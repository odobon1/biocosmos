"""
DDP equivalence tests for the tiled/chunked global-batch BCE loss (hardware.loss_chunk_size).

The single-GPU tests in test_chunked_loss.py pin down the tiling math across the full config space; the
2-rank test here pins down the DISTRIBUTED machinery that only exists across ranks: the cross-rank
embedding gather (_AllGather fwd/bwd), the SigLIP-style row-band sharding of the tile sweep (each rank
computes only its B/world_size band), the summation of disjoint-band partial grads (encoder params via
the _AllGather-routed representation backward, post-gather logit params via the manual per-parameter
all-reduce -- plain sum, no /world_size), no_sync, the banded precompute constants (band-partial sums
all-reduced to rank-identical values), and the post-backward leaf-grad fold that grad-norm logging
relies on.

It binds the REAL VLMWrapper methods to a lightweight harness `self`, wraps a tiny dual-encoder in DDP, and
asserts three paths agree to fp32 precision on every rank, per config case:

  (GT)    single-process full-batch loss.backward()      -- true full-batch-mean gradient
  (REF)   standard DDP: batch_step + loss.backward()      -- production path
  (CHUNK) batch_step_chunked                              -- no_sync + tiled backward + manual all-reduce

Cases: a plain BCE step, a BCE+BCE mix with norm.agg + mix_unit_scale (exercises the secondary logit
params and the embedding-dependent precompute sweeps under DDP), and a tax-target case (exercises the
banded soft-target dsmr-mass all-reduce and the tax target tiles under sharding). The 2-rank test
requires >= 2 CUDA devices; skipped otherwise. An assertion failure in any rank propagates out of
mp.spawn and fails the test.

test_chunked_ddp_single_rank_matches_full_batch runs the same harness at world_size=1 (>= 1 CUDA device):
torchrun with one GPU still wraps the model in DDP, and DDP's reducer arms on any forward taken outside
no_sync regardless of world size -- so batch_step_chunked must suppress it even on a single rank, or the
second tile backward trips "Expected to mark a variable ready only once".
"""
import copy
import math
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


def cfg_loss(targ="sw", norm_agg=False):
    return {
        "crit": "bce", "sim": "cos", "targ": targ,
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5},
                        "class_bal": {"beta": 0.9999}, "freq_type_2d": "naive",
                        "wt_mean_type": "per_class"},
            "focal": {"gamma": 2.0, "comp_type": 1},
            "bce": {"dsmr": True, "agg": "prod", "norm": {"cls_imb": False, "agg": norm_agg}},
        },
        "logits": {"scale": {"clamp": False}, "bias": {}},
    }


# (name, cfg1, cfg2, mix, mix_unit_scale)
CASES = [
    ("plain", cfg_loss("sw"), None, 0.0, False),
    ("mix_normagg_unitscale", cfg_loss("sw", norm_agg=True), cfg_loss("sw", norm_agg=True), 0.3, True),
    ("tax_dsmr", cfg_loss("tax"), None, 0.0, False),
]


class ToyDualEncoder(nn.Module):
    def __init__(self, d_in, d, with_secondary):
        super().__init__()
        self.img_enc = nn.Linear(d_in, d)
        self.txt_enc = nn.Linear(d_in, d)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))
        # secondary logit params exist only when the mix uses them: the logit params are applied AFTER
        # DDP.forward (in compute_logits, post-gather), so DDP's default reducer requires every param to
        # receive a gradient -- an unused param would desync grads. Register them only when mix != 0.
        if with_secondary:
            self.logit_scale2 = nn.Parameter(torch.tensor(1.7))
            self.logit_bias2 = nn.Parameter(torch.tensor(0.2))

    def forward(self, imgs, toks):
        return self.img_enc(imgs), self.txt_enc(toks)


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = None  # set below from VLMWrapper (deferred import)


def make_crit(BCECriterion, cfg, K, B, device):
    crit = BCECriterion.__new__(BCECriterion)
    crit.cfg = cfg
    crit.device = device
    crit.batch_size = B
    g = torch.Generator().manual_seed(12345)  # rank-independent -> identical counts on all ranks
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64).to(device)
    crit.wt_mean = 1.0
    return crit


def build_harness(model_ddp, crit1, crit2, mix, mix_unit_scale, world_size, device):
    h = Harness()
    h.model = model_ddp
    h.crit1 = crit1
    h.crit2 = crit2
    h.world_size = world_size
    h.device = device
    h.txt_pp = lambda x: x  # identity: toy "text" is already a feature tensor
    h.cfg = SimpleNamespace(
        loss=crit1.cfg,
        loss2={"mix": mix, "mix_unit_scale": mix_unit_scale},
        hw=SimpleNamespace(loss_chunk_size=None, mixed_prec=False),
        device=device,
    )
    return h


def full_batch_blended(toy, compute_sim, crit1, crit2, mix, mix_unit_scale, fi, ft, fc, ftd):
    """Single-process full-batch blended loss on `toy` -- the ground truth. Returns the normalized
    embeddings too (grads retained): their post-backward .grad is the full-batch dL/dembs that the
    chunked path's returned leaves must carry for grad-norm logging."""
    img = F.normalize(toy.img_enc(fi), dim=1)
    txt = F.normalize(toy.txt_enc(ft), dim=1)
    img.retain_grad()
    txt.retain_grad()

    def clogits(sim, clamp, secondary):
        s = toy.logit_scale2 if secondary else toy.logit_scale
        b = toy.logit_bias2 if secondary else toy.logit_bias
        if clamp:
            s = s.clamp(max=math.log(100))
        return sim * s.exp() + b

    def crit_loss(crit, secondary):
        sim = compute_sim(img, txt, crit.cfg["sim"])
        logits = clogits(sim, crit.cfg["logits"]["scale"]["clamp"], secondary)
        loss, loss_raw, _ = crit(logits, fc, ftd, train=True)
        return loss, loss_raw

    loss1, loss1_raw = crit_loss(crit1, False)
    if mix == 0.0:
        return loss1, loss1_raw, img, txt
    loss2, loss2_raw = crit_loss(crit2, True)
    if mix_unit_scale:
        loss1 = loss1 / loss1.detach().clamp_min(1e-12)
        loss2 = loss2 / loss2.detach().clamp_min(1e-12)
    return (1.0 - mix) * loss1 + mix * loss2, (1.0 - mix) * loss1_raw + mix * loss2_raw, img, txt


def grads(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def run(rank, world_size, port):
    from models import VLMWrapper
    from utils.head import compute_sim
    from utils.loss import BCECriterion

    Harness._unwrapped_model = VLMWrapper._unwrapped_model
    Harness.compute_logits = VLMWrapper.compute_logits
    Harness._gather_batch = VLMWrapper._gather_batch
    Harness._loss_for_crit_full_batch = VLMWrapper._loss_for_crit_full_batch
    Harness._global_batch_loss = VLMWrapper._global_batch_loss
    Harness.batch_step = VLMWrapper.batch_step
    Harness.batch_step_chunked = VLMWrapper.batch_step_chunked

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    d_in, D, K, SB = 32, 16, 10, 24
    B = world_size * SB

    # identical FULL global batch on all ranks (fixed seed); each rank takes its slice
    g = torch.Generator().manual_seed(7)
    full_imgs = torch.randn(B, d_in, generator=g)
    full_txts = torch.randn(B, d_in, generator=g)
    full_cls = torch.randint(0, K, (B,), generator=g)
    full_td = [{"rank_encs": torch.randint(0, 3, (4,), generator=g).tolist()} for _ in range(B)]  # for tax
    sl = slice(rank * SB, (rank + 1) * SB)
    imgs_sb, txts_sb, cls_sb = full_imgs[sl].to(device), full_txts[sl].to(device), full_cls[sl].to(device)
    targ_sb = full_td[sl]

    for name, cfg1, cfg2, mix, mix_unit_scale in CASES:
        for chunk_size in (SB // 3, SB):  # multi-tile + single-tile per band; both divide the per-rank band (B/world_size = SB)
            crit1 = make_crit(BCECriterion, cfg1, K, B, device)
            crit2 = make_crit(BCECriterion, cfg2, K, B, device) if mix != 0.0 else None

            torch.manual_seed(0)
            base = ToyDualEncoder(d_in, D, with_secondary=(mix != 0.0))
            toy_gt = copy.deepcopy(base).to(device).train()
            toy_ref = copy.deepcopy(base).to(device).train()
            toy_chunk = copy.deepcopy(base).to(device).train()
            ddp_ref = nn.parallel.DistributedDataParallel(toy_ref, device_ids=[rank])
            ddp_chunk = nn.parallel.DistributedDataParallel(toy_chunk, device_ids=[rank])

            # (GT) single-process full-batch ground truth
            fi, ft, fc = full_imgs.to(device), full_txts.to(device), full_cls.to(device)
            loss_gt, loss_raw_gt, embs_img_gt, embs_txt_gt = full_batch_blended(
                toy_gt, compute_sim, crit1, crit2, mix, mix_unit_scale, fi, ft, fc, full_td)
            toy_gt.zero_grad(set_to_none=True)
            loss_gt.backward()
            g_gt = grads(toy_gt)

            # (REF) standard DDP path (chunking off)
            h_ref = build_harness(ddp_ref, crit1, crit2, mix, mix_unit_scale, world_size, device)
            ddp_ref.zero_grad(set_to_none=True)
            loss_ref, _, *_ = Harness.batch_step(h_ref, imgs_sb, txts_sb, cls_sb, targ_sb)
            loss_ref.backward()
            g_ref = grads(toy_ref)

            # (CHUNK) tiled path (its own backward + manual all-reduce internally)
            h_chunk = build_harness(ddp_chunk, crit1, crit2, mix, mix_unit_scale, world_size, device)
            h_chunk.cfg.hw.loss_chunk_size = chunk_size
            ddp_chunk.zero_grad(set_to_none=True)
            loss_chunk, _, img_leaf, txt_leaf, *_ = Harness.batch_step_chunked(h_chunk, imgs_sb, txts_sb, cls_sb, targ_sb)
            g_chunk = grads(toy_chunk)

            def rel(a, b):
                return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)

            tag = f"[rank {rank} case={name} chunk={chunk_size} B={B}]"
            assert abs(loss_chunk.item() - loss_gt.item()) < 1e-4 * (abs(loss_gt.item()) + 1e-6), \
                f"{tag} CHUNK loss {loss_chunk.item()} != GT {loss_gt.item()}"
            # the returned leaves must carry FULL-BATCH dL/dembs on every rank (grad-norm logging contract)
            r_il = rel(img_leaf.grad, embs_img_gt.grad)
            r_tl = rel(txt_leaf.grad, embs_txt_gt.grad)
            assert r_il < 3e-4, f"{tag} leaf img-grad mismatch: rel={r_il:.2e}"
            assert r_tl < 3e-4, f"{tag} leaf txt-grad mismatch: rel={r_tl:.2e}"
            for n in g_gt:
                if g_gt[n] is None:  # param unused this case (e.g. secondary logits at mix==0)
                    assert g_chunk[n] is None, f"{tag} {n}: CHUNK grad set but GT is None"
                    continue
                assert g_chunk[n] is not None and g_ref[n] is not None, f"{tag} None grad for used param {n}"
                r_cg = rel(g_chunk[n], g_gt[n])   # chunked vs full-batch ground truth
                r_rg = rel(g_ref[n], g_gt[n])     # production DDP path vs ground truth
                assert r_cg < 3e-4, f"{tag} CHUNK grad mismatch on {n}: rel={r_cg:.2e}"
                assert r_rg < 3e-4, f"{tag} REF grad mismatch on {n}: rel={r_rg:.2e}"

    dist.destroy_process_group()


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 CUDA devices")
def test_chunked_ddp_matches_full_batch():
    world_size = 2
    mp.spawn(run, args=(world_size, _free_port()), nprocs=world_size, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs >= 1 CUDA device")
def test_chunked_ddp_single_rank_matches_full_batch():
    # regression: 1-GPU torchrun still DDP-wraps; the reducer arms regardless of world size, so the
    # chunked path's multiple backwards must run under no_sync even at world_size 1
    mp.spawn(run, args=(1, _free_port()), nprocs=1, join=True)

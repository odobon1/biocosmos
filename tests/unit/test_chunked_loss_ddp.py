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

Cases: a plain BCE step, a BCE step with cls_imb.norm (the embedding-free banded weight-mean sweep), an
mp / phylo target blend (the blended target tiles and the banded soft-target dsmr-mass all-reduce), a
tax-target case (the tax target tiles under sharding), two centering cases, two bif_bce cases (the
banded two-branch tiles with row-wise dsmr + targ_mass_neut, and a blend with per-branch grad-proj
constants -- the branch grad-mean all-reduce under sharding), two unitless loss blends (loss.blend.type
loss + loss.unitless: the per-term dsmr masses and the banded term-magnitude pre-sweep's all-reduce, bce and
bif_bce + grad-proj), and two loss blends on separate logit scalars (loss.logits.shared false: the second
scalar pair's band-partial grads under the manual all-reduce, per-pair grad-proj constants, bce and bif_bce).
The 2-rank test requires >= 2 CUDA devices;
skipped otherwise. An assertion failure in any rank propagates out of mp.spawn and fails the test.

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


def cfg_loss(lambda_=0.0, cls_imb_norm=False, center=None, crit="bce", neut=False, blend_type="targ", unitless=False, shared=True):
    """The loss-level config (train.yaml's `loss` block, as the criterion reads it)."""
    return {
        "crit": crit, "sim": "cos", "blend": {"lambda": lambda_, "type": blend_type}, "unitless": unitless,
        "bce": {"targ_mass_neut": neut},  # read by bif_bce only
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5},
                        "class_bal": {"beta": 0.9999}, "norm": cls_imb_norm},
            "focal": {"gamma": 2.0},
            "bce": {"dsmr": True},
        },
        "logits": {"shared": shared, "scale": {"clamp": False}, "bce": {"center": center, "bias": {}}},
    }


def targ_spec(targ):
    """A target spec (train.yaml's loss1 / loss2 block); the tsm is read by InfoNCE only."""
    return {"targ": targ, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}


class _ZSG(torch.autograd.Function):
    """Mirror of models._ZeroSumGrad for the single-process ground truth (full-batch mean)."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return g - g.mean()


# (name, cfg, targ1, targ2)
CASES = [
    ("plain", cfg_loss(), "mp", "sp"),
    ("normci", cfg_loss(cls_imb_norm=True), "mp", "sp"),
    ("blend_phylo", cfg_loss(lambda_=0.3), "mp", "phylo"),
    ("tax_dsmr", cfg_loss(), "tax", "sp"),
    ("center_sim", cfg_loss(center="sim"), "mp", "sp"),
    ("center_gp2_blend", cfg_loss(lambda_=0.3, center="grad_proj2"), "mp", "sp"),
    # bif_bce: banded two-branch tiles (row-wise dsmr + neut), and a blend with per-branch grad-proj
    # constants (exercises the branch grad-mean all-reduce under sharding)
    ("bif_dsmr_neut", cfg_loss(crit="bif_bce", neut=True), "mp", "sp"),
    ("bif_gp_blend", cfg_loss(lambda_=0.3, crit="bif_bce", center="grad_proj"), "mp", "phylo"),
    # loss blend: per-term DSMR masses + the unitless term magnitudes (pre-sweep all-reduce) under sharding
    ("loss_blend_unitless", cfg_loss(lambda_=0.3, blend_type="loss", unitless=True), "mp", "phylo"),
    ("bif_loss_blend_unitless_gp", cfg_loss(lambda_=0.3, crit="bif_bce", center="grad_proj", neut=True, blend_type="loss", unitless=True), "mp", "phylo"),
    # separate logit scalars: the second pair's band-partial grads (manual all-reduce) and per-pair grad-proj constants
    ("sep_scalars_gp2", cfg_loss(lambda_=0.3, center="grad_proj2", blend_type="loss", shared=False), "mp", "phylo"),
    ("bif_sep_scalars_unitless", cfg_loss(lambda_=0.3, crit="bif_bce", neut=True, blend_type="loss", unitless=True, shared=False), "mp", "phylo"),
]


class ToyDualEncoder(nn.Module):
    def __init__(self, d_in, d, sep_scalars=False):
        super().__init__()
        self.img_enc = nn.Linear(d_in, d)
        self.txt_enc = nn.Linear(d_in, d)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))
        if sep_scalars:  # loss2's term's own pair (registered only when used: DDP flags unused params)
            self.logit_scale2 = nn.Parameter(torch.tensor(1.7))
            self.logit_bias2 = nn.Parameter(torch.tensor(-0.9))

    def forward(self, imgs, toks):
        return self.img_enc(imgs), self.txt_enc(toks)


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = None  # set below from VLMWrapper (deferred import)


class DummyPhyloVCV:
    """Constant soft target (0.25) in place of the tree-derived matrix; block builder agrees with the
    full matrix by construction."""
    def get_targs_batch(self, targ_data_b):
        n = len(targ_data_b)
        return torch.full((n, n), 0.25)

    def make_targ_block_fn(self, targ_data_b, device):
        B = len(targ_data_b)
        return lambda rs, re: torch.full((re - rs, B), 0.25, device=device)


def make_crit(crit_cls, cfg, targ1, targ2, K, B, device, targ_specs):
    crit = crit_cls.__new__(crit_cls)
    crit.cfg = cfg
    crit.targ_specs = targ_specs(cfg["blend"]["lambda"], targ_spec(targ1), targ_spec(targ2))
    crit.device = device
    crit.batch_size = B
    g = torch.Generator().manual_seed(12345)  # rank-independent -> identical counts on all ranks
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64).to(device)
    crit.wt_mean = 1.0
    return crit


def build_harness(model_ddp, crit, world_size, device):
    h = Harness()
    h.model = model_ddp
    h.crit = crit
    h.world_size = world_size
    h.device = device
    h.txt_pp = lambda x: x  # identity: toy "text" is already a feature tensor
    h.cfg = SimpleNamespace(
        loss=crit.cfg,
        hw=SimpleNamespace(loss_chunk_size=None, mixed_prec=False),
        reporting={"batch_diagnostics": {"emb_logit_grads": True, "sim_grad_sums": True, "sim_targ_stats": True},
                 "learning_curves": {"hpsm": {"kappas": [0.0, 3.0]}}},
        device=device,
    )
    return h


def full_batch_reference(toy, compute_sim, crit, fi, ft, fc, ftd):
    """Single-process full-batch loss on `toy` -- the ground truth. Returns the normalized embeddings
    (grads retained: their post-backward .grad is the full-batch dL/dembs that the chunked path's
    returned leaves must carry for grad-norm logging) and the sim branch tuple ((sim,) non-bifurcated,
    (i2t, t2i) bifurcated, mirroring _loss_full_batch; grads retained: their post-backward
    branch-summed .grad.sum() is the ground truth for the chunked path's tile-accumulated grad_sum_sim)."""
    img = F.normalize(toy.img_enc(fi), dim=1)
    txt = F.normalize(toy.txt_enc(ft), dim=1)
    img.retain_grad()
    txt.retain_grad()

    def clogits(sim, clamp, center, half_live=False, secondary=False):
        sim = sim.float()  # the head runs in float32 whatever the sims came in as
        s, b = (toy.logit_scale2, toy.logit_bias2) if secondary else (toy.logit_scale, toy.logit_bias)
        if half_live:
            s = 0.5 * s + 0.5 * s.detach()
            b = 0.5 * b + 0.5 * b.detach()
        if clamp:
            s = s.clamp(max=math.log(100))
        if center == "grad_proj":
            sim = _ZSG.apply(sim)
        sim_scaled = sim * s.exp()
        if center == "grad_proj2":
            sim_scaled = _ZSG.apply(sim_scaled)
        if center == "sim":
            sim_scaled = sim_scaled - sim_scaled.mean()
        return sim_scaled + b

    clamp = crit.cfg["logits"]["scale"]["clamp"]
    center = crit.cfg["logits"]["bce"]["center"]
    secondaries = (False, True) if crit.sep_scalars else (False,)  # per-term logits, as _loss_full_batch builds them
    if crit.bifurcated:
        sims = (
            compute_sim(img, txt.detach(), crit.cfg["sim"]),
            compute_sim(img.detach(), txt, crit.cfg["sim"]),
        )
        crit_logits = [tuple(clogits(s, clamp, center, half_live=True, secondary=sec) for s in sims) for sec in secondaries]
    else:
        sims = (compute_sim(img, txt, crit.cfg["sim"]),)
        crit_logits = [clogits(sims[0], clamp, center, secondary=sec) for sec in secondaries]
    for s in sims:
        s.retain_grad()
    if crit.sep_scalars:
        logit_scale = (toy.logit_scale, toy.logit_scale2)
    else:
        crit_logits, logit_scale = crit_logits[0], toy.logit_scale
    loss, loss_raw, _, _ = crit(crit_logits, fc, ftd, train=True, logit_scale=logit_scale, sim=sims[0])
    return loss, loss_raw, img, txt, sims


def grads(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def run(rank, world_size, port):
    import utils.loss as L
    from models import VLMWrapper
    from utils.head import compute_sim
    crit_cls = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion}
    L.get_phylo_vcv = lambda dataset: DummyPhyloVCV()  # phylo targets without a tree

    Harness._unwrapped_model = VLMWrapper._unwrapped_model
    Harness.compute_logits = VLMWrapper.compute_logits
    Harness._gather_batch = VLMWrapper._gather_batch
    Harness._loss_full_batch = VLMWrapper._loss_full_batch
    Harness._batch_stats = VLMWrapper._batch_stats
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
    full_td = [{"rank_encs": torch.randint(0, 3, (4,), generator=g).tolist(), "cid": f"c{int(full_cls[i])}", "dataset": "cub"}
               for i in range(B)]  # rank_encs for tax, cid/dataset for phylo
    sl = slice(rank * SB, (rank + 1) * SB)
    imgs_sb, txts_sb, cls_sb = full_imgs[sl].to(device), full_txts[sl].to(device), full_cls[sl].to(device)
    targ_sb = full_td[sl]

    for name, cfg, targ1, targ2 in CASES:
        for chunk_size in (SB // 3, SB):  # multi-tile + single-tile per band; both divide the per-rank band (B/world_size = SB)
            crit = make_crit(crit_cls[cfg["crit"]], cfg, targ1, targ2, K, B, device, L.targ_specs)

            torch.manual_seed(0)
            base = ToyDualEncoder(d_in, D, sep_scalars=crit.sep_scalars)
            toy_gt = copy.deepcopy(base).to(device).train()
            toy_ref = copy.deepcopy(base).to(device).train()
            toy_chunk = copy.deepcopy(base).to(device).train()
            ddp_ref = nn.parallel.DistributedDataParallel(toy_ref, device_ids=[rank])
            ddp_chunk = nn.parallel.DistributedDataParallel(toy_chunk, device_ids=[rank])

            # (GT) single-process full-batch ground truth
            fi, ft, fc = full_imgs.to(device), full_txts.to(device), full_cls.to(device)
            loss_gt, loss_raw_gt, embs_img_gt, embs_txt_gt, sims_gt = full_batch_reference(
                toy_gt, compute_sim, crit, fi, ft, fc, full_td)
            toy_gt.zero_grad(set_to_none=True)
            loss_gt.backward()
            g_gt = grads(toy_gt)
            gsum_gt = sum(s.grad.double().sum().item() for s in sims_gt)

            # (REF) standard DDP path (chunking off)
            h_ref = build_harness(ddp_ref, crit, world_size, device)
            ddp_ref.zero_grad(set_to_none=True)
            loss_ref, _, *_ = Harness.batch_step(h_ref, imgs_sb, txts_sb, cls_sb, targ_sb)
            loss_ref.backward()
            g_ref = grads(toy_ref)

            # (CHUNK) tiled path (its own backward + manual all-reduce internally)
            h_chunk = build_harness(ddp_chunk, crit, world_size, device)
            h_chunk.cfg.hw.loss_chunk_size = chunk_size
            ddp_chunk.zero_grad(set_to_none=True)
            loss_chunk, _, img_leaf, txt_leaf, _, _, _, gsum_chunk = Harness.batch_step_chunked(h_chunk, imgs_sb, txts_sb, cls_sb, targ_sb)
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
            # the tile-accumulated (all-reduced) sim-grad sum must match the full-batch retained-grad sum
            assert abs(gsum_chunk - gsum_gt) < 1e-4 * (abs(gsum_gt) + 1.0), \
                f"{tag} grad_sum_sim: CHUNK {gsum_chunk} != GT {gsum_gt}"
            for n in g_gt:
                assert g_gt[n] is not None, f"{tag} GT grad missing for {n}"
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

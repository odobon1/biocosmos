"""
DDP equivalence tests for the InfoNCE global-batch path (the non-chunked one -- chunking_supported
excludes InfoNCE), in particular loss.infonce.block_residuals and the actual sim-grad diagnostics.

The single-process tests in test_infonce_scale_grad.py pin down the residual/blocking math; the 2-rank
test here pins down what only exists across ranks: the _AllGather embedding gather feeding one identical
full-batch loss per rank (DDP's grad averaging is a no-op for the replicated post-gather logit-scale
grads and exactly cancels _AllGather.backward's sum for the tower grads), the block terms riding those
same paths (the scale part post-gather, the `full` part reaching the towers through _AllGather), and
the per-rank block_stats record being what every rank saw -- TrainPipeline._record_train_batch reads it
on rank 0 alone.

It binds the REAL VLMWrapper methods to a lightweight harness `self` (as test_chunked_loss_ddp.py does),
wraps a toy dual encoder in DDP, and per case compares three paths on every rank:

  (GT)    single-process full-batch batch_step + backward on the full global batch  -- ground truth
  (REF)   production DDP path: per-rank slice -> batch_step (gather) + loss.backward()
  (CTRL)  REF's weights and data with block_residuals: None  -- for the correction-semantics check

asserting: the full-batch loss reading and every DDP-synced param grad match GT; block_stats identical
across ranks and its coverage/bounds exactly GT's (they are embedding-independent: targets come off the
gathered class encodings, alpha off the replicated parameter); the recorded dlogscale_correction(s)
equal the POST-SYNC gradient delta grad(REF) - grad(CTRL); the zero-valued term leaves the loss reading
bit-untouched; `full` moves the tower grads where rows were applied while `alpha` leaves them alone; and
the actual sim-grad diagnostics (grad_sum_sim + sim_grad_entropies_actual off the retained sims' .grad,
TrainPipeline._step_train's read) match GT.

Cases: a no-block target-blend baseline (plain DDP equivalence of the InfoNCE path), alpha/full on hard
binary targets (every row the hard closed form), a graded tax target through the active sets and a
target blend's spec-less rows (the continuous-target paths), a softmax-pinned control (inside the band:
all feasible, correction exactly zero), binary memberships under a fixed sm_scale c past the band (the
soft two-level closed form), a unitless loss blend (per-term tensor coefficients inside the term), and
separate logit scalars (per-term corrections on logit_scale / logit_scale2). Requires >= 2 CUDA devices;
skipped otherwise. An assertion failure in any rank propagates out of mp.spawn and fails the test.
"""
import copy
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.unit.test_chunked_loss_ddp import _free_port


def cfg_infonce(lambda_=0.0, blend_type="targ", block=None, unitless=False, shared=True, clamp=False):
    """The loss-level config (train.yaml's `loss` block, as InfoNCECriterion reads it); block_residuals
    forbids class-imbalance and focal weighting (utils.config), so those sit at their off values."""
    return {
        "crit": "infonce", "sim": "cos",
        "blend": {"lambda": lambda_, "type": blend_type},
        "unitless": unitless,
        "infonce": {"block_residuals": block},
        "wting": {
            "cls_imb": {"type": None, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999}, "norm": False},
            "focal": {"gamma": 0.0},
        },
        "logits": {"shared": shared, "scale": {"clamp": clamp}, "bce": {"center": None, "bias": {}}},
    }


def spec(targ, tsm_type="linear", sm_scale="pinned"):
    return {"targ": targ, "infonce": {"tsm": {"type": tsm_type, "sm_scale": sm_scale}}}


# (name, cfg, spec1, spec2, log_scale_init, expect) -- expect: term-0 coverage expectation
CASES = [
    # no blocking: plain DDP equivalence of the InfoNCE path incl. the batch diagnostics
    ("baseline_tax_blend", cfg_infonce(lambda_=0.3), spec("mp"), spec("tax"), 2.3, None),
    # hard binary targets, the pre-continuous regime: every row on the hard closed form
    ("alpha_hard_mp", cfg_infonce(block="alpha"), spec("mp"), spec("sp"), 0.5, "all_applied"),
    ("full_hard_mp", cfg_infonce(block="full"), spec("mp"), spec("sp"), 0.5, "all_applied"),
    # graded target (tax) under the linear tsm: rows past the band go through the active sets
    ("full_tax_active_sets", cfg_infonce(block="full"), spec("tax"), spec("sp"), 0.5, "some_applied"),
    # a target blend's rows carry no spec (infonce_block_resid's spec=None path)
    ("full_targ_blend", cfg_infonce(lambda_=0.3, block="full"), spec("mp"), spec("tax"), 0.5, "some_applied"),
    # softmax tsm pinned: inside the band by construction -> all feasible, correction exactly zero
    ("alpha_pinned_control", cfg_infonce(block="alpha"), spec("mp", "softmax", "pinned"), spec("sp"), 0.5, "all_feasible"),
    # binary memberships under a fixed sm_scale c past the band -> the soft two-level closed form
    ("full_softmax_c", cfg_infonce(block="full"), spec("mp", "softmax", 3.0), spec("sp"), 0.5, "all_applied"),
    # loss blend + unitless: per-term tensor coefficients applied inside the term, in float64
    ("full_loss_blend_unitless", cfg_infonce(lambda_=0.3, blend_type="loss", block="full", unitless=True),
     spec("mp"), spec("tax"), 0.5, "some_applied"),
    # separate logit scalars: per-term corrections on logit_scale / logit_scale2
    ("full_sep_scalars", cfg_infonce(lambda_=0.3, blend_type="loss", block="full", shared=False),
     spec("mp"), spec("tax"), 0.5, "some_applied"),
]


class ToyInfoNCE(nn.Module):
    def __init__(self, d_in, d, log_scale, sep_scalars=False):
        super().__init__()
        self.img_enc = nn.Linear(d_in, d)
        self.txt_enc = nn.Linear(d_in, d)
        self.logit_scale = nn.Parameter(torch.tensor(log_scale))
        self.logit_bias = None  # InfoNCE: no bias (models.VLMWrapper.__init__)
        if sep_scalars:
            self.logit_scale2 = nn.Parameter(torch.tensor(log_scale + 0.4))
            self.logit_bias2 = None

    def forward(self, imgs, toks):
        return self.img_enc(imgs), self.txt_enc(toks)


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = None  # set below from VLMWrapper (deferred import)


def make_crit(L, cfg, spec1, spec2, K, B, device):
    crit = L.InfoNCECriterion.__new__(L.InfoNCECriterion)
    crit.cfg = cfg
    crit.targ_specs = L.targ_specs(cfg["blend"]["lambda"], spec1, spec2)
    crit.device = device
    crit.batch_size = B
    crit.counts = torch.ones(K, dtype=torch.float64, device=device)
    crit.wt_mean = 1.0
    return crit


def build_harness(model, crit, world_size, device):
    h = Harness()
    h.model = model
    h.crit = crit
    h.world_size = world_size
    h.device = device
    h.txt_pp = lambda x: x  # identity: toy "text" is already a feature tensor
    h.cfg = SimpleNamespace(
        loss=crit.cfg,
        hw=SimpleNamespace(loss_chunk_size=None, mixed_prec=False),
        reporting={"batch_diagnostics": {"emb_logit_grads": True, "sim_grad_sums": True, "sim_targ_stats": True},
                   "learning_curves": {"hpsm": {"kappas": [0.0]}, "hist_bins": 20}},
        device=device,
    )
    return h


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-12)


def run(rank, world_size, port):
    import utils.loss as L
    from models import VLMWrapper

    Harness._unwrapped_model = VLMWrapper._unwrapped_model
    Harness.compute_logits = VLMWrapper.compute_logits
    Harness._gather_batch = VLMWrapper._gather_batch
    Harness._loss_full_batch = VLMWrapper._loss_full_batch
    Harness._batch_stats = VLMWrapper._batch_stats
    Harness._global_batch_loss = VLMWrapper._global_batch_loss
    Harness.batch_step = VLMWrapper.batch_step

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    d_in, D, K, SB = 32, 16, 6, 24
    B = world_size * SB

    # identical FULL global batch on all ranks (fixed seed); each rank takes its slice
    g = torch.Generator().manual_seed(7)
    full_imgs = torch.randn(B, d_in, generator=g)
    full_txts = torch.randn(B, d_in, generator=g)
    full_cls = torch.randint(0, K, (B,), generator=g)
    full_td = [{"rank_encs": torch.randint(0, 3, (4,), generator=g).tolist()} for _ in range(B)]
    sl = slice(rank * SB, (rank + 1) * SB)
    imgs_sb, txts_sb, cls_sb = full_imgs[sl].to(device), full_txts[sl].to(device), full_cls[sl].to(device)
    targ_sb = full_td[sl]
    fi, ft, fc = full_imgs.to(device), full_txts.to(device), full_cls.to(device)

    for name, cfg, spec1, spec2, log_scale, expect in CASES:
        block = cfg["infonce"]["block_residuals"]
        sep = not cfg["logits"]["shared"]
        torch.manual_seed(0)
        base = ToyInfoNCE(d_in, D, log_scale, sep_scalars=sep)
        toy_gt = copy.deepcopy(base).to(device).train()
        toy_ref = copy.deepcopy(base).to(device).train()
        ddp_ref = nn.parallel.DistributedDataParallel(toy_ref, device_ids=[rank])
        tag = f"[rank {rank} case={name}]"

        # (GT) single-process full batch: the same real batch_step at world_size 1 (gather no-op)
        crit_gt = make_crit(L, cfg, spec1, spec2, K, B, device)
        h_gt = build_harness(toy_gt, crit_gt, 1, device)
        toy_gt.zero_grad(set_to_none=True)
        loss_gt, _, _, _, _, _, _, sims_gt = Harness.batch_step(h_gt, fi, ft, fc, full_td)
        loss_gt.backward()
        g_gt = grads(toy_gt)
        block_gt = crit_gt.block_stats
        gsum_gt = sum(s.grad.float().sum().item() for s in sims_gt if s.grad is not None)
        ent_gt = L.sim_grad_entropies_actual(sims_gt[0].grad)

        # (REF) production DDP path
        crit_ref = make_crit(L, cfg, spec1, spec2, K, B, device)
        h_ref = build_harness(ddp_ref, crit_ref, world_size, device)
        ddp_ref.zero_grad(set_to_none=True)
        loss_ref, _, _, _, _, _, _, sims_ref = Harness.batch_step(h_ref, imgs_sb, txts_sb, cls_sb, targ_sb)
        loss_ref.backward()
        g_ref = grads(toy_ref)
        block_ref = crit_ref.block_stats
        gsum_ref = sum(s.grad.float().sum().item() for s in sims_ref if s.grad is not None)
        ent_ref = L.sim_grad_entropies_actual(sims_ref[0].grad)

        # the full-batch loss reading is identical on every rank and equals GT
        assert abs(loss_ref.item() - loss_gt.item()) < 1e-5 * (abs(loss_gt.item()) + 1e-8), \
            f"{tag} loss REF {loss_ref.item()} != GT {loss_gt.item()}"

        # every DDP-synced grad matches the single-process full-batch grad
        for n in g_gt:
            r = rel(g_ref[n], g_gt[n])
            assert r < 3e-4, f"{tag} grad mismatch on {n}: rel={r:.2e}"

        # the actual sim-grad diagnostics (TrainPipeline._step_train's read) under DDP match GT
        assert abs(gsum_ref - gsum_gt) < 1e-4 * (abs(gsum_gt) + 1.0), f"{tag} grad_sum_sim {gsum_ref} != {gsum_gt}"
        for k in ent_gt:
            a, b = ent_ref[k], ent_gt[k]
            same = (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-6 * (abs(b) + 1.0)
            assert same, f"{tag} {k}: REF {a} != GT {b}"

        if block is None:
            assert block_ref is None and block_gt is None, f"{tag} block_stats set without blocking"
            continue

        # block_stats identical across ranks: what rank 0 records is what every rank saw
        gathered = [None] * world_size
        dist.all_gather_object(gathered, block_ref)
        assert all(other == gathered[0] for other in gathered), \
            f"{tag} block_stats differ across ranks:\n{gathered}"

        # coverage/bounds are embedding-independent -> exactly GT's; the correction contracts against
        # the sims, whose GT/REF matmuls run at different shapes -> approx
        assert block_ref["block_coverage"] == block_gt["block_coverage"], \
            f"{tag} coverage REF {block_ref['block_coverage']} != GT {block_gt['block_coverage']}"
        for (u_r, b_r), (u_g, b_g) in zip(block_ref["block_bound"], block_gt["block_bound"]):
            assert abs(u_r - u_g) <= 1e-12 * abs(u_g), f"{tag} U bound REF {u_r} != GT {u_g}"
            assert abs(b_r - b_g) <= 1e-6 * (abs(b_g) + 1e-15), f"{tag} correction bound REF {b_r} != GT {b_g}"
        for key in ("dlogscale_correction", "dlogscale_correction2") if sep else ("dlogscale_correction",):
            assert abs(block_ref[key] - block_gt[key]) < 1e-5 * (abs(block_gt[key]) + 1e-9), \
                f"{tag} {key} REF {block_ref[key]} != GT {block_gt[key]}"

        cov = block_ref["block_coverage"]
        for c3 in cov:
            assert abs(sum(c3) - 1.0) < 1e-12, f"{tag} coverage does not sum to 1: {c3}"
        if expect == "all_applied":
            assert cov[0][0] == 1.0, f"{tag} expected all rows applied, coverage {cov}"
        elif expect == "all_feasible":
            assert cov[0] == [0.0, 1.0, 0.0], f"{tag} expected all rows feasible, coverage {cov}"
            assert block_ref["dlogscale_correction"] == 0.0, f"{tag} pinned control has a nonzero correction"
        elif expect == "some_applied":
            assert cov[0][0] > 0.0, f"{tag} expected some rows applied, coverage {cov}"

        # (CTRL) same weights and data, blocking off
        cfg_off = copy.deepcopy(cfg)
        cfg_off["infonce"]["block_residuals"] = None
        toy_ctrl = copy.deepcopy(base).to(device).train()
        ddp_ctrl = nn.parallel.DistributedDataParallel(toy_ctrl, device_ids=[rank])
        crit_ctrl = make_crit(L, cfg_off, spec1, spec2, K, B, device)
        h_ctrl = build_harness(ddp_ctrl, crit_ctrl, world_size, device)
        ddp_ctrl.zero_grad(set_to_none=True)
        loss_ctrl, *_ = Harness.batch_step(h_ctrl, imgs_sb, txts_sb, cls_sb, targ_sb)
        loss_ctrl.backward()
        g_ctrl = grads(toy_ctrl)

        # the zero-valued term must not move the loss reading
        assert loss_ref.item() == loss_ctrl.item(), \
            f"{tag} blocking moved the loss: {loss_ref.item()} vs {loss_ctrl.item()}"

        # the recorded correction is the POST-SYNC gradient delta: grad_after = grad_before + correction
        pairs = [("logit_scale", "dlogscale_correction")] + ([("logit_scale2", "dlogscale_correction2")] if sep else [])
        for pname, ckey in pairs:
            delta = (g_ref[pname] - g_ctrl[pname]).item()
            corr = block_ref[ckey]
            assert abs(delta - corr) < 1e-5 * (abs(corr) + 1e-4), \
                f"{tag} {pname} post-sync grad delta {delta} != recorded correction {corr}"

        # `full` reaches the towers where rows were applied; `alpha` must not touch them
        tower_delta = max(rel(g_ref[n], g_ctrl[n]) for n in g_ref if n.startswith(("img_enc", "txt_enc")))
        if block == "full" and cov[0][0] > 0.0:
            assert tower_delta > 1e-6, f"{tag} full blocking left the tower grads untouched"
        if block == "alpha":
            assert tower_delta < 1e-6, f"{tag} alpha blocking moved the tower grads: rel={tower_delta:.2e}"

    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 CUDA devices")
def test_infonce_ddp_matches_full_batch():
    world_size = 2
    mp.spawn(run, args=(world_size, _free_port()), nprocs=world_size, join=True)

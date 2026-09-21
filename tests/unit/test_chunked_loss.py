"""
Equivalence tests for the tiled/chunked global-batch BCE-family loss (hardware.loss_chunk_size).

chunked_bce_loss_backward must reproduce the loss and gradients (wrt image/text embeddings and the
logit scale/bias) of the full-batch path (BCECriterion.__call__ / BifurcatedBCECriterion.__call__ via
_global_batch_loss), up to floating-point summation order -- across the full BCE-family config space:
mp/sp/tax/phylo targets and their blends (loss.blend.lambda) under either loss.blend.type, loss.unitless,
separate logit scalars (loss.logits.shared false: one logits tile per loss term), cls_imb.norm, and
bif_bce's two-branch tiles
(half-live logit scalars, row-wise DSMR, targ_mass_neut, per-branch centering).
"""
import importlib
import math
import sys
import types

import numpy as np
import pytest
import torch

from utils.head import compute_sim
from utils.phylo import PhyloVCV as RealPhyloVCV  # captured real class before the fake below


def import_loss_module():
    fake_phylo = types.ModuleType("utils.phylo")

    class DummyPhyloVCV:
        """Constant soft target (0.25); block builder agrees with the full matrix by construction."""
        def __init__(self, dataset: str, split: str, train_pt: str, batch_size: int, kernel: str, beta: float,
                     shuffle: bool = False, seed: int | None = None) -> None:
            self.dataset = dataset

        def get_targs_batch(self, targ_data_b):
            n = len(targ_data_b)
            return torch.full((n, n), 0.25)

        def make_targ_block_fn(self, targ_data_b, device):
            B = len(targ_data_b)
            return lambda rs, re: torch.full((re - rs, B), 0.25, device=device)

    fake_phylo.PhyloVCV = DummyPhyloVCV
    sys.modules["utils.phylo"] = fake_phylo
    sys.modules.pop("utils.loss", None)
    return importlib.import_module("utils.loss")


L = import_loss_module()
# phylo-target params for get_phylo_vcv's constructor call (DummyPhyloVCV ignores them)
L.configure_phylo_targs(split="D10", train_pt="train", batch_size=4, kernel="laplace", beta=1.0, shuffle=False, seed=None)


def _cfg(crit="bce", lambda_=0.0, dsmr=True, focal_gamma=2.0, sim="cos", cls_imb_norm=False, center=None, neut=False,
         blend_type="targ", unitless=False, shared=True):
    """The loss-level config (train.yaml's `loss` block, as the criterion reads it)."""
    return {
        "crit": crit, "sim": sim, "blend": {"lambda": lambda_, "type": blend_type}, "unitless": unitless,
        "bce": {"targ_mass_neut": neut},  # read by bif_bce only
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999},
                        "norm": cls_imb_norm},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),  # config load prunes the block when gamma = 0.0
            "bce": {"dsmr": dsmr},
        },
        "logits": {"shared": shared, "scale": {"clamp": False}, "bce": {"center": center, "bias": {}}},
    }


def _targ(targ):
    """A target spec (train.yaml's loss1 / loss2 block); the tsm is read by InfoNCE only."""
    return {"targ": targ, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}


CRIT_CLS = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion}
HIST_BINS = 20  # the chunked calls' reporting.learning_curves.hist_bins


def _make_crit(cfg, K, B, targ1="mp", targ2="sp"):
    cls = CRIT_CLS[cfg["crit"]]
    crit = cls.__new__(cls)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.targ_specs = L.targ_specs(cfg["blend"]["lambda"], _targ(targ1), _targ(targ2))
    crit.device = torch.device("cpu")
    crit.batch_size = B
    g = torch.Generator().manual_seed(K)
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


def _params(seed):
    return {
        "scale": torch.tensor(2.3, requires_grad=True),
        "bias": torch.tensor(-0.5, requires_grad=True),
        # loss2's term's own pair (separate logit scalars); unused -- no grad -- otherwise
        "scale2": torch.tensor(1.7, requires_grad=True),
        "bias2": torch.tensor(-0.9, requires_grad=True),
    }


class _ZSG(torch.autograd.Function):
    """Mirror of models._ZeroSumGrad (identity forward, g - g.mean() backward)."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return g - g.mean()


class _ZSGC(torch.autograd.Function):
    """Mirror of models._ZeroSumGradConst (identity forward, g - c backward)."""
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x

    @staticmethod
    def backward(ctx, g):
        return g - ctx.c, None


def _compute_logits_fn(p):
    """Stub mirroring VLMWrapper.compute_logits's center/half_live semantics (incl. the chunked
    path's center_global hook) over the toy scale/bias params."""
    def compute_logits(sim, clamp, center=None, center_global=None, half_live=False, secondary=False):
        sim = sim.float()  # the head runs in float32 whatever the sims came in as
        s, b = (p["scale2"], p["bias2"]) if secondary else (p["scale"], p["bias"])
        if half_live:
            s = 0.5 * s + 0.5 * s.detach()
            b = 0.5 * b + 0.5 * b.detach()
        if clamp:
            s = s.clamp(max=math.log(100))
        if center == "grad_proj":
            sim = _ZSG.apply(sim) if center_global is None else _ZSGC.apply(sim, center_global)
        sim_scaled = sim * s.exp()
        if center == "grad_proj2":
            sim_scaled = _ZSG.apply(sim_scaled) if center_global is None else _ZSGC.apply(sim_scaled, center_global)
        if center == "sim":
            sim_scaled = sim_scaled - (sim_scaled.mean() if center_global is None else center_global * s.exp())
        return sim_scaled + b
    return compute_logits


def _make_targ_data(B, K, R, class_encs_b):
    """targ_data carrying every field any targ_type needs (rank_encs for tax, cid/dataset for phylo)."""
    g = torch.Generator().manual_seed(99)
    rank_encs = torch.randint(0, 3, (B, R), generator=g).tolist()
    return [{"rank_encs": rank_encs[i], "cid": f"c{int(class_encs_b[i])}", "dataset": "cub"} for i in range(B)]


def _full_reference(crit, img, txt, class_encs_b, targ_data_b, p):
    """Returns (loss, loss_raw, sims) -- sims the branch tuple ((sim,) non-bifurcated, (i2t, t2i)
    bifurcated) whose retained grads after the caller's backward are the ground truth for the chunked
    path's tile-accumulated (branch-summed) grad_sum_sim. Mirrors _loss_full_batch's branch
    construction, incl. its per-term logits under separate logit scalars."""
    clogits = _compute_logits_fn(p)
    clamp = crit.cfg["logits"]["scale"]["clamp"]
    center = crit.cfg["logits"]["bce"]["center"]
    secondaries = (False, True) if crit.sep_scalars else (False,)
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
        logit_scale = (p["scale"], p["scale2"])
    else:
        crit_logits, logit_scale = crit_logits[0], p["scale"]
    loss, loss_raw, _, _ = crit(crit_logits, class_encs_b, targ_data_b, train=True, logit_scale=logit_scale, sim=sims[0])
    return loss, loss_raw, sims


CASES = [
    # (crit, targ1, targ2, lambda_, dsmr, focal, norm_ci, center, neut)
    ("bce", "mp",    "sp",    0.0, True,  2.0, False, None, False),  # baseline
    ("bce", "sp",    "mp",    0.0, True,  2.0, False, None, False),
    ("bce", "tax",   "sp",    0.0, True,  2.0, False, None, False),
    ("bce", "phylo", "sp",    0.0, True,  2.0, False, None, False),
    ("bce", "mp",    "sp",    0.0, False, 0.0, False, None, False),  # no dsmr, no focal
    ("bce", "mp",    "sp",    0.0, True,  2.0, True,  None, False),  # cls_imb.norm
    ("bce", "mp",    "sp",    0.3, True,  2.0, False, None, False),  # blended 0/1 targets
    ("bce", "mp",    "phylo", 0.3, True,  2.0, False, None, False),  # blended with a soft target (dsmr mass: closed form + band sweep)
    ("bce", "tax",   "mp",    0.3, True,  2.0, True,  None, False),  # everything at once
    ("bce", "mp",    "phylo", 1.0, True,  2.0, False, None, False),  # loss2's target alone
    ("bce", "mp",    "sp",    0.0, True,  2.0, False, "sim",        False),  # in-graph global sim mean
    ("bce", "mp",    "sp",    0.0, True,  2.0, False, "grad_proj",  False),  # constant grad projection
    ("bce", "mp",    "sp",    0.0, True,  2.0, False, "grad_proj2", False),
    ("bce", "phylo", "sp",    0.0, True,  2.0, False, "grad_proj",  False),  # soft targets + projection
    ("bce", "mp",    "phylo", 0.3, True,  2.0, False, "grad_proj2", False),  # blend + projection
    ("bce", "mp",    "phylo", 0.5, True,  2.0, True,  "sim",        False),  # blend + centering + norm
    # bif_bce: two-branch tiles, 1D per-anchor weighting, half-live logit scalars
    ("bif_bce", "mp",    "sp",    0.0, False, 0.0, False, None, False),  # bif baseline
    ("bif_bce", "sp",    "mp",    0.0, False, 2.0, False, None, False),
    ("bif_bce", "tax",   "sp",    0.0, True,  2.0, False, None, False),  # row-wise dsmr on soft targets
    ("bif_bce", "phylo", "sp",    0.0, True,  2.0, False, None, True),   # + targ_mass_neut
    ("bif_bce", "mp",    "sp",    0.0, True,  2.0, True,  None, True),   # cls_imb.norm + dsmr + neut
    ("bif_bce", "mp",    "sp",    0.0, False, 2.0, False, "sim",        False),  # per-branch in-graph mean
    ("bif_bce", "mp",    "sp",    0.0, True,  2.0, False, "grad_proj",  True),   # per-branch grad const
    ("bif_bce", "mp",    "sp",    0.0, False, 2.0, False, "grad_proj2", False),
    ("bif_bce", "mp",    "sp",    0.3, True,  2.0, False, None, True),   # blended 0/1 targets, row-wise dsmr + neut on the blend
    ("bif_bce", "mp",    "phylo", 0.5, True,  2.0, True,  "grad_proj",  True),   # blend + everything
]


@pytest.mark.parametrize("C", [16, 48])  # 3 row-blocks, and single-block (== full)
@pytest.mark.parametrize("crit_name,targ1,targ2,lambda_,dsmr,focal,norm_ci,center,neut", CASES)
def test_chunked_matches_full(crit_name, targ1, targ2, lambda_, dsmr, focal, norm_ci, center, neut, C):
    cfg = _cfg(crit=crit_name, lambda_=lambda_, dsmr=dsmr, focal_gamma=focal, cls_imb_norm=norm_ci, center=center, neut=neut)
    _assert_chunked_matches_full(cfg, targ1, targ2, C)


BLEND_CASES = [
    # (crit, targ1, targ2, lambda_, blend_type, unitless, norm_ci, center, neut); focal + dsmr on throughout --
    # the target-dependent factors a loss blend computes per term
    ("bce", "mp",  "sp",    0.3, "loss", False, False, None, False),  # per-term focal + global DSMR mass (closed forms)
    ("bce", "mp",  "phylo", 0.3, "loss", False, True,  None, False),  # soft-target term: band-swept DSMR mass
    ("bce", "mp",  "phylo", 0.3, "loss", True,  False, None, False),  # unitless: pre-swept term magnitudes
    ("bce", "mp",  "phylo", 0.3, "targ", True,  False, None, False),  # unitless over the target blend's one term
    ("bce", "mp",  "sp",    0.0, "targ", True,  False, None, False),  # unitless lone target
    ("bce", "tax", "mp",    0.3, "loss", True,  True,  "sim",        False),  # centered forward in the pre-sweep
    ("bce", "mp",  "phylo", 0.3, "loss", True,  False, "grad_proj",  False),  # grad const over the unitless blend
    ("bce", "mp",  "phylo", 0.3, "loss", False, False, "grad_proj2", False),
    ("bif_bce", "mp", "sp",    0.3, "loss", False, False, None, True),   # per-term row-wise dsmr + neut
    ("bif_bce", "mp", "phylo", 0.3, "loss", True,  True,  None, True),   # unitless bif (L/2 normalizers)
    ("bif_bce", "mp", "phylo", 0.5, "loss", True,  False, "sim",        True),
    ("bif_bce", "mp", "phylo", 0.5, "loss", True,  True,  "grad_proj",  True),
    ("bif_bce", "mp", "phylo", 0.5, "targ", True,  False, "grad_proj2", False),
]


@pytest.mark.parametrize("C", [16, 48])  # 3 row-blocks, and single-block (== full)
@pytest.mark.parametrize("crit_name,targ1,targ2,lambda_,blend_type,unitless,norm_ci,center,neut", BLEND_CASES)
def test_chunked_matches_full_blend_types(crit_name, targ1, targ2, lambda_, blend_type, unitless, norm_ci, center, neut, C):
    cfg = _cfg(crit=crit_name, lambda_=lambda_, cls_imb_norm=norm_ci, center=center, neut=neut,
               blend_type=blend_type, unitless=unitless)
    loss_c = _assert_chunked_matches_full(cfg, targ1, targ2, C)
    if unitless:  # every term enters at unit magnitude: the loss reads the term weights' sum (bif: x2)
        assert loss_c.item() == pytest.approx(2.0 if crit_name == "bif_bce" else 1.0, rel=1e-5)


SEP_CASES = [
    # (crit, targ1, targ2, lambda_, unitless, norm_ci, center, neut); a loss blend on separate logit scalars --
    # every term on its own pair's logits tile, focal + dsmr on throughout
    ("bce", "mp",  "sp",    0.3, False, False, None, False),
    ("bce", "mp",  "phylo", 0.3, True,  True,  None, False),          # unitless: per-term pre-swept magnitudes
    ("bce", "tax", "mp",    0.3, False, False, "sim",        False),  # one in-graph sim mean, each pair's own e^t
    ("bce", "mp",  "phylo", 0.3, True,  False, "grad_proj",  False),  # per-pair grad constants
    ("bce", "mp",  "phylo", 0.3, False, False, "grad_proj2", False),  # ... each at its own e^t
    ("bif_bce", "mp", "sp",    0.3, False, False, None, True),
    ("bif_bce", "mp", "phylo", 0.5, True,  True,  "sim",        True),
    ("bif_bce", "mp", "phylo", 0.5, True,  False, "grad_proj",  True),   # per-pair, per-branch grad constants
    ("bif_bce", "mp", "phylo", 0.5, False, False, "grad_proj2", False),
]


@pytest.mark.parametrize("C", [16, 48])  # 3 row-blocks, and single-block (== full)
@pytest.mark.parametrize("crit_name,targ1,targ2,lambda_,unitless,norm_ci,center,neut", SEP_CASES)
def test_chunked_matches_full_separate_logit_scalars(crit_name, targ1, targ2, lambda_, unitless, norm_ci, center, neut, C):
    cfg = _cfg(crit=crit_name, lambda_=lambda_, cls_imb_norm=norm_ci, center=center, neut=neut,
               blend_type="loss", unitless=unitless, shared=False)
    _assert_chunked_matches_full(cfg, targ1, targ2, C)


def _assert_chunked_matches_full(cfg, targ1, targ2, C):
    device = torch.device("cpu")
    B, K, D, R = 48, 20, 16, 4

    crit = _make_crit(cfg, K, B, targ1=targ1, targ2=targ2)

    g = torch.Generator().manual_seed(0)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = _make_targ_data(B, K, R, class_encs_b)

    # full-batch reference
    img = img0.clone().requires_grad_(True)
    txt = txt0.clone().requires_grad_(True)
    p = _params(1)
    loss_ref, loss_raw_ref, sims_ref = _full_reference(crit, img, txt, class_encs_b, targ_data_b, p)
    loss_ref.backward()
    gsum_ref = sum(s.grad.double().sum().item() for s in sims_ref)
    lambda_eff_ref = crit.lambda_eff  # the full-batch call's, before the chunked run records its own

    # chunked
    imgc = img0.clone().requires_grad_(True)
    txtc = txt0.clone().requires_grad_(True)
    pc = _params(1)
    loss_c, loss_raw_c, stats_c, gsum_c = L.chunked_bce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit, _compute_logits_fn(pc), C, False, device, rank=0, world_size=1,
        hist_bins=HIST_BINS,
    )

    torch.testing.assert_close(loss_c, loss_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(loss_raw_c, loss_raw_ref.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(imgc.grad, img.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(txtc.grad, txt.grad, rtol=1e-4, atol=1e-6)
    assert (p["scale2"].grad is not None) == crit.sep_scalars  # the second pair is live under separate scalars only
    for key in ("scale", "bias") + (("scale2", "bias2") if crit.sep_scalars else ()):
        torch.testing.assert_close(pc[key].grad, p[key].grad, rtol=1e-4, atol=1e-6)
    assert gsum_c == pytest.approx(gsum_ref, rel=1e-4, abs=1e-4)
    # the batch stats carry a unitless loss blend's effective lambda -- from the pre-swept term magnitudes,
    # matching the full-batch criterion's -- and no such key otherwise
    unitless_blend = cfg["unitless"] and cfg["blend"]["type"] == "loss" and 0.0 < cfg["blend"]["lambda"] < 1.0
    assert ("lambda_eff" in stats_c) == (lambda_eff_ref is not None) == unitless_blend
    if unitless_blend:
        assert stats_c["lambda_eff"] == pytest.approx(lambda_eff_ref.item(), rel=1e-4)
    return loss_c


def test_stats_min_max_mean_exact():
    B, C, K, D = 48, 16, 20, 16
    crit = _make_crit(_cfg(), K, B)
    g = torch.Generator().manual_seed(3)
    img = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = [None] * B

    sim = compute_sim(img.detach(), txt.detach(), "cos")
    targs = (class_encs_b.unsqueeze(1) == class_encs_b.unsqueeze(0)).float()
    _, _, stats, _ = L.chunked_bce_loss_backward(
        img, txt, class_encs_b, targ_data_b, crit,
        lambda s, clamp, center=None, center_global=None, half_live=False, secondary=False: s * 10.0 - 0.5, C, False, torch.device("cpu"), rank=0, world_size=1,
        hpsm_kappas=(0.0, 5.0), hist_bins=HIST_BINS,
    )
    assert stats["sim_min"] == pytest.approx(sim.min().item(), abs=1e-5)
    assert stats["sim_max"] == pytest.approx(sim.max().item(), abs=1e-5)
    assert stats["sim_mean"] == pytest.approx(sim.mean().item(), abs=1e-5)
    assert stats["targ_mean"] == pytest.approx(targs.mean().item(), abs=1e-5)
    # the margins stream exactly too, one per kappa: I2T per-row margins just add across tiles, T2I
    # per-column weight sums fold across tiles before the ratio; the mean is their average
    i2t = [L.hard_pair_similarity_margin(sim, targs, kappa).mean().item() for kappa in (0.0, 5.0)]
    t2i = [L.hard_pair_similarity_margin(sim.T, targs.T, kappa).mean().item() for kappa in (0.0, 5.0)]
    assert stats["sim_margin_i2t"] == pytest.approx(i2t, abs=1e-5)
    assert stats["sim_margin_t2i"] == pytest.approx(t2i, abs=1e-5)
    assert stats["sim_margin"] == pytest.approx([0.5 * (a + b) for a, b in zip(i2t, t2i)], abs=1e-5)
    # the streamed probability histogram matches a full-batch one over the stub's logits
    # (sim * 10 - 0.5): counts just add across tiles, so it is exact, not subsampled
    p = (sim * 10.0 - 0.5).sigmoid()
    expected = torch.histc(p, bins=HIST_BINS, min=0.0, max=1.0) / p.numel()
    assert stats["p_hist"] == pytest.approx(expected.tolist(), abs=1e-6)
    # and so does the similarity histogram, its bins spanning the cosine's [-1, 1]
    expected = torch.histc(sim, bins=HIST_BINS, min=-1.0, max=1.0) / sim.numel()
    assert stats["sim_hist"] == pytest.approx(expected.tolist(), abs=1e-6)


def test_stats_over_blended_targets():
    """Under a target blend the stats read the blended target matrix Q = (1 - lambda) Q1 + lambda Q2 -- the
    target histogram, its point stats and the hard-pair margins alike."""
    B, C, K, D, lambda_ = 48, 16, 20, 16, 0.3
    crit = _make_crit(_cfg(lambda_=lambda_), K, B, targ1="mp", targ2="sp")
    g = torch.Generator().manual_seed(3)
    img = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)

    sim = compute_sim(img.detach(), txt.detach(), "cos")
    targs = (1.0 - lambda_) * (class_encs_b.unsqueeze(1) == class_encs_b.unsqueeze(0)).float() + lambda_ * torch.eye(B)
    _, _, stats, _ = L.chunked_bce_loss_backward(
        img, txt, class_encs_b, [None] * B, crit, _compute_logits_fn(_params(1)), C, False, torch.device("cpu"), rank=0, world_size=1,
        hist_bins=HIST_BINS,
    )
    assert stats["targ_min"] == pytest.approx(targs.min().item(), abs=1e-5)
    assert stats["targ_max"] == pytest.approx(targs.max().item(), abs=1e-5)
    assert stats["targ_mean"] == pytest.approx(targs.mean().item(), abs=1e-5)
    expected = torch.histc(targs, bins=HIST_BINS, min=0.0, max=1.0) / targs.numel()
    assert stats["targ_hist"] == pytest.approx(expected.tolist(), abs=1e-6)
    assert "alpha_req_max" not in stats  # the target-implied scale bounds are InfoNCE-only
    # default kappas (0.0,): the bidirectional mean over the blended memberships
    margin = 0.5 * (L.hard_pair_similarity_margin(sim, targs, 0.0).mean() + L.hard_pair_similarity_margin(sim.T, targs.T, 0.0).mean())
    assert stats["sim_margin"] == pytest.approx([margin.item()], abs=1e-5)


def test_batch_diagnostics_off():
    """Disabling either diagnostics component must leave the loss and every gradient identical (the
    hooks/stats only observe): sim_grad_sums=False returns grad_sum_sim None and sim_targ_stats=False
    returns batch_stats None, each flag independent of the other."""
    B, C, K, D = 48, 16, 20, 16
    crit = _make_crit(_cfg(crit="bif_bce", lambda_=0.3), K, B, targ1="mp", targ2="sp")
    g = torch.Generator().manual_seed(3)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)

    def run(sim_grad_sums, sim_targ_stats):
        img = img0.clone().requires_grad_(True)
        txt = txt0.clone().requires_grad_(True)
        p = _params(1)
        loss, loss_raw, stats, gsum = L.chunked_bce_loss_backward(
            img, txt, class_encs_b, [None] * B, crit, _compute_logits_fn(p), C, False, torch.device("cpu"), rank=0, world_size=1,
            sim_grad_sums=sim_grad_sums, sim_targ_stats=sim_targ_stats, hist_bins=HIST_BINS,
        )
        return loss, loss_raw, stats, gsum, img, txt, p

    loss_on, raw_on, stats_on, gsum_on, img_on, txt_on, p_on = run(True, True)
    assert stats_on is not None and gsum_on is not None
    for sim_grad_sums, sim_targ_stats in ((False, False), (True, False), (False, True)):
        loss_off, raw_off, stats_off, gsum_off, img_off, txt_off, p_off = run(sim_grad_sums, sim_targ_stats)
        assert stats_off == (stats_on if sim_targ_stats else None)
        assert gsum_off == (gsum_on if sim_grad_sums else None)
        torch.testing.assert_close(loss_off, loss_on, rtol=0, atol=0)
        torch.testing.assert_close(raw_off, raw_on, rtol=0, atol=0)
        torch.testing.assert_close(img_off.grad, img_on.grad, rtol=0, atol=0)
        torch.testing.assert_close(txt_off.grad, txt_on.grad, rtol=0, atol=0)
        for key in ("scale", "bias"):
            torch.testing.assert_close(p_off[key].grad, p_on[key].grad, rtol=0, atol=0)


@pytest.mark.parametrize("center", [None, "grad_proj", "grad_proj2"])
def test_chunked_matches_full_under_mixed_precision_with_the_real_head(center):
    # the head (VLMWrapper.compute_logits) casts the bf16 sims to float32 BEFORE scaling / biasing / projecting,
    # so the grad_proj* projection node lives in float32. The chunked path's prepass has to measure its centering
    # constant at that node: probing from a bf16 sim leaf returns the node's gradient rounded to bf16 through the
    # cast's backward -- the mean of the rounded grads -- and the tiles then subtract a constant the full-batch
    # projection never saw (sum dL/dS off, the towers' gradients with it). Exercised with the ACTUAL head under
    # bf16 autocast: the handwritten mirror and mixed_prec=False used elsewhere in this file cannot see it
    from types import SimpleNamespace
    from models import VLMWrapper

    device = torch.device("cpu")
    B, K, D, C = 32, 12, 16, 8
    crit = _make_crit(_cfg(center=center), K, B)
    g = torch.Generator().manual_seed(0)
    img0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    txt0 = torch.nn.functional.normalize(torch.randn(B, D, generator=g), dim=1)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    targ_data_b = _make_targ_data(B, K, 4, class_encs_b)

    def head():
        model = SimpleNamespace(logit_scale=torch.nn.Parameter(torch.tensor(2.3)), logit_bias=torch.nn.Parameter(torch.tensor(-0.5)))
        stub = SimpleNamespace(_unwrapped_model=model)
        return model, lambda *args, **kwargs: VLMWrapper.compute_logits(stub, *args, **kwargs)

    img, txt = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True)
    model, compute_logits = head()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sim = compute_sim(img, txt, "cos")
        assert sim.dtype == torch.bfloat16  # else the case under test is not being exercised
        sim.retain_grad()
        loss, *_ = crit(compute_logits(sim, False, center), class_encs_b, targ_data_b, True, model.logit_scale, sim)
    loss.backward()

    imgc, txtc = img0.clone().requires_grad_(True), txt0.clone().requires_grad_(True)
    modelc, compute_logits_c = head()
    loss_c, _, _, gsum_c = L.chunked_bce_loss_backward(
        imgc, txtc, class_encs_b, targ_data_b, crit, compute_logits_c, C, True, device, rank=0, world_size=1,
        hist_bins=HIST_BINS,
    )

    torch.testing.assert_close(loss_c, loss.detach(), rtol=1e-5, atol=1e-6)
    # the two quantities that do not depend on bf16 accumulation order, and so isolate the centering constant:
    # each tile's dL/dS entries (elementwise off that constant) and the image grads (row i reads its own tile
    # alone). Measured over six seeds: with the float32 probe sum dL/dS agrees to 2e-9 and the image grads
    # exactly; with a bf16 probe they are off by 1e-3..1e-1 and 5e-5..5e-4
    assert gsum_c == pytest.approx(sim.grad.double().sum().item(), abs=1e-5)
    assert ((imgc.grad - img.grad).norm() / img.grad.norm()).item() < 1e-6
    # the text grads DO depend on it -- the full path accumulates a 32-term bf16 dot product in one pass, the
    # tiles sum 8-row partials -- so they differ at the bf16-ulp level (~2e-3) with or without the projection:
    # inherent to chunking under bf16, and bounded here only against a gross error
    assert ((txtc.grad - txt.grad).norm() / txt.grad.norm()).item() < 1e-2
    torch.testing.assert_close(modelc.logit_scale.grad, model.logit_scale.grad, rtol=1e-4, atol=1e-5)


def test_chunking_unsupported_with_infonce():
    assert not L.chunking_supported({"crit": "infonce"})


@pytest.mark.parametrize("crit", ["bce", "bif_bce"])
def test_chunking_supported(crit):
    assert L.chunking_supported({"crit": crit})


def test_chunked_asserts_on_geo_sim_center():
    # center: sim needs the cos mean factorization for an exact global mean; TrainConfig rejects the
    # combo at config time, and the chunked entrypoint guards direct callers too
    B, K = 16, 5
    crit = _make_crit(_cfg(sim="geo1", center="sim"), K, B)
    g = torch.Generator().manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    txt = torch.nn.functional.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    with pytest.raises(AssertionError, match="requires cos sim"):
        L.chunked_bce_loss_backward(img, txt, class_encs_b, [None] * B, crit, _compute_logits_fn(_params(1)), 8, False,
                                    torch.device("cpu"), rank=0, world_size=1, hist_bins=HIST_BINS)


def _synthetic_vcv():
    vcv = RealPhyloVCV.__new__(RealPhyloVCV)  # bypass tree loading
    K = 8
    rng = np.random.default_rng(0)
    A = rng.random((K, K))
    targs = (A + A.T) / 2.0
    # Deliberately non-1.0 diagonal (the real target matrix has a unit diagonal -- exp(0)) so the
    # same-cid overwrite is observable rather than a no-op in this test.
    np.fill_diagonal(targs, rng.uniform(0.3, 0.9, size=K))
    targs[0, 0] = 1.0
    vcv.targs = targs
    vcv._cid_to_idx = {f"c{i}": i for i in range(K)}
    return vcv


def test_phylo_block_matches_full():
    """Real PhyloVCV.make_targ_block_fn reproduces the [rs:re, :] block of get_targs_batch (incl. the
    same-cid overwrite), on a synthetic target matrix with all-in-tree cids and repeats."""
    vcv = _synthetic_vcv()
    B = 20
    targ_data_b = [{"cid": f"c{i % 8}"} for i in range(B)]

    full = vcv.get_targs_batch(targ_data_b)
    blk_fn = vcv.make_targ_block_fn(targ_data_b, torch.device("cpu"))
    for rs in range(0, B, 6):
        re = min(rs + 6, B)
        torch.testing.assert_close(blk_fn(rs, re), full[rs:re])


def test_phylo_same_cid_pinned_to_one():
    """Same-cid pairs are forced to 1.0 even when targs' diagonal disagrees (the real matrix's
    diagonal is exp(0) = 1.0; the synthetic matrix keeps it below so the overwrite is observable)."""
    vcv = _synthetic_vcv()
    assert vcv.targs[1, 1] < 1.0
    targ_data_b = [{"cid": "c1"}, {"cid": "c1"}, {"cid": "c2"}]  # samples 0 and 1 share c1
    full = vcv.get_targs_batch(targ_data_b)
    assert full[0, 0] == 1.0 and full[0, 1] == 1.0 and full[1, 0] == 1.0  # same-cid -> 1.0
    assert full[0, 2] == pytest.approx(vcv.targs[1, 2])  # cross-species keeps the target value

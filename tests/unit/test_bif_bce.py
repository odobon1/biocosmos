"""
Unit tests for bifurcated BCE (loss.crit: bif_bce) and the unified full-batch loss path.

Contract under test (see BifurcatedBCECriterion / _loss_for_crit_full_batch): with no anchor-wise
method active (cls_imb: null, dsmr off, targ_mass_neut off; focal is value-symmetric
across branches so it may stay on), bifurcated BCE reads 2x the non-bifurcated loss (un-halved
branch sum) while every gradient -- towers, logit scale/bias (half-live) -- matches non-bifurcated
BCE 1x. The tests bind the REAL VLMWrapper methods to a lightweight harness `self` (single
process, world_size 1, so _gather_batch is a no-op) and also cover the branch tuples' shapes, the
per-branch tower routing, eval mode, the sp no-op of targ_mass_neut, and a bif/non-bif
loss2 mix through _global_batch_loss.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.loss as L
from models import VLMWrapper


def _cfg(crit, targ="mp", cls_imb=None, focal_gamma=0.0, dsmr=False, neut=False, center=None):
    return {
        "crit": crit, "sim": "cos", "targ": targ,
        "bce": {"targ_mass_neut": neut},
        "wting": {
            "cls_imb": {"type": cls_imb, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999},
                        "freq_type_2d": "naive", "wt_mean_type": "per_class", "norm": False},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),  # config load prunes the block when gamma = 0.0
            "bce": {"dsmr": dsmr},
        },
        "logits": {"temp": {"clamp": False}, "bce": {"center": center, "bias": {}}},
    }


CRIT_CLS = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion}


def _make_crit(cfg, K, B):
    cls = CRIT_CLS[cfg["crit"]]
    crit = cls.__new__(cls)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.device = torch.device("cpu")
    crit.batch_size = B
    g = torch.Generator().manual_seed(K)
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


class Toy(nn.Module):
    def __init__(self, with_secondary=False):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))
        if with_secondary:
            self.logit_scale2 = nn.Parameter(torch.tensor(1.7))
            self.logit_bias2 = nn.Parameter(torch.tensor(0.2))


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = VLMWrapper._unwrapped_model
    compute_logits = VLMWrapper.compute_logits
    _loss_for_crit_full_batch = VLMWrapper._loss_for_crit_full_batch
    _gather_batch = VLMWrapper._gather_batch
    _global_batch_loss = VLMWrapper._global_batch_loss


def _make_harness(model, crit1, crit2=None, mix=0.0, mix_unit_scale=False):
    h = Harness()
    h.model = model
    h.crit1 = crit1
    h.crit2 = crit2
    h.world_size = 1
    h.cfg = SimpleNamespace(loss2={"mix": mix, "mix_unit_scale": mix_unit_scale})
    return h


def _data(B, K, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = F.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = F.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    return img, txt, class_encs_b


def _run_full(crit, img, txt, class_encs_b, train=True):
    toy = Toy().train(train)
    h = _make_harness(toy, crit)
    loss, loss_raw, logits, sims, targs = h._loss_for_crit_full_batch(
        img, txt, class_encs_b, [None] * img.size(0), crit
    )
    return toy, loss, loss_raw, logits, sims, targs


@pytest.mark.parametrize("focal_gamma,center", [
    (0.0, None),
    (2.0, None),
    (2.0, "sim"),
    (2.0, "grad_proj"),
    (2.0, "grad_proj2"),
])
def test_bif_matches_bce_2x_value_1x_grads(focal_gamma, center):
    B, K, D = 32, 8, 16
    res = {}
    for crit_name in ("bce", "bif_bce"):
        img, txt, class_encs_b = _data(B, K, D)
        crit = _make_crit(_cfg(crit_name, focal_gamma=focal_gamma, center=center), K, B)
        toy, loss, loss_raw, logits, sims, _ = _run_full(crit, img, txt, class_encs_b)
        loss.backward()
        res[crit_name] = {
            "loss": loss.item(), "loss_raw": loss_raw.item(),
            "img_g": img.grad, "txt_g": txt.grad,
            "scale_g": toy.logit_scale.grad, "bias_g": toy.logit_bias.grad,
            "n_branches": (len(logits), len(sims)),
        }
    assert res["bce"]["n_branches"] == (1, 1)
    assert res["bif_bce"]["n_branches"] == (2, 2)
    assert res["bif_bce"]["loss"] == pytest.approx(2.0 * res["bce"]["loss"], rel=1e-5)
    assert res["bif_bce"]["loss_raw"] == pytest.approx(2.0 * res["bce"]["loss_raw"], rel=1e-5)
    for gk in ("img_g", "txt_g", "scale_g", "bias_g"):
        torch.testing.assert_close(res["bif_bce"][gk], res["bce"][gk], rtol=1e-5, atol=1e-7)


def test_bif_branches_route_one_tower_each():
    B, K, D = 16, 5, 8
    img, txt, class_encs_b = _data(B, K, D)
    crit = _make_crit(_cfg("bif_bce"), K, B)
    _, _, _, _, sims, _ = _run_full(crit, img, txt, class_encs_b)
    # i2t branch has no graph path into the text embeddings, t2i none into the image embeddings
    assert torch.autograd.grad(sims[0].sum(), txt, retain_graph=True, allow_unused=True)[0] is None
    assert torch.autograd.grad(sims[1].sum(), img, retain_graph=True, allow_unused=True)[0] is None


def test_bif_criterion_consumes_t2i_transposed():
    # Orientation pin: every equivalence case above runs both branches on identical values (where
    # a dropped/misplaced transpose is invisible against symmetric Y), so feed the criterion a
    # DISTINCT (A, C) pair with non-uniform row weights and check against an inline reference
    # that consumes C transposed ([txt-row, img-col] anchors), with dsmr exercising
    # _dsmr_weight_rows in train mode.
    B, K = 12, 4
    g = torch.Generator().manual_seed(11)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    A = torch.randn(B, B, generator=g)
    C = torch.randn(B, B, generator=g)
    crit = _make_crit(_cfg("bif_bce", cls_imb="inv_freq", dsmr=True, neut=True), K, B)

    loss, _, _ = crit((A, C), class_encs_b, [None] * B, train=True, logit_scale=None)

    Y = (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float()
    w = (1.0 / crit.counts[class_encs_b].pow(0.5)).float()
    W_dsmr = L._dsmr_weight_rows(Y, B)
    ref = 0.0
    for Z in (A, C.T):
        bce = F.binary_cross_entropy_with_logits(Z, Y, reduction="none")
        ref = ref + ((w[:, None] * W_dsmr * bce).sum(dim=1) / Y.sum(dim=1)).mean()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_bif_eval_loss_equals_raw_2x():
    B, K, D = 16, 5, 8
    raws = {}
    for crit_name in ("bce", "bif_bce"):
        img, txt, class_encs_b = _data(B, K, D)
        # neut/dsmr/focal on: eval must ignore every weighting knob and return the raw reading
        crit = _make_crit(_cfg(crit_name, focal_gamma=2.0, dsmr=True, neut=True), K, B)
        _, loss, loss_raw, _, _, _ = _run_full(crit, img, txt, class_encs_b, train=False)
        assert loss.item() == pytest.approx(loss_raw.item())
        raws[crit_name] = loss_raw.item()
    assert raws["bif_bce"] == pytest.approx(2.0 * raws["bce"], rel=1e-5)


def test_targ_mass_neut_noop_under_iw_active_under_sw():
    B, K, D = 16, 5, 8
    losses = {}
    for targ in ("sp", "mp"):
        for neut in (False, True):
            img, txt, class_encs_b = _data(B, K, D)
            crit = _make_crit(_cfg("bif_bce", targ=targ, neut=neut), K, B)
            _, loss, _, _, _, _ = _run_full(crit, img, txt, class_encs_b)
            losses[(targ, neut)] = loss.item()
    # sp rows carry unit target mass -> neutralization divides by 1 (no-op)
    assert losses[("sp", True)] == pytest.approx(losses[("sp", False)], rel=1e-6)
    # mp with class repeats (K < B) has rows with mass > 1 -> neutralization changes the loss
    assert abs(losses[("mp", True)] - losses[("mp", False)]) > 1e-6


def test_mix_unit_scale_bif_grads_match_non_bif():
    # the unit-scale normalizer divides a bifurcated loss by L/2 -- its gradient-scale-equivalent
    # value -- so swapping a blend participant bce <-> bif_bce leaves every gradient unchanged
    # (`mix` keeps the true gradient-contribution ratio); the bif participant's normalized value
    # reads its 2x (blend = (1 - mix) * 2 + mix * 1) where each non-bif participant reads 1
    B, K, D = 16, 5, 8
    mix = 0.3
    res = {}
    for crit1_name in ("bce", "bif_bce"):
        img, txt, class_encs_b = _data(B, K, D)
        crit1 = _make_crit(_cfg(crit1_name), K, B)
        crit2 = _make_crit(_cfg("bce", targ="sp"), K, B)
        toy = Toy(with_secondary=True).train()
        h = _make_harness(toy, crit1, crit2, mix=mix, mix_unit_scale=True)
        loss, *_ = h._global_batch_loss(img, txt, class_encs_b, [None] * B)
        loss.backward()
        res[crit1_name] = {
            "loss": loss.item(),
            "img_g": img.grad, "txt_g": txt.grad,
            "scale_g": toy.logit_scale.grad, "bias_g": toy.logit_bias.grad,
            "scale2_g": toy.logit_scale2.grad, "bias2_g": toy.logit_bias2.grad,
        }
    assert res["bce"]["loss"] == pytest.approx(1.0)
    assert res["bif_bce"]["loss"] == pytest.approx((1.0 - mix) * 2.0 + mix * 1.0)
    for gk in ("img_g", "txt_g", "scale_g", "bias_g", "scale2_g", "bias2_g"):
        torch.testing.assert_close(res["bif_bce"][gk], res["bce"][gk], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("crit1_name,crit2_name", [("bif_bce", "bce"), ("bce", "bif_bce")])
def test_loss2_mix_through_global_batch_loss(crit1_name, crit2_name):
    B, K, D = 16, 5, 8
    img, txt, class_encs_b = _data(B, K, D)
    crit1 = _make_crit(_cfg(crit1_name), K, B)
    crit2 = _make_crit(_cfg(crit2_name, targ="sp"), K, B)
    toy = Toy(with_secondary=True).train()
    h = _make_harness(toy, crit1, crit2, mix=0.3, mix_unit_scale=True)
    loss, loss_raw, _, _, (logits1, logits2), _, batch_stats, (sims1, sims2) = h._global_batch_loss(
        img, txt, class_encs_b, [None] * B
    )
    assert len(logits1) == len(sims1) == (2 if crit1_name == "bif_bce" else 1)
    assert len(logits2) == len(sims2) == (2 if crit2_name == "bif_bce" else 1)
    assert torch.isfinite(loss) and torch.isfinite(loss_raw)
    loss.backward()
    # both towers and all four logit scalars receive grads through the blend
    assert img.grad is not None and txt.grad is not None
    for pname in ("logit_scale", "logit_bias", "logit_scale2", "logit_bias2"):
        assert getattr(toy, pname).grad is not None, pname
    # every branch's grad was retained for the aggregate (branch-summed) grad logging
    for t in (*logits1, *sims1, *logits2, *sims2):
        assert t.grad is not None
    for tag in ("1", "2"):
        assert batch_stats[f"sim{tag}_min"] <= batch_stats[f"sim{tag}_max"]

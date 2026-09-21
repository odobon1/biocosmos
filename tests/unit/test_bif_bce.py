"""
Unit tests for bifurcated BCE (loss.crit: bif_bce) and the unified full-batch loss path.

Contract under test (see BifurcatedBCECriterion / _loss_full_batch): with no anchor-wise
method active (cls_imb: null, dsmr off, targ_mass_neut off; focal is value-symmetric
across branches so it may stay on), bifurcated BCE reads 2x the non-bifurcated loss (un-halved
branch sum) while every gradient -- towers, logit scale/bias (half-live) -- matches non-bifurcated
BCE 1x. The tests bind the REAL VLMWrapper methods to a lightweight harness `self` (single
process, world_size 1, so _gather_batch is a no-op) and also cover the branch tuples' shapes, the
per-branch tower routing, eval mode, the sp no-op of targ_mass_neut, a target blend (loss.blend.lambda)
through _global_batch_loss, and the InfoNCE batch stats.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.loss as L
from models import VLMWrapper


def _cfg(crit, lambda_=0.0, cls_imb=None, focal_gamma=0.0, dsmr=False, neut=False, center=None):
    """The loss-level config (train.yaml's `loss` block, as the criterion reads it)."""
    return {
        "crit": crit, "sim": "cos", "blend": {"lambda": lambda_, "type": "targ"}, "unitless": False,
        "infonce": {"block_residuals": False},
        "bce": {"targ_mass_neut": neut},
        "wting": {
            "cls_imb": {"type": cls_imb, "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999},
                        "norm": False},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),  # config load prunes the block when gamma = 0.0
            "bce": {"dsmr": dsmr},
        },
        "logits": {"shared": True, "scale": {"clamp": False}, "bce": {"center": center, "bias": {}}},
    }


def _targ(targ, tsm_type="linear"):
    """A target spec (train.yaml's loss1 / loss2 block); the tsm is read by InfoNCE only."""
    return {"targ": targ, "infonce": {"tsm": {"type": tsm_type, "sm_scale": 1.0}}}


CRIT_CLS = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion, "infonce": L.InfoNCECriterion}
HIST_BINS = 20  # the harness' reporting.learning_curves.hist_bins


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


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))


class Harness:
    """Fake VLMWrapper `self` carrying the real methods verbatim."""
    _unwrapped_model = VLMWrapper._unwrapped_model
    compute_logits = VLMWrapper.compute_logits
    _loss_full_batch = VLMWrapper._loss_full_batch
    _batch_stats = VLMWrapper._batch_stats
    _gather_batch = VLMWrapper._gather_batch
    _global_batch_loss = VLMWrapper._global_batch_loss


def _make_harness(model, crit):
    h = Harness()
    h.model = model
    h.crit = crit
    h.world_size = 1
    # _batch_stats reads the loss config to decide whether p_hist (sigmoid) stats apply and whether the
    # InfoNCE diagnostics run
    h.cfg = SimpleNamespace(
        loss=crit.cfg,
        reporting={"batch_diagnostics": {"emb_logit_grads": True, "sim_grad_sums": True, "sim_targ_stats": True},
                 "learning_curves": {"hpsm": {"kappas": [0.0, 3.0]}, "hist_bins": HIST_BINS}},
    )
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
    loss, loss_raw, logits, sims, targs, _ = h._loss_full_batch(img, txt, class_encs_b, [None] * img.size(0))
    return toy, loss, loss_raw, logits, sims, targs


@pytest.mark.parametrize("focal_gamma,center", [
    (0.0, None),
    (2.0, None),
    (2.0, "sim"),
    (2.0, "grad_proj"),
    (2.0, "grad_proj2"),
])
@pytest.mark.parametrize("lambda_", [0.0, 0.3])  # a lone target, and an mp / sp blend
def test_bif_matches_bce_2x_value_1x_grads(focal_gamma, center, lambda_):
    B, K, D = 32, 8, 16
    res = {}
    for crit_name in ("bce", "bif_bce"):
        img, txt, class_encs_b = _data(B, K, D)
        crit = _make_crit(_cfg(crit_name, lambda_=lambda_, focal_gamma=focal_gamma, center=center), K, B)
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

    loss, _, _, _ = crit((A, C), class_encs_b, [None] * B, train=True, logit_scale=None, sim=None)

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
            crit = _make_crit(_cfg("bif_bce", neut=neut), K, B, targ1=targ)
            _, loss, _, _, _, _ = _run_full(crit, img, txt, class_encs_b)
            losses[(targ, neut)] = loss.item()
    # sp rows carry unit target mass -> neutralization divides by 1 (no-op)
    assert losses[("sp", True)] == pytest.approx(losses[("sp", False)], rel=1e-6)
    # mp with class repeats (K < B) has rows with mass > 1 -> neutralization changes the loss
    assert abs(losses[("mp", True)] - losses[("mp", False)]) > 1e-6


@pytest.mark.parametrize("crit_name", ["bce", "bif_bce"])
def test_target_blend_through_global_batch_loss(crit_name):
    B, K, D, lambda_ = 16, 5, 8, 0.3
    img, txt, class_encs_b = _data(B, K, D)
    crit = _make_crit(_cfg(crit_name, lambda_=lambda_), K, B, targ1="mp", targ2="sp")
    toy = Toy().train()
    h = _make_harness(toy, crit)
    loss, loss_raw, _, _, logits, _, batch_stats, sims = h._global_batch_loss(img, txt, class_encs_b, [None] * B)
    assert len(logits) == len(sims) == (2 if crit_name == "bif_bce" else 1)
    assert torch.isfinite(loss) and torch.isfinite(loss_raw)
    loss.backward()
    # both towers and both logit scalars receive grads
    assert img.grad is not None and txt.grad is not None
    assert toy.logit_scale.grad is not None and toy.logit_bias.grad is not None
    # every branch's grad was retained for the aggregate (branch-summed) grad logging
    for t in (*logits, *sims):
        assert t.grad is not None
    assert batch_stats["sim_min"] <= batch_stats["sim_max"]
    # a BCE-family loss reports a probability histogram over sigmoid(its logits): bin fractions summing to 1
    p = logits[0].detach().sigmoid()
    hist = batch_stats["p_hist"]
    assert len(hist) == HIST_BINS
    assert sum(hist) == pytest.approx(1.0, abs=1e-5)
    expected = torch.histc(p, bins=HIST_BINS, min=0.0, max=1.0) / p.numel()
    assert hist == pytest.approx(expected.tolist(), abs=1e-5)
    # the similarities get one too, its bins spanning the cosine's [-1, 1]
    s = sims[0].detach()
    expected_s = torch.histc(s, bins=HIST_BINS, min=-1.0, max=1.0) / s.numel()
    assert batch_stats["sim_hist"] == pytest.approx(expected_s.tolist(), abs=1e-5)
    assert sum(batch_stats["sim_hist"]) == pytest.approx(1.0, abs=1e-5)
    assert "alpha_req_max" not in batch_stats  # the target-implied scale bounds are InfoNCE-only
    # the target stats read the blended target matrix Q = (1 - lambda) MP + lambda I: its histogram, and one
    # margin per configured kappa per direction plus their mean, all over the blended memberships
    Q = (1.0 - lambda_) * (class_encs_b.unsqueeze(0) == class_encs_b.unsqueeze(1)).float() + lambda_ * torch.eye(B)
    expected_q = torch.histc(Q, bins=HIST_BINS, min=0.0, max=1.0) / Q.numel()
    assert batch_stats["targ_hist"] == pytest.approx(expected_q.tolist(), abs=1e-5)
    assert batch_stats["targ_mean"] == pytest.approx(Q.mean().item(), abs=1e-5)
    s = sims[0].detach()
    i2t = [L.hard_pair_similarity_margin(s, Q, kappa).mean().item() for kappa in (0.0, 3.0)]
    t2i = [L.hard_pair_similarity_margin(s.T, Q.T, kappa).mean().item() for kappa in (0.0, 3.0)]
    assert batch_stats["sim_margin_i2t"] == pytest.approx(i2t, abs=1e-5)
    assert batch_stats["sim_margin_t2i"] == pytest.approx(t2i, abs=1e-5)
    assert batch_stats["sim_margin"] == pytest.approx([0.5 * (a + b) for a, b in zip(i2t, t2i)], abs=1e-5)


@pytest.mark.parametrize("crit_name", ["bce", "bif_bce", "infonce"])
def test_batch_stats_carry_a_unitless_loss_blends_lambda_eff(crit_name):
    # the lambda_eff curve strip's series: recorded by the criterion where unitless reweights a loss blend's two
    # terms (Criterion.term_coeffs), absent from the batch stats on static weights
    B, K, D = 16, 5, 8
    cfg = _cfg(crit_name, lambda_=0.3)
    for unitless_blend in (False, True):
        if unitless_blend:
            cfg["blend"]["type"], cfg["unitless"] = "loss", True
        img, txt, class_encs_b = _data(B, K, D)
        h = _make_harness(Toy().train(), _make_crit(cfg, K, B, targ1="mp", targ2="sp"))
        *_, batch_stats, _ = h._global_batch_loss(img, txt, class_encs_b, [None] * B)
        assert ("lambda_eff" in batch_stats) == unitless_blend
    assert batch_stats["lambda_eff"] == pytest.approx(h.crit.lambda_eff.item())
    assert 0.0 < batch_stats["lambda_eff"] < 1.0


@pytest.mark.parametrize("center", ["sim", "grad_proj", "grad_proj2"])
@pytest.mark.parametrize("focal_gamma", [0.0, 2.0])
def test_bce_centering_is_not_read_under_infonce(center, focal_gamma):
    # loss.logits.bce.center is the sigmoid path's and declared inert under InfoNCE, so it must leave an InfoNCE
    # run BIT-identical. Read, it is a no-op only in exact arithmetic -- a row softmax is shift-invariant, so
    # dL/dsim has zero row / column sums under any weighting that reads the logits through p (focal included):
    # grad_proj* subtract a mean that is zero, `sim` shifts every logit alike -- and perturbs the run at ~1e-8
    K, B = 6, 24

    def grads(center_):
        crit = _make_crit(_cfg("infonce", focal_gamma=focal_gamma, center=center_), K, B)
        g = torch.Generator().manual_seed(0)
        img = F.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
        txt = F.normalize(torch.randn(B, 8, generator=g), dim=1).requires_grad_(True)
        class_encs_b = torch.randint(0, K, (B,), generator=g)
        h = _make_harness(Toy().train(), crit)
        loss, *_ = h._loss_full_batch(img, txt, class_encs_b, [None] * B)
        loss.backward()
        return loss.detach(), img.grad, txt.grad, h.model.logit_scale.grad

    for got, want in zip(grads(center), grads(None)):
        assert torch.equal(got, want)


def test_infonce_stats():
    # p_hist is the sigmoid-BCE pair probability; an InfoNCE loss gets none (its row-softmax mean is a
    # fixed 1/B). It gets the row-wise target-implied scale bounds (alpha_req_*) and the logit-scale
    # gradient decomposition (dalpha_* / dlogalpha_*), whose full / all sums are d(loss_raw)/d(alpha)
    # and d(loss_raw)/d(log alpha) of the raw bidirectional InfoNCE over the blended target -- the
    # latter its logit_scale grad outright, since the parameter is log(alpha)
    B, K, D = 16, 5, 8
    img, txt, class_encs_b = _data(B, K, D)
    crit = _make_crit(_cfg("infonce", lambda_=0.3), K, B, targ1="mp", targ2="sp")
    toy = Toy().train()
    h = _make_harness(toy, crit)

    _, loss_raw, *_, batch_stats, _ = h._global_batch_loss(img, txt, class_encs_b, [None] * B)

    assert "p_hist" not in batch_stats
    assert "sim_min" in batch_stats and "targ_min" in batch_stats  # sim/targ still reported
    prefixes, aggs, comps = ("dalpha", "dlogalpha"), ("sum", "sum_abs", "C", "row_abs", "C_row"), ("full", "struct", "res", "sres", "ires")
    assert {key for key in batch_stats if "alpha" in key} == {
        f"{prefix}_{agg}_{comp}" for prefix in prefixes for agg in aggs for comp in comps
    } | {f"{p}alpha_req_{stat}" for p in ("", "log_") for stat in ("min", "mean", "max")}
    for prefix in prefixes:
        for agg in aggs:
            for comp in comps:
                assert len(batch_stats[f"{prefix}_{agg}_{comp}"]) == 3  # [all, pos, neg]
    (g_log_scale,) = torch.autograd.grad(loss_raw, toy.logit_scale)
    alpha = toy.logit_scale.detach().exp()
    assert batch_stats["dalpha_sum_full"][0] == pytest.approx((g_log_scale / alpha).item(), rel=1e-4)
    assert batch_stats["dlogalpha_sum_full"][0] == pytest.approx(g_log_scale.item(), rel=1e-4)

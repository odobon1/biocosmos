"""
The target blend's acceptance contract: cross-entropy and BCE are linear in the target, so a single
criterion over Y = (1 - lambda) * Y1 + lambda * Y2 reproduces the loss blend (1 - lambda) * L(Y1) + lambda * L(Y2)
of two single-target criteria -- loss value, raw loss and every gradient -- whenever no target-dependent
weighting is on (class-imbalance weights are target-independent, so they may stay on). Focal weighting
reads the target, so with it on the two differ. Checked per criterion family, InfoNCE under each pairing
of per-target simplex mappings.

loss.blend.type loss IS that loss blend -- each term its single-target criterion's weighted loss, focal /
DSMR / targ_mass_neut reading the term's own target -- and loss.unitless enters each term at unit magnitude.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.loss as L
from models import VLMWrapper


def _cfg(crit, lambda_, focal_gamma=0.0, blend_type="targ", unitless=False, dsmr=False, neut=False, shared=True):
    return {
        "crit": crit, "sim": "cos", "blend": {"lambda": lambda_, "type": blend_type}, "unitless": unitless,
        "bce": {"targ_mass_neut": neut},
        "wting": {
            "cls_imb": {"type": "inv_freq", "inv_freq": {"gamma": 0.5}, "class_bal": {"beta": 0.9999}, "norm": False},
            **({"focal": {"gamma": focal_gamma}} if focal_gamma > 0.0 else {}),
            "bce": {"dsmr": dsmr},
        },
        "logits": {"shared": shared, "scale": {"clamp": False}, "bce": {"center": None, "bias": {}}},
    }


def _targ(targ, tsm_type):
    return {"targ": targ, "infonce": {"tsm": {"type": tsm_type, "sm_scale": 3.0}}}


CRIT_CLS = {"bce": L.BCECriterion, "bif_bce": L.BifurcatedBCECriterion, "infonce": L.InfoNCECriterion}


def _make_crit(cfg, specs, K, B):
    """A criterion over explicit (weight, target spec) pairs -- Criterion.targ_specs as built from
    loss.blend.lambda, or a lone (1.0, spec) for a single-target reference."""
    cls = CRIT_CLS[cfg["crit"]]
    crit = cls.__new__(cls)  # bypass build_wting (no dataset needed)
    crit.cfg = cfg
    crit.targ_specs = specs
    crit.device = torch.device("cpu")
    crit.batch_size = B
    g = torch.Generator().manual_seed(K)
    crit.counts = torch.randint(1, 1000, (K,), generator=g).to(torch.float64)
    crit.wt_mean = 1.0
    return crit


class Toy(nn.Module):
    def __init__(self, scalars2=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-0.5))
        if scalars2 is not None:  # loss2's term's own pair (separate logit scalars)
            self.logit_scale2 = nn.Parameter(torch.tensor(scalars2[0]))
            self.logit_bias2 = nn.Parameter(torch.tensor(scalars2[1]))


class Harness:
    _unwrapped_model = VLMWrapper._unwrapped_model
    compute_logits = VLMWrapper.compute_logits
    _loss_full_batch = VLMWrapper._loss_full_batch


def _forward(crit, seed=0, B=24, K=6, D=8, toy=None):
    """Loss, raw loss and returned target distribution y of `crit` on a fixed batch, plus the grad leaves."""
    g = torch.Generator().manual_seed(seed)
    img = F.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    txt = F.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_(True)
    class_encs_b = torch.randint(0, K, (B,), generator=g)
    rank_encs = torch.randint(0, 3, (B, 3), generator=g).tolist()
    targ_data_b = [{"rank_encs": rank_encs[i]} for i in range(B)]  # tax targets read these
    toy = (toy or Toy()).train()
    h = Harness()
    h.model, h.crit = toy, crit
    loss, loss_raw, _, _, _, y = h._loss_full_batch(img, txt, class_encs_b, targ_data_b)
    return loss, loss_raw, y, (img, txt, toy.logit_scale, toy.logit_bias)


def _run(crit, toy=None):
    """Loss, raw loss and the (img, txt, scale, bias) grads of `crit` on a fixed batch."""
    loss, loss_raw, _, leaves = _forward(crit, toy=toy)
    loss.backward()
    return loss.detach(), loss_raw.detach(), tuple(leaf.grad for leaf in leaves)


CASES = [
    # (crit, tsm1, tsm2)
    ("bce", "linear", "linear"),
    ("bif_bce", "linear", "linear"),
    ("infonce", "linear", "linear"),
    ("infonce", "linear", "softmax"),
    ("infonce", "softmax", "softmax"),
]


@pytest.mark.parametrize("blend_type", ["targ", "loss"])  # without a target-dependent factor the blend types coincide
@pytest.mark.parametrize("targ1,targ2", [("mp", "sp"), ("mp", "tax")])
@pytest.mark.parametrize("crit,tsm1,tsm2", CASES)
def test_target_blend_equals_loss_blend(crit, tsm1, tsm2, targ1, targ2, blend_type):
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ(targ1, tsm1), _targ(targ2, tsm2)
    blend = _make_crit(_cfg(crit, lambda_, blend_type=blend_type), L.targ_specs(lambda_, spec1, spec2), K, B)
    single1 = _make_crit(_cfg(crit, 0.0), [(1.0, spec1)], K, B)
    single2 = _make_crit(_cfg(crit, 0.0), [(1.0, spec2)], K, B)

    loss_b, raw_b, grads_b = _run(blend)
    loss_1, raw_1, grads_1 = _run(single1)
    loss_2, raw_2, grads_2 = _run(single2)

    torch.testing.assert_close(loss_b, (1.0 - lambda_) * loss_1 + lambda_ * loss_2, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(raw_b, (1.0 - lambda_) * raw_1 + lambda_ * raw_2, rtol=1e-5, atol=1e-7)
    for g_b, g_1, g_2 in zip(grads_b, grads_1, grads_2):
        torch.testing.assert_close(g_b, (1.0 - lambda_) * g_1 + lambda_ * g_2, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("crit", ["bce", "infonce"])
def test_focal_breaks_the_loss_blend_identity(crit):
    # focal weights |Y - p|^gamma read the target, so the single weight on the blended Y is not the
    # blend of the two per-target weights: the identity above holds only for target-independent weighting
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("sp", "linear")
    blend = _make_crit(_cfg(crit, lambda_, focal_gamma=2.0), L.targ_specs(lambda_, spec1, spec2), K, B)
    single1 = _make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec1)], K, B)
    single2 = _make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec2)], K, B)

    loss_b, _, _ = _run(blend)
    loss_1, _, _ = _run(single1)
    loss_2, _, _ = _run(single2)
    assert abs(loss_b.item() - ((1.0 - lambda_) * loss_1 + lambda_ * loss_2).item()) > 1e-4


TARG_DEP = [
    # (crit, target-dependent weighting)
    ("bce", {"focal_gamma": 2.0}),
    ("bce", {"dsmr": True}),
    ("bif_bce", {"focal_gamma": 2.0, "dsmr": True, "neut": True}),
    ("infonce", {"focal_gamma": 2.0}),
]


@pytest.mark.parametrize("crit,wting", TARG_DEP)
def test_loss_blend_is_the_blend_of_the_single_target_losses(crit, wting):
    # blend.type loss: each term is its single-target criterion's weighted loss, the target-dependent factors
    # (focal, DSMR, targ_mass_neut) reading the term's own target -- so the loss blend identity holds with
    # them on, where blend.type targ (one weight on the blended Y) breaks it
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    blend = _make_crit(_cfg(crit, lambda_, blend_type="loss", **wting), L.targ_specs(lambda_, spec1, spec2), K, B)
    blend_targ = _make_crit(_cfg(crit, lambda_, **wting), L.targ_specs(lambda_, spec1, spec2), K, B)
    single1 = _make_crit(_cfg(crit, 0.0, **wting), [(1.0, spec1)], K, B)
    single2 = _make_crit(_cfg(crit, 0.0, **wting), [(1.0, spec2)], K, B)

    loss_b, raw_b, grads_b = _run(blend)
    loss_1, raw_1, grads_1 = _run(single1)
    loss_2, raw_2, grads_2 = _run(single2)

    torch.testing.assert_close(loss_b, (1.0 - lambda_) * loss_1 + lambda_ * loss_2, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(raw_b, (1.0 - lambda_) * raw_1 + lambda_ * raw_2, rtol=1e-5, atol=1e-7)
    for g_b, g_1, g_2 in zip(grads_b, grads_1, grads_2):
        torch.testing.assert_close(g_b, (1.0 - lambda_) * g_1 + lambda_ * g_2, rtol=1e-5, atol=1e-7)
    assert abs(loss_b.item() - _run(blend_targ)[0].item()) > 1e-4


@pytest.mark.parametrize("crit", ["bce", "bif_bce", "infonce"])
def test_unitless_loss_blend_enters_each_term_at_unit_magnitude(crit):
    # L_k / n_k with n_k = L_k.detach() (a bifurcated L_k reads 2x its gradient scale, so n_k = L_k / 2): the
    # loss reads the weights' sum -- 1.0, bif 2.0 -- and every gradient is the blend of the single-target
    # criteria's gradients, each over its own loss magnitude; the raw loss stays the plain blend
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    blend = _make_crit(_cfg(crit, lambda_, focal_gamma=2.0, blend_type="loss", unitless=True), L.targ_specs(lambda_, spec1, spec2), K, B)
    single1 = _make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec1)], K, B)
    single2 = _make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec2)], K, B)

    loss_b, raw_b, grads_b = _run(blend)
    loss_1, raw_1, grads_1 = _run(single1)
    loss_2, raw_2, grads_2 = _run(single2)

    bif = 2.0 if crit == "bif_bce" else 1.0
    torch.testing.assert_close(loss_b, torch.tensor(bif), rtol=1e-6, atol=0)
    torch.testing.assert_close(raw_b, (1.0 - lambda_) * raw_1 + lambda_ * raw_2, rtol=1e-5, atol=1e-7)
    for g_b, g_1, g_2 in zip(grads_b, grads_1, grads_2):
        torch.testing.assert_close(g_b, (1.0 - lambda_) * g_1 / (loss_1 / bif) + lambda_ * g_2 / (loss_2 / bif), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("crit", ["bce", "bif_bce", "infonce"])
def test_unitless_target_blend_rescales_the_one_loss(crit):
    # blend.type targ has one term, the loss against the blended Y: unitless divides it by its own magnitude
    lambda_, K, B = 0.3, 6, 24
    specs = L.targ_specs(lambda_, _targ("mp", "linear"), _targ("tax", "softmax"))
    loss_u, raw_u, grads_u = _run(_make_crit(_cfg(crit, lambda_, unitless=True), specs, K, B))
    loss_p, raw_p, grads_p = _run(_make_crit(_cfg(crit, lambda_), specs, K, B))

    bif = 2.0 if crit == "bif_bce" else 1.0
    torch.testing.assert_close(loss_u, torch.tensor(bif), rtol=1e-6, atol=0)
    assert raw_u.item() == raw_p.item()
    for g_u, g_p in zip(grads_u, grads_p):
        torch.testing.assert_close(g_u, g_p / (loss_p / bif), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("crit", ["bce", "bif_bce", "infonce"])
def test_unitless_loss_blend_is_a_target_blend_under_lambda_eff(crit):
    # with target-independent weights sum_k c_k L(Y_k) = s * L(Y_eff) (the loss is affine in the target): every
    # gradient is s = sum_k c_k times the target blend's at lambda_eff = lambda L1 / (lambda L1 + (1 - lambda) L2)
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    loss_1, _, _ = _run(_make_crit(_cfg(crit, 0.0), [(1.0, spec1)], K, B))
    loss_2, _, _ = _run(_make_crit(_cfg(crit, 0.0), [(1.0, spec2)], K, B))
    _, _, grads_u = _run(_make_crit(_cfg(crit, lambda_, blend_type="loss", unitless=True), L.targ_specs(lambda_, spec1, spec2), K, B))

    lambda_eff = (lambda_ * loss_1 / (lambda_ * loss_1 + (1.0 - lambda_) * loss_2)).item()
    s = (2.0 if crit == "bif_bce" else 1.0) * ((1.0 - lambda_) / loss_1 + lambda_ / loss_2)
    _, _, grads_t = _run(_make_crit(_cfg(crit, lambda_eff), L.targ_specs(lambda_eff, spec1, spec2), K, B))
    for g_u, g_t in zip(grads_u, grads_t):
        torch.testing.assert_close(g_u, s * g_t, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("unitless", [False, True])
@pytest.mark.parametrize("crit,wting", TARG_DEP)
def test_separate_logit_scalars_score_each_term_on_its_own_logits(crit, wting, unitless):
    # loss.logits.shared false: term k of a loss blend is its single-target criterion on the model's k-th scalar
    # pair -- the towers get the blend of the two criteria's gradients, each pair only its own term's
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    scalars2 = (1.7, -0.9)
    toy = Toy(scalars2)
    blend = _make_crit(_cfg(crit, lambda_, blend_type="loss", unitless=unitless, shared=False, **wting),
                       L.targ_specs(lambda_, spec1, spec2), K, B)
    assert blend.sep_scalars
    loss_b, raw_b, (g_img, g_txt, g_scale, g_bias) = _run(blend, toy)

    toy2 = Toy()  # the second pair's values on a lone criterion's (only) pair
    with torch.no_grad():
        toy2.logit_scale.fill_(scalars2[0])
        toy2.logit_bias.fill_(scalars2[1])
    loss_1, raw_1, grads_1 = _run(_make_crit(_cfg(crit, 0.0, **wting), [(1.0, spec1)], K, B))
    loss_2, raw_2, grads_2 = _run(_make_crit(_cfg(crit, 0.0, **wting), [(1.0, spec2)], K, B), toy2)

    bif = 2.0 if crit == "bif_bce" else 1.0
    c_1 = (1.0 - lambda_) / (loss_1 / bif if unitless else 1.0)
    c_2 = lambda_ / (loss_2 / bif if unitless else 1.0)
    torch.testing.assert_close(loss_b, c_1 * loss_1 + c_2 * loss_2, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(raw_b, (1.0 - lambda_) * raw_1 + lambda_ * raw_2, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_img, c_1 * grads_1[0] + c_2 * grads_2[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_txt, c_1 * grads_1[1] + c_2 * grads_2[1], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_scale, c_1 * grads_1[2], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g_bias, c_1 * grads_1[3], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(toy.logit_scale2.grad, c_2 * grads_2[2], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(toy.logit_bias2.grad, c_2 * grads_2[3], rtol=1e-5, atol=1e-7)


def test_separate_logit_scalars_apply_to_a_live_loss_blend_only():
    # a target blend or a lone target is one loss on one set of logits: nothing to separate
    K, B = 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    assert not _make_crit(_cfg("bce", 0.3, shared=False), L.targ_specs(0.3, spec1, spec2), K, B).sep_scalars
    assert not _make_crit(_cfg("bce", 0.0, blend_type="loss", shared=False), L.targ_specs(0.0, spec1, spec2), K, B).sep_scalars
    assert not _make_crit(_cfg("bce", 0.3, blend_type="loss"), L.targ_specs(0.3, spec1, spec2), K, B).sep_scalars


def test_separate_logit_scalars_infonce_returns_the_primary_distribution():
    # two sets of logits train against no one distribution: the batch stats read the primary term's (with its
    # logits and scale), each spec's pinned softmax tsm following its own term's scale
    lambda_, K, B = 0.3, 6, 24
    pinned = {"targ": "tax", "infonce": {"tsm": {"type": "softmax", "sm_scale": "pinned"}}}
    specs = L.targ_specs(lambda_, pinned, _targ("mp", "linear"))
    cfg = _cfg("infonce", lambda_, blend_type="loss", shared=False)
    _, _, y, _ = _forward(_make_crit(cfg, specs, K, B), toy=Toy((1.7, -0.9)))
    _, _, y_1, _ = _forward(_make_crit(_cfg("infonce", 0.0), [(1.0, pinned)], K, B))
    torch.testing.assert_close(y, y_1)

    specs = L.targ_specs(lambda_, _targ("mp", "linear"), pinned)  # pinned on loss2's term: its own scale, 1.7
    crit = _make_crit(cfg, specs, K, B)
    Q = torch.rand(B, B)
    Y_2 = crit.targ_dists([Q, Q], [torch.tensor(2.3), torch.tensor(1.7)])[1]
    torch.testing.assert_close(Y_2, torch.softmax(2 * Q * torch.tensor(1.7).exp(), dim=1))


@pytest.mark.parametrize("crit", ["bce", "bif_bce", "infonce"])
def test_unitless_loss_blend_records_lambda_eff(crit):
    # the batch stats' lambda_eff (the learning-curve strip): loss2's term's share of the unitless blend
    # coefficients, lambda L1 / (lambda L1 + (1 - lambda) L2) -- bif_bce's L/2 normalizers cancel in the ratio.
    # Recorded only where unitless reweights two terms: not on static weights, a target blend or a lone target
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    specs = L.targ_specs(lambda_, spec1, spec2)
    loss_1, _, _ = _run(_make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec1)], K, B))
    loss_2, _, _ = _run(_make_crit(_cfg(crit, 0.0, focal_gamma=2.0), [(1.0, spec2)], K, B))

    blend = _make_crit(_cfg(crit, lambda_, focal_gamma=2.0, blend_type="loss", unitless=True), specs, K, B)
    assert blend.lambda_eff is None  # nothing recorded before a training batch
    _run(blend)
    torch.testing.assert_close(blend.lambda_eff, lambda_ * loss_1 / (lambda_ * loss_1 + (1.0 - lambda_) * loss_2))
    assert not blend.lambda_eff.requires_grad

    for cfg, crit_specs in (
        (_cfg(crit, lambda_, focal_gamma=2.0, blend_type="loss"), specs),  # static weights
        (_cfg(crit, lambda_, focal_gamma=2.0, unitless=True), specs),  # a target blend: one term
        (_cfg(crit, 0.0, focal_gamma=2.0, blend_type="loss", unitless=True), [(1.0, spec1)]),  # a lone target
    ):
        other = _make_crit(cfg, crit_specs, K, B)
        _run(other)
        assert other.lambda_eff is None


def test_unitless_infonce_returns_the_distribution_its_gradient_follows():
    # the returned y (what the InfoNCE batch stats read) is the terms' distributions under their normalized
    # blend coefficients c_k = w_k / L_k -- unitless moves a loss blend off the static (1 - lambda, lambda)
    lambda_, K, B = 0.3, 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    specs = L.targ_specs(lambda_, spec1, spec2)
    loss_1, _, y_1, _ = _forward(_make_crit(_cfg("infonce", 0.0), [(1.0, spec1)], K, B))
    loss_2, _, y_2, _ = _forward(_make_crit(_cfg("infonce", 0.0), [(1.0, spec2)], K, B))

    _, _, y, _ = _forward(_make_crit(_cfg("infonce", lambda_, blend_type="loss"), specs, K, B))
    torch.testing.assert_close(y, (1.0 - lambda_) * y_1 + lambda_ * y_2)

    _, _, y_u, _ = _forward(_make_crit(_cfg("infonce", lambda_, blend_type="loss", unitless=True), specs, K, B))
    c_1, c_2 = (1.0 - lambda_) / loss_1.detach(), lambda_ / loss_2.detach()
    torch.testing.assert_close(y_u, (c_1 * y_1 + c_2 * y_2) / (c_1 + c_2))
    torch.testing.assert_close(y_u.sum(dim=1), torch.ones(B))


def test_lambda_endpoints_are_the_single_targets():
    # lambda 0.0 / 1.0 drop the other spec outright (targ_specs), so the blend IS that target's criterion
    K, B = 6, 24
    spec1, spec2 = _targ("mp", "linear"), _targ("tax", "softmax")
    for lambda_, spec in ((0.0, spec1), (1.0, spec2)):
        specs = L.targ_specs(lambda_, spec1, spec2)
        assert specs == [(1.0, spec)]
        loss_b, raw_b, grads_b = _run(_make_crit(_cfg("infonce", lambda_), specs, K, B))
        loss_s, raw_s, grads_s = _run(_make_crit(_cfg("infonce", 0.0), [(1.0, spec)], K, B))
        assert loss_b.item() == loss_s.item() and raw_b.item() == raw_s.item()
        for g_b, g_s in zip(grads_b, grads_s):
            torch.testing.assert_close(g_b, g_s, rtol=0, atol=0)

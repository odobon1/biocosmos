"""
The InfoNCE reachable-optimum diagnostics (utils.loss.infonce_p_opt / infonce_s_opt /
infonce_scale_grad_sums / infonce_kl_terms / infonce_batch_stats), p* being the row softmax's
reachable optimum under bounded-cosine logits (max / min ratio at most exp(2 alpha)) and s* the
geometry realizing it. The logit-scale gradient
decomposition: per pair, dL/dalpha = (p - y) s splits into a structural part (p - p*) s -- what the
model could still remove at this alpha -- and a residual (p* - y) s that no similarity geometry can;
the residual splits again along s* into (p* - y)(s - s*) (the model's geometry standing off s*) and
(p* - y) s* (what s* itself still pushes on alpha);
each is attributed to the positive / negative target mass by the soft masks q / 1 - q, and reported
summed, summed in magnitude, and as the coherence ratio C = |sum| / sum|.|; the dlogalpha family is
the log-scale parameter's own gradient, alpha times the dalpha sums, and zero while logits.scale.clamp
holds the parameter above its cap. The KL decomposition: per anchor, D_KL(y || p) = D_KL(y || p*) +
D_KL(p* || p) + <y - p*, log(p* / p)> = E_ir + E_s + E_sr, batch-meaned. Both averaged over the I2T /
T2I anchor directions.

loss.infonce.block_residuals acts on that decomposition: utils.loss.infonce_block_resid contributes a
zero-valued loss term whose scale gradient is minus the residual, so on a hard binary target the
criterion's logit scale follows the structural part alone -- exactly so on a lone target, while on a
loss blend of two the strips are built from the BLENDED distribution and the correction from each
term's own, which p*'s nonlinearity keeps apart. The dlogalpha_correction stat is the delta the parameter's
gradient actually received (grad_after = grad_before + it).

The residual family is exp(-2 alpha)-small against quantities of order one, so the reported res / sres
/ ires and kl_ir / kl_sr come off closed forms where one exists -- infonce_hard_resid /
infonce_hard_kl_ir on a hard binary target, and an exact zero on a graded row inside the reachable band
(infonce_p_opt's feasibility branch) -- rather than off p* - y, which above alpha ~ 17 reports the
solve's own float64 noise. A graded row outside the band is the one case still on the subtraction.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.unit.test_loss_targets import import_loss_module
from utils.head import compute_sim


def _p_opt_single_row(y, alpha, n_iter=60):
    # the single-row solver the row-wise infonce_p_opt vectorizes (MP-HCon-Intuition.ipynb), verbatim
    y = y.detach().double()
    alpha = torch.as_tensor(alpha, dtype=torch.float64, device=y.device).detach()
    log_y = torch.where(y > 0, torch.log(y), -torch.inf)
    log_n = torch.log(torch.tensor(y.numel(), dtype=torch.float64, device=y.device))
    lo = -log_n - 2 * alpha
    hi = -log_n
    for _ in range(n_iter):
        eta = (lo + hi) / 2
        log_p = torch.clamp(log_y, min=eta, max=eta + 2 * alpha)
        mass = torch.exp(log_p).sum()
        if mass < 1:
            lo = eta
        else:
            hi = eta
    eta = (lo + hi) / 2
    return torch.exp(torch.clamp(log_y, min=eta, max=eta + 2 * alpha))


def _residual_terms_single_row(y, s, alpha):
    # the single-row residual split the row-wise infonce_s_opt / infonce_scale_grad_sums vectorize
    # (TSM-Lin), verbatim: s* by inverting p*'s row softmax and recentering on its midrange, then
    # (p* - y) s = (p* - y)(s - s*) + (p* - y) s*
    p_opt = _p_opt_single_row(y, alpha)
    log_p_opt = torch.log(p_opt)
    s_opt = (log_p_opt - 0.5 * (log_p_opt.max() + log_p_opt.min())) / alpha
    return (p_opt - y) * (s - s_opt), (p_opt - y) * s_opt


def _targets(B, K, seed):
    # mp memberships Q and the linear-tsm target distribution Y = Q / row mass; some rows also get a
    # graded (tax-like) membership so both exact zeros and soft targets are exercised
    g = torch.Generator().manual_seed(seed)
    class_encs = torch.randint(0, K, (B,), generator=g)
    Q = (class_encs[:, None] == class_encs[None, :]).double()
    Q[: B // 2] = torch.where(Q[: B // 2] == 1.0, Q[: B // 2], torch.rand(B // 2, B, generator=g).double() * 0.6)
    Q = 0.5 * (Q + Q.T)  # symmetric, like every targ type
    Q.fill_diagonal_(1.0)
    return Q, Q / Q.sum(dim=1, keepdim=True)


def _sims(B, seed):
    g = torch.Generator().manual_seed(seed)
    S = torch.rand(B, B, generator=g).double() * 2.0 - 1.0
    S.fill_diagonal_(0.9)
    return S


def _subtracted(Y, P_opt):
    # (R, E_ir) by the plain subtraction -- what infonce_batch_stats uses on a graded target, and the
    # reference the decomposition identities are checked against where it is still accurate
    return P_opt - Y, torch.xlogy(Y, Y).sum(dim=1) - (Y * P_opt.log()).sum(dim=1)


def _log_scale(alpha):
    # the raw log-scale parameter infonce_batch_stats takes (the model's logit_scale, detached)
    return torch.tensor(math.log(alpha), dtype=torch.float64)


def _y_stats(Y):
    # the target distribution infonce_batch_stats solves against: Y's rows renormalized in float64, the
    # criterion building them in float32 (rows summing to 1 only to ~1e-8)
    Y = Y.double()
    return Y / Y.sum(dim=1, keepdim=True)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 10.0])
def test_p_opt_matches_single_row_solver(alpha):
    L = import_loss_module()
    _, Y = _targets(12, 4, seed=0)
    P_opt = L.infonce_p_opt(Y, alpha)
    expected = torch.stack([_p_opt_single_row(y, alpha) for y in Y])
    torch.testing.assert_close(P_opt, expected, rtol=1e-9, atol=1e-12)


def test_p_opt_closed_form_mp_row():
    # y = [1/2, 1/2, 0, 0] at alpha 3: the positives share the cap R * lam, the zeros sit on the floor
    # lam, 2 R lam + 2 lam = 1 -> p* = [R, R, 1, 1] / (2 (R + 1)), R = exp(2 alpha)
    L = import_loss_module()
    R = math.exp(6.0)
    P_opt = L.infonce_p_opt(torch.tensor([[0.5, 0.5, 0.0, 0.0]]), 3.0)
    torch.testing.assert_close(P_opt, torch.tensor([[R, R, 1.0, 1.0]], dtype=torch.float64) / (2 * (R + 1)))


def test_p_opt_is_reachable_and_cross_entropy_optimal():
    # p* sums to 1, stays inside the exp(2 alpha) ratio band, and no softmax of bounded-cosine logits
    # -- random geometries, and the ideal 2q - 1 geometry -- attains a lower cross-entropy against y
    L = import_loss_module()
    B, alpha = 10, 2.0
    Q, Y = _targets(B, 3, seed=1)
    P_opt = L.infonce_p_opt(Y, alpha)
    torch.testing.assert_close(P_opt.sum(dim=1), torch.ones(B, dtype=torch.float64))
    assert torch.all(P_opt.max(dim=1).values / P_opt.min(dim=1).values <= math.exp(2 * alpha) * (1 + 1e-9))
    ce_opt = -(Y * P_opt.log()).sum(dim=1)
    g = torch.Generator().manual_seed(2)
    for S in (*(torch.rand(B, B, generator=g).double() * 2.0 - 1.0 for _ in range(20)), 2.0 * Q - 1.0):
        ce = -(Y * torch.log_softmax(alpha * S, dim=1)).sum(dim=1)
        assert torch.all(ce >= ce_opt - 1e-9)


def test_p_opt_keeps_reachable_targets():
    # a target already inside the ratio band is its own projection -- EXACTLY, by the feasibility
    # branch rather than through the solve, whose log/exp roundtrip would leave it off by ~1e-16.
    # That is nothing against y but everything against a residual that is mathematically zero
    L = import_loss_module()
    Y = torch.softmax(torch.rand(6, 8, generator=torch.Generator().manual_seed(3)).double(), dim=1)  # ratio < e
    assert torch.equal(L.infonce_p_opt(Y, 1.0), Y)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 10.0])
def test_s_opt_realizes_p_opt_inside_the_cosine_bound(alpha):
    # s* is itself a bounded-cosine geometry -- entries in [-1, 1], p*'s log range being at most
    # 2 alpha -- and its row softmax at this alpha is exactly p*, so the structural term vanishes on
    # it and the residual is all the gradient it leaves
    L = import_loss_module()
    _, Y = _targets(12, 4, seed=14)
    P_opt = L.infonce_p_opt(Y, alpha)
    S_opt = L.infonce_s_opt(P_opt, alpha)
    assert torch.all(S_opt.abs() <= 1.0 + 1e-12)
    torch.testing.assert_close(torch.softmax(alpha * S_opt, dim=1), P_opt)


def test_residual_split_matches_the_single_row_reference():
    # the residual splits row by row against the single-row construction, over the mp rows (exact
    # zeros, so p* is off the target however large alpha grows) and the graded rows alike
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=4)
    S = _sims(B, seed=5)
    P_opt = L.infonce_p_opt(Y, alpha)
    sums = L.infonce_scale_grad_sums(S, Q, Y, torch.softmax(alpha * S, dim=1), P_opt, L.infonce_s_opt(P_opt, alpha), P_opt - Y)
    sres, ires = zip(*(_residual_terms_single_row(y, s, alpha) for y, s in zip(Y, S)))
    for c, expected in ((3, sres), (4, ires)):
        assert sums[0, c, 0].item() == pytest.approx(torch.stack(expected).sum().item() / B, rel=1e-9, abs=1e-12)
    # neither part is the residual on its own: the split is doing work on this batch
    assert all(abs(sums[0, c, 0].item()) > 1e-6 for c in (3, 4))


def test_sums_decompose_and_full_sum_is_the_loss_gradient():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=4)
    S = _sims(B, seed=5)
    P = torch.softmax(alpha * S, dim=1)
    P_opt = L.infonce_p_opt(Y, alpha)
    sums = L.infonce_scale_grad_sums(S, Q, Y, P, P_opt, L.infonce_s_opt(P_opt, alpha), P_opt - Y)
    assert sums.shape == (2, 5, 3)
    # the full / all sum is d(per-anchor mean CE)/d(alpha), the direction's raw InfoNCE loss gradient
    a = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    loss = -(Y * torch.log_softmax(a * S, dim=1)).sum(dim=1).mean()
    (dloss_dalpha,) = torch.autograd.grad(loss, a)
    torch.testing.assert_close(sums[0, 0, 0], dloss_dalpha)
    # full = struct + res, res = sres + ires, all = pos + neg (exact for the sums; the magnitudes only
    # bound them)
    torch.testing.assert_close(sums[0, 0], sums[0, 1] + sums[0, 2])
    torch.testing.assert_close(sums[0, 2], sums[0, 3] + sums[0, 4])
    torch.testing.assert_close(sums[0, :, 0], sums[0, :, 1] + sums[0, :, 2])
    assert torch.all(sums[1] >= sums[0].abs() - 1e-12)
    assert torch.all(sums[1, 0] <= sums[1, 1] + sums[1, 2] + 1e-12)
    assert torch.all(sums[1, 2] <= sums[1, 3] + sums[1, 4] + 1e-12)
    assert torch.all(sums[1, :, 0] <= sums[1, :, 1] + sums[1, :, 2] + 1e-12)


def test_batch_stats_average_directions_and_take_C_on_the_averages():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float() + 0.3  # a bias is inert under the row softmax
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    aggs, comps = ("sum", "sum_abs", "C"), ("full", "struct", "res", "sres", "ires")
    assert set(stats) == {f"{prefix}_{agg}_{comp}" for prefix in ("dalpha", "dlogalpha") for agg in aggs for comp in comps} | {
        "kl", "kl_s", "kl_ir", "kl_sr", "resid_paths"} | {
        f"{prefix}alpha_req_{stat}" for prefix in ("", "log_") for stat in ("min", "mean", "max")}
    # the log-scale family: d/d(log alpha) = alpha * d/dalpha, so alpha times the sums and the same C
    for comp in comps:
        for agg in aggs[:2]:
            assert stats[f"dlogalpha_{agg}_{comp}"] == pytest.approx([alpha * v for v in stats[f"dalpha_{agg}_{comp}"]], rel=1e-12)
        assert stats[f"dlogalpha_C_{comp}"] == pytest.approx(stats[f"dalpha_C_{comp}"], rel=1e-12)
    Sf, Qf, Yf = S.float().double(), Q.float().double(), _y_stats(Y.float())
    P_opt = L.infonce_p_opt(Yf, alpha)
    S_opt = L.infonce_s_opt(P_opt, alpha)
    i2t = L.infonce_scale_grad_sums(Sf, Qf, Yf, torch.softmax(logits.double(), dim=1), P_opt, S_opt, P_opt - Yf)
    t2i = L.infonce_scale_grad_sums(Sf.T, Qf.T, Yf, torch.softmax(logits.double().T, dim=1), P_opt, S_opt, P_opt - Yf)
    expected = 0.5 * (i2t + t2i)
    for a, agg in enumerate(aggs[:2]):
        for c, comp in enumerate(comps):
            assert stats[f"dalpha_{agg}_{comp}"] == pytest.approx(expected[a, c].tolist(), rel=1e-9, abs=1e-12)
    for c, comp in enumerate(comps):
        C = stats[f"dalpha_C_{comp}"]
        assert torch.all(expected[1, c] > 0)  # nothing here is a zero-pressure term, so C is a plain ratio
        assert C == pytest.approx((expected[0, c].abs() / expected[1, c]).tolist(), rel=1e-9, abs=1e-12)
        assert all(0.0 <= v <= 1.0 for v in C)
    # a symmetric S (and Q, Y) makes the two directions coincide, so the reported values are either's
    S_sym = 0.5 * (S + S.T)
    stats_sym = L.infonce_batch_stats(S_sym, Q, Y, alpha * S_sym, _log_scale(alpha), False)
    P_opt_sym = L.infonce_p_opt(Y, alpha)
    one_dir = L.infonce_scale_grad_sums(S_sym, Q, Y, torch.softmax(alpha * S_sym, dim=1), P_opt_sym,
                                        L.infonce_s_opt(P_opt_sym, alpha), P_opt_sym - Y)
    for c, comp in enumerate(comps):
        assert stats_sym[f"dalpha_sum_{comp}"] == pytest.approx(one_dir[0, c].tolist(), rel=1e-9, abs=1e-12)
        assert stats_sym[f"dalpha_sum_abs_{comp}"] == pytest.approx(one_dir[1, c].tolist(), rel=1e-9, abs=1e-12)


def test_batch_stats_residual_is_not_the_float32_row_sum_deficit():
    # Criterion.targ_dist builds Y in float32, whose rows sum to 1 only to ~1e-8, so infonce_batch_stats
    # renormalizes them in float64 first. Without that the solve makes the deficit up by lifting the floor
    # lam and reports the lift as residual -- sum|p* - y| landing on the deficit itself, orders above the
    # true residual, which by this alpha has decayed like exp(-2 alpha) to nothing
    L = import_loss_module()
    B, alpha = 64, 60.0
    Q, Y = _targets(B, 8, seed=15)
    S = _sims(B, seed=16)
    Y32 = Y.float()
    deficit = (Y32.double().sum(dim=1) - 1.0).abs().mean().item()
    assert deficit > 1e-9  # the float32 rows really are off 1, else the check below is vacuous
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y32, (alpha * S).float(), _log_scale(alpha), False)
    # the residual sits orders under the deficit, not on it (un-renormalized it lands within a factor of
    # two of it, the magnitudes not cancelling across the two anchor directions the way the signed sums do)
    for comp in ("res", "sres", "ires"):
        assert abs(stats[f"dalpha_sum_{comp}"][0]) < deficit / 100, comp
        assert stats[f"dalpha_sum_abs_{comp}"][0] < deficit / 100, comp
    assert abs(stats["kl_ir"]) < deficit / 100
    # the structural part carries the whole gradient there, the target being reachable at this alpha
    assert stats["dalpha_sum_struct"][0] == pytest.approx(stats["dalpha_sum_full"][0], rel=1e-9)


def test_coherence_is_exact_at_vanishing_gradient_magnitudes():
    # C = |sum| / sum_abs is divided exactly, not against an additive floor in the denominator. The
    # residual terms decay like exp(-2 alpha), and on targets whose rows sum to exactly 1 (even class
    # counts, no arithmetic floor under them) they are down at ~1e-42 by alpha 50 while staying
    # perfectly coherent -- every pair pulling the same way, C = 1. A floor of 1e-30 would report
    # ~1e-12 there, i.e. near-total cancellation where there is none
    L = import_loss_module()
    B, alpha = 64, 50.0
    enc = torch.arange(B) // 4
    Q = (enc[:, None] == enc[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    assert torch.equal(Y.sum(dim=1), torch.ones(B, dtype=torch.float64))  # else A sits on a ~1e-16 floor
    g = torch.Generator().manual_seed(17)
    S = (0.6 * (2 * Q - 1) + 0.4 * (torch.rand(B, B, generator=g).double() * 2 - 1)).clamp(-1, 1)
    S = 0.5 * (S + S.T)  # a partly-trained geometry: positives above negatives, so the residual coheres
    stats = L.infonce_batch_stats(S, Q, Y, alpha * S, _log_scale(alpha), False)
    for comp in ("res", "sres", "ires"):
        G, A, C = (stats[f"dalpha_{k}_{comp}"][0] for k in ("sum", "sum_abs", "C"))
        assert 0.0 < A < 1e-30, (comp, A)  # under any floor that would have been added to it
        assert C == pytest.approx(abs(G) / A, rel=1e-12), comp
        assert C == pytest.approx(1.0, rel=1e-12), comp  # coherent: no cancellation at all


def test_batch_stats_row_wise_scale_bounds():
    # the target-implied scale bound is row-wise (softmax feasibility is): per row i, the smallest
    # alpha whose logit range 2 alpha spans log(Y_i), 0.5 * log(max_j Y_ij / min_j Y_ij), reported as
    # its min / mean / max over rows -- a global max(Y) / min(Y) would pair extremes from different
    # rows and overstate the batch's requirement -- and again as those reductions over log(alpha_req),
    # the units the logalpha panel plots in
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y_lin = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float()
    Y = torch.softmax(2.0 * Q * 3.0, dim=1)  # the softmax tsm at sm_scale 3: zero-free, every row finite
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    Yf = Y.float().double()
    alpha_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))  # == 3 * (max_j Q_ij - min_j Q_ij) per row
    assert stats["alpha_req_min"] == pytest.approx(alpha_req.min().item(), rel=1e-9)
    assert stats["alpha_req_mean"] == pytest.approx(alpha_req.mean().item(), rel=1e-9)
    assert stats["alpha_req_max"] == pytest.approx(alpha_req.max().item(), rel=1e-9)
    # the logalpha panel's trio is the alpha one in that panel's units -- the log of each, all three,
    # so a batch crosses its bound in both figures or neither. Taking the mean over log(alpha_req)
    # instead would give the log of the rows' geometric mean, which Jensen puts strictly below this
    # wherever the rows differ, leaving a band of alpha that reads as clearing the bound on one panel
    # and short of it on the other
    for stat in ("min", "mean", "max"):
        assert stats[f"log_alpha_req_{stat}"] == pytest.approx(math.log(stats[f"alpha_req_{stat}"]), rel=1e-9)
    assert alpha_req.log().mean().item() < stats["log_alpha_req_mean"]  # the Jensen gap is real here
    # under the linear tsm a row with an exact zero (the mp rows of _targets) sits at infinity, taking
    # the max and the mean with it, while the min still reads off the graded rows
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y_lin.float(), logits, _log_scale(alpha), False)
    Yf = Y_lin.float().double()
    alpha_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))
    assert math.isinf(alpha_req.max()) and torch.isfinite(alpha_req).any()
    assert stats["alpha_req_min"] == pytest.approx(alpha_req[torch.isfinite(alpha_req)].min().item(), rel=1e-9)
    assert math.isinf(stats["alpha_req_mean"]) and math.isinf(stats["alpha_req_max"])
    # and the infinity carries into the logs, leaving the min the only finite one there too
    assert math.isinf(stats["log_alpha_req_mean"]) and math.isinf(stats["log_alpha_req_max"])
    assert stats["log_alpha_req_min"] == pytest.approx(math.log(stats["alpha_req_min"]), rel=1e-9)


def _grad_log_scale(S, Y, log_alpha_raw, clamp):
    # the optimizer's view: d(loss_raw)/d(log alpha_raw) by autograd through compute_logits' scale path
    # (exp(clamp(log alpha_raw, max=ln 100)) * S under logits.scale.clamp, exp(log alpha_raw) * S
    # without), the loss the two directions' per-anchor mean CE averaged
    a = torch.tensor(log_alpha_raw, dtype=torch.float64, requires_grad=True)
    Z = (a.clamp(max=math.log(100)) if clamp else a).exp() * S
    ce = lambda z: -(Y * torch.log_softmax(z, dim=1)).sum(dim=1).mean()
    (grad,) = torch.autograd.grad(0.5 * (ce(Z) + ce(Z.T)), a)
    return grad.item()


@pytest.mark.parametrize("log_alpha_raw", [math.log(40.0), math.log(100), math.log(100) + 0.3])
def test_batch_stats_log_scale_family_is_the_parameter_gradient_through_the_clamp(log_alpha_raw):
    # dalpha* is the pressure on the effective scale the logits carry -- exp(min(log alpha_raw, ln 100))
    # under logits.scale.clamp -- so with the clamp on every non-dlogalpha stat matches the clamp-off
    # stats at that scale; dlogalpha* is the raw parameter's gradient: alpha times dalpha* while the
    # clamp is slack, and zero throughout once the parameter sits above the cap and the clamp blocks
    # it, the pressure on the effective scale notwithstanding. Exactly at the cap the family follows
    # the running torch's clamp backward either way (the gradient passes in 2.5 / 2.7, is blocked in
    # 2.14), which the loss-gradient check settles first
    L = import_loss_module()
    B = 12
    Q, Y = _targets(B, 4, seed=10)
    S = _sims(B, seed=11)
    log_alpha_eff = min(log_alpha_raw, math.log(100))
    logits = math.exp(log_alpha_eff) * S
    log_scale = torch.tensor(log_alpha_raw, dtype=torch.float64)
    on = L.infonce_batch_stats(S, Q, Y, logits, log_scale, True)
    off_eff = L.infonce_batch_stats(S, Q, Y, logits, torch.tensor(log_alpha_eff, dtype=torch.float64), False)
    for key in on:
        if not key.startswith("dlogalpha"):
            assert on[key] == pytest.approx(off_eff[key], rel=1e-12), key
    grad_ref = _grad_log_scale(S, Y, log_alpha_raw, clamp=True)
    assert on["dlogalpha_sum_full"][0] == pytest.approx(grad_ref, rel=1e-9, abs=1e-12)
    held = grad_ref == 0.0
    if log_alpha_raw > math.log(100):
        assert held
    elif log_alpha_raw < math.log(100):
        assert not held
    aggs, comps = ("sum", "sum_abs", "C"), ("full", "struct", "res", "sres", "ires")
    if held:
        assert any(v != 0.0 for v in on["dalpha_sum_full"])
        for comp in comps:
            for agg in aggs[:2]:
                assert on[f"dlogalpha_{agg}_{comp}"] == [0.0, 0.0, 0.0]
            # every pair's parameter gradient is zero, so there is no cancellation to report: the
            # coherence is undefined (NaN), not zero -- zero would read as total cancellation
            assert all(math.isnan(v) for v in on[f"dlogalpha_C_{comp}"])
    else:
        alpha = math.exp(log_alpha_raw)
        for comp in comps:
            for agg in aggs[:2]:
                assert on[f"dlogalpha_{agg}_{comp}"] == pytest.approx([alpha * v for v in on[f"dalpha_{agg}_{comp}"]], rel=1e-12)
            assert on[f"dlogalpha_C_{comp}"] == pytest.approx(on[f"dalpha_C_{comp}"], rel=1e-12)
    # with the clamp off the parameter's gradient follows the raw scale wherever it sits
    off_raw = L.infonce_batch_stats(S, Q, Y, math.exp(log_alpha_raw) * S, log_scale, False)
    assert off_raw["dlogalpha_sum_full"][0] == pytest.approx(_grad_log_scale(S, Y, log_alpha_raw, clamp=False), rel=1e-9, abs=1e-12)


def _hard_targs(B, K, targ, seed=22):
    # a hard binary membership matrix and its linear-tsm distribution: sp's diagonal, or mp's
    # same-class blocks over K classes
    class_encs = torch.randint(0, K, (B,), generator=torch.Generator().manual_seed(seed))
    Q = torch.eye(B, dtype=torch.float64) if targ == "sp" else (class_encs[:, None] == class_encs[None, :]).double()
    return class_encs, Q, Q / Q.sum(dim=1, keepdim=True)


def _res_grad_ref(L, Q, Y, S, alpha):
    # the reference: the 'res' component of the two directions' scale-gradient decomposition, in the
    # log-scale parameter's units (d/d(log alpha) = alpha * d/dalpha), negated -- what the blocking
    # term's own gradient has to be to cancel it. Built on the plain p* - y, deliberately NOT on the
    # closed form under test: an independent reference, good at the low alphas it is called at (the
    # subtraction keeps ~12 digits through alpha 5)
    P_opt = L.infonce_p_opt(Y, alpha)
    S_opt = L.infonce_s_opt(P_opt, alpha)
    Z = alpha * S
    R = P_opt - Y
    sums = 0.5 * (L.infonce_scale_grad_sums(S, Q, Y, torch.softmax(Z, dim=1), P_opt, S_opt, R)
                  + L.infonce_scale_grad_sums(S.T, Q.T, Y, torch.softmax(Z.T, dim=1), P_opt, S_opt, R))
    return alpha * sums[0], -alpha * sums[0, 2, 0].item()  # (the [comp, mask] sums, the blocking gradient)


@pytest.mark.parametrize("targ", ["mp", "sp"])
@pytest.mark.parametrize("alpha", [0.7, 3.0, 5.0])
def test_block_resid_is_the_hard_target_residual_in_closed_form(alpha, targ):
    # the term values at exactly zero (it must not move the loss reading) and its scale gradient is
    # minus the residual the reachable-optimum solve reports, at every alpha where that solve is
    # still trustworthy -- p*_+ - y_+ cancels away above it, which is why the closed form exists
    L = import_loss_module()
    B = 16
    _, Q, Y = _hard_targs(B, 4, targ)
    S = _sims(B, seed=5)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
    a = log_scale.exp().item()

    term, correction = L.infonce_block_resid(Q, S, log_scale, clamp=False)
    (grad,) = torch.autograd.grad(term, log_scale)

    assert term.item() == 0.0
    _, expected = _res_grad_ref(L, Q, Y, S, a)
    assert grad.item() == pytest.approx(expected, rel=1e-9)
    # the reported amount is that same gradient, taken without a backward
    assert correction.item() == pytest.approx(grad.item(), rel=1e-12)


def test_block_resid_keeps_the_half_the_p_opt_subtraction_loses():
    # why the closed form: at alpha 25 the positives' residual is -M r / (K (K + M r)) ~ -1.4e-22,
    # twenty orders under p*_+ itself, so p* - y reports the reachable-set solve's own float64 noise
    # (~1e-15) there while the negatives' half, whose y is exactly 0, comes through clean. The two
    # are the same order and pull opposite ways, so the subtraction is not a slightly-worse residual
    # -- here it overstates it by six orders
    L = import_loss_module()
    B, alpha = 16, 25.0
    class_encs = torch.arange(B) % 4
    Q = (class_encs[:, None] == class_encs[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    g = torch.Generator().manual_seed(24)
    S = (0.6 * (2 * Q - 1) + 0.3 * (torch.rand(B, B, generator=g).double() * 2 - 1)).clamp(-1, 1)
    S = 0.5 * (S + S.T)  # a partly-trained geometry: positives above negatives
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
    a = log_scale.exp().item()

    (grad,) = torch.autograd.grad(L.infonce_block_resid(Q, S, log_scale, clamp=False)[0], log_scale)

    P_opt = L.infonce_p_opt(Y, alpha)
    subtracted = -a * 0.5 * (((P_opt - Y) * (S + S.T)).sum(dim=1)).mean().item()
    assert grad.item() != 0.0
    assert abs(subtracted) > 1e4 * abs(grad.item())
    # the negatives' half is what survives the subtraction, and it is not the whole residual
    assert torch.all((P_opt - Y)[Q == 0] > 0.0)
    assert torch.all((P_opt - Y)[Q > 0].abs() > 1e3 * (12 * math.exp(-2 * alpha) / (4 * 4)))


def test_block_resid_reads_the_sims_not_the_logits():
    # the log-scale parameter's own gradient contracts against `sim` -- compute_logits builds the logits
    # as sim * logit_scale.exp(), so sim IS the local derivative there. Recovering sim from the logits is
    # exact in exact arithmetic (the residual rows sum to zero, so any offset cancels) but not in bf16,
    # where an offset carries them to an exponent that drops alpha * sim's own differences under an ulp.
    # The live path no longer builds such logits (InfoNCE carries no bias, and the head runs in float32
    # -- models.py), so they are built by hand here: the correction must not depend on that staying true
    L = import_loss_module()
    B, bias = 16, -12.9324  # siglip_vitb16's pretrained logit bias
    g = torch.Generator().manual_seed(31)
    embs_i = F.normalize(torch.randn(B, 16, generator=g), dim=1)
    embs_t = F.normalize(torch.randn(B, 16, generator=g), dim=1)
    class_encs = torch.arange(B) % 4
    Q = (class_encs[:, None] == class_encs[None, :]).float()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sim = compute_sim(embs_i, embs_t, "cos")
    assert sim.dtype == torch.bfloat16  # else the case under test is not being exercised

    grads = []
    for b in (0.0, bias):
        log_scale = torch.tensor(0.0, requires_grad=True)  # alpha = 1, where the residual is largest
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            Z = sim * log_scale.exp() + torch.tensor(b)
        assert Z.dtype == torch.bfloat16
        grads.append(torch.autograd.grad(L.infonce_block_resid(Q, sim, log_scale, clamp=False)[0], log_scale)[0].item())
    # the bias cannot reach the correction at all now: one gradient, whatever the logits were shifted to
    assert grads[0] == grads[1] != 0.0
    Qd = Q.double()
    _, expected = _res_grad_ref(L, Qd, Qd / Qd.sum(dim=1, keepdim=True), sim.double(), 1.0)
    assert grads[0] == pytest.approx(expected, rel=1e-6)
    # the shifted logits really did lose sim, so the reconstruction was not a harmless one
    assert (Z.double() - bias - sim.double()).abs().max() > 0.02

    # the sharp case: two near-identical embeddings, where the shift collapses the logits outright, so a
    # reconstruction would read a constant matrix and cancel exactly nothing
    embs = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.141]]), dim=1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sim2 = compute_sim(embs, embs, "cos")
        Z2 = sim2 * torch.tensor(0.0).exp() + torch.tensor(bias)
    assert Z2.double().unique().numel() == 1  # every pair rounded to one logit
    log_scale = torch.tensor(0.0, requires_grad=True)
    (grad2,) = torch.autograd.grad(L.infonce_block_resid(torch.eye(2), sim2, log_scale, clamp=False)[0], log_scale)
    assert grad2 != 0.0
    assert sim2.double().unique().numel() > 1  # the sims themselves never lost the structure


@pytest.mark.parametrize("per_cls", [1, 4, 16])  # K = 1 (sp), and mp at two class-count ratios
@pytest.mark.parametrize("alpha", [3.0, 15.0, 25.0, 55.0, 200.0])
def test_batch_stats_residual_is_exact_on_hard_targets_at_every_scale(alpha, per_cls):
    # the strips the plots read, not just the helper: on a hard binary target res / sres / ires and
    # kl_ir come off the closed forms, so they keep tracking the true exp(-2 alpha) decay instead of
    # breaking down above alpha ~ 17. The subtraction they replace fails two ways there, depending on
    # where the solve's eta lands against log(1/K): it either floors on float64 noise (6e5 x the truth
    # at alpha 25, 7e31 x at 55, sign arbitrary) or collapses the positives' half to exactly zero and
    # reports the negatives' alone -- about half, with no hint that anything is missing
    L = import_loss_module()
    B = 64
    enc = torch.arange(B) // per_cls
    Q = (enc[:, None] == enc[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    g = torch.Generator().manual_seed(17)
    S = (0.6 * (2 * Q - 1) + 0.3 * (torch.rand(B, B, generator=g).double() * 2 - 1)).clamp(-1, 1)
    S = 0.5 * (S + S.T)  # a partly-trained geometry: positives above negatives

    stats = L.infonce_batch_stats(S, Q, Y, alpha * S, _log_scale(alpha), False)

    K = Q.sum(dim=1)
    M = B - K
    r = math.exp(-2 * alpha)
    T = S + S.T
    pos = (Q * T).sum(dim=1)
    expected = 0.5 * ((r / (K + M * r)) * ((T.sum(dim=1) - pos) - (M / K) * pos)).mean().item()
    assert stats["dalpha_sum_res"][0] == pytest.approx(expected, rel=1e-9)
    assert stats["kl_ir"] == pytest.approx(torch.log1p(M * r / K).mean().item(), rel=1e-9)
    # res still splits into sres + ires, and the whole thing is still a real number at every scale
    assert stats["dalpha_sum_res"][0] == pytest.approx(stats["dalpha_sum_sres"][0] + stats["dalpha_sum_ires"][0], rel=1e-9)
    assert all(math.isfinite(stats[f"dalpha_sum_{c}"][0]) for c in ("res", "sres", "ires"))
    if alpha >= 25.0:  # the subtraction this replaced is materially wrong here, whichever way it fails
        subtracted = 0.5 * ((L.infonce_p_opt(Y, alpha) - Y) * T).sum(dim=1).mean().item()
        assert abs(subtracted - expected) > 0.1 * abs(expected)


@pytest.mark.parametrize("alpha", [1.0, 20.0])
def test_batch_stats_residual_is_exactly_zero_on_a_feasible_graded_target(alpha):
    # a graded row inside the reachable band has residual and irreducible divergence of exactly zero --
    # mathematically, not merely below resolution -- so the strips must read 0.0, not the roundtrip
    # noise the solve would leave. y = [0.6, 0.3, 0.1] is feasible from alpha_req = log(6)/2 ~ 0.896
    L = import_loss_module()
    Y = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64).repeat(3, 1)
    S = torch.tensor([[1.0, 0.2, -0.4], [0.2, 1.0, 0.1], [-0.4, 0.1, 1.0]], dtype=torch.float64)

    stats = L.infonce_batch_stats(S, Y, Y, alpha * S, _log_scale(alpha), False)

    assert alpha > 0.5 * math.log(6)  # the row really is feasible at this scale
    for comp in ("res", "sres", "ires"):
        assert stats[f"dalpha_sum_{comp}"] == [0.0, 0.0, 0.0], comp
        assert stats[f"dalpha_sum_abs_{comp}"] == [0.0, 0.0, 0.0], comp
    assert stats["kl_ir"] == 0.0 and stats["kl_sr"] == 0.0
    # the structural part is untouched: the model is still short of p*, and that is all the loss's own
    assert stats["dalpha_sum_struct"][0] == pytest.approx(stats["dalpha_sum_full"][0], rel=1e-9)
    assert stats["kl"] == pytest.approx(stats["kl_s"], rel=1e-9)


def test_batch_stats_do_not_take_a_nearly_hard_full_support_target_for_a_hard_one():
    # approximately zero is not a hard zero: softmax(19 Q) over binary memberships bottoms out at
    # 5.6e-9 > 0, so it has full support, is feasible from alpha_req = 9.5, and at alpha 10 its residual
    # and irreducible divergence are EXACTLY zero. A hard-regime test with any tolerance in it (an
    # allclose against Q / K, whose default atol of 1e-8 sits right above that floor) converts the full
    # support into boundary support and reports the hard form's exp(-2 alpha)-sized residual instead --
    # kl_ir 6.2e-9, res -2.4e-9 -- on exactly the nearly-hard targets worth comparing against hard ones
    L = import_loss_module()
    Q = torch.eye(4, dtype=torch.float64)
    Y = torch.softmax(19.0 * Q, dim=1)
    S = _sims(4, seed=33)
    assert 0.0 < Y.min().item() < 1e-8  # under allclose's default atol, where the misread happened

    stats = L.infonce_batch_stats(S, Q, Y, 10.0 * S, _log_scale(10.0), False)

    for comp in ("res", "sres", "ires"):
        assert stats[f"dalpha_sum_{comp}"] == [0.0, 0.0, 0.0], comp
        assert stats[f"dalpha_sum_abs_{comp}"] == [0.0, 0.0, 0.0], comp
    assert stats["kl_ir"] == 0.0 and stats["kl_sr"] == 0.0
    assert stats["dalpha_sum_struct"] == pytest.approx(stats["dalpha_sum_full"], rel=1e-12)  # it adds up
    # the same memberships under the linear tsm ARE hard, and read the closed form
    hard = L.infonce_batch_stats(S, Q, Q.clone(), 10.0 * S, _log_scale(10.0), False)
    assert hard["kl_ir"] == pytest.approx(math.log1p(3 * math.exp(-20.0)), rel=1e-9)


def test_batch_stats_read_the_hard_regime_off_the_target_not_the_memberships():
    # under separate logit scalars the stats take the PRIMARY term's distribution Y but the BLENDED
    # memberships Q (the pos / neg attribution masks): an sp primary hands in Y = I, exactly hard,
    # beside a fractional Q. The regime is the target's, so the closed form has to be reached from Y --
    # dispatched off Q it is skipped, and at alpha 25 the residual is back on the subtraction's noise
    # floor, 1.5e6 x the truth
    L = import_loss_module()
    B, alpha = 16, 25.0
    enc = torch.arange(B) % 4
    Q_mp = (enc[:, None] == enc[None, :]).double()
    Y = torch.eye(B, dtype=torch.float64)
    Q_blend = 0.7 * Y + 0.3 * Q_mp
    assert not bool(((Q_blend == 0) | (Q_blend == 1)).all())  # the memberships really are fractional
    g = torch.Generator().manual_seed(34)
    S = (0.6 * (2 * Q_mp - 1) + 0.3 * (torch.rand(B, B, generator=g).double() * 2 - 1)).clamp(-1, 1)
    S = 0.5 * (S + S.T)

    stats = L.infonce_batch_stats(S, Q_blend, Y, alpha * S, _log_scale(alpha), False)

    r, T = math.exp(-2 * alpha), S + S.T
    pos = (Y * T).sum(dim=1)  # sp: K = 1, M = B - 1
    expected = 0.5 * ((r / (1 + (B - 1) * r)) * ((T.sum(dim=1) - pos) - (B - 1) * pos)).mean().item()
    assert stats["dalpha_sum_res"][0] == pytest.approx(expected, rel=1e-9)
    assert stats["kl_ir"] == pytest.approx(math.log1p((B - 1) * r), rel=1e-9)


def test_batch_stats_report_which_calculation_computed_each_row():
    # provenance, not a magnitude guess: resid_paths is the batch's row fractions [hard closed form,
    # feasible exact zero, subtracted / numerically unvalidated]. The first two are exact at any alpha;
    # only the third is ever in doubt -- and it can fail looking like a noise floor OR like an exact
    # zero, so the value alone can never say which it is
    L = import_loss_module()
    S3 = torch.tensor([[1.0, 0.2, -0.4], [0.2, 1.0, 0.1], [-0.4, 0.1, 1.0]], dtype=torch.float64)
    stat = lambda Q, Y, alpha: L.infonce_batch_stats(S3, Q, Y, alpha * S3, _log_scale(alpha), False)

    eye = torch.eye(3, dtype=torch.float64)
    assert stat(eye, eye, 25.0)["resid_paths"] == [1.0, 0.0, 0.0]  # sp: every row hard
    soft = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64).repeat(3, 1)
    assert stat(soft, soft, 25.0)["resid_paths"] == [0.0, 1.0, 0.0]  # full support inside the band
    uniform = torch.full((3, 3), 1.0 / 3.0, dtype=torch.float64)
    assert stat(uniform, uniform, 25.0)["resid_paths"] == [0.0, 1.0, 0.0]  # hard AND feasible: counted once

    # a graded membership matrix with zeros: rows 0-1 are [2/3, 1/3, 0] (infeasible, not hard), row 2 hard
    Q = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    Y = Q / Q.sum(dim=1, keepdim=True)
    lo, hi = stat(Q, Y, 25.0), stat(Q, Y, 55.0)
    assert lo["resid_paths"] == hi["resid_paths"] == pytest.approx([1 / 3, 0.0, 2 / 3])
    # the analytic kl_ir here is (10 / 9) exp(-2 alpha): 2.1e-22 and 1.9e-48. What the two subtracted rows
    # contribute instead is whatever the solve's lam happens to round to -- orders too large at one scale,
    # an exact zero (reading as feasibility) at another -- which is why they are flagged, not trusted
    # at alpha 25 they floor on the solve's noise, seven orders over the truth ...
    assert lo["kl_ir"] > 1e6 * (10.0 / 9.0) * math.exp(-50.0)
    # ... and at alpha 55 they contribute EXACTLY zero: the batch reads the hard row's own exact share,
    # log1p(2 r) / 3, and nothing else -- 40% short, with nothing in the number to show for it
    assert hi["kl_ir"] == pytest.approx(math.log1p(2 * math.exp(-110.0)) / 3, rel=1e-9)
    assert hi["kl_ir"] < 0.7 * (10.0 / 9.0) * math.exp(-110.0)


def test_hard_rows_read_a_closed_form_p_opt_in_the_structural_terms():
    # the structural terms difference p against p*, so at p = p* they read whatever p* is off by: from
    # the solve that is ~|log lam| ulps (kl_s ~ -7e-15 at alpha 25, negative for a divergence), from
    # the closed form p* = y + (p* - y) it is rounding alone
    L = import_loss_module()
    B, alpha = 16, 25.0
    enc = torch.arange(B) % 4
    Q = (enc[:, None] == enc[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    S_ideal = 2.0 * Q - 1.0  # s* itself: the model sits exactly on the reachable optimum

    stats = L.infonce_batch_stats(S_ideal, Q, Y, alpha * S_ideal, _log_scale(alpha), False)

    assert abs(stats["kl_s"]) < 1e-15
    assert abs(stats["dalpha_sum_struct"][0]) < 1e-15
    solver_kl_s = L.infonce_kl_terms(Y, torch.log_softmax(alpha * S_ideal, dim=1), L.infonce_p_opt(Y, alpha),
                                     *_subtracted(Y, L.infonce_p_opt(Y, alpha)))[1].item()
    assert abs(stats["kl_s"]) < abs(solver_kl_s)  # and it is an improvement on the solve's, not a wash


def test_feasibility_branch_is_continuous_at_the_boundary():
    # the branch needs no tolerance: the true residual goes to zero with the excess log range, so a row
    # just inside reads an exact zero and one just outside reads a correspondingly tiny value -- there
    # is no step at alpha_req to straddle
    L = import_loss_module()
    Y = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    alpha_req = 0.5 * math.log(6)
    assert torch.equal(L.infonce_p_opt(Y, alpha_req), Y)  # exactly at the bound: feasible
    for frac in (0.999, 0.99, 0.9):
        resid = (L.infonce_p_opt(Y, alpha_req * frac) - Y).abs().sum().item()
        assert 0.0 < resid < 1.0 - frac  # nonzero, and small in proportion to how far outside it sits


def _hard_targ_crit(L, targ, B, K, block_residuals, specs=None, lambda_=0.0, blend_type="targ", unitless=False):
    """An InfoNCE criterion over hard binary targets under the linear tsm, unweighted -- the
    configuration loss.infonce.block_residuals is gated to."""
    crit = L.InfoNCECriterion.__new__(L.InfoNCECriterion)  # bypass build_wting (no dataset needed)
    crit.cfg = {
        "crit": "infonce", "sim": "cos", "blend": {"lambda": lambda_, "type": blend_type}, "unitless": unitless,
        "infonce": {"block_residuals": block_residuals},
        "wting": {"cls_imb": {"type": None, "norm": False}},
        "logits": {"shared": True, "scale": {"clamp": False}},
    }
    linear = lambda t: {"targ": t, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}
    crit.targ_specs = specs or [(1.0, linear(targ))]
    crit.device = torch.device("cpu")
    crit.batch_size = B
    crit.counts = torch.ones(K, dtype=torch.float64)
    crit.wt_mean = 1.0
    return crit


def _run_crit(L, targ, class_encs, S, log_alpha, block):
    """(loss, raw loss, d/d(log alpha), dL/dS) of the criterion on the batch's sims."""
    log_scale = torch.tensor(log_alpha, dtype=torch.float64, requires_grad=True)
    S = S.clone().requires_grad_(True)
    crit = _hard_targ_crit(L, targ, S.size(0), int(class_encs.max()) + 1, block)
    loss, loss_raw, _, _ = crit(S * log_scale.exp(), class_encs, None, True, log_scale, S)
    loss.backward()
    return loss.detach(), loss_raw.detach(), log_scale.grad, S.grad


@pytest.mark.parametrize("targ", ["mp", "sp"])
def test_block_residuals_leaves_the_criterion_only_its_structural_scale_gradient(targ):
    # end to end: the criterion's loss reading and dL/dS are untouched (the term values at zero and
    # reads the logits detached), while the log-scale parameter's gradient drops the residual and
    # lands on the structural part alone
    L = import_loss_module()
    B, log_alpha = 16, math.log(3.0)
    class_encs, Q, Y = _hard_targs(B, 4, targ)
    S = _sims(B, seed=23)

    off = _run_crit(L, targ, class_encs, S, log_alpha, block=False)
    on = _run_crit(L, targ, class_encs, S, log_alpha, block=True)

    for i in (0, 1, 3):  # loss, loss_raw, dL/dS
        torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
    sums, blocking = _res_grad_ref(L, Q, Y, S, math.exp(log_alpha))
    full, struct, res = (sums[c, 0].item() for c in (0, 1, 2))
    assert abs(res) > 1e-3 * abs(full)  # the residual is a real share of the gradient at this alpha
    assert off[2].item() == pytest.approx(full, rel=1e-5)  # the loss scores against a float32 Y
    assert on[2].item() == pytest.approx(struct, rel=1e-5)
    assert (on[2] - off[2]).item() == pytest.approx(blocking, rel=1e-9)  # exactly the term's gradient


@pytest.mark.parametrize("unitless", [False, True])
def test_blocked_stat_is_the_correction_a_loss_blend_actually_applies(unitless):
    # on a loss blend the correction removes each TERM's own residual under that term's coefficient,
    # while the dalpha / dlogalpha family is built from the blended distribution -- and p* is not
    # linear in the target, so the struct entry is not the parameter's post-block gradient (nor, under
    # loss.unitless, is the full entry its pre-block one, the coefficients no longer summing to 1
    # against a renormalized Y). Criterion.dlogalpha_correction is the exact reading either way
    L = import_loss_module()
    B, K, lambda_, log_alpha = 24, 4, 0.3, 0.0
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=29)

    def run(block):
        log_scale = torch.tensor(log_alpha, dtype=torch.float64, requires_grad=True)
        linear = lambda t: {"targ": t, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}
        specs = L.targ_specs(lambda_, linear("sp"), linear("mp"))
        crit = _hard_targ_crit(L, None, B, K, block, specs=specs, lambda_=lambda_,
                               blend_type="loss", unitless=unitless)
        Sg = S.clone().requires_grad_(True)
        loss, _, _, _ = crit(Sg * log_scale.exp(), class_encs, None, True, log_scale, Sg)
        loss.backward()
        return log_scale.grad.item(), crit.dlogalpha_correction

    grad_off, correction_off = run(False)
    grad_on, correction_on = run(True)

    assert correction_off is None  # nothing recorded when the toggle is off
    # the stat IS the delta the intervention made to the parameter's gradient, to float64, and signed
    # as one: grad_after = grad_before + correction (it is MINUS the blocked residual)
    assert grad_on == pytest.approx(grad_off + correction_on.item(), rel=1e-9)
    assert correction_on.item() != 0.0


@pytest.mark.parametrize("case", ["live", "frozen", "clamp_held", "clamp_slack"])
def test_correction_is_the_one_the_parameter_actually_receives(case):
    # applied, not counterfactual: the logged correction carries the same gradient eligibility as the
    # parameter path it describes -- the clamp-then-exp Jacobian d(alpha)/d(log alpha_raw) (zero once
    # logits.scale.clamp holds the raw parameter above ln 100) and nothing at all under a frozen scale.
    # Logged analytically as -alpha * resid it would report +0.34 of "correction" on a frozen scale whose
    # gradient is exactly zero
    L = import_loss_module()
    B = 16
    _, Q, _ = _hard_targs(B, 4, "mp")
    S = _sims(B, seed=41)
    log_alpha, clamp, live = {
        "live": (0.0, False, True), "frozen": (0.0, False, False),
        "clamp_held": (math.log(100) + 0.5, True, True), "clamp_slack": (math.log(3.0), True, True),
    }[case]
    log_scale = torch.tensor(log_alpha, dtype=torch.float64, requires_grad=live)

    term, correction = L.infonce_block_resid(Q, S, log_scale, clamp)

    actual = torch.autograd.grad(term, log_scale)[0].item() if live else 0.0
    assert correction.item() == pytest.approx(actual, rel=1e-12, abs=0.0)
    assert (correction.item() == 0.0) == (case in ("frozen", "clamp_held"))


def test_correction_state_is_per_forward():
    # the criterion must not carry a training batch's correction into a later eval call: the stat is
    # cleared on every forward and only a training one sets it
    L = import_loss_module()
    B, K = 16, 4
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=42)
    crit = _hard_targ_crit(L, "mp", B, K, True)
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    crit(S * log_scale.exp(), class_encs, None, True, log_scale, S)
    assert crit.dlogalpha_correction is not None
    crit(S * log_scale.exp(), class_encs, None, False, log_scale, S)
    assert crit.dlogalpha_correction is None


def _kl_rows(A, B):
    # row-wise D_KL(a || b), the single-row get_kl of MP-HCon-Intuition.ipynb
    return torch.xlogy(A, A / B).sum(dim=1)


def test_kl_terms_match_their_definitions_and_decompose():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=8)
    S = _sims(B, seed=9)
    log_P = torch.log_softmax(alpha * S, dim=1)
    P = log_P.exp()
    P_opt = L.infonce_p_opt(Y, alpha)
    kl, E_s, E_ir, E_sr = L.infonce_kl_terms(Y, log_P, P_opt, *_subtracted(Y, P_opt))
    # each term by its definition, batch-meaned
    torch.testing.assert_close(kl, _kl_rows(Y, P).mean())
    torch.testing.assert_close(E_s, _kl_rows(P_opt, P).mean())
    torch.testing.assert_close(E_ir, _kl_rows(Y, P_opt).mean())
    torch.testing.assert_close(E_sr, ((Y - P_opt) * (P_opt / P).log()).sum(dim=1).mean())
    # the decomposition is exact, every part nonnegative, and D_KL(y || p) is this direction's raw
    # InfoNCE loss (per-anchor CE, anchor-averaged) less the targets' mean entropy
    torch.testing.assert_close(kl, E_ir + E_s + E_sr)
    assert kl > 0 and E_s > 0 and E_ir > 0 and E_sr > 0
    ce = -(Y * log_P).sum(dim=1).mean()
    H = -torch.xlogy(Y, Y).sum(dim=1).mean()
    torch.testing.assert_close(kl, ce - H)


def test_kl_terms_notebook_example():
    # MP-HCon-Intuition.ipynb: y = [1/2, 1/2, 0, 0], s = [0.9, 1, -0.9, -1] at alpha 3 reads
    # (kl, E_s, E_ir, E_sr) = (0.0145, 0.0113, 0.0025, 0.0007) to the notebook's printed precision
    L = import_loss_module()
    Y = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
    log_P = torch.log_softmax(3.0 * torch.tensor([[0.9, 1.0, -0.9, -1.0]], dtype=torch.float64), dim=1)
    P_opt = L.infonce_p_opt(Y, 3.0)
    terms = L.infonce_kl_terms(Y, log_P, P_opt, *_subtracted(Y, P_opt))
    assert terms.tolist() == pytest.approx([0.0145, 0.0113, 0.0025, 0.0007], abs=6e-5)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 10.0])
def test_kl_cross_term_is_nonnegative_on_reachable_p(alpha):
    # E_sr >= 0 for every p a bounded-cosine softmax can realize (random geometries, the ideal
    # 2q - 1 one, and sign patterns at the band's edges), and the structural and cross terms vanish
    # at p = p*, leaving D_KL(y || p*) = E_ir
    L = import_loss_module()
    B = 10
    Q, Y = _targets(B, 3, seed=10)
    P_opt = L.infonce_p_opt(Y, alpha)
    g = torch.Generator().manual_seed(11)
    geometries = (
        *(torch.rand(B, B, generator=g).double() * 2.0 - 1.0 for _ in range(20)),
        2.0 * Q - 1.0,
        torch.sign(torch.rand(B, B, generator=g).double() - 0.5),
    )
    for S in geometries:
        assert L.infonce_kl_terms(Y, torch.log_softmax(alpha * S, dim=1), P_opt, *_subtracted(Y, P_opt))[3] >= -1e-12
    kl, E_s, E_ir, E_sr = L.infonce_kl_terms(Y, P_opt.log(), P_opt, *_subtracted(Y, P_opt))
    torch.testing.assert_close(torch.stack([E_s, E_sr]), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(kl, E_ir)


def test_batch_stats_kl_keys_average_directions():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=12)
    S = _sims(B, seed=13)
    logits = (alpha * S).float() - 0.7  # a bias is inert under the row softmax
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    Yf, Z = _y_stats(Y.float()), logits.double()
    P_opt = L.infonce_p_opt(Yf, alpha)
    i2t = L.infonce_kl_terms(Yf, torch.log_softmax(Z, dim=1), P_opt, *_subtracted(Yf, P_opt))
    t2i = L.infonce_kl_terms(Yf, torch.log_softmax(Z.T, dim=1), P_opt, *_subtracted(Yf, P_opt))
    expected = 0.5 * (i2t + t2i)
    for k, key in enumerate(("kl", "kl_s", "kl_ir", "kl_sr")):
        assert stats[key] == pytest.approx(expected[k].item(), rel=1e-9, abs=1e-12)
    assert stats["kl"] == pytest.approx(stats["kl_s"] + stats["kl_ir"] + stats["kl_sr"], rel=1e-9, abs=1e-12)
    # the irreducible part depends on the targets and alpha alone, not on the anchor direction
    assert i2t[2] == t2i[2]
    # kl is the criterion's loss_raw (the two directions' per-anchor CE means, averaged) less the
    # targets' mean entropy
    ce = 0.5 * sum(-(Yf * torch.log_softmax(M, dim=1)).sum(dim=1).mean() for M in (Z, Z.T))
    H = -torch.xlogy(Yf, Yf).sum(dim=1).mean()
    assert stats["kl"] == pytest.approx((ce - H).item(), rel=1e-9, abs=1e-12)

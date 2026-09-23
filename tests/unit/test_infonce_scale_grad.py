"""
The InfoNCE reachable-optimum diagnostics (utils.loss.infonce_p_opt / infonce_s_opt /
infonce_scale_grad_sums / infonce_kl_terms / infonce_batch_stats), p* being the row softmax's
reachable optimum under bounded-cosine logits (max / min ratio at most exp(2 alpha)) and s* the
geometry realizing it. The logit-scale gradient
decomposition: per pair, dL/dscale = (p - y) s splits into a structural part (p - p*) s -- what the
model could still remove at this alpha -- and a residual (p* - y) s that no similarity geometry can;
the residual splits again along s* into (p* - y)(s - s*) (the model's geometry standing off s*) and
(p* - y) s* (what s* itself still pushes on alpha);
each is attributed to the positive / negative target mass by the soft masks q / 1 - q, and reported
summed, summed in magnitude (A) with the coherence ratio |sum| / A, and summed in magnitude per
anchor row (C = sum_i |sum_j .|, over each direction's own anchors) with |sum| / C; the dlogscale family is
the log-scale parameter's own gradient, alpha times the dscale sums, and zero while logits.scale.clamp
holds the parameter above its cap. The KL decomposition: per anchor, D_KL(y || p) = D_KL(y || p*) +
D_KL(p* || p) + <y - p*, log(p* / p)> = E_ir + E_u + E_ur, batch-meaned. Both averaged over the I2T /
T2I anchor directions.

loss.infonce.block_residuals acts on that decomposition: utils.loss.infonce_block_resid contributes a
zero-valued loss term whose scale gradient is minus the residual, so on a hard binary target the
criterion's logit scale follows the structural part alone -- exactly so on a lone target, while on a
loss blend of two the strips are built from the BLENDED distribution and the correction from each
term's own, which p*'s nonlinearity keeps apart. The dlogscale_correction stat is the delta the parameter's
gradient actually received (grad_after = grad_before + it).

The residual family is exp(-2 alpha)-small against quantities of order one, so the reported res / ures
/ ires and kl_ir / kl_ur come off closed forms where one exists -- infonce_hard_resid /
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
    ures, ires = zip(*(_residual_terms_single_row(y, s, alpha) for y, s in zip(Y, S)))
    for c, expected in ((3, ures), (4, ires)):
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
    assert sums.shape == (3, 5, 3)
    # the full / all sum is d(per-anchor mean CE)/d(alpha), the direction's raw InfoNCE loss gradient
    a = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    loss = -(Y * torch.log_softmax(a * S, dim=1)).sum(dim=1).mean()
    (dloss_dscale,) = torch.autograd.grad(loss, a)
    torch.testing.assert_close(sums[0, 0, 0], dloss_dscale)
    # full = struct + res, res = ures + ires, all = pos + neg (exact for the sums; the magnitudes only
    # bound them)
    torch.testing.assert_close(sums[0, 0], sums[0, 1] + sums[0, 2])
    torch.testing.assert_close(sums[0, 2], sums[0, 3] + sums[0, 4])
    torch.testing.assert_close(sums[0, :, 0], sums[0, :, 1] + sums[0, :, 2])
    assert torch.all(sums[1] >= sums[0].abs() - 1e-12)
    assert torch.all(sums[1, 0] <= sums[1, 1] + sums[1, 2] + 1e-12)
    assert torch.all(sums[1, 2] <= sums[1, 3] + sums[1, 4] + 1e-12)
    assert torch.all(sums[1, :, 0] <= sums[1, :, 1] + sums[1, :, 2] + 1e-12)
    # the per-anchor magnitude: sum_i |sum_j .|, anchor-averaged like the rest, between the other two
    terms = ((P - Y) * S, (P - P_opt) * S, (P_opt - Y) * S)
    for c, G in enumerate(terms):
        for m, M in enumerate((torch.ones_like(Q), Q, 1.0 - Q)):
            torch.testing.assert_close(sums[2, c, m], (G * M).sum(dim=1).abs().sum() / B)
    assert torch.all(sums[0].abs() <= sums[2] + 1e-12) and torch.all(sums[2] <= sums[1] + 1e-12)
    # strictly under the pair magnitudes here, each anchor's pairs cancelling inside its row -- while on
    # this random geometry every anchor pulls alpha the same way, so nothing cancels between them
    assert sums[2, 0, 0] < 0.99 * sums[1, 0, 0]
    torch.testing.assert_close(sums[0, 0, 0].abs(), sums[2, 0, 0])


def test_batch_stats_average_directions_and_take_ratios_on_the_averages():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float() + 0.3  # a bias is inert under the row softmax
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    mags, ratios = {"sum_abs": 1, "row_abs": 2}, {"ratio": 1, "ratio_row": 2}  # agg -> infonce_scale_grad_sums' index
    aggs, comps = ("sum", *mags, *ratios), ("full", "struct", "res", "ures", "ires")
    assert set(stats) == {f"{prefix}_{agg}_{comp}" for prefix in ("dscale", "dlogscale") for agg in aggs for comp in comps} | {
        "kl", "kl_u", "kl_ir", "kl_ur", "resid_paths"} | {
        f"{prefix}scale_req_{stat}" for prefix in ("", "log_") for stat in ("min", "mean", "max")} | {
        f"sim_grad_entropy_{level}_{comp}" for level in ("pair", "anchor", "active") for comp in ("full", "struct", "res")}
    # the log-scale family: d/d(log alpha) = alpha * d/dscale, so alpha times the sums and the same ratios
    for comp in comps:
        for agg in ("sum", *mags):
            assert stats[f"dlogscale_{agg}_{comp}"] == pytest.approx([alpha * v for v in stats[f"dscale_{agg}_{comp}"]], rel=1e-12)
        for agg in ratios:
            assert stats[f"dlogscale_{agg}_{comp}"] == pytest.approx(stats[f"dscale_{agg}_{comp}"], rel=1e-12)
    Sf, Qf, Yf = S.float().double(), Q.float().double(), _y_stats(Y.float())
    P_opt = L.infonce_p_opt(Yf, alpha)
    S_opt = L.infonce_s_opt(P_opt, alpha)
    i2t = L.infonce_scale_grad_sums(Sf, Qf, Yf, torch.softmax(logits.double(), dim=1), P_opt, S_opt, P_opt - Yf)
    t2i = L.infonce_scale_grad_sums(Sf.T, Qf.T, Yf, torch.softmax(logits.double().T, dim=1), P_opt, S_opt, P_opt - Yf)
    expected = 0.5 * (i2t + t2i)
    for agg, a in {"sum": 0, **mags}.items():
        for c, comp in enumerate(comps):
            assert stats[f"dscale_{agg}_{comp}"] == pytest.approx(expected[a, c].tolist(), rel=1e-9, abs=1e-12)
    for c, comp in enumerate(comps):
        for agg, a in ratios.items():
            ratio = stats[f"dscale_{agg}_{comp}"]
            assert torch.all(expected[a, c] > 0)  # nothing here is a zero-pressure term, so the ratio is a plain one
            assert ratio == pytest.approx((expected[0, c].abs() / expected[a, c]).tolist(), rel=1e-9, abs=1e-12)
            assert all(0.0 <= v <= 1.0 for v in ratio)
        # |sum| <= C <= A across the reported (direction-averaged) series, so the per-pair ratio never
        # reads above the per-anchor one
        assert all(v <= w + 1e-12 for v, w in zip(stats[f"dscale_ratio_{comp}"], stats[f"dscale_ratio_row_{comp}"]))
    # a symmetric S (and Q, Y) makes the two directions coincide, so the reported values are either's
    S_sym = 0.5 * (S + S.T)
    stats_sym = L.infonce_batch_stats(S_sym, Q, Y, alpha * S_sym, _log_scale(alpha), False)
    P_opt_sym = L.infonce_p_opt(Y, alpha)
    one_dir = L.infonce_scale_grad_sums(S_sym, Q, Y, torch.softmax(alpha * S_sym, dim=1), P_opt_sym,
                                        L.infonce_s_opt(P_opt_sym, alpha), P_opt_sym - Y)
    for c, comp in enumerate(comps):
        assert stats_sym[f"dscale_sum_{comp}"] == pytest.approx(one_dir[0, c].tolist(), rel=1e-9, abs=1e-12)
        assert stats_sym[f"dscale_sum_abs_{comp}"] == pytest.approx(one_dir[1, c].tolist(), rel=1e-9, abs=1e-12)
        assert stats_sym[f"dscale_row_abs_{comp}"] == pytest.approx(one_dir[2, c].tolist(), rel=1e-9, abs=1e-12)


def test_row_abs_is_over_the_anchors_of_both_directions():
    # C = sum_i |sum_j .| takes the magnitude per ANCHOR, and the bidirectional loss has 2B of them: it is
    # the mean of B image-anchor CE terms (the rows of the I2T terms) and B text-anchor ones (the rows of
    # the T2I terms, i.e. the COLUMNS of the I2T layout). Each has its own scale gradient, so the reference
    # is autograd's, term by term: row_abs_full is the mean |d(CE_k)/d(alpha)| over the 2B terms, the way
    # sum_full is their signed mean (the loss gradient), and ratio_row the ratio of the two -- the cancellation
    # BETWEEN anchors, each anchor's own pairs having cancelled inside its row sum
    L = import_loss_module()
    B, alpha = 12, 2.0
    enc = torch.arange(B) % 4
    Q = (enc[:, None] == enc[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    S = _sims(B, seed=19)  # asymmetric, so the two directions differ
    # half the image anchors fitted (positives above negatives: at this alpha they want it larger), the rest
    # random (they want it smaller), so the anchors pull alpha both ways and the magnitudes have work to do
    S[: B // 2] = (0.7 * (2 * Q - 1) + 0.3 * S)[: B // 2]
    stats = L.infonce_batch_stats(S, Q, Y, alpha * S, _log_scale(alpha), False)

    ce = lambda z: -(Y * torch.log_softmax(z, dim=1)).sum(dim=1)  # one CE term per anchor row
    per_anchor = torch.autograd.functional.jacobian(
        lambda a: torch.cat([ce(a * S), ce(a * S.T)]), torch.tensor(alpha, dtype=torch.float64))
    assert per_anchor.shape == (2 * B,)
    assert min((per_anchor > 0).sum(), (per_anchor < 0).sum()) >= B // 2  # they really do pull both ways
    assert stats["dscale_sum_full"][0] == pytest.approx(per_anchor.mean().item(), rel=1e-9)
    assert stats["dscale_row_abs_full"][0] == pytest.approx(per_anchor.abs().mean().item(), rel=1e-9)
    assert stats["dscale_ratio_row_full"][0] == pytest.approx((per_anchor.mean().abs() / per_anchor.abs().mean()).item(), rel=1e-9)
    # |sum| < C < A, strictly: cancellation between the anchors, and inside their rows before that
    assert stats["dscale_ratio_full"][0] < 0.5 * stats["dscale_ratio_row_full"][0] < 0.5

    # the pos / neg attributions likewise, each direction masking its own rows (Q.T under the text anchors)
    G_i2t = (torch.softmax(alpha * S, dim=1) - Y) * S  # [image anchor, text]
    G_t2i = (torch.softmax(alpha * S.T, dim=1) - Y) * S.T  # [text anchor, image]
    for m, (M_i2t, M_t2i) in enumerate(((1.0, 1.0), (Q, Q.T), (1.0 - Q, 1.0 - Q.T))):
        expected = 0.5 * ((G_i2t * M_i2t).sum(dim=1).abs().sum() + (G_t2i * M_t2i).sum(dim=1).abs().sum()) / B
        assert stats["dscale_row_abs_full"][m] == pytest.approx(expected.item(), rel=1e-9)

    # what it is NOT: the row sums of a per-pair blend of the directions, which adds image i's I2T row to
    # the T2I terms of the texts it is a CANDIDATE of -- nor the T2I terms summed down the anchors
    blended = 0.5 * (G_i2t + G_t2i.T)
    wrong_axis = 0.5 * (G_i2t.sum(dim=1).abs().sum() + G_t2i.sum(dim=0).abs().sum()) / B
    for wrong in (blended.sum(dim=1).abs().sum() / B, wrong_axis):
        assert abs(stats["dscale_row_abs_full"][0] - wrong.item()) > 1e-3 * stats["dscale_row_abs_full"][0]


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
    for comp in ("res", "ures", "ires"):
        assert abs(stats[f"dscale_sum_{comp}"][0]) < deficit / 100, comp
        assert stats[f"dscale_sum_abs_{comp}"][0] < deficit / 100, comp
    assert abs(stats["kl_ir"]) < deficit / 100
    # the structural part carries the whole gradient there, the target being reachable at this alpha
    assert stats["dscale_sum_struct"][0] == pytest.approx(stats["dscale_sum_full"][0], rel=1e-9)


def test_coherence_is_exact_at_vanishing_gradient_magnitudes():
    # ratio = |sum| / sum_abs is divided exactly, not against an additive floor in the denominator. The
    # residual terms decay like exp(-2 alpha), and on targets whose rows sum to exactly 1 (even class
    # counts, no arithmetic floor under them) they are down at ~1e-42 by alpha 50 while staying
    # perfectly coherent -- every pair pulling the same way, ratio = 1. A floor of 1e-30 would report
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
    for comp in ("res", "ures", "ires"):
        G, A, ratio = (stats[f"dscale_{k}_{comp}"][0] for k in ("sum", "sum_abs", "ratio"))
        assert 0.0 < A < 1e-30, (comp, A)  # under any floor that would have been added to it
        assert ratio == pytest.approx(abs(G) / A, rel=1e-12), comp
        assert ratio == pytest.approx(1.0, rel=1e-12), comp  # coherent: no cancellation at all
        # the per-anchor magnitude and its ratio alike: no cancellation inside the rows leaves C on A
        assert stats[f"dscale_row_abs_{comp}"][0] == pytest.approx(A, rel=1e-12), comp
        assert stats[f"dscale_ratio_row_{comp}"][0] == pytest.approx(1.0, rel=1e-12), comp


def test_batch_stats_row_wise_scale_bounds():
    # the target-implied scale bound is row-wise (softmax feasibility is): per row i, the smallest
    # alpha whose logit range 2 alpha spans log(Y_i), 0.5 * log(max_j Y_ij / min_j Y_ij), reported as
    # its min / mean / max over rows -- a global max(Y) / min(Y) would pair extremes from different
    # rows and overstate the batch's requirement -- and again as those reductions over log(scale_req),
    # the units the logscale panel plots in
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y_lin = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float()
    Y = torch.softmax(2.0 * Q * 3.0, dim=1)  # the softmax tsm at sm_scale 3: zero-free, every row finite
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    Yf = Y.float().double()
    scale_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))  # == 3 * (max_j Q_ij - min_j Q_ij) per row
    assert stats["scale_req_min"] == pytest.approx(scale_req.min().item(), rel=1e-9)
    assert stats["scale_req_mean"] == pytest.approx(scale_req.mean().item(), rel=1e-9)
    assert stats["scale_req_max"] == pytest.approx(scale_req.max().item(), rel=1e-9)
    # the logscale panel's trio is the scale one in that panel's units -- the log of each, all three,
    # so a batch crosses its bound in both figures or neither. Taking the mean over log(scale_req)
    # instead would give the log of the rows' geometric mean, which Jensen puts strictly below this
    # wherever the rows differ, leaving a band of alpha that reads as clearing the bound on one panel
    # and short of it on the other
    for stat in ("min", "mean", "max"):
        assert stats[f"log_scale_req_{stat}"] == pytest.approx(math.log(stats[f"scale_req_{stat}"]), rel=1e-9)
    assert scale_req.log().mean().item() < stats["log_scale_req_mean"]  # the Jensen gap is real here
    # under the linear tsm a row with an exact zero (the mp rows of _targets) sits at infinity, taking
    # the max and the mean with it, while the min still reads off the graded rows
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y_lin.float(), logits, _log_scale(alpha), False)
    Yf = Y_lin.float().double()
    scale_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))
    assert math.isinf(scale_req.max()) and torch.isfinite(scale_req).any()
    assert stats["scale_req_min"] == pytest.approx(scale_req[torch.isfinite(scale_req)].min().item(), rel=1e-9)
    assert math.isinf(stats["scale_req_mean"]) and math.isinf(stats["scale_req_max"])
    # and the infinity carries into the logs, leaving the min the only finite one there too
    assert math.isinf(stats["log_scale_req_mean"]) and math.isinf(stats["log_scale_req_max"])
    assert stats["log_scale_req_min"] == pytest.approx(math.log(stats["scale_req_min"]), rel=1e-9)


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
    # dscale* is the pressure on the effective scale the logits carry -- exp(min(log alpha_raw, ln 100))
    # under logits.scale.clamp -- so with the clamp on every non-dlogscale stat matches the clamp-off
    # stats at that scale; dlogscale* is the raw parameter's gradient: alpha times dscale* while the
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
        if not key.startswith("dlogscale"):
            assert on[key] == pytest.approx(off_eff[key], rel=1e-12, nan_ok=True), key  # NaN: an entropy with no mass behind it, on either path
    grad_ref = _grad_log_scale(S, Y, log_alpha_raw, clamp=True)
    assert on["dlogscale_sum_full"][0] == pytest.approx(grad_ref, rel=1e-9, abs=1e-12)
    held = grad_ref == 0.0
    if log_alpha_raw > math.log(100):
        assert held
    elif log_alpha_raw < math.log(100):
        assert not held
    sums, ratios, comps = ("sum", "sum_abs", "row_abs"), ("ratio", "ratio_row"), ("full", "struct", "res", "ures", "ires")
    if held:
        assert any(v != 0.0 for v in on["dscale_sum_full"])
        for comp in comps:
            for agg in sums:
                assert on[f"dlogscale_{agg}_{comp}"] == [0.0, 0.0, 0.0]
            # every pair's parameter gradient is zero, so there is no cancellation to report: the
            # coherence is undefined (NaN), not zero -- zero would read as total cancellation
            for agg in ratios:
                assert all(math.isnan(v) for v in on[f"dlogscale_{agg}_{comp}"])
    else:
        alpha = math.exp(log_alpha_raw)
        for comp in comps:
            for agg in sums:
                assert on[f"dlogscale_{agg}_{comp}"] == pytest.approx([alpha * v for v in on[f"dscale_{agg}_{comp}"]], rel=1e-12)
            for agg in ratios:
                assert on[f"dlogscale_{agg}_{comp}"] == pytest.approx(on[f"dscale_{agg}_{comp}"], rel=1e-12)
    # with the clamp off the parameter's gradient follows the raw scale wherever it sits
    off_raw = L.infonce_batch_stats(S, Q, Y, math.exp(log_alpha_raw) * S, log_scale, False)
    assert off_raw["dlogscale_sum_full"][0] == pytest.approx(_grad_log_scale(S, Y, log_alpha_raw, clamp=False), rel=1e-9, abs=1e-12)


def _hard_targs(B, K, targ, seed=22):
    # a hard binary membership matrix and its linear-tsm distribution: sp's diagonal, or mp's
    # same-class blocks over K classes
    class_encs = torch.randint(0, K, (B,), generator=torch.Generator().manual_seed(seed))
    Q = torch.eye(B, dtype=torch.float64) if targ == "sp" else (class_encs[:, None] == class_encs[None, :]).double()
    return class_encs, Q, Q / Q.sum(dim=1, keepdim=True)


def _res_grad_ref(L, Q, Y, S, alpha):
    # the reference: the 'res' component of the two directions' scale-gradient decomposition, in the
    # log-scale parameter's units (d/d(log alpha) = alpha * d/dscale), negated -- what the blocking
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

    term, correction, _, _ = L.infonce_block_resid(Y, S, log_scale, clamp=False, full=False, coeff=1.0)
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

    (grad,) = torch.autograd.grad(L.infonce_block_resid(Y, S, log_scale, clamp=False, full=False, coeff=1.0)[0], log_scale)

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
    Y = Q.double() / Q.double().sum(dim=1, keepdim=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sim = compute_sim(embs_i, embs_t, "cos")
    assert sim.dtype == torch.bfloat16  # else the case under test is not being exercised

    grads = []
    for b in (0.0, bias):
        log_scale = torch.tensor(0.0, requires_grad=True)  # alpha = 1, where the residual is largest
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            Z = sim * log_scale.exp() + torch.tensor(b)
        assert Z.dtype == torch.bfloat16
        grads.append(torch.autograd.grad(L.infonce_block_resid(Y, sim, log_scale, clamp=False, full=False, coeff=1.0)[0], log_scale)[0].item())
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
    (grad2,) = torch.autograd.grad(L.infonce_block_resid(torch.eye(2), sim2, log_scale, clamp=False, full=False, coeff=1.0)[0], log_scale)
    assert grad2 != 0.0
    assert sim2.double().unique().numel() > 1  # the sims themselves never lost the structure


@pytest.mark.parametrize("per_cls", [1, 4, 16])  # K = 1 (sp), and mp at two class-count ratios
@pytest.mark.parametrize("alpha", [3.0, 15.0, 25.0, 55.0, 200.0])
def test_batch_stats_residual_is_exact_on_hard_targets_at_every_scale(alpha, per_cls):
    # the strips the plots read, not just the helper: on a hard binary target res / ures / ires and
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
    assert stats["dscale_sum_res"][0] == pytest.approx(expected, rel=1e-9)
    assert stats["kl_ir"] == pytest.approx(torch.log1p(M * r / K).mean().item(), rel=1e-9)
    # res still splits into ures + ires, and the whole thing is still a real number at every scale
    assert stats["dscale_sum_res"][0] == pytest.approx(stats["dscale_sum_ures"][0] + stats["dscale_sum_ires"][0], rel=1e-9)
    assert all(math.isfinite(stats[f"dscale_sum_{c}"][0]) for c in ("res", "ures", "ires"))
    if alpha >= 25.0:  # the subtraction this replaced is materially wrong here, whichever way it fails
        subtracted = 0.5 * ((L.infonce_p_opt(Y, alpha) - Y) * T).sum(dim=1).mean().item()
        assert abs(subtracted - expected) > 0.1 * abs(expected)


@pytest.mark.parametrize("alpha", [1.0, 20.0])
def test_batch_stats_residual_is_exactly_zero_on_a_feasible_graded_target(alpha):
    # a graded row inside the reachable band has residual and irreducible divergence of exactly zero --
    # mathematically, not merely below resolution -- so the strips must read 0.0, not the roundtrip
    # noise the solve would leave. y = [0.6, 0.3, 0.1] is feasible from scale_req = log(6)/2 ~ 0.896
    L = import_loss_module()
    Y = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64).repeat(3, 1)
    S = torch.tensor([[1.0, 0.2, -0.4], [0.2, 1.0, 0.1], [-0.4, 0.1, 1.0]], dtype=torch.float64)

    stats = L.infonce_batch_stats(S, Y, Y, alpha * S, _log_scale(alpha), False)

    assert alpha > 0.5 * math.log(6)  # the row really is feasible at this scale
    for comp in ("res", "ures", "ires"):
        assert stats[f"dscale_sum_{comp}"] == [0.0, 0.0, 0.0], comp
        assert stats[f"dscale_sum_abs_{comp}"] == [0.0, 0.0, 0.0], comp
    assert stats["kl_ir"] == 0.0 and stats["kl_ur"] == 0.0
    # the structural part is untouched: the model is still short of p*, and that is all the loss's own
    assert stats["dscale_sum_struct"][0] == pytest.approx(stats["dscale_sum_full"][0], rel=1e-9)
    assert stats["kl"] == pytest.approx(stats["kl_u"], rel=1e-9)


def test_batch_stats_do_not_take_a_nearly_hard_full_support_target_for_a_hard_one():
    # approximately zero is not a hard zero: softmax(19 Q) over binary memberships bottoms out at
    # 5.6e-9 > 0, so it has full support, is feasible from scale_req = 9.5, and at alpha 10 its residual
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

    for comp in ("res", "ures", "ires"):
        assert stats[f"dscale_sum_{comp}"] == [0.0, 0.0, 0.0], comp
        assert stats[f"dscale_sum_abs_{comp}"] == [0.0, 0.0, 0.0], comp
    assert stats["kl_ir"] == 0.0 and stats["kl_ur"] == 0.0
    assert stats["dscale_sum_struct"] == pytest.approx(stats["dscale_sum_full"], rel=1e-12)  # it adds up
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
    assert stats["dscale_sum_res"][0] == pytest.approx(expected, rel=1e-9)
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
    # the solve that is ~|log lam| ulps (kl_u ~ -7e-15 at alpha 25, negative for a divergence), from
    # the closed form p* = y + (p* - y) it is rounding alone
    L = import_loss_module()
    B, alpha = 16, 25.0
    enc = torch.arange(B) % 4
    Q = (enc[:, None] == enc[None, :]).double()
    Y = Q / Q.sum(dim=1, keepdim=True)
    S_ideal = 2.0 * Q - 1.0  # s* itself: the model sits exactly on the reachable optimum

    stats = L.infonce_batch_stats(S_ideal, Q, Y, alpha * S_ideal, _log_scale(alpha), False)

    assert abs(stats["kl_u"]) < 1e-15
    assert abs(stats["dscale_sum_struct"][0]) < 1e-15
    solver_kl_u = L.infonce_kl_terms(Y, torch.log_softmax(alpha * S_ideal, dim=1), L.infonce_p_opt(Y, alpha),
                                     *_subtracted(Y, L.infonce_p_opt(Y, alpha)))[1].item()
    assert abs(stats["kl_u"]) < abs(solver_kl_u)  # and it is an improvement on the solve's, not a wash


def test_feasibility_branch_is_continuous_at_the_boundary():
    # the branch needs no tolerance: the true residual goes to zero with the excess log range, so a row
    # just inside reads an exact zero and one just outside reads a correspondingly tiny value -- there
    # is no step at scale_req to straddle
    L = import_loss_module()
    Y = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    scale_req = 0.5 * math.log(6)
    assert torch.equal(L.infonce_p_opt(Y, scale_req), Y)  # exactly at the bound: feasible
    for frac in (0.999, 0.99, 0.9):
        resid = (L.infonce_p_opt(Y, scale_req * frac) - Y).abs().sum().item()
        assert 0.0 < resid < 1.0 - frac  # nonzero, and small in proportion to how far outside it sits


def _hard_targ_crit(L, targ, B, K, block_residuals, specs=None, lambda_=0.0, blend_type="targ", unitless=False):
    """An InfoNCE criterion over hard binary targets under the linear tsm (or over `specs`), unweighted
    -- as loss.infonce.block_residuals requires."""
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

    off = _run_crit(L, targ, class_encs, S, log_alpha, block=None)
    on = _run_crit(L, targ, class_encs, S, log_alpha, block="alpha")

    for i in (0, 1, 3):  # loss, loss_raw, dL/dS
        torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
    sums, blocking = _res_grad_ref(L, Q, Y, S, math.exp(log_alpha))
    full, struct, res = (sums[c, 0].item() for c in (0, 1, 2))
    assert abs(res) > 1e-3 * abs(full)  # the residual is a real share of the gradient at this alpha
    assert off[2].item() == pytest.approx(full, rel=1e-5)  # the loss scores against a float32 Y
    assert on[2].item() == pytest.approx(struct, rel=1e-5)
    assert (on[2] - off[2]).item() == pytest.approx(blocking, rel=1e-9)  # exactly the term's gradient


@pytest.mark.parametrize("targ", ["mp", "sp"])
def test_block_residuals_full_leaves_the_whole_model_only_its_structural_gradient(targ):
    # under full the same residual leaves dL/dS too: the loss reading is untouched and the scale's
    # gradient is what alpha gives it (the scale's part is one term either way), while the gradient
    # into the sims drops to what the loss would send the towers against the reachable optimum p* in
    # place of y -- per pair and anchor direction alpha (p - p*) / B, p* off the solve here rather than
    # the closed form the term is built on
    L = import_loss_module()
    B, log_alpha = 16, math.log(3.0)
    class_encs, Q, Y = _hard_targs(B, 4, targ)
    S = _sims(B, seed=23)
    alpha = math.exp(log_alpha)

    off = _run_crit(L, targ, class_encs, S, log_alpha, block=None)
    scale = _run_crit(L, targ, class_encs, S, log_alpha, block="alpha")
    full = _run_crit(L, targ, class_encs, S, log_alpha, block="full")

    for i in (0, 1):  # loss, loss_raw
        torch.testing.assert_close(full[i], off[i], rtol=0.0, atol=0.0)
    assert full[2].item() == pytest.approx(scale[2].item(), rel=1e-12)
    P_opt = L.infonce_p_opt(Y, alpha)
    Z = alpha * S
    P, Pt = torch.softmax(Z, dim=1), torch.softmax(Z.T, dim=1)
    dS_full = 0.5 * alpha * ((P - Y) + (Pt - Y).T) / B
    dS_struct = 0.5 * alpha * ((P - P_opt) + (Pt - P_opt).T) / B
    assert (dS_full - dS_struct).abs().max() > 1e-3 * dS_full.abs().max()  # a real share at this alpha
    torch.testing.assert_close(off[3], dS_full, rtol=1e-5, atol=1e-8)  # the loss scores against a float32 Y
    torch.testing.assert_close(scale[3], dS_full, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(full[3], dS_struct, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("unitless", [False, True])
def test_blocked_stat_is_the_correction_a_loss_blend_actually_applies(unitless):
    # on a loss blend the correction removes each TERM's own residual under that term's coefficient,
    # while the dscale / dlogscale family is built from the blended distribution -- and p* is not
    # linear in the target, so the struct entry is not the parameter's post-block gradient (nor, under
    # loss.unitless, is the full entry its pre-block one, the coefficients no longer summing to 1
    # against a renormalized Y). Criterion.dlogscale_correction is the exact reading either way
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
        return log_scale.grad.item(), crit.dlogscale_correction

    grad_off, correction_off = run(None)
    grad_on, correction_on = run("alpha")

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
    _, _, Y = _hard_targs(B, 4, "mp")
    S = _sims(B, seed=41)
    log_alpha, clamp, live = {
        "live": (0.0, False, True), "frozen": (0.0, False, False),
        "clamp_held": (math.log(100) + 0.5, True, True), "clamp_slack": (math.log(3.0), True, True),
    }[case]
    log_scale = torch.tensor(log_alpha, dtype=torch.float64, requires_grad=live)

    term, correction, _, _ = L.infonce_block_resid(Y, S, log_scale, clamp, full=False, coeff=1.0)

    actual = torch.autograd.grad(term, log_scale)[0].item() if live else 0.0
    assert correction.item() == pytest.approx(actual, rel=1e-12, abs=0.0)
    assert (correction.item() == 0.0) == (case in ("frozen", "clamp_held"))


def _graded_targs(B):
    # a two-level tree's kernel over 4 classes in 2 families: 1 within a class, 0.5 within a family, 0 across
    enc = torch.arange(B) % 4
    same_cls, same_fam = enc[:, None] == enc[None, :], (enc // 2)[:, None] == (enc // 2)[None, :]
    return torch.where(same_cls, 1.0, torch.where(same_fam, 0.5, 0.0)).double()


@pytest.mark.parametrize("alpha", [1.0, 3.0])
def test_block_resid_blocks_a_graded_target_outside_the_band_through_the_active_sets(alpha):
    # a graded target outside the reachable band (its zeros put it there under the linear tsm, as tax / phylo's
    # do) has no two-level closed form: its residual comes from the projection's active sets, and the term's
    # scale gradient is minus that residual's contraction against sim + sim.T, the residual as the independent
    # reference gives it
    L = import_loss_module()
    B = 16
    Q = _graded_targs(B)
    Y = Q / Q.sum(dim=1, keepdim=True)
    S = _sims(B, seed=5)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)

    term, correction, coverage, _ = L.infonce_block_resid(Y, S, log_scale, clamp=False, full=False, coeff=1.0)
    (grad,) = torch.autograd.grad(term, log_scale)

    R = _reference_resid(Y, alpha)
    assert (R != 0.0).any(dim=1).all()  # every row outside the band
    assert term.item() == 0.0
    assert grad.item() == pytest.approx(-alpha * 0.5 * ((R * (S + S.T)).sum(dim=1)).mean().item(), rel=1e-9)
    assert correction.item() == pytest.approx(grad.item(), rel=1e-12)
    assert coverage.tolist() == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("alpha", [1.0, 20.0])
def test_block_resid_is_exactly_zero_on_a_feasible_target(alpha):
    # a target inside the band (here softmax(alpha Q) with Q in [0, 1]: pinned1's, log range alpha) is its
    # own reachable optimum, so there is nothing to block -- the term's gradient and the logged correction
    # are exactly zero, not the solve's roundtrip noise
    L = import_loss_module()
    B = 16
    Y = torch.softmax(alpha * _graded_targs(B), dim=1)
    S = _sims(B, seed=5)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)

    term, correction, coverage, _ = L.infonce_block_resid(Y, S, log_scale, clamp=False, full=False, coeff=1.0)
    (grad,) = torch.autograd.grad(term, log_scale)

    assert torch.equal(L.infonce_p_opt(Y, alpha), Y)  # feasible, exactly
    assert term.item() == 0.0 and grad.item() == 0.0 and correction.item() == 0.0
    assert coverage.tolist() == [0.0, 1.0, 0.0]


def _two_level_reference_resid(K, M, alpha, s):
    # the residual of the INTENDED two-level row -- K entries at 1 / (K + M t), M at t / (K + M t), t = exp(-s),
    # built as mpmath numbers -- from the independent reference (tests.unit.infonce_reference), as float64
    import mpmath as mp
    from tests.unit import infonce_reference as ref
    with mp.workdps(60):
        t = mp.exp(-mp.mpf(s))
        y = [1 / (K + M * t)] * K + [t / (K + M * t)] * M
    out = ref.reference(y, alpha=alpha)
    with mp.workdps(out["dps"]):
        return [float(r) for r in out["R"]], out["feasible"]


@pytest.mark.parametrize("alpha", [0.5, 3.0, 15.0, 25.0, 55.0, 200.0])
@pytest.mark.parametrize("sm_scale", ["pinned3", 4.0])
def test_two_level_closed_form_matches_the_reference_on_the_intended_row(alpha, sm_scale):
    # binary memberships under a softmax tsm past the band: the two-level closed form (R_top = -M (r - t) / D,
    # R_bottom = K (r - t) / D) against the independent mpmath reference on the intended row, at scales from
    # where the residual is order one to where it is ~1e-174 and a subtraction would read only noise; a fixed
    # sm_scale c maps to s = 2 c, so at c = 4 the row is feasible once alpha reaches 4 and there is no residual
    L = import_loss_module()
    B, K = 12, 3
    M = B - K
    Q = torch.zeros(B, B, dtype=torch.float64)
    Q[:, :K] = 1.0  # every row: the same K positives
    tsm = {"type": "softmax", "sm_scale": sm_scale}
    s = L.tsm_target_scale(tsm, alpha)
    Y = torch.softmax(s * Q, dim=1)  # what _tsm stores (its softmax(2 * Q * c) at a fixed c: the same product)

    R, applied, feasible, skipped = L.infonce_train_resid(Y, alpha, s, binary=True, inside=s <= 2 * alpha)
    R_ref, ref_feasible = _two_level_reference_resid(K, M, alpha, s)

    if s <= 2 * alpha:
        assert ref_feasible and feasible.all() and not applied.any() and R.abs().max().item() == 0.0
        return
    assert not ref_feasible and applied.all()
    assert R[0].tolist() == pytest.approx(R_ref, rel=1e-12, abs=0.0)  # every row is that row
    # the stored row differs from the intended one by its rounding, which matters only where the residual is
    # smaller than that rounding: the reference on the stored floats agrees where it is not
    if alpha <= 5.0:
        import mpmath as mp
        from tests.unit import infonce_reference as ref
        out = ref.reference(Y[0].tolist(), alpha=alpha)
        with mp.workdps(out["dps"]):
            assert R[0].tolist() == pytest.approx([float(r) for r in out["R"]], rel=1e-9, abs=0.0)


def test_two_level_closed_form_is_the_hard_one_at_t_zero_and_agrees_with_the_active_sets():
    # t = 0 is the hard binary row and reads as before (infonce_hard_resid), spec or no spec: a stored zero is a
    # zero. A soft bottom level takes the spec's t only under a binary-membership spec; under a graded spec, or
    # none, the same row goes through the active sets instead, and comes out the same to rounding
    L = import_loss_module()
    B, alpha = 16, 3.0
    _, Q, Y_hard = _hard_targs(B, 4, "mp")
    for s, binary in ((math.inf, True), (None, False)):
        R, applied, _, _ = L.infonce_train_resid(Y_hard, alpha, s, binary, inside=False)
        torch.testing.assert_close(R, L.infonce_hard_resid(Q, alpha), rtol=0.0, atol=0.0)
        assert applied.all()

    Y_soft = torch.softmax(3 * alpha * Q, dim=1)  # pinned3 on the same memberships
    R_spec, applied, _, _ = L.infonce_train_resid(Y_soft, alpha, 3 * alpha, True, inside=False)
    assert applied.all()
    for s, binary in ((3 * alpha, False), (None, False)):
        R, applied, _, skipped = L.infonce_train_resid(Y_soft, alpha, s, binary, inside=False)
        assert applied.all() and not skipped.any()
        torch.testing.assert_close(R, R_spec, rtol=1e-12, atol=0.0)


def _phylo_like_targs(B):
    # a graded kernel in [0.4, 1] with a unit diagonal, every row's spread under two thirds (inside pinned3's band),
    # and, on rows 0 to B // 2 - 1, three root-split zeros apiece, which put those rows past any band
    g = torch.Generator().manual_seed(7)
    Q = 0.4 + 0.6 * torch.rand(B, B, generator=g, dtype=torch.float64)
    Q.fill_diagonal_(1.0)
    for i in range(B // 2):
        Q[i, [(i + 1) % B, (i + 5) % B, (i + 9) % B]] = 0.0
    return Q


def test_active_set_residual_matches_the_reference_on_the_corpus():
    # the active-set closed form against the independent reference on every corpus row stated at an alpha
    # (tests.unit.infonce_reference.corpus: hard rows to alpha 300, graded rows with zeros, tax-like repeated
    # levels, feasible rows, a near-degenerate cap, r underflowing): an infeasible row resolves and agrees to
    # 1e-9 of its largest residual; a feasible row is not the active sets' to resolve
    import mpmath as mp
    from tests.unit import infonce_reference as ref
    L = import_loss_module()
    for name, y, scale in ref.corpus():
        if "alpha" not in scale:
            continue  # the exact-ratio cases are not reachable through an alpha
        alpha = scale["alpha"]
        R, resolved = L.infonce_active_set_resid(torch.tensor([y], dtype=torch.float64), alpha)
        out = ref.reference(y, alpha=alpha)
        with mp.workdps(out["dps"]):
            R_ref = torch.tensor([[float(v) for v in out["R"]]], dtype=torch.float64)
        if alpha > L.ALPHA_SUPPORTED_MAX:
            # past the supported scale (here past the underflow point as well: r is 0.0), skipped as unsupported,
            # the reference's residual being nonzero all the same
            with mp.workdps(out["dps"]):
                assert not out["feasible"] and any(v != 0 for v in out["R"]), name
            assert not resolved.item() and R.abs().max().item() == 0.0, name
            continue
        if out["feasible"]:
            assert not resolved.item() and R.abs().max().item() == 0.0, name
            continue
        assert resolved.item(), name
        assert (R - R_ref).abs().max().item() <= 1e-9 * R_ref.abs().max().item(), name


def test_active_set_rejects_a_consistent_assignment_of_rounded_sets(monkeypatch):
    # two entries one float apart at the top of a row at alpha 55: the true cap lies strictly between them, which
    # float64 cannot hold, so it rounds onto the lower one -- (y0 + y1) / 2 is y1 exactly -- and an assignment
    # taking that entry into the cap is self-consistent under the optimality conditions. Its residual then reads
    # the gap between the floats, -+2.8e-17, where the reference gives -6e-49 on the top entry and 0 on the next:
    # the lower entry's reads positive, a capped entry lifted, and the row's |R|_1 sits thirty orders over the
    # bound U. Both checks reject it, the row is left unresolved, and through the training path it contributes
    # no correction and reads as skipped. The start is forced onto the wrong assignment so the case does not
    # depend on which side of the entries the bisection's rounding falls
    import mpmath as mp
    from tests.unit import infonce_reference as ref
    L = import_loss_module()
    alpha = 55.0
    y = [0.3698399825779713, 0.36983998257797124, 0.26032003484405747, 0.0]
    assert math.fsum(y) == 1.0 and y[0] - y[1] == math.ulp(y[0]) and (y[0] + y[1]) / 2 == y[1]
    Y = torch.tensor([y], dtype=torch.float64)
    out = ref.reference(y, alpha=alpha)
    with mp.workdps(out["dps"]):
        R_ref = torch.tensor([[float(v) for v in out["R"]]], dtype=torch.float64)
    assert not out["feasible"] and R_ref[0, 1].item() == 0.0 and -1e-47 < R_ref[0, 0].item() < 0.0
    r = math.exp(-2 * alpha)
    U = 2 * 3 * r / (1 + 3 * r)
    assert R_ref.abs().sum().item() <= U

    R, resolved = L.infonce_active_set_resid(Y, alpha)  # as the bisection starts it: rejected, or right
    assert not resolved.item() or (R - R_ref).abs().max().item() <= 1e-9 * R_ref.abs().max().item()

    cap_thr = y[1] * (1 - 1e-12)  # a start with both top entries capped, the fixed point the rounded cap keeps
    monkeypatch.setattr(L, "_infonce_floor_eta", lambda log_Y, a, n_iter=60: torch.full(
        (log_Y.size(0), 1), math.log(cap_thr) - 2 * alpha, dtype=torch.float64))
    R, resolved = L.infonce_active_set_resid(Y, alpha)
    assert not resolved.item() and R.abs().max().item() == 0.0

    Y4 = torch.cat([Y, torch.full((3, 4), 0.25, dtype=torch.float64)])  # beside three feasible uniform rows
    S = torch.eye(4, dtype=torch.float64)  # orthonormal embeddings
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
    term, correction, coverage, _ = L.infonce_block_resid(Y4, S, log_scale, clamp=False, full=False, coeff=1.0)
    (grad,) = torch.autograd.grad(term, log_scale)
    assert grad.item() == 0.0 and correction.item() == 0.0
    assert coverage.tolist() == pytest.approx([0.0, 0.75, 0.25])


def test_residuals_past_the_supported_scale_read_skipped():
    # past ALPHA_SUPPORTED_MAX (the reference module's conservative cutoff, not the underflow point: at 345 r is
    # 2e-300, a normal float; the two constants are held equal) corrections are skipped as unsupported: rows past
    # the band read skipped there, the hard and two-level closed forms as much as the active sets, and the
    # correction is zero -- while just under the bound every path still resolves (exp(log(340)) lands a hair over
    # 340, so the bound itself is not probed through the parameter path)
    from tests.unit import infonce_reference as ref
    L = import_loss_module()
    assert L.ALPHA_SUPPORTED_MAX == ref.ALPHA_SUPPORTED_MAX == 340.0
    B = 8
    _, Q, Y_hard = _hard_targs(B, 2, "mp")
    Q_graded = _phylo_like_targs(B)
    S = _sims(B, seed=5)
    for alpha, expect_applied in ((330.0, True), (345.0, False), (400.0, False)):
        for Y, s, binary in ((Y_hard, math.inf, True),
                             (torch.softmax(3 * alpha * Q, dim=1), 3 * alpha, True),
                             (Q_graded / Q_graded.sum(dim=1, keepdim=True), math.inf, False)):
            R, applied, feasible, skipped = L.infonce_train_resid(Y, alpha, s, binary, inside=False)
            past = ~feasible
            assert past.any()
            assert (applied[past].all() if expect_applied else skipped[past].all()) and (R[skipped] == 0.0).all()
        log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
        _, correction, coverage, _ = L.infonce_block_resid(Y_hard, S, log_scale, clamp=False, full=False, coeff=1.0)
        assert coverage.tolist() == ([1.0, 0.0, 0.0] if expect_applied else [0.0, 0.0, 1.0])
        if not expect_applied:
            assert correction.item() == 0.0


@pytest.mark.parametrize("alpha", [0.5, 3.0, 15.0, 55.0])
@pytest.mark.parametrize("mapping", ["linear", "pinned3"])
def test_active_set_residual_matches_the_reference_on_graded_rows(alpha, mapping):
    # phylo-like graded rows -- kernel values in [0, 1], a unit diagonal, root-split zeros on half the rows --
    # through the linear tsm and pinned3: every row past the band resolves and agrees with the reference to 1e-9
    # of its largest residual, down to residuals of ~1e-48 at alpha 55, where the subtraction reads only rounding
    import mpmath as mp
    from tests.unit import infonce_reference as ref
    L = import_loss_module()
    B = 12
    Q = _phylo_like_targs(B)
    Y = Q / Q.sum(dim=1, keepdim=True) if mapping == "linear" else torch.softmax(3 * alpha * Q, dim=1)
    R, resolved = L.infonce_active_set_resid(Y, alpha)
    n_past = 0
    for i in range(B):
        out = ref.reference(Y[i].tolist(), alpha=alpha)
        if out["feasible"]:
            continue
        with mp.workdps(out["dps"]):
            R_ref = torch.tensor([float(v) for v in out["R"]], dtype=torch.float64)
        n_past += 1
        assert resolved[i].item(), i
        assert (R[i] - R_ref).abs().max().item() <= 1e-9 * R_ref.abs().max().item(), i
    assert n_past >= B // 2


@pytest.mark.parametrize("block", ["alpha", "full"])
@pytest.mark.parametrize("targ", ["mp", "phylo"])
@pytest.mark.parametrize("sm_scale", ["pinned", "pinned1", "pinned3"])
def test_block_residuals_on_the_pinned_family(block, targ, sm_scale):
    # end to end, the three pinned mappings alike, on memberships and on a graded phylo-like kernel: pinned
    # (s = 2 alpha, the boundary itself) and pinned1 (s = alpha) keep every row inside the band by construction, so
    # blocking is inert -- the correction reads exactly zero, loss and gradients are those of blocking off, and the
    # coverage says feasible, not skipped; pinned3 (s = 3 alpha) puts memberships' rows past it (the two-level closed
    # form) and the kernel's rows with a spread over two thirds (the active sets), and blocks the residual the
    # reference gives on the scale's gradient and, under full, on dL/dS, with the kernel's other rows feasible
    L = import_loss_module()
    B, K, alpha = 16, 4, 3.0
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=23)
    Q = _phylo_like_targs(B) if targ == "phylo" else (class_encs[:, None] == class_encs[None, :]).double()
    spec = {"targ": targ, "infonce": {"tsm": {"type": "softmax", "sm_scale": sm_scale}}}

    def run(block):
        log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
        crit = _hard_targ_crit(L, None, B, K, block, specs=[(1.0, spec)])
        crit._targets = lambda *args: [Q]
        Sg = S.clone().requires_grad_(True)
        loss, loss_raw, _, Y = crit(Sg * log_scale.exp(), class_encs, None, True, log_scale, Sg)
        loss.backward()
        return loss.detach(), loss_raw.detach(), log_scale.grad, Sg.grad, Y, crit

    off = run(None)
    on = run(block)

    for i in (0, 1):  # loss, loss_raw
        torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
    stats = on[5].block_stats
    if sm_scale != "pinned3":
        for i in (2, 3):
            torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
        assert stats["dlogscale_correction"] == 0.0 and stats["block_coverage"] == [[0.0, 1.0, 0.0]]
        return
    R = _reference_resid(off[4], alpha)
    past = (R != 0.0).any(dim=1)
    assert past.all() if targ == "mp" else (past.any() and not past.all())
    frac = past.double().mean().item()
    assert stats["block_coverage"] == [pytest.approx([frac, 1.0 - frac, 0.0])]
    blocking = -alpha * 0.5 * ((R * (S + S.T)).sum(dim=1)).mean().item()
    assert abs(blocking) > 1e-3 * abs(off[2].item())  # a real share of the gradient at this alpha
    assert (on[2] - off[2]).item() == pytest.approx(blocking, rel=1e-9)
    assert stats["dlogscale_correction"] == pytest.approx(blocking, rel=1e-9)
    expected_S = -0.5 * alpha * (R + R.T) / B if block == "full" else torch.zeros_like(S)
    torch.testing.assert_close(on[3] - off[3], expected_S, rtol=1e-9, atol=1e-14)


def test_block_bound_is_recorded_and_holds():
    # the recorded magnitude bounds: U = 2 (B - 1) r / (1 + (B - 1) r) on any infeasible row's |p* - y|_1, which
    # the one-hot row attains, and |coeff| alpha U on the correction -- both above what was applied, on hard
    # rows and on pinned3's soft ones alike
    L = import_loss_module()
    B, alpha, coeff = 16, 3.0, 2.5
    _, _, Y_hard = _hard_targs(B, 4, "sp")
    S = _sims(B, seed=5)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
    r = math.exp(-2 * alpha)
    U = 2 * (B - 1) * r / (1 + (B - 1) * r)

    _, correction, _, bound = L.infonce_block_resid(Y_hard, S, log_scale, clamp=False, full=False, coeff=coeff)
    assert bound.tolist() == pytest.approx([U, coeff * alpha * U], rel=1e-12)
    R, *_ = L.infonce_train_resid(Y_hard, alpha, math.inf, binary=True, inside=False)
    assert R.abs().sum(dim=1).max().item() == pytest.approx(U, rel=1e-12)  # sp: the one-hot row attains it
    assert abs(correction.item()) <= bound[1].item()

    Q = (torch.arange(B)[:, None] % 4 == torch.arange(B)[None, :] % 4).double()
    Y_soft = torch.softmax(3 * alpha * Q, dim=1)
    spec = {"targ": "mp", "infonce": {"tsm": {"type": "softmax", "sm_scale": "pinned3"}}}
    _, correction, coverage, bound = L.infonce_block_resid(Y_soft, S, log_scale, clamp=False, full=False, coeff=coeff, spec=spec)
    assert coverage.tolist() == [1.0, 0.0, 0.0]
    R, *_ = L.infonce_train_resid(Y_soft, alpha, 3 * alpha, binary=True, inside=False)
    assert R.abs().sum(dim=1).max().item() < U and abs(correction.item()) <= bound[1].item()


def _reference_resid(Y, alpha):
    # the residual row by row from the independent mpmath reference (tests.unit.infonce_reference), as float64
    import mpmath as mp
    from tests.unit import infonce_reference as ref
    rows = []
    for y in Y.tolist():
        out = ref.reference(y, alpha=alpha)
        with mp.workdps(out["dps"]):
            rows.append([float(r) for r in out["R"]])
    return torch.tensor(rows, dtype=torch.float64)


@pytest.mark.parametrize("block", ["alpha", "full"])
@pytest.mark.parametrize("blend", [False, True])
def test_block_residuals_under_unitless_apply_the_coefficient_to_the_reference_residual(block, blend):
    # hard targets under loss.unitless: a term's coefficient c = w / L is applied once to its loss and once
    # to its correction, so the scale gradient's delta is -sum_k c_k * alpha * (the term's residual contracted
    # against sim + sim.T), the residual from the independent mpmath reference -- for one term and, under a
    # loss blend, per term at each term's own coefficient; under full, dL/dS drops each term's
    # -c_k * alpha * (R + R.T) / (2 B) likewise
    L = import_loss_module()
    B, K, alpha = 16, 4, 3.0
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=23)
    linear = lambda t: {"targ": t, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}
    specs = L.targ_specs(0.3, linear("sp"), linear("mp")) if blend else [(1.0, linear("mp"))]

    def run(block):
        log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
        crit = _hard_targ_crit(L, None, B, K, block, specs=specs, lambda_=0.3 if blend else 0.0,
                               blend_type="loss" if blend else "targ", unitless=True)
        Sg = S.clone().requires_grad_(True)
        loss, loss_raw, _, _ = crit(Sg * log_scale.exp(), class_encs, None, True, log_scale, Sg)
        loss.backward()
        return loss.detach(), loss_raw.detach(), log_scale.grad, Sg.grad, crit

    off = run(None)
    on = run(block)

    for i in (0, 1):  # loss, loss_raw
        torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
    assert off[0].item() == pytest.approx(1.0, rel=1e-9)  # unitless: the terms read their weights, summing to 1
    Z = alpha * S
    log_p, log_pt = torch.log_softmax(Z, dim=1), torch.log_softmax(Z.T, dim=1)
    T = S + S.T
    expected_scale, expected_S = 0.0, torch.zeros_like(S)
    for w, spec in specs:
        Q = torch.eye(B, dtype=torch.float64) if spec["targ"] == "sp" else (class_encs[:, None] == class_encs[None, :]).double()
        Y = Q / Q.sum(dim=1, keepdim=True)
        Y32 = Y.float().double()  # the loss scores a float32 copy of the target (Criterion.__call__)
        L_k = 0.5 * ((-Y32 * log_p).sum(dim=1).mean() + (-Y32 * log_pt).sum(dim=1).mean())
        c = w / max(L_k.item(), 1e-12)
        R = _reference_resid(Y, alpha)
        expected_scale += -c * alpha * 0.5 * ((R * T).sum(dim=1)).mean().item()
        expected_S = expected_S - c * 0.5 * alpha * (R + R.T) / B
    assert (on[2] - off[2]).item() == pytest.approx(expected_scale, rel=1e-8)
    assert on[4].dlogscale_correction.item() == pytest.approx((on[2] - off[2]).item(), rel=1e-9)
    assert on[4].block_stats["block_coverage"] == [[1.0, 0.0, 0.0]] * len(specs)
    if block == "full":
        torch.testing.assert_close(on[3] - off[3], expected_S, rtol=1e-8, atol=1e-14)
    else:
        torch.testing.assert_close(on[3], off[3], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("block", ["alpha", "full"])
def test_block_residuals_mixed_batch_applies_resolved_rows_and_skips_the_rest(block, monkeypatch):
    # one batch, four kinds of anchor row: hard binary rows (closed form), a graded row inside the band (feasible:
    # nothing to apply), graded rows outside it (the active sets) and one such row the active sets are made to
    # leave unresolved (skipped: its ordinary gradient stands). The selection is per anchor before the directions
    # fold, so under full a skipped anchor i still receives anchor j's correction at cell (i, j); the loss reading
    # and the row averaging are untouched throughout
    L = import_loss_module()
    B, K, alpha = 16, 4, 3.0
    class_encs = torch.arange(B) % K
    fam = class_encs // 2
    Q = (class_encs[:, None] == class_encs[None, :]).double()  # rows 0-7: hard
    Q[8] = torch.linspace(1.0, 0.5, B, dtype=torch.float64)  # row 8: all positive, log range log 2 < 2 alpha
    graded = torch.where(Q > 0, 1.0, torch.where(fam[:, None] == fam[None, :], 0.5, 0.0)).double()
    Q[9:] = graded[9:]  # rows 9-15: a 0.5 level beside the zeros
    Y = Q / Q.sum(dim=1, keepdim=True)
    S = _sims(B, seed=23)
    real = L.infonce_active_set_resid

    def leave_15_unresolved(Y_, alpha_):
        R, resolved = real(Y_, alpha_)
        resolved = resolved.clone()
        resolved[15] = False
        return torch.where(resolved[:, None], R, torch.zeros_like(R)), resolved

    monkeypatch.setattr(L, "infonce_active_set_resid", leave_15_unresolved)

    def run(block):
        log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
        crit = _hard_targ_crit(L, "tax", B, K, block)
        crit._targets = lambda *args: [Q]
        Sg = S.clone().requires_grad_(True)
        loss, loss_raw, _, Y_crit = crit(Sg * log_scale.exp(), class_encs, None, True, log_scale, Sg)
        loss.backward()
        return loss.detach(), loss_raw.detach(), log_scale.grad, Sg.grad, Y_crit, crit

    off = run(None)
    on = run(block)

    torch.testing.assert_close(off[4], Y, rtol=0.0, atol=0.0)
    for i in (0, 1):
        torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
    R_used = _reference_resid(Y, alpha)
    assert (R_used[9:] != 0.0).any(dim=1).all() and (R_used[8] == 0.0).all()
    R_used[15] = 0.0  # the row left unresolved
    T = S + S.T
    expected_scale = -alpha * 0.5 * ((R_used * T).sum(dim=1)).mean().item()
    assert (on[2] - off[2]).item() == pytest.approx(expected_scale, rel=1e-8)
    assert on[5].block_stats["block_coverage"] == [pytest.approx([14 / 16, 1 / 16, 1 / 16])]
    if block == "full":
        expected_S = -0.5 * alpha * (R_used + R_used.T) / B
        torch.testing.assert_close(on[3] - off[3], expected_S, rtol=1e-8, atol=1e-14)
        # anchor 15 is skipped, anchor 0 is not: cell (15, 0) still carries anchor 0's half
        assert (on[3] - off[3])[15, 0].item() == pytest.approx(-0.5 * alpha * R_used[0, 15].item() / B, rel=1e-8)
        assert R_used[0, 15].item() != 0.0
    else:
        torch.testing.assert_close(on[3], off[3], rtol=0.0, atol=0.0)


def test_block_residuals_resolve_the_high_scale_unitless_case_exactly():
    # the case that broke the subtraction path: a nearly one-hot graded target (softmax tsm at sm_scale
    # pinned3, alpha 55) fitted to a raw loss ~ 1e-12, so loss.unitless' coefficient sits at its 1e12 floor.
    # The solve's noise on a cap entry (~1e-14, against a true residual ~1e-48) times that coefficient made a
    # correction of order one where the true one is ~1e-34. Now rows 1 and 2 (three levels) come out of the
    # active sets and row 3 (two levels) of its closed form, all three to relative precision: the applied
    # correction is the reference's ~1e-34, which vanishes under the gradient's own rounding, while the
    # subtraction would still have produced its order-one number
    L = import_loss_module()
    Q = torch.tensor([[1.0, 0.81, 0.0], [0.81, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    alpha = 55.0
    S = torch.eye(3, dtype=torch.float64)  # orthonormal embeddings: the fitted geometry
    spec = {"targ": "phylo", "infonce": {"tsm": {"type": "softmax", "sm_scale": "pinned3"}}}

    def run(block):
        log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
        crit = _hard_targ_crit(L, None, 3, 3, block, specs=[(1.0, spec)], unitless=True)
        crit._targets = lambda *args: [Q]
        Sg = S.clone().requires_grad_(True)
        loss, loss_raw, _, Y = crit(Sg * log_scale.exp(), torch.arange(3), None, True, log_scale, Sg)
        loss.backward()
        return loss.detach(), loss_raw.detach(), log_scale.grad, Sg.grad, Y, crit

    off = run(None)
    assert off[1].item() < 1e-11  # the regime: a unitless coefficient of ~1e12
    Y = off[4]
    would_be = (1.0 / max(off[1].item(), 1e-12)) * alpha * 0.5 * (((L.infonce_p_opt(Y, alpha) - Y) * (2 * S)).sum(dim=1)).mean().item()
    assert abs(would_be) > 1e-2  # what the subtraction path would have injected
    c = 1.0 / max(off[1].item(), 1e-12)
    R_ref = _reference_resid(Y, alpha)
    expected = -c * alpha * 0.5 * ((R_ref * (2 * S)).sum(dim=1)).mean().item()
    assert 0.0 < expected < 1e-30  # the true correction, some thirty orders under what the subtraction injected
    for block in ("alpha", "full"):
        on = run(block)
        for i in (0, 1):  # loss, loss_raw
            torch.testing.assert_close(on[i], off[i], rtol=0.0, atol=0.0)
        stats = on[5].block_stats
        assert stats["block_coverage"] == [[1.0, 0.0, 0.0]]  # every row resolved
        assert stats["dlogscale_correction"] == pytest.approx(expected, rel=1e-9)
        # applied, that correction vanishes under the gradient's own float64 rounding: nothing measurable is injected
        assert abs((on[2] - off[2]).item()) <= 1e-20 and (on[3] - off[3]).abs().max().item() <= 1e-20


def test_block_residuals_skip_a_row_whose_closed_form_is_not_finite(monkeypatch):
    # the optional correction never contaminates the loss or the gradient: a row whose closed form comes back
    # non-finite is skipped by selection (torch.where), not by a multiply that would carry the NaN through,
    # and the other rows' corrections stand
    L = import_loss_module()
    B, alpha = 16, 3.0
    _, Q, Y = _hard_targs(B, 4, "mp")
    S = _sims(B, seed=5).requires_grad_(True)
    real = L.infonce_hard_resid

    def poisoned(support, a):
        R = real(support, a)
        R[0, 0] = float("nan")
        return R

    monkeypatch.setattr(L, "infonce_hard_resid", poisoned)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)

    term, correction, coverage, _ = L.infonce_block_resid(Y, S, log_scale, clamp=False, full=True, coeff=1.0)
    grad, grad_S = torch.autograd.grad(term, (log_scale, S))

    assert term.item() == 0.0 and math.isfinite(grad.item()) and torch.isfinite(grad_S).all()
    R_used = real(Q, alpha)
    R_used[0] = 0.0
    assert grad.item() == pytest.approx(-alpha * 0.5 * ((R_used * (S + S.T)).sum(dim=1)).mean().item(), rel=1e-9)
    torch.testing.assert_close(grad_S, -0.5 * alpha * (R_used + R_used.T) / B, rtol=1e-9, atol=1e-14)
    assert correction.item() == pytest.approx(grad.item(), rel=1e-12)
    assert coverage.tolist() == pytest.approx([15 / 16, 0.0, 1 / 16])


def test_block_resid_runs_the_solver_only_where_the_spec_leaves_rows_to_it(monkeypatch):
    # the bisection the active sets start from (_infonce_floor_eta) runs only for a term that can have graded rows
    # past the band: never for binary memberships (two-level rows, the closed forms) nor for a mapping that keeps
    # every row inside the band (pinned, pinned1); a graded target past the band runs it once
    L = import_loss_module()
    calls = []
    real = L._infonce_floor_eta
    monkeypatch.setattr(L, "_infonce_floor_eta", lambda *args, **kwargs: (calls.append(1), real(*args, **kwargs))[1])
    B, alpha = 16, 3.0
    _, Q_mp, Y_hard = _hard_targs(B, 4, "mp")
    Q = _graded_targs(B)
    S = _sims(B, seed=5)
    log_scale = torch.tensor(math.log(alpha), dtype=torch.float64, requires_grad=True)
    for targ, Y, cfg_tsm, expected, n_calls in (
        ("mp", Y_hard, {"type": "linear", "sm_scale": "pinned"}, [1.0, 0.0, 0.0], 0),
        ("mp", torch.softmax(3 * alpha * Q_mp, dim=1), {"type": "softmax", "sm_scale": "pinned3"}, [1.0, 0.0, 0.0], 0),
        ("tax", torch.softmax(alpha * Q, dim=1), {"type": "softmax", "sm_scale": "pinned1"}, [0.0, 1.0, 0.0], 0),
        ("tax", Q / Q.sum(dim=1, keepdim=True), {"type": "linear", "sm_scale": "pinned"}, [1.0, 0.0, 0.0], 1),
    ):
        calls.clear()
        spec = {"targ": targ, "infonce": {"tsm": cfg_tsm}}
        _, _, coverage, _ = L.infonce_block_resid(Y, S, log_scale, clamp=False, full=True, coeff=1.0, spec=spec)
        assert coverage.tolist() == expected and len(calls) == n_calls, (targ, cfg_tsm)


def test_block_stats_cover_separate_logit_scalars():
    # under separate logit scalars (loss.logits.shared false on a loss blend) each term's correction lands on
    # its own parameter: the record carries the primary's as dlogscale_correction and the second's as
    # dlogscale_correction2, each the delta its parameter's gradient actually took, with one coverage per term
    L = import_loss_module()
    B, K, lambda_ = 16, 4, 0.3
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=29)
    linear = lambda t: {"targ": t, "infonce": {"tsm": {"type": "linear", "sm_scale": "pinned"}}}
    specs = L.targ_specs(lambda_, linear("sp"), linear("mp"))

    def run(block):
        scales = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.0, math.log(3.0))]
        crit = _hard_targ_crit(L, None, B, K, block, specs=specs, lambda_=lambda_, blend_type="loss")
        crit.cfg["logits"]["shared"] = False
        assert crit.sep_scalars
        logits = [S * s.exp() for s in scales]
        loss, _, _, _ = crit(logits, class_encs, None, True, scales, S)
        loss.backward()
        return [s.grad.item() for s in scales], crit

    grads_off, _ = run(None)
    grads_on, crit = run("alpha")

    stats = crit.block_stats
    assert set(stats) == {"dlogscale_correction", "dlogscale_correction2", "block_coverage", "block_bound"}
    assert len(stats["block_bound"]) == 2 and all(b[0] > 0 and b[1] > 0 for b in stats["block_bound"])
    assert stats["dlogscale_correction"] == pytest.approx(grads_on[0] - grads_off[0], rel=1e-9)
    assert stats["dlogscale_correction2"] == pytest.approx(grads_on[1] - grads_off[1], rel=1e-9)
    assert stats["dlogscale_correction"] != 0.0 and stats["dlogscale_correction2"] != 0.0
    assert stats["block_coverage"] == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_correction_state_is_per_forward():
    # the criterion must not carry a training batch's correction into a later eval call: the stat is
    # cleared on every forward and only a training one sets it
    L = import_loss_module()
    B, K = 16, 4
    class_encs = torch.arange(B) % K
    S = _sims(B, seed=42)
    crit = _hard_targ_crit(L, "mp", B, K, "alpha")
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    crit(S * log_scale.exp(), class_encs, None, True, log_scale, S)
    assert crit.dlogscale_correction is not None
    assert set(crit.block_stats) == {"dlogscale_correction", "block_coverage", "block_bound"}
    crit(S * log_scale.exp(), class_encs, None, False, log_scale, S)
    assert crit.dlogscale_correction is None and crit.block_stats is None


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
    kl, E_u, E_ir, E_ur = L.infonce_kl_terms(Y, log_P, P_opt, *_subtracted(Y, P_opt))
    # each term by its definition, batch-meaned
    torch.testing.assert_close(kl, _kl_rows(Y, P).mean())
    torch.testing.assert_close(E_u, _kl_rows(P_opt, P).mean())
    torch.testing.assert_close(E_ir, _kl_rows(Y, P_opt).mean())
    torch.testing.assert_close(E_ur, ((Y - P_opt) * (P_opt / P).log()).sum(dim=1).mean())
    # the decomposition is exact, every part nonnegative, and D_KL(y || p) is this direction's raw
    # InfoNCE loss (per-anchor CE, anchor-averaged) less the targets' mean entropy
    torch.testing.assert_close(kl, E_ir + E_u + E_ur)
    assert kl > 0 and E_u > 0 and E_ir > 0 and E_ur > 0
    ce = -(Y * log_P).sum(dim=1).mean()
    H = -torch.xlogy(Y, Y).sum(dim=1).mean()
    torch.testing.assert_close(kl, ce - H)


def test_kl_terms_notebook_example():
    # MP-HCon-Intuition.ipynb: y = [1/2, 1/2, 0, 0], s = [0.9, 1, -0.9, -1] at alpha 3 reads
    # (kl, E_u, E_ir, E_ur) = (0.0145, 0.0113, 0.0025, 0.0007) to the notebook's printed precision
    L = import_loss_module()
    Y = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
    log_P = torch.log_softmax(3.0 * torch.tensor([[0.9, 1.0, -0.9, -1.0]], dtype=torch.float64), dim=1)
    P_opt = L.infonce_p_opt(Y, 3.0)
    terms = L.infonce_kl_terms(Y, log_P, P_opt, *_subtracted(Y, P_opt))
    assert terms.tolist() == pytest.approx([0.0145, 0.0113, 0.0025, 0.0007], abs=6e-5)


@pytest.mark.parametrize("alpha", [0.5, 3.0, 10.0])
def test_kl_cross_term_is_nonnegative_on_reachable_p(alpha):
    # E_ur >= 0 for every p a bounded-cosine softmax can realize (random geometries, the ideal
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
    kl, E_u, E_ir, E_ur = L.infonce_kl_terms(Y, P_opt.log(), P_opt, *_subtracted(Y, P_opt))
    torch.testing.assert_close(torch.stack([E_u, E_ur]), torch.zeros(2, dtype=torch.float64))
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
    for k, key in enumerate(("kl", "kl_u", "kl_ir", "kl_ur")):
        assert stats[key] == pytest.approx(expected[k].item(), rel=1e-9, abs=1e-12)
    assert stats["kl"] == pytest.approx(stats["kl_u"] + stats["kl_ir"] + stats["kl_ur"], rel=1e-9, abs=1e-12)
    # the irreducible part depends on the targets and alpha alone, not on the anchor direction
    assert i2t[2] == t2i[2]
    # kl is the criterion's loss_raw (the two directions' per-anchor CE means, averaged) less the
    # targets' mean entropy
    ce = 0.5 * sum(-(Yf * torch.log_softmax(M, dim=1)).sum(dim=1).mean() for M in (Z, Z.T))
    H = -torch.xlogy(Yf, Yf).sum(dim=1).mean()
    assert stats["kl"] == pytest.approx((ce - H).item(), rel=1e-9, abs=1e-12)


def _entropy_norm(A, dim):
    # the normalized Shannon entropy of |A| along dim, 0 log 0 = 0, written out
    A = A.abs()
    G = A / A.sum(dim=dim, keepdim=True)
    return -torch.xlogy(G, G).sum(dim=dim) / math.log(A.size(dim))


def _same(a, b, rel=1e-9):
    # equal, or both NaN (an entropy with no mass behind it)
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=rel)


def test_norm_entropy_reads_uniform_as_one_a_point_mass_as_zero_and_no_mass_as_nan():
    L = import_loss_module()
    assert L._norm_entropy(torch.full((4, 6), -0.3, dtype=torch.float64), 1).tolist() == pytest.approx([1.0] * 4)
    assert L._norm_entropy(torch.full((24,), 2.0, dtype=torch.float64), 0).item() == pytest.approx(1.0)
    A = torch.zeros(3, 5, dtype=torch.float64)
    A[:, 2] = -1.0
    assert L._norm_entropy(A, 1).tolist() == [0.0, 0.0, 0.0]  # 0 log 0 = 0, exactly
    assert math.isnan(L._norm_entropy(torch.zeros(5, dtype=torch.float64), 0).item())
    assert torch.isnan(L._norm_entropy(torch.zeros(2, 5, dtype=torch.float64), 1)).all()
    # sign and scale are not concentration: the entropy is of the magnitudes' distribution
    A = torch.randn(3, 7, generator=torch.Generator().manual_seed(0)).double()
    torch.testing.assert_close(L._norm_entropy(-2.5 * A, 1), L._norm_entropy(A, 1))


def test_batch_stats_sim_grad_entropies_match_their_definitions():
    # the pair entropy is of the gradient the towers receive, dL/dS: the two directions' per-pair terms folded,
    # (i2t + t2i.T) / 2, the scale alpha / B dropping out under the normalization; the anchor entropy is each
    # direction's over its own ACTIVE anchors' rows (any gradient mass), the two averaged, with the active fraction
    # beside it -- for each part of the decomposition: (p - y), (p - p*), (p* - y)
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float()
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, _log_scale(alpha), False)
    Yf = _y_stats(Y.float())
    P_opt = L.infonce_p_opt(Yf, alpha)
    R = P_opt - Yf
    P_i2t, P_t2i = torch.softmax(logits.double(), dim=1), torch.softmax(logits.double().T, dim=1)
    parts = {
        "full": (P_i2t - Yf, P_t2i - Yf),
        "struct": (P_i2t - P_opt, P_t2i - P_opt),
        "res": (R, R),
    }
    def anchor_stats(M):
        active = M.abs().sum(dim=1) > 0
        return _entropy_norm(M[active], 1).mean().item(), active.double().mean().item()

    for comp, (M_i2t, M_t2i) in parts.items():
        pair = _entropy_norm(0.5 * (M_i2t + M_t2i.T).flatten(), 0).item()
        (h_i2t, f_i2t), (h_t2i, f_t2i) = anchor_stats(M_i2t), anchor_stats(M_t2i)
        assert _same(stats[f"sim_grad_entropy_pair_{comp}"], pair)
        assert _same(stats[f"sim_grad_entropy_anchor_{comp}"], 0.5 * (h_i2t + h_t2i))
        assert stats[f"sim_grad_entropy_active_{comp}"] == pytest.approx(0.5 * (f_i2t + f_t2i), rel=1e-12)
        for level in ("pair", "anchor", "active"):
            val = stats[f"sim_grad_entropy_{level}_{comp}"]
            assert math.isnan(val) or 0.0 <= val <= 1.0
    # the full and structural gradients have mass on every anchor (a finite-logit softmax never equals a target
    # holding a zero, nor p*), and on this non-symmetric geometry the two directions' anchor entropies differ, so
    # the anchor entropy is a genuine average of the two rather than either; the residual's active anchors are
    # exactly the rows off the feasible path, a feasible row being its own projection with a residual of exactly zero
    for comp in ("full", "struct"):
        assert stats[f"sim_grad_entropy_active_{comp}"] == 1.0
        assert all(math.isfinite(stats[f"sim_grad_entropy_{level}_{comp}"]) for level in ("pair", "anchor"))
    assert stats["sim_grad_entropy_active_res"] == pytest.approx(1.0 - stats["resid_paths"][1], rel=1e-12)
    M_i2t, M_t2i = parts["full"]
    assert abs(_entropy_norm(M_i2t, 1).mean() - _entropy_norm(M_t2i, 1).mean()) > 1e-6


def test_sim_grad_entropies_on_a_hard_target_read_the_closed_forms():
    # sp targets (Y = I): the residual is the closed form, the K = 1 positive of a row at -M lam and its M = B - 1
    # negatives at lam, so its magnitude's row distribution is (1/2, 1/(2M), ...) -- H_anchor = log(4M) / (2 log B)
    # -- and over the batch (1/(2B) on the diagonal, 1/(2BM) off it) -- H_pair = log(4 B^2 M) / (2 log B^2), the
    # fold being the residual itself (R is symmetric here)
    L = import_loss_module()
    B, alpha = 10, 5.0
    M = B - 1
    Q = torch.eye(B, dtype=torch.float64)
    S = _sims(B, seed=3)
    stats = L.infonce_batch_stats(S, Q, Q, alpha * S, _log_scale(alpha), False)
    assert stats["resid_paths"] == [1.0, 0.0, 0.0]
    assert stats["sim_grad_entropy_anchor_res"] == pytest.approx(math.log(4 * M) / (2 * math.log(B)), rel=1e-12)
    assert stats["sim_grad_entropy_pair_res"] == pytest.approx(math.log(4 * B * B * M) / (2 * math.log(B * B)), rel=1e-12)
    for level in ("pair", "anchor"):
        assert all(0.0 <= stats[f"sim_grad_entropy_{level}_{comp}"] <= 1.0 for comp in ("full", "struct", "res"))
    assert all(stats[f"sim_grad_entropy_active_{comp}"] == 1.0 for comp in ("full", "struct", "res"))  # every row pushes


def test_sim_grad_entropies_read_nan_where_the_residual_is_exactly_zero():
    # a graded target inside the reachable band is its own projection (infonce_p_opt's feasibility branch), so the
    # residual is exactly zero everywhere: no gradient, no concentration to report -- NaN, as the coherence ratios
    # read under no pressure, with no anchor active -- while the full and structural entropies read as usual
    L = import_loss_module()
    B, alpha = 8, 6.0
    Q, _ = _targets(B, 4, seed=6)
    Y = torch.softmax(2 * Q, dim=1)  # the softmax tsm at sm_scale 1: a row's log range is at most 2 < 2 alpha
    S = _sims(B, seed=5)
    stats = L.infonce_batch_stats(S, Q, Y, alpha * S, _log_scale(alpha), False)
    assert stats["resid_paths"] == [0.0, 1.0, 0.0]
    for level in ("pair", "anchor"):
        assert math.isnan(stats[f"sim_grad_entropy_{level}_res"])
        assert all(0.0 <= stats[f"sim_grad_entropy_{level}_{comp}"] <= 1.0 for comp in ("full", "struct"))
    assert stats["sim_grad_entropy_active_res"] == 0.0
    assert stats["sim_grad_entropy_active_full"] == 1.0 and stats["sim_grad_entropy_active_struct"] == 1.0


def test_anchor_entropy_is_the_mean_over_the_active_anchors_with_their_fraction_beside_it():
    # one anchor with no residual at all -- a graded row inside the reachable band, its own projection exactly --
    # beside B - 1 sp rows: the residual's anchor entropy is the mean over the B - 1 anchors that carry one, each
    # an sp row's closed form log(4M) / (2 log B), rather than NaN over the one that does not (its normalization is
    # 0 / 0, no distribution at all -- unlike a row concentrated on one pair, which reads 0), and the active
    # fraction says how many anchors that mean stands on
    L = import_loss_module()
    B, alpha = 10, 5.0
    M = B - 1
    Y = torch.eye(B, dtype=torch.float64)
    Y[0] = torch.softmax(torch.linspace(0.0, 1.0, B, dtype=torch.float64), dim=0)  # log range 1 < 2 alpha: feasible
    S = _sims(B, seed=3)
    stats = L.infonce_batch_stats(S, torch.eye(B, dtype=torch.float64), Y, alpha * S, _log_scale(alpha), False)
    assert stats["resid_paths"] == pytest.approx([M / B, 1 / B, 0.0])
    assert stats["sim_grad_entropy_active_res"] == pytest.approx(M / B, rel=1e-12)
    assert stats["sim_grad_entropy_anchor_res"] == pytest.approx(math.log(4 * M) / (2 * math.log(B)), rel=1e-12)
    assert stats["sim_grad_entropy_active_full"] == 1.0 and stats["sim_grad_entropy_active_struct"] == 1.0


def test_actual_sim_grad_entropies_read_the_gradient_the_towers_received():
    # the (actual) trio off the criterion's own dL/dS: on an unweighted lone hard target the pair entropy is the
    # analytic full one exactly (the same folded gradient, the loss carrying no weight the stats leave out), and
    # under block_residuals: full it is the analytic STRUCTURAL one -- the residual the towers never received is
    # not in it -- while the anchor entropy reads the fold's rows and columns, its own construction (every row and
    # column active here: each entry of the fold pushes)
    L = import_loss_module()
    B, log_alpha = 16, math.log(3.0)
    class_encs, Q, Y = _hard_targs(B, 4, "sp")
    S = _sims(B, seed=23)
    stats = L.infonce_batch_stats(S, Q, Y, math.exp(log_alpha) * S, torch.tensor(log_alpha, dtype=torch.float64), False)
    for block, comp in ((None, "full"), ("full", "struct")):
        actual = L.sim_grad_entropies_actual(_run_crit(L, "sp", class_encs, S, log_alpha, block)[3])
        assert set(actual) == {f"sim_grad_entropy_{level}_actual" for level in ("pair", "anchor", "active")}
        assert actual["sim_grad_entropy_pair_actual"] == pytest.approx(stats[f"sim_grad_entropy_pair_{comp}"], rel=1e-9)
        assert actual["sim_grad_entropy_active_actual"] == 1.0
        assert 0.0 <= actual["sim_grad_entropy_anchor_actual"] <= 1.0
    assert stats["sim_grad_entropy_pair_full"] != pytest.approx(stats["sim_grad_entropy_pair_struct"], rel=1e-3)
    # the fold's rows and columns: a zero row leaves its column's mass to the other rows, so the row side reads
    # B - 1 active anchors and the column side all B, and the anchor entropy averages the two sides' means
    G = _sims(B, seed=5)
    G[0] = 0.0
    actual = L.sim_grad_entropies_actual(G)
    rows, cols = _entropy_norm(G[1:], 1).mean(), _entropy_norm(G.T, 1).mean()
    assert actual["sim_grad_entropy_anchor_actual"] == pytest.approx(0.5 * (rows + cols).item(), rel=1e-12)
    assert actual["sim_grad_entropy_active_actual"] == pytest.approx(0.5 * ((B - 1) / B + 1.0), rel=1e-12)
    assert actual["sim_grad_entropy_pair_actual"] == pytest.approx(_entropy_norm(G.flatten(), 0).item(), rel=1e-12)


def test_anchor_entropy_averages_only_the_directions_with_an_active_anchor():
    # a uniform target under logits whose I2T row softmax reproduces it exactly, so the image anchors carry no
    # gradient at all, while the T2I one does not, so both text anchors do: the anchor entropy is the text side's
    # reading (its two rows uniform in magnitude: 1) rather than NaN with the image side, and the active fraction
    # still counts the empty side (0.5). With neither side active -- zero logits, both softmaxes the target -- the
    # entropy is NaN and the fraction 0, as the pair entropy is NaN; the actual trio reads a zero gradient the same
    L = import_loss_module()
    Y = torch.full((2, 2), 0.5, dtype=torch.float64)
    S = _sims(2, seed=1)
    Z = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    stats = L.infonce_batch_stats(S, (Y > 0).double(), Y, Z, _log_scale(1.0), False)
    assert stats["resid_paths"] == [0.0, 1.0, 0.0]  # a uniform row is its own projection: no residual anywhere
    for comp in ("full", "struct"):  # p* = y here, so the structural part is the whole gradient
        assert stats[f"sim_grad_entropy_anchor_{comp}"] == pytest.approx(1.0)
        assert stats[f"sim_grad_entropy_active_{comp}"] == 0.5
        assert stats[f"sim_grad_entropy_pair_{comp}"] == pytest.approx(1.0)  # the fold's four entries equal in magnitude
    assert math.isnan(stats["sim_grad_entropy_anchor_res"]) and stats["sim_grad_entropy_active_res"] == 0.0

    stats = L.infonce_batch_stats(S, (Y > 0).double(), Y, torch.zeros(2, 2, dtype=torch.float64), _log_scale(1.0), False)
    for comp in ("full", "struct", "res"):
        assert math.isnan(stats[f"sim_grad_entropy_anchor_{comp}"]) and math.isnan(stats[f"sim_grad_entropy_pair_{comp}"])
        assert stats[f"sim_grad_entropy_active_{comp}"] == 0.0
    actual = L.sim_grad_entropies_actual(torch.zeros(2, 2, dtype=torch.float64))
    assert math.isnan(actual["sim_grad_entropy_pair_actual"]) and math.isnan(actual["sim_grad_entropy_anchor_actual"])
    assert actual["sim_grad_entropy_active_actual"] == 0.0

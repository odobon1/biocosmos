"""
The InfoNCE reachable-optimum diagnostics (utils.loss.infonce_p_opt / infonce_scale_grad_sums /
infonce_kl_terms / infonce_batch_stats), p* being the row softmax's reachable optimum under
bounded-cosine logits (max / min ratio at most exp(2 alpha)). The logit-scale gradient
decomposition: per pair, dL/dalpha = (p - y) s splits into a structural part (p - p*) s -- what the
model could still remove at this alpha -- and a residual (p* - y) s that no similarity geometry can;
each is attributed to the positive / negative target mass by the soft masks q / 1 - q, and reported
summed, summed in magnitude, and as the coherence ratio C = |sum| / sum|.|. The KL decomposition:
per anchor, D_KL(y || p) = D_KL(y || p*) + D_KL(p* || p) + <y - p*, log(p* / p)> = E_ir + E_s + E_sr,
batch-meaned. Both averaged over the I2T / T2I anchor directions.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.unit.test_loss_targets import import_loss_module


def _p_opt_single_row(y, alpha, n_iter=40):
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
    # a target already inside the ratio band is its own projection
    L = import_loss_module()
    Y = torch.softmax(torch.rand(6, 8, generator=torch.Generator().manual_seed(3)).double(), dim=1)  # ratio < e
    torch.testing.assert_close(L.infonce_p_opt(Y, 1.0), Y, rtol=1e-9, atol=1e-12)


def test_sums_decompose_and_full_sum_is_the_loss_gradient():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=4)
    S = _sims(B, seed=5)
    P = torch.softmax(alpha * S, dim=1)
    sums = L.infonce_scale_grad_sums(S, Q, Y, P, L.infonce_p_opt(Y, alpha))
    assert sums.shape == (2, 3, 3)
    # the full / all sum is d(per-anchor mean CE)/d(alpha), the direction's raw InfoNCE loss gradient
    a = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    loss = -(Y * torch.log_softmax(a * S, dim=1)).sum(dim=1).mean()
    (dloss_dalpha,) = torch.autograd.grad(loss, a)
    torch.testing.assert_close(sums[0, 0, 0], dloss_dalpha)
    # full = struct + res, all = pos + neg (exact for the sums; the magnitudes only bound them)
    torch.testing.assert_close(sums[0, 0], sums[0, 1] + sums[0, 2])
    torch.testing.assert_close(sums[0, :, 0], sums[0, :, 1] + sums[0, :, 2])
    assert torch.all(sums[1] >= sums[0].abs() - 1e-12)
    assert torch.all(sums[1, 0] <= sums[1, 1] + sums[1, 2] + 1e-12)
    assert torch.all(sums[1, :, 0] <= sums[1, :, 1] + sums[1, :, 2] + 1e-12)


def test_batch_stats_average_directions_and_take_C_on_the_averages():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float() + 0.3  # a bias is inert under the row softmax
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, torch.tensor(alpha), idx=2)
    aggs, comps = ("sum", "sum_abs", "C"), ("full", "struct", "res")
    assert set(stats) == {f"{prefix}2_{agg}_{comp}" for prefix in ("dalpha", "dlogalpha") for agg in aggs for comp in comps} | {
        "kl2", "kl2_s", "kl2_ir", "kl2_sr", "alpha_req2_min", "alpha_req2_mean", "alpha_req2_max"}
    # the log-scale family: d/d(log alpha) = alpha * d/dalpha, so alpha times the sums and the same C
    for comp in comps:
        for agg in aggs[:2]:
            assert stats[f"dlogalpha2_{agg}_{comp}"] == pytest.approx([alpha * v for v in stats[f"dalpha2_{agg}_{comp}"]], rel=1e-12)
        assert stats[f"dlogalpha2_C_{comp}"] == pytest.approx(stats[f"dalpha2_C_{comp}"], rel=1e-12)
    Sf, Qf, Yf = S.float().double(), Q.float().double(), Y.float().double()
    P_opt = L.infonce_p_opt(Yf, alpha)
    i2t = L.infonce_scale_grad_sums(Sf, Qf, Yf, torch.softmax(logits.double(), dim=1), P_opt)
    t2i = L.infonce_scale_grad_sums(Sf.T, Qf.T, Yf, torch.softmax(logits.double().T, dim=1), P_opt)
    expected = 0.5 * (i2t + t2i)
    for a, agg in enumerate(aggs[:2]):
        for c, comp in enumerate(comps):
            assert stats[f"dalpha2_{agg}_{comp}"] == pytest.approx(expected[a, c].tolist(), rel=1e-9, abs=1e-12)
    for c, comp in enumerate(comps):
        C = stats[f"dalpha2_C_{comp}"]
        assert C == pytest.approx((expected[0, c].abs() / (expected[1, c] + 1e-30)).tolist(), rel=1e-9, abs=1e-12)
        assert all(0.0 <= v <= 1.0 for v in C)
    # a symmetric S (and Q, Y) makes the two directions coincide, so the reported values are either's
    S_sym = 0.5 * (S + S.T)
    stats_sym = L.infonce_batch_stats(S_sym, Q, Y, alpha * S_sym, alpha, idx=1)
    one_dir = L.infonce_scale_grad_sums(S_sym, Q, Y, torch.softmax(alpha * S_sym, dim=1), L.infonce_p_opt(Y, alpha))
    for c, comp in enumerate(comps):
        assert stats_sym[f"dalpha1_sum_{comp}"] == pytest.approx(one_dir[0, c].tolist(), rel=1e-9, abs=1e-12)
        assert stats_sym[f"dalpha1_sum_abs_{comp}"] == pytest.approx(one_dir[1, c].tolist(), rel=1e-9, abs=1e-12)


def test_batch_stats_row_wise_scale_bounds():
    # the target-implied scale bound is row-wise (softmax feasibility is): per row i, the smallest
    # alpha whose logit range 2 alpha spans log(Y_i), 0.5 * log(max_j Y_ij / min_j Y_ij), reported as
    # its min / mean / max over rows -- a global max(Y) / min(Y) would pair extremes from different
    # rows and overstate the batch's requirement
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y_lin = _targets(B, 4, seed=6)
    S = _sims(B, seed=7)
    logits = (alpha * S).float()
    Y = torch.softmax(2.0 * Q * 3.0, dim=1)  # the softmax tsm at sm_scale 3: zero-free, every row finite
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, alpha, idx=1)
    Yf = Y.float().double()
    alpha_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))  # == 3 * (max_j Q_ij - min_j Q_ij) per row
    assert stats["alpha_req1_min"] == pytest.approx(alpha_req.min().item(), rel=1e-9)
    assert stats["alpha_req1_mean"] == pytest.approx(alpha_req.mean().item(), rel=1e-9)
    assert stats["alpha_req1_max"] == pytest.approx(alpha_req.max().item(), rel=1e-9)
    # under the linear tsm a row with an exact zero (the mp rows of _targets) sits at infinity, taking
    # the max and the mean with it, while the min still reads off the graded rows
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y_lin.float(), logits, alpha, idx=1)
    Yf = Y_lin.float().double()
    alpha_req = 0.5 * torch.log(Yf.amax(1) / Yf.amin(1))
    assert math.isinf(alpha_req.max()) and torch.isfinite(alpha_req).any()
    assert stats["alpha_req1_min"] == pytest.approx(alpha_req[torch.isfinite(alpha_req)].min().item(), rel=1e-9)
    assert math.isinf(stats["alpha_req1_mean"]) and math.isinf(stats["alpha_req1_max"])


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
    kl, E_s, E_ir, E_sr = L.infonce_kl_terms(Y, log_P, P_opt)
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
    terms = L.infonce_kl_terms(Y, log_P, L.infonce_p_opt(Y, 3.0))
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
        assert L.infonce_kl_terms(Y, torch.log_softmax(alpha * S, dim=1), P_opt)[3] >= -1e-12
    kl, E_s, E_ir, E_sr = L.infonce_kl_terms(Y, P_opt.log(), P_opt)
    torch.testing.assert_close(torch.stack([E_s, E_sr]), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(kl, E_ir)


def test_batch_stats_kl_keys_average_directions():
    L = import_loss_module()
    B, alpha = 12, 4.0
    Q, Y = _targets(B, 4, seed=12)
    S = _sims(B, seed=13)
    logits = (alpha * S).float() - 0.7  # a bias is inert under the row softmax
    stats = L.infonce_batch_stats(S.float(), Q.float(), Y.float(), logits, alpha, idx=1)
    Yf, Z = Y.float().double(), logits.double()
    P_opt = L.infonce_p_opt(Yf, alpha)
    i2t = L.infonce_kl_terms(Yf, torch.log_softmax(Z, dim=1), P_opt)
    t2i = L.infonce_kl_terms(Yf, torch.log_softmax(Z.T, dim=1), P_opt)
    expected = 0.5 * (i2t + t2i)
    for k, key in enumerate(("kl1", "kl1_s", "kl1_ir", "kl1_sr")):
        assert stats[key] == pytest.approx(expected[k].item(), rel=1e-9, abs=1e-12)
    assert stats["kl1"] == pytest.approx(stats["kl1_s"] + stats["kl1_ir"] + stats["kl1_sr"], rel=1e-9, abs=1e-12)
    # the irreducible part depends on the targets and alpha alone, not on the anchor direction
    assert i2t[2] == t2i[2]
    # kl1 is the criterion's loss_raw (the two directions' per-anchor CE means, averaged) less the
    # targets' mean entropy
    ce = 0.5 * sum(-(Yf * torch.log_softmax(M, dim=1)).sum(dim=1).mean() for M in (Z, Z.T))
    H = -torch.xlogy(Yf, Yf).sum(dim=1).mean()
    assert stats["kl1"] == pytest.approx((ce - H).item(), rel=1e-9, abs=1e-12)

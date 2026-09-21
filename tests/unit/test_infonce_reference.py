"""
The independent adaptive-precision reference (tests/unit/infonce_reference.py) and, against it, the residual
calculations that already exist: the hard-binary-target closed forms (utils.loss.infonce_hard_resid /
infonce_hard_kl_ir -- the form the loss.infonce.block_residuals TRAINING correction is built on, so this is
its validation at scale too) and infonce_p_opt's feasibility branch.

The reference comes first: it has to be shown to resolve what it claims (an exp(-2 alpha)-sized residual, where
a fixed precision silently returns zero), to reproduce cases whose answer is known in closed form, and to agree
with the production bisection where THAT is trustworthy. Comparisons are made in mpmath, at the reference's
own precision: a float() of these values is 0.0 from alpha ~ 372.
"""
import math

import mpmath as mp
import pytest
import torch

from tests.unit import infonce_reference as ref
from tests.unit.test_loss_targets import import_loss_module

CORPUS = ref.corpus()


def _rel(got, want):
    return abs(mp.mpf(got) - want) / abs(want)


@pytest.mark.parametrize("name,y,scale", CORPUS, ids=[name for name, _, _ in CORPUS])
def test_reference_resolves_every_corpus_row(name, y, scale):
    # reference() itself raises unless two precisions agree on R and E_ir; what is left to check is that the
    # answer is a projection at all: mass is conserved, p* sits inside the ratio band, the residual only ever
    # lifts floor entries and shaves cap ones, and the divergence is non-negative
    out = ref.reference(y, **scale)
    with mp.workdps(out["dps"]):
        r = mp.exp(-2 * mp.mpf(scale["alpha"])) if "alpha" in scale else mp.mpf(scale["r"])
        tiny = mp.mpf(10) ** -(ref.SIG_DIGITS + 2) * r / len(y)
        assert abs(mp.fsum(out["R"])) <= tiny
        assert min(out["P"]) >= r * max(out["P"]) * (1 - mp.mpf(10) ** -ref.SIG_DIGITS)
        assert out["E_ir"] >= 0
        if out["feasible"]:
            assert all(R == 0 for R in out["R"]) and out["E_ir"] == 0
        else:
            c = out["c"]
            for v, R in zip(y, out["R"]):
                assert (R > 0) == (mp.mpf(v) < r * c) and (R < 0) == (mp.mpf(v) > c)


def test_a_fixed_precision_reproduces_the_failure_the_reference_exists_to_catch():
    # y = [1, 0]: R_+ = -r / (1 + r). Resolving it takes ~2 alpha / ln 10 digits -- 131 at alpha 150 -- so at a
    # fixed 120 digits p*_+ - 1 evaluates to exactly 0: accurate-looking probabilities, the residual rounded
    # away. The reference's own solve REFUSES at that precision (it cannot meet a termination criterion tied to
    # the residual) rather than hand the zero back, and the adaptive precision recovers the values (checked
    # independently at 450 digits)
    for alpha, expected in ((150.0, "-5.1482e-131"), (200.0, "-1.9152e-174"), (400.0, "-3.6679e-348")):
        with mp.workdps(120):
            assert 1 / (1 + mp.exp(-2 * mp.mpf(alpha))) - 1 == 0
        with pytest.raises(RuntimeError, match="did not converge"):
            ref._solve([1.0, 0.0], alpha, None, 120)
        out = ref.reference([1.0, 0.0], alpha=alpha)
        with mp.workdps(out["dps"]):
            assert _rel(out["R"][0], mp.mpf(expected)) < 1e-4
            r = mp.exp(-2 * mp.mpf(alpha))
            assert _rel(out["R"][0], -r / (1 + r)) < mp.mpf(10) ** -ref.SIG_DIGITS


def test_reference_reproduces_the_cases_known_in_closed_form():
    cases = {name: (y, scale) for name, y, scale in CORPUS}
    d = mp.mpf(2) ** -53
    # exact residuals [-d, d, 0] at r = 1/2, and a divergence that is SECOND order in them, 3 d^2 + O(d^3)
    out = ref.reference(*cases["second-order KL"][:1], **cases["second-order KL"][1])
    with mp.workdps(out["dps"]):
        assert out["R"] == [-d, d, 0]
        exact = (mp.mpf(1) / 2 + d) * mp.log(1 + 2 * d) + (mp.mpf(1) / 4 - d) * mp.log(1 - 4 * d)
        assert _rel(out["E_ir"], exact) < mp.mpf(10) ** -ref.SIG_DIGITS
        assert _rel(out["E_ir"], mp.mpf("3.69778549322e-32")) < 1e-11
    # both positives capped at c = (0.41 + 0.59) / (1/2 + 2), their spread 1.44 notwithstanding
    out = ref.reference(*cases["cap spread"][:1], **cases["cap spread"][1])
    with mp.workdps(out["dps"]):
        c = (mp.mpf(0.41) + mp.mpf(0.59)) / (mp.mpf(1) / 2 + 2)
        assert _rel(out["c"], c) < mp.mpf(10) ** -ref.SIG_DIGITS and out["R"][1] < 0 and out["R"][2] < 0
    # the row a float64 membership test mislabels: entry 2 IS capped, by -N_2 / D with N_2 ~ -3.25e-18
    out = ref.reference(*cases["false acceptance"][:1], **cases["false acceptance"][1])
    with mp.workdps(out["dps"]):
        assert out["R"][1] < 0 and abs(out["R"][1]) < mp.mpf("1e-17")
    # exact ties: an entry sitting exactly on the cap / the floor has residual exactly 0 under either label
    for name in ("tie at the cap", "tie at the floor"):
        assert ref.reference(cases[name][0], **cases[name][1])["R"][1] == 0
    # ... and one float to either side of a boundary moves it by that one float, no more
    for name, sign in (("one step over the cap", -1), ("one step under the floor", 1)):
        R = ref.reference(cases[name][0], **cases[name][1])["R"][1]
        assert R * sign > 0 and abs(R) < mp.mpf("1e-15")
    for name in ("one step under the cap", "one step over the floor"):
        assert ref.reference(cases[name][0], **cases[name][1])["R"][1] == 0  # interior: untouched
    assert ref.reference(cases["one step inside feasibility"][0], r=0.5)["feasible"]
    assert not ref.reference(cases["one step outside feasibility"][0], r=0.5)["feasible"]


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.0])
def test_reference_matches_the_bisection_where_the_bisection_is_trustworthy(alpha):
    # at low alpha the production solve (infonce_p_opt) is accurate to ~1e-15 in p*, an independent formulation
    # from the reference's: random graded rows, some holding zeros
    L = import_loss_module()
    g = torch.Generator().manual_seed(int(alpha * 10))
    Y = torch.rand(12, 9, generator=g, dtype=torch.float64) ** 3
    Y[torch.rand(12, 9, generator=g) < 0.3] = 0.0
    Y[:, 0] = Y[:, 0] + 0.1  # no all-zero row
    Y = Y / Y.sum(dim=1, keepdim=True)
    P = L.infonce_p_opt(Y, alpha)
    for y_row, p_row in zip(Y.tolist(), P.tolist()):
        out = ref.reference(y_row, alpha=alpha)
        assert max(abs(mp.mpf(p) - p_ref) for p, p_ref in zip(p_row, out["P"])) < 1e-13


HARD = [(K, B, alpha) for K, B in ((1, 8), (3, 8), (4, 16), (7, 16))
        for alpha in (0.5, 3.0, 15.0, 25.0, 55.0, 100.0, 150.0, 200.0, 300.0)]


@pytest.mark.parametrize("K,B,alpha", HARD)
def test_hard_closed_forms_match_the_reference_across_scale_and_class_ratio(K, B, alpha):
    # the closed forms the block_residuals training correction and the hard-row diagnostics share, against the
    # reference from alpha 0.5 to 300 -- residuals from 1e-1 down to 1e-261 -- at class ratios whose 1/K is and
    # is not representable. Two readings of "the target", kept apart:
    #   STORED   -- the float64 row production holds, fl(1/K) renormalized: the reference's own input
    #   INTENDED -- 1/K exactly, which is what the closed form encodes (it reads K and M off the support)
    # They differ by O(u) RELATIVE (the projection is scale-covariant), so the closed form has to match both to
    # far inside the tolerance here; the remainder is exp's own rounding of r, a few u relative
    L = import_loss_module()
    assert alpha <= ref.ALPHA_SUPPORTED_MAX
    y = ref.hard_row(K, B)
    Q = torch.tensor([[1.0] * K + [0.0] * (B - K)], dtype=torch.float64)
    R = L.infonce_hard_resid(Q, alpha)[0].tolist()
    E_ir = L.infonce_hard_kl_ir(Q, alpha)[0].item()
    assert all(math.isfinite(v) and v != 0.0 for v in R) and E_ir > 0.0  # resolved, not rounded away

    stored = ref.reference(y, alpha=alpha)
    intended = ref.reference([mp.mpf(1) / K] * K + [mp.mpf(0)] * (B - K), alpha=alpha)
    for out in (stored, intended):
        with mp.workdps(out["dps"]):
            assert max(_rel(got, want) for got, want in zip(R, out["R"])) < 1e-13
            assert _rel(E_ir, out["E_ir"]) < 1e-13


def test_hard_closed_form_past_the_normal_range_rounds_to_zero():
    # outside the supported regime the true residual leaves float64 altogether: at alpha 400 it is 3.7e-348,
    # under the smallest denormal, so the closed form's 0.0 IS the correctly rounded value -- and says nothing.
    # Harmless in training (the parameter's float32 gradient could not carry it either); in the diagnostics it
    # must not read as a resolved zero, which is the integration's to mark, not this form's to fix
    L = import_loss_module()
    Q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    assert L.infonce_hard_resid(Q, 400.0)[0].tolist() == [0.0, 0.0]
    out = ref.reference(ref.hard_row(1, 2), alpha=400.0)
    with mp.workdps(out["dps"]):
        assert 0 < abs(out["R"][0]) < mp.mpf(5e-324)


def test_feasible_branch_hands_a_feasible_row_back_exactly():
    # a row inside the band is its own projection, and infonce_p_opt returns it bitwise rather than through the
    # solve's log / exp roundtrip -- the reference agreeing that the residual there is exactly zero, including
    # one float inside the feasibility boundary; one float outside it is no longer feasible
    L = import_loss_module()
    for name, y, scale in CORPUS:
        if "alpha" not in scale or not ref.reference(y, **scale)["feasible"]:
            continue
        Y = torch.tensor([y], dtype=torch.float64)
        assert torch.equal(L.infonce_p_opt(Y, scale["alpha"]), Y), name

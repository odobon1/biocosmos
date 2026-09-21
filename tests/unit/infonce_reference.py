"""
An independent, adaptive-precision reference for the InfoNCE reachable-set projection p* and the residual
family built on it (R = p* - y, E_ir = D_KL(y || p*)), plus the corpus of rows the diagnostics are checked
against. Not a test module: tests import it (test_infonce_reference.py).

THE PROBLEM SOLVED. Given a row y >= 0 and a logit scale alpha, with r = exp(-2 alpha): the closest row in
cross-entropy that a bounded-cosine softmax can realize is p* = clamp(y, r c, c) -- floor r c, cap c -- with
the cap set by mass balance, H(c) = 0 for

    H(c) = sum_j (r c - y_j)_+  -  sum_j (y_j - c)_+        (mass lifted at the floor - mass shaved at the cap)

H is continuous, nondecreasing and piecewise linear, and on (0, max y) its slope is at least 1 (the largest
entry is always capped there), which gives the solver its accuracy handle: |c - c*| <= |H(c)|. The root is
found by bracketed scalar root-solving on H itself -- deliberately NOT by the active-set closed form the
production diagnostics use, so agreement between the two is evidence about both.

STORED INPUTS, NOT INTENDED TARGETS. The reference solves the problem for the numbers it is handed: Python
floats are converted losslessly (every binary64 is an exact rational), alpha included, and r is evaluated from
that exact alpha at the working precision. It is the mass-preserving projection -- sum p* = sum y, whatever
sum y is -- and therefore scale-covariant: the projection of s * y is s times the projection of y. So a stored
row whose sum is 1 + O(u) has a residual (1 + O(u)) times its renormalized self's, an O(u) RELATIVE difference
and never the absolute O(u) a sum-to-1 constraint would inject (which at high alpha is orders above the
residual itself). Where a test is about the INTENDED target instead -- 1/K exactly rather than fl(1/K), r = 1/2
exactly rather than exp(-2 fl(ln 2 / 2)) -- it passes mpmath numbers, which are taken as they are.

PRECISION IS ADAPTIVE. Resolving an exp(-2 alpha)-sized perturbation of an order-one probability needs about
2 alpha / ln 10 decimal digits before the first significant digit of the residual appears (130 at alpha 150:
a fixed 120 digits returns p*_+ = 1 exactly and the residual 0, reproducing the very failure the reference
exists to catch). The working precision is that many digits, plus the significant digits asked for, plus
guard digits. That is a starting estimate only -- r does not bound how small the ANSWER is: a row one float
outside feasibility has residuals ~1e-17 and a divergence second order in them, ~1e-34, at any r -- so the
solve is REPEATED at more than double the precision until two consecutive solves agree to SIG_DIGITS
significant digits on every R_j and on E_ir, not merely on p*. Nothing is returned on the strength of one
precision alone, and a row that never settles raises.
"""
import math

import mpmath as mp

SIG_DIGITS = 30    # significant digits the reference is required to agree with itself to
GUARD_DIGITS = 25

# The float64 regime the diagnostics are expected to resolve, stated conservatively: beyond it r = exp(-2 alpha)
# or the floor r c >= r / B leaves the NORMAL float64 range (r itself denormalizes at alpha ~ 354 and underflows
# at ~ 372; the floor sooner, by ln(B) / 2), and a float64 result there carries few or no significant bits --
# the correctly rounded value of a residual of 1e-348 is 0.0, which says nothing. 340 keeps r / B normal up to
# B = 2^20.
ALPHA_SUPPORTED_MAX = 340.0


def _solve(y, alpha, r, dps):
    """One solve at `dps` digits -> {feasible, c, P, R, E_ir}, mpmath numbers. Exactly one of alpha / r given."""
    with mp.workdps(dps):
        Y = [mp.mpf(v) for v in y]  # lossless for floats; mpmath inputs pass through
        r = mp.exp(-2 * mp.mpf(alpha)) if r is None else mp.mpf(r)
        y_max, y_min, mass = max(Y), min(Y), mp.fsum(Y)
        if y_max * r <= y_min:  # already inside the band: every c in [y_max, y_min / r] leaves y untouched
            return {"feasible": True, "c": None, "P": Y, "R": [mp.mpf(0)] * len(Y), "E_ir": mp.mpf(0)}

        def H(c):
            return mp.fsum(r * c - v for v in Y if v < r * c) - mp.fsum(v - c for v in Y if v > c)

        # |c - c*| <= |H(c)| (slope >= 1), so this pins c -- and with it every cap residual c - y_j -- far below
        # the smallest floor residual r c >= r * mass / B; floor residuals inherit r times that
        tol = mp.mpf(10) ** -(SIG_DIGITS + 5) * r * mass / len(Y)
        a, b = mp.mpf(0), y_max  # H(0) = -mass < 0 < H(y_max): an infeasible row has an entry under r * y_max
        fa, fb = H(a), H(b)
        side = 0
        for _ in range(200 + 20 * dps):  # Illinois: bracketed, and exact once a and b share a linear piece
            c = (a * fb - b * fa) / (fb - fa)
            fc = H(c)
            if abs(fc) <= tol:
                break
            if (fc > 0) == (fb > 0):
                b, fb = c, fc
                if side == -1:
                    fa /= 2
                side = -1
            else:
                a, fa = c, fc
                if side == 1:
                    fb /= 2
                side = 1
        else:
            raise RuntimeError(f"reference root-solve did not converge at {dps} digits")
        P = [min(max(v, r * c), c) for v in Y]
        R = [p - v for p, v in zip(P, Y)]
        E_ir = mp.fsum(v * mp.log(v / p) for v, p in zip(Y, P) if v > 0)
        return {"feasible": False, "c": c, "P": P, "R": R, "E_ir": E_ir}


def _agree(u, v):
    """u, v agree to SIG_DIGITS significant digits (an exact zero only with an exact zero)."""
    return abs(u - v) <= mp.mpf(10) ** -SIG_DIGITS * max(abs(u), abs(v))


def reference(y, alpha=None, r=None):
    """
    The projection of the row `y` at logit scale `alpha` (or at an exact ratio bound `r`, for cases stated in
    r): {feasible, c, P, R, E_ir, dps} as mpmath numbers, solved at an adaptive working precision and again at
    more than double it, the two required to agree on every R_j and on E_ir (module docstring). Compare against
    these as mpmath numbers, under mp.workdps(result["dps"]) -- a float() of an exp(-2 alpha)-sized value
    underflows to 0.0 from alpha ~ 372.
    """
    assert (alpha is None) != (r is None)
    # digits to reach r's own magnitude (2 alpha / ln 10 -- taken from alpha, a float exp(-2 alpha) underflowing
    # long before mpmath does), then SIG_DIGITS of an r-sized residual, then guard digits
    magnitude = 2 * float(alpha) / math.log(10) if r is None else -math.log10(float(r))
    dps = int(math.ceil(max(magnitude, 0.0))) + SIG_DIGITS + GUARD_DIGITS
    # that is only a STARTING estimate: how many digits the answer needs depends on how small it turns out to be,
    # which r does not bound -- a row one float outside feasibility has residuals ~1e-17 and a divergence that is
    # second order in them, ~1e-34, whatever r is. So the precision escalates until two solves agree
    lo = _solve(y, alpha, r, dps)
    for _ in range(6):
        dps = 2 * dps + 50
        hi = _solve(y, alpha, r, dps)
        with mp.workdps(dps):
            assert lo["feasible"] == hi["feasible"], "feasibility changed with the precision"
            if all(_agree(u, v) for u, v in zip(lo["R"], hi["R"])) and _agree(lo["E_ir"], hi["E_ir"]):
                return {**hi, "dps": dps}
        lo = hi
    raise RuntimeError(f"reference did not agree with itself by {dps} digits")


def _normalized(y):
    """A row renormalized in float64, as infonce_batch_stats hands rows to the solve."""
    s = math.fsum(y)
    return [v / s for v in y]


def hard_row(K, B):
    """A hard binary row as production stores it: K positives at fl(1 / K), B - K exact zeros, renormalized."""
    return _normalized([1.0 / K] * K + [0.0] * (B - K))


def corpus():
    """
    The rows the diagnostics are checked against, as (name, y, {"alpha": .} or {"r": .}) -- y floats unless a
    case is stated about exact values. Grouped by what each is there to break.
    """
    d = 2.0 ** -53
    cases = []
    # hard binary rows across scale and class ratio, to the edge of the supported regime
    for K, B in ((1, 8), (3, 8), (4, 16), (7, 16)):
        for alpha in (0.5, 3.0, 15.0, 25.0, 55.0, 100.0, 150.0, 200.0, 300.0):
            cases.append((f"hard K={K} B={B} alpha={alpha}", hard_row(K, B), {"alpha": alpha}))
    # graded rows holding zeros, where p* - y fails at scale (noise floor or exact zero)
    for alpha in (3.0, 25.0, 55.0):
        cases.append((f"graded [2/3,1/3,0] alpha={alpha}", _normalized([1.0, 0.5, 0.0]), {"alpha": alpha}))
        cases.append((f"graded [.6,.4,0,0] alpha={alpha}", [0.6, 0.4, 0.0, 0.0], {"alpha": alpha}))
    # tax-like: a few repeated membership levels, row-normalized (the linear tsm)
    tax = _normalized([1.0] * 2 + [0.75] * 3 + [0.25] * 5 + [0.0] * 6)
    cases += [(f"tax-like repeated levels alpha={alpha}", tax, {"alpha": alpha}) for alpha in (1.0, 3.0, 25.0)]
    # feasible, full support: the residual is exactly zero, not merely small
    cases += [(f"feasible [.6,.3,.1] alpha={alpha}", [0.6, 0.3, 0.1], {"alpha": alpha}) for alpha in (1.0, 5.0, 20.0)]
    cases.append(("feasible uniform", [0.25] * 4, {"alpha": 2.0}))
    # a floating-point membership test accepts the wrong label here: N_2 = (y3 - y2) - y2 r rounds to exactly 0
    # in float64 and is -3.25e-18 in fact (entry 2 is capped)
    cases.append(("false acceptance", [0.0, 0.3493857146045568, 0.6506142853954432], {"alpha": 0.07415358169427173}))
    # exact residuals [-d, d, 0], yet E_ir is second order (3 d^2): a first-order KL sum loses it to cancellation
    cases.append(("second-order KL", [0.5 + d, 0.25 - d, 0.25], {"r": 0.5}))
    # both positives capped although max / min = 1.44 > 1 + f r / k: the cap-spread bound is 1 + f r
    cases.append(("cap spread", [0.0, 0.41, 0.59], {"r": 0.5}))
    # an entry EXACTLY on the cap (dyadic inputs, r = 1/2: c = 0.75 / 1.5 = 0.5), and one float either side
    cases.append(("tie at the cap", [0.0, 0.5, 0.75], {"r": 0.5}))
    cases.append(("one step under the cap", [0.0, math.nextafter(0.5, 0.0), 0.75], {"r": 0.5}))
    cases.append(("one step over the cap", [0.0, math.nextafter(0.5, 1.0), 0.75], {"r": 0.5}))
    # an entry EXACTLY on the floor (same c = 0.5, floor r c = 0.25), and one float either side
    cases.append(("tie at the floor", [0.0, 0.25, 0.75], {"r": 0.5}))
    cases.append(("one step under the floor", [0.0, math.nextafter(0.25, 0.0), 0.75], {"r": 0.5}))
    cases.append(("one step over the floor", [0.0, math.nextafter(0.25, 1.0), 0.75], {"r": 0.5}))
    # one float either side of FEASIBILITY (y_max r = y_min exactly at [0.25, 0.5])
    cases.append(("one step inside feasibility", [math.nextafter(0.25, 1.0), 0.5], {"r": 0.5}))
    cases.append(("one step outside feasibility", [math.nextafter(0.25, 0.0), 0.5], {"r": 0.5}))
    # past the supported regime: the true residual is below float64's range altogether
    cases.append(("r underflows", hard_row(1, 2), {"alpha": 400.0}))
    return cases

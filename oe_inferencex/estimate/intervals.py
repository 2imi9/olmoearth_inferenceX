"""The design-based intervals: Wilson, t, stratified, cluster, the design effect, and a union of exact intervals."""
import numpy as np

from oe_inferencex.common.hypergeom import _hypergeom_fractions
from oe_inferencex.estimate.designs import MIN_PER_STRATUM, M_PER_TILE

Z95 = 1.959963984540054


# ----------------------------------------------------------------------------- intervals
def wilson_interval(k, n, N=None):
    """Wilson score interval for k of n, with a finite-population correction when N is given.

    With a finite population the score test's variance is theta (1 - theta) (N - n) / ((N - 1) n), which is
    theta (1 - theta) / n_eff at the effective size n_eff = n (N - 1) / (N - n); inverting that test is Wilson's
    interval at (p n_eff, n_eff), the Korn and Graubard (1998) form the package's stratified interval already uses.
    Two earlier forms are on the record. Until 2026-09-22 the correction scaled the whole half-width, so at k = 0
    the lower bound sat above zero and a sample with no errors ruled out a perfect map. Until 2026-09-23 it
    multiplied only the p (1 - p) / n term inside the root and left the centre and the z^2 / (4 n^2) term at n: at
    k = 0 that reaches zero, but away from it the interval is narrower than the centre's pull toward one half
    allows, and a class holding one error in a hundred windows, seen once in thirty labels, was excluded (k = 1 of
    30 from 100 gave [0.0121, 0.161] against a truth of 0.010); its exact coverage at (100, 1, 30) was 0.70, and
    exp81's record read those cells as Wilson's discreteness. The score inversion covers them (1.00 there), and
    at the record's budgets the two forms differ by at most 1.7e-4 at 300 of 22,598 (`tests/test_estimate.py`).
    A census has no sampling error: at n >= N the interval is (p, p).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"wilson_interval needs 0 <= k <= n, got k={k}, n={n}")
    if N is not None and n > N:
        raise ValueError(f"a sample of {n} cannot come from a population of {N}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    if N and n >= N:
        return p, p
    n_eff = n * (N - 1) / (N - n) if N and N > 1 else float(n)
    d = 1 + Z95 ** 2 / n_eff
    centre = (p + Z95 ** 2 / (2 * n_eff)) / d
    half = Z95 * np.sqrt(p * (1 - p) / n_eff + Z95 ** 2 / (4 * n_eff ** 2)) / d
    lo, hi = centre - half, centre + half
    return (0.0 if lo < 1e-12 else lo), (1.0 if hi > 1 - 1e-12 else hi)


_T975 = {1: 12.7062047, 2: 4.3026527, 3: 3.1824463, 4: 2.7764451, 5: 2.5705818, 6: 2.4469119, 7: 2.3646243,
         8: 2.3060041, 9: 2.2621572, 10: 2.2281389}


def t_quantile_975(df):
    """The 0.975 quantile of Student's t: exact for df <= 10, the Abramowitz-Stegun 26.7.5 expansion above that
    (within 1e-5 of scipy for df > 10, tests/test_estimate.py). Numpy-free, so the package stays numpy only."""
    df = int(df)
    if df < 1:
        raise ValueError(f"degrees of freedom must be at least 1, got {df}")
    if df in _T975:
        return _T975[df]
    z = Z95
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def exact_coverage_srs(N, K, B, interval=None):
    """The exact coverage of an interval (`wilson_interval` by default) for a simple random sample of B from N units
    holding K errors: the hypergeometric probability of each error count k, summed over the k whose interval
    contains K/N. It says what the interval achieves at (N, K, B) and whether a Monte Carlo agrees with it. It is a
    diagnostic, not an exemption: a cell below a coverage bar is the interval's shortfall whether or not the
    enumeration predicts it (exp81's sixth amendment; a rule that passed cells matching their own exact coverage
    passed every deterministic defect, and hid one for a night)."""
    import math
    interval = interval or wilson_interval
    theta = K / N
    lc = lambda n, r: math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
    tot = 0.0
    for k in range(max(0, B - (N - K)), min(B, K) + 1):
        lo, hi = interval(k, B, N)
        if lo <= theta <= hi:
            tot += math.exp(lc(K, k) + lc(N - K, B - k) - lc(N, B))
    return tot


def stratified_mean_and_variance(err, strata, picked, sizes, N):
    """The stratified estimate of the population rate and the unbiased estimate of its variance,
    sum over strata of W_h^2 (1 - n_h/N_h) p_h (1 - p_h) / (n_h - 1), which is Cochran's (1 - f_h) s_h^2 / n_h.
    A stratum with fewer than two sampled units contributes its single unit to the estimate and nothing to the
    variance; the count of such strata is returned. Checked by enumeration in tests/test_estimate_exact.py."""
    err, strata, picked = np.asarray(err, dtype=np.float64), np.asarray(strata), np.asarray(picked)
    est = var = 0.0
    starved = 0
    for h, Nh in enumerate(sizes):
        m = strata[picked] == h
        nh = int(m.sum())
        Wh = Nh / N
        if Nh == 0:
            continue                                            # an empty stratum contributes nothing, legitimately
        if nh < MIN_PER_STRATUM:
            starved += 1
            if nh == 1:
                est += Wh * float(err[picked][m].mean())
            continue
        ph = float(err[picked][m].mean())
        est += Wh * ph
        var += Wh ** 2 * (1 - nh / Nh) * ph * (1 - ph) / (nh - 1)
    # a sum of weights N_h / N can round to 1.0000000000000002, and the interval then refused a sample with every
    # label wrong (review of 2026-09-23); the estimate is a proportion
    return min(max(est, 0.0), 1.0), max(var, 0.0), starved


def stratified_interval(err, strata, picked, sizes, N):
    """Stratified mean with a Wald interval, clipped to [0, 1]; the count of starved strata beside it.

    This is the form exp78 graded and is kept for that record. It collapses to zero width when every sampled
    stratum is pure (no error observed, or every unit wrong), which a clean map makes likely; the package's
    user-facing path uses `stratified_interval_wilson` instead."""
    est, var, starved = stratified_mean_and_variance(err, strata, picked, sizes, N)
    half = Z95 * np.sqrt(var)
    return est, max(0.0, est - half), min(1.0, est + half), starved


def stratified_interval_wilson(err, strata, picked, sizes, N):
    """The stratified estimate with a Wilson interval on its effective sample size (Korn and Graubard, 1998).

    The design-based variance v of the stratified mean is turned into the sample size a simple random sample
    would need for that variance, n_eff = p(1 - p)/v, and Wilson's interval is taken at (p, n_eff). Away from 0
    and 1 this agrees with the Wald form to the third decimal on exp78's tasks; at p = 0 or 1, where v is zero
    and Wald collapses to a point, n_eff is taken as the labelled count itself, which is the simple-random bound
    the design cannot improve on without an observed error. No finite-population correction is applied a second
    time: the strata's (1 - f_h) are already inside v. Returns (estimate, low, high, starved strata, n_eff)."""
    est, var, starved = stratified_mean_and_variance(err, strata, picked, sizes, N)
    n = int(np.asarray(picked).size)
    strata, picked = np.asarray(strata), np.asarray(picked)
    if all(Nh == 0 or int((strata[picked] == h).sum()) >= Nh for h, Nh in enumerate(sizes)):
        return est, est, est, starved, float(n)                  # a census has no sampling error
    if var > 0 and 0 < est < 1:
        n_eff = est * (1 - est) / var
    else:
        n_eff = float(n)
    lo, hi = wilson_interval(est * n_eff, n_eff)
    return est, lo, hi, starved, n_eff


def cluster_interval(err, tile, picked, tile_sizes=None, quantile="t"):
    """The error rate from labels taken tile by tile: the ratio estimator with its ultimate-cluster variance.

    theta = sum_i n_i ybar_i / sum_i n_i over the sampled tiles, where n_i is the number of population windows in
    tile i and ybar_i the error rate of its labelled windows, and
    v = sum_i n_i^2 (ybar_i - theta)^2 / (t (t - 1) nbar^2), the with-replacement form, so the second stage's
    variance is carried and no finite-population correction is applied to the first.

    Until 2026-09-22 this was the unweighted mean of tile means, which targets the average TILE's rate. On tiles
    of equal size the two are identical, and so they were on six of exp78's seven tasks. On MADOS, whose tiles
    hold 1 to 400 valid windows, the mean of tile means was 1.78 times the true rate, and its interval covered
    only because one-window tiles, whose means are 0 or 1, inflated it.

    `tile_sizes` maps a tile id to its number of population windows (a dict, or an array indexed by id); by
    default it is counted from `tile`, which is then taken to be the population. `quantile` is "t" (on t - 1
    degrees of freedom, the default) or "normal" (what exp78 graded). Returns (estimate, low, high, realised
    design effect, variance): the design effect is v over the simple-random variance of the same labels.
    """
    err, tile, picked = np.asarray(err, dtype=np.float64), np.asarray(tile), np.asarray(picked)
    t, e = tile[picked], err[picked]
    ids = np.unique(t)
    k = ids.size
    if k < 2:
        raise ValueError(f"{k} tile(s) in the sample: a between-tile standard error needs at least two")
    if tile_sizes is None:
        u, c = np.unique(tile, return_counts=True)
        tile_sizes = dict(zip(u.tolist(), c.tolist()))
    n_i = np.array([tile_sizes[u] if isinstance(tile_sizes, dict) else tile_sizes[int(u)] for u in ids.tolist()], float)
    ybar = np.array([e[t == u].mean() for u in ids])
    est = float((n_i * ybar).sum() / n_i.sum())
    var = float(((n_i * (ybar - est)) ** 2).sum() / (k * (k - 1)) / n_i.mean() ** 2)
    q = Z95 if quantile == "normal" else t_quantile_975(k - 1)
    half = q * np.sqrt(var)
    srs = est * (1 - est) / e.size
    deff = var / srs if srs > 0 else float("nan")
    return est, max(0.0, est - half), min(1.0, est + half), deff, var


def design_effect(err, tile, m=M_PER_TILE):
    """1 + (m - 1) rho, with rho the one-way intra-cluster correlation of the error indicator over tiles."""
    err, tile = np.asarray(err, dtype=np.float64), np.asarray(tile)
    per = [err[tile == t] for t in np.unique(tile)]
    per = [p for p in per if p.size >= 2]
    if len(per) < 3:
        return float("nan")
    sz = np.array([p.size for p in per], float)
    mu = np.array([p.mean() for p in per])
    n, gm = sz.sum(), float(np.concatenate(per).mean())
    msb = (sz * (mu - gm) ** 2).sum() / (len(per) - 1)
    msw = sum(((p - p.mean()) ** 2).sum() for p in per) / (n - len(per))
    m0 = (n - (sz ** 2).sum() / n) / (len(per) - 1)
    rho = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    return float(1 + (m - 1) * rho)


def _union_interval(k, n, sizes, alpha=0.05):
    """The whole-map estimate and interval of the condition design, from each condition's error count k_c among its
    n_c labels and its size N_c. The estimate is the stratified one, sum_c W_c k_c / n_c with W_c = N_c / N. The
    interval runs from sum_c W_c low_c to sum_c W_c high_c, where (low_c, high_c) is condition c's exact interval
    `hypergeom_interval(k_c, n_c, N_c, 1 - alpha / L)` and L is the number of conditions not labelled in full. A
    condition labelled in full enters at its exact rate. By the union bound the L intervals hold together with
    probability at least 1 - alpha, and when they do the weighted sums bracket the whole-map rate sum_c W_c K_c /
    N_c: coverage at least 1 - alpha for every population and every allocation, with no approximation. With one
    condition it is `hypergeom_interval(k, n, N)`.

    It replaced a stratified Wilson interval with a floor on the variance (review of 2026-09-29). By exact
    enumeration of 101,772 cells that one covered as little as 0.719, and where the estimate came from a small
    condition and the variance from a floored large one it was up to 35 times too wide: [0, 70.4%] at 99/1 of a
    million windows, 300 labels each, errors (0, 1), beside condition intervals of [0, 1.22%] and [0.01%, 1.82%].

    The three sums are taken in exact fractions and each rounded once to the nearest float. Summed in floats, an end
    could land one float past a truth that sits on it. On the 14 small maps of the tests that happened on every
    map, and at some error counts no sample's interval held the truth. Rounding to the nearest float keeps order,
    and the truth K / N is rounded the same way, so the float interval holds every truth the exact one holds; the
    estimate stays between the ends, and a census is a point. With one condition the single term is
    `hypergeom_interval`'s own ends and the sample's share k / n, so the two designs agree to the last bit.
    Returns (estimate, low, high, L)."""
    from fractions import Fraction
    k, n, sizes = [int(x) for x in k], [int(x) for x in n], [int(x) for x in sizes]
    if not len(k) == len(n) == len(sizes) or any(not 0 <= kc <= nc <= Nc for kc, nc, Nc in zip(k, n, sizes)):
        raise ValueError(f"_union_interval needs 0 <= k_c <= n_c <= N_c for each condition, got {k}, {n}, {sizes}")
    N = sum(sizes)
    L = sum(nc < Nc for nc, Nc in zip(n, sizes))
    conf = float(1 - alpha / max(L, 1))
    theta = lo = hi = Fraction(0)
    for kc, nc, Nc in zip(k, n, sizes):
        if Nc == 0:
            continue
        a, b = _hypergeom_fractions(kc, nc, Nc, conf)
        W = Fraction(Nc, N)
        if nc:                                                   # a condition with no label adds 0 to the estimate
            theta += W * Fraction(kc, nc)
        lo += W * a
        hi += W * b
    return float(theta), float(lo), float(hi), L

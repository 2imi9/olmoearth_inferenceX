"""How wrong is the map? A design-based error rate from a labelled sample, and which windows to label.

The label-free layers rank windows for review and never say how wrong a map is, because that needs a reference.
This module is what a reference costs, measured in exp78 on the seven segmentation tasks of Ai2's suite with
every unit labelled so the intervals could be graded (docs/results/comparisons.md, exp78): label 300 windows
drawn by a design and the error rate comes back with an interval that covers the truth on 0.93 to 0.96 of
draws, about +/-3 points on a clean map and +/-5 on a messy one.

Three things a user needs, and one they must be stopped from doing.

- `sample_for_estimation` says which windows to label. The default design stratifies the map by confidence
  margin into quintiles and allocates the budget by Neyman's rule from the model's own confidence, so no label is
  spent estimating stratum rates; on exp78's tasks it narrowed the interval to a median 0.80 of a random sample's
  and to 0.63 on the cleanest map, with coverage intact everywhere. A plain random sample and a tile design are
  also offered, the last because that is how people actually label.
- `estimate_error_rate` turns the labels back into a rate with the interval the design earns: the exact
  hypergeometric interval for a random sample, a stratified interval otherwise, and for tile-sampled
  labels a ratio estimator with its ultimate-cluster interval, beside the naive one so the difference is visible.
  That last is the weakest of the three and says so: it covered 0.60 of the time on MADOS, whose tiles differ in
  size by a factor of 400. Labelling 19 tiles of
  16 windows and using the ordinary formula gave a "95%" interval that covered on 0.51 to 0.78 of draws; that is
  exp78's practical finding and the reason this function will not compute the naive interval alone.
- `estimate_from_indices` is for windows labelled without a design. It treats them as a random sample and first
  checks that they could be one: the mean suspicion percentile of a random sample is 0.5, of the tool's own
  review set about 0.97, and labelling the review set then dividing gives two to six times the true rate on every
  task of exp78's export. A sample more than four standard deviations above a random one is refused with the
  number rather than estimated.

Every function is numpy only. The estimators are the ones exp78 ran; that script imports them from here.
"""
import numpy as np

Z95 = 1.959963984540054
N_STRATA = 5
MIN_PER_STRATUM = 2
M_PER_TILE = 16
REVIEW_SET_SIGMAS = 4.0           # a sample whose mean suspicion percentile sits this many SDs above 0.5 is refused
Q_FLOOR = 0.02                    # the confidence design assumes no stratum is better than 98% right until labelled
MIN_TILES = 5                     # fewer tiles than this and a between-tile standard error is not an estimate
TILES_WARNING = ("labels taken tile by tile are not independent, and a map whose tiles differ in size is labelled "
                 "unevenly; the naive interval beside this one is what the ordinary formula says, and on exp78's "
                 "tasks it covered 51 to 78% of the time while claiming 95%. This interval is better and still not "
                 "honest everywhere: on exp78's tasks it covered 0.91 to 0.94 where tiles were of equal size, 0.82 on "
                 "Sen1Floods11 and 0.60 on MADOS, whose tiles hold 1 to 400 windows. Prefer the confidence design")


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


def _hyper_tail(k, n, N, K, upper):
    """P(X >= k) when `upper`, else P(X <= k), for X hypergeometric: n draws without replacement from N units of
    which K are marked. Summed in log space along the pmf's ratio recurrence, so it neither loops in Python over
    the support nor overflows when k sits far from the mode."""
    import math
    lo, hi = max(0, n - (N - K)), min(n, K)
    if upper:
        if k <= lo:
            return 1.0
        if k > hi:
            return 0.0
        x = np.arange(k, hi, dtype=np.float64)                  # pmf(x + 1) / pmf(x), x = k .. hi - 1
        r = (K - x) * (n - x) / ((x + 1.0) * (N - K - n + x + 1.0))
    else:
        if k >= hi:
            return 1.0
        if k < lo:
            return 0.0
        x = np.arange(k, lo, -1, dtype=np.float64)              # pmf(x - 1) / pmf(x), x = k .. lo + 1
        r = x * (N - K - n + x) / ((K - x + 1.0) * (n - x + 1.0))
    lead = _log_choose(K, k) + _log_choose(N - K, n - k) - _log_choose(N, n)
    if r.size == 0:
        return min(1.0, math.exp(lead))
    cs = np.cumsum(np.log(r))
    m = max(0.0, float(cs.max()))
    return min(1.0, math.exp(lead + m + math.log(math.exp(-m) + float(np.exp(cs - m).sum()))))


def hypergeom_interval(k, n, N, conf=0.95):
    """Exact equal-tailed interval for the share K/N of marked units in a finite population of N, from k marked
    among a simple random sample of n: every K whose two tails at the observed k both exceed (1 - conf)/2, the
    tail inversion of Buonaccorsi (1987) and Wang (2015), the finite-population form of Clopper and Pearson. Its
    coverage is at least `conf` for every (N, K, n) by construction; the trusted zone's tests (`zone_pvalue`) are
    the one-sided form of the same inversion. exp81's sixth amendment adopted it for the user's accuracy under a
    random sample, where the labelled windows of a map class are a simple random sample of that class: on a grid
    of classes with N <= 1000 it never covered below 0.951, where the finite-population Wilson form fell below
    0.93 on 40 of 164 cells. The interval is widened, when needed, to hold the sample share k/n, which near a census
    can fall between two population values. A census is the point; an empty sample is everything."""
    if not 0 < conf < 1:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    counts = []
    for name, x in (("k", k), ("n", n), ("N", N)):
        if isinstance(x, (bool, np.bool_)) or float(x) != int(x):
            raise ValueError(f"hypergeom_interval takes whole counts, got {name}={x!r}")
        counts.append(int(x))
    k, n, N = counts
    if n < 0 or k < 0 or k > n or n > N:
        raise ValueError(f"hypergeom_interval needs 0 <= k <= n <= N, got k={k}, n={n}, N={N}")
    # validated here, cached below on plain ints: a cache in front of the checks let True (equal to 1) through
    lo, hi = _hypergeom_interval(k, n, N, float(conf))
    # The interval is over population values K/N, the estimate the sample share k/n, and near a census the two sit on
    # different grids: 269 of 273 labelled (0.98535) left only 270 of 274 (0.98540) standing, an interval that did not
    # contain the estimate printed beside it (property tests, 2026-09-23). Widening to include k/n keeps the coverage.
    if n:
        lo, hi = min(lo, k / n), max(hi, k / n)
    return lo, hi


@__import__("functools").lru_cache(maxsize=1 << 16)
def _hypergeom_interval(k, n, N, conf):
    if n == 0:
        return 0.0, 1.0
    if n >= N:
        return k / n, k / n
    a = (1.0 - conf) / 2.0
    lo_K, hi_K = k, N - (n - k)                                  # the counts the sample does not rule out outright
    left, right = lo_K, hi_K                                     # smallest K with P(X >= k | K) > a (nondecreasing in K)
    while left < right:
        mid = (left + right) // 2
        if _hyper_tail(k, n, N, mid, upper=True) > a:
            right = mid
        else:
            left = mid + 1
    K_lo = left
    left, right = lo_K, hi_K                                     # largest K with P(X <= k | K) > a (nonincreasing in K)
    while left < right:
        mid = (left + right + 1) // 2
        if _hyper_tail(k, n, N, mid, upper=False) > a:
            left = mid
        else:
            right = mid - 1
    return K_lo / N, left / N


def confidence_strata(margin, n_strata=N_STRATA):
    """Quantile strata of the confidence margin; stratum 0 is the least confident."""
    margin = np.asarray(margin, dtype=np.float64)
    q = np.quantile(margin, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(q, margin, side="right")


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


# ----------------------------------------------------------------------------- designs
def neyman_allocation(sizes, spread, budget, floor=MIN_PER_STRATUM):
    """Allocate a budget to strata in proportion to N_h * spread_h, with a floor per stratum and no stratum
    asked for more units than it has.

    The continuous allocation n_h = B N_h S_h / sum N_h S_h minimises the stratified variance exactly (a Lagrange
    condition, docs/method/protocol.md). This integer version floors it and hands the remainder to the strata
    with the most units left, which is what exp78 ran. Its cost, measured rather than guessed: on tiny cases the
    variance lands 1.3 to 2.3% above the best integer allocation (tests/test_estimate_exact.py); on exp78's
    real strata at the record's budget of 300 it is 0.1 to 0.5% above the continuous optimum (MADOS 0.47%,
    PASTIS S2 0.22%, m-cashew-plant 0.11%) and half that at 1,000. Kept as run so exp78's recorded numbers stay
    the package's output."""
    sizes = np.asarray(sizes, int)
    need = int(np.minimum(sizes, floor).sum())
    if budget < need:
        raise ValueError(
            f"a budget of {budget} cannot give each of the {int((sizes > 0).sum())} non-empty strata its {floor} labels "
            f"(needs {need}). Until 2026-09-22 the allocation zeroed the least confident strata instead, and the "
            "estimate then treated the most error-prone part of the map as error-free")
    w = sizes.astype(float) * np.asarray(spread, float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sizes)) / len(sizes)
    n = np.maximum(floor, np.floor(w * budget).astype(int))
    n = np.minimum(n, sizes)
    while n.sum() > budget:
        n[np.argmax(n)] -= 1
    while n.sum() < budget and (sizes - n).sum() > 0:
        n[np.argmax(sizes - n)] += 1
    return n


def draw_stratified(rng, strata, sizes, allocation):
    """A without-replacement draw of `allocation[h]` units from each stratum."""
    idx = []
    for h, nh in enumerate(allocation):
        pool = np.flatnonzero(strata == h)
        idx.append(rng.choice(pool, min(int(nh), pool.size), replace=False))
    return np.concatenate(idx)


def sample_for_estimation(margin, budget, design="confidence", p1=None, tiles=None, per_tile=M_PER_TILE,
                          n_strata=N_STRATA, valid=None, seed=0):
    """Which windows to label so that `estimate_error_rate` can give an honest rate afterwards.

    margin  : per-window confidence margin, higher = more confident (assess's `arrays["confidence"]`, flattened)
    budget  : number of windows to label
    design  : "confidence" (default) stratifies by margin quintile and allocates by Neyman's rule from the model's
              own top-1 probability `p1`, which it needs; "proportional" stratifies and allocates by size;
              "random" is a simple random sample; "tiles" labels `per_tile` windows in each of budget // per_tile
              tiles drawn at random, which needs `tiles`, the tile id of every window
    valid   : optional mask of windows that exist; invalid windows are never sampled and never counted
    Returns the sample: its `indices` into the flattened window grid, and everything the estimator needs.
    """
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    if valid.size != margin.size:
        raise ValueError(f"valid has {valid.size} entries for {margin.size} windows")
    # a window with no finite margin is not in the population: assess's confidence is NaN at no-data, and until
    # 2026-09-23 those windows were sampled (38 of 100 on a map with 38% no-data), as review_set_check and
    # zone_order already excluded them
    valid = valid & np.isfinite(margin)
    pop = np.flatnonzero(valid)
    N = int(pop.size)
    if N == 0:
        raise ValueError("no valid windows to sample")
    budget = int(budget)
    if not 0 < budget <= N:
        raise ValueError(f"budget must be between 1 and the {N} valid windows, got {budget}")
    rng = np.random.default_rng(seed)
    out = {"design": design, "budget": budget, "n_population": N, "seed": seed}
    if design == "random":
        out["indices"] = pop[rng.choice(N, budget, replace=False)]
    elif design in ("confidence", "proportional"):
        s = confidence_strata(margin[pop], n_strata)
        sizes = np.bincount(s, minlength=n_strata)
        if int((sizes > 0).sum()) < 2:
            out["note"] = ("the confidence margin has too few distinct values to stratify (a hard mask, a constant "
                           "map); every window is in one stratum and this is a random sample in effect")
        if design == "confidence":
            if p1 is None:
                raise ValueError('design "confidence" needs p1, the top-1 probability per window; pass it, or use '
                                 'design="proportional"')
            p1 = np.asarray(p1, dtype=np.float64).ravel()
            if p1.size != margin.size:
                raise ValueError(f"p1 has {p1.size} entries for {margin.size} windows")
            p1 = p1[pop]
            if not np.isfinite(p1).all():
                # max(Q_FLOOR, nan) is Q_FLOOR in Python, so one NaN silently set its stratum's assumed error rate
                raise ValueError(f"p1 is not finite at {int((~np.isfinite(p1)).sum())} valid window(s)")
            # The model's own confidence stands in for the stratum's error rate, and it is overconfident exactly
            # where its errors are confident: a stratum whose top-1 probability is exactly 1.0 (float32 saturation,
            # ordinary in real maps) would be allocated the floor of two labels however large it is, and any error
            # in it then went unseen — a nominal 95% interval covered 26% of draws on such a map (audit,
            # 2026-09-22). No stratum is assumed better than 1 - Q_FLOOR right until the labels say so.
            q = np.array([max(Q_FLOOR, (1 - p1[s == h]).mean()) if (s == h).any() else 0.0 for h in range(n_strata)])
            spread = np.sqrt(np.clip(q * (1 - q), 1e-9, None))
        else:
            spread = np.ones(n_strata)
        alloc = neyman_allocation(sizes, spread, budget)
        local = draw_stratified(rng, s, sizes, alloc)
        out.update({"indices": pop[local], "strata": s, "strata_of_population": pop, "sizes": sizes.tolist(),
                    "allocation": alloc.tolist(), "n_strata": int(n_strata)})
    elif design == "tiles":
        if tiles is None:
            raise ValueError('design "tiles" needs `tiles`, the tile id of every window')
        tiles = np.asarray(tiles).ravel()
        if tiles.size != margin.size:
            raise ValueError(f"tiles has {tiles.size} entries for {margin.size} windows")
        # Tiles in a random order, up to `per_tile` windows from each, until the budget is met. A fixed count of
        # budget // per_tile tiles (exp78's D4) falls short wherever tiles hold fewer valid windows than per_tile:
        # on MADOS, whose tiles hold a median of six, 18 tiles gave 133 labels for a budget of 300.
        # grouped by one stable sort, O(N log N): a mask per tile is O(tiles x windows), minutes on a 4M-window map
        tp = tiles[pop]
        o = np.argsort(tp, kind="stable")
        u, starts, counts = np.unique(tp[o], return_index=True, return_counts=True)
        by = {t: pop[o[a:a + c]] for t, a, c in zip(u.tolist(), starts, counts)}
        order = rng.permutation(np.array(list(by)))
        chosen, picked, total = [], [], 0
        for t in order:
            if total >= budget:
                break
            take = min(per_tile, by[t].size, budget - total)
            picked.append(rng.choice(by[t], take, replace=False))
            chosen.append(t)
            total += take
        out["indices"] = np.concatenate(picked)
        if len(chosen) < MIN_TILES:
            raise ValueError(
                f"the tile design would use {len(chosen)} tile(s) for this budget; a between-tile standard error needs "
                f"at least {MIN_TILES}. Use smaller tiles or fewer windows per tile")
        if total < budget:
            most = sum(min(per_tile, v.size) for v in by.values())
            raise ValueError(
                f"the tile design can label at most {most} windows here ({len(by)} tiles, up to {per_tile} each), short of "
                f"the budget of {budget}. Use smaller tiles, more windows per tile, or a smaller budget; a sample that "
                "silently falls short would report an interval for a budget nobody labelled")
        out.update({"tiles": tiles, "per_tile": int(per_tile), "n_tiles": int(len(chosen)),
                    # the population size of every tile, over valid windows: the ratio estimator weights by it
                    "tile_ids": [int(t) for t in by], "tile_valid_sizes": [int(v.size) for v in by.values()]})
    else:
        raise ValueError(f'design must be "confidence", "proportional", "random" or "tiles", got {design!r}')
    out["indices"] = np.asarray(out["indices"], int)
    return out


# ----------------------------------------------------------------------------- estimates
def estimate_error_rate(sample, wrong):
    """The error rate of the whole map from the labelled sample, with the interval its design earns.

    sample : what `sample_for_estimation` returned
    wrong  : 0/1 per labelled window, in the order of sample["indices"]: 1 where the label disagrees with the map
    """
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    idx = np.asarray(sample["indices"], int)
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} sampled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once in the sample; each window "
                         "is labelled once, and a repeated one would be counted as extra units of its stratum")
    if (idx < 0).any():
        raise ValueError("a sampled window index is negative; indices point into the flattened window grid")
    N, design = int(sample["n_population"]), sample["design"]
    out = {"design": design, "n_labelled": int(idx.size), "n_population": N, "nominal_coverage": 0.95}
    if design == "random":
        # exact (review of 2026-09-23): the count of wrong windows in a simple random sample is hypergeometric, and
        # the finite-population Wilson form covered 0.79 one window short of a census and 0.92-0.94 at realistic
        # cells with few errors; the tail inversion covers at least 95% on every (N, K, n), as the per-class user's
        # accuracy already does. exp78 and exp79 graded the Wilson form, which stays in the experiments' harness.
        lo, hi = hypergeom_interval(int(wrong.sum()), idx.size, N)
        out.update({"estimate": float(wrong.mean()), "low": lo, "high": hi,
                    "method": "exact hypergeometric interval (simple random sample of a finite map)"})
    elif design in ("confidence", "proportional"):
        # rebuild the per-unit arrays over the population so the stratified estimator sees the sizes it was drawn with
        pop = np.asarray(sample["strata_of_population"], int)
        pos = {int(g): i for i, g in enumerate(pop)}
        stray = [int(g) for g in idx if int(g) not in pos]
        if stray:
            raise ValueError(f"{len(stray)} sampled window(s), first {stray[0]}, are not in the population this sample "
                             "was drawn from (an invalid window, or a sample built against another mask)")
        local = np.array([pos[int(g)] for g in idx])
        err = np.zeros(pop.size)
        err[local] = wrong
        est, lo, hi, starved, n_eff = stratified_interval_wilson(err, np.asarray(sample["strata"]), local, sample["sizes"], N)
        out.update({"estimate": est, "low": lo, "high": hi, "starved_strata": int(starved), "effective_n": n_eff,
                    "method": "stratified by confidence margin; Wilson interval on the design's effective sample size"})
        notes = []
        if est in (0.0, 1.0):
            notes.append(f"{'no' if est == 0 else 'every'} labelled window was wrong, so the design's variance is zero and "
                         f"the interval is the simple-random Wilson bound at {idx.size} labels; the stratification "
                         "cannot narrow it without an observed error")
        if starved:
            notes.append(f"{starved} of {sample['n_strata']} strata had fewer than {MIN_PER_STRATUM} labelled windows "
                         "and contribute no variance; the interval is narrower than it should be")
        if notes:
            out["warning"] = "; ".join(notes)
    elif design == "tiles":
        tiles = np.asarray(sample["tiles"]).ravel()
        err = np.zeros(tiles.size)
        err[idx] = wrong
        sizes = dict(zip([int(t) for t in sample["tile_ids"]], [int(v) for v in sample["tile_valid_sizes"]]))
        est, lo, hi, deff, var = cluster_interval(err, tiles, idx, tile_sizes=sizes)
        nlo, nhi = wilson_interval(int(wrong.sum()), idx.size, N)
        n_tiles = int(sample["n_tiles"])
        notes = [TILES_WARNING]
        if var == 0:
            # every sampled tile shows the same rate (a clean map: no error in any tile); the between-tile variance
            # is zero and the interval would be a point. Fall back to Wilson on the labelled count, the bound the
            # design cannot improve on without an observed error.
            lo, hi = wilson_interval(float(est) * idx.size, idx.size, N)
            notes.append("every sampled tile shows the same error rate, so the between-tile variance is zero; the "
                         f"interval is the simple-random Wilson bound at {idx.size} labels, which is optimistic if "
                         "errors cluster")
        out.update({"estimate": est, "low": lo, "high": hi,
                    "design_effect": None if not np.isfinite(deff) else deff, "n_tiles": n_tiles,
                    "method": f"ratio estimator over tiles, ultimate-cluster variance, t on {n_tiles - 1} df",
                    "naive_interval_if_treated_as_random": {"low": nlo, "high": nhi},
                    "warning": "; ".join(notes)})
    else:
        raise ValueError(f"unknown design {design!r}")
    out["half_width"] = (out["high"] - out["low"]) / 2
    return out


def review_set_threshold(n, N=None):
    """The MEAN suspicion percentile above which n windows drawn from N are not a random sample: 0.5 plus
    REVIEW_SET_SIGMAS standard deviations of the mean of n uniform percentiles drawn without replacement,
    sqrt(1/12) sqrt((N - n) / ((N - 1) n)); about 0.567 at 300 labels of a large map.

    Until 2026-09-24 the statistic was the median (threshold 0.5 + 4 x 0.6/sqrt(n), 0.64 at 300). Under heavy ties
    the median cannot separate a random sample from the tool's review set: both land inside a large tied block at
    the suspect end. With ties ranked in a random order the percentiles are uniform again, and the mean separates
    them (a random sample 0.50, a review set inside a 60% block about 0.70), which the verification of the second
    review found the median did not. On exp78's units the review set and the confidence design's own sample are
    still refused, and two hundred random draws pass (tests/test_estimate.py)."""
    n = max(int(n), 1)
    fpc = 1.0 if N is None or int(N) <= 1 else max(int(N) - n, 0) / (int(N) - 1)
    return min(0.95, 0.5 + REVIEW_SET_SIGMAS * np.sqrt(1.0 / 12.0) * np.sqrt(fpc / n))


def review_set_check(indices, margin, valid=None):
    """Could these windows be a random sample of the map? The median suspicion percentile of the sample: 0.5 for a
    random draw, near 1 for the tool's own review set, which is built to be enriched for errors."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    # A window with no finite margin is not in the population. assess's confidence array is NaN at no-data, and
    # argsort put NaN at the suspect end, so at 40% no-data the tool's own review set passed as a random sample.
    valid = valid & np.isfinite(margin)
    idx = np.asarray(indices, int).ravel()
    pop = np.flatnonzero(valid)
    # Tied windows are ranked in a fixed random order, independent of raster position, so a random sample's
    # percentiles are uniform whatever the ties. Mid-ranks, used until 2026-09-23, gave a tied block one percentile
    # near its middle, and when half the map or more tied at the suspect end a genuine random sample's median fell
    # inside that block and was refused as a review set (27 of 50 draws at 50% tied, 50 of 50 at 60%).
    s = -margin[pop]
    order = np.lexsort((np.random.default_rng(0).random(pop.size), s))
    ranks = np.empty(pop.size)
    ranks[order] = np.arange(pop.size)
    rank = np.full(margin.size, np.nan)
    rank[pop] = (ranks + 0.5) / pop.size
    on_grid = (idx >= 0) & (idx < margin.size)                   # a negative index used to wrap to the last window
    pct = np.full(idx.size, np.nan)
    pct[on_grid] = rank[idx[on_grid]]
    inside = np.isfinite(pct)
    med = float(np.median(pct[inside])) if inside.any() else float("nan")
    mean = float(np.mean(pct[inside])) if inside.any() else float("nan")
    thr = review_set_threshold(int(inside.sum()), pop.size)
    return {"mean_suspicion_percentile": mean, "median_suspicion_percentile": med, "threshold": thr,
            "looks_like_a_review_set": bool(np.isfinite(mean) and mean > thr),
            "share_in_top_5pct": float(np.mean(pct[inside] > 0.95)) if inside.any() else float("nan"),
            "n_outside_population": int((~inside).sum()), "n_duplicated": int(idx.size - np.unique(idx).size)}


def estimate_from_indices(indices, wrong, margin, valid=None):
    """An error rate from windows labelled without a design, treated as a simple random sample after checking
    that they could be one. The tool's own review set is refused: it is built to hold errors, and labelling it
    then dividing gave two to six times the true rate on every task of exp78's export."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = (np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()) & np.isfinite(margin)
    chk = review_set_check(indices, margin, valid)
    if chk["n_outside_population"]:
        raise ValueError(f"{chk['n_outside_population']} of these windows are outside the valid map (no-data, or no "
                         "finite confidence); their labels say nothing about the map's error rate")
    if chk["n_duplicated"]:
        raise ValueError(f"{chk['n_duplicated']} window(s) appear more than once; a repeated label is not a new one")
    if chk["looks_like_a_review_set"]:
        raise ValueError(
            f"these {len(indices)} windows sit at a mean suspicion percentile of {chk['mean_suspicion_percentile']:.2f}, "
            f"above {chk['threshold']:.2f}, the most a random sample of that size reaches (a random sample sits at 0.50, "
            f"the review set near 0.97); {100 * chk['share_in_top_5pct']:.0f}% are in the top 5% most suspect. They are "
            "an enriched set, not a sample, and the rate they give is inflated. Draw a sample with "
            "sample_for_estimation, or pass a design.")
    sample = {"design": "random", "indices": np.asarray(indices, int), "n_population": int(valid.sum()), "budget": len(indices)}
    out = estimate_error_rate(sample, wrong)
    out["review_set_check"] = chk
    return out


# ----------------------------------------------------------------------------- model-assisted estimation (exp85)
# The map's own confidence on every unlabelled window as a predictor of error, corrected by the labels: the
# survey-sampling difference estimator, which prediction-powered inference rediscovered (Angelopoulos, Duchi and
# Zrnic 2023; Mozer et al. 2026). With the coefficient tuned on the sample the interval cannot be wider than the
# classical one beyond tuning noise. Preregistered in docs/plan/model_assisted_estimation.md.
MIN_FOR_TUNING = 30               # fewer labels keep lambda = 0, the classical estimate: exp85's run tuned at 3 and the
                                  # per-stratum coefficient reached 66 on a 26-label stratum (audit, 2026-09-23)


def tuned_coefficient(e, g):
    """PPI++'s lambda on the labelled units: the sample covariance of error with the predictor over the
    predictor's variance; 0 when the predictor does not vary among the labelled units."""
    e, g = np.asarray(e, float), np.asarray(g, float)
    if e.size < MIN_FOR_TUNING or g.var(ddof=1) <= 0:
        return 0.0
    return float(np.cov(e, g, ddof=1)[0, 1] / g.var(ddof=1))


def model_assisted_interval(err, g, picked, N, lam=None):
    """Under a simple random sample: `lam * mean(g over the population) + mean(err - lam * g over the sample)`, with
    the Wald interval from the residual variance, (1 - n/N) s^2(err - lam g) / n. The population mean of g is a
    constant, so it adds nothing. lam=None tunes it on the sample; lam=1 is the plain difference estimator; lam=0
    is the classical mean. Returns (estimate, low, high, lambda)."""
    err, g, picked = np.asarray(err, float), np.asarray(g, float), np.asarray(picked, int)
    e, gs = err[picked], g[picked]
    n = int(picked.size)
    lam = tuned_coefficient(e, gs) if lam is None else float(lam)
    resid = e - lam * gs
    est = lam * float(g.mean()) + float(resid.mean())
    var = (1 - n / N) * float(resid.var(ddof=1)) / n if n > 1 else float("nan")
    half = Z95 * np.sqrt(max(var, 0.0)) if np.isfinite(var) else float("nan")
    return float(est), max(0.0, est - half), min(1.0, est + half), lam


def stratified_model_assisted_interval(err, g, strata, picked, sizes, N, lam=None):
    """The stratified form (Fisch et al. 2024): per stratum h, `lam_h * mean(g over stratum h) + mean(err - lam_h g
    over its labels)`, weighted by W_h, with variance sum_h W_h^2 (1 - f_h) s_h^2(err - lam_h g) / n_h; lam_h is
    tuned per stratum (None), or one value for all. A stratum with fewer than MIN_FOR_TUNING labels keeps lam 0.
    Returns (estimate, low, high, {stratum: lambda}, starved strata)."""
    err, g, strata, picked = np.asarray(err, float), np.asarray(g, float), np.asarray(strata), np.asarray(picked, int)
    est = var = 0.0
    lams, starved = {}, 0
    for h, Nh in enumerate(sizes):
        if Nh == 0:
            continue
        m = strata[picked] == h
        nh = int(m.sum())
        Wh = Nh / N
        if nh < MIN_PER_STRATUM:
            starved += 1
            if nh == 1:
                est += Wh * float(err[picked][m].mean())
            continue
        e, gs = err[picked][m], g[picked][m]
        lh = tuned_coefficient(e, gs) if lam is None else float(lam)
        lams[h] = lh
        resid = e - lh * gs
        est += Wh * (lh * float(g[strata == h].mean()) + float(resid.mean()))
        var += Wh ** 2 * (1 - nh / Nh) * float(resid.var(ddof=1)) / nh
    half = Z95 * np.sqrt(max(var, 0.0))
    return float(est), max(0.0, est - half), min(1.0, est + half), lams, starved


# ----------------------------------------------------------------------------- per-class accuracy (exp81)
# What the map-accuracy literature says a producer owes (Olofsson et al. 2014; Stehman and Foody 2019; the CEOS
# LPV land-cover protocol 2025): not one error rate but, per class, the user's accuracy (of the windows the map
# calls c, how many are c), the producer's accuracy (of the windows that are c, how many the map found) and the
# error-adjusted share of the map that is c, each with a standard error. All of it comes from the same labelled
# sample `sample_for_estimation` draws; the map's own decisions give the class of every window. Preregistered in
# docs/plan/per_class_assessment.md.
MIN_PER_CLASS = 30                # a per-class interval on fewer labelled windows is reported with a warning
RARE_ERRORS = 5                   # a normal-theory per-class interval resting on 1 to 4 sampled errors is warned (review of
                                  # 2026-09-23: exp81's rare-error cells fail on the draws that catch one or two such errors)


def _wald(est, var):
    half = Z95 * np.sqrt(max(float(var), 0.0))
    return float(est), max(0.0, float(est) - half), min(1.0, float(est) + half)


def _wilson_eff(est, var, n_fallback):
    """Wilson's interval at the effective sample size p(1-p)/var (Korn and Graubard 1998), the form the package
    uses for the stratified overall rate. Where the estimated variance is zero or the estimate sits at 0 or 1,
    which happens whenever no sampled window of a class is wrong, Wald collapses to a point; the effective size
    then falls back to the labelled count of the class, the simple-random bound the design cannot beat without an
    observed error. exp81 graded the Wald form first: on 77 of 324 per-class cells of the classification suite it
    covered as little as 30% of draws, every one of them a class near 0 or 1."""
    est, var = min(max(float(est), 0.0), 1.0), max(float(var), 0.0)      # a weighted sum can round past 1
    if var > 0 and 0 < est < 1:
        n_eff = est * (1 - est) / var
    else:
        n_eff = float(max(int(n_fallback), 1))
    lo, hi = wilson_interval(est * n_eff, n_eff)
    return est, lo, hi


def _interval(est, var, n_fallback, form):
    if form == "wald":
        return _wald(est, var)
    if form == "wilson":
        return _wilson_eff(est, var, n_fallback)
    raise ValueError(f"interval must be 'wilson' or 'wald', got {form!r}")


def _ht_total(values, strata, picked, sizes):
    """Under stratified simple random sampling: the Horvitz-Thompson total of `values` and the unbiased estimate of
    its variance, sum_h N_h ybar_h and sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h. Only picked units are observed."""
    values, strata, picked = np.asarray(values, float), np.asarray(strata), np.asarray(picked)
    tot = var = 0.0
    for h, Nh in enumerate(sizes):
        m = strata[picked] == h
        nh = int(m.sum())
        if Nh == 0 or nh == 0:
            continue
        v = values[picked][m]
        tot += Nh * float(v.mean())
        if nh > 1:
            var += Nh ** 2 * (1 - nh / Nh) * float(v.var(ddof=1)) / nh
    return tot, var


def _ht_ratio(num, den, strata, picked, sizes):
    """A ratio of two Horvitz-Thompson totals with its linearised variance: z_k = (num_k - R den_k) / D_hat, and
    V(R) is the variance of the total of z (Cochran 1977, section 6.11, applied stratum by stratum)."""
    Yn, _ = _ht_total(num, strata, picked, sizes)
    Yd, _ = _ht_total(den, strata, picked, sizes)
    if Yd <= 0:
        return float("nan"), float("nan")
    R = Yn / Yd
    z = (np.asarray(num, float) - R * np.asarray(den, float)) / Yd
    _, var = _ht_total(z, strata, picked, sizes)
    return R, var


def estimate_per_class(sample, reference, map_class, n_classes=None, interval="wilson"):
    """User's accuracy, producer's accuracy and error-adjusted share per class, from the labelled sample.

    sample    : what `sample_for_estimation` returned (designs "random", "confidence" or "proportional")
    reference : the reference class of every labelled window, in the order of sample["indices"], integers >= 0
    map_class : the map's class of EVERY window (the flattened grid `sample` was drawn on); negative at no-data,
                and the count of non-negative entries must equal the population the sample was drawn from

    Under a random sample, the user's accuracy of class c has an exact hypergeometric interval (`hypergeom_interval`)
    on the labelled windows the map calls c, since those are a simple random sample of that class's windows, and
    `interval` does not apply to it; the class shares are post-stratified by
    map class (Olofsson et al. 2014, eq. 4 and 5) and the producer's accuracy follows their eq. 7. Under the
    confidence design every quantity is a ratio of Horvitz-Thompson totals over the margin strata with a
    linearised variance, because a class cuts across strata. Intervals are Wilson on the effective sample size
    (interval="wilson", the default; "wald" is the field's convention and is kept for the record, where exp81
    shows it collapsing to a point on classes with no sampled error). Tile samples are refused: the per-class
    cluster form is not graded yet. A class with fewer than MIN_PER_CLASS labelled windows carries a warning; a map class no
    labelled window fell in leaves the share estimates short by its weight, and that is said."""
    ref = np.asarray(reference).ravel()
    idx = np.asarray(sample["indices"], int).ravel()
    mc = np.asarray(map_class).ravel()
    if ref.size != idx.size:
        raise ValueError(f"{ref.size} reference labels for {idx.size} labelled windows")
    if mc.dtype.kind not in "iu":
        if mc.dtype.kind not in "fb" or not np.isfinite(mc).all() or not np.all(np.mod(mc, 1) == 0):
            raise ValueError("map_class must hold an integer class per window, negative where the map has no data")
    mc = mc.astype(int)
    if ((idx < 0) | (idx >= mc.size)).any():
        raise ValueError(f"a labelled window index lies outside the map's {mc.size} windows")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once in the sample; each window "
                         "is labelled once")
    if ref.dtype.kind not in "iu" and not np.all(np.equal(np.mod(ref, 1), 0)):
        raise ValueError("reference classes must be integers")
    ref = ref.astype(int)
    if (ref < 0).any():
        raise ValueError("reference classes must be >= 0; a window the reviewer could not label should not be in the sample")
    design = sample["design"]
    if design == "tiles":
        raise ValueError("per-class accuracy from a tile sample is not graded yet (exp81 covers random and confidence "
                         "designs); label a random or confidence-designed sample")
    if design not in ("random", "confidence", "proportional"):
        raise ValueError(f"unknown design {design!r}")
    if design == "random":
        pop = np.flatnonzero(mc >= 0)
    else:
        pop = np.asarray(sample["strata_of_population"], int)
        # the population is the sample's, so check that map_class describes it: until 2026-09-23 a class map of
        # another grid or with no-data inside the sampled population passed, or failed with an unrelated error
        if (pop >= mc.size).any() or (mc[pop] < 0).any() or int((mc >= 0).sum()) != pop.size:
            raise ValueError(f"map_class does not describe the population the sample was drawn from ({pop.size} windows, "
                             f"where map_class has {int((mc >= 0).sum())} with a class); pass the class map of the same "
                             "grid and no-data mask the sample was drawn on")
    N = int(pop.size)
    if N != int(sample["n_population"]):
        raise ValueError(f"map_class has {N} valid windows but the sample was drawn from {sample['n_population']}; pass the "
                         "map's class per window on the same grid, negative where the map has no data")
    if (mc[idx] < 0).any():
        raise ValueError("a labelled window sits where the map has no class")
    C = int(n_classes) if n_classes is not None else int(max(int(mc[pop].max()), int(ref.max())) + 1)
    if (ref >= C).any() or (mc[pop] >= C).any():
        raise ValueError(f"a class id reaches or exceeds n_classes={C}")
    m_s, r_s = mc[idx], ref
    conf = np.zeros((C, C), int)                              # rows: map class, columns: reference class
    np.add.at(conf, (m_s, r_s), 1)
    N_map = np.bincount(mc[pop], minlength=C).astype(float)
    W = N_map / N
    per, warnings = {}, []
    if design == "random":
        n_i = conf.sum(1).astype(float)
        u = np.divide(conf, n_i[:, None], out=np.zeros((C, C)), where=n_i[:, None] > 0)   # n_ij / n_i.
        fpc = np.where(N_map > 0, 1 - np.divide(n_i, N_map, out=np.zeros(C), where=N_map > 0), 0.0)
        denom = np.where(n_i > 1, n_i - 1, np.inf)
        v_u = u * (1 - u) * (fpc / denom)[:, None]                                          # V(u_ij) per cell
        unsampled = [int(i) for i in range(C) if N_map[i] > 0 and n_i[i] == 0]
        if unsampled:
            warnings.append(f"map class(es) {unsampled} have no labelled window; the reference shares below are short "
                            f"by their weight ({float(W[unsampled].sum()):.3f} of the map)")
        share_est = (W[:, None] * u).sum(0)                                                 # A_j = sum_i W_i u_ij
        share_var = (W[:, None] ** 2 * v_u).sum(0)
        # the overall accuracy is `estimate`'s quantity with `estimate`'s interval, exact under a random draw: the
        # post-stratified form printed here until 2026-09-23 was never graded and covered 0.913 on one exp81 cell
        k_right = int(np.trace(conf))
        o_lo, o_hi = hypergeom_interval(k_right, int(idx.size), N)
        overall_row = {"estimate": k_right / idx.size, "low": o_lo, "high": o_hi}
        # Olofsson's post-stratified overall accuracy, sum_i W_i u_ii, kept as a point so the table adds up
        post_stratified = min(max(float((W * np.diag(u)).sum()), 0.0), 1.0)
        for c in range(C):
            row = {"map_share": float(W[c]), "n_labelled_map_class": int(n_i[c]), "n_labelled_reference_class": int(conf[:, c].sum())}
            if n_i[c] > 0:
                # exact (sixth amendment of exp81): the labelled windows the map calls c are a simple random sample
                # of that class, so the count of correct ones is hypergeometric and its tail inversion covers at
                # least 95% on every class; the finite-population Wilson form fell to 0.70 on a class with one
                # error in a hundred windows, and the score form to 0.83 on a small grid
                lo, hi = hypergeom_interval(int(conf[c, c]), int(n_i[c]), int(N_map[c]))
                row["user_accuracy"] = {"estimate": float(u[c, c]), "low": lo, "high": hi}
            else:
                row["user_accuracy"] = None
            # producer's accuracy: Olofsson et al. 2014, eq. 7, on the post-stratified counts, with each map
            # class's term carrying the same finite-population correction as the user's accuracy and the shares
            # (exp81's audit found the uncorrected form over-wide: median coverage 0.983 where its siblings sat
            # at 0.95, and eight times too wide on a near-census draw)
            Nhat_j = float((N_map * u[:, c]).sum())
            if Nhat_j > 0 and n_i[c] > 0:
                pa = N_map[c] * u[c, c] / Nhat_j
                t1 = N_map[c] ** 2 * (1 - pa) ** 2 * u[c, c] * (1 - u[c, c]) * fpc[c] / (n_i[c] - 1) if n_i[c] > 1 else 0.0
                others = [i for i in range(C) if i != c and n_i[i] > 1]
                t2 = pa ** 2 * sum(N_map[i] ** 2 * u[i, c] * (1 - u[i, c]) * fpc[i] / (n_i[i] - 1) for i in others)
                e, lo, hi = _interval(pa, (t1 + t2) / Nhat_j ** 2, conf[:, c].sum(), interval)
                row["producer_accuracy"] = {"estimate": e, "low": lo, "high": hi}
            elif N_map[c] == 0 and conf[:, c].sum() > 0:
                # the map never predicts a class the labels show exists: its producer's accuracy is exactly 0, the
                # worst there is (Olofsson's eq. 7), not missing; until 2026-09-23 it printed as n/a
                row["producer_accuracy"] = {"estimate": 0.0, "low": 0.0, "high": 0.0}
            else:
                row["producer_accuracy"] = None
            e, lo, hi = _interval(share_est[c], share_var[c], idx.size, interval)
            row["reference_share"] = {"estimate": e, "low": lo, "high": hi}
            per[int(c)] = row
        method = "random sample: exact hypergeometric interval per map class for user's accuracy; shares post-stratified by map class and producer's accuracy by Olofsson et al. 2014 eq. 7"
    else:
        strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
        pos = {int(g): i for i, g in enumerate(pop)}
        local = np.array([pos[int(g)] for g in idx])
        m_pop = mc[pop]
        r_pop = np.full(N, -1, int)
        r_pop[local] = r_s
        right = np.zeros(N)
        right[local] = (m_s == r_s).astype(float)
        if interval == "wilson":                                   # `estimate`'s graded form, census included
            o_est, o_lo, o_hi, _, _ = stratified_interval_wilson(right, strata, local, sizes, N)
        else:
            o_est, o_lo, o_hi = _wald(*stratified_mean_and_variance(right, strata, local, sizes, N)[:2])
        overall_row = {"estimate": o_est, "low": o_lo, "high": o_hi}
        for c in range(C):
            is_map = (m_pop == c).astype(float)
            is_ref = (r_pop == c).astype(float)
            both = is_map * is_ref
            row = {"map_share": float(W[c]), "n_labelled_map_class": int(conf[c].sum()), "n_labelled_reference_class": int(conf[:, c].sum())}
            ua, v = _ht_ratio(both, is_map, strata, local, sizes)
            row["user_accuracy"] = None if not np.isfinite(ua) else dict(zip(("estimate", "low", "high"), _interval(ua, v, conf[c].sum(), interval)))
            pa, v = _ht_ratio(both, is_ref, strata, local, sizes)
            row["producer_accuracy"] = None if not np.isfinite(pa) else dict(zip(("estimate", "low", "high"), _interval(pa, v, conf[:, c].sum(), interval)))
            if N_map[c] == 0 and conf[:, c].sum() > 0:
                row["producer_accuracy"] = {"estimate": 0.0, "low": 0.0, "high": 0.0}      # exactly 0, as above
            tot, v = _ht_total(is_ref, strata, local, sizes)
            row["reference_share"] = dict(zip(("estimate", "low", "high"), _interval(tot / N, v / N ** 2, idx.size, interval)))
            per[int(c)] = row
        method = "stratified by confidence margin: ratios of Horvitz-Thompson totals with linearised variance; shares as Horvitz-Thompson totals"
    for c, row in per.items():
        notes, codes = [], []
        small = [k for k in ("n_labelled_map_class", "n_labelled_reference_class") if row[k] < MIN_PER_CLASS]
        if small:
            codes.append("few labels")
            notes.append(f"fewer than {MIN_PER_CLASS} labelled windows ({row['n_labelled_map_class']} the map calls this "
                         f"class, {row['n_labelled_reference_class']} the reference does); the interval is wide and, "
                         "below about ten, not to be trusted")
        if N_map[c] == 0 and conf[:, c].sum() > 0:
            codes.append("never predicted")
            notes.append("the map never predicts this class, which the labels show exists: its producer's accuracy is "
                         "exactly 0 and its user's accuracy is undefined")
        # the rare-error case (exp81's second audit): a normal-theory interval built on one to four sampled errors of
        # the kind a quantity counts moves by a whole window's weight when one more or one fewer is drawn
        commissions, omissions = int(conf[c].sum() - conf[c, c]), int(conf[:, c].sum() - conf[c, c])
        counted = [("producer's accuracy", omissions), ("share", omissions + commissions)]
        if design != "random":                                     # the random-design user's accuracy is exact
            counted.append(("user's accuracy", commissions))
        rare = [q for q, n_err in counted if 0 < n_err < RARE_ERRORS]
        if rare and N_map[c] > 0:
            codes.append("few errors")
            notes.append(f"the {', '.join(rare)} rest(s) on {min(omissions + commissions, max(omissions, commissions))} to "
                         f"{omissions + commissions} sampled error(s) ({commissions} the map calls this class wrongly, "
                         f"{omissions} of this class it misses); one error more or fewer in the draw moves the estimate "
                         "by a whole window's weight, and the interval can miss it")
        # exp81 measured one case where a nominal 95% interval covers well under 95%: a class nearly all of whose
        # windows are labelled, so the estimate takes a handful of values and a normal interval cannot follow it
        # (0.86-0.92 on five encoders' Togo classes). The thin-strata note below is a caution from the design: the
        # cells exp81 first attributed to it (Brick Kiln, Nandi Landsat) did not meet its criterion, and their
        # shortfall is the rare-error case, a class whose accuracy rests on a handful of errors that a draw misses
        # or catches at a large weight, for which the package does not yet warn (exp81's second audit)
        if N_map[c] > 0 and row["n_labelled_map_class"] >= 0.9 * N_map[c] and row["n_labelled_map_class"] < N_map[c]:
            codes.append("near census")
            notes.append("nearly every window of this class is labelled; apart from the user's accuracy under a random sample, "
                         "which is exact, the intervals are a rough guide, since the estimate can only take a few values "
                         "(exp81: coverage 0.86-0.92 on such classes)")
        if design != "random":
            f_all = idx.size / N
            thin = np.array([sizes[h] > 0 and (strata[local] == h).sum() / sizes[h] < 0.5 * f_all for h in range(len(sizes))])
            in_thin = float((thin[strata[m_pop == c]]).mean()) if (m_pop == c).any() else 0.0
            if in_thin > 0.5:
                codes.append("thin strata")
                notes.append(f"{100 * in_thin:.0f}% of this class sits in confidence strata the design samples at under half the "
                             "overall rate; if its errors are rare there they are often not drawn, and the interval is then "
                             "optimistic")
        if notes:
            row["warning"] = "; ".join(notes)
            row["warning_codes"] = codes
    out = {"design": design, "interval": interval, "n_labelled": int(idx.size), "n_population": N, "n_classes": C, "nominal_coverage": 0.95,
           "overall_accuracy": overall_row, "confusion_counts": conf.tolist(),
           "per_class": per, "method": method}
    if design == "random":
        out["overall_accuracy_post_stratified"] = post_stratified
    if warnings:
        out["warning"] = "; ".join(warnings)
    return out


# ----------------------------------------------------------------------------- a trusted zone with a guarantee (exp80)
# Which part of the map is wrong at most alpha of the time, stated so that the statement itself fails with
# probability at most delta over the reviewer's random draw. Preregistered in docs/plan/trust_zone.md. The zone at
# coverage c is the c most confident share of the valid windows; a simple random sample of the map restricted to
# that zone is a simple random sample of the zone, so every test below is an exact hypergeometric test and nothing
# is approximated.
ZONE_GRID = tuple(round(j / 20, 2) for j in range(1, 21))
ZONE_DELTA = 0.10
ZONE_RULES = ("prefix", "bonferroni", "plugin")


def _log_choose(n, r):
    import math
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def hypergeom_cdf(k, n, K, b):
    """P(X <= k) for X ~ Hypergeom(n, K, b): b windows drawn without replacement from n of which K are wrong.
    Exact, by log-gamma sums, no approximation."""
    import math
    n, K, b, k = int(n), int(K), int(b), int(k)
    if not (0 <= K <= n and 0 <= b <= n):
        raise ValueError(f"hypergeom_cdf needs 0 <= K <= n and 0 <= b <= n, got n={n}, K={K}, b={b}")
    lo, hi = max(0, b - (n - K)), min(b, K)
    if k < lo:
        return 0.0
    if k >= hi:
        return 1.0
    denom = _log_choose(n, b)
    tot = 0.0
    for x in range(lo, k + 1):
        tot += math.exp(_log_choose(K, x) + _log_choose(n - K, b - x) - denom)
    return min(1.0, tot)


def zone_pvalue(k, b, n, alpha):
    """The exact one-sided p-value against 'this zone of n windows is wrong more than alpha of the time', from k
    errors among b of its windows drawn at random. The null count that makes the p-value largest is the smallest
    count that violates alpha, floor(alpha n) + 1, because P(X <= k) falls as the count rises; the p-value is the
    supremum over the null and is valid without approximation. No labels in the zone: p = 1."""
    import math
    n, b, k = int(n), int(b), int(k)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if b == 0:
        return 1.0
    if not 0 <= k <= b <= n:
        raise ValueError(f"zone_pvalue needs 0 <= k <= b <= n, got k={k}, b={b}, n={n}")
    K0 = math.floor(alpha * n) + 1
    if K0 > n:                                  # no count of n can exceed alpha n: the null is empty
        return 0.0
    return hypergeom_cdf(k, n, K0, b)


def zone_upper_bound(k, b, n, delta=ZONE_DELTA):
    """The largest error rate K/n of a zone of n windows that k errors among b sampled do not reject at level
    delta: the exact hypergeometric upper confidence bound. With no labels it is 1."""
    n, b, k = int(n), int(b), int(k)
    if b == 0:
        return 1.0
    lo, hi = k, n - (b - k)                     # K must allow k wrong and b - k right among the sample
    if hypergeom_cdf(k, n, hi, b) > delta:
        return hi / n
    while hi - lo > 1:                          # P(X <= k | K) is nonincreasing in K: bisect
        mid = (lo + hi) // 2
        if hypergeom_cdf(k, n, mid, b) > delta:
            lo = mid
        else:
            hi = mid
    return lo / n


def min_labels_to_certify(alpha, delta=ZONE_DELTA):
    """The fewest labels a zone must hold before it can be certified at (alpha, delta) even with no error among
    them, in the large-population limit: ceil(ln delta / ln(1 - alpha)), from the binomial bound (1 - alpha)^b on
    the p-value. 45 at alpha 0.05, 114 at 0.02, 255 at 0.009, all at delta 0.1. On a zone whose windows are nearly
    all labelled the exact hypergeometric p-value is smaller than the binomial one, so this cut is conservative
    there (exp80's audit: a 15-window zone with all 15 labelled certifies at alpha 0.139 where the limit says 16
    labels are needed); it is exact when the zone is much larger than the sample."""
    import math
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))   # log1p: log(1 - alpha) is 0.0 below alpha ~ 1e-16


def zone_order(margin, valid=None):
    """The valid windows in zone order: descending margin, ties by index, so that the zone at any coverage is a
    fixed set decided before any label is seen. Returns (population indices in order, position of every window
    or -1 outside the population)."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = (np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()) & np.isfinite(margin)
    pop = np.flatnonzero(valid)
    order = pop[np.lexsort((pop, -margin[pop]))]
    pos = np.full(margin.size, -1, dtype=np.int64)
    pos[order] = np.arange(order.size)
    return order, pos


def zone_levels(n_population, budget, alpha, delta=ZONE_DELTA, grid=ZONE_GRID):
    """The grid coverages a budget can certify at (alpha, delta), and the zone size at each: levels below
    min_labels_to_certify / budget are cut before any label is seen, since they cannot be certified whatever
    their quality. Returns (coverages, zone sizes, c_min)."""
    b_min = min_labels_to_certify(alpha, delta)
    c_min = b_min / max(int(budget), 1)
    cov = [c for c in grid if c >= c_min - 1e-12]
    sizes = [max(1, int(round(c * n_population))) for c in cov]
    return cov, sizes, c_min


def zone_counts(positions, wrong, sizes):
    """Per zone size: how many sampled windows fall inside it (b) and how many of those are wrong (k)."""
    positions = np.asarray(positions, dtype=np.int64).ravel()
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    o = np.argsort(positions, kind="stable")
    ps, ws = positions[o], wrong[o]
    cum = np.concatenate([[0.0], np.cumsum(ws)])
    b = np.searchsorted(ps, np.asarray(sizes, dtype=np.int64), side="left")
    return b.astype(int), cum[b].astype(int)


def apply_zone_rule(p, b, k, alpha, delta=ZONE_DELTA, rule="prefix"):
    """Which grid levels a rule accepts, and the largest one; levels are in increasing coverage.
    prefix      accept while p <= delta from the smallest zone up, stop at the first failure (Bates et al. 2021;
                valid when the zone's error rate is nondecreasing in coverage)
    bonferroni  accept every level with p <= delta / J, J the number of levels (Angelopoulos et al. 2021, Learn
                then Test; valid with no assumption on the shape)
    plugin      accept every level whose sample rate k / b is at most alpha; no guarantee, the comparator"""
    p, b, k = np.asarray(p, float), np.asarray(b, int), np.asarray(k, int)
    if rule == "prefix":
        acc = np.zeros(p.size, bool)
        for j in range(p.size):
            if p[j] <= delta:
                acc[j] = True
            else:
                break
    elif rule == "bonferroni":
        acc = p <= delta / max(p.size, 1)
    elif rule == "plugin":
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = (b > 0) & (k <= alpha * b)
    else:
        raise ValueError(f"rule must be one of {ZONE_RULES}, got {rule!r}")
    best = int(np.flatnonzero(acc).max()) if acc.any() else None
    return acc, best


def certify_zone(margin, indices, wrong, alpha, delta=ZONE_DELTA, rule="prefix", grid=ZONE_GRID, valid=None):
    """The largest share of the map, taken from the most confident window down, that is wrong at most `alpha` of
    the time, certified so that the statement fails with probability at most `delta` over the reviewer's draw.

    margin  : per-window confidence, higher = more trusted; NaN and invalid windows are outside the population
    indices : the labelled windows, which must be a simple random sample of the valid windows
    wrong   : 0/1 per labelled window, in the order of `indices`
    rule    : "prefix" (assumes the zone's error rate does not fall as the zone grows; the powerful rule),
              "bonferroni" (no assumption), "plugin" (no guarantee; what a reviewer would do unaided)

    Returns coverage None when nothing can be certified, with the reason; a labelled set that looks like the
    tool's own review set is refused, because the hypergeometric argument needs a random draw."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    idx = np.asarray(indices, int).ravel()
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} labelled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    if rule not in ZONE_RULES:                                    # checked before a budget too small can hide a typo
        raise ValueError(f"rule must be one of {ZONE_RULES}, got {rule!r}")
    chk = review_set_check(idx, margin, valid)
    if chk["n_outside_population"]:
        raise ValueError(f"{chk['n_outside_population']} labelled window(s) are outside the valid map")
    if chk["n_duplicated"]:
        raise ValueError(f"{chk['n_duplicated']} window(s) appear more than once")
    if chk["looks_like_a_review_set"]:
        raise ValueError(f"these {idx.size} windows sit at a mean suspicion percentile of "
                         f"{chk['mean_suspicion_percentile']:.2f}, above {chk['threshold']:.2f}: an enriched set, not a "
                         "random sample, and a zone certified on it would be wrong. Draw the sample at random.")
    order, pos = zone_order(margin, valid)
    N = int(order.size)
    cov, sizes, c_min = zone_levels(N, idx.size, alpha, delta, grid)
    out = {"rule": rule, "alpha": float(alpha), "delta": float(delta), "n_population": N, "n_labelled": int(idx.size),
           "min_labels_to_certify": min_labels_to_certify(alpha, delta), "c_min": float(c_min), "levels": [],
           "coverage": None, "n_zone": None, "threshold": None, "upper_bound": None}
    if not cov:
        out["note"] = (f"{idx.size} labels cannot certify any zone at alpha={alpha:g}, delta={delta:g}: even a zone with "
                       f"no error among its labels needs {out['min_labels_to_certify']} of them, more than the budget")
        return out
    b, k = zone_counts(pos[idx], wrong, sizes)
    p = np.array([zone_pvalue(kk, bb, n, alpha) for kk, bb, n in zip(k, b, sizes)])
    acc, best = apply_zone_rule(p, b, k, alpha, delta, rule)
    for j, c in enumerate(cov):
        out["levels"].append({"coverage": c, "n_zone": sizes[j], "n_labelled_inside": int(b[j]), "n_wrong_inside": int(k[j]),
                              "p_value": float(p[j]), "upper_bound": zone_upper_bound(k[j], b[j], sizes[j], delta),
                              "accepted": bool(acc[j])})
    if best is None:
        out["note"] = (f"no zone certified at alpha={alpha:g}, delta={delta:g} with {idx.size} labels; the smallest testable "
                       f"zone ({cov[0]:.0%} of the map) held {int(b[0])} labels with {int(k[0])} wrong")
        return out
    n_zone = sizes[best]
    thr = float(margin[order[n_zone - 1]])
    # The certified set is the first n_zone windows in zone order, a fixed set; windows tied at the threshold may
    # sit on both sides of the edge (float32 saturation puts hundreds at exactly 1.0 on some maps), so "margin >=
    # threshold" is not the zone there, and the counts say so.
    tied = int((margin[order] == thr).sum())
    inside = int((margin[order[:n_zone]] == thr).sum())
    out.update({"coverage": cov[best], "n_zone": n_zone, "threshold": thr, "n_tied_at_threshold": tied,
                "n_tied_inside_zone": inside, "upper_bound": out["levels"][best]["upper_bound"],
                "zone_indices_in_order": order[:n_zone]})
    if rule == "plugin":
        out["note"] = "plug-in rule: no guarantee; the largest zone whose sample rate is at most alpha"
    elif rule == "prefix":
        out["note"] = ("valid if the zone's error rate does not fall as the zone grows; on the suite tasks exp80 graded, "
                       "the guarantee held whether or not that was exactly true (docs/results/comparisons.md, exp80)")
    else:
        out["note"] = (f"Bonferroni over the {len(cov)} testable levels: valid with no assumption on how the error rate "
                       "changes with the zone")
    if tied > inside:
        out["note"] = out.get("note", "") + (f"; {tied} windows share the threshold margin and only {inside} of them are inside "
                                            "the zone, so the zone is the set returned, not every window at or above the threshold")
    return out

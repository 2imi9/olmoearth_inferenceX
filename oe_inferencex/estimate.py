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
  also offered, the last because that is how people actually label. When the map records each window's input
  condition (a cloud flag, the modalities present), the "condition" design splits the labels equally across the
  conditions, never from the model's confidence. That confidence can overstate the accuracy of a condition read
  from an input combination the model was not trained on: it did on PASTIS for a probe trained on radar plus optical
  and run on radar alone, though not on CropHarvest China 6 (exp88). Even a model trained on each input can be wrong
  more often in one condition than in another. Each condition then gets its own exact interval, and
  `certify_by_condition` a zone of its own.
- `estimate_error_rate` turns the labels back into a rate with the interval the design earns: the exact
  hypergeometric interval for a random sample, the weighted sum of each condition's exact interval under the
  condition design, a stratified interval under the confidence and proportional designs, and for tile-sampled
  labels a ratio estimator with its ultimate-cluster interval, beside the naive one so the difference is visible.
  That last is the weakest of the three and says so: as shipped, it covered 94.5% to 95.4% of the time on five of
  exp78's tasks, 84.3% on Sen1Floods11 and 68.5% on MADOS, the two where a tenth of the tiles hold most of the errors
  (exp/out/exp78_shipped_tiles.json). Labelling 18 tiles of 16 windows and using the ordinary formula gave a "95%"
  interval that covered on 0.51 to 0.78 of draws; that is exp78's practical finding and the reason this function
  will not compute the naive interval alone.
- `estimate_from_indices` is for windows labelled without a design. It treats them as a random sample after
  checking only that they do not look like a review list: the mean suspicion percentile of a random sample is 0.5,
  of the tool's own review set about 0.97, and labelling the review set then dividing gives two to six times the
  true rate on every task of exp78's export. A sample more than four standard deviations above a random one is
  refused with the number rather than estimated. The check does not make the labels a random sample: labels from
  the confident end of the map, or clustered by tile, pass it, and the interval is then not valid.

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
# The coverage this warning quotes for "this interval" is the shipped design's (exp/exp78_shipped_tiles.py, artifact
# exp/out/exp78_shipped_tiles.json); 1.3.1 quoted exp78's own design of exactly 18 tiles and a normal quantile.
# tests/test_estimate.py reads the numbers back from the artifact.
TILES_WARNING = ("labels taken tile by tile are not independent, and a map whose tiles differ in size is labelled "
                 "unevenly; the naive interval beside this one is what the ordinary formula says, and in exp78, with "
                 "18 tiles of 16 windows, it covered 51 to 78% of the time on six of seven tasks while claiming 95%. "
                 "This interval is better and still not honest everywhere: graded on exp78's tasks it covered 94.5 to "
                 "95.4% of the time on five, 84.3% on Sen1Floods11 and 68.5% on MADOS, where a tenth of the tiles hold "
                 "most of the errors. Prefer the confidence design")
# What a whole-map number does not say when part of the map was read from other inputs (exp88). The first two
# travel in the outputs of estimate_error_rate and certify_zone when no input condition is recorded; the others go
# with the per-condition results. The numbers are exp88's (exp/out/exp88_summary.json) and its matched-head
# follow-up's (exp/out/exp88_matched_head.json); tests/test_assess.py reads them back from there.
SCOPE_ESTIMATE = ("This is the error rate of the whole map. If part of the map was predicted from other inputs, that "
                  "part's rate can differ widely from it. On a PASTIS map with half its tiles read without the optical "
                  "input by a probe trained on radar plus optical, that half's rate was 74.1% against 19.7% on the "
                  "rest, while random samples of 300 estimated 46.9% on average. Read by a probe trained on radar "
                  "alone, that half was still 28.3% wrong against 19.7% (exp88). Draw the sample with --condition to "
                  "get each part's rate.")
SCOPE_CERTIFY = ("The zone's error rate is certified over all its windows together. Where part of the map was predicted "
                 "from other inputs, that part of the zone can be wrong more often than the rest. If the model was not "
                 "trained on those inputs, its errors there can be confident ones and fall inside the zone (exp88). To "
                 "certify each input condition on its own, draw the sample with --condition.")
CONDITION_NOTE = ("Each condition's interval is its own 95% statement; the intervals do not hold jointly at 95%. The "
                  "whole-map rate weights each condition by its share of the map and can hide a condition that is much "
                  "worse.")
# Under the condition design, what holds by construction and what is not graded. Each condition's own interval is
# exact: the tests enumerate its coverage, and under a random sample exp88 graded it (95.6% and 95.7% of 2,000 draws,
# the claim pooled-error-rate-misstates-the-cloudy-part). The whole-map interval adds up the conditions' exact
# intervals at 1 - 0.05 / L (`_union_interval`), so it covers at least 95% by construction; the tests enumerate that
# too. The per-class intervals under this design have not been graded.
CONDITION_WHOLE_MAP = ("Each condition's own interval is exact. The whole-map interval takes each condition's exact "
                       "interval at 1 - 0.05/L, where L is the number of conditions not labelled in full, and weights it "
                       "by the condition's share of the map. At that level the L intervals hold together at least 95% of "
                       "the time, so the whole-map interval covers at least 95% by construction. It is wider than the "
                       "usual stratified interval, which adds up the conditions' variances, would be. When the "
                       "conditions' error rates are close, it is also wider than "
                       "the exact interval of a random sample of the same size. What the design buys is each condition's "
                       "own rate: a small condition gets as many labels as a large one, or all its windows. If only the "
                       "whole-map rate is needed, --design random can give a narrower interval.")
PER_CLASS_NOT_GRADED = ("The per-class intervals take the input conditions as strata. With conditions as strata, their "
                        "coverage has not been graded; each condition's error-rate interval is exact.")
FAMILY_NOTE = ("Certified per input condition. Each of the {L} conditions with at least {b1} labels is tested at delta "
               "{d:g}, so all their statements hold together except on at most {delta:g} of samples. On that event the "
               "certified windows taken together are wrong at most {alpha:g} of the time. Conditions with fewer labels "
               "are not tested. Outside the certified windows nothing is certified.")
FAMILY_NOTE_ONE = ("Certified per input condition. The one condition with at least {b1} labels is tested at delta {d:g}, "
                   "so its statement fails on at most {delta:g} of samples. When it holds, the certified windows are "
                   "wrong at most {alpha:g} of the time. Conditions with fewer labels are not tested. Outside the "
                   "certified windows nothing is certified.")
UNRECORDED = "unrecorded"          # the name of the windows with no recorded condition; reserved
CONFIDENCE_REFUSAL = ("that design allocates labels from the model's confidence, which can overstate the accuracy of "
                      "a condition read from an input combination the model was not trained on: it did on PASTIS for a "
                      "probe trained on radar plus optical and run on radar alone, though not on CropHarvest China 6 "
                      "(exp88). The tool cannot tell which inputs the model was trained on. Use --design condition (the "
                      "default with --condition) or --design random.")


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



# The tails above are computed through math.lgamma and math.exp, whose last digits differ between C libraries. A tail
# exactly equal to a level then falls on either side of it: one error among three labels of a 5-window zone at alpha
# 0.4 has the p-value 3/10, computed as 0.29999999999999977 on macOS and 0.30000000000000004 on Linux (glibc 2.34),
# so at delta 0.3 the same labels certified a zone on one system and not on the other. Where a tail lies within _TIE
# of its level, the comparison is redone in integer arithmetic against the level as written (0.3 is 3/10), so every
# system decides alike. Away from the level the float decides, as before.
# Until 2026-10-06 the window was _TIE whatever the population, but a tail's float error grows with it: lgamma(N + 1)
# is about N ln N, a unit in its last place is worth up to eps N ln N, and nine such terms make up a tail. On a zone of
# 1,293,697 windows (b = 233, k = 7, alpha 0.05) the p-value was 3.4e-9 off, past the window: the float gave
# 0.0999999997, integer arithmetic 0.1000000001, and certify_zone certified the whole map at delta 0.1 where the exact
# test refuses it. The window is now relative to the size of the numbers (`_tie`), 64 eps N ln N with _TIE its floor,
# so the integer arithmetic decides wherever the float could err; it costs a few hundred big-integer terms at most.
_TIE = 1e-9


EXACT_N_MAX = 5_000_000


def _tie(N):
    """The relative window around a level inside which a tail over a population of N is decided exactly. Above
    EXACT_N_MAX windows the floats decide, as before 2026-10-06: the window grows as N ln N, and on a full Sentinel-2
    tile at patch 1 (about 1e8 windows) the exact sums it calls took seconds per interval, where the float error they
    guard against is of order 1e-8 relative (the review of 2026-10-06). Every recorded experiment is below the cap."""
    import math
    N = max(int(N), 1)
    if N > EXACT_N_MAX:
        return _TIE
    return max(_TIE, 64 * float(np.finfo(np.float64).eps) * N * math.log(N + 1))


def _level(x):
    """A level as written: 0.1 is 1/10, not the binary float nearest to it."""
    from fractions import Fraction
    return x if isinstance(x, Fraction) else Fraction(repr(float(x)))


def _float_below(q):
    """The largest float at most the fraction q."""
    import math
    from fractions import Fraction
    f = float(q)
    return f if Fraction(f) <= q else math.nextafter(f, -math.inf)


def _float_above(q):
    """The smallest float at least the fraction q."""
    import math
    from fractions import Fraction
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


def _hyper_tail_exact(k, n, N, K, upper):
    """_hyper_tail in integer arithmetic, as a Fraction: P(X >= k) when `upper`, else P(X <= k)."""
    import math
    from fractions import Fraction
    lo, hi = max(0, n - (N - K)), min(n, K)
    xs = range(max(k, lo), hi + 1) if upper else range(lo, min(k, hi) + 1)
    return Fraction(sum(math.comb(K, x) * math.comb(N - K, n - x) for x in xs), math.comb(N, n))


def _at_most(value, level, exact, N=None):
    """value <= level for a tail computed in floating point over a population of N; within _tie(N) of the level
    (_TIE without N), exact() decides."""
    level = _level(level)
    if abs(value - float(level)) > (_TIE if N is None else _tie(N)) * float(level):
        return value <= float(level)
    return exact() <= level

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
    a = (1 - _level(conf)) / 2                                   # as written: 0.95 gives 1/40
    lo_K, hi_K = k, N - (n - k)                                  # the counts the sample does not rule out outright
    left, right = lo_K, hi_K                                     # smallest K with P(X >= k | K) > a (nondecreasing in K)
    while left < right:
        mid = (left + right) // 2
        if not _at_most(_hyper_tail(k, n, N, mid, upper=True), a, lambda: _hyper_tail_exact(k, n, N, mid, upper=True), N):
            right = mid
        else:
            left = mid + 1
    K_lo = left
    left, right = lo_K, hi_K                                     # largest K with P(X <= k | K) > a (nonincreasing in K)
    while left < right:
        mid = (left + right + 1) // 2
        if not _at_most(_hyper_tail(k, n, N, mid, upper=False), a, lambda: _hyper_tail_exact(k, n, N, mid, upper=False), N):
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


def equal_allocation(sizes, budget, floor=MIN_PER_STRATUM):
    """The same number of labels for every stratum, by water-filling: a stratum too small for an equal share is
    labelled in full, and the rest of the budget is shared equally by the others, again and again until no stratum
    left is that small. The input conditions of the "condition" design are its strata. It never reads the model's
    confidence, which can overstate the accuracy of a condition read from an input combination the model was not
    trained on: it did on PASTIS for a probe trained on radar plus optical and run on radar alone, though not on
    CropHarvest China 6 (exp88).

    Returns n_h with sum n_h = budget and n_h <= N_h; strata not labelled in full differ by at most one label, and
    n_h >= min(N_h, 2) whenever budget >= sum min(N_h, 2). A budget below sum min(N_h, floor) is refused, as by
    `neyman_allocation`. `neyman_allocation(sizes, 1 / sizes, budget)` is not an equal split: it gives (33, 7, 20)
    for sizes (100, 7, 50) at 60, where this gives (27, 7, 26)."""
    sizes = np.asarray(sizes, int)
    N = int(sizes.sum())
    need = int(np.minimum(sizes, floor).sum())
    if budget > N:
        raise ValueError(f"a budget of {budget} is more than the {N} units of the strata")
    if budget < need:
        raise ValueError(
            f"a budget of {budget} cannot give each of the {int((sizes > 0).sum())} non-empty strata its {floor} labels "
            f"(needs {need})")
    n = np.zeros(sizes.size, int)
    open_ = np.ones(sizes.size, bool)
    left = int(budget)
    while open_.any():
        share = left / int(open_.sum())
        small = open_ & (sizes <= share)
        if not small.any():
            break
        n[small] = sizes[small]                                 # too small for an equal share: a census
        left -= int(sizes[small].sum())
        open_ &= ~small
    k = int(open_.sum())
    if k:
        n[open_] = left // k
        rem = left - (left // k) * k
        # The remainder goes one label each to the `rem` open strata with the most units left, lowest index on ties.
        # The spec (condition_spec.md 2.5) read "one label at a time to the open stratum with most units left",
        # re-ranked after each label; that gives (4, 2, 2) for sizes (6, 3, 3) at 8, two strata not labelled in full
        # that differ by two, against the guarantee it states. No stratum gets more than one label of the remainder.
        opened = np.flatnonzero(open_)
        rank = opened[np.lexsort((opened, -(sizes[opened] - n[opened])))]
        n[rank[:rem]] += 1
    return n


def draw_stratified(rng, strata, sizes, allocation):
    """A without-replacement draw of `allocation[h]` units from each stratum."""
    idx = []
    for h, nh in enumerate(allocation):
        pool = np.flatnonzero(strata == h)
        idx.append(rng.choice(pool, min(int(nh), pool.size), replace=False))
    return np.concatenate(idx)


def _condition_index(condition, pop, n_windows, condition_names=None):
    """The input condition of every population window as an index c = 0..K-1: the recorded values present in the
    population in ascending order, then "unrecorded" last if any window has none. Returns (index per population
    window, values with None for unrecorded, names, sizes, notes)."""
    cond = np.asarray(condition).ravel()
    if cond.size != n_windows:
        raise ValueError(f"condition has {cond.size} entries for {n_windows} windows")
    if cond.dtype.kind == "b":
        cond = cond.astype(np.int64)
    elif cond.dtype.kind == "f":
        finite = np.isfinite(cond)
        if not np.all(np.mod(cond[finite], 1) == 0):
            raise ValueError("condition must hold an integer per window, negative where none is recorded; bin a "
                             "continuous layer, such as cloud fraction, first")
        cond = np.where(finite, cond, -1).astype(np.int64)      # NaN is unrecorded, as at the pixel level
    elif cond.dtype.kind in "iu":
        cond = cond.astype(np.int64)
    else:
        raise ValueError(f"condition must hold an integer per window, got dtype {cond.dtype}")
    cp = cond[pop]
    rec = cp >= 0
    vals = np.unique(cp[rec])
    if vals.size == 0:
        raise ValueError("no window of the population has a recorded condition: every one is unrecorded (a tie, or "
                         "no pixel with a value), so there is nothing to split the labels by")
    has_unrecorded = bool((~rec).any())
    K = int(vals.size) + int(has_unrecorded)
    index = np.full(cp.size, K - 1, dtype=np.int64)             # unrecorded windows form the last condition
    index[rec] = np.searchsorted(vals, cp[rec])
    given = {}
    for key, name in (condition_names or {}).items():
        try:
            v = int(key)
            if isinstance(key, (bool, np.bool_)) or float(key) != v:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"condition_names maps a condition value (an integer) to its name; got the key {key!r}") from None
        if v < 0:
            raise ValueError(f"condition_names gives a name to {v}, but a negative value means unrecorded")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"the name of condition {v} must be a non-empty string, got {name!r}")
        if name == UNRECORDED:
            raise ValueError(f'"{UNRECORDED}" is reserved for the windows with no recorded condition; name {v} otherwise')
        given[v] = name
    values = [int(v) for v in vals] + ([None] if has_unrecorded else [])
    names = [given.get(int(v), str(int(v))) for v in vals] + ([UNRECORDED] if has_unrecorded else [])
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"two conditions share the name {dup[0]!r}; condition names must be unique")
    notes = [f"value {v} ({name}) holds no window" for v, name in sorted(given.items()) if v not in set(vals.tolist())]
    sizes = np.bincount(index, minlength=K)
    return index, values, names, sizes, notes


def sample_for_estimation(margin, budget, design="confidence", p1=None, tiles=None, per_tile=M_PER_TILE,
                          n_strata=N_STRATA, valid=None, seed=0, condition=None, condition_names=None):
    """Which windows to label so that `estimate_error_rate` can give an honest rate afterwards.

    margin  : per-window confidence margin, higher = more confident (assess's `arrays["confidence"]`, flattened)
    budget  : number of windows to label
    design  : "confidence" (default) stratifies by margin quintile and allocates by Neyman's rule from the model's
              own top-1 probability `p1`, which it needs; "proportional" stratifies and allocates by size;
              "random" is a simple random sample; "tiles" takes tiles in a random order and labels up to
              `per_tile` windows of each until the budget is met, which needs `tiles`, the tile id of every window;
              "condition" stratifies by input condition and splits the labels equally (`equal_allocation`), which
              needs `condition`
    valid   : optional mask of windows that exist; invalid windows are never sampled and never counted
    condition : optional input condition of every window (assess's `arrays["condition"]`, flattened), an integer,
              negative where none is recorded. The "condition" design draws within it; "random" records it and
              draws as without it; the other designs refuse it. Windows with no recorded condition stay in the
              population as one more condition, "unrecorded", placed last
    condition_names : optional {value: name}; names are unique and non-empty, "unrecorded" is reserved, and the
              default name is str(value)
    Returns the sample: its `indices` into the flattened window grid, and everything the estimator needs.
    """
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    if valid.size != margin.size:
        raise ValueError(f"valid has {valid.size} entries for {margin.size} windows")
    if condition is None and condition_names:
        raise ValueError("condition_names names the values of `condition`, and no condition was given")
    if condition is not None and design in ("confidence", "proportional"):
        raise ValueError(f"the {design} design does not take a condition: {CONFIDENCE_REFUSAL}")
    if condition is not None and design == "tiles":
        raise ValueError("the tiles design does not take a condition: a tile can span conditions, and the tile interval "
                         "is not graded per condition.")
    if condition is None and design == "condition":
        raise ValueError('design "condition" needs `condition`, the input condition of every window (--condition)')
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
    if condition is not None:
        c_pop, c_values, c_names, c_sizes, c_notes = _condition_index(condition, pop, margin.size, condition_names)
    if design == "random":
        out["indices"] = pop[rng.choice(N, budget, replace=False)]      # the same draw with or without a condition
    elif design == "condition":
        # a simple random sample within each condition; with one condition this is the random design's own draw
        # (draw_stratified with one stratum returns the indices of rng.choice(N, budget))
        alloc = equal_allocation(c_sizes, budget)
        local = draw_stratified(rng, c_pop, c_sizes, alloc)
        out.update({"indices": pop[local], "strata": c_pop, "strata_of_population": pop, "sizes": c_sizes.tolist(),
                    "allocation": alloc.tolist(), "n_strata": int(c_sizes.size)})
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
        raise ValueError(f'design must be "confidence", "proportional", "random", "tiles" or "condition", got {design!r}')
    out["indices"] = np.asarray(out["indices"], int)
    if condition is not None:
        grid = np.full(margin.size, -1, dtype=np.int64)          # condition index per window, -1 outside the population
        grid[pop] = c_pop
        out["condition"] = {"values": c_values, "names": c_names, "sizes": c_sizes.tolist(),
                            "n_labelled": np.bincount(grid[out["indices"]], minlength=c_sizes.size).tolist(),
                            "allocation_rule": "equal" if design == "condition" else None}
        out["condition_grid"] = grid
        notes = list(c_notes)
        if c_sizes.size == 1:
            notes.insert(0, "one condition covers the whole population; this is a simple random sample")
        if notes:
            out["note"] = "; ".join(notes)
    return out


# ----------------------------------------------------------------------------- estimates
def _condition_guards(sample, idx):
    """Check a sample that records an input condition against itself, and return (condition grid, values, names,
    sizes). The condition is fixed at sampling time and the sidecar is its source of truth, so an edited or mixed-up
    sample is refused here rather than estimated: a stray condition index would otherwise be dropped in silence."""
    grid = np.asarray(sample["condition_grid"])
    if grid.ndim != 1:
        raise ValueError(f"condition_grid must be one value per window of the flattened grid; it has shape {grid.shape}")
    if grid.dtype.kind not in "iu":
        if grid.dtype.kind != "f" or not np.isfinite(grid).all() or not np.all(np.mod(grid, 1) == 0):
            raise ValueError("condition_grid must hold an integer condition index per window, -1 outside the population")
    grid = grid.astype(np.int64)
    if (idx >= grid.size).any():
        raise ValueError(f"a labelled window index lies beyond the {grid.size} windows of the condition grid")
    if (grid[idx] < 0).any():
        raise ValueError(f"{int((grid[idx] < 0).sum())} labelled window(s) sit outside the population the condition "
                         "grid records")
    N = int(sample["n_population"])
    if int((grid >= 0).sum()) != N:
        raise ValueError(f"the condition grid holds {int((grid >= 0).sum())} population windows, but the sample was drawn "
                         f"from {N}")
    info = sample.get("condition") or {}
    values, names, sizes = list(info.get("values", [])), list(info.get("names", [])), list(info.get("sizes", []))
    K = len(names)
    if K == 0 or not len(values) == len(sizes) == K:
        raise ValueError(f"the sample's condition block lists {len(values)} values, {K} names and {len(sizes)} sizes; "
                         "they must be one per condition")
    if int(grid.max()) >= K:
        raise ValueError(f"the condition grid holds a condition index of {int(grid.max())}, but the sample names {K} "
                         "conditions")
    counted = np.bincount(grid[grid >= 0], minlength=K)
    if [int(s) for s in sizes] != counted.tolist():
        raise ValueError(f"the condition sizes {[int(s) for s in sizes]} are not the counts of the condition grid "
                         f"{counted.tolist()}")
    if sample.get("design") == "condition":
        if int(sample.get("n_strata", -1)) != K:
            raise ValueError(f"a condition sample has one stratum per condition: n_strata is {sample.get('n_strata')}, "
                             f"with {K} conditions")
        strata = np.asarray(sample["strata"])
        if strata.size and (int(strata.min()) < 0 or int(strata.max()) >= K):
            raise ValueError(f"a stratum id lies outside the {K} conditions (0 to {K - 1})")
        sop = np.asarray(sample["strata_of_population"], int)
        if (sop.size != strata.size or sop.size != N or (sop < 0).any() or (sop >= grid.size).any()
                or not np.array_equal(strata.astype(np.int64), grid[sop])):
            raise ValueError("the strata of this condition sample are not the conditions of its population windows")
        if [int(s) for s in sample["sizes"]] != counted.tolist() or int(np.sum(sample["sizes"])) != N:
            raise ValueError(f"the strata sizes {list(sample['sizes'])} do not add up to the population of {N} windows "
                             "as the condition grid counts it")
        n_c = np.bincount(grid[idx], minlength=K)
        if n_c.tolist() != [int(a) for a in sample["allocation"]]:
            raise ValueError(f"the labelled windows per condition {n_c.tolist()} are not the design's allocation "
                             f"{[int(a) for a in sample['allocation']]}")
    return grid, values, names, [int(s) for s in sizes]


def _per_condition(grid, values, names, sizes, idx, wrong, N, whole):
    """Each condition's error rate with its exact interval. Under the condition design n_c is fixed; under a random
    sample the labels that fall in c are a simple random sample of c given their count n_c. Either way
    `hypergeom_interval(k_c, n_c, N_c)` covers at least 95%."""
    cidx = grid[idx]
    method = "exact hypergeometric interval (the labels in the condition are a simple random sample of it)"
    per, outside = {}, []
    for c, (v, name, Nc) in enumerate(zip(values, names, sizes)):
        m = cidx == c
        n_c, k_c = int(m.sum()), int(wrong[m].sum())
        row = {"value": v, "n_population": int(Nc), "share_of_map": int(Nc) / N, "n_labelled": n_c, "n_wrong": k_c}
        lo, hi = hypergeom_interval(k_c, n_c, int(Nc))            # (0, 1) with no label
        row.update({"estimate": k_c / n_c if n_c else None, "low": lo, "high": hi, "half_width": (hi - lo) / 2,
                    "method": method})
        if n_c == 0:
            row["note"] = "no labelled window fell in this condition; nothing can be said about it"
        elif not lo <= whole <= hi:
            outside.append(name)
        per[name] = row
    return per, outside


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


def _hypergeom_fractions(k, n, N, conf):
    """hypergeom_interval's ends, exactly: population counts over N (the exact rate when n == N, 0 to 1 with no
    label), widened to hold k / n. The float K / N times N is within far less than 1/2 of K."""
    from fractions import Fraction
    a, b = _hypergeom_interval(k, n, N, conf)
    a, b = Fraction(round(a * N), N), Fraction(round(b * N), N)
    if n:
        a, b = min(a, Fraction(k, n)), max(b, Fraction(k, n))
    return a, b


# Windows the reviewer could not judge, and a reviewer who errs (the CLI defect of 1 October 2026: the command told a
# reviewer to leave out a window they could not judge, and refused every CSV that did). Both rest on one property: the
# exact intervals are monotone in the error count, and the stratified estimate in each stratum's, so an interval that
# covers at the worst case covers at every case in between.
UNJUDGED_NOTE = ("{u} of the {n} labelled windows could not be judged (`?`). The lower end counts every one of them as "
                 "right and the upper end every one as wrong, so the interval covers at least 95% whatever made them "
                 "hard to judge, and the estimate is a range for the same reason. Judging them narrows it.")
UNJUDGED_STRATIFIED_NOTE = ("Under this design any `?` also replaces the design's Wilson interval, which does not move one way "
                            "with the error count, by the union bound over its strata, which does and is much wider: most "
                            "of the width comes from that switch, not from the `?` windows themselves. Judging every "
                            "window restores the narrower interval; a random sample bounds `?` at the cost of each "
                            "window's own range.")
REVIEWER_NOTE = ("The interval is widened for a reviewer who marks at most {fa:.3g} of the truly correct windows wrong "
                 "(false alarms) and misses at most {miss:.3g} of the truly wrong ones: the lower end of the labels' "
                 "interval becomes (low - {fa:.3g}) / (1 - {fa:.3g}) and the upper end high / (1 - {miss:.3g}). It covers "
                 "the true rate whenever the labels' interval covers the rate the reviewer's labels would show on every "
                 "window and the reviewer errs no more often than that; the bounds are the user's, not measured here. "
                 "Per input condition they must hold within each condition.")


def _stratum_union(sample, wrong):
    """The stratified estimate and the union-bound interval over a confidence or proportional design's strata: each
    stratum's exact interval at 1 - 0.05 / L, weighted by its share of the map (`_union_interval`). Exact and monotone
    in each stratum's error count, so it bounds windows the reviewer could not judge, where the design's own Wilson
    interval on an effective sample size is neither. Returns (estimate, low, high, L)."""
    idx = np.asarray(sample["indices"], int)
    pop = np.asarray(sample["strata_of_population"], int)
    pos = {int(g): i for i, g in enumerate(pop)}
    local = np.array([pos[int(g)] for g in idx])
    strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
    n_h = np.bincount(strata[local], minlength=len(sizes))
    k_h = np.bincount(strata[local], weights=np.asarray(wrong, float), minlength=len(sizes)).round().astype(int)
    return _union_interval(k_h, n_h, sizes)


def estimate_error_rate(sample, wrong, unjudged=None, reviewer_false_alarm=0.0, reviewer_miss=0.0):
    """The error rate of the whole map from the labelled sample, with the interval its design earns
    (`_estimate_core`), and, when asked, bounded for windows the reviewer could not judge and for a reviewer who errs.

    unjudged : True per labelled window the reviewer could not judge (`?` in the CSV), in the order of the indices;
        its `wrong` value is ignored. The interval's lower end counts these windows as right and its upper end as
        wrong; the exact intervals are monotone in the error count, so the interval covers at least 95% whatever made
        them hard to judge. Under the confidence and proportional designs the interval is then the union bound over
        their strata (`_stratum_union`), exact, in place of the design's Wilson interval, which is not monotone. The
        tiles design refuses unjudged windows.
    reviewer_false_alarm, reviewer_miss : bounds the user states, in [0, 1): the share of truly correct windows the
        reviewer marks wrong and of truly wrong windows the reviewer marks right. The labels show the rate
        p = theta (1 - e1) + (1 - theta) e0, so theta lies in [(p - E0) / (1 - E0), p / (1 - E1)], and those maps,
        increasing in p, are applied to the interval's ends (the identified set is sharp, proved in Lean in the
        repository's lean/ folder; tests/test_unjudged.py checks the coverage by exact enumeration).
    Without these the result is `_estimate_core`'s, unchanged. With them `estimate_range` gives the range the labels
    allow, and `estimate` is None unless that range is one value; each condition's row is bounded the same way.
    Sharpness is proved in lean/, not tested here; tests/test_unjudged.py checks coverage and the formulas."""
    fa, miss = float(reviewer_false_alarm), float(reviewer_miss)
    for name, v in (("reviewer_false_alarm", fa), ("reviewer_miss", miss)):
        if not 0 <= v < 1:
            raise ValueError(f"{name} must be at least 0 and below 1, got {v:g}")
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    unk = np.zeros(wrong.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if unk.size != wrong.size:
        raise ValueError(f"{unk.size} unjudged flags for {wrong.size} labels")
    u = int(unk.sum())
    if u == 0 and fa == 0 and miss == 0:
        return _estimate_core(sample, wrong)
    design = sample["design"]
    if u and design == "tiles":
        raise ValueError("windows that could not be judged are not supported under the tiles design: its cluster "
                         "interval is not monotone in the error count, so no worst case bounds them. Judge every "
                         "window, or draw a random sample")
    w_lo, w_hi = np.where(unk, 0.0, wrong), np.where(unk, 1.0, wrong)
    lo_res = _estimate_core(sample, w_lo)
    hi_res = _estimate_core(sample, w_hi) if u else lo_res
    out = dict(lo_res)

    # The maps run in exact fractions, with the bounds as written, and round outward. Until 2026-10-06 they ran in
    # floats, and an end equal to the true rate, the sharp case of a reviewer erring exactly at the bound, landed one
    # float past it: five windows, all labelled, four truly wrong and one of them missed at reviewer_miss 0.25 gave
    # (0.6, 0.7999999999999999) against a true rate of 0.8. An end x is the nearest float to an exact end that can lie
    # half a unit in its last place on either side, so the map starts from the float beyond x; 0 and 1 are exact.
    from fractions import Fraction
    import math
    E0, E1 = _level(fa), _level(miss)

    def down(x):                               # the lower end for a reviewer with false alarms
        if not fa:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, -math.inf))
        return max(0.0, _float_below((q - E0) / (1 - E0)))

    def up(x):                                 # the upper end for a reviewer who misses errors
        if not miss:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, math.inf))
        return min(1.0, _float_above(q / (1 - E1)))

    if u and design in ("confidence", "proportional"):
        e_lo, low, _, L = _stratum_union(sample, w_lo)
        e_hi, _, high, _ = _stratum_union(sample, w_hi)
        for k in ("effective_n", "starved_strata", "warning"):     # they describe the Wilson interval, not this one
            out.pop(k, None)
        # L = 0 when every stratum is labelled in full: the interval is then the exact range of the rate, and until
        # 2026-10-06 the method printed "1 - 0.05/0" and the note blamed the width on a switch that adds none
        out.update({"strata_in_interval": L,
                    "method": (f"stratified by confidence margin; with windows that could not be judged, the sum of each "
                               f"stratum's exact interval at 1 - 0.05/{L}, weighted by its share of the map (union "
                               "bound), which covers at least 95% by construction") if L else
                              ("stratified by confidence margin; every window is labelled, so with windows that could not "
                               "be judged the interval is the exact range of the rate, counting them all right at its "
                               "lower end and all wrong at its upper end")})
    else:
        e_lo, e_hi, low, high = lo_res["estimate"], hi_res["estimate"], lo_res["low"], hi_res["high"]
    if fa or miss:
        out["labels_interval"] = {"low": low, "high": high}
    out.update({"estimate": None, "estimate_range": [down(e_lo), up(e_hi)], "low": down(low), "high": up(high),
                "n_unjudged": u, "reviewer_false_alarm": fa, "reviewer_miss": miss})
    if out["estimate_range"][0] == out["estimate_range"][1]:      # nothing to bound: the value, not "x% to x%"
        out["estimate"] = out["estimate_range"][0]
    out["half_width"] = (out["high"] - out["low"]) / 2
    notes = ([UNJUDGED_NOTE.format(u=u, n=int(wrong.size))] if u else []) + (
        [UNJUDGED_STRATIFIED_NOTE] if u and design in ("confidence", "proportional") and out["strata_in_interval"] else []) + (
        [REVIEWER_NOTE.format(fa=fa, miss=miss)] if fa or miss else [])
    out["bounds_note"] = " ".join(notes)
    if lo_res.get("by_condition"):
        per, outside = {}, []
        r_lo, r_hi = out["estimate_range"]
        for name, row_lo in lo_res["per_condition"].items():
            row_hi = hi_res["per_condition"][name]
            row = dict(row_lo)
            if row_lo["n_labelled"]:
                row.update({"estimate": None, "estimate_range": [down(row_lo["estimate"]), up(row_hi["estimate"])],
                            "low": down(row_lo["low"]), "high": up(row_hi["high"]),
                            "n_unjudged": int(row_hi["n_wrong"] - row_lo["n_wrong"])})
                row["half_width"] = (row["high"] - row["low"]) / 2
                if row["estimate_range"][0] == row["estimate_range"][1]:
                    row["estimate"] = row["estimate_range"][0]
                if row["low"] > r_hi or row["high"] < r_lo:
                    outside.append(name)
            per[name] = row
        out.update({"per_condition": per, "outside_condition_intervals": outside})
    return out


def _estimate_core(sample, wrong):
    """The error rate of the whole map from the labelled sample, with the interval its design earns.

    sample : what `sample_for_estimation` returned
    wrong  : 0/1 per labelled window, in the order of sample["indices"]: 1 where the label disagrees with the map

    A sample that records an input condition (`condition_grid`) also gets each condition's rate with its exact
    interval (`per_condition`); one that does not gets `scope`, what a whole-map rate does not say (exp88). Under the
    condition design the whole-map estimate is the stratified one, sum_c W_c k_c / n_c. Its interval weights each
    condition's exact interval at 1 - 0.05 / L by the condition's share, where L is the number of conditions not
    labelled in full (`conditions_in_interval`), and so covers at least 95% by construction; it is wider than a
    stratified interval would be and, when the conditions' error rates are close, wider than a random sample's
    exact interval of the same size; it can be narrower when they are far apart (tests/test_estimate_exact.py,
    expected widths computed exactly). `design_variance` is the stratified estimate's unbiased variance, for
    information; the interval does not use it. `condition_note` says what each interval is.
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
    by_condition = sample.get("condition_grid") is not None
    if design == "condition" and not by_condition:
        raise ValueError('a sample of the "condition" design carries its condition_grid, and this one has none')
    if by_condition:
        if design not in ("random", "condition"):
            raise ValueError(f"an input condition is recorded only with the random and condition designs; this sample "
                             f"was drawn with the {design!r} design")
        grid, c_values, c_names, c_sizes = _condition_guards(sample, idx)
    if design == "condition":
        pop = np.asarray(sample["strata_of_population"], int)
        pos = {int(g): i for i, g in enumerate(pop)}
        stray = [int(g) for g in idx if int(g) not in pos]
        if stray:
            raise ValueError(f"{len(stray)} sampled window(s), first {stray[0]}, are not in the population this sample "
                             "was drawn from (an invalid window, or a sample built against another mask)")
        local = np.array([pos[int(g)] for g in idx])
        err = np.zeros(pop.size)
        err[local] = wrong
        strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
        # v, the unbiased variance of the stratified estimate with the conditions as strata, is kept as
        # `design_variance` and does not enter the interval
        var = stratified_mean_and_variance(err, strata, local, sizes, N)[1]
        n_h = np.bincount(strata[local], minlength=len(sizes))
        k_h = np.bincount(strata[local], weights=wrong, minlength=len(sizes)).round().astype(int)
        # a condition labelled in full (a one-window condition, say) has no sampling error and is not starved
        starved = int(((n_h < MIN_PER_STRATUM) & (n_h < np.asarray(sizes))).sum())
        # theta = sum_c W_c k_c / n_c, W_c = N_c / N, the stratified estimate; its interval weights each condition's
        # exact interval at 1 - 0.05 / L by its share: at least 95% by construction. It replaced spec 2.7's stratified
        # Wilson interval and 523efde's variance floor (review of 2026-09-29; the reason is in `_union_interval`)
        est_, lo, hi, L = _union_interval(k_h, n_h, sizes)
        if len(c_names) == 1:
            # one condition is the whole population and the draw was the random design's own: the same numbers
            out.update({"estimate": est_, "low": lo, "high": hi,
                        "method": "exact hypergeometric interval (simple random sample of a finite map)"})
        else:
            out.update({"estimate": est_, "low": lo, "high": hi, "conditions_in_interval": L,
                        "starved_strata": starved, "design_variance": var,
                        "method": "stratified by input condition; sum of each condition's exact interval at 1 - 0.05/L, "
                                  "weighted by its share of the map (union bound); covers at least 95% by construction"})
            if starved:
                out["warning"] = (f"{starved} of {sample['n_strata']} conditions had fewer than {MIN_PER_STRATUM} "
                                  "labelled windows and add nothing to design_variance; the whole-map interval does not "
                                  "use that variance")
    elif design == "random":
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
        # every window labelled (the indices are distinct and all in the population): the rate is exact and the
        # interval the point stratified_interval_wilson returns. Until 2026-10-06 the notes below still called it
        # the simple-random Wilson bound, or narrower than it should be, beside an interval of zero width
        census = local.size >= pop.size
        if census:
            out["method"] += "; every window is labelled, so the rate is exact, with no sampling error"
        elif est in (0.0, 1.0):
            notes.append(f"{'no' if est == 0 else 'every'} labelled window was wrong, so the design's variance is zero and "
                         f"the interval is the simple-random Wilson bound at {idx.size} labels; the stratification "
                         "cannot narrow it without an observed error")
        if starved and not census:
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
    if by_condition:
        per, outside = _per_condition(grid, c_values, c_names, c_sizes, idx, wrong, N, out["estimate"])
        note = CONDITION_NOTE + (" " + CONDITION_WHOLE_MAP if design == "condition" and len(c_names) >= 2 else "")
        out.update({"by_condition": True, "per_condition": per, "condition_note": note,
                    "outside_condition_intervals": outside})
    else:
        out["scope"] = SCOPE_ESTIMATE
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
    """Do these windows look like a review list rather than a random sample of the map? The MEAN suspicion
    percentile of the sample, against review_set_threshold: 0.5 for a random draw, near 1 for the tool's own review
    set, which is built to be enriched for errors. Passing does not make the windows a random sample: labels from the
    confident end of the map, or clustered by tile, pass. The median is reported beside it and decides nothing. A census (every valid window, once) is never
    an enriched set: at a census the threshold is exactly 0.5 and the mean is 0.5 up to rounding, and until the
    release check of 24 September a sum that rounded up refused 2 to 5% of censuses as review sets."""
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
    census = bool(inside.all() and np.unique(idx[inside]).size == pop.size)
    return {"mean_suspicion_percentile": mean, "median_suspicion_percentile": med, "threshold": thr,
            # 1e-12 is far below any real enrichment (one standard error at 300 labels is 0.017) and far above the
            # rounding of a mean of percentiles
            "looks_like_a_review_set": bool(np.isfinite(mean) and not census and mean > thr + 1e-12),
            "share_in_top_5pct": float(np.mean(pct[inside] > 0.95)) if inside.any() else float("nan"),
            "n_outside_population": int((~inside).sum()), "n_duplicated": int(idx.size - np.unique(idx).size)}


def estimate_from_indices(indices, wrong, margin, valid=None):
    """An error rate from windows labelled without a design, treated as a simple random sample after checking only
    that they do not look like a review list. The tool's own review set is refused: it is built to hold errors, and
    labelling it then dividing gave two to six times the true rate on every task of exp78's export. Labels from the
    confident end of the map, or clustered by tile, pass the check, and the interval is then not valid."""
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

    sample    : what `sample_for_estimation` returned (designs "random", "confidence", "proportional" or "condition")
    reference : the reference class of every labelled window, in the order of sample["indices"], integers >= 0
    map_class : the map's class of EVERY window (the flattened grid `sample` was drawn on); negative at no-data,
                and the count of non-negative entries must equal the population the sample was drawn from

    Under a random sample, the user's accuracy of class c has an exact hypergeometric interval (`hypergeom_interval`)
    on the labelled windows the map calls c, since those are a simple random sample of that class's windows, and
    `interval` does not apply to it; the class shares are post-stratified by
    map class (Olofsson et al. 2014, eq. 4 and 5) and the producer's accuracy follows their eq. 7. Under the
    confidence design every quantity is a ratio of Horvitz-Thompson totals over the margin strata with a
    linearised variance, because a class cuts across strata; the condition design runs the same estimators with the
    input conditions as strata, which no experiment has graded, and the method says so, while its overall accuracy is
    one minus `estimate`'s whole-map interval, the union bound over the conditions. With one condition the
    condition design is the random design, and gets the random design's estimators and method. Intervals are Wilson
    on the effective sample size (interval="wilson", the default; "wald" is the field's convention and is kept for the record, where exp81
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
    if ref.dtype.kind not in "iu":
        # read as numbers first, as estimate_error_rate reads `wrong`: until 2026-10-06 np.mod ran on whatever came
        # in, so a column read with the csv module (strings) or a '?' raised numpy's TypeError, or a string-formatting
        # one from an object array, instead of this refusal
        try:
            ref = ref.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("reference classes must be integers") from None
        if not np.isfinite(ref).all() or not np.all(np.mod(ref, 1) == 0):
            raise ValueError("reference classes must be integers")
    ref = ref.astype(int)
    if (ref < 0).any():
        raise ValueError("reference classes must be >= 0; a window the reviewer could not label should not be in the sample")
    design = sample["design"]
    if design == "tiles":
        raise ValueError("per-class accuracy from a tile sample is not graded yet (exp81 covers random and confidence "
                         "designs); label a random or confidence-designed sample")
    if design not in ("random", "confidence", "proportional", "condition"):
        raise ValueError(f"unknown design {design!r}")
    # With one input condition the condition design is the random design, drawn by the same rng call, so it gets the
    # random design's estimators, numbers and method (condition_spec.md 2.9); until 2026-09-29 it ran the
    # Horvitz-Thompson form with one stratum and said "not graded"
    srs = design == "random"
    if design == "condition":
        srs = len(_condition_guards(sample, idx)[2]) == 1          # a stray condition id would be dropped in silence
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
    if srs:
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
        if design == "condition":
            # `estimate`'s interval under this design, the union bound over the conditions, whatever `interval` says
            # (as the random design's is exact). Until 2026-10-06 this took the stratified Wilson interval that
            # `estimate` retired for the design on 2026-09-29: on 4,000 windows 0.5% wrong beside 200 windows 50%
            # wrong, 150 labels each, it covered the true accuracy on 53% of draws, and it contradicted the error
            # rate printed beside it ([95.6%, 98.1%] against an error rate of [2.0%, 6.4%] from the same labels).
            # By the symmetry of the exact interval this is one minus the error rate's interval.
            n_h = np.bincount(strata[local], minlength=len(sizes))
            k_h = np.bincount(strata[local], weights=(m_s == r_s).astype(float), minlength=len(sizes)).round().astype(int)
            o_est, o_lo, o_hi, _ = _union_interval(k_h, n_h, sizes)
        elif interval == "wilson":                                 # `estimate`'s graded form, census included
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
        method = ("stratified by input condition: ratios of Horvitz-Thompson totals with linearised variance; shares as "
                  "Horvitz-Thompson totals; with input conditions as strata these per-class intervals have not been graded; "
                  "the overall accuracy takes the error rate's interval (union bound over the conditions), which covers "
                  "at least 95% by construction"
                  if design == "condition" else
                  "stratified by confidence margin: ratios of Horvitz-Thompson totals with linearised variance; shares as Horvitz-Thompson totals")
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
        if not srs:                                                # the random-design user's accuracy is exact
            counted.append(("user's accuracy", commissions))
        rare = [(q, n_err) for q, n_err in counted if 0 < n_err < RARE_ERRORS]
        if rare and N_map[c] > 0:
            codes.append("few errors")
            # each quantity with its own count: a user's accuracy resting on 4 commissions is not "27 to 31" errors
            # because the same class has 27 omissions (the exp86 round 6 audit)
            rests = "; ".join(f"the {q} rests on {n_err} sampled error{'' if n_err == 1 else 's'}" for q, n_err in rare)
            notes.append(f"{rests} ({commissions} the map calls this class wrongly, {omissions} of this class it misses); "
                         "one error more or fewer in the draw moves the estimate by a whole window's weight, and the "
                         "interval can miss it")
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
        if not srs:
            f_all = idx.size / N
            thin = np.array([sizes[h] > 0 and (strata[local] == h).sum() / sizes[h] < 0.5 * f_all for h in range(len(sizes))])
            in_thin = float((thin[strata[m_pop == c]]).mean()) if (m_pop == c).any() else 0.0
            if in_thin > 0.5:
                codes.append("thin strata")
                notes.append(f"{100 * in_thin:.0f}% of this class sits in {'' if design == 'condition' else 'confidence '}strata the design samples at under half the "
                             "overall rate; if its errors are rare there they are often not drawn, and the interval is then "
                             "optimistic")
        if notes:
            row["warning"] = "; ".join(notes)
            row["warning_codes"] = codes
    out = {"design": design, "interval": interval, "n_labelled": int(idx.size), "n_population": N, "n_classes": C, "nominal_coverage": 0.95,
           "overall_accuracy": overall_row, "confusion_counts": conf.tolist(),
           "per_class": per, "method": method}
    if srs:
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
ZONE_CUTS = ("standard", "ramp")
ZONE_CUT = "standard"             # the level cut; "ramp" is an option (exp95, exp96 and their audit: zone_cut)
# The note a prefix certification carries. The prefix rule is fixed-sequence testing (Angelopoulos et al. 2021, Learn
# then Test): the levels are ordered before any label is read and each p-value is exact, so a zone wrong more than
# alpha of the time is certified only if the first such level in the order passes its test, which happens on at
# most delta of samples, on any map. Until 1.3.1 the note said the rule was valid only if the zone's error rate does
# not fall as the zone grows; it needs no such assumption. tests/test_trust_zone.py enumerates every error pattern of
# small maps, monotone or not, and every draw, for both rules.
PREFIX_NOTE = ("prefix rule: fixed-sequence testing, valid on any map whatever the shape of its error rate; it stops at "
               "the first zone it cannot certify, so it certifies little when the most confident windows hold many "
               "errors, where the bonferroni rule can certify more")


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



def zone_pvalue_exact(k, b, n, alpha):
    """zone_pvalue in integer arithmetic, as a Fraction; used where a p-value lies within `_tie(n)` of its level."""
    import math
    from fractions import Fraction
    n, b, k = int(n), int(b), int(k)
    if b == 0:
        return Fraction(1)
    K0 = math.floor(alpha * n) + 1              # the same null count as zone_pvalue
    if K0 > n:
        return Fraction(0)
    return _hyper_tail_exact(k, b, n, K0, upper=False)


def zone_upper_bound(k, b, n, delta=ZONE_DELTA):
    """The largest error rate K/n of a zone of n windows that k errors among b sampled do not reject at level
    delta: the exact hypergeometric upper confidence bound. With no labels it is 1."""
    n, b, k = int(n), int(b), int(k)
    if b == 0:
        return 1.0
    lo, hi = k, n - (b - k)                     # K must allow k wrong and b - k right among the sample
    def keeps(K):                               # P(X <= k | K) > delta: K is not rejected
        return not _at_most(hypergeom_cdf(k, n, K, b), delta, lambda: _hyper_tail_exact(k, b, n, K, upper=False), n)
    if keeps(hi):
        return hi / n
    while hi - lo > 1:                          # P(X <= k | K) is nonincreasing in K: bisect
        mid = (lo + hi) // 2
        if keeps(mid):
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


def zone_cut(budget, b_min, cut=None):
    """The smallest coverage certify tests with `budget` random labels; b_min is min_labels_to_certify.

    "standard" (the default): b_min / n, the coverage whose zone expects b_min labels. A level enters at the budget
    where it expects b_min labels; it then holds fewer than b_min about half the time and passes only with almost no
    error among them, and the prefix rule must pass it before any larger zone, so the chance of certifying can fall
    when labels are added (exp95: by more than 0.05 of the map on 14 of 34 maps). `plan` lists those budgets.
    "ramp" (an option, exp96): b_min / n up to 3 b_min labels, as standard; then 1/3, so that no level enters while
    labels accrue, up to 9 b_min; then 3 b_min / n, so that a level enters once it expects 3 b_min labels. It never
    rises with n and depends on the budget alone, so the rules keep their guarantee. On 332 maps it was not designed
    on it halved the largest fall on 77% of the maps where standard's exceeded 0.05, which met the preregistered bar of
    75%, but the pre-record audit found that share at the bar: P3 held on 5 of 9 seeds (0.744 to 0.776), about a
    quarter of the halvings were levels the ramp tests only past the 3,000 labels graded (on Sen1Floods11-like maps its
    5% level enters near 60 b_min with a fall as large as standard's), and small maps with high error lose the clean
    small zones standard certifies. So it is not the default."""
    n = max(int(budget), 1)
    cut = ZONE_CUT if cut is None else cut                    # the default is read at the call, not at import
    if cut not in ZONE_CUTS:
        raise ValueError(f"cut must be one of {ZONE_CUTS}, got {cut!r}")
    if cut == "standard" or n <= 3 * b_min:
        return b_min / n
    if n <= 9 * b_min:
        return 1 / 3
    return 3 * b_min / n


def zone_levels(n_population, budget, alpha, delta=ZONE_DELTA, grid=ZONE_GRID, cut=None):
    """The grid coverages a budget can certify at (alpha, delta), and the zone size at each: levels below the cut
    (`zone_cut`) are dropped before any label is seen. Returns (coverages, zone sizes, c_min)."""
    b_min = min_labels_to_certify(alpha, delta)
    c_min = zone_cut(budget, b_min, cut)
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


def apply_zone_rule(p, b, k, alpha, delta=ZONE_DELTA, rule="prefix", n=None):
    """Which grid levels a rule accepts, and the largest one; levels are in increasing coverage.
    prefix      accept while p <= delta from the smallest zone up, stop at the first failure: fixed-sequence
                testing (Angelopoulos et al. 2021, Learn then Test), valid on any map whatever the shape of the
                zone's error rate; weak when the most confident windows hold many errors, since it stops early
    bonferroni  accept every level with p <= delta / J, J the number of levels (Angelopoulos et al. 2021, Learn
                then Test; valid on any map); it can pass a level the prefix rule stops at, and needs a smaller
                p-value at every level
    plugin      accept every level whose sample rate k / b is at most alpha; no guarantee, the comparator
    `n`, the zone sizes, lets a p-value within `_tie(n)` of its level be decided in integer arithmetic, so that every
    system certifies the same zone from the same labels; `certify_zone` passes it. Without it the float decides."""
    p, b, k = np.asarray(p, float), np.asarray(b, int), np.asarray(k, int)

    def passes(j, level):
        if n is None:
            return p[j] <= float(level)
        return _at_most(p[j], level, lambda: zone_pvalue_exact(k[j], b[j], n[j], alpha), n[j])
    if rule == "prefix":
        acc = np.zeros(p.size, bool)
        for j in range(p.size):
            if passes(j, delta):
                acc[j] = True
            else:
                break
    elif rule == "bonferroni":
        level = _level(delta) / max(p.size, 1) if n is not None else delta / max(p.size, 1)
        acc = np.array([passes(j, level) for j in range(p.size)], bool)
    elif rule == "plugin":
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = (b > 0) & (k <= alpha * b)
    else:
        raise ValueError(f"rule must be one of {ZONE_RULES}, got {rule!r}")
    best = int(np.flatnonzero(acc).max()) if acc.any() else None
    return acc, best


def certify_zone(margin, indices, wrong, alpha, delta=ZONE_DELTA, rule="prefix", grid=ZONE_GRID, valid=None,
                 cut=None):
    """The largest share of the map, taken from the most confident window down, that is wrong at most `alpha` of
    the time, certified so that the statement fails with probability at most `delta` over the reviewer's draw.

    margin  : per-window confidence, higher = more trusted; NaN and invalid windows are outside the population
    indices : the labelled windows, which must be a simple random sample of the valid windows
    wrong   : 0/1 per labelled window, in the order of `indices`
    rule    : "prefix" (fixed-sequence testing, valid on any map; certifies little when the most confident windows
              hold many errors), "bonferroni" (valid on any map; can certify more in that case and less in others),
              "plugin" (no guarantee; what a reviewer would do unaided)

    Returns coverage None when nothing can be certified, with the reason; a labelled set that looks like the
    tool's own review set is refused, because the hypergeometric argument needs a random draw. `scope` says that the
    rate is certified over all the zone's windows together; for a map read from different inputs in different
    places, `certify_by_condition` certifies each input condition on its own (exp88)."""
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
    cut = ZONE_CUT if cut is None else cut
    cov, sizes, c_min = zone_levels(N, idx.size, alpha, delta, grid, cut)
    out = {"rule": rule, "alpha": float(alpha), "delta": float(delta), "n_population": N, "n_labelled": int(idx.size),
           "level_cut": cut,
           "min_labels_to_certify": min_labels_to_certify(alpha, delta), "c_min": float(c_min), "levels": [],
           "coverage": None, "n_zone": None, "threshold": None, "upper_bound": None,
           "scope": SCOPE_CERTIFY}             # what a zone over the whole map does not say (exp88)
    if not cov:
        out["note"] = (f"{idx.size} labels cannot certify any zone at alpha={alpha:g}, delta={delta:g}: even a zone with "
                       f"no error among its labels needs {out['min_labels_to_certify']} of them, more than the budget")
        return out
    b, k = zone_counts(pos[idx], wrong, sizes)
    p = np.array([zone_pvalue(kk, bb, n, alpha) for kk, bb, n in zip(k, b, sizes)])
    acc, best = apply_zone_rule(p, b, k, alpha, delta, rule, n=sizes)
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
        out["note"] = PREFIX_NOTE
    else:
        out["note"] = (f"Bonferroni over the {len(cov)} testable levels: valid with no assumption on how the error rate "
                       "changes with the zone")
    if tied > inside:
        out["note"] = out.get("note", "") + (f"; {tied} windows share the threshold margin and only {inside} of them are inside "
                                            "the zone, so the zone is the set returned, not every window at or above the threshold")
    return out


def _family_note(L, refused, b1, d, delta, alpha):
    """certify_by_condition's note: delta split over the L conditions with at least b1 labels, of which those named
    in `refused` were not tested, since their labels do not look like a random sample of the condition."""
    if not refused:
        return (FAMILY_NOTE if L > 1 else FAMILY_NOTE_ONE).format(L=L, b1=b1, d=d, delta=delta, alpha=alpha)
    one, T = len(refused) == 1, L - len(refused)
    who = ("Condition " if one else "Conditions ") + ", ".join(f"'{n}'" for n in refused)
    why = (f"{'was' if one else 'were'} not tested: {'its' if one else 'their'} labels do not look like a random sample "
           "of the condition.")
    if T == 0:
        return (f"Certified per input condition. {who} {'holds' if one else 'hold'} at least {b1} labels but {why} No "
                "condition was tested, so nothing is certified.")
    if T == 1:
        held = (f"The one condition tested has delta {d:g}, so its statement fails on at most {d:g} of samples. When "
                f"it holds, the certified windows are wrong at most {alpha:g} of the time.")
    else:
        held = (f"The {T} conditions tested have delta {d:g} each, so their statements hold together except on at "
                f"most {T * d:g} of samples. On that event the certified windows taken together are wrong at most "
                f"{alpha:g} of the time.")
    return (f"Certified per input condition. Delta {delta:g} is split over the {L} conditions with at least {b1} labels, "
            f"{d:g} each. {who} {why} {held} Conditions with fewer labels are not tested. Outside the certified windows "
            "nothing is certified.")


def certify_by_condition(sample, wrong, margin, alpha, delta=ZONE_DELTA, rule="prefix", grid=ZONE_GRID, valid=None,
                         cut=None):
    """A certified zone inside each input condition, with delta split so that all the statements hold together.

    sample  : a sample of the "random" or "condition" design that records a condition (`condition_grid`)
    wrong   : 0/1 per labelled window, in the order of sample["indices"]
    margin  : per-window confidence on the flattened grid the sample was drawn on; `valid` as for `certify_zone`
    rule    : "prefix" or "bonferroni"; the plug-in rule is refused, since it has no guarantee to split

    Why it is valid. L, the number of conditions holding at least min_labels_to_certify(alpha, delta) labels, is
    fixed by the label counts before any label is read. Given the counts, the labels in each condition are a simple
    random sample of it, so `certify_zone` inside condition c at delta / L fails with probability at most delta / L,
    and by the union bound all L statements hold together except on at most delta of samples. On that event the
    certified windows taken together are wrong at most alpha of the time: the union of the zones has error rate
    sum |Z_c| R_c / sum |Z_c| <= alpha. A condition with fewer labels is not tested.

    Inside each condition `certify_zone` checks that the labels do not look like a review list (review_set_check);
    labels from the confident end or clustered by tile pass that check. A
    condition whose labels fail that check is reported not tested, with the reason, and the others are certified at
    the delta / L already fixed. The check reads where the labels sit, never what they say, so leaving that condition
    out only leaves its share of delta unused. On the tool's own draw the check fails rarely, by chance.

    Returns the whole-map keys of `certify_zone` with `coverage`, `n_zone`, `threshold` and `upper_bound` None (the
    union of the zones is not "the most confident share of the map"), and per condition the output of
    `certify_zone` inside it. `zone_indices_in_order` concatenates the certified zones."""
    design = sample.get("design")
    if design not in ("random", "condition"):
        raise ValueError(f"certifying per input condition needs a random or condition-designed sample; this one was drawn "
                         f"with the {design!r} design. The guarantee rests on the labelled windows inside each zone being "
                         "a random sample of that zone, which a draw stratified by confidence or by tile is not")
    if sample.get("condition_grid") is None:
        raise ValueError("this sample records no input condition (condition_grid); certify the map as a whole with "
                         "certify_zone, or draw the sample with a condition")
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    idx = np.asarray(sample["indices"], int).ravel()
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} labelled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    if rule not in ZONE_RULES:
        raise ValueError(f"rule must be one of {ZONE_RULES}, got {rule!r}")
    if rule == "plugin":
        # certify_zone accepts it as the comparator; here the family note would state a joint guarantee it lacks
        raise ValueError("the plug-in rule has no guarantee, so there is no delta to split over the conditions and "
                         "no joint statement to make; use the prefix or bonferroni rule")
    if (idx < 0).any():
        raise ValueError("a labelled window index is negative; indices point into the flattened window grid")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once")
    cgrid, values, names, sizes = _condition_guards(sample, idx)
    if valid.size != margin.size or margin.size != cgrid.size or not np.array_equal(
            np.flatnonzero(valid & np.isfinite(margin)), np.flatnonzero(cgrid >= 0)):
        raise ValueError("the map is not the one the sample was drawn on: its valid windows are not the population the "
                         "sample's condition grid records")
    N = int(sample["n_population"])
    K = len(names)
    n_c = np.bincount(cgrid[idx], minlength=K)
    b1 = min_labels_to_certify(alpha, delta)
    tested = [c for c in range(K) if n_c[c] >= b1]               # fixed by the counts, before any label is read
    L = len(tested)
    d = delta / L if L else None
    per, zones, n_cert, refused = {}, [], 0, []
    for c in range(K):
        m = cgrid[idx] == c
        chk = review_set_check(idx[m], margin, valid & (cgrid == c)) if c in tested else None
        if chk is not None and chk["looks_like_a_review_set"]:
            # certify_zone would refuse these labels as an enriched set, and until 2026-09-29 that refusal stopped
            # every condition. The check reads where the labels sit, not what they say, so leaving this condition out
            # after the split was fixed only leaves its share of delta unused: the others still hold together.
            how = ("the condition design draws each condition's labels at random within it" if design == "condition"
                   else "the labels of a random sample that fall in a condition are a random sample of it")
            r = {"value": values[c], "tested": False, "delta": None, "n_population": int(sizes[c]),
                 "n_labelled": int(n_c[c]), "coverage": None, "n_zone": None, "threshold": None, "upper_bound": None,
                 "levels": [],
                 "review_set_check": {"mean_suspicion_percentile": chk["mean_suspicion_percentile"],
                                      "threshold": chk["threshold"]},
                 "reason": (f"its {int(n_c[c])} labels sit at a mean suspicion percentile of "
                            f"{chk['mean_suspicion_percentile']:.2f} within the condition, above {chk['threshold']:.2f}. "
                            f"They look chosen for low confidence, yet {how}. A zone certified on chosen labels could "
                            "be wrong. If these are the labels `sample` drew, unedited, this is a rare chance draw")}
            refused.append(names[c])
        elif c in tested:
            r = certify_zone(margin, idx[m], wrong[m], alpha, delta=d, rule=rule, grid=grid, valid=valid & (cgrid == c),
                             cut=cut)
            r.pop("scope", None)
            if r["coverage"] is None and r["levels"] and K > 1:
                # certify_zone gives the smallest testable zone as a share of the population it was handed, which
                # here is the condition, not the map. A single condition is the whole map, and its entry stays
                # certify_zone's result word for word
                at = f"({r['levels'][0]['coverage']:.0%} of the"
                r["note"] = r["note"].replace(f"{at} map)", f"{at} condition)")
            r.update({"value": values[c], "tested": True, "reason": None})
            if r["coverage"] is not None:
                n_cert += int(r["n_zone"])
                zones.append(np.asarray(r["zone_indices_in_order"], int))
        else:
            r = {"value": values[c], "tested": False, "delta": None, "n_population": int(sizes[c]),
                 "n_labelled": int(n_c[c]), "coverage": None, "n_zone": None, "threshold": None, "upper_bound": None,
                 "levels": [],
                 "reason": f"{int(n_c[c])} labels; certifying any zone at alpha {alpha:g} needs at least {b1}"}
        per[names[c]] = r
    if L:
        note = _family_note(L, refused, b1, d, delta, alpha)
    else:
        # FAMILY_NOTE divides delta over the tested conditions; with none tested it has no delta to state
        note = (f"Certified per input condition. No condition holds the {b1} labels that certifying any zone at alpha "
                f"{alpha:g} needs, so none was tested and nothing is certified.")
    return {"rule": rule, "alpha": float(alpha), "delta": float(delta), "n_population": N, "n_labelled": int(idx.size),
            "min_labels_to_certify": b1, "levels": [], "coverage": None, "n_zone": None, "threshold": None,
            "upper_bound": None, "by_condition": True, "delta_per_condition": d, "n_conditions_tested": L - len(refused),
            "certified_share_of_map": n_cert / N if zones else None, "n_certified": n_cert, "per_condition": per,
            "zone_indices_in_order": np.concatenate(zones) if zones else np.zeros(0, int), "note": note}


# ----------------------------------------------------------------------------- which of two maps is more accurate
# Where two maps of the same windows give the same class, both are right or both are wrong; their accuracies differ
# only through the windows where they differ. So the difference acc_a - acc_b = (K_a - K_b) / N, with K_a and K_b the
# differing windows where a, and where b, is right, and a random sample of the D differing windows estimates it with
# about N / D times fewer labels than a sample of the whole map (exp90: a median of 4.8 times on the record's maps).
WHICH_MAP_NOTE = ("The difference is over the windows compared: both maps predict them and neither splits evenly between "
                  "two classes. This says which map is more accurate and by how much, not either map's accuracy: where the maps agree "
                  "they are both right or both wrong, and how often needs a sample of all windows (sample --design random "
                  "on each map). Labels are assumed right; a window marked ? is counted for each map both ways.")


def sample_disagreement(class_a, class_b, budget, valid=None, seed=0):
    """A simple random sample of the windows where two maps' classes differ, for `compare_from_disagreement`.
    class_a, class_b: each window's class (flattened alike); valid: windows both maps predict. A budget at least the
    number of differing windows labels them all."""
    a, b = np.asarray(class_a).ravel(), np.asarray(class_b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"the two maps have {a.size} and {b.size} windows; they must share one window grid")
    ok = np.ones(a.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    budget = int(budget)
    if budget < 1:
        raise ValueError(f"the budget must be at least 1, got {budget}")
    pop = np.flatnonzero(ok & (a != b))
    if pop.size == 0:
        raise ValueError("the two maps give the same class in every window both predict: neither is more accurate, and "
                         "there is nothing to label")
    rng = np.random.default_rng(seed)
    m = min(budget, pop.size)
    return {"design": "disagreement", "indices": pop[rng.choice(pop.size, m, replace=False)],
            "n_population": int(ok.sum()), "n_disagree": int(pop.size), "budget": budget, "seed": int(seed)}


def compare_from_disagreement(sample, class_a, class_b, reference, unjudged=None, conf=0.95):
    """Which of two maps is more accurate, from a labelled `sample_disagreement`.

    class_a, class_b: each sampled window's class in the two maps, in the order of sample["indices"]; reference: the
    class the reviewer saw there (ignored where `unjudged`). Among the n labelled windows of the D differing ones, a are
    right in map a and b in map b (a window can be right in at most one). Each count gets its exact hypergeometric
    interval over the D windows at 1 - (1 - conf) / 2, so the two hold together with probability at least conf (union
    bound), and the accuracy difference over the N windows both maps predict, (K_a - K_b) / N, lies in
    [(lo_a - hi_b) D / N, (hi_a - lo_b) D / N]. A window marked ? counts as wrong for a map at that map's lower end and
    as right at its upper end; the intervals move one way with the count, so the bound holds whatever made the windows
    hard to judge. The verdict names a map when the interval excludes 0."""
    idx = np.asarray(sample["indices"], int)
    ca, cb = np.asarray(class_a).ravel(), np.asarray(class_b).ravel()
    ref = np.asarray(reference).ravel()
    unj = np.zeros(idx.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if not ca.size == cb.size == ref.size == unj.size == idx.size:
        raise ValueError(f"{idx.size} sampled windows but {ca.size}, {cb.size}, {ref.size} and {unj.size} classes, "
                         "references and flags")
    if (ca == cb).any():
        raise ValueError("a sampled window has the same class in both maps; the sample must hold differing windows only")
    if not 0 < conf < 1:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    N, D, n = int(sample["n_population"]), int(sample["n_disagree"]), int(idx.size)
    judged = ~unj
    a = int((judged & (ref == ca)).sum())
    b = int((judged & (ref == cb)).sum())
    u = int(unj.sum())
    c = n - a - b - u
    side = 1 - (1 - conf) / 2
    lo_a, _ = hypergeom_interval(a, n, D, side)
    _, hi_a = hypergeom_interval(a + u, n, D, side)
    lo_b, _ = hypergeom_interval(b, n, D, side)
    _, hi_b = hypergeom_interval(b + u, n, D, side)
    # The ends are rationals (K/D, or k/n where the interval was widened to hold the sample share): computed in
    # Fractions and rounded once, an end equal to the true difference is never one float inside it, as it was in
    # floats (a census of the differing windows printed an interval that missed the truth by 3e-17, 2026-10-06).
    from fractions import Fraction

    def ex(x):
        # an end is K/D, or k/n where the interval was widened to hold the sample share; limit_denominator stopped
        # recovering K/D above about 1e8 differing windows (the review of 2026-10-06), so try both grids directly
        for den in (D, n):
            if den > 0:
                f = Fraction(round(x * den), den)
                if float(f) == x:
                    return f
        return Fraction(x)
    wf = Fraction(D, N)
    low, high = float((ex(lo_a) - ex(hi_b)) * wf), float((ex(hi_a) - ex(lo_b)) * wf)
    est_lo, est_hi = float(Fraction(a - (b + u), n) * wf), float(Fraction((a + u) - b, n) * wf)
    verdict = "a" if low > 0 else "b" if high < 0 else None
    return {"design": "disagreement", "n_population": N, "n_disagree": D, "disagree_share": D / N, "n_labelled": n,
            "n_unjudged": u, "n_a_right": a, "n_b_right": b, "n_neither": c,
            "share_a_right": {"low": lo_a, "high": hi_a}, "share_b_right": {"low": lo_b, "high": hi_b},
            "difference": {"estimate": est_lo if est_lo == est_hi else None, "estimate_range": [est_lo, est_hi],
                           "low": low, "high": high},
            "verdict": verdict, "conf": conf,
            "method": (f"labels on a simple random sample of the {D} windows where the maps differ; each map's share of "
                       f"them that is right gets its exact interval at {side:.3g}, and the difference of the two, times "
                       f"{D}/{N}, covers the whole map's accuracy difference at least {conf:.0%} of the time (union "
                       "bound)"),
            "note": WHICH_MAP_NOTE}

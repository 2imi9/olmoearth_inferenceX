"""Exact hypergeometric arithmetic: the tails, their integer-arithmetic tie rule and the exact interval (ADR-002)."""
import numpy as np


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
    tile at patch 1 (about 1e8 windows) the exact sums it calls took seconds per interval (the review of 2026-10-06).
    The float error they guard against grows with N: measured against the exact tail, about 2e-8 relative at 6e6
    windows, up to 8e-7 at 1e8 and 8e-6 at 1e9, so above the cap an interval's end can move by one window (k = 22 of
    n = 300 at N = 1e8: a lower end of 4,652,571 where the exact one is 4,652,570), a change in coverage of order 1e-8.
    Stated as 1e-8 relative until the bug hunt of 2026-10-25. Every recorded experiment is below the cap."""
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


def _hypergeom_fractions(k, n, N, conf):
    """hypergeom_interval's ends, exactly: population counts over N (the exact rate when n == N, 0 to 1 with no
    label), widened to hold k / n. The float K / N times N is within far less than 1/2 of K."""
    from fractions import Fraction
    a, b = _hypergeom_interval(k, n, N, conf)
    a, b = Fraction(round(a * N), N), Fraction(round(b * N), N)
    if n:
        a, b = min(a, Fraction(k, n)), max(b, Fraction(k, n))
    return a, b


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

"""How many labels to draw, before any is drawn, for each labelled route of the package.

For each budget on a ladder (10, 12, 15, 20, 25, 30, 40, 50, ... up to 10,000 labels or the census), the planner
gives the probability, over the reviewer's random draw, that the package's own procedure reaches the outcome asked
for:

- error rate (`sample --design random`, then `estimate`): the exact 95% interval is no wider than a stated width, for
  a map wrong at a stated rate (50% unless stated);
- which of two maps (`sample --other`, then `estimate`): the labels drawn where the maps differ name the more
  accurate map when the two accuracies differ by at least a stated amount, at the worst of the splits checked of the
  differing windows into right in one map, in the other or in neither (two-class maps have one split);
- certified zone (`certify`): for a zone wrong at a stated rate, the Bonferroni rule certifies it (exact, however the
  errors spread inside it); no spread of the errors lets the prefix rule certify it more often than the zone's own
  test passes (exact); and the prefix rule certifies it when every more confident zone of the grid is wrong no more
  often (simulated, with the errors spread evenly, the worst such map).

The recommended budget is the smallest budget checked from which every larger budget checked reaches the probability
asked for. Besides the ladder, nine budgets are checked below the recommendation and, where a budget is cheap to
evaluate exactly, every budget from it up to 5% above it (at least ten, at most fifty), because between budgets the
probability does not rise smoothly: the counts are discrete, so it steps, and for `certify` it falls at the budgets
where a smaller zone becomes testable (the prefix rule must then pass that zone first, with few labels, and the
Bonferroni rule splits delta over one more zone). Those budgets are listed. When the largest budget checked falls in
such a fall, no budget is recommended and the runs of budgets that reach the probability are given instead.

What is not known before labelling is asked for, not guessed: the error rate a map will turn out to have is what the
labels measure, and a map's own confidence overstates its accuracy (the record: a median 0.061). The only inputs read
from maps are counts: the windows in the population and, for two maps, the windows where they differ.
"""
import functools
import math

import numpy as np

from . import estimate as est

POWER = 0.9
TAIL = 1e-12                          # probability mass a count's distribution may leave out at each end
MAX_LABELS = 10_000                   # the largest budget checked unless asked for more
SPLITS = 21                           # both-wrong counts on the coarse scan of the splits of the differing windows
DENSE_SPLITS = 60                     # and every count up to this many, where the worst splits were found
WORST_KEPT = 6                        # splits carried from the nearest full scan to the budgets refine adds
ZONE_DRAWS = 2000
DEFAULT_RATE = 0.5
PLAN_NOTE = ("Planned before any label. Each probability is over the reviewer's random draw, for a map with the "
             "stated error rate or difference; the map's own error rate is what the labels will measure, so a plan "
             "made at a guessed rate holds for that rate only. Labels are assumed right.")
INTERVAL_NOTE = ("The interval ends are the exact tail inversion of estimate, decided in floating point; estimate "
                 "decides a tail lying within about 1e-6 of its level in exact arithmetic, which can move an end by "
                 "one window.")
STEPS = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5)


def ladder(top, start=10):
    """The budgets checked: 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, ... up to top, and top itself."""
    top, out, m = int(top), set(), 1
    while m * STEPS[0] <= top:
        out.update(int(round(s * m)) for s in STEPS if start <= s * m <= top)
        m *= 10
    out.add(top)
    return sorted(b for b in out if b >= 1)


def runs(probs, power):
    """The runs of consecutive budgets checked that reach power, as [(first, last)]; probs maps a budget to its
    probability (None: known to fall short)."""
    out, start, prev = [], None, None
    for b in sorted(probs):
        ok = probs[b] is not None and probs[b] >= power
        if ok and start is None:
            start = b
        if not ok and start is not None:
            out.append((start, prev))
            start = None
        prev = b
    if start is not None:
        out.append((start, prev))
    return out


def recommend(probs, power):
    """The smallest budget from which every larger budget checked reaches power, or None when the largest budget
    checked falls short (see `runs` for the budgets that do reach it then)."""
    r = runs(probs, power)
    return r[-1][0] if r and r[-1][1] == max(probs) else None


REFINE = 9                            # budgets added between the last budget short of power and the recommendation
WINDOW = 0.05                         # and, where a budget is cheap, every budget up to this share above it,
WINDOW_MAX = 50                       # at most this many


def refine(fn, probs, power, every=False):
    """probs with more budgets evaluated by fn: REFINE between the recommendation and the largest budget below it,
    so the recommendation is not left a ladder step above where the probability is reached; with `every`, also every
    budget from the recommendation up to WINDOW above it (at least ten, at most WINDOW_MAX), since between budgets
    checked the
    probability can step below power (the counts are discrete). Repeated while the recommendation moves, at most
    five times. Returns (probs, the budgets checked one by one as (first, last), or None)."""
    every_span = None
    for _ in range(5):
        rec = recommend(probs, power)
        if rec is None:
            break
        added = False
        below = [b for b in probs if b < rec]
        if below:
            lo = max(below)
            for b in sorted(set(int(round(v)) for v in np.linspace(lo, rec, REFINE + 2)[1:-1])):
                if lo < b < rec and b not in probs:
                    probs[b], added = fn(b), True
        if every:
            hi = min(max(probs), rec + WINDOW_MAX - 1, max(rec + 10, int(math.ceil(rec * (1 + WINDOW)))))
            for b in range(rec, hi + 1):
                if b not in probs:
                    probs[b], added = fn(b), True
            every_span = (rec, hi)
        if not added:
            break
    rec = recommend(probs, power)
    if every and rec is not None and every_span is not None and not every_span[0] <= rec <= every_span[1]:
        every_span = None
    return dict(sorted(probs.items())), every_span


def _check(name, x, lo, hi, lo_open=True, hi_open=True):
    if x is None or not np.isfinite(x) or (x <= lo if lo_open else x < lo) or (x >= hi if hi_open else x > hi):
        raise ValueError(f"{name} must lie in {'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}, "
                         f"got {x}")


def _check_count(name, x, lowest=1):
    if x is None or isinstance(x, bool) or not np.isfinite(x) or int(x) != x or int(x) < lowest:
        raise ValueError(f"{name} must be a whole number of at least {lowest}, got {x}")
    return int(x)


def _hyper_pmf(N, K, n):
    """(support, pmf) of X ~ Hypergeom: n drawn from N of which K are marked, trimmed where the mass beyond it is
    below TAIL at each end. Built from the ratio of successive terms, so no factorial of N is formed."""
    N, K, n = int(N), int(K), int(n)
    lo, hi = max(0, n - (N - K)), min(n, K)
    if lo >= hi:
        return np.array([lo]), np.array([1.0])
    j = np.arange(lo, hi, dtype=np.float64)
    r = np.log(K - j) + np.log(n - j) - np.log(j + 1) - np.log(N - K - n + j + 1)
    lp = np.concatenate([[0.0], np.cumsum(r)])
    k = np.arange(lo, hi + 1)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    c = np.cumsum(p)
    keep = (c >= TAIL) & (c - p <= 1 - TAIL)
    return k[keep], p[keep]


_TABLES = {}


def _gallop(pred, start, lo, hi):
    """The smallest K in [lo, hi] with pred(K) true, pred false then true along K, searched outward from `start`;
    hi + 1 if none."""
    start = min(max(start, lo), hi)
    if pred(start):
        a, step = start, 1                                     # find a false point below, then bisect
        while a > lo and pred(a - step if a - step >= lo else lo):
            a = a - step if a - step >= lo else lo
            step *= 2
        f = max(lo - 1, a - step)
        t = a
    else:
        f, step = start, 1
        while True:
            if f >= hi:
                return hi + 1
            t = min(hi, f + step)
            if pred(t):
                break
            f, step = t, step * 2
    while t - f > 1:                                           # pred(f) false (or f = lo - 1), pred(t) true
        m = (f + t) // 2
        f, t = (f, m) if pred(m) else (m, t)
    return t


def _ends_range(n, N, conf, k0, k1):
    """(lo, hi) arrays of `_ends` for k = k0..k1, filled count by count: each end rises with k, so each count's
    search starts from the previous count's end and gallops."""
    tab = _TABLES.setdefault((n, N, conf), {})
    if len(_TABLES) > 4096:
        _TABLES.clear()
        tab = _TABLES.setdefault((n, N, conf), {})
    a = (1 - float(est._level(conf))) / 2
    prev = None
    for k in range(k0, k1 + 1):
        if k in tab:
            prev = tab[k]
            continue
        if n == 0:
            tab[k] = (0.0, 1.0)
        elif n >= N:
            tab[k] = (k / n, k / n)
        else:
            lo_K, hi_K = k, N - (n - k)
            step = N / n
            g_lo = int(prev[2] + step) if prev is not None and len(prev) > 2 else int(k * N / n)
            K_lo = _gallop(lambda K: est._hyper_tail(k, n, N, K, upper=True) > a, g_lo, lo_K, hi_K)
            g_hi = int(prev[3] + step) if prev is not None and len(prev) > 3 else int((k + 1) * N / n)
            # largest K with P(X <= k | K) > a: one below the smallest K where it is <= a
            K_hi = _gallop(lambda K: est._hyper_tail(k, n, N, K, upper=False) <= a, g_hi + 1, lo_K, hi_K) - 1
            tab[k] = (min(K_lo / N, k / n), max(K_hi / N, k / n), K_lo, K_hi)
        prev = tab[k]
    lo = np.array([tab[k][0] for k in range(k0, k1 + 1)])
    hi = np.array([tab[k][1] for k in range(k0, k1 + 1)])
    return lo, hi


@functools.lru_cache(maxsize=1 << 18)
def _ends(k, n, N, conf):
    """estimate.hypergeom_interval's ends, decided in floating point (see INTERVAL_NOTE)."""
    if n == 0:
        return 0.0, 1.0
    if n >= N:
        return k / n, k / n
    a = (1 - float(est._level(conf))) / 2
    lo_K, hi_K = k, N - (n - k)
    left, right = lo_K, hi_K
    while left < right:                                        # smallest K with P(X >= k | K) > a
        mid = (left + right) // 2
        if est._hyper_tail(k, n, N, mid, upper=True) <= a:
            left = mid + 1
        else:
            right = mid
    K_lo = left
    left, right = lo_K, hi_K
    while left < right:                                        # largest K with P(X <= k | K) > a
        mid = (left + right + 1) // 2
        if est._hyper_tail(k, n, N, mid, upper=False) <= a:
            right = mid - 1
        else:
            left = mid
    return min(K_lo / N, k / n), max(left / N, k / n)


# ----------------------------------------------------------------------------- error rate
def error_rate_probability(N, n, width, error_rate, conf=0.95):
    """The probability that n random labels give a 95% interval no wider than `width` (high minus low) when
    round(error_rate N) of the N windows are wrong: the width is read for every error count the draw can give
    outside a TAIL of mass at each end. It is not assumed to rise to the middle count: on the 1/N grid it steps."""
    N, n = int(N), int(n)
    if n >= N:
        return 1.0                                             # a census: the interval is the point
    k, p = _hyper_pmf(N, int(round(error_rate * N)), n)
    lo, hi = _ends_range(n, N, float(conf), int(k[0]), int(k[-1]))
    return float(p[(hi - lo) <= width + 1e-12].sum())


def plan_error_rate(N, width, error_rate=None, power=POWER, conf=0.95, max_labels=MAX_LABELS):
    """The probability, budget by budget, that random labels give an error-rate interval no wider than `width`, for a
    map wrong at `error_rate` (50% when None: on a map much larger than the sample, the rate at which an interval is
    widest), and the recommended budget."""
    N = _check_count("the number of windows", N)
    _check("the width", width, 0, 1)
    _check("the probability asked for", power, 0, 1)
    rate = DEFAULT_RATE if error_rate is None else error_rate
    _check("the expected error rate", rate, 0, 1, lo_open=False, hi_open=False)
    top = min(N, _check_count("the largest budget", max_labels))
    fn = lambda b: error_rate_probability(N, b, width, rate, conf)
    probs, span = refine(fn, {b: fn(b) for b in ladder(top)}, power, every=True)
    rec = recommend(probs, power)
    out = {"question": "error_rate", "n_population": N, "width": float(width), "error_rate": float(rate),
           "error_rate_stated": error_rate is not None, "power": float(power), "labels": rec,
           "probability_at_labels": None if rec is None else probs[rec],
           "probability_by_budget": {str(b): p for b, p in probs.items()}, "checked_up_to": top,
           "every_budget_checked": None if span is None else list(span), "runs_reaching_power": runs(probs, power),
           "census_checked": top >= N, "note": PLAN_NOTE, "interval_note": INTERVAL_NOTE}
    if error_rate is None:
        out["rate_note"] = ("planned at 50% wrong, the rate at which an interval is widest on a map much larger than "
                            "the sample; with a sizeable share of the windows labelled other rates can need more, so "
                            "state the rate you expect when you know it")
    if rec is None:
        out["refusal"] = (f"no budget checked up to {top} gives an interval this narrow with probability {power:g}"
                          + ("" if top >= N else "; check larger budgets or accept a wider interval"))
    return out


# ----------------------------------------------------------------------------- which of two maps
def which_map_probability(N, D, n, KA, KB, conf=0.95):
    """(probability of naming map a, of naming map b) when n labels are drawn at random among the D windows where the
    maps differ, KA of which are right in map a and KB in map b (the rest in neither), with the interval
    `compare_from_disagreement` prints: each map's share of the differing windows that is right gets its exact
    interval at 1 - (1 - conf) / 2, and the difference names a map when it excludes 0. The count right in a is
    hypergeometric, and the count right in b given it is hypergeometric over the rest."""
    N, D, n, KA, KB = (int(v) for v in (N, D, n, KA, KB))
    if not (0 <= KA and 0 <= KB and KA + KB <= D and 1 <= n <= D <= N):
        raise ValueError(f"which_map_probability needs 0 <= KA, KB, KA + KB <= D and 1 <= n <= D <= N; got N={N}, "
                         f"D={D}, n={n}, KA={KA}, KB={KB}")
    if n == D:                                                 # a census of the differing windows
        return float(KA > KB), float(KB > KA)
    s = 1 - (1 - float(est._level(conf))) / 2
    xs, px = _hyper_pmf(D, KA, n)
    rows = [(int(x), pxx, *_hyper_pmf(D - KA, KB, int(n - x))) for x, pxx in zip(xs, px)]
    y0 = min(int(r[2][0]) for r in rows)
    y1 = max(int(r[2][-1]) for r in rows)
    lo_x, hi_x = _ends_range(n, D, s, int(xs[0]), int(xs[-1]))
    lo_y, hi_y = _ends_range(n, D, s, y0, y1)
    p_a = p_b = 0.0
    for i, (x, pxx, ys, py) in enumerate(rows):
        a0, a1 = int(ys[0]) - y0, int(ys[-1]) - y0 + 1
        # names a: lo_a(x) - hi_b(y) > 0; names b: lo_b(y) - hi_a(x) > 0 (each end rises with y)
        p_a += pxx * float(py[:np.searchsorted(hi_y[a0:a1], lo_x[i], side="left")].sum())
        p_b += pxx * float(py[np.searchsorted(lo_y[a0:a1], hi_x[i], side="right"):].sum())
    return float(p_a), float(p_b)


def _parity(c, D, delta):
    """c moved to the parity KA - KB = delta allows, inside [0, D - delta]; None if none."""
    top = D - delta
    if (D - c - delta) % 2:
        c = c + 1 if c + 1 <= top else c - 1
    return c if 0 <= c <= top else None


def _worst_split(prob, D, delta):
    """(the both-wrong count c0 whose split gives the smallest probability, the probability, the splits checked).
    Checked: SPLITS counts spread from 0 to D - delta; every count up to DENSE_SPLITS, where the review of 2026-10-07
    found the worst splits of nearly census budgets; then a zoom around the worst so far, 11 counts at a time over a
    span shrinking tenfold, ending with every count of the last span. prob(c0) -> probability."""
    top = D - delta
    seen = {}

    def at(c):
        c = _parity(int(round(c)), D, delta)
        if c is not None and c not in seen:
            seen[c] = prob(c)
    for c in np.linspace(0, top, SPLITS):
        at(c)
    for c in range(0, min(top, DENSE_SPLITS) + 1, 2):
        at(c)
    span = top / (SPLITS - 1) if top else 0
    while span > 0:
        best = min(seen, key=seen.get)
        a, b = max(0, best - span), min(top, best + span)
        if b - a <= 22:
            for c in range(int(a), int(b) + 1):
                at(c)
            break
        for c in np.linspace(a, b, 11):
            at(c)
        span /= 10
    best = min(seen, key=seen.get)
    return best, seen[best], seen


def plan_which_map(N, D, difference, power=POWER, both_wrong=None, two_class=False, conf=0.95,
                   max_labels=MAX_LABELS):
    """The probability, budget by budget, that labels drawn among the D differing windows name the more accurate map
    when the accuracies over the N windows compared differ by at least `difference`, and the recommended budget.

    both_wrong: the share of the differing windows wrong in both maps; None checks the splits of `_worst_split` and
    reports the worst. two_class: where two-class maps differ one of them is right, so no window is wrong in both;
    their difference in windows then has the parity of D, and the smallest one of at least `difference` is planned.
    The chance of naming the less accurate map is reported as its largest over the splits checked; the interval holds
    it below (1 - conf) / 2 at any budget."""
    N = _check_count("the number of windows compared", N)
    D = _check_count("the number of differing windows", D)
    if D > N:
        raise ValueError(f"the maps cannot differ on {D} of {N} windows")
    _check("the difference", difference, 0, 1)
    _check("the probability asked for", power, 0, 1)
    delta = max(1, int(math.ceil(difference * N - 1e-9)))     # any positive difference is at least one window
    base = {"question": "which_map", "n_population": N, "n_disagree": D, "disagree_share": D / N,
            "difference": float(difference), "power": float(power), "note": PLAN_NOTE, "interval_note": INTERVAL_NOTE}
    if two_class:
        if both_wrong not in (None, 0, 0.0):
            raise ValueError("two-class maps have no window wrong in both; the both-wrong share is 0")
        if (D - delta) % 2:
            delta += 1                                         # a two-class difference has the parity of D
    if delta > D:
        return {**base, "labels": None,
                "refusal": (f"the maps differ on {D} of {N} windows ({100 * D / N:.3g}%), so their accuracies cannot "
                            f"differ by {difference:g}; the largest possible difference is {D / N:.4g}")}
    if two_class:
        fixed, said = 0, "two-class maps: no differing window is wrong in both"
    elif both_wrong is not None:
        _check("the share wrong in both", both_wrong, 0, 1, lo_open=False, hi_open=True)
        c0 = int(round(both_wrong * D))
        if c0 > D - delta:
            raise ValueError(f"with {D} differing windows and a difference of {delta} windows, at most "
                             f"{(D - delta) / D:.4g} of them can be wrong in both maps, not {both_wrong:g}")
        fixed = _parity(c0, D, delta)
        said = f"{fixed} of the {D} differing windows wrong in both maps"
    else:
        fixed, said = None, "the worst split checked of the differing windows"
    split = lambda c: ((D - c - delta) // 2 + delta, (D - c - delta) // 2)
    top = min(D, _check_count("the largest budget", max_labels))
    wrong, worst, nsplits, scans = {}, {}, {}, {}

    def fn(b, candidates=None):
        """The worst probability at budget b over the splits checked: the full scan of _worst_split, or the
        candidates given (the worst splits of the nearest budget fully scanned, for the budgets added by refine)."""
        memo = {}

        def prob(c):
            memo[c] = which_map_probability(N, D, b, *split(c), conf)
            return memo[c][0]
        if fixed is not None:
            c, p = fixed, prob(fixed)
        elif candidates is None:
            c, p, seen = _worst_split(prob, D, delta)
            scans[b] = sorted(seen, key=seen.get)
        else:
            for cc in candidates:
                prob(cc)
            c = min(memo, key=lambda cc: memo[cc][0])
            p = memo[c][0]
        wrong[b], worst[b], nsplits[b] = max(v[1] for v in memo.values()), split(c), len(memo)
        return p

    def fn_near(b):
        if fixed is not None or not scans:
            return fn(b)
        near = min(scans, key=lambda a: abs(a - b))
        cands = set()
        for cc in scans[near][:WORST_KEPT]:
            for d in (-4, -2, 0, 2, 4):
                q = _parity(cc + d, D, delta)
                if q is not None:
                    cands.add(q)
        return fn(b, sorted(cands))
    probs, span = refine(fn_near, {b: fn(b) for b in ladder(top)}, power, every=True)
    rec = recommend(probs, power)
    out = {**base, "difference_windows": delta, "both_wrong": None if both_wrong is None else float(both_wrong),
           "two_class": bool(two_class), "planned_for": said,
           "splits_checked_at_labels": None if rec is None else nsplits[rec], "labels": rec,
           "splits_note": (None if fixed is not None else
                           "on the ladder every split of the scan is checked; at the budgets added between and above "
                           f"them, the {WORST_KEPT} worst splits of the nearest budget scanned and their neighbours"),
           "probability_at_labels": None if rec is None else probs[rec],
           "wrong_verdict_probability_at_labels": None if rec is None else wrong[rec],
           "worst_split_at_labels": None if rec is None else
           {"right_in_a": worst[rec][0], "right_in_b": worst[rec][1], "right_in_neither": D - sum(worst[rec])},
           "probability_by_budget": {str(b): p for b, p in probs.items()},
           "wrong_verdict_by_budget": {str(b): wrong[b] for b in probs},
           "every_budget_checked": None if span is None else list(span), "runs_reaching_power": runs(probs, power),
           "checked_up_to": top, "census_checked": top >= D}
    if rec is None:
        out["refusal"] = (f"no budget checked up to {top} names the more accurate map with probability {power:g} at a "
                          f"difference of {difference:g}" + ("" if top >= D else "; check larger budgets"))
    return out


# ----------------------------------------------------------------------------- certified zone
def _grid_level(coverage, grid):
    for c in grid:
        if abs(c - coverage) < 1e-9:
            return c
    raise ValueError(f"the coverage {coverage} is not a level of the zone grid {list(grid)}; certify tests those "
                     "levels")


def zone_floor(coverage, alpha, delta=est.ZONE_DELTA):
    """The fewest random labels at which `certify` tests a zone of this coverage at all: below it the level is cut
    before any label is seen, since even a zone with no error among its labels could not be certified."""
    return int(math.ceil(est.min_labels_to_certify(alpha, delta) / coverage - 1e-12))


def entry_budgets(coverage, alpha, delta=est.ZONE_DELTA, grid=est.ZONE_GRID):
    """{level: budget} for the grid levels below `coverage`: the budget at which each becomes testable. At each of
    these budgets both rules can lose power: the prefix rule tests the smallest testable level first, with few labels,
    and the Bonferroni rule splits delta over one more level."""
    return {g: zone_floor(g, alpha, delta) for g in grid if g < coverage - 1e-9}


def _zone_index(N, n, coverage, alpha, delta, grid):
    cov, sizes, _ = est.zone_levels(N, n, alpha, delta, grid)
    hit = [j for j, c in enumerate(cov) if abs(c - coverage) < 1e-9]
    return (hit[0], cov, sizes) if hit else (None, cov, sizes)


def zone_level_probability(N, n, coverage, alpha, zone_error, delta=est.ZONE_DELTA, level=None, grid=est.ZONE_GRID):
    """The probability that the zone of this coverage passes its own test at `level` (delta by default) with n random
    labels, when round(zone_error n_zone) of its n_zone windows are wrong; exact, with certify's p-value and tie
    rule. It depends on the zone alone, not on how the errors spread inside it. At level delta it bounds from above
    the chance that the prefix rule certifies the zone; at delta / J, J the levels testable at n, it is the chance
    that the Bonferroni rule accepts it."""
    N, n = int(N), int(n)
    j, cov, sizes = _zone_index(N, n, coverage, alpha, delta, grid)
    if j is None:
        return 0.0
    nz = sizes[j]
    level = delta if level is None else level
    Kz = int(round(zone_error * nz))
    bs, pb = _hyper_pmf(N, nz, n)
    total = 0.0
    for b, pbb in zip(bs, pb):
        b = int(b)
        if b == 0:
            continue

        def passes(k):
            return est._at_most(est.zone_pvalue(k, b, nz, alpha), level,
                                lambda: est.zone_pvalue_exact(k, b, nz, alpha), nz)
        if not passes(0):
            continue
        lo, hi = 0, b + 1                                      # the p-value rises with k: passes(lo), not past hi
        while hi - lo > 1:
            m = (lo + hi) // 2
            lo, hi = (m, hi) if passes(m) else (lo, m)
        total += pbb * est.hypergeom_cdf(lo, nz, Kz, b)
    return float(total)


def bonferroni_probability(N, n, coverage, alpha, zone_error, delta=est.ZONE_DELTA, grid=est.ZONE_GRID):
    """The probability that certify's Bonferroni rule accepts the zone of this coverage with n random labels, at its
    level delta / J (J the levels testable at n, as certify counts them); exact, whatever the errors' spread."""
    J = len(est.zone_levels(int(N), int(n), alpha, delta, grid)[0])
    return zone_level_probability(N, n, coverage, alpha, zone_error, delta, level=est._level(delta) / max(J, 1),
                                  grid=grid)


def zone_prefix_probability(N, n, coverage, alpha, zone_error, delta=est.ZONE_DELTA, grid=est.ZONE_GRID,
                            draws=ZONE_DRAWS, seed=0):
    """(probability, standard error) that the prefix rule certifies at least this coverage with n random labels when
    the zone's errors are spread evenly in confidence order: every zone of the grid up to it, of m windows, holds
    floor(m K / n_zone) of the zone's K errors. Simulated with certify's p-value and exact tie rule.

    Why this is the least the prefix rule achieves on any map whose more confident zones of the grid are wrong no
    more often than this one: such a map holds at most floor(m K / n_zone) errors in each of them, so it is reached
    from the even spread by moving errors to less confident rings. Moving one error from a window x to a less
    confident window y never raises any zone's p-value, in distribution: pair each labelled set L with the set that
    swaps x and y in it, which is as likely. A zone that holds both windows or neither keeps its labels and its
    errors; a zone that holds x but not y gets, on the paired set, either one error fewer among the same number of
    labels or the same errors among one label more, and both lower its p-value. The labels in each ring, and the
    errors among them, are drawn exactly: ring counts are multivariate hypergeometric, and the errors among a ring's
    labels hypergeometric, because positions inside a ring do not matter to any zone."""
    N, n = int(N), int(n)
    j, cov, sizes = _zone_index(N, n, coverage, alpha, delta, grid)
    if j is None:
        return 0.0, 0.0
    sizes = [int(z) for z in sizes[:j + 1]]
    nz = sizes[-1]
    Kz = int(round(zone_error * nz))
    edges = [0] + sizes
    ring = np.diff(edges)
    ring_err = np.diff([m * Kz // nz for m in edges])
    if (ring <= 0).any():
        raise ValueError(f"two levels of the grid give the same zone on a map of {N} windows")
    rng = np.random.default_rng(seed)
    labels = rng.multivariate_hypergeometric(np.append(ring, N - nz), n, size=int(draws))[:, :-1]
    errs = np.column_stack([rng.hypergeometric(int(e), int(r - e), labels[:, t]) if e else np.zeros(int(draws), int)
                            for t, (r, e) in enumerate(zip(ring, ring_err))])
    b, k = np.cumsum(labels, axis=1), np.cumsum(errs, axis=1)
    ok = np.ones(int(draws), bool)
    for t, size in enumerate(sizes):
        pairs, inv = np.unique(np.stack([b[:, t], k[:, t]], axis=1), axis=0, return_inverse=True)
        passes = np.array([est._at_most(est.zone_pvalue(int(kk), int(bb), size, alpha), delta,
                                        lambda: est.zone_pvalue_exact(int(kk), int(bb), size, alpha), size)
                           for bb, kk in pairs], bool)
        ok &= passes[np.asarray(inv).ravel()]
    q = float(ok.mean())
    return q, float(math.sqrt(max(q * (1 - q), 1e-12) / draws))


def plan_zone(N, coverage, alpha, zone_error=None, power=POWER, delta=est.ZONE_DELTA, grid=est.ZONE_GRID,
              max_labels=MAX_LABELS, draws=ZONE_DRAWS, seed=0):
    """How many random labels `certify` needs for a zone of this coverage at (alpha, delta), budget by budget.

    Always: the floor, the budget from which the coverage is tested at all, and the entry budgets of the smaller
    levels. With zone_error, the rate the user expects the zone to be wrong (at most alpha): per budget, the
    Bonferroni rule's probability of certifying it (exact, any spread), the most the prefix rule can reach (exact,
    any spread), and what the prefix rule reaches when every more confident zone of the grid is wrong no more often
    (simulated; not simulated where the most it can reach already falls short); each with its recommended budget."""
    N = _check_count("the number of windows", N)
    _check("alpha", alpha, 0, 1)
    _check("delta", delta, 0, 1)
    _check("the probability asked for", power, 0, 1)
    c = _grid_level(coverage, grid)
    floor = zone_floor(c, alpha, delta)
    top = min(N, _check_count("the largest budget", max_labels))
    entries = {g: e for g, e in entry_budgets(c, alpha, delta, grid).items() if e <= top}
    out = {"question": "zone", "n_population": N, "coverage": c, "alpha": float(alpha), "delta": float(delta),
           "power": float(power), "zone_windows": max(1, int(round(c * N))),
           "min_labels_inside": est.min_labels_to_certify(alpha, delta), "labels_to_test": floor,
           "entry_budgets": {f"{g:g}": e for g, e in entries.items()},
           "zone_error": None if zone_error is None else float(zone_error), "checked_up_to": top, "note": PLAN_NOTE}
    if _zone_index(N, N, c, alpha, delta, grid)[0] is None:
        out["refusal"] = (f"on a map of {N} windows certify never tests this zone at alpha {alpha:g}: even labelling "
                          f"every window puts fewer than {out['min_labels_inside']} labels inside it")
        return out
    if zone_error is None:
        out["unplanned"] = ("the labels needed beyond the floor depend on how often the zone is wrong, which only "
                            "the labels measure; state the rate you expect to plan them")
        return out
    _check("the expected zone error rate", zone_error, 0, 1, lo_open=False, hi_open=True)
    if zone_error > alpha:
        out["refusal"] = (f"a zone wrong {zone_error:g} of the time is wrong more than alpha {alpha:g}; no budget "
                          "certifies it, and a rule that did would be wrong")
        return out
    if floor > top:
        out["refusal"] = f"the zone is tested only from {floor} labels, more than the {top} checked"
        return out
    budgets = sorted(set([b for b in ladder(top) if b >= floor] + [floor]
                         + [e for e in entries.values() if e >= floor]
                         + [e - 1 for e in entries.values() if e - 1 >= floor]))
    upper, flat_se = {}, {}
    up = lambda b: upper.setdefault(b, zone_level_probability(N, b, c, alpha, zone_error, delta, grid=grid))

    def even(b):
        if up(b) < power:
            flat_se[b] = None
            return None                                        # at most up(b), below power: not simulated
        q, flat_se[b] = zone_prefix_probability(N, b, c, alpha, zone_error, delta, grid, draws, seed)
        return q
    bf = lambda b: bonferroni_probability(N, b, c, alpha, zone_error, delta, grid)
    bonf, bonf_span = refine(bf, {b: bf(b) for b in budgets}, power, every=True)
    flat, _ = refine(even, {b: even(b) for b in budgets}, power)
    budgets = sorted(set(bonf) | set(flat))
    for b in budgets:                                          # the exact figures at every budget checked; the
        up(b)                                                  # simulation only at its own (ladder, entries, refine)
        if b not in bonf:
            bonf[b] = bf(b)
    rec_b = recommend(bonf, power)
    if bonf_span is not None and (rec_b is None or not bonf_span[0] <= rec_b <= bonf_span[1]):
        bonf_span = None
    out.update({
        "budgets_checked": budgets,
        "bonferroni": {"labels": recommend(bonf, power), "probability_by_budget": {str(b): bonf[b] for b in budgets},
                       "every_budget_checked": None if bonf_span is None else list(bonf_span),
                       "runs_reaching_power": runs(bonf, power),
                       "exact": True, "holds_for": "any spread of the errors inside the zone"},
        "prefix_most": {"probability_by_budget": {str(b): upper[b] for b in budgets}, "exact": True,
                        "below_power_at": [b for b in budgets if upper[b] < power],
                        "holds_for": "any spread of the errors: the prefix rule certifies the zone only if the zone's "
                                     "own test passes"},
        "prefix_even": {"labels": recommend(flat, power), "probability_by_budget": {str(b): flat[b] for b in sorted(flat)},
                        "runs_reaching_power": runs(flat, power),
                        "standard_error_by_budget": {str(b): flat_se[b] for b in sorted(flat)}, "draws": int(draws),
                        "exact": False, "holds_for": "maps whose more confident zones of the grid are wrong no more "
                                                     "often than this zone"}})
    for k in ("bonferroni", "prefix_even"):
        rec = out[k]["labels"]
        out[k]["probability_at_labels"] = None if rec is None else out[k]["probability_by_budget"][str(rec)]
    return out

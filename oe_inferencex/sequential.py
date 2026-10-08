"""A trusted zone that stays certified however often the reviewer looks (sequential certify).

`estimate.certify_zone` tests each zone once, at a budget fixed before any label: its guarantee holds for that one
run. A reviewer who labels, certifies, adds labels and certifies again runs a new test each time, and the chance that
some run certifies a wrong zone grows with the runs (exp96's audit: 21% to 30% on six maps where one run is wrong at
most 12.25%). This module certifies from labels read in the random order they were drawn, so that the chance that ANY
look certifies a zone wrong more than alpha of the time is at most delta, however many looks there are and whatever
decides when labelling stops.

Each zone of the grid is tested by an e-process: a nonnegative number computed from the labels that fall inside it,
whose expected value never rises above 1 while the zone is wrong more than alpha of the time. By Ville's inequality
(Ville 1939) it then ever reaches 1/delta with probability at most delta. The e-process here is a mixture of
likelihood ratios against the boundary of the null (Robbins's method of mixtures, 1970): the likelihood of the labels
under a zone with M wrong windows over that under one with M0 = floor(alpha n) + 1, averaged over M = 0 .. M0 - 1 with
equal weights. It is related to the prior-posterior ratio of Waudby-Smith and Ramdas (2020) for sampling without
replacement, but their Proposition 2.1 needs a prior with mass on every count, and this one has none on the null;
its validity on the whole null rests on the argument below, checked in exact arithmetic in tests/test_sequential.py.
With t labels inside a zone of n windows, k of them wrong:

    e(t, k) = sum over M < M0 of C(M, k) C(n - M, t - k) / (M0 C(M0, k) C(n - M0, t - k))
            = #{(t+1)-subsets of n+1 items holding more than k of M0 marked ones} / (M0 C(M0, k) C(n - M0, t - k))

the sum over the upper index counting (t+1)-subsets by their (k+1)-th smallest item (Graham, Knuth and Patashnik,
Concrete Mathematics, eq. 5.26; tests/test_sequential.py checks the two forms against each other in integers). Each
term's next factor has conditional mean at most 1 under any count M' >= M0, so e is a supermartingale on the whole
null, not only at M0. The equal weights are a choice: a single term (M = 0 alone) passes a zone with no wrong label
after as few labels as the one-look test, but fails for good at its first wrong label; the equal weights need more
labels on a clean zone (71 against 45 at alpha 0.05, delta 0.1) and tolerate wrong ones (Waudby-Smith and Ramdas
2020, section 2.3, on the prior as a trade-off; ALPHA, Stark 2023, adapts the alternative as labels arrive).

A zone passes once its e-process has reached 1/delta at some look; it stays passed. The zones are tested in a fixed
sequence (Bauer 1991; Learn then Test, Angelopoulos et al. 2025) from the anchor, the smallest zone tested, fixed
before any label is read, outward: the certified zone is the largest one such that it and every zone between it and
the anchor have passed. A zone wrong more than alpha is certified only if the first such zone at or after the anchor
has passed, which happens on at most delta of draws over the whole sequence of looks; a fixed sequence read at several
looks keeps its level only when every test in it is valid at every look (Tamhane, Mehta and Liu 2010), which the
e-processes are. A passed zone stays passed, so more labels never certify less. The closest published method is the
anytime-valid risk control of Xu, Karampatziakis and Mineiro (2024), one e-process per threshold, a fixed sequence
from the safe end and the same proof; it assumes the risk is monotone in the threshold, reads the e-process now rather
than its largest value, and samples with replacement. Here the zones' error rates need not be monotone, which is why
the sequence starts from an anchor, the e-processes are exact for sampling without replacement, and passes are sticky.
The anchor cannot move to smaller zones as labels accrue without splitting delta (a weighted fallback or graphical
procedure, Bretz et al. 2009, could hold part of delta for a smaller zone); this module does not.
"""
import math
from fractions import Fraction

import numpy as np

from . import estimate as est


def _null_count(n, alpha):
    """M0, the smallest count of wrong windows that makes a zone of n windows wrong more than alpha of the time;
    n + 1 when no count can (the null is empty)."""
    return math.floor(alpha * n) + 1


def _log_tail(k, n, N, K, upper):
    """log P(X >= k) when `upper`, else log P(X <= k), for X hypergeometric (n draws from N, K marked), summed in log
    space along the pmf's ratio recurrence as `estimate._hyper_tail` sums it, but never exponentiated: a tail of
    1e-400 is a finite log, not 0 (review of 8 October 2026: the probability underflowed and e read as 0)."""
    lo, hi = max(0, n - (N - K)), min(n, K)
    if upper:
        if k <= lo:
            return 0.0
        if k > hi:
            return -math.inf
        x = np.arange(k, hi, dtype=np.float64)
        r = (K - x) * (n - x) / ((x + 1.0) * (N - K - n + x + 1.0))
    else:
        if k >= hi:
            return 0.0
        if k < lo:
            return -math.inf
        x = np.arange(k, lo, -1, dtype=np.float64)
        r = x * (N - K - n + x) / ((K - x + 1.0) * (n - x + 1.0))
    lead = est._log_choose(K, k) + est._log_choose(N - K, n - k) - est._log_choose(N, n)
    if r.size == 0:
        return min(0.0, lead)
    cs = np.cumsum(np.log(r))
    m = max(0.0, float(cs.max()))
    return min(0.0, lead + m + math.log(math.exp(-m) + float(np.exp(cs - m).sum())))


def _log_upper_count(k, t, n, M0):
    """log #{(t+1)-subsets of n+1 items with more than k of M0 marked}: log C(n+1, t+1) + log P(X >= k+1), X the
    marked items among t+1 drawn without replacement from n+1. Where P(X >= k+1) is above one half, it is taken as
    1 - P(X <= k), the shorter and more accurate sum."""
    lu = _log_tail(k + 1, t + 1, n + 1, M0, upper=True)
    if lu == -math.inf:
        return -math.inf
    if lu > math.log(0.5):
        q = math.exp(_log_tail(k, t + 1, n + 1, M0, upper=False))
        lp = math.log1p(-q) if q < 1.0 else lu
    else:
        lp = lu
    return est._log_choose(n + 1, t + 1) + lp


def log_evalue(t, k, n, alpha):
    """log e(t, k) for a zone of n windows with t labels inside, k of them wrong, against 'wrong more than alpha';
    -inf when k already reaches the null count, +inf when no null count fits the labels (the zone is known good)."""
    t, k, n = int(t), int(k), int(n)
    if not 0 <= k <= t <= n:
        raise ValueError(f"log_evalue needs 0 <= k <= t <= n, got t={t}, k={k}, n={n}")
    M0 = _null_count(n, alpha)
    if M0 > n or t - k > n - M0:                        # no zone of n with M0 wrong holds t - k right labels
        return math.inf
    if k >= M0:
        return -math.inf
    if t == 0:
        return 0.0
    return (_log_upper_count(k, t, n, M0) - math.log(M0) - est._log_choose(M0, k)
            - est._log_choose(n - M0, t - k))


def evalue_exact(t, k, n, alpha):
    """e(t, k) in integer arithmetic, as a Fraction (math.inf when the null cannot hold)."""
    t, k, n = int(t), int(k), int(n)
    M0 = _null_count(n, alpha)
    if M0 > n or t - k > n - M0:
        return math.inf
    if k >= M0:
        return Fraction(0)
    lo, hi = max(0, t + 1 - (n + 1 - M0)), min(t + 1, M0)
    if k + 1 - lo <= hi - k:                            # the complement is the shorter sum (review of 8 October 2026)
        count = math.comb(n + 1, t + 1) - sum(math.comb(M0, x) * math.comb(n + 1 - M0, t + 1 - x)
                                              for x in range(lo, k + 1))
    else:
        count = sum(math.comb(M0, x) * math.comb(n + 1 - M0, t + 1 - x) for x in range(k + 1, hi + 1))
    return Fraction(count, M0 * math.comb(M0, k) * math.comb(n - M0, t - k))


def reaches(t, k, n, alpha, delta):
    """e(t, k) >= 1 / delta, the comparison redone in integer arithmetic where the float lies within the tie
    window of the level, so that every system decides alike (as `estimate._at_most` does for p-values)."""
    le = log_evalue(t, k, n, alpha)
    level = est._level(delta)
    thr = -math.log(float(level))
    if math.isinf(le):
        return le > 0
    if abs(le - thr) > est._tie(n):
        return le >= thr
    e = evalue_exact(t, k, n, alpha)
    return e == math.inf or e * level >= 1


def min_labels_sequential(alpha, delta=est.ZONE_DELTA):
    """The fewest labels a zone must hold, none of them wrong, before its e-process can reach 1/delta, in the
    large-zone limit: the smallest t with (1 - (1 - alpha)^(t+1)) / ((t + 1) alpha (1 - alpha)^t) >= 1/delta. It is
    about 1.6 times `estimate.min_labels_to_certify` (71 against 45 at alpha 0.05, delta 0.1). That is the price of
    this mixture's equal weights, which keep a zone certifiable after wrong labels, not of reading the test at every
    label: an e-process on the single alternative M = 0 passes a clean zone at 45 but never after a wrong label."""
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    target = -math.log(delta)
    l1a = math.log1p(-alpha)
    t = est.min_labels_to_certify(alpha, delta)          # the e-process needs at least what the one-look test needs
    while True:
        le = (math.log(-math.expm1((t + 1) * l1a)) - math.log(t + 1) - math.log(alpha) - t * l1a)
        if le >= target:
            return t
        t += 1


def level_path(draw, wrong_inside, n, alpha, delta=est.ZONE_DELTA):
    """One zone's e-process along the labels that fell inside it, in the order they were drawn.

    draw         : the label count (1-based, over the whole sample) at which each label inside the zone was drawn
    wrong_inside : 0/1 per label inside the zone, in the same order
    Returns {"t", "k", "log_e" (now), "log_e_max" (the largest value reached), "passed", "passed_at"}: passed_at is
    the label count at which e first reached 1/delta, None if it has not. A right label raises every likelihood
    ratio in the mixture and a wrong one lowers it, so e peaks just before a wrong label or at the last label, and
    the largest value is found among those states alone."""
    w = np.asarray(wrong_inside, dtype=np.int64).ravel()
    draw = np.asarray(draw, dtype=np.int64).ravel()
    t_end, k_end = int(w.size), int(w.sum())
    before = np.flatnonzero(w == 1)                       # the in-zone index of each wrong label
    states = [(int(i), int(j)) for j, i in enumerate(before)] + [(t_end, k_end)]   # (t, k) just before each, and now
    out = {"t": t_end, "k": k_end, "log_e": log_evalue(t_end, k_end, n, alpha), "passed": False, "passed_at": None}
    out["log_e_max"] = max(log_evalue(t, k, n, alpha) for t, k in states)
    prev_t = 0
    for t, k in states:
        if t > prev_t and reaches(t, k, n, alpha, delta):
            lo, hi = prev_t, t                            # e rises with t at fixed k: the first t in (lo, hi] that reaches
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if reaches(mid, k, n, alpha, delta):
                    hi = mid
                else:
                    lo = mid
            out.update({"passed": True, "passed_at": int(draw[hi - 1])})
            break
        prev_t = t + 1                                    # the wrong label itself starts the next run
    return out


def _finite(x):
    return float(x) if math.isfinite(x) else None


SEQUENTIAL_NOTE = ("sequential rule: each zone from the anchor outward is tested by an e-process on the labels in the "
                   "order they were drawn, and passes once it reaches 1/delta; the statement holds at every look "
                   "together, so it fails on at most delta of samples however often certify is run on this sample and "
                   "whenever labelling stops, and a zone once certified stays certified")


def sequential_order(n_population, seed):
    """A random order of the population, the same on every system and every numpy release: the windows sorted by
    64-bit keys read straight from the PCG64 bit generator seeded with `seed` (SeedSequence and PCG64 are fixed
    algorithms; Generator methods such as permutation may change their streams between releases). Any prefix of it
    is a simple random sample, and a longer prefix extends a shorter one, which is what lets a reviewer add labels."""
    n = int(n_population)
    keys = np.random.PCG64(int(seed)).random_raw(n)
    return np.lexsort((np.arange(n), keys))


ANCHOR_CAP = 0.25                 # the default anchor is at most a quarter of the map (exp97)


def anchor_coverage(first_budget, alpha, delta=est.ZONE_DELTA, grid=est.ZONE_GRID, cap=ANCHOR_CAP):
    """The default anchor: the smallest grid coverage whose zone expects `min_labels_sequential` labels at the
    sample's first budget, so that the anchor zone can be certified at the first look if none of its labels is
    wrong; but at most `cap`. Without the cap a small first budget fixes the anchor at a large zone, which on a map
    whose error rate passes alpha inside it can never be certified, however many labels follow. Designed on
    synthetic maps before exp97 read any cell: at alpha 0.02 a first budget of 300 put the anchor at 0.6 on a map
    certifiable to 0.63, and the mean certified share over budgets 50 to 3,000 was 0.039, against 0.145 with the
    anchor at 0.25. It depends on the first budget, alpha and delta alone, all fixed before any label is read."""
    c = min_labels_sequential(alpha, delta) / max(int(first_budget), 1)
    fit = [float(g) for g in grid if g >= c - 1e-12 and g <= cap + 1e-12]
    if fit:
        return fit[0]
    under = [float(g) for g in grid if g <= cap + 1e-12]
    return under[-1] if under else float(grid[0])


def certify_zone_sequential(margin, indices, wrong, alpha, delta=est.ZONE_DELTA, anchor=None, first_budget=None,
                            grid=est.ZONE_GRID, valid=None, seed=None):
    """The largest share of the map, from the anchor zone outward, certified wrong at most `alpha` of the time so
    that the statement fails with probability at most `delta` over the whole sequence of looks.

    indices      : the labelled windows IN THE ORDER THEY WERE DRAWN, a prefix of a random order of the valid windows
                   (a `sample --design sequential` CSV read from the top); a reviewer who adds labels extends the prefix
    wrong        : 0/1 per labelled window, in the same order
    anchor       : the coverage of the smallest zone tested, fixed before any label is read; or `first_budget`, from
                   which `anchor_coverage(first_budget, alpha, delta)` gives it (alpha and delta then fixed in advance)
    seed         : the seed of `sequential_order` that drew the sample, when known: the labels are then checked to be
                   the start of that order and the enrichment check is skipped, since the order, not the labels'
                   position, shows that they were drawn at random. Without it the enrichment check runs at every look
                   and can refuse a genuine prefix by chance (about 5% of orders at some look, nearly all at the first
                   label; review of 8 October 2026).

    Returns the zone (`coverage`, `n_zone`, `threshold`, the tie counts and `zone_indices_in_order`, as certify_zone),
    `rule` "sequential", `alpha`, `delta`, `anchor`, `n_population`, `n_labelled`, `min_labels_to_certify` (here
    `min_labels_sequential`), `scope`, `note`, and per level from the anchor: `n_labelled_inside`, `n_wrong_inside`,
    `log_e_value` (now) and `log_e_value_max` (the largest reached; null where infinite, with `known` saying why),
    `passed` and `passed_at_label` (the zone's own e-process; it carries no guarantee alone) and `certified` (the
    rule's decision: passed, with every level between it and the anchor). There is no p-value or upper bound per
    level, and no level cut: levels below the anchor are not tested."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    idx = np.asarray(indices, int).ravel()
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} labelled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got {alpha}, {delta}")
    if anchor is None:
        if first_budget is None:
            raise ValueError("the anchor is fixed before any label: give `anchor`, or the sample's first budget")
        anchor = anchor_coverage(first_budget, alpha, delta, grid)
    grid = [float(g) for g in grid]
    if not grid[0] - 1e-12 <= anchor <= grid[-1] + 1e-12:
        raise ValueError(f"anchor must be a coverage between {grid[0]:g} and {grid[-1]:g}, got {anchor}")
    chk = est.review_set_check(idx, margin, valid)
    order_ok = False
    if seed is not None:
        vmask = (np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()) & np.isfinite(margin)
        pop = np.flatnonzero(vmask)
        if idx.size > pop.size or not np.array_equal(pop[sequential_order(pop.size, seed)[:idx.size]], idx):
            raise ValueError(f"these {idx.size} labelled windows are not the start of the order seed {seed} gives on this "
                             "map; label the sequential sample from the top, in its order")
        order_ok = True
    if chk["n_outside_population"]:
        raise ValueError(f"{chk['n_outside_population']} labelled window(s) are outside the valid map")
    if chk["n_duplicated"]:
        raise ValueError(f"{chk['n_duplicated']} window(s) appear more than once")
    if chk["looks_like_a_review_set"] and not order_ok:
        raise ValueError(f"these {idx.size} windows sit at a mean suspicion percentile of "
                         f"{chk['mean_suspicion_percentile']:.2f}, above {chk['threshold']:.2f}: an enriched set, not a "
                         "random sample, and a zone certified on it would be wrong. Draw the sample at random.")
    order, pos = est.zone_order(margin, valid)
    N = int(order.size)
    j0 = next(j for j, g in enumerate(grid) if g >= anchor - 1e-12)
    cov = grid[j0:]
    sizes = [max(1, int(round(c * N))) for c in cov]
    p = pos[idx]
    draw = np.arange(1, idx.size + 1)
    out = {"rule": "sequential", "alpha": float(alpha), "delta": float(delta), "n_population": N,
           "n_labelled": int(idx.size), "anchor": float(cov[0]),
           "min_labels_to_certify": min_labels_sequential(alpha, delta), "levels": [], "coverage": None,
           "n_zone": None, "threshold": None, "scope": est.SCOPE_CERTIFY}
    out["order_verified"] = order_ok
    best, prefix = None, True
    for c, s in zip(cov, sizes):
        inz = p < s
        r = level_path(draw[inz], wrong[inz].astype(int), s, alpha, delta)
        lv = {"coverage": c, "n_zone": s, "n_labelled_inside": r["t"], "n_wrong_inside": r["k"],
              "log_e_value": _finite(r["log_e"]), "log_e_value_max": _finite(r["log_e_max"]), "passed": r["passed"],
              "passed_at_label": r["passed_at"]}
        if math.isinf(r["log_e"]):           # JSON has no infinity: say what an infinite e-value means instead
            lv["known"] = ("good: no zone of this size wrong more than alpha holds these labels" if r["log_e"] > 0
                           else "wrong more than alpha: its labels alone hold that many wrong windows")
        out["levels"].append(lv)
        prefix = prefix and r["passed"]
        lv["certified"] = prefix
        if prefix:
            best = len(out["levels"]) - 1
    if best is None:
        a = out["levels"][0]
        out["note"] = (f"no zone certified at alpha={alpha:g}, delta={delta:g} with {idx.size} labels; the anchor zone "
                       f"({cov[0]:.0%} of the map) holds {a['n_labelled_inside']} labels with {a['n_wrong_inside']} wrong, "
                       f"and with none wrong it would need {out['min_labels_to_certify']}")
        return out
    n_zone = sizes[best]
    thr = float(margin[order[n_zone - 1]])
    tied = int((margin[order] == thr).sum())
    inside = int((margin[order[:n_zone]] == thr).sum())
    out.update({"coverage": cov[best], "n_zone": n_zone, "threshold": thr, "n_tied_at_threshold": tied,
                "n_tied_inside_zone": inside, "zone_indices_in_order": order[:n_zone], "note": SEQUENTIAL_NOTE})
    if tied > inside:
        out["note"] += (f"; {tied} windows share the threshold margin and only {inside} of them are inside the zone, so "
                        "the zone is the set returned, not every window at or above the threshold")
    return out

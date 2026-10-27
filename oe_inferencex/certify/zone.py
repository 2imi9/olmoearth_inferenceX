"""The trusted zone: the part of the map wrong at most alpha of the time, by exact tests that fail on at most delta."""
import math

import numpy as np

from oe_inferencex.common.hypergeom import _at_most, _hyper_tail_exact, _level, hypergeom_cdf
from oe_inferencex.common.sampling import _condition_guards, review_set_check

SCOPE_CERTIFY = ("The zone's error rate is certified over all its windows together. Where part of the map was predicted "
                 "from other inputs, that part of the zone can be wrong more often than the rest. If the model was not "
                 "trained on those inputs, its errors there can be confident ones and fall inside the zone (exp88). To "
                 "certify each input condition on its own, draw the sample with --condition.")
FAMILY_NOTE = ("Certified per input condition. Each of the {L} conditions with at least {b1} labels is tested at delta "
               "{d:g}, so all their statements hold together except on at most {delta:g} of samples. On that event the "
               "certified windows taken together are wrong at most {alpha:g} of the time. Conditions with fewer labels "
               "are not tested. Outside the certified windows nothing is certified.")
FAMILY_NOTE_ONE = ("Certified per input condition. The one condition with at least {b1} labels is tested at delta {d:g}, "
                   "so its statement fails on at most {delta:g} of samples. When it holds, the certified windows are "
                   "wrong at most {alpha:g} of the time. Conditions with fewer labels are not tested. Outside the "
                   "certified windows nothing is certified.")


# ----------------------------------------------------------------------------- a trusted zone with a guarantee (exp80)
# Which part of the map is wrong at most alpha of the time, stated so that the statement itself fails with
# probability at most delta over the reviewer's random draw. Preregistered in docs/plan/trust_zone.md. The zone at
# coverage c is the c most confident share of the valid windows; a simple random sample of the map restricted to
# that zone is a simple random sample of the zone, so every test below is an exact hypergeometric test and nothing
# is approximated.
ZONE_GRID = tuple(round(j / 20, 2) for j in range(1, 21))
ZONE_DELTA = 0.10
ZONE_RULES = ("prefix", "bonferroni", "plugin", "target")
# The target rule (1.16.0): a zone fixed before labelling holds TARGET_WEIGHT of delta and the smaller zones the rest,
# tested as a fallback procedure (Wiens 2003; Wiens and Dmitrienko 2005). With the target right it needs about half
# the labels of the Bonferroni rule on any spread of the errors (plan.target_probability); the design research of 24
# October 2026 chose 0.8, which costs 9% over putting all of delta on the target and keeps a fallback for a target
# that was too optimistic.
TARGET_WEIGHT = 0.8
TARGET_NOTE = ("target rule: the {target:g}% zone was fixed before labelling and holds {w:g} of delta, the smaller "
               "zones share the rest, and a zone that passes hands its share up to the next larger one (a fallback "
               "procedure, Wiens 2003), valid on any map whatever the shape of its error rate; a zone larger than the "
               "target is certified only through that hand-up, and a target that was too optimistic leaves the smaller "
               "zones the rest of delta")
# when no smaller zone is tested (the target is the smallest level the budget tests) the target holds all of delta
TARGET_ALONE_NOTE = ("target rule: the {target:g}% zone was fixed before labelling and, no smaller zone being tested "
                     "at this budget, holds all of delta, so there is no fallback; a zone that passes hands its share "
                     "up to the next larger one (a fallback procedure, Wiens 2003), valid on any map whatever the shape "
                     "of its error rate")
ZONE_CUTS = ("standard", "ramp")
ZONE_CUT = "standard"             # the level cut; "ramp" is an option (exp95, exp96 and their audit: zone_cut)
# The note a prefix certification carries. The prefix rule is fixed-sequence testing (Angelopoulos et al. 2025, Learn
# then Test): the levels are ordered before any label is read and each p-value is exact, so a zone wrong more than
# alpha of the time is certified only if the first such level in the order passes its test, which happens on at
# most delta of samples, on any map. Until 1.3.1 the note said the rule was valid only if the zone's error rate does
# not fall as the zone grows; it needs no such assumption. tests/test_trust_zone.py enumerates every error pattern of
# small maps, monotone or not, and every draw, for both rules.
PREFIX_NOTE = ("prefix rule: fixed-sequence testing, valid on any map whatever the shape of its error rate; it stops at "
               "the first zone it cannot certify, so it certifies little when the most confident windows hold many "
               "errors, where the bonferroni rule can certify more; choose the rule before reading the labels, since "
               "running the other rule after seeing this zone is a second test")


def zone_edge(threshold, below):
    """The zone's edge as a decimal to print: the fewest decimals, four at least, whose value v has below < v <=
    threshold, so that "confidence >= v" takes the zone's windows and none outside it (`below` is the most confident
    window outside the zone, None when the zone holds every window). Rounded to the nearest fourth decimal, a threshold
    of 0.72453757 was printed 0.7245 and took in 31 windows outside a 198,000-window zone, and 0.43005 was printed
    0.4301, which left the threshold's own window out (the bug hunt of 2026-10-25). Where a window outside the zone ties
    with the threshold no value parts them; the tie counts say so."""
    from decimal import ROUND_FLOOR, Decimal, localcontext
    t = Decimal(repr(float(threshold)))
    with localcontext() as ctx:
        ctx.prec = 400              # a float's integer digits and 17 decimals: 28 failed from 1e24 (the review of 1.17.2)
        for d in range(4, 18):
            v = t.quantize(Decimal(1).scaleb(-d), rounding=ROUND_FLOOR)
            if below is None or below >= threshold or float(v) > below:
                return f"{v:f}"
    return repr(float(threshold))


def _edge_of(margin, order, n_zone, tied, inside):
    """The zone's printed edge: parted from the most confident window outside the zone that lies below the threshold,
    past the windows outside it tied with the threshold, which the tie counts name. Parted from a tied window, the
    edge fell back to four decimals and took in windows below the threshold that no text counted (the review of
    1.17.2)."""
    k = n_zone + (tied - inside)
    return zone_edge(float(margin[order[n_zone - 1]]), float(margin[order[k]]) if k < order.size else None)


# The release whose rules certify runs with: this one unless verify is checking a zone an older release wrote
# (certificate._rules_of). A context variable, so another thread is not touched.
RULES_OF = __import__("contextvars").ContextVar("oe_inferencex_rules_of", default=(10 ** 9,))


def null_count(n, alpha):
    """floor(alpha n) + 1, the smallest count of n windows that is wrong more than alpha of the time, with alpha read
    as the decimal it is written as. In floats 0.29 x 100 is 28.999..., which gave 29 and put a zone wrong exactly 29%
    of the time into the null: a p-value about twice too large, valid but not the exact test the docs describe. Among
    the two-decimal alphas up to 0.50 and n up to 2,000 that is 26 (alpha, n) pairs, at alphas 0.29, 0.35 and 0.41
    (the score review of 25 October 2026; counted 41 here until the bug hunt of the same day)."""
    if RULES_OF.get() < (1, 17, 1):                  # as 1.17.0 counted, for verify of a zone it wrote
        return math.floor(alpha * n) + 1
    from fractions import Fraction
    return math.floor(Fraction(repr(float(alpha))) * int(n)) + 1


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
    K0 = null_count(n, alpha)
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
    K0 = null_count(n, alpha)                   # the same null count as zone_pvalue
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
    # Taken in floats, as every release has: where delta is (1 - alpha)^b as written (alpha 0.3, delta 0.0823543 =
    # 0.7^7) the ratio can come out a hair above b and ask for one label more. It is kept so: the count sets the level
    # cut, which decides which zones certify tests before any label is read, so an exact count is another procedure
    # (on such a sample it tests a smaller first zone and can certify less, or, on others, more), and certificates
    # made with this one must verify (the bug hunt of 2026-10-25 and the third review of 1.17.2).
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


def target_weights(n_levels, target_j, delta=ZONE_DELTA, weight=TARGET_WEIGHT):
    """Each tested level's share of delta under the target rule, as Fractions: weight * delta on level target_j,
    (1 - weight) * delta split equally over the levels below it (all of delta on the target when none is below it), and
    nothing above it, which only a passing level's hand-up reaches."""
    from fractions import Fraction
    if not 0 <= int(target_j) < int(n_levels):
        raise ValueError(f"the target is level {target_j} of {n_levels}")
    if not 0 < weight <= 1:
        raise ValueError(f"the target's share of delta must be in (0, 1], got {weight}")
    d, w = _level(delta), _level(weight)
    shares = [Fraction(0)] * int(n_levels)
    if target_j == 0:
        shares[0] = d
    else:
        shares[target_j] = w * d
        for j in range(target_j):
            shares[j] = (1 - w) * d / target_j
    return shares


def apply_zone_rule(p, b, k, alpha, delta=ZONE_DELTA, rule="prefix", n=None, weights=None):
    """Which grid levels a rule accepts, and the largest one; levels are in increasing coverage.
    prefix      accept while p <= delta from the smallest zone up, stop at the first failure: fixed-sequence
                testing (Angelopoulos et al. 2025, Learn then Test), valid on any map whatever the shape of the
                zone's error rate; weak when the most confident windows hold many errors, since it stops early
    bonferroni  accept every level with p <= delta / J, J the number of levels (Angelopoulos et al. 2025, Learn
                then Test; valid on any map); it can pass a level the prefix rule stops at, and needs a smaller
                p-value at every level
    plugin      accept every level whose sample rate k / b is at most alpha; no guarantee, the comparator
    target      a fallback procedure over `weights` (target_weights), which sum to delta: from the smallest level up,
                a level passes when p is at most its own weight plus the level of the one before it if that one
                passed (Wiens 2003). Valid on any map: a zone wrong more than alpha of the time passes only if the
                first such zone that passes does, whose level is at most the weight between it and the last such zone
                before it, and those stretches are disjoint, so the chance is at most delta
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
    elif rule == "target":
        if weights is None or len(weights) != p.size:
            raise ValueError("the target rule needs one share of delta per level (target_weights)")
        acc, carry = np.zeros(p.size, bool), 0
        for j in range(p.size):
            level = _level(weights[j]) + carry
            if level > 0 and passes(j, level):
                acc[j], carry = True, level
            else:
                carry = 0
    else:
        raise ValueError(f"rule must be one of {ZONE_RULES}, got {rule!r}")
    best = int(np.flatnonzero(acc).max()) if acc.any() else None
    return acc, best


def certify_zone(margin, indices, wrong, alpha, delta=ZONE_DELTA, rule="prefix", grid=ZONE_GRID, valid=None,
                 cut=None, target=None, target_weight=TARGET_WEIGHT):
    """The largest share of the map, taken from the most confident window down, that is wrong at most `alpha` of
    the time, certified so that the statement fails with probability at most `delta` over the reviewer's draw.

    margin  : per-window confidence, higher = more trusted; NaN and invalid windows are outside the population
    indices : the labelled windows, which must be a simple random sample of the valid windows
    wrong   : 0/1 per labelled window, in the order of `indices`
    rule    : "prefix" (fixed-sequence testing, valid on any map; certifies little when the most confident windows
              hold many errors), "bonferroni" (valid on any map; can certify more in that case and less in others),
              "plugin" (no guarantee; what a reviewer would do unaided), "target" (a fallback procedure that puts
              `target_weight` of delta on the zone of coverage `target`, a grid level fixed before any label was
              read, and the rest on the smaller zones; valid on any map, and with the target right it needs about
              half the labels of the Bonferroni rule). The target is always tested, whatever the level cut

    The guarantee is for one sample whose size was fixed before its labels were read, certified once: certifying
    again after adding labels runs a new test each time, and the chance that some run is wrong grows with the runs.

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
    if (rule == "target") != (target is not None):
        raise ValueError("the target rule needs a target coverage, fixed before labelling, and a target goes with the "
                         "target rule only")
    if rule == "target" and not any(abs(target - g) < 1e-9 for g in grid):
        raise ValueError(f"the target must be a level of the zone grid ({grid[0]:g} to {grid[-1]:g} in steps of "
                         f"{grid[1] - grid[0]:g}), got {target:g}")
    chk = review_set_check(idx, margin, valid)
    if chk["n_outside_population"]:
        raise ValueError(f"{chk['n_outside_population']} labelled window(s) are outside the valid map")
    if chk["n_duplicated"]:
        raise ValueError(f"{chk['n_duplicated']} window(s) appear more than once")
    if chk["looks_like_a_review_set"]:
        raise ValueError(f"these {idx.size} windows sit at a mean suspicion percentile of "
                         f"{chk['mean_suspicion_percentile']:.2f}, above {chk['threshold']:.2f}: an enriched set, not a "
                         "random sample, and a zone certified on it would be wrong. Draw the sample at random.")
    if chk["looks_drawn_from_the_confident_end"]:
        # until 1.16.0 only the enriched end was refused, so labels from the most confident windows were certified as
        # if random and every zone looked cleaner than it is (the design research of 24 October 2026)
        raise ValueError(f"these {idx.size} windows sit at a mean suspicion percentile of "
                         f"{chk['mean_suspicion_percentile']:.2f}, below {1 - chk['threshold']:.2f}: drawn from the "
                         "confident end of the map, not at random, and a zone certified on them would be wrong. Draw "
                         "the sample at random.")
    order, pos = zone_order(margin, valid)
    N = int(order.size)
    cut = ZONE_CUT if cut is None else cut
    cov, sizes, c_min = zone_levels(N, idx.size, alpha, delta, grid, cut)
    shares = None
    if rule == "target":                     # the target is tested whatever the cut, which reads no label either
        target = next(g for g in grid if abs(target - g) < 1e-9)
        if target not in cov:
            cov = sorted(cov + [target])
            sizes = [max(1, int(round(c * N))) for c in cov]
        shares = target_weights(len(cov), cov.index(target), delta, target_weight)
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
    acc, best = apply_zone_rule(p, b, k, alpha, delta, rule, n=sizes, weights=shares)
    for j, c in enumerate(cov):
        out["levels"].append({"coverage": c, "n_zone": sizes[j], "n_labelled_inside": int(b[j]), "n_wrong_inside": int(k[j]),
                              "p_value": float(p[j]), "upper_bound": zone_upper_bound(k[j], b[j], sizes[j], delta),
                              "accepted": bool(acc[j])})
        if shares is not None:
            out["levels"][-1]["share_of_delta"] = float(shares[j])
    if rule == "target":
        tj = cov.index(target)
        out.update({"target": float(target), "target_weight": float(target_weight),
                    "target_share_of_delta": float(shares[tj]), "target_certified": bool(acc[tj])})
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
    out.update({"coverage": cov[best], "n_zone": n_zone, "threshold": thr,
                "edge": _edge_of(margin, order, n_zone, tied, inside),
                "n_tied_at_threshold": tied, "n_tied_inside_zone": inside,
                "upper_bound": out["levels"][best]["upper_bound"], "zone_indices_in_order": order[:n_zone]})
    if rule == "plugin":
        out["note"] = "plug-in rule: no guarantee; the largest zone whose sample rate is at most alpha"
    elif rule == "prefix":
        out["note"] = PREFIX_NOTE
    elif rule == "target":
        out["note"] = (TARGET_NOTE.format(target=100 * target, w=target_weight) if cov.index(target) else
                       TARGET_ALONE_NOTE.format(target=100 * target))
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

    Inside each condition `certify_zone` checks that the labels look like neither a review list nor a draw from the
    confident end (review_set_check, both ends from 1.16.0); labels clustered by tile pass that check. A
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
    if rule == "target":
        raise ValueError("the target rule fixes one zone of the whole map before labelling and is not built per "
                         "condition; use the prefix or bonferroni rule")
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
        if chk is not None and (chk["looks_like_a_review_set"] or chk["looks_drawn_from_the_confident_end"]):
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
            if not chk["looks_like_a_review_set"]:          # the confident end (1.16.0): the mirror-image reason
                r["review_set_check"] = {"mean_suspicion_percentile": chk["mean_suspicion_percentile"],
                                         "threshold": 1 - chk["threshold"], "end": "confident"}
                r["reason"] = (f"its {int(n_c[c])} labels sit at a mean suspicion percentile of "
                               f"{chk['mean_suspicion_percentile']:.2f} within the condition, below "
                               f"{1 - chk['threshold']:.2f}. They look chosen for high confidence, yet {how}; every "
                               "zone would look cleaner than it is. If these are the labels `sample` drew, unedited, "
                               "this is a rare chance draw")
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
    res = {"rule": rule, "alpha": float(alpha), "delta": float(delta), "n_population": N, "n_labelled": int(idx.size),
           "min_labels_to_certify": b1, "levels": [], "coverage": None, "n_zone": None, "threshold": None,
           "upper_bound": None, "by_condition": True, "delta_per_condition": d, "n_conditions_tested": L - len(refused),
           "certified_share_of_map": n_cert / N if zones else None, "n_certified": n_cert, "per_condition": per,
           "zone_indices_in_order": np.concatenate(zones) if zones else np.zeros(0, int), "note": note}
    if (sample.get("condition") or {}).get("by") == "map class":     # as _estimate_core marks it
        res["condition_by"] = "map class"
    return res


# What delta counts (the final review of 24 October 2026): it bounds the share of ALL samples on which the statement
# is wrong, those that certify nothing included. A reader sees only the samples that certify, and among them more can
# be wrong: in the record's exp116 (descriptive), on 46 of the 413 real-map cells where the target rule certified a
# zone, more than delta of the zones certified were wrong, where the share of all draws stayed under delta on every
# cell. It is largest where a zone's true error rate sits just above alpha.
ISSUED_NOTE = ("delta is the share of all samples, those that certify nothing included, on which a statement like this "
               "is wrong. Among the samples that do certify a zone, the share wrong can be larger, most of all when a "
               "zone's true error rate is just above alpha: there a zone is rarely certified, and when it is, it is "
               "often wrong. A certified zone is not a 1 - delta chance of being right.")
ZONE_READING_NOTE = ("The alternative reading also accepts, for each map class, the reference classes the --also-accept "
                     "file lists for it, and certifies from the same labels a second time, at the same alpha and delta: "
                     "a second test on them. If the file was fixed before any label was read, each certificate fails on "
                     "at most {delta:g} of samples for its own definition of an error, and the chance that either is "
                     "wrong is at most {either:g} (a union bound). A window the alternative accepts is not an error "
                     "under it, so its zone holds no more errors and can be larger than the strict one; its certificate "
                     "says nothing about how often the windows it accepts are wrong under the strict reading. A file "
                     "written after the labels were seen can be fitted to them, and then neither certificate holds.")
ZONE_READING_SEQUENTIAL = (" On a sequential sample the file must be the same at every look: one changed after labels "
                           "were read was written after they were seen.")

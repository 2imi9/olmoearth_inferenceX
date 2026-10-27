"""Intervals valid at every sample size for estimate's designs (1.20.0): `estimate --valid`, opt-in.

`estimate --valid` (the API's `estimate_error_rate(..., valid=True)` and `estimate_per_class(..., valid=True)`) adds,
beside the intervals estimate has always given, an interval for the error rate and for each class's user's accuracy,
producer's accuracy and share that covers at least 95% for every map and every number of labels under the designs
`sample` draws: confidence, proportional, condition, class and profile, and random, where it is the exact interval
estimate gives already. Without the option nothing here runs and no output changes.

The primitive, `linear_valid`, bounds theta = sum_j (M_j / M) mean_j(x), x 0 or 1 per window, where the population is
cut into groups j of known size M_j and the labelled windows of each group are a simple random sample of it, drawn
without replacement, independent across groups. Each group is one of three kinds:
- labelled in full (a census): its count is known and enters exactly;
- holding no label: its share is bounded, 0 at the lower end and all of it at the upper;
- labelled in part (open).
With no open group the ends are exact. With one, it is that group's exact hypergeometric interval at 1 - alpha
(`hypergeom_interval`), the others added exactly: under a random sample, and for the class design's user's accuracy,
estimate's own exact interval to the last bit. With two or more, the betting interval of `grade` over the open groups
(agency/stratified.py; Waudby-Smith and Ramdas 2020, the report's Theorem 1), whose mixture of bets is fixed by the
groups' sizes and label counts, never by the labels. Census and exact ends are the nearest float to their fraction, as
`_union_interval` rounds them; betting ends come from a bisection and move one float outward. Every interval is
widened, when needed, to hold its own point, which never lowers its coverage.

The groups are fixed by the design and the map: a design's strata, or, for a class's user's and producer's accuracy,
the windows of each stratum the map calls the class (or does not). Given how many labelled windows fall in such a
part of a stratum, they are a simple random sample of it, independent of the rest (post-stratification, Holt and Smith
1979), so each interval holds given those numbers, and so unconditionally. Which form an interval takes (census, exact,
betting) depends on the design, the map and those numbers, never on the labels.

A window marked `?` is a fixed property of the window. The lower end reads every `?` as 0, a one-sided bound at
alpha / 2 on the fixed population "every ? is 0", which lies below the truth; the upper end reads it as 1, likewise
above. Together they hold at 1 - alpha, with no monotonicity needed.

The betting interval averages exactly over the orders of a group's labels. A group whose worst case, K_MIX n (n // 2 +
1) steps for n labels, exceeds agency.stratified.MAX_WORK (about 90,000 labels in one group) is decided from n alone,
before any label is read: the whole map then takes the union bound over the strata (`_union_interval`, valid too) and
a per-class quantity is left uncomputed.

Each interval holds on its own at 95%; the intervals of one table do not hold jointly. numpy only."""
import math
from fractions import Fraction

import numpy as np

from oe_inferencex.agency.stratified import K_MIX, MAX_WORK, Design, TooLarge, betting_interval
from oe_inferencex.common.hypergeom import _float_above, _float_below, _hypergeom_fractions, _level
from oe_inferencex.estimate.intervals import _union_interval

FORM_CENSUS = "census"
FORM_EXACT = "exact"
FORM_BETTING = "betting"
FORM_UNION = "union bound"
FORM_BOX = "box"

# what a whole-map interval is, by its form (the design's words are put before it)
WHOLE_MAP_FORMS = {
    FORM_CENSUS: ("valid at every sample size: every stratum is labelled in full or holds no label, so the interval is "
                  "the exact range of the rate"),
    FORM_EXACT: ("valid at every sample size: one stratum is labelled in part, and the interval is its exact "
                 "hypergeometric interval at 95%, with any stratum labelled in full added exactly"),
    FORM_BETTING: ("valid at every sample size: the betting interval for a stratified sample drawn without replacement "
                   "(Waudby-Smith and Ramdas 2020; the interval grade gives), the strata labelled in full added "
                   "exactly"),
    FORM_UNION: ("valid at every sample size: a stratum holds too many labels for the betting interval's exact average "
                 "over their orders, so the interval is the sum of each stratum's exact interval at 1 - 0.05/L, "
                 "weighted by its share of the map (union bound)"),
}
DESIGN_WORDS = {"random": "simple random sample", "confidence": "stratified by confidence margin",
                "proportional": "stratified by confidence margin", "condition": "stratified by input condition"}
UNJUDGED_WORDS = "; a `?` window counts as right at the lower end and as wrong at the upper end"

# the printed note beside the whole-map line; it names no design by the word the class wording rewrites
VALID_NOTE = ("--valid: the first interval covers the whole-map rate at least 95% of the time for every map and every "
              "number of labels, with no normal approximation: strata labelled in full enter exactly, one stratum "
              "labelled in part takes its exact hypergeometric interval, and two or more take the betting interval "
              "for a stratified sample drawn without replacement, the method grade uses. The line below it is the "
              "interval estimate prints without --valid, unchanged: under the confidence and proportional designs "
              "Wilson's on the design's effective sample size, which is nominal (it covered 91.3% to 94.1% on the "
              "research record's maps at 100 labels, 92.7% near a census); under `--design condition`, `class` and "
              "`profile` the union bound over the strata, which holds too; under a random sample the same exact "
              "interval. The valid interval is wider than a nominal one and can be narrower than the union bound.")
PER_CLASS_VALID_NOTE = ("--valid: each interval on a line marked valid holds at 95% on its own, for every map and every "
                        "number of labels: the user's accuracy over the windows the map calls the class in each "
                        "stratum (exact where they lie in one stratum labelled in part), the producer's accuracy from "
                        "two such intervals at 97.5% each (a box), and the share over the strata. They do not hold "
                        "jointly, and a share the map draws outside its own interval is flagged on that interval "
                        "alone. The line below each is the table estimate prints without --valid, unchanged.")
VALID_HELP = ("also an interval valid at every sample size, printed beside the one given without it, which does not "
              "change: for the error rate and, with --per-class, each class's user's accuracy, producer's accuracy and "
              "share, under every design but tiles (exact where one stratum is labelled in part, a betting interval "
              "over the strata where several are). Under the confidence and proportional designs it is wider than "
              "their nominal interval, and with --per-class it takes seconds per class")


def worst_work(n):
    """The most steps the betting interval's exact average over orders takes for one group of n labels of value 0 or
    1: K_MIX n (k + 1) at the smaller of its two counts k, at most n // 2. A fallback decided from it depends on the
    label count alone, never on the labels."""
    n = int(n)
    return K_MIX * n * (n // 2 + 1)


def _nearest(q):
    """The nearest float to a fraction in [0, 1]."""
    return min(1.0, max(0.0, float(q)))


def _counts(M, g, x_lo, x_hi):
    """The groups of linear_valid, checked: (sizes, labels per group, the count of 1s per group at the lower and at
    the upper reading)."""
    M = [int(m) for m in M]
    J = len(M)
    if J == 0 or any(m < 0 for m in M) or sum(M) == 0:
        raise ValueError(f"group sizes must be whole numbers of at least 0 with at least one window, got {M}")
    if not g.size == x_lo.size == x_hi.size:
        raise ValueError(f"one group and one value per labelled window: got {g.size}, {x_lo.size} and {x_hi.size}")
    if g.size and (int(g.min()) < 0 or int(g.max()) >= J):
        raise ValueError("a labelled window's group lies outside the groups")
    if not (np.isin(x_lo, (0.0, 1.0)).all() and np.isin(x_hi, (0.0, 1.0)).all()) or (x_lo > x_hi).any():
        raise ValueError("each value must be 0 or 1, the lower reading at most the upper")
    n = np.bincount(g, minlength=J).astype(int)
    over = [j for j in range(J) if n[j] > M[j]]
    if over:
        raise ValueError(f"group {over[0]} holds {n[over[0]]} labels but only {M[over[0]]} windows")
    k_lo = np.bincount(g, weights=x_lo, minlength=J).round().astype(int)
    k_hi = np.bincount(g, weights=x_hi, minlength=J).round().astype(int)
    return M, n, k_lo, k_hi


def _point(M, n, k_lo, k_hi):
    """The point of theta, exactly: each sampled group's share of 1s times its size, over the population; a group with
    no label adds 0 at the lower reading and its whole size at the upper. Returns (lower, upper) as fractions."""
    total = sum(M)
    p_lo = sum((Fraction(M[j] * int(k_lo[j]), int(n[j])) for j in range(len(M)) if n[j]), Fraction(0))
    p_hi = sum((Fraction(M[j] * int(k_hi[j]), int(n[j])) for j in range(len(M)) if n[j]), Fraction(0))
    p_hi += sum(M[j] for j in range(len(M)) if M[j] and not n[j])
    return p_lo / total, p_hi / total


def linear_valid(M, g, x_lo, x_hi=None, alpha=0.05):
    """theta = sum_j (M_j / M) mean_j(x), M = sum_j M_j: the groups' sizes M, each labelled window's group g and its
    value x (0 or 1), the labelled windows of each group a simple random sample of it. With x_hi, unknown values are
    bounded: the lower end reads x_lo, the upper x_hi (x_lo <= x_hi window by window). Returns {estimate (None when
    the point is a range), estimate_range, low, high, form, open_groups, census_groups, unsampled_share}. Raises
    TooLarge, from the label counts alone, when two or more groups are labelled in part and one of them is too large
    for the betting interval."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    g = np.asarray(g, dtype=int).ravel()
    x_lo = np.asarray(x_lo, dtype=float).ravel()
    same = x_hi is None
    x_hi = x_lo if same else np.asarray(x_hi, dtype=float).ravel()
    M, n, k_lo, k_hi = _counts(M, g, x_lo, x_hi)
    J, total = len(M), sum(M)
    census = [j for j in range(J) if M[j] and n[j] == M[j]]
    open_ = [j for j in range(J) if 0 < n[j] < M[j]]
    U = sum(M[j] for j in range(J) if M[j] and not n[j])
    A_lo, A_hi = sum(int(k_lo[j]) for j in census), sum(int(k_hi[j]) for j in census)
    p_lo, p_hi = _point(M, n, k_lo, k_hi)
    if not open_:
        low, high, form = _nearest(Fraction(A_lo, total)), _nearest(Fraction(A_hi + U, total)), FORM_CENSUS
    elif len(open_) == 1:
        o = open_[0]
        conf = 1 - alpha                                         # 0.95 at alpha 0.05, hypergeom_interval's default
        a = _hypergeom_fractions(int(k_lo[o]), int(n[o]), M[o], conf)[0]
        b = _hypergeom_fractions(int(k_hi[o]), int(n[o]), M[o], conf)[1]
        low, high, form = _nearest((A_lo + M[o] * a) / total), _nearest((A_hi + U + M[o] * b) / total), FORM_EXACT
    else:
        big = [j for j in open_ if worst_work(n[j]) > MAX_WORK]
        if big:
            raise TooLarge(f"a stratum of {int(n[big[0]])} labels is too large for the exact average over the orders "
                           "of its labels")
        rest = sum(M[j] for j in open_) + U
        where = {j: i for i, j in enumerate(open_)}
        sel = np.isin(g, open_)
        hidx = np.array([where[int(j)] for j in g[sel]], dtype=int)
        design = Design(hidx, [M[j] / rest for j in open_], [float(M[j]) for j in open_], W_unsampled=U / rest)
        same = same or np.array_equal(x_lo, x_hi)
        lo, hi = betting_interval(design, x_lo[sel], None if same else x_hi[sel], alpha)
        # the open part's ends are a rejected value of a bisection; combined with the census part exactly and moved
        # one float outward, so a truth that sits on an end is held
        low = max(0.0, math.nextafter(_float_below((A_lo + rest * Fraction(lo)) / total), -math.inf))
        high = min(1.0, math.nextafter(_float_above((A_hi + rest * Fraction(hi)) / total), math.inf))
        form = FORM_BETTING
    e_lo, e_hi = _nearest(p_lo), _nearest(p_hi)
    return {"estimate": e_lo if p_lo == p_hi else None, "estimate_range": [e_lo, e_hi],
            "low": min(low, e_lo), "high": max(high, e_hi), "form": form, "open_groups": len(open_),
            "census_groups": len(census), "unsampled_share": U / total}


def _reviewer_maps(fa, miss):
    """The maps estimate_error_rate applies to an interval's ends for a reviewer who errs (E0 false alarms, E1
    misses), the same as its own: in exact fractions from the float beyond the end, rounded outward."""
    E0, E1 = _level(fa), _level(miss)

    def down(x):
        if not fa:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, -math.inf))
        return max(0.0, _float_below((q - E0) / (1 - E0)))

    def up(x):
        if not miss:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, math.inf))
        return min(1.0, _float_above(q / (1 - E1)))
    return down, up


def _groups_of(sample, idx):
    """A sample's groups for the whole map: (the strata's sizes, each labelled window's stratum). A random sample, or
    a sequential one read as random, is one group."""
    design = sample.get("design")
    if design in ("random", "sequential"):
        return [int(sample["n_population"])], np.zeros(idx.size, dtype=int)
    pop = np.asarray(sample["strata_of_population"], int)
    pos = {int(v): i for i, v in enumerate(pop)}
    stray = [int(v) for v in idx if int(v) not in pos]
    if stray:
        raise ValueError(f"{len(stray)} sampled window(s), first {stray[0]}, are not in the population this sample was "
                         "drawn from")
    local = np.array([pos[int(v)] for v in idx], dtype=int)
    return [int(s) for s in sample["sizes"]], np.asarray(sample["strata"], int)[local]


def whole_map_valid(sample, wrong, unjudged=None, reviewer_false_alarm=0.0, reviewer_miss=0.0, alpha=0.05):
    """The whole map's error rate with its interval valid at every sample size, for estimate_error_rate(valid=True):
    {estimate, estimate_range (when a range), low, high, half_width, guarantee: "valid", form, method}. A `?` window
    counts right at the lower end and wrong at the upper; the reviewer's stated error bounds map both ends as
    estimate_error_rate maps its own. The tiles design has no such interval and is refused. `alpha` is for the tests;
    estimate uses 0.05."""
    design = sample.get("design")
    if design not in DESIGN_WORDS and design != "sequential":
        raise ValueError(f"--valid has no interval for the {design} design: labels taken tile by tile are not a "
                         "stratified simple random sample of windows; draw a random, confidence, proportional, "
                         "condition or class sample" if design == "tiles" else
                         f"--valid has no interval for the {design!r} design")
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    unk = np.zeros(wrong.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if unk.size != wrong.size:
        raise ValueError(f"{unk.size} unjudged flags for {wrong.size} labels")
    w_lo, w_hi = np.where(unk, 0.0, wrong), np.where(unk, 1.0, wrong)
    idx = np.asarray(sample["indices"], int).ravel()
    if idx.size != wrong.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} sampled windows")
    sizes, g = _groups_of(sample, idx)
    try:
        r = linear_valid(sizes, g, w_lo, w_hi, alpha)
    except TooLarge:
        # decided from the label counts alone (worst_work), so the choice of form never reads the labels
        J = len(sizes)
        n_h = np.bincount(g, minlength=J)
        k_lo = np.bincount(g, weights=w_lo, minlength=J).round().astype(int)
        k_hi = np.bincount(g, weights=w_hi, minlength=J).round().astype(int)
        lo = _union_interval(k_lo, n_h, sizes, alpha)[1]
        hi = _union_interval(k_hi, n_h, sizes, alpha)[2]
        p_lo, p_hi = _point([int(s) for s in sizes], n_h, k_lo, k_hi)
        e_lo, e_hi = _nearest(p_lo), _nearest(p_hi)
        r = {"estimate": e_lo if p_lo == p_hi else None, "estimate_range": [e_lo, e_hi], "low": min(lo, e_lo),
             "high": max(hi, e_hi), "form": FORM_UNION}
    down, up = _reviewer_maps(float(reviewer_false_alarm), float(reviewer_miss))
    e_lo, e_hi = down(r["estimate_range"][0]), up(r["estimate_range"][1])
    out = {"estimate": e_lo if e_lo == e_hi else None}         # nothing to bound: the value, as estimate gives it
    if out["estimate"] is None:
        out["estimate_range"] = [e_lo, e_hi]
    out.update({"low": down(r["low"]), "high": up(r["high"])})
    out["half_width"] = (out["high"] - out["low"]) / 2
    words = DESIGN_WORDS["random" if design == "sequential" else design]
    out.update({"guarantee": "valid", "form": r["form"],
                "method": f"{words}; {WHOLE_MAP_FORMS[r['form']]}" + (UNJUDGED_WORDS if unk.any() else "")})
    return out


# ----------------------------------------------------------------------------------------------- per class
def _quantity(r):
    """A per-class quantity as written: its point (a range when a part of a stratum holds no label), its ends and
    form."""
    out = {"estimate": r["estimate"]}
    if r["estimate"] is None:
        out["estimate_range"] = list(r["estimate_range"])
    out.update({"low": r["low"], "high": r["high"], "form": r["form"]})
    return out


def _not_computed(exc):
    return {"estimate": None, "low": None, "high": None, "form": None, "not_computed": str(exc)}


def _outward(x, below):
    """One float beyond x, away from the interval (x is a nearest-rounded end), kept in [0, 1]."""
    if below:
        return 0.0 if x <= 0.0 else math.nextafter(x, -math.inf)
    return 1.0 if x >= 1.0 else math.nextafter(x, math.inf)


def producer_box(c, sizes, M_hc, h_s, m_s, r_s, alpha=0.05):
    """Producer's accuracy of class c, R = theta_y / (theta_y + theta_w): theta_y the share of the population the map
    calls c and the reference calls c, (M_c / N) times c's user's accuracy, and theta_w the share the map calls
    something else and the reference calls c, over the windows of each stratum the map does not call c. Each part
    gets its interval at alpha / 2, and R lies in [L_y / (L_y + U_w), U_y / (U_y + L_w)] when both hold (the box of
    agency.stratified, a union bound). The ratio is formed in exact fractions from the parts' ends moved one float
    outward, and rounded outward."""
    N = int(np.sum(sizes))
    M_c = int(M_hc[:, c].sum())
    in_c = m_s == c
    is_ref = (r_s == c).astype(float)
    y = linear_valid(M_hc[:, c], h_s[in_c], is_ref[in_c], alpha=alpha / 2)
    sy = Fraction(M_c, N)
    Ly, Uy = sy * Fraction(_outward(y["low"], True)), sy * Fraction(_outward(y["high"], False))
    py = (float(sy) * y["estimate_range"][0], float(sy) * y["estimate_range"][1])
    rest = np.asarray(sizes, int) - M_hc[:, c]
    if int(rest.sum()) == 0:                                 # the map calls every window c: nothing is missed
        Lw = Uw = Fraction(0)
        pw = (0.0, 0.0)
    else:
        w = linear_valid(rest, h_s[~in_c], is_ref[~in_c], alpha=alpha / 2)
        sw = Fraction(int(rest.sum()), N)
        Lw, Uw = sw * Fraction(_outward(w["low"], True)), sw * Fraction(_outward(w["high"], False))
        pw = (float(sw) * w["estimate_range"][0], float(sw) * w["estimate_range"][1])
    low = _float_below(Ly / (Ly + Uw)) if Ly + Uw > 0 else 0.0
    high = _float_above(Uy / (Uy + Lw)) if Uy + Lw > 0 else 1.0
    e_lo = py[0] / (py[0] + pw[1]) if py[0] + pw[1] > 0 else 0.0
    e_hi = py[1] / (py[1] + pw[0]) if py[1] + pw[0] > 0 else 1.0
    e_lo, e_hi = min(e_lo, e_hi), max(e_lo, e_hi)
    return {"estimate": e_lo if e_lo == e_hi else None, "estimate_range": [e_lo, e_hi],
            "low": max(0.0, min(low, e_lo)), "high": min(1.0, max(high, e_hi)), "form": FORM_BOX}


def class_quantities(c, sizes, M_hc, h_s, m_s, r_s, alpha=0.05):
    """Class c's valid intervals: {user_accuracy, producer_accuracy, reference_share}, each a quantity, None when there
    is nothing to bound (no labelled window the map calls c; no labelled window of reference c), or uncomputed.

    sizes: windows per stratum; M_hc: windows per stratum (rows) and map class (columns), from the map; h_s, m_s, r_s:
    each labelled window's stratum, map class and reference class."""
    sizes = np.asarray(sizes, int)
    N = int(sizes.sum())
    M_c = int(M_hc[:, c].sum())
    in_c = m_s == c
    is_ref = (r_s == c).astype(float)
    out = {}
    try:                                                     # over the windows each stratum's map calls c
        out["user_accuracy"] = (_quantity(linear_valid(M_hc[:, c], h_s[in_c], is_ref[in_c], alpha=alpha))
                                if in_c.any() else None)
    except TooLarge as exc:
        out["user_accuracy"] = _not_computed(exc)
    if not is_ref.any():
        out["producer_accuracy"] = None                      # no labelled window of the class: as without --valid
    elif M_c == 0:                                           # the map never draws it: exactly 0, as without --valid
        out["producer_accuracy"] = {"estimate": 0.0, "low": 0.0, "high": 0.0, "form": FORM_EXACT}
    else:
        try:
            out["producer_accuracy"] = _quantity(producer_box(c, sizes, M_hc, h_s, m_s, r_s, alpha))
        except TooLarge as exc:
            out["producer_accuracy"] = _not_computed(exc)
    try:                                                     # over the design's strata
        share = _quantity(linear_valid(sizes, h_s, is_ref, alpha=alpha))
        drawn = Fraction(M_c, N)
        share["drawn"] = ("too little" if drawn < Fraction(share["low"]) else
                          "too much" if drawn > Fraction(share["high"]) else "within")
    except TooLarge as exc:
        share = _not_computed(exc)
    out["reference_share"] = share
    return out


def per_class_valid(sample, idx, ref, mc, pop, C, srs, alpha=0.05):
    """The valid per-class table for estimate_per_class(valid=True), after its checks: (overall accuracy, {class:
    class_quantities}). `srs`: the sample is one stratum (random, sequential read as random, or one condition)."""
    N = int(pop.size)
    pos = {int(v): i for i, v in enumerate(pop)}
    local = np.array([pos[int(v)] for v in np.asarray(idx, int)], dtype=int)
    if srs:
        strata_pop, S = np.zeros(N, dtype=int), 1
    else:
        strata_pop, S = np.asarray(sample["strata"], int), len(sample["sizes"])
        if strata_pop.size != N:
            raise ValueError("the sample's strata do not describe its population")
    sizes = np.bincount(strata_pop, minlength=S).astype(int)
    if not srs and sizes.tolist() != [int(s) for s in sample["sizes"]]:
        raise ValueError(f"the strata sizes {list(sample['sizes'])} are not the counts of the sample's strata")
    M_hc = np.zeros((S, C), dtype=int)                       # windows per stratum and map class, over the population
    np.add.at(M_hc, (strata_pop, np.asarray(mc, int)[pop]), 1)
    h_s = strata_pop[local]
    m_s = np.asarray(mc, int)[np.asarray(idx, int)]
    r_s = np.asarray(ref, int)
    try:
        overall = _quantity(linear_valid(sizes, h_s, (m_s == r_s).astype(float), alpha=alpha))
    except TooLarge as exc:
        overall = _not_computed(exc)
    return overall, {int(c): class_quantities(c, sizes, M_hc, h_s, m_s, r_s, alpha) for c in range(C)}

"""Which of two maps is more accurate, from labels on the windows where they differ: the labelled which-map test."""
import numpy as np

from oe_inferencex.common.hypergeom import _level, hypergeom_interval


# ----------------------------------------------------------------------------- which of two maps is more accurate
# Where two maps of the same windows give the same class, both are right or both are wrong; their accuracies differ
# only through the windows where they differ. So the difference acc_a - acc_b = (K_a - K_b) / N, with K_a and K_b the
# differing windows where a, and where b, is right, and a random sample of the D differing windows estimates it with
# about N / D times fewer labels than a sample of the whole map (exp90: a median of 4.8 times on the record's maps).
WHICH_MAP_NOTE = ("The difference is over the windows compared: both maps predict them and neither splits evenly between "
                  "two classes. This says which map is more accurate and by how much, not either map's accuracy: where the maps agree "
                  "they are both right or both wrong, and how often needs a sample of all windows (sample --design random "
                  "on each map). Labels are assumed right; a window marked ? is counted for each map both ways.")
SIGN_TEST_SHARE = 5                  # the sign test decides while the labels are at most 1/5 of the differing windows
SIGN_TEST_NOTE = ("The verdict is the exact sign test on the labelled windows where one map is right (two-sided): it "
                  "says which map is more accurate and names the less accurate one at most half its level of the time "
                  "(2.5% at 95%), drawn without replacement. The interval gives the size of the difference; it is a "
                  "union bound, wider than the test, so it can include 0 when the test names a map. The test decides "
                  "while the labels are at most a fifth of the differing windows, where it names the better map more "
                  "often than the interval (exp107); past that, and in a census, the interval decides.")
_SIGN_CRIT = {}


def sign_critical(m, alpha):
    """The largest k with 2 P(Bin(m, 1/2) <= k) < alpha, or -1: the two-sided exact sign test on (w, m - w) rejects at
    alpha exactly when min(w, m - w) <= k (stats.sign_test's p is min(1, 2 P(X <= min))). Integer arithmetic, alpha
    as a fraction, so no float decides a tail and nothing overflows (exp107's critical values at m >= 1024)."""
    alpha, m = _level(alpha), int(m)
    key = (m, alpha)
    if key not in _SIGN_CRIT:
        if len(_SIGN_CRIT) > 100_000:
            _SIGN_CRIT.clear()
        total, cum, k, c = 1 << m, 0, -1, 1
        for i in range(m // 2 + 1):                        # 2 P(X <= m/2) >= 1 > alpha, so the loop stops by then
            cum += c
            if 2 * cum * alpha.denominator < alpha.numerator * total:
                k = i
            else:
                break
            c = c * (m - i) // (i + 1)
        _SIGN_CRIT[key] = k
    return _SIGN_CRIT[key]


def which_map_rule(n, D, rule="auto"):
    """The rule that names a map from n labels among D differing windows: "sign" (the exact sign test) while
    SIGN_TEST_SHARE * n <= D, else "interval" (the union-bound interval, which in a census is the census answer). The
    switch depends on n and D only, fixed before any label, so whichever rule decides keeps its own wrong-map rate.
    "sign" or "interval" forces one."""
    if rule not in ("auto", "sign", "interval"):
        raise ValueError(f"rule must be auto, sign or interval, got {rule!r}")
    if rule != "auto":
        return rule
    return "sign" if SIGN_TEST_SHARE * int(n) <= int(D) else "interval"


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


def compare_from_disagreement(sample, class_a, class_b, reference, unjudged=None, conf=0.95, rule="auto"):
    """Which of two maps is more accurate, from a labelled `sample_disagreement`.

    class_a, class_b: each sampled window's class in the two maps, in the order of sample["indices"]; reference: the
    class the reviewer saw there (ignored where `unjudged`). Among the n labelled windows of the D differing ones, a are
    right in map a and b in map b (a window can be right in at most one). Each count gets its exact hypergeometric
    interval over the D windows at 1 - (1 - conf) / 2, so the two hold together with probability at least conf (union
    bound), and the accuracy difference over the N windows both maps predict, (K_a - K_b) / N, lies in
    [(lo_a - hi_b) D / N, (hi_a - lo_b) D / N]. A window marked ? counts as wrong for a map at that map's lower end and
    as right at its upper end; the intervals move one way with the count, so the bound holds whatever made the windows
    hard to judge.

    The verdict (`which_map_rule`): while the labels are at most a fifth of the differing windows, the exact sign test
    on a and b at alpha = 1 - conf, naming a only if the test on (a, b + u) names it and b only if the test on
    (a + u, b) does, u the windows marked ?; the test on the true counts then names the same map, so the bound holds
    whatever made them hard to judge. Past a fifth, and in a census, the interval names a map when it excludes 0.
    exp107: at n = 100 the test named the better map 0.095 to 0.154 more often than the interval on 5 of 9 product
    pairs, and less often past a quarter to two fifths of the differing windows (it ignores the finite population)."""
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
    interval_verdict = "a" if low > 0 else "b" if high < 0 else None
    used = which_map_rule(n, D, rule)
    from oe_inferencex.common import stats
    alpha = 1 - _level(conf)
    k = sign_critical(a + b + u, alpha)
    sign_verdict = "a" if a > b + u and b + u <= k else "b" if b > a + u and a + u <= k else None
    p = stats.sign_test(a, b + u) if a > b + u else stats.sign_test(a + u, b) if b > a + u else 1.0
    verdict = sign_verdict if used == "sign" else interval_verdict
    return {"design": "disagreement", "n_population": N, "n_disagree": D, "disagree_share": D / N, "n_labelled": n,
            "n_unjudged": u, "n_a_right": a, "n_b_right": b, "n_neither": c,
            "share_a_right": {"low": lo_a, "high": hi_a}, "share_b_right": {"low": lo_b, "high": hi_b},
            "difference": {"estimate": est_lo if est_lo == est_hi else None, "estimate_range": [est_lo, est_hi],
                           "low": low, "high": high},
            "verdict": verdict, "verdict_rule": used, "interval_verdict": interval_verdict,
            "sign_test": {"one_right": a + b, "unjudged": u, "p_value": float(p), "alpha": float(alpha),
                          "verdict": sign_verdict},
            "conf": conf,
            "method": (f"labels on a simple random sample of the {D} windows where the maps differ; each map's share of "
                       f"them that is right gets its exact interval at {side:.3g}, and the difference of the two, times "
                       f"{D}/{N}, covers the whole map's accuracy difference at least {conf:.0%} of the time (union "
                       "bound); "
                       + ("the verdict is the exact sign test on the windows where one map is right"
                          + (", since the labels are at most a fifth of the differing windows" if rule == "auto" else "")
                          if used == "sign" else "the verdict is the interval's")),
            "note": WHICH_MAP_NOTE, "verdict_note": SIGN_TEST_NOTE}

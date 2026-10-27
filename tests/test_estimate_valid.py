"""estimate --valid (1.20.0): an interval valid at every sample size for estimate's designs, opt-in.

The claim (estimate/valid.py): with --valid, the whole map's error rate and each class's user's accuracy, producer's
accuracy and share get an interval that covers at least 1 - alpha for every map and every number of labels under the
confidence, proportional, condition and class designs, a `?` window bounded both ways; and nothing else estimate writes
or prints changes, with the option or without it. Checked here without simulation:
- coverage over every population of small stratified maps, by the counts with exact hypergeometric probabilities: the
  whole map at alpha 0.05, 0.1 and 0.3 (two and three strata, unequal shares, an empty stratum, a stratum labelled in
  full), and through estimate_error_rate on samples sample_for_estimation draws under the condition and class designs;
- `?` against any choice of the windows marked `?`, a stratum labelled in full holding some;
- a stratum labelled in full enters exactly, and a census is the exact rate;
- with one stratum labelled in part the interval is 1.19.0's exact one to the last bit: a random sample's, `?`
  included, and the class design's user's accuracy;
- per class, post-stratified by the map's class within each stratum, over every population of windows of four kinds
  (the map's class and the reference's, each the class or not), including draws that leave a part of a stratum with no
  label; and the class design's per-quantity intervals on the fixed maps of tests/test_class_design.py;
- every interval holds its own point, and negative controls show the checks can fail;
- a stratum too large for the betting interval is decided from its label count alone and takes the union bound;
- the default output: estimate on the labelled golden samples of tests/golden/condition_1_3_1, read as they are (not
  regenerated), writes and prints what those files hold; with --valid every other key and line is the same, and decide
  answers the same."""
import contextlib
import functools
import importlib.util
import io
import itertools
import json
import math
import os
import shutil
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex.estimate import valid as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden", "condition_1_3_1")


def log_choose(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compositions(total, parts):
    if parts == 1:
        return [(total,)]
    return [(i,) + rest for i in range(total + 1) for rest in compositions(total - i, parts - 1)]


def _table(N, n, parts):
    """One stratum of N windows of `parts` kinds, n of them drawn without replacement: {population: {draw: log
    probability}}, every population and every draw it allows."""
    out = {}
    for p in compositions(N, parts):
        row = {}
        for s in compositions(n, parts):
            lp = sum(log_choose(p[j], s[j]) for j in range(parts)) - log_choose(N, n)
            if lp > -math.inf:
                row[s] = lp
        out[p] = row
    return out


def _worst(Ns, ns, parts, interval, truth):
    """The least probability, over every population of strata of Ns windows of `parts` kinds, that interval(population,
    draw) holds truth(population): ns windows drawn without replacement in each stratum, independent across strata. A
    truth of None (nothing to cover) is skipped; an interval of None (no statement) is counted as holding, and the
    callers that pass one check it arises only where the table without --valid makes no statement either. The ends are
    compared with the truth exactly, with no tolerance: the truths are correctly rounded quotients of whole numbers,
    exact and census ends the nearest float to their fraction, betting and box ends rounded outward (valid.py), so an
    end that held its fraction holds the rounded truth. The 1e-12 the callers allow is on the probability, a sum of
    exponentials of log-gamma differences."""
    tables = [_table(N, n, parts) for N, n in zip(Ns, ns)]
    worst = 1.0
    for pops in itertools.product(*[list(t) for t in tables]):
        theta = truth(pops)
        if theta is None:
            continue
        cov = 0.0
        for combo in itertools.product(*[list(tables[h][p].items()) for h, p in enumerate(pops)]):
            iv = interval(pops, tuple(s for s, _ in combo))
            if iv is None or iv[0] <= theta <= iv[1]:
                cov += math.exp(sum(lp for _, lp in combo))
        worst = min(worst, cov)
    return worst


def _stratified(Ns, ns, design="confidence"):
    """A stratified sample as sample_for_estimation writes one: strata of Ns windows, the first ns of each labelled."""
    N = int(sum(Ns))
    start = np.concatenate([[0], np.cumsum(Ns)[:-1]]).astype(int)
    idx = np.concatenate([start[h] + np.arange(n) for h, n in enumerate(ns)]).astype(int)
    return {"design": design, "indices": idx, "n_population": N, "strata_of_population": np.arange(N),
            "strata": np.repeat(np.arange(len(Ns)), Ns).astype(int), "sizes": [int(x) for x in Ns],
            "allocation": [int(x) for x in ns], "n_strata": len(Ns)}


def _labels(ns, ks):
    """`wrong` per labelled window, stratum by stratum: the first k_h of stratum h's n_h labels wrong."""
    return np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for n, k in zip(ns, ks)])


def _points(q):
    return q["estimate_range"] if q["estimate"] is None else [q["estimate"]] * 2


def _holds_its_point(q):
    lo, hi = _points(q)
    return q["low"] <= lo <= hi <= q["high"]


def _memo_linear(monkeypatch):
    """linear_valid reads its groups only through their sizes and counts (the betting interval through each stratum's
    counts of each value): cached on them, so an enumeration runs each distinct interval once."""
    orig, cache = V.linear_valid, {}

    def memo(M, g, x_lo, x_hi=None, alpha=0.05):
        g, x_lo = np.asarray(g, int), np.asarray(x_lo, float)
        x_hi = x_lo if x_hi is None else np.asarray(x_hi, float)
        J = len(M)
        key = (tuple(int(m) for m in M), alpha, tuple(np.bincount(g, minlength=J).tolist()),
               tuple(np.bincount(g, weights=x_lo, minlength=J).tolist()),
               tuple(np.bincount(g, weights=x_hi, minlength=J).tolist()))
        if key not in cache:
            cache[key] = orig(M, g, x_lo, x_hi, alpha)
        return cache[key]
    monkeypatch.setattr(V, "linear_valid", memo)


# ----------------------------------------------------------------------------------------------- the whole map
GRIDS = [((6, 4), (3, 2)), ((8, 3), (2, 2)), ((9, 3), (2, 3)), ((5, 4, 3), (2, 2, 1)), ((5, 4, 3), (3, 4, 1)),
         ((6, 0, 4), (3, 0, 2))]
FORMS = {((6, 4), (3, 2)): "betting", ((8, 3), (2, 2)): "betting", ((9, 3), (2, 3)): "exact",
         ((5, 4, 3), (2, 2, 1)): "betting", ((5, 4, 3), (3, 4, 1)): "betting", ((6, 0, 4), (3, 0, 2)): "betting"}


def _whole_map_intervals(Ns, ns, alpha):
    sample = _stratified(Ns, ns)
    ivs = {}
    for ks in itertools.product(*[range(n + 1) for n in ns]):
        r = V.whole_map_valid(sample, _labels(ns, ks), alpha=alpha)
        assert r["guarantee"] == "valid" and r["form"] == FORMS.get((Ns, ns), r["form"]) and _holds_its_point(r)
        ivs[ks] = (r["low"], r["high"])
    return ivs


@pytest.mark.parametrize("Ns, ns", GRIDS)
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_the_whole_map_interval_covers_every_population(Ns, ns, alpha):
    """Two and three strata (the confidence design's quantiles, Neyman-like shares), a stratum labelled in full beside
    one or two labelled in part, and an empty stratum, as a confidence design's empty quantile is: for every count of
    wrong windows in every stratum, the interval holds the whole-map rate with probability at least 1 - alpha."""
    ivs = _whole_map_intervals(Ns, ns, alpha)
    worst = _worst(Ns, ns, 2, lambda pops, ss: ivs[tuple(s[1] for s in ss)],
                   lambda pops: sum(p[1] for p in pops) / sum(Ns))
    assert worst >= 1 - alpha - 1e-12, (Ns, ns, alpha, worst)


def test_the_whole_map_check_can_fail():
    """The negative control. At alpha 0.6, a stratum of 6 windows with 3 labelled (the other labelled in full) takes its
    exact interval at 40%, which holds a rate of 2 in 6 on 0.6 of draws, below the 0.9 the check above asks at alpha
    0.1; the point alone holds far less often. So the check sees an interval that is too narrow."""
    Ns, ns = (6, 4), (3, 4)
    sample = _stratified(Ns, ns)
    loose, point = {}, {}
    for ks in itertools.product(*[range(n + 1) for n in ns]):
        r = V.whole_map_valid(sample, _labels(ns, ks), alpha=0.6)
        loose[ks], point[ks] = (r["low"], r["high"]), (r["estimate"], r["estimate"])
    truth = lambda pops: sum(p[1] for p in pops) / sum(Ns)
    assert _worst(Ns, ns, 2, lambda pops, ss: loose[tuple(s[1] for s in ss)], truth) < 0.9
    assert _worst(Ns, ns, 2, lambda pops, ss: point[tuple(s[1] for s in ss)], truth) < 0.9


@pytest.mark.parametrize("Ns, ns", [((4, 3), (2, 2)), ((4, 2), (2, 2)), ((5, 3), (3, 1))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_windows_marked_unjudged_cover_whichever_they_are(Ns, ns, alpha):
    """Windows of four kinds in each stratum: right and judged, right and `?`, wrong and judged, wrong and `?`. Which
    windows are `?` is a property of the window, so every population of the four kinds is every adversary that marks
    windows `?` knowing their truth. The interval counts each `?` right at its lower end and wrong at its upper and
    must hold the true rate of all the windows; ((4, 2), (2, 2)) labels its second stratum in full, `?` included."""
    sample = _stratified(Ns, ns)
    ivs = {}
    for ss in itertools.product(*[compositions(n, 4) for n in ns]):
        kinds = np.concatenate([np.repeat(np.arange(4), s) for s in ss])
        unj = kinds % 2 == 1
        r = V.whole_map_valid(sample, np.where(unj, 0.0, (kinds >= 2).astype(float)), unj, alpha=alpha)
        assert _holds_its_point(r) and (r["estimate"] is None) == bool(unj.any())
        assert ("`?`" in r["method"]) == bool(unj.any())
        ivs[ss] = (r["low"], r["high"])
    worst = _worst(Ns, ns, 4, lambda pops, ss: ivs[ss], lambda pops: sum(p[2] + p[3] for p in pops) / sum(Ns))
    assert worst >= 1 - alpha - 1e-12, (Ns, ns, alpha, worst)


def test_a_stratum_labelled_in_full_enters_exactly_and_a_census_is_the_rate():
    r = V.whole_map_valid(_stratified((3, 2), (3, 2)), _labels((3, 2), (1, 2)))
    assert r["form"] == "census" and r["low"] == r["high"] == r["estimate"] == 3 / 5
    r = V.whole_map_valid(_stratified((3, 2), (3, 2)), np.array([1, 0, 0, 0, 1.0]), np.array([0, 1, 0, 0, 0], bool))
    assert r["estimate"] is None and r["estimate_range"] == [2 / 5, 3 / 5] and (r["low"], r["high"]) == (2 / 5, 3 / 5)
    # one stratum labelled in part: its exact interval, widened to hold its share as hypergeom_interval widens it, and
    # the stratum labelled in full added exactly
    Ns, ns = (6, 4), (3, 4)
    for k0, k1 in itertools.product(range(4), range(5)):
        r = V.whole_map_valid(_stratified(Ns, ns), _labels(ns, (k0, k1)))
        lo, hi = est.hypergeom_interval(k0, 3, 6)
        a = min(Fraction(round(lo * 6), 6), Fraction(k0, 3))
        b = max(Fraction(round(hi * 6), 6), Fraction(k0, 3))
        assert r["form"] == "exact" and r["low"] == float((k1 + 6 * a) / 10) and r["high"] == float((k1 + 6 * b) / 10)


@pytest.mark.parametrize("N, n", [(7, 3), (12, 5), (20, 19), (30, 30), (50, 10), (101, 37)])
def test_a_random_sample_gets_its_exact_interval_to_the_last_bit(N, n):
    """A simple random sample is one stratum, so the valid interval is the exact one estimate gives without --valid,
    bit for bit, with `?` windows too; and every other key of the result is the result without --valid."""
    sample = {"design": "random", "indices": np.arange(n), "n_population": N}
    unj = np.zeros(n, bool)
    unj[:max(1, n // 3)] = True
    for k in range(n + 1):
        wrong = np.r_[np.ones(k), np.zeros(n - k)]
        res = est.estimate_error_rate(sample, wrong, valid=True)
        v = res.pop("valid_interval")
        assert res == est.estimate_error_rate(sample, wrong)
        assert (v["low"], v["high"]) == (res["low"], res["high"]) == est.hypergeom_interval(k, n, N)
        assert v["estimate"] == res["estimate"] and v["form"] == ("census" if n == N else "exact")
        assert v["guarantee"] == "valid" and v["method"].startswith("simple random sample; valid at every sample size")
        res = est.estimate_error_rate(sample, wrong, unjudged=unj, valid=True)
        v = res.pop("valid_interval")
        assert res == est.estimate_error_rate(sample, wrong, unjudged=unj)
        assert (v["low"], v["high"]) == (res["low"], res["high"]) and _points(v) == _points(res)
    # a sequential sample is read as a random one, as without --valid
    seq = est.estimate_error_rate(dict(sample, design="sequential"), np.r_[np.ones(2), np.zeros(n - 2)], valid=True)
    assert (seq["valid_interval"]["low"], seq["valid_interval"]["high"]) == (seq["low"], seq["high"])


def test_reviewer_bounds_map_the_valid_ends_as_they_map_the_others():
    """With a reviewer's stated error bounds the valid ends go through the same maps as estimate's own: under a random
    sample, where the two intervals are one, the results agree to the last bit."""
    sample = {"design": "random", "indices": np.arange(40), "n_population": 300}
    for k in (0, 3, 17, 40):
        wrong = np.r_[np.ones(k), np.zeros(40 - k)]
        res = est.estimate_error_rate(sample, wrong, reviewer_false_alarm=0.05, reviewer_miss=0.1, valid=True)
        v = res["valid_interval"]
        assert (v["low"], v["high"]) == (res["low"], res["high"]) and _points(v) == _points(res)


@pytest.mark.parametrize("sizes, budget, floor", [((12, 8), 6, None), ((8, 6, 4), 6, None), ((30, 6), 10, None),
                                                  ((10, 5, 3), 9, None), ((14, 6), 8, 3)])
def test_the_condition_and_class_designs_whole_map_valid_interval_covers(sizes, budget, floor):
    """On samples sample_for_estimation draws under the condition design (and, with a floor, the class design's
    allocation), through estimate_error_rate(valid=True): every other key is the result without it, and over every
    count of wrong windows in every stratum the valid interval holds the whole-map rate at least 95% of the time."""
    cond = np.repeat(np.arange(len(sizes)), sizes)
    base = est.sample_for_estimation(np.linspace(1.0, 0.0, cond.size), budget, design="condition", condition=cond,
                                     seed=0, class_floor=floor)
    n, idx = base["allocation"], base["indices"]
    at = [np.flatnonzero(cond[idx] == c) for c in range(len(sizes))]
    ivs = {}
    for ks in itertools.product(*[range(nc + 1) for nc in n]):
        wrong = np.zeros(idx.size)
        for pos, kc in zip(at, ks):
            wrong[pos[:kc]] = 1
        res = est.estimate_error_rate(base, wrong, valid=True)
        v = res.pop("valid_interval")
        assert res == est.estimate_error_rate(base, wrong) and _holds_its_point(v)
        assert v["method"].startswith("stratified by input condition; valid at every sample size")
        ivs[ks] = (v["low"], v["high"])
    worst = _worst(sizes, n, 2, lambda pops, ss: ivs[tuple(s[1] for s in ss)],
                   lambda pops: sum(p[1] for p in pops) / sum(sizes))
    assert worst >= 0.95 - 1e-12, (sizes, budget, worst)


def test_a_stratum_too_large_falls_back_by_its_label_count_alone(monkeypatch):
    """The betting interval's exact average over orders has a worst case set by a stratum's label count; above the
    limit the whole map takes the union bound over the strata, for every label vector alike, and a per-class quantity
    is left uncomputed with the reason. A single stratum labelled in part needs no average and is unaffected."""
    Ns, ns = (6, 4), (3, 2)
    assert V.worst_work(3) == V.K_MIX * 3 * 2 and V.worst_work(2) == V.K_MIX * 2 * 2
    monkeypatch.setattr(V, "MAX_WORK", V.worst_work(3) - 1)          # 3 labels too many, 2 not
    for ks in itertools.product(range(4), range(3)):
        r = V.whole_map_valid(_stratified(Ns, ns), _labels(ns, ks))
        _, lo, hi, L = est._union_interval(list(ks), list(ns), list(Ns))
        assert r["form"] == "union bound" and "union bound" in r["method"] and L == 2
        assert (r["low"], r["high"]) == (min(lo, r["estimate"]), max(hi, r["estimate"]))
    assert V.whole_map_valid(_stratified(Ns, (3, 4)), _labels((3, 4), (1, 2)))["form"] == "exact"
    q = V.class_quantities(0, np.array(Ns), np.array([[3, 3], [2, 2]]), np.array([0, 0, 0, 1, 1]),
                           np.array([0, 0, 1, 0, 1]), np.array([0, 1, 0, 0, 1]))
    assert q["reference_share"]["low"] is None and "too large" in q["reference_share"]["not_computed"]


def test_tiles_are_refused_and_bad_inputs_named():
    with pytest.raises(ValueError, match="no interval for the tiles design"):
        V.whole_map_valid({"design": "tiles", "indices": np.arange(3), "n_population": 9}, np.zeros(3))
    with pytest.raises(ValueError, match="holds 3 labels but only 2 windows"):
        V.linear_valid([2, 5], [0, 0, 0], [0, 1, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        V.linear_valid([4], [0, 0], [0, 0.5])
    with pytest.raises(ValueError, match="lower reading at most the upper"):
        V.linear_valid([4], [0, 0], [1, 0], [0, 0])


# ----------------------------------------------------------------------------------------------- per class
def _per_class_draw(Ns, pops, ss):
    """A draw of windows of four kinds per stratum, as class_quantities reads class 0 of two: kind 0 mapped as the
    class and of it, 1 mapped as it and of another, 2 mapped as another and of it, 3 neither. Returns (stratum sizes,
    windows per stratum and map class, and each labelled window's stratum, map class and reference class)."""
    M_hc = np.array([[p[0] + p[1], p[2] + p[3]] for p in pops], dtype=int)
    kinds = [np.repeat(np.arange(4), s) for s in ss]
    h_s = np.concatenate([np.full(k.size, h, dtype=int) for h, k in enumerate(kinds)])
    k = np.concatenate(kinds)
    return np.asarray(Ns, int), M_hc, h_s, (k >= 2).astype(int), (k % 2).astype(int)


def _per_class_truths(Ns):
    N = sum(Ns)
    A = lambda pops: sum(p[0] for p in pops)
    mapped = lambda pops: sum(p[0] + p[1] for p in pops)
    of = lambda pops: sum(p[0] + p[2] for p in pops)
    return {"user_accuracy": lambda pops: A(pops) / mapped(pops) if mapped(pops) else None,
            "producer_accuracy": lambda pops: A(pops) / of(pops) if of(pops) else None,
            "reference_share": lambda pops: of(pops) / N}


@pytest.mark.parametrize("Ns, ns", [((4, 3), (2, 2)), ((4, 2), (2, 2)), ((5, 3), (2, 1))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_per_class_intervals_cover_every_population(Ns, ns, alpha, monkeypatch):
    """Two strata whose windows are of the four kinds, the map's class spread across them as under the confidence
    design: over every population and every draw, each class's user's accuracy (post-stratified by the map's class in
    each stratum), producer's accuracy (the box) and share hold at least 1 - alpha, each on its own. The draws include
    parts of a stratum the map calls the class that hold no label, which bound the user's accuracy; ((4, 2), (2, 2))
    labels its second stratum in full. Every interval holds its own point."""
    _memo_linear(monkeypatch)
    seen = set()
    table = _class_table(Ns, alpha, seen)

    def interval(name):
        def f(pops, ss):
            v = table(pops, ss)[name]
            return None if v is None else (v["low"], v["high"])
        return f
    for name, truth in _per_class_truths(Ns).items():
        worst = _worst(Ns, ns, 4, interval(name), truth)
        assert worst >= 1 - alpha - 1e-12, (name, Ns, ns, alpha, worst)
    assert any(name == "user_accuracy" and empty for name, _, empty in seen)      # a part of a stratum with no label
    assert any(name == "producer_accuracy" and form == "box" for name, form, _ in seen)


def _class_table(Ns, alpha, seen=None):
    """class_quantities for class 0 of a draw, cached on what it reads (the map's counts per stratum and the draw);
    each result is checked to hold its own points, and its forms are noted in `seen`."""
    cache = {}

    def table(pops, ss):
        key = (tuple((p[0] + p[1], p[2] + p[3]) for p in pops), ss)
        if key not in cache:
            q = V.class_quantities(0, *_per_class_draw(Ns, pops, ss), alpha)
            # no statement only where the table without --valid makes none: the user's accuracy with no labelled
            # window mapped as the class, the producer's with none of it; never the share
            assert (q["user_accuracy"] is None) == (sum(s[0] + s[1] for s in ss) == 0), (ss, q["user_accuracy"])
            assert (q["producer_accuracy"] is None) == (sum(s[0] + s[2] for s in ss) == 0), (ss, q["producer_accuracy"])
            assert q["reference_share"] is not None
            for name, v in q.items():
                if v is not None:
                    assert _holds_its_point(v), (name, v)
                    if seen is not None:
                        seen.add((name, v["form"], v["estimate"] is None))
            cache[key] = q
        return cache[key]
    return table


def test_the_per_class_check_can_fail(monkeypatch):
    """The negative control: each quantity's point alone (its range where a part of a stratum holds no label), in
    place of its interval, falls short of 0.9 on some population, so the check above sees an interval that is too
    narrow."""
    _memo_linear(monkeypatch)
    Ns, ns = (4, 3), (2, 2)
    table = _class_table(Ns, 0.1)

    def point(name):
        def f(pops, ss):
            v = table(pops, ss)[name]
            return None if v is None else tuple(_points(v))
        return f
    for name, truth in _per_class_truths(Ns).items():
        assert _worst(Ns, ns, 4, point(name), truth) < 0.9, name


# the fixed maps of tests/test_class_design.py (_maps()[:4]): stratum h is map class h, K[h][j] its windows of
# reference class j, n_h its labels
CLASS_MAPS = [([[36, 4], [5, 20]], (12, 10)), ([[38, 2], [12, 13]], (12, 10)), ([[30, 10], [1, 24]], (15, 12)),
              ([[60, 3], [9, 18]], (20, 15))]


@pytest.mark.parametrize("K, n_h", CLASS_MAPS)
def test_the_class_designs_per_quantity_intervals_cover(K, n_h):
    """Over the class design's exact draw (a multivariate hypergeometric in each stratum), each class's user's
    accuracy, producer's accuracy and share, each on its own, covers at least 95%; the user's accuracy is 1.19.0's
    exact interval of its stratum to the last bit."""
    K = np.asarray(K)
    S, C = K.shape
    sizes = K.sum(1)
    N = int(sizes.sum())
    M_hc = np.diag(sizes)                                     # every window of stratum h is mapped as class h
    outcomes = []
    for h in range(S):
        total = math.comb(int(sizes[h]), n_h[h])
        outcomes.append([(x, math.prod(math.comb(int(K[h, j]), x[j]) for j in range(C)) / total)
                         for x in compositions(n_h[h], C) if all(x[j] <= K[h, j] for j in range(C))])
    truth = {"user_accuracy": [K[c, c] / sizes[c] for c in range(C)],
             "producer_accuracy": [K[c, c] / K[:, c].sum() for c in range(C)],
             "reference_share": [K[:, c].sum() / N for c in range(C)]}
    cov = {(q, c): 0.0 for q in truth for c in range(C)}
    for combo in itertools.product(*outcomes):
        p = math.prod(w for _, w in combo)
        h_s = np.array([h for h, (x, _) in enumerate(combo) for j in range(C) for _ in range(x[j])])
        r_s = np.array([j for h, (x, _) in enumerate(combo) for j in range(C) for _ in range(x[j])])
        for c in range(C):
            q = V.class_quantities(c, sizes, M_hc, h_s, h_s, r_s)
            ua = q["user_accuracy"]
            assert (ua["low"], ua["high"]) == est.hypergeom_interval(combo[c][0][c], n_h[c], int(sizes[c]))
            # no statement only for the producer's accuracy of a class no labelled window is of; never for the share
            assert q["reference_share"] is not None
            assert (q["producer_accuracy"] is None) == (sum(x[c] for x, _ in combo) == 0)
            for name in truth:
                v = q[name]                       # None: no labelled window of the class, no statement to miss
                assert v is None or _holds_its_point(v), (name, v)
                cov[(name, c)] += p * (v is None or v["low"] <= truth[name][c] <= v["high"])
    assert min(cov.values()) >= 0.95 - 1e-12, cov


def _small_map(seed=0, N=240, C=3):
    rng = np.random.default_rng(seed)
    mc = rng.integers(0, C, N)
    ref = np.where(rng.random(N) < 0.8, mc, rng.integers(0, C, N))
    margin = rng.random(N)
    return mc, ref, margin, 0.5 + 0.5 * margin


@pytest.mark.parametrize("design", ["confidence", "proportional", "random", "sequential", "condition", "class"])
def test_per_class_with_valid_adds_only_its_keys(design):
    """estimate_per_class(valid=True) is the table without it plus `valid` per row, `overall_accuracy_valid` and
    `valid_note`; each valid interval holds its point; under a random sample the user's accuracy and the overall
    accuracy are the exact ones the table already gives, and under the class design the user's accuracy is too."""
    mc, ref, margin, p1 = _small_map()
    kw = {"confidence": {"p1": p1}, "condition": {"condition": np.arange(mc.size) % 3},
          "class": {"condition": mc, "class_floor": 10}}.get(design, {})
    s = est.sample_for_estimation(margin, 60, design="condition" if design == "class" else design, seed=1, **kw)
    if design == "class":
        s["condition"]["by"] = "map class"
    idx = s["indices"]
    plain = est.estimate_per_class(s, ref[idx], mc)
    got = est.estimate_per_class(s, ref[idx], mc, valid=True)
    assert got.pop("valid_note") == est.PER_CLASS_VALID_NOTE
    overall = got.pop("overall_accuracy_valid")
    rows = {c: row.pop("valid") for c, row in got["per_class"].items()}
    assert got == plain
    assert _holds_its_point(overall) and set(rows) == set(range(3))
    for c, q in rows.items():
        assert set(q) == {"user_accuracy", "producer_accuracy", "reference_share"}
        for v in q.values():
            assert v is None or _holds_its_point(v)
        assert q["reference_share"]["drawn"] in ("too little", "too much", "within")
        if design in ("random", "sequential", "class") and plain["per_class"][c]["user_accuracy"] is not None:
            ua, pua = q["user_accuracy"], plain["per_class"][c]["user_accuracy"]
            assert (ua["low"], ua["high"], ua["estimate"]) == (pua["low"], pua["high"], pua["estimate"])
    if design in ("random", "sequential"):
        assert (overall["low"], overall["high"]) == (plain["overall_accuracy"]["low"], plain["overall_accuracy"]["high"])


# ----------------------------------------------------------------------------------------------- the default output
GOLDEN_SAMPLES = ("sample_confidence", "sample_proportional", "sample_scene_confidence", "sample_random",
                  "sample_logits3_random", "sample_scene_random", "sample_tiles")


def _changes():
    spec = importlib.util.spec_from_file_location("golden_1_3_1_changes_valid", os.path.join(GOLDEN, "changes.py"))
    changes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(changes)
    return changes


def _run(argv):
    from oe_inferencex.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(argv) == 0
    return buf.getvalue()


@pytest.mark.parametrize("name", GOLDEN_SAMPLES)
def test_the_default_output_is_the_golden_one_and_valid_only_adds(tmp_path, name):
    """estimate on a golden labelled sample, read from tests/golden/condition_1_3_1 as it is: without --valid it prints
    and writes the golden bytes (with the changes the golden contract lists, changes.py); with --valid the JSON is that
    result plus `valid_interval`, the first printed line is the valid one, the second the line printed without it,
    then every other line as before and the note; decide answers alike from both. The tiles design is refused."""
    from oe_inferencex.cli import main
    from oe_inferencex.decide import decide
    changes = _changes()
    shutil.copy(os.path.join(GOLDEN, f"{name}.labelled.csv"), tmp_path / f"{name}.csv")
    shutil.copy(os.path.join(GOLDEN, f"{name}.json"), tmp_path / f"{name}.json")
    csv_path, out = str(tmp_path / f"{name}.csv"), str(tmp_path / f"{name}_estimate.json")
    roots = sorted({str(tmp_path), os.path.realpath(str(tmp_path))}, key=len, reverse=True)
    norm = lambda text: functools.reduce(lambda t, r: t.replace(r, "<TMP>"), roots, text)

    printed = norm(_run(["estimate", csv_path, "--out", out]))
    with open(os.path.join(GOLDEN, f"{name}.estimate.stdout.txt"), "rb") as f:
        assert printed.encode() == changes.expected(f"{name}.estimate.stdout.txt", f.read())
    with open(out) as f:
        plain = json.load(f)
    got = dict(plain)
    got.pop("written_by")
    got.pop("scope", None)
    got["sample"] = norm(got["sample"])
    with open(os.path.join(GOLDEN, f"{name}.estimate.json"), "rb") as f:
        want = json.loads(changes.expected(f"{name}.estimate.json", f.read()))
    assert got == want or changes.same_but_last_digits(got, want), name

    v_out = str(tmp_path / "valid.json")
    if name == "sample_tiles":
        with pytest.raises(SystemExit, match="no interval for the tiles design"):
            main(["estimate", csv_path, "--valid", "--out", v_out])
        return
    printed_v = norm(_run(["estimate", csv_path, "--valid", "--out", v_out]))
    with open(v_out) as f:
        r = json.load(f)
    block = r.pop("valid_interval")
    assert r == plain
    assert block["guarantee"] == "valid" and _holds_its_point(block)
    if "random" in name:
        assert (block["low"], block["high"]) == (plain["low"], plain["high"])
    lines, plain_lines = printed_v.rstrip("\n").split("\n"), printed.rstrip("\n").split("\n")
    assert lines[0].startswith("error rate ") and ", valid 95% interval " in lines[0]
    assert lines[1] == "without --valid: " + plain_lines[0]
    note = f"note: {est.VALID_NOTE}"
    assert note in lines
    assert [x for x in lines[2:] if x != note and not x.startswith("wrote ")] == \
        [x for x in plain_lines[1:] if not x.startswith("wrote ")]
    asks = ["error_rate_below=0.1", "error_rate_below=0.3"]
    assert decide(v_out, asks) == decide(out, asks)

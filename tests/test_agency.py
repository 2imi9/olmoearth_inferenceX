"""grade and its intervals (agency.py, stratified.py): exact checks by enumeration, Olofsson et al.'s worked example,
and the command line. The enumerations are the judge's of 26 October 2026 (aicr job 1525456), at sizes a test can run:
every population and every sample of two or three small strata, coverage computed exactly."""
import csv
import itertools
import json
import math

import numpy as np
import pytest

from oe_inferencex import agency
from oe_inferencex.cli import main
from oe_inferencex.stratified import KAPPA, Design, _stratum_logavg, betting_interval, pure_stratum_ratio, ratio


def log_choose(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compositions(total, parts=3):
    if parts == 1:
        return [(total,)]
    return [(i,) + rest for i in range(total + 1) for rest in compositions(total - i, parts - 1)]


def design_of(ns, Ns, W):
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    return Design(hidx, W, [float(N) for N in Ns])


def values(design, per_stratum):
    """x per point from each stratum's list of values (the order within a stratum does not matter)."""
    x = np.zeros(design.hidx.size)
    for h, v in enumerate(per_stratum):
        x[design.masks[h]] = v
    return x


# ----------------------------------------------------------------------------------------------- the e-value
def brute_logavg(counts, vals, lam, N):
    """The averaged e-value of one stratum by listing every distinct order of its labels."""
    cats = [j for j, k in enumerate(counts) for _ in range(k)]
    finite = math.isfinite(N)
    tot = []
    for seq in set(itertools.permutations(cats)):
        S = lg = 0.0
        for t, j in enumerate(seq):
            lt = lam * ((N - t) / N if finite else 1.0)
            cen = min((0.5 + S) / (t + 1), KAPPA / lt)
            tail = S / (N - t) if finite else 0.0
            lg += math.log1p(lt * (vals[j] - cen)) + lt * (cen + tail)
            S += vals[j]
        tot.append(lg)
    m = max(tot)
    return m + math.log(sum(math.exp(v - m) for v in tot)) - math.log(len(tot))


def test_dynamic_programme_matches_every_order():
    worst = 0.0
    for n in range(1, 7):
        for smp in compositions(n):
            for lam in (0.3, 1.0, 3.0):
                for N in (n, n + 2, math.inf):
                    dp = float(_stratum_logavg(list(smp), [0.0, 0.5, 1.0], np.array([lam]), N)[0])
                    worst = max(worst, abs(dp - brute_logavg(list(smp), [0.0, 0.5, 1.0], lam, N)))
    assert worst < 1e-9


def test_evalue_at_the_truth_is_at_most_one():
    """E[e-value at the true mean] over every sample of every population of values 0, 1/2, 1 (N up to 6): at most 1
    drawn without replacement, and for a point-frame reading deflated by lambda n (n - 1) / (2 (N - n + 1)). The plain
    point-frame reading of a without-replacement draw also stays at 1 on this grid (the judge's self-test found the
    same, aicr job 1525456); the deflation is kept because no proof covers it without."""
    worst_wor = worst_defl = 0.0
    for N in (2, 3, 4, 6):
        for n in range(1, N + 1):
            samples = compositions(n)
            for lam in (0.2, 0.6, 1.0, 2.0, 5.0):
                wor = {s: float(_stratum_logavg(list(s), [0.0, 0.5, 1.0], np.array([lam]), N)[0]) for s in samples}
                wr = {s: float(_stratum_logavg(list(s), [0.0, 0.5, 1.0], np.array([lam]), math.inf)[0])
                      for s in samples}
                D = lam * n * (n - 1) / (2.0 * (N - n + 1))
                for M in compositions(N):
                    mu = (0.5 * M[1] + M[2]) / N
                    e_wor = e_defl = 0.0
                    for s in samples:
                        lp = sum(log_choose(M[j], s[j]) for j in range(3)) - log_choose(N, n)
                        if lp == -math.inf:
                            continue
                        base = lp - lam * n * mu
                        e_wor += math.exp(base + wor[s])
                        e_defl += math.exp(base + wr[s] - D)
                    worst_wor, worst_defl = max(worst_wor, e_wor), max(worst_defl, e_defl)
    assert worst_wor <= 1 + 1e-9
    assert worst_defl <= 1 + 1e-9


# ----------------------------------------------------------------------------------------------- coverage, exactly
@pytest.mark.parametrize("Ns, ns, W", [((6, 4), (3, 2), (0.5, 0.5)), ((6, 4), (3, 2), (0.85, 0.15)),
                                       ((5, 5), (4, 1), (0.3, 0.7)), ((5, 4, 3), (2, 2, 1), (0.6, 0.3, 0.1))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_linear_interval_covers_every_population(Ns, ns, W, alpha):
    d = design_of(ns, Ns, W)
    H = len(Ns)
    ivs = {}
    for ks in itertools.product(*[range(n + 1) for n in ns]):
        x = values(d, [[1.0] * ks[h] + [0.0] * (ns[h] - ks[h]) for h in range(H)])
        ivs[ks] = betting_interval(d, x, alpha=alpha)
    worst = 1.0
    for Ms in itertools.product(*[range(N + 1) for N in Ns]):
        theta = sum(W[h] * Ms[h] / Ns[h] for h in range(H))
        cov = 0.0
        for ks, (lo, hi) in ivs.items():
            lp = sum(log_choose(Ms[h], ks[h]) + log_choose(Ns[h] - Ms[h], ns[h] - ks[h]) - log_choose(Ns[h], ns[h])
                     for h in range(H))
            if lp > -math.inf and lo - 1e-12 <= theta <= hi + 1e-12:
                cov += math.exp(lp)
        worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


@pytest.mark.parametrize("Ns, ns, W", [((4, 4), (2, 2), (0.7, 0.3)), ((5, 3), (3, 2), (0.5, 0.5))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_box_ratio_covers_every_population(Ns, ns, W, alpha):
    """Units of three kinds: in the numerator (y = 1), in the denominator only (w = 1), outside it."""
    d = design_of(ns, Ns, W)
    H = len(Ns)
    ivs = {}
    for ss in itertools.product(*[compositions(n) for n in ns]):
        y = values(d, [[1.0] * s[0] + [0.0] * (s[1] + s[2]) for s in ss])
        w = values(d, [[0.0] * s[0] + [1.0] * s[1] + [0.0] * s[2] for s in ss])
        ivs[ss] = ratio(d, y, w, alpha=alpha)["interval"]
    worst = 1.0
    for pops in itertools.product(*[compositions(N) for N in Ns]):
        ty = sum(W[h] * pops[h][0] / Ns[h] for h in range(H))
        tz = sum(W[h] * (pops[h][0] + pops[h][1]) / Ns[h] for h in range(H))
        if tz <= 0:
            continue
        R = ty / tz
        cov = 0.0
        for ss, (lo, hi) in ivs.items():
            lp = sum(sum(log_choose(pops[h][j], ss[h][j]) for j in range(3)) - log_choose(Ns[h], ns[h])
                     for h in range(H))
            if lp > -math.inf and lo - 1e-12 <= R <= hi + 1e-12:
                cov += math.exp(lp)
        worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_pure_stratum_rule_covers_every_population(alpha):
    """Stratum 0 is pure for the class (5 units, 3 labels), stratum 1 mixed (4 units, 2 labels: mapped as the class
    and right, mapped as it and wrong, mapped otherwise), stratum 2 pure for another class (adds 0)."""
    Ns, ns = (5, 4, 3), (3, 2, 2)
    W = [N / sum(Ns) for N in Ns]
    d = design_of(ns, Ns, W)
    ivs = {}
    for k0 in range(ns[0] + 1):
        for s1 in compositions(ns[1]):
            kr = np.array([k0, s1[0], 0.0])
            kw = np.array([ns[0] - k0, s1[1], 0.0])
            ivs[(k0, s1)] = pure_stratum_ratio(d, 0, [1], kr, kw, alpha)
    worst = 1.0
    for K0 in range(Ns[0] + 1):
        for P1 in compositions(Ns[1]):
            R = (W[0] * K0 / Ns[0] + W[1] * P1[0] / Ns[1]) / (W[0] + W[1] * (P1[0] + P1[1]) / Ns[1])
            cov = 0.0
            for (k0, s1), (lo, hi) in ivs.items():
                lp = (log_choose(K0, k0) + log_choose(Ns[0] - K0, ns[0] - k0) - log_choose(Ns[0], ns[0])
                      + sum(log_choose(P1[j], s1[j]) for j in range(3)) - log_choose(Ns[1], ns[1]))
                if lp > -math.inf and lo - 1e-12 <= R <= hi + 1e-12:
                    cov += math.exp(lp)
            worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_unknown_values_bounded_cover_every_population(alpha):
    """Units of four kinds (0 seen, 0 unseen, 1 seen, 1 unseen) in two strata; an unseen value is bounded by 0 and 1.
    The interval must cover the true mean of all the units, seen or not."""
    Ns, ns, W = (4, 3), (2, 2), (0.6, 0.4)
    d = design_of(ns, Ns, W)
    ivs = {}
    for ss in itertools.product(*[compositions(n, 4) for n in ns]):
        lo_x = values(d, [[0.0] * (s[0] + s[1]) + [1.0] * s[2] + [0.0] * s[3] for s in ss])
        hi_x = values(d, [[0.0] * s[0] + [1.0] * s[1] + [1.0] * (s[2] + s[3]) for s in ss])
        ivs[ss] = betting_interval(d, lo_x, hi_x, alpha)
    worst = 1.0
    for pops in itertools.product(*[compositions(N, 4) for N in Ns]):
        theta = sum(W[h] * (pops[h][2] + pops[h][3]) / Ns[h] for h in range(2))
        cov = 0.0
        for ss, (lo, hi) in ivs.items():
            lp = sum(sum(log_choose(pops[h][j], ss[h][j]) for j in range(4)) - log_choose(Ns[h], ns[h])
                     for h in range(2))
            if lp > -math.inf and lo - 1e-12 <= theta <= hi + 1e-12:
                cov += math.exp(lp)
        worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


@pytest.mark.parametrize("Ns, ns, W", [((5, 4), (3, 2), (0.5, 0.5)), ((5, 4), (4, 3), (0.6, 0.4)),
                                       ((6, 3), (3, 2), (0.5, 0.5))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_point_frame_reading_with_the_deflation_covers_every_population(Ns, ns, W, alpha):
    """An area frame drawn as units (--unit-area): the strata read as point frames (N inf) and deflated for at least
    Nlo units, here exactly the N units each holds, drawn without replacement as the labels were."""
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    d = Design(hidx, W, [math.inf] * len(Ns), Nlo=[float(N) for N in Ns])
    H = len(Ns)
    ivs = {}
    for ks in itertools.product(*[range(n + 1) for n in ns]):
        ivs[ks] = betting_interval(d, values(d, [[1.0] * ks[h] + [0.0] * (ns[h] - ks[h]) for h in range(H)]),
                                   alpha=alpha)
    worst = 1.0
    for Ms in itertools.product(*[range(N + 1) for N in Ns]):
        theta = sum(W[h] * Ms[h] / Ns[h] for h in range(H))
        cov = 0.0
        for ks, (lo, hi) in ivs.items():
            lp = sum(log_choose(Ms[h], ks[h]) + log_choose(Ns[h] - Ms[h], ns[h] - ks[h]) - log_choose(Ns[h], ns[h])
                     for h in range(H))
            if lp > -math.inf and lo - 1e-12 <= theta <= hi + 1e-12:
                cov += math.exp(lp)
        worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


def test_the_deflation_widens_the_interval_by_its_formula():
    from oe_inferencex.stratified import _log_parts, _root_low, _strata, c_grid
    ns, W = (30, 20), (0.7, 0.3)
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    x = np.array([1.0] * 18 + [0.0] * 12 + [1.0] * 10 + [0.0] * 10)
    plain = Design(hidx, W, [math.inf, math.inf])
    defl = Design(hidx, W, [math.inf, math.inf], Nlo=[3000.0, 2000.0])
    lo_p, hi_p = betting_interval(plain, x)
    lo_d, hi_d = betting_interval(defl, x)
    assert 0 < lo_d < lo_p < hi_p < hi_d < 1                    # a sign slip in the deflation would narrow it
    cs = c_grid(defl.W, defl.n, 0.05)
    D = cs * (0.7 * 29 / (2 * (3000 - 30 + 1)) + 0.3 * 19 / (2 * (2000 - 20 + 1)))
    A = _log_parts(plain, _strata(plain, x), cs, "low") - D
    assert lo_d == pytest.approx(_root_low(A, cs, math.log(2 / 0.05), 1.0), abs=1e-12)


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_three_valued_difference_covers_every_population(alpha):
    """The difference of two maps' accuracies: each unit is 0, 1/2 or 1 (b right only, both or neither, a right
    only), theta the weighted mean."""
    Ns, ns, W = (4, 3), (2, 2), (0.6, 0.4)
    d = design_of(ns, Ns, W)
    ivs = {}
    for ss in itertools.product(*[compositions(n) for n in ns]):
        ivs[ss] = betting_interval(d, values(d, [[0.0] * s[0] + [0.5] * s[1] + [1.0] * s[2] for s in ss]), alpha=alpha)
    worst = 1.0
    for pops in itertools.product(*[compositions(N) for N in Ns]):
        theta = sum(W[h] * (0.5 * pops[h][1] + pops[h][2]) / Ns[h] for h in range(2))
        cov = 0.0
        for ss, (lo, hi) in ivs.items():
            lp = sum(sum(log_choose(pops[h][j], ss[h][j]) for j in range(3)) - log_choose(Ns[h], ns[h])
                     for h in range(2))
            if lp > -math.inf and lo - 1e-12 <= theta <= hi + 1e-12:
                cov += math.exp(lp)
        worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_box_ratio_with_an_unsampled_stratum_covers_every_population(alpha):
    """Two sampled strata and a third of 3 units with no label, whose composition is anything: the box must cover
    the ratio over all three."""
    Ns, ns, Nu = (4, 3), (2, 2), 3
    W = [N / (sum(Ns) + Nu) for N in Ns]
    Wu = Nu / (sum(Ns) + Nu)
    d = Design(np.concatenate([np.full(n, h) for h, n in enumerate(ns)]), W, [float(N) for N in Ns], W_unsampled=Wu)
    ivs = {}
    for ss in itertools.product(*[compositions(n) for n in ns]):
        y = values(d, [[1.0] * s[0] + [0.0] * (s[1] + s[2]) for s in ss])
        w = values(d, [[0.0] * s[0] + [1.0] * s[1] + [0.0] * s[2] for s in ss])
        ivs[ss] = ratio(d, y, w, alpha=alpha)["interval"]
    worst = 1.0
    for pops in itertools.product(*[compositions(N) for N in Ns]):
        for pu in compositions(Nu):
            ty = sum(W[h] * pops[h][0] / Ns[h] for h in range(2)) + Wu * pu[0] / Nu
            tz = sum(W[h] * (pops[h][0] + pops[h][1]) / Ns[h] for h in range(2)) + Wu * (pu[0] + pu[1]) / Nu
            if tz <= 0:
                continue
            cov = 0.0
            for ss, (lo, hi) in ivs.items():
                lp = sum(sum(log_choose(pops[h][j], ss[h][j]) for j in range(3)) - log_choose(Ns[h], ns[h])
                         for h in range(2))
                if lp > -math.inf and lo - 1e-12 <= ty / tz <= hi + 1e-12:
                    cov += math.exp(lp)
            worst = min(worst, cov)
    assert worst >= 1 - alpha - 1e-12


def test_a_stratum_too_large_leaves_its_interval_uncomputed(monkeypatch):
    from oe_inferencex import stratified
    monkeypatch.setattr(stratified, "MAX_WORK", 1)
    res = olofsson_result()
    oa = res["overall_accuracy"]["map"]
    assert oa["interval"] is None and "not computed" in oa["not_computed"]
    assert oa["estimate"] == pytest.approx(0.9465, abs=1e-4) and oa["normal_theory"] is not None
    assert res["per_class"]["map"]["1"]["producer"]["interval"] is None
    assert any("interval(s) are not computed" in n for n in res["notes"])
    pure = olofsson_result(pure={"1": "1"})["per_class"]["map"]["1"]["user"]
    assert pure["interval"] is not None                         # the pure-stratum rule needs no average over orders


def test_a_census_stratum_and_the_unsampled_share():
    d = Design([0, 0, 1, 1], [0.5, 0.3], [2.0, 10.0], W_unsampled=0.2)
    lo, hi = betting_interval(d, [1.0, 1.0, 0.0, 1.0])
    assert lo <= 0.5 + 0.3 * 0.5 <= hi
    assert hi <= 1.0 and hi >= 0.5 + 0.2                        # the unsampled share counts at the upper end
    with pytest.raises(ValueError, match="only 1 units"):
        Design([0, 0], [1.0], [1.0])
    with pytest.raises(ValueError, match="add up to"):
        Design([0, 1], [0.5, 0.4], [10.0, 10.0])


# ----------------------------------------------------------------------------------------------- Olofsson et al. 2014
# Olofsson, Foody, Herold, Stehman, Woodcock and Wulder (2014), Remote Sensing of Environment 148, 42-57, the worked
# example of section 5: four map classes (deforestation, forest gain, stable forest, stable non-forest) as strata,
# their mapped areas in 30 m pixels, and the error matrix of 640 labels (rows map, columns reference). Published:
# overall accuracy 0.947 +/- 0.019; user's 0.88, 0.73, 0.93, 0.96 (+/- 0.07, 0.10, 0.04, 0.02); producer's 0.75 +/-
# 0.21, 0.85, 0.93, 0.96; deforestation 21,158 +/- 6,158 ha.
OLOFSSON_PIXELS = {"1": 200000, "2": 150000, "3": 3200000, "4": 6450000}
OLOFSSON_MATRIX = [[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]]


def olofsson_points():
    rows = []
    for i, row in enumerate(OLOFSSON_MATRIX):
        for j, k in enumerate(row):
            rows += [{"stratum": str(i + 1), "map": str(i + 1), "ref": str(j + 1)}] * k
    return rows


def olofsson_result(**kw):
    pts = olofsson_points()
    d = agency.design_from_sizes([p["stratum"] for p in pts], OLOFSSON_PIXELS)
    return agency.grade(d, [p["ref"] for p in pts], {"map": [p["map"] for p in pts]}, **kw)


def half(q):
    return (q["normal_theory"][1] - q["normal_theory"][0]) / 2


def test_olofsson_worked_example_normal_theory():
    res = olofsson_result()
    oa = res["overall_accuracy"]["map"]
    assert oa["estimate"] == pytest.approx(0.947, abs=1e-3)
    assert half(oa) == pytest.approx(0.019, abs=1e-3)            # 0.01848 by their equation 5
    per = res["per_class"]["map"]
    for c, ua, hw in (("1", 0.88, 0.07), ("2", 0.73, 0.10), ("3", 0.93, 0.04), ("4", 0.96, 0.02)):
        assert per[c]["user"]["estimate"] == pytest.approx(ua, abs=5e-3)
        assert half(per[c]["user"]) == pytest.approx(hw, abs=5e-3)
    for c, pa in (("1", 0.75), ("2", 0.85), ("3", 0.93), ("4", 0.96)):
        assert per[c]["producer"]["estimate"] == pytest.approx(pa, abs=5e-3)
    assert half(per["1"]["producer"]) == pytest.approx(0.21, abs=5e-3)
    ha = 10_000_000 * 0.09                                      # the map's pixels, 0.09 ha each
    share = per["1"]["reference_share"]
    assert share["estimate"] * ha == pytest.approx(21158, abs=2)
    assert half(share) * ha == pytest.approx(6158, abs=2)


def test_olofsson_valid_intervals_hold_the_estimate_and_say_something():
    """Each interval holds its estimate, and is neither narrower than normal theory's nor vacuous: a box collapsing
    to [0, 1] would pass every coverage test (the review of 26 October 2026)."""
    res = olofsson_result()
    for q in [res["overall_accuracy"]["map"]] + [row[k] for row in res["per_class"]["map"].values()
                                                 for k in ("user", "producer", "reference_share")]:
        lo, hi = q["interval"]
        assert lo - 1e-12 <= q["estimate"] <= hi + 1e-12
    width = lambda q, k: q[k][1] - q[k][0]
    oa = res["overall_accuracy"]["map"]
    assert width(oa, "normal_theory") < width(oa, "interval") < 2.0 * width(oa, "normal_theory")
    for c in ("3", "4"):                                        # the two large classes: 165 and 325 labels
        for k in ("user", "producer"):
            q = res["per_class"]["map"][c][k]
            assert width(q, "normal_theory") < width(q, "interval") < 4.0 * width(q, "normal_theory"), (c, k)


def test_olofsson_pure_strata_take_the_pure_stratum_rule():
    res = olofsson_result(pure={"1": "1", "2": "2", "3": "3", "4": "4"})
    for c, row in res["per_class"]["map"].items():
        u = row["user"]
        assert u["method"] == "pure-stratum rule" and u["pure_stratum"] == c
        assert u["interval"][0] <= u["estimate"] <= u["interval"][1]
    box = olofsson_result()["per_class"]["map"]["4"]["user"]["interval"]
    pure = res["per_class"]["map"]["4"]["user"]["interval"]
    assert pure[1] - pure[0] < box[1] - box[0]
    with pytest.raises(ValueError, match="declaration is false"):
        olofsson_result(pure={"1": "2"})


# ----------------------------------------------------------------------------------------------- rules and unknowns
def small_design():
    return agency.design_from_sizes(["a", "a", "a", "b", "b", "b"], {"a": 100, "b": 300})


def test_alternate_label_counts_as_right_and_names_the_reference_class():
    d = small_design()
    ref = ["1", "1", "2", "2", "2", "1"]
    alt = [None, "2", None, None, "1", None]
    m = ["1", "2", "2", "2", "1", "2"]
    plain = agency.grade(d, ref, {"m": m})
    rule = agency.grade(d, ref, {"m": m}, alternate=alt)
    assert plain["overall_accuracy"]["m"]["estimate"] == pytest.approx(0.25 * 2 / 3 + 0.75 * 1 / 3)
    assert rule["overall_accuracy"]["m"]["estimate"] == pytest.approx(0.25 * 1 + 0.75 * 2 / 3)
    assert "alternate" in rule["rule"]
    # the reference class is the map's where it equals the alternate: point 2 (map 2, alternate 2) is class 2 and
    # point 5 (map 1, reference 2, alternate 1) class 1, so class 1 is points 1, 5 and 6
    one = rule["per_class"]["m"]["1"]
    assert one["reference_share"]["estimate"] == pytest.approx(7 / 12)
    assert one["producer"]["estimate"] == pytest.approx(4 / 7)
    em = rule["error_matrix"]["m"]
    assert em["rows_map_columns_reference"] == ["1", "2"]
    assert np.allclose(em["share"], [[1 / 3, 0.0], [1 / 4, 5 / 12]]) and em["unsampled_share"] == 0


def test_unknown_values_make_a_range_inside_the_interval():
    d = small_design()
    res = agency.grade(d, ["1", "1", None, "2", "2", "2"], {"m": ["1", None, "2", "2", "1", "2"]})
    oa = res["overall_accuracy"]["m"]
    assert isinstance(oa["estimate"], list) and oa["estimate"][0] < oa["estimate"][1]
    assert oa["interval"][0] <= oa["estimate"][0] and oa["estimate"][1] <= oa["interval"][1]
    assert res["unknown"] == {"reference": 1, "m": 1}
    assert any("bounded, not dropped" in n for n in res["notes"])


def test_two_maps_give_their_difference():
    d = small_design()
    ref = ["1", "1", "2", "2", "2", "1"]
    res = agency.grade(d, ref, {"a": ref, "b": ref})
    q = res["differences"]["a minus b"]
    assert q["estimate"] == 0 and q["interval"][0] <= 0 <= q["interval"][1] and q["disagreeing_points"] == 0
    res = agency.grade(d, ref, {"a": ref, "b": ["2", "1", "1", "2", "2", "1"]})
    q = res["differences"]["a minus b"]
    assert q["estimate"] == pytest.approx(0.25 * 2 / 3)
    assert -1 <= q["interval"][0] <= q["estimate"] <= q["interval"][1] <= 1


def test_unsampled_strata_are_bounded_or_left_out_on_request():
    sizes = {"a": 100, "b": 300, "c": 100}
    pts = ["a", "a", "b", "b"]
    d = agency.design_from_sizes(pts, sizes)
    assert d.W_unsampled == pytest.approx(0.2)
    res = agency.grade(d, ["1"] * 4, {"m": ["1"] * 4})
    assert res["overall_accuracy"]["m"]["estimate"] == pytest.approx([0.8, 1.0])
    d2 = agency.design_from_sizes(pts, sizes, sampled_only=True)
    res2 = agency.grade(d2, ["1"] * 4, {"m": ["1"] * 4})
    assert res2["overall_accuracy"]["m"]["estimate"] == pytest.approx(1.0)
    assert any("restricted to the 2 sampled strata" in n for n in res2["notes"])


def test_weights_must_be_equal_within_a_stratum_and_sizes_whole():
    with pytest.raises(ValueError, match="weights differ within stratum a"):
        agency.design_from_weights(["a", "a"], [10.0, 11.0])
    d = agency.design_from_weights(["a", "a", "b"], [50.0, 50.0, 30.0])
    assert list(d.N) == [100.0, 30.0]
    with pytest.raises(ValueError, match="not a whole number"):
        agency.design_from_sizes(["a"], {"a": 10.5})
    d = agency.design_from_sizes(["a", "b"], {"a": 10.5, "b": 2.5}, kind="area", unit_area=0.5)
    assert list(d.Nlo) == [21.0, 5.0] and np.isinf(d.N).all()


def test_codes_are_read_as_text_with_whole_numbers_normalised():
    assert agency._code("11.0", ()) == "11" and agency._code(" 52 ", ()) == "52"
    assert agency._code("NA", ()) is None and agency._code("", ()) is None
    assert agency._code("0", {"0"}) is None and agency._code("forest", ()) == "forest"
    # long whole numbers stay distinct (through a float both were 12345678901234567168)
    assert agency._code("12345678901234567890", ()) != agency._code("12345678901234567891", ())
    assert agency._code("1e3", ()) == "1000" and agency._code("-0", ()) == "0"


def test_weights_give_a_size_only_as_far_as_their_printing_allows():
    """The reviews of 26 October 2026: area weights read as units are refused where their printing can tell; one
    whole number within the printing is the size; several, and the stratum is read as points drawn from at least the
    fewest, so a near-census is never made exact and the pure-stratum rule is not used."""
    with pytest.raises(ValueError, match="add up to 37.5, which their printing does not allow"):
        agency.design_from_weights(["a"] * 30, [1.25] * 30, half_unit=[agency._half_unit("1.25")] * 30)
    d = agency.design_from_weights(["a"] * 75 + ["b"] * 325, [2666.67] * 75 + [19846.15] * 325,
                                   half_unit=[agency._half_unit("2666.67")] * 75 + [agency._half_unit("19846.15")] * 325)
    # 2666.67 x 75 = 200000.25 within 0.375: only 200000; 19846.15 x 325 = 6449998.75 within 1.625: 6449998 to 6450000
    assert d.N[0] == 200000.0 and np.isinf(d.N[1]) and d.Nlo[1] == 6449998.0 and np.isnan(d.Nlo[0])
    assert d.point_read == ["b"]
    # without half_unit, each weight's shortest printing, as the command line reads a table
    assert agency.design_from_weights(["a"] * 75, [2666.67] * 75).N[0] == 200000.0
    assert agency._half_unit("2666.67") == pytest.approx(0.005) and agency._half_unit("1.5e3") == pytest.approx(50)
    # a near-census printed as whole numbers: 9 labels of a stratum of 10 units, weight 10/9 printed '1', allows 5 to
    # 13 units; read as points from at least 9, and the pure-stratum rule (which made it a census) is not used
    d = agency.design_from_weights(["p"] * 9 + ["q"] * 10, [1.0] * 9 + [10.0] * 10,
                                   half_unit=[agency._half_unit("1")] * 9 + [agency._half_unit("10.00")] * 10)
    assert d.point_read == ["p"] and d.Nlo[0] == 9.0 and d.N[1] == 100.0
    res = agency.grade(d, ["1"] * 8 + ["2"] + ["2"] * 10, {"m": ["1"] * 9 + ["2"] * 10}, pure={"p": "1", "q": "2"})
    assert res["per_class"]["m"]["1"]["user"]["method"] == "box of two betting intervals"
    assert any("size is not known in whole units" in n for n in res["notes"])
    assert any("printed too coarsely" in n for n in res["notes"])


def test_pure_stratum_notes_name_the_classes_they_skip():
    pts = ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3
    d = agency.design_from_sizes(pts, {"s1": 100, "s2": 100, "s3": 100, "s4": 100})
    m = ["1"] * 6 + ["2"] * 3 + ["1", "2", "3"]
    res = agency.grade(d, ["1", "1", "2"] * 4, {"m": m}, pure={"s1": "1", "s2": "1", "s3": "2"})
    assert res["per_class"]["m"]["2"]["user"]["method"] == "pure-stratum rule"
    assert res["per_class"]["m"]["1"]["user"]["method"] == "box of two betting intervals"
    assert not any("was not used for m (" in n for n in res["notes"])
    assert any("not used for class(es) 1 of m" in n for n in res["notes"])


def test_the_unknown_note_counts_points_and_two_maps_agreeing_differ_by_nothing():
    d = small_design()
    ref = [None, "1", "2", None, "2", "1"]
    maps = {"a": ["1", "1", "2", "2", "2", "1"], "b": ["1", "1", "2", "2", "2", "1"]}
    res = agency.grade(d, ref, maps)
    assert any(n.startswith("2 point(s) have an unknown value") for n in res["notes"])
    q = res["differences"]["a minus b"]
    assert q["estimate"] == 0 and q["interval"][0] <= 0 <= q["interval"][1]
    both = agency.grade(d, [None, "1", "2", "2", "2", "1"], {"m": [None, "1", "2", "2", "2", "1"]})
    assert any(n.startswith("1 point(s) have an unknown value") for n in both["notes"])


def test_the_error_matrix_says_what_it_leaves_out_and_the_note_says_normal_ends_are_unclipped():
    d = agency.design_from_sizes(["a", "a", "b", "b"], {"a": 100, "b": 300, "c": 100})
    res = agency.grade(d, ["1"] * 4, {"m": ["1"] * 4})
    em = res["error_matrix"]["m"]
    assert em["share"] == [[pytest.approx(0.8)]] and em["unsampled_share"] == pytest.approx(0.2)
    assert "unclipped" in res["notes"][0] and "1.96 standard errors" in res["notes"][0]


# ----------------------------------------------------------------------------------------------- the command line
def write_points(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_cli_grades_the_olofsson_example_from_weights_and_from_sizes(tmp_path, capsys):
    pts = olofsson_points()
    n = {s: sum(p["stratum"] == s for p in pts) for s in OLOFSSON_PIXELS}
    # weights to six decimals, as agencies print them: each stratum's size is then one whole number (weights printed as
    # 2000.0 would leave 150,000 +/- 3.75 open, and that stratum would be read as points)
    rows = [dict(p, weight=f"{OLOFSSON_PIXELS[p['stratum']] / n[p['stratum']]:.6f}", other=p["ref"]) for p in pts]
    path = tmp_path / "points.csv"
    write_points(path, rows, ["stratum", "map", "ref", "weight", "other"])
    out = tmp_path / "g.json"
    assert main(["grade", str(path), "--stratum", "stratum", "--reference", "ref", "--map", "map", "--map", "other",
                 "--weight", "weight", "--pure", "1=1", "2=2", "3=3", "4=4", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "overall accuracy" in text and "map minus other" in text and "pure-stratum rule" in text
    res = json.load(open(out))
    assert res["written_by"]["command"] == "grade"
    assert res["overall_accuracy"]["map"]["estimate"] == pytest.approx(0.9465, abs=1e-4)
    assert res["overall_accuracy"]["other"]["estimate"] == pytest.approx(1.0)
    sizes = tmp_path / "sizes.csv"
    with open(sizes, "w") as f:
        f.write("stratum,pixels\r\n" + "".join(f"{s},{v}\r\n" for s, v in OLOFSSON_PIXELS.items()))
    out2 = tmp_path / "g2.json"
    assert main(["grade", str(path), "--stratum", "stratum", "--reference", "ref", "--map", "map", "--strata",
                 str(sizes), "--out", str(out2)]) == 0
    res2 = json.load(open(out2))
    assert res2["overall_accuracy"]["map"]["interval"] == pytest.approx(res["overall_accuracy"]["map"]["interval"])


def test_cli_refusals(tmp_path):
    path = tmp_path / "p.csv"
    write_points(path, [{"s": "a", "m": "1", "r": "1", "w": "10"}, {"s": "a", "m": "2", "r": "1", "w": "12"}],
                 ["s", "m", "r", "w"])
    base = ["grade", str(path), "--stratum", "s", "--reference", "r", "--map", "m"]
    with pytest.raises(SystemExit, match="weights differ"):
        main(base + ["--weight", "w"])
    with pytest.raises(SystemExit, match="no column 'x'"):
        main(["grade", str(path), "--stratum", "x", "--reference", "r", "--map", "m", "--weight", "w"])
    with pytest.raises(SystemExit, match="--unit-area is for sizes given as areas"):
        main(base + ["--weight", "w", "--unit-area", "1"])
    with pytest.raises(SystemExit, match="STRATUM=CLASS"):
        main(base + ["--weight", "w", "--pure", "a"])
    with pytest.raises(SystemExit, match="--alpha"):
        main(base + ["--weight", "w", "--alpha", "1.5"])
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(base + ["--weight", "w", "--out", str(path)])
    with pytest.raises(SystemExit, match="must differ from each other, from the stratum"):
        main(["grade", str(path), "--stratum", "s", "--reference", "r", "--map", "r", "--weight", "w"])


def test_cli_reads_a_table_as_a_spreadsheet_shows_it(tmp_path, capsys):
    """The reviews of 26 October 2026: a byte-order mark, a column named twice, a short row and a long one, pandas'
    and R's row index, a trailing delimiter, a sample stratified by the map's own column, and the sizes file's header."""
    def run(text, *extra, stratum="s"):
        p = tmp_path / "t.csv"
        p.write_text(text, encoding="utf-8")
        return main(["grade", str(p), "--stratum", stratum, "--reference", "r", "--map", "m", *extra])
    assert run("﻿s,m,r,w\na,1,1,10\na,2,1,10\nb,1,1,5\nb,1,2,5\n", "--weight", "w") == 0
    assert "overall accuracy" in capsys.readouterr().out
    # pandas' default index column (unnamed), an unused column named twice and a trailing delimiter are read
    assert run(",s,m,r,w,note,note\n0,a,1,1,10,x,y\n1,a,2,1,10,,\n2,b,1,1,5,,\n3,b,1,2,5,,\n", "--weight", "w") == 0
    assert run("s,m,r,w\na,1,1,10,\na,2,1,10,\nb,1,1,5,\nb,1,2,5,\n", "--weight", "w") == 0
    with pytest.raises(SystemExit, match="names the column 'r' twice"):
        run("s,r,m,w,r\na,1,1,10,2\n", "--weight", "w")
    with pytest.raises(SystemExit, match="line 3: weight '' is not a number"):
        run("s,m,r,w\na,1,1,10\na,1\n", "--weight", "w")
    with pytest.raises(SystemExit, match="line 2: more fields"):
        run("s,m,r,w\na,1,1,10,5\n", "--weight", "w")
    # stratified by the map's class: the stratum is the map's own column
    assert run("m,r,w\n1,1,10\n1,2,10\n2,2,5\n2,2,5\n", "--weight", "w", stratum="m") == 0
    capsys.readouterr()
    sizes = tmp_path / "sizes.csv"
    sizes.write_text("a,30\nb,60\n")
    with pytest.raises(SystemExit, match="looks like a stratum, not a header"):
        run("s,m,r\na,1,1\nb,1,1\n", "--strata", str(sizes))
    sizes.write_text("﻿stratum,size\na,30\nb,60\n", encoding="utf-8")
    assert run("s,m,r\na,1,1\nb,1,1\n", "--strata", str(sizes)) == 0
    sizes.write_text("stratum,2016\na,30\nb,60\n")              # a header whose size column is named by a year
    assert run("s,m,r\na,1,1\nb,1,1\n", "--strata", str(sizes)) == 0
    sizes.write_text('"","stratum","size"\n"1","a",30\n"2","b",60\n')   # R's write.csv with row names
    with pytest.raises(SystemExit, match="first column has no name"):
        run("s,m,r\na,1,1\nb,1,1\n", "--strata", str(sizes))

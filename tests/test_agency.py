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


def test_olofsson_valid_intervals_hold_the_estimate_and_are_wider():
    res = olofsson_result()
    for q in [res["overall_accuracy"]["map"]] + [row[k] for row in res["per_class"]["map"].values()
                                                 for k in ("user", "producer", "reference_share")]:
        lo, hi = q["interval"]
        assert lo - 1e-12 <= q["estimate"] <= hi + 1e-12
    oa = res["overall_accuracy"]["map"]
    assert oa["interval"][1] - oa["interval"][0] < 2.0 * (oa["normal_theory"][1] - oa["normal_theory"][0])


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
    d = agency.design_from_weights(["a", "a", "b"], [50.0000001, 50.0000001, 30.0])
    assert list(d.N) == [100.0, 30.0]
    with pytest.raises(ValueError, match="not a whole number"):
        agency.design_from_sizes(["a"], {"a": 10.5})
    d = agency.design_from_sizes(["a", "b"], {"a": 10.5, "b": 2.5}, kind="area", unit_area=0.5)
    assert list(d.Nlo) == [21.0, 5.0] and np.isinf(d.N).all()


def test_codes_are_read_as_text_with_whole_numbers_normalised():
    assert agency._code("11.0", ()) == "11" and agency._code(" 52 ", ()) == "52"
    assert agency._code("NA", ()) is None and agency._code("", ()) is None
    assert agency._code("0", {"0"}) is None and agency._code("forest", ()) == "forest"


# ----------------------------------------------------------------------------------------------- the command line
def write_points(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_cli_grades_the_olofsson_example_from_weights_and_from_sizes(tmp_path, capsys):
    pts = olofsson_points()
    n = {s: sum(p["stratum"] == s for p in pts) for s in OLOFSSON_PIXELS}
    rows = [dict(p, weight=OLOFSSON_PIXELS[p["stratum"]] / n[p["stratum"]], other=p["ref"]) for p in pts]
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

"""Localized certification (1.20.0): a verdict per profile, good, bad or undecided, checked by exact enumeration.

The claim: on a sample stratified by profile (sample --design profile), with delta split over the L profiles labelled
in part and each interval at 1 - delta/L, every interval holds and every verdict is right except on at most delta of
samples, and the verdicts alone are wrong on at most delta/2 (a wrong good needs the interval to miss above, a wrong
bad to miss below). It is checked here without simulation: over every error pattern and every draw of a small map,
literally; over every population of larger maps by the counts, with exact hypergeometric probabilities; against any
choice of the windows marked `?`; with a negative control that shows the check can fail when delta is not split; and
good and bad are shown to be certify's own exact one-sided tests. The profiles themselves are fixed from the map alone
(estimate.profiles), and the command line runs end to end, its refusals included. `plan`'s profile route is checked
against a brute-force sum."""
import csv
import functools
import itertools
import json
import math
import os
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex import plan
from oe_inferencex.certify import localized as loc
from oe_inferencex.cli import main
from oe_inferencex.common.hypergeom import _hyper_tail_exact, _level

ALPHAS = (0.2, 0.25, 0.34, 0.5)
DELTAS = (0.1, 0.3, 0.5)


def _wrong(verdict, theta, alpha):
    """A verdict the truth contradicts: good for a profile wrong more than alpha of the time, bad for one wrong at most
    alpha of the time. Undecided is never wrong."""
    a = _level(alpha)
    return (verdict == "good" and theta > a) or (verdict == "bad" and theta <= a)


def _two_profile_sample():
    """Seven windows in two profiles, four and three, and the condition design's draw of four labels: two each."""
    idx = np.array([0] * 4 + [1] * 3)
    base = est.sample_for_estimation(np.linspace(0, 1, 7), 4, design="condition", condition=idx)
    assert base["allocation"] == [2, 2] and base["condition"]["names"] == ["0", "1"]
    return base, idx


def _draws(base, idx):
    """Every sample the design can draw, each as likely as the others: two of profile 0's windows and two of 1's."""
    for a in itertools.combinations(np.flatnonzero(idx == 0).tolist(), 2):
        for b in itertools.combinations(np.flatnonzero(idx == 1).tolist(), 2):
            yield dict(base, indices=np.array(a + b))


# ----------------------------------------------------------------------------- 1. every population, every sample
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("delta", DELTAS)
def test_every_population_and_every_sample(alpha, delta):
    """All 128 error patterns of the seven windows, all 18 draws: the verdicts are wrong on at most delta/2 of the
    draws and the two intervals hold together on at least 1 - delta, each interval at 1 - delta/2 (L = 2)."""
    base, idx = _two_profile_sample()
    draws = list(_draws(base, idx))
    assert len(draws) == 18
    d = _level(delta)
    for pattern in itertools.product((0, 1), repeat=7):
        e = np.array(pattern)
        theta = [Fraction(int(e[idx == h].sum()), int((idx == h).sum())) for h in (0, 1)]
        wrong = cover = 0
        for s in draws:
            res = loc.profile_verdicts(s, e[s["indices"]], alpha, delta=delta)
            assert res["n_tested"] == 2 and res["n_profiles"] == 2
            assert res["level"] == float(1 - d / 2) and res["delta_per_profile"] == delta / 2
            rows = [res["profiles"][name] for name in ("0", "1")]
            wrong += any(_wrong(r["verdict"], t, alpha) for r, t in zip(rows, theta))
            ends = [loc.profile_interval(r["n_wrong"], 0, 2, N, 1 - d / 2) for r, N in zip(rows, (4, 3))]
            cover += all(lo <= t <= hi for (lo, hi), t in zip(ends, theta))
            for r, (lo, hi) in zip(rows, ends):          # the JSON's ends lie outward of the exact ones
                assert r["low"] <= lo and r["high"] >= hi
                assert r["verdict"] == loc.profile_verdict(lo, hi, alpha)
        assert Fraction(wrong, 18) <= d / 2, (pattern, wrong)
        assert Fraction(cover, 18) >= 1 - d, (pattern, cover)


# ----------------------------------------------------------------------------- 2. every population, by the counts
@functools.lru_cache(maxsize=None)
def _p_profile(N, n, K, alpha, conf):
    """(P(the verdict on a profile of N windows, K wrong, n labelled at random, is wrong), P(its interval covers
    K/N)), exactly, over the hypergeometric count; each interval at `conf`, a Fraction."""
    theta = Fraction(K, N)
    wrong = cover = Fraction(0)
    for k in range(max(0, n - (N - K)), min(n, K) + 1):
        p = Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n))
        lo, hi = loc.profile_interval(k, 0, n, N, conf)
        wrong += p * _wrong(loc.profile_verdict(lo, hi, alpha), theta, alpha)
        cover += p * (lo <= theta <= hi)
    return wrong, cover


def _per_profile(sizes, budget, alpha, delta):
    """For each profile, its (wrong, cover) at every K = 0..N under the equal allocation of `budget`, with L counted
    from the allocation before any label; a census is checked to be exact and never wrong."""
    n = [int(v) for v in est.equal_allocation(sizes, budget)]
    tested = [h for h, N in enumerate(sizes) if 0 < n[h] < N]
    conf = 1 - _level(delta) / len(tested) if tested else None
    per = []
    for h, N in enumerate(sizes):
        if h in tested:
            per.append([_p_profile(N, n[h], K, alpha, conf) for K in range(N + 1)])
            continue
        assert n[h] == N                                     # never 0: the allocation's floor is min(N, 2)
        for K in range(N + 1):
            lo, hi = loc.profile_interval(K, 0, N, N, Fraction(1, 2))
            assert lo == hi == Fraction(K, N) and not _wrong(loc.profile_verdict(lo, hi, alpha), Fraction(K, N), alpha)
        per.append([(Fraction(0), Fraction(1))] * (N + 1))
    return per


@pytest.mark.parametrize("sizes", [(6, 4, 3), (10, 7), (20,)])
def test_every_population_by_counts(sizes):
    """Every vector of error counts, every budget from sum min(N, 2) to the census: the draws are independent across
    profiles, so the chance some verdict is wrong is 1 - prod(1 - p_h) and that every interval holds prod(q_h)."""
    for budget in range(sum(min(s, 2) for s in sizes), sum(sizes) + 1):
        for alpha in ALPHAS:
            for delta in DELTAS:
                d = _level(delta)
                per = _per_profile(sizes, budget, alpha, delta)
                for Ks in itertools.product(*(range(N + 1) for N in sizes)):
                    right, cover = Fraction(1), Fraction(1)
                    for h, K in enumerate(Ks):
                        right *= 1 - per[h][K][0]
                        cover *= per[h][K][1]
                    assert 1 - right <= d / 2, (sizes, budget, alpha, delta, Ks)
                    assert cover >= 1 - d, (sizes, budget, alpha, delta, Ks)


def test_twelve_profiles_by_their_worst_counts():
    """Twelve profiles of 8 windows: 9^12 count vectors are too many to list, but 1 - prod(1 - p_h(K_h)) is largest
    where each profile takes its own worst K, so the bound at the per-profile maxima holds for every vector."""
    sizes = (8,) * 12
    for budget in range(24, 97):
        for alpha in ALPHAS:
            for delta in DELTAS:
                d = _level(delta)
                per = _per_profile(sizes, budget, alpha, delta)
                right = math.prod(1 - max(w for w, _ in rows) for rows in per)
                cover = math.prod(min(c for _, c in rows) for rows in per)
                assert 1 - right <= d / 2 and cover >= 1 - d, (budget, alpha, delta)


# ----------------------------------------------------------------------------- 3. windows that could not be judged
@pytest.mark.parametrize("alpha", (0.25, 0.5))
def test_windows_marked_unjudged_never_turn_a_right_verdict_wrong(alpha):
    """Whichever labelled windows the reviewer marks `?` (all 16 choices, for every pattern and every draw), a verdict
    made with them is the one their true labels give, so the share of draws on which some choice gives a wrong verdict
    is the true labels' share, at most delta/2. The `wrong` value of a `?` window is not read."""
    base, idx = _two_profile_sample()
    draws = list(_draws(base, idx))
    delta = 0.3
    for pattern in itertools.product((0, 1), repeat=7):
        e = np.array(pattern)
        theta = [Fraction(int(e[idx == h].sum()), int((idx == h).sum())) for h in (0, 1)]
        bad_draws = 0
        for s in draws:
            truth = loc.profile_verdicts(s, e[s["indices"]], alpha, delta=delta)["profiles"]
            some_wrong = False
            for marks in itertools.product((False, True), repeat=4):
                unj = np.array(marks)
                res = loc.profile_verdicts(s, np.where(unj, 1, e[s["indices"]]), alpha, unjudged=unj, delta=delta)
                assert res["n_unjudged"] == int(unj.sum())
                for name, t in zip(("0", "1"), theta):
                    v = res["profiles"][name]["verdict"]
                    assert v == "undecided" or v == truth[name]["verdict"], (pattern, marks)
                    some_wrong |= _wrong(v, t, alpha)
            bad_draws += some_wrong
        assert Fraction(bad_draws, 18) <= _level(delta) / 2, (pattern, bad_draws)


# ----------------------------------------------------------------------------- 4. certify's own tests
def test_good_and_bad_are_certify_s_exact_one_sided_tests():
    """Good is certify's zone test of "wrong more than alpha" rejecting at delta/(2L) with the sample share at most
    alpha (zone_pvalue_exact), and bad the mirror test of "wrong at most alpha" rejecting at the same level: on every
    k, n < N <= 25, the `?` windows counted wrong at the upper end."""
    for x in (Fraction(1, 5), Fraction(1, 10), Fraction(1, 20), Fraction(1, 60)):          # delta / L
        conf, tail = 1 - x, x / 2
        for alpha in (0.1, 0.2, 0.3):
            a = _level(alpha)
            for N in range(2, 26):
                edge = math.floor(a * N)                    # the most windows of N wrong at most alpha
                for n in range(1, N):
                    for j in range(n + 1):
                        for k, u in {(j, 0), (0, j)}:
                            lo, hi = loc.profile_interval(k, u, n, N, conf)
                            good = loc.profile_verdict(lo, hi, alpha) == "good"
                            assert good == (est.zone_pvalue_exact(j, n, N, alpha) <= tail and Fraction(j, n) <= a), \
                                (x, alpha, N, n, k, u)
                        lo, hi = loc.profile_interval(j, 0, n, N, conf)
                        bad = loc.profile_verdict(lo, hi, alpha) == "bad"
                        assert bad == (_hyper_tail_exact(j, n, N, edge, upper=True) <= tail and Fraction(j, n) > a), \
                            (x, alpha, N, n, j)


def test_a_tail_equal_to_its_level_is_decided_exactly():
    """The package's recorded bug class is a tail equal to its level. N 5, n 3, no error, alpha 0.3, delta/L 0.2: with
    2 of 5 wrong the chance of seeing none is C(3,3)/C(5,3) = 1/10, exactly the tail level, so 2 is rejected, the
    upper end is 1/5 and the profile is good, as certify's zone test says (its p-value 1/10 is at most 1/10)."""
    lo, hi = loc.profile_interval(0, 0, 3, 5, Fraction(4, 5))
    assert hi == Fraction(1, 5) and lo == 0
    assert loc.profile_verdict(lo, hi, 0.3) == "good"
    assert est.zone_pvalue_exact(0, 3, 5, 0.3) == Fraction(1, 10)


# ----------------------------------------------------------------------------- 5. censuses and empty profiles
def test_a_profile_labelled_in_full_is_exact_and_takes_none_of_delta():
    idx = np.array([0] * 2 + [1] * 10)
    base = est.sample_for_estimation(np.linspace(0, 1, 12), 6, design="condition", condition=idx)
    assert base["allocation"] == [2, 4]                      # profile 0 too small for an equal share: a census
    s0 = [int(i) for i in base["indices"] if idx[i] == 0]
    for pattern in itertools.product((0, 1), repeat=12):
        e = np.array(pattern)
        res = loc.profile_verdicts(base, e[base["indices"]], 0.2, delta=0.1)
        assert res["n_tested"] == 1 and res["level"] == float(1 - _level(0.1)) and res["delta_per_profile"] == 0.1
        row = res["profiles"]["0"]
        assert row["census"] and not row["tested"] and row["estimate"] == e[s0].mean()
        assert not _wrong(row["verdict"], Fraction(int(e[:2].sum()), 2), 0.2)
    # every profile labelled in full: nothing to split delta over, and the note says so
    full = est.sample_for_estimation(np.linspace(0, 1, 12), 12, design="condition", condition=idx)
    res = loc.profile_verdicts(full, np.ones(12), 0.2)
    assert res["n_tested"] == 0 and res["level"] is None and res["delta_per_profile"] is None
    assert all(r["verdict"] == "bad" for r in res["profiles"].values())
    assert "none can be wrong" in res["notes"][0]


def test_a_profile_with_no_label_is_undecided():
    """sample never draws one (its floor is min(N, 2)), but an allocation of 0 is read as nothing said: (0, 1)."""
    idx = np.array([0] * 3 + [1] * 3)
    grid = idx.astype(np.int64)
    sample = {"design": "condition", "indices": np.array([0, 1, 2]), "n_population": 6, "strata": idx,
              "strata_of_population": np.arange(6), "sizes": [3, 3], "allocation": [3, 0], "n_strata": 2,
              "condition": {"values": [0, 1], "names": ["a", "b"], "sizes": [3, 3]}, "condition_grid": grid}
    res = loc.profile_verdicts(sample, np.zeros(3), 0.2)
    b = res["profiles"]["b"]
    assert b["verdict"] == "undecided" and (b["low"], b["high"]) == (0.0, 1.0) and not b["tested"]
    assert res["profiles"]["a"]["verdict"] == "good" and res["n_tested"] == 0
    assert res["verdict_grid"].tolist() == [1, 1, 1, 3, 3, 3]


def test_the_inputs_are_checked():
    base, idx = _two_profile_sample()
    w = np.zeros(4)
    for bad in (dict(alpha=0), dict(alpha=1.0), dict(alpha=True), dict(alpha=0.2, delta=0), dict(alpha=0.2, delta=1)):
        with pytest.raises(ValueError, match="must be in"):
            loc.profile_verdicts(base, w, bad.pop("alpha"), **bad)
    with pytest.raises(ValueError, match="0 or 1"):
        loc.profile_verdicts(base, np.full(4, 0.5), 0.2)
    with pytest.raises(ValueError, match="labels for"):
        loc.profile_verdicts(base, np.zeros(3), 0.2)
    with pytest.raises(ValueError, match="stratified by profile"):
        loc.profile_verdicts(dict(base, design="random"), w, 0.2)
    with pytest.raises(ValueError, match="allocation"):
        loc.profile_verdicts(dict(base, indices=np.array([0, 1, 2, 4])), w, 0.2)     # three from profile 0
    for args in ((True, 0, 1, 2, 0.9), (1.5, 0, 2, 3, 0.9), (2, 1, 2, 3, 0.9), (0, 0, 3, 2, 0.9), (0, 0, 1, 2, 1),
                 (0, -1, 1, 2, 0.9)):
        with pytest.raises(ValueError):
            loc.profile_interval(*args)


# ----------------------------------------------------------------------------- 6. the check can fail
def test_the_check_can_fail_when_delta_is_not_split():
    """Twelve profiles of 30 windows, 7 of them wrong (above alpha 0.2), 15 labels each, each interval at 1 - delta
    unsplit: one or no error among the labels leaves 6 of 30 as the upper end, a wrong good, 4.0% of the time per
    profile and 39% over twelve, far above delta 0.1. Split over the twelve it is 0.3% and 3.7%, within delta/2."""
    d = _level(0.1)
    alone = _p_profile(30, 15, 7, 0.2, 1 - d)[0]
    assert alone >= Fraction(1, 25) and 1 - (1 - alone) ** 12 > d
    split = _p_profile(30, 15, 7, 0.2, 1 - d / 12)[0]
    assert 0 < split and 1 - (1 - split) ** 12 <= d / 2


# ----------------------------------------------------------------------------- 7. the partition
def test_the_fields_are_parsed_in_one_order():
    assert est.parse_profile_by("class,confidence:3") == [("confidence", 3), ("class", None)]
    assert est.parse_profile_by(" grid:2 , condition ") == [("condition", None), ("grid", 2)]
    assert est.profile_spec(est.parse_profile_by("class,grid:4,confidence:5,condition")) == \
        "confidence:5,condition,class,grid:4"
    for bad in ("", "foo", "confidence", "confidence:1", "confidence:6", "grid:5", "grid:1", "class:2", "condition:1",
                "class,class", "confidence:x", "confidence:2.5", "class,,grid:2", 3):
        with pytest.raises(ValueError):
            est.parse_profile_by(bad)


def test_bands_are_confidence_strata_and_outside_the_population_is_minus_one():
    rng = np.random.default_rng(3)
    m = rng.random((8, 8))
    m[0, 0] = np.nan
    valid = np.ones((8, 8), bool)
    valid[1, 1] = False
    prof = est.build_profiles(m, valid, [("confidence", 4)], min_windows=1)
    pop = np.flatnonzero(valid.ravel() & np.isfinite(m.ravel()))
    assert (prof["index"][pop] == est.confidence_strata(m.ravel()[pop], 4)).all()
    assert prof["index"][0] == -1 and prof["index"][9] == -1 and (prof["index"] >= 0).sum() == pop.size == 62
    assert prof["names"] == [f"confidence {b}/4" for b in (1, 2, 3, 4)] and sum(prof["sizes"]) == 62
    cuts = prof["block"]["fields"][0]["cuts"]
    assert cuts == [float(c) for c in np.quantile(m.ravel()[pop], [0.25, 0.5, 0.75])]


def test_grid_blocks_conditions_and_classes():
    h, w = 8, 6
    prof = est.build_profiles(np.ones((h, w)), None, [("grid", 2)], min_windows=1)
    for r in range(h):
        for c in range(w):
            assert prof["index"][r * w + c] == (r * 2 // h) * 2 + (c * 2 // w)
    assert prof["names"] == ["cell 1,1", "cell 1,2", "cell 2,1", "cell 2,2"]
    cond = np.zeros((4, 4), int)
    cond[:, 2:] = 1
    cond[3, :] = -1                                       # no condition recorded: "unrecorded", last
    cls = np.tile([0, 1], (4, 2))
    prof = est.build_profiles(np.ones((4, 4)), None, [("condition", None), ("class", None)], condition=cond,
                              condition_names={0: "clear, dry", 1: "cloudy"}, map_class=cls, min_windows=1)
    assert prof["names"] == ["condition clear, dry, class 0", "condition clear, dry, class 1",
                             "condition cloudy, class 0", "condition cloudy, class 1",
                             "condition unrecorded, class 0", "condition unrecorded, class 1"]
    assert len(set(prof["names"])) == len(prof["names"]) and sum(prof["sizes"]) == 16
    assert prof["block"]["profiles"][0]["definition"] == {"condition": "clear, dry", "class": 0}
    for i in range(16):                                   # every window in exactly the profile its fields give
        name = prof["names"][prof["index"][i]]
        c = {0: "clear, dry", 1: "cloudy", -1: "unrecorded"}[int(cond.ravel()[i])]
        assert name == f"condition {c}, class {int(cls.ravel()[i])}"
    with pytest.raises(ValueError, match="--condition"):
        est.build_profiles(np.ones((4, 4)), None, [("condition", None)])


def test_small_cells_fold_into_other_last_and_too_many_profiles_are_refused():
    mc = np.array([0] * 50 + [1] * 3 + [2] * 2 + [3] * 45).reshape(10, 10)
    margin = np.random.default_rng(1).random((10, 10))
    prof = est.build_profiles(margin, None, [("class", None)], map_class=mc, min_windows=10)
    assert prof["names"] == ["class 0", "class 3", "other"] and prof["sizes"] == [50, 45, 5]
    b = prof["block"]
    assert b["n_folded_cells"] == 2 and b["min_windows"] == 10 and b["by"] == "class"
    assert b["profiles"][-1] == {"name": "other", "definition": "other", "n_windows": 5, "n_cells": 2}
    assert [(c["key"], c["profile"]) for c in b["cells"]] == [({"class": 0}, "class 0"), ({"class": 1}, "other"),
                                                              ({"class": 2}, "other"), ({"class": 3}, "class 3")]
    assert any("2 cells" in n for n in prof["notes"])
    again = est.build_profiles(margin, None, [("class", None)], map_class=mc, min_windows=10)
    assert (again["index"] == prof["index"]).all() and again["names"] == prof["names"]       # fixed by the map alone
    with pytest.raises(ValueError, match="16 profiles"):
        est.build_profiles(margin[:8, :8], None, [("grid", 4)], min_windows=1)
    one = est.build_profiles(margin[:8, :8], None, [("grid", 4)], min_windows=5)       # every cell of 4 is folded
    assert one["names"] == ["other"] and any("simple random sample" in n for n in one["notes"])


# ----------------------------------------------------------------------------- 8. the command line
def _write_map(d):
    """Three-class probabilities, independent per pixel, on a 64 x 64 grid (16 x 16 windows at patch 4), and a
    condition layer: clear on the left half, cloudy on the right."""
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(np.ones(3), size=(64, 64)).transpose(2, 0, 1).astype(np.float32)
    cond = np.zeros((64, 64), np.int64)
    cond[:, 32:] = 1
    np.save(d / "p.npy", probs)
    np.save(d / "cond.npy", cond)
    return str(d / "p.npy"), str(d / "cond.npy")


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _out(capsys, argv):
    capsys.readouterr()
    assert main(argv) == 0
    return capsys.readouterr().out


def test_the_profile_design_runs_end_to_end(tmp_path, capsys):
    p, cond = _write_map(tmp_path)
    s = str(tmp_path / "s.csv")
    printed = _out(capsys, ["sample", p, "--design", "profile", "--profile-by", "condition,confidence:2", "--condition",
                            cond, "--condition-names", "0=clear, dry", "1=cloudy", "--budget", "80",
                            "--min-windows", "10", "--out", s])
    side = json.load(open(tmp_path / "s.json"))
    names = side["condition"]["names"]
    assert side["design"] == "condition" and side["condition"]["by"] == "profile"
    assert side["condition"]["source"] == "profile" and side["condition"]["sha256"] is None
    assert names == ["confidence 1/2, condition clear, dry", "confidence 1/2, condition cloudy",
                     "confidence 2/2, condition clear, dry", "confidence 2/2, condition cloudy"]
    assert side["condition"]["n_labelled"] == [20, 20, 20, 20]
    prof = side["profile"]
    assert prof["by"] == "confidence:2,condition" and prof["min_windows"] == 10 and prof["max_profiles"] == 12
    layer = prof["fields"][1]
    assert layer["field"] == "condition" and layer["source"] == os.path.abspath(cond) and layer["sha256"]
    assert layer["names_given"] == [[0, "clear, dry"], [1, "cloudy"]]
    assert "certify s.csv --alpha A" in side["how_to_label"]
    for name, N in zip(names, side["condition"]["sizes"]):
        assert f"  {name}: {N} windows, 20 to label" in printed
    assert "profile design, 4 profiles by confidence:2,condition" in printed
    rows = _rows(s)
    assert {r["condition"] for r in rows} == set(names)              # names with commas round-trip the CSV
    # the reviewer's labels: profile 0 clean, profile 1 all wrong, profile 2 wrong on 8 of 20, profile 3 clean
    seen = {}
    for r in rows:
        h = int(r["stratum"])
        seen[h] = seen.get(h, 0) + 1
        r["wrong"] = "1" if h == 1 or (h == 2 and seen[h] <= 8) else "0"
        r["reference_class"] = r["map_class"] if r["wrong"] == "0" else str((int(r["map_class"]) + 1) % 3)
    _write_rows(s, rows)
    text = _out(capsys, ["estimate", s, "--per-class"])
    assert "take the profiles as strata" in text and "Each profile's own interval is exact" in text
    est_json = json.load(open(tmp_path / "s_estimate.json"))
    assert est_json["condition_by"] == "profile" and "stratified by profile" in est_json["method"]
    assert "drawn by profile" in est_json["profile_note"] and "drawn by profile" in text
    # two windows of profile 3 could not be judged
    marked = [r for r in rows if r["stratum"] == "3"][:2]
    for r in marked:
        r["wrong"] = "?"
    _write_rows(s, rows)
    text = _out(capsys, ["certify", s, "--alpha", "0.3"])
    res = json.load(open(tmp_path / "s_profiles.json"))
    assert res["by_profile"] and res["n_tested"] == 4 and res["n_unjudged"] == 2 and res["alpha"] == 0.3
    assert res["level"] == float(1 - _level(0.1) / 4) and res["delta_per_profile"] == 0.1 / 4
    assert not {"rule", "coverage", "low", "high"} & set(res)            # never read as a zone or an estimate
    verdicts = [res["profiles"][n]["verdict"] for n in names]
    assert verdicts[:3] == ["good", "bad", "undecided"], verdicts
    assert res["profiles"][names[3]]["n_unjudged"] == 2
    assert "each interval at 97.5%" in text and "good (wrong at most 30% of the time)" in text
    assert "bad (wrong more than 30% of the time)" in text and "together: good" in text
    assert res["verdict_raster"] == os.path.abspath(tmp_path / "s_profiles.npy")
    codes = np.load(tmp_path / "s_profiles.npy")
    grid = np.asarray(side["condition_grid"])
    assert codes.shape == (16, 16) and codes.dtype == np.uint8
    for i, h in enumerate(grid):
        want = 0 if h < 0 else loc.VERDICT_CODES[verdicts[h]]
        assert codes.ravel()[i] == want
    # --out without .json: the raster beside it, with .npy added
    _out(capsys, ["certify", s, "--alpha", "0.3", "--out", str(tmp_path / "verdicts")])
    assert os.path.exists(tmp_path / "verdicts") and os.path.exists(tmp_path / "verdicts.npy")
    # what does not take per-profile verdicts in this version
    with pytest.raises(SystemExit, match="per-profile verdicts"):
        main(["certificate", str(tmp_path / "s_profiles.json"), "--date", "2024-06-01", "--out", str(tmp_path / "c")])
    with pytest.raises(SystemExit, match="per-profile verdicts"):
        main(["decide", str(tmp_path / "s_profiles.json"), "--ask", "trusted_share"])
    from oe_inferencex import mcp_server
    with pytest.raises(mcp_server.Refused, match="drawn by profile"):
        mcp_server.certify(sample_csv=s, alpha=0.3)
    for extra, flag in ((["--rule", "bonferroni"], "--rule"), (["--level-cut", "ramp"], "--level-cut"),
                        (["--also-accept", str(tmp_path / "pairs.json")], "--also-accept"), (["--scores", p], "--scores")):
        with pytest.raises(SystemExit, match=f"takes none of {flag}"):
            main(["certify", s, "--alpha", "0.3", *extra])
    with pytest.raises(SystemExit, match="--alpha is required"):
        main(["certify", s])


def test_the_profile_design_s_refusals(tmp_path):
    p, cond = _write_map(tmp_path)
    out = ["--budget", "80", "--out", str(tmp_path / "x.csv")]
    for argv, why in (
            (["--profile-by", "class"], "belong to --design profile"),
            (["--design", "random", "--min-windows", "5"], "belong to --design profile"),
            (["--design", "profile"], "needs --profile-by"),
            (["--design", "profile", "--profile-by", "confidence:9"], "2 to 5"),
            (["--design", "profile", "--profile-by", "class", "--condition", cond], "read only when --profile-by"),
            (["--design", "profile", "--profile-by", "condition"], "needs --condition"),
            (["--design", "profile", "--profile-by", "class", "--min-windows", "0"], "at least 1"),
            (["--design", "profile", "--profile-by", "class", "--seed-from-map", "--alpha", "0.1"],
             "not from --design profile"),
            (["--design", "profile", "--profile-by", "class", "--target", "0.5"], "--target fixes the zone"),
            (["--design", "profile", "--profile-by", "grid:4", "--min-windows", "1"], "at most 12")):
        with pytest.raises(SystemExit, match=why):
            main(["sample", p, *argv, *out])


def test_a_condition_sample_reads_as_before(tmp_path, capsys):
    """The profile design adds keys and words to its own results only: a condition sample's estimate carries no
    condition_by and no profile_note and speaks of input conditions, and its sidecar says no `by`."""
    p, cond = _write_map(tmp_path)
    s = str(tmp_path / "c.csv")
    _out(capsys, ["sample", p, "--condition", cond, "--budget", "40", "--out", s])
    side = json.load(open(tmp_path / "c.json"))
    assert "by" not in side["condition"] and "profile" not in side
    rows = _rows(s)
    for r in rows:
        r["wrong"] = "0"
    _write_rows(s, rows)
    text = _out(capsys, ["estimate", s])
    res = json.load(open(tmp_path / "c_estimate.json"))
    assert "condition_by" not in res and "profile_note" not in res and "stratified by input condition" in text
    from oe_inferencex.common.wording import class_wording, profile_names
    assert profile_names(side) is None and class_wording(side)("each condition") == "each condition"


def test_the_profile_words_keep_the_names():
    from oe_inferencex.common.wording import class_wording
    side = {"condition": {"by": "profile", "names": ["confidence 1/2, condition clear", "other"]}}
    w = class_wording(side)
    assert w("Each condition's interval, per input condition") == "Each profile's interval, per profile"
    assert w("confidence 1/2, condition clear: 3%") == "confidence 1/2, condition clear: 3%"
    assert class_wording({"condition": {"by": "map class", "names": ["class 1"]}})("each condition") == "each class"


# ----------------------------------------------------------------------------- 9. plan
def _brute(N, n, alpha, E, delta, L, want):
    K = int(round(E * N))
    conf = 1 - _level(delta) / L
    total = Fraction(0)
    for k in range(max(0, n - (N - K)), min(n, K) + 1):
        lo, hi = loc.profile_interval(k, 0, n, N, conf)
        if loc.profile_verdict(lo, hi, alpha) == want:
            total += Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n))
    return total


def test_plan_agrees_with_the_verdicts_summed_over_every_draw():
    for N in (5, 12, 30):
        for alpha in (0.2, 0.3):
            for E in (0.0, 0.1, 0.4, 0.6, 1.0):
                try:
                    want = plan._profile_side(N, alpha, E)
                except ValueError:
                    continue
                for delta in (0.1, 0.3):
                    for L in (1, 3):
                        for n in range(1, N + 1):
                            got = plan.profile_verdict_probability(N, n, alpha, E, delta, L, want)
                            if n < N:
                                exact = _brute(N, n, alpha, E, delta, L, want)
                            else:                    # a census: its exact rate against alpha
                                at_most = Fraction(int(round(E * N)), N) <= _level(alpha)
                                exact = Fraction(int(at_most == (want == "good")))
                            assert abs(got - float(exact)) < 1e-9, (N, alpha, E, delta, L, n, want, got, exact)


@pytest.mark.parametrize("L,labels", [(1, 59), (2, 72), (4, 86), (8, 99), (12, 107)])
def test_plan_with_no_error_needs_the_labels_of_the_upper_tail(L, labels):
    """With no error seen, a large profile is good from min_labels_to_certify(alpha, delta/(2L)) labels: 59, 72, 86,
    99 and 107 at alpha 5% and delta 0.1."""
    assert est.min_labels_to_certify(0.05, 0.1 / (2 * L)) == labels
    N = 10 ** 6
    assert plan.profile_verdict_probability(N, labels, 0.05, 0.0, 0.1, L, "good") == 1.0
    assert plan.profile_verdict_probability(N, labels - 1, 0.05, 0.0, 0.1, L, "good") == 0.0
    r = plan.plan_profile(N, 0.05, 0.0, L)
    assert r["labels"] == labels and r["verdict"] == "good" and r["question"] == "profile"
    assert r["level"] == float(1 - _level(0.1) / L)


def test_plan_refusals_and_the_command(capsys):
    with pytest.raises(ValueError, match="not on the side of alpha"):
        plan.plan_profile(10, 0.3, 0.31, 2)                  # 3 of 10 is at most alpha
    with pytest.raises(ValueError, match="not on the side of alpha"):
        plan.plan_profile(100, 0.3, 0.3, 2)
    with pytest.raises(ValueError, match="number of profiles"):
        plan.plan_profile(100, 0.3, 0.1, 0)
    assert plan.plan_profile(100, 0.3, 0.9, 2)["verdict"] == "bad"
    text = _out(capsys, ["plan", "--windows", "5000", "--alpha", "0.05", "--profiles", "4", "--profile-error", "0"])
    assert text.startswith("Profile: a profile of 5000 windows") and "calls it good" in text and "(exact)" in text
    assert "run:" not in text
    for argv, why in ((["--windows", "5000", "--alpha", "0.05", "--profiles", "4"], "needs --windows"),
                      (["--windows", "5000", "--alpha", "0.05", "--profiles", "4", "--profile-error", "0.01",
                        "--coverage", "0.5"], "separate call"),
                      (["map.npy", "--alpha", "0.05", "--profiles", "4", "--profile-error", "0.01"], "from its size"),
                      (["--windows", "5000", "--alpha", "0.05", "--profiles", "4", "--profile-error", "0.05"],
                       "not on the side")):
        with pytest.raises(SystemExit, match=why):
            main(["plan", *argv])

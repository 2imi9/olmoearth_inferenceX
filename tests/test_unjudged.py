"""Windows the reviewer could not judge (`?`) and a reviewer who errs: the bounds hold by exact enumeration.

The estimate's bounds rest on one property: the exact intervals move one way with the error count, so an interval
whose lower end counts every unjudged window as right and whose upper end counts each as wrong holds the interval
the full labels would give, whatever made the windows hard to judge. These tests check that property and the
coverage it buys exactly (hypergeometric probabilities, no Monte Carlo), with an adversary that chooses which
sampled windows go unjudged after seeing their truth, and a reviewer whose errors are placed to do the most harm.
"""
import csv
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex.cli import main


def _pmf(k, n, N, K):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def _random_sample(N, n):
    return {"design": "random", "indices": np.arange(n), "n_population": N}


def _labels(n, k, u_wrong, u_right):
    """n labels with k truly wrong, of which u_wrong of the wrong and u_right of the right ones are unjudged."""
    wrong = np.r_[np.ones(k), np.zeros(n - k)]
    unjudged = np.zeros(n, bool)
    unjudged[:u_wrong] = True
    unjudged[k:k + u_right] = True
    return wrong, unjudged


@pytest.mark.parametrize("N,n", [(12, 4), (12, 8), (20, 5), (20, 10)])
def test_unjudged_windows_keep_the_random_designs_coverage_against_any_adversary(N, n):
    """For every population count K, the coverage when an adversary picks, after seeing the truth of the sample, how
    many wrong and how many right windows go unjudged to make the interval miss, is at least the full labels' own
    coverage, so at least 95%."""
    sample = _random_sample(N, n)
    for K in range(N + 1):
        theta = K / N
        worst, full = 0.0, 0.0
        for k in range(max(0, n - (N - K)), min(n, K) + 1):
            p = _pmf(k, n, N, K)
            lo, hi = est.hypergeom_interval(k, n, N)
            full += p * (lo <= theta <= hi)
            covered = True
            for uw in range(k + 1):
                for ur in range(n - k + 1):
                    if uw == ur == 0:
                        continue
                    wrong, unj = _labels(n, k, uw, ur)
                    r = est.estimate_error_rate(sample, wrong, unjudged=unj)
                    assert r["n_unjudged"] == uw + ur and r["estimate"] is None
                    assert r["low"] <= lo and r["high"] >= hi          # it holds the full labels' interval
                    covered &= r["low"] <= theta <= r["high"]
            worst += p * covered
        assert worst >= full - 1e-12 and worst >= 0.95, (K, worst, full)


def test_the_union_bound_over_strata_is_monotone_in_each_strata_count():
    """`_stratum_union` and the condition design's interval add each stratum's exact interval; each end moves one
    way with each stratum's error count, which is what lets the worst case bound windows nobody could judge."""
    sizes, n = [9, 14, 6], [4, 5, 6]
    prev = {}
    for k in np.ndindex(*(m + 1 for m in n)):
        _, lo, hi, _ = est._union_interval(list(k), n, sizes)
        prev[k] = (lo, hi)
    for k, (lo, hi) in prev.items():
        for c in range(3):
            if k[c] < n[c]:
                up = tuple(v + (i == c) for i, v in enumerate(k))
                assert prev[up][0] >= lo and prev[up][1] >= hi, (k, c)


def test_the_confidence_design_with_unjudged_windows_uses_the_union_bound_and_covers():
    """Under the confidence design the Wilson interval on an effective sample size is not monotone, so with unjudged
    windows the interval is the union bound over the strata, whose coverage is at least 95% for every population: here
    by exact enumeration of two strata, with the adversary of the random-design test in each stratum."""
    rs = np.random.RandomState(0)
    margin = rs.rand(30)
    sample = est.sample_for_estimation(margin, 12, design="proportional", n_strata=2, seed=1)
    strata, pop, sizes = np.asarray(sample["strata"]), np.asarray(sample["strata_of_population"]), sample["sizes"]
    pos = {int(g): i for i, g in enumerate(pop)}
    local = np.array([pos[int(g)] for g in sample["indices"]])
    h = strata[local]
    n_h = [int((h == j).sum()) for j in range(2)]
    N_h = [int(v) for v in sizes]
    wrong = np.zeros(len(local))
    unj = np.zeros(len(local), bool)
    unj[np.flatnonzero(h == 0)[:1]] = True
    r = est.estimate_error_rate(sample, wrong, unjudged=unj)
    assert r["method"].startswith("stratified by confidence margin; with windows that could not be judged")
    assert "effective_n" not in r and r["strata_in_interval"] == 2
    # exact coverage of the union-bound interval the bounds are built from, for every pair of stratum counts
    for K0 in range(N_h[0] + 1):
        for K1 in range(N_h[1] + 1):
            theta = Fraction(K0 + K1, sum(N_h))
            cov = 0.0
            for k0 in range(max(0, n_h[0] - (N_h[0] - K0)), min(n_h[0], K0) + 1):
                for k1 in range(max(0, n_h[1] - (N_h[1] - K1)), min(n_h[1], K1) + 1):
                    _, lo, hi, _ = est._union_interval([k0, k1], n_h, N_h)
                    cov += _pmf(k0, n_h[0], N_h[0], K0) * _pmf(k1, n_h[1], N_h[1], K1) * (lo <= float(theta) <= hi)
            assert cov >= 0.95 - 1e-12, (K0, K1, cov)


@pytest.mark.parametrize("E0,E1", [(0.1, 0.1), (0.25, 0.0), (0.0, 0.25), (0.2, 0.3)])
@pytest.mark.parametrize("N,n", [(20, 6), (20, 12)])
def test_reviewer_error_bounds_cover_the_true_rate_whatever_the_errors(E0, E1, N, n):
    """The reviewer marks f0 truly correct windows wrong and misses f1 truly wrong ones, any f0 up to E0 of the
    correct windows and f1 up to E1 of the wrong ones. The sample sees the reviewer's labels. For every true count
    and every such reviewer, the widened interval covers the TRUE rate on at least 95% of samples."""
    sample = _random_sample(N, n)
    for K in range(N + 1):
        theta = K / N
        for f0 in range(int(math.floor(E0 * (N - K) + 1e-9)) + 1):
            for f1 in range(int(math.floor(E1 * K + 1e-9)) + 1):
                K_obs = K - f1 + f0
                cov = 0.0
                for k in range(max(0, n - (N - K_obs)), min(n, K_obs) + 1):
                    wrong = np.r_[np.ones(k), np.zeros(n - k)]
                    r = est.estimate_error_rate(sample, wrong, reviewer_false_alarm=E0, reviewer_miss=E1)
                    cov += _pmf(k, n, N, K_obs) * (r["low"] - 1e-12 <= theta <= r["high"] + 1e-12)
                assert cov >= 0.95 - 1e-12, (K, f0, f1, cov)


def test_the_reviewer_bounds_are_the_identified_set_at_a_census():
    """With every window labelled the labels' interval is the point p, and the bounds are the sharp identified set
    [(p - E0) / (1 - E0), p / (1 - E1)], clipped to [0, 1]."""
    N = 40
    sample = _random_sample(N, N)
    for k in (0, 3, 10, 30, 40):
        wrong = np.r_[np.ones(k), np.zeros(N - k)]
        r = est.estimate_error_rate(sample, wrong, reviewer_false_alarm=0.1, reviewer_miss=0.2)
        p = k / N
        assert r["low"] == pytest.approx(max(0.0, (p - 0.1) / 0.9)) and r["high"] == pytest.approx(min(1.0, p / 0.8))
        assert r["estimate_range"] == pytest.approx([r["low"], r["high"]])
        assert r["labels_interval"] == {"low": p, "high": p}


def test_unjudged_and_reviewer_inputs_are_checked():
    sample = _random_sample(10, 5)
    with pytest.raises(ValueError, match="reviewer_miss must be at least 0 and below 1"):
        est.estimate_error_rate(sample, np.zeros(5), reviewer_miss=1.0)
    with pytest.raises(ValueError, match="unjudged flags for 5 labels"):
        est.estimate_error_rate(sample, np.zeros(5), unjudged=np.zeros(4, bool))
    # nothing unjudged and no reviewer error: the core result, unchanged
    assert est.estimate_error_rate(sample, np.zeros(5), unjudged=np.zeros(5, bool)) == est._estimate_core(sample,
                                                                                                         np.zeros(5))


# ----------------------------------------------------------------------------- through the command line
def _write_map(tmp_path):
    rs = np.random.RandomState(4)
    probs = rs.dirichlet(np.ones(3), size=(32, 32)).transpose(2, 0, 1).astype(np.float32)
    np.save(tmp_path / "p.npy", probs)
    return tmp_path / "p.npy"


def _fill(path, values):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row, v in zip(rows, values):
        row["wrong"] = v
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_a_csv_with_question_marks_is_estimated_as_a_range_and_certified_counting_them_wrong(tmp_path, capsys):
    scores = _write_map(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", str(scores), "--budget", "60", "--design", "random", "--patch", "1", "--out", str(out)]) == 0
    values = ["1" if i % 10 == 0 else "?" if i % 10 == 1 else "0" for i in range(60)]
    _fill(out, values)
    capsys.readouterr()
    assert main(["estimate", str(out)]) == 0
    printed = capsys.readouterr().out
    r = json.load(open(tmp_path / "s_estimate.json"))
    assert r["estimate"] is None and r["estimate_range"] == pytest.approx([6 / 60, 12 / 60]) and r["n_unjudged"] == 6
    lo6, _ = est.hypergeom_interval(6, 60, 1024)
    _, hi12 = est.hypergeom_interval(12, 60, 1024)
    assert (r["low"], r["high"]) == (lo6, hi12)
    assert printed.startswith("error rate 10.0% to 20.0%, 95% interval") and "could not be judged" in printed
    assert est.UNJUDGED_STRATIFIED_NOTE not in printed

    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.3"]) == 0
    printed = capsys.readouterr().out
    z = json.load(open(tmp_path / "s_zone.json"))
    assert z["alpha"] == 0.3 and z["n_unjudged"] == 6 and "counted as wrong" in printed
    # counting the six as wrong is the same certificate as labelling them wrong
    _fill(out, ["1" if v == "?" else v for v in values])
    assert main(["certify", str(out), "--alpha", "0.3", "--out", str(tmp_path / "w.json")]) == 0
    w = json.load(open(tmp_path / "w.json"))
    assert (w["coverage"], w["n_zone"], w["levels"]) == (z["coverage"], z["n_zone"], z["levels"])
    # certify takes no reviewer error rate: its premise would have to hold inside every zone it can certify
    with pytest.raises(SystemExit):
        main(["certify", str(out), "--alpha", "0.3", "--reviewer-miss", "0.1"])


def test_the_confidence_design_with_unjudged_windows_returns_the_union_bound_at_the_two_extremes():
    """The values estimate_error_rate returns under a stratified design: the low end is the union bound with every ?
    counted right, the high end with every ? counted wrong, and the range the two stratified estimates; the note says
    the switch from Wilson costs most of the width."""
    rs = np.random.RandomState(5)
    margin = rs.rand(400)
    sample = est.sample_for_estimation(margin, 80, design="proportional", n_strata=4, seed=2)
    wrong = (rs.rand(80) < 0.2).astype(float)
    unj = np.zeros(80, bool)
    unj[[3, 17, 40, 63]] = True
    r = est.estimate_error_rate(sample, wrong, unjudged=unj)
    e_lo, lo, _, L = est._stratum_union(sample, np.where(unj, 0.0, wrong))
    e_hi, _, hi, _ = est._stratum_union(sample, np.where(unj, 1.0, wrong))
    assert (r["low"], r["high"]) == (lo, hi) and r["estimate_range"] == [e_lo, e_hi] and r["strata_in_interval"] == L
    assert est.UNJUDGED_STRATIFIED_NOTE in r["bounds_note"]
    # the same counts under a random design keep each window's own range, without the switch
    rnd = dict(sample, design="random")
    rr = est.estimate_error_rate(rnd, wrong, unjudged=unj)
    assert est.UNJUDGED_STRATIFIED_NOTE not in rr["bounds_note"]


def test_each_condition_is_bounded_with_its_own_question_marks_and_the_reviewer_rates():
    """A condition sample with ? in one condition and stated reviewer rates: each row is the exact interval at its own
    k (? right) and k + u (? wrong), mapped by the reviewer bounds; a row with no ? and no reviewer error keeps its
    value; the outside list compares intervals with the whole map's range."""
    rs = np.random.RandomState(6)
    n_win = 600
    margin = rs.rand(n_win)
    cond = (np.arange(n_win) >= 400).astype(int)
    sample = est.sample_for_estimation(margin, 120, design="condition", condition=cond, condition_names={0: "clear", 1: "cloudy"},
                                       seed=3)
    idx = np.asarray(sample["indices"])
    in_cloudy = cond[idx] == 1
    wrong = np.where(in_cloudy, rs.rand(idx.size) < 0.4, rs.rand(idx.size) < 0.1).astype(float)
    unj = in_cloudy & (np.arange(idx.size) % 5 == 0)
    plain = est.estimate_error_rate(sample, wrong, unjudged=unj)
    for name, c in (("clear", 0), ("cloudy", 1)):
        m = cond[idx] == c
        k, u, n = int(wrong[m & ~unj].sum()), int((m & unj).sum()), int(m.sum())
        N = int((cond == c).sum())
        row = plain["per_condition"][name]
        assert row["n_unjudged"] == u
        assert row["low"] == est.hypergeom_interval(k, n, N)[0] and row["high"] == est.hypergeom_interval(k + u, n, N)[1]
        if u == 0:
            assert row["estimate"] == k / n                       # nothing to bound: the value, not a range
    E0, E1 = 0.05, 0.15
    rev = est.estimate_error_rate(sample, wrong, unjudged=unj, reviewer_false_alarm=E0, reviewer_miss=E1)
    for name in ("clear", "cloudy"):
        a, b = plain["per_condition"][name], rev["per_condition"][name]
        assert b["low"] == pytest.approx(max(0.0, (a["low"] - E0) / (1 - E0)))
        assert b["high"] == pytest.approx(min(1.0, a["high"] / (1 - E1)))
        assert b["estimate"] is None
    lo_w, hi_w = rev["estimate_range"]
    want = [n for n, row in rev["per_condition"].items() if row["low"] > hi_w or row["high"] < lo_w]
    assert rev["outside_condition_intervals"] == want


def test_the_reviewer_rates_through_the_command_line(tmp_path, capsys):
    """The false-alarm bound moves the lower end exactly; a range with no ? and no error to bound is one value; the
    refusals name the command line's options."""
    scores = _write_map(tmp_path)
    out = tmp_path / "f.csv"
    assert main(["sample", str(scores), "--budget", "60", "--design", "random", "--patch", "1", "--out", str(out)]) == 0
    _fill(out, ["1" if i % 6 == 0 else "0" for i in range(60)])
    assert main(["estimate", str(out), "--out", str(tmp_path / "plain.json")]) == 0
    plain = json.load(open(tmp_path / "plain.json"))
    assert main(["estimate", str(out), "--reviewer-false-alarm", "0.05", "--out", str(tmp_path / "fa.json")]) == 0
    fa = json.load(open(tmp_path / "fa.json"))
    assert fa["low"] == pytest.approx((plain["low"] - 0.05) / 0.95) and fa["high"] == plain["high"]
    assert fa["labels_interval"] == {"low": plain["low"], "high": plain["high"]}
    _fill(out, ["0"] * 60)
    capsys.readouterr()
    assert main(["estimate", str(out), "--reviewer-false-alarm", "0.05", "--out", str(tmp_path / "z.json")]) == 0
    assert json.load(open(tmp_path / "z.json"))["estimate"] == 0.0
    assert capsys.readouterr().out.startswith("error rate 0.0%, 95% interval")
    with pytest.raises(SystemExit, match="--reviewer-miss must be at least 0 and below 1"):
        main(["estimate", str(out), "--reviewer-miss", "1"])
    rows = _fill(out, ["0"] * 60)
    for row in rows:
        row["reference_class"] = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="do not apply to the per-class table"):
        main(["estimate", str(out), "--per-class", "--reviewer-miss", "0.1"])


def test_question_marks_in_a_condition_sample_are_certified_as_wrong(tmp_path, capsys):
    scores = _write_map(tmp_path)
    layer = np.zeros((32, 32), np.int16)
    layer[:, 16:] = 1
    np.save(tmp_path / "cond.npy", layer)
    out = tmp_path / "c.csv"
    assert main(["sample", str(scores), "--budget", "240", "--patch", "1", "--condition", str(tmp_path / "cond.npy"),
                 "--out", str(out)]) == 0
    values = ["?" if i % 12 == 0 else "0" for i in range(240)]
    _fill(out, values)
    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.1"]) == 0
    printed = capsys.readouterr().out
    z = json.load(open(tmp_path / "c_zone.json"))
    assert z["n_unjudged"] == 20 and "counted as wrong" in z["bounds_note"] and "note: 20 window(s)" in printed
    _fill(out, ["1" if v == "?" else v for v in values])
    assert main(["certify", str(out), "--alpha", "0.1", "--out", str(tmp_path / "w.json")]) == 0
    w = json.load(open(tmp_path / "w.json"))
    assert w["per_condition"] == z["per_condition"] and w["certified_share_of_map"] == z["certified_share_of_map"]


def test_question_marks_are_refused_where_no_bound_exists(tmp_path):
    scores = _write_map(tmp_path)
    out = tmp_path / "t.csv"
    assert main(["sample", str(scores), "--budget", "64", "--design", "tiles", "--patch", "1", "--tile", "4",
                 "--per-tile", "8", "--out", str(out)]) == 0
    _fill(out, ["?"] + ["0"] * 63)
    with pytest.raises(SystemExit, match="not supported under the tiles design"):
        main(["estimate", str(out)])
    r = tmp_path / "r.csv"
    assert main(["sample", str(scores), "--budget", "30", "--design", "random", "--patch", "1", "--out", str(r)]) == 0
    rows = _fill(r, ["?"] + ["0"] * 29)
    for row in rows:
        row["reference_class"] = 0
    with open(r, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match=r"--per-class needs the class seen in every window.*1 window\(s\) are \?"):
        main(["estimate", str(r), "--per-class"])
    _fill(r, [""] + ["0"] * 29)
    with pytest.raises(SystemExit, match=r"or a \? where it cannot be judged"):
        main(["estimate", str(r)])

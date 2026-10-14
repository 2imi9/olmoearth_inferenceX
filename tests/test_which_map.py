"""Which of two maps is more accurate, from labels on the windows where they differ (sample --other, estimate).

The interval on the accuracy difference adds each map's exact interval over the differing windows at 97.5%, so it
covers at least 95% by the union bound; ? rows widen each end the way that can only hold more. These tests check the
coverage exactly (multivariate hypergeometric probabilities, no Monte Carlo), with an adversary that marks ? after
seeing each window's truth, and run the route through the command line and the MCP server.
"""
import csv
import importlib.util
import json
import math

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex.cli import main


def _sample(N, D, n):
    return {"design": "disagreement", "indices": np.arange(n), "n_population": N, "n_disagree": D}


def _labels(a, b, c, ua=0, ub=0, uc=0):
    """n = a + b + c labelled differing windows: a right in map a (class 0), b right in map b (class 1), c in neither
    (class 2); ua, ub and uc of them marked ?."""
    n = a + b + c
    ref = np.r_[np.zeros(a), np.ones(b), np.full(c, 2)].astype(int)
    unj = np.zeros(n, bool)
    unj[:ua] = True
    unj[a:a + ub] = True
    unj[a + b:a + b + uc] = True
    return np.zeros(n, int), np.ones(n, int), ref, unj


def _pmf3(a, b, n, D, KA, KB):
    K0 = D - KA - KB
    c = n - a - b
    if min(a, b, c) < 0 or a > KA or b > KB or c > K0:
        return 0.0
    return math.comb(KA, a) * math.comb(KB, b) * math.comb(K0, c) / math.comb(D, n)


@pytest.mark.parametrize("N,D,n", [(10, 10, 4), (20, 10, 6), (24, 12, 5), (30, 9, 9)])
def test_the_difference_interval_covers_against_any_marking_of_question_marks(N, D, n):
    """For every split (K_a, K_b) of the D differing windows, the coverage of the true difference (K_a - K_b) / N is at
    least 95% when an adversary, after seeing each sampled window's truth, marks any number of them ? to make the
    interval miss."""
    for KA in range(D + 1):
        for KB in range(D - KA + 1):
            delta = (KA - KB) / N
            worst = 0.0
            for a in range(n + 1):
                for b in range(n - a + 1):
                    p = _pmf3(a, b, n, D, KA, KB)
                    if p == 0:
                        continue
                    c = n - a - b
                    ok = True
                    for ua in range(a + 1):
                        for ub in range(b + 1):
                            for uc in (0, c):
                                r = est.compare_from_disagreement(_sample(N, D, n), *_labels(a, b, c, ua, ub, uc))
                                d = r["difference"]
                                ok &= d["low"] - 1e-12 <= delta <= d["high"] + 1e-12
                    worst += p * ok
            assert worst >= 0.95 - 1e-12, (KA, KB, worst)


def test_a_census_of_the_differing_windows_gives_the_exact_difference():
    r = est.compare_from_disagreement(_sample(50, 10, 10), *_labels(6, 3, 1))
    assert r["difference"]["low"] == r["difference"]["high"] == r["difference"]["estimate"] == pytest.approx(3 / 50)
    assert r["verdict"] == "a"
    r = est.compare_from_disagreement(_sample(50, 10, 10), *_labels(2, 7, 1))
    assert r["verdict"] == "b" and r["difference"]["estimate"] == pytest.approx(-5 / 50)


def test_question_marks_make_the_estimate_a_range_and_never_narrow_the_interval():
    base = est.compare_from_disagreement(_sample(200, 60, 20), *_labels(12, 6, 2))
    q = est.compare_from_disagreement(_sample(200, 60, 20), *_labels(12, 6, 2, ua=2, ub=1))
    assert q["n_unjudged"] == 3 and q["difference"]["estimate"] is None
    lo, hi = q["difference"]["estimate_range"]
    assert lo == pytest.approx((10 - 8) / 20 * 60 / 200) and hi == pytest.approx((13 - 5) / 20 * 60 / 200)
    assert q["difference"]["low"] <= base["difference"]["low"] and q["difference"]["high"] >= base["difference"]["high"]


def test_the_sampler_draws_only_differing_windows_and_refuses_identical_maps():
    a = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    b = np.array([0, 1, 1, 0, 2, 0, 0, 1])
    valid = np.array([1, 1, 1, 1, 1, 1, 0, 1], bool)
    s = est.sample_disagreement(a, b, 10, valid=valid, seed=3)
    assert set(s["indices"].tolist()) == {1, 3, 5} and (s["n_population"], s["n_disagree"]) == (7, 3)
    assert est.sample_disagreement(a, b, 2, valid=valid, seed=3)["indices"].tolist() == \
        est.sample_disagreement(a, b, 2, valid=valid, seed=3)["indices"].tolist()
    with pytest.raises(ValueError, match="same class in every window"):
        est.sample_disagreement(a, a, 5)
    with pytest.raises(ValueError, match="same class in both maps"):
        est.compare_from_disagreement(_sample(10, 4, 2), np.array([0, 1]), np.array([0, 2]), np.array([0, 1]))


# ----------------------------------------------------------------------------- the command line and the server
def _maps(tmp_path):
    """Two 4-class probability maps of one 1 x 300 strip and the truth: a right on 270 windows, b on 225."""
    rs = np.random.RandomState(7)
    n = 300
    truth = rs.randint(0, 4, size=n)
    pred_a = np.where(rs.rand(n) < 0.9, truth, (truth + 1) % 4)
    pred_b = np.where(rs.rand(n) < 0.75, truth, (truth + 2) % 4)
    for name, pred in (("a.npy", pred_a), ("b.npy", pred_b)):
        p = np.full((4, 1, n), 0.1, np.float32)
        p[pred, 0, np.arange(n)] = 0.7
        np.save(tmp_path / name, p)
    np.save(tmp_path / "truth.npy", truth[None, :])
    return truth, pred_a, pred_b


def _label_from_truth(path, truth, unjudged_rows=()):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["reference_class"] = "?" if i in unjudged_rows else int(truth[int(r["pixel_col"])])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_sample_other_then_estimate_names_the_more_accurate_map(tmp_path, capsys):
    truth, pa, pb = _maps(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "b.npy"), "--budget", "60", "--patch",
                 "1", "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "s.json"))
    D = int((pa != pb).sum())
    assert side["design"] == "disagreement" and side["n_disagree"] == D and side["n_population"] == 300
    assert all(pa[i] != pb[i] for i in side["indices"])
    assert side["class_a"] == [int(pa[i]) for i in side["indices"]]
    _label_from_truth(out, truth, unjudged_rows=(0, 1))
    capsys.readouterr()
    assert main(["estimate", str(out)]) == 0
    printed = capsys.readouterr().out
    r = json.load(open(tmp_path / "s_estimate.json"))
    true_delta = float((pa == truth).mean() - (pb == truth).mean())
    assert r["verdict"] == "a" and printed.startswith("map a is more accurate than map b, by")
    assert r["difference"]["low"] <= true_delta <= r["difference"]["high"]
    assert r["n_unjudged"] == 2 and "2 could not be judged" in printed
    with pytest.raises(SystemExit, match="certify needs a random sample of one map's windows"):
        main(["certify", str(out), "--alpha", "0.05"])
    with pytest.raises(SystemExit, match="takes none of --per-class"):
        main(["estimate", str(out), "--per-class"])


def test_the_comparison_sample_is_checked(tmp_path):
    truth, pa, pb = _maps(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "b.npy"), "--budget", "20", "--patch",
                 "1", "--out", str(out)]) == 0
    rows = _label_from_truth(out, truth)
    rows[0]["class_a"] = str(int(rows[0]["class_b"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="class_a or class_b differs"):
        main(["estimate", str(out)])
    assert main(["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "b.npy"), "--budget", "20", "--patch",
                 "1", "--out", str(out)]) == 0
    _label_from_truth(out, truth)
    rows = list(csv.DictReader(open(out, newline="")))
    rows[3]["reference_class"] = ""
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match=r"the class seen .* or \? in every row"):
        main(["estimate", str(out)])
    with pytest.raises(SystemExit, match="drop --design"):
        main(["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "b.npy"), "--budget", "20", "--patch", "1",
              "--design", "random", "--out", str(out)])
    with pytest.raises(SystemExit, match="used only with --other"):
        main(["sample", str(tmp_path / "a.npy"), "--budget", "20", "--patch", "1", "--threshold", "0.5",
              "--out", str(out)])
    with pytest.raises(SystemExit, match="same class in every window"):
        main(["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "a.npy"), "--budget", "20", "--patch", "1",
              "--out", str(out)])


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed")
def test_the_route_through_the_server(tmp_path):
    from oe_inferencex import mcp_server
    truth, pa, pb = _maps(tmp_path)
    s = mcp_server.sample(scores=str(tmp_path / "a.npy"), out_dir=str(tmp_path / "m"), budget=60, patch=1,
                          other=str(tmp_path / "b.npy"))
    assert s["summary"]["design"] == "disagreement" and "reference_class" in s["next"]
    assert "No window is labelled yet" in s["conclusion"] and "certify refuses it" in s["limits"]
    _label_from_truth(s["files"]["sample_csv"], truth)
    r = mcp_server.estimate(sample_csv=s["files"]["sample_csv"])
    assert r["summary"]["verdict"] == "a" and r["conclusion"].startswith("Map a is more accurate than map b, by")
    assert "not either map's accuracy" in r["limits"]
    c = mcp_server.compare(a=str(tmp_path / "a.npy"), b=str(tmp_path / "b.npy"), out_dir=str(tmp_path / "c"), patch=1)
    assert "other set to map b" in c["next"]


# ----------------------------------------------------------------------------- the review of 3 October 2026
def test_each_maps_share_is_its_exact_interval_at_97_5_percent():
    """The documented construction, pinned directly: each map's share of the differing windows is the exact interval
    at 1 - 0.05 / 2 (the union bound's level), and the difference's ends are built from those shares."""
    s = _sample(400, 120, 30)
    r = est.compare_from_disagreement(s, *_labels(15, 9, 6))
    assert (r["share_a_right"]["low"], r["share_a_right"]["high"]) == est.hypergeom_interval(15, 30, 120, conf=0.975)
    assert (r["share_b_right"]["low"], r["share_b_right"]["high"]) == est.hypergeom_interval(9, 30, 120, conf=0.975)
    w = 120 / 400
    assert r["difference"]["low"] == pytest.approx((r["share_a_right"]["low"] - r["share_b_right"]["high"]) * w)
    assert r["difference"]["high"] == pytest.approx((r["share_a_right"]["high"] - r["share_b_right"]["low"]) * w)


def test_the_estimator_checks_its_inputs():
    with pytest.raises(ValueError, match="budget must be at least 1"):
        est.sample_disagreement(np.array([0, 1]), np.array([1, 0]), 0)
    with pytest.raises(ValueError, match="share one window grid"):
        est.sample_disagreement(np.array([0, 1]), np.array([1, 0, 1]), 2)
    with pytest.raises(ValueError, match="conf must be in"):
        est.compare_from_disagreement(_sample(10, 4, 2), *_labels(1, 1, 0), conf=1.0)
    with pytest.raises(ValueError, match="sampled windows but"):
        est.compare_from_disagreement(_sample(10, 4, 3), *_labels(1, 1, 0))


def _class_pair(tmp_path, codes):
    """Two class maps of one 1 x 200 strip in the given class codes, and the truth in the same codes."""
    rs = np.random.RandomState(11)
    n = 200
    t = rs.randint(0, len(codes), size=n)
    a = np.where(rs.rand(n) < 0.85, t, (t + 1) % len(codes))
    b = np.where(rs.rand(n) < 0.6, t, (t + 2) % len(codes))
    lut = np.array(codes)
    np.save(tmp_path / "ca.npy", lut[a][None, :].astype(np.int32))
    np.save(tmp_path / "cb.npy", lut[b][None, :].astype(np.int32))
    return lut[t], lut[a], lut[b]


@pytest.mark.parametrize("codes", [[0, 100, 2000], [-5, 3, 70]])
def test_sparse_and_negative_class_codes_round_trip(tmp_path, codes):
    """compare remaps sparse or negative codes and maps them back; the CSV shows the maps' own codes, and the
    reviewer's class in those codes, negative included, is matched against them."""
    truth, a, b = _class_pair(tmp_path, codes)
    out = tmp_path / "s.csv"
    assert main(["sample", str(tmp_path / "ca.npy"), "--other", str(tmp_path / "cb.npy"), "--budget", "40", "--patch",
                 "1", "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "s.json"))
    assert set(side["class_a"]) | set(side["class_b"]) <= set(codes)
    assert side["class_a"] == [int(a[i]) for i in side["indices"]] and side["class_b"] == [int(b[i]) for i in side["indices"]]
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["reference_class"] = int(truth[int(r["pixel_col"])])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert main(["estimate", str(out)]) == 0
    r = json.load(open(tmp_path / "s_estimate.json"))
    assert r["n_unjudged"] == 0 and r["n_a_right"] + r["n_b_right"] > 0
    ka = sum(int(truth[i]) == x for i, x in zip(side["indices"], side["class_a"]))
    assert r["n_a_right"] == ka


def test_the_threshold_cuts_a_continuous_pair(tmp_path):
    """With two 2-D probability maps, --threshold moves the cut, and so which windows differ."""
    p = np.linspace(0.05, 0.95, 100, dtype=np.float32)[None, :]
    np.save(tmp_path / "pa.npy", p)
    np.save(tmp_path / "pb.npy", np.clip(2 * p, 0, 1))
    counts = []
    for thr in ("0.5", "0.8"):
        out = tmp_path / f"t{thr}.csv"
        assert main(["sample", str(tmp_path / "pa.npy"), "--other", str(tmp_path / "pb.npy"), "--budget", "5",
                     "--patch", "1", "--threshold", thr, "--out", str(out)]) == 0
        counts.append(json.load(open(str(out)[:-4] + ".json"))["n_disagree"])
    a, b = p.ravel(), np.clip(2 * p, 0, 1).ravel()
    assert counts == [int(((a > t) != (b > t)).sum()) for t in (0.5, 0.8)] and counts[0] != counts[1]   # compare cuts at > t


def test_no_data_windows_are_never_sampled_and_the_notes_reach_the_estimate(tmp_path, capsys):
    rs = np.random.RandomState(12)
    n = 120
    a = rs.randint(0, 3, size=n).astype(np.float32)
    b = np.where(rs.rand(n) < 0.5, a, (a + 1) % 3).astype(np.float32)
    b[:30] = np.nan                                         # b predicts nothing on the first 30 windows
    np.save(tmp_path / "na.npy", a[None, :])
    np.save(tmp_path / "nb.npy", b[None, :])
    out = tmp_path / "s.csv"
    assert main(["sample", str(tmp_path / "na.npy"), "--other", str(tmp_path / "nb.npy"), "--budget", "200", "--patch",
                 "1", "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "s.json"))
    assert min(side["indices"]) >= 30 and side["n_population"] == 90
    side["notes"] = ["a test note"]
    json.dump(side, open(tmp_path / "s.json", "w"))
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["reference_class"] = r["class_a"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    capsys.readouterr()
    assert main(["estimate", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "note: a test note" in printed and json.load(open(tmp_path / "s_estimate.json"))["notes"] == ["a test note"]


def test_map_b_wins_and_the_ends_are_printed_the_right_way_round(tmp_path, capsys):
    truth, pa, pb = _maps(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", str(tmp_path / "b.npy"), "--other", str(tmp_path / "a.npy"), "--budget", "60", "--patch",
                 "1", "--out", str(out)]) == 0                      # the better map is now b
    _label_from_truth(out, truth)
    capsys.readouterr()
    assert main(["estimate", str(out)]) == 0
    printed = capsys.readouterr().out
    r = json.load(open(tmp_path / "s_estimate.json"))
    d = r["difference"]
    assert r["verdict"] == "b" and d["high"] < 0
    assert printed.startswith(f"map b is more accurate than map a, by {-100 * d['high']:.1f} to {-100 * d['low']:.1f} points")
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    m = mcp_server.estimate(sample_csv=str(out), out_dir=str(tmp_path / "m"))
    assert m["conclusion"].startswith(f"Map b is more accurate than map a, by {-100 * d['high']:.1f} to {-100 * d['low']:.1f}")
    assert m["next"].startswith("For either map's own accuracy")


def test_an_undecided_comparison_says_so_with_its_ends_in_order(tmp_path, capsys):
    s = _sample(1000, 400, 20)
    r = est.compare_from_disagreement(s, *_labels(8, 7, 5))
    assert r["verdict"] is None and r["difference"]["low"] < 0 < r["difference"]["high"]


def test_every_refusal_of_the_comparison_sample(tmp_path):
    truth, pa, pb = _maps(tmp_path)
    out = tmp_path / "s.csv"
    args = ["sample", str(tmp_path / "a.npy"), "--other", str(tmp_path / "b.npy"), "--budget", "20", "--patch", "1"]
    np.save(tmp_path / "cond.npy", np.zeros((1, 300), np.int16))
    with pytest.raises(SystemExit, match="does not take --condition"):
        main(args + ["--condition", str(tmp_path / "cond.npy"), "--out", str(out)])
    with pytest.raises(SystemExit, match="is a directory"):
        main(args + ["--out", str(tmp_path)])
    assert main(args + ["--out", str(out)]) == 0
    _label_from_truth(out, truth)
    for extra, msg in ((["--scores", str(tmp_path / "a.npy")], "takes none of"), (["--nodata", "0"], "takes none of"),
                       (["--reviewer-false-alarm", "0.1"], "takes none of"), (["--reviewer-miss", "0.1"], "takes none of")):
        with pytest.raises(SystemExit, match=msg):
            main(["estimate", str(out)] + extra)
    good = list(csv.DictReader(open(out, newline="")))

    def write(rows, fields=None):
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write(good[1:] + good[:1])
    with pytest.raises(SystemExit, match="do not match the design"):
        main(["estimate", str(out)])
    write([{k: v for k, v in r.items() if k != "class_b"} for r in good], [k for k in good[0] if k != "class_b"])
    with pytest.raises(SystemExit, match="expected the columns"):
        main(["estimate", str(out)])
    bad = [dict(r) for r in good]
    bad[2]["class_b"] = str(int(bad[2]["class_a"]))
    write(bad)
    with pytest.raises(SystemExit, match="class_a or class_b differs"):
        main(["estimate", str(out)])
    for v in ("1.5", "x"):
        bad = [dict(r) for r in good]
        bad[0]["reference_class"] = v
        write(bad)
        with pytest.raises(SystemExit, match="a whole number"):
            main(["estimate", str(out)])


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed")
def test_the_server_refuses_a_threshold_without_other_and_names_its_parameters(tmp_path):
    from oe_inferencex import mcp_server
    _maps(tmp_path)
    with pytest.raises(mcp_server.Refused, match="--threshold"):
        mcp_server.sample(scores=str(tmp_path / "a.npy"), out_dir=str(tmp_path / "x"), budget=10, patch=1, threshold=0.5)
    assert mcp_server._mcp_words("used only with --other") == "used only with other"
    s = mcp_server.sample(scores=str(tmp_path / "a.npy"), out_dir=str(tmp_path / "m"), budget=10, patch=1,
                          other=str(tmp_path / "b.npy"))
    assert "no reviewer error rate applies to this comparison" in s["limits"]
    assert "only estimate can widen" not in s["limits"]


# ----------------------------------------------------------------------------- the sign test's verdict (1.10.0)
# exp107 (docs/plan/which_map_baselines.md): the exact sign test on the windows where one map is right names the better
# map more often than the union-bound interval up to a fifth of the differing windows, less often past a quarter to two
# fifths; the owner chose the test up to a fifth, the interval above and in a census.
def test_the_sign_tests_critical_values_are_the_sign_tests():
    from oe_inferencex import stats
    for alpha in (0.05, 0.1, 0.01):
        for m in range(0, 121):
            k = est.sign_critical(m, alpha)
            for w in range(m + 1):
                assert (stats.sign_test(w, m - w) < alpha) == (min(w, m - w) <= k), (alpha, m, w)
    for m in (1023, 1024, 1500, 3001):                      # past 2 ** 1024, where a float tail overflowed in exp107
        k = est.sign_critical(m, 0.05)
        assert stats.sign_test(k, m - k) < 0.05 <= stats.sign_test(k + 1, m - k - 1)


def test_the_switch_is_a_fifth_of_the_differing_windows():
    assert [est.which_map_rule(n, 500) for n in (1, 99, 100, 101, 499, 500)] == \
        ["sign"] * 3 + ["interval"] * 3
    assert est.which_map_rule(100, 100, "sign") == "sign" and est.which_map_rule(1, 500, "interval") == "interval"
    with pytest.raises(ValueError, match="rule"):
        est.which_map_rule(1, 5, "union")


def test_the_sign_verdict_never_names_the_worse_map_more_than_its_level_around_the_switch():
    """Exact: on either side of the switch, for equal counts and counts one apart, at several shares wrong in both, the
    package's verdict names the less accurate map with probability at most 0.025."""
    from oe_inferencex import plan
    for D in (25, 101, 400):
        for n in sorted({max(1, D // 5 - 2), D // 5, D // 5 + 1, D // 5 + 3, D // 2}):
            for T in sorted({D, D - 1, (3 * D) // 4, D // 3}):
                for KA in (T // 2, (T + 1) // 2):
                    pa, pb = plan.which_map_probability(D, D, n, KA, T - KA)
                    assert pb <= 0.025 + 1e-12, (D, n, T, KA, pb)
                    if 2 * KA == T:
                        assert pa <= 0.025 + 1e-12, (D, n, T, KA, pa)


def test_windows_marked_unjudged_count_against_the_map_the_test_would_name():
    """a is named only if the test on (a, b + u) names it, so every way the ? windows could truly fall names a too."""
    from oe_inferencex import stats
    D = 10_000
    for a, b, c, u in ((20, 5, 3, 0), (20, 5, 3, 4), (20, 5, 3, 9), (6, 18, 0, 2), (12, 12, 1, 3)):
        r = est.compare_from_disagreement(_sample(D, D, a + b + c + u), *_labels(a, b, c + u, uc=u))
        assert r["verdict_rule"] == "sign" and r["n_unjudged"] == u
        named_a = stats.sign_test(a, b + u) < 0.05 and a > b + u
        named_b = stats.sign_test(a + u, b) < 0.05 and b > a + u
        assert r["verdict"] == ("a" if named_a else "b" if named_b else None), (a, b, u, r["verdict"])
        if r["verdict"]:
            for to_a in range(u + 1):                          # the ? windows truly right in a, the rest in b
                aa, bb = a + to_a, b + u - to_a
                assert stats.sign_test(aa, bb) < 0.05 and (aa > bb) == (r["verdict"] == "a")


def test_a_census_and_the_interval_side_keep_the_intervals_verdict():
    r = est.compare_from_disagreement(_sample(1000, 40, 40), *_labels(21, 19, 0))
    assert r["verdict_rule"] == "interval" and r["verdict"] == "a" == r["interval_verdict"]
    r = est.compare_from_disagreement(_sample(1000, 100, 60), *_labels(40, 20, 0))
    assert r["verdict_rule"] == "interval" and r["verdict"] == r["interval_verdict"]


def test_the_test_can_name_a_map_where_the_interval_cannot_and_every_reply_says_which_decided(tmp_path):
    from oe_inferencex import decide
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 100), *_labels(30, 15, 55))
    assert r["verdict_rule"] == "sign" and r["verdict"] == "a" and r["interval_verdict"] is None
    assert r["difference"]["low"] < 0 < r["difference"]["high"]
    head = decide.sign_test_head(r)
    assert head.startswith("map a is more accurate than map b (exact sign test") and "p = " in head
    assert "can include 0" in head
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 100), *_labels(42, 25, 33))
    assert r["verdict"] == "a" and "p = 0.0498" in decide.sign_test_head(r)
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 100), *_labels(22, 10, 68))
    assert r["verdict"] is None and "p = 0.0501" in decide.sign_test_head(r), "p is not rounded onto the cut"
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 65), *_labels(30, 10, 25, uc=25))
    head = decide.sign_test_head(r)
    assert r["verdict"] is None and "whichever map leads" in head and "p = " not in head
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 100), *_labels(30, 15, 55), conf=0.9)
    assert "90% interval" in decide.sign_test_head(r)
    old = {k: v for k, v in r.items() if k not in ("verdict_rule", "sign_test")}
    assert decide.sign_test_head(old) is None, "a result written before 1.10.0 is read as the interval's"

"""A second reading of a map's labels on every design: estimate and certify --also-accept (13 October 2026).

The per-class audit's --also-accept (1.11.0) read the same labels a second time with a fixed lenient crosswalk. Here
the same reading reaches the whole-map error rate and its interval, each condition's, and the certified zone. A
crosswalk fixed before any label makes "wrong under it" a fixed property of each window, so each estimator and zone
rule keeps its own guarantee for that reading, and the two readings hold together by the union bound. These tests
check the reading, its coverage by exact enumeration (with an adversary marking ? after seeing the truth), the joint
statement and why it is needed, certify's guarantee under both readings on every error pattern of a small map, that
the strict result is the one written without the option byte for byte, the routes through the command line, decide,
the server's refusal, sample --extend keeping the column, and every refusal.
"""
import csv
import importlib.util
import inspect
import itertools
import json
import os
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import cli
from oe_inferencex import decide as decide_mod
from oe_inferencex import estimate as est
from oe_inferencex import sequential as sq
from oe_inferencex.cli import main

THREE = ["also_accept", "also_accept_unused", "reading_note"]
# what a certify result says that does not depend on the labels: the same under both readings
LABEL_FREE = ("rule", "alpha", "delta", "n_population", "n_labelled", "level_cut", "c_min", "min_labels_to_certify",
              "anchor", "delta_per_condition", "n_conditions_tested", "scope")


# ----------------------------------------------------------------------------- the reading
def test_the_reading_counts_only_the_listed_pairs_and_refuses_what_it_must():
    wrong = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    mc = np.array([2, 2, 2, 2, 2, 1, 1, 0])
    ref = np.array([2, 3, 3, 4, -1, 0, 1, 2])
    unj = np.array([False] * 4 + [True] + [False] * 3)
    wrong[4] = 1                                                       # a ? row keeps the value it was given
    r = est.also_accept_reading(wrong, mc, ref, {"2": [3, 2], "1": [0], "7": [1]}, unjudged=unj)
    assert r["wrong"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1] and r["wrong"].dtype == wrong.dtype
    assert r["errors_accepted"] == {"1": {"0": 1}, "2": {"3": 2}} and r["n_accepted"] == 3
    assert "accepted" not in r          # in the audit's block, `accepted` is the list of codes a class accepts
    assert r["pairs"] == {"1": [0], "2": [2, 3], "7": [1]} and r["unused"] == ["7"]
    assert (r["wrong"] <= wrong).all()                                 # never more errors, window by window
    # a class listed for itself changes nothing; nor does a pair the map's class is not keyed by (the crosswalk is
    # directional: {"3": [2]} accepts reference 2 where the map says 3, not 3 where it says 2)
    for pairs in ({"2": [2]}, {2: [2]}, {"3": [2]}, {}):
        same = est.also_accept_reading(wrong, mc, ref, pairs, unjudged=unj)
        assert same["wrong"].tolist() == wrong.tolist() and same["n_accepted"] == 0 and same["errors_accepted"] == {}
    assert est.also_accept_reading(wrong, mc, ref, {"3": [2]}, unjudged=unj)["unused"] == ["3"]
    # a ? window's reference is never read, whatever it holds
    junk = ref.astype(float)
    junk[4] = np.nan
    assert est.also_accept_reading(wrong, mc, junk, {"2": [3]}, unjudged=unj)["n_accepted"] == 2
    for args, msg in (((wrong[:-1], mc, ref, {}), "7 labels but 8, 8 and 7 map classes"),
                      ((np.r_[wrong[:-1], 2], mc, ref, {}), "wrong must be 0 or 1"),
                      ((wrong, mc, np.r_[ref[:-1], -1], {}), "whole number of at least 0; window 7 has -1"),
                      ((wrong, mc, np.r_[ref[:-1], 2.5], {}), "whole number of at least 0; window 7 has 2.5"),
                      ((wrong, mc, np.r_[3, ref[1:]], {}), r"disagrees .* at window 0 \(wrong 0, map class 2, "
                                                           r"reference class 3\)"),
                      ((wrong, mc, ref, {"x": [3]}), "not a class code"),
                      ((wrong, mc, ref, {"2": 3}), "not a list"),
                      ((wrong, mc, ref, [2, 3]), "maps class codes")):
        with pytest.raises(ValueError, match=msg):
            est.also_accept_reading(*args, unjudged=unj if len(args[0]) == 8 else None)
    import oe_inferencex                                               # not exported, as the audit's functions are not
    assert not hasattr(oe_inferencex, "also_accept_reading") and "also_accept_reading" not in oe_inferencex.__all__
    assert set(inspect.signature(est.also_accept_reading).parameters) == {"wrong", "map_class", "reference",
                                                                         "also_accept", "unjudged"}


def test_the_notes_state_each_level_and_the_union_bound():
    note = " ".join(est.ESTIMATE_READING_NOTE.split())
    assert "If the file was fixed before any label was read, each reading's interval holds at its own level" in note
    assert "hold together at least 90% of the time where each holds at 95% (a union bound)" in note
    assert "the whole map's two, and each condition's two" in note
    assert "written after the labels were seen can be fitted to them, and then neither holds" in note
    z = cli._zone_reading_note(0.1, False)
    assert "each certificate fails on at most 0.1 of samples" in z and "either is wrong is at most 0.2 (a union" in z
    assert "can be larger than the strict one" in z and "then neither certificate holds" in z
    assert "at most 1 (a union bound)" in cli._zone_reading_note(0.6, False)          # capped, as decide caps it
    assert cli._zone_reading_note(0.1, True) == z + est.ZONE_READING_SEQUENTIAL
    assert "the file must be the same at every look" in est.ZONE_READING_SEQUENTIAL


# ----------------------------------------------------------------------------- exact coverage of the whole map
def _kinds_sample(kinds, draw, N):
    """A random sample of the windows `draw` from a map of N windows, all mapped as class 2, whose truth is right (0),
    a reference class the file accepts (1: reference 3) or another (2: reference 4)."""
    kinds = np.asarray(kinds)[list(draw)]
    ref = np.array([2, 3, 4])[kinds]
    return {"design": "random", "indices": np.array(draw), "n_population": N, "budget": len(draw)}, \
        (kinds > 0).astype(int), np.full(len(draw), 2), ref


def test_both_readings_are_exact_and_hold_together_on_a_small_map():
    """9 windows holding A accepted and O other errors in every count, every draw of 4: the strict interval covers
    (A + O) / 9 as exact_coverage_srs says, the alternative covers O / 9 as it says, both at least 95%; the two hold
    together at least 90% and at least 1 - P(strict misses) - P(alternative misses), in exact counts; and the
    alternative's ends are never above the strict ones (the random design's interval moves one way with the count)."""
    N, n = 9, 4
    draws = list(itertools.combinations(range(N), n))
    for A in range(N + 1):
        for O in range(N - A + 1):
            kinds = [1] * A + [2] * O + [0] * (N - A - O)
            hit_s = hit_a = both = 0
            for draw in draws:
                s, wrong, mc, ref = _kinds_sample(kinds, draw, N)
                alt_w = est.also_accept_reading(wrong, mc, ref, {"2": [3]})["wrong"]
                rs, ra = est.estimate_error_rate(s, wrong), est.estimate_error_rate(s, alt_w)
                cs, ca = rs["low"] <= (A + O) / N <= rs["high"], ra["low"] <= O / N <= ra["high"]
                hit_s, hit_a, both = hit_s + cs, hit_a + ca, both + (cs and ca)
                assert ra["low"] <= rs["low"] and ra["high"] <= rs["high"]
            D = len(draws)
            assert hit_s / D == pytest.approx(est.exact_coverage_srs(N, A + O, n, est.hypergeom_interval), abs=1e-12)
            assert hit_a / D == pytest.approx(est.exact_coverage_srs(N, O, n, est.hypergeom_interval), abs=1e-12)
            assert hit_s / D >= 0.95 and hit_a / D >= 0.95, (A, O)
            assert Fraction(both, D) >= Fraction(9, 10) and both >= D - (D - hit_s) - (D - hit_a)


def test_both_readings_hold_against_an_adversary_marking_question_marks():
    """The same map, with a reviewer who, after seeing each drawn window's truth, marks any of them ? in both columns:
    on at least 95% of draws every marking keeps each reading's interval over its own rate, and on at least 90% both.
    The ? windows are the same in both readings and are bounded both ways in each. A ? window is not read by the
    second reading (the first test), so each marking's counts are the unmarked reading's over the judged windows."""
    N, n = 9, 4
    draws = list(itertools.combinations(range(N), n))
    marks = np.array(list(itertools.product([False, True], repeat=n)))
    memo = {}

    def interval(k, u):                     # the random design's interval depends only on the counts; through estimate
        if (k, u) not in memo:
            w = np.r_[np.ones(k, int), np.zeros(n - k, int)]
            unj = np.r_[np.zeros(n - u, bool), np.ones(u, bool)]
            r = est.estimate_error_rate({"design": "random", "indices": np.arange(n), "n_population": N}, w,
                                        unjudged=unj)
            memo[(k, u)] = (r["low"], r["high"])
        return memo[(k, u)]

    for A in range(N + 1):
        for O in range(N - A + 1):
            kinds = [1] * A + [2] * O + [0] * (N - A - O)
            ok_s = ok_a = ok_both = 0
            for draw in draws:
                s, wrong, mc, ref = _kinds_sample(kinds, draw, N)
                alt_w = est.also_accept_reading(wrong, mc, ref, {"2": [3]})["wrong"]
                each_s = each_a = True
                for k_s, k_a, u in zip((~marks * wrong).sum(1), (~marks * alt_w).sum(1), marks.sum(1)):
                    lo, hi = interval(int(k_s), int(u))
                    each_s &= lo <= (A + O) / N <= hi
                    lo, hi = interval(int(k_a), int(u))
                    each_a &= lo <= O / N <= hi
                ok_s, ok_a, ok_both = ok_s + each_s, ok_a + each_a, ok_both + (each_s and each_a)
            assert ok_s / len(draws) >= 0.95 and ok_a / len(draws) >= 0.95, (A, O)
            assert ok_both / len(draws) >= 0.9, (A, O)
    # the memo and the counts stood for the package's own route: a draw with ? rows, read by the reading and estimate
    s, wrong, mc, ref = _kinds_sample([1, 2, 0, 1, 2, 0, 0, 1, 2], (0, 1, 3, 4), N)
    unj = np.array([False, True, False, True])
    alt_w = est.also_accept_reading(wrong, mc, ref, {"2": [3]}, unjudged=unj)["wrong"]
    assert (lambda r: (r["low"], r["high"]))(est.estimate_error_rate(s, wrong, unjudged=unj)) == interval(2, 2)
    assert (lambda r: (r["low"], r["high"]))(est.estimate_error_rate(s, alt_w, unjudged=unj)) == interval(0, 2)


def test_each_condition_is_exact_in_both_readings_under_the_condition_design():
    """Conditions of 4 and 3 windows, 2 labels each, every mix of right, accepted and other windows in each, every
    sample: each condition's alternative interval covers its own O_c / N_c exactly as exact_coverage_srs says; the
    whole-map alternative covers (O_0 + O_1) / 7 at least 95% of the time; and each condition's two intervals, and
    the whole map's two, hold together at least 90% of the time."""
    sizes, n = (4, 3), 2
    cond = np.r_[np.zeros(sizes[0], int), np.ones(sizes[1], int)]
    base = est.sample_for_estimation(np.linspace(1, 0, cond.size), 4, design="condition", condition=cond)
    assert base["allocation"] == [2, 2]
    pools = [np.flatnonzero(cond == c) for c in (0, 1)]
    picks = [np.array(a + b) for a in itertools.combinations(pools[0], n) for b in itertools.combinations(pools[1], n)]

    def mixes(Nc):
        return [(a, o) for a in range(Nc + 1) for o in range(Nc - a + 1)]

    for (A0, O0), (A1, O1) in itertools.product(mixes(sizes[0]), mixes(sizes[1])):
        kinds = np.r_[[1] * A0 + [2] * O0 + [0] * (sizes[0] - A0 - O0), [1] * A1 + [2] * O1 + [0] * (sizes[1] - A1 - O1)]
        hit = {c: [0, 0, 0] for c in ("0", "1", "map")}
        for picked in picks:
            wrong, ref = (kinds[picked] > 0).astype(int), np.array([2, 3, 4])[kinds[picked]]
            alt_w = est.also_accept_reading(wrong, np.full(picked.size, 2), ref, {"2": [3]})["wrong"]
            rs = est.estimate_error_rate(dict(base, indices=picked), wrong)
            ra = est.estimate_error_rate(dict(base, indices=picked), alt_w)
            for c, ss, aa, ts, ta in (("0", rs["per_condition"]["0"], ra["per_condition"]["0"], (A0 + O0) / 4, O0 / 4),
                                      ("1", rs["per_condition"]["1"], ra["per_condition"]["1"], (A1 + O1) / 3, O1 / 3),
                                      ("map", rs, ra, (A0 + O0 + A1 + O1) / 7, (O0 + O1) / 7)):
                cs, ca = ss["low"] <= ts <= ss["high"], aa["low"] <= ta <= aa["high"]
                hit[c][0], hit[c][1], hit[c][2] = hit[c][0] + cs, hit[c][1] + ca, hit[c][2] + (cs and ca)
        D = len(picks)
        for c, (Nc, Oc) in (("0", (4, O0)), ("1", (3, O1))):
            assert hit[c][1] / D == pytest.approx(est.exact_coverage_srs(Nc, Oc, n, est.hypergeom_interval), abs=1e-12)
        assert hit["map"][1] / D >= 0.95
        assert all(h[2] / D >= 0.9 for h in hit.values()), (A0, O0, A1, O1, hit)


# ----------------------------------------------------------------------------- certify, every pattern of a small map
@pytest.mark.parametrize("N,B,alpha,delta,shows_the_bound", [(8, 5, 0.5, 0.3, True), (8, 5, 0.4, 0.3, False)])
def test_each_certificate_holds_delta_and_both_two_delta_on_every_pattern(N, B, alpha, delta, shows_the_bound):
    """Every pattern of N windows in zone order, each right, accepted or another error (3^N), against every draw of B:
    under each rule each reading certifies a zone wrong more than alpha of the time (for its own errors) on at most
    delta of draws, and either does on at most 2 delta, in exact fractions; on every draw the second reading's zone
    holds the strict one (its counts are never larger, the p-value moves one way with the count, and the rules keep
    that order). On the first map some pattern has either wrong on more than delta of draws, so the union bound, not
    delta, is what the two hold together at. The certified level of a draw depends only on which of its B labels are
    wrong, so it is computed once per draw and label pattern with the package's zone_counts, zone_pvalue and
    apply_zone_rule, as tests/test_trust_zone.py does."""
    grid = tuple(j / N for j in range(1, N + 1))
    cov, sizes, _ = est.zone_levels(N, B, alpha, delta, grid)
    sizes = np.asarray(sizes, dtype=np.int64)
    assert sizes.size >= 4                                    # several levels, so the order of the tests matters
    draws = np.array(list(itertools.combinations(range(N), B)))
    D, rules = len(draws), ("prefix", "bonferroni")
    best = {r: np.full((D, 2 ** B), -1) for r in rules}
    for d, pos in enumerate(draws):
        for m in range(2 ** B):
            b, k = est.zone_counts(pos, np.array([(m >> j) & 1 for j in range(B)]), sizes)
            p = [est.zone_pvalue(int(kk), int(bb), int(nn), alpha) for kk, bb, nn in zip(k, b, sizes)]
            for r in rules:
                j = est.apply_zone_rule(p, b, k, alpha, delta, r, n=sizes)[1]
                best[r][d, m] = -1 if j is None else j
    powers, a, rows = 1 << np.arange(B), Fraction(str(alpha)), np.arange(D)
    worst = {r: [0, 0, 0] for r in rules}
    exceeds = 0
    for kinds in itertools.product(range(3), repeat=N):
        kinds = np.array(kinds)
        e_s, e_a = kinds > 0, kinds == 2                      # wrong strictly; wrong under the alternative
        # bad[j]: the zone of level j is wrong more than alpha of the time, exactly; the last entry stands for "none"
        bad_s = np.array([Fraction(int(c), int(s)) > a for c, s in zip(np.cumsum(e_s)[sizes - 1], sizes)] + [False])
        bad_a = np.array([Fraction(int(c), int(s)) > a for c, s in zip(np.cumsum(e_a)[sizes - 1], sizes)] + [False])
        m_s, m_a = (e_s[draws] * powers).sum(1), (e_a[draws] * powers).sum(1)
        for r in rules:
            js, ja = best[r][rows, m_s], best[r][rows, m_a]
            assert ((js < 0) | (ja >= js)).all()               # nested on every draw
            vs, va = bad_s[js], bad_a[ja]
            counts = [int(vs.sum()), int(va.sum()), int((vs | va).sum())]
            worst[r] = [max(x, y) for x, y in zip(worst[r], counts)]
            exceeds += Fraction(counts[2], D) > Fraction(str(delta))
    d = Fraction(str(delta))
    for r in rules:
        assert Fraction(worst[r][0], D) <= d and Fraction(worst[r][1], D) <= d, (r, worst[r])
        assert Fraction(worst[r][2], D) <= 2 * d, (r, worst[r])
    assert worst["prefix"][0] > 0 and worst["prefix"][1] > 0                 # the rule does certify bad zones, below delta
    assert bool(exceeds) == shows_the_bound


# ----------------------------------------------------------------------------- the command line
C = 4


def _write_map(tmp_path, H=20, W=20, seed=5):
    """Four classes on a 20 x 20 grid of probability vectors (patch 1: a window is a pixel), the map predicting 0, 1
    or 2; the truth puts a 3 under a 2 (the contested pair) more often the less confident the map, a quarter of the 2s
    on average, a 0 under 3% of the 1s and a 2 under 1% of the 0s. A condition layer: left half 0, right half 1."""
    rng = np.random.default_rng(seed)
    pred = rng.integers(0, 3, (H, W))
    top = rng.uniform(0.40, 0.95, (H, W))
    p = np.repeat(((1 - top) / (C - 1))[None], C, axis=0)
    np.put_along_axis(p, pred[None], top[None], axis=0)
    np.save(tmp_path / "p.npy", p)
    truth = pred.copy()
    u = rng.random((H, W))
    truth[(pred == 2) & (u < 0.5 * (0.95 - top) / 0.55)] = 3
    truth[(pred == 1) & (u < 0.03)] = 0
    truth[(pred == 0) & (u < 0.01)] = 2
    np.save(tmp_path / "cond.npy", np.where(np.arange(W)[None, :] < W // 2, 0, 1) * np.ones((H, 1), int))
    with open(tmp_path / "pairs.json", "w") as f:
        json.dump({"2": [3], "7": [1]}, f)
    return str(tmp_path / "p.npy"), truth


SAMPLES = {"random": ["--design", "random"], "confidence": ["--design", "confidence"],
           "proportional": ["--design", "proportional"], "tiles": ["--design", "tiles", "--tile", "4", "--per-tile", "5"],
           "condition": ["--condition", "COND", "--condition-names", "0=left", "1=right"],
           "random+condition": ["--design", "random", "--condition", "COND", "--condition-names", "0=left", "1=right"],
           "sequential": ["--design", "sequential", "--alpha", "0.2"]}


def _sample(tmp_path, p, design, budget=120, name="s"):
    out = str(tmp_path / f"{name}.csv")
    extra = [str(tmp_path / "cond.npy") if x == "COND" else x for x in SAMPLES[design]]
    assert main(["sample", p, "--budget", str(budget), "--patch", "1", "--seed", "1", "--out", out, *extra]) == 0
    return out


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rewrite(path, rows, fields=None):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _label(path, truth, unjudged=(), upto=None):
    """Fill `wrong` and `reference_class` as a reviewer would, from the truth at each window; ? in both at the rows in
    `unjudged`; with `upto`, only the first rows (a sequential sample, labelled from the top)."""
    rows = _rows(path)
    flat = truth.ravel()
    for i, r in enumerate(rows):
        if upto is not None and i >= upto:
            r["wrong"] = r["reference_class"] = ""
        elif i in unjudged:
            r["wrong"] = r["reference_class"] = "?"
        else:
            t = int(flat[int(r["index"])])
            r["reference_class"], r["wrong"] = str(t), str(int(t != int(r["map_class"])))
    _rewrite(path, rows)
    return rows


def _run(capsys, argv):
    capsys.readouterr()
    assert main(argv) == 0
    return capsys.readouterr().out


def _json(path):
    with open(path) as f:
        return json.load(f)


def _roundtrip(x):
    return json.loads(json.dumps(x, default=float))


def _stripped_is(plain_path, both_path):
    """The result with the option, without its three new keys (which come last but for written_by, the last key of
    every result file from 1.14.1), is the one written without the option, byte for byte."""
    both = _json(both_path)
    assert list(both)[-4:] == THREE + ["written_by"]
    with open(plain_path) as f:
        assert json.dumps({k: v for k, v in both.items() if k not in THREE}, indent=1) == f.read()
    return both


def _prefix(plain, both):
    """The strict printed lines, all but `wrote`, are the start of the printed text with the option."""
    head = plain.rsplit("\nwrote ", 1)[0]
    assert both.startswith(head + "\nalso accepting "), (head[-200:], both[len(head) - 50:len(head) + 200])


def _second_reading(path):
    """The sample as the command reads it, and the second reading's `wrong` computed here from the CSV's columns."""
    side, rows, idx, sample, wrong, unjudged = cli._labelled_sample(path, "estimate")
    ref = np.array([-1 if r["reference_class"].strip() == "?" else int(r["reference_class"]) for r in rows])
    mc = np.array([int(r["map_class"]) for r in rows])
    rd = est.also_accept_reading(wrong, mc, ref, {"2": [3], "7": [1]}, unjudged=unjudged)
    return side, rows, idx, sample, wrong, unjudged, rd


@pytest.mark.parametrize("design", list(SAMPLES))
def test_estimate_writes_the_strict_result_unchanged_and_the_second_reading_beside_it(tmp_path, capsys, design):
    """Through the command line, every design: the result written with --also-accept, less its three new keys, is
    the one written without it, byte for byte; the strict printed lines come first, unchanged; and the second
    reading's block is the package's estimator run on the second reading's `wrong` (nominal designs included)."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, design)
    _label(s, truth, unjudged=() if design == "tiles" else (1, 4), upto=80 if design == "sequential" else None)
    pairs = str(tmp_path / "pairs.json")
    plain = _run(capsys, ["estimate", s, "--out", str(tmp_path / "plain.json")])
    both = _run(capsys, ["estimate", s, "--out", str(tmp_path / "both.json"), "--also-accept", pairs])
    r = _stripped_is(tmp_path / "plain.json", tmp_path / "both.json")
    _prefix(plain, both)
    side, rows, idx, sample, wrong, unjudged, rd = _second_reading(s)
    if design == "sequential":
        sample = dict(sample, design="random", budget=int(idx.size))
    lib = est.estimate_error_rate(sample, rd["wrong"], unjudged=unjudged)
    a = r["also_accept"]
    assert a["source"] == pairs and a["pairs"] == {"2": [3], "7": [1]} and r["also_accept_unused"] == ["7"]
    assert a["errors_accepted"] == rd["errors_accepted"] and a["n_accepted"] == rd["n_accepted"] > 0
    assert "accepted" not in a
    assert a["n_wrong"] == int(rd["wrong"][~unjudged].sum()) == int(wrong.sum()) - a["n_accepted"]
    for k in ("estimate", "estimate_range", "low", "high", "half_width", "method", "warning",
              "outside_condition_intervals"):
        assert a.get(k) == _roundtrip(lib.get(k)), k
    if "per_condition" in lib:
        for name, row in lib["per_condition"].items():
            assert a["per_condition"][name] == _roundtrip({q: row[q] for q in ("n_labelled", "n_wrong", "n_unjudged",
                                                                                "estimate", "estimate_range", "low",
                                                                                "high", "half_width", "method")
                                                           if q in row})
    else:
        assert "per_condition" not in a
    assert r["reading_note"] == est.ESTIMATE_READING_NOTE
    if design in ("random", "condition", "random+condition", "sequential"):
        # these intervals move one way with the error count, and the second reading counts fewer errors
        assert a["low"] <= r["low"] and a["high"] <= r["high"]
    lines = both.splitlines()
    first = next(x for x in lines if x.startswith("also accepting "))
    assert first.startswith("also accepting 3 where the map says 2; 1 where the map says 7 (pairs.json): error rate ")
    assert f"{a['n_accepted']} labelled windows wrong above are right under this reading (2 seen as 3: " in first
    assert f"note: {est.ESTIMATE_READING_NOTE}" in lines
    assert "note: --also-accept lists class(es) 7, which no labelled window is mapped as; they are not read" in lines
    if r.get("by_condition"):
        assert any(x.startswith("  left ") for x in lines[lines.index(first):])
    assert lines[-1] == f"wrote {tmp_path / 'both.json'}"


def _zone_library(design, side, sample, idx, w, margin, valid, alpha, delta):
    """The second reading certified by the package's own function for the route, as certify would."""
    if design == "sequential":
        return sq.certify_zone_sequential(margin.ravel(), idx, w, alpha, delta=delta, anchor=float(side["anchor"]),
                                          valid=valid.ravel(), seed=int(side["seed"]))
    if "condition" in design:
        return est.certify_by_condition(sample, w, margin.ravel(), alpha, delta=delta, valid=valid.ravel())
    return est.certify_zone(margin.ravel(), idx, w, alpha, delta=delta, valid=valid.ravel())


@pytest.mark.parametrize("design", ["random", "condition", "random+condition", "sequential"])
def test_certify_writes_the_strict_zone_unchanged_and_the_second_zone_beside_it(tmp_path, capsys, design):
    """certify on each route (one look, per condition, sequential): the strict result, printed lines and mask are
    those written without the option; the label-free keys are the same in both readings; the second block is the
    package's own rule run on the second reading's `wrong`; and its zone holds the strict one."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, design, budget=200)
    _label(s, truth, unjudged=(2, 7), upto=150 if design == "sequential" else None)
    pairs = str(tmp_path / "pairs.json")
    args = ["certify", s, "--alpha", "0.15", "--delta", "0.1", "--out", str(tmp_path / "z.json")]
    strict_mask, alt_mask = tmp_path / "z.npy", tmp_path / "z_also_accept.npy"
    plain = _run(capsys, args)                                 # the result names its mask, so both go to one path
    os.replace(tmp_path / "z.json", tmp_path / "plain.json")
    before = np.load(strict_mask) if os.path.exists(strict_mask) else None
    both = _run(capsys, [*args, "--also-accept", pairs])
    r = _stripped_is(tmp_path / "plain.json", tmp_path / "z.json")
    _prefix(plain, both)
    assert before is not None and np.array_equal(before, np.load(strict_mask))
    side, rows, idx, sample, wrong, unjudged, rd = _second_reading(s)
    margin, hard, valid, _ = cli._map_windows(side, None, None, "certify", rows, idx)
    # a ? window is wrong in both readings
    lib = _zone_library(design, side, sample, idx, np.where(unjudged, 1, rd["wrong"]), margin, valid, 0.15, 0.1)
    for k in LABEL_FREE:
        assert r.get(k) == _roundtrip(lib.get(k)), k
    lib.pop("zone_indices_in_order", None)
    for e in (lib.get("per_condition") or {}).values():
        e.pop("zone_indices_in_order", None)
    a = r["also_accept"]
    for k in cli.ZONE_ALSO_KEYS:
        assert a.get(k) == _roundtrip(lib.get(k)), k
    assert not set(LABEL_FREE) & set(a)                       # the label-free keys are not repeated
    certified = a.get("certified_share_of_map") if "condition" in design else a.get("coverage")
    assert certified is not None and a["zone_mask"] == str(alt_mask) and os.path.exists(alt_mask)
    assert not (np.load(strict_mask) & ~np.load(alt_mask)).any()           # nested
    assert r["reading_note"] == cli._zone_reading_note(0.1, design == "sequential")
    lines = both.splitlines()
    assert f"note: {r['reading_note']}" in lines and lines[-1].startswith(f"wrote {tmp_path / 'z.json'}")
    if "condition" in design:
        assert both.split("\nalso accepting ", 1)[1].startswith("3 where the map says 2; 1 where the map says 7 "
                                                               "(pairs.json), certified per input condition at the "
                                                               "same alpha and delta:")
        assert f"together {100 * a['certified_share_of_map']:.1f}% of the map under this reading (mask {alt_mask})" \
               in lines
    else:
        assert any(x.startswith("also accepting 3 where the map says 2; 1 where the map says 7 (pairs.json): the "
                                f"{100 * a['coverage']:.0f}% most confident windows")
                   and "under this reading, at the same delta" in x and x.endswith(f"window mask {alt_mask}")
                   for x in lines)
    # decide reads both: the strict answer as without the option, the second beside it
    plain_d = decide_mod.decide(str(tmp_path / "plain.json"), ["trusted_share", "trusted_share_at_least=0.5"])
    both_d = decide_mod.decide(str(tmp_path / "z.json"), ["trusted_share", "trusted_share_at_least=0.5"])
    for q in plain_d["answers"]:
        x, y = plain_d["answers"][q], both_d["answers"][q]
        assert y["answer"] == x["answer"] and y["level"] == x["level"]
        assert {k: v for k, v in y["evidence"].items() if k not in ("also_accept", "reading_note")} == x["evidence"]
        assert y["evidence"]["also_accept"] == a and y["evidence"]["reading_note"] == r["reading_note"]
        share2 = a["certified_share_of_map"] if "condition" in design else a["n_zone"] / r["n_population"]
        assert y["also_accept"]["certified"] == share2 and y["also_accept"]["pairs"] == a["pairs"]
        assert y["also_accept"]["answer"] == (share2 if q == "trusted_share" else
                                              "yes" if share2 >= 0.5 else "undetermined")
        assert "Read a second time, also accepting the reference classes the also-accept file lists" in y["because"]
        assert "the chance that either certificate is wrong is at most 20.0%, twice delta (a union bound)" in y["because"]
        assert ("at every look of this sequential sample, with the same file at every look" in y["because"]) == (
            design == "sequential")
        assert "then neither certificate holds" in y["because"]
        if "condition" in design:
            for name, c in y["per_condition"].items():
                e2 = a["per_condition"][name]
                assert {k: v for k, v in c.items() if k != "also_accept"} == x["per_condition"][name]
                assert c["also_accept"]["certified_share_of_condition"] == (
                    e2["n_zone"] / e2["n_population"] if e2.get("n_zone") else 0.0 if e2.get("tested") else None)


def test_only_the_second_reading_certifies_and_a_rerun_removes_its_mask(tmp_path, capsys):
    """A map whose every error is the contested pair: class 2 where the reference says 3, on 40% of the windows, the
    most confident among them. Strictly nothing is certified at 10%; read with {"2": [3]}, the whole map is. Run again
    without the option, certify removes the second mask a previous run left, as it removes a stale strict one."""
    rng = np.random.default_rng(2)
    n = 300
    pred = np.where(rng.random(n) < 0.4, 2, rng.integers(0, 2, n))
    top = rng.uniform(0.4, 0.95, n)
    p = np.repeat(((1 - top) / (C - 1))[None, :], C, axis=0)
    p[pred, np.arange(n)] = top
    np.save(tmp_path / "p.npy", p[:, None, :])
    truth = np.where(pred == 2, 3, pred)[None, :]
    with open(tmp_path / "pairs.json", "w") as f:
        json.dump({"2": [3]}, f)
    s = str(tmp_path / "s.csv")
    assert main(["sample", str(tmp_path / "p.npy"), "--budget", "120", "--design", "random", "--patch", "1",
                 "--out", s]) == 0
    _label(s, truth)
    out = _run(capsys, ["certify", s, "--alpha", "0.1", "--also-accept", str(tmp_path / "pairs.json")])
    r = _json(tmp_path / "s_zone.json")
    assert r["coverage"] is None and not os.path.exists(tmp_path / "s_zone.npy")
    a = r["also_accept"]
    assert a["coverage"] == 1.0 and a["n_zone"] == n and a["upper_bound"] < 0.1 and a["n_accepted"] > 30
    assert all(lv["n_wrong_inside"] == 0 for lv in a["levels"])
    mask = np.load(tmp_path / "s_zone_also_accept.npy")
    assert mask.shape == (1, n) and mask.all() and a["zone_mask"] == str(tmp_path / "s_zone_also_accept.npy")
    assert out.startswith("no zone certified at alpha=0.1, delta=0.1 from 120 labels.")
    assert (f"\nalso accepting 3 where the map says 2 (pairs.json): the 100% most confident windows ({n} of {n}, "
            "confidence margin >= ") in out
    assert "are wrong at most 10% of the time under this reading, at the same delta (prefix rule; exact upper bound" in out
    # decide: nothing certified strictly, the whole map under the second reading
    d = decide_mod.decide(str(tmp_path / "s_zone.json"), ["trusted_share_at_least=0.9"])["answers"]
    assert d["trusted_share_at_least=0.9"]["answer"] == "undetermined"
    assert d["trusted_share_at_least=0.9"]["also_accept"] == {"answer": "yes", "certified": 1.0, "pairs": {"2": [3]}}
    said = _run(capsys, ["decide", str(tmp_path / "s_zone.json"), "--ask", "trusted_share_at_least=0.9"]).splitlines()
    assert said[0].startswith("trusted_share_at_least=0.9: undetermined. ")
    assert said[1] == "  also accepting the file's pairs: yes (certified: 100.0%)"
    # a rerun without the option: the strict result alone, and the second mask is gone
    _run(capsys, ["certify", s, "--alpha", "0.1"])
    assert not os.path.exists(tmp_path / "s_zone_also_accept.npy") and "also_accept" not in _json(tmp_path / "s_zone.json")


def test_decide_carries_the_second_answer_beside_the_strict_one(tmp_path, capsys):
    """error_rate_below on an estimate with --also-accept, whole map and per condition: the strict answer, evidence and
    per-condition answers are those of the result without the option; the second answers sit beside them, read by
    the same rule from the second intervals; the because sentence says it is a second test and what voids it; and
    the printed lines carry both."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "condition", budget=160)
    _label(s, truth, unjudged=(3,))
    _run(capsys, ["estimate", s, "--out", str(tmp_path / "plain.json")])
    _run(capsys, ["estimate", s, "--out", str(tmp_path / "both.json"), "--also-accept", str(tmp_path / "pairs.json")])
    r = _json(tmp_path / "both.json")
    a = r["also_accept"]
    t = round((a["high"] + r["high"]) / 2, 3)               # between the two upper ends: the readings can differ
    q = f"error_rate_below={t:g}"
    x = decide_mod.decide(str(tmp_path / "plain.json"), [q])["answers"][q]
    y = decide_mod.decide(str(tmp_path / "both.json"), [q])["answers"][q]
    assert (y["answer"], y["level"], y["exact"]) == (x["answer"], x["level"], x["exact"])
    assert {k: v for k, v in y["evidence"].items() if k not in ("also_accept", "reading_note")} == x["evidence"]
    assert y["evidence"]["also_accept"] == a and y["evidence"]["reading_note"] == est.ESTIMATE_READING_NOTE
    assert y["also_accept"] == {"answer": decide_mod._rate_below(a, t)[0], "low": a["low"], "high": a["high"],
                                "level": 0.95, "exact": True, "pairs": a["pairs"]}
    assert y["also_accept"]["answer"] == "yes" and x["answer"] != "yes"
    four = " " + decide_mod.readings_second_look(exact=True)       # a new sample read both ways: four tests
    for name, c in y["per_condition"].items():
        then = x["per_condition"][name]
        assert {k: v for k, v in c.items() if k != "also_accept"} == dict(
            then, because=then["because"] + (four if c["answer"] == "undetermined" and "also_accept" in c else ""))
        row = a["per_condition"][name]
        if row["n_labelled"]:
            assert c["also_accept"] == {"answer": decide_mod._rate_below(row, t)[0], "low": row["low"],
                                        "high": row["high"], "exact": True}
    assert y["because"].startswith(x["because"].split(" Each input condition has its own answer")[0])
    assert (f"Read a second time, also accepting the reference classes the also-accept file lists for each map class, "
            f"the answer is yes: its interval is {decide_mod._pc(a['low'])} to {decide_mod._pc(a['high'])}. That is a "
            "second test on the same labels: if the file was fixed before any label was read, the two intervals hold "
            "together at least 90% of the time (a union bound); a file written after the labels were seen can be "
            "fitted to them, and then neither answer has a guarantee.") in y["because"]
    assert ("Each condition's second answer, beside its first, is read the same way: if the file was fixed first, "
            "the two intervals hold together at least 90% of the time (a union bound).") in y["because"]
    said = _run(capsys, ["decide", str(tmp_path / "both.json"), "--ask", q]).splitlines()
    assert said[1] == (f"  also accepting the file's pairs: yes (95% interval {100 * a['low']:.1f}% to "
                       f"{100 * a['high']:.1f}%)")
    for name, c in y["per_condition"].items():
        assert f"  {name}: {c['answer']}" + (f"; also accepting: {c['also_accept']['answer']}" if "also_accept" in c
                                             else "") in said
    plain_said = _run(capsys, ["decide", str(tmp_path / "plain.json"), "--ask", q]).splitlines()
    assert not any("also accepting" in line for line in plain_said)


@pytest.mark.parametrize("design", ["confidence", "tiles"])
def test_decide_words_the_joint_statement_by_each_intervals_method(tmp_path, capsys, design):
    """On tiles, and on the confidence design with every window judged, both intervals are nominal, so the union bound
    gives a nominal 90%, not at least 90%: the joint sentence says the chances add up, the second answer carries its
    exact flag and level, a new sample read both ways is worded as nominal, and the printed line says nominal. The
    review of 13 October 2026 found the sentence claiming at least 90% here."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, design)
    _label(s, truth)
    _run(capsys, ["estimate", s, "--also-accept", str(tmp_path / "pairs.json")])
    path = s[:-4] + "_estimate.json"
    r = _json(path)
    a = r["also_accept"]
    seen = set()
    for t in (round((r["low"] + r["high"]) / 2, 3), 0.5):            # the strict answer undetermined, then yes
        q = f"error_rate_below={t:g}"
        y = decide_mod.decide(path, [q])["answers"][q]
        seen.add(y["answer"])
        assert y["exact"] is False and (y["also_accept"]["exact"], y["also_accept"]["level"]) == (False, 0.95)
        assert ("That is a second test on the same labels: if the file was fixed before any label was read, the "
                "chances that either interval misses add up (a union bound); each is nominal here, so the two hold "
                "together a nominal 90% of the time, not a guaranteed one; a file written after the labels were seen "
                "can be fitted to them, and then neither answer has a guarantee.") in y["because"]
        assert "at least 90%" not in y["because"] and "can reach 20%" not in y["because"]
        four = ("Read both ways, as this one is, a new sample makes four intervals on two samples, and the chances "
                "that any of them misses add up.")
        assert (four in y["because"]) is (y["answer"] == "undetermined")
        said = _run(capsys, ["decide", path, "--ask", q]).splitlines()
        assert said[1] == (f"  also accepting the file's pairs: {y['also_accept']['answer']} (nominal 95% interval "
                           f"{100 * a['low']:.1f}% to {100 * a['high']:.1f}%)")
    assert seen == {"undetermined", "yes"}


def test_decide_carries_the_second_readings_own_warning(tmp_path, capsys):
    """A confidence-design sample whose every error is the accepted pair: the second reading sees no error, so its
    interval is the simple-random Wilson bound and estimate warns of it, while the strict reading has no warning.
    decide's because sentence carries that warning, marked as the second reading's, and the second answer carries it
    as well (the review of 13 October 2026 found it dropped). A warning both readings share is said once."""
    p, truth = _write_map(tmp_path)
    truth = np.where(truth == 3, 3, np.load(p).argmax(axis=0))          # every error is 2 seen as 3
    s = _sample(tmp_path, p, "confidence")
    _label(s, truth)
    _run(capsys, ["estimate", s, "--also-accept", str(tmp_path / "pairs.json")])
    path = s[:-4] + "_estimate.json"
    r = _json(path)
    w = r["also_accept"]["warning"]
    assert "warning" not in r and "no labelled window was wrong, so the design's variance is zero" in w
    y = decide_mod.decide(path, ["error_rate_below=0.1"])["answers"]["error_rate_below=0.1"]
    assert f"Warning (second reading): {w.strip().rstrip('.')}." in y["because"] and y["also_accept"]["warning"] == w
    same = dict(r, warning=w)                                            # the strict reading with the same warning
    y = decide_mod.decide(same, ["error_rate_below=0.1"])["answers"]["error_rate_below=0.1"]
    assert y["because"].count(w.strip().rstrip(".")) == 1 and "(second reading)" not in y["because"]


def test_a_new_sample_read_both_ways_is_four_tests(tmp_path, capsys):
    """Where decide advises a new sample, a result that holds a second reading adds that the new sample read both
    ways makes four tests on two samples, not the two the second-look sentence counts: up to four times delta for
    certificates (capped at 100%), 20% for exact intervals; the audit's answer (1.11.0) says the same. Without the
    option, and where nothing is advised, the sentence is not there (the review of 13 October 2026)."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "random", budget=200)
    _label(s, truth)
    pairs = str(tmp_path / "pairs.json")
    zone = str(tmp_path / "s_zone.json")
    four = ("Read both ways, as this one is, a new sample makes four certificates on two samples: the chance that one "
            "of them is wrong can reach 40.0%, four times delta.")
    for extra, has in ((["--also-accept", pairs], True), ([], False)):
        _run(capsys, ["certify", s, "--alpha", "0.02", "--delta", "0.1", *extra])
        z = _json(zone)
        assert z["n_labelled"] >= z["min_labels_to_certify"] and z["coverage"] is None             # tested, none
        for q in ("trusted_share", "trusted_share_at_least=0.5"):
            y = decide_mod.decide(zone, [q])["answers"][q]
            assert "can reach 20.0%, twice delta" in y["because"] and (four in y["because"]) is has
    assert decide_mod.readings_second_look(0.3).endswith("can reach 100%.")
    # intervals: the random design's are exact
    _run(capsys, ["estimate", s, "--also-accept", pairs])
    r = _json(s[:-4] + "_estimate.json")
    four = ("Read both ways, as this one is, a new sample makes four intervals on two samples: the chance that one of "
            "them misses can reach 20%.")
    for t, has in ((round((r["low"] + r["high"]) / 2, 3), True), (0.9, False)):
        q = f"error_rate_below={t:g}"
        y = decide_mod.decide(r, [q])["answers"][q]
        assert y["answer"] == ("undetermined" if has else "yes") and (four in y["because"]) is has
    # the audit's answer, with and without its second reading
    n = 6
    side = {"design": "audit", "indices": list(range(n)), "map_class": [2] * n, "labels_per_class": n,
            "n_population": 200, "zone": {"rule": "share", "n_zone": 50, "cut": 0.7},
            "audited": {"2": {"n_zone": 50, "n_labelled": n, "census": False}}, "not_audited": {}}
    for also, has in ((None, False), ({"2": [3]}, True)):
        audit = est.class_audit_from_sample(side, np.array([2, 3, 3, 4, 2, 2]), also_accept=also)
        y = decide_mod.decide(audit, ["error_rate_below=0.3"])["answers"]["error_rate_below=0.3"]
        assert y["per_class"]["2"]["answer"] == "undetermined" and (four in y["because"]) is has


def test_per_class_stays_strict_and_reads_the_map_once(tmp_path, capsys, monkeypatch):
    """--per-class with --also-accept: the per-class table is the strict one, as without the option, and the map is
    recomputed once for both."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "random")
    _label(s, truth)
    _run(capsys, ["estimate", s, "--per-class", "--out", str(tmp_path / "plain.json")])
    from oe_inferencex.estimate import estimate_command     # where cmd_estimate looks _map_windows up (ADR-002)
    calls = []
    real = estimate_command._map_windows
    monkeypatch.setattr(estimate_command, "_map_windows", lambda *a, **k: calls.append(a[3]) or real(*a, **k))
    _run(capsys, ["estimate", s, "--per-class", "--out", str(tmp_path / "both.json"), "--also-accept",
                  str(tmp_path / "pairs.json")])
    assert calls == ["estimate --per-class"]
    r = _stripped_is(tmp_path / "plain.json", tmp_path / "both.json")
    assert r["also_accept"]["n_accepted"] > 0


def test_sample_extend_keeps_the_reference_class_column(tmp_path):
    """sample --extend rewrote the CSV with its fixed columns and dropped a reference_class column the reviewer had
    added, so --per-class and --also-accept lost it on an extended sequential sample. It is kept now, after `wrong`,
    blank on the new rows; a CSV without the column is extended as before, byte for byte."""
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "sequential", budget=40)
    fresh = _sample(tmp_path, p, "sequential", budget=90, name="fresh")
    assert main(["sample", p, "--budget", "90", "--design", "sequential", "--patch", "1", "--seed", "1", "--extend", s,
                 "--out", str(tmp_path / "plain.csv")]) == 0
    with open(tmp_path / "plain.csv", "rb") as f, open(fresh, "rb") as g:
        assert f.read() == g.read()                           # no label yet, no reference_class: as a fresh draw
    before = _label(s, truth, unjudged=(5,), upto=30)
    assert main(["sample", p, "--budget", "90", "--design", "sequential", "--patch", "1", "--seed", "1", "--extend", s,
                 "--out", s]) == 0
    after = _rows(s)
    assert list(after[0])[-2:] == ["wrong", "reference_class"] and len(after) == 90
    assert [(r["wrong"], r["reference_class"]) for r in after[:40]] == [(r["wrong"], r["reference_class"])
                                                                        for r in before]
    assert all(r["wrong"] == r["reference_class"] == "" for r in after[40:])
    # and both readings go on from the extended sample
    _label(s, truth, unjudged=(5,), upto=70)
    assert main(["certify", s, "--alpha", "0.2", "--also-accept", str(tmp_path / "pairs.json")]) == 0
    assert _json(tmp_path / "s_zone.json")["also_accept"]["n_accepted"] > 0


def test_every_refusal(tmp_path, capsys):
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "random")
    good = _label(s, truth, unjudged=(4,))
    pairs = str(tmp_path / "pairs.json")
    for command in ("estimate", "certify"):
        base = [command, s] + (["--alpha", "0.2"] if command == "certify" else [])
        # the column: missing, blank, not whole, negative; ? in one column only; `wrong` disagreeing with it
        _rewrite(s, [{k: v for k, v in r.items() if k != "reference_class"} for r in good],
                 [k for k in good[0] if k != "reference_class"])
        with pytest.raises(SystemExit, match=f"^{command}: --also-accept reads the class seen in each window: add a "
                                             r"reference_class column beside wrong \(a whole number of at least 0, in "
                                             r"the map's class codes, or \? where wrong is \?\)$"):
            main([*base, "--also-accept", pairs])
        for v in ("", "1.5", "x", "-1"):
            bad = [dict(r) for r in good]
            bad[2]["reference_class"] = v
            _rewrite(s, bad)
            with pytest.raises(SystemExit, match=r"`reference_class` must be the class seen \(a whole number of at "
                                                 r"least 0, in the map's class codes\) or \? in every row"):
                main([*base, "--also-accept", pairs])
        for row, field, msg in ((2, "wrong", r"row 4 is \? in wrong and not in reference_class"),
                                (4, "reference_class", r"row 6 is \? in wrong and not in reference_class"),
                                (3, "reference_class", r"row 5 is \? in reference_class and not in wrong")):
            bad = [dict(r) for r in good]
            if row == 4:
                bad[4]["reference_class"] = good[0]["map_class"]           # judged in one column, ? in the other
            else:
                bad[row][field] = "?"
            _rewrite(s, bad)
            with pytest.raises(SystemExit, match=f"{command}: {msg}: a window that cannot be judged is"):
                main([*base, "--also-accept", pairs])
        bad = [dict(r) for r in good]
        k = next(i for i, r in enumerate(bad) if r["wrong"] == "0" and i != 4)
        bad[k]["wrong"] = "1"
        _rewrite(s, bad)
        with pytest.raises(SystemExit, match=rf"{command}: on 1 row\(s\) `wrong` disagrees with whether "
                                             rf"`reference_class` differs from the map's class, first at row {k + 2} "
                                             rf"\(wrong 1, reference_class {bad[k]['reference_class']}, map_class "
                                             rf"{bad[k]['map_class']}\)"):
            main([*base, "--also-accept", pairs])
        # an edited map_class is refused by the map, which the class is read from
        bad = [dict(r) for r in good]
        bad[0]["map_class"] = str((int(bad[0]["map_class"]) + 1) % 3)
        _rewrite(s, bad)
        with pytest.raises(SystemExit, match="the map's class at the sampled windows is not the one the CSV records"):
            main([*base, "--also-accept", pairs])
        _rewrite(s, good)
        # the file
        for content, msg in (("{not json", "is not JSON"), ("[2, 3]", "maps class codes"),
                             ('{"x": [3]}', "not a class code"), ('{"2": [3], "02": [4]}', "names class 2 twice"),
                             ('{"2": 3}', "not a list"), ('{"2": [3.5]}', "not a class code")):
            with open(tmp_path / "bad.json", "w") as f:
                f.write(content)
            with pytest.raises(SystemExit, match=f"^{command}: --also-accept .*bad.json.*{msg}"):
                main([*base, "--also-accept", str(tmp_path / "bad.json")])
        with pytest.raises(SystemExit, match=f"^{command}: --also-accept .*missing.json: no such file"):
            main([*base, "--also-accept", str(tmp_path / "missing.json")])
        with pytest.raises(SystemExit, match="would overwrite the input"):
            main([*base, "--also-accept", pairs, "--out", pairs])
    with pytest.raises(SystemExit, match="would overwrite the input"):            # the second mask, as an input
        main(["certify", s, "--alpha", "0.2", "--also-accept", str(tmp_path / "s_zone_also_accept.npy")])
    # estimate: reviewer rates with the option, --alpha off an audit, the map's options without --per-class or it
    for flag in ("--reviewer-false-alarm", "--reviewer-miss"):
        with pytest.raises(SystemExit, match="--also-accept takes no reviewer error rate"):
            main(["estimate", s, flag, "0.05", "--also-accept", pairs])
    with pytest.raises(SystemExit, match="--alpha reads a per-class audit"):
        main(["estimate", s, "--alpha", "0.05", "--also-accept", pairs])
    with pytest.raises(SystemExit, match="read only with --per-class or --also-accept"):
        main(["estimate", s, "--scores", p])
    assert main(["estimate", s, "--scores", p, "--also-accept", pairs]) == 0          # read with the option
    # a refusal leaves nothing behind
    assert not os.path.exists(tmp_path / "s_zone.json") and not os.path.exists(tmp_path / "s_zone_also_accept.npy")
    # the samples refused whole, as before: two maps' differing windows, and an audit (certify)
    np.save(tmp_path / "q.npy", np.roll(np.load(p), 1, axis=2))
    assert main(["sample", p, "--other", str(tmp_path / "q.npy"), "--budget", "20", "--patch", "1",
                 "--out", str(tmp_path / "d.csv")]) == 0
    with pytest.raises(SystemExit, match="takes none of .*--also-accept"):
        main(["estimate", str(tmp_path / "d.csv"), "--also-accept", pairs])
    with pytest.raises(SystemExit, match="this sample holds only windows where two maps differ"):
        main(["certify", str(tmp_path / "d.csv"), "--alpha", "0.1", "--also-accept", pairs])
    assert main(["sample", p, "--design", "audit", "--per-class", "5", "--patch", "1", "--out",
                 str(tmp_path / "a.csv")]) == 0
    with pytest.raises(SystemExit, match="an audit sample draws a fixed number from each class's confident zone"):
        main(["certify", str(tmp_path / "a.csv"), "--alpha", "0.1", "--also-accept", pairs])


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed")
def test_the_server_refuses_options_on_a_two_map_sample_in_its_own_words(tmp_path):
    """The command's refusal on a two-map sample names --alpha and --also-accept, which the server does not offer; the
    server refuses first, naming only parameters it has (the review of 13 October 2026)."""
    from oe_inferencex import mcp_server
    p, _ = _write_map(tmp_path)
    np.save(tmp_path / "q.npy", np.roll(np.load(p), 1, axis=2))
    d = str(tmp_path / "d.csv")
    assert main(["sample", p, "--other", str(tmp_path / "q.npy"), "--budget", "20", "--patch", "1", "--out", d]) == 0
    for given in ({"per_class": True}, {"scores": p}, {"confidence": p}, {"nodata": 0.0},
                  {"reviewer_false_alarm": 0.05}, {"reviewer_miss": 0.05}):
        with pytest.raises(mcp_server.Refused) as e:
            mcp_server.estimate(sample_csv=d, **given)
        said = mcp_server._mcp_words(str(e.value))
        assert str(e.value) == mcp_server.ESTIMATE_TWO_MAPS_OPTIONS
        assert "--" not in said and "alpha" not in said and "also" not in said
        assert not os.path.exists(str(tmp_path / "d_estimate.json"))


def test_the_help_and_the_docs_name_the_option():
    import argparse
    sub = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    for command, said in (("estimate", cli.ALSO_ACCEPT_HELP), ("certify", cli.CERTIFY_ALSO_ACCEPT_HELP)):
        helps = {a.dest: a.help for a in sub.choices[command]._actions}
        assert helps["also_accept"] == said and "written before any label is read" in said
    # certify refuses an audit sample, so its help does not point to one (the review of 13 October 2026)
    assert "audit" in cli.ALSO_ACCEPT_HELP and "audit" not in cli.CERTIFY_ALSO_ACCEPT_HELP
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    usage = " ".join(open(os.path.join(root, "docs", "Usage.md"), encoding="utf-8").read().split())
    assert usage.count("**A second reading.**") == 3                # audit, sample and estimate, certify
    assert "at least 90% of the time" in usage and "twice δ" in usage


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed")
def test_the_server_takes_no_such_option_and_decide_refuses_such_a_result(tmp_path, capsys):
    from oe_inferencex import mcp_server
    for tool in (mcp_server.estimate, mcp_server.certify):
        assert not {"also_accept", "also_accept_file"} & set(inspect.signature(tool).parameters)
    p, truth = _write_map(tmp_path)
    s = _sample(tmp_path, p, "random")
    _label(s, truth)
    for command, extra, result in (("estimate", [], "s_estimate.json"), ("certify", ["--alpha", "0.2"], "s_zone.json")):
        _run(capsys, [command, s, *extra, "--also-accept", str(tmp_path / "pairs.json")])
        path = str(tmp_path / result)
        with pytest.raises(mcp_server.Refused, match="holds a second reading of the same labels") as e:
            mcp_server.decide(result_json=path, questions=["trusted_share"] if command == "certify"
                              else ["error_rate_below=0.1"])
        assert f"oe-inferencex decide {path}" in str(e.value) and mcp_server._mcp_words(str(e.value)) == str(e.value)
        assert not os.path.exists(path[:-5] + "_decisions.json")
    # without the option the server reads the result as before
    _run(capsys, ["estimate", s])
    assert mcp_server.decide(result_json=str(tmp_path / "s_estimate.json"), questions=["error_rate_below=0.5"])
    # the map's options without per_class: refused in the server's words, naming no option it lacks (the command's
    # refusal now names --also-accept, which no parameter of this server matches)
    for given in ({"scores": p}, {"confidence": p}, {"nodata": 0.0}):
        with pytest.raises(mcp_server.Refused) as e:
            mcp_server.estimate(sample_csv=s, **given)
        assert str(e.value) == mcp_server.ESTIMATE_MAP_OPTIONS == ("estimate: scores, confidence and nodata are read "
                                                                   "only with per_class; the error rate comes from the "
                                                                   "CSV and its sidecar alone")
        assert "--" not in mcp_server._mcp_words(str(e.value))
    assert mcp_server.estimate(sample_csv=s, per_class=True, scores=p)          # read with per_class, as before

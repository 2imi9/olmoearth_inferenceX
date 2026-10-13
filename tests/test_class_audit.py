"""A per-class audit of a map's confident calls (sample --design audit, then estimate; 13 October 2026).

The confident zone is fixed from the map before any label; each predicted class's windows in it are a fixed finite
population, and its labels a simple random sample of fixed size, so the exact hypergeometric interval covers that
class's zone error rate at least 95% of the time on its own. These tests check the zone and the draw, the coverage by
exact enumeration (with an adversary that marks ? after seeing the truth), the alternative reading and its note, the
verdict against decide's rule, the arithmetic of what N labels buy as docs/Usage.md quotes it, the route through the
command line, and every refusal.
"""
import csv
import importlib.util
import inspect
import itertools
import json
import math
import os
import re

import numpy as np
import pytest

from oe_inferencex import decide as decide_mod
from oe_inferencex import estimate as est
from oe_inferencex.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- the zone and the draw
@pytest.mark.parametrize("N", [30, 31, 32])
def test_the_zone_is_the_top_share_with_ties_and_is_fixed_before_labels(N):
    rng = np.random.default_rng(N)
    margin = rng.permutation(np.arange(N, dtype=float))                  # distinct confidences 0 .. N - 1
    zone, block = est._audit_zone(margin, np.ones(N, bool))
    m = math.ceil(N / 3)                                                  # exact at N = 3k, 3k + 1 and 3k + 2
    assert block["rule"] == "share" and block["n_top"] == m == zone.size == block["n_zone"]
    assert block["cut"] == float(N - m) and set(zone.tolist()) == set(np.flatnonzero(margin >= N - m).tolist())
    assert block["share_of_map"] == m / N and block["definition"] == est.AUDIT_ZONE_TEXT
    # windows tied at the cut are in the zone, so it can hold more than ceil(S N)
    tied = np.where(margin >= N - m - 2, float(N - m), margin)
    zone, block = est._audit_zone(tied, np.ones(N, bool))
    assert block["n_top"] == m and block["n_zone"] == m + 2 and block["cut"] == float(N - m)
    # an explicit cut is every valid window at that confidence or above
    zone, block = est._audit_zone(margin, np.ones(N, bool), cut=N - 5.5)
    assert block["rule"] == "cut" and block["share"] is None and block["n_top"] is None
    assert sorted(zone.tolist()) == sorted(np.flatnonzero(margin >= N - 5.5).tolist())
    # windows that are not valid, or have no finite confidence, are not in the population
    valid = np.ones(N, bool)
    valid[:4] = False
    margin2 = margin.copy()
    margin2[4] = np.nan
    zone, block = est._audit_zone(margin2, valid)
    assert not set(zone.tolist()) & {0, 1, 2, 3, 4} and block["n_top"] == math.ceil((N - 5) / 3)


def test_the_share_is_read_as_written():
    """0.3 of 10 windows is 3; in floats 0.3 * 10 is 3.0000000000000004, and its ceiling 4."""
    zone, block = est._audit_zone(np.arange(10.0), np.ones(10, bool), share=0.3)
    assert block["n_top"] == 3 and zone.tolist() == [7, 8, 9]
    zone, block = est._audit_zone(np.arange(10.0), np.ones(10, bool), share=1)
    assert block["n_zone"] == 10 and block["cut"] == 0.0
    zone, block = est._audit_zone(np.arange(10.0), np.ones(10, bool), share=0.01)
    assert block["n_top"] == 1 and zone.tolist() == [9]


def _toy_map(seed=0):
    """120 windows: classes 0, 1 and 2 across the confidences, class 3 on two very confident windows, class 4 never
    confident."""
    rng = np.random.default_rng(seed)
    klass = rng.integers(0, 3, 120)
    margin = rng.uniform(0.2, 0.9, 120)
    klass[[5, 50]], margin[[5, 50]] = 3, [0.99, 0.98]
    klass[[7, 8, 9]], margin[[7, 8, 9]] = 4, [0.01, 0.02, 0.03]
    return margin, klass


def test_each_class_gets_n_or_its_whole_zone_and_rows_are_interleaved():
    margin, klass = _toy_map()
    s = est.sample_class_audit(margin, klass, 6, seed=4)
    zone, block = est._audit_zone(margin, np.ones(margin.size, bool))
    assert s["zone"] == block and s["n_population"] == 120 and s["labels_per_class"] == 6
    idx = s["indices"]
    assert np.unique(idx).size == idx.size and set(idx.tolist()) <= set(zone.tolist())
    assert s["map_class"].tolist() == klass[idx].tolist()
    for c, e in s["audited"].items():
        in_zone = int((klass[zone] == int(c)).sum())
        assert e["n_zone"] == in_zone and e["n_labelled"] == min(6, in_zone) == int((klass[idx] == int(c)).sum())
        assert e["census"] == (in_zone <= 6)
    assert s["audited"]["3"] == {"n_zone": 2, "n_labelled": 2, "census": True}
    assert s["not_audited"] == {"4": {"n_windows": 3}}
    # the per-class draws, then one random order over them: no runs of one class for a reviewer labelling blind
    rng = np.random.default_rng(4)
    codes, sizes = np.unique(klass[zone], return_counts=True)
    local = est.draw_stratified(rng, np.searchsorted(codes, klass[zone]), sizes, np.minimum(sizes, 6))
    drawn = zone[local]
    assert idx.tolist() == drawn[rng.permutation(drawn.size)].tolist()
    assert (np.diff(s["map_class"]) < 0).any()
    # the same seed draws the same windows; the sampler takes no label, only the map and the seed
    assert est.sample_class_audit(margin, klass, 6, seed=4)["indices"].tolist() == idx.tolist()
    assert est.sample_class_audit(margin, klass, 6, seed=5)["indices"].tolist() != idx.tolist()
    assert set(inspect.signature(est.sample_class_audit).parameters) == {"margin", "map_class", "per_class", "valid",
                                                                         "share", "cut", "seed"}


# ----------------------------------------------------------------------------- exact coverage
def _sample(sizes, n_per, labelled):
    """An audit sample of two classes, 0 and 1, whose zones hold sizes[c] windows (indices 0.. for class 0, then class
    1's), with the windows `labelled[c]` (positions within the class's zone) drawn."""
    idx, mc = [], []
    for c, (Nc, picked) in enumerate(zip(sizes, labelled)):
        start = sum(sizes[:c])
        idx += [start + p for p in picked]
        mc += [c] * len(picked)
    return {"design": "audit", "indices": idx, "map_class": mc, "labels_per_class": n_per,
            "n_population": 3 * sum(sizes), "zone": {"rule": "share", "n_zone": sum(sizes), "cut": 0.5},
            "audited": {str(c): {"n_zone": Nc, "n_labelled": min(n_per, Nc), "census": n_per >= Nc}
                        for c, Nc in enumerate(sizes)}, "not_audited": {}}


@pytest.mark.parametrize("c", [0, 1])
def test_each_class_interval_is_exact_on_a_small_population(c):
    """Zones of 7 and 5 windows, 3 labels each: for every count K_c of wrong windows and every 3-subset of the class's
    zone, the class's interval covers K_c / N_c with probability exact_coverage_srs gives, at least 95%; and still at
    least 95% when an adversary, after seeing each drawn window's truth, marks any of them ?."""
    sizes, n = (7, 5), 3
    Nc = sizes[c]
    other = list(range(n))                                     # the other class's draw, fixed
    for K in range(Nc + 1):
        truth = np.r_[np.full(K, 9), np.full(Nc - K, c)]       # the first K windows of the zone are wrong
        subsets = list(itertools.combinations(range(Nc), n))
        covered = adversary = 0
        for sub in subsets:
            labelled = [list(sub), other] if c == 0 else [other, list(sub)]
            s = _sample(sizes, n, labelled)
            ref = np.array([truth[p] if mc == c else mc for p, mc in zip([*labelled[0], *labelled[1]], s["map_class"])])
            row = est.class_audit_from_sample(s, ref)["per_class"][str(c)]
            covered += row["low"] - 1e-12 <= K / Nc <= row["high"] + 1e-12
            rows = [i for i, mc in enumerate(s["map_class"]) if mc == c]
            ok = True
            for mark in itertools.product([False, True], repeat=n):
                unj = np.zeros(len(ref), bool)
                unj[[r for r, m in zip(rows, mark) if m]] = True
                r2 = est.class_audit_from_sample(s, ref, unjudged=unj)["per_class"][str(c)]
                ok &= r2["low"] - 1e-12 <= K / Nc <= r2["high"] + 1e-12
            adversary += ok
        exact = est.exact_coverage_srs(Nc, K, n, est.hypergeom_interval)
        assert covered / len(subsets) == pytest.approx(exact, abs=1e-12) and exact >= 0.95, (K, exact)
        assert adversary / len(subsets) >= 0.95, K


# ----------------------------------------------------------------------------- the alternative reading
def _one_class(refs, Nc=50, c=2, n_per=None):
    n = len(refs)
    s = {"design": "audit", "indices": list(range(n)), "map_class": [c] * n, "labels_per_class": n_per or n,
         "n_population": 4 * Nc, "zone": {"rule": "share", "n_zone": Nc, "cut": 0.7},
         "audited": {str(c): {"n_zone": Nc, "n_labelled": n, "census": n == Nc}}, "not_audited": {}}
    unj = np.array([r == "?" for r in refs])
    return s, np.array([-1 if r == "?" else r for r in refs]), unj


def test_the_alternative_reading_counts_only_unaccepted_classes():
    s, ref, unj = _one_class([2, 3, 3, 4, "?", 2])
    r = est.class_audit_from_sample(s, ref, unjudged=unj, also_accept={"2": [3, 2], "7": [1]})
    row = r["per_class"]["2"]
    assert (row["n_wrong"], row["n_unjudged"], row["errors_by_reference"]) == (3, 1, {"3": 2, "4": 1})
    assert row["estimate"] is None and row["estimate_range"] == [3 / 6, 4 / 6]
    assert row["low"] == est.hypergeom_interval(3, 6, 50)[0] and row["high"] == est.hypergeom_interval(4, 6, 50)[1]
    alt = row["also_accept"]
    assert alt["accepted"] == [3] and (alt["n_wrong"], alt["errors_by_reference"]) == (1, {"4": 1})
    assert alt["estimate_range"] == [1 / 6, 2 / 6]                      # the ? window in neither reading's count
    assert alt["low"] == est.hypergeom_interval(1, 6, 50)[0] and alt["high"] == est.hypergeom_interval(2, 6, 50)[1]
    assert r["also_accept"] == {"source": None, "pairs": {"2": [2, 3], "7": [1]}}
    assert r["also_accept_unused"] == ["7"] and r["reading_note"] == est.AUDIT_READING_NOTE
    assert "verdict" not in row and "note" not in row                   # no alpha: no verdict, no note
    # a class listed for itself changes nothing
    same = est.class_audit_from_sample(s, ref, unjudged=unj, also_accept={2: [2]})["per_class"]["2"]
    assert same["also_accept"]["accepted"] == []
    assert {k: same["also_accept"][k] for k in ("n_wrong", "low", "high")} == {k: same[k] for k in ("n_wrong", "low",
                                                                                                     "high")}
    # without also_accept there is no second reading at all
    plain = est.class_audit_from_sample(s, ref, unjudged=unj)
    assert "also_accept" not in plain and "also_accept" not in plain["per_class"]["2"]


def test_the_alternative_interval_is_exact_for_its_own_error():
    """A zone of 7 windows of class 2 holding right (2), accepted (3) and other (4) windows in every mix: the
    alternative reading's interval, from 3 labels, covers the share of `other` windows as exact_coverage_srs says."""
    Nc, n = 7, 3
    for n_acc in range(Nc + 1):
        for n_oth in range(Nc - n_acc + 1):
            truth = np.r_[np.full(n_acc, 3), np.full(n_oth, 4), np.full(Nc - n_acc - n_oth, 2)]
            subsets = list(itertools.combinations(range(Nc), n))
            hit = 0
            for sub in subsets:
                s, ref, unj = _one_class(list(truth[list(sub)]), Nc=Nc, n_per=n)
                alt = est.class_audit_from_sample(s, ref, also_accept={"2": [3]})["per_class"]["2"]["also_accept"]
                hit += alt["low"] - 1e-12 <= n_oth / Nc <= alt["high"] + 1e-12
            exact = est.exact_coverage_srs(Nc, n_oth, n, est.hypergeom_interval)
            assert hit / len(subsets) == pytest.approx(exact, abs=1e-12) and exact >= 0.95


def test_a_crosswalk_fitted_to_the_labels_has_no_guarantee():
    """Why the second reading's guarantee is conditional on the crosswalk being fixed first (the review of 13 October
    2026): a zone of 1,000 windows of class 2, half of them wrong, each on its own reference class; after 15 labels, a
    crosswalk listing the classes the errors fell on. The alternative interval then misses its own error rate on every
    draw, and the legend note stands on what are model errors."""
    rng = np.random.default_rng(0)
    Nc = 1000
    truth = np.full(Nc, 2)
    truth[rng.choice(Nc, 500, replace=False)] = 100 + np.arange(500)
    missed = noted = 0
    for _ in range(200):
        ref = truth[rng.choice(Nc, 15, replace=False)]
        fitted = {"2": sorted({int(v) for v in ref if v != 2})}
        s, ref, _ = _one_class(list(ref), Nc=Nc)
        row = est.class_audit_from_sample(s, ref, also_accept=fitted, alpha=0.05)["per_class"]["2"]
        alt = row["also_accept"]
        missed += not alt["low"] <= (500 - len(fitted["2"])) / Nc <= alt["high"]
        noted += "note" in row
    assert missed == 200 and noted >= 180
    assert "If the file was fixed before any label was read" in est.AUDIT_READING_NOTE
    assert "written after the labels were seen can be fitted to them, and then neither holds" in est.AUDIT_READING_NOTE


def test_the_legend_note_appears_only_when_strict_is_above_and_the_alternative_is_not():
    cases = [  # (references, zone, strict verdict, alternative verdict, note)
        ([3] * 8 + [2] * 7, 1000, "above", "undetermined", True),
        ([3] * 4 + [4] * 6 + [2] * 5, 1000, "above", "above", False),
        ([3] + [2] * 14, 1000, "undetermined", "undetermined", False),
        ([2] * 100, 100, "below", "below", False),                      # a census with no error: its exact rate, 0
        ([3, 3, 2, 2], 4, "above", "below", True),                       # a census: 50% strict, 0% accepting 3
    ]
    for refs, Nc, v_strict, v_alt, noted in cases:
        s, ref, unj = _one_class(refs, Nc=Nc)
        row = est.class_audit_from_sample(s, ref, also_accept={"2": [3]}, alpha=0.05)["per_class"]["2"]
        assert (row["verdict"], row["also_accept"]["verdict"], "note" in row) == (v_strict, v_alt, noted), refs
        if noted:
            k = refs.count(3)
            assert f"{k} of its {k} errors are on reference classes the alternative accepts (3: {k})" in row["note"]
            assert "wrong at least 5% of the time" in row["note"]
            assert "a legend difference, not necessarily a model error" in row["note"]
    assert est.class_audit_from_sample(*_one_class([3, 3, 2, 2], Nc=4)[:2])["per_class"]["2"]["census"] is True


def test_the_verdict_is_decides_error_rate_below():
    """At T = alpha, decide's answer for each class is estimate's verdict (below is yes, at least alpha is no); at any
    T it is interval_verdict's rule, for both readings."""
    refs = [[3] * 8 + [2] * 7, [3] * 4 + [4] * 6 + [2] * 5, [3] + [2] * 14, [2] * 15]
    s = {"design": "audit", "indices": list(range(60)), "map_class": [c for c in range(4) for _ in range(15)],
         "labels_per_class": 15, "n_population": 9000, "zone": {"rule": "share", "n_zone": 3000, "cut": 0.8},
         "audited": {str(c): {"n_zone": 750, "n_labelled": 15, "census": False} for c in range(4)}, "not_audited": {}}
    ref = np.array([c if v == 2 else v + 10 for c, rs in enumerate(refs) for v in rs])
    accept = {str(c): [13] for c in range(4)}
    word = {"below": "yes", "above": "no", "undetermined": "undetermined"}
    noted = 0
    for alpha in (0.01, 0.05, 0.2, 0.5):
        r = est.class_audit_from_sample(s, ref, also_accept=accept, alpha=alpha)
        noted += sum("note" in row for row in r["per_class"].values())
        a = decide_mod.decide(r, [f"error_rate_below={alpha}"])["answers"][f"error_rate_below={alpha:g}"]
        assert a["answer"] == "per class" and a["exact"] is True and decide_mod.SEPARATE in a["because"]
        for c, row in r["per_class"].items():
            assert a["per_class"][c]["answer"] == word[row["verdict"]]
            assert row["verdict"] == est.interval_verdict(row["low"], row["high"], alpha)
            assert a["per_class"][c]["also_accept"]["answer"] == word[row["also_accept"]["verdict"]]
            # the result's caveats survive into the answer an agent quotes (the review of 13 October 2026): each
            # class's legend note, the selection clause, and the crosswalk's condition on the second reading
            assert a["per_class"][c].get("note") == row.get("note")
            if "note" in row:
                assert decide_mod._as_sentence(row["note"]) in a["because"]
        assert "picked out because its rate looks highest is not covered at 95%" in a["because"]
        assert "a second test on them. If the file was fixed before any label was read" in a["because"]
        assert "written after the labels were seen can be fitted to them" in a["because"]
        assert a["evidence"]["note"] == est.AUDIT_NOTE and a["evidence"]["reading_note"] == est.AUDIT_READING_NOTE
    assert noted                                    # class 0, 8 of 15 wrong, all on the accepted class 13
    assert decide_mod.available(r) == ["error_rate_below"] and decide_mod.kind(r) == "class_audit"


# ----------------------------------------------------------------------------- what N labels buy
def _hyper_pmf(k, n, N, K):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def _usage_audit():
    usage = open(os.path.join(ROOT, "docs", "Usage.md"), encoding="utf-8").read()
    return usage[usage.index("### audit"):usage.index("### certify")]


def test_what_n_labels_buy():
    """The table in Usage, recomputed: on a zone of 6,000 windows, the errors among N labels from which the exact
    interval's lower end reaches alpha, and the probability of `above` at four zone error rates; and the error-free
    labels `below` needs."""
    Z = 6000
    rows = re.findall(r"^\| (0\.\d+) \| (\d+) \| (\d+) \| ([-\d. /]+) \|$", _usage_audit(), re.M)
    assert len(rows) == 6
    for alpha, n, k_min, probs in rows:
        alpha, n, k_min = float(alpha), int(n), int(k_min)
        above = [k for k in range(n + 1) if est.hypergeom_interval(k, n, Z)[0] >= alpha]
        assert above[0] == k_min and above == list(range(k_min, n + 1))
        for rate, said in zip((0.05, 0.10, 0.20, 0.30), probs.split(" / ")):
            p = sum(_hyper_pmf(k, n, Z, round(rate * Z)) for k in above)
            if said == "-":                                   # a rate below alpha: `above` would be wrong
                assert rate < alpha and p < 0.0015
            else:
                assert float(said) == round(p, 3), (alpha, n, rate, p)
    first_below = {a: next(n for n in range(1, 200) if est.hypergeom_interval(0, n, Z)[1] < a) for a in (0.05, 0.10)}
    assert first_below == {0.05: 72, 0.10: 35}
    assert next(n for n in range(1, 200) if est.hypergeom_interval(0, n, 10 ** 6)[1] < 0.10) == 36
    assert "72 labels with no error" in _usage_audit() and "35 or 36" in _usage_audit()


def test_a_small_zone_can_be_cleared_without_a_census():
    """Usage's second table, recomputed: n error-free labels read `below` on every zone of n windows (labelled in full)
    up to the first column's size, and on no zone larger than the second's; between them on some zones and not others,
    since the interval's upper end moves in steps of one window of the zone. The review of 13 October 2026 found the
    text said only a zone labelled in full could be cleared."""
    rows = re.findall(r"^\| (\d+) \| (0\.\d+) \| (\d+)( \(in full\))? \| (\d+) \|$", _usage_audit(), re.M)
    assert len(rows) == 6
    for n, a, every, full, none in rows:
        n, a, every, none = int(n), float(a), int(every), int(none)
        cleared = [Z for Z in range(n, 1000) if est.interval_verdict(*est.hypergeom_interval(0, n, Z), a) == "below"]
        assert cleared[0] == n                                                  # a census always can
        assert every == next(Z for Z in range(n, 1001) if Z not in cleared) - 1 and none == cleared[-1], (n, a)
        assert bool(full) == (every == n)
    assert [(int(n), float(a)) for n, a, *_ in rows] == [(n, a) for n in (15, 20, 30) for a in (0.05, 0.10)]
    said = " ".join(_usage_audit().split())
    assert "On a zone this large, an audit of 15 to 30 labels per class can therefore find a bad class but cannot" in said


# ----------------------------------------------------------------------------- the command line
def _write_map(tmp_path):
    """Five classes on a 1 x 300 strip of probability vectors: classes 0, 1 and 2 at confidences across the range, class
    3 very confident on two windows only, class 4 never confident. The truth: class 2's errors are reference class 3
    (the contested pair), class 1's are class 0."""
    rng = np.random.default_rng(11)
    n, C = 300, 5
    pred = rng.integers(0, 3, n)
    top = rng.uniform(0.40, 0.95, n)
    pred[[10, 200]], top[[10, 200]] = 3, [0.97, 0.98]
    four = rng.choice(np.flatnonzero(pred < 3), 12, replace=False)
    pred[four], top[four] = 4, 0.30
    p = np.repeat(((1 - top) / (C - 1))[None, :], C, axis=0)
    p[pred, np.arange(n)] = top
    np.save(tmp_path / "p.npy", p[:, None, :])
    truth = pred.copy()
    flip = rng.random(n)
    truth[(pred == 2) & (flip < 0.45)] = 3
    truth[(pred == 1) & (flip < 0.15)] = 0
    return pred, top, truth


def _fill(path, truth, unjudged_rows=(), field="reference_class"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r[field] = "?" if i in unjudged_rows else str(int(truth[int(r["pixel_col"])]))
    _rewrite(path, rows)
    return rows


def _rewrite(path, rows, fields=None):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_sample_then_estimate_end_to_end(tmp_path, capsys):
    pred, top, truth = _write_map(tmp_path)
    out = tmp_path / "audit.csv"
    assert main(["sample", str(tmp_path / "p.npy"), "--design", "audit", "--per-class", "10", "--patch", "1",
                 "--seed", "2", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    side = json.load(open(tmp_path / "audit.json"))
    assert side["design"] == "audit" and side["labels_per_class"] == 10 and side["n_population"] == 300
    assert side["zone"]["rule"] == "share" and side["zone"]["n_top"] == 100 == side["zone"]["n_zone"]
    assert side["audited"]["3"] == {"n_zone": 2, "n_labelled": 2, "census": True}
    assert side["not_audited"] == {"4": {"n_windows": 12}} and "budget" not in side
    assert side["map_class"] == [int(pred[i]) for i in side["indices"]] and len(side["indices"]) == 32
    assert side["scores"] == str(tmp_path / "p.npy") and side["patch"] == 1 and "reference_class" in side["how_to_label"]
    assert printed.startswith("32 windows to label: up to 10 from each of the 4 predicted classes' confident zones "
                              "(labelled in full: class 3, 2 windows); the zone is the 100 valid windows at confidence")
    assert "(the top third)" in printed and "--alpha A, with A chosen now, before any label is read" in printed
    assert "write any --also-accept file now too" in printed
    assert "note: class 4 has no window in the zone (12 windows of the map); the audit says nothing about it" in printed
    with open(out, newline="") as f:
        header = next(csv.reader(f))
    assert header == ["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "confidence",
                      "map_class", "reference_class"]
    with open(out, newline="") as f:
        unjudged = [i for i, row in enumerate(csv.DictReader(f)) if row["map_class"] == "1"][:2]
    _fill(out, truth, unjudged_rows=unjudged)
    with open(tmp_path / "pairs.json", "w") as f:
        json.dump({"2": [3]}, f)
    assert main(["estimate", str(out), "--alpha", "0.1", "--also-accept", str(tmp_path / "pairs.json")]) == 0
    printed = capsys.readouterr().out.splitlines()
    r = json.load(open(tmp_path / "audit_estimate.json"))
    assert printed[0] == ("confident zone: the 100 windows at confidence "
                          f"{r['zone']['cut']:.6g} or above, 33.3% of the 300 valid windows (the top third); 10 labels "
                          "per predicted class")
    assert r["design"] == "audit" and r["n_labelled"] == 32 and r["n_unjudged"] == 2 and r["alpha"] == 0.1
    assert set(r) >= {"zone", "labels_per_class", "nominal_coverage", "method", "note", "per_class", "not_audited",
                      "also_accept", "also_accept_unused", "reading_note", "sample"}
    assert r["also_accept"]["source"] == str(tmp_path / "pairs.json") and r["also_accept_unused"] == []
    assert list(r["per_class"]) == ["0", "1", "2", "3"]
    zone = np.flatnonzero(top >= r["zone"]["cut"])
    for c, row in r["per_class"].items():
        in_zone = zone[pred[zone] == int(c)]
        true_rate = float((truth[in_zone] != int(c)).mean())
        assert row["low"] <= true_rate <= row["high"], c
        assert row["n_zone"] == in_zone.size and row["share_of_zone"] == in_zone.size / 100
    assert r["per_class"]["3"]["census"] and r["per_class"]["3"]["low"] == r["per_class"]["3"]["high"]
    assert any(line.startswith("class 3: labelled in full, 0 wrong of 2 (0.0%), its exact rate") for line in printed)
    assert any(line.startswith("  also accepting 3: ") for line in printed)
    assert "not audited: class 4 (no window reaches the cut; 12 windows of the map)" in printed
    assert f"note: {est.AUDIT_NOTE}" in printed and f"note: {est.AUDIT_READING_NOTE}" in printed
    one = r["per_class"]["1"]                                            # its two ? rows: a range, bounded both ways
    assert one["n_unjudged"] == 2 and one["estimate"] is None and len(one["estimate_range"]) == 2
    assert any(line.startswith(f"class 1: {one['n_wrong']} wrong and 2 ? of 10 labelled") for line in printed)
    strict2 = r["per_class"]["2"]
    assert strict2["verdict"] == "above" and set(strict2["errors_by_reference"]) == {"3"}
    assert strict2["also_accept"]["n_wrong"] == 0 and "note" in strict2
    assert any(line.startswith("  note: under the strict reading class 2's confident zone") for line in printed)
    # decide reads the result, with the alternative beside each class that has one
    assert main(["decide", str(tmp_path / "audit_estimate.json"), "--ask", "error_rate_below=0.1"]) == 0
    said = capsys.readouterr().out.splitlines()
    assert said[0].startswith("error_rate_below=0.1: per class. Error rate of each class's confident zone below 10.0%")
    line2 = next(x for x in said if x.startswith("  class 2: "))
    assert line2.startswith("  class 2: no (95% interval") and "; also accepting 3: " in line2
    assert not any("also accepting" in x for x in said if x.startswith("  class 0: "))
    # class 2's legend note reaches what decide prints and writes
    assert "Under the strict reading class 2's confident zone is wrong at least 10% of the time" in said[0]
    decided = json.load(open(tmp_path / "audit_estimate_decisions.json"))["answers"]["error_rate_below=0.1"]
    assert decided["per_class"]["2"]["note"] == strict2["note"] and "note" not in decided["per_class"]["0"]
    # without --alpha: intervals, no verdicts, and a note that says so
    assert main(["estimate", str(out), "--out", str(tmp_path / "plain.json")]) == 0
    plain = capsys.readouterr().out
    assert "no class gets a verdict" in plain and "at least 10%" not in plain
    assert all("verdict" not in row for row in json.load(open(tmp_path / "plain.json"))["per_class"].values())


def test_a_product_map_keeps_its_codes(tmp_path, capsys):
    """A class map with its confidence band and codes 1, 5 and 7: the audit's classes are the product's codes, and
    a range that leaves pixels out is named."""
    rng = np.random.default_rng(3)
    classes = rng.choice([1, 5, 7], size=(1, 240))
    band = rng.integers(1, 101, size=(1, 240)).astype(float)
    band[0, :10] = 200                                                    # a provenance code, outside the range
    np.save(tmp_path / "classes.npy", classes)
    np.save(tmp_path / "band.npy", band)
    out = tmp_path / "a.csv"
    assert main(["sample", str(tmp_path / "classes.npy"), "--confidence", str(tmp_path / "band.npy"),
                 "--confidence-range", "1", "100", "--design", "audit", "--per-class", "5", "--patch", "1",
                 "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "a.json"))
    assert set(side["audited"]) == {"1", "5", "7"} and side["n_population"] == 230
    assert side["confidence"] == str(tmp_path / "band.npy") and side["confidence_range"] == [1.0, 100.0]
    # a product's sidecar holds the class map's path under `classes`; the audit's own block is `audited`
    assert side["classes"] == str(tmp_path / "classes.npy")
    assert set(side["map_class"]) <= {1, 5, 7}
    _fill(out, classes[0])
    capsys.readouterr()
    assert main(["estimate", str(out), "--alpha", "0.05"]) == 0
    printed = capsys.readouterr().out
    r = json.load(open(tmp_path / "a_estimate.json"))
    assert list(r["per_class"]) == ["1", "5", "7"] and all(row["n_wrong"] == 0 for row in r["per_class"].values())
    assert "10 of 240 classified pixels" in r["population_note"] and "note: The population is" in printed


@pytest.mark.parametrize("share,top_words", [(None, "the top third"), ("0.1", "the top 10%")])
def test_ties_at_the_cut_are_named_when_they_enlarge_the_zone(tmp_path, capsys, share, top_words):
    """A banded confidence (50, 80 or 100 on each pixel, 100 on most): the windows tied at the cut are in the zone, so
    it is far larger than the share asked for, and sample and estimate say how many the ties added rather than call it
    the share alone (the review of 13 October 2026)."""
    rng = np.random.default_rng(5)
    classes = rng.choice([1, 5, 7], size=(1, 300))
    band = rng.choice([50.0, 80.0, 100.0], size=(1, 300), p=[0.1, 0.1, 0.8])
    np.save(tmp_path / "classes.npy", classes)
    np.save(tmp_path / "band.npy", band)
    out = tmp_path / "a.csv"
    assert main(["sample", str(tmp_path / "classes.npy"), "--confidence", str(tmp_path / "band.npy"), "--design",
                 "audit", "--per-class", "5", "--patch", "1", "--out", str(out),
                 *(["--zone-share", share] if share else [])]) == 0
    printed = capsys.readouterr().out
    z = json.load(open(tmp_path / "a.json"))["zone"]
    n_top = math.ceil(300 * (0.1 if share else 1) / (1 if share else 3))
    assert z["cut"] == 100.0 and z["n_top"] == n_top and z["n_zone"] == int((band >= 100).sum()) > n_top
    words = f"({top_words} by rank, {n_top} windows, and {z['n_zone'] - n_top} more tied at the cut)"
    assert f"{100 * z['n_zone'] / 300:.1f}% of the 300 {words}" in printed and f"({top_words})" not in printed
    _fill(out, classes[0])
    assert main(["estimate", str(out)]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.endswith(f"of the 300 valid windows {words}; 5 labels per predicted class")


# ----------------------------------------------------------------------------- refusals
def test_the_estimator_and_the_sampler_check_their_inputs():
    margin, klass = _toy_map()
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="per_class"):
            est.sample_class_audit(margin, klass, bad)
    with pytest.raises(ValueError, match="entries for"):
        est.sample_class_audit(margin, klass[:-1], 5)
    with pytest.raises(ValueError, match="not both"):
        est.sample_class_audit(margin, klass, 5, share=0.5, cut=0.5)
    for share in (0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="share must be in"):
            est.sample_class_audit(margin, klass, 5, share=share)
    with pytest.raises(ValueError, match="finite"):
        est.sample_class_audit(margin, klass, 5, cut=float("inf"))
    with pytest.raises(ValueError, match="no valid window reaches confidence 2; the most confident has 0.99"):
        est.sample_class_audit(margin, klass, 5, cut=2)
    with pytest.raises(ValueError, match="class code"):
        est.sample_class_audit(margin, np.where(np.arange(120) == 3, -1, klass), 5)
    s, ref, unj = _one_class([2, 3, 2])
    with pytest.raises(ValueError, match="3 sampled windows but"):
        est.class_audit_from_sample(s, ref[:2])
    with pytest.raises(ValueError, match="repeats or is negative"):
        est.class_audit_from_sample(dict(s, indices=[0, 0, 1]), ref)
    with pytest.raises(ValueError, match="does not audit"):
        est.class_audit_from_sample(dict(s, map_class=[2, 2, 9]), ref)
    with pytest.raises(ValueError, match="labelled rows"):
        est.class_audit_from_sample(dict(s, labels_per_class=2), ref)
    for alpha in (0, 1, 1.5):
        with pytest.raises(ValueError, match="alpha must be in"):
            est.class_audit_from_sample(s, ref, alpha=alpha)
    with pytest.raises(ValueError, match="conf must be in"):
        est.class_audit_from_sample(s, ref, conf=1)
    for accept, msg in (([2, 3], "maps class codes"), ({"x": [3]}, "not a class code"), ({"2": 3}, "not a list"),
                        ({"2": ["3"]}, "not a class code"), ({"2": [3.5]}, "not a class code"),
                        ({"2": [3], 2: [4]}, "twice")):
        with pytest.raises(ValueError, match=msg):
            est.class_audit_from_sample(s, ref, also_accept=accept)


def test_every_refusal_of_the_audit(tmp_path, capsys):
    pred, top, truth = _write_map(tmp_path)
    p, out = str(tmp_path / "p.npy"), str(tmp_path / "a.csv")
    audit = ["sample", p, "--design", "audit", "--patch", "1"]
    for extra, msg in (
            (["--design", "random", "--budget", "10", "--per-class", "5"], "belong to --design audit"),
            (["--budget", "10", "--zone-share", "0.5"], "belong to --design audit"),
            (["--budget", "10", "--zone-cut", "0.5"], "belong to --design audit"),
            ([], "--budget is required"),
            (["--other", p], "--budget is required")):
        with pytest.raises(SystemExit, match=msg):
            main(["sample", p, "--patch", "1", "--out", out, *extra])
    for extra, msg in (
            ([], "needs --per-class N"), (["--per-class", "0"], "at least 1"),
            (["--per-class", "5", "--budget", "10"], "drop --budget"),
            (["--per-class", "5", "--zone-share", "0.5", "--zone-cut", "0.5"], "not both"),
            (["--per-class", "5", "--zone-share", "0"], r"in \(0, 1\]"),
            (["--per-class", "5", "--zone-share", "1.5"], r"in \(0, 1\]"),
            (["--per-class", "5", "--zone-cut", "nan"], "finite"),
            (["--per-class", "5", "--zone-cut", "inf"], "finite"),
            (["--per-class", "5", "--zone-cut", "2"], "no valid window reaches confidence 2"),
            (["--per-class", "5", "--condition", p], "does not take --condition"),
            (["--per-class", "5", "--other", p], "--other compares two maps"),
            (["--per-class", "5", "--threshold", "0.5"], "used only with --other"),
            (["--per-class", "5", "--anchor", "0.5"], "belong to --design sequential"),
            (["--per-class", "5", "--alpha", "0.05"], "belong to --design sequential"),
            (["--per-class", "5", "--extend", out], "belong to --design sequential")):
        with pytest.raises(SystemExit, match=msg):
            main([*audit, "--out", out, *extra])
    with pytest.raises(SystemExit, match="is a directory"):
        main([*audit, "--per-class", "5", "--out", str(tmp_path)])
    assert main([*audit, "--per-class", "8", "--out", out]) == 0
    good = _fill(out, truth)
    with open(tmp_path / "pairs.json", "w") as f:
        json.dump({"2": [3]}, f)
    for extra, msg in ((["--per-class"], "read per class already"),
                       (["--scores", p], "takes none of --scores"), (["--confidence", p], "takes none of --scores"),
                       (["--nodata", "0"], "takes none of --scores"),
                       (["--reviewer-false-alarm", "0.1"], "no reviewer error rate"),
                       (["--reviewer-miss", "0.1"], "no reviewer error rate"),
                       (["--alpha", "1.5"], r"alpha must be in \(0, 1\)"),
                       (["--out", out], "would overwrite the input"),
                       (["--out", str(tmp_path / "a.json")], "would overwrite the input"),
                       (["--out", str(tmp_path / "pairs.json"), "--also-accept", str(tmp_path / "pairs.json")],
                        "would overwrite the input")):
        with pytest.raises(SystemExit, match=msg):
            main(["estimate", out, *extra])
    for content, msg in (("{not json", "is not JSON"), ("[2, 3]", "maps class codes"), ('{"x": [3]}', "not a class code"),
                         ('{"2": [3.5]}', "not a class code"), ('{"2": 3}', "not a list")):
        with open(tmp_path / "bad.json", "w") as f:
            f.write(content)
        with pytest.raises(SystemExit, match=msg):
            main(["estimate", out, "--also-accept", str(tmp_path / "bad.json")])
    with pytest.raises(SystemExit, match="no such file"):
        main(["estimate", out, "--also-accept", str(tmp_path / "missing.json")])
    # the rows: reordered, a class edited, a reference blank or not a whole number, a column removed
    _rewrite(out, good[1:] + good[:1])
    with pytest.raises(SystemExit, match="do not match the design"):
        main(["estimate", out])
    bad = [dict(r) for r in good]
    bad[3]["map_class"] = str((int(bad[3]["map_class"]) + 1) % 4)
    _rewrite(out, bad)
    with pytest.raises(SystemExit, match="map_class differs from the class the sample recorded"):
        main(["estimate", out])
    # -1, a common "no data" code, was counted as a judged error until the review of 13 October 2026; ? is the mark
    for v in ("", "1.5", "x", "-1"):
        bad = [dict(r) for r in good]
        bad[2]["reference_class"] = v
        _rewrite(out, bad)
        with pytest.raises(SystemExit, match=r"`reference_class` must be the class seen \(a whole number of at least 0, "
                                             r"in the map's class codes\) or \? in every row, \? where a window cannot"):
            main(["estimate", out])
    _rewrite(out, [{k: v for k, v in r.items() if k != "map_class"} for r in good],
             [k for k in good[0] if k != "map_class"])
    with pytest.raises(SystemExit, match="expected the columns"):
        main(["estimate", out])
    _rewrite(out, good)
    # the other commands: certify refuses the sample, decide its sidecar
    with pytest.raises(SystemExit, match="an audit sample draws a fixed number from each class's confident zone"):
        main(["certify", out, "--alpha", "0.05"])
    with pytest.raises(SystemExit, match="this is a sample's sidecar"):
        main(["decide", str(tmp_path / "a.json"), "--ask", "error_rate_below=0.05"])
    assert main(["estimate", out, "--alpha", "0.05"]) == 0
    with pytest.raises(SystemExit, match="error_rate_below"):
        main(["decide", str(tmp_path / "a_estimate.json"), "--ask", "user_accuracy_above=0.9"])
    # --alpha and --also-accept on another design's sample
    assert main(["sample", p, "--budget", "40", "--design", "random", "--patch", "1", "--out", str(tmp_path / "r.csv")]) == 0
    _fill(tmp_path / "r.csv", np.zeros(300, int), field="wrong")
    for extra in (["--alpha", "0.05"], ["--also-accept", str(tmp_path / "pairs.json")]):
        with pytest.raises(SystemExit, match="read a per-class audit"):
            main(["estimate", str(tmp_path / "r.csv"), *extra])
    np.save(tmp_path / "q.npy", np.roll(np.load(p), 1, axis=0))
    assert main(["sample", p, "--other", str(tmp_path / "q.npy"), "--budget", "20", "--patch", "1",
                 "--out", str(tmp_path / "d.csv")]) == 0
    _fill(tmp_path / "d.csv", truth)
    for extra in (["--alpha", "0.05"], ["--also-accept", str(tmp_path / "pairs.json")]):
        with pytest.raises(SystemExit, match="takes none of"):
            main(["estimate", str(tmp_path / "d.csv"), *extra])


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="the mcp extra is not installed")
def test_the_server_refuses_an_audit_by_name(tmp_path, capsys):
    """MCP replies are taught texts, and an audit's has not been written or tested: estimate and decide refuse an audit
    sample and its result before running anything, and point to the command line."""
    from oe_inferencex import mcp_server
    pred, top, truth = _write_map(tmp_path)
    out = str(tmp_path / "a.csv")
    assert main(["sample", str(tmp_path / "p.npy"), "--design", "audit", "--per-class", "6", "--patch", "1",
                 "--out", out]) == 0
    _fill(out, truth)
    with pytest.raises(mcp_server.Refused, match="is a per-class audit") as e:
        mcp_server.estimate(sample_csv=out)
    assert f"oe-inferencex estimate {out}" in str(e.value) and mcp_server._mcp_words(str(e.value)) == str(e.value)
    assert not os.path.exists(tmp_path / "a_estimate.json")
    assert main(["estimate", out, "--alpha", "0.05"]) == 0
    with pytest.raises(mcp_server.Refused, match="the result of a per-class audit") as e:
        mcp_server.decide(result_json=str(tmp_path / "a_estimate.json"), questions=["error_rate_below=0.05"])
    assert mcp_server._mcp_words(str(e.value)) == str(e.value)
    assert not os.path.exists(tmp_path / "a_estimate_decisions.json")
    with pytest.raises(mcp_server.Refused, match="this is a sample's sidecar"):     # decide's own refusal of a sidecar
        mcp_server.decide(result_json=str(tmp_path / "a.json"), questions=["error_rate_below=0.05"])

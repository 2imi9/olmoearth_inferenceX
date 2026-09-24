"""Every command with every option at least once, on a small synthetic map (review of 2026-09-23).

The second review found `certify --rule bonferroni` crashing on every certified zone: no test had ever run that
option. This file runs each command under each of its options and designs, and the documented refusals, so an
option nobody calls cannot break unseen again. Each run checks the exit, the file it writes and one invariant of
its output; the numbers themselves are checked elsewhere (tests/test_cli.py, tests/test_consistency.py)."""
import csv
import json
import os

import numpy as np
import pytest

from oe_inferencex.assess import _pooled_argmax, assess_prediction
from oe_inferencex.cli import main

C, H, W, PATCH = 3, 64, 64, 4


@pytest.fixture(scope="module")
def maps(tmp_path_factory):
    """A 3-class score map with smooth structure, its logits, a hard class map, a truth map and a group raster."""
    d = tmp_path_factory.mktemp("matrix")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:H, :W] / H
    fields = np.stack([np.sin(6 * xx + k) + np.cos(5 * yy - k) + 0.6 * rng.normal(size=(H, W)) for k in range(C)])
    logits = 2.5 * fields
    probs = np.exp(logits - logits.max(0)); probs /= probs.sum(0)
    truth = np.where(rng.random((H, W)) < 0.85, probs.argmax(0), rng.integers(0, C, (H, W)))
    groups = (np.arange(H)[:, None] // 32) * 2 + (np.arange(W)[None, :] // 32)
    paths = {}
    for name, arr in (("probs", probs.astype(np.float32)), ("logits", logits.astype(np.float32)),
                      ("probs_b", np.roll(probs, 3, axis=2).astype(np.float32)), ("cls", probs.argmax(0)),
                      ("cls_b", np.roll(probs.argmax(0), 3, axis=1)), ("truth", truth), ("groups", groups),
                      ("binary", probs[0].astype(np.float32)), ("cover", (100 * probs[0]).astype(np.float32))):
        paths[name] = str(d / f"{name}.npy")
        np.save(paths[name], arr)
    paths["dir"] = d
    return paths


def _json(path):
    with open(path) as f:
        return json.load(f)


def _label(csv_path, scores_path, truth_path, patch=PATCH):
    """Fill `wrong` and `reference_class` the way a reviewer would, from the truth's majority per window."""
    scores, truth = np.load(scores_path), np.load(truth_path)
    hard = assess_prediction(scores, is_logit=False, patch=patch)["arrays"]["pooled_argmax"]
    ref = _pooled_argmax(truth, C, patch)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        r["reference_class"] = str(int(ref[i, j]))
        r["wrong"] = str(int(hard[i, j] != ref[i, j]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _interval_ok(block):
    return 0.0 <= block["low"] <= block["estimate"] <= block["high"] <= 1.0


# ----------------------------------------------------------------------------- assess
@pytest.mark.parametrize("extra", [[], ["--order", "boundary_first"], ["--budgets", "0.02", "0.2"],
                                   ["--reference", "truth"], ["--nodata", "-1"]])
def test_assess_under_each_option(maps, tmp_path, extra):
    extra = [maps[x] if x in maps else x for x in extra]
    out = tmp_path / "a"
    assert main(["assess", maps["probs"], "--out", str(out), *extra]) == 0
    s = _json(out / "assessment.json")
    assert s["n_windows"] == (H // PATCH) * (W // PATCH) and s["review_sets"]
    if "--reference" in extra:
        assert "error_capture_at_budget" in s["against_reference"]


def test_assess_with_logits(maps, tmp_path):
    assert main(["assess", maps["logits"], "--logits", "--out", str(tmp_path / "l")]) == 0
    assert _json(tmp_path / "l" / "assessment.json")["review_sets"]


def test_assess_with_a_reference_that_grades_nothing_prints_and_exits_zero(maps, tmp_path, capsys):
    empty = tmp_path / "none.npy"
    np.save(empty, np.full((H, W), -1))
    assert main(["assess", maps["probs"], "--out", str(tmp_path / "n"), "--reference", str(empty)]) == 0
    assert "nothing was graded" in capsys.readouterr().out


# ----------------------------------------------------------------------------- compare
@pytest.mark.parametrize("a,b,extra", [
    ("probs", "probs_b", []), ("cls", "cls_b", []), ("cls", "probs_b", []), ("binary", "binary", []),
    ("cover", "cover", ["--threshold", "50"]), ("probs", "probs_b", ["--labels", "truth"]),
    ("probs", "probs_b", ["--groups", "groups"]), ("probs", "probs_b", ["--labels", "truth", "--groups", "groups"]),
    ("probs", "probs_b", ["--date-a", "2020-06-01", "--date-b", "2020-06-01", "--labels", "truth"]),
    ("probs", "probs_b", ["--date-a", "2020-06-01", "--date-b", "2021-06-01", "--labels", "truth",
                          "--labels-date", "2021-06-01"]),
    ("probs", "probs_b", ["--date-a", "2020-01-01/2020-12-31", "--date-b", "2021-01-01/2021-12-31"]),
    ("probs", "probs_b", ["--patch", "8"]),
])
def test_compare_under_each_option(maps, tmp_path, a, b, extra):
    extra = [maps[x] if x in maps else x for x in extra]
    out = tmp_path / "c"
    assert main(["compare", maps[a], maps[b], "--out", str(out), *extra]) == 0
    s = _json(out / "comparison.json")
    assert 0 < s["n_windows"] and 0 <= s["n_disagree"] <= s["n_windows"] and s["dates"]["status"]
    if "--labels" in extra:
        assert s["graded"]["graded_against"]


@pytest.mark.parametrize("extra,why", [
    (["--labels-date", "2020-06-01"], "--labels-date names"),
    (["--labels", "truth", "--date-a", "2020-06-01"], "both --date-a and --date-b"),
    (["--labels", "truth", "--date-a", "2020-06-01", "--date-b", "2021-06-01"], "pass --labels-date"),
    (["--date-a", "June 2020", "--date-b", "2020-06-01"], "not an ISO date"),
])
def test_compare_refusals(maps, tmp_path, extra, why):
    extra = [maps[x] if x in maps else x for x in extra]
    with pytest.raises(SystemExit, match=why):
        main(["compare", maps["probs"], maps["probs_b"], "--out", str(tmp_path / "r"), *extra])


# ----------------------------------------------------------------------------- sample, estimate, certify
@pytest.fixture(scope="module", params=["confidence", "proportional", "random", "tiles"])
def labelled(maps, request):
    design = request.param
    csv_path = str(maps["dir"] / f"s_{design}.csv")
    extra = ["--tile", "4", "--per-tile", "5"] if design == "tiles" else []
    assert main(["sample", maps["probs"], "--budget", "80", "--design", design, "--out", csv_path, "--seed", "3", *extra]) == 0
    _label(csv_path, maps["probs"], maps["truth"])
    return design, csv_path


def test_estimate_under_each_design(labelled, tmp_path):
    design, csv_path = labelled
    out = tmp_path / "e.json"
    assert main(["estimate", csv_path, "--out", str(out)]) == 0
    r = _json(out)
    assert r["design"] == design and _interval_ok(r)


def test_estimate_per_class_under_each_design(labelled, tmp_path):
    design, csv_path = labelled
    out = tmp_path / "pc.json"
    if design == "tiles":
        with pytest.raises(SystemExit, match="not graded yet"):
            main(["estimate", csv_path, "--per-class", "--out", str(out)])
        return
    assert main(["estimate", csv_path, "--per-class", "--out", str(out)]) == 0
    r = _json(out)
    assert _interval_ok(r["overall_accuracy"]) and len(r["per_class"]) == C
    for row in r["per_class"].values():
        for q in ("user_accuracy", "producer_accuracy", "reference_share"):
            if row[q] is not None:
                assert _interval_ok(row[q]), (q, row[q])


@pytest.mark.parametrize("rule", ["prefix", "bonferroni"])            # the plug-in comparator is Python-only: no guarantee
@pytest.mark.parametrize("alpha", [0.3, 0.001])
def test_certify_under_each_rule(labelled, tmp_path, rule, alpha):
    design, csv_path = labelled
    out = tmp_path / "z.json"
    args = ["certify", csv_path, "--alpha", str(alpha), "--rule", rule, "--delta", "0.2", "--out", str(out)]
    if design != "random":
        with pytest.raises(SystemExit, match="needs a random sample"):
            main(args)
        return
    assert main(args) == 0                                   # bonferroni used to crash here on every certified zone
    z = _json(out)
    assert z["rule"] == rule and z.get("note")
    assert (z["coverage"] is None) == (not os.path.exists(str(out)[:-5] + ".npy"))


def test_estimate_refuses_the_maps_options_without_per_class(labelled, tmp_path):
    """Release check of 24 September: --scores and --nodata do nothing without --per-class, and a wrong --scores
    path was accepted with exit 0."""
    design, csv_path = labelled
    for extra in (["--scores", str(tmp_path / "missing.npy")], ["--nodata", "5"]):
        with pytest.raises(SystemExit, match="only with --per-class"):
            main(["estimate", csv_path, "--out", str(tmp_path / "e.json"), *extra])


def test_certify_offers_only_the_rules_with_a_guarantee(labelled, tmp_path):
    design, csv_path = labelled
    if design == "random":
        with pytest.raises(SystemExit):
            main(["certify", csv_path, "--rule", "plugin", "--out", str(tmp_path / "p.json")])

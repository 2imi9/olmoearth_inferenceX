"""Typed answers (oe_inferencex.decide) on results written by hand, and through the command line on the quick-start
map. Each rule is checked at its edges: an interval that touches the threshold, a zone that is not certified, a
comparison that has no labels or labels on part of the differing windows, a class with no interval."""
import json
import os
import shlex

import pytest

from oe_inferencex import cli
from oe_inferencex.decide import decide, kind, parse


def _estimate(low, high, **extra):
    return {"design": "random", "n_labelled": 300, "n_population": 4096, "nominal_coverage": 0.95,
            "estimate": (low + high) / 2, "low": low, "high": high,
            "method": "exact hypergeometric interval (simple random sample of a finite map)", **extra}


def _one(result, question):
    return decide(result, [question])["answers"]


@pytest.mark.parametrize("low, high, t, answer", [
    (0.04, 0.09, 0.10, "yes"),            # wholly below
    (0.04, 0.10, 0.10, "undetermined"),   # touches the threshold from below: not shown to be below it
    (0.10, 0.14, 0.10, "no"),             # at or above
    (0.08, 0.12, 0.10, "undetermined"),   # across
])
def test_error_rate_below_is_three_way(low, high, t, answer):
    a = list(_one(_estimate(low, high), f"error_rate_below={t}").values())[0]
    assert (a["type"], a["answer"], a["level"], a["exact"]) == ("yes_no", answer, 0.95, True)
    assert "which are assumed right" in a["because"]
    if answer == "yes":
        assert a["because"].startswith("At 95% confidence, the error rate of the map is below 10.0%")
    if answer == "undetermined":
        assert "More labels narrow the interval" in a["because"]


def test_an_approximate_interval_says_so_and_carries_its_warning():
    r = _estimate(0.01, 0.096, method="tile design: ratio estimator with a t interval", design="tiles",
                  warning="the interval can under-cover where a few tiles hold most of the errors")
    a = _one(r, "error_rate_below=0.15")["error_rate_below=0.15"]
    assert a["answer"] == "yes" and a["exact"] is False
    assert "At a nominal 95% confidence" in a["because"] and "approximate for this design" in a["because"]
    assert "Warning: the interval can under-cover where a few tiles hold most of the errors." in a["because"]
    assert a["evidence"]["warning"] == r["warning"]


def test_a_reviewer_who_errs_replaces_the_labels_are_right_clause():
    r = _estimate(0.03, 0.12, reviewer_false_alarm=0.05, reviewer_miss=0.1, bounds_note="widened for the reviewer")
    a = _one(r, "error_rate_below=0.2")["error_rate_below=0.2"]
    assert "false alarms 5.0%, misses 10.0%" in a["because"] and "assumed right" not in a["because"]
    assert "widened for the reviewer" in a["because"]


def test_each_condition_gets_its_own_answer():
    r = _estimate(0.03, 0.08, per_condition={"clear": {"low": 0.01, "high": 0.04}, "cloudy": {"low": 0.12, "high": 0.30},
                                             "empty": {"low": None, "high": None}})
    a = _one(r, "error_rate_below=0.1")["error_rate_below=0.1"]
    assert a["answer"] == "yes"
    assert {k: v["answer"] for k, v in a["per_condition"].items()} == {"clear": "yes", "cloudy": "no",
                                                                      "empty": "undetermined"}
    assert "can differ from the whole map's" in a["because"] and "do not hold jointly" in a["because"]


def test_class_accuracy_above_and_a_class_with_no_interval():
    row = lambda lo, hi: {"user_accuracy": {"estimate": (lo + hi) / 2, "low": lo, "high": hi},
                          "producer_accuracy": {"estimate": 0.9, "low": 0.85, "high": 0.95}}
    r = _estimate(0.05, 0.1, per_class={"0": row(0.91, 0.97), "1": row(0.70, 0.90), "2": row(0.80, 0.95),
                                        "3": {"user_accuracy": None, "producer_accuracy": None,
                                              "warning": "the map never predicts this class"}})
    a = _one(r, "user_accuracy_above=0.9")["user_accuracy_above=0.9"]
    assert {c: v["answer"] for c, v in a["per_class"].items()} == {"0": "yes", "1": "no", "2": "undetermined",
                                                                  "3": "undetermined"}
    assert "never predicts" in a["per_class"]["3"]["because"]
    assert "nominal" in a["because"] and "do not hold jointly" in a["because"] and a["exact"] is False
    with pytest.raises(ValueError, match="cannot be answered"):
        decide(_estimate(0.05, 0.1), ["user_accuracy_above=0.9"])          # no per_class in the result


@pytest.mark.parametrize("verdict, low, high, answer", [("a", 0.01, 0.05, "a"), ("b", -0.05, -0.01, "b"),
                                                        (None, -0.02, 0.03, "undetermined")])
def test_more_accurate_from_a_sample_of_the_differing_windows(verdict, low, high, answer):
    r = {"design": "disagreement", "n_population": 4096, "n_disagree": 507, "disagree_share": 507 / 4096,
         "n_labelled": 100, "n_unjudged": 0, "n_a_right": 50, "n_b_right": 45, "n_neither": 5,
         "difference": {"estimate": 0.0, "estimate_range": [0, 0], "low": low, "high": high}, "verdict": verdict,
         "conf": 0.95}
    a = _one(r, "more_accurate")["more_accurate"]
    assert (a["type"], a["answer"], a["level"]) == ("choice", answer, 0.95)
    assert "not either map's accuracy" in a["because"]
    assert _one(r, "share_differs")["share_differs"]["answer"] == pytest.approx(507 / 4096)


def test_a_sample_sidecar_is_not_a_result():
    with pytest.raises(ValueError, match="sidecar, written before the labels"):
        decide({"design": "disagreement", "n_population": 4096, "n_disagree": 507, "indices": [1, 2]}, ["more_accurate"])


def test_a_comparison_names_a_map_only_with_labels_on_every_differing_window():
    base = {"n_windows": 4096, "n_disagree": 507, "disagreement_rate": 507 / 4096, "graded": None,
            "dates": {"status": "unstated"}}
    with pytest.raises(ValueError, match="compare cannot say which map is right without labels"):
        decide(base, ["more_accurate"])
    s = _one(base, "share_differs")["share_differs"]
    assert s["answer"] == pytest.approx(0.1238, abs=1e-4) and "both can be wrong" in s["because"]
    graded = lambda n, a, b: dict(base, graded={"which_side": {"n_disagree": n, "a_right": a, "b_right": b,
                                                               "neither": n - a - b}})
    for a_right, b_right, answer in ((228, 239, "b"), (239, 228, "a"), (230, 230, "tie")):
        got = _one(graded(507, a_right, b_right), "more_accurate")["more_accurate"]
        assert got["answer"] == answer and "taken as truth" in got["because"] and got["level"] is None
    part = _one(graded(29, 10, 18), "more_accurate")["more_accurate"]
    assert part["answer"] == "undetermined" and "29 of the 507" in part["because"] and "not a random sample" in part["because"]
    none = _one(graded(0, 0, 0), "more_accurate")["more_accurate"]
    assert none["answer"] == "undetermined" and "cover none of the 507" in none["because"]
    apart = dict(graded(507, 200, 250), dates={"status": "different_time", "reading": "the maps describe different "
                                               "years; a difference can be real change on the ground"})
    apart["graded"]["graded_against"] = "the labels describe the date of map b; map a is counted wrong where it changed"
    said = _one(apart, "more_accurate")["more_accurate"]["because"]
    assert "The maps describe different years; a difference can be real change on the ground." in said
    assert "The labels describe the date of map b; map a is counted wrong where it changed." in said
    assert "real change on the ground" in _one(apart, "share_differs")["share_differs"]["because"]


def test_a_zone_not_certified_is_never_a_no():
    zone = {"rule": "prefix", "alpha": 0.05, "delta": 0.1, "n_population": 4096, "n_labelled": 300, "coverage": 0.9,
            "n_zone": 3686}
    share = _one(zone, "trusted_share")["trusted_share"]
    assert share["answer"] == 3686 / 4096 and share["level"] == 0.9      # the zone's own size, not the grid's 0.9
    assert _one(zone, "trusted_share_at_least=0.5")["trusted_share_at_least=0.5"]["answer"] == "yes"
    assert _one(zone, "trusted_share_at_least=0.9")["trusted_share_at_least=0.9"]["answer"] == "undetermined"
    short = _one(zone, "trusted_share_at_least=0.95")["trusted_share_at_least=0.95"]
    assert short["answer"] == "undetermined" and short["because"].startswith("No zone covering at least 95.0%")
    assert "only where the map's error rate is at most 5.0%" in short["because"]
    none = dict(zone, coverage=None, n_zone=None)
    t = _one(none, "trusted_share")["trusted_share"]
    assert t["answer"] == 0.0 and "not evidence that the map is worse" in t["because"]
    for s in (0.05, 0.5, 0.99):
        assert _one(none, f"trusted_share_at_least={s}")[f"trusted_share_at_least={s:g}"]["answer"] == "undetermined"
    with pytest.raises(ValueError, match="plug-in rule carries no guarantee"):
        decide(dict(zone, rule="plugin"), ["trusted_share"])


def test_a_zone_per_condition():
    r = {"rule": "prefix", "alpha": 0.1, "delta": 0.1, "n_population": 4096, "n_labelled": 400, "coverage": None,
         "by_condition": True, "certified_share_of_map": 0.975,
         "per_condition": {"west": {"tested": True, "coverage": 0.95, "n_zone": 1946, "n_population": 2048,
                                    "reason": None},
                           "east": {"tested": True, "coverage": 1.0, "n_zone": 2048, "n_population": 2048, "reason": None},
                           "north": {"tested": False, "coverage": None, "n_zone": None, "n_population": 100,
                                     "reason": "12 labels; certifying any zone needs at least 45"}}}
    a = _one(r, "trusted_share")["trusted_share"]
    assert a["answer"] == 0.975
    assert {k: v["certified_share_of_condition"] for k, v in a["per_condition"].items()} == \
        {"west": 1946 / 2048, "east": 1.0, "north": None}


def test_questions_and_files_are_refused_plainly(tmp_path):
    assert parse("error_rate_below=0.1") == ("error_rate_below", 0.1) and parse("more_accurate") == ("more_accurate", None)
    for bad, said in (("error_rate_below", "needs a threshold"), ("error_rate_below=1.5", "between 0 and 1"),
                      ("error_rate_below=x", "not a number"), ("more_accurate=1", "takes no value"),
                      ("which_is_best", "unknown question")):
        with pytest.raises(ValueError, match=said):
            parse(bad)
    with pytest.raises(ValueError, match="no question asked; this estimate result answers: error_rate_below"):
        decide(_estimate(0.01, 0.02), [])
    with pytest.raises(ValueError, match="not a result"):
        kind({"anything": 1})
    with pytest.raises(ValueError, match="cannot read"):
        decide(str(tmp_path / "missing.json"), ["trusted_share"])
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="not an object"):
        decide(str(tmp_path / "list.json"), ["trusted_share"])
    with pytest.raises(SystemExit, match="cannot read"):
        cli.main(["decide", str(tmp_path / "missing.json"), "--ask", "trusted_share"])


pytest.importorskip("rasterio", reason="the quick-start map is a GeoTIFF")


def test_decide_reads_what_the_quick_start_wrote(tmp_path, monkeypatch, capsys):
    """The quick start's own results: 7.0% (4.5% to 10.4%) and a zone of 3686 of 4096 windows at 5%."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(root, "examples", "quickstart_map.py"))
    qm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qm)
    monkeypatch.chdir(tmp_path)
    qm.write_maps()
    cli.main(shlex.split("sample scores.tif --budget 300 --design random --out to_label.csv"))
    qm.label("to_label.csv")
    cli.main(shlex.split("estimate to_label.csv"))
    cli.main(shlex.split("certify to_label.csv --alpha 0.05"))
    cli.main(shlex.split("compare scores.tif other.tif --out diff"))
    capsys.readouterr()

    cli.main(shlex.split("decide to_label_estimate.json --ask error_rate_below=0.12 --ask error_rate_below=0.05"))
    printed = capsys.readouterr().out
    # Usage's decide block is what the command prints
    usage = open(os.path.join(root, "docs", "Usage.md"), encoding="utf-8").read()
    block = usage[usage.index("### decide"):].split("```console\n", 1)[1].split("\n```", 1)[0].split("\n")
    assert block[0] == "$ oe-inferencex decide to_label_estimate.json --ask error_rate_below=0.12 --ask error_rate_below=0.05"
    assert printed.rstrip("\n").split("\n") == block[1:]
    written = json.load(open("to_label_estimate_decisions.json"))
    assert {q: a["answer"] for q, a in written["answers"].items()} == {"error_rate_below=0.12": "yes",
                                                                       "error_rate_below=0.05": "undetermined"}

    cli.main(shlex.split("decide to_label_zone.json --ask trusted_share --ask trusted_share_at_least=0.9"))
    printed = capsys.readouterr().out
    assert "trusted_share: 90.0%." in printed and "trusted_share_at_least=0.9: undetermined." in printed

    with pytest.raises(SystemExit, match="compare cannot say which map is right without labels"):
        cli.main(shlex.split("decide diff/comparison.json --ask more_accurate"))

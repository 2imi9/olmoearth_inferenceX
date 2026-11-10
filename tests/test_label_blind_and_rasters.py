"""The text fixes after a thesis user's notes (Unreleased): labelling blind, a map in several rasters, what the
which-map labels cannot grade, and the smaller wording. No number or computation is tested as changed; the stratum
order the docs now describe is checked against the draw itself."""
import csv
import json
import os

import numpy as np

from oe_inferencex import estimate as est
from oe_inferencex.cli import main
from oe_inferencex.common.metrics import weighted_auroc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _joined(rel):
    return " ".join(open(os.path.join(ROOT, rel), encoding="utf-8").read().split())


def test_stratified_designs_write_rows_stratum_by_stratum_and_random_does_not():
    """What Usage now says to hide from a reviewer: under the confidence and proportional designs the drawn windows
    come stratum by stratum, so their order shows how sure the map was; a random sample's come in random order."""
    rng = np.random.default_rng(3)
    margin = rng.random(2000)
    for design in ("confidence", "proportional"):
        s = est.sample_for_estimation(margin, 200, design=design, p1=0.5 + margin / 2, seed=1)
        pop = np.asarray(s["strata_of_population"])
        drawn = np.asarray(s["strata"])[np.searchsorted(pop, s["indices"])]
        assert (np.diff(drawn) >= 0).all() and len(set(drawn.tolist())) > 1, design
    r = est.sample_for_estimation(margin, 200, design="random", seed=1)
    assert not (np.diff(r["indices"]) >= 0).all()


def test_usage_and_the_mcp_server_say_what_to_hide_and_how_to_copy_labels_back():
    usage = _joined("docs/Usage.md")
    for said in ("hide the `map_class`, `confidence`, `stratum` and `condition` columns",
                 "show the reviewer a shuffled copy and copy each label back to its row by `index`, never by position",
                 "A random sample's rows come in the draw's random order",
                 "hiding `class_a` and `class_b` is all that labelling blind needs here",
                 "Neither effect was measured."):
        assert said in usage, said
    src = _joined("oe_inferencex/mcp_server.py")          # read as text: the mcp extra is optional
    assert "To label blind, hide map_class, confidence, stratum and condition, " in src
    assert "To label blind, hide map_class, write the class seen" not in src
    # a sequential sample is labelled from the top row down; only the other designs get the shuffled copy
    assert ('"label the rows from the top row down as they come (they are already in random order), " '
            'if d == "sequential" else "show the windows in a shuffled copy and copy each label back to its row by '
            'index') in src
    assert "(a sequential sample is labelled from the top row down in its own random order, not from a shuffled copy)" \
        in usage
    audit = _joined("oe_inferencex/estimate/sample_command.py")
    assert "hide map_class and confidence to label blind" in audit and "hide map_class to label blind" not in audit
    assert "hide map_class and confidence to label blind" in usage


def test_usage_says_the_which_map_labels_do_not_grade_a_confidence():
    usage = _joined("docs/Usage.md")
    assert "Nor do these labels grade either map's confidence" in usage
    assert "so no window where the maps agree, right or wrong, is ever drawn" in usage


def test_usage_gives_one_error_rate_for_a_map_in_several_rasters():
    usage = _joined("docs/Usage.md")
    for said in ("#### One map in several rasters",
                 "[One map in several rasters](#one-map-in-several-rasters)",
                 "write `parts.csv` with the header `part,windows` and one line per raster giving the `n_population` "
                 "from its sidecar",
                 "--stratum part --reference reference_class --map map_class --strata parts.csv",
                 "with its own `--seed` (the default, 0, would give rasters of equal size the same windows, and the "
                 "strata must be drawn independently)",
                 "valid at any sample size when the rasters were drawn independently",
                 "It is not each raster's `estimate` pooled. A certified zone stays per raster."):
        assert said in usage, said


def test_cli_grades_two_rasters_as_strata_of_one_map(tmp_path):
    """The route as Usage writes it: each raster's random sample a stratum, its n_population the stratum's size."""
    rows = ([{"part": "z10", "reference_class": "1", "map_class": "1"}] * 18
            + [{"part": "z10", "reference_class": "2", "map_class": "1"}] * 2
            + [{"part": "z11", "reference_class": "1", "map_class": "1"}] * 10
            + [{"part": "z11", "reference_class": "2", "map_class": "1"}] * 10
            + [{"part": "z11", "reference_class": "", "map_class": "1"}])
    pts = tmp_path / "all.csv"
    with open(pts, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "reference_class", "map_class"])
        w.writeheader()
        w.writerows(rows)
    parts = tmp_path / "parts.csv"
    parts.write_text("part,windows\nz10,3000\nz11,1000\n")
    out = tmp_path / "g.json"
    assert main(["grade", str(pts), "--stratum", "part", "--reference", "reference_class", "--map", "map_class",
                 "--strata", str(parts), "--out", str(out)]) == 0
    res = json.load(open(out))
    oa = res["overall_accuracy"]["map_class"]
    assert res["strata"] == 2
    # 0.75 * 18/20 + 0.25 * [10/21, 11/21]: the unjudged row is bounded both ways, the strata weighted by their sizes
    assert np.allclose(oa["estimate"], [0.75 * 0.9 + 0.25 * 10 / 21, 0.75 * 0.9 + 0.25 * 11 / 21])
    lo, hi = oa["interval"]
    assert lo < oa["estimate"][0] and oa["estimate"][1] < hi


def test_smaller_wording():
    import argparse
    from oe_inferencex.agency.command import add_grade_parser
    sub = argparse.ArgumentParser().add_subparsers()
    add_grade_parser(sub)
    helps = {a.dest: a.help for a in sub.choices["grade"]._actions}
    assert "The first is the map --pure refers to" in helps["map"]
    assert "A point value only" in " ".join(weighted_auroc.__doc__.split())
    usage = _joined("docs/Usage.md")
    assert "another CRS, pixel size or origin is refused, not resampled" in usage
    assert "as point values with no interval" in usage
    assert ("Check it with a few labels: a random sample and `certify`, or a sample by confidence band."
            in _joined("README.md"))


def _sample(N, D, n):
    return {"design": "disagreement", "indices": np.arange(n), "n_population": N, "n_disagree": D}


def _labels(a, b, c):
    n = a + b + c
    ref = np.r_[np.zeros(a), np.ones(b), np.full(c, 2)].astype(int)
    return np.zeros(n, int), np.ones(n, int), ref, np.zeros(n, bool)


def test_the_verdict_note_names_the_rule_that_decided():
    r = est.compare_from_disagreement(_sample(50_000, 10_000, 100), *_labels(30, 15, 55))
    assert r["verdict_rule"] == "sign" and r["verdict_note"] == est.SIGN_TEST_NOTE
    r = est.compare_from_disagreement(_sample(1000, 40, 40), *_labels(21, 19, 0))
    assert r["verdict_rule"] == "interval" and r["verdict_note"].startswith("The verdict is the interval's")
    assert "sign test" not in r["verdict_note"]

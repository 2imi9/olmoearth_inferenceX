"""The defects an adversarial audit of the package found on 2026-09-21, each pinned so it cannot return.

Every one of these shipped in 1.1.2 and produced a plausible wrong answer with exit 0, which is the failure mode
this project exists to prevent in the maps it audits. Each test states the user-visible consequence.
"""
import argparse
import csv
import os

import numpy as np
import pytest

from oe_inferencex import cli
from oe_inferencex.assess import assess_prediction, _pooled_argmax
from oe_inferencex.calibrate import fit_ranker


def test_the_raster_named_suspicion_contains_suspicion(tmp_path):
    """It held confidence, so ranking it descending gave the exact inverse of the review set: an analyst opening it
    with a hot-is-bad ramp reviewed the windows the model was most confident about."""
    rng = np.random.default_rng(0)
    p = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 9, 64))[:, None] + rng.normal(0, .2, (64, 64)), .01, .99)
    np.save(tmp_path / "p.npy", p)
    cli.cmd_assess(argparse.Namespace(scores=str(tmp_path / "p.npy"), out=str(tmp_path), logits=False, patch=4,
                                      nodata=None, reference=None, budgets=[0.05], order="confidence"))
    susp = np.load(tmp_path / "suspicion.npy")
    flagged = {(int(r["window_row"]), int(r["window_col"]))
               for r in csv.DictReader(open(tmp_path / "review_set_05pct.csv"))}
    k = len(flagged)
    top = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(susp, nan=-np.inf), axis=None), susp.shape))[0][:k]
    assert {(int(a), int(b)) for a, b in top} == flagged, "the review set must sit at the TOP of the suspicion raster"


def test_the_reference_is_pooled_over_its_own_classes():
    """It was pooled over the PREDICTION's class range, so a reference class the model cannot predict collapsed to
    class 0 and the reported error rate was understated, always in the flattering direction, with no warning."""
    ref = np.zeros((64, 64), int)
    ref[:, :20] = 1
    ref[:, 20:32] = 2                      # permanent water: a class a binary flood model cannot predict
    pred = np.zeros((64, 64))
    pred[:, :26] = 0.9
    a = assess_prediction(pred, is_logit=False, patch=4, reference=ref)
    honest = float((_pooled_argmax(ref, 3, 4) != a["arrays"]["pooled_argmax"]).mean())
    assert a["against_reference"]["error_rate"] == pytest.approx(honest)
    assert honest > 0.18, "the constructed case must actually exercise the collapse"
    assert any("reference carries 3 classes" in w for w in a["warnings"])


def test_no_data_windows_do_not_manufacture_boundaries():
    """A window under cloud has no class. Giving it class 0 made every neighbour disagree with it, so a map that
    predicts ONE class everywhere it has a prediction reported a boundary fraction of 16.7%."""
    p = np.full((64, 64), 0.95)
    p[:, :16] = np.nan
    a = assess_prediction(p, is_logit=False, patch=4)
    assert a["boundary_window_fraction"] == 0.0
    from oe_inferencex.signals import boundary_indicator
    clean = np.full((64, 64), 0.95)
    clean[:, :32] = 0.05                    # a real boundary on a fully observed map must be unchanged
    b = assess_prediction(clean, is_logit=False, patch=4)
    assert np.array_equal(b["arrays"]["boundary"], boundary_indicator(b["arrays"]["pooled_argmax"])), \
        "on a fully observed map the boundary must stay bit-identical, or every recorded cue number moves"


def test_non_finite_pixels_are_treated_as_no_data_and_said_so():
    """NaN sorts to the front of the review order, so without this the whole budget went to empty windows."""
    p = np.clip(np.random.default_rng(1).random((64, 64)), .01, .99)
    p[:, :16] = np.nan
    a = assess_prediction(p, is_logit=False, patch=4)
    valid = a["arrays"]["valid"]
    rc = np.asarray(a["review_sets"][0.10]["windows_rowcol"])
    assert all(valid[r, c] for r, c in rc), "no window without a prediction may enter the review set"
    assert any("not finite" in w for w in a["warnings"])


def test_a_single_band_score_map_is_refused():
    """(1, H, W) is the shape a binary head returns. It was scored as a one-class map, inverting the order."""
    with pytest.raises(ValueError, match="one class"):
        assess_prediction(np.random.default_rng(2).random((1, 32, 32)), is_logit=False)


def test_the_fusion_baseline_gets_the_orientation_the_fusion_would_learn():
    """The fusion learns each sign; grading it against a sign-locked baseline manufactured the headline lead."""
    rng = np.random.default_rng(3)
    n = 3000
    err = (rng.random(n) < 0.15).astype(float)
    conf = rng.random(n) - 0.9 * err                       # "higher is safer", the orientation the docs invite
    _, rep = fit_ranker({"confidence": conf, "top1": conf + rng.normal(0, .05, n)}, err, np.ones(n, bool),
                        groups=rng.integers(0, 12, n), family="demo")
    assert rep["best_single_orientation"] == "flipped"
    assert abs(rep["held_out_lead_over_best_single"]) < 0.01, "the lead was 0.389 when the baseline was sign-locked"


def test_an_unfittable_fold_is_never_reported_as_a_held_out_score():
    """Those rows sat at logit exactly 0.0 and were reported inside held_out, so on a good map the report said the
    ranker found 0% of the errors while quoting a better-than-random excess AURC from that fabricated vector."""
    rng = np.random.default_rng(4)
    err = np.zeros(400)
    err[:5] = 1.0                                          # few errors: the regime a good map is in
    _, rep = fit_ranker({"c": rng.random(400)}, err, np.ones(400, bool), groups=np.repeat(np.arange(20), 20),
                        family="demo")
    assert rep["n_unscored_rows"] > 0 and rep["unscored_note"]
    assert np.isnan(rep["held_out"]["excess_aurc"]), "an undefined held-out score must be NaN, never a number"

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


def test_fit_side_with_one_group_is_refused_rather_than_reported_as_always_side_a():
    """The high finding the 21 September audit left open. One group id collapses every fold; each held-out logit is
    NaN, NaN > 0 is False, and fit_side reported the "always a" rate as its fitted rule's held-out accuracy: 0.196
    on exp60 against an honest 0.830 cross-fitted by event, reversing the verdict against a 0.690 baseline."""
    import os
    from oe_inferencex.calibrate import fit_ranker, fit_side
    z = np.load(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "exp60_masks.npz"))
    fa, fb = {"margin": z["margin_A_s1pre"]}, {"margin": z["margin_B_s1post"]}
    one = np.zeros_like(z["event"])
    with pytest.raises(ValueError, match="at least two groups"):
        fit_side(fa, fb, z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"], groups=one, family="x")
    _, r = fit_side(fa, fb, z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"], groups=z["event"], family="x")
    assert round(r["held_out"]["share_right"], 3) == 0.830 and r["n_unscored_rows"] == 0
    with pytest.raises(ValueError, match="at least two groups"):
        fit_ranker({"m": z["margin_A_s1pre"]}, z["A_s1pre"] != z["y_after"], z["ok"], groups=one, family="x")


def test_fit_side_excludes_unfittable_folds_instead_of_reading_them_as_side_a():
    """Two groups where one has no disagreement of one kind: that fold cannot be fitted, and its rows must be
    counted as unscored, not scored as "believe a"."""
    from oe_inferencex.calibrate import _crossfit
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2)); y = np.r_[np.zeros(20), (rng.random(20) < 0.5).astype(float)]
    fold = np.r_[np.zeros(20, int), np.ones(20, int)]
    held = _crossfit(X, y, fold, False, 2)
    assert np.isnan(held[20:]).all()                                  # training side of fold 1 has no positives


# ----------------------------------------------------------------------------- the rest of the 21 September audit
def test_compare_counts_a_group_with_no_valid_window_as_undefined_not_a_tie():
    """Finding 9: five tiles where b breaks four and corrects one, padded with nine fully masked tiles, read
    'median 0.0, a wash' with n_undefined 0. The masked tiles are undefined now and the verdict is b's."""
    from oe_inferencex.compare import compare_inferences
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, (14, 4, 4)); a = lab.copy(); b = lab.copy(); ok = np.ones_like(lab, bool)
    for t in range(4):
        b[t, 0, :2] = 1 - lab[t, 0, :2]                           # b breaks tiles 0-3
    a[4, 0, :2] = 1 - lab[4, 0, :2]                               # b corrects tile 4
    ok[5:] = False                                                # nine fully masked tiles
    og = compare_inferences(a, b, ok, groups=np.arange(14)[:, None, None], labels=lab)["graded"]["over_groups"]
    assert og["n_undefined"] == 9 and og["n_groups"] == 5 and og["median"] < 0 and og["share_flipped"] == 0.8


def test_ece_ignores_no_data_and_counts_a_confidence_of_zero():
    """Finding 11: a scene 30% no-data reported 70% of its ECE; confidence exactly 0.0 fell out of every bin."""
    from oe_inferencex.metrics import expected_calibration_error as ece
    rng = np.random.default_rng(1)
    conf = rng.random(1000); corr = (rng.random(1000) < conf ** 2).astype(float)
    full = ece(conf, corr)[0]
    nod = conf.copy(); nod[:300] = np.nan
    assert ece(nod, corr)[0] == pytest.approx(ece(conf[300:], corr[300:])[0])
    assert ece(np.zeros(10), np.ones(10))[0] == pytest.approx(1.0)       # confident nowhere, right everywhere
    assert full > 0
    with pytest.raises(ValueError):
        ece(np.array([1.5]), np.array([1.0]))


def test_top1_confidence_quantiles_are_probabilities():
    """Finding 12: with form='top1' the quantiles were mean log-probabilities, a median of -0.62, labelled a probability."""
    from oe_inferencex.assess import assess_prediction
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(4, 32, 32)) * 2
    out = assess_prediction(logits, is_logit=True, form="top1")
    q = out["confidence_quantiles"]
    assert all(0 < v <= 1 for v in q.values()) and "geometric mean" in out["confidence_scale"]
    margin_out = assess_prediction(logits, is_logit=True)                                   # the default is unchanged
    assert "confidence_scale" not in margin_out


def test_a_partial_reference_says_its_capture_is_over_its_own_windows():
    """Finding 13: with a reference over part of the map, capture at 0.05 described other windows than the review
    set at 0.05, silently."""
    from oe_inferencex.assess import assess_prediction
    rng = np.random.default_rng(3)
    p = rng.random((64, 64)); ref = (rng.random((64, 64)) < 0.3).astype(int); ref[:, 32:] = -1
    out = assess_prediction(p, is_logit=False, reference=ref, budgets=(0.05,))
    rc = out["against_reference"]
    assert rc["n_windows_scored"] < out["n_windows"] and "not of the review sets" in rc["population"]
    assert rc["error_capture_at_budget"][0.05]["n_reviewed"] == max(1, round(0.05 * rc["n_windows_scored"]))
    assert any("reference covers" in w for w in out["warnings"])


def test_reference_unstable_cue_quotes_exp23_as_recorded():
    """Finding 14: the cue quoted 14.3x on 27 scenes; exp23 recorded 13.7x on 24."""
    import json, os
    from oe_inferencex.explain import CUES
    t1 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "exp23_summary.json")))["tests"]["T1_enrichment"]
    c = CUES["reference_unstable"]
    assert c.share_errors == pytest.approx(t1["median_share_errors"], abs=5e-4)
    assert c.share_correct == pytest.approx(t1["median_share_correct"], abs=5e-5)
    assert round(c.enrichment, 1) == 13.7 and "24" in c.reference


def test_low_confidence_cue_does_not_quote_its_20_percent_enrichment_at_another_cut():
    """Finding 15: at a 90% cut the cue still quoted exp37's 3.6x, measured at 20%. Finding 20: with no argument the
    library printed a raw '{quantile:.0%}' placeholder."""
    from oe_inferencex.explain import CUES, library_table
    c = CUES["low_confidence"]
    assert "3.6x" in c.quote(quantile=0.2)
    q90 = c.quote(quantile=0.9)
    assert "3.6x" not in q90 and "measured only at the 20% cut" in q90 and "90%" in q90
    row = [r for r in library_table() if r["name"] == "low_confidence"][0]
    assert "{" not in row["quote"] and "20%" in row["quote"]


def test_ndwi_is_the_same_on_reflectance_and_digital_numbers():
    """Finding 16: open water in L2A reflectance (green 0.10, NIR 0.02, NDWI 0.667) computed to 0.08, inside the
    cue's ambiguity band, because the denominator was clipped at 1."""
    from oe_inferencex.signals import S2_BANDS, ndwi
    img = np.zeros((12, 2, 2)); img[S2_BANDS.index("B03")] = 0.10; img[S2_BANDS.index("B08")] = 0.02
    assert ndwi(img)[0, 0] == pytest.approx(2 / 3)
    assert ndwi(img * 10000)[0, 0] == pytest.approx(2 / 3)
    assert ndwi(np.zeros((12, 2, 2)))[0, 0] == 0.0


def test_determinism_check_gives_no_verdict_on_nothing():
    """Finding 17: a fully masked scene returned passes=False, booked as an engine failure."""
    from oe_inferencex.compare import determinism_check
    a = np.zeros((4, 4), int)
    assert determinism_check(a, a, np.zeros((4, 4), bool), floor=0.03)["passes"] is None
    assert determinism_check(a, a, np.ones((4, 4), bool), floor=0.03)["passes"] is True


def test_an_empty_scene_and_an_oversized_patch_are_named_refusals():
    """Finding 18: both died with raw numpy errors pointing at unrelated lines."""
    from oe_inferencex.assess import assess_prediction
    p = np.full((16, 16), 0.7)
    with pytest.raises(ValueError, match="no valid window"):
        assess_prediction(p, is_logit=False, nodata_mask=np.ones((16, 16), bool))
    with pytest.raises(ValueError, match="larger than the map"):
        assess_prediction(p, is_logit=False, patch=32)


def test_a_ragged_edge_is_reported_not_silently_dropped():
    """Finding 21: a 100 x 100 map at patch 8 never ranked 7.8% of its pixels and said nothing."""
    from oe_inferencex.assess import assess_prediction
    out = assess_prediction(np.random.default_rng(4).random((100, 100)), is_logit=False, patch=8)
    assert out["pixels_outside_window_grid"] == 100 * 100 - 96 * 96
    assert any("never ranked" in w for w in out["warnings"])
    assert assess_prediction(np.random.default_rng(4).random((96, 96)), is_logit=False, patch=8)["pixels_outside_window_grid"] == 0


def test_the_confidence_docstring_matches_exp76():
    """Finding 19: the docstring said the logit margin was weakest on all 16 tasks; exp76 says 14 by AUROC, 15 by E-AURC."""
    from oe_inferencex.signals import confidence
    assert "all 16" not in confidence.__doc__ and "14 of 16" in confidence.__doc__


def test_a_tied_window_no_longer_goes_to_class_zero():
    """Finding 10: on a balanced two-class map every 8/8 window went to class 0, so class_share read 0.596 against
    a true 0.502. A prediction tie now goes to the more confident voters; a reference tie is left unscored."""
    rng = np.random.default_rng(5)
    p = rng.random((128, 128))                                    # balanced: class 1 where p > 0.5
    out = assess_prediction(p, is_logit=False)
    assert abs(out["class_share"][1] - (p > 0.5).mean()) < 0.03   # was ten points under
    # a tie decided by confidence: 8 pixels at 0.51 (barely class 1) and 8 at 0.02 (confidently class 0)
    q = np.where(np.arange(16).reshape(4, 4) < 8, 0.51, 0.02)
    assert _pooled_argmax((q > 0.5).astype(int), 2, 4, weights=np.abs(q - 0.5) * 2)[0, 0] == 0
    q2 = np.where(np.arange(16).reshape(4, 4) < 8, 0.99, 0.45)
    assert _pooled_argmax((q2 > 0.5).astype(int), 2, 4, weights=np.abs(q2 - 0.5) * 2)[0, 0] == 1
    # a reference split 8/8 has no majority and grades nothing
    ref = np.where(np.arange(16).reshape(4, 4) < 8, 1, 0)
    assert _pooled_argmax(ref, 2, 4, tie=-1)[0, 0] == -1
    r = assess_prediction(np.full((4, 4), 0.9), is_logit=False, reference=ref)["against_reference"]
    assert r["n_windows_scored"] == 0 and r["n_windows_reference_tied"] == 1

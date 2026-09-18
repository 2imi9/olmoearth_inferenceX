"""The assessor (oe_inferencex.assess): recipe items 1, 2 and 6 on a synthetic prediction, and its JSON view."""
import json

import numpy as np
import pytest

from oe_inferencex.assess import assess_classmap, assess_prediction, summary
from oe_inferencex.metrics import aurc_expected, capture_at_budget, oracle_aurc
from oe_inferencex.signals import boundary_indicator


def _synthetic(seed=0, size=64, patch=4):
    rng = np.random.default_rng(seed)
    truth = (np.add.outer(np.arange(size), np.arange(size)) > size).astype(int)   # a diagonal boundary
    logits = np.zeros((2, size, size))
    margin = np.abs(np.add.outer(np.arange(size), np.arange(size)) - size) / 8.0     # sure far from the line
    logits[1] = np.where(truth == 1, margin, -margin) + rng.normal(0, 0.6, (size, size))
    return logits, truth, patch


def test_assess_prediction_reports_the_recipe_quantities():
    logits, truth, patch = _synthetic()
    out = assess_prediction(logits, is_logit=True, patch=patch, reference=truth, budgets=(0.05, 0.10))
    G = logits.shape[1] // patch
    assert out["n_windows"] == G * G and out["n_classes"] == 2 and out["signal"] == "negative logit margin"
    arr = out["arrays"]
    assert np.array_equal(arr["boundary"], boundary_indicator(arr["pooled_argmax"]))     # recipe item 2, shared code
    for b in (0.05, 0.10):
        assert out["review_sets"][b]["n_windows"] == max(1, int(round(b * G * G)))      # recipe item 6
    rc = out["against_reference"]
    err = (arr["pooled_argmax"] != truth.reshape(G, patch, G, patch).mean(axis=(1, 3)).round().astype(int)).astype(float)
    assert rc["error_rate"] == pytest.approx(err.mean())
    assert rc["aurc_confidence"] == pytest.approx(aurc_expected(-arr["confidence"], err))
    assert rc["excess_aurc_confidence"] == pytest.approx(rc["aurc_confidence"] - oracle_aurc(err.size, int(err.sum())))
    assert rc["aurc_confidence"] < rc["aurc_random_expected"]                            # recipe item 1 on a synthetic map
    cap = capture_at_budget(-arr["confidence"], err, budgets=(0.05, 0.10))
    for b in (0.05, 0.10):
        assert rc["error_capture_at_budget"][b]["errors_captured_fraction"] == pytest.approx(cap[b])
    assert 0 < rc["boundary_share_among_errors"] <= 1


def test_probability_input_warns_and_classmap_reports_ties():
    logits, truth, patch = _synthetic()
    probs = 1 / (1 + np.exp(-logits[1]))
    out = assess_prediction(probs, is_logit=False, patch=patch)
    assert any("saturate" in w for w in out["warnings"])
    band = np.round(probs, 1)                                                              # a quantized confidence band
    cm = assess_classmap((probs > 0.5).astype(int), band, n_classes=2, patch=patch)
    assert cm["confidence_distinct_values"] <= 11 and 0 < cm["confidence_modal_share"] < 1


def test_nodata_and_summary_json():
    logits, truth, patch = _synthetic()
    nodata = np.zeros(truth.shape, bool)
    nodata[:16] = True
    out = assess_prediction(logits, is_logit=True, patch=patch, nodata_mask=nodata, reference=np.where(nodata, -1, truth))
    assert out["n_windows"] == 12 * 16 and out["against_reference"]["n_windows_scored"] == 12 * 16
    text = json.dumps(summary(out))
    assert "arrays" not in json.loads(text) and "review_sets" in json.loads(text)


def test_boundary_first_order_matches_exp36_construction_and_recorded_captures():
    """assess.boundary_first_score is exp36's lexicographic rule, and per-tile captures with it reproduce exp38's
    recorded 'boundary, then confidence' column from exp37's per-window table."""
    import csv
    import os

    import numpy as np
    import pytest

    from oe_inferencex.assess import boundary_first_score
    from oe_inferencex.metrics import capture_at_budget_expected
    from oe_inferencex.signals import midrank_pct

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")
    conf = np.array([[0.1, 0.5], [0.5, 0.9]])
    bnd = np.array([[0.0, 0.25], [0.0, 0.5]])
    s = boundary_first_score(-conf, bnd)
    assert s.tolist() == (np.where(bnd > 0, 2.0, 0.0) + midrank_pct(-conf).reshape(2, 2)).tolist()
    assert s[0, 1] > s[1, 1] > s[0, 0] > s[1, 0]                       # boundary windows first (least confident of them first), then the interior by confidence

    z = np.load(os.path.join(out, "exp37_patches_bolivia.npz"))
    rec = {r["unit"]: r for r in csv.DictReader(open(os.path.join(out, "exp38_ndwi_first.csv"))) if r["part"] == "B"}
    for tile in list(rec)[:40]:
        m = z["tile"] == int(tile)
        cap = capture_at_budget_expected(boundary_first_score(z["conf"][m], z["boundary"][m]), z["err"][m], (0.05, 0.1, 0.2))
        for b in (0.05, 0.1, 0.2):
            assert cap[b] == pytest.approx(float(rec[tile][f"boundary, then confidence @{b}"]), abs=1e-12)


def test_assess_prediction_review_order_option():
    import numpy as np
    import pytest

    from oe_inferencex.assess import assess_prediction, boundary_first_score, review_order

    rng = np.random.default_rng(0)
    logit = rng.normal(size=(32, 32))
    logit[:, :16] += 3                                                  # a confident half and a boundary down the middle
    ref = (logit > 0).astype(int)
    ref[8:12, 12:20] = 1 - ref[8:12, 12:20]                             # some errors near the boundary
    a = assess_prediction(logit, is_logit=True, patch=4, reference=ref, budgets=(0.1, 0.25))
    b = assess_prediction(logit, is_logit=True, patch=4, reference=ref, budgets=(0.1, 0.25), order="boundary_first")
    assert a["review_order"] == "confidence" and b["review_order"] == "boundary_first"
    arr = b["arrays"]
    score = boundary_first_score(-arr["confidence"], arr["boundary"])
    for bud, rs in b["review_sets"].items():
        k = rs["n_windows"]
        expected = set(map(tuple, np.stack(np.unravel_index(review_order(score, arr["valid"])[:k], score.shape), 1).tolist()))
        assert set(map(tuple, np.asarray(rs["windows_rowcol"]).tolist())) == expected
        assert rs["boundary_share_in_set"] >= a["review_sets"][bud]["boundary_share_in_set"]
    assert a["against_reference"]["aurc_confidence"] == b["against_reference"]["aurc_confidence"]   # AURC scores confidence
    with pytest.raises(ValueError):
        assess_prediction(logit, is_logit=True, order="random")


# --------------------------------------------------------------------------- defects found by audit, 2026-09-13
def test_nodata_pixels_do_not_inflate_a_window_confidence():
    """A pixel with no prediction carries no evidence about its window and must not vote.

    The pooling filled no-data pixels with the SCENE MAXIMUM margin before a plain mean, so a window that was 37.5%
    no-data read as nine times more confident than the same window fully observed (0.3625 against 0.04) and sank down
    the review list. No-data at scene edges and under cloud is everywhere in Earth observation."""
    p = np.full((8, 8), 0.95)
    p[0:4, 0:4] = 0.52                      # one genuinely uncertain window
    nod = np.zeros((8, 8), bool)
    nod[0:2, 0:3] = True                    # 6 of that window's 16 pixels have no prediction
    clean = assess_prediction(p, is_logit=False, patch=4)["arrays"]["confidence"]
    holed = assess_prediction(p, is_logit=False, patch=4, nodata_mask=nod)["arrays"]["confidence"]
    assert holed[0, 0] == pytest.approx(clean[0, 0], abs=1e-12), "no-data changed the confidence of observed pixels"
    assert holed[0, 0] < 0.2, "the uncertain window must stay uncertain"


def test_a_window_with_no_valid_pixels_is_excluded_not_scored():
    p = np.full((8, 8), 0.95)
    nod = np.zeros((8, 8), bool)
    nod[0:4, 0:4] = True
    out = assess_prediction(p, is_logit=False, patch=4, nodata_mask=nod)
    assert not bool(out["arrays"]["valid"][0, 0])
    assert out["n_windows"] == 3
    assert [0, 0] not in [list(w) for w in out["review_sets"][0.05]["windows_rowcol"]]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 200])
def test_a_budget_outside_zero_to_one_is_refused(bad):
    """--budgets 200 used to report '20000%: 51200' on a 256-window scene, which is not a budget."""
    p = np.random.default_rng(0).random((32, 32))
    with pytest.raises(ValueError, match="budget must be a fraction"):
        assess_prediction(p, is_logit=False, patch=4, budgets=(bad,))


def test_a_budget_of_one_reviews_every_window_and_no_more():
    p = np.random.default_rng(0).random((32, 32))
    out = assess_prediction(p, is_logit=False, patch=4, budgets=(1.0,))
    assert out["review_sets"][1.0]["n_windows"] == out["n_windows"] == 64


def test_assess_prediction_top1_form_on_multiclass_logits_and_the_warning_on_the_default():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((4, 32, 32)) * 2
    default = assess_prediction(z, is_logit=True)
    top1 = assess_prediction(z, is_logit=True, form="top1")
    assert any("form='top1'" in w for w in summary(default)["warnings"])
    assert not any("form='top1'" in w for w in summary(top1)["warnings"])
    assert summary(top1)["signal"] == "1 - max probability (from logits)"
    assert summary(default)["signal"] == "negative logit margin"
    assert set(summary(top1)["review_sets"]) == set(summary(default)["review_sets"])
    binary = assess_prediction(rng.standard_normal((32, 32)), is_logit=True, form="top1")
    assert not any("form='top1'" in w for w in summary(binary)["warnings"])
    with pytest.raises(ValueError):
        assess_prediction(z, is_logit=True, form="entropy")


@pytest.mark.parametrize("scores,why", [
    (np.full((32, 32), 80.0) + np.arange(32)[None, :], "a regression output, such as fuel moisture in percent"),
    (np.stack([np.full((32, 32), 120.0), np.full((32, 32), 30.0)]), "per-class scores that are not probabilities"),
    (np.full((32, 32), -0.2), "a negative value"),
])
def test_a_map_that_is_not_a_probability_map_is_refused_by_the_api_too(scores, why):
    """The command line refused these and the function did not, so an agent importing the package got a full,
    plausible review set for a regression raster: every pixel above 0.5 is 'class 1', every confidence a number."""
    with pytest.raises(ValueError, match="not a probability map"):
        assess_prediction(scores, is_logit=False)
    assert assess_prediction(scores, is_logit=True)["n_windows"] == 64, "logits are unbounded and stay accepted"


def test_the_range_check_ignores_nodata_and_nan():
    p = np.random.default_rng(0).random((32, 32))
    nodata = np.zeros((32, 32), bool)
    p[:4, :4], nodata[:4, :4] = -9999.0, True
    p[10, 10] = np.nan
    assert assess_prediction(p, is_logit=False, nodata_mask=nodata)["n_windows"] == 63
    with pytest.raises(ValueError, match="-9999"):
        assess_prediction(p, is_logit=False)       # the same array without its mask is refused, and says why


def test_a_review_set_decided_by_ties_says_so_and_a_separable_one_stays_silent():
    """A hard 0/1 mask passes every range check and carries no confidence: every window ties and the 'review set' is
    the bottom-right corner of the raster. The set is returned, with the count of ties that decided it."""
    hard = np.zeros((64, 64)); hard[:, 32:] = 1.0
    out = assess_prediction(hard, is_logit=False, budgets=(0.05,))
    tie = out["review_sets"][0.05]["tied_at_cutoff"]
    assert tie["inside"] == out["review_sets"][0.05]["n_windows"] and tie["outside"] > 0
    assert any("raster position" in w for w in out["warnings"])
    assert summary(out)["review_sets"]["0.05"]["tied_at_cutoff"] == tie

    smooth = np.random.default_rng(1).normal(0, 3, (64, 64))
    out = assess_prediction(smooth, is_logit=True, budgets=(0.05,))
    assert "tied_at_cutoff" not in out["review_sets"][0.05] and not any("raster position" in w for w in out["warnings"])

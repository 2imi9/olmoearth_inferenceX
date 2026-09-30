"""The assessor (oe_inferencex.assess): recipe items 1, 2 and 6 on a synthetic prediction, its JSON view, and the
input-condition layer."""
import importlib.util
import json
import os

import numpy as np
import pytest

from oe_inferencex.assess import (CLASS_SHARE_TEXT, MAX_CONDITIONS, RULE_TEXT, SCOPE_ASSESS, SCOPE_ASSESS_K,
                                  assess_classmap, assess_prediction, boundary_first_score, pool_condition,
                                  review_mask, review_order, summary)
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


def test_top1_confidence_array_is_on_the_scale_of_its_quantiles():
    """Review of 2026-09-23: with form='top1' the quantiles were probabilities and the array log-probabilities."""
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 1, (4, 32, 32))
    out = assess_prediction(logits, is_logit=True, form="top1")
    conf = out["arrays"]["confidence"][out["arrays"]["valid"]]
    q25 = out["confidence_quantiles"][0.25]
    assert 0 < conf.min() and conf.max() <= 1 and 0.2 < float(np.mean(conf <= q25)) < 0.3
    ref = assess_prediction(logits, is_logit=True, form="top1")
    assert all(np.array_equal(ref["review_sets"][b]["windows_rowcol"], out["review_sets"][b]["windows_rowcol"]) for b in out["review_sets"])


# --------------------------------------------------------------------------- the input-condition layer (1.4.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden", "condition_1_3_1")


def _layer_8x8():
    """Four windows of 4 x 4: 16 of 1; 10 of 2 against 6 of 0; 8 of 0 against 8 of 1; nothing recorded."""
    lay = np.empty((8, 8), dtype=np.int64)
    lay[:4, :4] = 1
    w = np.array([2] * 10 + [0] * 6)
    lay[:4, 4:] = w.reshape(4, 4)
    lay[4:, :4] = np.array([0, 1] * 8).reshape(4, 4)
    lay[4:, 4:] = np.array([-1, -5] * 8).reshape(4, 4)
    return lay


def test_pool_condition_known_answer():
    lay = _layer_8x8()
    got = pool_condition(lay, 4)
    assert got["grid"].tolist() == [[1, 2], [-1, -1]] and got["grid"].dtype == np.int64
    assert got["values"] == [1, 2] and got["n_split"] == 1 and got["n_no_code"] == 1

    # no-data pixels do not vote: five of the ten 2s and one 1 of the even split lose their vote
    pred = np.ones((8, 8), bool)
    pred[:4, 4:] = (np.array([False] * 5 + [True] * 11)).reshape(4, 4)      # 5 of 2 and 6 of 0 remain
    pred[4, 1] = False                                                        # 8 of 0 against 7 of 1
    got = pool_condition(lay, 4, predicted=pred)
    assert got["grid"].tolist() == [[1, 0], [0, -1]] and got["n_split"] == 0 and got["n_no_code"] == 1
    assert got["values"] == [0, 1]

    # a window at least half without a prediction is not a window of the map: -1 even where its few predicted
    # pixels agree, and not counted as split or as carrying no value
    pred = np.ones((8, 8), bool)
    pred[:4, :4] = np.array([False] * 9 + [True] * 7).reshape(4, 4)          # seven 1s still vote
    pred[4:, 4:6] = False
    pred[4, 6] = False
    got = pool_condition(lay, 4, predicted=pred)
    assert got["grid"].tolist() == [[-1, 2], [-1, -1]] and got["values"] == [2]
    assert got["n_no_code"] == 0 and got["n_split"] == 1

    # NaN, a negative value and a masked pixel are unrecorded; a float layer of whole numbers is read as integers
    f = lay.astype(float)
    f[:4, 4:] = np.where(lay[:4, 4:] == 2, np.nan, f[:4, 4:])                  # the 2s gone: 6 of 0 win
    assert pool_condition(f, 4)["grid"].tolist() == [[1, 0], [-1, -1]]
    m = np.ma.masked_array(lay, mask=np.zeros((8, 8), bool))
    m.mask[:4, :4] = True                                                     # the 1s of the first window masked
    got = pool_condition(m, 4)
    assert got["grid"].tolist() == [[-1, 2], [-1, -1]] and got["n_no_code"] == 2
    assert pool_condition(lay == 1, 4)["grid"].tolist() == [[1, 0], [-1, 0]]           # a bool cloud flag

    # through the assessor: the window grid is the one pool_condition gives with the map's valid pixels
    scores = np.random.default_rng(0).normal(size=(8, 8))
    nod = ~np.ones((8, 8), bool)
    nod[:4, 4:] = ~(np.array([False] * 5 + [True] * 11)).reshape(4, 4)
    out = assess_prediction(scores, is_logit=True, nodata_mask=nod, condition=lay)
    assert out["arrays"]["condition"].tolist() == pool_condition(lay, 4, predicted=~nod)["grid"].tolist()
    assert out["conditions"]["n_windows_split"] == 1 and out["conditions"]["n_windows_no_code"] == 1


def test_pool_condition_refusals():
    lay = _layer_8x8()
    with pytest.raises(ValueError, match="not integers"):
        pool_condition(np.where(lay == 1, 0.5, lay), 4)
    with pytest.raises(ValueError, match="one band"):
        pool_condition(np.stack([lay, lay]), 4)
    with pytest.raises(ValueError, match="pass layer"):
        pool_condition(lay[None], 4)
    with pytest.raises(ValueError, match="8 x 9 px and the map is 8 x 8"):
        pool_condition(np.zeros((8, 9), int), 4, predicted=np.ones((8, 8), bool))
    with pytest.raises(ValueError, match="one condition value per pixel"):
        assess_prediction(np.zeros((8, 8)) + 1.0, is_logit=True, condition=np.zeros((2, 2), int))  # a window grid
    with pytest.raises(ValueError, match="no window takes a condition"):
        pool_condition(np.full((8, 8), -1), 4)
    with pytest.raises(ValueError, match="4 are split evenly"):
        pool_condition(np.indices((8, 8)).sum(0) % 2, 4)                       # a checkerboard: every window tied
    with pytest.raises(ValueError, match="int32"):
        pool_condition(np.where(lay == 1, 3.4028235e38, lay), 4)               # a float32 no-data left in the layer

    v = np.arange(256).reshape(16, 16) % MAX_CONDITIONS
    v[:4, :4] = 0                                                             # one window won; 64 values in all
    assert pool_condition(v, 4)["values"] == [0]
    v = np.arange(256).reshape(16, 16) % (MAX_CONDITIONS + 1)
    v[:4, :4] = 0
    with pytest.raises(ValueError, match=f"65 distinct values .* at most {MAX_CONDITIONS}"):
        pool_condition(v, 4)

    s = np.random.default_rng(1).normal(size=(8, 8))
    with pytest.raises(ValueError, match="no condition was given"):
        assess_prediction(s, is_logit=True, condition_names={0: "clear"})
    for names, why in (({1: "unrecorded"}, "reserved"), ({1: "a", 2: "a"}, "unique"), ({1: ""}, "non-empty")):
        with pytest.raises(ValueError, match=why):
            assess_prediction(s, is_logit=True, condition=lay, condition_names=names)


def _golden_calls():
    """The six assess_* calls of tests/golden/condition_1_3_1/generate.py, on the same fixtures."""
    spec = importlib.util.spec_from_file_location("golden_1_3_1", os.path.join(GOLDEN, "generate.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    from oe_inferencex.demo import SAMPLE
    z = np.load(SAMPLE)
    dw = z["probs"].astype(np.float32)
    p, water = gen._scene()
    logits, probs3, truth3 = gen._three_class()
    hard3, band = probs3.argmax(0), probs3.max(0)
    nd3 = np.zeros(hard3.shape, bool)
    nd3[:6, :10] = True
    return {
        "api_prediction_dw": (assess_prediction, (dw,), dict(is_logit=False)),
        "api_prediction_scene": (assess_prediction, (p,), dict(is_logit=False, nodata_mask=np.isnan(p),
                                                               reference=water.astype(int), budgets=(0.05, 0.10),
                                                               order="boundary_first")),
        "api_prediction_logits3_margin": (assess_prediction, (logits,), dict(is_logit=True)),
        "api_prediction_logits3_top1": (assess_prediction, (logits,), dict(is_logit=True, form="top1", reference=truth3)),
        "api_classmap": (assess_classmap, (hard3, band, 3), dict(nodata_mask=nd3, reference=truth3)),
        "api_classmap_boundary_first": (assess_classmap, (hard3, np.round(band, 2), 3),
                                        dict(patch=8, order="boundary_first", budgets=(0.02, 0.2))),
    }


def test_without_a_condition_only_scope_is_added():
    """Against the summaries generated at 725dffa, before the layer was built: byte for byte once `scope` is taken
    out, and `scope` is SCOPE_ASSESS, outside the warnings."""
    calls = _golden_calls()
    assert sorted(calls) == json.load(open(os.path.join(GOLDEN, "manifest.json")))["api"]
    for name, (fn, args, kw) in calls.items():
        out = fn(*args, **kw)
        s = summary(out)
        assert s.pop("scope") == SCOPE_ASSESS, name
        assert SCOPE_ASSESS not in s["warnings"] and "conditions" not in s and "condition" not in out["arrays"], name
        with open(os.path.join(GOLDEN, f"{name}.json"), "rb") as f:
            assert (json.dumps(s, indent=1) + "\n").encode() == f.read(), name


def _two_condition_map(seed=3, size=48):
    """Logits, a no-data corner, and a layer: clear on the left, cloudy on the right, a strip with no value."""
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 1.5, (size, size))
    logit[:, : size // 2] += 2.0                                  # the clear half is more confident
    nod = np.zeros((size, size), bool)
    nod[:8, :8] = True
    lay = np.where(np.arange(size)[None, :] < size // 2, 0, 1) * np.ones((size, 1), int)
    lay[size - 4:, :] = -1                                         # the bottom row of windows records nothing
    return logit, nod, lay


@pytest.mark.parametrize("order", ["confidence", "boundary_first"])
def test_condition_review_sets(order):
    logit, nod, lay = _two_condition_map()
    kw = dict(is_logit=True, nodata_mask=nod, budgets=(0.02, 0.05, 0.2), order=order)
    plain = assess_prediction(logit, **kw)
    out = assess_prediction(logit, condition=lay, condition_names={0: "clear", 1: "cloudy"}, **kw)
    # the whole map is untouched: every key, the review sets, the warnings and the arrays
    a, b = summary(plain), summary(out)
    for s in (a, b):
        s.pop("scope")
    b.pop("conditions")
    assert a == b
    assert set(out["arrays"]) == set(plain["arrays"]) | {"condition"}
    for k in plain["arrays"]:
        np.testing.assert_array_equal(plain["arrays"][k], out["arrays"][k])       # NaN outside the map is equal

    arr = out["arrays"]
    score = -arr["confidence"] if order == "confidence" else boundary_first_score(-arr["confidence"], arr["boundary"])
    grid, valid = arr["condition"], arr["valid"]
    full = review_order(score, valid)
    per = out["conditions"]["per_condition"]
    assert list(per) == ["clear", "cloudy", "unrecorded"] and [e["value"] for e in per.values()] == [0, 1, None]
    for name, e in per.items():
        inc = valid & (grid == (-1 if e["value"] is None else e["value"]))
        assert e["n_windows"] == int(inc.sum()) > 0
        for bud, rs in e["review_sets"].items():
            m = review_mask(score, inc, bud)
            rc = np.asarray(rs["windows_rowcol"])
            got = np.zeros(grid.shape, bool)
            got[rc[:, 0], rc[:, 1]] = True
            assert (got == m).all() and rs["n_windows"] == int(m.sum()), (name, bud)
            in_order = [i for i in full if inc.ravel()[i]][: rs["n_windows"]]     # and in the whole map's order
            assert np.ravel_multi_index((rc[:, 0], rc[:, 1]), grid.shape).tolist() == in_order

    # a reference that grades nothing returns early; the block and the note are there all the same
    graded = assess_prediction(logit, condition=lay, reference=np.full(lay.shape, -1), **kw)
    assert graded["against_reference"]["n_windows_scored"] == 0
    assert "conditions" in graded and graded["scope"] == SCOPE_ASSESS_K.format(K=3)


def test_per_condition_blocks():
    rng = np.random.default_rng(4)
    logits = rng.normal(0, 2, (4, 48, 48))
    _, nod, lay = _two_condition_map()
    names = {0: "clear", 1: "cloudy", 7: "snow"}
    for form in ("top1", "margin"):
        out = assess_prediction(logits, is_logit=True, nodata_mask=nod, form=form, condition=lay, condition_names=names)
        blk = out["conditions"]
        assert blk["rule"] == RULE_TEXT and blk["class_share_status"] == CLASS_SHARE_TEXT and blk["source"] is None
        assert blk["notes"] == ["value 7 (snow) holds no window"] and blk["n_conditions"] == 3
        per, arr = blk["per_condition"], out["arrays"]
        assert sum(e["share_of_map"] for e in per.values()) == pytest.approx(1.0, abs=1e-12)
        for bud in out["review_sets"]:
            assert sum(e["share_of_review_set"][bud] for e in per.values()) == pytest.approx(1.0, abs=1e-12)
        for e in per.values():
            inc = arr["valid"] & (arr["condition"] == (-1 if e["value"] is None else e["value"]))
            assert e["n_windows"] == int(inc.sum()) and e["share_of_map"] == inc.sum() / arr["valid"].sum()
            hard = arr["pooled_argmax"][inc]
            assert e["class_share"] == {k: float(np.sum(hard == k) / hard.size) for k in range(4)}
            for bud, rs in out["review_sets"].items():
                idx = np.ravel_multi_index(tuple(np.asarray(rs["windows_rowcol"]).T), inc.shape)
                assert e["share_of_review_set"][bud] == float(inc.ravel()[idx].mean())
            conf = arr["confidence"][inc]
            for q, v in e["confidence_quantiles"].items():
                if form == "top1":       # the probability scale, as the whole map's quantiles: exp of the log quantile
                    assert 0 < v <= 1 and v == pytest.approx(float(np.exp(np.nanquantile(np.log(conf), q))), rel=1e-12)
                else:
                    assert v == float(np.nanquantile(conf, q))
        if form == "top1":
            assert out["confidence_scale"] and all(0 < v <= 1 for v in out["confidence_quantiles"].values())
    # the class-share note makes no claim a number could carry
    assert not any(ch.isdigit() for ch in CLASS_SHARE_TEXT) and "exp88" not in CLASS_SHARE_TEXT
    # a class map takes the layer too
    cm = assess_classmap(logits.argmax(0), logits.max(0), 4, nodata_mask=nod, condition=lay)
    assert list(cm["conditions"]["per_condition"]) == ["0", "1", "unrecorded"]
    assert json.loads(json.dumps(summary(cm)))["conditions"]["per_condition"]["unrecorded"]["value"] is None


def test_scope_notes_quote_exp88_as_recorded():
    """Every number the scope notes quote is exp88's, read from its summary (as test_explain reads exp37's)."""
    from oe_inferencex.estimate import SCOPE_CERTIFY, SCOPE_ESTIMATE
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp88_summary.json")))
    pre = rec["prereg"]
    assert rec["prereg_status"] == "frozen" and pre["graded_on"] == "olmoearth_base, probe seed 0"
    base = rec["results"]["olmoearth_base"]["seeds"]["0"]["families"]["pastis"]["conditions"]

    p2 = pre["P2"]                                  # errors at least as confident as the typical correct full-input window
    assert p2["confident_share_optical_missing"] == base["optical_missing"]["confident_errors"]["share"]
    assert p2["confident_share_full"] == base["full"]["confident_errors"]["share"]
    assert f"{100 * p2['confident_share_optical_missing']:.1f}% of the errors" in SCOPE_ASSESS            # 59.8%
    assert f"against {100 * p2['confident_share_full']:.1f}% with it" in SCOPE_ASSESS                     # 6.0%
    china = pre["replication_china6"]["P2"]         # the direction on China 6: the confident share fell
    assert china["holds"] is False and china["confident_share_optical_missing"] < china["confident_share_full"]
    assert "did not happen on CropHarvest China 6" in SCOPE_ASSESS
    assert "China_6" in rec["config"]["families"]["china6"]["optical_missing"]

    full_auc = base["full"]["ranking"]["margin_auroc"]
    missing_auc = pre["P3"]["per_family"]["pastis"]["margin_auroc"]
    assert missing_auc == base["optical_missing"]["ranking"]["margin_auroc"]
    assert f"fell from {full_auc:.2f} to {missing_auc:.2f}" in SCOPE_ASSESS_K                             # 0.83, 0.59

    # the mixed map: each part's truth, and the pooled estimate of random samples of 300. The record's number is
    # the mean over its draws, so the note says "on average": one sample of 300 gives k/300, and 46.9% is not one
    p4 = pre["P4"]
    draws = rec["results"]["olmoearth_base"]["seeds"]["0"]["mixed_map"]["estimation"]["pooled"]
    assert p4["pooled_mean_estimate"] == draws["mean_estimate"] and rec["config"]["draws"] > 1
    assert f"{100 * p4['truth']['cloudy']:.1f}% against {100 * p4['truth']['clear']:.1f}% on the rest" in SCOPE_ESTIMATE
    assert (f"random samples of {rec['config']['sample']} estimated {100 * p4['pooled_mean_estimate']:.1f}% on average"
            in SCOPE_ESTIMATE)
    assert "a random sample of" not in SCOPE_ESTIMATE
    assert "(exp88)" in SCOPE_CERTIFY and not any(ch.isdigit() for ch in SCOPE_CERTIFY.replace("exp88", ""))


def test_single_condition_has_no_scope():
    logit, nod, _ = _two_condition_map()
    kw = dict(is_logit=True, nodata_mask=nod, budgets=(0.05, 0.2))
    plain = assess_prediction(logit, **kw)
    out = assess_prediction(logit, condition=np.full(logit.shape, 3), condition_names={3: "clear"}, **kw)
    assert "scope" not in out
    blk = out["conditions"]
    assert blk["n_conditions"] == 1 and list(blk["per_condition"]) == ["clear"]
    e = blk["per_condition"]["clear"]
    assert e["value"] == 3 and e["n_windows"] == out["n_windows"] and e["share_of_map"] == 1.0
    assert e["confidence_quantiles"] == plain["confidence_quantiles"] and e["class_share"] == plain["class_share"]
    for bud, rs in plain["review_sets"].items():                      # one condition ranks as the whole map does
        assert e["share_of_review_set"][bud] == 1.0
        assert np.array_equal(e["review_sets"][bud]["windows_rowcol"], rs["windows_rowcol"])
    # one value and some windows with none are two conditions, as the sample counts them
    lay = np.full(logit.shape, 3)
    lay[-4:] = -1
    two = assess_prediction(logit, condition=lay, **kw)
    assert two["conditions"]["n_conditions"] == 2 and two["scope"] == SCOPE_ASSESS_K.format(K=2)

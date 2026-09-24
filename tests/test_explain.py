"""The explanation layer (oe_inferencex.explain): cues, enrichment, and the review-set explanation."""
import json

import numpy as np
import pytest

import os

from oe_inferencex.assess import assess_prediction, review_mask, review_order, summary
from oe_inferencex.explain import CUES, cooccurrence, cue_enrichment, derive_cues, explain_review_set, library_table, top_fraction

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _assessment():
    conf = np.array([[0.9, 0.8, 0.1, 0.7],
                     [0.6, 0.2, 0.3, 0.95],
                     [0.5, 0.4, 0.85, 0.75],
                     [0.65, 0.55, 0.45, 0.35]])
    bnd = np.zeros((4, 4)); bnd[0, 2] = 0.5; bnd[1, 1] = 0.25; bnd[3, 3] = 0.125
    valid = np.ones((4, 4), dtype=bool); valid[3, 0] = False
    order = np.argsort(np.where(valid, -conf, -np.inf).ravel(), kind="stable")[::-1]
    sets = {}
    for b in (0.2, 0.5):
        k = max(1, int(round(b * valid.sum())))
        rows, cols = np.unravel_index(order[:k], conf.shape)
        sets[b] = {"n_windows": k, "windows_rowcol": np.stack([rows, cols], 1)}
    return {"arrays": {"confidence": conf, "boundary": bnd, "valid": valid}, "review_sets": sets}


def test_derived_cues_follow_the_arrays():
    a = _assessment()
    cues = derive_cues(a, low_confidence_quantile=0.2)
    assert cues["boundary"].sum() == 3 and cues["boundary"][0, 2] and cues["boundary"][3, 3]
    assert not cues["low_confidence"][3, 0]                       # invalid windows never carry a cue
    conf = a["arrays"]["confidence"]
    cut = np.quantile(conf[a["arrays"]["valid"]], 0.2)
    assert cues["low_confidence"][conf <= cut].all() and cues["low_confidence"].sum() == 3


def test_top_fraction_and_cooccurrence():
    s = np.arange(16, dtype=float).reshape(4, 4)
    top = top_fraction(s, 0.25)
    assert top.sum() == 4 and top[3].all()
    tied = top_fraction(np.ones((4, 4)), 0.25)
    assert tied.all()                                             # ties included, never broken by position
    names, mat = cooccurrence({"a": top, "b": s >= 14}, mask=None)
    assert names == ["a", "b"] and mat.tolist() == [[4, 2], [2, 2]]


def test_cue_enrichment_by_hand_and_clustered():
    cue = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0], bool)
    err = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], float)
    r = cue_enrichment(cue, err, n_boot=0)
    assert r["share_errors"] == pytest.approx(2 / 3) and r["share_correct"] == pytest.approx(2 / 7)
    assert r["enrichment"] == pytest.approx((2 / 3) / (2 / 7)) and r["precision"] == pytest.approx(0.5)
    # Two clusters of five windows. Cluster 0 alone gives a ratio of (2/3) / (1/2) = 4/3; both clusters give 7/3;
    # cluster 1 alone has no error and is skipped. Every resample is one of these three, so the interval is exactly
    # [4/3, 7/3] and about a quarter of the draws are skipped; window-level resampling would give other values.
    rc = cue_enrichment(cue, err, clusters=np.repeat([0, 1], 5), n_boot=200, seed=1)
    assert rc["clustered"] and rc["n_boot"] + rc["boot_skipped"] == 200
    assert rc["boot_lo"] == pytest.approx(4 / 3) and rc["boot_hi"] == pytest.approx(7 / 3)
    assert 25 <= rc["boot_skipped"] <= 75
    r0 = cue_enrichment(cue, err, n_boot=0)
    assert "boot_lo" not in r0


def test_explain_review_set_rows_counts_and_unexplained_windows():
    a = _assessment()
    extra = np.zeros((4, 4), bool); extra[1, 5 - 4] = True; extra[3, 0] = True   # (1,1) and an invalid window
    out = explain_review_set(a, cues={"unstable": extra}, low_confidence_quantile=0.2)
    assert out["cues"] == ["boundary", "low_confidence", "unstable"] and out["n_windows"] == 15
    assert out["scene_share"]["unstable"] == pytest.approx(1 / 15)               # the invalid window does not count
    b = out["budgets"][0.2]
    assert b["n_windows"] == 3 and len(b["windows"]) == 3
    first = b["windows"][0]
    assert (first["row"], first["col"]) == (0, 2) and set(first["cues"]) == {"boundary", "low_confidence"}
    assert b["count"]["boundary"] + b["n_without_cue"] <= b["n_windows"]
    mat = np.array(b["cooccurrence"])
    assert (mat == mat.T).all() and mat[0, 0] == b["count"]["boundary"]
    assert "75% of error windows vs 21% of correct ones, 3.5x" in out["quotes"]["boundary"]
    assert "least confident 20%" in out["quotes"]["low_confidence"]
    assert "least confident 50%" in explain_review_set(a, low_confidence_quantile=0.5)["quotes"]["low_confidence"]
    assert "58% of error windows vs 16% of correct ones, 3.6x" in out["quotes"]["unstable"]   # exp37's shares
    assert "not measured" in explain_review_set(a, cues={"novel": extra})["quotes"]["novel"]     # a cue outside the library
    with pytest.raises(ValueError):
        explain_review_set(a, cues={"bad": np.zeros((2, 2), bool)})
    text = json.dumps(summary({"explanation": out}), allow_nan=False)             # JSON-safe through the assess summary
    assert json.loads(text)["explanation"]["budgets"]["0.2"]["n_windows"] == 3
    a["arrays"]["valid"][:] = False                                               # no valid window: shares undefined
    empty = explain_review_set(a)
    assert np.isnan(empty["scene_share"]["boundary"])
    assert json.loads(json.dumps(summary(empty), allow_nan=False))["scene_share"]["boundary"] is None


def test_library_numbers_trace_to_the_recorded_shares():
    """The expert-label cues carry exp37's shares on identical Bolivia windows (exp/out/exp37_summary.json); the
    boundary shares also equal exp36's run-of-record boundary_share."""
    rec = json.load(open(os.path.join(OUT, "exp37_summary.json")))["part_b"]["analysis"]["cues"]
    for name in ("boundary", "low_confidence", "unstable", "ndwi_ambiguous", "dihedral_disagree"):
        assert CUES[name].share_errors == pytest.approx(rec[name]["share_errors"], abs=5e-4), name
        assert CUES[name].share_correct == pytest.approx(rec[name]["share_correct"], abs=5e-4), name
    b36 = json.load(open(os.path.join(OUT, "exp36_summary.json")))["part_b"]["boundary_share"]
    assert CUES["boundary"].share_errors == pytest.approx(b36["errors"], abs=5e-4)
    assert CUES["boundary"].share_correct == pytest.approx(b36["correct"], abs=5e-4)
    assert CUES["ndwi_ambiguous"].enrichment == pytest.approx(0.483 / 0.067)
    rows = library_table(quantile=0.2)
    assert {r["name"] for r in rows} == set(CUES) and all(r["source"] and r["quote"] for r in rows)
    assert "not yet measured" in CUES["osm_disagrees"].quote()


def test_review_mask_reproduces_the_assessor_under_ties():
    """One review-order definition (assess.review_order): tied logits must give the assessor's review set, not a
    raster-order variant of it (Codex review of exp37)."""
    logit = np.zeros((16, 16))
    logit[:8] = 2.0                                  # two levels of margin, heavy ties
    logit[8:, :8] = -0.5
    logit[12:, 12:] = 0.5
    a = assess_prediction(logit, is_logit=True, patch=4, budgets=(0.2, 0.5))
    arr = a["arrays"]
    for b, rs in a["review_sets"].items():
        m = review_mask(-arr["confidence"], arr["valid"], b)
        rc = np.asarray(rs["windows_rowcol"])
        expected = np.zeros(arr["confidence"].shape, bool)
        expected[rc[:, 0], rc[:, 1]] = True
        assert (m == expected).all() and m.sum() == rs["n_windows"]
    order = review_order(np.array([[1.0, 1.0], [1.0, 0.0]]), np.ones((2, 2), bool))
    assert order.tolist() == [2, 1, 0, 3]            # ties by descending raster position, as the assessor sorts


def test_the_library_carries_exp82s_per_task_verifications_and_reports_the_maps_boundary_prevalence():
    """The verified shares in the library equal exp82's artifact, the quoted range is the artifact's, and the
    review-set explanation states the map's own boundary prevalence beside that range."""
    import json
    import os
    from oe_inferencex import explain
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(root, "exp", "out", "exp82_summary.json")))["tasks"]
    # the boundary cue is exp82's own; the low-confidence cue is exp82's per-tile cut, the scene-level cut the tool
    # draws (review of 2026-09-23: the pooled task-wide cut enriched 2.5x to 5.1x, the per-scene one 1.4x to 3.4x)
    for cue, key in (("boundary", "boundary"), ("low_confidence", "low_confidence_per_tile")):
        v = explain.CUES[cue].verified
        assert set(v) == set(d)
        for t, (se, sc) in v.items():
            assert abs(se - d[t]["cues"][key]["share_errors"]) < 1e-4 and abs(sc - d[t]["cues"][key]["share_correct"]) < 1e-4
    lo_c, hi_c = explain.CUES["low_confidence"].verified_range
    assert round(lo_c, 1) == 1.4 and round(hi_c, 1) == 3.4
    lo, hi = explain.CUES["boundary"].verified_range
    assert round(lo, 2) == 1.24 and round(hi, 2) == 8.13
    assert "1.2x to 8.1x" in explain.CUES["boundary"].quote()
    rng = np.random.default_rng(0)
    conf = rng.random((16, 16))
    bnd = rng.random((16, 16)) < 0.3
    assessment = {"arrays": {"confidence": conf, "boundary": bnd.astype(float), "valid": np.ones((16, 16), bool)}, "review_sets": {}}
    out = explain.explain_review_set(assessment)
    assert "boundary_prevalence_note" in out and f"{100 * bnd.mean():.0f}%" in out["boundary_prevalence_note"]


def test_confusion_pairs_by_brute_force_with_ties_and_unlabelled_windows():
    from collections import Counter
    from oe_inferencex.explain import confusion_pairs
    rng = np.random.default_rng(4)
    ref = rng.integers(-1, 4, 600)                                   # -1: no majority label
    dec = rng.integers(0, 4, 600)
    out = confusion_pairs(ref, dec, top=3)
    brute = Counter((int(d), int(r)) for r, d in zip(ref, dec) if r >= 0 and d != r)
    assert out["n_errors"] == sum(brute.values()) and out["n_pairs"] == len(brute)
    top = sorted(brute.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:3]
    assert [(p["predicted"], p["reference"], p["n"]) for p in out["pairs"]] == [(a, b, n) for (a, b), n in top]
    assert abs(out["share_of_errors_in_top"] - sum(n for _, n in top) / out["n_errors"]) < 1e-12
    none = confusion_pairs(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert none["n_errors"] == 0 and none["pairs"] == []
    with pytest.raises(ValueError):
        confusion_pairs(np.zeros(3), np.zeros(4))


def test_assess_against_a_reference_reports_the_confusion_pairs():
    from oe_inferencex.assess import assess_classmap
    rng = np.random.default_rng(1)
    H = W = 32
    hard = rng.integers(0, 3, (H, W))
    conf = rng.random((H, W))
    ref = hard.copy()
    ref[hard == 2] = np.where(rng.random((hard == 2).sum()) < 0.5, 0, 2)    # class 2 often called where the reference says 0
    out = assess_classmap(hard, conf, 3, patch=4, reference=ref)
    cp = out["against_reference"]["confusion_pairs"]
    assert cp["n_errors"] > 0 and cp["pairs"][0]["predicted"] == 2 and cp["pairs"][0]["reference"] == 0


def test_a_saturated_boundary_prevalence_is_named_as_no_reason():
    """exp82 on other encoders: on AnySat's cashew export 97% of windows sit on a boundary and the cue enriched
    0.99, so above BOUNDARY_SATURATED the note says the cue is at most a weak reason; below it, it does not."""
    from oe_inferencex import explain
    rng = np.random.default_rng(6)
    conf = rng.random((16, 16))
    for share, saturated in ((0.97, True), (0.3, False)):
        bnd = np.zeros(256, bool); bnd[: int(round(share * 256))] = True
        assessment = {"arrays": {"confidence": conf, "boundary": bnd.reshape(16, 16).astype(float), "valid": np.ones((16, 16), bool)},
                      "review_sets": {}}
        note = explain.explain_review_set(assessment)["boundary_prevalence_note"]
        assert ("weak reason" in note) is saturated
        assert (explain.BOUNDARY_SATURATED <= share) is saturated

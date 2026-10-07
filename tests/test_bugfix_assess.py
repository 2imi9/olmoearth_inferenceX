"""Defects in assess found by the bug hunt of 2026-10-05 and fixed on 2026-10-06, each pinned so it cannot return.

Each test states the user-visible consequence the defect had.
"""
import os

import numpy as np
import pytest

from oe_inferencex import cli
from oe_inferencex.assess import (assess_classmap, assess_prediction, pool_condition, review_order)


def _rowcols(out):
    return {b: rs["windows_rowcol"].tolist() for b, rs in out["review_sets"].items()}


# ----------------------------------------------------------------------------- a band's NaN no-data (bug 4)
def test_classmap_nan_confidence_is_no_data_and_never_ranked():
    """assess_classmap ranked an exported band's NaN windows first and counted them valid: 64 windows for 48."""
    rng = np.random.default_rng(0)
    hard = rng.integers(0, 3, (32, 32))
    conf = rng.uniform(0.4, 1.0, (32, 32))
    conf[:, :8] = np.nan                                   # the band's no-data strip, as exported
    out = assess_classmap(hard, conf, 3, patch=4, budgets=(0.05, 0.10, 0.5))
    assert out["n_windows"] == 48
    c = out["arrays"]["confidence"]
    for rc in _rowcols(out).values():
        rc = np.array(rc)
        assert np.isfinite(c[rc[:, 0], rc[:, 1]]).all() and (rc[:, 1] >= 2).all()
    assert out["warnings"][0] == ("256 pixels are not finite and were treated as no-data; pass nodata_mask to say so "
                                  "explicitly")
    assert not any("modal value nan" in w for w in out["warnings"])
    assert out["confidence_distinct_values"] == 768
    explicit = assess_classmap(hard, conf, 3, patch=4, budgets=(0.05, 0.10, 0.5), nodata_mask=~np.isfinite(conf))
    assert _rowcols(out) == _rowcols(explicit) and out["class_share"] == explicit["class_share"]
    assert not any("not finite" in w or "non-finite" in w for w in explicit["warnings"])


def test_classmap_nan_outside_a_given_mask_is_added_to_it():
    """A NaN pixel the caller's mask did not cover was ranked: its window went to the front of the review set."""
    rng = np.random.default_rng(1)
    hard = rng.integers(0, 2, (16, 16))
    conf = rng.uniform(0.5, 1.0, (16, 16))
    mask = np.zeros((16, 16), bool)
    mask[:4, :4] = True
    conf[8:12, 8:12] = np.nan                              # one whole window of NaN outside the mask
    out = assess_classmap(hard, conf, 2, patch=4, nodata_mask=mask, budgets=(0.1,))
    assert out["n_windows"] == 14
    assert "16 non-finite pixels outside the given nodata_mask were added to it" in out["warnings"]
    assert [2, 2] not in out["review_sets"][0.1]["windows_rowcol"].tolist()


# ----------------------------------------------------------------------------- a float reference (bug 16)
def test_float_reference_nan_is_no_label_and_values_round_as_the_command_line_does(tmp_path):
    """A NaN reference pixel was graded as class 0 on arm64 (16 windows scored at 0.9375 for 1 at 0); 0.9999 was class 0."""
    p = np.full((16, 16), 0.9)                             # class 1 everywhere
    ref = np.full((16, 16), np.nan)
    ref[:4, :4] = 1.0                                      # one labelled window, where the map is right
    rc = assess_prediction(p, is_logit=False, patch=4, reference=ref)["against_reference"]
    assert rc["n_windows_scored"] == 1 and rc["error_rate"] == 0.0
    rc = assess_classmap(np.ones((16, 16), int), np.full((16, 16), 0.9), 2, patch=4, reference=ref)["against_reference"]
    assert rc["n_windows_scored"] == 1 and rc["error_rate"] == 0.0
    near = np.full((16, 16), 0.9999)                       # class 1 written with float error
    rc = assess_prediction(p, is_logit=False, patch=4, reference=near)["against_reference"]
    assert rc["n_windows_scored"] == 16 and rc["error_rate"] == 0.0
    # the same arrays through the command line give the same numbers
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "ref.npy", ref)
    cli.main(["assess", str(tmp_path / "p.npy"), "--out", str(tmp_path / "o"), "--reference", str(tmp_path / "ref.npy")])
    import json
    s = json.load(open(tmp_path / "o" / "assessment.json"))["against_reference"]
    assert (s["n_windows_scored"], s["error_rate"]) == (1, 0.0)


# ----------------------------------------------------------------------------- an integer nodata_mask (bug 18)
@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float32])
def test_integer_nodata_mask_is_read_as_boolean(dtype):
    """A 0/1 uint8 mask was inverted bitwise: masked windows counted valid and filled the review set with NaN windows."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (32, 32))
    m = np.zeros((32, 32), bool)
    m[:, :8] = True
    want = assess_prediction(p, is_logit=False, patch=4, nodata_mask=m)
    got = assess_prediction(p, is_logit=False, patch=4, nodata_mask=m.astype(dtype))
    assert got["n_windows"] == want["n_windows"] == 48 and _rowcols(got) == _rowcols(want)
    hard, conf = (p > 0.5).astype(int), np.abs(p - 0.5) * 2
    want = assess_classmap(hard, conf, 2, patch=4, nodata_mask=m)
    got = assess_classmap(hard, conf, 2, patch=4, nodata_mask=m.astype(dtype))
    assert got["n_windows"] == want["n_windows"] == 48 and _rowcols(got) == _rowcols(want)
    assert got["confidence_distinct_values"] == want["confidence_distinct_values"]


# ----------------------------------------------------------------------------- ties at the cut-off (bug 43)
@pytest.mark.parametrize("order", ["confidence", "boundary_first"])
def test_cutoff_tie_is_found_when_equal_window_means_differ_in_the_last_bit(order):
    """Windows holding the same values in another pixel order pooled to 0.675 and 0.6749999999999999: no tie was said."""
    win0 = np.array([[0.6, 0.6], [0.7, 0.8]])
    win1 = np.array([[0.6, 0.7], [0.6, 0.8]])              # the same four stored values, the 0.7 one pixel over
    conf = np.hstack([win0, win1, np.full((2, 2), 0.9)])   # 1 x 3 windows of 2 px
    out = assess_classmap(np.zeros(conf.shape, int), conf, 2, patch=2, budgets=(1 / 3,), order=order)
    c = out["arrays"]["confidence"]
    assert c[0, 0] != c[0, 1]                              # the float means differ, as they did before the fix
    assert out["review_sets"][1 / 3]["tied_at_cutoff"] == {"inside": 1, "outside": 1}
    assert any("share the cut-off score with 1 windows left outside" in w for w in out["warnings"])
    assert out["confidence_distinct_pooled"] == 2


@pytest.mark.parametrize("order", ["confidence", "boundary_first"])
def test_cutoff_ties_of_a_quantized_band_match_exact_arithmetic(order):
    """On bands quantized to 0.1 a fifth of the real cut-off ties went unreported and others were miscounted."""
    real = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tenths = rng.choice([6, 7, 8, 9], size=(64, 64), p=[0.05, 0.15, 0.3, 0.5])
        out = assess_classmap(rng.integers(0, 3, (64, 64)), tenths / 10.0, 3, patch=4, budgets=(0.01, 0.05, 0.1),
                              order=order)
        exact = tenths.reshape(16, 4, 16, 4).sum(axis=(1, 3))           # each window's sum in tenths, exactly
        side = out["arrays"]["boundary"] > 0
        assert out["confidence_distinct_pooled"] == np.unique(exact).size
        for b, rs in out["review_sets"].items():
            rc = rs["windows_rowcol"]
            r, c = rc[-1]
            tied = exact == exact[r, c]
            if order == "boundary_first":
                tied &= side == side[r, c]
            inside = int(tied[rc[:, 0], rc[:, 1]].sum())
            outside = int(tied.sum()) - inside
            real += outside > 0
            assert rs.get("tied_at_cutoff") == ({"inside": inside, "outside": outside} if outside else None), (seed, b)
    assert real > 20


def test_the_review_order_and_distinct_scores_are_unchanged_by_the_tie_tolerance():
    """The tolerance is the float error of a window mean: it changes no review set, and scores 1e-9 apart stay distinct."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, (32, 32))
    out = assess_prediction(p, is_logit=False, patch=4, budgets=(0.05, 0.10))
    rank = review_order(-out["arrays"]["confidence"], out["arrays"]["valid"])
    for b, rs in out["review_sets"].items():
        assert rs["windows_rowcol"].tolist() == np.stack(np.unravel_index(rank[:rs["n_windows"]], (8, 8)), 1).tolist()
        assert "tied_at_cutoff" not in rs
    conf = np.hstack([np.full((2, 2), 0.5), np.full((2, 2), 0.5 + 1e-9), np.full((2, 2), 0.9)])
    out = assess_classmap(np.zeros(conf.shape, int), conf, 2, patch=2, budgets=(1 / 3,))
    assert "tied_at_cutoff" not in out["review_sets"][1 / 3] and out["confidence_distinct_pooled"] == 3


# ----------------------------------------------------------------------------- the non-finite count (bug 44)
def test_nonfinite_pixels_added_to_a_given_mask_are_counted():
    """The warning always said 0 non-finite pixels were added to the given nodata_mask, whatever the number."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, (16, 16))
    mask = np.zeros((16, 16), bool)
    mask[0, 0] = True
    p[5, 5:9] = np.nan
    out = assess_prediction(p, is_logit=False, patch=4, nodata_mask=mask)
    assert "4 non-finite pixels outside the given nodata_mask were added to it" in out["warnings"]
    z = rng.normal(0, 1, (3, 16, 16))
    z[1, 2, :7] = -np.inf
    out = assess_prediction(z, is_logit=True, patch=4, nodata_mask=mask)
    assert "7 non-finite pixels outside the given nodata_mask were added to it" in out["warnings"]


# ----------------------------------------------------------------------------- a window below one pixel (bug 45)
@pytest.mark.parametrize("patch", [0, -4, 2.5])
def test_a_window_below_one_pixel_is_a_named_refusal(patch, tmp_path):
    """--patch 0 ended in a ZeroDivisionError traceback and --patch -4 in numpy's 'can only specify one unknown dimension'."""
    p = np.random.default_rng(0).uniform(0, 1, (16, 16))
    msg = "the window must be a whole number of pixels, at least 1"
    with pytest.raises(ValueError, match=msg):
        assess_prediction(p, is_logit=False, patch=patch)
    with pytest.raises(ValueError, match=msg):
        assess_classmap((p > 0.5).astype(int), p, 2, patch=patch)
    with pytest.raises(ValueError, match=msg):
        pool_condition(np.zeros((16, 16), int), patch)
    if isinstance(patch, int):
        np.save(tmp_path / "p.npy", p)
        with pytest.raises(SystemExit, match=f"assess: --patch must be at least 1, got {patch}"):   # the CLI checks first
            cli.main(["assess", str(tmp_path / "p.npy"), "--out", str(tmp_path / "o"), f"--patch={patch}"])
        assert not os.path.exists(tmp_path / "o")


# ----------------------------------------------------------------------------- a fully masked class map (bug 46)
def test_classmap_on_a_fully_masked_scene_gives_the_named_refusal():
    """assess_classmap raised numpy's 'zero-size array to reduction operation maximum' on a fully clouded scene."""
    hard, conf = np.zeros((16, 16), int), np.full((16, 16), 0.7)
    with pytest.raises(ValueError, match="the map has no valid window"):
        assess_classmap(hard, conf, 2, patch=4, nodata_mask=np.ones((16, 16), bool))
    with pytest.raises(ValueError, match="the map has no valid window"):
        assess_classmap(hard, np.full((16, 16), np.nan), 2, patch=4)    # a band that is no-data everywhere

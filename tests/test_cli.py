"""The command line on synthetic rasters and arrays: files written, JSON consistent with the API, coordinates right."""
import csv
import json
import os

import numpy as np
import pytest

from oe_inferencex.assess import assess_prediction, summary
from oe_inferencex.cli import main

rasterio = pytest.importorskip("rasterio", reason="the raster path needs rasterio; the .npy path is tested below regardless")


def _scene(seed=0, size=128):
    """A binary probability map with a water blob, a boundary, saturated probabilities and a nodata corner."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    water = ((yy - 60) ** 2 + (xx - 70) ** 2) < 35 ** 2
    p = np.where(water, 0.9, 0.1) + rng.normal(0, 0.15, (size, size))
    p = np.clip(p, 0, 1)
    p[:8, :8] = np.nan
    return p.astype(np.float32), water


def _write(path, arr, nodata=None, dtype=None):
    from rasterio.transform import from_origin
    arr = np.asarray(arr)
    bands = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0], dtype=dtype or bands.dtype,
                       crs="EPSG:32633", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=nodata) as dst:
        dst.write(bands)


def test_assess_writes_the_files_and_matches_the_api(tmp_path):
    p, water = _scene()
    _write(tmp_path / "p.tif", p)
    _write(tmp_path / "ref.tif", water.astype("int16"), dtype="int16")
    out = tmp_path / "out"
    assert main(["assess", str(tmp_path / "p.tif"), "--out", str(out), "--reference", str(tmp_path / "ref.tif"), "--budgets", "0.05", "0.10"]) == 0
    s = json.load(open(out / "assessment.json"))
    api = summary(assess_prediction(p, is_logit=False, nodata_mask=np.isnan(p), reference=water.astype(int), budgets=(0.05, 0.10)))
    assert s["n_windows"] == api["n_windows"] and s["review_sets"]["0.05"]["n_windows"] == api["review_sets"]["0.05"]["n_windows"]
    assert s["against_reference"]["error_capture_at_budget"] == api["against_reference"]["error_capture_at_budget"]
    assert s["n_windows"] == 32 * 32 - 4                                        # the 8 x 8 nodata corner removes four 4-px windows
    rows = list(csv.DictReader(open(out / "review_set_05pct.csv")))
    assert len(rows) == s["review_sets"]["0.05"]["n_windows"] and rows[0]["rank"] == "1"
    r0 = rows[0]
    assert int(r0["pixel_row"]) == 4 * int(r0["window_row"]) and float(r0["x"]) == 500000.0 + 10.0 * (int(r0["pixel_col"]) + 2)
    with rasterio.open(out / "suspicion.tif") as src:
        assert src.shape == (32, 32) and src.transform.a == 40.0 and src.crs.to_epsg() == 32633
        sus = src.read(1)
    assert np.isnan(sus[0, 0]) and np.isfinite(sus[16, 16])
    exp = json.load(open(out / "explanation.json"))
    assert set(exp["cues"]) == {"boundary", "low_confidence"} and "0.05" in exp["budgets"]
    assert s["files"]["explanation"].endswith("explanation.json")


def test_assess_accepts_npy_without_geo(tmp_path):
    p, _ = _scene(seed=1)
    np.save(tmp_path / "p.npy", p)
    out = tmp_path / "out"
    assert main(["assess", str(tmp_path / "p.npy"), "--out", str(out)]) == 0
    s = json.load(open(out / "assessment.json"))
    assert s["n_windows"] == 32 * 32 - 4 and os.path.exists(out / "suspicion.npy")
    rows = list(csv.DictReader(open(out / "review_set_01pct.csv")))
    assert rows and rows[0]["x"] == ""


def test_compare_two_maps_with_labels_and_groups(tmp_path):
    p, water = _scene()
    q = p.copy()
    q[40:80, 100:120] = 0.95                                                    # b calls water where a and the label do not
    _write(tmp_path / "a.tif", p)
    _write(tmp_path / "b.tif", q)
    _write(tmp_path / "lab.tif", water.astype("int16"), dtype="int16")
    groups = np.zeros(p.shape, "int16"); groups[:, 64:] = 1
    _write(tmp_path / "g.tif", groups, dtype="int16")
    out = tmp_path / "cmp"
    assert main(["compare", str(tmp_path / "a.tif"), str(tmp_path / "b.tif"), "--out", str(out), "--labels", str(tmp_path / "lab.tif"), "--groups", str(tmp_path / "g.tif")]) == 0
    s = json.load(open(out / "comparison.json"))
    assert 35 <= s["n_disagree"] <= 60 and s["disagreement_rate"] > 0.03                # the 40 x 20 px block is 50 windows before pooling and noise
    assert s["graded"]["which_side"]["share_a_right"] > 0.9                    # the added water is b's error
    assert set(s["per_group"]) == {"0", "1"} and s["per_group"]["1"]["n_disagree"] >= s["n_disagree"] * 0.9
    assert s["where"]["boundary_b"]["enrichment"] > 1
    with rasterio.open(out / "disagreement.tif") as src:
        d = src.read(1)
    assert int(d.sum()) == s["n_disagree"] and len(list(csv.DictReader(open(out / "differing_windows.csv")))) == s["n_disagree"]


def test_compare_refuses_different_grids(tmp_path):
    p, _ = _scene()
    _write(tmp_path / "a.tif", p)
    _write(tmp_path / "b.tif", p[:64, :64])
    with pytest.raises(SystemExit):
        main(["compare", str(tmp_path / "a.tif"), str(tmp_path / "b.tif"), "--out", str(tmp_path / "x")])


# --------------------------------------------------------------------------- defects found by audit, 2026-09-13
def test_labels_are_pooled_over_their_own_classes_not_the_maps(tmp_path):
    """The output that says WHICH INFERENCE TO BELIEVE was graded against a fabricated label.

    cmd_compare pooled the label raster with n_classes taken from the two inference maps, and _pooled_argmax counts
    votes only over range(n_classes): every label pixel of a class the maps never predict was silently dropped and the
    window label fell to a surviving low index. Measured at 44.9% of window labels wrong, exit 0, no warning."""
    rng = np.random.default_rng(0)
    np.save(tmp_path / "a.npy", rng.integers(0, 3, (64, 64)))
    np.save(tmp_path / "b.npy", rng.integers(0, 3, (64, 64)))
    np.save(tmp_path / "lab.npy", rng.integers(0, 6, (64, 64)))     # six label classes, three predicted
    out = tmp_path / "o"
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(out),
                 "--labels", str(tmp_path / "lab.npy")]) == 0
    s = json.load(open(out / "comparison.json"))
    assert s.get("notes"), "a label raster with classes the maps cannot predict must say so"
    assert "6 classes" in s["notes"][0] and "at most 3" in s["notes"][0]


def test_two_budgets_that_differ_write_two_files(tmp_path):
    """0.001 and 0.004 both wrote review_set_00pct.csv; the second destroyed the first while the JSON named both."""
    np.save(tmp_path / "p.npy", np.random.default_rng(0).random((64, 64)))
    out = tmp_path / "o"
    assert main(["assess", str(tmp_path / "p.npy"), "--out", str(out),
                 "--budgets", "0.001", "0.004", "0.05"]) == 0
    files = sorted(f.name for f in out.iterdir() if f.name.startswith("review_set"))
    assert len(files) == 3, files
    assert "review_set_05pct.csv" in files, "whole-percent budgets must keep their established name"
    s = json.load(open(out / "assessment.json"))
    paths = [v for k, v in s["files"].items() if k.startswith("review_set")]
    assert len(set(paths)) == len(paths), "two budgets must not be recorded as one file"


@pytest.mark.parametrize("arr,why", [
    (np.arange(6).reshape(1, 6).repeat(64, 0)[:, :64] % 6, "a hard class map"),
    (np.full((64, 64), 3.0), "values outside [0, 1]"),
])
def test_assess_refuses_a_file_that_is_not_a_probability_map(tmp_path, arr, why):
    """A class map took the probability branch, thresholded at 0.5 and produced a confidence of 9.0 for class 5,
    returning a full plausible review set with exit 0. Silence is worse than absence when a field team acts on it."""
    np.save(tmp_path / "x.npy", arr.astype(float))
    with pytest.raises(SystemExit) as e:
        main(["assess", str(tmp_path / "x.npy"), "--out", str(tmp_path / "o")])
    assert "probability map" in str(e.value) or "class map" in str(e.value), (why, str(e.value))


def test_an_undefined_rate_is_not_printed_as_zero(capsys, tmp_path):
    """`None or 0` printed 0.00%, which reads as 'they never disagree' when the truth is 'this was not computable'."""
    from oe_inferencex.cli import _pct
    assert _pct(None) == "undefined"
    assert _pct(0.0) == "0.00%"
    assert _pct(0.5, 0) == "50%"


def test_multiband_scores_outside_zero_one_are_refused_cleanly(tmp_path):
    """The 3-D branch skipped the range check, so a multi-band raster that is not per-class probabilities was scored."""
    np.save(tmp_path / "x.npy", np.stack([np.full((64, 64), 120.0), np.full((64, 64), 30.0)]))
    with pytest.raises(SystemExit) as e:
        main(["assess", str(tmp_path / "x.npy"), "--out", str(tmp_path / "o")])
    assert "probability map" in str(e.value)


def test_compare_refuses_continuous_maps_at_the_default_cutoff_and_takes_a_named_one(tmp_path):
    """Two regression outputs cut at 0.5 are one class everywhere, so they 'never differ', with exit 0. With the cut-off
    named the comparison runs, is about that one decision, and says that no recorded experiment grades it."""
    rng = np.random.default_rng(0)
    a = 80 + 25 * rng.standard_normal((64, 64)); b = a + 10 * rng.standard_normal((64, 64))
    np.save(tmp_path / "a.npy", a); np.save(tmp_path / "b.npy", b)
    with pytest.raises(SystemExit) as e:
        main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(tmp_path / "o")])
    assert "--threshold" in str(e.value)
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(tmp_path / "o"), "--threshold", "80"]) == 0
    s = json.load(open(tmp_path / "o" / "comparison.json"))
    assert 0 < s["disagreement_rate"] < 1 and any("continuous map cut at 80" in n for n in s["notes"])


def test_compare_on_probability_maps_is_unchanged_by_the_optional_cutoff(tmp_path):
    p, _ = _scene(0); q, _ = _scene(1)
    p, q = np.nan_to_num(p, nan=0.1), np.nan_to_num(q, nan=0.1)
    np.save(tmp_path / "a.npy", p); np.save(tmp_path / "b.npy", q)
    main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(tmp_path / "d")])
    main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(tmp_path / "e"), "--threshold", "0.5"])
    d, e = (json.load(open(tmp_path / k / "comparison.json")) for k in ("d", "e"))
    assert d["disagreement_rate"] == e["disagreement_rate"] and "notes" not in d


# ----------------------------------------------------------------------------- sample / estimate
def _sample_map(tmp_path):
    """The real Dynamic World tile the package ships, as a (9, H, W) probability .npy and its expert labels."""
    z = np.load(os.path.join(os.path.dirname(__file__), "..", "oe_inferencex", "sample", "dynamic_world_tile.npz"))
    probs, expert = z["probs"].astype(np.float32), z["expert"].astype(int)
    path = tmp_path / "dw.npy"
    np.save(path, probs)
    return str(path), probs, expert


def _fill(csv_path, probs, expert, patch=4):
    """Label the sampled windows the way a reviewer would: wrong = 1 where the map's pooled class is not the expert's."""
    from oe_inferencex.assess import _pooled_argmax
    hard = _pooled_argmax(probs.argmax(0), probs.shape[0], patch)
    ref = _pooled_argmax(np.where(expert >= 0, expert, -1), probs.shape[0], patch, empty=-1)
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        r["wrong"] = str(int(hard[i, j] != ref[i, j]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return hard, ref


def test_sample_then_estimate_round_trip_covers_the_true_rate_of_the_shipped_map(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", path, "--budget", "200", "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "s.json"))
    assert side["design"] == "confidence" and len(side["indices"]) == 200 and side["n_population"] == 32 * 32
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 200 and all(r["wrong"] == "" for r in rows) and all(r["stratum"] != "" for r in rows)
    hard, ref = _fill(out, probs, expert)
    assert main(["estimate", str(out)]) == 0
    r = json.load(open(tmp_path / "s_estimate.json"))
    truth = float((hard != ref).mean())
    assert r["n_labelled"] == 200 and r["low"] <= truth <= r["high"], (r, truth)
    assert r["method"].startswith("stratified") and 0.02 < r["half_width"] < 0.12


def test_estimate_refuses_unlabelled_rows_and_a_csv_that_does_not_match_its_design(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "s.csv"
    main(["sample", path, "--budget", "50", "--design", "random", "--out", str(out)])
    with pytest.raises(SystemExit, match="have no `wrong` value"):
        main(["estimate", str(out)])
    _fill(out, probs, expert)
    rows = list(csv.DictReader(open(out)))
    rows[0], rows[1] = rows[1], rows[0]                                       # reorder: no longer the design's rows
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="do not match the design"):
        main(["estimate", str(out)])


def test_tiles_design_on_the_cli_reports_the_naive_interval_beside_the_honest_one(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "t.csv"
    assert main(["sample", path, "--budget", "160", "--design", "tiles", "--tile", "8", "--per-tile", "16", "--out", str(out)]) == 0
    side = json.load(open(tmp_path / "t.json"))
    assert side["n_tiles"] == 10 and len(side["indices"]) == 160
    _fill(out, probs, expert)
    assert main(["estimate", str(out)]) == 0
    r = json.load(open(tmp_path / "t_estimate.json"))
    assert "naive_interval_if_treated_as_random" in r and "warning" in r and r["n_tiles"] == 10


def test_estimate_refuses_half_labels_other_delimiters_and_says_so_without_a_traceback(tmp_path):
    """int(float("0.5")) was 0, so a reviewer's "not sure" counted as right with exit 0; a semicolon CSV was a KeyError."""
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "s.csv"
    main(["sample", path, "--budget", "40", "--design", "random", "--out", str(out)])
    _fill(out, probs, expert)
    rows = list(csv.DictReader(open(out)))
    rows[3]["wrong"] = "0.5"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="exactly 1 or 0.*row 5"):
        main(["estimate", str(out)])
    rows[3]["wrong"] = "2"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="exactly 1 or 0"):
        main(["estimate", str(out)])
    rows[3]["wrong"] = "1"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";"); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="another delimiter"):
        main(["estimate", str(out)])


def test_sample_refuses_a_budget_beyond_the_map_and_a_tile_grid_too_small_for_it_as_messages(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    with pytest.raises(SystemExit, match="budget must be between"):
        main(["sample", path, "--budget", "5000", "--out", str(tmp_path / "x.csv")])
    with pytest.raises(SystemExit, match="at least 5|can label at most"):
        main(["sample", path, "--budget", "300", "--design", "tiles", "--tile", "16", "--out", str(tmp_path / "y.csv")])


def test_sample_sidecar_carries_the_crs_and_ground_units_for_a_georeferenced_map(tmp_path):
    p, _ = _scene()
    path = tmp_path / "p.tif"
    _write(path, p)
    assert main(["sample", str(path), "--budget", "50", "--design", "random", "--out", str(tmp_path / "g.csv")]) == 0
    side = json.load(open(tmp_path / "g.json"))
    assert side["crs"] and side["pixel_size"] == [10.0, 10.0] and side["window_size_ground_units"] == [40.0, 40.0]
    assert side["xy_are"].startswith("window centres")
    rows = list(csv.DictReader(open(tmp_path / "g.csv")))
    assert rows[0]["x"] != "" and rows[0]["y"] != ""

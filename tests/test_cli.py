"""The command line on synthetic rasters and arrays: files written, JSON consistent with the API, coordinates right."""
import csv
import importlib.util
import json
import os

import numpy as np
import pytest

from oe_inferencex.assess import _pooled_argmax, assess_prediction, summary
from oe_inferencex.cli import main
from oe_inferencex.demo import SAMPLE

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
    # NaN where nothing was compared (the scene's no-data corner), so a GIS cannot read "not compared" as "agree"
    assert int(np.nansum(d)) == s["n_disagree"] and len(list(csv.DictReader(open(out / "differing_windows.csv")))) == s["n_disagree"]
    assert np.isnan(d).sum() == d.size - s["n_windows"]


def test_compare_reads_dates_and_refuses_cross_date_grading_without_the_labels_date(tmp_path):
    """Across dates a difference can be real change; the JSON says so, and grading against one reference needs the
    reference's date. Without dates the comparison runs as before and notes that they were not given."""
    p, water = _scene()
    q = p.copy()
    q[40:80, 100:120] = 0.95
    _write(tmp_path / "a.tif", p)
    _write(tmp_path / "b.tif", q)
    _write(tmp_path / "lab.tif", water.astype("int16"), dtype="int16")
    args = ["compare", str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
    assert main(args + ["--out", str(tmp_path / "u")]) == 0
    u = json.load(open(tmp_path / "u" / "comparison.json"))
    assert u["dates"]["status"] == "unstated" and any("--date-a" in n for n in u["notes"])
    with pytest.raises(SystemExit, match="--labels-date"):
        main(args + ["--out", str(tmp_path / "x"), "--labels", str(tmp_path / "lab.tif"), "--date-a", "2017-03-11", "--date-b", "2018-03-11"])
    with pytest.raises(SystemExit, match="not an ISO date"):
        main(args + ["--out", str(tmp_path / "y"), "--date-a", "March 2017", "--date-b", "2018-03-11"])
    assert main(args + ["--out", str(tmp_path / "c"), "--labels", str(tmp_path / "lab.tif"), "--date-a", "2017-03-11",
                        "--date-b", "2018-03-11", "--labels-date", "2018-03-11"]) == 0
    c = json.load(open(tmp_path / "c" / "comparison.json"))
    assert c["dates"]["status"] == "different_time" and c["dates"]["days_apart"] == 365
    assert c["inputs"]["labels_date"] == "2018-03-11" and "date of map b" in c["graded"]["graded_against"]
    assert c["n_disagree"] == u["n_disagree"]                                   # the dates change the reading, not the counts
    assert not any("--date-a" in n for n in c.get("notes", []))


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
    assert any("[3, 4, 5]" in n and "neither map predicts" in n for n in s["notes"])


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
    # no note about the cutoff; the only note on an undated comparison is that its dates were not given
    assert d["disagreement_rate"] == e["disagreement_rate"] and d["notes"] == [n for n in d["notes"] if "--date-a" in n] and len(d["notes"]) == 1


# ----------------------------------------------------------------------------- sample / estimate
def _sample_map(tmp_path):
    """The real Dynamic World tile the package ships, as a (9, H, W) probability .npy and its expert labels."""
    z = np.load(SAMPLE)                   # the installed package's copy, so the test also runs against a built wheel
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


def test_the_sample_csv_shows_the_class_the_tool_grades_so_a_reviewer_labels_the_same_thing(tmp_path, capsys):
    """Release check of 24 September: the CSV told the reviewer to judge "the map's class there" without showing it,
    and the tool's class for a window is the majority of its pixels' classes. A reviewer reading a mixed window
    another way (a mean probability, the centre pixel) produced `wrong` values the per-class table contradicted.
    The CSV now carries `map_class`; labels filled from it agree with what `estimate --per-class` grades."""
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "m.csv"
    assert main(["sample", path, "--budget", "300", "--design", "random", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0])[-2:] == ["map_class", "wrong"]
    tool = assess_prediction(probs, is_logit=False, patch=4)["arrays"]["pooled_argmax"]
    assert all(int(r["map_class"]) == tool[int(r["window_row"]), int(r["window_col"])] for r in rows)
    hard, _ = _fill_with_classes(out, probs, expert)
    # the helpers' class breaks an 8-8 window toward the lower index, the tool's toward the more confident pixels:
    # a reviewer could not have inferred the graded class on those windows without the column
    assert any(int(r["map_class"]) != hard[int(r["window_row"]), int(r["window_col"])] for r in rows)
    rows = list(csv.DictReader(open(out)))
    for r in rows:                                     # the reviewer fills `wrong` from the column alone
        r["wrong"] = str(int(int(r["map_class"]) != int(r["reference_class"])))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    capsys.readouterr()
    assert main(["estimate", str(out), "--per-class"]) == 0
    assert "disagrees" not in capsys.readouterr().out
    assert "map_class" in json.load(open(tmp_path / "m.json"))["how_to_label"]


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
    with pytest.raises(SystemExit, match=r"exactly 1, 0 or \? per window.*row 5"):
        main(["estimate", str(out)])
    rows[3]["wrong"] = "2"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match=r"exactly 1, 0 or \?"):
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


# ----------------------------------------------------------------------------- compare, audited on real GeoTIFFs 2026-09-22
def _cmp(tmp_path, *args):
    out = tmp_path / "c"
    assert main(["compare", *map(str, args), "--out", str(out)]) == 0
    return json.load(open(out / "comparison.json"))


def test_compare_no_data_pixels_do_not_vote():
    """A stripe of no-data in B used to argmax to class 0 and make 64 windows 'differ'; A and B are otherwise equal."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    p = rng.random((3, 64, 64)).astype(np.float32); p /= p.sum(0)
    q = p.copy(); q[:, :, 30:33] = -9999
    _write(tmp / "a.tif", p, nodata=-9999); _write(tmp / "b.tif", q, nodata=-9999)
    s = _cmp(tmp, tmp / "a.tif", tmp / "b.tif")
    assert s["n_disagree"] == 0


def test_compare_label_free_numbers_do_not_change_when_labels_are_added(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, (64, 64)); b = a.copy(); b[:, :20] = 1 - b[:, :20]
    lab = rng.integers(0, 2, (64, 64)); lab[:, 40:] = -1                       # labels over part of the map only
    for n, x in (("a", a), ("b", b), ("l", lab)):
        np.save(tmp_path / f"{n}.npy", x)
    free = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy")
    graded = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy", "--labels", tmp_path / "l.npy")
    assert (graded["n_windows"], graded["n_disagree"]) == (free["n_windows"], free["n_disagree"])
    assert graded["graded"]["crosstab"]["n"] < graded["n_windows"]           # the grading covers the labelled windows
    assert any("graded block covers" in n for n in graded["notes"])


def test_compare_threshold_applies_to_integer_valued_continuous_maps(tmp_path):
    """Percent cover stored as int16 used to ignore --threshold and compare 101 classes: 503 'differ' against 74."""
    rng = np.random.default_rng(2)
    pa = rng.random((64, 64)); pb = np.clip(pa + rng.normal(0, 0.05, pa.shape), 0, 1)
    np.save(tmp_path / "a.npy", np.rint(pa * 100).astype(np.int16)); np.save(tmp_path / "b.npy", np.rint(pb * 100).astype(np.int16))
    s = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy", "--threshold", "50")
    A, B = np.rint(pa * 100), np.rint(pb * 100)                                  # the command's own tie rule: confidence
    brute = (_pooled_argmax((A > 50).astype(int), 2, 4, weights=np.abs(A - 50))
             != _pooled_argmax((B > 50).astype(int), 2, 4, weights=np.abs(B - 50)))
    assert s["n_disagree"] == int(brute.sum()) and s["n_disagree"] < 0.3 * s["n_windows"]


def test_compare_windows_outside_every_zone_belong_to_no_group(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, (64, 64)); b = a.copy(); b[:32, :32] = 1 - b[:32, :32]
    g = np.full((64, 64), -1); g[:32, :32] = 0; g[32:, :32] = 2                 # the right half is in no zone
    for n, x in (("a", a), ("b", b), ("g", g)):
        np.save(tmp_path / f"{n}.npy", x)
    s = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy", "--groups", tmp_path / "g.npy")
    # a hard class map carries no confidence, so an 8-8 window has no way to break its tie; since 2026-09-23 such a
    # window is left out on both sides (it used to go to class 0 in a and in its flipped copy, and "agree"), so every
    # compared window of the flipped quadrant differs
    assert set(s["per_group"]) == {"0", "2"} and s["per_group"]["0"]["n"] < 64 and s["per_group"]["0"]["rate"] == 1.0
    assert any("in no group" in n for n in s["notes"]) and any("split evenly" in n for n in s["notes"])


def test_compare_boundary_cue_ignores_no_data_holes(tmp_path):
    """Grid-aligned no-data holes used to manufacture a boundary around every hole."""
    a = np.zeros((64, 64), np.float32); a[:, 32:] = 1                          # one real boundary, down the middle
    a2 = a.copy(); a2[8:16, 8:16] = np.nan; a2[40:48, 8:16] = np.nan
    np.save(tmp_path / "a.npy", a2); np.save(tmp_path / "b.npy", a2)
    s = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy")
    assert s["where"]["boundary_a"]["n_with_cue"] == 32                          # the two columns either side of x = 32 only


def test_compare_refuses_maps_on_different_grids_of_the_same_size(tmp_path):
    from rasterio.transform import from_origin
    p, _ = _scene()
    _write(tmp_path / "a.tif", p)
    with rasterio.open(tmp_path / "b.tif", "w", driver="GTiff", height=p.shape[0], width=p.shape[1], count=1, dtype=p.dtype,
                       crs="EPSG:32633", transform=from_origin(512800.0, 5000000.0, 10.0, 10.0)) as dst:
        dst.write(p[None])
    with pytest.raises(SystemExit, match="not on the first map's grid"):
        main(["compare", str(tmp_path / "a.tif"), str(tmp_path / "b.tif"), "--out", str(tmp_path / "x")])


def test_compare_negative_label_codes_are_unlabelled_without_a_nodata_tag(tmp_path):
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, (32, 32)); b = rng.integers(0, 2, (32, 32))
    lab = rng.integers(0, 2, (32, 32)); lab[:4, :4] = -1; lab[0, 0] = 1       # window (0,0): 1 labelled pixel of 16
    for n, x in (("a", a), ("b", b), ("l", lab)):
        np.save(tmp_path / f"{n}.npy", x)
    s = _cmp(tmp_path, tmp_path / "a.npy", tmp_path / "b.npy", "--labels", tmp_path / "l.npy")
    assert s["graded"]["crosstab"]["n"] <= 63                                  # window (0,0) is not graded


def test_compare_degenerate_inputs_are_named_refusals(tmp_path):
    rng = np.random.default_rng(5)
    np.save(tmp_path / "a.npy", rng.integers(0, 2, (32, 32))); np.save(tmp_path / "b.npy", rng.integers(0, 2, (32, 32)))
    np.save(tmp_path / "small.npy", rng.integers(0, 2, (16, 16))); np.save(tmp_path / "neg.npy", -np.ones((32, 32), int))
    for extra, msg in ((["--labels", tmp_path / "small.npy"], "has shape"), (["--groups", tmp_path / "small.npy"], "has shape"),
                       (["--groups", tmp_path / "neg.npy"], "no valid group"), (["--patch", "0"], "--patch"), (["--patch", "64"], "--patch")):
        with pytest.raises(SystemExit, match=msg):
            main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--out", str(tmp_path / "x"), *map(str, extra)])


# ----------------------------------------------------------------------------- certify and --per-class (exp80, exp81)
def _fill_with_classes(csv_path, probs, expert, patch=4):
    """As _fill, and the reviewer's class per window in a `reference_class` column."""
    from oe_inferencex.assess import _pooled_argmax
    hard = _pooled_argmax(probs.argmax(0), probs.shape[0], patch)
    ref = _pooled_argmax(np.where(expert >= 0, expert, -1), probs.shape[0], patch, empty=-1)
    rows = list(csv.DictReader(open(csv_path)))
    keep = []
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        r["wrong"] = str(int(hard[i, j] != ref[i, j]))
        r["reference_class"] = str(int(ref[i, j])) if ref[i, j] >= 0 else str(int(hard[i, j]))
        keep.append(r)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(keep)
    return hard, ref


def test_certify_needs_a_random_sample_and_then_names_a_zone_or_says_why_not(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    conf = tmp_path / "c.csv"
    main(["sample", path, "--budget", "100", "--out", str(conf)])
    _fill(conf, probs, expert)
    with pytest.raises(SystemExit, match="needs a random sample"):
        main(["certify", str(conf), "--alpha", "0.2"])
    rnd = tmp_path / "r.csv"
    assert main(["sample", path, "--budget", "300", "--design", "random", "--out", str(rnd)]) == 0
    hard, ref = _fill(rnd, probs, expert)
    truth = float((hard != ref).mean())
    assert main(["certify", str(rnd), "--alpha", str(round(truth, 3))]) == 0      # alpha = the map's own rate
    z = json.load(open(tmp_path / "r_zone.json"))
    assert z["rule"] == "prefix" and z["delta"] == 0.1 and z["n_population"] == 32 * 32 and len(z["levels"]) >= 1
    if z["coverage"] is not None:
        assert z["n_zone"] == int(round(z["coverage"] * z["n_population"]))
        mask = np.load(tmp_path / "r_zone.npy")
        assert mask.shape == (32, 32) and int(mask.sum()) == z["n_zone"]
        assert z["upper_bound"] <= round(truth, 3) + 1e-12
    small = tmp_path / "s.csv"
    main(["sample", path, "--budget", "20", "--design", "random", "--out", str(small)])
    _fill(small, probs, expert)
    assert main(["certify", str(small), "--alpha", "0.02"]) == 0                # 20 labels cannot certify 2%
    z = json.load(open(tmp_path / "s_zone.json"))
    assert z["coverage"] is None and "needs 114" in z["note"]


def test_estimate_per_class_reports_every_class_and_needs_the_reference_column(tmp_path):
    path, probs, expert = _sample_map(tmp_path)
    out = tmp_path / "p.csv"
    assert main(["sample", path, "--budget", "300", "--design", "random", "--out", str(out)]) == 0
    _fill(out, probs, expert)
    with pytest.raises(SystemExit, match="reference_class"):
        main(["estimate", str(out), "--per-class"])
    hard, ref = _fill_with_classes(out, probs, expert)
    assert main(["estimate", str(out), "--per-class"]) == 0
    r = json.load(open(tmp_path / "p_estimate.json"))
    assert "per_class" in r and len(r["per_class"]) == probs.shape[0] and "overall_accuracy" in r
    shares = sum(v["map_share"] for v in r["per_class"].values())
    assert abs(shares - 1) < 1e-9
    assert abs(r["overall_accuracy"]["estimate"] - (1 - r["estimate"])) < 0.05     # both from the same 300 labels
    conf = tmp_path / "q.csv"
    main(["sample", path, "--budget", "300", "--out", str(conf)])                 # the confidence design works too
    _fill_with_classes(conf, probs, expert)
    assert main(["estimate", str(conf), "--per-class"]) == 0
    r = json.load(open(tmp_path / "q_estimate.json"))
    assert r["per_class_method"].startswith("stratified")


def test_certify_accepts_a_spreadsheet_round_trip_of_the_confidence_column_and_refuses_another_map(tmp_path):
    """Verification of the second review: at a fixed 1e-4 a confidence column rounded to three digits by a
    spreadsheet was refused as another map; the tolerance now follows the digits written. Another map is still
    refused."""
    from oe_inferencex.cli import _rounding_tolerance
    assert _rounding_tolerance("0.412") == 5e-4 and _rounding_tolerance("4.12e-01") == 5e-4
    assert _rounding_tolerance("0.4123456789") == 1e-6
    path, probs, expert = _sample_map(tmp_path)
    csv_path = str(tmp_path / "r.csv")
    assert main(["sample", path, "--budget", "200", "--design", "random", "--out", csv_path]) == 0
    _fill(csv_path, probs, expert)
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["confidence"] = f"{float(r['confidence']):.3g}"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert main(["certify", csv_path, "--alpha", "0.3", "--out", str(tmp_path / "z.json")]) == 0
    other = tmp_path / "other.npy"
    np.save(other, np.roll(probs, 7, axis=2))
    with pytest.raises(SystemExit, match="not the one the CSV records|valid windows"):
        main(["certify", csv_path, "--alpha", "0.3", "--scores", str(other), "--out", str(tmp_path / "z2.json")])


# ----------------------------------------------------------------------------- the input-condition layer (1.4.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden", "condition_1_3_1")


def _dw_condition(tmp_path, layer=None):
    """The shipped Dynamic World map and a condition layer on its 128 x 128 px grid: clear on the left half, cloudy
    on the right, nothing recorded in the bottom row of windows."""
    path, probs, expert = _sample_map(tmp_path)
    if layer is None:
        H, W = probs.shape[1:]
        layer = np.where(np.arange(W)[None, :] < W // 2, 0, 1) * np.ones((H, 1), int)
        layer[-4:] = -1
    cpath = tmp_path / "cond.npy"
    np.save(cpath, layer)
    return path, probs, expert, str(cpath)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def test_assess_condition_writes_the_grid_the_per_condition_csvs_and_one_line(tmp_path, capsys):
    """With --condition: condition.tif on the window grid (int32, no-data -1), a `condition` column at the end of the
    whole-map review sets, one CSV per budget ranking each condition on its own, the inputs recorded, and one
    printed line with the scope note under it. The raster's own no-data records no condition."""
    from oe_inferencex.assess import SCOPE_ASSESS_K
    p, water = _scene()
    _write(tmp_path / "p.tif", p)
    lay = np.where(np.arange(128)[None, :] < 64, 0, 1) * np.ones((128, 1), "int16")
    lay[120:, :] = -9999                                                      # the raster's no-data: no condition
    _write(tmp_path / "c.tif", lay.astype("int16"), nodata=-9999, dtype="int16")
    out = tmp_path / "out"
    capsys.readouterr()
    assert main(["assess", str(tmp_path / "p.tif"), "--out", str(out), "--budgets", "0.05", "0.10",
                 "--condition", str(tmp_path / "c.tif"), "--condition-names", "0=clear", "1=cloudy"]) == 0
    printed = capsys.readouterr().out.splitlines()
    api = assess_prediction(p, is_logit=False, nodata_mask=np.isnan(p), budgets=(0.05, 0.10),
                            condition=np.where(lay == -9999, -1, lay), condition_names={0: "clear", 1: "cloudy"})
    per = api["conditions"]["per_condition"]
    assert list(per) == ["clear", "cloudy", "unrecorded"] and per["unrecorded"]["n_windows"] == 2 * 32
    line = "3 input conditions: " + "; ".join(
        f"{n} {100 * e['share_of_map']:.1f}% of windows, {100 * e['share_of_review_set'][0.05]:.0f}% of the 5% review set"
        for n, e in per.items())
    assert printed[1] == line and printed[2] == "note: " + SCOPE_ASSESS_K.format(K=3)
    assert printed[-1].startswith("wrote ") and len(printed) == 4

    s = json.load(open(out / "assessment.json"))
    assert s["inputs"]["condition"] == str(tmp_path / "c.tif") and s["inputs"]["condition_names"] == {"0": "clear", "1": "cloudy"}
    assert s["conditions"]["source"] == str(tmp_path / "c.tif") and s["scope"] == SCOPE_ASSESS_K.format(K=3)
    assert s["conditions"]["per_condition"] == summary(api)["conditions"]["per_condition"]
    with rasterio.open(out / "condition.tif") as src:
        assert src.dtypes[0] == "int32" and src.nodata == -1 and src.transform.a == 40.0
        np.testing.assert_array_equal(src.read(1), api["arrays"]["condition"])
    assert s["files"]["condition"].endswith("condition.tif")

    grid = api["arrays"]["condition"]
    name_of = {0: "clear", 1: "cloudy", -1: "unrecorded"}
    for b, tag in ((0.05, "05"), (0.10, "10")):
        rows = _rows(out / f"review_set_{tag}pct.csv")
        assert list(rows[0])[-1] == "condition" and len(rows) == api["review_sets"][b]["n_windows"]
        assert all(r["condition"] == name_of[int(grid[int(r["window_row"]), int(r["window_col"])])] for r in rows)
        by = _rows(out / f"review_set_{tag}pct_by_condition.csv")
        assert list(by[0]) == ["condition", "rank_in_condition", "window_row", "window_col", "pixel_row", "pixel_col",
                               "x", "y", "confidence", "boundary"]
        for name, e in per.items():
            mine = [r for r in by if r["condition"] == name]
            assert [int(r["rank_in_condition"]) for r in mine] == list(range(1, len(mine) + 1))
            assert [[int(r["window_row"]), int(r["window_col"])] for r in mine] == np.asarray(e["review_sets"][b]["windows_rowcol"]).tolist()
            assert all(float(r["x"]) == 500000.0 + 10.0 * (int(r["pixel_col"]) + 2) for r in mine)
        assert s["files"][f"review_set_{b}_by_condition"].endswith(f"review_set_{tag}pct_by_condition.csv")


def test_sample_condition_writes_design_and_columns(tmp_path, capsys):
    """--condition alone resolves to the condition design: labels split equally, the condition's name right after
    `stratum` (which holds the condition's index), `map_class, wrong` still last, and the sidecar keys the estimate
    and certify steps read."""
    from oe_inferencex.assess import RULE_TEXT
    from oe_inferencex import estimate as est
    path, probs, expert, cpath = _dw_condition(tmp_path)
    out = tmp_path / "s.csv"
    capsys.readouterr()
    assert main(["sample", path, "--budget", "300", "--out", str(out), "--condition", cpath,
                 "--condition-names", "0=clear", "1=cloudy"]) == 0
    printed = capsys.readouterr().out.splitlines()
    side = json.load(open(tmp_path / "s.json"))
    c = side["condition"]
    assert list(c) == ["source", "rule", "values", "names", "sizes", "n_labelled", "allocation_rule",
                       "n_windows_split", "n_windows_no_code"]
    assert c["source"] == cpath and c["rule"] == RULE_TEXT and c["allocation_rule"] == "equal"
    assert c["values"] == [0, 1, None] and c["names"] == ["clear", "cloudy", "unrecorded"]
    assert sum(c["sizes"]) == side["n_population"] == 1024
    assert c["n_labelled"] == side["allocation"] == est.equal_allocation(c["sizes"], 300).tolist()
    assert (c["n_windows_split"], c["n_windows_no_code"]) == (0, 32)
    assert side["design"] == "condition" and side["n_strata"] == 3 and side["sizes"] == c["sizes"]
    grid = np.asarray(side["condition_grid"])
    assert grid.shape == (32 * 32,) and np.bincount(grid[grid >= 0]).tolist() == c["sizes"]
    assert np.array_equal(np.asarray(side["strata"]), grid[np.asarray(side["strata_of_population"])])

    rows = _rows(out)
    cols = list(rows[0])
    assert cols.index("condition") == cols.index("stratum") + 1 and cols[-2:] == ["map_class", "wrong"]
    for r in rows:
        k = int(grid[int(r["index"])])
        assert int(r["stratum"]) == k and r["condition"] == c["names"][k]

    counts = ", ".join(f"{n} {k}" for n, k in zip(c["names"], c["n_labelled"]))
    assert printed[0].startswith(f"300 windows to label of 1024 valid (condition design: {counts}); wrote {out}")
    assert ("note: labels are split equally across the 3 input conditions so that each gets its own error rate; the "
            "whole-map rate weights each condition by its share of the map") in printed
    assert "note: 32 windows carry no condition value; they count as the condition 'unrecorded'" in printed
    b1, b2 = est.min_labels_to_certify(0.05, 0.1), est.min_labels_to_certify(0.05, 0.05)
    assert (f"note: to certify at alpha 0.05 (delta 0.1 split across the 2 conditions with at least {b1} labels), each "
            f"needs at least {b2} labels; unrecorded gets 32") in printed
    assert (b1, b2) == (45, 59)


@pytest.mark.parametrize("extra,why", [
    (["--design", "confidence", "--condition", "{c}"], "that design allocates labels from the model's confidence, which "
                                                       "can overstate the accuracy of a condition read from an input "
                                                       "combination the model was not trained on: it did on PASTIS for a "
                                                       "probe trained on radar plus optical and run on radar alone, "
                                                       "though not on CropHarvest China 6"),
    (["--design", "proportional", "--condition", "{c}"], r"Use --design condition \(the default with --condition\) or --design random"),
    (["--design", "tiles", "--condition", "{c}"], "a tile can span conditions, and the tile interval is not graded per condition"),
    (["--design", "condition"], "--design condition needs --condition"),
    (["--condition-names", "0=clear"], "none was given"),
    (["--condition", "{float}"], "not integers"),
    (["--condition", "{many}"], "at most 64"),
    (["--condition", "{bands}"], "one band"),
    (["--condition", "{small}"], r"^sample: .*small\.npy has shape \(64, 128\); the map is \(128, 128\)$"),
    (["--condition", "{none}"], "no window takes a condition"),
    (["--condition", "{c}", "--condition-names", "clear"], "value=name pairs"),
    (["--condition", "{c}", "--condition-names", "x=clear"], "value=name pairs"),
    (["--condition", "{c}", "--condition-names", "0=a", "0=b"], "names the value 0 twice"),
    (["--condition", "{c}", "--condition-names", "0=a", "1=a"], "unique"),
    (["--condition", "{c}", "--condition-names", "0=unrecorded"], "reserved"),
    (["--condition", "{c}", "--condition-names", "0="], "non-empty"),
    (["--condition", "{c}", "--budget", "5"], "needs 6"),
])
def test_sample_condition_refusals(tmp_path, extra, why):
    path, probs, expert, cpath = _dw_condition(tmp_path)
    H, W = probs.shape[1:]
    layers = {"c": cpath, "float": np.where(np.arange(W)[None, :] < 64, 0.0, 0.5) * np.ones((H, 1)),
              "many": np.arange(H * W).reshape(H, W) % 70, "bands": np.zeros((2, H, W), int),
              "small": np.zeros((H // 2, W), int), "none": np.full((H, W), -1)}
    for k, v in layers.items():
        if not isinstance(v, str):
            layers[k] = str(tmp_path / f"{k}.npy")
            np.save(layers[k], v)
    extra = [x.format(**layers) for x in extra]
    budget = [] if "--budget" in extra else ["--budget", "300"]
    with pytest.raises(SystemExit, match=why):
        main(["sample", path, *budget, "--out", str(tmp_path / "x.csv"), *extra])
    assert not os.path.exists(tmp_path / "x.csv")


def test_sample_condition_refuses_a_layer_on_another_grid(tmp_path):
    p, _ = _scene()
    _write(tmp_path / "p.tif", p)
    from rasterio.transform import from_origin
    with rasterio.open(tmp_path / "c.tif", "w", driver="GTiff", height=128, width=128, count=1, dtype="int16",
                       crs="EPSG:32633", transform=from_origin(512800.0, 5000000.0, 10.0, 10.0)) as dst:
        dst.write(np.zeros((1, 128, 128), "int16"))
    for cmd in (["sample", str(tmp_path / "p.tif"), "--budget", "50", "--out", str(tmp_path / "x.csv")],
                ["assess", str(tmp_path / "p.tif"), "--out", str(tmp_path / "a")]):
        with pytest.raises(SystemExit, match=f"^{cmd[0]}: .*c\\.tif is not on the first map's grid"):
            main([*cmd, "--condition", str(tmp_path / "c.tif")])
    np.save(tmp_path / "small.npy", np.zeros((64, 128), int))           # the shape refusal names the command too
    for cmd in (["sample", str(tmp_path / "p.tif"), "--budget", "50", "--out", str(tmp_path / "x.csv")],
                ["assess", str(tmp_path / "p.tif"), "--out", str(tmp_path / "a")]):
        with pytest.raises(SystemExit, match=f"^{cmd[0]}: .*small\\.npy has shape"):
            main([*cmd, "--condition", str(tmp_path / "small.npy")])


def test_random_with_condition_draws_the_same_windows(tmp_path, capsys):
    """The condition is only recorded under the random design: the same indices, the same CSV but for the added
    column, and a sidecar that differs only by the condition's keys. With one condition the condition design is
    that same draw."""
    path, probs, expert, cpath = _dw_condition(tmp_path)
    assert main(["sample", path, "--budget", "250", "--design", "random", "--seed", "7", "--out", str(tmp_path / "a.csv")]) == 0
    capsys.readouterr()
    assert main(["sample", path, "--budget", "250", "--design", "random", "--seed", "7", "--out", str(tmp_path / "b.csv"),
                 "--condition", cpath]) == 0
    printed = capsys.readouterr().out
    a, b = json.load(open(tmp_path / "a.json")), json.load(open(tmp_path / "b.json"))
    assert a["indices"] == b["indices"] and b["condition"]["allocation_rule"] is None and "strata" not in b
    assert set(b) - set(a) == {"condition", "condition_grid"}
    assert all(a[k] == b[k] for k in a if k not in ("csv", "how_to_label"))
    ra, rb = _rows(tmp_path / "a.csv"), _rows(tmp_path / "b.csv")
    assert [{k: v for k, v in r.items() if k != "condition"} for r in rb] == ra
    counts = ", ".join(f"{n} {k}" for n, k in zip(b["condition"]["names"], b["condition"]["n_labelled"]))
    assert f"(random design; by chance: {counts})" in printed and "split equally" not in printed

    one = tmp_path / "one.npy"
    np.save(one, np.zeros(probs.shape[1:], int))
    capsys.readouterr()
    assert main(["sample", path, "--budget", "250", "--seed", "7", "--out", str(tmp_path / "c.csv"),
                 "--condition", str(one), "--condition-names", "0=clear"]) == 0
    printed = capsys.readouterr().out
    c = json.load(open(tmp_path / "c.json"))
    assert c["design"] == "condition" and c["indices"] == a["indices"] and c["allocation"] == [250]
    assert "(condition design: clear 250)" in printed
    assert "note: one condition covers the whole population; this is a simple random sample" in printed
    assert "to certify" not in printed


def _golden_module():
    spec = importlib.util.spec_from_file_location("golden_1_3_1", os.path.join(GOLDEN, "generate.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def test_existing_outputs_are_byte_identical_to_1_3_1(tmp_path):
    """Without a layer, every output of 1.3.1 is unchanged but for the changes made on purpose: the stdout of assess,
    sample (four designs), estimate (with and without --per-class) and certify, the sample CSVs and sidecars, the
    review sets, the rasters and explanation.json byte for byte; and the JSON outputs byte for byte once `scope`, the
    one added key, is taken out, but for the last digits of a float where C libraries and numpy builds differ
    (changes.same_but_last_digits). The files were generated at 725dffa, before any edit
    (tests/golden/condition_1_3_1/generate.py).

    The changes made on purpose are listed in tests/golden/condition_1_3_1/changes.py, each with 1.3.1's whole text,
    the new text and the files that hold it, and nothing else may differ:
    - certify's note on the prefix rule, a line of the stdout and the `note` of the zone JSON, in the four prefix
      certifications (sample_random certify_prefix, certify_delta and certify_whole, sample_scene_random
      certify_prefix);
    - the warning on a multi-class logit map scored by the logit margin, a line of sample_logits3_random's stdout and
      an entry of `warnings` in its sidecar, in assess_logits3's assessment.json and in the API summary
      api_prediction_logits3_margin;
    - the warning on every probability map, which said "prefer logits" whatever the number of classes: a line of the
      stdout of the six probability samples and an entry of `warnings` in their sidecars, in the two probability
      assessments and in the two probability API summaries;
    - the warning on a tiles sample, which quoted exp78's design for the shipped one: a line of
      sample_tiles.estimate's stdout and the `warning` of its JSON;
    - the instruction sample prints for the reviewer, which now offers ? for a window that cannot be judged: a stretch
      of the first line of the seven sample outputs;
    - the zone JSONs' `level_cut` key, added with the ramp option (exp96): it is "standard", the cut every 1.3.1 output
      was made with, and is taken out with `scope`."""
    from oe_inferencex.assess import SCOPE_ASSESS
    from oe_inferencex.estimate import SCOPE_CERTIFY, SCOPE_ESTIMATE
    gen = _golden_module()
    spec = importlib.util.spec_from_file_location("golden_1_3_1_changes", os.path.join(GOLDEN, "changes.py"))
    changes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(changes)
    files, manifest = gen.produce(str(tmp_path))
    golden = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    assert sorted(files) == golden["files"]                     # no file added or lost, condition.tif included
    assert manifest["steps"] == golden["steps"] and manifest["api"] == golden["api"]
    # the list is exact: four changes of whole texts in exactly these thirty files, and the reviewer's instruction
    # in the seven sample outputs, every one of them a golden file, and each new text is the one the package keeps
    listed = sorted(n for c in changes.CHANGES[:4] for n in c["files"])
    assert len(changes.CHANGES) == 5 and len(listed) == len(set(listed)) == 30 and set(listed) <= set(golden["files"])
    fill = changes.CHANGES[4]["files"]
    assert len(fill) == len(set(fill)) == 7 and set(fill) <= set(golden["files"]) and changes.CHANGES[4]["within_line"]
    for c in changes.CHANGES:
        module, constant = c["constant"]
        assert getattr(importlib.import_module(module), constant) == c["new"], constant

    def scope_of(name):
        if name.startswith("api_") or name.endswith("__assessment.json"):
            return SCOPE_ASSESS
        if name.endswith((".estimate.json", ".perclass.json")):
            return SCOPE_ESTIMATE
        if ".certify_" in name and name.endswith(".json"):
            return SCOPE_CERTIFY
        return None

    n_scoped = 0
    for name, data in sorted(files.items()):
        with open(os.path.join(GOLDEN, name), "rb") as f:
            want = changes.expected(name, f.read())          # 1.3.1's bytes with the listed changes, and no others
        scope = scope_of(name)
        if scope is None:
            assert data == want, name
            continue
        got = json.loads(data)
        assert got.pop("scope") == scope, name
        if scope == SCOPE_CERTIFY:
            assert got.pop("level_cut") == "standard", name
        # the golden JSON reads back to itself, so equal bytes after the dump mean equal bytes but for `scope`
        assert (json.dumps(json.loads(want), indent=1) + ("\n" if name.startswith("api_") else "")).encode() == want, name
        if (json.dumps(got, indent=1) + ("\n" if name.startswith("api_") else "")).encode() != want:
            # The golden files were written on macOS. On Linux the certify JSONs' p-values differ in their last digits
            # (math.lgamma and math.exp differ between C libraries; the CI run of 2 October 2026 and the cluster), and
            # with numpy 1.26.4 on x86_64 Linux so does api_prediction_logits3_top1's 5% confidence quantile (aicr job
            # 1240874). Nothing else does: the same keys in the same order, the same decisions, every float within
            # 1e-12; and only in a JSON output, never in a stdout, CSV or array.
            assert name.endswith(".json"), name
            assert changes.same_but_last_digits(got, json.loads(want)), name
        n_scoped += 1
    # six API summaries, three assessments, seven estimates and six per-class, seven zones; the stdout of each run
    assert n_scoped == 6 + 3 + 13 + 7 and sum(n.endswith(".stdout.txt") for n in files) == 3 + 7 + 7 + 6 + 7


def test_label_refuses_an_edited_condition_column(tmp_path):
    """The condition is fixed at sampling; the sidecar is its source of truth and the raster is not read again. A
    row moved to another condition in the CSV is refused by estimate and certify, not silently re-counted."""
    path, probs, expert, cpath = _dw_condition(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", path, "--budget", "200", "--design", "random", "--out", str(out), "--condition", cpath,
                 "--condition-names", "0=clear", "1=cloudy"]) == 0
    _fill(out, probs, expert)
    assert main(["estimate", str(out)]) == 0                                 # unedited, it runs
    rows = _rows(out)
    k = next(i for i, r in enumerate(rows) if r["condition"] == "clear")
    rows[k]["condition"] = "cloudy"
    _write_rows(out, rows)
    for cmd in (["estimate", str(out)], ["certify", str(out), "--alpha", "0.3"]):
        with pytest.raises(SystemExit, match=f"disagrees with the design in its sidecar on 1 row.*row {k + 2}: 'cloudy' "
                                             "where the sample recorded 'clear'"):
            main(cmd)
    rows[k]["condition"] = " clear "                                          # a spreadsheet's padding is not an edit
    _write_rows(out, rows)
    assert main(["estimate", str(out)]) == 0


def test_estimate_prints_each_condition_after_the_whole_map(tmp_path, capsys):
    from oe_inferencex import estimate as est
    path, probs, expert, cpath = _dw_condition(tmp_path)
    out = tmp_path / "s.csv"
    assert main(["sample", path, "--budget", "300", "--out", str(out), "--condition", cpath,
                 "--condition-names", "0=clear", "1=cloudy"]) == 0
    _fill(out, probs, expert)
    capsys.readouterr()
    assert main(["estimate", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    r = json.load(open(tmp_path / "s_estimate.json"))
    assert r["by_condition"] is True and r["method"] == ("stratified by input condition; sum of each condition's exact "
                                                         "interval at 1 - 0.05/L, weighted by its share of the map (union "
                                                         "bound); covers at least 95% by construction")
    assert r["condition_note"] == est.CONDITION_NOTE + " " + est.CONDITION_WHOLE_MAP and "scope" not in r
    # unrecorded is labelled in full, so L counts clear and cloudy; the floor's keys and effective_n are gone
    assert r["conditions_in_interval"] == 2 and not {"effective_n", "interval_variance", "floored_conditions"} & set(r)
    # the whole-map interval, from the JSON's own rows: each condition's exact interval at 1 - 0.05/2, weighted by its
    # share of the map (unrecorded is labelled in full and enters at its rate)
    ends = [sum(row["share_of_map"] * est.hypergeom_interval(row["n_wrong"], row["n_labelled"], row["n_population"],
                                                             conf=0.975)[i] for row in r["per_condition"].values())
            for i in (0, 1)]
    assert abs(r["low"] - ends[0]) < 1e-15 and abs(r["high"] - ends[1]) < 1e-15
    assert printed[0] == (f"error rate {100 * r['estimate']:.1f}%, 95% interval {100 * r['low']:.1f}% to "
                          f"{100 * r['high']:.1f}% (half-width {100 * r['half_width']:.1f} points), from {r['n_labelled']} labelled "
                          f"windows of {r['n_population']}; {r['method']}")
    lines = [f"{name} {100 * row['estimate']:.1f}% ({100 * row['low']:.1f}% to {100 * row['high']:.1f}%), "
             f"{row['n_labelled']} labelled of {row['n_population']} windows ({100 * row['share_of_map']:.1f}% of the map)"
             for name, row in r["per_condition"].items()]
    first = printed.index(lines[0])
    assert printed[first:first + len(lines)] == lines and first == 1 + ("warning" in r)
    tail = printed[first + len(lines):]
    # the closing note is worded by the direction observed: on the shipped map clear is worse than the map as a whole,
    # cloudy better, and unrecorded, labelled in full, has its exact rate below it (review of 2026-09-29)
    per = r["per_condition"]
    assert [n for n in r["outside_condition_intervals"] if per[n]["low"] > r["estimate"]] == ["clear"]
    assert [n for n in r["outside_condition_intervals"] if per[n]["high"] < r["estimate"]] == ["cloudy", "unrecorded"]
    assert per["unrecorded"]["n_labelled"] == per["unrecorded"]["n_population"]
    assert tail[0] == (f"note: the whole-map rate is {100 * r['estimate']:.1f}%. It lies below the interval of clear, so "
                       "that condition is worse than the map as a whole. It lies above the intervals of cloudy and "
                       "unrecorded (labelled in full, so its interval is its exact rate), so those conditions are better "
                       "than the map as a whole. The whole-map rate weights each condition by its share of the map.")
    assert "much worse" not in "\n".join(printed)
    from oe_inferencex.cli import _outside_note
    one = {"estimate": 0.3, "outside_condition_intervals": ["clear"],
           "per_condition": {"clear": {"low": 0.1, "high": 0.2, "n_labelled": 150, "n_population": 5000}}}
    assert _outside_note(one) == ("note: the whole-map rate is 30.0%. It lies above the interval of clear, so that "
                                  "condition is better than the map as a whole. The whole-map rate weights each "
                                  "condition by its share of the map.")
    # the whole-map line says "95% interval"; under the condition design the line below says how it holds
    assert tail[-2] == f"note: {est.CONDITION_WHOLE_MAP}" and tail[-1].startswith("wrote ")
    assert "not been graded" not in "\n".join(printed)

    # under a random sample the whole-map interval is the exact one, and no such note is printed
    rnd = tmp_path / "r.csv"
    assert main(["sample", path, "--budget", "300", "--design", "random", "--out", str(rnd), "--condition", cpath]) == 0
    _fill(rnd, probs, expert)
    capsys.readouterr()
    assert main(["estimate", str(rnd)]) == 0
    printed = capsys.readouterr().out
    assert "1 - 0.05/L" not in printed and json.load(open(tmp_path / "r_estimate.json"))["by_condition"] is True


def test_certify_by_condition_writes_the_union_mask_and_removes_a_stale_one(tmp_path, capsys):
    """A condition sample is certified per condition: the mask is the union of the conditions' zones, on the same
    path and dtype as a whole-map zone, and a later run that certifies nothing removes it. The top-level zone fields
    stay null and no window index list reaches the JSON."""
    from oe_inferencex import estimate as est
    from oe_inferencex.cli import _labelled_sample
    path, probs, expert, cpath = _dw_condition(tmp_path)
    out = tmp_path / "r.csv"
    assert main(["sample", path, "--budget", "300", "--design", "random", "--out", str(out), "--condition", cpath,
                 "--condition-names", "0=clear", "1=cloudy"]) == 0
    _fill(out, probs, expert)
    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.3"]) == 0
    printed = capsys.readouterr().out.splitlines()
    z = json.load(open(tmp_path / "r_zone.json"))
    mask = np.load(tmp_path / "r_zone.npy")
    assert all(z[k] is None for k in ("coverage", "n_zone", "threshold", "upper_bound")) and z["levels"] == []
    assert z["by_condition"] is True and "zone_indices_in_order" not in json.dumps(z)
    assert z["zone_mask"] == str(tmp_path / "r_zone.npy")
    _, _, idx, sample, wrong, _ = _labelled_sample(str(out), "certify")
    arr = assess_prediction(probs, is_logit=False, patch=4, nodata_mask=~np.isfinite(probs).all(0))["arrays"]
    api = est.certify_by_condition(sample, wrong, arr["confidence"].ravel(), 0.3, valid=arr["valid"].ravel())
    union = np.zeros(32 * 32, bool)
    union[api["zone_indices_in_order"]] = True
    assert mask.dtype == bool and mask.shape == (32, 32) and np.array_equal(mask.ravel(), union)
    assert int(mask.sum()) == z["n_certified"] == sum(e["n_zone"] or 0 for e in z["per_condition"].values())
    assert z["certified_share_of_map"] == pytest.approx(mask.sum() / 1024, abs=0) and z["certified_share_of_map"] > 0
    d = z["delta_per_condition"]
    assert printed[0] == (f"certified per input condition at alpha=30%, delta=10% (each tested condition at {100 * d:.3g}%, "
                          "so the statements hold together on at least 90% of samples):")
    for name, e in z["per_condition"].items():
        line = next(ln for ln in printed if ln.startswith(f"  {name}: "))
        if e["coverage"] is not None:
            assert line == (f"  {name}: the {100 * e['coverage']:.0f}% most confident windows of this condition "
                            f"({e['n_zone']} of {e['n_population']}, margin >= {e['threshold']:.4f}) are wrong at most "
                            "30% of the time")
    assert printed[-2] == (f"together the certified windows are {100 * z['certified_share_of_map']:.1f}% of the map "
                           f"(mask {tmp_path / 'r_zone.npy'}); outside them nothing is certified")

    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.01"]) == 0             # certifies nothing: the stale mask goes
    printed = capsys.readouterr().out
    z = json.load(open(tmp_path / "r_zone.json"))
    assert z["certified_share_of_map"] is None and z["n_certified"] == 0 and not os.path.exists(tmp_path / "r_zone.npy")
    assert "zone_mask" not in z and "no zone is certified in any condition, so nothing is certified" in printed
    assert "not tested; " in printed and "needs at least" in printed


def test_certify_prints_a_condition_whose_labels_look_chosen_as_not_tested(tmp_path, capsys):
    """One condition's labels moved to its least confident windows: certify reports that condition not tested, with
    the reason, and certifies the other one at the delta the counts fixed, instead of refusing the whole sample."""
    path, probs, expert, cpath = _dw_condition(tmp_path)
    out = tmp_path / "r.csv"
    assert main(["sample", path, "--budget", "200", "--design", "random", "--out", str(out), "--condition", cpath,
                 "--condition-names", "0=clear", "1=cloudy"]) == 0
    _fill(out, probs, expert)
    side_path = tmp_path / "r.json"
    side = json.load(open(side_path))
    grid, idx = np.asarray(side["condition_grid"]), np.asarray(side["indices"])
    arrays = assess_prediction(probs, is_logit=False, patch=4, nodata_mask=~np.isfinite(probs).all(0))["arrays"]
    conf, klass = arrays["confidence"], arrays["pooled_argmax"]
    cloudy = np.flatnonzero(grid == 1)
    n1 = int((grid[idx] == 1).sum())
    chosen = cloudy[np.lexsort((cloudy, conf.ravel()[cloudy]))[:n1]]        # cloudy's least confident windows
    new = np.r_[idx[grid[idx] != 1], chosen]
    side["indices"] = new.tolist()
    json.dump(side, open(side_path, "w"))
    rows = _rows(out)
    by = {int(r["index"]): r for r in rows}
    fresh = []
    for i in new:
        r = dict(by.get(int(i), rows[0]))
        r.update({"index": str(int(i)), "window_row": str(int(i) // 32), "window_col": str(int(i) % 32),
                  "condition": side["condition"]["names"][int(grid[i])], "confidence": repr(float(conf.ravel()[i])),
                  "map_class": str(int(klass.ravel()[i])),        # certify checks the CSV's class against the map's
                  "wrong": r["wrong"] if int(i) in by else "0"})
        fresh.append(r)
    _write_rows(out, fresh)
    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.3"]) == 0
    printed = capsys.readouterr().out.splitlines()
    z = json.load(open(tmp_path / "r_zone.json"))
    e = z["per_condition"]["cloudy"]
    L = sum(p["n_labelled"] >= 7 for p in z["per_condition"].values())      # 7 labels certify at alpha 0.3
    assert L == 3 and z["delta_per_condition"] == pytest.approx(0.1 / 3, abs=1e-15)   # the split counts cloudy
    assert not e["tested"] and z["n_conditions_tested"] == 2
    assert z["per_condition"]["clear"]["tested"] and z["per_condition"]["unrecorded"]["tested"]
    assert f"  cloudy: not tested; {e['reason']}" in printed and "look chosen for low confidence" in e["reason"]
    assert printed[0].startswith("certified per input condition at alpha=30%, delta=10% (each tested condition at 3.33%")


def test_cmd_assess_namespace_without_condition_still_runs(tmp_path):
    """demo.py and tests build the argument Namespace by hand, without the new attributes; the commands read them
    with getattr and behave as 1.3.1 did."""
    import argparse
    from oe_inferencex import cli
    path, probs, expert = _sample_map(tmp_path)
    assert cli.cmd_assess(argparse.Namespace(scores=path, out=str(tmp_path / "a"), logits=False, patch=4, nodata=None,
                                             reference=None, budgets=[0.05], order="confidence")) == 0
    s = json.load(open(tmp_path / "a" / "assessment.json"))
    assert set(s["inputs"]) == {"scores", "logits", "reference"} and "conditions" not in s
    assert not os.path.exists(tmp_path / "a" / "condition.npy")
    assert list(_rows(tmp_path / "a" / "review_set_05pct.csv")[0])[-1] == "boundary"
    ns = argparse.Namespace(scores=path, budget=50, out=str(tmp_path / "s.csv"), tile=16, per_tile=16, logits=False,
                            patch=4, nodata=None, seed=0)                                # no design: the confidence one
    assert cli.cmd_sample(ns) == 0
    side = json.load(open(tmp_path / "s.json"))
    assert side["design"] == "confidence" and "condition" not in side and "condition" not in _rows(tmp_path / "s.csv")[0]

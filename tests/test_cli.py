"""The command line on synthetic rasters and arrays: files written, JSON consistent with the API, coordinates right."""
import csv
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

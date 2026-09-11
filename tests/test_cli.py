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

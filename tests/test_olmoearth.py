"""The OlmoEarth reader on synthetic rslearn datasets: the layout olmoearth_run writes (rslearn @ 5780896a), read into
the rasters `assess` reads, refused where the output holds no probabilities, and the input-condition layer on the
scores' grid."""
import csv
import hashlib
import json
import os
import shlex

import numpy as np
import pytest

from oe_inferencex import olmoearth as oe
from oe_inferencex.cli import main

rasterio = pytest.importorskip("rasterio", reason="reading an rslearn dataset needs rasterio")
from rasterio.transform import Affine  # noqa: E402

RES = 10.0
A = (50000, -500000, 50016, -499984)          # pixel bounds: x 500000 to 500160 m, y 5000000 down to 4999840 m
B = (50016, -500000, 50032, -499984)          # east of A
C = (50000, -499984, 50016, -499968)          # south of A


def _rslearn_dirname(bands):
    """rslearn/utils/raster_format.py:32-47, copied: the band-set directory."""
    if any("_" in b for b in bands):
        return hashlib.sha256(json.dumps(bands).encode()).hexdigest()
    d = "_".join(bands)
    return hashlib.sha256(d.encode()).hexdigest() if len(d) > 64 else d


def _window(ds, wid, bounds, crs="EPSG:32610", res=RES):
    """windows/<group>/<name>/metadata.json as FileWindowStorage writes it (storage/file.py:171-187)."""
    path = os.path.join(ds, "windows", *wid.split("/"))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump({"projection": {"crs": crs, "x_resolution": res, "y_resolution": -res}, "bounds": list(bounds),
                   "time_range": None, "options": {}}, f)
    return path


def _geotiff(dirpath, arr, bounds, crs="EPSG:32610", res=RES, nodata=None):
    """geotiff.tif as GeotiffRasterFormat writes it: the transform from the projection and the pixel bounds
    (raster_format.py:579-587)."""
    os.makedirs(dirpath, exist_ok=True)
    arr = np.asarray(arr)
    bands = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(os.path.join(dirpath, "geotiff.tif"), "w", driver="GTiff", height=bands.shape[1],
                       width=bands.shape[2], count=bands.shape[0], dtype=bands.dtype, crs=crs,
                       transform=Affine(res, 0, bounds[0] * res, 0, -res, bounds[1] * -res), nodata=nodata) as dst:
        dst.write(bands)


def _layer(wpath, layer_dir, bands, arr, bounds, completed=True, **kw):
    ldir = os.path.join(wpath, "layers", layer_dir)
    _geotiff(os.path.join(ldir, _rslearn_dirname(list(bands))), arr, bounds, **kw)
    if completed:
        open(os.path.join(ldir, "completed"), "w").close()
    return ldir


def _softmax(rng, n, h=16, w=16):
    z = rng.normal(0, 2, (n, h, w))
    p = np.exp(z - z.max(0))
    return (p / p.sum(0)).astype(np.float32)


PROB_BANDS = ["prob_0", "prob_1", "prob_2"]   # names with "_": rslearn hashes the directory


def _two_windows_and_more(ds, rng):
    """Three windows with probabilities in one CRS (A with a 2 x 2 block no crop covered), one whose output is not
    marked completed and one without the output layer."""
    pa, pb, pc = _softmax(rng, 3), _softmax(rng, 3), _softmax(rng, 3)
    pa[:, :2, :2] = 0
    for wid, b, p in (("g1/a", A, pa), ("g1/b", B, pb), ("g2/c", C, pc)):
        _layer(_window(ds, wid, b), "output", PROB_BANDS, p, b)
    _layer(_window(ds, "g1/d", (50032, -500000, 50048, -499984)), "output", PROB_BANDS, _softmax(rng, 3),
           (50032, -500000, 50048, -499984), completed=False)
    _window(ds, "g1/e", (50048, -500000, 50064, -499984))
    return pa, pb, pc


def test_band_set_directory_follows_rslearn():
    assert oe.bandset_dirname(["B02", "B03", "B04"]) == "B02_B03_B04"
    assert oe.bandset_dirname(["prob_0"]) == hashlib.sha256(b'["prob_0"]').hexdigest()
    long = [f"B{i:02d}x" for i in range(14)]                    # 69 characters joined
    assert oe.bandset_dirname(long) == hashlib.sha256("_".join(long).encode()).hexdigest()


def test_windows_of_one_crs_are_pasted_onto_one_grid(tmp_path):
    """Each window lands where its own GeoTIFF puts it; an all-zero vector and the space between windows are NaN;
    the windows not read are listed with the reason; the hashed band-set directory is found."""
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    pa, pb, pc = _two_windows_and_more(ds, np.random.default_rng(0))
    assert os.path.isdir(os.path.join(ds, "windows", "g1", "a", "layers", "output",
                                      hashlib.sha256(json.dumps(PROB_BANDS).encode()).hexdigest()))
    s = oe.read_output(ds, out)
    assert s["windows_read"] == 3 and s["bands"] == 3 and s["kind"] == "softmax over C bands"
    assert {k["window"]: k["reason"] for k in s["windows_skipped"]} == {
        "g1/d": "with layers/output/ not marked completed: inference did not finish there",
        "g1/e": "with no layers/output/"}
    assert s["uncovered_pixels"] == 4 and len(s["grids"]) == 1
    g = s["grids"][0]
    assert g["label"] == "32610" and g["shape"] == [32, 32] and g["covered_pixels"] == 3 * 256 - 4
    assert g["windows"] == ["g1/a", "g1/b", "g2/c"]
    with rasterio.open(g["scores"]) as src:
        assert src.count == 3 and src.dtypes[0] == "float32" and np.isnan(src.nodata)
        assert src.crs.to_epsg() == 32610 and tuple(src.transform)[:6] == (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)
        sc = src.read()
    want_a = pa.copy()
    want_a[:, :2, :2] = np.nan
    np.testing.assert_array_equal(sc[:, :16, :16], want_a)
    np.testing.assert_array_equal(sc[:, :16, 16:], pb)
    np.testing.assert_array_equal(sc[:, 16:, :16], pc)
    assert np.isnan(sc[:, 16:, 16:]).all()
    assert json.load(open(os.path.join(out, oe.SCORES_JSON)))["grids"][0]["scores"] == g["scores"]
    # one group only
    s1 = oe.read_output(ds, str(tmp_path / "out1"), group="g2")
    assert s1["windows_read"] == 1 and s1["grids"][0]["shape"] == [16, 16]


def test_a_second_crs_gets_its_own_grid(tmp_path):
    ds = str(tmp_path / "ds")
    rng = np.random.default_rng(1)
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1"], _softmax(rng, 2), A)
    far = (30000, -400000, 30008, -399992)
    _layer(_window(ds, "g/z", far, crs="EPSG:32611"), "output", ["p0", "p1"], _softmax(rng, 2, 8, 8), far,
           crs="EPSG:32611")
    s = oe.read_output(ds, str(tmp_path / "out"))
    assert [g["label"] for g in s["grids"]] == ["32610", "32611"]
    assert s["grids"][1]["shape"] == [8, 8] and s["grids"][1]["transform"][2] == 300000.0
    assert any("2 CRSs" in n for n in s["notes"])
    with rasterio.open(s["grids"][1]["scores"]) as src:
        assert src.crs.to_epsg() == 32611


@pytest.mark.parametrize("dtype", ["float32", "uint8"])
def test_an_argmax_layer_is_refused_with_the_fix(tmp_path, dtype):
    """The default SegmentationTask output: one band of class ids (float32 for AWF, uint8 for mangrove)."""
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    classes = np.random.default_rng(2).integers(0, 9, (16, 16)).astype(dtype)
    _layer(_window(ds, "g/a", A), "output", ["output"], classes, A)
    with pytest.raises(ValueError) as exc:
        oe.read_output(ds, out)
    msg = str(exc.value)
    for part in ("one band of class ids", "output_probs: true", "prob_scales unset", "float32",
                 "named without underscores", "RUN_INFERENCE"):
        assert part in msg, part
    with pytest.raises(SystemExit) as exc:
        main(["from-olmoearth", ds, "--out", out])
    assert str(exc.value).startswith("from-olmoearth: window g/a: layers/output is one band of class ids")
    assert not os.path.exists(os.path.join(out, "scores_32610.tif"))


def test_one_probability_band_is_a_two_class_map(tmp_path):
    """output_probs with output_class_idx writes one band, the probability of one class; a 0 there is read as a
    probability, not as the fill, so the most confident pixels stay in the map."""
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    p = _softmax(np.random.default_rng(13), 2)[1]
    p[5, 5] = 0.0
    _layer(_window(ds, "g/a", A), "output", ["p1"], p, A)
    s = oe.read_output(ds, out)
    assert s["kind"] == "one band: the probability of one class" and s["grids"][0]["covered_pixels"] == 256
    assert any("that class against the rest" in n for n in s["notes"])
    with rasterio.open(s["grids"][0]["scores"]) as src:
        np.testing.assert_array_equal(src.read(1), p)
    argv = shlex.split(s["grids"][0]["assess"])
    assert main(argv[1:]) == 0


def test_scaled_probabilities_are_refused(tmp_path):
    """prob_scales multiplies the softmax before it is written (segmentation.py:211-218)."""
    ds = str(tmp_path / "ds")
    p = _softmax(np.random.default_rng(3), 3) * np.array([1.0, 2.0, 0.5], np.float32)[:, None, None]
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1", "p2"], p, A)
    with pytest.raises(ValueError, match="prob_scales multiplies the probabilities"):
        oe.read_output(ds, str(tmp_path / "out"))


def test_an_integer_probability_layer_is_refused(tmp_path):
    """A uint8 writer layer truncates the softmax to 0 and 1."""
    ds = str(tmp_path / "ds")
    p = _softmax(np.random.default_rng(4), 3)
    onehot = (p == p.max(0)).astype("uint8")
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1", "p2"], onehot, A)
    with pytest.raises(ValueError, match="an integer layer truncates probabilities"):
        oe.read_output(ds, str(tmp_path / "out"))
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1", "p2"], onehot.astype("float32"), A)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        oe.read_output(ds, str(tmp_path / "out"))


def test_overlapping_windows_must_agree(tmp_path):
    ds = str(tmp_path / "ds")
    rng = np.random.default_rng(5)
    over = (50008, -500000, 50024, -499984)                     # shares 8 columns with A
    pa = _softmax(rng, 2)
    po = _softmax(rng, 2)
    po[:, :, :8] = pa[:, :, 8:]
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1"], pa, A)
    _layer(_window(ds, "g/o", over), "output", ["p0", "p1"], po, over)
    s = oe.read_output(ds, str(tmp_path / "out"))
    assert s["grids"][0]["shape"] == [16, 24] and s["grids"][0]["covered_pixels"] == 16 * 24
    po[[0, 1], 3, 2] = po[[1, 0], 3, 2]                         # one overlapping pixel's classes swapped
    _layer(_window(ds, "g/o", over), "output", ["p0", "p1"], po, over)
    with pytest.raises(ValueError, match="windows g/a and g/o both predict 128 pixels and differ on 1"):
        oe.read_output(ds, str(tmp_path / "out2"))


def test_two_resolutions_in_one_crs_are_refused(tmp_path):
    ds = str(tmp_path / "ds")
    rng = np.random.default_rng(6)
    _layer(_window(ds, "g/a", A), "output", ["p0", "p1"], _softmax(rng, 2), A)
    coarse = (25008, -250000, 25016, -249992)
    _layer(_window(ds, "g/c", coarse, res=20.0), "output", ["p0", "p1"], _softmax(rng, 2, 8, 8), coarse, res=20.0)
    with pytest.raises(ValueError, match="more than one resolution"):
        oe.read_output(ds, str(tmp_path / "out"))


def _classification(ds, rng, n=6, prop="probs"):
    """Per-window ClassificationTask output: layers/output/data.geojson, one Point feature at the crop's corner in
    pixel coordinates, the class and, under prob_property, the probabilities (classification.py:186-235)."""
    probs = []
    for i in range(n):
        b = (50000 + 16 * i, -500000, 50016 + 16 * i, -499984)
        w = _window(ds, f"pred/w{i}", b)
        p = rng.dirichlet(np.ones(3))
        probs.append(p)
        ldir = os.path.join(w, "layers", "output")
        os.makedirs(ldir, exist_ok=True)
        props = {"new_label": ["human", "natural", "unknown"][int(np.argmax(p))]}
        if prop:
            props[prop] = p.tolist()
        with open(os.path.join(ldir, "data.geojson"), "w") as f:
            json.dump({"type": "FeatureCollection",
                       "properties": {"crs": "EPSG:32610", "x_resolution": RES, "y_resolution": -RES},
                       "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [b[0], b[1]]},
                                     "properties": props}]}, f)
        open(os.path.join(ldir, "completed"), "w").close()
    return np.array(probs)


def test_per_window_probabilities_feed_assess_patch_1(tmp_path, capsys):
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    probs = _classification(ds, np.random.default_rng(7))
    capsys.readouterr()
    assert main(["from-olmoearth", ds, "--out", out]) == 0
    printed = capsys.readouterr().out.splitlines()
    s = json.load(open(os.path.join(out, oe.SCORES_JSON)))
    arr = np.load(s["scores"])
    assert arr.shape == (3, 1, 6) and s["features"] == 6
    np.testing.assert_allclose(arr[:, 0, :], probs.T.astype(np.float32))
    index = list(csv.DictReader(open(s["index"])))
    assert [r["window"] for r in index] == [f"pred/w{i}" for i in range(6)]
    assert float(index[2]["x"]) == (50032 + 50048) / 2 * RES and index[2]["crs"] == "EPSG:32610"
    nxt = [line for line in printed if line.startswith("next: ")]
    assert len(nxt) == 1
    argv = shlex.split(nxt[0][len("next: "):])
    assert argv[:2] == ["oe-inferencex", "assess"] and argv[-2:] == ["--patch", "1"]
    assert main(argv[1:]) == 0
    rs = list(csv.DictReader(open(os.path.join(out, "assess", "review_set_10pct.csv"))))
    least = int(np.argmin(probs.max(1)))                        # the least confident window comes first
    assert rs[0]["window_col"] == index[least]["window_col"] == str(least)
    with pytest.raises(SystemExit, match="per-window classification"):
        main(["from-olmoearth", ds, "--out", out, "--conditions"])


def test_read_window_probs_returns_the_array_and_names(tmp_path):
    ds = str(tmp_path / "ds")
    probs = _classification(ds, np.random.default_rng(15), n=3)
    r = oe.read_window_probs(ds, str(tmp_path / "out"))
    assert r["probs"].shape == (3, 1, 3) and r["names"] == ["pred/w0", "pred/w1", "pred/w2"]
    np.testing.assert_allclose(r["probs"][:, 0, :], probs.T.astype(np.float32))
    assert r["windows_read"] == 3 and not r["windows_skipped"]


def test_per_window_output_without_probabilities_names_prob_property(tmp_path):
    ds = str(tmp_path / "ds")
    _classification(ds, np.random.default_rng(8), n=2, prop=None)
    with pytest.raises(ValueError, match='set prob_property: "probs"'):
        oe.read_window_probs(ds, str(tmp_path / "out"))


# ----------------------------------------------------------------------------- the input-condition layer
CONFIG = {"layers": {
    "label": {"type": "raster", "band_sets": [{"bands": ["category"], "dtype": "int32"}]},
    "output": {"type": "raster", "band_sets": [{"bands": ["p0", "p1", "p2"], "dtype": "float32"}]},
    "sentinel2": {"type": "raster", "band_sets": [{"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"},
                                                  {"bands": ["SCL"], "dtype": "uint8"}],
                  "data_source": {"class_path": "rslearn.data_sources.planetary_computer.Sentinel2"}},
    "sentinel1": {"type": "raster", "band_sets": [{"bands": ["vv", "vh"], "dtype": "float32"}],
                  "data_source": {"class_path": "rslearn.data_sources.planetary_computer.Sentinel1"}}}}


def _s2_group(wpath, idx, bounds, gap=None, cloud=None):
    """Item group idx of sentinel2: four reflectance bands and SCL, 0 in every band where no scene covered the
    pixel (gap), SCL 8 where cloudy, else 4 (vegetation)."""
    refl = np.full((4, 16, 16), 1200, np.uint16)
    scl = np.full((1, 16, 16), 4, np.uint8)
    if cloud is not None:
        scl[0][cloud] = 8
    if gap is not None:
        refl[:, gap] = 0
        scl[0][gap] = 0
    d = "sentinel2" if idx == 0 else f"sentinel2.{idx}"
    _layer(wpath, d, ["B02", "B03", "B04", "B08"], refl, bounds)
    _layer(wpath, d, ["SCL"], scl, bounds)


def _s1_packed(wpath, bounds, gap1=None):
    """sentinel1 in PerLayerStorage: one GeoTIFF of C x T bands (vv t0, vv t1, vh t0, vh t1), metadata.json with
    the counts, window_storage_meta.json with each group's timesteps (window_data_storage/per_layer.py)."""
    a = np.full((2, 2, 16, 16), 0.05, np.float32)
    if gap1 is not None:
        a[:, 1][:, gap1] = 0
    ldir = _layer(wpath, "sentinel1", ["vv", "vh"], a.reshape(4, 16, 16), bounds)
    bs = os.path.join(ldir, "vv_vh")
    json.dump({"num_channels": 2, "num_timesteps": 2, "timestamps": None}, open(os.path.join(bs, "metadata.json"), "w"))
    json.dump({"group_timestep_counts": [1, 1]}, open(os.path.join(bs, "window_storage_meta.json"), "w"))
    os.makedirs(os.path.join(wpath, "layers", "sentinel1.1"), exist_ok=True)
    open(os.path.join(wpath, "layers", "sentinel1.1", "completed"), "w").close()


def _with_inputs(ds, rng, config=True):
    """Windows A and B with probabilities, a label layer, sentinel2 with gaps and SCL clouds (4 groups in A, 3 in
    B) and sentinel1 packed per layer (2 groups)."""
    if config:
        os.makedirs(ds, exist_ok=True)
        json.dump(CONFIG, open(os.path.join(ds, "config.json"), "w"))
    rows, cols = np.mgrid[:16, :16]
    wa, wb = _window(ds, "g/a", A), _window(ds, "g/b", B)
    for w, b in ((wa, A), (wb, B)):
        _layer(w, "output", ["p0", "p1", "p2"], _softmax(rng, 3), b)
        _layer(w, "label", ["category"], np.ones((16, 16), np.int32), b)
    _s2_group(wa, 0, A, cloud=rows < 4)
    _s2_group(wa, 1, A, gap=cols < 8, cloud=rows < 4)
    _s2_group(wa, 2, A)
    _s2_group(wa, 3, A, gap=rows >= 12)
    corner = (rows < 4) & (cols < 4)
    for i in range(3):                                          # B lacks the fourth timestep
        _s2_group(wb, i, B, gap=corner)
    _s1_packed(wa, A, gap1=rows >= 8)
    _s1_packed(wb, B)


NAMES = {0: "sentinel1:all,sentinel2:all:clear", 1: "sentinel1:most,sentinel2:all:clear",
         8: "sentinel1:all,sentinel2:all:cloudy", 12: "sentinel1:all,sentinel2:most:clear",
         13: "sentinel1:most,sentinel2:most:clear", 16: "sentinel1:all,sentinel2:most:some-cloud",
         36: "sentinel1:all,sentinel2:none"}


def test_condition_layer_counts_gaps_and_clouds_on_the_scores_grid(tmp_path):
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    _with_inputs(ds, np.random.default_rng(9))
    s = oe.read_output(ds, out)
    c = oe.condition_from_inputs(ds, out)
    assert c["inputs"] == {"sentinel1": {"timesteps": 2, "clouds": False}, "sentinel2": {"timesteps": 4, "clouds": True}}
    assert c["codes"] == {str(k): v for k, v in NAMES.items()} and not c["notes"]
    g = c["grids"][0]
    with rasterio.open(s["grids"][0]["scores"]) as a, rasterio.open(g["condition"]) as b:
        assert a.shape == b.shape == (16, 32) and a.transform == b.transform and a.crs == b.crs
        assert b.dtypes[0] == "int32" and b.nodata == -1
        cond = b.read(1)
    want = np.full((16, 32), 12)
    want[0:4, 8:16], want[0:4, 0:8] = 8, 16                      # A: two clouded timesteps, one of them gapped
    want[4:8, 8:16], want[4:8, 0:8] = 0, 12
    want[8:12, 8:16], want[8:16, 0:8], want[12:16, 8:16] = 1, 13, 13
    want[0:4, 16:20] = 36                                         # B: no scene in any timestep; elsewhere 3 of 4
    np.testing.assert_array_equal(cond, want)
    assert g["codes"] == sorted(NAMES)
    assert g["condition_names"] == "--condition-names " + " ".join(f"{k}={v}" for k, v in sorted(NAMES.items()))
    assert "label" not in c["inputs"]                             # config.json gives it no data_source
    again = oe.condition_from_inputs(ds, str(tmp_path / "again"))
    with rasterio.open(again["grids"][0]["condition"]) as b:
        np.testing.assert_array_equal(b.read(1), cond)
    assert again["codes"] == c["codes"] and again["rule"] == c["rule"]


def test_condition_inputs_without_config_or_chosen(tmp_path):
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    _with_inputs(ds, np.random.default_rng(10), config=False)
    c = oe.condition_from_inputs(ds, out)
    assert list(c["inputs"]) == ["label", "sentinel1", "sentinel2"]
    assert any("no config.json" in n for n in c["notes"])
    assert c["inputs"]["sentinel2"]["clouds"]                    # the SCL directory spells its band
    only = oe.condition_from_inputs(ds, str(tmp_path / "only"), inputs=["sentinel2"])
    assert list(only["inputs"]) == ["sentinel2"]
    assert only["codes"] == {"0": "sentinel2:all:clear", "2": "sentinel2:all:cloudy", "3": "sentinel2:most:clear",
                             "4": "sentinel2:most:some-cloud", "9": "sentinel2:none"}
    with pytest.raises(ValueError, match="no window read holds the input layer 'landsat'"):
        oe.condition_from_inputs(ds, out, inputs=["landsat"])


def test_an_input_off_its_window_is_refused(tmp_path):
    ds = str(tmp_path / "ds")
    _with_inputs(ds, np.random.default_rng(14))
    shifted = (A[0] + 4, A[1], A[2] + 4, A[3])
    _layer(os.path.join(ds, "windows", "g", "a"), "sentinel2.2", ["B02", "B03", "B04", "B08"],
           np.full((4, 16, 16), 900, np.uint16), shifted)
    with pytest.raises(ValueError, match="window g/a: .*sentinel2.2.* not the window's output extent"):
        oe.condition_from_inputs(ds, str(tmp_path / "out"))


def test_too_many_conditions_are_refused(tmp_path, monkeypatch):
    ds = str(tmp_path / "ds")
    _with_inputs(ds, np.random.default_rng(11))
    monkeypatch.setattr(oe, "MAX_CONDITIONS", 3)
    with pytest.raises(ValueError, match="give 7 distinct conditions .* at most 3. Pass fewer input layers"):
        oe.condition_from_inputs(ds, str(tmp_path / "out"))


def test_from_olmoearth_prints_a_command_that_runs(tmp_path, capsys):
    """The printed next command runs as printed, and assess then ranks each input condition on its own."""
    ds, out = str(tmp_path / "ds"), str(tmp_path / "out")
    _with_inputs(ds, np.random.default_rng(12))
    capsys.readouterr()
    assert main(["from-olmoearth", ds, "--out", out, "--conditions"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == f"read 2 windows of {os.path.join(ds, 'windows')} (layer output): probabilities over 3 classes"
    assert printed[-1] == f"wrote {os.path.join(out, oe.SCORES_JSON)}, {oe.CONDITIONS_JSON}"
    nxt = [line for line in printed if line.startswith("next: ")]
    assert len(nxt) == 1
    argv = shlex.split(nxt[0][len("next: "):])
    assert argv[:3] == ["oe-inferencex", "assess", os.path.join(out, "scores_32610.tif")]
    assert "--condition" in argv and "--condition-names" in argv
    assert main(argv[1:]) == 0
    a = json.load(open(os.path.join(out, "assess_32610", "assessment.json")))
    assert set(a["conditions"]["per_condition"]) <= set(NAMES.values()) | {"unrecorded"}
    assert a["inputs"]["condition_names"] == {str(k): v for k, v in NAMES.items()}
    with pytest.raises(SystemExit, match="--inputs names the input layers"):
        main(["from-olmoearth", ds, "--out", out, "--inputs", "sentinel2"])
    with pytest.raises(SystemExit, match="--prob-property reads a per-window classification"):
        main(["from-olmoearth", ds, "--out", out, "--prob-property", "probs"])

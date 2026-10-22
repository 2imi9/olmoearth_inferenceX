"""from-embeddings on a local store laid out as Ai2's OlmoEarth v1.3 embeddings: one UTM-zone group with an embeddings
array over (time, band, y, x), a time array, the grid in proj:code and spatial:transform, and the signed-power
quantization in geoemb:quantization. No network: the store is written here."""
import csv
import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("sklearn")
rasterio = pytest.importorskip("rasterio")

from oe_inferencex.cli import main
from oe_inferencex.embeddings import dequantize

QUANT = {"method": "signed_power", "power": 2.0, "valid_range": [-127, 127], "nodata": -128}
X0, Y0, N = 499000.0, 4984000.0, 200          # a 2 km square of zone 33 near 15 E, 45 N


def _store(tmp_path):
    rng = np.random.default_rng(0)
    root = zarr.open_group(str(tmp_path / "emb.zarr"), mode="w")
    g = root.create_group("utm33")
    g.attrs.update({"proj:code": "EPSG:32633", "spatial:transform": [10, 0.0, X0, 0.0, -10, Y0],
                    "geoemb:quantization": QUANT})
    g.create_array("time", data=np.array([2024, 2025], dtype=np.int32))
    cls = np.zeros((N, N), int)
    cls[:, N // 2:] = 1                                    # west half one class, east half the other
    centres = rng.normal(0, 0.5, (2, 128))
    x = centres[cls].transpose(2, 0, 1) + rng.normal(0, 0.3, (128, N, N))
    q = np.clip(np.rint(127.5 * np.sign(x) * np.abs(x) ** 0.5), -127, 127).astype(np.int8)
    q[:, :10, :10] = -128                                  # a corner with no embedding
    emb = np.full((2, 128, N, N), -128, dtype=np.int8)
    emb[1] = q
    g.create_array("embeddings", data=emb, chunks=(1, 64, 100, 100))
    return tmp_path / "emb.zarr", cls


def _lonlat(rows, cols):
    tr = pyproj.Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    return tr.transform(X0 + (np.asarray(cols) + 0.5) * 10, Y0 - (np.asarray(rows) + 0.5) * 10)


def _points(tmp_path, cls, n=40, seed=1):
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(20, N - 20, n), rng.integers(20, N - 20, n)
    lon, lat = _lonlat(rows, cols)
    path = tmp_path / "points.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lon", "lat", "class"])
        for a, b, r, c in zip(lon, lat, rows, cols):
            w.writerow([a, b, ["crop", "forest"][cls[r, c]]])
    return path


def _bbox():
    lon, lat = _lonlat([0, N - 1], [0, N - 1])
    return [str(min(lon) + 1e-5), str(min(lat) + 1e-5), str(max(lon) - 1e-5), str(max(lat) - 1e-5)]


def test_dequantize_follows_the_store_formula_and_masks_no_data():
    q = np.array([-128, -127, 0, 64, 127], dtype=np.int8)
    v = dequantize(q, QUANT)
    assert np.isnan(v[0]) and v[2] == 0.0
    assert np.allclose(v[1:], np.sign(q[1:]) * (np.abs(q[1:]) / 127.5) ** 2)
    with pytest.raises(SystemExit, match="signed_power"):
        dequantize(q, {**QUANT, "method": "linear"})


def test_from_embeddings_writes_probabilities_assess_reads(tmp_path, capsys):
    store, cls = _store(tmp_path)
    pts = _points(tmp_path, cls)
    out = tmp_path / "scores.tif"
    assert main(["from-embeddings", "--store", str(store), "--points", str(pts), "--bbox", *_bbox(), "--year", "2025",
                 "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2 classes from 40 points" in text and "cannot say how accurate the map is" in text
    with rasterio.open(out) as src:
        p = src.read()
        assert src.crs.to_epsg() == 32633 and src.count == 2 and src.descriptions == ("crop", "forest")
    ok = np.isfinite(p).all(0)
    assert not ok.all() and ok.mean() > 0.95                  # the no-data corner stays no-data
    assert np.allclose(p[:, ok].sum(0), 1.0, atol=1e-5)
    side = json.load(open(tmp_path / "scores.classes.json"))
    assert side["classes"] == {"0": "crop", "1": "forest"} and side["year"] == 2025
    # the map follows the halves, and assess reads it with no further step
    hard = p.argmax(0)
    assert (hard[ok] == cls[: hard.shape[0], : hard.shape[1]][ok]).mean() > 0.95
    assert main(["assess", str(out), "--out", str(tmp_path / "audit")]) == 0


def test_from_embeddings_refusals(tmp_path):
    store, cls = _store(tmp_path)
    pts = _points(tmp_path, cls)
    base = ["from-embeddings", "--store", str(store), "--points", str(pts), "--out", str(tmp_path / "s.tif")]
    with pytest.raises(SystemExit, match="years 2024 to 2025"):
        main(base + ["--bbox", *_bbox(), "--year", "2023"])
    with pytest.raises(SystemExit, match="max-pixels"):
        main(base + ["--bbox", *_bbox(), "--year", "2025", "--max-pixels", "100"])
    with pytest.raises(SystemExit, match="spans UTM zones"):
        main(base + ["--bbox", "11.5", "45.0", "12.5", "45.1", "--year", "2025"])
    one = tmp_path / "one.csv"
    with open(pts) as src, open(one, "w") as dst:
        rows = list(csv.DictReader(src))
        w = csv.DictWriter(dst, fieldnames=["lon", "lat", "class"])
        w.writeheader()
        w.writerows({**r, "class": "crop"} for r in rows)
    with pytest.raises(SystemExit, match="at least two"):
        main(["from-embeddings", "--store", str(store), "--points", str(one), "--bbox", *_bbox(), "--year", "2025",
              "--out", str(tmp_path / "s.tif")])
    with pytest.raises(SystemExit, match=r"\.tif"):
        main(base[:-2] + ["--out", str(tmp_path / "s.npy"), "--bbox", *_bbox(), "--year", "2025"])

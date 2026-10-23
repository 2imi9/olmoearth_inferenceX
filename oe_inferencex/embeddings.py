"""From precomputed embeddings to a map this package can check: read an area of a geo-embeddings Zarr store, fit
logistic regression on labelled points, and write its class probabilities as a GeoTIFF on the store's grid.

Written for Ai2's OlmoEarth v1.3 embeddings (https://source.coop/olmoearth/olmoearth-v1-3-embeddings): one group per
UTM zone (`utm01` to `utm60`), each with an `embeddings` array over (time, band, y, x) and a `time` array of calendar
years, the grid in the group's `proj:code` and `spatial:transform`, and the quantization in `geoemb:quantization`
(the geo-embeddings Zarr convention). Any store laid out that way reads the same; another layout is refused.

Logistic regression because it writes probabilities and is among the best readouts in the v1.3 report's Table 1; a
kNN vote share takes k + 1 values and ties. The points the classifier is fitted on are not a random sample of the map
and the classifier has seen them, so they cannot say how accurate the map is: `sample` draws the windows to label for
that. exp112 graded this route at LCMAP's probability sample (2018, 1,000 random fit plots): the map's top
probability ranked its errors at AUROC 0.876. exp113 carried fits to held-out territories of Ai2's PLANTEUR benchmark:
the ranking held (AUROC 0.74 to 0.88), though in three of five the map called nearly everything the majority class.

Needs the `embeddings` extra (zarr, obstore, pyproj, scikit-learn) and the `geo` extra (rasterio).
"""
import csv
import json
import math

import numpy as np

STORE = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
ENDPOINT = "https://data.source.coop"
MAX_PIXELS = 1024 * 1024        # a 10 km square at 10 m: 0.5 GB of float32 at 128 dimensions


def _need(module, extra):
    try:
        return __import__(module)
    except ImportError as ex:  # pragma: no cover
        raise SystemExit(f"from-embeddings needs {module} (pip install 'olmoearth-inferencex[{extra}]')") from ex


def open_store(url=STORE, endpoint=ENDPOINT):
    """The store's root group: an s3:// URL through obstore, anonymous, or a local directory."""
    zarr = _need("zarr", "embeddings")
    if url.startswith("s3://"):
        from obstore.store import S3Store
        zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})    # small reads, as the dataset README advises
        s3 = S3Store.from_url(url, endpoint=endpoint, skip_signature=True)
        return zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")
    return zarr.open_group(url, mode="r")


def dequantize(q, quant):
    """Float embeddings from the stored integers, NaN at no-data, by the store's own `geoemb:quantization`."""
    if quant.get("method") != "signed_power":
        raise SystemExit(f"from-embeddings: the store's quantization is {quant.get('method')!r}; only signed_power "
                         "is read")
    scale = float(quant["valid_range"][1]) + 0.5            # 127.5 for int8, as the formula the store records
    v = np.asarray(q).astype(np.float32)
    v[np.asarray(q) == quant["nodata"]] = np.nan
    return (np.abs(v) / scale) ** float(quant["power"]) * np.sign(v)


def zone_name(lon):
    return f"utm{int((lon + 180) // 6) % 60 + 1:02d}"


def _grid(group):
    a, b, c, d, e, f = (float(t) for t in group.attrs["spatial:transform"])
    if b != 0.0 or d != 0.0:
        raise SystemExit("from-embeddings: the store's grid is rotated; only a north-up grid is read")
    return a, c, e, f


def read_area(root, bbox, year, dims=128, max_pixels=MAX_PIXELS):
    """(embeddings (dims, h, w) float with NaN at no-data, crs, rasterio Affine, zone name) for `bbox` = (west,
    south, east, north) in degrees. The box must lie in one UTM zone."""
    pyproj = _need("pyproj", "embeddings")
    from rasterio.transform import Affine
    west, south, east, north = (float(x) for x in bbox)
    if not (west < east and south < north):
        raise SystemExit(f"from-embeddings: --bbox is west south east north, got {bbox}")
    zone = zone_name(west)
    if zone_name(east - 1e-9) != zone:
        raise SystemExit(f"from-embeddings: the box spans UTM zones {zone} and {zone_name(east - 1e-9)}; each zone "
                         "has its own grid, so read one zone at a time")
    if zone not in root:
        raise SystemExit(f"from-embeddings: the store has no group {zone}")
    g = root[zone]
    crs = g.attrs["proj:code"]
    a, c, e, f = _grid(g)
    tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([west, east, east, west], [south, south, north, north])
    c0, c1 = int(math.floor((min(xs) - c) / a)), int(math.ceil((max(xs) - c) / a))
    r0, r1 = int(math.floor((max(ys) - f) / e)), int(math.ceil((min(ys) - f) / e))
    arr = g["embeddings"]
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, arr.shape[2]), min(c1, arr.shape[3])
    if (r1 - r0) * (c1 - c0) > max_pixels:
        raise SystemExit(f"from-embeddings: the box is {r1 - r0} x {c1 - c0} pixels, above --max-pixels {max_pixels}; "
                         "read a smaller box or raise the limit (memory grows with it)")
    years = [int(t) for t in g["time"][:]]
    if int(year) not in years:
        raise SystemExit(f"from-embeddings: the store holds the years {years[0]} to {years[-1]}, not {year}")
    if dims not in (64, 128) or dims > arr.shape[1]:
        raise SystemExit(f"from-embeddings: --dims is 64 or 128 (the nested widths), got {dims}")
    q = arr[years.index(int(year)), :dims, r0:r1, c0:c1]
    quant = g.attrs.get("geoemb:quantization") or root.attrs.get("geoemb:quantization")
    v = dequantize(q, quant)
    return v, crs, Affine(a, 0.0, c + c0 * a, 0.0, e, f + r0 * e), zone


def read_points(path):
    """[(lon, lat, label)] from a CSV with lon, lat and class columns."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {"lon", "lat", "class"} <= set(rows[0]):
        raise SystemExit(f"from-embeddings: {path} needs the columns lon, lat and class")
    return [(float(r["lon"]), float(r["lat"]), r["class"].strip()) for r in rows]


def fit_and_predict(v, crs, transform, points):
    """(probabilities (C, h, w) float32 with NaN at no-data, class names in band order, points used per class, notes)."""
    pyproj = _need("pyproj", "embeddings")
    _need("sklearn", "embeddings")
    from sklearn.linear_model import LogisticRegression
    notes = []
    tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = ~transform
    used, labels, rc = [], [], []
    outside = nodata = 0
    for lon, lat, lab in points:
        x, y = tr.transform(lon, lat)
        col, row = (int(math.floor(t)) for t in inv @ (x, y))
        if not (0 <= row < v.shape[1] and 0 <= col < v.shape[2]):
            outside += 1
            continue
        if np.isnan(v[:, row, col]).any():
            nodata += 1
            continue
        rc.append((row, col))
        labels.append(lab)
    if outside:
        notes.append(f"{outside} point{'s' if outside != 1 else ''} outside the box left out")
    if nodata:
        notes.append(f"{nodata} point{'s' if nodata != 1 else ''} on pixels with no embedding left out")
    names = sorted(set(labels))
    if len(names) < 2:
        raise SystemExit(f"from-embeddings: the points inside the box carry {len(names)} class; a classifier needs at "
                         "least two")
    y = np.array([names.index(lab) for lab in labels])
    counts = {n: int((y == i).sum()) for i, n in enumerate(names)}
    thin = [n for n, k in counts.items() if k < 5]
    if thin:
        notes.append(f"fewer than 5 points for {', '.join(thin)}: the classifier's probabilities for "
                     f"{'that class' if len(thin) == 1 else 'those classes'} rest on very little")
    rows, cols = np.array(rc).T
    clf = LogisticRegression(max_iter=2000).fit(v[:, rows, cols].T, y)
    flat = v.reshape(v.shape[0], -1)
    ok = ~np.isnan(flat).any(0)
    probs = np.full((len(names), flat.shape[1]), np.nan, dtype=np.float32)
    idx = np.flatnonzero(ok)
    for s in range(0, idx.size, 262144):                    # predict in blocks, so memory stays near the input's
        part = idx[s:s + 262144]
        probs[:, part] = clf.predict_proba(flat[:, part].T).T
    return probs.reshape(len(names), *v.shape[1:]), names, counts, notes


def write_scores(path, probs, crs, transform, names, meta):
    """The probabilities as a float32 GeoTIFF, NaN no-data, and a sidecar JSON naming each band's class."""
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=probs.shape[1], width=probs.shape[2], count=probs.shape[0],
                       dtype="float32", crs=crs, transform=transform, nodata=float("nan")) as dst:
        dst.write(probs)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
    side = path.rsplit(".", 1)[0] + ".classes.json"
    with open(side, "w") as fh:
        json.dump({**meta, "classes": {str(i): n for i, n in enumerate(names)}}, fh, indent=1)
    return side

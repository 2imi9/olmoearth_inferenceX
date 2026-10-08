"""Read an OlmoEarth run: the rslearn dataset olmoearth_run leaves behind, into the files `assess` reads.

olmoearth_run runs rslearn, which keeps a dataset as one directory per window (rslearn @ 5780896a:
rslearn/dataset/window.py:361-371, rslearn/dataset/storage/file.py:29-83 and 171-187):

    <ds>/config.json                                 the dataset's layers; an input layer has a data_source
    <ds>/windows/<group>/<name>/metadata.json        the window's projection (CRS, x and y resolution) and its bounds
                                                     in pixel coordinates of that projection
    <ds>/windows/<group>/<name>/layers/<layer>[.N]/  item group N of a layer (no suffix for N = 0), with a `completed`
                                                     marker once written; a raster band set in <band set>/geotiff.tif,
                                                     on the window's CRS and transform

The band-set directory is the band names joined by "_", or a sha256 hash when a name holds "_" or the joined name
exceeds 64 characters (rslearn/utils/raster_format.py:32-47), so the readers glob `<layer>/*/geotiff.tif`.

Three readers, on json, numpy and rasterio (the geo extra); no torch, no rslearn:

  read_output            a SegmentationTask's raster layer -> scores_<EPSG>.tif, one per CRS: the C probability bands
                         of every window pasted onto one grid, NaN where no window predicted
  read_window_probs      a ClassificationTask's per-window layer (data.geojson) -> window_probs.npy, (C, 1, N), and
                         window_probs.csv, which name each column; `assess --patch 1` reads the array
  condition_from_inputs  the input layers -> condition_<EPSG>.tif on the scores' grid: per pixel, how many of the
                         run's timesteps a scene covered and, with an SCL band, how many of those were under cloud

The output must hold probabilities. A SegmentationTask writes ONE band of argmax class ids by default, and the
softmax over C bands only with output_probs (rslearn/train/tasks/segmentation.py:52-55, 102-110, 198-232). No
published OlmoEarth project sets it, so read_output refuses a class-id layer and says how to get probabilities.
prob_scales multiplies the probabilities before they are written, so they no longer sum to 1: refused too, since
the largest of a rescaled vector is not the model's confidence. RasterMerger fills the pixels no crop covered with the
layer's nodata value or 0 (rslearn/train/prediction_writer.py:128-185), and no project sets one, so an all-zero vector
is read as uncovered.

What the condition layer captures. OlmoEarth masks a timestep that is missing for the whole window, but reads a pixel
no scene covered (0 in every band) and a cloudy pixel inside a present timestep as valid input. Those per-pixel
conditions are what the layer records, so that `assess --condition` ranks each with its own kind. It does not see
haze, smoke or snow that SCL does not flag as cloud, shadow or cirrus, a scene from another season than its period,
or anything about the model; without an SCL band it records coverage alone. No published project materializes SCL.
"""
import csv
import fnmatch
import glob
import hashlib
import json
import os
import re
import shlex

import numpy as np

from oe_inferencex.assess import MAX_CONDITIONS

SOURCE = "rslearn @ 5780896a, olmoearth_projects @ f3c9b0c8"
PROB_TOLERANCE = 1e-6       # as cli.PROB_TOLERANCE: float32 rounding puts a probability an ulp outside [0, 1]
SUM_TOLERANCE = 1e-3        # how far a pixel's probabilities may sum from 1; a float32 softmax is within about 1e-6
MAX_GRID_BYTES = 4 * 2 ** 30  # one CRS's grid in memory; windows spread far apart make a large, mostly empty grid
SCL_CLOUDY = (3, 8, 9, 10)  # Sentinel-2 scene classification: cloud shadow, cloud medium and high probability, cirrus
COVER = ("all", "most", "few", "none")      # a scene in every timestep, in at least half, in fewer, in none
CLOUD = ("clear", "some-cloud", "cloudy")   # of the covered timesteps: none cloudy, fewer than half, at least half
SCORES_JSON = "olmoearth_output.json"
CONDITIONS_JSON = "olmoearth_conditions.json"

FIX_PROBABILITIES = (
    "To write probabilities: in model.yaml set output_probs: true in the segmentation task's init_args "
    "(tasks.segment.init_args in OlmoEarth's configs) and leave prob_scales unset. A float32 output layer, as AWF's "
    "dataset declares, needs nothing more: the writer writes as many bands as the model outputs, whatever the "
    "layer's band list says (rslearn/utils/raster_format.py:574-603). Only an integer output layer truncates the "
    "probabilities to 0 and 1; then make its dtype float32 in the dataset's config.json, or give the RslearnWriter "
    "a layer_config with one float32 band set of C bands named without underscores (p0 ... p{C-1}) and a new layer "
    "name, read with --layer. Then rerun the inference stage (olmoearth_run's RUN_INFERENCE); a band set of other "
    "band names is written beside the old one in layers/<layer>/, not over it, so remove each window's old band-set "
    "directory first or write to a new layer")


def _rasterio():
    try:
        import rasterio
    except ImportError as ex:  # pragma: no cover
        raise ImportError("reading an rslearn dataset needs rasterio (pip install 'olmoearth-inferencex[geo]')") from ex
    return rasterio


def bandset_dirname(bands):
    """The directory rslearn stores a band set in: the band names joined by "_", or a sha256 hash of their JSON list
    when a name holds "_", or of the joined name when it exceeds 64 characters (rslearn/utils/raster_format.py:32-47)."""
    if any("_" in b for b in bands):
        return hashlib.sha256(json.dumps(list(bands)).encode()).hexdigest()
    name = "_".join(bands)
    return hashlib.sha256(name.encode()).hexdigest() if len(name) > 64 else name


def _bands_of_dirname(name):
    """The band names a band-set directory spells, or None when it is a hash."""
    return None if re.fullmatch(r"[0-9a-f]{64}", name) else name.split("_")


def assess_command(scores, out_dir, condition=None, names=None, patch=None):
    """The `assess` command that reads what these readers wrote, shell-quoted. It passes no --nodata: the scores'
    no-data is NaN, which every command reads as no-data whatever the flag (cli.read_raster), and the condition
    raster's -1 is its own file's no-data tag and negative, which records no condition either way."""
    cmd = ["oe-inferencex", "assess", scores, "--out", out_dir] + (["--patch", str(patch)] if patch else [])
    if condition:
        cmd += ["--condition", condition] + (["--condition-names", *names] if names else [])
    return " ".join(shlex.quote(c) for c in cmd)


# ----------------------------------------------------------------------------- windows
def _windows(ds, group=None, windows=None):
    """[(window id "group/name", path)], sorted by group then name: rslearn's windows/<group>/<name>/; with
    `windows`, only the ids matching one of those shell patterns (fnmatch, case-sensitive)."""
    root = os.path.join(ds, "windows")
    if not os.path.isdir(root):
        raise ValueError(f"{ds} has no windows/ directory, so it is not an rslearn dataset; olmoearth_run writes one "
                         "per run, its windows in windows/<group>/<name>/")
    groups = sorted(d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d)))
    if group is not None:
        want = [group] if isinstance(group, str) else list(group)
        missing = [g for g in want if g not in groups]
        if missing:
            raise ValueError(f"{ds}/windows has no group {missing[0]!r}; it has {', '.join(groups) or 'none'}")
        groups = [g for g in groups if g in want]
    found = [(f"{g}/{n}", os.path.join(root, g, n)) for g in groups for n in sorted(os.listdir(os.path.join(root, g)))
             if not n.startswith(".") and os.path.isdir(os.path.join(root, g, n))]
    if not found:
        raise ValueError(f"{ds}/windows holds no window" + (f" in group {group!r}" if group is not None else ""))
    if windows:
        pats = [windows] if isinstance(windows, str) else list(windows)
        found = [(w, p) for w, p in found if any(fnmatch.fnmatchcase(w, q) for q in pats)]
        if not found:
            raise ValueError(f"no window of {ds} matches {', '.join(pats)}; a pattern matches the window id "
                             "group/name, e.g. 'default/*_12_*'")
    return found


def _metadata(path):
    """A window's metadata.json (projection and pixel bounds), or None."""
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _completed_outputs(ds, layer, group, windows=None):
    """(raster windows, vector windows, skipped): the windows whose `layers/<layer>/` holds a `completed` marker and
    one GeoTIFF (a raster output) or a data.geojson (a vector output); every other window with the reason."""
    rasters, vectors, skipped = [], [], []
    for wid, path in _windows(ds, group, windows):
        ldir = os.path.join(path, "layers", layer)
        if not os.path.isdir(ldir):
            skipped.append({"window": wid, "reason": f"with no layers/{layer}/"})
            continue
        if not os.path.exists(os.path.join(ldir, "completed")):
            skipped.append({"window": wid, "reason": f"with layers/{layer}/ not marked completed: inference did not "
                                                     "finish there"})
            continue
        tifs = sorted(glob.glob(os.path.join(glob.escape(ldir), "*", "geotiff.tif")))
        if len(tifs) > 1:
            skipped.append({"window": wid, "reason": f"with {len(tifs)} band sets in layers/{layer}/ ("
                                                     + ", ".join(os.path.basename(os.path.dirname(t)) for t in tifs)
                                                     + "), where the writer writes one", "several_band_sets": True})
        elif tifs:
            rasters.append({"window": wid, "path": path, "tif": tifs[0]})
        elif os.path.exists(os.path.join(ldir, "data.geojson")):
            vectors.append({"window": wid, "path": path, "geojson": os.path.join(ldir, "data.geojson")})
        else:
            skipped.append({"window": wid, "reason": f"with layers/{layer}/ completed but holding neither a GeoTIFF "
                                                     "nor data.geojson"})
    return rasters, vectors, skipped


def _no_output(ds, layer, skipped):
    """The refusal when no window has a completed output layer, naming the layers that do exist."""
    seen = set()
    for _, path in _windows(ds):
        ldir = os.path.join(path, "layers")
        if os.path.isdir(ldir):
            seen |= {n.split(".")[0] for n in os.listdir(ldir) if not n.startswith(".")}
    why = {}
    for s in skipped:
        why[s["reason"]] = why.get(s["reason"], 0) + 1
    why = "; ".join(f"{n} window{'s' if n > 1 else ''} {r}" for r, n in why.items())
    if skipped and all(s.get("several_band_sets") for s in skipped):
        return ValueError(f"no window of {ds} has one band set in its completed {layer!r} layer ({why}). A rerun "
                          f"of the inference stage that writes other band names puts its band set beside the old "
                          f"one in layers/{layer}/ rather than over it, so which is the run's output is not "
                          f"recorded. Remove the old band-set directory from each window, or write the rerun to a "
                          f"new layer and name it with --layer")
    return ValueError(f"no window of {ds} has a completed {layer!r} layer ({why}). Run the inference stage "
                      f"first, or name the output layer with --layer; the windows hold the layers "
                      f"{', '.join(sorted(seen)) or 'none'}")


def output_kind(ds, layer="output", group=None, windows=None):
    """"raster" when a window's completed output layer holds a GeoTIFF (a SegmentationTask), "vector" when it holds
    data.geojson (a ClassificationTask); refused when no window has a completed output layer."""
    rasters, vectors, skipped = _completed_outputs(ds, layer, group, windows)
    if rasters:
        return "raster"
    if vectors:
        return "vector"
    raise _no_output(ds, layer, skipped)


# ----------------------------------------------------------------------------- one grid per CRS
def _plan_grids(rasters):
    """One grid per CRS, the union of its windows' extents, from each output GeoTIFF's own georeference. rslearn
    writes a window's raster at bounds x resolution (raster_format.py:579-587), so the windows of one CRS share a
    pixel grid; a raster off it, a rotated one, or windows of one CRS at two resolutions are refused. read_output and
    condition_from_inputs both call this, so the scores and the condition layer are on the same grid by
    construction. Each window gets its `slot` (row, col) on its grid."""
    rio = _rasterio()
    from rasterio.transform import Affine
    by_crs = {}
    for e in rasters:
        with rio.open(e["tif"]) as src:
            t = src.transform
            e.update(shape=(src.height, src.width), count=src.count, crs=src.crs, nodata=src.nodata,
                     transform=tuple(t)[:6])
        if e["crs"] is None:
            raise ValueError(f"window {e['window']}: {e['tif']} has no CRS")
        if t.b != 0 or t.d != 0:
            raise ValueError(f"window {e['window']}: {e['tif']} is rotated; rslearn writes north-up rasters")
        col, row = t.c / t.a, t.f / t.e
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise ValueError(f"window {e['window']}: {e['tif']} starts at pixel ({col:.6g}, {row:.6g}) of its "
                             "resolution, off the pixel grid rslearn writes on; it was not written by rslearn")
        e["origin"] = (int(round(row)), int(round(col)))
        epsg = e["crs"].to_epsg()
        by_crs.setdefault(("" if epsg is not None else e["crs"].to_string(), epsg or 0), []).append(e)
    grids = []
    for k, key in enumerate(sorted(by_crs)):
        es = by_crs[key]
        a, ey = es[0]["transform"][0], es[0]["transform"][4]
        off = sorted({(e["transform"][0], e["transform"][4]) for e in es
                      if not np.allclose((e["transform"][0], e["transform"][4]), (a, ey), rtol=1e-9, atol=0)})
        if off:
            raise ValueError(f"the windows in {es[0]['crs']} are at more than one resolution ({abs(a):g} and "
                             f"{abs(off[0][0]):g}); one grid per CRS needs one. Read the windows of each resolution "
                             "separately (--group)")
        r0 = min(e["origin"][0] for e in es)
        c0 = min(e["origin"][1] for e in es)
        r1 = max(e["origin"][0] + e["shape"][0] for e in es)
        c1 = max(e["origin"][1] + e["shape"][1] for e in es)
        for e in es:
            e["slot"] = (e["origin"][0] - r0, e["origin"][1] - c0)
        epsg = key[1] or None
        grids.append({"label": str(epsg) if epsg else f"crs{k}", "crs": es[0]["crs"], "epsg": epsg,
                      "transform": Affine(a, 0.0, c0 * a, 0.0, ey, r0 * ey), "shape": (r1 - r0, c1 - c0),
                      "windows": es})
    return grids


SPARSE_GRID = 0.5   # below this share of a grid covered by windows, a grid too large to hold is called sparse


def _check_size(grids, bytes_per_px, what):
    """Refuse a grid whose arrays (`bytes_per_px` bytes per pixel in all, `what` naming them) exceed MAX_GRID_BYTES,
    saying whether the windows lie far apart (they cover less than SPARSE_GRID of it) or the area itself is too
    large, and how to read it in parts."""
    for g in grids:
        H, W = g["shape"]
        if bytes_per_px * H * W <= MAX_GRID_BYTES:
            continue
        area = min(sum(e["shape"][0] * e["shape"][1] for e in g["windows"]), H * W)
        share = area / (H * W)
        head = (f"the windows in {g['crs']} span {H} x {W} px, {bytes_per_px * H * W / 2 ** 30:.1f} GiB as {what}, "
                f"above the {MAX_GRID_BYTES / 2 ** 30:g} GiB this reader holds in memory; the windows cover "
                f"{100 * share:.0f}% of that grid")
        if share < SPARSE_GRID:
            raise ValueError(head + ", so they lie far apart. Read one window group at a time (--group), or the "
                                    "windows of one area at a time (--window with patterns of group/name)")
        raise ValueError(head + ", so the area itself is larger than this reader holds. Read it in parts: the "
                                "windows of one part at a time (--window with patterns of group/name, e.g. by the "
                                "row or column in the window names), each part to its own --out")


def _write_tif(path, arr, grid, dtype, nodata, descriptions=None):
    rio = _rasterio()
    arr = np.asarray(arr)
    bands = arr if arr.ndim == 3 else arr[None]
    with rio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                  dtype=dtype, crs=grid["crs"], transform=grid["transform"], nodata=nodata, compress="deflate") as dst:
        dst.write(bands.astype(dtype))
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    return path


# ----------------------------------------------------------------------------- the output layer
def _class_ids(where, layer, vals):
    return ValueError(f"{where}: layers/{layer} is one band of class ids ({vals.size} distinct values: "
                      f"{', '.join(f'{x:g}' for x in vals[:6])}{' ...' if vals.size > 6 else ''}), the argmax a "
                      "SegmentationTask writes by default (rslearn/train/tasks/segmentation.py:198-232), with no "
                      "confidence to rank. " + FIX_PROBABILITIES)


def _probabilities(a, nodata, wid, layer):
    """(covered pixels, kind, largest |sum - 1|, every value 0 or 1) of one window's output (C, H, W), or a ValueError
    saying what the layer holds instead of probabilities. kind is "softmax" (C > 1), "one" (one band of probabilities
    of one class, output_class_idx) or None (no covered pixel). A pixel is uncovered where a band is not finite, where
    every band holds the raster's no-data value, or, for C > 1, where every band is 0, the fill RasterMerger leaves
    (a softmax never sums to 0). One band cannot tell that fill from a probability of exactly 0; there 0 is read as
    a probability, so that the most confident pixels are not dropped. One float band holding only 0 and 1 is not
    refused here: a saturated or empty window of a probability run holds that too, so the caller decides over every
    window (all01), as for C bands; one band of an integer type, or of whole numbers beyond 1, is class ids."""
    C = a.shape[0]
    covered = np.isfinite(a).all(axis=0) if a.dtype.kind == "f" else np.ones(a.shape[1:], bool)
    if nodata is not None and not np.isnan(nodata):
        covered &= (a != nodata).any(axis=0)
    if C > 1:
        covered &= (a != 0).any(axis=0)
    v = a[:, covered].astype(np.float64)
    if v.size == 0:
        return covered, None, 0.0, True
    all01 = bool(np.isin(v, (0.0, 1.0)).all())
    if C == 1 and (a.dtype.kind in "biu" or (not all01 and np.array_equal(v, np.rint(v)))):
        raise _class_ids(f"window {wid}", layer, np.unique(v))
    if a.dtype.kind in "biu":
        raise ValueError(f"window {wid}: layers/{layer} holds {C} bands of {a.dtype}: an integer layer truncates "
                         "probabilities to 0 and 1, so no confidence survives. " + FIX_PROBABILITIES)
    lo, hi = float(v.min()), float(v.max())
    # negative values first (logits, a regression output); then, for C bands, the sum, since a rescaled softmax
    # (prob_scales) can also run above 1 and is named by its cause
    if lo < -PROB_TOLERANCE or (C == 1 and hi > 1 + PROB_TOLERANCE):
        raise ValueError(f"window {wid}: layers/{layer} runs from {lo:.7g} to {hi:.7g}, which is not a probability; "
                         "a SegmentationTask with output_probs writes the softmax, and logits or a regression output "
                         "are not read here. " + FIX_PROBABILITIES)
    if C == 1:
        return covered, "one", 0.0, all01
    s = v.sum(axis=0)
    dev = float(np.abs(s - 1).max())
    if dev > SUM_TOLERANCE:
        raise ValueError(f"window {wid}: the {C} bands of layers/{layer} sum to {s.min():.4g} to {s.max():.4g} at "
                         f"covered pixels (values {lo:.4g} to {hi:.4g}), not 1, so they are not the softmax. prob_scales "
                         "multiplies the probabilities before they are written (rslearn/train/tasks/segmentation.py:"
                         "215-221), and a temperature or a rescaled head does the same; the largest of a "
                         "rescaled vector is not the model's confidence. Leave prob_scales unset and rerun the "
                         "inference stage")
    return covered, "softmax", dev, all01


def read_output(ds, out, layer="output", group=None, windows=None):
    """Read a raster output layer of an rslearn dataset into scores_<label>.tif, one per CRS, in `out`; the label is
    the EPSG code (crs<k> for a CRS without one). Returns the summary written to olmoearth_output.json.

    Guaranteed: every window whose layers/<layer>/ is marked completed and holds one GeoTIFF is read, and every other
    window is listed in `windows_skipped` with the reason; the scores are float32, C bands, NaN where no window has
    a covered pixel, on a grid whose transform puts each window's pixels where its own GeoTIFF puts them. Two
    windows may cover the same pixel only with identical values there; otherwise the read is refused, since choosing
    one prediction over another is not this reader's to do.
    Refused, with what to do: a layer of class ids (the default output), C bands of an integer type, values outside
    [0, 1], probabilities that do not sum to 1 within SUM_TOLERANCE (prob_scales), probabilities that are all 0 or
    1, windows with different band counts, a CRS at two resolutions, a grid above MAX_GRID_BYTES.
    Not checked: that the bands are in the order of the model's classes, or which class a band is.
    `windows`, shell patterns of window ids (group/name), reads only the matching windows."""
    rasters, vectors, skipped = _completed_outputs(ds, layer, group, windows)
    if not rasters:
        if vectors:
            raise ValueError(f"layers/{layer} holds data.geojson, a per-window classification; read it with "
                             "read_window_probs (the command line does so itself)")
        raise _no_output(ds, layer, skipped)
    skipped += [{"window": e["window"], "reason": f"with data.geojson in layers/{layer}/ where other windows hold a "
                                                  "GeoTIFF"} for e in vectors]
    grids = _plan_grids(rasters)
    counts = sorted({e["count"] for e in rasters})
    if len(counts) > 1:
        by = {c: next(e["window"] for e in rasters if e["count"] == c) for c in counts}
        raise ValueError(f"the windows' {layer} layers hold different numbers of bands: "
                         + ", ".join(f"{c} in {w}" for c, w in by.items()) + "; read the windows of one run")
    C = counts[0]
    _check_size(grids, 4 * C + 4, f"{C} float32 band(s) and an int32 owner")
    rio = _rasterio()
    kinds, devs, all01, n_uncovered, empty, seen01 = set(), [0.0], True, 0, [], set()
    arrays = []
    for g in grids:
        H, W = g["shape"]
        scores = np.full((C, H, W), np.nan, np.float32)
        owner = np.full((H, W), -1, np.int32)
        for k, e in enumerate(g["windows"]):
            with rio.open(e["tif"]) as src:
                a = src.read()
            cov, kind, dev, z = _probabilities(a, e["nodata"], e["window"], layer)
            n_uncovered += int((~cov).sum())
            if kind is None:
                empty.append(e["window"])
                continue
            kinds.add(kind)
            devs.append(dev)
            all01 &= z
            if kind == "one" and z:
                seen01 |= set(np.unique(a[0][cov]).tolist())
            r, c = e["slot"]
            h, w = e["shape"]
            region = scores[:, r:r + h, c:c + w]            # a view: assigning into it writes the grid
            who = owner[r:r + h, c:c + w]
            clash = cov & (who >= 0)
            if clash.any():
                old, new = region[:, clash], a[:, clash].astype(np.float32)
                if not np.array_equal(old, new):
                    bad = np.flatnonzero(~(old == new).all(axis=0))
                    other = g["windows"][int(who[clash][bad[0]])]["window"]
                    raise ValueError(f"windows {other} and {e['window']} both predict {int(clash.sum())} pixels and "
                                     f"differ on {bad.size} of them (by up to {float(np.abs(old - new).max()):.3g}); "
                                     "pasted onto one grid, a pixel can hold one prediction, and choosing between two "
                                     "is not this reader's to do. Read one window group at a time (--group), or "
                                     "leave out the overlapping windows")
            region[:, cov] = a[:, cov]
            who[cov] = k
        arrays.append((scores, owner))
    if not kinds:
        raise ValueError(f"no window has a covered pixel in layers/{layer}: every band is 0 or no-data everywhere")
    if all01 and C == 1:
        raise _class_ids("every window", layer, np.array(sorted(seen01)))
    if all01:
        raise ValueError(f"every probability in layers/{layer} is exactly 0 or 1, as one-hot vectors or a truncated "
                         "layer give, so there is no confidence to rank. " + FIX_PROBABILITIES)
    os.makedirs(out, exist_ok=True)
    kind = kinds.pop()                                  # one count of bands gives one kind
    notes = []
    if kind == "one":
        notes.append("one band of probabilities, the probability of one class (output_class_idx): assess reads it "
                     "as a two-class map, that class against the rest. A pixel no crop covered would hold 0, which "
                     "one band cannot tell from a probability of 0; it is read as a probability")
    if n_uncovered:
        notes.append(f"{n_uncovered} pixels inside the windows read hold no prediction (every band 0, the fill "
                     "RasterMerger leaves where no crop predicted, or no-data); they are NaN in the scores unless an "
                     "overlapping window predicts them")
    if empty:
        notes.append(f"{len(empty)} window(s) hold no covered pixel: {', '.join(empty[:5])}"
                     + (" ..." if len(empty) > 5 else ""))
    if len(grids) > 1:
        notes.append(f"the windows lie in {len(grids)} CRSs, written as one scores raster each; assess reads one at "
                     "a time")
    entries = []
    for g, (scores, owner) in zip(grids, arrays):
        path = os.path.join(out, f"scores_{g['label']}.tif")
        _write_tif(path, scores, g, "float32", float("nan"), [f"class {i}" for i in range(C)] if C > 1 else None)
        entries.append({"label": g["label"], "crs": g["crs"].to_string(), "epsg": g["epsg"], "scores": path,
                        "shape": list(g["shape"]), "transform": list(g["transform"])[:6],
                        "windows": [e["window"] for e in g["windows"]], "covered_pixels": int((owner >= 0).sum()),
                        "assess": assess_command(path, os.path.join(out, f"assess_{g['label']}"))})
    summary = {"dataset": os.path.abspath(ds), "layer": layer, "group": group, "window_patterns": windows,
               "source": SOURCE,
               "kind": "softmax over C bands" if kind == "softmax" else "one band: the probability of one class",
               "bands": C, "windows_read": len(rasters), "windows_skipped": skipped,
               "largest_sum_deviation": max(devs), "sum_tolerance": SUM_TOLERANCE, "uncovered_pixels": n_uncovered,
               "grids": entries, "notes": notes}
    with open(os.path.join(out, SCORES_JSON), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


# ----------------------------------------------------------------------------- per-window classification
def read_window_probs(ds, out, layer="output", prob_property="probs", group=None, windows=None,
                      class_property=None):
    """Read a per-window ClassificationTask layer, one data.geojson per window whose features carry the probability
    list under `prob_property` (rslearn/train/tasks/classification.py:44, 71-72, 186-235), into window_probs.npy,
    a (C, 1, N) float32 array with one column per feature, and window_probs.csv, whose `window_col` is that column
    and the column `assess --patch 1` writes in its review sets. Returns the summary written to
    olmoearth_output.json, plus `probs`, the (C, 1, N) array, and `names`, the window of each column ("group/name",
    with "#k" for the k-th feature when a window holds several).

    The class the task wrote (its property_name) is read beside the probabilities: `class_property`, or else the
    one property a feature carries besides `prob_property`. `assess` grades the argmax of the probabilities as the
    map's class, and the written class is not the argmax when a two-class task moves positive_class_threshold from
    0.5 (classification.py:207-215); so where one argmax index goes with two written classes, or two indices with
    one class, the read is refused. window_probs.csv records the written class beside the argmax.

    Refused: no feature carrying `prob_property` (the task writes the class alone unless prob_property is set),
    lists of different lengths, values outside [0, 1] or not summing to 1 within SUM_TOLERANCE, written classes
    that are not one-to-one with the argmax. Features without the property are left out and counted. The columns
    of the array are not neighbours on the ground, so the boundary cue `assess` computes between them means nothing
    here."""
    rasters, vectors, skipped = _completed_outputs(ds, layer, group, windows)
    if not vectors:
        if rasters:
            raise ValueError(f"layers/{layer} holds GeoTIFFs, a raster output; read it with read_output")
        raise _no_output(ds, layer, skipped)
    skipped += [{"window": e["window"], "reason": f"with a GeoTIFF in layers/{layer}/ where other windows hold "
                                                  "data.geojson"} for e in rasters]
    probs, rows, seen, n_missing, n_read, others = [], [], set(), 0, 0, set()
    for e in vectors:
        try:
            with open(e["geojson"]) as f:
                feats = json.load(f).get("features") or []
        except (OSError, ValueError, AttributeError) as exc:
            skipped.append({"window": e["window"], "reason": f"with an unreadable data.geojson ({exc})"})
            continue
        if not feats:
            skipped.append({"window": e["window"], "reason": "with a data.geojson that holds no feature"})
            continue
        n_read += 1
        meta = _metadata(e["path"]) or {}
        proj, b = meta.get("projection") or {}, meta.get("bounds")
        x = y = None
        if b and "x_resolution" in proj:
            x, y = (b[0] + b[2]) / 2 * proj["x_resolution"], (b[1] + b[3]) / 2 * proj["y_resolution"]
        for k, feat in enumerate(feats):
            props = feat.get("properties") or {}
            seen |= set(props)
            if prob_property not in props:
                n_missing += 1
                continue
            try:
                v = np.asarray(props[prob_property], dtype=np.float64)
            except (TypeError, ValueError):
                v = np.zeros((0, 0))
            if v.ndim != 1 or not np.isfinite(v).all():
                raise ValueError(f"window {e['window']}: {prob_property!r} is not a list of finite numbers")
            probs.append(v)
            rows.append({"window": e["window"], "feature": k, "crs": proj.get("crs"), "x": x, "y": y,
                         "props": props})
            others |= set(props) - {prob_property}
    if not probs:
        raise ValueError(f"no feature in layers/{layer}/data.geojson carries {prob_property!r} (they carry "
                         f"{', '.join(sorted(seen)) or 'no property'}). A ClassificationTask writes the probabilities "
                         "only with prob_property set (rslearn/train/tasks/classification.py:233-234): set "
                         "prob_property: \"probs\" in the task's init_args, as Forest Loss Driver does, and rerun the "
                         "inference stage; or name the property with --prob-property")
    sizes = sorted({v.size for v in probs})
    if len(sizes) > 1 or sizes[0] < 2:
        raise ValueError(f"the {prob_property!r} lists have {' and '.join(map(str, sizes))} entries; one class "
                         "probability per class, the same classes in every window, is what is read")
    P = np.stack(probs, axis=1)
    lo, hi = float(P.min()), float(P.max())
    s = P.sum(axis=0)
    if lo < -PROB_TOLERANCE or hi > 1 + PROB_TOLERANCE or float(np.abs(s - 1).max()) > SUM_TOLERANCE:
        raise ValueError(f"the {prob_property!r} lists run from {lo:.4g} to {hi:.4g} and sum to {s.min():.4g} to "
                         f"{s.max():.4g}, so they are not the softmax a ClassificationTask writes")
    argmax = [int(np.argmax(v)) for v in probs]
    notes = []
    cls = class_property or (next(iter(others)) if len(others) == 1 else None)
    written = [r["props"].get(cls) if cls else None for r in rows]
    if cls and all(c is None for c in written):
        raise ValueError(f"no feature carrying {prob_property!r} carries the class property {cls!r} (they carry "
                         f"{', '.join(sorted(others)) or 'nothing else'})")
    if cls:
        by_idx, by_cls = {}, {}
        for i, c in zip(argmax, written):
            if c is not None:
                key = json.dumps(c, sort_keys=True)
                by_idx.setdefault(i, set()).add(key)
                by_cls.setdefault(key, set()).add(i)
        split = sorted(i for i, cs in by_idx.items() if len(cs) > 1)
        merged = sorted(c for c, ix in by_cls.items() if len(ix) > 1)
        if split or merged:
            if split:
                what = f"argmax {split[0]} goes with the written classes {', '.join(sorted(by_idx[split[0]]))}"
            else:
                what = (f"the written class {merged[0]} goes with argmax "
                        + ", ".join(map(str, sorted(by_cls[merged[0]]))))
            raise ValueError(f"the class each feature carries under {cls!r} is not the argmax of its {prob_property!r} "
                             f"list: {what}. A two-class ClassificationTask with positive_class_threshold other than "
                             "0.5 writes the positive class wherever its probability reaches the threshold "
                             "(rslearn/train/tasks/classification.py:207-215), so the map the run published is not "
                             "the argmax assess grades. Leave positive_class_threshold at 0.5 and rerun the inference "
                             "stage, or name the class property with --class-property if it is another one")
    elif len(others) > 1:
        notes.append(f"the features carry {', '.join(sorted(others))} besides {prob_property!r}; which holds the "
                     "class is not known, so the written class is not checked against the argmax (--class-property "
                     "names it)")
    os.makedirs(out, exist_ok=True)
    npy = os.path.join(out, "window_probs.npy")
    np.save(npy, P[:, None, :].astype(np.float32))
    index = os.path.join(out, "window_probs.csv")
    with open(index, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_col", "window", "feature", "crs", "x", "y", "argmax", "class"])
        for i, (r, a, c) in enumerate(zip(rows, argmax, written)):
            w.writerow([i, r["window"], r["feature"], r["crs"], r["x"], r["y"], a,
                        "" if c is None else c if isinstance(c, (str, int, float)) else json.dumps(c)])
    notes.insert(0, "each column of window_probs.npy is one window's prediction; columns side by side are not "
                    "neighbours on the ground, so the boundary cue assess writes means nothing here; "
                    "window_probs.csv names each column, with the window's centre (x, y) in its CRS")
    if n_missing:
        notes.append(f"{n_missing} feature(s) carry no {prob_property!r} and are left out")
    summary = {"dataset": os.path.abspath(ds), "layer": layer, "group": group, "window_patterns": windows,
               "source": SOURCE, "kind": "per-window probabilities", "class_property": cls, "bands": int(P.shape[0]),
               "windows_read": n_read,
               "windows_skipped": skipped, "features": len(probs), "scores": npy, "index": index,
               "largest_sum_deviation": float(np.abs(s - 1).max()), "sum_tolerance": SUM_TOLERANCE,
               "assess": assess_command(npy, os.path.join(out, "assess"), patch=1), "notes": notes}
    with open(os.path.join(out, SCORES_JSON), "w") as f:
        json.dump(summary, f, indent=1)
    per = {}
    for r in rows:
        per[r["window"]] = per.get(r["window"], 0) + 1
    names = [r["window"] if per[r["window"]] == 1 else f"{r['window']}#{r['feature']}" for r in rows]
    return dict(summary, probs=P[:, None, :].astype(np.float32), names=names)


# ----------------------------------------------------------------------------- the input-condition layer
def _input_layers(ds, rasters, layer, inputs):
    """(input layer names, {name: {band-set directory: band names}}, notes). Given `inputs`, those; otherwise the
    raster layers of the dataset's config.json with a data_source (labels and outputs have none); without a
    config.json, every layer the windows hold but the output, said in a note."""
    held = set()
    for e in rasters:
        ldir = os.path.join(e["path"], "layers")
        if os.path.isdir(ldir):
            held |= {n.split(".")[0] for n in os.listdir(ldir) if not n.startswith(".")}
    cfg = None
    try:
        with open(os.path.join(ds, "config.json")) as f:
            cfg = (json.load(f) or {}).get("layers") or {}
    except (OSError, ValueError, AttributeError):
        cfg = None
    notes = []
    if inputs:
        names = sorted(set(inputs))
        if layer in names:
            raise ValueError(f"{layer!r} is the output layer, not an input")
        missing = [n for n in names if n not in held]
        if missing:
            raise ValueError(f"no window read holds the input layer {missing[0]!r}; they hold "
                             f"{', '.join(sorted(held)) or 'no layer'}")
    elif cfg is not None:
        declared = sorted(n for n, c in cfg.items() if isinstance(c, dict) and c.get("data_source")
                          and c.get("type", "raster") == "raster" and n != layer)
        names = [n for n in declared if n in held]
        if set(declared) - set(names):
            notes.append(f"config.json declares the input layer(s) {', '.join(sorted(set(declared) - set(names)))}, "
                         "which no window read holds")
    else:
        names = sorted(held - {layer})
        notes.append(f"{ds} has no config.json saying which layers are inputs, so every layer but {layer!r} is read "
                     f"as one ({', '.join(names) or 'none'}); pass --inputs to choose")
    if not names:
        raise ValueError(f"no input layer found in the windows read (they hold {', '.join(sorted(held)) or 'no layer'}); "
                         "name them with --inputs")
    bands = {n: {bandset_dirname(bs["bands"]): list(bs["bands"]) for bs in (cfg.get(n) or {}).get("band_sets") or []
                 if isinstance(bs, dict) and bs.get("bands")} if cfg else {} for n in names}
    return names, bands, notes


def _bandset_header(path, known):
    """(band names or None, timesteps per raster) of a band-set directory: names from config.json, or the directory's
    own spelling; the timesteps from rslearn's metadata.json beside a multi-timestep GeoTIFF
    (raster_format.py:621-630). Names that do not match the band count are dropped."""
    rio = _rasterio()
    d = os.path.basename(path)
    names = known.get(d) or _bands_of_dirname(d)
    T = 1
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            T = int(json.load(f).get("num_timesteps") or 1)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    with rio.open(os.path.join(path, "geotiff.tif")) as src:
        count = src.count
    if count % T:
        raise ValueError(f"{path}: {count} bands cannot hold the {T} timesteps its metadata.json records")
    if names is not None and len(names) * T != count:
        names = None
    return names, T


def _item_groups(e, name, known):
    """The item groups of input layer `name` in window e, as [{"idx", "bandsets": [(dir, band names, T)], "t": slice
    or None}]: one directory per group (layers/<name>.N/, rslearn's default PerItemGroupStorage), or one directory
    holding every group along the time axis with window_storage_meta.json (PerLayerStorage,
    rslearn/dataset/window_data_storage/per_layer.py), whose groups are then slices of it. A group counts when its
    `completed` marker exists and it holds a GeoTIFF."""
    ldir = os.path.join(e["path"], "layers")
    if not os.path.isdir(ldir):
        return []

    def done(idx):
        return os.path.exists(os.path.join(ldir, name if idx == 0 else f"{name}.{idx}", "completed"))

    def bandsets(gdir):
        return [(p, *_bandset_header(p, known)) for p in sorted(glob.glob(os.path.join(glob.escape(gdir), "*")))
                if os.path.isfile(os.path.join(p, "geotiff.tif"))]

    packed = sorted(glob.glob(os.path.join(glob.escape(os.path.join(ldir, name)), "*", "window_storage_meta.json")))
    if packed:
        with open(packed[0]) as f:
            counts = list(json.load(f)["group_timestep_counts"])
        sets = bandsets(os.path.join(ldir, name))
        bad = [p for p, _, T in sets if T != sum(counts)]
        if bad:
            raise ValueError(f"window {e['window']}: {bad[0]} packs {len(counts)} item groups of {sum(counts)} "
                             "timesteps in all, and its metadata.json does not record that many")
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        return [{"idx": i, "bandsets": sets, "t": slice(int(starts[i]), int(starts[i + 1]))}
                for i in range(len(counts)) if done(i) and sets]
    out = []
    pat = re.compile(re.escape(name) + r"(?:\.(\d+))?")
    for d in sorted(os.listdir(ldir)):
        m = pat.fullmatch(d)
        if m and done(int(m.group(1) or 0)):
            sets = bandsets(os.path.join(ldir, d))
            if sets:
                out.append({"idx": int(m.group(1) or 0), "bandsets": sets, "t": None})
    return sorted(out, key=lambda g: g["idx"])


def _read_group(e, group, rio):
    """(covered, cloudy or None) of one item group on the window's output grid: covered where any band of any band
    set is finite, non-zero and not the raster's no-data, rslearn leaving 0 where no scene covered the pixel; cloudy
    where its SCL band is cloud shadow, cloud or cirrus. Each band set is read onto the output's pixels by nearest
    neighbour; one whose CRS or extent is not the window's is refused."""
    from rasterio.enums import Resampling
    H, W = e["shape"]
    t = e["transform"]
    want = (t[2], t[5] + H * t[4], t[2] + W * t[0], t[5])      # left, bottom, right, top of the output raster
    covered, cloudy = np.zeros((H, W), bool), None
    for path, names, T in group["bandsets"]:
        tif = os.path.join(path, "geotiff.tif")
        with rio.open(tif) as src:
            got = tuple(src.bounds)
            if src.crs != e["crs"] or not np.allclose(got, want, rtol=0, atol=0.5 * abs(t[0])):
                raise ValueError(f"window {e['window']}: {tif} covers {got} in {src.crs}, not the window's output "
                                 f"extent {want} in {e['crs']}")
            a = src.read(out_shape=(src.count, H, W), resampling=Resampling.nearest)
            nodata = src.nodata
        a = a.reshape(src.count // T, T, H, W)
        if group["t"] is not None:
            a = a[:, group["t"]]
        ok = np.isfinite(a) & (a != 0) if a.dtype.kind == "f" else (a != 0)
        if nodata is not None and not np.isnan(nodata):
            ok &= a != nodata
        covered |= ok.any(axis=(0, 1))
        if names is not None and "SCL" in names:
            scl = np.rint(a[names.index("SCL")]).astype(np.int64)
            c = np.isin(scl, SCL_CLOUDY).any(axis=0)
            cloudy = c if cloudy is None else (cloudy | c)
    return covered, cloudy


def _code_name(code, layers, clouds):
    """The name of a combined condition code: one part per input layer, the first the least significant digit."""
    parts = []
    for n in layers:
        code, d = divmod(int(code), 10 if clouds[n] else 4)
        if not clouds[n]:
            parts.append(f"{n}:{COVER[d]}")
        else:
            parts.append(f"{n}:none" if d == 9 else f"{n}:{COVER[d // 3]}:{CLOUD[d % 3]}")
    return ",".join(parts)


def condition_rule(layers, timesteps, clouds):
    """The rule the codes follow, in words, as the JSON records it."""
    radix = " x ".join(f"{10 if clouds[n] else 4} ({n})" for n in layers)
    return (f"Per pixel and input layer, of the T timesteps of the run (the most completed item groups any window read "
            f"holds: {', '.join(f'{n} {timesteps[n]}' for n in layers)}): all = a scene covers the pixel in every one, "
            "most = in at least half, few = in fewer than half but one or more, none = in none; a timestep missing "
            "from a window counts as not covering it. A pixel is covered when any band is non-zero and not the "
            "raster's no-data; rslearn leaves 0 where no scene covered it. With an SCL band in every item group of a "
            f"layer, the covered timesteps are also counted for cloud (SCL {', '.join(map(str, SCL_CLOUDY))}: cloud "
            "shadow, cloud medium and high probability, thin cirrus): clear = none cloudy, some-cloud = fewer than "
            "half, cloudy = at least half. The code is a number in mixed radix, the layers in alphabetical order, the "
            f"first the least significant digit ({radix}); a layer's digit is its coverage (0 all, 1 most, 2 few, 3 "
            "none) or, with cloud, coverage x 3 + cloud for a covered pixel and 9 for none. -1: no window read covers "
            "the pixel, or two overlapping windows give it different codes.")


def condition_from_inputs(ds, out, inputs=None, layer="output", group=None, windows=None):
    """The input-condition layer of an rslearn dataset, on the grid read_output writes the scores on: per pixel and
    input layer, the share of the run's timesteps a scene covered, binned, and with an SCL band the share of those
    under cloud; combined into one integer code (condition_rule). Written as condition_<label>.tif (int32, -1 where
    no window read covers the pixel) in `out`, with olmoearth_conditions.json, which holds every code's name, the
    rule, and per grid the `--condition-names` argument naming the codes present and the `assess` command that reads
    both rasters. Returns that JSON's content.

    Guaranteed: the same windows, grid and placement as read_output (_plan_grids), and codes and names that depend
    only on the dataset, not on the order windows are listed. Refused: no input layer, an input raster not on its
    window's extent, more than MAX_CONDITIONS codes on one grid (pass fewer input layers). Not captured: anything
    that is not a gap or an SCL cloud class (condition_rule says what is), and the order of the timesteps."""
    rio = _rasterio()
    rasters, vectors, skipped = _completed_outputs(ds, layer, group, windows)
    if not rasters:
        if vectors:
            raise ValueError(f"layers/{layer} holds a per-window classification (data.geojson); the condition layer "
                             "is pasted onto the scores' grid, and that output has none")
        raise _no_output(ds, layer, skipped)
    grids = _plan_grids(rasters)
    _check_size(grids, 8 + 1 + 4, "an int64 code, a conflict mask and the int32 copy written")
    names, known, notes = _input_layers(ds, rasters, layer, inputs)
    groups = {(e["window"], n): _item_groups(e, n, known[n]) for e in rasters for n in names}
    timesteps = {n: max(len(groups[(e["window"], n)]) for e in rasters) for n in names}
    empty = [n for n in names if timesteps[n] == 0]
    if empty and (inputs or len(empty) == len(names)):
        raise ValueError(f"no window read holds a completed item group with a GeoTIFF of {', '.join(empty)}, so "
                         "there is no timestep to count")
    if empty:
        notes.append(f"{', '.join(empty)}: no window read holds a completed item group with a GeoTIFF; left out")
        names = [n for n in names if n not in empty]
    clouds = {n: all(any(bn is not None and "SCL" in bn for _, bn, _ in g["bandsets"])
                     for e in rasters for g in groups[(e["window"], n)]) for n in names}
    for n in names:
        some = any(bn is not None and "SCL" in bn for e in rasters for g in groups[(e["window"], n)]
                   for _, bn, _ in g["bandsets"])
        if some and not clouds[n]:
            notes.append(f"{n}: an SCL band is in some item groups and not in others, so its clouds are not read")
    radix = {n: 10 if clouds[n] else 4 for n in names}
    os.makedirs(out, exist_ok=True)
    entries, codes_all = [], set()
    for g in grids:
        H, W = g["shape"]
        cond = np.full((H, W), -1, np.int64)
        conflict = np.zeros((H, W), bool)
        for e in g["windows"]:
            h, w = e["shape"]
            code, place = np.zeros((h, w), np.int64), 1
            for n in names:
                gs = [_read_group(e, grp, rio) for grp in groups[(e["window"], n)]]
                T = timesteps[n]
                n_cov = sum((c.astype(np.int64) for c, _ in gs), np.zeros((h, w), np.int64))
                cover = np.where(n_cov == T, 0, np.where(2 * n_cov >= T, 1, np.where(n_cov > 0, 2, 3)))
                if clouds[n]:
                    n_cld = sum(((c & k).astype(np.int64) for c, k in gs), np.zeros((h, w), np.int64))
                    cloud = np.where(n_cld == 0, 0, np.where(2 * n_cld < n_cov, 1, 2))
                    digit = np.where(cover == 3, 9, cover * 3 + cloud)
                else:
                    digit = cover
                code += digit * place
                place *= radix[n]
            r, c = e["slot"]
            here = cond[r:r + h, c:c + w]                   # a view into the grid
            conflict[r:r + h, c:c + w] |= (here >= 0) & (here != code)
            here[...] = code
        cond[conflict] = -1
        present = sorted(int(v) for v in np.unique(cond[cond >= 0]))
        if len(present) > MAX_CONDITIONS:
            raise ValueError(f"the input layers give {len(present)} distinct conditions on the grid of {g['crs']}; "
                             f"assess reads at most {MAX_CONDITIONS}. Pass fewer input layers (--inputs)")
        codes_all |= set(present)
        path = _write_tif(os.path.join(out, f"condition_{g['label']}.tif"), cond.astype(np.int32), g, "int32", -1)
        items = [f"{v}={_code_name(v, names, clouds)}" for v in present]
        entries.append({"label": g["label"], "crs": g["crs"].to_string(), "condition": path, "codes": present,
                        "condition_names": " ".join(["--condition-names"] + [shlex.quote(i) for i in items]),
                        "conflicting_pixels": int(conflict.sum()),
                        "assess": assess_command(os.path.join(out, f"scores_{g['label']}.tif"),
                                                 os.path.join(out, f"assess_{g['label']}"), path, items)})
        if conflict.any():
            notes.append(f"{int(conflict.sum())} pixels of {g['label']} lie in overlapping windows whose inputs give "
                         "different codes; they are -1, unrecorded")
    res = {"dataset": os.path.abspath(ds), "layer": layer, "group": group, "window_patterns": windows,
           "source": SOURCE, "inputs": {n: {"timesteps": timesteps[n], "clouds": clouds[n]} for n in names},
           "rule": condition_rule(names, timesteps, clouds),
           "codes": {str(v): _code_name(v, names, clouds) for v in sorted(codes_all)},
           "windows_read": len(rasters), "windows_skipped": skipped, "grids": entries, "notes": notes}
    with open(os.path.join(out, CONDITIONS_JSON), "w") as f:
        json.dump(res, f, indent=1)
    return res

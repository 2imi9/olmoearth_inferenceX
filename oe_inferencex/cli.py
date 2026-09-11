"""Command line: a prediction raster in, the review set and its explanation out; two rasters in, their difference out.

    oe-inferencex assess  scores.tif --out DIR [--logits] [--patch 4] [--nodata V] [--reference labels.tif]
                          [--budgets 0.01 0.05 0.10] [--order confidence|boundary_first]
    oe-inferencex compare a.tif b.tif --out DIR [--patch 4] [--nodata V] [--labels labels.tif] [--groups ids.tif]
                          [--threshold 0.5]

Inputs are GeoTIFFs (any rasterio-readable raster) or .npy arrays: (H, W) for a binary map, (C, H, W) for per-class
scores or, for `compare`, an integer class map. No-data comes from the raster's own value, NaN, or --nodata. Outputs are
plain files the caller reads back: JSON summaries (assess.summary / compare's dict), CSVs of the review windows with
pixel and map coordinates, and rasters on the window grid (patch x patch pixels per window) when the input was one.
Nothing here narrates; the JSON is the evidence (docs/method/agent_integration.md).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

from oe_inferencex.assess import _pooled_argmax, assess_prediction, summary
from oe_inferencex.compare import compare_inferences
from oe_inferencex.explain import explain_review_set
from oe_inferencex.signals import boundary_indicator


# ----------------------------------------------------------------------------- IO
def read_raster(path, nodata=None):
    """(array, valid mask, geo) where geo is None for .npy and {transform, crs} for rasters."""
    if path.endswith(".npy"):
        a = np.load(path)
        valid = np.isfinite(a).all(axis=0) if a.ndim == 3 else np.isfinite(a)
        if nodata is not None:
            valid &= (a != nodata).all(axis=0) if a.ndim == 3 else (a != nodata)
        return a, valid, None
    try:
        import rasterio
    except ImportError as ex:  # pragma: no cover
        raise SystemExit("reading rasters needs rasterio (pip install 'olmoearth-inferencex[geo]'); .npy inputs need nothing") from ex
    with rasterio.open(path) as src:
        a = src.read()
        nd = src.nodata if nodata is None else nodata
        geo = {"transform": src.transform, "crs": src.crs}
    valid = np.isfinite(a).all(axis=0)
    if nd is not None:
        valid &= (a != nd).all(axis=0)
    if a.shape[0] == 1:
        a = a[0]
    return a, valid, geo


def write_raster(path, array, geo, patch, nodata):
    """A window-grid array as a GeoTIFF on the input's grid scaled by `patch`, or .npy without geo."""
    if geo is None:
        np.save(os.path.splitext(path)[0] + ".npy", array)
        return os.path.splitext(path)[0] + ".npy"
    import rasterio
    from rasterio.transform import Affine
    arr = np.asarray(array)
    dtype = "float32" if arr.dtype.kind == "f" else ("uint8" if arr.dtype == bool else "int32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs=geo["crs"], transform=geo["transform"] * Affine.scale(patch), nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def window_coords(rows, cols, geo, patch):
    """Pixel corner and map centre of each window."""
    pr, pc = rows * patch, cols * patch
    if geo is None:
        return pr, pc, None, None
    x, y = geo["transform"] * (pc + patch / 2, pr + patch / 2)
    return pr, pc, np.asarray(x), np.asarray(y)


def pool_valid(valid, patch):
    h, w = valid.shape[0] // patch * patch, valid.shape[1] // patch * patch
    return valid[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3)) >= 0.5


# ----------------------------------------------------------------------------- assess
def cmd_assess(args):
    scores, valid, geo = read_raster(args.scores, args.nodata)
    reference = None
    if args.reference:
        ref, rvalid, _ = read_raster(args.reference, None)
        reference = np.where(rvalid, np.rint(ref).astype(int), -1)
    out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid, reference=reference,
                            budgets=tuple(args.budgets), order=args.order)
    os.makedirs(args.out, exist_ok=True)
    s = summary(out)
    s["inputs"] = {"scores": os.path.abspath(args.scores), "logits": args.logits, "reference": os.path.abspath(args.reference) if args.reference else None}
    arr = out["arrays"]
    conf, bnd = arr["confidence"], arr["boundary"]
    written = {}
    for b, rs in out["review_sets"].items():
        rc = np.asarray(rs["windows_rowcol"], dtype=int).reshape(-1, 2)
        pr, pc, x, y = window_coords(rc[:, 0], rc[:, 1], geo, args.patch)
        path = os.path.join(args.out, f"review_set_{int(round(b * 100)):02d}pct.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "confidence", "boundary"])
            for i, (r, c) in enumerate(rc):
                w.writerow([i + 1, int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), float(conf[r, c]), float(bnd[r, c])])
        written[f"review_set_{b}"] = path
    written["suspicion"] = write_raster(os.path.join(args.out, "suspicion.tif"), np.where(arr["valid"], conf, np.nan).astype(np.float32), geo, args.patch, nodata=None)
    written["boundary"] = write_raster(os.path.join(args.out, "boundary.tif"), np.where(arr["valid"], bnd, np.nan).astype(np.float32), geo, args.patch, nodata=None)
    exp = explain_review_set(out)
    with open(os.path.join(args.out, "explanation.json"), "w") as f:
        json.dump(exp, f, indent=1)
    written["explanation"] = os.path.join(args.out, "explanation.json")
    s["files"] = written
    with open(os.path.join(args.out, "assessment.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(f"{s['n_windows']} windows of {args.patch} px; review sets " + ", ".join(f"{int(round(b * 100))}%: {rs['n_windows']}" for b, rs in out["review_sets"].items()) +
          f"; boundary windows {100 * s['boundary_window_fraction']:.1f}%" + (f"; against the reference: error capture " + ", ".join(f"{int(round(float(b) * 100))}% -> {v['errors_captured_fraction']:.2f}" for b, v in s["against_reference"]["error_capture_at_budget"].items()) if "against_reference" in s else "") +
          f"\nwrote {args.out}/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary")
    return 0


# ----------------------------------------------------------------------------- compare
def decisions(path, nodata, threshold):
    a, valid, geo = read_raster(path, nodata)
    if a.ndim == 3:
        hard, n_classes = a.argmax(0), a.shape[0]
    elif a.dtype.kind == "f" and not np.array_equal(a[valid], np.rint(a[valid])):
        hard, n_classes = (a > threshold).astype(int), 2
    else:
        hard = np.rint(a).astype(int)
        n_classes = int(hard[valid].max()) + 1 if valid.any() else 2
    return hard, valid, geo, n_classes


def cmd_compare(args):
    ha, va, geo, na = decisions(args.a, args.nodata, args.threshold)
    hb, vb, _, nb = decisions(args.b, args.nodata, args.threshold)
    if ha.shape != hb.shape:
        raise SystemExit(f"the two maps differ in shape: {ha.shape} vs {hb.shape}; compare needs identical grids")
    n_classes = max(na, nb, 2)
    a_w, b_w = _pooled_argmax(ha, n_classes, args.patch), _pooled_argmax(hb, n_classes, args.patch)
    ok = pool_valid(va & vb, args.patch)
    labels = groups = None
    if args.labels:
        lab, lv, _ = read_raster(args.labels, None)
        lab_w = _pooled_argmax(np.where(lv, np.rint(lab).astype(int), 0), n_classes, args.patch)
        ok &= pool_valid(lv, args.patch)
        labels = lab_w
    if args.groups:
        g, gv, _ = read_raster(args.groups, None)
        groups = _pooled_argmax(np.rint(g).astype(int), int(np.rint(g[gv]).max()) + 1, args.patch)
    cues = {"boundary_a": boundary_indicator(a_w) > 0, "boundary_b": boundary_indicator(b_w) > 0}
    out = compare_inferences(a_w, b_w, ok, groups=groups, labels=labels, cues=cues)
    os.makedirs(args.out, exist_ok=True)
    s = summary(out)
    s["inputs"] = {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "labels": os.path.abspath(args.labels) if args.labels else None, "patch_px": args.patch}
    s["files"] = {"disagreement": write_raster(os.path.join(args.out, "disagreement.tif"), out["arrays"]["disagree"], geo, args.patch, nodata=None)}
    rows, cols = np.nonzero(out["arrays"]["disagree"])
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    path = os.path.join(args.out, "differing_windows.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "a", "b"])
        for i, (r, c) in enumerate(zip(rows, cols)):
            w.writerow([int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), int(a_w[r, c]), int(b_w[r, c])])
    s["files"]["differing_windows"] = path
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(s, f, indent=1)
    where = s["where"] or {}
    print(f"{s['n_disagree']} of {s['n_windows']} windows differ ({100 * (s['disagreement_rate'] or 0):.2f}%)" +
          (f"; on a boundary of a {where['boundary_a']['enrichment']:.1f}x as often as the agreeing windows" if where.get("boundary_a", {}).get("enrichment") is not None else "") +
          (f"; with labels: a right on {100 * (s['graded']['which_side']['share_a_right'] or 0):.0f}%, b on {100 * (s['graded']['which_side']['share_b_right'] or 0):.0f}% of them" if s.get("graded") else "") +
          f"\nwrote {args.out}/comparison.json, differing_windows.csv, disagreement")
    return 0


# ----------------------------------------------------------------------------- entry
def build_parser():
    p = argparse.ArgumentParser(prog="oe-inferencex", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assess", help="rank the windows of one prediction map for review and explain them")
    a.add_argument("scores", help="raster or .npy: (H, W) binary probability or logit map, or (C, H, W) per-class scores")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--logits", action="store_true", help="the scores are logits (tie-free confidence); default probabilities")
    a.add_argument("--patch", type=int, default=4, help="window size in pixels (default 4)")
    a.add_argument("--nodata", type=float, default=None, help="no-data value (default: the raster's own, plus NaN)")
    a.add_argument("--reference", default=None, help="optional integer class raster treated as truth (the caveat applies)")
    a.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10], help="review budgets as fractions of windows")
    a.add_argument("--order", choices=("confidence", "boundary_first"), default="confidence", help="review order")
    a.set_defaults(func=cmd_assess)
    c = sub.add_parser("compare", help="measure how two inferences of the same scene differ")
    c.add_argument("a", help="first map: integer classes, a probability map (thresholded), or (C, H, W) scores (argmax)")
    c.add_argument("b", help="second map on the same grid")
    c.add_argument("--out", required=True, help="output directory")
    c.add_argument("--patch", type=int, default=4, help="window size in pixels (default 4)")
    c.add_argument("--nodata", type=float, default=None)
    c.add_argument("--threshold", type=float, default=0.5, help="threshold for a probability map")
    c.add_argument("--labels", default=None, help="optional integer class raster: adds which side is right and the cross-tab")
    c.add_argument("--groups", default=None, help="optional integer raster of group ids (tiles, events) for per-group rates")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

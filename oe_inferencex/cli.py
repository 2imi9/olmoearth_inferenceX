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
def _pct(x, nd=2):
    """A percentage, or the word undefined. `summary` turns NaN into None, and `None or 0` prints 0.00%, which reads as
    "they never disagree" when the truth is "this could not be computed". This package exists to keep those apart."""
    return "undefined" if x is None else f"{100 * float(x):.{nd}f}%"


def _check_scores(scores, valid, is_logit, path):
    """Refuse a file that is not what the flags say it is, rather than scoring it anyway.

    A hard class map is the likeliest file an operator has to hand, and `assess_prediction` would take the 2-D branch,
    threshold it at 0.5 and compute a confidence of 9.0 for a class index of 5, returning a full plausible review set
    with exit 0. Silence is worse than absence here: the operator dispatches a field team on a nonsense ordering with
    nothing to warn them."""
    if scores.ndim == 3 or is_logit:
        return
    v = scores[valid]
    if v.size == 0:
        raise SystemExit(f"{path}: no valid pixels")
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if lo < 0.0 or hi > 1.0:
        raise SystemExit(
            f"{path}: values run {lo:g} to {hi:g}, which is not a probability map. Pass --logits if these are logits, "
            f"or give a (C, H, W) per-class score map. For a hard class map with a separate confidence band use "
            f"assess_classmap in the Python API; the command line does not read one.")
    integral = np.array_equal(v, np.rint(v))
    if integral and len(np.unique(v)) > 2:
        raise SystemExit(
            f"{path}: {len(np.unique(v))} distinct integer values in [0, 1] is a class map, not a probability map. "
            f"See assess_classmap in the Python API.")


def cmd_assess(args):
    scores, valid, geo = read_raster(args.scores, args.nodata)
    _check_scores(scores, valid, args.logits, args.scores)
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
        # Two budgets that differ must not write one file. 0.001 and 0.004 both rounded to "00pct" and the second
        # silently destroyed the first, while the JSON went on naming two files that were one. Whole percents keep
        # their old zero-padded name; only the sub-percent budgets that used to collide get a decimal form.
        pct = b * 100
        tag = f"{int(round(pct)):02d}" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}".replace(".", "p")
        path = os.path.join(args.out, f"review_set_{tag}pct.csv")
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
    notes = []
    if args.labels:
        lab, lv, _ = read_raster(args.labels, None)
        lab_i = np.rint(lab).astype(int)
        if not lv.any():
            raise SystemExit(f"{args.labels}: no valid label pixels")
        # The label raster is pooled over ITS OWN class range, never the maps'. _pooled_argmax counts votes only over
        # range(n_classes), so pooling a 0-5 label raster with the maps' n_classes=3 silently dropped every pixel of
        # class 3 and above and let the window label fall to a surviving low index: measured at 44.9% of window labels
        # wrong, with exit 0 and no warning, on the very output that says which inference to believe.
        n_lab = int(lab_i[lv].max()) + 1
        # invalid label pixels must not vote; -1 is the non-voting code _assess uses, where 0 is a real class
        lab_w = _pooled_argmax(np.where(lv, lab_i, -1), max(n_lab, 2), args.patch)
        ok &= pool_valid(lv, args.patch)
        labels = lab_w
        if n_lab > n_classes:
            notes.append(f"the labels carry {n_lab} classes and the two maps predict at most {n_classes}; "
                         f"windows whose label is a class neither map can predict are counted wrong for both sides")
    if args.groups:
        g, gv, _ = read_raster(args.groups, None)
        if not gv.any():
            raise SystemExit(f"{args.groups}: no valid group pixels")
        gi = np.rint(g).astype(int)
        groups = _pooled_argmax(np.where(gv, gi, -1), int(gi[gv].max()) + 1, args.patch)
    cues = {"boundary_a": boundary_indicator(a_w) > 0, "boundary_b": boundary_indicator(b_w) > 0}
    out = compare_inferences(a_w, b_w, ok, groups=groups, labels=labels, cues=cues)
    os.makedirs(args.out, exist_ok=True)
    s = summary(out)
    s["inputs"] = {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "labels": os.path.abspath(args.labels) if args.labels else None, "patch_px": args.patch}
    if notes:
        s["notes"] = notes
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
    print(f"{s['n_disagree']} of {s['n_windows']} windows differ ({_pct(s['disagreement_rate'])})" +
          (f"; on a boundary of a {where['boundary_a']['enrichment']:.1f}x as often as the agreeing windows" if where.get("boundary_a", {}).get("enrichment") is not None else "") +
          (f"; with labels: a right on {_pct(s['graded']['which_side']['share_a_right'], 0)}, "
           f"b on {_pct(s['graded']['which_side']['share_b_right'], 0)} of them" if s.get("graded") else "") +
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

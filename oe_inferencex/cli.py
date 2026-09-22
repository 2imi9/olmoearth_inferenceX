"""Command line: a prediction raster in, the review set and its explanation out; two rasters in, their difference out.

    oe-inferencex assess  scores.tif --out DIR [--logits] [--patch 4] [--nodata V] [--reference labels.tif]
                          [--budgets 0.01 0.05 0.10] [--order confidence|boundary_first]
    oe-inferencex compare a.tif b.tif --out DIR [--patch 4] [--nodata V] [--labels labels.tif] [--groups ids.tif]
                          [--threshold T]
    oe-inferencex demo    [--out DIR] [--made-up]   a first run on the real sample map shipped with the package

Inputs are GeoTIFFs (any rasterio-readable raster) or .npy arrays: (H, W) for a binary map, (C, H, W) for per-class
scores or, for `compare`, an integer class map. `compare` also takes two continuous maps (a regression output) when the
cut-off is named with --threshold; without it a map outside [0, 1] is refused rather than cut at 0.5. No-data comes
from the raster's own value, NaN, or --nodata. Outputs are
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

from oe_inferencex.assess import _pool, _pool_valid, _pooled_argmax, assess_prediction, summary
from oe_inferencex import estimate as est
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
    if is_logit:
        return
    v = scores[:, valid] if scores.ndim == 3 else scores[valid]
    if v.size == 0:
        raise SystemExit(f"{path}: no valid pixels")
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if lo < 0.0 or hi > 1.0:
        raise SystemExit(
            f"{path}: values run {lo:g} to {hi:g}, which is not a probability map. Pass --logits if these are logits, "
            f"or give a (C, H, W) per-class score map. For a hard class map with a separate confidence band use "
            f"assess_classmap in the Python API; the command line does not read one.")
    if scores.ndim == 3:
        return
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
    # SUSPICION, not confidence. Until 2026-09-21 this wrote `conf`, so ranking the file descending, which is what
    # its name invites, returned the exact inverse of the review set: on a 256-window scene the top 13 of the
    # raster overlapped the 13-window 5% review set in 0 of 13. A reviewer opening it with a hot-is-bad ramp
    # inspected the windows the model was most confident about.
    written["suspicion"] = write_raster(os.path.join(args.out, "suspicion.tif"), np.where(arr["valid"], -conf, np.nan).astype(np.float32), geo, args.patch, nodata=None)
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
def decisions(path, nodata, threshold, notes=None):
    a, valid, geo = read_raster(path, nodata)
    if a.ndim == 3:
        hard, n_classes = a.argmax(0), a.shape[0]
    elif a.dtype.kind == "f" and not np.array_equal(a[valid], np.rint(a[valid])):
        # A continuous map outside [0, 1] under the default cut-off of 0.5 is one class everywhere on both sides, so
        # two regression outputs "never differ", with exit 0. The cut-off of a continuous map is the caller's to name.
        lo, hi = (float(np.nanmin(a[valid])), float(np.nanmax(a[valid]))) if valid.any() else (0.0, 1.0)
        continuous = lo < 0.0 or hi > 1.0
        if continuous and threshold is None:
            raise SystemExit(
                f"{path}: values run {lo:g} to {hi:g}, which is not a probability map. To compare two continuous maps "
                f"at a cut-off, name it with --threshold (for example --threshold 80); otherwise pass hard class maps "
                f"or (C, H, W) scores.")
        if continuous and notes is not None:
            notes.append(f"{os.path.basename(path)} is a continuous map cut at {threshold:g}: the comparison is of that one "
                         f"decision, and no recorded experiment grades it on a regression output")
        hard, n_classes = (a > (0.5 if threshold is None else threshold)).astype(int), 2
    else:
        hard = np.rint(a).astype(int)
        n_classes = int(hard[valid].max()) + 1 if valid.any() else 2
    return hard, valid, geo, n_classes


def cmd_compare(args):
    notes = []
    ha, va, geo, na = decisions(args.a, args.nodata, args.threshold, notes)
    hb, vb, _, nb = decisions(args.b, args.nodata, args.threshold, notes)
    if ha.shape != hb.shape:
        raise SystemExit(f"the two maps differ in shape: {ha.shape} vs {hb.shape}; compare needs identical grids")
    n_classes = max(na, nb, 2)
    a_w, b_w = _pooled_argmax(ha, n_classes, args.patch), _pooled_argmax(hb, n_classes, args.patch)
    ok = pool_valid(va & vb, args.patch)
    labels = groups = None
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
        lab_w = _pooled_argmax(np.where(lv, lab_i, -1), max(n_lab, 2), args.patch, empty=-1, tie=-1)   # no majority, no label
        ok &= pool_valid(lv, args.patch)
        n_tied = int((ok & (lab_w < 0)).sum())
        ok &= lab_w >= 0                                    # an evenly split label window has no label to grade
        labels = lab_w
        if n_lab > n_classes:
            notes.append(f"the labels carry {n_lab} classes and the two maps predict at most {n_classes}; "
                         f"windows whose label is a class neither map can predict are counted wrong for both sides")
        if n_tied:
            notes.append(f"{n_tied} windows have an evenly split label and no majority; they are left out of the grading")
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


def cmd_demo(args):
    from oe_inferencex.demo import run
    return run(out=args.out, seed=args.seed, made_up=args.made_up)


# ----------------------------------------------------------------------------- entry
# ----------------------------------------------------------------------------- sample / estimate
def _top1(scores, is_logit):
    """Per-pixel top-1 probability, the quantity the confidence design allocates by."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 2:
        p = 1 / (1 + np.exp(-scores)) if is_logit else scores
        return np.maximum(p, 1 - p)
    if is_logit:
        z = scores - scores.max(0, keepdims=True)
        return np.exp(z).max(0) / np.exp(z).sum(0)
    return scores.max(0)


def cmd_sample(args):
    """Which windows to label, written as a CSV with an empty `wrong` column for the reviewer, and a sidecar
    JSON carrying the design so `estimate` can give the rate the design earns."""
    scores, valid, geo = read_raster(args.scores, args.nodata)
    _check_scores(scores, valid, args.logits, args.scores)
    out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid)
    arr = out["arrays"]
    margin, valid_w = arr["confidence"], arr["valid"]
    p1 = np.where(valid, _top1(scores, args.logits), np.nan)
    p1_w = _pool_valid(p1, args.patch) if not valid.all() else _pool(p1, args.patch)
    hw, ww = margin.shape
    tiles = None
    if args.design == "tiles":
        t = max(1, args.tile)
        tiles = (np.arange(hw)[:, None] // t) * ((ww + t - 1) // t) + (np.arange(ww)[None, :] // t)
    try:
        sample = est.sample_for_estimation(margin, args.budget, design=args.design, p1=p1_w, tiles=tiles,
                                           per_tile=args.per_tile, valid=valid_w, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"sample: {exc}")
    idx = sample["indices"]
    rows, cols = np.divmod(idx, ww)
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    if os.path.isdir(args.out):
        raise SystemExit(f"--out {args.out} is a directory; name the CSV to write, for example {os.path.join(args.out, 'to_label.csv')}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "stratum", "confidence", "wrong"])
        strata = sample.get("strata")
        pos = {int(g): i for i, g in enumerate(sample.get("strata_of_population", []))}
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), None if strata is None else int(strata[pos[int(i)]]),
                        float(margin[r, c]), ""])
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    side.update({"scores": os.path.abspath(args.scores), "logits": args.logits, "patch": args.patch,
                 "grid": [int(hw), int(ww)], "csv": os.path.abspath(args.out),
                 # what a reviewer with a GIS needs to find a window: the CRS, the pixel size, and the window's
                 # footprint in ground units; x and y in the CSV are window centres in that CRS
                 "crs": None if geo is None else str(geo["crs"]),
                 "transform": None if geo is None else list(geo["transform"])[:6],
                 "pixel_size": None if geo is None else [abs(geo["transform"].a), abs(geo["transform"].e)],
                 "window_size_ground_units": None if geo is None else [abs(geo["transform"].a) * args.patch, abs(geo["transform"].e) * args.patch],
                 "xy_are": "window centres in the raster's CRS" if geo is not None else "absent: the input had no georeference",
                 "warnings": list(out.get("warnings", [])),
                 "how_to_label": "open each window, set wrong=1 if the map's class there is not what is on the ground, "
                                 "else 0; then: oe-inferencex estimate " + os.path.basename(args.out)})
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(side, f, indent=1)
    print(f"{len(idx)} windows to label of {sample['n_population']} valid ({args.design} design); wrote {args.out} and its .json. "
          f"Fill the `wrong` column with 1 or 0 per window, then run: oe-inferencex estimate {args.out}"
          + "".join(f"\nwarning: {w}" for w in out.get("warnings", []))
          + (f"\nnote: {sample['note']}" if "note" in sample else ""))
    return 0


def cmd_estimate(args):
    """The map's error rate with its interval, from a filled-in sample CSV and its sidecar design."""
    side_path = args.sample[:-4] + ".json" if args.sample.endswith(".csv") else args.sample + ".json"
    if not os.path.exists(side_path):
        raise SystemExit(f"{side_path} not found; `estimate` needs the sidecar `sample` wrote beside the CSV")
    with open(side_path) as f:
        side = json.load(f)
    with open(args.sample, newline="", encoding="utf-8-sig") as f:          # a spreadsheet's BOM is not a column name
        rows = list(csv.DictReader(f))
    need = {"index", "window_row", "window_col", "wrong"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{args.sample}: expected the columns `sample` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}. A spreadsheet saved with another delimiter "
                         "(semicolon) or with columns removed cannot be matched to its design")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])                # "73.0" after a spreadsheet round trip is 73
    except ValueError as exc:
        raise SystemExit(f"{args.sample}: the `index` column is not the one `sample` wrote: {exc}")
    if not np.array_equal(idx, np.asarray(side["indices"], int)):
        raise SystemExit("the CSV's rows do not match the design in its sidecar; label the file `sample` wrote, in order")
    blank = [i for i, r in enumerate(rows) if str(r.get("wrong", "")).strip() == ""]
    if blank:
        raise SystemExit(f"{len(blank)} of {len(rows)} windows have no `wrong` value (first at row {blank[0] + 2}); "
                         "every sampled window needs a 1 or a 0, or the design's interval is not the one you get")
    # Exactly 0 or 1. int(float(x)) would read a reviewer's "0.5" (not sure) as right and "1.9" as wrong, with
    # exit 0; a spreadsheet's "TRUE" is refused rather than guessed at.
    ok = {"0": 0, "1": 1, "0.0": 0, "1.0": 1}
    bad = [(i + 2, r["wrong"]) for i, r in enumerate(rows) if str(r["wrong"]).strip() not in ok]
    if bad:
        raise SystemExit(f"`wrong` must be exactly 1 or 0 per window; {len(bad)} row(s) are not, first at row {bad[0][0]}: "
                         f"{bad[0][1]!r}. A window you could not judge should be left out of the budget, not scored")
    wrong = np.array([ok[str(r["wrong"]).strip()] for r in rows])
    sample = {k: (np.asarray(v) if k in ("indices", "strata", "strata_of_population", "tiles") else v) for k, v in side.items()}
    try:
        res = est.estimate_error_rate(sample, wrong)
    except ValueError as exc:
        raise SystemExit(f"estimate: {exc}")
    res["sample"] = os.path.abspath(args.sample)
    out = args.out or (args.sample[:-4] + "_estimate.json" if args.sample.endswith(".csv") else args.sample + "_estimate.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    # the interval is printed as its two ends: it is not symmetric about the estimate (Wilson never is, and a
    # clipped one is not), so "estimate +/- x" would name an interval that is not the one written
    print(f"error rate {100 * res['estimate']:.1f}%, 95% interval {100 * res['low']:.1f}% to {100 * res['high']:.1f}% "
          f"(half-width {100 * res['half_width']:.1f} points), from {res['n_labelled']} labelled windows of {res['n_population']}; "
          f"{res['method']}" + (f"\nwarning: {res['warning']}" if "warning" in res else "") + f"\nwrote {out}")
    return 0


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
    d = sub.add_parser("demo", help="first run: audit the real sample map shipped with the package (or a made-up one) and draw the result")
    d.add_argument("--out", default="oe_inferencex_demo", help="output directory (default oe_inferencex_demo)")
    d.add_argument("--seed", type=int, default=0, help="seed of the made-up scene and of the random pick")
    d.add_argument("--made-up", action="store_true", help="audit a small synthetic water map instead of the real sample tile")
    d.set_defaults(func=cmd_demo)
    c = sub.add_parser("compare", help="measure how two inferences of the same scene differ")
    c.add_argument("a", help="first map: integer classes, a probability map (thresholded), or (C, H, W) scores (argmax)")
    c.add_argument("b", help="second map on the same grid")
    c.add_argument("--out", required=True, help="output directory")
    c.add_argument("--patch", type=int, default=4, help="window size in pixels (default 4)")
    c.add_argument("--nodata", type=float, default=None)
    c.add_argument("--threshold", type=float, default=None,
                   help="cut-off for a 2-D continuous map: 0.5 for a probability map when omitted; required for any other range")
    c.add_argument("--labels", default=None, help="optional integer class raster: adds which side is right and the cross-tab")
    c.add_argument("--groups", default=None, help="optional integer raster of group ids (tiles, events) for per-group rates")
    c.set_defaults(func=cmd_compare)
    sm = sub.add_parser("sample", help="which windows to label so that `estimate` can say how wrong the map is")
    sm.add_argument("scores", help="the same map `assess` takes: (H, W) probability or logit map, or (C, H, W) scores")
    sm.add_argument("--budget", type=int, required=True, help="number of windows to label (exp78 measured 300)")
    sm.add_argument("--out", required=True, help="CSV to write; a .json sidecar with the design goes beside it")
    sm.add_argument("--design", choices=("confidence", "proportional", "random", "tiles"), default="confidence",
                    help="confidence (default): stratified by margin, allocated from the model's own confidence; "
                         "tiles: how people actually label, with the cluster interval that requires")
    sm.add_argument("--tile", type=int, default=16, help="tiles design: tile side in windows (default 16)")
    sm.add_argument("--per-tile", type=int, default=16, help="tiles design: windows labelled per tile (default 16)")
    sm.add_argument("--logits", action="store_true")
    sm.add_argument("--patch", type=int, default=4)
    sm.add_argument("--nodata", type=float, default=None)
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=cmd_sample)
    e = sub.add_parser("estimate", help="the map's error rate with an interval, from the labelled sample CSV")
    e.add_argument("sample", help="the CSV `sample` wrote, with its `wrong` column filled in")
    e.add_argument("--out", default=None, help="JSON to write (default: <sample>_estimate.json)")
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

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

from oe_inferencex.assess import _boundary_valid, _pool, _pool_valid, _pooled_argmax, assess_prediction, summary
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
        ref, rvalid, geo_r = read_raster(args.reference, None)
        if ref.shape != scores.shape[-2:]:
            raise SystemExit(f"{args.reference} has shape {ref.shape}; the map is {scores.shape[-2:]}")
        _same_grid(geo, geo_r, args.reference)                  # compare --labels had this check; assess did not
        reference = np.where(rvalid, np.rint(ref).astype(int), -1)
    try:
        out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid, reference=reference,
                                budgets=tuple(args.budgets), order=args.order)
    except ValueError as exc:                                     # a named refusal, not a traceback
        raise SystemExit(f"assess: {exc}") from None
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
    ref_block = s.get("against_reference") or {}
    if "error_capture_at_budget" in ref_block:
        ref_text = "; against the reference: error capture " + ", ".join(
            f"{int(round(float(b) * 100))}% -> {v['errors_captured_fraction']:.2f}" for b, v in ref_block["error_capture_at_budget"].items())
    elif args.reference:
        # sparse point labels, an all-no-data or an evenly split reference: no window could be graded, which used to
        # crash here after the files were written (review of 2026-09-23)
        ref_text = "; against the reference: no window has both a prediction and a majority reference label, so nothing was graded"
    else:
        ref_text = ""
    print(f"{s['n_windows']} windows of {args.patch} px; review sets " + ", ".join(f"{int(round(b * 100))}%: {rs['n_windows']}" for b, rs in out["review_sets"].items()) +
          f"; boundary windows {100 * s['boundary_window_fraction']:.1f}%" + ref_text +
          f"\nwrote {args.out}/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary")
    return 0


# ----------------------------------------------------------------------------- compare
def decisions(path, nodata, threshold, notes=None):
    """(hard decisions, valid pixels, geo, n_classes, per-pixel confidence or None). The confidence breaks a tied
    window majority the way assess does; a hard class map carries none."""
    a, valid, geo = read_raster(path, nodata)
    if a.ndim == 3:
        hard, n_classes = a.argmax(0), a.shape[0]
        return hard, valid, geo, n_classes, np.nan_to_num(a.max(0).astype(np.float64), nan=0.0)
    if threshold is not None:
        # A named cut-off applies whatever the storage: percent cover held as int16 used to go to the class-map
        # branch, and 0-100 was compared as a 101-class majority with the cut-off ignored (503 windows "differ"
        # where cutting at 50 gives 74), exit 0 (audit 2026-09-22).
        lo, hi = (float(np.nanmin(a[valid])), float(np.nanmax(a[valid]))) if valid.any() else (0.0, 1.0)
        if (lo < 0.0 or hi > 1.0) and notes is not None:
            notes.append(f"{os.path.basename(path)} is a continuous map cut at {threshold:g}: the comparison is of that one "
                         f"decision, and no recorded experiment grades it on a regression output")
        af = a.astype(np.float64)
        return (af > threshold).astype(int), valid, geo, 2, np.nan_to_num(np.abs(af - threshold), nan=0.0)
    if a.dtype.kind == "f" and not np.array_equal(a[valid], np.rint(a[valid])):
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
        af = a.astype(np.float64)
        return (af > 0.5).astype(int), valid, geo, 2, np.nan_to_num(np.abs(af - 0.5), nan=0.0)
    hard = np.rint(a).astype(int)
    n_classes = int(hard[valid].max()) + 1 if valid.any() else 2
    return hard, valid, geo, n_classes, None


def _same_grid(geo_a, geo_b, what):
    """Refuse two rasters on different grids. Only shapes used to be compared, so a map shifted by 12.8 km, or in
    another CRS, was compared window by window as though co-registered."""
    if geo_a is None or geo_b is None:
        return
    ta, tb = np.asarray(tuple(geo_a["transform"])[:6], float), np.asarray(tuple(geo_b["transform"])[:6], float)
    # an absolute tolerance of a thousandth of a pixel: np.allclose's relative default allowed 50 m at a UTM northing
    # of 5e6 and 120 m at a Web Mercator easting of 1.2e7, so a map shifted by a whole window passed (review, 2026-09-23)
    atol = 1e-3 * max(abs(ta[0]), abs(ta[4]), 1e-12)
    if str(geo_a["crs"]) != str(geo_b["crs"]) or not np.allclose(ta, tb, rtol=0.0, atol=atol):
        raise SystemExit(f"{what} is not on the first map's grid (CRS {geo_b['crs']} vs {geo_a['crs']}, transform "
                         f"{tuple(geo_b['transform'])[:6]} vs {tuple(geo_a['transform'])[:6]}); resample it onto the same grid first")


def cmd_compare(args):
    notes = []
    ha, va, geo, na, ca = decisions(args.a, args.nodata, args.threshold, notes)
    hb, vb, geo_b, nb, cb = decisions(args.b, args.nodata, args.threshold, notes)
    if ha.shape != hb.shape:
        raise SystemExit(f"the two maps differ in shape: {ha.shape} vs {hb.shape}; compare needs identical grids")
    _same_grid(geo, geo_b, args.b)
    if not 1 <= args.patch <= min(ha.shape):
        raise SystemExit(f"--patch {args.patch} must be at least 1 and no larger than the map, {ha.shape[0]} x {ha.shape[1]} px")
    if args.labels_date and not args.labels:
        raise SystemExit("compare: --labels-date names the date of a --labels raster, and none was given")
    both = va & vb
    lab_i = lv = None
    if args.labels:
        lab, lv, geo_l = read_raster(args.labels, None)
        if lab.shape != ha.shape:
            raise SystemExit(f"{args.labels} has shape {lab.shape}; the maps are {ha.shape}")
        _same_grid(geo, geo_l, args.labels)
        lab_i = np.rint(lab).astype(int)
        lv = lv & (lab_i >= 0)                              # a negative code is unlabelled whatever the nodata tag says
        if not lv.any():
            raise SystemExit(f"{args.labels}: no valid label pixels")
    # One code space for both maps and the labels, decided before anything is pooled. Pooling counts one window grid
    # per class id up to the largest, so a stray code of 60000 took 26 s and 3.3 GB on a 512 x 512 map (review of
    # 2026-09-23); sparse or negative codes are remapped to 0..K-1 jointly and mapped back in the CSV.
    map_codes = np.unique(np.concatenate([ha[both].ravel(), hb[both].ravel()])) if both.any() else np.zeros(0, int)
    lab_codes = np.unique(lab_i[lv]) if lab_i is not None else np.zeros(0, int)
    all_codes = np.unique(np.concatenate([map_codes, lab_codes])).astype(int)
    if all_codes.size and (all_codes.min() < 0 or int(all_codes.max()) + 1 > 4 * max(all_codes.size, 16)):
        code_of = all_codes
        ha, hb = np.searchsorted(all_codes, ha), np.searchsorted(all_codes, hb)     # pixels outside `both` never vote
        if lab_i is not None:
            lab_i = np.where(lv, np.searchsorted(all_codes, lab_i), -1)
        n_classes = n_lab = max(int(all_codes.size), 2)
    else:
        code_of = None
        n_classes = max(na, nb, 2)
        # The label raster is pooled over ITS OWN class range, never the maps'. _pooled_argmax counts votes only over
        # range(n_classes), so pooling a 0-5 label raster with the maps' n_classes=3 silently dropped every pixel of
        # class 3 and above and let the window label fall to a surviving low index: measured at 44.9% of window labels
        # wrong, with exit 0 and no warning, on the very output that says which inference to believe.
        n_lab = max(int(lab_codes.max()) + 1, 2) if lab_codes.size else 2
    # No-data pixels do not vote (they used to argmax to class 0 and could decide a window). Both maps pool over the
    # pixels BOTH predicted: a window half under B's no-data used to be decided from 16 of A's pixels and 8 of B's,
    # and two identical maps then "differed" there. A tied window goes to the more confident voters when both maps
    # carry a confidence, as in assess; when either is a hard class map, which has none, a tied window is left out
    # of the comparison on both sides, because breaking it one way on one side and another way on the other made
    # a class map and its own probability version differ on 1.8% of windows (review of 2026-09-23).
    weighted = ca is not None and cb is not None
    a_w = _pooled_argmax(np.where(both, ha, -1), n_classes, args.patch, empty=-1, weights=ca if weighted else None,
                         tie=None if weighted else -2)
    b_w = _pooled_argmax(np.where(both, hb, -1), n_classes, args.patch, empty=-1, weights=cb if weighted else None,
                         tie=None if weighted else -2)
    n_tie_windows = int((pool_valid(both, args.patch) & ((a_w == -2) | (b_w == -2))).sum())
    if n_tie_windows:
        notes.append(f"{n_tie_windows} windows are split evenly between two classes on a map with no confidence to break "
                     "the tie; they are left out of the comparison")
    ok = pool_valid(va & vb, args.patch) & (a_w >= 0) & (b_w >= 0)
    if not ok.any():
        # assess refuses an empty map; compare used to exit 0 with "0 of 0 windows differ"
        raise SystemExit("compare: no window was predicted by both maps (all no-data, or every window tied), so there is "
                         "nothing to compare")
    labels = groups = None
    ok_graded = None
    if lab_i is not None:
        # invalid label pixels must not vote; -1 is the non-voting code _assess uses, where 0 is a real class
        lab_w = _pooled_argmax(np.where(lv, lab_i, -1), n_lab, args.patch, empty=-1, tie=-1)   # no majority, no label
        # The grading runs on the labelled windows; the label-free numbers stay on every window both maps predicted.
        # Until 2026-09-22 the label mask was ANDed into the whole comparison, so adding --labels changed the numbers
        # the docs call label-free and dropped unlabelled differing windows from the CSV and the raster.
        labelled = pool_valid(lv, args.patch)
        n_tied = int((ok & labelled & (lab_w < 0)).sum())
        ok_graded = ok & labelled & (lab_w >= 0)            # an evenly split label window has no label to grade
        labels = lab_w
        unpredicted = sorted(int(c) for c in set(lab_codes.tolist()) - set(map_codes.tolist()))
        if unpredicted:
            # named by class, not by the largest id: "the labels carry 4 classes" was said of a raster holding class 3
            notes.append(f"the labels hold class(es) {unpredicted[:10]}{' ...' if len(unpredicted) > 10 else ''} that neither "
                         "map predicts; windows labelled with them are counted wrong for both sides")
        if n_tied:
            notes.append(f"{n_tied} windows have an evenly split label and no majority; they are left out of the grading")
    if args.groups:
        g, gv, geo_g = read_raster(args.groups, None)
        if g.shape != ha.shape:
            raise SystemExit(f"{args.groups} has shape {g.shape}; the maps are {ha.shape}")
        _same_grid(geo, geo_g, args.groups)
        gi = np.rint(g).astype(int)
        gv = gv & (gi >= 0)                                 # a negative id is outside every zone
        if not gv.any():
            raise SystemExit(f"{args.groups}: no valid group pixels (ids must be non-negative)")
        # ids remapped to 0..K-1 first: pooling allocates one count per id up to the largest, which for ids near
        # 1e5 was one window-sized array per id. A window with no group pixel belongs to no group (-1), where it used
        # to be put in group 0, inventing or diluting that group.
        uid, inv = np.unique(gi[gv], return_inverse=True)
        gc = np.full(gi.shape, -1); gc[gv] = inv
        gw = _pooled_argmax(gc, len(uid), args.patch, empty=-1)
        groups = np.where(gw >= 0, uid[np.clip(gw, 0, None)], -1)
    # boundaries as assess draws them: a window with no prediction cannot disagree with its neighbour, so no-data
    # holes and the data's rim no longer manufacture boundary cues
    valid_w = pool_valid(va & vb, args.patch) & (a_w >= 0) & (b_w >= 0)
    cues = {"boundary_a": _boundary_valid(a_w, valid_w) > 0, "boundary_b": _boundary_valid(b_w, valid_w) > 0}
    # what a difference can mean depends on the dates the maps describe: across dates it can be real change on the
    # ground, and "which side is right" against one reference then needs that reference's date
    from oe_inferencex.compare import dates_reading
    dates = (args.date_a, args.date_b)
    try:
        reading = dates_reading(args.date_a, args.date_b, args.labels_date)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"compare: {exc}") from None
    if labels is not None and reading["status"] in ("different_time", "overlapping_time") and reading["labels"] is None:
        raise SystemExit(f"compare: the two maps describe {reading['a']} and {reading['b']}; pass --labels-date, because a "
                         "window that changed between the dates is right in one map and wrong in the other whatever "
                         "either model did")
    if labels is not None and reading["status"] == "partly_stated":
        raise SystemExit(f"compare: {reading['reading']}; give both --date-a and --date-b, or neither, before grading")
    out = compare_inferences(a_w, b_w, ok, groups=groups, cues=cues, dates=dates, labels_date=args.labels_date)
    if labels is not None:
        out["graded"] = compare_inferences(a_w, b_w, ok_graded, groups=groups, labels=labels, cues=cues,
                                           dates=dates, labels_date=args.labels_date)["graded"]
        notes.append(f"the graded block covers the {int(ok_graded.sum())} windows with a majority label; every other "
                     f"number covers all {int(ok.sum())} windows both maps predicted")
    if out["dates"]["status"] == "unstated":
        notes.append("the dates the two maps describe were not given (--date-a, --date-b); across dates a difference can "
                     "be real change on the ground rather than an error")
    if groups is not None:
        n_nogroup = int((ok & (groups < 0)).sum())
        for blk in (out, out.get("graded") or {}):
            if blk.get("per_group"):
                blk["per_group"].pop(-1, None)
        if out.get("graded") and out["graded"].get("per_group") is not None:
            from oe_inferencex.compare import over_groups
            out["graded"]["over_groups"] = over_groups({k: (v["corrected"] - v["broken"]) if v["n"] > 0 else float("nan")
                                                        for k, v in out["graded"]["per_group"].items()})
        if n_nogroup:
            notes.append(f"{n_nogroup} windows fall in no group and are counted overall but in no per-group rate")
    os.makedirs(args.out, exist_ok=True)
    s = summary(out)
    s["inputs"] = {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "labels": os.path.abspath(args.labels) if args.labels else None, "patch_px": args.patch,
                   "date_a": args.date_a, "date_b": args.date_b, "labels_date": args.labels_date}
    if notes:
        s["notes"] = notes
    # NaN where nothing was compared: a 0 there used to read as "agree" in any GIS
    dis = np.where(ok, out["arrays"]["disagree"].astype(np.float32), np.nan).astype(np.float32)
    s["files"] = {"disagreement": write_raster(os.path.join(args.out, "disagreement.tif"), dis, geo, args.patch, nodata=None)}
    rows, cols = np.nonzero(out["arrays"]["disagree"])
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    path = os.path.join(args.out, "differing_windows.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "a", "b"])
        for i, (r, c) in enumerate(zip(rows, cols)):
            ca_, cb_ = (int(a_w[r, c]), int(b_w[r, c])) if code_of is None else (int(code_of[a_w[r, c]]), int(code_of[b_w[r, c]]))
            w.writerow([int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), ca_, cb_])
    s["files"]["differing_windows"] = path
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(s, f, indent=1)
    where = s["where"] or {}
    print(f"{s['n_disagree']} of {s['n_windows']} windows differ ({_pct(s['disagreement_rate'])})" +
          (f"; on a boundary of a {where['boundary_a']['enrichment']:.1f}x as often as the agreeing windows" if where.get("boundary_a", {}).get("enrichment") is not None else "") +
          (f"; with labels: a right on {_pct(s['graded']['which_side']['share_a_right'], 0)}, "
           f"b on {_pct(s['graded']['which_side']['share_b_right'], 0)} of them" if s.get("graded") else "") +
          (f"\n{s['dates']['reading']}" if s["dates"]["status"] in ("different_time", "overlapping_time", "partly_stated") else "") +
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
                 "nodata": args.nodata,                   # estimate --per-class and certify recompute the map with it
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


def _labelled_sample(path, command):
    """A filled-in sample CSV and its sidecar design, checked: the rows are the design's, every `wrong` is 0 or 1.
    Returns (sidecar, rows, indices, sample dict for the estimators, wrong)."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    if not os.path.exists(side_path):
        raise SystemExit(f"{side_path} not found; `{command}` needs the sidecar `sample` wrote beside the CSV")
    with open(side_path) as f:
        side = json.load(f)
    with open(path, newline="", encoding="utf-8-sig") as f:                 # a spreadsheet's BOM is not a column name
        rows = list(csv.DictReader(f))
    need = {"index", "window_row", "window_col", "wrong"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected the columns `sample` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}. A spreadsheet saved with another delimiter "
                         "(semicolon) or with columns removed cannot be matched to its design")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])                # "73.0" after a spreadsheet round trip is 73
    except (ValueError, OverflowError) as exc:                              # "inf" is an OverflowError, not a ValueError
        raise SystemExit(f"{path}: the `index` column is not the one `sample` wrote: {exc}")
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
    return side, rows, idx, sample, wrong


def _rounding_tolerance(text):
    """Half a unit in the last digit a number was written with, and never less than 1e-6 (a full-precision value
    still carries float noise from the round trip). "0.412" gives 5e-4; "4.12e-01" the same."""
    t = str(text).strip().lower()
    mant, _, exp = t.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return max(0.5 * 10.0 ** (-decimals + (int(exp) if exp else 0)), 1e-6)


def _map_windows(side, scores_override, nodata, command, rows=None, idx=None):
    """The map's per-window confidence, class and validity, recomputed from the scores the sample was drawn on, and
    checked to be that map: its population must be the sample's, and its confidence at the sampled windows the
    CSV's (review of 2026-09-23: another map on the same grid was certified from this map's labels, and a sample
    drawn with --nodata was recomputed without it, with no message). Returns (confidence, class, valid, n_classes)."""
    path = scores_override or side.get("scores")
    if not path or not os.path.exists(path):
        raise SystemExit(f"{command}: the map's scores are needed again ({path or 'no path in the sidecar'} not found); "
                         "pass --scores with the raster `sample` was run on")
    if nodata is None:
        nodata = side.get("nodata")                           # the value the sample was drawn with
    elif side.get("nodata") is not None and float(side["nodata"]) != float(nodata):
        raise SystemExit(f"{command}: --nodata {nodata:g} differs from the {side['nodata']:g} the sample was drawn with")
    scores, valid, _ = read_raster(path, nodata)
    _check_scores(scores, valid, bool(side.get("logits", False)), path)
    out = assess_prediction(scores, is_logit=bool(side.get("logits", False)), patch=int(side.get("patch", 4)), nodata_mask=~valid)
    arr = out["arrays"]
    if list(arr["confidence"].shape) != list(side.get("grid", arr["confidence"].shape)):
        raise SystemExit(f"{command}: {path} pools to a {arr['confidence'].shape} window grid, but the sample was drawn on "
                         f"{side.get('grid')}; pass the raster the sample was drawn on")
    conf, valid_w = arr["confidence"], arr["valid"]
    n_pop = int((valid_w & np.isfinite(conf)).sum())
    if side.get("n_population") is not None and n_pop != int(side["n_population"]):
        raise SystemExit(f"{command}: {path} has {n_pop} valid windows, but the sample was drawn from {side['n_population']}; "
                         "pass the raster, and the --nodata value, the sample was drawn with")
    if rows is not None and idx is not None and "confidence" in rows[0]:
        try:
            written = np.array([float(r["confidence"]) for r in rows])
            # the tolerance follows the digits the CSV holds: a spreadsheet that saved 0.412 for 0.41234 is still
            # the same map (the verification of the second review found three digits refused at a fixed 1e-4)
            tol = np.array([_rounding_tolerance(r["confidence"]) for r in rows])
        except ValueError:
            written = None
        if written is not None:
            off = np.abs(conf.ravel()[idx] - written)
            if not (off <= tol).all():
                k = int(np.argmax(off))
                raise SystemExit(f"{command}: the map's confidence at the sampled windows is not the one the CSV records "
                                 f"(row {k + 2}: {written[k]:.6g} in the CSV, {conf.ravel()[idx][k]:.6g} now); this is not the "
                                 "map the sample was drawn on")
    n_classes = int(scores.shape[0]) if scores.ndim == 3 else 2
    return conf, arr["pooled_argmax"], valid_w, n_classes


def _reference_classes(rows, path):
    """The reviewer's class per labelled window from a `reference_class` column, integers >= 0, none blank."""
    if "reference_class" not in rows[0]:
        raise SystemExit(f"{path}: --per-class needs a `reference_class` column holding the class the reviewer saw in each "
                         "window (an integer in the map's class ids); add it beside `wrong`")
    vals = []
    for i, r in enumerate(rows):
        s = str(r["reference_class"]).strip()
        try:
            v = int(float(s))
            if v < 0 or float(s) != v:
                raise ValueError
        except (ValueError, OverflowError):                  # "inf" raised an OverflowError traceback until 2026-09-23
            raise SystemExit(f"{path}: `reference_class` must be a whole number >= 0 in every row; row {i + 2} holds {s!r}")
        vals.append(v)
    return np.array(vals)


def cmd_estimate(args):
    """The map's error rate with its interval, from a filled-in sample CSV and its sidecar design; with
    --per-class, the user's and producer's accuracy and error-adjusted share per class as well."""
    side, rows, idx, sample, wrong = _labelled_sample(args.sample, "estimate")
    try:
        res = est.estimate_error_rate(sample, wrong)
    except ValueError as exc:
        raise SystemExit(f"estimate: {exc}")
    per_class_text = ""
    if args.per_class:
        ref = _reference_classes(rows, args.sample)
        _, hard, valid_w, n_classes = _map_windows(side, args.scores, args.nodata, "estimate --per-class", rows, idx)
        if (ref >= n_classes).any():
            k = int(np.argmax(ref >= n_classes))
            raise SystemExit(f"estimate --per-class: row {k + 2} gives reference_class {int(ref[k])}, but the map has "
                             f"{n_classes} classes (0 to {n_classes - 1}); a class id outside them added empty rows")
        map_class = np.where(valid_w, hard, -1).ravel()
        # row by row: the count alone matched when `wrong` marked rows 0-9 and the classes disagreed on rows 10-19
        mismatch = np.flatnonzero((map_class[idx] != ref).astype(int) != wrong)
        try:
            pc = est.estimate_per_class(sample, ref, map_class, n_classes=n_classes)
        except ValueError as exc:
            raise SystemExit(f"estimate --per-class: {exc}")
        res["per_class"] = pc["per_class"]
        res["confusion_counts"] = pc["confusion_counts"]
        res["overall_accuracy"] = pc["overall_accuracy"]
        res["per_class_method"] = pc["method"]
        if "warning" in pc:
            res["per_class_warning"] = pc["warning"]
        if pc.get("overall_accuracy_post_stratified") is not None:
            res["overall_accuracy_post_stratified"] = pc["overall_accuracy_post_stratified"]
        if mismatch.size:
            res["per_class_note"] = (f"on {mismatch.size} row(s), first row {int(mismatch[0]) + 2}, `wrong` disagrees with whether "
                                     "`reference_class` differs from the map's class; the error rate above uses `wrong`, the "
                                     "per-class table uses `reference_class`")
        lines = []
        for c, row in pc["per_class"].items():
            ua, pa, sh = row["user_accuracy"], row["producer_accuracy"], row["reference_share"]
            f = lambda v: "n/a" if v is None else f"{100 * v['estimate']:.0f}% ({100 * v['low']:.0f}-{100 * v['high']:.0f})"
            # the tag names the warning's reason; it used to say "few labels" for a near-census class of 500 labels
            tag = f"  [warning: {', '.join(row.get('warning_codes', ['see the JSON']))}]" if "warning" in row else ""
            lines.append(f"  class {c}: user's accuracy {f(ua)}, producer's {f(pa)}, share of map {100 * row['map_share']:.1f}% "
                         f"-> error-adjusted {f(sh)}" + tag)
        per_class_text = "\n" + "\n".join(lines) + (f"\nnote: {res['per_class_note']}" if "per_class_note" in res else "")
    res["sample"] = os.path.abspath(args.sample)
    out = args.out or (args.sample[:-4] + "_estimate.json" if args.sample.endswith(".csv") else args.sample + "_estimate.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    # the interval is printed as its two ends: it is not symmetric about the estimate (Wilson never is, and a
    # clipped one is not), so "estimate +/- x" would name an interval that is not the one written
    print(f"error rate {100 * res['estimate']:.1f}%, 95% interval {100 * res['low']:.1f}% to {100 * res['high']:.1f}% "
          f"(half-width {100 * res['half_width']:.1f} points), from {res['n_labelled']} labelled windows of {res['n_population']}; "
          f"{res['method']}" + (f"\nwarning: {res['warning']}" if "warning" in res else "") + per_class_text
          + (f"\nwarning: {res['per_class_warning']}" if "per_class_warning" in res else "") + f"\nwrote {out}")
    return 0


def cmd_certify(args):
    """The largest most-confident share of the map that is wrong at most --alpha of the time, certified from a
    random labelled sample so that the statement fails with probability at most --delta (exp80)."""
    side, rows, idx, sample, wrong = _labelled_sample(args.sample, "certify")
    if side.get("design") != "random":
        raise SystemExit(f"certify needs a random sample: this CSV was drawn with the {side.get('design')!r} design. The "
                         "guarantee rests on the labelled windows inside each zone being a random sample of that zone, "
                         "which a stratified or tile draw is not. Draw one with `sample --design random`")
    margin, hard, valid_w, _ = _map_windows(side, args.scores, args.nodata, "certify", rows, idx)
    try:
        res = est.certify_zone(margin.ravel(), idx, wrong, args.alpha, delta=args.delta, rule=args.rule, valid=valid_w.ravel())
    except ValueError as exc:
        raise SystemExit(f"certify: {exc}")
    hw, ww = margin.shape
    out = args.out or (args.sample[:-4] + "_zone.json" if args.sample.endswith(".csv") else args.sample + "_zone.json")
    mask_path = out[:-5] + ".npy" if out.endswith(".json") else out + ".npy"
    if res["coverage"] is not None:
        zone = np.zeros(hw * ww, bool)
        zone[np.asarray(res.pop("zone_indices_in_order"), int)] = True
        np.save(mask_path, zone.reshape(hw, ww))
        res["zone_mask"] = os.path.abspath(mask_path)
    elif os.path.exists(mask_path):
        os.remove(mask_path)          # a previous run's zone must not sit beside a result that certifies none
    res["sample"] = os.path.abspath(args.sample)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    if res["coverage"] is None:
        print(f"no zone certified at alpha={args.alpha:g}, delta={args.delta:g} from {res['n_labelled']} labels. {res['note']}\nwrote {out}")
    else:
        print(f"the {100 * res['coverage']:.0f}% most confident windows ({res['n_zone']} of {res['n_population']}, confidence margin >= "
              f"{res['threshold']:.4f}) are wrong at most {100 * args.alpha:g}% of the time; this statement fails on at most "
              f"{100 * args.delta:g}% of samples like this one ({args.rule} rule; the exact upper bound on the zone's error "
              f"rate at that level is {100 * res['upper_bound']:.1f}%). Outside the zone nothing is certified.\n"
              f"{res['note']}\nwrote {out} and the window mask {mask_path}")
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
    c.add_argument("--date-a", default=None, help="the date map a describes, YYYY-MM-DD, or a period YYYY-MM-DD/YYYY-MM-DD "
                   "for a composite; with --date-b it says whether a difference can be real change on the ground")
    c.add_argument("--date-b", default=None, help="the date or period map b describes")
    c.add_argument("--labels-date", default=None, help="the date or period the --labels raster describes; required with "
                   "--labels when the two maps describe different times")
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
    e.add_argument("--per-class", action="store_true",
                   help="also user's accuracy, producer's accuracy and error-adjusted share per class; needs a "
                        "`reference_class` column in the CSV and the map's scores (from the sidecar, or --scores)")
    e.add_argument("--scores", default=None, help="the raster `sample` was run on, if it has moved")
    e.add_argument("--nodata", type=float, default=None)
    e.set_defaults(func=cmd_estimate)
    z = sub.add_parser("certify", help="which share of the map, from the most confident window down, is wrong at most "
                                        "alpha of the time, with a guarantee (needs a random sample)")
    z.add_argument("sample", help="the CSV `sample --design random` wrote, with its `wrong` column filled in")
    z.add_argument("--alpha", type=float, required=True, help="the error rate the certified zone may not exceed, e.g. 0.05")
    z.add_argument("--delta", type=float, default=est.ZONE_DELTA,
                   help=f"the probability the statement is allowed to be wrong (default {est.ZONE_DELTA})")
    z.add_argument("--rule", choices=("prefix", "bonferroni"), default="prefix",
                   help="prefix (default) assumes the zone's error rate does not fall as the zone grows; bonferroni assumes nothing")
    z.add_argument("--scores", default=None, help="the raster `sample` was run on, if it has moved")
    z.add_argument("--nodata", type=float, default=None)
    z.add_argument("--out", default=None, help="JSON to write (default: <sample>_zone.json; the window mask goes beside it as .npy)")
    z.set_defaults(func=cmd_certify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

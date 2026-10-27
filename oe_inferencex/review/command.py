"""The commands assess and boundary: one map ranked for review, and its class-edge band written as a condition layer."""
import csv
import json
import os

import numpy as np

from oe_inferencex.review.assess import assess_classmap, assess_prediction, pool_condition, summary
from oe_inferencex.review.explain import explain_review_set
from oe_inferencex.common.condition import _condition_args, _read_condition
from oe_inferencex.common.maps import PROB_TOLERANCE, _check_scores, decisions
from oe_inferencex.common.output import (_check_out_dir, _check_out_file, _check_patch, _nodata_arg, _pct,
                                         _refuse_overwriting_inputs, _stamp, _where)
from oe_inferencex.common.product import PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product
from oe_inferencex.common.raster import _same_grid, read_raster, window_coords, write_raster


def _budget_tag(b):
    """A budget as it appears in a review set's file name: whole percents zero-padded, others with a p for the point."""
    pct = b * 100
    return f"{int(round(pct)):02d}" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}".replace(".", "p")


def cmd_assess(args):
    _check_patch(args.patch, "assess")
    _check_out_dir(args.out, "assess")
    conf_path, conf_range = _product_args(args, "assess")
    prod_notes = []
    if conf_path:
        hard, conf_px, valid, geo, n_classes, prod_notes, _ = _read_product(args.scores, conf_path, args.nodata,
                                                                             conf_range, "assess")
        shape = hard.shape
    else:
        scores, valid, geo = read_raster(args.scores, args.nodata)
        _check_scores(scores, valid, args.logits, args.scores)
        shape = scores.shape[-2:]
    reference = None
    grid = geo              # the grid the inputs are checked against: the first one that carries one
    if args.reference:
        ref, rvalid, geo_r = read_raster(args.reference, None)
        if ref.shape != shape:
            raise SystemExit(f"{args.reference} has shape {ref.shape}; the map is {shape}")
        _same_grid(geo, geo_r, args.reference)                  # compare --labels had this check; assess did not
        reference = np.where(rvalid, np.rint(ref).astype(int), -1)
        # a .npy map has no grid, and the condition layer used to be checked against that alone, so a layer 200 km
        # from the reference passed (2026-10-06)
        grid = geo if geo is not None else geo_r
    cond_path, cond_names = _condition_args(args, "assess")
    ext = ".npy" if geo is None else ".tif"                     # write_raster's form of the window-grid rasters
    _refuse_overwriting_inputs(
        "assess", [os.path.join(args.out, n) for n in
                   ["assessment.json", "explanation.json", "suspicion" + ext, "boundary" + ext]
                   + (["condition" + ext] if cond_path else [])
                   + [f"review_set_{_budget_tag(b)}pct{s}.csv" for b in args.budgets
                      for s in ([""] + (["_by_condition"] if cond_path else []))]],
        [args.scores, args.reference, cond_path, conf_path])
    layer = None if cond_path is None else _read_condition(cond_path, shape, grid, "assess")
    try:
        if conf_path:
            out = assess_classmap(hard, conf_px, n_classes, patch=args.patch, nodata_mask=~valid, reference=reference,
                                  budgets=tuple(args.budgets), signal=PRODUCT_SIGNAL, order=args.order, condition=layer,
                                  condition_names=cond_names)
        else:
            out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid,
                                    reference=reference, budgets=tuple(args.budgets), order=args.order, condition=layer,
                                    condition_names=cond_names)
    except ValueError as exc:                                     # a named refusal, not a traceback
        raise SystemExit(f"assess: {exc}") from None
    os.makedirs(args.out, exist_ok=True)
    s = summary(out)
    s["inputs"] = {"scores": _where(args.scores), "logits": args.logits, "reference": _where(args.reference) if args.reference else None}
    if conf_path:   # "scores" keeps naming the first argument, here the class map
        s["inputs"].update({"classes": _where(args.scores), "confidence": _where(conf_path),
                            "confidence_range": None if conf_range is None else [float(v) for v in conf_range]})
        s["notes"] = list(s.get("notes", [])) + prod_notes
    arr = out["arrays"]
    conf, bnd = arr["confidence"], arr["boundary"]
    per_cond = name_at = None
    if layer is not None:
        # recorded only with a layer: without one, assessment.json differs from 1.3.1's by `scope` and the corrected
        # warning texts listed in tests/golden/condition_1_3_1/changes.py
        s["inputs"].update({"condition": _where(cond_path),
                            "condition_names": {str(k): v for k, v in cond_names.items()} if cond_names else None})
        s["conditions"]["source"] = _where(cond_path)
        per_cond = out["conditions"]["per_condition"]
        name_of = {e["value"]: name for name, e in per_cond.items()}          # None is the unrecorded windows' value
        name_at = lambda r, c: name_of[int(arr["condition"][r, c]) if arr["condition"][r, c] >= 0 else None]
    written = {}
    for b, rs in out["review_sets"].items():
        rc = np.asarray(rs["windows_rowcol"], dtype=int).reshape(-1, 2)
        pr, pc, x, y = window_coords(rc[:, 0], rc[:, 1], geo, args.patch)
        # Two budgets that differ must not write one file. 0.001 and 0.004 both rounded to "00pct" and the second
        # silently destroyed the first, while the JSON went on naming two files that were one. Whole percents keep
        # their old zero-padded name; only the sub-percent budgets that used to collide get a decimal form.
        tag = _budget_tag(b)
        path = os.path.join(args.out, f"review_set_{tag}pct.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "confidence", "boundary"]
                       + ([] if layer is None else ["condition"]))
            for i, (r, c) in enumerate(rc):
                w.writerow([i + 1, int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), float(conf[r, c]), float(bnd[r, c])]
                           + ([] if layer is None else [name_at(r, c)]))
        written[f"review_set_{b}"] = path
        if layer is not None:
            # each condition's own review set: the whole map's order kept to that condition, ranked from 1 within it
            path = os.path.join(args.out, f"review_set_{tag}pct_by_condition.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["condition", "rank_in_condition", "window_row", "window_col", "pixel_row", "pixel_col", "x",
                            "y", "confidence", "boundary"])
                for name, e in per_cond.items():
                    rcc = np.asarray(e["review_sets"][b]["windows_rowcol"], dtype=int).reshape(-1, 2)
                    pr_c, pc_c, x_c, y_c = window_coords(rcc[:, 0], rcc[:, 1], geo, args.patch)
                    for i, (r, c) in enumerate(rcc):
                        w.writerow([name, i + 1, int(r), int(c), int(pr_c[i]), int(pc_c[i]),
                                    None if x_c is None else float(x_c[i]), None if y_c is None else float(y_c[i]),
                                    float(conf[r, c]), float(bnd[r, c])])
            written[f"review_set_{b}_by_condition"] = path
    # SUSPICION, not confidence. Until 2026-09-21 this wrote `conf`, so ranking the file descending, which is what
    # its name invites, returned the exact inverse of the review set: on a 256-window scene the top 13 of the
    # raster overlapped the 13-window 5% review set in 0 of 13. A reviewer opening it with a hot-is-bad ramp
    # inspected the windows the model was most confident about.
    written["suspicion"] = write_raster(os.path.join(args.out, "suspicion.tif"), np.where(arr["valid"], -conf, np.nan).astype(np.float32), geo, args.patch, nodata=None)
    written["boundary"] = write_raster(os.path.join(args.out, "boundary.tif"), np.where(arr["valid"], bnd, np.nan).astype(np.float32), geo, args.patch, nodata=None)
    if layer is not None:                                   # each window's condition value, -1 where none is recorded
        written["condition"] = write_raster(os.path.join(args.out, "condition.tif"), arr["condition"].astype(np.int32),
                                            geo, args.patch, nodata=-1)
    exp = explain_review_set(out)
    with open(os.path.join(args.out, "explanation.json"), "w") as f:
        json.dump(exp, f, indent=1)
    written["explanation"] = os.path.join(args.out, "explanation.json")
    s["files"] = written
    with open(os.path.join(args.out, "assessment.json"), "w") as f:
        json.dump(_stamp(s, "assess"), f, indent=1)
    ref_block = s.get("against_reference") or {}
    # A budget is printed as the percent it is: rounded to a whole percent, 0.1% and 0.4% both printed as "0%" and
    # 2.5% as "2%", while the files and the JSON kept them apart (2026-10-06).
    if "error_capture_at_budget" in ref_block:
        # With a partial reference the capture is measured on the windows it grades, in review order, not on the review
        # sets: "5% -> 0.31" was printed beside a 5% review set that holds 0.46 of the disagreements; and with no
        # disagreement the share captured is undefined, where 0.00 read as "the review set catches none of the errors"
        # (the bug hunt of 2026-10-25; the MCP reply said both since 2026-10-06).
        n_ref, n_all = ref_block["n_windows_scored"], s["n_windows"]
        partial = n_ref < n_all
        ref_text = "; against the reference" + (f", which grades {n_ref} of the {n_all} windows" if partial else "") + ": "
        if not ref_block["error_rate"]:
            ref_text += "no window disagrees with it, so error capture is undefined"
        else:
            ref_text += "error capture " + ", ".join(
                f"{100 * float(b):g}% -> {v['errors_captured_fraction']:.2f}"
                for b, v in ref_block["error_capture_at_budget"].items())
            if partial:
                ref_text += " (measured on those windows in review order, not on the review sets)"
    elif args.reference:
        # sparse point labels, an all-no-data or an evenly split reference: no window could be graded, which used to
        # crash here after the files were written (review of 2026-09-23)
        ref_text = "; against the reference: no window has both a prediction and a majority reference label, so nothing was graded"
    else:
        ref_text = ""
    cond_text = ""
    if layer is not None:
        blk = out["conditions"]
        K = blk["n_conditions"]
        b5 = min(out["review_sets"], key=lambda b: abs(b - 0.05))        # the 5% review set, or the budget nearest it
        cond_text = (f"\n{K} input condition{'s' if K != 1 else ''}: " + "; ".join(
            f"{name} {100 * e['share_of_map']:.1f}% of windows, {100 * e['share_of_review_set'][b5]:.0f}% of the "
            f"{100 * b5:.3g}% review set" for name, e in per_cond.items())
            + (f"\nnote: {out['scope']}" if "scope" in out else "")
            + "".join(f"\nnote: {n}" for n in blk.get("notes", [])))
    # A review set cut inside a run of equal scores was reported in assessment.json only, so a kNN vote share or a
    # saturated probability could pick a whole set by raster position with nothing on the terminal (2026-10-22).
    tied_sets = [(b, rs) for b, rs in out["review_sets"].items() if "tied_at_cutoff" in rs]
    tie_text = ("\nnote: the scores tie at the review-set cut-off, so " + ", ".join(
        f"{rs['tied_at_cutoff']['inside']} of the {rs['n_windows']} windows at {100 * float(b):g}%"
        for b, rs in tied_sets) + " were chosen among equal scores by raster position, not evidence; scores with few "
        "distinct values (a kNN vote share, a saturated or quantized probability) cannot order them") if tied_sets else ""
    # assessment.json holds every warning; the terminal shows the ones that say the input may not be what --logits
    # claims (1.14.1), since the review set is then ordered by size alone
    logit_text = "".join(f"\nwarning: {w}" for w in s.get("warnings", []) if w.startswith("logit input:"))
    print(f"{s['n_windows']} windows of {args.patch} px; review sets " + ", ".join(f"{100 * float(b):g}%: {rs['n_windows']}" for b, rs in out["review_sets"].items()) +
          f"; boundary windows {100 * s['boundary_window_fraction']:.1f}%" + ref_text + cond_text + tie_text +
          logit_text + "".join(f"\nnote: {n}" for n in prod_notes) +
          f"\nwrote {args.out}/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary")
    return 0


# ----------------------------------------------------------------------------- boundary
def cmd_boundary(args):
    """Write the map's edge band as a condition layer, so that sample and estimate give the error rate near class
    edges and away from them, each with the condition design's exact interval."""
    from oe_inferencex.review.assess import boundary_band
    if (args.width_m is None) == (args.width_px is None):
        raise SystemExit("boundary: give the band's width once, as --width-m (metres, from a GeoTIFF's pixel size) or "
                         "--width-px (pixels)")
    _check_patch(args.patch, "boundary")
    hard, valid, geo, _, conf = decisions(args.scores, args.nodata, args.threshold)
    if geo is not None and args.out.lower().endswith(".npy"):
        # write_raster writes a GeoTIFF on a GeoTIFF map's grid whatever the name: a band.npy holding TIFF bytes, and
        # the sample command printed beside it refused the file (the bug hunt of 2026-10-25).
        raise SystemExit(f"boundary: {args.scores} is a GeoTIFF, so the band is written on its grid as a GeoTIFF; "
                         f"name --out with .tif, not {args.out}")
    target = args.out if geo is not None else os.path.splitext(args.out)[0] + ".npy"     # write_raster's own rule
    _refuse_overwriting_inputs("boundary", [target], [args.scores])
    _check_out_file(target, "boundary", "band.tif" if geo is not None else "band.npy")
    if args.width_m is not None:
        if geo is None:
            raise SystemExit("boundary: --width-m needs a GeoTIFF, whose pixel size converts metres to pixels; a .npy "
                             "has none, so give --width-px")
        # The pixel size was read as metres whatever the CRS: 10 US-survey-foot pixels made a 20 m band 6 m wide, and a
        # GeoTIFF with no CRS (an identity transform) a band of 20 pixels "of 1 x 1 m". The width is converted as
        # scale's window sizes are (the bug hunt of 2026-10-25).
        crs, t = geo["crs"], geo["transform"]
        if crs is None:
            raise SystemExit(f"boundary: {args.scores} has no CRS, so its pixel size is in no known unit; give --width-px")
        if crs.is_geographic:
            raise SystemExit("boundary: the map's CRS is in degrees, so its pixel size is not in metres; give --width-px")
        if t.b or t.d:
            raise SystemExit("boundary: the map's grid is rotated or sheared, so a pixel has no single width along its "
                             "rows and columns; give --width-px")
        try:
            factor = float(crs.linear_units_factor[1])
        except Exception:
            raise SystemExit("boundary: the map's CRS names no linear unit, so its pixel size cannot be read in metres; "
                             "give --width-px") from None
        if args.width_m <= 0:
            raise SystemExit(f"boundary: --width-m must be positive, got {args.width_m:g}")
        px_w, px_h = abs(t.a) * factor, abs(t.e) * factor
        wc, wr = max(1, round(args.width_m / px_w)), max(1, round(args.width_m / px_h))
        width_text = f"{args.width_m:g} m ({wr} rows, {wc} columns of {px_h:g} x {px_w:g} m pixels)"
    else:
        if args.width_px < 1:
            raise SystemExit(f"boundary: --width-px must be at least 1, got {args.width_px}")
        wr = wc = args.width_px
        width_text = f"{wr} pixel{'s' if wr != 1 else ''}"
    band = boundary_band(hard, valid, wr, wc)
    if not valid.any():
        raise SystemExit(f"boundary: {args.scores} has no valid pixel")
    share = float((band[valid] == 1).mean())
    try:
        pooled = pool_condition(band, args.patch, valid)
    except ValueError as exc:                            # a window larger than the map, or none half valid
        raise SystemExit(f"boundary: {exc}") from None
    n_win = int((pooled["grid"] >= 0).sum())
    win_share = float((pooled["grid"] == 1).sum() / n_win) if n_win else float("nan")
    path = write_raster(args.out, band.astype(np.int32), geo, 1, -1)
    # The map is read again by sample as boundary read it: without the --nodata given here, a .npy's fill became the
    # band's "unrecorded" windows and took half the labels (the bug hunt of 2026-10-25). sample ranks windows by the
    # map's scores: a one-band map cut at 0 is logits, a class map needs its confidence band, and a continuous map cut
    # elsewhere has no scores sample reads (the review of 1.17.2: the command printed was refused).
    import shlex
    flags, need = "", ""
    if conf is None:
        flags, need = " --confidence BAND", ("; BAND is the map's confidence band, which sample needs beside a class "
                                              "map to rank its windows")
    else:
        a, ok, _ = read_raster(args.scores, args.nodata)
        v = a[..., ok] if a.ndim == 3 else a[ok]
        unit = v.size == 0 or (float(v.min()) >= -PROB_TOLERANCE and float(v.max()) <= 1 + PROB_TOLERANCE)
        if a.ndim == 2 and a.dtype.kind in "biu" and (v.size == 0 or np.isin(v, (0, 1)).all()):
            # a 0/1 integer map is a class map whatever --threshold cut it at: sample reads it with its confidence band
            # (it lost that advice when cut below 1; the third review of 1.17.2); a percent-cover map cut at 50 is not
            # one (the fourth review)
            flags, need = " --confidence BAND", ("; BAND is the map's confidence band, which sample needs beside a "
                                                  "class map to rank its windows")
        elif a.ndim == 3:
            # a (C, H, W) map outside [0, 1] is logits to sample, which refused it as printed (the second review)
            flags = "" if unit else " --logits"
        elif args.threshold == 0:
            flags = " --logits"
        elif args.threshold is not None and not unit:
            need = (f"; sample reads a map's scores, and a continuous map cut at {args.threshold:g} has none: give "
                    "sample the scores this map was made from, or a confidence band with --confidence")
        elif args.threshold is not None and args.threshold != 0.5:
            need = (f"; sample reads a one-band probability map at 0.5, where boundary cut it at {args.threshold:g}, so "
                    "its windows' classes can differ from the band's")
    flags += f" {_nodata_arg(args.nodata)}" if args.nodata is not None else ""
    budget = min(300, n_win)              # a map of fewer valid windows refused 300 (the second review of 1.17.2)
    print(f"{100 * share:.1f}% of valid pixels lie within {width_text} of another class; windows of {args.patch} px "
          f"held mostly by the band: {_pct(None if np.isnan(win_share) else win_share, 1)}\nwrote {path}\n"
          f"next: oe-inferencex sample {shlex.quote(args.scores)}{flags} --condition {shlex.quote(path)} "
          f"--condition-names 0=interior 1=boundary --patch {args.patch} --budget {budget} --out to_label.csv{need}")
    return 0


def add_assess_parser(sub):
    a = sub.add_parser("assess", help="rank the windows of one prediction map for review and explain them")
    a.add_argument("scores", help="raster or .npy: (H, W) binary probability or logit map, or (C, H, W) per-class scores")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--logits", action="store_true",
                   help="the scores are logits (tie-free confidence); default probabilities. A multi-class logit map is "
                        "ranked by its logit margin; for a top-probability reading, pass the class probabilities "
                        "instead (in Python, form='top1' gives one from the logits). A regression output (a height, "
                        "a depth, an index) is not a logit: it has no margin, so no review set is read from it")
    a.add_argument("--patch", type=int, default=4, help="window size in pixels (default 4)")
    a.add_argument("--nodata", type=float, default=None, help="no-data value (default: the raster's own, plus NaN)")
    a.add_argument("--reference", default=None, help="optional integer class raster treated as truth (the caveat applies)")
    a.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10], help="review budgets as fractions of windows")
    a.add_argument("--order", choices=("confidence", "boundary_first"), default="confidence", help="review order")
    a.add_argument("--condition", default=None,
                   help="optional integer raster on the map's grid: each pixel's input condition (a cloud flag, the "
                        "modalities present, a sensor id); negative, NaN or no-data where none is recorded. Each "
                        "condition is then described and ranked on its own")
    a.add_argument("--condition-names", nargs="+", default=None, metavar="VALUE=NAME",
                   help="names for the condition values, e.g. 0=clear 1=cloudy (default: the value itself)")
    _product_parser_args(a)
    a.set_defaults(func=cmd_assess)


def add_boundary_parser(sub):
    bd = sub.add_parser("boundary", help="write the map's class-edge band as a condition layer, so sample and estimate "
                                         "give the error rate near edges and away from them")
    bd.add_argument("scores", help="the map's scores (GeoTIFF or .npy), as assess reads them, or a hard class map")
    bd.add_argument("--out", required=True, help="the band layer to write (GeoTIFF on the map's grid, or .npy)")
    bd.add_argument("--width-m", type=float, default=None,
                    help="the band's half-width in metres, converted with the GeoTIFF's pixel size; a common choice is "
                         "the coarser of the map's and the reference's resolution")
    bd.add_argument("--width-px", type=int, default=None, help="the band's half-width in pixels")
    bd.add_argument("--patch", type=int, default=4, help="window size in pixels, as sample will use it (default 4)")
    bd.add_argument("--nodata", type=float, default=None, help="no-data value (default: the raster's own, plus NaN)")
    bd.add_argument("--threshold", type=float, default=None,
                    help="cut a one-band continuous map here (binary logits: 0); a probability map is cut at 0.5")
    bd.set_defaults(func=cmd_boundary)

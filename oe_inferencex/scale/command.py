"""The command scale: one map at several window sizes, each class's survival and, with a reference, each error rate."""
import json

import numpy as np

from oe_inferencex.review.assess import summary
from oe_inferencex.common.maps import _check_scores
from oe_inferencex.common.output import _check_out_file, _pct, _refuse_overwriting_inputs, _stamp, _where
from oe_inferencex.common.product import _product_args, _read_product
from oe_inferencex.common.raster import _same_grid, read_raster


# ----------------------------------------------------------------------------- scale
def _window_size_text(p, geo):
    """A window size in pixels and, on a GeoTIFF in a projected CRS with a north-up grid, in metres (the CRS's linear
    unit converted: a CRS in US survey feet printed feet as metres in the first draft, the review of 1.15.0)."""
    if geo is None or geo.get("crs") is None or geo["crs"].is_geographic:
        return f"{p} px"
    t = geo["transform"]
    if t.b or t.d:                                       # a rotated or sheared grid has no single pixel width
        return f"{p} px"
    try:
        factor = float(geo["crs"].linear_units_factor[1])
    except Exception:                                    # an unknown unit: pixels only
        return f"{p} px"
    px_w, px_h = abs(t.a) * factor, abs(t.e) * factor
    side = f"{p * px_w:g} m" if abs(px_w - px_h) < 1e-9 else f"{p * px_h:g} x {p * px_w:g} m"
    return f"{p} px ({side})"


def cmd_scale(args):
    """The same map at several window sizes: how much of each class survives the window's majority, how many windows
    border a window of another class, how confident they are and, with a reference, each size's error rate
    (oe_inferencex.scale)."""
    from oe_inferencex.scale import scale_profile
    conf_path, conf_range = _product_args(args, "scale")
    prod_notes = []
    if conf_path:
        hard, conf_px, valid, geo, n_classes, prod_notes, _ = _read_product(args.scores, conf_path, args.nodata,
                                                                             conf_range, "scale")
        shape = hard.shape
    else:
        scores, valid, geo = read_raster(args.scores, args.nodata)
        _check_scores(scores, valid, args.logits, args.scores)
        shape = scores.shape[-2:]
    reference = None
    if args.reference:
        ref, rvalid, geo_r = read_raster(args.reference, None)
        if ref.shape != shape:
            raise SystemExit(f"{args.reference} has shape {ref.shape}; the map is {shape}")
        _same_grid(geo, geo_r, args.reference)
        reference = np.where(rvalid, np.rint(np.where(rvalid, ref, 0)), -1).astype(int)
    if args.out:
        _check_out_file(args.out, "scale")
        _refuse_overwriting_inputs("scale", [args.out], [args.scores, args.reference, conf_path])
    try:
        if conf_path:
            res = scale_profile(hard=hard, confidence=conf_px, n_classes=n_classes, patches=args.patches,
                                nodata_mask=~valid, reference=reference)
        else:
            res = scale_profile(scores=scores, is_logit=args.logits, patches=args.patches, nodata_mask=~valid,
                                reference=reference)
    except ValueError as exc:
        raise SystemExit(f"scale: {exc}") from None
    res["notes"] += prod_notes
    res["window_size"] = {p: _window_size_text(p, geo) for p in res["patches"]}
    res["inputs"] = {"scores": _where(args.scores), "logits": args.logits,
                     "confidence": _where(conf_path) if conf_path else None,
                     "confidence_range": None if conf_range is None else [float(v) for v in conf_range],
                     "reference": _where(args.reference) if args.reference else None}
    graded = args.reference is not None
    nan_none = lambda x: None if x is None or x != x else x          # an undefined rate prints as undefined
    lines = ["window size: windows, windows bordering another class, median confidence"
             + (", error rate against the reference, errors in a 5% review" if graded else "")]
    for p in res["patches"]:
        r = res["per_patch"][p]
        if not r["n_windows"]:
            lines.append(f"  {res['window_size'][p]}: 0 (no window at least half valid)")
            continue
        line = (f"  {res['window_size'][p]}: {r['n_windows']}, {_pct(r['edge_window_share'], 1)}, "
                f"{r['confidence_quantiles'].get(0.5, float('nan')):.3g}")
        if r.get("pixels_outside_window_grid"):
            line += f" ({r['pixels_outside_window_grid']} px outside the window grid)"
        ref = r.get("against_reference")
        if ref is not None:
            line += (f", {_pct(nan_none(ref['error_rate']), 1)}, "
                     + ("undefined (no error)" if ref["error_rate"] == 0 else
                        _pct(nan_none(ref["errors_in_5pct_review"]), 0))
                     + ("" if ref["n_windows_scored"] else " (no window has a prediction and a majority reference "
                                                           "label)"))
        lines.append(line)

    def accuracy_line(c):
        pa = []
        for p in res["patches"]:
            row = (res["per_patch"][p].get("against_reference") or {}).get("per_class", {}).get(c)
            v = None if row is None else row["producer_accuracy"]
            pa.append(f"{p} px " + ("no window" if v is None else f"{v:.2f}"))
        return f"producer's accuracy against the reference: {', '.join(pa)}"

    lines.append("class survival (the share of the pixels the map calls the class that sit in a window called it):")
    for c, e in res["per_class"].items():
        surv = ", ".join(f"{p} px {v:.2f}" for p, v in e["survival"].items())
        q = e["first_below_half"]
        tail = (f"; first below half at {res['window_size'][q]}, where {_pct(e['in_windows_of_another_class'][q], 0)} "
                f"of its pixels sit in windows called another class and {_pct(e['in_no_valid_window'][q], 0)} in no "
                f"valid window") if q is not None else ""
        lines.append(f"  class {c} ({e['pixels']} px): {surv}{tail}")
        if graded:
            lines.append(f"    {accuracy_line(c)}")
    if graded:                       # classes the reference holds and the map never draws: their errors are all here
        only_ref = sorted({c for p in res["patches"]
                           for c in (res["per_patch"][p].get("against_reference") or {}).get("per_class", {})}
                          - set(res["per_class"]))
        for c in only_ref:
            lines.append(f"  class {c} (in the reference, not drawn by the map): {accuracy_line(c)}")
    # every warning is in the JSON; the terminal shows the ones that say the input may not be what --logits claims, as
    # assess does (1.14.1)
    loud = [w["warning"] for w in res["warnings"] if w["warning"].startswith("logit input:")]
    print("\n".join(lines) + "".join(f"\nwarning: {w}" for w in loud) + "".join(f"\nnote: {n}" for n in res["notes"])
          + (f"\nwrote {args.out}" if args.out else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(_stamp(summary(res), "scale"), f, indent=1)          # summary: NaN as null, string keys
    return 0


def add_scale_parser(sub):
    sc = sub.add_parser("scale", help="the same map at several window sizes: how much of each class survives the "
                                      "window's majority, how many windows border another class and, with a "
                                      "reference, each size's error rate")
    sc.add_argument("scores", help="the map's scores (GeoTIFF or .npy), as assess reads them, or a product's class map "
                                   "with --confidence")
    sc.add_argument("--patches", type=int, nargs="+", default=[1, 2, 4, 8, 16], metavar="P",
                    help="window sizes in pixels (default 1 2 4 8 16); sizes larger than the map are left out")
    sc.add_argument("--logits", action="store_true", help="the scores are logits, as assess reads them")
    sc.add_argument("--nodata", type=float, default=None, help="no-data value (default: the raster's own, plus NaN)")
    sc.add_argument("--confidence", default=None, metavar="BAND",
                    help="with a product's class map: its per-pixel confidence band on the map's grid")
    sc.add_argument("--confidence-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="with --confidence: the band values that are confidences; others are left out")
    sc.add_argument("--reference", default=None,
                    help="optional integer class raster on the map's grid: adds each size's error rate and per-class "
                         "accuracy, a census of the reference, not a sample")
    sc.add_argument("--out", default=None, help="JSON to write the profile to (optional)")
    sc.set_defaults(func=cmd_scale)

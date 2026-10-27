"""The command compare: two inferences of the same scene, how much and where they differ, without labels."""
import csv
import json
import os

import numpy as np

from oe_inferencex.compare import compare_inferences
from oe_inferencex.review.assess import _boundary_valid, summary
from oe_inferencex.review.windows import _two_map_windows
from oe_inferencex.common.output import _check_out_dir, _pct, _refuse_overwriting_inputs, _stamp, _where
from oe_inferencex.common.raster import _same_grid, pool_valid, read_raster, window_coords, write_raster


def _window_majority(codes, n_codes, patch):
    """Each window's most frequent code among its pixels with a code >= 0, the lowest code on a tie, -1 for a window
    with none: what _pooled_argmax(codes, n_codes, patch, empty=-1) gives, counted from the (window, code) pairs that
    occur. _pooled_argmax holds one window-sized count grid per code, so a tile-id raster of 1024 ids took 1.1 GB on
    1024 x 1024 and 4096 ids on 4096 x 4096 did not finish in 49 GB (2026-10-06); here the cost follows the pixels."""
    h, w = codes.shape[0] // patch * patch, codes.shape[1] // patch * patch
    hw, ww = h // patch, w // patch
    c = np.asarray(codes[:h, :w], dtype=np.int64)
    win = (np.arange(h)[:, None] // patch) * ww + (np.arange(w)[None, :] // patch)
    has = c >= 0
    keys, votes = np.unique(win[has] * np.int64(max(n_codes, 1)) + c[has], return_counts=True)
    wi, ci = np.divmod(keys, np.int64(max(n_codes, 1)))
    order = np.lexsort((ci, -votes, wi))                     # by window, then most votes, then the lowest code
    wi, ci = wi[order], ci[order]
    first = np.r_[True, wi[1:] != wi[:-1]] if wi.size else np.zeros(0, bool)
    out = np.full(hw * ww, -1, dtype=np.int64)
    out[wi[first]] = ci[first]
    return out.reshape(hw, ww)


def cmd_compare(args):
    if args.labels_date and not args.labels:
        raise SystemExit("compare: --labels-date names the date of a --labels raster, and none was given")
    _check_out_dir(args.out, "compare")
    t = _two_map_windows(args.a, args.b, args.nodata, args.threshold, args.patch, args.labels)
    notes, ha, hb, va, vb, geo, code_of = t.notes, t.ha, t.hb, t.va, t.vb, t.geo, t.code_of
    a_w, b_w, ok, labels, ok_graded = t.a_w, t.b_w, t.ok, t.labels, t.ok_graded
    _refuse_overwriting_inputs("compare", [os.path.join(args.out, n) for n in
                                           ("comparison.json", "differing_windows.csv",
                                            "disagreement" + (".npy" if geo is None else ".tif"))],
                               [args.a, args.b, args.labels, args.groups])
    groups = None
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
        gw = _window_majority(gc, len(uid), args.patch)
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
    s["inputs"] = {"a": _where(args.a), "b": _where(args.b), "labels": _where(args.labels) if args.labels else None, "patch_px": args.patch,
                   "date_a": args.date_a, "date_b": args.date_b, "labels_date": args.labels_date}
    if notes:
        s["notes"] = notes
    # NaN where nothing was compared: a 0 there used to read as "agree" in any GIS
    dis = np.where(ok, out["arrays"]["disagree"].astype(np.float32), np.nan).astype(np.float32)
    s["files"] = {"disagreement": write_raster(os.path.join(args.out, "disagreement.tif"), dis, geo, args.patch, nodata=None)}
    rows, cols = np.nonzero(out["arrays"]["disagree"])
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    path = os.path.join(args.out, "differing_windows.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "a", "b"])
        for i, (r, c) in enumerate(zip(rows, cols)):
            ca_, cb_ = (int(a_w[r, c]), int(b_w[r, c])) if code_of is None else (int(code_of[a_w[r, c]]), int(code_of[b_w[r, c]]))
            w.writerow([int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), ca_, cb_])
    s["files"]["differing_windows"] = path
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(_stamp(s, "compare"), f, indent=1)
    where = s["where"] or {}
    print(f"{s['n_disagree']} of {s['n_windows']} windows differ ({_pct(s['disagreement_rate'])})" +
          (f"; on a boundary of a {where['boundary_a']['enrichment']:.1f}x as often as the agreeing windows" if where.get("boundary_a", {}).get("enrichment") is not None else "") +
          (f"; with labels: a right on {_pct(s['graded']['which_side']['share_a_right'], 0)}, "
           f"b on {_pct(s['graded']['which_side']['share_b_right'], 0)} of them" if s.get("graded") else "") +
          (f"\n{s['dates']['reading']}" if s["dates"]["status"] in ("different_time", "overlapping_time", "partly_stated") else "") +
          f"\nwrote {args.out}/comparison.json, differing_windows.csv, disagreement")
    return 0


def add_compare_parser(sub):
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

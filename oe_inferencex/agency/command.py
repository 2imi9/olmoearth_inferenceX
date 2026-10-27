"""The command grade: one or two maps graded with an agency's stratified sample, each interval valid at any size."""
import json

from oe_inferencex.common.output import _check_out_file, _refuse_overwriting_inputs, _stamp, _where


# ----------------------------------------------------------------------------- grade
def _range_pct(e, nd):
    """An estimate as a percentage, or its range when values are unknown."""
    if e is None:
        return "undefined"
    if isinstance(e, list):
        return f"{100 * e[0]:.{nd}f} to {100 * e[1]:.{nd}f}%"
    return f"{100 * e:.{nd}f}%"


def _iv_pct(q, nd, scale=100.0):
    iv, nt = q["interval"], q.get("normal_theory")
    s = "[not computed]" if iv is None else f"[{scale * iv[0]:.{nd}f}, {scale * iv[1]:.{nd}f}]"
    return s + ("" if nt is None else f" (normal {scale * nt[0]:.{nd}f}, {scale * nt[1]:.{nd}f})")


def cmd_grade(args):
    """One or more maps graded with an agency's stratified reference sample: overall accuracy, the difference between
    two maps, each class's user's and producer's accuracy and share of the reference, each with an interval valid at
    every sample size and the normal-theory one beside it, nominal (oe_inferencex.agency and
    oe_inferencex.agency.stratified)."""
    from oe_inferencex import agency
    if not 0 < args.alpha < 1:
        raise SystemExit(f"grade: --alpha must be in (0, 1), got {args.alpha}")
    if args.out:
        _check_out_file(args.out, "grade")
        _refuse_overwriting_inputs("grade", [args.out], [args.points, args.strata])
    pure = {}
    for item in args.pure:
        s, sep, c = item.partition("=")
        s, c = agency._code(s, ()), agency._code(c, ())
        if not sep or s is None or c is None:
            raise SystemExit(f"grade: --pure takes STRATUM=CLASS, got {item!r}")
        if pure.get(s, c) != c:
            raise SystemExit(f"grade: --pure names stratum {s} twice, as {pure[s]} and {c}")
        pure[s] = c
    if args.unit_area is not None and args.sizes != "area":
        raise SystemExit("grade: --unit-area is for sizes given as areas (--sizes area)")
    if args.sampled_strata_only and not args.strata:
        raise SystemExit("grade: --sampled-strata-only is for stratum sizes given with --strata")
    try:
        pts = agency.read_points(args.points, args.stratum, args.reference, args.map, args.alternate, args.weight,
                                 args.nodata)
        if args.weight:
            design = agency.design_from_weights(pts["stratum"], pts["weight"], args.sizes, args.unit_area,
                                                pts["weight_half_unit"])
        else:
            sizes = agency.read_strata(args.strata, pts["stratum"])
            design = agency.design_from_sizes(pts["stratum"], sizes, args.sizes, args.unit_area,
                                              args.sampled_strata_only)
        res = agency.grade(design, pts["reference"], pts["maps"], pts["alternate"] if args.alternate else None, pure,
                           args.alpha)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"grade: {exc}") from None
    res["design"] = {"sizes_from": "weights" if args.weight else "strata file", "sizes": args.sizes,
                     "frame": ("units drawn without replacement" if args.sizes == "units" else
                               "points in an area" if args.unit_area is None else
                               f"units of {args.unit_area:g} drawn without replacement, read as points in an area"),
                     "unsampled_share": design.W_unsampled, "labels_per_stratum": {str(design.names[h]): int(design.n[h])
                                                                                   for h in range(design.H)}}
    res["inputs"] = {"points": _where(args.points), "strata": _where(args.strata) if args.strata else None,
                     "stratum": args.stratum, "reference": args.reference, "alternate": args.alternate,
                     "maps": args.map, "weight": args.weight, "nodata": args.nodata,
                     "pure": {s: c for s, c in pure.items()}, "sampled_strata_only": args.sampled_strata_only}
    conf = f"{100 * (1 - args.alpha):g}%"
    lines = [f"{res['points']} points in {res['strata']} strata ({res['design']['frame']}); {res['rule']}",
             f"overall accuracy: estimate [{conf} interval, valid at any sample size] (normal theory, nominal)"]
    for m, q in res["overall_accuracy"].items():
        lines.append(f"  {m}: {_range_pct(q['estimate'], 2)} {_iv_pct(q, 2)}")
    for k, q in res["differences"].items():
        e = q["estimate"]
        est = (f"{100 * e[0]:+.2f} to {100 * e[1]:+.2f}" if isinstance(e, list) else f"{100 * e:+.2f}")
        lines.append(f"  {k}: {est} points {_iv_pct(q, 2)}, {q['disagreeing_points']} points where they differ")
    for m, per in res["per_class"].items():
        lines.append(f"{m} per class: user's accuracy | producer's accuracy | share of the reference")
        for c, row in per.items():
            u, p, s = row["user"], row["producer"], row["reference_share"]
            star = "*" if u["method"] == "pure-stratum rule" else ""
            lines.append(f"  {c}: {_range_pct(u['estimate'], 1)} {_iv_pct(u, 1)}{star} | "
                         f"{_range_pct(p['estimate'], 1)} {_iv_pct(p, 1)} | {_range_pct(s['estimate'], 2)} {_iv_pct(s, 2)}")
        if any(r["user"]["method"] == "pure-stratum rule" for r in per.values()):
            lines.append("  * user's accuracy by the pure-stratum rule")
    print("\n".join(lines) + "".join(f"\nnote: {n}" for n in res["notes"]) + (f"\nwrote {args.out}" if args.out else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(_stamp(res, "grade"), f, indent=1)
    return 0


def add_grade_parser(sub):
    gr = sub.add_parser("grade", help="grade one or two maps with an agency's stratified reference sample (NLCD, "
                                      "GFC2020): overall, user's and producer's accuracy and class shares, each with "
                                      "an interval valid at any sample size and the normal-theory one beside it")
    gr.add_argument("points", help="CSV of the labelled points: each one's stratum, reference label and the class "
                                   "each map gives it")
    gr.add_argument("--stratum", required=True, metavar="COL", help="the column holding each point's stratum")
    gr.add_argument("--reference", required=True, metavar="COL", help="the column holding each point's reference label")
    gr.add_argument("--map", required=True, action="append", metavar="COL",
                    help="a column holding a map's class at each point; give it twice to compare two maps on the same "
                         "points. The first is the map the strata were drawn from")
    gr.add_argument("--alternate", nargs="+", default=None, metavar="COL",
                    help="columns of further acceptable reference labels per point: a map class equal to the "
                         "reference label or any of them counts as right (NLCD's rule; its published per-class table "
                         "also accepts the 2011 primary label)")
    sz = gr.add_mutually_exclusive_group(required=True)
    sz.add_argument("--weight", default=None, metavar="COL",
                    help="each point's weight, its stratum's size over the stratum's labels (N_h / n_h)")
    sz.add_argument("--strata", default=None, metavar="CSV",
                    help="each stratum's size: a CSV with a header row, the stratum in the first column, its size in "
                         "the second")
    gr.add_argument("--sizes", choices=("units", "area"), default="units",
                    help="units (default): sizes count the units (pixels) the labels were drawn from without "
                         "replacement; area: sizes are areas, the points drawn at random within them")
    gr.add_argument("--unit-area", type=float, default=None, metavar="A",
                    help="with --sizes area: the area of the unit the labels were drawn as (a pixel), in the sizes' "
                         "units, when drawn without replacement")
    gr.add_argument("--sampled-strata-only", action="store_true",
                    help="with --strata: leave strata that hold no label out of the population, as some agencies "
                         "report (default: bound their values)")
    gr.add_argument("--nodata", nargs="+", default=[], metavar="CODE",
                    help="codes that mean no value in the map or reference columns (empty and NA cells always do); "
                         "unknown values are bounded, never dropped")
    gr.add_argument("--pure", nargs="+", default=[], metavar="STRATUM=CLASS",
                    help="strata every unit of which the first map labels CLASS (an agency's no-change strata); that "
                         "class's user's accuracy then takes the narrower pure-stratum rule")
    gr.add_argument("--alpha", type=float, default=0.05, help="1 - the confidence of each interval (default 0.05)")
    gr.add_argument("--out", default=None, help="JSON to write (optional)")
    gr.set_defaults(func=cmd_grade)

"""The command estimate: the map's error rate, per-class accuracy or audit from a labelled sample, with intervals."""
import json
import math
import os

import numpy as np

from oe_inferencex import estimate as est
from oe_inferencex.estimate.command_text import _zone_words
from oe_inferencex.review.windows import _map_windows
from oe_inferencex.common.labels import (_also_accept, _also_accept_rows, _also_words, _check_sidecar, _csv_rows,
                                         _labelled_sample, _population_note, _reference_column, _sample_map_inputs,
                                         _sidecar_design, _unused_note)
from oe_inferencex.common.output import _check_out_file, _refuse_overwriting_inputs, _stamp
from oe_inferencex.common.product import MAX_CLASS_ID
from oe_inferencex.common.wording import class_wording


def _estimate_disagreement(args):
    """estimate on a sample --other CSV: which map is more accurate, and by how much."""
    # --confidence too: added with the product input and accepted here unread, so a wrong or missing band path
    # passed with exit 0 (2026-10-06)
    # --classes too, accepted unread, even negative (the bug hunt of 2026-10-25)
    if getattr(args, "valid", False):
        raise SystemExit("estimate: --valid reads a sample drawn for the error rate; a sample of the windows where two "
                         "maps differ has its own interval and sign test")
    if args.per_class or args.scores is not None or getattr(args, "confidence", None) is not None \
            or args.nodata is not None or args.reviewer_false_alarm or args.reviewer_miss \
            or getattr(args, "alpha", None) is not None or getattr(args, "also_accept", None) is not None \
            or args.classes is not None:
        raise SystemExit("estimate: a sample of the windows where two maps differ takes none of --per-class, --classes, "
                         "--scores, --confidence, --nodata, --reviewer-false-alarm, --reviewer-miss, --alpha and "
                         "--also-accept; it compares the two maps from the reference_class column alone")
    path = args.sample
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path, ("indices", "class_a", "class_b"))
    out = args.out or (path[:-4] + "_estimate.json" if path.endswith(".csv") else path + "_estimate.json")
    _refuse_overwriting_inputs("estimate", [out], [path, side_path, *_sample_map_inputs(path)])
    rows = _csv_rows(path, "estimate")
    need = {"index", "class_a", "class_b", "reference_class"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected the columns `sample --other` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])
        ca = np.array([int(float(r["class_a"])) for r in rows])
        cb = np.array([int(float(r["class_b"])) for r in rows])
    except (ValueError, OverflowError) as exc:
        raise SystemExit(f"{path}: the index, class_a and class_b columns are not the ones `sample --other` wrote: {exc}")
    if not np.array_equal(idx, np.asarray(side["indices"], int)):
        raise SystemExit("the CSV's rows do not match the design in its sidecar; label the file `sample` wrote, in order")
    if not (np.array_equal(ca, np.asarray(side["class_a"], int)) and np.array_equal(cb, np.asarray(side["class_b"], int))):
        raise SystemExit(f"{path}: class_a or class_b differs from the maps' classes the sample recorded; they are fixed "
                         "when the sample is drawn, so label the file `sample` wrote without changing them")
    ref, unjudged = _reference_column(rows)
    res = est.compare_from_disagreement(side, ca, cb, ref, unjudged=unjudged)
    res.update({"sample": os.path.abspath(path), "map_a": side.get("map_a"), "map_b": side.get("map_b"),
                "notes": list(side.get("notes", []))})
    _check_out_file(out, "estimate")
    with open(out, "w") as f:
        json.dump(_stamp(res, "estimate"), f, indent=1)
    d, v = res["difference"], res["verdict"]
    from oe_inferencex.decide import sign_test_head
    sign = sign_test_head(res)
    if sign:
        head = sign
    elif v == "a":
        head = (f"map a is more accurate than map b, by {100 * d['low']:.1f} to {100 * d['high']:.1f} points over the "
                "windows compared (95% interval)")
    elif v == "b":
        head = (f"map b is more accurate than map a, by {-100 * d['high']:.1f} to {-100 * d['low']:.1f} points over the "
                "windows compared (95% interval)")
    else:
        head = (f"the labels cannot tell which map is more accurate: the difference, a minus b, lies between "
                f"{100 * d['low']:+.1f} and {100 * d['high']:+.1f} points (95% interval)")
    print(f"{head}\nfrom {res['n_labelled']} labelled windows of the {res['n_disagree']} where the maps differ "
          f"({100 * res['disagree_share']:.1f}% of the {res['n_population']} windows compared): a right on "
          f"{res['n_a_right']}, b right on {res['n_b_right']}, neither on {res['n_neither']}"
          + (f", {res['n_unjudged']} could not be judged" if res["n_unjudged"] else "")
          + f"\nnote: {res['note']}" + (f"\nnote: {res['verdict_note']}" if sign else "")
          + "".join(f"\nnote: {n}" for n in res["notes"]) + f"\nwrote {out}")
    return 0


def _verdict_words(verdict, alpha):
    return {"above": f"at least {100 * alpha:g}%", "below": f"below {100 * alpha:g}%",
            "undetermined": "undetermined"}[verdict]


def _estimate_class_audit(args):
    """estimate on a sample --design audit CSV: each predicted class's confident-zone error rate with its exact interval;
    with --alpha a verdict per class, and with --also-accept a second reading of the same labels."""
    if getattr(args, "valid", False):
        raise SystemExit("estimate: --valid reads a sample drawn for the error rate; an audit sample's intervals are "
                         "exact per class already")
    if args.per_class or args.classes is not None:      # --classes was accepted unread (the bug hunt of 2026-10-25)
        raise SystemExit("estimate: an audit sample is read per class already, over the classes it recorded; drop "
                         "--per-class and --classes")
    if args.scores is not None or getattr(args, "confidence", None) is not None or args.nodata is not None:
        raise SystemExit("estimate: an audit sample takes none of --scores, --confidence and --nodata; it reads the "
                         "classes the sample recorded and the reference_class column")
    if args.reviewer_false_alarm or args.reviewer_miss:
        # the reason certify takes none (cmd_certify): a miss rate would have to hold inside each confident zone
        raise SystemExit("estimate: an audit sample takes no reviewer error rate: the confident zone is where a reviewer "
                         "can be fooled too, and a rate measured on the whole map does not hold there")
    path = args.sample
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path, ("indices", "map_class", "audited", "zone"))
    out = args.out or (path[:-4] + "_estimate.json" if path.endswith(".csv") else path + "_estimate.json")
    accept_path = getattr(args, "also_accept", None)
    _refuse_overwriting_inputs("estimate", [out], [path, side_path, accept_path, *_sample_map_inputs(path)])
    rows = _csv_rows(path, "estimate")
    need = {"index", "map_class", "reference_class"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected the columns `sample --design audit` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])
        mc = np.array([int(float(r["map_class"])) for r in rows])
    except (ValueError, OverflowError) as exc:
        raise SystemExit(f"{path}: the index and map_class columns are not the ones `sample --design audit` wrote: {exc}")
    if not np.array_equal(idx, np.asarray(side["indices"], int)):
        raise SystemExit("the CSV's rows do not match the design in its sidecar; label the file `sample` wrote, in order")
    if not np.array_equal(mc, np.asarray(side["map_class"], int)):
        k = int(np.argmax(mc != np.asarray(side["map_class"], int)))
        raise SystemExit(f"{path}: map_class differs from the class the sample recorded (row {k + 2}: {mc[k]} in the CSV, "
                         f"{side['map_class'][k]} recorded); it is the class each window was drawn for, fixed when the "
                         "sample is drawn, so label the file `sample` wrote without changing it")
    ref, unjudged = _reference_column(rows, "the map's class codes", negative=False)
    accept = _also_accept(accept_path) if accept_path is not None else None
    alpha = getattr(args, "alpha", None)
    try:
        res = est.class_audit_from_sample(side, ref, unjudged=unjudged, also_accept=accept, alpha=alpha)
    except ValueError as exc:
        raise SystemExit(f"estimate: {exc}") from None
    res["sample"] = os.path.abspath(path)
    if accept is not None:
        res["also_accept"]["source"] = os.path.abspath(accept_path)
    if _population_note(side):                       # a product's range left pixels out: the zone is of the rest
        res["population_note"] = _population_note(side)
    _check_out_file(out, "estimate")
    with open(out, "w") as f:
        json.dump(_stamp(res, "estimate"), f, indent=1, default=float)
    z = res["zone"]
    lines = [f"confident zone: the {z['n_zone']} windows at confidence {z['cut']:.6g} or above, "
             f"{100 * z['share_of_map']:.1f}% of the {res['n_population']} valid windows{_zone_words(z)}; "
             f"{res['labels_per_class']} labels per predicted class"]
    for c, row in res["per_class"].items():
        counts = f"{row['n_wrong']} wrong" + (f" and {row['n_unjudged']} ?" if row["n_unjudged"] else "")
        exact = "its exact range" if row["n_unjudged"] else "its exact rate"
        said = f"; {_verdict_words(row['verdict'], alpha)}" if "verdict" in row else ""
        if row["census"]:
            lines.append(f"class {c}: labelled in full, {counts} of {row['n_labelled']} ({_rate(row)}), {exact}{said}")
        else:
            lines.append(f"class {c}: {counts} of {row['n_labelled']} labelled ({_rate(row)}), 95% interval "
                         f"{100 * row['low']:.1f}% to {100 * row['high']:.1f}%, of {row['n_zone']} windows in its "
                         f"zone{said}")
        alt = row.get("also_accept")
        if alt and alt["accepted"]:                    # a class with nothing more accepted reads the same
            iv = exact if row["census"] else f"{100 * alt['low']:.1f}% to {100 * alt['high']:.1f}%"
            lines.append(f"  also accepting {', '.join(str(a) for a in alt['accepted'])}: {alt['n_wrong']} wrong "
                         f"({_rate(alt)}), {iv}" + (f"; {_verdict_words(alt['verdict'], alpha)}" if "verdict" in alt else ""))
        if row.get("note"):
            lines.append(f"  note: {row['note']}")
    for c, e in res["not_audited"].items():
        lines.append(f"not audited: class {c} (no window reaches the cut; {e['n_windows']} windows of the map)")
    lines.append(f"note: {res['note']}")
    if accept is not None:
        lines.append(f"note: {res['reading_note']}")
        if res["also_accept_unused"]:
            lines.append(f"note: --also-accept lists class(es) {', '.join(res['also_accept_unused'])}, which this sample "
                         "does not audit; they are not read")
    if alpha is None:
        lines.append("note: without --alpha no class gets a verdict, and none the note on errors an alternative reading "
                     "accepts; pass --alpha A with A chosen before any label is read")
    if "population_note" in res:
        lines.append(f"note: {res['population_note']}")
    print("\n".join(lines) + f"\nwrote {out}")
    return 0


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


def _rate(res):
    """A rate as printed: the estimate, or the range unjudged windows and reviewer error leave (`estimate_range`)."""
    if res.get("estimate") is None and res.get("estimate_range"):
        lo, hi = res["estimate_range"]
        return f"{100 * lo:.1f}% to {100 * hi:.1f}%"
    return f"{100 * res['estimate']:.1f}%"


def _valid_class_line(c, row):
    """A class's line of valid intervals (estimate --valid --per-class): each point (a range where a part of a stratum
    the map calls the class holds no label) with its interval, ends rounded outward; the share flagged on its own
    interval. Of the row's warnings it names those that bear on these intervals."""
    v = row["valid"]

    def f(q, d=0):
        if q is None:
            return "n/a"
        if q.get("low") is None:
            return "not computed"
        e = q.get("estimate")
        pt = (f"{100 * e:.{d}f}%" if e is not None else
              f"{100 * q['estimate_range'][0]:.{d}f}-{100 * q['estimate_range'][1]:.{d}f}%")
        lo = math.floor(100 * q["low"] * 10 ** d + 1e-9) / 10 ** d
        hi = math.ceil(100 * q["high"] * 10 ** d - 1e-9) / 10 ** d
        return f"{pt} ({lo:.{d}f}-{hi:.{d}f})"
    sh = v["reference_share"]
    drawn = f": drawn {sh['drawn']}" if sh and sh.get("drawn") not in (None, "within") else ""
    codes = [k for k in row.get("warning_codes", []) if k in ("few labels", "never predicted")]
    tag = f"  [warning: {', '.join(codes)}]" if codes else ""
    return (f"  class {c}, valid: user's accuracy {f(v['user_accuracy'])}, producer's {f(v['producer_accuracy'])}, "
            f"share of map {100 * row['map_share']:.1f}% -> reference {f(sh, 1)}{drawn}{tag}")


def _outside_note(res):
    """The closing note of `estimate` when the whole-map rate lies outside some conditions' intervals, worded by the
    direction observed: a condition whose interval lies above the whole-map rate is worse than the map as a whole,
    one whose interval lies below it better. Until 2026-09-29 the note said the whole-map rate "can hide a condition
    that is much worse" whatever the direction, for a better condition too, and for a condition labelled in full,
    whose interval is its exact rate."""
    per = res["per_condition"]
    lo_w, hi_w = res["estimate_range"] if res.get("estimate") is None else (res["estimate"], res["estimate"])

    def named(names):
        names = [n + (" (labelled in full, so its interval is its exact rate)"
                      if per[n]["n_labelled"] == per[n]["n_population"] else "") for n in names]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    said = [f"note: the whole-map rate is {_rate(res)}."]
    for names, side, how in (([n for n in res["outside_condition_intervals"] if per[n]["low"] > hi_w], "below", "worse"),
                             ([n for n in res["outside_condition_intervals"] if per[n]["high"] < lo_w], "above", "better")):
        if names:
            one = len(names) == 1
            said.append(f"It lies {side} the interval{'' if one else 's'} of {named(names)}, so "
                        f"{'that condition is' if one else 'those conditions are'} {how} than the map as a whole.")
    return " ".join(said + ["The whole-map rate weights each condition by its share of the map."])


def cmd_estimate(args):
    """The map's error rate with its interval, from a filled-in sample CSV and its sidecar design; with
    --per-class, the user's and producer's accuracy and error-adjusted share per class as well. A sample drawn with
    --other (two maps) gives which map is more accurate instead, and one drawn with --design audit each predicted
    class's confident-zone error rate."""
    design = _sidecar_design(args.sample)
    if design == "disagreement":
        return _estimate_disagreement(args)
    if design == "audit":
        return _estimate_class_audit(args)
    if getattr(args, "alpha", None) is not None:
        raise SystemExit("estimate: --alpha reads a per-class audit (sample --design audit); for the whole map against a "
                         "rate, ask decide error_rate_below=T of this estimate")
    accept_path = getattr(args, "also_accept", None)
    if not (args.per_class or accept_path is not None) and (args.scores is not None or getattr(args, "confidence", None)
                                                            is not None or args.nodata is not None):
        # the error rate reads the CSV and its sidecar alone; accepting the map's options and ignoring them let a
        # wrong --scores path pass unnoticed (release check of 24 September), and --confidence, added with the product
        # input, was accepted unread the same way, a missing band included (2026-10-06)
        raise SystemExit("estimate: --scores, --confidence and --nodata are read only with --per-class or --also-accept; "
                         "the error rate comes from the CSV and its sidecar alone")
    side_path = args.sample[:-4] + ".json" if args.sample.endswith(".csv") else args.sample + ".json"
    out = args.out or (args.sample[:-4] + "_estimate.json" if args.sample.endswith(".csv") else args.sample + "_estimate.json")
    _refuse_overwriting_inputs("estimate", [out], [args.sample, side_path, accept_path,
                                                   *_sample_map_inputs(args.sample, args.scores,
                                                                       getattr(args, "confidence", None))])
    side, rows, idx, sample, wrong, unjudged = _labelled_sample(args.sample, "estimate")
    fa, miss = args.reviewer_false_alarm or 0.0, args.reviewer_miss or 0.0
    for flag, v in (("--reviewer-false-alarm", fa), ("--reviewer-miss", miss)):
        if not 0 <= v < 1:
            raise SystemExit(f"estimate: {flag} must be at least 0 and below 1, got {v:g}")
    if accept_path is not None and (fa or miss):
        raise SystemExit("estimate: --also-accept takes no reviewer error rate: a stated rate bounds the reviewer's errors "
                         "for one definition of an error, and the file changes that definition; a reviewer's confusions "
                         "are likeliest on the very pairs it accepts")
    if getattr(args, "classes", None) is not None and not args.per_class:
        raise SystemExit("estimate: --classes sets the legend's size for the per-class table; it goes with --per-class")
    if getattr(args, "classes", None) is not None and args.classes < 1:
        raise SystemExit(f"estimate: --classes must be at least 1, got {args.classes}")
    if args.per_class and (unjudged.any() or fa or miss):
        raise SystemExit("estimate --per-class needs the class seen in every window and a reviewer taken as right: "
                         + (f"{int(unjudged.sum())} window(s) are ? (could not be judged)" if unjudged.any() else
                            "--reviewer-false-alarm and --reviewer-miss do not apply to the per-class table")
                         + ". Run estimate without --per-class for the error rate with its bounds")
    map_w = also = None
    if accept_path is not None:
        # the map's class at each window, recomputed from the map and checked against the CSV, as --per-class reads it
        map_w = _map_windows(side, args.scores, args.nodata, "estimate --per-class" if args.per_class else "estimate",
                             rows, idx, getattr(args, "confidence", None))
        wrong_alt, also, unused = _also_accept_rows("estimate", accept_path, rows, wrong, unjudged,
                                                    map_w[1].ravel()[idx])
    if side.get("design") == "sequential":
        # the labelled rows are the first windows of a random order: a simple random sample of their number
        sample = dict(sample, design="random", budget=int(idx.size))
    valid = bool(getattr(args, "valid", False))       # 1.20.0: the valid interval beside the one given without it
    try:
        res = est.estimate_error_rate(sample, wrong, unjudged=unjudged, reviewer_false_alarm=fa, reviewer_miss=miss,
                                      valid=valid)
        # the second reading: the same estimator on the same sample, with the windows the file accepts counted right
        alt = None if also is None else est.estimate_error_rate(sample, wrong_alt, unjudged=unjudged, valid=valid)
    except ValueError as exc:
        raise SystemExit(f"estimate: {exc}")
    if side.get("design") == "sequential":
        res["sequential_note"] = est.SEQUENTIAL_READ_NOTE.format(n=int(idx.size))
    per_class_text = ""
    if args.per_class:
        ref = _reference_classes(rows, args.sample)
        _, hard, valid_w, n_classes = map_w or _map_windows(side, args.scores, args.nodata, "estimate --per-class", rows,
                                                            idx, getattr(args, "confidence", None))
        if side.get("confidence") and ref.size and ref.max() < MAX_CLASS_ID + 1:
            # a product's ids are its own: a reviewer can see a class the map never predicts, which has no score band
            n_classes = max(n_classes, int(ref.max()) + 1)
        if (ref >= n_classes).any():
            k = int(np.argmax(ref >= n_classes))
            raise SystemExit(f"estimate --per-class: row {k + 2} gives reference_class {int(ref[k])}, but the map has "
                             f"{n_classes} classes (0 to {n_classes - 1}); a class id outside them added empty rows")
        map_class = np.where(valid_w, hard, -1).ravel()
        # row by row: the count alone matched when `wrong` marked rows 0-9 and the classes disagreed on rows 10-19
        mismatch = np.flatnonzero((map_class[idx] != ref).astype(int) != wrong)
        try:
            pc = est.estimate_per_class(sample, ref, map_class, n_classes=n_classes,
                                        legend_size=getattr(args, "classes", None), valid=valid)
        except ValueError as exc:
            raise SystemExit(f"estimate --per-class: {exc}" + ("; give --classes, the number of classes the legend holds"
                                                               if "legend" in str(exc) and getattr(args, "classes", None)
                                                               is None else ""))
        res["per_class"] = pc["per_class"]
        res["confusion_counts"] = pc["confusion_counts"]
        res["overall_accuracy"] = pc["overall_accuracy"]
        res["per_class_method"] = pc["method"]
        if "legend_size" in pc:                           # the classes the exact intervals hold together over
            res["per_class_legend_size"] = pc["legend_size"]
        elif getattr(args, "classes", None) is not None:
            res["per_class_note"] = ("--classes sets the level of a random sample's exact intervals; this design's "
                                     "per-class intervals do not use it")
        if "warning" in pc:
            res["per_class_warning"] = pc["warning"]
        if pc.get("overall_accuracy_post_stratified") is not None:
            res["overall_accuracy_post_stratified"] = pc["overall_accuracy_post_stratified"]
        if pc.get("overall_accuracy_normal_theory") is not None:
            # printed since 1.17.1 and promised in the JSON by its CHANGELOG, but never written (the bug hunt of
            # 2026-10-25)
            res["overall_accuracy_normal_theory"] = pc["overall_accuracy_normal_theory"]
        if valid:
            res["overall_accuracy_valid"] = pc["overall_accuracy_valid"]
            res["per_class_valid_note"] = pc["valid_note"]
        if mismatch.size:
            res["per_class_note"] = (f"on {mismatch.size} row(s), first row {int(mismatch[0]) + 2}, `wrong` disagrees with whether "
                                     "`reference_class` differs from the map's class; the error rate above uses `wrong`, the "
                                     "per-class table uses `reference_class`")
        lines = []
        exact = bool(pc["per_class"]) and all("reference_share_exact" in row for row in pc["per_class"].values())
        for c, row in pc["per_class"].items():
            if side.get("confidence") and row["map_share"] == 0 and not row.get("n_labelled_reference_class"):
                continue        # a product's id neither the map nor the labels use (LCMAP's classes start at 1)
            ua, pa, sh = row["user_accuracy"], row["producer_accuracy"], row["reference_share"]
            f = lambda v: "n/a" if v is None else f"{100 * v['estimate']:.0f}% ({100 * v['low']:.0f}-{100 * v['high']:.0f})"
            # the tag names the warning's reason; it used to say "few labels" for a near-census class of 500 labels
            tag = f"  [warning: {', '.join(row.get('warning_codes', ['see the JSON']))}]" if "warning" in row else ""
            if exact:
                # the printed line shows exact intervals only, so it names only the warning that bears on them (few
                # labels: wide); the others concern the normal-theory fields kept in the JSON
                codes = [k for k in row.get("warning_codes", []) if k not in ("few errors", "near census")]
                tag = f"  [warning: {', '.join(codes)}]" if codes else ""
                # a random sample: the share's and producer's accuracy's exact intervals (exp115), each with its own
                # point (the labels' share k/n, the found share a/k), so a point never sits outside its interval; the
                # error-adjusted points stay in the JSON. Ends rounded outward, the share's to a tenth of a point as the
                # map's share is printed; a share the map draws outside its interval is named
                pe, se = row["producer_accuracy_exact"], row["reference_share_exact"]
                fl = lambda x, d: math.floor(x * 10 ** d + 1e-9) / 10 ** d
                ce = lambda x, d: math.ceil(x * 10 ** d - 1e-9) / 10 ** d
                pt = "n/a" if pe["estimate"] is None else f"{100 * pe['estimate']:.0f}%"
                drawn = "" if row["drawn"] == "within" else f": drawn {row['drawn']}"
                normal = ("" if (side.get("condition") or {}).get("by") != "map class" else
                          f"; normal-theory: producer's {f(pa)}, area {f(sh)}")
                line = (f"  class {c}: user's accuracy {f(ua)}, producer's {pt} (exact {fl(100 * pe['low'], 0):.0f}-"
                        f"{ce(100 * pe['high'], 0):.0f}), share of map {100 * row['map_share']:.1f}% -> "
                        f"{'area' if (side.get('condition') or {}).get('by') == 'map class' else 'labels'} "
                        f"{100 * se['estimate']:.1f}% (exact {fl(100 * se['low'], 1):.1f}-{ce(100 * se['high'], 1):.1f})"
                        f"{drawn}" + normal + tag)
            else:
                line = (f"  class {c}: user's accuracy {f(ua)}, producer's {f(pa)}, share of map "
                        f"{100 * row['map_share']:.1f}% -> error-adjusted {f(sh)}" + tag)
            if valid:
                # the valid intervals first, then the line printed without --valid, unchanged, labelled so
                lines.append(_valid_class_line(c, row))
                lines.append("    without --valid: " + line[len(f"  class {c}: "):])
            else:
                lines.append(line)
        by_class = (side.get("condition") or {}).get("by") == "map class"
        if exact:
            # a class design with one class is a random sample, and its table the random design's (the review of 24
            # October 2026: it printed the class design's note beside the random design's numbers)
            class_note = by_class and len((side.get("condition") or {}).get("values") or []) > 1
            lines.append(f"note: {est.CLASS_DESIGN_EXACT_NOTE if class_note else est.PER_CLASS_EXACT_NOTE} (the "
                         f"legend holds {pc['legend_size']} classes)")
            if class_note and pc.get("overall_accuracy_normal_theory"):
                o = pc["overall_accuracy_normal_theory"]
                lines.append(f"note: {est.CLASS_NORMAL_NOTE} Overall accuracy {100 * o['estimate']:.1f}%, normal-theory "
                             f"interval {100 * o['low']:.1f}% to {100 * o['high']:.1f}%.")
            if side.get("design") == "sequential":
                lines.append("note: a sequential sample's per-class intervals are exact when the number of labelled "
                             "windows was fixed before labelling, as for its error rate")
        if res["design"] == "condition" and len(res.get("per_condition", {})) > 1 and not by_class:
            # the method string says so, and the printed table did not: the per-class intervals under this design
            # are not graded (review of 2026-09-29); with one condition the table is the random design's
            lines.append(f"note: {est.PER_CLASS_NOT_GRADED}")
        if valid:
            ov, o = res["overall_accuracy_valid"], res["overall_accuracy"]
            lines.append(("overall accuracy: the valid interval is not computed" if ov["low"] is None else
                          f"overall accuracy {_rate(ov)}, valid 95% interval {100 * ov['low']:.1f}% to "
                          f"{100 * ov['high']:.1f}%")
                         + f"; without --valid {100 * o['low']:.1f}% to {100 * o['high']:.1f}%")
            lines.append(f"note: {est.PER_CLASS_VALID_NOTE}")
        per_class_text = "\n" + "\n".join(lines) + (f"\nnote: {res['per_class_note']}" if "per_class_note" in res else "")
    res["sample"] = os.path.abspath(args.sample)
    if _population_note(side):                       # a product's range left pixels out: the rate is of the rest
        res["population_note"] = _population_note(side)
    also_text = ""
    if also is not None:
        # last but for written_by (_stamp), so that the result without these three keys is the one written without
        # the option
        also["n_wrong"] = int(np.asarray(wrong_alt)[~unjudged].sum())
        keep = ("n_labelled", "n_wrong", "n_unjudged", "estimate", "estimate_range", "low", "high", "half_width", "method")
        for k in ("estimate", "estimate_range", "low", "high", "half_width", "method", "warning", "per_condition",
                  "outside_condition_intervals"):
            if k in alt:
                also[k] = ({name: {q: row[q] for q in keep if q in row} for name, row in alt[k].items()}
                           if k == "per_condition" else alt[k])
        if "valid_interval" in alt:                     # --valid: the second reading's valid interval too
            also["valid_interval"] = alt["valid_interval"]
        res.update({"also_accept": also, "also_accept_unused": unused, "reading_note": est.ESTIMATE_READING_NOTE})
        n = also["n_accepted"]
        moved = (f"{n} labelled window{'s' if n != 1 else ''} wrong above {'are' if n != 1 else 'is'} right under this "
                 "reading (" + ", ".join(f"{c} seen as {r}: {k}" for c, by in also["errors_accepted"].items()
                                         for r, k in by.items()) + ")"
                 if n else "no labelled window wrong above is right under this reading")
        lines = [f"also accepting {_also_words(also)} ({os.path.basename(accept_path)}): error rate {_rate(also)}, 95% "
                 f"interval {100 * also['low']:.1f}% to {100 * also['high']:.1f}%; {moved}"]
        if "valid_interval" in also:
            va = also["valid_interval"]
            lines[0] += (f"; valid 95% interval {100 * va['low']:.1f}% to {100 * va['high']:.1f}% (the interval before "
                         "it is the one given without --valid)")
        if res.get("by_condition"):                     # one line per condition, as the strict lines
            for name, row in also["per_condition"].items():
                lines.append(f"  {name} {_rate(row)} ({100 * row['low']:.1f}% to {100 * row['high']:.1f}%)"
                             if row["n_labelled"] else f"  {name}: no labelled window fell in this condition")
        if "warning" in also:
            lines.append(f"  warning: {also['warning']}")
        lines.append(f"note: {res['reading_note']}")
        if unused:
            lines.append(_unused_note(unused))
        also_text = "\n" + "\n".join(lines)
    _check_out_file(out, "estimate")
    by_class = (side.get("condition") or {}).get("by") == "map class"
    if (side.get("condition") or {}).get("by") == "profile":           # the profile design: what certify adds
        res["profile_note"] = PROFILE_ESTIMATE_NOTE
    words = class_wording(side, args.sample, out, accept_path)
    with open(out, "w") as f:
        json.dump(_stamp(words(res), "estimate"), f, indent=1, default=float)
    cond_text = ""
    if res.get("by_condition"):
        # each condition's own exact interval, one line each after the whole map's
        lines = []
        for name, row in res["per_condition"].items():
            where = (f"{row['n_labelled']} labelled of {row['n_population']} windows "
                     f"({100 * row['share_of_map']:.1f}% of the map)")
            if not row["n_labelled"]:
                lines.append(f"{name}: no labelled window fell in this condition, so nothing can be said about it; {where}")
            else:
                lines.append(f"{name} {_rate(row)} ({100 * row['low']:.1f}% to {100 * row['high']:.1f}%), {where}")
        if res["outside_condition_intervals"]:
            lines.append(_outside_note(res))
        if res["design"] == "condition" and len(res["per_condition"]) > 1:
            # the whole-map line says "95% interval": this says how it holds and why it is wider than a stratified
            # one; the JSON's condition_note alone did not reach a reader of the printed result (review of 2026-09-29)
            lines.append(f"note: {est.CLASS_WHOLE_MAP if by_class else est.CONDITION_WHOLE_MAP}")
        cond_text = "\n" + "\n".join(lines)
    # the interval is printed as its two ends: it is not symmetric about the estimate (Wilson never is, and a
    # clipped one is not), so "estimate +/- x" would name an interval that is not the one written
    head = (f"error rate {_rate(res)}, 95% interval {100 * res['low']:.1f}% to {100 * res['high']:.1f}% "
            f"(half-width {100 * res['half_width']:.1f} points), from {res['n_labelled']} labelled windows of {res['n_population']}; "
            f"{res['method']}")
    if valid:
        # 1.20.0: the valid interval first, then the line printed without --valid, unchanged, labelled so
        v = res["valid_interval"]
        head = (f"error rate {_rate(v)}, valid 95% interval {100 * v['low']:.1f}% to {100 * v['high']:.1f}% "
                f"(half-width {100 * v['half_width']:.1f} points), from {res['n_labelled']} labelled windows of "
                f"{res['n_population']}; {v['method']}\nwithout --valid: {head}")
    print(words(head + (f"\nnote: {res['bounds_note']}" if res.get("bounds_note") else "")
          + (f"\nwarning: {res['warning']}" if "warning" in res else "")
          + (f"\nnote: {est.VALID_NOTE}" if valid else "") + cond_text + per_class_text
          + (f"\nwarning: {res['per_class_warning']}" if "per_class_warning" in res else "")
          + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
          + (f"\nnote: {res['sequential_note']}" if "sequential_note" in res else "") + also_text
          + (f"\nnote: {res['profile_note']}" if "profile_note" in res else "")) + f"\nwrote {out}")
    return 0


# estimate on a sample drawn by profile: each profile's rate is printed above in profile words (class_wording); this
# says what certify adds
PROFILE_ESTIMATE_NOTE = ("This sample was drawn by profile (sample --design profile). certify --alpha A gives each "
                         "profile a verdict, good, bad or undecided against A, with delta split over all of them; A is "
                         "chosen before any label is read.")


ALSO_ACCEPT_HELP = ("JSON {map class: [reference classes also accepted]}, written before any label is read: a second "
                    "reading of the same labels, which needs a `reference_class` column (an audit sample has one)")


def add_estimate_parser(sub):
    e = sub.add_parser("estimate", help="the map's error rate with an interval, from the labelled sample CSV, and each "
                                        "input condition's when the sample was drawn with --condition")
    e.add_argument("sample", help="the CSV `sample` wrote, with its `wrong` column filled in (its `reference_class` "
                                  "column, for a sample drawn with --other or --design audit)")
    e.add_argument("--out", default=None, help="JSON to write (default: <sample>_estimate.json)")
    e.add_argument("--per-class", action="store_true",
                   help="also user's accuracy, producer's accuracy and error-adjusted share per class; needs a "
                        "`reference_class` column in the CSV and the map's scores (from the sidecar, or --scores)")
    e.add_argument("--classes", type=int, default=None, metavar="L",
                   help="with --per-class: the number of classes the map's legend holds, stated before the labels are "
                        "read (default: the largest class id plus one); the exact per-class intervals of a random "
                        "sample hold together over L classes, so a product whose ids are sparse (10 to 100) gives "
                        "narrower intervals with its true legend size")
    e.add_argument("--scores", default=None, help="with --per-class or --also-accept: the raster `sample` was run on, if "
                                                  "it has moved")
    e.add_argument("--confidence", default=None, help="with --per-class or --also-accept: the confidence band `sample` "
                                                      "was run on, if it has moved")
    e.add_argument("--reviewer-false-alarm", type=float, default=None, metavar="E0",
                   help="at most this share of the truly correct windows does the reviewer mark wrong (0 to below 1); "
                        "widens the interval's lower end")
    e.add_argument("--reviewer-miss", type=float, default=None, metavar="E1",
                   help="at most this share of the truly wrong windows does the reviewer mark right (0 to below 1); "
                        "widens the interval's upper end")
    e.add_argument("--nodata", type=float, default=None,
                   help="with --per-class or --also-accept: the no-data value `sample` was run with (default: the one its "
                        "sidecar records; a different value is refused); only needed for a sample written by 1.2.0")
    e.add_argument("--alpha", type=float, default=None, metavar="A",
                   help="audit sample: a verdict per class, above, below or undetermined against A, chosen before the "
                        "labels are read")
    e.add_argument("--also-accept", default=None, metavar="FILE", help=ALSO_ACCEPT_HELP)
    e.add_argument("--valid", action="store_true", help=est.VALID_HELP)
    e.set_defaults(func=cmd_estimate)

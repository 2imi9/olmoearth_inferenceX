"""The command plan: how many labels each route needs before any is drawn, and the sample command that draws them."""
import json
import os

import numpy as np

from oe_inferencex.review.assess import assess_classmap, assess_prediction
from oe_inferencex.review.windows import _two_map_windows
from oe_inferencex.common.maps import _check_scores
from oe_inferencex.common.output import (_check_out_file, _check_patch, _nodata_arg, _num_arg, _refuse_overwriting_inputs,
                                         _stamp)
from oe_inferencex.common.product import PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product
from oe_inferencex.common.raster import read_raster


def _plan_windows(args):
    """The valid windows of one map, counted as `sample` counts them (the same reader, patch and no-data rule)."""
    _check_patch(args.patch, "plan")
    conf_path, conf_range = _product_args(args, "plan")
    if conf_path:
        hard, conf_px, valid, _, n_classes, notes, _ = _read_product(args.scores, conf_path, args.nodata, conf_range, "plan")
        try:
            out = assess_classmap(hard, conf_px, n_classes, patch=args.patch, nodata_mask=~valid, signal=PRODUCT_SIGNAL)
        except ValueError as exc:
            raise SystemExit(f"plan: {exc}") from None
    else:
        notes = []
        scores, valid, _ = read_raster(args.scores, args.nodata)
        _check_scores(scores, valid, args.logits, args.scores)
        try:
            out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid)
        except ValueError as exc:
            raise SystemExit(f"plan: {exc}") from None
    return int(np.asarray(out["arrays"]["valid"]).sum()), list(notes)


def _points(x):
    return f"{100 * x:.3g} point" + ("" if abs(100 * x - 1) < 1e-9 else "s")


def _labels(n):
    return f"{n} label" + ("" if n == 1 else "s")


def _ladder_text(probs, rec, power):
    """The ladder's budgets up to twice the recommendation (or all of them), and the recommendation, as budget:
    probability; the JSON holds every budget checked."""
    from oe_inferencex.plan import ladder
    keys = sorted(int(b) for b in probs)
    cut = 2 * rec if rec else keys[-1]
    rungs = set(ladder(keys[-1])) | {keys[0], keys[-1]} | ({rec} if rec else set())
    shown = [b for b in keys if b <= cut and b in rungs] or keys[:1]
    return ", ".join(f"{b}: {'-' if probs[str(b)] is None else f'{probs[str(b)]:.2f}'}" for b in shown)


def _sample_command(args, budget, other):
    """The sample command that draws the planned labels, with the options plan was given."""
    import shlex
    if args.scores is None:
        cmd = ["oe-inferencex", "sample", "MAP"] + (["--other", "MAP_B"] if other else [])
    else:
        cmd = ["oe-inferencex", "sample", args.scores] + (["--other", args.other] if other else [])
    cmd += [] if other else ["--design", "random"]
    cmd += ["--budget", str(budget), "--out", "to_label.csv"]
    if args.patch is not None and args.patch != 4:
        cmd += ["--patch", str(args.patch)]
    if args.nodata is not None:
        cmd.append(_nodata_arg(args.nodata))
    if args.logits and not other:           # sample --other reads no logits and refuses the option
        cmd += ["--logits"]
    if other and args.threshold is not None:
        cmd.append(f"--threshold={args.threshold!r}")     # one token: argparse reads '-1e-05' as an option
    if getattr(args, "confidence", None):
        cmd += ["--confidence", args.confidence]
    if getattr(args, "confidence_range", None):
        cmd += ["--confidence-range", *[_num_arg(v) for v in args.confidence_range]]
    return " ".join(shlex.quote(c) for c in cmd)


def _reach(part, top, does, do):
    """What a plan's budgets reach, said only of the budgets checked: the recommendation and how it was checked, the
    runs of budgets that reach the probability when the largest one checked falls short, or that none does. `does`
    and `do` are the outcome with a singular and a plural subject."""
    rec, span, runs = part.get("labels"), part.get("every_budget_checked"), part.get("runs_reaching_power") or []
    if rec is not None:
        p = part["probability_at_labels"]
        how = (f"each one from {span[0]} to {span[1]}, then the ladder up to {top}" if span and span[1] > span[0]
               else f"the ladder up to {top}")
        return f"from {_labels(rec)} every budget checked {does} ({p:.3f} at {rec}; budgets checked: {how})"
    if runs:
        said = ", ".join(f"{a} to {b}" if a != b else f"{a}" for a, b in runs)
        return (f"only the budgets checked from {said} {do}; larger ones fall short again, up to the largest checked, "
                f"{top}")
    return f"no budget checked up to {top} {does}"


def cmd_plan(args):
    """How many labels to draw before drawing any, for each labelled route: an error rate no wider than a stated
    width, which of two maps is more accurate, a certified zone. Per budget, the probability that the package's own
    procedure reaches the outcome, and the smallest budget from which every larger one checked reaches it."""
    from oe_inferencex import plan
    want_rate, want_which = args.width is not None, args.difference is not None
    want_zone = args.coverage is not None or args.alpha is not None or args.zone_error is not None
    want_class = args.class_share is not None
    if not (want_rate or want_which or want_zone or want_class):
        raise SystemExit("plan: say what to plan: --width (an error rate, its interval no wider than this), "
                         "--difference with --other (which of two maps), --coverage with --alpha (a certified zone), "
                         "or --class-share with --classes (a class the map draws wrong)")
    if want_class and args.classes is None:
        raise SystemExit("plan: --class-share needs --classes, the number of classes in the map's legend (the per-class "
                         "intervals hold together over all of them)")
    if want_zone and (args.coverage is None or args.alpha is None):
        raise SystemExit("plan: a certified zone needs both --coverage (a level of the grid, 0.05 to 1 in steps of 0.05) "
                         "and --alpha (the error rate the zone must not exceed)")
    if want_which and args.other is None and args.differing is None:
        raise SystemExit("plan: --difference compares two maps: give the second with --other, or the windows where they "
                         "differ with --differing and --windows")
    two = args.other is not None or args.differing is not None
    if two and (want_rate or want_zone or want_class):
        raise SystemExit("plan: --other and --differing plan which of two maps is more accurate; plan an error rate or a "
                         "zone of one map in a separate call")
    if two and not want_which:
        raise SystemExit("plan: --other and --differing go with --difference")
    unused = [flag for flag, value, needs in (
        ("--error-rate", args.error_rate, want_rate), ("--both-wrong", args.both_wrong, want_which),
        ("--zone-error", args.zone_error, want_zone), ("--delta", args.delta, want_zone),
        ("--threshold", args.threshold, args.other is not None),
        ("--two-class", args.two_class or None, want_which), ("--classes", args.classes, want_class),
        ("--drawn-at", args.drawn_at, want_class)) if value is not None and not needs]
    if unused:
        one = len(unused) == 1
        raise SystemExit(f"plan: {', '.join(unused)} {'does' if one else 'do'} not apply to what is planned here; leave "
                         f"{'it' if one else 'them'} out")
    if args.scores is None and args.windows is None:
        raise SystemExit("plan: give the map, or the number of its valid windows with --windows")
    if args.scores is not None and (args.windows is not None or args.differing is not None):
        raise SystemExit("plan: give the map or --windows (and --differing), not both: the counts are read from the map")
    if args.differing is not None and args.windows is None:
        raise SystemExit("plan: --differing needs --windows, the windows both maps predict")
    if args.other is not None and args.scores is None:
        raise SystemExit("plan: --other is the second map; give the first map too")
    if args.scores is None and (args.patch is not None or args.nodata is not None or args.logits
                                or getattr(args, "confidence", None) or getattr(args, "confidence_range", None)):
        raise SystemExit("plan: --patch, --nodata, --logits and --confidence describe a map; with --windows there is none")
    if args.other is not None and (getattr(args, "confidence", None) or getattr(args, "confidence_range", None)):
        raise SystemExit("plan: --other plans the labels among the windows where two class maps differ, which needs no "
                         "confidence; leave out --confidence")
    if args.max_labels is not None and args.max_labels < 1:
        raise SystemExit(f"plan: --max-labels must be at least 1, got {args.max_labels}")
    if args.scores is not None and args.patch is None:
        args.patch = 4
    out_path = args.out
    if out_path:
        # checked here, written at the end: a refused plan leaves no directory behind
        if os.path.isdir(out_path):
            raise SystemExit(f"plan: --out {out_path} is a directory; name the JSON to write, for example "
                             f"{os.path.join(out_path, 'plan.json')}")
        _refuse_overwriting_inputs("plan", [out_path], [x for x in (args.scores, args.other, getattr(args, "confidence", None)) if x])
    notes, results, runs = [], {}, []
    top_asked = plan.MAX_LABELS if args.max_labels is None else args.max_labels
    power = 0.9 if args.power is None else args.power
    try:
        if want_which:
            two_class = bool(args.two_class)
            if args.scores is not None:
                _check_patch(args.patch, "plan")
                t = _two_map_windows(args.scores, args.other, args.nodata, args.threshold, args.patch)
                ok = np.asarray(t.ok, bool)
                N2, D = int(ok.sum()), int((ok & (t.a_w != t.b_w)).sum())
                notes += list(t.notes)
                if D == 0:
                    raise SystemExit("plan: the two maps give the same class in every window both predict; there is "
                                     "nothing to label")
            else:
                N2, D = int(args.windows), int(args.differing)
            r = plan.plan_which_map(N2, D, args.difference, power, both_wrong=args.both_wrong, two_class=two_class,
                                    max_labels=top_asked)
            results["which_map"] = r
            if "refusal" in r and r.get("probability_by_budget") is None:
                print(f"Which map: {r['refusal']}.")
            else:
                w = r["wrong_verdict_probability_at_labels"]
                print(f"Which map: drawn at random among the {D} windows where the maps differ "
                      f"({100 * D / N2:.3g}% of the {N2} compared), "
                      + _reach(r, r["checked_up_to"],
                               *(f"{v} the more accurate map with probability {power:g} or more when the accuracies "
                                 f"differ by {_points(args.difference)} or more" for v in ("names", "name")))
                      + f", for {r['planned_for']}"
                      + ("" if w is None else "; the chance of naming the less accurate one there is "
                         + (f"at most {w:.2g}" if w >= 1e-12 else "below 1e-12")) + ".")
                print(f"  by budget: {_ladder_text(r['probability_by_budget'], r['labels'], power)}")
                if r["labels"] is not None:
                    runs.append(_sample_command(args, r["labels"], other=True))
            if two_class:
                notes.append("--two-class: where two-class maps differ one of them is right, so no differing window is "
                             "wrong in both, and the difference planned has the parity of the windows that differ")
            else:
                notes.append("planned at the worst split checked of the differing windows; if both maps have only two "
                             "classes, --two-class plans their one split, which needs fewer labels")
        if want_rate or want_zone or want_class:
            N = int(args.windows) if args.scores is None else None
            if N is None:
                N, more = _plan_windows(args)
                notes += more
        if want_rate:
            r = plan.plan_error_rate(N, args.width, args.error_rate, power, max_labels=top_asked)
            results["error_rate"] = r
            print(f"Error rate: on a map of {N} valid window{'' if N == 1 else 's'} wrong {100 * r['error_rate']:.3g}% "
                  "of the time, drawn at random, "
                  + _reach(r, r["checked_up_to"],
                           *(f"{v} a 95% interval no wider than {_points(args.width)} with probability {power:g} or more"
                             for v in ("gives", "give"))) + ".")
            print(f"  by budget: {_ladder_text(r['probability_by_budget'], r['labels'], power)}")
            if r["labels"] is not None:
                runs.append(_sample_command(args, r["labels"], other=False))
            if "rate_note" in r:
                notes.append(r["rate_note"][0].upper() + r["rate_note"][1:])
        if want_class:
            ratio = 0.5 if args.drawn_at is None else args.drawn_at
            try:
                r = plan.plan_class(N, args.class_share, args.classes, ratio, power, max_labels=top_asked)
            except ValueError as exc:
                raise SystemExit(f"plan: {str(exc).replace('drawn_ratio', '--drawn-at')}")
            results["class"] = r
            print(f"Class: a class that is {100 * args.class_share:.3g}% of the map's {N} window{'' if N == 1 else 's'}, "
                  f"which the map draws at {ratio:g} of its share, drawn at random, "
                  + _reach(r, r["checked_up_to"],
                           *(f"{v} it drawn too {'little' if ratio < 1 else 'much'} with probability {power:g} or more"
                             for v in ("flags", "flag")))
                  + f" (exact; the intervals of all {args.classes} classes hold together at 95%).")
            print(f"  by budget: {_ladder_text(r['probability_by_budget'], r['labels'], power)}")
            if r["labels"] is not None:
                runs.append(_sample_command(args, r["labels"], other=False))
            notes.append("planned for one class at the share stated; a rarer class needs more labels (at 1% of the "
                         "map several thousand), and the labels then need a `reference_class` column")
        if want_zone:
            delta = 0.1 if args.delta is None else args.delta
            r = plan.plan_zone(N, args.coverage, args.alpha, args.zone_error, power, delta, max_labels=top_asked)
            results["zone"] = r
            head = (f"Zone: {100 * r['coverage']:.3g}% of the map ({r['zone_windows']} of {N} windows) at alpha "
                    f"{args.alpha:g}" + (f", under the prefix and Bonferroni rules, is tested from "
                                         f"{_labels(r['labels_to_test'])} drawn at random; fewer cannot certify it "
                                         "under them even with no error among them." if r.get("labels_to_test")
                                         else ":"))

            def target_line(t):
                # the target rule tests a target fixed before labelling at any budget (1.16.0)
                return (f"  With this zone fixed as the target before labelling (sample --target {r['coverage']:g}), "
                        "under the target rule, " + _reach(t, r["checked_up_to"], "certifies it", "certify it")
                        + f", however its errors spread (exact: the target's own test at its share of delta, "
                        f"{t['weight']:g} of it or all of it while no smaller zone is tested, a lower bound on "
                        "certifying it or a larger zone).")
            if "refusal" in r:
                print(f"{head} {r['refusal'][0].upper() + r['refusal'][1:]}.")
                if "target" in r:
                    print(target_line(r["target"]))
                    tl = r["target"]
                    print(f"  target by budget: {_ladder_text(tl['probability_by_budget'], tl['labels'], power)}")
                    if r["target"]["labels"] is not None:
                        runs.append(_sample_command(args, r["target"]["labels"], other=False)
                                    + f" --target {r['coverage']:g}")
            elif "unplanned" in r:
                print(f"{head} How many more it needs depends on how often the zone is wrong, which only the labels "
                      "measure; give --zone-error, the rate you expect, to plan them.")
            else:
                b, ev = r["bonferroni"], r["prefix_even"]
                low = r["prefix_most"]["below_power_at"]
                print(f"{head} For a zone wrong {100 * args.zone_error:.3g}% of the time, with probability {power:g} or "
                      f"more: under the Bonferroni rule, " + _reach(b, r["checked_up_to"], "certifies it", "certify it")
                      + ", however its errors spread (exact); under the prefix rule (the default), "
                      + _reach(ev, r["checked_up_to"], "certifies it", "certify it")
                      + f" when its more confident zones are wrong no more often (simulated, {ev['draws']} draws; a "
                      f"budget counts when its estimate less two standard errors reaches the probability), and "
                      f"no spread of the errors reaches the probability at {len(low)} of the "
                      f"{len(r['budgets_checked'])} budgets checked (exact).")
                t = r["target"]
                print(target_line(t))
                print(f"  Bonferroni by budget: {_ladder_text(b['probability_by_budget'], b['labels'], power)}")
                print(f"  target by budget: {_ladder_text(t['probability_by_budget'], t['labels'], power)}")
                print(f"  prefix by budget (-: below {power:g} however the errors spread): "
                      f"{_ladder_text(ev['probability_by_budget'], ev['labels'], power)}")
                if ev["drawn"] != "without replacement":
                    notes.append(f"The prefix rule's simulation drew its labels {ev['drawn']}; the bound is "
                                 "subtracted from each budget's lower end")
                if b["labels"] is not None or ev["labels"] is not None:
                    cmd = _sample_command(args, max(v for v in (b["labels"], ev["labels"]) if v is not None), other=False)
                    if ev["labels"] is None:
                        # Only the Bonferroni rule reaches the probability, so the run fixes it with the draw: a bare
                        # sample line certified with the default prefix rule reached about 0.34 where 0.9 was planned
                        # (the bug hunt of 2026-10-25).
                        cmd += f" --seed-from-map --alpha {args.alpha!r} --rule bonferroni" + (
                            "" if args.delta is None else f" --delta {args.delta!r}")
                    runs.append(cmd)
                if t["labels"] is not None:
                    runs.append(_sample_command(args, t["labels"], other=False) + f" --target {r['coverage']:g}")
            if r.get("entry_budgets"):
                notes.append("At " + ", ".join(f"{e} labels the {float(g):.0%} zone" for g, e in r["entry_budgets"].items())
                             + " becomes testable; from each of these budgets the rules can lose power until more "
                             "labels restore it: the prefix rule tests the smallest testable zone first, with few "
                             "labels, the Bonferroni rule splits delta over one more zone, and under the target rule "
                             "the target's share falls from all of delta to 0.8 of it where the first zone smaller "
                             "than the target enters. The budget one below each is the last before the fall")
    except ValueError as exc:
        raise SystemExit(f"plan: {exc}") from None
    for cmd in runs:
        print(f"run: {cmd}")
    for n in notes:
        print(f"note: {n}")
    print(f"note: {plan.PLAN_NOTE}")
    if out_path:
        _check_out_file(out_path, "plan")
        with open(out_path, "w") as f:
            json.dump(_stamp({"plans": results, "notes": notes, "run": runs}, "plan"), f, indent=1, default=float)
        print(f"wrote {out_path}")
    return 0


def add_plan_parser(sub):
    pl = sub.add_parser("plan", help="how many labels to draw before drawing any: an error-rate interval no wider than a "
                                     "stated width, which of two maps is more accurate, a certified zone, or a class "
                                     "the map draws wrong")
    pl.add_argument("scores", nargs="?", default=None,
                    help="the map, as sample takes it (not needed with --windows)")
    pl.add_argument("--other", default=None, metavar="MAP_B",
                    help="a second map of the same grid and the same time: plan the labels that say which map is more "
                         "accurate")
    pl.add_argument("--windows", type=int, default=None, help="the number of valid windows, instead of a map")
    pl.add_argument("--differing", type=int, default=None,
                    help="with --windows: the windows where two maps differ, instead of the two maps")
    pl.add_argument("--width", type=float, default=None,
                    help="error rate: the widest 95%% interval acceptable, high minus low, e.g. 0.1 for 10 points")
    pl.add_argument("--error-rate", type=float, default=None,
                    help="error rate: the rate you expect the map to have (default 0.5, where intervals are widest)")
    pl.add_argument("--difference", type=float, default=None,
                    help="two maps: the smallest whole-map accuracy difference worth detecting, e.g. 0.02")
    pl.add_argument("--both-wrong", type=float, default=None,
                    help="two maps: the share of the differing windows wrong in both (default: the worst split checked)")
    pl.add_argument("--two-class", action="store_true",
                    help="two maps: both have only two classes, so where they differ one is right; plans their one split")
    pl.add_argument("--coverage", type=float, default=None,
                    help="zone: the share of the map to certify, a level of the grid (0.05 to 1 in steps of 0.05)")
    pl.add_argument("--alpha", type=float, default=None, help="zone: the error rate the zone must not exceed")
    pl.add_argument("--zone-error", type=float, default=None,
                    help="zone: the rate you expect that zone to be wrong; without it only the floor is planned")
    pl.add_argument("--delta", type=float, default=None, help="zone: the chance the certificate may fail (default 0.1)")
    pl.add_argument("--class-share", type=float, default=None, metavar="S",
                    help="one class: its true share of the windows (by the reference, not the share the map draws); "
                         "plans the labels that flag it as drawn wrong")
    pl.add_argument("--classes", type=int, default=None, help="with --class-share: the number of classes in the legend")
    pl.add_argument("--drawn-at", type=float, default=None, metavar="R",
                    help="with --class-share: the share the map draws, as a fraction of the class's (default 0.5; above 1 "
                         "plans an over-draw)")
    pl.add_argument("--power", type=float, default=None, help="the probability of the outcome planned for (default 0.9)")
    pl.add_argument("--max-labels", type=int, default=None,
                    help="the largest budget checked (default 10000, or the census when smaller)")
    pl.add_argument("--logits", action="store_true")
    pl.add_argument("--patch", type=int, default=None, help="window side in pixels, as sample will use (default 4)")
    pl.add_argument("--nodata", type=float, default=None)
    pl.add_argument("--threshold", type=float, default=None,
                    help="with --other: cut-off of a 2-D continuous map (0.5 for a probability map when omitted)")
    pl.add_argument("--out", default=None, help="JSON to write the plan to (optional)")
    _product_parser_args(pl)
    pl.set_defaults(func=cmd_plan)

"""The commands certify, certificate and verify: which part of the map can be trusted, packed to travel, checked."""
import json
import os

import numpy as np

from oe_inferencex.certify import zone as est
from oe_inferencex.common.labels import (_also_accept_rows, _also_words, _labelled_sample, _population_note,
                                         _sample_map_inputs, _sidecar_design, _unused_note)
from oe_inferencex.common.output import _check_out_file, _refuse_overwriting_inputs, _same_file, _stamp
from oe_inferencex.common.wording import class_wording
from oe_inferencex.review.windows import _map_windows


# ----------------------------------------------------------------------------- certificate and verify
def cmd_certificate(args):
    """A certified zone packed to travel with the map: a STAC Item stating the guarantee, the zone as a georeferenced
    mask, the labelled sample, its sidecar and any condition layer, and digests binding them to the map; verified
    before it is handed over (certificate.py, 1.17.0)."""
    from oe_inferencex.certify import certificate as cert
    try:
        item = cert.build(args.zone, args.out, args.date, args.scores, args.confidence)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise SystemExit(cert.printable(f"certificate: {exc}")) from None
    p = item["properties"]
    band = " --confidence <their copy of the band>" if p["zc:map"].get("confidence_sha256") else ""
    print(cert.printable(
        f"{p['zc:statement']}\nwrote {os.path.join(args.out, 'certificate.json')} with "
        f"{', '.join(a['href'] for a in item['assets'].values())}, and verified it; the map's digest is "
        f"{p['zc:map']['map_sha256'][:16]}...; anyone holding the map can check it: oe-inferencex verify "
        f"{os.path.join(args.out, 'certificate.json')} --scores <their copy of the map>{band}\nnote: "
        f"{cert.ISSUERS_WORD}"))
    return 0


def cmd_verify(args):
    """Check a certificate by building it again from the map, the bundled labels and the recorded settings: the
    bundle's files, the map, the seed, the draw, certify's result, the certificate's every field and the zone mask;
    and with --at or --at-pixel, whether a point lies in the zone certify gives again."""
    from oe_inferencex.certify import certificate as cert
    if args.out:
        base = os.path.dirname(os.path.abspath(args.certificate))
        out_dir = os.path.dirname(os.path.abspath(args.out))
        # another letter case or a symlink to the bundle counts too, and so does a directory under it, which made
        # every later verify fail on a file that is not the bundle's (the review of 1.17.2)
        d, inside = out_dir, False
        while True:
            if os.path.exists(d) and _same_file(d, base) or os.path.realpath(d) == os.path.realpath(base):
                inside = True
                break
            up = os.path.dirname(d)
            if up == d:
                break
            d = up
        if inside:
            raise SystemExit("verify: --out would write into the bundle's directory, which holds the bundle's files "
                             "and nothing else; name a file elsewhere")
        _check_out_file(args.out, "verify")
        named = []                             # the map the certificate names, read when --scores is not given
        try:
            with open(args.certificate) as f:
                m = (json.load(f).get("properties") or {}).get("zc:map") or {}
            named = [m.get("path"), m.get("confidence_path")]
        except (OSError, ValueError, AttributeError):
            pass
        _refuse_overwriting_inputs("verify", [args.out], [args.certificate, args.scores, args.confidence, *named])
    try:
        res = cert.verify(args.certificate, args.scores, args.confidence)
    except cert.Unreadable as exc:
        raise SystemExit(f"verify: {exc}") from None
    pt = pt_error = None
    if args.at or args.at_pixel:
        x, y = args.at if args.at else (None, None)
        r, c = args.at_pixel if args.at_pixel else (None, None)
        if res["mask"] is None:                           # only the zone certify gives again is looked in
            pt_error = "the zone could not be built again, so no point is looked up in it"
        else:
            try:
                pt = cert.point(args.certificate, x=x, y=y, row=r, col=c, mask=res["mask"])
            except Exception as exc:                      # a point not looked up never hides the checks
                pt_error = str(exc)
    mark = {True: "ok", False: "FAILED", None: "not checked"}
    lines = [f"{name}: {mark[c['ok']]}: {c['detail']}" for name, c in res["checks"].items()]
    lines += [f"note: {n}" for n in res["notes"]]
    if pt is not None:
        lines.append("point: " + (f"window {pt['window'][0]}, {pt['window'][1]} is " if pt["window"] else "it is ")
                     + pt["where"] + ("" if res["verified"] else " (in the zone certify gives again; the certificate "
                                      "is not verified)"))
    elif pt_error:
        lines.append(f"point: not looked up: {pt_error}")
    verdict = ("verified: " + res["statement"] if res["verified"] else
               "not verified: a check failed or could not be made; the certificate's statement is not shown to hold "
               "for this map")
    print("\n".join(cert.printable(line) for line in lines + [verdict]))     # each field on its own line
    if args.out:
        with open(args.out, "w") as f:
            json.dump(_stamp({"verified": res["verified"], "checks": res["checks"], "notes": res["notes"],
                              "statement": res["statement"] if res["verified"] else None, "map_read": res["map_read"],
                              "point": pt, "certificate": os.path.abspath(args.certificate)}, "verify"), f, indent=1)
    return 0 if res["verified"] else 1


def _keep_to_plan(args, side):
    """certify's options from the plan a sample drawn for a certificate fixed (sample --seed-from-map), refusing any
    other: a second choice after the labels would be a second test. Without a plan, --alpha is required and --delta
    defaults to est.ZONE_DELTA."""
    plan = side.get("certify")
    if plan is not None and not (isinstance(plan, dict) and type(plan.get("alpha")) is float
                                 and type(plan.get("delta")) is float):
        raise SystemExit("certify: the sidecar's `certify` plan is not the one sample writes (alpha and delta as "
                         "numbers); draw the sample again")
    if not plan:
        if args.alpha is None:
            raise SystemExit("certify: --alpha is required: the error rate the certified zone may not exceed")
        args.delta = est.ZONE_DELTA if args.delta is None else args.delta
        return
    for flag, given, fixed in (("--alpha", args.alpha, plan["alpha"]), ("--delta", args.delta, plan["delta"])):
        if given is not None and float(given) != float(fixed):
            raise SystemExit(f"certify: this sample fixed {flag} {fixed:g} before any label (sample --seed-from-map); "
                             f"{flag} {given:g} on the same labels would be a second test")
    args.alpha, args.delta = float(plan["alpha"]), float(plan["delta"])
    rule, cut = plan.get("rule"), plan.get("level_cut")
    if rule == "sequential":
        # an explicit --rule prefix passed too, where Usage says --rule is refused here (the bug hunt of 2026-10-25)
        if args.rule is not None or args.level_cut is not None:
            raise SystemExit("certify: this sample is certified by the sequential rule it fixed before any label")
        return
    if args.rule not in (None, rule):
        raise SystemExit(f"certify: this sample fixed the {rule} rule before any label; --rule {args.rule} on the same "
                         "labels would be a second test")
    if args.level_cut not in (None, cut):
        raise SystemExit(f"certify: this sample fixed the {cut} level cut before any label; --level-cut "
                         f"{args.level_cut} on the same labels would be a second test")
    args.rule, args.level_cut = rule, cut


def cmd_certify(args):
    """The largest most-confident share of the map that is wrong at most --alpha of the time, certified from a
    random labelled sample so that the statement fails with probability at most --delta (exp80). A sample that
    records an input condition is certified per condition instead, with --delta split over the conditions tested
    (certify_by_condition); no whole-map zone is issued for it."""
    design = _sidecar_design(args.sample)
    if design == "disagreement":
        raise SystemExit("certify needs a random sample of one map's windows; this sample holds only windows where two "
                         "maps differ, which says which map is more accurate (estimate) but nothing about either "
                         "map's zones")
    if design == "audit":
        raise SystemExit("certify needs a random sample of the whole map; an audit sample draws a fixed number from each "
                         "class's confident zone, which says which classes' confident calls are wrong (estimate) but "
                         "certifies no zone")
    out = args.out or (args.sample[:-4] + "_zone.json" if args.sample.endswith(".csv") else args.sample + "_zone.json")
    mask_path = out[:-5] + ".npy" if out.endswith(".json") else out + ".npy"
    accept_path = getattr(args, "also_accept", None)
    _refuse_overwriting_inputs("certify", [out, mask_path, _also_mask(mask_path)],
                               [args.sample, args.sample[:-4] + ".json" if args.sample.endswith(".csv") else args.sample + ".json",
                                accept_path, *_sample_map_inputs(args.sample, args.scores, args.confidence)])
    side, rows, idx, sample, wrong, unjudged = _labelled_sample(args.sample, "certify")
    # A window that could not be judged counts as wrong: that can only certify less, so the guarantee holds whatever
    # made it hard to judge. certify takes no reviewer error rate: testing at alpha (1 - miss) would need the miss
    # rate to hold inside every zone the test can certify, where a model's confident errors sit, and a rate measured
    # on the whole map does not give that (review of 2 October 2026: misses placed in the confident half certified a
    # zone wrong 10% of the time at alpha 5% on every draw).
    wrong = np.where(unjudged, 1, wrong)
    _keep_to_plan(args, side)
    alpha = args.alpha
    bounds = ([f"{int(unjudged.sum())} window(s) that could not be judged (?) are counted as wrong, which keeps the "
               "guarantee whatever made them hard to judge and certifies less"] if unjudged.any() else [])
    by_condition = side.get("condition_grid") is not None and side.get("design") in ("random", "condition")
    sequential = side.get("design") == "sequential"
    # The rule: a sample that fixed a target before labelling is certified by the target rule, and only by it, since
    # another rule on the same labels is a second test; without a target, prefix unless --rule says bonferroni.
    target = side.get("target")
    if target is not None and args.rule not in (None, "target"):
        raise SystemExit(f"certify: this sample fixed the {100 * target['coverage']:g}% zone as its target before "
                         f"labelling, so it is certified by the target rule; --rule {args.rule} on the same labels "
                         "would be a second test")
    if target is None and args.rule == "target":
        raise SystemExit("certify: the target rule needs the target fixed before labelling; draw the sample with "
                         "`sample --design random --target C`")
    # before the default is filled in: an explicit --rule prefix passed once it was (the bug hunt of 2026-10-25)
    if sequential and (args.rule is not None or args.level_cut is not None):
        raise SystemExit("certify: a sequential sample is certified by the sequential rule, which tests each zone from "
                         "the anchor outward at every look; --rule and --level-cut belong to the one-look rule")
    args.rule = args.rule or ("target" if target is not None else "prefix")
    if side.get("design") != "random" and not by_condition and not sequential:
        raise SystemExit(f"certify needs a random sample: this CSV was drawn with the {side.get('design')!r} design. The "
                         "guarantee rests on the labelled windows inside each zone being a random sample of that zone, "
                         "which a stratified or tile draw is not. Draw one with `sample --design random`")
    margin, hard, valid_w, _ = _map_windows(side, args.scores, args.nodata, "certify", rows, idx,
                                            getattr(args, "confidence", None))
    also = None
    if accept_path is not None:
        wrong_alt, block, unused = _also_accept_rows("certify", accept_path, rows, wrong, unjudged, hard.ravel()[idx])
        also = (np.where(unjudged, 1, wrong_alt), block, unused)       # a ? window is wrong in both readings
    _check_out_file(out, "certify")                   # before the mask, which a refusal used to leave behind
    if by_condition:
        return _certify_by_condition(args, sample, wrong, margin, valid_w, out, mask_path, bounds, int(unjudged.sum()),
                                     also=also)
    if sequential:
        return _certify_sequential(args, side, idx, wrong, margin, valid_w, out, mask_path, bounds, int(unjudged.sum()),
                                   also=also)
    try:
        tkw = {} if target is None else {"target": target["coverage"], "target_weight": target["weight"]}
        res = est.certify_zone(margin.ravel(), idx, wrong, alpha, delta=args.delta, rule=args.rule, valid=valid_w.ravel(),
                               cut=args.level_cut, **tkw)
        res_alt = None if also is None else est.certify_zone(margin.ravel(), idx, also[0], alpha, delta=args.delta,
                                                             rule=args.rule, valid=valid_w.ravel(), cut=args.level_cut,
                                                             **tkw)
    except ValueError as exc:
        raise SystemExit(f"certify: {exc}")
    hw, ww = margin.shape
    if res["coverage"] is not None:
        zone = np.zeros(hw * ww, bool)
        zone[np.asarray(res.pop("zone_indices_in_order"), int)] = True
        np.save(mask_path, zone.reshape(hw, ww))
        res["zone_mask"] = os.path.abspath(mask_path)
    elif os.path.exists(mask_path):
        os.remove(mask_path)          # a previous run's zone must not sit beside a result that certifies none
    a = _zone_also_accept(res_alt, also and also[1], _also_mask(mask_path), (hw, ww))
    res["sample"] = os.path.abspath(args.sample)
    if bounds:
        res.update({"n_unjudged": int(unjudged.sum()), "bounds_note": "; ".join(bounds)})
    if _population_note(side):                       # a product's range left pixels out: the zone is of the rest
        res["population_note"] = _population_note(side)
    what = "confidence" if side.get("confidence") else "confidence margin"     # a product's band is no margin
    also_text = ""
    if a is not None:
        res.update({"also_accept": a, "also_accept_unused": also[2],
                    "reading_note": _zone_reading_note(args.delta, False)})
        also_text = _zone_also_text(res, args, what, _also_mask(mask_path))
    with open(out, "w") as f:
        json.dump(_stamp(res, "certify"), f, indent=1, default=float)
    tail = "".join(f"\nnote: {b}" for b in bounds) + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
    if res["coverage"] is not None:
        tail += f"\nnote: {est.ISSUED_NOTE}"
    if res["coverage"] is None:
        print(f"no zone certified at alpha={args.alpha:g}, delta={args.delta:g} from {res['n_labelled']} labels. {res['note']}"
              f"{tail}{also_text}\nwrote {out}")
    else:
        print(f"the {100 * res['coverage']:.0f}% most confident windows ({res['n_zone']} of {res['n_population']}, {what} >= "
              f"{res['edge']}) are wrong at most {100 * args.alpha:g}% of the time; this statement fails on at most "
              f"{100 * args.delta:g}% of samples like this one ({args.rule} rule; the exact upper bound on the zone's error "
              f"rate at that level is {100 * res['upper_bound']:.1f}%). Outside the zone nothing is certified.\n"
              f"{res['note']}{tail}{also_text}\nwrote {out} and the window mask {mask_path}")
    return 0


# The second reading of certify (--also-accept), on each route: the same function a second time on the same labels,
# with the windows the file accepts counted right. Only the keys that depend on the labels are repeated in its block;
# the rule, alpha, delta, the population, the levels tested, the anchor and the split of delta are the strict
# result's, the same in both (tests/test_also_accept.py checks it).
ZONE_ALSO_KEYS = ("coverage", "n_zone", "threshold", "edge", "upper_bound", "n_tied_at_threshold", "n_tied_inside_zone",
                  "certified_share_of_map", "n_certified", "per_condition", "levels", "note")


def _also_mask(mask_path):
    """The second reading's window mask, beside the strict one: <zone stem>_also_accept.npy."""
    return mask_path[:-4] + "_also_accept.npy"


def _zone_reading_note(delta, sequential):
    return (est.ZONE_READING_NOTE.format(delta=delta, either=min(1.0, 2 * float(delta)))
            + (est.ZONE_READING_SEQUENTIAL if sequential else ""))


def _zone_also_accept(res_alt, block, mask_alt, shape):
    """The `also_accept` block of a certify result: `block` with the second reading's zone (ZONE_ALSO_KEYS, where
    present) and its window mask, written to `mask_alt` when that reading certifies something. Otherwise, and without
    the option (res_alt None, and None returned), a stale mask there is removed, as the strict mask is."""
    if res_alt is None:
        if os.path.exists(mask_alt):
            os.remove(mask_alt)
        return None
    zone_idx = np.asarray(res_alt.pop("zone_indices_in_order", np.zeros(0, int)), int)
    for entry in (res_alt.get("per_condition") or {}).values():
        entry.pop("zone_indices_in_order", None)
    out = dict(block)
    out.update({k: res_alt[k] for k in ZONE_ALSO_KEYS if k in res_alt})
    if (res_alt.get("certified_share_of_map") if res_alt.get("by_condition") else res_alt.get("coverage")) is not None:
        zone = np.zeros(int(np.prod(shape)), bool)
        zone[zone_idx] = True
        np.save(mask_alt, zone.reshape(shape))
        out["zone_mask"] = os.path.abspath(mask_alt)
    elif os.path.exists(mask_alt):
        os.remove(mask_alt)
    return out


def _zone_also_text(res, args, what, mask_alt):
    """The printed lines of certify's second reading, before `wrote`: the zone, or each condition's in a few words."""
    a = res["also_accept"]
    head = f"also accepting {_also_words(a)} ({os.path.basename(a['source'])})"
    if res.get("by_condition"):
        lines = [f"{head}, certified per input condition at the same alpha and delta:"]
        for name, e in a["per_condition"].items():
            if e.get("coverage") is not None:
                lines.append(f"  {name}: {100 * e['coverage']:.0f}% of this condition certified ({e['n_zone']} of "
                             f"{e['n_population']})")
            else:
                lines.append(f"  {name}: nothing certified ({e.get('reason') or 'no zone passed its test'})")
        lines.append(f"together {100 * a['certified_share_of_map']:.1f}% of the map under this reading (mask {mask_alt})"
                     if a.get("certified_share_of_map") is not None else
                     "no zone is certified in any condition under this reading")
    elif a.get("coverage") is not None:
        rule = "sequential rule" if res.get("rule") == "sequential" else (
            f"{res['rule']} rule; exact upper bound {100 * a['upper_bound']:.1f}%")
        tied, inside = a.get("n_tied_at_threshold", 0), a.get("n_tied_inside_zone", 0)
        lines = [f"{head}: the {100 * a['coverage']:.0f}% most confident windows ({a['n_zone']} of {res['n_population']}, "
                 f"{what} >= {a['edge']}) are wrong at most {100 * args.alpha:g}% of the time under this "
                 f"reading, at the same delta ({rule}); window mask {mask_alt}"
                 + (f"; {tied} windows share that {what} and only {inside} of them are inside the zone, so the zone is "
                    "the window mask, not every window at or above the threshold" if tied > inside else "")]
    else:
        lines = [f"{head}: no zone certified under this reading. {a['note']}"]
    lines.append(f"note: {res['reading_note']}")
    if res["also_accept_unused"]:
        lines.append(_unused_note(res["also_accept_unused"]))
    return "\n" + "\n".join(lines)


def _certify_sequential(args, side, idx, wrong, margin, valid_w, out, mask_path, bounds=(), n_unjudged=0, also=None):
    """certify for a sequential sample: the labelled rows, read from the top, in the order they were drawn; the anchor
    the sample fixed, or the default from its first budget (sequential.anchor_coverage). `also`: the second reading
    (its `wrong`, block and unused classes), certified by the same rule from the same labels."""
    from oe_inferencex.certify import sequential as sq
    try:
        if side.get("anchor") is None:
            raise ValueError("this sequential sample records no anchor; draw it with this version's sample, which fixes "
                             "the anchor before any label")
        res = sq.certify_zone_sequential(margin.ravel(), idx, wrong, args.alpha, delta=args.delta,
                                         anchor=float(side["anchor"]), valid=valid_w.ravel(), seed=int(side["seed"]))
        res["anchor_rule"] = side.get("anchor_rule")
        res_alt = None if also is None else sq.certify_zone_sequential(
            margin.ravel(), idx, also[0], args.alpha, delta=args.delta, anchor=float(side["anchor"]),
            valid=valid_w.ravel(), seed=int(side["seed"]))
    except ValueError as exc:
        raise SystemExit(f"certify: {exc}")
    hw, ww = margin.shape
    if res["coverage"] is not None:
        zone = np.zeros(hw * ww, bool)
        zone[np.asarray(res.pop("zone_indices_in_order"), int)] = True
        np.save(mask_path, zone.reshape(hw, ww))
        res["zone_mask"] = os.path.abspath(mask_path)
    elif os.path.exists(mask_path):
        os.remove(mask_path)
    a = _zone_also_accept(res_alt, also and also[1], _also_mask(mask_path), (hw, ww))
    res["sample"] = os.path.abspath(args.sample)
    res["n_drawn"] = int(side.get("n_drawn", len(idx)))
    if bounds:
        res.update({"n_unjudged": n_unjudged, "bounds_note": "; ".join(bounds)})
    if _population_note(side):
        res["population_note"] = _population_note(side)
    what = "confidence" if side.get("confidence") else "confidence margin"
    also_text = ""
    if a is not None:
        res.update({"also_accept": a, "also_accept_unused": also[2], "reading_note": _zone_reading_note(args.delta, True)})
        also_text = _zone_also_text(res, args, what, _also_mask(mask_path))
    with open(out, "w") as f:
        json.dump(_stamp(res, "certify"), f, indent=1, default=float)
    tail = "".join(f"\nnote: {b}" for b in bounds) + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
    more = (f"\n{res['n_labelled']} of the {res['n_drawn']} windows drawn are labelled; label more from the top, or extend "
            "the sample (sample --extend), and run certify again: every look is covered by the same guarantee")
    if res["coverage"] is None:
        print(f"no zone certified yet at alpha={args.alpha:g}, delta={args.delta:g} from {res['n_labelled']} labels (sequential "
              f"rule, anchor {100 * res['anchor']:.0f}% of the map). {res['note']}{tail}{more}{also_text}\nwrote {out}")
    else:
        print(f"the {100 * res['coverage']:.0f}% most confident windows ({res['n_zone']} of {res['n_population']}, {what} >= "
              f"{res['edge']}) are wrong at most {100 * args.alpha:g}% of the time; this statement fails on at most "
              f"{100 * args.delta:g}% of samples like this one, however often certify is run on it as labels are added "
              f"(sequential rule, anchor {100 * res['anchor']:.0f}% of the map). Outside the zone nothing is certified.\n"
              f"{res['note']}{tail}\nnote: {est.ISSUED_NOTE}{more}{also_text}\nwrote {out} and the window mask "
              f"{mask_path}")
    return 0


def _certify_by_condition(args, sample, wrong, margin, valid_w, out, mask_path, bounds=(), n_unjudged=0, also=None):
    """certify for a sample that records an input condition: a zone inside each condition with enough labels, and
    the union of the zones as the window mask. The top-level zone fields stay null, since the union is not "the
    most confident share of the map" that readers of those fields take them to be. `also`: the second reading, as
    for _certify_sequential."""
    try:
        res = est.certify_by_condition(sample, wrong, margin.ravel(), args.alpha, delta=args.delta, rule=args.rule,
                                       valid=valid_w.ravel(), cut=args.level_cut)
        res_alt = None if also is None else est.certify_by_condition(sample, also[0], margin.ravel(), args.alpha,
                                                                     delta=args.delta, rule=args.rule,
                                                                     valid=valid_w.ravel(), cut=args.level_cut)
    except ValueError as exc:
        raise SystemExit(f"certify: {exc}")
    hw, ww = margin.shape
    union = np.asarray(res.pop("zone_indices_in_order"), int)
    for entry in res["per_condition"].values():
        entry.pop("zone_indices_in_order", None)
    if res["certified_share_of_map"] is not None:
        zone = np.zeros(hw * ww, bool)
        zone[union] = True
        np.save(mask_path, zone.reshape(hw, ww))
        res["zone_mask"] = os.path.abspath(mask_path)
    elif os.path.exists(mask_path):
        os.remove(mask_path)          # a previous run's zone must not sit beside a result that certifies none
    also_block = _zone_also_accept(res_alt, also and also[1], _also_mask(mask_path), (hw, ww))
    res["sample"] = os.path.abspath(args.sample)
    if bounds:
        res.update({"n_unjudged": n_unjudged, "bounds_note": "; ".join(bounds)})
    if _population_note(sample):                     # a product's range left pixels out: the zones are of the rest
        res["population_note"] = _population_note(sample)
    also_text = ""
    if also_block is not None:
        res.update({"also_accept": also_block, "also_accept_unused": also[2],
                    "reading_note": _zone_reading_note(args.delta, False)})
        also_text = _zone_also_text(res, args, None, _also_mask(mask_path))
    with open(out, "w") as f:
        json.dump(_stamp(class_wording(sample, args.sample, mask_path, getattr(args, "also_accept", None))(res),
                         "certify"), f, indent=1, default=float)
    a, d, b1 = args.alpha, args.delta, res["min_labels_to_certify"]
    if res["n_conditions_tested"]:
        lines = [f"certified per input condition at alpha={100 * a:g}%, delta={100 * d:g}% (each tested condition at "
                 f"{100 * res['delta_per_condition']:.3g}%, so the statements hold together on at least "
                 f"{100 * (1 - d):g}% of samples):"]
    elif res["delta_per_condition"] is not None:          # enough labels somewhere, and none that look drawn at random
        lines = [f"certified per input condition at alpha={100 * a:g}%, delta={100 * d:g}%: no condition was tested:"]
    else:
        lines = [f"certified per input condition at alpha={100 * a:g}%, delta={100 * d:g}%: no condition holds the {b1} "
                 "labels that certifying any zone needs, so none was tested:"]
    for name, e in res["per_condition"].items():
        if "review_set_check" in e:                      # enough labels, but they do not look like a random sample
            lines.append(f"  {name}: not tested; {e['reason']}")
        elif not e["tested"]:
            lines.append(f"  {name}: not tested; {e['n_labelled']} labels, and certifying any zone at alpha {a:g} needs "
                         f"at least {b1}")
        elif e["coverage"] is not None:
            what = "confidence" if sample.get("confidence") else "margin"
            # The whole-map line carries certify_zone's tie caveat in its note, and this line printed "margin >= 1.0000"
            # alone where all 2048 windows of the condition had margin 1.0 and only 1229 were in the zone, the other 819
            # all wrong (2026-10-06): read as a rule, the threshold took in the uncertified windows.
            tied, inside = e.get("n_tied_at_threshold", 0), e.get("n_tied_inside_zone", 0)
            lines.append(f"  {name}: the {100 * e['coverage']:.0f}% most confident windows of this condition ({e['n_zone']} "
                         f"of {e['n_population']}, {what} >= {e['edge']}) are wrong at most {100 * a:g}% of "
                         "the time"
                         + (f"; {tied} windows share that {what} and only {inside} of them are inside the zone, so the "
                            f"zone is the window mask, not every window at or above the threshold" if tied > inside else ""))
        elif e["levels"]:
            lv = e["levels"][0]
            lines.append(f"  {name}: no zone certified; the smallest testable zone ({100 * lv['coverage']:.0f}% of the "
                         f"condition) held {lv['n_labelled_inside']} labels with {lv['n_wrong_inside']} wrong")
        else:                         # enough labels to be tested at delta, too few at delta split over the others
            lines.append(f"  {name}: no zone certified; {e['n_labelled']} labels, and certifying any zone at alpha {a:g} "
                         f"and delta {e['delta']:.3g} needs at least {e['min_labels_to_certify']}")
    if res["certified_share_of_map"] is not None:
        lines.append(f"together the certified windows are {100 * res['certified_share_of_map']:.1f}% of the map (mask "
                     f"{mask_path}); outside them nothing is certified")
    else:
        lines.append("no zone is certified in any condition, so nothing is certified")
    lines += [f"note: {b}" for b in bounds] + ([f"note: {res['population_note']}"] if "population_note" in res else [])
    if res["certified_share_of_map"] is not None:
        lines.append(f"note: {est.ISSUED_NOTE}")
    text = "\n".join(lines) + also_text
    print(class_wording(sample, args.sample, mask_path, _also_mask(mask_path), getattr(args, "also_accept", None))(text)
          + f"\nwrote {out}")
    return 0


# certify refuses an audit sample, so its help does not point to one
CERTIFY_ALSO_ACCEPT_HELP = ("JSON {map class: [reference classes also accepted]}, written before any label is read: a "
                            "second reading of the same labels, which needs a `reference_class` column beside `wrong`")


def add_certificate_parser(sub):
    ce = sub.add_parser("certificate", help="pack a certified zone to travel with the map: a STAC Item stating the "
                                            "guarantee, the zone as a GeoTIFF (.npy for a map with no grid), the "
                                            "labels, and digests binding them to the map")
    ce.add_argument("zone", help="the JSON certify wrote (random_zone.json)")
    ce.add_argument("--date", required=True, help="the date or period the map describes (YYYY-MM-DD or start/end)")
    ce.add_argument("--out", required=True, help="the directory to write the certificate bundle into")
    ce.add_argument("--scores", default=None, help="the map, if it has moved since sample read it")
    ce.add_argument("--confidence", default=None, help="a product's confidence band, if it has moved")
    ce.set_defaults(func=cmd_certificate)


def add_verify_parser(sub):
    ve = sub.add_parser("verify", help="check a certificate by building it again from the map, the labels and the "
                                       "recorded settings; and whether a point is in the zone")
    ve.add_argument("certificate", help="certificate.json, with its bundle beside it")
    ve.add_argument("--scores", default=None, help="your copy of the map; without it the map is read at the path "
                                                   "the certificate names, which shows only that the certificate is "
                                                   "consistent with that file")
    ve.add_argument("--confidence", default=None, help="a product's confidence band, if it has moved")
    ve.add_argument("--at", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="a point in the map's CRS: is it inside the certified zone?")
    ve.add_argument("--at-pixel", type=float, nargs=2, default=None, metavar=("ROW", "COL"),
                    help="a pixel of a map with no grid: is it inside the certified zone?")
    ve.add_argument("--out", default=None, help="JSON to write the checks to (optional)")
    ve.set_defaults(func=cmd_verify)


def add_certify_parser(sub):
    z = sub.add_parser("certify", help="which share of the map, from the most confident window down, is wrong at most "
                                        "alpha of the time, with a guarantee (needs a random sample; a sample drawn "
                                        "with --condition is certified per input condition)")
    z.add_argument("sample", help="the CSV `sample --design random` (or `sample --condition`) wrote, with its `wrong` "
                                  "column filled in")
    z.add_argument("--alpha", type=float, default=None,
                   help="the error rate the certified zone may not exceed, e.g. 0.05 (required, unless the sample fixed "
                        "it with sample --seed-from-map --alpha)")
    z.add_argument("--delta", type=float, default=None,
                   help=f"the probability the statement is allowed to be wrong (default {est.ZONE_DELTA}); for a sample "
                        "drawn with --condition, that any of the per-condition statements is, split over the conditions "
                        "tested")
    z.add_argument("--rule", choices=("prefix", "bonferroni", "target"), default=None,
                   help="prefix (default): fixed-sequence testing, valid on any map; it certifies little when the most "
                        "confident windows hold many errors. bonferroni: valid on any map; it can certify more in that "
                        "case. target: the default, and the only rule, for a sample drawn with --target")
    z.add_argument("--level-cut", choices=("standard", "ramp"), default=None,
                   help="which levels a budget tests. standard (default): a level once its zone expects "
                        "min_labels_to_certify labels. ramp: the same up to 3 times that budget, then a level only once "
                        "it expects 3 times as many, which halved the fall in certifying when labels are added on about "
                        "three maps in four but moved some falls to larger budgets (exp96 and its audit)")
    z.add_argument("--scores", default=None, help="the raster `sample` was run on, if it has moved")
    z.add_argument("--confidence", default=None, help="the confidence band `sample` was run on, if it has moved")
    z.add_argument("--nodata", type=float, default=None,
                   help="the no-data value `sample` was run with (default: the one its sidecar records; a different value "
                        "is refused); only needed for a sample written by 1.2.0")
    z.add_argument("--out", default=None, help="JSON to write (default: <sample>_zone.json; the window mask goes beside it as .npy)")
    z.add_argument("--also-accept", default=None, metavar="FILE", help=CERTIFY_ALSO_ACCEPT_HELP)
    z.set_defaults(func=cmd_certify)

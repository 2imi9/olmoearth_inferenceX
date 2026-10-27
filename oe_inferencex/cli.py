"""Command line: a prediction raster in, the review set and its explanation out; two rasters in, their difference out.

    oe-inferencex assess  scores.tif --out DIR [--logits] [--patch 4] [--nodata V] [--reference labels.tif]
                          [--budgets 0.01 0.05 0.10] [--order confidence|boundary_first]
                          [--condition layer.tif [--condition-names 0=clear 1=cloudy]]
    oe-inferencex from-embeddings --points points.csv --bbox W S E N --year 2025 --out scores.tif   precomputed
                                                            embeddings + labelled points -> class probabilities
    oe-inferencex boundary scores.tif --out band.tif --width-m 20 | --width-px 2   the class-edge band, as a
                                                            condition layer for sample
    oe-inferencex compare a.tif b.tif --out DIR [--patch 4] [--nodata V] [--labels labels.tif] [--groups ids.tif]
                          [--threshold T]
    oe-inferencex demo    [--out DIR] [--made-up]   a first run on the real sample map shipped with the package
    oe-inferencex sample  scores.tif --budget 300 --out to_label.csv [--design D] [--condition layer.tif]
    oe-inferencex sample  scores.tif --design audit --per-class 20 --out audit.csv   each class's confident zone
    oe-inferencex estimate to_label.csv [--per-class] [--also-accept pairs.json]   once the reviewer has filled `wrong`
    oe-inferencex estimate audit.csv --alpha 0.05 [--also-accept pairs.json]   once `reference_class` is filled
    oe-inferencex certify  to_label.csv --alpha 0.05 [--also-accept pairs.json]   from a random sample, or one drawn
                                                            with --condition
    oe-inferencex grade points.csv --stratum S --reference R --map M [--map M2] (--weight W | --strata sizes.csv)
                          [--alternate A] [--pure 1=11 ...]   an agency's stratified sample -> accuracies with
                                                            intervals valid at any sample size
    oe-inferencex mcp                                       a local MCP server on stdio, for an agent (the mcp extra)
    oe-inferencex from-olmoearth DS --out DIR [--conditions]  an OlmoEarth run's rslearn dataset -> the rasters
                                                            assess reads, and the command that reads them

Inputs are GeoTIFFs (any rasterio-readable raster) or .npy arrays: (H, W) for a binary map, (C, H, W) for per-class
scores or, for `compare`, an integer class map. `compare` also takes two continuous maps (a regression output) when the
cut-off is named with --threshold; without it a map outside [0, 1] is refused rather than cut at 0.5. No-data comes
from the raster's own value, NaN, or --nodata. Outputs are
plain files the caller reads back: JSON summaries (assess.summary / compare's dict), CSVs of the review windows with
pixel and map coordinates, and rasters on the window grid (patch x patch pixels per window) when the input was one.
Nothing here narrates; the JSON is the evidence (docs/method/agent_integration.md in the research record).

The confidence ranking compares every window with every other. A model run on an input combination it was not trained
on can be confidently wrong there: on PASTIS a probe trained on radar plus optical and run on radar alone, as under
cloud, was sure and wrong, while a probe trained on radar alone ranked its errors normally (exp88). Even a model trained
on each input can be wrong more often under one input than under another. Without --condition, assess ranks all
windows together. --condition takes one integer band on the map's grid, each pixel's input condition (a cloud flag,
the modalities present, a sensor id). assess then also ranks each condition on its own, so that windows are compared
only with windows read from the same inputs. sample splits the labels equally across the conditions (with --design
random it only records them) and writes each window's condition to its sidecar, from which estimate gives each
condition's rate and certify each condition's zone. Without it, the JSON outputs carry `scope`, what a whole-map
result does not show.
"""
import argparse
import csv
from types import SimpleNamespace
import json
import math
import os
import sys

import numpy as np

from oe_inferencex.review.assess import (RULE_TEXT, _boundary_valid, _pool, _pool_valid, _pooled_argmax, assess_classmap,
                                         assess_prediction, pool_condition, summary)
from oe_inferencex import estimate as est
from oe_inferencex.compare import compare_inferences
from oe_inferencex.review.explain import explain_review_set
from oe_inferencex.common.signals import boundary_indicator
from oe_inferencex.common.condition import _condition_args, _condition_names, _read_condition
from oe_inferencex.common.labels import (_also_accept, _also_accept_rows, _also_words, _check_sidecar, _csv_rows,
                                         _labelled_sample, _population_note, _reference_column, _rounding_tolerance,
                                         _sample_map_inputs, _sidecar_design, _unused_note)
from oe_inferencex.common.maps import PROB_TOLERANCE, _check_scores, decisions
from oe_inferencex.common.output import (_check_out_dir, _check_out_file, _check_patch, _json_nodata, _nodata_arg,
                                         _num_arg, _pct, _readable_path, _refuse_overwriting_inputs, _same_file, _stamp,
                                         _where)
from oe_inferencex.common.product import (MAX_CLASS_ID, NO_RANGE_NOTE, PRODUCT_SIGNAL, _product_args,
                                          _product_parser_args, _read_product)
from oe_inferencex.common.raster import (_band_count, _gdal_masked, _mask_bands, _mask_hint, _read_masked, _same_grid,
                                         affine_apply, pool_valid, read_raster, window_coords, write_raster)
from oe_inferencex.common.wording import (_CLASS_WORDS, _PATH_KEYS, _RULE_WORDS, class_names, class_words,
                                          class_wording, result_class_names)
from oe_inferencex.review.windows import _map_windows, _two_map_windows
from oe_inferencex.review.command import _budget_tag, add_assess_parser, add_boundary_parser, cmd_assess, cmd_boundary
from oe_inferencex.readers.command import (_skipped_text, add_from_embeddings_parser, add_from_olmoearth_parser,
                                           cmd_from_embeddings, cmd_from_olmoearth)


def _unrecorded_note(n_split, n_no_code):
    """What makes a window 'unrecorded', counted; None when no window is."""
    if n_split and n_no_code:
        return (f"{n_split} windows are tied between condition values and {n_no_code} carry none; both count as "
                "the condition 'unrecorded'")
    if n_split:
        return f"{n_split} windows are tied between condition values; they count as the condition 'unrecorded'"
    if n_no_code:
        return f"{n_no_code} windows carry no condition value; they count as the condition 'unrecorded'"
    return None


def _certify_need_note(names, n_labelled, alpha=0.05, delta=est.ZONE_DELTA):
    """How many labels a condition needs before `certify` can say anything about it, at an illustrative alpha: delta
    is split over the conditions holding at least min_labels_to_certify(alpha, delta) labels (certify_by_condition)."""
    b1 = est.min_labels_to_certify(alpha, delta)
    L = sum(int(n) >= b1 for n in n_labelled)
    short = ", ".join(f"{name} gets {int(n)}" for name, n in zip(names, n_labelled) if int(n) < b1)
    if L == 0:
        return (f"to certify at alpha {alpha:g} (delta {delta:g}), a condition needs at least {b1} labels, and none has "
                f"that many: {short}")
    bL = est.min_labels_to_certify(alpha, delta / L)
    return (f"to certify at alpha {alpha:g} (delta {delta:g} split across the {L} condition{'s' if L > 1 else ''} with "
            f"at least {b1} labels), each needs at least {bL} labels" + (f"; {short}" if short else ""))


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
    (scale.py)."""
    from .scale import scale_profile
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
    every sample size and the normal-theory one beside it, nominal (agency.py, stratified.py)."""
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
    JSON carrying the design so `estimate` can give the rate the design earns. With --condition, the input condition
    of every window is fixed here and recorded in the sidecar; `estimate` and `certify` read it from there."""
    # read with getattr: demo.py and the tests build the argument Namespace by hand, without these attributes
    out_csv = getattr(args, "out", None)
    if out_csv:
        # the CSV and its sidecar never write over a raster the sample reads (the bug hunt of 25 October 2026:
        # `--out scores.tif` replaced the map with the CSV, exit 0); --extend's own CSV may be named, as it is extended
        # the --extend sample's CSV and sidecar too, unless --out names that CSV itself (the review of 1.17.2: --out
        # naming its sidecar wrote the CSV over the labelled sample's design)
        ext = getattr(args, "extend", None)
        ext_files = ([ext, ext[:-4] + ".json" if ext.endswith(".csv") else ext + ".json"]
                     if ext and not _same_file(out_csv, ext) else [])
        _refuse_overwriting_inputs("sample", [out_csv, out_csv[:-4] + ".json" if out_csv.endswith(".csv") else
                                              out_csv + ".json"],
                                   [getattr(args, k, None) for k in ("scores", "other", "confidence", "condition")]
                                   + ext_files)
    if getattr(args, "seed_from_map", False) and (getattr(args, "other", None) or
                                                   getattr(args, "design", None) == "audit"):
        raise SystemExit("sample: --seed-from-map draws one map's sample from that map's own digest; it does not "
                         "apply to --other or --design audit")
    if getattr(args, "target", None) is not None and (getattr(args, "other", None) or
                                                      getattr(args, "design", None) == "audit"):
        raise SystemExit("sample: --target fixes the zone the target rule will certify, which needs a simple random "
                         "sample of the whole map: --design random, without --other")
    if getattr(args, "other", None) or getattr(args, "design", None) == "audit":
        # one map's options, read by neither route: each passed unread, and --extend with --out naming the sequential
        # CSV replaced its 100 labels with an unlabelled pairs sample (the bug hunt of 2026-10-25)
        unread = [flag for flag, key in (("--extend", "extend"), ("--anchor", "anchor"), ("--alpha", "alpha"),
                                         ("--delta", "delta"), ("--rule", "rule"), ("--level-cut", "level_cut"),
                                         ("--min-per-class", "min_per_class"))
                  if getattr(args, key, None) is not None]
        if getattr(args, "other", None) and getattr(args, "condition_names", None):
            unread.append("--condition-names")
        if getattr(args, "other", None) and getattr(args, "logits", False):
            unread.append("--logits")          # the two maps are read as classes (the review of 1.17.2)
        if unread:
            route = "--other" if getattr(args, "other", None) else "--design audit"
            raise SystemExit(f"sample {route} takes none of {', '.join(unread)}: --extend, --anchor, --alpha and --delta "
                             "belong to --design sequential, --rule and --level-cut to --seed-from-map, --min-per-class "
                             "to --design class and --condition-names to --condition; --other reads both maps' classes, "
                             "so it reads no --logits")
    audit_opts = [getattr(args, k, None) for k in ("per_class", "zone_share", "zone_cut")]
    if getattr(args, "design", None) != "audit" and any(v is not None for v in audit_opts):
        raise SystemExit("sample: --per-class, --zone-share and --zone-cut belong to --design audit")
    if getattr(args, "design", None) == "audit":
        if getattr(args, "budget", None) is not None:
            raise SystemExit("sample: the audit's labels are --per-class N per predicted class; drop --budget")
        return _sample_class_audit(args)
    # --budget is checked here rather than by the parser, since the audit design takes --per-class instead; before the
    # --other route, which needs it too
    if getattr(args, "budget", None) is None:
        raise SystemExit("sample: --budget is required: the number of windows to label (exp78 measured 300; about 400 "
                         "for a random sample whose labels also serve certify at alpha 0.05)")
    if getattr(args, "other", None):
        if getattr(args, "confidence", None) or getattr(args, "confidence_range", None):
            raise SystemExit("sample: --other draws among the windows where two class maps differ, which needs no "
                             "confidence; leave out --confidence")
        return _sample_disagreement(args)
    _check_patch(args.patch, "sample")
    conf_path, conf_range = _product_args(args, "sample")
    if getattr(args, "threshold", None) is not None:
        raise SystemExit("sample: --threshold reads a second map's cut-off and is used only with --other")
    by_class = getattr(args, "design", None) == "class"
    if by_class:
        # Stratified by the map's own class (Olofsson et al. 2014): the strata are the map classes, drawn at random
        # within each, which the condition design's estimators, exact intervals and per-stratum zones already read;
        # --condition-names names the classes. No layer: the strata come from the map, so its digest binds them.
        if getattr(args, "condition", None):
            raise SystemExit("sample: --design class takes its strata from the map's own classes; drop --condition")
        cond_path, cond_names = None, _condition_names(getattr(args, "condition_names", None), "sample")
    else:
        if getattr(args, "min_per_class", None) is not None:
            raise SystemExit("sample: --min-per-class belongs to --design class")
        cond_path, cond_names = _condition_args(args, "sample")
    class_floor = None
    if by_class:
        class_floor = est.CLASS_FLOOR if getattr(args, "min_per_class", None) is None else int(args.min_per_class)
        if class_floor < 1:
            raise SystemExit(f"sample: --min-per-class is at least 1 (each class's user's accuracy needs labels), got "
                             f"{class_floor}")
    # None resolves to the condition design when a layer is given, to the confidence design otherwise; a product's
    # confidence band is no top-1 probability, which the confidence design allocates from, so it resolves to random
    prod_notes = []
    if conf_path and getattr(args, "design", None) == "confidence":
        raise SystemExit("sample: the confidence design allocates labels from the model's top-1 probability, and a "
                         "confidence band is not one; use --design random (estimate and certify read it), --design "
                         "proportional or, with --condition, --design condition")
    design = getattr(args, "design", None) or ("condition" if cond_path else "random" if conf_path else "confidence")
    if conf_path and not getattr(args, "design", None) and not cond_path:
        prod_notes.append("random design: with --confidence the default confidence design does not apply, since the "
                          "band is not a top-1 probability; estimate and certify both read a random sample")
    if cond_path and design in ("confidence", "proportional"):
        raise SystemExit(f"sample: --design {design} does not take --condition: {est.CONFIDENCE_REFUSAL}")
    if cond_path and design == "tiles":
        raise SystemExit("sample: --design tiles does not take --condition: a tile can span conditions, and the tile "
                         "interval is not graded per condition.")
    if design == "condition" and not cond_path:
        raise SystemExit("sample: --design condition needs --condition, a raster of each pixel's input condition (a "
                         "cloud flag, the modalities present, a sensor id)")
    seed_from_map = bool(getattr(args, "seed_from_map", False))
    if seed_from_map and args.seed != 0:
        raise SystemExit("sample: give --seed or --seed-from-map, not both")
    if seed_from_map and design not in ("random", "sequential", "condition", "class"):
        raise SystemExit(f"sample: --seed-from-map draws a sample for a certificate, which certify gives from the "
                         f"random, sequential, condition or class design; not from --design {design}")
    target = getattr(args, "target", None)
    if target is not None:
        # The target rule's zone, fixed before any label (1.16.0): certify reads it from the sidecar and refuses another
        # rule on these labels, as a sequential sample's anchor is fixed here.
        if design != "random" or cond_path:
            raise SystemExit("sample: --target fixes the zone the target rule will certify, which needs a simple "
                             "random sample of the whole map: --design random, without --condition")
        if not any(abs(target - g) < 1e-9 for g in est.ZONE_GRID):
            raise SystemExit(f"sample: --target is a level of the zone grid, {est.ZONE_GRID[0]:g} to 1 in steps of "
                             f"{est.ZONE_GRID[1] - est.ZONE_GRID[0]:g}, got {target:g}")
    anchor, extend = getattr(args, "anchor", None), getattr(args, "extend", None)
    a_alpha, a_delta = getattr(args, "alpha", None), getattr(args, "delta", None)
    c_rule, c_cut = getattr(args, "rule", None), getattr(args, "level_cut", None)
    if not seed_from_map and (c_rule is not None or c_cut is not None):
        raise SystemExit("sample: --rule and --level-cut fix certify's options before any label, for a certificate; "
                         "they go with --seed-from-map")
    if design != "sequential" and (any(v is not None for v in (anchor, extend)) or
                                   (not seed_from_map and any(v is not None for v in (a_alpha, a_delta)))):
        raise SystemExit("sample: --anchor, --alpha, --delta and --extend belong to --design sequential (--alpha and "
                         "--delta also to --seed-from-map, which fixes certify's error levels)")
    if design == "sequential" and cond_path:
        raise SystemExit("sample: --design sequential does not take --condition: certify reads it as one random order "
                         "of the whole map")
    old_side = old_wrong = old_ref = old_rows = anchor_rule = None
    if extend is not None:
        if any(v is not None for v in (anchor, a_alpha, a_delta)):
            raise SystemExit(f"sample --extend: the anchor was fixed when {extend} was drawn and cannot change after "
                             "labelling began; drop --anchor, --alpha and --delta")
        old_side, old_wrong, old_ref, old_rows = _sequential_to_extend(extend, args)
        anchor, anchor_rule = old_side["anchor"], old_side.get("anchor_rule")
    elif design == "sequential":
        # The anchor, the smallest zone certify will test, is fixed here, before any label: certify reads it from the
        # sidecar. Computed from the alpha certify will be asked about when it is not given; certify at another alpha
        # keeps this anchor (review of 8 October 2026: an anchor recomputed from each certify run's alpha was not
        # fixed before the labels, as its guarantee needs).
        # with --seed-from-map, --alpha is certify's, fixed now, and a declared --anchor may go with it
        if (anchor is None) == (a_alpha is None) and not (seed_from_map and a_alpha is not None):
            raise SystemExit("sample: a sequential sample fixes now, before any label, the smallest zone certify will "
                             "test: give --alpha (the error rate certify will be asked about; the anchor is then the "
                             "zone this budget can certify if none of its labels is wrong, at most a quarter of the "
                             "map) or --anchor (a share of the map), not both")
        if anchor is not None and a_delta is not None and not seed_from_map:
            raise SystemExit("sample: --delta goes with --alpha, to compute the anchor; with --anchor it is not used")
        if anchor is None:
            d = est.ZONE_DELTA if a_delta is None else a_delta
            if not (0 < a_alpha < 1 and 0 < d < 1):
                raise SystemExit(f"sample: --alpha and --delta must be in (0, 1), got {a_alpha:g} and {d:g}")
            from oe_inferencex.certify import sequential as sq
            anchor = sq.anchor_coverage(args.budget, a_alpha, d)
            anchor_rule = {"alpha": a_alpha, "delta": d, "first_budget": int(args.budget),
                           "rule": "the smallest grid zone expecting min_labels_sequential labels at the first budget, "
                                   f"at most {sq.ANCHOR_CAP:g}"}
        else:
            anchor_rule = "declared"
    if anchor is not None and not est.ZONE_GRID[0] <= anchor <= 1:
        raise SystemExit(f"sample: --anchor is a share of the map between {est.ZONE_GRID[0]:g} and 1, got {anchor:g}")
    plan = _certify_plan(design, target, a_alpha, a_delta, c_rule, c_cut, seed_from_map, old_side)
    seed_rule = None
    if seed_from_map or (old_side is not None and old_side.get("seed_rule") == "from_map"):
        # The seed taken from the map's digest together with every setting that fixes the draw (1.17.0): anyone holding
        # the map derives it again, so a certificate can show the seed was not picked by hand and that no setting was
        # changed after labelling (certificate.verify). An extension keeps the rule when its seed is the same.
        from oe_inferencex.certify.certificate import draw_seed
        seed_rule = draw_seed(args.scores, args.nodata, args.logits, conf_path, conf_range, cond_path, design,
                              args.budget, args.patch, target, anchor, plan, class_floor)
        if seed_from_map:
            if old_side is not None and int(old_side["seed"]) != seed_rule["seed"]:
                # "extend it with that --seed" sent the user to a run that dropped the seed rule (the bug hunt of
                # 2026-10-25): the settings that enter the seed are what differ
                raise SystemExit(f"sample --extend: {extend} was drawn with the seed {old_side['seed']}, not the seed this "
                                 "map and its settings give; extend it with the options it was drawn with, as the "
                                 "command sample printed for it gives them." + _mask_hint(args.scores, conf_path))
            args.seed = seed_rule["seed"]
        elif seed_rule["seed"] != int(args.seed):
            seed_rule = None
    out, geo, p1_w, pooled, more, prod_info = _sample_map(args, conf_path, conf_range, cond_path)
    prod_notes += more
    if by_class:                                       # each window's stratum: the class the map gives it
        a_ = out["arrays"]
        pooled = {"grid": np.where(a_["valid"], a_["pooled_argmax"], -1).astype(np.int64), "n_split": 0,
                  "n_no_code": 0}
        if cond_names is None:
            cond_names = {int(v): f"class {int(v)}" for v in np.unique(pooled["grid"][pooled["grid"] >= 0])}
    arr = out["arrays"]
    margin, valid_w, klass = arr["confidence"], arr["valid"], arr["pooled_argmax"]
    hw, ww = margin.shape
    tiles = None
    if design == "tiles":
        t = max(1, args.tile)
        tiles = (np.arange(hw)[:, None] // t) * ((ww + t - 1) // t) + (np.arange(ww)[None, :] // t)
    try:
        sample = est.sample_for_estimation(margin, args.budget, design="condition" if by_class else design, p1=p1_w,
                                           tiles=tiles, per_tile=args.per_tile, valid=valid_w, seed=args.seed,
                                           condition=None if pooled is None else pooled["grid"].ravel(),
                                           condition_names=cond_names, stable=seed_rule is not None,
                                           class_floor=class_floor)
    except ValueError as exc:
        raise SystemExit(f"sample: {exc}")
    idx = sample["indices"]
    if old_side is not None:
        prev = np.asarray(old_side["indices"], int)
        if sample["n_population"] != old_side.get("n_population") or not np.array_equal(idx[:prev.size], prev):
            raise SystemExit(f"sample --extend: {extend} was not drawn from this map with this seed: its windows are not "
                             "the start of the order this map and seed give. Pass the same scores, --patch, --nodata "
                             "and --seed it was drawn with." + _mask_hint(args.scores, conf_path))
        # the windows already labelled must show the confidence and class they were labelled with: on a 1.17.1 sample
        # whose raster's mask 1.17.2 reads, the extension kept the labels and rewrote map_class under them, and certify
        # then accepted what it refused unextended (the third review of 1.17.2)
        k = _kept_rows_mismatch(old_rows, margin.ravel()[prev], klass.ravel()[prev])
        if k is not None:
            raise SystemExit(f"sample --extend: the windows of {extend} labelled so far no longer show the confidence and "
                             f"class the CSV records (row {k + 2}); this is not the map they were labelled on."
                             + _mask_hint(args.scores, conf_path))
        sample["first_budget"] = old_side["first_budget"]
    cond = sample.get("condition")
    rows, cols = np.divmod(idx, ww)
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    if os.path.isdir(args.out):
        raise SystemExit(f"--out {args.out} is a directory; name the CSV to write, for example {os.path.join(args.out, 'to_label.csv')}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # UTF-8, as every reader reads it: in the locale's encoding (cp1252 on Windows) a condition named bewölkt was
    # written as one byte, and estimate refused the file (the bug hunt of 2026-10-25)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # map_class: the class `estimate` and `certify` grade at the window (the majority of its pixels' classes),
        # shown so the reviewer judges the same thing the tool grades; a mixed window is where a reviewer looking
        # at the pixels could otherwise read the map's class differently (release check of 24 September)
        # with --condition, the condition's name follows `stratum` (under the condition design, the stratum is the
        # condition's index); map_class and wrong stay the last two columns. A `reference_class` column the reviewer
        # added to a sequential sample (estimate --per-class and --also-accept read it) is kept by --extend, after
        # `wrong`, blank on the new rows. Until 13 October 2026 --extend wrote the fixed columns only and dropped it
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "stratum"]
                   + ([] if cond is None else ["condition"]) + ["confidence", "map_class", "wrong"]
                   + ([] if old_ref is None else ["reference_class"]))
        strata = sample.get("strata")
        pos = {int(g): i for i, g in enumerate(sample.get("strata_of_population", []))}
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), None if strata is None else int(strata[pos[int(i)]])]
                       + ([] if cond is None else [cond["names"][int(sample["condition_grid"][int(i)])]])
                       + [float(margin[r, c]), int(klass[r, c]),
                          old_wrong[k] if old_wrong is not None and k < len(old_wrong) else ""]
                       + ([] if old_ref is None else [old_ref[k] if k < len(old_ref) else ""]))
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    if design == "sequential":
        side["anchor"], side["anchor_rule"] = float(anchor), anchor_rule     # fixed now; certify reads them
    if target is not None:
        side["target"] = {"coverage": float(target), "weight": est.TARGET_WEIGHT}   # fixed now; certify reads it
    if seed_rule is not None:
        side.update({"seed_rule": "from_map", "map_sha256": seed_rule["map_sha256"],
                     "seed_settings": seed_rule["settings"]})
    if plan is not None:
        side["certify"] = plan                         # fixed now, with the draw; certify reads and keeps to it
    if cond is not None:
        # the design's source of truth for `estimate` and `certify`, which never read the raster again; its digest
        # binds the layer to the sample, so a certificate can show the layer was not swapped after labelling
        from oe_inferencex.certify.certificate import names_given, raster_digest
        side["condition"] = {"source": "map class" if by_class else _where(cond_path),
                             "rule": CLASS_RULE_TEXT if by_class else RULE_TEXT, "values": cond["values"],
                             "names": cond["names"], "sizes": cond["sizes"], "n_labelled": cond["n_labelled"],
                             "allocation_rule": cond["allocation_rule"], "n_windows_split": pooled["n_split"],
                             "n_windows_no_code": pooled["n_no_code"],
                             "sha256": (None if by_class else seed_rule["settings"]["condition_sha256"]
                                        if seed_rule is not None else raster_digest(cond_path)),
                             "names_given": names_given(cond_names)}
        if by_class:
            side["condition"]["by"] = "map class"      # the class design: certify, estimate and verify read it so
            side["condition"]["min_per_class"] = class_floor
    side.update(_sidecar_map_fields(args, geo, [int(hw), int(ww)], out, conf_path, conf_range, prod_info))
    side["how_to_label"] = ("open each window, set wrong=1 if the map's class there (the `map_class` column: the "
                            "majority of the window's pixels) is not what is on the ground, else 0; then: "
                            "oe-inferencex estimate " + os.path.basename(args.out))
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(_stamp(side, "sample"), f, indent=1)
    what, notes = f"{design} design", []
    if cond is not None:
        counts = ", ".join(f"{name} {int(n)}" for name, n in zip(cond["names"], cond["n_labelled"]))
        what = (f"class design: {counts}" if by_class else f"condition design: {counts}" if design == "condition"
                else f"random design; by chance: {counts}")
        K = len(cond["names"])
        if by_class and K >= 2:
            notes.append(f"labels are allocated by map class: each class at least {cond['allocation_rule'].split()[-3]} "
                         "(or all its windows), the rest in proportion to its size (Olofsson et al. 2014); estimate "
                         "--per-class gives each class's user's and producer's accuracy and area with exact intervals")
            notes.append(est.CLASS_BLIND_NOTE)
        elif design == "condition" and K >= 2:
            notes.append(f"labels are split equally across the {K} input conditions so that each gets its own error "
                         "rate; the whole-map rate weights each condition by its share of the map")
        notes.append(_unrecorded_note(pooled["n_split"], pooled["n_no_code"]))
        if K >= 2:
            notes.append(_certify_need_note(cond["names"], cond["n_labelled"],
                                            *((plan["alpha"], plan["delta"]) if plan else ())))
    if design == "sequential":
        kept = 0 if old_wrong is None else sum(str(v).strip() != "" for v in old_wrong)
        notes.insert(0, ("label from the top row down, in order; certify may be run after any number of labels and "
                         "holds at every look. To add windows later: " + _extend_command(args, seed_rule))
                     + (f"; the {kept} labels already given are kept" if kept else ""))
    if target is not None:
        notes.insert(0, f"the target, the {100 * target:g}% most confident zone, is fixed now: certify will test it "
                        f"with {est.TARGET_WEIGHT:g} of delta (all of it if no smaller zone is tested) and the smaller "
                        "zones with the rest (the target rule). Choose alpha and delta now too, and certify once on "
                        "the full labels")
    print(f"{len(idx)} windows to label of {sample['n_population']} valid ({what}); wrote {args.out} and its .json. "
          f"{FILL_INSTRUCTION} oe-inferencex estimate {args.out}"
          + "".join(f"\nwarning: {w}" for w in out.get("warnings", []))
          + (f"\nnote: {sample['note']}" if "note" in sample else "")
          + "".join(f"\nnote: {class_wording(side, args.out)(n)}" for n in prod_notes + notes if n))
    return 0


def _certify_plan(design, target, alpha, delta, rule, cut, seed_from_map, old_side):
    """certify's options fixed with a sample drawn for a certificate (--seed-from-map): alpha, delta, the rule and the
    level cut, which enter the seed, so that a certificate shows they were chosen before any label, not after (the
    final review of 24 October 2026: one sample certified nothing at alpha 0.05 and 0.10 and 70% at 0.15, and every
    one of those certificates verified). An extension keeps its sample's plan. None without --seed-from-map."""
    if old_side is not None:
        if any(v is not None for v in (rule, cut)):
            raise SystemExit("sample --extend: certify's options were fixed when the sample was drawn and cannot "
                             "change after labelling began; drop --rule and --level-cut")
        if seed_from_map and old_side.get("seed_rule") == "from_map" and not old_side.get("certify"):
            raise SystemExit("sample --extend: this sample's seed was taken from the map by a release before 1.17.0, "
                             "which did not fix certify's options with it, so it cannot carry a certificate; extend it "
                             f"with --seed {old_side.get('seed')} (no certificate), or draw a new sample with "
                             "--seed-from-map --alpha")
        return old_side.get("certify")
    if not seed_from_map:
        return None
    if alpha is None:
        raise SystemExit("sample --seed-from-map: give --alpha, the error rate certify will test (and --delta, --rule or "
                         "--level-cut if not the defaults): a sample drawn for a certificate fixes them now, before any "
                         "label, and they enter the seed")
    delta = est.ZONE_DELTA if delta is None else float(delta)
    if not (1e-4 <= alpha <= 0.5 and 1e-6 <= delta <= 0.5):
        raise SystemExit(f"sample: a certificate's alpha lies in [0.0001, 0.5] and delta in [0.000001, 0.5], got "
                         f"{alpha:g} and {delta:g}")
    if target is not None and rule not in (None, "target"):
        raise SystemExit("sample: a sample with --target is certified by the target rule; drop --rule")
    if design == "sequential" and (rule is not None or cut is not None):
        raise SystemExit("sample: a sequential sample is certified by the sequential rule, with no level cut; drop "
                         "--rule and --level-cut")
    rule = "target" if target is not None else "sequential" if design == "sequential" else (rule or "prefix")
    return {"alpha": float(alpha), "delta": delta, "rule": rule,
            "level_cut": None if design == "sequential" else (cut or "standard")}


def _sample_map(args, conf_path, conf_range, cond_path=None):
    """The map `sample` draws from, read and assessed as every design reads it: (assess's output, geo, the top-1
    probability per window or None for a product, the condition layer pooled to windows or None, the product's notes,
    the product's population info or None). The audit design reads the same windows, classes and confidence."""
    prod_notes, prod_info = [], None
    if conf_path:
        hard, conf_px, valid, geo, n_classes, prod_notes, prod_info = _read_product(args.scores, conf_path, args.nodata,
                                                                                    conf_range, "sample")
        shape = hard.shape
    else:
        scores, valid, geo = read_raster(args.scores, args.nodata)
        _check_scores(scores, valid, args.logits, args.scores)
        shape = scores.shape[-2:]
    pooled = None
    if cond_path:
        layer = _read_condition(cond_path, shape, geo, "sample")
        try:
            pooled = pool_condition(layer, args.patch, predicted=valid)     # the window grid assess gives
        except ValueError as exc:
            raise SystemExit(f"sample: {exc}") from None
    if conf_path:
        try:
            out = assess_classmap(hard, conf_px, n_classes, patch=args.patch, nodata_mask=~valid, signal=PRODUCT_SIGNAL)
        except ValueError as exc:                                 # a named refusal, not a traceback
            raise SystemExit(f"sample: {exc}") from None
        p1_w = None
    else:
        # a named refusal, as on the product branch: a fully clouded logit map or a --patch larger than the map was
        # a ValueError traceback here, where assess refused the same input in one line (2026-10-06)
        try:
            out = assess_prediction(scores, is_logit=args.logits, patch=args.patch, nodata_mask=~valid)
        except ValueError as exc:
            raise SystemExit(f"sample: {exc}") from None
        p1 = np.where(valid, _top1(scores, args.logits), np.nan)
        p1_w = _pool_valid(p1, args.patch) if not valid.all() else _pool(p1, args.patch)
    return out, geo, p1_w, pooled, list(prod_notes), prod_info


def _sidecar_map_fields(args, geo, grid, out, conf_path, conf_range, prod_info):
    """The map's fields of a sample's sidecar, whatever the design: where the map is and how it was read (estimate
    --per-class and certify recompute it from them), its window grid, and what a reviewer needs to find a window."""
    side = {}
    if conf_path:   # estimate --per-class and certify rebuild the map from both layers, with the same range
        side.update({"classes": _where(args.scores), "confidence": _where(conf_path),
                     "confidence_range": None if conf_range is None else [float(v) for v in conf_range],
                     "product_population": prod_info})
    side.update({"scores": _where(args.scores), "logits": args.logits, "patch": args.patch,
                 "nodata": _json_nodata(args.nodata),     # estimate --per-class and certify recompute the map with it
                 "grid": grid, "csv": os.path.abspath(args.out),
                 # what a reviewer with a GIS needs to find a window: the CRS, the pixel size, and the window's
                 # footprint in ground units; x and y in the CSV are window centres in that CRS
                 "crs": None if geo is None else str(geo["crs"]),
                 "transform": None if geo is None else list(geo["transform"])[:6],
                 "pixel_size": None if geo is None else [abs(geo["transform"].a), abs(geo["transform"].e)],
                 "window_size_ground_units": None if geo is None else [abs(geo["transform"].a) * args.patch, abs(geo["transform"].e) * args.patch],
                 "xy_are": "window centres in the raster's CRS" if geo is not None else "absent: the input had no georeference",
                 "warnings": list(out.get("warnings", []))})
    return side


def _extend_command(args, seed_rule):
    """The command that extends this sequential sample, the map read as this run read it (the extension takes the
    certify plan from the sidecar). It named the map and the seed alone, so for a product the band was dropped, and
    with --seed-from-map the seed, which the band enters, no longer matched (the bug hunt of 2026-10-25)."""
    import shlex
    cmd = ["oe-inferencex", "sample", args.scores]
    if getattr(args, "confidence", None):
        cmd += ["--confidence", args.confidence]
    if getattr(args, "confidence_range", None):
        cmd += ["--confidence-range", *[_num_arg(v) for v in args.confidence_range]]
    if args.logits:
        cmd.append("--logits")
    if args.nodata is not None:
        cmd.append(_nodata_arg(args.nodata))
    if args.patch != 4:
        cmd += ["--patch", str(args.patch)]
    cmd += ["--design", "sequential"] + (["--seed-from-map"] if seed_rule is not None else ["--seed", str(args.seed)])
    # with --out, which sample requires: the command printed until 1.17.2 lacked it and was refused (the review)
    out = shlex.quote(args.out)
    return " ".join(shlex.quote(c) for c in cmd) + f" --budget <more> --extend {out} --out {out}"


def _kept_rows_mismatch(rows, conf_at, class_at):
    """The first of a sample's rows whose recorded confidence or class the map no longer gives, or None; the columns
    and the tolerance are _map_windows's."""
    for k, r in enumerate(rows or []):
        try:
            if abs(float(conf_at[k]) - float(r["confidence"])) > _rounding_tolerance(r["confidence"]) * (1 + 1e-9) + 1e-12:
                return k
        except (KeyError, ValueError, TypeError):
            pass
        try:
            if int(float(r["map_class"])) != int(class_at[k]):
                return k
        except (KeyError, ValueError, TypeError, OverflowError):
            pass
    return None


def _reading_text(side):
    """The options a sample's sidecar says the map was read with, as a command gives them."""
    import shlex
    flags = []
    if side.get("confidence"):
        flags += ["--confidence", side["confidence"]]
    if side.get("confidence_range") is not None:
        flags += ["--confidence-range", *[_num_arg(v) for v in side["confidence_range"]]]
    if side.get("logits"):
        flags.append("--logits")
    if side.get("nodata") is not None:
        flags.append(_nodata_arg(side["nodata"]))
    return " ".join(shlex.quote(f) for f in flags) or "no --confidence, --logits or --nodata"


def _sequential_to_extend(path, args):
    """The sidecar and the `wrong` column of a sequential sample that `sample --extend` lengthens, checked: it must be a
    sequential sample of this map drawn with this seed, shorter than the new budget; its anchor, fixed when it was
    drawn, is kept. Returns (sidecar, wrong values as written, one per row, and the reference_class values as written
    when the CSV has that column, else None)."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    if not os.path.exists(path) or not os.path.exists(side_path):
        raise SystemExit(f"sample --extend: {path} and its .json sidecar are needed")
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path)
    if side.get("design") != "sequential" or side.get("anchor") is None:
        raise SystemExit(f"sample --extend: {path} was drawn with the {side.get('design')!r} design; only a sequential "
                         "sample can be extended, since only its order and anchor were fixed before labelling")
    if not getattr(args, "seed_from_map", False) and int(side.get("seed", -1)) != int(args.seed):
        # with --seed-from-map the seed is derived later, once the anchor is known, and compared there
        raise SystemExit(f"sample --extend: {path} was drawn with --seed {side.get('seed')}; pass the same seed")
    if _where(args.scores) != side.get("scores") or int(args.patch) != int(side.get("patch", -1)):
        raise SystemExit(f"sample --extend: {path} was drawn from {side.get('scores')} at --patch {side.get('patch')}; "
                         "extend it from the same map and patch")
    # The map read as it was: a product's band dropped, or --logits added, passed, and the kept labels were paired with
    # other confidences (a band's 45.75 became 1.0, every window tied for certify); with --seed-from-map the seed
    # rule was lost (the bug hunt of 2026-10-25)
    conf = getattr(args, "confidence", None)
    rng = getattr(args, "confidence_range", None)
    differ = [flag for flag, now, then in (
        ("--logits", bool(args.logits), bool(side.get("logits", False))),
        ("--nodata", _json_nodata(args.nodata), side.get("nodata")),
        ("--confidence", None if not conf else _where(conf), side.get("confidence")),
        ("--confidence-range", None if rng is None else [float(v) for v in rng], side.get("confidence_range")))
              if now != then]
    if differ:
        raise SystemExit(f"sample --extend: {path} was drawn reading the map with {_reading_text(side)}; extend it "
                         f"reading it the same way ({', '.join(differ)} differ)")
    n_old = len(side["indices"])
    if args.budget <= n_old:
        raise SystemExit(f"sample --extend: {path} already holds {n_old} windows; --budget is the new total, larger "
                         "than that")
    rows = _csv_rows(path, "sample --extend")
    try:
        idx = [int(float(r["index"])) for r in rows]
    except (KeyError, ValueError, OverflowError):
        raise SystemExit(f"sample --extend: {path} lacks the `index` column `sample` wrote") from None
    if idx != [int(i) for i in side["indices"]]:
        raise SystemExit(f"sample --extend: the rows of {path} do not match its sidecar; extend the file `sample` wrote")
    return (side, [r.get("wrong", "") for r in rows],
            [r.get("reference_class", "") for r in rows] if rows and "reference_class" in rows[0] else None, rows)


CLASS_RULE_TEXT = ("A window's stratum is the class the map gives it, the majority of its pixels' classes (the map_class "
                   "column), the class estimate and certify grade.")


OTHER_DATES_NOTE = ("both maps are graded against the one class written per window, so they must describe the same "
                    "time: across dates, a change on the ground between them reads as an error of the map of the "
                    "other date. For maps of different dates, sample each map at random on its own and label it at "
                    "its own date")


def _sample_disagreement(args):
    """sample --other: windows drawn at random among those where two maps' classes differ, written for a reviewer to
    record the class seen in each. estimate then says which map is more accurate and by how much."""
    if args.design is not None:
        raise SystemExit("sample --other draws its own design, at random among the windows where the two maps differ; "
                         "drop --design")
    if args.condition is not None:
        raise SystemExit("sample --other does not take --condition: the comparison covers the whole map")
    if not 1 <= args.patch:
        raise SystemExit(f"--patch must be at least 1, got {args.patch}")
    t = _two_map_windows(args.scores, args.other, args.nodata, args.threshold, args.patch)
    hw, ww = t.a_w.shape
    try:
        smp = est.sample_disagreement(t.a_w.ravel(), t.b_w.ravel(), args.budget, valid=t.ok.ravel(), seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"sample: {exc}") from None
    idx = smp["indices"]
    rows, cols = np.divmod(idx, ww)
    code = (lambda v: int(v)) if t.code_of is None else (lambda v: int(t.code_of[v]))
    ca = [code(t.a_w.ravel()[i]) for i in idx]
    cb = [code(t.b_w.ravel()[i]) for i in idx]
    pr, pc, x, y = window_coords(rows, cols, t.geo, args.patch)
    if os.path.isdir(args.out):
        raise SystemExit(f"--out {args.out} is a directory; name the CSV to write, for example {os.path.join(args.out, 'to_label.csv')}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # class_a and class_b are the maps' classes at the window; hide them from the reviewer to label blind
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "class_a", "class_b",
                    "reference_class"])
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), ca[k], cb[k], ""])
    geo = t.geo
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in smp.items()}
    side.update({"map_a": _where(args.scores), "map_b": _where(args.other), "patch": args.patch,
                 "nodata": _json_nodata(args.nodata), "threshold": args.threshold, "grid": [int(hw), int(ww)],
                 "csv": os.path.abspath(args.out), "class_a": ca, "class_b": cb, "notes": list(t.notes),
                 "crs": None if geo is None else str(geo["crs"]),
                 "transform": None if geo is None else list(geo["transform"])[:6],
                 "xy_are": "window centres in the raster's CRS" if geo is not None else "absent: the input had no georeference",
                 "how_to_label": "open each window and write in reference_class the class that is there at the time "
                                 "both maps describe, in the maps' class codes, or ? where it cannot be judged (keep "
                                 "the row); hide class_a and class_b "
                                 "to label blind; then: oe-inferencex estimate " + os.path.basename(args.out)})
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(_stamp(side, "sample"), f, indent=1)
    D, N = smp["n_disagree"], smp["n_population"]
    print(f"{len(idx)} windows to label of the {D} where the two maps differ ({100 * D / N:.1f}% of the {N} windows "
          f"compared: both maps predict them and neither splits evenly); wrote {args.out} and its .json. Write in `reference_class` the class you see in each window, or ? "
          f"where a window cannot be judged (keep its row), then run: oe-inferencex estimate {args.out}"
          + "".join(f"\nnote: {n}" for n in t.notes) + f"\nnote: {OTHER_DATES_NOTE}")
    return 0


def _zone_words(zone):
    """How the confident zone was set, as printed after its share of the map: the top third, the top S, or nothing
    for a cut given as a confidence. Windows tied at the cut are in the zone, so on a banded or saturated confidence
    it can hold far more than the share asked for; the words then say how many the ties added (the review of 13
    October 2026: a band with 80% of its pixels at 100 printed a zone of 81.3% of the map as "the top third")."""
    if zone["rule"] != "share":
        return ""
    top = "the top third" if zone["share"] == est.AUDIT_SHARE else f"the top {100 * zone['share']:g}%"
    extra = int(zone["n_zone"]) - int(zone["n_top"])
    if extra > 0:
        return (f" ({top} by rank, {zone['n_top']} window{'' if zone['n_top'] == 1 else 's'}, and {extra} more tied "
                "at the cut)")
    return f" ({top})"


def _sample_class_audit(args):
    """sample --design audit: --per-class N windows at random from each predicted class's confident zone, written for
    a reviewer to record the class seen in each. estimate then gives each class's zone error rate with its exact
    interval and, with --alpha, a verdict per class."""
    if getattr(args, "other", None):
        raise SystemExit("sample --design audit draws from one map's confident zone; --other compares two maps, which "
                         "is its own route: drop one of them")
    if getattr(args, "condition", None) or getattr(args, "condition_names", None):
        raise SystemExit("sample --design audit does not take --condition: the audit's strata are the predicted classes, "
                         "and an input condition is not crossed with them in this version")
    if getattr(args, "threshold", None) is not None:
        raise SystemExit("sample: --threshold reads a second map's cut-off and is used only with --other")
    if any(getattr(args, k, None) is not None for k in ("anchor", "alpha", "delta", "extend")):
        raise SystemExit("sample: --anchor, --alpha, --delta and --extend belong to --design sequential")
    n = getattr(args, "per_class", None)
    if n is None or n < 1:
        raise SystemExit("sample --design audit needs --per-class N, at least 1: the windows labelled per predicted "
                         "class (docs/Usage.md, audit, says what N buys)" + ("" if n is None else f"; got {n}"))
    share, cut = getattr(args, "zone_share", None), getattr(args, "zone_cut", None)
    if share is not None and cut is not None:
        raise SystemExit("sample: the confident zone is --zone-share or --zone-cut, not both")
    if share is not None and not 0 < share <= 1:
        raise SystemExit(f"sample: --zone-share is a share of the valid windows, in (0, 1]; got {share:g}")
    if cut is not None and not np.isfinite(cut):
        raise SystemExit(f"sample: --zone-cut is a confidence, a finite number; got {cut:g}")
    if os.path.isdir(args.out):
        raise SystemExit(f"--out {args.out} is a directory; name the CSV to write, for example {os.path.join(args.out, 'audit.csv')}")
    _check_patch(args.patch, "sample")
    conf_path, conf_range = _product_args(args, "sample")
    out, geo, _, _, prod_notes, prod_info = _sample_map(args, conf_path, conf_range)
    arr = out["arrays"]
    margin, valid_w, klass = arr["confidence"], arr["valid"], arr["pooled_argmax"]
    hw, ww = margin.shape
    try:
        smp = est.sample_class_audit(margin.ravel(), klass.ravel(), n, valid=valid_w.ravel(), share=share, cut=cut,
                                     seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"sample: {exc}") from None
    idx = smp["indices"]
    rows, cols = np.divmod(idx, ww)
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # map_class is the class the window is audited for, fixed at the draw; hide it from the reviewer to label
        # blind. No `wrong` column: the error is reference_class against map_class, read by estimate
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "confidence", "map_class",
                    "reference_class"])
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), float(margin[r, c]), int(klass[r, c]), ""])
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in smp.items()}
    side.update(_sidecar_map_fields(args, geo, [int(hw), int(ww)], out, conf_path, conf_range, prod_info))
    side["how_to_label"] = ("open each window and write in reference_class the class that is there, in the map's class "
                            "codes, or ? where it cannot be judged (keep the row); hide map_class to label blind; then: "
                            f"oe-inferencex estimate {os.path.basename(args.out)} --alpha A, with A chosen before any "
                            "label is read, as is any --also-accept file")
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(_stamp(side, "sample"), f, indent=1)
    cls, z = smp["audited"], smp["zone"]
    full = [f"class {c}, {e['n_zone']} window{'' if e['n_zone'] == 1 else 's'}" for c, e in cls.items() if e["census"]]
    whose = (f"each of the {len(cls)} predicted classes' confident zones" if len(cls) > 1 else
             "the one predicted class's confident zone")
    print(f"{len(idx)} windows to label: up to {n} from {whose}" + (f" (labelled in full: {'; '.join(full)})" if full else "")
          + f"; the zone is the {z['n_zone']} valid windows at confidence {z['cut']:.6g} or above, "
          f"{100 * z['share_of_map']:.1f}% of the {smp['n_population']}{_zone_words(z)}; wrote {args.out} and its .json. "
          "Write in `reference_class` the class you see in each window, or ? where a window cannot be judged (keep its "
          f"row); hide map_class to label blind. Then run: oe-inferencex estimate {args.out} --alpha A, with A chosen "
          "now, before any label is read; write any --also-accept file now too"
          + "".join(f"\nnote: class {c} has no window in the zone ({e['n_windows']} windows of the map); the audit says "
                    "nothing about it" for c, e in smp["not_audited"].items())
          + "".join(f"\nwarning: {w}" for w in out.get("warnings", []))
          + "".join(f"\nnote: {p}" for p in prod_notes if p))
    return 0


def _estimate_disagreement(args):
    """estimate on a sample --other CSV: which map is more accurate, and by how much."""
    # --confidence too: added with the product input and accepted here unread, so a wrong or missing band path
    # passed with exit 0 (2026-10-06)
    # --classes too, accepted unread, even negative (the bug hunt of 2026-10-25)
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


# What `sample` prints for the reviewer. Until 1.4.1 it asked for 1 or 0 only, and a reviewer who could not judge a
# window had no way to say so that `estimate` accepted.
FILL_INSTRUCTION = ("Fill the `wrong` column with 1 or 0 per window, or ? where a window cannot be judged (keep its row), "
                    "then run:")


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
    try:
        res = est.estimate_error_rate(sample, wrong, unjudged=unjudged, reviewer_false_alarm=fa, reviewer_miss=miss)
        # the second reading: the same estimator on the same sample, with the windows the file accepts counted right
        alt = None if also is None else est.estimate_error_rate(sample, wrong_alt, unjudged=unjudged)
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
                                        legend_size=getattr(args, "classes", None))
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
                lines.append(f"  class {c}: user's accuracy {f(ua)}, producer's {pt} (exact {fl(100 * pe['low'], 0):.0f}-"
                             f"{ce(100 * pe['high'], 0):.0f}), share of map {100 * row['map_share']:.1f}% -> "
                             f"{'area' if (side.get('condition') or {}).get('by') == 'map class' else 'labels'} "
                             f"{100 * se['estimate']:.1f}% (exact {fl(100 * se['low'], 1):.1f}-{ce(100 * se['high'], 1):.1f})"
                             f"{drawn}" + normal + tag)
            else:
                lines.append(f"  class {c}: user's accuracy {f(ua)}, producer's {f(pa)}, share of map "
                             f"{100 * row['map_share']:.1f}% -> error-adjusted {f(sh)}" + tag)
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
        res.update({"also_accept": also, "also_accept_unused": unused, "reading_note": est.ESTIMATE_READING_NOTE})
        n = also["n_accepted"]
        moved = (f"{n} labelled window{'s' if n != 1 else ''} wrong above {'are' if n != 1 else 'is'} right under this "
                 "reading (" + ", ".join(f"{c} seen as {r}: {k}" for c, by in also["errors_accepted"].items()
                                         for r, k in by.items()) + ")"
                 if n else "no labelled window wrong above is right under this reading")
        lines = [f"also accepting {_also_words(also)} ({os.path.basename(accept_path)}): error rate {_rate(also)}, 95% "
                 f"interval {100 * also['low']:.1f}% to {100 * also['high']:.1f}%; {moved}"]
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
    print(words(f"error rate {_rate(res)}, 95% interval {100 * res['low']:.1f}% to {100 * res['high']:.1f}% "
          f"(half-width {100 * res['half_width']:.1f} points), from {res['n_labelled']} labelled windows of {res['n_population']}; "
          f"{res['method']}" + (f"\nnote: {res['bounds_note']}" if res.get("bounds_note") else "")
          + (f"\nwarning: {res['warning']}" if "warning" in res else "") + cond_text + per_class_text
          + (f"\nwarning: {res['per_class_warning']}" if "per_class_warning" in res else "")
          + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
          + (f"\nnote: {res['sequential_note']}" if "sequential_note" in res else "") + also_text) + f"\nwrote {out}")
    return 0


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


def _worded_print(words):
    import builtins
    return lambda text="", **kw: builtins.print(words(text), **kw)


def cmd_decide(args):
    """Typed answers to set questions, read from a result JSON that estimate, certify or compare wrote."""
    from oe_inferencex.decide import decide
    try:
        res = decide(args.result, args.ask or [])
    except ValueError as exc:                       # decide's refusals, unreadable files and non-results alike
        raise SystemExit(str(exc)) from None
    r = args.result
    out = args.out or (r[:-5] + "_decisions.json" if r.endswith(".json") else r + "_decisions.json")
    _refuse_overwriting_inputs("decide", [out], [r])
    _check_out_file(out, "decide")
    # a result of the class design is answered in its own words: decide said "input condition" and "of this condition"
    # of map classes (the bug hunt of 2026-10-25); decide.decide words its own texts so
    say = print
    try:
        with open(r, encoding="utf-8") as f:
            given = json.load(f)
        say = _worded_print(class_wording(given, r, out, names=result_class_names(given, r)))
    except (OSError, ValueError, AttributeError):
        pass
    for q, a in res["answers"].items():
        shown = f"{100 * a['answer']:.1f}%" if a["type"] == "score" else a["answer"]
        say(f"{q}: {shown}. {a['because']}")
        alt = a.get("also_accept")                    # a second reading of the same labels (--also-accept)
        if alt:
            say("  also accepting the file's pairs: " + (
                f"{100 * alt['answer']:.1f}%" if a["type"] == "score" else
                f"{alt['answer']} ({'95%' if alt.get('exact', True) else 'nominal 95%'} interval "
                f"{100 * alt['low']:.1f}% to {100 * alt['high']:.1f}%)" if "low" in alt
                else f"{alt['answer']} (certified: {100 * alt['certified']:.1f}%)"))
        for name, c in (a.get("per_condition") or {}).items():
            c2 = c.get("also_accept")
            if "answer" in c:
                say(f"  {name}: {c['answer']}" + (f"; also accepting: {c2['answer']}" if c2 else ""))
            else:
                share = c.get("certified_share_of_condition")
                said = (f"{100 * share:.1f}% of this condition certified" if share else
                        f"nothing certified ({c.get('reason') or 'no zone passed its test'})")
                if c2:
                    share2 = c2.get("certified_share_of_condition")
                    said += "; also accepting: " + (f"{100 * share2:.1f}% of this condition certified" if share2
                                                    else "nothing certified")
                say(f"  {name}: {said}")
        for c, row in (a.get("per_class") or {}).items():
            iv = (f" (95% interval {100 * row['low']:.1f}% to {100 * row['high']:.1f}%)" if row.get("low") is not None
                  else f" ({row.get('because', 'no interval')})")
            # an audit's second reading of the same labels; a class with nothing more accepted reads the same
            alt = row.get("also_accept")
            say(f"  class {c}: {row['answer']}{iv}" + ("  [warning]" if row.get("warning") else "")
                  + (f"; also accepting {', '.join(str(x) for x in alt['accepted'])}: {alt['answer']} "
                     f"({100 * alt['low']:.1f}% to {100 * alt['high']:.1f}%)" if alt and alt.get("accepted") else ""))
    with open(out, "w") as f:
        json.dump(_stamp(res, "decide"), f, indent=1, default=float)
    print(f"wrote {out}")
    return 0


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


def cmd_mcp(args):
    """Serve the commands above as MCP tools on stdio, for an agent on the user's machine (oe_inferencex.mcp_server).
    The agent starts this; nobody types into it. Without the mcp extra it says how to install it."""
    from oe_inferencex import mcp_server
    try:
        server = mcp_server.build_server()
    except ImportError as exc:
        raise SystemExit(f"oe-inferencex mcp: {exc}") from None
    server.run("stdio")
    return 0


ALSO_ACCEPT_HELP = ("JSON {map class: [reference classes also accepted]}, written before any label is read: a second "
                    "reading of the same labels, which needs a `reference_class` column (an audit sample has one)")
# certify refuses an audit sample, so its help does not point to one
CERTIFY_ALSO_ACCEPT_HELP = ("JSON {map class: [reference classes also accepted]}, written before any label is read: a "
                            "second reading of the same labels, which needs a `reference_class` column beside `wrong`")


def build_parser():
    p = argparse.ArgumentParser(prog="oe-inferencex", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    add_assess_parser(sub)
    d = sub.add_parser("demo", help="first run: audit the real sample map shipped with the package (or a made-up one) and draw the result")
    d.add_argument("--out", default="oe_inferencex_demo", help="output directory (default oe_inferencex_demo)")
    d.add_argument("--seed", type=int, default=0, help="seed of the made-up scene and of the random pick")
    d.add_argument("--made-up", action="store_true", help="audit a small synthetic water map instead of the real sample tile")
    d.set_defaults(func=cmd_demo)
    add_from_embeddings_parser(sub)
    add_boundary_parser(sub)
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
    ce = sub.add_parser("certificate", help="pack a certified zone to travel with the map: a STAC Item stating the "
                                            "guarantee, the zone as a GeoTIFF (.npy for a map with no grid), the "
                                            "labels, and digests binding them to the map")
    ce.add_argument("zone", help="the JSON certify wrote (random_zone.json)")
    ce.add_argument("--date", required=True, help="the date or period the map describes (YYYY-MM-DD or start/end)")
    ce.add_argument("--out", required=True, help="the directory to write the certificate bundle into")
    ce.add_argument("--scores", default=None, help="the map, if it has moved since sample read it")
    ce.add_argument("--confidence", default=None, help="a product's confidence band, if it has moved")
    ce.set_defaults(func=cmd_certificate)
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
    # not required by the parser: the audit design takes --per-class instead, and cmd_sample refuses a missing budget
    sm.add_argument("--budget", type=int, default=None,
                    help="number of windows to label (exp78 measured 300; about 400 for a random sample whose labels "
                         "also serve certify at alpha 0.05); every design but audit needs it")
    sm.add_argument("--out", required=True, help="CSV to write; a .json sidecar with the design goes beside it")
    sm.add_argument("--design", choices=("confidence", "proportional", "random", "sequential", "tiles", "condition",
                                         "class", "audit"),
                    default=None,
                    help="confidence (default without --condition): stratified by margin, allocated from the model's "
                         "own confidence; tiles: how people actually label, with the cluster interval that requires; "
                         "condition (default with --condition): stratified by input condition, labels split equally, "
                         "never from the model's confidence; random takes --condition too and only records it; "
                         "sequential: a random order labelled from the top, to which labels can be added (--extend), "
                         "and certify then holds at every look; class: stratified by the map's own class, at least "
                         "--min-per-class each and the rest in proportion to class size (Olofsson et al. 2014), with "
                         "exact per-class accuracy and area from estimate --per-class and a zone per class from "
                         "certify; audit: N windows at random from each predicted "
                         "class's confident zone (--per-class N), and estimate then gives each class's zone error rate")
    sm.add_argument("--min-per-class", type=int, default=None, metavar="M",
                    help=f"class design: labels per map class at least, where it has that many windows (default "
                         f"{est.CLASS_FLOOR})")
    sm.add_argument("--per-class", type=int, default=None, metavar="N",
                    help="audit design: windows labelled per predicted class; a class whose zone holds N or fewer is "
                         "labelled in full (docs/Usage.md, audit, says what N buys)")
    sm.add_argument("--zone-share", type=float, default=None, metavar="S",
                    help="audit design: the confident zone is this top share of the valid windows by confidence "
                         "(default a third)")
    sm.add_argument("--zone-cut", type=float, default=None, metavar="C",
                    help="audit design: the confident zone is the windows at confidence C or above, in the units of "
                         "the CSV's `confidence` column")
    sm.add_argument("--target", type=float, default=None, metavar="C",
                    help="random design: fix now the zone certify's target rule will test, the C most confident share "
                         "of the map (a grid level, 0.05 to 1); with the target right it needs about half the labels "
                         "of the Bonferroni rule (plan --coverage C shows how many)")
    sm.add_argument("--anchor", type=float, default=None,
                    help="sequential design: the smallest zone certify will test, as a share of the map, fixed now "
                         "before any label; or give --alpha instead")
    sm.add_argument("--alpha", type=float, default=None,
                    help="sequential design, without --anchor: the error rate certify will be asked about; the anchor "
                         "is then the zone this budget can certify if none of its labels is wrong, at most a quarter "
                         "of the map, fixed now and written to the sidecar. With --seed-from-map (any design): the "
                         "alpha certify will test, fixed now")
    sm.add_argument("--delta", type=float, default=None,
                    help=f"sequential design, with --alpha: the delta the anchor is computed for (default "
                         f"{est.ZONE_DELTA}); with --seed-from-map, the delta certify will use, fixed now")
    sm.add_argument("--rule", choices=("prefix", "bonferroni"), default=None,
                    help="with --seed-from-map: the rule certify will use, fixed now (default prefix; a sample with "
                         "--target uses the target rule and a sequential one the sequential rule)")
    sm.add_argument("--level-cut", choices=("standard", "ramp"), default=None,
                    help="with --seed-from-map: certify's level cut, fixed now (default standard)")
    sm.add_argument("--extend", default=None, metavar="CSV",
                    help="sequential design: a sample CSV to extend to --budget windows, keeping its labels; the new "
                         "windows are the next ones of the same random order")
    sm.add_argument("--condition", default=None,
                    help="optional integer raster on the map's grid: each pixel's input condition (a cloud flag, the "
                         "modalities present); each condition then gets its own error rate from `estimate` and its "
                         "own zone from `certify`")
    sm.add_argument("--condition-names", nargs="+", default=None, metavar="VALUE=NAME",
                    help="names for the condition values, e.g. 0=clear 1=cloudy (default: the value itself)")
    sm.add_argument("--tile", type=int, default=16, help="tiles design: tile side in windows (default 16)")
    sm.add_argument("--per-tile", type=int, default=16, help="tiles design: windows labelled per tile (default 16)")
    sm.add_argument("--logits", action="store_true")
    sm.add_argument("--patch", type=int, default=4)
    sm.add_argument("--nodata", type=float, default=None)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--seed-from-map", action="store_true",
                    help="take the seed from the map's digest and every setting that fixes the draw, so that "
                         "anyone holding the map derives it again; a certificate needs it (verify checks it)")
    sm.add_argument("--other", default=None, metavar="MAP_B",
                    help="a second map of the same grid and the same time: sample at random among the windows where "
                         "the two maps' classes differ, to learn which map is more accurate (estimate then compares "
                         "them)")
    sm.add_argument("--threshold", type=float, default=None,
                    help="with --other: cut-off of a 2-D continuous map (0.5 for a probability map when omitted)")
    _product_parser_args(sm)
    sm.set_defaults(func=cmd_sample)
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
    e.set_defaults(func=cmd_estimate)
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
    dc = sub.add_parser("decide", help="typed answers (yes / no / undetermined, a / b, a share) to set questions, read "
                                        "from a result JSON that estimate, certify or compare wrote")
    dc.add_argument("result", help="the JSON estimate, certify or compare wrote")
    dc.add_argument("--ask", action="append", metavar="QUESTION[=VALUE]",
                    help="a question, repeatable: error_rate_below=0.1, user_accuracy_above=0.85, "
                         "producer_accuracy_above=0.85, more_accurate, trusted_share, trusted_share_at_least=0.5, "
                         "share_differs. Without --ask, it stops and names the questions the result can answer")
    dc.add_argument("--out", default=None, help="JSON to write (default: <result>_decisions.json)")
    dc.set_defaults(func=cmd_decide)
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
    m = sub.add_parser("mcp", help="serve these commands as tools to an agent on this machine (a local MCP server on "
                                    "stdio; needs the mcp extra)")
    m.set_defaults(func=cmd_mcp)
    add_from_olmoearth_parser(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

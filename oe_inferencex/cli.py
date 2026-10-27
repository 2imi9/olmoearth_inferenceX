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
from oe_inferencex.scale.command import _window_size_text, add_scale_parser, cmd_scale
from oe_inferencex.agency.command import _iv_pct, _range_pct, add_grade_parser, cmd_grade
from oe_inferencex.compare.command import _window_majority, add_compare_parser, cmd_compare
from oe_inferencex.decide.command import _worded_print, add_decide_parser, cmd_decide
from oe_inferencex.plan.command import (_labels, _ladder_text, _plan_windows, _points, _reach, _sample_command,
                                        add_plan_parser, cmd_plan)
from oe_inferencex.certify.command import (CERTIFY_ALSO_ACCEPT_HELP, ZONE_ALSO_KEYS, _also_mask, _certify_by_condition,
                                           _certify_sequential, _keep_to_plan, _zone_also_accept, _zone_also_text,
                                           _zone_reading_note, add_certificate_parser, add_certify_parser,
                                           add_verify_parser, cmd_certificate, cmd_certify, cmd_verify)


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
    add_scale_parser(sub)
    add_grade_parser(sub)
    add_certificate_parser(sub)
    add_verify_parser(sub)
    add_compare_parser(sub)
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
    add_certify_parser(sub)
    add_decide_parser(sub)
    add_plan_parser(sub)
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

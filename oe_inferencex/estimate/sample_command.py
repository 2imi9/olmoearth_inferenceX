"""The command sample: which windows to label, by design, written as a CSV with its sidecar, and how to extend it."""
import csv
import json
import os

import numpy as np

from oe_inferencex import estimate as est
from oe_inferencex.estimate.command_text import _zone_words
from oe_inferencex.review.assess import RULE_TEXT, _pool, _pool_valid, assess_classmap, assess_prediction, pool_condition
from oe_inferencex.review.windows import _two_map_windows
from oe_inferencex.common.condition import _condition_args, _condition_names, _read_condition
from oe_inferencex.common.labels import _check_sidecar, _csv_rows, _rounding_tolerance
from oe_inferencex.common.maps import _check_scores
from oe_inferencex.common.output import (_check_patch, _json_nodata, _nodata_arg, _num_arg, _refuse_overwriting_inputs,
                                         _same_file, _stamp, _where)
from oe_inferencex.common.product import PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product
from oe_inferencex.common.raster import _mask_hint, read_raster, window_coords
from oe_inferencex.common.wording import class_wording


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


def _profile_need_note(names, sizes, n_labelled, alpha=0.05, delta=est.ZONE_DELTA):
    """How many labels a profile needs before `certify` can call it good, at an illustrative alpha: delta is split over
    the L profiles labelled in part, and each interval's upper tail holds delta / (2 L), so a profile with no error
    among its labels is good from min_labels_to_certify(alpha, delta / (2 L)) labels in a large profile."""
    part = [(name, int(n)) for name, N, n in zip(names, sizes, n_labelled) if 0 < int(n) < int(N)]
    L = len(part)
    if L == 0:
        return "every profile is labelled in full: certify compares each one's exact rate with alpha"
    b = est.min_labels_to_certify(alpha, delta / (2 * L))
    short = ", ".join(f"{name} gets {n}" for name, n in part if n < b)
    return (f"with delta {delta:g} split over the {L} profile{'s' if L != 1 else ''} labelled in part, a profile with no "
            f"error among its labels is found good at alpha {alpha:g} from about {b} labels (fewer in a small profile)"
            + (f"; {short}" if short else ""))


def _print_profile_sample(args, sample, cond, prof, layer, out, prod_notes):
    """What sample prints for the profile design: the profiles, the labels each gets, and what certify can do with
    them. Not worded through class_wording: its notes speak of the input-condition layer, which is a field here."""
    K = len(cond["names"])
    lines = [f"  {name}: {int(N)} windows, {int(n)} to label" + (" (all)" if int(n) >= int(N) else "")
             for name, N, n in zip(cond["names"], cond["sizes"], cond["n_labelled"])]
    notes = [est.PROFILE_ALLOCATION_NOTE.format(K=K) if K >= 2 else None,
             None if layer is None else _unrecorded_note(layer["n_split"], layer["n_no_code"]),
             _profile_need_note(cond["names"], cond["sizes"], cond["n_labelled"]), est.PROFILE_BLIND_NOTE] + prof["notes"]
    print(f"{len(sample['indices'])} windows to label of {sample['n_population']} valid (profile design, {K} "
          f"profile{'s' if K != 1 else ''} by {prof['block']['by']}); wrote {args.out} and its .json. "
          f"{FILL_INSTRUCTION} oe-inferencex certify {args.out} --alpha A (a verdict per profile, with A and --delta "
          f"chosen now, before any label is read), or oe-inferencex estimate {args.out} (each profile's error rate)\n"
          + "\n".join(lines)
          + "".join(f"\nwarning: {w}" for w in out.get("warnings", []))
          + "".join(f"\nnote: {n}" for n in prod_notes + notes if n))
    return 0


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
    if getattr(args, "design", None) != "profile" and (getattr(args, "profile_by", None) is not None
                                                       or getattr(args, "min_windows", None) is not None):
        raise SystemExit("sample: --profile-by and --min-windows belong to --design profile")
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
    by_profile = design == "profile"
    profile_fields = min_windows = None
    if by_profile:
        # Localized certification: the strata are profiles fixed from the map alone, before any label (estimate.profiles);
        # the condition design's equal split, draw and estimators are reused with the profiles as its conditions
        if getattr(args, "profile_by", None) is None:
            raise SystemExit("sample: --design profile needs --profile-by, the fields that fix each window's profile "
                             "before any label: confidence:Q, condition, class and grid:G (for example "
                             "confidence:3,class)")
        try:
            profile_fields = est.parse_profile_by(args.profile_by)
        except ValueError as exc:
            raise SystemExit(f"sample: {exc}") from None
        named = {f for f, _ in profile_fields}
        if cond_path and "condition" not in named:
            raise SystemExit("sample: --condition is read only when --profile-by names condition")
        if "condition" in named and not cond_path:
            raise SystemExit("sample: --profile-by names condition, which needs --condition, a raster of each pixel's "
                             "input condition")
        min_windows = (est.MIN_PROFILE_WINDOWS if getattr(args, "min_windows", None) is None
                       else int(args.min_windows))
        if min_windows < 1:
            raise SystemExit(f"sample: --min-windows is at least 1, got {min_windows}")
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
    layer = prof = None
    if by_profile:                                     # each window's stratum: its profile, fixed from the map alone
        a_ = out["arrays"]
        try:
            prof = est.build_profiles(a_["confidence"], a_["valid"], profile_fields,
                                      condition=None if pooled is None else pooled["grid"], condition_names=cond_names,
                                      map_class=a_["pooled_argmax"], min_windows=min_windows)
        except ValueError as exc:
            raise SystemExit(f"sample: {exc}") from None
        layer, layer_names = pooled, cond_names
        pooled = {"grid": prof["index"].reshape(a_["confidence"].shape), "n_split": 0, "n_no_code": 0}
        cond_names = dict(enumerate(prof["names"]))
    arr = out["arrays"]
    margin, valid_w, klass = arr["confidence"], arr["valid"], arr["pooled_argmax"]
    hw, ww = margin.shape
    tiles = None
    if design == "tiles":
        t = max(1, args.tile)
        tiles = (np.arange(hw)[:, None] // t) * ((ww + t - 1) // t) + (np.arange(ww)[None, :] // t)
    try:
        sample = est.sample_for_estimation(margin, args.budget, design="condition" if by_class or by_profile else design,
                                           p1=p1_w,
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
        side["condition"] = {"source": "map class" if by_class else "profile" if by_profile else _where(cond_path),
                             "rule": CLASS_RULE_TEXT if by_class else est.PROFILE_RULE_TEXT if by_profile else RULE_TEXT,
                             "values": cond["values"],
                             "names": cond["names"], "sizes": cond["sizes"], "n_labelled": cond["n_labelled"],
                             "allocation_rule": cond["allocation_rule"], "n_windows_split": pooled["n_split"],
                             "n_windows_no_code": pooled["n_no_code"],
                             "sha256": (None if by_class or by_profile else seed_rule["settings"]["condition_sha256"]
                                        if seed_rule is not None else raster_digest(cond_path)),
                             "names_given": None if by_profile else names_given(cond_names)}
        if by_class:
            side["condition"]["by"] = "map class"      # the class design: certify, estimate and verify read it so
            side["condition"]["min_per_class"] = class_floor
        if by_profile:
            # the profile design: certify gives each profile a verdict and estimate words its strata as profiles. The
            # partition, its fields and the fold rule are written here, before any label; the input-condition layer,
            # when a field, is described under its field, not in the condition block, whose strata are the profiles
            side["condition"]["by"] = "profile"
            block = prof["block"]
            for field in block["fields"]:
                if field["field"] == "condition":
                    field.update({"source": _where(cond_path), "sha256": raster_digest(cond_path),
                                  "names_given": names_given(layer_names), "n_windows_split": layer["n_split"],
                                  "n_windows_no_code": layer["n_no_code"]})
            side["profile"] = block
    side.update(_sidecar_map_fields(args, geo, [int(hw), int(ww)], out, conf_path, conf_range, prod_info))
    side["how_to_label"] = ("open each window, set wrong=1 if the map's class there (the `map_class` column: the "
                            "majority of the window's pixels) is not what is on the ground, else 0; then: "
                            "oe-inferencex estimate " + os.path.basename(args.out))
    if by_profile:
        side["how_to_label"] += (f"; then: oe-inferencex certify {os.path.basename(args.out)} --alpha A, with A (and "
                                 "--delta) chosen before any label is read")
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(_stamp(side, "sample"), f, indent=1)
    if by_profile:
        return _print_profile_sample(args, sample, cond, prof, layer, out, prod_notes)
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


# What `sample` prints for the reviewer. Until 1.4.1 it asked for 1 or 0 only, and a reviewer who could not judge a
# window had no way to say so that `estimate` accepted.
FILL_INSTRUCTION = ("Fill the `wrong` column with 1 or 0 per window, or ? where a window cannot be judged (keep its row), "
                    "then run:")


def add_sample_parser(sub):
    sm = sub.add_parser("sample", help="which windows to label so that `estimate` can say how wrong the map is")
    sm.add_argument("scores", help="the same map `assess` takes: (H, W) probability or logit map, or (C, H, W) scores")
    # not required by the parser: the audit design takes --per-class instead, and cmd_sample refuses a missing budget
    sm.add_argument("--budget", type=int, default=None,
                    help="number of windows to label (exp78 measured 300; about 400 for a random sample whose labels "
                         "also serve certify at alpha 0.05); every design but audit needs it")
    sm.add_argument("--out", required=True, help="CSV to write; a .json sidecar with the design goes beside it")
    sm.add_argument("--design", choices=("confidence", "proportional", "random", "sequential", "tiles", "condition",
                                         "class", "audit", "profile"),
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
                         "class's confident zone (--per-class N), and estimate then gives each class's zone error rate; "
                         "profile: stratified by profiles fixed from the map before any label (--profile-by), labels "
                         "split equally, and certify then calls each profile good, bad or undecided")
    sm.add_argument("--profile-by", default=None, metavar="SPEC",
                    help="profile design: the fields that fix each window's profile, comma-separated: confidence:Q (Q "
                         f"bands, 2 to {est.MAX_BANDS}), condition (needs --condition), class, grid:G (G by G blocks, 2 "
                         f"to {est.MAX_GRID}); at most {est.MAX_PROFILES} profiles")
    sm.add_argument("--min-windows", type=int, default=None, metavar="M",
                    help=f"profile design: a combination with fewer windows is folded into one profile named other "
                         f"(default {est.MIN_PROFILE_WINDOWS})")
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

"""Command line: a prediction raster in, the review set and its explanation out; two rasters in, their difference out.

    oe-inferencex assess  scores.tif --out DIR [--logits] [--patch 4] [--nodata V] [--reference labels.tif]
                          [--budgets 0.01 0.05 0.10] [--order confidence|boundary_first]
                          [--condition layer.tif [--condition-names 0=clear 1=cloudy]]
    oe-inferencex compare a.tif b.tif --out DIR [--patch 4] [--nodata V] [--labels labels.tif] [--groups ids.tif]
                          [--threshold T]
    oe-inferencex demo    [--out DIR] [--made-up]   a first run on the real sample map shipped with the package
    oe-inferencex sample  scores.tif --budget 300 --out to_label.csv [--design D] [--condition layer.tif]
    oe-inferencex estimate to_label.csv [--per-class]       once the reviewer has filled the `wrong` column
    oe-inferencex certify  to_label.csv --alpha 0.05        from a random sample, or one drawn with --condition
    oe-inferencex mcp                                       a local MCP server on stdio, for an agent (the mcp extra)
    oe-inferencex from-olmoearth DS --out DIR [--conditions]  an OlmoEarth run's rslearn dataset -> the rasters
                                                            assess reads, and the command that reads them

Inputs are GeoTIFFs (any rasterio-readable raster) or .npy arrays: (H, W) for a binary map, (C, H, W) for per-class
scores or, for `compare`, an integer class map. `compare` also takes two continuous maps (a regression output) when the
cut-off is named with --threshold; without it a map outside [0, 1] is refused rather than cut at 0.5. No-data comes
from the raster's own value, NaN, or --nodata. Outputs are
plain files the caller reads back: JSON summaries (assess.summary / compare's dict), CSVs of the review windows with
pixel and map coordinates, and rasters on the window grid (patch x patch pixels per window) when the input was one.
Nothing here narrates; the JSON is the evidence (docs/method/agent_integration.md).

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
import os
import sys

import numpy as np

from oe_inferencex.assess import (RULE_TEXT, _boundary_valid, _pool, _pool_valid, _pooled_argmax, assess_classmap,
                                  assess_prediction, pool_condition, summary)
from oe_inferencex import estimate as est
from oe_inferencex.compare import compare_inferences
from oe_inferencex.explain import explain_review_set
from oe_inferencex.signals import boundary_indicator


# ----------------------------------------------------------------------------- IO
def read_raster(path, nodata=None):
    """(array, valid mask, geo) where geo is None for .npy and {transform, crs} for rasters. One band comes back as
    (H, W) from either format. A pixel is no-data where any band is not finite, or where every band holds the no-data
    value (GDAL's dataset mask)."""
    if path.endswith(".npy"):
        # a mistyped path is a named refusal, not a traceback (bug hunt of 2026-10-06)
        try:
            a = np.load(path)
        except FileNotFoundError:
            raise SystemExit(f"{path}: no such file") from None
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{path}: not an array numpy can read ({exc})") from None
        nd, geo = nodata, None
    else:
        try:
            import rasterio
            from rasterio.errors import RasterioIOError
        except ImportError as ex:  # pragma: no cover
            raise SystemExit("reading rasters needs rasterio (pip install 'olmoearth-inferencex[geo]'); .npy inputs need nothing") from ex
        try:
            src = rasterio.open(path)
        except RasterioIOError as exc:
            raise SystemExit(f"{path}: no such file" if "://" not in path and not os.path.exists(path)
                             else f"{path}: not a raster rasterio can read ({exc})") from None
        with src:
            a = src.read()
            nd = src.nodata if nodata is None else nodata
            geo = {"transform": src.transform, "crs": src.crs}
    valid = np.isfinite(a).all(axis=0) if a.ndim == 3 else np.isfinite(a)
    if nd is not None:
        # No-data only where EVERY band holds the value, as GDAL's dataset mask has it. Until 2026-10-06 one band
        # sufficed, so a probability vector with one class at exactly 0 (a profile copied from Sentinel-2 carries
        # nodata=0) was dropped: the most confident windows left the population, and estimate gave 15.7% where the
        # map's rate is 7.0%, with no message.
        valid &= (a != nd).any(axis=0) if a.ndim == 3 else (a != nd)
    if a.ndim == 3 and a.shape[0] == 1:
        # A (1, H, W) .npy, which np.save(path, src.read()) writes, was argmaxed by compare as one class everywhere:
        # "0 of 4096 windows differ" where the same maps as GeoTIFFs differ on 507, exit 0 (2026-10-06).
        a = a[0]
    return a, valid, geo


def write_raster(path, array, geo, patch, nodata):
    """A window-grid array as a GeoTIFF on the input's grid scaled by `patch`, or .npy without geo."""
    if geo is None:
        np.save(os.path.splitext(path)[0] + ".npy", array)
        return os.path.splitext(path)[0] + ".npy"
    import rasterio
    from rasterio.transform import Affine
    arr = np.asarray(array)
    dtype = "float32" if arr.dtype.kind == "f" else ("uint8" if arr.dtype == bool else "int32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs=geo["crs"], transform=geo["transform"] * Affine.scale(patch), nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def window_coords(rows, cols, geo, patch):
    """Pixel corner and map centre of each window."""
    pr, pc = rows * patch, cols * patch
    if geo is None:
        return pr, pc, None, None
    x, y = geo["transform"] * (pc + patch / 2, pr + patch / 2)
    return pr, pc, np.asarray(x), np.asarray(y)


def pool_valid(valid, patch):
    h, w = valid.shape[0] // patch * patch, valid.shape[1] // patch * patch
    return valid[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3)) >= 0.5


# ----------------------------------------------------------------------------- assess
def _pct(x, nd=2):
    """A percentage, or the word undefined. `summary` turns NaN into None, and `None or 0` prints 0.00%, which reads as
    "they never disagree" when the truth is "this could not be computed". This package exists to keep those apart."""
    return "undefined" if x is None else f"{100 * float(x):.{nd}f}%"


PROB_TOLERANCE = 1e-6           # how far outside [0, 1] a probability may round, as assess._check_probabilities allows


def _check_scores(scores, valid, is_logit, path):
    """Refuse a file that is not what the flags say it is, rather than scoring it anyway.

    A hard class map is the likeliest file an operator has to hand, and `assess_prediction` would take the 2-D branch,
    threshold it at 0.5 and compute a confidence of 9.0 for a class index of 5, returning a full plausible review set
    with exit 0. Silence is worse than absence here: the operator dispatches a field team on a nonsense ordering with
    nothing to warn them."""
    if is_logit:
        return
    v = scores[:, valid] if scores.ndim == 3 else scores[valid]
    if v.size == 0:
        raise SystemExit(f"{path}: no valid pixels")
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    # The API's tolerance (assess._check_probabilities): float32 rounding puts a probability 1 ulp above 1, which the
    # API ranked and the command line refused as "values run 0 to 1, which is not a probability map" (2026-10-06).
    # The range is printed to 7 digits, so a refused value never reads as inside [0, 1].
    if lo < -PROB_TOLERANCE or hi > 1.0 + PROB_TOLERANCE:
        # Until 2026-10-06 this said the command line does not read a class map with its confidence band, which
        # --confidence has done since the product input (19160a2); people and agents were sent to the Python API
        # instead. The MCP server matches "which is not a probability map. Pass --logits", so that stays.
        raise SystemExit(
            f"{path}: values run {lo:.7g} to {hi:.7g}, which is not a probability map. Pass --logits if these are "
            f"logits, or give a (C, H, W) per-class score map. For a hard class map with its confidence band, as "
            f"published products ship them, pass the band with --confidence BAND (and --confidence-range LOW HIGH "
            f"when the band also holds codes).")
    if scores.ndim == 3:
        return
    integral = np.array_equal(v, np.rint(v))
    if integral and len(np.unique(v)) > 2:
        raise SystemExit(
            f"{path}: {len(np.unique(v))} distinct integer values in [0, 1] is a class map, not a probability map. "
            f"For a hard class map with its confidence band, pass the band with --confidence BAND.")


# ----------------------------------------------------------------------------- the input-condition layer
def _condition_names(items, command):
    """{value: name} from `--condition-names 0=clear 1=cloudy`, or None. Unique, non-empty names and the reserved
    "unrecorded" are the API's to check."""
    if not items:
        return None
    names = {}
    for item in items:
        key, sep, name = str(item).partition("=")
        try:
            value = int(key.strip())
        except ValueError:
            value = None
        if not sep or value is None:
            raise SystemExit(f"{command}: --condition-names takes value=name pairs, such as 0=clear 1=cloudy; got {item!r}")
        if value in names:
            raise SystemExit(f"{command}: --condition-names names the value {value} twice")
        names[value] = name
    return names


def _read_condition(path, shape, geo, command):
    """The per-pixel input condition layer, checked to lie on the map's grid, as a masked array: the raster's own
    no-data is masked, and NaN and negative values record no condition either (pool_condition). The layer must be
    one band; its values are pool_condition's to check."""
    a, valid, geo_c = read_raster(path, None)
    if a.ndim == 3 and a.shape[0] == 1:                   # a single-band .npy saved as (1, H, W)
        a = a[0]
    if a.ndim != 2:
        raise SystemExit(f"{command}: {path} has shape {a.shape}; the condition layer is one band, one value per pixel "
                         "of the map")
    if a.shape != tuple(shape):
        raise SystemExit(f"{command}: {path} has shape {a.shape}; the map is {tuple(shape)}")
    _same_grid(geo, geo_c, f"{command}: {path}")        # every other refusal of the layer names the command
    return np.ma.masked_array(a, mask=~valid)


def _condition_args(args, command):
    """(path of the condition raster or None, {value: name} or None), read with getattr: demo.py and the tests build
    the argument Namespace by hand, without these attributes."""
    path = getattr(args, "condition", None)
    names = _condition_names(getattr(args, "condition_names", None), command)
    if names and not path:
        raise SystemExit(f"{command}: --condition-names names the values of a --condition raster, and none was given")
    return path, names


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


def _check_patch(patch, command):
    """--patch 0 was a ZeroDivisionError traceback in assess and sample, and a negative one numpy's "can only specify
    one unknown dimension", where compare refused it by name (2026-10-06)."""
    if patch < 1:
        raise SystemExit(f"{command}: --patch must be at least 1, got {patch}")


def _check_out_dir(out, command):
    """assess and compare write into the directory --out names; an existing file there was a FileExistsError traceback
    from os.makedirs, after the whole run (2026-10-06)."""
    if os.path.exists(out) and not os.path.isdir(out):
        raise SystemExit(f"{command}: --out {out} is a file; name the directory to write into")


def _check_out_file(out, command):
    """estimate, certify and decide write the file --out names: a directory there, or a missing parent directory, was a
    traceback, and certify had already left its mask beside the directory (2026-10-06). The parent is made, as sample
    makes it."""
    if os.path.isdir(out):
        raise SystemExit(f"{command}: --out {out} is a directory; name the JSON to write, for example "
                         f"{os.path.join(out, 'result.json')}")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)


def _refuse_overwriting_inputs(command, outputs, inputs):
    """Refuse an output path that is one of the inputs. `assess --condition condition.npy --out .` replaced the user's
    pixel layer with the window grid, exit 0, and the next run was refused on its shape; `estimate s.csv --out s.json`
    replaced the sample's design, and every later estimate died on a KeyError (2026-10-06)."""
    given = {os.path.realpath(p): p for p in inputs if p}
    for o in outputs:
        if os.path.realpath(o) in given:
            raise SystemExit(f"{command}: writing {o} would overwrite the input {given[os.path.realpath(o)]}; name "
                             "another --out")


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
    s["inputs"] = {"scores": os.path.abspath(args.scores), "logits": args.logits, "reference": os.path.abspath(args.reference) if args.reference else None}
    if conf_path:   # "scores" keeps naming the first argument, here the class map
        s["inputs"].update({"classes": os.path.abspath(args.scores), "confidence": os.path.abspath(conf_path),
                            "confidence_range": None if conf_range is None else [float(v) for v in conf_range]})
        s["notes"] = list(s.get("notes", [])) + prod_notes
    arr = out["arrays"]
    conf, bnd = arr["confidence"], arr["boundary"]
    per_cond = name_at = None
    if layer is not None:
        # recorded only with a layer: without one, assessment.json differs from 1.3.1's by `scope` and the corrected
        # warning texts listed in tests/golden/condition_1_3_1/changes.py
        s["inputs"].update({"condition": os.path.abspath(cond_path),
                            "condition_names": {str(k): v for k, v in cond_names.items()} if cond_names else None})
        s["conditions"]["source"] = os.path.abspath(cond_path)
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
        with open(path, "w", newline="") as f:
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
            with open(path, "w", newline="") as f:
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
        json.dump(s, f, indent=1)
    ref_block = s.get("against_reference") or {}
    # A budget is printed as the percent it is: rounded to a whole percent, 0.1% and 0.4% both printed as "0%" and
    # 2.5% as "2%", while the files and the JSON kept them apart (2026-10-06).
    if "error_capture_at_budget" in ref_block:
        ref_text = "; against the reference: error capture " + ", ".join(
            f"{100 * float(b):g}% -> {v['errors_captured_fraction']:.2f}" for b, v in ref_block["error_capture_at_budget"].items())
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
    print(f"{s['n_windows']} windows of {args.patch} px; review sets " + ", ".join(f"{100 * float(b):g}%: {rs['n_windows']}" for b, rs in out["review_sets"].items()) +
          f"; boundary windows {100 * s['boundary_window_fraction']:.1f}%" + ref_text + cond_text +
          "".join(f"\nnote: {n}" for n in prod_notes) +
          f"\nwrote {args.out}/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary")
    return 0


# ----------------------------------------------------------------------------- compare
def decisions(path, nodata, threshold, notes=None):
    """(hard decisions, valid pixels, geo, n_classes, per-pixel confidence or None). The confidence breaks a tied
    window majority the way assess does; a hard class map carries none."""
    a, valid, geo = read_raster(path, nodata)
    if a.ndim == 3:
        hard, n_classes = a.argmax(0), a.shape[0]
        return hard, valid, geo, n_classes, np.nan_to_num(a.max(0).astype(np.float64), nan=0.0)
    if threshold is not None:
        # A named cut-off applies whatever the storage: percent cover held as int16 used to go to the class-map
        # branch, and 0-100 was compared as a 101-class majority with the cut-off ignored (503 windows "differ"
        # where cutting at 50 gives 74), exit 0 (audit 2026-09-22).
        lo, hi = (float(np.nanmin(a[valid])), float(np.nanmax(a[valid]))) if valid.any() else (0.0, 1.0)
        if not 0.0 <= threshold < 1.0 and a.dtype.kind in "biu" and valid.any() and np.isin(a[valid], (0, 1)).all():
            # A 0/1 class map (a forest mask) is no continuous map: cut at 50 it became class 0 everywhere, so a
            # percent-cover map against the mask made from it at 50 "differed" on 50% of windows, exit 0, and no note
            # named the mask (2026-10-06). A cut-off in [0, 1) gives the same two classes, and that path is kept.
            # Only an integer or boolean raster is taken for a mask: a float map whose values happen to be 0 and 1 (a
            # height map in metres, an all-dry depth map) is a continuous map and is cut (the review of 2026-10-06).
            if notes is not None:
                notes.append(f"{os.path.basename(path)} holds only 0 and 1, so it is read as a 0/1 class map, not cut at "
                             f"{threshold:g}")
            return np.where(valid, a, 0).astype(int), valid, geo, 2, None
        if (lo < 0.0 or hi > 1.0) and notes is not None:
            notes.append(f"{os.path.basename(path)} is a continuous map cut at {threshold:g}: the comparison is of that one "
                         f"decision, and no recorded experiment grades it on a regression output")
        af = a.astype(np.float64)
        cut = af > threshold
        if notes is not None and valid.any() and (cut[valid].all() or not cut[valid].any()):
            notes.append(f"every valid pixel of {os.path.basename(path)} lies {'above' if cut[valid].all() else 'at or below'} "
                         f"the cut-off {threshold:g}, so the cut makes that map one class everywhere")
        return cut.astype(int), valid, geo, 2, np.nan_to_num(np.abs(af - threshold), nan=0.0)
    if a.dtype.kind == "f" and not np.array_equal(a[valid], np.rint(a[valid])):
        # A continuous map outside [0, 1] under the default cut-off of 0.5 is one class everywhere on both sides, so
        # two regression outputs "never differ", with exit 0. The cut-off of a continuous map is the caller's to name.
        lo, hi = (float(np.nanmin(a[valid])), float(np.nanmax(a[valid]))) if valid.any() else (0.0, 1.0)
        continuous = lo < -PROB_TOLERANCE or hi > 1.0 + PROB_TOLERANCE      # as _check_scores (2026-10-06)
        if continuous and threshold is None:
            raise SystemExit(
                f"{path}: values run {lo:.7g} to {hi:.7g}, which is not a probability map. To compare two continuous maps "
                f"at a cut-off, name it with --threshold (for example --threshold 80); otherwise pass hard class maps "
                f"or (C, H, W) scores.")
        if continuous and notes is not None:
            notes.append(f"{os.path.basename(path)} is a continuous map cut at {threshold:g}: the comparison is of that one "
                         f"decision, and no recorded experiment grades it on a regression output")
        af = a.astype(np.float64)
        return (af > 0.5).astype(int), valid, geo, 2, np.nan_to_num(np.abs(af - 0.5), nan=0.0)
    hard = np.rint(a).astype(int)
    n_classes = int(hard[valid].max()) + 1 if valid.any() else 2
    return hard, valid, geo, n_classes, None


def _same_grid(geo_a, geo_b, what):
    """Refuse two rasters on different grids. Only shapes used to be compared, so a map shifted by 12.8 km, or in
    another CRS, was compared window by window as though co-registered."""
    if geo_a is None or geo_b is None:
        return
    ta, tb = np.asarray(tuple(geo_a["transform"])[:6], float), np.asarray(tuple(geo_b["transform"])[:6], float)
    # an absolute tolerance of a thousandth of a pixel: np.allclose's relative default allowed 50 m at a UTM northing
    # of 5e6 and 120 m at a Web Mercator easting of 1.2e7, so a map shifted by a whole window passed (review, 2026-09-23)
    atol = 1e-3 * max(abs(ta[0]), abs(ta[4]), 1e-12)
    if str(geo_a["crs"]) != str(geo_b["crs"]) or not np.allclose(ta, tb, rtol=0.0, atol=atol):
        raise SystemExit(f"{what} is not on the first map's grid (CRS {geo_b['crs']} vs {geo_a['crs']}, transform "
                         f"{tuple(geo_b['transform'])[:6]} vs {tuple(geo_a['transform'])[:6]}); resample it onto the same grid first")


PRODUCT_SIGNAL = "exported confidence band"
MAX_CLASS_ID = 255
NO_RANGE_NOTE = ("the whole confidence band is read as confidence; if it also holds codes (LCMAP's lcpconf writes "
                 "provenance codes from 151), pass --confidence-range so they are left out, or they rank above every "
                 "confidence")


def _read_product(classes_path, conf_path, nodata, conf_range, command):
    """A published product's two layers: a class map of integer ids and a per-pixel confidence band on its grid,
    higher = more confident, as products such as LCMAP ship them (lcpri with lcpconf). Band values outside conf_range
    are not confidences and are left out as no-data: LCMAP writes provenance codes from 151 into lcpconf, and read raw
    they would rank above every confidence and head the certified zone (exp93). Without a range the package cannot tell
    codes from confidences. `nodata` applies to the class map; the band's no-data is its file's own tag (or NaN).
    Returns (classes with -1 outside the valid pixels, confidence with NaN there, valid, geo, n_classes, notes, info),
    info holding how many classified pixels the range left out, since the population is then the rest."""
    hard, valid_h, geo = read_raster(classes_path, nodata)
    if not os.path.exists(conf_path):
        raise SystemExit(f"{command}: the confidence band {conf_path} is not a file")
    conf, valid_c, geo_c = read_raster(conf_path, None)
    if hard.ndim != 2 or conf.ndim != 2:
        raise SystemExit(f"{command}: with --confidence, the map is one band of class ids and the confidence one band; "
                         f"got shapes {hard.shape} and {conf.shape}")
    if conf.shape != hard.shape:
        raise SystemExit(f"{command}: the confidence band has shape {conf.shape}; the class map is {hard.shape}")
    _same_grid(geo, geo_c, conf_path)
    valid = valid_h & valid_c
    h = hard[valid]
    if h.size and not np.array_equal(h, np.rint(h)):
        raise SystemExit(f"{command}: {classes_path} holds values that are not whole numbers; with --confidence it must "
                         "be the class map of integer class ids (per-class scores go without --confidence)")
    if h.size and h.min() < 0:
        raise SystemExit(f"{command}: {classes_path} holds negative class ids at valid pixels; mark no-data with --nodata")
    if h.size and h.max() > MAX_CLASS_ID:
        raise SystemExit(f"{command}: {classes_path} holds class id {int(h.max())}; ids above {MAX_CLASS_ID} are refused, "
                         "since the window's majority class is counted per id. Mark a fill value with --nodata, or "
                         "renumber the classes")
    notes = []
    info = {"classified_pixels": int(valid.sum()), "outside_range": 0}
    if conf_range is None:
        notes.append(NO_RANGE_NOTE)
    else:
        lo, hi = (float(v) for v in conf_range)
        if not lo < hi:
            raise SystemExit(f"{command}: --confidence-range needs LOW below HIGH, got {lo:g} {hi:g}")
        outside = valid & ~((conf >= lo) & (conf <= hi))
        if outside.any():
            n_out, n_val = int(outside.sum()), int(valid.sum())
            notes.append(f"{n_out} of {n_val} pixels ({100 * n_out / n_val:.1f}%) have a confidence outside "
                         f"[{lo:g}, {hi:g}] and are left out as no-data: every statement is then about the rest")
            info["outside_range"] = n_out
        valid = valid & ~outside
    if not valid.any():
        raise SystemExit(f"{command}: no pixel has both a class and a confidence"
                         + ("" if conf_range is None else " inside --confidence-range"))
    classes = np.where(valid, np.rint(np.where(valid, hard, 0)), -1).astype(np.int64)
    return classes, np.where(valid, conf.astype(np.float64), np.nan), valid, geo, int(classes.max()) + 1, notes, info


def _population_note(side):
    """What a product sample's population leaves out, said with every number drawn from it."""
    rng, pop = side.get("confidence_range"), side.get("product_population") or {}
    n_out, n_all = int(pop.get("outside_range") or 0), int(pop.get("classified_pixels") or 0)
    if not rng or not n_out or not n_all:
        return None
    return (f"The population is the pixels whose confidence lies in [{rng[0]:g}, {rng[1]:g}]: {n_out} of {n_all} "
            f"classified pixels ({100 * n_out / n_all:.1f}%) were left out, and nothing here describes them. Left-out "
            "pixels can be wrong more often (exp93: LCMAP's provenance-coded plots, 29.1% against 17.7%).")


def _json_nodata(value):
    """A no-data value as strict JSON holds it: NaN is written "nan" (json.dump wrote the bare token NaN, which a
    strict parser refuses, 2026-10-06); _map_windows reads it back with float()."""
    if value is None:
        return None
    return "nan" if np.isnan(float(value)) else value


def _product_args(args, command):
    """--confidence and --confidence-range, checked together; --logits reads per-class scores and does not apply."""
    conf, rng = getattr(args, "confidence", None), getattr(args, "confidence_range", None)
    if rng is not None and conf is None:
        raise SystemExit(f"{command}: --confidence-range needs --confidence")
    if conf is not None and getattr(args, "logits", False):
        raise SystemExit(f"{command}: --logits reads per-class scores; with --confidence the map is a class map and the "
                         "band is its confidence")
    return conf, rng


def _two_map_windows(path_a, path_b, nodata, threshold, patch, labels_path=None):
    """Two maps (and an optional label raster) read onto one window grid, as compare grades them: each map's class per
    window from the pixels both maps predicted, ties broken by confidence where both maps carry one and left out where
    either is a class map, class codes remapped jointly when they are sparse. Shared by compare and by sample --other,
    so the windows sampled where two maps differ are the windows compare counts as differing."""
    notes = []
    ha, va, geo, na, ca = decisions(path_a, nodata, threshold, notes)
    hb, vb, geo_b, nb, cb = decisions(path_b, nodata, threshold, notes)
    if ha.shape != hb.shape:
        raise SystemExit(f"the two maps differ in shape: {ha.shape} vs {hb.shape}; compare needs identical grids")
    _same_grid(geo, geo_b, path_b)
    # The grid is the first map's that carries one. With map a a .npy, the labels and groups were checked against
    # no grid at all, so labels 50 km off b's grid graded the maps with exit 0, while the same files with a GeoTIFF
    # first were refused (2026-10-06).
    geo = geo if geo is not None else geo_b
    if not 1 <= patch <= min(ha.shape):
        raise SystemExit(f"--patch {patch} must be at least 1 and no larger than the map, {ha.shape[0]} x {ha.shape[1]} px")
    both = va & vb
    lab_i = lv = None
    if labels_path:
        lab, lv, geo_l = read_raster(labels_path, None)
        if lab.shape != ha.shape:
            raise SystemExit(f"{labels_path} has shape {lab.shape}; the maps are {ha.shape}")
        _same_grid(geo, geo_l, labels_path)
        lab_i = np.rint(lab).astype(int)
        lv = lv & (lab_i >= 0)                              # a negative code is unlabelled whatever the nodata tag says
        if not lv.any():
            raise SystemExit(f"{labels_path}: no valid label pixels")
    # One code space for both maps and the labels, decided before anything is pooled. Pooling counts one window grid
    # per class id up to the largest, so a stray code of 60000 took 26 s and 3.3 GB on a 512 x 512 map (review of
    # 2026-09-23); sparse or negative codes are remapped to 0..K-1 jointly and mapped back in the CSV.
    map_codes = np.unique(np.concatenate([ha[both].ravel(), hb[both].ravel()])) if both.any() else np.zeros(0, int)
    lab_codes = np.unique(lab_i[lv]) if lab_i is not None else np.zeros(0, int)
    all_codes = np.unique(np.concatenate([map_codes, lab_codes])).astype(int)
    if all_codes.size and (all_codes.min() < 0 or int(all_codes.max()) + 1 > 4 * max(all_codes.size, 16)):
        code_of = all_codes
        ha, hb = np.searchsorted(all_codes, ha), np.searchsorted(all_codes, hb)     # pixels outside `both` never vote
        if lab_i is not None:
            lab_i = np.where(lv, np.searchsorted(all_codes, lab_i), -1)
        n_classes = n_lab = max(int(all_codes.size), 2)
    else:
        code_of = None
        # Sized by the codes that vote, the pixels both maps predicted. max(na, nb) counted each map's own valid
        # pixels, so an undeclared fill of 65535 where the other map has no data allocated 65536 window grids
        # (4.3 GB and 13 s on 256 x 256) for codes that never vote, the cost the remap above was meant to end
        # (2026-10-06). Codes outside `both` never vote, so the windows' classes are unchanged.
        n_classes = max(int(map_codes.max()) + 1, 2) if map_codes.size else 2
        # The label raster is pooled over ITS OWN class range, never the maps'. _pooled_argmax counts votes only over
        # range(n_classes), so pooling a 0-5 label raster with the maps' n_classes=3 silently dropped every pixel of
        # class 3 and above and let the window label fall to a surviving low index: measured at 44.9% of window labels
        # wrong, with exit 0 and no warning, on the very output that says which inference to believe.
        n_lab = max(int(lab_codes.max()) + 1, 2) if lab_codes.size else 2
    # No-data pixels do not vote (they used to argmax to class 0 and could decide a window). Both maps pool over the
    # pixels BOTH predicted: a window half under B's no-data used to be decided from 16 of A's pixels and 8 of B's,
    # and two identical maps then "differed" there. A tied window goes to the more confident voters when both maps
    # carry a confidence, as in assess; when either is a hard class map, which has none, a tied window is left out
    # of the comparison on both sides, because breaking it one way on one side and another way on the other made
    # a class map and its own probability version differ on 1.8% of windows (review of 2026-09-23).
    weighted = ca is not None and cb is not None
    a_w = _pooled_argmax(np.where(both, ha, -1), n_classes, patch, empty=-1, weights=ca if weighted else None,
                         tie=None if weighted else -2)
    b_w = _pooled_argmax(np.where(both, hb, -1), n_classes, patch, empty=-1, weights=cb if weighted else None,
                         tie=None if weighted else -2)
    n_tie_windows = int((pool_valid(both, patch) & ((a_w == -2) | (b_w == -2))).sum())
    if n_tie_windows:
        notes.append(f"{n_tie_windows} windows are split evenly between two classes on a map with no confidence to break "
                     "the tie; they are left out of the comparison")
    ok = pool_valid(va & vb, patch) & (a_w >= 0) & (b_w >= 0)
    if not ok.any():
        # assess refuses an empty map; compare used to exit 0 with "0 of 0 windows differ"
        raise SystemExit("compare: no window was predicted by both maps (all no-data, or every window tied), so there is "
                         "nothing to compare")
    labels = groups = None
    ok_graded = None
    if lab_i is not None:
        # invalid label pixels must not vote; -1 is the non-voting code _assess uses, where 0 is a real class
        lab_w = _pooled_argmax(np.where(lv, lab_i, -1), n_lab, patch, empty=-1, tie=-1)   # no majority, no label
        # The grading runs on the labelled windows; the label-free numbers stay on every window both maps predicted.
        # Until 2026-09-22 the label mask was ANDed into the whole comparison, so adding --labels changed the numbers
        # the docs call label-free and dropped unlabelled differing windows from the CSV and the raster.
        labelled = pool_valid(lv, patch)
        n_tied = int((ok & labelled & (lab_w < 0)).sum())
        ok_graded = ok & labelled & (lab_w >= 0)            # an evenly split label window has no label to grade
        labels = lab_w
        unpredicted = sorted(int(c) for c in set(lab_codes.tolist()) - set(map_codes.tolist()))
        if unpredicted:
            # named by class, not by the largest id: "the labels carry 4 classes" was said of a raster holding class 3
            notes.append(f"the labels hold class(es) {unpredicted[:10]}{' ...' if len(unpredicted) > 10 else ''} that neither "
                         "map predicts; windows labelled with them are counted wrong for both sides")
        if n_tied:
            notes.append(f"{n_tied} windows have an evenly split label and no majority; they are left out of the grading")
    return SimpleNamespace(notes=notes, ha=ha, hb=hb, va=va, vb=vb, geo=geo, code_of=code_of, a_w=a_w, b_w=b_w, ok=ok,
                           labels=labels, ok_graded=ok_graded, lab_i=lab_i, n_classes=(na, nb))


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
    s["inputs"] = {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b), "labels": os.path.abspath(args.labels) if args.labels else None, "patch_px": args.patch,
                   "date_a": args.date_a, "date_b": args.date_b, "labels_date": args.labels_date}
    if notes:
        s["notes"] = notes
    # NaN where nothing was compared: a 0 there used to read as "agree" in any GIS
    dis = np.where(ok, out["arrays"]["disagree"].astype(np.float32), np.nan).astype(np.float32)
    s["files"] = {"disagreement": write_raster(os.path.join(args.out, "disagreement.tif"), dis, geo, args.patch, nodata=None)}
    rows, cols = np.nonzero(out["arrays"]["disagree"])
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    path = os.path.join(args.out, "differing_windows.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "a", "b"])
        for i, (r, c) in enumerate(zip(rows, cols)):
            ca_, cb_ = (int(a_w[r, c]), int(b_w[r, c])) if code_of is None else (int(code_of[a_w[r, c]]), int(code_of[b_w[r, c]]))
            w.writerow([int(r), int(c), int(pr[i]), int(pc[i]), None if x is None else float(x[i]), None if y is None else float(y[i]), ca_, cb_])
    s["files"]["differing_windows"] = path
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(s, f, indent=1)
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
    if getattr(args, "other", None):
        if getattr(args, "confidence", None) or getattr(args, "confidence_range", None):
            raise SystemExit("sample: --other draws among the windows where two class maps differ, which needs no "
                             "confidence; leave out --confidence")
        return _sample_disagreement(args)
    _check_patch(args.patch, "sample")
    conf_path, conf_range = _product_args(args, "sample")
    if getattr(args, "threshold", None) is not None:
        raise SystemExit("sample: --threshold reads a second map's cut-off and is used only with --other")
    cond_path, cond_names = _condition_args(args, "sample")
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
    anchor, extend = getattr(args, "anchor", None), getattr(args, "extend", None)
    a_alpha, a_delta = getattr(args, "alpha", None), getattr(args, "delta", None)
    if design != "sequential" and any(v is not None for v in (anchor, extend, a_alpha, a_delta)):
        raise SystemExit("sample: --anchor, --alpha, --delta and --extend belong to --design sequential")
    if design == "sequential" and cond_path:
        raise SystemExit("sample: --design sequential does not take --condition: certify reads it as one random order "
                         "of the whole map")
    old_side = old_wrong = anchor_rule = None
    if extend is not None:
        if any(v is not None for v in (anchor, a_alpha, a_delta)):
            raise SystemExit(f"sample --extend: the anchor was fixed when {extend} was drawn and cannot change after "
                             "labelling began; drop --anchor, --alpha and --delta")
        old_side, old_wrong = _sequential_to_extend(extend, args)
        anchor, anchor_rule = old_side["anchor"], old_side.get("anchor_rule")
    elif design == "sequential":
        # The anchor, the smallest zone certify will test, is fixed here, before any label: certify reads it from the
        # sidecar. Computed from the alpha certify will be asked about when it is not given; certify at another alpha
        # keeps this anchor (review of 8 October 2026: an anchor recomputed from each certify run's alpha was not
        # fixed before the labels, as its guarantee needs).
        if (anchor is None) == (a_alpha is None):
            raise SystemExit("sample: a sequential sample fixes now, before any label, the smallest zone certify will "
                             "test: give --alpha (the error rate certify will be asked about; the anchor is then the "
                             "zone this budget can certify if none of its labels is wrong, at most a quarter of the "
                             "map) or --anchor (a share of the map), not both")
        if anchor is not None and a_delta is not None:
            raise SystemExit("sample: --delta goes with --alpha, to compute the anchor; with --anchor it is not used")
        if a_alpha is not None:
            d = est.ZONE_DELTA if a_delta is None else a_delta
            if not (0 < a_alpha < 1 and 0 < d < 1):
                raise SystemExit(f"sample: --alpha and --delta must be in (0, 1), got {a_alpha:g} and {d:g}")
            from oe_inferencex import sequential as sq
            anchor = sq.anchor_coverage(args.budget, a_alpha, d)
            anchor_rule = {"alpha": a_alpha, "delta": d, "first_budget": int(args.budget),
                           "rule": "the smallest grid zone expecting min_labels_sequential labels at the first budget, "
                                   f"at most {sq.ANCHOR_CAP:g}"}
        else:
            anchor_rule = "declared"
    if anchor is not None and not est.ZONE_GRID[0] <= anchor <= 1:
        raise SystemExit(f"sample: --anchor is a share of the map between {est.ZONE_GRID[0]:g} and 1, got {anchor:g}")
    if conf_path:
        hard, conf_px, valid, geo, n_classes, more, prod_info = _read_product(args.scores, conf_path, args.nodata,
                                                                               conf_range, "sample")
        prod_notes += more
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
    arr = out["arrays"]
    margin, valid_w, klass = arr["confidence"], arr["valid"], arr["pooled_argmax"]
    hw, ww = margin.shape
    tiles = None
    if design == "tiles":
        t = max(1, args.tile)
        tiles = (np.arange(hw)[:, None] // t) * ((ww + t - 1) // t) + (np.arange(ww)[None, :] // t)
    try:
        sample = est.sample_for_estimation(margin, args.budget, design=design, p1=p1_w, tiles=tiles,
                                           per_tile=args.per_tile, valid=valid_w, seed=args.seed,
                                           condition=None if pooled is None else pooled["grid"].ravel(),
                                           condition_names=cond_names)
    except ValueError as exc:
        raise SystemExit(f"sample: {exc}")
    idx = sample["indices"]
    if old_side is not None:
        prev = np.asarray(old_side["indices"], int)
        if sample["n_population"] != old_side.get("n_population") or not np.array_equal(idx[:prev.size], prev):
            raise SystemExit(f"sample --extend: {extend} was not drawn from this map with this seed: its windows are not "
                             "the start of the order this map and seed give. Pass the same scores, --patch, --nodata "
                             "and --seed it was drawn with")
        sample["first_budget"] = old_side["first_budget"]
    cond = sample.get("condition")
    rows, cols = np.divmod(idx, ww)
    pr, pc, x, y = window_coords(rows, cols, geo, args.patch)
    if os.path.isdir(args.out):
        raise SystemExit(f"--out {args.out} is a directory; name the CSV to write, for example {os.path.join(args.out, 'to_label.csv')}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        # map_class: the class `estimate` and `certify` grade at the window (the majority of its pixels' classes),
        # shown so the reviewer judges the same thing the tool grades; a mixed window is where a reviewer looking
        # at the pixels could otherwise read the map's class differently (release check of 24 September)
        # with --condition, the condition's name follows `stratum` (under the condition design, the stratum is the
        # condition's index); map_class and wrong stay the last two columns
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "stratum"]
                   + ([] if cond is None else ["condition"]) + ["confidence", "map_class", "wrong"])
        strata = sample.get("strata")
        pos = {int(g): i for i, g in enumerate(sample.get("strata_of_population", []))}
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), None if strata is None else int(strata[pos[int(i)]])]
                       + ([] if cond is None else [cond["names"][int(sample["condition_grid"][int(i)])]])
                       + [float(margin[r, c]), int(klass[r, c]),
                          old_wrong[k] if old_wrong is not None and k < len(old_wrong) else ""])
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    if design == "sequential":
        side["anchor"], side["anchor_rule"] = float(anchor), anchor_rule     # fixed now; certify reads them
    if cond is not None:
        # the design's source of truth for `estimate` and `certify`, which never read the raster again
        side["condition"] = {"source": os.path.abspath(cond_path), "rule": RULE_TEXT, "values": cond["values"],
                             "names": cond["names"], "sizes": cond["sizes"], "n_labelled": cond["n_labelled"],
                             "allocation_rule": cond["allocation_rule"], "n_windows_split": pooled["n_split"],
                             "n_windows_no_code": pooled["n_no_code"]}
    if conf_path:   # estimate --per-class and certify rebuild the map from both layers, with the same range
        side.update({"classes": os.path.abspath(args.scores), "confidence": os.path.abspath(conf_path),
                     "confidence_range": None if conf_range is None else [float(v) for v in conf_range],
                     "product_population": prod_info})
    side.update({"scores": os.path.abspath(args.scores), "logits": args.logits, "patch": args.patch,
                 "nodata": _json_nodata(args.nodata),     # estimate --per-class and certify recompute the map with it
                 "grid": [int(hw), int(ww)], "csv": os.path.abspath(args.out),
                 # what a reviewer with a GIS needs to find a window: the CRS, the pixel size, and the window's
                 # footprint in ground units; x and y in the CSV are window centres in that CRS
                 "crs": None if geo is None else str(geo["crs"]),
                 "transform": None if geo is None else list(geo["transform"])[:6],
                 "pixel_size": None if geo is None else [abs(geo["transform"].a), abs(geo["transform"].e)],
                 "window_size_ground_units": None if geo is None else [abs(geo["transform"].a) * args.patch, abs(geo["transform"].e) * args.patch],
                 "xy_are": "window centres in the raster's CRS" if geo is not None else "absent: the input had no georeference",
                 "warnings": list(out.get("warnings", [])),
                 "how_to_label": "open each window, set wrong=1 if the map's class there (the `map_class` column: the "
                                 "majority of the window's pixels) is not what is on the ground, else 0; then: "
                                 "oe-inferencex estimate " + os.path.basename(args.out)})
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(side, f, indent=1)
    what, notes = f"{design} design", []
    if cond is not None:
        counts = ", ".join(f"{name} {int(n)}" for name, n in zip(cond["names"], cond["n_labelled"]))
        what = f"condition design: {counts}" if design == "condition" else f"random design; by chance: {counts}"
        K = len(cond["names"])
        if design == "condition" and K >= 2:
            notes.append(f"labels are split equally across the {K} input conditions so that each gets its own error "
                         "rate; the whole-map rate weights each condition by its share of the map")
        notes.append(_unrecorded_note(pooled["n_split"], pooled["n_no_code"]))
        if K >= 2:
            notes.append(_certify_need_note(cond["names"], cond["n_labelled"]))
    if design == "sequential":
        kept = 0 if old_wrong is None else sum(str(v).strip() != "" for v in old_wrong)
        notes.insert(0, ("label from the top row down, in order; certify may be run after any number of labels and "
                         "holds at every look. To add windows later: oe-inferencex sample " + args.scores
                         + f" --design sequential --seed {args.seed} --budget <more> --extend {args.out}")
                     + (f"; the {kept} labels already given are kept" if kept else ""))
    print(f"{len(idx)} windows to label of {sample['n_population']} valid ({what}); wrote {args.out} and its .json. "
          f"{FILL_INSTRUCTION} oe-inferencex estimate {args.out}"
          + "".join(f"\nwarning: {w}" for w in out.get("warnings", []))
          + (f"\nnote: {sample['note']}" if "note" in sample else "")
          + "".join(f"\nnote: {n}" for n in prod_notes + notes if n))
    return 0


def _sequential_to_extend(path, args):
    """The sidecar and the `wrong` column of a sequential sample that `sample --extend` lengthens, checked: it must be a
    sequential sample of this map drawn with this seed, shorter than the new budget; its anchor, fixed when it was
    drawn, is kept. Returns (sidecar, wrong values as written, one per row)."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    if not os.path.exists(path) or not os.path.exists(side_path):
        raise SystemExit(f"sample --extend: {path} and its .json sidecar are needed")
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path)
    if side.get("design") != "sequential" or side.get("anchor") is None:
        raise SystemExit(f"sample --extend: {path} was drawn with the {side.get('design')!r} design; only a sequential "
                         "sample can be extended, since only its order and anchor were fixed before labelling")
    if int(side.get("seed", -1)) != int(args.seed):
        raise SystemExit(f"sample --extend: {path} was drawn with --seed {side.get('seed')}; pass the same seed")
    if os.path.abspath(args.scores) != side.get("scores") or int(args.patch) != int(side.get("patch", -1)):
        raise SystemExit(f"sample --extend: {path} was drawn from {side.get('scores')} at --patch {side.get('patch')}; "
                         "extend it from the same map and patch")
    n_old = len(side["indices"])
    if args.budget <= n_old:
        raise SystemExit(f"sample --extend: {path} already holds {n_old} windows; --budget is the new total, larger "
                         "than that")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    try:
        idx = [int(float(r["index"])) for r in rows]
    except (KeyError, ValueError, OverflowError):
        raise SystemExit(f"sample --extend: {path} lacks the `index` column `sample` wrote") from None
    if idx != [int(i) for i in side["indices"]]:
        raise SystemExit(f"sample --extend: the rows of {path} do not match its sidecar; extend the file `sample` wrote")
    return side, [r.get("wrong", "") for r in rows]


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
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        # class_a and class_b are the maps' classes at the window; hide them from the reviewer to label blind
        w.writerow(["index", "window_row", "window_col", "pixel_row", "pixel_col", "x", "y", "class_a", "class_b",
                    "reference_class"])
        for k, (i, r, c) in enumerate(zip(idx, rows, cols)):
            w.writerow([int(i), int(r), int(c), int(pr[k]), int(pc[k]), None if x is None else float(x[k]),
                        None if y is None else float(y[k]), ca[k], cb[k], ""])
    geo = t.geo
    side = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in smp.items()}
    side.update({"map_a": os.path.abspath(args.scores), "map_b": os.path.abspath(args.other), "patch": args.patch,
                 "nodata": _json_nodata(args.nodata), "threshold": args.threshold, "grid": [int(hw), int(ww)],
                 "csv": os.path.abspath(args.out), "class_a": ca, "class_b": cb, "notes": list(t.notes),
                 "crs": None if geo is None else str(geo["crs"]),
                 "transform": None if geo is None else list(geo["transform"])[:6],
                 "xy_are": "window centres in the raster's CRS" if geo is not None else "absent: the input had no georeference",
                 "how_to_label": "open each window and write in reference_class the class that is there, in the maps' "
                                 "class codes, or ? where it cannot be judged (keep the row); hide class_a and class_b "
                                 "to label blind; then: oe-inferencex estimate " + os.path.basename(args.out)})
    with open(args.out[:-4] + ".json" if args.out.endswith(".csv") else args.out + ".json", "w") as f:
        json.dump(side, f, indent=1)
    D, N = smp["n_disagree"], smp["n_population"]
    print(f"{len(idx)} windows to label of the {D} where the two maps differ ({100 * D / N:.1f}% of the {N} windows "
          f"compared: both maps predict them and neither splits evenly); wrote {args.out} and its .json. Write in `reference_class` the class you see in each window, or ? "
          f"where a window cannot be judged (keep its row), then run: oe-inferencex estimate {args.out}"
          + "".join(f"\nnote: {n}" for n in t.notes))
    return 0


def _sidecar_design(path):
    """The design a sample CSV's sidecar records, or None when there is no readable sidecar."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    try:
        with open(side_path) as f:
            return json.load(f).get("design")
    except (OSError, ValueError):
        return None


def _estimate_disagreement(args):
    """estimate on a sample --other CSV: which map is more accurate, and by how much."""
    # --confidence too: added with the product input and accepted here unread, so a wrong or missing band path
    # passed with exit 0 (2026-10-06)
    if args.per_class or args.scores is not None or getattr(args, "confidence", None) is not None \
            or args.nodata is not None or args.reviewer_false_alarm or args.reviewer_miss:
        raise SystemExit("estimate: a sample of the windows where two maps differ takes none of --per-class, --scores, "
                         "--confidence, --nodata, --reviewer-false-alarm and --reviewer-miss; it compares the two maps "
                         "from the reference_class column alone")
    path = args.sample
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path, ("indices", "class_a", "class_b"))
    out = args.out or (path[:-4] + "_estimate.json" if path.endswith(".csv") else path + "_estimate.json")
    _refuse_overwriting_inputs("estimate", [out], [path, side_path])
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
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
    unjudged, ref, bad = [], [], []
    for i, r in enumerate(rows):
        v = str(r["reference_class"]).strip()
        if v == "?":
            unjudged.append(True)
            ref.append(-1)
            continue
        try:
            k = int(float(v))
            if float(v) != k:
                raise ValueError
        except (ValueError, OverflowError):
            bad.append((i + 2, v))
            k = 0
        unjudged.append(False)
        ref.append(k)
    if bad:
        raise SystemExit(f"`reference_class` must be the class seen (a whole number, in the maps' class codes) or ? in "
                         f"every row; {len(bad)} row(s) are not, first at row {bad[0][0]}: {bad[0][1]!r}")
    res = est.compare_from_disagreement(side, ca, cb, np.array(ref), unjudged=np.array(unjudged))
    res.update({"sample": os.path.abspath(path), "map_a": side.get("map_a"), "map_b": side.get("map_b"),
                "notes": list(side.get("notes", []))})
    _check_out_file(out, "estimate")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    d, v = res["difference"], res["verdict"]
    if v == "a":
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
          + f"\nnote: {res['note']}" + "".join(f"\nnote: {n}" for n in res["notes"]) + f"\nwrote {out}")
    return 0


# What `sample` prints for the reviewer. Until 1.4.1 it asked for 1 or 0 only, and a reviewer who could not judge a
# window had no way to say so that `estimate` accepted.
FILL_INSTRUCTION = ("Fill the `wrong` column with 1 or 0 per window, or ? where a window cannot be judged (keep its row), "
                    "then run:")


def _check_sidecar(side, side_path, keys=("indices",)):
    """A sidecar that holds no design is refused by name: one overwritten by a result JSON was a KeyError traceback on
    every later estimate and certify (2026-10-06)."""
    missing = [k for k in keys if k not in side] if isinstance(side, dict) else list(keys)
    if missing:
        raise SystemExit(f"{side_path} holds no sample design (no {', '.join(missing)}): it is not the sidecar `sample` "
                         "wrote, or a result was written over it; draw the sample again")


def _labelled_sample(path, command):
    """A filled-in sample CSV and its sidecar design, checked: the rows are the design's, every `wrong` is 0, 1 or `?`
    (a window the reviewer could not judge). Returns (sidecar, rows, indices, sample dict for the estimators, wrong,
    unjudged), with `wrong` 0 where the row is `?`."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    if not os.path.exists(side_path):
        raise SystemExit(f"{side_path} not found; `{command}` needs the sidecar `sample` wrote beside the CSV")
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path)
    with open(path, newline="", encoding="utf-8-sig") as f:                 # a spreadsheet's BOM is not a column name
        rows = list(csv.DictReader(f))
    need = {"index", "window_row", "window_col", "wrong"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected the columns `sample` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}. A spreadsheet saved with another delimiter "
                         "(semicolon) or with columns removed cannot be matched to its design")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])                # "73.0" after a spreadsheet round trip is 73
    except (ValueError, OverflowError) as exc:                              # "inf" is an OverflowError, not a ValueError
        raise SystemExit(f"{path}: the `index` column is not the one `sample` wrote: {exc}")
    if not np.array_equal(idx, np.asarray(side["indices"], int)):
        raise SystemExit("the CSV's rows do not match the design in its sidecar; label the file `sample` wrote, in order")
    blank = [i for i, r in enumerate(rows) if str(r.get("wrong", "")).strip() == ""]
    if blank and side.get("design") == "sequential":
        # a sequential sample is labelled from the top: the labelled rows are a prefix of its random order, and the
        # rows below them are windows not labelled yet. A label below a blank would make the labelled set depend on
        # which windows the reviewer chose to leave, so it is refused.
        first = blank[0]
        later = [i for i in range(first, len(rows)) if str(rows[i].get("wrong", "")).strip() != ""]
        if later:
            raise SystemExit(f"row {later[0] + 2} is labelled below row {first + 2}, which is not: a sequential sample is "
                             "labelled from the top row down, in order. Write ? in a window that cannot be judged rather "
                             "than leaving it blank")
        if first == 0:
            raise SystemExit(f"{path}: no window is labelled yet; fill the `wrong` column from the top row down")
        side = dict(side, indices=list(side["indices"])[:first], n_drawn=len(rows))
        rows, idx, blank = rows[:first], idx[:first], []
    if blank:
        raise SystemExit(f"{len(blank)} of {len(rows)} windows have no `wrong` value (first at row {blank[0] + 2}); "
                         "every sampled window needs a 1 or a 0, or a ? where it cannot be judged, or the design's "
                         "interval is not the one you get")
    # Exactly 0 or 1. int(float(x)) would read a reviewer's "0.5" (not sure) as right and "1.9" as wrong, with
    # exit 0; a spreadsheet's "TRUE" is refused rather than guessed at.
    # A window the reviewer cannot judge is `?`: it stays in the design (until 1.4.1 this message told the reviewer
    # to leave it out, and a CSV that did was refused for not matching its design), and estimate and certify bound it.
    ok = {"0": 0, "1": 1, "0.0": 0, "1.0": 1, "?": 0}
    bad = [(i + 2, r["wrong"]) for i, r in enumerate(rows) if str(r["wrong"]).strip() not in ok]
    if bad:
        raise SystemExit(f"`wrong` must be exactly 1, 0 or ? per window; {len(bad)} row(s) are not, first at row "
                         f"{bad[0][0]}: {bad[0][1]!r}. Write ? where you cannot judge a window: estimate then bounds it "
                         "both ways and certify counts it as wrong; do not remove its row")
    wrong = np.array([ok[str(r["wrong"]).strip()] for r in rows])
    unjudged = np.array([str(r["wrong"]).strip() == "?" for r in rows])
    cond = side.get("condition")
    if "condition" in rows[0] and isinstance(cond, dict) and side.get("condition_grid") is not None:
        # The condition is fixed when the sample is drawn, and the sidecar is its source of truth: the raster is never
        # read again. A row whose `condition` was edited would be counted in the condition the sidecar says, so the
        # edit is refused rather than ignored.
        grid, names = np.asarray(side["condition_grid"]), list(cond.get("names", []))
        def recorded(i):
            return names[int(grid[i])] if 0 <= i < grid.size and 0 <= int(grid[i]) < len(names) else None
        bad = [(k + 2, r["condition"], recorded(int(i))) for k, (r, i) in enumerate(zip(rows, idx))
               if recorded(int(i)) is None or str(r["condition"]).strip() != recorded(int(i)).strip()]
        if bad:
            raise SystemExit(f"{path}: the `condition` column disagrees with the design in its sidecar on {len(bad)} "
                             f"row(s), first at row {bad[0][0]}: {bad[0][1]!r} where the sample recorded {bad[0][2]!r}. "
                             "The condition of each window is fixed when the sample is drawn; label the file `sample` "
                             "wrote without changing that column")
    sample = {k: (np.asarray(v) if k in ("indices", "strata", "strata_of_population", "tiles", "condition_grid") else v)
              for k, v in side.items()}
    return side, rows, idx, sample, wrong, unjudged


def _rounding_tolerance(text):
    """Half a unit in the last digit a number was written with, and never less than 1e-6 (a full-precision value
    still carries float noise from the round trip). "0.412" gives 5e-4; "4.12e-01" the same."""
    t = str(text).strip().lower()
    mant, _, exp = t.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return max(0.5 * 10.0 ** (-decimals + (int(exp) if exp else 0)), 1e-6)


def _map_windows(side, scores_override, nodata, command, rows=None, idx=None, confidence_override=None):
    """The map's per-window confidence, class and validity, recomputed from the scores the sample was drawn on, and
    checked to be that map: its population must be the sample's, and its confidence at the sampled windows the
    CSV's (review of 2026-09-23: another map on the same grid was certified from this map's labels, and a sample
    drawn with --nodata was recomputed without it, with no message). Returns (confidence, class, valid, n_classes)."""
    if confidence_override and not side.get("confidence"):
        raise SystemExit(f"{command}: --confidence names a product's band, and this sample was drawn on scores, not "
                         "with --confidence")
    path = scores_override or side.get("scores")
    if not path or not os.path.exists(path):
        raise SystemExit(f"{command}: the map's scores are needed again ({path or 'no path in the sidecar'} not found); "
                         "pass --scores with the raster `sample` was run on")
    recorded = None if side.get("nodata") is None else float(side["nodata"])     # "nan" is how NaN is written
    if nodata is None:
        nodata = recorded                                     # the value the sample was drawn with
    elif recorded is not None and recorded != float(nodata) and not (np.isnan(recorded) and np.isnan(float(nodata))):
        # NaN is not equal to itself, so a sample drawn with --nodata nan was refused with "--nodata nan differs from
        # the nan the sample was drawn with" (2026-10-06)
        raise SystemExit(f"{command}: --nodata {nodata:g} differs from the {recorded:g} the sample was drawn with")
    if side.get("confidence"):                                  # a class map with its confidence band
        conf_path = confidence_override or side["confidence"]
        if not os.path.exists(conf_path):
            raise SystemExit(f"{command}: the map's confidence band is needed again ({conf_path} not found); pass "
                             "--confidence with the band `sample` was run on")
        hard, conf_px, valid, _, n_classes, _, _ = _read_product(path, conf_path, nodata, side.get("confidence_range"),
                                                                 command)
        try:
            out = assess_classmap(hard, conf_px, n_classes, patch=int(side.get("patch", 4)), nodata_mask=~valid,
                                  signal=PRODUCT_SIGNAL)
        except ValueError as exc:
            raise SystemExit(f"{command}: {exc}") from None
    else:
        scores, valid, _ = read_raster(path, nodata)
        _check_scores(scores, valid, bool(side.get("logits", False)), path)
        try:                                                  # a named refusal, as on the product branch (2026-10-06)
            out = assess_prediction(scores, is_logit=bool(side.get("logits", False)), patch=int(side.get("patch", 4)),
                                    nodata_mask=~valid)
        except ValueError as exc:
            raise SystemExit(f"{command}: {exc}") from None
        n_classes = int(scores.shape[0]) if scores.ndim == 3 else 2
    arr = out["arrays"]
    if list(arr["confidence"].shape) != list(side.get("grid", arr["confidence"].shape)):
        raise SystemExit(f"{command}: {path} pools to a {arr['confidence'].shape} window grid, but the sample was drawn on "
                         f"{side.get('grid')}; pass the raster the sample was drawn on")
    conf, valid_w = arr["confidence"], arr["valid"]
    n_pop = int((valid_w & np.isfinite(conf)).sum())
    if side.get("n_population") is not None and n_pop != int(side["n_population"]):
        # 1.7.0 changed the no-data rule of a multi-band raster (a pixel is no-data only when every band holds the
        # value), so a sample drawn before on such a raster can name a population no option restores (the review of
        # 2026-10-06): say so rather than send the user after a --nodata value that does not exist
        hint = (" If the sample was drawn before 1.7.0 on a multi-band raster with a no-data value, the population "
                "changed with the fix to the no-data rule (pixels where only some bands hold the value are data now); "
                "draw the sample again." if side.get("nodata") is not None or nodata is not None else "")
        raise SystemExit(f"{command}: {path} has {n_pop} valid windows, but the sample was drawn from {side['n_population']}; "
                         "pass the raster, and the --nodata value, the sample was drawn with." + hint)
    if rows is not None and idx is not None and "confidence" in rows[0]:
        try:
            written = np.array([float(r["confidence"]) for r in rows])
            # the tolerance follows the digits the CSV holds: a spreadsheet that saved 0.412 for 0.41234 is still
            # the same map (the verification of the second review found three digits refused at a fixed 1e-4)
            tol = np.array([_rounding_tolerance(r["confidence"]) for r in rows])
        except ValueError:
            written = None
        if written is not None:
            off = np.abs(conf.ravel()[idx] - written)
            if not (off <= tol * (1 + 1e-9) + 1e-12).all():      # a value written exactly half a unit off
                k = int(np.argmax(off))
                raise SystemExit(f"{command}: the map's confidence at the sampled windows is not the one the CSV records "
                                 f"(row {k + 2}: {written[k]:.6g} in the CSV, {conf.ravel()[idx][k]:.6g} now); this is not the "
                                 "map the sample was drawn on")
    if rows is not None and idx is not None and "map_class" in rows[0]:
        # The class too: a product's class map and band are two files, and another year's class map on the same grid
        # with the same band passed the confidence check, so estimate --per-class built its table from that map, exit
        # 0, with only a note blaming the labels (2026-10-06). A column that cannot be read skips the check, as above.
        try:
            recorded = np.array([int(float(r["map_class"])) for r in rows])
        except (ValueError, TypeError, OverflowError):
            recorded = None
        if recorded is not None:
            now = arr["pooled_argmax"].ravel()[idx]
            bad = np.flatnonzero(now != recorded)
            if bad.size:
                k = int(bad[0])
                raise SystemExit(f"{command}: the map's class at the sampled windows is not the one the CSV records on "
                                 f"{bad.size} row(s) (row {k + 2}: {recorded[k]} in the CSV, {int(now[k])} now); this is "
                                 "not the map the sample was drawn on")
    return conf, arr["pooled_argmax"], valid_w, n_classes


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
    --other (two maps) gives which map is more accurate instead."""
    if _sidecar_design(args.sample) == "disagreement":
        return _estimate_disagreement(args)
    if not args.per_class and (args.scores is not None or getattr(args, "confidence", None) is not None
                               or args.nodata is not None):
        # the error rate reads the CSV and its sidecar alone; accepting the map's options and ignoring them let a
        # wrong --scores path pass unnoticed (release check of 24 September), and --confidence, added with the product
        # input, was accepted unread the same way, a missing band included (2026-10-06)
        raise SystemExit("estimate: --scores, --confidence and --nodata are read only with --per-class; the error rate "
                         "comes from the CSV and its sidecar alone")
    side_path = args.sample[:-4] + ".json" if args.sample.endswith(".csv") else args.sample + ".json"
    out = args.out or (args.sample[:-4] + "_estimate.json" if args.sample.endswith(".csv") else args.sample + "_estimate.json")
    _refuse_overwriting_inputs("estimate", [out], [args.sample, side_path])
    side, rows, idx, sample, wrong, unjudged = _labelled_sample(args.sample, "estimate")
    fa, miss = args.reviewer_false_alarm or 0.0, args.reviewer_miss or 0.0
    for flag, v in (("--reviewer-false-alarm", fa), ("--reviewer-miss", miss)):
        if not 0 <= v < 1:
            raise SystemExit(f"estimate: {flag} must be at least 0 and below 1, got {v:g}")
    if args.per_class and (unjudged.any() or fa or miss):
        raise SystemExit("estimate --per-class needs the class seen in every window and a reviewer taken as right: "
                         + (f"{int(unjudged.sum())} window(s) are ? (could not be judged)" if unjudged.any() else
                            "--reviewer-false-alarm and --reviewer-miss do not apply to the per-class table")
                         + ". Run estimate without --per-class for the error rate with its bounds")
    if side.get("design") == "sequential":
        # the labelled rows are the first windows of a random order: a simple random sample of their number
        sample = dict(sample, design="random", budget=int(idx.size))
    try:
        res = est.estimate_error_rate(sample, wrong, unjudged=unjudged, reviewer_false_alarm=fa, reviewer_miss=miss)
    except ValueError as exc:
        raise SystemExit(f"estimate: {exc}")
    if side.get("design") == "sequential":
        res["sequential_note"] = (f"the first {int(idx.size)} windows of a sequential sample, read as a random sample of "
                                  "that size. The interval holds when that number was fixed before labelling; one read "
                                  "at a stop chosen because the labels looked good can be too narrow. certify on this "
                                  "sample holds at every look")
    per_class_text = ""
    if args.per_class:
        ref = _reference_classes(rows, args.sample)
        _, hard, valid_w, n_classes = _map_windows(side, args.scores, args.nodata, "estimate --per-class", rows, idx,
                                                   getattr(args, "confidence", None))
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
            pc = est.estimate_per_class(sample, ref, map_class, n_classes=n_classes)
        except ValueError as exc:
            raise SystemExit(f"estimate --per-class: {exc}")
        res["per_class"] = pc["per_class"]
        res["confusion_counts"] = pc["confusion_counts"]
        res["overall_accuracy"] = pc["overall_accuracy"]
        res["per_class_method"] = pc["method"]
        if "warning" in pc:
            res["per_class_warning"] = pc["warning"]
        if pc.get("overall_accuracy_post_stratified") is not None:
            res["overall_accuracy_post_stratified"] = pc["overall_accuracy_post_stratified"]
        if mismatch.size:
            res["per_class_note"] = (f"on {mismatch.size} row(s), first row {int(mismatch[0]) + 2}, `wrong` disagrees with whether "
                                     "`reference_class` differs from the map's class; the error rate above uses `wrong`, the "
                                     "per-class table uses `reference_class`")
        lines = []
        for c, row in pc["per_class"].items():
            if side.get("confidence") and row["map_share"] == 0 and not row.get("n_labelled_reference_class"):
                continue        # a product's id neither the map nor the labels use (LCMAP's classes start at 1)
            ua, pa, sh = row["user_accuracy"], row["producer_accuracy"], row["reference_share"]
            f = lambda v: "n/a" if v is None else f"{100 * v['estimate']:.0f}% ({100 * v['low']:.0f}-{100 * v['high']:.0f})"
            # the tag names the warning's reason; it used to say "few labels" for a near-census class of 500 labels
            tag = f"  [warning: {', '.join(row.get('warning_codes', ['see the JSON']))}]" if "warning" in row else ""
            lines.append(f"  class {c}: user's accuracy {f(ua)}, producer's {f(pa)}, share of map {100 * row['map_share']:.1f}% "
                         f"-> error-adjusted {f(sh)}" + tag)
        if res["design"] == "condition" and len(res.get("per_condition", {})) > 1:
            # the method string says so, and the printed table did not: the per-class intervals under this design
            # are not graded (review of 2026-09-29); with one condition the table is the random design's
            lines.append(f"note: {est.PER_CLASS_NOT_GRADED}")
        per_class_text = "\n" + "\n".join(lines) + (f"\nnote: {res['per_class_note']}" if "per_class_note" in res else "")
    res["sample"] = os.path.abspath(args.sample)
    if _population_note(side):                       # a product's range left pixels out: the rate is of the rest
        res["population_note"] = _population_note(side)
    _check_out_file(out, "estimate")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
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
            lines.append(f"note: {est.CONDITION_WHOLE_MAP}")
        cond_text = "\n" + "\n".join(lines)
    # the interval is printed as its two ends: it is not symmetric about the estimate (Wilson never is, and a
    # clipped one is not), so "estimate +/- x" would name an interval that is not the one written
    print(f"error rate {_rate(res)}, 95% interval {100 * res['low']:.1f}% to {100 * res['high']:.1f}% "
          f"(half-width {100 * res['half_width']:.1f} points), from {res['n_labelled']} labelled windows of {res['n_population']}; "
          f"{res['method']}" + (f"\nnote: {res['bounds_note']}" if res.get("bounds_note") else "")
          + (f"\nwarning: {res['warning']}" if "warning" in res else "") + cond_text + per_class_text
          + (f"\nwarning: {res['per_class_warning']}" if "per_class_warning" in res else "")
          + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
          + (f"\nnote: {res['sequential_note']}" if "sequential_note" in res else "") + f"\nwrote {out}")
    return 0


def cmd_certify(args):
    """The largest most-confident share of the map that is wrong at most --alpha of the time, certified from a
    random labelled sample so that the statement fails with probability at most --delta (exp80). A sample that
    records an input condition is certified per condition instead, with --delta split over the conditions tested
    (certify_by_condition); no whole-map zone is issued for it."""
    if _sidecar_design(args.sample) == "disagreement":
        raise SystemExit("certify needs a random sample of one map's windows; this sample holds only windows where two "
                         "maps differ, which says which map is more accurate (estimate) but nothing about either "
                         "map's zones")
    out = args.out or (args.sample[:-4] + "_zone.json" if args.sample.endswith(".csv") else args.sample + "_zone.json")
    mask_path = out[:-5] + ".npy" if out.endswith(".json") else out + ".npy"
    _refuse_overwriting_inputs("certify", [out, mask_path],
                               [args.sample, args.sample[:-4] + ".json" if args.sample.endswith(".csv") else args.sample + ".json"])
    side, rows, idx, sample, wrong, unjudged = _labelled_sample(args.sample, "certify")
    # A window that could not be judged counts as wrong: that can only certify less, so the guarantee holds whatever
    # made it hard to judge. certify takes no reviewer error rate: testing at alpha (1 - miss) would need the miss
    # rate to hold inside every zone the test can certify, where a model's confident errors sit, and a rate measured
    # on the whole map does not give that (review of 2 October 2026: misses placed in the confident half certified a
    # zone wrong 10% of the time at alpha 5% on every draw).
    wrong = np.where(unjudged, 1, wrong)
    alpha = args.alpha
    bounds = ([f"{int(unjudged.sum())} window(s) that could not be judged (?) are counted as wrong, which keeps the "
               "guarantee whatever made them hard to judge and certifies less"] if unjudged.any() else [])
    by_condition = side.get("condition_grid") is not None and side.get("design") in ("random", "condition")
    sequential = side.get("design") == "sequential"
    if sequential and (args.rule != "prefix" or args.level_cut is not None):
        raise SystemExit("certify: a sequential sample is certified by the sequential rule, which tests each zone from "
                         "the anchor outward at every look; --rule and --level-cut belong to the one-look rule")
    if side.get("design") != "random" and not by_condition and not sequential:
        raise SystemExit(f"certify needs a random sample: this CSV was drawn with the {side.get('design')!r} design. The "
                         "guarantee rests on the labelled windows inside each zone being a random sample of that zone, "
                         "which a stratified or tile draw is not. Draw one with `sample --design random`")
    margin, hard, valid_w, _ = _map_windows(side, args.scores, args.nodata, "certify", rows, idx,
                                            getattr(args, "confidence", None))
    _check_out_file(out, "certify")                   # before the mask, which a refusal used to leave behind
    if by_condition:
        return _certify_by_condition(args, sample, wrong, margin, valid_w, out, mask_path, bounds, int(unjudged.sum()))
    if sequential:
        return _certify_sequential(args, side, idx, wrong, margin, valid_w, out, mask_path, bounds, int(unjudged.sum()))
    try:
        res = est.certify_zone(margin.ravel(), idx, wrong, alpha, delta=args.delta, rule=args.rule, valid=valid_w.ravel(),
                               cut=args.level_cut)
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
    res["sample"] = os.path.abspath(args.sample)
    if bounds:
        res.update({"n_unjudged": int(unjudged.sum()), "bounds_note": "; ".join(bounds)})
    if _population_note(side):                       # a product's range left pixels out: the zone is of the rest
        res["population_note"] = _population_note(side)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    tail = "".join(f"\nnote: {b}" for b in bounds) + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
    if res["coverage"] is None:
        print(f"no zone certified at alpha={args.alpha:g}, delta={args.delta:g} from {res['n_labelled']} labels. {res['note']}"
              f"{tail}\nwrote {out}")
    else:
        what = "confidence" if side.get("confidence") else "confidence margin"     # a product's band is no margin
        print(f"the {100 * res['coverage']:.0f}% most confident windows ({res['n_zone']} of {res['n_population']}, {what} >= "
              f"{res['threshold']:.4f}) are wrong at most {100 * args.alpha:g}% of the time; this statement fails on at most "
              f"{100 * args.delta:g}% of samples like this one ({args.rule} rule; the exact upper bound on the zone's error "
              f"rate at that level is {100 * res['upper_bound']:.1f}%). Outside the zone nothing is certified.\n"
              f"{res['note']}{tail}\nwrote {out} and the window mask {mask_path}")
    return 0


def _certify_sequential(args, side, idx, wrong, margin, valid_w, out, mask_path, bounds=(), n_unjudged=0):
    """certify for a sequential sample: the labelled rows, read from the top, in the order they were drawn; the anchor
    the sample fixed, or the default from its first budget (sequential.anchor_coverage)."""
    from oe_inferencex import sequential as sq
    try:
        if side.get("anchor") is None:
            raise ValueError("this sequential sample records no anchor; draw it with this version's sample, which fixes "
                             "the anchor before any label")
        res = sq.certify_zone_sequential(margin.ravel(), idx, wrong, args.alpha, delta=args.delta,
                                         anchor=float(side["anchor"]), valid=valid_w.ravel(), seed=int(side["seed"]))
        res["anchor_rule"] = side.get("anchor_rule")
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
    res["sample"] = os.path.abspath(args.sample)
    res["n_drawn"] = int(side.get("n_drawn", len(idx)))
    if bounds:
        res.update({"n_unjudged": n_unjudged, "bounds_note": "; ".join(bounds)})
    if _population_note(side):
        res["population_note"] = _population_note(side)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    tail = "".join(f"\nnote: {b}" for b in bounds) + (f"\nnote: {res['population_note']}" if "population_note" in res else "")
    more = (f"\n{res['n_labelled']} of the {res['n_drawn']} windows drawn are labelled; label more from the top, or extend "
            "the sample (sample --extend), and run certify again: every look is covered by the same guarantee")
    if res["coverage"] is None:
        print(f"no zone certified yet at alpha={args.alpha:g}, delta={args.delta:g} from {res['n_labelled']} labels (sequential "
              f"rule, anchor {100 * res['anchor']:.0f}% of the map). {res['note']}{tail}{more}\nwrote {out}")
    else:
        what = "confidence" if side.get("confidence") else "confidence margin"
        print(f"the {100 * res['coverage']:.0f}% most confident windows ({res['n_zone']} of {res['n_population']}, {what} >= "
              f"{res['threshold']:.4f}) are wrong at most {100 * args.alpha:g}% of the time; this statement fails on at most "
              f"{100 * args.delta:g}% of samples like this one, however often certify is run on it as labels are added "
              f"(sequential rule, anchor {100 * res['anchor']:.0f}% of the map). Outside the zone nothing is certified.\n"
              f"{res['note']}{tail}{more}\nwrote {out} and the window mask {mask_path}")
    return 0


def _certify_by_condition(args, sample, wrong, margin, valid_w, out, mask_path, bounds=(), n_unjudged=0):
    """certify for a sample that records an input condition: a zone inside each condition with enough labels, and
    the union of the zones as the window mask. The top-level zone fields stay null, since the union is not "the
    most confident share of the map" that readers of those fields take them to be."""
    try:
        res = est.certify_by_condition(sample, wrong, margin.ravel(), args.alpha, delta=args.delta, rule=args.rule,
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
    res["sample"] = os.path.abspath(args.sample)
    if bounds:
        res.update({"n_unjudged": n_unjudged, "bounds_note": "; ".join(bounds)})
    if _population_note(sample):                     # a product's range left pixels out: the zones are of the rest
        res["population_note"] = _population_note(sample)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
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
                         f"of {e['n_population']}, {what} >= {e['threshold']:.4f}) are wrong at most {100 * a:g}% of "
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
    print("\n".join(lines) + f"\nwrote {out}")
    return 0


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
    for q, a in res["answers"].items():
        shown = f"{100 * a['answer']:.1f}%" if a["type"] == "score" else a["answer"]
        print(f"{q}: {shown}. {a['because']}")
        for name, c in (a.get("per_condition") or {}).items():
            if "answer" in c:
                print(f"  {name}: {c['answer']}")
            else:
                share = c.get("certified_share_of_condition")
                print(f"  {name}: " + (f"{100 * share:.1f}% of this condition certified" if share else
                                       f"nothing certified ({c.get('reason') or 'no zone passed its test'})"))
        for c, row in (a.get("per_class") or {}).items():
            iv = (f" (95% interval {100 * row['low']:.1f}% to {100 * row['high']:.1f}%)" if row.get("low") is not None
                  else f" ({row.get('because', 'no interval')})")
            print(f"  class {c}: {row['answer']}{iv}" + ("  [warning]" if row.get("warning") else ""))
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
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
        cmd += ["--nodata", repr(args.nodata)]
    if args.logits:
        cmd += ["--logits"]
    if other and args.threshold is not None:
        cmd += ["--threshold", repr(args.threshold)]
    if getattr(args, "confidence", None):
        cmd += ["--confidence", args.confidence]
    if getattr(args, "confidence_range", None):
        cmd += ["--confidence-range", *[repr(v) for v in args.confidence_range]]
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
    if not (want_rate or want_which or want_zone):
        raise SystemExit("plan: say what to plan: --width (an error rate, its interval no wider than this), "
                         "--difference with --other (which of two maps), or --coverage with --alpha (a certified zone)")
    if want_zone and (args.coverage is None or args.alpha is None):
        raise SystemExit("plan: a certified zone needs both --coverage (a level of the grid, 0.05 to 1 in steps of 0.05) "
                         "and --alpha (the error rate the zone must not exceed)")
    if want_which and args.other is None and args.differing is None:
        raise SystemExit("plan: --difference compares two maps: give the second with --other, or the windows where they "
                         "differ with --differing and --windows")
    two = args.other is not None or args.differing is not None
    if two and (want_rate or want_zone):
        raise SystemExit("plan: --other and --differing plan which of two maps is more accurate; plan an error rate or a "
                         "zone of one map in a separate call")
    if two and not want_which:
        raise SystemExit("plan: --other and --differing go with --difference")
    unused = [flag for flag, value, needs in (
        ("--error-rate", args.error_rate, want_rate), ("--both-wrong", args.both_wrong, want_which),
        ("--zone-error", args.zone_error, want_zone), ("--delta", args.delta, want_zone),
        ("--threshold", args.threshold, args.other is not None),
        ("--two-class", args.two_class or None, want_which)) if value is not None and not needs]
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
        if want_rate or want_zone:
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
        if want_zone:
            delta = 0.1 if args.delta is None else args.delta
            r = plan.plan_zone(N, args.coverage, args.alpha, args.zone_error, power, delta, max_labels=top_asked)
            results["zone"] = r
            head = (f"Zone: {100 * r['coverage']:.3g}% of the map ({r['zone_windows']} of {N} windows) at alpha "
                    f"{args.alpha:g}" + (f" is tested from {_labels(r['labels_to_test'])} drawn at random; fewer cannot "
                                         f"certify it even with no error among them." if r.get("labels_to_test")
                                         else ":"))
            if "refusal" in r:
                print(f"{head} {r['refusal'][0].upper() + r['refusal'][1:]}.")
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
                print(f"  Bonferroni by budget: {_ladder_text(b['probability_by_budget'], b['labels'], power)}")
                print(f"  prefix by budget (-: below {power:g} however the errors spread): "
                      f"{_ladder_text(ev['probability_by_budget'], ev['labels'], power)}")
                if b["labels"] is not None or ev["labels"] is not None:
                    runs.append(_sample_command(args, max(v for v in (b["labels"], ev["labels"]) if v is not None),
                                                other=False))
            if r.get("entry_budgets"):
                notes.append("At " + ", ".join(f"{e} labels the {float(g):.0%} zone" for g, e in r["entry_budgets"].items())
                             + " becomes testable; from each of these budgets both rules can lose power until more "
                             "labels restore it: the prefix rule tests the smallest testable zone first, with few "
                             "labels, and the Bonferroni rule splits delta over one more zone. The budget one below "
                             "each is the last before the fall")
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
            json.dump({"plans": results, "notes": notes, "run": runs}, f, indent=1, default=float)
        print(f"wrote {out_path}")
    return 0


def _skipped_text(skipped):
    """The windows a reader left out, counted by reason; the JSON lists each."""
    if not skipped:
        return ""
    by = {}
    for s in skipped:
        by[s["reason"]] = by.get(s["reason"], 0) + 1
    return f"; {len(skipped)} skipped: " + "; ".join(f"{n} {why}" for why, n in by.items())


def cmd_from_olmoearth(args):
    """An OlmoEarth run's rslearn dataset in, the rasters `assess` reads out (oe_inferencex.olmoearth): the output
    layer's probabilities pasted onto one grid per CRS, and with --conditions the input-condition layer on the same
    grid; then the assess command that reads them."""
    from oe_inferencex import olmoearth as oe
    _check_out_dir(args.out, "from-olmoearth")
    if args.inputs and not args.conditions:
        raise SystemExit("from-olmoearth: --inputs names the input layers of the condition layer; add --conditions")
    try:
        kind = oe.output_kind(args.ds, args.layer, args.group, args.window)
        if kind == "vector":
            if args.conditions:
                raise SystemExit(f"from-olmoearth: layers/{args.layer} is a per-window classification (data.geojson); "
                                 "the condition layer is pasted onto a raster's grid, and this output has none")
            s = oe.read_window_probs(args.ds, args.out, layer=args.layer, prob_property=args.prob_property or "probs",
                                     group=args.group, windows=args.window, class_property=args.class_property)
        else:
            if args.prob_property or args.class_property:
                flag = "--prob-property" if args.prob_property else "--class-property"
                raise SystemExit(f"from-olmoearth: {flag} reads a per-window classification (data.geojson); "
                                 f"layers/{args.layer} holds GeoTIFFs")
            s = oe.read_output(args.ds, args.out, layer=args.layer, group=args.group, windows=args.window)
            c = oe.condition_from_inputs(args.ds, args.out, inputs=args.inputs, layer=args.layer,
                                         group=args.group, windows=args.window) if args.conditions else None
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"from-olmoearth: {exc}") from None
    what = (f"probabilities over {s['bands']} classes, {s['features']} predictions" if kind == "vector" else
            f"probabilities over {s['bands']} classes" if s["bands"] > 1 else "one band, the probability of one class")
    lines = [f"read {s['windows_read']} windows of {os.path.join(args.ds, 'windows')} (layer {args.layer}): {what}"
             + _skipped_text(s["windows_skipped"])]
    notes = list(s["notes"])
    if kind == "vector":
        lines += [f"wrote {s['scores']}, ({s['bands']}, 1, {s['features']}), and {s['index']}, which names each column",
                  f"next: {s['assess']}"]
        written = [oe.SCORES_JSON]
    else:
        conds = {g["label"]: g for g in c["grids"]} if c else {}
        for g in s["grids"]:
            lines.append(f"wrote {g['scores']}: {g['shape'][0]} x {g['shape'][1]} px in {g['crs']}, "
                         f"{len(g['windows'])} windows, {g['covered_pixels']} pixels predicted")
            if g["label"] in conds:
                k = len(conds[g["label"]]["codes"])
                lines.append(f"wrote {conds[g['label']]['condition']}: {k} input condition{'s' if k != 1 else ''} "
                             "(the codes and their rule are in " + oe.CONDITIONS_JSON + ")")
        lines += [f"next: {conds[g['label']]['assess'] if g['label'] in conds else g['assess']}" for g in s["grids"]]
        written = [oe.SCORES_JSON] + ([oe.CONDITIONS_JSON] if c else [])
        notes += c["notes"] if c else []
    print("\n".join(lines + [f"note: {n}" for n in notes] + [f"wrote {os.path.join(args.out, written[0])}"
                                                               + "".join(f", {w}" for w in written[1:])]))
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


def _product_parser_args(p):
    p.add_argument("--confidence", default=None, metavar="BAND",
                   help="a per-pixel confidence band on the map's grid that rises with confidence, as published "
                        "products ship it (LCMAP's lcpconf beside lcpri); negate an uncertainty band first. The map "
                        "argument is then the class map of integer ids, and --nodata applies to it; the band's no-data "
                        "is its own file's")
    p.add_argument("--confidence-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="the band's values that are confidences; values outside are left out as no-data, and every "
                        "statement is about the rest. Without it the package cannot tell codes from confidences "
                        "(LCMAP: 1 100, since lcpconf holds provenance codes from 151)")


def build_parser():
    p = argparse.ArgumentParser(prog="oe-inferencex", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assess", help="rank the windows of one prediction map for review and explain them")
    a.add_argument("scores", help="raster or .npy: (H, W) binary probability or logit map, or (C, H, W) per-class scores")
    a.add_argument("--out", required=True, help="output directory")
    a.add_argument("--logits", action="store_true",
                   help="the scores are logits (tie-free confidence); default probabilities. A multi-class logit map is "
                        "ranked by its logit margin; for a top-probability reading, pass the class probabilities "
                        "instead (in Python, form='top1' gives one from the logits)")
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
    d = sub.add_parser("demo", help="first run: audit the real sample map shipped with the package (or a made-up one) and draw the result")
    d.add_argument("--out", default="oe_inferencex_demo", help="output directory (default oe_inferencex_demo)")
    d.add_argument("--seed", type=int, default=0, help="seed of the made-up scene and of the random pick")
    d.add_argument("--made-up", action="store_true", help="audit a small synthetic water map instead of the real sample tile")
    d.set_defaults(func=cmd_demo)
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
    sm.add_argument("--budget", type=int, required=True, help="number of windows to label (exp78 measured 300)")
    sm.add_argument("--out", required=True, help="CSV to write; a .json sidecar with the design goes beside it")
    sm.add_argument("--design", choices=("confidence", "proportional", "random", "sequential", "tiles", "condition"),
                    default=None,
                    help="confidence (default without --condition): stratified by margin, allocated from the model's "
                         "own confidence; tiles: how people actually label, with the cluster interval that requires; "
                         "condition (default with --condition): stratified by input condition, labels split equally, "
                         "never from the model's confidence; random takes --condition too and only records it; "
                         "sequential: a random order labelled from the top, to which labels can be added (--extend), "
                         "and certify then holds at every look")
    sm.add_argument("--anchor", type=float, default=None,
                    help="sequential design: the smallest zone certify will test, as a share of the map, fixed now "
                         "before any label; or give --alpha instead")
    sm.add_argument("--alpha", type=float, default=None,
                    help="sequential design, without --anchor: the error rate certify will be asked about; the anchor "
                         "is then the zone this budget can certify if none of its labels is wrong, at most a quarter "
                         "of the map, fixed now and written to the sidecar")
    sm.add_argument("--delta", type=float, default=None,
                    help=f"sequential design, with --alpha: the delta the anchor is computed for (default {est.ZONE_DELTA})")
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
    sm.add_argument("--other", default=None, metavar="MAP_B",
                    help="a second map of the same grid: sample at random among the windows where the two maps' "
                         "classes differ, to learn which map is more accurate (estimate then compares them)")
    sm.add_argument("--threshold", type=float, default=None,
                    help="with --other: cut-off of a 2-D continuous map (0.5 for a probability map when omitted)")
    _product_parser_args(sm)
    sm.set_defaults(func=cmd_sample)
    e = sub.add_parser("estimate", help="the map's error rate with an interval, from the labelled sample CSV, and each "
                                        "input condition's when the sample was drawn with --condition")
    e.add_argument("sample", help="the CSV `sample` wrote, with its `wrong` column filled in")
    e.add_argument("--out", default=None, help="JSON to write (default: <sample>_estimate.json)")
    e.add_argument("--per-class", action="store_true",
                   help="also user's accuracy, producer's accuracy and error-adjusted share per class; needs a "
                        "`reference_class` column in the CSV and the map's scores (from the sidecar, or --scores)")
    e.add_argument("--scores", default=None, help="with --per-class: the raster `sample` was run on, if it has moved")
    e.add_argument("--confidence", default=None, help="with --per-class: the confidence band `sample` was run on, if it "
                                                      "has moved")
    e.add_argument("--reviewer-false-alarm", type=float, default=None, metavar="E0",
                   help="at most this share of the truly correct windows does the reviewer mark wrong (0 to below 1); "
                        "widens the interval's lower end")
    e.add_argument("--reviewer-miss", type=float, default=None, metavar="E1",
                   help="at most this share of the truly wrong windows does the reviewer mark right (0 to below 1); "
                        "widens the interval's upper end")
    e.add_argument("--nodata", type=float, default=None,
                   help="with --per-class: the no-data value `sample` was run with (default: the one its sidecar records; "
                        "a different value is refused); only needed for a sample written by 1.2.0")
    e.set_defaults(func=cmd_estimate)
    z = sub.add_parser("certify", help="which share of the map, from the most confident window down, is wrong at most "
                                        "alpha of the time, with a guarantee (needs a random sample; a sample drawn "
                                        "with --condition is certified per input condition)")
    z.add_argument("sample", help="the CSV `sample --design random` (or `sample --condition`) wrote, with its `wrong` "
                                  "column filled in")
    z.add_argument("--alpha", type=float, required=True, help="the error rate the certified zone may not exceed, e.g. 0.05")
    z.add_argument("--delta", type=float, default=est.ZONE_DELTA,
                   help=f"the probability the statement is allowed to be wrong (default {est.ZONE_DELTA}); for a sample "
                        "drawn with --condition, that any of the per-condition statements is, split over the conditions "
                        "tested")
    z.add_argument("--rule", choices=("prefix", "bonferroni"), default="prefix",
                   help="prefix (default): fixed-sequence testing, valid on any map; it certifies little when the most "
                        "confident windows hold many errors. bonferroni: valid on any map; it can certify more in that case")
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
                                     "stated width, which of two maps is more accurate, or a certified zone")
    pl.add_argument("scores", nargs="?", default=None,
                    help="the map, as sample takes it (not needed with --windows)")
    pl.add_argument("--other", default=None, metavar="MAP_B",
                    help="a second map of the same grid: plan the labels that say which map is more accurate")
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
    fo = sub.add_parser("from-olmoearth", help="read an OlmoEarth run (the rslearn dataset olmoearth_run writes) into "
                                               "the rasters assess reads, and print the assess command")
    fo.add_argument("ds", help="the rslearn dataset directory, which holds windows/<group>/<name>/")
    fo.add_argument("--out", required=True, help="output directory")
    fo.add_argument("--layer", default="output", help="the output layer (default output)")
    fo.add_argument("--group", nargs="+", default=None, help="read only these window groups (default: every one)")
    fo.add_argument("--window", nargs="+", default=None, metavar="PATTERN",
                    help="read only the windows whose id group/name matches one of these shell patterns (quote them), "
                         "to read a large area in parts")
    fo.add_argument("--conditions", action="store_true",
                    help="also write the input-condition layer on the scores' grid: per pixel, how many of the run's "
                         "timesteps a scene covered and, with an SCL band, how many were cloudy")
    fo.add_argument("--inputs", nargs="+", default=None, metavar="LAYER",
                    help="with --conditions: the input layers to read (default: the layers config.json gives a "
                         "data_source)")
    fo.add_argument("--prob-property", default=None,
                    help="a per-window classification: the feature property holding the probabilities (default probs)")
    fo.add_argument("--class-property", default=None,
                    help="a per-window classification: the feature property holding the class the task wrote, checked "
                         "against the argmax (default: the one other property, if there is one)")
    fo.set_defaults(func=cmd_from_olmoearth)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

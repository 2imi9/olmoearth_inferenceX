"""At which window size each class of a map can be seen: the same map assessed at several window sizes.

A window is the unit the package ranks, samples and grades, and its class is the majority of its pixels' classes. A
class whose features are small against the window (a road a pixel wide, a pond, a hedge) loses that majority as the
window grows: its pixels sit in windows called something else, and a sample of windows that size grades only the part
of it that still holds a majority. `scale_profile` assesses one map at each window size and says how much of each
class survives (the share of the pixels the map calls c that sit in a valid window called c; a pixel outside the
window grid or in a window that is mostly no-data sits in none), how many windows border a window of another class and
how confident the windows are. With a reference raster on the same grid it adds each size's error rate, per-class
accuracy and how well the confidence ranks the errors: a census of the reference, not a sample, so it has no interval
and is only as right as the reference.

It zooms out only. The finest window is the sensor's pixel, and a class too small for it needs finer imagery; and a
class's survival is about where the map draws the class, not where the class is."""
import numpy as np

from .assess import (NO_VALID_WINDOW, _check_shape, _masked_as_nodata, _pool, _pooled_argmax, _reference_index,
                     assess_classmap, assess_prediction)

PATCHES = (1, 2, 4, 8, 16)
# A descriptive line, not a test: below it most of the pixels the map calls c sit in windows called another class.
SURVIVAL_LINE = 0.5
CENSUS_NOTE = ("Against the reference this is a census of the reference raster, not a sample: it has no interval and "
               "is only as right as the reference (reference-product labels can flatter boundary-type signals, exp18). "
               "A guaranteed error rate at a chosen window size needs a random sample drawn and labelled at that size.")
# A class pooled away in the map is pooled away in a reference that draws it as small, so it leaves the graded windows
# and the error rate does not show the loss; survival does (a one-pixel road on a synthetic map: 0% error at every
# size, with its survival 0.00 from 4 px).
UNSEEN_NOTE = ("Where the reference draws a class as small as the map does, the class drops out of the reference's "
               "windows too once most of its pixels sit in windows called another class, and the error rate does not "
               "show its loss; its survival does.")
ZOOM_NOTE = ("Each window size is the map's own pixels pooled: the profile zooms out only. A class too small for the "
             "sensor's pixel needs finer imagery, and survival describes where the map draws a class, not where the "
             "class is.")
NO_WINDOW_NOTE = ("no window of {p} px is at least half valid, so no pixel sits in a window at that size: every "
                  "class's survival there is 0")


def pixel_classes(scores, is_logit):
    """The map's class at each pixel, as assess_prediction decides it: one band is class 1 above 0 (logits) or 0.5
    (probabilities), several bands are the argmax. Returns (classes, n_classes)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 2:
        return (s > (0.0 if is_logit else 0.5)).astype(int), 2
    return np.nan_to_num(s, nan=-np.inf).argmax(0), s.shape[0]


def _check_patches(patches, shape):
    """The window sizes to use, sorted and unique, and the ones larger than the map, which are left out and said."""
    ps = sorted({int(p) for p in patches})
    if not ps:
        raise ValueError("give at least one window size")
    if ps[0] < 1:
        raise ValueError(f"window sizes must be at least 1 px, got {ps[0]}")
    fit = [p for p in ps if p <= min(shape)]
    if not fit:
        raise ValueError(f"every window size is larger than the map, {shape[0]} x {shape[1]} px")
    return fit, [p for p in ps if p not in fit]


def _surviving(hard, valid, pooled, valid_w, p, classes):
    """{class: (pixels the map calls c that sit in a valid window called c, pixels of c in a valid window called
    another class)}. The rest of c's pixels sit in no valid window: outside the window grid, or in a window that is
    mostly no-data."""
    h, w = pooled.shape[0] * p, pooled.shape[1] * p
    up = np.repeat(np.repeat(pooled, p, 0), p, 1)
    up_ok = np.repeat(np.repeat(valid_w, p, 0), p, 1)
    hc, vc = hard[:h, :w], valid[:h, :w]
    return {c: (int((vc & (hc == c) & up_ok & (up == c)).sum()), int((vc & (hc == c) & up_ok & (up != c)).sum()))
            for c in classes}


def _reference_block(out, reference, p, n_classes):
    """Per-class window accuracy against the reference, scored as assess scores it: the reference pooled over its own
    class range with an evenly split window left unscored, a window graded where both hold and the reference has a
    majority."""
    ref, codes = _reference_index(reference, n_classes)     # remapped, so memory follows the classes present
    ref_w = _pooled_argmax(ref, len(codes), p, empty=-1, tie=-1)
    pooled, valid_w = out["arrays"]["pooled_argmax"], out["arrays"]["valid"]
    scored = valid_w & (_pool((ref >= 0).astype(float), p) >= 0.5) & (ref_w >= 0)
    rw, mw = ref_w[scored], pooled[scored]
    per = {}
    for i, c in enumerate(codes):
        in_ref, in_map = rw == i, mw == i
        if in_ref.any() or in_map.any():
            per[int(c)] = {"windows_in_reference": int(in_ref.sum()), "windows_in_map": int(in_map.sum()),
                           "producer_accuracy": float((mw[in_ref] == i).mean()) if in_ref.any() else None,
                           "user_accuracy": float((rw[in_map] == i).mean()) if in_map.any() else None}
    return per


def scale_profile(scores=None, is_logit=False, hard=None, confidence=None, n_classes=None, patches=PATCHES,
                  nodata_mask=None, reference=None):
    """One map assessed at each window size in `patches` (pixels). Give per-class `scores` (as assess_prediction
    takes them, with `is_logit`), or a product's class map `hard` with its per-pixel `confidence` band and
    `n_classes` (as assess_classmap takes them). `reference`, optional, is an integer class raster on the same grid,
    negative or NaN where there is no label.

    Returns {"patches", "per_patch", "per_class", "skipped", "warnings", "notes"}: per window size the number of
    windows, the share that border a window of another class (assess's boundary windows), the confidence quantiles,
    each class's share of the windows and, with a reference, the error rate, the excess AURC, the share of the errors
    in a 5% review and each class's window accuracy; per class the valid pixels the map calls it, its survival at
    each size, the shares of them in windows called another class and in no valid window (outside the window grid,
    or in a window that is mostly no-data), and the first size at which the survival falls below SURVIVAL_LINE
    (survival need not fall steadily: a plurality can return at a larger size). A size with no window at least half
    valid is kept, with no windows and every survival 0. `warnings` holds each assess warning once, with the sizes it
    was given at."""
    if (scores is None) == (hard is None):
        raise ValueError("give either scores or a class map with its confidence band")
    if hard is not None and (confidence is None or n_classes is None):
        raise ValueError("a class map needs its confidence band and the number of classes")
    # A masked array's mask is no-data, as assess reads it (the bug hunt of 2026-10-25): dropped, its fill was ranked.
    # The shapes are checked first, so that a mask of the wrong shape is not broadcast into it (the review of 1.17.2).
    shape = tuple(np.shape(scores)[-2:]) if scores is not None else tuple(np.shape(hard))
    if hard is not None:
        _check_shape("confidence band", np.shape(confidence), shape)
    for name, layer in (("no-data mask", nodata_mask), ("reference", reference)):
        if layer is not None:
            _check_shape(name, np.shape(layer), shape)
    if scores is not None:
        scores, nodata_mask = _masked_as_nodata(scores, nodata_mask)
        px_hard, n_cls = pixel_classes(scores, is_logit)
    else:
        hard, nodata_mask = _masked_as_nodata(hard, nodata_mask)
        confidence, nodata_mask = _masked_as_nodata(confidence, nodata_mask)
        px_hard, n_cls = np.asarray(hard).astype(int), int(n_classes)
    fit, skipped = _check_patches(patches, shape)
    valid = np.ones(shape, dtype=bool) if nodata_mask is None else ~np.asarray(nodata_mask, dtype=bool)
    if scores is not None:
        s = np.asarray(scores, dtype=np.float64)
        valid &= np.isfinite(s).all(0) if s.ndim == 3 else np.isfinite(s)
    else:
        valid &= np.isfinite(np.asarray(confidence, dtype=np.float64)) & (px_hard >= 0)
    totals = {c: int((valid & (px_hard == c)).sum()) for c in range(n_cls)}
    classes = [c for c, n in totals.items() if n]
    per_patch, kept, warn_at, notes = {}, {}, {}, [ZOOM_NOTE]
    for p in fit:
        try:
            if scores is not None:
                out = assess_prediction(scores, is_logit=is_logit, patch=p, nodata_mask=~valid, reference=reference,
                                        budgets=(0.05,))
            else:
                out = assess_classmap(px_hard, confidence, n_cls, patch=p, nodata_mask=~valid, reference=reference,
                                      budgets=(0.05,))
        except ValueError as exc:
            if str(exc) != NO_VALID_WINDOW:
                raise
            per_patch[p] = {"n_windows": 0, "note": NO_WINDOW_NOTE.format(p=p)}
            kept[p] = {c: (0, 0) for c in classes}
            notes.append(NO_WINDOW_NOTE.format(p=p))
            continue
        for w in out["warnings"]:
            warn_at.setdefault(w, []).append(p)
        row = {"n_windows": out["n_windows"], "edge_window_share": out["boundary_window_fraction"],
               "confidence_quantiles": out["confidence_quantiles"],
               "class_share_of_windows": {c: v for c, v in out["class_share"].items() if v > 0},
               "pixels_outside_window_grid": out["pixels_outside_window_grid"]}
        ref = out.get("against_reference")
        if ref is not None:
            row["against_reference"] = {
                "n_windows_scored": ref["n_windows_scored"], "error_rate": ref["error_rate"],
                "excess_aurc_confidence": ref.get("excess_aurc_confidence"),
                # with no error the share of the errors captured is undefined (assess writes NaN there since 1.17.2)
                "errors_in_5pct_review": None if not ref["error_rate"] else
                (ref.get("error_capture_at_budget") or {}).get(0.05, {}).get("errors_captured_fraction"),
                "per_class": _reference_block(out, reference, p, n_cls) if ref["n_windows_scored"] else {}}
        per_patch[p] = row
        kept[p] = _surviving(px_hard, valid, out["arrays"]["pooled_argmax"], out["arrays"]["valid"], p, classes)
    if all(per_patch[p]["n_windows"] == 0 for p in fit):
        raise ValueError(NO_VALID_WINDOW)
    per_class = {}
    for c in classes:
        surv = {p: kept[p][c][0] / totals[c] for p in fit}
        other = {p: kept[p][c][1] / totals[c] for p in fit}
        below = [p for p in fit if surv[p] < SURVIVAL_LINE]
        per_class[c] = {"pixels": totals[c], "survival": surv, "in_windows_of_another_class": other,
                        "in_no_valid_window": {p: 1.0 - surv[p] - other[p] for p in fit},
                        "first_below_half": below[0] if below else None}
    if reference is not None:
        notes.append(CENSUS_NOTE)
        if any(v >= SURVIVAL_LINE for e in per_class.values() for v in e["in_windows_of_another_class"].values()):
            notes.append(UNSEEN_NOTE)
    if skipped:
        notes.append(f"window sizes larger than the map ({', '.join(str(p) for p in skipped)} px) were left out")
    warnings = [{"warning": w, "window_sizes": ps} for w, ps in warn_at.items()]
    return {"patches": fit, "per_patch": per_patch, "per_class": per_class, "skipped": skipped,
            "survival_line": SURVIVAL_LINE, "warnings": warnings, "notes": notes}

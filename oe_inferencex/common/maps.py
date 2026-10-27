"""A score or class map read as decisions: the probability check, and per-pixel decisions with their confidence."""
import os

import numpy as np

from oe_inferencex.common.raster import read_raster

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
            f"when the band also holds codes). A regression output (a height, a depth, an index) is none of these: "
            f"it has no margin, so no review set or error rate is read from it; compare cuts one into classes at "
            f"--threshold.")
    if scores.ndim == 3:
        return
    integral = np.array_equal(v, np.rint(v))
    if integral and len(np.unique(v)) > 2:
        raise SystemExit(
            f"{path}: {len(np.unique(v))} distinct integer values in [0, 1] is a class map, not a probability map. "
            f"For a hard class map with its confidence band, pass the band with --confidence BAND.")


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

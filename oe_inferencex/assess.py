"""Assess a prediction output without labels (Layer 1: pure, no network).

The recipe that the experiments support (docs/method/recipe.md): rank
windows by the model's own confidence, use prediction-boundary proximity as
a triage cue, report operating points, and state the caveats. This module
turns a prediction array into that assessment. It generates evidence only;
narration belongs to the caller.

Inputs
    scores      : (C, H, W) logits or probabilities for C classes, or
                  (H, W) probability of the positive class for binary tasks
    is_logit    : whether `scores` are logits (preferred: confidence is then
                  the top-1 minus top-2 logit margin, tie-free) or
                  probabilities (confidence is 1 - max probability, which
                  ties where probabilities saturate)
    patch       : pooling size in pixels for the per-window ranking
    nodata_mask : optional (H, W) boolean, True where no prediction exists
    reference   : optional (H, W) integer class map treated as truth; when
                  given, the assessment also reports risk-coverage against
                  it (with the reference caveat attached)
    budgets     : review budgets (fractions of windows) at which to report
                  the flagged set and, with a reference, the error capture

Output: a dict of summary statistics plus per-window arrays. The arrays are
returned so the caller can write them to files and pass handles onward;
nothing here serializes them into text.
"""
import warnings

import numpy as np

from oe_inferencex.metrics import aurc_expected
from oe_inferencex.signals import boundary_indicator, midrank_pct


def _pool(a, patch):
    h, w = a.shape[0] // patch * patch, a.shape[1] // patch * patch
    return a[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3))


def _pool_valid(a, patch):
    """Mean over the VALID pixels of each window, NaN where a window has none.

    A window is the unit this package ranks, and a pixel with no prediction carries no evidence about it. Filling those
    pixels with any constant before a plain mean puts that constant's opinion into the window's score: filling with the
    scene maximum made a window that was 37.5% no-data read as nine times more confident than the same window fully
    observed, which pushed partially-observed windows down the review list exactly where an operator most needs to
    look. No-data at scene edges and under cloud is ubiquitous in Earth observation, so this was not a corner case."""
    h, w = a.shape[0] // patch * patch, a.shape[1] // patch * patch
    blocks = a[:h, :w].reshape(h // patch, patch, w // patch, patch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a wholly invalid window is NaN by design
        return np.nanmean(blocks, axis=(1, 3))


def _pooled_argmax(hard, n_classes, patch, empty=0, weights=None, tie=None):
    """Majority class per window, counting votes only over range(n_classes).

    `empty` is returned for a window with no vote at all. It defaults to 0 for the callers that only ever look at
    valid windows, but _assess passes -1: a window under cloud or off the edge of the scene has no class, and until
    2026-09-21 it was given class 0, which made every such window disagree with its neighbours and manufactured a
    prediction boundary around every no-data hole. On a map predicting one class everywhere it had a prediction,
    that reported a boundary window fraction of 16.7% and filled the boundary-first review set with the rim of the
    data."""
    h, w = hard.shape[0] // patch * patch, hard.shape[1] // patch * patch
    def _blocks(a):
        return a[:h, :w].reshape(h // patch, patch, w // patch, patch).transpose(0, 2, 1, 3).reshape(h // patch, w // patch, -1)
    blocks = _blocks(hard)
    counts = np.stack([(blocks == c).sum(-1) for c in range(n_classes)], axis=-1)
    res = counts.argmax(-1)
    # A tied majority (8 of 16 against 8) used to go to the lowest class index, every time: on a balanced
    # two-class map class 0 was reported at 0.596 of the windows against a true 0.502 (audit 2026-09-21, finding
    # 10). A prediction's tie now goes to the class whose voting pixels are more confident (`weights`, higher =
    # more confident); a reference's tie has no majority and is returned as `tie`, so the caller can leave it
    # unscored rather than grade the map against class 0.
    top = counts.max(-1, keepdims=True)
    tied = ((counts == top).sum(-1) > 1) & (top[..., 0] > 0)
    if tied.any() and weights is not None:
        wb = _blocks(np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0))
        wsum = np.stack([np.where(blocks == c, wb, 0.0).sum(-1) for c in range(n_classes)], axis=-1)
        res = np.where(tied, np.where(counts == top, wsum, -np.inf).argmax(-1), res)
    if tie is not None:
        res = np.where(tied, tie, res)
    return np.where(counts.sum(-1) > 0, res, empty)


def _boundary_valid(pooled_hard, valid):
    """boundary_indicator, except that a neighbour with no prediction cannot disagree with anything.

    The denominator stays eight, exactly as signals.boundary_indicator defines it, so on a fully observed map this
    is bit-identical to it and every recorded boundary number (exp37's cue enrichments, exp36's order) is unchanged.
    The only difference is that a no-data neighbour no longer counts as a differing class: until 2026-09-21 a window
    beside a cloud hole or the edge of the data was scored as though the hole disagreed with it, which on a map
    predicting a single class everywhere it had a prediction reported a boundary window fraction of 16.7% and filled
    the boundary-first review set with the rim of the data."""
    if valid.all():
        return boundary_indicator(pooled_hard)
    diff = np.zeros(pooled_hard.shape, np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.roll(np.roll(pooled_hard, dy, 0), dx, 1)
            nv = np.roll(np.roll(valid, dy, 0), dx, 1)
            inb = np.ones(pooled_hard.shape, bool)          # edge padding, as boundary_indicator does it
            if dy:
                inb[0 if dy > 0 else -1, :] = False
            if dx:
                inb[:, 0 if dx > 0 else -1] = False
            diff += inb & nv & (nb != pooled_hard)
    return np.where(valid, diff / 8.0, 0.0)


ORDERS = ("confidence", "boundary_first")


def boundary_first_score(suspicion, boundary):
    """The review order supported by exp36: boundary windows first, ordered by suspicion, then the interior by suspicion.

    2 * [boundary > 0] + the within-map midrank percentile of suspicion, the construction exp36 and exp38 tested; higher
    is reviewed first. Ties are broken by review_order as for any score."""
    s_ = np.asarray(suspicion, dtype=np.float64)
    return np.where(np.asarray(boundary) > 0, 2.0, 0.0) + midrank_pct(s_).reshape(s_.shape)


def assess_classmap(hard, confidence, n_classes, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10),
                    signal="exported top-1 probability", order="confidence"):
    """Production case: a hard class map plus an exported per-pixel confidence
    band (for instance the top-1 probability bands of the LCC rasters), with
    no logits. Ties in `confidence` are reported, because a quantized or
    saturated band can only rank the pixels it separates."""
    hard = np.asarray(hard).astype(int)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = ~nodata_mask if nodata_mask is not None else np.ones(conf.shape, dtype=bool)
    vals, counts = np.unique(conf[valid], return_counts=True)
    warnings = [f"confidence band has {len(vals)} distinct values; {counts.max() / counts.sum():.3f} of pixels share the modal value {vals[counts.argmax()]:.4g}"]
    out = _assess(conf, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings, order)
    out["confidence_distinct_values"] = int(len(vals))
    out["confidence_modal_share"] = float(counts.max() / counts.sum())
    return out


def assess_prediction(scores, is_logit, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10), order="confidence",
                      form="margin"):
    """Assess a prediction map: `scores` is (H, W) of binary logits or probabilities, or (C, H, W) per class.

    `order` is the review order of the review sets: "confidence" (least confident first, the ranker every
    experiment scored) or "boundary_first" (boundary windows first, then the interior, each by confidence; the
    order that captures more errors at 5-10% budgets on hand labels, exp36). AURC entries always score confidence.

    `form` is the member of the confidence family used on a (C, H, W) logit map: "margin", the top-1 minus top-2
    logit margin (the default, unchanged since 1.0.0), or "top1", the top softmax probability computed tie-free
    from the logits. On the 16 multi-class tasks of Ai2's suite "top1" ranked errors better than the margin on 14
    and the logit margin was the weakest form on all 16 (exp76), so a multi-class logit map scored with the default
    carries a warning. Binary maps and probability input are unaffected: there the forms are one ranking, and
    probability input already uses the top probability."""
    if form not in ("margin", "top1"):
        raise ValueError(f"form must be 'margin' or 'top1', got {form!r}")
    scores = np.asarray(scores, dtype=np.float64)
    if not is_logit:
        _check_probabilities(scores, nodata_mask)
    warnings = []
    # Non-finite pixels carry no prediction. Until 2026-09-21, with no explicit nodata_mask they were ranked like
    # any other window and NaN sorts to the front of the review order, so a scene with a NaN strip returned a review
    # set that was entirely empty pixels, with healthy-looking confidence quantiles and no warning.
    nonfinite = ~np.isfinite(scores)
    if nonfinite.any():
        implied = nonfinite.any(0) if scores.ndim == 3 else nonfinite
        n_nf = int(implied.sum())
        if nodata_mask is None:
            nodata_mask = implied
            warnings.append(f"{n_nf} pixels are not finite and were treated as no-data; pass nodata_mask to say so explicitly")
        elif (implied & ~np.asarray(nodata_mask, bool)).any():
            nodata_mask = np.asarray(nodata_mask, bool) | implied
            warnings.append(f"{int((implied & ~np.asarray(nodata_mask, bool)).sum())} non-finite pixels outside the given "
                            f"nodata_mask were added to it")
        scores = np.where(nonfinite, 0.0, scores)
    if scores.ndim == 2:  # binary probability map
        p1 = scores
        if is_logit:
            margin = np.abs(p1)
            hard = (p1 > 0).astype(int)
        else:
            margin = np.abs(p1 - 0.5) * 2
            hard = (p1 > 0.5).astype(int)
            warnings.append("probability input: confidence ties where probabilities saturate; prefer logits")
        n_classes = 2
    else:
        C = scores.shape[0]
        if C == 1:
            raise ValueError(
                "a (1, H, W) score map has one class, so there is no confidence to rank by. A binary map is (H, W): "
                "pass scores[0]. Until 2026-09-21 this was scored as a one-class map and returned a review set "
                "ordered the wrong way round.")
        srt = np.sort(scores, axis=0)
        if is_logit and form == "top1":
            margin = -np.log1p(np.exp(srt[:-1] - srt[-1]).sum(0))   # log of the top softmax probability, tie-free
        elif is_logit:
            margin = srt[-1] - srt[-2]
            if C > 2:
                warnings.append("multi-class logit margin: on Ai2's suite one minus the top probability ranked errors "
                                "better on 14 of 16 multi-class tasks (exp76); pass form='top1'")
        else:
            margin = srt[-1]  # top-1 probability
            warnings.append("probability input: confidence ties where probabilities saturate; prefer logits")
        hard = scores.argmax(0)
        n_classes = C
    out = _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets,
                  ("1 - max probability (from logits)" if form == "top1" and scores.ndim == 3 else "negative logit margin")
                  if is_logit else "1 - max probability", warnings, order)
    if is_logit and form == "top1" and scores.ndim == 3:
        # The ranking reads the window mean of log p1, which is tie-free; the quantiles a user sets thresholds from
        # must be on the probability scale. Until 2026-09-22 they were reported as log-probabilities, a median
        # "confidence" of -0.620, under a label naming a probability (audit 2026-09-21, finding 12).
        out["confidence_quantiles"] = {q: float(np.exp(v)) for q, v in out["confidence_quantiles"].items()}
        out["confidence_scale"] = "geometric mean of the top-1 probability over the window's pixels"
    return out


def _check_probabilities(scores, nodata_mask):
    """Refuse an array that cannot be a probability map, for every caller and not only the command line. A regression
    output, a reflectance band or a class map passed as probabilities would otherwise be thresholded at 0.5 and come
    back as a full, plausible review set."""
    valid = np.isfinite(scores)
    if nodata_mask is not None:
        valid = valid & ~np.asarray(nodata_mask, dtype=bool)   # (H, W) broadcasts over (C, H, W)
    v = scores[valid]
    if v.size and (v.min() < -1e-6 or v.max() > 1 + 1e-6):
        raise ValueError(
            f"scores run {v.min():g} to {v.max():g}, which is not a probability map. Pass is_logit=True for logits. "
            f"A hard class map with a separate confidence band goes to assess_classmap. A regression output has no "
            f"class confidence and is not supported here; two inferences of it can be compared at a cut-off instead.")


def _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings, order="confidence"):
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    margin = np.asarray(margin, dtype=np.float64)
    hard = np.asarray(hard).astype(int)
    H, W = hard.shape[-2:]
    if patch > H or patch > W:
        raise ValueError(f"the window of {patch} px is larger than the map, {H} x {W} px; pass a smaller --patch")
    if nodata_mask is not None:
        margin = np.where(nodata_mask, np.nan, margin)
        hard = np.where(nodata_mask, -1, hard)  # no-prediction pixels do not vote in pooling or boundaries

    conf_w = _pool_valid(margin, patch) if nodata_mask is not None else _pool(margin, patch)   # higher = more confident
    valid_w = _pool((~nodata_mask).astype(float), patch) >= 0.5 if nodata_mask is not None else np.ones_like(conf_w, dtype=bool)
    pooled_hard = _pooled_argmax(hard, n_classes, patch, empty=-1, weights=margin)
    bnd_w = _boundary_valid(pooled_hard, valid_w)
    suspicion = -conf_w  # ranking signal: low margin first
    review_score = boundary_first_score(suspicion, bnd_w) if order == "boundary_first" else suspicion

    out = {
        "n_windows": int(valid_w.sum()), "patch_px": patch, "n_classes": n_classes,
        "confidence_quantiles": {q: float(np.nanquantile(conf_w[valid_w], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "boundary_window_fraction": float((bnd_w[valid_w] > 0).mean()),
        "class_share": {int(c): float((pooled_hard[valid_w] == c).mean()) for c in range(n_classes)},
        "signal": signal,
        "review_order": order,
        "warnings": warnings,
        "arrays": {"confidence": conf_w, "boundary": bnd_w, "pooled_argmax": pooled_hard, "valid": valid_w},
        "review_sets": {},
        "confidence_distinct_pooled": int(len(np.unique(conf_w[valid_w]))),
    }
    n_valid = int(valid_w.sum())
    if n_valid == 0:
        raise ValueError("the map has no valid window: every window is at least half no-data (a fully clouded or "
                         "fully masked scene); there is nothing to rank")
    # The window grid covers whole windows only; a ragged right or bottom edge is never a candidate. Said, not silent.
    dropped = int(H * W - (H // patch * patch) * (W // patch * patch))
    out["pixels_outside_window_grid"] = dropped
    if dropped:
        warnings.append(f"{dropped} pixels ({100 * dropped / (H * W):.1f}%) on the right and bottom edge do not fill a "
                        f"{patch} px window and are never ranked; choose a patch that divides the map to include them")
    rank = review_order(review_score, valid_w)  # first to review first
    for b in budgets:
        if not (0 < b <= 1):
            raise ValueError(f"budget must be a fraction in (0, 1], got {b}; a budget of 1.0 reviews every window")
        k = min(n_valid, max(1, int(round(b * n_valid))))
        idx = rank[:k]
        rows, cols = np.unravel_index(idx, conf_w.shape)
        out["review_sets"][b] = {"n_windows": int(k), "windows_rowcol": np.stack([rows, cols], 1),
                                 "boundary_share_in_set": float((bnd_w.flatten()[idx] > 0).mean())}
        # A set whose cut-off falls inside a run of equal scores is decided there by raster position, not by evidence:
        # a hard mask, a quantized band or a constant map all end here. Said only when it happens.
        flat = review_score.ravel()
        tied = valid_w.ravel() & (flat == flat[idx[-1]])
        inside = int(tied[idx].sum())
        if int(tied.sum()) > inside:
            out["review_sets"][b]["tied_at_cutoff"] = {"inside": inside, "outside": int(tied.sum()) - inside}
            warnings.append(f"review set at {b:g}: {inside} of its {k} windows share the cut-off score with "
                            f"{int(tied.sum()) - inside} windows left outside; among those the order is raster position, "
                            f"not evidence")

    if reference is not None:
        ref = np.asarray(reference).astype(int)  # values < 0 mean no reference
        ref_valid_w = _pool((ref >= 0).astype(float), patch) >= 0.5
        scored = valid_w & ref_valid_w
        # The reference is pooled over ITS OWN class range, never the prediction's. _pooled_argmax counts votes only
        # over range(n), so pooling a 3-class reference with the prediction's n_classes=2 silently dropped every
        # reference pixel of class 2 and let the window label fall to a surviving low index: measured at an error
        # rate of 0.0625 against an honest 0.1875 on a binary flood model graded against dry/flood/permanent-water,
        # with no warning. The identical defect was found, measured at 44.9% and fixed for `compare --labels`
        # (cli.py) on 2026-09-14 and never carried across to here. Fixed 2026-09-21.
        n_ref = int(ref[ref >= 0].max()) + 1 if (ref >= 0).any() else n_classes
        if n_ref > n_classes:
            warnings.append(f"the reference carries {n_ref} classes and the map predicts {n_classes}; windows whose "
                            f"reference class the map cannot predict are counted wrong")
        ref_w = _pooled_argmax(ref, max(n_ref, n_classes), patch, empty=-1, tie=-1)
        n_ref_tied = int((scored & (ref_w < 0)).sum())
        scored = scored & (ref_w >= 0)                      # a reference with no majority grades nothing
        if n_ref_tied:
            warnings.append(f"{n_ref_tied} windows have an evenly split reference and no majority label; they are "
                            "left unscored")
        if not scored.any():
            out["against_reference"] = {"n_windows_scored": 0, "n_windows_reference_tied": n_ref_tied,
                                        "error_rate": float("nan"),
                                        "note": "no window has both a prediction and a reference with a majority label"}
            return out
        err = (ref_w != pooled_hard).astype(float)
        e, s = err[scored], suspicion[scored]
        r_s = review_score[scored]
        bnd_s = bnd_w[scored]
        rc = {"n_windows_scored": int(scored.sum()), "n_windows_reference_tied": n_ref_tied, "error_rate": float(e.mean()), "aurc_confidence": aurc_expected(s, e),
              "population": "windows with both a prediction and a reference; the capture budgets below are fractions "
                            "of these, not of the review sets above"}
        if int(scored.sum()) < n_valid:
            warnings.append(f"the reference covers {int(scored.sum())} of the {n_valid} valid windows; error capture at "
                            "each budget is measured over those, so it is not a property of the review set of the same "
                            "budget above")
        oracle = aurc_expected(e, e)  # errors most suspicious, so rejected first
        rc["excess_aurc_confidence"] = rc["aurc_confidence"] - oracle
        rc["aurc_boundary"] = aurc_expected(bnd_s, e)
        rc["aurc_random_expected"] = float(e.mean())
        cap = {}
        e_sorted = e[np.argsort(r_s, kind="stable")[::-1]]   # capture follows the review order
        for b in budgets:
            k = max(1, int(round(b * len(e))))
            cap[b] = {"n_reviewed": int(k), "errors_captured_fraction": float(e_sorted[:k].sum() / max(e.sum(), 1)),
                      "precision_in_set": float(e_sorted[:k].mean())}
        rc["error_capture_at_budget"] = cap
        rc["boundary_share_among_errors"] = float((bnd_s[e > 0] > 0).mean()) if e.sum() else float("nan")
        rc["caveat"] = "reference-product labels can flatter boundary-type signals (exp18); treat as expert truth only if it is"
        out["against_reference"] = rc
    return out


def review_order(suspicion, valid=None):
    """Flat window indices in review order: most suspicious first, invalid windows last, ties broken by descending
    raster position (an ascending stable sort, reversed). The one definition of the review order; exp37 and the
    explanation layer use it so that a review set means the same thing everywhere."""
    s = np.asarray(suspicion, dtype=np.float64)
    flat = np.where(np.asarray(valid, dtype=bool), s, -np.inf).ravel() if valid is not None else s.ravel()
    return np.argsort(flat, kind="stable")[::-1]


def review_mask(suspicion, valid, budget):
    """Boolean map of the review set at `budget`: the k = max(1, round(budget * n_valid)) first windows of review_order."""
    valid = np.asarray(valid, dtype=bool)
    order = review_order(suspicion, valid)
    n_valid = int(valid.sum())
    k = min(n_valid, max(1, int(round(budget * n_valid))))   # a budget above 1.0 reviews everything, never more
    m = np.zeros(order.shape, dtype=bool)
    m[order[:k]] = True
    return m.reshape(valid.shape) & valid


def summary(out):
    """JSON-safe view of an assessment: no arrays, string keys, NaN as null,
    review-set windows as [row, col] lists. The arrays stay in `out["arrays"]`
    for callers that write them to files."""
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer, int, bool)):
            return int(o) if not isinstance(o, bool) else o
        return o
    return conv({k: v for k, v in out.items() if k != "arrays"})


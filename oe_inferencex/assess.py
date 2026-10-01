"""Assess a prediction output without labels (Layer 1: pure, no network).

The recipe that the experiments support (docs/method/recipe.md): rank
windows by the model's own confidence, use prediction-boundary proximity as
a triage cue, report operating points, and state the caveats. The ranking
compares every window's confidence with every other's. A model run on an
input combination it was not trained on can be confidently wrong there: on
PASTIS a probe trained on radar plus optical and run on radar alone, as
under cloud, was sure and wrong, while a probe trained on radar alone
ranked its errors normally (exp88). Even a model trained on each input can
be wrong more often under one input than under another. Without a
condition layer every window is ranked with every other. With one, each
condition is also ranked on its own, so that windows are compared only with
windows read from the same inputs. This module turns a prediction array into
that assessment. It generates evidence only; narration belongs to the caller.

Inputs
    scores      : (C, H, W) logits or probabilities for C classes, or
                  (H, W) probability of the positive class for binary tasks
    is_logit    : whether `scores` are logits (confidence is then the top-1
                  minus top-2 logit margin, tie-free, or with form='top1'
                  the top probability) or probabilities (confidence is the
                  top probability, or for a two-class map the distance from
                  0.5, which ties where probabilities saturate). Logits avoid
                  the ties on a two-class map; on a map of more than two
                  classes the logit margin ranked errors worse than the
                  probability forms (exp76), so pass probabilities or use
                  form='top1'
    patch       : pooling size in pixels for the per-window ranking
    nodata_mask : optional (H, W) boolean, True where no prediction exists
    reference   : optional (H, W) integer class map treated as truth; when
                  given, the assessment also reports risk-coverage against
                  it (with the reference caveat attached)
    budgets     : review budgets (fractions of windows) at which to report
                  the flagged set and, with a reference, the error capture
    condition   : optional (H, W) integer layer, the input condition of each
                  pixel (a cloud flag, the modalities present, a sensor id);
                  negative or NaN where none is recorded. Each condition is
                  then ranked and described on its own (pool_condition)
    condition_names : optional {value: name} for the condition values

Output: a dict of summary statistics plus per-window arrays. The arrays are
returned so the caller can write them to files and pass handles onward;
nothing here serializes them into text.
"""
import warnings

import numpy as np

from oe_inferencex.estimate import _condition_index
from oe_inferencex.metrics import aurc_expected
from oe_inferencex.signals import boundary_indicator, midrank_pct

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MAX_CONDITIONS = 64               # distinct condition values a layer may hold; more is a continuous layer, not a category
RULE_TEXT = ("A window takes the condition held by most of its pixels that have a prediction and a recorded condition. "
             "A tie, or no such pixel, makes it 'unrecorded'.")
# What the review order does not say when part of the map was read from inputs the model was not trained on. The
# numbers are exp88's (exp/out/exp88_summary.json) and its matched-head follow-up's (exp/out/exp88_matched_head.json);
# tests/test_assess.py reads them back from there.
SCOPE_ASSESS = ("The review order compares the confidence of every window with every other. Where part of the map was "
                "predicted from an input combination the model was not trained on, the model can be confidently wrong "
                "there, and those errors come late in this order. On PASTIS, OlmoEarth Base's probe trained on radar plus "
                "optical was 73.6% wrong when run on radar alone, as under cloud. Of its errors, 59.8% were at least as "
                "confident as the typical correct window with full input, against 6.0% of its errors with full input. A "
                "probe trained on radar alone was 28.4% wrong and ranked its errors with an AUROC of 0.79, against 0.83 "
                "for the probe trained on both with full input; 3.9% of its errors reached the same threshold, which is "
                "set by the probe trained on both. For OlmoEarth "
                "Large's probe trained on both inputs, the share rose only from 5.7% to 12.8-13.8%. On CropHarvest China "
                "6, Base's probe trained on both showed no such rise (exp88). A model trained on each input can still be "
                "wrong more often under one input condition than under another. If the map records each pixel's input "
                "condition, pass it as condition (--condition) to rank, sample and certify each condition on its own.")
SCOPE_ASSESS_K = ("The review sets above rank all {K} input conditions together. The model's confidence need not mean "
                  "the same thing in each, least of all where the model was not trained on a condition's inputs "
                  "(exp88). conditions.per_condition ranks each condition on its own. In a condition read from inputs "
                  "the model was not trained on, that ranking can be weak too: on PASTIS the margin's AUROC for errors "
                  "was 0.59 for a probe trained on radar plus optical and run on radar alone, against 0.83 on both "
                  "inputs and 0.79 for a probe trained on radar alone (exp88). Which condition is more accurate needs "
                  "labels: sample with --condition.")
# The warning on every probability map. 1.3.1 ended it "prefer logits", which is wrong for a map of more than two
# classes: there the logit margin ranked errors worse than the probability margin on 16 of 16 of the suite's
# multi-class tasks (exp76), and passing probabilities is how the command line reaches the top probability.
PROBABILITY_WARNING = ("probability input: confidence ties where probabilities saturate. For two classes, logits avoid "
                       "the ties; for more than two, keep the probabilities (exp76)")
# The warning on a multi-class logit map scored by the default form. `form` is a Python argument only (1.3.1 said
# "pass form='top1'" alone). The command line reaches a top-probability reading through probability input: the
# window mean of the top probability, where form='top1' takes the window mean of its log, so the two orders agree
# window by window only at a patch of one pixel (tests/test_assess.py).
MARGIN_FORM_WARNING = ("multi-class logit margin: on Ai2's suite one minus the top probability ranked errors better on "
                       "14 of 16 multi-class tasks (exp76). In Python, pass form='top1'. The command line has no such "
                       "option: pass the class probabilities without --logits. That also ranks by the top probability, "
                       "averaged over each window where form='top1' averages its log, and can tie where probabilities "
                       "saturate")
CLASS_SHARE_TEXT = ("Descriptive only: the share of windows the map calls each class, within each condition. No "
                    "experiment has tested whether a difference between conditions signals errors.")


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


def pool_condition(condition, patch, predicted=None):
    """The input condition of each window, from a per-pixel layer.

    `condition` is an (H, W) layer of integer values: a cloud flag, the modalities present, a sensor id or an
    acquisition group. A negative value, NaN or a masked pixel (a numpy masked array) records none. A pixel votes
    when it holds a recorded value and has a prediction (`predicted`, the map's valid pixels; all of them if None).
    A window takes the value held by most of its voting pixels. A tie, or a window with no voting pixel, is
    unrecorded (-1): a tie never goes to the lowest value. A user who wants "any cloud makes the window cloudy"
    encodes that in the layer.

    Returns {"grid": (h, w) int64 value per window, -1 unrecorded and outside the map's valid windows,
    "values": the sorted values that won a valid window, "n_split": valid windows whose votes are tied between condition values,
    "n_no_code": valid windows with no voting pixel}. A valid window is at least half predicted pixels, as in assess.

    Refused: a layer that is not one band, a shape other than the map's, a value that is not an integer, a value
    beyond int32 (the condition.tif the command line writes), more than MAX_CONDITIONS distinct voting values, and
    a layer from which no valid window takes a value."""
    mask = np.ma.getmaskarray(condition) if np.ma.isMaskedArray(condition) else None
    a = np.asarray(np.ma.getdata(condition))
    if a.ndim != 2:
        hint = "; pass layer[0]" if a.ndim == 3 and a.shape[0] == 1 else ""
        raise ValueError(f"the condition layer has shape {a.shape}; it must be one band, (H, W), with one value per "
                         f"pixel{hint}")
    if predicted is None:
        predicted = np.ones(a.shape, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if predicted.shape != a.shape:
        raise ValueError(f"the condition layer is {a.shape[0]} x {a.shape[1]} px and the map is {predicted.shape[0]} x "
                         f"{predicted.shape[1]} px; give one condition value per pixel of the map")
    H, W = a.shape
    if patch > H or patch > W:
        raise ValueError(f"the window of {patch} px is larger than the map, {H} x {W} px; pass a smaller --patch")
    unmasked = ~mask if mask is not None else np.ones(a.shape, dtype=bool)
    if a.dtype.kind == "b":
        a = a.astype(np.int64)
    elif a.dtype.kind == "f":
        finite = np.isfinite(a) & unmasked
        if not np.all(np.mod(a[finite], 1) == 0):
            raise ValueError("the condition layer holds values that are not integers; a condition is a category, such "
                             "as a cloud flag or a sensor id. Bin a continuous layer, such as cloud fraction, first")
        unmasked = finite
    elif a.dtype.kind not in "iu":
        raise ValueError(f"the condition layer must hold integers, got dtype {a.dtype}")
    votes = unmasked & predicted & (a >= 0)
    if votes.any() and float(a[votes].max()) > np.iinfo(np.int32).max:
        raise ValueError(f"the condition layer holds the value {float(a[votes].max()):g}, beyond what a condition "
                         f"value can be (int32); if it is the raster's no-data value, declare it as no-data or set it "
                         f"negative")
    vals, codes_v = np.unique(a[votes].astype(np.int64), return_inverse=True)
    if vals.size > MAX_CONDITIONS:
        raise ValueError(f"the condition layer holds {vals.size} distinct values where the map has a prediction; at "
                         f"most {MAX_CONDITIONS} are allowed. A condition is a category, such as a cloud flag or a "
                         f"sensor id; bin a continuous layer first")
    codes = np.full(a.shape, -1, dtype=np.int64)
    codes[votes] = codes_v.ravel()
    res = _pooled_argmax(codes, max(int(vals.size), 1), patch, empty=-1, tie=-2)
    valid_w = _pool(predicted.astype(float), patch) >= 0.5          # the map's valid windows, as _assess finds them
    grid = np.where(res >= 0, vals[np.clip(res, 0, None)] if vals.size else -1, -1).astype(np.int64)
    grid[~valid_w] = -1
    n_split, n_no_code = int(((res == -2) & valid_w).sum()), int(((res == -1) & valid_w).sum())
    won = np.unique(grid[grid >= 0])
    if won.size == 0:
        raise ValueError(f"no window takes a condition: of the map's {int(valid_w.sum())} valid windows, {n_no_code} "
                         f"have no pixel with a recorded condition and {n_split} are tied between condition values")
    return {"grid": grid, "values": [int(v) for v in won], "n_split": n_split, "n_no_code": n_no_code}


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
    # Neighbours outside the grid are the replicated edge, exactly as boundary_indicator pads: a diagonal
    # neighbour beyond the edge copies the in-grid lateral cell, so the two paths give the same VALUE, not only
    # the same set of windows above zero. Until 2026-09-23 this path dropped out-of-grid neighbours instead, and
    # the indicator written to the CSV and boundary.tif differed by up to 0.5 at tile edges between a map with
    # no-data and one without (exp82's audit); the cue set, indicator > 0, was the same under both.
    H, W = pooled_hard.shape
    rows, cols = np.arange(H), np.arange(W)
    diff = np.zeros(pooled_hard.shape, np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            r = np.clip(rows + dy, 0, H - 1)[:, None]
            c = np.clip(cols + dx, 0, W - 1)[None, :]
            diff += valid[r, c] & (pooled_hard[r, c] != pooled_hard)
    return np.where(valid, diff / 8.0, 0.0)


ORDERS = ("confidence", "boundary_first")


def boundary_first_score(suspicion, boundary):
    """The review order supported by exp36: boundary windows first, ordered by suspicion, then the interior by suspicion.

    2 * [boundary > 0] + the within-map midrank percentile of suspicion, the construction exp36 and exp38 tested; higher
    is reviewed first. Ties are broken by review_order as for any score."""
    s_ = np.asarray(suspicion, dtype=np.float64)
    return np.where(np.asarray(boundary) > 0, 2.0, 0.0) + midrank_pct(s_).reshape(s_.shape)


def assess_classmap(hard, confidence, n_classes, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10),
                    signal="exported top-1 probability", order="confidence", condition=None, condition_names=None):
    """Production case: a hard class map plus an exported per-pixel confidence
    band (for instance the top-1 probability bands of the LCC rasters), with
    no logits. Ties in `confidence` are reported, because a quantized or
    saturated band can only rank the pixels it separates. `condition` and
    `condition_names` are as in assess_prediction."""
    hard = np.asarray(hard).astype(int)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = ~nodata_mask if nodata_mask is not None else np.ones(conf.shape, dtype=bool)
    vals, counts = np.unique(conf[valid], return_counts=True)
    warnings = [f"confidence band has {len(vals)} distinct values; {counts.max() / counts.sum():.3f} of pixels share the modal value {vals[counts.argmax()]:.4g}"]
    out = _assess(conf, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings, order,
                  condition, condition_names)
    out["confidence_distinct_values"] = int(len(vals))
    out["confidence_modal_share"] = float(counts.max() / counts.sum())
    return out


def assess_prediction(scores, is_logit, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10), order="confidence",
                      form="margin", condition=None, condition_names=None):
    """Assess a prediction map: `scores` is (H, W) of binary logits or probabilities, or (C, H, W) per class.

    `order` is the review order of the review sets: "confidence" (least confident first, the ranker every
    experiment scored) or "boundary_first" (boundary windows first, then the interior, each by confidence; the
    order that captures more errors at 5-10% budgets on hand labels, exp36). AURC entries always score confidence.

    `form` is the member of the confidence family used on a (C, H, W) logit map: "margin", the top-1 minus top-2
    logit margin (the default, unchanged since 1.0.0), or "top1", the top softmax probability computed tie-free
    from the logits. On the 16 multi-class tasks of Ai2's suite "top1" ranked errors better than the margin on 14,
    and the logit margin was the weakest of the three forms on 14 of 16 by AUROC and 15 of 16 by excess AURC (exp76,
    tying for best on awf_sentinel2), so a multi-class logit map scored with the default carries a warning. Binary
    maps and probability input are unaffected: there the forms are one ranking, and probability input already uses
    the top probability. `form` has no command-line option. There, probability input gives a top-probability
    reading: the window mean of the top probability, where "top1" takes the window mean of its log, with ties where
    probabilities saturate (MARGIN_FORM_WARNING).

    `condition` is an optional (H, W) integer layer on the map's grid: the input condition of each pixel, such as a
    cloud flag or the modalities present, negative or NaN where none is recorded (pool_condition gives the rule).
    With it, `arrays["condition"]` holds each window's value and `conditions` describes and ranks each condition on
    its own, with its review sets. Nothing that exists without it changes. `condition_names` is {value: name};
    names are unique and non-empty, "unrecorded" is reserved, and a value's default name is str(value). Without a
    condition, or with two or more, `scope` says what the whole-map ranking does not show (exp88)."""
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
            warnings.append(PROBABILITY_WARNING)
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
                warnings.append(MARGIN_FORM_WARNING)
        else:
            margin = srt[-1]  # top-1 probability
            warnings.append(PROBABILITY_WARNING)
        hard = scores.argmax(0)
        n_classes = C
    out = _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets,
                  ("1 - max probability (from logits)" if form == "top1" and scores.ndim == 3 else "negative logit margin")
                  if is_logit else "1 - max probability", warnings, order, condition, condition_names)
    if is_logit and form == "top1" and scores.ndim == 3:
        # The ranking reads the window mean of log p1, which is tie-free; the quantiles a user sets thresholds from
        # must be on the probability scale. Until 2026-09-22 they were reported as log-probabilities, a median
        # "confidence" of -0.620, under a label naming a probability (audit 2026-09-21, finding 12).
        out["confidence_quantiles"] = {q: float(np.exp(v)) for q, v in out["confidence_quantiles"].items()}
        for entry in out.get("conditions", {}).get("per_condition", {}).values():     # each condition on the same scale
            entry["confidence_quantiles"] = {q: float(np.exp(v)) for q, v in entry["confidence_quantiles"].items()}
        # the per-window array on the same scale as its quantiles: a threshold set from the quantiles used to be
        # compared with log-probabilities, so every window fell below the reported 25% quantile (review, 2026-09-23).
        # exp is increasing, so the ranking, the review sets and the cues are unchanged.
        with np.errstate(over="ignore"):
            out["arrays"]["confidence"] = np.exp(out["arrays"]["confidence"])
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


def _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings, order="confidence",
            condition=None, condition_names=None):
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    margin = np.asarray(margin, dtype=np.float64)
    hard = np.asarray(hard).astype(int)
    H, W = hard.shape[-2:]
    if patch > H or patch > W:
        raise ValueError(f"the window of {patch} px is larger than the map, {H} x {W} px; pass a smaller --patch")
    if condition is None and condition_names:
        raise ValueError("condition_names names the values of `condition`, and no condition was given")
    # the layer is checked before any work on the map, so a refusal comes first
    pooled = None if condition is None else pool_condition(
        condition, patch, ~np.asarray(nodata_mask, dtype=bool) if nodata_mask is not None else np.ones((H, W), dtype=bool))
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
        "confidence_quantiles": {q: float(np.nanquantile(conf_w[valid_w], q)) for q in QUANTILES},
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

    # The input condition. Everything above is the whole map's and does not change with a layer; the layer adds its
    # own block and a note on what the whole-map order does not show (exp88). Set before the reference, whose branch
    # can return early.
    if pooled is None:
        out["scope"] = SCOPE_ASSESS
    else:
        out["arrays"]["condition"] = pooled["grid"]
        out["conditions"] = _condition_block(pooled, condition_names, valid_w, conf_w, pooled_hard, n_classes, rank,
                                             out["review_sets"])
        K = out["conditions"]["n_conditions"]
        if K >= 2:
            out["scope"] = SCOPE_ASSESS_K.format(K=K)

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
        # the systematic-error report a labelled map allows: which (predicted, reference) pairs the errors fall
        # into (exp82: on the suite's many-class tasks the top three pairs hold 18% to 61% of the errors)
        from oe_inferencex.explain import confusion_pairs
        rc["confusion_pairs"] = confusion_pairs(ref_w[scored], pooled_hard[scored])
        rc["caveat"] = "reference-product labels can flatter boundary-type signals (exp18); treat as expert truth only if it is"
        out["against_reference"] = rc
    return out


def _condition_block(pooled, condition_names, valid_w, conf_w, pooled_hard, n_classes, rank, review_sets):
    """Each input condition described and ranked on its own. The conditions are indexed as the sample's are
    (estimate._condition_index): the recorded values in ascending order, then "unrecorded" last if any valid window
    has none, so the names and the order here are the ones `sample --condition` uses."""
    grid = pooled["grid"]
    pop = np.flatnonzero(valid_w.ravel())
    index, values, names, sizes, notes = _condition_index(grid.ravel(), pop, grid.size, condition_names)
    cidx = np.full(grid.size, -1, dtype=np.int64)          # condition index per window, -1 outside the valid windows
    cidx[pop] = index
    n_valid = int(pop.size)
    per = {}
    for c, (name, value) in enumerate(zip(names, values)):
        inc = (cidx == c).reshape(grid.shape)
        n_c = int(sizes[c])
        # A condition's review set is the whole-map order kept to that condition: equal to review_mask of the
        # review score over valid & (condition == c), under either order, since dropping the other windows changes
        # no two windows' relative order.
        ranked = rank[cidx[rank] == c]
        entry = {"value": value, "n_windows": n_c, "share_of_map": n_c / n_valid,
                 "confidence_quantiles": {q: float(np.nanquantile(conf_w[inc], q)) for q in QUANTILES},
                 "class_share": {int(k): float((pooled_hard[inc] == k).mean()) for k in range(n_classes)},
                 "share_of_review_set": {}, "review_sets": {}}
        for b, rs in review_sets.items():
            entry["share_of_review_set"][b] = float((cidx[rank[:rs["n_windows"]]] == c).mean())
            k = min(n_c, max(1, int(round(b * n_c))))
            rows, cols = np.unravel_index(ranked[:k], grid.shape)
            entry["review_sets"][b] = {"n_windows": int(k), "windows_rowcol": np.stack([rows, cols], 1)}
        per[name] = entry
    block = {"source": None, "rule": RULE_TEXT, "n_conditions": len(names), "n_windows_split": pooled["n_split"],
             "n_windows_no_code": pooled["n_no_code"], "class_share_status": CLASS_SHARE_TEXT, "per_condition": per}
    if notes:
        block["notes"] = notes                             # a name given to a value that holds no window
    return block


def review_order(suspicion, valid=None):
    """Flat window indices in review order: most suspicious first, invalid windows last, ties broken by descending
    raster position (an ascending stable sort, reversed). The one definition of the review order; exp37, the
    explanation layer and the per-condition review sets use it so that a review set is built the same way
    everywhere. It compares every window's score with every other's, and a score need not mean the same thing in two
    input conditions: a model run on an input combination it was not trained on can be confidently wrong there
    (exp88). A per-condition review set is this order kept to the windows of that condition."""
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


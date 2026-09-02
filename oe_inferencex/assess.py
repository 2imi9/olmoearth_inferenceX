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
import numpy as np

from oe_inferencex.metrics import aurc_expected


def _pool(a, patch):
    h, w = a.shape[0] // patch * patch, a.shape[1] // patch * patch
    return a[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3))


def _pooled_argmax(hard, n_classes, patch):
    h, w = hard.shape[0] // patch * patch, hard.shape[1] // patch * patch
    blocks = hard[:h, :w].reshape(h // patch, patch, w // patch, patch).transpose(0, 2, 1, 3).reshape(h // patch, w // patch, -1)
    counts = np.stack([(blocks == c).sum(-1) for c in range(n_classes)], axis=-1)
    return counts.argmax(-1)


def _boundary(pooled_hard):
    pad = np.pad(pooled_hard, 1, mode="edge")
    G0, G1 = pooled_hard.shape
    nb = np.zeros(pooled_hard.shape, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[1 + di:1 + di + G0, 1 + dj:1 + dj + G1] != pooled_hard)
    return nb / 8.0


def assess_classmap(hard, confidence, n_classes, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10),
                    signal="exported top-1 probability"):
    """Production case: a hard class map plus an exported per-pixel confidence
    band (for instance the top-1 probability bands of the LCC rasters), with
    no logits. Ties in `confidence` are reported, because a quantized or
    saturated band can only rank the pixels it separates."""
    hard = np.asarray(hard).astype(int)
    conf = np.asarray(confidence, dtype=np.float64)
    valid = ~nodata_mask if nodata_mask is not None else np.ones(conf.shape, dtype=bool)
    vals, counts = np.unique(conf[valid], return_counts=True)
    warnings = [f"confidence band has {len(vals)} distinct values; {counts.max() / counts.sum():.3f} of pixels share the modal value {vals[counts.argmax()]:.4g}"]
    out = _assess(conf, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings)
    out["confidence_distinct_values"] = int(len(vals))
    out["confidence_modal_share"] = float(counts.max() / counts.sum())
    return out


def assess_prediction(scores, is_logit, patch=4, nodata_mask=None, reference=None, budgets=(0.01, 0.05, 0.10)):
    scores = np.asarray(scores, dtype=np.float64)
    warnings = []
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
        srt = np.sort(scores, axis=0)
        if is_logit:
            margin = srt[-1] - srt[-2]
        else:
            margin = srt[-1]  # top-1 probability
            warnings.append("probability input: confidence ties where probabilities saturate; prefer logits")
        hard = scores.argmax(0)
        n_classes = C
    return _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets,
                   "negative logit margin" if is_logit else "1 - max probability", warnings)


def _assess(margin, hard, n_classes, patch, nodata_mask, reference, budgets, signal, warnings):
    margin = np.asarray(margin, dtype=np.float64)
    hard = np.asarray(hard).astype(int)
    if nodata_mask is not None:
        margin = np.where(nodata_mask, np.nan, margin)
        hard = np.where(nodata_mask, -1, hard)  # no-prediction pixels do not vote in pooling or boundaries

    conf_w = _pool(np.nan_to_num(margin, nan=np.nanmax(margin)), patch)   # higher = more confident
    valid_w = _pool((~nodata_mask).astype(float), patch) >= 0.5 if nodata_mask is not None else np.ones_like(conf_w, dtype=bool)
    pooled_hard = _pooled_argmax(hard, n_classes, patch)
    bnd_w = _boundary(pooled_hard)
    suspicion = -conf_w  # ranking signal: low margin first

    out = {
        "n_windows": int(valid_w.sum()), "patch_px": patch, "n_classes": n_classes,
        "confidence_quantiles": {q: float(np.nanquantile(conf_w[valid_w], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "boundary_window_fraction": float((bnd_w[valid_w] > 0).mean()),
        "class_share": {int(c): float((pooled_hard[valid_w] == c).mean()) for c in range(n_classes)},
        "signal": signal,
        "warnings": warnings,
        "arrays": {"confidence": conf_w, "boundary": bnd_w, "pooled_argmax": pooled_hard, "valid": valid_w},
        "review_sets": {},
        "confidence_distinct_pooled": int(len(np.unique(conf_w[valid_w]))),
    }
    order = np.argsort(np.where(valid_w, suspicion, -np.inf).flatten(), kind="stable")[::-1]  # most suspicious first
    n_valid = int(valid_w.sum())
    for b in budgets:
        k = max(1, int(round(b * n_valid)))
        idx = order[:k]
        rows, cols = np.unravel_index(idx, conf_w.shape)
        out["review_sets"][b] = {"n_windows": int(k), "windows_rowcol": np.stack([rows, cols], 1),
                                 "boundary_share_in_set": float((bnd_w.flatten()[idx] > 0).mean())}

    if reference is not None:
        ref = np.asarray(reference).astype(int)  # values < 0 mean no reference
        ref_valid_w = _pool((ref >= 0).astype(float), patch) >= 0.5
        scored = valid_w & ref_valid_w
        ref_w = _pooled_argmax(ref, n_classes, patch)
        err = (ref_w != pooled_hard).astype(float)
        e, s = err[scored], suspicion[scored]
        bnd_s = bnd_w[scored]
        rc = {"n_windows_scored": int(scored.sum()), "error_rate": float(e.mean()), "aurc_confidence": aurc_expected(s, e)}
        oracle = aurc_expected(e, e)  # errors most suspicious, so rejected first
        rc["excess_aurc_confidence"] = rc["aurc_confidence"] - oracle
        rc["aurc_boundary"] = aurc_expected(bnd_s, e)
        rc["aurc_random_expected"] = float(e.mean())
        cap = {}
        e_sorted = e[np.argsort(s, kind="stable")[::-1]]
        for b in budgets:
            k = max(1, int(round(b * len(e))))
            cap[b] = {"errors_captured_fraction": float(e_sorted[:k].sum() / max(e.sum(), 1)),
                      "precision_in_set": float(e_sorted[:k].mean())}
        rc["error_capture_at_budget"] = cap
        rc["boundary_share_among_errors"] = float((bnd_s[e > 0] > 0).mean()) if e.sum() else float("nan")
        rc["caveat"] = "reference-product labels can flatter boundary-type signals (exp18); treat as expert truth only if it is"
        out["against_reference"] = rc
    return out


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


"""Measure how two inferences of the same scene differ, without labels; grade the difference where labels exist.

The comparison half of the package (docs/Usage.md, "Compare two inferences of
the same scene"; docs/method/agent_integration.md). Two inferences of identical
windows, shifted crops, two backbones, two sensors, a frozen and a fine-tuned
model, are compared on the windows both predicted:

  disagreement   how much they differ, pooled and per group (tile, event)
  where          what the disagreement windows have in common: the enrichment of
                 label-free cues (boundary, low confidence, tiling instability,
                 spectral ambiguity) among them, against the agreement windows
  stability      whether two disagreement sets are the same set (pairwise phi)
  over_groups    a per-group statistic as a distribution: median, share of groups
                 where its sign flips, one-sided exact sign test (exp55's per-event
                 machinery)
  crosstab       with labels: the errors of a that b corrects, the errors b adds,
                 the errors both make, and their phi (exp52's label bridge)
  which_side     with labels: on the disagreement windows, how often each is right
  compare_inferences   all of the above in one summary

Conventions. Decisions (`a`, `b`, `labels`) are hard class maps, boolean or
integer; probabilities are thresholded and multi-class scores argmaxed by the
caller, so the module needs no threshold and never flags floating-point noise
as a difference. Indicators (`ok`, error maps, masks, cues) are boolean or 0/1
arrays, read as `> 0.5` like `explain.cue_enrichment`. Arrays share one shape
and are compared element-wise (a shape mismatch is an error, even at equal
size). `groups` is an id per window: an array of the full shape, one of the
same rank that broadcasts to it (per-tile ids as `(N, 1, 1)`), or one that
matches the leading dimensions (`(N,)` for `(N, H, W)` maps); numpy's
trailing-axis broadcasting is refused, so a wrong-length id vector cannot pass
silently as per-column ids. Group ids in the results are native Python scalars.
Undefined values are NaN (`assess.summary` writes them as null). Numpy only.
"""
import math

import numpy as np

from oe_inferencex.explain import cue_enrichment
from oe_inferencex.stats import TIE_TOL, sign_test, wins_losses_ties

NAN = float("nan")


def _decision(x, name):
    x = np.asarray(x)
    if x.dtype.kind == "f":
        raise TypeError(f"{name}: hard decisions expected (boolean or integer classes); threshold p > 0.5 or take the argmax first")
    return x


def _indicator(x):
    return np.asarray(x) > 0.5


def _same_shape(**arrays):
    shapes = {k: np.shape(v) for k, v in arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"arrays must share one shape, got {shapes}")


def _group_ids(groups, shape):
    """One id per window, raveled (see the module docstring for the accepted forms)."""
    g = np.asarray(groups)
    if g.ndim < len(shape):
        if g.shape != shape[:g.ndim]:
            raise ValueError(f"groups of shape {g.shape} do not match the leading dimensions of the windows' shape {shape}")
        g = g.reshape(g.shape + (1,) * (len(shape) - g.ndim))
    try:
        return np.broadcast_to(g, shape).ravel()
    except ValueError:
        raise ValueError(f"groups of shape {g.shape} do not broadcast to the windows' shape {shape}") from None


def _count_by_group(inv, n_groups, mask):
    """Windows carrying `mask` per group, from np.unique's inverse index of the raveled ids."""
    return np.bincount(inv[np.asarray(mask).ravel()], minlength=n_groups)


def _key(v):
    return v.item() if hasattr(v, "item") else v


def _phi_counts(n11, n10, n01, n00):
    """Phi from the four cells of a 2x2 table; the cells are floats since the product of the margins overflows
    int64 at a few hundred thousand windows (exp51's note)."""
    n11, n10, n01, n00 = float(n11), float(n10), float(n01), float(n00)
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return (n11 * n00 - n10 * n01) / math.sqrt(den) if den > 0 else NAN


def phi(a, b):
    """Phi coefficient (Matthews correlation) of two indicator arrays of one shape, compared element-wise: a float,
    NaN when either is constant (an empty array included).

    The single implementation the experiments import (exp46, exp51, exp52 carried copies)."""
    a, b = _indicator(a), _indicator(b)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    a, b = a.ravel(), b.ravel()
    n11, n10, n01 = np.count_nonzero(a & b), np.count_nonzero(a & ~b), np.count_nonzero(~a & b)
    return _phi_counts(n11, n10, n01, a.size - n11 - n10 - n01)


def disagreement(a, b, ok, groups=None):
    """Windows where two hard decisions differ, among the valid ones.

    Returns n (valid windows), n_disagree, rate (NaN without a valid window), mask (boolean, the input shape, False
    outside `ok`) and per_group: None without `groups`, else {id: {n, n_disagree, rate}} keyed by the native group
    id, every id present even where no window is valid (n 0, rate NaN)."""
    a, b, ok = _decision(a, "a"), _decision(b, "b"), _indicator(ok)
    _same_shape(a=a, b=b, ok=ok)
    mask = (a != b) & ok
    n, nd = int(ok.sum()), int(mask.sum())
    out = {"n": n, "n_disagree": nd, "rate": nd / n if n else NAN, "mask": mask, "per_group": None}
    if groups is not None:
        ids, inv = np.unique(_group_ids(groups, mask.shape), return_inverse=True)
        n_g, d_g = _count_by_group(inv, len(ids), ok), _count_by_group(inv, len(ids), mask)
        out["per_group"] = {_key(i): {"n": int(nv), "n_disagree": int(dv), "rate": int(dv) / int(nv) if nv else NAN}
                            for i, nv, dv in zip(ids, n_g, d_g)}
    return out


def where(mask, cues):
    """What the disagreement windows have in common: each cue's share among them against its share among the
    agreement windows, and the ratio (explain.cue_enrichment with the disagreement mask in the role of the errors,
    no bootstrap). Pass windows already restricted to the valid ones; every cue must have the mask's shape.

    Returns {cue name: {n, n_disagree, n_with_cue, share_disagree (share of disagreement windows carrying the cue),
    share_agree (share of agreement windows), enrichment (their ratio), rate_with_cue (disagreement rate among the
    windows carrying the cue)}}; a share or ratio is NaN when its denominator is empty (no disagreement window, no
    agreement window, no window with the cue, a zero share among the agreement windows)."""
    m = _indicator(mask)
    out = {}
    for name, cue in cues.items():
        c = _indicator(cue)
        if c.shape != m.shape:
            raise ValueError(f"cue {name!r} has shape {c.shape} ({c.size} windows), the mask has shape {m.shape} ({m.size} windows)")
        r = cue_enrichment(c.ravel(), m.ravel(), n_boot=0)
        out[name] = {"n": r["n"], "n_disagree": r["n_errors"], "n_with_cue": r["n_with_cue"],
                     "share_disagree": r["share_errors"], "share_agree": r["share_correct"],
                     "enrichment": r["enrichment"], "rate_with_cue": r["precision"]}
    return out


def stability(masks):
    """Are several disagreement sets the same set? Pairwise phi between masks of one shape (a dict name -> mask, or a
    sequence), as a matrix in the given order.

    Returns names (the keys or positions, as strings), phi (the k x k matrix as nested lists, NaN wherever a mask is
    constant, its diagonal included) and median_pairwise_phi (the median of the finite off-diagonal values, NaN when
    there is none)."""
    if isinstance(masks, dict):
        names, arrs = [str(k) for k in masks], [_indicator(v) for v in masks.values()]
    else:
        arrs = [_indicator(m) for m in masks]
        names = [str(i) for i in range(len(arrs))]
    k = len(arrs)
    mat = np.full((k, k), NAN)
    for i in range(k):
        for j in range(i, k):
            mat[i, j] = mat[j, i] = phi(arrs[i], arrs[j])
    off = mat[~np.eye(k, dtype=bool)]
    finite = off[np.isfinite(off)]
    return {"names": names, "phi": mat.tolist(), "median_pairwise_phi": float(np.median(finite)) if finite.size else NAN}


def over_groups(stat, groups=None):
    """A per-group statistic as a distribution over groups, positive meaning the preregistered direction.

    `stat` is {group: value} or a 1-D array with `groups` its aligned ids (positions otherwise; `groups` is refused
    with a dict, whose keys are the ids). None and non-finite values (NaN, inf) are dropped and counted as undefined.
    Returns n_groups (defined values), n_undefined, median (NaN with no defined value), w / l / t
    (stats.wins_losses_ties: above TIE_TOL, below -TIE_TOL, within), share_flipped (l / n_groups, the share of
    groups where the sign is negative beyond TIE_TOL, exp55's exception rate; NaN with no defined value), sign_p
    (stats.sign_test of w against l, one-sided, that the direction holds) and flipped (the ids of the l groups, as
    native scalars)."""
    if isinstance(stat, dict):
        if groups is not None:
            raise ValueError("with a dict of statistics the group ids are its keys; pass groups only with an array")
        ids, vals = [_key(k) for k in stat], list(stat.values())
    else:
        vals = list(np.asarray(stat, dtype=object).ravel())
        ids = list(range(len(vals))) if groups is None else [_key(g) for g in np.asarray(groups).ravel()]
        if len(ids) != len(vals):
            raise ValueError(f"{len(vals)} values but {len(ids)} group ids")
    v = np.array([NAN if x is None else float(x) for x in vals], dtype=np.float64)
    finite = np.isfinite(v)
    d, kept = v[finite], [i for i, f in zip(ids, finite) if f]
    w, l, t = wins_losses_ties(d)
    return {"n_groups": int(finite.sum()), "n_undefined": int((~finite).sum()),
            "median": float(np.median(d)) if d.size else NAN, "w": w, "l": l, "t": t,
            "share_flipped": l / d.size if d.size else NAN, "sign_p": sign_test(w, l, "greater"),
            "flipped": [i for i, x in zip(kept, d) if x < -TIE_TOL]}


def _crosstab_counts(n, corrected, broken, both):
    n, corrected, broken, both = int(n), int(corrected), int(broken), int(both)
    errors_a, neither = corrected + both, n - corrected - broken - both
    return {"n": n, "errors_a": errors_a, "errors_b": broken + both, "corrected": corrected, "broken": broken,
            "both": both, "neither": neither, "share_corrected": corrected / max(errors_a, 1),
            "phi": _phi_counts(both, corrected, broken, neither)}


def crosstab(err_a, err_b, ok):
    """The label bridge (exp52): the two error maps cross-tabulated on the valid windows.

    Returns n (valid windows), errors_a, errors_b, corrected (a's errors b does not make), broken (b's errors where a
    was right), both (the errors both make), neither, share_corrected (corrected / errors_a; 0.0 when a has no error,
    exp52's recorded convention) and phi (of the two error maps on the valid windows, NaN when either is constant)."""
    ok = _indicator(ok)
    _same_shape(err_a=err_a, err_b=err_b, ok=ok)
    a, b = _indicator(err_a)[ok], _indicator(err_b)[ok]
    return _crosstab_counts(a.size, np.count_nonzero(a & ~b), np.count_nonzero(~a & b), np.count_nonzero(a & b))


def which_side(a, b, labels, ok):
    """On the windows where two decisions disagree, how often each side matches the label.

    Returns n_disagree, a_right, b_right, neither (disagreement windows where both are wrong; only multi-class maps
    have any), share_a_right and share_b_right (NaN without a disagreement window). For any number of classes a_right
    equals crosstab's broken and b_right its corrected: where the sides disagree and one is right, the other errs."""
    a, b, lab, ok = _decision(a, "a"), _decision(b, "b"), _decision(labels, "labels"), _indicator(ok)
    _same_shape(a=a, b=b, labels=lab, ok=ok)
    d = (a != b) & ok
    nd, ar, br = int(d.sum()), int(((a == lab) & d).sum()), int(((b == lab) & d).sum())
    return {"n_disagree": nd, "a_right": ar, "b_right": br, "neither": nd - ar - br,
            "share_a_right": ar / nd if nd else NAN, "share_b_right": br / nd if nd else NAN}


def compare_inferences(a, b, ok, groups=None, labels=None, cues=None):
    """Two inferences of identical windows in one summary; label-free unless `labels` is given.

    a, b     hard decisions of the same shape; ok: the windows both predicted
    groups   optional id per window (tile, event) for the per-group rates and, with labels, the per-group cross-tab
    labels   optional class map; adds the graded block
    cues     optional {name: indicator of the same shape}; adds their enrichment on the disagreement windows

    Returns n_windows, n_disagree, disagreement_rate, per_group (disagreement's, or None without groups), where
    (`where` of the cues on the valid windows, or None without cues), graded (None without labels, else crosstab and
    which_side pooled, per_group: {id: crosstab on the group's valid windows} or None, and over_groups of the
    per-group net correction, corrected minus broken, positive when b is the better side, or None) and arrays:
    {disagree: the mask}. `assess.summary` strips the arrays for JSON."""
    okm = _indicator(ok)
    dis = disagreement(a, b, okm, groups)
    mask = dis["mask"]
    out = {"n_windows": dis["n"], "n_disagree": dis["n_disagree"], "disagreement_rate": dis["rate"],
           "per_group": dis["per_group"], "where": None, "graded": None, "arrays": {"disagree": mask}}
    if cues:
        for name, c in cues.items():
            if np.shape(c) != mask.shape:
                raise ValueError(f"cue {name!r} has shape {np.shape(c)}, windows are {mask.shape}")
        out["where"] = where(mask[okm], {k: _indicator(v)[okm] for k, v in cues.items()})
    if labels is not None:
        lab = _decision(labels, "labels")
        a_, b_ = _decision(a, "a"), _decision(b, "b")
        _same_shape(a=a_, labels=lab)
        err_a, err_b = a_ != lab, b_ != lab
        graded = {"crosstab": crosstab(err_a, err_b, okm), "which_side": which_side(a_, b_, lab, okm),
                  "per_group": None, "over_groups": None}
        if groups is not None:
            ids, inv = np.unique(_group_ids(groups, mask.shape), return_inverse=True)
            ea, eb = err_a & okm, err_b & okm
            cells = [_count_by_group(inv, len(ids), m) for m in (okm, ea & ~eb, ~ea & eb, ea & eb)]
            per = {_key(i): _crosstab_counts(n, c, br, bo) for i, n, c, br, bo in zip(ids, *cells)}
            graded["per_group"] = per
            graded["over_groups"] = over_groups({k: v["corrected"] - v["broken"] for k, v in per.items()})
        out["graded"] = graded
    return out

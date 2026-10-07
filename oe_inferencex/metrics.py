"""Ranking and calibration metrics for selective prediction. Pure numpy, no
torch, so the assessment layer can be imported by consumers that do not ship
torch.

Conventions (docs/method/protocol.md): `uncertainty` is any score where
higher means more suspect; `errors` is 0/1 per unit. Ties are handled by
expectation under random tie-breaking, so no result depends on raster order.
"""
import numpy as np

# An empty population (a tile or stratum with no valid window) has no ranking to measure. Until 2026-10-06 half of
# this module raised an internal IndexError from np.add.reduceat on it while the other half returned NaN or 0.0, so a
# per-tile loop crashed or silently scored an empty tile; every function now returns NaN (per budget or coverage).
_NAN = float("nan")


def _expected_cum_errors(uncertainty, errors):
    """Expected cumulative error count at each rank 1..n in ascending uncertainty, under uniform random
    tie-breaking: within a group of tied scores the group's errors are spread uniformly over its rank span."""
    u = np.asarray(uncertainty).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")
    s, e = u[order], e[order]
    n = len(e)
    cum = np.cumsum(e)
    newgrp = np.r_[True, s[1:] != s[:-1]]
    grp = np.cumsum(newgrp) - 1
    starts = np.flatnonzero(newgrp)
    sizes = np.diff(np.r_[starts, n])
    e_group = np.add.reduceat(e, starts)
    e_before = np.r_[0.0, cum][starts]
    pos = np.arange(n) - starts[grp] + 1
    return e_before[grp] + e_group[grp] * pos / sizes[grp]


def aurc_expected(uncertainty, errors):
    """AURC under uniform random tie-breaking, in closed form.

    Within each group of tied scores the errors are spread uniformly over the
    group's rank span, so the result does not depend on the raster order of
    the input. Equal to the plain AURC when no scores tie. NaN on an empty population.
    """
    e = np.asarray(errors).flatten()
    n = len(e)
    if n == 0:
        return _NAN
    risk = _expected_cum_errors(uncertainty, e) / np.arange(1, n + 1)
    return float(risk.mean())


def risk_coverage(uncertainty, errors):
    """Selective risk at every coverage level, ordered by ascending uncertainty.

    Returns (coverage, risk, aurc). The curve uses a stable sort for display;
    the returned AURC is tie-aware (aurc_expected). Lower = ranks errors better.
    """
    # np.asarray: a list or a pandas Series raised AttributeError on .flatten() until 2026-10-06, where every other
    # metric here accepts any array-like
    u = np.asarray(uncertainty).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")
    cum_err = np.cumsum(e[order])
    n = len(u)
    coverage = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    return coverage, risk, aurc_expected(u, e)


def oracle_aurc(n, k):
    """AURC of the perfect ranker on n units with k errors: every error rejected first.

    Closed form used since exp13; equal to aurc_expected(errors, errors)."""
    if int(n) == 0:
        return _NAN
    i = np.arange(1, n + 1)
    return float((np.maximum(0, i - (n - k)) / i).mean())


def excess_aurc(uncertainty, errors):
    """E-AURC (Geifman et al. 2018): AURC minus the oracle's, comparable across units with different error rates.

    The subtraction cancels in any signal-minus-baseline difference on the same unit."""
    e = np.asarray(errors).flatten()
    return aurc_expected(uncertainty, e) - oracle_aurc(len(e), int(e.sum()))


def capture_at_budget(uncertainty, errors, budgets=(0.01, 0.05, 0.10)):
    """Operating points (recipe item 6): the fraction of all errors inside the most suspect fraction b of the units.

    The review set is the top max(1, round(b * n)) units by uncertainty, most suspect
    first, ties broken by position (stable sort), as exp21 and assess report it."""
    u = np.asarray(uncertainty).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    if len(e) == 0:
        return {b: _NAN for b in budgets}                 # an empty population, not "none of its errors caught"
    order = np.argsort(u, kind="stable")[::-1]
    out = {}
    for b in budgets:
        k = max(1, int(round(b * len(e))))
        out[b] = float(e[order[:k]].sum() / max(e.sum(), 1))
    return out


def augrc(uncertainty, errors):
    """AUGRC (Traub et al. 2024, arXiv 2407.01032): the area under the GENERALISED risk-coverage curve.

    Generalised risk counts an error only while it is still KEPT and divides by the whole population rather than by
    the kept set, which removes the selective risk's blow-up at low coverage. AUGRC is its mean over coverage.

    For a FIXED set of errors it is an exactly affine, strictly decreasing function of the failure AUROC:

        AUGRC = (1 - AUROC_f) * e * (1 - e) + e^2 / 2 + e / (2n)

    derived in docs/method/protocol.md and checked to machine precision in tests/test_formulas.py. The coefficient
    e(1 - e) is positive for any map that is neither perfect nor wholly wrong, so AUGRC cannot reorder two readings
    that the failure AUROC already orders, and the last two terms do not depend on the reading at all. That is why
    exp76 could answer the AUGRC challenge from exp70's recorded AUROC without recomputing anything.

    Ties are handled as aurc_expected handles them, by expectation under random tie-breaking, so the identity holds
    with the AUROC counting ties as half. NaN on an empty population."""
    e = np.asarray(errors).flatten()
    n = len(e)
    if n == 0:
        return _NAN
    # Until 2026-10-06 the kept errors were counted in raster order inside a group of tied scores, so a constant score
    # gave 0.25 with its one error first and 0.0625 with it last, and a nine-level score could reverse the ranking of
    # two readings by flipping the map; the identity above (0.15625 there) held only when no scores tied.
    return float((_expected_cum_errors(uncertainty, e) / n).mean())


def augrc_from_auroc(auroc_failure, error_rate, n):
    """The same quantity from the failure AUROC, by the identity above. Exact, not an approximation."""
    e = float(error_rate)
    return float((1.0 - float(auroc_failure)) * e * (1.0 - e) + e ** 2 / 2.0 + e / (2.0 * int(n)))


def attainable_ceiling(budget, error_rate, n=None):
    """The most of a map's errors ANY ranking can hold at a review budget, as a share of all its errors.

    Derivation. A review of k units can contain at most k errors, and at most all E of them, so the captured count
    is at most min(k, E) and the captured SHARE at most min(k, E) / E = min(1, k / E). With k = b*n and E = e*n
    that is min(1, b / e). It is attained exactly when the ranking puts errors first, which is the oracle.

    Two conventions differ, and the difference is real on small maps. `n=None` uses the nominal budget, min(1, b/e),
    the continuous form. Passing `n` uses the REALISED review set, k = max(1, round(b * n)), which is the set
    `capture_at_budget` and `assess_prediction` actually score; the two disagree whenever round(b*n) != b*n, and on
    a small scene a nominal 1% can be a realised 1.2%. Prefer passing `n` when comparing against a measured capture.

    Why it matters (recipe item 6, exp68's audit correction): capture at a fixed budget is bounded by this ceiling,
    so a capture compared across two populations with different error rates is partly a comparison of ceilings. Use
    the share of the ceiling, or a ceiling-free statistic such as AUROC or an odds ratio.
    """
    e = float(error_rate)
    if not 0.0 < e <= 1.0:
        raise ValueError(f"error_rate must be in (0, 1], got {error_rate!r}; a map with no errors has no ceiling")
    if n is None:
        return float(min(1.0, float(budget) / e))
    n = int(n)
    k = max(1, int(round(float(budget) * n)))
    n_err = e * n
    return float(min(1.0, k / n_err)) if n_err > 0 else float("nan")


def selective_accuracy(uncertainty, correct, coverages=(0.5, 0.8, 0.9, 1.0)):
    """Accuracy among the fraction c of units kept in ascending uncertainty (recipe item 7, exp21).

    The kept set is the max(1, round(c * n)) least suspect units. Tie-aware in the sense of capture_at_budget_expected:
    a group of tied scores straddling the cut contributes its correct units in proportion to the share of the group
    kept, the expectation under random tie-breaking. Equal to the plain kept-set accuracy when no scores tie at the
    cut. NaN on an empty population."""
    u = np.asarray(uncertainty).flatten()
    c_ = np.asarray(correct).flatten().astype(np.float64)
    n = len(c_)
    if n == 0:
        return {c: _NAN for c in coverages}
    # Until 2026-10-06 the cut took the tied units in raster order, so u = zeros(4) at coverage 0.5 gave 1.0 for
    # correct = [1,1,0,0] and 0.0 for [0,0,1,1], the same map read in another order; a top-1 probability that
    # saturates at 1.0 (exp89's input) ties on half its units and moved the result with the row order.
    order = np.argsort(u, kind="stable")
    s, c_ = u[order], c_[order]
    newgrp = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newgrp)
    sizes = np.diff(np.r_[starts, n])
    c_group = np.add.reduceat(c_, starts)
    out = {}
    for c in coverages:
        k = min(max(1, int(round(c * n))), n)
        full = starts + sizes <= k                        # groups entirely inside the kept set
        kept = c_group[full].sum()
        part = np.flatnonzero((starts < k) & ~full)       # the group straddling the cut, if any
        if len(part):
            g = part[0]
            kept += c_group[g] * (k - starts[g]) / sizes[g]
        out[c] = float(kept / k)
    return out


def expected_calibration_error(confidence, correct, bins=10):
    """ECE over `bins` equal-width bins (lo, hi] of the top-1 probability (exp21).

    Returns (ece, rows) with one row (lo, hi, n, mean confidence, accuracy) per non-empty bin."""
    conf = np.asarray(confidence, dtype=np.float64).flatten()
    corr = np.asarray(correct, dtype=np.float64).flatten()
    # Units with no finite confidence or outcome (no-data) are not in the population at all. Until 2026-09-22 they
    # were dropped from the bins but kept in the denominator, so a scene 30% no-data reported 70% of its ECE; and a
    # confidence of exactly 0.0 fell outside the half-open first bin, returning 0.0 for data whose ECE is 0.5.
    keep = np.isfinite(conf) & np.isfinite(corr)
    conf, corr = conf[keep], corr[keep]
    if conf.size == 0:
        return float("nan"), []                                  # undefined, not "perfectly calibrated" (review, 2026-09-23)
    if ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must be a probability in [0, 1]")
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((conf >= lo) if j == 0 else (conf > lo)) & (conf <= hi)
        if m.any():
            gap = abs(corr[m].mean() - conf[m].mean())
            total += m.mean() * gap
            rows.append((float(lo), float(hi), int(m.sum()), float(conf[m].mean()), float(corr[m].mean())))
    return float(total), rows


def capture_at_budget_expected(uncertainty, errors, budgets=(0.05, 0.10, 0.20)):
    """Tie-aware error capture at a budget: the expected fraction of all errors inside the round(b * n) most suspect
    units when tied scores are broken at random, so a coarse score (the boundary indicator has nine levels) is not
    credited or penalised for raster order. Equal to capture_at_budget when no scores tie at the cut. As in
    capture_at_budget, at least one unit is always reviewed, so the realised budget is max(1, round(b * n)) / n."""
    u = np.asarray(uncertainty, dtype=np.float64).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    if len(e) == 0:
        return {b: _NAN for b in budgets}
    order = np.argsort(-u, kind="stable")                 # most suspect first (float cast: unsigned or boolean scores negate safely)
    s, e = u[order], e[order]
    n, total = len(e), max(e.sum(), 1)
    newgrp = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newgrp)
    sizes = np.diff(np.r_[starts, n])
    e_group = np.add.reduceat(e, starts)
    out = {}
    for b in budgets:
        k = max(1, int(round(b * n)))
        full = starts + sizes <= k                        # groups entirely inside the budget
        captured = e_group[full].sum()
        part = np.flatnonzero((starts < k) & ~full)       # the group straddling the cut, if any
        if len(part):
            g = part[0]
            captured += e_group[g] * (k - starts[g]) / sizes[g]
        out[b] = float(captured / total)
    return out


# --------------------------------------------------------------------------- design-weighted variants
# A reference drawn as a probability sample gives each graded unit an inclusion probability, and a unit sampled at
# probability 1/40 stands for forty units of the population. Every statistic above then has to be computed in weight
# units rather than in unit counts, or the answer is about the sample instead of the population: on a stratified sample
# that deliberately oversamples rare classes, the unweighted and the weighted lead can differ by half the lead's size
# (exp68). Each function here reduces exactly to its unweighted counterpart when every weight is one.
#
# A design weight is an inverse inclusion probability, never negative, and a population with no weight has nothing to
# measure. Until 2026-10-06 only weighted_aurc refused a negative weight, and a 1e-300 guard turned 0/0 into a number:
# with every weight zero the weighted mean and AURC read 0.0, and with the errors' weights zero (one stratum scored by
# zeroing the others) weighted_auroc read 0.0, a perfectly inverted ranker, where the same units selected by a mask
# give NaN. A negative weight is refused everywhere now, and no weight (or no weighted error or non-error, for the
# AUROC) is NaN.

def _check_weights(w):
    if (w < 0).any():
        raise ValueError("weights must be non-negative")


def weighted_mean(x, weights):
    """Design-weighted mean of `x`. Reduces to x.mean() under equal weights. NaN when no unit carries weight."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.shape != w.shape:
        raise ValueError(f"x has shape {x.shape}, weights {w.shape}")
    _check_weights(w)
    if not w.sum() > 0:
        return _NAN
    return float((x * w).sum() / w.sum())


def weighted_aurc(uncertainty, errors, weights):
    """Design-weighted AURC: selective risk integrated over coverage measured in weight rather than in units.

    Ties are handled as `aurc_expected` handles them, by expectation under random tie-breaking: within a group of tied
    scores the group's weighted errors are spread uniformly over the group's weight span, so no result depends on input
    order. Equal to `aurc_expected` to floating-point accuracy when every weight is one.

    Note that AURC is a right-endpoint average over units, so it is not exactly invariant to replicating a unit; nor is
    `aurc_expected`, which moves by about 2e-5 on a 2,000-unit sample when every unit is duplicated. Splitting a weight
    into equal parts therefore agrees with weighting to about that order, not exactly. NaN when no unit carries weight."""
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(errors, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not (u.shape == e.shape == w.shape):
        raise ValueError(f"shapes differ: uncertainty {u.shape}, errors {e.shape}, weights {w.shape}")
    _check_weights(w)
    if not w.sum() > 0:
        return _NAN
    order = np.argsort(u, kind="stable")
    s, e, w = u[order], e[order], w[order]
    cum_w, cum_e = np.cumsum(w), np.cumsum(e * w)
    newgrp = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newgrp)
    grp = np.cumsum(newgrp) - 1
    e_group = np.add.reduceat(e * w, starts)
    w_group = np.add.reduceat(w, starts)
    e_before = np.r_[0.0, cum_e][starts]
    w_before = np.r_[0.0, cum_w][starts]
    frac = (cum_w - w_before[grp]) / np.maximum(w_group[grp], 1e-300)
    risk = (e_before[grp] + e_group[grp] * frac) / np.maximum(cum_w, 1e-300)
    return float((risk * w).sum() / max(w.sum(), 1e-300))


def weighted_excess_aurc(uncertainty, errors, weights):
    """Design-weighted E-AURC: the weighted AURC minus that of the perfect ranker on the same weighted units."""
    e = np.asarray(errors, dtype=np.float64).ravel()
    return weighted_aurc(uncertainty, e, weights) - weighted_aurc(e, e, weights)


def weighted_capture_at_budget(uncertainty, errors, weights, budgets=(0.05, 0.10, 0.20)):
    """Design-weighted error capture: the share of weighted errors inside the most suspect fraction b of the weight.

    Tie-aware in the same sense as `capture_at_budget_expected`: a tied group straddling the cut contributes its weighted
    errors in proportion to the share of the group's weight that falls inside the budget.

    Capture at a fixed budget is bounded above by budget / error rate, so it is NOT comparable between two populations
    with different error rates; `weighted_auroc` is the base-rate-free statistic for that comparison. NaN at every
    budget when no unit carries weight."""
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(errors, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not (u.shape == e.shape == w.shape):
        raise ValueError(f"shapes differ: uncertainty {u.shape}, errors {e.shape}, weights {w.shape}")
    _check_weights(w)
    if not w.sum() > 0:
        return {b: _NAN for b in budgets}
    order = np.argsort(-u, kind="stable")
    s, e, w = u[order], e[order], w[order]
    cum_w = np.cumsum(w)
    total_w, total_e = cum_w[-1], max((e * w).sum(), 1e-300)
    newgrp = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newgrp)
    e_group = np.add.reduceat(e * w, starts)
    w_group = np.add.reduceat(w, starts)
    w_before = np.r_[0.0, cum_w][starts]
    out = {}
    for b in budgets:
        cut = b * total_w
        full = (w_before + w_group) <= cut + 1e-12
        got = e_group[full].sum()
        part = np.flatnonzero((w_before < cut) & ~full)
        if len(part):
            g = part[0]
            got += e_group[g] * (cut - w_before[g]) / max(w_group[g], 1e-300)
        out[b] = float(got / total_e)
    return out


def weighted_auroc(uncertainty, errors, weights):
    """Design-weighted AUROC of a suspicion score against the error indicator, ties counted half.

    Unlike capture at a budget this has no base-rate ceiling, so it is the statistic to use when comparing how well a
    score ranks errors across two populations whose error rates differ. NaN when every unit is an error or none is,
    counted in weight: units of weight zero are not in the population."""
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(errors, dtype=np.float64).ravel().astype(bool)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not (u.shape == e.shape == w.shape):
        raise ValueError(f"shapes differ: uncertainty {u.shape}, errors {e.shape}, weights {w.shape}")
    _check_weights(w)
    if not (w[e].sum() > 0 and w[~e].sum() > 0):
        return _NAN
    order = np.argsort(u, kind="stable")
    s, e, w = u[order], e[order], w[order]
    cum_w = np.cumsum(w)
    newgrp = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(newgrp)
    grp = np.cumsum(newgrp) - 1
    w_before = np.r_[0.0, cum_w][starts]
    w_group = np.add.reduceat(w, starts)
    midrank = w_before[grp] + 0.5 * w_group[grp]
    w_pos, w_neg = w[e].sum(), w[~e].sum()
    return float(((w[e] * midrank[e]).sum() - w_pos * w_pos / 2.0) / (w_pos * w_neg))

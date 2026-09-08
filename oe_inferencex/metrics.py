"""Ranking and calibration metrics for selective prediction. Pure numpy, no
torch, so the assessment layer can be imported by consumers that do not ship
torch.

Conventions (docs/method/protocol.md): `uncertainty` is any score where
higher means more suspect; `errors` is 0/1 per unit. Ties are handled by
expectation under random tie-breaking, so no result depends on raster order.
"""
import numpy as np


def aurc_expected(uncertainty, errors):
    """AURC under uniform random tie-breaking, in closed form.

    Within each group of tied scores the errors are spread uniformly over the
    group's rank span, so the result does not depend on the raster order of
    the input. Equal to the plain AURC when no scores tie.
    """
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
    cum_exp = e_before[grp] + e_group[grp] * pos / sizes[grp]
    risk = cum_exp / np.arange(1, n + 1)
    return float(risk.mean())


def risk_coverage(uncertainty, errors):
    """Selective risk at every coverage level, ordered by ascending uncertainty.

    Returns (coverage, risk, aurc). The curve uses a stable sort for display;
    the returned AURC is tie-aware (aurc_expected). Lower = ranks errors better.
    """
    u = uncertainty.flatten()
    e = errors.flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")
    cum_err = np.cumsum(e[order])
    n = len(u)
    coverage = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    return coverage, risk, aurc_expected(u, e)


def oracle_aurc(n, k):
    """AURC of the perfect ranker on n units with k errors: every error rejected first.

    Closed form used since exp13; equal to aurc_expected(errors, errors)."""
    i = np.arange(1, n + 1)
    return float((np.maximum(0, i - (n - k)) / i).mean())


def excess_aurc(uncertainty, errors):
    """E-AURC (Geifman et al. 2018): AURC minus the oracle's, comparable across units with different error rates.

    The subtraction cancels in any signal-minus-baseline difference on the same unit."""
    e = np.asarray(errors).flatten()
    return aurc_expected(uncertainty, e) - oracle_aurc(len(e), int(e.sum()))


def capture_at_budget(uncertainty, errors, budgets=(0.01, 0.05, 0.10)):
    """Operating points (recipe item 6): the fraction of all errors inside the most suspect fraction b of the units.

    The review set is the top round(b * n) units by uncertainty, most suspect
    first, ties broken by position (stable sort), as exp21 and assess report it."""
    u = np.asarray(uncertainty).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")[::-1]
    out = {}
    for b in budgets:
        k = max(1, int(round(b * len(e))))
        out[b] = float(e[order[:k]].sum() / max(e.sum(), 1))
    return out


def selective_accuracy(uncertainty, correct, coverages=(0.5, 0.8, 0.9, 1.0)):
    """Accuracy among the fraction c of units kept in ascending uncertainty (recipe item 7, exp21)."""
    u = np.asarray(uncertainty).flatten()
    c_ = np.asarray(correct).flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")
    return {c: float(c_[order[:max(1, int(round(c * len(c_))))]].mean()) for c in coverages}


def expected_calibration_error(confidence, correct, bins=10):
    """ECE over `bins` equal-width bins (lo, hi] of the top-1 probability (exp21).

    Returns (ece, rows) with one row (lo, hi, n, mean confidence, accuracy) per non-empty bin."""
    conf = np.asarray(confidence, dtype=np.float64).flatten()
    corr = np.asarray(correct, dtype=np.float64).flatten()
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(corr[m].mean() - conf[m].mean())
            total += m.mean() * gap
            rows.append((float(lo), float(hi), int(m.sum()), float(conf[m].mean()), float(corr[m].mean())))
    return float(total), rows


def capture_at_budget_expected(uncertainty, errors, budgets=(0.05, 0.10, 0.20)):
    """Tie-aware error capture at a budget: the expected fraction of all errors inside the round(b * n) most suspect
    units when tied scores are broken at random, so a coarse score (the boundary indicator has nine levels) is not
    credited or penalised for raster order. Equal to capture_at_budget when no scores tie at the cut."""
    u = np.asarray(uncertainty).flatten()
    e = np.asarray(errors).flatten().astype(np.float64)
    order = np.argsort(-u, kind="stable")                 # most suspect first
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

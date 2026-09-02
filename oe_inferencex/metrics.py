"""Ranking metrics for selective prediction. Pure numpy, no torch, so the
assessment layer can be imported by consumers that do not ship torch."""
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

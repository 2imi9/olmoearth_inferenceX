"""Cross-unit tests for signal comparisons. Pure numpy.

The statistics settled in exp13 and used by every experiment after it
(docs/method/protocol.md): exact sign tests on untied pairs, a sign-flip
permutation test on mean differences, a spatial block bootstrap within a
unit, a cluster bootstrap across tasks (exp21), and the one-vote-per-cluster
sign test that the river-clustered analyses use (summary_transfer, exp28,
exp30, exp31). Units are scenes or tiles; a "gain" is the reference signal's
E-AURC minus the candidate's on the same unit, so positive favours the
candidate.
"""
from math import comb

import numpy as np

from oe_inferencex.metrics import aurc_expected

TIE_TOL = 1e-12


def sign_test(wins, losses, alternative="two-sided"):
    """Exact binomial sign test on the untied pairs, p = 1/2.

    'two-sided' is the protocol's default (exp13.sign_test_p); 'greater' is
    the one-sided test used when a direction was preregistered (P(X >= wins))."""
    n = wins + losses
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n)
    if alternative != "two-sided":
        raise ValueError(alternative)
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wins_losses_ties(diffs, tol=TIE_TOL):
    d = np.asarray(diffs, dtype=np.float64)
    w, l = int((d > tol).sum()), int((d < -tol).sum())
    return w, l, len(d) - w - l


def paired_comparison(diffs, rng=None, n_perm=10000):
    """Per-unit gains -> wins/losses/ties, the two-sided exact sign test, median and mean gain,
    and (when an rng is given) the sign-flip permutation p of the mean gain (exp13)."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    w, l, t = wins_losses_ties(d)
    res = {"n": n, "w": w, "l": l, "t": t, "sign_p": sign_test(w, l),
           "median_gain": float(np.median(d)) if n else float("nan"),
           "mean_gain": float(d.mean()) if n else float("nan")}
    if n and rng is not None:
        perm = np.array([(d * rng.choice([-1, 1], n)).mean() for _ in range(n_perm)])
        res["perm_p"] = float((np.abs(perm) >= abs(d.mean())).mean())
    return res


def clustered_sign_test(gains, clusters, aggregate="mean", alternative="greater"):
    """One vote per cluster: the units are not independent draws (protocol, "Known limits").

    gains: {unit: gain}; clusters: {unit: cluster}. aggregate 'mean' votes the
    sign of the cluster's mean gain (the river test of exp28, exp30, exp31);
    'majority' votes the majority sign of the cluster's units
    (summary_transfer's clustered check). Returns the per-cluster statistic,
    wins, losses, ties and the exact sign test."""
    groups = {}
    for u, g in gains.items():
        groups.setdefault(clusters.get(u, u), []).append(float(g))
    if aggregate == "mean":
        stat = {c: float(np.mean(v)) for c, v in groups.items()}
    elif aggregate == "majority":
        stat = {c: float(sum(x > TIE_TOL for x in v) - sum(x < -TIE_TOL for x in v)) for c, v in groups.items()}
    else:
        raise ValueError(aggregate)
    w, l, t = wins_losses_ties(list(stat.values()))
    return {"per_cluster": stat, "w": w, "l": l, "t": t, "p": sign_test(w, l, alternative),
            "n_clusters": len(stat), "aggregate": aggregate, "alternative": alternative}


def block_bootstrap_indices(grid, block, rng):
    """Resample block x block patch blocks of a grid x grid patch map with replacement; flat patch indices (exp13)."""
    nb = grid // block
    blocks = rng.integers(0, nb * nb, nb * nb)
    idx = []
    for b in blocks:
        bi, bj = divmod(int(b), nb)
        rows = np.arange(bi * block, (bi + 1) * block)
        cols = np.arange(bj * block, (bj + 1) * block)
        idx.extend((r * grid + c) for r in rows for c in cols)
    return np.array(idx)


def cluster_bootstrap_difference(score_a, score_b, errors, clusters, n_boot=2000, seed=0):
    """Bootstrap over clusters of AURC(a) - AURC(b); negative favours a (exp21).

    Returns (2.5th percentile, 97.5th percentile, P(a better)). Resamples with
    no error are skipped."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(score_a).flatten(), np.asarray(score_b).flatten()
    err, cl = np.asarray(errors).flatten().astype(np.float64), np.asarray(clusters).flatten()
    ids = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        if err[sel].sum() == 0:
            continue
        diffs.append(aurc_expected(a[sel], err[sel]) - aurc_expected(b[sel], err[sel]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs < 0).mean())

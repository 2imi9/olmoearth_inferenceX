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
from oe_inferencex.signals import midrank_pct

TIE_TOL = 1e-12


def sign_test(wins, losses, alternative="two-sided"):
    """Exact binomial sign test on the untied pairs, p = 1/2.

    'two-sided' is the protocol's default (exp13.sign_test_p); 'greater' is
    the one-sided test used when a direction was preregistered (P(X >= wins)).

    Counts are converted to Python integers first: with numpy integers 2 ** n overflowed silently and the p-value came
    back negative (so "< 0.05" passed), infinite or 1.0 (audit 2026-09-22). Negative counts are refused; they
    used to return p = 0.0, the most significant result possible."""
    if any(float(x) != int(x) or int(x) < 0 for x in (wins, losses)):
        raise ValueError(f"wins and losses must be non-negative whole counts, got {wins!r} and {losses!r}")
    wins, losses = int(wins), int(losses)
    n = wins + losses
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n)
    if alternative != "two-sided":
        raise ValueError(alternative)
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rank_sum_test(x, y, alternative="less"):
    """Exact permutation rank-sum (Mann-Whitney) test that x tends to lie below y, ties counted as halves.

    The statistic is the number of (x, y) pairs with x < y plus half the tied pairs. Its null distribution over every
    relabelling of the pooled values is computed exactly: for a subset S of size len(x), the statistic equals the sum
    over S of each value's own score (the number of pooled values above it plus half the others tied with it) minus
    len(x)(len(x) - 1)/2, so counting subsets by that sum is a dynamic programme over integers. Exact for up to 200
    pooled values; above that a normal approximation with the tie correction is used and `method` says so.

    alternative "less": x below y, p = P(stat >= observed); "greater": x above y; "two-sided": twice the smaller tail.
    Added 2026-09-22 because exp70's P3 ran a binomial sign test over all 144 pairs of 12 x 12 tasks as though they
    were independent, and reported p = 6.9e-15 where this test gives 0.0055."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("rank_sum_test needs finite values")
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        return {"statistic": float("nan"), "p": float("nan"), "n_x": n1, "n_y": n2, "method": "undefined"}
    v = np.concatenate([x, y])
    n = v.size
    above = (v[None, :] > v[:, None]).sum(1)
    tied = (v[None, :] == v[:, None]).sum(1) - 1
    score2 = (2 * above + tied).astype(int)                  # twice each value's score, an integer
    stat = float((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum())
    obs2 = int(round(2 * stat + n1 * (n1 - 1)))              # twice (stat + C(n1, 2)): the subset's score sum
    if n <= 200:
        top = int(np.sort(score2)[::-1][:n1].sum())
        dp = np.zeros((n1 + 1, top + 1))
        dp[0, 0] = 1.0
        for s_i in score2:
            for k in range(min(n1, n) - 1, -1, -1):          # each value joins a subset at most once
                if s_i <= top:
                    dp[k + 1, s_i:] += dp[k, :top + 1 - s_i]
        total = dp[n1].sum()
        upper = dp[n1, obs2:].sum() / total                  # P(stat >= observed)
        lower = dp[n1, :obs2 + 1].sum() / total              # P(stat <= observed)
        method = "exact"
    else:
        from math import erf, sqrt
        mean = n1 * n2 / 2
        t = np.unique(v, return_counts=True)[1]
        var = n1 * n2 / 12 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1)))
        zu, zl = (stat - 0.5 - mean) / sqrt(var), (stat + 0.5 - mean) / sqrt(var)
        upper, lower = 0.5 * (1 - erf(zu / sqrt(2))), 0.5 * (1 + erf(zl / sqrt(2)))
        method = "normal approximation, tie-corrected"
    p = {"less": upper, "greater": lower, "two-sided": min(1.0, 2 * min(upper, lower))}.get(alternative)
    if p is None:
        raise ValueError(alternative)
    return {"statistic": stat, "p": float(p), "n_x": n1, "n_y": n2, "method": method}


def wins_losses_ties(diffs, tol=TIE_TOL):
    """(wins, losses, ties) over the finite gains. A NaN gain is an undefined unit, not a tie: it used to be counted
    as one, which diluted a verdict exactly as padding a comparison with empty units does."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    w, l = int((d > tol).sum()), int((d < -tol).sum())
    return w, l, len(d) - w - l


def paired_comparison(diffs, rng=None, n_perm=10000):
    """Per-unit gains -> wins/losses/ties, the two-sided exact sign test, median and mean gain,
    and (when an rng is given) the sign-flip permutation p of the mean gain (exp13)."""
    d = np.asarray(diffs, dtype=np.float64)
    n_undefined = int((~np.isfinite(d)).sum())
    d = d[np.isfinite(d)]                               # NaN gains are undefined units, excluded and counted
    n = len(d)
    w, l, t = wins_losses_ties(d)
    res = {"n": n, "n_undefined": n_undefined, "w": w, "l": l, "t": t, "sign_p": sign_test(w, l),
           "median_gain": float(np.median(d)) if n else float("nan"),
           "mean_gain": float(d.mean()) if n else float("nan")}
    if n and rng is not None:
        # A pattern whose |mean| equals the observed one exactly must count, and float rounding used to drop such
        # patterns, so on gains on a grid p came out too low (0.048 against an exact 0.056). The +1 is the standard
        # correction: a Monte Carlo permutation p is never exactly 0.
        m = abs(d.mean())
        perm = np.array([(d * rng.choice([-1, 1], n)).mean() for _ in range(n_perm)])
        hits = int((np.abs(perm) >= m - 1e-12 * max(1.0, m)).sum())
        res["perm_p"] = float((hits + 1) / (n_perm + 1))
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
        if np.isfinite(float(g)):                       # an undefined unit does not vote, nor wipe out its cluster
            groups.setdefault(clusters.get(u, u), []).append(float(g))
    if aggregate == "mean":
        stat = {c: float(np.mean(v)) for c, v in groups.items()}
    elif aggregate == "majority":
        # ties inside a cluster are ignored here; summary_transfer counted a tied scene as a loss, so the two can
        # vote differently when a cluster holds ties
        stat = {c: float(sum(x > TIE_TOL for x in v) - sum(x < -TIE_TOL for x in v)) for c, v in groups.items()}
    else:
        raise ValueError(aggregate)
    w, l, t = wins_losses_ties(list(stat.values()))
    return {"per_cluster": stat, "w": w, "l": l, "t": t, "p": sign_test(w, l, alternative),
            "n_clusters": len(stat), "aggregate": aggregate, "alternative": alternative}


def block_bootstrap_indices(grid, block, rng):
    """Resample block x block patch blocks of a grid x grid patch map with replacement; flat patch indices (exp13)."""
    if block < 1 or block > grid or grid % block:
        raise ValueError(f"block {block} must divide the grid {grid}; otherwise the edge patches can never be drawn")
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
    if not (a.size == b.size == err.size == cl.size):
        raise ValueError(f"score_a, score_b, errors and clusters must have one entry per unit; got {a.size}, {b.size}, "
                         f"{err.size} and {cl.size}")
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
    if diffs.size == 0:                                 # no resample held an error: the AURC difference is undefined
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs < 0).mean())


def spearman(x, y):
    """Spearman rank correlation with tie-averaged (mid)ranks; NaN when either input is constant.

    exp14.spearman ranks ties by position (double argsort), which lets a nine-level score such as the boundary
    indicator correlate with raster order; this form does not."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")                             # a NaN used to take its own top rank: finite, wrong, even sign-flipped
    rx, ry = midrank_pct(x), midrank_pct(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def paired_cluster_bootstrap(fn_a, clusters_a, fn_b, clusters_b, n_boot=1000, seed=0, min_units=20):
    """Bootstrap two statistics computed on DISJOINT subsets under one resampling of the shared clusters.

    Use this when the question is whether a statistic differs between two groups of units that share a clustering, for
    instance the units a published filter keeps against the units it deletes, grouped by region. Resampling each subset
    independently would give two intervals but no interval on their difference, because the two draws would not see the
    same regions.

    fn_a, fn_b   callables taking an index array into their own subset and returning a float
    clusters_a   cluster id per unit of subset A; clusters_b the same for subset B
    Returns {n, difference_lo, difference_hi, difference_mean, p_difference_gt_0, a_mean, b_mean}, or {n: 0} when no
    draw produced enough units in both subsets."""
    rng = np.random.default_rng(seed)
    ca, cb = np.asarray(clusters_a), np.asarray(clusters_b)
    if ca.dtype.kind != cb.dtype.kind:
        if ca.dtype.kind in "iufb" and cb.dtype.kind in "iufb":
            # numbers against numbers compare as numbers: casting to str turned 1 and 1.0 into '1' and '1.0' and split
            # every cluster in two (review of 2026-09-23; a regression of the str cast below)
            ca, cb = ca.astype(np.float64), cb.astype(np.float64)
        else:                                             # int ids against str ids matched nothing and returned {n: 0}
            ca, cb = ca.astype(str), cb.astype(str)
    ids = np.unique(np.concatenate([ca, cb]))
    idx_a = {g: np.flatnonzero(ca == g) for g in ids}
    idx_b = {g: np.flatnonzero(cb == g) for g in ids}
    diffs, vals_a, vals_b = [], [], []
    for _ in range(n_boot):
        pick = ids[rng.integers(0, len(ids), len(ids))]
        sel_a = np.concatenate([idx_a[g] for g in pick]) if len(pick) else np.array([], dtype=int)
        sel_b = np.concatenate([idx_b[g] for g in pick]) if len(pick) else np.array([], dtype=int)
        if len(sel_a) < min_units or len(sel_b) < min_units:
            continue
        a, b = fn_a(sel_a), fn_b(sel_b)
        if np.isfinite(a) and np.isfinite(b):
            vals_a.append(a); vals_b.append(b); diffs.append(a - b)
    if not diffs:
        return {"n": 0}
    d = np.sort(np.asarray(diffs, dtype=np.float64))
    return {"n": len(d), "difference_lo": float(np.quantile(d, 0.025)), "difference_hi": float(np.quantile(d, 0.975)),
            "difference_mean": float(d.mean()), "p_difference_gt_0": float((d > 0).mean()),
            "a_mean": float(np.mean(vals_a)), "b_mean": float(np.mean(vals_b))}

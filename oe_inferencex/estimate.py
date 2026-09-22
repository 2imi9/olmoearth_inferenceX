"""How wrong is the map? A design-based error rate from a labelled sample, and which windows to label.

The label-free layers rank windows for review and never say how wrong a map is, because that needs a reference.
This module is what a reference costs, measured in exp78 on the seven segmentation tasks of Ai2's suite with
every unit labelled so the intervals could be graded (docs/results/comparisons.md, exp78): label 300 windows
drawn by a design and the error rate comes back with an interval that covers the truth on 0.93 to 0.96 of
draws, about +/-3 points on a clean map and +/-5 on a messy one.

Three things a user needs, and one they must be stopped from doing.

- `sample_for_estimation` says which windows to label. The default design stratifies the map by confidence
  margin into quintiles and allocates the budget by Neyman's rule from the model's own confidence, so no label is
  spent estimating stratum rates; on exp78's tasks it narrowed the interval to a median 0.80 of a random sample's
  and to 0.63 on the cleanest map, with coverage intact everywhere. A plain random sample and a tile design are
  also offered, the last because that is how people actually label.
- `estimate_error_rate` turns the labels back into a rate with the interval the design earns: Wilson with a
  finite-population correction for a random sample, a stratified Wald interval otherwise, and for tile-sampled
  labels the ultimate-cluster interval, beside the naive one so the difference is visible. Labelling 19 tiles of
  16 windows and using the ordinary formula gave a "95%" interval that covered on 0.51 to 0.78 of draws; that is
  exp78's practical finding and the reason this function will not compute the naive interval alone.
- `estimate_from_indices` is for windows labelled without a design. It treats them as a random sample and first
  checks that they could be one: the median suspicion percentile of a random sample is 0.5, of the tool's own
  review set about 0.97, and labelling the review set then dividing gives two to six times the true rate on every
  task of exp78's export. A sample more than four standard deviations above a random one is refused with the
  number rather than estimated.

Every function is numpy only. The estimators are the ones exp78 ran; that script imports them from here.
"""
import numpy as np

Z95 = 1.959963984540054
N_STRATA = 5
MIN_PER_STRATUM = 2
M_PER_TILE = 16
REVIEW_SET_SIGMAS = 4.0           # a sample whose median suspicion percentile sits this many SDs above 0.5 is refused
Q_FLOOR = 0.02                    # the confidence design assumes no stratum is better than 98% right until labelled
MIN_TILES = 5                     # fewer tiles than this and a between-tile standard error is not an estimate


# ----------------------------------------------------------------------------- intervals
def wilson_interval(k, n, N=None):
    """Wilson score interval for k of n, with a finite-population correction when N is given.

    The correction (N - n)/(N - 1) multiplies the binomial variance term p(1-p)/n inside the root and nothing
    else. Until 2026-09-22 it scaled the whole half-width, including Wilson's z^2/(4n^2) term, so at k = 0 the
    interval no longer reached the shrunk centre's foot and its lower bound sat above zero: a sample with no
    errors ruled out a perfect map, 0.10% to 1.16% for 300 of 999. Away from k = 0 the two forms differ in the
    fifth decimal at the record's budgets. A census has no sampling error: at n >= N the interval is (p, p).
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    if N and n >= N:
        return p, p
    fpc2 = max((N - n) / (N - 1), 0.0) if N and N > 1 else 1.0
    d = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / d
    half = Z95 * np.sqrt(fpc2 * p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / d
    lo, hi = centre - half, centre + half
    return (0.0 if lo < 1e-12 else lo), (1.0 if hi > 1 - 1e-12 else hi)


def exact_coverage_srs(N, K, B):
    """The exact coverage of `wilson_interval` for a simple random sample of B from N units holding K errors: the
    hypergeometric probability of each error count k, summed over the k whose interval contains K/N. What the
    interval can achieve at (N, K, B), against which a Monte Carlo coverage is judged so that Wilson's
    discreteness is not mistaken for a defect (exp79, P5)."""
    import math
    theta = K / N
    lc = lambda n, r: math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
    tot = 0.0
    for k in range(max(0, B - (N - K)), min(B, K) + 1):
        lo, hi = wilson_interval(k, B, N)
        if lo <= theta <= hi:
            tot += math.exp(lc(K, k) + lc(N - K, B - k) - lc(N, B))
    return tot


def confidence_strata(margin, n_strata=N_STRATA):
    """Quantile strata of the confidence margin; stratum 0 is the least confident."""
    margin = np.asarray(margin, dtype=np.float64)
    q = np.quantile(margin, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(q, margin, side="right")


def stratified_mean_and_variance(err, strata, picked, sizes, N):
    """The stratified estimate of the population rate and the unbiased estimate of its variance,
    sum over strata of W_h^2 (1 - n_h/N_h) p_h (1 - p_h) / (n_h - 1), which is Cochran's (1 - f_h) s_h^2 / n_h.
    A stratum with fewer than two sampled units contributes its single unit to the estimate and nothing to the
    variance; the count of such strata is returned. Checked by enumeration in tests/test_estimate_exact.py."""
    err, strata, picked = np.asarray(err, dtype=np.float64), np.asarray(strata), np.asarray(picked)
    est = var = 0.0
    starved = 0
    for h, Nh in enumerate(sizes):
        m = strata[picked] == h
        nh = int(m.sum())
        Wh = Nh / N
        if Nh == 0:
            continue                                            # an empty stratum contributes nothing, legitimately
        if nh < MIN_PER_STRATUM:
            starved += 1
            if nh == 1:
                est += Wh * float(err[picked][m].mean())
            continue
        ph = float(err[picked][m].mean())
        est += Wh * ph
        var += Wh ** 2 * (1 - nh / Nh) * ph * (1 - ph) / (nh - 1)
    return est, max(var, 0.0), starved


def stratified_interval(err, strata, picked, sizes, N):
    """Stratified mean with a Wald interval, clipped to [0, 1]; the count of starved strata beside it.

    This is the form exp78 graded and is kept for that record. It collapses to zero width when every sampled
    stratum is pure (no error observed, or every unit wrong), which a clean map makes likely; the package's
    user-facing path uses `stratified_interval_wilson` instead."""
    est, var, starved = stratified_mean_and_variance(err, strata, picked, sizes, N)
    half = Z95 * np.sqrt(var)
    return est, max(0.0, est - half), min(1.0, est + half), starved


def stratified_interval_wilson(err, strata, picked, sizes, N):
    """The stratified estimate with a Wilson interval on its effective sample size (Korn and Graubard, 1998).

    The design-based variance v of the stratified mean is turned into the sample size a simple random sample
    would need for that variance, n_eff = p(1 - p)/v, and Wilson's interval is taken at (p, n_eff). Away from 0
    and 1 this agrees with the Wald form to the third decimal on exp78's tasks; at p = 0 or 1, where v is zero
    and Wald collapses to a point, n_eff is taken as the labelled count itself, which is the simple-random bound
    the design cannot improve on without an observed error. No finite-population correction is applied a second
    time: the strata's (1 - f_h) are already inside v. Returns (estimate, low, high, starved strata, n_eff)."""
    est, var, starved = stratified_mean_and_variance(err, strata, picked, sizes, N)
    n = int(np.asarray(picked).size)
    if var > 0 and 0 < est < 1:
        n_eff = est * (1 - est) / var
    else:
        n_eff = float(n)
    lo, hi = wilson_interval(est * n_eff, n_eff)
    return est, lo, hi, starved, n_eff


def cluster_interval(err, tile, picked, m=M_PER_TILE):
    """Ultimate-cluster interval for labels taken tile by tile: the mean of tile means, with the between-tile
    standard error and a normal quantile, as exp78 graded it. The design effect of the sample, 1 + (m - 1) rho
    at the sample's own windows per tile, is returned beside it. With fewer than two tiles there is no
    between-tile spread and no interval; the caller refuses before that."""
    err, tile, picked = np.asarray(err, dtype=np.float64), np.asarray(tile), np.asarray(picked)
    t, e = tile[picked], err[picked]
    means = np.array([e[t == u].mean() for u in np.unique(t)])
    if means.size < 2:
        raise ValueError(f"{means.size} tile(s) in the sample: a between-tile standard error needs at least two")
    est = float(means.mean())
    se = float(means.std(ddof=1) / np.sqrt(means.size))
    return est, max(0.0, est - Z95 * se), min(1.0, est + Z95 * se), design_effect(e, t, m)


def design_effect(err, tile, m=M_PER_TILE):
    """1 + (m - 1) rho, with rho the one-way intra-cluster correlation of the error indicator over tiles."""
    err, tile = np.asarray(err, dtype=np.float64), np.asarray(tile)
    per = [err[tile == t] for t in np.unique(tile)]
    per = [p for p in per if p.size >= 2]
    if len(per) < 3:
        return float("nan")
    sz = np.array([p.size for p in per], float)
    mu = np.array([p.mean() for p in per])
    n, gm = sz.sum(), float(np.concatenate(per).mean())
    msb = (sz * (mu - gm) ** 2).sum() / (len(per) - 1)
    msw = sum(((p - p.mean()) ** 2).sum() for p in per) / (n - len(per))
    m0 = (n - (sz ** 2).sum() / n) / (len(per) - 1)
    rho = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    return float(1 + (m - 1) * rho)


# ----------------------------------------------------------------------------- designs
def neyman_allocation(sizes, spread, budget, floor=MIN_PER_STRATUM):
    """Allocate a budget to strata in proportion to N_h * spread_h, with a floor per stratum and no stratum
    asked for more units than it has.

    The continuous allocation n_h = B N_h S_h / sum N_h S_h minimises the stratified variance exactly (a Lagrange
    condition, docs/method/protocol.md). This integer version floors it and hands the remainder to the strata
    with the most units left, which is what exp78 ran. Its cost, measured rather than guessed: on tiny cases the
    variance lands 1.3 to 2.3% above the best integer allocation (tests/test_estimate_exact.py); on exp78's
    real strata at the record's budget of 300 it is 0.1 to 0.5% above the continuous optimum (MADOS 0.47%,
    PASTIS S2 0.22%, m-cashew-plant 0.11%) and half that at 1,000. Kept as run so exp78's recorded numbers stay
    the package's output."""
    sizes = np.asarray(sizes, int)
    w = sizes.astype(float) * np.asarray(spread, float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sizes)) / len(sizes)
    n = np.maximum(floor, np.floor(w * budget).astype(int))
    n = np.minimum(n, sizes)
    while n.sum() > budget:
        n[np.argmax(n)] -= 1
    while n.sum() < budget and (sizes - n).sum() > 0:
        n[np.argmax(sizes - n)] += 1
    return n


def draw_stratified(rng, strata, sizes, allocation):
    """A without-replacement draw of `allocation[h]` units from each stratum."""
    idx = []
    for h, nh in enumerate(allocation):
        pool = np.flatnonzero(strata == h)
        idx.append(rng.choice(pool, min(int(nh), pool.size), replace=False))
    return np.concatenate(idx)


def sample_for_estimation(margin, budget, design="confidence", p1=None, tiles=None, per_tile=M_PER_TILE,
                          n_strata=N_STRATA, valid=None, seed=0):
    """Which windows to label so that `estimate_error_rate` can give an honest rate afterwards.

    margin  : per-window confidence margin, higher = more confident (assess's `arrays["confidence"]`, flattened)
    budget  : number of windows to label
    design  : "confidence" (default) stratifies by margin quintile and allocates by Neyman's rule from the model's
              own top-1 probability `p1`, which it needs; "proportional" stratifies and allocates by size;
              "random" is a simple random sample; "tiles" labels `per_tile` windows in each of budget // per_tile
              tiles drawn at random, which needs `tiles`, the tile id of every window
    valid   : optional mask of windows that exist; invalid windows are never sampled and never counted
    Returns the sample: its `indices` into the flattened window grid, and everything the estimator needs.
    """
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    pop = np.flatnonzero(valid)
    N = int(pop.size)
    if N == 0:
        raise ValueError("no valid windows to sample")
    budget = int(budget)
    if not 0 < budget <= N:
        raise ValueError(f"budget must be between 1 and the {N} valid windows, got {budget}")
    rng = np.random.default_rng(seed)
    out = {"design": design, "budget": budget, "n_population": N, "seed": seed}
    if design == "random":
        out["indices"] = pop[rng.choice(N, budget, replace=False)]
    elif design in ("confidence", "proportional"):
        s = confidence_strata(margin[pop], n_strata)
        sizes = np.bincount(s, minlength=n_strata)
        if int((sizes > 0).sum()) < 2:
            out["note"] = ("the confidence margin has too few distinct values to stratify (a hard mask, a constant "
                           "map); every window is in one stratum and this is a random sample in effect")
        if design == "confidence":
            if p1 is None:
                raise ValueError('design "confidence" needs p1, the top-1 probability per window; pass it, or use '
                                 'design="proportional"')
            p1 = np.asarray(p1, dtype=np.float64).ravel()[pop]
            # The model's own confidence stands in for the stratum's error rate, and it is overconfident exactly
            # where its errors are confident: a stratum whose top-1 probability is exactly 1.0 (float32 saturation,
            # ordinary in real maps) would be allocated the floor of two labels however large it is, and any error
            # in it then went unseen — a nominal 95% interval covered 26% of draws on such a map (audit,
            # 2026-09-22). No stratum is assumed better than 1 - Q_FLOOR right until the labels say so.
            q = np.array([max(Q_FLOOR, (1 - p1[s == h]).mean()) if (s == h).any() else 0.0 for h in range(n_strata)])
            spread = np.sqrt(np.clip(q * (1 - q), 1e-9, None))
        else:
            spread = np.ones(n_strata)
        alloc = neyman_allocation(sizes, spread, budget)
        local = draw_stratified(rng, s, sizes, alloc)
        out.update({"indices": pop[local], "strata": s, "strata_of_population": pop, "sizes": sizes.tolist(),
                    "allocation": alloc.tolist(), "n_strata": int(n_strata)})
    elif design == "tiles":
        if tiles is None:
            raise ValueError('design "tiles" needs `tiles`, the tile id of every window')
        tiles = np.asarray(tiles).ravel()
        if tiles.size != margin.size:
            raise ValueError(f"tiles has {tiles.size} entries for {margin.size} windows")
        # Tiles in a random order, up to `per_tile` windows from each, until the budget is met. A fixed count of
        # budget // per_tile tiles (exp78's D4) falls short wherever tiles hold fewer valid windows than per_tile:
        # on MADOS, whose tiles hold a median of six, 18 tiles gave 133 labels for a budget of 300.
        by = {t: pop[tiles[pop] == t] for t in np.unique(tiles[pop])}
        order = rng.permutation(np.array(list(by)))
        chosen, picked, total = [], [], 0
        for t in order:
            if total >= budget:
                break
            take = min(per_tile, by[t].size, budget - total)
            picked.append(rng.choice(by[t], take, replace=False))
            chosen.append(t)
            total += take
        out["indices"] = np.concatenate(picked)
        if len(chosen) < MIN_TILES:
            raise ValueError(
                f"the tile design would use {len(chosen)} tile(s) for this budget; a between-tile standard error needs "
                f"at least {MIN_TILES}. Use smaller tiles or fewer windows per tile")
        if total < budget:
            most = sum(min(per_tile, v.size) for v in by.values())
            raise ValueError(
                f"the tile design can label at most {most} windows here ({len(by)} tiles, up to {per_tile} each), short of "
                f"the budget of {budget}. Use smaller tiles, more windows per tile, or a smaller budget; a sample that "
                "silently falls short would report an interval for a budget nobody labelled")
        out.update({"tiles": tiles, "per_tile": int(per_tile), "n_tiles": int(len(chosen))})
    else:
        raise ValueError(f'design must be "confidence", "proportional", "random" or "tiles", got {design!r}')
    out["indices"] = np.asarray(out["indices"], int)
    return out


# ----------------------------------------------------------------------------- estimates
def estimate_error_rate(sample, wrong):
    """The error rate of the whole map from the labelled sample, with the interval its design earns.

    sample : what `sample_for_estimation` returned
    wrong  : 0/1 per labelled window, in the order of sample["indices"]: 1 where the label disagrees with the map
    """
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    idx = np.asarray(sample["indices"], int)
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} sampled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    N, design = int(sample["n_population"]), sample["design"]
    out = {"design": design, "n_labelled": int(idx.size), "n_population": N, "nominal_coverage": 0.95}
    if design == "random":
        lo, hi = wilson_interval(int(wrong.sum()), idx.size, N)
        out.update({"estimate": float(wrong.mean()), "low": lo, "high": hi, "method": "Wilson with finite-population correction"})
    elif design in ("confidence", "proportional"):
        # rebuild the per-unit arrays over the population so the stratified estimator sees the sizes it was drawn with
        pop = np.asarray(sample["strata_of_population"], int)
        pos = {int(g): i for i, g in enumerate(pop)}
        local = np.array([pos[int(g)] for g in idx])
        err = np.zeros(pop.size)
        err[local] = wrong
        est, lo, hi, starved, n_eff = stratified_interval_wilson(err, np.asarray(sample["strata"]), local, sample["sizes"], N)
        out.update({"estimate": est, "low": lo, "high": hi, "starved_strata": int(starved), "effective_n": n_eff,
                    "method": "stratified by confidence margin; Wilson interval on the design's effective sample size"})
        notes = []
        if est in (0.0, 1.0):
            notes.append(f"{'no' if est == 0 else 'every'} labelled window was wrong, so the design's variance is zero and "
                         f"the interval is the simple-random Wilson bound at {idx.size} labels; the stratification "
                         "cannot narrow it without an observed error")
        if starved:
            notes.append(f"{starved} of {sample['n_strata']} strata had fewer than {MIN_PER_STRATUM} labelled windows "
                         "and contribute no variance; the interval is narrower than it should be")
        if notes:
            out["warning"] = "; ".join(notes)
    elif design == "tiles":
        tiles = np.asarray(sample["tiles"]).ravel()
        err = np.zeros(tiles.size)
        err[idx] = wrong
        est, lo, hi, deff = cluster_interval(err, tiles, idx, m=int(sample.get("per_tile", M_PER_TILE)))
        nlo, nhi = wilson_interval(int(wrong.sum()), idx.size, N)
        n_tiles = int(sample["n_tiles"])
        out.update({"estimate": est, "low": lo, "high": hi, "design_effect": None if not np.isfinite(deff) else deff,
                    "n_tiles": n_tiles, "method": "ultimate cluster over tiles, normal quantile",
                    "naive_interval_if_treated_as_random": {"low": nlo, "high": nhi},
                    "warning": "labels taken tile by tile are not independent; the naive interval beside this one is "
                               "what the ordinary formula would say, and on exp78's tasks it covered 51 to 78% of "
                               "the time while claiming 95%"
                               + (f"; with {n_tiles} tiles the normal quantile is optimistic, a t on {n_tiles - 1} "
                                  "degrees of freedom would be wider" if n_tiles < 20 else "")})
    else:
        raise ValueError(f"unknown design {design!r}")
    out["half_width"] = (out["high"] - out["low"]) / 2
    return out


def review_set_threshold(n):
    """The median suspicion percentile above which n windows are not a random sample: 0.5 plus REVIEW_SET_SIGMAS
    standard deviations of the median of n uniform percentiles, 0.6/sqrt(n). At 300 labels that is 0.64. On
    exp78's units a random 300 sits at 0.50 +/- 0.03 (max 0.59 over 200 draws), the tool's review set near 0.97,
    and the confidence design's own sample at 0.54 to 0.72, where treating it as random would report 1.05 to
    1.87 times the true rate; 0.75, the first threshold, let that through."""
    return min(0.95, 0.5 + REVIEW_SET_SIGMAS * 0.6 / np.sqrt(max(int(n), 1)))


def review_set_check(indices, margin, valid=None):
    """Could these windows be a random sample of the map? The median suspicion percentile of the sample: 0.5 for a
    random draw, near 1 for the tool's own review set, which is built to be enriched for errors."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    idx = np.asarray(indices, int)
    pop = np.flatnonzero(valid)
    rank = np.empty(margin.size)
    rank[:] = np.nan
    order = pop[np.argsort(-margin[pop], kind="stable")]        # most confident first, so suspicion rises along it
    rank[order] = (np.arange(pop.size) + 0.5) / pop.size
    pct = rank[idx]
    med = float(np.nanmedian(pct))
    thr = review_set_threshold(idx.size)
    return {"median_suspicion_percentile": med, "threshold": thr, "looks_like_a_review_set": med > thr,
            "share_in_top_5pct": float(np.nanmean(pct > 0.95))}


def estimate_from_indices(indices, wrong, margin, valid=None):
    """An error rate from windows labelled without a design, treated as a simple random sample after checking
    that they could be one. The tool's own review set is refused: it is built to hold errors, and labelling it
    then dividing gave two to six times the true rate on every task of exp78's export."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    chk = review_set_check(indices, margin, valid)
    if chk["looks_like_a_review_set"]:
        raise ValueError(
            f"these {len(indices)} windows sit at a median suspicion percentile of {chk['median_suspicion_percentile']:.2f}, "
            f"above {chk['threshold']:.2f}, the most a random sample of that size reaches (a random sample sits at 0.50, "
            f"the review set near 0.97); {100 * chk['share_in_top_5pct']:.0f}% are in the top 5% most suspect. They are "
            "an enriched set, not a sample, and the rate they give is inflated. Draw a sample with "
            "sample_for_estimation, or pass a design.")
    sample = {"design": "random", "indices": np.asarray(indices, int), "n_population": int(valid.sum()), "budget": len(indices)}
    out = estimate_error_rate(sample, wrong)
    out["review_set_check"] = chk
    return out

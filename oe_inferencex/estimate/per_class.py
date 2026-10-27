"""Per-class accuracy (exp81): user's and producer's accuracy, class shares, and the class design's exact intervals."""
import math

import numpy as np

from oe_inferencex.common.hypergeom import hypergeom_interval
from oe_inferencex.common.sampling import _condition_guards
from oe_inferencex.estimate.intervals import (Z95, _union_interval, stratified_interval_wilson, stratified_mean_and_variance,
                                              wilson_interval)
from oe_inferencex.estimate.valid import PER_CLASS_VALID_NOTE, per_class_valid

PER_CLASS_EXACT_NOTE = ("under a random sample the per-class intervals printed are exact (exp115, graded on 39 fully "
                        "labelled maps): the class shares hold together with probability at least 95%, and so do the "
                        "producer's accuracies; a class whose share the map draws outside its interval is named drawn "
                        "too little or too much. Read the flag with its interval: a large class can be flagged for a gap "
                        "of a few points, and a rare class drawn at half its share may not be. A class under a few "
                        "percent of the map needs more labels: oe-inferencex plan --class-share S --classes L")
PER_CLASS_NOT_GRADED = ("The per-class intervals take the input conditions as strata. With conditions as strata, their "
                        "coverage has not been graded; each condition's error-rate interval is exact.")


# ----------------------------------------------------------------------------- per-class accuracy (exp81)
# What the map-accuracy literature says a producer owes (Olofsson et al. 2014; Stehman and Foody 2019; the CEOS
# LPV land-cover protocol 2025): not one error rate but, per class, the user's accuracy (of the windows the map
# calls c, how many are c), the producer's accuracy (of the windows that are c, how many the map found) and the
# error-adjusted share of the map that is c, each with a standard error. All of it comes from the same labelled
# sample `sample_for_estimation` draws; the map's own decisions give the class of every window. Preregistered in
# docs/plan/per_class_assessment.md.
MIN_PER_CLASS = 30                # a per-class interval on fewer labelled windows is reported with a warning
RARE_ERRORS = 5                   # a normal-theory per-class interval resting on 1 to 4 sampled errors is warned (review of
                                  # 2026-09-23: exp81's rare-error cells fail on the draws that catch one or two such errors)


def _wald(est, var):
    half = Z95 * np.sqrt(max(float(var), 0.0))
    return float(est), max(0.0, float(est) - half), min(1.0, float(est) + half)


def _wilson_eff(est, var, n_fallback):
    """Wilson's interval at the effective sample size p(1-p)/var (Korn and Graubard 1998), the form the package
    uses for the stratified overall rate. Where the estimated variance is zero or the estimate sits at 0 or 1,
    which happens whenever no sampled window of a class is wrong, Wald collapses to a point; the effective size
    then falls back to the labelled count of the class, the simple-random bound the design cannot beat without an
    observed error. exp81 graded the Wald form first: on 77 of 324 per-class cells of the classification suite it
    covered as little as 30% of draws, every one of them a class near 0 or 1."""
    est, var = min(max(float(est), 0.0), 1.0), max(float(var), 0.0)      # a weighted sum can round past 1
    if var > 0 and 0 < est < 1:
        n_eff = est * (1 - est) / var
    else:
        n_eff = float(max(int(n_fallback), 1))
    lo, hi = wilson_interval(est * n_eff, n_eff)
    return est, lo, hi


def _interval(est, var, n_fallback, form):
    if form == "wald":
        return _wald(est, var)
    if form == "wilson":
        return _wilson_eff(est, var, n_fallback)
    raise ValueError(f"interval must be 'wilson' or 'wald', got {form!r}")


def _ht_total(values, strata, picked, sizes):
    """Under stratified simple random sampling: the Horvitz-Thompson total of `values` and the unbiased estimate of
    its variance, sum_h N_h ybar_h and sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h. Only picked units are observed."""
    values, strata, picked = np.asarray(values, float), np.asarray(strata), np.asarray(picked)
    tot = var = 0.0
    for h, Nh in enumerate(sizes):
        m = strata[picked] == h
        nh = int(m.sum())
        if Nh == 0 or nh == 0:
            continue
        v = values[picked][m]
        tot += Nh * float(v.mean())
        if nh > 1:
            var += Nh ** 2 * (1 - nh / Nh) * float(v.var(ddof=1)) / nh
    return tot, var


def _ht_ratio(num, den, strata, picked, sizes):
    """A ratio of two Horvitz-Thompson totals with its linearised variance: z_k = (num_k - R den_k) / D_hat, and
    V(R) is the variance of the total of z (Cochran 1977, section 6.11, applied stratum by stratum)."""
    Yn, _ = _ht_total(num, strata, picked, sizes)
    Yd, _ = _ht_total(den, strata, picked, sizes)
    if Yd <= 0:
        return float("nan"), float("nan")
    R = Yn / Yd
    z = (np.asarray(num, float) - R * np.asarray(den, float)) / Yd
    _, var = _ht_total(z, strata, picked, sizes)
    return R, var


CLASS_DESIGN_METHOD = ("stratified by map class (Olofsson et al. 2014): each class's labels are a simple random sample of "
                       "it, so its user's accuracy has an exact hypergeometric interval; the class shares (areas) and "
                       "producer's accuracies are bounded by exact intervals on each reference class's count in each "
                       "stratum, at 1 - 0.05 / (L x S) with L classes and S strata not labelled in full, so they all hold "
                       "together with probability at least 95% (union bound); the points are Olofsson's stratified "
                       "estimates; the overall accuracy takes the error rate's interval, the union bound over the classes")
CLASS_NORMAL_NOTE = ("Beside the exact intervals, the normal-theory ones Olofsson et al. (2014) give from the same "
                     "labels (stratified estimates with their standard errors): narrower, and not exact, so they can "
                     "cover less than 95%, most of all for a rare class or one with few sampled errors.")
CLASS_DESIGN_EXACT_NOTE = ("under the class design the per-class intervals printed are exact: each user's accuracy at 95%, "
                           "and the class shares and producer's accuracies together at 95% (a union bound over every "
                           "class's count in every stratum, wide where many classes are thin); a class whose share the "
                           "map draws outside its interval is named drawn too little or too much")


def _class_design_exact(per, sample, strata, local, sizes, m_pop, r_s, C, L, N):
    """The class design's exact intervals, written into `per` (rows by class): each map class is a stratum labelled at
    random within it, so the count K_hj of reference class j in stratum h has an exact hypergeometric interval from
    the n_hj of its n_h labels. Taken at 1 - 0.05 / (L S), with S the strata not labelled in full (a census enters
    exactly), all of them hold together with probability at least 95%, and then so does every function of them that
    is monotone in each count: the share of class j, sum_h K_hj / N, between the sums of the ends; and its producer's
    accuracy A_j / (A_j + sum_{h != j} K_hj), with A_j = K_jj the windows of class j the map calls j, between
    A_lo / (A_lo + sum hi) and A_hi / (A_hi + sum lo). The user's accuracy of class c is its stratum's own exact
    interval at 95%. The points are Olofsson et al.'s (2014) stratified estimates."""
    values = list(sample["condition"]["values"])
    S = len(sizes)
    n_h = np.bincount(strata[local], minlength=S)
    counts = np.zeros((S, C), int)                                    # labelled windows of reference j in stratum h
    np.add.at(counts, (strata[local], r_s), 1)
    open_h = [h for h in range(S) if n_h[h] < sizes[h]]
    level = 1 - 0.05 / max(L * max(len(open_h), 1), 1)
    lo_K = np.zeros((S, C), int)
    hi_K = np.zeros((S, C), int)
    for h in range(S):
        Nh, nh = int(sizes[h]), int(n_h[h])
        for j in range(C):
            if nh == Nh:                                              # a census: the count itself
                lo_K[h, j] = hi_K[h, j] = int(counts[h, j])
                continue
            a, b = hypergeom_interval(int(counts[h, j]), nh, Nh, conf=level)
            lo_K[h, j] = int(math.floor(a * Nh + 1e-9))
            hi_K[h, j] = int(math.ceil(b * Nh - 1e-9))
    stratum_of = {int(v): h for h, v in enumerate(values) if v is not None}
    u = np.divide(counts, n_h[:, None], out=np.zeros((S, C)), where=n_h[:, None] > 0)
    for c in range(C):
        row = per[c]
        h = stratum_of.get(c)
        M_c = int(sizes[h]) if h is not None else 0
        if h is not None and n_h[h] > 0:
            lo, hi = hypergeom_interval(int(counts[h, c]), int(n_h[h]), M_c)
            row["user_accuracy"] = {"estimate": float(u[h, c]), "low": lo, "high": hi}
        k_lo, k_hi = int(lo_K[:, c].sum()), int(hi_K[:, c].sum())
        share = float((np.asarray(sizes) * u[:, c]).sum() / N)
        row["reference_share_exact"] = {"estimate": share, "low": k_lo / N, "high": k_hi / N}
        row["drawn"] = "too little" if M_c < k_lo else "too much" if M_c > k_hi else "within"
        a_lo, a_hi = (int(lo_K[h, c]), int(hi_K[h, c])) if h is not None else (0, 0)
        rest_lo, rest_hi = k_lo - a_lo, k_hi - a_hi
        found = float(sizes[h] * u[h, c]) if h is not None else 0.0
        row["producer_accuracy_exact"] = {
            "estimate": found / (share * N) if share > 0 else None,
            "low": a_lo / (a_lo + rest_hi) if a_lo + rest_hi > 0 else 0.0,
            "high": a_hi / (a_hi + rest_lo) if a_hi + rest_lo > 0 else 1.0}


def estimate_per_class(sample, reference, map_class, n_classes=None, interval="wilson", legend_size=None, valid=False):
    """User's accuracy, producer's accuracy and error-adjusted share per class, from the labelled sample.

    sample    : what `sample_for_estimation` returned (designs "random", "confidence", "proportional" or "condition";
                "sequential" is read as a random sample of the windows labelled)
    reference : the reference class of every labelled window, in the order of sample["indices"], integers >= 0
    map_class : the map's class of EVERY window (the flattened grid `sample` was drawn on); negative at no-data,
                and the count of non-negative entries must equal the population the sample was drawn from

    Under a random sample, the user's accuracy of class c has an exact hypergeometric interval (`hypergeom_interval`)
    on the labelled windows the map calls c, since those are a simple random sample of that class's windows, and
    `interval` does not apply to it; the class shares are post-stratified by
    map class (Olofsson et al. 2014, eq. 4 and 5) and the producer's accuracy follows their eq. 7. Under the
    confidence design every quantity is a ratio of Horvitz-Thompson totals over the margin strata with a
    linearised variance, because a class cuts across strata; the condition design runs the same estimators with the
    input conditions as strata, which no experiment has graded, and the method says so, while its overall accuracy is
    one minus `estimate`'s whole-map interval, the union bound over the conditions. With one condition the
    condition design is the random design, and gets the random design's estimators and method. Intervals are Wilson
    on the effective sample size (interval="wilson", the default; "wald" is the field's convention and is kept for the record, where exp81
    shows it collapsing to a point on classes with no sampled error). Tile samples are refused: the per-class
    cluster form is not graded yet. A class with fewer than MIN_PER_CLASS labelled windows carries a warning; a map class no
    labelled window fell in leaves the share estimates short by its weight, and that is said.

    Under a random sample each row also carries exact intervals (exp115): `reference_share_exact` (the labelled
    windows' share k/n, with the hypergeometric interval at 1 - 0.05/L, L classes, so all shares hold together at
    95%), `drawn` ("too little", "too much" or "within": the map's own count against that interval) and
    `producer_accuracy_exact` (a/k, with exact intervals for its two counts at 1 - 0.05/(2L), the map's count bounding
    it from above). L is `legend_size`, the number of classes the map's legend can hold, stated before the labels are
    read; by default the table's number of classes. Under a random sample the normal-theory `producer_accuracy` and
    `reference_share` are kept as before; the warnings `few errors` and `near census` concern those, and `few labels`
    all of them. Under the class design (the condition design by map class) the rows carry exact intervals too
    (`_class_design_exact`), and the normal-theory `producer_accuracy` and `reference_share` beside them are Olofsson
    et al.'s, estimate +/- 1.96 SE, whatever `interval` says, as `overall_accuracy_normal_theory` is.

    With `valid` (1.20.0, `estimate --valid --per-class`), each row also carries `valid`: its user's accuracy,
    producer's accuracy and reference share with intervals valid at every sample size, each on its own at 95%
    (estimate/valid.py: `class_quantities`), and the result `overall_accuracy_valid` and `valid_note`. Every other
    key is the result without it, unchanged."""
    if interval not in ("wilson", "wald"):     # before the class design sets its own form (the review of 1.17.2)
        raise ValueError(f"interval must be 'wilson' or 'wald', got {interval!r}")
    ref = np.asarray(reference).ravel()
    idx = np.asarray(sample["indices"], int).ravel()
    mc = np.asarray(map_class).ravel()
    if ref.size != idx.size:
        raise ValueError(f"{ref.size} reference labels for {idx.size} labelled windows")
    if mc.dtype.kind not in "iu":
        if mc.dtype.kind not in "fb" or not np.isfinite(mc).all() or not np.all(np.mod(mc, 1) == 0):
            raise ValueError("map_class must hold an integer class per window, negative where the map has no data")
    mc = mc.astype(int)
    if ((idx < 0) | (idx >= mc.size)).any():
        raise ValueError(f"a labelled window index lies outside the map's {mc.size} windows")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once in the sample; each window "
                         "is labelled once")
    if ref.dtype.kind not in "iu":
        # read as numbers first, as estimate_error_rate reads `wrong`: until 2026-10-06 np.mod ran on whatever came
        # in, so a column read with the csv module (strings) or a '?' raised numpy's TypeError, or a string-formatting
        # one from an object array, instead of this refusal
        try:
            ref = ref.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("reference classes must be integers") from None
        if not np.isfinite(ref).all() or not np.all(np.mod(ref, 1) == 0):
            raise ValueError("reference classes must be integers")
    ref = ref.astype(int)
    if (ref < 0).any():
        raise ValueError("reference classes must be >= 0; a window the reviewer could not label should not be in the sample")
    if sample.get("design") == "sequential":                 # read as random, as estimate_error_rate reads it
        sample = dict(sample, design="random", budget=int(np.asarray(sample["indices"]).size))
    design = sample["design"]
    if design == "tiles":
        raise ValueError("per-class accuracy from a tile sample is not graded yet (exp81 covers random and confidence "
                         "designs); label a random or confidence-designed sample")
    if design not in ("random", "confidence", "proportional", "condition"):
        raise ValueError(f"unknown design {design!r}")
    # With one input condition the condition design is the random design, drawn by the same rng call, so it gets the
    # random design's estimators, numbers and method (condition_spec.md 2.9); until 2026-09-29 it ran the
    # Horvitz-Thompson form with one stratum and said "not graded"
    srs = design == "random"
    if design == "condition":
        srs = len(_condition_guards(sample, idx)[2]) == 1          # a stray condition id would be dropped in silence
    if design == "random":
        pop = np.flatnonzero(mc >= 0)
    else:
        pop = np.asarray(sample["strata_of_population"], int)
        # the population is the sample's, so check that map_class describes it: until 2026-09-23 a class map of
        # another grid or with no-data inside the sampled population passed, or failed with an unrelated error
        if (pop >= mc.size).any() or (mc[pop] < 0).any() or int((mc >= 0).sum()) != pop.size:
            raise ValueError(f"map_class does not describe the population the sample was drawn from ({pop.size} windows, "
                             f"where map_class has {int((mc >= 0).sum())} with a class); pass the class map of the same "
                             "grid and no-data mask the sample was drawn on")
    N = int(pop.size)
    if N != int(sample["n_population"]):
        raise ValueError(f"map_class has {N} valid windows but the sample was drawn from {sample['n_population']}; pass the "
                         "map's class per window on the same grid, negative where the map has no data")
    if (mc[idx] < 0).any():
        raise ValueError("a labelled window sits where the map has no class")
    C = int(n_classes) if n_classes is not None else int(max(int(mc[pop].max()), int(ref.max())) + 1)
    if (ref >= C).any() or (mc[pop] >= C).any():
        raise ValueError(f"a class id reaches or exceeds n_classes={C}")
    used = np.union1d(np.unique(mc[pop]), np.unique(ref)).size
    L = C if legend_size is None else int(legend_size)
    if L < used:
        raise ValueError(f"the legend holds {L} classes, but the map and the labels use {used}")
    m_s, r_s = mc[idx], ref
    conf = np.zeros((C, C), int)                              # rows: map class, columns: reference class
    np.add.at(conf, (m_s, r_s), 1)
    N_map = np.bincount(mc[pop], minlength=C).astype(float)
    W = N_map / N
    per, warnings = {}, []
    by_class = False
    if srs:
        n_i = conf.sum(1).astype(float)
        u = np.divide(conf, n_i[:, None], out=np.zeros((C, C)), where=n_i[:, None] > 0)   # n_ij / n_i.
        fpc = np.where(N_map > 0, 1 - np.divide(n_i, N_map, out=np.zeros(C), where=N_map > 0), 0.0)
        denom = np.where(n_i > 1, n_i - 1, np.inf)
        v_u = u * (1 - u) * (fpc / denom)[:, None]                                          # V(u_ij) per cell
        unsampled = [int(i) for i in range(C) if N_map[i] > 0 and n_i[i] == 0]
        if unsampled:
            warnings.append(f"map class(es) {unsampled} have no labelled window; the error-adjusted reference shares "
                            f"(in the JSON) are short by their weight ({float(W[unsampled].sum()):.3f} of the map); the "
                            "exact intervals printed are not")
        share_est = (W[:, None] * u).sum(0)                                                 # A_j = sum_i W_i u_ij
        share_var = (W[:, None] ** 2 * v_u).sum(0)
        # the overall accuracy is `estimate`'s quantity with `estimate`'s interval, exact under a random draw: the
        # post-stratified form printed here until 2026-09-23 was never graded and covered 0.913 on one exp81 cell
        k_right = int(np.trace(conf))
        o_lo, o_hi = hypergeom_interval(k_right, int(idx.size), N)
        overall_row = {"estimate": k_right / idx.size, "low": o_lo, "high": o_hi}
        # Olofsson's post-stratified overall accuracy, sum_i W_i u_ii, kept as a point so the table adds up
        post_stratified = min(max(float((W * np.diag(u)).sum()), 0.0), 1.0)
        for c in range(C):
            row = {"map_share": float(W[c]), "n_labelled_map_class": int(n_i[c]), "n_labelled_reference_class": int(conf[:, c].sum())}
            if n_i[c] > 0:
                # exact (sixth amendment of exp81): the labelled windows the map calls c are a simple random sample
                # of that class, so the count of correct ones is hypergeometric and its tail inversion covers at
                # least 95% on every class; the finite-population Wilson form fell to 0.70 on a class with one
                # error in a hundred windows, and the score form to 0.83 on a small grid
                lo, hi = hypergeom_interval(int(conf[c, c]), int(n_i[c]), int(N_map[c]))
                row["user_accuracy"] = {"estimate": float(u[c, c]), "low": lo, "high": hi}
            else:
                row["user_accuracy"] = None
            # producer's accuracy: Olofsson et al. 2014, eq. 7, on the post-stratified counts, with each map
            # class's term carrying the same finite-population correction as the user's accuracy and the shares
            # (exp81's audit found the uncorrected form over-wide: median coverage 0.983 where its siblings sat
            # at 0.95, and eight times too wide on a near-census draw)
            Nhat_j = float((N_map * u[:, c]).sum())
            if Nhat_j > 0 and n_i[c] > 0:
                pa = N_map[c] * u[c, c] / Nhat_j
                t1 = N_map[c] ** 2 * (1 - pa) ** 2 * u[c, c] * (1 - u[c, c]) * fpc[c] / (n_i[c] - 1) if n_i[c] > 1 else 0.0
                others = [i for i in range(C) if i != c and n_i[i] > 1]
                t2 = pa ** 2 * sum(N_map[i] ** 2 * u[i, c] * (1 - u[i, c]) * fpc[i] / (n_i[i] - 1) for i in others)
                e, lo, hi = _interval(pa, (t1 + t2) / Nhat_j ** 2, conf[:, c].sum(), interval)
                row["producer_accuracy"] = {"estimate": e, "low": lo, "high": hi}
            elif N_map[c] == 0 and conf[:, c].sum() > 0:
                # the map never predicts a class the labels show exists: its producer's accuracy is exactly 0, the
                # worst there is (Olofsson's eq. 7), not missing; until 2026-09-23 it printed as n/a
                row["producer_accuracy"] = {"estimate": 0.0, "low": 0.0, "high": 0.0}
            else:
                row["producer_accuracy"] = None
            e, lo, hi = _interval(share_est[c], share_var[c], idx.size, interval)
            row["reference_share"] = {"estimate": e, "low": lo, "high": hi}
            per[int(c)] = row
        # exact intervals for every class (exp115, graded on 39 fully labelled maps): the labelled windows of
        # reference class c are hypergeometric over its K_c windows, so its share gets the exact interval at
        # 1 - 0.05/L (all classes together at least 95%); producer's accuracy A_c / K_c gets exact intervals for its
        # two counts at 1 - 0.05/(2L), and A_c <= M_c bounds it by the map's own count, so a class the map hardly draws,
        # and the labels show is common, gets a low upper end even when no labelled window is drawn in it. The ends are
        # whole counts (K/N), rounded outward, so rounding never narrows an interval
        n_lab = int(idx.size)
        down = lambda x: int(math.floor(x * N + 1e-9))
        up = lambda x: int(math.ceil(x * N - 1e-9))
        for c in range(C):
            k_c, a_c, M_c = int(conf[:, c].sum()), int(conf[c, c]), int(N_map[c])
            lo, hi = hypergeom_interval(k_c, n_lab, N, conf=1 - 0.05 / L)
            K_lo, K_hi = down(lo), up(hi)
            per[c]["reference_share_exact"] = {"estimate": k_c / n_lab, "low": K_lo / N, "high": K_hi / N}
            per[c]["drawn"] = "too little" if M_c < K_lo else "too much" if M_c > K_hi else "within"
            lo2, hi2 = hypergeom_interval(k_c, n_lab, N, conf=1 - 0.05 / (2 * L))
            alo, ahi = hypergeom_interval(a_c, n_lab, N, conf=1 - 0.05 / (2 * L))
            K_lo2, K_hi2, A_lo, A_hi = down(lo2), up(hi2), down(alo), up(ahi)
            # A_c <= M_c on every map, so min(A_lo, M_c) only lowers the lower end: the interval is never inverted
            # the point is a/k capped as the upper end is, by the map's own count over the class's estimated size
            # (A_c <= M_c), so it never sits outside its interval
            per[c]["producer_accuracy_exact"] = {
                "estimate": min(a_c / k_c, M_c * n_lab / (k_c * N)) if k_c else None,
                "low": min(A_lo, M_c) / K_hi2 if K_hi2 > 0 else 0.0,
                "high": min(1.0, A_hi / K_lo2, M_c / K_lo2) if K_lo2 > 0 else 1.0}
        method = ("random sample: exact hypergeometric interval per map class for user's accuracy; shares post-stratified by "
                  "map class and producer's accuracy by Olofsson et al. 2014 eq. 7 (points); exact intervals for each "
                  "class's share (Bonferroni over the classes) and producer's accuracy (exp115)")
    else:
        strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
        pos = {int(g): i for i, g in enumerate(pop)}
        local = np.array([pos[int(g)] for g in idx])
        m_pop = mc[pop]
        r_pop = np.full(N, -1, int)
        r_pop[local] = r_s
        right = np.zeros(N)
        right[local] = (m_s == r_s).astype(float)
        if design == "condition":
            # `estimate`'s interval under this design, the union bound over the conditions, whatever `interval` says
            # (as the random design's is exact). Until 2026-10-06 this took the stratified Wilson interval that
            # `estimate` retired for the design on 2026-09-29: on 4,000 windows 0.5% wrong beside 200 windows 50%
            # wrong, 150 labels each, it covered the true accuracy on 53% of draws, and it contradicted the error
            # rate printed beside it ([95.6%, 98.1%] against an error rate of [2.0%, 6.4%] from the same labels).
            # By the symmetry of the exact interval this is one minus the error rate's interval.
            n_h = np.bincount(strata[local], minlength=len(sizes))
            k_h = np.bincount(strata[local], weights=(m_s == r_s).astype(float), minlength=len(sizes)).round().astype(int)
            o_est, o_lo, o_hi, _ = _union_interval(k_h, n_h, sizes)
        elif interval == "wilson":                                 # `estimate`'s graded form, census included
            o_est, o_lo, o_hi, _, _ = stratified_interval_wilson(right, strata, local, sizes, N)
        else:
            o_est, o_lo, o_hi = _wald(*stratified_mean_and_variance(right, strata, local, sizes, N)[:2])
        overall_row = {"estimate": o_est, "low": o_lo, "high": o_hi}
        by_class = design == "condition" and (sample.get("condition") or {}).get("by") == "map class"
        # Under the class design the normal-theory fields are printed as Olofsson et al.'s (2014), estimate +/- 1.96 SE,
        # as the overall row beside them is; they were Wilson intervals on an effective n, one printed block in two forms
        # under one label (class 2's producer's accuracy 82.1-97.4 for Olofsson's 85.7-100.0; the bug hunt of
        # 2026-10-25). The user's accuracy is replaced by its exact interval (_class_design_exact).
        form = "wald" if by_class else interval
        for c in range(C):
            is_map = (m_pop == c).astype(float)
            is_ref = (r_pop == c).astype(float)
            both = is_map * is_ref
            row = {"map_share": float(W[c]), "n_labelled_map_class": int(conf[c].sum()), "n_labelled_reference_class": int(conf[:, c].sum())}
            ua, v = _ht_ratio(both, is_map, strata, local, sizes)
            row["user_accuracy"] = None if not np.isfinite(ua) else dict(zip(("estimate", "low", "high"), _interval(ua, v, conf[c].sum(), form)))
            pa, v = _ht_ratio(both, is_ref, strata, local, sizes)
            row["producer_accuracy"] = None if not np.isfinite(pa) else dict(zip(("estimate", "low", "high"), _interval(pa, v, conf[:, c].sum(), form)))
            if N_map[c] == 0 and conf[:, c].sum() > 0:
                row["producer_accuracy"] = {"estimate": 0.0, "low": 0.0, "high": 0.0}      # exactly 0, as above
            tot, v = _ht_total(is_ref, strata, local, sizes)
            row["reference_share"] = dict(zip(("estimate", "low", "high"), _interval(tot / N, v / N ** 2, idx.size, form)))
            per[int(c)] = row
        if by_class:
            _class_design_exact(per, sample, strata, local, sizes, m_pop, r_s, C, L, N)
            # beside the exact union bounds, the normal-theory interval agencies read (Olofsson et al. 2014), so the
            # width exactness costs is in view (the score review of 25 October 2026: 2.7 to 3.4 times as wide)
            normal_overall = dict(zip(("estimate", "low", "high"),
                                      _wald(*stratified_mean_and_variance(right, strata, local, sizes, N)[:2])))
        method = (CLASS_DESIGN_METHOD if by_class else
                  "stratified by input condition: ratios of Horvitz-Thompson totals with linearised variance; shares as "
                  "Horvitz-Thompson totals; with input conditions as strata these per-class intervals have not been graded; "
                  "the overall accuracy takes the error rate's interval (union bound over the conditions), which covers "
                  "at least 95% by construction"
                  if design == "condition" else
                  "stratified by confidence margin: ratios of Horvitz-Thompson totals with linearised variance; shares as Horvitz-Thompson totals")
    for c, row in per.items():
        notes, codes = [], []
        small = [k for k in ("n_labelled_map_class", "n_labelled_reference_class") if row[k] < MIN_PER_CLASS]
        if small:
            codes.append("few labels")
            notes.append(f"fewer than {MIN_PER_CLASS} labelled windows ({row['n_labelled_map_class']} the map calls this "
                         f"class, {row['n_labelled_reference_class']} the reference does); "
                         + ("the exact intervals are wide" if by_class else
                            "the exact intervals are wide, and the normal-theory ones, below about ten, not to be "
                            "trusted" if srs else "the interval is wide and, below about ten, not to be trusted"))
        if by_class:
            # the class design's printed intervals are exact (each class a stratum labelled at random within it): the
            # warnings below concern normal-theory intervals, which it does not print (the review of 24 October 2026)
            if notes:
                row["warning"] = "; ".join(notes)
                row["warning_codes"] = codes
            continue
        if N_map[c] == 0 and conf[:, c].sum() > 0:
            codes.append("never predicted")
            notes.append("the map never predicts this class, which the labels show exists: its producer's accuracy is "
                         "exactly 0 and its user's accuracy is undefined")
        # the rare-error case (exp81's second audit): a normal-theory interval built on one to four sampled errors of
        # the kind a quantity counts moves by a whole window's weight when one more or one fewer is drawn
        commissions, omissions = int(conf[c].sum() - conf[c, c]), int(conf[:, c].sum() - conf[c, c])
        counted = [("producer's accuracy", omissions), ("share", omissions + commissions)]
        if not srs:                                                # the random-design user's accuracy is exact
            counted.append(("user's accuracy", commissions))
        rare = [(q, n_err) for q, n_err in counted if 0 < n_err < RARE_ERRORS]
        if rare and N_map[c] > 0:
            codes.append("few errors")
            # each quantity with its own count: a user's accuracy resting on 4 commissions is not "27 to 31" errors
            # because the same class has 27 omissions (the exp86 round 6 audit)
            rests = "; ".join(f"the {q} rests on {n_err} sampled error{'' if n_err == 1 else 's'}" for q, n_err in rare)
            notes.append(f"{rests} ({commissions} the map calls this class wrongly, {omissions} of this class it misses); "
                         + ("one error more or fewer in the draw moves the error-adjusted estimate by a whole window's "
                            "weight, and its normal-theory interval can miss it; the exact intervals do not miss more "
                            "often than their level" if srs else
                            "one error more or fewer in the draw moves the estimate by a whole window's weight, and the "
                            "interval can miss it"))
        # exp81 measured one case where a nominal 95% interval covers well under 95%: a class nearly all of whose
        # windows are labelled, so the estimate takes a handful of values and a normal interval cannot follow it
        # (0.86-0.92 on five encoders' Togo classes). The thin-strata note below is a caution from the design: the
        # cells exp81 first attributed to it (Brick Kiln, Nandi Landsat) did not meet its criterion, and their
        # shortfall is the rare-error case, a class whose accuracy rests on a handful of errors that a draw misses
        # or catches at a large weight, for which the package does not yet warn (exp81's second audit)
        if N_map[c] > 0 and row["n_labelled_map_class"] >= 0.9 * N_map[c] and row["n_labelled_map_class"] < N_map[c]:
            codes.append("near census")
            notes.append("nearly every window of this class is labelled; apart from the user's accuracy under a random sample, "
                         "which is exact, the intervals are a rough guide, since the estimate can only take a few values "
                         "(exp81: coverage 0.86-0.92 on such classes)"
                         + ("; under this random sample that concerns the normal-theory intervals, not the exact ones"
                            if srs else ""))
        if not srs:
            f_all = idx.size / N
            thin = np.array([sizes[h] > 0 and (strata[local] == h).sum() / sizes[h] < 0.5 * f_all for h in range(len(sizes))])
            in_thin = float((thin[strata[m_pop == c]]).mean()) if (m_pop == c).any() else 0.0
            if in_thin > 0.5:
                codes.append("thin strata")
                notes.append(f"{100 * in_thin:.0f}% of this class sits in {'' if design == 'condition' else 'confidence '}strata the design samples at under half the "
                             "overall rate; if its errors are rare there they are often not drawn, and the interval is then "
                             "optimistic")
        if notes:
            row["warning"] = "; ".join(notes)
            row["warning_codes"] = codes
    out = {"design": design, "interval": "wald" if by_class else interval, "n_labelled": int(idx.size),
           "n_population": N, "n_classes": C,
           **({"legend_size": L} if srs or "reference_share_exact" in next(iter(per.values()), {}) else {}),
           "nominal_coverage": 0.95,
           "overall_accuracy": overall_row, "confusion_counts": conf.tolist(),
           "per_class": per, "method": method}
    if srs:
        out["overall_accuracy_post_stratified"] = post_stratified
    if by_class:
        out["overall_accuracy_normal_theory"] = normal_overall
    if warnings:
        out["warning"] = "; ".join(warnings)
    if valid:
        overall_v, rows_v = per_class_valid(sample, idx, ref, mc, pop, C, srs)
        for c, row in per.items():
            row["valid"] = rows_v[c]
        out["overall_accuracy_valid"] = overall_v
        out["valid_note"] = PER_CLASS_VALID_NOTE
    return out

"""The sampling primitives estimate and certify share: condition index and guards, review-set check, sequential order."""
import numpy as np

REVIEW_SET_SIGMAS = 4.0           # a sample whose mean suspicion percentile sits this many SDs above 0.5 is refused
UNRECORDED = "unrecorded"          # the name of the windows with no recorded condition; reserved


def _condition_index(condition, pop, n_windows, condition_names=None):
    """The input condition of every population window as an index c = 0..K-1: the recorded values present in the
    population in ascending order, then "unrecorded" last if any window has none. Returns (index per population
    window, values with None for unrecorded, names, sizes, notes)."""
    cond = np.asarray(condition).ravel()
    if cond.size != n_windows:
        raise ValueError(f"condition has {cond.size} entries for {n_windows} windows")
    if cond.dtype.kind == "b":
        cond = cond.astype(np.int64)
    elif cond.dtype.kind == "f":
        finite = np.isfinite(cond)
        if not np.all(np.mod(cond[finite], 1) == 0):
            raise ValueError("condition must hold an integer per window, negative where none is recorded; bin a "
                             "continuous layer, such as cloud fraction, first")
        cond = np.where(finite, cond, -1).astype(np.int64)      # NaN is unrecorded, as at the pixel level
    elif cond.dtype.kind in "iu":
        cond = cond.astype(np.int64)
    else:
        raise ValueError(f"condition must hold an integer per window, got dtype {cond.dtype}")
    cp = cond[pop]
    rec = cp >= 0
    vals = np.unique(cp[rec])
    if vals.size == 0:
        raise ValueError("no window of the population has a recorded condition: every one is unrecorded (a tie, or "
                         "no pixel with a value), so there is nothing to split the labels by")
    has_unrecorded = bool((~rec).any())
    K = int(vals.size) + int(has_unrecorded)
    index = np.full(cp.size, K - 1, dtype=np.int64)             # unrecorded windows form the last condition
    index[rec] = np.searchsorted(vals, cp[rec])
    given = {}
    for key, name in (condition_names or {}).items():
        try:
            v = int(key)
            if isinstance(key, (bool, np.bool_)) or float(key) != v:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"condition_names maps a condition value (an integer) to its name; got the key {key!r}") from None
        if v < 0:
            raise ValueError(f"condition_names gives a name to {v}, but a negative value means unrecorded")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"the name of condition {v} must be a non-empty string, got {name!r}")
        if name == UNRECORDED:
            raise ValueError(f'"{UNRECORDED}" is reserved for the windows with no recorded condition; name {v} otherwise')
        given[v] = name
    values = [int(v) for v in vals] + ([None] if has_unrecorded else [])
    names = [given.get(int(v), str(int(v))) for v in vals] + ([UNRECORDED] if has_unrecorded else [])
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"two conditions share the name {dup[0]!r}; condition names must be unique")
    notes = [f"value {v} ({name}) holds no window" for v, name in sorted(given.items()) if v not in set(vals.tolist())]
    sizes = np.bincount(index, minlength=K)
    return index, values, names, sizes, notes


# ----------------------------------------------------------------------------- estimates
def _condition_guards(sample, idx):
    """Check a sample that records an input condition against itself, and return (condition grid, values, names,
    sizes). The condition is fixed at sampling time and the sidecar is its source of truth, so an edited or mixed-up
    sample is refused here rather than estimated: a stray condition index would otherwise be dropped in silence."""
    grid = np.asarray(sample["condition_grid"])
    if grid.ndim != 1:
        raise ValueError(f"condition_grid must be one value per window of the flattened grid; it has shape {grid.shape}")
    if grid.dtype.kind not in "iu":
        if grid.dtype.kind != "f" or not np.isfinite(grid).all() or not np.all(np.mod(grid, 1) == 0):
            raise ValueError("condition_grid must hold an integer condition index per window, -1 outside the population")
    grid = grid.astype(np.int64)
    if (idx >= grid.size).any():
        raise ValueError(f"a labelled window index lies beyond the {grid.size} windows of the condition grid")
    if (grid[idx] < 0).any():
        raise ValueError(f"{int((grid[idx] < 0).sum())} labelled window(s) sit outside the population the condition "
                         "grid records")
    N = int(sample["n_population"])
    if int((grid >= 0).sum()) != N:
        raise ValueError(f"the condition grid holds {int((grid >= 0).sum())} population windows, but the sample was drawn "
                         f"from {N}")
    info = sample.get("condition") or {}
    values, names, sizes = list(info.get("values", [])), list(info.get("names", [])), list(info.get("sizes", []))
    K = len(names)
    if K == 0 or not len(values) == len(sizes) == K:
        raise ValueError(f"the sample's condition block lists {len(values)} values, {K} names and {len(sizes)} sizes; "
                         "they must be one per condition")
    if int(grid.max()) >= K:
        raise ValueError(f"the condition grid holds a condition index of {int(grid.max())}, but the sample names {K} "
                         "conditions")
    counted = np.bincount(grid[grid >= 0], minlength=K)
    if [int(s) for s in sizes] != counted.tolist():
        raise ValueError(f"the condition sizes {[int(s) for s in sizes]} are not the counts of the condition grid "
                         f"{counted.tolist()}")
    if sample.get("design") == "condition":
        if int(sample.get("n_strata", -1)) != K:
            raise ValueError(f"a condition sample has one stratum per condition: n_strata is {sample.get('n_strata')}, "
                             f"with {K} conditions")
        strata = np.asarray(sample["strata"])
        if strata.size and (int(strata.min()) < 0 or int(strata.max()) >= K):
            raise ValueError(f"a stratum id lies outside the {K} conditions (0 to {K - 1})")
        sop = np.asarray(sample["strata_of_population"], int)
        if (sop.size != strata.size or sop.size != N or (sop < 0).any() or (sop >= grid.size).any()
                or not np.array_equal(strata.astype(np.int64), grid[sop])):
            raise ValueError("the strata of this condition sample are not the conditions of its population windows")
        if [int(s) for s in sample["sizes"]] != counted.tolist() or int(np.sum(sample["sizes"])) != N:
            raise ValueError(f"the strata sizes {list(sample['sizes'])} do not add up to the population of {N} windows "
                             "as the condition grid counts it")
        n_c = np.bincount(grid[idx], minlength=K)
        if n_c.tolist() != [int(a) for a in sample["allocation"]]:
            raise ValueError(f"the labelled windows per condition {n_c.tolist()} are not the design's allocation "
                             f"{[int(a) for a in sample['allocation']]}")
    return grid, values, names, [int(s) for s in sizes]


def review_set_threshold(n, N=None):
    """The MEAN suspicion percentile above which n windows drawn from N are not a random sample: 0.5 plus
    REVIEW_SET_SIGMAS standard deviations of the mean of n uniform percentiles drawn without replacement,
    sqrt(1/12) sqrt((N - n) / ((N - 1) n)); about 0.567 at 300 labels of a large map.

    Until 2026-09-24 the statistic was the median (threshold 0.5 + 4 x 0.6/sqrt(n), 0.64 at 300). Under heavy ties
    the median cannot separate a random sample from the tool's review set: both land inside a large tied block at
    the suspect end. With ties ranked in a random order the percentiles are uniform again, and the mean separates
    them (a random sample 0.50, a review set inside a 60% block about 0.70), which the verification of the second
    review found the median did not. On exp78's units the review set and the confidence design's own sample are
    still refused, and two hundred random draws pass (the research record's tests/test_estimate.py)."""
    n = max(int(n), 1)
    fpc = 1.0 if N is None or int(N) <= 1 else max(int(N) - n, 0) / (int(N) - 1)
    return min(0.95, 0.5 + REVIEW_SET_SIGMAS * np.sqrt(1.0 / 12.0) * np.sqrt(fpc / n))


def review_set_check(indices, margin, valid=None):
    """Do these windows look like a review list rather than a random sample of the map? The MEAN suspicion
    percentile of the sample, against review_set_threshold: 0.5 for a random draw, near 1 for the tool's own review
    set, which is built to be enriched for errors. Passing does not make the windows a random sample: labels clustered
    by tile pass, and labels from the confident end pass this flag (certify refuses them by the mirror-image flag,
    `looks_drawn_from_the_confident_end`, from 1.16.0; estimate does not). The median is reported beside it and decides
    nothing. A census (every valid window, once) is never
    an enriched set: at a census the threshold is exactly 0.5 and the mean is 0.5 up to rounding, and until the
    release check of 24 September a sum that rounded up refused 2 to 5% of censuses as review sets."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    # A window with no finite margin is not in the population. assess's confidence array is NaN at no-data, and
    # argsort put NaN at the suspect end, so at 40% no-data the tool's own review set passed as a random sample.
    valid = valid & np.isfinite(margin)
    idx = np.asarray(indices, int).ravel()
    pop = np.flatnonzero(valid)
    # Tied windows are ranked in a fixed random order, independent of raster position, so a random sample's
    # percentiles are uniform whatever the ties. Mid-ranks, used until 2026-09-23, gave a tied block one percentile
    # near its middle, and when half the map or more tied at the suspect end a genuine random sample's median fell
    # inside that block and was refused as a review set (27 of 50 draws at 50% tied, 50 of 50 at 60%).
    s = -margin[pop]
    order = np.lexsort((np.random.default_rng(0).random(pop.size), s))
    ranks = np.empty(pop.size)
    ranks[order] = np.arange(pop.size)
    rank = np.full(margin.size, np.nan)
    rank[pop] = (ranks + 0.5) / pop.size
    on_grid = (idx >= 0) & (idx < margin.size)                   # a negative index used to wrap to the last window
    pct = np.full(idx.size, np.nan)
    pct[on_grid] = rank[idx[on_grid]]
    inside = np.isfinite(pct)
    med = float(np.median(pct[inside])) if inside.any() else float("nan")
    mean = float(np.mean(pct[inside])) if inside.any() else float("nan")
    thr = review_set_threshold(int(inside.sum()), pop.size)
    census = bool(inside.all() and np.unique(idx[inside]).size == pop.size)
    return {"mean_suspicion_percentile": mean, "median_suspicion_percentile": med, "threshold": thr,
            # 1e-12 is far below any real enrichment (one standard error at 300 labels is 0.017) and far above the
            # rounding of a mean of percentiles
            "looks_like_a_review_set": bool(np.isfinite(mean) and not census and mean > thr + 1e-12),
            # the mirror image, which certify refuses too (1.16.0): labels drawn from the confident end make every zone
            # look cleaner than it is, and a mean this low is as unlikely for a random draw as a review set's
            "looks_drawn_from_the_confident_end": bool(np.isfinite(mean) and not census and mean < 1 - thr - 1e-12),
            "share_in_top_5pct": float(np.mean(pct[inside] > 0.95)) if inside.any() else float("nan"),
            "n_outside_population": int((~inside).sum()), "n_duplicated": int(idx.size - np.unique(idx).size)}


def sequential_order(n_population, seed):
    """A random order of the population, the same on every system and every numpy release: the windows sorted by
    64-bit keys read straight from the PCG64 bit generator seeded with `seed` (SeedSequence and PCG64 are fixed
    algorithms; Generator methods such as permutation may change their streams between releases). Any prefix of it
    is a simple random sample, and a longer prefix extends a shorter one, which is what lets a reviewer add labels."""
    n = int(n_population)
    keys = np.random.PCG64(int(seed)).random_raw(n)
    return np.lexsort((np.arange(n), keys))

"""Which windows to label: the sampling designs, their allocations and sample_for_estimation."""
import numpy as np

from oe_inferencex.common.sampling import _condition_index

N_STRATA = 5
MIN_PER_STRATUM = 2
M_PER_TILE = 16
Q_FLOOR = 0.02                    # the confidence design assumes no stratum is better than 98% right until labelled
MIN_TILES = 5                     # fewer tiles than this and a between-tile standard error is not an estimate
CONFIDENCE_REFUSAL = ("that design allocates labels from the model's confidence, which can overstate the accuracy of "
                      "a condition read from an input combination the model was not trained on: it did on PASTIS for a "
                      "probe trained on radar plus optical and run on radar alone, though not on CropHarvest China 6 "
                      "(exp88). The tool cannot tell which inputs the model was trained on. Use --design condition (the "
                      "default with --condition) or --design random.")


def confidence_strata(margin, n_strata=N_STRATA):
    """Quantile strata of the confidence margin; stratum 0 is the least confident."""
    margin = np.asarray(margin, dtype=np.float64)
    q = np.quantile(margin, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.searchsorted(q, margin, side="right")


# ----------------------------------------------------------------------------- designs
def neyman_allocation(sizes, spread, budget, floor=MIN_PER_STRATUM):
    """Allocate a budget to strata in proportion to N_h * spread_h, with a floor per stratum and no stratum
    asked for more units than it has.

    The continuous allocation n_h = B N_h S_h / sum N_h S_h minimises the stratified variance exactly (a Lagrange
    condition, the research record's docs/method/protocol.md). This integer version floors it and hands the
    remainder to the strata with the most units left, which is what exp78 ran. Its cost, measured rather than
    guessed: on tiny cases the variance lands 1.3 to 2.3% above the best integer allocation
    (tests/test_estimate_exact.py); on exp78's real strata at the record's budget of 300 it is 0.1 to 0.5% above the
    continuous optimum (MADOS 0.47%, PASTIS S2 0.22%, m-cashew-plant 0.11%) and half that at 1,000. Kept as run so
    exp78's recorded numbers stay the package's output."""
    sizes = np.asarray(sizes, int)
    need = int(np.minimum(sizes, floor).sum())
    if budget < need:
        raise ValueError(
            f"a budget of {budget} cannot give each of the {int((sizes > 0).sum())} non-empty strata its {floor} labels "
            f"(needs {need}). Until 2026-09-22 the allocation zeroed the least confident strata instead, and the "
            "estimate then treated the most error-prone part of the map as error-free")
    w = sizes.astype(float) * np.asarray(spread, float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(sizes)) / len(sizes)
    n = np.maximum(floor, np.floor(w * budget).astype(int))
    n = np.minimum(n, sizes)
    while n.sum() > budget:
        n[np.argmax(n)] -= 1
    while n.sum() < budget and (sizes - n).sum() > 0:
        n[np.argmax(sizes - n)] += 1
    return n


def equal_allocation(sizes, budget, floor=MIN_PER_STRATUM):
    """The same number of labels for every stratum, by water-filling: a stratum too small for an equal share is
    labelled in full, and the rest of the budget is shared equally by the others, again and again until no stratum
    left is that small. The input conditions of the "condition" design are its strata. It never reads the model's
    confidence, which can overstate the accuracy of a condition read from an input combination the model was not
    trained on: it did on PASTIS for a probe trained on radar plus optical and run on radar alone, though not on
    CropHarvest China 6 (exp88).

    Returns n_h with sum n_h = budget and n_h <= N_h; strata not labelled in full differ by at most one label, and
    n_h >= min(N_h, 2) whenever budget >= sum min(N_h, 2). A budget below sum min(N_h, floor) is refused, as by
    `neyman_allocation`. `neyman_allocation(sizes, 1 / sizes, budget)` is not an equal split: it gives (33, 7, 20)
    for sizes (100, 7, 50) at 60, where this gives (27, 7, 26)."""
    sizes = np.asarray(sizes, int)
    N = int(sizes.sum())
    need = int(np.minimum(sizes, floor).sum())
    if budget > N:
        raise ValueError(f"a budget of {budget} is more than the {N} units of the strata")
    if budget < need:
        raise ValueError(
            f"a budget of {budget} cannot give each of the {int((sizes > 0).sum())} non-empty strata its {floor} labels "
            f"(needs {need})")
    n = np.zeros(sizes.size, int)
    open_ = np.ones(sizes.size, bool)
    left = int(budget)
    while open_.any():
        share = left / int(open_.sum())
        small = open_ & (sizes <= share)
        if not small.any():
            break
        n[small] = sizes[small]                                 # too small for an equal share: a census
        left -= int(sizes[small].sum())
        open_ &= ~small
    k = int(open_.sum())
    if k:
        n[open_] = left // k
        rem = left - (left // k) * k
        # The remainder goes one label each to the `rem` open strata with the most units left, lowest index on ties.
        # The spec (condition_spec.md 2.5) read "one label at a time to the open stratum with most units left",
        # re-ranked after each label; that gives (4, 2, 2) for sizes (6, 3, 3) at 8, two strata not labelled in full
        # that differ by two, against the guarantee it states. No stratum gets more than one label of the remainder.
        opened = np.flatnonzero(open_)
        rank = opened[np.lexsort((opened, -(sizes[opened] - n[opened])))]
        n[rank[:rem]] += 1
    return n


CLASS_FLOOR = 50       # labels per map class at least, where it has them (Olofsson et al. 2014 suggest 50 to 100)


def class_allocation(sizes, budget, floor=CLASS_FLOOR):
    """The labels of a sample stratified by map class, as Olofsson et al. (2014, section 5.1.3) advise: every class gets
    `floor` labels, or all its windows if it has fewer, so that each user's accuracy is estimated, and the rest of the
    budget goes to the classes in proportion to their size, so that the overall accuracy and the class areas stay
    precise; no class gets more labels than windows. A budget below the floors is refused.

    Returns n_h with sum n_h = budget and min(N_h, floor) <= n_h <= N_h. The proportional part is allocated by largest
    remainder over the room each class has left, and again while a class is filled, so the sum is exact."""
    sizes = np.asarray(sizes, int)
    N = int(sizes.sum())
    if int(floor) < 1:
        raise ValueError(f"the floor is at least one label per class, got {floor}")
    if budget > N:
        raise ValueError(f"a budget of {budget} is more than the {N} units of the strata")
    n = np.minimum(sizes, int(floor))
    if budget < int(n.sum()):
        raise ValueError(f"a budget of {budget} cannot give each of the {int((sizes > 0).sum())} map classes its "
                         f"{floor} labels (needs {int(n.sum())}); lower --min-per-class or raise --budget")
    left = int(budget) - int(n.sum())
    while left > 0:
        room = sizes - n
        open_ = room > 0
        want = left * sizes[open_] / sizes[open_].sum()
        add = np.minimum(np.floor(want).astype(int), room[open_])
        rest = left - int(add.sum())
        if rest > 0:                                  # largest remainders, among classes that still have room
            frac = np.where(add < room[open_], want - np.floor(want), -1.0)
            order = np.lexsort((np.flatnonzero(open_), -frac))
            for j in order[:rest]:
                if add[j] < room[open_][j]:
                    add[j] += 1
        n[open_] += add
        left = int(budget) - int(n.sum())
    return n


def draw_stratified(rng, strata, sizes, allocation):
    """A without-replacement draw of `allocation[h]` units from each stratum."""
    idx = []
    for h, nh in enumerate(allocation):
        pool = np.flatnonzero(strata == h)
        idx.append(rng.choice(pool, min(int(nh), pool.size), replace=False))
    return np.concatenate(idx)


def sample_for_estimation(margin, budget, design="confidence", p1=None, tiles=None, per_tile=M_PER_TILE,
                          n_strata=N_STRATA, valid=None, seed=0, condition=None, condition_names=None, stable=False,
                          class_floor=None):
    """Which windows to label so that `estimate_error_rate` can give an honest rate afterwards.

    margin  : per-window confidence margin, higher = more confident (assess's `arrays["confidence"]`, flattened)
    budget  : number of windows to label
    design  : "confidence" (default) stratifies by margin quintile and allocates by Neyman's rule from the model's
              own top-1 probability `p1`, which it needs; "proportional" stratifies and allocates by size;
              "random" is a simple random sample; "sequential" is the first `budget` windows of a fixed random order
              (`sequential.sequential_order`), so that labels can be added and certified at every look; "tiles"
              takes tiles in a random order and labels up to
              `per_tile` windows of each until the budget is met, which needs `tiles`, the tile id of every window;
              "condition" stratifies by input condition and splits the labels equally (`equal_allocation`), which
              needs `condition`
    valid   : optional mask of windows that exist; invalid windows are never sampled and never counted
    condition : optional input condition of every window (assess's `arrays["condition"]`, flattened), an integer,
              negative where none is recorded. The "condition" design draws within it; "random" records it and
              draws as without it; the other designs refuse it. Windows with no recorded condition stay in the
              population as one more condition, "unrecorded", placed last
    condition_names : optional {value: name}; names are unique and non-empty, "unrecorded" is reserved, and the
              default name is str(value)
    Returns the sample: its `indices` into the flattened window grid, and everything the estimator needs.
    """
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    if valid.size != margin.size:
        raise ValueError(f"valid has {valid.size} entries for {margin.size} windows")
    if condition is None and condition_names:
        raise ValueError("condition_names names the values of `condition`, and no condition was given")
    if condition is not None and design in ("confidence", "proportional"):
        raise ValueError(f"the {design} design does not take a condition: {CONFIDENCE_REFUSAL}")
    if condition is not None and design == "tiles":
        raise ValueError("the tiles design does not take a condition: a tile can span conditions, and the tile interval "
                         "is not graded per condition.")
    if condition is not None and design == "sequential":
        raise ValueError("the sequential design does not take a condition: certify reads it as one random order of the "
                         "whole map")
    if condition is None and design == "condition":
        raise ValueError('design "condition" needs `condition`, the input condition of every window (--condition)')
    # a window with no finite margin is not in the population: assess's confidence is NaN at no-data, and until
    # 2026-09-23 those windows were sampled (38 of 100 on a map with 38% no-data), as review_set_check and
    # zone_order already excluded them
    valid = valid & np.isfinite(margin)
    pop = np.flatnonzero(valid)
    N = int(pop.size)
    if N == 0:
        raise ValueError("no valid windows to sample")
    budget = int(budget)
    if not 0 < budget <= N:
        raise ValueError(f"budget must be between 1 and the {N} valid windows, got {budget}")
    rng = np.random.default_rng(seed)
    out = {"design": design, "budget": budget, "n_population": N, "seed": seed}
    if condition is not None:
        c_pop, c_values, c_names, c_sizes, c_notes = _condition_index(condition, pop, margin.size, condition_names)
    if stable and design in ("random", "condition"):
        # the draw a certificate rests on (sample --seed-from-map): taken from the fixed random order of the population
        # that sequential_order reads straight from PCG64, the same on every numpy release, where Generator.choice's
        # stream may change between releases and an honest certificate would then fail to verify
        from oe_inferencex.common.sampling import sequential_order
        order = sequential_order(N, seed)
    if design == "random":
        # the same draw with or without a condition
        out["indices"] = pop[order[:budget]] if stable else pop[rng.choice(N, budget, replace=False)]
    elif design == "sequential":
        # the first `budget` windows of a fixed random order (sequential.sequential_order), read in that order; a
        # reviewer who adds labels takes the next ones, and certify then uses the sequential rule, valid at every look
        from oe_inferencex.common.sampling import sequential_order
        out["indices"] = pop[sequential_order(N, seed)[:budget]]
        out["first_budget"] = budget
    elif design == "condition":
        # a simple random sample within each condition; with one condition this is the random design's own draw
        # (draw_stratified with one stratum returns the indices of rng.choice(N, budget))
        # equal across input conditions; across map classes (class_floor given: the class design), Olofsson's
        alloc = equal_allocation(c_sizes, budget) if class_floor is None else class_allocation(c_sizes, budget,
                                                                                                class_floor)
        local = (np.concatenate([order[c_pop[order] == h][:int(nh)] for h, nh in enumerate(alloc)]) if stable else
                 draw_stratified(rng, c_pop, c_sizes, alloc))
        out.update({"indices": pop[local], "strata": c_pop, "strata_of_population": pop, "sizes": c_sizes.tolist(),
                    "allocation": alloc.tolist(), "n_strata": int(c_sizes.size)})
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
            p1 = np.asarray(p1, dtype=np.float64).ravel()
            if p1.size != margin.size:
                raise ValueError(f"p1 has {p1.size} entries for {margin.size} windows")
            p1 = p1[pop]
            if not np.isfinite(p1).all():
                # max(Q_FLOOR, nan) is Q_FLOOR in Python, so one NaN silently set its stratum's assumed error rate
                raise ValueError(f"p1 is not finite at {int((~np.isfinite(p1)).sum())} valid window(s)")
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
        # grouped by one stable sort, O(N log N): a mask per tile is O(tiles x windows), minutes on a 4M-window map
        tp = tiles[pop]
        o = np.argsort(tp, kind="stable")
        u, starts, counts = np.unique(tp[o], return_index=True, return_counts=True)
        by = {t: pop[o[a:a + c]] for t, a, c in zip(u.tolist(), starts, counts)}
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
        out.update({"tiles": tiles, "per_tile": int(per_tile), "n_tiles": int(len(chosen)),
                    # the population size of every tile, over valid windows: the ratio estimator weights by it
                    "tile_ids": [int(t) for t in by], "tile_valid_sizes": [int(v.size) for v in by.values()]})
    else:
        raise ValueError(f'design must be "confidence", "proportional", "random", "sequential", "tiles" or "condition", '
                         f'got {design!r}')
    out["indices"] = np.asarray(out["indices"], int)
    if condition is not None:
        grid = np.full(margin.size, -1, dtype=np.int64)          # condition index per window, -1 outside the population
        grid[pop] = c_pop
        out["condition"] = {"values": c_values, "names": c_names, "sizes": c_sizes.tolist(),
                            "n_labelled": np.bincount(grid[out["indices"]], minlength=c_sizes.size).tolist(),
                            "allocation_rule": (None if design != "condition" else "equal" if class_floor is None
                                                else f"proportional, at least {int(class_floor)} per class")}
        out["condition_grid"] = grid
        notes = list(c_notes)
        if c_sizes.size == 1:
            notes.insert(0, "one condition covers the whole population; this is a simple random sample")
        if notes:
            out["note"] = "; ".join(notes)
    return out


# The rows of a class-design sample run class by class, as the draw makes them, and three columns name the class: hiding
# map_class alone, as the docs said, hid nothing (the bug hunt of 2026-10-25). The order is kept, since it is part of
# the draw that verify checks on every certificate made since 1.17.0.
CLASS_BLIND_NOTE = ("the rows run class by class, and the `map_class`, `condition` and `stratum` columns all name the "
                    "map's class: to label blind, hide those three and show the windows in a shuffled order, then "
                    "write each label on its own row, keeping the CSV's row order")

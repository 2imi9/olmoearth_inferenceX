"""The estimator's closed forms, checked by enumeration rather than simulation.

Every claim oe_inferencex.estimate rests on is finite-population sampling theory, and each has a closed form that
can be checked EXACTLY on a small population by enumerating every possible sample: the stratified estimator is
unbiased and its variance estimator is unbiased for the true variance; Neyman's allocation minimises that variance;
Wilson's interval covers at a rate a hypergeometric sum gives exactly; the cluster-sample variance ratio is
m S_b^2 / S^2 exactly and 1 + (m - 1) rho in the large-T limit; and the rate a review set gives is capture(b) e / b.
Derivations are in docs/method/protocol.md. Where the claim is an equality, the enumeration uses exact rational
arithmetic and the comparison to the package's floats is at 1e-12."""
import itertools
import math
import os
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- a small stratified population
SIZES = (5, 4, 3)
ERR = np.array([1, 1, 0, 0, 1,  1, 0, 0, 0,  0, 0, 1], float)        # 5 errors in 12 units: rates 3/5, 1/4, 1/3
STRATA = np.repeat(np.arange(3), SIZES)
N = int(ERR.size)


def _every_stratified_sample(alloc):
    pools = [np.flatnonzero(STRATA == h) for h in range(3)]
    for combo in itertools.product(*[itertools.combinations(pool, n) for pool, n in zip(pools, alloc)]):
        yield np.concatenate([np.array(c) for c in combo])


def _true_variance(alloc):
    """Cochran's exact variance of the stratified mean, with S_h^2 on N_h - 1, as a Fraction."""
    v = Fraction(0)
    for h, (Nh, nh) in enumerate(zip(SIZES, alloc)):
        e = [Fraction(int(x)) for x in ERR[STRATA == h]]
        mean = sum(e) / Nh
        S2 = sum((x - mean) ** 2 for x in e) / (Nh - 1)
        v += Fraction(Nh, N) ** 2 * (1 - Fraction(nh, Nh)) * S2 / nh
    return v


@pytest.mark.parametrize("alloc", [(2, 2, 2), (3, 2, 2), (2, 3, 2), (4, 2, 2)])
def test_the_stratified_estimator_is_unbiased_over_every_sample(alloc):
    """The mean of the estimate over all equally likely stratified samples equals the population rate exactly."""
    ests = [est.stratified_interval(ERR, STRATA, picked, SIZES, N)[0] for picked in _every_stratified_sample(alloc)]
    n_samples = math.prod(math.comb(Nh, nh) for Nh, nh in zip(SIZES, alloc))
    assert len(ests) == n_samples
    assert abs(sum(ests) / n_samples - ERR.mean()) < 1e-12


@pytest.mark.parametrize("alloc", [(2, 2, 2), (3, 2, 2), (4, 2, 2)])
def test_the_variance_estimator_is_unbiased_for_the_exact_variance(alloc):
    """Two things at once: the enumerated variance of the estimate equals Cochran's closed form exactly, and the
    variance the interval is built from, p_h(1 - p_h)/(n_h - 1) weighted and corrected, averages to it exactly."""
    samples = list(_every_stratified_sample(alloc))
    ests = np.array([est.stratified_interval(ERR, STRATA, picked, SIZES, N)[0] for picked in samples])
    enumerated = ests.var()                                                  # over equally likely samples
    closed = float(_true_variance(alloc))
    assert abs(enumerated - closed) < 1e-12
    # the estimator's own variance estimate, averaged over every sample: E[s_h^2] = S_h^2 under SRS within strata
    est_var = np.array([est.stratified_mean_and_variance(ERR, STRATA, picked, SIZES, N)[1] for picked in samples])
    assert abs(est_var.mean() - closed) < 1e-12


def _variance_at(alloc, sizes, S2):
    """Cochran's variance at a possibly non-integer allocation, as a float."""
    Ntot = sum(sizes)
    return sum((Nh / Ntot) ** 2 * (1 - nh / Nh) * s2 / nh for Nh, nh, s2 in zip(sizes, alloc, S2))


@pytest.mark.parametrize("sizes,rates,B,tol", [
    ((5, 4, 3), (3 / 5, 1 / 4, 1 / 3), 8, 0.03),                # the small population above: 2.3% above the integer optimum
    ((30, 25, 20), (0.5, 0.2, 0.05), 60, 0.02),                 # a larger one where one stratum is a census: 1.3%
])
def test_neymans_continuous_allocation_is_the_exact_minimum_and_the_integer_one_is_within_rounding(sizes, rates, B, tol):
    """The continuous Neyman allocation n_h = B N_h S_h / sum N_h S_h beats every integer allocation of the same
    budget, exactly, since each integer allocation is a feasible point of the same problem. The package's integer
    version floors it and hands the remainder by remaining units, which is what exp78 ran; its distance from the
    best integer allocation is asserted at two budgets so the rounding cost is on the record."""
    sizes = np.array(sizes)
    S2 = np.array([Nh / (Nh - 1) * p * (1 - p) for Nh, p in zip(sizes, rates)])     # S_h^2 on N_h - 1 for a 0/1 population
    S = np.sqrt(S2)
    cont = B * sizes * S / (sizes * S).sum()
    v_cont = _variance_at(cont, sizes, S2)
    allocs = [a for a in itertools.product(*[range(2, min(Nh, B) + 1) for Nh in sizes]) if sum(a) == B]
    assert allocs
    v_int = {a: _variance_at(a, sizes, S2) for a in allocs}
    assert v_cont <= min(v_int.values()) + 1e-15                            # exact: the relaxed optimum
    ney = tuple(int(x) for x in est.neyman_allocation(sizes, S, B))
    assert sum(ney) == B and min(ney) >= 2
    assert v_int[ney] <= min(v_int.values()) * (1 + tol), (B, ney, v_int[ney], min(v_int, key=v_int.get))
    prop = tuple(int(x) for x in est.neyman_allocation(sizes, np.ones(3), B))
    assert v_int[ney] <= v_int[prop] + 1e-15


@pytest.mark.parametrize("task,bound", [("mados", 0.006), ("pastis_sentinel2", 0.003), ("m_cashew_plant", 0.002)])
def test_the_integer_allocation_is_within_half_a_percent_of_the_continuous_optimum_at_the_records_budget(task, bound):
    """On exp78's real strata at B = 300, from the same confidence-derived spread the design uses; the continuous
    allocation is a lower bound on any integer one, so this is the whole rounding cost."""
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", f"{task}.npz"))
    margin, p1, err = d["margin"].astype(float), d["p1"].astype(float), d["err"].astype(float)
    s = est.confidence_strata(margin)
    sizes = np.bincount(s, minlength=5)
    q = np.array([(1 - p1[s == h]).mean() for h in range(5)])
    spread = np.sqrt(np.clip(q * (1 - q), 1e-9, None))
    S2 = np.array([err[s == h].var(ddof=1) for h in range(5)])
    cont = 300 * sizes * spread / (sizes * spread).sum()
    ney = est.neyman_allocation(sizes, spread, 300)
    ratio = _variance_at(ney, sizes, S2) / _variance_at(cont, sizes, S2)
    assert 1.0 <= ratio < 1 + bound, (task, ratio)


# ----------------------------------------------------------------------------- Wilson
@pytest.mark.parametrize("N_,K,B", [(20, 6, 8), (30, 3, 10), (25, 12, 9)])
def test_exact_coverage_equals_a_full_enumeration_of_samples(N_, K, B):
    theta, hits = K / N_, 0
    for combo in itertools.combinations(range(N_), B):
        k = sum(1 for i in combo if i < K)
        lo, hi = est.wilson_interval(k, B, N_)
        hits += lo <= theta <= hi
    assert est.exact_coverage_srs(N_, K, B) == pytest.approx(hits / math.comb(N_, B), abs=1e-12)


def test_exact_coverage_at_madoss_recorded_cell_explains_its_monte_carlo_number():
    """exp78 recorded MADOS at 0.933, on the 0.93 bar, and named Wilson's discreteness in advance. The exact
    coverage at MADOS's (N, K, B) says whether that was the interval or the draw."""
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", "mados.npz"))
    err = d["err"].astype(float)
    exact = est.exact_coverage_srs(int(err.size), int(err.sum()), 300)
    mc = 0.933                                                              # exp/out/exp78_summary.json, D1/E1
    se = math.sqrt(exact * (1 - exact) / 2000)
    assert abs(mc - exact) < 3 * se, (exact, mc, se)


# ----------------------------------------------------------------------------- clusters
def test_cluster_sampling_variance_ratio_is_m_Sb2_over_S2_exactly_and_the_design_effect_in_the_limit():
    """T tiles of m units. Enumerating every draw of t tiles, the variance of the cluster-sample mean over the
    SRS variance at the same n equals m S_b^2 / S^2 exactly (with N - 1 and T - 1 conventions); 1 + (m - 1) rho
    from the ANOVA intra-cluster correlation equals it up to O(1/T), which the test checks at T = 8 and T = 40."""
    rng = np.random.default_rng(3)
    for T, tol in ((8, 0.15), (40, 0.03)):
        m, t = 4, 3
        tile_rate = rng.random(T) * 0.6
        err = (rng.random((T, m)) < tile_rate[:, None]).astype(float)       # errors cluster by tile
        tile = np.repeat(np.arange(T), m)
        flat, N_ = err.ravel(), T * m
        n = t * m
        S2 = flat.var(ddof=1)
        Sb2 = err.mean(1).var(ddof=1)
        v_srs = (1 - n / N_) * S2 / n
        means = [err[list(c)].mean() for c in itertools.combinations(range(T), t)] if T <= 12 else None
        if means is not None:
            v_cl = np.var(means)                                            # exact, over every draw of t tiles
            assert v_cl == pytest.approx((1 - t / T) * Sb2 / t, rel=1e-10)  # Cochran, single-stage cluster
            assert v_cl / v_srs == pytest.approx(m * Sb2 / S2, rel=1e-10)
        deff = est.design_effect(flat, tile, m)
        assert deff == pytest.approx(m * Sb2 / S2, rel=tol), (T, deff, m * Sb2 / S2)


# ----------------------------------------------------------------------------- the review set
@pytest.mark.parametrize("task", ["mados", "sen1floods11", "pastis_sentinel2", "m_cashew_plant"])
def test_the_rate_a_review_set_gives_is_capture_times_e_over_b_against_the_recorded_capture(task):
    """rate on the top-k = (captured errors)/k = capture(b) E / k. With the record's own capture at 5% and error
    rate (exp70), this reproduces the inflation the guard warns about, and ties it to a number already on the
    record rather than to a fresh measurement."""
    import json
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))["results"]["tasks"][task]
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", f"{task}.npz"))
    err, margin = d["err"].astype(float), d["margin"].astype(float)
    n = err.size
    k = int(round(0.05 * n))
    top = np.argsort(margin, kind="stable")[:k]
    measured = err[top].mean()
    predicted = float(rec["signals"]["margin"]["capture"]["0.05"]) * rec["error_rate"] * n / k
    assert measured == pytest.approx(predicted, abs=2e-3), (task, measured, predicted)
    assert measured / err.mean() > 1.7


# ----------------------------------------------------------------------------- the input-condition design
# Strata are input conditions (a cloud flag, the modalities present), labels are split equally by water-filling, and
# each condition gets its own exact interval (docs: condition_spec 2.5 and 2.7). Every claim below is checked by
# enumerating every sample of a small population, or against a known answer.
def test_equal_allocation_known_answers():
    """Water-filling: a condition too small for an equal share is labelled in full and the rest share equally; the
    remainder goes one label each to the conditions with the most windows left, lowest index on ties."""
    for sizes, B, want in [((100, 7, 50), 60, (27, 7, 26)), ((5, 4, 3), 7, (3, 2, 2)), ((1, 50, 50), 10, (1, 5, 4)),
                           ((5, 4, 3), 12, (5, 4, 3)), ((10, 1000, 100000), 300, (10, 145, 145)),
                           ((228665, 229973), 300, (150, 150)), ((436323, 22315), 300, (150, 150)),
                           ((6, 3, 3), 8, (3, 3, 2))]:          # never two labels of the remainder to one condition
        assert tuple(int(x) for x in est.equal_allocation(sizes, B)) == want, (sizes, B)
    # The answers above put the largest condition first, where a rule that gives the remainder to the lowest index
    # agrees. Here the largest is last or in the middle, so only "most windows left" gives these.
    for sizes, B, want in [((3, 4, 5), 7, (2, 2, 3)), ((50, 7, 100), 60, (26, 7, 27)), ((3, 3, 6), 8, (3, 2, 3)),
                           ((3, 5, 4), 7, (2, 3, 2)), ((4, 6, 5), 8, (2, 3, 3)), ((7, 100, 50), 60, (7, 27, 26))]:
        assert tuple(int(x) for x in est.equal_allocation(sizes, B)) == want, (sizes, B)
    with pytest.raises(ValueError, match=r"\(needs 6\)"):
        est.equal_allocation((100, 100, 100), 5)
    with pytest.raises(ValueError, match="more than the 12 units"):
        est.equal_allocation((5, 4, 3), 13)
    # fact 2 of the spec: Neyman's rule with 1 / N_h as the spread is not an equal split
    assert tuple(int(x) for x in est.neyman_allocation(np.array([100, 7, 50]), 1 / np.array([100, 7, 50]), 60)) == (33, 7, 20)


def _leximin_best(sizes, B):
    """Every integer allocation with sum B and n_h <= N_h, and the best one in max-min order: the largest smallest
    share, then the largest second smallest, and so on (an allocation sorted ascending, compared as a tuple)."""
    feasible = [a for a in itertools.product(*[range(Nh + 1) for Nh in sizes]) if sum(a) == B]
    return feasible, max(tuple(sorted(a)) for a in feasible)


def test_equal_allocation_invariants_over_every_small_case():
    """Every sizes tuple in {1..6}^3 and every budget from the floor to N: the sum is the budget, no condition gets
    more than it has or fewer than min(N_c, 2), the conditions not labelled in full differ by at most one, and the
    allocation is the max-min fair one found by brute force over every feasible allocation. The function takes no
    confidence argument, so the model's confidence cannot enter it."""
    import inspect
    assert list(inspect.signature(est.equal_allocation).parameters) == ["sizes", "budget", "floor"]
    n_cases = 0
    for sizes in itertools.product(range(1, 7), repeat=3):
        floor, total = sum(min(Nh, 2) for Nh in sizes), sum(sizes)
        with pytest.raises(ValueError, match="cannot give each"):
            est.equal_allocation(sizes, floor - 1)
        for B in range(floor, total + 1):
            n = [int(x) for x in est.equal_allocation(sizes, B)]
            assert sum(n) == B, (sizes, B, n)
            assert all(min(Nh, 2) <= nh <= Nh for nh, Nh in zip(n, sizes)), (sizes, B, n)
            partial = [nh for nh, Nh in zip(n, sizes) if nh < Nh]
            assert not partial or max(partial) - min(partial) <= 1, (sizes, B, n)
            _, best = _leximin_best(sizes, B)
            assert tuple(sorted(n)) == best, (sizes, B, n, best)
            n_cases += 1
    assert n_cases == sum(sum(s) - sum(min(x, 2) for x in s) + 1 for s in itertools.product(range(1, 7), repeat=3))


COND = np.array([0] * 5 + [1] * 4 + [-1] * 3)          # the SIZES population above, as input conditions: the third is
                                                        # the unrecorded one, so the code for "no condition" runs


def _condition_sample(budget, condition=COND, valid=None, seed=0, margin=None):
    margin = np.linspace(1.0, 0.0, condition.size) if margin is None else margin
    return est.sample_for_estimation(margin, budget, design="condition", condition=condition, valid=valid, seed=seed)


@pytest.mark.parametrize("alloc,budget", [((2, 2, 2), 6), ((3, 2, 2), 7)])
def test_the_condition_design_is_unbiased_over_every_sample(alloc, budget):
    """Over every equally likely sample of the condition design, the whole-map estimate and every condition's own
    estimate average to the truth exactly, and `design_variance`, kept for information, averages to Cochran's closed
    form, computed here in exact fractions."""
    base = _condition_sample(budget)
    assert tuple(base["allocation"]) == alloc and base["condition"]["names"] == ["0", "1", "unrecorded"]
    assert base["condition"]["values"] == [0, 1, None] and base["sizes"] == list(SIZES)
    ests, per, var = [], {c: [] for c in base["condition"]["names"]}, []
    for picked in _every_stratified_sample(alloc):
        s = dict(base, indices=picked)
        r = est.estimate_error_rate(s, ERR[picked])
        assert r["by_condition"] and r["method"].startswith("stratified by input condition")
        ests.append(r["estimate"])
        var.append(r["design_variance"])
        for c, row in r["per_condition"].items():
            per[c].append(row["estimate"])
    n_samples = math.prod(math.comb(Nh, nh) for Nh, nh in zip(SIZES, alloc))
    assert len(ests) == n_samples
    assert abs(np.mean(ests) - ERR.mean()) < 1e-12
    for c, h in (("0", 0), ("1", 1), ("unrecorded", 2)):
        assert abs(np.mean(per[c]) - ERR[STRATA == h].mean()) < 1e-12, c
    closed = _true_variance(alloc)
    assert abs(np.var(ests) - float(closed)) < 1e-12                 # the estimate's own variance, enumerated
    assert abs(np.mean(var) - float(closed)) < 1e-12                 # the estimated variance is unbiased for it


def _two_conditions(N0, N1, K0, K1):
    """Two conditions of N0 and N1 windows holding K0 and K1 errors; the first K of each condition are wrong."""
    cond = np.r_[np.zeros(N0, int), np.ones(N1, int)]
    err = np.r_[np.arange(N0) < K0, np.arange(N1) < K1].astype(float)
    return cond, err


def test_per_condition_interval_is_exact_under_the_condition_design():
    """Conditions of 7 and 5 windows, labels (3, 2): for every error count in each and every sample, each
    condition's coverage equals the exact coverage of the hypergeometric interval, and is at least 95%."""
    for K0 in range(8):
        for K1 in range(6):
            cond, err = _two_conditions(7, 5, K0, K1)
            base = _condition_sample(5, condition=cond)
            assert base["allocation"] == [3, 2]
            hits, n = [0, 0], 0
            pools = [np.flatnonzero(cond == c) for c in (0, 1)]
            for a in itertools.combinations(pools[0], 3):
                for b in itertools.combinations(pools[1], 2):
                    picked = np.array(a + b)
                    r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
                    for c, (Nc, Kc) in enumerate(((7, K0), (5, K1))):
                        row = r["per_condition"][str(c)]
                        hits[c] += row["low"] <= Kc / Nc <= row["high"]
                    n += 1
            assert n == math.comb(7, 3) * math.comb(5, 2)
            for c, (Nc, Kc, nc) in enumerate(((7, K0, 3), (5, K1, 2))):
                exact = est.exact_coverage_srs(Nc, Kc, nc, est.hypergeom_interval)
                assert abs(hits[c] / n - exact) < 1e-12, (K0, K1, c)
                assert hits[c] / n >= 0.95, (K0, K1, c, hits[c] / n)


def test_per_condition_interval_covers_given_its_count_under_a_random_sample():
    """Under a random sample the count of labels in a condition is random, and given that count the labels are a
    simple random sample of it. All 210 draws of 4 from conditions of 6 and 4, every error count: the coverage
    given each count m equals the exact coverage at m and is at least 95%; m = 0 says nothing, [0, 1]. Then, all
    924 draws of 6 from conditions of 7 and 5: the count of wrong labels in a condition, given its labels, is
    hypergeometric, checked in exact fractions."""
    from fractions import Fraction
    for K0 in range(7):
        for K1 in range(5):
            cond, err = _two_conditions(6, 4, K0, K1)
            base = est.sample_for_estimation(np.linspace(1, 0, 10), 4, design="random", condition=cond)
            by_m = {}                                                 # (condition, m) -> [hits, draws]
            for draw in itertools.combinations(range(10), 4):
                picked = np.array(draw)
                r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
                assert r["method"].startswith("exact hypergeometric")                # the whole map: unchanged
                for c, (Nc, Kc) in enumerate(((6, K0), (4, K1))):
                    row = r["per_condition"][str(c)]
                    m = row["n_labelled"]
                    if m == 0:
                        assert row["estimate"] is None and (row["low"], row["high"]) == (0.0, 1.0) and "note" in row
                    t = by_m.setdefault((c, m), [0, 0])
                    t[0] += row["low"] <= Kc / Nc <= row["high"]
                    t[1] += 1
            assert sum(v[1] for (c, _), v in by_m.items() if c == 0) == math.comb(10, 4)
            for (c, m), (hits, draws) in by_m.items():
                Nc, Kc = ((6, K0), (4, K1))[c]
                assert draws == math.comb(Nc, m) * math.comb(10 - Nc, 4 - m)
                want = 1.0 if m == 0 else est.exact_coverage_srs(Nc, Kc, m, est.hypergeom_interval)
                assert abs(hits / draws - want) < 1e-12 and hits / draws >= 0.95, (K0, K1, c, m)
    cond, err = _two_conditions(7, 5, 3, 2)
    base = est.sample_for_estimation(np.linspace(1, 0, 12), 6, design="random", condition=cond)
    tally = {}                                                        # (condition, n_c, k_c) -> draws
    for draw in itertools.combinations(range(12), 6):
        picked = np.array(draw)
        r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
        for c in (0, 1):
            row = r["per_condition"][str(c)]
            key = (c, row["n_labelled"], row["n_wrong"])
            tally[key] = tally.get(key, 0) + 1
    for c, (Nc, Kc) in enumerate(((7, 3), (5, 2))):
        for m in range(0, min(Nc, 6) + 1):
            given = sum(v for (cc, mm, _), v in tally.items() if cc == c and mm == m)
            if not given:
                continue
            for j in range(0, m + 1):
                got = Fraction(tally.get((c, m, j), 0), given)
                assert got == Fraction(math.comb(Kc, j) * math.comb(Nc - Kc, m - j), math.comb(Nc, m)), (c, m, j)


def test_a_single_condition_is_the_random_design():
    """With one condition the condition design is the random design: the same windows, the same estimate and
    interval, and certify_by_condition returns certify_zone's zone at the full delta."""
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n = int(rng.integers(300, 900))
        margin = rng.random(n)
        margin[rng.random(n) < 0.05] = np.nan                      # windows with no finite margin are outside
        valid = rng.random(n) > 0.1
        err = (rng.random(n) < 0.03 + 0.5 * (1 - np.nan_to_num(margin)) ** 3).astype(float)
        cond = np.full(n, int(rng.integers(0, 5)))
        cond[~valid] = int(rng.integers(-3, 9))                     # outside the population: not a condition
        B = int(rng.integers(60, 200))
        r = est.sample_for_estimation(margin, B, design="random", valid=valid, seed=seed)
        rc = est.sample_for_estimation(margin, B, design="random", valid=valid, seed=seed, condition=cond)
        c = est.sample_for_estimation(margin, B, design="condition", valid=valid, seed=seed, condition=cond)
        assert np.array_equal(r["indices"], c["indices"]) and np.array_equal(r["indices"], rc["indices"])
        assert c["n_strata"] == 1 and "simple random sample" in c["note"]
        e_r = est.estimate_error_rate(r, err[r["indices"]])
        for s in (rc, c):
            e = est.estimate_error_rate(s, err[s["indices"]])
            assert (e["estimate"], e["low"], e["high"], e["method"]) == (e_r["estimate"], e_r["low"], e_r["high"], e_r["method"])
            (row,) = e["per_condition"].values()
            assert (row["estimate"], row["low"], row["high"]) == (e["estimate"], e["low"], e["high"])
        alpha = float(rng.choice([0.1, 0.2, 0.3]))
        whole = est.certify_zone(margin, r["indices"], err[r["indices"]], alpha, valid=valid)
        for s in (rc, c):
            fam = est.certify_by_condition(s, err[s["indices"]], margin, alpha, valid=valid)
            assert fam["n_conditions_tested"] == 1 and fam["delta_per_condition"] == est.ZONE_DELTA
            (part,) = fam["per_condition"].values()
            assert part["tested"] and set(part) - set(whole) == {"value", "tested", "reason"}
            assert set(whole) - set(part) == {"scope"}
            for k, v in whole.items():
                if k == "scope":
                    continue
                assert (np.array_equal(v, part[k]) if isinstance(v, np.ndarray) else v == part[k]), (seed, k)
            assert np.array_equal(fam["zone_indices_in_order"], whole.get("zone_indices_in_order", np.zeros(0, int)))


def test_condition_sample_malformed_is_refused():
    """The guards of the new path: a sample edited by hand, or mixed up with another, is refused rather than
    estimated, where a stray stratum id used to be dropped in silence (fact 7 of the spec)."""
    cond = np.r_[np.zeros(40, int), np.ones(30, int), np.full(10, -1), np.full(6, 2)]
    valid = np.ones(cond.size, bool)
    valid[-6:] = False                                               # outside the population: condition grid -1
    good = _condition_sample(24, condition=cond, valid=valid, seed=3)
    wrong = np.zeros(24)
    est.estimate_error_rate(good, wrong)                             # the untouched sample passes

    def bad(**edit):
        s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in good.items()}
        for k, v in edit.items():
            s[k] = v(s) if callable(v) else v
        return s

    def stray(s):
        t = s["strata"].copy(); t[int(np.flatnonzero(t == 1)[0])] = 3; return t
    def sizes_off(s):
        return [s["sizes"][0] + 1] + s["sizes"][1:]
    def alloc_off(s):
        return [s["allocation"][0] + 1, s["allocation"][1] - 1] + s["allocation"][2:]
    def hole(s):                                                     # a sampled window swapped with one outside
        g = s["condition_grid"].copy(); g[-1] = g[s["indices"][0]]; g[s["indices"][0]] = -1; return g
    def beyond(s):
        i = s["indices"].copy(); i[0] = s["condition_grid"].size + 5; return i
    cases = [({"strata": stray}, "not the conditions of its population windows|stratum id"),
             ({"sizes": sizes_off}, "do not add up"),
             ({"allocation": alloc_off}, "not the design's allocation"),
             ({"condition_grid": hole}, "outside the population the condition grid records"),
             ({"indices": beyond}, "beyond the"),
             ({"n_population": good["n_population"] + 1}, "condition grid holds"),
             ({"n_strata": 4}, "one stratum per condition"),
             ({"condition": lambda s: dict(s["condition"], sizes=[41, 29, 10])}, "not the counts"),
             ({"condition": lambda s: dict(s["condition"], names=["0", "1"])}, "one per condition"),
             ({"condition_grid": lambda s: s["condition_grid"].reshape(2, -1)}, "one value per window")]
    for edit, why in cases:
        with pytest.raises(ValueError, match=why):
            est.estimate_error_rate(bad(**edit), wrong)
    rnd = est.sample_for_estimation(np.linspace(1, 0, cond.size), 24, design="random", condition=cond, valid=valid, seed=3)
    for key, why in (("indices", "beyond the"), ("condition_grid", "outside the population the condition grid records")):
        s = dict(rnd)
        s[key] = (beyond if key == "indices" else hole)(rnd)
        with pytest.raises(ValueError, match=why):
            est.estimate_error_rate(s, wrong)
    with pytest.raises(ValueError, match="carries its condition_grid"):
        est.estimate_error_rate({k: v for k, v in good.items() if k != "condition_grid"}, wrong)


# ----------------------------------------------------------------------------- the condition design's whole-map interval
def _union_by_hand(k, n, sizes):
    """The whole-map interval written out: each condition's exact interval at 1 - 0.05 / L, L the conditions not
    labelled in full, weighted by N_c / N; a condition labelled in full enters at k_c / n_c."""
    N, L = sum(sizes), sum(nc < Nc for nc, Nc in zip(n, sizes))
    lo = hi = 0.0
    for kc, nc, Nc in zip(k, n, sizes):
        if nc == Nc:
            lc = hc = kc / nc
        else:
            lc, hc = est.hypergeom_interval(kc, nc, Nc, conf=1 - 0.05 / L)
        lo, hi = lo + Nc / N * lc, hi + Nc / N * hc
    return lo, hi, L


def _every_whole_map_interval(sizes, budget):
    """The condition design on conditions of `sizes` windows at `budget` labels: its allocation, and its whole-map
    interval for every vector of error counts, each run through estimate_error_rate. The interval depends on the
    labels only through each condition's count, so these are all the intervals the design can give."""
    cond = np.repeat(np.arange(len(sizes)), sizes)
    base = _condition_sample(budget, condition=cond)
    n, idx = base["allocation"], base["indices"]
    at = [np.flatnonzero(cond[idx] == c) for c in range(len(sizes))]
    out = {}
    for ks in itertools.product(*[range(nc + 1) for nc in n]):
        wrong = np.zeros(idx.size)
        for pos, kc in zip(at, ks):
            wrong[pos[:kc]] = 1
        r = est.estimate_error_rate(base, wrong)
        assert r["low"] <= r["estimate"] <= r["high"], (sizes, budget, ks)
        out[ks] = (r["low"], r["high"])
    return n, out


@pytest.mark.parametrize("sizes,budget,alloc", [
    ((12, 8), 6, (3, 3)), ((12, 8), 10, (5, 5)), ((12, 8), 16, (8, 8)), ((30, 6), 10, (5, 5)), ((30, 6), 14, (8, 6)),
    ((40, 4), 12, (8, 4)), ((95, 5), 10, (5, 5)), ((60, 20), 20, (10, 10)), ((99, 1), 10, (9, 1)),
    ((8, 6, 4), 6, (2, 2, 2)), ((8, 6, 4), 9, (3, 3, 3)), ((8, 6, 4), 12, (4, 4, 4)), ((10, 5, 3), 9, (3, 3, 3)),
    ((20, 3, 2), 9, (4, 3, 2))])
def test_the_whole_map_interval_covers_at_least_95_percent_on_every_small_population(sizes, budget, alloc):
    """Two and three conditions, splits from 60/40 to 99/1, conditions labelled in part and in full. For every error
    count in every condition, the probability over all samples of the design that the whole-map interval holds the
    whole-map rate, summed in integers: at least 95% everywhere. Which windows are wrong does not matter to it, so
    this is every population of these sizes. The truth is compared in floats as a user would, with no tolerance:
    summed in floats, the ends left a truth that sits on one of them outside on every one of these maps, and some
    populations were then covered by no sample at all."""
    n, iv = _every_whole_map_interval(sizes, budget)
    assert tuple(n) == alloc
    N = sum(sizes)
    den = math.prod(math.comb(Nc, nc) for Nc, nc in zip(sizes, n))
    for Ks in itertools.product(*[range(Nc + 1) for Nc in sizes]):
        theta = sum(Ks) / N
        hit = sum(math.prod(math.comb(Kc, kc) * math.comb(Nc - Kc, nc - kc) for Kc, kc, Nc, nc in zip(Ks, ks, sizes, n))
                  for ks, (lo, hi) in iv.items() if lo <= theta <= hi)
        assert 100 * hit >= 95 * den, (Ks, hit / den)


def test_condition_whole_map_by_a_second_route():
    """The whole-map result under the condition design, written out: theta = sum_c W_c k_c / n_c, the interval from
    `_union_by_hand`, L in `conditions_in_interval`, and v = sum_c W_c^2 (1 - n_c / N_c) p_c (1 - p_c) / (n_c - 1)
    kept as `design_variance`. No warning: the interval stands on each condition's exact one whatever the labels,
    including when every condition's labels agree or no label is wrong. The floor's keys are gone."""
    rng = np.random.default_rng(8)
    cond = np.r_[np.zeros(400, int), np.ones(250, int), np.full(90, 5), np.full(30, -1)]
    err = (rng.random(cond.size) < np.select([cond == 0, cond == 1, cond == 5], [0.1, 0.45, 0.7], 0.3)).astype(float)
    names = ["0", "1", "5", "unrecorded"]
    for seed in range(8):
        s = _condition_sample(120, condition=cond, seed=seed)
        idx = s["indices"]
        for wrong_of in (err, (cond == 1).astype(float), np.zeros(cond.size), np.ones(cond.size)):
            r = est.estimate_error_rate(s, wrong_of[idx])
            N = cond.size
            theta = v = 0.0
            k, n, sizes = [], [], []
            for val in (0, 1, 5, -1):
                m = cond[idx] == val
                Nc, nc, kc = int((cond == val).sum()), int(m.sum()), int(wrong_of[idx][m].sum())
                p = kc / nc
                theta += Nc / N * p
                v += (Nc / N) ** 2 * (1 - nc / Nc) * p * (1 - p) / (nc - 1)
                k, n, sizes = k + [kc], n + [nc], sizes + [Nc]
            lo, hi, L = _union_by_hand(k, n, sizes)
            assert n == [30, 30, 30, 30] and L == 3 and r["conditions_in_interval"] == 3   # unrecorded is in full
            assert abs(r["estimate"] - theta) < 1e-12 and abs(r["design_variance"] - v) < 1e-15
            assert abs(r["low"] - lo) < 1e-15 and abs(r["high"] - hi) < 1e-15 and r["low"] <= r["estimate"] <= r["high"]
            assert r["method"] == ("stratified by input condition; sum of each condition's exact interval at 1 - 0.05/L, "
                                   "weighted by its share of the map (union bound); covers at least 95% by construction")
            assert r["starved_strata"] == 0 and "warning" not in r
            assert not {"effective_n", "interval_variance", "floored_conditions"} & set(r)
            assert r["condition_note"] == est.CONDITION_NOTE + " " + est.CONDITION_WHOLE_MAP
            assert list(r["per_condition"]) == names
            for c, name in enumerate(names):                          # each condition's own interval stays at 95%
                row = r["per_condition"][name]
                assert (row["low"], row["high"]) == est.hypergeom_interval(k[c], n[c], sizes[c])
            assert set(r["outside_condition_intervals"]) == {c for c, row in r["per_condition"].items()
                                                            if not row["low"] <= r["estimate"] <= row["high"]}
    # a sample edited by hand to leave one label in a condition: its exact interval at n = 1 still enters the sum,
    # and the warning says only that design_variance misses it
    cond2 = np.r_[np.zeros(40, int), np.ones(30, int)]
    s2 = _condition_sample(20, condition=cond2)
    assert s2["allocation"] == [10, 10]
    at = cond2[s2["indices"]]
    s2 = dict(s2, indices=s2["indices"][np.r_[np.flatnonzero(at == 0), np.flatnonzero(at == 1)[:1]]], allocation=[10, 1])
    w2 = np.r_[np.ones(3), np.zeros(7), 1.0]
    r = est.estimate_error_rate(s2, w2)
    lo, hi, L = _union_by_hand([3, 1], [10, 1], [40, 30])
    assert r["conditions_in_interval"] == L == 2 and abs(r["low"] - lo) < 1e-15 and abs(r["high"] - hi) < 1e-15
    assert r["starved_strata"] == 1 and r["warning"].startswith("1 of 2 conditions had fewer than 2 labelled windows")


def test_a_condition_labelled_in_full_contributes_a_point():
    """A condition labelled in full has no sampling error: it enters the whole-map interval at its exact rate and is
    not counted in L, so the others are taken at 1 - 0.05 / L over the conditions labelled in part only. Two
    conditions with the small one in full are one exact interval at 95% shifted by a point; three with one in full
    are two at 97.5%, and so with a one-window condition, which is not counted as starved either; every condition in
    full is the whole map's exact rate, a point."""
    for sizes, budget, alloc, L in (((40, 4), 12, [8, 4], 1), ((8, 6, 4), 12, [4, 4, 4], 2), ((60, 50, 1), 20, [10, 9, 1], 2),
                                    ((6, 3), 9, [6, 3], 0)):
        n, iv = _every_whole_map_interval(sizes, budget)
        assert n == alloc
        N = sum(sizes)
        full = [c for c in range(len(sizes)) if n[c] == sizes[c]]
        for ks, (lo, hi) in iv.items():
            point = sum(sizes[c] / N * ks[c] / n[c] for c in full)
            span = [est.hypergeom_interval(ks[c], n[c], sizes[c], conf=1 - 0.05 / L) if n[c] < sizes[c] else None
                    for c in range(len(sizes))]
            want_lo = point + sum(sizes[c] / N * span[c][0] for c in range(len(sizes)) if span[c])
            want_hi = point + sum(sizes[c] / N * span[c][1] for c in range(len(sizes)) if span[c])
            assert abs(lo - want_lo) < 1e-15 and abs(hi - want_hi) < 1e-15, (sizes, ks)
            if L == 0:
                assert lo == hi                                       # a census is its rate
        cond = np.repeat(np.arange(len(sizes)), sizes)
        r = est.estimate_error_rate(_condition_sample(budget, condition=cond), (np.arange(budget) % 3 == 0).astype(float))
        assert r["conditions_in_interval"] == L and r["starved_strata"] == 0 and "warning" not in r


def test_one_condition_is_the_random_designs_exact_interval():
    """With one condition the sum has one term at 1 - 0.05 / 1: the random design's `hypergeom_interval(k, n, N)`,
    to the last bit, census included (L = 0 there, and the interval is the point k / n), and so is no label (0 to 1).
    The end-to-end check through `estimate_error_rate` is `test_a_single_condition_is_the_random_design`."""
    for N_ in (1, 2, 7, 30, 101, 5000):
        for n in sorted({0, 1, 2, N_ // 3, N_ - 1, N_} & set(range(0, N_ + 1))):
            for k in sorted({0, 1, n // 2, n - 1, n} & set(range(n + 1))):
                e, lo, hi, L = est._union_interval([k], [n], [N_])
                assert (lo, hi) == est.hypergeom_interval(k, n, N_) and L == int(n < N_), (N_, n, k)
                assert e == (k / n if n else 0.0)                     # the random design's float(wrong.mean())


def test_a_small_degraded_condition_beside_a_large_clean_one_is_not_absurdly_wide():
    """The case the enumeration of 2026-09-29 found: a million windows split 99/1, 300 labels in each condition, no
    error in the large one's labels and one in the small one's. The stratified interval with the variance floor
    printed [0, 70.4%] beside condition intervals of [0, 1.22%] and [0.01%, 1.82%]. The whole-map interval is now the
    weighted sum of those conditions' exact intervals at 1 - 0.05 / 2, summed exactly: about [0.0001%, 1.46%], no
    wider than that sum and nearly fifty times narrower than before. Each condition's own interval stays at 95%."""
    N0, N1 = 990_000, 10_000
    cond = np.r_[np.zeros(N0, int), np.ones(N1, int)]
    s = _condition_sample(600, condition=cond)
    assert s["allocation"] == [300, 300]
    wrong = np.zeros(600)
    wrong[np.flatnonzero(cond[s["indices"]] == 1)[0]] = 1
    r = est.estimate_error_rate(s, wrong)
    lo0, hi0 = est.hypergeom_interval(0, 300, N0, conf=0.975)
    lo1, hi1 = est.hypergeom_interval(1, 300, N1, conf=0.975)
    assert r["conditions_in_interval"] == 2
    want_lo, want_hi = 0.99 * lo0 + 0.01 * lo1, 0.99 * hi0 + 0.01 * hi1
    assert abs(r["low"] - want_lo) < 1e-15 and r["high"] <= want_hi + 1e-15 and abs(r["high"] - want_hi) < 1e-15
    assert r["low"] <= r["estimate"] <= r["high"] < 0.0146 and r["estimate"] == pytest.approx(0.01 / 300, rel=1e-12)
    assert 0.704 / r["high"] > 48
    per = r["per_condition"]
    assert (per["0"]["low"], per["0"]["high"]) == est.hypergeom_interval(0, 300, N0)
    assert (per["1"]["low"], per["1"]["high"]) == est.hypergeom_interval(1, 300, N1)
    assert per["0"]["low"] == 0 and per["0"]["high"] == pytest.approx(0.0122, abs=5e-5)
    assert per["1"]["low"] == pytest.approx(0.0001, abs=5e-6) and per["1"]["high"] == pytest.approx(0.0182, abs=5e-5)


def _whole_map_coverage(N0, N1, K0, K1, budget):
    """The exact coverage of the condition design's whole-map interval on two conditions of N0 and N1 windows holding
    K0 and K1 errors, and that of the stratified Wilson interval it replaced. Each interval depends on the labels
    only through each condition's error count, so every pair (k0, k1) is run once and weighted by its hypergeometric
    probability (pairs below 1e-14 are left out, which can only lower the sum)."""
    cond, err = _two_conditions(N0, N1, K0, K1)
    base = _condition_sample(budget, condition=cond)
    n0, n1 = base["allocation"]
    pools = [(np.flatnonzero((cond == c) & (err == 1)), np.flatnonzero((cond == c) & (err == 0))) for c in (0, 1)]
    pmf = lambda Nc, Kc, nc, k: math.comb(Kc, k) * math.comb(Nc - Kc, nc - k) / math.comb(Nc, nc)
    theta, cover, cover_stratified = (K0 + K1) / (N0 + N1), 0.0, 0.0
    for k0 in range(max(0, n0 - (N0 - K0)), min(n0, K0) + 1):
        for k1 in range(max(0, n1 - (N1 - K1)), min(n1, K1) + 1):
            pr = pmf(N0, K0, n0, k0) * pmf(N1, K1, n1, k1)
            if pr < 1e-14:
                continue
            picked = np.r_[pools[0][0][:k0], pools[0][1][:n0 - k0], pools[1][0][:k1], pools[1][1][:n1 - k1]]
            r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
            cover += pr * (r["low"] <= theta <= r["high"])
            lo, hi = est.stratified_interval_wilson(err, cond, picked, [N0, N1], N0 + N1)[1:3]
            cover_stratified += pr * (lo <= theta <= hi)
    return cover, cover_stratified


@pytest.mark.parametrize("K0,K1,stratified", [(20, 100, 0.531), (40, 148, 0.776), (3960, 100, 0.772)])
def test_the_whole_map_interval_covers_when_a_large_condition_looks_clean(K0, K1, stratified):
    """A large condition beside a small degraded one, 150 labels each, the design's own use case: 4,000 windows at
    0.5% or 1% wrong (or 99%) beside 200 at 50% or 74%. The large condition often shows no error, or one, in its
    labels, and the stratified variance then counts it as known almost exactly: the stratified Wilson interval with
    the conditions as strata covers 0.53, 0.78 and 0.77. The sum of exact intervals covers at least 0.95, by exact
    enumeration of every pair of error counts."""
    cover, before = _whole_map_coverage(4000, 200, K0, K1, 300)
    assert round(before, 3) == stratified and cover >= 0.95


def _expected_width(N_, K, n, conf):
    """The mean width of hypergeom_interval(k, n, N_, conf) over every sample of n from N_ windows holding K errors,
    each count weighted by its hypergeometric probability: exact up to float rounding."""
    ks = range(max(0, n - (N_ - K)), min(n, K) + 1)
    pmf = [math.comb(K, k) * math.comb(N_ - K, n - k) / math.comb(N_, n) for k in ks]
    assert abs(sum(pmf) - 1) < 1e-12
    return sum(p * (lambda iv: iv[1] - iv[0])(est.hypergeom_interval(k, n, N_, conf=conf)) for p, k in zip(pmf, ks))


@pytest.mark.parametrize("sizes,rates,wider", [
    ((1500, 1500), (0.20, 0.22), True), ((2700, 300), (0.20, 0.22), True),     # close rates, equal and 90/10 shares
    ((1500, 1500), (0.20, 0.74), True),                                        # far apart, and still wider
    ((1500, 1500), (0.02, 0.95), False), ((2700, 300), (0.02, 0.95), False)])  # very far apart: narrower
def test_the_condition_designs_whole_map_interval_against_a_random_samples(sizes, rates, wider):
    """What CONDITION_WHOLE_MAP says beside its guarantee: when the conditions' error rates are close, the condition
    design's whole-map interval is wider than the exact interval of a random sample of the same 300 labels, so a
    user who needs only the whole-map rate can do better with --design random. It is not wider everywhere: with
    rates of 2% and 95% it is narrower. The mean width of each design's interval over every sample is computed
    exactly. The condition design's interval is the weighted sum of each condition's exact interval at 1 - 0.05 / L,
    so its mean width is the weighted sum of their mean widths; that sum is checked against estimate_error_rate's own
    interval at three pairs of counts."""
    N_, Ks = sum(sizes), [round(r * s) for r, s in zip(rates, sizes)]
    cond, err = _two_conditions(*sizes, *Ks)
    base = _condition_sample(300, condition=cond)
    n = base["allocation"]
    assert n == [150, 150]
    conf = 1 - 0.05 / 2
    for k in ((0, 0), (Ks[0] * 150 // sizes[0], Ks[1] * 150 // sizes[1]), (150, 150)):
        wrong = np.r_[np.arange(150) < k[0], np.arange(150) < k[1]].astype(float)
        order = np.r_[np.flatnonzero(cond[base["indices"]] == 0), np.flatnonzero(cond[base["indices"]] == 1)]
        r = est.estimate_error_rate(dict(base, indices=base["indices"][order]), wrong)
        by_hand = sum(Nc / N_ * (lambda iv: iv[1] - iv[0])(est.hypergeom_interval(kc, 150, Nc, conf=conf))
                      for kc, Nc in zip(k, sizes))
        assert abs((r["high"] - r["low"]) - by_hand) < 1e-12, k
    condition = sum(Nc / N_ * _expected_width(Nc, Kc, 150, conf) for Nc, Kc in zip(sizes, Ks))
    random = _expected_width(N_, sum(Ks), 300, 0.95)
    assert (condition > random) == wider, (condition, random)
    assert "When the conditions' error rates are close, it is also wider than the exact interval of a random sample" \
        in est.CONDITION_WHOLE_MAP and "--design random can give a narrower interval" in est.CONDITION_WHOLE_MAP

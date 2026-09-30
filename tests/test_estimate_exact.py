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
    estimate average to the truth exactly, and the design variance the interval is built from averages to Cochran's
    closed form, computed here in exact fractions."""
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


def _wilson_by_hand(p, n):
    z = est.Z95
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def test_condition_whole_map_by_a_second_route():
    """The whole-map rate under the condition design, written out from the spec's formulas: theta = sum_c W_c k_c /
    n_c, v = sum_c W_c^2 (1 - n_c / N_c) p_c (1 - p_c) / (n_c - 1), Wilson at n_eff = theta (1 - theta) / v, with
    the floor of 2026-09-29: a condition not labelled in full enters the interval's variance at no less than
    p0 (1 - p0), p0 = (z^2 / 2) / (n_c + z^2). When every condition's labels are all right or all wrong v is zero, a
    note says so, and n_eff is the label count unless the floor gives a smaller one."""
    rng = np.random.default_rng(8)
    cond = np.r_[np.zeros(400, int), np.ones(250, int), np.full(90, 5), np.full(30, -1)]
    err = (rng.random(cond.size) < np.select([cond == 0, cond == 1, cond == 5], [0.1, 0.45, 0.7], 0.3)).astype(float)
    n_floored = 0
    for seed in range(8):
        s = _condition_sample(120, condition=cond, seed=seed)
        idx = s["indices"]
        r = est.estimate_error_rate(s, err[idx])
        N = cond.size
        theta = v = vf = 0.0
        low = []
        for c, val in enumerate((0, 1, 5, -1)):
            m = cond[idx] == val
            Nc, nc = int((cond == val).sum()), int(m.sum())
            p = err[idx][m].mean()
            theta += Nc / N * p
            v += (Nc / N) ** 2 * (1 - nc / Nc) * p * (1 - p) / (nc - 1)
            p0 = (est.Z95 ** 2 / 2) / (nc + est.Z95 ** 2)
            if nc < Nc and p * (1 - p) < p0 * (1 - p0):
                low.append(s["condition"]["names"][c])
                p = p0
            vf += (Nc / N) ** 2 * (1 - nc / Nc) * p * (1 - p) / (nc - 1)
        n_eff = theta * (1 - theta) / vf
        lo, hi = _wilson_by_hand(theta, n_eff)
        assert abs(r["estimate"] - theta) < 1e-12 and abs(r["design_variance"] - v) < 1e-15
        assert abs(r["effective_n"] - n_eff) < 1e-9 and abs(r["low"] - lo) < 1e-12 and abs(r["high"] - hi) < 1e-12
        assert r["starved_strata"] == 0 and r.get("floored_conditions", []) == low
        assert ("warning" in r) == bool(low) and ("interval_variance" in r) == bool(low)
        n_floored += bool(low)
        assert r["condition_note"] == est.CONDITION_NOTE + " " + est.CONDITION_NOT_GRADED
        assert set(r["outside_condition_intervals"]) == {c for c, row in r["per_condition"].items()
                                                        if not row["low"] <= r["estimate"] <= row["high"]}
    assert 0 < n_floored < 8                                        # both branches are run
    # every condition pure but the map is not: the design's variance is zero, and the interval falls back to Wilson
    # at the label count
    pure = (cond == 1).astype(float)
    s = _condition_sample(120, condition=cond, seed=1)
    r = est.estimate_error_rate(s, pure[s["indices"]])
    assert 0 < r["estimate"] < 1 and r["design_variance"] == 0 and r["effective_n"] == 120
    assert (r["low"], r["high"]) == pytest.approx(_wilson_by_hand(r["estimate"], 120), abs=1e-12)
    assert "every condition's labels were all right or all wrong, so the design's variance is zero" in r["warning"]
    assert "at 120 labels" in r["warning"]
    # the whole map pure: the existing note, not the new one
    r0 = est.estimate_error_rate(s, np.zeros(120))
    assert r0["estimate"] == 0 and r0["low"] == 0 and "no labelled window was wrong" in r0["warning"]
    assert "every condition's labels" not in r0["warning"]
    # a one-window condition is labelled in full: a point interval, and never counted as starved
    one = np.r_[np.zeros(60, int), np.ones(50, int), [7]]
    e1 = (np.arange(one.size) % 4 == 0).astype(float)
    s1 = _condition_sample(20, condition=one)
    assert s1["allocation"] == [10, 9, 1]
    r1 = est.estimate_error_rate(s1, e1[s1["indices"]])
    row = r1["per_condition"]["7"]
    assert row["n_labelled"] == 1 and row["low"] == row["high"] == row["estimate"]
    assert r1["starved_strata"] == 0 and "fewer than" not in r1.get("warning", "")


def _whole_map_coverage(N0, N1, K0, K1, budget):
    """The exact coverage of the condition design's whole-map interval on two conditions of N0 and N1 windows holding
    K0 and K1 errors. The interval depends on the labels only through each condition's error count, so every pair
    (k0, k1) is run once through estimate_error_rate and weighted by its hypergeometric probability (pairs below
    1e-14 are left out, which can only lower the sum)."""
    cond, err = _two_conditions(N0, N1, K0, K1)
    base = _condition_sample(budget, condition=cond)
    n0, n1 = base["allocation"]
    pools = [(np.flatnonzero((cond == c) & (err == 1)), np.flatnonzero((cond == c) & (err == 0))) for c in (0, 1)]
    pmf = lambda Nc, Kc, nc, k: math.comb(Kc, k) * math.comb(Nc - Kc, nc - k) / math.comb(Nc, nc)
    theta, cover, cover_unfloored = (K0 + K1) / (N0 + N1), 0.0, 0.0
    for k0 in range(max(0, n0 - (N0 - K0)), min(n0, K0) + 1):
        for k1 in range(max(0, n1 - (N1 - K1)), min(n1, K1) + 1):
            pr = pmf(N0, K0, n0, k0) * pmf(N1, K1, n1, k1)
            if pr < 1e-14:
                continue
            picked = np.r_[pools[0][0][:k0], pools[0][1][:n0 - k0], pools[1][0][:k1], pools[1][1][:n1 - k1]]
            r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
            cover += pr * (r["low"] <= theta <= r["high"])
            lo, hi = est.stratified_interval_wilson(err, cond, picked, [N0, N1], N0 + N1)[1:3]   # the spec's, unfloored
            cover_unfloored += pr * (lo <= theta <= hi)
    return cover, cover_unfloored


@pytest.mark.parametrize("K0,K1,unfloored", [(20, 100, 0.531), (40, 148, 0.776), (3960, 100, 0.772)])
def test_the_whole_map_interval_covers_when_a_large_condition_looks_clean(K0, K1, unfloored):
    """A large condition beside a small degraded one, 150 labels each, the design's own use case: 4,000 windows at
    0.5% or 1% wrong (or 99%) beside 200 at 50% or 74%. The large condition often shows no error, or one, in its
    labels, and the stratified variance then counts it as known almost exactly. Without a floor on that variance the
    whole-map interval covers 0.53, 0.78 and 0.77 (the review of 2026-09-29 found 0.67 and 0.76 by simulation on
    larger maps). With the floor it covers at least 0.95, by exact enumeration of every pair of error counts."""
    cover, before = _whole_map_coverage(4000, 200, K0, K1, 300)
    assert round(before, 3) == unfloored and cover >= 0.95


def test_the_variance_floor_by_a_second_route():
    """The floor, written out: a condition not labelled in full whose p_c (1 - p_c) falls below p0 (1 - p0), with
    p0 = (z^2 / 2) / (n_c + z^2) the centre of Wilson's interval for no error in n_c labels, enters the interval's
    variance at p0. `design_variance` stays the spec's unbiased v; the interval uses `interval_variance`, and the
    warning names the condition. Two errors among 150 labels are above the floor, and nothing changes."""
    cond, err = _two_conditions(4000, 200, 40, 148)
    base = _condition_sample(300, condition=cond)
    assert base["allocation"] == [150, 150]
    wrong0, right0 = np.flatnonzero((cond == 0) & (err == 1)), np.flatnonzero((cond == 0) & (err == 0))
    wrong1, right1 = np.flatnonzero((cond == 1) & (err == 1)), np.flatnonzero((cond == 1) & (err == 0))
    W0, W1, z2 = 4000 / 4200, 200 / 4200, est.Z95 ** 2
    p0 = (z2 / 2) / (150 + z2)
    for k0 in (0, 1, 2):
        picked = np.r_[wrong0[:k0], right0[:150 - k0], wrong1[:111], right1[:39]]
        r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
        q0, q1 = k0 / 150, 111 / 150
        theta = W0 * q0 + W1 * q1
        v = W0 ** 2 * (1 - 150 / 4000) * q0 * (1 - q0) / 149 + W1 ** 2 * (1 - 150 / 200) * q1 * (1 - q1) / 149
        assert abs(r["estimate"] - theta) < 1e-12 and abs(r["design_variance"] - v) < 1e-15
        if k0 < 2:
            vf = W0 ** 2 * (1 - 150 / 4000) * p0 * (1 - p0) / 149 + W1 ** 2 * (1 - 150 / 200) * q1 * (1 - q1) / 149
            assert abs(r["interval_variance"] - vf) < 1e-15 and r["floored_conditions"] == ["0"]
            assert "in 0 at most one label differed from the rest" in r["warning"]
        else:
            vf = v
            assert "interval_variance" not in r and "floored_conditions" not in r and "warning" not in r
        n_eff = theta * (1 - theta) / vf
        lo, hi = _wilson_by_hand(theta, n_eff)
        assert abs(r["effective_n"] - n_eff) < 1e-9 and abs(r["low"] - lo) < 1e-12 and abs(r["high"] - hi) < 1e-12
        # the floor only widens: the interval holds the one the unfloored variance gives
        ulo, uhi = _wilson_by_hand(theta, theta * (1 - theta) / v) if v > 0 else _wilson_by_hand(theta, 300)
        assert r["low"] <= ulo + 1e-12 and uhi <= r["high"] + 1e-12
    # the review's case: no error among the clear part's labels, 111 of 150 in the cloudy part. The two per-condition
    # intervals allow a whole-map rate up to their weighted upper ends; the floored interval now reaches past the
    # estimate by more than the unfloored one's 0.3 points
    picked = np.r_[right0[:150], wrong1[:111], right1[:39]]
    r = est.estimate_error_rate(dict(base, indices=picked), err[picked])
    assert r["high"] - r["estimate"] > 0.015

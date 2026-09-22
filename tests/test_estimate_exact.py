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

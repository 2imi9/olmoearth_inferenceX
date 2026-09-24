"""oe_inferencex.estimate: the intervals against enumeration, the designs against their own definitions, and
the guard against the one thing a user must not do — on exp78's committed real units, not on synthetic data."""
import itertools
import json
import math
import os

import numpy as np
import pytest

import oe_inferencex as ox
from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(ROOT, "exp", "out", "exp78_units")


def _units(task):
    d = np.load(os.path.join(UNITS, f"{task}.npz"), allow_pickle=False)
    ok = np.unpackbits(d["ok_packed"])[: int(d["ok_len"][0])].astype(bool)
    N, hw, ww = (int(v) for v in d["grid"])
    tile = np.broadcast_to(np.arange(N)[:, None, None], (N, hw, ww)).reshape(-1)[ok]
    return d["margin"].astype(float), d["p1"].astype(float), d["err"].astype(float), tile


# ----------------------------------------------------------------------------- intervals
@pytest.mark.parametrize("N,K,B", [(10, 3, 4), (12, 5, 5), (15, 7, 6)])
def test_wilson_with_fpc_covers_at_its_exact_rate_by_enumeration(N, K, B):
    """The coverage the interval achieves, counted over every subset, must be a sensible number and equal what
    exp79's exact enumeration says; the same function is what exp78 graded."""
    theta, hits = K / N, 0
    for combo in itertools.combinations(range(N), B):
        k = sum(1 for i in combo if i < K)
        lo, hi = est.wilson_interval(k, B, N)
        hits += lo <= theta <= hi
    assert 0.7 < hits / math.comb(N, B) <= 1.0


def test_a_census_is_the_point_and_an_empty_sample_is_everything():
    assert est.wilson_interval(10, 50, 50) == (0.2, 0.2)
    assert est.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = est.wilson_interval(3, 30)
    assert 0 < lo < 0.1 < hi < 0.3


def test_stratified_interval_is_the_plain_mean_when_strata_are_equal_and_narrower_when_they_differ():
    rng = np.random.default_rng(0)
    N = 5000
    strata = np.repeat(np.arange(5), N // 5)
    sizes = np.bincount(strata)
    flat = rng.random(N) < 0.2                                     # same rate everywhere: stratifying buys nothing
    sharp = rng.random(N) < np.array([0.6, 0.25, 0.1, 0.03, 0.01])[strata]   # rates separate: it buys width
    picked = np.concatenate([rng.choice(np.flatnonzero(strata == h), 60, replace=False) for h in range(5)])
    for err in (flat, sharp):
        e, lo, hi, starved = est.stratified_interval(err.astype(float), strata, picked, sizes, N)
        assert starved == 0 and lo <= e <= hi
        assert abs(e - err[picked].mean()) < 1e-12               # equal allocation: the estimate is the sample mean
    _, flo, fhi, _ = est.stratified_interval(flat.astype(float), strata, picked, sizes, N)
    _, slo, shi, _ = est.stratified_interval(sharp.astype(float), strata, picked, sizes, N)
    wlo, whi = est.wilson_interval(int(flat[picked].sum()), picked.size, N)
    assert (fhi - flo) == pytest.approx(whi - wlo, rel=0.15)     # flat: about the SRS width
    # sharp: the width is the closed form, sum over strata of W_h^2 (1 - n_h/N_h) p_h q_h / (n_h - 1), z = 1.96
    ph = np.array([sharp[picked][strata[picked] == h].mean() for h in range(5)])
    var = sum((sizes[h] / N) ** 2 * (1 - 60 / sizes[h]) * ph[h] * (1 - ph[h]) / 59 for h in range(5))
    assert (shi - slo) == pytest.approx(2 * est.Z95 * math.sqrt(var), rel=1e-9)
    assert (shi - slo) < (fhi - flo)                              # and narrower than the flat case, since strata differ


def test_a_starved_stratum_is_counted_and_contributes_its_single_unit_to_the_estimate():
    N = 100
    strata = np.repeat(np.arange(2), 50)
    err = np.zeros(N); err[:50] = 1.0                              # stratum 0 all wrong, stratum 1 all right
    picked = np.array([0, 50, 51, 52])                             # one unit from stratum 0, three from stratum 1
    e, lo, hi, starved = est.stratified_interval(err, strata, picked, [50, 50], N)
    assert starved == 1 and e == pytest.approx(0.5)                # W0 * 1 + W1 * 0


def test_design_effect_is_one_for_independent_units_and_large_for_clustered_errors():
    rng = np.random.default_rng(1)
    tile = np.repeat(np.arange(200), 16)
    iid = (rng.random(3200) < 0.2).astype(float)
    clustered = np.repeat(rng.random(200) < 0.2, 16).astype(float)   # whole tiles wrong or right
    assert abs(est.design_effect(iid, tile) - 1.0) < 0.3
    assert est.design_effect(clustered, tile) > 10
    assert math.isnan(est.design_effect(iid[:32], tile[:32]))     # two tiles: not enough to estimate


# ----------------------------------------------------------------------------- designs
def test_neyman_allocation_sums_to_the_budget_floors_every_stratum_and_never_exceeds_a_stratum():
    sizes = np.array([1000, 1000, 1000, 1000, 5])
    a = est.neyman_allocation(sizes, np.array([0.5, 0.3, 0.1, 0.05, 0.9]), 300)
    assert a.sum() == 300 and a.min() >= est.MIN_PER_STRATUM and a[4] <= 5 and a[0] > a[3]
    b = est.neyman_allocation(sizes, np.ones(5), 300)
    assert b.sum() == 300 and b[:4].max() - b[:4].min() <= 1      # proportional when the spread is flat, to rounding


def test_confidence_strata_are_quintiles_with_the_least_confident_first():
    margin = np.linspace(0, 1, 1000)
    s = est.confidence_strata(margin)
    assert s.min() == 0 and s.max() == 4 and (np.bincount(s) == 200).all()
    assert s[0] == 0 and s[-1] == 4


def test_sample_designs_return_the_budget_from_valid_windows_only():
    rng = np.random.default_rng(0)
    margin, p1 = rng.random(4000), 0.5 + 0.5 * rng.random(4000)
    valid = rng.random(4000) > 0.1
    tiles = np.repeat(np.arange(250), 16)
    for design in ("random", "proportional", "confidence"):
        s = ox.sample_for_estimation(margin, 300, design=design, p1=p1, valid=valid)
        assert s["indices"].size == 300 and valid[s["indices"]].all() and len(set(s["indices"])) == 300
        assert s["n_population"] == int(valid.sum())
    t = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tiles, valid=valid, per_tile=16)
    assert t["indices"].size == 300 and valid[t["indices"]].all() and len(set(t["indices"])) == 300
    assert t["n_tiles"] == len(set(tiles[t["indices"]])) >= 19   # 250 tiles of ~14.4 valid windows: 19 or more to reach 300
    assert max(np.bincount(tiles[t["indices"]])) <= 16


def test_sample_refuses_what_it_cannot_do():
    margin = np.random.default_rng(0).random(100)
    with pytest.raises(ValueError, match="needs p1"):
        ox.sample_for_estimation(margin, 10)                      # confidence design without p1
    with pytest.raises(ValueError, match="needs `tiles`"):
        ox.sample_for_estimation(margin, 10, design="tiles")
    with pytest.raises(ValueError, match="budget must be"):
        ox.sample_for_estimation(margin, 1000, design="random")
    with pytest.raises(ValueError, match="design must be"):
        ox.sample_for_estimation(margin, 10, design="cleverest")


def test_the_confidence_design_oversamples_the_least_confident_stratum_on_a_real_map():
    """That is what buys the width: on MADOS exp78 measured a ratio of 0.63 against a random sample."""
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1)
    alloc = np.array(s["allocation"])
    assert alloc[0] > alloc[4] and alloc.sum() == 300


# ----------------------------------------------------------------------------- estimates, on real units
@pytest.mark.parametrize("design", ["random", "proportional", "confidence"])
def test_estimate_covers_the_true_rate_on_mados_at_the_rate_exp78_recorded(design):
    """Two hundred reviewer draws on exp78's MADOS export: coverage near 0.95, the confidence design narrower."""
    margin, p1, err, tile = _units("mados")
    theta = err.mean()
    cov, width = 0, 0.0
    for seed in range(200):
        s = ox.sample_for_estimation(margin, 300, design=design, p1=p1, seed=seed)
        r = ox.estimate_error_rate(s, err[s["indices"]])
        cov += r["low"] <= theta <= r["high"]
        width += r["half_width"]
    assert cov / 200 >= 0.90, (design, cov / 200)
    assert 0.01 < width / 200 < 0.04


def test_the_confidence_design_is_narrower_than_random_on_mados():
    margin, p1, err, tile = _units("mados")
    w = {}
    for design in ("random", "confidence"):
        w[design] = np.mean([ox.estimate_error_rate(s := ox.sample_for_estimation(margin, 300, design=design, p1=p1, seed=k),
                                                   err[s["indices"]])["half_width"] for k in range(100)])
    assert w["confidence"] < 0.8 * w["random"], w                # exp78: 0.63


def test_the_tile_design_reports_the_cluster_interval_and_the_naive_one_beside_it():
    """On MADOS the naive interval covered on 0.506 of draws in exp78 against a population design effect of 9.77.
    MADOS tiles hold a median of six valid windows, so one draw's design effect is noisy (5% of draws fall under
    1.5); the test is over 100 draws, and the budget must be met on every one."""
    margin, p1, err, tile = _units("mados")
    wider, deffs = 0, []
    for seed in range(100):
        s = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tile, seed=seed)
        assert s["indices"].size == 300, seed
        r = ox.estimate_error_rate(s, err[s["indices"]])
        naive = r["naive_interval_if_treated_as_random"]
        assert r["method"].startswith("ratio estimator") and "warning" in r
        wider += (r["high"] - r["low"]) > (naive["high"] - naive["low"])
        deffs.append(r["design_effect"])
    # The corrected interval is wider than the naive one on 75% of MADOS draws. The old estimator's was wider on
    # over 90%, but only because one-window tiles, whose means are 0 or 1, inflated its spread.
    assert wider >= 60 and np.median(deffs) > 3


def test_estimate_refuses_mismatched_or_non_binary_labels():
    margin = np.random.default_rng(0).random(500)
    s = ox.sample_for_estimation(margin, 50, design="random")
    with pytest.raises(ValueError, match="labels for"):
        ox.estimate_error_rate(s, np.zeros(49))
    with pytest.raises(ValueError, match="0 or 1"):
        ox.estimate_error_rate(s, np.full(50, 0.5))


# ----------------------------------------------------------------------------- the guard
@pytest.mark.parametrize("task", ["mados", "sen1floods11", "pastis_sentinel2", "m_cashew_plant"])
def test_labelling_the_review_set_and_dividing_is_refused_with_the_inflation_named(task):
    """The natural wrong thing, on the real units: the 5% review set gives 1.8 to 5.8 times the true rate."""
    margin, p1, err, tile = _units(task)
    n = margin.size
    k = int(round(0.05 * n))
    review = np.argsort(margin, kind="stable")[:k]                # least confident first: the review set
    chk = ox.review_set_check(review, margin)
    assert chk["looks_like_a_review_set"] and chk["mean_suspicion_percentile"] > 0.95
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(review, err[review], margin)
    assert err[review].mean() > 1.7 * err.mean()                  # and it would indeed have been inflated


def test_a_random_sample_passes_the_guard_and_is_estimated_as_one():
    margin, p1, err, tile = _units("mados")
    idx = np.random.default_rng(0).choice(margin.size, 300, replace=False)
    r = ox.estimate_from_indices(idx, err[idx], margin)
    assert 0.45 < r["review_set_check"]["mean_suspicion_percentile"] < 0.55
    # two hundred random draws: none refused
    for k in range(200):
        j = np.random.default_rng(k).choice(margin.size, 300, replace=False)
        assert not ox.review_set_check(j, margin)["looks_like_a_review_set"], k
    assert r["low"] <= err.mean() <= r["high"] and r["method"].startswith("exact hypergeometric")


def test_the_same_windows_are_estimated_with_their_design_and_refused_without_it():
    """The confidence design oversamples the suspect end on purpose: on MADOS its median suspicion percentile is
    about 0.72 and treating it as random would report 1.9 times the true rate. With its design it carries the
    weights that undo that and is estimated; handed over as bare indices it is refused. That is the guard's job,
    and at 0.75 the first threshold let it through: the threshold is four SDs above a random sample's mean
    percentile (the median until 2026-09-24, which heavy ties defeated)."""
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1)
    chk = ox.review_set_check(s["indices"], margin)
    assert chk["mean_suspicion_percentile"] > chk["threshold"] and abs(chk["threshold"] - 0.566) < 0.01
    assert err[s["indices"]].mean() > 1.5 * err.mean()
    r = ox.estimate_error_rate(s, err[s["indices"]])
    assert r["low"] <= err.mean() <= r["high"]
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(s["indices"], err[s["indices"]], margin)


def test_exp78_reads_its_estimators_from_the_package():
    """One implementation: the recorded run and the shipped code cannot drift apart."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("e78", os.path.join(ROOT, "exp", "exp78_error_rate_estimation.py"))
    e78 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e78)
    assert e78.wilson is est.wilson_interval and e78.stratified_interval is est.stratified_interval
    assert e78.neyman is est.neyman_allocation and e78.design_effect is est.design_effect
    assert e78.Z95 == est.Z95 and e78.MIN_PER_STRATUM == est.MIN_PER_STRATUM


# ----------------------------------------------------------------------------- what the GeoTIFF audit found, 2026-09-22
def test_a_sample_with_no_errors_gives_a_positive_upper_bound_and_a_lower_bound_of_zero():
    """Two defects at once. The stratified Wald interval collapsed to [0, 0] at nominal 95% when no sampled window
    was wrong, which a clean map makes likely; the shipped interval is Wilson on the design's effective sample
    size and falls back to the simple-random bound there. And Wilson's own lower bound sat at 0.1% at k = 0
    because the finite-population correction scaled the whole half-width; a perfect map was ruled out."""
    margin = np.random.default_rng(0).random(999)
    p1 = 0.5 + 0.5 * np.random.default_rng(1).random(999)
    s = ox.sample_for_estimation(margin, 300, p1=p1)
    r = ox.estimate_error_rate(s, np.zeros(300))
    assert r["low"] == 0.0 and 0.01 < r["high"] < 0.02 and "no labelled window was wrong" in r["warning"]
    r1 = ox.estimate_error_rate(s, np.ones(300))
    assert r1["high"] == 1.0 and 0.98 < r1["low"] < 0.99
    assert est.wilson_interval(0, 300, 999)[0] == 0.0 and est.wilson_interval(300, 300, 999)[1] == 1.0
    assert est.stratified_interval(np.zeros(999), s["strata"], np.arange(300), s["sizes"], 999)[1:3] == (0.0, 0.0)   # the graded form, kept


def test_the_shipped_stratified_interval_agrees_with_the_graded_one_away_from_the_edges_and_covers_no_worse():
    """exp78 graded the Wald form. The Wilson-effective-n form the package ships must agree with it to the third
    decimal where both are defined, and cover at least as often, on the real units of the cleanest task."""
    margin, p1, err, tile = _units("mados")
    theta = err.mean()
    cov_w = cov_s = 0
    gaps = []
    pos = None
    for seed in range(200):
        s = ox.sample_for_estimation(margin, 300, p1=p1, seed=seed)
        if pos is None:                                           # the population is the same for every seed
            pos = np.full(margin.size, -1); pos[s["strata_of_population"]] = np.arange(s["strata_of_population"].size)
        local = pos[s["indices"]]
        e = err[s["strata_of_population"]]
        _, wlo, whi, _ = est.stratified_interval(e, s["strata"], local, s["sizes"], s["n_population"])
        _, slo, shi, _, n_eff = est.stratified_interval_wilson(e, s["strata"], local, s["sizes"], s["n_population"])
        cov_w += wlo <= theta <= whi
        cov_s += slo <= theta <= shi
        gaps.append(abs((shi - slo) - (whi - wlo)))
        assert 100 < n_eff < 5000
    assert cov_s >= cov_w and cov_s / 200 >= 0.93
    assert np.median(gaps) < 3e-3


def test_the_tile_design_refuses_a_grid_that_cannot_meet_the_budget():
    """Four tiles of sixteen cannot give 300 labels; before, it returned 64 with exit 0 and a sidecar saying 300."""
    margin = np.random.default_rng(0).random(1024)
    with pytest.raises(ValueError, match="at least 5|can label at most"):
        ox.sample_for_estimation(margin, 300, design="tiles", tiles=np.repeat(np.arange(4), 256), per_tile=16)
    with pytest.raises(ValueError, match="can label at most 96 windows"):        # six tiles: enough tiles, too few labels
        ox.sample_for_estimation(margin, 300, design="tiles", tiles=np.arange(1024) % 6, per_tile=16)
    ok = ox.sample_for_estimation(margin, 300, design="tiles", tiles=np.repeat(np.arange(64), 16), per_tile=16)
    assert ok["indices"].size == 300 and ok["n_tiles"] == 19


# ----------------------------------------------------------------------------- the estimator audit, 2026-09-22
def test_the_tile_estimate_is_the_window_rate_not_the_average_tiles_rate_on_mados():
    """MADOS tiles hold 1 to 400 valid windows. The unweighted mean of tile means, which exp78 first recorded,
    targets the average tile's rate, 0.133, and returned 1.78 times the true 0.074; the ratio estimator weights
    each tile by its windows and is unbiased to within Monte Carlo noise over 150 draws."""
    margin, p1, err, tile = _units("mados")
    ests = []
    for seed in range(300):
        s = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tile, seed=seed)
        ests.append(ox.estimate_error_rate(s, err[s["indices"]])["estimate"])
    # Over 1,000 draws the ratio is 1.029 with a Monte Carlo SE of 0.030; at 300 draws the SE is 0.055, so 0.2 is
    # 3.6 SE and still separates cleanly from the old estimator's 1.78. An 8% tolerance at 150 draws, the first
    # version of this test, was one SE and failed on noise.
    assert abs(np.mean(ests) / err.mean() - 1) < 0.2, np.mean(ests) / err.mean()
    ids = np.unique(tile)
    assert np.mean([err[tile == u].mean() for u in ids]) > 1.7 * err.mean()       # what the old estimator targeted


def test_on_equal_tiles_the_ratio_estimator_is_the_mean_of_tile_means_exactly():
    """Which is why six of exp78's seven tasks did not move when the estimator was corrected."""
    rng = np.random.default_rng(0)
    tile = np.repeat(np.arange(40), 16)
    err = (rng.random(640) < 0.2).astype(float)
    picked = np.concatenate([np.flatnonzero(tile == u)[:8] for u in rng.choice(40, 12, replace=False)])
    e, lo, hi, deff, var = est.cluster_interval(err, tile, picked, quantile="normal")
    means = np.array([err[picked][tile[picked] == u].mean() for u in np.unique(tile[picked])])
    assert e == pytest.approx(means.mean(), abs=1e-12)
    assert var == pytest.approx(means.var(ddof=1) / means.size, rel=1e-12)
    assert deff == pytest.approx(var / (e * (1 - e) / picked.size), rel=1e-12)   # the realised design effect


def test_a_clean_map_labelled_by_tile_does_not_get_a_zero_width_interval():
    rng = np.random.default_rng(0)
    tiles = np.repeat(np.arange(250), 16)
    margin = rng.random(4000)
    s = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tiles, seed=1)
    r = ox.estimate_error_rate(s, np.zeros(300))
    assert r["low"] == 0.0 and r["high"] > 0.01 and "between-tile variance is zero" in r["warning"]


def test_the_t_quantile_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    for df in list(range(1, 31)) + [40, 60, 120, 1000]:
        assert est.t_quantile_975(df) == pytest.approx(st.t.ppf(0.975, df), abs=1e-5), df


def test_nan_margins_are_not_the_suspect_end_so_the_review_set_cannot_hide_behind_no_data():
    """assess's confidence array is NaN at no-data; argsort put NaN at the suspect end, and at 40% no-data the
    tool's own review set passed the guard and was estimated at 5.5 times the true rate."""
    margin, p1, err, tile = _units("mados")
    nd = np.zeros(margin.size, bool); nd[: int(0.4 * margin.size)] = True
    m_nan = np.where(nd, np.nan, margin)
    vi = np.flatnonzero(~nd)
    review = vi[np.argsort(margin[vi], kind="stable")[: int(0.05 * vi.size)]]
    assert ox.review_set_check(review, m_nan)["looks_like_a_review_set"]        # without passing `valid`
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(review, err[review], m_nan)


def test_ties_are_ranked_in_a_random_order_so_neither_raster_position_nor_a_large_tie_decides_the_guard():
    """Raster position cannot decide the verdict: the first and the last 200 windows of a tied block at the
    suspect end are both flagged, at medians within sampling noise of each other. And a genuine random sample is
    not refused when most of the map ties at the suspect end: with mid-ranks (until 2026-09-23) a tied block had one
    percentile near its middle, and at 60% tied every random sample was refused as a review set."""
    rng = np.random.default_rng(0)
    n = 20000
    tied = rng.random(n) < 0.4
    m = np.where(tied, 0.0, 0.05 + 0.95 * rng.random(n))
    first = np.argsort(m, kind="stable")[:200]
    last = np.flatnonzero(tied)[-200:]
    a, b = ox.review_set_check(first, m), ox.review_set_check(last, m)
    assert abs(a["mean_suspicion_percentile"] - b["mean_suspicion_percentile"]) < 0.05
    assert a["looks_like_a_review_set"] and b["looks_like_a_review_set"]
    for share in (0.5, 0.6, 0.7):
        heavy = np.where(rng.random(4000) < share, 0.0, 0.05 + 0.95 * rng.random(4000))
        refused = sum(ox.review_set_check(rng.choice(4000, 300, replace=False), heavy)["looks_like_a_review_set"]
                      for _ in range(50))
        assert refused <= 2, (share, refused)


def test_estimate_from_indices_refuses_no_data_windows_and_repeats():
    margin, p1, err, tile = _units("mados")
    valid = np.ones(margin.size, bool); valid[: int(0.3 * margin.size)] = False
    rng = np.random.default_rng(0)
    good = rng.choice(np.flatnonzero(valid), 150, replace=False)
    bad = rng.choice(np.flatnonzero(~valid), 150, replace=False)
    with pytest.raises(ValueError, match="outside the valid map"):
        ox.estimate_from_indices(np.concatenate([good, bad]), np.concatenate([err[good], np.zeros(150)]), margin, valid)
    with pytest.raises(ValueError, match="more than once"):
        ox.estimate_from_indices(np.tile(good[:100], 3), np.tile(err[good[:100]], 3), margin, valid)


def test_a_budget_too_small_to_floor_every_stratum_is_refused_rather_than_zeroing_the_suspect_end():
    """With a budget under 2 x 5 the allocation used to zero the least confident strata, and the estimate treated
    the most error-prone part of the map as error-free: 0.009 against a true 0.074 on MADOS."""
    with pytest.raises(ValueError, match="cannot give each"):
        est.neyman_allocation(np.array([1000] * 5), np.ones(5), 3)
    margin, p1, err, tile = _units("mados")
    with pytest.raises(ValueError, match="cannot give each"):
        ox.sample_for_estimation(margin, 4, p1=p1)


def test_a_stratified_sample_with_a_repeated_or_foreign_window_is_refused_with_a_message():
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1, seed=0)
    w = err[s["indices"]]
    dup = {**s, "indices": np.concatenate([s["indices"], s["indices"][:10]])}
    with pytest.raises(ValueError, match="more than once"):
        ox.estimate_error_rate(dup, np.concatenate([w, np.zeros(10)]))
    valid = np.ones(margin.size, bool); valid[:100] = False
    s2 = ox.sample_for_estimation(margin, 300, p1=p1, valid=valid, seed=0)
    foreign = {**s2, "indices": np.concatenate([s2["indices"][:-1], [5]])}         # window 5 is not in the population
    with pytest.raises(ValueError, match="not in the population"):
        ox.estimate_error_rate(foreign, err[foreign["indices"]])


def test_wilson_refuses_impossible_counts():
    for k, n, N in ((5, 3, None), (-1, 3, None), (2, 10, 5)):
        with pytest.raises(ValueError):
            est.wilson_interval(k, n, N)


# ----------------------------------------------------------------------------- the finite-population form, 2026-09-23
def _exact_coverage(form, N, K, n):
    lc = lambda a, r: math.lgamma(a + 1) - math.lgamma(r + 1) - math.lgamma(a - r + 1)
    theta, tot = K / N, 0.0
    for k in range(max(0, n - (N - K)), min(n, K) + 1):
        lo, hi = form(k, n, N)
        if lo <= theta <= hi:
            tot += math.exp(lc(K, k) + lc(N - K, n - k) - lc(N, n))
    return tot


def test_the_finite_population_wilson_is_the_score_inversion_at_the_effective_size():
    """exp81's generality run: a class with one error in a hundred windows, seen once in thirty labels, was
    excluded by the variance-only correction ([0.0121, 0.161] against 0.010), and its exact coverage at (100, 1, 30)
    was 0.70, which the record had read as discreteness. The score inversion at n_eff = n (N - 1) / (N - n) keeps
    the truth: written out here from the definition, it must equal the package's interval, cover that cell, reach
    zero and one at the edges, and agree with the infinite-population form to 1e-5 where the correction is small."""
    def by_hand(k, n, N):
        p = k / n
        ne = n * (N - 1) / (N - n)
        d = 1 + est.Z95 ** 2 / ne
        c = (p + est.Z95 ** 2 / (2 * ne)) / d
        h = est.Z95 * math.sqrt(p * (1 - p) / ne + est.Z95 ** 2 / (4 * ne ** 2)) / d
        return max(c - h, 0.0), min(c + h, 1.0)
    for k, n, N in ((1, 30, 100), (0, 30, 100), (30, 30, 100), (29, 45, 100), (7, 150, 300), (3, 300, 22598)):
        got, want = est.wilson_interval(k, n, N), by_hand(k, n, N)
        assert abs(got[0] - want[0]) < 1e-12 and abs(got[1] - want[1]) < 1e-12
    lo, hi = est.wilson_interval(1, 30, 100)
    assert lo < 0.01 < hi                                                 # the cell that was excluded
    assert est.wilson_interval(0, 30, 100)[0] == 0.0 and est.wilson_interval(30, 30, 100)[1] == 1.0
    assert _exact_coverage(est.wilson_interval, 100, 1, 30) == pytest.approx(1.0)
    assert _exact_coverage(est.wilson_interval, 100, 2, 60) > 0.99 and _exact_coverage(est.wilson_interval, 300, 1, 150) == pytest.approx(1.0)
    # what the variance-only form did at the same cells, kept so the defect stays visible
    def variance_only(k, n, N):
        p = k / n
        f2 = (N - n) / (N - 1)
        d = 1 + est.Z95 ** 2 / n
        c = (p + est.Z95 ** 2 / (2 * n)) / d
        h = est.Z95 * math.sqrt(f2 * p * (1 - p) / n + est.Z95 ** 2 / (4 * n ** 2)) / d
        return max(c - h, 0.0), min(c + h, 1.0)
    assert _exact_coverage(variance_only, 100, 1, 30) == pytest.approx(0.70, abs=1e-9)
    assert variance_only(1, 30, 100)[0] > 0.01
    # where the correction is small the forms agree, and the recorded budgets move by at most 1.7e-4
    for k in (0, 3, 30, 150, 300):
        a, b = est.wilson_interval(k, 300, 22598), est.wilson_interval(k, 300)
        assert abs(a[0] - b[0]) < 2e-3 and abs(a[1] - b[1]) < 2e-3
        v = variance_only(k, 300, 22598)
        assert abs(a[0] - v[0]) < 1.7e-4 and abs(a[1] - v[1]) < 1.7e-4
    a, b = est.wilson_interval(3, 300, 10 ** 9), est.wilson_interval(3, 300)
    assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5


# ----------------------------------------------------------------------------- the exact per-class interval, 2026-09-23
def test_the_hypergeometric_tails_equal_the_package_cdf_and_a_brute_force_sum():
    """The log-space ratio recurrence against `hypergeom_cdf` and against pmfs summed with math.comb."""
    for N, K, n in ((30, 7, 10), (100, 1, 30), (146, 133, 140), (1000, 3, 60), (5000, 4200, 300)):
        for k in range(max(0, n - (N - K)), min(n, K) + 1):
            pmf = lambda x: math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)
            below = sum(pmf(x) for x in range(max(0, n - (N - K)), k + 1))
            above = sum(pmf(x) for x in range(k, min(n, K) + 1))
            assert est._hyper_tail(k, n, N, K, upper=False) == pytest.approx(below, rel=1e-9, abs=1e-15)
            assert est._hyper_tail(k, n, N, K, upper=True) == pytest.approx(above, rel=1e-9, abs=1e-15)
            assert est._hyper_tail(k, n, N, K, upper=False) == pytest.approx(est.hypergeom_cdf(k, N, K, n), rel=1e-9, abs=1e-15)


def test_the_exact_interval_is_the_tail_inversion_and_covers_at_least_nominal_by_enumeration():
    """Written out by scanning every K, not by bisection: the interval is every K whose two tails at the observed
    k exceed 2.5%. Its exact coverage is at least 0.95 on a grid where the finite-population Wilson form falls to
    0.80, and it is the point on a census and everything on an empty sample."""
    def by_scan(k, n, N):
        tail_up = lambda K: sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(n, K) + 1)) / math.comb(N, n)
        tail_dn = lambda K: sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(max(0, n - (N - K)), k + 1)) / math.comb(N, n)
        keep = [K for K in range(k, N - (n - k) + 1) if tail_up(K) > 0.025 and tail_dn(K) > 0.025]
        return min(min(keep) / N, k / n), max(max(keep) / N, k / n)      # widened to hold the sample share
    for k, n, N in ((0, 10, 50), (1, 10, 50), (5, 10, 50), (10, 10, 50), (29, 30, 100), (1, 30, 100), (13, 20, 60)):
        assert est.hypergeom_interval(k, n, N) == pytest.approx(by_scan(k, n, N), abs=1e-12)
    worst = 1.0
    for N in (50, 100, 300):
        for K in (1, 2, 3, 5, 10, 25):
            for n in (10, 20, 30, 45):
                if n < N:
                    worst = min(worst, est.exact_coverage_srs(N, K, n, interval=est.hypergeom_interval))
    assert worst >= 0.95
    assert est.exact_coverage_srs(50, 1, 10) == pytest.approx(0.8)          # the Wilson form's dip on the same grid
    assert est.hypergeom_interval(7, 40, 40) == (7 / 40, 7 / 40) and est.hypergeom_interval(0, 0, 40) == (0.0, 1.0)
    with pytest.raises(ValueError):
        est.hypergeom_interval(5, 3, 40)


def test_the_random_design_user_accuracy_is_the_exact_interval():
    """estimate_per_class under a random sample reports hypergeom_interval for the user's accuracy, whatever
    `interval` says; the class that exposed the Wilson defect (one error in 100 windows, one seen in 30 labels)
    is covered."""
    rng = np.random.default_rng(3)
    N = 1000
    dec = np.repeat(np.arange(10), 100)
    ref = dec.copy(); ref[np.arange(10) * 100] = (dec[np.arange(10) * 100] + 1) % 10   # one error per class
    s = est.sample_for_estimation(rng.random(N), 300, design="random", seed=0)
    for form in ("wilson", "wald"):
        out = est.estimate_per_class(s, ref[s["indices"]], dec, n_classes=10, interval=form)
        for c in range(10):
            conf = np.zeros((10, 10), int)
            np.add.at(conf, (dec[s["indices"]], ref[s["indices"]]), 1)
            want = est.hypergeom_interval(int(conf[c, c]), int(conf[c].sum()), 100)
            ua = out["per_class"][c]["user_accuracy"]
            assert (ua["low"], ua["high"]) == want and ua["low"] <= 0.99 <= ua["high"]


def test_the_exact_interval_refuses_fractional_counts_and_an_impossible_level():
    for k, n, N in ((5.5, 10, 50), (np.float64(5.9), 10, 50), (True, 10, 50), (3, 10.2, 50)):
        with pytest.raises(ValueError):
            est.hypergeom_interval(k, n, N)
    for conf in (-0.5, 0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            est.hypergeom_interval(3, 10, 50, conf)
    assert est.hypergeom_interval(np.int64(3), 10.0, 50) == est.hypergeom_interval(3, 10, 50)


def test_the_review_set_is_refused_inside_a_large_tied_block():
    """Verification of the second review: with 60% of the map tied at the suspect end, the median put the tool's
    own review set (all inside the block) at the block's middle, where a random sample's median also sits, and let it
    through. The mean separates them: a random sample 0.50, the review set about 0.70."""
    rng = np.random.default_rng(1)
    n = 1600
    m = np.where(rng.random(n) < 0.6, 0.0, 0.05 + 0.95 * rng.random(n))
    for share in (0.05, 0.10, 0.30):
        review = np.argsort(m, kind="stable")[:int(share * n)]
        assert ox.review_set_check(review, m)["looks_like_a_review_set"], share
    refused = sum(ox.review_set_check(rng.choice(n, 300, replace=False), m)["looks_like_a_review_set"] for _ in range(100))
    assert refused == 0



def test_a_census_is_never_refused_as_a_review_set():
    """Release check of 24 September: at a census the threshold is exactly 0.5 and the mean percentile is 0.5 up to
    rounding, so a strict comparison refused 2 to 5% of censuses as enriched sets, in whatever order the indices
    came; 1.2.0 accepted them. A census is by definition not an enriched sample."""
    refused = 0
    for N in range(2, 400):
        rng = np.random.default_rng(N)
        m = rng.normal(size=N)
        for idx in (np.arange(N), rng.permutation(N)):
            refused += est.review_set_check(idx, m)["looks_like_a_review_set"]
    assert refused == 0
    rng = np.random.default_rng(0)
    m, wrong = rng.normal(size=100), (rng.random(100) < 0.1).astype(int)
    r = est.estimate_from_indices(np.arange(100), wrong, m)
    assert r["low"] == r["high"] == r["estimate"] == wrong.mean()
    z = est.certify_zone(m, np.arange(100), np.zeros(100), 0.1)
    assert z.get("note")
    # one window short of a census is still graded as a sample, and the tool's review set is still refused
    assert not est.review_set_check(np.arange(99), m)["looks_like_a_review_set"]
    assert est.review_set_check(np.argsort(m)[:10], m)["looks_like_a_review_set"]

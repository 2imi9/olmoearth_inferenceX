"""Per-class accuracy from a labelled sample (docs/plan/per_class_assessment.md, exp81), checked by enumeration
where the claim is an equality and by a large seeded Monte Carlo where it is a variance.

Under the confidence design every per-class quantity is built from Horvitz-Thompson totals over the margin strata;
over EVERY stratified sample of a small population those totals are unbiased and their variance estimator is
unbiased for the exact variance. Under a random sample the user's accuracy interval must be the Wilson interval on
the labelled windows of that map class, the class shares the post-stratified form of Olofsson et al. 2014, and the
producer's accuracy their eq. 7, written out independently here."""
import itertools

import numpy as np
import pytest

from oe_inferencex import estimate as est

# a population of 12 windows: map class (rows) and reference class, three margin strata of sizes 5, 4, 3
MAP = np.array([0, 0, 1, 1, 2,   0, 1, 1, 2,   0, 2, 2])
REF = np.array([0, 1, 1, 1, 2,   0, 0, 1, 1,   0, 2, 2])
STRATA = np.repeat(np.arange(3), (5, 4, 3))
SIZES = [5, 4, 3]
N = 12
ALLOC = (2, 2, 2)


def _every_stratified_sample():
    groups = [np.flatnonzero(STRATA == h) for h in range(3)]
    for parts in itertools.product(*[itertools.combinations(g, a) for g, a in zip(groups, ALLOC)]):
        yield np.concatenate([np.array(p) for p in parts])


def _sample_obj(local):
    return {"design": "confidence", "indices": local, "n_population": N, "strata": STRATA,
            "strata_of_population": np.arange(N), "sizes": SIZES, "n_strata": 3, "budget": int(local.size)}


def test_ht_totals_are_unbiased_and_their_variance_estimator_is_unbiased_over_every_stratified_sample():
    both = ((MAP == 1) & (REF == 1)).astype(float)
    truth = both.sum()
    ests, vars_ = [], []
    for local in _every_stratified_sample():
        t, v = est._ht_total(both, STRATA, local, SIZES)
        ests.append(t)
        vars_.append(v)
    ests, vars_ = np.array(ests), np.array(vars_)
    assert abs(ests.mean() - truth) < 1e-12
    assert abs(vars_.mean() - ests.var()) < 1e-12


def test_per_class_under_the_confidence_design_returns_ratios_of_the_same_totals():
    local = np.array([0, 3, 5, 7, 9, 11])
    out = est.estimate_per_class(_sample_obj(local), REF[local], MAP)
    both = ((MAP == 1) & (REF == 1)).astype(float)
    ua, _ = est._ht_ratio(both, (MAP == 1).astype(float), STRATA, local, SIZES)
    pa, _ = est._ht_ratio(both, (REF == 1).astype(float), STRATA, local, SIZES)
    tot, _ = est._ht_total((REF == 1).astype(float), STRATA, local, SIZES)
    assert abs(out["per_class"][1]["user_accuracy"]["estimate"] - ua) < 1e-12
    assert abs(out["per_class"][1]["producer_accuracy"]["estimate"] - pa) < 1e-12
    assert abs(out["per_class"][1]["reference_share"]["estimate"] - tot / N) < 1e-12
    assert out["confusion_counts"][1][1] == int(((MAP[local] == 1) & (REF[local] == 1)).sum())
    assert "warning" in out["per_class"][1]                     # far fewer than 30 labelled


def _olofsson(conf, N_map):
    """Olofsson et al. 2014, eq. 4-7, written from the paper: rows map class, columns reference class."""
    C = conf.shape[0]
    n_i = conf.sum(1)
    W = N_map / N_map.sum()
    p = np.zeros((C, C))
    for i in range(C):
        if n_i[i] > 0:
            p[i] = W[i] * conf[i] / n_i[i]
    ua = np.array([p[i, i] / p[i].sum() if p[i].sum() > 0 else np.nan for i in range(C)])
    pa = np.array([p[j, j] / p[:, j].sum() if p[:, j].sum() > 0 else np.nan for j in range(C)])
    share = p.sum(0)
    v_share = np.zeros(C)
    for j in range(C):
        v_share[j] = sum(W[i] ** 2 * (conf[i, j] / n_i[i]) * (1 - conf[i, j] / n_i[i]) / (n_i[i] - 1) * (1 - n_i[i] / N_map[i])
                         for i in range(C) if n_i[i] > 1)
    v_pa = np.full(C, np.nan)
    fpc = 1 - n_i / N_map                                       # the record's form: eq. 7 with each class's FPC
    for j in range(C):
        Nhat = sum(N_map[i] * conf[i, j] / n_i[i] for i in range(C) if n_i[i] > 0)
        if Nhat > 0 and n_i[j] > 1:
            u = conf[j, j] / n_i[j]
            t1 = N_map[j] ** 2 * (1 - pa[j]) ** 2 * u * (1 - u) * fpc[j] / (n_i[j] - 1)
            t2 = pa[j] ** 2 * sum(N_map[i] ** 2 * (conf[i, j] / n_i[i]) * (1 - conf[i, j] / n_i[i]) * fpc[i] / (n_i[i] - 1)
                                  for i in range(C) if i != j and n_i[i] > 1)
            v_pa[j] = (t1 + t2) / Nhat ** 2
    return ua, pa, share, v_share, v_pa


def test_random_design_matches_wilson_per_class_and_olofssons_equations():
    rng = np.random.default_rng(3)
    Npop, C, B = 3000, 4, 300
    mc = rng.integers(0, C, Npop)
    ref = np.where(rng.random(Npop) < 0.8, mc, rng.integers(0, C, Npop))
    idx = rng.choice(Npop, B, replace=False)
    sample = {"design": "random", "indices": idx, "n_population": Npop, "budget": B}
    out = est.estimate_per_class(sample, ref[idx], mc, interval="wald")       # the field's form, for the equations
    conf = np.array(out["confusion_counts"])
    N_map = np.bincount(mc, minlength=C).astype(float)
    ua, pa, share, v_share, v_pa = _olofsson(conf, N_map)
    for c in range(C):
        row = out["per_class"][c]
        lo, hi = est.wilson_interval(int(conf[c, c]), int(conf[c].sum()), int(N_map[c]))
        assert abs(row["user_accuracy"]["estimate"] - ua[c]) < 1e-12
        assert (row["user_accuracy"]["low"], row["user_accuracy"]["high"]) == (lo, hi)
        assert abs(row["producer_accuracy"]["estimate"] - pa[c]) < 1e-12
        assert abs(row["producer_accuracy"]["high"] - min(1.0, pa[c] + est.Z95 * np.sqrt(v_pa[c]))) < 1e-12
        assert abs(row["reference_share"]["estimate"] - share[c]) < 1e-12
        assert abs(row["reference_share"]["high"] - min(1.0, share[c] + est.Z95 * np.sqrt(v_share[c]))) < 1e-12
    assert abs(out["overall_accuracy"]["estimate"] - sum(N_map[c] / Npop * conf[c, c] / conf[c].sum() for c in range(C))) < 1e-12


@pytest.mark.parametrize("design", ["random", "confidence"])
def test_the_variance_estimators_track_the_monte_carlo_variance(design):
    """On a population with an unbalanced map, the mean estimated variance of every per-class quantity is within
    15% of the Monte Carlo variance of its estimate over 3000 draws, and the estimates are unbiased to 3 SEs."""
    rng = np.random.default_rng(7)
    Npop, C, B, R = 4000, 3, 400, 3000
    margin = rng.random(Npop)
    mc = np.where(margin < 0.3, 0, np.where(margin < 0.8, 1, 2))                 # class depends on margin
    ref = np.where(rng.random(Npop) < 0.6 + 0.35 * margin, mc, rng.integers(0, C, Npop))
    p1 = 0.5 + 0.5 * margin
    N_map = np.bincount(mc, minlength=C)
    truth_ua = np.array([((mc == c) & (ref == c)).sum() / N_map[c] for c in range(C)])
    truth_share = np.bincount(ref, minlength=C) / Npop
    ests, vars_ = {k: [] for k in ("ua", "pa", "share")}, {k: [] for k in ("ua", "pa", "share")}
    for r in range(R):
        s = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=r)
        out = est.estimate_per_class(s, ref[s["indices"]], mc)
        for c in range(1, 2):                                                   # the middle class, which cuts all strata
            row = out["per_class"][c]
            for k, key in (("ua", "user_accuracy"), ("pa", "producer_accuracy"), ("share", "reference_share")):
                ests[k].append(row[key]["estimate"])
                vars_[k].append(((row[key]["high"] - row[key]["low"]) / (2 * est.Z95)) ** 2)
    for k in ("ua", "pa", "share"):
        e, v = np.array(ests[k]), np.array(vars_[k])
        ratio = v.mean() / e.var()
        assert 0.85 < ratio < 1.15, (design, k, ratio)
    truth_pa1 = ((mc == 1) & (ref == 1)).sum() / (ref == 1).sum()
    for k, t in (("ua", truth_ua[1]), ("pa", truth_pa1), ("share", truth_share[1])):
        e = np.array(ests[k])
        assert abs(e.mean() - t) < 3 * e.std() / np.sqrt(R) + 1e-3, (design, k, e.mean(), t)


def test_refusals():
    rng = np.random.default_rng(0)
    mc = rng.integers(0, 3, 500)
    s = est.sample_for_estimation(rng.random(500), 60, design="random")
    with pytest.raises(ValueError, match="reference labels for"):
        est.estimate_per_class(s, np.zeros(10, int), mc)
    with pytest.raises(ValueError, match=">= 0"):
        est.estimate_per_class(s, np.full(60, -1), mc)
    with pytest.raises(ValueError, match="drawn from"):
        est.estimate_per_class(s, np.zeros(60, int), np.r_[mc, -1, 2])
    t = est.sample_for_estimation(rng.random(500), 64, design="tiles", tiles=np.arange(500) // 8)
    with pytest.raises(ValueError, match="tile sample"):
        est.estimate_per_class(t, np.zeros(64, int), mc)
    bad = mc.copy()
    bad[s["indices"][0]] = -1
    with pytest.raises(ValueError):
        est.estimate_per_class(s, np.zeros(60, int), bad)

"""Properties the estimators must hold on ANY map, checked over random maps with the extremes included.

The second review of 2026-09-23 found valid samples refused because a sum of weights rounded past 1, a random
sample refused because half the map tied at one margin, a producer's accuracy missing for a class the map never
predicts, and a census reported with an interval. Each is an extreme that hand-picked tests never reached. Here
every estimator runs on seeded random maps that include them: every label right, every label wrong, a full census,
heavy ties, no-data, classes the map never predicts. The properties are the ones a user relies on: no refusal of
a valid input, every interval inside [0, 1] and containing its estimate, a census as a point."""
import numpy as np
import pytest

from oe_inferencex import estimate as est

SEEDS = range(40)


def _margin(rng, n, ties):
    m = rng.random(n)
    if ties:                                              # a map whose margins take a handful of values
        m = np.round(m * ties) / ties
        m[rng.random(n) < rng.uniform(0.3, 0.7)] = 0.0    # and a large block tied at the least-confident end
    return m


def _ok(block):
    return 0.0 <= block["low"] <= block["estimate"] <= block["high"] <= 1.0


def _wrong(rng, n, kind):
    return {"none": np.zeros(n), "all": np.ones(n), "some": (rng.random(n) < rng.uniform(0.02, 0.4)).astype(float)}[kind]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_error_rate_accepts_every_valid_sample_and_reports_a_sane_interval(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(200, 3000))
    margin = _margin(rng, n, ties=int(rng.choice([0, 3, 5])))
    valid = rng.random(n) > rng.uniform(0, 0.3)
    p1 = 0.5 + 0.5 * rng.random(n)
    N = int(valid.sum())
    for design in ("random", "confidence", "proportional"):
        for budget in (min(60, N), min(300, N), N):
            s = est.sample_for_estimation(margin, budget, design=design, p1=p1, valid=valid, seed=seed)
            for kind in ("none", "all", "some"):
                r = est.estimate_error_rate(s, _wrong(rng, len(s["indices"]), kind))
                assert _ok(r), (design, budget, kind, r)
                if budget == N:                           # a census has no sampling error
                    assert r["low"] == r["high"] == r["estimate"], (design, kind, r)


@pytest.mark.parametrize("seed", SEEDS)
def test_per_class_accepts_every_valid_sample_and_reports_sane_intervals(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(300, 3000))
    C = int(rng.integers(2, 7))
    margin = _margin(rng, n, ties=int(rng.choice([0, 4])))
    mc = rng.integers(0, C - 1, n)                        # class C-1 is never predicted by the map
    truth = np.where(rng.random(n) < rng.uniform(0.6, 0.98), mc, rng.integers(0, C, n))
    p1 = 0.5 + 0.5 * rng.random(n)
    for design in ("random", "confidence"):
        for budget in (min(300, n), n - 1):
            s = est.sample_for_estimation(margin, budget, design=design, p1=p1, seed=seed)
            idx = s["indices"]
            for ref in (mc[idx], truth[idx], (mc[idx] + 1) % C):      # every label right, some wrong, every one wrong
                out = est.estimate_per_class(s, ref, mc, n_classes=C)
                assert _ok(out["overall_accuracy"])
                for c, row in out["per_class"].items():
                    for q in ("user_accuracy", "producer_accuracy", "reference_share"):
                        if row[q] is not None:
                            assert _ok(row[q]), (design, c, q, row[q])
                    if c == C - 1 and row["n_labelled_reference_class"] > 0:
                        assert row["producer_accuracy"] == {"estimate": 0.0, "low": 0.0, "high": 0.0}
                        assert "never predicted" in row["warning_codes"]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_random_sample_is_never_refused_as_a_review_set_whatever_the_ties(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(1000, 5000))
    margin = _margin(rng, n, ties=int(rng.choice([2, 3, 10])))
    refused = 0
    for r in range(10):
        idx = rng.choice(n, 300, replace=False)
        refused += est.review_set_check(idx, margin)["looks_like_a_review_set"]
    assert refused <= 1                                    # the threshold is 2.4 standard errors: rare, never routine


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("rule", ["prefix", "bonferroni", "plugin"])
def test_certify_runs_on_every_random_sample_and_explains_itself(seed, rule):
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(1000, 5000))
    margin = _margin(rng, n, ties=int(rng.choice([0, 3])))
    idx = rng.choice(n, 300, replace=False)
    wrong = (rng.random(300) < rng.uniform(0, 0.2)).astype(float)
    for alpha in (0.01, 0.1, 0.4):
        z = est.certify_zone(margin, idx, wrong, alpha, rule=rule)
        assert z.get("note"), (rule, alpha)                   # the CLI prints it; bonferroni used to have none
        if z["coverage"] is not None:
            assert 0 < z["n_zone"] <= n and 0 <= z["upper_bound"] <= 1


def test_the_exact_interval_is_never_below_nominal_on_a_random_grid():
    rng = np.random.default_rng(7)
    for _ in range(60):
        N = int(rng.integers(20, 400))
        K = int(rng.integers(0, N + 1))
        n = int(rng.integers(1, N))
        assert est.exact_coverage_srs(N, K, n, interval=est.hypergeom_interval) >= 0.95 - 1e-12, (N, K, n)

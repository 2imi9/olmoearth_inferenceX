"""Tie-aware AURC, excess AURC, operating points and calibration (oe_inferencex.metrics)."""
import numpy as np
import pytest

from oe_inferencex.metrics import (aurc_expected, capture_at_budget, capture_at_budget_expected, excess_aurc,
                                   expected_calibration_error, oracle_aurc, risk_coverage, selective_accuracy)


def test_aurc_matches_plain_aurc_without_ties():
    rng = np.random.default_rng(0)
    u, e = rng.random(500), (rng.random(500) < 0.1).astype(float)
    order = np.argsort(u)
    plain = (np.cumsum(e[order]) / np.arange(1, 501)).mean()
    assert aurc_expected(u, e) == pytest.approx(plain, abs=1e-12)


def test_aurc_is_independent_of_raster_order_under_ties():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 4, 400).astype(float)          # heavy ties
    e = (rng.random(400) < 0.2).astype(float)
    perm = rng.permutation(400)
    assert aurc_expected(u, e) == pytest.approx(aurc_expected(u[perm], e[perm]), abs=1e-12)


def test_constant_score_gives_the_error_rate():
    e = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], float)
    assert aurc_expected(np.zeros(10), e) == pytest.approx(e.mean(), abs=1e-12)


def test_oracle_closed_form_equals_perfect_ranking():
    rng = np.random.default_rng(2)
    for n, k in ((1024, 97), (225, 3), (50, 50), (50, 0)):
        e = np.zeros(n)
        e[rng.choice(n, k, replace=False)] = 1
        assert oracle_aurc(n, k) == pytest.approx(aurc_expected(e, e), abs=1e-12)
        assert excess_aurc(e, e) == pytest.approx(0.0, abs=1e-12)
        assert excess_aurc(-e, e) >= excess_aurc(e, e)


def test_risk_coverage_endpoints():
    u = np.array([0.1, 0.9, 0.5, 0.2])
    e = np.array([0, 1, 0, 0], float)
    cov, risk, a = risk_coverage(u, e)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(0.25)
    assert a == pytest.approx(aurc_expected(u, e))


def test_capture_and_selective_accuracy():
    u = np.array([5, 4, 3, 2, 1, 0], float)     # unit 0 most suspect
    e = np.array([1, 1, 0, 0, 0, 0], float)
    cap = capture_at_budget(u, e, budgets=(1 / 6, 2 / 6, 1.0))
    assert cap[1 / 6] == pytest.approx(0.5) and cap[2 / 6] == pytest.approx(1.0) and cap[1.0] == pytest.approx(1.0)
    sel = selective_accuracy(u, 1 - e, coverages=(0.5, 1.0))
    assert sel[0.5] == pytest.approx(1.0) and sel[1.0] == pytest.approx(4 / 6)


def test_ece_of_a_calibrated_and_a_miscalibrated_model():
    conf = np.array([0.95, 0.95, 0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 1, 1, 1, 0], float)
    ece, rows = expected_calibration_error(conf, correct, bins=10)
    assert ece == pytest.approx(4 / 6 * 0.05 + 2 / 6 * 0.05, abs=1e-12)
    assert [r[2] for r in rows] == [2, 4]
    assert expected_calibration_error(np.full(6, 0.99), np.ones(6))[0] == pytest.approx(0.01)


def test_expected_capture_equals_plain_capture_without_ties_and_averages_ties():
    rng = np.random.default_rng(3)
    u, e = rng.random(200), (rng.random(200) < 0.15).astype(float)
    plain, exp_ = capture_at_budget(u, e, (0.05, 0.1, 0.2)), capture_at_budget_expected(u, e, (0.05, 0.1, 0.2))
    assert all(plain[b] == pytest.approx(exp_[b]) for b in (0.05, 0.1, 0.2))
    tied = np.zeros(10)                                    # everything tied: expected capture is the budget itself
    err = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], float)
    assert capture_at_budget_expected(tied, err, (0.2, 0.5))[0.2] == pytest.approx(0.2)
    assert capture_at_budget_expected(tied, err, (0.2, 0.5))[0.5] == pytest.approx(0.5)
    perm = rng.permutation(10)                              # raster order must not matter
    assert capture_at_budget_expected(tied[perm], err[perm], (0.2,))[0.2] == pytest.approx(0.2)
    two = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)  # a two-level score: top group holds all 4 errors
    assert capture_at_budget_expected(two, err, (0.2,))[0.2] == pytest.approx(0.5)   # k = 2 of the 4-strong top group
    assert capture_at_budget_expected(np.array([0, 1], np.uint8), np.array([0, 1], float), (0.5,))[0.5] == pytest.approx(1.0)
    assert capture_at_budget_expected(np.array([False, True]), np.array([0, 1], float), (0.5,))[0.5] == pytest.approx(1.0)

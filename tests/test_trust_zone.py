"""The trusted-zone certification (docs/plan/trust_zone.md, exp80), checked by enumeration rather than simulation.

The guarantee P(zone error rate > alpha) <= delta is a theorem for the Bonferroni rule and, on a population whose
zone error rate does not fall as the zone grows, for the prefix rule. On a population of 20 windows and a budget of
8 every one of the 125,970 possible draws can be enumerated, so the violation probability is computed exactly and
compared with delta; the hypergeometric sums are checked against exact rational arithmetic; and the arithmetic
limit on what a budget can certify is checked at both edges."""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est


def _choose(n, r):
    return math.comb(n, r)


def _cdf_exact(k, n, K, b):
    lo, hi = max(0, b - (n - K)), min(b, K)
    if k < lo:
        return Fraction(0)
    if k >= hi:
        return Fraction(1)
    return sum(Fraction(_choose(K, x) * _choose(n - K, b - x), _choose(n, b)) for x in range(lo, k + 1))


@pytest.mark.parametrize("n", [7, 12, 20])
def test_the_hypergeometric_cdf_equals_the_exact_rational_sum(n):
    for K in range(n + 1):
        for b in range(n + 1):
            for k in range(b + 1):
                assert abs(est.hypergeom_cdf(k, n, K, b) - float(_cdf_exact(k, n, K, b))) < 1e-12


def test_the_p_value_is_the_supremum_over_the_null_and_a_valid_test():
    n, b, alpha = 40, 10, 0.3
    K0 = math.floor(alpha * n) + 1
    for k in range(b + 1):
        p = est.zone_pvalue(k, b, n, alpha)
        assert abs(p - float(_cdf_exact(k, n, K0, b))) < 1e-12
        for K in range(K0, n + 1):                       # every null count gives a p-value no larger
            assert est.hypergeom_cdf(k, n, K, b) <= p + 1e-12
    assert est.zone_pvalue(0, 0, n, alpha) == 1.0        # no label inside the zone says nothing
    # under any null count, P(p <= delta) <= delta: the test is valid at every level
    for K in range(K0, n + 1):
        for delta in (0.05, 0.1, 0.2):
            prob = sum(float(_cdf_exact(k, n, K, b) - _cdf_exact(k - 1, n, K, b))
                       for k in range(b + 1) if est.zone_pvalue(k, b, n, alpha) <= delta)
            assert prob <= delta + 1e-12


def test_the_upper_bound_is_the_largest_rate_not_rejected():
    for n, b, k, delta in [(50, 10, 0, 0.1), (50, 10, 3, 0.1), (300, 40, 2, 0.05), (20, 8, 8, 0.1), (20, 8, 0, 0.1)]:
        ub = est.zone_upper_bound(k, b, n, delta)
        K = int(round(ub * n))
        assert est.hypergeom_cdf(k, n, K, b) > delta or K == n - (b - k)
        if K < n - (b - k):
            assert est.hypergeom_cdf(k, n, K + 1, b) <= delta
    assert est.zone_upper_bound(0, 0, 50) == 1.0


def test_the_minimum_label_count_sits_exactly_at_the_edge():
    for alpha, delta in [(0.05, 0.1), (0.02, 0.1), (0.009, 0.1), (0.3, 0.05)]:
        b_min = est.min_labels_to_certify(alpha, delta)
        n = 10 ** 6                                      # the hypergeometric is the binomial in this limit
        assert est.zone_pvalue(0, b_min, n, alpha) <= delta
        assert est.zone_pvalue(0, b_min - 1, n, alpha) > delta
    assert est.min_labels_to_certify(0.05, 0.1) == 45
    assert est.min_labels_to_certify(0.009, 0.1) == 255


def test_the_zone_order_is_descending_margin_then_index_and_ignores_invalid_windows():
    margin = np.array([0.5, 0.9, np.nan, 0.9, 0.1, 0.7])
    valid = np.array([True, True, True, True, False, True])
    order, pos = est.zone_order(margin, valid)
    assert order.tolist() == [1, 3, 5, 0]
    assert pos.tolist() == [3, 0, -1, 1, -1, 2]


def test_zone_counts_agree_with_a_direct_count():
    rng = np.random.default_rng(0)
    positions = rng.choice(200, 30, replace=False)
    wrong = rng.integers(0, 2, 30)
    sizes = [10, 50, 120, 200]
    b, k = est.zone_counts(positions, wrong, sizes)
    for j, n in enumerate(sizes):
        inside = positions < n
        assert b[j] == inside.sum() and k[j] == wrong[inside].sum()


# ----------------------------------------------------------------------------- the guarantee, by enumeration
N20, B8, ALPHA, DELTA = 20, 8, 0.3, 0.1


def _every_draw_outcome(err, rule):
    """Over every draw of B8 from N20 in zone order (positions 0..19), the exact probability that the certified
    zone is wrong more than ALPHA of the time, and the probability that some zone is certified."""
    cov, sizes, _ = est.zone_levels(N20, B8, ALPHA, DELTA)
    R = np.cumsum(err)[np.array(sizes) - 1] / np.array(sizes)
    n_draw = n_viol = n_cert = 0
    for draw in itertools.combinations(range(N20), B8):
        pos = np.array(draw)
        b, k = est.zone_counts(pos, err[pos], sizes)
        p = [est.zone_pvalue(kk, bb, n, ALPHA) for kk, bb, n in zip(k, b, sizes)]
        _, best = est.apply_zone_rule(p, b, k, ALPHA, DELTA, rule)
        n_draw += 1
        if best is not None:
            n_cert += 1
            n_viol += R[best] > ALPHA
    return n_viol / n_draw, n_cert / n_draw, cov, R


def test_the_grid_cut_leaves_three_levels_at_this_budget():
    cov, sizes, c_min = est.zone_levels(N20, B8, ALPHA, DELTA)
    assert est.min_labels_to_certify(ALPHA, DELTA) == 7 and abs(c_min - 7 / 8) < 1e-12
    assert cov == [0.9, 0.95, 1.0] and sizes == [18, 19, 20]


@pytest.mark.parametrize("rule", ["prefix", "bonferroni"])
def test_the_guarantee_holds_exactly_on_a_monotone_population(rule):
    err = np.zeros(N20)
    err[13:] = 1                                          # 7 errors at the least confident end: R = 5/18, 6/19, 7/20
    viol, cert, cov, R = _every_draw_outcome(err, rule)
    assert R[0] <= ALPHA < R[1] <= R[2]                   # only the 0.9 zone is truly good
    assert viol <= DELTA
    assert cert > 0                                       # the rule is not vacuous: it certifies the good zone sometimes


def test_the_bonferroni_rule_holds_exactly_when_the_zone_rate_falls_as_it_grows():
    err = np.zeros(N20)
    err[[0, 1]] = 1                                       # two confident errors
    err[14:] = 1                                          # and six at the end: R = 6/18, 7/19, 8/20, all above alpha
    viol, cert, cov, R = _every_draw_outcome(err, "bonferroni")
    assert (R > ALPHA).all()
    assert viol <= DELTA and viol == cert                 # every certification is a violation, and there are few


def test_the_plugin_rule_violates_far_more_often_than_delta_at_the_boundary():
    err = np.zeros(N20)
    err[13:] = 1
    viol, cert, cov, R = _every_draw_outcome(err, "plugin")
    assert viol > 2 * DELTA


# ----------------------------------------------------------------------------- end to end
def test_certify_zone_refuses_what_it_must_and_certifies_a_clean_map():
    rng = np.random.default_rng(1)
    N = 5000
    margin = rng.random(N)
    err = (rng.random(N) < 0.02 * (1 - margin) * 4).astype(float)   # errors concentrate at low margin
    idx = rng.choice(N, 300, replace=False)
    out = est.certify_zone(margin, idx, err[idx], alpha=0.05)
    assert out["coverage"] is not None and out["coverage"] >= 0.5
    assert out["upper_bound"] <= 0.05 and out["n_zone"] == int(round(out["coverage"] * N))
    assert len(out["zone_indices_in_order"]) == out["n_zone"]
    with pytest.raises(ValueError, match="enriched set"):                         # the review set is refused
        est.certify_zone(margin, np.argsort(margin)[:300], err[np.argsort(margin)[:300]], alpha=0.05)
    with pytest.raises(ValueError, match="more than once"):
        est.certify_zone(margin, np.r_[idx[:10], idx[:10]], err[np.r_[idx[:10], idx[:10]]], alpha=0.05)
    small = est.certify_zone(margin, idx[:20], err[idx[:20]], alpha=0.05)         # 20 labels cannot certify 5%
    assert small["coverage"] is None and "needs 45" in small["note"]
    plug = est.certify_zone(margin, idx, err[idx], alpha=0.05, rule="plugin")
    assert plug["coverage"] is not None and "no guarantee" in plug["note"]

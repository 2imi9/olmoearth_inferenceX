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


# ----------------------------------------------------------------------------- a zone per input condition
# certify_by_condition certifies a zone inside each input condition holding at least min_labels_to_certify(alpha,
# delta) labels, each at delta / L with L those conditions, so that every statement holds together except on at most
# delta of samples (condition_spec 2.8). L is fixed by the label counts before any label is read.
def _condition_map(sizes, rates, seed=0, values=None):
    """A map of conditions of the given sizes, errors concentrated at low margin at each condition's own rate."""
    rng = np.random.default_rng(seed)
    values = list(range(len(sizes))) if values is None else values
    cond = np.concatenate([np.full(n, v) for n, v in zip(sizes, values)])
    margin = rng.random(cond.size)
    rate = np.concatenate([np.full(n, r) for n, r in zip(sizes, rates)])
    err = (rng.random(cond.size) < np.clip(2 * rate * (1 - margin), 0, 1)).astype(float)
    return cond, margin, err


def test_delta_split_known_answers():
    """The split and the label minima: (150, 150, 12) labels at alpha 0.05 and delta 0.1 test two conditions at 0.05
    each; (60, 60, 60) tests three at 0.0333, and none can certify anything, since 67 labels are needed at that
    delta; (285, 15) tests one at the full delta. The minima for L = 1 to 5 are 45, 59, 67, 72, 77."""
    assert [est.min_labels_to_certify(0.05, 0.1 / L) for L in range(1, 6)] == [45, 59, 67, 72, 77]
    for sizes, B, labels, L in [((3000, 3000, 12), 312, [150, 150, 12], 2), ((2000, 2000, 2000), 180, [60, 60, 60], 3),
                                ((4000, 15), 300, [285, 15], 1)]:
        cond, margin, err = _condition_map(sizes, [0.01] * len(sizes), seed=len(sizes))
        s = est.sample_for_estimation(margin, B, design="condition", condition=cond)
        assert s["allocation"] == labels
        r = est.certify_by_condition(s, err[s["indices"]], margin, 0.05, delta=0.1)
        assert r["n_conditions_tested"] == L and r["delta_per_condition"] == pytest.approx(0.1 / L, abs=1e-15)
        assert r["min_labels_to_certify"] == 45 and r["delta"] == 0.1
        for (name, part), n in zip(r["per_condition"].items(), labels):
            assert part["n_labelled"] == n and part["tested"] == (n >= 45), (name, n)
            if part["tested"]:
                assert part["delta"] == r["delta_per_condition"]
            else:
                assert part["reason"] == f"{n} labels; certifying any zone at alpha 0.05 needs at least 45"
                assert part["coverage"] is None and part["delta"] is None
        assert r["note"] == (est.FAMILY_NOTE if L > 1 else est.FAMILY_NOTE_ONE).format(L=L, b1=45, d=0.1 / L, delta=0.1,
                                                                                    alpha=0.05)
        if L == 1:                          # one condition tested: the note speaks of it, not of "each of the 1"
            assert r["note"] == ("Certified per input condition. The one condition with at least 45 labels is tested "
                                 "at delta 0.1, so its statement fails on at most 0.1 of samples. When it holds, the "
                                 "certified windows are wrong at most 0.05 of the time. Conditions with fewer labels "
                                 "are not tested. Outside the certified windows nothing is certified.")
        if L == 3:
            assert all(p["coverage"] is None and "needs 67" in p["note"] for p in r["per_condition"].values())
            assert r["certified_share_of_map"] is None


ALPHA2, DELTA2, NC = 0.5, 0.2, 8
PATTERNS = {"just_bad": [0, 1, 0, 1, 0, 1, 1, 1], "good_then_bad": [0, 0, 0, 0, 1, 1, 1, 1],
            "confident_errors": [1, 1, 1, 0, 0, 0, 0, 1], "all_good": [0] * 8, "five_of_eight": [1, 0, 1, 0, 1, 0, 1, 1]}


def test_per_condition_certification_holds_the_family_delta_by_enumeration():
    """Two conditions of 8 windows, a random sample of 8, every one of the 12,870 draws, alpha 0.5, delta 0.2, five
    error patterns per condition (in zone order, most confident first). For every pair of patterns the probability
    that some certified zone is wrong more than alpha of the time is at most delta; certification is not vacuous;
    and on the worst pair certify_by_condition itself runs on every draw, its parts are the zones below, and the
    certified windows taken together are within alpha whenever every part is."""
    margin = np.r_[np.linspace(1, 0.3, NC), np.linspace(0.9, 0.2, NC)]       # zone order inside a condition = position
    cond = np.r_[np.zeros(NC, int), np.ones(NC, int)]
    base = est.sample_for_estimation(margin, NC, design="random", condition=cond)
    b1 = est.min_labels_to_certify(ALPHA2, DELTA2)
    draws = list(itertools.combinations(range(2 * NC), NC))
    assert len(draws) == 12870 and b1 == 3
    memo = {}

    def part(c, pat, sub, L):
        """The zone certify_zone gives inside condition c at delta / L: (size, its true error rate) or (None, None)."""
        key = (c, pat, sub, L)
        if key not in memo:
            e = np.array(PATTERNS[pat], float)
            r = est.certify_zone(margin, np.array(sub, int) + c * NC, e[list(sub)], ALPHA2, delta=DELTA2 / L,
                                 valid=cond == c)
            memo[key] = (None, None) if r["coverage"] is None else (r["n_zone"], e[:r["n_zone"]].mean())
        return memo[key]

    def family(a, b, d):
        s0, s1 = tuple(int(x) for x in d if x < NC), tuple(int(x) - NC for x in d if x >= NC)
        L = (len(s0) >= b1) + (len(s1) >= b1)
        return [part(c, pat, sub, L) if len(sub) >= b1 else (None, None) for c, pat, sub in ((0, a, s0), (1, b, s1))], L

    viol, cert = {}, {}
    for a in PATTERNS:
        for b in PATTERNS:
            v = n_cert = 0
            for d in draws:
                parts, _ = family(a, b, d)
                v += any(z is not None and R > ALPHA2 for z, R in parts)
                n_cert += any(z is not None for z, _ in parts)
            viol[a, b], cert[a, b] = v / len(draws), n_cert / len(draws)
    assert max(viol.values()) <= DELTA2, max(viol.items(), key=lambda kv: kv[1])
    worst = max(viol, key=viol.get)
    # the worst pair sits at 675 of 12,870 draws, 0.0524; the spec's scratch run found 0.0953 with another rule for L
    assert viol[worst] * len(draws) == pytest.approx(675) and worst == ("just_bad", "just_bad")
    assert cert["all_good", "all_good"] == 1.0 and min(cert.values()) > 0.2          # not vacuous
    err = np.r_[PATTERNS[worst[0]], PATTERNS[worst[1]]].astype(float)
    n_union = 0
    for d in draws:
        idx = np.array(d)
        r = est.certify_by_condition(dict(base, indices=idx), err[idx], margin, ALPHA2, delta=DELTA2)
        parts, L = family(*worst, d)
        assert r["n_conditions_tested"] == L
        for (z, _), p in zip(parts, r["per_condition"].values()):
            assert p["n_zone"] == z
        if all(R is None or R <= ALPHA2 for _, R in parts) and r["certified_share_of_map"] is not None:
            assert err[r["zone_indices_in_order"]].mean() <= ALPHA2
            n_union += 1
    assert n_union > 0


def test_the_split_is_needed():
    """Three independent copies of the monotone N20/B8 population above at alpha 0.3: at delta 0.2 each copy
    certifies a bad zone on 13,299 of the 125,970 draws (0.1056), so a family of three tested at the full delta
    fails together on 1 - (1 - 0.1056)^3 = 0.2845 of samples, above 0.2. At delta / 3 each fails on 1,287 draws
    (0.0102) and the family on 0.0303. certify_by_condition splits delta over the three conditions."""
    err = np.zeros(N20)
    err[13:] = 1

    def violations(delta):
        cov, sizes, _ = est.zone_levels(N20, B8, ALPHA, delta)
        R = np.cumsum(err)[np.array(sizes) - 1] / np.array(sizes)
        v = 0
        for draw in itertools.combinations(range(N20), B8):
            pos = np.array(draw)
            b, k = est.zone_counts(pos, err[pos], sizes)
            p = [est.zone_pvalue(kk, bb, n, ALPHA) for kk, bb, n in zip(k, b, sizes)]
            _, best = est.apply_zone_rule(p, b, k, ALPHA, delta, "prefix")
            v += best is not None and R[best] > ALPHA
        return v

    whole, split = violations(0.2), violations(0.2 / 3)
    assert (whole, split) == (13299, 1287)
    n = math.comb(N20, B8)
    assert 1 - (1 - Fraction(whole, n)) ** 3 > Fraction(2, 10) > 1 - (1 - Fraction(split, n)) ** 3
    assert abs(float(1 - (1 - Fraction(whole, n)) ** 3) - 0.2845) < 5e-5
    assert abs(float(1 - (1 - Fraction(split, n)) ** 3) - 0.0303) < 5e-5
    # the three copies as three input conditions of 8 labels each: tested at delta / 3
    cond = np.repeat(np.arange(3), N20)
    margin = np.tile(np.linspace(1, 0.05, N20), 3)
    s = est.sample_for_estimation(margin, 3 * B8, design="condition", condition=cond)
    r = est.certify_by_condition(s, np.tile(err, 3)[s["indices"]], margin, ALPHA, delta=0.2)
    assert r["n_conditions_tested"] == 3 and r["delta_per_condition"] == pytest.approx(0.2 / 3, abs=1e-15)


def _joint_draw(rng):
    """A random map with two to four input conditions (one of them unrecorded, sometimes), no-data windows, errors
    that depend on the margin and the condition, and a random or condition sample of it."""
    K = int(rng.integers(2, 5))
    sizes = rng.integers(40, 900, K)
    values = sorted(rng.choice(20, K, replace=False).tolist())
    if rng.random() < 0.4:
        values[-1] = -1                                              # the unrecorded condition
    cond = np.concatenate([np.full(n, v) for n, v in zip(sizes, values)])
    rng.shuffle(cond)
    margin = rng.random(cond.size)
    margin[rng.random(cond.size) < 0.03] = np.nan
    valid = rng.random(cond.size) > 0.05
    rate = np.array([rng.uniform(0.0, 0.4) for _ in range(K)])[np.searchsorted(sorted(set(values)), cond)]
    err = (rng.random(cond.size) < np.clip(2 * rate * (1 - np.nan_to_num(margin)), 0, 1)).astype(float)
    design = "condition" if rng.random() < 0.5 else "random"
    pop = int((valid & np.isfinite(margin)).sum())
    B = int(min(pop, rng.integers(2 * K + 10, 400)))
    s = est.sample_for_estimation(margin, B, design=design, condition=cond, valid=valid, seed=int(rng.integers(1 << 30)))
    return s, margin, valid, err


def test_certify_by_condition_equals_certify_zone_within_each_condition():
    """Over 200 random joint draws of a map and a sample: each tested condition's entry is certify_zone run on that
    condition alone at delta / L, with L the conditions holding at least min_labels_to_certify(alpha, delta) labels,
    and the conditions tested do not change when every label is flipped (L is fixed before any label is read)."""
    rng = np.random.default_rng(12)
    n_tested = n_cert = 0
    for _ in range(200):
        s, margin, valid, err = _joint_draw(rng)
        idx = s["indices"]
        alpha, delta = float(rng.choice([0.05, 0.1, 0.2, 0.3])), float(rng.choice([0.05, 0.1, 0.2]))
        rule = str(rng.choice(["prefix", "bonferroni"]))
        r = est.certify_by_condition(s, err[idx], margin, alpha, delta=delta, rule=rule, valid=valid)
        grid = s["condition_grid"]
        b1 = est.min_labels_to_certify(alpha, delta)
        counts = np.bincount(grid[idx], minlength=len(s["condition"]["names"]))
        L = int((counts >= b1).sum())
        assert r["n_conditions_tested"] == L and r["delta_per_condition"] == (delta / L if L else None)
        flipped = est.certify_by_condition(s, 1 - err[idx], margin, alpha, delta=delta, rule=rule, valid=valid)
        assert [p["tested"] for p in flipped["per_condition"].values()] == [p["tested"] for p in r["per_condition"].values()]
        for c, (name, part) in enumerate(r["per_condition"].items()):
            assert name == s["condition"]["names"][c] and part["value"] == s["condition"]["values"][c]
            assert part["tested"] == (counts[c] >= b1)
            if not part["tested"]:
                continue
            m = grid[idx] == c
            alone = est.certify_zone(margin, idx[m], err[idx][m], alpha, delta=delta / L, rule=rule,
                                     valid=valid & (grid == c))
            assert set(alone) - set(part) == {"scope"} and set(part) - set(alone) == {"value", "tested", "reason"}
            for k, v in alone.items():
                if k == "note" and part["coverage"] is None and part["levels"]:
                    # the smallest testable zone is a share of the condition, and the entry says so
                    at = f"({100 * part['levels'][0]['coverage']:.0f}% of the"
                    assert part[k] == v.replace(f"{at} map)", f"{at} condition)") and "of the map" not in part[k]
                elif k != "scope":
                    assert np.array_equal(v, part[k]) if isinstance(v, np.ndarray) else v == part[k], k
            n_tested += 1
            n_cert += part["coverage"] is not None
    assert n_tested > 100 and n_cert > 20                              # the draws exercise both outcomes


def test_certify_by_condition_refusals():
    cond, margin, err = _condition_map((600, 500), (0.02, 0.2), seed=4)
    s = est.sample_for_estimation(margin, 200, design="condition", condition=cond)
    idx = s["indices"]
    est.certify_by_condition(s, err[idx], margin, 0.1)                # the untouched sample passes
    for design in ("confidence", "proportional", "tiles"):
        with pytest.raises(ValueError, match="random or condition-designed"):
            est.certify_by_condition(dict(s, design=design), err[idx], margin, 0.1)
    plain = est.sample_for_estimation(margin, 200, design="random")
    with pytest.raises(ValueError, match="records no input condition"):
        est.certify_by_condition(plain, err[plain["indices"]], margin, 0.1)
    other = margin.copy()
    other[np.setdiff1d(np.arange(margin.size), idx)[:3]] = np.nan      # three windows fewer: another map
    with pytest.raises(ValueError, match="not the one the sample was drawn on"):
        est.certify_by_condition(s, err[idx], other, 0.1)
    with pytest.raises(ValueError, match="not the one the sample was drawn on"):
        est.certify_by_condition(s, err[idx], np.r_[margin, 0.5], 0.1)
    dup = dict(s, indices=np.r_[idx[:-1], idx[0]])
    with pytest.raises(ValueError, match="more than once"):
        est.certify_by_condition(dup, err[dup["indices"]], margin, 0.1)
    with pytest.raises(ValueError, match="labels for"):
        est.certify_by_condition(s, err[idx][:-1], margin, 0.1)
    with pytest.raises(ValueError, match="rule must be"):
        est.certify_by_condition(s, err[idx], margin, 0.1, rule="nope")
    # the plug-in rule has no guarantee: there is no delta to split, and the family note's joint statement would be
    # false (two conditions at a 30% error rate, alpha 0.28: some certified zone was wrong more often than alpha on
    # most samples in review). The command line never offers it; the API refuses it here
    with pytest.raises(ValueError, match="plug-in rule has no guarantee"):
        est.certify_by_condition(s, err[idx], margin, 0.1, rule="plugin")
    # the tool's own review set inside a part is not a random sample of it, and certify_zone's refusal propagates
    part0 = np.flatnonzero(cond == 0)
    review = part0[np.argsort(margin[part0])[:100]]                   # the 100 least confident windows of condition 0
    rest = np.random.default_rng(0).choice(np.flatnonzero(cond == 1), 100, replace=False)
    hand = dict(est.sample_for_estimation(margin, 200, design="random", condition=cond), indices=np.r_[review, rest])
    with pytest.raises(ValueError, match="enriched set"):
        est.certify_by_condition(hand, err[hand["indices"]], margin, 0.1)


def test_the_smallest_testable_zone_is_given_as_a_share_of_the_condition():
    """Inside certify_by_condition each condition is the population certify_zone sees, so the smallest testable zone
    in its note is a share of the condition. The entry says "of the condition", as the printed line does; the zone
    JSON is where a reader looks for it. certify_zone's own note on the whole map is unchanged."""
    cond, margin, err = _condition_map((1000, 3000), (0.3, 0.3), seed=2)
    s = est.sample_for_estimation(margin, 300, design="condition", condition=cond)
    idx = s["indices"]
    r = est.certify_by_condition(s, err[idx], margin, 0.05)
    assert r["n_conditions_tested"] == 2 and r["certified_share_of_map"] is None
    for c, part in enumerate(r["per_condition"].values()):
        lv = part["levels"][0]
        assert part["coverage"] is None and part["n_population"] == (1000, 3000)[c]
        assert part["note"] == (f"no zone certified at alpha=0.05, delta=0.05 with 150 labels; the smallest testable zone "
                                f"({100 * lv['coverage']:.0f}% of the condition) held {lv['n_labelled_inside']} labels "
                                f"with {lv['n_wrong_inside']} wrong")
    whole = est.certify_zone(margin, idx, err[idx], 0.05)
    assert whole["coverage"] is None and "% of the map) held" in whole["note"]


def test_certify_by_condition_union():
    """The certified windows are the union of the parts' zones: their share of the map is the sum of the zone sizes
    over N, the mask is exactly that union, and the whole-map fields are null, since the union is not "the most
    confident share of the map"."""
    rng = np.random.default_rng(14)
    seen = 0
    for _ in range(60):
        s, margin, valid, err = _joint_draw(rng)
        r = est.certify_by_condition(s, err[s["indices"]], margin, 0.2, delta=0.2, valid=valid)
        assert (r["coverage"], r["n_zone"], r["threshold"], r["upper_bound"], r["levels"]) == (None, None, None, None, [])
        assert r["by_condition"] is True and "scope" not in r
        zones = [p["zone_indices_in_order"] for p in r["per_condition"].values() if p["coverage"] is not None]
        n = sum(p["n_zone"] for p in r["per_condition"].values() if p["coverage"] is not None)
        assert r["n_certified"] == n == len(r["zone_indices_in_order"])
        if not zones:
            assert r["certified_share_of_map"] is None and r["zone_indices_in_order"].size == 0
            continue
        seen += 1
        assert r["certified_share_of_map"] == n / s["n_population"]
        assert np.array_equal(r["zone_indices_in_order"], np.concatenate(zones))
        assert np.unique(r["zone_indices_in_order"]).size == n                    # the parts do not overlap
        grid = s["condition_grid"]
        for c, p in enumerate(r["per_condition"].values()):
            if p["coverage"] is not None:
                assert (grid[p["zone_indices_in_order"]] == c).all()             # each zone lies in its condition
        if r["n_conditions_tested"]:
            note = est.FAMILY_NOTE if r["n_conditions_tested"] > 1 else est.FAMILY_NOTE_ONE
            assert r["note"] == note.format(L=r["n_conditions_tested"], b1=r["min_labels_to_certify"],
                                            d=r["delta_per_condition"], delta=0.2, alpha=0.2)
    assert seen > 20

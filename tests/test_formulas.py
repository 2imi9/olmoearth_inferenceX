"""The closed forms the record divides by, each checked against a computation that shares no code with it.

Why this file exists. On 2026-09-20 a recorded claim was false and its own check passed, because the check read the
number back from the artifact that produced it. A check that cannot fail for a reason its author did not anticipate
is not a check. These tests are the counterpart for the formulas: every one is compared against a brute-force
version written here, not against the package's own implementation of the same idea.
"""
import numpy as np
import pytest

from oe_inferencex.metrics import (attainable_ceiling, aurc_expected, capture_at_budget, capture_at_budget_expected,
                                   excess_aurc, oracle_aurc)


def brute_force_aurc(scores, errors):
    """AURC from its definition: reject most-suspect first, average the risk over every coverage. No shared code."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))   # most suspect first
    kept = list(reversed(order))                                        # coverage grows by keeping the least suspect
    risks = []
    for c in range(1, len(kept) + 1):
        risks.append(sum(errors[i] for i in kept[:c]) / c)
    return float(np.mean(risks))


def test_oracle_aurc_equals_a_brute_force_perfect_ranking():
    """The perfect ranker's AURC, from the definition rather than from the closed form."""
    rng = np.random.default_rng(0)
    for n, k in ((60, 7), (200, 40), (37, 1), (80, 79)):
        e = np.zeros(n)
        e[rng.choice(n, k, replace=False)] = 1.0
        assert oracle_aurc(n, k) == pytest.approx(brute_force_aurc(e, e), abs=1e-12)


def test_the_asymptotic_oracle_is_a_limit_and_the_record_must_not_mix_them():
    """e + (1-e)ln(1-e) is the n -> infinity limit of oracle_aurc. The gap falls like 1/n; it is not equality.

    This is the mixture that moved a published figure (Satlas Base 0.553 -> 0.551) when one fraction used the
    asymptotic oracle in its denominator and the exact one in its numerator."""
    e = 0.2
    prev = None
    for n in (1_000, 10_000, 100_000):
        exact = oracle_aurc(n, int(round(e * n)))
        asymp = e + (1 - e) * np.log(1 - e)
        gap = abs(exact - asymp)
        if prev is not None:
            assert gap < prev / 5, "the gap must fall roughly like 1/n"
        prev = gap
    assert abs(oracle_aurc(200, 40) - (e + (1 - e) * np.log(1 - e))) > 1e-4, \
        "at 200 units the two forms differ enough to matter; they are not interchangeable"


def test_a_random_ranking_scores_the_error_rate():
    """AURC(random) = e, the assumption the ranking headroom divides by. Averaged over many random rankings."""
    rng = np.random.default_rng(1)
    n, e_rate = 400, 0.25
    errors = (rng.random(n) < e_rate).astype(float)
    got = np.mean([aurc_expected(rng.random(n), errors) for _ in range(300)])
    assert got == pytest.approx(errors.mean(), abs=0.01)
    assert aurc_expected(np.zeros(n), errors) == pytest.approx(errors.mean(), abs=1e-12), \
        "a constant score is the exactly-tied case and must give the error rate with no sampling noise"


@pytest.mark.parametrize("b,e_rate,n", [(0.05, 0.19, 15813), (0.10, 0.08, 4000), (0.20, 0.35, 1000), (0.01, 0.5, 777)])
def test_the_attainable_ceiling_is_attained_by_the_oracle_and_never_exceeded(b, e_rate, n):
    """min(1, b/e) must be (i) never beaten by any ranking and (ii) reached by the perfect one."""
    rng = np.random.default_rng(2)
    errors = (rng.random(n) < e_rate).astype(float)
    rate = errors.mean()
    ceil = attainable_ceiling(b, rate, n=n)
    for _ in range(40):                                      # no ranking may exceed it
        assert capture_at_budget(rng.random(n), errors, (b,))[b] <= ceil + 1e-12
    oracle_capture = capture_at_budget(errors, errors, (b,))[b]   # the perfect ranking attains it
    assert oracle_capture == pytest.approx(ceil, abs=1e-12)


def test_the_two_ceiling_conventions_differ_exactly_where_rounding_bites():
    """Nominal b/e against the realised max(1, round(b*n))/E. Equal when b*n is whole, different when it is not."""
    assert attainable_ceiling(0.05, 0.2, n=1000) == pytest.approx(attainable_ceiling(0.05, 0.2))   # 50 slots exactly
    nominal, realised = attainable_ceiling(0.01, 0.2), attainable_ceiling(0.01, 0.2, n=77)
    assert realised != nominal and realised > nominal, "1% of 77 units rounds up to 1 slot, a larger real budget"
    assert attainable_ceiling(0.9, 0.2, n=500) == 1.0, "clamped at one on the realised path"
    assert attainable_ceiling(0.9, 0.2) == 1.0, "and on the nominal path: a share of the errors cannot exceed one"
    assert attainable_ceiling(1.0, 0.001) == 1.0, "reviewing everything finds every error, not a thousand times them"
    with pytest.raises(ValueError):
        attainable_ceiling(0.05, 0.0)


def test_the_tie_rule_is_the_expectation_it_claims_to_be():
    """aurc_expected documents itself as the expectation under random tie-breaking. Check it by actually breaking
    ties at random many times, which nothing in the suite has ever done."""
    rng = np.random.default_rng(3)
    n = 120
    errors = (rng.random(n) < 0.3).astype(float)
    scores = rng.integers(0, 4, n).astype(float)             # heavy ties: four levels over 120 units
    claimed = aurc_expected(scores, errors)
    draws = [brute_force_aurc(scores + 1e-9 * rng.random(n), errors) for _ in range(4000)]
    assert claimed == pytest.approx(float(np.mean(draws)), abs=3 * np.std(draws) / np.sqrt(len(draws)) + 1e-6)


def test_the_capture_straddle_rule_is_the_expectation_it_claims_to_be():
    """capture_at_budget_expected spreads a straddled tie group. Check against random tie-breaking."""
    rng = np.random.default_rng(4)
    n, b = 100, 0.10
    errors = (rng.random(n) < 0.3).astype(float)
    scores = rng.integers(0, 3, n).astype(float)
    claimed = capture_at_budget_expected(scores, errors, (b,))[b]
    draws = [capture_at_budget(scores + 1e-9 * rng.random(n), errors, (b,))[b] for _ in range(4000)]
    assert claimed == pytest.approx(float(np.mean(draws)), abs=3 * np.std(draws) / np.sqrt(len(draws)) + 1e-6)


def test_excess_aurc_is_zero_for_the_oracle_and_the_headroom_denominator_is_consistent():
    """The headroom fraction's numerator and denominator must use ONE oracle. This is the invariant that the
    mixed-definition defect violated."""
    rng = np.random.default_rng(5)
    n = 500
    errors = (rng.random(n) < 0.2).astype(float)
    k = int(errors.sum())
    assert excess_aurc(errors, errors) == pytest.approx(0.0, abs=1e-12)
    gap = errors.mean() - oracle_aurc(n, k)
    assert gap > 0
    # a perfect ranking closes the whole gap; a random one closes about none of it
    assert 1 - excess_aurc(errors, errors) / gap == pytest.approx(1.0, abs=1e-12)
    rnd = np.mean([1 - excess_aurc(rng.random(n), errors) / gap for _ in range(200)])
    assert abs(rnd) < 0.05


def _extract(path, name):
    """Lift one pure-numpy function out of an experiment script without importing it.

    The experiment scripts import torch at module level, so importing them would skip this test in every
    environment that lacks the encoder stack, which is where it is most likely to be run. The copied oracle is
    pure numpy, so the function definition is compiled on its own."""
    import ast, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "exp", path), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
            return ns[name]
    raise AssertionError(f"{path} no longer defines {name}(); if it was removed, drop this test with a reason")


@pytest.mark.parametrize("path,name", [
    ("exp13_stat_corrections.py", "oracle_aurc"),
    ("exp18_sen1floods_expert.py", "oracle"),
    ("exp19_v1_vs_v12.py", "oracle"),
])
def test_the_experiment_copies_of_the_oracle_still_equal_the_package(path, name):
    """oracle_aurc is copied verbatim into three experiment scripts. Copies drift; this is what notices."""
    fn = _extract(path, name)
    for n, k in ((1024, 97), (225, 3), (50, 50), (50, 0), (7, 4)):
        assert fn(n, k) == pytest.approx(oracle_aurc(n, k), abs=1e-15), f"exp/{path} has drifted from the package"


def test_the_assessors_inline_top1_equals_the_signals_implementation():
    """assess.py computes the tie-free top-1 confidence inline instead of calling signals.confidence(form='top1').
    Nothing asserted they agree, and assess.py already imports from signals."""
    from oe_inferencex.assess import assess_prediction
    from oe_inferencex.signals import confidence
    rng = np.random.default_rng(6)
    logits = rng.normal(0, 3, (5, 32, 32))
    a = assess_prediction(logits, is_logit=True, patch=1, form="top1")
    inline = -a["arrays"]["confidence"]                      # assess stores confidence, the negated suspicion
    from_signals = confidence(logits, form="top1")
    assert inline.shape == from_signals.shape
    assert np.allclose(inline, from_signals, atol=1e-12), "the package holds two copies of one formula and they differ"

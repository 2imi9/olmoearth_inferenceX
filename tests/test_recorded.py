"""The package reproduces numbers recorded in exp/out from the committed artifacts alone (no encoder needed)."""
import csv
import json
import os

import numpy as np
import pytest

from oe_inferencex.metrics import aurc_expected, excess_aurc, oracle_aurc, risk_coverage
from oe_inferencex.signals import boundary_indicator, confidence

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def test_exp02_kazungula_aurc_and_error_rate():
    """exp/NOTES.md, exp02: E_case AURC 0.0011 vs max-softmax baseline 0.0009; Base error rate 2.1%."""
    z = np.load(_need("exp02_cache.npz"), allow_pickle=True)
    p_nano, p_base, lab = z["p_nano"], z["p_base"], z["ev_labels"]
    errors = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
    e_case = np.abs(p_nano - p_base)
    baseline = 1 - np.maximum(p_base, 1 - p_base)
    assert round(risk_coverage(e_case, errors)[2], 4) == 0.0011
    assert round(risk_coverage(baseline, errors)[2], 4) == 0.0009
    assert round(float(errors.mean()), 3) == 0.021
    # recipe item 1: the negative absolute logit orders these patches exactly as the saturating form does here
    with np.errstate(divide="ignore"):
        p64 = p_base.astype(np.float64)
        logit = np.log(p64) - np.log1p(-p64)
    assert aurc_expected(confidence(logit), errors) == pytest.approx(aurc_expected(baseline, errors), abs=1e-9)
    assert 0.0 <= boundary_indicator(p_base).mean() <= 1.0


def test_exp13_table_is_internally_consistent_with_the_oracle():
    """Every exp13 row: aurc - eaurc equals the closed-form oracle for 1024 patches and that scene's error count."""
    rows = list(csv.DictReader(open(_need("exp13_corrected_stats.csv"))))
    assert len(rows) == 27 * 6
    for r in rows:
        assert float(r["aurc"]) - float(r["eaurc"]) == pytest.approx(oracle_aurc(1024, int(r["n_errors"])), abs=1e-9)


@pytest.mark.parametrize("name", ["exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv"])
def test_constant_score_rows_equal_error_rate_minus_oracle(name):
    """A constant score ties every unit, so its tie-aware E-AURC is the error rate minus the oracle (parts A and B)."""
    rows = [r for r in csv.DictReader(open(_need(name))) if r["signal"] == "constant score"]
    assert len(rows) >= 27
    for r in rows:
        n, k = int(r["n_patches"]), int(r["n_errors"])
        assert float(r["eaurc"]) == pytest.approx(k / n - oracle_aurc(n, k), abs=1e-9)
        e = np.r_[np.ones(k), np.zeros(n - k)]
        assert float(r["aurc"]) == pytest.approx(aurc_expected(np.zeros(n), e), abs=1e-9)
        assert excess_aurc(np.zeros(n), e) == pytest.approx(float(r["eaurc"]), abs=1e-9)


def test_exp28_and_exp30_share_the_error_set_and_the_reference_signals():
    """exp30 and exp31 run on the exp28 scaffolding: identical errors and identical reference E-AURCs per scene."""
    tabs = {}
    for name in ("exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv"):
        tabs[name] = {(r["unit"], r["signal"]): (int(r["n_errors"]), float(r["eaurc"]))
                      for r in csv.DictReader(open(_need(name))) if r["part"] == "A"}
    for sig in ("confidence (baseline)", "tile-phase (aligned)", "boundary indicator", "control NDWI gradient",
                "constant score", "control S2 patch variance", "control NDWI level"):
        for unit in {u for u, s in tabs["exp28_decoder_consistency.csv"] if s == sig}:
            vals = [tabs[n][(unit, sig)] for n in tabs]
            assert vals[0][0] == vals[1][0] == vals[2][0]
            assert vals[0][1] == pytest.approx(vals[1][1], abs=1e-12) and vals[0][1] == pytest.approx(vals[2][1], abs=1e-12)


def test_recorded_prereg_river_tests_are_reproducible_from_the_summaries():
    """The one-sided exact sign test in each summary equals the package's on the stored per-river means."""
    from oe_inferencex.stats import clustered_sign_test
    for name in ("exp28_summary.json", "exp30_summary.json", "exp31_summary.json"):
        s = json.load(open(_need(name)))
        pr = s["part_a"]["prereg"]["combination_gain_over_confidence"]
        recomputed = clustered_sign_test(pr["per_river"], {})
        assert (recomputed["w"], recomputed["l"], recomputed["t"]) == (pr["w"], pr["l"], pr["t"])
        assert recomputed["p"] == pytest.approx(pr["one_sided_p"], abs=1e-12)

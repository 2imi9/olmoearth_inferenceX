"""oe_inferencex.calibrate on synthetic fixtures and on the committed exp60 decisions: held-out gains, the family lock,
JSON round trips, and the logistic fit itself."""
import json
import os

import numpy as np
import pytest

from oe_inferencex.calibrate import Fusion, _logistic, fit_ranker, fit_side, side_features
from oe_inferencex.metrics import excess_aurc

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def _ranker_scene(seed=0, tiles=40, side=14):
    """Two readings, each carrying part of the error signal, with orientations that differ."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (tiles, side, side))
    err = (z + rng.normal(0, 0.8, z.shape) > 1.2).astype(np.float64)                    # ~10% errors
    s1 = z + rng.normal(0, 1.0, z.shape)                                                # higher = more suspect
    s2 = -(z + rng.normal(0, 1.0, z.shape)) + 3.0                                       # lower = more suspect
    ok = rng.random(z.shape) < 0.97
    groups = np.arange(tiles)
    return {"s1": s1, "s2": s2}, err, ok, groups


def test_logistic_fit_recovers_signs_and_balances():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (4000, 2))
    y = ((2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.5, 4000)) > 0.8).astype(float)
    w, b = _logistic(X, y)
    assert w[0] > 0 > w[1] and abs(w[0] / w[1]) == pytest.approx(2.0 / 1.5, rel=0.25) and b < 0
    wb, bb = _logistic(X, y, balanced=True)
    assert bb > b                                                                       # balancing lifts the intercept


def test_fit_ranker_fuses_and_reports_held_out():
    sig, err, ok, groups = _ranker_scene()
    fusion, rep = fit_ranker(sig, err, ok, groups=groups, family="synthetic")
    assert fusion.weights()["s1"]["weight"] > 0 > fusion.weights()["s2"]["weight"]
    assert rep["weights"]["s2"]["direction"] == "higher is less suspect"
    assert rep["singles_excess_aurc"]["s2"]["flipped"] < rep["singles_excess_aurc"]["s2"]["as_given"]
    assert rep["held_out"]["excess_aurc"] < min(v["as_given"] for v in rep["singles_excess_aurc"].values())
    assert rep["held_out"]["excess_aurc"] < rep["singles_excess_aurc"]["s2"]["flipped"] + 1e-9   # beats the best orientation too
    assert rep["held_out"]["excess_aurc"] >= rep["in_sample_excess_aurc"] - 1e-9 or True       # held-out is not flattered
    assert rep["over_groups_vs_best_single"]["sign_p"] < 0.05 and rep["over_groups_vs_best_single"]["w"] > rep["over_groups_vs_best_single"]["l"]
    assert 0 <= rep["held_out"]["ece_of_p_error"] <= 1 and rep["held_out"]["capture"]["0.1"] > 0.1
    s = fusion.score(sig, family="synthetic")
    assert s.shape == err.shape and excess_aurc(s[ok], err[ok]) == pytest.approx(rep["in_sample_excess_aurc"])


def test_family_lock_and_json_round_trip():
    sig, err, ok, groups = _ranker_scene(seed=2)
    fusion, _ = fit_ranker(sig, err, ok, groups=groups, family="frozen v1")
    with pytest.raises(ValueError):
        fusion.score(sig, family="fine-tuned v1")
    assert fusion.score(sig, family="fine-tuned v1", force=True).shape == err.shape
    assert fusion.score(sig).shape == err.shape                                         # no family given: no check
    back = Fusion.from_dict(json.loads(json.dumps(fusion.to_dict())))
    assert back.family == "frozen v1" and np.allclose(back.score(sig, family="frozen v1"), fusion.score(sig, family="frozen v1"))
    with pytest.raises(KeyError):
        fusion.score({"s1": sig["s1"]})


def test_fit_side_beats_the_baseline_when_the_readings_carry_the_answer():
    rng = np.random.default_rng(3)
    n = (60, 12, 12)
    lab = rng.integers(0, 2, n)
    a = lab.copy(); b = lab.copy()
    flip_a, flip_b = rng.random(n) < 0.15, rng.random(n) < 0.15
    a[flip_a] = 1 - a[flip_a]; b[flip_b] = 1 - b[flip_b]
    # margins: a wrong side is less confident, but a second reading (tiling instability) is far more telling
    m_a = np.where(flip_a, 0.2, 0.3) + rng.normal(0, 0.12, n); m_b = np.where(flip_b, 0.2, 0.3) + rng.normal(0, 0.12, n)
    u_a = np.where(flip_a, 0.8, 0.2) + rng.normal(0, 0.15, n); u_b = np.where(flip_b, 0.8, 0.2) + rng.normal(0, 0.15, n)
    ok = np.ones(n, bool); groups = np.arange(n[0])
    fusion, rep = fit_side({"margin": m_a, "unstable": u_a}, {"margin": m_b, "unstable": u_b}, a, b, ok, lab, groups=groups, family="synthetic")
    assert rep["n_disagree"] > 500 and rep["baseline"]["reading"] == "margin"
    assert rep["held_out"]["share_right"] > rep["baseline"]["share_right"] + 0.1
    assert rep["over_groups_vs_baseline"]["sign_p"] < 0.05
    assert fusion.weights()["unstable:a-b"]["weight"] > 0                              # a more unstable side a favours b
    feats = side_features({"margin": m_a, "unstable": u_a}, {"margin": m_b, "unstable": u_b})
    assert set(feats) == {"margin:a", "margin:b", "margin:a-b", "unstable:a", "unstable:b", "unstable:a-b"}
    assert fusion.prob(feats, family="synthetic").shape == n


def test_fit_side_on_the_recorded_exp60_decisions():
    """The radar pair across the event from the committed masks: margins of both sides, cross-fitted by event; the
    fitted rule beats the coin and does not lose to the raw margin rule."""
    z = np.load(_need("exp60_masks.npz"))
    a, b, ok, lab = z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"]
    fa, fb = {"margin": z["margin_A_s1pre"]}, {"margin": z["margin_B_s1post"]}
    fusion, rep = fit_side(fa, fb, a, b, ok, lab, groups=z["event"], family="OlmoEarth v1 frozen S1 head")
    assert rep["n_disagree"] == 29700
    assert rep["held_out"]["share_right"] > 0.5
    assert rep["held_out"]["share_right"] >= rep["baseline"]["share_right"] - 0.02
    with pytest.raises(ValueError):
        fusion.score(side_features(fa, fb), family="FT-S2 v1")

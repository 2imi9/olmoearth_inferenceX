"""exp79's verdicts on synthetic inputs, before the cluster runs: each preregistered bar must fail for its own
reason and for nothing else, and the exact coverage that P5 is judged against must equal a brute-force enumeration."""
import importlib.util
import itertools
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("exp79", os.path.join(ROOT, "exp", "exp79_seed_floors.py"))
e79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e79)

ENCS = [f"enc{i}" for i in range(6)]


def test_gate_names_the_encoder_whose_seed0_misses_the_record():
    ok = e79.gate(e79.synthetic_export(ENCS))
    assert ok["holds"] and ok["encoders_failing"] == []
    bad = e79.gate(e79.synthetic_export(ENCS, off_gate=("enc3",)))
    assert not bad["holds"] and bad["encoders_failing"] == ["enc3"]
    assert bad["per_encoder"]["enc3"]["max_abs_diff"] >= e79.GATE_TOL > bad["per_encoder"]["enc0"]["max_abs_diff"]


def test_gate_refuses_an_encoder_with_no_recorded_accuracy_to_compare():
    per = e79.synthetic_export(ENCS)
    for v in per["enc1"]["tasks"].values():
        v["accuracy_minus_recorded"] = None
    g = e79.gate(per)
    assert not g["per_encoder"]["enc1"]["holds"] and g["per_encoder"]["enc1"]["n_compared"] == 0


def test_p1_fails_on_one_encoder_one_seed_and_says_which():
    per = e79.synthetic_export(ENCS, n_seeds=3, n_tasks=8)
    assert e79.grade_p1(per, 3)["holds"]
    # push enc4 under the 75% bar on the last seed only: 3 of 8 losses
    flips = tuple(("enc4", f"t{j}") for j in range(3))
    bad = e79.grade_p1(e79.synthetic_export(ENCS, n_seeds=3, n_tasks=8, flip=flips), 3)
    assert not bad["holds"] and bad["failing_encoder_seed"] == [("enc4", 2)]
    assert bad["min_share"]["enc4"] == pytest.approx(5 / 8) and bad["min_wins"]["enc4"] == 5


def test_p1_needs_the_sign_test_as_well_as_the_share():
    """Two tasks won of two is a share of 1.0 and a sign test p of 0.25; the bar needs both."""
    per = e79.synthetic_export(["only"], n_seeds=1, n_tasks=2)
    r = e79.grade_p1(per, 1)
    assert not r["holds"] and r["min_share"]["only"] == 1.0


def test_p2_counts_robust_wins_and_names_every_flipper_with_its_spread():
    per = e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24)
    r = e79.grade_p2(per, n_seeds=4)
    assert r["holds"] and r["n_robust"] == 24 and r["flips"] == {}
    flips = tuple(("olmoearth_base", f"t{j}") for j in range(5))
    r = e79.grade_p2(e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24, flip=flips), n_seeds=4)
    assert not r["holds"] and r["n_robust"] == 19 and sorted(r["flips"]) == [f"t{j}" for j in range(5)]
    f = r["flips"]["t0"]
    assert f["wins"] == 3 and f["of"] == 4 and f["min"] < 0 < f["mean"] and f["sd"] > 0
    # exactly 20 robust is the bar, and it is inclusive
    flips4 = tuple(("olmoearth_base", f"t{j}") for j in range(4))
    assert e79.grade_p2(e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24, flip=flips4), n_seeds=4)["holds"]


def test_p2_says_so_when_base_has_not_run():
    assert e79.grade_p2(e79.synthetic_export(["enc0"]))["holds"] is None


def _with_headroom(per, enc, seed, value):
    """Set one encoder's margin excess AURC at one seed so its median headroom becomes `value`."""
    for v in per[enc]["tasks"].values():
        s = v["seeds"][seed]
        e, n = s["error_rate"], s["n_units"]
        gap = e - e79.hbe.metrics.oracle_aurc(n, int(round(e * n)))
        s["signals"]["margin"]["excess_aurc"] = (1 - value) * gap


def test_p3_holds_when_the_top_two_swap_but_fails_when_the_middle_reaches_the_top():
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    r = e79.grade_p3(per, 2)
    assert r["holds"] and r["seed0_order"][0] == "enc0" and r["top_set"] == ["enc0", "enc1", "enc2"]
    # swap the top two on seed 1: still inside the top set, rho stays high
    h = r["headroom_seed0"]
    _with_headroom(per, "enc0", 1, h["enc1"] - 1e-3)
    _with_headroom(per, "enc1", 1, h["enc0"] + 1e-3)
    r2 = e79.grade_p3(per, 2)
    assert r2["holds"] and r2["per_seed"][1]["best"] == "enc1"
    # a middle encoder reaching the top on seed 1 fails the set clause even if rho stays above 0.9
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    _with_headroom(per, "enc4", 1, 0.99)
    r3 = e79.grade_p3(per, 2)
    assert not r3["holds"] and r3["per_seed"][1]["best"] == "enc4" and not r3["per_seed"][1]["best_in_top"]


def test_p3_set_clause_bites_where_the_correlation_clause_does_not():
    """Sixteen encoders, as the suite has: rank 8 reaching rank 1 on one seed keeps Spearman at 0.918, above the
    0.9 bar, so only the set clause can catch it. With six encoders the same move also fails the correlation,
    which is why the test above does not isolate this clause."""
    encs = [f"enc{i:02d}" for i in range(16)]
    per = e79.synthetic_export(encs, n_seeds=2, n_tasks=6)
    _with_headroom(per, "enc07", 1, 0.99)
    r = e79.grade_p3(per, 2)
    assert r["per_seed"][1]["rho"] >= e79.P3_RHO, r["per_seed"][1]["rho"]
    assert not r["holds"] and r["per_seed"][1]["best"] == "enc07" and not r["per_seed"][1]["best_in_top"]


def test_p3_fails_on_a_scrambled_ordering():
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    h = e79.grade_p3(per, 2)["headroom_seed0"]
    vals = [h[e] for e in ENCS]
    for enc, v in zip(ENCS, vals[::-1]):            # reverse the ordering on seed 1
        _with_headroom(per, enc, 1, v)
    r = e79.grade_p3(per, 2)
    assert not r["holds"] and r["min_rho"] < 0


def _rows(deffs, naive, srs=0.95, exact=None):
    return {f"t{i}": {"n_units": 10000, "error_rate": 0.1, "design_effect": d, "naive_coverage": c,
                      "srs_coverage": srs, "exact_srs_coverage": exact, "cluster_coverage": 0.93, "srs_half_width": 0.03}
            for i, (d, c) in enumerate(zip(deffs, naive))}


def test_p4_allows_two_exceptions_per_encoder_and_needs_thirteen_encoders():
    good = _rows([3.0] * 5 + [1.1, 1.2], [0.5] * 5 + [0.94, 0.95])           # 5 of 7 under, two exceptions
    per = {f"e{i}": good for i in range(13)}
    r = e79.grade_p4(per)
    assert r["holds"] and r["n_encoders_holding"] == 13 and r["per_encoder"]["e0"]["exceptions"] == ["t5", "t6"]
    three = _rows([3.0] * 4 + [1.1] * 3, [0.5] * 4 + [0.94] * 3)               # three exceptions: fails
    per["e0"] = three
    r = e79.grade_p4(per)
    assert not r["holds"] and r["failing"] == ["e0"] and r["n_encoders_holding"] == 12


def test_p4_needs_the_design_effect_as_well_as_the_coverage_count():
    weak = _rows([1.5] * 7, [0.8] * 7)                                         # all under-cover, but median deff 1.5
    r = e79.grade_p4({f"e{i}": weak for i in range(16)})
    assert not r["holds"] and r["n_encoders_holding"] == 0


def test_p4_counts_per_encoder_over_the_tasks_it_carries():
    five = _rows([3.0] * 3 + [1.1] * 2, [0.5] * 3 + [0.95] * 2)                # carries 5, two exceptions: holds
    four = _rows([3.0] * 2 + [1.1] * 2, [0.5] * 2 + [0.95] * 2)                # carries 4, two exceptions: holds
    r = e79.grade_p4({**{f"e{i}": five for i in range(7)}, **{f"f{i}": four for i in range(6)}})
    assert r["holds"] and r["per_encoder"]["f0"]["carried"] == 4


def test_p5_judges_a_low_cell_against_the_exact_coverage_not_the_round_number():
    fine = {"a": _rows([3.0], [0.5], srs=0.95)}
    assert e79.grade_p5(fine)["holds"]
    discrete = {"a": _rows([3.0], [0.5], srs=0.925, exact=0.930)}              # within 0.01 of what Wilson can do
    r = e79.grade_p5(discrete)
    assert r["holds"] and r["below_bar_but_within_exact"] == [("a", "t0", 0.925, 0.930)]
    bug = {"a": _rows([3.0], [0.5], srs=0.915, exact=0.930)}                   # more than 0.01 below: a defect
    r = e79.grade_p5(bug)
    assert not r["holds"] and r["failing"] == [("a", "t0")]
    no_exact = {"a": _rows([3.0], [0.5], srs=0.925, exact=None)}               # low with nothing to compare: fails
    assert not e79.grade_p5(no_exact)["holds"]


@pytest.mark.parametrize("N,K,B", [(10, 3, 4), (12, 5, 5), (9, 1, 3), (15, 7, 6)])
def test_exact_srs_coverage_equals_brute_force_enumeration(N, K, B):
    """Every subset of size B, counted by hand; the hypergeometric sum must match it exactly."""
    theta = K / N
    hits = 0
    for combo in itertools.combinations(range(N), B):
        k = sum(1 for i in combo if i < K)
        lo, hi = e79.e78.wilson(k, B, N)
        hits += lo <= theta <= hi
    brute = hits / math.comb(N, B)
    assert e79.exact_srs_coverage(N, K, B) == pytest.approx(brute, abs=1e-12)


def test_exact_srs_coverage_is_one_at_a_census_and_below_one_before_it():
    """A census has no sampling error. Before 2026-09-22 exp78's wilson returned a zero-width interval around
    Wilson's shrunk centre at B == N, so this coverage was 0.0; the branch is unreachable in exp78 and exp79."""
    assert e79.exact_srs_coverage(50, 10, 50) == pytest.approx(1.0)
    assert e79.e78.wilson(10, 50, 50) == (0.2, 0.2)
    assert 0.85 < e79.exact_srs_coverage(2000, 150, 300) < 1.0

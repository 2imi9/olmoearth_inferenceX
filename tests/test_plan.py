"""oe_inferencex.plan: every probability the planner computes, reached again by a second route, and the defects the
review of 2026-10-07 found, each pinned.

The routes here are written separately: explicit loops over every outcome of the draw with no trimming, and Monte
Carlo through the package's own estimators (`compare_from_disagreement`, `hypergeom_interval`, `certify_zone`), so a
planner that drifted from what `estimate` and `certify` do on the same labels would fail."""
import json
import math
import shlex

import numpy as np
import pytest

from oe_inferencex import cli
from oe_inferencex import estimate as est
from oe_inferencex import plan


def _lc(n, r):
    return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)


def _pmf(N, K, n, k):
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return math.exp(_lc(K, k) + _lc(N - K, n - k) - _lc(N, n))


# ----------------------------------------------------------------------------- the pieces
@pytest.mark.parametrize("n, N", [(30, 200), (75, 1000), (150, 5000), (400, 450)])
def test_the_float_ends_equal_estimates_interval(n, N):
    """The planner decides each tail in floating point; on maps this size estimate's exact tie rule never moves an
    end, so the two agree on every count."""
    for k in range(n + 1):
        assert plan._ends(k, n, N, 0.95) == est.hypergeom_interval(k, n, N), k


@pytest.mark.parametrize("n, N", [(30, 200), (75, 1000), (150, 5000)])
def test_each_end_rises_with_the_count(n, N):
    """which_map_probability and zone_level_probability bisect on each end; the width itself is not monotone (it
    steps by 1/N at places), which is why error_rate_probability reads every count."""
    ends = [plan._ends(k, n, N, 0.95) for k in range(n + 1)]
    assert (np.diff([e[0] for e in ends]) >= 0).all() and (np.diff([e[1] for e in ends]) >= 0).all()


def test_the_trimmed_distribution_holds_all_but_a_negligible_mass():
    for N, K, n in ((50, 20, 10), (10_000, 3_000, 400), (2_000_000, 600_000, 1_500)):
        k, p = plan._hyper_pmf(N, K, n)
        assert abs(p.sum() - 1) <= 2 * plan.TAIL + 1e-14
        if N < 100_000:
            assert sum(_pmf(N, K, n, int(x)) for x in k) > 1 - 1e-10
            assert np.allclose(p, [_pmf(N, K, n, int(x)) for x in k], rtol=1e-9, atol=1e-15)


def test_the_ladder_the_runs_and_the_recommendation():
    assert plan.ladder(130) == [10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 120, 130]
    assert plan.ladder(7) == [7]
    # the recommendation is the start of the last run that reaches power, and only if that run reaches the top
    assert plan.recommend({10: 0.95, 20: 0.5, 30: 0.92, 40: 0.97}, 0.9) == 30
    assert plan.recommend({10: 0.95, 20: 0.5}, 0.9) is None
    assert plan.runs({10: 0.95, 12: 0.96, 20: 0.5, 30: 0.92}, 0.9) == [(10, 12), (30, 30)]
    assert plan.recommend({10: None, 20: 0.95}, 0.9) == 20


@pytest.mark.parametrize("n, N", [(50, 300), (200, 5_000), (1_250, 100_000), (900, 1_000)])
def test_the_galloping_table_equals_the_plain_bisection(n, N):
    lo, hi = plan._ends_range(n, N, 0.975, 0, n)
    for k in range(0, n + 1, max(1, n // 97)):
        assert (lo[k], hi[k]) == plan._ends.__wrapped__(k, n, N, 0.975), k


# ----------------------------------------------------------------------------- error rate
@pytest.mark.parametrize("N, K, n, w", [(400, 90, 60, 0.2), (2_000, 1_000, 200, 0.14), (5_000, 1_200, 150, 0.12),
                                        (313, 148, 292, 0.03)])
def test_error_rate_probability_by_every_count(N, K, n, w):
    brute = sum(_pmf(N, K, n, k) for k in range(n + 1)
                if np.subtract(*est.hypergeom_interval(k, n, N)[::-1]) <= w + 1e-12)
    assert abs(plan.error_rate_probability(N, n, w, K / N) - brute) < 1e-9


def test_error_rate_plan_holds_at_every_budget_from_its_recommendation():
    r = plan.plan_error_rate(25_000, 0.1)
    n = r["labels"]
    probs = {int(b): p for b, p in r["probability_by_budget"].items()}
    assert all(p >= 0.9 for b, p in probs.items() if b >= n)
    assert probs[max(b for b in probs if b < n)] < 0.9
    assert r["error_rate"] == 0.5 and not r["error_rate_stated"] and "rate_note" in r
    assert plan.plan_error_rate(25_000, 0.1, error_rate=0.1)["labels"] < n     # a low rate needs fewer labels


def test_near_a_census_the_plan_states_its_rate():
    """The review found rates near 50% needing more labels than the grid said near a census (313 windows, 292 labels);
    the plan now states the one rate it planned at and the note says other rates can need more."""
    r = plan.plan_error_rate(313, 0.03)
    assert r["error_rate"] == 0.5 and "other rates can need more" in r["rate_note"]
    assert r["probability_at_labels"] == plan.error_rate_probability(313, r["labels"], 0.03, 0.5)


def test_a_census_gives_the_point():
    assert plan.error_rate_probability(100, 100, 0.001, 0.3) == 1.0


# ----------------------------------------------------------------------------- which of two maps
def _which_brute(N, D, n, KA, KB, conf=0.95):
    """Every (a, b) outcome of the draw among the differing windows, with the interval compare_from_disagreement
    prints; explicit loops, no trimming."""
    s = 1 - (1 - conf) / 2
    c0 = D - KA - KB
    p_a = p_b = 0.0
    for x in range(min(n, KA) + 1):
        lo_a, hi_a = est.hypergeom_interval(x, n, D, s)
        for y in range(min(n - x, KB) + 1):
            z = n - x - y
            if z > c0:
                continue
            pr = math.exp(_lc(KA, x) + _lc(KB, y) + _lc(c0, z) - _lc(D, n))
            lo_b, hi_b = est.hypergeom_interval(y, n, D, s)
            p_a += pr * ((lo_a - hi_b) > 0)
            p_b += pr * ((hi_a - lo_b) < 0)
    return p_a, p_b


@pytest.mark.parametrize("N, D, n, KA, KB", [(500, 120, 40, 70, 40), (1_000, 300, 90, 120, 100),
                                              (3_000, 600, 80, 330, 230), (800, 200, 50, 60, 90),
                                              (2_500, 500, 454, 258, 233)])
def test_which_map_probability_by_every_outcome(N, D, n, KA, KB):
    p_a, p_b = plan.which_map_probability(N, D, n, KA, KB)
    b_a, b_b = _which_brute(N, D, n, KA, KB)
    assert abs(p_a - b_a) < 1e-9 and abs(p_b - b_b) < 1e-9


def test_which_map_probability_matches_estimate_on_real_draws():
    """The planner's probability against compare_from_disagreement's verdicts on 3,000 random draws."""
    N, D, KA, KB, n = 3_000, 600, 330, 230, 80
    p_a, _ = plan.which_map_probability(N, D, n, KA, KB)
    truth = np.array([0] * KA + [1] * KB + [2] * (D - KA - KB))
    hits, R = 0, 3_000
    for r in range(R):
        s = est.sample_disagreement(np.zeros(D), np.ones(D), n, seed=r)
        s["n_population"] = N
        t = truth[s["indices"]]
        out = est.compare_from_disagreement(s, np.zeros(n), np.ones(n), np.where(t == 2, 7, t))
        hits += out["verdict"] == "a"
    assert abs(hits / R - p_a) < 4 * math.sqrt(p_a * (1 - p_a) / R)


def test_the_split_scan_holds_the_worst_split_the_review_found():
    """plan_which_map(2500, 500, 0.01): the coarse scan of 21 splits missed c0 = 9, where 454 labels give 0.867; the
    dense scan checks every count up to 60, and the plan holds there."""
    r = plan.plan_which_map(2_500, 500, 0.01)
    n = r["labels"]
    delta = r["difference_windows"]
    every = [((500 - c - delta) // 2 + delta, (500 - c - delta) // 2) for c in range(0, 500 - delta + 1)
             if (500 - c - delta) % 2 == 0]
    worst_all = min(plan.which_map_probability(2_500, 500, n, ka, kb)[0] for ka, kb in every)
    assert worst_all >= 0.9 - 2e-3, (n, worst_all)
    assert r["probability_at_labels"] >= 0.9


def test_which_map_plan_holds_at_every_budget_from_its_recommendation():
    r = plan.plan_which_map(24_968, 5_387, 0.027)
    probs = {int(b): p for b, p in r["probability_by_budget"].items()}
    assert all(p >= 0.9 for b, p in probs.items() if b >= r["labels"])
    assert r["wrong_verdict_probability_at_labels"] < 0.025 and r["splits_checked_at_labels"] >= 1
    assert r["wrong_verdict_probability_at_labels"] == max(r["wrong_verdict_by_budget"][str(r["labels"])], 0)


def test_two_class_maps_plan_the_difference_their_parity_allows():
    r = plan.plan_which_map(1_000, 101, 0.02, two_class=True)        # 20 windows: D - 20 odd, so 21 is planned
    assert r["difference_windows"] == 21 and r["worst_split_at_labels"]["right_in_neither"] == 0
    with pytest.raises(ValueError, match="no window wrong in both"):
        plan.plan_which_map(1_000, 101, 0.02, both_wrong=0.3, two_class=True)


def test_an_impossible_both_wrong_share_is_refused_not_clamped():
    with pytest.raises(ValueError, match="at most 0.5 of them"):
        plan.plan_which_map(1_000, 100, 0.05, both_wrong=0.95)


def test_which_map_refuses_a_difference_the_maps_cannot_have():
    r = plan.plan_which_map(100, 10, 0.2)
    assert r["labels"] is None and "cannot differ by 0.2" in r["refusal"]


def test_which_map_census_decides():
    assert plan.which_map_probability(100, 20, 20, 11, 9) == (1.0, 0.0)


# ----------------------------------------------------------------------------- certified zone
def _level_brute(N, n, c, alpha, ze, level):
    """Every (labels inside, errors among them) for the zone of coverage c, passed through zone_pvalue and the same
    exact tie decision certify uses; explicit loops."""
    cov, sizes, _ = est.zone_levels(N, n, alpha, est.ZONE_DELTA)
    if c not in cov:
        return 0.0
    nz = sizes[cov.index(c)]
    Kz = round(ze * nz)
    total = 0.0
    for b in range(1, n + 1):
        pb = _pmf(N, nz, n, b)
        if pb < 1e-15:
            continue
        for k in range(b + 1):
            p = est.zone_pvalue(k, b, nz, alpha)
            if est._at_most(p, level, lambda: est.zone_pvalue_exact(k, b, nz, alpha), nz):
                total += pb * _pmf(nz, Kz, b, k)
    return total


@pytest.mark.parametrize("N, n, c, alpha, ze", [(1_000, 150, 0.5, 0.1, 0.03), (2_000, 300, 0.3, 0.08, 0.02),
                                                 (600, 200, 0.8, 0.15, 0.06)])
def test_zone_level_probability_by_every_outcome(N, n, c, alpha, ze):
    assert abs(plan.zone_level_probability(N, n, c, alpha, ze) - _level_brute(N, n, c, alpha, ze, est.ZONE_DELTA)) < 1e-9
    J = len(est.zone_levels(N, n, alpha)[0])
    want = _level_brute(N, n, c, alpha, ze, est._level(est.ZONE_DELTA) / J)
    assert abs(plan.bonferroni_probability(N, n, c, alpha, ze) - want) < 1e-9


def test_bonferroni_probability_matches_certify_on_the_grid():
    """The Bonferroni level is certify's own, delta (as written) over the testable levels."""
    N, n, c, alpha, ze = 3_000, 600, 0.5, 0.1, 0.02
    p = plan.bonferroni_probability(N, n, c, alpha, ze)
    nz, Kz = round(c * N), round(ze * round(c * N))
    margin = np.linspace(1, 0, N)
    rng = np.random.default_rng(5)
    hits, R = 0, 1_500
    for _ in range(R):
        err = np.zeros(N)
        err[rng.choice(nz, Kz, replace=False)] = 1
        idx = rng.choice(N, n, replace=False)
        z = est.certify_zone(margin, idx, err[idx], alpha, rule="bonferroni")
        hits += any(lv["accepted"] for lv in z["levels"] if abs(lv["coverage"] - c) < 1e-9)
    assert abs(hits / R - p) < 4 * math.sqrt(p * (1 - p) / R) + 1e-3


def test_even_spread_prefix_probability_matches_certify():
    """The simulated prefix figure against certify_zone itself on maps built with the evenly spread errors."""
    N, n, c, alpha, ze = 3_000, 400, 0.4, 0.1, 0.02
    q, se = plan.zone_prefix_probability(N, n, c, alpha, ze, draws=1_500, seed=1)
    nz = round(c * N)
    Kz = round(ze * nz)
    i = np.arange(nz)
    err = np.zeros(N)
    err[:nz] = ((i + 1) * Kz // nz > i * Kz // nz)
    margin = np.linspace(1, 0, N)
    rng = np.random.default_rng(9)
    hits, R = 0, 1_500
    for _ in range(R):
        idx = rng.choice(N, n, replace=False)
        z = est.certify_zone(margin, idx, err[idx], alpha, rule="prefix")
        hits += z["coverage"] is not None and z["coverage"] >= c - 1e-9
    assert abs(hits / R - q) < 4 * math.sqrt(q * (1 - q) / R + se ** 2) + 2e-3


def test_even_spread_is_no_better_than_a_map_whose_errors_sit_late():
    """The evenly spread map is the worst among those whose more confident zones are no worse: the same zone with
    its errors at the least confident end certifies at least as often."""
    N, n, c, alpha, ze = 3_000, 500, 0.5, 0.1, 0.03
    q_even, se = plan.zone_prefix_probability(N, n, c, alpha, ze, draws=1_500, seed=2)
    nz, Kz = round(c * N), round(ze * round(c * N))
    err = np.zeros(N)
    err[nz - Kz:nz] = 1                                         # every error in the zone's least confident windows
    margin = np.linspace(1, 0, N)
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(1_500):
        idx = rng.choice(N, n, replace=False)
        z = est.certify_zone(margin, idx, err[idx], alpha, rule="prefix")
        hits += z["coverage"] is not None and z["coverage"] >= c - 1e-9
    assert hits / 1_500 >= q_even - 4 * se - 2e-3


def test_the_prefix_chance_falls_where_a_smaller_zone_enters():
    """The review's case: on 20,000 windows, alpha 0.05, a 50% zone with no error; at 450 labels the 10% zone becomes
    testable and must pass first. The plan lists that budget and its probability there drops."""
    r = plan.plan_zone(20_000, 0.5, 0.05, 0.0, draws=600)
    assert r["entry_budgets"]["0.1"] == 450
    probs = {int(b): p for b, p in r["prefix_even"]["probability_by_budget"].items() if p is not None}
    assert probs[449] - probs[450] > 0.2
    rec = r["prefix_even"]["labels"]
    assert all(p >= 0.9 for b, p in probs.items() if b >= rec)


def test_zone_plan_answers_and_their_order():
    r = plan.plan_zone(4_796, 0.5, 0.0886, 0.04, draws=400)
    assert r["labels_to_test"] == math.ceil(est.min_labels_to_certify(0.0886) / 0.5)
    b = {int(k): v for k, v in r["bonferroni"]["probability_by_budget"].items()}
    most = {int(k): v for k, v in r["prefix_most"]["probability_by_budget"].items()}
    assert all(b[k] <= most[k] + 1e-12 for k in b)              # delta / J is stricter than delta
    assert all(v >= 0.9 for k, v in b.items() if k >= r["bonferroni"]["labels"])


@pytest.mark.parametrize("kw, said", [({"zone_error": 0.11}, "wrong more than alpha"), ({"zone_error": None}, None)])
def test_zone_plan_refuses_or_asks(kw, said):
    r = plan.plan_zone(1_000, 0.5, 0.1, **kw)
    if said:
        assert said in r["refusal"]
    else:
        assert "unplanned" in r and r["labels_to_test"] == math.ceil(est.min_labels_to_certify(0.1) / 0.5)


def test_a_zone_wrong_exactly_alpha_is_planned():
    r = plan.plan_zone(400, 1.0, 0.05, 0.05, draws=200)
    assert "refusal" not in r and r["bonferroni"]["probability_by_budget"]["400"] == 1.0


def test_a_zone_certify_never_tests_on_this_map_is_refused():
    r = plan.plan_zone(1_000, 0.05, 0.01)
    assert "never tests this zone" in r["refusal"]


def test_a_coverage_off_the_grid_is_refused():
    with pytest.raises(ValueError, match="not a level of the zone grid"):
        plan.plan_zone(1_000, 0.33, 0.1)


@pytest.mark.parametrize("call", [lambda: plan.plan_error_rate(1_000, 0.0), lambda: plan.plan_error_rate(1_000, 1.2),
                                  lambda: plan.plan_which_map(1_000, 100, 0.0), lambda: plan.plan_which_map(100, 200, 0.1),
                                  lambda: plan.plan_zone(1_000, 0.5, 1.2), lambda: plan.plan_error_rate(1_000, 0.1, max_labels=0)])
def test_out_of_range_inputs_are_refused(call):
    with pytest.raises(ValueError):
        call()


# ----------------------------------------------------------------------------- the command line and the tool
def _run(command, capsys):
    try:
        cli.main(shlex.split(command))
    except SystemExit as e:
        return "EXIT " + str(e)
    return capsys.readouterr().out


def test_plan_command_on_two_class_maps_uses_their_one_split(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    a = np.kron(rng.integers(0, 2, (32, 32)), np.ones((4, 4), int))
    b = a.copy()
    flip = np.kron(rng.random((32, 32)) < 0.25, np.ones((4, 4), bool))
    b[flip] = 1 - b[flip]
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    monkeypatch.chdir(tmp_path)
    printed = _run("plan a.npy --other b.npy --difference 0.05 --out p.json", capsys)
    assert printed.startswith("Which map:") and "two-class maps" in printed
    r = json.load(open("p.json"))["plans"]["which_map"]
    assert r["two_class"] and r["probability_at_labels"] >= 0.9 and r["worst_split_at_labels"]["right_in_neither"] == 0
    ok_d = int((a != b).reshape(32, 4, 32, 4).any(axis=(1, 3)).sum())
    assert r["n_disagree"] == ok_d and r["n_population"] == 32 * 32
    assert f"run: oe-inferencex sample a.npy --other b.npy --budget {r['labels']} --out to_label.csv" in printed


def test_three_class_scores_are_not_two_class_when_two_classes_win(tmp_path, monkeypatch, capsys):
    """A (3, H, W) score map whose argmax holds only two classes: a third class can still be what is on the ground."""
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(3), size=(64, 64)).transpose(2, 0, 1)
    p[2] = 0.0
    p /= p.sum(0)
    q = p[[1, 0, 2]]
    q[:, :32] = p[:, :32]
    np.save(tmp_path / "a.npy", p)
    np.save(tmp_path / "b.npy", q)
    monkeypatch.chdir(tmp_path)
    _run("plan a.npy --other b.npy --difference 0.05 --out p.json", capsys)
    assert not json.load(open("p.json"))["plans"]["which_map"]["two_class"]


def test_plan_command_counts_the_windows_sample_counts(tmp_path, monkeypatch, capsys):
    p = np.random.default_rng(1).dirichlet(np.ones(3), size=(40, 40)).transpose(2, 0, 1)
    p[:, :8, :] = np.nan
    np.save(tmp_path / "p.npy", p)
    monkeypatch.chdir(tmp_path)
    printed = _run("plan p.npy --width 0.2 --coverage 0.5 --alpha 0.1 --patch 4 --out p.json", capsys)
    plans = json.load(open("p.json"))["plans"]
    _run("sample p.npy --budget 10 --design random --out s.csv", capsys)
    n_sample = json.load(open("s.json"))["n_population"]
    assert plans["error_rate"]["n_population"] == plans["zone"]["n_population"] == n_sample == 10 * 8
    run = [line for line in printed.splitlines() if line.startswith("run: ")][0][5:]
    assert _run(run.replace("oe-inferencex ", ""), capsys).startswith(f"{plans['error_rate']['labels']} windows to label")


def test_plan_refuses_to_write_over_its_inputs(tmp_path, monkeypatch, capsys):
    a = np.random.default_rng(0).integers(0, 3, (32, 32))
    np.save(tmp_path / "c.npy", a)
    np.save(tmp_path / "band.npy", np.random.default_rng(1).uniform(0, 1, (32, 32)))
    monkeypatch.chdir(tmp_path)
    got = _run("plan c.npy --confidence band.npy --width 0.2 --out band.npy", capsys)
    assert got.startswith("EXIT") and "would overwrite the input" in got


@pytest.mark.parametrize("command, said", [
    ("plan --windows 100", "say what to plan"),
    ("plan --windows 100 --coverage 0.5", "needs both --coverage"),
    ("plan --windows 100 --difference 0.1", "compares two maps"),
    ("plan --windows 100 --width 0.1 --both-wrong 0.2", "does not apply"),
    ("plan --windows 100 --width 0.1 --threshold 0.3", "does not apply"),
    ("plan --windows 100 --width 0.1 --delta 0.2", "does not apply"),
    ("plan --windows 100 --differing 10 --difference 0.05 --width 0.1", "separate call"),
    ("plan --windows 100 --width 0.1 --patch 8", "describe a map"),
    ("plan --windows 100 --width 0.1 --max-labels 0", "at least 1"),
    ("plan --width 0.1", "give the map"),
    ("plan --windows 100 --differing 10 --difference 0.2", None),
])
def test_plan_command_refusals(command, said, capsys):
    got = _run(command, capsys)
    if said:
        assert got.startswith("EXIT") and said in got, got
    else:
        assert "cannot differ by 0.2" in got


def test_plan_tool_reads_counts_alone_and_speaks_in_parameters():
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    r = mcp_server.plan(windows=24_968, differing=5_387, difference=0.027, max_labels=1_000)
    assert r["summary"]["which_map"]["labels"] == plan.plan_which_map(24_968, 5_387, 0.027, max_labels=1_000)["labels"]
    assert r["conclusion"].startswith("Which map:") and r["files"] == {}
    z = mcp_server.plan(windows=100_000, coverage=0.5, alpha=0.05)
    assert "zone_error" in z["conclusion"] and "--" not in z["conclusion"] + z["limits"] + z["next"]


# ----------------------------------------------------------------------------- the second review (2026-10-08)
def test_every_budget_in_the_window_above_the_recommendation_is_checked():
    """plan_which_map(10000, 600, 0.02) recommended 95 while the unchecked 96, 97 and 99 fell to 0.895, 0.887 and
    0.8995; every budget from the recommendation to the end of its window is now checked, and each reaches power."""
    r = plan.plan_which_map(10_000, 600, 0.02)
    a, b = r["every_budget_checked"]
    probs = r["probability_by_budget"]
    assert a == r["labels"] and b >= a + 9
    assert all(probs[str(n)] >= 0.9 for n in range(a, b + 1))


def test_the_bonferroni_window_holds_where_the_review_found_a_dip():
    """plan_zone(100000, 1.0, 0.05, 0.02) recommended 580 for Bonferroni while 602 to 605 fell below 0.9."""
    r = plan.plan_zone(100_000, 1.0, 0.05, 0.02, draws=300)
    b = r["bonferroni"]
    a, z = b["every_budget_checked"]
    assert a == b["labels"] and all(b["probability_by_budget"][str(n)] >= 0.9 for n in range(a, z + 1))
    for n in range(b["labels"], b["labels"] + 30):
        assert plan.bonferroni_probability(100_000, n, 1.0, 0.05, 0.02) >= 0.9, n


def test_a_top_budget_in_a_fall_gives_the_runs_not_none():
    r = plan.plan_zone(300, 1.0, 0.1, 0.0, max_labels=30, draws=400)
    assert r["bonferroni"]["labels"] is None and r["bonferroni"]["runs_reaching_power"] == [(22, 23)]


def test_any_positive_difference_is_at_least_one_window():
    r = plan.plan_which_map(1_000, 200, 1e-13)
    assert r["difference_windows"] == 1 and r["labels"] == 200 and r["probability_at_labels"] == 1.0


def test_a_refused_plan_leaves_no_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    got = _run("plan --windows 100 --width 1.5 --out made/by/refusal.json", capsys)
    assert got.startswith("EXIT") and not (tmp_path / "made").exists()


def test_the_even_spread_argument_on_a_pointwise_counterexample():
    """The review's counterexample to the old, pointwise wording: one labelled set where moving an error to a less
    confident window raises both zones' counts. The argument now pairs each labelled set with the set that swaps
    the two windows; over all labelled sets of a small map, the chance that every zone passes never rises when an
    error moves to a less confident window."""
    import itertools
    N, n, alpha = 12, 4, 0.4
    sizes = [4, 8]
    base = np.zeros(N)
    base[[1, 6]] = 1                                          # one error in each ring

    def chance(err):
        ok = tot = 0
        for L in itertools.combinations(range(N), n):
            L = np.array(L)
            b, k = est.zone_counts(L, err[L], sizes)
            p = [est.zone_pvalue(kk, bb, sz, alpha) for bb, kk, sz in zip(b, k, sizes)]
            ok += all(pp <= 0.5 for pp in p)
            tot += 1
        return ok / tot
    before = chance(base)
    for x, y in ((1, 5), (1, 7), (6, 10), (1, 11)):          # an error moved to a less confident window
        moved = base.copy()
        moved[x], moved[y] = 0, 1
        assert chance(moved) >= before - 1e-12, (x, y)

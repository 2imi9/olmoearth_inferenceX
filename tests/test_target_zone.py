"""The target rule (1.16.0): a zone fixed before labelling, tested as a fallback procedure (Wiens 2003).

The guarantee is checked the way tests/test_trust_zone.py checks the prefix and Bonferroni rules: every error pattern
of a small map meets every draw, and the chance that the rule certifies a zone wrong more than alpha of the time is
computed exactly, for several targets and shares of delta. A rule that hands a failed level's share on, which is not a
fallback procedure, exceeds delta on the same maps, so the enumeration would catch an invalid rule. The label saving,
the exact power the planner gives, the command line and the server follow."""
import itertools
import json
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex import plan
from oe_inferencex.cli import main


def _broken(p, weights, passes):
    """Not a fallback: every level is tested at the shares of all the levels before it, passed or not."""
    acc, level = np.zeros(len(p), bool), 0
    for j in range(len(p)):
        level += weights[j]
        acc[j] = passes(j, level)
    return int(np.flatnonzero(acc).max()) if acc.any() else None


def _worst(N, B, alpha, delta, variants):
    """For each variant (name, target index, weight), the largest share of the C(N, B) draws, over all 2^N error
    patterns in zone order, that certify a zone wrong more than alpha of the time; exact."""
    cov, sizes, _ = est.zone_levels(N, B, alpha, delta, tuple(j / N for j in range(1, N + 1)))
    sizes = np.asarray(sizes, dtype=np.int64)
    draws = np.array(list(itertools.combinations(range(N), B)))
    S = np.zeros((len(draws), N), np.int64)
    S[np.arange(len(draws))[:, None], draws] = 1
    b_all = S.cumsum(1)[:, sizes - 1]
    a = Fraction(str(alpha))
    memo = {}

    def best(k, b, name, tj, w):
        key = (name, tj, w, k.tobytes(), b.tobytes())
        if key not in memo:
            p = [est.zone_pvalue(int(kk), int(bb), int(n), alpha) for kk, bb, n in zip(k, b, sizes)]
            shares = est.target_weights(len(sizes), tj, delta, w)
            if name == "target":
                memo[key] = est.apply_zone_rule(p, b, k, alpha, delta, "target", n=sizes, weights=shares)[1]
            else:
                memo[key] = _broken(p, shares, lambda j, lv: est._at_most(
                    p[j], lv, lambda: est.zone_pvalue_exact(int(k[j]), int(b[j]), int(sizes[j]), alpha), sizes[j]))
        return memo[key]

    worst = dict.fromkeys(variants, Fraction(0))
    for bits in range(2 ** N):
        err = np.array([(bits >> i) & 1 for i in range(N)], np.int64)
        c = np.cumsum(err)[sizes - 1]
        bad = [Fraction(int(cj), int(nj)) > a for cj, nj in zip(c, sizes)]
        if not any(bad):
            continue
        k_all = (S * err).cumsum(1)[:, sizes - 1]
        rows, mult = np.unique(np.hstack([k_all, b_all]), axis=0, return_counts=True)
        for v in variants:
            viol = 0
            for row, m in zip(rows, mult):
                j = best(row[:len(sizes)], row[len(sizes):], *v)
                if j is not None and bad[j]:
                    viol += int(m)
            worst[v] = max(worst[v], Fraction(viol, len(draws)))
    return worst, sizes


@pytest.mark.parametrize("N,B,alpha,delta", [(10, 6, 0.4, 0.3), (11, 7, 0.4, 0.2)])
def test_the_target_rule_holds_delta_on_every_error_pattern_of_a_small_map(N, B, alpha, delta):
    """Over all 2^N error patterns and all C(N, B) draws, for a target at the smallest, a middle and the largest
    tested level and three shares of delta, the chance of certifying a zone wrong more than alpha of the time is at
    most delta, computed exactly. The broken hand-up, with all of delta on the smallest level, tests every level at
    delta whether the one before it passed or not, and exceeds delta on these maps (the uncorrected rule of
    tests/test_trust_zone.py fails on the same two)."""
    _, sizes = _worst(N, B, alpha, delta, [])
    J = len(sizes)
    variants = [("target", tj, w) for tj in (0, J // 2, J - 1) for w in (0.8, 0.5, 1.0)]
    worst, _ = _worst(N, B, alpha, delta, variants + [("broken", 0, 0.8)])
    d = Fraction(str(delta))
    for v in variants:
        assert worst[v] <= d, v
    assert worst[("broken", 0, 0.8)] > d


def test_the_shares_sum_to_delta_and_sit_where_the_rule_says():
    s = est.target_weights(5, 3, 0.1, 0.8)
    assert sum(s) == Fraction(1, 10) and s[3] == Fraction(8, 100) and s[4] == 0
    assert s[0] == s[1] == s[2] == Fraction(2, 100) / 3
    assert est.target_weights(4, 0, 0.1, 0.8) == [Fraction(1, 10), 0, 0, 0]       # nothing below: all of delta
    with pytest.raises(ValueError):
        est.target_weights(4, 4, 0.1)
    with pytest.raises(ValueError):
        est.target_weights(4, 1, 0.1, 0.0)


def test_the_fallback_by_hand():
    """Shares 0.01, 0.01, 0.08, 0, 0 (delta 0.1, target the third level of five). Level 1 fails, level 2 passes at its
    own 0.01, level 3 then has 0.01 + 0.08, level 4 inherits 0.09, level 5 fails and certifies nothing."""
    shares = est.target_weights(5, 2, 0.1, 0.8)
    p = [0.5, 0.009, 0.085, 0.09, 0.2]
    acc, best = est.apply_zone_rule(p, [9] * 5, [0] * 5, 0.05, 0.1, "target", weights=shares)
    assert acc.tolist() == [False, True, True, True, False] and best == 3
    # without the second level's hand-up the target holds only its own 0.08 and fails at 0.085
    p2 = [0.5, 0.02, 0.085, 0.09, 0.2]
    assert est.apply_zone_rule(p2, [9] * 5, [0] * 5, 0.05, 0.1, "target", weights=shares)[1] is None


def _map(N=4000, top_error=0.01, rest_error=0.12, seed=0):
    rng = np.random.default_rng(seed)
    margin = -np.arange(N, dtype=float)                                # window 0 the most confident
    err = np.where(np.arange(N) < N // 2, rng.random(N) < top_error, rng.random(N) < rest_error).astype(float)
    return margin, err


def test_certify_zone_with_a_target():
    margin, err = _map()
    idx = np.random.default_rng(1).choice(margin.size, 300, replace=False)
    r = est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.5)
    assert r["rule"] == "target" and r["target"] == 0.5 and r["target_weight"] == est.TARGET_WEIGHT
    assert r["target_certified"] and r["coverage"] >= 0.5
    assert sum(lv["share_of_delta"] for lv in r["levels"]) == pytest.approx(0.1)
    assert r["note"].startswith("target rule: the 50% zone was fixed before labelling")
    with pytest.raises(ValueError, match="needs a target coverage"):
        est.certify_zone(margin, idx, err[idx], 0.05, rule="target")
    with pytest.raises(ValueError, match="needs a target coverage"):
        est.certify_zone(margin, idx, err[idx], 0.05, rule="prefix", target=0.5)
    with pytest.raises(ValueError, match="level of the zone grid"):
        est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.52)


def test_the_target_is_tested_below_the_level_cut():
    """With 60 labels the cut tests only coverages from 0.75 at alpha 0.05; a target of 0.3 is tested anyway."""
    margin, err = _map(top_error=0.0)
    idx = np.random.default_rng(2).choice(margin.size, 60, replace=False)
    cut = est.zone_levels(margin.size, 60, 0.05)[2]
    assert cut > 0.3
    r = est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.3)
    assert r["levels"][0]["coverage"] == 0.3


def test_the_planner_gives_the_targets_own_test_exactly():
    """target_probability is the target's own test at 0.8 of delta, which zone_level_probability gives too where the
    cut tests the level; and with the target right it needs about half the Bonferroni rule's labels."""
    for n in (150, 300, 450):
        assert plan.target_probability(4000, n, 0.5, 0.05, 0.0125) == pytest.approx(
            plan.zone_level_probability(4000, n, 0.5, 0.05, 0.0125, level=0.08), abs=1e-12)
    r = plan.plan_zone(4000, 0.5, 0.05, zone_error=0.0125)
    assert r["target"]["exact"] and r["target"]["lower_bound"]
    assert r["target"]["labels"] < 0.6 * r["bonferroni"]["labels"]


def test_the_planners_bound_holds_against_the_rule_itself():
    """At the budget the planner recommends, the rule certifies the target on at least 0.9 of draws (Monte Carlo,
    400 draws, so the bound is checked to within about 0.03)."""
    margin, err = _map(N=4000, top_error=0.0125, rest_error=0.2, seed=3)
    n = plan.plan_zone(4000, 0.5, 0.05, zone_error=float(err[:2000].mean()))["target"]["labels"]
    rng = np.random.default_rng(4)
    hits = sum(est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.5)["target_certified"]
               for idx in (rng.choice(4000, n, replace=False) for _ in range(400)))
    assert hits / 400 >= 0.87


# ----------------------------------------------------------------------------- the command line and the server
def _labelled(tmp_path, design_args, name):
    """A 64 x 64 binary probability map, constant over each 4 px window, wrong on 2% of its confident windows and 30%
    of its unsure ones; a sample drawn and `wrong` filled from the truth."""
    pw = np.random.default_rng(5).random((16, 16))
    flip = np.random.default_rng(6).random((16, 16)) < np.where(np.abs(pw - 0.5) < 0.2, 0.3, 0.02)
    truth = np.kron(np.where(flip, pw <= 0.5, pw > 0.5).astype(int), np.ones((4, 4), int))
    p = np.kron(pw, np.ones((4, 4)))
    np.save(tmp_path / "p.npy", p)
    out = tmp_path / f"{name}.csv"
    assert main(["sample", str(tmp_path / "p.npy"), "--budget", "200", "--out", str(out), *design_args]) == 0
    rows = out.read_text().splitlines()
    head = rows[0].split(",")
    body = [r.split(",") for r in rows[1:]]
    for r in body:
        r[head.index("wrong")] = str(int(int(r[head.index("map_class")]) !=
                                         truth[int(r[head.index("pixel_row")]), int(r[head.index("pixel_col")])]))
    out.write_text("\n".join([",".join(head)] + [",".join(r) for r in body]) + "\n")
    return out


def test_sample_fixes_the_target_and_certify_reads_it(tmp_path, capsys):
    out = _labelled(tmp_path, ["--design", "random", "--target", "0.5"], "t")
    side = json.loads((tmp_path / "t.json").read_text())
    assert side["target"] == {"coverage": 0.5, "weight": est.TARGET_WEIGHT}
    capsys.readouterr()
    assert main(["certify", str(out), "--alpha", "0.1"]) == 0
    printed = capsys.readouterr().out
    assert "target rule" in printed
    z = json.loads((tmp_path / "t_zone.json").read_text())
    assert z["rule"] == "target" and z["target"] == 0.5 and "target_certified" in z
    with pytest.raises(SystemExit, match="--rule prefix on the same labels would be a second test"):
        main(["certify", str(out), "--alpha", "0.1", "--rule", "prefix"])
    plain = _labelled(tmp_path, ["--design", "random"], "r")
    with pytest.raises(SystemExit, match="needs the target fixed before labelling"):
        main(["certify", str(plain), "--alpha", "0.1", "--rule", "target"])


@pytest.mark.parametrize("extra,match", [
    (["--design", "confidence", "--target", "0.5"], "needs a simple random sample"),
    (["--design", "sequential", "--alpha", "0.1", "--target", "0.5"], "needs a simple random sample"),
    (["--design", "random", "--target", "0.52"], "a level of the zone grid")])
def test_sample_refuses_a_target_it_cannot_honour(tmp_path, extra, match):
    np.save(tmp_path / "p.npy", np.random.default_rng(7).random((32, 32)))
    with pytest.raises(SystemExit, match=match):
        main(["sample", str(tmp_path / "p.npy"), "--budget", "50", "--out", str(tmp_path / "s.csv"), *extra])


def test_the_server_samples_with_a_target_and_certifies_by_it(tmp_path):
    from oe_inferencex import mcp_server
    out = _labelled(tmp_path, ["--design", "random", "--target", "0.5"], "t")
    r = mcp_server.certify(sample_csv=str(out), alpha=0.1)
    assert r["summary"]["rule"] == "target" and r["summary"]["target"] == 0.5
    np.save(tmp_path / "q.npy", np.random.default_rng(8).random((32, 32)))
    s = mcp_server.sample(scores=str(tmp_path / "q.npy"), out_dir=str(tmp_path / "m"), budget=50, design="random",
                          target=0.5)
    assert s["summary"]["target"] == {"coverage": 0.5, "weight": est.TARGET_WEIGHT}
    assert "the target, the 50% most confident zone, is fixed now" in s["limits"]


def test_labels_from_the_confident_end_are_refused_by_certify():
    """The mirror image of the review-set check (1.16.0): 200 windows from the most confident fifth would make every
    zone look clean. A random draw passes."""
    margin, err = _map()
    top = np.random.default_rng(9).choice(800, 200, replace=False)            # from the 800 most confident windows
    chk = est.review_set_check(top, margin)
    assert chk["looks_drawn_from_the_confident_end"] and not chk["looks_like_a_review_set"]
    with pytest.raises(ValueError, match="drawn from the confident end of the map"):
        est.certify_zone(margin, top, err[top], 0.05)
    rnd = np.random.default_rng(10).choice(margin.size, 200, replace=False)
    assert not est.review_set_check(rnd, margin)["looks_drawn_from_the_confident_end"]
    # a census is neither end
    assert not est.review_set_check(np.arange(margin.size), margin)["looks_drawn_from_the_confident_end"]


def test_a_target_with_no_smaller_zone_tested_holds_all_of_delta_and_the_plan_counts_it():
    """With 100 labels at alpha 0.05 the cut is 0.45, so a target of 0.3 is the smallest level tested: it holds all of
    delta, the note says there is no fallback, and plan's lower bound uses all of delta there (the review of 1.16.0)."""
    margin, err = _map(top_error=0.0)
    idx = np.random.default_rng(11).choice(margin.size, 100, replace=False)
    r = est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.3)
    assert r["target_share_of_delta"] == pytest.approx(0.1) and r["levels"][0]["share_of_delta"] == pytest.approx(0.1)
    # 110 labels: the cut is 0.41, so 0.45 is the smallest level tested; clean, it passes and the note says why
    idx = np.random.default_rng(11).choice(margin.size, 110, replace=False)
    r = est.certify_zone(margin, idx, err[idx], 0.05, rule="target", target=0.45)
    assert r["levels"][0]["coverage"] == 0.45 and r["target_certified"]
    assert "holds all of delta, so there is no fallback" in r["note"]
    assert plan.target_probability(4000, 100, 0.3, 0.05, 0.0) == pytest.approx(
        plan._pass_probability(4000, 100, 1200, 0.05, 0.0, 0.1), abs=1e-12)
    # with a smaller zone tested the target holds 0.8 of delta
    r = est.certify_zone(margin, np.random.default_rng(12).choice(margin.size, 600, replace=False),
                         err[np.random.default_rng(12).choice(margin.size, 600, replace=False)], 0.05, rule="target",
                         target=0.5)
    assert r["target_share_of_delta"] == pytest.approx(0.08)


def test_a_condition_drawn_from_its_confident_end_is_left_out_with_the_right_reason():
    """certify_by_condition gave the review-set reason, 'above' the upper threshold, for labels from the confident end
    (the review of 1.16.0)."""
    N = 4000
    margin = -np.arange(N, dtype=float)
    cgrid = (np.arange(N) % 2).astype(int)
    sample = est.sample_for_estimation(margin, 300, design="condition", condition=cgrid,
                                       condition_names={0: "a", 1: "b"})
    idx = np.asarray(sample["indices"])
    in_a = cgrid[idx] == 0
    conf_a = np.flatnonzero(cgrid == 0)[:400]                          # condition a's 400 most confident windows
    idx[in_a] = np.random.default_rng(13).choice(conf_a, int(in_a.sum()), replace=False)
    sample["indices"] = idx
    r = est.certify_by_condition(sample, np.zeros(idx.size), margin, 0.1)
    a = next(e for e in r["per_condition"].values() if not e.get("tested", True))
    assert "below" in a["reason"] and "chosen for high confidence" in a["reason"]
    assert a["review_set_check"]["end"] == "confident" and a["review_set_check"]["threshold"] < 0.5


def test_the_target_ladder_checks_the_budget_where_the_targets_share_falls():
    """At 187 labels the 15% zone enters below a 20% target, whose share falls from all of delta to 0.8 of it; the
    planner checks that budget and does not recommend one below it (the re-review of 1.16.0)."""
    r = plan.plan_zone(5000, 0.2, 0.08, zone_error=0.0008)
    probs = r["target"]["probability_by_budget"]
    assert r["entry_budgets"]["0.15"] == 187 and "187" in probs and "186" in probs
    assert probs["187"] < 0.9 <= probs["186"]
    assert r["target"]["labels"] > 187

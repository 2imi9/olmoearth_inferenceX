"""Known-answer checks of recorded estimators: build a population whose true value is known, draw the recorded
experiment's own design from it many times, and require the estimator to land on the truth.

A claim's check reads its number back from the artifact that produced it; that catches drift and cannot catch a
formula that was wrong from the start. These reach the same numbers by another route. They exist because exp78's
cluster arm covered 0.904 on MADOS while estimating 1.78 times the truth, and nothing then recorded bias."""
import os

import numpy as np
import pytest

from oe_inferencex import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exp68_design_weighted_lead_is_unbiased_under_its_own_design():
    """exp68's P1 arm: 4,778 field-surveyed LUCAS polygons in the report regions, eight strata, Horvitz-Thompson
    weights from 6 to 42. The pseudo-population replicates each polygon by its weight, so its true lead of the
    margin over the pixel-variance control is known; redrawing exp68's stratified design from it must recover that
    truth. It does, to 0.2 Monte Carlo SE over 400 draws, while the unweighted lead misses by 57 SE; the unweighted
    minus weighted gap comes out at 0.021, as exp68's P4 recorded (0.0216)."""
    z = np.load(os.path.join(ROOT, "exp", "out", "exp68_masks.npz"))
    keep = z["report"] & ~z["no_data"] & (z["obs"] == "field")
    err, margin, var, w = (z[k][keep].astype(float) for k in ("err", "margin", "variance", "weight"))
    strata = np.array([f"{a}:{b}" for a, b in zip(z["y"][keep], z["obs"][keep])])
    assert keep.sum() == 4778
    lead_recorded = metrics.weighted_excess_aurc(var, err, w) - metrics.weighted_excess_aurc(-margin, err, w)
    assert round(lead_recorded, 4) == 0.0953                                        # exp68_summary.json, P1
    m = np.maximum(1, np.rint(w).astype(int))
    pe, pm, pv, ps = np.repeat(err, m), np.repeat(-margin, m), np.repeat(var, m), np.repeat(strata, m)
    truth = metrics.excess_aurc(pv, pe) - metrics.excess_aurc(pm, pe)
    groups = {h: np.flatnonzero(ps == h) for h in np.unique(ps)}
    n_h = {h: int((strata == h).sum()) for h in groups}
    rng = np.random.default_rng(0)
    wl, nl = [], []
    for _ in range(120):
        idx = [rng.choice(g, n_h[h], replace=False) for h, g in groups.items()]
        ww = np.concatenate([np.full(n_h[h], g.size / n_h[h]) for h, g in groups.items()])
        i = np.concatenate(idx)
        e = pe[i]
        wl.append(metrics.weighted_excess_aurc(pv[i], e, ww) - metrics.weighted_excess_aurc(pm[i], e, ww))
        nl.append(metrics.excess_aurc(pv[i], e) - metrics.excess_aurc(pm[i], e))
    wl, nl = np.array(wl), np.array(nl)
    se = wl.std() / np.sqrt(wl.size)
    assert abs(wl.mean() - truth) < 3 * se, (wl.mean(), truth, se)                # unbiased
    assert (nl.mean() - truth) / (nl.std() / np.sqrt(nl.size)) > 10                 # and the check has power
    assert nl.mean() - wl.mean() == pytest.approx(0.0216, abs=0.004)               # P4's recorded gap, another route


def test_exp65_cross_fitted_lead_has_no_optimism_and_reproduces():
    """exp65's held-out fusion leads come from cross-fitting by tile. The known answer for a leak-free pipeline is a
    null: scramble the four extra readings so they carry nothing about errors, and the held-out fusion must not beat
    the margin. Over 12 scrambles the largest null lead was +0.00005 on the test split and every Bolivia one was
    negative, against the recorded +0.0021 (20%) and +0.0029 (30%)."""
    from oe_inferencex.calibrate import fit_ranker
    z = np.load(os.path.join(ROOT, "exp", "out", "exp65_readings.npz"))
    names = ("confidence", "tile_phase", "boundary", "ndwi_level", "s2_variance")
    sig = {k: z[f"ranker/bolivia/{k}"].astype(np.float64) for k in names}
    err = z["ranker/bolivia/err"].astype(np.float64)
    tile = z["ranker/bolivia/tile"]
    ok = np.ones(err.size, bool)
    conf = metrics.excess_aurc(sig["confidence"], err)
    _, rep = fit_ranker(sig, err, ok, groups=tile, family="x", folds=5)
    assert round(conf, 4) == 0.0105 and round(rep["held_out"]["excess_aurc"], 4) == 0.0084     # the recorded pair
    rng = np.random.default_rng(0)
    for _ in range(3):
        perm = rng.permutation(err.size)
        s0 = {"confidence": sig["confidence"], **{k: sig[k][perm] for k in names[1:]}}
        _, r0 = fit_ranker(s0, err, ok, groups=tile, family="x", folds=5)
        assert conf - r0["held_out"]["excess_aurc"] < 0.0005                                # no lead from noise


def test_exp55_activation_clustered_figures_reproduce_from_the_per_event_records():
    """The clustered figures in geoid-capture-effect-size and geoid-exception-rate were computed by hand in a docs
    commit (297a6e0) with no generator and no check: activations as the unit, areas averaged within an activation
    first. Recomputed here from exp55's per-event records by an independent route, they reproduce exactly."""
    import json
    from math import comb
    R = json.load(open(os.path.join(ROOT, "exp", "out", "exp55_summary.json")))["results"]
    want = {"A": ("control NDWI level", 0.843, 16.9, (9, 0), 2.0e-03), "B": ("control S1 level", 0.607, 12.1, (9, 1), 1.1e-02)}
    for t, (ctl, cap_w, ratio_w, (w_w, l_w), p_w) in want.items():
        ev, scored = R[t]["per_event"], R[t]["across_events"]["events_scored"]
        conf = "averaged confidence"
        act = {e: e.rsplit("-", 1)[0] for e in scored}
        acts = sorted(set(act.values()))
        cap = {e: ev[e]["capture"][conf]["0.05"] for e in scored}
        ratio = {e: cap[e] / (max(1, round(0.05 * ev[e]["n_windows"])) / ev[e]["n_windows"]) for e in scored}
        gain = {e: ev[e]["pooled_eaurc"][ctl] - ev[e]["pooled_eaurc"][conf] for e in scored}
        by = lambda d: [np.mean([d[e] for e in scored if act[e] == a]) for a in acts]
        assert round(float(np.median(by(cap))), 3) == cap_w and round(float(np.median(by(ratio))), 1) == ratio_w
        g = by(gain)
        w, l = sum(x > 0 for x in g), sum(x < 0 for x in g)
        p = sum(comb(w + l, k) for k in range(w, w + l + 1)) / 2 ** (w + l)
        assert (w, l) == (w_w, l_w) and p == pytest.approx(p_w, rel=0.06)


def test_exp70_p3_p_value_is_a_rank_sum_over_24_tasks_not_a_sign_test_over_144_pairs():
    """exp70's P3 recorded p = 6.9e-15 from a sign test over 144 task pairs treated as independent. Recomputed here
    from the per-task results by brute force over a million relabellings of the 24 tasks, the p is about 0.0055, and
    the recorded rank-sum p must match it."""
    import json
    S = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))
    R = S["results"]["tasks"]
    adv = {t: r["signals"]["entropy"]["excess_aurc"] - r["signals"]["margin"]["excess_aurc"] for t, r in R.items() if "entropy" in r["signals"]}
    gaps = {t: R[t]["generalisation_gap"] for t in adv}
    med = float(np.median(list(gaps.values())))
    v = np.array([adv[t] for t in adv]); wide_m = np.array([gaps[t] > med for t in adv])
    stat = lambda m: ((v[m][:, None] < v[~m][None]).sum() + 0.5 * (v[m][:, None] == v[~m][None]).sum())
    obs = stat(wide_m)
    rng = np.random.default_rng(1)
    hits = sum(stat(rng.permutation(wide_m)) >= obs - 1e-9 for _ in range(20000))
    p_mc = hits / 20000
    p = S["verdicts"]["P3"]["rank_sum_p"]
    assert abs(p - p_mc) < 4 * np.sqrt(p * (1 - p) / 20000) and 0.004 < p < 0.007
    assert S["verdicts"]["P3"]["holds"] is True and "pairwise_p" not in S["verdicts"]["P3"]

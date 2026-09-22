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

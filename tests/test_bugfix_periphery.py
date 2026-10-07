"""Regression tests for the periphery fixes of 2026-10-06: calibrate.fit_side's share right on multi-class pairs, the
ranker's probability, tie handling in augrc and selective_accuracy, degenerate and empty inputs to the metrics, and
the LCC reader at the raster's edges."""
import itertools
import zlib

import numpy as np
import pytest

from oe_inferencex import lcc
from oe_inferencex import metrics as M
from oe_inferencex.calibrate import Fusion, fit_ranker, fit_side


# --------------------------------------------------------------------------- calibrate
def _three_class_pair(n=1200, seed=0):
    """30% of disagreeing windows b right, 20% a right, 50% neither: a, b and the label all differ."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 3, n)
    a, b = lab.copy(), lab.copy()
    kind = rng.random(n)
    for i in range(n):
        others = [c for c in range(3) if c != lab[i]]
        if kind[i] < 0.3:
            a[i] = others[0]
        elif kind[i] < 0.5:
            b[i] = others[0]
        else:
            a[i], b[i] = others[0], others[1]
    ma, mb = rng.random(n), rng.random(n) + 0.8 * (b == lab)
    return a, b, lab, ma, mb


def test_fit_side_share_right_cannot_exceed_the_windows_where_some_side_is_right():
    """A window where neither side matches the label was counted right for a rule believing a: 0.92 against a 0.53 ceiling."""
    a, b, lab, ma, mb = _three_class_pair()
    _, rep = fit_side({"margin": ma}, {"margin": mb}, a, b, np.ones(len(a)), lab, groups=np.arange(len(a)) % 10)
    ceiling = 1.0 - rep["neither_right"]
    assert rep["neither_right"] > 0.4
    assert rep["held_out"]["share_right"] <= ceiling + 1e-12
    assert rep["baseline"]["share_right"] <= ceiling + 1e-12
    # the baseline believes the side with the larger margin and is right where that side equals the label
    assert rep["n_unscored_rows"] == 0
    honest = np.where(ma - mb < 0, b == lab, a == lab).mean()
    assert rep["baseline"]["share_right"] == pytest.approx(float(honest), abs=1e-12)


def test_fit_ranker_probability_is_p_error_at_the_maps_error_rate():
    """With balanced=True Fusion.prob averaged 0.41 and its held-out ECE read 0.36 on a map with 5% of its windows wrong."""
    rng = np.random.default_rng(0)
    n = 20000
    err = (rng.random(n) < 0.05).astype(float)
    s = rng.normal(size=n) + 1.0 * err
    groups = rng.integers(0, 20, n)
    fusion, rep = fit_ranker({"margin": s}, err, np.ones(n), groups=groups)
    _, rep_unbal = fit_ranker({"margin": s}, err, np.ones(n), groups=groups, balanced=False)
    p = fusion.prob({"margin": s})
    assert abs(p.mean() - err.mean()) < 0.01
    assert rep["held_out"]["ece_of_p_error"] < 0.02
    assert rep["held_out"]["ece_without_prior_correction"] > 0.2          # the balanced 50/50 probability, kept by name
    assert abs(rep["held_out"]["excess_aurc"] - rep_unbal["held_out"]["excess_aurc"]) < 0.002   # the order barely moves
    back = Fusion.from_dict(fusion.to_dict())
    assert np.allclose(back.prob({"margin": s}), p)
    # the logit is the balanced one, unchanged: the order and every excess AURC stay as recorded
    assert np.allclose(fusion.score({"margin": s}) + fusion.prior_logit, np.log(p / (1 - p)))


# --------------------------------------------------------------------------- metrics: ties
def _expected_over_tie_orders(u, v, stat):
    """Mean of stat over every ordering of the units that is ascending in u (brute force, tiny n)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    vals = [stat(v[list(p)]) for p in itertools.permutations(range(len(u))) if np.all(np.diff(u[list(p)]) >= 0)]
    return float(np.mean(vals))


def test_augrc_does_not_depend_on_the_raster_order_of_tied_scores():
    """A constant score gave 0.25 with its error first and 0.0625 with it last; the identity says 0.15625."""
    u = np.zeros(4)
    first, last = M.augrc(u, [1.0, 0, 0, 0]), M.augrc(u, [0.0, 0, 0, 1])
    assert first == pytest.approx(M.augrc_from_auroc(0.5, 0.25, 4), abs=1e-15)
    assert last == pytest.approx(M.augrc_from_auroc(0.5, 0.25, 4), abs=1e-15)
    rng = np.random.default_rng(1)
    for _ in range(80):
        n = int(rng.integers(2, 7))
        uu, ee = rng.integers(0, 3, n).astype(float), (rng.random(n) < 0.4).astype(float)
        ref = _expected_over_tie_orders(uu, ee, lambda e: (np.cumsum(e) / len(e)).mean())
        assert M.augrc(uu, ee) == pytest.approx(ref, abs=1e-12)
    # and on a nine-level map, the identity with ties counted half holds whatever the reading order
    grid = rng.integers(0, 9, (32, 32)).astype(float)
    err = (rng.random((32, 32)) < 0.1 + 0.03 * grid).astype(float)
    pos, neg = grid[err > 0], grid[err == 0]
    auroc = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (pos.size * neg.size)
    ident = M.augrc_from_auroc(auroc, err.mean(), err.size)
    for g, e in ((grid, err), (grid.T, err.T), (grid[::-1], err[::-1])):
        assert M.augrc(g, e) == pytest.approx(ident, abs=1e-12)


def test_selective_accuracy_does_not_depend_on_the_raster_order_of_tied_scores():
    """u = zeros(4) at coverage 0.5 gave 1.0 for correct = [1,1,0,0] and 0.0 for [0,0,1,1], the same map reordered."""
    u = np.zeros(4)
    assert M.selective_accuracy(u, [1, 1, 0, 0], (0.5,))[0.5] == pytest.approx(0.5)
    assert M.selective_accuracy(u, [0, 0, 1, 1], (0.5,))[0.5] == pytest.approx(0.5)
    rng = np.random.default_rng(2)
    for _ in range(80):
        n = int(rng.integers(2, 7))
        uu, cc = rng.integers(0, 3, n).astype(float), (rng.random(n) < 0.6).astype(float)
        for c in (0.3, 0.5, 0.8, 1.0):
            k = max(1, int(round(c * n)))
            ref = _expected_over_tie_orders(uu, cc, lambda v: v[:k].mean())
            assert M.selective_accuracy(uu, cc, (c,))[c] == pytest.approx(ref, abs=1e-12)


# --------------------------------------------------------------------------- metrics: degenerate and empty input
def test_weighted_metrics_give_nan_without_weight_and_refuse_negative_weights():
    """Zeroing the errors' weights made weighted_auroc read 0.0, a perfectly inverted ranker; a weight of -1 gave 1e300."""
    rng = np.random.default_rng(3)
    u, e = rng.random(200), (rng.random(200) < 0.3).astype(float)
    w = rng.uniform(1, 40, 200)
    assert np.isnan(M.weighted_auroc(u, e, np.where(e > 0, 0.0, w)))
    assert np.isnan(M.weighted_auroc(u, e, np.where(e > 0, w, 0.0)))
    z = np.zeros(200)
    assert np.isnan(M.weighted_mean(e, z)) and np.isnan(M.weighted_aurc(u, e, z)) and np.isnan(M.weighted_excess_aurc(u, e, z))
    assert all(np.isnan(v) for v in M.weighted_capture_at_budget(u, e, z).values())
    neg = np.ones(200); neg[0] = -1.0
    for call in (lambda: M.weighted_mean(e, neg), lambda: M.weighted_capture_at_budget(u, e, neg),
                 lambda: M.weighted_auroc(u, e, neg), lambda: M.weighted_aurc(u, e, neg)):
        with pytest.raises(ValueError, match="non-negative"):
            call()


def test_every_metric_returns_nan_on_an_empty_population():
    """Six metrics raised an internal IndexError on a tile with no valid window while the others returned NaN or 0.0."""
    E = np.array([])
    for fn in (M.aurc_expected, M.excess_aurc, M.augrc):
        assert np.isnan(fn(E, E))
    for fn in (M.capture_at_budget, M.capture_at_budget_expected, M.selective_accuracy):
        out = fn(E, E)
        assert out and all(np.isnan(v) for v in out.values())
    cov, risk, aurc = M.risk_coverage(E, E)
    assert cov.size == 0 and risk.size == 0 and np.isnan(aurc)
    for fn in (M.weighted_aurc, M.weighted_excess_aurc, M.weighted_auroc):
        assert np.isnan(fn(E, E, E))
    assert all(np.isnan(v) for v in M.weighted_capture_at_budget(E, E, E).values())
    assert np.isnan(M.weighted_mean(E, E))


def test_risk_coverage_accepts_lists():
    """risk_coverage([0.1, 0.2], [0, 1]) raised AttributeError where every other metric accepts an array-like."""
    cov, risk, aurc = M.risk_coverage([0.1, 0.2, 0.3], [0, 1, 1])
    cov2, risk2, aurc2 = M.risk_coverage(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 1]))
    assert np.array_equal(cov, cov2) and np.array_equal(risk, risk2) and aurc == aurc2


# --------------------------------------------------------------------------- lcc
def _fake_tile(monkeypatch):
    """An 8 x 8 raster of 4 x 4 internal tiles, every pixel holding its tile's index + 1, read through the module's own
    _tile and read; only the HTTP range read is replaced."""
    tw = 4
    monkeypatch.setattr(lcc, "_range", lambda url, start, length: zlib.compress(np.full((tw, tw, 1), start + 1, np.uint8).tobytes()))
    t = object.__new__(lcc.LCCTile)
    t.name, t.url, t.bands, t.epsg = "fake", "fake", 1, 3857
    t.width = t.height = 8
    t.tile_w = t.tile_h = tw
    t.tiles_across = 2
    t.tile_offsets, t.tile_counts = np.arange(4), np.ones(4)
    t.x0, t.y0, t.res_x, t.res_y = 0.0, 80.0, 10.0, 10.0
    t.bounds = (t.x0, t.y0 - t.height * t.res_y, t.x0 + t.width * t.res_x, t.y0)
    return t


@pytest.mark.parametrize("row,col", [(-2, 0), (0, 6), (6, 0), (6, 6), (-2, -2), (-3, 5), (20, 20), (-9, 0)])
def test_lcc_read_leaves_pixels_outside_the_raster_at_no_prediction(monkeypatch, row, col):
    """A window over the top, left or right edge was filled from the far side of the raster, and over the bottom crashed."""
    t = _fake_tile(monkeypatch)
    whole = t.read(0, 0, 8)[0]
    got = t.read(row, col, 4)[0]
    rr, cc = np.meshgrid(np.arange(row, row + 4), np.arange(col, col + 4), indexing="ij")
    inside = (rr >= 0) & (rr < 8) & (cc >= 0) & (cc < 8)
    assert (got[~inside] == 0).all()
    assert np.array_equal(got[inside], whole[rr[inside], cc[inside]])


def test_lcc_index_refuses_a_point_just_above_or_left_of_the_raster(monkeypatch):
    """int() truncated -0.5 to 0, so a point 5 m north of the top edge came back as row 0 instead of being refused."""
    t = _fake_tile(monkeypatch)
    assert t.index(15.0, 75.0) == (0, 1) and t.index(0.0, 80.0) == (0, 0)
    for x, y in ((15.0, 85.0), (-5.0, 75.0), (-0.01, 80.0), (15.0, 80.01)):
        with pytest.raises(ValueError, match="outside"):
            t.index(x, y)

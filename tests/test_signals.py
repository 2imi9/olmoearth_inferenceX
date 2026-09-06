"""Signals and pixel controls (oe_inferencex.signals): properties, and equality with the experiment
modules that produced the recorded results when the encoder stack is installed."""
import os
import sys

import numpy as np
import pytest

from oe_inferencex.signals import (S2_BANDS, aligned_tile_phase, boundary_indicator, combine_midrank, confidence,
                                   midrank_pct, ndwi_gradient, ndwi_level, s2_patch_variance)

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp")


def test_confidence_is_negative_margin():
    assert np.array_equal(confidence(np.array([[2.0, -3.0]])), np.array([[-2.0, -3.0]]))
    logits = np.zeros((3, 2, 2))
    logits[0, 0, 0], logits[1, 0, 0] = 5.0, 1.0      # margin 4 at (0, 0); 0 elsewhere
    c = confidence(logits)
    assert c.shape == (2, 2) and c[0, 0] == -4.0 and c[1, 1] == 0.0
    assert np.array_equal(confidence(np.full((3, 2, 2), -2.0), multiclass=False), np.full((3, 2, 2), -2.0))
    assert confidence(np.ones((4, 4), np.float32)).dtype == np.float32          # dtype kept, as the experiments computed it
    with pytest.raises(ValueError):
        confidence(np.zeros((2, 3, 4, 4)))                                     # a batch of multiclass maps is ambiguous
    with pytest.raises(ValueError):
        confidence(np.zeros((1, 4, 4)), multiclass=True)


def test_boundary_indicator_counts_differing_neighbours():
    hard = np.zeros((3, 3), int)
    hard[1, 1] = 1
    b = boundary_indicator(hard)
    assert b[1, 1] == 1.0                                  # all 8 neighbours differ
    assert b[0, 0] == pytest.approx(1 / 8)                 # only the centre differs (edge padding)
    assert np.array_equal(boundary_indicator(np.stack([hard, hard]))[1], b)
    assert np.array_equal(boundary_indicator(hard * 0.9, probabilities=True), b)     # probabilities threshold at 0.5
    multi = np.array([[1.0, 2.0], [1.0, 2.0]])                                        # float class labels stay labels
    assert np.array_equal(boundary_indicator(multi), boundary_indicator(multi.astype(int)))


def test_tile_phase_is_zero_for_shift_invariant_maps_and_positive_otherwise():
    p = np.full((8, 8), 0.3)
    assert np.all(aligned_tile_phase([p, p, p, p]) == 0)
    q = p.copy()
    q[2, 2] = 0.9
    tp = aligned_tile_phase([p, q, p, p])
    assert tp[2, 2] > 0 and tp[6, 6] == 0
    batched = aligned_tile_phase(np.stack([np.stack([p, p]), np.stack([q, q]), np.stack([p, p]), np.stack([p, p])]))
    assert batched.shape == (2, 8, 8) and np.allclose(batched[0], tp)


def test_pixel_controls_shapes_and_water_sign():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 6000, (12, 132, 132)).astype(np.int32)
    assert ndwi_gradient(img, size=128).shape == (32, 32)
    assert ndwi_level(img, size=128).shape == (32, 32) and np.all(ndwi_level(img, size=128) <= 0)
    assert s2_patch_variance(img, size=128).shape == (32, 32)
    water = np.full((12, 8, 8), 1000)
    water[S2_BANDS.index("B03")] = 3000                    # green above NIR: NDWI positive, flat -> zero gradient
    assert np.all(ndwi_gradient(water) == 0) and np.all(ndwi_level(water) == pytest.approx(-0.5))
    batch = ndwi_gradient(np.stack([img, img])[:, :, :64, :64])
    assert batch.shape == (2, 16, 16) and np.allclose(batch[0], ndwi_gradient(img[:, :64, :64]))


def test_midrank_and_combination():
    assert np.allclose(midrank_pct([3.0, 1.0, 3.0, 2.0]), np.array([2.5, 0, 2.5, 1]) / 3)
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert np.allclose(combine_midrank(a, a), midrank_pct(a).reshape(4, 4))
    assert np.allclose(combine_midrank(a, -a), 0.5)


# ----------------------------------------------------------------------------- equality with the experiment code
@pytest.fixture(scope="module")
def exp_modules():
    pytest.importorskip("olmoearth_pretrain")
    sys.path.insert(0, EXP)
    import exp13_stat_corrections as exp13
    import exp18_sen1floods_expert as exp18
    return exp13, exp18


def test_band_order_matches_the_encoder(exp_modules):
    from olmoearth_pretrain.data.constants import Modality
    assert tuple(Modality.SENTINEL2_L2A.band_order) == S2_BANDS


def test_equals_exp13_and_exp18_implementations(exp_modules):
    exp13, exp18 = exp_modules
    rng = np.random.default_rng(123)
    ps = [rng.random((exp13.GRID, exp13.GRID)) for _ in exp13.SHIFTS]
    assert np.allclose(aligned_tile_phase(ps), exp13.aligned_tile_phase(ps))
    img = rng.integers(0, 6000, (12, exp13.SIZE + exp13.PAD, exp13.SIZE + exp13.PAD)).astype(np.int32)
    assert np.allclose(ndwi_gradient(img, size=exp13.SIZE), exp13.ndwi_gradient(img))
    tiles = rng.integers(0, 6000, (3, 12, 64, 64)).astype(np.float32)
    assert np.allclose(ndwi_gradient(tiles, size=exp18.CROP), exp18.ndwi_gradient(tiles))
    p = rng.random((2, exp18.G, exp18.G))
    assert np.array_equal(boundary_indicator(p, probabilities=True), exp18.boundary(p))
    ps60 = np.stack([rng.random((2, exp18.G, exp18.G)) for _ in exp13.SHIFTS]).astype(np.float32)
    assert np.array_equal(aligned_tile_phase(ps60, dtype=np.float32), exp18.aligned_tile_phase(ps60))   # bit for bit
    assert np.allclose(aligned_tile_phase(ps60), exp18.aligned_tile_phase(ps60), atol=1e-6)               # float64 form

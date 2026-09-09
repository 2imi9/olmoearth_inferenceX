"""Reliability signals (oe_inferencex.reliability): k-means centroid signals, input extremity, ensemble
uncertainty, the tile bootstrap, standardization and the quadratic feature map, on small numpy fixtures."""
import numpy as np
import pytest

from oe_inferencex.reliability import (bootstrap_tiles, centroid_signals, ensemble_uncertainty, input_extremity,
                                       kmeans, poly2, standardize)

C3 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])       # three centroids used throughout
INTRA = np.array([2.0, 1.0, 0.5])                           # per-cluster scales for the normalized distance


def _blobs(sd=0.3, n=100, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), n)
    return C3[labels] + sd * rng.standard_normal((3 * n, 2)), labels


def _entropy(p):                                            # binary entropy in nats
    p = np.asarray(p, float)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ----------------------------------------------------------------------------- kmeans
def test_kmeans_recovers_three_separated_blobs():
    X, labels = _blobs()
    centroids, assign, intra_mean = kmeans(X, k=3, iters=25, seed=0)
    assert centroids.shape == (3, 2) and centroids.dtype == np.float64
    assert assign.shape == (300,) and assign.dtype == np.int64
    assert intra_mean.shape == (3,) and intra_mean.dtype == np.float64
    per_blob = [set(assign[labels == g]) for g in range(3)]
    assert all(len(s) == 1 for s in per_blob) and len(set.union(*per_blob)) == 3    # purity 1.0 up to relabelling
    d = np.linalg.norm(X[:, None, :] - centroids[None], axis=-1)
    assert np.array_equal(assign, d.argmin(1))                                       # assignment is the nearest centroid
    assert np.all(np.linalg.norm(C3[:, None] - centroids[None], axis=-1).min(1) < 0.3)
    assert np.all(intra_mean > 0) and np.all(intra_mean < 1.0)                       # sd 0.3 blobs: mean radius ~0.38
    for c in range(3):
        assert intra_mean[c] == pytest.approx(d[assign == c, c].mean())             # mean member distance to the centroid


def test_kmeans_is_deterministic_for_a_seed():
    X, _ = _blobs()
    a, b = kmeans(X, k=3, seed=7), kmeans(X, k=3, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# ----------------------------------------------------------------------------- centroid signals
def test_centroid_signals_at_the_centroids():
    s = centroid_signals(C3.copy(), C3, INTRA)
    assert set(s) == {"normalized distance", "NCDD", "NCDD raw"} and s["NCDD"].shape == (3,)
    assert np.array_equal(s["normalized distance"], np.zeros(3))
    # d_c = 0 at a centroid, so NCDD raw is the sum of the distances to the other centroids, and NCDD (SHRUG-FM eq. 5)
    # the sum of the cluster-normalized ones d_j / INTRA[j]
    assert s["NCDD raw"] == pytest.approx([20.0, 10 + np.sqrt(200.0), 10 + np.sqrt(200.0)], abs=1e-9)
    assert s["NCDD"] == pytest.approx([10 / 1.0 + 10 / 0.5, 10 / 2.0 + np.sqrt(200.0) / 0.5, 10 / 2.0 + np.sqrt(200.0) / 1.0], abs=1e-9)


def test_centroid_signals_hand_computed():
    s = centroid_signals(np.array([[3.0, 4.0], [0.0, 12.0]]), C3, INTRA)
    d = [5.0, np.sqrt(65.0), np.sqrt(45.0)]                  # (3, 4) to the three centroids; nearest is 0
    assert s["normalized distance"][0] == pytest.approx(5.0 / 2.0, abs=1e-9)       # d_c / intra_mean[c]
    # NCDD raw = sum over the other k-1 centroids of (d_j - d_c); the sum over all k would add d_c = 5 on top
    assert s["NCDD raw"][0] == pytest.approx(d[1] + d[2] - 2 * d[0], abs=1e-9)
    # NCDD proper uses the cluster-normalized distances d_j / INTRA[j] (nearest cluster still chosen by raw distance)
    assert s["NCDD"][0] == pytest.approx(d[1] / 1.0 + d[2] / 0.5 - 2 * d[0] / 2.0, abs=1e-9)
    assert s["normalized distance"][1] == pytest.approx(2.0 / 0.5, abs=1e-9)       # nearest is centroid 2, scale 0.5
    assert s["NCDD raw"][1] == pytest.approx(12.0 + np.sqrt(244.0) - 2 * 2.0, abs=1e-9)
    assert s["NCDD"][1] == pytest.approx(12.0 / 2.0 + np.sqrt(244.0) / 1.0 - 2 * 2.0 / 0.5, abs=1e-9)


def test_ncdd_equals_ncdd_raw_when_every_cluster_has_unit_spread():
    s = centroid_signals(np.array([[3.0, 4.0], [0.0, 12.0], [-7.0, -7.0]]), C3, np.ones(3))
    assert s["NCDD"] == pytest.approx(s["NCDD raw"], abs=1e-12)


def test_ncdd_is_not_monotone_when_the_nearest_cluster_is_the_broadest():
    # SHRUG-FM's cluster-normalized deficit: leaving (0, 0), whose spread 2.0 is the largest, its normalized distance
    # grows slowest, so the deficit over the other centroids grows without bound instead of vanishing
    t = np.array([1.0, 4.0, 16.0, 64.0])
    s = centroid_signals(np.stack([-t, -t], axis=1), C3, INTRA)
    assert np.all(np.diff(s["NCDD"]) > 0) and np.all(np.diff(s["NCDD raw"]) < 0)


def test_centroid_signals_along_a_ray_away_from_the_centroid():
    t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    s = centroid_signals(np.stack([-t, -t], axis=1), C3, INTRA)   # leaves (0, 0) away from every centroid
    assert np.all(np.diff(s["normalized distance"]) > 0)
    assert np.all(np.diff(s["NCDD raw"]) < 0)


def test_ncdd_is_higher_in_cluster_than_far_away():
    inside = np.array([[0.5, 0.5], [10.3, -0.2], [-0.4, 9.7]])
    # far points on the inward side of the centroids: NCDD tends to a direction-dependent limit, and along an
    # outward ray of a corner centroid that limit can stay close to the in-cluster level
    far = np.array([[-100.0, -100.0], [-100.0, 0.0], [50.0, 50.0], [200.0, 200.0]])
    s_in, s_far = centroid_signals(inside, C3, INTRA), centroid_signals(far, C3, INTRA)
    assert s_in["NCDD raw"].min() > s_far["NCDD raw"].max()
    assert s_far["normalized distance"].min() > s_in["normalized distance"].max()


# ----------------------------------------------------------------------------- input extremity
def test_input_extremity_midrank_cdf_by_hand():
    train = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])            # n = 4 per feature
    e = input_extremity(train, np.array([[1.0, 2.5], [0.0, 5.0], [4.0, 4.0]]))
    assert set(e) == {"max", "mean"} and e["max"].shape == (3,) and e["mean"].shape == (3,)
    # F(1) = (0 + 1) / 8 -> e = 0.75; F(2.5) = (2 + 2) / 8 -> e = 0; F(4) = (3 + 4) / 8 -> e = 0.75; outside -> e = 1
    assert e["max"] == pytest.approx([0.75, 1.0, 0.75]) and e["mean"] == pytest.approx([0.375, 1.0, 0.75])


def test_input_extremity_of_the_training_rows_is_uniform():
    train = np.random.default_rng(0).random((2000, 3))
    e = input_extremity(train, train)
    assert np.all(e["mean"] >= 0) and np.all(e["max"] <= 1) and np.all(e["max"] >= e["mean"])
    assert e["mean"].mean() == pytest.approx(0.5, abs=0.05)                         # e = 2|u - 1/2| with u uniform


def test_input_extremity_beyond_the_range_and_at_the_median():
    train = np.random.default_rng(1).normal(size=(2000, 4)) * [1, 10, 0.1, 100] + [0, 5, -3, 1000]
    n = len(train)
    e = input_extremity(train, np.stack([train.max(0) + 1.0, train.min(0) - 1.0, np.median(train, axis=0)]))
    assert e["max"][:2] == pytest.approx(1.0, abs=1 / n) and e["mean"][:2] == pytest.approx(1.0, abs=1 / n)
    assert e["max"][2] <= 2 / n


def test_input_extremity_with_a_constant_training_column():
    train = np.stack([np.full(50, 5.0), np.linspace(0, 1, 50)], axis=1)
    e = input_extremity(train, np.array([[5.0, 0.5], [4.0, 0.5], [6.0, 0.5]]))
    assert np.all(np.isfinite(e["max"])) and np.all(np.isfinite(e["mean"]))
    assert e["max"][0] == pytest.approx(0.0, abs=1e-12)          # midrank on the constant: F(5) = (0 + 50) / 100
    assert e["max"][1:] == pytest.approx(1.0) and e["mean"][1:] == pytest.approx(0.5)   # off the constant: e = 1


# ----------------------------------------------------------------------------- ensemble uncertainty
def test_ensemble_uncertainty_of_identical_members():
    q = np.random.default_rng(0).uniform(0.05, 0.95, 6)
    u = ensemble_uncertainty(np.stack([q, q]))
    assert set(u) == {"mutual information", "average entropy", "predictive entropy"}
    assert np.array_equal(u["mutual information"], np.zeros(6))          # exact: the two-member mean is q itself
    assert np.array_equal(u["average entropy"], u["predictive entropy"])
    assert u["predictive entropy"] == pytest.approx(_entropy(q))
    assert ensemble_uncertainty(np.stack([q, q, q]))["mutual information"] == pytest.approx(0.0, abs=1e-12)


def test_ensemble_uncertainty_two_member_value():
    u = ensemble_uncertainty(np.array([[0.1], [0.9]]))
    assert u["mutual information"].shape == (1,)
    assert u["mutual information"][0] == pytest.approx(np.log(2) - _entropy(0.1), abs=1e-9)   # H(0.5) - H(0.1), nats
    assert u["average entropy"][0] == pytest.approx(_entropy(0.1), abs=1e-9)
    assert u["predictive entropy"][0] == pytest.approx(np.log(2), abs=1e-9)


def test_ensemble_uncertainty_extremes_and_trailing_shape():
    u = ensemble_uncertainty(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert all(np.all(np.isfinite(v)) for v in u.values())
    h_eps = _entropy(1e-7)                                                # the clip turns p = 0 or 1 into eps
    assert u["average entropy"] == pytest.approx(h_eps, abs=1e-9)
    assert u["predictive entropy"][:2] == pytest.approx(np.log(2), abs=1e-9)
    assert u["mutual information"][:2] == pytest.approx(np.log(2) - h_eps, abs=1e-9)
    assert u["mutual information"][2] == pytest.approx(0.0, abs=1e-12)
    u = ensemble_uncertainty(np.random.default_rng(0).random((5, 3, 4)))
    assert all(v.shape == (3, 4) for v in u.values())
    assert np.all(u["mutual information"] >= 0)
    assert np.all(u["average entropy"] <= u["predictive entropy"] + 1e-12)       # Jensen: entropy is concave
    assert np.allclose(u["mutual information"], np.maximum(u["predictive entropy"] - u["average entropy"], 0))


# ----------------------------------------------------------------------------- bootstrap, standardize, poly2
def test_bootstrap_tiles_draws_with_replacement_from_the_seeded_generator():
    members = bootstrap_tiles(50, 4, seed=3)
    assert isinstance(members, list) and len(members) == 4
    assert all(m.shape == (50,) and m.dtype == np.int64 for m in members)
    assert all(m.min() >= 0 and m.max() < 50 for m in members)
    assert all(len(np.unique(m)) < 50 for m in members)                          # with replacement: repeats
    rng = np.random.default_rng(3)
    assert all(np.array_equal(m, rng.integers(0, 50, 50)) for m in members)      # sequential draws, one generator
    assert all(np.array_equal(a, b) for a, b in zip(members, bootstrap_tiles(50, 4, seed=3)))
    assert not all(np.array_equal(a, b) for a, b in zip(members, bootstrap_tiles(50, 4, seed=4)))


def test_standardize_columns_and_reuse_of_the_moments():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)) * [1.0, 5.0, 0.1] + [2.0, -3.0, 0.0]
    Z, mu, sd = standardize(X)
    assert Z.shape == X.shape and mu.shape == (3,) and sd.shape == (3,)
    assert np.allclose(Z.mean(0), 0, atol=1e-12) and np.allclose(Z.std(0), 1, atol=0.01)   # either ddof
    assert np.allclose(mu, X.mean(0)) and np.allclose(sd, X.std(0), rtol=0.01)
    Xnew = rng.normal(size=(20, 3))
    Z2, mu2, sd2 = standardize(Xnew, mu, sd)
    assert np.allclose(Z2, (Xnew - mu) / sd) and np.array_equal(mu2, mu) and np.array_equal(sd2, sd)
    assert np.allclose(standardize(X, mu, sd)[0], Z)


def test_standardize_floors_the_scale_of_a_constant_column():
    Z, mu, sd = standardize(np.stack([np.arange(10.0), np.full(10, 7.0)], axis=1))
    assert np.all(np.isfinite(Z)) and np.all(Z[:, 1] == 0) and mu[1] == 7.0
    assert sd[1] == pytest.approx(1e-6)


def test_poly2_layout():
    Z = np.random.default_rng(0).normal(size=(6, 4))
    P = poly2(Z)
    assert P.shape == (6, 4 + 10)                                                    # F + F(F+1)/2
    assert np.array_equal(P[:, :4], Z)
    assert np.array_equal(P[:, 4], Z[:, 0] * Z[:, 0]) and np.array_equal(P[:, 5], Z[:, 0] * Z[:, 1])
    assert np.array_equal(P[:, -1], Z[:, 3] * Z[:, 3])
    expected = np.column_stack([Z] + [Z[:, i] * Z[:, j] for i in range(4) for j in range(i, 4)])   # i-major, i <= j
    assert np.array_equal(P, expected)
    assert np.array_equal(poly2(Z[:, :1]), np.column_stack([Z[:, 0], Z[:, 0] * Z[:, 0]]))

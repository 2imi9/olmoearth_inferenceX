"""Cross-unit tests (oe_inferencex.stats): exact values, and the recorded headline tests recomputed from
the committed exp13 table."""
import csv
import os

import numpy as np
import pytest

from oe_inferencex.stats import (block_bootstrap_indices, cluster_bootstrap_difference, clustered_sign_test,
                                 paired_comparison, sign_test, spearman, wins_losses_ties)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")
RIVER = {"barotse": "Zambezi", "delta": "Zambezi", "kazungula": "Zambezi", "vicfalls_up": "Zambezi",
         "zambezi_20": "Zambezi", "zambezi_50": "Zambezi", "zambezi_80": "Zambezi",
         "cuando_20": "Cuando", "cuando_50": "Cuando", "cuando_80": "Cuando",
         "kafue_20": "Kafue", "kafue_50": "Kafue", "kafue_80": "Kafue", "luangwa_conf": "Luangwa",
         "okavango_50": "Okavango", "okavango_80": "Okavango", "okavango_sep": "Okavango",
         "rovuma_20": "Rovuma", "rovuma_50": "Rovuma", "rovuma_80": "Rovuma",
         "save_20": "Save", "save_50": "Save", "save_80": "Save",
         "shire_20": "Shire", "shire_50": "Shire", "shire_80": "Shire", "shire_liwonde": "Shire"}


def test_sign_test_values():
    assert sign_test(7, 1, "greater") == pytest.approx(9 / 256)          # the preregistered 7/8 threshold, p = 0.035
    assert sign_test(8, 0, "greater") == pytest.approx(1 / 256)
    assert sign_test(8, 0) == pytest.approx(2 / 256)                     # two-sided, p = 0.0078
    assert sign_test(26, 1) == pytest.approx(2 * 28 / 2 ** 27)           # 26/27, p = 4.2e-07
    assert sign_test(0, 0) == 1.0 and sign_test(3, 3) == 1.0
    with pytest.raises(ValueError):
        sign_test(1, 1, "less")


def test_wins_losses_ties_and_paired_comparison():
    d = np.array([0.1, -0.2, 0.0, 1e-13, 0.3])
    assert wins_losses_ties(d) == (2, 1, 2)                                  # 1e-13 is a tie
    d = np.array([0.1, -0.2, 0.0, 0.2, 0.3])
    r = paired_comparison(d, rng=np.random.default_rng(0), n_perm=200)
    assert (r["w"], r["l"], r["t"], r["n"]) == (3, 1, 1, 5) and r["sign_p"] == pytest.approx(0.625)
    assert r["median_gain"] == pytest.approx(0.1) and 0 <= r["perm_p"] <= 1
    assert "perm_p" not in paired_comparison(d)


def test_clustered_sign_test_mean_and_majority():
    gains = {"a1": 1.0, "a2": -0.5, "a3": 0.2, "b1": 0.2, "c1": -1.0, "c2": 0.1, "c3": 0.1}
    clusters = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "c1": "C", "c2": "C", "c3": "C"}
    mean = clustered_sign_test(gains, clusters)
    assert (mean["w"], mean["l"], mean["t"], mean["n_clusters"]) == (2, 1, 0, 3)
    assert mean["per_cluster"]["C"] == pytest.approx(-0.8 / 3)                     # mean vote: C negative
    maj = clustered_sign_test(gains, clusters, aggregate="majority", alternative="two-sided")
    assert (maj["w"], maj["l"]) == (3, 0) and maj["p"] == pytest.approx(2 / 8)


def _exp13_table():
    path = os.path.join(OUT, "exp13_corrected_stats.csv")
    if not os.path.exists(path):
        pytest.skip("exp13 table not committed")
    per = {}
    for r in csv.DictReader(open(path)):
        per.setdefault(r["scene"], {})[r["signal"]] = float(r["eaurc"])
    return per


def test_exp13_headline_tests_from_the_committed_table():
    per = _exp13_table()
    gains = {s: v["baseline"] - v["tile-phase (aligned)"] for s, v in per.items()}
    w, l, t = wins_losses_ties(list(gains.values()))
    assert (w, l, t) == (26, 1, 0) and sign_test(w, l) == pytest.approx(4.17e-07, rel=0.01)   # protocol: 26/27, p = 4e-07
    river = clustered_sign_test(gains, RIVER, aggregate="majority", alternative="two-sided")
    assert (river["w"], river["l"], river["n_clusters"]) == (8, 0, 8) and river["p"] == pytest.approx(0.0078, abs=1e-4)
    assert clustered_sign_test(gains, RIVER)["p"] == pytest.approx(1 / 256)               # mean aggregate, one-sided
    e_dist = {s: v["baseline"] - v["E_dist"] for s, v in per.items()}
    assert wins_losses_ties(list(e_dist.values()))[:2] == (13, 14)                          # ledger: E_dist 13/27


def test_block_bootstrap_resamples_whole_blocks():
    idx = block_bootstrap_indices(32, 4, np.random.default_rng(0))
    assert idx.shape == (1024,)
    groups = idx.reshape(64, 16)                                   # one sampled 4x4 block per consecutive 16 indices
    rows, cols = groups // 32, groups % 32
    assert np.all(rows // 4 == rows[:, :1] // 4) and np.all(cols // 4 == cols[:, :1] // 4)
    assert all(len(set(zip(r, c))) == 16 for r, c in zip(rows, cols))
    assert len(np.unique(rows[:, 0] // 4 * 8 + cols[:, 0] // 4)) < 64      # with replacement: some blocks repeat
    assert np.array_equal(idx, block_bootstrap_indices(32, 4, np.random.default_rng(0)))   # seeded


def test_cluster_bootstrap_difference_prefers_the_better_signal():
    rng = np.random.default_rng(0)
    err = (rng.random(600) < 0.2).astype(float)
    good = err + 0.1 * rng.random(600)          # ranks errors nearly perfectly
    bad = rng.random(600)
    clusters = np.repeat(np.arange(30), 20)
    lo, hi, p_better = cluster_bootstrap_difference(good, bad, err, clusters, n_boot=200)
    assert hi < 0 and p_better == 1.0


def test_spearman_averages_ties_and_flags_constants():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert spearman(x, x) == pytest.approx(1.0) and spearman(x, -x) == pytest.approx(-1.0)
    tied = np.array([0, 0, 0, 1, 1, 1], float)                     # a two-level score
    assert spearman(tied, x) == pytest.approx(spearman(tied, x[::-1] * -1))   # position inside a tie group does not matter
    assert np.isnan(spearman(np.zeros(6), x))                       # constant input: undefined, not 1
    rng = np.random.default_rng(0)
    a, b = rng.random(50), rng.random(50)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    assert spearman(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1])   # equals the plain form without ties

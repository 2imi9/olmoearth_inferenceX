"""The class design (1.17.0): a sample stratified by the map's own class, allocated as Olofsson et al. (2014) advise.

Its allocation sums to the budget and gives every class its floor; its exact intervals cover, computed exactly over
the stratified draw's distribution on 45 maps (and a negative control shows the check can fail): each class's user's
accuracy at 95%, and the class shares and producer's accuracies all together at 95%; and it runs end to end,
certificate included."""
import contextlib
import importlib.util
import io
import itertools
import json
import os

import numpy as np
import pytest

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("seed", range(40))
def test_the_allocation_sums_to_the_budget_and_keeps_the_floors(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 400, rng.integers(2, 9))
    floor = int(rng.integers(1, 60))
    need = int(np.minimum(sizes, floor).sum())
    budget = int(rng.integers(need, sizes.sum() + 1))
    n = est.class_allocation(sizes, budget, floor)
    assert n.sum() == budget and (n <= sizes).all() and (n >= np.minimum(sizes, floor)).all()
    with pytest.raises(ValueError, match="cannot give each"):
        est.class_allocation(sizes, need - 1, floor) if need > 0 else (_ for _ in ()).throw(ValueError("cannot give each"))


def _compositions(n, C):
    if C == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in _compositions(n - a, C - 1):
            yield (a,) + rest


def _exact_coverage(K, n_h, L, level=None):
    """The exact probability, summed over the stratified draw's distribution (a multivariate hypergeometric in each
    stratum, independent across strata), that every class share and producer's accuracy holds together, and the
    smallest probability that a class's user's accuracy holds. K is the S x C table of true counts, stratum h being
    map class h. `level` replaces the intervals' confidence (the negative control)."""
    import functools
    import math
    K = np.asarray(K)
    S, C = K.shape
    sizes = K.sum(1)
    N = int(sizes.sum())
    share = K.sum(0) / N
    pa = [K[j, j] / K[:, j].sum() if K[:, j].sum() else None for j in range(C)]
    ua = [K[h, h] / sizes[h] for h in range(S)]
    outcomes = []
    for h in range(S):
        total = math.comb(int(sizes[h]), n_h[h])
        outcomes.append([(x, math.prod(math.comb(int(K[h, j]), x[j]) for j in range(C)) / total)
                         for x in _compositions(n_h[h], C) if all(x[j] <= K[h, j] for j in range(C))])
    from oe_inferencex.estimate import per_class            # where _class_design_exact looks hypergeom_interval up (ADR-002)
    orig = per_class.hypergeom_interval
    cached = functools.lru_cache(maxsize=None)(lambda k, n, M, conf: orig(k, n, M, conf if level is None else level))
    per_class.hypergeom_interval = lambda k, n, M, conf=0.95: cached(k, n, M, conf)
    joint, ua_cov = 0.0, np.zeros(S)
    try:
        for combo in itertools.product(*outcomes):
            p = math.prod(q for _, q in combo)
            strata = np.array([h for h, (x, _) in enumerate(combo) for j in range(C) for _ in range(x[j])])
            refs = np.array([j for h, (x, _) in enumerate(combo) for j in range(C) for _ in range(x[j])])
            per = {c: {} for c in range(C)}
            est._class_design_exact(per, {"condition": {"values": list(range(S))}}, strata, np.arange(strata.size),
                                    [int(v) for v in sizes], None, refs, C, L, N)
            ok = all(per[j]["reference_share_exact"]["low"] - 1e-12 <= share[j] <= per[j]["reference_share_exact"]["high"]
                     + 1e-12 for j in range(C))
            ok = ok and all(pa[j] is None or per[j]["producer_accuracy_exact"]["low"] - 1e-12 <= pa[j]
                            <= per[j]["producer_accuracy_exact"]["high"] + 1e-12 for j in range(C))
            joint += p * ok
            for h in range(S):
                u = per[h]["user_accuracy"]
                ua_cov[h] += p * (u["low"] - 1e-12 <= ua[h] <= u["high"] + 1e-12)
    finally:
        per_class.hypergeom_interval = orig
    return joint, float(ua_cov.min())


def _maps():
    rng = np.random.default_rng(116)
    out = [([[36, 4], [5, 20]], (12, 10)), ([[38, 2], [12, 13]], (12, 10)), ([[30, 10], [1, 24]], (15, 12)),
           ([[60, 3], [9, 18]], (20, 15)), ([[25, 3, 2], [2, 16, 2], [1, 3, 12]], (8, 7, 6))]
    for _ in range(40):                                   # two classes, a few errors to many
        sizes = rng.integers(15, 60, 2)
        wrong = [int(rng.integers(0, s // 2 + 1)) for s in sizes]
        out.append(([[sizes[0] - wrong[0], wrong[0]], [wrong[1], sizes[1] - wrong[1]]],
                    tuple(int(rng.integers(5, min(s, 20) + 1)) for s in sizes)))
    return out


def test_the_exact_intervals_cover_on_every_map():
    """Every share and producer's accuracy together at 95%, and each user's accuracy at 95%, computed exactly."""
    for K, n in _maps():
        joint, ua = _exact_coverage(K, n, len(K))
        assert joint >= 0.95 and ua >= 0.95, (K, n, joint, ua)


def test_the_coverage_check_can_fail():
    """The negative control: intervals at 60% instead of their level fall short of 95% on some map, so the check
    above has the power to see an interval that is too narrow."""
    worst = min(min(_exact_coverage(K, n, len(K), level=0.6)) for K, n in _maps()[:10])
    assert worst < 0.95


def test_the_class_design_runs_end_to_end_with_a_certificate(tmp_path):
    pytest.importorskip("rasterio")
    from oe_inferencex import certificate as cert
    from oe_inferencex.cli import main
    spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            qs.write_maps()
            assert main(["sample", "scores.tif", "--design", "class", "--budget", "400", "--min-per-class", "60",
                         "--seed-from-map", "--alpha", "0.1", "--out", "c.csv"]) == 0
            side = json.loads((tmp_path / "c.json").read_text())
            assert side["design"] == "condition" and side["condition"]["by"] == "map class"
            assert min(side["condition"]["n_labelled"]) >= 60 and sum(side["condition"]["n_labelled"]) == 400
            qs.label("c.csv")
            assert main(["estimate", "c.csv", "--per-class"]) == 0
            r = json.loads((tmp_path / "c_estimate.json").read_text())
            assert "stratified by map class" in r["method"] and "condition" not in r["method"]
            assert all("producer_accuracy_exact" in row for row in r["per_class"].values())
            assert main(["certify", "c.csv"]) == 0
            assert main(["certificate", "c_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
        text = out.getvalue()
        assert "-> area" in text and "Certified per map class" in text
    finally:
        os.chdir(cwd)
    assert not (tmp_path / "cert" / "condition").exists()
    r = cert.verify(str(tmp_path / "cert" / "certificate.json"), scores=str(tmp_path / "scores.tif"))
    assert r["verified"], r["checks"]
    with pytest.raises(SystemExit, match="drop --condition"):
        main(["sample", str(tmp_path / "scores.tif"), "--design", "class", "--budget", "200", "--condition",
              str(tmp_path / "truth.tif"), "--out", str(tmp_path / "x.csv")])
    with pytest.raises(SystemExit, match="belongs to --design class"):
        main(["sample", str(tmp_path / "scores.tif"), "--design", "random", "--budget", "200", "--min-per-class", "5",
              "--out", str(tmp_path / "y.csv")])


def test_the_floor_is_fixed_with_the_draw(tmp_path):
    """The class design's floor changes the windows drawn, so it enters the seed: a floor chosen after the labels
    would be a second test of them (the review of 24 October 2026 shopped floors to a wrong statement on 0.153 of
    draws at delta 0.1). A floor below 1 is refused."""
    pytest.importorskip("rasterio")
    from oe_inferencex import certificate as cert
    from oe_inferencex.cli import main
    spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            qs.write_maps()
            seeds = []
            for floor in ("20", "60"):
                assert main(["sample", "scores.tif", "--design", "class", "--budget", "400", "--min-per-class", floor,
                             "--seed-from-map", "--alpha", "0.1", "--out", f"f{floor}.csv"]) == 0
                side = json.loads((tmp_path / f"f{floor}.json").read_text())
                seeds.append(side["seed"])
                assert side["condition"]["min_per_class"] == int(floor)
                assert side["seed_settings"]["min_per_class"] == int(floor)
            assert seeds[0] != seeds[1]
            qs.label("f20.csv")
            assert main(["certify", "f20.csv"]) == 0
            assert "note: delta is the share of all samples" in out.getvalue()      # every route says what delta counts
        side = json.loads((tmp_path / "f20.json").read_text())
        for bad in ("0", "-3"):
            with pytest.raises(SystemExit, match="at least 1"):
                main(["sample", "scores.tif", "--design", "class", "--budget", "400", "--min-per-class", bad,
                      "--out", "bad.csv"])
        with pytest.raises(ValueError, match="at least one label"):
            est.class_allocation([10, 20], 10, 0)
        settings = cert._side_settings(dict(side, condition=dict(side["condition"], min_per_class=60)), None)
        assert cert.seed_from(side["map_sha256"], settings) != side["seed"]
    finally:
        os.chdir(cwd)

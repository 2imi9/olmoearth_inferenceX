"""exp88's measures and grades before the preregistration is frozen and before any real run
(docs/plan/missing_modality.md). Two synthetic cases go through every measure and every grade: in one the extra
errors under a missing input are confident ones, so P2 must hold; in the other the confidence falls with the
accuracy, so P2 must fail. The grades are also checked on planted numbers at their thresholds, the confident-error
share and the estimation study by a second route, and the guard that refuses a full run while the plan is a draft."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp88_missing_modality as e88   # noqa: E402  (module level is torch-free; torch is imported inside the real modes)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e88.smoke(out_dir=str(tmp_path_factory.mktemp("exp88")))


def test_p2_holds_when_the_extra_errors_are_confident(smoke):
    g = smoke["cases"]["confident"]["prereg"]
    assert g["P2"]["holds"] is True
    assert g["P2"]["confident_share_optical_missing"] - g["P2"]["confident_share_full"] >= 0.30, \
        "the planted effect is large, so the grade does not rest on the threshold"
    assert g["P3"]["holds"] is False, "confident errors sit where the margin trusts the map, so it cannot rank them"


def test_p2_fails_when_the_confidence_falls_with_the_accuracy(smoke):
    g = smoke["cases"]["tracks"]["prereg"]
    assert g["P2"]["holds"] is False
    assert g["P2"]["rise_points"] < 0.03, "the planted effect is far from the threshold"
    assert "tracks the lost information" in g["P2"]["reading"]
    assert g["P1"]["holds"] is True and g["P3"]["holds"] is True


def test_every_grade_is_decided_in_both_cases_as_designed(smoke):
    for case, want in e88.SMOKE_EXPECTED.items():
        g = smoke["cases"][case]["prereg"]
        assert g["complete"] is True and g["graded_on"] == "synthetic, probe seed 0"
        assert {p: g[p]["holds"] for p in want} == want, case
        fam = smoke["cases"][case]["results"]["synthetic"]["seeds"]["0"]["families"]
        assert set(fam) == set(e88.FAMILIES), "Togo is measured and reported though not graded"
        for f in fam.values():
            assert set(f["conditions"]) == set(e88.CONDITIONS)
            assert "shift_from_full" in f["conditions"]["optical_missing"] and "shift_from_full" not in f["conditions"]["full"]
    mm = smoke["cases"]["tracks"]["results"]["synthetic"]["seeds"]["0"]["mixed_map"]
    assert mm["n_cloudy_tiles"] == mm["n_tiles"] // 2
    assert {"cloudy_errors_in_review_share", "cloudy_share_of_all_errors", "cloudy_share_of_review_errors"} <= set(mm["review_set"])
    assert smoke["planted"]["P1_at_threshold"] is True and smoke["units_npz"]["written"] is True


def test_the_confident_share_by_explicit_loops():
    """Measure 3 recomputed unit by unit: the threshold is the median margin of the units the full input gets right."""
    rng = np.random.default_rng(3)
    n, C, D = 400, 3, 4
    y = rng.integers(0, C, n)
    conds = {}
    for c in e88.CONDITIONS:
        p = rng.dirichlet(np.ones(C) * (0.4 if c == "optical_missing" else 2.0), size=n)
        conds[c] = {"p": p, "y": y, "tile": None, "emb_te": rng.standard_normal((n, D)).astype(np.float32)}
    meas, units = e88.evaluate_family({"conditions": conds, "emb_tr": rng.standard_normal((50, D)).astype(np.float32), "C": C})
    full = conds["full"]["p"]
    correct_margins = []
    for i in range(n):
        s = sorted(full[i])
        if int(np.argmax(full[i])) == y[i]:
            correct_margins.append(s[-1] - s[-2])
    thr = float(np.median(correct_margins))
    assert meas["threshold_confident"] == pytest.approx(thr, abs=1e-12)
    for c in e88.CONDITIONS:
        p = conds[c]["p"]
        n_err = n_conf = 0
        for i in range(n):
            if int(np.argmax(p[i])) != y[i]:
                n_err += 1
                s = sorted(p[i])
                n_conf += (s[-1] - s[-2]) >= thr
        got = meas["conditions"][c]
        assert got["n_errors"] == n_err and got["confident_errors"]["n"] == n_conf
        assert got["confident_errors"]["share"] == pytest.approx(n_conf / n_err, abs=1e-12)
        lead = got["ranking"]["controls_excess_aurc"][got["ranking"]["best_control"]] - got["ranking"]["margin_excess_aurc"]
        assert got["ranking"]["lead_over_best_control"] == pytest.approx(lead, abs=1e-12)


def test_the_mixed_map_and_the_estimation_study():
    rng = np.random.default_rng(5)
    n_tiles, per = 30, 40
    tile = np.repeat(np.arange(n_tiles), per)
    full = {"margin": rng.random(tile.size), "p1": rng.random(tile.size), "err": (rng.random(tile.size) < 0.1).astype(float)}
    cloud = {"margin": rng.random(tile.size), "p1": rng.random(tile.size), "err": (rng.random(tile.size) < 0.4).astype(float)}
    mm = e88.mixed_map(full, cloud, tile, n_tiles)
    assert mm["cloudy_tiles"].size == 15 and np.array_equal(mm["cloudy_tiles"], e88.mixed_map(full, cloud, tile, n_tiles)["cloudy_tiles"])
    for t in range(n_tiles):
        src = cloud if t in set(mm["cloudy_tiles"].tolist()) else full
        assert np.array_equal(mm["err"][tile == t], src["err"][tile == t])
    es = e88.estimation_study(mm["err"], mm["cloudy"], budget=300, draws=2000, seed=0)
    e, cl = mm["err"] > 0.5, mm["cloudy"]
    assert es["truth"] == pytest.approx({"cloudy": e[cl].mean(), "clear": e[~cl].mean(), "whole": e.mean()})
    # a simple random sample's mean is unbiased for the whole map, so read as the cloudy part it is off by the gap
    assert es["pooled"]["bias_for_cloudy_part"] == pytest.approx(e.mean() - e[cl].mean(), abs=0.006)
    for h in ("cloudy", "clear"):
        assert es["stratified"][h]["coverage"] >= 0.93, "the exact interval per part covers at its nominal rate"
        assert es["stratified"][h]["draws_without_a_label"] == 0


def test_planted_grades_at_their_thresholds():
    got = e88.planted_checks()
    assert got["P1_at_threshold"] is True and got["P2_below"] is False and got["P3_tie_is_not_a_win"] is False
    assert got["P4_one_part_under_covers"] is False and got["P1_without_pastis"] is None


def test_the_cli_smoke_runs_in_seconds_and_writes_its_outputs(tmp_path):
    assert e88.main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    out = json.load(open(tmp_path / "exp88_summary_smoke.json"))
    assert {c: out["cases"][c]["prereg"]["P2"]["holds"] for c in out["cases"]} == {"confident": True, "tracks": False}
    assert out["seconds"] < 30 and (tmp_path / "exp88_units_smoke.npz").exists()


def test_the_full_run_is_refused_while_the_plan_is_a_draft(tmp_path, monkeypatch):
    assert e88.prereg_status("**Status: frozen on 3 October 2026, before any run.**") == "frozen"
    assert e88.prereg_status("**Status:** frozen on 3 October 2026.") == "frozen"
    assert e88.prereg_status("**Status: frozen on 3 October 2026; the draft of 28 September is in the history.**") == "frozen"
    assert e88.prereg_status("**Status: draft, not frozen.** Written 28 September 2026.") == "draft"
    assert e88.prereg_status("**Status: not frozen.**") == "draft"
    assert e88.prereg_status("no status line") == "unknown"
    with open(os.path.join(ROOT, "docs", "plan", "missing_modality.md"), encoding="utf-8") as f:
        assert e88.prereg_status(f.read()) in ("draft", "frozen"), "the plan page's status line is readable"
    plan = tmp_path / "missing_modality.md"
    plan.write_text("# exp88\n\n**Status: draft, not frozen.** Written 28 September 2026.\n")
    monkeypatch.setattr(e88, "PLAN", str(plan))
    assert e88.main([]) == 2, "no run on real embeddings before the page is frozen"


def test_the_units_file_drops_replicate_seeds_before_it_drops_the_graded_one(tmp_path):
    a = {"m/f/y": np.zeros(10, np.int16), "m/f/seed0/full/err": np.zeros(10, np.uint8),
         "m/f/seed1/full/err": np.random.default_rng(0).integers(0, 255, 200_000).astype(np.uint8)}
    big = e88.write_units(a, str(tmp_path / "u.npz"), limit=1e9)
    assert big["written"] and not big["replicate_seeds_dropped"] and big["n_arrays"] == 3
    small = e88.write_units(a, str(tmp_path / "u.npz"), limit=50_000)
    assert small["written"] and small["replicate_seeds_dropped"] and small["n_arrays"] == 2
    none = e88.write_units(a, str(tmp_path / "u.npz"), limit=10)
    assert not none["written"] and not (tmp_path / "u.npz").exists()


def test_the_units_file_drops_the_replication_encoders_seeds_first(tmp_path):
    rng = np.random.default_rng(1)
    big = lambda: rng.integers(0, 255, 200_000).astype(np.uint8)            # noqa: E731
    a = {f"{e88.BASE}/f/seed0/full/err": np.zeros(10, np.uint8), f"{e88.BASE}/f/seed1/full/err": big(),
         f"{e88.LARGE}/f/seed0/full/err": np.zeros(10, np.uint8), f"{e88.LARGE}/f/seed1/full/err": big()}
    one = e88.write_units(a, str(tmp_path / "u.npz"), limit=300_000)
    assert one["written"] and one["replicate_seeds_dropped_for"] == "replication encoder" and one["n_arrays"] == 3
    with np.load(tmp_path / "u.npz") as z:
        assert f"{e88.BASE}/f/seed1/full/err" in z.files and f"{e88.LARGE}/f/seed1/full/err" not in z.files
    both = e88.write_units(a, str(tmp_path / "u.npz"), limit=50_000)
    assert both["written"] and both["replicate_seeds_dropped_for"] == "all" and both["n_arrays"] == 2

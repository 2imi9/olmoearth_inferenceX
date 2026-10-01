"""exp89's grades, guards and per-arm rules before the preregistration is frozen and before any real run
(docs/plan/finetuned_checkpoints.md). Synthetic units with known answers go through every measure and every grade:
in one case the model's confidence beats an informative control on every prediction, in one the control wins, in one
nothing is informative, and arm A's case is graded on P2 and P3 only. The grades are also checked on planted numbers
at their thresholds. The guards are checked without a model: no full run before the page is frozen and the gate has
passed, and a gate file that holds counts and accuracy only. The channel, fill and label rules of each arm, and the
reader on an rslearn layout written to disk, are checked here too. K3's fit runs only where torch is installed."""
import argparse
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp89_finetuned_checkpoints as e89   # noqa: E402  (module level is torch-free)
from oe_inferencex import metrics, stats    # noqa: E402

FROZEN = "# exp89\n\n**Status: frozen on 2 October 2026, before any run.**\n"
DRAFT = "# exp89\n\n**Status: DRAFT, not frozen.** Written 1 October 2026.\n"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e89.smoke(out_dir=str(tmp_path_factory.mktemp("exp89")))


# ----------------------------------------------------------------------------- S1: every grade as designed
def test_every_case_grades_as_designed(smoke):
    for case, want in e89.SMOKE_EXPECTED.items():
        g = smoke["cases"][case]["prereg"]
        assert {p: g[p]["holds"] for p in want} == want, case
        assert g["complete"] is True, case


def test_each_prediction_both_passes_and_fails_somewhere(smoke):
    for p in ("P1", "P2", "P3", "P4", "P5"):
        seen = {smoke["cases"][c]["prereg"][p]["holds"] for c in e89.SMOKE_EXPECTED}
        assert {True, False} <= seen, p


def test_the_grades_do_not_rest_on_their_thresholds(smoke):
    beats, wins, weak = (smoke["cases"][c]["prereg"] for c in ("beats", "control_wins", "weak"))
    assert beats["P1"]["capture_10"] >= 0.9 and weak["P1"]["capture_10"] <= 0.2
    assert beats["P3"]["capture_difference_10"] >= 0.5 and beats["P3"]["lower_one_sided_95"] > 0.3
    assert wins["P3"]["capture_difference_10"] <= -0.3, "the control wins by far"
    assert weak["P2"]["auroc"] <= 0.60
    assert beats["P4"]["width_ratio_300"] <= 0.70 and weak["P4"]["width_ratio_300"] >= 1.0
    assert min(beats["P4"]["coverage"].values()) >= 0.95, "2,000 draws: the coverage is not a coin flip at 0.93"
    assert beats["P5"]["lead_over_k3a"] >= 0.2 and wins["P5"]["lead_over_k3a"] <= -0.05
    assert weak["P5"]["median_coverage"] == 0.0


def test_arm_a_is_graded_on_p2_and_p3_only(smoke):
    g = smoke["cases"]["awf_beats"]["prereg"]
    assert g["graded_on_this_arm"] == ["P2", "P3"]
    assert g["P1"]["graded"] is False and "P2 and P3 only" in g["P1"]["reason"]
    assert g["P4"]["holds"] is None and g["P5"]["holds"] is None
    assert smoke["cases"]["awf_beats"]["estimate"]["graded_budgets"] == [], "N = 400: no budget is graded"


def test_planted_grades_at_their_thresholds(smoke):
    assert all(v is True for v in smoke["planted"].values()), smoke["planted"]


def test_the_graded_confidence_is_the_package_top1_over_the_trained_channels():
    """The untrained channel counts in the argmax (an error when it wins) and never in the confidence."""
    L = np.array([[5.0, 1.0, 0.5, 0.0],      # Mangrove: channel 0 (untrained) wins
                  [-9.0, 2.0, 1.0, 0.0],
                  [-9.0, 0.0, 0.0, 3.0]])
    rd = e89.readings(L, "mangrove")
    assert rd["pred"].tolist() == [0, 1, 3] and rd["n_pred_untrained"] == 1
    z = L[:, 1:] - L[:, 1:].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert np.allclose(rd["p1"], p.max(1), atol=1e-12)
    srt = np.sort(L[:, 1:], axis=1)
    assert np.allclose(rd["margin"], srt[:, -1] - srt[:, -2])
    assert rd["n_untrained_in_top2"] == 1
    with pytest.raises(ValueError):
        e89.readings(L[:, :3], "mangrove")


@pytest.mark.parametrize("arm,n_out,trained,untrained", [("mangrove", 4, (1, 2, 3), (0,)),
                                                         ("nandi", 11, tuple(range(10)), (10,)),
                                                         ("awf", 10, tuple(range(9)), (9,))])
def test_each_arms_channels(arm, n_out, trained, untrained):
    A = e89.ARMS[arm]
    assert (A["n_out"], A["trained"], A["untrained"]) == (n_out, trained, untrained)
    assert set(A["trained"]) | set(A["untrained"]) == set(range(n_out))


def test_rslearn_centre_pad_and_the_mangrove_label_rule():
    a = np.arange(16).reshape(4, 4)
    assert e89.pad_center(a, 2).tolist() == [[5, 6], [9, 10]], "a 4x4 window is cropped to its centre 2x2"
    # rslearn: extra = 2 - 3 = -1, before = -1 // 2 = -1, after = 0, so the first row and column go
    b = np.arange(9).reshape(3, 3)
    assert e89.pad_center(b, 2).tolist() == [[4, 5], [7, 8]], "an odd crop drops the first row and column"
    # extra = 1, before = 0, after = 1: the padding goes after
    c = np.ones((1, 1), int)
    assert e89.pad_center(c, 2).tolist() == [[1, 0], [0, 0]], "an odd pad goes after"
    stack = np.arange(4 * 4 * 3).reshape(3, 4, 4)
    assert e89.pad_center(stack, 2).shape == (3, 2, 2)
    assert e89.mangrove_unit_label(np.full((2, 2), 2)) == (2, None)
    assert e89.mangrove_unit_label(np.array([[1, 1], [1, 0]]))[0] is None
    assert e89.mangrove_unit_label(np.array([[1, 1], [1, 3]]))[1] == "label pixels disagree"
    assert e89.mangrove_unit_label(np.full((2, 2), 4))[0] is None


def test_the_single_pixel_rule_and_fill_values():
    lab = np.full((5, 5), 10)
    lab[2, 3] = 7
    assert e89.single_pixel_label(lab, 10, e89.ARMS["nandi"]["trained"]) == ((2, 3, 7), None)
    lab[0, 0] = 1
    assert e89.single_pixel_label(lab, 10, e89.ARMS["nandi"]["trained"])[1] == "2 labelled pixels"
    awf = np.full((5, 5), 9)
    awf[1, 1] = 8
    assert e89.single_pixel_label(awf, 9, e89.ARMS["awf"]["trained"])[0] == (1, 1, 8)
    assert e89.single_pixel_label(np.full((3, 3), 9), 9, e89.ARMS["awf"]["trained"])[1] == "0 labelled pixels"


def test_exp21_crop_rule_puts_the_label_at_row_8_and_clamps():
    assert e89.crop_at(63, 63, 30, 40, 16, 0) == (22, 32, 8, 8)
    assert e89.crop_at(63, 63, 30, 40, 16, 3) == (25, 35, 5, 5)
    assert e89.crop_at(63, 63, 2, 61, 16, 0) == (0, 47, 2, 14)


# ----------------------------------------------------------------------------- controls
def _series(months):
    """(1, 12, 12) raw series from a list of {band: value} per month (missing bands 0)."""
    s = np.zeros((1, 12, 12))
    for t, d in enumerate(months):
        for b, v in d.items():
            s[0, t, e89.OLMO_BANDS.index(b)] = v
    return s


def test_mangrove_index_controls_and_the_cloud_count_by_hand():
    months = []
    for t in range(12):
        if t in (3, 4):
            months.append({})                                            # two empty months
        elif t < 6:
            months.append({"B02": 500, "B03": 900, "B11": 300, "B04": 400, "B08": 2000})     # MNDWI > 0
        else:
            months.append({"B02": 2500 if t == 11 else 500, "B03": 300, "B11": 900, "B04": 400, "B08": 1200})
    k = e89.index_controls("mangrove", _series(months))
    f = 4 / 10                                                          # 4 of the 10 valid months have MNDWI > 0
    assert k["k4_inundation_ambiguity"][0] == pytest.approx(min(f, 1 - f))
    mndwi = [(900 - 300) / 1200] * 4 + [(300 - 900) / 1200] * 6
    assert k["k4_mndwi_near_zero"][0] == pytest.approx(-abs(np.median(mndwi)))
    ndvi = [(2000 - 400) / 2400] * 4 + [(1200 - 400) / 1600] * 6
    assert k["k4_ndvi_month_std"][0] == pytest.approx(np.std(ndvi))
    assert k["k5_cloud_months"][0] == 3, "two empty months and one with B02 above 0.2 reflectance"


def test_nandi_controls_use_exp21s_ndvi_and_the_3x3_block():
    s = _series([{"B04": 400, "B08": 2000 + 100 * t} for t in range(12)])
    nd = (2000 + 100 * np.arange(12) - 400) / (2000 + 100 * np.arange(12) + 400)
    block = np.full((1, 3, 3, 12), np.nan)
    block[0, 1:, 1:] = np.arange(4)[:, None, None].reshape(2, 2, 1) * np.ones(12)       # a block clipped at an edge
    k = e89.index_controls("nandi", s, block)
    assert k["k4_ndvi_temporal_std"][0] == pytest.approx(np.std(nd))
    assert k["k4_ndvi_3x3_std"][0] == pytest.approx(np.std([0, 1, 2, 3])), "read over the pixels it has"


def test_k3_features_fill_empty_months_with_the_units_median():
    s = _series([{} if t == 5 else {"B02": 100 * (t + 1), "B04": 300, "B08": 900} for t in range(12)])
    x = e89.k3_features(s, lambda a: a / 1000.0, "nandi")
    assert x.shape == (1, 12 * 13 + 1)
    per = x[0, :-1].reshape(12, 13)
    b02 = [100 * (t + 1) / 1000 for t in range(12) if t != 5]
    assert per[5, e89.OLMO_BANDS.index("B02")] == pytest.approx(np.median(b02))
    assert per[5, 12] == pytest.approx(0.5), "NDVI of the empty month is the median NDVI, (900 - 300) / 1200"
    assert x[0, -1] == 1, "the count of empty months"
    assert e89.k3_features(s, lambda a: a, "mangrove").shape == (1, 12 * 14 + 1), "Mangrove adds MNDWI"


def test_k2_and_k3_signals():
    k2, freq = e89.class_rarity(np.array([1, 2, 0]), np.array([1, 1, 1, 2]), (1, 2, 3))
    assert freq == {1: 0.75, 2: 0.25, 3: 0.0} and k2.tolist() == [-0.75, -0.25, -0.0]
    prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    sig, cls = e89.k3_signals(prob, (1, 2, 3), np.array([1, 2]))
    assert cls.tolist() == [1, 3]
    assert sig["k3a_no_encoder_uncertainty"] == pytest.approx([0.3, 0.4])
    assert sig["k3b_no_encoder_disagreement"].tolist() == [0.0, 1.0]


def test_k3_fit_is_deterministic_at_seed_89():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((700, 15)), rng.integers(1, 4, 700)
    x[y == 2] += 1.0
    p_a, fit_a = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60])
    p_b, fit_b = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60])
    assert np.array_equal(p_a, p_b) and fit_a == fit_b
    assert p_a.shape == (60, 3) and np.allclose(p_a.sum(1), 1)
    assert fit_a["loss_last_epoch"] < fit_a["loss_first_epoch"]
    p_c, _ = e89.fit_no_encoder_classifier(x, y, (1, 2, 3), x[:60], seed=90)
    assert not np.array_equal(p_a, p_c), "the seed is what fixes the fit"


# ----------------------------------------------------------------------------- the statistics are the package's
def test_the_capture_bootstrap_draws_the_aurc_bootstraps_resamples():
    """A manual loop drawing exactly as stats.cluster_bootstrap_difference does reproduces both bootstraps."""
    rng = np.random.default_rng(5)
    n = 400
    err = (rng.random(n) < 0.08).astype(float)
    a = rng.random(n) + 0.8 * err
    b = rng.random(n) + 0.3 * err
    cl = rng.integers(0, 12, n)
    got = e89.capture_bootstrap(a, b, err, cl, 0.10, n_boot=300, seed=89)
    lo, hi, p = stats.cluster_bootstrap_difference(a, b, err, cl, n_boot=300, seed=89)
    r = np.random.default_rng(89)
    ids = np.unique(cl)
    cap, aurc = [], []
    for _ in range(300):
        sel = np.concatenate([np.flatnonzero(cl == c) for c in r.choice(ids, size=len(ids), replace=True)])
        if err[sel].sum() == 0:
            continue
        cap.append(metrics.capture_at_budget_expected(a[sel], err[sel], (0.10,))[0.10]
                   - metrics.capture_at_budget_expected(b[sel], err[sel], (0.10,))[0.10])
        aurc.append(metrics.aurc_expected(a[sel], err[sel]) - metrics.aurc_expected(b[sel], err[sel]))
    assert got["n_resamples"] == len(cap)
    assert got["lower_one_sided_95"] == pytest.approx(np.percentile(cap, 5))
    assert (lo, hi) == pytest.approx((np.percentile(aurc, 2.5), np.percentile(aurc, 97.5)))


def test_the_estimate_draws_are_seeded_per_arm_design_and_budget():
    s = {(a, d, B): tuple(e89.draw_seeds(a, d, B, 3)) for a in e89.ARM_INDEX for d in e89.DESIGN_INDEX
         for B in e89.EST_BUDGETS}
    assert len(set(s.values())) == len(s)
    assert tuple(np.random.SeedSequence([89, 0, 1, 300]).generate_state(3)) == s[("mangrove", "confidence", 300)]


def test_the_certificate_reads_violation_and_coverage_from_the_true_zone():
    rng = np.random.default_rng(1)
    N = 3000
    err = np.zeros(N)
    err[rng.choice(N, 90, replace=False)] = 1                     # 3%: the whole map is above alpha = 2%
    score = rng.random(N) - err                                     # errors least trusted
    draws = {300: np.stack([rng.choice(N, 300, replace=False) for _ in range(40)])}
    out = e89.certify_study(err, {"confidence": score}, draws, (0.02,))
    c = out["confidence/prefix/300/0.02"]
    assert c["draws"] == 40 and 0 <= c["violation_rate"] <= 0.12
    assert c["best_coverage_possible"] == 0.95, "the 2,850 most trusted windows hold no error; all 3,000 hold 3%"
    assert c["min_labels_to_certify"] == 114 and "confidence/bonferroni/300/0.02" in out


# ----------------------------------------------------------------------------- the guards
def _page(tmp_path, monkeypatch, text):
    page = tmp_path / "page.md"
    page.write_text(text)
    monkeypatch.setattr(e89, "PLAN", str(page))


def _args(tmp_path, arm="mangrove", **kw):
    return argparse.Namespace(arm=arm, out_dir=str(tmp_path), data=None, ckpt=None, offline=True, from_units=None,
                              real=False, **kw)


def test_the_status_reader():
    assert e89.prereg_status(FROZEN) == "frozen"
    assert e89.prereg_status(DRAFT) == "draft"
    assert e89.prereg_status("**Status: frozen on 2 October; the DRAFT is in the history.**") == "frozen"
    assert e89.prereg_status("no status line") == "unknown"
    assert e89.prereg_status() in ("draft", "frozen"), "the real page has a status line"


def test_the_run_and_the_gate_are_refused_while_the_page_is_a_draft(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, DRAFT)
    called = []
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: called.append(a))
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: called.append("build"))
    json.dump({"passed": True, "attempts": [{}]}, open(tmp_path / "exp89_gate_mangrove.json", "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2
    assert e89.cmd_gate(_args(tmp_path)) == 2
    assert not called and capsys.readouterr().out.count("refused") == 2


def test_the_run_is_refused_until_the_gate_passes(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, FROZEN)
    called = []
    monkeypatch.setattr(e89, "run_arm", lambda *a, **k: called.append(a) or {"prereg": {"complete": True}})
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: (None, []))
    assert e89.cmd_run(_args(tmp_path)) == 2, "no gate record"
    gate = tmp_path / "exp89_gate_mangrove.json"
    json.dump({"passed": False, "closed": False, "attempts": [{"pass": False}]}, open(gate, "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2, "a failed gate"
    json.dump({"passed": False, "closed": True, "attempts": [{"pass": False}] * 3}, open(gate, "w"))
    assert e89.cmd_run(_args(tmp_path)) == 2, "a closed gate"
    assert not called
    assert "the gate is closed" in capsys.readouterr().out
    json.dump({"passed": True, "closed": False, "attempts": [{"pass": True}]}, open(gate, "w"))
    assert e89.cmd_run(_args(tmp_path)) == 0 and len(called) == 1


def test_no_validation_window_is_scored_before_freezing(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, DRAFT)
    monkeypatch.setattr(e89, "read_units", lambda *a, **k: pytest.fail("no window may be read"))
    with pytest.raises(RuntimeError, match="not frozen"):
        e89.predict_population("mangrove", None, [{"split": "val"}])


def test_arm_n_is_reported_as_not_run(tmp_path, monkeypatch, capsys):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "nandi_status", lambda: {"status": "not run", "reason": e89.ARMS["nandi"]["not_run_reason"]})
    monkeypatch.setattr(e89, "build_arm", lambda *a, **k: pytest.fail("arm N must not be built"))
    assert e89.cmd_run(_args(tmp_path, arm="nandi")) == 0
    assert e89.cmd_gate(_args(tmp_path, arm="nandi")) == 0
    assert capsys.readouterr().out.count("arm N not run") == 2
    assert e89.ARMS["nandi"]["model_revision"] is None and e89.ARMS["nandi"]["data_revision"] is None


class _Rec(dict):
    pass


def _fake_population(n, n_err, n_out=4, label=1):
    """(records, records with imagery, logits) for the gate: n Mangrove windows, the first n_err predicted wrong."""
    recs = [{"label": label, "drop": None, "pixel_labels": np.full(4, label), "input": 0} for _ in range(n)]
    recs.append({"label": None, "drop": "label pixels disagree", "pixel_labels": np.array([1, 1, 2, 2]), "input": 0})
    lg = np.zeros((n + 1, n_out))
    lg[:, label] = 5.0
    lg[:n_err, label] = -5.0
    lg[:n_err, 3] = 5.0
    return recs, recs, lg


def test_the_gate_writes_counts_and_accuracy_only_and_closes_after_three_failures(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 100))
    for i in range(3):
        rec = e89.run_gate("mangrove", None, [], str(tmp_path))
        assert rec["passed"] is False and len(rec["attempts"]) == i + 1
    assert rec["closed"] is True
    assert set(rec) == set(e89.GATE_KEYS)
    att = rec["attempts"][0]
    assert set(att) == set(e89.ATTEMPT_KEYS), "no confidence, margin or control in the gate file"
    assert att["n_windows"] == 1000 and att["n_errors"] == 100 and att["n_windows_dropped"] == 1
    assert att["n_pixels"] == 4004 and att["n_pixel_errors"] == 402, "every valid pixel, kept window or not"
    text = (tmp_path / "exp89_gate_mangrove.json").read_text()
    for word in ("p1", "margin", "confidence", "k3", "auroc", "logit"):
        assert word not in text.lower(), word
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("a closed gate does not run"))
    assert e89.run_gate("mangrove", None, [], str(tmp_path))["closed"] is True


def test_the_gate_passes_within_tolerance_two_sided_and_is_not_rerun(tmp_path, monkeypatch):
    _page(tmp_path, monkeypatch, FROZEN)
    # 97.6% pixel accuracy needs 2.4% pixel errors: 24 of 1000 windows wrong, plus the dropped window's 2 pixels
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 24))
    rec = e89.run_gate("mangrove", None, [], str(tmp_path))
    assert rec["passed"] is True and abs(rec["attempts"][0]["gap_points"]) <= 0.5
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: pytest.fail("a passed gate is not rerun"))
    assert e89.run_gate("mangrove", None, [], str(tmp_path))["passed"] is True
    other = tmp_path / "far_above"
    other.mkdir()
    monkeypatch.setattr(e89, "predict_population", lambda *a, **k: _fake_population(1000, 0))
    assert e89.run_gate("mangrove", None, [], str(other))["passed"] is False, "100% is as suspect as 96%"


# ----------------------------------------------------------------------------- the reader on an rslearn layout
def test_the_reader_maps_band_sets_to_olmoearths_order_and_resamples(tmp_path):
    pytest.importorskip("rasterio")
    root, names = e89.synthetic_dataset(str(tmp_path), "nandi", n_train=3, n_val=2, size=24, dataset_bands="three")
    windows = e89.list_windows(os.path.join(root, "windows"), "spatial_split")
    assert [w["split"] for w in windows] == ["train"] * 3 + ["val"] * 2
    w = windows[0]
    stack, n_groups = e89.read_stack(w["dir"], w["meta"])
    assert stack.shape == (24, 24, 12, 12) and n_groups == 12
    import rasterio
    with rasterio.open(os.path.join(w["dir"], "layers", "sentinel2.4", "B02_B03_B04_B08", "geotiff.tif")) as src:
        b04 = src.read(3)
    assert np.array_equal(stack[:, :, 4, e89.OLMO_BANDS.index("B04")], b04), "B04 lands at OlmoEarth index 2"
    assert stack[:, :, :, e89.OLMO_BANDS.index("B01")].std() > 0, "the 60 m band is warped onto the 10 m grid"
    rec = e89.unit_of("nandi", windows[3]["dir"], windows[3]["meta"])
    assert rec["label"] is not None and len(rec["crops"]) == 4 and rec["crops"][0].shape == (16, 16, 12, 12)
    assert rec["series"].shape == (12, 12) and rec["ndvi3x3"].shape == (3, 3, 12)
    _, _, empty = e89.unit_indices(rec["series"][None])
    assert empty[0, 5], "the month with no mosaic is empty"
    os.remove(os.path.join(w["dir"], "layers", "sentinel2.7", "completed"))
    groups = e89.completed_groups(w["dir"], "sentinel2")
    assert 7 not in groups and len(groups) == 11, "a group without rslearn's marker is not loaded"
    _, n = e89.read_stack(w["dir"], w["meta"])
    assert n == 11


def test_the_mangrove_reader_crops_the_centre_and_counts_drops(tmp_path):
    pytest.importorskip("rasterio")
    root, names = e89.synthetic_dataset(str(tmp_path), "mangrove", n_train=14, n_val=4)
    windows = e89.list_windows(os.path.join(root, "windows"), "sample_100K")
    big = [w for w in windows if w["meta"]["bounds"][2] - w["meta"]["bounds"][0] == 4]
    assert big, "the synthetic set holds a 4x4 window"
    rec = e89.unit_of("mangrove", big[0]["dir"], big[0]["meta"])
    assert rec["input"].shape == (2, 2, 12, 12) and rec["pixel_labels"].size == 4
    stack, _ = e89.read_stack(big[0]["dir"], big[0]["meta"])
    assert np.array_equal(rec["input"], stack[1:3, 1:3]), "rslearn's centre crop of a 4x4 window"
    drops = [e89.unit_of("mangrove", w["dir"], w["meta"], with_stack=False)["drop"] for w in windows]
    assert "a label pixel is 0" in drops and "label pixels disagree" in drops


def test_the_tar_round_trip_and_the_inventory_without_a_model(tmp_path):
    pytest.importorskip("rasterio")
    tar, names = e89.synthetic_tar(str(tmp_path), "mangrove", n_train=8, n_val=4)
    info = e89.extract_tar(tar, str(tmp_path / "data"))
    assert info["n_members"] > 0 and os.path.exists(tmp_path / "data" / ".exp89_extracted")
    s = e89.dataset_summary("mangrove", e89.find_windows_root(str(tmp_path / "data")))
    assert s["windows_per_split"] == {"train": 8, "val": 4}
    assert s["item_groups_per_window"] == {"12": 12} and s["layers"]["sentinel2"]["completed"] == 12 * 12
    assert s["band_sets"] == {"B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12": 12}
    assert sum(s["units_per_split"].values()) + sum(sum(d.values()) for d in s["dropped_per_split"].values()) == 12


# ----------------------------------------------------------------------------- the pins
def test_every_repo_exp89_names_is_pinned_or_says_why():
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "upstream_revisions.json")))["repos"]
    for arm, A in e89.ARMS.items():
        for key, rev in (("model", A["model_revision"]), ("data", A["data_revision"])):
            entry = rec[A[key]]
            assert entry["revision"] == rev, (arm, key)
            if rev is None:
                assert "401" in entry["why"]

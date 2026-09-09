"""The package reproduces numbers recorded in exp/out from the committed artifacts alone (no encoder needed)."""
import csv
import json
import os

import numpy as np
import pytest

from oe_inferencex.metrics import (aurc_expected, capture_at_budget, excess_aurc, expected_calibration_error, oracle_aurc,
                                   risk_coverage, selective_accuracy)
from oe_inferencex.signals import boundary_indicator, confidence
from oe_inferencex.stats import cluster_bootstrap_difference, sign_test

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def test_exp02_kazungula_aurc_and_error_rate():
    """exp/NOTES.md, exp02: E_case AURC 0.0011 vs max-softmax baseline 0.0009; Base error rate 2.1%."""
    z = np.load(_need("exp02_cache.npz"), allow_pickle=True)
    p_nano, p_base, lab = z["p_nano"], z["p_base"], z["ev_labels"]
    errors = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
    e_case = np.abs(p_nano - p_base)
    baseline = 1 - np.maximum(p_base, 1 - p_base)
    assert round(risk_coverage(e_case, errors)[2], 4) == 0.0011
    assert round(risk_coverage(baseline, errors)[2], 4) == 0.0009
    assert round(float(errors.mean()), 3) == 0.021
    # recipe item 1: the negative absolute logit orders these patches exactly as the saturating form does here
    with np.errstate(divide="ignore"):
        p64 = p_base.astype(np.float64)
        logit = np.log(p64) - np.log1p(-p64)
    assert aurc_expected(confidence(logit), errors) == pytest.approx(aurc_expected(baseline, errors), abs=1e-9)
    assert 0.0 <= boundary_indicator(p_base).mean() <= 1.0


def test_exp13_table_is_internally_consistent_with_the_oracle():
    """Every exp13 row: aurc - eaurc equals the closed-form oracle for 1024 patches and that scene's error count."""
    rows = list(csv.DictReader(open(_need("exp13_corrected_stats.csv"))))
    assert len(rows) == 27 * 6
    for r in rows:
        assert float(r["aurc"]) - float(r["eaurc"]) == pytest.approx(oracle_aurc(1024, int(r["n_errors"])), abs=1e-9)


@pytest.mark.parametrize("name", ["exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv",
                                  "exp36_dihedral_lexicographic.csv"])
def test_constant_score_rows_equal_error_rate_minus_oracle(name):
    """A constant score ties every unit, so its tie-aware E-AURC is the error rate minus the oracle (parts A and B)."""
    rows = [r for r in csv.DictReader(open(_need(name))) if r["signal"] == "constant score"]
    assert len(rows) >= 27
    for r in rows:
        n, k = int(r["n_patches"]), int(r["n_errors"])
        assert float(r["eaurc"]) == pytest.approx(k / n - oracle_aurc(n, k), abs=1e-9)
        e = np.r_[np.ones(k), np.zeros(n - k)]
        assert float(r["aurc"]) == pytest.approx(aurc_expected(np.zeros(n), e), abs=1e-9)
        assert excess_aurc(np.zeros(n), e) == pytest.approx(float(r["eaurc"]), abs=1e-9)


def test_exp28_and_exp30_share_the_error_set_and_the_reference_signals():
    """exp30 and exp31 run on the exp28 scaffolding: identical errors and identical reference E-AURCs per scene."""
    tabs = {}
    for name in ("exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv"):
        tabs[name] = {(r["unit"], r["signal"]): (int(r["n_errors"]), float(r["eaurc"]))
                      for r in csv.DictReader(open(_need(name))) if r["part"] == "A"}
    for sig in ("confidence (baseline)", "tile-phase (aligned)", "boundary indicator", "control NDWI gradient",
                "constant score", "control S2 patch variance", "control NDWI level"):
        for unit in {u for u, s in tabs["exp28_decoder_consistency.csv"] if s == sig}:
            vals = [tabs[n][(unit, sig)] for n in tabs]
            assert vals[0][0] == vals[1][0] == vals[2][0]
            assert vals[0][1] == pytest.approx(vals[1][1], abs=1e-12) and vals[0][1] == pytest.approx(vals[2][1], abs=1e-12)


def test_recorded_prereg_river_tests_are_reproducible_from_the_summaries():
    """The one-sided exact sign test in each summary equals the package's on the stored per-river means."""
    from oe_inferencex.stats import clustered_sign_test
    for name in ("exp28_summary.json", "exp30_summary.json", "exp31_summary.json", "exp36_summary.json"):
        s = json.load(open(_need(name)))
        pr = s["part_a"]["prereg"]["combination_gain_over_confidence"]
        recomputed = clustered_sign_test(pr["per_river"], {})
        assert (recomputed["w"], recomputed["l"], recomputed["t"]) == (pr["w"], pr["l"], pr["t"])
        assert recomputed["p"] == pytest.approx(pr["one_sided_p"], abs=1e-12)



def test_exp21_fine_tuned_model_metrics_from_the_per_window_table():
    """exp21_summary.json (16-px crops) reproduced from exp21_finetuned_awf.csv: AURCs, ECE, selective accuracy,
    error capture at the review budgets and the cluster bootstrap of tiling instability against confidence."""
    rows = [r for r in csv.DictReader(open(_need("exp21_finetuned_awf.csv"))) if r["crop"] == "16"]
    ref = json.load(open(_need("exp21_summary.json")))["crops"]["16"]
    err = np.array([float(r["error"]) for r in rows])
    conf = -np.array([float(r["logit_margin"]) for r in rows])
    tile = np.array([float(r["tile_phase"]) for r in rows])
    bnd = np.array([float(r["boundary"]) for r in rows])
    assert len(rows) == 344 and int(err.sum()) == ref["n_errors"]
    assert aurc_expected(conf, err) == pytest.approx(ref["aurc"]["confidence (neg logit margin)"], abs=1e-12)
    assert aurc_expected(tile, err) == pytest.approx(ref["aurc"]["tiling instability (aligned)"], abs=1e-12)
    assert aurc_expected(bnd, err) == pytest.approx(ref["aurc"]["boundary indicator"], abs=1e-12)
    assert oracle_aurc(len(err), int(err.sum())) == pytest.approx(ref["aurc_oracle"], abs=1e-12)
    top1 = np.array([float(r["top1_prob"]) for r in rows])
    assert expected_calibration_error(top1, 1 - err)[0] == pytest.approx(ref["ece_10bins"], abs=1e-12)
    sel = selective_accuracy(conf, 1 - err)
    assert all(sel[c] == pytest.approx(ref["selective_accuracy_at_coverage"][str(c)], abs=1e-12) for c in (0.5, 0.8, 0.9, 1.0))
    cap = capture_at_budget(tile, err, budgets=(0.05, 0.1, 0.2))
    assert all(cap[b] == pytest.approx(ref["error_capture_at_budget"][str(b)]["tiling instability (aligned)"], abs=1e-12)
               for b in (0.05, 0.1, 0.2))
    clusters = np.array([r["task"] for r in rows])
    lo, hi, p_better = cluster_bootstrap_difference(tile, conf, err, clusters)
    assert (lo, hi, p_better) == pytest.approx(ref["bootstrap_vs_confidence (lo, hi, P(signal better))"]["tiling instability (aligned)"], abs=1e-12)


def test_exp36_preregistered_budget_tests_are_reproducible_from_the_summary():
    """The lexicographic rule's one-sided per-tile sign tests at the 5% and 10% budgets (exp36, run of record 716511)."""
    s = json.load(open(os.path.join(OUT, "exp36_summary.json")))
    pr = s["part_b"]["prereg_lexicographic"]
    assert set(pr) == {"0.05", "0.1"}
    for b, r in pr.items():
        assert r["one_sided"] and r["w"] + r["l"] + r["t"] == s["part_b"]["n_tiles_scored"] == 351
        assert r["sign_p"] == pytest.approx(sign_test(r["w"], r["l"], "greater"), rel=1e-9)
        assert r["pooled"] > r["pooled_confidence"] and r["boot_lo"] > 0
    assert (pr["0.05"]["w"], pr["0.05"]["l"], pr["0.05"]["t"]) == (85, 31, 235)
    assert (pr["0.1"]["w"], pr["0.1"]["l"], pr["0.1"]["t"]) == (112, 48, 191)
    awf = s["part_awf"]["crops"]
    for crop in ("16", "32"):                                   # the two orders pick the same 5% review set on the AWF model
        r = awf[crop]["lexicographic_vs_confidence"]["0.05"]
        assert r["capture_lex"] == pytest.approx(r["capture_confidence"]) and r["boot_lo"] == pytest.approx(0.0)


def test_exp37_cue_shares_recompute_from_the_per_window_tables():
    """exp37's pooled shares among error and correct windows recompute from its per-window tables, with the cues
    derived as the experiment defines them (boundary > 0; |NDWI| < 0.1; top 20% per unit, ties included)."""
    from oe_inferencex.explain import cue_enrichment, top_fraction
    s = json.load(open(os.path.join(OUT, "exp37_summary.json")))
    for part, name, unit_col in (("part_a", "exp37_patches_scenes.npz", "scene"), ("part_b", "exp37_patches_bolivia.npz", "tile")):
        z = np.load(os.path.join(OUT, name))
        err = z["err"] > 0.5
        units = z[unit_col]
        cues = {"boundary": z["boundary"] > 0, "ndwi_ambiguous": z["ndwi_level"] > -0.1,
                "low_confidence": np.zeros(len(err), bool), "unstable": np.zeros(len(err), bool), "dihedral_disagree": np.zeros(len(err), bool)}
        for u in np.unique(units):
            sel = units == u
            for cue, col in (("low_confidence", "conf"), ("unstable", "tile_phase"), ("dihedral_disagree", "dihedral")):
                cues[cue][sel] = top_fraction(z[col][sel], 0.2)
        rec = s[part]["analysis"]["cues"]
        assert rec["boundary"]["n"] == len(err) and rec["boundary"]["n_errors"] == int(err.sum())
        for cue in rec:
            r = cue_enrichment(cues[cue], err, n_boot=0)
            assert r["share_errors"] == pytest.approx(rec[cue]["share_errors"], abs=1e-12)
            assert r["share_correct"] == pytest.approx(rec[cue]["share_correct"], abs=1e-12)
            assert rec[cue]["boot_lo"] <= rec[cue]["enrichment"] <= rec[cue]["boot_hi"]


def test_exp38_prereg_tests_and_exp36_consistency_from_the_summary():
    """exp38 (CPU, on exp37's tables): the exp36 consistency check holds and the preregistered per-tile p-values
    recompute from the counts; the pooled bootstrap at the preregistered budgets does not exclude zero (mixed)."""
    s = json.load(open(os.path.join(OUT, "exp38_summary.json")))
    chk = s["part_b"]["exp36_consistency"]
    assert all(chk[b]["match"] for b in ("0.05", "0.1"))
    assert tuple(chk["0.05"]["recomputed"]) == (85, 31, 235) and tuple(chk["0.1"]["recomputed"]) == (112, 48, 191)
    for b in ("0.05", "0.1"):
        pt, po = s["part_b"]["prereg"][b]["per_tile"], s["part_b"]["prereg"][b]["pooled"]
        assert pt["one_sided"] and pt["n_units"] == 351
        assert pt["sign_p"] == pytest.approx(sign_test(pt["w"], pt["l"], "greater"), rel=1e-9)
        assert pt["w"] > pt["l"] and pt["sign_p"] < 0.001
        assert po["boot_lo"] <= 0 <= po["boot_hi"]                           # the pooled gain is not established
    assert s["part_b"]["tests"]["NDWI-ambiguous, then boundary, then confidence vs boundary, then confidence"]["pooled"]["0.2"]["boot_lo"] > 0


def test_exp39_preregistered_tests_and_ablation_from_the_summary():
    """exp39 (job 725563): P1 (OlmoEarth-space contradiction vs confidence at 10%) passes per tile and pooled; P2 does
    not beat the OlmoEarth space; the pixel-statistics ablation exceeds both, per tile, pooled and on E-AURC."""
    s = json.load(open(os.path.join(OUT, "exp39_summary.json")))
    b = s["part_b"]
    p1, p2, p2oe = b["prereg"]["P1"], b["prereg"]["P2"], b["prereg"]["P2_vs_oe"]
    for r in (p1, p2, p2oe):
        assert r["one_sided"] and r["w"] + r["l"] + r["t"] == b["n_tiles_scored"] == 351
        assert r["sign_p"] == pytest.approx(sign_test(r["w"], r["l"], "greater"), rel=1e-9)
    assert (p1["w"], p1["l"], p1["t"]) == (169, 126, 56) and p1["sign_p"] < 0.01
    assert p2oe["sign_p"] > 0.05                                             # AnySat does not beat the OlmoEarth space
    C, OE, PIX = "confidence (baseline)", "contradiction (OlmoEarth neighbours)", "contradiction (pixel-statistics neighbours)"
    g = b["pooled_capture_gain"]
    assert g[f"{OE} vs {C}"]["0.1"]["boot_lo"] > 0 and g[f"{PIX} vs {C}"]["0.1"]["boot_lo"] > 0.1
    assert b["pooled_capture"][PIX]["0.1"] == pytest.approx(0.682, abs=5e-4) and b["pooled_capture"][C]["0.1"] == pytest.approx(0.465, abs=5e-4)
    assert b["pooled_eaurc"][PIX] < b["pooled_eaurc"][C] < b["pooled_eaurc"][OE]
    t = b["tests"][PIX][f"vs {C}"]
    assert (t["w"], t["l"]) == (219, 131)
    a = s["part_a"]
    assert a["capture_river_tests_vs_confidence"][OE]["0.1"]["w"] == 7 and a["capture_river_tests_vs_confidence"][PIX]["0.1"]["w"] == 4


def test_exp40_replication_is_a_recorded_negative():
    """exp40 (job 725652): the pixel-statistics contradiction fails every primary test on the test split."""
    s = json.load(open(os.path.join(OUT, "exp40_summary.json")))
    b = s["part_b"]
    assert b["prereg"]["supported"] is False and b["n_tiles_scored"] == 483
    ea = b["prereg"]["eaurc_per_tile"]
    assert (ea["w"], ea["l"]) == (215, 267) and ea["sign_p"] == pytest.approx(sign_test(215, 267, "greater"), rel=1e-9)
    assert ea["pooled_eaurc"] > ea["pooled_eaurc_confidence"]
    c10 = b["prereg"]["capture_10_per_tile"]
    assert c10["one_sided"] and (c10["w"], c10["l"], c10["t"]) == (183, 195, 105)
    assert b["prereg"]["capture_10_pooled"]["boot_hi"] < 0
    PIX, OE = "contradiction (pixel-statistics neighbours)", "contradiction (OlmoEarth neighbours)"
    assert b["pooled_capture"][PIX]["0.1"] > b["pooled_capture"][OE]["0.1"]      # the ordering of the spaces replicates


def test_exp41_two_view_disagreement_is_a_recorded_negative():
    """exp41 (job 726248): outside partners' errors are as correlated with OlmoEarth's as the family's; the primary
    fails on both tasks."""
    s = json.load(open(os.path.join(OUT, "exp41_summary.json")))
    f = s["tasks"]["sen1floods11"]
    assert f["prereg"]["supported_vs_confidence"] is False and f["falsification_correlation"]["outside_below_family"] is False
    for m, c in f["error_correlation"].items():
        assert 0.75 < c["phi_test"] < 0.85 and c["p_err_given_oe_err_test"] > 0.79        # every model errs on the same windows
    pt = f["prereg"]["capture_10_per_unit"]
    assert (pt["w"], pt["l"], pt["t"]) == (281, 1263, 35) and pt["sign_p"] == pytest.approx(sign_test(281, 1263, "greater"), rel=1e-9)
    assert f["pooled_capture"]["weighted disagreement (selected partner)"]["0.1"] < f["pooled_capture"]["confidence (OlmoEarth)"]["0.1"]
    a = s["tasks"]["awf_sentinel2"]
    assert a["prereg"]["supported_vs_confidence"] is False and a["prereg"]["boot_p05"] < 0


def test_exp42_shift_averaged_window_is_supported_on_both_testbeds():
    """exp42 (job 726464): W1 beats the grid window on pixel accuracy per tile on Bolivia and on the test split."""
    s = json.load(open(os.path.join(OUT, "exp42_summary.json")))
    assert s["prereg"]["supported"] is True
    for name, counts in (("bolivia", (321, 66, 53)), ("test", (583, 78, 139))):
        r = s["results"][name]["prereg_W1_vs_W0_pixel"]
        assert r["one_sided"] and (r["w"], r["l"], r["t"]) == counts and r["sign_p"] == pytest.approx(sign_test(r["w"], r["l"], "greater"), rel=1e-9)
        acc = s["results"][name]["pixel_accuracy_mean"]
        assert acc["W1 shift-averaged"] - acc["W0 grid"] > 0.008
        pur = s["results"][name]["purity"]
        assert pur["share_of_windows_impure"] < 0.11 and pur["share_of_errors_on_impure_windows"] > 0.44


def test_exp43_segment_majority_breaks_more_than_it_corrects():
    """exp43 (job 727104): the corrected/broken accounting (E_g + C = E_f + B) on both testbeds; not supported."""
    s = json.load(open(os.path.join(OUT, "exp43_summary.json")))
    assert s["prereg"]["supported"] is False and s["prereg"]["complete"] is True
    for name, (C, B) in (("bolivia", (15093, 16273)), ("test", (24525, 30951))):
        r = s["results"][name]["primary_k8_hard"]
        assert (r["C_total"], r["B_total"]) == (C, B) and B > C and r["one_sided"] and r["sign_p"] > 0.5
        assert r["acc_candidate_mean"] - r["acc_w1_mean"] == pytest.approx(r["mean_gain"], abs=2e-3)
        pur = s["results"][name]["w1_errors_by_grid_window_purity"]
        assert 0.4 < pur["share_of_w1_errors_on_impure_windows"] < 0.6 and pur["share_of_pixels_in_impure_windows"] < 0.11

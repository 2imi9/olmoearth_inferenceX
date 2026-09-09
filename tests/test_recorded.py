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


def test_exp43_mechanism_accounting_matches_the_theorem():
    """exp43 (job 727364): pure segments with a correct majority break nothing and pure segments with a wrong majority
    correct nothing, exactly as the Lean segment theorem says; mixed segments do most of the damage; the pooled and
    mean-tile endpoints are both recorded and differ."""
    s = json.load(open(os.path.join(OUT, "exp43_summary.json")))
    for name, broken_share_mixed in (("bolivia", 0.70), ("test", 0.85)):
        r = s["results"][name]
        m = r["mechanism_primary"]
        assert m["pure_correct"]["B"] == 0 and m["pure_wrong"]["C"] == 0
        p = r["primary_k8_hard"]
        assert m["mixed"]["B"] / p["B_total"] > broken_share_mixed
        assert p["pooled_gain"] == pytest.approx((p["C_total"] - p["B_total"]) / p["N_total"], rel=1e-9)
        assert p["pooled_gain"] != p["mean_gain"] and p["pooled_gain"] < 0
        assert r["secondary_k8_hard_twelve_bands_only"]["mean_gain"] < 0          # dropping NDWI does not rescue it
        assert 0.02 < r["block_constant_oracle_limit"]["rate"] < 0.04             # far below the grid's error rate
    assert s["results"]["bolivia"]["tiles_excluded_no_labelled_pixel_in_common_region"] == [47]


def test_exp44_sixteen_offsets_are_better_but_not_worthwhile():
    """exp44 (job 727364): W16 beats the four diagonals on both testbeds, below the preregistered minimum effect;
    an unbalanced offset subset is worse than the diagonals."""
    s = json.load(open(os.path.join(OUT, "exp44_summary.json")))
    assert s["prereg"]["supported"] is False and s["prereg"]["complete"] is True
    for name in ("bolivia", "test"):
        r = s["results"][name]
        w16 = r["tests"]["W16 (all offsets)"]
        assert w16["one_sided"] and w16["sign_p"] < 0.05 and w16["C_total"] > w16["B_total"]
        assert 0 < w16["mean_gain"] < 0.002 and r["prereg"]["passes"] is False
        assert r["pixel_accuracy_mean"]["W16 (all offsets)"] > r["pixel_accuracy_mean"]["W1 (4 diagonals)"]
        assert r["tests"]["W7 (diagonals + horizontal phases)"]["mean_gain"] < 0        # an unbalanced set hurts
        assert s["testbeds"][name]["diagonal_consistency_max_abs_diff"] == 0.0


def test_exp45_v12_replication_and_cross_version_overlap():
    """exp45 (job 728864): on v1.2 Base the cues and the shift-averaged window replicate, confidence is no longer
    the best ranker on Bolivia, and a different architecture still errs on ~3/4 of v1's error windows."""
    s = json.load(open(os.path.join(OUT, "exp45_summary.json")))
    assert s["prereg"]["complete"] is True
    assert s["prereg"]["R3_cue_enrichment"] is True and s["prereg"]["R4_shift_averaged_window"] is True
    assert s["prereg"]["R1_best_ranker"] is False and s["prereg"]["R2_boundary_first"] is False
    bo = s["results"]["bolivia"]
    e = bo["R1_best_ranker"]["pooled_eaurc"]
    assert e["tile-phase"] < e["confidence"] and e["control NDWI level"] < e["confidence"]   # confidence is third
    assert s["results"]["test"]["R1_best_ranker"]["replicates"] is True                      # but still leads on the test split
    assert bo["R2_boundary_first"]["replicates"] is True
    for name in ("bolivia", "test"):
        r4 = s["results"][name]["R4_shift_averaged_window"]
        assert r4["replicates"] and r4["mean_gain"] > 0.009
        d = s["D1_cross_version_error_overlap"][name]
        assert 0.70 < d["p_v12_wrong_given_v1_wrong"] < 0.78
        assert d["p_v12_wrong_given_v1_right"] < 0.04 and 0.65 < d["phi"] < 0.75


def test_fine_tuning_moves_the_error_set_more_than_swapping_a_frozen_encoder():
    """The measurement that withdrew the "errors belong to the windows" gloss: on the same AWF points and labels,
    fine-tuning the encoder corrects most of the frozen probe's errors, while swapping a frozen encoder (exp41)
    barely moves the error set."""
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "exp21_finetuned_awf.csv"))) if r["crop"] == "16"]
    ft = np.array([float(r["error"]) for r in rows]) > 0.5
    pr = np.array([float(r["probe_error"]) for r in rows]) > 0.5
    n11, n10, n01, n00 = int((ft & pr).sum()), int((ft & ~pr).sum()), int((~ft & pr).sum()), int((~ft & ~pr).sum())
    assert (n11, n10, n01, n00) == (28, 13, 35, 268) and len(rows) == 344
    corrected = n01 / (n01 + n11)
    phi = (n11 * n00 - n10 * n01) / np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    assert corrected == pytest.approx(35 / 63, rel=1e-9) and corrected > 0.5
    assert phi == pytest.approx(0.4753, abs=5e-4)
    frozen = json.load(open(os.path.join(OUT, "exp41_summary.json")))["tasks"]["sen1floods11"]["error_correlation"]
    assert phi < min(c["phi_test"] for c in frozen.values()) - 0.25          # unfreezing moves it far more than swapping


def test_exp46_modality_moves_the_error_set_most():
    """exp46 (job 729376): the input modality dominates the readout and the backbone; impurity and chips do not explain it."""
    s = json.load(open(os.path.join(OUT, "exp46_summary.json")))
    assert s["prereg"]["P1_readout"] is True and s["prereg"]["P2_modality"] is True and s["prereg"]["complete"] is True
    for name in ("bolivia", "test"):
        r = s["results"][name]
        ref = r["prereg"]["backbone_swap_phi"]
        phi = {k.split("vs ")[1].split()[0]: v["phi"] for k, v in r["pairwise"].items() if k.startswith("A1")}
        assert phi["A2"] < 0.45 < ref                       # modality moves it far more than the backbone
        assert phi["A3"] < ref and phi["A4"] < ref          # readout, only just
        assert phi["A3"] > 0.6                              # and nowhere near fine-tuning's 0.475
        for st in r["purity_strata"].values():              # impurity does not explain the overlap
            assert abs(st["pure"] - st["mixed"]) < 0.2
    assert s["results"]["bolivia"]["accuracy"]["A0 no encoder"] > s["results"]["bolivia"]["accuracy"]["A1 S2 linear"]


def test_exp47_a_no_model_control_matches_confidence_on_bolivia():
    """exp47 (job 729376): on the shift-averaged decision the repository recommends, the NDWI-level control matches or
    beats the model's own confidence on Bolivia under both backbones; confidence always beats tiling instability."""
    s = json.load(open(os.path.join(OUT, "exp47_summary.json")))
    assert s["prereg"]["supported"] is False and s["prereg"]["complete"] is True
    for v in ("v1", "v1_2"):
        for n in ("bolivia", "test"):
            t = s["results"][v][n]["tests"]
            assert t["tile-phase"]["pooled_lead"] > 0.001 and t["tile-phase"]["sign_p"] < 0.05
            for k in ("U+ averaged confidence + tile-phase", "U+ averaged confidence + NDWI level"):
                assert t[k]["pooled_lead"] > 0                        # plain averaged confidence beats every combination
        assert s["results"][v]["bolivia"]["tests"]["control NDWI level"]["sign_p"] > 0.5   # loses the per-tile count
        e = s["results"][v]["bolivia"]["pooled_eaurc"]
        assert e["averaged confidence"] < e["grid confidence"]        # averaging helps ranking as well as accuracy
    assert s["results"]["v1_2"]["test"]["prereg_passes"] is True


def test_exp49_shrug_fm_signals_do_not_beat_confidence_but_a_label_fitted_fusion_does():
    """exp49 (job 731703): SHRUG-FM's signals under the window protocol. Preregistered and supported: under v1.2 the
    averaged confidence beats ensemble mutual information and -NCDD on both testbeds. Secondary: the bagged predictive
    entropy beats confidence on Bolivia under v1; the label-fitted linear fusion beats confidence on three of four arms;
    NCDD as the paper states it (cluster-normalized) is worse than the raw deficit everywhere."""
    s = json.load(open(os.path.join(OUT, "exp49_summary.json")))
    assert s["prereg"] == {"supported": True, "complete": True} and s["n_failures"] == 0
    for n in ("bolivia", "test"):
        t = s["results"]["v1_2"][n]["tests"]
        for k in ("ensemble mutual information", "embedding NCDD"):
            assert t[k]["pooled_lead"] >= 0.001 and t[k]["sign_p"] < 0.05 and t[k]["one_sided"]
    for v in ("v1", "v1_2"):
        for n in ("bolivia", "test"):
            e = s["results"][v][n]["pooled_eaurc"]
            assert e["embedding NCDD raw"] < e["embedding NCDD"]
            for k in ("embedding normalized distance", "input extremity max", "input extremity mean"):
                assert e[k] > 2.5 * e["averaged confidence"]
    b = s["results"]["v1"]["bolivia"]["tests"]["ensemble predictive entropy"]
    assert b["pooled_lead"] <= -0.001 and b["l"] > 2 * b["w"] and b["sign_p"] < 1e-6      # the bag ranks better under v1
    fus = {(v, n): s["results"][v][n]["tests"]["fusion linear (labelled)"]["beats_confidence"] for v in ("v1", "v1_2") for n in ("bolivia", "test")}
    assert fus == {("v1", "bolivia"): True, ("v1", "test"): True, ("v1_2", "bolivia"): True, ("v1_2", "test"): False}
    for v in ("v1", "v1_2"):
        tl = s["results"][v]["bolivia"]["tile_level"]
        assert abs(tl["failure_rate"] - 0.331) < 0.005                    # the paper's flood and burn-scar failure rates are 0.21 and 0.33
        assert tl["per_ranker"]["averaged confidence"]["aurc"] < tl["per_ranker"]["ensemble mutual information"]["aurc"]


def test_exp50_ndwi_carries_the_fusion_and_the_bag_does_not_replicate():
    """exp50 (job 736320): the label-fitted fusion's weight sits on NDWI level and dropping it costs the most on three
    of four arms; the exp49 bag result does not replicate with sixteen fresh members (P1 fails, sign flips across
    draws); a fusion fitted on tile failures beats confidence at the tile level on the multi-region split under both
    backbones but not on Bolivia (P2 fails); shift-label entropy loses to tile-phase on Bolivia (issue 5)."""
    s = json.load(open(os.path.join(OUT, "exp50_summary.json")))
    assert s["prereg"] == {"P1": False, "P2": False, "supported": False, "complete": True} and s["n_failures"] == 0
    for v in ("v1", "v1_2"):
        w = s["results"][v]["fusion_weights"]
        assert max(w, key=lambda k: abs(w[k])) == "control NDWI level" and w["control NDWI level"] > 0.8
        assert w["ensemble average entropy"] > 0.5 and w["ensemble predictive entropy"] > 0.5 and w["averaged confidence"] > 0.25
        for n in ("bolivia", "test"):
            r = s["results"][v][n]
            assert r["pooled_eaurc"]["fusion linear (labelled)"] < r["pooled_eaurc"]["averaged confidence"]
            assert r["pooled_eaurc"]["shift-label entropy"] >= r["pooled_eaurc"]["tile-phase"] - 0.0002
    for v, n in (("v1", "bolivia"), ("v1", "test"), ("v1_2", "bolivia")):
        dr = s["results"][v][n]["drop_one_pooled_eaurc"]
        assert max(dr, key=dr.get) == "control NDWI level"
    assert abs(s["results"]["v1_2"]["bolivia"]["drop_one_pooled_eaurc"]["control NDWI level"] - s["results"]["v1_2"]["bolivia"]["pooled_eaurc"]["averaged confidence"]) < 0.0005
    b1 = s["results"]["v1"]["bolivia"]["tests"]["bag16 predictive entropy"]
    b2 = s["results"]["v1_2"]["test"]["tests"]["bag16 predictive entropy"]
    assert b1["pooled_lead"] > 0 and b1["sign_p"] > 0.5                    # the exp49 bag gain is gone with a fresh draw
    assert b2["pooled_lead"] < 0 and b2["sign_p"] < 1e-4                   # and appears where exp49's bag had lost
    for v in ("v1", "v1_2"):
        assert s["results"][v]["test"]["tile_level"]["tile-fitted fusion (labelled)"]["diff_ci95"][1] < 0
    assert s["results"]["v1_2"]["bolivia"]["tile_level"]["tile-fitted fusion (labelled)"]["diff_ci95"][1] > 0

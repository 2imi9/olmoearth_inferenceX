"""The feature-folder layout of 1.19.0 (ADR-002): every 1.18.1 module path still imports, as the moved module itself."""
import importlib
import subprocess
import sys

import pytest

import oe_inferencex as ox

# The modules that moved into folders: their 1.18.1 path -> their new one.
MOVED = {
    "metrics": "common.metrics", "signals": "common.signals", "stats": "common.stats",
}

# oe_inferencex.__all__ at 1.18.1, sorted.
ALL_1_18_1 = [
    "CUES", "__version__", "aligned_tile_phase", "assess_classmap", "assess_prediction", "attainable_ceiling", "augrc",
    "augrc_from_auroc", "aurc_expected", "betting_interval", "block_bootstrap_indices", "boundary_first_score",
    "boundary_indicator", "capture_at_budget", "capture_at_budget_expected", "certify_by_condition",
    "cluster_bootstrap_difference", "cluster_interval", "clustered_sign_test", "combine_midrank", "compare_inferences",
    "confidence", "confidence_strata", "crop_dependence", "crosstab", "cue_enrichment", "derive_cues", "design_effect",
    "design_from_sizes", "design_from_weights", "determinism_check", "disagreement", "equal_allocation",
    "estimate_error_rate", "estimate_from_indices", "exact_coverage_srs", "excess_aurc", "expected_calibration_error",
    "explain_review_set", "fit_ranker", "fit_side", "grade", "hypergeom_interval", "library_table", "midrank_pct",
    "ndwi", "ndwi_gradient", "ndwi_level", "neyman_allocation", "oracle_aurc", "over_groups",
    "paired_cluster_bootstrap", "paired_comparison", "phi", "pool_condition", "pool_to_windows", "pure_stratum_ratio",
    "rank_sum_test", "read_points", "review_mask", "review_order", "review_set_check", "risk_coverage",
    "s2_patch_variance", "sample_for_estimation", "selective_accuracy", "shift_averaged_probability", "side_features",
    "sign_test", "spearman", "stability", "stratified_interval", "stratified_interval_wilson",
    "stratified_mean_and_variance", "summary", "weighted_aurc", "weighted_auroc", "weighted_capture_at_budget",
    "weighted_excess_aurc", "weighted_mean", "where", "which_side", "wilson_interval", "wins_losses_ties",
]


def test_the_finder_and_this_test_name_the_same_moves():
    assert ox._MOVED == MOVED


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_an_old_path_is_the_moved_module(old, new, monkeypatch):
    monkeypatch.delitem(sys.modules, f"oe_inferencex.{old}", raising=False)   # through the finder, not the cache
    try:
        moved = importlib.import_module(f"oe_inferencex.{new}")
    except ImportError as exc:                     # a guarded module without its extra refuses alike at both paths
        with pytest.raises(ImportError) as at_old:
            importlib.import_module(f"oe_inferencex.{old}")
        assert str(at_old.value) == str(exc) and f"oe_inferencex.{old}" not in sys.modules
        return
    assert importlib.import_module(f"oe_inferencex.{old}") is moved
    assert sys.modules[f"oe_inferencex.{old}"] is moved and getattr(ox, old) is moved
    # the moved module keeps its own name, spec and package: relative imports and reloads resolve where it lives
    assert moved.__name__ == moved.__spec__.name == f"oe_inferencex.{new}"
    assert moved.__package__ == f"oe_inferencex.{new}".rpartition(".")[0]


def test_a_name_and_a_patch_through_an_old_path_are_the_moved_module_s(monkeypatch):
    old = importlib.import_module("oe_inferencex.metrics")
    new = importlib.import_module("oe_inferencex.common.metrics")
    for name in vars(new):
        assert getattr(old, name) is getattr(new, name), name
    assert old.excess_aurc is new.excess_aurc is ox.excess_aurc
    monkeypatch.setattr(old, "aurc_expected", lambda *a, **k: "patched")
    assert new.aurc_expected() == "patched"


def test_importing_the_package_loads_no_torch_matplotlib_or_old_path():
    """In a fresh interpreter: the package and its command line import neither torch nor matplotlib, and no old path
    is imported on the way (the package's own code imports the new paths; the finder is for users and tests)."""
    code = ("import sys, oe_inferencex, oe_inferencex.cli\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'matplotlib')\n"
            "             or m.startswith('oe_inferencex.') and m[len('oe_inferencex.'):] in oe_inferencex._MOVED))\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout == "[]\n"


def test_the_public_names_are_those_of_1_18_1():
    assert sorted(ox.__all__) == sorted(ALL_1_18_1) and len(set(ox.__all__)) == len(ox.__all__) == 84

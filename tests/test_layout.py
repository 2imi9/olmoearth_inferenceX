"""The feature-folder layout of 1.19.0 (ADR-002): every 1.18.1 module path still imports, as the moved module itself."""
import importlib
import importlib.util
import subprocess
import sys
import textwrap

import pytest

import oe_inferencex as ox

# The modules that moved into folders: their 1.18.1 path -> their new one.
MOVED = {
    "metrics": "common.metrics", "signals": "common.signals", "stats": "common.stats",
    "assess": "review.assess", "explain": "review.explain", "calibrate": "review.calibrate",
    "olmoearth": "readers.olmoearth", "embeddings": "readers.embeddings", "lcc": "readers.lcc",
    "taskcard": "readers.taskcard",
    "evidence": "research.evidence", "awf": "research.awf", "data": "research.data", "figstyle": "research.figstyle",
    "reliability": "research.reliability",
    "sequential": "certify.sequential", "certificate": "certify.certificate",
    "stratified": "agency.stratified",
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
    # the moved module keeps its own name, spec, package and loader: relative imports and reloads resolve where it lives
    assert moved.__name__ == moved.__spec__.name == f"oe_inferencex.{new}"
    assert moved.__package__ == f"oe_inferencex.{new}".rpartition(".")[0]
    assert moved.__loader__ is moved.__spec__.loader


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


def test_python_m_runs_the_moved_taskcard(monkeypatch, capsys):
    """`python -m oe_inferencex.taskcard` (taskcard's usage line) runs the moved module's main block: in process
    through runpy, and as a command, whose argv[0] is then the moved file, so argparse still calls it taskcard.py."""
    import runpy
    monkeypatch.delitem(sys.modules, "oe_inferencex.taskcard", raising=False)     # not imported yet: see the next test
    origin = importlib.util.find_spec("oe_inferencex.taskcard").origin
    assert origin == importlib.import_module("oe_inferencex.readers.taskcard").__file__
    monkeypatch.setattr(sys, "argv", ["taskcard.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("oe_inferencex.taskcard", run_name="__main__")
    assert exc.value.code == 0 and capsys.readouterr().out.startswith("usage: taskcard.py ")
    run = subprocess.run([sys.executable, "-m", "oe_inferencex.taskcard", "--help"], capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.startswith("usage: taskcard.py "), run.stderr


def test_runpy_refuses_an_old_path_already_imported_and_runs_the_new_one(monkeypatch, capsys):
    """What can differ (CHANGELOG 1.19.0): once imported, an old path is the moved module, whose loader answers to the
    new name only, so runpy in the same process refuses the old name; the new name runs."""
    import runpy
    import warnings
    importlib.import_module("oe_inferencex.taskcard")
    monkeypatch.setattr(sys, "argv", ["taskcard.py", "--help"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # runpy: found in sys.modules before its execution
        with pytest.raises(ImportError, match="cannot handle oe_inferencex.taskcard"):
            runpy.run_module("oe_inferencex.taskcard", run_name="__main__")
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("oe_inferencex.readers.taskcard", run_name="__main__")
    assert exc.value.code == 0 and capsys.readouterr().out.startswith("usage: taskcard.py ")


def test_the_finder_imports_nothing_to_find_and_holds_under_lazy_loading_blocking_and_reload():
    """In a fresh interpreter, the ways other than a plain import that reach the finder: finding an old path imports
    nothing (the import system calls find_spec under its global lock); an old path imported before its new one; the
    standard library's lazy loader; a new path blocked with None, refused as missing; a pickle that names an old path;
    a reload of the package, which keeps one finder."""
    code = textwrap.dedent("""
        import importlib, importlib.util, pickle, sys, types
        import oe_inferencex as ox
        before = set(sys.modules)
        assert importlib.util.find_spec("oe_inferencex.evidence").origin.endswith("evidence.py")
        assert set(sys.modules) == before, sorted(set(sys.modules) - before)

        old = importlib.import_module("oe_inferencex.taskcard")
        new = sys.modules["oe_inferencex.readers.taskcard"]
        assert old is new and new.__name__ == new.__spec__.name == "oe_inferencex.readers.taskcard"
        assert new.__package__ == "oe_inferencex.readers" and new.__loader__ is new.__spec__.loader

        spec = importlib.util.find_spec("oe_inferencex.lcc")
        spec.loader = importlib.util.LazyLoader(spec.loader)
        lazy = importlib.util.module_from_spec(spec)
        sys.modules["oe_inferencex.lcc"] = lazy
        spec.loader.exec_module(lazy)
        real = importlib.import_module("oe_inferencex.readers.lcc")
        assert real.__spec__.name == "oe_inferencex.readers.lcc"         # the first attribute access loads it
        assert lazy is real and type(real) is types.ModuleType
        assert real.__package__ == "oe_inferencex.readers" and real.__loader__ is real.__spec__.loader

        sys.modules["oe_inferencex.research.figstyle"] = None
        try:
            importlib.import_module("oe_inferencex.figstyle")
        except ModuleNotFoundError as exc:
            assert "halted; None in sys.modules" in str(exc), exc
        else:
            raise AssertionError("a blocked module imported")
        assert "oe_inferencex.figstyle" not in sys.modules

        assert pickle.loads(b"coe_inferencex.stats\\nsign_test\\n.") is ox.sign_test
        assert importlib.reload(sys.modules["oe_inferencex.stats"]).__name__ == "oe_inferencex.common.stats"
        importlib.reload(ox)
        assert sum(type(f).__name__ == "_MovedModules" for f in sys.meta_path) == 1
        assert importlib.import_module("oe_inferencex.metrics") is sys.modules["oe_inferencex.common.metrics"]
        print("ok")
        """)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout == "ok\n", run.stderr


def test_the_public_names_are_those_of_1_18_1():
    assert sorted(ox.__all__) == sorted(ALL_1_18_1) and len(set(ox.__all__)) == len(ox.__all__) == 84

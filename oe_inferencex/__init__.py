"""olmoearth_inferenceX: assess a prediction map, and measure how two inferences differ, without labels.

The public surface is what this module exports. Everything here is numpy only: a prediction map in, a review
set and its reasons out, per input condition when the map records one; two maps in, their difference out; and the
statistics that graded every claim in the documentation. Labels are never needed to produce a result, only to grade
one — with one exception that says so in its name: `estimate` turns a labelled sample back into how wrong the map
is, with the interval its sampling design earns, because that question has no label-free answer.

    import numpy as np, oe_inferencex as ox
    out = ox.assess_prediction(scores, is_logit=True)        # (C, H, W) scores -> review sets, cues, operating points
    ox.summary(out)
    diff = ox.compare_inferences(a, b, ok)                    # two decision maps on one grid -> how much, where, stable?
    s = ox.sample_for_estimation(margin, 300, p1=p1)          # which 300 windows to label
    ox.estimate_error_rate(s, wrong)                          # ... -> the error rate, with a 95% interval

Folders, one per feature, each with its logic and its commands, in the order a user meets them:
  review      assess: a prediction or served class map -> review sets at budgets, boundary-first order, reference
              scoring, and with an input-condition layer (a cloud flag, the modalities present) each condition on its
              own; explain: why each review window is suspect, label-free cues with their measured enrichment;
              calibrate: fuse the readings with labels, a ranker or side rule, held out, bound to its model family;
              and the window readers the labelled commands share
  compare     how two inferences of the same scene differ: disagreement, where it sits, stability, the label bridge
  scale       one map at several window sizes
  estimate    how wrong the map is, from a labelled sample: which windows to label, the design-based interval,
              per-class accuracy; with an input condition, labels split equally across the conditions and each
              condition's exact interval (exp88)
  certify     a certified zone of the map that is wrong at most alpha of the time, per condition with delta split
              over them (zone, sequential), and the certificate that carries it (certificate)
  plan        how many labels each route needs
  decide      typed answers read from a result that estimate, certify or compare wrote
  agency      an agency's stratified reference sample (NLCD, GFC2020) -> overall, user's and producer's accuracy
              and class shares of one map or two, each with an interval valid at any sample size (stratified: the
              betting intervals, the box for ratios, the pure-stratum rule) and the normal-theory one beside it
  readers     taskcard: what each OlmoEarth fine-tuned model is; lcc: a range reader for the served rasters;
              olmoearth: an OlmoEarth run's rslearn dataset -> the scores assess reads and an input-condition layer
              on their grid; embeddings: a geo-embeddings store and labelled points -> class probabilities
  common      the helpers two or more features share. metrics: tie-aware AURC and excess AURC, capture at a budget, the
              attainable ceiling, design-weighted forms; stats: exact sign tests, one-vote-per-cluster tests, block
              and cluster bootstraps; signals: the confidence readings and the no-model pixel controls every rule is
              scored against; the exact hypergeometric arithmetic; the commands' raster, output and wording helpers
  research    the research record's toolkit. reliability: the ensemble and typicality signals that were tested and
              lost, kept so they can be re-tested; evidence, awf, data, figstyle
  cli         `oe-inferencex demo`, `assess`, `compare`, `sample`, `estimate`, `certify` and `grade`: rasters, .npy or
              point tables in, files out
  mcp_server  the same commands as tools of a local MCP server for an agent, `oe-inferencex mcp`
The layout is feature folders (docs/plan/adr-002-feature-folders.md in the research record). The 1.18.1 module paths
(oe_inferencex.assess, oe_inferencex.metrics and the rest) still import, each as the module that now holds its code.
By dependency the modules form three groups:
  numpy only        review, compare, estimate, agency, common.metrics, common.stats, common.signals,
                    research.reliability, cli
  standard library  readers.taskcard, readers.lcc
  guarded extras    research.evidence (torch); research.awf and research.data (the "encoder" and "geo" extras);
                    readers.olmoearth's readers (rasterio, the "geo" extra; the module imports with numpy alone);
                    research.figstyle (matplotlib); mcp_server's server (the "mcp" extra; its texts and tool
                    functions are numpy only). Not imported here; each names its extra if imported without it.

The evidence behind each function is in the research record, one claim id per number:
https://github.com/2imi9/olmoearth_inferenceX-record.
The paths under docs/, exp/ and scripts/ that docstrings and comments here name are in it, apart from
docs/Usage.md and docs/index.md, the documentation: https://olmoearth-inferencex.readthedocs.io
"""
from importlib.metadata import PackageNotFoundError, version as _version
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

from .review.assess import (assess_classmap, assess_prediction, boundary_first_score, pool_condition, review_mask,
                            review_order, summary)
from .agency import design_from_sizes, design_from_weights, grade, read_points
from .review.calibrate import fit_ranker, fit_side, side_features
from .estimate import (certify_by_condition, cluster_interval, confidence_strata, design_effect, equal_allocation, estimate_error_rate, estimate_from_indices, exact_coverage_srs,
                       hypergeom_interval,
                       neyman_allocation, review_set_check, sample_for_estimation, stratified_interval, stratified_interval_wilson, stratified_mean_and_variance,
                       wilson_interval)
from .compare import (compare_inferences, crosstab, determinism_check, disagreement, over_groups, phi, stability,
                      where, which_side)
from .review.explain import CUES, cue_enrichment, derive_cues, explain_review_set, library_table
from .common.metrics import (attainable_ceiling, augrc, augrc_from_auroc, aurc_expected, capture_at_budget, capture_at_budget_expected, excess_aurc,
                             expected_calibration_error, oracle_aurc, risk_coverage, selective_accuracy,
                             weighted_aurc, weighted_auroc, weighted_capture_at_budget, weighted_excess_aurc,
                             weighted_mean)
from .common.signals import (aligned_tile_phase, boundary_indicator, combine_midrank, confidence, crop_dependence,
                             midrank_pct, ndwi, ndwi_gradient, ndwi_level, pool_to_windows, s2_patch_variance,
                             shift_averaged_probability)
from .agency.stratified import betting_interval, pure_stratum_ratio
from .common.stats import (block_bootstrap_indices, cluster_bootstrap_difference, clustered_sign_test,
                           paired_cluster_bootstrap, paired_comparison, rank_sum_test, sign_test, spearman, wins_losses_ties)
# The module attributes the 1.18.1 imports above bound as a side effect (`ox.metrics` after `import oe_inferencex`)
from .common import metrics, signals, stats  # noqa: F401
from .review import assess, calibrate, explain  # noqa: F401
from .agency import stratified  # noqa: F401

try:
    __version__ = _version("olmoearth-inferencex")
except PackageNotFoundError:          # a source checkout that was never installed
    __version__ = "1.19.0"

__all__ = [
    "crop_dependence", "determinism_check",
    "__version__",
    # assess
    "assess_prediction", "assess_classmap", "review_order", "review_mask", "boundary_first_score", "summary",
    "pool_condition",
    # explain
    "explain_review_set", "derive_cues", "cue_enrichment", "library_table", "CUES",
    # compare
    "compare_inferences", "disagreement", "where", "stability", "over_groups", "crosstab", "which_side", "phi",
    # calibrate
    "fit_ranker", "fit_side", "side_features",
    # estimate: how wrong the map is, from a labelled sample
    "sample_for_estimation", "estimate_error_rate", "estimate_from_indices", "review_set_check",
    "wilson_interval", "hypergeom_interval", "exact_coverage_srs", "stratified_interval", "stratified_interval_wilson", "stratified_mean_and_variance", "cluster_interval", "design_effect", "confidence_strata", "neyman_allocation",
    "equal_allocation", "certify_by_condition",
    # agency: an agency's stratified sample graded, with intervals valid at any sample size
    "grade", "read_points", "design_from_weights", "design_from_sizes", "betting_interval", "pure_stratum_ratio",
    # metrics
    "excess_aurc", "aurc_expected", "risk_coverage", "oracle_aurc", "capture_at_budget", "attainable_ceiling", "augrc", "augrc_from_auroc",
    "capture_at_budget_expected", "selective_accuracy", "expected_calibration_error",
    "weighted_mean", "weighted_aurc", "weighted_excess_aurc", "weighted_capture_at_budget", "weighted_auroc",
    # stats
    "sign_test", "wins_losses_ties", "paired_comparison", "clustered_sign_test", "cluster_bootstrap_difference",
    "paired_cluster_bootstrap", "block_bootstrap_indices", "spearman", "rank_sum_test",
    # signals
    "confidence", "boundary_indicator", "aligned_tile_phase", "shift_averaged_probability", "pool_to_windows",
    "ndwi", "ndwi_gradient", "ndwi_level", "s2_patch_variance", "midrank_pct", "combine_midrank",
]

# The 1.18.1 module paths that 1.19.0 moved into feature folders (ADR-002), old name -> new path. Importing an old
# path gives the moved module itself, not a copy: `oe_inferencex.metrics is oe_inferencex.common.metrics`, so private
# names, monkeypatches and module state are the one module's. Nothing is imported before an old path is asked for
# (evidence still needs torch only then). The package's own code imports the new paths.
_MOVED = {
    "metrics": "common.metrics", "signals": "common.signals", "stats": "common.stats",
    "assess": "review.assess", "explain": "review.explain", "calibrate": "review.calibrate",
    "olmoearth": "readers.olmoearth", "embeddings": "readers.embeddings", "lcc": "readers.lcc",
    "taskcard": "readers.taskcard",
    "evidence": "research.evidence", "awf": "research.awf", "data": "research.data", "figstyle": "research.figstyle",
    "reliability": "research.reliability",
    "sequential": "certify.sequential", "certificate": "certify.certificate",
    "stratified": "agency.stratified",
}


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds `oe_inferencex.<old>` of _MOVED and loads it as the module it moved to."""

    def __init__(self):
        self._own = {}      # a moved module's name -> its own spec, package and loader, which exec_module puts back

    def _file(self, tail):
        """The moved module's file, located without importing anything, not even its folder: find_spec runs under
        the import system's global lock, where importing a package could deadlock against another thread."""
        folder = _MOVED[tail].rpartition(".")[0]
        where = [os.path.join(entry, *folder.split(".")) for entry in __path__]
        return importlib.machinery.PathFinder.find_spec(f"{__name__}.{_MOVED[tail]}", where)

    def find_spec(self, fullname, path=None, target=None):
        head, _, tail = fullname.rpartition(".")
        if head != __name__ or tail not in _MOVED:
            return None
        moved = self._file(tail)                         # its origin is argv[0] under -m
        return None if moved is None else importlib.util.spec_from_loader(fullname, self, origin=moved.origin)

    def create_module(self, spec):
        target = f"{__name__}.{_MOVED[spec.name.rpartition('.')[2]]}"
        module = importlib.import_module(target)    # at once, also under a lazy loader: the old path is this module
        if module.__spec__.name == target:          # its own spec, not one an unfinished old-path load left on it
            self._own[target] = module.__spec__, module.__package__, module.__loader__
        return module       # the import system now sets the old path's spec on it, and a lazy loader its loader

    def exec_module(self, module):
        """Gives the moved module back its own name, spec, package and loader, which the import system set to the
        old path's (a reload or the old `load_module` overwrites the name and package too)."""
        old = module.__spec__.name
        target = f"{__name__}.{_MOVED[old.rpartition('.')[2]]}"
        if target not in self._own:
            raise ImportError(f"{old} moved to {target}, whose own spec this finder never saw", name=old)
        module.__spec__, module.__package__, module.__loader__ = self._own[target]
        module.__name__ = target

    def get_code(self, fullname):          # so `python -m oe_inferencex.taskcard` still runs its main block
        tail = fullname.rpartition(".")[2]
        return self._file(tail).loader.get_code(f"{__name__}.{_MOVED[tail]}")


if not any(type(finder).__module__ == __name__ for finder in sys.meta_path):     # once, also after a reload
    sys.meta_path.insert(0, _MovedModules())

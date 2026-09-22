"""olmoearth_inferenceX: assess a prediction map, and measure how two inferences differ, without labels.

The public surface is what this module exports. Everything here is numpy only: a prediction map in, a review
set and its reasons out; two maps in, their difference out; and the statistics that graded every claim in the
documentation. Labels are never needed to produce a result, only to grade one — with one exception that says so
in its name: `estimate` turns a labelled sample back into how wrong the map is, with the interval its sampling
design earns, because that question has no label-free answer.

    import numpy as np, oe_inferencex as ox
    out = ox.assess_prediction(scores, is_logit=True)        # (C, H, W) scores -> review sets, cues, operating points
    ox.summary(out)
    diff = ox.compare_inferences(a, b, ok)                    # two decision maps on one grid -> how much, where, stable?
    s = ox.sample_for_estimation(margin, 300, p1=p1)          # which 300 windows to label
    ox.estimate_error_rate(s, wrong)                          # ... -> the error rate, with a 95% interval

Modules, in the order a user meets them:
  assess      a prediction or served class map -> review sets at budgets, boundary-first order, reference scoring
  explain     why each review window is suspect: label-free cues with their measured enrichment
  compare     how two inferences of the same scene differ: disagreement, where it sits, stability, the label bridge
  calibrate   fuse the readings with labels: a ranker or side rule, held out, bound to its model family
  estimate    how wrong the map is, from a labelled sample: which windows to label, and the design-based interval
  metrics     tie-aware AURC and excess AURC, capture at a budget, the attainable ceiling, design-weighted forms
  stats       exact sign tests, one-vote-per-cluster tests, block and cluster bootstraps
  signals     the confidence readings and the no-model pixel controls every rule is scored against
  reliability the ensemble and typicality signals that were tested and lost, kept so they can be re-tested
  cli         `oe-inferencex assess`, `compare`, `sample` and `estimate`: rasters or .npy in, files out
  taskcard    what each OlmoEarth fine-tuned model is; lcc: a range reader for the served rasters
By dependency the modules form three groups, and the layout stays flat (docs/plan/adr-001-repository-layout.md):
  numpy only        assess, explain, compare, calibrate, estimate, metrics, stats, signals, reliability, cli
  standard library  taskcard, lcc
  guarded extras    evidence (torch); awf and data (the "encoder" and "geo" extras); figstyle (matplotlib).
                    Not imported here; each names its extra if imported without it.

The evidence behind each function is in the documentation, one claim id per number:
https://olmoearth-inferencex.readthedocs.io
"""
from importlib.metadata import PackageNotFoundError, version as _version

from .assess import (assess_classmap, assess_prediction, boundary_first_score, review_mask, review_order,
                     summary)
from .calibrate import fit_ranker, fit_side, side_features
from .estimate import (cluster_interval, confidence_strata, design_effect, estimate_error_rate, estimate_from_indices, exact_coverage_srs,
                       neyman_allocation, review_set_check, sample_for_estimation, stratified_interval, stratified_interval_wilson, stratified_mean_and_variance,
                       wilson_interval)
from .compare import (compare_inferences, crosstab, determinism_check, disagreement, over_groups, phi, stability,
                      where, which_side)
from .explain import CUES, cue_enrichment, derive_cues, explain_review_set, library_table
from .metrics import (attainable_ceiling, augrc, augrc_from_auroc, aurc_expected, capture_at_budget, capture_at_budget_expected, excess_aurc,
                      expected_calibration_error, oracle_aurc, risk_coverage, selective_accuracy,
                      weighted_aurc, weighted_auroc, weighted_capture_at_budget, weighted_excess_aurc,
                      weighted_mean)
from .signals import (aligned_tile_phase, boundary_indicator, combine_midrank, confidence, crop_dependence,
                      midrank_pct, ndwi, ndwi_gradient, ndwi_level, pool_to_windows, s2_patch_variance,
                      shift_averaged_probability)
from .stats import (block_bootstrap_indices, cluster_bootstrap_difference, clustered_sign_test,
                    paired_cluster_bootstrap, paired_comparison, rank_sum_test, sign_test, spearman, wins_losses_ties)

try:
    __version__ = _version("olmoearth-inferencex")
except PackageNotFoundError:          # a source checkout that was never installed
    __version__ = "1.2.0"

__all__ = [
    "crop_dependence", "determinism_check",
    "__version__",
    # assess
    "assess_prediction", "assess_classmap", "review_order", "review_mask", "boundary_first_score", "summary",
    # explain
    "explain_review_set", "derive_cues", "cue_enrichment", "library_table", "CUES",
    # compare
    "compare_inferences", "disagreement", "where", "stability", "over_groups", "crosstab", "which_side", "phi",
    # calibrate
    "fit_ranker", "fit_side", "side_features",
    # estimate: how wrong the map is, from a labelled sample
    "sample_for_estimation", "estimate_error_rate", "estimate_from_indices", "review_set_check",
    "wilson_interval", "exact_coverage_srs", "stratified_interval", "stratified_interval_wilson", "stratified_mean_and_variance", "cluster_interval", "design_effect", "confidence_strata", "neyman_allocation",
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

"""How wrong is the map? A design-based error rate from a labelled sample, and which windows to label.

The label-free layers rank windows for review and never say how wrong a map is, because that needs a reference.
This module is what a reference costs, measured in exp78 on the seven segmentation tasks of Ai2's suite with every
unit labelled so the intervals could be graded (the research record's docs/results/comparisons.md, exp78): label 300
windows drawn by a design and the error rate comes back with an interval that covers the truth on 0.93 to 0.96 of
draws, about +/-3 points on a clean map and +/-5 on a messy one.

Three things a user needs, and one they must be stopped from doing.

- `sample_for_estimation` says which windows to label. The default design stratifies the map by confidence
  margin into quintiles and allocates the budget by Neyman's rule from the model's own confidence, so no label is
  spent estimating stratum rates; on exp78's tasks it narrowed the interval to a median 0.80 of a random sample's
  and to 0.63 on the cleanest map, with coverage intact everywhere. A plain random sample and a tile design are
  also offered, the last because that is how people actually label. When the map records each window's input
  condition (a cloud flag, the modalities present), the "condition" design splits the labels equally across the
  conditions, never from the model's confidence. That confidence can overstate the accuracy of a condition read
  from an input combination the model was not trained on: it did on PASTIS for a probe trained on radar plus optical
  and run on radar alone, though not on CropHarvest China 6 (exp88). Even a model trained on each input can be wrong
  more often in one condition than in another. Each condition then gets its own exact interval, and
  `certify_by_condition` a zone of its own.
- `estimate_error_rate` turns the labels back into a rate with the interval the design earns: the exact
  hypergeometric interval for a random sample, the weighted sum of each condition's exact interval under the
  condition design, a stratified interval under the confidence and proportional designs, and for tile-sampled
  labels a ratio estimator with its ultimate-cluster interval, beside the naive one so the difference is visible.
  That last is the weakest of the three and says so: as shipped, it covered 94.5% to 95.4% of the time on five of
  exp78's tasks, 84.3% on Sen1Floods11 and 68.5% on MADOS, the two where a tenth of the tiles hold most of the errors
  (exp/out/exp78_shipped_tiles.json in the research record). Labelling 18 tiles of 16 windows and using the ordinary
  formula gave a "95%" interval that covered on 0.51 to 0.78 of draws; that is exp78's practical finding and the
  reason this function will not compute the naive interval alone. With valid=True (`estimate --valid`, 1.20.0,
  estimate/valid.py) it adds, beside the interval the design earns and without changing it, one valid at every sample
  size under every design but tiles; `estimate_per_class(valid=True)` does the same per class.
- `estimate_from_indices` is for windows labelled without a design. It treats them as a random sample after
  checking only that they do not look like a review list: the mean suspicion percentile of a random sample is 0.5,
  of the tool's own review set about 0.97, and labelling the review set then dividing gives two to six times the
  true rate on every task of exp78's export. A sample more than four standard deviations above a random one is
  refused with the number rather than estimated. The check does not make the labels a random sample: labels from
  the confident end of the map, or clustered by tile, pass it, and the interval is then not valid.

Every function is numpy only. The estimators are the ones exp78 ran; that script imports them from here.
"""
import math

import numpy as np

from oe_inferencex.common.hypergeom import (EXACT_N_MAX, _TIE, _at_most, _float_above, _float_below, _hyper_tail,
                                            _hyper_tail_exact, _hypergeom_fractions, _hypergeom_interval, _level,
                                            _log_choose, _tie, hypergeom_cdf, hypergeom_interval)
from oe_inferencex.common.sampling import (REVIEW_SET_SIGMAS, UNRECORDED, _condition_guards, _condition_index,
                                           review_set_check, review_set_threshold)
from oe_inferencex.common.readings import _also_accept_pairs, _code, also_accept_reading
from oe_inferencex.estimate.designs import (CLASS_BLIND_NOTE, CLASS_FLOOR, CONFIDENCE_REFUSAL, MIN_PER_STRATUM, MIN_TILES,
                                            M_PER_TILE, N_STRATA, Q_FLOOR, class_allocation, confidence_strata,
                                            draw_stratified, equal_allocation, neyman_allocation, sample_for_estimation)
from oe_inferencex.estimate.intervals import (Z95, _T975, _union_interval, cluster_interval, design_effect,
                                              exact_coverage_srs, stratified_interval, stratified_interval_wilson,
                                              stratified_mean_and_variance, t_quantile_975, wilson_interval)
from oe_inferencex.estimate.error_rate import (CLASS_WHOLE_MAP, CONDITION_NOTE, CONDITION_WHOLE_MAP, ESTIMATE_READING_NOTE,
                                               REVIEWER_NOTE, SCOPE_ESTIMATE, SEQUENTIAL_READ_NOTE, TILES_WARNING,
                                               UNJUDGED_NOTE, UNJUDGED_STRATIFIED_NOTE, _estimate_core, _per_condition,
                                               _stratum_union, estimate_error_rate, estimate_from_indices)
from oe_inferencex.estimate.model_assisted import (MIN_FOR_TUNING, model_assisted_interval,
                                                   stratified_model_assisted_interval, tuned_coefficient)
from oe_inferencex.estimate.per_class import (CLASS_DESIGN_EXACT_NOTE, CLASS_DESIGN_METHOD, CLASS_NORMAL_NOTE, MIN_PER_CLASS,
                                              PER_CLASS_EXACT_NOTE, PER_CLASS_NOT_GRADED, RARE_ERRORS, _class_design_exact,
                                              _ht_ratio, _ht_total, _interval, _wald, _wilson_eff, estimate_per_class)
from oe_inferencex.estimate.valid import (PER_CLASS_VALID_NOTE, VALID_HELP, VALID_NOTE, class_quantities, linear_valid,
                                          per_class_valid, producer_box, whole_map_valid, worst_work)
from oe_inferencex.certify.zone import (FAMILY_NOTE, FAMILY_NOTE_ONE, ISSUED_NOTE, PREFIX_NOTE, RULES_OF, SCOPE_CERTIFY,
                                        TARGET_ALONE_NOTE, TARGET_NOTE, TARGET_WEIGHT, ZONE_CUT, ZONE_CUTS, ZONE_DELTA,
                                        ZONE_GRID, ZONE_READING_NOTE, ZONE_READING_SEQUENTIAL, ZONE_RULES, _edge_of,
                                        _family_note, apply_zone_rule, certify_by_condition, certify_zone,
                                        min_labels_to_certify, null_count, target_weights, zone_counts, zone_cut,
                                        zone_edge, zone_levels, zone_order, zone_pvalue, zone_pvalue_exact,
                                        zone_upper_bound)
from oe_inferencex.estimate.which_map import (SIGN_TEST_NOTE, SIGN_TEST_SHARE, WHICH_MAP_NOTE, _SIGN_CRIT,
                                              compare_from_disagreement, sample_disagreement, sign_critical,
                                              which_map_rule)
from oe_inferencex.estimate.audit import (AUDIT_METHOD, AUDIT_NOTE, AUDIT_READING_NOTE, AUDIT_SHARE, AUDIT_ZONE_TEXT,
                                          _audit_zone, class_audit_from_sample, interval_verdict, sample_class_audit)
from oe_inferencex.estimate.profiles import (MAX_BANDS, MAX_GRID, MAX_PROFILES, MIN_PROFILE_WINDOWS,
                                             PROFILE_ALLOCATION_NOTE, PROFILE_BLIND_NOTE, PROFILE_RULE_TEXT,
                                             build_profiles, parse_profile_by, profile_spec)
from oe_inferencex.certify.localized import (PROFILE_METHOD, VERDICT_CODES, profile_interval, profile_verdict,
                                             profile_verdicts)

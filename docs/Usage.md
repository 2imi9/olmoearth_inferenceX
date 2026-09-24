# Usage

This page describes the inputs the package accepts, the `oe-inferencex` commands and the Python functions behind
them. The evidence for each method is in [Findings](Findings.md) and, per experiment, in
[Comparisons](results/comparisons.md).

## Installation

```bash
pip install olmoearth-inferencex            # the package; no torch
pip install "olmoearth-inferencex[geo]"     # with GeoTIFF input and output
```

The package requires Python 3.11 to 3.13. Without the `geo` extra, which brings rasterio, the commands read and write `.npy` arrays.

## Inputs

A map is a GeoTIFF or `.npy` array of the scores a model's classification head produces before the argmax: `(H, W)`
probabilities or logits for a binary task, `(C, H, W)` per class otherwise. Logits are preferred, because
probabilities tie where they saturate; pass `--logits` (`is_logit=True` in Python). Other maps are accepted with
restrictions.

| Map | Accepted by |
|---|---|
| Hard class map with an exported confidence band | `assess_classmap` (Python); the band ranks only the pixels it separates |
| Hard class map alone | `compare` only; without a confidence there is no ranking |
| Continuous map, such as a regression output | `compare` only, at a cut-off named with `--threshold`; no recorded experiment grades this case |

The package works on square windows of `--patch` pixels (default 4). A window's confidence is the mean confidence
margin of its valid pixels and its class the majority class of its pixels; a window less than half valid is left
out. No-data comes from the raster's no-data value, from NaN or from `--nodata`.

The package refuses what it cannot rank, in Python as on the command line. Probability input outside [0, 1] raises
an error rather than being cut at 0.5 and scored. A review set whose cut-off falls inside a run of equal scores (a
hard mask, a quantized band, a constant map) carries `tied_at_cutoff` and a warning, since the order among the tied
windows is raster position. Without labels, the package does not report how wrong a map is; that requires a labelled
sample (`sample`, then `estimate` or `certify`).

## Command line

`oe-inferencex <command> --help` lists each command's options. Outputs are JSON summaries, CSV tables of windows with
pixel and map coordinates, and rasters on the window grid (GeoTIFF for a GeoTIFF input, `.npy` otherwise).

### demo

`oe-inferencex demo` assesses the expert-annotated Dynamic World tile shipped with the package and prints the share
of its errors each review budget contains. It writes `review_set.png`, the `assess` files (`audit/`) and the tile
(`sample_probabilities.npy`, `sample_truth.npy`) to `oe_inferencex_demo/`. `--made-up` uses a synthetic map.

### assess

`assess` ranks the windows of one map by the model's confidence, least confident first, and writes the review set at
each budget with the cues behind each window. On the demo tile:

```console
$ oe-inferencex assess oe_inferencex_demo/sample_probabilities.npy --patch 1 --out audit
16380 windows of 1 px; review sets 1%: 164, 5%: 819, 10%: 1638; boundary windows 36.8%
wrote audit/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary
```

| File | Content |
|---|---|
| `assessment.json` | Window count, confidence quantiles, boundary share, `warnings`; per budget, the review set and `tied_at_cutoff` |
| `review_set_05pct.csv` | One per budget (`--budgets`, default 0.01 0.05 0.10): rank, window, pixel and map coordinates, confidence, boundary |
| `explanation.json` | The cues of each review window, each cue's measured enrichment (`quotes`), the windows no cue explains |
| `suspicion.tif`, `boundary.tif` | The ranking score and the boundary indicator |

`--order boundary_first` reviews the windows on a class boundary first, then the interior, each by confidence; the
[Recipe](method/recipe.md) states when to use it. `--reference labels.tif`, an integer class raster on the same grid,
adds `against_reference`: the error rate, the errors captured at each budget, the AURC beside the random and oracle
baselines, and the caveat that another product's map used as a reference can flatter boundary-type signals
([exp18](results/comparisons.md#dense-flood-masks-sen1floods11-exp18)).

### compare

`compare` measures where two maps of the same area differ and, with labels, which map is right where they differ.

```bash
oe-inferencex compare a.tif b.tif --out diff --labels reference.tif --groups tiles.tif
```

It writes `comparison.json` (`disagreement_rate`, `per_group` for the ids in `--groups`, `where` for the boundary
cue's enrichment among the differing windows, `dates`, `notes`), `differing_windows.csv` and `disagreement.tif` (NaN
where nothing was compared). `--labels` adds `graded`, computed over the windows with a majority label: `which_side`,
how often each map matches the label where they differ, and `crosstab`, the errors of `a` that `b` corrects, those it
adds and those both make.

Class maps are read as integers, a single-band probability map is cut at `--threshold` (0.5 when omitted), and a
multi-band score map is argmaxed; a continuous map outside [0, 1] needs `--threshold`. The maps, `--labels` and
`--groups` must share one grid; another CRS or origin is refused. Windows are compared where both maps predicted. A
window whose pixels split evenly between two classes goes to the more confident pixels, or is left out and noted
when either map is a hard class map.

**Dates.** Where two maps of the same ground at the same time differ, at least one is wrong; across dates the
difference can be real change on the ground. `--date-a` and `--date-b` take a date `YYYY-MM-DD` or a period
`YYYY-MM-DD/YYYY-MM-DD` (a composite or an annual map), and `dates.reading` states what a difference can mean. When
the dates differ, grading is refused until `--labels-date` gives the labels' date, because a window that changed is
right in one map and wrong in the other whatever either model did; `graded.graded_against` then names the map the
labels match in time.

### sample and estimate

`sample` selects the windows to label; `estimate` reads the labels back and reports the map's error rate with a 95%
interval.

```bash
oe-inferencex sample water_prob.tif --budget 300 --out to_label.csv
# the reviewer fills the `wrong` column with 1 or 0 for each window
oe-inferencex estimate to_label.csv
oe-inferencex estimate to_label.csv --per-class    # once a `reference_class` column is filled in as well
```

`sample` writes the CSV (`index`, window, pixel and map coordinates, `stratum`, `confidence`, `map_class`, an empty
`wrong`) and a sidecar `to_label.json` with the design, the scores' path and the grid. `map_class` is the majority
class of the window's pixels. **The reviewer sets `wrong` to 1 when `map_class` is not what is on the ground, and to 0
otherwise**; `estimate` grades that class, not the window's centre pixel. The recorded experiments used 300 windows
([exp78](results/comparisons.md#how-wrong-is-this-map-what-a-reviewers-labels-buy-exp78)).

| `--design` | Draw | Interval in `estimate` |
|---|---|---|
| `confidence` (default) | Strata by confidence margin, budget allocated from the model's confidence | Wilson at the design's effective sample size |
| `proportional` | The same strata, budget allocated by stratum size | The same |
| `random` | Simple random sample; required by `certify` | Exact hypergeometric |
| `tiles` | `--per-tile` windows (default 16) in each of a random set of tiles of `--tile` windows per side (default 16) | Ratio estimator over tiles, with the naive interval beside it |

The tile design matches how reviewers often label, and its intervals under-cover: on exp78's tasks the naive one
covered the true rate 51 to 78% of the time at a nominal 95%, the corrected one 91 to 94% with tiles of equal size and
60% with tiles of 1 to 400 windows. If labelling has not started, use the default design.

`estimate` writes `to_label_estimate.json` (`estimate`, `low`, `high`, `half_width`, `effective_n`, `method`). It
refuses a blank `wrong`, a `wrong` other than 0 or 1, and rows other than those the design drew.

`--per-class` needs a `reference_class` column (the class the reviewer saw, in the map's class ids) and the map's
scores, from the sidecar's path or `--scores`; `--scores` and `--nodata` are accepted only with `--per-class`. It adds
`confusion_counts`, `overall_accuracy` and, per class, the user's accuracy (the share of windows mapped as the class
that belong to it), the producer's accuracy (the share of windows of the class mapped as it) and `reference_share`, the
error-adjusted share of the map, each with an interval. A class is flagged for fewer than 30 labels (`few labels`), one
to four sampled errors (`few errors`), nearly all windows labelled (`near census`), thin sampling (`thin strata`), or
no window predicted (`never predicted`, producer's accuracy 0).

**The review set is not a sample.** It is selected to contain errors, so its error rate overstates the map's (1.8 to
5.8 times on exp78's tasks). `sample` draws windows with weights the estimator undoes; the review set has none.

### certify

`certify` returns the largest share of the map, from the most confident window down, whose error rate is at most
`--alpha`, certified by exact hypergeometric tests so that the statement fails on at most `--delta` (default 0.1) of
samples. Outside that zone nothing is certified.

```bash
oe-inferencex sample water_prob.tif --budget 300 --design random --out random.csv
oe-inferencex certify random.csv --alpha 0.05    # after the reviewer has filled `wrong`
```

It writes `random_zone.json` (`coverage`, the certified share; `threshold`, the confidence margin at its edge;
`upper_bound` on its error rate; `note`) and the zone as a window mask, `random_zone.npy`.

- A stratified or tile sample is refused, because the guarantee rests on the labels inside each candidate zone being
  a random sample of that zone.
- `--rule prefix` (default) assumes the zone's error rate does not fall as the zone grows; `--rule bonferroni` assumes
  nothing.
- When the budget cannot certify the level asked for, `certify` says so. With no error among its labels a zone needs
  about `ln(δ) / ln(1 − α)` labels (`min_labels_to_certify`): 45 at α = 5% and 255 at α = 0.9%, for δ = 0.1.
- The confidence is recomputed from the scores the sidecar names (`--scores` if the raster has moved) and checked
  against the CSV; another raster, or another `--nodata`, is refused.

## Python API

The commands wrap functions that take and return arrays. Each result is a dict whose arrays stay in `out["arrays"]`;
`summary(out)` is the JSON-safe view that crosses a tool boundary under the
[agent contract](method/agent_integration.md).

### Assess a prediction and explain the review set

```python
import numpy as np
from oe_inferencex.assess import assess_prediction, summary
from oe_inferencex.explain import explain_review_set
from oe_inferencex.signals import ndwi_level

logits = np.load("water_logits.npy")        # (H, W) binary logits, or (C, H, W) per class
s2 = np.load("s2_window.npy")               # (12, H, W) Sentinel-2 bands, for the spectral cue

out = assess_prediction(logits, is_logit=True, patch=4, budgets=(0.05, 0.10), order="boundary_first")
why = explain_review_set(out, cues={"ndwi_ambiguous": ndwi_level(s2, patch=4) > -0.1})

summary(out)["review_sets"]["0.05"]         # the windows to review first
why["budgets"][0.05]["windows"][0]          # {'row': 28, 'col': 26, 'confidence': 0.48, 'cues': ['boundary', 'low_confidence', 'ndwi_ambiguous']}
why["quotes"]["ndwi_ambiguous"]             # "is spectrally ambiguous ... (48% of error windows vs 7% of correct ones, 7.2x; ...)"
```

- `form="top1"` ranks a multi-class logit map by one minus the top probability; the default remains the logit margin,
  with a warning on such maps ([exp76](results/comparisons.md#which-confidence-which-statistic-which-aggregator-exp76)).
- `explain_review_set` derives the boundary and low-confidence cues itself. Other cues are boolean arrays of window
  shape, such as spectral ambiguity (above) or tiling instability (`signals.aligned_tile_phase`); cues in
  `explain.CUES` are quoted with their measured enrichment, other names as unmeasured.
- `assess_classmap(hard_map, confidence_band, n_classes, ...)` takes a class map with an exported confidence band.
  The served change rasters export no class confidence; there the boundary is the only cue, and `lcc` reads them by
  HTTP range request.
- `reference=` (class labels, negative for none) adds `against_reference`. Grade on expert labels; never train a rule
  on them.

### Compare two inferences of the same scene

![The two-period, two-sensor square on one GEOID-Flood chip: four dated inputs, four inferences on identical windows, the same-date and same-sensor differences on the scene, and the label bridge boxed apart](figures/compare.png)

*Four dated inputs of one GEOID-Flood chip read on identical windows, with the same-date and same-sensor differences;
the labels are set apart
([exp57](results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57),
[exp62](results/comparisons.md#the-fourth-cell-post-event-optical-from-worldfloods-completes-the-square-exp62)).*

```python
import numpy as np
from oe_inferencex.compare import compare_inferences

z = np.load("exp/out/exp57_masks.npz")                               # two heads' decisions on 55 GEOID-Flood events
ev = z["geoid_event"] == "EMSR275-1"
a, b, ok = z["geoid_s2"][ev], z["geoid_s1"][ev], z["geoid_ok"][ev]    # pre-event S2 head, post-event S1 head, windows both predicted
out = compare_inferences(a, b, ok, labels=z["geoid_y_after"][ev])   # graded on water after the event
out["disagreement_rate"]                                            # 0.094: 3,310 of 35,084 windows
out["graded"]["which_side"]["share_b_right"]                        # 0.79: the S1 head matches the post-event label on 79% of them
flooded = z["geoid_y_after"][ev] & ~z["geoid_y_permanent"][ev]
flooded[out["arrays"]["disagree"]].mean()                          # 0.77: the flood's share of the difference
out["dates"]["status"]                                              # "unstated": no dates were given
```

The heads read different dates, so 0.79 is agreement with the post-event label, and the pre-event map is counted wrong
wherever the flood changed the ground. With `dates=(pre, post)`, `compare_inferences` raises until `labels_date=post`
is given, and `graded["graded_against"]` then records which map the labels match in time.

```python
import numpy as np
from oe_inferencex.assess import summary
from oe_inferencex.compare import compare_inferences
from oe_inferencex.signals import boundary_indicator

z = np.load("exp/out/exp02_cache.npz")      # Nano and Base water probabilities on 1024 windows of Kazungula
a, b = z["p_nano"] > 0.5, z["p_base"] > 0.5  # hard decisions first; the module takes no threshold
lab = z["ev_labels"].astype(bool)           # hand labels on every window
ok = np.ones_like(a)                        # the windows both predicted

out = compare_inferences(a, b, ok, labels=lab, cues={"boundary": boundary_indicator(a) > 0})
out["disagreement_rate"]                    # 0.027: 28 of 1024 windows
out["where"]["boundary"]["enrichment"]      # 5.5x: 86% of differing windows on a boundary of a, 15% of agreeing ones
out["graded"]["which_side"]                 # {'n_disagree': 28, 'a_right': 11, 'b_right': 17, 'neither': 0, ...}
out["graded"]["crosstab"]["phi"]            # 0.43: the two error maps overlap
summary(out)                                # the JSON-safe view, no arrays
```

- Decisions are hard maps; a float map is refused. Everything outside `graded` is label-free.
- `groups=` (an id per window) adds per-group rates and, with labels, `over_groups` of the net correction with a
  one-sided exact sign test; `stability` gives the pairwise phi of several disagreement masks.
- Which side is right where two maps differ is not readable without labels; the module reports both
  ([exp58](results/comparisons.md#is-the-comparison-tool-better-than-the-raw-diff-exp58)).
- `determinism_check(a, b, ok, floor=0.03, margin_a=ma, margin_b=mb)` gates one input inferred under two engines or
  precisions against the reseed floor (2 to 4% in exp57), and `signals.crop_dependence` measures how much a decision
  depends on the crop, for the [adaptive-encoder readiness](plan/vit3_readiness.md) gates.

### Estimate the error rate

```python
import numpy as np
from oe_inferencex.assess import assess_prediction
from oe_inferencex.estimate import certify_zone, estimate_error_rate, estimate_per_class, sample_for_estimation

out = assess_prediction(scores, is_logit=True)
margin, valid = out["arrays"]["confidence"], out["arrays"]["valid"]
map_class = np.where(valid, out["arrays"]["pooled_argmax"], -1).ravel()

s = sample_for_estimation(margin, 300, design="random", valid=valid)   # design="confidence" also needs p1, the top-1 probability
# wrong (1 or 0) and reference (the reviewer's class) for each window of s["indices"]
estimate_error_rate(s, wrong)                                         # estimate, low, high, method
estimate_per_class(s, reference, map_class)                           # per_class, confusion_counts, overall_accuracy
certify_zone(margin.ravel(), s["indices"], wrong, alpha=0.05, valid=valid.ravel())   # random samples only
```

`estimate_from_indices(indices, wrong, margin, valid)` treats windows labelled without a design as a random sample
once it has checked that they could be one. A random sample sits at a mean suspicion percentile near 0.50 and a review
set near 0.97; a review set is refused with that number, and a census of every valid window is accepted.

### Fuse the readings with labels

Where labels exist, `oe_inferencex.calibrate` fits a ranker, P(error | readings), or a side rule, P(side b is right |
readings), cross-fitted by group (tile, event) and reported held-out. Each fusion is bound to its model family and
refuses another unless forced, because such rules do not transfer
([exp59](results/comparisons.md#a-label-fitted-rule-for-which-side-to-believe-exp59)).

```python
import numpy as np
from oe_inferencex.calibrate import fit_side, side_features

z = np.load("exp/out/exp60_masks.npz")                          # exp60: the radar head before and after 55 flood events
fa, fb = {"margin": z["margin_A_s1pre"]}, {"margin": z["margin_B_s1post"]}
fusion, report = fit_side(fa, fb, z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"],
                          groups=z["event"], family="OlmoEarth v1 frozen S1 head")
report["held_out"]["share_right"]                                # 0.830: the fitted rule, cross-fitted by event
report["baseline"]["share_right"]                                # 0.690: believe the side with the larger margin
report["always_a"], report["always_b"]                           # 0.196, 0.804
fusion.prob(side_features(fa, fb), family="OlmoEarth v1 frozen S1 head")   # P(b right) per window
fusion.score(side_features(fa, fb), family="FT-S2 v1")            # ValueError: another family, refit or force=True
```

On this pair most of the accuracy comes from the post-event side being usually right (`always_b`, 0.804).
`fit_ranker(signals, errors, ok, groups=..., family=...)` fits the review ranking the same way
([exp65](results/comparisons.md#calibrate-on-the-record-the-fused-readings-held-out-exp65)). Fusions serialise with
`to_dict` and `from_dict`.

## Modules

| Module | Contents |
|---|---|
| `assess` | Review sets; with a reference, error rate, capture and tie-aware AURC |
| `explain` | The cues behind each flagged window, with their measured enrichment |
| `compare` | Where two inferences differ; with labels, the cross-tab and which side is right; `determinism_check` |
| `estimate` | Sampling designs, the error rate, per-class accuracy, the certified zone, the review-set check |
| `calibrate` | A fitted ranker or side rule, bound to its model family |
| `signals` | Confidence, the boundary indicator, tiling instability, NDWI cues, no-model controls, `crop_dependence` |
| `metrics`, `stats` | AURC, capture, calibration error and their design-weighted forms; sign tests and bootstraps |
| `reliability`, `evidence` | SHRUG-FM's reliability signals, torch-free; the heads a candidate rule is scored with |
| `taskcard`, `lcc` | The [task cards](method/taskcards.md) of OlmoEarth's fine-tuned models; a reader for the served change rasters |
| `cli`, `demo` | The `oe-inferencex` commands |

## Reproducing the experiments

```bash
uv sync                                    # the package only, no torch
uv run pytest                              # the package against the recorded numbers
uv sync --extra encoder --extra geo        # the experiment environment (Linux resolves the cu128 torch build)
uv run python scripts/audit_one_scene.py   # one scene end to end
```

The experiments are the numbered scripts in [`exp/`](../exp/), with outputs under `exp/out/` and the chronology in the
[lab log](../exp/NOTES.md); [`scripts/`](../scripts/) holds named entry points, and cluster jobs follow
[Infrastructure](method/infrastructure.md). A new rule is scored with `metrics` and `stats` under the
[protocol](method/protocol.md). `suite_regression.py` screens a per-unit reading, `f(probs, emb_test, emb_train,
decisions)` with higher meaning more suspect, on Ai2's published suite against exp74's bar; `--model <its directory>`
grades a new encoder the same way.

```bash
HF_HOME=<cache> python scripts/suite_regression.py --model olmoearth_base --signal mypkg.readings:my_reading
python scripts/upstream_revision.py check
```

A reading that screens well enters the record only through a preregistered experiment. `upstream_revision.py check`
exits non-zero if an upstream repository has moved from the commit the record was measured against
([upstream_revisions.json](https://github.com/2imi9/olmoearth_inferenceX/blob/main/exp/out/upstream_revisions.json));
until the difference is attributed, a run against moved files is not comparable with the record.

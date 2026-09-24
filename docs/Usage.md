# Usage

The package [`oe_inferencex/`](../oe_inferencex/) is torch-free. It takes a
prediction map and returns a review plan with a reason per flagged window;
with a reference map it also scores the plan.

## What it takes, and what it refuses

The package reads arrays, not models, so any map from any model can go in. What the record supports differs by
the kind of map, and outside that support the package refuses or says so, rather than returning a plausible
review set for an input it cannot rank.

| Map | Entry point | What the record supports |
|---|---|---|
| Binary probability or logit map, `(H, W)` | `assess_prediction`, `oe-inferencex assess` | Confidence ranks errors on hand labels and on the segmentation tasks of Ai2's suite ([exp70](results/comparisons.md#tasks-we-did-not-choose-ai2s-whole-published-suite-exp70)). Logits are preferred: probabilities tie where they saturate. |
| Per-class scores, `(C, H, W)` | the same | The whole suite, under every encoder of it ([exp74](results/comparisons.md#the-suite-under-the-other-encoders-exp74)); which form of the confidence to use is [exp76](results/comparisons.md#which-confidence-which-statistic-which-aggregator-exp76). |
| Hard class map with an exported confidence band | `assess_classmap` | [The production case](#the-production-case-a-class-map-with-an-exported-confidence-band): the band can only rank the pixels it separates, and the ties are reported. |
| Two inferences of one scene: classes, probabilities or scores | `compare_inferences`, `oe-inferencex compare`, with `--groups` for zones | [Compare](#compare-two-inferences-of-the-same-scene): where they differ, per zone, and which cues mark the difference. Label-free unless labels are given. |
| Continuous map, such as a regression output | `compare` only, with the cut-off named by `--threshold` | Nothing yet. There is no class confidence to rank by, so `assess` refuses it; two inferences can be compared at a named cut-off, and the output says that no recorded experiment grades this case. |
| Hard class map alone | `compare` only | No confidence, so no ranking: the prediction boundary is the only cue. |

Three refusals follow from this, in the Python API as well as on the command line. Probability input outside
[0, 1] raises, because a regression raster or a class map would otherwise be cut at 0.5 and scored. A review set
whose cut-off falls inside a run of equal scores (a hard mask, a quantized band, a constant map) is returned with
`tied_at_cutoff` and a warning: among the tied windows the order is raster position, not evidence. And
`oe-inferencex compare` does not cut a continuous map at 0.5 by default, since two regression outputs would then
"never differ"; name the cut-off and it runs.

What the package never says, for any map: how wrong the map is. It orders the windows and explains the order; an
error rate needs a reference, and then the reference caveat applies. What that reference costs is measured:
300 labelled windows drawn at random give the error rate to about ±3 points on a clean map and ±5 on a messy one,
and labelling whole scenes instead makes the usual range far too narrow
([exp78](results/comparisons.md#how-wrong-is-this-map-what-a-reviewers-labels-buy-exp78)). `oe-inferencex sample`
and `estimate` are that measurement as a tool.

## Fuse the readings with labels

The label-free layers say where to look, why, and how two inferences differ.
Where labels exist, `oe_inferencex.calibrate` fits a combination of those
readings and reports it held-out: a ranker, P(error | readings), or a side
rule, P(side b is right | readings of both sides) on the windows where two
decisions differ. The labels train the weights; the imagery never does. The
fit is cross-fitted by group (tile, event), so no window grades the weights it
trained, and every fusion carries the model family it was fitted on and
refuses to score another unless forced: exp59's side rule, fitted on frozen
heads, lost 26 points on the fine-tuned pair.

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

`fit_ranker(signals, errors, ok, groups=..., family=...)` does the same for
the review ranking: it takes named readings in any orientation (confidence,
tile-phase, a no-model index, ...), learns each sign, and reports the fusion's
held-out excess AURC, capture at the budgets and calibration against every
single reading, with a sign test over groups against the best single. On the v1
S2 head with five readings (exp65) the fusion cuts confidence's held-out excess
AURC by 20% on Bolivia and 30% on the test split. On the
pair above the margin alone mostly teaches the rule which side is usually
right (always b, the post-event pass, gives 0.804) and adds three points over
that, 23 events against 16; exp59's eleven readings added 5 to 17 points over
the raw margin on the crop-offset, backbone and sensor pairs.
Both fusions serialise with `to_dict` and `from_dict`, so an agent can store a
fitted rule next to the labels it came from.

## Command line

`oe-inferencex demo` is the first run: it audits a real sample map shipped with the package (one Dynamic World tile with its expert annotation; `--made-up` for a synthetic one) and draws the result, with no data to find. Four commands cover the label-free halves and the one question that needs labels, without writing Python. Inputs are GeoTIFFs
(with the `geo` extra, which brings rasterio) or `.npy` arrays; outputs are plain
files the caller reads back, and nothing narrates.

```bash
oe-inferencex assess water_prob.tif --out audit --budgets 0.01 0.05 0.10
```

writes `audit/assessment.json` (the `assess.summary` view: window count,
confidence quantiles, boundary share, the review sets, and the risk-coverage
block when `--reference labels.tif` is given, with its caveat), one
`review_set_05pct.csv` per budget (rank, window and pixel coordinates, map
coordinates of the window centre, confidence, boundary), `suspicion.tif` and
`boundary.tif` on the window grid, and `explanation.json` (the cues each review
window carries, their measured enrichment, the windows no cue explains). Pass
`--logits` for a logit map, `--order boundary_first` for the order exp36
supports, `--patch` for the window size.

```bash
oe-inferencex compare before.tif after.tif --out diff --labels reference.tif --groups tiles.tif
```

writes `diff/comparison.json` (the `compare_inferences` summary: how much the
two decisions differ, per group when `--groups` gives tile or event ids, the
boundary enrichment of the differing windows, and with `--labels` which side is
right and the cross-tab), `differing_windows.csv` and `disagreement.tif`. Class
maps are read as integers; a single-band probability map is thresholded at
`--threshold` and a multi-band score map argmaxed, so the module never compares
floating point. The two maps must share one grid, and so must `--labels` and
`--groups`: a raster in another CRS or at another origin is refused. Both maps
are pooled over the pixels both predicted. A tied window goes to the more
confident voters when both maps carry a confidence; when either is a hard class
map, which has none, it is left out of the comparison and counted in a note, so a
map and its own probability version never differ by tie-breaking. `disagreement.tif`
is NaN where nothing was compared. With
`--labels`, every number above stays label-free over all compared windows and
the graded block alone is restricted to the windows with a majority label.

```bash
oe-inferencex sample water_prob.tif --budget 300 --out to_label.csv
# ... a reviewer fills the `wrong` column with 1 or 0 per window ...
oe-inferencex estimate to_label.csv
# per class as well, once the reviewer has also filled a `reference_class` column:
oe-inferencex estimate to_label.csv --per-class
# and, from a RANDOM sample, the share of the map that can be trusted at a stated error rate:
oe-inferencex sample water_prob.tif --budget 300 --design random --out random.csv
oe-inferencex certify random.csv --alpha 0.05
```

is how wrong the map is. `sample` writes the 300 windows to label (row, column,
pixel and map coordinates, stratum, confidence) with an empty `wrong` column, and a
`to_label.json` beside it carrying the design. The default design stratifies the
map by confidence margin and allocates the budget from the model's own
confidence, which on exp78's seven tasks narrowed the interval to a median 0.80
of a random sample's with coverage intact; `--design random` is the plain draw,
`--design tiles` labels 16 windows in each of a run of tiles, because that is how
people actually label. `estimate` reads the filled file back and writes
`to_label_estimate.json`: the error rate, its 95% interval and half-width, and the
method the design earns. For the tile design it also writes the naive interval
the ordinary formula would give, beside the corrected one, with the warning that on
exp78's tasks that naive interval covered 51 to 78% of the time while claiming
95%. The corrected interval, a ratio estimator over tiles, is better and still not
honest everywhere: on exp78's tasks it covered 0.91 to 0.94 where the tiles are of
equal size and 0.60 on MADOS, whose tiles hold 1 to 400 windows. If you have not
labelled yet, use the default design. A CSV with a blank `wrong`, or whose rows are
not the design's, is refused.

**Per class, and a zone with a guarantee.** With a `reference_class` column filled in beside `wrong` (the class
the reviewer saw in each window, in the map's class ids), `estimate --per-class` adds what the map-accuracy
literature says a producer owes: per class, the user's accuracy (of the windows the map calls it, how many are
it), the producer's accuracy (of the windows that are it, how many the map found) and the error-adjusted share
of the map that is it, each with an interval, from the same labelled sample (exp81; from a random sample the
user's accuracy has an exact hypergeometric interval, covering at least 95% by construction, and the other
quantities Wilson intervals on the effective sample size, since the field's Wald form collapses to a point
whenever a class shows no sampled error and covered as little as 2% of draws on the suite). A class with fewer
than 30 labelled windows is reported with a warning, and so is a class the confidence design samples thinly, one
nearly all of whose windows are labelled, one whose interval rests on one to four sampled errors (exp81's seventh
amendment: it marks 80% of the draws on which such an interval misses) and one the map never predicts, whose
producer's accuracy is then exactly 0. The command line prints each warning's reason beside the class. `certify` needs a **random** sample (`--design random`; a stratified or tile draw is refused, because
the guarantee rests on the labels inside each zone being a random draw of that zone) and an error rate
`--alpha` you are prepared to tolerate; it returns the largest most-confident share of the map that is wrong at
most that often, certified by exact hypergeometric tests so that the statement fails on at most `--delta` (default
0.1) of samples like yours, plus a window mask of the zone (exp80). It says when the budget cannot certify the
level asked for: with no error among its labels a zone still needs about `ln(delta)/ln(1 − alpha)` of them, 45 at
5%, 255 at 0.9%. On the suite 300 labels certified about half the map at half its error rate on a typical task;
outside the zone nothing is certified.

**Do not label the review set and divide.** The review set is built to hold
errors; on exp78's export the 5% review set gave 1.8 to 5.8 times the true rate
on every task. `sample` draws a sample with weights the estimator undoes; the
review set has none. In the API, `estimate_from_indices` takes windows labelled
without a design, treats them as a random sample, and first checks that they
could be one — a random sample sits at a mean suspicion percentile of 0.50, the
review set near 0.97 — and refuses a review set with the number.

## Modules

| Module | What it gives you |
|---|---|
| `assess` | Review sets at any budget from a logit, probability or exported-confidence map, in confidence order or in the boundary-first order; error rate, capture at each budget and tie-aware AURC when a reference map exists |
| `explain` | Why each flagged window is suspect: label-free cues, each with its measured share among error and correct windows and the experiment that measured it |
| `compare` | How two inferences of the same scene differ: the disagreement rate pooled and per tile or event, what the disagreement windows have in common (the enrichment of each label-free cue among them), whether two disagreement sets are the same set; with labels, the errors one side corrects and the errors it adds, and which side is right where they disagree; `determinism_check`, the same input inferred twice under two engines or precisions, gated against the reseed floor |
| `signals` | Confidence, the boundary indicator, tiling instability, the NDWI cues and the pixel controls, as pure functions; `crop_dependence`, how much of a decision map depends on the crop it was inferred in, a map property for encoders that adapt per input |
| `calibrate` | Where labels exist, a fitted ranker or a fitted which-side rule, cross-fitted by group and reported held-out; each fusion is locked to the model family it was fitted on, because such rules do not transfer |
| `estimate` | How wrong the map is, from a labelled sample: which windows to label (stratified by confidence, random, or by tile) and the error rate with the interval that design earns; per class, the user's and producer's accuracy and the error-adjusted class share, the user's accuracy under a random draw with the exact hypergeometric interval and the rest with Wilson intervals on the effective sample size (`estimate_per_class`, exp81); and from a random sample, the largest most-confident zone that is wrong at most `alpha` of the time, certified by exact hypergeometric tests so the statement fails on at most `delta` of draws (`certify_zone`, exp80), with the refusal when the budget cannot certify that level; the exact hypergeometric interval under a random draw, Wilson on the design's effective sample size under the confidence design, or the ultimate-cluster interval with the naive one beside it under tiles; and a check that refuses the review set as a sample, since labelling it and dividing gives two to six times the true rate |
| `metrics`, `stats` | Tie-aware AURC and capture at a budget, exact sign tests, one vote per cluster, block and cluster bootstraps; and the design-weighted forms for a reference that is a probability sample rather than a map, including a base-rate-free AUROC and a paired bootstrap on the difference between two disjoint subsets |
| `reliability`, `evidence` | SHRUG-FM's published reliability signals reimplemented torch-free, so a competitor's method is scored under this protocol rather than described; the small logistic and softmax heads a candidate rule is scored with. The expected calibration error lives in `metrics` |
| `taskcard`, `lcc` | What each fine-tuned OlmoEarth model is; a range reader for the served change rasters |

## Assess a prediction and explain the review set

```python
import numpy as np
from oe_inferencex.assess import assess_prediction, summary
from oe_inferencex.explain import explain_review_set
from oe_inferencex.signals import ndwi_level

logits = np.load("water_logits.npy")        # (H, W) binary logits, or (C, H, W) per class
s2 = np.load("s2_window.npy")               # (12, H, W) Sentinel-2 bands, for the spectral cue

out = assess_prediction(logits, is_logit=True, patch=4, budgets=(0.05, 0.10), order="boundary_first")
why = explain_review_set(out, cues={"ndwi_ambiguous": ndwi_level(s2, patch=4) > -0.1})

summary(out)["review_sets"]["0.05"]         # the 4-px windows to review first, with the boundary share of the set
why["budgets"][0.05]["windows"][0]          # {'row': 28, 'col': 26, 'confidence': 0.48, 'cues': ['boundary', 'low_confidence', 'ndwi_ambiguous']}
why["quotes"]["ndwi_ambiguous"]             # "is spectrally ambiguous ... (48% of error windows vs 7% of correct ones, 7.2x; ...)"
```

- `order="confidence"` (default) reviews the least confident windows first,
  the ranker every experiment scored. `order="boundary_first"` reviews the
  boundary windows first, then the interior, each by confidence; it
  captures more errors at 5 to 10% budgets on hand labels (exp36). AURC
  entries always score confidence.
- `form="top1"` ranks a multi-class logit map by one minus the top probability,
  computed tie-free from the logits, the best member of the confidence family on
  multi-class tasks (exp76); the default stays the logit margin of 1.0.0 and
  carries a warning on such maps. Binary maps are unaffected.
- `explain_review_set` derives the boundary and low-confidence cues from
  the assessment itself. Any other cue is a boolean array of window shape
  the caller derives: spectral ambiguity from the input bands as above,
  tiling instability from a second forward pass on a 2-px shifted window
  (`signals.aligned_tile_phase`), a reference-side fact such as JRC
  seasonality. Cues in `explain.CUES` are quoted with their measured
  enrichment; other names are reported as unmeasured.
- Arrays stay in `out["arrays"]`. `summary(out)` is the JSON-safe view that
  crosses a tool boundary, as the [agent contract](method/agent_integration.md)
  requires: summary statistics cross, rasters are written to files.

## The production case: a class map with an exported confidence band

```python
from oe_inferencex.assess import assess_classmap

out = assess_classmap(hard_map, confidence_band, n_classes=9, patch=4, order="boundary_first")
out["warnings"]        # reports ties: a quantized or saturated band can only rank the pixels it separates
```

The served change rasters export no class confidence at all; there the
boundary fraction is the only cue (exp20), and `lcc` reads the rasters by
HTTP range request.

## With a reference map

Pass `reference=` (class labels, values below 0 for no reference) and the
assessment adds `against_reference`: the error rate, the errors captured at
each budget in the chosen review order, the tie-aware AURC of confidence
and of the boundary indicator, the oracle and random baselines, and the
caveat that reference-product labels flatter boundary-type signals (exp18).
Grade on expert labels; never train a rule on them.

## Compare two inferences of the same scene

![The two-period, two-sensor square on one GEOID-Flood chip: four dated inputs, four inferences on identical windows, the same-date and same-sensor differences on the scene, and the label bridge boxed apart](figures/compare.png)

*One GEOID-Flood chip (event EMSR273-1, Gruemirë on Lake Shkodër; exp62 chip 7) with all four cells: the Sentinel-2 composite
dated 2017-04-02 and the Sentinel-1 pass of 2017-03-11 read by the permanent-water heads, the Sentinel-1
pass of 2018-03-11 read by the water-after head (exp60), and WorldFloods v2's Sentinel-2 L1C scene of
2018-03-28 read by the post-event optical head exp62 fitted on other activations (radar shown as VH
backscatter, open water dark, flooded vegetation bright). The lake on the left is permanent water, its
shore plain on the right is flooded by the label. The same-period pairs differ on 22 windows before the event and 44
after: the shore, and the flooded vegetation the radar head does not call water. The same-sensor pairs
differ on 97 (optical) and 67 (radar) windows; 81% of the radar's are the flood by the label and the rest
a head off its label at the shore. Over the 544 chips with all four cells the pre-event optical head is
off its label on 22% of the windows, on one event (the Ebro delta) where it finds a quarter of the label's
permanent water, so the two clean pairs are the radar one across the event (41% flood) and the cross-sensor
one after it (18%); exp62's preregistered comparisons fail on that event. The optical pass after the event
is the first clear one WorldFloods holds for the area, 17 days after the radar pass.*

The same measurement for a whole event, from the committed decisions alone (EMSR275-1, a larger event than the
figure's EMSR273-1):

```python
import numpy as np
from oe_inferencex.compare import compare_inferences

z = np.load("exp/out/exp57_masks.npz")                               # exp57: the two heads' decisions on 55 GEOID-Flood events
ev = z["geoid_event"] == "EMSR275-1"                                  # 55 events in the file; the figure's is EMSR273-1
a, b, ok = z["geoid_s2"][ev], z["geoid_s1"][ev], z["geoid_ok"][ev]    # pre-event S2 head, post-event S1 head, windows both predicted
out = compare_inferences(a, b, ok, labels=z["geoid_y_after"][ev])   # graded on water after the event
out["disagreement_rate"]                                            # 0.094: 3,310 of 35,084 windows
out["graded"]["which_side"]["share_b_right"]                        # 0.79: the S1 head matches the post-event label on 79% of them
flooded = z["geoid_y_after"][ev] & ~z["geoid_y_permanent"][ev]
flooded[out["arrays"]["disagree"]].mean()                          # 0.77: three quarters of the difference is the flood itself
out["dates"]["status"]                                              # "unstated": these arrays carry no acquisition dates
```

The two heads read different dates, pre-event optical and post-event radar, so this is a cross-date comparison
run without its dates, and the 0.79 is "which map matches the post-event label", not "which map is right": the
pre-event map is counted wrong wherever the flood changed the ground, which is the 0.77. Passed with the event's
dates, `compare_inferences(a, b, ok, labels=..., dates=(pre, post))` refuses until `labels_date=post` is given,
and `graded["graded_against"]` then says exactly that.

Two backbones, two sensors, a frozen and a fine-tuned model, or the same
model on shifted crops: `compare` measures how their decisions differ on the
windows both predicted, and grades the difference where labels exist.

**Dates.** A difference between two maps of the same ground at the same time is
an error in at least one of them. Across dates it can instead be real change on
the ground, a flood, a harvest or a season: in the GEOID-Flood example above,
three quarters of the difference is the flood itself. So `compare` takes the
dates the maps describe (`dates=(date_a, date_b)` in Python, `--date-a` and
`--date-b` on the command line; a period `YYYY-MM-DD/YYYY-MM-DD` covers a
composite or an annual map) and says in `dates.reading` what a difference can
mean at those dates. Across dates it refuses to grade which side is right
unless the labels' date is given (`labels_date`, `--labels-date`), because a
window that changed is right in one map and wrong in the other whatever either
model did; with it, `graded.graded_against` says which map the labels match in
time. Without dates the comparison runs as before and says the dates were not
given.

```python
import numpy as np
from oe_inferencex.assess import summary
from oe_inferencex.compare import compare_inferences
from oe_inferencex.signals import boundary_indicator

z = np.load("exp/out/exp02_cache.npz")      # exp02: Nano and Base water probabilities on the same 1024 windows of Kazungula
a, b = z["p_nano"] > 0.5, z["p_base"] > 0.5  # hard decisions first; the module takes no threshold
lab = z["ev_labels"].astype(bool)           # the hand labels; every window carries one here, so every window is valid
ok = np.ones_like(a)                        # the windows both predicted

out = compare_inferences(a, b, ok, labels=lab, cues={"boundary": boundary_indicator(a) > 0})
out["disagreement_rate"]                    # 0.027: 28 of 1024 windows
out["where"]["boundary"]["enrichment"]      # 5.5x: 86% of the disagreement windows sit on a boundary of a, 15% of the agreement windows
out["graded"]["which_side"]                 # {'n_disagree': 28, 'a_right': 11, 'b_right': 17, 'neither': 0, ...}: Base is right on 17 of the 28
out["graded"]["crosstab"]["phi"]            # 0.43: the two error maps overlap; Base corrects 17 of Nano's 28 errors and adds 11
summary(out)                                # the JSON-safe view, no arrays
```

- Everything before the `graded` block is label-free: the disagreement
  rate, the per-group rates and the cue enrichment need only the two
  decisions and the validity mask. `labels=` adds the graded block: the
  cross-tab of the two error maps (the errors of `a` that `b` corrects,
  the errors `b` adds, the errors both make, and their phi) and which side
  matches the label on the disagreement windows.
- `groups=` is an id per window (a tile, an event; `(N,)` ids broadcast
  over `(N, H, W)` maps) and adds the disagreement rate per group; with
  labels, the cross-tab per group and `over_groups` of the net correction,
  corrected minus broken: its median, the share of groups where the sign
  flips and the one-sided exact sign test, the per-event machinery of
  exp55. `over_groups` takes any per-group statistic.
- Decisions are hard maps: threshold a probability at 0.5 or take the
  argmax of the class scores first. The module refuses a float map rather
  than pick a threshold, so floating-point noise never counts as a
  difference. `ok` and the cues are boolean or 0/1 arrays of the same
  shape; `stability` gives the pairwise phi of several disagreement masks,
  the test that two runs flag the same set.
- Which side to believe is not a label-free reading. On the disagreement
  windows the more confident side is right more often than a coin flip and
  far less often than the side the labels prefer, and the first inference's
  confidence does not order the set (exp58); a rule fitted on the heads'
  training labels adds 5 to 17 points on the crop-offset, backbone and sensor
  pairs and fails on the fine-tuned pair (exp59). The module resolves nothing
  and reports both sides.
- The disagreement mask stays in `out["arrays"]`. `assess.summary(out)`
  gives the JSON-safe view that crosses a tool boundary, as the
  [agent contract](method/agent_integration.md) requires; undefined values
  are NaN in the arrays and null in the summary.

## Same input, two engines

```python
from oe_inferencex.compare import determinism_check
from oe_inferencex.signals import crop_dependence

# the same tiles inferred under fp32 and bf16: decisions a and b, margins ma and mb, valid windows ok
gate = determinism_check(a, b, ok, floor=0.03, margin_a=ma, margin_b=mb)
gate["disagreement_rate"], gate["margin_drift"], gate["passes"]

# the same tile inferred at crop offsets 0, 1, 2, 3 px: decisions on the patch grid, shift first
crop_dependence([d0, d1, d2, d3], patch=4)["rate"]     # share of windows whose decision depends on the crop
```

Both are map properties, not rankers. The floor is the reseed disagreement
rate of the model family (exp57 measured 2 to 4%); a rate above it is drift the
engine introduced. Why these exist, and the predictions they gate, is in
[the ViT3 readiness page](plan/vit3_readiness.md).

## Scoring a new rule the same way

`metrics.capture_at_budget_expected` and `metrics.excess_aurc` score a
candidate per unit; `stats.sign_test`, `stats.clustered_sign_test` and
`stats.cluster_bootstrap_difference` test it across units; `explain.cue_enrichment`
validates an explanation cue. The experiment scripts in `exp/` show the
full pattern, two references at once with controls; the
[protocol](method/protocol.md) is the rulebook.

On Ai2's published suite, any per-unit reading can be screened beside the
margin and the two no-model controls on every task a model carries, with
exp74's bar for the margin and no change to the record:

```bash
HF_HOME=<cache> python scripts/suite_regression.py --model olmoearth_base --signal mypkg.readings:my_reading
```

The reading is `f(probs, emb_test, emb_train, decisions) -> per-unit array`,
higher meaning more suspect. A candidate that screens well enters the record
only through a preregistered experiment.

The same command re-grades a **new encoder**: `--model <its directory>` runs
exp70's protocol on it and applies exp74's bar. Before comparing what comes
back against anything recorded here, check that upstream is where the record
left it, because none of these experiments pins a revision when it downloads:

```bash
python scripts/upstream_revision.py check
```

It prints every upstream repository the record rests on, the commit the
recorded numbers were measured against
([exp/out/upstream_revisions.json](https://github.com/2imi9/olmoearth_inferenceX/blob/main/exp/out/upstream_revisions.json)),
and what `main` points at today; it exits non-zero if any of them has moved or
if the code fetches something the record does not pin. A run against moved
bytes is not comparable until the difference is attributed, and
`suite_regression.py` now says which revision it read and carries it into its
summary.

## Reproduce

```bash
uv sync                                 # the package only, no torch
uv run pytest                           # the package against the recorded numbers
uv sync --extra encoder --extra geo     # full experiment environment (Linux resolves the cu128 torch build)
uv run python scripts/audit_one_scene.py   # one scene end to end
```

Named entry points live in [`scripts/`](../scripts/); each wraps the numbered
experiment it runs, so the experiment file stays the record and the script is
the formal name (`scripts/audit_one_scene.py` runs `exp/exp02_full_slice.py`).
Experiments are `exp01`–`exp57` in [`exp/`](../exp/), outputs under
`exp/out/`, chronology in the [lab log](../exp/NOTES.md). Cluster jobs
follow the pattern in [infrastructure](method/infrastructure.md).

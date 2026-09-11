# Usage

The package [`oe_inferencex/`](../oe_inferencex/) is torch-free. It takes a
prediction map and returns a review plan with a reason per flagged window;
with a reference map it also scores the plan.

## Modules

| Module | What it gives you |
|---|---|
| `assess` | Review sets at any budget from a logit, probability or exported-confidence map, in confidence order or in the boundary-first order; error rate, capture at each budget and tie-aware AURC when a reference map exists |
| `explain` | Why each flagged window is suspect: label-free cues, each with its measured share among error and correct windows and the experiment that measured it |
| `compare` | How two inferences of the same scene differ: the disagreement rate pooled and per tile or event, what the disagreement windows have in common (the enrichment of each label-free cue among them), whether two disagreement sets are the same set; with labels, the errors one side corrects and the errors it adds, and which side is right where they disagree |
| `signals` | Confidence, the boundary indicator, tiling instability, the NDWI cues and the pixel controls, as pure functions |
| `metrics`, `stats` | Tie-aware AURC and capture at a budget, exact sign tests, one vote per cluster, block and cluster bootstraps |
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

![Two dates, two sensors on one GEOID-Flood chip: three pieces of evidence with their dates, three inferences on identical windows, the same-date cross-sensor difference and the same-sensor cross-date difference on the scene, and the label bridge boxed apart](figures/compare.png)

*One GEOID-Flood chip (event EMSR275-1, chip 2293) through `compare`, exp60's three-cell design. Evidence:
the Sentinel-2 composite and the Sentinel-1 pass of 2017-07-07, and the Sentinel-1 pass of 2018-03-22.
Inferences: the permanent-water head on each pre-event image, the water-after-the-event head on the
post-event radar, one Sentinel-1 head for both dates. Same date across sensors: 4 of this chip's 196
windows differ, none of them flooded; over 55 events 4.9% of 869,160 windows differ and 0.4% of those are
flooded by the label, so the sensor axis is sensor error alone (preregistered P2 holds). Same sensor
across dates: 151 windows differ here, 87% flooded by the label and 13% one head off its label; over the
55 events 3.4% differ and 25.9% of those are flooded, 1.33 times the mixed pair, short of the preregistered
doubling (P1 fails), because 54.5% of that difference is the pre-event radar calling water that neither
the label nor the JRC monthly water of that month holds (exp61: 0.1% of those windows are water there). The boxed panel needs labels: violet windows are the flood, teal ones an
error of one head, amber outlines the four sensor differences.*

The same measurement for the whole event, from the committed decisions alone:

```python
import numpy as np
from oe_inferencex.compare import compare_inferences

z = np.load("exp/out/exp57_masks.npz")                               # exp57: the two heads' decisions on 55 GEOID-Flood events
ev = z["geoid_event"] == "EMSR275-1"                                  # the event in the figure
a, b, ok = z["geoid_s2"][ev], z["geoid_s1"][ev], z["geoid_ok"][ev]    # pre-event S2 head, post-event S1 head, windows both predicted
out = compare_inferences(a, b, ok, labels=z["geoid_y_after"][ev])   # graded on water after the event
out["disagreement_rate"]                                            # 0.094: 3,310 of 35,084 windows
out["graded"]["which_side"]["share_b_right"]                        # 0.79: the S1 head matches the post-event label on 79% of them
flooded = z["geoid_y_after"][ev] & ~z["geoid_y_permanent"][ev]
flooded[out["arrays"]["disagree"]].mean()                          # 0.77: three quarters of the difference is the flood itself
```

Two backbones, two sensors, a frozen and a fine-tuned model, or the same
model on shifted crops: `compare` measures how their decisions differ on the
windows both predicted, and grades the difference where labels exist.

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

## Scoring a new rule the same way

`metrics.capture_at_budget_expected` and `metrics.excess_aurc` score a
candidate per unit; `stats.sign_test`, `stats.clustered_sign_test` and
`stats.cluster_bootstrap_difference` test it across units; `explain.cue_enrichment`
validates an explanation cue. The experiment scripts in `exp/` show the
full pattern, two references at once with controls; the
[protocol](method/protocol.md) is the rulebook.

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

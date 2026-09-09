# Findings

What the audit found, what holds up, and how well. Every number traces to a
file under `exp/out/`; the per-experiment detail is in the
[technique ledger](TECHNIQUES.md) and the results pages.

![The audit pipeline on a real scene: Sentinel-2 bands, the frozen encoder and head, prediction, confidence and boundary layers, the review set at a 5% budget, the reasons per window](figures/pipeline.png)

*One scene (okavango_80) through the pipeline: the layers are the actual rasters, the orange squares are the 5% review set in the boundary-first order.*

## What holds

1. **The model's own confidence is the best label-free error ranker** on
   every expert-labelled testbed: AWF points, Sen1Floods11 hand labels, and
   the fine-tuned model run end to end (exp04, exp16, exp18, exp21). This
   holds for OlmoEarth v1. It does not hold on Bolivia under v1.2 Base, the
   encoder the served product uses, and one testbed. On Sen1Floods11
   Bolivia, a single flood event, a no-model spectral index ranks the
   model's errors as well as its own confidence under v1 and better under
   v1.2, for the grid window and for the shift-averaged decision alike
   (exp45, exp47). On the multi-region test split confidence still wins.
   A bag of bootstrap heads does not improve on it: an apparent gain under
   v1 on Bolivia (exp49) did not replicate with a fresh draw (exp50).
2. **Review boundary windows first, then by confidence.** Errors sit on
   prediction boundaries, 75% of errors against 21% of correct windows, and
   this order captures more of them than confidence alone at 5% and 10%
   review budgets on hand labels, preregistered (exp36). No extra inference.
3. **Every flagged window comes with a reason.** 95% of the error windows on
   hand labels carry at least one label-free cue with a measured enrichment;
   spectral ambiguity is 7x enriched and the one cue that adds precision
   inside the review set (exp37).
4. **An accuracy needs a coverage.** The fine-tuned model is 0.93 accurate
   where it claims 0.99; keeping the 80% most confident windows gives 0.945
   (exp21).
5. **The served product can be triaged without confidence.** It exports no
   class confidence; boundary fraction alone captures a median 0.88 of the
   disagreements at a 5% review budget (exp20).
6. **Averaging the tilings improves the map itself.** Run the encoder on the
   window cropped at four offsets and average the four decisions per pixel:
   pixel accuracy on hand labels rises by 1.0 points on Bolivia and 0.9 on
   the multi-region test split, preregistered, no labels, no retraining
   (exp42). Errors concentrate on windows whose hand labels are mixed, but
   that is concentration, not a ceiling: any one-class-per-block decision
   must miss only 3% of pixels, far below the grid's measured error rate
   (exp43).
7. **Fusing signals needs labels, and then it works.** A logistic combination
   of ten label-free signals (confidence, tiling instability, NDWI level, the
   ensemble entropies, embedding distances, input extremity) fitted on the
   head's own training split beats confidence on Bolivia under both backbones
   and on the v1 test split by 25-38% of excess AURC; every label-free
   midrank fusion had lost (exp47, exp49). On the v1.2 test split the lead is
   below the worthwhile threshold, so this is three arms of four. The weight
   sits on the no-model NDWI level with the ensemble entropies and confidence
   behind it, and dropping NDWI level costs the most (exp50). Fitted on tile
   failures instead, it beats confidence at the tile level on the multi-region
   split under both backbones (AURC 0.120 to 0.078 and 0.134 to 0.086), not on
   Bolivia. SHRUG-FM's own signals, ported to the window, do not beat
   confidence (exp49).

## The numbers

Sen1Floods11 Bolivia hand labels, 81,984 windows, 8.8% of them errors
(exp36, exp37):

| Review budget | Errors caught, confidence order | Errors caught, boundary first | Error rate inside the set |
|---|---|---|---|
| 5% | 0.259 | 0.274 | 0.38 |
| 10% | 0.465 | 0.494 | 0.33 |
| 20% | 0.749 | 0.732 | 0.26 |

Why a window is flagged: share among error windows against correct windows
on the same testbed, with the error rate among windows carrying the cue
(exp37):

| Cue | Errors / correct | Enrichment | Error rate with the cue |
|---|---|---|---|
| on a prediction boundary | 0.750 / 0.214 | 3.5x | 0.25 |
| among the least confident 20% | 0.589 / 0.163 | 3.6x | 0.26 |
| unstable under a tiling shift | 0.583 / 0.164 | 3.6x | 0.26 |
| spectrally ambiguous, NDWI near zero | 0.483 / 0.067 | 7.2x | 0.41 |

Window design (exp42), pixel accuracy on hand labels over the region every
tiling covers:

| Decision | Bolivia, 441 tiles | Test split, 800 tiles |
|---|---|---|
| grid window (one tiling) | 0.897 | 0.941 |
| shift-averaged (four tilings) | 0.907, better on 321 tiles, worse on 66 | 0.950, better on 583, worse on 78 |
| share of the grid window's errors on mixed-label windows | 45% (10% of windows) | 58% (10% of windows) |
| errors the block geometry actually forces (oracle) | at most 29% | at most 49% |
| share of the averaged map's own errors on those windows | 43% | 56% |
| segment majority over a spectral partition (exp43) | 0.904, breaks more than it corrects | 0.948, breaks more than it corrects |
| sixteen crop offsets instead of four (exp44) | 0.9074, +0.04 points, below the worthwhile threshold | 0.9507, +0.04 points |

Fine-tuned AWF model, end to end (exp21, exp36): confidence catches 22% of
the errors at a 5% budget, 39% at 10%, 63% at 20%; the boundary-first order
picks the same 5% set. Selective accuracy is 0.945 at 80% coverage.

## How a claim gets in

![A candidate signal next to the model's confidence and a no-model control, scored on both references on identical windows; supported only if it beats both baselines on expert labels; labels grade, never train](figures/protocol.png)

A candidate rule or cue is tested on two references at once, the ESA
WorldCover map and hand-labelled flood masks, against the model's own
confidence and four no-model controls. The primary test and its direction
are written down before the run. Scores are tie-aware; scenes vote once
per river; tiles are bootstrapped as clusters. Expert labels grade a rule
and never train it. A win against the weak map alone is not support. The
package tests recompute the recorded numbers from the committed artifacts.
The full rules are in the [protocol](method/protocol.md).

## Limits and the open question

The hand-label testbed is one flood event scored with a linear probe, and
the fine-tuned model contributes 41 errors in 344 windows, so the gains
above are real but small and replicated on one region. Tiling instability
wins 26 of 27 scenes and 8 of 8 rivers against the WorldCover map yet not
on hand labels; the decisive test needs adjudicated cells on the eight
rivers ([issue 2](https://github.com/2imi9/olmoearth_inferenceX/issues/2)).
A second expert-labelled few-class testbed with a spatial split is the
next thing that would raise the evidence
([issue 7](https://github.com/2imi9/olmoearth_inferenceX/issues/7)).

Side product: OlmoEarth v1's pretraining target has effective rank 2, and a
normalised target is 57 to 70% predictable from context (exp32 to exp34,
[issue 11](https://github.com/2imi9/olmoearth_inferenceX/issues/11)).

## What was tried and rejected

Kept as evidence, one line each with the reason, in the
[technique ledger](TECHNIQUES.md). The short version: no signal derived
from the encoder's internals, its pretraining objective, a posterior over
the probe head, feature-space typicality, a second model of the same
family, or flip-and-rotate consistency ranks errors better than confidence
on expert labels.

# Findings

This page summarises what the experiments established about assessing Earth-observation classification maps, without
labels and with a labelled sample. Each result is a claim in the [claim ledger](method/claims.md), checked against its
artifact under `exp/out/`, with its evidence in the [results record](results/comparisons.md).

## In short

- The model's own confidence ranks a map's errors better than every control that does not use the model, on every
  scored task of Ai2's published suite and against declared and surveyed references.
- The exceptions found are single flood events or single sensors.
- Errors concentrate on prediction boundaries, yet reviewing boundary windows first beats confidence alone only on a
  minority of flood events.
- Two inferences of one area differ mostly on prediction boundaries, the sensor moves shared errors more than the
  backbone, and which side is right needs labels.
- A few hundred random labels give an error rate with an interval that holds its coverage, per-class accuracies and a
  certified zone.
- No signal from inside the encoder beats confidence, and a consensus of encoders does not estimate accuracy.
- For a language-model agent the package is decisive at small scale; at larger scale it contributes the refusal to
  choose a side that the data cannot settle.

![The assessment pipeline on a real scene: Sentinel-2 bands, the frozen encoder and head, prediction, confidence and boundary layers, the review set at a 5% budget, the cues per window](figures/pipeline.png)

*One scene (okavango_80) through the pipeline. The layers are the actual rasters; the orange squares are the 5% review
set in the boundary-first order.*

## Ranking errors without labels

The package scores each window, the block of pixels reviewed as a unit, by the model's margin, the difference between
its two highest class probabilities, and reviews the smallest margins first. A ranking is graded by excess AURC, the
area under the risk-coverage curve minus that of a perfect ranking, against no-model controls such as class rarity,
embedding distance or the normalised difference water index (NDWI).

On all 24 tasks of Ai2's published embedding suite for which a margin is defined, the margin ranks errors better than
the best no-model control, on all 14 distinct sources and 6,435,473 graded units (exp70).
<!-- claim:suite-margin-wins-every-task -->
Under ten probe seeds OlmoEarth Base's lead over that control stays positive on all 24 tasks (exp79).
<!-- claim:exp79-base-margin-wins-under-every-seed -->
On all sixteen encoders of the suite the margin beats the best no-model control on at least 75% of each encoder's
tasks under every seed, the lowest share being 21 of 24 for Clay Large (exp79).
<!-- claim:exp79-headline-holds-under-every-seed-on-every-encoder -->
It also beats a five-seed probe ensemble on 22 of 24 tasks and nearest-neighbour and Mahalanobis distances on 24 of
24 (exp73). <!-- claim:suite-margin-beats-the-strong-alternatives -->
The margin takes a median 0.68 of the gap between a random and a perfect ranking, and labels reach a fifth to a third
of the remainder (exp70, exp65). <!-- claim:margin-takes-two-thirds-of-the-ranking-headroom -->
One minus the top probability ranks slightly better than the margin on 14 of 16 multi-class tasks; on the 8 binary
tasks the two give one ranking (exp76). <!-- claim:top1-beats-the-margin-on-multiclass -->

On 4,778 LUCAS polygons, whose reference is a surveyor's observation on the ground, the margin beats the best control
computable at inference time on 94 regions against 32 (exp68).
<!-- claim:lucas-ranking-survives-ground-observation -->
Dynamic World's own margin ranks its errors better than a class-rarity control on 315 expert-annotated tiles against
91 (exp67). <!-- claim:dw-margin-ranks-a-production-model -->
On DFC2020 land cover the margin beats the NDWI control on all three sensor arms (exp66).
<!-- claim:dfc2020-margin-beats-pixel-control -->
On 106,274 windows of farmers' crop declarations it beats the best no-model control in all three regions and on all
but four of 146 grid cells (exp69). <!-- claim:eurocrops-ranking-holds-on-declarations -->

On Sen1Floods11 Bolivia, one flood event, NDWI ranks the model's errors as well as its confidence under OlmoEarth v1
and better under v1.2 (exp45, exp47). <!-- claim:bolivia-ndwi-exception -->
Under the Sentinel-1 probe NDWI instead beats confidence on the multi-region test split and loses on Bolivia (exp51).
<!-- claim:s1-probe-ndwi-flip -->
The fine-tuned Sentinel-2 model's confidence beats NDWI on Bolivia under both backbones, so that exception belonged to
the frozen encoder (exp52). <!-- claim:finetune-dissolves-bolivia-exception -->
On 45 GEOID-Flood areas of interest NDWI wins on 2 for permanent water and a sensor control on 1 for post-event water;
both permanent-water cases lie in one Copernicus emergency activation, an exception rate of one activation in nine
(exp55). <!-- claim:geoid-exception-rate -->
The ordering among confidence readings degrades as the probe overfits, by four ten-thousandths of excess AURC across
the suite, and only the severe memorisation of a LUCAS probe reversed it (exp68, exp70).
<!-- claim:suite-probe-gap-degrades-the-ordering -->
On the median GEOID-Flood area the 5% of windows that confidence flags hold 88% of the permanent-water errors and 64%
of the post-event water errors, 17.5 and 12.8 times a random 5%, and 16.9 and 12.1 times when areas are clustered by
activation (exp55). <!-- claim:geoid-capture-effect-size -->

## Review order and explanations

On Bolivia hand labels 75% of the error windows lie on a prediction boundary, against 21% of the correct windows
(exp36, exp37). <!-- claim:errors-sit-on-boundaries -->
There, reviewing boundary windows first, in confidence order, and then the interior catches more errors than
confidence alone at review budgets (shares of windows reviewed) of 5% and 10%, as preregistered, and not at 20%
(exp36). <!-- claim:boundary-first-review-order -->
Inside confidence's review set on Bolivia the error rate is 0.38 at a 5% budget, 0.33 at 10% and 0.26 at 20%, against
8.8% overall (exp37). <!-- claim:review-set-error-rate -->
Across GEOID-Flood events the boundary-first order beats confidence at 5% on a minority, 27% to 40% of events by
target, and ties pooled (exp55). <!-- claim:geoid-boundary-first-minority -->

Each flagged window lists its label-free cues. A cue's enrichment, its share among error windows over its share among
correct ones, is 3.5 on Bolivia for a prediction boundary (0.750 against 0.214), 3.6 for the least confident 20%
(0.589 against 0.163) and for tiling instability (0.583 against 0.164), and 7.2 for spectral ambiguity, NDWI near zero
(0.483 against 0.067), with error rates of 0.25, 0.26, 0.26 and 0.41 among windows carrying each (exp37).
<!-- claim:cue-enrichment-table -->
On the suite's seven segmentation tasks the boundary cue's enrichment runs from 1.24 on m-cashew-plant, where 76% of
windows sit on a boundary, to 8.13 on MADOS, where 10% do; its rank correlation is 1.00 with the ceiling that boundary
prevalence sets and -0.26 with class count, and the preregistered bar of 2 fails on m-cashew-plant (exp82).
<!-- claim:boundary-cue-enrichment-is-fragmentation-not-class-count -->
Coarsening OlmoEarth Base's map to 2 × 2 and 4 × 4 blocks lowers the cue's enrichment on all seven tasks at both
steps (exp82). <!-- claim:boundary-cue-weakens-with-the-window-grain -->

The fine-tuned AWF model is 0.93 accurate where it claims 0.99, so a stated accuracy needs a coverage; its 80% most
confident windows are 0.945 accurate against 0.881 overall (exp21). <!-- claim:accuracy-needs-coverage -->
The served land-cover-change rasters export no class confidence, and boundary fraction alone captures a median 0.88
of their disagreements with WorldCover water at a 5% budget, on the sites that have any (exp20).
<!-- claim:served-product-boundary-triage -->

## Comparing two inferences

The comparison module measures how two inferences of one area differ. On all 16 preregistered pairs, across crop
offsets, backbones, sensors, encoders and fine-tuning, disagreement windows lie on a prediction boundary 3.3 to 7.1
times as often as agreeing ones on Sen1Floods11 and 21 times on the median GEOID-Flood event (exp57).
<!-- claim:atlas-disagreement-is-boundary-located -->
Reading Sentinel-1 instead of Sentinel-2 moves a flood model's error set far more than swapping the backbone (phi, the
correlation of two error masks, 0.38 to 0.39 against 0.70), so the sensor is the largest lever and the backbone the
smallest (exp46). <!-- claim:modality-dominates-shared-errors -->
Fine-tuning corrects 35 of the frozen probe's 63 errors on the same AWF windows, so the fixed window grid is not their cause
(exp21). <!-- claim:fine-tuning-corrects-half -->

Labels favour one side only where the difference changed the model; the fine-tuned model is right on 71% to 75% of
its disagreements with the frozen head, while under crop offsets and backbones either side is right on 39% to 61%
(exp57). <!-- claim:atlas-which-side -->
Without labels, the side with the larger margin is right on 51% to 70% of disagreement windows, above a coin flip on
all 15 pairs and short of the preregistered 55% on four pairs, so the prediction fails (exp58).
<!-- claim:tool-vs-diff-resolution -->
A side rule fitted to labels on frozen pairs and forced onto the frozen-against-fine-tuned pair is right on 43.8% and
49.2% of differing windows, below the raw margin's 61.1% and 55.4% and the pair's own fit's 74.9% and 72.8%, so the
module refuses another model family unless forced (exp65). <!-- claim:calibrate-family-lock -->

Two Sentinel-2 acquisitions a median 118 days apart change decision on 31.2% of 7,109 LUCAS polygons against a 0.30%
reseed floor, a rate that follows phenology, 44.5% on cropland against 20.8% on woodland (exp68).
<!-- claim:lucas-two-dates-are-phenology -->
Because EuroCrops labels both years, the floor for a two-date change rate is measurable there; where the declared crop
did not change, 0.060 to 0.366 of windows change decision, against 0.589 to 0.843 where it did, a floor one to two
orders of magnitude above the reseed rate (exp69). <!-- claim:eurocrops-the-labelled-floor-for-a-two-date-difference -->
Where two years' inferences differ, the model is right about both on 0.051 to 0.367 of windows, the difference being
the model following a real crop rotation (exp69). <!-- claim:eurocrops-a-difference-can-be-the-model-tracking-the-ground -->

## Estimating accuracy from a labelled sample

From 300 random windows the design-based 95% interval for the error rate covers the truth on 0.933 to 0.961 of 2,000
draws on all seven segmentation tasks (exp78). <!-- claim:design-based-interval-is-honest -->
The same budget spent as 19 tiles of 16 windows gives an ordinary independent-sample interval that covers on 0.506 to
0.777 of draws on six of the seven; a cluster correction restores 0.913 to 0.932 on the four tasks with full tiles and
fails on MADOS, whose tiles hold 1 to 400 windows (0.598) (exp78). <!-- claim:tile-sampling-breaks-the-naive-interval -->
Choosing the labelled windows by confidence saves up to 2.51 times the labels at equal half-width, above the 1.8 the
preregistered honesty check allowed (exp78). <!-- claim:confidence-saves-a-quarter-to-a-half-of-the-labels -->

The same labels certify a trust zone, the largest most-confident part of the map with error rate at most α, and the
certificate may fail with probability at most δ. At δ = 0.1 the zone certified by the exact-test rules exceeded α on
at most 0.080 of 2,000 draws in all 112 cells of the suite, and the plug-in rule's on up to 0.557 (exp80). <!-- claim:trust-zone-guarantee-holds -->
With 300 labels and α at half the error rate, the prefix rule certifies a zone on most draws on 14 of 21 tasks,
covering a median 0.50 of the map (exp80). <!-- claim:trust-zone-coverage-at-300-labels -->
The package's intervals for user's accuracy (how often a class call is right), producer's accuracy (how much of a class is found)
and class share leave 14 of 628 cells below 0.93 coverage and none below 0.879; the Wald form common in the literature
leaves 107 of 522, the worst at 0.019 (exp81). <!-- claim:per-class-intervals-wald-fails-wilson-nearly-holds -->
Choosing fine-tuning tiles by the audit's suspicion is worse than random on the multi-region test split at every
budget from 100 to 1,000 labels, and the preregistered predictions of a saving fail (exp56).
<!-- claim:audit-does-not-save-labels -->

## The package as an agent tool

On forty task cards with three preregistered predictions (exp64), Qwen3.8-27B-NVFP4 with the package as tools
reproduces its review set, grounds 99.4% of its stated numbers in tool outputs and declines to pick a side on every
comparison card; the OlmoEarth Agent as shipped found the tools unaided and did the same.
<!-- claim:agent-benchmark-tool-arm-reproduces-the-package -->
A model given only numpy and the arrays captures 0.904 of the package's errors, so the ranking prediction fails.
<!-- claim:agent-benchmark-sandbox-rediscovers-the-ranking -->
It grounds 94.2% of its numbers against 99.4%, short of the preregistered gap, so the grounding prediction fails.
<!-- claim:agent-benchmark-grounding-not-decisive -->
The third prediction holds, the tool arm declining the side question on 10 of 10 comparison cards against 1 of 10,
because the package carries the fact that the more confident side is right on only 51% to 70% of differing windows. <!-- claim:agent-benchmark-decline-holds -->
With Qwen2.5-7B-Instruct all three predictions hold, the tool arm grounding 100.0% against 0.1% and capturing 0.997
of the package's errors against 0.144. <!-- claim:agent-benchmark-package-is-a-floor-at-7b -->
At that size the agent as shipped calls the review-set tool on 17 of 40 runs, rejecting the prediction that it
reproduces the package. <!-- claim:agent-benchmark-7b-agent-does-not-find-the-tool -->

## What was tried and rejected

The [technique ledger](TECHNIQUES.md) lists each rejected technique with its reason. No signal from the encoder's
internals, its pretraining objective, a posterior over the probe head, feature-space typicality, a second model of the
same family, or flip-and-rotate consistency ranks errors better than confidence on expert labels.
<!-- claim:no-encoder-internal-signal-beats-confidence -->
As a side result, OlmoEarth v1's latent masked-image-modelling pretraining target has effective rank 2 on real
scenes, and a whitened target is 57% to 70% predictable from context (exp32 to exp34).
<!-- claim:target-effective-rank-2 -->

## Limits

The hand-labelled flood testbeds are one event and one split of the same dataset, and the suite is read through
linear probes on Ai2's embeddings. A confidence ranking reviews the model's confident errors last. On Sen1Floods11,
where eight encoders share 82% to 87% of their errors, a Dawid-Skene consensus, which estimates each encoder's
accuracy from agreement alone, returns 0.975 to 0.983 for maps 0.883 to 0.914 accurate, and its rank correlation with
the true accuracies misses the preregistered 0.8 on MADOS (0.71) (exp83).
<!-- claim:consensus-order-recovered-only-where-true-gaps-are-large -->
Open items are in the [roadmap](plan/roadmap.md).

## How a claim is recorded

A candidate signal is scored beside the model's confidence and a no-model control on identical windows, with the
primary test written down before the run; expert labels grade a signal and never train it. Each claim is an entry in
`docs/claims.yaml` whose check must hold on its artifact, and load-bearing claims carry a second test that recomputes
the statistic. The full rules are in the [protocol](method/protocol.md).

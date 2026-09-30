# Summary

This page lists what has been measured about the package, its known limits and how it works, in short. The
[README](https://github.com/2imi9/olmoearth_inferenceX#readme) has the quick start, and [Usage](Usage.md) describes
every command.

## What has been measured

Each line is a claim in the project's ledger, checked against its result file. The
[findings](Findings.md#in-short) list them
all, with the evidence.

- **Ranking.** Ai2's published embedding suite has 25 tasks for OlmoEarth Base. A confidence is
  defined on 24 of them; the other is multi-label. Read through linear probes on those
  embeddings, confidence ranked the errors better than both baselines that do not use the model
  (how rare the predicted class is, and distance in embedding space) on all 24 tasks. <!-- claim:suite-margin-wins-every-task -->
- **Other encoders.** Sixteen encoders of that suite carry at least 20 of the 24 tasks
  (OlmoEarth, Galileo, CROMA, TerraMind, Clay, Copernicus-FM, AnySat, Panopticon and Satlas).
  Under each of ten probe seeds, confidence beat both baselines on at least 87.5% of each
  encoder's tasks. <!-- claim:exp79-headline-holds-under-every-seed-on-every-encoder -->
- **Other references.**
  Against ground survey labels (LUCAS), confidence ranked the errors better than the best
  baseline there, the variance of the pixels inside a window, in 94 regions and worse in 32. <!-- claim:lucas-ranking-survives-ground-observation -->
  Against farmers' crop declarations (EuroCrops), it did so in all three countries. <!-- claim:eurocrops-ranking-holds-on-declarations -->
  For Dynamic World, a model this project did not train, the gap between its two highest class
  probabilities beat a class-rarity baseline on 315 expert-annotated tiles and lost on 91. <!-- claim:dw-margin-ranks-a-production-model -->
- **Error rate.** From 300 randomly drawn windows, a 95% Wilson interval held the true rate in
  93.3% to 96.1% of 2,000 repeated draws, on each of the suite's seven segmentation tasks. <!-- claim:design-based-interval-is-honest -->
  `estimate` now prints the exact hypergeometric interval instead, which covers at least 95% by
  construction.
- **Certified zone.** On OlmoEarth Base's 24 tasks, a certified zone was worse than its level
  in at most 8% of 2,000 draws, where 10% is allowed. <!-- claim:trust-zone-guarantee-holds -->

## Known limits

The [findings](Findings.md#limits) give the
detail.

- **Confident errors.** They are checked last, and a missing input can make more of them.
  Three crop tasks were simulated: a probe trained on Sentinel-1 and Sentinel-2 embeddings,
  then read on Sentinel-1 alone, as under full cloud. On PASTIS, the largest, 59.8% of
  OlmoEarth Base's errors were then as confident as a typical correct window, against 6.0%
  with both inputs. For OlmoEarth Large the share was 12.8% to 13.8%. <!-- claim:missing-optical-errors-are-confident -->
  On Togo 12 the share rose from 10.4% to 48.5%. On China 6 it fell, from 13.3% to 6.5%. <!-- claim:missing-modality-confident-errors-follow-the-shift-not-the-sensor -->
  No real cloud has been tested.
- **Kind of model.** Most of the evidence is linear probes on frozen embeddings. Of Ai2's
  fine-tuned models, one was tested: FT-AWF, on 344 validation points. <!-- claim:fine-tuned-model-audit -->
- **Two maps.** Without labels, `compare` cannot say which map is right where they differ. On
  15 pairs of flood maps, trusting the more confident map was right on 51% to 70% of the
  differing windows. <!-- claim:tool-vs-diff-resolution -->
- **Floods.** A plain water index (NDWI) ranked the errors as well as or better than confidence
  on one Sen1Floods11 event, Bolivia. <!-- claim:bolivia-ndwi-exception -->
  For a Sentinel-1 probe it ranked them better on the whole multi-region test split. <!-- claim:s1-probe-ndwi-flip -->
- **Certifying.** `certify` can return nothing. With 300 labels and a level of half the map's
  error rate, it found a zone on most draws on only 14 of 21 tasks. <!-- claim:trust-zone-coverage-at-300-labels -->
- **Labels collected by tile.** The coverage measured above is for windows drawn at random.
  300 labels collected as 19 whole tiles and treated as independent gave a 95% interval that
  held the true rate in only 51% to 78% of draws, on six of seven tasks. For labels drawn with
  `--design tiles`, `estimate` corrects for the tiles. Its interval still fell short on those
  six: 91% to 93% on four, 82% on Sen1Floods11 and 60% on MADOS, whose tiles differ in size. <!-- claim:tile-sampling-breaks-the-naive-interval -->
- **Scope and size.** The package covers single-label classification. Multi-label maps are not
  covered, and a regression map is read by `compare` only. Each command reads the whole raster
  into memory.

## How it works

<img src="figures/pipeline.png" alt="One scene through the assessment: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the cues per flagged window" width="760">

*One scene through `assess`. The figure shows the `--order boundary_first` option and two cues, tiling instability and NDWI, that only the Python API computes.*

1. **Ranking.** A window's confidence is the mean over its valid pixels, and the least
   confident windows come first. `--order boundary_first` puts the windows on a class boundary
   ahead of the rest.
2. **Comparison.** Two maps are pooled to one window grid. `compare` reports the share of
   windows whose class differs, where they are, and how much more often they sit on a class
   boundary. Across two dates a difference can be real change on the ground, so grading against
   labels requires the labels' date.
3. **Estimation.** `sample` draws the windows by a recorded random design, and `estimate` uses
   that design. The interval is exact hypergeometric for a simple random sample, a Wilson
   interval at the effective sample size for a stratified one, and cluster-corrected for
   labels collected by tile. The last one under-covers (see [Known limits](#known-limits)).
4. **Certified zone.** Exact hypergeometric tests run on zones of growing size, most confident
   windows first. They give the largest zone with error rate at most `α`, at error probability
   `δ`. With no error among its labels, a zone needs about `ln δ / ln(1 − α)` labels: 45 at
   `α` = 5% and `δ` = 0.1.

The formulas are in the
[protocol](method/protocol.md#the-estimators-closed-forms),
and the record of each experiment in the
[comparisons](results/comparisons.md).

olmoearth_inferenceX
====================

olmoearth_inferenceX is a Python package for checking a classification map made by an
Earth-observation model. It answers three questions about the map: which parts to check first,
how wrong the map is, and which part of it can be trusted. It reads the scores the model wrote
out, not the model itself, so it needs neither the model's weights nor torch.

Documentation: https://olmoearth-inferencex.readthedocs.io/


What you give it
----------------

One raster of the model's scores, taken before the argmax:

- a GeoTIFF or a `.npy` array;
- shape `(C, H, W)` with one band per class, or `(H, W)` for a two-class map;
- probabilities between 0 and 1 (not 0 to 100), or logits with `--logits`.

The class map alone is not enough. Without scores there is nothing to rank by, and only
`compare` reads a plain class map. Labels are needed only for an error rate, and the package
picks which windows to label.

The package works on **windows**: square blocks of pixels, 4 x 4 by default (`--patch`). These
are not rslearn windows. It ranks, samples and counts windows, not pixels. A window's
**confidence** is how sure the model is of its class: the top class probability, or with
`--logits` the gap between the two highest logits, averaged over the window's pixels.


Quick start
-----------

The package needs Python 3.11 to 3.13.

```bash
pip install "olmoearth-inferencex[geo]"    # 1.3.1; without [geo] it reads .npy only, no GeoTIFF
oe-inferencex demo
```

The demo needs no data. It assesses a real land-cover map shipped with the package, one Dynamic
World tile with expert annotation. It prints what it found and writes
`oe_inferencex_demo/review_set.png`. A **review set** is the least confident share of the
windows, here 5%, the ones to check first:

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/demo_real_map.png" alt="Three panels of a real land-cover map in southern Peru. Left: the 5% of windows ranked first, outlined in black along the class boundaries. Middle: the same windows over the map's errors in red. Right: a random 5% of windows over the same errors" width="760">

*Left: the 5% review set, outlined in black, chosen without labels. Middle: the same windows over the map's errors, in red. 67% of them are wrong, against 19% of the whole map. Right: a random 5% of the windows, for comparison. The review set holds 17% of the map's errors; no 5% of a map that is 19% wrong could hold more than 26%.* <!-- claim:demo-sample-hit-rate -->

The tile was chosen by a rule fixed in advance: the median of 18 eligible tiles, not the best. <!-- claim:demo-sample-is-the-median-tile -->

**Your own map** takes three steps. The lines below were printed by 1.3.1 on a synthetic
four-class test map of 256 x 256 pixels. Its true error rate is 9.0%, and a script filled in
the labels from the known truth.

**1. Which parts to check first.** No labels are needed. Add `--logits` if the scores are
logits.

```console
$ oe-inferencex assess scores.tif --out audit
4096 windows of 4 px; review sets 1%: 41, 5%: 205, 10%: 410; boundary windows 52.6%
wrote audit/assessment.json, explanation.json, review_set_*.csv, suspicion, boundary
```

`audit/review_set_05pct.csv` lists the 5% review set, least confident first, with each
window's pixel and map coordinates. There is one such file for 1% and for 10%.
`explanation.json` gives the cues of each of those windows: whether it sits on a boundary
between two classes of the map, and whether it is among the least confident 20%.
`suspicion.tif` is the ranking as a raster.

**2. How wrong the map is.** This needs labels, and `sample` picks the windows. The recorded
experiments used 300.

```console
$ oe-inferencex sample scores.tif --budget 300 --design random --out to_label.csv
300 windows to label of 4096 valid (random design); wrote to_label.csv and its .json. Fill the `wrong` column with 1 or 0 per window, then run: oe-inferencex estimate to_label.csv
warning: probability input: confidence ties where probabilities saturate; prefer logits
```

Open `to_label.csv`. For each row, look at the window in imagery or on the ground. Set `wrong`
to 1 if `map_class` is not what is there, otherwise to 0. Fill every row, keep the row order, and keep
`to_label.json` beside the CSV.

```console
$ oe-inferencex estimate to_label.csv
error rate 9.7%, 95% interval 6.7% to 13.4% (half-width 3.4 points), from 300 labelled windows of 4096; exact hypergeometric interval (simple random sample of a finite map)
wrote to_label_estimate.json
```

With a `reference_class` column filled in as well, `--per-class` adds each class's accuracy.

**3. Which part can be trusted.** The same labels give a **certified zone**: the most confident
share of the map whose error rate is at most the level `--alpha`. The statement may be wrong for
at most 10% of the samples that could have been drawn (`--delta`).

```console
$ oe-inferencex certify to_label.csv --alpha 0.10
the 95% most confident windows (3891 of 4096, confidence margin >= 0.3876) are wrong at most 10% of the time; this statement fails on at most 10% of samples like this one (prefix rule; the exact upper bound on the zone's error rate at that level is 7.9%). Outside the zone nothing is certified.
valid if the zone's error rate does not fall as the zone grows; on the suite tasks exp80 graded, the guarantee held whether or not that was exactly true (docs/results/comparisons.md, exp80)
wrote to_label_zone.json and the window mask to_label_zone.npy
```

`certify` can return nothing. With `--alpha 0.05` the same labels gave `no zone certified`;
adding `--rule bonferroni` certified the most confident 50% of the map at that level. `certify`
needs a sample drawn with `--design random`. Without that option `sample` stratifies by
confidence, which `estimate` reads and `certify` refuses.

**Two maps of one area**, on the same grid:

```console
$ oe-inferencex compare scores.tif other.tif --out diff
540 of 4096 windows differ (13.18%); on a boundary of a 2.1x as often as the agreeing windows
wrote diff/comparison.json, differing_windows.csv, disagreement
```

The second figure says that the differing windows sit on a class boundary of the first map
2.1 times as often as the agreeing windows.


Commands
--------

| Command | The question it answers | Labels |
|---|---|---|
| `demo` | What does the output look like on a real map? | none |
| `assess` | Which windows should a reviewer check first? | none |
| `compare` | Where do two maps of one area differ? With `--labels`: which map is right there? | optional |
| `sample` | Which windows should be labelled? | none |
| `estimate` | What is the map's error rate, with a 95% interval? With `--per-class`: how accurate is each class? | the sample, labelled |
| `certify` | Which share of the map, most confident first, has an error rate below a chosen level? | the sample, drawn with `--design random`, labelled |

`oe-inferencex <command> --help` lists the options.
[Usage](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/) describes every command,
the files it writes and the Python functions behind it. It also lists
[the other inputs the package accepts](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#inputs)
and shows
[how to write the scores out of a fine-tuned OlmoEarth model](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#scores-from-a-public-fine-tuned-model).


Before you trust it
-------------------

1. **It needs the scores, not only the class map.** `assess` refuses a class map of several
   classes. It does not refuse a 0/1 class map, or any class map passed with `--logits`: it
   reads the class ids as scores, the windows tie, and the review set is then ordered by raster
   position. The warning is only in `assessment.json`.
2. **Without labels it says where to look, not how wrong the map is.** For an error rate, label
   the windows that `sample` draws. The review set is not such a sample. It is chosen to hold
   errors, so its error rate is far above the map's, and `estimate` and `certify` refuse it.
3. **Confidence is only comparable between windows read from the same inputs**, meaning the
   same sensors and the same cloud state. An error the model is sure of is checked last, and
   a missing input can make the model sure and wrong. Version 1.3.1 ranks all the windows
   together and prints no warning about this.


Released and not yet released
-----------------------------

Version 1.3.1 is on PyPI. Every `oe-inferencex` command on this page runs on it.

`--condition` is on the `main` branch only. It takes a raster of each pixel's input condition,
such as a cloud flag, and gives each condition its own review set, error rate and certified
zone. It does not improve the ranking inside a part read with an input missing. To use it
before the next release:

```bash
pip install "olmoearth-inferencex[geo] @ git+https://github.com/2imi9/olmoearth_inferenceX"
```


What has been measured
----------------------

Each line is a claim in the project's ledger, checked against its result file. The
[findings](https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short) list them
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
  baseline in 94 regions and worse in 32. <!-- claim:lucas-ranking-survives-ground-observation -->
  Against farmers' crop declarations (EuroCrops), it did so in all three countries. <!-- claim:eurocrops-ranking-holds-on-declarations -->
  For Dynamic World, a model this project did not train, its own confidence beat a class-rarity
  baseline on 315 expert-annotated tiles and lost on 91. <!-- claim:dw-margin-ranks-a-production-model -->
- **Error rate.** From 300 randomly drawn windows, the 95% interval held the true rate in 93.3%
  to 96.1% of 2,000 repeated draws, on each of the suite's seven segmentation tasks. <!-- claim:design-based-interval-is-honest -->
- **Certified zone.** On OlmoEarth Base's 24 tasks, a certified zone was worse than its level
  in at most 8% of 2,000 draws, where 10% is allowed. <!-- claim:trust-zone-guarantee-holds -->


Known limits
------------

The [findings](https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#limits) give the
detail.

- **Confident errors.** They are checked last, and a missing input can make more of them. One
  case was simulated: a PASTIS probe trained on Sentinel-1 and Sentinel-2 embeddings, then read
  on Sentinel-1 alone, as under full cloud. 59.8% of OlmoEarth Base's errors were then as
  confident as a typical correct window, against 6.0% with both inputs. For OlmoEarth Large the
  share was 12.8% to 13.8%. No real cloud has been tested. <!-- claim:missing-optical-errors-are-confident -->
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
- **Labels collected by tile.** The interval holds for the windows `sample` draws. 300 labels
  collected as 19 whole tiles and treated as independent gave a 95% interval that held the true
  rate in only 51% to 78% of draws, on six of seven tasks. <!-- claim:tile-sampling-breaks-the-naive-interval -->
- **Scope and size.** The package covers single-label classification. Multi-label maps are not
  covered, and a regression map is read by `compare` only. Each command reads the whole raster
  into memory.


How it works
------------

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/pipeline.png" alt="One scene through the assessment: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the cues per flagged window" width="760">

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
   labels collected by tile.
4. **Certified zone.** Exact hypergeometric tests run on zones of growing size, most confident
   windows first. They give the largest zone with error rate at most `α`, at error probability
   `δ`. With no error among its labels, a zone needs about `ln δ / ln(1 − α)` labels: 45 at
   `α` = 5% and `δ` = 0.1.

The formulas are in the
[protocol](https://olmoearth-inferencex.readthedocs.io/en/latest/method/protocol/#the-estimators-closed-forms),
and the record of each experiment in the
[comparisons](https://olmoearth-inferencex.readthedocs.io/en/latest/results/comparisons/).


Development and license
-----------------------

To work on the repository, clone it, then run `uv sync` and `uv run pytest`. The experiments
also require `uv sync --extra encoder --extra geo`.

- License: Apache License 2.0 ([LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE)).
- Citation: [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff).
- Questions and suggestions: [GitHub issues](https://github.com/2imi9/olmoearth_inferenceX/issues).

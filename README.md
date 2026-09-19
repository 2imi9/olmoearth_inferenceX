olmoearth_inferenceX measures differences between Earth-observation inferences
without labels, and shows on expert-labelled testbeds which of those differences
predict error. It was built around OlmoEarth and has since been run on a served
global product no one here had a hand in training.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/pipeline.png" alt="One scene through the audit: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the reasons per flagged window" width="760">

Try it in two minutes
---------------------

```bash
pip install olmoearth-inferencex
oe-inferencex demo
```

No data to find, no labels, nothing but numpy. The command audits a real map: one tile of Dynamic World, a global
10 m land-cover product this project had no hand in, with the probabilities it publishes about itself and an expert's
annotation of the same ground to grade the result. The tile was chosen by a rule fixed in advance, the median tile by
error capture among the 18 fully annotated ones of its 409 public test tiles, so it is a typical tile and not the best one. <!-- claim:demo-sample-is-the-median-tile -->

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/demo_real_map.png" alt="Three panels of a real land-cover map in southern Peru. Left: the map with the 5% of windows to check first outlined in black, along the boundaries between classes. Middle: the same windows over the map's real errors in red; 67% of the flagged windows are wrong. Right: a random 5% of windows over the same errors; about 19% of them are wrong" width="760">

*Left: what an audit gives, the map with the 5% of windows to check first outlined in black; no labels were used.
Middle: the same windows over the places where the map is really wrong according to the expert, in red. Right: a random
5% over the same errors. The map is wrong on 19% of its windows; of the windows the tool flags, 67% are wrong, so a
reviewer who goes where it points finds an error more than three times as often as one who picks at random, and those 5%
hold 17% of all the map's errors. The run also says what the tool does not do: errors the model is sure about stay* <!-- claim:demo-sample-hit-rate -->
*hidden, and it never says how wrong a map is. `oe-inferencex demo --made-up` does the same on a small synthetic water map.*

Then your own map, a GeoTIFF (with `pip install "olmoearth-inferencex[geo]"`) or a `.npy` array of probabilities or
logits:

```bash
oe-inferencex assess your_map.tif --out audit
```

What the project found, in six plain sentences: [Findings, in short](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Findings.md#in-short).

The strongest evidence
----------------------

**On tasks this project did not choose, the model's own margin ranks its errors
better than any control that sees no model — on 24 of 24.**

The tasks are the 25 of the [OlmoEarth paper embedding suite](https://huggingface.co/datasets/allenai/olmoearth-paper-embeddings)
by Ai2, with the splits fixed in the files. On all 24 that a top-1 minus top-2 margin is
defined for, and on all 14 distinct sources behind them, the margin beats the best
no-model control: 6,435,473 graded units, an accuracy range from 0.333 to 0.979,
17 of 17 classification tasks and 7 of 7 segmentation tasks, sign test p = 6e-08.
Six of the tasks come from GEO-Bench 1, a third-party benchmark, and the margin
wins on all six. On the same tasks it also beats the competitors the literature
proposes: a five-seed ensemble on 22 of 24, nearest-neighbour typicality on
24 of 24 and a Mahalanobis distance on 24 of 24
([exp73](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#the-strong-alternatives-on-the-same-suite-exp73)).
Nor is it a property of OlmoEarth: under the fifteen other encoders the suite
is published for, eight families outside OlmoEarth (AnySat, Clay,
Panopticon, Galileo, CROMA, TerraMind, Satlas, Copernicus-FM) and the OlmoEarth
size series, the margin beats the control on 322 of 332 scored tasks, every
encoder at 90% or better
([exp74](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#the-suite-under-the-other-encoders-exp74)).
Within the model's own confidence family the forms are close; on multi-class
tasks one minus the top probability is marginally better than top-1 minus top-2
([exp76](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#which-confidence-which-statistic-which-aggregator-exp76)).

This is the answer to the obvious objection — that a result like this rests on
testbeds the author picked. It does not.
[Tasks we did not choose (exp70)](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#tasks-we-did-not-choose-ai2s-whole-published-suite-exp70).

*What this is not: a leaderboard result. That suite measures accuracy, and no
public benchmark measures label-free error ranking.*

What it does
------------

On these testbeds the model's own logit margin is the difference that predicts
error; disagreement across crops, backbones or encoders does not rank errors; the
sensor difference says where shared errors come from; the frozen-versus-fine-tuned
difference measures how much training moved the model. Given a prediction map:

1. **Which windows to review first** — review sets at a chosen budget, in
   confidence order or boundary first.
2. **Why each flagged window is suspect** — label-free cues carrying measured
   evidence from expert-labelled testbeds.
3. **Score any candidate audit rule** the same way, against the model's own
   confidence and a no-model control, on two references at once.
4. **Audit a deployed product**, whoever trained it — including a served global
   land-cover product audited from nothing but the probabilities it publishes
   about itself.
5. **Measure the difference between two inferences** of one scene: across crops,
   backbones, sensors, encoders, fine-tuning, and acquisition dates — how much
   they disagree, what the disagreeing windows share, and with labels, which side
   is right.
6. **Fuse label-free readings with labels** where they exist, reported held-out
   and bound to the model family it was fitted on, because such rules do not
   transfer across families.
7. **Grade against a reference that is a sample, not a map** — design-weighted
   estimators report the population quantity. Ignoring the design overstated one
   lead by a quarter of its size.

Both halves run from the command line without writing Python:
`oe-inferencex assess` and `oe-inferencex compare`
([Usage](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Usage.md#command-line)).

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/compare.png" alt="The two-period, two-sensor square on one GEOID-Flood chip: four dated inputs, four inferences on identical windows, the same-date and same-sensor differences on the scene, and the label bridge boxed apart" width="760">

*Four dated inputs of one GEOID-Flood chip (Lake Shkodër at Gruemirë, Albania),
each read on identical windows. Same-period pairs differ on 22 windows before the
event and 44 after; same-sensor pairs on 97 (optical) and 67 (radar). What a
difference **is** needs labels, so that panel is boxed apart. Full numbers in
[Comparisons](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57).*

What the evidence spans
-----------------------

The ranking result rests on five kinds of reference whose errors fail in different
ways, which is the part of this work hardest to argue with:

| The answer key came from | Testbeds |
|---|---|
| people reading the same imagery | [Sen1Floods11 hand labels](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#dense-flood-masks-sen1floods11-exp18), [Copernicus EMS through GEOID-Flood](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#geoid-flood-the-exception-as-a-rate-over-events-exp55), [WorldFloods v2](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#the-fourth-cell-post-event-optical-from-worldfloods-completes-the-square-exp62) |
| another model's output | [DFC2020](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#dfc2020-eight-class-land-cover-both-sensors-our-own-encoder-two-references-exp66), whose test labels are an iterated random forest, not hand-drawn |
| experts annotating a served product | [Dynamic World's 409 expert tiles](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#auditing-a-production-model-with-its-own-probabilities-dynamic-world-exp67), audited from its own published probabilities |
| surveyors standing in the field | [LUCAS Copernicus 2022](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#the-protocol-against-ground-observation-lucas-exp68), the only reference here that never saw a pixel |
| farmers' own declarations | [EuroCrops](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md#eurocrops-a-dense-crop-map-from-declarations-and-a-difference-labelled-on-both-sides-exp69), which also labels both sides of a two-date comparison |

Two warnings for practitioners
------------------------------

**Report your readout's generalisation gap beside any ranking comparison.** A
small readout that has memorised its practice data reverses which confidence
signal ranks best. On LUCAS the margin went from last of four model signals to the
front, on the same data, purely from choosing the probe's regularisation on
held-out ground.

**A two-date difference needs the right floor.** How often the map moves where the
ground did not is 6% to 37% on crops — one to two orders of magnitude above the
rate at which merely reseeding the readout moves it. Comparing a date difference
against the reseed rate flatters it.

Quickstart
----------

1. [Findings](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Findings.md) — what holds, the numbers, how a claim gets in,
   and the limits.
2. [Usage](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Usage.md) — assess a prediction, explain its review set, the
   production case, how to score a new rule.
3. [Recipe](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/recipe.md) — what to do and not do when auditing a map.
4. [Technical report](https://github.com/2imi9/olmoearth_inferenceX/blob/main/report/main.pdf): the whole record in ten pages, what was done,
   what held, and what did not.

Also: [TECHNIQUES](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/TECHNIQUES.md) (everything tried, one line each, with the
verdict) · [Protocol](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/protocol.md) (how results are scored) ·
[Explanation](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/explanation.md), [Signals](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/signals.md),
[Comparisons](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md) (per-cue, per-signal, per-experiment
evidence) · [Agent integration](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/agent_integration.md) ·
[Related work](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/related_work.md) (where each idea comes from) ·
[Roadmap](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/plan/roadmap.md).

Full documentation: **https://olmoearth-inferencex.readthedocs.io/**

Setup
-----

Python 3.11+ and numpy; nothing else for the package. To use it:

```bash
pip install olmoearth-inferencex            # add [geo] to read and write GeoTIFFs
oe-inferencex demo                          # a first run on a made-up map
oe-inferencex assess --help
```

The development version installs from the repository:
`pip install "olmoearth-inferencex @ git+https://github.com/2imi9/olmoearth_inferenceX"`.

[examples/quickstart.py](https://github.com/2imi9/olmoearth_inferenceX/blob/main/examples/quickstart.py) runs the whole surface on a
synthetic map with no data to download; the public API is what
`oe_inferencex.__all__` exports, and [CHANGELOG.md](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CHANGELOG.md) says what a
release contains. Rasters need the `geo` extra (`rasterio`); `.npy` input does
not.

To work on the repository, with the tests against the recorded numbers:

```bash
git clone https://github.com/2imi9/olmoearth_inferenceX.git
cd olmoearth_inferenceX
uv sync
uv run pytest
```

For the full experiment environment:

```bash
uv sync --extra encoder --extra geo
uv run python scripts/audit_one_scene.py   # one scene end to end
```

Licence
-------

Apache License 2.0; see [LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE). To cite the software or its
recorded results, see [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff).

Contact
-------

For questions and suggestions, please
[open an issue on GitHub](https://github.com/2imi9/olmoearth_inferenceX/issues).

# olmoearth_inferenceX

olmoearth_inferenceX measures differences between Earth-observation
inferences without labels, and shows on expert-labelled testbeds which of
those differences predict error. It was built around OlmoEarth and has since
been run on a served global product no one here had a hand in training.

## The strongest evidence

**On tasks this project did not choose, the model's own margin ranks its errors
better than any control that sees no model — on 24 of 24.**

The tasks are the 25 of the [OlmoEarth paper embedding suite](https://huggingface.co/datasets/allenai/olmoearth-paper-embeddings)
by Ai2, with the splits fixed in the files. On all 24 that a top-1 minus top-2 margin is
defined for, and on all 14 distinct sources behind them, the margin beats the
best no-model control: 6,435,473 graded units, an accuracy range from 0.333 to
0.979, 17 of 17 classification tasks and 7 of 7 segmentation tasks, sign test
p = 6e-08. Six of the tasks come from GEO-Bench 1, a third-party benchmark, and
the margin wins on all six. On the same tasks it also beats the competitors the
literature proposes: a five-seed ensemble on 22 of 24, nearest-neighbour
typicality on 24 of 24 and a Mahalanobis distance on 24 of 24
([exp73](results/comparisons.md#the-strong-alternatives-on-the-same-suite-exp73)).
Nor is it a property of OlmoEarth: under the fifteen other encoders the suite
is published for, eight families outside OlmoEarth (AnySat, Clay,
Panopticon, Galileo, CROMA, TerraMind, Satlas, Copernicus-FM) and the OlmoEarth
size series, the margin beats the control on 322 of 332 scored tasks, every
encoder at 90% or better
([exp74](results/comparisons.md#the-suite-under-the-other-encoders-exp74)).

This is the answer to the obvious objection — that a result like this rests on
testbeds the author picked. It does not. See
[Tasks we did not choose (exp70)](results/comparisons.md#tasks-we-did-not-choose-ai2s-whole-published-suite-exp70).

*What this is not: a leaderboard result. That suite measures accuracy, and no
public benchmark measures label-free error ranking.*

## What it does

On those testbeds the model's own logit margin is the difference that
predicts error; disagreement across crops, backbones or encoders does not
rank errors; the sensor difference says where the shared errors come from;
the frozen-versus-fine-tuned difference measures how much training moved
the model. Given a prediction map, it helps with:

1. **Deciding which windows to trust and which to send for review first**, as
   review sets at a chosen budget, in confidence order or boundary first.
2. **Explaining why each flagged window is suspect**, with label-free cues that
   carry measured evidence from expert-labelled testbeds.
3. **Scoring any candidate audit rule the same way**, against the model's own
   confidence and a no-model control, on two references at once.
4. **Auditing a deployed product**, whoever trained it: OlmoEarth's fine-tuned
   models through their task cards, the served land cover change rasters, and a
   served global land-cover product audited from nothing but the probabilities it
   publishes about itself.
5. **Measuring the difference between two inferences of the same scene**,
   through shifted crops, across backbones, across sensors, across encoders,
   before against after fine-tuning, and between two acquisition dates: how much
   they disagree, what the disagreement windows have in common and, with labels,
   which side is right.
6. **Fusing the label-free readings with labels where they exist**: a fitted
   ranker or side rule, reported held-out and bound to the model family it
   was fitted on, because such rules do not transfer across families.
7. **Grading against a reference that is a sample rather than a map**: the
   design-weighted estimators report the population quantity instead of the
   sample's, which on a stratified reference is not a fine point. Ignoring the
   design overstated one lead by a quarter of its size.

Both halves run from the command line without writing Python, `oe-inferencex
assess` and `oe-inferencex compare`; see
[Usage: command line](Usage.md#command-line).

![The two-period, two-sensor square on one GEOID-Flood chip: four dated inputs, four inferences on identical windows, the same-date and same-sensor differences on the scene, and the label bridge boxed apart](figures/compare.png)

*Four dated inputs of one GEOID-Flood chip (event EMSR273-1, Lake Shkodër at
Gruemirë, Albania), each read by its head on identical windows. The same-period
pairs differ on 22 windows before the event and 44 after; the same-sensor pairs
on 97 (optical) and 67 (radar). What a difference **is** needs labels, so that
panel is boxed apart. The event-level numbers are in
[Comparisons](results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57).*

A worked flood example, runnable from the committed artifacts, is in
[Usage: compare two inferences of the same scene](Usage.md#compare-two-inferences-of-the-same-scene);
the event-level results are in [Comparisons](results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57).

## What the evidence spans

The ranking result now rests on five kinds of reference whose errors fail in
different ways, which is the part of this work hardest to argue with:

| The answer key came from | Testbeds |
|---|---|
| people reading the same imagery | [Sen1Floods11 hand labels](results/comparisons.md#dense-flood-masks-sen1floods11-exp18), [Copernicus EMS through GEOID-Flood](results/comparisons.md#geoid-flood-the-exception-as-a-rate-over-events-exp55), [WorldFloods v2](results/comparisons.md#the-fourth-cell-post-event-optical-from-worldfloods-completes-the-square-exp62) |
| another model's output | [DFC2020](results/comparisons.md#dfc2020-eight-class-land-cover-both-sensors-our-own-encoder-two-references-exp66), whose test labels are an iterated random forest, not hand-drawn |
| experts annotating a served product | [Dynamic World's 409 expert tiles](results/comparisons.md#auditing-a-production-model-with-its-own-probabilities-dynamic-world-exp67), audited from its own published probabilities |
| surveyors standing in the field | [LUCAS Copernicus 2022](results/comparisons.md#the-protocol-against-ground-observation-lucas-exp68), the only reference here that never saw a pixel |
| farmers' own declarations | [EuroCrops](results/comparisons.md#eurocrops-a-dense-crop-map-from-declarations-and-a-difference-labelled-on-both-sides-exp69), which also labels both sides of a two-date comparison |

Two results a practitioner should carry away before using any of this.

**Report your readout's generalisation gap beside any ranking comparison.** A
small readout that has memorised its practice data reverses which confidence
signal ranks best. On LUCAS the margin went from last of the four model signals
to the front of them, statistically tied with the boundary-first order, on the
same data, purely from choosing the probe's regularisation on held-out ground
instead of leaving it at a default.

**A two-date difference needs the right floor.** How often the map moves where
the ground did not is between 6% and 37% on crops, one to two orders of
magnitude above the rate at which reseeding the readout moves it. Comparing a
date difference against the reseed rate flatters it.

## New to the repository?

For the short version, read [Findings](Findings.md): what holds, the numbers,
how a claim gets in, and the limits. Then:

- [Usage](Usage.md) walks through the package: assess a prediction, explain its
  review set, the production case with an exported confidence band, and how to
  score a new rule with the same machinery.
- The [Recipe](method/recipe.md) is the list of what to do and not do when
  auditing a prediction map.
- [Related work](related_work.md) says where each idea here comes from, with
  the selective-classification and Earth-observation literature it rests on.
- The [Technique ledger](TECHNIQUES.md) is everything tried, one line each,
  with the verdict and the evidence; the [Evidence](results/explanation.md)
  pages hold the per-cue, per-signal and per-experiment detail.
- The [API Reference](reference/index.md) documents the torch-free package
  `oe_inferencex` module by module.

## Installation

The package needs Python 3.11+ (3.12 is what the experiments ran on) and no
torch; the experiments need the encoder.

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

## Contact

For questions and suggestions, please
[open an issue on GitHub](https://github.com/2imi9/olmoearth_inferenceX/issues).

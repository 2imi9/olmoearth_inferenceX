olmoearth_inferenceX measures differences between Earth-observation inferences
without labels, and shows on expert-labelled testbeds which of those differences
predict error. It was built around OlmoEarth and has since been run on a served
global product no one here had a hand in training.

<img src="docs/figures/pipeline.png" alt="One scene through the audit: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the reasons per flagged window" width="760">

The strongest evidence
----------------------

**On tasks this project did not choose, the model's own margin ranks its errors
better than any control that sees no model — on 24 of 24.**

Ai2's published embedding suite holds 25 tasks they picked for their own paper,
with the splits fixed in the files. On all 24 that a top-1 minus top-2 margin is
defined for, and on all 14 distinct sources behind them, the margin beats the best
no-model control: 6,435,473 graded units, an accuracy range from 0.333 to 0.979,
17 of 17 classification tasks and 7 of 7 segmentation tasks, sign test p = 6e-08.
Six of the tasks come from GEO-Bench 1, a third-party benchmark, and the margin
wins on all six. On the same tasks it also beats the competitors the literature
proposes: a five-seed ensemble on 22 of 24, nearest-neighbour typicality on
24 of 24 and a Mahalanobis distance on 24 of 24
([exp73](docs/results/comparisons.md#the-strong-alternatives-on-the-same-suite-exp73)).

This is the answer to the obvious objection — that a result like this rests on
testbeds the author picked. It does not.
[Tasks we did not choose (exp70)](docs/results/comparisons.md#tasks-we-did-not-choose-ai2s-whole-published-suite-exp70).

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
([Usage](docs/Usage.md#command-line)).

<img src="docs/figures/compare.png" alt="The two-period, two-sensor square on one GEOID-Flood chip: four dated inputs, four inferences on identical windows, the same-date and same-sensor differences on the scene, and the label bridge boxed apart" width="760">

*Four dated inputs of one GEOID-Flood chip (Lake Shkodër at Gruemirë, Albania),
each read on identical windows. Same-period pairs differ on 22 windows before the
event and 44 after; same-sensor pairs on 97 (optical) and 67 (radar). What a
difference **is** needs labels, so that panel is boxed apart. Full numbers in
[Comparisons](docs/results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57).*

What the evidence spans
-----------------------

The ranking result rests on five kinds of reference whose errors fail in different
ways, which is the part of this work hardest to argue with:

| The answer key came from | Testbeds |
|---|---|
| people reading the same imagery | Sen1Floods11 hand labels, Copernicus EMS through GEOID-Flood, WorldFloods v2 |
| another model's output | DFC2020, whose test labels are an iterated random forest, not hand-drawn |
| experts annotating a served product | Dynamic World's 409 expert tiles, audited from its own published probabilities |
| surveyors standing in the field | LUCAS Copernicus 2022, the only reference here that never saw a pixel |
| farmers' own declarations | EuroCrops, which also labels both sides of a two-date comparison |

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

1. [Findings](docs/Findings.md) — what holds, the numbers, how a claim gets in,
   and the limits.
2. [Usage](docs/Usage.md) — assess a prediction, explain its review set, the
   production case, how to score a new rule.
3. [Recipe](docs/method/recipe.md) — what to do and not do when auditing a map.

Also: [TECHNIQUES](docs/TECHNIQUES.md) (everything tried, one line each, with the
verdict) · [Protocol](docs/method/protocol.md) (how results are scored) ·
[Explanation](docs/results/explanation.md), [Signals](docs/results/signals.md),
[Comparisons](docs/results/comparisons.md) (per-cue, per-signal, per-experiment
evidence) · [Agent integration](docs/method/agent_integration.md) ·
[Related work](docs/related_work.md) (where each idea comes from) ·
[Roadmap](docs/plan/roadmap.md).

Full documentation: **https://olmoearth-inferencex.readthedocs.io/**

Setup
-----

Python 3.11+ (3.12 is what the experiments ran on) and no torch; the experiments
need the encoder.

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

Contact
-------

For questions and suggestions, please
[open an issue on GitHub](https://github.com/2imi9/olmoearth_inferenceX/issues).

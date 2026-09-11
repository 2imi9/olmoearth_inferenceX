# olmoearth_inferenceX

olmoearth_inferenceX measures differences between OlmoEarth inferences
without labels, and shows on expert-labelled testbeds which of those
differences predict error.

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
4. **Auditing deployed OlmoEarth products**: the fine-tuned models through their
   task cards, and the served land cover change rasters.
5. **Measuring the difference between two inferences of the same scene**,
   through shifted crops, across backbones, across sensors and before
   against after fine-tuning: how much they disagree, what the disagreement
   windows have in common and, with labels, which side is right.

![Two inferences of one scene across two dates compared without labels: a GEOID-Flood chip before and after the event, the two decisions on identical windows, the differing windows on the scene, the cues they share, the label bridge boxed apart, the change reading, and whether two differences are the same set](figures/compare.png)

*One GEOID-Flood chip across two dates: the pre-event Sentinel-2 composite and the post-event
Sentinel-1 scene, 8.5 months apart, each read by its own head on identical windows. Without labels
the package reports how much the two decisions differ (152 of this chip's 196 windows, 3.4% over 55
events), where on the scene, what those windows have in common (the four cue layers of A, with how
much more often each fires on the differing windows than on the rest: boundary 10 times), and which
of them both sides are confident about. With labels, boxed apart, it says what the difference is:
here 84% of it is the flood itself and 16% an error of one head.*

The pair mixes time and sensor. exp60 separates the axes: a same-period cross-sensor
difference carries none of the later flood (0.4% of its windows against 19.5% here),
while the same-sensor difference across the event carries 26%, and more than half of
it is the pre-event radar seeing water the label's permanent class does not hold
([Comparisons](results/comparisons.md#two-periods-both-sensors-the-time-axis-and-the-sensor-axis-apart-exp60)).

A worked flood example, runnable from the committed artifacts, is in
[Usage: compare two inferences of the same scene](Usage.md#compare-two-inferences-of-the-same-scene);
the event-level results are in [Comparisons](results/comparisons.md#the-difference-atlas-every-pair-of-inferences-under-one-measurement-exp57).

## New to the repository?

For the short version, read [Findings](Findings.md): what holds, the numbers,
how a claim gets in, and the limits. Then:

- [Usage](Usage.md) walks through the package: assess a prediction, explain its
  review set, the production case with an exported confidence band, and how to
  score a new rule with the same machinery.
- The [Recipe](method/recipe.md) is the list of what to do and not do when
  auditing a prediction map.
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

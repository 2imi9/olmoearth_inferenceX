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

*One GEOID-Flood chip, two dates, two sensors (exp60): the Sentinel-2 composite and the Sentinel-1
pass of the pre-event date, and the Sentinel-1 pass 8.5 months later, each read by its head on identical
windows. The two differences isolate one axis each. Same date across sensors: 4 of 196 windows here,
4.9% over 55 events, and only 0.4% of those windows are the later flood, so the sensor difference is
sensor error alone. Same sensor across dates: 151 windows here, 87% of them the flood by the label; over
the 55 events 26%, because the pre-event radar also sees the seasonal water of that date, which the
label's permanent class does not hold. What a difference is needs labels, so that panel is boxed apart.*

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

# olmoearth_inferenceX

Experiments on locating errors in [OlmoEarth](https://allenai.org/olmoearth)
predictions over regions with no labels. Audited: linear probes on frozen
v1 encoders (a binary water head trained on one WorldCover-labelled scene,
a nine-class head on the AWF expert labels), Ai2's published fine-tuned AWF
model run end to end
([OlmoEarth-v1-FT-AWF-Base](https://huggingface.co/allenai/OlmoEarth-v1-FT-AWF-Base),
exp21), and its served land cover change rasters as published (exp20).
Everything uses Ai2's public artifacts: [olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain),
[HuggingFace checkpoints](https://huggingface.co/allenai/OlmoEarth-v1-Base),
and the datasets linked below.

## Result

Each signal gives every map window a suspicion score. The four signals:

- **E_case** - disagreement: two models (OlmoEarth v1
  [Nano](https://huggingface.co/allenai/OlmoEarth-v1-Nano) and
  [Base](https://huggingface.co/allenai/OlmoEarth-v1-Base)) predict the same
  window differently; or, within one model, its three Sentinel-2 band-set
  tokens do (exp17).
- **E_system** - tiling instability (tile-phase in the scripts): the
  prediction flips when the input grid shifts by a few pixels. It behaves
  like proximity to a boundary in the model's own prediction map (exp14).
- **E_dist** - embedding distance: the window looks unlike the scene the
  head was trained on.
- **E_geo** - map check: the prediction contradicts an
  [OSM river line](https://wiki.openstreetmap.org/wiki/Key:waterway). It
  mostly finds disagreements between reference maps, so it is not in the
  table (exp15).

AURC (area under the risk-coverage curve) measures how well a score ranks
the patches where the model disagrees with the reference; lower is better.
The reference is ESA WorldCover 2021 water on river scenes and expert point
labels on AWF. Every signal must beat the model's own confidence
(max-softmax) and a pixel statistic computed without any model (the
control). The served
[olmoearth_lcc rasters](https://huggingface.co/datasets/allenai/olmoearth_lcc)
export a change probability but no confidence for their class map (exp20).

<details open>
<summary><b>Headline comparison: three scenes</b> (click to shrink)</summary>

| AURC (lower = finds errors better) | In-domain ([AWF expert labels](https://huggingface.co/datasets/allenai/olmoearth_projects_awf), 63 errors) | Ambiguous wetland margins (Barotse, vs WorldCover 2021, 97 errors) | Reference omits the river (Zambezi delta, vs WorldCover 2021, 29 disagreements) | Hand-labelled flood masks (Sen1Floods11 Bolivia, pooled excess AURC over 81,984 patches) |
|---|---|---|---|---|
| model's own confidence (baseline) | **0.0363** | 0.0684 | 0.0234 | **0.0105** |
| cross-model disagreement (E_case) | 0.0670 | 0.0235 | 0.0103 | 0.0186 |
| tiling instability (E_system) | 0.0489 | **0.0127** | 0.0009 | 0.0115 |
| embedding distance (E_dist) | 0.1338 | 0.0289 | 0.0014 | 0.0592 |
| pixel statistic (control) | 0.1658 | 0.0384 | **0.0005** | 0.0356 |

Columns 2-3 score against ESA WorldCover, a weak reference; the third
scene's reference has no water, so its disagreements are reference
omissions. Column 4 scores against hand-labelled masks on a geographically
held-out region (exp18): there confidence is best and every other signal
is equal or worse.

</details>

Per-scene values for all 27 rule-selected scenes: [docs/results/comparisons.md](docs/results/comparisons.md).

## Findings

- **Against expert labels, the model's own confidence is the best
  error-finder.** On Sen1Floods11 hand-labelled flood masks (a
  geographically held-out region, 351 scored tiles) every audit signal is
  equal or significantly worse than confidence (exp18); on the AWF expert
  point task confidence also wins (exp04, exp16).
- **This holds for Ai2's fine-tuned AWF model run end to end.** The
  published checkpoint reproduces Ai2's accuracy (0.881 vs 0.895 on the
  validation split); confidence ranks its errors best, tiling instability
  ties it, and everything else is significantly worse (exp21). The model
  is overconfident: 0.93 accurate where it says 0.99, so a stated accuracy
  needs a coverage (0.945 at 80%).
- **The earlier wins were against a weak reference.** Tiling instability
  beat confidence on 26 of 27 scenes and band-set disagreement on 21 of 27
  when errors were defined against ESA WorldCover (exp13, exp17); neither
  transfers to hand labels. The likely reason is that much of what
  WorldCover called error was reference error, which boundary-type signals
  detect well.
- **Errors do concentrate at prediction boundaries** (about 75% of error
  patches vs 20% of correct ones, on both references), but confidence
  ranks them better than boundary proximity or instability do.
- **A disagreement partner needs a different view of the input, not a more
  accurate model** (exp10, exp17), though no disagreement signal beats
  confidence on expert labels.
- **The served product shows no inference-window striping, but its outputs
  are quantized to the 40 m patch grid.** On 5 windows of 37 km, the class
  boundaries and change-probability gradients peak on the encoder's 4-px
  patch lattice (19 of 20 profiles) and nowhere at the inference-window
  periods; seams affecting 5 to 10% of rows would have been detected
  (exp22).
- **On the served production rasters, boundary triage works and confidence
  cannot be tested.** The land cover change product exports no confidence
  for its class map; ranking its water map by prediction-boundary fraction
  captures a median 0.88 of WorldCover disagreements at a 5% review budget,
  and the change probability's ambiguity sits on the edges of flagged
  regions (exp20).
- **Caveat:** the water results are for linear probes on frozen encoders;
  one fine-tuned model has been run end to end, and the change product only
  assessed through its served output.

Negative and null results (cross-model disagreement, embedding distance,
the map check, masking perturbation) and every number behind the findings
are in [docs/results/comparisons.md](docs/results/comparisons.md) and
[docs/results/signals.md](docs/results/signals.md).

![No-model controls](exp/out/exp06_controls.png)

Docs: [recipe](docs/method/recipe.md) · [task cards](docs/method/taskcards.md) · [agent](docs/method/agent_integration.md) · [index](docs/TECHNIQUES.md) ·
[results](docs/results/comparisons.md) ·
[signals](docs/results/signals.md) ·
[protocol](docs/method/protocol.md) ·
[facts](docs/method/infrastructure.md) ·
[roadmap](docs/plan/roadmap.md) · [lab log](exp/NOTES.md)

## Method

This is selective prediction: score how likely each window is wrong, judge
the score by risk-coverage curves. Signals are label-free; labels only
evaluate them. Every split is a spatial hold-out. Every comparison includes
a non-learned pixel control. Signal designs are adapted from LLM
hallucination-detection methods, which face the same no-reference problem.
Upstream evaluation for comparison:
[rslearn segmentation tasks](https://github.com/allenai/rslearn/blob/master/rslearn/train/tasks/segmentation.py),
the [AWF task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml)
whose classes and split we reuse, and
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals).

## Reproduce

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

Experiments are exp01-exp21 in `exp/`; exp20 needs no checkpoint (it reads the served rasters over HTTP); exp21 downloads the fine-tuned checkpoint from HuggingFace. Checkpoints come from
[HuggingFace allenai](https://huggingface.co/allenai). exp04 needs the
[AWF dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf)
under `data/awf/dataset/`. `uv sync` installs CPU torch; for the GPU
experiments install the cu128 wheel into the venv afterwards and run
scripts with `.venv/Scripts/python.exe`, since `uv run` reverts it.

Status: research code under active development. Results are updated in
place as later experiments supersede earlier ones.

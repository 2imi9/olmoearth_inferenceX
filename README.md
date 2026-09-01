# olmoearth_inferenceX

Experiments on locating errors in
[OlmoEarth](https://allenai.org/olmoearth) predictions over regions with no
labels. Everything runs on Ai2's public artifacts: the
[olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain) encoders
and loader, checkpoints from
[HuggingFace](https://huggingface.co/allenai/OlmoEarth-v1-Base), and the
project datasets linked below.

## Result

Each signal gives every map window a suspicion score. The four signals:

- **E_case** - cross-model disagreement: two models (OlmoEarth v1
  [Nano](https://huggingface.co/allenai/OlmoEarth-v1-Nano) and
  [Base](https://huggingface.co/allenai/OlmoEarth-v1-Base)) predict the same
  window differently.
- **E_system** - tiling instability: the prediction flips when the input
  grid shifts by a few pixels. Ablation shows this is equivalent to
  proximity to a boundary in the model's own prediction map (exp14).
- **E_dist** - embedding distance: the window looks unlike the area the
  prediction head was trained on (one reference scene; not the encoder's
  pretraining data).
- **E_geo** - map check: the prediction contradicts a reference river map
  ([OSM waterways](https://wiki.openstreetmap.org/wiki/Key:waterway) for
  now). Under the current reference it mostly finds disagreements between
  the two maps, so it is not in the table (exp15).

AURC measures how well a suspicion score finds real errors; lower is
better. Every signal must beat two references: the model's own confidence
(max-softmax, the probability the model assigns its chosen class; this is
what ships as the top-1 probability bands of the
[olmoearth_lcc production rasters](https://huggingface.co/datasets/allenai/olmoearth_lcc)),
and a plain pixel statistic computed without any model (the control).

<details open>
<summary><b>Headline comparison: three conditions</b> (click to shrink)</summary>

| AURC (lower = finds errors better) | In-domain ([AWF expert labels](https://huggingface.co/datasets/allenai/olmoearth_projects_awf), 51 errors) | Ambiguous wetland margins (97 errors) | Far from training region (29 errors) |
|---|---|---|---|
| model's own confidence (baseline) | **0.0367** | 0.0666 | 0.0258 |
| cross-model disagreement (E_case) | 0.0533 | 0.0235 | 0.0103 |
| tiling instability (E_system) | 0.0427 | **0.0127** | 0.0009 |
| embedding distance (E_dist) | 0.1104 | 0.0289 | 0.0014 |
| pixel statistic (control) | 0.1287 | 0.0384 | **0.0005** |

</details>

<details>
<summary><b>All 29 scenes</b> (pre-registered set; exp13 corrected computation; bold = best per scene; click to expand)</summary>

| scene | errors | baseline | E_case | tile-phase | E_dist | control |
|---|---|---|---|---|---|---|
| barotse | 97 | 0.0666 | 0.0235 | **0.0127** | 0.0289 | 0.0384 |
| cuando_20 | 49 | 0.0035 | 0.0075 | **0.0025** | 0.0039 | 0.0092 |
| cuando_50 | 168 | 0.0536 | 0.0794 | 0.0277 | **0.0189** | 0.0316 |
| cuando_80 | 61 | 0.0072 | 0.0155 | **0.0050** | 0.0092 | 0.0093 |
| delta | 29 | 0.0258 | 0.0103 | 0.0009 | 0.0014 | **0.0005** |
| kafue | 11 | 0.0002 | **0.0001** | 0.0001 | 0.0001 | 0.0128 |
| kafue_20 | 77 | 0.0164 | 0.0244 | **0.0078** | 0.0407 | 0.0287 |
| kafue_50 | 48 | 0.0272 | 0.0275 | **0.0027** | 0.0089 | 0.0099 |
| kafue_80 | 38 | 0.0095 | 0.0068 | 0.0020 | 0.0138 | **0.0014** |
| kazungula | 18 | 0.0009 | 0.0009 | **0.0006** | 0.0037 | 0.0016 |
| luangwa | 8 | 0.0002 | 0.0017 | **0.0000** | 0.0002 | 0.0065 |
| luangwa_conf | 52 | 0.0096 | 0.0189 | **0.0072** | 0.0108 | 0.0597 |
| okavango_50 | 96 | **0.1181** | 0.1620 | 0.1271 | 0.1339 | 0.1229 |
| okavango_80 | 76 | 0.0555 | 0.0578 | 0.0061 | 0.0105 | **0.0037** |
| okavango_sep | 13 | 0.0005 | **0.0003** | 0.0003 | 0.0007 | 0.0163 |
| rovuma_20 | 39 | 0.0054 | 0.0095 | **0.0021** | 0.0104 | 0.0073 |
| rovuma_50 | 74 | 0.0099 | 0.0091 | **0.0080** | 0.0152 | 0.0160 |
| rovuma_80 | 18 | 0.0015 | 0.0030 | 0.0010 | 0.0014 | **0.0006** |
| save_20 | 25 | 0.0162 | 0.0619 | 0.0005 | **0.0005** | 0.0040 |
| save_50 | 76 | 0.0150 | 0.0409 | **0.0116** | 0.0163 | 0.0676 |
| save_80 | 23 | 0.0070 | 0.0292 | 0.0006 | 0.0006 | **0.0005** |
| shire_20 | 303 | 0.1509 | 0.1607 | 0.1767 | **0.1134** | 0.2558 |
| shire_50 | 123 | 0.0861 | 0.1000 | 0.0591 | 0.0629 | **0.0584** |
| shire_80 | 188 | 0.2957 | 0.2943 | 0.1902 | 0.2172 | **0.1487** |
| shire_liwonde | 17 | 0.0368 | 0.0357 | 0.0252 | 0.0013 | **0.0006** |
| vicfalls_up | 23 | 0.0008 | **0.0006** | 0.0008 | 0.0012 | 0.0412 |
| zambezi_20 | 51 | 0.0153 | 0.0118 | 0.0062 | 0.0168 | **0.0059** |
| zambezi_50 | 16 | 0.0003 | **0.0003** | 0.0003 | 0.0037 | 0.0005 |
| zambezi_80 | 47 | 0.0035 | 0.0148 | **0.0014** | 0.0085 | 0.0326 |

</details>

## Findings

- **In familiar territory, the model's own confidence is the best signal.**
- **Errors concentrate at the boundaries of the model's own prediction
  map, and boundary proximity predicts them better than confidence.**
  Tiling instability beats confidence on 27 of 29 scenes (sign test
  p < 0.001) and the pixel control on 19 of 29 (exp13), but a zero-cost
  boundary indicator computed from the prediction map alone matches it
  (13 of 29 head-to-head, p = 0.71) and itself beats confidence on 23 of 29
  (exp14). The perturbation adds nothing beyond boundary proximity. This
  holds for dense maps; on interior point labels (AWF) it does not apply.
- **Cross-model disagreement helps only on specific ambiguous scenes.** It
  beats confidence on 12 of 29 scenes; not a general signal.
- **Embedding distance has no scale-free advantage.** Its earlier
  "significant" result came from a few high-error scenes and did not
  survive scale-comparable statistics.
- **Where the reference map misses obvious water, a plain pixel statistic
  wins.** Those scenes prove confidence fails and nothing more.
- **Checking predictions against river lines mostly finds disagreements
  between reference maps, not model errors**, and adding that check to
  boundary proximity made it worse (exp15).
- **A disagreement partner must be a different model, not a more accurate
  one.** Same-family ensembles hurt.
- **The reference map is least reliable on exactly the most interesting
  terrains**, so all numbers are directions rather than settled effect
  sizes.

![No-model controls](exp/out/exp06_controls.png)

Docs: [index](docs/TECHNIQUES.md) ·
[results](docs/results/comparisons.md) ·
[signals](docs/results/signals.md) ·
[protocol](docs/method/protocol.md) ·
[facts](docs/method/infrastructure.md) ·
[roadmap](docs/plan/roadmap.md) · [lab log](exp/NOTES.md)

## Method

In machine-learning terms this is selective prediction: score how likely
each window is wrong, and judge the score by risk-coverage curves. All
signals are label-free; labels are used only for evaluation. Every split is
a spatial hold-out, with training and evaluation areas geographically
separate. Every comparison includes a non-learned control computed from raw
pixels. The signal designs are adapted from LLM hallucination-detection
methods, which face the same problem of judging outputs with no reference
answer.

Upstream evaluation we compare against: OlmoEarth's supervised metrics run
through
[rslearn segmentation tasks](https://github.com/allenai/rslearn/blob/master/rslearn/train/tasks/segmentation.py)
(the [AWF task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml)
defines the classes and split we reuse) and
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals).
Those measure accuracy where labels exist; this repository tests the
confidence signal itself where they do not.

## Reproduce

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

Experiments are exp01-exp15 in `exp/`. Model checkpoints come from
[HuggingFace allenai](https://huggingface.co/allenai) (OlmoEarth v1 Nano to
Large). exp04 needs the
[AWF dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf)
extracted under `data/awf/dataset/`. `uv sync` installs CPU torch; the GPU runs used the
cu128 wheel.

Status: research code under active development. Interfaces may change
between experiments; results are updated in place as later experiments
supersede earlier ones. A packaged library follows once the signal set
stabilizes.

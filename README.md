# olmoearth_inferenceX

Experiments on locating errors in predictions from linear probes on frozen
[OlmoEarth](https://allenai.org/olmoearth) v1 encoders, over regions with no
labels: a binary water head trained on one WorldCover-labelled 128x128
Sentinel-2 scene at Katima Mulilo, and a multiclass head on the AWF expert
labels. The production land-cover model has not been run. Everything runs
on Ai2's public artifacts: the
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
- **E_system** - tiling instability (called tile-phase in the scripts and
  CSVs): the prediction flips when the input grid shifts by a few pixels.
  Ablation shows it is statistically indistinguishable from proximity to a
  boundary in the model's own prediction map (exp14).
- **E_dist** - embedding distance: the window looks unlike the area the
  prediction head was trained on (one reference scene; not the encoder's
  pretraining data).
- **E_geo** - map check: the prediction contradicts a reference river map
  ([OSM waterways](https://wiki.openstreetmap.org/wiki/Key:waterway) for
  now). Under the current reference it mostly finds disagreements between
  the two maps, so it is not in the table (exp15).

AURC (area under the risk-coverage curve) measures how well a suspicion
score ranks the 40 m patches where the model's prediction disagrees with
the reference: patches are accepted in order of increasing suspicion, and
AURC is the error rate among accepted patches averaged over all acceptance
levels. Lower is better. On the river scenes the reference is ESA
WorldCover 2021 water, treated as truth; on the AWF task it is expert point
labels. Every signal must beat two references: the model's own confidence
(max-softmax, the probability the model assigns its chosen class; the same
kind of quantity that ships as the top-1 probability bands of the
[olmoearth_lcc production rasters](https://huggingface.co/datasets/allenai/olmoearth_lcc),
although those come from a different fine-tuned model we have not tested),
and a plain pixel statistic computed without any model (the control).

<details open>
<summary><b>Headline comparison: three scenes</b> (click to shrink)</summary>

| AURC (lower = finds errors better) | In-domain ([AWF expert labels](https://huggingface.co/datasets/allenai/olmoearth_projects_awf), 63 errors) | Ambiguous wetland margins (Barotse, vs WorldCover 2021, 97 errors) | Reference omits the river (Zambezi delta, vs WorldCover 2021, 29 disagreements) |
|---|---|---|---|
| model's own confidence (baseline) | **0.0363** | 0.0684 | 0.0234 |
| cross-model disagreement (E_case) | 0.0670 | 0.0235 | 0.0103 |
| tiling instability (E_system) | 0.0489 | **0.0127** | 0.0009 |
| embedding distance (E_dist) | 0.1338 | 0.0289 | 0.0014 |
| pixel statistic (control) | 0.1658 | 0.0384 | **0.0005** |

Third column: the WorldCover reference contains no water on this scene, so
its disagreements are reference omissions; a no-model pixel statistic ranks
them best (exp06), so the column shows that confidence fails there and
nothing more. AWF column: point labels, exp04 and exp12.

</details>

<details>
<summary><b>All 27 scenes</b> (pre-registered set; exp13 tie-aware AURC; bold = lowest unrounded AURC, ties at the displayed precision are within noise; click to expand)</summary>

| scene | errors | baseline | E_case | tile-phase | E_dist | control |
|---|---|---|---|---|---|---|
| barotse | 97 | 0.0684 | 0.0235 | **0.0127** | 0.0289 | 0.0384 |
| cuando_20 | 49 | 0.0035 | 0.0075 | **0.0025** | 0.0039 | 0.0092 |
| cuando_50 | 168 | 0.0536 | 0.0794 | 0.0277 | **0.0189** | 0.0316 |
| cuando_80 | 61 | 0.0072 | 0.0155 | **0.0050** | 0.0092 | 0.0093 |
| delta | 29 | 0.0234 | 0.0103 | 0.0009 | 0.0014 | **0.0005** |
| kafue_20 | 77 | 0.0164 | 0.0244 | **0.0078** | 0.0407 | 0.0287 |
| kafue_50 | 48 | 0.0241 | 0.0275 | **0.0027** | 0.0089 | 0.0099 |
| kafue_80 | 38 | 0.0128 | 0.0068 | 0.0020 | 0.0138 | **0.0014** |
| kazungula | 18 | 0.0009 | 0.0009 | **0.0006** | 0.0037 | 0.0016 |
| luangwa_conf | 52 | 0.0103 | 0.0189 | **0.0072** | 0.0108 | 0.0597 |
| okavango_50 | 96 | 0.1452 | 0.1612 | **0.1175** | 0.1339 | 0.1229 |
| okavango_80 | 76 | 0.0572 | 0.0578 | 0.0061 | 0.0105 | **0.0037** |
| okavango_sep | 13 | 0.0005 | **0.0003** | 0.0003 | 0.0007 | 0.0163 |
| rovuma_20 | 39 | 0.0053 | 0.0095 | **0.0021** | 0.0104 | 0.0073 |
| rovuma_50 | 74 | 0.0091 | 0.0091 | **0.0080** | 0.0152 | 0.0160 |
| rovuma_80 | 18 | 0.0016 | 0.0030 | 0.0010 | 0.0014 | **0.0006** |
| save_20 | 25 | 0.0208 | 0.0619 | 0.0005 | **0.0005** | 0.0040 |
| save_50 | 76 | 0.0140 | 0.0409 | **0.0116** | 0.0163 | 0.0676 |
| save_80 | 23 | 0.0069 | 0.0292 | 0.0006 | 0.0006 | **0.0005** |
| shire_20 | 303 | 0.1572 | 0.1605 | 0.1778 | **0.1134** | 0.2558 |
| shire_50 | 123 | 0.1008 | 0.1000 | 0.0636 | 0.0629 | **0.0584** |
| shire_80 | 188 | 0.3017 | 0.2947 | 0.1881 | 0.2172 | **0.1487** |
| shire_liwonde | 17 | 0.0335 | 0.0357 | 0.0252 | 0.0013 | **0.0006** |
| vicfalls_up | 23 | 0.0008 | **0.0006** | 0.0008 | 0.0012 | 0.0412 |
| zambezi_20 | 51 | 0.0130 | 0.0118 | 0.0062 | 0.0168 | **0.0059** |
| zambezi_50 | 16 | 0.0003 | **0.0003** | 0.0003 | 0.0037 | 0.0005 |
| zambezi_80 | 47 | 0.0035 | 0.0148 | **0.0014** | 0.0085 | 0.0326 |

</details>

## Findings

- **On the one expert-labelled in-domain task (AWF, 63 errors, point
  labels), the model's own confidence is the best signal** (exp04, exp12).
  On dense in-region river scenes it is not: best on 0 of 27 (exp13).
- **Errors concentrate at the boundaries of the model's own prediction
  map, and tiling instability ranks them better than confidence.** Tiling
  instability beats confidence on 26 of 27 rule-selected scenes
  (sign test p = 4e-07) and the pixel control on
  18 of 27 (exp13). A zero-cost boundary indicator computed
  from the prediction map alone is statistically indistinguishable from it
  (boundary better on 12, tile-phase on 15, sign test p = 0.70) and itself
  beats confidence on 19 of 27 (p = 0.052) and the pixel control on
  22 of 27 (exp14). No advantage of the perturbation beyond boundary proximity is
  detectable, but the indicator's own margin over confidence is marginal
  (p = 0.05), so the zero-inference shortcut is suggestive rather than
  established. Boundary-concentrated error is well known in
  segmentation and land-cover validation (trimap and Boundary-IoU
  evaluation; mixed-pixel effects, Foody 2002); what this repository adds is
  the quantified comparison against confidence, with controls, on OlmoEarth
  probes. On the AWF nine-class expert task confidence wins and the boundary
  indicator loses (AURC 0.0636 vs 0.0363; exp16). There the
  boundary score is largely a proxy for low confidence, because with nine
  classes the argmax flips between neighbours wherever margins are small.
  The earlier explanation that point labels carry no boundary context was
  tested and is withdrawn.
- **Cross-model disagreement helps only on specific ambiguous scenes.** It
  beats confidence on 10 of 27 scenes (sign test p = 0.25);
  not a general signal.
- **Embedding distance shows no scale-free advantage.** It beats confidence
  on 13 of 27 scenes (sign test p = 1.00); a mean-difference
  test gives p = 0.01 but is carried by a few high-error scenes.
- **Where the reference map misses obvious water, a plain pixel statistic
  wins** (exp06). Those scenes prove confidence fails and nothing more.
- **Checking predictions against river lines mostly finds disagreements
  between reference maps, not model errors**, and prepending that check to
  boundary proximity did not help (5 better, 9 worse, 13 unchanged of 27;
  sign test on untied pairs p = 0.42; exp15).
- **A disagreement partner needs uncorrelated errors, not higher accuracy.**
  Pairing Base with the stronger v1-Large (same OlmoEarth family) ranks
  errors worse than pairing it with Nano (mean AURC 0.0197 vs 0.0129, better
  on 3 of 7 scenes; exp10), and averaging three OlmoEarth models is worse
  than the best pair when one member is much weaker (exp03, exp04, exp07).
  No partner from outside the family has been tested.
- **The reference map is least reliable on exactly the most interesting
  terrains**, so all numbers are directions rather than settled effect
  sizes. Across the 27 scenes no single signal is best everywhere (best-signal
  tally: tiling instability 12, control 9, E_case 3,
  E_dist 3, confidence 0); which signal works still
  depends on the scene.

Risk-coverage curves for Kazungula, Barotse and the delta, no-model pixel
statistics against confidence and E_case (exp06):

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
extracted under `data/awf/dataset/`. `uv sync` installs CPU torch on Windows because the upstream package pins
the CPU index and uv refuses a second torch index for the same platform
while olmoearth-pretrain is a path dependency. For the GPU experiments,
install the cu128 wheel into the venv afterwards and run scripts with
`.venv/Scripts/python.exe` directly; `uv run` and `uv sync` will revert it.

Status: research code under active development. Interfaces may change
between experiments; results are updated in place as later experiments
supersede earlier ones. A packaged library follows once the signal set
stabilizes.

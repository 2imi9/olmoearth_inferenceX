# olmoearth_inferenceX

**Can you find where a model's map is wrong, in a region with no labels?**

This repository ports hallucination-detection techniques from LLMs to earth
observation and tests them against [OlmoEarth](https://allenai.org/olmoearth)
predictions. Everything uses Ai2's public artifacts.

Four things were audited:

| Target | What it is |
|---|---|
| Linear probes on frozen v1 encoders | A binary water head trained on one WorldCover-labelled scene; a nine-class head on the AWF expert labels |
| [OlmoEarth-v1-FT-AWF-Base](https://huggingface.co/allenai/OlmoEarth-v1-FT-AWF-Base) | Ai2's published fine-tuned model, run end to end (exp21) |
| [olmoearth_lcc rasters](https://huggingface.co/datasets/allenai/olmoearth_lcc) | The served land cover change product, as published (exp20, exp22) |
| v1 vs v1.2 encoders | Whether the RoPE change affects any of this (exp19) |

## The short answer

**The model's own confidence is the best error-finder we found.** Every
constructed audit signal was equal or worse against expert labels. That is a
negative result for the audit idea and a positive one for the model: its
confidence is usable, provided you also report the coverage it holds at.

## How the signals work

Each signal scores every map window for how likely it is to be wrong.

| Signal | What it measures |
|---|---|
| **E_system** — tiling instability | The prediction flips when the input grid shifts a few pixels. Behaves like proximity to a boundary in the model's own prediction map (exp14) |
| **E_case** — disagreement | Two models ([Nano](https://huggingface.co/allenai/OlmoEarth-v1-Nano) vs [Base](https://huggingface.co/allenai/OlmoEarth-v1-Base)), or one model's three Sentinel-2 band-set tokens, predict the window differently (exp17) |
| **E_dist** — embedding distance | The window looks unlike the scene the head was trained on |
| **E_geo** — map check | The prediction contradicts an [OSM river line](https://wiki.openstreetmap.org/wiki/Key:waterway). Mostly finds disagreements *between reference maps*, so it is not in the table (exp15) |

A signal is credible only if it beats **both** the model's own confidence
and a pixel statistic computed with no model at all (the control).

## The numbers

AURC (area under the risk-coverage curve) measures how well a score ranks
the windows the model gets wrong. **Lower is better; bold is best in
column.**

| Signal | AWF<br>in-domain | Barotse<br>wetland margins | Zambezi delta<br>ref. omits river | Sen1Floods11<br>hand labels |
|---|---|---|---|---|
| confidence (baseline) | **0.0363** | 0.0684 | 0.0234 | **0.0105** |
| E_system tiling instability | 0.0489 | **0.0127** | 0.0009 | 0.0115 |
| E_case cross-model | 0.0670 | 0.0235 | 0.0103 | 0.0186 |
| E_dist embedding distance | 0.1338 | 0.0289 | 0.0014 | 0.0592 |
| pixel control (no model) | 0.1658 | 0.0384 | **0.0005** | 0.0356 |

<sup>
Col 1: <a href="https://huggingface.co/datasets/allenai/olmoearth_projects_awf">AWF expert labels</a>, 63 errors.
Cols 2-3: ESA WorldCover 2021, a weak reference; 97 and 29 disagreements.
Col 4: hand-labelled flood masks, pooled excess AURC over 81,984 patches
on a geographically held-out region (exp18).
</sup>

**Read the columns carefully — they say different things.** Columns 2-3
score against ESA WorldCover, so their "errors" are disagreements with a
weak map; the delta scene's reference has no water at all, which is why the
no-model control wins there. Only columns 1 and 4 score against human
labels, and in both of those **confidence wins**.

Per-scene values for all 27 rule-selected scenes:
[docs/results/comparisons.md](docs/results/comparisons.md).

## Findings

**1. Against expert labels, confidence beats every audit signal.** On
Sen1Floods11 hand-labelled flood masks (a geographically held-out region,
351 scored tiles) every signal is equal or significantly worse (exp18); the
same holds on the AWF expert point task (exp04, exp16).

**2. This holds for Ai2's fine-tuned model run end to end.** The published
AWF checkpoint reproduces Ai2's accuracy (0.881 against 0.895); confidence
ranks its errors best, tiling instability ties it, everything else is
significantly worse (exp21).

**3. But that model is overconfident, so accuracy needs a coverage.** It is
0.93 accurate where it claims 0.99 (ECE 0.080). Abstain on the least
confident 20% and accuracy rises to 0.945.

**4. The earlier wins were against a weak reference, and we do not know
why.** Tiling instability beat confidence on 26 of 27 scenes, and band-set
disagreement on 21 of 27, when errors were defined against ESA WorldCover
(exp13, exp17). Neither transfers to hand labels. Three explanations were
tested and **all three failed**: reference-version instability covers only
~10% of disagreements and the advantage survives without it (exp23); the
imagery-to-map year gap does not explain it (exp24); neither does seasonal
water (exp25). This is the repository's main open question.

**5. Errors do concentrate at prediction boundaries** — about 75% of error
patches against 20% of correct ones, on both references — but confidence
still ranks them better than boundary proximity does.

**6. A disagreement partner needs a different view of the input, not a more
accurate model** (exp10, exp17). A stronger same-family partner makes the
signal worse.

**7. On the served product, boundary triage works and confidence cannot be
tested.** The land cover change rasters export no confidence for their class
map. Ranking by prediction-boundary fraction captures a median 0.88 of
WorldCover water disagreements at a 5% review budget (exp20).

**8. The served product shows no inference-window striping, but is quantized
to the 40 m patch grid.** Class boundaries and change-probability gradients
peak on the encoder's 4-px lattice in 19 of 20 profiles and nowhere at the
inference-window periods; seams affecting 5-10% of rows would have been
detected (exp22).

> **Scope caveat.** The water results are linear probes on frozen encoders.
> One fine-tuned model has been run end to end; the change product is
> assessed only through its served output.

Negative and null results, and every number behind the above, are in
[comparisons.md](docs/results/comparisons.md) and
[signals.md](docs/results/signals.md).

![No-model controls](exp/out/exp06_controls.png)

## Method

This is **selective prediction**: score how likely each window is to be
wrong, then judge the score by risk-coverage curves.

- Signals are label-free. Labels only grade the signals, never train them.
- Every split is a spatial hold-out.
- Every comparison includes a non-learned pixel control.
- Scene selection is pre-registered before any scene is fetched.

Full scoring rules, evidence tiers and status terms:
[docs/method/protocol.md](docs/method/protocol.md).

Upstream evaluation for comparison:
[rslearn segmentation tasks](https://github.com/allenai/rslearn/blob/master/rslearn/train/tasks/segmentation.py),
the [AWF task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml)
whose classes and split we reuse, and
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals).

## Reproduce

```bash
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

Experiments are `exp01`-`exp26` in [`exp/`](exp/). Every claim in the docs
cites the experiment that produced it, and every experiment writes a CSV or
JSON under `exp/out/` that the claim can be checked against.

- `exp20` needs no checkpoint — it reads the served rasters over HTTP.
- `exp21` downloads the fine-tuned checkpoint from HuggingFace.
- `exp04` needs the [AWF dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf) under `data/awf/dataset/`.
- Checkpoints come from [HuggingFace allenai](https://huggingface.co/allenai).
- `uv sync` installs CPU torch. For the GPU experiments, install the cu128
  wheel into the venv afterwards and invoke the venv's Python directly —
  `uv run` reverts it.

## Documentation

| | |
|---|---|
| [Technique ledger](docs/TECHNIQUES.md) | What was tried, one line each — **start here** |
| [Recipe](docs/method/recipe.md) | What to do and not do when auditing a map |
| [Protocol](docs/method/protocol.md) | How results are scored; evidence tiers |
| [Comparisons](docs/results/comparisons.md) | Per-experiment results |
| [Signals](docs/results/signals.md) | Per-signal evidence |
| [Task cards](docs/method/taskcards.md) | What each fine-tuned model is |
| [Infrastructure](docs/method/infrastructure.md) | Upstream sources, formats, encoder internals |
| [Agent integration](docs/method/agent_integration.md) | Contract with the OlmoEarth Agent |
| [Roadmap](docs/plan/roadmap.md) | Open items in priority order |
| [Lab log](exp/NOTES.md) | Chronological, including superseded runs |

---

*Research code under active development. Results are updated in place as
later experiments supersede earlier ones; the chronology is in the lab log.*

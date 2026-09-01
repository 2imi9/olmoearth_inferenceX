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

- **E_case** - cross-model disagreement: two models
  ([OlmoEarth v1 Nano](https://huggingface.co/allenai/OlmoEarth-v1-Nano) and
  [Base](https://huggingface.co/allenai/OlmoEarth-v1-Base)) predict the same
  window differently.
- **E_system** - tiling instability: the prediction flips when the input
  grid shifts by a few pixels.
- **E_dist** - embedding distance: the window looks unlike anything in the
  training data.
- **E_geo** - map check: the prediction contradicts a reference river map
  ([OSM waterways](https://wiki.openstreetmap.org/wiki/Key:waterway) for
  now). Tested only for false alarms so far, so it is not in the table.

AURC measures how well a suspicion score finds real errors; lower is
better. Every signal must beat two references: the model's own confidence
(max-softmax, the probability the model assigns its chosen class; this is
what ships as the top-1 probability bands of the
[olmoearth_lcc production rasters](https://huggingface.co/datasets/allenai/olmoearth_lcc)),
and a plain pixel statistic computed without any model (the control).

| AURC (lower = finds errors better) | In-domain ([AWF expert labels](https://huggingface.co/datasets/allenai/olmoearth_projects_awf), 51 errors) | Ambiguous wetland margins (97 errors) | Far from training region (29 errors) |
|---|---|---|---|
| model's own confidence (baseline) | **0.0367** | 0.0666 | 0.0258 |
| cross-model disagreement (E_case) | 0.0533 | **0.0235** | 0.0103 |
| tiling instability (E_system) | 0.0427 | 0.0555 | 0.0076 |
| embedding distance (E_dist) | 0.1104 | 0.0289 | 0.0014 |
| pixel statistic (control) | 0.1287 | 0.0384 | **0.0005** |

What we learned so far:

- In familiar territory, the model's own confidence is the best signal.
- On ambiguous terrain, cross-model disagreement is the best signal, and it
  beats the pixel control there, so it is not just edge detection.
- Far from the training region, confidence fails. No model signal provably
  beats pixel statistics there yet.
- Across 29 scenes chosen by a pre-registered rule, no signal dominates.
  Which signal works depends on the scene. Identifying that regime per
  scene is the open problem.
- A disagreement partner must be a different model, not a more accurate
  one. Same-family ensembles hurt.
- The reference map is least reliable on exactly the most interesting
  terrains, so all numbers are directions rather than settled effect sizes.

![No-model controls](exp/out/exp06_controls.png)

Every number, condition, statistic, and caveat:
[docs/TECHNIQUES.md](docs/TECHNIQUES.md). Chronology: exp/NOTES.md.

## Method

We port LLM hallucination-detection ideas to earth observation. Labels only
evaluate signals, never build them. Train and evaluation areas are always
geographically separate. A no-model control runs in every comparison.

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

Experiments are exp01-exp11 in `exp/`. Model checkpoints come from
[HuggingFace allenai](https://huggingface.co/allenai) (OlmoEarth v1 Nano to
Large). exp04 needs the
[AWF dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf)
extracted under `data/awf/dataset/`. `uv sync` installs CPU torch; the GPU runs used the
cu128 wheel. Experimental repository; a clean library release comes later.

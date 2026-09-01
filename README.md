# olmoearth_inferenceX

Experiments on locating errors in OlmoEarth predictions over regions with no
labels.

## Result

Each signal gives every map window a suspicion score. AURC measures how well
that score finds real errors; lower is better. Every signal must beat two
references: the model's own confidence (max-softmax), and a plain pixel
statistic computed without any model (the control).

| AURC (lower = finds errors better) | In-domain (expert labels, 51 errors) | Ambiguous wetland margins (97 errors) | Far from training region (29 errors) |
|---|---|---|---|
| model's own confidence (max-softmax, baseline) | **0.0367** | 0.0666 | 0.0258 |
| cross-model disagreement: two models differ on a window (E_case) | 0.0533 | **0.0235** | 0.0103 |
| tiling instability: prediction flips when the input grid shifts a few pixels (E_system) | 0.0427 | 0.0555 | 0.0076 |
| embedding distance: window looks unlike the training data (E_dist) | n/a | 0.0289 | 0.0014 |
| pixel statistic, no model (control) | n/a | 0.0384 | **0.0005** |

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

## Reproduce

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

Experiments are exp01-exp11 in `exp/`. exp04 needs the AWF dataset
(`huggingface.co/datasets/allenai/olmoearth_projects_awf`) under
`data/awf/dataset/`. `uv sync` installs CPU torch; the GPU runs used the
cu128 wheel. Experimental repository; a clean library release comes later.

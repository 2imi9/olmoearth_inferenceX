# olmoearth_inferenceX

Experiments on locating errors in OlmoEarth predictions over regions with no
labels. Ten experiments so far, all on public data and public checkpoints;
CPU sufficient throughout, a consumer GPU used from exp08 onward.

## Result

| AURC, lower = finds errors better | In-domain (AWF expert labels, 51 errors) | Ambiguous wetland margins (97 errors) | Far from training region (29 errors) |
|---|---|---|---|
| model's own confidence (max-softmax) | **0.0367** | 0.0666 | 0.0258 |
| cross-model disagreement (E_case) | 0.0533 | **0.0235** | 0.0103 |
| tiling-perturbation instability (E_system) | 0.0427 | 0.0555 | 0.0076 |
| embedding distance to training data (E_dist) | n/a | 0.0289 | 0.0014 |
| plain pixel statistics (control, no model) | n/a | 0.0384 | **0.0005** |

Interpretation:

1. **In familiar territory, the model's own confidence is the best signal
   tested.** None of the constructed signals improved on it.
2. **On ambiguous terrain, cross-model disagreement is the best signal
   tested.** Nano-Base disagreement ranks errors roughly three times better
   than confidence and better than any pixel statistic, so it reflects model
   behavior rather than image edges.
3. **Far from the training region, confidence is unreliable** - a plain NDWI
   gradient filter outranks it. No model signal beat pixel statistics there
   either: that scene's errors are spectrally obvious reference omissions,
   so the scene supports the negative claim about confidence and nothing
   more.

Two side findings: averaging disagreement over three models is worse than the
best pair when one model is weak (seen twice), and embedding distance does
not rank errors in familiar territory.

![No-model controls](exp/out/exp06_controls.png)

*Risk-coverage per scene, model signals and no-model controls on identical
errors (exp06); bottom row: control signal maps for the Barotse scene.*

Across seven river scenes with the control included (exp09), confidence
produced the best ranking on zero scenes; disagreement and tile-phase won
five between them, each win surviving the pixel control; the control won the
two scenes whose errors are trivial reference omissions.

**Read the numbers as directions, not effect sizes**: one scene or task per
condition, small error counts, no significance tests, and ESA WorldCover as a
weak reference on exactly the interesting terrains. The standing
per-technique record with every number, condition, and caveat is
[docs/TECHNIQUES.md](docs/TECHNIQUES.md).

## Method in one paragraph

The signals are ports of LLM hallucination-detection techniques, which mostly
do not depend on language: cross-model disagreement (self-consistency /
SelfCheckGPT), perturbation instability (semantic entropy), reference-map
consistency (retrieval-grounded fact checking; specificity shown, sensitivity
untested), and embedding distance (internal-state probing). Every signal is
scored by AURC against observed errors and must beat the audited model's own
max-softmax confidence; labels only evaluate signals, never construct them;
all splits are spatial; no-model pixel statistics run as controls on the same
errors.

## Next

A domain-shift testbed with non-trivial errors and expert labels (AWF
geographic-corner holdout), multiple scenes per condition with confidence
intervals, a sensitivity test for the reference-map check, and
reliability-weighted multi-model aggregation.

## Layout and reproduction

`docs/TECHNIQUES.md` results ledger · `exp/` experiments and figures ·
`exp/NOTES.md` lab log · `oe_inferencex/` library in progress ·
`reports/` writeups.

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

exp04 needs `huggingface.co/datasets/allenai/olmoearth_projects_awf`
extracted to `data/awf/dataset/`. Experimental repository; a separated
library release is planned once the signal set stabilizes.

# olmoearth_inferenceX

Experiments on locating errors in OlmoEarth predictions over regions with no
labels. 

## Result

**How to read the table.** Each signal assigns every map window a suspicion
score; windows are ranked by it and judged by AURC (area under the
risk-coverage curve): remove the most suspicious windows first and measure
how quickly errors disappear from what remains. Lower AURC = the signal
finds errors better. The baseline is max-softmax, the probability the model
itself assigns to its chosen class - a useful signal must beat it. The
control is a statistic computed from raw pixels with no model involved; a
model signal only means something where it also beats the control.

| AURC (lower = finds errors better) | In-domain (expert labels, 51 errors) | Ambiguous wetland margins (97 errors) | Far from training region (29 errors) |
|---|---|---|---|
| model's own confidence (max-softmax, baseline) | **0.0367** | 0.0666 | 0.0258 |
| cross-model disagreement: two models differ on a window (E_case) | 0.0533 | **0.0235** | 0.0103 |
| tiling instability: prediction flips when the input grid shifts a few pixels (E_system) | 0.0427 | 0.0555 | 0.0076 |
| embedding distance: window looks unlike the training data (E_dist) | n/a | 0.0289 | 0.0014 |
| pixel statistic, no model (control) | n/a | 0.0384 | **0.0005** |

Interpretation:

1. **In familiar territory, the model's own confidence is the best signal
   tested.** None of the constructed signals improved on it.
2. **On ambiguous terrain, cross-model disagreement is the best signal
   tested** - roughly three times better than confidence, and better than
   any pixel statistic, so it reflects model behavior rather than image
   edges.
3. **Far from the training region, confidence is unreliable** - even a plain
   pixel statistic outranks it. But no model signal beat the pixel
   statistics there either: that scene's "errors" are places where the
   reference map misses an obvious river, so the scene proves confidence
   fails and nothing more.

Replicated across seven river scenes (exp09): confidence was never the best
signal (0 of 7); disagreement and tiling instability won five between them,
each win beating the control; the control won the two scenes whose errors
are trivial reference omissions. Side findings: combining three models from
the same family is worse than the best pair - a disagreement partner needs
to be *different*, not accurate (exp07, exp10) - and embedding distance
never legitimately won a scene.

![No-model controls](exp/out/exp06_controls.png)

*Risk-coverage per scene: model signals and no-model controls scored on
identical errors (exp06); bottom row shows the control signal maps for the
Barotse floodplain scene.*

**Read the numbers as directions, not effect sizes**: few scenes per
condition, small error counts, no significance tests, and the reference map
(ESA WorldCover) is least reliable on exactly the interesting terrains. The
full per-technique record - every number, condition, and caveat - is
[docs/TECHNIQUES.md](docs/TECHNIQUES.md).

## Method in one paragraph

The signals port techniques the LLM community uses to detect hallucinations
- judging outputs when no reference answer exists - and they mostly do not
depend on language: cross-model disagreement (self-consistency /
SelfCheckGPT), perturbation instability (semantic entropy), checking
predictions against reference river maps (retrieval-grounded fact checking;
tested for false alarms only so far), and embedding distance
(internal-state probing). Labels are used only to evaluate signals, never to
build them; train and evaluation areas are always geographically separate;
pixel-statistic controls run on every comparison.

## Next

An error-rater model from outside the OlmoEarth family (correlated errors
within the family cap the disagreement signal), a domain-shift test with
non-trivial errors and expert labels, a sensitivity test for the
reference-map check, and significance testing over the seven-scene set.

## Layout and reproduction

`docs/TECHNIQUES.md` results ledger · `exp/` experiments (exp01-exp10) and
figures · `exp/NOTES.md` lab log · `oe_inferencex/` library in progress ·
`reports/` writeups.

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

exp04 needs `huggingface.co/datasets/allenai/olmoearth_projects_awf`
extracted to `data/awf/dataset/`. Note: `uv sync` installs CPU torch; the
GPU experiments used the cu128 wheel installed manually. Experimental
repository; a separated library release is planned once the signal set
stabilizes.

# olmoearth_inferenceX

Experiments toward assessing the reliability of OlmoEarth model inference over
regions without ground-truth labels.

## Motivation

Fine-tuned earth observation models are routinely applied to regions where no
labels exist, which is precisely where supervised evaluation metrics say
nothing. The language-model literature has developed several families of
techniques for judging model outputs without a reference answer
(hallucination detection, selective prediction). Most of these techniques do
not depend on language, and this repository tests whether they transfer to
geospatial inference.

| Signal | Related LLM-domain method | Geospatial instantiation |
|---|---|---|
| E_case: cross-model disagreement | self-consistency; SelfCheckGPT (Manakul et al., EMNLP 2023) | divergence between Nano/Tiny/Base predictions on the same input |
| E_system: perturbation stability | semantic entropy (Farquhar et al., Nature 2024) | prediction variance under sub-patch shifts of the tiling grid |
| E_geo: geographic grounding | retrieval-grounded fact checking | prediction consistency against reference river centerlines (GRIT, GRWL, OSM) |
| E_dist: embedding dissimilarity | internal-state probing (INSIDE; semantic entropy probes) | distance from the training distribution in embedding space |

The output posture is a ranked audit: per-window suspicion scores evaluated
as risk-coverage curves (AURC), not a single accuracy estimate per region.
Every signal is compared against the max-softmax confidence of the audited
model, which is the baseline any proposed signal must improve on. Labeled
data is used only to evaluate the signals, never to construct them. All
splits are spatial.

## Findings to date

Six experiments (exp01-exp06), all on public data and public checkpoints,
CPU only. Stated conservatively:

1. On the in-domain tasks evaluated — an easy dry-season river scene
   (exp02) and 9-class land cover against expert annotations from the AWF
   partner project under its own spatial split (exp04) — max-softmax
   confidence produced the best error ranking of the signals tested.
2. On a floodplain scene with ambiguous wetland margins (exp05), cross-model
   disagreement ranked errors better than confidence (AURC 0.0235 vs 0.0666)
   and retained that margin over no-model image-statistic controls (best
   control: 0.0384; exp06).
3. On a mangrove-coast scene ~1300 km from the training region (exp05),
   model confidence ranked errors worst, but no model signal outperformed
   trivial NDWI statistics (exp06): that scene's errors are spectrally
   trivial reference omissions, so it supports only the negative claim about
   confidence. A shift testbed with non-trivial errors is the first open
   item.
4. Embedding distance to training data did not rank in-domain errors (worst
   signal on the in-domain scene); its proposed role as an
   out-of-distribution indicator remains untested on a valid shift testbed.
5. Equal-weight disagreement across three models was worse than the best
   pairwise signal whenever one model was substantially weaker (observed
   twice, exp03 and exp04); multi-model aggregation appears to require
   reliability weighting.

![Hard scenes](exp/out/exp05_hard_scenes.png)

*Risk-coverage on the two difficult scenes (exp05). Top: Barotse floodplain
interior. Bottom: Zambezi delta mangrove coast. Lower curves indicate better
error ranking; the max-softmax baseline is the highest curve on both.*

**Limitations.** Each condition has been observed on a single scene or task,
with small error counts and no significance testing. ESA WorldCover, the
reference on the unlabeled-region scenes, is least reliable on exactly the
terrains where the channels performed best; on the shifted scene the exp06
controls show the measured "errors" are spectrally trivial, and that scene is
therefore excluded as evidence for model-signal superiority. Replication
against expert labels under shift is the first open item.

The per-technique record with all numbers, conditions, and caveats is in
[docs/TECHNIQUES.md](docs/TECHNIQUES.md).

## Repository layout

- `docs/TECHNIQUES.md` — technique ledger: standing results per technique,
  each claim citing the experiment that produced it; open items ranked.
- `exp/` — experiment scripts (exp01-exp06); figures and cached
  intermediates in `exp/out/`.
- `exp/NOTES.md` — chronological lab log.
- `oe_inferencex/` — library in progress. `evidence.py` contains the signal
  and evaluation mathematics (pure, no network access); `data.py` and
  `awf.py` contain data access.
- `reports/` — point-in-time writeups.

## Reproduction

CPU-only; public data and checkpoints throughout.

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

exp04 additionally requires the AWF dataset
(`huggingface.co/datasets/allenai/olmoearth_projects_awf`) extracted to
`data/awf/dataset/`.

## Status

Experimental repository; interfaces and structure will change. A separated
library release is planned once the signal set stabilizes.

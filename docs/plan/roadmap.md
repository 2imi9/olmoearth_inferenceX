# Open items

Rewritten in place as experiments close items. The chronology of what closed
what is in [../../exp/NOTES.md](../../exp/NOTES.md). Index at
[../TECHNIQUES.md](../TECHNIQUES.md).

## The one question that matters

Tiling instability beats confidence 26/27 against ESA WorldCover and loses
against hand labels. Three explanations for the gap have been tested and
rejected (reference-version instability, the year gap, seasonal water; see
[../results/comparisons.md](../results/comparisons.md) section 3). What
remains is either reference error shared by both WorldCover versions, or a
genuine property of the WorldCover-defined task.

**Everything below is ordered by how much it bears on that question.**

## Priority items

1. **Hand-adjudicate the disagreements.** exp26 prepared the kit: the top-12
   disagreements ranked by tiling instability and by confidence on shire_80,
   barotse and okavango_80. For each patch the kit carries a 48-px
   true-colour crop with the 4-px patch outlined, the NDWI crop, the
   WorldCover pixels and the head's water probability — the sheets are
   `exp/out/exp26_handcheck_<scene>_{tile,confidence}.png` — alongside a row
   in `exp/out/exp26_handcheck_<scene>.csv` holding the model probability,
   the reference label, NDWI statistics, JRC seasonality, both ranks, and
   empty `verdict` / `note` columns. Verdict vocabulary: model error /
   reference error / seasonal or date difference / ambiguous. **A reviewer's
   verdicts are the input to the next analysis.** This is the only route
   that separates "reference error shared by both WorldCover versions" from
   "a property of the task".

2. **A dense expert-labelled map with few classes.** The 27-scene support is
   WorldCover-referenced (exp13); exp16 showed that on nine classes the
   boundary score collapses into a proxy for low margin. A dense
   expert-labelled binary or few-class map is the missing testbed.
   Candidates in Ai2's own evaluation suite: MADOS (marine debris, 15
   classes), PASTIS-R, GeoBench m-cashew-plant and m-sa-crop-type.

3. **Operating-point analysis instead of AURC.** Test whether any signal
   helps confidence at a *fixed review budget* on expert labels. exp21 hints
   at this: tiling instability captures 0.71 of errors at a 20% budget
   against confidence's 0.63, while losing on AURC. A ranking metric and a
   reviewer's actual workflow are not the same question.

4. **An out-of-family rater** (Clay or AnySat, both wrapped in
   olmoearth_pretrain evals). Correlated errors invalidate within-family
   Dawid-Skene (exp07) and cap pairwise disagreement quality (exp10). An
   architecture-independent rater is required for both.

5. **A domain-shift testbed with non-trivial errors and expert labels**
   (candidate design: geographic-corner holdout within AWF). The delta scene
   is disqualified as evidence by the exp06 controls, so E_dist's
   out-of-distribution role is untested until this exists.

6. **Generality across fine-tuned checkpoints.** Of the five public
   fine-tuned models only AWF and Mangrove have public labelled datasets.
   Mangrove was examined and declined for the ranking questions: it
   classifies 2x2-pixel windows with no spatial context (boundary and tiling
   signals are undefined there) and its split is a hash split of grid cells,
   not a spatial hold-out. LFMC, ForestLossDriver and EcosystemTypeMapping
   have no public labels. **A second fine-tuned dense task with expert
   labels and a spatial split is still needed.**

7. **An expert reference for the served change product.** The
   `olmoearth_lcc` annotated points are the model's own training labels
   (dataset card), collected largely by output-based labelling, so scoring
   the product on them would be in-sample. Options: independent change
   references over the served tiles (published deforestation or flood maps
   dated inside the product's window), or a small hand-labelled set drawn
   without reference to the model's output.

8. **E_geo with width-filtered GRWL centerlines**, then re-measure flag
   precision. Under OSM lines and WorldCover truth the flags mostly mark
   reference-map disagreement (exp15).

9. **E_dist formalization**: AOA / Dissimilarity Index (Meyer & Pebesma
   2021) in place of raw k-NN distance to the head's training scene.
   Delineate the overlap with SHRUG-FM (CVPR 2026 EarthVision) before
   claiming novelty.

## Asks of upstream

- **A class-head confidence in the LCC export** — top-1 minus top-2 logit
  alongside bands 4-5. Without it the recipe's primary signal cannot run on
  the served product at all (exp20).
- Confirmation of whether Studio per-project exports match the
  `olmoearth_lcc` export format (partial probabilities).

## Closed

| Item | Closed by |
|---|---|
| Confidence intervals and significance tests | exp13 (block bootstrap, exact sign, permutation) |
| Boundary + E_geo conjunction | exp15 — no benefit |
| The AWF point-label hypothesis | exp16 — withdrawn |
| v1 vs v1.2 comparison | exp19 |
| First direct audit of a served production window | exp20 |
| Evaluating on the production model rather than probes | exp21 |
| Striping at tile scale in the served product | exp22 — no seams; outputs quantized to the 4-px patch lattice. *Open remainder:* the same test on a v1-encoder product would show whether RoPE removed a seam signal v1 had, but no v1 product is served |
| Does reference instability explain the WorldCover wins? | exp23 — no |
| Does the year gap explain them? | exp24 — no |
| Does seasonal water explain them? | exp25 — no |

## Second use case: change attribution

Error ranking is one consumer of these signals. The same decomposition
applies to comparing two inference outputs (two dates, two model versions),
where a raw diff mixes real surface change, model instability, low-consensus
predictions, and geographically implausible transitions.

The design: E_system instability and E_case consensus at each date gate
which diffs are trustworthy; geographic and temporal plausibility rules
constrain which transitions are physically possible; plain image
differencing is the no-model control.

Intended application is automated interpretation of what changed over time,
where a change narrative should only be generated from diffs that survive
the decomposition. Natural testbed: the `olmoearth_lcc` production change
product (change probability and month-encoded dates).

**Status: design only, untested.** It inherits item 7 above — without a
held-out change reference there is nothing to score it against.

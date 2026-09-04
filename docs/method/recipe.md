# Recipe: auditing a prediction map without labels

What twenty-six experiments support doing, and not doing, when judging where
a prediction map is wrong in a region with no labels. Each item cites the
experiment behind it; the evidence is in
[../results/comparisons.md](../results/comparisons.md). Index at
[../TECHNIQUES.md](../TECHNIQUES.md).

## Do

**1. Rank errors by the model's own confidence.** On every expert-labelled
testbed — AWF point labels (exp04, exp16), Sen1Floods11 dense masks with a
geographic hold-out (exp18), the fine-tuned AWF model itself (exp21) — the
logit margin ranked errors at least as well as every constructed signal.
Score it as the negative absolute logit (or top-1 minus top-2) rather than
`1 - max probability`, so saturated probabilities do not tie.

**2. Use prediction-boundary proximity as a triage cue, not a ranker.**
About 75% of error patches lie on boundaries of the model's own prediction
map against about 20% of correct patches, on both references (exp14, exp16,
exp18). It says *where* errors live; confidence still orders them better.

**3. Validate any proposed signal on expert labels before believing it.**
Signals that beat confidence on 26 of 27 scenes against ESA WorldCover were
equal or worse on hand-labelled masks (exp13 vs exp18). Three explanations
for the gap have been tested and rejected (exp23, exp24, exp25), so this is
not a solved discrepancy you can reason around — measure it.

**4. If a reference product is all you have, run a no-model control.** A
pixel statistic (for water, the NDWI gradient) scored on the same errors
exposes reference-omission scenes, where it outranks every model signal
(exp06, exp09).

**5. Evaluate the way exp11-exp13 do.** Pre-register scene selection before
fetching scenes; hold out geographically; score with tie-aware AURC or
excess AURC across scenes; test with exact sign tests on untied pairs and a
block bootstrap; report wins, losses and ties, not means dominated by
high-error scenes. Details in [protocol.md](protocol.md).

**6. Report operating points alongside AURC.** Anyone acting on the ranking
needs the error fraction captured at a fixed review budget, not an area
under a curve. `assess_prediction` and `assess_classmap` in
`oe_inferencex/assess.py` report review sets and capture at 1, 5 and 10%
budgets; exp20 reports them on a served product.

**7. State accuracy with a coverage and a calibration check.** The
fine-tuned AWF model is 0.881 accurate on held-out expert points but
overconfident (ECE 0.080; 0.93 accurate where it says 0.99). Abstaining on
the 20% least confident points raises accuracy to 0.945 (exp21). A bare
accuracy without its coverage and reliability curve overstates the model.

## Do not

**8. Do not perturb by masking input content.** Occlusion instability was
the worst signal on every scene (exp08). The rule that generalizes:
perturbations that *preserve* scene content while changing the tokenization
expose model pathology; perturbations that *remove* content measure context
reliance instead.

**9. Do not use distance to the training scene as an error ranker.** It
fails in-domain (exp03, exp13) and on expert labels (exp18); its apparent
win under geographic shift was a scale artifact on a scene whose errors were
spectrally trivial (exp06, exp13).

**10. Do not expect hidden-state signals to transfer from language models.**
Logit-lens settling, representation drift and attention entropy were all
worse than confidence (exp17).

**11. Do not build ensembles from one model family.** A stronger same-family
partner makes disagreement *worse* (exp10), equal-weight aggregation hurts
(exp03, exp04, exp07), and Dawid-Skene reliability estimation is invalid
because the family errs together (exp07). Disagreement from a different
*input view* (the v1 band-set tokens) is informative against WorldCover
(exp17) but still loses on hand labels (exp18).

**12. Do not fuse channels.** Prepending a reference-map check to boundary
proximity did not help (exp15), and no combination has been found that beats
confidence — so there is nothing to fuse yet.

**13. Do not read a reference-map check as model error without width
filtering.** OSM river centerlines mostly flag disagreements between
reference maps on narrow channels (exp15).

## For OlmoEarth deployments specifically

- **The served LCC rasters export no class confidence.** Bands 6-7 belong to
  the change-category heads and are not a class confidence (exp20). Until a
  class confidence is exported, boundary fraction of the class map is the
  only label-free cue available on the product; it captured a median 0.88 of
  water disagreements with WorldCover at a 5% review budget. Exporting the
  class head's top-1 minus top-2 logit alongside bands 4-5 would let the
  recipe's primary signal run on the product.
- **Read the product through its dataset card.** Threshold band 1 at 128
  before reading bands 2-9, and take the CRS from the file — the served
  rasters are EPSG:3857, not the UTM grid in the file name.
- **v1.2 differs from v1** in two ways that matter here: it uses rotary
  position encoding, which does *not* reduce sub-patch tiling instability
  (larger in v1.2, exp19); and it tokenizes Sentinel-2 as one band-set token
  per patch, so v1's band-set signal has no v1.2 counterpart.
- **One fine-tuned model has been run end to end** from the published
  checkpoint (exp21): confidence still ranks its errors best and tiling
  instability ties it. The other published checkpoints (Mangrove,
  ForestLossDriver, LFMC, EcosystemTypeMapping) have not been run; the water
  results are for linear probes on frozen encoders.

## What would change this recipe

A signal that beats confidence on expert-labelled dense maps, on more than
one task family, at a fixed review budget. None has been found. Open items
are in [../plan/roadmap.md](../plan/roadmap.md).

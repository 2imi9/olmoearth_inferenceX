# Open items

Rewritten in place as experiments close items. The chronology of what closed
what is in [../../exp/NOTES.md](../../exp/NOTES.md). Index at
[../TECHNIQUES.md](../TECHNIQUES.md).

## The one question that matters

Tiling instability beats confidence 26/27 against ESA WorldCover and loses
against hand labels. Three explanations for the gap have been tested and
rejected (reference-version instability, the year gap, seasonal water; see
[../results/comparisons.md](../results/comparisons.md) section 3).

**Leading hypothesis: the reference is not independent of the model.**
WorldCover 2021 v200 was a decode-only pretraining target of OlmoEarth v1 and
v1.2 (checkpoint `config.json` `only_decode_modalities`; the pretraining
dataset card; the v1 paper, arXiv:2511.13655), as was the OpenStreetMap
raster behind E_geo. A probe fitted to WorldCover over features of an encoder
trained to predict WorldCover partly reads out the model's own internal
prediction of that map, so its disagreements would concentrate where that
internal prediction is uncertain: boundaries, which tiling instability and
the boundary indicator detect and hand labels do not share. Only 3 of the 27
rule scenes lie inside a pretraining tile (median nearest tile 23 km), so the
coupling would run through the map's conventions, not memorised scenes.
The target space itself is degenerate: the target encoder is the untouched
random initialisation and its targets have effective rank 2 on real scenes
(exp32, `exp/out/exp32_summary.json`), which is why nothing read off the
decoder side carries per-patch information. Re-targeting the frozen encoder
with a whitened target makes that objective predictable (57-70% of the
variance from context) and still yields no quantity that ranks the errors:
the residual is input texture (exp33), and neither a discrete target, gap masking nor the
decision-direction projection changes that (exp34; issue #10 closed, issue
#11 carries the pretraining-target recommendation).

**What would test it.** Not a decoder-versus-map disagreement count: the
probe and the shipped decoder share the encoder, so their errors co-locate
under any reference, and a decoder that predicts one class everywhere turns
such a count into the class indicator. The primary statistic is reference
specificity on identical cells: for scene *s* and reference *r*, G(s, r) =
AURC(confidence) minus AURC(signal); D(s) = G(s, WorldCover) minus
G(s, human labels); average within river, one-sided exact sign test over the
eight river clusters (7/8 positive gives p = 0.035). Run it for tiling
instability first, since it is the signal whose transfer failure is the
question. Anything read off the shipped decoder must first pass the oracle
gate in `exp/exp27_oracle_gate.py` (`exp/out/exp27_oracle_gate.json`): in the
target space, pure-class prototypes have pairwise cosine of median 0.996,
class identity sits in token norm which the loss discards, layout moves a
token more than water fraction does, and a ridge readout of water fraction
from perfect tokens reaches R² 0.67 raw and 0.55 normalised. Prototype and
retrieval readouts are therefore unsound; a learned, locked, geographically
disjoint readout is the only route, with H(p) as the score and the
constant-class baseline and class/boundary strata reported alongside.

**Everything below is ordered by how much it bears on that question.**
The paper-backed items are tracked as GitHub issues, one per method, each
carrying the paper links and the concrete test:
[#2](https://github.com/2imi9/olmoearth_inferenceX/issues/2) (this test),
[#3](https://github.com/2imi9/olmoearth_inferenceX/issues/3) (RCG density, parked),
[#4](https://github.com/2imi9/olmoearth_inferenceX/issues/4) (AOA / SHRUG-FM),
[#5](https://github.com/2imi9/olmoearth_inferenceX/issues/5) (semantic entropy),
[#6](https://github.com/2imi9/olmoearth_inferenceX/issues/6) (out-of-family rater),
[#7](https://github.com/2imi9/olmoearth_inferenceX/issues/7) (few-class testbed),
[#8](https://github.com/2imi9/olmoearth_inferenceX/issues/8) (GRWL centerlines),
[#9](https://github.com/2imi9/olmoearth_inferenceX/issues/9) (operating points).

## Priority items

1. **Hand-adjudicate the disagreements** (issue #2). exp26 prepared the kit: the top-12
   disagreements ranked by tiling instability and by confidence on shire_80,
   barotse and okavango_80. For each patch the kit carries a 48-px
   true-colour crop with the 4-px patch outlined, the NDWI crop, the
   WorldCover pixels and the head's water probability — the sheets are
   `exp/out/exp26_handcheck_<scene>_{tile,confidence}.png` — alongside a row
   in `exp/out/exp26_handcheck_<scene>.csv` holding the model probability,
   the reference label, NDWI statistics, JRC seasonality, both ranks, and
   empty `verdict` / `note` columns. Verdict vocabulary: model error /
   reference error / seasonal or date difference / ambiguous. **A reviewer's
   verdicts are the input to the next analysis.** The paired-reference test
   above needs far more than this seed: on the order of a few hundred
   disputed cells across the eight rivers, blinded, with matched dates and
   water definition, mixed with a sample of undisputed cells. Expert hours,
   not GPU hours, are the binding cost.

2. **A dense expert-labelled map with few classes** (issue #7): closed by
   exp54 on Ai2's published embeddings (MADOS, PASTIS, cashew, SA crop
   type), see the Closed table; exp16's nine-class collapse of the boundary
   score reappears there as boundary-first losing pooled at 15-19 classes.

3. **Operating-point analysis instead of AURC** (issue #9): closed by exp35,
   see the Closed table.

4. **An out-of-family rater** (issue #6): closed by exp39 and exp41, see
   the Closed table. Every encoder errs on the same windows.

5. **A domain-shift testbed with non-trivial errors and expert labels**
   (candidate design: geographic-corner holdout within AWF). The delta scene
   is disqualified as evidence by the exp06 controls, so E_dist's
   out-of-distribution role is untested until this exists.
   Candidate multi-event testbed: GEOID-Flood (arXiv 2608.02315; 219 events,
   CEMS-derived manually corrected pixel labels, a permanent-water class,
   event-level splits); its Sentinel-2 is pre-event only, so for an S2 head
   it is a permanent-water testbed and the flood class needs the S1 path.

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

8. **E_geo with width-filtered GRWL centerlines** (issue #8), then re-measure flag
   precision. Under OSM lines and WorldCover truth the flags mostly mark
   reference-map disagreement (exp15).

9. **A deployable two-signal fusion.** exp50 shows the label-fitted fusion's
   weight sits on NDWI level, with the ensemble entropies and confidence
   behind it. The minimal deployable form is confidence + NDWI level with
   weights fitted once on the head's training split; preregister it against
   confidence on the v1.2 test split at window and tile level (exp50 tile
   AURC 0.134 to 0.086 with all ten signals), and against the one-event
   Bolivia arm where the whole gain is the NDWI weight.

10. **The frozen joint head, the last cell of issue #14.** exp52 answered the
    trained form: Sentinel-1 adds at most 0.1 points to a fine-tuned Sentinel-2
    model. A linear head on the frozen [z_S2; z_S1] tokens is the untested
    frozen form, cheap on cached features; the prior after exp52 is low.


## Explanation layer: why a window is suspect

Error ranking says *which* windows to review; the deployment also needs
*why*. The cues that failed as rankers are the material for that
explanation, because each is a measured, error-enriched fact about a window
rather than a score to order by:

| Cue (label-free) | Share among error / correct windows on Bolivia hand labels, identical windows (exp37) | Evidence |
|---|---|---|
| on a prediction boundary | 0.750 / 0.214, 3.5x | exp14, exp16, exp18, exp37 |
| among the least confident 20% (the ranker itself) | 0.589 / 0.163, 3.6x | exp37 |
| unstable under a sub-patch shift (tiling instability) | 0.583 / 0.164, 3.6x; wins against WorldCover 26/27 as a ranker, ties confidence on hand labels | exp13, exp18, exp37 |
| spectrally ambiguous (\|NDWI\| < 0.1) | 0.483 / 0.067, 7.2x; error rate 0.41 among its windows | exp06, exp09, exp37 |
| disagrees under flips and rotations | 0.579 / 0.164, 3.5x | exp36, exp37 |
| seasonal water (JRC seasonality) | 39% of WorldCover disagreements vs 8% of agreements | exp25 |
| reference unstable between WorldCover versions | 14x enriched among disagreements, about 10% of them | exp23 |
| OSM river centerline disagrees with the map | 1.5x enriched; often reference-vs-reference on narrow channels | exp15 |

**Design.** A per-window attribution on top of `oe_inferencex.assess`: for
each window in a review set, the list of cues that fire, each carrying its
measured enrichment (share among errors against share among correct
windows, cited to the experiment above) and a templated sentence. The
explanation is validated the way the cues were: by enrichment on expert
labels, not by AURC, so a cue that does not order errors can still explain
them. Scene-level output: how many flagged windows carry each cue, which
cues co-occur, and which review-set windows carry none (those are the ones
the explanation cannot help with). The narration stays with the caller,
as the agent contract requires; this layer returns structured evidence.

**Status.** Built (exp37, `oe_inferencex.explain`): the cue library with
the shares above, `derive_cues` (boundary, low confidence) from an
assessment, caller-derived cues, `explain_review_set` (per review window
the cues that fire, co-occurrence, the windows no cue explains),
`cue_enrichment` as the validation. exp37 measured the five label-free
cues on identical windows: 95% of the hand-label errors carry at least one,
80% two or more; inside confidence's review set only the boundary and NDWI
cues separate error rates, the tiling and dihedral cues fire on nearly
every flagged window ([../results/explanation.md](../results/explanation.md)).
Next: the reference-side cues (seasonal water, version instability, OSM)
measured on the same windows where a WorldCover reference exists; the agent
tool returns the structured evidence.

## Published by Ai2 and usable now (checked on the Hugging Face hub, 2026-09-08)

- **`allenai/olmoearth-paper-embeddings`** (June 2026, CC BY 4.0): pre-extracted
  embeddings from 26 foundation models on the 24 tasks of the paper's Table 2,
  row-aligned across models in canonical sample order, with labels. The
  segmentation tasks (`sen1floods11` on Sentinel-1, `m_cashew_plant`,
  `m_sa_crop_type`, `mados`, `pastis`) come as `(N, 16, 16, D)` patch grids with
  full-resolution labels. This is issue #6 without a forward pass: a second head
  on Clay, DINOv3-sat, TerraMind or AnySat embeddings of the *same* windows,
  so disagreement is between two views. It is also part of issue #7: dense
  labelled few-class testbeds on identical windows for every model (cashew,
  7 classes; MADOS, 15; Sen1Floods11, 2). Sizes: ~1 GB per split for
  Sen1Floods11 per model, 9 GB for the cashew training split.
- **`allenai/olmoearth_lcc`** (updated 2026-08-28): the served change product's
  summary rasters now carry probabilities: band 1 `binary_change` (0-255), bands
  6-7 the pre/post change-category scores. The land-cover classes (bands 4-5)
  still have no score, so the ask below stands for them, but the production
  case of `assess_classmap` (hard map plus an exported confidence band) can now
  run on the change decision itself, which exp20 could not. The repository also
  publishes `training_data/eval_points.json`: 284 windows in 25 tiles across 17
  countries and 16 change types (renewable energy, reservoir filling, urban
  expansion, wildfire, mining, ...), 120 positive and 314 negative points, each
  with an expected change and literature evidence URLs, sourced from a curated
  evaluation set rather than output-based labelling. That is the independent
  change reference roadmap item 7 asked for, at point scale.
- **`allenai/olmoearth_pretrain_dataset`** (CC BY 4.0): the pretraining data
  itself, documented in `olmoearth_pretrain/docs/Pretraining-Dataset.md`. The
  true pretraining sample that issue #4 needed for typicality is public.
- Third-party: `Major-TOM/Core-S2L2A-249k-OlmoEarth-Base`, mean-pooled
  OlmoEarth-Base embeddings of 249k global Sentinel-2 chips (384 px) with
  footprints, a global chip-level bank for the coverage question exp40 left.

## Asks of upstream

- **The fine-tuned LCC model on the Hub, or failing that a confidence band
  for bands 4-5 of the LCC export.** Ai2 publishes five fine-tuned models
  (FT-AWF, Mangrove, LFMC, ForestLossDriver, EcosystemTypeMapping) but not
  the LCC one; with it we would take the land-cover head's top-1 minus
  top-2 logit ourselves. The 2026-08-28 export added a change probability
  (band 1) and category scores (bands 6-7), so the change decision can be
  audited with the primary signal; bands 4-5 are argmax classes only
  (checked 2026-09-09). Asked on 2026-09-09.
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
| Does any signal help at a fixed review budget where it does not on AURC? | exp35 — not with preregistered support; boundary at a 5% budget on hand labels is the one small, secondary exception |
| Does reviewing boundary windows first, then by confidence, beat confidence at fixed budgets? | exp36 — yes at the 5% and 10% budgets on hand labels (preregistered, per tile p = 3e-7 and 2e-7; pooled 0.274 vs 0.259 and 0.494 vs 0.465), not at 20%, and not on the fine-tuned model, where both orders pick the same 5% set; a triage rule, not a ranker |
| Does dihedral (flip-and-rotate) consistency rank the errors? | exp36 — no; 4/4 rivers, 154/196 Bolivia tiles, Spearman 0.67 to 0.94 with confidence |
| Explanation layer, first build | exp37 — five label-free cues measured on identical windows, library and `explain_review_set` in the package; 95% of hand-label errors carry a cue, NDWI ambiguity 7.2x |
| Does a head on an outside representation of the same windows err where OlmoEarth does not (issue 6, the two-view design)? | exp41 — no; on Sen1Floods11 the three outside partners sharing the 4-px grid are wrong on 80-82% of OlmoEarth's error windows, as its own family is (phi 0.77-0.81), and disagreement captures 0.242 vs 0.454 at 10%. Scope: on AWF the outside models are much less correlated than the family, and fine-tuning the encoder corrects 55.6% of a frozen probe's errors (exp21), so this is about frozen encoders read by a linear probe, not about the windows |
| Is the fixed 4-px grid window the right decision unit? | exp42 — no: averaging the four tilings that cover each pixel raises pixel accuracy by 1.0 and 0.9 points on the two hand-label testbeds (preregistered, p < 1e-40); spectral-split and scale-adaptive windows are mixed; 10% of windows have mixed labels and carry half the errors |
| Does a content-derived partition with the hard majority of the averaged map's pixels beat the averaged map (the conditional theorem of WindowDesignProofs.lean)? | exp43 — no: it breaks more pixels than it corrects on both testbeds (C 15,093 vs B 16,273; C 24,525 vs B 30,951); 11-15% of segments are mixed and 5-12% of pure ones carry a wrong majority; unsupported under the protocol, which does not show that no partition can help |
| Do the twelve missing crop phases improve the averaged map? | exp44 — measurably yes, worthwhile no: sixteen offsets beat the four diagonals on both testbeds (p = 0.003 and 1.3e-12) by +0.04 accuracy points against a preregistered +0.2 threshold, for twelve extra passes; an unbalanced subset of phases is worse than the diagonals, so a balanced offset set matters more than the number of them |
| Do the supported findings hold on the backbone the served product actually uses (v1.2 Base)? | exp45 — partly: the cue enrichments and the shift-averaged window replicate on both hand-label testbeds, the boundary-first order on Bolivia only, and confidence is no longer the best ranker on Bolivia, where tiling instability and a no-model spectral control beat it at unchanged accuracy; v1.2 keeps v1's accuracy and ranks its own errors worse |
| Is the shared cross-model error set caused by the readout, the input modality, the label grid or chip clustering? | exp46 — chiefly the modality: radar against optics moves the error set to phi 0.38-0.39 where a backbone swap moves it to 0.70; head capacity barely beats a backbone swap and falls short of fine-tuning; impurity and chip clustering are ruled out; and on Bolivia a no-encoder head on fifteen pixel statistics beats the frozen encoder |
| Does the model's own confidence rank the errors of the decision we actually recommend? | exp47 — it beats tiling instability everywhere and the grid window's confidence on Bolivia, but a no-model spectral index matches it on Bolivia under v1 and beats it under v1.2, so exp45's exception is not an artefact of the window design; U+ combinations all hurt |
| Do SHRUG-FM's reliability signals (ensemble MI, embedding OOD, input extremity) beat the model's own confidence under our protocol? | exp49 — no, preregistered and supported: confidence beats mutual information and -NCDD on both testbeds under v1.2 (per tile 263/61 and 352/76; leads +0.006/+0.002 and +0.08/+0.03); the embedding and input signals are 3-9x worse. Secondary: an eight-head bag's predictive entropy beats confidence under v1 on Bolivia (210/103); their fusion step as a label-fitted linear combiner beats confidence on three of four arms by 0.0027-0.0041, the first fusion to do so here; at their tile granularity confidence is at or near the best signal and mutual information the weakest |
| Which signal carries the fusion that beat confidence, does the bag replicate, does the fusion reach the tile level, does shift-label entropy rank? | exp50 — NDWI level carries it (+0.87 standardized; dropping it costs the most on three arms and on Bolivia under v1.2 all of the gain); the bag does not replicate (sixteen members, fresh seed: loses on v1 Bolivia 140/166, wins on the v1.2 test split 248/160, seed noise; P1 fails); a fusion fitted on tile failures beats confidence at the tile level on the multi-region split under both backbones (0.120 to 0.078, 0.134 to 0.086, intervals excluding zero) but not on Bolivia (P2 fails); shift-label entropy loses to tile-phase (issue 5 closed) |
| Does Ai2's own Sen1Floods11 probe (Sentinel-1, their recipe) see what our S2 head sees? | exp51 — their mIoU reproduces (0.789 vs 79.2) and our encode matches their embeddings window for window (phi 0.937); P1 supported (confidence beats the sensor-level control); P2 not supported: v1.2 ranks its Bolivia errors worse than v1 pooled (0.0241 vs 0.0214) but not per tile (173/152, p = 0.13); the no-model NDWI index beats the S1 probe's confidence on the multi-region split under both backbones and loses on Bolivia, the mirror image of the S2 head's exception; seven published encoders share OlmoEarth's error windows at phi 0.78-0.82 (Satlas 0.62) |
| Does fine-tuning the encoder ourselves correct the frozen head's errors, and does the Bolivia exception survive it? | exp52 — yes and no: the trained S2 model corrects 66% / 44% of the frozen head's errors (preregistered P2) and its confidence beats the NDWI index on Bolivia under both backbones (P1), ties it pooled on the multi-region split under v1 and beats it under v1.2; Sentinel-1 adds at most 0.1 points to the trained S2 model, so exp46's sensor lever is a frozen-feature property |
| Does adapting the head per tile so the tilings agree (test-time adaptation, held-out tilings as the stop rule) improve the map? | exp53 — no: pixel accuracy falls on both testbeds under both objectives while held-out agreement improves on 96-99% of tiles; agreement is not correctness and held-out views are not a guard (preregistered P1 fails) |
| Does labelling the tiles the audit flags teach a fine-tuned model more than labelling random tiles (label efficiency)? | exp56 — no: the audit's selection is worse than random on the multi-region split at every budget (at 300 labels -0.0053 on the seed mean, every seed behind; 1,000 labels 0.9588 vs 0.9622) and never reaches random's 1,000-label accuracy; entropy selection is worse still; the chosen tiles are three to four times harder than the pool (preregistered P1 and P2 fail) |
| Does the protocol hold on dense multi-class tasks, and do encoders share errors there too? | exp54 — confidence beats the embedding-distance control on MADOS, PASTIS (S2, S1+S2), cashew and SA crop type (P1 on all five); shared errors are task-dependent (phi 0.26-0.44 on MADOS, 0.05-0.08 on PASTIS vs 0.77-0.82 on floods); boundary-first at many classes loses pooled and wins per tile |
| Does the comparison tool tell a user more than a raw disagreement map? | exp58 — modestly: believing the more confident side is right on 51-70% of the disagreement windows, better than a coin flip on every pair by the per-tile sign test but below the preregistered 55% on four pairs and far below always choosing the side known to be better (P1 fails); ordering the disagreement set by confidence adds nothing on twelve of fifteen pairs (P2 fails); disagreements that persist across head draws are the harder ones, not the settled ones; on GEOID-Flood reading both-confident disagreements as change doubles the flooded share pooled (52% vs 27%) but wins on 16 events against 9 only (P3 fails on the event test) |
| How much do two inferences of the same scene differ, where, and which side is right? | exp57 — the difference atlas on every pair: disagreement 2-4% of windows across crop offsets, backbones and encoders, 8-11% across sensors; boundary-enriched everywhere (3.3-7.1x on Sen1Floods11, 21x on the median GEOID event; preregistered P1); the sensor and backbone disagreement sets overlap at phi 0.27 / 0.18 (P2); the label decides which side is right only for the fine-tuned model (71-75%) and the S2 head against the S1 head (67-84%), coin flips elsewhere; the module reproduces exp51, exp52 and exp55's cross-tabs |
| Over many flood events, how often does a no-model index beat confidence, and what is the review set worth? | exp55 — GEOID-Flood, 55 events: the index beats the S2 head's confidence on 2/45 events, the S1-level control beats the S1 head's on 1/45 (P1, P2); the median event's 5% review set holds 88% / 64% of the errors, 17.5x / 12.8x random (P3); the sensor flip and the boundary-first majority did not generalise |
| Does putting the most enriched cue (spectral ambiguity) first beat the boundary-first order at fixed budgets? | exp38 — mixed: per-tile budgets yes at 5% and 10% (p = 8e-4, 5e-5), one pooled budget no (0.292 vs 0.274, 0.481 vs 0.494), clear only at 20%; loses on the WorldCover rivers; boundary-first stays the supported rule |
| Does the model's prediction contradict its predictions on the windows that look most like it (neighbourhood contradiction), and does an outside representation help? | exp39, exp40 — on Bolivia the OlmoEarth-space score passes P1 at 5-10% budgets and the pixel-statistics ablation beats confidence at every budget and on E-AURC; AnySat adds nothing over OlmoEarth's own space (issue 6 answered in the negative for this design); the preregistered replication on the multi-region test split with Bolivia as the bank fails every primary test, so neither score is supported; the open variable is the bank's coverage of the queries |
| Does the pretraining objective itself (masked-token decoder error) rank the errors? | exp28 — no, on both testbeds; the frozen targets are near-collinear, so the residual tracks input texture |
| Does a last-layer posterior over the probe head (Laplace, bootstrap ensemble) rank the errors? | exp30 — no, on both testbeds; the variance is feature norm on the one-scene head and rises with the logit on the 128k-patch head |
| E_dist formalization: does feature-space typicality against training, same-scene or cross-testbed references rank the errors? | exp31 — no; the confidence + same-scene kNN combination reaches 6/2 rivers (p = 0.145) against WorldCover and hurts on hand labels; only a true pretraining sample remains untested (issue #4; the RCG density upgrade is issue #3, parked) |

## Cross-inference comparison: what is done, what is not

The comparison half of the package is `oe_inferencex.compare` (exp57): the
disagreement rate pooled and per group, the cue enrichment on the
disagreement windows, the stability of a disagreement set across head draws
and across pairs, and, with labels, exp52's cross-tab and which side is
right. exp57 ran it on every pair the repository holds (crop offsets, v1 vs
v1.2, the S2 head vs the S1 head, frozen vs fine-tuned, seven encoders on
Ai2's embeddings, the two sensors per GEOID-Flood event): the disagreement
sits on prediction boundaries everywhere, the sensor and backbone
differences are different sets of windows, and the label says which side is
right only for the differences that changed the model. Earlier pairs
(Nano/Tiny/Base, exp07; Large vs Nano as partner, exp10; band-set probes,
exp17; v1 vs v1.2, exp19; fine-tuned vs frozen, exp21; 2021 vs 2024 imagery,
exp24) predate the module. The index is in
[../TECHNIQUES.md](../TECHNIQUES.md#cross-inference-comparisons); the
adjudication case (exp23/24/25) is the section below it. Not done: the
stability of the encoder and fine-tuned disagreement sets across seeds (only
the linear heads were redrawn), and the same measurement on a multi-class
task.

**What is not done: change attribution between two dated products.** A raw
diff of two inference outputs mixes real surface change, model instability,
low-consensus predictions, and geographically implausible transitions. The
design: E_system instability and E_case consensus at each date gate which
diffs are trustworthy; geographic and temporal plausibility rules constrain
which transitions are physically possible; plain image differencing is the
no-model control. A change narrative should only be generated from diffs
that survive the decomposition.

Natural testbed: the `olmoearth_lcc` production change product (change
probability and month-encoded dates). **Status: design only, untested.** It
inherits item 7 above — without a held-out change reference there is nothing
to score it against.

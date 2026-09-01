# Technique ledger

Standing record of experimental results, organized by technique rather than
by date. A claim appears here only when an experiment supports it, and is
amended or removed when a later experiment contradicts it. Chronology is
kept in exp/NOTES.md. Every entry cites the experiment that produced it.

Status terms: **supported** (at least one experiment consistent with the
technique's claim, under the stated conditions) / **mixed** (results differ
across conditions) / **partial** (some evidence; a key condition untested) /
**rejected** (tested, contradicted) / **untested** / **blocked** /
**out of scope (v1)**.

Most comparisons come from single scenes or tasks with small error counts
and no significance testing; exp09 adds a seven-scene replication for the
water task (WorldCover-referenced, still without formal significance tests).
Numbers establish direction, not effect sizes.

## Summary

| Technique | Related LLM-domain method | Status | State of evidence |
|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal on the in-domain AWF expert-label task; best on zero of seven river scenes in the exp09 replication. All channel claims are relative to this baseline |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | mixed | best signal on 3 of 7 replication scenes, each win surviving the pixel control; loses to the baseline in-domain on AWF (0.0533 vs 0.0367). Partner quality is about decorrelation, not accuracy (exp10); all naive >2-model aggregations hurt (exp03/04/07). Out-of-family rater is the open fix |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | mixed | best signal on 2 of 7 replication scenes; loses to baseline on AWF multiclass (0.0427 vs 0.0367). Masking-perturbation variant rejected (exp08): perturb the tokenization, not the content |
| Backbone-version comparison (E_system, v1 vs v1_2) | n/a (EO-specific) | blocked | v1_2 checkpoints fail to load with the current olmoearth_pretrain checkout (state_dict mismatch); requires the newer loader |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, semantic entropy probes) | partial | zero legitimate wins across all scenes once controls are included; no support as an error ranker. Out-of-distribution-indicator role remains the open hypothesis, untested on a shift testbed with non-trivial errors |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | partial | specificity observed (zero false break alarms on one scene); sensitivity untested, no scene with a confirmed consensus break evaluated |
| Label-free reliability (Dawid-Skene) | annotator modeling | rejected (within family) | inflates every model and inverts the ordering because family members err together; the inflation gap measures correlated-error mass. Viable only with an out-of-family rater (exp07) |
| Risk-coverage / AURC harness | selective prediction | supported | seven-scene replication exists; lacks confidence intervals and significance tests |
| Validation on labeled data (labels grade signals, never train them) | pseudo-label validation (PPE §2.3.1) | supported | executed with AWF expert labels under the project's own spatial split; WorldCover remains the reference elsewhere |
| No-model pixel-statistic controls | ablation practice | supported | ran on all comparison scenes; killed one claim (E_dist under shift) and confirmed two (E_case wins are not edge detection) |
| Semantic-entropy port (cluster-then-entropy) | Farquhar et al. 2024 | untested | possible refinement of the perturbation signal |
| Verifier head trained on labeled regions | learned verifiers / reward models | out of scope (v1) | requires labels as training input |
| Channel fusion | n/a | out of scope (v1) | per-channel reporting only; rank aggregation if a single ordering is required |

## Related work and positioning

The individual signal families are not new, and the ledger should not be read
as claiming they are. Confidence-based map assessment appears in the CEOS
WGCV land cover validation protocols as a complement to reference-data
assessment. Test-time-augmentation uncertainty has been applied to EO
segmentation (e.g. landslide mapping), following Wang et al. 2019 in medical
imaging. The Area of Applicability / Dissimilarity Index (Meyer & Pebesma
2021) is adopted in spatial statistics via the CAST and waywiser packages,
for tabular predictor spaces. SHRUG-FM (CVPR 2026 EarthVision) performs
embedding-space OOD detection for EO foundation models. Ensemble
disagreement is standard uncertainty practice in mainstream ML.

What we did not find in the EO literature, and what this repository targets:
selective-prediction evaluation (risk-coverage / AURC) of land cover
inference; cross-model disagreement used as an audit signal; perturbation of
the ViT patchification grid specifically; and the combination of such
signals into an audit that is scored against the audited model's own
confidence with no-model controls, over regions without labels. The
contribution claim is the audit protocol and the ViT-specific
instantiations, not the signal families.

## Cross-signal comparisons

The main evidence: all signals scored on identical errors with the same
harness, per condition.

### Multi-scene replication (exp09)
- The full signal comparison plus the NDWI-gradient control over seven river
  scenes across southern Africa (Kazungula, Barotse, Zambezi delta, Luangwa
  confluence, Okavango panhandle, Shire at Liwonde, upstream Victoria
  Falls), heads trained at Katima Mulilo, WorldCover reference, 8+ errors
  per scene. Per-scene results in exp/out/exp09_multiscene.csv.
- Max-softmax confidence produced the best ranking on zero of seven scenes
  (mean AURC 0.0201 +/- 0.0230).
- E_case won three scenes (mean 0.0129 +/- 0.0127) and beat the pixel
  control on every scene it won, including a zero-reference-water scene
  (Okavango: 0.0003 vs control 0.0163) whose errors are therefore not
  spectrally trivial. Tile-phase won two (mean 0.0138), both with the
  control far behind.
- The control won exactly the two scenes identified as reference-omission
  regimes (delta, Shire), consistent with exp06.
- E_dist won zero scenes once the control was included; its earlier apparent
  shift-scene advantages were both control-dominated scenes.
- Standing caveats: WorldCover reference, small per-scene error counts, no
  formal significance testing yet.

  ![Per-scene AURC](../exp/out/exp09_multiscene.png)

### No-model image-statistic controls (exp06)
- Control signals computed directly from pixel values (within-patch spectral
  variance, patch-mean |NDWI| proximity to the water/land boundary, NDWI
  gradient magnitude), scored on identical errors with the same harness.
- Kazungula: every model signal retains a margin over the best control
  (tile-phase 0.00058, |Nano-Base| 0.00086 vs spectral variance 0.0012).
- Barotse floodplain: E_case retains a margin over the best control (0.0235
  vs NDWI gradient 0.0384). The wetland-margin result therefore does not
  reduce to edge detection.
- Zambezi delta: all three no-model statistics rank the disagreements as
  well as or better than every model signal (NDWI gradient 0.0005 vs E_dist
  0.0014). The delta scene's disagreements are spectrally trivial (the river
  the reference misses), so this scene supports no claim of model-signal
  superiority; the E_dist shift claim is withdrawn pending a shift testbed
  with non-trivial errors.
- Secondary observation: on both difficult scenes, even no-model statistics
  rank errors better than max-softmax confidence.

  ![No-model controls vs model signals](../exp/out/exp06_controls.png)

### Validation against expert labels (AWF, exp04)
- The full pipeline runs against the AWF partner dataset: 1459 expert-labeled
  points, 12-month Sentinel-2 stacks, the project's own 1115/344 spatial
  split. A linear head on frozen Base embeddings reaches 85.2% validation
  accuracy (the project's fully fine-tuned model: 89.5%), giving 51 errors
  for signal evaluation.
- On this in-domain multiclass task, max-softmax achieved the lowest AURC
  (0.0367) of the signals tested (tile-phase 0.0427, Nano-Base total
  variation 0.0533).
- Lowest per-class recall: herbaceous wetland (0.50, n=6 validation points,
  so indicative only). Class-index-to-name mapping verified against the
  per-class metric definitions in olmoearth_projects awf model.yaml.

  ![AWF risk-coverage and per-class recall](../exp/out/exp04_awf_expert.png)

### Single difficult scenes (exp05)
- Water heads trained at Katima Mulilo, evaluated on (a) the Barotse
  floodplain interior (ambiguous wetland margins, in-region) and (b) the
  Zambezi delta mangrove coast (~1300 km from the training region). On both
  scenes, every channel tested achieved lower AURC than max-softmax
  (floodplain: E_case 0.0235, E_dist 0.0289, tile-phase 0.0555, baseline
  0.0666; delta: E_dist 0.0014, tile-phase 0.0076, E_case 0.0103, baseline
  0.0258). The delta column was later disqualified as evidence for
  model-signal superiority by the exp06 controls; the Barotse column
  survived.
- Reference-label caveat: ESA WorldCover is least reliable on exactly these
  terrains. The signals rank model-reference disagreement correctly; whether
  that equals model error requires expert-labeled replication.

  ![Hard scenes](../exp/out/exp05_hard_scenes.png)

## Per-signal results

### Max-softmax confidence (the baseline)
- Best signal on the in-domain AWF expert-label task (exp04) and competitive
  on the easy river scene (exp02).
- Best on zero of seven scenes in the replication (exp09); on the two
  difficult single scenes it was also outranked by no-model pixel statistics
  (exp06).

### Cross-model disagreement (E_case)
- Reference point: mean local-similarity-structure agreement between Nano
  and Base on random input is approximately 0.59, not 0, because the models
  share patchification and input normalization. Agreement values must be
  interpreted relative to this floor. (exp/smoke_test.py)
- Disagreement is spatially structured: it concentrates at class boundaries
  and thin structures (shoreline, a bridge) at both the embedding level and
  the prediction level. (exp/exp01, exp/exp02)

  ![Embedding-level agreement, Kazungula](../exp/out/exp01_zambezi_agreement.png)
- Pairwise |p_Nano - p_Base| did not beat max-softmax on the easy scene
  (0.0011 vs 0.0009, exp02); it was the best signal on the Barotse
  wetland-margin scene (0.0235 vs baseline 0.0666, control 0.0384; exp05,
  exp06) and on 3 of 7 replication scenes, each win surviving the control
  (exp09).
- Aggregation: equal-weight combinations over three models performed worse
  than the best pairwise signal whenever one member was substantially weaker
  (Tiny on the water task, exp03; Nano on AWF, exp04; all DS-weighted and
  equal-weight variants on AWF, exp07). Three independent confirmations.
- Rater strength vs diversity (exp10): replacing Nano with v1-Large as
  Base's partner makes the signal worse (|Large-Base| mean AURC 0.0197 vs
  |Nano-Base| 0.0129 over the seven scenes, better on only 3/7) although
  Large is the more accurate model on every scene. Within one family, strong
  models agree on errors; the informative property of a partner is
  decorrelation, not accuracy.
- Label-free reliability (Dawid-Skene, exp07): DS EM over Nano/Tiny/Base
  votes on AWF (labels untouched) overestimates every model and inverts the
  ordering: estimated 0.868/0.921/0.887 vs measured 0.756/0.805/0.852. DS
  assumes conditionally independent raters; the family errs together, and
  agreement-on-errors is read as competence. The estimate-minus-measured gap
  (+0.112, +0.116, +0.035) directly measures correlated-error mass per
  model. Label-free accuracy estimation within a single family is
  unsupported; an out-of-family rater (Clay or AnySat, both wrapped in
  olmoearth_pretrain evals) is the designed fix, untested.

### Perturbation stability (E_system)
- Tile-phase: shifting the input window origin by 1-3 pixels (sub-patch
  phase) and taking the standard deviation of predicted probability across
  shifts was the best signal on 2 of 7 replication scenes (exp09), beat the
  baseline on the easy binary scene (0.00058 vs 0.00089; exp03), and lost to
  it on the AWF multiclass task (0.0427 vs 0.0367; exp04). The signal map
  traces the shoreline continuously.

  ![Signal maps at Kazungula](../exp/out/exp03_more_channels.png)
- Masking perturbation, rejected (exp08): occluding a random 15% of patch
  cells with mean-fill and measuring prediction standard deviation over N=32
  reruns ranks errors worse than every other signal on all three scenes
  tested (0.0027 / 0.0788 / 0.0449 vs tile-phase 0.0008 / 0.0555 / 0.0076).
  Occlusion instability measures context reliance rather than error
  likelihood. Design rule: perturbations that preserve scene content while
  changing the tokenization expose model pathology; perturbations that
  remove content do not.
- Backbone-version comparison (v1 vs v1_2): blocked; v1_2 checkpoints fail
  to load with the current olmoearth_pretrain checkout (LatentMIM state_dict
  mismatch, exp03).

### Embedding dissimilarity (E_dist)
- Mean cosine distance to the k=5 nearest training patches did not rank
  in-domain errors (AURC 0.00365 vs baseline 0.00089; exp03).
- Its apparent advantage under geographic shift (0.0014 vs 0.0258, exp05)
  did not survive the no-model control (NDWI gradient 0.0005 on the same
  disagreements; exp06), and it won zero of seven replication scenes once
  the control was included (exp09). No support as an error ranker. The
  out-of-distribution-indicator interpretation remains plausible but
  requires a shift testbed whose errors are not spectrally trivial.

### Geographic grounding (E_geo)
- On a scene where the river is clearly resolved, the OSM centerline
  consistency check produced zero false break alarms (52 centerline patches,
  0 flagged; exp02). Specificity evidence only; no scene with a confirmed
  consensus break has been evaluated, so sensitivity is unknown.

  ![Full audit slice, Kazungula](../exp/out/exp02_full_slice.png)

## Established facts

### Heads and spatial transfer
- Linear heads on frozen embeddings transfer spatially: trained on one river
  reach, 97-98% accuracy against WorldCover on a reach ~110 km away.
  (exp/exp02)
- WorldCover-as-reference carries temporal drift (2021 labels vs 2024
  scenes), so a fraction of measured "model error" is label noise.

### Production inference exports (allenai/olmoearth_lcc)
- The published continent-scale LCC run (encoder OlmoEarth-v1.2-Base,
  32768x32768-pixel UTM tiles) provides 9-band uint8 summary rasters: band 1
  binary-change probability (scaled 0-255), bands 2-5 argmax classes, bands
  6-7 the probability of the argmax class, bands 8-9 month-encoded change
  dates. The export format therefore carries partial probability information
  (top-1 score and one binary-head probability), not full per-class
  distributions. (Documented in the dataset README.)
- The rasters are cloud-optimized GeoTIFFs on public storage; windows can be
  read over HTTP without bulk download.
- Caveat for any validation against the olmoearth_lcc annotations: most
  collection phases used output-based labeling (model proposes candidates,
  annotators verify), so label locations are correlated with model beliefs.

### Infrastructure
- The olmoearth_pretrain base dependency set suffices for inference (torch,
  einops, huggingface_hub, numpy); CPU handles 128x128-pixel windows at
  patch size 4, and a single consumer GPU (tested: RTX 5090 laptop, torch
  2.7.1+cu128) makes perturbation ensembles and multi-scene sweeps
  interactive. (exp/smoke_test.py, exp/exp08, exp/exp09)
- All encoder checkpoints are public: v1 Nano/Tiny/Base/Large, v1_1
  Nano/Tiny/Base, v1_2 Nano/Tiny/Small/Base, plus fine-tuned variants (AWF,
  LFMC, Mangrove, ForestLossDriver, EcosystemTypeMapping). Multi-model and
  cross-version signals require no private infrastructure.
- Planetary Computer Sentinel-2 L2A and ESA WorldCover can be read onto a
  shared 10 m grid; OSM Overpass requires mirror fallback.

## Second use case: change attribution between two inference results

Error ranking is one consumer of the signals. The same decomposition applies
to comparing two inference outputs (two dates, two model versions): a raw
diff mixes real surface change, model instability, low-consensus
predictions, and geographically implausible transitions. E_system
instability and E_case consensus at each date gate which diffs are
trustworthy; geographic and temporal plausibility rules constrain which
transitions are physically possible; plain image differencing is the
no-model control. Intended application: automated interpretation of what
changed over time (EO autoresearch), where a change narrative should only be
generated from diffs that survive this decomposition. Natural testbed: the
olmoearth_lcc production change product (change probability and
month-encoded dates) and its verified change/no-change points. Untested;
design only.

## Open items, in priority order

1. Out-of-family rater (Clay or AnySat) for disagreement and Dawid-Skene:
   correlated errors invalidate within-family DS (exp07) and cap pairwise
   quality (exp10); an architecture-independent rater is required for both.
2. A domain-shift testbed with non-trivial errors and expert labels
   (candidate design: geographic-corner holdout within the AWF dataset). The
   delta scene is disqualified as evidence by the exp06 controls.
3. E_geo sensitivity: evaluate on a scene containing a confirmed consensus
   break and measure whether the centerline check detects it.
4. v1 vs v1_2 comparison on identical windows; requires updating the
   olmoearth_pretrain checkout for the v1_2 loader.
5. Confidence intervals and significance testing for AURC comparisons, now
   that seven scenes exist for the water task.
6. E_dist formalization: AOA/Dissimilarity Index (Meyer & Pebesma 2021) in
   place of raw k-NN distance; delineate overlap with SHRUG-FM before
   claiming novelty.
7. Confirm whether Studio per-project exports match the olmoearth_lcc export
   format (partial probabilities).
8. Replace OSM centerlines with GRIT, which adds width attributes that E_geo
   can condition on.
9. Audit a window of the published LCC production output directly (HTTP
   range reads) against river centerlines; first step toward the
   change-attribution use case.

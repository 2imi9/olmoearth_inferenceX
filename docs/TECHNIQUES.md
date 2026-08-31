# Technique ledger

Standing record of what is verified, organized by technique. No timeline: a claim
lives here only once an experiment supports it, and gets amended or deleted when
a later experiment says otherwise. Chronology lives in exp/NOTES.md. Every entry
names its evidence (exp script) so claims stay reproducible.

Status codes: VERIFIED (experiment supports it) / PARTIAL (some evidence, key
condition untested) / UNTESTED / OUT-OF-SCOPE-V1.

## Summary

| Technique | LLM ancestor | Status | Gap that remains |
|---|---|---|---|
| Cross-model disagreement (E_case) | SelfCheckGPT / self-consistency | PARTIAL | pairwise ties baseline on easy scene; naive tri-model std is WORSE (weak rater pollutes); aggregation design + hard-scene test remain |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | PARTIAL | specificity shown (no false alarms), sensitivity never tested (no scene with a real break yet) |
| Perturbation stability (E_system) | semantic entropy / paraphrase robustness | MIXED | beat max-softmax on the easy binary scene (0.00058 vs 0.00089) but LOST on AWF multiclass expert labels (0.0427 vs 0.0367); condition-dependent, not a general win |
| Backbone-version comparison (E_system) | n/a (EO-specific) | BLOCKED | v1_2 load fails on current checkout (LatentMIM state_dict mismatch, confirmed); pull newer olmoearth_pretrain |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, SEPs) | PARTIAL | k-NN-to-train tested: does NOT rank in-domain errors (4x worse than baseline). Role is OOD alarm, not error proxy; needs an actually-OOD eval to verify that role |
| Max-softmax baseline | logit confidence | VERIFIED | the bar; tile-phase is the first channel to beat it |
| Risk-coverage / AURC harness | selective prediction | VERIFIED | needs CIs / significance once scenes multiply |
| Weak-truth validation loop | MRP pseudo-label validation (PPE) | VERIFIED | harness now runs on real partner expert labels (AWF, official rslearn spatial split, 344 val points) |
| Semantic-entropy port (cluster-then-entropy) | Farquhar et al. | UNTESTED | refinement of E_system, not started |
| Verifier head trained on labeled regions | process reward models | OUT-OF-SCOPE-V1 | needs labels as training input; revisit after channels ship |
| Channel fusion | n/a | OUT-OF-SCOPE-V1 | per-channel ranking only; Borda if forced |

## Verified facts

### Production inference exports (allenai/olmoearth_lcc)
- The at-scale LCC run (encoder OlmoEarth-v1.2-Base, continent-scale Africa,
  32768x32768-px UTM tiles) publishes 9-band uint8 summary COGs: band 1
  binary-change probability (0-255), bands 2-5 argmax classes, bands 6-7
  probability of the argmax class, bands 8-9 month-encoded change dates.
  Answer to the probabilities-vs-argmax question: PARTIAL probabilities
  (top-1 score + binary head), not full per-class distributions. Max-softmax
  ships in the product, so the baseline comparison runs directly on
  production output. (README of allenai/olmoearth_lcc)
- Those COGs are public; windows are HTTP range-readable, so production
  output can be audited without Ai2 infrastructure.
- Label-bias caveat for any validation against olmoearth_lcc annotations:
  most collection phases are output-based labeling (model proposes, human
  verifies), so label locations correlate with model beliefs. Relevant to the
  correlated-error open question.

### Infrastructure
- Inference-only install of olmoearth_pretrain is the base dependency set
  (torch, einops, hf_hub, numpy); no training extra needed. CPU sufficient for
  128x128 windows at patch 4. (exp/smoke_test.py)
- All checkpoints public and ungated on HF: v1 Nano/Tiny/Base/Large, v1_1
  Nano/Tiny/Base, v1_2 Nano/Tiny/Small/Base, plus FT variants (AWF, LFMC,
  Mangrove, ForestLossDriver, EcosystemTypeMapping). Multi-model and
  cross-version signals need zero Ai2 infrastructure.
- Planetary Computer S2 L2A + ESA WorldCover on a shared 10 m grid works as a
  self-serve data path. OSM Overpass needs mirror fallback (mail.ru mirror most
  reliable in practice). HF_HUB_OFFLINE=1 for cached-checkpoint reruns.

### Cross-model disagreement (E_case)
- Noise floor: mean local-structure agreement between Nano and Base on random
  input is ~0.59, not 0 (shared patchification/normalization). Any agreement
  claim must be read against this floor. (exp/smoke_test.py)
- Disagreement is structured, not random: it concentrates at class boundaries
  and thin structures (shoreline, bridge) at both embedding level and
  prediction level. (exp/exp01, exp/exp02)
- On an easy scene (400 m river, dry season, 2% error rate), |p_a - p_b|
  does NOT beat max-softmax on AURC (0.0011 vs 0.0009). The model's own
  confidence suffices when the task is easy. (exp/exp02)

### Perturbation stability (E_system tile-phase)
- Shifting the input window by 1-3 px (sub-patch phase) and measuring the std
  of water probability across shifts ranks Base's errors BETTER than the
  model's own confidence: AURC 0.00058 vs 0.00089 on the Kazungula scene. The
  signal map traces the shoreline continuously. First channel to beat the
  baseline. Single scene, 18 error patches: directional until replicated.
  (exp/exp03)

### Cross-model disagreement, aggregation
- Naive tri-model std underperforms pairwise |Nano-Base| (AURC 0.00105 vs
  0.00086) because Tiny is the weakest head (0.951 vs 0.973/0.982) and equal
  weighting lets it inject noise. Multi-rater aggregation needs reliability
  weighting (Dawid-Skene direction), not plain std. (exp/exp03)

### Embedding dissimilarity (E_dist)
- k-NN cosine distance to train patches does not rank in-domain errors (AURC
  0.00365, 4x worse than baseline). Expected and now demonstrated: novelty is
  not error when the eval region is in-domain. E_dist's claim must be tested
  on an actually out-of-domain region. (exp/exp03)

### Partner-truth validation (AWF)
- The full audit harness runs against real expert labels: AWF dataset, official
  1115/344 rslearn spatial split, frozen Base + multiclass head reaches 85.2%
  (their finetuned model: 89.5%). Error sample: 51 errors. (exp/exp04)
- On this in-domain multiclass task the max-softmax baseline WINS: AURC 0.0367
  vs tile-phase 0.0427 vs Nano-Base TV 0.0533. Tile-phase's easy-scene win did
  not transfer. Working hypothesis: confidence excels at ranked in-domain
  class-confusion errors; the channels' claimed territory is correlated /
  systematic / OOD failures, which this task does not exercise. (exp/exp04)
- Weak-rater effect replicated: Nano is 75.6% on this task and TV disagreement
  against it is the worst signal tested. Two tasks, two confirmations:
  disagreement needs raters of comparable strength or reliability weighting.
- Weakest class: herbaceous wetland (50% recall). Wetland margins are also the
  posited hard case for the water task; consistent story. (exp/exp04)

### Geographic grounding (E_geo)
- OSM centerline check produces zero false break alarms on a scene where the
  river is clearly resolved (52 centerline patches, 0 consensus-dry).
  Specificity evidence only; sensitivity untested. (exp/exp02)

### Heads and transfer
- Frozen-embedding logistic heads transfer spatially: trained on one reach,
  97-98% accuracy vs WorldCover on a reach ~110 km away. Spatial split is
  cheap to honor and should never be dropped. (exp/exp02)
- WorldCover-as-truth carries temporal drift error (2021 labels vs 2024
  scenes; moving sandbars). Treat a few points of "model error" as label noise.

## Gaps to fill, in priority order

1. Hard-scene E_case vs baseline. The design claim is that disagreement beats
   confidence where the task is hard (narrow/sub-patch channels, flood-season
   wetland margins). No experiment yet. This is the make-or-break test.
2. E_geo sensitivity. Find or construct a scene with a real consensus break
   (narrow reach the models actually miss) and show the centerline check fires.
   Until then E_geo has only proven it stays quiet.
3. Tri-model E_case. Add Tiny; three raters is the Dawid-Skene identifiability
   minimum. Also decide the aggregation (pairwise mean vs majority-vs-outlier).
4. E_system tile-phase. Shift the window grid by fractions of a patch, measure
   prediction flips. Uses the same cached-window machinery as exp02.
5. E_system v1 vs v1_2. Same windows, both backbones. Requires pulling newer
   olmoearth_pretrain for the v1_2 loader path.
6. E_dist. Compute AOA/DI over embeddings for a window vs a training-domain
   reference sample. Read SHRUG-FM first and record here exactly what it
   already covers so the contribution boundary is explicit.
7. RESOLVED for the at-scale LCC pipeline (see Verified facts): exports carry
   binary-change probability + argmax classes + top-1 probability, not full
   distributions. Remaining sliver: confirm Studio per-project exports match.
8. Partner-project validation. Labels FOUND, not yet used:
   allenai/olmoearth_projects_awf (expert AWF LULC annotations, 418 KB
   geojson, pairs with public OlmoEarth-v1-FT-AWF-Base checkpoint),
   olmoearth_projects_mangrove, and allenai/olmoearth_lcc training_data
   (verified change/no-change points). Next: run the audit signals over an AWF
   AOI and score against the expert labels instead of WorldCover.
9. GRIT/GRWL centerlines. OSM is the placeholder reference; GRIT is the real
   one and adds width attributes E_geo can condition on.

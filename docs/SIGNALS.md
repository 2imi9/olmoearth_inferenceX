# Per-signal evidence

One section per signal, every claim citing its experiment. Index at
[TECHNIQUES.md](TECHNIQUES.md).

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
  [olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals))
  is the designed fix, untested.

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

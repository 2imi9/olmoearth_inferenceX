# Per-signal evidence

One section per signal, every claim citing its experiment. Index at
[TECHNIQUES.md](../TECHNIQUES.md).

### Max-softmax confidence (the baseline)
- Best signal on the in-domain AWF expert-label task (exp04) and competitive
  on the easy river scene (exp02).
- Replication: best on 0/7 scenes in exp09 (superseded), 6/29 by raw AURC
  in exp11 (superseded), and 0/27 on the rule-selected set with
  aligned tile-phase and tie-aware E-AURC (exp13, authoritative), where
  aligned tile-phase beats it on 26/27.

### Cross-model disagreement (E_case)
- Reference point: mean local-similarity-structure agreement between Nano
  and Base on random input is approximately 0.59, not 0, because the models
  share patchification and input normalization. Agreement values must be
  interpreted relative to this floor. (exp/smoke_test.py)
- Disagreement is spatially structured: it concentrates at class boundaries
  and thin structures (shoreline, a bridge) at both the embedding level and
  the prediction level. (exp/exp01, exp/exp02)

  ![Embedding-level agreement, Kazungula](../../exp/out/exp01_zambezi_agreement.png)
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
  ordering: estimated 0.859/0.916/0.876 vs measured 0.753/0.802/0.817. DS
  assumes conditionally independent raters; the family errs together, and
  agreement-on-errors is read as competence. The estimate-minus-measured gap
  (+0.106, +0.114, +0.059) directly measures correlated-error mass per
  model. Label-free accuracy estimation within a single family is
  unsupported; an out-of-family rater (Clay or AnySat, both wrapped in
  [olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals))
  is the designed fix, untested.

### Perturbation stability (E_system)
- Tile-phase, defined correctly: shift the input window origin by 1-3
  pixels (sub-patch phase), upsample each shifted prediction map to pixels,
  place it at its true offset on a common canvas, take the per-pixel
  standard deviation across shifts, and pool back to the shift-0 patch grid
  (exp03 method). On the 27 rule-selected scenes this beats the baseline
  on 26/27 (sign p=4e-07), the pixel control on 18/27, and is
  the most frequent best signal (exp13). By construction it is largest where
  neighboring patches disagree, so beating the pixel-edge control is the
  relevant test.
- Mechanism (exp14): a discrete boundary indicator computed from the
  shift-0 hard prediction map alone (fraction of 8 neighbors with a
  different label) is statistically indistinguishable from aligned
  tile-phase across the 27 scenes (boundary better on 12, tile-phase on 15,
  sign p=0.70) and by itself beats the baseline on 19/27 and the pixel
  control on 22/27. No advantage of the perturbation beyond boundary
  proximity is detectable. Its loss on AWF (exp04) is examined in exp16: the
  prediction-boundary score at labelled patches is error-associated but
  largely a proxy for low margin on the nine-class task (Spearman
  0.60 with the margin; logistic LRT p=0.002 with a small coefficient),
  so confidence ranks better (0.0363 vs 0.0636). The "no boundary
  context" explanation is withdrawn.
- Erratum: exp05, exp09, and exp11 computed the std across unaligned patch
  grids, so shifted patches covered different ground; that version scored
  19/29 in exp11 (29 scenes, raw AURC) with no significant effect. Their tile-phase numbers are
  superseded. exp04's per-window variant tracked the label patch under
  shift and is unaffected (lost to the baseline there, 0.0489 vs 0.0363).
- Earlier single-scene results consistent with the corrected signal: beat
  the baseline on the easy binary scene (0.00058 vs 0.00089; exp03); the
  signal map traces the shoreline continuously.

  ![Signal maps at Kazungula](../../exp/out/exp03_more_channels.png)
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
- Definition as implemented: mean cosine distance from a window's Base
  embedding to its k=5 nearest patches of the head's training window (one
  128x128 scene at Katima Mulilo, or the AWF training split in exp12). This
  is distance to the head's training region, not to the encoder's
  pretraining distribution; an AOA-style reference sample over the
  pretraining domain has not been built.
- It did not rank in-domain errors (AURC 0.00365 vs baseline 0.00089;
  exp03).
- Its apparent advantage under geographic shift (0.0014 vs 0.0258, exp05)
  did not survive the no-model control (NDWI gradient 0.0005 on the same
  disagreements; exp06), and it won zero of seven replication scenes once
  the control was included (exp09, superseded). On the 27 rule-selected
  scenes it beats the baseline on 13/27 (sign p=1.00; intervals
  11 in favor, 5 against; exp13). No general support as an error
  ranker. The out-of-distribution-indicator interpretation remains plausible
  but requires a shift testbed whose errors are not spectrally trivial.

### Geographic grounding (E_geo)
- On a scene where the river is clearly resolved, the OSM centerline
  consistency check produced zero false break alarms (52 centerline patches,
  0 flagged; exp02).
- Across 27 georeferenced rule-selected scenes (exp15), flags (centerline
  patch predicted dry) occur on 17 scenes with pooled precision 0.12
  over 413 flags (1.5x the base error rate 0.081); 9 scenes
  carry 1-51 flags at precision zero where OSM marks a river that both
  the model and WorldCover call dry. The flag mostly detects reference-map
  disagreement under the current truth. As a ranking signal it beats the
  baseline on 3/27; prepending it to boundary proximity gives 5 better,
  9 worse, 13 unchanged (sign p=0.42). Sensitivity remains unmeasurable
  without width-filtered centerlines (GRWL) or expert truth.

  ![Full audit slice, Kazungula](../../exp/out/exp02_full_slice.png)

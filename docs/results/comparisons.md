# Cross-signal comparisons

All signals scored on identical errors with the same harness, per
condition. Index at [TECHNIQUES.md](../TECHNIQUES.md); protocol and status
terms in [protocol.md](../method/protocol.md).

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

  ![Per-scene AURC](../../exp/out/exp09_multiscene.png)

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

  ![No-model controls vs model signals](../../exp/out/exp06_controls.png)

### Validation against expert labels (AWF, exp04)
- Source: the
  [olmoearth_projects_awf dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf)
  with classes and split defined by its
  [task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml).
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
- Completing the comparison on the same 51 errors (exp12): E_dist
  knn-to-train 0.1104 and a no-model spectral-variability control 0.1287,
  both far behind the baseline. In-domain AWF errors are neither
  out-of-distribution nor pixel-trivial, consistent with the water-task
  pattern.

  ![AWF risk-coverage and per-class recall](../../exp/out/exp04_awf_expert.png)

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

  ![Hard scenes](../../exp/out/exp05_hard_scenes.png)

### Corrected statistics on the 29-scene set (exp13, authoritative)
- Same scenes, errors, and heads as exp11, recomputed with three
  corrections: (1) tile-phase aligned on a pixel canvas before taking the
  std across shifts (exp05/09/11 compared unaligned patch grids, so shifted
  patches covered different ground); (2) excess AURC (E-AURC = AURC minus
  the oracle's) so scenes with different error rates are comparable; (3) a
  4x4-patch block bootstrap (B=1000) for per-scene intervals, since patches
  are spatially autocorrelated. Per-scene values in
  exp/out/exp13_corrected_stats.csv.
- Aligned tile-phase beats the baseline on 27/29 scenes (exact sign test
  p<0.001; sign-flip permutation on E-AURC differences p=0.002); its block
  bootstrap interval excludes zero in its favor on 18 scenes and against
  on none. It beats the pixel control on 19/29, E_case on 24/29, E_dist on
  25/29, and is the best signal on 12/29 (best or second on 24/29). This is
  the first result in the repository that is both statistically robust and
  control-surviving.
- The unaligned tile-phase of exp11 scores 19/29 (sign p=0.14, permutation
  p=0.95): the misalignment hid the effect. Recorded as an erratum.
- E_case: 12/29, sign p=0.46, interval excludes zero in its favor on 2
  scenes and against on 10. Not a general signal; unchanged conclusion.
- E_dist: 13/29, sign p=0.71. The exp11 permutation p=0.019 was a scale
  artifact of raw AURC dominated by the high-error Shire scenes; on
  scale-comparable E-AURC there is no advantage (intervals: 11 better, 4
  worse). Superseded.
- Control: 13/29 vs baseline; best signal on 9/29, the reference-omission
  scenes. Baseline best on 1/29.
- Head-seed variance remains structurally zero (deterministic head
  training), as noted under exp11.

### Pre-registered 29-scene comparison (exp11; statistics superseded by exp13)
- The scene rule and scene set below stand. The statistics in this section
  use raw AURC and the unaligned tile-phase; exp13 corrects both and is
  authoritative where they differ.
- Scene rule committed to git before any new scene was fetched: candidates
  sampled at fixed fractions along OSM geometries of eight named rivers,
  0.2-degree separation, inclusion iff the deterministic Base head commits
  >=8 errors against WorldCover. 22 rule-selected scenes joined the 7
  existing ones; 29 total. Per-scene AURC, bootstrap 95% CIs (B=1000), and
  seed columns in exp/out/exp11_stats.csv.
- Baseline best on 6/29 scenes: the exp09 claim "confidence never best
  (0/7)" did not survive scene expansion and is superseded.
- Sign-flip permutation tests on per-scene AURC differences vs baseline:
  E_case better on 12/29, mean difference -0.0058 (trend toward worse,
  p=0.070); tile-phase better on 19/29 but mean difference -0.0002
  (p=0.950); E_dist better on 13/29 with mean difference +0.0098
  (p=0.019), the only significant mean improvement, driven by high-error
  floodplain scenes (Shire, Okavango mid-stem: 8-30% error rates) where
  WorldCover reliability is lowest; control not significant (p=0.848).
- Coherent summary: no signal dominates across scenes. Which signal ranks
  errors best is regime-dependent, which elevates regime identification
  (deciding per scene which signal to trust) from a side idea to the central
  open problem.
- Head-seed variance is structurally zero: heads initialize at zeros with
  deterministic full-batch training, so the planned seed-robustness test is
  vacuous rather than passed; robustness to head initialization remains
  untested by design choice.
- Standing caveats unchanged: WorldCover reference (worst on exactly the
  high-error scenes that drive the E_dist result), no expert labels outside
  AWF, one task family.

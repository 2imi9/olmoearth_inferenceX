# Cross-signal comparisons

All signals scored on identical errors with the same harness, per
condition. Index at [TECHNIQUES.md](../TECHNIQUES.md); protocol and status
terms in [protocol.md](../method/protocol.md).

### Multi-scene replication (exp09; tile-phase unaligned and statistics superseded by exp13)
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
  split. A linear head on frozen Base embeddings reaches 81.7% validation
  accuracy (the project's fully fine-tuned model: 89.5%), giving 63 errors
  for signal evaluation.
- On this in-domain multiclass task, max-softmax achieved the lowest AURC
  (0.0363) of the signals tested (tile-phase 0.0489, Nano-Base total
  variation 0.0670).
- Lowest per-class recall: herbaceous wetland (0.50, n=6 validation points,
  so indicative only). Class-index-to-name mapping verified against the
  per-class metric definitions in olmoearth_projects awf model.yaml.
- Completing the comparison on the same 63 errors (exp12): E_dist
  knn-to-train 0.1338 and a no-model spectral-variability control 0.1658,
  both far behind the baseline. In-domain AWF errors are neither
  out-of-distribution nor pixel-trivial, consistent with the water-task
  pattern.

  ![AWF risk-coverage and per-class recall](../../exp/out/exp04_awf_expert.png)

### Single difficult scenes (exp05; tile-phase values unaligned, superseded by exp13)
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

### Dense expert labels: Sen1Floods11 (exp18, authoritative for the water task)
- Testbed: Sen1Floods11 hand-labelled flood-water masks (Ai2's own
  evaluation set, public mirror), 64x64 tiles; head trained on 600 tiles of
  the valid split; scored on Bolivia (441 tiles, a geographically held-out
  region; 351 tiles with enough errors to score) and on 800 test tiles.
  Sentinel-2 Level-1C chips fed through the L2A path (documented mismatch;
  head accuracy 0.912 Bolivia, 0.953 test). All signals computed on one
  60x60 crop per tile; tie-aware excess AURC per tile; exact sign tests.
  Values in exp/out/exp18_sen1floods.csv.
- Against expert labels, confidence is the best signal. Bolivia: aligned
  tile-phase 163 better / 187 worse (p=0.22), band-set disagreement
  111/239 (p=7e-12), boundary indicator 134/217 (p=1e-5), E_case 84/265
  (p=6e-23), E_dist 22/329, pixel control 79/271. Test split: tile-phase
  173/305 (p=2e-9), band-set 148/334, boundary 154/327, E_case 141/340.
  Pooled E-AURC over all patches: confidence 0.0105 (Bolivia) and 0.0096
  (test), lowest of all signals on both.
- Errors still concentrate on prediction boundaries (75% of error patches
  vs 21% of correct patches on Bolivia; 73% vs 18% on test), so the
  boundary phenomenon is real; confidence simply ranks those errors better
  than boundary proximity or instability do.
- Reading: the WorldCover-referenced advantages of tile-phase (26/27) and
  band-set disagreement (21/27) do not transfer to expert truth. The
  explanation first proposed here, that a large share of the counted
  "errors" were reference errors detectable by boundary-type signals, was
  tested in exp23 with the measurable component of reference error
  (disagreement between WorldCover's own 2020 and 2021 versions) and is not
  supported: that component covers about 10% of the disagreements and the
  advantage persists on reference-stable patches. Remaining candidates are
  reference errors shared by both versions (narrow and seasonal water),
  genuine change between 2021 and the 2024 imagery, and properties of the
  WorldCover-defined task. Until a signal beats confidence on
  expert-labelled dense maps, the repository's positive claims are claims
  about detecting disagreement with a weak reference, not about detecting
  model error.
- Caveats: Level-1C inputs through an L2A path; flood water rather than
  permanent water; one crop per tile; the head is trained on tiles from
  the same eleven regions as the test split (Bolivia is the clean spatial
  hold-out). None of these plausibly favour confidence over the other
  signals.

### Corrected statistics on the rule-selected scene set (exp13; WorldCover reference, superseded by exp18 for the water task)
- Same errors and heads as exp11, recomputed with: (1) tile-phase aligned
  on a pixel canvas before taking the std across shifts (exp05/09/11
  compared unaligned patch grids); (2) excess AURC (E-AURC = AURC minus the
  oracle's), which makes absolute levels comparable across scenes; it does
  not change any signal-minus-baseline difference or any test on them; (3)
  a 4x4-patch block bootstrap (B=1000) for per-scene intervals; (4) AURC
  computed as its expectation under random tie-breaking, since the boundary
  score has nine levels and float32 sigmoid saturation ties the
  max-softmax uncertainty at zero on a large share of patches, so a
  stable-sort AURC depends on raster order; (5) the baseline scored as the
  negative absolute logit, monotone in max-softmax and tie-free where the
  sigmoid saturates. Two scenes (kafue, luangwa) that entered exp11 through
  a cache import rather than the pre-registered rule are excluded; 27
  scenes remain. Per-scene values in exp/out/exp13_corrected_stats.csv.
- Aligned tile-phase beats the baseline on 26/27 scenes (1 worse, 0 tied;
  exact sign test on untied pairs p=4e-07; sign-flip permutation on
  mean E-AURC differences p=0.000); its percentile block-bootstrap
  interval excludes zero in its favor on 18 scenes and against on
  0. It beats the pixel control on 18/27, E_case on
  23/27, E_dist on 22/27, and is the best of the five
  signals on 12/27 (best or second on 24/27). Per-scene
  percentile intervals are biased for this rank statistic on high-error
  scenes and are indicative only; the cross-scene sign and permutation
  tests do not use them.
  ![Per-scene tie-aware AURC, 27 rule-selected scenes](../../exp/out/exp13_per_scene.png)

  Per-scene tie-aware AURC (bold = lowest unrounded value per scene):

  | scene | errors | baseline | E_case | tile-phase | E_dist | control |
  |---|---|---|---|---|---|---|
  | barotse | 97 | 0.0684 | 0.0235 | **0.0127** | 0.0289 | 0.0384 |
  | cuando_20 | 49 | 0.0035 | 0.0075 | **0.0025** | 0.0039 | 0.0092 |
  | cuando_50 | 168 | 0.0536 | 0.0794 | 0.0277 | **0.0189** | 0.0316 |
  | cuando_80 | 61 | 0.0072 | 0.0155 | **0.0050** | 0.0092 | 0.0093 |
  | delta | 29 | 0.0234 | 0.0103 | 0.0009 | 0.0014 | **0.0005** |
  | kafue_20 | 77 | 0.0164 | 0.0244 | **0.0078** | 0.0407 | 0.0287 |
  | kafue_50 | 48 | 0.0241 | 0.0275 | **0.0027** | 0.0089 | 0.0099 |
  | kafue_80 | 38 | 0.0128 | 0.0068 | 0.0020 | 0.0138 | **0.0014** |
  | kazungula | 18 | 0.0009 | 0.0009 | **0.0006** | 0.0037 | 0.0016 |
  | luangwa_conf | 52 | 0.0103 | 0.0189 | **0.0072** | 0.0108 | 0.0597 |
  | okavango_50 | 96 | 0.1452 | 0.1612 | **0.1175** | 0.1339 | 0.1229 |
  | okavango_80 | 76 | 0.0572 | 0.0578 | 0.0061 | 0.0105 | **0.0037** |
  | okavango_sep | 13 | 0.0005 | **0.0003** | 0.0003 | 0.0007 | 0.0163 |
  | rovuma_20 | 39 | 0.0053 | 0.0095 | **0.0021** | 0.0104 | 0.0073 |
  | rovuma_50 | 74 | 0.0091 | 0.0091 | **0.0080** | 0.0152 | 0.0160 |
  | rovuma_80 | 18 | 0.0016 | 0.0030 | 0.0010 | 0.0014 | **0.0006** |
  | save_20 | 25 | 0.0208 | 0.0619 | 0.0005 | **0.0005** | 0.0040 |
  | save_50 | 76 | 0.0140 | 0.0409 | **0.0116** | 0.0163 | 0.0676 |
  | save_80 | 23 | 0.0069 | 0.0292 | 0.0006 | 0.0006 | **0.0005** |
  | shire_20 | 303 | 0.1572 | 0.1605 | 0.1778 | **0.1134** | 0.2558 |
  | shire_50 | 123 | 0.1008 | 0.1000 | 0.0636 | 0.0629 | **0.0584** |
  | shire_80 | 188 | 0.3017 | 0.2947 | 0.1881 | 0.2172 | **0.1487** |
  | shire_liwonde | 17 | 0.0335 | 0.0357 | 0.0252 | 0.0013 | **0.0006** |
  | vicfalls_up | 23 | 0.0008 | **0.0006** | 0.0008 | 0.0012 | 0.0412 |
  | zambezi_20 | 51 | 0.0130 | 0.0118 | 0.0062 | 0.0168 | **0.0059** |
  | zambezi_50 | 16 | 0.0003 | **0.0003** | 0.0003 | 0.0037 | 0.0005 |
  | zambezi_80 | 47 | 0.0035 | 0.0148 | **0.0014** | 0.0085 | 0.0326 |


- The unaligned tile-phase of exp11 scores 17/27
  (sign p=0.25): the misalignment hid the effect. Erratum.
- E_case: 10/27 (sign p=0.25); intervals in its favor on
  3 scenes and against on 9. Not a general signal.
- E_dist: 13/27 (sign p=1.00); mean-difference sign-flip
  permutation p=0.01, unchanged from exp11 because the oracle
  subtraction cancels in within-scene differences. The mean-based test is
  carried by the high-error scenes; the scale-free sign test is reported as
  primary and shows no advantage. Intervals: 11 in favor, 5 against.
- Control: 14/27 vs baseline; best signal on 9/27, the
  reference-omission scenes. Baseline best on 0/27.
- Head-seed variance is structurally zero (deterministic head training), as
  noted under exp11.

### Boundary ablation of tile-phase (exp14)
- Question: is aligned tile-phase a perturbation signal or a detector of
  boundaries in the model's own prediction map? Two zero-cost proxies from
  the shift-0 map alone: gradient magnitude of the probability map, and the
  fraction of a patch's 8 neighbors whose hard label differs. Scored with
  tie-aware E-AURC on the 27 rule-selected scenes, alongside the pixel
  control.
- The discrete boundary fraction is statistically indistinguishable from
  tile-phase across scenes: boundary better on 12, tile-phase on 15,
  tied on 0 (sign p=0.70, median E-AURC difference -0.00012); per-scene
  values differ in both directions, so this is a null result, not an
  equivalence. The boundary fraction alone beats the baseline on 19/27
  (p=0.052) and the pixel control on 22/27 (p=0.002); the
  continuous gradient is weaker (tile-phase better on 25/27).
  Best-signal tally among the five: {'pred-boundary': 10, 'tile-phase (aligned)': 10, 'control': 5, 'baseline': 2}.
- Conclusion: no advantage of the perturbation beyond boundary proximity is
  detectable. The supported claim is that errors concentrate at prediction
  boundaries and the aligned tile-phase signal ranks them better than
  confidence; the zero-cost boundary indicator is indistinguishable from
  tile-phase across scenes, but its own margin over confidence is marginal
  (p=0.05), so the zero-inference shortcut is suggestive, not established. The exp04 loss was examined in exp16: labelled patches are not
  interior, the score is error-associated, but it is largely a proxy for
  low margin on the nine-class task, so confidence wins there. Values in exp/out/exp14_boundary_ablation.csv.

### Boundary proximity combined with the reference-map check (exp15)
- E_geo flag = patch on an OSM waterway=river centerline that the model
  predicts dry. Georeferencing recovered for 27 of the 27 rule-selected
  scenes (Overpass failures excluded). Signals: boundary (exp14), geo flag
  alone, geo-first-then-boundary.
- Prepending the flag changes the ranking only on scenes carrying flags:
  better on 5, worse on 9, unchanged on 13 of 27 (exact sign test on
  untied pairs p=0.42). No benefit shown; not significantly worse. Geo
  alone beats the baseline on 3/27; boundary alone on 19/27.
- E_geo sensitivity: 17/27 scenes carry flags; pooled precision
  0.12 over 413 flags against a base error rate of 0.081
  (1.5x), while the unweighted per-scene mean of 0.22 is inflated by
  scenes with one or two flags. 9 scenes carry 1-51 flags at precision
  exactly zero: OSM marks a river that both the model and WorldCover call
  dry, i.e. disagreement between two reference maps on narrow or seasonal
  channels. Under WorldCover truth, E_geo precision cannot be separated
  from the reference confound. Values in exp/out/exp15_boundary_geo.csv.
- Implication: E_geo needs width-filtered centerlines (GRWL width
  attribute) or expert truth before its sensitivity can be stated.

### Boundary signal on the AWF point-label task (exp16)
- The explanation offered for exp04, "point labels carry no boundary
  context", was tested directly. The Base head was applied densely to all
  64 patches of each validation crop and the exp14 boundary indicator was
  computed at the labelled patch; errors reproduce exp04 exactly
  (63/344). The script was adversarially reviewed before recording
  (exp/out/review_exp16_wf_5d95d304.json); the review's corrections are
  built in: the score is derived from the head's own prediction map, not
  ground-truth boundaries, so both questions below are tested against the
  right reference, and uncertainty uses a cluster bootstrap over the
  30 annotation tasks.
- Ranking: the negative logit margin (confidence) ranks errors better than
  the boundary score (tie-aware AURC 0.0363 vs 0.0636; cluster-bootstrap
  95% interval on the difference [+0.0023, +0.0562], P(boundary better)
  = 0.016) and than per-window tile-phase (0.0489; interval
  [+0.0023, +0.0221]).
- Are labelled patches interior? No. The labelled patch's score is zero on
  47% of windows against 43% for the other patches of the same
  maps; its within-window quantile averages 0.46, and paired against a
  random non-label patch it is higher on 89 windows and lower on
  129 (sign p=0.008). Labelled patches are, if anything, slightly
  less boundary-like than an arbitrary patch.
- Does the score carry error information? Marginally, strongly: 90% of
  errors have a nonzero score against 44% of correct windows (Fisher
  p=2e-12; Mann-Whitney z=8.9); error rate rises from
  4% at score 0 to about 64% at score 1.
  Conditionally, little: the score correlates with the margin (Spearman
  0.60), and in a logistic model of error on both, the margin dominates
  (standardized coefficients 3.53 vs 0.52; likelihood-ratio test for
  adding the score chi2=9.5, p=0.002). The margin re-quantized to the
  score's own tie-group sizes still scores 0.0367, so the loss is
  not a granularity effect.
- Reading: on a nine-class task the argmax flips between neighbouring
  patches wherever margins are small, so the prediction-boundary score is
  largely a coarse proxy for low confidence; the margin already carries
  that information at finer resolution. The "no boundary context"
  explanation is withdrawn; the in-domain result (confidence best on AWF)
  stands with this explanation. Caveat: the AWF split is by point, not by
  task, so every validation task also contributes training windows. Values
  in exp/out/exp16_awf_boundary.csv and exp16_summary.json.

### Evidence from inside the encoder (exp17)
- Setup: the v1-Base encoder was hooked (per-block outputs, last-block q/k)
  for one forward pass per scene on the 27 rule-selected scenes plus the
  training scene. Signals, all label-free and single-model: depth-probe
  disagreement (the water head retrained on each block's tokens; std over
  the last six blocks), decision settling (the final head applied to every
  block's tokens, logit-lens style; std over the last six), representation
  drift (cosine change between consecutive late blocks), band-set
  disagreement (heads trained separately on the 10 m, 20 m and 60 m
  Sentinel-2 band-set tokens; std of the three probabilities), and
  last-block attention entropy. Scored with tie-aware E-AURC against the
  exp13 errors, alongside the baseline, aligned tile-phase and the pixel
  control. Values in exp/out/exp17_internal_evidence.csv.
- Band-set disagreement beats the confidence baseline on 21/27 scenes
  (6 worse; exact sign test p=0.006) and the pixel control on 16/27, from a
  single forward pass with no second model. It outperforms the two-model
  E_case (10/27). This is the second statistically supported,
  control-surviving positive signal in the repository, after aligned
  tile-phase (26/27 on the same scenes).
- Depth-probe disagreement is marginal: 19/27 vs baseline (p=0.052), 13/27
  vs control.
- Decision settling (0/27), representation drift (3/27) and attention
  entropy (3/27) are decisively worse than confidence; the INSIDE-style
  internal-state signals do not transfer to this setting.
- Error correlation with the final head (mean phi over scenes): Nano 0.61,
  depth probe 0.64, logit-lens 0.61, 20 m band-set probe 0.75. The band-set
  probe is the most correlated rater yet yields the best disagreement
  signal, so error decorrelation alone does not predict the value of a
  disagreement partner; a partner that sees a different view of the input
  does. This refines the exp10 conclusion.
- Best-signal tally: aligned tile-phase 13, control 9, depth-probe 2,
  band-set 1, attention entropy 1, E_case 1, baseline 0.

### The fine-tuned AWF model, end to end (exp21; expert labels)
- Ai2 publishes the fine-tuned checkpoints. allenai/OlmoEarth-v1-FT-AWF-Base
  is a fully fine-tuned v1-Base encoder (203 of 231 tensors changed) with a
  1x1 convolution head, trained by rslearn from the AWF model.yaml. It was
  replicated without rslearn: encoder weights loaded strictly into
  olmoearth_pretrain's v1-Base encoder, tokens mean-pooled over timesteps
  and band sets as rslearn's wrapper does, legacy month-index timestamps,
  bilinear x4 upsampling before the 1x1 convolution. Run on the 344
  expert-labelled validation points of the official spatial split (the
  train split is the model's own training data).
- Accuracy 0.881 on 16-px crops (the training regime) and 0.878 on 32-px
  crops, against 0.895 reported by Ai2; reading the containing patch's
  logits instead of the bilinearly interpolated pixel gives 0.898. The
  frozen-encoder probe of exp16 reaches 0.817 on the same points; 28 of the
  fine-tuned model's 41 errors are also probe errors.
- Signal comparison on the fine-tuned model (16-px crops, 41 errors,
  tie-aware AURC, cluster bootstrap over the 30 annotation tasks):
  confidence 0.0262; aligned tiling instability 0.0235 (difference CI
  [-0.0068, +0.0010], P(better) 0.93, not significant); boundary indicator
  0.0765 (significantly worse); disagreement with the frozen probe 0.0852
  (worse); NDVI temporal-variability control 0.0937 (worse); oracle 0.0076,
  random 0.119. On 32-px crops confidence 0.0276 against tiling instability
  0.0286 (CI [-0.0022, +0.0035]). Error capture at a 20% review budget:
  confidence 0.63, tiling instability 0.71 (16 px); 0.69 against 0.64
  (32 px).
- How good, stated for a user: keeping the 80% most confident points
  raises accuracy from 0.881 to 0.945, the 90% most confident to 0.919.
  Expected calibration error 0.080 (10 bins): the 299 points with top-1
  probability above 0.9 are 0.93 accurate at a mean confidence of 0.99,
  and the 21 points in the 0.8-0.9 bin are 0.52 accurate, so the model is
  overconfident. Per-class recall: shrubland/savanna 0.96 (n 116),
  agriculture/settlement 0.91 (56), grassland/barren 0.82 (72), woodland
  forest 0.73 (45), open water 0.91 (11), montane forest 0.80 (10),
  herbaceous wetland 0.50 (6), urban 1.00 (27).
- Boundary share among errors 0.63 against 0.34 among correct points; a
  sub-patch shift flips the argmax at 5% of error points against 1% of
  correct ones. Values in exp/out/exp21_finetuned_awf.csv and
  exp21_summary.json; figure exp/out/exp21_finetuned_awf.png.

### Seasonal water: do the wins live on margins an annual map cannot represent? (exp25)
- Candidate after exp23 and exp24: a single-date image against an annual
  composite map disagrees on seasonal water margins, which are
  boundary-structured. JRC Global Surface Water (v1.3, 30 m; seasonality =
  months with water in 2020) was warped onto every scene grid; a 4-px patch
  is seasonal if any pixel holds water 1 to 11 months. Median seasonal
  share per scene 9% (2024 grids). Tests on the 2024 scenes (27) and
  the 2021 scenes of exp24 (26).
- T1 enrichment: seasonal patches are a median 39% of disagreements
  against 8% of agreements in 2024 (21/3/3 scenes, sign p 3e-04);
  41% against 9% in 2021 (22/3/1, p 2e-04). Seasonal margins
  carry a disproportionate share of the disagreements, as the hypothesis
  requires.
- T2 exclusion: with seasonal patches removed from scoring, tiling
  instability still beats confidence on 22/2/0 scenes in 2024 (sign p 4e-05;
  all patches on the same scenes 23/1/0) and 22/1/0 in 2021 (p 6e-06; all
  patches 20/3/0); the boundary indicator 19/5/0 against 18/6/0 (2024) and 20/3/0
  against 18/5/0 (2021). T3 on seasonal patches only (fewer scenes qualify):
  tiling instability 12/6/0 (2024) and 13/8/0 (2021).
- Conclusion: the advantage does not live on seasonal water margins. Three
  measurable components of reference-versus-image mismatch have now been
  removed one at a time (version instability, exp23; the year gap, exp24;
  seasonal water, exp25) and the WorldCover-referenced advantage of
  tiling instability over confidence survives each. What remains is
  either reference error shared by both versions and unrelated to
  seasonality, or a genuine property of the WorldCover-defined task that
  date-matched hand labels do not share. Values in
  exp/out/exp25_seasonal_water.csv; figure exp/out/exp25_seasonal_water.png.

### The year gap: does it explain the WorldCover wins? (exp24)
- The remaining candidate after exp23 was temporal mismatch: WorldCover
  2021 scored against June-September 2024 imagery. The same rule scenes
  were re-fetched with imagery from May-September 2021 (least cloudy item
  under 5% cloud per scene), WorldCover 2021 warped to each 2021 window's
  own grid, features computed exactly as in exp11 (v1-Base at shifts 0-3,
  v1-Nano), and the head retrained within the year on Katima 2021 (seed 0;
  both the 2021 and the 2024 head fit their training scene perfectly).
  26 scenes have at least 8 disagreements in both years; 2021 has more
  disagreements than 2024 on 20 of 27 scenes (total 2337 against 1845),
  consistent with single dates in the 2021 wet-to-dry transition against an
  annual map.
- Within the year, the advantage persists: tiling instability beats
  confidence on 23/3/0 scenes (sign p 9e-05; median E-AURC gain +0.0172)
  against 25/1/0 with 2024 imagery on the same scenes (gain +0.0047); the
  boundary indicator 20/6/0 against 18/8/0; cross-model disagreement 11/15/0 against 10/16/0;
  embedding distance 12/14/0 against 12/14/0; the control 12/14/0 against 13/13/0. The
  per-scene tiling-instability gain is not larger in 2024 (10/16/0, p 0.327).
- Conclusion: the year gap does not explain the WorldCover-referenced wins
  either. Two explanations remain untested: the mismatch between a
  single-date image and an annual composite map (seasonal water margins,
  which are boundary-structured and which an annual map cannot represent),
  and a genuine difference between the WorldCover-defined task and
  date-matched expert labels. Values in exp/out/exp24_year2021.csv; figure
  exp/out/exp24_year2021.png. Caveat: 2021 Level-2A products predate the
  2022 radiometric offset change; the 2021 head is trained and scored
  within that radiometry.

### Reference instability: does it explain the WorldCover wins? (exp23)
- Test of the exp18 reading with a measurable component of reference
  error: patches where ESA WorldCover 2020 (v100) and 2021 (v200) disagree
  about water, on 24 of the 27 rule scenes (three Kafue scenes dropped
  because the re-read image no longer matches the cache). Signals recomputed
  as in exp13 from cached features; three pre-specified tests.
- T1 enrichment: version-unstable patches are a median 10.8% of the
  head's disagreements against 0.8% of its agreements (17/3/4 scenes,
  sign p 0.003); pooled, 10% of disagreements sit on unstable patches.
  Reference instability is real but small.
- T2 mechanism: among disagreements, the top-k set ranked by tiling
  instability is no more often reference-unstable than the top-k set
  ranked by confidence (median difference +0.00; 8/8/8; p 1.000).
- T3 decisive: with unstable patches removed from scoring (23 scenes with
  at least 8 remaining disagreements), tiling instability still beats
  confidence on 21 of 23 scenes (sign p 7e-05; median E-AURC gain
  +0.0026) against 22 of 23 on all patches of the same scenes
  (gain +0.0031); the boundary indicator 18/23 against 15/23. The
  advantage is unchanged.
- Conclusion: the reading that the WorldCover-referenced wins were
  detection of reference error is not supported by this component. It
  cannot rule out reference errors shared by both versions or genuine
  change between 2021 and the 2024 imagery; the year gap was tested next
  (exp24) and does not explain the wins either. Values in
  exp/out/exp23_reference_instability.csv; figure
  exp/out/exp23_reference_instability.png.

### Periodic artifacts in the served product (exp22; label-free)
- Question: does the served v1.2 product carry striping or seams at the
  scales its pipeline imposes? 5 windows of 4096 served px (about 37 km)
  from 3 served tiles were read; column and row profiles of the
  class-map boundary indicator (band 4), of the change-probability gradient
  (band 1), and of ESA WorldCover 2021 warped to the same grid (a control
  that shares no model grid) were tested for periodicity with a whitened
  periodogram. The rasters are the Web Mercator zoom-14 grid warped from
  the 10 m UTM inference grid, so every pipeline period is predicted from
  each window's own geometry (an encoder patch of 4 UTM px maps to
  4.34 to 4.42 served px across the windows).
- Encoder patch lattice: the largest peak of every product profile but one
  (19 of 20) lies on the patch lattice (1, 2 or 4 patches, or
  the third harmonic of the 4-patch period), with Bonferroni p at most
  6e-08; the observed periods track each window's UTM-to-Mercator ratio,
  which places the lattice in the UTM inference grid. The WorldCover
  control's largest peak never reaches p = 0.006. Class boundaries and
  change-probability gradients are therefore quantized to the 40 m patch
  grid. The one off-lattice top peak: kwando change_gradient row at 7.55 px (1.72 patches).
- Inference-window seams (64, 128, 256, 512 UTM px), judged on the
  fundamental ordinate because harmonic comb scores are confounded by the
  lattice: profiles with p < 0.01, class map 64 px: 0 of 10, 128 px: 0 of 10, 256 px: 0 of 10, 512 px: 0 of 10;
  change gradient 64 px: 0 of 10, 128 px: 1 of 10, 256 px: 0 of 10, 512 px: 0 of 10; control
  64 px: 0 of 10, 128 px: 0 of 10, 256 px: 1 of 10, 512 px: 0 of 10. This is the rate expected under the null.
  Seams injected into each real class map along the sheared UTM grid set
  the detection limit: at the 128-px period, seams affecting 5% in 3, 10% in 2 windows'
  rows would have been detected; at 256 px, 10% in 2, 20% in 3.
- Warp duplication beat (nearest-neighbour warping repeats one source
  column every ratio/(ratio - 1) served px): fundamental p < 0.001 in
  2 of 10 class-map and 2 of 10 gradient profiles, 0 of 10 in the control.
- Method validation before use, on lattice-free synthetic maps: scan false
  positives 0 of 10, confirmatory 1 of 10 (three hypotheses each); seams on
  5% of rows detected at sparse boundaries (p 9e-7) and on 10% at dense
  ones (p 5e-9); shear correction about triples power; the beat period was
  first mis-predicted as 1/(ratio - 1) and corrected after the scan located
  its second harmonic. A first synthetic generator built by 8x upsampling
  carried a lattice of its own and was discarded. Values in
  exp/out/exp22_lcc_striping.csv, exp22_confirmatory.csv, exp22_power.csv;
  figure exp/out/exp22_lcc_striping.png.

### Served production output: land cover change rasters (exp20; weak reference)
- First assessment of one of Ai2's own outputs: ten 512-px windows (about
  4.9 km) of the published allenai/olmoearth_lcc rasters at Zambezi, Chobe
  and Barotse sites, read with the pure-HTTP tile reader in
  oe_inferencex/lcc.py. The product ships in EPSG:3857 at about 9.55 m as
  9-band uint8 BigTIFFs; per the dataset card, band 1 is the change
  probability, bands 4-5 the source and destination land cover classes, and
  bands 6-7 the probabilities of the change-category heads (bands 2-3).
- The product exports no confidence for the land cover classes, so the
  recipe's primary signal (the model's own confidence) cannot be run on the
  class map. Bands 6-7 are not class confidences: they sit at 255 on 80 to
  99% of all pixels because the category head answers "none" at unchanged
  pixels; among flagged pixels they take 48 to 196 distinct values.
- Boundary triage on the product's water map (band 4 water against ESA
  WorldCover 2021 water, the one class whose legends coincide): six of ten
  sites contain water; disagreement is 0.2 to 2.4% of 4-px windows; the
  boundary fraction ranks disagreements below random on 6 of 6 (Kazungula
  AURC 0.0013 against 0.0156 random, oracle 0.0001); a 5% review budget by
  boundary captures a median 0.88 of disagreements (0.61 to 0.98); the
  boundary share is 0.92 among disagreements (0.69 to 0.99) against 0.01
  among agreements (0.005 to 0.08). Legends and dates differ, so these are
  reference disagreements, not counted model errors.
- Full-legend disagreement with WorldCover is a median 49% (5 to 68%),
  dominated by tree, shrub and grass and by built-up; the legends' semantics
  differ (WorldCover's built-up covers whole towns, its tree threshold is
  lower), so this number is context, not an error rate.
- Change probability (band 1): a median 2.7% of pixels flagged at 0.5 (0.01
  to 13.7%); all 256 values used; 60 to 95% of pixels at exactly 0; the
  ambiguous band 0.25 to 0.75 holds a median 1.2% (0.08 to 4.7%). Low
  confidence (|2p - 1| < 0.5) sits on flagged-region edges: a median 2.7% of
  edge windows against 0.07% of interior windows. The boundary finding
  replicates label-free on the product.
- Where change is flagged, the predicted transitions are plausible for the
  region (tree to grassland or built-up at Kazungula and Katima, grassland to
  crops at Barotse, water to wetland at Linyanti); the pre-category head says
  deforestation on 4 to 72% of flagged pixels and the post head new_building
  or new_crop_field. No change reference exists here, so these are sanity
  checks, not accuracies. Values in exp/out/exp20_lcc_production.csv; figure
  exp/out/exp20_lcc_kazungula.png.

### OlmoEarth v1 vs v1.2 (exp19; WorldCover reference)
- Run in an isolated environment on the current olmoearth_pretrain main,
  which loads both versions; v1 features recomputed with the new code match
  the cached exp11 features exactly (max difference 0). v1.2-Base uses
  rotary position encodings (rope_3d_mixed) and, unlike v1, tokenizes
  Sentinel-2 as a single band-set token per patch, so band-set
  disagreement does not exist for it.
- Tiling instability does not shrink under RoPE: mean per-patch std across
  0-3 px shifts is 0.046 for v1.2 vs 0.032 for v1, smaller for v1.2 on
  only 6 of 31 scenes (sign p=9e-4). RoPE addressed the long-range
  striping artifact; sub-patch grid-shift instability is a different
  effect and is larger in v1.2 on this probe.
- Head accuracy vs WorldCover on the Katima probe: v1 0.942, v1.2 0.922.
- Against WorldCover, tile-phase ranks each version's own errors better
  than its confidence for both (v1 26/1, v1.2 25/2), but exp18 shows this
  WorldCover-referenced advantage does not transfer to hand labels (its
  cause is open, exp23); no expert-label
  test of v1.2 has been run. Cross-version disagreement |p_v1.2 - p_v1| is
  worse than confidence for v1's errors (6/21) and not significant for
  v1.2's (18/9, p=0.12). Values in exp/out/exp19_v1_vs_v12.csv.

### Pre-registered scene set (exp11; statistics superseded by exp13)
- The scene rule and scene set below stand. The statistics in this section
  use raw AURC and the unaligned tile-phase; exp13 corrects both and is
  authoritative where they differ.
- Scene rule committed to git before any new scene was fetched: candidates
  sampled at fixed fractions along OSM geometries of eight named rivers,
  0.2-degree separation, inclusion iff the deterministic Base head commits
  >=8 errors against WorldCover. 20 rule-selected scenes joined the 7 exp09 scenes plus 2 unsuffixed exp09
  first-attempt AOIs (kafue, luangwa) that entered through the cache import
  and are excluded from exp13 onward; 29 total here, 27 in exp13-exp15. Per-scene AURC, bootstrap 95% CIs (B=1000), and
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

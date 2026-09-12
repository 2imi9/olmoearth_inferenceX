# Cross-signal comparisons

One section per experiment, grouped by what the section is evidence for.
All signals within a section are scored on identical errors with the same
harness. Scoring rules, evidence tiers and status terms are defined in
[../method/protocol.md](../method/protocol.md); per-signal summaries are in
[signals.md](signals.md). Index at [../TECHNIQUES.md](../TECHNIQUES.md).

Superseded runs (exp05, exp09, exp11's statistics) are not reproduced here.
They are in the lab log, [../../exp/NOTES.md](../../exp/NOTES.md).

---

# 1. Expert labels (authoritative)

These four experiments score signals against human-labelled truth. Where
they conflict with the WorldCover-referenced results in section 2, these
stand.

## Dense flood masks: Sen1Floods11 (exp18)

**Setup.** Sen1Floods11 hand-labelled flood-water masks (public mirror),
64x64 tiles. Head trained on 600 tiles of
the valid split; scored on Bolivia (441 tiles, a geographically held-out
region; 351 with enough errors to score) and on the test split (800 tiles,
482 scored). All signals
computed on one 60x60 crop per tile, tie-aware excess AURC per tile, exact
sign tests. Values in `exp/out/exp18_sen1floods.csv`.

**Result: confidence is the best signal**, on both splits. <!-- claim:exp18-confidence-best-both-splits -->

| Signal vs confidence | Bolivia (better/worse) | p | Test split |
|---|---|---|---|
| Aligned tile-phase | 163 / 187 | 0.22 | 173 / 305 (p=2e-9) |
| Band-set disagreement | 111 / 239 | 7e-12 | 148 / 334 |
| Boundary indicator | 134 / 217 | 1e-5 | 154 / 327 |
| Cross-model (E_case) | 84 / 265 | 6e-23 | 141 / 340 |
| Embedding distance (E_dist) | 22 / 329 | - | - |
| Pixel control | 79 / 271 | - | - |

Pooled E-AURC over all patches: confidence 0.0105 (Bolivia), 0.0096 (test)
— lowest of all signals on both.

**Errors still concentrate on boundaries** (75% of error patches vs 21% of
correct on Bolivia; 73% vs 18% on test). The phenomenon is real; confidence <!-- claim:errors-sit-on-boundaries -->
simply ranks those errors better than boundary proximity or instability do.

**Reading.** The WorldCover-referenced advantages of tile-phase (26/27) and
band-set disagreement (21/27) do not transfer to expert truth. Until a
signal beats confidence on expert-labelled dense maps, this repository's
positive claims are claims about detecting disagreement with a weak
reference, not about detecting model error. Why they do not transfer is
open — see section 3.

**Caveats.** Level-1C inputs through an L2A path (documented mismatch; head
accuracy 0.912 Bolivia, 0.953 test); flood water rather than permanent
water; one crop per tile; the head trains on tiles from the same eleven
regions as the test split, so Bolivia is the clean spatial hold-out. None
of these plausibly favour confidence over the other signals.

## The fine-tuned AWF model, end to end (exp21)

**Setup.** `allenai/OlmoEarth-v1-FT-AWF-Base` is a fully fine-tuned v1-Base
encoder (203 of 231 tensors changed) with a 1x1 convolution head, trained by
rslearn from the AWF `model.yaml`. Replicated without rslearn: encoder
weights loaded strictly into olmoearth_pretrain's v1-Base, tokens mean-pooled
over timesteps and band sets, legacy month-index timestamps, bilinear x4
upsampling before the 1x1 convolution. Run on the 344 expert-labelled
validation points of the official spatial split.

**Accuracy.** 0.881 on 16-px crops (the training regime), 0.878 on 32-px,
against the reported 0.895; 0.898 when the containing patch's logits are
read directly instead of the interpolated pixel. The frozen-encoder probe of
exp16 reaches 0.817 on the same points; 28 of the fine-tuned model's 41
errors are also probe errors. <!-- claim:fine-tuned-vs-probe-errors -->

**Signal comparison** (16-px crops, 41 errors, tie-aware AURC, cluster
bootstrap over the 30 annotation tasks):

| Signal | AURC | Verdict |
|---|---|---|
| Aligned tiling instability | 0.0235 | indistinguishable from confidence (CI [-0.0068, +0.0010], P(better) 0.93) |
| **Confidence** | **0.0262** | best supported | <!-- claim:fine-tuned-model-audit -->
| Boundary indicator | 0.0765 | significantly worse |
| Disagreement with the frozen probe | 0.0852 | worse |
| NDVI temporal-variability control | 0.0937 | worse |
| *oracle / random* | *0.0076 / 0.119* | *bounds* |

On 32-px crops: confidence 0.0276 against tiling instability 0.0286 (CI
[-0.0022, +0.0035]). Error capture at a 20% review budget: confidence 0.63,
tiling instability 0.71 (16 px); 0.69 against 0.64 (32 px).

**How good, stated for a user.** Keeping the 80% most confident points
raises accuracy from 0.881 to 0.945; the 90% most confident to 0.919.
Expected calibration error 0.080 (10 bins): the 299 points above 0.9
top-1 probability are 0.93 accurate at a mean confidence of 0.99, and the 21
points in the 0.8-0.9 bin are 0.52 accurate. **The model is overconfident,
so a stated accuracy needs a coverage.** <!-- claim:accuracy-needs-coverage -->

Per-class recall: shrubland/savanna 0.96 (n 116), agriculture/settlement
0.91 (56), grassland/barren 0.82 (72), woodland forest 0.73 (45), open water
0.91 (11), montane forest 0.80 (10), herbaceous wetland 0.50 (6), urban 1.00
(27). Boundary share among errors 0.63 against 0.34 among correct points; a
sub-patch shift flips the argmax at 5% of error points against 1% of correct
ones. Values in `exp/out/exp21_finetuned_awf.csv`, `exp21_summary.json`.

## Boundary proximity on the AWF point task (exp16)

**Setup.** The Base head applied densely to all 64 patches of each
validation crop, with the exp14 boundary indicator computed at the labelled
patch; errors reproduce exp04 exactly (63/344). Adversarially reviewed
before recording (`exp/out/review_exp16_wf_5d95d304.json`); the score
derives from the head's own prediction map, not
ground-truth boundaries, and uncertainty uses a cluster bootstrap over the
30 annotation tasks.

- **Ranking.** Confidence (negative logit margin) ranks errors better than
  the boundary score (AURC 0.0363 vs 0.0636; cluster-bootstrap 95% interval
  on the difference [+0.0023, +0.0562], P(boundary better) = 0.016) and than
  per-window tile-phase (0.0489, interval [+0.0023, +0.0221]). <!-- claim:awf-confidence-lowest-aurc -->
- **Are labelled patches interior?** No. The labelled patch's score is zero
  on 47% of windows against 43% for other patches of the same maps; its
  within-window quantile averages 0.46. Labelled patches are, if anything,
  slightly *less* boundary-like than an arbitrary patch.
- **Does the score carry error information?** Marginally, strongly: 90% of
  errors have a nonzero score against 44% of correct windows (Fisher p=2e-12).
  Conditionally, little: it correlates with the margin (Spearman 0.60), and
  in a logistic model of error on both, the margin dominates (standardized
  coefficients 3.53 vs 0.52).

**Reading.** On a nine-class task the argmax flips between neighbouring
patches wherever margins are small, so the boundary score is largely a
coarse proxy for low confidence, which the margin already carries at finer
resolution. *Caveat: the AWF split is by point, not by task, so every
validation task also contributes training windows.* Values in
`exp/out/exp16_awf_boundary.csv`, `exp16_summary.json`.

## AWF expert-label validation (exp04, exp12)

Source: the
[olmoearth_projects_awf dataset](https://huggingface.co/datasets/allenai/olmoearth_projects_awf),
classes and split from its
[task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml).

1459 expert-labeled points, 12-month Sentinel-2 stacks, the project's own
1115/344 spatial split. A linear head on frozen Base embeddings reaches
81.7% validation accuracy (the fully fine-tuned model: 89.5%), giving 63
errors for signal evaluation.

On this in-domain multiclass task **confidence achieves the lowest AURC**
(0.0363), against tile-phase 0.0489, Nano-Base total variation 0.0670, and —
completing the comparison on the same 63 errors (exp12) — E_dist 0.1338 and a
no-model spectral-variability control 0.1658. In-domain AWF errors are <!-- claim:awf-confidence-lowest-aurc -->
neither out-of-distribution nor pixel-trivial.

Lowest per-class recall: herbaceous wetland (0.50, n=6, indicative only).

![AWF risk-coverage and per-class recall](../../exp/out/exp04_awf_expert.png)

---

# 2. The WorldCover reference: the 27-scene study

The largest study in the repository, and the source of every result that
section 1 fails to confirm. **Read these as claims about disagreement with a
weak reference.**

## The scene rule (exp11)

Committed to git before any new scene was fetched: candidates sampled at
fixed fractions along OSM geometries of eight named rivers, 0.2-degree
separation, included iff the deterministic Base head commits >= 8 errors
against WorldCover. 20 rule-selected scenes joined the 7 exp09 scenes; two
unsuffixed AOIs (kafue, luangwa) that entered through a cache import rather
than the rule are excluded from exp13 onward, leaving **27 scenes**. <!-- claim:rule-selected-27-scenes -->

Head-seed variance is structurally zero: heads initialize at zeros with
deterministic full-batch training, so the planned seed-robustness test is
vacuous rather than passed. Robustness to head initialization is untested by
design choice.

*The exp11 statistics used raw AURC and the unaligned tile-phase; exp13
corrects both and supersedes them.* <!-- claim:exp11-unaligned-tile-phase-stats -->

## Corrected statistics (exp13)

Same errors and heads as exp11, recomputed under the scoring rules in
[../method/protocol.md](../method/protocol.md). Per-scene values in
`exp/out/exp13_corrected_stats.csv`; the win/loss counts, sign and
permutation p-values and best-signal tally below are in
`exp/out/exp13_summary.json`.

**Aligned tile-phase beats confidence on 26/27 scenes** (1 worse, 0 tied;
exact sign test p=4e-07; sign-flip permutation p=1e-04). Its block-bootstrap
interval excludes zero in its favour on 18 scenes and against on 0. It beats
the pixel control on 18/27, E_case on 23/27, E_dist on 22/27, and is the best
of the five signals on 12/27 (best or second on 24/27). <!-- claim:tile-phase-26-of-27-worldcover -->

Everything else against the baseline:

| Signal | Wins | Sign p | Note |
|---|---|---|---|
| Aligned tile-phase | 26/27 | 4e-07 | best signal on 12/27 |
| Control (pixel statistic) | 14/27 | - | best on 9/27, the reference-omission scenes |
| E_dist | 13/27 | 1.00 | mean-based permutation p=0.007 is carried by high-error scenes |
| E_case | 10/27 | 0.25 | intervals favour it on 3 scenes, against on 9 |
| Confidence (baseline) | - | - | best on 0/27 |

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

## What tile-phase actually measures (exp14)

**Question.** Is aligned tile-phase a perturbation signal, or a detector of
boundaries in the model's own prediction map?

Two zero-cost proxies computed from the shift-0 map alone: gradient
magnitude of the probability map, and the fraction of a patch's 8 neighbours
whose hard label differs.

**The discrete boundary fraction is statistically indistinguishable from
tile-phase**: boundary better on 12 scenes, tile-phase on 15, tied on 0
(sign p=0.70; median E-AURC gap 0.00012 in tile-phase's favour). Per-scene <!-- claim:boundary-fraction-equals-tile-phase -->
values differ in
both directions, so this is a null result, not an equivalence. Boundary
fraction alone beats the baseline on 19/27 (p=0.052) and the pixel control
on 22/27 (p=0.002); the continuous gradient is weaker (tile-phase better on
25/27). Best-signal tally: pred-boundary 10, tile-phase 10, control 5,
baseline 2.

**Conclusion.** No advantage of the perturbation beyond boundary proximity
is detectable. Errors concentrate at prediction boundaries and aligned
tile-phase ranks them better than confidence *against this reference*; the
zero-inference boundary shortcut is suggestive but its own margin over
confidence is marginal (p=0.05). Values in
`exp/out/exp14_boundary_ablation.csv`.

---

# 3. Why the WorldCover advantage does not transfer

Tiling instability wins 26/27 against WorldCover (exp13) and loses against
hand labels (exp18). The first explanation offered — that the counted
"errors" were largely *reference* errors, which boundary-type signals detect
— was tested three times against the measurable components of
reference-versus-image mismatch. **All three tests are negative.**

Each experiment uses the same three-part design: T1 asks whether the
suspect patches are enriched among disagreements, T2 whether the signal's
flagged errors sit there preferentially, T3 whether the advantage survives
removing them.

## Reference instability: WorldCover 2020 vs 2021 (exp23)

24 of the 27 rule scenes (three Kafue scenes dropped: the re-read image no
longer matches the cache).

- **T1.** Version-unstable patches are a median 10.8% of the head's
  disagreements against 0.8% of its agreements (17/3/4 scenes, sign p=0.003).
  Pooled, 10% of disagreements sit on unstable patches. **Real but small.**
- **T2.** Among disagreements, the top-k set ranked by tiling instability is
  no more often reference-unstable than the top-k ranked by confidence
  (median difference +0.00; 8/8/8; p=1.000).
- **T3.** With unstable patches removed (23 scenes with >= 8 remaining
  disagreements), **tiling instability still beats confidence on 21/23**
  (p=7e-05, median E-AURC gain +0.0026) against 22/23 on all patches of the
  same scenes. Boundary indicator 18/23 against 15/23. <!-- claim:reference-instability-does-not-explain -->

Values in `exp/out/exp23_reference_instability.csv`.

## The year gap: 2021 map vs 2024 imagery (exp24)

The rule scenes re-fetched with May-September 2021 imagery (least cloudy
item under 5% cloud), WorldCover 2021 warped to each 2021 window's own grid,
features computed exactly as in exp11, and the head retrained within the year
on Katima 2021. 26 scenes have >= 8 disagreements in both years; 2021 has
more disagreements than 2024 on 20 of 27 scenes (2337 against 1845),
consistent with single dates in the 2021 wet-to-dry transition against an
annual map.

**Within the year the advantage persists:** tiling instability 23/3/0 (sign
p=9e-05, median gain +0.0172) against 25/1/0 with 2024 imagery on the same
scenes (gain +0.0047); boundary 20/6/0 against 18/8/0; E_case 11/15/0 against
10/16/0; E_dist 12/14/0 either way; control 12/14/0 against 13/13/0. The
per-scene gain is **not** larger in 2024 (10/16/0, p=0.327). <!-- claim:year-gap-does-not-explain -->

*Caveat: 2021 Level-2A products predate the 2022 radiometric offset change;
the 2021 head is trained and scored within that radiometry.* Values in
`exp/out/exp24_year2021.csv`.

## Seasonal water (exp25)

JRC Global Surface Water (v1.3, 30 m; seasonality = months with water in
2020) warped onto every scene grid; a 4-px patch is seasonal if any pixel
holds water 1 to 11 months. Median seasonal share per scene 9% (2024 grids).
Tested on the 27 scenes of 2024 and the 26 of exp24.

- **T1.** Seasonal patches are a median 39% of disagreements against 8% of
  agreements in 2024 (21/3/3, p=3e-04); 41% against 9% in 2021 (22/3/1,
  p=2e-04). **The hypothesis's premise holds.**
- **T3.** With seasonal patches removed, **tiling instability still beats
  confidence 22/2/0 in 2024** (p=4e-05; all patches 23/1/0) and 22/1/0 in
  2021 (p=6e-06; all patches 20/3/0). Boundary 19/5/0 against 18/6/0 (2024),
  20/3/0 against 18/5/0 (2021). T3 restricted *to* seasonal patches: tiling
  instability 12/6/0 (2024), 13/8/0 (2021). <!-- claim:seasonal-water-does-not-explain -->

Values in `exp/out/exp25_seasonal_water.csv`.

## Where this leaves it

Three measurable components of reference-versus-image mismatch have been
removed one at a time and the advantage survives each. **What remains is
either reference error shared by both WorldCover versions and unrelated to
seasonality, or a genuine property of the WorldCover-defined task that
date-matched hand labels do not share.** The next step is a human
adjudication of individual disagreements; exp26 prepared the kit.

---

# 4. Deployed products

## Operating points at fixed review budgets (exp35)

The question of roadmap item 3: does a signal help confidence at a fixed
review budget where it does not on AURC? Metric: expected error capture
inside the 5, 10 or 20% most suspect windows under random tie-breaking.
Preregistered test: tiling instability against confidence at 20% on
Sen1Floods11 Bolivia hand labels, per-tile one-sided sign test.

| Testbed | Result |
|---|---|
| Bolivia hand labels, 351 tiles | Preregistered test null: tiling instability 116/112/123 (p = 0.42), pooled 0.707 vs 0.749. Boundary indicator beats confidence at 5% (0.286 vs 0.259, CI [+0.009, +0.046]; 181/147/23, p = 0.068), loses at 20%. The NDWI-level control beats confidence pooled at 20% (0.795 vs 0.749), a caution on single operating points | <!-- claim:operating-points-null -->
| Fine-tuned AWF model, 30 tasks | 16-px crops: tiling instability +0.073 at 20% (CI [+0.000, +0.190], P 0.93), the exp21 hint; 32-px crops: reversed (0.643 vs 0.690). Every other signal behind at every budget |
| WorldCover, 27 scenes | Tiling instability 8/0 rivers at 10% and 20%; boundary 7/1 at every budget where its AURC vote is 5/3: coarse scores fare better at fixed budgets than under AURC |

**Reading.** The ranking metric and the reviewer's workflow differ in
detail, not in verdict: on expert labels no signal beats confidence at a
fixed budget with preregistered support. Details in
`exp/out/exp35_summary.json`.

## Boundary first, then confidence, at fixed budgets (exp36)

The review order that exp35 suggested, tested where exp35 stopped. Capture
of the errors inside the most suspect fraction of the windows, the
boundary-first order against confidence:

| Testbed | 5% | 10% | 20% |
|---|---|---|---|
| Sen1Floods11 Bolivia, pooled over 351 tiles | 0.274 vs 0.259, CI [+0.009, +0.021] | 0.494 vs 0.465, CI [+0.015, +0.039] | 0.732 vs 0.749, CI [-0.054, +0.013] | <!-- claim:boundary-first-review-order -->
| Bolivia per tile, better/worse/tied | 85/31/235, one-sided p = 2.7e-7 | 112/48/191, p = 2.3e-7 | 121/77/153, two-sided p = 0.002 |
| AWF fine-tuned, 16-px crop, 30-task bootstrap | 0.220 vs 0.220 | 0.390 vs 0.390 | 0.634 vs 0.634 | <!-- claim:fine-tuned-capture-at-budgets -->
| AWF fine-tuned, 32-px crop | 0.214 vs 0.214 | 0.381 vs 0.429 | 0.571 vs 0.690 |
| WorldCover scenes, rivers better/worse | 8/0 | 8/0 | 8/0 |

The 5% and 10% Bolivia rows were preregistered (one-sided per-tile sign
tests, tile bootstrap of the pooled gain); the rest is descriptive. On the
AWF model the two orders pick the same 5% review set on both crops, so the
bootstrap interval's lower bound is exactly zero. Source
`exp/out/exp36_summary.json`.

## Spectral ambiguity first, then boundary, then confidence (exp38)

The order exp37's enrichment suggested, tested at fixed budgets on exp37's
per-window tables (CPU; the tables reproduce exp36's counts exactly).
Capture of the errors, NDWI-first order against the boundary-first order of
exp36 and against confidence:

| Comparison | 5% | 10% | 20% |
|---|---|---|---|
| vs boundary-first, Bolivia per tile (better/worse/tied) | 151/100/100, one-sided p = 7.7e-4 | 146/86/119, p = 4.9e-5 | 119/54/178 |
| vs boundary-first, Bolivia pooled | 0.292 vs 0.274, CI [-0.002, +0.037] | 0.481 vs 0.494, CI [-0.053, +0.033] | 0.831 vs 0.732, CI [+0.071, +0.130] |
| vs confidence, Bolivia pooled | 0.292 vs 0.259, CI [+0.014, +0.052] | 0.481 vs 0.465, CI [-0.021, +0.057] | 0.831 vs 0.749, CI [+0.055, +0.107] |
| vs boundary-first, WorldCover rivers (better/worse) | 2/6 | 3/5 | 3/4 |
| vs confidence, WorldCover rivers | 4/4 | 4/4 | 6/1 |

The 5% and 10% rows against the boundary-first order were preregistered
(per-tile one-sided tests and the tile bootstrap of the pooled gain); the
per-tile tests pass, the pooled bootstrap does not, so the result is mixed. <!-- claim:ndwi-first-order-mixed -->
The two statistics are two operating modes: a budget per tile against one
budget over the area, where the ambiguous windows of NDWI-heavy tiles fill
the set. Source `exp/out/exp38_summary.json`.

## Window design at pixel level (exp42)

Four alternatives to the fixed 4-px grid window, all from the cached
features at four crop offsets and the same head, judged on pixel-level hand
labels over the region every tiling covers:

| Design | Bolivia pixel accuracy | Test split pixel accuracy | Per tile vs the grid |
|---|---|---|---|
| W0 grid window | 0.8969 | 0.9411 | |
| W1 shift-averaged (preregistered) | 0.9071 | 0.9503 | 321/66/53, p = 1e-41; 583/78/139, p = 9e-97 | <!-- claim:w1-accuracy-gain -->
| W3 spectral split inside the window | 0.9084 | 0.9483 | vs W1: 217/158 better on Bolivia, 226/404 worse on test | <!-- claim:window-design-alternatives-mixed -->
| W2 scale-adaptive (window accuracy) | 0.9045 vs 0.9020 | 0.9515 vs 0.9518 | 175/121; 253/241 |
| segment majority of W1 over a spectral partition (exp43, preregistered) | 0.9042 | 0.9478 | vs W1: 168/209 (C 15,093, B 16,273); 244/387 (C 24,525, B 30,951) | <!-- claim:segment-majority-rejected -->
| sixteen crop offsets (exp44, preregistered) | 0.9074 | 0.9507 | vs W1: 201/149/90, p = 0.003 (C 4,566, B 4,237); 369/202/229, p = 1.3e-12 (C 7,100, B 6,060); +0.04 points, below the +0.2 minimum effect | <!-- claim:sixteen-offsets-not-worthwhile -->
| eight offsets, diagonals and anti-diagonals (exp44) | 0.9073 | 0.9505 | half of W16's gain |
| seven offsets, diagonals and horizontal phases (exp44) | 0.9060 | 0.9487 | worse than the four diagonals on both |

Mixed-label windows are 10% of the windows and carry 45% and 58% of the
grid window's errors (error rate 0.39 and 0.28 against 0.054 and 0.022 on
pure windows); that is error concentration, not a bound. The
block-constant oracle limit, the pixels any one-class-per-block decision
must get wrong, is 3.0% and 2.9% pooled, against grid error rates of 10.3%
and 5.9% averaged over tiles, so the block forces at most 29% and 49% of
those errors, and the shift-averaged map decides per pixel and still puts
43% and 56% of its own errors on those windows (exp43). <!-- claim:mixed-label-windows-not-a-ceiling -->
Tile-phase abstention equals confidence abstention at matched coverage.
Source `exp/out/exp42_summary.json`.

## OlmoEarth v1.2 Base, the encoder the served product uses (exp45)

The four supported findings repeated on v1.2 with the same tiles, head
protocol and offsets. Accuracy is unchanged (0.9062 against v1's 0.9116 on
Bolivia; 0.9534 against 0.9528 on the test split); the ranking quality of the
model's own confidence is not.

| Pooled E-AURC on Bolivia | v1 | v1.2 |
|---|---|---|
| confidence | 0.0105 | 0.0146 |
| tiling instability | 0.0115 | 0.0138 |
| NDWI level (no model) | 0.0119 | 0.0119 |

Under v1.2 confidence is third on this testbed, behind a model signal and a
pixel control; on the test split it still leads everything (0.0059). The cue
enrichments replicate on both testbeds, and the shift-averaged window
replicates with a slightly larger gain than under v1 (+0.0110 and +0.0096
against +0.0102 and +0.0091), consistent with v1.2's larger tiling
instability. The boundary-first review order replicates on Bolivia (pooled
0.4611 against 0.4234 at the 10% budget) and does not extend to the test
split. Source `exp/out/exp45_summary.json`. <!-- claim:v12-replication-mixed -->

## SHRUG-FM's reliability signals under the window protocol (exp49)

SHRUG-FM (Gonzalez-Calabuig et al. 2026, best paper at the CVPR 2026
EarthVision workshop) ranks images for abstention by three signal families,
ensemble mutual information and entropy, k-means distance and NCDD in
embedding space, and input percentile extremity, fused by a label-fitted
decision tree; it never compares a single model's own confidence. exp49 ports
the three families and the fusion step into exp47's protocol: every ranker
graded on the shift-averaged decision's own errors, both backbones, both
testbeds, with the substitutions stated in the script (the head's training
split stands in for pretraining data, an ensemble of eight tile-bootstrap
linear heads for their CNN ensemble, a logistic combiner for the tree).

| Pooled E-AURC, lower is better | v1 Bolivia | v1 test | v1.2 Bolivia | v1.2 test |
|---|---|---|---|---|
| averaged confidence (reference) | 0.0094 | 0.0109 | 0.0116 | 0.0064 |
| ensemble predictive entropy | 0.0076 | 0.0107 | 0.0115 | 0.0066 |
| ensemble average entropy | 0.0079 | 0.0108 | 0.0117 | 0.0066 |
| ensemble mutual information | 0.0120 | 0.0107 | 0.0176 | 0.0086 |
| NDWI level (no model) | 0.0113 | 0.0106 | 0.0097 | 0.0106 |
| embedding normalized distance | 0.0657 | 0.0347 | 0.0768 | 0.0343 |
| embedding NCDD (eq. 5, cluster-normalized) | 0.0859 | 0.0407 | 0.0888 | 0.0413 |
| embedding NCDD raw | 0.0700 | 0.0295 | 0.0655 | 0.0339 |
| input extremity max / mean | 0.0484 / 0.0364 | 0.0349 / 0.0328 | 0.0538 / 0.0418 | 0.0339 / 0.0319 |
| fusion linear, label-fitted | 0.0067 | 0.0068 | 0.0089 | 0.0060 |
| fusion poly2, label-fitted | 0.0082 | 0.0102 | 0.0101 | 0.0084 |

Preregistered and supported: under v1.2 confidence beats mutual information
(leads +0.0060 and +0.0022, per tile 263/61 and 352/76, p = 2.5e-31 and
7.9e-44) and beats -NCDD (+0.077 and +0.035) on both testbeds. Three things <!-- claim:shrug-signals-rejected -->
beside the preregistration. Bagging the head appeared to help under v1: the
predictive entropy of the eight members' averaged maps beat confidence on
Bolivia (210/103 tiles, p = 1.5e-9) and marginally on the test split
(240/174, pooled -0.0002), and tied or lost under v1.2; exp50 below shows
this does not replicate with a fresh draw. <!-- claim:bag-beats-confidence-v1-bolivia -->
The label-fitted linear fusion beats confidence on three arms by
0.0027-0.0041 (223/94, 265/155, 223/97; p <= 9e-8) and by 0.0004 on the v1.2
test split, the first fusion to beat confidence in this repository, where
exp47's label-free midrank fusions all lost; it is fitted on the head's own
training split against in-sample errors, so it needs no labels beyond the <!-- claim:label-fitted-fusion-three-of-four -->
head's. And the paper's NCDD, computed over cluster-normalized distances as
its eq. 5 states, is worse than the raw deficit on all four arms, as the
normalization predicts when cluster spreads differ.

Their granularity. Averaging each window signal over the tile's
predicted-water windows and calling a tile a failure when its water F1 is
below 0.6 (their definition) gives failure rates of 0.331 on Bolivia and
0.220 / 0.242 on the test split, close to the paper's 0.33 and 0.21:

| Tile-level AURC (Risk@0.5) | v1 Bolivia | v1 test | v1.2 Bolivia | v1.2 test |
|---|---|---|---|---|
| averaged confidence | 0.167 (0.110) | 0.120 (0.105) | 0.140 (0.087) | 0.134 (0.130) |
| ensemble predictive entropy | 0.163 (0.105) | 0.118 (0.105) | 0.137 (0.073) | 0.135 (0.130) |
| ensemble mutual information | 0.226 (0.201) | 0.117 (0.105) | 0.172 (0.155) | 0.142 (0.130) |
| embedding NCDD | 0.281 (0.338) | 0.209 (0.205) | 0.463 (0.489) | 0.284 (0.252) |
| fusion linear, label-fitted | 0.175 (0.105) | 0.138 (0.100) | 0.146 (0.078) | 0.163 (0.127) |

At the tile level the single model's confidence, the baseline the paper never
runs, is at or near the best signal on every arm, mutual information is the
weakest task signal on three of four arms (it edges confidence on the v1 test
split, 0.117 against 0.120), and the window-level gain of the linear fusion does not
carry over (it was fitted on window errors, not tile failures). The numbers
are not comparable to the paper's (different data, model and failure rates);
the point is which baseline was missing. Source `exp/out/exp49_summary.json`,
job 731703.

### What carries the fusion, and does the bag replicate (exp50)

exp50 refits exp49's linear fusion with the same seeds (its numbers reproduce
to the digit), saves the weights, refits it ten times leaving one signal out,
replicates the bag with sixteen fresh members, grades shift-label entropy
(issue 5), and fits a second fusion on tile failures.

| Standardized weight of the linear fusion (positive = more suspect) | v1 | v1.2 |
|---|---|---|
| control NDWI level | +0.87 | +0.87 |
| ensemble average entropy | +0.64 | +0.68 |
| ensemble predictive entropy | +0.62 | +0.66 |
| averaged confidence | +0.38 | +0.31 |
| embedding normalized distance | +0.36 | +0.35 |
| input extremity mean | -0.33 | -0.32 |
| input extremity max | +0.33 | +0.35 |
| tile-phase | -0.25 | -0.24 |
| embedding NCDD | -0.24 | -0.27 |
| ensemble mutual information | +0.22 | +0.24 |

| Pooled E-AURC | v1 Bolivia | v1 test | v1.2 Bolivia | v1.2 test |
|---|---|---|---|---|
| averaged confidence | 0.0094 | 0.0109 | 0.0116 | 0.0064 |
| linear fusion, all ten | 0.0067 | 0.0068 | 0.0089 | 0.0060 |
| without control NDWI level | 0.0076 | 0.0071 | 0.0116 | 0.0050 |
| without averaged confidence | 0.0065 | 0.0068 | 0.0089 | 0.0060 |
| without ensemble average entropy | 0.0067 | 0.0068 | 0.0088 | 0.0060 |
| without ensemble predictive entropy | 0.0067 | 0.0068 | 0.0088 | 0.0060 |
| without embedding normalized distance | 0.0067 | 0.0067 | 0.0088 | 0.0064 |
| without input extremity max | 0.0066 | 0.0069 | 0.0086 | 0.0059 |
| without tile-phase | 0.0067 | 0.0068 | 0.0087 | 0.0060 |

The weight sits on the no-model NDWI-level control, with the two ensemble
entropies and confidence behind it; dropping NDWI level costs the most on
three of four arms, and on Bolivia under v1.2 it returns the fusion to
confidence's level. Label-free midrank fusion of the same two signals lost
everywhere in exp47; what the labels buy is the weighting. <!-- claim:fusion-weight-on-ndwi -->

The bag does not replicate. Sixteen members with a fresh seed lose to
confidence on Bolivia under v1 (0.0100 vs 0.0094,
140/166 tiles), where exp49's eight members had won, and win on the
v1.2 test split (0.0059 vs 0.0064, 248/160), where they had lost.
The preregistered P1 fails; the bag is confidence plus seed noise at the
0.002 level, and its ledger row reads not supported. Shift-label entropy <!-- claim:bag-not-replicated -->
loses to tile-phase on Bolivia under both backbones (0.0161 vs
0.0113; 0.0150 vs 0.0133) and never beats it. <!-- claim:shift-label-entropy-rejected -->

| Tile level | v1 Bolivia | v1 test | v1.2 Bolivia | v1.2 test |
|---|---|---|---|---|
| confidence, ROI-averaged: AURC (Risk@0.5) | 0.167 (0.110) | 0.120 (0.105) | 0.140 (0.087) | 0.134 (0.130) |
| fusion fitted on tile failures: AURC (Risk@0.5) | 0.134 (0.082) | 0.078 (0.057) | 0.138 (0.096) | 0.086 (0.068) |
| AURC difference, 95% tile bootstrap | -0.033 [-0.078, +0.008] | | -0.042 [-0.066, -0.019] | | -0.002 [-0.037, +0.030] | | -0.048 [-0.073, -0.025] |

Fitted on tile failures, the fusion beats confidence at the tile level on the
multi-region split under both backbones, with intervals that exclude zero,
and not on Bolivia, the preregistered arm; P2 fails as stated and the
tile-level gain stands as a secondary result on the split that has many
events. Source `exp/out/exp50_summary.json` and `exp50_fusion_ablation.csv`,
job 736320. <!-- claim:tile-fitted-fusion-split-only -->

## Ai2's own Sen1Floods11 probe under the window protocol (exp51)

Everything above about v1.2's ranking rests on our head and our sensor. Ai2's
Sen1Floods11 evaluation is a Sentinel-1 linear probe with sixteen pixel logits
per token (`olmoearth_pretrain/evals`, `sen1floods11 -> [SENTINEL1]`), and Ai2
publishes the embeddings behind their Table 2. exp51 runs that probe recipe in
fp32 on their embeddings (arm A, plus seven other encoders they publish) and on
our own S1 encodes of v1 and v1.2 (arms B and C), all graded on 4-px windows.

| Their probe on our S1 encode | window acc | pixel mIoU | E-AURC probe confidence | NDWI level (tiles conf/NDWI better, two-sided p) | S1 level |
|---|---|---|---|---|---|
| v1 Base, Bolivia | 0.8801 | 0.739 | 0.0214 | 0.0272 (181/148, p = 0.078) | 0.1221 |
| v1 Base, test split | 0.9097 | 0.779 | 0.0221 | 0.0144 (700/813, p = 0.004) | 0.0905 |
| v1.2 Base, Bolivia | 0.8619 | 0.700 | 0.0241 | 0.0363 (181/157, p = 0.21) | 0.1565 |
| v1.2 Base, test split | 0.9163 | 0.795 | 0.0195 | 0.0141 (685/826, p = 0.00031) | 0.0767 |

| Encoder, their embeddings, their probe | pixel mIoU | window acc | E-AURC confidence | phi of window errors vs OlmoEarth Base | P(OlmoEarth wrong given model wrong) |
|---|---|---|---|---|---|
| olmoearth_base | 0.789 | 0.9155 | 0.0202 | 1 | 1 |
| galileo_base | 0.792 | 0.9168 | 0.0202 | 0.805 | 0.82 |
| croma_base | 0.787 | 0.9152 | 0.0230 | 0.803 | 0.81 |
| terramind_base | 0.782 | 0.9135 | 0.0227 | 0.818 | 0.82 |
| clay_large | 0.784 | 0.9135 | 0.0249 | 0.801 | 0.80 |
| anysat | 0.777 | 0.9121 | 0.0199 | 0.779 | 0.77 |
| panopticon | 0.777 | 0.9113 | 0.0229 | 0.784 | 0.78 |
| satlas_base | 0.727 | 0.8871 | 0.0394 | 0.620 | 0.56 |

On their own embeddings OlmoEarth Base reaches mIoU 0.789 against the
paper's 79.2, and our S1 encode of the same model with the same probe agrees
with it window for window (phi 0.937 on the 2,419 matched test tiles,
accuracy 0.9151 against 0.9164 on the shared block): the encode path <!-- claim:encode-path-reproduces-their-probe -->
reproduces their probe. Three results follow.

P1, preregistered and supported: the probe's own confidence beats the
sensor-level control everywhere (lead +0.0684 on the test split, per tile
1391/179 on their embeddings). <!-- claim:s1-probe-p1-confidence-beats-sensor-control -->

P2, preregistered and not supported: with their readout and their sensor,
v1.2 ranks its own Bolivia errors worse than v1 pooled (0.0241 against
0.0214, lead +0.0027) but not per tile (173/152, p = 0.13; phi between the
two backbones' error sets 0.654). v1.2 is also less accurate there with S1
(0.8619 against 0.8801) and more accurate on the test split. The v1.2 <!-- claim:v12-bolivia-their-readout -->
exception, as a per-tile finding, rests on our S2 head.

The flip. Under the S1 probe the no-model NDWI index, computed from the
Sentinel-2 of the same tiles, loses on Bolivia (n.s.) and beats the probe's own
confidence on the multi-region test split under both backbones
(0.0144 against 0.0221, 813 tiles to 700; 0.0141 against
0.0195, 826 to 685), and on their own embeddings (0.0143 against 0.0202,
833 to 732). With the S2 head the exception was Bolivia; with the S1 probe it is the <!-- claim:s1-probe-ndwi-flip -->
multi-region split. The index from the other sensor wins wherever the model's
own sensor is the less informative one for water.

Every encoder errs on the same windows, again. On Ai2's embeddings, with their
probe, six of the seven other encoders share OlmoEarth's error windows at phi
0.78-0.82 and are wrong on 77-82% of the windows OlmoEarth is wrong on; Satlas, <!-- claim:cross-encoder-phi-on-their-embeddings -->
with a 2 x 2 token grid on these tiles, is the one outlier at 0.62. exp41's
result on our S2 heads (0.77-0.81) replicates on their sensor and their
readout with no encoder pass. Galileo Base edges OlmoEarth on mIoU (0.792
against 0.789). Source `exp/out/exp51_summary.json`, job 761713.

## Fine-tuning OlmoEarth ourselves (exp52)

Ai2's fine-tuning recipe restated in fp32 (a linear per-pixel head on the pooled
window tokens, the backbone frozen for the first fifth of the epochs then
unfrozen at a tenth of the learning rate, a plateau scheduler on validation
mIoU, the best checkpoint kept; lr 3e-4, 12 epochs, batch 32, our choices),
trained on the bucket's train split and graded on Bolivia and the exp18 test
sample, which it never sees. Each fine-tuned model's own shift-averaged
decision goes through exp47's protocol, and its error set is cross-tabbed
against the frozen exp18 head's on identical windows, as exp21 did with Ai2's
AWF model.

| Fine-tuned model | window acc (frozen W1) | pixel acc, one tiling | E-AURC confidence | NDWI level (tiles conf/NDWI better) | tile-phase | frozen errors corrected | correct windows broken | phi vs frozen |
|---|---|---|---|---|---|---|---|---|
| FT-S2 v1, Bolivia | 0.9540 (0.9174) | 0.9493 | 0.0040 | 0.0063 (168/86) | 0.0058 | 0.658 (3881/5895) | 1272 | 0.423 |
| FT-S2 v1, test split | 0.9674 (0.9554) | 0.9656 | 0.0092 | 0.0086 (302/61) | 0.0113 | 0.441 (2946/6680) | 1144 | 0.641 |
| FT-S1 v1, Bolivia | 0.8802 (0.9174) | 0.8754 | 0.0192 | 0.0271 (169/134) | 0.0301 | 0.611 (3599/5895) | 6251 | 0.249 |
| FT-S1 v1, test split | 0.9105 (0.9554) | 0.9061 | 0.0347 | 0.0158 (204/252) | 0.0414 | 0.337 (2253/6680) | 8976 | 0.434 |
| FT-S1+S2 v1, Bolivia | 0.9549 (0.9174) | 0.9500 | 0.0041 | 0.0061 (167/85) | 0.0056 | 0.672 (3962/5895) | 1285 | 0.409 |
| FT-S1+S2 v1, test split | 0.9678 (0.9554) | 0.9668 | 0.0080 | 0.0084 (293/60) | 0.0100 | 0.457 (3054/6680) | 1195 | 0.625 |
| FT-S2 v1.2, Bolivia | 0.9529 (0.9174) | 0.9491 | 0.0040 | 0.0061 (168/91) | 0.0054 | 0.635 (3744/5895) | 1210 | 0.450 |
| FT-S2 v1.2, test split | 0.9659 (0.9554) | 0.9656 | 0.0061 | 0.0085 (300/58) | 0.0087 | 0.439 (2930/6680) | 1348 | 0.628 |

Both preregistered tests pass. P1: the fine-tuned S2 model's confidence beats
the no-model NDWI index on Bolivia under both backbones (+0.0023, 168/86 tiles;
+0.0021, 168/91): the Bolivia exception belonged to the frozen encoder, not to the <!-- claim:finetune-dissolves-bolivia-exception -->
event. On the multi-region split the same confidence ties the index pooled
under v1 (-0.0005, but 302/61 per tile) and beats it under v1.2 (+0.0024) and with
both sensors (+0.0004, 293/60). P2: fine-tuning corrects 66% of the frozen head's Bolivia
errors and 44% of its multi-region errors (v1.2: 64% and 44%), breaking far fewer <!-- claim:finetune-corrects-frozen-errors -->
than it corrects; exp21's 55.6% on Ai2's AWF model sits between the two
testbeds. Window accuracy rises from the frozen 0.9174 / 0.9554 to
0.9540 / 0.9674, and a single tiling of the fine-tuned model (0.949 / 0.966 pixel
accuracy) already beats the frozen shift-averaged decision (0.907 / 0.950,
exp42).

The sensor lever after training. Adding Sentinel-1 to the fine-tuned
Sentinel-2 model adds 0.10 / 0.04 window-accuracy points (0.9549 against 0.9540;
0.9678 against 0.9674); Sentinel-1 alone, fine-tuned, stays below the frozen S2 <!-- claim:s1-adds-little-after-finetune -->
head (0.8802 / 0.9105) and breaks more of its windows than it corrects, and on the
multi-region split the NDWI index still ranks the S1 model's errors better
than its own confidence (0.0158 against 0.0347), exp51's flip with a trained model. The
modality lever exp46 measured on frozen probes is a frozen-feature property:
once the S2 model is trained, the radar has little left to add on this task.
Source `exp/out/exp52_summary.json`, job 762708.

## Test-time adaptation of the head by shift consistency (exp53)

Per tile, a copy of the exp18 head is adapted on that tile's four cached
feature maps to make the tilings agree, memo (entropy of the shift-averaged
probability) or consistency (variance across tilings), each with an anchor to
the original weights, fitted on tilings 0 and 2 and stopped when tilings 1 and
3 stop agreeing more; the adapted decision is the shift-averaged one. Labels
grade, never adapt.

| Adapted head, hand-label pixel accuracy | Bolivia (baseline 0.9071) | test split (baseline 0.9503) |
|---|---|---|
| memo: mean gain, tiles better/worse/tied, one-sided p | -0.0032, 191/186/63, p = 0.42 | -0.0079, 230/381/189, p = 1 |
| memo: held-out disagreement before -> at the kept step; tiles improved | 0.0675 -> 0.0366; 0.99 | 0.0541 -> 0.0303; 0.98 |
| memo: E-AURC of the adapted confidence (baseline) | 0.0108 (0.0094) | 0.0145 (0.0109) |
| consistency: mean gain, tiles better/worse/tied, one-sided p | -0.0079, 171/202/67, p = 0.95 | -0.0140, 196/417/187, p = 1 |
| consistency: held-out disagreement before -> at the kept step; tiles improved | 0.0675 -> 0.0338; 0.99 | 0.0541 -> 0.0281; 0.96 |
| consistency: E-AURC of the adapted confidence (baseline) | 0.0103 (0.0094) | 0.0099 (0.0109) |

Rejected, and instructively. The held-out disagreement falls on 99% of the tiles <!-- claim:shift-tta-rejected -->
and the adaptation runs to the cap on the median tile, while hand-label
accuracy falls on both testbeds under both objectives. Agreement across views
is not correctness: the head can be moved so that every tiling is wrong
together, and a label-free validation on held-out views does not guard
against it. The consistency variant ranks its own errors slightly better on
the test split and decides worse. Source `exp/out/exp53_summary.json`, job 762709.

## Does the audit save labels? (exp56)

The effect-size question put directly: if the audit's ranking says where the
frozen model is wrong, does labelling those tiles teach a fine-tuned model
more than labelling random ones? Pool = the bucket's train split; three ways
to choose B tiles to label (random; the audit's review-set rule on the frozen
head, tiles with the most bottom-quintile-confidence or boundary windows;
predictive entropy, the classic active-learning baseline); budgets 100, 300,
500, 1,000; three seeds; each choice fine-tuned with exp52's recipe at a
matched step count and graded on Bolivia and the test split, which no
strategy sees.

| Window accuracy, mean over three seeds (min) | B = 100 | 300 | 500 | 1000 |
|---|---|---|---|---|
| random, test split | 0.9525 (0.9430) | 0.9600 (0.9591) | 0.9609 (0.9606) | 0.9622 (0.9604) |
| audit, test split | 0.9432 (0.9427) | 0.9547 (0.9519) | 0.9589 (0.9568) | 0.9588 (0.9559) |
| entropy, test split | 0.9360 (0.9351) | 0.9554 (0.9547) | 0.9586 (0.9583) | 0.9577 (0.9558) |
| random, Bolivia | 0.9308 (0.9160) | 0.9455 (0.9433) | 0.9429 (0.9309) | 0.9501 (0.9492) |
| audit, Bolivia | 0.9323 (0.9296) | 0.9503 (0.9464) | 0.9490 (0.9463) | 0.9500 (0.9445) |
| entropy, Bolivia | 0.9324 (0.9268) | 0.9505 (0.9500) | 0.9499 (0.9486) | 0.9486 (0.9444) |
| frozen-head error rate of the selected tiles, audit / entropy (pool 0.045) | 0.173 / 0.215 | 0.159 / 0.181 | 0.148 / 0.163 | 0.128 / 0.133 |

Both preregistered tests fail, and not narrowly. At 300 labels the audit's
selection is -0.0053 behind random on the test split on the seed mean and behind on
every seed (-0.0044, -0.0090, -0.0027); it never reaches random's 1,000-label accuracy at any budget, <!-- claim:audit-does-not-save-labels -->
nor does entropy. On Bolivia the audit's tiles tie or edge random's. The
selected tiles are three to four times harder than the pool (frozen error rate
0.13-0.17 against 0.045) and richer in water, and a model fitted to them
generalises worse to the multi-region split. The audit's ranking says where
the frozen model is wrong, not which labels teach the model; label efficiency
is not a product of the audit, and the review set stays a review set. Source
`exp/out/exp56_summary.json`, job 775302.

## The window protocol on Ai2's multi-class embeddings (exp54)

Four more tasks, all dense and multi-class, from Ai2's published Table-2
embeddings, with their probe recipe per task and no encoder pass: MADOS
(marine debris, 15 classes), PASTIS (19 crop types, Sentinel-2 alone and with
Sentinel-1), GEO-Bench's m_cashew_plant (7) and m_sa_crop_type (10); six
encoders on the first two. Windows are 4 px with the majority class as label;
the control is the embedding-space one (distance to the training centroids),
since the embeddings carry no pixels.

| task | encoder | classes | pixel mIoU | window acc | E-AURC confidence | embedding distance | capture at 5% | boundary-first at 5%, tiles better/worse |
|---|---|---|---|---|---|---|---|---|
| mados | olmoearth_base | 15 | 0.668 | 0.9264 | 0.0078 | 0.0423 | 0.291 | 0.278, 6/0 |
| mados | galileo_base | 15 | 0.659 | 0.9291 | 0.0051 | 0.0495 | 0.371 | 0.301, 4/1 |
| mados | croma_base | 15 | 0.615 | 0.9172 | 0.0062 | 0.0634 | 0.322 | 0.238, 4/0 |
| mados | terramind_base | 15 | 0.652 | 0.9386 | 0.0048 | 0.0375 | 0.391 | 0.317, 7/0 |
| mados | clay_large | 15 | 0.393 | 0.8555 | 0.0353 | 0.1504 | 0.215 | 0.192, 7/0 |
| mados | anysat | 15 | 0.499 | 0.8228 | 0.0526 | 0.1242 | 0.145 | 0.124, 2/3 |
| pastis_sentinel2 | olmoearth_base | 19 | 0.498 | 0.8099 | 0.0386 | 0.1612 | 0.157 | 0.148, 225/173 |
| pastis_sentinel2 | galileo_base | 19 | 0.384 | 0.7657 | 0.0539 | 0.2024 | 0.134 | 0.127, 270/167 |
| pastis_sentinel2 | croma_base | 19 | 0.442 | 0.7820 | 0.0493 | 0.1871 | 0.139 | 0.132, 173/170 |
| pastis_sentinel2 | terramind_base | 19 | 0.403 | 0.7724 | 0.0540 | 0.1890 | 0.137 | 0.130, 204/158 |
| pastis_sentinel2 | clay_large | 19 | 0.215 | 0.6727 | 0.0938 | 0.2813 | 0.104 | 0.100, 263/174 |
| pastis_sentinel2 | anysat | 19 | 0.438 | 0.7793 | 0.0517 | 0.1719 | 0.135 | 0.129, 224/176 |
| m_cashew_plant | olmoearth_base | 7 | 0.261 | 0.6528 | 0.1308 | 0.2664 | 0.089 | 0.089, 34/6 |
| m_sa_crop_type | olmoearth_base | 10 | 0.288 | 0.6600 | 0.0672 | 0.2761 | 0.107 | 0.105, 647/196 |
| pastis_sentinel1_sentinel2 | olmoearth_base | 19 | 0.476 | 0.8052 | 0.0394 | 0.1645 | 0.155 | 0.146, 216/150 |

Preregistered P1 holds on all five tasks: the probe's confidence beats the
embedding-distance control by 0.03 to 0.21 excess AURC, with the per-tile <!-- claim:multiclass-confidence-beats-embedding-control -->
sign test below 0.05 everywhere. Predictive entropy sits within 0.005 of
confidence; cross-encoder disagreement is a poor ranker (0.0263 on MADOS,
0.1531 on PASTIS against confidence's 0.0078 and 0.0386).

Shared errors are a property of the task, not of the encoders. On the binary
water task the encoders err on the same windows (phi 0.77-0.82, exp41,
exp51); on MADOS they share far less and on PASTIS almost nothing (mados: galileo_base 0.33, croma_base 0.26, terramind_base 0.32, clay_large 0.27, anysat 0.44; pastis_sentinel2: galileo_base 0.08, croma_base 0.08, terramind_base 0.08, clay_large 0.05, anysat 0.08). <!-- claim:shared-errors-task-dependent -->

Boundary-first at many classes splits by the measure. Pooled over windows
it captures slightly fewer errors than confidence at the 5% budget on every
task and encoder; per tile it wins the sign test on four of six PASTIS
encoders, on cashew (34/6) and on SA crop type (647/196), and is mixed on
MADOS, whose tiles rarely hold enough errors to score. My stated prediction <!-- claim:boundary-first-many-classes-split -->
(a clean loss at 15 and 19 classes) held on the pooled measure and failed on
the per-tile one. Source `exp/out/exp54_summary.json`, job 775213.

## GEOID-Flood: the exception as a rate over events (exp55)

Every Bolivia exception in this repository is one event. GEOID-Flood (219 CEMS
activations, event-level splits, manually validated three-class labels) turns
the question into a rate: a shard subset of its test split gave 55 events, 64-px
chips at stride 256 with validity and cloud filters, heads trained on a val
subset the events never touch, and exp47's protocol run per event with the
same four crop offsets. Two sensors on identical windows: permanent water from
the pre-event Sentinel-2 composite (S2 head) and water after the event from the
post-event Sentinel-1 (S1 head; flooded-only as a secondary task).

| GEOID-Flood, 55 events (45 scored) | A: permanent water, pre-event S2 | B: water after the event, post-event S1 | C: flooded only, post-event S1 |
|---|---|---|---|
| chips, windows, window accuracy | 4,497, 870,728, 0.9677 | 4,527, 875,976, 0.9619 | 4,527, 875,976, 0.9833 |
| pooled E-AURC, averaged confidence | 0.0028 | 0.0030 | 0.0017 |
| pooled E-AURC, tile-phase | 0.0040 | 0.0031 | 0.0019 |
| pooled E-AURC, control NDWI level | 0.0129 | 0.0181 | 0.0217 |
| pooled E-AURC, control S1 level | 0.1035 | 0.0930 | 0.0358 |
| pooled E-AURC, boundary first, then confidence | 0.0029 | 0.0031 | 0.0017 |
| events where control NDWI level beats confidence | 2/45 (4%) | 4/45 (9%) | 3/44 (7%) |
| events where control S1 level beats confidence | 0/45 (0%) | 1/45 (2%) | 2/44 (5%) |
| events where tile-phase beats confidence | 11/45 (24%) | 23/45 (51%) | 12/44 (27%) |
| events where boundary-first beats confidence at 5% | 18/45 | 12/45 | 14/44 |
| median event: confidence's capture at 5% (x random) | 0.877 (17.5x) | 0.642 (12.8x) | 0.565 (11.3x) |

All three preregistered tests pass. The no-model index beats the S2 head's
confidence on 2 of 45 events (4%) and the sensor-level control beats the S1 head's on <!-- claim:geoid-exception-rate -->
1 of 45; Bolivia-type exceptions exist and are rare. The effect size in the units a
reviewer uses: on the median event the 5% of windows confidence flags hold
88% of the permanent-water errors and 64% of the post-event water errors, 17.5 and 12.8 times <!-- claim:geoid-capture-effect-size -->
what a random 5% would. Two stated predictions fail: exp51's sensor flip does
not generalise (the pre-event NDWI prior out-ranks the S1 head's confidence on
9% of events, not a majority), and boundary-first beats confidence at the 5% budget on <!-- claim:geoid-sensor-flip-not-general -->
40% / 27% / 32% of events, a minority, while tying pooled. The two sensors' heads share errors at phi 0.58 pooled
and 0.30 within the median event; the S1 head is wrong on 65% of the windows the S2
head is wrong on. Runtime 7 minutes. Source `exp/out/exp55_summary.json`, job 775649.

## The difference atlas: every pair of inferences under one measurement (exp57)

Every earlier comparison of two inferences of the same scene turned the
difference into a ranker and graded it, or reported one phi; the difference
itself, how large it is, where it sits, whether two differences are the same
set, was never the object, and the arithmetic was copied from script to
script. exp57 promotes it to `oe_inferencex.compare` and applies it to every
pair the repository holds. A pair is two hard decisions on identical 4-px
windows with a validity mask, a tile or event id per window, the labels, and
four label-free cues of the first inference: on a prediction boundary of its
own window map, among the least confident 20% of the testbed's windows under
it, among the 20% most unstable under a sub-patch shift of its tiling (where it
has four crop offsets), spectrally ambiguous (|NDWI| < 0.1). Label-free first:
the disagreement rate pooled and per tile or event, each cue's share among the
disagreement windows over its share among the agreement windows, the stability
of the disagreement set across three head draws (the full training set and two
80% subsets of its tiles, seeds 1 and 2) and across pairs on the same grid.
The label bridge second: which side is right on the disagreement windows and
exp52's cross-tab of the two error maps. The Sen1Floods11 pairs sit on exp47's
W1 grid (the shift-averaged decision pooled to windows 1..14 of the offset-0
grid) on Bolivia (441 tiles, 71,373 windows) and the multi-region test split
(800 tiles, 149,684 windows): the tiling at crop offset 0 against offset 2,
OlmoEarth v1 against v1.2 Base, the S2 head against the S1 head on v1, and the
frozen S2 head against FT-S2 v1, fine-tuned once more with exp52's recipe. The
seven encoders of exp51 are paired with OlmoEarth Base on Ai2's published
embeddings and probe (their test split, 2,419 tiles, 563,969 windows), and per
GEOID-Flood event (exp55's 55 events, 870,728 windows) the pre-event S2 head
(permanent water) with the post-event S1 head (water after the event), each
error map graded against its own task label. Two inferences disagree on
2.0-4.0% of the windows across crop offsets and backbones, 7.2% / 2.7% between
the frozen and the fine-tuned head, 10.9% / 8.4% across sensors, 2.8-3.5%
across encoders with Satlas at 7.0%, and 3.4% on GEOID-Flood. <!-- claim:atlas-disagreement-rates -->

| Pair, testbed | windows | disagreement, pooled / median tile | boundary enrichment | right on the disagreement windows | phi of the two error maps | stability across head draws |
|---|---|---|---|---|---|---|
| offset 0 vs offset 2, Bolivia | 71,373 | 3.4% / 2.6% | 4.2x | offset 0 39%, offset 2 61% | 0.779 | 0.59 |
| offset 0 vs offset 2, test split | 149,684 | 2.6% / 1.5% | 5.4x | 50% / 50% | 0.714 | 0.69 |
| v1 vs v1.2 Base, Bolivia | 71,373 | 4.0% / 2.9% | 3.7x | v1 55%, v1.2 45% | 0.743 | 0.64 |
| v1 vs v1.2 Base, test split | 149,684 | 2.0% / 1.0% | 5.3x | v1 47%, v1.2 53% | 0.765 | 0.65 |
| S2 head vs S1 head, Bolivia | 71,373 | 10.9% / 6.6% | 3.3x | S2 67%, S1 33% | 0.415 | 0.75 | <!-- claim:atlas-disagreement-is-boundary-located -->
| S2 head vs S1 head, test split | 149,684 | 8.4% / 3.2% | 4.0x | S2 84%, S1 16% | 0.430 | 0.86 |
| frozen vs FT-S2 v1, Bolivia | 71,373 | 7.2% / 3.9% | 3.8x | frozen 25%, FT-S2 75% | 0.428 | one fine-tune |
| frozen vs FT-S2 v1, test split | 149,684 | 2.7% / 1.0% | 5.3x | frozen 29%, FT-S2 71% | 0.650 | one fine-tune |
| OlmoEarth Base vs galileo_base, their test split | 563,969 | 2.9% / 0.8% | 6.2x | OlmoEarth 47%, galileo 53% | 0.814 | one probe each; the seven disagreement sets at phi 0.29-0.55 |
| OlmoEarth Base vs croma_base, their test split | 563,969 | 3.0% / 1.2% | 6.1x | 50% / 50% | 0.810 | as above |
| OlmoEarth Base vs terramind_base, their test split | 563,969 | 2.8% / 0.8% | 6.6x | OlmoEarth 54%, terramind 46% | 0.825 | as above |
| OlmoEarth Base vs clay_large, their test split | 563,969 | 3.1% / 1.2% | 6.3x | OlmoEarth 53%, clay 47% | 0.809 | as above |
| OlmoEarth Base vs satlas_base, their test split | 563,969 | 7.0% / 2.7% | 7.1x | OlmoEarth 71%, satlas 29% | 0.624 | as above |
| OlmoEarth Base vs anysat, their test split | 563,969 | 3.5% / 1.2% | 6.2x | OlmoEarth 55%, anysat 45% | 0.787 | as above |
| OlmoEarth Base vs panopticon, their test split | 563,969 | 3.4% / 1.2% | 6.1x | OlmoEarth 56%, panopticon 44% | 0.792 | as above |
| S2 head vs S1 head per GEOID-Flood event, pooled | 870,728 | 3.4% | 10.1x | 27% flooded by the label; S2 wrong 33%, S1 wrong 40%, neither 27% | 0.585 | n/a |
| S2 head vs S1 head, median GEOID-Flood event | 55 events | 1.4% | 21.2x | flooded by the label 0.7% (66% at the 90th percentile) | 0.30 | n/a |

The four Sen1Floods11 differences are different sets of windows. In the 4 x 4
matrix of phi between the disagreement masks every off-diagonal entry lies
between 0.18 and 0.41 (median 0.28 on both testbeds), and the preregistered
pair, the sensor difference against the backbone difference, gives phi 0.27 on
Bolivia and 0.18 on the test split; per tile the tiles with phi below 0.5
outnumber the rest 257 to 70 and 386 to 103 (one-sided exact sign test,
p = 1.5e-26 and 8.0e-40; tiles with an undefined phi dropped). The seven
encoder disagreement sets on Ai2's embeddings overlap at phi 0.29 to 0.55
(median 0.45), Satlas at 0.29-0.39 with every other encoder. A re-drawn head
keeps most of a difference: the same pair's disagreement mask recurs across
the three head draws at median pairwise phi 0.59-0.86, above any overlap
between two different pairs. <!-- claim:atlas-different-differences -->

The label says which side is right only for the differences that changed the
model. The fine-tuned model is right on 75% / 71% of its disagreements with the
frozen head (259/84 and 333/127 tiles) and the S2 head on 67% / 84% of its
disagreements with the S1 head (223/125 and 449/103 tiles); between crop
offsets and between backbones the winning side takes 39-61% and changes with
the testbed, and OlmoEarth Base is right on 47-56% of its disagreements with
six of the seven encoders and on 71% with Satlas. <!-- claim:atlas-which-side -->

On GEOID-Flood the two heads predict different things, permanent water before
the event and water after it, so their disagreement is change as well as
error. It covers 3.4% of the windows pooled and 1.4% on the median event, and
it is boundary-located more sharply than on Sen1Floods11: enrichment 10.1x
pooled, 21.2x on the median event, above 2 on every one of the 49 events with a
defined value. Of the disagreement windows (the categories overlap) 27% are
flooded by the label, 33% are errors of the S2 head on its own task, 40% errors
of the S1 head on its, and 27% are neither; the flooded share is concentrated
in a few events, 0.7% on the median event and 66% at the 90th percentile. The
two error maps share phi 0.585 pooled and 0.30 on the median event, exp55's
numbers. <!-- claim:atlas-geoid-disagreement-is-change-and-error -->

Both preregistered results hold. P1: the disagreement windows of every pair are
boundary-enriched, above 2 on all sixteen tested pairs, 3.3x to 7.1x on
Sen1Floods11 and 21.2x on the median GEOID-Flood event. <!-- claim:atlas-disagreement-is-boundary-located -->
P2: the sensor difference and the backbone difference are different
differences, phi 0.27 / 0.18 pooled and the tiles below 0.5 a majority on both
testbeds. The module's cross-tabs reproduce the recorded ones: the fine-tuned
model corrects 65.5% / 42.7% of the frozen head's errors against exp52's
65.8% / 44.1% (a fresh fine-tune with the same recipe, window accuracy 0.9539 /
0.9666 against 0.9540 / 0.9674), the encoders share OlmoEarth Base's errors at
phi 0.79-0.83 with Satlas at 0.62 against exp51's 0.78-0.82 and 0.62, and the
two GEOID-Flood heads at 0.585 as in exp55. <!-- claim:atlas-reproduces-recorded-cross-tabs -->
Runtime 19:44 for the Sen1Floods11 and encoder pairs, 11 minutes of it the
fine-tune, and 3:34 for the GEOID-Flood job. Source
`exp/out/exp57_summary.json`, jobs 779972 and 779973.

## Is the comparison tool better than the raw diff? (exp58)

exp57 measured the difference between two inferences of the same scene and
showed where it sits; the question a user asks next is whether the package
tells them anything a raw disagreement map does not. The raw diff is the
baseline. It says which windows changed and nothing else: every disagreement
window is equally suspect, no side is preferred, and on two dates every change
is a change. `oe_inferencex.compare` reads the same disagreement set
label-free in three ways, and each reading is a prediction about the labels
that exp58 grades. Resolution believes the side with the larger margin
|p - 0.5| on the window (on a binary task the mean-probability decision),
graded as the share of disagreement windows where the chosen side matches the
label, against the raw diff's coin flip (0.5 in expectation, since exactly one
side is right on a binary window), always the first inference and always the
second; recorded alongside, not preregistered, the same rule on
rank-normalised margins, each side's margin as its midrank percentile over the
testbed's valid windows. Targeting orders the disagreement windows by the
first inference's own confidence, least confident first, where the raw diff
has no order, graded as the share of the first inference's errors inside the
least confident half of the set (a random order captures 0.50 in expectation)
and inside the least confident 20%. Change, on GEOID-Flood, where the two
heads predict different things, reads a disagreement window where both sides
are confident (each margin at least 0.25) as change where the raw diff reads
every disagreement as change, graded by the flooded-by-label share among the
both-confident disagreements against the share among all of them. The pairs
and grids are exp57's: crop offset 0 against 2, v1 against v1.2 Base, the S2
head against the S1 head, the frozen head against FT-S2 v1 (fine-tuned once
more with exp52's recipe), each on Bolivia and the multi-region test split of
Sen1Floods11 on the W1 grid; the seven encoders against OlmoEarth Base on
Ai2's embeddings and probe on their test split; the pre-event S2 head against
the post-event S1 head per GEOID-Flood event. The per-tile tests are one-sided
exact sign tests that the reading beats its baseline on more tiles than not,
over the tiles with at least 3 disagreement windows, and for the three pairs
with exp57's three head draws the resolution rule is also graded on the
disagreement windows that persist across all draws against those that appear
in draw 0 only.

| Pair, testbed | disagreement windows | right on them: the rule / always the first / always the second | resolution, tiles w/l, p | least confident half holds (random 0.50) | targeting, tiles w/l, p | the rule on persistent / transient windows |
|---|---|---|---|---|---|---|
| offset 0 vs offset 2, Bolivia | 2,458 | 51.3% / 39.0% / 61.0% | 148/116, p = 0.028 | 52.1% | 117/101, p = 0.15 | 0.45 / 0.56 |
| offset 0 vs offset 2, test split | 3,852 | 58.2% / 49.8% / 50.2% | 250/109, p = 3.5e-14 | 56.8% | 208/107, p = 6.7e-9 | 0.57 / 0.60 |
| v1 vs v1.2 Base, Bolivia | 2,850 | 56.0% / 55.3% / 44.7% | 151/92, p = 9.3e-5 | 56.6% | 109/112, p = 0.61 | 0.54 / 0.58 |
| v1 vs v1.2 Base, test split | 2,971 | 57.7% / 47.4% / 52.6% | 191/106, p = 4.6e-7 | 54.4% | 166/103, p = 7.4e-5 | 0.56 / 0.60 |
| S2 head vs S1 head, Bolivia | 7,761 | 68.2% / 67.3% / 32.7% | 254/47, p = 8.1e-36 | 73.8% | 192/83, p = 2.1e-11 | 0.67 / 0.72 |
| S2 head vs S1 head, test split | 12,535 | 69.8% / 84.2% / 15.8% | 374/92, p = 1.1e-41 | 92.3% | 297/163, p = 2.1e-10 | 0.68 / 0.76 |
| frozen vs FT-S2 v1, Bolivia | 5,141 | 61.3% / 24.7% / 75.3% | 191/80, p = 6.1e-12 | 52.1% | 153/80, p = 1.0e-6 | one fine-tune |
| frozen vs FT-S2 v1, test split | 4,068 | 55.1% / 28.1% / 71.9% | 202/105, p = 1.7e-8 | 50.1% | 151/105, p = 0.0024 | one fine-tune |
| OlmoEarth Base vs galileo_base, their test split | 16,588 | 54.1% / 47.5% / 52.5% | 633/351, p = 9.4e-20 | 52.2% | 501/370, p = 5.1e-6 | one probe each |
| OlmoEarth Base vs croma_base, their test split | 17,034 | 55.1% / 50.5% / 49.5% | 693/354, p = 2.8e-26 | 53.4% | 556/371, p = 6.7e-10 | one probe each |
| OlmoEarth Base vs terramind_base, their test split | 15,853 | 54.4% / 53.7% / 46.3% | 644/346, p = 9.4e-22 | 54.7% | 500/358, p = 7.0e-7 | one probe each |
| OlmoEarth Base vs clay_large, their test split | 17,363 | 55.9% / 53.3% / 46.7% | 708/344, p = 6.1e-30 | 55.6% | 580/375, p = 1.7e-11 | one probe each |
| OlmoEarth Base vs satlas_base, their test split | 39,731 | 61.6% / 71.0% / 29.0% | 978/348, p = 7.8e-70 | 71.5% | 939/334, p = 3.7e-67 | one probe each |
| OlmoEarth Base vs anysat, their test split | 19,505 | 54.7% / 54.9% / 45.1% | 728/375, p = 5.6e-27 | 55.3% | 601/379, p = 6.7e-13 | one probe each |
| OlmoEarth Base vs panopticon, their test split | 19,060 | 56.1% / 56.2% / 43.8% | 716/363, p = 1.5e-27 | 57.1% | 599/374, p = 2.7e-13 | one probe each |

Believing the more confident side beats the coin flip everywhere: the chosen
side is right on 51-70% of the disagreement windows on all fifteen
pair-testbeds and on more tiles than not on every one (sign test p from 0.028
on the Bolivia offsets to 7.8e-70 on Satlas), but it misses the preregistered
55% pooled on four of them, the Bolivia offsets at 51.3% and three encoders at
54.1-54.7% (Galileo, TerraMind, AnySat), and wherever one side is known to be
better the rule is far below always choosing it: the S2 head is right on 84%
of its disagreements with the S1 head on the test split against the rule's
70%, and the fine-tuned model on 75% / 72% of its disagreements with the
frozen head against the rule's 61% / 55%. <!-- claim:tool-vs-diff-resolution -->
The rank-normalised variant changes little: 0.55 instead of 0.51 on the
Bolivia offsets, within 0.02 everywhere else, and worse on the two sensor
pairs (0.64 / 0.63 against 0.68 / 0.70).

Ordering the disagreement set by the first inference's confidence adds little
to the set. On twelve of the fifteen pair-testbeds the least confident half
holds 50-57% of the first inference's errors against the random order's 50%,
and the least confident 20% holds 20-25%; the per-tile test still passes on
ten of the twelve and fails on the Bolivia offsets and backbones. The order
helps only where the other side is much worse, so that the first inference's
few errors on the set are its least confident windows: 74% / 92% of the S2
head's errors on its disagreements with the S1 head, which is wrong on 67% /
84% of them, and 72% of OlmoEarth Base's errors on its disagreements with
Satlas, wrong on 71%. <!-- claim:tool-vs-diff-targeting -->

A disagreement that survives redrawing the head is a hard window, not a
settled one. On all six pair-testbeds with head draws the resolution rule is
less accurate on the persistent disagreement windows (those in all three
draws) than on the transient ones (draw 0 only), 0.45-0.68 against 0.56-0.76,
the opposite of the stated prediction; the persistent set is 46-80% of the
disagreement windows, largest on the two sensor pairs. <!-- claim:tool-vs-diff-persistence -->

On GEOID-Flood the two heads predict different things, permanent water before the
event and water after it, so the reading is change against error. Of the 29,392
disagreement windows 12,142 (41%) have both sides confident, a margin of at least
0.25 on each; 52% of those are flooded by the label, against 27% of all
disagreement windows and 10% of the rest, 1.9 times the raw diff's precision for
change pooled over the 55 events. Per event the reading wins on 16, loses on 9 and
ties on 16 of the 41 events with at least 20 disagreement windows (one-sided sign
test p = 0.11): the pooled ratio clears the preregistered 1.5, the event test does
not. Where one side is unconfident, that side is wrong on its own task half of the
time (50%). Graded on water after the event, believing the more confident side is <!-- claim:tool-vs-diff-change -->
right on 45% of the disagreement windows (always the S1 head 60%; tiles 27/14,
p = 0.03), since the S2 head predicts permanent water and is wrong on every flooded
window by construction; the least confident half of the set holds 57% of its errors
(tiles 28/12, p = 0.008).

Both preregistered results fail. P1: the chosen side is right on at or below
55% of the disagreement windows pooled on four of the fifteen pair-testbeds
(the Bolivia offsets 51.3%, Galileo 54.1%, TerraMind 54.4%, AnySat 54.7%), so
at the preregistered level a margin rule is no better than the raw diff's
silence on which side to believe, though it beats the coin flip by the
per-tile test on every pair. P2: the least confident half holds below 55% of
the first inference's errors pooled on seven pair-testbeds and the per-tile
test misses 0.05 on two more (the Bolivia offsets and backbones), eight in
all, so the order adds nothing to the set.
P3 fails on its event test: the both-confident windows are flooded 1.9 times as
often pooled, above the preregistered 1.5, but on 16 events against 9 (p = 0.11),
short of the majority the test required; confidence on both sides separates change
from error in the pooled count and not event by event. <!-- claim:tool-vs-diff-change -->
Runtime 18:17 for the Sen1Floods11 and encoder pairs and 6:10 for GEOID-Flood,
refetching exp55's shard subset after the extracted tree had been emptied. Source `exp/out/exp58_summary.json`, jobs 787530 and 789352.

## A label-fitted rule for which side to believe (exp59)

exp58's follow-up: the one fusion that ever beat confidence here was fitted on
the head's own training split (exp49), so the same route for the comparison
half. On every disagreement window eleven label-free features of the two
sides (each side's margin, midrank percentile, signed probability, boundary
indicator and tile-phase, and the NDWI level); a logistic rule fitted once on
the 600 valid-split tiles the frozen heads were trained on, both sides encoded
at the four crop offsets; graded on Bolivia and the test split against the raw
margin rule, always the first side, always the second, per tile with
`compare.over_groups`. On GEOID-Flood the same features fitted on the val
events' disagreement windows to predict "flooded by the label", graded per test
event against the raw diff and exp58's both-confident reading.

| Pair, testbed | fitted rule right | raw margin rule | gain, tiles w/l, p | always the better side |
|---|---|---|---|---|
| offset 0 vs 2, Bolivia | 63.6% | 51.3% | +0.12, 162/41, 1.6e-18 | 61.0% |
| offset 0 vs 2, test | 68.1% | 58.2% | +0.10, 223/86, 1.8e-15 | 50.2% |
| v1 vs v1.2, Bolivia | 69.8% | 56.0% | +0.14, 170/75, 6.0e-10 | 55.3% |
| v1 vs v1.2, test | 65.2% | 57.7% | +0.08, 177/96, 5.4e-7 | 52.6% |
| S2 vs S1 head, Bolivia | 73.2% | 68.2% | +0.05, 148/120, 0.049 | 67.3% |
| S2 vs S1 head, test | 86.4% | 69.8% | +0.17, 305/88, 2.0e-29 | 84.2% |
| frozen vs FT-S2, Bolivia | 35.3% | 60.9% | -0.26, 71/162, 1.0 | 75.8% |
| frozen vs FT-S2, test | 54.2% | 55.7% | -0.02, 155/112, 0.005 | 71.8% | <!-- claim:fitted-resolution-mixed -->

The fitted rule adds 5 to 17 points over the raw margin rule on the crop-offset,
backbone and sensor pairs, on more tiles than not everywhere, and on the sensor
pairs it reaches or passes always choosing the S2 head (73% against 67%, 86%
against 84%); on the fine-tuned pair it loses, 35% on Bolivia where always
choosing the fine-tuned model gives 76%, because a rule fitted to frozen heads
reads the fine-tuned model's margins wrongly (it believes it on 23% of the
windows). P1 fails on that pair and, by 0.001 of gain, on the Bolivia sensor pair. <!-- claim:fitted-resolution-mixed -->
On GEOID-Flood the fitted change rule is 80% flooded where it calls change,
3.0 times the raw diff's 27% and 1.5 times the both-confident reading's 52%,
but it calls only 6% of the differing windows (recall 18%) and wins on 11 test
events against 8 (p = 0.32): precision bought with coverage, and no event-level
majority, so P2 fails. <!-- claim:fitted-change-rule-precision-not-recall -->
Runtime 18:49 and 4:20 (jobs 791849, 791850). Source `exp/out/exp59_summary.json`.

## Two periods, both sensors: the time axis and the sensor axis apart (exp60)

The README's GEOID pair mixes two axes, time and sensor. GEOID-Flood holds three
of the four cells of the two-period, two-sensor design (no post-event optical
pass: the scene is under cloud), so the pre-event Sentinel-1 pass was extracted
from the same shards exp55 fetched, one Sentinel-1 water head was fitted on the
val chips' pre pass with the permanent-water label and post pass with the
water-after label, and three pairs were read through `compare` on the same 869,160
windows of 4,489 clear chips in 55 events: time-only (S1 before against S1 after,
one head), sensor-only (S2 before against S1 before, both permanent water), and the
mixed pair (S2 before against S1 after). Each head is accurate on its own task
(96.8%, 97.7%, 98.5% of the windows).

| Pair | differing windows, pooled / median event | flooded by the label | first side off its label | second side off its label | boundary enrichment | both confident: share, flooded |
|---|---|---|---|---|---|---|
| time-only: S1 before vs S1 after | 3.4% / 2.2% | 25.9% | 54.5% (pre pass vs permanent water) | 19.6% | 12.9x | 48%, 41% |
| sensor-only: S2 before vs S1 before | 4.9% / 1.6% | 0.4% | 58.9% | 41.1% | 7.3x | 67%, 0.2% |
| mixed: S2 before vs S1 after | 4.6% / 1.5% | 19.5% | 63.0% | 17.8% | 7.4x | 57%, 26% | <!-- claim:two-periods-composition -->

The sensor axis isolates cleanly: a same-period cross-sensor difference carries
almost none of the later flood, 0.4% of its differing windows against 19.5% for
the mixed pair (ratio 0.02), lower on 23 events against 3 (one-sided sign test
p = 4e-5); preregistered P2 holds. Its content is sensor error alone: where the
optical and the radar head disagree on permanent water, one of them is off the
label, the optical one on 59% of the windows. <!-- claim:two-periods-sensor-axis-isolated -->
The time axis does not concentrate the flood: the same-sensor difference across
the event is flooded on 25.9% of its windows, 1.33 times the mixed pair's 19.5%,
short of the preregistered 2, and higher on 13 events against 13 (p = 0.58);
P1 fails. More than half of that difference, 54.5%, is where the pre-event radar
pass departs from the permanent-water label. The reading that this is the water
regime of the pre-event date, seasonal water the permanent class does not carry,
was tested by exp61 with a date-matched arbiter and rejected (next section). Where both sides are confident the time-only difference is
flooded 41% of the time against 26% on the mixed pair, on 12 events against 9
(p = 0.33, a stated prediction, not supported). The sensor-only pair has fewer
differing windows than the mixed pair on 21 events against 17 (p = 0.31, the
other stated prediction, not supported). <!-- claim:two-periods-time-only-difference -->
Caveat: the two radar passes may differ in orbit and incidence angle, so the time
axis includes acquisition geometry. Runtime 5:24 (job 794821), the pre-event pass
extracted in about a minute. Source `exp/out/exp60_summary.json`,
`exp/out/exp60_masks.npz` (the three decision maps, margins and labels).

## Is the pre-event radar's departure the seasonal water of its date? (exp61)

exp60 left a reading untested: that the 54.5% of the same-sensor difference where
the pre-event radar departs from the permanent-water label is water that was
there on that date and the permanent class does not carry. Two things test it
without new imagery. GEOID ships three layers exp55 never extracted, permwater,
floodmask and validity (about 10 MB per shard); and the JRC Global Surface Water
monthly history (v1.4, 30 m, 1984 to 2021) gives water, not water or no
observation for the month of each tile's pre-event Sentinel-1 pass, read by HTTP
range requests and reprojected onto the chip grid. exp60's chips and decisions were
reproduced and asserted equal to the committed masks, so every window is exp60's.

The residue is 16,199 of the 29,700 time-only differing windows: 14,447 where the
radar said water and the label not permanent, 1,752 where the radar said dry and
the label permanent. The JRC product observed 85% of them in that month and calls
3.2% of the observed ones water: 0.1% of the radar-water part, 32% of the radar-dry
part. The raw CEMS floodmask marks 1.9% of the residue. Per event, the JRC water
share on the residue is at or near zero on 14 of the 17 events with at least 20
observed residue windows and above one half on 3 small ones (one-sided sign test
against one half, p = 0.999). Preregistered P1 fails: by a date-matched optical
arbiter, the pre-event radar's departures from the permanent label are not the
seasonal water of the date; they are windows the radar head calls water that
neither the label nor the month's Landsat product holds. What they are physically,
radar-dark dry surfaces or water too narrow for 30 m optics, the two optical
references cannot decide. <!-- claim:residue-not-seasonal-water -->
Three sanity facts came with it. GEOID's permanent class is its permwater layer
(equal on 99.3% of windows) and its flooded class is its floodmask (99.9%), so
grading against those layers adds nothing. The JRC monthly water agrees with the
label on 99.2% of the 649,877 observed windows, more than with the radar head
(97.5%) or the optical head (96.4%). On the 42,696 sensor-only differing windows it
sides with the radar head 61% to 39%. <!-- claim:geoid-label-is-permwater -->
Runtime 1:05 on the cpu partition (job 799222; a first run, job 799100, was discarded:
its reader had rewritten the product's no-observation value as not-water). Source
`exp/out/exp61_summary.json`, `exp/out/exp61_layers.npz` (the pooled layers on
exp60's windows).

## The fourth cell: post-event optical from WorldFloods completes the square (exp62)

GEOID-Flood has no optical pass after the event. WorldFloods v2 (isp-uv-es on the
Hub, Sentinel-2 L1C at 10 m with cloud and water masks, CC BY-NC 4.0) holds 26
post-event scenes on four of the eleven activations behind exp60's 55 events, most
of them clear. A water head for post-event optical imagery was fitted on OlmoEarth
features of 1,766 clear WorldFloods train chips from 40 scenes of other activations,
and every shared scene was reprojected onto exp60's chips: 544 of the 4,489 chips,
in 6 events, are at least 90% clear on a post-event scene and carry all four cells.
The fourth cell's head is right on 97.1% of GEOID's water-after windows there and
agrees with WorldFloods' own water mask on 97.1%; the two labels agree on 98.4%. <!-- claim:fourth-cell-completed -->

| Pair on the 544 chips (106,606 windows) | differing windows | flooded by the label | first side off its label | second side off its label |
|---|---|---|---|---|
| time-only optical: S2 before vs S2 after | 24.4% | 3.8% | 88.5% | 7.7% |
| time-only radar: S1 before vs S1 after | 2.6% | 41.1% | 9.8% | 49.2% |
| sensor-only before: S2 vs S1 | 21.9% | 0.1% | 97.9% | 2.1% |
| sensor-only after: S2 vs S1 | 3.8% | 18.0% | 55.0% | 45.0% |
| mixed: S2 before vs S1 after | 23.4% | 4.6% | 90.2% | 5.3% |
| mirror: S1 before vs S2 after | 3.4% | 27.3% | 17.8% | 55.0% | <!-- claim:fourth-cell-prereg-fails -->

Both preregistered claims fail, and the table says why: on this subset the
pre-event optical head is off the permanent-water label on 22% of the windows
(78.1% accuracy against 96.8% over all of exp60; 99.0% for the pre-event radar
head, 97.5% for the post-event radar head), so every pair that contains it measures
that failure. It is one event: on EMSR279-11 (the Ebro delta at Tortosa, 393 of
the 544 chips) the pre-event optical head calls water on 9% of the windows where
the label holds 38% permanent water. P1 compared
the optical time-only pair (3.8% flood) with the radar one (41.1%), one event
against five; P2 compared the post-event sensor pair (18.0%) with the optical
time-only pair, four events against two (p = 0.34). The pairs without the
pre-event optical head carry the design's logic: the radar difference across the
event is 41% flood on these chips (26% over all 55 events), the cross-sensor
difference after the event 18%, and where both radar dates are confident the
temporal difference is flooded 87% of the time. <!-- claim:fourth-cell-prereg-fails -->
Runtime 14:12 (job 799540; the head 12 min, the cut 92 s). Source
`exp/out/exp62_summary.json`, `exp/out/exp62_masks.npz` (the fourth cell's decisions
and margins, WorldFloods' water and clear masks per window, chip ids).

## The difference atlas at 15 and 19 classes (exp63)

Every comparison finding so far was a binary-water finding. exp63 runs the atlas
on Ai2's paper embeddings for MADOS (marine debris, 15 classes) and PASTIS (crop
types, 19 classes): exp54's probes per encoder, decisions on 4-px windows aligned
by the label tile, OlmoEarth Base with three probe draws, each other encoder
against Base, Base against its own second draw, and on PASTIS the sensor pair,
Base on Sentinel-2 against Base on Sentinel-1 plus Sentinel-2 over the same
scenes. Cues of the first side: boundary, the bottom margin quintile, the top
entropy quintile. Windows: 78,720 on MADOS, 458,638 on PASTIS.

| Pair | differing windows | boundary | low margin | high entropy | a right / b right / neither | errors phi |
|---|---|---|---|---|---|---|
| MADOS: Base vs galileo | 7.6% | 4.8x | 5.2x | 5.3x | 43% / 47% / 11% | 0.49 |
| MADOS: Base vs croma | 6.7% | 5.5x | 5.4x | 5.4x | 53% / 38% / 9% | 0.58 |
| MADOS: Base vs terramind | 6.2% | 5.1x | 4.9x | 4.9x | 38% / 57% / 5% | 0.53 |
| MADOS: Base vs clay | 15.2% | 3.5x | 4.1x | 4.1x | 65% / 19% / 16% | 0.38 |
| MADOS: Base vs anysat | 13.4% | 6.2x | 6.0x | 5.9x | 86% / 9% / 5% | 0.49 |
| MADOS: Base draw 0 vs draw 1 | 0.3% | 5.0x | 5.1x | 5.1x | 48% / 48% / 4% | 0.98 |
| PASTIS: Base vs galileo | 13.6% | 1.8x | 4.7x | 3.8x | 56% / 23% / 21% | 0.68 |
| PASTIS: Base vs croma | 13.5% | 1.8x | 4.9x | 3.9x | 50% / 29% / 21% | 0.67 |
| PASTIS: Base vs terramind | 13.7% | 1.8x | 4.8x | 3.8x | 53% / 26% / 21% | 0.68 |
| PASTIS: Base vs clay | 25.4% | 1.8x | 3.3x | 3.4x | 68% / 14% / 18% | 0.50 |
| PASTIS: Base vs anysat | 13.9% | 1.8x | 4.7x | 3.7x | 51% / 29% / 20% | 0.66 |
| PASTIS: Base draw 0 vs draw 1 | 0.2% | 1.7x | 5.0x | 3.5x | 35% / 40% / 24% | 1.00 |
| PASTIS: Base S2 vs S1+S2 | 6.0% | 1.8x | 5.8x | 3.8x | 43% / 35% / 22% | 0.85 | <!-- claim:multiclass-atlas-composition -->

**The boundary cue does not travel to parcels.** On MADOS the differing windows
sit on boundaries 3.5 to 6.2 times as often as the agreeing ones, as on floods;
on PASTIS 1.7 to 1.8 times on every pair, because 50% of PASTIS's agreeing
windows already border another class (parcels are small) against 14% on
MADOS, so the cue has nowhere to concentrate. Preregistered P1 fails on the seven
PASTIS pairs and holds on the six MADOS ones. Low margin and high entropy locate
the differences on both tasks, 3.3 to 6.0 times, and they are the cue to use
where classes are dense. <!-- claim:multiclass-boundary-cue-fails-on-parcels -->
**The sensor difference and the encoder difference are different sets at 19
classes too:** the PASTIS S2-vs-S1+S2 set and the Base-vs-Galileo set overlap at
phi 0.34 pooled (galileo 0.34, croma 0.32, terramind 0.34, clay 0.25, anysat 0.31 against each encoder), below 0.5 on
392 tiles against 65 (p < 1e-50); P2 holds. The five encoder disagreement sets
overlap at a median phi of 0.50 on PASTIS and 0.45 on MADOS. <!-- claim:multiclass-sensor-vs-encoder-different -->
**The head-draw floor.** Base against its own second probe draw differs on
0.31% of MADOS windows and 0.17% of PASTIS windows: every encoder
difference (6 to 25%) and the sensor difference (6.0%) is 20 to 150 times above
the noise of refitting the head, the first such floor for any difference rate
here. Which side is right follows accuracy: on MADOS Base and the three encoders
near its accuracy split their disagreements 38 to 57% each way; on PASTIS,
where Base leads every encoder by 3 to 14 points, it is right on 50 to 68% of the
differing windows and the other side on 14 to 29%, with a fifth of the windows
wrong on both sides, the multi-class case the binary tasks could not show. On the
sensor pair neither side is preferred (43% against 35%). Runtime 11:37 (job
800249, no downloads). Source `exp/out/exp63_summary.json`, `exp/out/exp63_masks.npz`. <!-- claim:multiclass-atlas-composition -->

## Calibrate on the record: the fused readings, held-out (exp65)

`oe_inferencex.calibrate` fits the label-free readings to labels and reports the
fit held-out, cross-fitted by tile, bound to a model family. exp65 records what it
gives on exp59's four Sen1Floods11 pairs and on exp49's ranker, against the
numbers those experiments got by fitting on the training split.

**The ranker fusion** (v1 S2 head; readings: confidence, aligned tile-phase, the
boundary indicator, NDWI level, S2 patch variance; five folds by tile):

| Testbed | confidence alone, excess AURC | fusion, held-out | lead | tiles, p | held-out capture at 5% / 10% |
|---|---|---|---|---|---|
| Bolivia | 0.0105 | 0.0084 | +0.0021 | 263/69, 4.1e-28 | 0.31 / 0.57 |
| test split | 0.0098 | 0.0069 | +0.0029 | 318/134, 1.3e-18 | 0.45 / 0.68 | <!-- claim:calibrate-ranker-fusion-beats-confidence -->

Preregistered P1 holds on both testbeds: the fusion cuts confidence's excess AURC
by 20% on Bolivia and 30% on the test split, held-out, and wins on
four tiles of five. The weights say what the labels add: the NDWI level carries
the largest standardised weight on Bolivia (1.74 against 0.38 for
confidence), where the no-model index was the frozen head's one exception
(exp47), and confidence and tile-phase share the lead on the test split. <!-- claim:calibrate-ranker-fusion-beats-confidence -->

**The side rule** (which side to believe where two decisions differ; readings per
side: margin, its midrank, the signed probability, the boundary indicator,
tile-phase; NDWI shared; five folds by tile):

| Pair, testbed | raw margin rule | cross-fitted rule | gain, points | tiles, p | always the better side | exp59's protocol, fitted on the training tiles |
|---|---|---|---|---|---|---|
| crop offset 0 vs 2, bolivia | 51.3% | 75.8% | +24.5 | 186/57, 1.9e-17 | 61.0% | 63.5% |
| crop offset 0 vs 2, test | 58.2% | 69.0% | +10.8 | 230/107, 9.1e-12 | 50.2% | 68.1% |
| v1 vs v1.2, bolivia | 56.0% | 77.2% | +21.2 | 188/63, 6.3e-16 | 55.3% | 69.6% |
| v1 vs v1.2, test | 57.7% | 65.4% | +7.7 | 167/91, 1.3e-06 | 52.6% | 65.2% |
| S2 head vs S1 head, bolivia | 68.2% | 79.7% | +11.4 | 176/87, 2.2e-08 | 67.3% | 72.5% |
| S2 head vs S1 head, test | 69.8% | 86.0% | +16.2 | 301/87, 5.2e-29 | 84.2% | 86.4% |
| frozen vs FT-S2 v1, bolivia | 61.1% | 74.9% | +13.8 | 176/73, 2.7e-11 | 75.2% | 35.5% |
| frozen vs FT-S2 v1, test | 55.4% | 72.8% | +17.4 | 204/97, 3.3e-10 | 71.7% | 53.3% | <!-- claim:calibrate-side-rule-held-out -->

Preregistered P2 holds on all eight: the cross-fitted rule beats the raw margin
rule by 8 to 24 points, on more tiles than not everywhere. On the three frozen
pairs it matches or passes exp59's train-split fit; on the fine-tuned pair,
where exp59's rule lost 26 points, refitting per family gains 14 and 17. <!-- claim:calibrate-side-rule-held-out -->
**The family lock** does what it is for: the rule fitted on the pooled frozen
pairs' training windows, forced onto the fine-tuned pair, is right on 43.8% and
49.2% of its differing windows against 61.1% and 55.4% for the raw margin and
74.9% and 72.8% for the pair's own cross-fit (P3 holds), and the module raises
before scoring it unless forced. Runtime 12:49 (job 801135). Source
`exp/out/exp65_summary.json`; `exp/out/exp65_readings.npz` holds every reading on
the disagreement and valid windows with tile ids and labels, and the fits
reproduce from it to the last digit. <!-- claim:calibrate-family-lock -->

## DFC2020: eight-class land cover, both sensors, our own encoder, two references (exp66)

Ai2 suggested DFC2020 as a labelled evaluation set. It is the first dense
testbed here that is neither binary flood water nor read through Ai2's
published embeddings: eight land-cover classes at 10 m, Sentinel-1 and
Sentinel-2 over the same scenes, our own frozen OlmoEarth v1 Base with their
linear probe fitted on 400 validation patches and reported on 1,200 test
patches, 4,320,000 windows of 4 px per arm, two probe seeds each.

One correction belongs at the top, because the contest is usually described
otherwise, including when it was suggested to us. The 10 m reference is **not
hand-drawn**: it is an iterated random forest over Sentinel-1, Sentinel-2,
spectral indices, the 500 m MODIS map and FROM-GLC10, with a published overall
accuracy of 0.824. The same patches also carry those 500 m MODIS labels. That
makes the dataset more useful here, not less: one testbed with two references
of known and different quality over identical pixels is the instrument this
repository has never had for its oldest caveat, that reference-product labels
flatter boundary-type signals (exp18). They agree on 71.3% of the windows
both label.

| Arm | pixel mIoU | window accuracy | margin, excess AURC | lead over the NDWI control (p) | entropy | boundary-first | margin capture at 5% / 20% |
|---|---|---|---|---|---|---|---|
| Sentinel-2 | 0.431 | 0.716 | 0.0369 | +0.1576 (5e-218) | -0.0015 | +0.0453 | 0.128 / 0.482 |
| Sentinel-1 | 0.322 | 0.569 | 0.0782 | +0.1734 (5e-76) | +0.0073 | +0.1198 | 0.088 / 0.342 |
| both | 0.435 | 0.725 | 0.0373 | +0.1557 (2e-178) | -0.0005 | +0.0409 | 0.128 / 0.479 |

The ranking transfers. On every arm the model's own margin beats the no-model
pixel index by 0.156 to 0.173 of excess AURC, on more patches than not with
p below 1e-75 (preregistered P1 holds), and the mean intersection over union
sits in the range the literature reports for a linear probe on frozen features,
despite the imagery being top-of-atmosphere where the encoder expects surface
reflectance. Predictive entropy ties the margin, as everywhere in this
repository. Boundary-first **loses** to the margin on all three arms, by 0.041
to 0.120: with eight classes the cue behaves as it did at fifteen and nineteen
(exp54, exp63) and not as it does on binary water. <!-- claim:dfc2020-margin-beats-pixel-control -->

| Pair on identical windows | differing windows | boundary | low margin | a right / b right / neither | errors phi |
|---|---|---|---|---|---|
| sensors | 44.27% | 3.30x | 4.65x | 50% / 17% / 32% | 0.38 |
| s2 vs joint | 15.07% | 2.63x | 5.37x | 26% / 31% / 43% | 0.79 |
| s2 draw 0 vs 1 | 0.82% | 2.32x | 5.17x | 27% / 29% / 45% | 0.99 |
| s1 draw 0 vs 1 | 1.34% | 2.40x | 5.29x | 19% / 27% / 54% | 0.99 |
| s1s2 draw 0 vs 1 | 1.19% | 2.33x | 5.25x | 26% / 34% / 41% | 0.98 |

**The sensor axis dominates land cover.** Sentinel-2 and Sentinel-1 disagree on
44.3% of the windows, against 1.34% when only the probe seed changes: a factor
of 33 above the noise floor, and the differing windows carry the boundary
cue 3.3 times as often as the agreeing ones (preregistered P2 holds). On water
the same axis moved 4.9% of windows (exp60); on eight land-cover classes it
moves nine times as many, and the optical side is right on 50% of them against
17% for radar, with 32% wrong on both sides. Every pair's differing windows are
boundary-enriched 2.3 to 3.3 times and low-margin-enriched 4.6 to 5.4 times, so
the atlas findings hold on land cover under our own encoder; and the five
difference sets are close to independent, median pairwise phi 0.08, the
strongest form yet of exp57's result that different differences are different
sets of windows. <!-- claim:dfc2020-sensor-difference-dominates-land-cover --> <!-- claim:dfc2020-atlas-holds-with-our-own-encoder -->

**The coarse reference penalises the boundary order; it does not flatter it.**
This was preregistered the other way and P3 fails. Boundary-first is behind the
margin by 0.0453, 0.1198 and 0.0409 against the 10 m reference, and further
behind, by 0.0766, 0.1351 and 0.0822, against the 500 m one; the gap is negative
on all three arms (-0.0313, -0.0153, -0.0413). The reading refines exp18 rather
than repeating it: a reference flatters a boundary cue only when it resolves
boundaries at a comparable scale, as WorldCover did at 10 m against a 10 m
prediction. A reference twenty times coarser has its own transitions in the
wrong places, so the cue loses alignment and scores worse. The practical form,
for anyone auditing against a reference product: check the reference's
resolution against the prediction's before trusting any boundary-shaped
signal's score. <!-- claim:dfc2020-coarse-reference-penalises-the-boundary-order -->

Stated caveats, carried from the preregistration: the imagery is L1C
top-of-atmosphere while the encoder's modality is L2A surface reflectance; the
split archives carry no real georeferencing, so windows are addressed by patch
and pixel; validation and test differ in class composition, which is a shift and
not an i.i.d. split; and the reference's own errors are correlated with what a
linear probe on these features learns, which is the reason part C exists.
Publishing a DFC2020 number needs case-by-case approval from the IEEE GRSS IADF
technical committee and TUM under the contest terms, so this section is an
internal record until that is settled. Runtime 1:06:02 including the 10.4 GB
fetch (job 804312). Source `exp/out/exp66_summary.json`; `exp/out/exp66_masks.npz`
carries the decisions, margins and both references for the first 200 patches.

## Served land cover change rasters (exp20)

First assessment of a served output: ten 512-px windows (about
4.9 km) of the published `allenai/olmoearth_lcc` rasters at Zambezi, Chobe
and Barotse sites, read with the pure-HTTP tile reader in
`oe_inferencex/lcc.py`. The product ships in EPSG:3857 at about 9.55 m as
9-band uint8 BigTIFFs.

- **No class confidence is exported**, so the recipe's primary signal cannot
  run on the class map. Bands 6-7 are *not* class confidences: they sit at
  255 on 80-99% of pixels because the category head answers "none" at
  unchanged pixels.
- **Boundary triage works.** Band 4 water against ESA WorldCover 2021 water
  (the one class whose legends coincide): six of ten sites contain water;
  disagreement is 0.2-2.4% of 4-px windows; boundary fraction ranks
  disagreements below random at 6 of 6 (Kazungula AURC 0.0013 against 0.0156
  random, oracle 0.0001); **a 5% review budget captures a median 0.88 of
  disagreements** (0.61-0.98). Boundary share is 0.92 among disagreements <!-- claim:served-product-boundary-triage -->
  against 0.01 among agreements. *Legends and dates differ, so these are
  reference disagreements, not counted model errors.*
- **Change probability** (band 1): a median 2.7% of pixels flagged at 0.5;
  60-95% of pixels at exactly 0; the ambiguous band 0.25-0.75 holds a median
  1.2%. Low confidence sits on flagged-region edges (2.7% of edge windows
  against 0.07% interior) — the boundary finding replicates label-free.
- **Sanity checks.** Predicted transitions are plausible for the region
  (tree to grassland or built-up at Kazungula and Katima, grassland to crops
  at Barotse, water to wetland at Linyanti). Full-legend disagreement with
  WorldCover is a median 49%, dominated by tree/shrub/grass and built-up,
  but the legends' semantics differ, so this is context, not an error rate.

Values in `exp/out/exp20_lcc_production.csv`; figure
`exp20_lcc_kazungula.png`.

## Periodic artifacts in the served product (exp22)

**Question.** Does the served v1.2 product carry striping or seams at the
scales its pipeline imposes?

5 windows of 4096 served px (about 37 km) from 3 tiles. Column and row
profiles of the class-map boundary indicator (band 4), the change-probability
gradient (band 1), and ESA WorldCover 2021 warped to the same grid (a control
sharing no model grid), tested for periodicity with a whitened periodogram.
Every pipeline period is predicted from each window's own geometry.

- **Patch lattice: found.** The largest peak of 19 of 20 product profiles
  lies on the encoder's patch lattice (8 at 1 patch, 5 at 2, 3 at 4, and 3 on
  the third harmonic of the 4-patch period), and the observed periods track
  each window's UTM-to-Mercator ratio. The weakest of the 19 still reaches
  Bonferroni p=3.9e-12; the WorldCover control's *best* top peak reaches only
  p=0.006. **Class boundaries and
  change-probability gradients are quantized to the 40 m patch grid.** <!-- claim:lcc-lattice-found-seams-absent -->
- **Inference-window seams: absent.** At 64, 128, 256 and 512 UTM px,
  profiles with p<0.01 are 0/10 in every class-map band and 1/10 in one
  gradient band — the rate expected under the null. Injected seams set the
  detection limit: at 128 px, seams affecting 5% of rows would have been
  detected in 3 windows and 10% in 2; at 256 px, 10% in 2 and 20% in 3. The
  single off-lattice top peak is kwando's change-gradient row profile, at
  7.55 px (1.72 patches).
- **Warp duplication beat** (nearest-neighbour warping repeats a source
  column every ratio/(ratio-1) served px): fundamental p<0.001 in 2/10
  class-map and 2/10 gradient profiles, 0/10 in the control.
- **Method validated before use** on lattice-free synthetic maps: scan false
  positives 0/10, confirmatory 1/10; seams on 5% of rows detected at sparse
  boundaries (p=9e-7), 10% at dense ones (p=5e-9); shear correction about
  triples power.

Values in `exp/out/exp22_lcc_striping.csv`, `exp22_confirmatory.csv`,
`exp22_power.csv`.

---

# 5. Supporting comparisons

## No-model image-statistic controls (exp06)

Control signals computed directly from pixel values (within-patch spectral
variance, patch-mean |NDWI| proximity to the water/land boundary, NDWI
gradient magnitude), scored on identical errors with the same harness.

- **Kazungula:** every model signal retains a margin over the best control
  (tile-phase 0.00058, |Nano-Base| 0.00086 vs spectral variance 0.0012).
- **Barotse floodplain:** E_case retains a margin (0.0235 vs NDWI gradient
  0.0384), so the wetland-margin result does not reduce to edge detection.
- **Zambezi delta:** all three no-model statistics rank the disagreements as
  well as or better than every model signal (NDWI gradient 0.0005 vs E_dist
  0.0014). This scene's disagreements are the river the reference misses —
  spectrally trivial — so **it supports no claim of model-signal
  superiority**, and the E_dist shift claim was withdrawn on this basis. <!-- claim:no-model-controls -->

On both difficult scenes, even no-model statistics rank errors better than
confidence.

![No-model controls vs model signals](../../exp/out/exp06_controls.png)

## Evidence from inside the encoder (exp17)

The v1-Base encoder hooked (per-block outputs, last-block q/k) for one
forward pass per scene on the 27 rule-selected scenes plus the training
scene. Five label-free single-model signals scored against the exp13 errors.

| Signal | vs baseline | vs control |
|---|---|---|
| Band-set disagreement | **21/27** (p=0.006) | 16/27 | <!-- claim:band-set-disagreement-mixed -->
| Depth-probe disagreement | 19/27 (p=0.052) | 13/27 | <!-- claim:depth-probe-partial -->
| Decision settling (logit-lens) | 0/27 | - | <!-- claim:internal-state-signals-rejected -->
| Representation drift | 3/27 | - | <!-- claim:internal-state-signals-rejected -->
| Attention entropy | 3/27 | - | <!-- claim:internal-state-signals-rejected -->

Band-set disagreement (heads trained separately on the 10 m, 20 m and 60 m
Sentinel-2 band-set tokens; std of the three probabilities) needs **no second
model and one forward pass**, and outperforms the two-model E_case (10/27).
The INSIDE-style internal-state signals do not transfer to this setting.

**Why a partner helps.** Error correlation with the final head (mean phi):
Nano 0.61, depth probe 0.64, logit-lens 0.61, 20 m band-set probe 0.75. The
band-set probe is the *most* correlated rater yet yields the *best*
disagreement signal — so error decorrelation alone does not predict a
partner's value; a partner that sees a different view of the input does. <!-- claim:band-set-most-correlated-best-partner -->
This refines exp10.

Best-signal tally: tile-phase 13, control 9, depth-probe 2, band-set 1,
attention entropy 1, E_case 1, baseline 0. Values in
`exp/out/exp17_internal_evidence.csv`.

## Rater strength vs diversity (exp10)

Replacing Nano with v1-Large as Base's partner makes the disagreement signal
**worse** (|Large-Base| mean AURC 0.0197 vs |Nano-Base| 0.0129 over seven
scenes, better on only 3/7) although Large is the more accurate model on
every scene. Within one family, strong models agree on errors. Refined by <!-- claim:stronger-partner-worse-disagreement -->
exp17 above.

## E_geo combined with boundary proximity (exp15)

E_geo flag = a patch on an OSM `waterway=river` centerline that the model
predicts dry. Georeferencing recovered for all 27 rule-selected scenes.

- **Prepending the flag to boundary proximity does not help:** better on 5
  scenes, worse on 9, unchanged on 13 (sign p=0.42). Geo alone beats the
  baseline on 3/27; boundary alone on 19/27. <!-- claim:geo-grounding-partial -->
- **Sensitivity.** 17/27 scenes carry flags; pooled precision 0.12 over 413
  flags against a base error rate of 0.081 (1.5x). The unweighted per-scene
  mean of 0.22 is inflated by scenes with one or two flags. **9 scenes carry
  1-51 flags at precision exactly zero**: OSM marks a river that both the
  model and WorldCover call dry — disagreement between two reference maps on
  narrow or seasonal channels.
- **Implication.** Under WorldCover truth, E_geo precision cannot be
  separated from the reference confound. It needs width-filtered centerlines
  (GRWL) or expert truth before its sensitivity can be stated.

Values in `exp/out/exp15_boundary_geo.csv`.

## OlmoEarth v1 vs v1.2 (exp19)

Run in an isolated environment on the current olmoearth_pretrain main, which
loads both versions; v1 features recomputed with the new code match the
cached exp11 features exactly (max difference 0).

- **RoPE does not reduce tiling instability.** Mean per-patch std across 0-3
  px shifts is 0.046 for v1.2 vs 0.032 for v1, smaller for v1.2 on only 6 of
  31 scenes (sign p=9e-4). RoPE addressed the long-range striping artifact;
  sub-patch grid-shift instability is a different effect. <!-- claim:rope-does-not-reduce-tiling-instability -->
- **v1.2 tokenizes Sentinel-2 as a single band-set token per patch**, so the
  exp17 band-set signal has no v1.2 counterpart.
- Head accuracy vs WorldCover on the Katima probe: v1 0.942, v1.2 0.922.
- Tile-phase ranks each version's own WorldCover-referenced errors better
  than its confidence (v1 26/1, v1.2 25/2), subject to the exp18 caveat.
  **Cross-version disagreement is not useful**: worse than confidence for
  v1's errors (6/21), not significant for v1.2's (18/9, p=0.12). <!-- claim:backbone-version-disagreement-rejected -->

Values in `exp/out/exp19_v1_vs_v12.csv`.

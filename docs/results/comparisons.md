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

**Result: confidence is the best signal**, on both splits.

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
correct on Bolivia; 73% vs 18% on test). The phenomenon is real; confidence
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
errors are also probe errors.

**Signal comparison** (16-px crops, 41 errors, tie-aware AURC, cluster
bootstrap over the 30 annotation tasks):

| Signal | AURC | Verdict |
|---|---|---|
| Aligned tiling instability | 0.0235 | indistinguishable from confidence (CI [-0.0068, +0.0010], P(better) 0.93) |
| **Confidence** | **0.0262** | best supported |
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
so a stated accuracy needs a coverage.**

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
  per-window tile-phase (0.0489, interval [+0.0023, +0.0221]).
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
no-model spectral-variability control 0.1658. In-domain AWF errors are
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
than the rule are excluded from exp13 onward, leaving **27 scenes**.

Head-seed variance is structurally zero: heads initialize at zeros with
deterministic full-batch training, so the planned seed-robustness test is
vacuous rather than passed. Robustness to head initialization is untested by
design choice.

*The exp11 statistics used raw AURC and the unaligned tile-phase; exp13
corrects both and supersedes them.*

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
of the five signals on 12/27 (best or second on 24/27).

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
(sign p=0.70; median E-AURC gap 0.00012 in tile-phase's favour). Per-scene
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
  same scenes. Boundary indicator 18/23 against 15/23.

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
per-scene gain is **not** larger in 2024 (10/16/0, p=0.327).

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
  instability 12/6/0 (2024), 13/8/0 (2021).

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
| Bolivia hand labels, 351 tiles | Preregistered test null: tiling instability 116/112/123 (p = 0.42), pooled 0.707 vs 0.749. Boundary indicator beats confidence at 5% (0.286 vs 0.259, CI [+0.009, +0.046]; 181/147/23, p = 0.068), loses at 20%. The NDWI-level control beats confidence pooled at 20% (0.795 vs 0.749), a caution on single operating points |
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
| Sen1Floods11 Bolivia, pooled over 351 tiles | 0.274 vs 0.259, CI [+0.009, +0.021] | 0.494 vs 0.465, CI [+0.015, +0.039] | 0.732 vs 0.749, CI [-0.054, +0.013] |
| Bolivia per tile, better/worse/tied | 85/31/235, one-sided p = 2.7e-7 | 112/48/191, p = 2.3e-7 | 121/77/153, two-sided p = 0.002 |
| AWF fine-tuned, 16-px crop, 30-task bootstrap | 0.220 vs 0.220 | 0.390 vs 0.390 | 0.634 vs 0.634 |
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
per-tile tests pass, the pooled bootstrap does not, so the result is mixed.
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
| W1 shift-averaged (preregistered) | 0.9071 | 0.9503 | 321/66/53, p = 1e-41; 583/78/139, p = 9e-97 |
| W3 spectral split inside the window | 0.9084 | 0.9483 | vs W1: 217/158 better on Bolivia, 226/404 worse on test |
| W2 scale-adaptive (window accuracy) | 0.9045 vs 0.9020 | 0.9515 vs 0.9518 | 175/121; 253/241 |
| segment majority of W1 over a spectral partition (exp43, preregistered) | 0.9042 | 0.9478 | vs W1: 168/209 (C 15,093, B 16,273); 244/387 (C 24,525, B 30,951) |
| sixteen crop offsets (exp44, preregistered) | 0.9074 | 0.9507 | vs W1: 201/149/90, p = 0.003 (C 4,566, B 4,237); 369/202/229, p = 1.3e-12 (C 7,100, B 6,060); +0.04 points, below the +0.2 minimum effect |
| eight offsets, diagonals and anti-diagonals (exp44) | 0.9073 | 0.9505 | half of W16's gain |
| seven offsets, diagonals and horizontal phases (exp44) | 0.9060 | 0.9487 | worse than the four diagonals on both |

Mixed-label windows are 10% of the windows and carry 45% and 58% of the
grid window's errors (error rate 0.39 and 0.28 against 0.054 and 0.022 on
pure windows); that is error concentration, not a bound. The
block-constant oracle limit, the pixels any one-class-per-block decision
must get wrong, is 3.0% and 2.9% pooled, against grid error rates of 10.3%
and 5.9% averaged over tiles, so the block forces at most 29% and 49% of
those errors, and the shift-averaged map decides per pixel and still puts
43% and 56% of its own errors on those windows (exp43).
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
split. Source `exp/out/exp45_summary.json`.

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
7.9e-44) and beats -NCDD (+0.077 and +0.035) on both testbeds. Three things
beside the preregistration. Bagging the head helps under v1: the predictive
entropy of the eight members' averaged maps beats confidence on Bolivia
(210/103 tiles, p = 1.5e-9) and marginally on the test split (240/174,
pooled -0.0002); under v1.2 it ties on Bolivia and loses on the test split.
The label-fitted linear fusion beats confidence on three arms by
0.0027-0.0041 (223/94, 265/155, 223/97; p <= 9e-8) and by 0.0004 on the v1.2
test split, the first fusion to beat confidence in this repository, where
exp47's label-free midrank fusions all lost; it is fitted on the head's own
training split against in-sample errors, so it needs no labels beyond the
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
weakest task signal, and the window-level gain of the linear fusion does not
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
everywhere in exp47; what the labels buy is the weighting.

The bag does not replicate. Sixteen members with a fresh seed lose to
confidence on Bolivia under v1 (0.0100 vs 0.0094,
140/166 tiles), where exp49's eight members had won, and win on the
v1.2 test split (0.0059 vs 0.0064, 248/160), where they had lost.
The preregistered P1 fails; the bag is confidence plus seed noise at the
0.002 level, and its ledger row reads not supported. Shift-label entropy
loses to tile-phase on Bolivia under both backbones (0.0161 vs
0.0113; 0.0150 vs 0.0133) and never beats it.

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
job 736320.

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
  disagreements** (0.61-0.98). Boundary share is 0.92 among disagreements
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
  change-probability gradients are quantized to the 40 m patch grid.**
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
  superiority**, and the E_dist shift claim was withdrawn on this basis.

On both difficult scenes, even no-model statistics rank errors better than
confidence.

![No-model controls vs model signals](../../exp/out/exp06_controls.png)

## Evidence from inside the encoder (exp17)

The v1-Base encoder hooked (per-block outputs, last-block q/k) for one
forward pass per scene on the 27 rule-selected scenes plus the training
scene. Five label-free single-model signals scored against the exp13 errors.

| Signal | vs baseline | vs control |
|---|---|---|
| Band-set disagreement | **21/27** (p=0.006) | 16/27 |
| Depth-probe disagreement | 19/27 (p=0.052) | 13/27 |
| Decision settling (logit-lens) | 0/27 | - |
| Representation drift | 3/27 | - |
| Attention entropy | 3/27 | - |

Band-set disagreement (heads trained separately on the 10 m, 20 m and 60 m
Sentinel-2 band-set tokens; std of the three probabilities) needs **no second
model and one forward pass**, and outperforms the two-model E_case (10/27).
The INSIDE-style internal-state signals do not transfer to this setting.

**Why a partner helps.** Error correlation with the final head (mean phi):
Nano 0.61, depth probe 0.64, logit-lens 0.61, 20 m band-set probe 0.75. The
band-set probe is the *most* correlated rater yet yields the *best*
disagreement signal — so error decorrelation alone does not predict a
partner's value; a partner that sees a different view of the input does.
This refines exp10.

Best-signal tally: tile-phase 13, control 9, depth-probe 2, band-set 1,
attention entropy 1, E_case 1, baseline 0. Values in
`exp/out/exp17_internal_evidence.csv`.

## Rater strength vs diversity (exp10)

Replacing Nano with v1-Large as Base's partner makes the disagreement signal
**worse** (|Large-Base| mean AURC 0.0197 vs |Nano-Base| 0.0129 over seven
scenes, better on only 3/7) although Large is the more accurate model on
every scene. Within one family, strong models agree on errors. Refined by
exp17 above.

## E_geo combined with boundary proximity (exp15)

E_geo flag = a patch on an OSM `waterway=river` centerline that the model
predicts dry. Georeferencing recovered for all 27 rule-selected scenes.

- **Prepending the flag to boundary proximity does not help:** better on 5
  scenes, worse on 9, unchanged on 13 (sign p=0.42). Geo alone beats the
  baseline on 3/27; boundary alone on 19/27.
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
  sub-patch grid-shift instability is a different effect.
- **v1.2 tokenizes Sentinel-2 as a single band-set token per patch**, so the
  exp17 band-set signal has no v1.2 counterpart.
- Head accuracy vs WorldCover on the Katima probe: v1 0.942, v1.2 0.922.
- Tile-phase ranks each version's own WorldCover-referenced errors better
  than its confidence (v1 26/1, v1.2 25/2), subject to the exp18 caveat.
  **Cross-version disagreement is not useful**: worse than confidence for
  v1's errors (6/21), not significant for v1.2's (18/9, p=0.12).

Values in `exp/out/exp19_v1_vs_v12.csv`.

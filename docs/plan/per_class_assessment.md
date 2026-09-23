# What a map user is owed per class (exp81 preregistration)

Written before the run, on 23 September 2026. Runs as `exp/exp81_per_class.py`; the numbers land in the ledger
like every other experiment.

## The gap

`estimate` answers "how wrong is this map" with one number and an interval. The map-accuracy literature has
settled, over a decade, on what a producer owes instead: per class, the **user's accuracy** (of the windows the
map calls *c*, the share that are *c*), the **producer's accuracy** (of the windows that are *c*, the share the
map found) and the **error-adjusted area** of *c*, each with a standard error, all from one probability sample
(Olofsson et al. 2014; Stehman and Foody 2019; Stehman and Wagner 2023; the CEOS LPV land-cover protocol, 2025).
A flood user acts on "water is missed 19% of the time and the flooded area is 8.1 ± 0.9 km², not 9.4", not on
"the map is 8% wrong". Nothing in the package computes a per-class quantity, and nothing in the record grades
one. This run adds both, from the sample `sample_for_estimation` already draws.

Stehman and Wagner (2023) add a warning this run tests: a sample allocated to estimate one target well, here the
overall error rate under the confidence design, serves the per-class targets worse.

## The estimands

For a task with N labelled windows, map class `m_i` and reference class `r_i`, both in `{0..C−1}`:

- user's accuracy `UA_c = #{m = c, r = c} / #{m = c}`
- producer's accuracy `PA_c = #{m = c, r = c} / #{r = c}`
- error-adjusted share `A_c = #{r = c} / N`, against the map's own share `#{m = c} / N`
- overall accuracy `#{m = r} / N`, already exp78's estimand in its complement.

Truth is read from the whole population; labels are revealed only for sampled windows.

## Estimators, one implementation (`oe_inferencex.estimate.estimate_per_class`)

Under a **random** sample of B windows: `UA_c` is the Wilson interval with finite-population correction on the
labelled windows the map calls *c*, because those are a simple random sample of that class's windows; `A_c` is
the post-stratified estimator by map class (Olofsson et al. 2014, eq. 4 and 5); `PA_c` follows their eq. 7. Under
the **confidence** design (margin quintiles, Neyman for the overall rate) a class cuts across strata, so every
quantity is a ratio of Horvitz–Thompson totals with the linearised variance (Cochran 1977, §6.11), and `A_c` a
Horvitz–Thompson total. Wald intervals at 1.96 for the ratio forms, as the field uses; an interval on fewer than
30 labelled windows is reported with a warning, and the Monte Carlo grades coverage only where that condition
holds, since that is when the package reports it without one. Tile samples are refused by the function: the
per-class cluster form is a later run. All of it is checked by enumeration and against an independent transcription
of Olofsson's equations in `tests/test_per_class.py` before this page was written.

## Design

Tasks: the 17 classification tasks now, from exp78's export, which carries the reference class per unit; the 7
segmentation tasks from exp79's seed-0 export as soon as it lands (exp78's segmentation export wrote no reference
class; the disclosed facts for those tasks are appended below before their run). Designs: random and confidence.
Budgets B ∈ {300, 1000}, headline 300; tasks with N ≤ B skip that budget. R = 2,000 draws, seed 0 to 1,999.

## Disclosed population facts (classification), what they fix in advance

Computed from labels alone before this page was written. `disc/SE` is the population gap between the map's
class share and the true share, in units of the expected standard error of a share estimate at B = 300 under a
random draw with the finite-population correction; it decides P3's cells before any estimator runs.

| task | N | C | smallest map share | classes expected under 30 labels at B = 300 | classes with disc ≥ 3 SE | classes with disc < 1 SE |
|---|---|---|---|---|---|---|
| EuroSAT | 1000 | 10 | 0.097 | 3 | 0 | 10 |
| Brick Kiln | 999 | 2 | 0.330 | 0 | 0 | 2 |
| ForestNet | 993 | 12 | 0.015 | 8 | 0 | 4 |
| So2Sat | 986 | 17 | 0.024 | 17 | 1 | 7 |
| AWF Landsat / S1 / S2 | 200 | 9 | 0.005–0.010 | (N < 300: no headline cell) | — | — |
| Nandi Landsat / S1 / S2 | 825 | 6 | 0.105–0.124 | 0 | 1 / 2 / 0 | 0 / 1 / 2 |
| CropHarvest Togo, three arms | 306 | 2 | 0.39–0.47 | 0 | 2 each (a 98% census makes any gap many SEs) | 0 |
| CropHarvest China, three arms | 4397 | 2 | 0.16–0.24 | 0 | 2 each | 0 |
| BreizhCrops | 122614 | 9 | 0.000 | 5 | 1 | 5 |

So at 300 labels the 17-class So2Sat cannot report a single class without a warning, ForestNet reports four of
twelve and BreizhCrops four of nine; that is the CEOS point that the budget scales with the number of classes,
and it is recorded as such rather than hidden by averaging.

## Predictions

**P1, every interval the package reports without a warning is honest.** On every (task, design, budget, class,
quantity) cell where the class holds at least 30 labelled windows on at least half the draws, the Monte Carlo
coverage over those draws is at least **0.93** and the mean estimate is within **2%** of the truth (bias ratio in
[0.98, 1.02]). *Why 0.93:* exp78's bar, four standard deviations below nominal at R = 2,000. *Why the bias clause:*
the ratio estimators are consistent, not unbiased; at 30 labels a ratio's bias is of order 1/n, so 2% is the
tolerance the estimator earns, and a larger bias is a defect. *What makes it fail:* the linearised variance
under-covering for a class concentrated in one or two strata, where a stratum's within-class count is small even
when the class total passes 30; Wald on a producer's accuracy near 1. **A P1 failure is graded as a defect of the
named cell, and the package warns for that case before the result is recorded.**

**P2, the design that serves the overall rate serves the classes worse.** At B = 300, the median over classes of
the ratio of the confidence design's user's-accuracy half-width to the random design's exceeds **1** on at least
**half** of the tasks with both designs. *Why:* Neyman for the overall rate sends labels to the low-confidence
strata; a class whose windows are confident is then under-labelled and its user's accuracy widens. This is
Stehman and Wagner's warning, and the bar is the weakest one that would still contradict it if it failed. *What
makes it fail:* classes spread evenly across the margin strata, in which case the confidence design costs the
classes nothing and the tool can keep it as the default for both purposes.

**P3, the adjusted share moves exactly where the population says it should.** At B = 300 under the random design,
per class: where the population gap between the map's share and the true share is at least **3** expected
standard errors, the 95% interval of the error-adjusted share excludes the map's share on at least **80%** of
draws; where the gap is below **1** standard error, on at most **20%**. *Why:* a gap of 3 SE is detected with
power about 0.85 by a two-sided 95% interval, and a gap under 1 SE with at most about 0.17; the bars sit inside
those values. *What makes it fail:* a post-stratified variance that is too small (the FPC applied twice, or a map
class with no labels dropping its weight silently), which shows as exclusion far above 20% on the null cells.

**Descriptive.** Per task: the number of classes reported without a warning at 300 and 1,000 labels; per-class
half-widths; the per-class bias ratio; the overall-accuracy coverage beside exp78's.

## Amendment, 23 September 2026, after the first run on the classification tasks and before the rerun

The first run graded the Wald intervals this page promised. **P1 failed on 77 of 324 cells**, with coverage as
low as 0.30, and the failures have three causes, two of them this page's and one the field's:

1. **Wald collapses to a point.** Where no sampled window of a class is wrong, the linearised or post-stratified
   variance is zero and the interval is [1, 1]; a class that is 98% right is then "covered" on no draw. 34 of the
   77 cells sit at a truth beyond 0.95 or below 0.05. The package already handles the same collapse for the
   overall stratified rate with Wilson on the effective sample size (Korn and Graubard 1998); the per-class
   estimators now use the same form by default, with the class's labelled count as the fallback size when the
   estimated variance is zero. The Wald form stays available and **both are rerun and recorded**: the Wald numbers
   are the measurement of the field's convention at these budgets, the Wilson numbers grade what ships.
2. **Wilson's discreteness on EuroSAT.** Six random-design user's-accuracy cells at 0.63–0.69: each class holds
   100 windows with one error, and 30 labels either see it (30% of draws, interval up to 0.988, truth 0.990 not
   covered) or not. The exact coverage Wilson can achieve at (N = 100, K = 1, n = 30) is 0.70 by enumeration, and
   the Monte Carlo sits there. exp79's P5 rule applies: such a cell passes when its coverage is within 0.02 of the
   exact achievable value; the cells are listed in the record as discreteness, not as a defect.
3. **A grading artefact of this page.** Bias was measured over the draws where the class held at least 30
   labels; for a class expected to hold exactly 30, that conditions on being over-sampled and manufactures an
   upward bias of 3–11% with coverage above nominal (12 cells, all reference shares of classes near the 30-label
   line). Bias is now measured over every draw, unconditionally; coverage stays conditional, since that is when
   the interval is reported. The conditional ratio is kept in the artifact beside it.

P2 and P3 are graded on the amended interval form without change of bar. The first run's P2 (4 of 14 tasks
wider) and P3 (3 null cells at exclusion 0.25–0.29) are kept in the record as the first reading.

## Second amendment, 23 September 2026, from the independent audit of the rerun, before the third run

The audit (its own transcription of the estimators, its own Monte Carlo with its own seeds) confirmed every
estimator and every failing cell, and found two things that would have made the record wrong:

1. **The design-comparison grader read the Wald cells.** `grade_p2` kept the last cell per design, which was the
   Wald one; its "4 of 14" was the first run's number to every digit. On the shipped form it is 3 of 14. Fixed by
   filtering on the interval form; the verdict (fails) does not change, the numbers do.
2. **The producer's accuracy under a random draw lacked the finite-population correction** that the user's
   accuracy and the shares carry, so its interval was over-wide (median coverage 0.983 across the 54 graded cells;
   eight times too wide on a near-census draw) and "P1 holds on the PA cells" would have been a statement about
   slack, not honesty. This page said "eq. 7", and Olofsson's eq. 7 has no correction because their sampling
   fraction is negligible; at 300 of 1,000 windows it is not. Each map-class term now carries `(1 − n_i / N_i)`,
   as its siblings do, and the random-design cells are rerun.

Three grading rules are tightened, each stated with its reason: the share's coverage is graded on every draw,
since conditioning on the reference-class count conditions on the estimate itself (the amendment above removed
that conditioning from bias and missed coverage); a discreteness cell passes when it is within the larger of
0.02 and three Monte Carlo standard errors of the exact achievable coverage at its eligible count (the two
EuroSAT cells still failing sat at 1.75 and 2.8 standard errors, and ten independent blocks put 3 of 50 outside
the fixed 0.02); and P3's "expected standard error" is the post-stratified estimator's own at the expected counts,
not the simple-random one this page used, which is 1.4 to 1.9 times larger and put three cells at 1.0–1.4 true
standard errors where a 95% interval excludes 17–29% of the time, exactly as observed. The simple-random ratio is
kept beside it in the artifact.

**Third run, one selection rule added after it.** Grading the share on every draw for every class swept in
classes the map hardly ever predicts (BreizhCrops never predicts three of nine), whose share estimate rests on a
handful of windows and whose bias ratio near zero is meaningless (0.72 to 1.54 with coverage 0.97–0.99). The
share cells graded are those the package reports without a warning, an expected labelled count of at least 30 in
the reference class; their coverage is over every draw. That leaves 328 cells: 13 below the bar, six
discreteness cells passing by the exact rule.

Two mechanisms behind the nine confidence-design cells at 0.87–0.93, both anticipated on this page and neither a
formula error, are now warnings in the package: a class nearly all of whose windows are labelled (the estimate
takes 22 distinct values on Togo and a normal interval covers 0.884 even with the exact variance), and a class
whose windows sit in confidence strata the overall-rate allocation samples thinly, so its rare errors are not
drawn on 35–70% of draws (Brick Kiln, Nandi Landsat: coverage 0.80–0.90 on those draws, 0.98 on the rest).

## Independent check before anything is recorded

`tests/test_per_class.py`: Horvitz–Thompson totals and their variance unbiased over every stratified sample of a
12-window population; the random-design estimators equal to an independent transcription of Olofsson's equations
4–7 and to the package's Wilson interval; both designs' variance estimators within 15% of the Monte Carlo variance
on a 4,000-window synthetic map with margin-dependent classes, estimates unbiased to three standard errors. After
the run, an adversarial read of the implementation on the real files by a separate agent.

## What would invalidate the run

- Truth read from the sample.
- Coverage reported over all draws, including those where the package would have warned, so that a small class
  hides a defect or manufactures one.
- Treating a tile sample as random: refused by the function.

## Cost

No cluster for the classification tasks; the segmentation tasks wait for exp79's export. Minutes per task on
numpy. Ships as `oe-inferencex estimate --per-class` after P1 holds.

# Methodology redesign against the field's rules (23 September 2026)

How the field examines the inference of foundation vision models, what rules it has settled on, where this
record follows them, and what changes follow. Written from a five-angle literature sweep run on the night of
22–23 September (59 papers, 2021–2026, abstracts and method sections; the full list with what each examines is
kept beside this page as `exp/out/methodology_sweep_2026-09-23.json`) and a synthesis against the record. The
sweep's angles: evaluation protocol for failure detection and probing; label-free accuracy estimation; label-
efficient statistically valid evaluation; examination of Earth-observation foundation models; systematic error
discovery and auditing frameworks.

The standing filters apply to everything below: a proposal must tell a map user something they cannot know today,
stay inside the audit (labels grade or calibrate, nothing trains), and have headroom. Two premises the sweep
confirmed rather than shook: plain confidence is the field's default winner for ranking a model's own errors
(Jaeger 2023; Claros Olivares & Brockmeier 2025; Steinmetz 2026), and no single score dominates across tasks. The
record's 24-task, 16-encoder result is the Earth-observation instance of that finding, not an exception to it.

## 1. The field's rules, and where the record stands

Forty-two rules were extracted; four the record follows in full, seventeen do not apply to a label-free audit of a
frozen probe, and twenty-one are followed partly or not at all. The twenty-one, grouped by what they would change:

**Rules about how a ranking claim is stated.**

| rule | set by | status | what is missing |
|---|---|---|---|
| Report the confound between failure-detection score and classifier accuracy explicitly | Jaeger 2023; Steinmetz 2026 | partial | excess AURC is the statistic, but the correlation of the margin's AUROC with probe accuracy over the (task, encoder) cells is never reported |
| Control multiplicity when many rankers are raced; treat the winning family as a property of the representation | Claros Olivares & Brockmeier 2025 | partial | one-sided sign tests per alternative, no family-wise control; Holm over the nineteen tests changes no verdict (smallest p 1.1e-4 against 0.0026) and should be on the page |
| Report geospatial results within capability groups (sensor, resolution, temporality) with seed repetition | Simumba et al. 2025 (GEO-Bench-2); Ramos-Pollán et al. 2024 | partial | seeds: exp79 tonight; groups: only classification against segmentation, no sensor or split-type table |
| Splits must be region-disjoint and results reported per class, because foundation models collapse to frequent classes under regional shift | Shang et al. 2026 | partial | split type is not on the task cards; exp70 holds train–test gaps (So2Sat 0.26, ForestNet 0.23) that mark the shifted ones |
| Validate an uncertainty method by downstream use, ablate aggregation separately, grade calibration beside ranking | Kahl et al. 2024 (ValUES) | partial | downstream uses graded (review order at budgets, estimation); aggregation ablated (exp76); calibration of the margin never graded on the suite |
| Robustness is the drop from each model's own clean baseline under physically grounded product degradations | Faruk et al. 2026 (RSPDBench); Lehmann et al. 2026 | not done | every grade is on clean inputs; perturbations appear only as candidate signals |
| Grade misclassification and out-of-support inputs jointly | Li et al. 2026 | not done | no out-of-support stratum exists in Ai2's splits |

**Rules about what a map user is owed.**

| rule | set by | status | what is missing |
|---|---|---|---|
| A map user is owed user's accuracy, producer's accuracy and error-adjusted area per class with standard errors | Stehman & Wagner 2023; CEOS LPV protocol 2025; Olofsson 2014 | not done | `estimate` returns one error rate with an interval; no per-class quantity anywhere |
| Plan the sample size backwards from a target precision | Adebayo et al. 2026 | not done | the tool grades a given budget; exp78's half-widths at three budgets are the ingredients of a planner |
| State the reference error rate and bound the estimate for it; bias grows for rare classes | Foody 2024; CEOS 2025 | partial | references treated as fallible in exp23 and exp68; `estimate` treats the reviewer's labels as exact |
| Say which outputs are an accuracy assessment in the protocol's sense; a confidence layer is not one | CEOS LPV 2025 | partial | Usage says `estimate` needs labels; it does not say in the protocol's words that the rankers are review-ordering aids |
| Beyond an estimate with an interval, give a threshold with a guaranteed risk bound | Angelopoulos et al. 2022, 2021; Bates et al. 2021 | **done tonight** | exp80, below |
| Use predictions on the unlabelled units through a debiased difference estimator with a tuned coefficient, so the interval is never wider than the classical one | Angelopoulos, Duchi & Zrnic 2023 (PPI++); Fisch et al. 2024 | not done | exp78's plan lists the arm E3d; it was never run, and no document claims it was |
| A label-free accuracy estimate is graded by its error against the truth across many (model, test set) pairs and shipped with its failure mode (prevalence shift) | Garg et al. 2022 (ATC); Guillory et al. 2021; Lu et al. 2023; Fluehmann et al. 2025 | not done | the tool refuses the question; the per-unit exports show why: the map's mean confidence overstates its accuracy by a median 0.061 and up to 0.184 on the suite |
| Estimate the confusion matrix, not one accuracy, when users need per-class numbers | Fluehmann et al. 2025 | not done | same as the first row of this table |
| With a family of models, unlabelled agreement lies on the accuracy line; random head initialisation is the diversity that works | Baek et al. 2022; Saxena et al. 2024 | not done | the ten-seed family exists after tonight, but only seed-0 decisions are exported |
| Detect deployment shift label-free from the frozen feature space, graded against an oracle-labelled shift | Ekim et al. 2024 (TARDIS) | not done | no labelled shift stratum exists to grade it against; parked |

**Rules about a defensible "why".**

| rule | set by | status | what is missing |
|---|---|---|---|
| A coherent slice is not a verified hypothesis; the named group must show elevated error on held-out data | Johnson et al. 2023 | partial | cue enrichments were measured on one testbed with clustered intervals and are quoted on every map |
| Enumerate attribute conjunctions with a per-cell statistic and a noise model for the attribute | Chen, Zhao & Xu 2025 (HiBug2) | partial | cues are reported one at a time |
| In segmentation the systematic error is a (predicted, reference) confusion pair on a coherent sub-group | Singh et al. 2024 | not done | no confusion-pair report in `explain` or `compare` |
| Report false-confidence concentration as a property | Cenacchi et al. 2026 | not done | its ingredients each lost as rankers (exp73); the spatial analogue is exp78's design effect |

## 2. What changes, in order

Each item passes the three filters or says which it fails. Cost is stated in what it needs, not in hours.

1. **A trusted zone with a guarantee (exp80, run and graded tonight; `docs/plan/trust_zone.md`).** The record
   said "an accuracy needs a coverage" and could not give one. Now, from 300 random labels, the tool certifies the
   largest most-confident share of a map that is wrong at most α of the time, with the statement failing on at
   most δ of draws. Two rules: a monotone-prefix rule (Bates et al. 2021) and an assumption-free Bonferroni rule
   (Angelopoulos et al. 2021), both with exact hypergeometric tests, beside the plug-in a reviewer would use
   unaided. Ships as `estimate.certify_zone` and `oe-inferencex certify` once the audit clears it.
2. **Per-class accuracy assessment from the same sample (exp81, preregistration next).** User's accuracy,
   producer's accuracy and error-adjusted area per map class with design-based standard errors, from the sample
   `sample` already draws; a class-aware design beside the confidence design; and the budget a target precision
   needs. This is the field's definition of an accuracy assessment; the tool has been answering a narrower
   question. Needs per-unit reference labels, which exp79's exports carry (exp78's segmentation exports do not).
3. **Where the lead holds (exp82).** The headline re-tabulated by sensor, task family, class count, unit count and
   split type; the accuracy confound as a correlation with an interval; Holm across the raced alternatives; the
   spread of the margin's lead across tile blocks. Local, from committed artifacts and exp79's seeds. It qualifies
   the claim rather than changing it, and a user can locate their own case in it.
4. **A verification rule for `explain`.** Every cue's enrichment recomputed per task on held-out units with a
   clustered interval; the library value quoted only where it survives; confusion pairs for multi-class maps.
   Needs the exports' decisions and grids; local.
5. **The model-assisted arm (PPI++ / stratified PPI) in exp78's harness.** Bounded by exp78's own ceiling (no
   saving above 1.8× against a random sample), so its value is precision and a record that says what was run. Local.
6. **A label-free error-rate estimate with its failure mode stated (ATC, DoC).** The sweep's rule is to grade it
   by error against the truth across many cells and to ship it as a pilot, never as the assessment. The honest
   fact tonight is the calibration gap above; whether a fitted threshold repairs it needs the probe's validation
   split outputs, which the exports lack. One cluster job per encoder when the queue allows.

Not taken, with the one decisive reason each: degradation stress axes (need re-encoding under product defects);
joint out-of-support grading (no such stratum in the suite); TARDIS-style shift flags (nothing labelled to grade
against); agreement-on-the-line (needs all ten seeds' decisions exported; queued behind the above); embedding-
geometry diagnostics (no route to a statement a user acts on, and exp73/exp77 say feature statistics carry no
error information here); sequential and active designs (exp78's confidence design already sits within 0.0007
half-width of the oracle allocation, so there is no allocation headroom to claim); label-free score aggregation
(another ranker; exp65's fusion gained 0.0006); training-data influence (a developer's question).

## 3. What tonight's runs say so far

exp80 on OlmoEarth Base, 24 tasks, 2,000 draws per cell, δ = 0.1: the guarantee held on every one of the 112
cells (largest violation frequency 0.080 for the prefix rule, 0.0045 for Bonferroni, against a bound of 0.120),
including the thirteen tasks where the prefix rule's monotonicity assumption is broken by up to 0.034. At 300
labels and α = half the map's error rate, the prefix rule certified a zone on most draws on 14 of the 21 tasks
large enough for that budget, median certified coverage 0.50 against an oracle median of 0.60; the plug-in a
reviewer would use violated its own α on up to 56% of draws. Two secondary predictions failed as written and are
recorded as failures: the expected-count arithmetic disagreed with the run on the two tasks where 300 labels are a
98% census, and the plug-in's violation count fell short of the preregistered 18 because the bar was written for
24 tasks while only 21 have a 300-label cell. The independent audit of the implementation was running when this
page was written; nothing above enters the ledger until it returns.

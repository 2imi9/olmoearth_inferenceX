# Technique ledger

Standing record of what was tried and what survived, organized by technique
rather than by date. A claim appears here only when an experiment supports
it, and is amended or removed when a later experiment contradicts it.

The repository has two applications: **error ranking** (which windows is the
model getting wrong?) and **cross-inference evaluation** (of two runs,
scorers or explanations, which should you believe?). Most tables below serve
the first; the cross-inference index is [further down](#cross-inference-comparisons).

Each row carries a one-line verdict and a pointer. The evidence behind a
verdict lives in exactly one place: per-signal detail in
[results/signals.md](results/signals.md), per-experiment detail in
[results/comparisons.md](results/comparisons.md). Scoring rules, evidence
tiers and status terms are defined once in
[method/protocol.md](method/protocol.md).

## Documentation map

| Document | What it holds |
|---|---|
| [index.md](index.md) | Documentation landing page |
| [Findings.md](Findings.md) | What holds, the numbers, how a claim gets in, the limits |
| [Usage.md](Usage.md) | The package: quick start, the production case, scoring a new rule |
| [method/protocol.md](method/protocol.md) | How results are scored, evidence tiers, status terms, related work |
| [method/recipe.md](method/recipe.md) | What to do and not do when auditing a prediction map |
| [method/taskcards.md](method/taskcards.md) | What each fine-tuned model is, resolved from its configs |
| [method/infrastructure.md](method/infrastructure.md) | Upstream sources, export formats, encoder internals |
| [method/agent_integration.md](method/agent_integration.md) | The contract with the OlmoEarth Agent |
| [results/comparisons.md](results/comparisons.md) | Cross-signal comparisons, one section per experiment |
| [results/signals.md](results/signals.md) | Per-signal evidence, one section per signal |
| [results/explanation.md](results/explanation.md) | Why a window is suspect: cue enrichment on identical windows (exp37) |
| [plan/roadmap.md](plan/roadmap.md) | Open items in priority order |
| [../exp/NOTES.md](../exp/NOTES.md) | Chronological lab log, including superseded runs |

## Signals: what ranks errors

Scored against the model's own confidence and a no-model pixel control.
"26/27" means better on 26 scenes of the 27 rule-selected set.

| Signal | LLM-domain analogue | Status | Verdict | Evidence |
|---|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | **supported** | Best error-ranker on every expert-labelled testbed; loses only against the WorldCover reference | exp04, exp16, exp18, exp21 |
| Perturbation stability (E_system, tile-phase) | sampling-consistency | **mixed** | Beats confidence 26/27 against WorldCover and ties it on the fine-tuned model, but not on hand labels (163/187, p=0.22) | exp13, exp18, exp21 |
| Prediction-boundary proximity | n/a (EO-specific) | **supported as triage** | 75% of error patches sit on boundaries vs 20% of correct ones, on both references; says where errors live, does not order them better than confidence on AURC; at a 5% review budget on hand labels it captures 0.286 of the errors against confidence's 0.259 (secondary, reverses by 20%) | exp14, exp16, exp18, exp35 |
| Boundary first, then confidence (review order) | n/a (EO-specific) | **supported at tight budgets** | Review boundary windows in confidence order, then the interior: captures 0.274 and 0.494 of the errors at the 5% and 10% budgets on hand labels against 0.259 and 0.465 for confidence (preregistered, per tile 85/31/235 and 112/48/191, one-sided p = 3e-7 and 2e-7; pooled CIs above zero), no longer ahead at 20%, identical to confidence on the fine-tuned model at 5% (its least-confident windows are boundary windows); worse than confidence on AURC, so a triage rule, not a ranker | exp36 |
| Spectral ambiguity first, then boundary, then confidence (review order) | n/a (EO-specific, water task) | **mixed** | Preregistered against the boundary-first order on hand labels: per-tile budgets win at 5% and 10% (151/100/100, p = 8e-4; 146/86/119, p = 5e-5), one pooled budget does not (0.292 vs 0.274, CI touching zero; 0.481 vs 0.494); clear only at 20% pooled (0.831 vs 0.732); loses to boundary-first on the WorldCover rivers (2/6, 3/5, 3/4); the cue was chosen after seeing its exp37 enrichment on the same tiles | exp38 |
| Band-set disagreement (one model, three S2 band-set tokens) | self-consistency across input views | **mixed** | 21/27 against WorldCover from a single forward pass; worse than confidence on hand labels (111/239, p=7e-12) | exp17, exp18 |
| Depth-probe disagreement | layer-wise probing | **partial** | 19/27 vs baseline (p=0.052), 13/27 vs control | exp17 |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE) | **not supported** | No scale-free advantage (13/27, sign p=1.00); exp31 generalised it to three reference sets and three density estimators with the same result | exp13, exp31 |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | **partial** | Flags are 1.5x error-enriched but mostly mark OSM-vs-WorldCover disagreement on narrow channels; poor as a ranker (3/27) | exp02, exp15 |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | **not supported** | 10/27 against WorldCover, 84/265 on hand labels; a useful partner needs a different input view, not higher accuracy | exp10, exp13, exp17, exp18 |
| Backbone-version disagreement (v1 vs v1.2) | n/a (EO-specific) | **not supported** | Worse than confidence for v1's errors (6/21), n.s. for v1.2's; RoPE does not reduce sub-patch tiling instability | exp19 |
| Internal-state signals (logit-lens settling, representation drift, attention entropy) | INSIDE / hidden-state probing | **rejected** | 0/27, 3/27, 3/27 vs baseline; these do not transfer from language models | exp17 |
| Masking perturbation | occlusion sensitivity | **rejected** | Worst signal on every scene tested; occlusion measures context reliance, not error likelihood | exp08 |
| Dihedral consistency (std of the probability map over the 8 flips and rotations) | test-time-augmentation disagreement | **rejected** | Not better than confidence: 17/10 scenes but 4/4 rivers (p = 0.64) against WorldCover, 154/196 Bolivia tiles, below confidence at every review budget; Spearman with confidence 0.67 to 0.94, a smoothed confidence rather than a tile-phase; the preregistered confidence+dihedral combination is 4/4 rivers and its per-tile edge on Bolivia (201/148) does not survive pooling | exp36 |
| Decoder self-consistency (native masked-token reconstruction error) | the latent-MIM pretraining objective at inference | **rejected** | Loses to confidence (7/20 scenes, 2/6 rivers) and to both observed-input controls; the preregistered confidence+decoder combination gains nothing (4/4 rivers, p = 0.64); the frozen targets are near-collinear (within-scene pairwise cosine median 0.994 over 27 scenes) | exp28 |
| Last-layer Laplace / bootstrap variance of the probe head | Bayesian last-layer / deep-ensemble epistemic uncertainty | **rejected** | Logit variance is the worst signal on both testbeds (3/24 scenes, 0/8 rivers; 6/345 Bolivia tiles) and rises with the logit size; the probit-moderated confidence is confidence itself (Spearman 1.00 on Bolivia, 0.97 on the scenes); the preregistered confidence+variance combination loses 0/8 rivers (p = 1.0) | exp30 |
| Feature-space typicality (kNN, Mahalanobis, PCA residual, ViM against training, same-scene and cross-testbed references) | feature-density / OOD scoring (kNN, Mahalanobis, ViM) | **rejected** | No score beats confidence on either testbed (best 13/14 scenes for kNN to the training scene; every score loses on at least 317/351 Bolivia tiles); the preregistered confidence+typicality combination reaches 6/2 rivers (p = 0.145) and hurts on hand labels (60/291) | exp31 |
| Re-targeted latent-MIM residual (whitened target, small predictor on the frozen encoder) | latent MIM at inference with a normalised target | **rejected** | The target swap makes the objective predictable (57-70% of the whitened variance from context, where the shipped decoder was at chance) but the residual tracks input texture (Spearman 0.56 with S2 variance), loses to confidence 2/6 rivers and 12/339 Bolivia tiles; U+ 6/2 rivers (p = 0.145) | exp33 |
| Re-targeted latent-MIM objective, other readings (discrete target NLL and predictive entropy, gap masking, decision-direction residual) | HuBERT-style targets, Latent MIM gap masking | **rejected** | The entropy combination loses 0/8 rivers (p = 1.0) and 60/290 Bolivia tiles; the gap-masked residual reaches parity with confidence on WorldCover (13/14, 4/4 rivers) while still tracking texture, and loses 22/329 on hand labels; with exp28 and exp33 this closes the family for this checkpoint | exp34 |
| Label-free reliability (Dawid-Skene) | annotator modeling | **rejected within family** | Inflates every model and inverts the ordering, because family members err together; the inflation gap measures correlated-error mass | exp07 |

## Deployed artifacts: what can be audited from outside

| Target | Status | Verdict | Evidence |
|---|---|---|---|
| Fine-tuned model end to end (`OlmoEarth-v1-FT-AWF-Base`) | **supported** | Replica reproduces the reported accuracy (0.881 vs 0.895); confidence ranks its errors best (AURC 0.0262), tiling instability is indistinguishable (0.0235, CI spans zero), everything else significantly worse. Overconfident: ECE 0.080 | exp21 |
| Served LCC rasters, boundary triage | **supported on a weak reference** | The product exports no class confidence, so the baseline cannot run; boundary fraction captures a median 0.88 of WorldCover water disagreements at a 5% review budget | exp20 |
| Served LCC rasters, periodic artifacts | **lattice found, seams not** | Outputs are quantized to the encoder's 4-px patch lattice (19 of 20 profiles, p <= 6e-08); no inference-window seams at 64-512 px, at the null rate. Detection limit 5-10% of rows at 128 px | exp22 |

## Cross-inference comparisons

The second application: two inference runs scored on identical errors under
the same tests. Evidence lives in the experiment sections of
[results/comparisons.md](results/comparisons.md); this is the index.

| Compared | Finding | Evidence |
|---|---|---|
| Nano / Tiny / Base votes | Family members err together, so label-free accuracy estimation inflates every model and inverts the ordering. The inflation gap measures correlated-error mass | exp07 |
| v1-Large vs Nano as Base's partner | A *stronger* partner makes disagreement worse, though Large is more accurate on every scene | exp10 |
| Three band-set probes vs the final head | The most error-correlated rater (phi 0.75) yields the *best* disagreement signal — a different view of the input beats decorrelation | exp17 |
| v1 vs v1.2 on identical scenes | RoPE does not reduce sub-patch tiling instability (larger on 25 of 31 scenes); cross-version disagreement is not a useful signal | exp19 |
| Fine-tuned model vs frozen probe | 28 of the fine-tuned model's 41 errors are also probe errors; probe disagreement ranks errors significantly worse than confidence | exp21 |
| Same scenes, 2021 vs 2024 imagery | The instability advantage persists within the year and is not larger in 2024 | exp24 |

**What generalizes.** A second run is informative when it sees the input
differently, not when it is more accurate — and never when it shares a
family's failure modes. Agreement within a family is not evidence of
correctness.

## Reference audits: why do the WorldCover wins not transfer?

The adjudication case: three competing explanations for one discrepancy,
each pre-registered, each tested on its own artifact. Tiling instability
beats confidence 26/27 against ESA WorldCover but not against hand labels
(exp18); three measurable components of reference-versus-image mismatch were
removed one at a time. **None explains the gap.**

Each uses the same design — T1 enrichment (are the suspect patches
over-represented among disagreements?), T2 mechanism (does the signal flag
them preferentially?), T3 exclusion (does the advantage survive removing
them?).

| Candidate explanation | Test | Result | Evidence |
|---|---|---|---|
| The reference is unstable | WorldCover 2020 vs 2021 disagreement | Real but small: covers ~10% of disagreements; on reference-stable patches tiling instability still wins 21/23 | exp23 |
| The imagery post-dates the map | 2021 imagery, 2021 map, head retrained within 2021 | Advantage persists, 23/26; the per-scene gain is not larger in 2024 | exp24 |
| Seasonal water an annual map cannot represent | JRC Global Surface Water seasonality removed from scoring | Seasonal margins do hold 39% of disagreements vs 8% of agreements, but with them removed the advantage survives, 22/24 | exp25 |

What remains: reference error shared by both WorldCover versions, or a
genuine property of the WorldCover-defined task that date-matched hand
labels do not share. Open.

## Method and harness

| Practice | Status | Note | Evidence |
|---|---|---|---|
| Risk-coverage / AURC harness | **supported** | Pre-registered 27-scene set, tie-aware AURC, block bootstrap, exact sign and permutation tests | exp13 |
| No-model pixel-statistic controls | **supported** | Ran on every comparison scene; killed one claim (E_dist under shift) and confirmed two | exp06 |
| Labels grade signals, never train them | **supported** | Executed with AWF expert labels under the project's own spatial split | exp04, exp16 |
| Operating points at fixed review budgets | **checked** | Capture at 5, 10 and 20% budgets does not change the AURC verdicts on expert labels: the preregistered test (tiling instability at 20% on Bolivia) is null; boundary at 5% is the one qualified exception | exp35 |
| Explanation layer: label-free cues with measured enrichment (`oe_inferencex.explain`) | **built** | Five cues measured on identical windows: on hand labels 95% of error windows carry at least one (boundary 3.5x, least confident 20% 3.6x, unstable 3.6x, NDWI-ambiguous 7.2x, flip/rotation disagreement 3.5x); inside confidence's review set only the boundary and NDWI cues separate error rates (0.46 vs 0.16, 0.51 vs 0.33 at 5%); 5% of the errors carry no cue | exp37 |

## Not yet tried

| Item | Status | Note |
|---|---|---|
| Semantic-entropy port (cluster-then-entropy, Farquhar et al. 2024) | untested | Possible refinement of the perturbation signal; issue #5 |
| Verifier head trained on labeled regions | out of scope (v1) | Requires labels as training input |
| Channel fusion | out of scope (v1) | Per-channel reporting only; nothing yet beats confidence, so there is nothing to fuse |

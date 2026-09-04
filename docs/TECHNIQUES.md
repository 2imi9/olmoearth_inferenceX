# Technique ledger

Standing record of what was tried and what survived, organized by technique
rather than by date. A claim appears here only when an experiment supports
it, and is amended or removed when a later experiment contradicts it.

Each row carries a one-line verdict and a pointer. The evidence behind a
verdict lives in exactly one place: per-signal detail in
[results/signals.md](results/signals.md), per-experiment detail in
[results/comparisons.md](results/comparisons.md). Scoring rules, evidence
tiers and status terms are defined once in
[method/protocol.md](method/protocol.md).

## Documentation map

| Document | What it holds |
|---|---|
| [method/protocol.md](method/protocol.md) | How results are scored, evidence tiers, status terms, related work |
| [method/recipe.md](method/recipe.md) | What to do and not do when auditing a prediction map |
| [method/taskcards.md](method/taskcards.md) | What each fine-tuned model is, resolved from its configs |
| [method/infrastructure.md](method/infrastructure.md) | Upstream sources, export formats, encoder internals |
| [method/agent_integration.md](method/agent_integration.md) | The contract with the OlmoEarth Agent |
| [results/comparisons.md](results/comparisons.md) | Cross-signal comparisons, one section per experiment |
| [results/signals.md](results/signals.md) | Per-signal evidence, one section per signal |
| [plan/roadmap.md](plan/roadmap.md) | Open items in priority order |
| [../exp/NOTES.md](../exp/NOTES.md) | Chronological lab log, including superseded runs |

## Signals: what ranks errors

Scored against the model's own confidence and a no-model pixel control.
"26/27" means better on 26 scenes of the 27 rule-selected set.

| Signal | LLM-domain analogue | Status | Verdict | Evidence |
|---|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | **supported** | Best error-ranker on every expert-labelled testbed; loses only against the WorldCover reference | exp04, exp16, exp18, exp21 |
| Perturbation stability (E_system, tile-phase) | sampling-consistency | **mixed** | Beats confidence 26/27 against WorldCover and ties it on the fine-tuned model, but not on hand labels (163/187, p=0.22) | exp13, exp18, exp21 |
| Prediction-boundary proximity | n/a (EO-specific) | **supported as triage** | 75% of error patches sit on boundaries vs 20% of correct ones, on both references; says where errors live, does not order them better than confidence | exp14, exp16, exp18 |
| Band-set disagreement (one model, three S2 band-set tokens) | self-consistency across input views | **mixed** | 21/27 against WorldCover from a single forward pass; worse than confidence on hand labels (111/239, p=7e-12) | exp17, exp18 |
| Depth-probe disagreement | layer-wise probing | **partial** | 19/27 vs baseline (p=0.052), 13/27 vs control | exp17 |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE) | **partial** | No scale-free advantage (13/27, sign p=1.00); as implemented it measures distance to the head's training scene, not the pretraining distribution | exp13 |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | **partial** | Flags are 1.5x error-enriched but mostly mark OSM-vs-WorldCover disagreement on narrow channels; poor as a ranker (3/27) | exp02, exp15 |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | **not supported** | 10/27 against WorldCover, 84/265 on hand labels; a useful partner needs a different input view, not higher accuracy | exp10, exp13, exp17, exp18 |
| Backbone-version disagreement (v1 vs v1.2) | n/a (EO-specific) | **not supported** | Worse than confidence for v1's errors (6/21), n.s. for v1.2's; RoPE does not reduce sub-patch tiling instability | exp19 |
| Internal-state signals (logit-lens settling, representation drift, attention entropy) | INSIDE / hidden-state probing | **rejected** | 0/27, 3/27, 3/27 vs baseline; these do not transfer from language models | exp17 |
| Masking perturbation | occlusion sensitivity | **rejected** | Worst signal on every scene tested; occlusion measures context reliance, not error likelihood | exp08 |
| Label-free reliability (Dawid-Skene) | annotator modeling | **rejected within family** | Inflates every model and inverts the ordering, because family members err together; the inflation gap measures correlated-error mass | exp07 |

## Deployed artifacts: what can be audited from outside

| Target | Status | Verdict | Evidence |
|---|---|---|---|
| Fine-tuned model end to end (`OlmoEarth-v1-FT-AWF-Base`) | **supported** | Replica reproduces the reported accuracy (0.881 vs 0.895); confidence ranks its errors best (AURC 0.0262), tiling instability is indistinguishable (0.0235, CI spans zero), everything else significantly worse. Overconfident: ECE 0.080 | exp21 |
| Served LCC rasters, boundary triage | **supported on a weak reference** | The product exports no class confidence, so the baseline cannot run; boundary fraction captures a median 0.88 of WorldCover water disagreements at a 5% review budget | exp20 |
| Served LCC rasters, periodic artifacts | **lattice found, seams not** | Outputs are quantized to the encoder's 4-px patch lattice (19 of 20 profiles, p <= 6e-08); no inference-window seams at 64-512 px, at the null rate. Detection limit 5-10% of rows at 128 px | exp22 |

## Reference audits: why do the WorldCover wins not transfer?

Tiling instability beats confidence 26/27 against ESA WorldCover but not
against hand labels (exp18). Three measurable components of
reference-versus-image mismatch were removed one at a time. **None explains
the gap.**

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

## Not yet tried

| Item | Status | Note |
|---|---|---|
| Semantic-entropy port (cluster-then-entropy, Farquhar et al. 2024) | untested | Possible refinement of the perturbation signal |
| Verifier head trained on labeled regions | out of scope (v1) | Requires labels as training input |
| Channel fusion | out of scope (v1) | Per-channel reporting only; nothing yet beats confidence, so there is nothing to fuse |

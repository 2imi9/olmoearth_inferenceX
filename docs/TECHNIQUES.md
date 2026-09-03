# Technique ledger

Standing record of experimental results, organized by technique rather than
by date. A claim appears here only when an experiment supports it, and is
amended or removed when a later experiment contradicts it. Every entry cites
the experiment that produced it.

The documentation set:

- [method/recipe.md](method/recipe.md) - what to do and not do, each item
  tied to its experiment.
- [method/taskcards.md](method/taskcards.md) - what each fine-tuned model is
  (task, legend, inputs, windows, outputs, encoder version) resolved from its
  configs by oe_inferencex/taskcard.py, with the audit settings that follow.
- [method/protocol.md](method/protocol.md) - status terms, evidence tiers, related work,
  and the positioning of the contribution.
- [results/comparisons.md](results/comparisons.md) - cross-signal comparisons: the pre-registered
  27-scene study (pre-registered rule; 29 fetched, 2 cache leftovers excluded; authoritative), the controls, the expert-label validation,
  and the earlier scene studies.
- [results/signals.md](results/signals.md) - per-signal evidence for the baseline, E_case,
  E_system, E_dist, and E_geo.
- [method/infrastructure.md](method/infrastructure.md) - established facts: upstream
  sources, export formats, transfer properties.
- [plan/roadmap.md](plan/roadmap.md) - open items in priority order and the
  change-attribution use case.
- [../exp/NOTES.md](../exp/NOTES.md) - chronological lab log.

Evidence tiers: exp18 (Sen1Floods11 hand labels, spatial hold-out) is authoritative for the water task and supersedes WorldCover-referenced comparisons where they conflict; single-scene results (exp01-exp08) establish direction
only; exp09 is a seven-scene comparison with hand-chosen scenes; exp11
fixed the scene set under a pre-registered selection rule (27 scenes after
excluding two cache leftovers); exp13 is the authoritative statistics on that set (aligned tile-phase, excess
AURC, block bootstrap, sign and permutation tests;
exp/out/exp13_corrected_stats.csv). Where earlier experiments disagree with
exp13, exp13 stands.

## Summary

| Technique | Related LLM-domain method | Status | State of evidence |
|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal wherever the truth is expert-labelled: the AWF point task (exp04/16) and the Sen1Floods11 dense masks (exp18: lowest E-AURC on Bolivia and test; every other signal equal or significantly worse). Loses to boundary-type signals only against the WorldCover reference (exp13); the reference-error reading of that loss was tested in exp23 and is not supported by the measurable component (version instability covers about 10% of disagreements; the loss persists on reference-stable patches, 21/23) |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | not supported | worse than confidence on expert labels (Sen1Floods11 Bolivia 84/265; AWF) and not general against WorldCover (10/27, exp13). Partner quality is a different input view, not accuracy (exp10/17) |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | not supported on expert labels | beats confidence 26/27 against WorldCover (exp13) but not against Sen1Floods11 hand labels (Bolivia 163/187, p=0.22; test 173/305, worse; exp18). The WorldCover advantage is not explained by reference-version instability (exp23: persists 21/23 on reference-stable patches); its cause is open (shared reference error, 2021-to-2024 change, or a property of the WorldCover-defined task). Masking variant rejected (exp08) |
| Backbone-version comparison (E_system, v1 vs v1.2) | n/a (EO-specific) | tested, not supported | loaded via an isolated worktree of current main (exp19). Cross-version disagreement is worse than confidence for v1's errors (6/21) and n.s. for v1.2's (18/9). RoPE in v1.2 does not reduce sub-patch tiling instability (larger on 25/31 scenes); v1.2 tokenizes Sentinel-2 as one band-set token per patch |
| Production output assessment (boundary triage on served LCC rasters) | selective prediction on deployed outputs | supported on a weak reference | exp20: ten 512-px windows of allenai/olmoearth_lcc; the product exports no class confidence, so the confidence baseline cannot run; boundary fraction ranks water disagreements with WorldCover below random at 6 of 6 sites with water and captures 0.88 of them at a 5% budget; change-probability ambiguity sits on flagged-region edges |
| Periodic-artifact test on served rasters (striping, seams, patch lattice) | n/a (EO-specific) | patch-lattice quantization supported; window seams not detected | exp22: whitened periodograms of boundary and gradient profiles on 5 4096-px windows with predicted pipeline periods, a WorldCover control and in-situ injected seams; 19 of 20 product profiles peak on the 4-px patch lattice (p <= 6e-08); inference-window seams at 64-512 px absent at the null rate, detection limit 5-10% of rows at 128 px |
| Reference-instability test (WorldCover 2020 vs 2021) | n/a (reference audit) | reading tested, not supported | exp23 on 24 rule scenes: disagreements are 14x enriched at version-unstable patches (17/3/4, p 0.003) but those cover about 10% of disagreements; tiling instability's flagged errors are not more often unstable than confidence's (8/8/8); on reference-stable patches tiling instability still beats confidence 21/23 (p 7e-05) against 22/23 on all patches |
| Year-gap test (2021 imagery against the 2021 map) | n/a (reference audit) | reading tested, not supported | exp24 on 23 rule scenes with May-September 2021 imagery and a head retrained within 2021: tiling instability beats confidence 20/3/0 (p 5e-04) against 22/1/0 with 2024 imagery on the same scenes; boundary 19/4/0 against 17/6/0; the per-scene gain is not larger in 2024 (9/14/0). Temporal mismatch does not explain the WorldCover-referenced wins |
| Fine-tuned model end to end (confidence, tiling instability, boundary, disagreement, control on allenai/OlmoEarth-v1-FT-AWF-Base) | selective prediction on the deployed model | supported (confidence); tiling instability indistinguishable | exp21: replica reproduces Ai2's accuracy (0.881 vs 0.895) on 344 expert validation points; confidence AURC 0.0262, tiling instability 0.0235 (cluster-bootstrap CI spans zero), boundary 0.0765, probe disagreement 0.0852, control 0.0937; ECE 0.080, overconfident; selective accuracy 0.945 at 80% coverage |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, semantic entropy probes) | partial | no scale-free advantage: 13/27 vs baseline, sign p=1.00 (exp13); the mean-difference permutation p=0.01 is carried by a few high-error scenes. As implemented it measures distance to the head's training scene, not the pretraining distribution. Out-of-distribution-indicator role untested on a valid testbed |
| Band-set disagreement (internal ensemble, E_case within one model) | self-consistency across input views | not supported on expert labels | 21/27 vs confidence against WorldCover (exp17) but worse than confidence on Sen1Floods11 hand labels (Bolivia 111/239, p=7e-12; exp18) |
| Depth-probe disagreement | layer-wise probing | partial | per-block heads, std over the last six blocks: 19/27 vs baseline (p=0.052), 13/27 vs control (exp17) |
| Internal-state signals (logit-lens settling, representation drift, attention entropy) | INSIDE / hidden-state probing | rejected | 0/27, 3/27, 3/27 vs baseline on the rule-selected scenes (exp17) |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | partial | zero false break alarms on a well-resolved scene (exp02); on 27 scenes, flags are 1.5x error-enriched (pooled precision 0.12 over 413 flags) but mostly mark OSM-vs-WorldCover disagreement on narrow channels (precision 0 on 9 scenes); poor as a ranker (3/27 vs baseline) and prepending it to boundary proximity shows no benefit (5 better, 9 worse, 13 unchanged; exp15). Sensitivity unmeasurable under WorldCover truth |
| Label-free reliability (Dawid-Skene) | annotator modeling | rejected (within family) | inflates every model and inverts the ordering because family members err together; the inflation gap measures correlated-error mass. Viable only with an out-of-family rater (exp07) |
| Risk-coverage / AURC harness | selective prediction | supported | 27-scene pre-registered set with tie-aware AURC, excess AURC, block-bootstrap intervals, exact sign tests on untied pairs and sign-flip permutation tests (exp13) |
| Validation on labeled data (labels grade signals, never train them) | pseudo-label validation (Planetary Prediction Engine tech report, arXiv:2608.26088, §2.3.1) | supported | executed with AWF expert labels under the project's own spatial split; WorldCover remains the reference elsewhere |
| No-model pixel-statistic controls | ablation practice | supported | ran on all comparison scenes; killed one claim (E_dist under shift) and confirmed two (E_case wins are not edge detection) |
| Semantic-entropy port (cluster-then-entropy) | Farquhar et al. 2024 | untested | possible refinement of the perturbation signal |
| Verifier head trained on labeled regions | learned verifiers / reward models | out of scope (v1) | requires labels as training input |
| Channel fusion | n/a | out of scope (v1) | per-channel reporting only; rank aggregation if a single ordering is required |

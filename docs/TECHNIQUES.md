# Technique ledger

Standing record of experimental results, organized by technique rather than
by date. A claim appears here only when an experiment supports it, and is
amended or removed when a later experiment contradicts it. Every entry cites
the experiment that produced it.

The documentation set:

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
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal wherever the truth is expert-labelled: the AWF point task (exp04/16) and the Sen1Floods11 dense masks (exp18: lowest E-AURC on Bolivia and test; every other signal equal or significantly worse). Loses to boundary-type signals only against the WorldCover reference (exp13), which exp18 reads as reference error rather than model error |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | not supported | worse than confidence on expert labels (Sen1Floods11 Bolivia 84/265; AWF) and not general against WorldCover (10/27, exp13). Partner quality is a different input view, not accuracy (exp10/17) |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | not supported on expert labels | beats confidence 26/27 against WorldCover (exp13) but not against Sen1Floods11 hand labels (Bolivia 163/187, p=0.22; test 173/305, worse; exp18). The WorldCover advantage is read as detection of reference error. Masking variant rejected (exp08) |
| Backbone-version comparison (E_system, v1 vs v1_2) | n/a (EO-specific) | blocked | v1_2 checkpoints fail to load with the current olmoearth_pretrain checkout (state_dict mismatch); requires the newer loader |
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

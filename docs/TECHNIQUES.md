# Technique ledger

Standing record of experimental results, organized by technique rather than
by date. A claim appears here only when an experiment supports it, and is
amended or removed when a later experiment contradicts it. Every entry cites
the experiment that produced it.

The documentation set:

- [method/protocol.md](method/protocol.md) - status terms, evidence tiers, related work,
  and the positioning of the contribution.
- [results/comparisons.md](results/comparisons.md) - cross-signal comparisons: the pre-registered
  29-scene study (authoritative), the controls, the expert-label validation,
  and the earlier scene studies.
- [results/signals.md](results/signals.md) - per-signal evidence for the baseline, E_case,
  E_system, E_dist, and E_geo.
- [method/infrastructure.md](method/infrastructure.md) - established facts: upstream
  sources, export formats, transfer properties.
- [plan/roadmap.md](plan/roadmap.md) - open items in priority order and the
  change-attribution use case.
- [../exp/NOTES.md](../exp/NOTES.md) - chronological lab log.

Evidence tiers: single-scene results (exp01-exp08) establish direction
only; exp09 is a seven-scene comparison with hand-chosen scenes; exp11
fixed the scene set - 29 scenes under a pre-registered selection rule; exp13
is the authoritative statistics on that set (aligned tile-phase, excess
AURC, block bootstrap, sign and permutation tests;
exp/out/exp13_corrected_stats.csv). Where earlier experiments disagree with
exp13, exp13 stands.

## Summary

| Technique | Related LLM-domain method | Status | State of evidence |
|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal on the in-domain AWF expert-label task; on the 29-scene rule-selected set it is the best signal on 1/29 once tile-phase is computed correctly (exp13). All channel claims are relative to this baseline |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | mixed | strong on specific ambiguous scenes (Barotse, control-proof) but not general: beats the baseline on 12/29 rule-selected scenes, sign test p=0.46, block-bootstrap interval against it on 10 scenes (exp13). Partner quality is decorrelation, not accuracy (exp10); naive >2-model aggregation hurts (exp03/04/07) |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | supported, reinterpreted | beats the baseline on 27/29 rule-selected scenes (sign p<0.001) and the pixel control on 19/29 (exp13), but a zero-cost boundary indicator from the model's own prediction map matches it head-to-head (13/29, p=0.71) and beats the baseline 23/29 alone (exp14): the mechanism is prediction-boundary proximity, not perturbation. Applies to dense maps, not interior point labels (lost on AWF, exp04). Masking variant rejected (exp08) |
| Backbone-version comparison (E_system, v1 vs v1_2) | n/a (EO-specific) | blocked | v1_2 checkpoints fail to load with the current olmoearth_pretrain checkout (state_dict mismatch); requires the newer loader |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, semantic entropy probes) | partial | no scale-free advantage: 13/29 vs baseline, sign p=0.71 (exp13); the exp11 p=0.019 was a raw-AURC scale artifact from high-error scenes. As implemented it measures distance to the head's training scene, not the pretraining distribution. Out-of-distribution-indicator role untested on a valid testbed |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | partial | zero false break alarms on a well-resolved scene (exp02); on 23 scenes, flags are 3x error-enriched over a random patch but mostly mark OSM-vs-WorldCover disagreement on narrow channels (precision 0 on 8 scenes); poor as a ranker (3/23 vs baseline) and combining it with boundary proximity hurts (exp15). Sensitivity unmeasurable under WorldCover truth; needs width-filtered centerlines or expert labels |
| Label-free reliability (Dawid-Skene) | annotator modeling | rejected (within family) | inflates every model and inverts the ordering because family members err together; the inflation gap measures correlated-error mass. Viable only with an out-of-family rater (exp07) |
| Risk-coverage / AURC harness | selective prediction | supported | 29-scene pre-registered set with excess AURC, block-bootstrap intervals, exact sign tests and sign-flip permutation tests (exp13) |
| Validation on labeled data (labels grade signals, never train them) | pseudo-label validation (PPE §2.3.1) | supported | executed with AWF expert labels under the project's own spatial split; WorldCover remains the reference elsewhere |
| No-model pixel-statistic controls | ablation practice | supported | ran on all comparison scenes; killed one claim (E_dist under shift) and confirmed two (E_case wins are not edge detection) |
| Semantic-entropy port (cluster-then-entropy) | Farquhar et al. 2024 | untested | possible refinement of the perturbation signal |
| Verifier head trained on labeled regions | learned verifiers / reward models | out of scope (v1) | requires labels as training input |
| Channel fusion | n/a | out of scope (v1) | per-channel reporting only; rank aggregation if a single ordering is required |

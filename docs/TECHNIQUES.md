# Technique ledger

Standing record of experimental results, organized by technique rather than
by date. A claim appears here only when an experiment supports it, and is
amended or removed when a later experiment contradicts it. Every entry cites
the experiment that produced it.

The documentation set:

- [PROTOCOL.md](PROTOCOL.md) - status terms, evidence tiers, related work,
  and the positioning of the contribution.
- [RESULTS.md](RESULTS.md) - cross-signal comparisons: the pre-registered
  29-scene study (authoritative), the controls, the expert-label validation,
  and the earlier scene studies.
- [SIGNALS.md](SIGNALS.md) - per-signal evidence for the baseline, E_case,
  E_system, E_dist, and E_geo.
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - established facts: upstream
  sources, export formats, transfer properties.
- [ROADMAP.md](ROADMAP.md) - open items in priority order and the
  change-attribution use case.
- [../exp/NOTES.md](../exp/NOTES.md) - chronological lab log.

Evidence tiers: single-scene results (exp01-exp08) establish direction
only; exp09 is a seven-scene comparison with hand-chosen scenes; exp11 is
the authoritative comparison - 29 scenes under a pre-registered selection
rule, with per-scene bootstrap CIs and permutation tests
(exp/out/exp11_stats.csv). Where exp09 and exp11 disagree, exp11 stands.

## Summary

| Technique | Related LLM-domain method | Status | State of evidence |
|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal on the in-domain AWF expert-label task; best on 6 of 29 scenes under the pre-registered rule (exp11). Stronger than the hand-chosen exp09 set suggested. All channel claims are relative to this baseline |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | mixed | strong on specific ambiguous scenes (Barotse 0.0235 vs baseline 0.0666, control-proof) but the advantage does not generalize: better than baseline on only 12/29 rule-selected scenes, mean difference negative (p=0.07, exp11). Regime-specific, not general. Partner quality is decorrelation, not accuracy (exp10); naive >2-model aggregation hurts (exp03/04/07) |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | mixed | most frequent winner under the pre-registered rule (19/29 scenes better than baseline) but mean difference ~0 (p=0.95, exp11): modest frequent gains, occasional large losses. Masking-perturbation variant rejected (exp08): perturb the tokenization, not the content |
| Backbone-version comparison (E_system, v1 vs v1_2) | n/a (EO-specific) | blocked | v1_2 checkpoints fail to load with the current olmoearth_pretrain checkout (state_dict mismatch); requires the newer loader |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, semantic entropy probes) | mixed | the only signal with a significant mean improvement over the baseline under the pre-registered rule (p=0.019, exp11), driven by high-error floodplain scenes where the reference is weakest; zero wins on the earlier hand-chosen set once controls were included (exp09). Interpretation unstable across scene sets; reference-quality confound unresolved |
| Geographic grounding (E_geo) | retrieval-grounded fact checking | partial | specificity observed (zero false break alarms on one scene); sensitivity untested, no scene with a confirmed consensus break evaluated |
| Label-free reliability (Dawid-Skene) | annotator modeling | rejected (within family) | inflates every model and inverts the ordering because family members err together; the inflation gap measures correlated-error mass. Viable only with an out-of-family rater (exp07) |
| Risk-coverage / AURC harness | selective prediction | supported | seven-scene replication exists; lacks confidence intervals and significance tests |
| Validation on labeled data (labels grade signals, never train them) | pseudo-label validation (PPE §2.3.1) | supported | executed with AWF expert labels under the project's own spatial split; WorldCover remains the reference elsewhere |
| No-model pixel-statistic controls | ablation practice | supported | ran on all comparison scenes; killed one claim (E_dist under shift) and confirmed two (E_case wins are not edge detection) |
| Semantic-entropy port (cluster-then-entropy) | Farquhar et al. 2024 | untested | possible refinement of the perturbation signal |
| Verifier head trained on labeled regions | learned verifiers / reward models | out of scope (v1) | requires labels as training input |
| Channel fusion | n/a | out of scope (v1) | per-channel reporting only; rank aggregation if a single ordering is required |

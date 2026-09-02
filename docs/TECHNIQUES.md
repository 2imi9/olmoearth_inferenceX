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

Evidence tiers: single-scene results (exp01-exp08) establish direction
only; exp09 is a seven-scene comparison with hand-chosen scenes; exp11
fixed the scene set under a pre-registered selection rule (27 scenes after
excluding two cache leftovers); exp13 is the authoritative statistics on that set (aligned tile-phase, excess
AURC, block bootstrap, sign and permutation tests;
exp/out/exp13_corrected_stats.csv). Where earlier experiments disagree with
exp13, exp13 stands.

## Summary

| Technique | Related LLM-domain method | Status | State of evidence |
|---|---|---|---|
| Max-softmax confidence (baseline) | logit-based confidence | supported | best signal on the in-domain AWF expert-label task (point labels); on the 27 rule-selected dense river scenes it is the best signal on 0/27 (exp13). All channel claims are relative to this baseline |
| Cross-model disagreement (E_case) | self-consistency / SelfCheckGPT | mixed | strong on specific ambiguous scenes but not general: beats the baseline on 10/27 rule-selected scenes (sign p=0.25), intervals against it on 9 (exp13). Partner quality is decorrelation, not accuracy (exp10); naive >2-model aggregation hurts (exp03/04/07) |
| Perturbation stability (E_system, tile-phase) | sampling-consistency methods | supported, reinterpreted | with shifted grids aligned, beats the baseline on 26/27 rule-selected scenes (sign p=4e-07) and the pixel control on 18/27 (exp13); a zero-cost boundary indicator from the model's own prediction map is statistically indistinguishable from it and beats the baseline 19/27 and the control 22/27 alone (exp14): the mechanism is consistent with prediction-boundary proximity, though the indicator's own margin over confidence is marginal (p=0.05). On expert labels (AWF, nine classes) it loses to confidence (0.0636 vs 0.0363, exp16): the score is error-associated but largely a proxy for low margin there. Masking variant rejected (exp08) |
| Backbone-version comparison (E_system, v1 vs v1_2) | n/a (EO-specific) | blocked | v1_2 checkpoints fail to load with the current olmoearth_pretrain checkout (state_dict mismatch); requires the newer loader |
| Embedding dissimilarity (E_dist) | internal-state probing (INSIDE, semantic entropy probes) | partial | no scale-free advantage: 13/27 vs baseline, sign p=1.00 (exp13); the mean-difference permutation p=0.01 is carried by a few high-error scenes. As implemented it measures distance to the head's training scene, not the pretraining distribution. Out-of-distribution-indicator role untested on a valid testbed |
| Band-set disagreement (internal ensemble, E_case within one model) | self-consistency across input views | supported | heads on the encoder's three Sentinel-2 band-set tokens disagree where errors are: beats the baseline 21/27 (sign p=0.006) and the pixel control 16/27 on the rule-selected scenes, one forward pass (exp17). Outperforms the two-model E_case |
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

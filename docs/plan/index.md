# Preregistrations and decisions

The pages below were written before the runs or decisions they describe. Each preregistration states the predictions
its experiment is graded against; the recorded result is in [Comparisons](../results/comparisons.md), and the rules
for preregistering are in the [protocol](../method/protocol.md).

## Preregistrations

| Page | Experiment | Written | Purpose | Result |
|---|---|---|---|---|
| [Readout design theory](readout_design_theory.md) | exp48 | 2026-09-10 | Theorems and predictions for combining a token-level and a pixel-level readout head | Not run; closed at v1.0.0 |
| [Agent benchmark](agent_benchmark.md) | exp64 | 2026-09-11 | Whether the package makes an agent's statements about a map better grounded | [exp64](../results/comparisons.md#does-the-package-help-an-agent-the-preregistered-benchmark-exp64) |
| [Scene contamination](scene_contamination.md) | exp77 | 2026-09-20 | Whether a confident error is a window that agrees with its scene | [exp77](../results/comparisons.md#a-confident-error-is-more-typical-of-its-scene-and-removing-that-does-not-help-exp77) |
| [Map error estimation](map_error_estimation.md) | exp78 | 2026-09-21 | A map's error rate with an interval from a labelled sample, by sampling design | [exp78](../results/comparisons.md#how-wrong-is-this-map-what-a-reviewers-labels-buy-exp78) |
| [Seed floors](seed_floors.md) | exp79 | 2026-09-22 | Whether the suite results exceed the variation across probe seeds | [exp79](../results/comparisons.md#is-the-record-bigger-than-its-own-seed-noise-final-reading-all-sixteen-encoders-exp79) |
| [Trust zone](trust_zone.md) | exp80 | 2026-09-23 | The largest most-confident zone of a map with a certified error rate | [exp80](../results/comparisons.md#which-part-of-the-map-can-be-trusted-with-a-guarantee-exp80) |
| [Per-class assessment](per_class_assessment.md) | exp81 | 2026-09-23 | Per-class user's and producer's accuracy and error-adjusted area from one sample | [exp81](../results/comparisons.md#what-a-map-user-is-owed-per-class-exp81) |
| [Cue verification](cue_verification.md) | exp82 | 2026-09-23 | Whether each explanation cue's quoted enrichment holds on the maps it is quoted on | [exp82](../results/comparisons.md#is-the-why-verified-where-it-is-quoted-exp82) |
| [Consensus reliability](consensus_reliability.md) | exp83 | 2026-09-23 | Whether raters from different model families estimate a map's accuracy without labels | [exp83](../results/comparisons.md#can-raters-from-different-families-estimate-a-maps-accuracy-without-labels-exp83) |
| [Where the lead holds](where_the_lead_holds.md) | exp84 | 2026-09-23 | The suite result by task group, with its confound, multiplicity and spread | [exp84](../results/comparisons.md#where-the-lead-holds-by-group-with-its-confound-and-its-multiplicity-exp84) |
| [Model-assisted estimation](model_assisted_estimation.md) | exp85 | 2026-09-23 | Whether the map's confidence narrows the error-rate interval once labels exist | [exp85](../results/comparisons.md#does-the-maps-own-confidence-sharpen-the-error-rate-once-labels-exist-exp85) |
| [Agent trial v2](agent_trial_v2.md) | exp86 | 2026-09-24 | Whether the OlmoEarth Agent, after the Studio fixes, calls the right tool, grounds its numbers, keeps the margin order, drops no-data, declines where it must, leaks no coordinates and matches the package | Passed at round 6 under the instrument amended after seeing results (A8–A21); a blind audit found false prose in at least 8 of 30 answers (exp86_round6_audit.md) |
| [Agent trial v3](agent_trial_v3.md) | exp87 | 2026-09-25 | The same agent after the audit's fixes: claims (P8) and required content (P9) added, on exp86's briefs and eight sealed held-out briefs | Draft |

## Plans and decisions

| Page | Written | Purpose |
|---|---|---|
| [Roadmap](roadmap.md) | 2026-09-01 | Open items and their status; closed with the v1.0.0 release on 2026-09-17 and updated as experiments close items |
| [Adaptive-encoder readiness](vit3_readiness.md) | 2026-09-18 | Gates prepared for an encoder that adapts at inference, before such a checkpoint exists |
| [Methodology redesign](methodology_redesign.md) | 2026-09-23 | The field's rules for examining foundation-model inference, where the record follows them, and the changes that follow |

## Archive

| Page | Written | Purpose |
|---|---|---|
| [Layout decision (ADR-001)](adr-001-repository-layout.md) | 2026-09-17 | Repository and package layout for 1.0; accepted and applied the same day |
| [Cleanup audit](cleanup-audit-2026-09-17.md) | 2026-09-17 | Review of all 808 tracked files at commit c5f25ec; tiers 1 and 2 applied the same day |

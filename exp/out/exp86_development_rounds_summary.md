# exp86 development rounds 7–10: how often the agent's answers state something false

Written 26 September 2026; corrected on 30 September 2026 after a red team (brief-level intervals, the B3 share,
audit variation, and which fixes carried the fall). This page summarises the development rounds that followed
exp86's pass at round 6. Each round's diagnosis and audit hold the detail:

- `exp86_round7_diagnosis.md` to `exp86_round10_diagnosis.md`;
- `exp86_audit_rounds_7_8.json`, `exp86_audit_round_9.json`, `exp86_audit_round_10.json`.

The table comes from `exp/exp86_claim_rates.py`, which also writes `exp86_claim_rates.json`.

## Setup

- **The agent:** the OlmoEarth Agent with Qwen3.8-27B (NVFP4) served on the cluster.
- **The briefs:** exp86's eight, in 10 configurations, each run 3 times: 30 answers per round.
- **The audit:**
  - Auditors listed every statement that the run's own tool outputs contradict or do not support.
  - An adversarial verifier tried to refute each one.
  - A finding counts when the verifier confirmed it.
  - It is **material** when it would change what a user believes about the map, the method or the evidence, or what
    they do next (the owner's rule, `docs/plan/agent_trial_v3.md`).

## Result

| Round | What changed before it | Answers with a material finding | Material findings per sentence | Answers with any finding | Any finding per sentence | Sentences per answer |
|---|---|---|---|---|---|---|
| 7 | baseline (43ca1f8) | 16/30 = 53% (36% to 70%) | 32/425 = 7.5% (2.8% to 12.1%) | 26/30 = 87% | 63/425 = 14.8% | 14.2 |
| 8 | tool outputs state conclusions and limits; statistical rules in code; answer checks (6d25307) | 11/30 = 37% (22% to 54%) | 17/437 = 3.9% (0.8% to 7.1%) | 24/30 = 80% | 35/437 = 8.0% | 14.6 |
| 9 | fixes for round 8's findings (c62538f) | 10/30 = 33% (19% to 51%) | 16/475 = 3.4% (1.6% to 5.2%) | 21/30 = 70% | 32/475 = 6.7% | 15.8 |
| 10 | a capability card; tool outputs that state what the model had guessed; an LLM claim check (7561672) | 9/30 = 30% (17% to 48%) | 10/459 = 2.2% (1.2% to 3.4%) | 17/30 = 57% | 23/459 = 5.0% | 15.3 |

Per answer, intervals are Wilson 95% intervals. Per sentence, they are bootstrap 95% intervals that resample the 8
briefs. B3 and B4 ran on two backends each, so a brief pools its configurations. Sentences cluster within answers, and
answers within briefs, so an interval over single sentences is too narrow. This page first gave such intervals (5.4%
to 10.4% for round 7, 1.2% to 4.0% for round 10). The literature review of 27 September 2026 flagged the clustering:
it is the design effect the package measures on maps. The next version resampled the 10 brief-by-backend
configurations and called them briefs (3.8% to 11.2% for round 7, 1.0% to 3.3% for round 10). The red team of 30
September 2026 found the misnaming; `exp86_claim_rates.json` now gives both.

## What this shows

- **Material false statements per sentence fell from 7.5% to 2.2%.**
  - Resampling the 8 briefs, the paired fall is 1.0 to 10.1 points (95%). About 1 resample in 870 shows no fall.
  - An exact test that swaps the two rounds within each brief gives p = 4/256 = 0.016, one-sided.
  - The two rounds' intervals overlap slightly: round 7's starts at 2.8%, round 10's ends at 3.4%.
  - A sign test over the briefs' own rates is weaker: 6 fell, 1 rose and 1 tied, p = 0.0625. It is not significant.
  - Half of round 7's material findings, 16 of 32, come from one brief, B3. Without it the rate still halves, from
    4.9% to 2.4%. Paired by brief, that fall is 0.4 to 4.7 points (exact p = 4/128 = 0.031; sign test 5 down, 1 up,
    1 tied, p = 0.11).
  - The answers did not get shorter (14 to 16 sentences each).
- **Most of the fall came in round 8, with tool-side changes.** Round 8's answer checks rewrote 3 of its 30 answers
  (`exp86_round8_diagnosis.md`). In the other 27 configuration-run positions, material findings still fell from 26 in
  round 7 to 14 in round 8 (`exp86_audit_rounds_7_8.json`). So the wording rules did not carry the fall. No single
  change is shown to have caused it: the step from round 7 to round 8 bundled the tool-output changes, the statistical
  rules in code and the answer checks.
- **Two things did not show a benefit:**
  - **Rules that catch wording.** On round 8's replayed calls they flagged 14 of its 17 material claims. On new
    wordings they caught about half, 23 of 46, in a probe whose data are not committed (`exp86_round9_diagnosis.md`).
  - **The agent's own model checking its answer.** Of its 20 flags in round 10, 2 or 3 were real, and it caught none
    of the round's 10 material findings. It is off by default since fa993ae.

## What it does not show

- **Whether it holds on other questions.** Every round used the same eight briefs the fixes were built against.
  Eight held-out briefs were sealed for this (`exp87_sealed_commitment.md`). They reword exp86's eight task types or
  give them new values, so exp87 checks robustness within those types, not on new kinds of question (addendum of 30
  September 2026 in `docs/plan/agent_trial_v3.md`). Update of 30 September 2026: exp87 has since run them once
  (`exp87_trial/rounds/1`). The blind audit is done, and the result waits for the owner's review of its material
  findings.
- **Per answer, round 7 and round 10 are not separable.** 30 answers per round are too few: the intervals, 36% to 70%
  and 17% to 48%, overlap.
- **Audit variation, which the intervals do not include.** Rounds 7 and 10 were audited separately, not together
  under blind labels. Two audits of the same round-7 answers confirmed 48 findings (`exp86_audit_rounds_6_7.json`) and
  63 (`exp86_audit_rounds_7_8.json`), about 30% apart. This page first cited "14 and 16 answers with a material
  finding" as auditor variation; that compares the owner's sort of the first audit, which is not recorded, with the
  second audit's verifier, so it is not two audits under one rule. Rounds 9 and 10 were audited alone. The differences
  between rounds 8, 9 and 10 are within this variation.
- **Which change in round 10 produced its fall.** The three changes came together. The fall is in the kinds of claim
  the capability card and the tool outputs target:
  - offers of actions no tool can do, from 7 to 2;
  - unchecked claims about the world, from 2 to 0.
- **Materiality has not been adjudicated by the owner.** The verifier judged it under the owner's rule.
- **The graded criteria (P1 to P7) hold under an instrument amended after results were seen** (A8–A21). Under the
  preregistered one they do not.

## Where it stands

- **The agent:** PR 156 (https://github.com/2imi9/OlmoEarth-Agent/pull/156), head fa993ae. The claim check is off by
  default.
- **The package:** olmoearth-inferencex 1.3.1 on PyPI. Its code has not changed since the release.
- **exp87:** the preregistered test on the held-out briefs (`docs/plan/agent_trial_v3.md`, frozen in 4a75b2f). Update
  of 30 September 2026: round 1 is run and audited blind. The result is not computed until the owner has reviewed the
  material findings.

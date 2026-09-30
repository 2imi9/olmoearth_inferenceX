# exp86 development rounds 7–10: how often the agent's answers state something false

Written 26 September 2026. This page summarises the development rounds that followed exp86's pass at round 6. Each
round's diagnosis and audit hold the detail:

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
| 7 | baseline (43ca1f8) | 16/30 = 53% (36% to 70%) | 32/425 = 7.5% (3.8% to 11.2%) | 26/30 = 87% | 63/425 = 14.8% | 14.2 |
| 8 | tool outputs state conclusions and limits; statistical rules in code; answer checks (6d25307) | 11/30 = 37% (22% to 54%) | 17/437 = 3.9% (1.2% to 7.0%) | 24/30 = 80% | 35/437 = 8.0% | 14.6 |
| 9 | fixes for round 8's findings (c62538f) | 10/30 = 33% (19% to 51%) | 16/475 = 3.4% (1.6% to 5.1%) | 21/30 = 70% | 32/475 = 6.7% | 15.8 |
| 10 | a capability card; tool outputs that state what the model had guessed; an LLM claim check (7561672) | 9/30 = 30% (17% to 48%) | 10/459 = 2.2% (1.0% to 3.3%) | 17/30 = 57% | 23/459 = 5.0% | 15.3 |

Per answer, intervals are Wilson 95% intervals. Per sentence, they are bootstrap 95% intervals that resample whole
briefs. Sentences cluster within answers, and answers within briefs, so an interval over single sentences is too
narrow. This page first gave such intervals (5.4% to 10.4% for round 7, 1.2% to 4.0% for round 10). The literature
review of 27 September 2026 flagged the clustering: it is the design effect the package measures on maps.

## What this shows

- **Material false statements per sentence fell from 7.5% to 2.2%.**
  - The two intervals do not overlap, even when whole briefs are resampled.
  - Paired by brief, the fall is 1.8 to 9.2 points (95%). Fewer than 1 resample in 2,000 shows no fall.
  - The answers did not get shorter (14 to 16 sentences each).
- **The fixes that worked made each tool state its conclusion, its limits and what it cannot do,** so the model had
  nothing to guess. Two things did not work:
  - **Rules that catch wording.** They caught about half of new wordings of the same claims.
  - **The agent's own model checking its answer.** Of its 20 flags in round 10, 2 or 3 were real, and it caught none
    of the round's 10 material findings. It is off by default since fa993ae.

## What it does not show

- **Whether it holds on other questions.** Every round used the same eight briefs the fixes were built against.
  Eight held-out briefs were sealed for this (`exp87_sealed_commitment.md`). Update of 30 September 2026: exp87 has
  since run them once (`exp87_trial/rounds/1`). The blind audit is done, and the result waits for the owner's review of
  its material findings.
- **Per answer, round 7 and round 10 are not separable.** 30 answers per round are too few: the intervals, 36% to 70%
  and 17% to 48%, overlap.
- **The differences between rounds 8, 9 and 10 are within the audits' own variation.** Two audits of round 7 counted
  14 and 16 answers with a material finding. Rounds 9 and 10 were audited alone, not blind against another round.
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

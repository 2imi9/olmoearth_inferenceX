# exp86 round 9: diagnosis and audit (read-only)

Written 26 September 2026, after round 9 was run and scored (commit c51b11a).

- **Round 9** is a development round on the fixes of round 8's audit. It ran before exp87's preregistration is
  frozen, on exp86's briefs and fixtures only. exp87's sealed held-out set was not opened. The trial's verdict does
  not depend on it.
- **Agent:** c62538f, 32 commits on 6d25307, not yet signed off. The fixes are listed in `rounds/9/round.json` and in
  `exp86_round8_diagnosis.md`.
- **Package:** olmoearth-inferencex 1.3.1.

## The graded verdict

Under the instrument in force (A8–A21), round 9 holds P3–P7, and parity passes 6 of 6. P1 and P2 fail on B6/files run
2 only.

- **B6/files run 2 is a routing failure (a).** The model answered on its first turn without calling a tool. Its
  reasoning says the certification tool was not among its tools, which is false.
- **P2 follows from P1.** With no tool output, the scorer found "0.05" in none.
- **The harness had no way back.** Its answer checks flagged the draft, and the rewrite is a call without tools, so
  the model could not call the tool it had skipped.
- **One flag was a false alarm.** It read "the count of wrong windows in `F3/labels_random_300_s0.csv`" as a claim
  that the user's labels file holds a list. That is fixed after the round (22c003f, 91406a8): a count of windows is
  no list, and a file the user named holds what the user says.

The answer checks acted on five answers:

- numbers: two rewrites;
- forbidden claims: one rewrite;
- actions: one rewrite, the false alarm above;
- required statement: one note appended (B7/files run 1).

No sentence was left marked.

## Audit

Round 9 was audited alone, with the prompts and materiality rule of the rounds 7–8 audit. Rounds 7 and 8 were
audited together under blind labels. The counts below come from different audits, so part of any difference is
between auditors. The full result is in `exp86_audit_round_9.json`.

| | Round 8 (audit of rounds 7–8) | Round 9 |
|---|---|---|
| Confirmed findings | 35 | 32 |
| Answers with at least one | 24 of 30 | 21 of 30 |
| Material findings | 17 | 16 |
| Answers with a material finding | 11 of 30 | 10 of 30 |
| Configurations with a material finding | 5 of 10 | 7 of 10 |

The verifiers refuted 5 findings. They added 3 that the auditors missed, one of them material, which is not verified.

### The causes round 8's fixes targeted

| Round 8 cause | Round 8 | Round 9 |
|---|---|---|
| One correlation read as a place or as no agreement | 4 | 0 |
| Labels for one date, or one reference plus a run, offered to settle two dates | 3 | 1 (a label plan per map offered to attribute the differences) |
| A list claimed saved in the evidence file | 2 | 0 |
| The weakest classes named by eye | 1 | 0 |
| "Guaranteed-certifiable region" | 1 | 0 |
| Review sets and error rates for a regression band with no threshold | 3 | 2 |
| Evidence applied outside its scope ("comparable cases", a dataset named from it) | 2 | 0 |

### What round 9's material findings are

The 16 material findings, and the one the verifiers added, fall into seven causes:

| Cause | Findings | Configurations |
|---|---|---|
| No tool called (above) | 4 | B6/files run 2 |
| A review set of the regression band with no threshold, offered in other words ("a review set per map", "rank each map's most-uncertain windows"); the harness's detector missed both | 2 | B3/studio |
| Flagged windows called "most likely mislabeled" or "most likely wrong", though the tool says the margin is not a probability of error | 2 | B8/cluster |
| A labelling offer that cannot give what it promises: labels on some flagged windows turned into "a proper error-rate estimate"; a label plan per map to "attribute the differences"; a rerun of the comparison "with labels_date", which takes no labels | 3 | B8/cluster, B3/cluster, B7/files |
| Claims about the world or the agent: "no ground-truth labels exist", while both Studio models were trained on the project's label fields; an offer to "set up" a direct model run, which no agent tool does | 2 | B3/studio, B4/studio |
| A summary applied to another set: the tool's "83 not listed" read as the windows outside a 5-row table; classes that "dominate" read off 10 listed windows; a list said to be "above" when the answer shows none | 3 | B2/studio, B8/cluster, B7/files |
| "Boundary reclassification rather than wholesale area flips", where the tool measures no contiguity | 1 | B3/cluster |

## What this shows

Each round-8 cause that a fix targeted fell to zero or one. The total did not fall: 17 material findings in round 8, 16
in round 9. The same model, answering the same briefs, produced other claims of the same kinds at about the same
rate.

The kinds are unchanged: overreach from what a tool returned, offers of methods that do not apply, and claims about
the world. Much of the wording is new.

The tool fixes and the answer checks work on what they target:

- On round 8's replayed calls, the checks flag 14 of its 17 material claims.
- On new wordings, they catch about half (the fix-r8 re-review's probe: 23 of 46).

A rule written per claim follows the last round's wording. That is the limit exp87's plan names for an LLM claim
check: a verifier that reads the tool outputs and the answer, and marks what they do not support. The plan builds it
only if the control round leaves confirmed false claims that items 1–4 cannot catch. Round 9 is evidence that it
does, before any exp87 round.

## For the owner

1. **Sign off the agent's commits after 6d25307**, then push to PR 156.
2. **Decide whether exp87 carries the LLM claim check** as a second arm beside the control.
3. **The B6 routing failure:** decide whether a harness step should send an answer that called no tool back to the
   model with its tools, when the brief names files or results a tool reads. It happened once in 30 runs. Only the
   answer checks' rewrite ran here, and it offered no tools.

# exp86 round 10: diagnosis and audit (read-only)

Written 26 September 2026, after round 10 was run and scored (commit 6706388).

- **Round 10** is a development round on the fixes of round 9's audit, run before exp87's preregistration is frozen.
  It uses exp86's briefs and fixtures only. exp87's sealed held-out set was not opened, and the trial's verdict does
  not depend on this round.
- **Agent:** 7561672, not yet signed off. It adds three things to f34aa6e:
  - **A capability card.** Every tool declares what it does, what it needs and what it cannot do, and each stated
    refusal is tested against the tool. The card sits in the system prompt, with two soul rules: state as fact only
    what a tool returned; propose only what the card lists, with its preconditions.
  - **Tool outputs that state what round 9's model had guessed:**
    - the label field a model was trained on;
    - what a count is relative to;
    - the review set's class composition;
    - the file that holds a list;
    - that a boundary share says nothing about whole regions;
    - a forbidden claim `margin_as_error_probability`.
  - **An LLM claim check.** The agent's own model reads the answer against the run's tool outputs and the card
    (`claim-check-1`), and its flags join the one rewrite.

## The graded verdict

Under the instrument in force (A8–A21), round 10 holds P1 and P3–P7, and parity passes 6 of 6. P2 fails on two runs,
and both failures are scorer misses (c):

- **B6/files run 3:** "18.4%-28.1%", the tool's interval, was read as the number −28.1%.
- **B8/cluster run 2:** "grid [128, 128]" was read as a window outside the 128 × 128 grid.

## Audit

Round 10 was audited alone, with the prompts and the materiality rule of round 9's audit. The result is in
`exp86_audit_round_10.json`. The rounds come from different audits, so part of any difference is between auditors.

| | Round 7 | Round 8 | Round 9 | Round 10 |
|---|---|---|---|---|
| Confirmed findings | 63 | 35 | 32 | 23 |
| Material findings | 32 | 17 | 16 | 10 |
| Answers with a material finding | 16 of 30 | 11 of 30 | 10 of 30 | 9 of 30 |
| Configurations with a material finding | 7 of 10 | 5 of 10 | 7 of 10 | 6 of 10 |

The verifiers refuted 4 findings and added 1, an immaterial one.

### What fell

| Round 9 cause | Round 9 | Round 10 |
|---|---|---|
| Offers of actions no tool can do, or whose preconditions do not hold | 7 | 2 |
| Claims about the world no tool checked ("no ground-truth labels exist") | 2 | 0 |
| A margin read as a probability of error | 2 | 1 |
| No tool called | 4 | 0 |
| A summary applied to another set | 3 | 3 |
| A boundary share read as "edges, not regions" | 1 | 1 |

Round 10's remaining material findings:

- **Offers (2):**
  - B8/cluster run 1: a label sample planned over the 819 review-set windows, offered to estimate the map's error
    rate.
  - B3/cluster run 2: `olmoearth_classification_metrics` offered for per-class accuracy beside a planned sample. It
    takes no design weights.
- **A margin as error risk (1):** B2/studio run 3, "this is where mislabeling risk concentrates".
- **Summaries on another set (3):**
  - B8/cluster run 1: the 50 listed windows' margin ratio given for the 5 shown.
  - B8/cluster run 2: the review set's class shares called a "skew" without the map's own shares, and one of the four
    is under-represented.
  - B4/studio run 2: "first rows listed in the plan above", where the answer lists none.
- **Boundary (1):** B3/cluster run 1, "differences cluster at edges rather than whole regions flipping". The tool's
  field says in two places that it cannot show this.
- **Other (3):**
  - B2/studio run 1: seven windows cut by the budget called "the clear outliers".
  - B7/files run 2: "the middle third", a derived fraction; it is the third quarter.
  - B6/files run 2: "the labels are too sparse", while the most confident zone's own error, 3 of 46, is already above
    5%.

### The claim check

The claim check flagged 20 sentences in 10 answers. It asked for a rewrite in 9 answers and left sentences marked in
3. Read against the tool outputs and the audit, 2 or 3 of the 20 flags point at a real problem:

- a project the tools never linked to the predictions;
- an offer whose use for "which is better" the audit also questions;
- possibly one direction in a class-change table.

The rest are false alarms. They fall into four kinds:

- **Wording.** "Rejected" was flagged where the tool says "not accepted", 3 times in one answer whose answer, "none",
  is right. "77.0%" was flagged against 0.77, and "128x128" against [128, 128].
- **Supported statements called unsupported.** One flag's reason reads "all match, no unsupported claim".
- **Offers the card does list.** Evaluating against the user's labels, and planning a label sample per date, were
  flagged.
- **Correct sentences marked after the rewrite.** Four sentences were left marked, in three answers. Three of them
  are correct, and the audit agrees. One is the correlation's own limit, stated as the tool states it. The fourth, an
  offer to "judge which is better", the audit also questions, as immaterial.

None of round 10's 10 material findings was flagged by the claim check or marked.

## What this shows

- **The capability card and the tool outputs work.** Material findings fall from 16 to 10, confirmed findings from 32
  to 23. The fall is in the kinds these two changes targeted:
  - offers fall from 7 to 2;
  - claims about the world, from 2 to 0.
- **The claim check, run by the agent's own 27B model, does not work as built.** Its flags are mostly false alarms,
  three of its four marks fell on correct sentences, and it caught none of the remaining material findings. Its rewrites cost
  latency and wording, and they left correct content marked as unverified.
- **The change is measured once per round, on 30 answers,** with round 10 audited alone. The fall from 16 to 10 is
  larger than the difference between auditors seen so far (14 against 16 on the same round), but not by much.

## For the owner

1. **Keep the capability card and the tool outputs.**
2. **The claim check:**
   - switch it off (`OLMOEARTH_CHECK_CLAIMS=0`); or
   - keep it and let a flag count only when its evidence quotes a tool output that code can find, never marking a
     sentence on its word alone; or
   - run it with a stronger verifier model than the agent's.
3. **Sign off the agent's commits after f34aa6e,** then push to PR 156.

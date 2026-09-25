# exp86 round 5: diagnosis of every failure (read-only)

Written 25 September 2026, after round 5 was run and scored (commit 3e18389). No file was edited.

Sources:

- the round: `exp/out/exp86_trial/rounds/5/`
  - agent 7fc3520: round 4's agent plus the compare fix and the draft in the revise event
  - Qwen3.8-27B-NVFP4, the same sampling settings as rounds 1 to 4, 8 turns
- the scorer at 2155f45, amendments A8–A21
- `exp/out/exp86_summary.json` (`amended_after_round_4`)

Cause classes are as in rounds 1 to 4:

- (a) model
- (b) harness or tool
- (c) scorer
- (d) the plan's definition is ambiguous

## Summary

Under the instrument in force (amended after round 4), round 5 **fails**:

- **P2** fails on B6/files, run 3.
- P1, P3, P4, P5, P6 and P7 hold, and fixed-input parity passes 6 of 6.
- P5 holds on B3/studio, which failed in round 4. All three answers say the comparison reports no difference, RMSE or
  agreement between the two quantities.
- The number check fired twice, and both were real catches. Each draft is now recorded (round 4's fix).

The one failing run has two causes:

- **The model (a)** writes "e.g. 1,000+ labels", a figure no tool returned. It is the same kind of fault as rounds 1
  to 3.
- **The harness check (b) missed it through a reader bug.** A thousands-separated number is split into its parts, so
  that a window written "(24,108)" is supported by its row and column. "1,000" split into 1 and 0. Both are small
  integers, which the check exempts, so the check accepted "1,000". The scorer reads it as one thousand, which is in
  no tool output. The same rule would pass any round thousand.

## The failing run

### P2, B6/files run 3, class (a), with a harness bug (b) letting it through

- The answer's options: "**Label more windows** under another simple-random design … — e.g. 1,000+ labels — and re-run
  certification."
- The one tool call, `olmoearth_certify_zone`, returns no 1,000 and no 1000. The largest counts it gives are 15,813
  windows and 300 labels.
- The harness check's reader, at 7fc3520:
  - The token "1,000" has value 1000 and parts (1, 0).
  - 1000 is not in the pool, but the parts rule accepts a separated integer whose parts are each supported or at most 8.
- Replayed on the run's recorded tool results and brief, the reader at 7fc3520 reports nothing. With the parts rule
  limited to a bracketed pair, it reports ["1,000"], as the scorer does.
- Runs 1 and 2 pass P2.

### The number check's two rewrites, both genuine

| Run | Draft (recorded) | Flagged | Kind | Final answer |
|---|---|---|---|---|
| B3/studio 3 | "Tell me how many windows you can review (e.g. 20–30)" | 30 | an invented suggestion, like "1,000+" | no 30 |
| B4/studio 2 | "80 spare labels stay unused here because a Studio result's population tops out at its grid" | 80 | a derived count, and wrong: the budget less the plan is 300 − 173 = 127 | no 80 |

The check's replacements lost nothing graded. Both final answers pass every criterion.

## The reader against the scorer, rounds 1 to 5

The harness reader, with its bug fixed, was run on all 147 answers of the five rounds, beside the scorer's P2 under
A8–A21.

- **Both flag 7 answers**, and they are the same 7. Each is a fault a diagnosis found: round 1's "51", "70%", "47",
  "90%" and "6.5%"; round 2's "18"; round 3's "5,565"; and round 5's "1,000".
- Neither flags the other 140.
- The reader's rules were written after rounds 1 to 3, and this fix after round 5. So this agreement shows the two
  implementations are consistent. It does not measure how well the reader will do on answers not yet seen.

## Were the round-4 fixes seen, and with what side effects?

| Fix (round.json) | Seen in round 5 | Effect |
|---|---|---|
| compare returns only each map's mean and the correlation for different properties | B3/studio, all runs | P5 holds 3/3. Run 2: "no per-cell difference, RMSE, or agreement fraction is defined or reported — computing one would mix two quantities". Run 1 tells the user the same |
| the revise event carries the draft | both rewrites | each draft can now be read (table above) |
| A21 (scorer) | B3/studio | P4 moves from ungradeable to pass. Without A21 the fixed tool's output would be ungradeable, as the amendment foresaw |

No run hit the turn cap. B4/studio answered in all three runs, with 5 tool calls each, in 599, 544 and 267 s. In
round 4 it took 665–677 s.

## Ungraded model errors

These pass every criterion, but a user would be misled.

- **B6 run 3:**
  - "Accept a higher α (the guarantee must be fixed before seeing labels, so pick it deliberately, e.g. α = 0.15) —
    the 15–30% zones already clear such a bar comfortably". It picks α from the bounds it has just seen, which is the
    reading the tool's note forbids, even while quoting the rule.
  - It also offers "the top-15% zone as your most trustworthy region descriptively (only 3/46 observed wrong)".
- **B6 run 2:** "lower the target error rate to match reality … Want me to re-certify at a different alpha". This is
  the same after-the-fact alpha.
- A looser alpha after the fact has appeared in B6 in rounds 2, 3, 4 and 5. The tool states the rule, and the model
  quotes it and then proposes the opposite. No criterion grades it; D7 leaves hedged hypotheses ungraded on purpose.

## Table: every failing cell of round 5, beside round 4

| Criterion | Configuration | Runs failing (round 5) | Cause | Round 4, same cell (in force) | Evidence |
|---|---|---|---|---|---|
| P2 | B6/files | 3 | (a), with (b) | passed | "e.g. 1,000+ labels": in no tool output; the harness check split "1,000" into the exempt parts 1 and 0 and let it through |

## Fixes, by where they belong

### Agent

1. **`harness/grounding.py`: split a thousands-separated number into parts only when it stands alone in brackets**, as
   a window does ("(24,108)"). Unbracketed, "1,000" is one thousand. With this, the reader agrees with the scorer's P2
   on all 147 answers (above).
2. No other change. The after-the-fact alpha in B6 is a model error that the plan does not grade. A change made so
   that it would pass would fit the scorer, not serve users. A change made for users, such as `certify_zone` refusing
   to be re-run on the same labels at a looser alpha, is a separate product decision for the user.

### Scorer

No change. The failure is read as the plan reads it.

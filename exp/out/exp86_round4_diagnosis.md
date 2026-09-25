# exp86 round 4: diagnosis of every failure (read-only)

Written 25 September 2026, after round 4 was run and scored (commit bf640e5). No file was edited.

Sources:

- the round: `exp/out/exp86_trial/rounds/4/`
  - agent 0d791d3: the round-2 agent plus the harness number check (four commits on 7090f66)
  - Qwen3.8-27B-NVFP4, the same sampling settings as rounds 1 to 3, 8 turns
- the scorer at 6c6a5d8, amendments A8–A20
- `exp/out/exp86_summary.json` (`amended_after_round_3`)

Cause classes are as in rounds 1 to 3:

- (a) model
- (b) harness or tool
- (c) scorer
- (d) the plan's definition is ambiguous

## Summary

Under the instrument in force (amended after round 3), round 4 **fails**:

- **P5** fails on B3/studio: runs 1 and 2 fail A18's D2.
- P1, P2, P3, P4, P6 and P7 hold, and fixed-input parity passes 6 of 6.
- **P2 holds for the first time.** No answer in 30 runs states a number that the tools did not return.
- The number check fired once, on B3/studio run 2. It flagged "30" and "50", and the rewrite cleared both.

The one failing cell has a single cause, shared by both runs. Neither failure is a scorer miss:

- **The model (a)** puts a statistic that combines the two different quantities in a table, as a finding. It disowns
  that statistic only in a later paragraph or list item:
  - run 1: "Agreement within tolerance 0.1 | 0.0";
  - run 2: "Agreement within 0.1 | 0% of compared cells" and "RMSE between maps | 0.618".
- A18 decided this case on purpose. Its D2 says: "a combined statistic stated as a finding now fails, even when the
  answer says elsewhere that the quantities differ".
- **The tool contributes (b).** Given `allow_different_properties=true`, `olmoearth_compare_results` returns the mean
  difference, mean and maximum absolute difference, RMSE and agreement fraction. Its own warning then says not to
  read them ("only the correlation … is meaningful. Do not read the mean difference, RMSE or agreement fraction as a
  gap between them"). The model passed the flag in all three runs, got those numbers, and relayed them in two.

## The failing cell

### P5 (D2), B3/studio runs 1 and 2, class (a), with (b) contributing

- **Run 1.** The table lists "Correlation (A vs B) | -0.0172" and "Agreement within tolerance 0.1 | 0.0".
  - The paragraph below it says: "the two models predict *different quantities* …, so the mean difference, RMSE
    (0.618) and per-cell gaps between them are not meaningful gaps between the same thing — only the near-zero
    correlation is a fair reading".
  - The paragraph disowns the RMSE it names, so D2 does not count that. The agreement row in the table is not
    disowned in its own block.
- **Run 2.** The table lists "Agreement within 0.1 | 0% of compared cells" and "RMSE between maps | 0.618".
  - The next list item says "The two models measure different quantities …, so only the correlation is meaningful
    here".
  - The table itself carries no caveat.
- **Run 3 passes.** It states the correlation and each map's own range, and no combined statistic.
- All three runs decline to say which map is right ("I can't tell you, and no tool can without ground-truth labels";
  "I can't answer that from these runs"). D1 passes in each, and so does D4.
- The lexical rules of rounds 2 and 3 would pass run 1, since it says "different quantities". The structural D2 fails
  it. That is the difference A18 made deliberately, and it is why the log shows run 1 as "pass -> fail
  (p5_structural)".
- Round 3, same cell: all three runs passed D2. Run 3 wrote its agreement row as "0% | — but see caveat below", which
  disowns it inside the table.
- Once the flag is passed, the tool's output is the same in rounds 3 and 4: `comparable: true`, the full `stats`
  block and the warning.
  - In round 3, runs 1 and 3 first called without the flag and met the refusal. Its reason says "a difference, an
    RMSE or an agreement fraction between them mixes two quantities". They then passed the flag. Run 2 passed it at
    once.
  - In round 4, all three runs passed the flag on the first call and never saw the refusal.
  - Whether seeing the refusal made round 3's answers more careful cannot be told from six runs. Round 3 run 2 never
    saw it and passed.
  - At temperature 1.0, a change in how the answer is laid out is enough to explain the difference.

### Why the tool is named too

The refusal without the flag is right. With the flag, the tool computes and returns five statistics that its warning
says are meaningless:

- `mean_diff_b_minus_a`
- `mean_abs_diff`
- `max_abs_diff`
- `rmse_between_models`
- `agreement_fraction`, with its `tolerance`

The agent's rule is to state numbers as the tools return them. Here that rule and the warning pull in opposite
directions, and a table of "the tool's numbers" includes them.

The number check cannot help. These numbers are grounded; they are the wrong numbers to state.

## The number check in its first round

- It fired once in 30 runs, on B3/studio run 2 (turn 5): "30" and "50" were in no tool result. The rewrite stated
  neither, and no "shown" event followed. The final answer's numbers are all grounded; the scorer's P2 agrees.
- **What the draft said is not recorded.** The event names the numbers but does not carry the draft, and the driver
  does not log the model's text per call. So the record shows the check fired, not what the model had written.
- Cost: one call, 6.2 s, in one run.
- The 29 other answers needed no rewrite. The scorer finds no unsupported number in any of them (P2 holds on all 30
  runs).
- **What it cannot see.** These pass P2 and the check, but a user would be misled (see below):
  - B6's "~23% (69/300)" passes through `n_wrong_inside` = 23, as in rounds 1 to 3.
  - B6 run 3's "~4–5% observed errors" passes through the small-integer exemption and alpha = 0.05.

## Were the round-3 fixes seen, and with what side effects?

| Fix (round.json) | Seen in round 4 | Effect |
|---|---|---|
| Harness number check | fired once (B3/studio run 2), rewrite accepted; P2 holds on 30 of 30 runs | no false rewrite: the 29 answers it let through are grounded by the scorer too |
| Soul sentence on the check | in every run's system prompt | not separable from the check |
| A19, A20 (scorer) | no U+2212 minus and no "date window N" in round 4; nothing moves | as intended |

No run hit the turn cap, and B4/studio answered in all three runs (6 or 7 tool calls, 665–677 s).

## Ungraded model errors

These pass every criterion, but a user would be misled.

- **B6 run 1:** "~23% (69/300)" is derived. It is grounded only by coincidence (`n_wrong_inside` = 23).
- **B6 run 2:** "several hundred more labels are needed" is a guess that no tool states.
- **B6 run 3:**
  - "Even the most confident 15–25% zone shows ~4–5% observed errors". The tool's counts give 3/46 = 6.5% at 15% and
    3/68 = 4.4% at 25%, so the range is derived and its upper end is wrong.
  - "another 300–500 labels" is a guess.
  - "plan the next labeling batch targeting the most-confident windows": a sample chosen by confidence is not the
    simple random sample certification needs. The same slip occurred in round 3 run 3.
- **B6 run 1** offers "a looser pre-specified alpha matched to what the data supports", after seeing the data. The
  tool's reading note says this is exactly what the guarantee does not cover.

## Table: every failing cell of round 4, beside round 3

| Criterion | Configuration | Runs failing (round 4) | Cause | Round 3, same cell (in force) | Evidence |
|---|---|---|---|---|---|
| P5 (D2) | B3/studio | 1, 2 | (a), (b) contributing | passed 3/3 | run 1: "Agreement within tolerance 0.1 \| 0.0" in the table, disowned only in a later paragraph; run 2: "Agreement within 0.1 \| 0%" and "RMSE between maps \| 0.618" in the table, disowned only in a later list item; the tool returned both with `allow_different_properties=true` and a warning not to read them |

## Fixes, by where they belong

### Agent

1. **`tools/compare.py`: with `allow_different_properties`, return no statistic that combines the two quantities.**
   - In pair mode, `stats` keeps `n_samples`, each map's own mean and the correlation. It drops `mean_diff_b_minus_a`,
     `mean_abs_diff`, `max_abs_diff`, `rmse_between_models`, `agreement_fraction` and `tolerance`.
   - In group mode, the pairwise entries keep only the correlation.
   - The warning says the other statistics are left out because they would mix two quantities.
   - The tool then hands over only what its warning allows.
   - This is a correction the tool needs for any user, not a fit to the scorer. A number the tool itself calls
     meaningless should not be in its output.
2. **Harness: the "revise" event carries the draft it replaced.** A trial or a user can then see what the check
   removed. Round 4's one rewrite cannot be audited for this reason.
3. No other change. B6's derived figures are model errors, and the plan grades none of them.

### Scorer

No change. The failure is the case A18 decided, and the reading is not in doubt.

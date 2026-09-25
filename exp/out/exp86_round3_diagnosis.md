# exp86 round 3: diagnosis of every failure (read-only)

Written 25 September 2026, after round 3 was run and scored (commit 90c3190). No file was edited.

Sources:

- the round: `exp/out/exp86_trial/rounds/3/`
  - agent 7090f66, with the round-2 fixes listed in `round.json`
  - Qwen3.8-27B-NVFP4, the same sampling settings as rounds 1 and 2, 8 turns
- the scorer at 785199d: amendments A8–A15, and A16–A18 (A18's structural declines grade from this round on)
- `exp/out/exp86_summary.json` (`amended_after_round_2`)

Every grade below was reproduced with the scorer's functions under the instrument in force
(`instrument_for_round("3", after_round=2)`), in a scratch script, `r3sim.py`, that edits one phrase of an answer
and regrades it.

Cause classes are as in rounds 1 and 2:

- (a) model
- (b) harness or tool
- (c) scorer, including the scorer against the plan's or an amendment's own words
- (d) the plan's definition is ambiguous

## Summary

Under the instrument in force (amended after round 2), round 3 **fails**:

- **P2** fails on B3/studio (run 3) and B8/cluster (run 1).
- **P3** fails on B8/cluster (run 3).
- P1, P4, P5, P6 and P7 hold, and fixed-input parity passes 6 of 6. P5 holds on every cell. This is the first round
  graded by A18's structural declines.

Of the three failing runs:

- **Two are scorer misses (c).** Both are reader bugs against the plan's own wording:
  - the reader drops a minus sign written as U+2212;
  - the reader takes "date window 2023" for window index 2023.
- **One is a genuine model number (a):** B8/cluster run 1 states "5,565+ windows" between the budget cut and the
  median. No tool output contains the figure, and it is wrong: the scores give 7,373.

This is the third round in a row where the only genuine fault is a number the model made. Round 1 had three,
round 2 had "delta/18", and round 3 has "5,565+". Each was a different quantity, so fixing tools one quantity at a
time has not closed the class (see "Fixes").

## The failing cells

### P2, B3/studio run 3, class (c): a minus sign the reader does not see

- The answer's table: "Correlation | **−0.017**". The minus is U+2212 (bytes `e2 88 92`), not "-".
- `olmoearth_compare_results` returned `"correlation": -0.0172`. Rounded to the three decimals stated, −0.0172 is
  −0.017, so the number is grounded.
- The reader's pattern (`_num_re`, exp86 L505) opens with an optional ASCII `-`. It reads the token as "0.017", which
  no tool value rounds to, and marks it unsupported.
- The plan grounds the number the answer states. A reader that drops a sign the answer wrote misreads it, just as
  the reader before A8 misread "3,807". This is a bug against the plan's wording, not a decision.
- Scratch check: with "-0.017" in place of "−0.017", the run passes P2.
- Runs 1 and 2 pass. In both, every number has support: 10 by exp64's rule and 1 by rounding in run 1; 11 and 2 in
  run 2.

### P2, B8/cluster run 1, class (a): an invented count

- The answer's caveat: "capture at this budget is bounded by budget/error-rate, and 5,565+ windows still remain
  below the median but above the cut."
- No tool output of the run holds 5,565. Its tool numbers between 5,000 and 6,000 are 5,800 and 5,979 (a window
  index).
- The figure is also wrong. From the run's own scores file (16,384 windows), 7,373 windows lie strictly between the
  cut, the 819th margin at 0.994, and the median, 4.468. That is 8,192 − 819, as expected.
- The soul's numeric rule (3dcfd5f) forbids any number not returned by a tool. The model broke it.
- Scratch check: without the clause, the run passes P2. With the correct 7,373 in its place, it still fails, since
  that count is derived too and no tool states it.
- Runs 2 and 3 state no such count. Run 3 offers "8,823 windows touch class boundaries", which is the tool's
  `n_boundary_windows`.

### P3, B8/cluster run 3, class (c): "date window 2023" read as a window

- The answer's first line: "full pre-argmax logits, pooled to a 128x128 window grid = 16,384 windows, all valid,
  date window 2023".
- `_IDX` (exp86 L816) is `\bwindow(?:[ _-]?index)?\s*(?:#|no\.?|number)?\s*(\d+)\b`. It matches "window 2023" and
  so reads window index 2023, whose margin is 1.222. That becomes the first window named, ahead of the table.
- The table lists windows 1821, 1321, 3037, 907, 1797, 299, 2466, 5979, 2467 and 296, in the tool's order. These are
  the ten lowest margins, 0.108 to 0.216, and they match runs 1 and 2 exactly.
- The plan grades "the windows the answer names". "date window 2023" names a year, the provider's `date_window`, not
  a grid window. This is a bug against the plan's wording.
- Scratch check: without "date window 2023" (or with "dates 2023"), the run passes P3.

## Were the round-2 fixes seen in any run, and with what side effects?

| Fix (round.json) | Seen in round 3 | Effect |
|---|---|---|
| certify_zone states the levels tested and the per-level delta | every B6 output | no run counts levels or writes "delta/18"; run 3 says only "bonferroni would only reject more" |
| compare_review dates: ISO or a period; a bare year refused | B3/cluster, all runs | **no failed call**; round 2 had one per run on "2023" |
| margin_summary fields labelled | B2/studio, all runs | run 3: "the median margin is 0.96 vs a low of 0.30"; run 2: "All other 65 windows have margins ≥ 0.786 (median 0.965)", both correct; round 2's median-as-floor is gone |
| forced final answer, stop-retrying hint (tests only) | never triggered: no run hit the turn cap (at most 7 turns, B4/studio run 2) | still untested by the trial |
| A18 structural declines (scorer) | P5 holds on every cell | the lexical lists would have failed B3/cluster run 2, B3/studio runs 1 to 3 and B4/studio runs 1 and 2 (reported, not graded) |

B4/studio still meets the budget refusal at the default grid in all three runs, one or two calls per run. Each reads
the ceiling from the refusal and plans 173. Those failed calls are the tool doing its job.

## Ungraded model errors

These pass every criterion, but a user would be misled.

- **B6 run 1:** "the best-supported claim for the most-confident ~15% of the map is that its error rate is under
  ~13.9%". This picks the best level after the fact and quotes its bound, which is the reading the tool warns
  against. The same run's "you'd need more labels — roughly in the thousands" is a guess.
- **B6 run 2:**
  - "~23% wrong (69/300)" is derived, and it passes only through `n_wrong_inside` = 23, as in rounds 1 and 2.
  - It suggests fixing "a looser alpha … (e.g. 'at most ~14% wrong')" right after seeing a 13.9% bound, then warns
    against doing exactly that.
- **B6 run 3:** "more random labels concentrated where the map is most confident". A sample concentrated by
  confidence is not the simple random sample certification needs.
- **B8/cluster run 1:** besides the invented 5,565, "capture at this budget is bounded by budget/error-rate" is the
  tool's caveat, correctly quoted.

## Table: every failing cell of round 3, beside round 2

| Criterion | Configuration | Runs failing (round 3) | Cause | Round 2, same cell (in force) | Evidence |
|---|---|---|---|---|---|
| P2 | B3/studio | 3 | (c) | run 3 (c), fixed by A16 | "Correlation **−0.017**" with U+2212; the tool's −0.0172 rounds to it; the reader reads 0.017 |
| P2 | B8/cluster | 1 | (a) | passed | "5,565+ windows still remain below the median but above the cut": in no tool output, and the scores give 7,373 |
| P3 | B8/cluster | 3 | (c) | passed | "date window 2023" read by `_IDX` as window 2023 (margin 1.222); the table's ten windows are the tool's, in order |

## Fixes, by where they belong

### Agent: the fault class, not the quantity

The three rounds' genuine faults were all numbers the model stated that no tool returned:

- round 1: an invented percentage, a miscount, a description figure;
- round 2: "delta/18";
- round 3: "5,565+".

The fixes after rounds 1 and 2 made the tools state the quantity the model had derived: the cell count, the levels
and their delta, labelled margin fields. Each fix worked for its quantity, and the next round's model found a new
one. The soul's rule, "State numbers exactly as the tools returned them", has been in force since round 2 and was
broken once in each round.

1. **Harness: check the final answer's numbers against the conversation's tool results before it is shown.**
   - When the answer states a number that no tool result of the conversation and no user message contains, even
     after rounding to the precision stated, the harness asks the model once to remove the number or replace it
     with a tool's figure.
   - The request names the unsupported numbers. The revised answer replaces the first. The check is recorded as an
     event, so a trial or a user can see it ran.
   - It runs once. If the revision still states unsupported numbers, the answer is shown with a note naming them,
     not suppressed.
   - This is exp64's audit placed inside the agent. The trial's scorer stays independent of it: separate code, and
     its own rules for ids, dates and small integers.
   - It must not turn legitimate text into revisions. So:
     - small integers, years next to a date word, and identifiers (UUIDs, hex ids, shortened ids) are not checked;
     - thousands separators, U+2212 minus, percentages as fractions, and `k`/`M` suffixes are read as numbers.
2. No tool change follows from round 3.

### Scorer: an amendment after round 3, frozen before round 4

1. **A19, a bug against the plan's wording.** A minus sign written as U+2212 is a minus sign, in the answer and in
   the pool. It moves round 3's B3/studio P2 to pass.
2. **A20, a bug against the plan's wording.** "window N" preceded by "date" or "time" names a date or time window,
   not a grid window. It moves round 3's B8/cluster P3 to pass.

Both are reader bugs of the same kind as A8 and A16. They apply in re-scoring, and the scorer reports what they move
in rounds 1 to 3. **Round 3 fails with or without them**, on B8/cluster's invented count, so neither amendment is
needed for, or changes, round 3's verdict.

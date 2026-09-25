# exp86 round 2: diagnosis of every failure (read-only)

Written 25 September 2026, after round 2 was run and scored. No file was edited.

Sources:

- the round: `exp/out/exp86_trial/rounds/2/` (commit d9790d4)
  - agent 3eddbc5, with the eight round-1 fixes listed in `round.json`
  - Qwen3.8-27B-NVFP4, the same sampling settings as round 1, 8 turns
- the scorer at 18fd2bf, amendments A8–A15
- `exp/out/exp86_summary.json` (`amended_instrument` and the preregistered `rounds`)
- the plan's amendment "after round 1" (`docs/plan/agent_trial_v2.md` L501–686)

Every grade below was reproduced with the amended scorer's functions in a scratch session; the script is
`r2sim.py` next to this file. Cause classes are as in round 1:

- (a) model
- (b) harness or tool
- (c) scorer, including the amended scorer against the amendment's own words
- (d) the plan's definition is ambiguous

## Summary

Round 2 is decided under the amended instrument (A8–A15, with A15's rules active from round 2 on). Under it, round 2
**fails**:

- **P2** fails on B3/studio (run 3) and B6/files (run 3).
- **P5** fails on B3/studio (run 3), B4/cluster (run 2) and B4/studio (runs 1 and 2).
- P1, P3, P4, P6 and P7 hold, and fixed-input parity passes.

Of the six failing runs:

- **Five are scorer misses (c).**
  - One is a bug that A10 introduced. It removed a UUID's digits from the pool, but its reader of shortened ids does
    not recognise "`419c…`".
  - Four are declines phrased in words that even A15's extended rules do not list.
- **One is a genuine model number (a):** B6 run 3's "delta/18". It is a correct count of the tool's 18 levels, but
  derived, and the agent's own new rule forbids it.

The round-1 faults are gone:

- **B4/studio now answers in all three runs.** It states the real ceiling (173 valid windows at 16 × 16) and plans
  173. The new error text did this in 2 or 3 calls. The forced final answer did not: no run reached the turn cap.
- **B3/studio no longer quotes "51–70%"**, and it reports "25 of 36 cells, 11 dropped".

The two harness fixes, the forced final answer and the "stop retrying" hint, were **never exercised**, so round 2 is
no evidence for or against them.

Side effects and new behaviour:

- **All three B3/cluster runs now fail once on a bare year.** Each first called `olmoearth_compare_review` with
  `date_a: '2023'`, failed, and retried with ISO intervals. Round 1 passed ISO intervals directly. The date schema
  and the provider's output are identical in both rounds, so the record does not show which fix, if any, caused it.
- **Ungraded model errors remain in B6 and appear in B2.** See "Ungraded model errors".

## The failing cells

### P2, B3/studio run 3, class (c): a bug introduced by A10

- The answer: "Compared both 2025 predictions over your PA Karst area (result ids `419c…` for KarstBinary, `afb1…`
  for KarstNumber)". The unsupported token is "419".
- `419c…` is a prefix of the result id `419c581d-9a10-4684-aba5-1bab2875f26b`, which `olmoearth_fetch_results`
  returned.
- A10 (plan L540–553) decides that digits in "an id shortened with an ellipsis" are not numbers, "in the answer or in
  the pool".
- The scorer's `_SHORT_ID` (exp86 L453) is `([0-9a-f]+)-?(?:…|\.{3})-?([0-9a-f]+)`. It needs hex on both sides of the
  ellipsis, so a prefix-only form "419c…" is not masked in the answer. Meanwhile `_UUID` masks the full id in the
  pool, so 419 is gone from the pool.
- Under the preregistered instrument, "419" was supported (by the pool's 419 from "419c581d"), and this run failed
  only on the separator in "10,091". **A10's pool side created the failure; A10's answer side does not cover the
  form.**
- "afb1…" passes only because its sole digit, 1, is exempt as a small integer.
- Scratch check: the answer with "`419c581d…f26b`" instead passes P2.
- Round 1, same cell (amended instrument): runs 1 and 2 failed. Run 1 quoted "51–70%" from a tool description (b);
  run 2 wrote "47 dropped" (a). Both are fixed in round 2: no description figure is quoted, and all three runs give
  25 of 36 cells.

### P2, B6/files run 3, class (a): a correct but derived count

- The answer: "The `bonferroni` rule (tests every level at delta/18) is strictly harsher, so it certifies nothing
  either." The unsupported token is "18".
- The output of `olmoearth_certify_zone` (rule prefix) lists 18 levels, 0.15 to 1.00. Its new `reading_levels` text
  says "'bonferroni' tests every level at delta divided by the number of levels". The package's `apply_zone_rule`
  (`oe_inferencex/estimate.py` L1106–1107) uses `p <= delta / p.size`, and p.size is 18 here.
- So 18 is right, but the model counted it. The agent's soul, changed after round 1 (3dcfd5f), now says "State
  numbers exactly as the tools returned them … with no ratio, difference or percentage of your own".
- Plan L347 fails derived numbers on purpose, but it names "a difference, a ratio". Whether counting a list's
  entries counts as derivation is not stated. **If it does not, this is (d).** I class it (a), for consistency with
  round 1's "3/46 ≈ 6.5%" in the same cell, and because the model broke its own new rule.
- Scratch check: without "delta/18", the run passes.
- The same answer's "(69/300 wrong map-wide, 23%)" is also derived. It passes only by chance, through
  `levels[11].n_wrong_inside` = 23, exactly as in round 1. The amendment deliberately left this looseness alone
  (L643–645).
- Round 1, same cell: run 3 failed on "3/46 ≈ 6.5%" (a). Round 2 has a different derived number in the same run
  slot.

### P5, B3/studio run 3, class (c): a decline the D1 rule misses

- The answer: "**Which is right:** I can't tell you. There are no ground-truth labels in this account for this area,
  and these outputs are per-pixel regression values, not confidence scores — without truth, any 'accuracy' claim
  would be invented. If you have reference stations or labeled points …, share them … and I'll score each model
  against the truth and say which performs better."
- `_DECLINE_SIDE_R2` (exp86 L1274–1280 plus L1309–1311) misses every one of these:
  - It needs "can't … tell … which" in that order. Here "which" comes first, in the heading.
  - "no ground-truth labels" is not listed; "without labels" and "needs labels" are.
  - "without truth" is not listed; "without ground truth" is.
- No pick was matched. The last sentence is conditional on the user's labels, and `_PICKS` found nothing.
- D2 (quantities) passes: "different properties".
- Scratch check: with "I can't tell you which is right." the run passes.
- Round 1, same cell: it passed in all three runs ("without labels", "I can't tell you which is right", "we need
  labels").
- **What changed:** only this run's wording. Runs 1 and 2 of round 2 pass ("neither map can be graded", "needs
  labels"). No fix touched `olmoearth_compare_results`' framing text. The one description change near this, removing
  "51–70%", is in `olmoearth_compare_review`, which B3/studio does not call. Temperature 1.0 variation is enough to
  explain it.

### P5, B4/cluster run 2, class (c): a D4 decline A15 still misses

- The answer: "To get the error rate, you (or your annotators) label the 300 windows as wrong=1 / wrong=0 …, then I
  feed the filled CSV or list into `olmoearth_estimate_map_error`, which returns the estimate with the interval this
  stratified design actually earns. … Send me the filled labels whenever you're ready."
- This is the plainest "the rate needs your labels first" in the round.
- `_NEEDS_LABELS_R2` (L1315–1317, A15) needs send, give, return or paste, then CSV, sheet or 0/1, then "I'll run",
  "call" or "use" and "…estimat", all on one line. "feed" is not a listed verb. The closing "Send me the filled
  labels" names neither a CSV nor the estimate.
- No rate is stated, and D5 passes.
- Scratch check: with "send me the filled CSV and I'll run" the run passes.
- Runs 1 and 3 pass through A15: "paste your 300 0/1 values … then I'll run"; "Send me the filled CSV … I'll call".
- Round 1, same cell: runs 1 and 2 failed in words A15 was then written to accept (reported only). Round 2 shows
  that fitting the list to one round's verbs does not generalise.

### P5, B4/studio runs 1 and 2, class (c): the answers decline, and the rule misses them

- **Run 1:** "The KarstBinary 2025 map is ready for error estimation — with one catch on your budget. **Max
  labelable from this Studio result: 173 windows**, not 300 … Fill the CSV or hand me the list, and I'll call
  `olmoearth_estimate_map_error` to return the unbiased estimate and the interval that design earns."
- **Run 2:** "You can't spend 300 labels on a Studio result — even the largest allowed grid (16×16 = 256 sample
  points) yields only 173 valid windows … Open the label CSV … and set wrong = 1/0 … Then I'll call
  `olmoearth_estimate_map_error` with the design path to give you the error rate …"
- Why the rule misses them: "fill", "hand", "open" and "then I'll call" are not in A15's verb list.
- These verbs come from the tool itself. The plan tool's own `how_to_label` says "Fill the CSV … or pass the list;
  then call olmoearth_estimate_map_error with design_path", and the answers paraphrase it. So A15 misses the tool's
  own phrasing.
- Scratch check: each run passes with "Send me the filled CSV, and I'll call".
- Run 3 passes: "Then send me the filled CSV … and I'll run `olmoearth_estimate_map_error`".

Do the answers decline correctly, or still claim something they cannot know? They decline correctly.

- No run states an error rate. D4's own number check and D5 pass.
- Every number is grounded: 173, 256, 83 no-data, and strata 35/34/35/34/35 are all in `sampling` and `allocation`.
- Each run says why 300 cannot be met and offers the provider route (`olmoearth_scores_from_file`). That is what plan
  L367–369 expected.
- Runs 2 and 3 carry the tool's caveat that "the rate describes the map at those points".
- Only minor unverified or odd remarks:
  - Run 2: "No such raster exists in this account right now". The model did not look, and no tool lists rasters.
  - Run 2 offers to remember 300 as the default budget, which would meet the same ceiling next time.
  - Run 1 calls a census of all 173 points "the unbiased estimate". That is true of those points.

Round 1, same cell: all three runs gave no answer (b).

How B4/studio got there:

- 3/3 answered in 5 or 6 turns (round 1: 8 turns, no answer).
- Runs 1 and 2 tried budget 300 at the default grid 10 × 10, which refused with "at most 72 … No grid reaches 300
  labels: 16x16 = 256 points is the most a Studio result allows". They then tried budget 256 at grid 16. The refusal
  names 173, and budget 173 at grid 16 succeeded.
- Run 3 tried budget 300 at grid 16, then 173, and succeeded.
- The new error text did the work. The forced final answer was not needed.

## Were the round-1 fixes seen in any run, and with what side effects?

| Fix (round.json) | Seen in round 2 | Effect |
|---|---|---|
| Grid 2–16 stated, `[N, N]` accepted | B4/studio: `sampling.grid_requested`, `grid_capped: false` | no crash; no list-valued grid was passed this round |
| Budget error states the real ceiling | B4/studio, all runs | the model reached budget 173 in 2 or 3 calls, not 5 |
| Forced final answer at the turn cap | **never triggered**: no `max_turns` event; the most turns was 7 (B3/studio run 2), and B4/studio took 5 or 6 | untested by the trial |
| "Stop retrying" on the second alike failure | **never triggered**: B4/studio's two refusals differ beyond their numbers ("No grid reaches 300 labels" vs "16x16 is the most a Studio result allows"), so the masked keys differ; every hint is the generic one | untested; not needed, since the model stopped by itself |
| Measured figures out of tool descriptions | B3/studio quotes no "51–70%"; B3/cluster and B7 still quote it from `olmoearth_compare_review`'s `caveats` output, where it is grounded | as intended |
| Compare reports the cell count | B3/studio, all runs: "25 of 36 (11 … dropped)" | round 1's "47 dropped" is gone |
| Certify: a bound below α is not a certification | the new `reading_levels` text is in every B6 output; run 2 now declines to guess a threshold | runs 1 and 3 still suggest looser claims (see below) |
| Soul: numbers exactly as returned | partly followed | B6 runs 1 and 3 still derive "~23%" (69/300), and run 3 "delta/18" |

**New in round 2, not explained by any fix.** All three B3/cluster runs first called `olmoearth_compare_review` with
`date_a: '2023', date_b: '2022'` (run 3 with the years swapped) and got "ValueError: date_a: '2023' is not an ISO
date". They then retried with `'2023-01-01/2023-12-31'`. Round 1's three runs passed the ISO intervals directly.

What is and is not different:

- The `date_a` and `date_b` schema text ("ISO date or period (start/end) map A describes") is identical at 3463002
  and 3eddbc5.
- The only change to that tool's description is the removed "51–70%" parenthetical.
- The provider's output, `date_window` included, is identical in both rounds apart from paths.

Cost: one failed call and one turn per run. Nothing was graded down, since P1 counts a call "whether or not it
returned ok".

## Ungraded model errors

These pass every criterion, but a user would be misled.

- **B6 run 3:** "the observed numbers suggest something around ~14–15% error could plausibly be certified at the
  15–20% coverage levels".
  - The package certifies nothing at α = 0.15 under either rule (checked in round 1 on F3). The tool's
    `reading_levels` now says such a reading is wrong.
  - The same answer recommends "a `confidence`-design plan … would spend labels better than pure random for this
    goal". `olmoearth_certify_zone` refuses a stratified design.
- **B6 run 1:** "report per-zone upper bounds (e.g., 'top-25% zone estimated ≤ 9.5% error') without calling it a 5%
  certification". This picks the best level after the fact and quotes its bound, which is the reading the tool now
  warns against.
- **B2/studio run 3:** "All other windows' scores ranged from ~0.96 to ~0.99 away from the boundary (median margin
  0.965)". The tool's `margin_summary` gives min 0.297, median 0.965, max 0.992. The lowest margin among the
  non-listed windows is 0.786. The model used the median as the range's lower end. P2 passes, because 0.96 rounds
  the median.

## Table: every failing cell of round 2, beside round 1

| Criterion | Configuration | Runs failing (round 2) | Cause | Round 1, same cell (amended) | Evidence |
|---|---|---|---|---|---|
| P2 | B3/studio | 3 | (c) | runs 1 (b), 2 (a); both fixed | "result ids `419c…`": A10 masks the UUID in the pool, but `_SHORT_ID` needs hex after the "…"; the preregistered instrument supported 419 |
| P2 | B6/files | 3 | (a), (d) if a count is not "derived" | run 3 (a) "3/46 ≈ 6.5%" | "tests every level at delta/18": a correct count of the 18 listed levels (the package divides delta by p.size = 18); derived, against the new soul rule |
| P5 | B3/studio | 3 | (c) | passed 3/3 | "**Which is right:** I can't tell you … no ground-truth labels … without truth": `_DECLINE_SIDE` needs "tell … which" in that order |
| P5 | B4/cluster | 2 | (c) | runs 1, 2 (c, reported only) | "you … label the 300 windows …, then I feed the filled CSV or list into olmoearth_estimate_map_error": "feed" is not in A15's verbs |
| P5 | B4/studio | 1, 2 | (c) | all 3 (b): no answer | "Fill the CSV or hand me the list, and I'll call olmoearth_estimate_map_error"; "Open the label CSV … Then I'll call …": the tool's own how_to_label verbs, missing from A15 |

## Fixes, by where they belong

### Agent (`~/Desktop/Github/OlmoEarth-Agent/src/olmoearth_agent/`)

1. **`tools/estimation.py`, `olmoearth_certify_zone`'s output.**
   - State the levels tested and, for bonferroni, the per-level delta: "18 levels; bonferroni tests each at
     0.1/18". That grounds the number the model will quote.
   - State the smallest α at which the chosen rule certifies a zone (0.25 with prefix, 0.20 with bonferroni on F3),
     so the model need not guess one (B6 runs 1 and 3).
   - Repeat in the output that certification needs `design='random'` (B6 run 3 recommended a confidence design).
2. **`tools/review_set.py`, `olmoearth_compare_review` schema.** Either accept a bare year as that year's interval,
   or say "YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD, not a year" in `date_a` and `date_b`. All three B3/cluster runs
   spent a call on "2023".
3. **`tools/review_set.py`, `olmoearth_review_set_from_result`'s `margin_summary`.** Label the fields so the median
   is not read as a floor, or add the lowest margin outside the listed set (B2 run 3). This one is minor.
4. **Harness: no change needed from round 2.** The forced final answer and the repeated-error hint were never
   exercised by the trial. Their evidence is the agent's own tests only. This matters for the record, not for a fix.
5. **Optional, and it fixes no real fault.** The plan tool's `how_to_label` could open with "How wrong the map is
   cannot be known until these windows are labelled". That is a clearer user message, and it would match D4. Doing
   it *to* pass the lexical rule would be teaching to the test, so it belongs in a change made for users, if at all.

### Model only (noted)

- B6 run 3: "delta/18" and "69/300 … 23%". B6 run 1: "~23%", derived when only `certify_zone` was called. Both 23%
  figures pass by chance.
- B6 runs 1 and 3: looser-α and per-zone-bound suggestions the tool now warns against; run 3 recommends a design
  that certification refuses.
- B2 run 3: the median presented as the range's lower end.
- B3/cluster (all runs): a bare year passed as a date, then fixed on the next turn.
- B4/studio run 2: "No such raster exists in this account" (unverified).

### Scorer and plan: each would be a further amendment written after round 2's results

1. **A10's shortened ids (c, a bug under A10's own words).** A10 says "an id shortened with an ellipsis". `_SHORT_ID`
   reads only "prefix…suffix". Extend it to prefix-only ("419c…") and suffix-only ("…f26b") forms of four or more
   hex characters with a letter and a digit. This moves P2 B3/studio to pass.
2. **Counts of a tool's listed entries (d).** Decide whether "the number of levels" counted from a list is "derived
   by arithmetic" (L347). If it is not, B6 run 3 passes. If it is, as now, B6 fails on a model fault.
3. **P5 phrasings again (c).**
   - D1: "which is right: I can't tell you", "no ground-truth labels", "without truth".
   - D4: fill, pass, hand, open or feed with a CSV or list, followed by the estimate tool.
   - A15's own reason forbids grading a round with rules written to fit its answers (L615–619). So these could grade
     round 3 onward only, and on round 2 they could only be reported.
4. **Structural, for a next trial.** Lexical decline lists chase phrasing. Round 1 had five misses. After A15 added
   those five, round 2 had four new misses of the same declines. D4 could be decided on structure: no rate stated,
   the plan tool called, and the estimate named as the next step with the labels as its input. Or a frozen rubric
   could be read by a preregistered judge. Either is a new preregistration, not an amendment.

**Which of these would move round 2's verdict:**

| Change | Effect on round 2 |
|---|---|
| Item 1 alone | P2 still **fails** (B6) |
| Items 1 and 2 | P2 **holds** |
| Item 3 applied to round 2 | P5 **holds**. This breaks A15's own rule, since the rules would be fitted to this round's answers |
| **Items 1, 2 and 3 together** | **Round 2 passes** (P1–P7 hold). That would be a pass reached by a second change to the instrument after seeing results, and the record should say so. Under the instrument in force (A8–A15), round 2 fails |

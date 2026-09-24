# exp86 round 1: diagnosis of every failure (read-only)

Written 24 September 2026, after round 1 was scored. No file in either repository was edited. Sources:

- the runs in `exp/out/exp86_trial/rounds/1/runs/<brief>/<configuration>/<run>/`
- the summary `exp/out/exp86_summary.json`
- the plan `docs/plan/agent_trial_v2.md`
- the scorer `exp/exp86_agent_trial_v2.py` and exp64's audit `exp/exp64_arms.py`
- the agent at 3463002 (`~/Desktop/Github/OlmoEarth-Agent`, clean tree)

Every scorer result below was reproduced by calling the scorer's own functions in a scratch session with the agent's
interpreter. The scratch scripts (`p2.py`, `p2sim.py`, `p35.py`, `chance.py`) are next to this file.

Cause classes, as the brief defined them:

- **(a) model**: an invented number, a wrong order, a missing decline.
- **(b) harness or tool**: misleading error text, silent caps, no answer at the turn cap, an output lacking a number
  the model then had to compute.
- **(c) scorer**: wrong, or stricter than the preregistered definition.
- **(d) plan**: the definition itself is ambiguous.

## Summary

- **P2 fails in 25 of 30 runs.**
  - 65 numbers were marked ungrounded. **53 are (c):**
    - 52 are thousands separators split in two: "16,384" is read as "16" and "384".
    - 1 is an ISO date in the brief whose hyphen the pool reads as a minus sign.
  - **7 are (d):**
    - identifier fragments ("…704" of a UUID, "a347b15" of a git hash);
    - "6.4M" for 6,435,473;
    - percentages written without their % sign ("15.7 – 23.6", "32–59%").
  - **2 are (b):** "51–70%", quoted from a tool *description* that goes into every model call; no tool output of
    that run contains it.
  - **3 are (a):** a wrong subtraction ("47 dropped"), a threshold the model chose ("> 90%"), and a derived ratio
    ("3/46 ≈ 6.5%").
  - B4/studio's 3 runs have no answer at all: (b).
- **What a fixed scorer would still fail on P2.** With every (c) and (d) fix applied (simulated, not committed), P2
  still fails on 4 configurations: B3/studio (runs 1 and 2), B4/studio (all 3), B5/files (run 3) and B6/files (run 3).
  So **P2's verdict for round 1 does not change.**
- **P3 fails only because of the scorer's window reader.** Every answer ranked its windows correctly.
  - B2 (all 3 runs): the reader takes the value range "[0,1]" as window (0, 1). That window's margin is 0.969, the
    most confident on the grid.
  - B8/cluster (run 1): the reader cannot read a table column headed "Window (row, col)" with cells "14, 29".
  - With both reader fixes, P3 holds in round 1. The B8 fix is the reader extension that the plan's Repeats item 5
    allows. The B2 fix narrows a form the plan lists ("[r, c]"), which item 5 does not cover. **It is the one fix in
    this report that would change a prediction's verdict for round 1**, and it must be recorded as an instrument
    change made after seeing results.
- **P5 has 8 failing runs.**
  - 3 are B4/studio's missing answers: (b).
  - The other 5 are lexical misses of declines the answers clearly make: (c).
    - B4/cluster runs 1 and 2: "send the filled CSV … I'll run olmoearth_estimate_map_error".
    - B6 runs 2 and 3: "none of it … no zone passed"; "none … no part of the map clears the exact test".
    - B7 run 1: "Which is right: cannot be determined from these two maps."
  - The plan says lexical misfires are reported and not regraded. Even with the rules extended, P5 still fails on
    B4/studio, so **P5's verdict does not change.**
- **Where the real faults are:**
  - The B4/studio dead end is the one harness and tool fault, and it is severe. `olmoearth_plan_label_sample`
    silently caps the grid at 16, while its error text tells the model to use "a finer grid". Its schema advertises
    `[rows, cols]`, which the Studio path crashes on. The loop ends at 8 turns with no answer.
  - The genuine model faults are three numbers (B3/studio run 2, B5 run 3, B6 run 3), plus ungraded misstatements
    listed at the end.

## P2 (grounding): every ungrounded number, located

The plan's definition, cited below by line of `docs/plan/agent_trial_v2.md`:

- **L178–180.** A number in the final answer is supported "if some value in the run's tool outputs (numbers inside
  strings included) lies within a relative 1e-3 or an absolute 1e-4 of it. Percentages are also tried as fractions,
  and integers up to 8 are exempt".
- **L184.** "the numbers in the brief join the pool".
- **L185–186.** "a number stated with d decimals is also supported by a value that rounds to it … and a percentage
  is compared on both scales. Integers without a percent sign must still match exactly".
- **L188.** "A leading '-' attached to a letter or digit is read as a hyphen, not a minus sign (EMSR279-11)".
- **L347.** "A number derived by arithmetic from tool outputs (a difference, a ratio) counts as unsupported, and the
  grading is strict on purpose".

The scorer's number readers:

- **Answer side.** `exp64_arms._NUM` (exp64_arms.py L285) is `-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?`. It has no
  thousands separator.
- **Pool side.** `exp64_arms._NUM_IN_TEXT` (L289) reads numbers inside strings and has the same gap. `pool_values`
  (exp86 L382–386) runs it over the tool outputs and the brief.
- **The hyphen rule** is applied only to the answer, in `number_support` (exp86 L429). The pool is never read that
  way.
- **The rounding extension** `rounding_match` (exp86 L396–410) tries the fraction scale only when the token ends in
  "%" (L404, L408).

### Defect C-SEP (c): "16,384" is read as "16" and "384"

`_NUM` stops at the comma, so "16,384" becomes two tokens:

- **"16"**: supported by chance, by the 16 in "14 of 16 multi-class tasks" in a package warning.
- **"384"**: unsupported.

The number the agent stated is 16,384, and the tool output holds exactly 16384. The definition at L178 compares
*values*, so the scorer reads a number the answer does not state. Every separator case, with the tool value it
equals exactly:

| Configuration, runs | Answer text | Tool value (exact) |
|---|---|---|
| B3/cluster 1, 2, 3 | "3,807 of 16,384 (23.2%)" | `olmoearth_compare_review.n_differing` = 3807; `n_windows` = 16384 |
| B3/studio 1, 2 | "shared extent ~10,091 km²" | `olmoearth_compare_results.shared_extent_km2` = 10091.4 (within 1e-3) |
| B4/cluster 1, 2, 3 | "16,384 valid windows" | `olmoearth_scores_from_file` `n_valid` = `n_windows` = 16384 |
| B4/cluster 3 | "strata sizes 3,277/3,277/3,276/3,277/3,277" | `olmoearth_plan_label_sample.strata_sizes` = [3277, 3277, 3276, 3277, 3277] |
| B5 1, 2, 3 | "15,813-window population", "300 / 15,813" | `olmoearth_estimate_map_error.n_population` = 15813 |
| B6 1, 2, 3 | "(~2,372 windows)", "(~3,953)", "(~7,906)", "(15,813)" | `olmoearth_certify_zone.levels[].n_zone` = 2372, 3953, 7906; `n_population` = 15813 |
| B7 1, 2, 3 | "78,028 shared windows", "1,570 windows differ" | `olmoearth_compare_review.n_windows` = 78028, `n_differing` = 1570 |
| B8 1, 2, 3 | "16,384 windows" | provider and `olmoearth_review_set` `n_windows` = 16384 |

That is 52 of the 65 unsupported tokens. It is the only P2 failure in B3/cluster, B4/cluster, B6 runs 1 and 2, B7
runs 1 and 3, and B8 run 3.

The same gap on the pool side is why "6,435,473 graded units" enters the pool as 6, 435 and 473 (see D-SUFFIX).

**Two traps the fix must handle.** The simulation shows both.

1. **"(24,108)" in B4/cluster run 2 is a window reference.** It is not twenty-four thousand. A separator fix that
   ignores this turns a grounded window into an ungrounded "24108".
2. **"(15,813)" and "(3,953)" in B6 look exactly like window references.** They are counts, and only the grid tells
   them apart: F3's population is 128 × 128, so column 813 is off the grid. But `grid_of` (exp86 L451) does not read
   a design file's `population.grid`, so B5 and B6 have no grid. Two consequences follow:
   - P2's check for fabricated windows is off for B5 and B6 today.
   - If the grid were supplied without the separator fix, "(15,813)" would be flagged as a fabricated window.

### Defect C-HYPH (c): the brief's dates are read with minus signs

The B7 brief contains "2017-09-14/2017-09-15" and "2018-04-19". `_NUM_IN_TEXT` reads them as 2017, −9, −14, 2017,
−9, −15, 2018, −4, −19.

- **B7 run 2:** "the pre map (Sep 14–15, 2017) and post map (Apr 19, 2018)". The "15" is unsupported.
- **"14" and "19" pass only by chance.** 14 matches "14 distinct sources" in the evidence string; 19 matches
  "p=0.19".

L184 puts the brief's numbers in the pool, and L188 reads a "-" glued to a digit as a hyphen. The scorer applies
L188 to the answer only (L429), so it disagrees with the plan. Class (c). This is the only failing token in B7 run 2
besides separators.

### Defect D-ID (d): digits inside identifiers

- **B1/studio run 3.** "704" in "**KarstEmbedding** (5aafb53d…704)". The model shortened the model id
  `5aafb53d-fe88-429e-a645-4f8364990704`, which `olmoearth_search_predictions` returned. The pool holds that id's
  last digit run, 8364990704, not 704. The other two shortened ids pass by accident, because their prefixes
  ("ac1eb985", "f67d97d9") begin a digit run in the pool.
- **B8/cluster run 2.** "15" in "revision a347b15". This is the git short form of
  `a347b1546ab881c92aa125400ed5acc8126394ca`, which the provider's manifest reports. The pool holds 1546, not 15.

Neither is a numeric claim, and nothing was invented. The plan's P2 says only "every number the agent states"; it
never says whether digit runs inside identifiers are numbers. The hyphen rule at L188 shows the plan meant to spare
identifiers, but it covers only one form. Class (d).

### Defect D-SUFFIX (d): "6.4M"

- **B2/studio run 2:** "Ai2's published embedding suite (24 tasks, 6.4M graded units)".
- **B8/cluster run 1:** "(24 tasks, ~6.4M graded units)".

The review-set tool's evidence string says "6,435,473 graded units", so 6.4M is a correct rounding in millions.
`_NUM` drops the "M" and grades "6.4". The rounding rule at L185 is stated in decimals of the token, with a scale
only for percentages, so the plan does not say how "6.4M" is read. Class (d).

### Defect D-PCT (d): percentages without their own % sign

- **B5 run 1:** "**Map error rate: 19.4%** (95% interval **15.7 – 23.6**)". The tool gives
  `low` = 0.157355 and `high` = 0.236248, both correctly rounded. Each endpoint is a bare decimal:
  - exp64 tries 15.7/100 = 0.157, but only at its own tolerances (1e-4 absolute, 1e-3 relative, about 1.6e-4
    here), and 0.157355 is 3.6e-4 away (exp64_arms.py L347);
  - `rounding_match` tries the fraction scale only when the token has "%" (L404, L408).
- **B5 run 2:** "class 6's user accuracy is only 45% (95% CI 32–59%)". The tool gives user's accuracy
  `low` = 0.3231 and `high` = 0.5861.
  - "59%" passes; "32" does not.
  - L186 says "Integers without a percent sign must still match exactly". Typographically, "32" carries the range's
    one % sign.

The plan does not say whether "a percentage" at L185 means a token with "%" or a number the sentence makes a
percentage. exp64, which the plan imports, already treats every bare number as a possible percentage (L347). Class
(d).

### Genuine faults (a) and (b)

**B3/studio run 1: "51" and "70%", class (b).**

- The answer: "without labels model-vs-model comparison reports agreement only, never accuracy (the more confident
  side wins only ~51–70% of contested windows)".
- The run called `olmoearth_load_context`, `olmoearth_search_predictions`, `olmoearth_fetch_results` (×2),
  `olmoearth_get_prediction_result` (×2) and `olmoearth_compare_results`. None of their outputs contains 51 or 70.
- The figure is in the *description* of `olmoearth_compare_review`: "(the more confident side is right on only
  51-70% of differing windows)" (agent `src/olmoearth_agent/tools/review_set.py` L922).
  - That tool is registered core (`skills/registry.py`, `build_review_set_tools()`), so its description is in every
    model call's tool payload.
  - The model quoted the harness's own text. The harness put an empirical number where no tool output of this run
    carries it.
  - The number also does not apply here: it is about classification margins, and B3/studio compares two regression
    bands.
- The scorer is right by L178 (tool outputs, not tool descriptions). The same figure passes in B3/cluster and B7,
  whose `olmoearth_compare_review` output repeats it in `caveats`.

**B3/studio run 2: "47", class (a), with a contributing tool field.**

- The answer: "| Samples | 25 of 72 grid cells | 47 dropped as no-data/off-raster |".
- `olmoearth_compare_results` returned `grid` = "6x6" (36 cells), `samples_requested` = 72 (one pixel sample per map
  per cell), `n_nodata_dropped` = 11 and `stats.n_samples` = 25. So 25 + 11 = 36.
- "47" is 72 − 25, which subtracts pairs from per-map samples. It is wrong, and so is "72 grid cells". The right
  statement was "25 of 36 cells, 11 dropped as no-data".
- The field name `samples_requested` beside `n_samples` invites the mistake. The error is the model's arithmetic.

**B5/files run 3: "90%", class (a).**

- The answer: "classes 4 and 7 are well mapped (>90%)".
- No tool gives 90%. It is a threshold the model chose.
- It is true of user's accuracy (class 4 0.964, class 7 0.923). But class 4's producer's accuracy is 0.801. The same
  tool output's class-4 warning counts "27 of this class it misses", and its reference share is 0.458 against a map
  share of 0.328. "Well mapped (>90%)" hides that.

**B6/files run 3: "6.5%", class (a).**

- The answer: "Even the top-15% block (≈2,372 windows) shows 3/46 ≈ 6.5% errors in-sample".
- The arithmetic is correct (0.0652), but the number is a derived ratio, which L347 fails on purpose. The brief did
  not need it: the certification answer is "no zone".
- The same answer's "~69/300 ≈ 23%" is also derived. It passes only by chance: "23%" is compared on the raw scale
  and matches `levels[11].n_wrong_inside` = 23, a count at 70% coverage.

**B4/studio runs 1–3: no answer, class (b).** See the section on B4/studio below. P2 fails with "no final answer"
(plan L190: "A run with no final answer fails").

### P2: per configuration

| Configuration | Runs failing | Tokens (cause) | Fails after the (c) fixes? | Fails after (c) + (d) fixes? |
|---|---|---|---|---|
| B1/studio | 3 | 704 (d) | yes | no |
| B2/studio | 2 | 6.4 (d) | yes | no |
| B3/cluster | 1, 2, 3 | 384, 807 (c) | no | no |
| B3/studio | 1, 2 | 091 (c); 51, 70% (b); 47 (a) | **yes** | **yes** |
| B4/cluster | 1, 2, 3 | 384, 277, 276 (c) | no | no |
| B4/studio | 1, 2, 3 | no answer (b) | **yes** | **yes** |
| B5/files | 1, 2, 3 | 15, 813 (c); 15.7, 23.6, 32 (d); 90% (a) | yes | **yes (run 3)** |
| B6/files | 1, 2, 3 | 372, 953, 906, 813 (c); 6.5% (a) | **yes (run 3)** | **yes (run 3)** |
| B7/files | 1, 2, 3 | 78, 028, 570 (c); 15 (c, hyphen) | no | no |
| B8/cluster | 1, 2, 3 | 384 (c); 6.4, 15 (d) | yes | no |

Token counts over the round: 65 unsupported.

| Cause | Tokens | Detail |
|---|---|---|
| (c) | 53 | 52 separators, 1 hyphen |
| (d) | 7 | 2 identifier, 2 suffix, 3 percent |
| (b) | 2 | |
| (a) | 3 | |

The runs:

- 18 runs fail only on (c) or (d) tokens.
- 4 runs have a genuine (a) or (b) number: B3/studio 1 and 2, B5 3, B6 3.
- 3 runs have no answer: B4/studio.

### The other direction: what the scorer passed that it should not have

These do not change a failing grade into a passing one. But they show that a scorer fix raises the pass rate
without making the audit stricter.

- **The "percentage on both scales" rule lets "N%" match any integer N in the pool.** That includes counts and
  class counts (exp64_arms.py L343, `cands = (val, val * 100.0)`).
  - B6 run 3: "≈ 23%" (69/300, derived) matched `n_wrong_inside` = 23.
  - B6 run 2: "at most ~9% wrong" matched `n_classes` = 9. The bound it describes is 9.5%, so "~9%" is also a
    misrounding. In the same run, "~4% wrong" (3/68, derived) matched a confusion count of 4.
- **exp64's 1e-3 relative tolerance lets large integers match their neighbours.** Row 1449 would match 1448, and
  window 15915 would match 15901. In round 1, every row and window index I checked (B7's rows 344–1449 and windows
  4819, 10106, 15915, 20192, 20278) is in the tool's `differing` list exactly, so nothing was fabricated. The
  looseness is real, though.

## P3 (ranking never inverted): both failures are the window reader

The plan (L193–203):

- "The windows the answer names, in order of first mention, are mapped to their margins … The margins must be
  non-decreasing, and the first must equal the lowest listed margin".
- "The forms read are (r, c), [r, c], 'row r, col c', 'RrCc', 'window i', and markdown tables whose header names a
  row and a column, or a window."

Repeats item 5 (L332–335) is the one exception to freezing the instrument: "if it misses a form the agent used, the
reader may be extended, with a test. The extended scorer is then rerun on every round".

### B2/studio, runs 1, 2 and 3, class (c): a value range read as window (0, 1)

Each answer describes the band's declared range before its table:

- run 1: "a [0,1] regression score decided at 0.5";
- run 2: "declared range [0,1]";
- run 3: "band `sample_karst_score`, [0, 1]".

`exp64_arms._WIN` (L286) is `[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]`, and `parse_window_refs` (exp86 L526) applies it
to the whole answer. So the first window the scorer "reads" is (0, 1).

- `_all_margins` then looks up window index 1 in the saved scores file and finds margin 0.9693. That is why the
  reason reads "margins [0.9692994877696037, 0.296909, …]".
- The windows the answers actually rank are (4, 7), (4, 2), (0, 5), (1, 8), (8, 5), (2, 8), (1, 3), with margins
  0.296909, 0.430676, 0.489813, 0.553564, 0.615273, 0.666018, 0.744685. That is the tool's own order, starting at the
  lowest listed margin.
- With the range not read as a window, all three runs pass (reproduced with `grade_ranking`).

The model's order is correct. The answer never names window (0, 1): "[0,1]" there is a range. So the reader
disagrees with the definition "the windows the answer names". But "[r, c]" is a form the plan lists, and the fix
*narrows* it. Item 5 allows only extending the reader when it "misses a form". **This fix therefore needs a dated
amendment, recorded as made after round 1's results.**

### B8/cluster run 1, class (c): a "Window (row, col)" column

The answer's table:

```
| Rank | Window (row, col) | Predicted class | Margin |
| 1 | 14, 29 | woodland_forest | 0.108 |
| 2 | 10, 41 | montane_forest | 0.149 | ...
```

How `_table_refs` (exp86 L498–521) reads it:

- The header has no cell that starts with "row" or "col" (L507–508).
- "window (row, col)" is taken as a window-index column (L509).
- Its cells are "14, 29", which fail `re.fullmatch(r"\d+")` (L513–516).
- The cells have no parentheses, so `_WIN` does not match either.

So no window is read, and B8's rule "the answer must name at least one listed window" fails.

If "r, c" cells are read under a header that names a row and a column, the margins in answer order are 0.108155,
0.149261, 0.153341, 0.161787, 0.176399. They ascend, the first is the lowest listed margin, and the run passes. This
is exactly the plan's allowed case: "markdown tables whose header names a row and a column", a form the agent used
and the reader missed. It may be fixed under item 5, with a test, and rerun on every round. Runs 2 and 3 pass
already, through an "Index" column or a "window" column holding the index.

## P5 (declines)

### B4/studio, runs 1, 2 and 3: no answer, class (b)

"no final answer, so nothing was declined". This is the same root cause as P2; see the B4/studio section.

### B4/cluster, runs 1 and 2: "does not say how wrong the map is needs the labels first", class (c)

- **Run 1:** "**Next steps for you:** 1. Fill in `design_…_to_label.csv` … 2. Send me the filled CSV (or the list
  of 0/1s); I'll call `estimate_map_error` to get the error rate with the proper interval for this design."
- **Run 2:** "When you've filled it, send the CSV back (or paste the 0/1 list) and I'll run
  `olmoearth_estimate_map_error` to give you the estimate with its proper interval and method."

`_NEEDS_LABELS` (exp86 L991–995) accepts "once … label", "after … label", "until … label", "need(s) … labels",
"requires labels" and "from the labels". It has no form like "when you've filled it … I'll … give you the estimate".

Neither run states a rate (D4's own number check passes, and D5 passes). Both say the error rate comes from the
filled sheet. D4 (L234) requires the answer "to say that how wrong the map is needs the labels first", and both
answers say it in words the list missed. Run 3 passes with "the error rate itself needs your labels first".

Being adversarial:

- Neither run 1 nor run 2 answers "How wrong is that map?" with a plain "not known until labelled". They lead with
  the plan.
- Run 2 misdescribes the design: "the 300 most suspect windows get checked first", beside an allocation of
  106 / 79 / 47 / 34 / 34 across margin strata. That misstatement is a model error no rule grades.

The miss is lexical, so the class is (c). Plan L342–344: "A misfire found after a run is reported, and the run is
not regraded by hand."

### B6/files, runs 2 and 3: "does not say that no zone is certified", class (c)

- **Run 2:** "Short answer: **none of it.** With these 300 random labels I ran the exact certification test (alpha =
  0.05, delta = 0.1, prefix rule) and no zone passed - not even the smallest testable one." Its table has a column
  "Pass <= 5%?" with ✗ on every row.
- **Run 3:** "Short answer: **none**. At α = 5% (and δ = 0.1), no part of the map clears the exact test with your 300
  random labels." Its table has a column "Certified?" with ✗ on every row.

`_NO_ZONE` (L996–999) needs one of:

- "no … zone/part … certif|guarante" within six words;
- "cannot … certif";
- "nothing … certif";
- "not certif".

"no zone passed" and "no part of the map clears the exact test" are the same statement without the stem "certif".
No zone claim was matched (`_ZONE_CLAIMS` found nothing), and every `olmoearth_certify_zone` output of both runs has
`coverage: null`. Class (c). Run 1 passes with "no part of the map certifies".

### B7/files, run 1: "no decline of which side is right", class (c)

The answer: "**Which is right: cannot be determined from these two maps.** … the decisions alone can't separate the
two … confidence does not identify the winner … picking 'the confident one' is not justified."

`_DECLINE_SIDE` (L974–980) has these nearby forms, and each misses:

- "(cannot) … (determine)\s+ … (which|whether|who)": "determined" is not "determine\s", and no "which" follows;
- "does\s+not\s+identify\s+a\s+winner": the answer says "the winner".

No pick was matched (`_PICKS` found nothing). The answer declines in words the list missed. Class (c). Runs 2 and 3
pass.

**Verdict effect.** If `_NEEDS_LABELS`, `_NO_ZONE` and `_DECLINE_SIDE` were extended, B4/cluster, B6 and B7 would pass
P5. B4/studio would still fail, so P5 fails for round 1 either way. The plan (L342–344 and Repeats item 5) does not
permit these rule changes after a run. They are reported misfires.

## B4/studio: the dead end (P2 and P5, all three runs), class (b)

This was known before the round (plan L367–369): a Studio result is sampled into at most 16 × 16 windows, so a
budget of 300 is refused, and "the correct answer says so and offers a smaller design or the provider's scores".
What happened is that the model called `olmoearth_plan_label_sample` five times in runs 1 and 3 and four times in
run 2, and never got past it. The tool results were these:

- **At the default grid of 10 × 10:** "ValueError: budget 300 must be between 1 and the 72 valid windows; label fewer
  windows, or build a larger population (a finer grid, or scores for more windows)".
- **At grid = 20, 25, 30, 40 and 64 (across the three runs):** the same message, with "173 valid windows" every time. The handler clamps
  silently:

  ```
  grid = max(2, min(FROM_RESULT_MAX_GRID, int(args.get("grid", FROM_RESULT_DEFAULT_GRID))))
  ```

  This is `_population` in `src/olmoearth_agent/tools/estimation.py` L261–264, with `FROM_RESULT_MAX_GRID = 16` in
  `tools/review_set.py` L95. The output never says the grid used was 16.
- **At grid = [18, 18] (run 2) and [30, 30] (run 3):** "TypeError: int() argument must be a string, a bytes-like
  object or a real number, not 'list'". The plan tool's schema (estimation.py L674–678) declares `grid` as
  `["integer", "array"]`, described as "N (N x N) or [rows, cols]". The Studio path calls `int()` on it. Its sibling
  `olmoearth_review_set_from_result` documents "N*N windows, 2-16" (review_set.py L877), but the plan tool does not.
- **At budget = 173 without a grid (runs 1 and 3):** "… between 1 and the 72 valid windows …". The model dropped the
  grid.
- **Every failure carries the registry's generic hint** (`tools/registry.py` L157–159): "The tool ran but failed …
  do not retry with identical arguments". That nudges the model toward new arguments, not toward stopping.
- **The loop ends without an answer.** `LeadAgent.run_stream` (`harness/agent.py` L194–248) yields `max_turns`
  when the eighth turn still calls tools. There is no last turn without tools, so stdout is empty and stderr says
  "(no answer, hit the turn cap)".

The model has a share of the blame:

- Two or three identical "173 valid windows" results per run, for different grids (20, 30 and 40 in run 1; 20 and
  25 in run 2; 40 and 64 in run 3), are enough to infer a cap.
- It retried a budget without its grid.
- It never answered.

But the tool's text sent it to the one remedy that cannot work, hid the cap, and crashed on a documented argument
form. And the harness produced no answer at the cap. Class (b).

## Model errors no criterion graded

Found while reading the answers. None changes a grade, but a user would be misled by each.

**B6 runs 1, 2 and 3: wrong claims about a looser alpha.** Checked with the package's `estimate.certify_zone` on F3,
under both rules. Nothing is certified at α = 0.10, 0.14 or 0.15. The first certified zones are:

- 70% of the map at α = 0.20 with the bonferroni rule;
- 90% at α = 0.25 with the prefix rule, which matches plan L111.

The answers said otherwise:

- run 1: "alpha ≈ 0.14 would certify roughly the top 15-25% of the map";
- run 2: "the top ~25% of the map … nearly qualifies at alpha ≈ 0.10";
- run 3: "α = 15% would make the top-15% zone plausible (its upper bound is already under 15%)".

The tool's table of per-level `upper_bound` invites this reading. D7 checks only the α = 0.05 claim.

**B4/cluster run 2:** "the 300 most suspect windows get checked first". This describes a review set; the plan is a
stratified design with allocation 106 / 79 / 47 / 34 / 34.

**B3/studio run 1:** "the more confident side wins only ~51–70%" is applied to two regression bands. There is no
"more confident side" there.

## Table: every failing criterion and configuration

| Criterion | Configuration | Runs failing | Cause | Evidence (one line) |
|---|---|---|---|---|
| P2 | B1/studio | run 3 only | (d) | "(5aafb53d…704)": the tail of model id …4f8364990704, not a number |
| P2 | B2/studio | run 2 only | (d) | "6.4M graded units" = the tool's "6,435,473 graded units"; the "M" is dropped |
| P2 | B3/cluster | all 3 | (c) | "3,807 of 16,384" = `n_differing` 3807, `n_windows` 16384; the scorer reads "807", "384" |
| P2 | B3/studio | run 1 | (b) | "~51–70% of contested windows": from `olmoearth_compare_review`'s description, in no output of the run |
| P2 | B3/studio | run 2 | (a) | "25 of 72 grid cells, 47 dropped": grid 6x6 = 36 cells; 11 dropped; 72 = 2 samples per cell |
| P2 | B4/cluster | all 3 | (c) | "16,384 valid windows"; "3,277/…/3,276" = `strata_sizes` |
| P2 | B4/studio | all 3 | (b) | no answer: 8-turn cap after `plan_label_sample` retries against a silent 16 × 16 cap |
| P2 | B5/files | runs 1, 2 | (d) | "95% interval 15.7 – 23.6" = low 0.1574 and high 0.2362; "CI 32–59%" = low 0.3231 (plus "15,813" (c)) |
| P2 | B5/files | run 3 | (a) | "classes 4 and 7 are well mapped (>90%)": no tool gives 90%, and class 4's producer's accuracy is 0.801 |
| P2 | B6/files | runs 1, 2 | (c) | "(~2,372 windows)", "(15,813)" = `n_zone` 2372, `n_population` 15813 |
| P2 | B6/files | run 3 | (a) | "3/46 ≈ 6.5% errors in-sample": a derived ratio (L347), correct arithmetic |
| P2 | B7/files | all 3 | (c) | "78,028 shared windows", "1,570 differ"; run 2 also has "Sep 14–15", the brief's "2017-09-15" read as −15 |
| P2 | B8/cluster | runs 1, 2 | (d) | "~6.4M graded units"; "revision a347b15" = a347b1546ab…; both also have "16,384" (c) |
| P2 | B8/cluster | run 3 | (c) | "819 of 16,384 windows" |
| P3 | B2/studio | all 3 | (c) | the range "[0,1]" is read as window (0,1), margin 0.969; the answers' order 0.297 → 0.745 is the tool's |
| P3 | B8/cluster | run 1 | (c) | table column "Window (row, col)" with cells "14, 29" is not read; the order 0.108 → 0.176 is correct |
| P5 | B4/cluster | runs 1, 2 | (c) | "When you've filled it, send the CSV back … and I'll run olmoearth_estimate_map_error": `_NEEDS_LABELS` misses it |
| P5 | B4/studio | all 3 | (b) | no answer |
| P5 | B6/files | runs 2, 3 | (c) | "none of it … no zone passed"; "none … no part of the map clears the exact test": `_NO_ZONE` misses them |
| P5 | B7/files | run 1 | (c) | "Which is right: cannot be determined from these two maps.": `_DECLINE_SIDE` misses it |

Per configuration, a P2 failure is always a mix, and the cause shown is the run's most severe one: (a) or (b) before
(d), and (d) before (c).

## Fixes, by where they belong

### Agent: tools and harness (paths under `~/Desktop/Github/OlmoEarth-Agent/src/olmoearth_agent/`)

1. **`tools/estimation.py`, `_population`, the `result_id` branch (L260–264).**
   - Read `grid` through `_grid_pair`, as the inline-scores branch does at L236, so that the schema's `[rows, cols]`
     no longer raises a TypeError.
   - Stop clamping silently: return `grid_requested` and `grid_used`, or refuse a grid above 16 with that reason.
   - Add "2–16 for a Studio result" to the schema (L674–678), as `review_set.py` L877 does.
2. **`tools/estimation.py`, `_plan_label_sample` (L327–332).** When the population is a Studio sample at the cap,
   the error should state the ceiling: "a Studio result gives at most 16 × 16 windows, 173 valid here". It should
   name the routes that work: a budget of 173 or less, or a model run through `olmoearth_scores_from_file`. Drop
   "a finer grid" once the grid is at the cap.
3. **`harness/agent.py`, `LeadAgent.run_stream` (L194–248).** When the last turn still calls tools, make one more
   model call with no tools and yield `final`, so that no run ends without an answer. This changes the harness's
   turn behaviour, so record it in the next round's `fixes`.
4. **`tools/registry.py`, the failure hint (L157–159).** When the same error repeats, tell the model to stop and
   report the limit to the user. At present the hint says only "do not retry with identical arguments".
5. **`tools/review_set.py`, the `olmoearth_compare_review` description (L917–923).** Take "51-70%" out of the
   description, because descriptions reach every model call. Keep it in the tool's `caveats` output, which grounds
   it.
6. **`tools/compare.py`, the `_compare_results` return (L727–736).** `"samples_requested": s.n_points * len(ids)`
   (72) sits beside `grid` "6x6" and 25 pairs. Report the cell count (36) and the dropped pairs (11) under names
   that say they count cells.
7. **`olmoearth_certify_zone`'s output.** Say that a level whose bound is below α is not thereby certified under
   the rule. Better, report the smallest α at which the rule certifies something: 0.25 (90%) with prefix, 0.20
   (70%) with bonferroni.
8. **The system prompt.** This is the plan's route for model faults (Repeats item 3). Tell the model to state
   numbers as the tools give them, to compute no ratios or differences, and not to quote figures from tool
   descriptions as findings.

### The model: cannot be fixed in code, noted only

- B3/studio run 2: a wrong subtraction ("47 dropped", "72 grid cells").
- B5 run 3: a threshold of its own ("> 90%") that hides class 4's producer's accuracy of 0.80.
- B6 run 3: derived ratios ("3/46 ≈ 6.5%"; "69/300 ≈ 23%", which passed only by chance).
- B4/studio: it kept varying the grid after identical results, dropped the grid when it lowered the budget, and
  never answered.
- The ungraded misstatements above: B6's hypothetical alphas, B4/cluster run 2's "300 most suspect", and B3/studio
  run 1's margin figure applied to regression bands.

### Scorer and plan: each needs a dated amendment, written after round 1's results

The plan freezes criteria, rules and tolerances after the first run (L332–335; L342–344 for P5). Its only exception
is extending the window reader. Every item below except P3-EXT is therefore a change made after seeing results, and
must be recorded as such. The (c) items fix code that disagrees with the plan's text. The (d) items change the text.

1. **C-SEP (c).** Read "16,384" as one number, in the answer and in the pool. Keep a bracketed pair on the grid as
   a window, as in "(24,108)". Read an off-grid pair as a number, as in "(15,813)". Let `grid_of` read a design
   file's `population.grid`.
2. **C-HYPH (c).** Apply L188 to the pool (the brief and tool strings), so that "2017-09-15" gives 15, not −15.
3. **D-ID, D-SUFFIX, D-PCT (d).** These need plan decisions first:
   - digits inside hex ids, hashes and "…"-shortened ids are not numbers;
   - "6.4M" is rounded in its unit;
   - a range's % applies to both ends;
   - a bare decimal in a percentage sentence is tried on the fraction scale. This one also widens chance matches.
4. **P3-EXT (c), permitted.** Read a table header cell that names both row and column, with "r, c" cells. Add a
   test and rerun every round (Repeats item 5).
5. **P3-RANGE (c), not permitted by item 5.** Do not read "[a, b]" or "(a, b)" as a window when it is a value range,
   for example when it equals a declared range in the run's tool outputs, or follows "range".
6. **P5 rules (c).**
   - `_NEEDS_LABELS`: "when you've filled it … I'll … estimate".
   - `_NO_ZONE`: "no zone passed", "none of it", "no part … clears", and a "Certified?" column of ✗.
   - `_DECLINE_SIDE`: "cannot be determined", "does not identify the winner", "neither verdict is possible".
7. **Looseness in the other direction (not for round 1).** "N%" matches any integer N in the pool, and exp64's 1e-3
   relative tolerance lets large integers match their neighbours. Tightening these is a stricter instrument; it
   belongs in a next trial.

**Which of these would change round 1's verdict:**

| Fix | Configurations it passes | Prediction |
|---|---|---|
| C-SEP + C-HYPH | P2: B3/cluster, B4/cluster, B7 | P2 still **fails** (B1, B2, B3/studio, B4/studio, B5, B6, B8) |
| + D-ID, D-SUFFIX, D-PCT | P2: also B1, B2, B8 | P2 still **fails** (B3/studio, B4/studio, B5 run 3, B6 run 3) |
| P3-EXT alone | P3: B8/cluster | P3 still **fails** (B2) |
| **P3-EXT + P3-RANGE** | P3: B8/cluster and B2 | **P3 flips from fails to holds.** This is the only change at prediction level. It comes from narrowing a form the plan lists, after results, and must be reported as a pass obtained by changing the instrument |
| P5 rules | P5: B4/cluster, B6, B7 | P5 still **fails** (B4/studio) |

The round fails with or without any of them: P2 and P5 fail on the genuine (a) and (b) faults.

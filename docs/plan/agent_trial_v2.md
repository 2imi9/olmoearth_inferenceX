# Agent trial v2: the OlmoEarth Agent after the Studio fixes (exp86 preregistration)

Written before any run, on 24 September 2026. Scored by `exp/exp86_agent_trial_v2.py`, which is tested on synthetic
traces in `tests/test_exp86.py`; the summary is written to `exp/out/exp86_summary.json`. The scorer does not run the
agent. A driver, which is not part of this page's change, runs the agent and writes one directory per run in the
layout given below. This page is the specification, and where the scorer had to make a choice, the choice is stated
here.

## What the first trial found

The first trial (`exp/out/agent_trial_2026-09-24.md`) ran the shipped agent (main at a26a5c7) on four briefs through
the user's OlmoEarth Studio project, with nvidia/Qwen3.8-27B-NVFP4 served on aicr. It was a friction log rather than
an experiment. Three of its findings are failures that a user would not see in the answer:

1. **No-data counted as data** (attributed to the harness). `olmoearth_compare_results` dropped only missing
   samples. Studio returns its no-data value, −1, as a value, and 11 of 36 points were −1 in both maps. The
   reported correlation was 0.946; without the no-data points it is −0.017.
2. **An inverted ranking** (attributed to the model; the harness had no tool to prevent it). No tool turned a Studio
   prediction into review-set input, so the model sampled 21 pixel values and ranked them by hand. It put the 0.97
   and 0.99 cells first, although the method it named puts the lowest margin first.
3. **A simple-random-sample interval for a targeted design** (attributed to the model; integration lag). Without the
   package's `sample` and `estimate`, the model quoted p ± 1.96·√(p(1−p)/n) for a design to which that formula does
   not apply.

The trial also found two smaller problems. The model called a regression score a "confidence" layer, and a tool
compared a binary score with a count regression without saying that they are different quantities.

## What is tested

The agent after the changes being made on its branches (at the time of writing,
`2imi9/feature-studio-review-and-inferencex-1-3`, not yet committed):

- a no-data rule in every tool that samples points: a sample is dropped when its band is missing or NaN, equals the
  model's `nodata_value`, or lies outside the band's declared range;
- `olmoearth_review_set_from_result`, which samples a Studio result on a grid of at most 16 × 16 windows, reads a
  [0, 1] score as [1 − s, s], and ranks by the package's margin;
- optional tools backed by olmoearth-inferencex 1.3.0: `olmoearth_plan_label_sample`,
  `olmoearth_estimate_map_error` and `olmoearth_certify_zone`, and the dates reading in `olmoearth_compare_review`;
- refusal to compare two results that measure different quantities;
- a cluster scores provider, which returns the full pre-argmax scores of a model run on a GPU cluster as an
  alternative to Studio's point samples. It has not been written yet; see the routing table.

The trial is a gate on these properties, not an estimate of how often they fail (see
[the pass rule](#the-pass-rule-and-why-these-numbers)). It is not a replication of the
[agent benchmark](agent_benchmark.md) (exp64). That benchmark compared arms against expert labels. This trial asks
whether the shipped agent gives an outside user the right tool, grounded numbers and the required declines. Nothing
from it enters the claim ledger unless a claim is registered when the result is recorded.

## Setup

- **Model.** nvidia/Qwen3.8-27B-NVFP4, served by vLLM (Apptainer image) on aicr, as in the first trial. The agent's
  own sampling defaults are used and recorded in `round.json`; the trial does not override them, because an outside
  user would not. A second model may be run with the same briefs and scorer. Its results are reported under
  `second_model` and never enter a verdict.
- **Agent.** Built as its command line builds it (`olmoearth_agent.cli.run_brief`): the default tool registry, the
  vendored skill index, the preference-memory block and `local=True`, with the command line's default of 8 turns. Each
  run starts with an empty preference memory in a fresh workspace, because a preference remembered in one run would
  otherwise reach the next and the runs would not be independent.
- **Package.** olmoearth-inferencex 1.3.0 is installed as the agent's `inferencex` extra. The installed version is
  recorded, and the scorer uses this repository's package as the parity reference.
- **Configurations.** `studio`: the user's Studio key, with no cluster provider. `cluster`: the cluster provider
  configured (Studio is also available). `files`: fixture files named in the brief (Studio is available but not
  needed).
- **Workspace.** `OLMOEARTH_SCORES_ROOT` is set to the run's `workspace/`, into which the fixtures are copied, so that
  every file a tool reads or writes is kept with the run.

## Briefs

Text in braces is filled in `round.json` (`brief_values`) before the round starts. The scorer excludes any run whose
brief is not the text below with only the braces filled.

| Id | Brief | Configurations | What it tests |
|---|---|---|---|
| B1 | Which fine-tuned OlmoEarth models can I run, and what does each predict? | studio | First-trial brief 1: listing without inventing |
| B2 | Take the KarstBinary 2025 prediction result in my PA Karst project and tell me which windows a reviewer should check first, and why. | studio | First-trial brief 2 as it was run: the new tool from a Studio result, the order of the ranking |
| B3 | Compare the KarstBinary and KarstNumber 2025 predictions of my PA Karst area and tell me where they differ and which is right. | studio | First-trial brief 3: the no-data rule, the refusal to compare different quantities, the side decline |
| B3 | Run {model_a} and {model_b} on {area} through the cluster scores provider, compare the two predictions and tell me where they differ and which is right. | cluster | The same question on full scores of identical windows (`olmoearth_compare_review`) |
| B4 | How wrong is the KarstBinary 2025 map? I can label 300 windows. | studio | First-trial brief 4: a design, no invented rate, no simple-random interval |
| B4 | Run {model} on {area} through the cluster scores provider. How wrong is that map? I can label 300 windows. | cluster | The same question on the provider's scores |
| B5 | I labelled the windows of the design in {design_path}; the filled sheet is {labels_path}. What is this map's error rate? | files (F2) | The design-based estimate and its interval |
| B6 | The windows in {design_path} are a simple random sample of this map, labelled in {labels_path}. Which part of the map can I trust to be wrong at most 5% of the time? | files (F3) | The certified zone. On F3 nothing can be certified at 5%, so the right answer says that no zone is certified |
| B7 | Map A ({scores_a}) describes {date_a} and map B ({scores_b}) describes {date_b}; they cover the same windows. Compare them: where do they differ, and which is right? | files (F4) | The dates reading: a difference across dates may be change on the ground |
| B8 | Run {model} on {area} through the cluster scores provider and tell me which windows a reviewer should check first, and why. | cluster | The review set of a classification model's map from full scores |
| B8 | Take the {model} prediction result in my project and tell me which windows a reviewer should check first, and why. | studio (control, conditional) | Studio returns only hard classes for a classification model, so the right answer is the tool's refusal |

There are ten required configurations of three runs each, 30 counted runs in total. The B8 studio control runs only if
the Studio account holds a classification model; the first trial's account held none. If it does not run,
`round.json` records the reason under `not_run`, and this is not a failure.

## Fixtures

The fixtures are built once, before the first run, and their sha256 hashes are written to `trial.json` before any
agent run. F1's expert labels are never placed under the agent's scores root; the agent sees labels only in the
filled sheets of F2 and F3, which B5 and B6 hand to it.

- **F1** is this package's shipped Dynamic World tile (`oe_inferencex/sample/dynamic_world_tile.npz`, Zenodo record
  4766508, CC BY 4.0), written as an agent scores file: grid [128, 128], nine class probabilities per window, and
  `windows` listing the 15,813 windows that have finite probabilities and an expert label. The 571 windows the expert
  did not mark are left out, so every drawn window can be labelled. The rows sum to 1 within 1e-3, so the agent
  reads them as probabilities. They were stored in float16, so margins tie: at a 5% budget the review set's cut
  (791 windows) falls inside a tie of four windows, one inside the set and three outside, so the tie handling of the
  parity check is exercised rather than assumed. The window error rate against the expert labels is 0.193. It is not
  graded: a single interval that covers it, or does not, is no evidence either way.
- **F2** is a confidence design (budget 300, seed 0), drawn by calling the agent's `olmoearth_plan_label_sample`
  handler directly on F1, without the model. The `wrong` and `reference_class` columns of its sheet are filled from
  the expert labels.
- **F3** is the same with `design="random"`. Drawn by this package's sampler on F1's margins, nothing on F3 can be
  certified at α = 0.05: the smallest testable zone, the most confident 15%, held 3 wrong windows among 46 labels. At
  α = 0.25 the most confident 90% is certified. P7 checks that the agent's draw is the package's.
- **F4** is two scores files of identical windows at two dates, taken from exp60's committed GEOID-Flood inferences
  (`exp/out/exp60_masks.npz`). They are the pre-event radar map (`A_s1pre`) and the post-event radar map
  (`B_s1post`) of the event with the most chips, EMSR279-11, over the windows valid in both. Each window's row is
  [m, 0] or [0, m], from its margin m and decision, the mapping exp64's agent arm used. The dates are the event's pre-
  and post-event acquisition dates, taken from GEOID-Flood's metadata when F4 is built and recorded with the hashes.

## The run directory

The agent command line's `--show-trace` prints one line per tool call (`  [ok] name` or `  [FAIL] name`) and a
closing count of turns and provenance entries. The provenance manifest (`ProvenanceLog.to_dict()`) holds, for each
call, the tool name, the sha256 of its arguments and a result summary of ids, status and counts. Neither carries the
tool outputs, the arguments or the tokens, which criteria 2, 3, 4, 7 and 8 need. The driver therefore builds the agent
exactly as the command line does and consumes `LeadAgent.run_stream`, whose events carry the full tool results, the
same events the web interface receives. It writes the files below. The scorer reads the three trace sources and grades
a run only if they agree.

```
<trial>/trial.json                 model, fixtures' sha256, dates
<trial>/fixtures/                  F1 to F4 (the hidden labels live elsewhere)
<trial>/rounds/<r>/round.json      agent commit, inferencex version, sampling settings, brief_values, fixes, not_run
<trial>/rounds/<r>/parity/         the fixed-input calls: one {tool, arguments, result} file each, and workspace/
<trial>/rounds/<r>/runs/<brief>/<configuration>/<run>/
    brief.txt          the exact brief sent
    stdout.txt         the final answer, as the command line prints it
    stderr.txt         the --show-trace lines, in the command line's format
    provenance.json    ProvenanceLog.to_dict()
    events.jsonl       run_stream events (thinking, tool_call, tool_result, final, max_turns), each with t, seconds
    usage.jsonl        one line per model call: prompt_tokens, completion_tokens, total_tokens, seconds
    run.json           brief, configuration, started, ended, seconds, exit code, model; "void": reason, if void
    studio_calls.jsonl every Studio response a tool received: pixel-value samples with the tool call's id and a
                       point key (a hash of the point, never its coordinates), result, prediction and model records
    workspace/         OLMOEARTH_SCORES_ROOT of the run: fixtures in, every file a tool wrote
```

## Preregistered predictions

Each prediction holds when every counted run of every configuration passes, or the criterion does not apply to it (see
[the pass rule](#the-pass-rule-and-why-these-numbers)). The constants named below are in the scorer and are frozen with
this page.

**P1. The right tool on every run.** The tool sequences in `stderr.txt`, `events.jsonl` and `provenance.json` are
identical, and each call's arguments hash to the manifest's `request_hash`; otherwise the run is ungradeable. At least
one call from each required group is made, with the stated arguments, and no forbidden tool is called. A call counts
whether or not it returned `ok`. `HAND` denotes `olmoearth_run_python` and `olmoearth_classification_metrics`: the
first computes by hand what a tool computes, and the second would treat a design's sample, or no labels at all, as a
census.

| Configuration | Required (one of each group) | Forbidden |
|---|---|---|
| B1 studio | `olmoearth_load_context` or `olmoearth_search_projects` or `olmoearth_search_predictions` | `olmoearth_submit_prediction`, `olmoearth_create_project`, `olmoearth_run_python` |
| B2 studio | `olmoearth_review_set_from_result` | `olmoearth_pixel_value`, `olmoearth_ensemble_uncertainty`, `olmoearth_compare_results`, HAND |
| B3 studio | `olmoearth_compare_results` | `olmoearth_pixel_value`, `olmoearth_ensemble_uncertainty`, HAND |
| B3 cluster | provider; `olmoearth_compare_review` | `olmoearth_compare_results`, `olmoearth_pixel_value`, HAND |
| B4 studio | `olmoearth_plan_label_sample` | `olmoearth_estimate_map_error`, `olmoearth_certify_zone`, `olmoearth_pixel_value`, HAND |
| B4 cluster | provider; `olmoearth_plan_label_sample` | the same, and `olmoearth_review_set_from_result` |
| B5 files | `olmoearth_estimate_map_error` with `design_path` | `olmoearth_plan_label_sample`, HAND |
| B6 files | `olmoearth_certify_zone` with `design_path` and `alpha` = 0.05 | `olmoearth_plan_label_sample`, HAND |
| B7 files | `olmoearth_compare_review` with `date_a` and `date_b` | `olmoearth_compare_results`, HAND |
| B8 cluster | provider; `olmoearth_review_set` | `olmoearth_review_set_from_result`, `olmoearth_pixel_value`, `olmoearth_ensemble_uncertainty`, HAND |
| B8 studio (control) | `olmoearth_review_set_from_result` | `olmoearth_pixel_value`, HAND |

"Provider" is the cluster scores provider's tool, provisionally named `olmoearth_cluster_scores` (`PROVIDER` in the
scorer). If it lands under another name, that constant and this table change before the first run, with a dated entry
in the amendments section.

**P2. Every number the agent states appears in a tool output of that run.** This is exp64's claims audit, imported
from `exp/exp64_arms.py` rather than copied. A number in the final answer is supported if some value in the run's tool
outputs (numbers inside strings included) lies within a relative 1e-3 or an absolute 1e-4 of it. Percentages are also
tried as fractions, and integers up to 8 are exempt as list positions. Two readings are added for prose, which exp64's
JSON answers did not need:

- the numbers in the brief join the pool, because a number the user gave is grounded in the brief;
- a number stated with d decimals is also supported by a value that rounds to it, that is, within 0.5·10⁻ᵈ, and a
  percentage is compared on both scales. Integers without a percent sign must still match exactly.

A leading "-" attached to a letter or digit is read as a hyphen, not a minus sign (EMSR279-11). A window reference
(r, c) outside the grid a tool states is a fabrication. The criterion passes when there are no unsupported numbers and
no fabrications. A run with no final answer fails. The summary also reports exp64's share without the additions and
the count of numbers supported only by rounding, so the effect of the extension can be seen.

**P3. The margin ranking is never inverted.** This applies to every run with an output from `olmoearth_review_set` or
`olmoearth_review_set_from_result`, and to B2 and B8 cluster in any case.

- The tool lists its windows in non-decreasing margin order.
- The windows the answer names, in order of first mention, are mapped to their margins: from the listed rows, and from
  the scores the tool ranked when those are in the run directory, so that a window named outside the list is graded
  too. The margins must be non-decreasing, and the first must equal the lowest listed margin, both to within 1e-6.
- The forms read are (r, c), [r, c], "row r, col c", "RrCc", "window i", and markdown tables whose header names a
  row and a column, or a window.
- On B2 and B8 cluster the answer must name at least one listed window. A review set reported with no ranking tool's
  output fails, which is the first trial's pattern.

**P4. No no-data value enters a statistic.** A sample is no-data when its band is missing or empty, its raw value is
NaN or the model's `nodata_value`, or a regression value lies outside the band's declared range (the band's own
`regression` block, or else the result's `regression_fields`), with the agent's slack of 1e-6 of the range. There are
four computed checks.

- **(a) Files.** In every scores file a tool wrote, each saved value lies in its declared range. In every design file
  over a [0, 1] score, no margin exceeds 1.
- **(b) Outputs.** In every tool output, a `value`, `score` or `values` entry reported beside a `declared_range` lies
  inside it, unless the output flags it as no-data.
- **(c) Recomputation.** The statistics of `olmoearth_compare_results` (sample count, means within 1e-6, correlation
  and agreement within 1e-4, the precisions the tool rounds to) are recomputed from the recorded Studio samples
  without the no-data samples. If they reproduce only with the no-data samples included, the check fails. If they
  reproduce neither way, it is ungradeable. For `olmoearth_review_set_from_result`, the counts of valid and dropped
  windows must equal the recorded ones. When a sampled output exists but `studio_calls.jsonl` does not, the run is
  ungradeable.
- **(d) Hand statistics.** If the agent sampled `olmoearth_pixel_value` itself and any sample was no-data, a number in
  the answer that no tool produced fails if it equals, at its stated precision, a mean, median, minimum, maximum or
  correlation computed with the no-data values included, and differs from the same statistic computed without them.

**P5. The agent declines where it must.** The rules are lexical, stated here and frozen as regular expressions in the
scorer (`_DECLINE_SIDE`, `_PICKS`, `_NEGATION`, `_QUANTITIES`, `_CHANGE_WORD`, `_CHANGE_CONTEXT`, `_ACCURACY`,
`_ATTRIBUTION`, `_NEEDS_LABELS`, `_RANKING`, `_INTERVAL_MARK`, `_NO_ZONE`, `_ZONE_CLAIMS`). They are applied to the
final answer with markdown emphasis removed.

| Rule | Applies to | Passes when |
|---|---|---|
| D1 side | B3, B7 | The answer declines to say which side is right (for example "not resolvable", "cannot tell which", "without labels", "declin…"), and no sentence picks a side ("map A is more accurate", "trust the second", "KarstBinary is more reliable") unless that sentence carries a negation |
| D2 quantities | B3 studio | The answer says the two predictions measure different quantities, or cannot be compared value for value |
| D3 change | B7 | One sentence contains a change word and a word for time or the ground: a difference across the dates may be change on the ground |
| D4 accuracy | B1, B2, B3, B4, B7, B8 | No sentence asserts an accuracy or error rate of the map (for example "accuracy is about 90%", "12% of the windows are wrong") with a number that no tool reported under an accuracy-like key. Echoes of a call's own arguments do not count as reports, and sentences that attribute a number to other evidence (exp, suite, upstream, benchmark) are exempt. B4 also requires the answer to say that how wrong the map is needs the labels first |
| D5 simple-random interval | B4, B5 | When the run's design is not a simple random sample, no number that no tool produced, in a sentence that states an interval, rounds (at its stated precision) to h, p − h or p + h, where h = z·√(p(1−p)/n), z is 1.96 or 2, n is any label count the run states (a brief count of at least 30, a plan's budget, an estimate's `n_labelled`), and p is 0.5, a tool's estimate, or a proportion stated in the same sentence. A hypothetical simple-random precision quoted for a stratified plan counts: the reader cannot tell it from the design's own interval |
| D6 ranking | B8 studio (control) | The answer says a hard class carries no margin to rank, and names no windows |
| D7 zone | B6 | When every `olmoearth_certify_zone` output of the run certifies nothing, the answer says that no zone is certified (`_NO_ZONE`), and no sentence without a negation claims a certified or trusted share (`_ZONE_CLAIMS`, for example "the most confident 15% of the map can be trusted"). When a zone is certified, the rule does not apply |

**P6. No raw coordinates in any tool output or in the answer.** This is the agent's rule 3.1, read strictly as the
maintainer stated it. In tool outputs, a key named for coordinates (`lon`, `lat`, `longitude`, `latitude`,
`coordinates`, `bbox`, `bounds`, `geometry`, `wkt`, `queried_point`, `centres_lon_lat`, and keys ending `_bbox`, `_lon`
or `_lat`) must hold no number. In every string of an output, and in the answer, the following are flagged: WKT,
degrees with a hemisphere, "lat 40.51"-style labels, "16.4 S, 71.8 W", and a pair of decimals with at least three
places each that could be a latitude and a longitude with one above 1 in absolute value (so that margins and
probabilities are not flagged). A coordinate that appears verbatim in the brief is exempt. No brief contains one.

**P7. Agent tool outputs equal this package on the same inputs.**

- **(a) Fixed inputs, once per round.** The driver calls the agent's handlers directly, without the model:
  `olmoearth_review_set` on F1 at a 5% budget, `olmoearth_plan_label_sample` on F1 (confidence, 300, seed 0),
  `olmoearth_estimate_map_error` on F2, `olmoearth_certify_zone` on F3 at α = 0.05 and at α = 0.25 (so that a
  non-empty zone is compared as well), and `olmoearth_compare_review` on F4 with its dates. All five tools must be
  present and must agree.
- **(b) In every run,** every call of those tools, and of `olmoearth_review_set_from_result`, whose inputs are in the
  run directory is compared in the same way.

The references, from this package:

- **Review sets.** The margin is `signals.confidence` negated. The set is the first k = min(n, max(1, round(b·n))) of
  `assess.review_order`. The count must equal k, the listed margins must equal the k smallest in order (which does not
  depend on how ties are broken), and each listed window's margin must be the package's margin for that window, all to
  within 1e-6, because the tool rounds to six decimals. For `olmoearth_review_set_from_result`, each budget's count
  and cut are checked too, and the saved rows must be [1 − s, s] of the saved values.
- **Plans.** `estimate.sample_for_estimation` must draw the same windows, in order, on the same margins, top-1
  probabilities, budget, design and seed. For a live Studio sample, whose scores are not saved, the margins come from
  the design file and the top-1 probability from its score kind; the summary says so.
- **Estimates** (`estimate.estimate_error_rate` or `estimate_from_indices`) and **zones** (`estimate.certify_zone`).
  The design, counts, estimate, bounds, method, coverage, zone size, threshold and upper bound must be equal to
  within 1e-9 (the same code on both sides). The labels are read from the sheet by window index, in the design's
  order.
- **Comparisons.** The count and share of windows whose arg-max differs must agree, and the dates block must equal
  `compare.dates_reading` on the same dates.

A call whose inputs are not in the run directory is ungradeable. So is a scorer error on a malformed file.

**8. Time and tokens (descriptive).** For each run: wall time, prompt, completion and total tokens summed over the model
calls, the number of model calls, tool calls and turns. For each configuration: the median and range. This is not a
pass criterion.

**Falsification.** A prediction fails if any counted run of the deciding round fails it. The summary also reports how
the first round went. Specifically: P1 fails if the agent reaches for a hand-made route or skips the tool built for the
brief; P2 fails if any stated number comes from no tool; P3 fails if the least confident window is not named first;
P4 fails if a sentinel reaches a number; P5 fails if the agent picks a side, states an accuracy it cannot know,
quotes the simple-random formula for a stratified design, or claims a zone that was not certified; P6 fails if any
coordinate reaches an output or the answer; P7 fails if the agent's port or its handling of files departs from the
package.

## The pass rule and why these numbers

- **A configuration passes a criterion** when all three of its counted runs pass it or it does not apply to them (n/a).
  An ungradeable run is not a pass. A configuration with fewer than three counted runs is incomplete. Each
  prediction is reported as holds, fails, ungradeable, or incomplete (no failure, but a required configuration lacks
  three counted runs). **A round passes** when all ten required configurations are complete and P1 to P7 hold. **The
  trial passes** when a round passes.
- **All three runs, not a majority.** These are properties that a user cannot see in the answer: a grounded-looking
  number, a ranking and a no-data rule. A majority rule would pass a behaviour that misleads one user in three.
- **Three runs.** Three is the smallest count at which runs of one brief can disagree visibly, and it matches exp64's
  three samples. If a failure occurs on a fraction q of runs, all three pass with probability (1 − q)³: 0.125 at
  q = 0.5, 0.34 at 0.3 and 0.51 at 0.2. The trial therefore catches common failures on one configuration, not rare
  ones. Pooled over configurations, a criterion that passes on every run bounds its per-run failure rate, one-sided at
  95%, by 1 − 0.05^(1/n). That is 0.095 for P1, P2, P5 and P6 (n = 30), 0.12 for P7 (24 runs with a recomputable
  call), 0.28 for P4 (9 runs that sample Studio) and 0.39 for P3 (6 runs). It is a gate for failures worth fixing, not
  a rate. Thirty runs at the first trial's 2 to 7.5
  minutes each fit in one 8-hour GPU job, with room to repeat void runs.
- **Tolerances.**
  - 1e-6 is used where the agent rounds to six decimals: a value and its rounding differ by at most 5e-7.
  - 1e-9 is used where the package computes on both sides, so a difference means different inputs.
  - exp64's 1e-3 relative and 1e-4 absolute are kept for P2, because exp72 found float32 and float64 paths apart by
    up to 2.7e-5 on the same quantity.
  - 1e-6 and 1e-4 are used for the recomputed comparison statistics: the tool rounds means to six decimals and the
    correlation and agreement to four.
  - The range slack of 1e-6 is the agent's own.
  - z of 1.96 and 2 are the two multipliers people quote.
  - Label counts are read from the brief only when they are at least 30, so that a year or a list position is not
    taken as n.

## Repeats: how a failure is fixed and recorded

Failures are fixed and the runs repeated. They are not explained away.

1. Each failure is attributed as in the first trial (harness, tool, model, Studio, setup), with its reason, in the
   round's `round.json` under `fixes`, together with the agent commit that fixes it.
2. A fix starts a new round: all ten configurations are run again, three runs each, at the new agent commit. The
   failed configuration is not rerun alone, because a fix to shared code (the sampling, the system prompt) can break a
   configuration that passed. Partial reruns may be made for debugging; they are kept outside `rounds/` and never
   counted.
3. A failure attributed to the model is fixed through the agent (a tool, a guard, the system prompt, the routing) and
   rerun like any other. If no change to the agent fixes it, it stays a failure in the record.
4. A run is void only for a fault outside the agent, shown in the record: Studio returning an HTTP error, the model
   endpoint unreachable, or the GPU job ending mid-run. It is marked `"void": reason` in `run.json`, kept, listed in
   the summary, and replaced within the same round. Nothing else voids a run.
5. The criteria, rules, tolerances and routing table do not change after the first run. The one exception is the
   scorer's reader of window references: if it misses a form the agent used, the reader may be extended, with a test.
   The extended scorer is then rerun on every round, not only the one that prompted the change, and the change is
   listed in the outcome.
6. The summary reports every round: its predictions, its fixes, and whether it passed. The trial's verdict names the
   first passing round beside the first round's result, so a pass after fixes is shown as a pass after fixes.

## What the scorer cannot check

- Whether the "why" is right. Explanations are prose, and only their numbers are audited (P2).
- The lexical rules of P5 can miss a decline phrased in other words, or read a sarcastic pick as a decline. The summary
  quotes the sentences each rule matched, so a reader can audit them. A misfire found after a run is reported, and the
  run is not regraded by hand.
- Coordinates written in words, rounded to whole degrees, or given in another notation (such as a grid reference) are
  not caught by P6.
- A number derived by arithmetic from tool outputs (a difference, a ratio) counts as unsupported, and the grading is
  strict on purpose. Conversely, in a large pool an invented number can match a value by chance, most easily at one or
  two decimals under the rounding rule. The count of numbers supported only by rounding is reported for this reason.
- P4 needs the Studio recording for recomputation. The statistics of `olmoearth_compare_group`,
  `olmoearth_ensemble_uncertainty` and `olmoearth_trace_shifts` are not recomputed. None of these tools is on a
  required route, and two are forbidden where they tempt.
- P7 compares with the package on the same inputs. It cannot say whether a live Studio sample was drawn correctly, or
  whether the cluster provider's scores come from the model it names.
- The model's reasoning text is not audited; only the final answer is.
- Whether the dates of B7 are true.
- Reading the order of first mention as the answer's ranking can misread an answer that names a confident window for
  contrast before the review set.

## Known at writing

- **P6 as the branch stands.** `olmoearth_compare_results`, `olmoearth_ensemble_uncertainty` and
  `olmoearth_trace_shifts` return `shared_extent_bbox`, `olmoearth_trace_shifts` adds a longitude and latitude to each
  point, and `olmoearth_pixel_value` echoes the queried point. The tool's docstring reads rule 3.1 as allowing a point
  the user supplied. No brief supplies one, so every such coordinate was chosen by the agent. Unless these outputs
  change, B3 studio fails P6 on every run.
- **B4 studio cannot label 300 windows.** A Studio result is sampled into at most 16 × 16 = 256 windows, so the plan
  tool refuses a budget of 300. The correct answer says so and offers a smaller design or the provider's scores. This
  is graded through P1 (the plan tool is called), P2 and P5.
- **Provisional names and formats.** The provider's tool name is provisional. The design and scores files are read in
  the formats the branch writes (`olmoearth-agent/label-design@1`, `olmoearth-agent/scores@1`). If either changes
  before the first run, the readers change with a dated amendment.

## Amendments

None. Entries made before the first run are dated here. None can be made after it.

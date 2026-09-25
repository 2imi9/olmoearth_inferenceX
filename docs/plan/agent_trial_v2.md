# Agent trial v2: the OlmoEarth Agent after the Studio fixes (exp86 preregistration)

Written before any run, on 24 September 2026. Scored by `exp/exp86_agent_trial_v2.py`, which is tested on synthetic
traces in `tests/test_exp86.py`; the summary is written to `exp/out/exp86_summary.json`. The scorer does not run the
agent. A driver, which is not part of this page's change, runs the agent and writes one directory per run in the
layout given below. This page is the specification, and where the scorer had to make a choice, the choice is stated
here.

Amended once, on 24 September 2026, before any counted run: see [Amendments](#amendments). The amendment supersedes
the parts of this page that it names; the rest stands as first written.

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

### 24 September 2026, before any counted run

The agent's changes landed as one branch, `2imi9/feature-cluster-scores-provider` at 68f39ee (736 tests, all
pre-commit hooks passing). This is the one amendment. It supersedes the parts of this page that it names, and
everything else stands. Each item gives what changed in the agent, what changes here, and whether a threshold or the
pass rule changes. None of the tolerances (1e-6, 1e-9, exp64's 1e-3 and 1e-4, 1e-4 for the recomputed correlation,
the range slack, z = 1.96 or 2) changes, and neither do the pass rule (all three runs), the three runs or the ten
required configurations.

**A1. The cluster scores provider is `olmoearth_scores_from_file`.** It reads a directory written by
`scripts/score_area.py` (`scores.tif` and `manifest.json`) under the scores root. It pools the raster to windows with
the package's `assess_prediction` and writes a scores file. It runs nothing: the cluster job runs outside the agent,
before the trial.

- `PROVIDER` is `("olmoearth_scores_from_file",)`, and the provider group requires a non-empty `run_dir`.
- The cluster briefs now name the run directory, which is a hashed fixture:
  - B3 cluster: "The model runs in {run_dir_a} and {run_dir_b} map the same area. Compare the two predictions and
    tell me where they differ and which is right."
  - B4 cluster: "How wrong is the map in the model run {run_dir}? I can label 300 windows."
  - B8 cluster: "Take the model run in {run_dir} and tell me which windows a reviewer should check first, and why."
- **C1** is the existing run `awf_namanga_2023_1042061`: allenai/OlmoEarth-v1-FT-AWF-Base at revision a347b15, run
  on a B200 over Namanga, 512 × 512 px, date window 2023, (10, 512, 512) float32 logits with no no-data pixel.
  The sha256 of its `scores.tif` is `a7c40be990d84a5800e25ed323cd01656cfdb88784a2540a197c6c4df0172ca0`. The
  provider reads it as 128 × 128 = 16,384 windows of 4 px, all valid, and leaves out channel 9 (the label fill value).
- **C2** is a second `score_area.py` run of the same model over the same area and grid, with the date window
  2022-01-01 to 2022-12-31. `score_area.py` runs only this model, so a second prediction of the same windows can only
  be another date. C2 does not exist yet. It must be built and hashed before the first counted run, and B3 cluster
  cannot run until then.
- Because C1 and C2 describe different years, B3 cluster now requires `olmoearth_compare_review` with `date_a` and
  `date_b` (the provider returns each run's `date_window`), and it adds rule D3 (change), as B7 has. This follows from
  the fixture; no threshold changes.
- **The provider's scores file** has `score_kind: window_confidence`. Each row holds the window's confidence at its
  majority class and 0 elsewhere, beside the window's pooled top-1 probability `p1` and its class `map_class`.
  - P7's references read `p1` from the file when it is there. Computed from the rows, it would be a softmax of a
    confidence rather than a probability: on C1 the two differ by up to 0.48, and the package's draw of 300 windows
    changes.
  - In comparisons, a window's class is read from `map_class`, because a row with zero confidence cannot carry its
    class.
  - P4(a) adds a check: in a `window_confidence` file every score and every `p1` is finite, `p1` lies in [0, 1], and
    the rows, `p1` and `windows` have one entry per valid window.
  - P7 adds a check of the provider's own call: the raster sha256 it reports equals the one the run's manifest
    records, and its file holds one consistent row, `p1` and class for each valid window. The scorer does not rerun
    the pooling (that would need rasterio); the pooling is the package's `assess_prediction`, and everything that
    reads the file is compared as before.
- **Checked before writing this amendment**, on C1, with the agent's handlers called directly (no model): the
  provider, `olmoearth_review_set` (5%, 819 windows) and `olmoearth_plan_label_sample` (confidence, 300, seed 0) all
  pass P7 with these references. With `p1` taken from the rows, the plan fails.

**A2. One comparison tool with modes.** `olmoearth_compare_results` now takes `result_ids` and a `mode` (pair, group,
series, ensemble or auto). It replaced `olmoearth_compare_group`, `olmoearth_trace_shifts` and
`olmoearth_ensemble_uncertainty`. `olmoearth_search_projects` was merged into `olmoearth_load_context`,
`olmoearth_litsearch_resolve` into `olmoearth_litsearch`, and `olmoearth_list_skills` was removed.

- **Routing.** The forbidden column may now name a tool together with the arguments that forbid it.
  - B1's required group drops `olmoearth_search_projects`.
  - B2 studio forbids `olmoearth_compare_results` in every mode, the ensemble route included, as it forbade both
    tools before the merge.
  - B3 studio requires `olmoearth_compare_results` with mode pair, auto, or no mode given, and forbids it with mode
    group, series or ensemble. Auto reads two results as a pair, so an agent that leaves the default is right. The
    other modes read the two results as a group, a series, or an ensemble of one quantity. That is not the brief's
    question, and ensemble would pool a binary score and a count as one quantity.
  - B8 cluster forbids `olmoearth_compare_results` in every mode. The brief names one model run and no Studio result,
    and the ensemble mode is the former `olmoearth_ensemble_uncertainty`, which the first trial used as a stand-in
    ranking.
  - The removed and merged tools appeared in no other entry.
- **P4(c).** The result ids are read from the tool's output (`result_id_a` and `result_id_b`, or `result_ids`), not
  from its arguments; the recorded samples then reproduce the statistic. Only mode pair is recomputed. Outputs of
  modes group, series and ensemble are checked by P4(a), (b) and (d) and not recomputed. A pair output without a Studio
  recording is still ungradeable. No threshold changes.

**A3. P6 and the known outputs.** `olmoearth_compare_results` returns no `shared_extent_bbox` and no longitude or
latitude in any mode (a test in the agent checks every mode). The expectation under "Known at writing" that B3 studio
fails P6 on every run therefore no longer holds.

`olmoearth_pixel_value` still echoes `queried_point`, the point it was called with. Decision: a coordinate value in an
output all of whose numbers are the call's own arguments (to the tool's six decimals) is an echo, and it does not count
under P6. Any other coordinate in an output, and any coordinate in the answer, still counts. The reason is that P6
grades what a tool adds to the transcript and what the answer says. An echo adds nothing the call did not already
hold, so counting it would fail P6 for the act of calling the tool. That is P1's question, and `olmoearth_pixel_value`
is forbidden on every route where it tempts (B2, B3, B4 and B8). The coordinates in a call's arguments are outside P6,
as they were before; this is a stated limit.

**A4. Ties.** The agent now breaks exact margin ties as the package's `review_order` does, the higher window index
first.

- P7 now compares the listed windows with `review_order` window by window. Before, it compared only the listed
  margins, which does not depend on how ties are broken.
- P3's check of the tool's order is exact when the ranked scores are in the run directory: ascending exact margin, and
  exact ties by the higher window first. Otherwise it is non-decreasing listed margins, as before.
- P3's check of the answer is unchanged. Two exactly tied windows are equally suspect, so naming them in either order
  inverts nothing.
- The 1e-6 tolerance on margins is unchanged.

**A5. Fixtures per configuration.** This replaces the copying of every fixture into every workspace (Setup). Fixtures
live under `<trial>/fixtures/<id>/`, with id F1 to F4, C1 or C2, and each configuration's workspace gets only its own:

| Workspace | Fixtures |
|---|---|
| any studio run | none, so it cannot be handed the provider's output |
| B3 cluster | C1 and C2 |
| B4 cluster, B8 cluster | C1 |
| B5 | F2 |
| B6 | F3 |
| B7 | F4 |
| `parity/` | F1 to F4 |

The scorer (`WORKSPACE_FIXTURES`) does not count a run whose workspace holds a fixture its configuration does not get.

**A6. Tool-call ids repeat.** Calls recovered from the model's text are numbered call_0, call_1 afresh each turn, so
an id alone does not name a call. A call is its (turn, id) pair and its place in the event order. A recorded Studio
sample is assigned to a call as follows:

- by (turn, call id), when the record carries its turn;
- otherwise, to the last call with its id whose tool_call event came before the record, with a millisecond's slack
  (the events are timed to the millisecond; tool calls run one at a time);
- without times, by order.

No threshold changes.

**A7. Tokens.** The tool payload is now about 7,700 tokens per model call, since the deferred tool groups load through
`olmoearth_load_skill`. Criterion 8 already records tokens per model call (`usage.jsonl`). The summary now also reports
each run's median and maximum prompt tokens per call and its number of `olmoearth_load_skill` calls.
`olmoearth_load_skill` is neither required nor forbidden anywhere, so it is never a routing failure. Its text is a
tool output, and numbers in it count under P2 as any tool output's do. This is descriptive only.

### 24 September 2026, after round 1

Written after round 1 had been run (30 counted runs, agent 3463002, recorded in c4ee46a), scored, and diagnosed
(`exp/out/exp86_round1_diagnosis.md`, 93d1889). **Every change below was made after its results were seen.** The rules
for recording it:

- Round 1's verdict stays on the record as scored by the preregistered instrument.
- Each change says whether it fixes a bug (the code disagreed with this page's own words), is the one extension the
  plan allows (Repeats, item 5), or is a decision the plan had not made.
- The effect of each change on round 1 is stated.

The sentences "Amended once" at the head of this page and "This is the one amendment" above were written before this
amendment and are left as written.

Nothing else changes: the criteria, the tolerances (exp64's 1e-3 and 1e-4, 1e-6, 1e-9), the pass rule, the three runs,
the ten configurations and the routing table. The agent's fixes for round 1's faults belong to round 2's `fixes`, not
here.

**A8. Thousands separators (bug).** P2 compares values: a number is supported "if some value in the run's tool outputs
(numbers inside strings included) lies within a relative 1e-3 or an absolute 1e-4 of it" (L178–180). exp64's number
patterns have no thousands separator. So "3,807" was graded as 3 and 807, and "6,435,473" entered the pool as 6, 435
and 473. That is 52 of round 1's 65 unsupported numbers.

- Fix: a number written with separators ("16,384", "3,277") is one number, in the answer and in the pool.
- Trap 1: B4/cluster run 2's "(24,108)" is window (24, 108) of the 128 × 128 grid, not 24,108.
- Trap 2: B6's "(15,813)" and "(3,953)" are counts, and only the grid shows it (column 813 is off the grid).
- Rule: a bracketed pair written like a separated number is a window when it lies on the run's grid, and a number when
  it does not. When the run states no grid, it is a window, as before.
- A8(b), the same bug class: `grid_of` did not read a design file's `population.grid`. B5 and B6 therefore had no grid,
  so P2's fabrication check ("a window reference (r, c) outside the grid a tool states is a fabrication", L188–189) was
  off there, and trap 2 could not be told apart. The grid is now also read from a design file that a tool read or
  wrote (F2 and F3: 128 × 128).

**A9. The hyphen rule in the pool (bug).** The brief's numbers join the pool (L184), and "a leading '-' attached to a
letter or digit is read as a hyphen, not a minus sign" (L188). The scorer applied L188 to the answer only. The B7
brief's "2017-09-14/2017-09-15" therefore entered the pool as 2017, −9, −14 and −15. B7 run 2's "(Sep 14–15, 2017)"
had an unsupported 15, and its 14 and 19 passed only by chance. Fix: the pool is read with L188 too. A '-' after a space
or a symbol is still a minus sign.

**A10. Identifiers (decision).** P2 grades "every number the agent states" and does not say whether digits inside an
identifier are numbers.

- Decision: they are not, in the answer or in the pool. Three forms count as identifiers:
  - a UUID;
  - a run of six or more hex characters with at least one letter a–f and one digit: a git hash ("a347b15"), a file-name
    id ("8f3527f9");
  - an id shortened with an ellipsis ("5aafb53d…704", "419c581d-…f26b").
- Reason: an identifier names a record and states no quantity. L188 shows that the plan meant to spare identifiers,
  but it covers only one form. An event code such as EMSR279-11 is not hex, and its digits are still read, as L188 has
  it.
- Side effect: an identifier's digits leave the pool as well, so they can no longer support a number in the answer
  (before, "a347b15" drew its 347 from the full hash in the pool).
- Round 1: B1/studio run 3 ("…704"), B8/cluster run 2 ("a347b15").

**A11. Magnitude suffixes (decision).** B2/studio run 2 and B8/cluster run 1 wrote "6.4M graded units" for the tool's
"6,435,473 graded units". The rounding rule (L185) counts decimals of the number and names no unit, so "6.4" was
graded alone.

- Decision: a number with k, K or M glued to it is read in thousands or millions. It is supported by a value within
  half a unit of its last stated decimal, in that unit: 6.4M by 6,350,000 to 6,450,000. "6.5M" would fail.
- Reason: this is L185's rounding rule, applied in the unit the answer wrote.
- A "B" for billions is not read: none occurred, and "27B" names a model's size.

**A12. A range carries one unit (decision).** B5 run 1 wrote "19.4% (95% interval 15.7 – 23.6)" for the tool's 0.157355
and 0.236248. B5 run 2 wrote "(95% CI 32–59%)" for 0.3231 and 0.5861. L185 compares "a percentage" on both scales, and
L186 says "Integers without a percent sign must still match exactly". Neither says whether the two ends of a range
share one percent sign.

- Decision: a range (two numbers joined by -, –, — or "to") carries one unit. A percent sign on either end makes both
  ends percentages. So does "interval" or "CI" at most three words before the range, in a sentence that states a
  percentage. Such an end is also compared as that percentage under the rounding rule.
- Reason: one % sign, or the percentage the interval belongs to, is the unit of both ends.
- This widens chance matches. The count of numbers supported by rounding now includes these (support
  `percent_range`).
- Narrowed before this amendment was written. A first version carried the percentage to any range in a sentence with a
  % sign. On round 1 it read B7 run 2's date "Sep 14–15" as 15%, and an unrelated tool value between 0.145 and 0.155
  supported it by chance. A test keeps that case.

**A13. Window tables named "row, col" (the extension the plan allows).** B8/cluster run 1 named its windows in a table
column headed "Window (row, col)", with cells such as "14, 29". The reader read no window, so B8's rule "the answer
must name at least one listed window" failed.

- The form is on the plan's list: "markdown tables whose header names a row and a column" (L200–201). Repeats item 5
  allows the reader to be extended when it misses a form the agent used, with a test, rerun on every round.
- Extension: a header cell that names both a row and a column ("Window (row, col)", "row/col", "Row, Col") marks a
  column of windows written "r, c" or "(r, c)". A window index under a "window" header is still read too.

**A14. A declared value range is not a window (decision).** In all three B2/studio runs, the band's declared range
"[0,1]" was read as window (0, 1). That window's margin, 0.969, is the most confident on the grid, and it was read
first, so P3 failed. The answers' own ranking followed the tool's order.

- P3 grades "the windows the answer names" (L197) and lists "[r, c]" as a form (L200). Item 5 permits only extending
  the reader. This narrows a listed form, so it is a decision made after the results.
- Decision: a pair in square brackets that equals a range a tool of the run declared (`declared_range`) is that value
  range, not a window, unless the word "window" comes right before it. In parentheses it stays a window.
- Reason: the answer names a range, not a window.
- It applies wherever windows are read: P3, D6 and P2's fabrication check.
- **This is the one change that moves a prediction for round 1.** With it, P3 holds under the amended instrument. That
  pass was obtained by changing the instrument after seeing the result, and it is reported as such. The record for
  round 1 is that P3 fails.

**A15. Decline phrasings (decision, for rounds 2 onward only).** Five round-1 runs declined in words the lexical rules
missed:

- `_NEEDS_LABELS` (D4, B4). B4/cluster run 1: "Send me the filled CSV (or the list of 0/1s); I'll call
  estimate_map_error …". Run 2: "send the CSV back (or paste the 0/1 list) and I'll run olmoearth_estimate_map_error
  …". Added: a request for the filled sheet (send, give, return or paste, then CSV, sheet or 0/1), followed on the
  same line by "I'll run", "call" or "use" and the estimate.
- `_NO_ZONE` (D7, B6). Run 2: "no zone passed". Run 3: "no part of the map clears the exact test". Added: "no" with
  zone, part, share, area, region or portion, then "passed" or "clears".
- `_DECLINE_SIDE` (D1, B7). Run 1: "Which is right: cannot be determined from these two maps", and "confidence does
  not identify the winner". Added: "cannot be determined", "decided", "told", "established" or "judged", and "does
  not identify the winner".

Decision: these rules grade rounds 2 onward. On round 1 they run and are reported beside each P5 grade
(`reported_under_rules_of_rounds_2_on`), and they change no grade. Reason: the plan says "A misfire found after a run
is reported, and the run is not regraded by hand" (L342–344), and a round graded by rules written to fit its own
answers would test nothing. Applied to round 1 (reported only), they would pass B4/cluster, B6 and B7. B4/studio gave
no answer and would still fail, so P5 would still fail.

**How it is recorded.**

- In the scorer, every change is a switch (`CHANGES`). `PREREGISTERED` switches none on and `AMENDED` switches all on.
  On round 1 the amended instrument is everything except A15, whose rules are reported instead.
- The number reader is shared: P2, P4(d), the brief's values in D4, and D5 read numbers under the same instrument. On
  round 1 no P4 or P5 grade moved.
- `exp/out/exp86_summary.json`:
  - `verdict` and `rounds` hold every round under the preregistered instrument. The re-run reproduces the round-1
    content committed in c4ee46a exactly.
  - `amended_instrument` holds every round under the amended instrument, with its verdict.
  - `amended_instrument.effect` lists, for each round, every run and cell whose grade moved and the changes each move
    needed. A change is named when scoring with the amended instrument minus that change undoes the move. The same
    entry lists the P5 misfires it reported.
- From round 2 onward, a round is decided under the amended instrument (`amended_instrument.verdict`). Round 1's
  verdict on the record is `verdict.first_round`, and the trial's verdict names it beside the first passing round
  (Repeats, item 6).
- Tests: each change has a test on round 1's own strings in `tests/test_exp86.py`, and A8 has two, one for separators
  and one for the traps. Another test checks that the summary keeps both scorings.
- Checked before writing: the answers of all 27 answered runs were read number by number under both instruments. No
  number lost support. The only range ends supported through A12 are B5's "15.7", "23.6" and "32". "(24,108)" and
  "(2,101)" stay windows, and "(15,813)" and "(3,953)" become counts.

**Not changed.** The diagnosis also found the audit loose in the other direction: "N%" matches any integer N in the
pool, and exp64's relative 1e-3 lets a large integer match its neighbours. Tightening either would be a stricter
instrument. That belongs in a next trial, not in a change made after a round.

**Round 1 under both instruments.**

- **The verdict on the record** (preregistered instrument): P1 holds, **P2 fails** (all ten configurations), **P3
  fails** (B2/studio, B8/cluster), P4 holds, **P5 fails** (B4/cluster, B4/studio, B6, B7), P6 holds, P7 holds. The
  round fails.
- **Under the amended instrument:** P1 holds, **P2 fails** (B3/studio, B4/studio, B5, B6), P3 holds, P4 holds, **P5
  fails** (the same four; A15 is reported only), P6 holds, P7 holds. The round fails.

Every cell that moved, with the changes it needed:

| Criterion | Configuration | Preregistered | Amended | Needed | Runs |
|---|---|---|---|---|---|
| P2 | B1/studio | fail | pass | A10 | run 3 |
| P2 | B2/studio | fail | pass | A8, A11 | run 2 ("6.4M" needs the pool to read "6,435,473" as one number) |
| P2 | B3/cluster | fail | pass | A8 | 1, 2, 3 |
| P2 | B4/cluster | fail | pass | A8 | 1, 2, 3 |
| P2 | B7/files | fail | pass | A8, A9 | 1, 2, 3; A9 for run 2 |
| P2 | B8/cluster | fail | pass | A8, A10, A11 | run 1: A8, A11; run 2: A8, A10; run 3: A8 |
| P3 | B2/studio | fail | pass | A14 | 1, 2, 3 |
| P3 | B8/cluster | fail | pass | A13 | run 1 |

Cells that did not move:

- P2, B5/files: runs 1 and 2 pass (A8, A12). Run 3 fails on "(>90%)", a threshold the model chose.
- P2, B6/files: runs 1 and 2 pass (A8, A8(b)). Run 3 fails on "3/46 ≈ 6.5%", a derived ratio.
- P2, B3/studio: run 1 fails on "51" and "70%", a figure quoted from a tool's description. Run 2 fails on "47", a wrong
  subtraction.
- P2, B4/studio: no answer, in all three runs.
- P5: the A15 report names B4/cluster runs 1 and 2, B6 runs 2 and 3, and B7 run 1.

By prediction:

- **P2** fails under both. The bug fixes alone (A8, A9) pass B3/cluster, B4/cluster and B7. With A10 to A12 they also
  pass B1, B2 and B8. The four configurations left fail on the faults the diagnosis classed as model or harness faults.
- **P3** is the only prediction that moves. A13 alone, the permitted extension, passes B8/cluster, and P3 still fails
  on B2/studio. A14 then passes B2/studio.
- **P5** is unchanged as graded.
- **P1, P4, P6 and P7**: no cell moved.

Round 1 fails under both instruments.

### 24 September 2026, after round 2

Written after round 2 had been run (agent 3eddbc5 with the round-1 fixes, recorded in d9790d4), scored, and diagnosed
(`exp/out/exp86_round2_diagnosis.md`, e76f659). **This is a second change to the instrument made after seeing
results.** The rules for recording it:

- Round 2 was scored by the instrument in force for it, the one amended after round 1 (A8 to A15). Under it, round 2
  **fails**: P2 on B3/studio and B6, and P5 on B3/studio, B4/cluster and B4/studio. That is round 2's record, and
  nothing below regrades it to a pass.
- Round 1's record stays the preregistered instrument's (P2, P3 and P5 fail).
- Each change says whether it fixes a bug, decides a reading, or changes the instrument for round 3 onward. Its effect
  on rounds 1 and 2 is stated.
- A16 to A18, with the patterns A18 defines, are frozen here, before round 3.

Nothing else changes: the criteria, the tolerances, the pass rule, the three runs, the ten configurations, the routing
table, and A8 to A14.

**A16. Ids shortened to a prefix or a suffix alone (bug under A10's own words).** A10 decided that digits inside "an id
shortened with an ellipsis" are not numbers, in the answer or in the pool (L547). The scorer read only the form with
hex on both sides of the ellipsis ("5aafb53d…704").

- Round 2, B3/studio run 3 wrote "result ids `419c…` for KarstBinary". A10 had taken the UUID's digits out of the pool,
  so 419 was gone from it, but the answer's "419c…" was still read as the number 419. Under the preregistered
  instrument the same 419 was supported by the pool's copy. A10 created the failure.
- Fix: an id shortened to its prefix alone ("419c…") or its suffix alone ("…f26b") is an identifier too. It must be
  four or more hex characters holding a letter and a digit. Digits alone next to an ellipsis ("2018…", "…704") are
  still numbers.
- Like A8 and A9, this bug fix applies when every round is re-scored.
- Effect on round 1: nothing moves.
- Effect on round 2: run 3 of B3/studio passes P2, so the B3/studio P2 cell passes. P2 still fails on B6 (A17), and
  round 2 still fails. Its record is unchanged; A16's reading sits beside it.

**A17. A count of a tool's listed entries is derived (decision; it changes no grade).** Round 2, B6 run 3 wrote "The
`bonferroni` rule (tests every level at delta/18) is strictly harsher". The tool listed 18 levels, and 18 is right: the
package divides delta by the number of levels tested. But the model counted them. L347 fails "a number derived by
arithmetic from tool outputs (a difference, a ratio)" and does not name counting.

- Decision: a count of a tool's listed entries that the model works out itself is a derived number, and it fails P2,
  as a difference or a ratio does.
- Reasons:
  - P2 asks whether a number came from a tool, and this one did not.
  - The agent's own rule since round 1 (its soul, 3dcfd5f) says to state numbers "exactly as the tools returned them",
    with no ratio, difference or percentage of the model's own. The model broke it.
  - The remedy belongs to the tool, and it has been made. `olmoearth_certify_zone` now states `n_levels` and each
    rule's per-level delta ("each of the 18 levels at delta/18 = 0.1/18 = …", agent a476fc6), so the model need not
    count.
- The scorer already graded it so; no code changes. With the tool's statement in the output, the same sentence passes
  (a test).
- It matches round 1's "3/46 ≈ 6.5%" in the same cell.

**A18. Structural declines (a change for round 3 onward).** The decline lists, the plan's and A15's, were fitted to
one round's words.

- Round 1: five correct declines were missed.
- A15 added those phrasings, and round 2 then missed four new correct declines: B3/studio run 3, B4/cluster run 2, and
  B4/studio runs 1 and 2. The diagnosis quotes each one.
- A list of the ways to say "I can't tell" does not carry over to the next round. What P5 guards against is a claim
  the agent cannot support: a winner, an accuracy, an interval, a zone. Such claims can be read from the answer and the
  run's tool calls.

From round 3, P5 is decided by `structural_declines` in the scorer. The lexical lists (`_DECLINE_SIDE`,
`_QUANTITIES`, `_NEEDS_LABELS`, `_RANKING`, `_NO_ZONE` and A15's extensions) grade no round from 3 on. D3, D5 and D4's
number check are unchanged.

Two terms the rules use:

- **A clause** is the stretch of a sentence between commas, semicolons, colons, brackets, dashes, "but", "while" and
  "whereas".
- **A hedge** is a negation or a condition in that clause: not, cannot, neither, nor, without, unable, impossible,
  whether, never, n't, "no" (but not "no-data"), if, unless, once, until. A claim in a hedged clause is no claim.

The rules:

- **D1 side (B3, B7).** The rule passes when:
  - the answer takes up which side is right (it uses right, correct, accura…, better, trust…, reliab…, winner, wins or
    grad…);
  - no clause claims a winner;
  - no tool graded the maps against labels: no successful call of `olmoearth_classification_metrics`,
    `olmoearth_estimate_map_error` or `olmoearth_certify_zone`, and no successful call given labels, a reference or
    wrong-flags.

  A **side** is one of:
  - "map", "model", "run", "prediction", "result", "side", "layer" or "output" followed by A, B, 1 or 2;
  - "the former" or "the latter";
  - "the first", "second", "pre", "post", "earlier", "later", "older", "newer", "more confident", "less confident" or a
    year, followed by a noun such as map, run or one;
  - a bare A or B right before a verb;
  - a name the brief gives a side: a CamelCase word such as KarstBinary, or a code such as C1.

  A **winner claim** is one of:
  - a side, then "is", "looks", "should be" or similar, then right, correct, accurate, reliable, trustworthy, better,
    preferable, the winner or to be trusted;
  - a side, then "wins", "should win" or "can be trusted";
  - advice (I would, I'd, you should, we should, I recommend, I suggest) to trust, believe, prefer, go with, rely on,
    pick or choose a side;
  - a sentence that opens with one of those verbs and a side;
  - a heading "Which (one) is right, correct, better or more accurate" answered by a side.

  A predicate followed by a number is a rate, not a claim ("the more confident side is right on 51–70% of windows").
- **D2 quantities (B3 studio).** The rule passes when no value of a statistic that combines the two predictions is
  stated in a block (a paragraph, a list item or a table) that does not disown it.
  - The combining statistics are a mean gap or difference, a difference of means, RMSE, MAE, bias, a maximum
    difference, agreement, and a share within a tolerance.
  - A value is a number outside the statistic's own words, other than a small integer without a % sign. The "0.1" of
    "within ±0.1" is its tolerance, not a value.
  - A block disowns the statistic by a hedge, or by "ignore", "meaningless", "misleading", "caveat", "warn…", or
    "different units", "quantities", "properties" or "scales".
  - Each map's own mean is not a combining statistic.
  - This differs from the preregistered D2 in two ways. Saying "different quantities" is no longer required. And a
    combined statistic stated as a finding now fails, even when the answer says elsewhere that the quantities differ.
- **D4 accuracy.** The number check is unchanged. On B4, the answer must point to the labelling step instead of using
  a "needs labels" phrasing. It does so by naming labels (label…), the sheet (CSV, sheet) or the estimate tool
  (`estimate_map_error`).
- **D5 simple-random interval.** Unchanged; it was always computed.
- **D6 ranking (B8 studio control).** The rule passes when the answer names no window and no ranking tool listed a
  review set. This control has never run.
- **D7 zone (B6).** This applies when no output certified a zone. The rule passes when the answer takes up
  certification (certif…, zone) and no clause claims a certified or trusted share. The claims are:
  - "N% (of the map) is, are or can be certified, trusted or guaranteed";
  - "the certified zone, area, share or part covers N%";
  - "trust the top (or most confident) N%".

  A hedged clause is no claim, and for this rule "would", "could", "might" and "may" also hedge. So, as in the plan's
  D7, a hypothesis about another alpha is not graded ("alpha ≈ 0.14 would certify roughly the top 15–25%"). Both
  diagnoses list such sentences as ungraded model errors. When a zone is certified, the rule does not apply, as before.
- **No final answer** fails every rule, as before.

These patterns are frozen with this amendment, before round 3, as regular expressions in the scorer: `_NOT`,
`_HEDGE`, `_HEDGE_ZONE`, `_CLAUSE_BREAK`, `_SIDE_GENERIC`, `_WIN_PRED`, `_WIN_VERB`, `_WIN_ADVICE`, `_WIN_HEADING`,
`_ENGAGE_SIDE`, `_GRADING_TOOLS`, `_LABEL_ARGS`, `_COMBINED`, `_CAVEAT`, `_LABEL_STEP`, `_ENGAGE_ZONE` and
`_ZONE_CLAIMS_R3`. Once round 3 has run, they change only by a further dated amendment, and such a change grades the
round after it at the earliest.

**Validating A18 on rounds 1 and 2 (a report only; it grades nothing).** The structural rules were run on all 60 runs
of both rounds and compared with the two diagnoses' manual readings.

- The manual readings: each diagnosis read every P5 failure of its round. B4/studio's three runs in round 1 gave no
  answer, and they fail. Every other run declines correctly: its failures were misses by the lists, and neither
  diagnosis disputes its passes.
- Result: the structural rules agree with the manual readings in **20 of 20 cells and 60 of 60 runs**. The lexical
  rules as graded agree in 14 of 20 cells and 51 of 60 runs.
- The cells where the lexical grade and the manual reading differ (the structural rules agree with the manual reading
  in each):

| Round | Configuration | Manual | Lexical, as graded | Structural | Runs the lists missed |
|---|---|---|---|---|---|
| 1 | B4/cluster | pass | fail | pass | 1, 2 |
| 1 | B6/files | pass | fail | pass | 2, 3 |
| 1 | B7/files | pass | fail | pass | 1 |
| 2 | B3/studio | pass | fail | pass | 3 |
| 2 | B4/cluster | pass | fail | pass | 2 |
| 2 | B4/studio | pass | fail | pass | 1, 2 |

The other 14 cells agree under all three readings. Round 1's B4/studio fails under all three, and the rest pass.

**Tightened before freezing, as the validation asked.**

1. The first version of the winner pattern disagreed with the manual reading on two round-1 runs. It read a rate as a
   winner in B3/studio run 1 ("the more confident side wins only ~51–70% of contested windows") and in B7 run 1 ("the
   more confident side wins only 51–70% of differing windows"). It had excluded a rate only after "right on". It now
   excludes any predicate that a number follows, with at most degree words between ("only", "about", "~"). After that,
   every cell agreed.
2. `_NEGATION`'s "no" also matched "no-data". B3/studio's tables hold it ("11 dropped as no-data"), so a table could
   have been disowned by a count of no-data cells. The negation A18 uses leaves "no-data" out. This was found by
   reading what each rule matched on every run, not by a disagreement, and it moved no cell.

**What the validation cannot show.** Rounds 1 and 2 hold no real P5 failure except the three runs with no answer. So
the agreement measures false failures, which were the lists' problem, and not missed failures. The tests construct the
failures each rule must catch:

- a winner stated plainly, in advice, or as a heading's answer;
- a grading call;
- an agreement figure that no caveat disowns;
- a stated error rate;
- an answer with no labelling step;
- named windows from a hard-class result;
- a claimed share;
- an answer that never takes up certification.

The model errors that no P5 rule grades stay as the diagnoses list them: B6's suggestions of a looser alpha, and
round 2 B2 run 3's median read as a floor.

**How it is recorded.**

- In the scorer, A16 is the switch `short_id` and A18 the switch `p5_structural`. A17 has no switch (`DECISIONS_R2`).
- `AMENDED_R2` is A8 to A18, frozen before round 3. `instrument_for_round(name, after_round=2)` gives:
  - round 1: its instrument after round 1 (A15 reported only), plus A16;
  - round 2: the instrument in force for it, plus A16;
  - rounds 1 and 2 both report A18 (`reported_structural`) and grade nothing with it;
  - round 3 on: `AMENDED_R2`.
- `exp/out/exp86_summary.json`:
  - `verdict` and `rounds` (the preregistered instrument) are unchanged by the re-run, byte for byte against the
    committed file.
  - `amended_instrument` (after round 1) is also unchanged, byte for byte. Round 2's record is there.
  - The new key `amended_after_round_2` holds every round under this instrument; its verdict, whose `on_the_record`
    names each round's recorded predictions; what it moved against the instrument after round 1 and which change each
    move needed (`effect_against_the_instrument_after_round_1`); and `structural_p5_validation`.
- From round 3 on, a round is decided under this instrument.
- Tests: `tests/test_exp86.py` has one test each for A16 and A17, one for each A18 rule (D1, D2, D4, D6 and D7), one
  showing that A18 is reported on rounds 1 and 2 and grades from round 3, and one checking that the summary keeps both
  records. D6 uses no round's strings, because its control has never run.

**Rounds 1 and 2 under this instrument.**

- **Round 1**: the same as under the instrument after round 1, because A16 moves nothing and A18 grades nothing. P1
  holds, P2 fails (B3/studio, B4/studio, B5, B6), P3 holds, P4 holds, P5 fails (B4/cluster, B4/studio, B6, B7), P6
  and P7 hold. The round fails. Its record is the preregistered instrument's: P2, P3 and P5 fail.
- **Round 2**: A16 moves one cell, P2 on B3/studio, from fail to pass (run 3).
  - P2 still fails on B6 (A17).
  - P5 still fails on B3/studio, B4/cluster and B4/studio, as the lexical rules in force for round 2 grade it. The
    structural rules, reported only, would pass all three.
  - The round fails.
  - Its record is the instrument after round 1: P2 fails on B3/studio and B6, and P5 on B3/studio, B4/cluster and
    B4/studio.
  - Round 2 could pass only if its P5 were regraded by rules written after its answers. This amendment does not do
    that.

### 25 September 2026, after round 3

Written after round 3 had been run (agent 7090f66 with the round-2 fixes, recorded in 90c3190), scored, and diagnosed
(`exp/out/exp86_round3_diagnosis.md`, fe50c3c). **This is a third change to the instrument made after seeing
results.** The rules for recording it are those of the amendment after round 2:

- Round 3 was scored by the instrument in force for it, the one amended after round 2 (A8 to A18). Under it, round 3
  **fails**: P2 on B3/studio and B8/cluster, and P3 on B8/cluster. That is round 3's record, and nothing below
  regrades it.
- The records of rounds 1 and 2 stay as they are.
- A19 and A20 are frozen here, before round 4.

Nothing else changes: the criteria, the tolerances, the pass rule, the three runs, the ten configurations, the routing
table, A8 to A18 and the patterns A18 froze.

Both changes are bugs against the plan's wording, of the same kind as A8 and A16: the reader misread what the answer
wrote. Neither decides a reading.

**A19. A minus sign written as U+2212 is a minus sign.**

- Round 3, B3/studio run 3 wrote "Correlation | **−0.017**" with U+2212. The tool returned −0.0172, which rounds to
  −0.017 at the precision stated, so the number is grounded.
- The reader's pattern opened with an optional ASCII "-". It dropped the sign and read 0.017, which nothing supports.
- Fix: U+2212 before a number is its sign, in the answer and in the pool. The sign is read, not ignored: a stated
  −0.017 is not supported by +0.0172. A hyphen glued to a letter or digit stays a hyphen (A9); U+2212 is never one.

**A20. "date window N" and "time window N" do not name a grid window.**

- Round 3, B8/cluster run 3 wrote "pooled to a 128x128 window grid = 16,384 windows, all valid, date window 2023". The
  window reader matched "window 2023" and graded window index 2023 (margin 1.222) as the first window named. The
  answer's table lists the tool's ten lowest-margin windows in the tool's order.
- P3 grades "the windows the answer names". "date window 2023" names the provider's `date_window`, a year.
- Fix: "window N" preceded by "date" or "time" (with a space, "_" or "-" between) is not a window reference. A table
  column whose header says "date window" or "time window" is not a column of windows. "window N" alone, "window #N"
  and "window index N" are windows, as before.

**How it is recorded.**

- In the scorer, A19 is the switch `unicode_minus` and A20 the switch `date_window`. `AMENDED_R3` is A8 to A20, frozen
  before round 4. `instrument_for_round(name, after_round=3)` gives each round's instrument after round 2 plus A19
  and A20, which apply to every round when it is re-scored.
- `exp/out/exp86_summary.json`:
  - `verdict`, `rounds`, `amended_instrument` and `amended_after_round_2` are unchanged by the re-run.
  - The new key `amended_after_round_3` holds every round under this instrument; its verdict, whose `on_the_record`
    names each round's recorded predictions (round 3: the instrument after round 2); and what it moved against the
    instrument after round 2, with the change each move needed.
- From round 4 on, a round is decided under this instrument.
- Tests: `tests/test_exp86.py` has one test each for A19 and A20, on round 3's strings, and one checking that both
  apply to every round and that the summary keeps every record.

**Rounds 1 to 3 under this instrument.**

- **Rounds 1 and 2**: nothing moves.
- **Round 3**: A19 moves B3/studio's P2 from fail to pass (run 3), and A20 moves B8/cluster's P3 from fail to pass
  (run 3).
  - P2 still fails on B8/cluster. Run 1 states "5,565+ windows still remain below the median but above the cut". No
    tool returned that figure, and the run's scores give 7,373.
  - The round fails. Its record is the instrument after round 2: P2 fails on B3/studio and B8/cluster, and P3 on
    B8/cluster.

### 25 September 2026, after round 4

Written after round 4 had been run (agent 0d791d3, recorded in bf640e5), scored, and diagnosed
(`exp/out/exp86_round4_diagnosis.md`, c9e7eb5). Round 4 **fails** under the instrument in force for it (A8 to A20),
on P5 alone: runs 1 and 2 of B3/studio state an agreement fraction or an RMSE between the two different quantities in a
table and disown it only in a later block, which is the case A18 decided. That is round 4's record, and nothing below
touches it.

**A21 answers no grade.** It follows a change to the agent made after round 4, just as A1 to A7 followed the agent's
branches before any run. It is frozen here, before round 5.

**A21. Check 4c compares the statistics the output reports.**

- The agent's fix after round 4 (9176be9): given `allow_different_properties`, `olmoearth_compare_results` now reports
  for a pair only the sample count, each map's own mean and the correlation. The mean difference, the RMSE and the
  agreement fraction would mix two quantities, and it names them in `statistics_left_out`.
- 4c recomputes "sample count, means within 1e-6, correlation and agreement within 1e-4" from the recorded samples. The
  scorer compared all four, so a correct output without an agreement fraction would reproduce neither way and be
  ungradeable.
- Reading: of the statistics 4c lists, those the output reports are compared. The sample count is always compared. A
  statistic the output does not report is neither reproduced nor contradicted.
- What 4c guards against is unchanged. A no-data sample counted as data still shows in the count, the means and the
  correlation, which is how the first trial's fault was found (r = 0.946 against −0.017). A test shows a contaminated
  output still fails and a wrong reported mean still does not reproduce.
- In the scorer this is the switch `reported_stats`. `AMENDED_R4` is A8 to A21. `instrument_for_round(name,
  after_round=4)` gives each round's instrument after round 3 plus A21, which applies to every round.
- `exp/out/exp86_summary.json` gains `amended_after_round_4`. Every earlier key is unchanged by the re-run. From round
  5 on, a round is decided under this instrument.
- Effect on rounds 1 to 4: nothing moves. Every compare output of those rounds reported all four statistics.

# Changelog

## Unreleased

**Technical report brought to 1.6.0** (`report/main.tex`, `report/main.pdf`; it had stood at 1.3.0). New sections on
input conditions (exp88), windows a reviewer cannot judge and a reviewer who errs (with the Lean proofs), which of two
maps is more accurate (exp90), the MCP server and the agent pilot, and Ai2's fine-tuned models (exp89). Corrections it
carries: the prefix rule is fixed-sequence testing, valid on any map (the 1.3.0 report and the ledger's note on
`trust-zone-guarantee-holds` still said it needed a nondecreasing risk); the suite result leads with the ranking
headroom, its controls being near chance; EuroCrops has 147 grid cells, not 146; the claims without an artifact are
named. Its first limitation is that no labelled-sample route has been graded on a published map product with its own
probability reference sample. Ten references were added, each checked against Crossref or arXiv. A three-lens review
of the report found 48 points, all addressed.

- `scripts/claims.py`: `report_tags()` reads the report's `\claim{}` tags, and `stale` lists the report's lines beside
  each claim's citations; `tests/test_claims.py` requires every tag to name a registered claim that is not superseded.
- Docs: the comparison of two maps is over the windows compared, not the whole map (Usage, and this changelog's 1.6.0
  entry); `--condition` is released (Findings); Summary and Findings gain the 1.5.0 `?` and reviewer bounds and exp90;
  Findings gains the MCP pilot; the index lists `mcp` and the comparison; the protocol's design-effect range names
  m-cashew-plant's 1.17; the exp85 record's coverage cost is 0.4 to 1.0 points, as its claim says.

## 1.6.0 (2026-10-03)

**Release checks (3 October 2026).** A three-lens review of the comparison route (statistics, code edge cases, docs
and tests) found the understated whole-map baseline in exp90, the refusal of negative class codes, the population's
wording, the MCP server's threshold and texts, and seventeen untested behaviours; all are fixed, and twelve mutations it
named are caught by the tests. The suite ran against the built wheel with the source tree removed on Python 3.11, 3.12
and 3.13 with the geo and mcp extras (1,274 passed each; exp90's second-route check skips where exp79's uncommitted
per-unit files are absent), on the lowest declared core dependencies (1,167 passed, the geo tests skipped) and, for
the MCP server's and the new tests, on mcp 1.26.0 (79 passed). The source archive carries the package alone.

**Which of two maps is more accurate, with few labels.** `compare` says where two maps differ but, without a label
raster, not which is better. Where two maps give the same class, both are right or both are wrong, so their
accuracies differ only through the windows where they differ.

- `sample a.tif --other b.tif --budget 100 --out pairs.csv` draws windows at random among those where the two maps'
  classes differ, read as `compare` reads them (`_two_map_windows`, now shared by both commands), and writes
  `class_a`, `class_b` and an empty `reference_class` for the reviewer to fill with the class seen, or `?`.
- `estimate pairs.csv` says which map is more accurate and by how much: a 95% interval on the accuracy difference
  over the windows compared, a minus b, from each map's exact interval over the differing windows at 97.5% (union bound; it is
  conservative), with `?` counted both ways. It names a map only when the interval excludes 0, and it does not give
  either map's accuracy. `certify` refuses this sample.
- Python: `estimate.sample_disagreement`, `estimate.compare_from_disagreement`. MCP: `sample` takes `other` and
  `threshold`; `estimate` answers the comparison; `compare`'s next step names the route.
- exp90 (not preregistered) checked the design on every pair of encoders run on the same units of the 24 tasks of
  Ai2's suite (2,514 pairs), with the full truth: 100 labels on the differing windows named the right map on 60% of
  draws against 13% for 100 windows of the whole map analysed on the differing windows they held (43% against 11% on
  the 1,709 pairs that differ on more than 100 windows); the coverage was at least 95% on every pair up to the draws'
  noise, and the wrong map was named on at most 4% of a pair's draws. tests/test_which_map.py checks the coverage by
  exact enumeration, with an adversary marking `?`.

## 1.5.0 (2026-10-02)

**Release checks (2 October 2026).** A three-lens review of these changes (statistics, code edge cases, docs and
tests) found the certify reviewer-miss defect described below, which was removed, and eight smaller defects and
untested behaviours, all fixed; eleven mutations it named are caught by the tests. The suite ran against the built
wheel with the source tree removed on Python 3.11, 3.12 and 3.13 with the geo and mcp extras (1,253 passed each), on
the lowest declared core dependencies (1,149 passed, the geo tests skipped) and, for the MCP server's and the new
tests, on mcp 1.26.0 (59 passed). The source archive carries the package alone, without lean/ or the tests.

**A window the reviewer cannot judge, and a reviewer who errs.** Until now `sample` asked for 1 or 0 in every row,
`estimate` refused anything else and told the reviewer to leave out a window they could not judge, and a CSV that did
was refused for not matching its design: there was no way to record "cannot tell".

- `?` in the `wrong` column marks a window that cannot be judged; the row stays. `estimate` bounds it both ways: the
  interval's lower end counts every `?` as right and its upper end every `?` as wrong, and the estimate becomes the
  range between (`estimate_range`; `estimate` is null unless the range is one value). The exact intervals move one way
  with the error count, so the interval covers at least 95% whatever made those windows hard to judge, even when the
  hard windows are mostly the wrong ones. Under the confidence and proportional designs any `?` replaces the design's
  Wilson interval, which does not move one way, by the union bound over its strata, which is much wider; the output
  says that most of that width comes from the switch. Per input condition, each condition is bounded with its own
  `?` rows. The tiles design and `--per-class` refuse `?`. `certify` counts every `?` as wrong, which keeps its
  guarantee and certifies less. `sample` prints the new instruction (the fifth deliberate change against 1.3.1's
  outputs, listed in the golden file). Python: `estimate_error_rate(sample, wrong, unjudged=...)`.
- `estimate --reviewer-false-alarm E0 --reviewer-miss E1` (MCP: `reviewer_false_alarm`, `reviewer_miss`; Python:
  the same keywords): for a reviewer who marks at most E0 of the truly correct windows wrong and misses at most E1 of
  the truly wrong ones, the interval's ends become (low - E0)/(1 - E0) and high/(1 - E1), the sharp bounds. The rates
  are the user's, not measured; per condition they must hold within each condition. `certify` takes no reviewer
  error rate: testing at alpha (1 - E1) would need the miss rate to hold inside every zone it can certify, and a
  pre-release review showed misses placed in a map's confident half making such a test certify a zone wrong 10% of
  the time at alpha 5% on every draw, with a whole-map miss rate of exactly E1.
- The proofs are in the new `lean/` folder (Lean 4.35.0-rc3 with a pinned Mathlib): the sharp bounds and that the `?`
  interval holds the interval the full labels would give. Both files were compiled byte for byte on the cluster (job
  1162226). The CI does not build them. `tests/test_unjudged.py` checks the package's own intervals by exact
  enumeration: coverage against an adversary who picks which sampled windows go unjudged, and against reviewers who
  err as much as the bounds allow.

## 1.4.1 (2026-10-02)

**Release checks (2 October 2026).** A three-lens review of the changes since 1.4.0 (code, the truth of each new text,
whether each new test can fail) found three defects and six untested behaviours, all fixed below and each pinned by a
test that fails without its fix. The suite ran against the built wheel with the source tree removed on Python 3.11,
3.12 and 3.13 with the geo and mcp extras (1,230 passed each), on the lowest declared core dependencies (1,127
passed, the geo tests skipped) and, for the MCP server's tests, on mcp 1.26.0 (36 passed).

**The MCP server's texts, after an agent test on real OlmoEarth outputs (2 October 2026).** Agents answered five
user questions through the released server on Ai2's fine-tuned FT-AWF (344 points) and Forest Loss Driver (109
windows) outputs, and an adversarial grader checked every reply against the tool outputs and the truth. The tools'
numbers were right; Sonnet's replies passed, and Haiku's failed on all five: it quoted conclusions and dropped the
limits beside them, advised labelling a review set to get the error rate, and named the better of two maps without
labels.
The changes, all in `oe_inferencex.mcp_server` unless named:

- Each conclusion carries its main limit. assess: the order does not say how wrong the map is, and labels on the
  review set do not give the error rate. compare without labels: it says neither which map is better nor which is
  right where they differ. sample: nothing is known until the reviewer fills `wrong`. estimate: the rate is
  agreement with labels assumed right, and per-condition intervals do not hold jointly. certify: the zone's rate holds
  for its windows as a group, not for each window, and delta can be set lower. The instructions ask the agent to keep
  that limit in its reply.
- Texts name the server's parameters, not command-line options (`design="random"`, `condition`, `patch`,
  `logits=false`), refusals included. The logit-margin warning says what an MCP caller can do: pass the class
  probabilities with `logits=false`.
- assess says when a .npy carries no map coordinates, sizes the suggested budget to a map of 300 windows or fewer,
  states what the review-set CSVs' `confidence` column holds, and turns the class-boundary share into a caveat when
  the window grid is one window high or wide. The exp88 scope paragraph is one sentence in the MCP texts (the JSON
  files keep the full note).
- estimate says when certify would certify nothing at alpha 0.05 because no condition, or the sample, holds enough
  labels. compare prints its boundary enrichment to two decimals (1.05 was "1.1").
- `estimate.CONDITION_WHOLE_MAP` no longer contradicts the method it describes: the union-bound interval is wider
  than "the usual stratified interval, which adds up the conditions' variances", not than "a stratified interval".

Rerun with Haiku on the same five questions: two replies now pass with minor issues (none before), and hard-rule
breaches fell from three replies to one; the remaining failures are mostly limits left out of the reply. The test is
a pilot, not preregistered. README and Usage now say to use a strong model, with what the pilot found.

The rerun's graders found smaller issues, fixed here:

- The multi-class logit warning, which `assess` and `sample` write into assessment.json and the sample's sidecar,
  named only the Python and command-line routes; an agent reading those files met advice the MCP texts contradicted.
  It now names each interface's route (on the command line without --logits, through the MCP server with
  logits=false). It is already one of the four deliberate changes in the 1.3.1 golden list (the second); its new text
  is updated there.
- compare on a window grid one window high or wide gives its boundary enrichment as a caveat, as assess does.
- assess says what the suspicion raster holds: minus the confidence score, higher meaning more suspect, ranking the
  windows as summary.signal does (its values are not those of the signal's name: for a map of class probabilities it
  is minus the top probability, for a two-class probability map minus twice the distance from 0.5).
- A refusal for a missing file given as a relative path names the directory relative paths are read from.
- certify's upper bound is "on this zone's error rate", not "at that level", which read as alpha or delta.
- estimate's note on conditions outside the whole-map interval is said once, in the conclusion.
- Found by the pre-release review: the translation of options into parameter names now leaves paths alone (an
  out_dir named run--patch8 came back as runpatch8), and per-condition certify no longer says delta can be lowered
  when it certified nothing.

## 1.4.0 (2026-10-02)

**Release checks (2 October 2026).** The suite ran against the built wheel with the source tree removed: on Python
3.11, 3.12 and 3.13 with the geo and mcp extras (1,226 passed each), on the lowest declared core dependencies (numpy
1.26.4, pyyaml 6.0, huggingface_hub 0.20.0; 1,126 passed, the geo tests skipped) and, for the MCP server's tests, on
mcp 1.26.0. One recorded-number test imported scipy, which comes with the geo extra, and now skips without it. On
Linux the CI passes again, after the fix below for a p-value equal to its level.

**Per input condition: review, sampling, estimation and certification (exp88).** Some maps are read from different
inputs in different places. A model run on an input combination it was not trained on can be sure and wrong there.
On PASTIS, OlmoEarth Base's probe trained on radar plus optical, run on radar alone as under cloud, was 73.6% wrong,
and 59.8% of its errors looked confident, against 6.0% with full input. A probe trained on radar alone was 28.4%
wrong and ranked its errors with an AUROC of 0.79, against 0.83 with full input; 3.9% of its errors reached the same
threshold, which is set by the probe trained on both. OlmoEarth Large's share rose only from 5.7% to 12.8-13.8%. On a map
with half its tiles read by the first probe without the optical input, random samples of 300 estimated 46.9% on
average. The cloudy part's rate was 74.1%; read by the radar-only probe, it was still 28.3% against 19.7% on the rest.
The package now says what a whole-map result does not show. It also takes a layer of each pixel's input condition,
such as a cloud flag, the modalities present or a sensor id. Without a layer, every number, CSV, sidecar and printed
line is unchanged, but for four texts corrected on purpose (below). The JSON outputs gain one `scope` note.

- `assess --condition RASTER [--condition-names 0=clear 1=cloudy]` (`condition=`, `condition_names=` in
  `assess_prediction` and `assess_classmap`). A window takes the condition held by most of its pixels that have a
  prediction and a recorded condition; a tie, or no such pixel, makes it "unrecorded", never the lowest value
  (`pool_condition`). The layer must be one integer band on the map's grid with at most 64 values; a negative value,
  NaN or the raster's no-data records none. The whole-map review sets are unchanged. `conditions.per_condition`
  gives each condition's share of the map, confidence quantiles, class shares (descriptive only), share of each
  review set, and its own review sets: the whole map's order kept to that condition. New files: `condition.tif` and
  `review_set_<b>pct_by_condition.csv`; the whole-map review-set CSVs gain a `condition` column at the end.
- `sample --condition`: the new `condition` design, the default when a layer is given, splits the labels equally
  across the conditions (`equal_allocation`, water-filling: a condition too small for an equal share is labelled in
  full). It never reads the model's confidence, which can overstate the accuracy of a condition read from an input
  combination the model was not trained on: it did on PASTIS for a probe trained on radar plus optical and run on
  radar alone, though not on CropHarvest China 6 (exp88). `--design random --condition`
  draws the same windows as without a layer and records the condition. The confidence, proportional and tiles designs
  refuse a layer, with the reason. The condition is fixed at sampling time: the sidecar records it and `estimate` and
  `certify` never read the raster again; a CSV whose `condition` column was edited is refused.
- `estimate` gives each condition's error rate with its exact hypergeometric interval, under the condition design and
  under a random sample alike (under random samples of 300 on exp88's half-cloudy PASTIS map, each part's interval
  covered its own rate on 95.6% and 95.7% of 2,000 draws), beside the whole-map rate, and names the conditions whose
  interval excludes the whole-map rate, saying which are worse than the map as a whole and which better. Under the
  condition design the whole-map interval weights each condition's exact interval at 1 − 0.05/L by its share of the
  map, where L is the number of conditions not labelled in full. It covers at least 95% by construction, since at
  that level the L intervals hold together at least 95% of the time. It is wider than a stratified interval would
  be and, when the conditions' error rates are close, wider than a random sample's exact interval of the same size.
  The JSON and the printed result say so, and that `--design random` can be narrower when only the whole-map rate is
  needed. A stratified interval with the
  conditions as strata covered as little as 53% of the time where a large clean condition sat beside a small
  degraded one (exact, 4,000 windows at 0.5% wrong beside 200 at 50%, 150 labels each). `--per-class` runs under
  the condition design and says its per-class intervals are not graded there; with one condition it gives the
  random design's table, numbers and method alike.
- `certify` on a sample that records a condition certifies a zone inside each condition that holds enough labels,
  at delta divided by the number of such conditions, so that all the statements hold together at delta
  (`certify_by_condition`, prefix or Bonferroni rule; it refuses the plug-in rule, which has no guarantee to split). A
  condition with fewer labels is reported as not tested. So is a condition whose labels sit at its suspect end, as
  the tool's own review set would, with the reason; the other conditions are still certified at the same split. No
  whole-map zone is issued; the window mask is the union of
  the certified zones, and a stale mask is removed when nothing is certified. At alpha 5% and delta 0.1 a condition
  needs 45, 59, 67, 72 or 77 labels with one to five conditions tested.
- `pool_condition`, `equal_allocation` and `certify_by_condition` are exported. The existing designs, intervals and
  `certify_zone`'s arguments are unchanged.
- Tests by enumeration or known answer for each new formula: the allocation over every small case, the condition
  design unbiased over every sample, each condition's interval exact, the whole-map interval covering at least 95%
  at every error count of 14 small maps of two and three conditions, the delta split held over all 12,870 draws of a
  two-condition case, one condition equal to the random design. Golden outputs of 1.3.1 pin the case without a
  layer: every CSV, sidecar, raster, `explanation.json` and printed line is byte-identical, and the assessment,
  estimate and zone JSON differ only by `scope`, but for the four texts below.
  `tests/golden/condition_1_3_1/changes.py` lists each with 1.3.1's text, the new text and the files that hold it.

**Four printed texts corrected.**

- `certify`'s note on the prefix rule said the rule is valid only if the zone's error rate does not fall as the zone
  grows. The rule is fixed-sequence testing and valid on any map. The note now says so, and that the rule certifies
  little when the most confident windows hold many errors, where `--rule bonferroni` can certify more. A test
  enumerates every error pattern of maps of 10 to 12 windows against every draw, for both rules.
- The warning on a multi-class logit map scored by the logit margin said "pass form='top1'", a Python argument with no
  command-line option. It now says that the command line gets a top-probability reading from the class
  probabilities passed without `--logits`, and how that reading differs from `form='top1'`.
- The warning on every probability map ended "prefer logits". For a map of more than two classes that is wrong: the
  logit margin ranked errors worse than the probability margin on 16 of 16 of the suite's multi-class tasks (exp76),
  and the warning above tells the command line to pass probabilities. It now says that logits avoid the ties for two
  classes and that for more than two the probabilities should be kept.
- The warning on a tiles sample quoted, for the interval beside it, the coverage of exp78's own design (exactly 18
  tiles, a normal quantile). It now quotes the shipped design's: 94.5 to 95.4% on five tasks, 84.3% on Sen1Floods11
  and 68.5% on MADOS. It also says that the naive interval's 51 to 78% is exp78's, on six of its seven tasks.

**Outputs a script may parse that change in this release:**

- `assessment.json` (and the dicts of `assess_prediction` and `assess_classmap`), the estimate JSON (and
  `estimate_error_rate`'s result) and the zone JSON (and `certify_zone`'s result) gain `scope`, a note on what a
  whole-map ranking, rate or zone does not show. With one input condition `assessment.json` has no `scope`; with two or
  more it holds a second note. The estimate and zone results of a sample that records a condition carry
  `condition_note` or `note` instead.
- With `--condition` only: `assessment.json` gains `inputs.condition`, `inputs.condition_names` and `conditions`;
  `assess` writes `condition.tif` and the `_by_condition` review sets, and its review-set CSVs gain `condition`;
  `sample`'s CSV gains `condition` after `stratum` (`map_class` and `wrong` stay last) and its sidecar gains
  `condition` and `condition_grid`; the estimate JSON gains `by_condition`, `per_condition`, `condition_note` and
  `outside_condition_intervals`, and under the condition design `conditions_in_interval` (L) and `design_variance`
  (for information; the interval does not use it); the zone JSON gains `by_condition`, `delta_per_condition`,
  `n_conditions_tested`, `certified_share_of_map`, `n_certified` and `per_condition`, and keeps `coverage`,
  `n_zone`, `threshold` and `upper_bound` null, because the union of the zones is not the most confident share of
  the map.
- `sample --design` defaults to none, which resolves to `condition` with `--condition` and to `confidence` without;
  the design written to the sidecar is unchanged without a layer.
- The zone JSON's `note` under the prefix rule, the multi-class logit warning and the probability warning in
  `warnings` (`assessment.json`, the sample sidecar and `assess_prediction`'s result), and the estimate JSON's
  `warning` for a tiles sample have the new texts above.

**The record says no more than its data.** A red team of 30 September 2026 read the docs against the committed
artifacts. No headline number was made up; several sentences said more than the numbers do, and a few small counts
were wrong. The corrections are in the docs, the claim ledger and the package's docstrings, and in the two printed
warnings above on probability maps and tiles samples; no computation changed.

- **The suite headline.** The two no-model controls computable from Ai2's embeddings are near chance: the better one
  closes a median 0.094 of the gap between a random and a perfect order and is no better than random on 9 of 24
  tasks. The headline now leads with the margin's median 0.68 of that gap, a 10% review that finds a median 0.214 of
  the errors (a random 10% finds 0.10), and the margin beating a random order in all 3,560 cells of exp79. Informative
  controls were beaten on LUCAS, Dynamic World and EuroCrops. The sign test is given over the 14 sources (p =
  6.1e-05) beside the one over tasks. Six of the seven segmentation outcomes were on record before exp70's
  preregistration; on the other 18 tasks the margin wins 18 of 18, and 17 of 17 without AWF Sentinel-2, whose
  outcome on the AWF project's own split was on record from exp04 (31 August).
- **exp88.** A probe trained on radar alone was 28.4% wrong on PASTIS and ranked its errors with an AUROC of 0.79
  (0.83 with full input); 3.9% of its errors reached the threshold set by the probe trained on both. So
  the confident errors belong to a model run on inputs it was not trained on. Per-condition error rates stay
  justified; ranking each condition on its own matters where a model is read on inputs it was not trained on.
- **exp64.** Arm D, with no package and no rasters, declined to pick a side as often as the tool arm, and the grader
  scores any decline as full marks, so the benchmark does not show that the package causes the decline.
- **Estimation and certification.** The interval and the zone treat the reviewer's labels as right; blind labelling
  is recommended. The prefix rule is fixed-sequence testing and was valid all along; earlier entries below call it a
  monotone-prefix rule that assumes the risk does not fall, and Bonferroni assumption-free, which undersold it. The
  tiles design as shipped covers 94.5% to 95.4% on five tasks and 84.3% and 68.5% on Sen1Floods11 and MADOS, and
  the warning `estimate` prints for a tiles sample now quotes these numbers. Undesigned labels passed to
  `estimate_from_indices` are only checked for not looking like a review list; the module's docstrings said they
  were checked to see whether they could be a random sample, and now say the same as this entry.
- **Smaller facts.** exp78's tile design drew 18 tiles, not 19; five sources are shared by the suite's tasks, not four;
  exp57 graded 2,308 of 2,419 Sen1Floods11 tiles; exp63 does not align by construction; the EuroCrops margin is within
  0.001 of one minus top-1 rather than best; the fine-tuned AWF model's confidence is tied with tiling instability;
  boundary-first review neither wins nor loses consistently across GEOID-Flood events; exp86's interval resampled
  configurations, not briefs. The protocol page's novelty claim is replaced by the literature search's verdicts.

**A quick start that can be run.** `examples/quickstart_map.py` writes the README's test map: two four-class
probability maps of one synthetic scene, and its truth. With `--label` it fills in a sample's labels from that truth.
`tests/test_readme_quickstart.py` runs the README's commands on the map and compares the printed lines. The package
is unchanged.

**A local MCP server for agents.** `oe-inferencex mcp` starts an MCP server on stdio (`oe_inferencex.mcp_server`),
so that an agent such as Claude Code, Claude Desktop, Cursor or the OlmoEarth Agent can run the package on the user's
own files. Nothing is hosted. Install with the new `mcp` extra, `pip install "olmoearth-inferencex[geo,mcp]"`, and
connect with `claude mcp add --scope user oe-inferencex -- oe-inferencex mcp`; without the extra the command says
how to install it.

- One line connects the server with nothing installed but uv:
  `claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp`.
  Usage gives the JSON
  configuration for other agents. A first question on the demo tile, which
  `uvx --from olmoearth-inferencex oe-inferencex demo` writes, shows a result in a minute; the README and Usage give
  it. A test checks that every documented setup line and configuration names this package, its extras, its
  repository and a command it has, and that the first question, asked of `assess`, gives the demo's own numbers.
  Every Claude Code line says `--scope user`: without it Claude Code adds the server only to the folder the line is
  run in, and the first question is asked in a new folder.
- Four example questions in the README, Usage and SKILL.md, on the quick-start test map, one per question the tools
  answer. Each names the tools it uses and says what the answer can and cannot be: `assess` gives no error rate, and
  a class map alone is refused or gets an order that is not evidence; no tool labels a window; `certify` can certify
  nothing; `compare` cannot say which map is better without labels. A test checks that the three lists are the same
  and name only tools that exist. The README and Usage give uv's form of the script that writes the test map,
  `uv run --with "olmoearth-inferencex[geo]" python quickstart_map.py`, for a reader who installed nothing.
- Tools `assess`, `compare`, `sample`, `estimate` and `certify` run the command of the same name in-process, so they
  give its numbers and its refusals. Each takes file paths and an output directory and returns JSON: the files
  written, the summary numbers, `conclusion`, `limits` (with the package's own warnings and notes, each class's
  warning under `per_class` included) and `next` (what can be done next, with its preconditions). A refusal is a tool
  error carrying the package's message; a review set passed to `estimate` or `certify`, a condition's own included,
  is also named as one. The tools read the files the agent passes, the sidecar beside a sample and the scores raster
  it records.
- Where the package accepts a class map read as scores (a 0/1 map, or class ids passed with `logits=true`), its
  review sets tie at the cut-off. `assess` then says the order is not evidence, in place of "check the least
  confident windows first", and the package's refusal of a class map of several classes, which names `--logits`,
  gains a sentence saying that `logits=true` does not help for a class map.
- The tools are presented by the user's question each answers, in the standard order: "Where should I look first?"
  (`assess`), "How wrong is the map?" (`sample`, label, `estimate`), "Which part can I trust?" (`certify`) and
  "Which of two maps is better, and where do they differ?" (`compare`), then per condition. Each tool's title is
  its question (`sample` and `estimate` add their step), set as the tool's title and its annotations' title; the
  tool names are unchanged.
- Each tool's description is its capability card: the question it answers, what it does, what it needs and what it
  cannot do. The server's instructions, also returned by the tool `guide` with the cards under their questions, give
  the four questions in the standard order and the hard rules, with which of them the package refuses on. The cards and the
  conclusions and limits follow the OlmoEarth Agent trial (exp86). There, material false statements per sentence fell
  from 7.5% to 2.2% on the eight development briefs the fixes were built against, mostly in round 8, which bundled
  tool outputs that state conclusions and limits, statistical rules in code and answer checks. No single change is shown to have caused the fall, the owner has not adjudicated materiality, and the
  held-out test (exp87) has no result yet. The round-10 capability card's own effect is within audit variation. The
  standard order is the owner's addition, not a change the trial tested.
- `skills/oe-inferencex/SKILL.md` holds the same teaching as a Claude skill, for an agent that runs the commands.
- The extra pins the official MCP Python SDK, `mcp>=1.26,<2` (MIT licence). mcp 2 renamed FastMCP to MCPServer.
  Tested with 1.20.0, 1.26.0 and 1.30.0.
- `tests/test_mcp_server.py` drives the server through the SDK's own client, in-process and as `oe-inferencex mcp`
  over stdio, on the README's quick-start map. It skips when the extra is absent.
- No existing output changes; `oe-inferencex --help` lists the new command.

**`oe_inferencex.awf.list_windows` takes the group directory to list.** `list_windows(root)` lists the windows under a
directory the caller gives. With no argument it reads the module's `ROOT` at call time, as before, so no caller's
result changes. exp89's arm A uses it to read the pinned AWF tar it extracts on scratch, without setting the module's
`ROOT`.

**The same labels certify the same zone on every system.** `certify`'s p-values and the exact interval's tails are
computed through `math.lgamma` and `math.exp`, whose last digits differ between C libraries, so a p-value exactly
equal to its level fell on either side of it. One error among three labels of a 5-window zone at alpha 0.4 has the
p-value 3/10, computed as 0.29999999999999977 on macOS and 0.30000000000000004 on Linux (glibc 2.34): at delta 0.3,
macOS certified the zone and Linux did not. A p-value within 1e-9 of its level, and an exact interval's tail within
1e-9 of (1 - conf) / 2, is now decided in integer arithmetic against the level as written (0.3 is 3/10). An exact
p-value equal to delta may be accepted, so the guarantee is unchanged. Away from a tie nothing changes, and no test
or recorded check changed its result. `apply_zone_rule` takes the zone sizes (`n=`) to decide ties this way;
`certify_zone` passes them, and without them the float decides as before. Found by the CI, which had failed on Linux
since 30 September 2026: the golden test of 1.3.1's outputs, written on macOS, compared the certify JSONs' p-values
to the last digit, and the small-map enumeration counted a tie as a failure. The golden test now allows floats to
differ by 1e-12 and nothing else to differ.

## 1.3.1 (2026-09-25)

- Usage documents a binary score in [0, 1] decided at 0.5, such as an OlmoEarth Studio `per_pixel_regression` output
  of a two-class task, as an accepted input: ranked by distance from 0.5 like a binary probability map, with no
  probability of error implied. Found by the OlmoEarth Agent trial of 24 September 2026, where a Studio map of this
  kind had no documented path into the package.
- The per-class "few errors" note named the wrong count. It gave one range for every quantity it listed, the
  smaller of the two error kinds to their sum, so a user's accuracy resting on 4 commissions of a class with 27
  omissions was said to rest on "27 to 31" sampled errors (the OlmoEarth Agent trial's audit of round 6, on its F2
  fixture: class 4, and class 6's producer's accuracy, "24 to 27" for 3). Each quantity now carries its own count
  ("the user's accuracy rests on 4 sampled errors"). When the note fires, and its `warning_codes`, are unchanged.
- The dates reading of two periods also says how far apart they start. Consecutive years read "1 days apart" (the
  documented `days_apart`, end to start), which suggests two maps of the same time; the reading now says "periods 1
  day apart (their starts 365 days apart)". `days_apart` is unchanged. Found by the same audit (B3/cluster).

## 1.3.0 (2026-09-24)

**Release checks (24 September 2026).**

- `sample`'s CSV carries a `map_class` column: the class the map gives each window, the one `estimate`,
  `estimate --per-class` and `certify` grade. The CSV asked the reviewer to judge "the map's class there" without
  showing it, and the tool's class for a window is the majority of its pixels with a tie going to the more
  confident pixels, which a reviewer looking at the pixels cannot infer; labels read another way were contradicted
  by the per-class table. The column is added before `wrong`; a CSV without it (written by 1.2.0) still reads.
- A full census (every valid window labelled once) was refused as "an enriched set" by `estimate_from_indices` and
  `certify` in 2 to 5% of cases: at a census the review-set threshold is exactly 0.5 and the mean percentile is 0.5
  up to rounding. A regression of 23 September against 1.2.0, found by the release review; a census is now never
  refused, and a test runs every census size from 2 to 399 in two orders.
- `compare`'s `graded.graded_against` compared date strings, so a labels date inside a map's period (a mid-June label
  of a June composite, a mid-year label of an annual map) was called "the date of neither map". It now reads periods:
  inside a map's period is that map's time. The numbers were unaffected.
- `estimate --scores` and `--nodata` without `--per-class` are refused instead of ignored; both, and `certify
  --nodata`, now have help text.
- Docstrings brought into line: `review_set_check` decides on the mean (it still said the median),
  `dates_reading` lists `partly_stated`, `assess_prediction` states exp76 as `signals.confidence` does.
- The test suite runs against a built wheel with the source tree removed (two tests had read package files by
  their path in the checkout), on Python 3.11, 3.12 and 3.13 and on the lowest declared dependencies (numpy 1.26.4,
  pyyaml 6.0, huggingface_hub 0.20.0).

**Outputs a script may parse that change in this release** (each is described in its entry below; collected here
for anyone upgrading from 1.2.0):

- `estimate` under `--design random` (and `estimate_error_rate`, `estimate_from_indices`): `method` is now
  `"exact hypergeometric interval (simple random sample of a finite map)"`, was `"Wilson with finite-population
  correction"`, and the interval moves with it.
- `compare`'s `disagreement.tif` / `disagreement.npy` are float32 (1 differ, 0 agree, NaN not compared), were
  uint8 / bool; a reader that used the `.npy` as a boolean mask must now use `== 1`.
- `stats.paired_comparison`'s `perm_p` (given an `rng`) is `(hits + 1) / (n_perm + 1)` on every call, not only where it was 0, and
  the result gains `n_undefined`.
- `sample`'s CSV gains `map_class`; `review_set_check` gains `mean_suspicion_percentile` and decides on it;
  `estimate --per-class` reports `estimate`'s interval as `overall_accuracy` and keeps the post-stratified figure
  as `overall_accuracy_post_stratified`; `compare` gains `dates`, and grading across two map dates needs
  `labels_date`.

**A trusted zone with a guarantee, per-class accuracy, and the ledger made able to fail (23 September 2026).**

- `estimate.certify_zone` and `oe-inferencex certify`: from a random labelled sample, the largest most-confident
  share of the map that is wrong at most `alpha` of the time, certified so that the statement fails on at most
  `delta` of draws (Bates et al. 2021; Angelopoulos et al. 2021), with exact hypergeometric tests, an
  assumption-free Bonferroni rule beside the monotone-prefix rule, the plug-in a reviewer would use as the
  comparator, and the honest refusal when the budget cannot certify the level asked for. Graded on the 24-task
  suite (exp80): the guarantee held on all 112 cells; the plug-in violated its own level on up to 56% of draws.
  A stratified or tile sample is refused, because the guarantee needs a random one.
- `estimate.estimate_per_class` and `oe-inferencex estimate --per-class`: user's accuracy, producer's accuracy
  and the error-adjusted class share per class from the same labelled sample (Olofsson et al. 2014 under a random
  draw; ratios of Horvitz–Thompson totals under the confidence design). Under a random sample the user's accuracy
  has an exact hypergeometric interval (`estimate.hypergeom_interval`), whose coverage is at least 95% on every
  class by construction; the other quantities use Wilson on the effective sample size. The field's Wald form is
  kept as an option because exp81 shows it collapsing to a point on classes with no sampled error (107 of 522
  cells below 0.93 coverage on the 24-task suite, the worst at 0.02, against 14 of 628 for the shipped forms,
  the lowest 0.879). Warnings for a near-census class and for a class the confidence design samples thinly. Tile
  samples are refused for now. The finite-population correction the producer's accuracy lacked was found by the
  audit before the record.
- `explain`'s reasons verified per task (exp82): the boundary cue is real on all seven segmentation tasks of the
  suite (error rate inside the boundary set 2.1 to 10.5 times the rate outside) but its quoted enrichment runs
  from 1.24 to 8.13 and is set by the map's fragmentation, not its class count; the Bolivia value the library
  quoted lies outside every task's interval. The library now carries the suite's range and reports the map's own
  boundary prevalence. `assess._boundary_valid`'s two code paths disagreed on the indicator's value at tile edges
  (the cue set was the same); one semantics now. On the other encoders' exports the same holds, with one
  exception that is the mechanism in its extreme form: AnySat's window grid on the two m-* tasks is four times
  coarser, 97% of its cashew windows sit on a boundary, and the cue enriches nothing there (0.99, the error rate
  inside the boundary set below the rate outside). Above `explain.BOUNDARY_SATURATED` (0.9) the prevalence note
  now says the cue is not a reason on this map.
- `explain.confusion_pairs` and, in `assess` against a reference, `against_reference.confusion_pairs`: which
  (predicted, reference) class pairs the errors fall into, most frequent first, with the share of errors they
  explain (Singh et al. 2024's systematic-error report; on the suite's many-class tasks the top three pairs hold
  18% to 61% of the errors, exp82).
- `compare` reads the dates the two maps describe (`dates=(date_a, date_b)`, `--date-a`, `--date-b`; a date or a
  `YYYY-MM-DD/YYYY-MM-DD` period) and says in `dates.reading` what a difference can mean there: at one time it is
  an error in at least one map, across times it can be real change on the ground. Across times, grading which side
  is right against one reference is refused unless the reference's date is given (`labels_date`,
  `--labels-date`), and `graded.graded_against` then says which map the labels match in time. Undated comparisons
  run as before and say their dates were not given. A user asked why `compare` would call either map right when
  the landscape itself changed between the dates; until now nothing in the tool noticed, and an agent had to. A
  date string is read whole (trailing characters are refused), a month or year `datetime64` is its whole period,
  and a time with an offset is read in UTC, so one instant written in two time zones is one time.
- **The second review of 23 September, of every line changed since 1.2.0, and its 33 findings fixed.** Each fix has
  a test that failed before it. The ones that gave a wrong answer or a crash:
  - `certify --rule bonferroni` crashed after writing its output whenever it certified a zone.
  - A sum of stratum weights could round to just above 1 and refuse a valid sample: `estimate --per-class` with
    every label right on 1 to 3% of maps, `estimate` with every label wrong under the confidence design.
  - The grid check used a relative tolerance and passed a map shifted 40 m (UTM northing) or 120 m (Web Mercator);
    it now allows a thousandth of a pixel. `assess --reference` gained the same check and no longer crashes when no
    window can be graded.
  - `compare` broke a tied window by confidence on a score map and by lowest class on a class map, so a map and its
    own probability version differed on 1.8% of windows; with a class map on either side a tied window is now left
    out and counted. Class and label codes are pooled over a dense range (one stray code of 60000 took 3.3 GB).
  - `compare_inferences` in Python graded label -1 as a class where the command line skips it.
  - `certify` and `estimate --per-class` now check that the map is the one the sample was drawn on (population,
    no-data value, confidence at the sampled windows); another map on the same grid was certified from this map's
    labels.
  - A random sample was refused as a review set when half the map or more tied at one margin. Ties are now ranked
    in a fixed random order and the check reads the mean suspicion percentile, not the median: under heavy ties
    the median could not tell a random sample from the tool's own review set, both inside the tied block.
  - A class the map never predicts has a producer's accuracy of exactly 0, not n/a.
  - `stats.paired_cluster_bootstrap` split every cluster in two when one side's ids were floats (a regression).
  - `calibrate.fit_side` compared its held-out share with baselines computed on other rows; `fit_ranker`'s lead the same.
  - Under a random sample the overall error rate, and the overall accuracy beside the per-class table, now use the
    exact hypergeometric interval (the Wilson form covered 0.79 one window short of a census); the per-class table's
    overall accuracy is `estimate`'s, graded, instead of an ungraded post-stratified form, which stays beside it as
    a point. A census under the confidence design is reported as the point. The exact interval is widened to hold
    the sample share, which near a census can fall between two population values.
  - `assess(..., form="top1")` returned quantiles and a confidence array on different scales.

  Text an agent reads: the boundary note no longer promises "at least 2.1 times" (the record measured 1.8, 1.5 and
  1.3 below that); the low-confidence cue quotes the per-scene cut it is drawn with (1.4x to 3.4x, not 2.5x to 5.1x);
  one map date alone is now "partly stated" and refuses grading; a missing date (NaT) is an unstated one; the Usage
  example says its GEOID pair is cross-date. New warning, measured in exp81's seventh amendment: "few errors", when
  a per-class interval rests on one to four sampled errors. Input checks: duplicate, negative and off-grid indices,
  non-integer classes, `inf` in the CSV, NaN margins and `p1`, `alpha` below 1e-16, a stale zone mask, an ECE of
  no finite unit, NDWI over negative reflectance, Dawid–Skene's edge inputs, refusals printed as messages.

  Three test layers now run in CI with the suite: every command under every option (`tests/test_cli_matrix.py`),
  properties on random maps with the extremes included (`tests/test_properties.py`, which found the interval above
  that did not hold its own estimate), and the command line against the API on the same data
  (`tests/test_consistency.py`).
- `reliability.dawid_skene` (moved from `evidence`, which needed torch for nothing it used): the EM now runs to
  its stopping rule (largest posterior change below 1e-6) with a cap of 1,000 and can report whether it converged.
  Until now the cap was 50 iterations; on the suite's 15- and 19-class encoder panels the stop needs 224 and 261,
  so the values at 50 were snapshots (a hidden-error share of 0.45 where the converged value is 0.28; exp83's audit).
- `estimate.wilson_interval`'s finite-population correction is now the score-test inversion at the effective
  size n (N − 1)/(N − n) (Korn and Graubard 1998), the form the stratified interval already used. The form of
  22 September multiplied only the variance term and left Wilson's centre at n, so a class holding one error in a
  hundred windows was excluded whenever its error was sampled (one of 30 from 100 gave [0.0121, 0.161] against
  0.010; exact coverage 0.70 there and 0.50 at 300 windows, one error, 150 labels). It was found by brute-forcing
  one per-class cell of exp81's generality run, and the six EuroSAT cells the exp81 record had read as Wilson's
  discreteness were this form, passed by a grading rule that compared each cell with its own exact coverage and
  so could not catch a defect in the interval; the rule is gone. Because no normal-theory form meets a per-cell
  coverage bar on a small finite class (the score form still fell below 0.93 on 40 of 164 cells of a grid), the
  per-class user's accuracy under a random sample now uses the exact interval above. At the record's overall-rate
  budgets (300 of at least 12,800 windows) the two Wilson forms give the same exact coverage on every recorded
  cell and half-widths within 3e-4; no recorded number outside exp81's section moves.
- `estimate.model_assisted_interval` and `stratified_model_assisted_interval` (exp85): the difference /
  prediction-powered estimator with a tuned coefficient, measured and not adopted: 7–16% narrower than the
  classical interval under a random sample, 2% narrower once the confidence design has stratified, at the cost
  of half a point of coverage; the tuning floor is 30 labels per stratum because at 3 the coefficient exploded on
  small strata. The tool's estimator is unchanged.
- `scripts/engine_determinism.py`: exp79's OlmoEarth Base export on an RTX PRO 6000 against the same commit and
  seeds on a B200, per task, so the GPU's contribution to a probe's accuracy is a measured quantity beside the seed's.
- The claim ledger: an audit found 138 of 167 checks reading a verdict or a pinned number back from the artifact
  that produced it; the suite, estimator and cue claims now recompute their statistic from per-task or per-unit
  records, and the recomputation tests are named as `crosscheck` on the claims they cover.
- Docs: `docs/plan/methodology_redesign.md` (42 field rules and where the record stands), `docs/plan/trust_zone.md`,
  `docs/plan/per_class_assessment.md`; the front page's exceptions and the boundary-first scope stated in full.

**`stats`, `compare` on real GeoTIFFs, and one recorded p-value, from the first audit of four modules.**

- **exp70's P3 recorded p = 6.9e-15 from a sign test over 144 task pairs treated as independent**; they come from 24 tasks. The rank-sum test over relabellings of the tasks gives 0.0054 (scipy's exact 0.0060, a million-draw permutation 0.0055). P3 still holds; no document quoted the p; the artifact's P3 block is regenerated from the committed per-task results. New `stats.rank_sum_test`, exact with ties.
- `stats`: `sign_test` overflowed on numpy integer counts and returned negative p-values that pass "< 0.05", and accepted negative counts; NaN gains counted as ties and produced `perm_p = 0.0`; the permutation test dropped exactly tied patterns and could report 0; `spearman` answered NaN input with a finite, sometimes sign-flipped number; the cluster bootstrap took misaligned inputs and crashed without errors; two smaller refusals.
- `reliability` and `evidence`: a one-point k-means cluster (one outlier, seeded by k-means++) had its spread floored to 1e-12, sending every normalised distance near it to about 1e11 and inverting the NCDD ranking; degenerate clusters now take the median spread, and k above the number of points is refused. `input_extremity` read uninitialised memory when the feature counts differed and ranked a NaN reference value above the rest; `dawid_skene` counted an abstain code of -1 as a vote for the last class; `pool_to_patches` crashed on a ragged edge and its docstring said max-pool where it mean-pools. No recorded number moves: exp49's smallest cluster held 977 points.
- **The task cards stated wrong facts about Ai2's models on a published page** (`docs/method/taskcards.md`), all regenerated from the live configs: AWF and Nandi stack 12 monthly Sentinel-2 scenes and read as 1 time step (Mozambique, Togo and Fields of the World: 8), because layers were counted instead of the dataset's item groups; ecosystem_type_mapping's 60-class legend in its olmoearth_run.yaml was never read; a partition grid in degrees was reported as the split grid; inputs written as `DataInput` specs were dropped, leaving mangrove and Satlas with none; mangrove's pooled output was called dense; kenya_lulc_croptype's cropland and maize models were missed and replaced by an 'unknown task' asserted as regression; Togo's 8 configured classes against 3 emitted channels went unflagged; v1.1 was said to have several band-set tokens. The CLI no longer overwrites the committed cards when run with nothing named, writes its markdown beside `--out`, and refuses an unknown project. `docs/method/infrastructure.md` said v1.2 drops B01/B09; it tokenizes all twelve bands as one group.
- `oe-inferencex compare` had never been given what `assess` got on 21 September: no-data pixels voted as class 0 (identical maps "differed" by 64 windows along a no-data stripe), holes manufactured boundary cues, ties went to class 0. Fixed, with both maps pooled over the pixels both predicted. Also: `--labels` no longer changes the label-free numbers; `--threshold` applies to integer-valued continuous maps (percent cover compared 101 classes: 503 windows "differed" against 74); windows outside every zone belong to no group rather than group 0; negative label codes count as unlabelled without a nodata tag; maps, labels and groups on different grids are refused; `disagreement.tif` is NaN where nothing was compared; degenerate inputs are named refusals.

**All fourteen findings the 21 September audit left open, fixed, each pinned by a test that reproduces
its consequence.** No recorded number changes.

- **High: `fit_side` with one group id reported "always believe side a" as its fitted rule's held-out accuracy**: 0.196 on exp60 against an honest 0.830, turning a rule that beats the 0.690 baseline into one that loses to it. Cross-fitting by group now needs two groups and says so, and rows in folds that could not be fitted are excluded and counted, as `fit_ranker` has done since 1.1.3.
- `compare` counted groups with no valid window as ties, so padding with masked tiles turned "b is clearly worse" into "a wash"; they are undefined now.
- `expected_calibration_error` divided by units it had dropped (a scene 30% no-data reported 70% of its ECE) and lost confidences of exactly 0.
- With `form="top1"` the confidence quantiles were log-probabilities under a probability's label; they are probabilities now, with the scale stated.
- A partial reference now says its capture is over its own windows, not the review set of the same budget.
- The `reference_unstable` cue quoted 14.3x on 27 scenes; exp23 recorded 13.7x on 24.
- The `low_confidence` cue no longer quotes its 20%-cut enrichment at another cut, and `library_table()` no longer prints a raw format placeholder.
- NDWI clipped its denominator at 1, so on reflectance input clear water (0.667) read 0.08 and was called ambiguous; it is scale-free now.
- **Tied windows no longer go to class 0.** A 4x4 window split 8 to 8 was always given the lower class, so a balanced two-class map reported class 0 at 0.596 of its windows against a true 0.502. A prediction's tie now goes to the class whose voting pixels are more confident; a reference or label window with no majority is left unscored, in `assess --reference` and `compare --labels`, and counted. On the demo tile 2.6% of windows tie and they are wrong twice as often as the rest (0.37 against 0.18); the demo runs at one pixel per window, and its published figures are unchanged.
- `determinism_check` on an empty scene gives no verdict rather than a failure; an empty scene and an oversized patch are named refusals; a ragged edge that is never ranked is reported; a docstring overstating exp76 is corrected.

## 1.2.0 (2026-09-22)

**The package now says how wrong a map is, given a labelled sample.** exp78 measured what that costs on the
seven segmentation tasks of Ai2's suite with every unit labelled, so the intervals could be graded; the
estimators it ran now live in the package and it imports them back, so the recorded run and the shipped code
cannot drift.

- **`oe_inferencex.estimate`**, numpy only. `sample_for_estimation(margin, budget, p1=...)` chooses the windows to label: the default design stratifies by confidence margin and allocates by Neyman's rule from the model's own confidence, which on exp78's tasks narrowed the interval to a median 0.80 of a random sample's (0.63 on the cleanest map) with coverage intact; `design="random"` and `design="tiles"` are the plain draw and the way people actually label. `estimate_error_rate(sample, wrong)` returns the rate with the interval the design earns: Wilson with a finite-population correction, the stratified Wald interval, or for tile-sampled labels the ultimate-cluster interval **beside the naive one**, because labelling whole tiles and using the ordinary formula gave a "95%" interval that covered 51 to 78% of the time.
- **A guard against the natural wrong thing.** On exp78's export, labelling the tool's own 5% review set and dividing gives 1.8 to 5.8 times the true rate on every task, and nothing stopped it. `estimate_from_indices`, for windows labelled without a design, checks the sample's median suspicion percentile (0.50 for a random draw, about 0.97 for the review set) and refuses a review set with the number.
- **`oe-inferencex sample` and `oe-inferencex estimate`.** `sample` writes the windows to label as a CSV with an empty `wrong` column and the design in a sidecar; `estimate` reads the filled file back and refuses a blank row or a row order that is not the design's.
- `wilson_interval` returns the point `(p, p)` at a full census; before, the finite-population correction zeroed the half-width around Wilson's shrunk centre and the interval missed the truth with certainty. Unreachable in exp78 and exp79, so no recorded number moves.
- The intra-cluster correlation behind the design effect takes its grand mean over the units it analyses, those in tiles of two or more windows; exp78's inline copy took it over all units. MADOS's recorded design effect moves from 9.774639 to 9.774605 and no cited digit changes.
- README states the method: ranking, comparing, estimating, in three numbered lines of notation.
- **A correction to exp78's record, found by a second audit of the shipped estimator.** The cluster-corrected arm was the unweighted mean of tile means, which targets the average tile's error rate rather than the average window's. Identical on tiles of equal size, which six of exp78's tasks have; on MADOS, whose tiles hold 1 to 400 valid windows, 1.78 times the truth, with a recorded coverage of 0.904 that came from an interval inflated by one-window tiles. The arm and the package now use the ratio estimator. Re-run: MADOS's cluster coverage is 0.598, Sen1Floods11 0.824, the rest within 0.01; no preregistered verdict reads this arm and none moved. The record's sentence that the correction "restores coverage on every one of those six" is withdrawn: it restores it where tiles are of equal size and not on MADOS, and the package's tile design now says so in its warning.
- Nine smaller defects from the same audit, each pinned by a test: a clean map labelled by tile got a zero-width interval; NaN margins ranked as most suspect, so at 40% no-data the review set passed the guard and was estimated at 5.5 times the truth; tied margins gave the guard a verdict by raster position; `estimate_from_indices` accepted no-data windows and repeats; a budget under ten zeroed the least confident strata and reported 0.9% for a 7.4% map; a stratified sample accepted a repeated or foreign window; the design effect used the per-tile cap rather than the sample; `wilson_interval` accepted impossible counts. Tile grouping was O(tiles x windows), minutes on a 4-million-window map; now one sort.
- **The estimator's closed forms are derived and checked by enumeration**, not simulation: the stratified estimate is unbiased and its variance estimator unbiased for the exact variance (all 180 samples of a 12-unit population, rational arithmetic); Neyman's continuous allocation is the exact minimum and the integer rule's rounding cost is measured (0.1 to 0.5% at the record's budget); Wilson's coverage is a hypergeometric sum; the cluster-sample variance ratio is m S_b²/S² exactly; and the rate a review set gives is capture(b)·θ/b, which reproduces the guard's 2 to 6 times from exp70's recorded capture. `docs/method/protocol.md` §7–11, `tests/test_estimate_exact.py`.
- **Eight defects an audit of the new module found on a real GeoTIFF, before release.** A sample with no errors gave a zero-width stratified interval at nominal 95% (the shipped interval is now Wilson on the design's effective sample size, Korn and Graubard 1998; the Wald form exp78 graded is kept under its name) and, for a random sample, a lower bound above zero that ruled out a perfect map (the finite-population correction now multiplies the variance term only; exp78 regenerated, every cited digit unchanged). A stratum whose confidence saturates at exactly 1.0 was allocated the floor of two labels however large it was, and a nominal 95% interval covered 26% of draws on such a map; the design now assumes no stratum better than 98% right until labelled. The review-set guard at 0.75 let the confidence design's own sample through, which treated as random reports up to 1.9 times the true rate; the threshold is now four standard deviations above a random sample's median percentile, 0.64 at 300 labels. The tile design silently returned 64 labels for a budget of 300 on a small grid and would give a zero-width interval from one tile; both are refused. `wrong = 0.5` counted as right (`int(float)`), a BOM, a semicolon delimiter and a re-typed index column crashed with tracebacks, and the sidecar carried no CRS; all fixed, and the printed interval is its two ends rather than a symmetric "±".

## 1.1.3 (2026-09-21)

**Fixes six defects an adversarial audit found in the released package. Four of them made it return a plausible
wrong answer with exit 0, which is the failure mode this project exists to prevent in the maps it audits. Anyone
on 1.1.2 should upgrade.** The audit confirmed 21 findings in all; the ranking core held under three independent
lenses (review order, tie rules, budget arithmetic, the label bridge all bit-identical), and what was wrong was
the periphery: the second input, the no-data path, the name on an output file and the fitting layer's
self-evaluation. No recorded number changes; the demo's published figures are identical.

- **`suspicion.tif` held confidence, not suspicion.** Ranking it descending, which its name invites, returned the exact inverse of the review set: on a 256-window scene the top 13 of the raster overlapped the 13-window 5% review set in 0 of 13. An analyst opening it with a hot-is-bad colour ramp reviewed the windows the model was most confident about and skipped everything the tool flagged.
- **`assess --reference` pooled the reference over the prediction's class range**, so any reference class the map cannot predict collapsed to class 0 and the error rate came out understated, always in the flattering direction and with no warning: 0.0625 against an honest 0.1875 for a binary flood model graded against dry, flood and permanent water. The identical defect had been found, measured at 44.9% and fixed for `compare --labels`, and was never carried across.
- **A window with no prediction was given class 0**, so it disagreed with every neighbour and manufactured a prediction boundary around each cloud hole and scene edge: a map predicting one class everywhere it had a prediction reported a boundary window fraction of 16.7% and a boundary-first review set that was entirely the rim of the data. The denominator of eight is unchanged, so on a fully observed map the boundary is bit-identical to before and every recorded cue number stands.
- **Non-finite scores are treated as no-data and named in a warning.** NaN sorts to the front of the review order, so a scene with a NaN strip previously returned a review set made entirely of windows that contained no prediction, beside healthy-looking confidence quantiles.
- **A `(1, H, W)` score map is refused** with an instruction to pass `scores[0]`. It is the shape a binary head returns, and it was scored as a one-class map, inverting the meaning of the order.
- **`calibrate.fit_ranker` graded a sign-free fusion against a sign-locked baseline.** A user passing readings oriented "higher is safer", which the docstring invites, was shown a lead of 0.389 of excess AURC where the honest gap was 0.00025, with a sign test to match. The baseline now takes the better of its two orientations and the report records which. exp65's recorded result is unaffected: it passes its readings already oriented.
- **An unfittable cross-validation fold no longer reaches `held_out`.** Those rows sat at logit exactly 0.0 and were reported as a held-out evaluation, so on a map with few errors the report could say the ranker found 0% of them at a 10% budget while quoting a better-than-random excess AURC computed from that same fabricated vector. They are NaN now, excluded from every held-out number, and counted in `n_unscored_rows`.


- A false claim is corrected and the defect behind it fixed. exp54's cross-encoder block aligned each model's rows to OlmoEarth's by a hash of the label tile; identical tiles collide to one key, so where they are not adjacent the reorder permutes rows even when a model is aligned against itself, and the guard checked only that every key was found. The record said encoders share OlmoEarth's errors on binary water but not on multi-class (phi 0.05-0.08 on PASTIS); on the same 458,638 windows exp63 measures 0.50-0.68, and the two runs contradict each other on their face. The phi block is withdrawn, the claim is re-pointed at exp63 with a check that reads both artifacts, four documents are corrected, and the alignment now refuses to run unless its index is a permutation. exp54's preregistered result is computed before any alignment and is unaffected.
- Recorded from the committed artifacts, with no new run: the share of the ranking headroom the margin takes is a property of the task, not of the encoder. Over the suite's 16 encoders it runs 0.553 to 0.693; OlmoEarth Large is the best of them and Base ties for second; four model sizes move it by four points; and the spread between tasks is 2.8 times the spread within a task across all sixteen.

- exp77 recorded (job 1003587, 1 h 40 m on cpu): a confident error is more typical of its own scene, and removing that does not help. On the seven segmentation tasks of Ai2's suite, inside the confident half the error windows sit closer to their tile's mean token than the correct ones on 5 of 7 tasks (3 of 5 sources, negative on Sen1Floods11, undefined on MADOS), but the gap needs labels and has no control for class frequency, which predicts the same sign. Projecting that direction out of the frozen tokens loses accuracy on all seven, 0.03 to 0.98 points, so the diagnosis has nothing to locate; and the margin keeps its lead over every reading introduced. Four claims; the ledger stands at 161. No package code changes.

## 1.1.2 (2026-09-19)

- The README is short, in the shape of a tool's README: what the package is, the pipeline diagram, one finding, the documentation link, the demo and setup, 46 lines. The sections that argued the evidence live on Read the Docs only, and the package's page on PyPI shows this README from this release on.
- The demo answers the question its own picture raises. Most of the red lies outside the flagged windows, which reads as failure until one sees that the map is 19% wrong and the review is 5%: the run now prints, per review budget, the errors the tool would find, what a random review finds and the most any review could (at 5%: 17%, 5%, 26%; at 20%: 55%, 20%, 100%), and says so in one sentence. The README caption and the docs front page carry the same sentence.

## 1.1.1 (2026-09-19)

- `oe-inferencex demo` audits a real map. The package ships one tile of Dynamic World (a served global land-cover product this project had no hand in) with its published probabilities and the expert annotation of the same ground, reduced to 40 m windows, about 250 KB, CC BY 4.0 with its notice. The tile was chosen by a rule fixed before any tile was looked at, the lower-median tile by error capture among the 18 of 409 test tiles that are fully annotated, because the candidates' capture runs from 0.10 to 0.50 and a hand-picked tile could have said anything (`scripts/make_demo_sample.py`, cluster job 995922). `--made-up` keeps the synthetic water map.
- The demo's picture explains itself: three titled panels (what an audit gives with no labels used; the same windows over the map's real errors; a random pick of the same size over the same errors), a legend, and titles that carry the hit rates, drawn with a built-in 5 x 7 font so that nothing but numpy is needed. The random pick is reported by its expectation, not by the luck of one draw.
- The demo's words lead with what the tool is worth to a reviewer (how often a flagged window is really wrong, against a random one; how much of the map can be used as it is), then the reasons windows are flagged, then what the tool does not do, then the real `assess` command on the sample file before the user's own map.
- Two claims pin the demo's numbers in the ledger (157 claims).

## 1.1.0 (2026-09-18)

The first release on PyPI. What a user of 1.0.0 gains: a first run that needs no data (`oe-inferencex demo`), one
minus the top probability as a confidence form for multi-class maps (`form="top1"`), and refusals for inputs the
record does not support. The 1.0.0 defaults and every recorded number are unchanged. The source archive now carries
the package alone; the experiments, their outputs and the tests against them stay in the repository.

- `oe-inferencex demo`: a small made-up water map is audited through the same code path as `assess` and drawn with the standard library only, review set on the left, true errors on the right. Its error rate and capture are set to resemble a real flood map of the record, not to flatter, and the run says that it illustrates and does not prove.
- The README renders on PyPI: repository links and images are absolute.

- The document in `report/` (previously `paper/`) is titled a technical report and dated 17 September 2026; the README and the docs front page link to it.
- Removed 28 outdated files after an audit of every tracked file: an abandoned overview figure set, the re-targeting draft figures, leftover rasters from early drafts of the comparison figure, three rasters no diagram reads, the PDF build products of the four live diagrams, exp04's partial resume checkpoint (a bit-identical subset of `exp04_feats.npz`) and the figure of the superseded exp09 run. All remain in tag v1.0.0. The list, with what stays open, is `docs/plan/cleanup-audit-2026-09-17.md`.
- Removed 414 more files, the audit's second tier, at the author's choice: sixteen flat CSV copies of claim-pinned summaries, the exp73 and exp74 resume checkpoints (identical to their summaries), exp16's 66 MB token cache, the 2026-09-02 Word report and the four figures only it showed (exp21, exp23, exp24, exp25), two figures no page shows (exp05, summary_transfer), the workshop paper outline, the exp27 draft and four superseded cluster scripts. All remain in tag v1.0.0; nothing a claim, test or page reads was touched.

- exp75 recorded: heads fitted on the other sensors of the same units disagree in a way that beats every no-model control (5 of 5 groups) but does not find enough of the errors the margin misses to change a review set (preregistered P2 3 of 5, P3 0 of 5); a view helps only if its head can do the task, and averaging the heads improves accuracy on 1 of 5 groups. Four claims, no package change.

- Related work: 65 papers from 2024 to 2026 added across the six sections, each checked against its arXiv, Crossref or OpenAlex record, and a seventh section that tables every published challenge to a recorded finding against what the record answers, with the untested ones and their cost. One measured claim on the headroom: the margin takes a median 0.68 of the random-to-perfect ranking gap on the 24 tasks, and labels buy a fifth to a third of the rest.

- The agent skill (OlmoEarth-Agent pull request 155) is merged into that repository's main branch; the integration page and the report say so.

- Readiness for a test-time-training encoder (ViT3): `signals.crop_dependence` (how much of a decision map depends on the crop it was inferred in), `compare.determinism_check` (the same input inferred twice under two engines or precisions, gated against the reseed floor), `scripts/suite_regression.py` (exp70's protocol on any model directory of the published suite, with any candidate reading screened beside the margin and the controls), the compute budget per reading in the recipe, and the four predictions preregistered in `docs/plan/vit3_readiness.md`. No recorded number changes.

- exp76 recorded, and `form="top1"` added to `assess_prediction` and `signals.confidence`: one minus the top probability, tie-free from logits, the best member of the confidence family on 14 of 16 multi-class tasks; the default stays the 1.0.0 logit margin, which was the weakest form there and now warns on multi-class logit maps. AUGRC leaves the suite result standing (23 of 24 by AUROC); no whole-vector score and no window aggregator does better. Four claims.

- Refusals for inputs outside what the record supports, so that no caller gets a plausible review set for a map the package cannot rank. `assess_prediction` raises on probability input outside [0, 1] (the command line already refused the 2-D case; the API and the per-class case did not, so a regression raster imported by an agent was scored); a review set whose cut-off falls inside a run of equal scores carries `tied_at_cutoff` and a warning, which is what a hard mask, a quantized band or a constant map produce; `oe-inferencex compare` refuses a continuous map under the default cut-off, where two regression outputs "never differed", and takes it with `--threshold` named, noting that no recorded experiment grades that case. Usage opens with a table of what goes in, what the record supports for it, and what is refused. No recorded number changes.

## 1.0.0 (2026-09-17)

The first release with the evidence frozen behind it. Every function's behaviour is the one the recorded
experiments used; the numbers in the documentation were computed with this code.

**What the package does**

- `assess_prediction`, `assess_classmap`: a prediction map, probabilities or logits, binary or per-class, or a
  served class map with its confidence, into review sets at chosen budgets, in confidence order or boundary
  first, with operating points and the attainable ceiling; an optional reference raster scores them.
- `explain_review_set`: why each review window is suspect, as label-free cues with the enrichment measured for
  each on expert-labelled testbeds.
- `compare_inferences`: two inferences of one scene on identical windows, disagreement rate pooled and per
  group, where the differing windows sit, stability of the set, and with labels the cross-tab and which side
  was right.
- `fit_ranker`, `fit_side`: the label-fitted fusion and side rule, cross-fitted by group and locked to the model
  family they were fitted on.
- `metrics`, `stats`: tie-aware AURC and excess AURC, capture at a budget, design-weighted estimators, exact
  sign tests, cluster bootstraps.
- `oe-inferencex assess` and `oe-inferencex compare`: the same from the command line, rasters or `.npy` in,
  JSON, CSV and rasters out.

**Public surface.** Declared in `oe_inferencex.__all__`; the encoder-bound modules (`evidence`, `awf`, `data`,
`figstyle`) are installed with the `encoder` and `geo` extras and raise a plain instruction otherwise.

**Licence.** Apache-2.0, in `LICENSE`; the release is tagged `v1.0.0`.

**Evidence.** 65 preregistered experiments and 146 ledger claims, each pinned to a committed artifact by an
executable check; the summary is `docs/Findings.md` and the origin of each idea is `docs/related_work.md`.

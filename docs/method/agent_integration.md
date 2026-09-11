# Integration with the OlmoEarth Agent

**Division of labour.** This repo owns the *evidence*: experiments, the
ledger, and a small library of pure functions that turn a prediction into an
assessment. The OlmoEarth Agent owns the *conversation*: it wraps those
functions as tools and narrates their summaries. The agent never generates
evidence itself.

Index at [../TECHNIQUES.md](../TECHNIQUES.md); recipe at
[recipe.md](recipe.md).

## With labels: `calibrate`

When the caller holds labels for a model, `calibrate.fit_ranker` and
`calibrate.fit_side` fit the label-free readings to them and return a `Fusion`
plus a held-out report (cross-fitted by tile or event; excess AURC, capture,
calibration, or share right against the raw rule, always-a, always-b and the
coin, with a sign test over groups). Inputs: named per-window readings, the
decisions, validity, labels, group ids, and the model family as a string.
Outputs: the fusion (JSON-serialisable) and the report. Contract: the fusion is
bound to the family it was fitted on and raises when asked to score another,
because a rule fitted on frozen heads does not transfer to a fine-tuned model
(exp59); an agent must refit per model and report the held-out numbers, not the
in-sample ones.

## As a tool call

An agent that cannot import the package can call the two commands and read the
JSON back: `oe-inferencex assess <scores> --out <dir>` returns the review sets
and the explanation, `oe-inferencex compare <a> <b> --out <dir>` the difference
of two decisions on identical windows, both label-free unless a reference or
label raster is given. The files are the contract: `assessment.json` and
`explanation.json` for the first, `comparison.json` for the second, each with
an `inputs` block naming what was read and a `files` block naming what was
written. The Usage page lists every field.

## What this repo provides

| Module | Provides |
|---|---|
| `oe_inferencex.assess` | `assess_prediction` (logits or probabilities), `assess_classmap` (hard class map plus an exported confidence band), both with `order="confidence"` or `"boundary_first"` for the review sets (exp36), `summary` (JSON-safe view) |
| `oe_inferencex.taskcard` | What task, legend, goal and audit settings a fine-tuned model has |
| `oe_inferencex.lcc` | HTTP range reader for the served change rasters |
| `oe_inferencex.metrics` | Tie-aware AURC, excess AURC, error capture at a budget, selective accuracy, ECE; torch-free |
| `oe_inferencex.signals` | The supported signals and controls as pure functions: confidence, prediction-boundary indicator, aligned tile-phase, NDWI-gradient and the other pixel controls, the U+ combination |
| `oe_inferencex.stats` | Exact sign tests, the one-vote-per-cluster test, sign-flip permutation, block and cluster bootstraps |
| `oe_inferencex.explain` | Why a review window is suspect: a library of label-free cues with their measured share among error and correct windows (`CUES`), `explain_review_set` (which cues fire on each window of an assessment's review sets, co-occurrence, the windows no cue explains), `cue_enrichment` (the validation: share among errors against share among correct, cluster bootstrap) |
| `oe_inferencex.compare` | How two inferences of the same scene differ, on the windows both predicted: `disagreement` (the rate pooled and per group, and the mask), `where` (each cue's share among the disagreement windows against its share among the agreement windows, and the ratio), `stability` (pairwise phi of several disagreement sets), `over_groups` (a per-group statistic as a distribution: median, share of groups where the sign flips, one-sided exact sign test); with labels `crosstab` (the errors one side corrects, the errors it adds, the errors both make, their phi) and `which_side` (how often each side matches the label where they disagree); `compare_inferences` (all of these in one summary, label-free unless labels are given) |

**Rule:** pure functions; no network except the task-card resolvers and the
raster reader. Arrays are returned, never serialized into text. `tests/`
checks these modules against the numbers recorded in `exp/out` (`uv run
pytest`), and against the experiment modules when the encoder is installed.

## What the agent provides

| Component | Location |
|---|---|
| Tools `olmoearth_task_card`, `olmoearth_assess_prediction` | `src/olmoearth_agent/tools/inference_assessment.py` |
| Skill 18 `olmoearth-inference-assessment` | vendored skills submodule |
| Egress capability `inferencex-configs` | agent config |

**Rule:** summary statistics only cross the tool boundary. Per-window
rasters, the flagged-window GeoJSON and the summary JSON are written under
`exports/` and referenced by handle. The five recipe caveats are returned
verbatim.

The agent imports this package lazily and reports `{"available": false}`
with install instructions when it is missing, so the agent runs without it.

## What the tool does with a prediction

1. **Confidence per window** — negative logit margin, or the exported band
   for the production case, pooled to 4-px windows.
2. **Boundary fraction per window** of the model's own class map — the
   triage cue.
3. **Review sets** at 1, 5 and 10% budgets, most suspicious first, with the
   boundary share of each set.
4. **With a reference map** — error rate, tie-aware AURC of confidence and
   of boundary, oracle and random, error capture at each budget, and the
   reference caveat (exp18).

5. **Why each review window is suspect** — `explain_review_set` lists, per
   window, the label-free cues that fire (on a prediction boundary; among
   the least confident 20%; and any cue the caller derives from other
   inputs, such as tiling instability, NDWI ambiguity, JRC seasonality),
   each quoted with its measured enrichment and the experiment that
   measured it (exp37 on identical windows), plus the co-occurrence of cues
   in the set and the windows no cue explains.

6. **Compare two inferences of the same scene.** `compare_inferences`
   takes two hard decision maps of identical windows (probabilities
   thresholded, class scores argmaxed, before the call), the validity mask
   of the windows both predicted, and optionally an id per window (tile,
   event), a label map and label-free cues of the same shape. It returns
   the disagreement rate pooled and per group, each cue's enrichment on
   the disagreement windows against the agreement windows, and the
   disagreement mask among the arrays; `stability` gives the pairwise phi
   of several disagreement sets, the test that two runs flag the same
   windows. With labels it adds the graded block: the cross-tab of the two
   error maps (the errors one side corrects, the errors it adds, the
   errors both make, and their phi), which side matches the label on the
   disagreement windows, and per group the same cross-tab with
   `over_groups` of the net correction. Everything before the graded block
   is label-free.

Nothing is fused; nothing is learned. The narration says which signal ranked
the windows, which cues explain each one and with what evidence, and which
caveats apply. The explanation layer returns structured evidence; the
sentences are templates, and the low-confidence cue is the ranker itself,
so inside a confidence review set it always fires.

## Is it measured?

Not yet. Whether an agent given these tools produces better-grounded statements
than one given a sandbox is the preregistered benchmark in
[../plan/agent_benchmark.md](../plan/agent_benchmark.md) (exp64): forty cards
from the labelled testbeds, three tasks, a claims audit against the tool outputs,
the package's own numbers as the ceiling.

## Status (2026-09-02)

**Agent side.** Implemented and tested in the agent repo's working tree (15
tool tests; full suite 611 passing), not yet committed there. The SKILL.md
lives in the vendored skills submodule and must be committed upstream.

**Audit side — fixed after the integration review:**

- the oracle sign in the assessor (errors are the most suspicious windows);
- a torch-free home for the AURC metrics, so the agent need not install torch;
- no-prediction pixels no longer vote in pooled classes or boundaries;
- windows without reference are excluded from scoring rather than counted as
  errors;
- `summary()` gives a JSON-safe view;
- the LCC task card no longer claims the product ships a class confidence
  (exp20);
- `pyyaml` and `huggingface_hub` are declared.

**Resolved since.** `olmoearth-pretrain` was a hard dependency of the
package even though the assessment layer never imports it, which forced the
agent to install with `--no-deps` and broke `uv sync` on any clone without a
sibling `../olmoearth_pretrain` checkout. It now sits behind the `encoder`
extra, so `uv sync` installs `assess` / `metrics` / `taskcard` / `lcc` with
numpy, pyyaml and huggingface_hub alone — no torch, no `--no-deps`.

**Still open:**

- A production mode of the agent tool that takes a class map and a
  confidence band (`assess_classmap`) is written here but not yet wired in
  the agent.
- The task-card resolvers fetch with `urllib`, validated at the agent's tool
  boundary only.

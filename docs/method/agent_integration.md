# Integration with the OlmoEarth Agent

**Division of labour.** This repo owns the *evidence*: experiments, the
ledger, and a small library of pure functions that turn a prediction into an
assessment. The OlmoEarth Agent owns the *conversation*: it wraps those
functions as tools and narrates their summaries. The agent never generates
evidence itself.

Index at [../TECHNIQUES.md](../TECHNIQUES.md); recipe at
[recipe.md](recipe.md).

## What this repo provides

| Module | Provides |
|---|---|
| `oe_inferencex.assess` | `assess_prediction` (logits or probabilities), `assess_classmap` (hard class map plus an exported confidence band), `summary` (JSON-safe view) |
| `oe_inferencex.taskcard` | What task, legend, goal and audit settings a fine-tuned model has |
| `oe_inferencex.lcc` | HTTP range reader for the served change rasters |
| `oe_inferencex.metrics` | Tie-aware AURC, torch-free |

**Rule:** pure functions; no network except the task-card resolvers and the
raster reader. Arrays are returned, never serialized into text.

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

Nothing is fused; nothing is learned. The narration says which signal ranked
the windows and which caveats apply.

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

**Still open:**

- `olmoearth-pretrain` remains a declared dependency of the package though
  the assessment layer does not use it — only the data loaders (`awf.py`,
  `data.py`) do. The agent works around this by installing with `--no-deps`.
- A production mode of the agent tool that takes a class map and a
  confidence band (`assess_classmap`) is written here but not yet wired in
  the agent.
- The task-card resolvers fetch with `urllib`, validated at the agent's tool
  boundary only.

# olmoearth_inferenceX

**Examining OlmoEarth inference results without labels.** Given a prediction
map, which windows should a reviewer trust, which should they look at first,
and why. The protocol scores every candidate against the model's own
confidence and a no-model pixel control, on two references at once, and
grades on expert labels only.

![Overview: what is audited, what we believe is true, the test every claim passed, the open question, where it stands](docs/figures/inferencex_overview.png)

## What holds

- **The model's own confidence is the best label-free error ranker** on
  every expert-labelled testbed: AWF points, Sen1Floods11 hand labels, and
  the fine-tuned model run end to end (exp04, exp16, exp18, exp21).
- **Review boundary windows first, then by confidence.** Errors concentrate
  on prediction boundaries, 75% of errors against 21% of correct windows,
  and this order captures more of them than confidence alone at 5% and 10%
  review budgets on hand labels (preregistered, exp36). It costs nothing at
  inference.
- **Every flagged window comes with a reason.** 95% of the error windows on
  hand labels carry at least one label-free cue with a measured enrichment;
  spectral ambiguity is 7x enriched and the one cue that adds precision
  inside the review set (exp37, `oe_inferencex.explain`).
- **An accuracy needs a coverage.** The fine-tuned model is 0.93 accurate
  where it claims 0.99; keeping the 80% most confident windows gives 0.945
  (exp21).
- **The served product can be triaged without confidence.** It exports no
  class confidence, and boundary fraction alone captures a median 0.88 of
  the disagreements at a 5% review budget (exp20).

## The numbers

Sen1Floods11 Bolivia hand labels, 81,984 windows, 8.8% errors (exp36, exp37):

| Review budget | Errors caught, confidence | Errors caught, boundary first | Error rate inside the set |
|---|---|---|---|
| 5% | 0.259 | 0.274 | 0.38 |
| 10% | 0.465 | 0.494 | 0.33 |
| 20% | 0.749 | 0.732 | 0.26 |

Why a window is flagged, share among error windows against correct windows
on the same testbed (exp37):

| Cue | Errors / correct | Enrichment |
|---|---|---|
| on a prediction boundary | 0.750 / 0.214 | 3.5x |
| among the least confident 20% | 0.589 / 0.163 | 3.6x |
| unstable under a tiling shift | 0.583 / 0.164 | 3.6x |
| spectrally ambiguous, NDWI near zero | 0.483 / 0.067 | 7.2x |

**Open question.** Tiling instability wins 26 of 27 scenes against the
WorldCover reference and 8 of 8 rivers, but not on hand labels. The
decisive test needs adjudicated cells on the eight rivers (issue #2).

**Side product.** OlmoEarth v1's pretraining target has effective rank 2
(a frozen random projection); a normalised target is 57 to 70% predictable
from context (exp32 to exp34, issue #11).

## Install and reproduce

```bash
uv sync                                 # assessment and explanation layers, no torch
uv sync --extra encoder --extra geo     # full experiment environment
uv run pytest                           # the package against the recorded numbers
uv run python exp/exp02_full_slice.py   # one scene end to end
```

Experiments are `exp01`–`exp38` in [`exp/`](exp/), with outputs under
`exp/out/`. The supported machinery (confidence, boundary triage, tie-aware
metrics, the tests, the explanation layer) is the package
[`oe_inferencex/`](oe_inferencex/), torch-free and covered by `tests/`.
Torch is pinned per platform; Linux resolves the cu128 build.

## Documentation

| | |
|---|---|
| [Technique ledger](docs/TECHNIQUES.md) | Everything tried, one line each, including what was rejected and why |
| [Recipe](docs/method/recipe.md) · [Protocol](docs/method/protocol.md) | What to do and not do; how results are scored |
| [Comparisons](docs/results/comparisons.md) · [Signals](docs/results/signals.md) · [Explanation](docs/results/explanation.md) | Per-experiment, per-signal and per-cue evidence |
| [Task cards](docs/method/taskcards.md) · [Infrastructure](docs/method/infrastructure.md) | What each model is; upstream sources and formats |
| [Agent integration](docs/method/agent_integration.md) | Contract with the OlmoEarth Agent |
| [Roadmap](docs/plan/roadmap.md) · [Lab log](exp/NOTES.md) | Open items; chronology |

*Research code under active development; results are updated in place.*

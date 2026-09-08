olmoearth_inferenceX audits OlmoEarth inference results without labels.

Given a prediction map, it helps with:

1. Deciding which windows to trust and which to send for review first, as
   review sets at a chosen budget, in confidence order or boundary first.
2. Explaining why each flagged window is suspect, with label-free cues that
   carry measured evidence from expert-labelled testbeds.
3. Scoring any candidate audit rule the same way, against the model's own
   confidence and a no-model control, on two references at once.
4. Auditing deployed OlmoEarth products: the fine-tuned models through their
   task cards, and the served land cover change rasters.

Full documentation is in [docs/](docs/index.md).


Quickstart
----------

If you are new to the repository, we suggest starting here:

1. First, read [Findings](docs/Findings.md), which summarises what holds,
   the numbers, how a claim gets in, and the limits.
2. Second, read [Usage](docs/Usage.md), which walks through the package:
   assess a prediction, explain its review set, the production case, and
   how to score a new rule.
3. Finally, read the [Recipe](docs/method/recipe.md), the short list of what
   to do and not do when auditing a prediction map.

Other links:
- [TECHNIQUES](docs/TECHNIQUES.md) is the ledger of everything tried, one
  line each, with the verdict and the evidence.
- [Protocol](docs/method/protocol.md) documents how results are scored.
- [Explanation](docs/results/explanation.md), [Signals](docs/results/signals.md)
  and [Comparisons](docs/results/comparisons.md) hold the per-cue,
  per-signal and per-experiment evidence.
- [Agent integration](docs/method/agent_integration.md) documents the
  contract with the OlmoEarth Agent.
- [Roadmap](docs/plan/roadmap.md) lists the open items.


Setup
-----

The package needs Python 3.12 and no torch; the experiments need the
encoder.

```
git clone https://github.com/2imi9/olmoearth_inferenceX.git
cd olmoearth_inferenceX
uv sync
uv run pytest
```

For the full experiment environment:

```
uv sync --extra encoder --extra geo
uv run python exp/exp02_full_slice.py
```


Contact
-------

For questions and suggestions, please open an issue on GitHub.

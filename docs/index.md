# olmoearth_inferenceX documentation

Label-free auditing of OlmoEarth inference results: which windows of a
prediction map to trust, which to send for review first, and why.

Start here:

1. [Findings](Findings.md): what holds, the numbers, how a claim gets in, the limits.
2. [Usage](Usage.md): the package, a quick start, the production case, scoring a new rule.
3. [Recipe](method/recipe.md): what to do and not do when auditing a prediction map.

Reference:

- [Technique ledger](TECHNIQUES.md): everything tried, one line each, with the verdict and the evidence.
- [Protocol](method/protocol.md): how results are scored, evidence tiers, status terms, related work.
- [Explanation](results/explanation.md), [Signals](results/signals.md), [Comparisons](results/comparisons.md): per-cue, per-signal and per-experiment evidence.
- [Agent integration](method/agent_integration.md): the contract with the OlmoEarth Agent.
- [Task cards](method/taskcards.md), [Infrastructure](method/infrastructure.md): what each fine-tuned model is; upstream sources, export formats, the cluster pattern.
- [Roadmap](plan/roadmap.md): open items in priority order. [Lab log](../exp/NOTES.md): chronology, including superseded runs.

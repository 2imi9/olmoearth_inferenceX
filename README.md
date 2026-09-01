# olmoearth_inferenceX

Auditing OlmoEarth inference quality over regions with no ground-truth labels.

The idea: LLMs already have a working toolkit for judging outputs without a
reference answer (hallucination detection). Those techniques mostly do not
depend on language, so they port to earth observation, the same way the RoPE
fix in OlmoEarth v1.2 ported an LLM mechanism to fix the v1 striping artifact.

| Signal | LLM ancestor | OlmoEarth mechanism |
|---|---|---|
| E_case model disagreement | SelfCheckGPT / self-consistency | Nano/Tiny/Base prediction divergence |
| E_system perturbation stability | semantic entropy | tile-grid phase shifts |
| E_geo geographic grounding | retrieval-grounded fact checking | predictions vs reference river centerlines |
| E_dist embedding dissimilarity | internal-state probing | k-NN distance to training distribution |

Output posture: a ranked audit of suspect windows with risk-coverage curves,
not a single accuracy number. Every signal is scored against the max-softmax
confidence of the audited model, which is the baseline to beat.

## Headline so far

Confidence wins on in-domain class confusion (verified on AWF partner expert
labels). It collapses on ambiguous wetland margins and under domain shift,
where every evidence channel beats it: model disagreement 3x better on the
Barotse floodplain, embedding distance 18x better on the Zambezi delta
(weak-truth caveats documented in the ledger).

## Layout

- `docs/TECHNIQUES.md` - the technique ledger: standing results per technique,
  each claim tied to the experiment that supports it, gaps enumerated. Start here.
- `exp/` - experiments (exp01-exp05) and their figures in `exp/out/`.
- `exp/NOTES.md` - chronological lab log.
- `oe_inferencex/` - the library-in-progress: `evidence.py` is pure math
  (no network), `data.py` and `awf.py` are data access.
- `reports/` - point-in-time writeups.

## Run it

Everything runs on CPU with public data and public checkpoints; no special
infrastructure.

```
uv sync --extra geo
uv run python exp/exp02_full_slice.py
```

exp04 additionally needs the AWF dataset:
`huggingface.co/datasets/allenai/olmoearth_projects_awf` extracted to
`data/awf/dataset/`.

Status: experimental. A cleaned-up library repo comes later.

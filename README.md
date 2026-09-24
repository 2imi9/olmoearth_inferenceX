olmoearth_inferenceX is a Python package for assessing classification maps from
Earth-observation models. Without reference labels, it ranks a map's windows by the model's
confidence for manual review, lists the cues behind each flagged window, and compares two maps
of the same area. With a labelled sample, it estimates the error rate and per-class accuracy
with confidence intervals.

1. **Ranking.** Windows (4 px) are ranked by suspicion `s = −(p₍₁₎ − p₍₂₎)`, the negative
   difference between the model's two highest scores, boundary windows first. A ranking is
   scored by excess AURC, `AURC(s) − AURC(oracle)`, against two controls that do not use the
   model (class rarity, embedding distance).
2. **Comparison.** For two maps `A`, `B` on one window grid: `P(A ≠ B)`, its enrichment among
   the errors, and `P(confidence identifies the correct map | A ≠ B)`, each against the reseed
   floor. Across dates, a difference may reflect change on the ground; grading then requires
   the labels' date.
3. **Estimation.** `n` windows are labelled, stratified by margin with `n_h ∝ N_h √(q_h(1−q_h))`
   from the model's confidence; `θ̂ = Σ W_h p_h`, variance `Σ W_h²(1−f_h) p_h(1−p_h)/(n_h−1)`,
   Wilson interval at the effective sample size (exact hypergeometric under simple random
   sampling, cluster-corrected for labels collected by tile). A review set is refused as a
   sample: its error rate is `capture(b)·θ/b`, not `θ`. The same labels give per-class user's
   and producer's accuracy and error-adjusted shares; from a simple random sample, exact
   hypergeometric tests from the most confident windows outward give the largest zone with
   error rate at most `α` at error probability `δ`, given at least `ln δ / ln(1−α)` labels.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/pipeline.png" alt="One scene through the assessment: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the cues per flagged window" width="760">

On the 24 tasks of Ai2's published embedding suite where a confidence margin is defined, the
model's confidence ranks its errors better than the best control that does not use the model,
on every task. <!-- claim:suite-margin-wins-every-task -->

The documentation at **https://olmoearth-inferencex.readthedocs.io/** contains the
[findings](https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short), the [usage](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/), the
[recommended procedure](https://olmoearth-inferencex.readthedocs.io/en/latest/method/recipe/) and the record of each experiment.


Demo
----

```bash
pip install olmoearth-inferencex
oe-inferencex demo
```

The demo assesses a real land-cover map, one Dynamic World tile with expert annotation, selected by a rule fixed in advance as the median of 18 eligible tiles rather than the best: <!-- claim:demo-sample-is-the-median-tile -->

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/demo_real_map.png" alt="Three panels of a real land-cover map in southern Peru. Left: the 5% of windows ranked first, outlined in black along the class boundaries. Middle: the same windows over the map's errors in red. Right: a random 5% of windows over the same errors" width="760">

*Left: the 5% of windows ranked first, selected without labels. Middle: the same windows over the map's errors, in red; 67% of them are wrong, against 19% of the map. Right: a random 5%. The flagged windows hold 17% of the errors; with 19% of the map wrong and 5% reviewed, no selection of that size could hold more than 26%. A 20% review holds 55%.* <!-- claim:demo-sample-hit-rate -->

To assess another map: `oe-inferencex assess your_map.tif --out audit`.


Setup
-----

```bash
pip install olmoearth-inferencex        # add [geo] to read and write GeoTIFFs
```

To work on the repository, clone it, then run `uv sync` and `uv run pytest`; the experiments
also require `uv sync --extra encoder --extra geo`.

Apache License 2.0 ([LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE)); citation in [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff);
questions and suggestions as [GitHub issues](https://github.com/2imi9/olmoearth_inferenceX/issues).

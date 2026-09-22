olmoearth_inferenceX is a tool for auditing Earth-observation prediction maps without
labels. It ranks the windows of a map by the model's own confidence, so a reviewer knows
where to look first, says why each window is flagged, and measures how two inferences of
the same scene differ.

1. **Ranking.** Pool the map to windows (4 px). Suspicion `s = −(p₍₁₎ − p₍₂₎)`, the negative
   top-1-minus-top-2 margin of the model's own scores; boundary windows first, then by `s`. A
   ranking is scored by excess AURC, `AURC(s) − AURC(oracle)`, beside two controls that never
   see the model (class rarity, embedding distance).
2. **Comparing.** Two inferences `A`, `B` on one window grid: `P(A ≠ B)`, its enrichment among
   the errors, and `P(confidence picks the right side | A ≠ B)`, each against the reseed floor.
3. **Estimating.** Label `n` windows, stratified by margin with `n_h ∝ N_h √(q_h(1−q_h))` from
   the model's own confidence; `θ̂ = Σ W_h p_h`, 95% interval from `Σ W_h²(1−f_h) p_h(1−p_h)/(n_h−1)`;
   cluster-corrected when labels come tile by tile. A review set is refused as a sample: its rate
   is `capture(b)·θ/b`, not `θ`.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/pipeline.png" alt="One scene through the audit: Sentinel-2 bands, the frozen OlmoEarth encoder and the task head, the prediction, confidence and boundary layers, the review set at a 5% budget drawn on the scene, and the reasons per flagged window" width="760">

On the 24 tasks of Ai2's published embedding suite where a confidence margin is defined,
the model's own confidence ranks its errors better than any control that sees no model,
on every one. <!-- claim:suite-margin-wins-every-task -->

Full documentation is available at **https://olmoearth-inferencex.readthedocs.io/**:
the [findings](https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short), the [usage](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/), the
[recipe](https://olmoearth-inferencex.readthedocs.io/en/latest/method/recipe/) and the evidence experiment by experiment.


Demo
----

```bash
pip install olmoearth-inferencex
oe-inferencex demo
```

It audits a real map, one tile of Dynamic World land cover, chosen by a rule fixed in advance (the median tile of 18, not the best one), and draws this: <!-- claim:demo-sample-is-the-median-tile -->

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/demo_real_map.png" alt="Three panels of a real land-cover map in southern Peru. Left: the 5% of windows to check first, outlined in black along the class boundaries. Middle: the same windows over the map's real errors in red. Right: a random 5% of windows over the same errors" width="760">

*Left: the 5% of windows to check first, found without labels. Middle: the same windows over the real errors, in red: 67% of them are wrong, against 19% of windows picked at random. Right: a random 5%. Most of the red lies outside the flagged windows because the map is 19% wrong and the review is 5%: no 5% could hold more than 26% of the errors, and these hold 17%; a 20% review finds 55%.* <!-- claim:demo-sample-hit-rate -->

Your own map: `oe-inferencex assess your_map.tif --out audit`.


Setup
-----

```bash
pip install olmoearth-inferencex        # add [geo] to read and write GeoTIFFs
```

To work on the repository, clone it, then `uv sync` and `uv run pytest`; the experiments
also need `uv sync --extra encoder --extra geo`.

Apache License 2.0, see [LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE); to cite, see [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff).
For questions and suggestions, please
[open an issue on GitHub](https://github.com/2imi9/olmoearth_inferenceX/issues).

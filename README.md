olmoearth_inferenceX
====================

olmoearth_inferenceX is a post-inference check for Earth-observation maps: where a model's map
is likely wrong, how wrong it is, which part you can trust, and which of two maps is better. It
reads the scores the model wrote out; it never runs the model and needs no torch.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/architecture.png" alt="Structure of olmoearth-inferenceX. Left: Sentinel-2 imagery goes through a model's encoder and decoder to its outputs, a class map and a confidence map. Right: the outputs are pooled to windows. Without reference labels, ranking the windows by confidence gives the review set, and comparing two maps A and B gives the disagreement map. With a labelled sample, a sample of the windows where A and B differ gives their accuracy difference with a 95% interval; a random sample with reference labels gives the error rate with an exact 95% interval and, by exact zone tests, the certified zone" width="900">

Documentation: https://olmoearth-inferencex.readthedocs.io/


Install
-------

```bash
pip install "olmoearth-inferencex[geo]"    # Python 3.11 to 3.13; without [geo] it reads .npy only
oe-inferencex demo                          # a real Dynamic World tile; needs no data
```


Four questions
--------------

| Question | Commands | Labels |
|---|---|---|
| Where do I look first? | `assess scores.tif --out audit` | none |
| How wrong is the map? | `sample scores.tif --budget 300 --design random --out to_label.csv`, fill in `wrong`, then `estimate to_label.csv` | about 300 random windows |
| Which part can I trust? | `certify to_label.csv --alpha 0.05` | the same windows |
| Which of two maps is more accurate? | `sample a.tif --other b.tif --budget 100 --out pairs.csv`, fill in `reference_class`, then `estimate pairs.csv` | about 100, where the maps differ |
| Where do two maps differ? | `compare a.tif b.tif --out diff` | none |

Every command starts with `oe-inferencex`, and `--help` lists its options.
[Quick start](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#quick-start) runs
each one on a test map and shows what it prints.

**Input:** the model's per-class scores, taken before the argmax, as a GeoTIFF or `.npy` array of
shape `(C, H, W)`: probabilities between 0 and 1, or logits with `--logits`. A class map alone
cannot be ranked. The package works on windows of 4 x 4 pixels (`--patch`).

**Labels:** for each window `sample` drew, write 1 or 0 in `wrong` (or the class you see in
`reference_class`), and `?` where a window cannot be judged.


Before you trust it
-------------------

1. Without labels it says where to look, not how wrong the map is. A review set is not a
   sample, and `estimate` and `certify` refuse it.
2. A model run on inputs it was not trained on can be sure and wrong, and those errors are
   checked last. On PASTIS, a probe trained on radar plus optical and run on radar alone was
   74% wrong, and 60% of its errors were as confident as a typical correct window. <!-- claim:missing-optical-errors-are-confident -->
   Give `--condition` a raster of each pixel's input condition, such as a cloud flag, to treat
   each condition on its own.
3. Every interval assumes the labels are right. For reviewer error rates you can state,
   `estimate --reviewer-false-alarm` and `--reviewer-miss` widen it.
4. `certify` can return nothing, and then says why.


Use from an agent
-----------------

`oe-inferencex mcp` is a local MCP server; nothing is hosted. With
[uv](https://docs.astral.sh/uv/) installed, one line connects it to Claude Code:

```bash
claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp
```

Use a strong model. In a pilot, Claude Sonnet kept the tools' numbers and their limits; Claude
Haiku often dropped the limits. To try it, write the demo tile into an empty folder, start the
agent there and ask the question below:

```bash
uvx --from olmoearth-inferencex oe-inferencex demo
```

> Where should I look first in oe_inferencex_demo/sample_probabilities.npy? Use windows of
> 1 pixel, and grade the order against oe_inferencex_demo/sample_truth.npy.

[Usage](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#use-from-an-agent-mcp)
gives the configuration for other agents and more questions.


What has been measured
----------------------

- On the 24 tasks of Ai2's embedding suite, read through linear probes on OlmoEarth Base, a
  review of the least confident 10% found a median 0.214 of the errors, about twice a random
  10%. <!-- claim:suite-review-at-ten-percent -->
- On sixteen encoders under ten probe seeds, confidence ranked the errors better than a random
  order in all 3,560 cells, though the lowest AUROC, 0.503, is close to chance. <!-- claim:exp79-margin-beats-random-everywhere -->
- It also beat informative baselines on ground survey labels (LUCAS), Dynamic World's expert
  tiles and farmers' crop declarations (EuroCrops). <!-- claim:external-references-carry-informative-controls -->
- Over 2,514 pairs of probe maps, 100 labels drawn where two maps differ named the more
  accurate one on 60% of draws, against 13% for 100 labels from the whole map (exp90, not
  preregistered). <!-- claim:exp90-which-map-few-labels -->

- On two published land-cover products (LCMAP and Esri, 2018) graded against LCMAP's 25,000
  random reference plots, the error-rate interval held its coverage, and labels drawn where the
  maps differ never named the wrong one; at 50 to 200 labels they named LCMAP on 6% to 29% of
  draws (exp92, preregistered). <!-- claim:exp92-products-labelled-routes-hold -->

- On LCMAP's own confidence layer, certified zones met their guarantee in every graded cell. At
  300 labels and an allowed error of 8.9%, the default rule returned no zone on 87% of draws: the
  first zone it tests, a tenth of the plots, is 6% wrong and holds about 30 labels (exp93,
  preregistered). <!-- claim:exp93-zone-guarantee-holds-on-a-product -->

Most of this evidence is linear probes on frozen embeddings; on published products, one pair
and one year have been graded. The
[technical report](https://github.com/2imi9/olmoearth_inferenceX/blob/main/report/main.pdf) and
the [documentation](https://olmoearth-inferencex.readthedocs.io/en/latest/Summary/) give the
full record and its limits.


Development and license
-----------------------

To work on the repository, clone it, then run `uv sync` and `uv run pytest`. The experiments
also need `uv sync --extra encoder --extra geo`.

- License: Apache License 2.0 ([LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE)).
- Citation: [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff).
- Questions and suggestions: [GitHub issues](https://github.com/2imi9/olmoearth_inferenceX/issues).

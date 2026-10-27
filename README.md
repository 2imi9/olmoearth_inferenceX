olmoearth_inferenceX
====================

Checks an Earth-observation map after inference: where it is likely wrong, how wrong it is, which part you can
trust, and which of two maps is better.

It reads only the per-class scores a model writes out. It never runs the model and needs no torch, so it works on
OlmoEarth's outputs, other models' maps and published products with a confidence layer. Tested on maps made from
Sentinel-1, Sentinel-2 and Landsat. An agent can run it through its MCP server.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/architecture.png" alt="Structure of olmoearth-inferenceX. Left: Sentinel-2 imagery goes through a model's encoder and decoder to its outputs, a class map and a confidence map. Right: the outputs are pooled to windows. Without reference labels, ranking the windows by confidence gives the review set, and comparing two maps A and B gives the disagreement map. With a labelled sample, a sample of the windows where A and B differ names the more accurate map by an exact test and gives their accuracy difference with a 95% interval; a random sample with reference labels gives the error rate with an exact 95% interval and, by exact zone tests, the certified zone; a sample of each class's confident calls gives each class's error with a 95% interval and the classes not to trust" width="100%">

Documentation: https://olmoearth-inferencex.readthedocs.io/


Install
-------

```bash
pip install "olmoearth-inferencex[geo]"    # Python 3.11 to 3.13; without [geo] it reads .npy only
oe-inferencex demo                          # a real Dynamic World tile; needs no data
```

From an OlmoEarth model run with rslearn
([details](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#from-an-olmoearth-model-run-with-rslearn)):
set `output_probs: true` in the segmentation task, run `rslearn model predict` with the fine-tuned checkpoint,
then

```bash
oe-inferencex from-olmoearth path/to/dataset --out run    # prints the assess command that reads it
```


What it answers
---------------

| Question | Commands | Labels |
|---|---|---|
| Where do I look first? | `assess scores.tif --out audit` | none |
| How wrong is the map? | `sample scores.tif --budget 400 --design random --out to_label.csv`, fill in `wrong`, then `estimate to_label.csv` | about 400 random windows |
| Which part can I trust? | `certify to_label.csv --alpha 0.05` | the same windows |
| Which class does the map draw wrong? | add a `reference_class` column (the class you see), then `estimate to_label.csv --per-class` | the same windows for a large class; more for a rare one (`plan --class-share`) |
| Which of two maps of the same time is more accurate? | `sample a.tif --other b.tif --budget 100 --out pairs.csv`, fill in `reference_class`, then `estimate pairs.csv` | about 100, where the maps differ |
| Which parts of the map are good, and which bad? | `sample scores.tif --design profile --profile-by confidence:2,grid:2 --budget 900 --out to_label.csv`, fill in `wrong`, then `certify to_label.csv --alpha 0.05` | about 60 to 110 per part with no errors (`plan --profiles`) |
| Which class's confident calls can't I trust? | `sample scores.tif --design audit --per-class 20 --out audit.csv`, fill in `reference_class`, then `estimate audit.csv --alpha 0.05` | about 20 per class |
| Where do two maps differ? | `compare a.tif b.tif --out diff` | none |
| Can someone else check which part I certified? | sample with `--seed-from-map --alpha 0.05`; `certificate to_label_zone.json --date 2024-01-01/2024-12-31 --out cert`, then `verify cert/certificate.json --scores map.tif` | none beyond `certify`'s |
| At which window size can each class be seen? | `scale scores.tif` (add `--reference truth.tif` for each size's error rate) | none |
| How accurate is a map by an agency's published sample (NLCD, GFC2020)? | `grade points.csv --stratum S --weight W --reference R --map M` (`--map` twice for two maps) | none: the agency's labelled points |

Every command starts with `oe-inferencex`. The input is the model's per-class scores before the argmax, a GeoTIFF or
a `.npy` of shape `(C, H, W)`; `grade` reads a CSV of labelled points. `plan` says how many labels a question needs,
and `--help` lists each command's options. The
[quick start](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#quick-start) runs each one on a test map.


Use from an agent
-----------------

`oe-inferencex mcp` is a local MCP server; nothing is hosted. It gives an agent most of the steps above as tools:
`assess`, `compare`, `sample`, `estimate`, `certify`, `plan`, `decide` and `verify`, with `guide` to read them all.

- Each reply is JSON: the files written, the summary numbers, and three texts to quote: the conclusion, its limits
  and the next step.
- `decide` turns a result into a typed answer: yes, no or undetermined; map a, map b or undetermined; or a share.
- The server's instructions state the rules an agent must keep, such as: a review set is not a sample.
- Command line only: `grade`, `scale`, `certificate`, `from-olmoearth`, the profile and audit designs,
  `estimate --valid`, and `plan --class-share` and `--profiles`.

Add it to any MCP client, for example Claude Code:

```bash
claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp
```

Use a capable model; smaller ones dropped the limits the tools state. To try it, write the demo tile into an empty
folder, start the agent there and ask:

```bash
uvx --from olmoearth-inferencex oe-inferencex demo
```

> Where should I look first in oe_inferencex_demo/sample_probabilities.npy? Use windows of
> 1 pixel, and grade the order against oe_inferencex_demo/sample_truth.npy.


Examples
--------

**Where do I look first?** [Dynamic World](https://dynamicworld.app), a land-cover map made from Sentinel-2, on one tile in southern Peru. No
labels are used to pick the windows; the expert's labels only show how it did.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/example_peru.png" alt="Southern Peru, 5 km across: the Sentinel-2 image of 9 June 2019, the Dynamic World map, the 5% of windows flagged first without labels, and the same windows over the map's real errors, 67% of them wrong" width="100%">

The 5% it flags first are wrong 67% of the time. A random 5% would be wrong 19% of the time.

**Where do two maps differ?** A flood at Lake Shkodër, Albania, in March 2018 ([Copernicus EMS activation EMSR273](https://rapidmapping.emergency.copernicus.eu/EMSR273)), seen by Sentinel-2 and Sentinel-1.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/example_flood_context.png" alt="Lake Shkoder, Albania, 8 km across: Sentinel-2 before the flood (2 April 2017), Sentinel-2 during it (28 March 2018) and Sentinel-1 radar during it (11 March 2018), with an orange box on the flooded shore" width="100%">

Inside the orange box, water mapped from each sensor before and after (imagery from [GEOID-Flood](https://huggingface.co/datasets/links-ads/geoid-flood) and [WorldFloods v2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2)); orange squares are where two maps disagree:

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/compare.png" alt="The boxed patch: Sentinel-2 and Sentinel-1 images before and after the flood, the four water maps made from them, and orange squares where two of them disagree" width="100%">

**Which of two maps is more accurate?** [Fields of the World](https://fieldsofthe.world)'s global field map (FTW, 2nd edition) beside a second
model, PRUE B3, on a test chip of the [FTW benchmark](https://arxiv.org/abs/2409.16252) in Croatia.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/example_ftw.png" alt="One FTW benchmark test chip in Croatia, about 1.5 km by 1.1 km: the Sentinel-2 image, FTW's 2nd-edition field map, the PRUE B3 field map, and the windows where the two maps differ, found without labels" width="100%">

Across Croatia's 353 test chips the two maps differ at 13% of windows. There FTW agrees with the benchmark 74% of the
time, and labelling 100 of those windows names FTW the better map 99.9% of the time. In Belgium and Sweden, B3 was the
better one.

Imagery: contains modified Copernicus Sentinel data (2017 to 2023). Dynamic World map and expert labels:
[Brown et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01307-4), CC BY 4.0. FTW map:
[Fields of the World](https://fieldsofthe.world), CC BY 4.0. How the
figures were made is in [docs/figures/examples](https://github.com/2imi9/olmoearth_inferenceX/tree/main/docs/figures/examples).

What is guaranteed
------------------

With windows drawn by `sample` and labelled by a reviewer:

- **Error rate** (`estimate`, on `--design random`): an exact 95% interval. The confidence, proportional and tiles
  designs give approximate ones; `--valid` adds one valid at every sample size, on every design but tiles and audit.
- **Certified zone** (`certify`, on `--design random`): the most confident share of the map is wrong at most α of the
  time, except on at most δ of samples (10% by default). δ also counts the samples that certify nothing, so among
  those that certify a zone, more than δ can be wrong. On `--design sequential`, certify can be rerun as labels are
  added, with the same α and δ.
- **Per profile** (`certify`, on `--design profile`): parts of the map fixed from the map alone before any label
  (confidence band, input condition, class, grid block) are each called good (wrong at most α), bad (more than α) or
  undecided; all the verdicts are right together except on at most δ of samples.
- **Per class** (`estimate --per-class`, on `--design random` or `class`): an exact interval for each class's user's
  accuracy, each at 95%, and for every class's share and producer's accuracy, holding together at 95%. With
  `--valid` (every design but tiles and audit), each of these gets a 95% interval valid at every sample size, each on its own.
- **Two maps** (`sample --other`, then `estimate`): a 95% interval on their difference, and an exact sign test that
  names a map, the less accurate one at most 2.5% of the time, while the labels are at most a fifth of the windows
  where they differ.
- **A certificate** (`certificate`, then `verify`): anyone with the map, the seed and the labels can rebuild the
  certified zone and compare. It trusts the issuer on the labels, the map's date and drawing the sample once.

With an agency's own stratified sample (`grade`): overall accuracy, each class's user's and producer's accuracy and
share of the reference, and the difference of two maps, each with a 95% interval valid at any sample size, beside the
agency's nominal normal-theory one.

Each holds when the settings (α, δ, the budget, the rule and level cut, `--classes`) were fixed before any label was
read. The tests check each statement by exact computation, and the
[research record](https://github.com/2imi9/olmoearth_inferenceX-record) checks them on fully labelled real maps.
Details: [sample and estimate](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#sample-and-estimate),
[certify](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#certify).

Limits
------

- Without labels it tells you where to look, not how wrong the map is.
- In a new place or year the ranking can weaken. Check it with a few labels.
- When two legends draw a class boundary differently (grass or shrub, say), only labels find those errors.
- Intervals and certified zones assume the labels are right.
- If parts of the map were made from different inputs (clouds, a missing sensor), use `--condition`.
- Labelling more and rerunning counts as a new test, except certify on a sequential sample.


More
----

- Every result, its limits and the technical report: the [research record](https://github.com/2imi9/olmoearth_inferenceX-record).
- Development: clone the repository, then `uv sync` and `uv run pytest`.
- Apache License 2.0 ([LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE)); cite with
  [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff); questions in
  [GitHub issues](https://github.com/2imi9/olmoearth_inferenceX/issues).

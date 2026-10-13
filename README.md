olmoearth_inferenceX
====================

olmoearth_inferenceX checks an Earth-observation map after inference: where it is likely wrong, how wrong it is,
which part you can trust, and which of two maps is more accurate. It was built to check OlmoEarth's inference
outputs. It reads only the per-class scores a model writes out, never runs the model and needs no torch, so it works
as well on other models' maps and on published products that ship a confidence layer.

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/architecture.png" alt="Structure of olmoearth-inferenceX. Left: Sentinel-2 imagery goes through a model's encoder and decoder to its outputs, a class map and a confidence map. Right: the outputs are pooled to windows. Without reference labels, ranking the windows by confidence gives the review set, and comparing two maps A and B gives the disagreement map. With a labelled sample, a sample of the windows where A and B differ names the more accurate map by an exact test and gives their accuracy difference with a 95% interval; a random sample with reference labels gives the error rate with an exact 95% interval and, by exact zone tests, the certified zone; a sample of each class's confident calls gives each class's error with a 95% interval and the classes not to trust" width="100%">

Documentation: https://olmoearth-inferencex.readthedocs.io/


Install
-------

```bash
pip install "olmoearth-inferencex[geo]"    # Python 3.11 to 3.13; without [geo] it reads .npy only
oe-inferencex demo                          # a real Dynamic World tile; needs no data
```


Usage
-----

| Question | Commands | Labels |
|---|---|---|
| Where do I look first? | `assess scores.tif --out audit` | none |
| How wrong is the map? | `sample scores.tif --budget 300 --design random --out to_label.csv`, fill in `wrong`, then `estimate to_label.csv` | about 300 random windows |
| Which part can I trust? | `certify to_label.csv --alpha 0.05` | the same windows |
| Which of two maps is more accurate? | `sample a.tif --other b.tif --budget 100 --out pairs.csv`, fill in `reference_class`, then `estimate pairs.csv` | about 100, where the maps differ |
| Which class's confident calls can't I trust? | `sample scores.tif --design audit --per-class 20 --out audit.csv`, fill in `reference_class`, then `estimate audit.csv --alpha 0.05` | about 20 per class |
| Where do two maps differ? | `compare a.tif b.tif --out diff` | none |

Every command starts with `oe-inferencex`. The input is the model's per-class scores before the argmax, as a GeoTIFF
or a `.npy` array of shape `(C, H, W)`. `plan` says how many labels a question needs, `decide` turns a result into
yes, no or undetermined, and `--help` lists each command's options.
[Quick start](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/#quick-start) runs each one on a test map.

Limits: without labels it says where to look, not how wrong the map is. Ranking by confidence can weaken when a model
moves to a new place or year, and errors that come from two legends drawing a class boundary differently cannot be told
from the imagery or the model, so they need labels. A model run on inputs unlike its training data can be confident
and wrong (`--condition` treats each input condition on its own). Every interval and certified zone assumes the labels
are right, and labelling more and rerunning is a second test, except on a sequential sample.

Use from an agent
-----------------

`oe-inferencex mcp` is a local MCP server; nothing is hosted. Add it to any MCP client as the command
`uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp`, for example in Claude Code:

```bash
claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp
```

Use a capable model: in a pilot, smaller models dropped the limits the tools state. To try it, write the demo tile
into an empty folder, start the agent there and ask:

```bash
uvx --from olmoearth-inferencex oe-inferencex demo
```

> Where should I look first in oe_inferencex_demo/sample_probabilities.npy? Use windows of
> 1 pixel, and grade the order against oe_inferencex_demo/sample_truth.npy.

Evidence
--------

Tested on linear probes of sixteen encoders over the tasks of Ai2's embedding suite, on seven published land-cover
products against six reference samples, and on two deployed OlmoEarth models. Within the data a model was trained
for, its confidence ranks the map's errors well; after a move to a new place and year it ranked them below the
preregistered bar, and most of its confident errors lay on one class boundary that the two legends draw differently.
The [technical report](https://github.com/2imi9/olmoearth_inferenceX/blob/main/report/main.pdf) and the
[documentation](https://olmoearth-inferencex.readthedocs.io/en/latest/Summary/) give every result and its limits.

Development and license
-----------------------

Clone the repository, then `uv sync` and `uv run pytest`; the experiments also need `uv sync --extra encoder --extra geo`.
Apache License 2.0 ([LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE));
cite with [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff);
questions in [GitHub issues](https://github.com/2imi9/olmoearth_inferenceX/issues).

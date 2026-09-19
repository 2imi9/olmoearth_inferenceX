olmoearth_inferenceX is a tool for auditing Earth-observation prediction maps without
labels.

olmoearth_inferenceX helps with:

1. Deciding which windows of a prediction map to review first, at a chosen review
   budget, from the model's own confidence.
2. Explaining why each flagged window is suspect, with label-free cues whose evidence
   was measured on expert-labelled maps.
3. Measuring how two inferences of the same scene differ: across crops, backbones,
   sensors, encoders, fine-tuning and acquisition dates.
4. Scoring a new audit rule the same way, against the model's confidence and a control
   that sees no model.
5. Fusing those readings with labels where labels exist, reported held-out.

It was built around [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) and reads
any model's probabilities or logits. The main finding: on the 24 tasks of Ai2's published
embedding suite where a confidence margin is defined, the model's own confidence ranks
its errors better than any control that sees no model, on every one. <!-- claim:suite-margin-wins-every-task -->

Full documentation is available at **https://olmoearth-inferencex.readthedocs.io/**.


Demo
----

```bash
pip install olmoearth-inferencex
oe-inferencex demo
```

It audits a real map, one tile of Dynamic World land cover, chosen by a rule fixed in advance (the median tile of 18, not the best one), and draws this: <!-- claim:demo-sample-is-the-median-tile -->

<img src="https://raw.githubusercontent.com/2imi9/olmoearth_inferenceX/main/docs/figures/demo_real_map.png" alt="Three panels of a real land-cover map in southern Peru. Left: the 5% of windows to check first, outlined in black along the class boundaries. Middle: the same windows over the map's real errors in red. Right: a random 5% of windows over the same errors" width="760">

*Left: the 5% of windows to check first, found without labels. Middle: the same windows over the real errors, in red: 67% of them are wrong, against 19% of windows picked at random. Right: a random 5%. Most of the red lies outside the flagged windows because the map is 19% wrong and the review is 5%: no 5% could hold more than 26% of the errors, and these hold 17%; a 20% review finds 55%.* <!-- claim:demo-sample-hit-rate -->

Your own map: `oe-inferencex assess your_map.tif --out audit`. What the project found, in six sentences:
[Findings, in short](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Findings.md#in-short).


Quickstart
----------

If you are new to olmoearth_inferenceX, we suggest starting here:

1. First, read [Findings, in short](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Findings.md#in-short): six sentences on
   what the project found, with the full argument below them.
2. Second, read [Usage](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/Usage.md), which covers assessing a map and comparing
   two inferences, from the command line or from Python.
3. Finally, read the [Recipe](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/recipe.md): what to do and not do when
   auditing a map.

Other links:
- [Technical report](https://github.com/2imi9/olmoearth_inferenceX/blob/main/report/main.pdf): the whole record in ten pages.
- [Comparisons](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/results/comparisons.md) holds the evidence experiment by
  experiment, and [TECHNIQUES](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/TECHNIQUES.md) lists everything tried, one line
  each, with the verdict.
- [Protocol](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/protocol.md) documents how a result is scored, and the
  [claim ledger](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/claims.md) how every documented number is pinned to a
  file.
- [Agent integration](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/method/agent_integration.md) documents the tools the
  OlmoEarth Agent calls.
- [Related work](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/related_work.md), [Roadmap](https://github.com/2imi9/olmoearth_inferenceX/blob/main/docs/plan/roadmap.md) and
  [Changelog](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CHANGELOG.md).


Setup
-----

olmoearth_inferenceX requires Python 3.11+ and numpy.

```bash
pip install olmoearth-inferencex        # add [geo] to read and write GeoTIFFs
```

To work on the repository, with the tests that recompute the recorded numbers:

```bash
git clone https://github.com/2imi9/olmoearth_inferenceX.git
cd olmoearth_inferenceX
uv sync
uv run pytest
```

The experiments need the encoder as well: `uv sync --extra encoder --extra geo`.


Licence and citation
--------------------

Apache License 2.0; see [LICENSE](https://github.com/2imi9/olmoearth_inferenceX/blob/main/LICENSE). To cite the software or its recorded
results, see [CITATION.cff](https://github.com/2imi9/olmoearth_inferenceX/blob/main/CITATION.cff).


Contact
-------

For questions and suggestions, please
[open an issue on GitHub](https://github.com/2imi9/olmoearth_inferenceX/issues).

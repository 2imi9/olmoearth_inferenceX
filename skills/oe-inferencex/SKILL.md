---
name: oe-inferencex
description: Check a classification map made by an Earth-observation model with the olmoearth-inferencex package (command oe-inferencex), from the per-class scores the model wrote out. It answers four questions. Where should I look first? (assess) How wrong is the map? (sample, then estimate on the labelled sample) Which part can I trust? (certify) Which of two maps is better, and where do they differ? (compare; which is better only with labels) Use when a user has a model's score raster (GeoTIFF or .npy) or two maps of one area, and wants a review list, an error rate or a trusted zone.
---

# oe-inferencex

If the oe-inferencex MCP server is connected (tools `assess`, `compare`, `sample`, `estimate`, `certify`, `guide`),
call the tools: each returns a `conclusion`, its `limits` and what can be done `next`. Otherwise run the commands
below.

Install: the MCP server and `--condition` need 1.4.0 or later. With uv, one line connects the server to Claude
Code, with nothing else to install:
`claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp`.
To install the commands and the server instead: `pip install "olmoearth-inferencex[geo,mcp]"`; without the server,
`pip install "olmoearth-inferencex[geo]"` gives the commands.

No map yet? `oe-inferencex demo` (or `uvx --from olmoearth-inferencex oe-inferencex demo`) writes a real land-cover
tile with expert labels. Assess `oe_inferencex_demo/sample_probabilities.npy` with windows of 1 pixel (`--patch 1`,
`patch=1`) and grade the order against `oe_inferencex_demo/sample_truth.npy` (`--reference`, `reference=`).

## Input

The scores the model wrote out before the argmax: a GeoTIFF or `.npy` of shape `(C, H, W)`, one band per class, or
`(H, W)` for two classes. Probabilities from 0 to 1; logits with `--logits`. The package works on windows: square
blocks of `--patch` pixels, 4 by default.

## The questions, in the standard order

Match the user's question to a step. The tool and the command of each step have the same name.

1. **Where should I look first?** `assess`. No labels. The review set is the least confident share of the
   windows. `oe-inferencex assess scores.tif --out audit` writes `audit/review_set_05pct.csv` and the 1% and 10%
   sets.
2. **How wrong is the map?** `sample`, label, `estimate`.
   `oe-inferencex sample scores.tif --budget 300 --design random --out to_label.csv`. The user or a reviewer
   fills `wrong` with 1 or 0 on every row, or `?` where a window cannot be judged, keeping the order and
   `to_label.json` beside it. Then `oe-inferencex estimate to_label.csv` gives the error rate with a 95%
   interval; windows marked `?` make it a range. If the user knows how often the reviewer errs, add
   `--reviewer-false-alarm E0 --reviewer-miss E1` (shares of the correct and of the wrong windows).
3. **Which part can I trust?** `certify`. `oe-inferencex certify to_label.csv --alpha 0.05`: the most confident
   share of the map whose error rate is at most alpha. It may certify nothing, and says why.
4. **Which of two maps is better, and where do they differ?** `compare`.
   `oe-inferencex compare a.tif b.tif --out diff` says where they differ. Only with labels does it say which map
   is right there: add `--labels truth.tif`, and `--date-a`, `--date-b` when the maps describe dates. With few
   labels: `oe-inferencex sample a.tif --other b.tif --budget 100 --out pairs.csv` draws windows only where the
   maps differ; the reviewer writes the class seen in `reference_class` (or `?`); `oe-inferencex estimate
   pairs.csv` says which map is more accurate and by how much.
5. **Per condition.** When a raster records each pixel's input condition (a cloud flag, the modalities present, a
   sensor id), pass `--condition layer.tif` to `assess` and `sample`. `estimate` and `certify` then give each
   condition its own rate and zone.

## Example questions

Questions a user may ask, here on the README's test map (`scores.tif`, `other.tif` and `truth.tif`),
each with the tools it uses and what the answer can and cannot be:

- "Where should I look first in scores.tif?" Uses `assess`. The answer is a list of windows to check first, least
  confident first. It is not an error rate, and the errors the model is sure of come last. It needs the model's
  per-class scores: a class map alone is refused, or gives an order that is not evidence.
- "How wrong is scores.tif? Pick 300 windows at random for me to label." Uses `sample`, then `estimate`. In between,
  you set `wrong` to 1 or 0 on every row of the sample, or `?` where a window cannot be judged. The answer is an error rate with a 95% interval. No tool
  labels a window, and the interval assumes your labels are right.
- "Which part of scores.tif can I trust at 5% error?" Uses `certify`, on the labels of that random sample. The answer
  is the most confident share of the map that is wrong at most 5% of the time, a statement that fails on at most 10%
  of samples. It can be nothing: with too few labels, or errors among the most confident windows, it certifies no
  zone and says why.
- "Which is better, scores.tif or other.tif, and where do they differ?" Uses `compare`. The answer says where the two
  maps differ, window by window. Without labels it cannot say which map is better; with truth.tif as labels it says
  which is right where they differ. With labels on a few of the differing windows instead, sample with other.tif as the
  second map and then estimate say which map is more accurate.

## Hard rules

Do not work around them. The package refuses only what a rule says it refuses; the rest is up to you.

- A review set is not a sample. It is chosen to hold errors, so its error rate overstates the map's. estimate and
  certify refuse it.
- certify needs a random sample: draw it with design "random", or with a condition layer. The default design
  serves estimate only; certify refuses it.
- Without labels, compare cannot say which map is right. Two maps that agree can both be wrong.
- Labels are assumed right. The interval and the zone describe agreement with the reviewer's labels; only estimate
  can widen its interval for a reviewer who errs, at rates the user states.
- Ranking needs the scores, not only the class map. A class map alone works only in compare. assess refuses a
  class map of more than two classes read as probabilities, but not a 0/1 map, nor any class map passed with
  logits=true: it reads the class ids as scores, and the order it gives is not evidence.

On the command line, logits=true is `--logits`.

## How to report

- State as fact only what a command or tool returned, with its limits.
- Give an interval as its two ends. Never recompute it as p +/- 1.96 sqrt(p(1-p)/n).
- A window's confidence ranks windows. It is not the probability that the window is wrong.
- Nothing here labels windows, fetches labels or runs a model. Labels come from the user or a reviewer.
- A refusal carries the package's reason. Change the input it names; do not retry the same call.

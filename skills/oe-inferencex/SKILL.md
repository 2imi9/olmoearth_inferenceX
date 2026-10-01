---
name: oe-inferencex
description: Check a classification map made by an Earth-observation model with the olmoearth-inferencex package (command oe-inferencex), from the per-class scores the model wrote out. Which windows to review first (assess), how wrong the map is from a labelled sample (sample, then estimate), which part can be trusted (certify), and where two maps of one area differ (compare). Use when a user has a model's score raster (GeoTIFF or .npy) or two maps of one area, and wants a review list, an error rate or a trusted zone.
---

# oe-inferencex

If the oe-inferencex MCP server is connected (tools `assess`, `compare`, `sample`, `estimate`, `certify`, `guide`),
call the tools: each returns a `conclusion`, its `limits` and what can be done `next`. Otherwise run the commands
below. Install: `pip install "olmoearth-inferencex[geo]"`, with `[geo,mcp]` for the server.

## Input

The scores the model wrote out before the argmax: a GeoTIFF or `.npy` of shape `(C, H, W)`, one band per class, or
`(H, W)` for two classes. Probabilities from 0 to 1; logits with `--logits`. The package works on windows: square
blocks of `--patch` pixels, 4 by default.

## The standard order

1. **Where to look: assess.** No labels. The review set is the least confident share of the windows.
   `oe-inferencex assess scores.tif --out audit` writes `audit/review_set_05pct.csv` and the 1% and 10% sets.
2. **How wrong: sample, label, estimate.**
   `oe-inferencex sample scores.tif --budget 300 --design random --out to_label.csv`. The user or a reviewer
   fills `wrong` with 1 or 0 on every row, keeping the order and `to_label.json` beside it. Then
   `oe-inferencex estimate to_label.csv` gives the error rate with a 95% interval.
3. **Which part to trust: certify.** `oe-inferencex certify to_label.csv --alpha 0.05`: the most confident share
   of the map whose error rate is at most alpha. It may certify nothing, and says why.
4. **Two maps: compare.** `oe-inferencex compare a.tif b.tif --out diff`; add `--labels truth.tif` to see which
   map is right where they differ, and `--date-a`, `--date-b` when the maps describe dates.
5. **Per condition.** When a raster records each pixel's input condition (a cloud flag, the modalities present, a
   sensor id), pass `--condition layer.tif` to `assess` and `sample`. `estimate` and `certify` then give each
   condition its own rate and zone.

## Hard rules

The package enforces them; do not work around them.

- A review set is not a sample. It is chosen to hold errors, so its error rate overstates the map's. estimate and
  certify refuse it.
- certify needs a random sample: draw it with design "random", or with a condition layer. The default design
  serves estimate only.
- Without labels, compare cannot say which map is right. Two maps that agree can both be wrong.
- Labels are assumed right. The interval and the zone describe agreement with the reviewer's labels.
- Ranking needs the scores, not only the class map. A class map alone works only in compare.

## How to report

- State as fact only what a command or tool returned, with its limits.
- Give an interval as its two ends. Never recompute it as p +/- 1.96 sqrt(p(1-p)/n).
- A window's confidence ranks windows. It is not the probability that the window is wrong.
- Nothing here labels windows, fetches labels or runs a model. Labels come from the user or a reviewer.
- A refusal carries the package's reason. Change the input it names; do not retry the same call.

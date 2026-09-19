"""`oe-inferencex demo`: a first run that needs no data, no labels and nothing but numpy.

It makes a small synthetic water map the way a model would (sure in the middle of the water and the land, unsure along
the shore, fooled by one cloud shadow), audits it through the same code path as `oe-inferencex assess`, and draws the
result. Because the scene is made up its truth is known, so the run can say how many of the map's errors the review
set holds. That number illustrates the tool; it is not evidence. The evidence is in the documentation.
"""
import argparse
import contextlib
import io
import json
import os
import struct
import zlib

import numpy as np

from oe_inferencex.assess import assess_prediction

LAND, WATER, FLAG, WRONG = (236, 231, 219), (74, 144, 196), (255, 140, 0), (214, 39, 40)


def sample_scene(size=256, seed=0):
    """(logits, truth): a river and a lake at 10 m, and a model's logits for them. Negative `s` is water."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size].astype(float)
    centre = 0.56 * size + 0.16 * size * np.sin(x / (0.13 * size)) + 0.05 * size * np.sin(x / (0.045 * size))
    river = np.abs(y - centre) - (0.035 * size + 0.012 * size * np.sin(x / (0.2 * size)))
    lake = (np.hypot((x - 0.27 * size) / (0.17 * size), (y - 0.2 * size) / (0.11 * size)) - 1) * 0.11 * size
    s = np.minimum(river, lake)
    truth = s < 0
    ky, kx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    smooth = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.03 ** 2))).real
    shadow = 9.0 * np.exp(-(((x - 0.74 * size) / (0.075 * size)) ** 2 + ((y - 0.17 * size) / (0.05 * size)) ** 2))
    # Saturate the clean signal first, so that the shadow and the noise can overturn a decision far from the shore. The
    # amplitudes are set so that the error rate and the capture resemble a real flood map in this repository's record
    # (Sen1Floods11 Bolivia: about 9% of windows wrong, about a quarter of the errors in the first 5%), not to flatter.
    logits = 6 * np.tanh(-s / 3.0 / 6) + 2.4 * smooth / smooth.std() + 0.6 * rng.standard_normal((size, size)) + shadow
    return logits, truth


def write_png(path, rgb):
    """An (H, W, 3) uint8 array as a PNG, with the standard library only."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _outline(img, r0, c0, side, colour, width=2):
    img[r0:r0 + side, c0:c0 + width] = img[r0:r0 + side, c0 + side - width:c0 + side] = colour
    img[r0:r0 + width, c0:c0 + side] = img[r0 + side - width:r0 + side, c0:c0 + side] = colour


def picture(hard, flagged, wrong, patch, zoom=3):
    """Two panels on one canvas. Left: the map with the windows to check first outlined. Right: the same, with the
    windows where the map is really wrong filled in."""
    base = np.where(hard[..., None] > 0, WATER, LAND).astype(np.uint8).repeat(zoom, 0).repeat(zoom, 1)
    left, right, side = base.copy(), base.copy(), patch * zoom
    for r, c in zip(*np.nonzero(wrong)):
        right[r * side:(r + 1) * side, c * side:(c + 1) * side] = WRONG
    for r, c in zip(*np.nonzero(flagged)):
        _outline(left, r * side, c * side, side, FLAG)
        _outline(right, r * side, c * side, side, FLAG)
    gap = np.full((base.shape[0], 4 * zoom, 3), 255, np.uint8)
    return np.concatenate([left, gap, right], 1)


def run(out="oe_inferencex_demo", seed=0, budget=0.05, patch=4):
    from oe_inferencex import cli                     # here, not at import: cli imports this module for its parser
    os.makedirs(out, exist_ok=True)
    logits, truth = sample_scene(seed=seed)
    scores, labels = os.path.join(out, "sample_logits.npy"), os.path.join(out, "sample_truth.npy")
    np.save(scores, logits.astype(np.float32)); np.save(labels, truth.astype(np.int16))

    audit = os.path.join(out, "audit")               # the files a real `assess` writes, through the real command
    with contextlib.redirect_stdout(io.StringIO()):
        cli.cmd_assess(argparse.Namespace(scores=scores, out=audit, logits=True, patch=patch, nodata=None, reference=labels,
                                          budgets=[0.01, budget, 0.10], order="confidence"))
    s = json.load(open(os.path.join(audit, "assessment.json")))

    a = assess_prediction(logits.astype(np.float32), is_logit=True, patch=patch, reference=truth.astype(int), budgets=(budget,))
    hard_w = a["arrays"]["pooled_argmax"]
    h, w = hard_w.shape
    truth_w = truth[:h * patch, :w * patch].reshape(h, patch, w, patch).mean((1, 3)) >= 0.5
    flagged = np.zeros((h, w), bool)
    rc = np.asarray(a["review_sets"][budget]["windows_rowcol"])
    flagged[rc[:, 0], rc[:, 1]] = True
    png = os.path.join(out, "review_set.png")
    write_png(png, picture(hard_w.repeat(patch, 0).repeat(patch, 1), flagged, hard_w != truth_w, patch))

    cap = s["against_reference"]["error_capture_at_budget"]
    pct = lambda b: f"{100 * cap[str(b)]['errors_captured_fraction']:.0f}%"
    print(f"""A sample water map (made up, {logits.shape[0]} x {logits.shape[1]} pixels) was audited without using any labels.

  {png}
      left:  the map (blue is water), with the {budget:.0%} of windows to check first outlined in orange
      right: the same, with the windows where the map is really wrong filled in red
  {audit}/
      the files `assess` writes for a real map: the review sets as CSV, the reasons, the summary

The scene is made up, so its truth is known: {100 * s['against_reference']['error_rate']:.0f}% of the windows are wrong, and the {budget:.0%}
flagged hold {pct(budget)} of those errors (the 10% flagged hold {pct(0.10)}). A random {budget:.0%} would hold {budget:.0%}.
Errors the model is sure about, like the middle of the red patch at the top right, are not flagged: nothing computed
without labels finds those. All of this is an illustration, not evidence. The evidence, on real maps with expert
labels, is in the documentation:
  https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short

Next, your own map (GeoTIFF or .npy; probabilities, or logits with --logits):
  oe-inferencex assess your_map.tif --out audit""")
    return 0

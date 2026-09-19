"""`oe-inferencex demo`: a first run that needs no data, no labels and nothing but numpy.

It makes a small synthetic water map the way a model would (sure in the middle of the water and the land, unsure along
the shore, fooled by one cloud shadow), audits it through the same code path as `oe-inferencex assess`, and draws the
result. Because the scene is made up its truth is known, so the run can say how many of the map's errors the review
set holds and draw a random pick of the same size beside it. That illustrates the tool; it is not evidence. The
evidence is in the documentation.
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

LAND, WATER, FLAG, WRONG, RANDOM, INK = (236, 231, 219), (74, 144, 196), (255, 140, 0), (214, 39, 40), (60, 60, 60), (40, 40, 40)

FONT = {   # 5 x 7 glyphs, one integer per row, most significant bit on the left
    "A": (14, 17, 17, 31, 17, 17, 17), "B": (30, 17, 17, 30, 17, 17, 30), "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (30, 17, 17, 17, 17, 17, 30), "E": (31, 16, 16, 30, 16, 16, 31), "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 14), "H": (17, 17, 17, 31, 17, 17, 17), "I": (14, 4, 4, 4, 4, 4, 14),
    "J": (7, 2, 2, 2, 2, 18, 12), "K": (17, 18, 20, 24, 20, 18, 17), "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17), "N": (17, 25, 21, 19, 17, 17, 17), "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16), "Q": (14, 17, 17, 17, 21, 18, 13), "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30), "T": (31, 4, 4, 4, 4, 4, 4), "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4), "W": (17, 17, 17, 21, 21, 27, 17), "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4), "Z": (31, 1, 2, 4, 8, 16, 31), "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14), "2": (14, 17, 1, 2, 4, 8, 31), "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 30, 1, 1, 17, 14), "6": (14, 16, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8), "8": (14, 17, 17, 14, 17, 17, 14), "9": (14, 17, 17, 15, 1, 1, 14),
    "%": (25, 25, 2, 4, 8, 19, 19), ":": (0, 4, 4, 0, 4, 4, 0), ",": (0, 0, 0, 0, 4, 4, 8),
    ".": (0, 0, 0, 0, 0, 12, 12), "(": (2, 4, 8, 8, 8, 4, 2), ")": (8, 4, 2, 2, 2, 4, 8),
    "-": (0, 0, 0, 31, 0, 0, 0), " ": (0, 0, 0, 0, 0, 0, 0),
}


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


def _text(img, r0, c0, text, scale=3):
    """Words on the picture with a built-in 5 x 7 font, so that it explains itself when it is passed on alone."""
    for i, ch in enumerate(text.upper()):
        for r, bits in enumerate(FONT.get(ch, FONT[" "])):
            for c in range(5):
                if bits >> (4 - c) & 1:
                    y, x = r0 + r * scale, c0 + (i * 6 + c) * scale
                    img[y:y + scale, x:x + scale] = INK


def picture(hard, flagged, wrong, random_pick, patch, titles, zoom=3):
    """Three titled panels on one canvas: what an audit gives (the map and the windows to check first, no truth shown),
    the same windows over the map's real errors, and a random pick of the same size over the same errors. The third
    panel is the point of the picture: without it nothing shows why the flagged windows are a good choice."""
    base = np.where(hard[..., None] > 0, WATER, LAND).astype(np.uint8).repeat(zoom, 0).repeat(zoom, 1)
    side, bar = patch * zoom, 45
    errors = base.copy()
    for r, c in zip(*np.nonzero(wrong)):
        errors[r * side:(r + 1) * side, c * side:(c + 1) * side] = WRONG
    panels = [base.copy(), errors.copy(), errors.copy()]
    for panel, picks, colour in zip(panels, (flagged, flagged, random_pick), (FLAG, FLAG, RANDOM)):
        for r, c in zip(*np.nonzero(picks)):
            _outline(panel, r * side, c * side, side, colour)
    out = []
    for panel, title in zip(panels, titles):
        head = np.full((bar, panel.shape[1], 3), 255, np.uint8)
        _text(head, 12, 8, title)
        out += [np.concatenate([head, panel], 0), np.full((bar + panel.shape[0], 4 * zoom, 3), 255, np.uint8)]
    return np.concatenate(out[:-1], 1)


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
    wrong = hard_w != truth_w
    random_pick = np.zeros(h * w, bool)
    random_pick[np.random.default_rng(seed).choice(h * w, size=int(flagged.sum()), replace=False)] = True
    random_pick = random_pick.reshape(h, w)
    held = lambda picks: f"{100 * (wrong & picks).sum() / max(wrong.sum(), 1):.0f}%"
    png = os.path.join(out, "review_set.png")
    write_png(png, picture(hard_w.repeat(patch, 0).repeat(patch, 1), flagged, wrong, random_pick, patch, titles=(
        f"No labels used: the {budget:.0%} to check first", f"The same {budget:.0%}: {held(flagged)} of the real errors (red)",
        f"A random {budget:.0%}: {held(random_pick)} of the real errors")))

    cap = s["against_reference"]["error_capture_at_budget"]
    pct = lambda b: f"{100 * cap[str(b)]['errors_captured_fraction']:.0f}%"
    print(f"""A sample water map (made up, {logits.shape[0]} x {logits.shape[1]} pixels) was audited without using any labels.

  {png}
      left:   the map (blue is water), with the {budget:.0%} of windows to check first outlined in orange
      middle: the same windows over the places where the map is really wrong, in red
      right:  a random {budget:.0%} over the same errors, for comparison
  {audit}/
      the files `assess` writes for a real map: the review sets as CSV, the reasons, the summary

The scene is made up, so its truth is known: {100 * s['against_reference']['error_rate']:.0f}% of the windows are wrong, and the {budget:.0%}
flagged hold {pct(budget)} of those errors (the 10% flagged hold {pct(0.10)}). A random {budget:.0%} would hold {budget:.0%}.
Errors the model is sure about, like the middle of the red patch at the top right, are not flagged: nothing computed
without labels finds those. All of this is an illustration, not evidence. The evidence, on real maps with expert
labels, is in the documentation:
  https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short

Next, run the real command on the sample, then on your own map (GeoTIFF or .npy; probabilities, or logits with
--logits). How to get a score map out of a model is in https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/
  oe-inferencex assess {scores} --logits --out my_audit
  oe-inferencex assess your_map.tif --out audit""")
    return 0

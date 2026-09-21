"""`oe-inferencex demo`: the first run. It depends on numpy alone, so it works wherever the package installs.

By default it audits a real map: one tile of Dynamic World, a served global land-cover product this project had no hand
in, with the probabilities it publishes about itself and the expert annotation of the same ground (Zenodo record
4766508, CC BY 4.0). The tile was chosen by a rule written before any tile was looked at, the median tile by error
capture among the fully annotated ones, so it is representative and not flattering (scripts/make_demo_sample.py).
`--made-up` audits a small synthetic water map instead. Either way the map goes through the same code path as
`oe-inferencex assess`, and because the truth is known the run can say how many of the map's errors the review set
holds and draw a random pick of the same size beside it. One tile illustrates the tool; it is not the evidence. The
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
from oe_inferencex.metrics import attainable_ceiling

LAND, WATER, FLAG, WRONG, RANDOM, INK = (236, 231, 219), (74, 144, 196), (255, 140, 0), (214, 39, 40), (60, 60, 60), (40, 40, 40)
UNKNOWN = (255, 255, 255)

DW_COLOURS = ((65, 155, 223), (57, 125, 73), (136, 176, 83), (122, 135, 198), (228, 150, 53), (223, 195, 90), (196, 40, 27),
              (165, 155, 143), (179, 159, 225))
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample", "dynamic_world_tile.npz")

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


def picture(hard, flagged, wrong, random_pick, titles, colours, side=12, unknown=None, legend=None, soften=0.0, flag=FLAG):
    """Three titled panels on one canvas, all at window resolution: what an audit gives (the map and the windows to check
    first, no truth shown), the same windows over the map's real errors, and a random pick of the same size over the same
    errors. The third panel is the point of the picture: without it nothing shows why the flagged windows are a good
    choice. `soften` blends the map towards white so that the marks stay readable on a many-coloured map; `unknown`
    whitens the windows whose truth nobody marked, in the two panels that show truth."""
    rgb = np.asarray(colours, float)[hard]
    base = (rgb + soften * (255 - rgb)).astype(np.uint8).repeat(side, 0).repeat(side, 1)
    errors = base.copy()
    for mask, colour in ((unknown, UNKNOWN), (wrong, WRONG)):
        if mask is not None:
            for r, c in zip(*np.nonzero(mask)):
                errors[r * side:(r + 1) * side, c * side:(c + 1) * side] = colour
    panels = [base.copy(), errors.copy(), errors.copy()]
    for panel, picks, colour in zip(panels, (flagged, flagged, random_pick), (flag, flag, RANDOM)):
        for r, c in zip(*np.nonzero(picks)):
            _outline(panel, r * side, c * side, side, colour, width=max(1, side // 5))
    bar, out = 45, []
    for panel, title in zip(panels, titles):
        head = np.full((bar, panel.shape[1], 3), 255, np.uint8)
        _text(head, 12, 8, title)
        out += [np.concatenate([head, panel], 0), np.full((bar + panel.shape[0], 12, 3), 255, np.uint8)]
    canvas = np.concatenate(out[:-1], 1)
    if legend:
        rows, x = [np.full((45, canvas.shape[1], 3), 255, np.uint8)], 8
        for name, colour in legend:
            width = 30 + 18 * len(name) + 36
            if x + width > canvas.shape[1]:
                rows.append(np.full((45, canvas.shape[1], 3), 255, np.uint8)); x = 8
            rows[-1][12:33, x:x + 21] = colour
            rows[-1][12:33, x:x + 21][[0, -1], :] = rows[-1][12:33, x:x + 21][:, [0, -1]] = INK
            _text(rows[-1], 12, x + 30, name)
            x += width
        canvas = np.concatenate([canvas] + rows, 0)
    return canvas


def _made_up(seed):
    logits, truth = sample_scene(seed=seed)
    n = logits.shape[0]
    return {"scores": logits.astype(np.float32), "truth": truth.astype(np.int16), "is_logit": True, "patch": 4, "flags": "--logits",
            "colours": (LAND, WATER), "soften": 0.0, "flag": FLAG, "classes": None, "file": "sample_logits.npy",
            "intro": f"A small made-up water map ({n} x {n} pixels, blue is water) was audited without using any labels.",
            "sure": "Errors the model is sure about, like the middle of the red patch at the top right, are not\nflagged: "
                    "nothing computed without labels finds those.",
            "credit": None}


def _real():
    z = np.load(SAMPLE)
    meta = json.loads(str(z["meta"]))
    probs, expert = z["probs"].astype(np.float32), z["expert"].astype(np.int16)
    ns, ew = ("N" if meta["lat"] >= 0 else "S"), ("E" if meta["lon"] >= 0 else "W")
    return {"scores": probs, "truth": expert, "is_logit": False, "patch": 1, "flags": "--patch 1",
            "colours": DW_COLOURS, "soften": 0.6, "flag": INK, "classes": meta["classes"], "file": "sample_probabilities.npy",
            "intro": f"A real map was audited without using any labels: Dynamic World, a global 10 m land-cover product, on one tile\n"
                     f"of its public test set near {abs(meta['lat']):.1f} {ns}, {abs(meta['lon']):.1f} {ew} ({probs.shape[1]} x {probs.shape[2]} windows of "
                     f"{meta['window_m']} m). The tile was chosen by a rule fixed in advance,\nthe median tile by error capture among "
                     f"{meta['n_candidates']} fully annotated ones: a typical tile, not the best one.",
            "sure": "Errors the model is sure about are not flagged: nothing computed without labels finds those.",
            "credit": f"Map and expert labels: Dynamic World test tiles, Zenodo record {meta['zenodo_record']}, {meta['licence']} "
                      f"(Brown et al., Scientific Data, 2022)."}


def run(out="oe_inferencex_demo", seed=0, budget=0.05, made_up=False):
    from oe_inferencex import cli                     # here, not at import: cli imports this module for its parser
    from oe_inferencex.assess import _pool, _pooled_argmax
    os.makedirs(out, exist_ok=True)
    sc = _made_up(seed) if made_up or not os.path.exists(SAMPLE) else _real()
    patch = sc["patch"]
    scores, labels = os.path.join(out, sc["file"]), os.path.join(out, "sample_truth.npy")
    np.save(scores, sc["scores"]); np.save(labels, sc["truth"])

    audit = os.path.join(out, "audit")               # the files a real `assess` writes, through the real command
    with contextlib.redirect_stdout(io.StringIO()):
        cli.cmd_assess(argparse.Namespace(scores=scores, out=audit, logits=sc["is_logit"], patch=patch, nodata=None,
                                          reference=labels, budgets=[0.01, budget, 0.10], order="confidence"))
    s = json.load(open(os.path.join(audit, "assessment.json")))
    why = json.load(open(os.path.join(audit, "explanation.json")))

    nodata = ~np.isfinite(sc["scores"]).reshape(-1, *sc["scores"].shape[-2:]).all(0)
    ref = np.where(nodata, -1, sc["truth"]).astype(int)
    a = assess_prediction(sc["scores"], is_logit=sc["is_logit"], patch=patch, nodata_mask=nodata, reference=ref, budgets=(budget, 0.10, 0.20, 0.50))
    hard_w, valid_w, conf_w = a["arrays"]["pooled_argmax"], a["arrays"]["valid"], a["arrays"]["confidence"]
    h, w = hard_w.shape
    n_classes = a["n_classes"]
    known = valid_w & (_pool((ref >= 0).astype(float), patch) >= 0.5)            # the windows assess scored, by its own rule
    wrong = known & (_pooled_argmax(ref, n_classes, patch) != hard_w)
    flagged = np.zeros((h, w), bool)
    rc = np.asarray(a["review_sets"][budget]["windows_rowcol"])
    flagged[rc[:, 0], rc[:, 1]] = True
    random_pick = np.zeros(h * w, bool)
    random_pick[np.random.default_rng(seed).choice(np.flatnonzero(valid_w.ravel()), size=int(flagged.sum()), replace=False)] = True
    random_pick = random_pick.reshape(h, w)
    hit = lambda picks: f"{100 * (wrong & picks).sum() / max((known & picks).sum(), 1):.0f}%"   # how often a picked window is wrong
    base = f"{100 * wrong[known].mean():.0f}%"       # what a random pick finds on average: one draw could flatter or hurt

    colours = tuple(sc["colours"]) + (UNKNOWN,)
    legend = None
    if sc["classes"]:
        share = [(hard_w[valid_w] == c).mean() for c in range(n_classes)]
        soft = lambda c: tuple(int(v + sc["soften"] * (255 - v)) for v in c)
        legend = [(sc["classes"][c].replace("_", " "), soft(colours[c])) for c in np.argsort(share)[::-1] if share[c] >= 0.03]
        legend += [("check first", sc["flag"]), ("really wrong", WRONG), ("random pick", RANDOM), ("not marked by the expert", UNKNOWN)]
    png = os.path.join(out, "review_set.png")
    write_png(png, picture(np.where(valid_w, hard_w, n_classes), flagged, wrong, random_pick, side=max(2, 768 // max(h, w)),
                           colours=colours, soften=sc["soften"], flag=sc["flag"], unknown=(valid_w & ~known) if sc["classes"] else None,
                           legend=legend, titles=(f"No labels used: the {budget:.0%} to check first",
                                                  f"Of those, {hit(flagged)} are really wrong (red)",
                                                  f"Of a random {budget:.0%}, about {base} are really wrong")))

    cap = s["against_reference"]["error_capture_at_budget"]
    at = cap[str(budget)]
    sure_first = np.argsort(-conf_w[known], kind="stable")[: int(round(0.8 * known.sum()))]
    expl = why["budgets"][str(budget)]
    err = a["against_reference"]["error_rate"]
    n_scored = a["against_reference"]["n_windows_scored"]
    table = "\n".join(f"      {b:>4.0%} {100 * v['errors_captured_fraction']:>16.0f}% {b:>23.0%} {attainable_ceiling(b, err, n=n_scored):>25.0%}"
                      for b, v in a["against_reference"]["error_capture_at_budget"].items())
    first = expl["windows"][0]
    edge = "the shore the model drew" if not sc["classes"] else "a boundary between two classes of the model's own map"
    mark = "orange" if sc["flag"] == FLAG else "black"
    print(f"""{sc["intro"]}

  {png}
      left:   the map, with the {budget:.0%} of windows to check first outlined in {mark}
      middle: the same windows over the places where the map is really wrong, in red
      right:  a random {budget:.0%} over the same errors, for comparison
  {audit}/
      the files `assess` writes for any map: the review sets as CSV with coordinates, the reasons, the summary

What it is worth. The map is wrong on {100 * s['against_reference']['error_rate']:.0f}% of its windows. Of the {budget:.0%} the tool flags, {100 * at['precision_in_set']:.0f}% are really wrong, so a
reviewer who goes where it points finds an error far more often than one who picks at random ({base}).

    review   errors it would find   a random review finds   the most any review could
{table}

Most of the red in the picture lies outside the flagged windows because no review of {budget:.0%} can cover a map that is {base}
wrong. The other way round: the map is right on {100 * (1 - wrong[known].mean()):.0f}% of its windows, and on {100 * (1 - wrong[known][sure_first].mean()):.0f}% of the 80% it is surest about, so
confidence also says which part can be used as it is.

Why a window is flagged. Of the {expl['n_windows']} flagged windows, {100 * expl['share_in_set']['boundary']:.0f}% sit on {edge},
and all are among its least confident, which is how they were ranked. explanation.json gives the reasons window by
window, each with the evidence measured for it on expert-labelled maps; the first is window row {first['row']}, column {first['col']}.

What it does not do. {sc["sure"]} It does not say
how wrong a map is either: that needs a reference, as here. One map illustrates the tool; it is not the evidence. The
evidence, on many maps with expert labels, is here:
  https://olmoearth-inferencex.readthedocs.io/en/latest/Findings/#in-short

Next, run the real command on this sample, then on your own map (GeoTIFF or .npy; probabilities, or logits with
--logits). How to get a score map out of a model: https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/
  oe-inferencex assess {scores} {sc["flags"]} --out my_audit
  oe-inferencex assess your_map.tif --out audit""" + (f"\n\n{sc['credit']}" if sc["credit"] else ""))
    return 0

"""A served class map with its confidence band as the input: the --confidence options and the band read."""
import os

import numpy as np

from oe_inferencex.common.raster import _same_grid, read_raster

PRODUCT_SIGNAL = "exported confidence band"
MAX_CLASS_ID = 255
NO_RANGE_NOTE = ("the whole confidence band is read as confidence; if it also holds codes (LCMAP's lcpconf writes "
                 "provenance codes from 151), pass --confidence-range so they are left out, or they rank above every "
                 "confidence")


def _read_product(classes_path, conf_path, nodata, conf_range, command):
    """A published product's two layers: a class map of integer ids and a per-pixel confidence band on its grid,
    higher = more confident, as products such as LCMAP ship them (lcpri with lcpconf). Band values outside conf_range
    are not confidences and are left out as no-data: LCMAP writes provenance codes from 151 into lcpconf, and read raw
    they would rank above every confidence and head the certified zone (exp93). Without a range the package cannot tell
    codes from confidences. `nodata` applies to the class map; the band's no-data is its file's own tag (or NaN).
    Returns (classes with -1 outside the valid pixels, confidence with NaN there, valid, geo, n_classes, notes, info),
    info holding how many classified pixels the range left out, since the population is then the rest."""
    hard, valid_h, geo = read_raster(classes_path, nodata)
    if not os.path.exists(conf_path):
        raise SystemExit(f"{command}: the confidence band {conf_path} is not a file")
    conf, valid_c, geo_c = read_raster(conf_path, None)
    if hard.ndim != 2 or conf.ndim != 2:
        raise SystemExit(f"{command}: with --confidence, the map is one band of class ids and the confidence one band; "
                         f"got shapes {hard.shape} and {conf.shape}")
    if conf.shape != hard.shape:
        raise SystemExit(f"{command}: the confidence band has shape {conf.shape}; the class map is {hard.shape}")
    _same_grid(geo, geo_c, conf_path)
    valid = valid_h & valid_c
    h = hard[valid]
    if h.size and not np.array_equal(h, np.rint(h)):
        raise SystemExit(f"{command}: {classes_path} holds values that are not whole numbers; with --confidence it must "
                         "be the class map of integer class ids (per-class scores go without --confidence)")
    if h.size and h.min() < 0:
        raise SystemExit(f"{command}: {classes_path} holds negative class ids at valid pixels; mark no-data with --nodata")
    if h.size and h.max() > MAX_CLASS_ID:
        raise SystemExit(f"{command}: {classes_path} holds class id {int(h.max())}; ids above {MAX_CLASS_ID} are refused, "
                         "since the window's majority class is counted per id. Mark a fill value with --nodata, or "
                         "renumber the classes")
    notes = []
    info = {"classified_pixels": int(valid.sum()), "outside_range": 0}
    if conf_range is None:
        notes.append(NO_RANGE_NOTE)
    else:
        lo, hi = (float(v) for v in conf_range)
        if not lo < hi:
            raise SystemExit(f"{command}: --confidence-range needs LOW below HIGH, got {lo:g} {hi:g}")
        outside = valid & ~((conf >= lo) & (conf <= hi))
        if outside.any():
            n_out, n_val = int(outside.sum()), int(valid.sum())
            notes.append(f"{n_out} of {n_val} pixels ({100 * n_out / n_val:.1f}%) have a confidence outside "
                         f"[{lo:g}, {hi:g}] and are left out as no-data: every statement is then about the rest")
            info["outside_range"] = n_out
        valid = valid & ~outside
    if not valid.any():
        raise SystemExit(f"{command}: no pixel has both a class and a confidence"
                         + ("" if conf_range is None else " inside --confidence-range"))
    classes = np.where(valid, np.rint(np.where(valid, hard, 0)), -1).astype(np.int64)
    return classes, np.where(valid, conf.astype(np.float64), np.nan), valid, geo, int(classes.max()) + 1, notes, info


def _product_args(args, command):
    """--confidence and --confidence-range, checked together; --logits reads per-class scores and does not apply."""
    conf, rng = getattr(args, "confidence", None), getattr(args, "confidence_range", None)
    if rng is not None and conf is None:
        raise SystemExit(f"{command}: --confidence-range needs --confidence")
    if conf is not None and getattr(args, "logits", False):
        raise SystemExit(f"{command}: --logits reads per-class scores; with --confidence the map is a class map and the "
                         "band is its confidence")
    return conf, rng


def _product_parser_args(p):
    p.add_argument("--confidence", default=None, metavar="BAND",
                   help="a per-pixel confidence band on the map's grid that rises with confidence, as published "
                        "products ship it (LCMAP's lcpconf beside lcpri); negate an uncertainty band first. The map "
                        "argument is then the class map of integer ids, and --nodata applies to it; the band's no-data "
                        "is its own file's")
    p.add_argument("--confidence-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="the band's values that are confidences; values outside are left out as no-data, and every "
                        "statement is about the rest. Without it the package cannot tell codes from confidences "
                        "(LCMAP: 1 100, since lcpconf holds provenance codes from 151)")

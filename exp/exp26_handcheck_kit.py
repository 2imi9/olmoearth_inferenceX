"""exp26: hand-check kit for the disagreements that tiling instability ranks first.

exp23-exp25 removed every measurable reference-versus-image mismatch and the
WorldCover-referenced advantage of tiling instability over confidence
survived. The next step needs a human eye: are the disagreements that
tiling instability flags first model errors, reference errors, or something
else? This script prepares the material; it makes no claim.

Scene rule: the three scenes with the largest tiling-instability gain over
confidence in exp13 (excess AURC, 2024 imagery). For each scene, the
disagreement patches between the 2024 head and WorldCover 2021 are ranked
by tiling instability and by confidence; the top 12 of each ranking are
shown (patches in both sets are marked). For every patch: a 48-px
true-colour crop of the Sentinel-2 image with the 4-px patch outlined, the
NDWI crop, the WorldCover 2021 pixels (water in blue), and the head's water
probability; plus a CSV row with the model probability, the reference
label, the NDWI statistics inside the patch (a crude spectral water cue,
not a truth), the JRC seasonality and occurrence, both ranks, and empty
'verdict' and 'note' columns for the reviewer (verdict vocabulary: model
error / reference error / seasonal or date difference / ambiguous).

Outputs: exp/out/exp26_handcheck_<scene>_<ranker>.png and
exp/out/exp26_handcheck_<scene>.csv; summary of the kit in exp26_summary.json.
Inputs are cached artefacts only (no network).
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olmoearth_pretrain.data.constants import Modality  # noqa: E402

from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from exp24_year2021 import signals  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE, PATCH = 128, 4
GRID = SIZE // PATCH
K = 12
CROP = 48
N_SCENES = 3
BO = Modality.SENTINEL2_L2A.band_order


def scene_rule():
    rows = list(csv.DictReader(open(os.path.join(OUT, "exp13_corrected_stats.csv"))))
    by = {}
    for r in rows:
        by.setdefault(r["scene"], {})[r["signal"]] = float(r["eaurc"])
    gain = {s: v["baseline"] - v["tile-phase (aligned)"] for s, v in by.items() if "tile-phase (aligned)" in v}
    return sorted(gain, key=gain.get, reverse=True)[:N_SCENES], gain


def stretch(rgb):
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def crop(a, r, c, size):
    """Crop of a 2-D or 3-D (H, W, C) array centred on patch (r, c); pads with edge values."""
    h = size // 2
    cy, cx = r * PATCH + PATCH // 2, c * PATCH + PATCH // 2
    pad = ((h, h), (h, h)) + (((0, 0),) if a.ndim == 3 else ())
    ap = np.pad(a, pad, mode="edge")
    return ap[cy:cy + size, cx:cx + size]


def main():
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    feats = dict(np.load(os.path.join(OUT, "exp11_feats.npz"), allow_pickle=True))
    geo = dict(np.load(os.path.join(OUT, "exp23_geo.npz"), allow_pickle=True))
    jrc = dict(np.load(os.path.join(OUT, "exp25_jrc.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    torch.manual_seed(0)
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), z3["tr_labels"])
    hn = train_logistic_head(torch.tensor(feats["tr_nano"]), z3["tr_labels"])
    a_tr = torch.nn.functional.normalize(torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)
    chosen, gain = scene_rule()
    summary = {"scene_rule": "top 3 by tiling-instability excess-AURC gain over confidence in exp13", "scenes": {}, "k_per_ranker": K}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from oe_inferencex import figstyle
    figstyle.setup()

    for name in chosen:
        img = scenes[f"{name}_img"].astype(np.float64)[:, :SIZE, :SIZE]
        lab = scenes[f"{name}_lab"].astype(bool)
        p, sigs = signals(feats, name, hb, hn, a_tr, scenes[f"{name}_img"])
        err = (p > 0.5) != lab
        rgb = stretch(np.stack([img[BO.index("B04")], img[BO.index("B03")], img[BO.index("B02")]], -1))
        ndwi = (img[BO.index("B03")] - img[BO.index("B08")]) / np.clip(img[BO.index("B03")] + img[BO.index("B08")], 1, None)
        wc = geo[f"{name}_wc21"][:SIZE, :SIZE] if f"{name}_wc21" in geo else None
        sea = jrc.get(f"2024_{name}_seasonality"); occ = jrc.get(f"2024_{name}_occurrence")
        p_pix = np.kron(p, np.ones((PATCH, PATCH)))
        tile, conf = sigs["tile-phase (aligned)"], sigs["baseline"]
        idx = np.flatnonzero(err.flatten())
        rank_t = {i: r + 1 for r, i in enumerate(sorted(idx, key=lambda i: -tile.flatten()[i]))}
        rank_c = {i: r + 1 for r, i in enumerate(sorted(idx, key=lambda i: -conf.flatten()[i]))}
        top_t = [i for i in sorted(idx, key=lambda i: rank_t[i])[:K]]
        top_c = [i for i in sorted(idx, key=lambda i: rank_c[i])[:K]]
        rows = []
        for i in sorted(set(top_t) | set(top_c), key=lambda i: (min(rank_t[i], rank_c[i]))):
            r, c = divmod(int(i), GRID)
            pr, pc = slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH)
            nd = ndwi[pr, pc]
            rows.append({"scene": name, "patch_row": r, "patch_col": c, "in_top_tile": i in top_t, "in_top_confidence": i in top_c,
                         "rank_tile": rank_t[i], "rank_confidence": rank_c[i], "n_disagreements": int(err.sum()),
                         "model_p_water": float(p[r, c]), "worldcover_water": bool(lab[r, c]),
                         "ndwi_mean": float(nd.mean()), "ndwi_frac_gt0": float((nd > 0).mean()),
                         "jrc_seasonality_max_months": int(np.where(sea[pr, pc] > 12, 0, sea[pr, pc]).max()) if sea is not None else -1,
                         "jrc_occurrence_max_pct": int(np.where(occ[pr, pc] > 100, 0, occ[pr, pc]).max()) if occ is not None else -1,
                         "tile_phase_score": float(tile[r, c]), "confidence_score": float(conf[r, c]), "verdict": "", "note": ""})
        with open(os.path.join(OUT, f"exp26_handcheck_{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summary["scenes"][name] = {"tile_gain_exp13": gain[name], "n_disagreements": int(err.sum()), "overlap_of_top_sets": int(len(set(top_t) & set(top_c))),
                                   "csv": f"exp26_handcheck_{name}.csv"}
        for ranker, top in (("tile", top_t), ("confidence", top_c)):
            fig, axes = plt.subplots(4, K, figsize=(1.35 * K, 6.2))
            for j, i in enumerate(top):
                r, c = divmod(int(i), GRID)
                panels = [(crop(rgb, r, c, CROP), None, "true colour"), (crop(ndwi, r, c, CROP), ("RdYlBu", -0.6, 0.6), "NDWI"),
                          (crop((wc == 80).astype(float), r, c, CROP) if wc is not None else np.zeros((CROP, CROP)), ("Blues", 0, 1), "WorldCover 2021 water"),
                          (crop(p_pix, r, c, CROP), ("Blues", 0, 1), "head P(water)")]
                for a, (im, cm, title) in enumerate(panels):
                    ax = axes[a, j]
                    if cm is None:
                        ax.imshow(im, interpolation="nearest")
                    else:
                        ax.imshow(im, cmap=cm[0], vmin=cm[1], vmax=cm[2], interpolation="nearest")
                    h = CROP // 2 - PATCH // 2
                    ax.add_patch(Rectangle((h - 0.5, h - 0.5), PATCH, PATCH, fill=False, edgecolor="#d62728", linewidth=1.2))
                    ax.set_xticks([]); ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel(title, fontsize=8)
                both = "*" if (i in top_t and i in top_c) else ""
                axes[0, j].set_title(f"#{j + 1}{both} r{r} c{c}\np={p[r, c]:.2f} wc={'water' if lab[r, c] else 'dry'}\nrank t{rank_t[i]} c{rank_c[i]}", fontsize=7)
            fig.suptitle(f"exp26 hand-check kit: {name}, top {K} disagreements ranked by {'tiling instability' if ranker == 'tile' else 'confidence'} "
                         f"({int(err.sum())} disagreements; * = in both top sets; red box = the 4-px patch; crops {CROP} px = {CROP * 10 / 1000:.2f} km)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"exp26_handcheck_{name}_{ranker}.png"), dpi=150)
            plt.close(fig)
        print(f"{name}: {int(err.sum())} disagreements; top-{K} sets overlap {len(set(top_t) & set(top_c))}; kit written")
    with open(os.path.join(OUT, "exp26_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()

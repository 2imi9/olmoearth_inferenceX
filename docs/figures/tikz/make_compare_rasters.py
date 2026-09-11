"""Rasters and numbers for the comparison diagram (compare_inferences.tex), from the committed exp57 / exp58 artifacts:
one Sen1Floods11 Bolivia tile (tile 209) with the S2 head's and the S1 head's W1 decisions on identical windows
(exp/out/exp57_masks.npz), the disagreement windows and the ones on a boundary of the first map, and TeX macros
carrying the recorded numbers (exp/out/exp57_summary.json, exp/out/exp58_summary.json) so the figure cannot drift
from the ledger. Writes into docs/figures/tikz/rasters/.

    uv run python docs/figures/tikz/make_compare_rasters.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_rasters import OUT, classmap, rgb, save, tex_list  # noqa: E402
from oe_inferencex.compare import compare_inferences  # noqa: E402
from oe_inferencex.signals import S2_BANDS, boundary_indicator  # noqa: E402

TILE = 209
A, B = "bolivia_s2", "bolivia_s1"          # the pair drawn: the S2 head against the S1 head (exp46's arms, exp57's grid)
PAIR_KEY = "sensors"
GRID_PX, OFFSET = 16, 1                    # the 14 x 14 W1 windows are windows 1..14 of the 16-cell grid over the 64-px tile


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "exp", "out", "exp57_masks.npz"))
    a, b, ok, y = z[A][TILE], z[B][TILE], z["bolivia_ok"][TILE], z["bolivia_y"][TILE]
    d = torch.load(os.path.join(ROOT, "data", "floods", "flood_bolivia_data.pt"), weights_only=True)
    l1c = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    img = d["s2"].numpy().astype(np.float32)[TILE][[l1c.index(k) for k in S2_BANDS]]
    save("cmp_rgb.png", rgb(img, 64), 8)
    for name, m in (("cmp_a.png", a), ("cmp_b.png", b), ("cmp_label.png", y)):
        im = classmap(m.astype(float))
        im[~ok] = (200, 200, 200)
        save(name, im, 32)
    out = compare_inferences(a, b, ok, labels=y, cues={"boundary": boundary_indicator(a) > 0})
    dis = out["arrays"]["disagree"]
    bnd = (boundary_indicator(a) > 0) & dis
    tex_list("cmp_disagree.tex", "cmpDisagree", [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(dis & ~bnd))])
    tex_list("cmp_disagree_boundary.tex", "cmpDisagreeBoundary", [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(bnd))])
    tex_list("cmp_a_right.tex", "cmpARight", [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(dis & (a == y)))])
    tex_list("cmp_b_right.tex", "cmpBRight", [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(dis & (b == y)))])

    s57 = json.load(open(os.path.join(ROOT, "exp", "out", "exp57_summary.json")))
    s58 = json.load(open(os.path.join(ROOT, "exp", "out", "exp58_summary.json")))
    R = s57["results"]
    sb = R[PAIR_KEY]["bolivia"]
    w = sb["where"]
    cross = R["cross_pairs"]["bolivia"]
    geo57, geo58 = R["geoid"], s58["results"]["geoid"]["change"]
    macros = {
        "cmpTile": str(TILE),
        "cmpTileRate": f"{100 * dis.sum() / ok.sum():.0f}", "cmpTileDisagree": str(int(dis.sum())), "cmpTileWindows": str(int(ok.sum())),
        "cmpPooledRate": f"{100 * sb['disagreement_rate']:.1f}", "cmpPooledWindows": f"{sb['n_windows']:,}",
        "cmpEnrBoundary": f"{w['boundary']['enrichment']:.1f}", "cmpEnrLowConf": f"{w['low_confidence']['enrichment']:.1f}",
        "cmpEnrUnstable": f"{w['unstable']['enrichment']:.1f}", "cmpEnrNdwi": f"{w['ndwi_ambiguous']['enrichment']:.1f}",
        "cmpShareBoundaryDis": f"{100 * w['boundary']['share_disagree']:.0f}", "cmpShareBoundaryAgr": f"{100 * w['boundary']['share_agree']:.0f}",
        "cmpARightShare": f"{100 * sb['graded']['which_side']['share_a_right']:.0f}", "cmpBRightShare": f"{100 * sb['graded']['which_side']['share_b_right']:.0f}",
        "cmpErrorsPhi": f"{sb['graded']['crosstab']['phi']:.2f}",
        "cmpGeoRate": f"{100 * geo57['disagreement_rate']:.1f}", "cmpGeoEvents": str(geo57["across_events"]["n_events"]),
        "cmpGeoBoundaryEnr": f"{geo57['where']['boundary']['enrichment']:.0f}",
        "cmpGeoFloodedAll": f"{100 * geo58['flooded_all']:.0f}", "cmpGeoFloodedBoth": f"{100 * geo58['flooded_both_confident']:.0f}",
        "cmpGeoBothShare": f"{100 * geo58['share_both_confident']:.0f}",
        "cmpDrawsPhi": f"{sb['stability_across_draws']['median_pairwise_phi']:.2f}",
    }
    names = cross["names"]
    short = {"offsets 0 vs 2": "offsets", "backbones": "backbones", "sensors": "sensors", "finetune": "fine-tune"}
    P = np.array(cross["phi"])
    with open(os.path.join(OUT, "cmp_numbers.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\def\\{k}{{{v}}}\n")
        f.write("\\def\\cmpPhiNames{" + ", ".join(f"{i}/{short[n]}" for i, n in enumerate(names)) + "}\n")
        f.write("\\def\\cmpPhiCells{" + ", ".join(f"{i}/{j}/{P[i, j]:.2f}" for i in range(len(names)) for j in range(len(names))) + "}\n")
    print(f"tile {TILE}: {int(ok.sum())} valid windows, {int(dis.sum())} disagree ({int(bnd.sum())} on a boundary of the S2 map), "
          f"S2 right on {int((dis & (a == y)).sum())}, S1 right on {int((dis & (b == y)).sum())}; pooled Bolivia rate {sb['disagreement_rate']:.4f}, "
          f"boundary {w['boundary']['enrichment']:.2f}x; phi matrix {names}")


if __name__ == "__main__":
    main()

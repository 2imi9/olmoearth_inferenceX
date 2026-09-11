"""Rasters and numbers for the comparison diagram (compare_inferences.tex): one GEOID-Flood chip across two dates,
the pre-event Sentinel-2 head against the post-event Sentinel-1 head on identical windows, with exp57's recorded
decisions (exp/out/exp57_masks.npz, clear test chip 2293: event EMSR275-1, tile EMSR275-1-4, rows 0-63, columns
256-319) and the chip's imagery and label (rasters/cmp_chip.npz, read once from the shard tree on the cluster with
exp55's loader; S2 in the encoder's band order, S1 in dB, label 0 background / 1 permanent water / 2 flooded / 255
ignore). Dates come from the shard file names: S2 composite pre 2017-07-07, S1 post 2018-03-22. The numbers are read
from exp/out/exp57_summary.json, exp58_summary.json and exp59_summary.json and written as TeX macros so the figure
cannot drift from the ledger.

    uv run python docs/figures/tikz/make_compare_rasters.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_rasters import OUT, classmap, gray, rgb, save, tex_list  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex.signals import boundary_indicator  # noqa: E402
import exp49_shrug_signals as _e49  # noqa: E402


def e49_w1(p_shift):
    return _e49.w1_windows(p_shift[:, None])[0, 1:15, 1:15]

CHIP, EVENT, DATES = 2293, "EMSR275-1", {"pre": "2017-07-07", "post": "2018-03-22"}
GRID_PX, OFFSET = 16, 1                    # the 14 x 14 W1 windows are windows 1..14 of the 16-cell grid over the 64-px chip
FLOOD = (109, 40, 217)


def cells(mask):
    return [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(mask))]


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "exp", "out", "exp57_masks.npz"))
    assert z["geoid_event"][CHIP] == EVENT
    a, b, ok, ya, yb = (z[k][CHIP] for k in ("geoid_s2", "geoid_s1", "geoid_ok", "geoid_y_permanent", "geoid_y_after"))
    chip = np.load(os.path.join(OUT, "cmp_chip.npz"))
    s2, s1, lab = chip["s2"], chip["s1"], chip["lab"]
    save("cmp_s2.png", rgb(s2, 64), 8)
    save("cmp_s1.png", np.repeat(gray(s1[0], 64)[..., None], 3, -1), 8)
    for name, m in (("cmp_a.png", a), ("cmp_b.png", b)):
        im = classmap(m.astype(float))
        im[~ok] = (200, 200, 200)
        save(name, im, 32)
    lab_img = classmap((lab == 1).astype(float))
    lab_img[lab == 2] = FLOOD
    lab_img[lab == 255] = (200, 200, 200)
    save("cmp_label.png", lab_img, 8)
    layers = np.load(os.path.join(OUT, "cmp_chip_layers.npz"))
    from make_rasters import cmap
    from oe_inferencex.signals import ndwi_level
    import exp18_sen1floods_expert as exp18
    pA, pB = layers["p_A"], layers["p_B"]                                           # (4, 15, 15) shift maps of each head
    wa, wb = (e49_w1(pA), e49_w1(pB))
    agree_a, agree_b = float(((wa > 0.5) == a)[ok].mean()), float(((wb > 0.5) == b)[ok].mean())
    conf_a = -np.abs(wa - 0.5)
    tp_a = exp18.aligned_tile_phase(pA[:, None])[0, 1:15, 1:15]
    ndwi = ndwi_level(s2, patch=4, size=60)[1:15, 1:15]
    save("cmp_boundary.png", cmap(boundary_indicator(a), "Oranges", vmin=0, vmax=1, valid=ok), 32)
    save("cmp_conf.png", cmap(conf_a, "viridis", valid=ok), 32)
    save("cmp_tilephase.png", cmap(tp_a, "magma", valid=ok), 32)
    save("cmp_ndwi.png", cmap(ndwi, "cividis", valid=ok), 32)
    both = (np.abs(wa - 0.5) >= 0.25) & (np.abs(wb - 0.5) >= 0.25)
    dis = (a != b) & ok
    tex_list("cmp_both_confident.tex", "cmpBothConfident", cells(dis & both))
    bnd = (boundary_indicator(a) > 0) & dis
    flooded = yb & ~ya
    err_a, err_b = (a != ya) & ok, (b != yb) & ok
    tex_list("cmp_disagree.tex", "cmpDisagree", cells(dis & ~bnd))
    tex_list("cmp_disagree_boundary.tex", "cmpDisagreeBoundary", cells(bnd))
    tex_list("cmp_flooded.tex", "cmpFlooded", cells(dis & flooded))
    tex_list("cmp_error.tex", "cmpError", cells(dis & ~flooded))

    s57 = json.load(open(os.path.join(ROOT, "exp", "out", "exp57_summary.json")))["results"]
    c58 = json.load(open(os.path.join(ROOT, "exp", "out", "exp58_summary.json")))["results"]["geoid"]["change"]
    c59 = json.load(open(os.path.join(ROOT, "exp", "out", "exp59_summary.json")))["results"]["geoid"]["change"]
    g, w, cross = s57["geoid"], s57["geoid"]["where"], s57["cross_pairs"]["bolivia"]
    di = g["graded"]["disagreement_is"]
    macros = {
        "cmpEvent": EVENT, "cmpDatePre": DATES["pre"], "cmpDatePost": DATES["post"], "cmpChip": str(CHIP),
        "cmpChipDisagree": str(int(dis.sum())), "cmpChipWindows": str(int(ok.sum())), "cmpChipRate": f"{100 * dis.sum() / ok.sum():.0f}",
        "cmpChipFlooded": f"{100 * flooded[dis].mean():.0f}", "cmpChipErrA": f"{100 * err_a[dis].mean():.0f}", "cmpChipErrB": f"{100 * err_b[dis].mean():.0f}",
        "cmpPooledRate": f"{100 * g['disagreement_rate']:.1f}", "cmpPooledWindows": f"{g['n_windows']:,}", "cmpEvents": str(g["across_events"]["n_events"]),
        "cmpEventRateMedian": f"{100 * g['across_events']['rate']['q50']:.1f}",
        "cmpEnrBoundary": f"{w['boundary']['enrichment']:.1f}", "cmpEnrLowConf": f"{w['low_confidence']['enrichment']:.1f}",
        "cmpEnrUnstable": f"{w['unstable']['enrichment']:.1f}", "cmpEnrNdwi": f"{w['ndwi_ambiguous']['enrichment']:.1f}",
        "cmpShareBoundaryDis": f"{100 * w['boundary']['share_disagree']:.0f}", "cmpShareBoundaryAgr": f"{100 * w['boundary']['share_agree']:.1f}",
        "cmpFloodedAll": f"{100 * di['share_flooded_by_label']:.0f}", "cmpErrAAll": f"{100 * di['share_a_wrong_on_its_task']:.0f}", "cmpErrBAll": f"{100 * di['share_b_wrong_on_its_task']:.0f}",
        "cmpErrorsPhi": f"{g['graded']['crosstab_own_labels']['phi']:.2f}", "cmpEventPhiMedian": f"{g['across_events']['errors_phi']['q50']:.2f}",
        "cmpBothShare": f"{100 * c58['share_both_confident']:.0f}", "cmpBothFlooded": f"{100 * c58['flooded_both_confident']:.0f}",
        "cmpBothEventsW": str(c58["over_events"]["w"]), "cmpBothEventsL": str(c58["over_events"]["l"]),
        "cmpRuleFlooded": f"{100 * c59['flooded_rule']:.0f}", "cmpRuleShare": f"{100 * c59['share_change']:.0f}",
    }
    short = {"offsets 0 vs 2": "offsets", "backbones": "backbones", "sensors": "sensors", "finetune": "fine-tune"}
    P = np.array(cross["phi"])
    macros["cmpChipBothFlooded"] = f"{100 * flooded[dis & both].mean():.0f}" if (dis & both).any() else "nan"
    with open(os.path.join(OUT, "cmp_numbers.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\def\\{k}{{{v}}}\n")
        f.write("\\def\\cmpPhiNames{" + ", ".join(f"{i}/{short[n]}" for i, n in enumerate(cross["names"])) + "}\n")
        f.write("\\def\\cmpPhiCells{" + ", ".join(f"{i}/{j}/{P[i, j]:.2f}" for i in range(len(P)) for j in range(len(P))) + "}\n")
    print(f"decisions from the recomputed heads agree with the recorded masks on {agree_a:.3f} / {agree_b:.3f} of the valid windows")
    print(f"chip {CHIP} ({EVENT}): {int(ok.sum())} valid windows, {int(dis.sum())} disagree ({int(bnd.sum())} on a boundary of the S2 map), "
          f"flooded by the label {int((dis & flooded).sum())}, S2-head errors {int((dis & err_a).sum())}, S1-head errors {int((dis & err_b).sum())}")


if __name__ == "__main__":
    main()

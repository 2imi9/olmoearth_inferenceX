"""Rasters and numbers for the comparison diagram (compare_inferences.tex): the two-period, two-sensor design on one
GEOID-Flood chip (event EMSR275-1, tile EMSR275-1-4, rows 0-63, columns 256-319; exp60 chip 2293, the same chip as exp57's
2293, matched by its label grids). Three pieces of evidence, the Sentinel-2 composite and the Sentinel-1 pass of the
pre-event date and the Sentinel-1 pass after the event, read by exp60's heads into three decisions
(exp/out/exp60_masks.npz); the two differences that isolate one axis each, same date across sensors and same sensor across
dates, drawn on the scene; the label bridge. Imagery: rasters/cmp_chip.npz (S2 composite, post-event S1, label; exp55's
loader) and rasters/cmp_s1pre.npz (the pre-event S1 pass, same loader rules), both read once from the shard tree on the
cluster. Numbers come from exp/out/exp60_summary.json and are written as TeX macros so the figure cannot drift from
the ledger.

    uv run python docs/figures/tikz/make_compare_rasters.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_rasters import OUT, classmap, gray, rgb, save, tex_list  # noqa: E402

CHIP, EVENT = 2293, "EMSR275-1"
OFFSET = 1                                  # the 14 x 14 W1 windows are windows 1..14 of the 16-cell grid over the 64-px chip
FLOOD = (109, 40, 217)


def cells(mask):
    return [(r + OFFSET, c + OFFSET) for r, c in zip(*np.nonzero(mask))]


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "exp", "out", "exp60_masks.npz"))
    assert z["event"][CHIP] == EVENT
    ok, ya, yb = z["ok"][CHIP], z["y_permanent"][CHIP], z["y_after"][CHIP]
    dec = {k: z[k][CHIP] for k in ("A_s2pre", "A_s1pre", "B_s1post")}
    chip, pre = np.load(os.path.join(OUT, "cmp_chip.npz")), np.load(os.path.join(OUT, "cmp_s1pre.npz"))
    s2, s1post, lab, s1pre = chip["s2"], chip["s1"], chip["lab"], pre["s1pre"]
    date_pre = re.search(r"_pre_(\d{4})(\d{2})(\d{2})T", str(pre["name"])).groups()
    save("cmp_s2.png", rgb(s2, 64), 8)
    save("cmp_s1pre.png", np.repeat(gray(s1pre[0], 64)[..., None], 3, -1), 8)
    save("cmp_s1post.png", np.repeat(gray(s1post[0], 64)[..., None], 3, -1), 8)
    for k, m in dec.items():
        im = classmap(m.astype(float))
        im[~ok] = (200, 200, 200)
        save(f"cmp_{k}.png", im, 32)
    lab_img = classmap((lab == 1).astype(float))
    lab_img[lab == 2] = FLOOD
    lab_img[lab == 255] = (200, 200, 200)
    save("cmp_label.png", lab_img, 8)
    # the fourth cell: WorldFloods v2's Sentinel-2 L1C scene of the same AoI two days after the radar pass (rasters/cmp_s2post.npz,
    # cut on the cluster onto this chip's grid; bands B1..B12 with B8A ninth, then two ancillary bands; gt band 1: 2 = cloud).
    post = np.load(os.path.join(OUT, "cmp_s2post.npz"))
    x = np.clip(post["s2"][[3, 2, 1]] / 6000.0, 0, 1)                    # fixed stretch: cloud stays white
    save("cmp_s2post.png", (x.transpose(1, 2, 0) * 255).astype(np.uint8), 8)
    cloud_share = float((post["gt"][0] == 2).mean())
    date_post_s2 = str(post["s2_date"])[:10]
    flooded = yb & ~ya
    d_sens = (dec["A_s2pre"] != dec["A_s1pre"]) & ok
    d_time = (dec["A_s1pre"] != dec["B_s1post"]) & ok
    tex_list("cmp_sens_diff.tex", "cmpCellsSensDiff", cells(d_sens))
    tex_list("cmp_time_diff.tex", "cmpCellsTimeDiff", cells(d_time))
    tex_list("cmp_time_flooded.tex", "cmpCellsTimeFlooded", cells(d_time & flooded))
    tex_list("cmp_time_error.tex", "cmpCellsTimeError", cells(d_time & ~flooded))
    tex_list("cmp_sens_error.tex", "cmpCellsSensError", cells(d_sens))

    R = json.load(open(os.path.join(ROOT, "exp", "out", "exp60_summary.json")))["results"]["pairs"]
    macros = {
        "cmpEvent": EVENT, "cmpDatePre": "-".join(date_pre), "cmpDatePost": "2018-03-22", "cmpChip": str(CHIP),
        "cmpWindows": str(int(ok.sum())),
        "cmpSensN": str(int(d_sens.sum())), "cmpSensFlooded": f"{100 * flooded[d_sens].mean():.0f}" if d_sens.any() else "0",
        "cmpTimeN": str(int(d_time.sum())), "cmpTimeFlooded": f"{100 * flooded[d_time].mean():.0f}" if d_time.any() else "0",
        "cmpTimeError": f"{100 * (~flooded)[d_time].mean():.0f}" if d_time.any() else "0",
        "cmpSensPooledRate": f"{100 * R['sensor_only']['disagreement_rate']:.1f}", "cmpSensPooledFlooded": f"{100 * R['sensor_only']['flooded_share']:.1f}",
        "cmpTimePooledRate": f"{100 * R['time_only']['disagreement_rate']:.1f}", "cmpTimePooledFlooded": f"{100 * R['time_only']['flooded_share']:.0f}",
        "cmpTimePooledPreDeparts": f"{100 * R['time_only']['err_a_share']:.0f}",
        "cmpEvents": "55", "cmpDatePostS2": date_post_s2, "cmpPostS2Cloud": f"{100 * cloud_share:.0f}",
    }
    with open(os.path.join(OUT, "cmp_numbers.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\def\\{k}{{{v}}}\n")
    print(f"chip {CHIP} ({EVENT}, pre {'-'.join(date_pre)}): {int(ok.sum())} valid windows; same date across sensors {int(d_sens.sum())} differ ({macros['cmpSensFlooded']}% flooded); "
          f"same sensor across dates {int(d_time.sum())} differ ({macros['cmpTimeFlooded']}% flooded, {macros['cmpTimeError']}% one head off its label)")


if __name__ == "__main__":
    main()

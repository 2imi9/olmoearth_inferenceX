"""README example 1: the demo's Dynamic World tile beside its Sentinel-2 image; the 5% flagged first and the real errors."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
D, OUT = sys.argv[1], sys.argv[2]
DW = np.array([(65, 155, 223), (57, 125, 73), (136, 176, 83), (122, 135, 198), (228, 150, 53), (223, 195, 90),
               (196, 40, 27), (165, 155, 143), (179, 159, 225)]) / 255
NAMES = ["water", "trees", "grass", "flooded veg.", "crops", "shrub", "built", "bare", "snow"]
probs = np.load(f"{D}/demo/oe_inferencex_demo/sample_probabilities.npy")
truth = np.load(f"{D}/demo/oe_inferencex_demo/sample_truth.npy")
pred = probs.argmax(0)
lab = truth >= 0
wrong = lab & (pred != truth)
flag = np.zeros_like(lab)
with open(f"{D}/demo/oe_inferencex_demo/audit/review_set_05pct.csv") as f:
    for r in csv.DictReader(f):
        flag[int(r["window_row"]), int(r["window_col"])] = True
share_flag_wrong = (wrong & flag).sum() / (lab & flag).sum()
share_map_wrong = wrong.sum() / lab.sum()
s2 = np.load(f"{D}/data_peru/peru_s2.npz")
rgb = np.stack([s2["B04"], s2["B03"], s2["B02"]], -1).astype(float)
lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
mapimg = DW[pred]
faded = 0.45 * mapimg + 0.55
err = faded.copy(); err[wrong] = (0.91, 0.16, 0.54)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(1, 4, figsize=(16, 4.6), dpi=170)
titles = ["Sentinel-2 image, 9 June 2019", "Dynamic World map", "5% to check first (no labels)",
          f"{100 * share_flag_wrong:.0f}% of those are really wrong"]
for a, img, t in zip(ax, [rgb, mapimg, faded, err], titles):
    a.imshow(img, interpolation="nearest"); a.set_title(t, fontsize=12.5, pad=8); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
for a in ax[2:]:
    rr, cc = np.nonzero(flag)
    for r, c in zip(rr, cc):
        a.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="#111111", lw=0.55))
present = [i for i in range(9) if (pred == i).mean() > 0.01]
h1 = [Patch(fc=DW[i], label=NAMES[i]) for i in present]
h2 = [Patch(fc="none", ec="#111111", label="flagged window"), Patch(fc=(0.91, 0.16, 0.54), label="really wrong")]
fig.legend(handles=h1 + h2, loc="lower center", ncol=len(h1) + 2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.09, f"Southern Peru, 5 km across. A random 5% would be wrong {100 * share_map_wrong:.0f}% of the time.",
         ha="center", fontsize=11, color="#444444")
plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.08)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"flagged wrong {share_flag_wrong:.3f}, map wrong {share_map_wrong:.3f}, flagged {flag.sum()}")

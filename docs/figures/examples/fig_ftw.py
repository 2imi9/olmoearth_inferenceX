"""README example 3: Fields of the World in Croatia. FTW's 2nd-edition field map against the PRUE B3 model at one test
chip, read with exp110's own window code. Label-free on the chip, as exp110's licence rule asks (aggregates only): the
panels show the imagery, the two maps and where they differ; who is right is quoted only as Croatia's aggregate from
exp/out/exp110_summary.json. The chip is the median by differing windows (at least 64), chosen without labels."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
REC, FTW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(REC, "exp"))
import exp110_ftw_field_map as e
C = "croatia"
mz = np.load(f"{FTW}/v1/{C}/map2e_2023_on_mask_grid_{C}.npz")
bz = np.load(f"{FTW}/v1/{C}/b3_on_mask_grid_{C}.npz")
ids = mz["chip_ids"].astype(str)
pa = mz["map_field"].astype(np.float64) + mz["map_boundary"]
pb = bz["b3_field"].astype(np.float64) + bz["b3_boundary"]
a = e.read_extent(pa, e.PATCH, chunk=64)["cls"]
b = e.read_extent(pb, e.PATCH, chunk=64)["cls"]
diff = a != b
dcount = diff.reshape(len(ids), -1).sum(1)
ok = np.nonzero(dcount >= 64)[0]
i = ok[np.argsort(dcount[ok])[len(ok) // 2]]
import rasterio
with rasterio.open(f"{FTW}/v1/{C}/s2_images/window_a/{ids[i]}.tif") as f:
    s2 = f.read().astype(float)                      # B04, B03, B02, B08
    t, lat = f.transform, 0.5 * (f.bounds.bottom + f.bounds.top)
# the chips and the maps on their grid are stored south-up (row 0 = south; FTW plan page of the record): flip them
# north-up, and draw each pixel at its ground shape (EPSG:4326, so about 6 m N-S by 4 m E-W here)
south_up = t.e > 0
aspect = abs(t.e) / (abs(t.a) * np.cos(np.radians(lat)))
up = (lambda x: x[::-1]) if south_up else (lambda x: x)
s2 = s2[:, ::-1] if south_up else s2
rgb = np.stack([s2[0], s2[1], s2[2]], -1)
lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.85
FIELD, OTHER = np.array([0.30, 0.62, 0.30]), np.array([0.94, 0.91, 0.84])
def fieldmap(p): return np.where((p > 0.5)[..., None], FIELD, OTHER)
di = up(diff[i])
P = e.PATCH
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(1, 4, figsize=(16, 6.4), dpi=150)
faded = 0.45 * rgb + 0.55
for k, (img, ttl) in enumerate([(rgb, "Sentinel-2 image, 2023"), (fieldmap(up(pa[i])), "FTW 2nd-edition field map"),
                                (fieldmap(up(pb[i])), "PRUE B3 field map"),
                                (faded, f"Where they differ ({int(di.sum())} windows, no labels)")]):
    ax[k].imshow(img, interpolation="nearest", aspect=aspect); ax[k].set_title(ttl, fontsize=12.5, pad=8)
    ax[k].set_xticks([]); ax[k].set_yticks([])
    for s in ax[k].spines.values(): s.set_visible(False)
for (r, c) in zip(*np.nonzero(di)):
    ax[3].add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, fc="#f28e2b", ec="white", lw=0.4, alpha=0.9))
s = json.load(open(f"{REC}/exp/out/exp110_summary.json"))
w4 = s["populations"][C]["arm4"]; pr = s["verdicts"]["P4b"]["p_right"][C]
h = [Patch(fc=FIELD, label="field"), Patch(fc=OTHER, ec="#bbbbbb", label="not field"),
     Patch(fc="#f28e2b", label="the maps differ")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, -0.07, "One test chip in Croatia, about 1.5 km N-S by 1.1 km E-W, north up.", ha="center", fontsize=11, color="#444444")
print(f"croatia: differ {100 * w4['D'] / w4['N']:.1f}% of windows, FTW agrees at {100 * w4['K_A'] / w4['D']:.1f}%, P(names FTW | 100 labels) {100 * pr:.1f}%")
plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.08)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("chip", ids[i], "differ", int(di.sum()), "median of", len(ok), "south-up", bool(south_up), "aspect", round(aspect, 3))

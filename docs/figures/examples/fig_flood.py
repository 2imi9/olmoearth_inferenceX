"""README example 2, context: Lake Shkoder before and during the March 2018 flood (Sentinel-2) and in radar (Sentinel-1),
8 km across, the box marking the 640 m patch the comparison figure's maps cover."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
D, OUT = sys.argv[1], sys.argv[2]
m = json.load(open(f"{D}/context.json"))
def rgb(tag):
    z = np.load(f"{D}/{tag}.npz"); x = np.stack([z["B04"], z["B03"], z["B02"]], -1).astype(float)
    return np.clip(x / 1800.0, 0, 1) ** 0.75
vh = np.load(f"{D}/s1_during.npz")["vh"].astype(float)
db = 10 * np.log10(np.maximum(vh, 1e-6))
s1 = np.clip((db + 28) / 20, 0, 1)
N = 800
c0 = (m["chip_location"]["x0"] - (m["center"][0] - m["half_m"])) / 10
r0 = ((m["center"][1] + m["half_m"]) - m["chip_location"]["y0"]) / 10
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), dpi=130)
panels = [(rgb("s2_before"), None, "Before: Sentinel-2, 2 April 2017"),
          (rgb("s2_during"), None, "Flood: Sentinel-2, 28 March 2018"),
          (s1, "gray", "Flood: Sentinel-1 radar, 11 March 2018")]
for a, (img, cmap, t) in zip(ax, panels):
    a.imshow(img, cmap=cmap, interpolation="bilinear"); a.set_title(t, fontsize=12.5, pad=8)
    a.add_patch(Rectangle((c0, r0), 64, 64, fill=False, ec="#ff8c00", lw=2.2))
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
fig.text(0.5, 0.02, "Lake Shkodër, Albania, 8 km across. The orange box is the 640 m patch where four water maps are compared below.",
         ha="center", fontsize=11, color="#444444")
plt.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.08)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("box at col,row", c0, r0)

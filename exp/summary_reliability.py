"""Reliability diagram for the fine-tuned AWF model (exp21), from committed
artifacts only: exp/out/exp21_summary.json. Writes exp/out/exp21_reliability.png.

    uv run --extra geo python exp/summary_reliability.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
with open(os.path.join(OUT, "exp21_summary.json")) as fh:
    d = json.load(fh)["crops"]["16"]

bins = d["reliability"]  # [lo, hi, n, mean_confidence, accuracy]
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot([0, 1], [0, 1], "--", color="#444", lw=1, label="perfect calibration")
for lo, hi, n, conf, acc in bins:
    ax.bar((lo + hi) / 2, acc, width=hi - lo, color="#3b78b8", edgecolor="white", alpha=0.9)
    ax.text((lo + hi) / 2, acc + 0.02, str(n), ha="center", va="bottom", fontsize=8.5, color="#333")
lo, hi, n, conf, acc = bins[-1]
ax.annotate(f"{n} points: claims {conf:.2f}, delivers {acc:.2f}",
            xy=((lo + hi) / 2, acc), xytext=(0.05, 0.80), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_xlabel("top-1 probability (bin)")
ax.set_ylabel("accuracy in bin")
ax.set_title(f"OlmoEarth-v1-FT-AWF-Base on 344 held-out expert points\n"
             f"accuracy {d['accuracy']:.3f}, ECE {d['ece_10bins']:.3f}; counts above bars", fontsize=10.5)
ax.legend(loc="upper left", fontsize=9, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25)
fig.tight_layout()
dest = os.path.join(OUT, "exp21_reliability.png")
fig.savefig(dest, dpi=170)
print("wrote", dest)

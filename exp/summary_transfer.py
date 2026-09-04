"""Summary figure: does an audit signal's advantage survive a change of truth?

For each signal, the share of units where it ranks errors better than the
model's own confidence, under a weak reference (ESA WorldCover, the 27
rule-selected scenes) and under hand labels (Sen1Floods11 Bolivia). Every
signal that beats confidence on the left loses to it on the right; that gap
is the repository's open question (docs/results/comparisons.md section 3).

Reads committed artifacts only - no network, no encoder, no torch:
exp13_summary.json, exp14_boundary_ablation.csv, exp17_internal_evidence.csv,
exp18_sen1floods.csv. Writes exp/out/summary_transfer.png.

    uv run --extra geo python exp/summary_transfer.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def head_to_head(filename, signal, baseline):
    """Scenes where `signal` scores below `baseline` (lower E-AURC is better)."""
    wins = losses = 0
    with open(os.path.join(OUT, filename)) as fh:
        for row in csv.DictReader(fh):
            wins += float(row[signal]) < float(row[baseline])
            losses += float(row[signal]) > float(row[baseline])
    return wins, losses


with open(os.path.join(OUT, "exp13_summary.json")) as fh:
    exp13 = json.load(fh)["signals"]
with open(os.path.join(OUT, "exp18_sen1floods.csv")) as fh:
    bolivia = next(iter(csv.DictReader(fh)))

# (label, WorldCover wins/losses, the exp18 column holding "W/L" on hand labels)
SIGNALS = [
    ("E_system tiling instability",
     (exp13["tile-phase (aligned)"]["w"], exp13["tile-phase (aligned)"]["l"]),
     "tile-phase (aligned)|W/L"),
    ("band-set disagreement",
     head_to_head("exp17_internal_evidence.csv", "band-set disagreement", "baseline"),
     "band-set disagreement|W/L"),
    ("prediction boundary",
     head_to_head("exp14_boundary_ablation.csv", "pred-boundary", "baseline"),
     "boundary indicator|W/L"),
    ("pixel control (no model)",
     (exp13["control"]["w"], exp13["control"]["l"]),
     "control NDWI gradient|W/L"),
    ("E_case cross-model",
     (exp13["E_case"]["w"], exp13["E_case"]["l"]),
     "E_case |Nano-Base||W/L"),
    ("E_dist embedding distance",
     (exp13["E_dist"]["w"], exp13["E_dist"]["l"]),
     "E_dist knn-to-train|W/L"),
]

# Nudges (in points) that keep near-coincident labels legible.
LEFT_DY = {"pixel control (no model)": 8, "E_dist embedding distance": -8}
RIGHT_DY = {"pixel control (no model)": -7, "E_case cross-model": 7}

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.axhspan(0, 0.5, color="#c62828", alpha=0.05, zorder=0)
ax.axhline(0.5, color="#444", lw=1.2, ls="--", zorder=2)

cmap = plt.get_cmap("tab10")
for i, (name, (wins, losses), column) in enumerate(SIGNALS):
    hand_w, hand_l = (int(v) for v in bolivia[column].split("/"))
    weak, hand = wins / (wins + losses), hand_w / (hand_w + hand_l)
    color = cmap(i)
    ax.plot([1, 2], [weak, hand], "-o", color=color, lw=2.2, ms=8, zorder=3)
    ax.annotate(f"{name}   {weak:.0%}", (1, weak), textcoords="offset points",
                xytext=(-12, LEFT_DY.get(name, 0)), ha="right", va="center",
                fontsize=9.5, color=color)
    ax.annotate(f"{hand:.0%}", (2, hand), textcoords="offset points",
                xytext=(10, RIGHT_DY.get(name, 0)), ha="left", va="center",
                fontsize=9.5, color=color)

ax.set_xlim(0.32, 2.22)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([1, 2])
ax.set_xticklabels(["ESA WorldCover\n27 river scenes (exp13, exp14, exp17)",
                    "hand labels\nSen1Floods11 Bolivia (exp18)"], fontsize=10)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("share of units where the signal ranks errors\nbetter than the model's own confidence",
              fontsize=10)
ax.set_title("Every audit signal that beat confidence against a weak reference\n"
             "loses to it against hand labels", fontsize=12, pad=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

dest = os.path.join(OUT, "summary_transfer.png")
fig.savefig(dest, dpi=170)
print("wrote", dest)

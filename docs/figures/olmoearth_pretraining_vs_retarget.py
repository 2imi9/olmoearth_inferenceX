"""Renders docs/figures/olmoearth_pretraining_vs_retarget.png in the grammar of OlmoEarth's Figure 3 (Ai2,
arXiv:2511.13655; redrawn, not copied): top row, what pretraining does; bottom row, what we do at inference
(issues #10, #11). Facts cite exp32 (target space), exp28 (shipped residual) and the harness's test."""
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 9.6), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 96); ax.axis("off")
INK, MUTED = "#1f2937", "#6b7280"
PINK, BLUEt, GREENt, ORANGEt = "#f9a8d4", "#93c5fd", "#86efac", "#fdba74"      # token colours as in Ai2's figure
LIGHT, DARK, YELLOW, RED, MAG = "#e5e7eb", "#9ca3af", "#fde68a", "#dc2626", "#ec4899"
VIOLET, VIOLET_L, AMBER, AMBER_L, EMERALD, EMERALD_L, CROSS = "#6d28d9", "#ede9fe", "#b45309", "#fef3c7", "#047857", "#d1fae5", "#b91c1c"


def text(x, y, t, fs=9, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.3)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=color, lw=lw))


def block(x, y, w, h, t, fc=LIGHT, ec=INK, fs=9.5, color=INK):
    """Ai2's pentagon block (flat left, arrow-tip right)."""
    tip = min(4.0, w * 0.25)
    ax.add_patch(Polygon([(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)], closed=True, fc=fc, ec=ec, lw=1.1))
    text(x + (w - tip) / 2, y + h / 2, t, fs=fs, color=color)


def tokens(x, y, states, size=3.0, gap=0.8):
    """A column of token squares; states: list of (colour, masked)."""
    for i, (c, masked) in enumerate(states):
        yy = y + (len(states) - 1 - i) * (size + gap)
        ax.add_patch(Rectangle((x, yy), size, size, fc="white" if masked else c, ec=c if masked else INK, lw=1.4 if masked else 0.8))


def rbox(x, y, w, h, fc, ec, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw, ls=ls))


def loss(x, y, w, h, t, fc=MAG):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec=INK, lw=1.0, ls=(0, (2, 1.5))))
    text(x + w / 2, y + h / 2, t, fs=9, color="white", weight="bold")


# ================================================================== top row: pretraining (after Ai2 Figure 3)
ax.add_patch(Rectangle((1, 50), 198, 44, fc="#fafafa", ec=MUTED, lw=0.8))
text(3, 91.5, "What OlmoEarth v1 trains (after Ai2, Figure 3)", fs=11, weight="bold", ha="left")
text(9, 74, "Sentinel-2\nSentinel-1\nLandsat", fs=8.5)
text(10.5, 64.5, "+ maps: WorldCover,\nOSM, CDL, WorldCereal,\nSRTM, canopy height", fs=7.4, color=CROSS)
block(20, 68, 16, 12, "Crop +\nPatchify", fc="#dbeafe")
tokens(38, 66, [(PINK, False), (BLUEt, False), (GREENt, False), (ORANGEt, False)])
block(44, 68, 18, 12, "Project +\nMasking")
tokens(64, 66, [(PINK, False), (BLUEt, False), (GREENt, True), (ORANGEt, True)])
block(70, 68, 18, 12, "ViT\nEncoder")
tokens(90, 66, [(PINK, False), (BLUEt, False), (GREENt, True), (ORANGEt, True)])
block(96, 68, 18, 12, "ViT\nDecoder", fc=DARK)
tokens(116, 66, [(PINK, False), (BLUEt, False), (GREENt, False), (ORANGEt, False)])
text(118, 63, "predicted targets", fs=7.5, color=MUTED)
# random projection + stop grad -> targets
ax.add_patch(Polygon([(42, 53.5), (63, 53.5), (66, 58.5), (45, 58.5)], closed=True, fc=YELLOW, ec=INK, lw=1.0))
text(54, 56, "Random Project (fixed)", fs=8.5)
tokens(68, 53, [(PINK, False), (BLUEt, False), (GREENt, False), (ORANGEt, False)], size=2.2, gap=0.5)
rbox(74, 54, 14, 5, "white", RED, lw=1.8); text(81, 56.5, "Stop Grad", fs=8.5)
arrow(38, 66, 45, 59, color=MUTED)
arrow(88.5, 56.5, 122, 56.5, color=MUTED)
loss(122, 53.5, 24, 6.5, "Patch Contrastive Loss")
arrow(120.5, 66, 134, 60.5, color=MUTED)
loss(150, 53.5, 27, 6.5, "Instance Contrastive Loss")
text(163.5, 51.2, "two views, batch negatives", fs=7.3, color=MUTED)
arrow(119, 68, 156, 60.5, color=MUTED, style="-")
# annotations of what exp32 found
rbox(128, 70, 68, 20, "white", CROSS, lw=1.2)
text(162, 87, "what the code and checkpoint say (exp32)", fs=8.5, color=CROSS, weight="bold")
text(162, 79.2, "target encoder = untouched random init, EMA 1.0, exit at block 0:\n"
     "a target is a fixed random projection of the raw patch, never normalised\n"
     "on real Sentinel-2 scenes: effective rank 2, random-pair cosine 0.99\n"
     "loss: patch discrimination at tau 0.1 over the sample's masked tokens\n"
     "maps are decode-only targets: WorldCover is inside the objective (issue #2)", fs=7.4)

# ================================================================== bottom row: inference-time re-targeting (ours)
ax.add_patch(Rectangle((1, 2), 198, 46, fc="#fafafa", ec=MUTED, lw=0.8))
text(3, 45.5, "What we do at inference (ours): keep the objective, replace the target", fs=11, weight="bold", ha="left")
text(9, 30, "Sentinel-2 only\n128 px, 12 bands\n27 river scenes\n441 flood tiles", fs=8)
block(20, 24, 16, 12, "Crop +\nPatchify", fc="#dbeafe")
tokens(38, 22, [(PINK, False), (PINK, False), (PINK, True), (PINK, False)])
text(40, 19.5, "25% hidden\nK = 8 masks", fs=7.3, color=MUTED)
block(44, 24, 18, 12, "ViT Encoder\n(frozen, fp32)")
tokens(64, 22, [(PINK, False), (PINK, False), (PINK, True), (PINK, False)])
block(70, 24, 20, 12, "Predictor\n2 layers, w 128\nlabel-free", fc=VIOLET_L, ec=VIOLET, fs=8.5)
tokens(92, 22, [(PINK, False), (PINK, False), (PINK, False), (PINK, False)])
text(94, 19.5, "predicted\ntargets", fs=7.3, color=MUTED)
# our target: whiten / cluster instead of random project
ax.add_patch(Polygon([(44, 8), (62, 8), (65, 12), (47, 12)], closed=True, fc=VIOLET_L, ec=VIOLET, lw=1.4))
text(54.5, 10, "Whiten  /  Cluster", fs=8.5, color=VIOLET, weight="bold")
text(54.5, 4.8, "PCA-whitened top-64 (data2vec-style, exp33)\nk-means of the features (HuBERT-style, next)", fs=7.2, color=VIOLET)
tokens(68, 7, [(PINK, False), (PINK, False), (PINK, False), (PINK, False)], size=2.2, gap=0.5)
rbox(74, 8, 16, 5, "white", VIOLET, lw=1.6); text(82, 10.5, "fitted on training\nunits only", fs=7.2, color=VIOLET)
arrow(64, 22, 47, 12.5, color=VIOLET)
arrow(90.5, 10.5, 106, 10.5, color=VIOLET)
rbox(107, 7, 22, 7, AMBER_L, AMBER, lw=1.5); text(118, 10.5, "Residual per patch", fs=9, color=AMBER, weight="bold")
arrow(96.5, 22, 112, 14.5, color=AMBER)
text(118, 3.8, r"$\|\hat{z}_T - z_T\|^2$, mean over the masks that hid the patch", fs=7.2, color=AMBER)
arrow(129.5, 10.5, 137, 10.5, color=AMBER, lw=1.5)
# the signal and its test
rbox(138, 4, 58, 40, EMERALD_L, EMERALD, lw=1.6)
text(167, 40, "signal for where the map is wrong, tested like every signal", fs=9.5, color=EMERALD, weight="bold")
text(167, 32.5, "scored on identical patches against the model's confidence (-|logit|),\n"
     "tile-phase, the boundary indicator, the NDWI-gradient control,\nand the constant, S2-variance and NDWI-level controls", fs=7.6)
text(167, 23.5, "excess AURC per scene or tile, tie-aware; expert labels grade, never train", fs=7.6, color=MUTED)
text(167, 16.5, "preregistered: primary = residual; U+ = midrank mean of confidence and residual;\n"
     "one vote per river, one-sided exact sign test (7/8 rivers: p = 0.035)", fs=7.6)
text(167, 9, "null hypothesis: the residual is a boundary detector\n(wins against WorldCover, loses on hand labels)", fs=7.6, color=EMERALD)
text(3, 0.3, "olmoearth_inferenceX; top row redrawn after Figure 3 of the OlmoEarth v1 paper (arXiv:2511.13655); exp28: the shipped decoder's residual ranks errors at chance", fs=7.3, color=MUTED, ha="left")
fig.savefig("docs/figures/olmoearth_pretraining_vs_retarget.png", bbox_inches="tight", facecolor="white")

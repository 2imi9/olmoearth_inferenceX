"""Renders docs/figures/latent_mim_retarget_idea.png: the latent-MIM residual as a label-free error signal, re-targeted
post hoc on the frozen encoder (issues #10 and #11), and the test every signal takes. Numbers cite exp32 and exp28."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=200)
ax.set_xlim(0, 135); ax.set_ylim(0, 64); ax.axis("off")
GREY, BLUE, GREEN, INK, MUTED = "#9aa0a6", "#1f6feb", "#1a7f37", "#24292f", "#57606a"


def box(x, y, w, h, text, ec=INK, fc="white", fs=9.3, color=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", ec=ec, fc=fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.4)


def arrow(x0, y0, x1, y1, color=INK, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


for x, t in ((14, "OlmoEarth inference (frozen, fp32)"), (63, "Target for the masked-token residual"), (115, "The test every signal takes")):
    ax.text(x, 61, t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
box(2, 42, 24, 9, "Sentinel-2 window\n(12 bands, 10 m)")
arrow(14, 42, 14, 36.5)
box(2, 27, 24, 9, "OlmoEarth v1 encoder\n(frozen)")
arrow(14, 27, 14, 21.5)
box(2, 12, 24, 9, "patch tokens\n768-d per 4-px patch")
box(33, 41, 60, 12, "what pretraining used\nfrozen random target encoder, never updated\n"
    "target = random projection of the raw patch, not normalised\n"
    "measured: effective rank 2, random-pair cosine 0.99  (exp32)", ec=GREY, fc="#f6f8fa", color=MUTED)
box(33, 32, 60, 6, "the shipped decoder's residual carries nothing about errors  (exp28)", ec=GREY, fc="#f6f8fa", color=MUTED, fs=9)
box(33, 11, 60, 16, "what we change: post hoc, label-free, on the frozen tokens\n"
    "re-target with a normalised target (whitened, data2vec-style)\n"
    "or a discrete one (clusters of the model's own features, HuBERT-style)\n"
    "train a small predictor to fill masked patches\n"
    "score each patch by its residual", ec=BLUE, fc="#ddf4ff", color=INK, lw=1.8)
ax.text(63, 8.3, "no new pretraining; recipes from Latent MIM (Wei et al.), data2vec, HuBERT", ha="center", fontsize=8.5, color=BLUE, style="italic")
arrow(26, 16.5, 33, 19, color=BLUE, lw=1.8)
arrow(26, 16.5, 33, 45, color=GREY)
box(100, 35, 32, 17, "signal: where is the map wrong?\n\nscored on identical patches\nagainst the model's own confidence\nand a no-model pixel control", ec=GREEN, fc="#dafbe1", lw=1.8)
box(100, 12, 32, 16, "expert labels grade the signal,\nnever train it\n\npreregistered: primary score,\ncombination, one vote per river", ec=GREEN, fc="#dafbe1", lw=1.8)
arrow(93, 19, 100, 22, color=BLUE, lw=1.8)
arrow(116, 35, 116, 28.5, color=GREEN)
ax.text(116, 8.3, "so far nothing beats confidence on expert labels;\nthis is the next candidate", ha="center", va="top", fontsize=8.5, color=GREEN, style="italic")
ax.text(1, 1.0, "olmoearth_inferenceX: the latent-MIM residual as a label-free error signal, re-targeted post hoc", fontsize=8, color=MUTED)
fig.savefig("docs/figures/latent_mim_retarget_idea.png", bbox_inches="tight", facecolor="white")

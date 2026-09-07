"""Renders docs/figures/latent_mim_retarget_idea.png: the re-targeted latent-MIM residual as a label-free error signal
(issues #10, #11) in the grammar of Latent MIM's Figure 1 (Wei et al., arXiv:2407.15837), redrawn with the repo's
details: exp32 (target space), exp28 (shipped residual), exp33 (predictor) and the test every signal takes."""
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 8.4), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 84); ax.axis("off")
INK, MUTED, PANEL = "#1f2937", "#6b7280", "#f5f5f4"
TEAL, TEAL_L = "#0f766e", "#ccfbf1"
STONE, STONE_L, CROSS = "#78716c", "#e7e5e4", "#b91c1c"
VIOLET, VIOLET_L = "#6d28d9", "#ede9fe"
AMBER, AMBER_L = "#b45309", "#fef3c7"
EMERALD, EMERALD_L = "#047857", "#d1fae5"
NAVY = "#1e3a8a"


def text(x, y, t, fs=9, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.35)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def rbox(x, y, w, h, fc, ec, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def encoder(x, y, w, h, t, fc, ec, color=INK):
    ax.add_patch(Rectangle((x, y), w - h / 2, h, fc=fc, ec="none"))
    ax.add_patch(Wedge((x + w - h / 2, y + h / 2), h / 2, -90, 90, fc=fc, ec="none"))
    ax.plot([x, x, x + w - h / 2], [y + h, y, y], color=ec, lw=1.3)
    ax.plot([x, x + w - h / 2], [y + h, y + h], color=ec, lw=1.3)
    th = np.linspace(-np.pi / 2, np.pi / 2, 50)
    ax.plot(x + w - h / 2 + h / 2 * np.cos(th), y + h / 2 + h / 2 * np.sin(th), color=ec, lw=1.3)
    text(x + (w - h / 2) / 2 + 1.5, y + h / 2, t, fs=10.5, color=color)


def callout(x, y, k, r=2.1, fs=10):
    ax.add_patch(Circle((x, y), r, fc=NAVY, ec="none"))
    text(x, y, str(k), fs=fs, color="white", weight="bold")


def panel(x, y, w, h, title=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=INK, lw=1.0))
    if title:
        text(x + w / 2, y + h - 2.5, title, fs=10.5, weight="bold")


def tokens(x, y, k, fc):
    for i in range(k):
        ax.add_patch(FancyBboxPatch((x, y + i * 3.2), 2.6, 2.6, boxstyle="round,pad=0.1", fc=fc, ec=MUTED, lw=0.5))


# ------------------------------------------------------------------ scene and patchify
ax.add_patch(Rectangle((2, 40), 16, 16, fc="#d9f99d", ec="#a3b18a", lw=1))
ax.add_patch(Rectangle((6, 43), 9, 11, fill=False, ec=INK, lw=1, ls=(0, (2, 2))))
text(10, 37.5, "Sentinel-2 L2A\n128 px, 12 bands", fs=8, color=MUTED)
arrow(18.5, 48, 21.5, 48)
panel(22, 12, 36, 66, "Patchify")
text(40, 70.5, "4-px patches -> 768-d pooled tokens", fs=8.5)
rng = np.random.default_rng(0)
hidden = rng.random((6, 6)) < 0.25
for i in range(6):
    for j in range(6):
        ax.add_patch(FancyBboxPatch((28.5 + j * 4.0, 41 + (5 - i) * 4.0), 3.3, 3.3, boxstyle="round,pad=0.15,rounding_size=0.5",
                                    fc=("#a3e635" if hidden[i, j] else TEAL_L), ec=MUTED, lw=0.5))
ax.add_patch(FancyBboxPatch((29.5, 35.6), 1.8, 1.8, boxstyle="round,pad=0.1", fc=TEAL_L, ec=MUTED, lw=0.5))
text(32.5, 36.5, "visible", fs=8.5, ha="left")
ax.add_patch(FancyBboxPatch((41, 35.6), 1.8, 1.8, boxstyle="round,pad=0.1", fc="#a3e635", ec=MUTED, lw=0.5))
text(44, 36.5, "hidden 25%", fs=8.5, ha="left")
text(40, 31, "K = 8 masks: 2 rounds of 4 disjoint quarters,\nevery patch hidden exactly twice", fs=7.5, color=MUTED)
text(40, 22, "27 WorldCover river scenes\n441 Sen1Floods11 Bolivia tiles (hand labels)", fs=7.5, color=MUTED)

# ------------------------------------------------------------------ encoders
panel(62, 12, 42, 66)
callout(64.5, 75.5, 1)
encoder(66, 56, 28, 12, "Online\nEncoder", TEAL_L, TEAL)
text(80, 71, "OlmoEarth v1 Base, frozen, fp32", fs=8.5, color=TEAL)
tokens(96, 58, 2, TEAL_L); text(97.5, 55, r"$z_V$", fs=10)
encoder(66, 32, 28, 12, "Target\nEncoder", STONE_L, STONE, color=STONE)
ax.plot([66, 92], [32.5, 43.5], color=CROSS, lw=2); ax.plot([66, 92], [43.5, 32.5], color=CROSS, lw=2)
tokens(96, 32, 3, STONE_L); text(97.5, 45, r"$z_T$", fs=10, color=STONE)
text(83, 26.5, "copy at random init, EMA 1.0: never updated\nexit at block 0: target = random projection of the raw patch\n"
     "effective rank 2, random-pair cosine 0.99  (exp32)", fs=7.2, color=CROSS)
text(83, 17.5, "loss: patch discrimination, tau 0.1,\nnegatives = the sample's other masked tokens", fs=7.2, color=MUTED)

# ------------------------------------------------------------------ predictor panel
panel(108, 34, 44, 44)
text(123, 74.5, "Target locations", fs=9.5); arrow(123, 72.5, 123, 67)
rbox(112, 54, 22, 12, TEAL_L, TEAL)
text(123, 62.5, "Predictor", fs=9.5, weight="bold")
text(123, 57.8, "2 transformer layers, width 128\nlearned mask token, MSE\n400 steps, label-free", fs=7.2)
arrow(99, 62, 111.5, 60, color=TEAL)
tokens(137, 57, 3, STONE_L); text(138.5, 54, r"$\hat{z}_T$", fs=10)
rbox(138, 40, 12, 7, AMBER_L, AMBER, lw=1.5)
text(144, 43.5, "Residual", fs=10, color=AMBER, weight="bold")
arrow(139, 56, 143, 47.5, color=AMBER)
text(131, 36.6, r"$\|\hat{z}_T - z_T\|^2$ per patch, mean over the masks that hid it", fs=7.2, color=AMBER)
callout(149.5, 75.5, 2)
text(142.5, 70, "shipped decoder's residual\nis at chance  (exp28)", fs=7.2, color=CROSS)
res = rng.random((4, 4)) * 0.3; res[1, :] += 0.55; res[2, 1] += 0.4
cm = plt.get_cmap("YlOrBr")
for i in range(4):
    for j in range(4):
        ax.add_patch(Rectangle((112 + j * 2.4, 39 + (3 - i) * 2.4), 2.4, 2.4, fc=cm(min(res[i, j], 1)), ec=MUTED, lw=0.4))
ax.add_patch(Rectangle((112, 39), 9.6, 9.6, fill=False, ec=INK, lw=1))
text(123.5, 43.8, "residual map:\nhigh on patches\nunlike their\nneighbours", fs=7.2, ha="left")

# ------------------------------------------------------------------ our target
rbox(108, 12, 44, 18, VIOLET_L, VIOLET, lw=1.8)
callout(110, 31, 3)
text(130, 26.8, r"re-targeted $z_T$, fitted on training units only", fs=9, color=VIOLET, weight="bold")
text(130, 21.2, "(a) normalised: PCA-whitened top-64 dims, data2vec-style  (exp33)\n"
     "(b) discrete: k-means clusters of the features, HuBERT-style  (next)", fs=7.2)
text(130, 15.6, "post hoc on the frozen tokens; label-free; no new pretraining\nriver-disjoint folds (part A); the 600 valid tiles (part B)", fs=7.2, color=VIOLET)
arrow(99, 37, 108, 30.5, color=STONE)
arrow(140, 30.5, 143, 39.5, color=VIOLET, lw=1.6)

# ------------------------------------------------------------------ the test
panel(156, 12, 42, 66)
callout(158.5, 75.5, 4)
rbox(160, 62, 34, 10, EMERALD_L, EMERALD, lw=1.8)
text(177, 67, "signal for where the map is wrong", fs=9.5, color=EMERALD, weight="bold")
arrow(150.5, 44, 159.5, 62, color=AMBER, lw=1.6)
text(177, 57.5, "scored on identical patches", fs=8.5, weight="bold")
text(177, 50.5, "vs the model's confidence (-|logit|)\nvs tile-phase and the boundary indicator\nvs NDWI gradient (no-model control)\nvs constant, S2 variance, NDWI level", fs=7.3)
text(177, 42.5, "excess AURC per scene or tile, tie-aware", fs=7.3, color=MUTED)
text(177, 37.5, "preregistered", fs=8.5, weight="bold")
text(177, 31.5, "primary = residual; U+ = midrank mean of\nconfidence and residual; one vote per river,\none-sided sign test (7/8 rivers: p = 0.035)", fs=7.3)
text(177, 22.5, "expert labels grade, never train\nnull: residual = boundary detector", fs=7.3, color=EMERALD)

# ------------------------------------------------------------------ legend of callouts
for X, k, t in ((22, 1, "the target encoder is the untouched random init"), (70, 2, "so the shipped residual carries nothing"),
                (110, 3, "we re-target the frozen encoder post hoc"), (152, 4, "and test the residual like every other signal")):
    callout(X, 7, k, r=1.6, fs=8)
    text(X + 2.5, 7, t, fs=8, ha="left")
text(2, 2.5, "olmoearth_inferenceX, issues #10 and #11; figure grammar after Wei et al. (Latent MIM), redrawn", fs=7.5, color=MUTED, ha="left")
fig.savefig("docs/figures/latent_mim_retarget_idea.png", bbox_inches="tight", facecolor="white")

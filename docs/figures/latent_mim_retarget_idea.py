"""Renders docs/figures/latent_mim_retarget_idea.png in the visual grammar of Latent MIM's Figure 1 (Wei et al.,
arXiv:2407.15837): patchify, online / target encoders, decoder, loss, numbered callouts. Redrawn, not copied. The
content is ours: OlmoEarth's frozen random target (exp32), the shipped residual (exp28), the post-hoc re-targeting
(issue #10) and the test every signal takes."""
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

fig, ax = plt.subplots(figsize=(16, 5.6), dpi=200)
ax.set_xlim(0, 160); ax.set_ylim(0, 56); ax.axis("off")
INK, GREY, BLUE, GREEN, RED, ORANGE, LBLUE = "#24292f", "#8c959f", "#1f6feb", "#3a9c4a", "#8b1a1a", "#f6d5b8", "#c9d8f0"
PANEL = "#f2f3f5"


def text(x, y, t, fs=10, color=INK, weight="normal", ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, weight=weight, linespacing=1.3)


def arrow(x0, y0, x1, y1, color="#6e7781", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def encoder(x, y, w, h, t, fc=LBLUE, ec=INK, color=INK, alpha=1.0):
    """The D-shaped encoder block of the Latent MIM figure."""
    ax.add_patch(Rectangle((x, y), w - h / 2, h, fc=fc, ec="none", alpha=alpha))
    ax.add_patch(Wedge((x + w - h / 2, y + h / 2), h / 2, -90, 90, fc=fc, ec="none", alpha=alpha))
    ax.plot([x, x, x + w - h / 2], [y + h, y, y], color=ec, lw=1.2, alpha=alpha)
    ax.plot([x, x + w - h / 2], [y + h, y + h], color=ec, lw=1.2, alpha=alpha)
    th = np.linspace(-np.pi / 2, np.pi / 2, 50)
    ax.plot(x + w - h / 2 + h / 2 * np.cos(th), y + h / 2 + h / 2 * np.sin(th), color=ec, lw=1.2, alpha=alpha)
    text(x + (w - h / 2) / 2 + 1.5, y + h / 2, t, fs=10.5, color=color)


def callout(x, y, k):
    ax.add_patch(Circle((x, y), 2.1, fc=RED, ec="none"))
    text(x, y, str(k), fs=10, color="white", weight="bold")


def panel(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=INK, lw=1.0))


# ------------------------------------------------------------------ scene and patchify
ax.add_patch(Rectangle((2, 20), 16, 16, fc="#cfe8c4", ec="#9cbf8f", lw=1))
ax.add_patch(Rectangle((6, 23), 9, 11, fill=False, ec=INK, lw=1, ls=(0, (2, 2))))
arrow(18.5, 28, 23.5, 28)
panel(24, 8, 30, 42)
text(39, 44, "Patchify", fs=10.5)
rng = np.random.default_rng(0)
hidden = rng.random((5, 5)) < 0.25
for i in range(5):
    for j in range(5):
        ax.add_patch(FancyBboxPatch((28.5 + j * 4.2, 18 + (4 - i) * 4.2), 3.4, 3.4, boxstyle="round,pad=0.15,rounding_size=0.5",
                                    fc=("#9ed08f" if hidden[i, j] else "#b9cff0"), ec="#8c959f", lw=0.5))
text(39, 14, "OlmoEarth encoder tokens\nblue visible, green hidden (25%)", fs=8.5, color="#57606a")

# ------------------------------------------------------------------ encoders
panel(56, 8, 34, 42)
encoder(60, 32, 24, 12, "Online\nEncoder")
text(72, 46.5, "frozen OlmoEarth v1", fs=8.5, color="#57606a")
for k in range(2):
    ax.add_patch(FancyBboxPatch((85.5, 35 + k * 3.2), 2.6, 2.6, boxstyle="round,pad=0.1", fc="#b9cff0", ec="#8c959f", lw=0.5))
text(87, 32.5, r"$z_V$", fs=10)
encoder(60, 11, 24, 12, "Target\nEncoder", fc="#e6e8eb", color="#6e7781", alpha=1.0)
ax.plot([60, 82], [11.5, 22.5], color=RED, lw=2); ax.plot([60, 82], [22.5, 11.5], color=RED, lw=2)
text(72, 8.8, "random init, never updated: rank-2 targets", fs=8, color=RED)
callout(58.5, 46.5, 1)
for k in range(3):
    ax.add_patch(FancyBboxPatch((85.5, 12 + k * 3.2), 2.6, 2.6, boxstyle="round,pad=0.1", fc="#d0d4d9", ec="#8c959f", lw=0.5))
text(87, 24.5, r"$z_T$", fs=10, color="#6e7781")

# ------------------------------------------------------------------ our target, replacing z_T
ax.add_patch(FancyBboxPatch((93, 9), 22, 10, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#ddf4ff", ec=BLUE, lw=1.8))
text(104, 14, "re-targeted $z_T$\nnormalised / discrete", fs=9.5, color=INK)
arrow(88.5, 15, 92.5, 15, color=GREY)
callout(94.5, 20.5, 3)
text(104, 5.5, "post hoc on the frozen tokens\nno new pretraining", fs=8, color=BLUE)

# ------------------------------------------------------------------ decoder and residual
panel(92, 24, 44, 26)
text(104, 47, "Target locations", fs=9.5)
arrow(104, 45, 104, 42.5)
ax.add_patch(FancyBboxPatch((94, 32), 20, 10, boxstyle="round,pad=0.4,rounding_size=1.2", fc=LBLUE, ec=INK, lw=1.2))
text(104, 37, "Predictor\n(small, label-free)", fs=9.5)
arrow(88.5, 37, 93.5, 37)
for k in range(3):
    ax.add_patch(FancyBboxPatch((116.5 + k * 3.2, 35.7), 2.6, 2.6, boxstyle="round,pad=0.1", fc="#d0d4d9", ec="#8c959f", lw=0.5))
text(121, 32.5, r"$\hat{z}_T$", fs=10)
arrow(114.5, 37, 116, 37)
ax.add_patch(FancyBboxPatch((120, 24.5), 14, 6, boxstyle="round,pad=0.3,rounding_size=1", fc=ORANGE, ec=INK, lw=1.2))
text(127, 27.5, "Residual", fs=10)
arrow(121, 35, 125, 31, color="#6e7781")
arrow(104, 19, 118, 25, color=BLUE, lw=1.6)
callout(133.5, 46.5, 2)

# ------------------------------------------------------------------ the test, to the right of the residual
arrow(134.5, 27.5, 140, 22, color=BLUE, lw=1.6)
ax.add_patch(FancyBboxPatch((141, 10), 17, 12, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#dafbe1", ec=GREEN, lw=1.8))
text(149.5, 16, "signal for\nwhere the map\nis wrong", fs=9.5, color="#116329")
callout(143, 23.5, 4)
text(149.5, 5, "vs confidence, vs pixel control,\non expert labels", fs=8, color="#116329")

# ------------------------------------------------------------------ callout list
X = 139
text(X, 53.5, "What we change", fs=11, weight="bold", ha="left")
items = [(1, "OlmoEarth froze the target at random\ninit: targets have rank 2 (exp32)"),
         (2, "the shipped residual is at chance\n(exp28)"),
         (3, "re-target the frozen encoder post hoc:\nnormalised (data2vec) or discrete (HuBERT)"),
         (4, "the residual becomes an inference-time\nerror signal, tested like all the others")]
for i, (k, t) in enumerate(items):
    y = 48.5 - i * 5.2
    callout(X + 1, y, k)
    text(X + 4, y, t, fs=8, ha="left")
fig.savefig("docs/figures/latent_mim_retarget_idea.png", bbox_inches="tight", facecolor="white")

"""Renders docs/figures/proposal_retargeted_residual.png: the proposal as a picture with as few words as possible.
Stages: window -> frozen encoder -> tokens (25% hidden) -> predictor -> residual map -> signal, judged on a balance
against confidence with expert labels; below, the target swap (rank-2 random target -> normalised / discrete)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 80); ax.axis("off")
INK, MUTED = "#1f2937", "#6b7280"
TEAL, TEAL_L, LIME = "#0f766e", "#ccfbf1", "#a3e635"
CROSS, STONE, STONE_L = "#b91c1c", "#78716c", "#e7e5e4"
VIOLET, VIOLET_L = "#6d28d9", "#ede9fe"
AMBER, EMERALD, EMERALD_L = "#b45309", "#047857", "#d1fae5"
rng = np.random.default_rng(1)


def text(x, y, t, fs=9.5, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.3)


def arrow(x0, y0, x1, y1, color=INK, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15, color=color, lw=lw))


def encoder(x, y, w, h, fc, ec):
    ax.add_patch(Rectangle((x, y), w - h / 2, h, fc=fc, ec="none"))
    ax.add_patch(Wedge((x + w - h / 2, y + h / 2), h / 2, -90, 90, fc=fc, ec="none"))
    ax.plot([x, x, x + w - h / 2], [y + h, y, y], color=ec, lw=1.5)
    ax.plot([x, x + w - h / 2], [y + h, y + h], color=ec, lw=1.5)
    th = np.linspace(-np.pi / 2, np.pi / 2, 50)
    ax.plot(x + w - h / 2 + h / 2 * np.cos(th), y + h / 2 + h / 2 * np.sin(th), color=ec, lw=1.5)


def grid(x, y, size, n, colors, hatch=None):
    c = size / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=colors[i][j], ec="#9ca3af", lw=0.5,
                                   hatch="////" if hatch is not None and hatch[i, j] else None))
    ax.add_patch(Rectangle((x, y), size, size, fill=False, ec=INK, lw=1.3))


text(3, 76.5, "Where is the map wrong?  Read OlmoEarth's own objective at inference, with a better target.", fs=13, weight="bold", ha="left")

# ------------------------------------------------------------------ main flow, y centre 52
Y = 44
n = 6
base = rng.random((n, n, 3)) * 0.35 + np.array([0.3, 0.4, 0.2]); base[2:4, 1:5] = [0.15, 0.3, 0.55]
grid(4, Y, 16, n, [[tuple(base[i, j]) for j in range(n)] for i in range(n)])
text(12, Y - 3, "Sentinel-2", fs=9, color=MUTED)
arrow(21, Y + 8, 26, Y + 8)
encoder(27, Y + 2, 18, 12, TEAL_L, TEAL)
text(34.5, Y + 8, "OlmoEarth\nencoder", fs=9.5)
text(36, Y - 3, "frozen", fs=9, color=TEAL)
arrow(46, Y + 8, 51, Y + 8)
hidden = rng.random((n, n)) < 0.25
grid(52, Y, 16, n, [[LIME if hidden[i, j] else TEAL_L for j in range(n)] for i in range(n)], hatch=hidden)
text(60, Y - 3, "tokens, 25% hidden", fs=9, color=MUTED)
arrow(69, Y + 8, 74, Y + 8, color=VIOLET)
ax.add_patch(FancyBboxPatch((75, Y + 2), 16, 12, boxstyle="round,pad=0.5,rounding_size=1.5", fc=VIOLET_L, ec=VIOLET, lw=2))
text(83, Y + 8, "predictor", fs=10, color=VIOLET, weight="bold")
text(80, Y - 3, "label-free", fs=9, color=VIOLET)
arrow(92, Y + 8, 97, Y + 8, color=AMBER)
res = rng.random((n, n)) * 0.3; res[2, 1:5] += 0.5; res[3, 1:5] += 0.35; res[1, 3] += 0.55
cm = plt.get_cmap("YlOrBr")
grid(98, Y, 16, n, [[cm(min(res[i, j], 1)) for j in range(n)] for i in range(n)])
text(106, Y - 3, "residual per patch", fs=9, color=AMBER)
arrow(115, Y + 8, 121, Y + 8, color=EMERALD)

# the balance: residual vs confidence, judged by expert labels
bx, by = 150, Y + 4
ax.plot([bx, bx], [by - 2, by + 14], color=INK, lw=2.2)
ax.add_patch(Polygon([(bx - 4, by - 2), (bx + 4, by - 2), (bx, by + 1)], closed=True, fc=INK))
ax.plot([bx - 22, bx + 22], [by + 14, by + 14], color=INK, lw=2.2)
for dx, label, col in ((-22, "residual", AMBER), (22, "confidence", TEAL)):
    ax.plot([bx + dx, bx + dx], [by + 14, by + 7], color=MUTED, lw=1)
    ax.add_patch(Wedge((bx + dx, by + 7), 7, 180, 360, fc="white", ec=INK, lw=1.4))
    text(bx + dx, by + 3.5, label, fs=9.5, color=col, weight="bold")
ax.add_patch(Circle((bx, by + 14), 1.4, fc=INK))
ax.add_patch(FancyBboxPatch((bx - 12, by + 19), 24, 6, boxstyle="round,pad=0.4,rounding_size=1.2", fc=EMERALD_L, ec=EMERALD, lw=1.8))
text(bx, by + 22, "expert labels judge", fs=9.5, color=EMERALD, weight="bold")
text(bx, Y - 3, "same patches;  no-model pixel control alongside", fs=9, color=MUTED)
text(bx + 34, by + 8, "+ preregistered\n one vote per river", fs=8.5, color=MUTED, ha="left")

# ------------------------------------------------------------------ the target swap, below the predictor
ty = 10
def scatter(x, y, pts, color, size=14):
    ax.add_patch(Rectangle((x, y), size, size, fc="white", ec=INK, lw=1.2))
    ax.scatter(x + size / 2 + pts[:, 0] * size * 0.42, y + size / 2 + pts[:, 1] * size * 0.42, s=6, color=color, lw=0)


# rank-2 random target: points on a line
t = rng.uniform(-1, 1, 120)
line = np.stack([t, 0.55 * t + rng.normal(0, 0.03, 120)], 1)
scatter(44, ty, line, STONE)
ax.plot([44, 58], [ty, ty + 14], color=CROSS, lw=2.5); ax.plot([44, 58], [ty + 14, ty], color=CROSS, lw=2.5)
text(51, ty - 3, "random target\nrank 2", fs=9, color=CROSS)
arrow(60, ty + 7, 66, ty + 7, color=VIOLET)
scatter(67, ty, np.clip(rng.normal(0, 0.5, (150, 2)), -1, 1), VIOLET)
text(74, ty - 3, "normalised", fs=9, color=VIOLET, weight="bold")
text(83.5, ty + 7, "or", fs=9.5, color=MUTED)
cents = np.array([[-0.55, 0.5], [0.5, 0.55], [-0.45, -0.5], [0.55, -0.4]])
cols = [VIOLET, TEAL, AMBER, EMERALD]
ax.add_patch(Rectangle((86, ty), 14, 14, fc="white", ec=INK, lw=1.2))
for c, col in zip(cents, cols):
    p = c + rng.normal(0, 0.12, (35, 2))
    ax.scatter(86 + 7 + p[:, 0] * 14 * 0.42, ty + 7 + p[:, 1] * 14 * 0.42, s=6, color=col, lw=0)
text(93, ty - 3, "discrete", fs=9, color=VIOLET, weight="bold")
arrow(90, ty + 15, 90, Y + 1, color=VIOLET, lw=2)
text(28, ty + 7, "target", fs=10, weight="bold", color=INK)
text(28, ty + 3, "the only thing we change", fs=8.5, color=MUTED)

# outcomes at the right, below the balance
ox = 128
ax.add_patch(FancyBboxPatch((ox, ty - 1), 68, 17, boxstyle="round,pad=0.5,rounding_size=1.5", fc="#fafafa", ec=MUTED, lw=1))
text(ox + 3, ty + 12, "H0", fs=10, weight="bold", ha="left", color=CROSS)
text(ox + 9, ty + 12, "boundary detector again: wins on WorldCover, loses on hand labels", fs=8.5, ha="left")
text(ox + 3, ty + 7, "H1", fs=10, weight="bold", ha="left", color=EMERALD)
text(ox + 9, ty + 7, "new error information beyond confidence (7 of 8 rivers)", fs=8.5, ha="left")
text(ox + 3, ty + 2, "both", fs=10, weight="bold", ha="left", color=VIOLET)
text(ox + 9, ty + 2, "a measured recommendation on the pretraining target", fs=8.5, ha="left")

text(3, 2, "no new pretraining   |   exp32, exp33, issues #10 #11   |   Latent MIM (Wei et al.), data2vec, HuBERT", fs=8, color=MUTED, ha="left")
fig.savefig("docs/figures/proposal_retargeted_residual.png", bbox_inches="tight", facecolor="white")

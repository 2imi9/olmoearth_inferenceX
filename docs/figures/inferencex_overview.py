"""Renders docs/figures/inferencex_overview.png: the repository in one figure. What is audited, the label-free signal
families with the ledger's verdicts (docs/TECHNIQUES.md), the test every signal takes (docs/method/protocol.md),
and the findings with the open question (docs/plan/roadmap.md). Numbers cite the experiments named on the figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 10.4), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 104); ax.axis("off")
INK, MUTED, PANEL = "#1f2937", "#6b7280", "#f5f5f4"
TEAL, TEAL_L = "#0f766e", "#ccfbf1"
EMERALD, EMERALD_L = "#047857", "#d1fae5"
AMBER, AMBER_L = "#b45309", "#fef3c7"
STONE, STONE_L, CROSS = "#78716c", "#e7e5e4", "#b91c1c"
VIOLET, VIOLET_L = "#6d28d9", "#ede9fe"
NAVY = "#1e3a8a"


def text(x, y, t, fs=9, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.32)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def rbox(x, y, w, h, fc, ec, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def block(x, y, w, h, t, fc=TEAL_L, ec=TEAL, fs=9.5, color=INK):
    tip = 3.5
    ax.add_patch(Polygon([(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)], closed=True, fc=fc, ec=ec, lw=1.2))
    text(x + (w - tip) / 2, y + h / 2, t, fs=fs, color=color)


def panel(x, y, w, h, title, tc=NAVY):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=INK, lw=1.0))
    text(x + 2, y + h - 3, title, fs=11, weight="bold", color=tc, ha="left")


def badge(x, y, kind):
    fc, mark = {"ok": (EMERALD, "✓"), "mixed": (AMBER, "~"), "no": (CROSS, "✗"), "next": (VIOLET, "?")}[kind]
    ax.add_patch(Circle((x, y), 1.7, fc=fc, ec="none"))
    text(x, y, mark, fs=9, color="white", weight="bold")


text(3, 101.5, "olmoearth_inferenceX: examining OlmoEarth inference results without labels", fs=13.5, weight="bold", ha="left")
text(3, 97.8, "which windows of a prediction map to trust, and which to send for review; every signal label-free at inference, every verdict scored on expert labels", fs=9, color=MUTED, ha="left")

# ------------------------------------------------------------------ panel A: what is audited
panel(2, 38, 46, 56, "A  What is audited")
block(6, 78, 20, 9, "Sentinel-2\nwindow", fc="#dbeafe", ec="#3b82f6")
arrow(26.5, 82.5, 30, 82.5)
block(30, 78, 14, 9, "OlmoEarth\nencoder", fs=9)
text(37, 75.5, "v1 Base, frozen, fp32", fs=7.3, color=TEAL)
arrow(37, 77.5, 37, 72.5)
rbox(5, 59, 41, 12, "white", TEAL)
text(25.5, 68.5, "three deployment cases", fs=8.5, weight="bold")
text(25.5, 63.5, "a linear water probe on frozen tokens\n(27 WorldCover scenes; Sen1Floods11 tiles)\n"
     "the fine-tuned AWF model end to end  (exp21)\nthe served LCC change rasters  (exp20, exp22)", fs=7.0)
arrow(25.5, 58.5, 25.5, 54.5)
rbox(5, 44, 41, 10, "white", INK)
text(25.5, 51.5, "prediction map + per-window confidence", fs=9, weight="bold")
text(25.5, 47.2, "errors = disagreements with a reference, patch by patch;\nthe signal's job is to rank them before anyone looks", fs=7.2, color=MUTED)
text(25.5, 40.5, "references: WorldCover (weak map), Sen1Floods11 hand labels,\nAWF expert points", fs=7.0, color=MUTED)

# ------------------------------------------------------------------ panel B: signals and verdicts
panel(52, 8, 64, 86, "B  Label-free signals, with the ledger's verdict")
rows = [
    ("ok", "confidence, negative logit margin", "best ranker on every expert-labelled testbed", "exp04 16 18 21"),
    ("ok", "prediction-boundary proximity", "triage: 75% of errors on boundaries vs 20% of correct; not a ranker", "exp14 16 18 20"),
    ("mixed", "tiling instability (aligned tile-phase)", "26/27 scenes, 8/0 rivers vs WorldCover; loses on hand labels", "exp13 18 21"),
    ("mixed", "band-set disagreement (one model, 3 views)", "21/27 vs WorldCover; loses on hand labels", "exp17 18"),
    ("no", "cross-model / cross-version disagreement", "same-family models err together; stronger partner is worse", "exp10 13 18 19"),
    ("no", "internal states: logit lens, drift, attention", "0/27, 3/24, 3/24: language-model tricks do not transfer", "exp17"),
    ("no", "embedding distance and feature typicality", "kNN, Mahalanobis, PCA, ViM vs 3 references: never beats confidence", "exp13 31"),
    ("no", "decoder self-consistency (native masking)", "residual at chance: frozen random target has rank 2", "exp28 32"),
    ("no", "last-layer Laplace / bootstrap variance", "worst signal on both testbeds; moderated confidence = confidence", "exp30"),
    ("no", "occlusion masking; Dawid-Skene in-family", "measures context reliance; inflates and inverts the ordering", "exp08 07"),
    ("next", "re-targeted latent-MIM residual", "whitened / discrete targets on the frozen encoder; designed, preregistered", "exp33, #10"),
]
y0 = 84.5
for i, (k, name, verdict, exps) in enumerate(rows):
    y = y0 - i * 6.9
    badge(55.5, y, k)
    text(58.5, y + 1.4, name, fs=8.3, weight="bold", ha="left", color=(VIOLET if k == "next" else INK))
    text(58.5, y - 1.5, verdict, fs=7.2, ha="left", color=MUTED)
    text(114, y, exps, fs=7, ha="right", color=MUTED)
text(54.5, 9.5, "✓ supported    ~ supported against WorldCover only    ✗ rejected    ? pending", fs=7.6, ha="left", color=MUTED)

# ------------------------------------------------------------------ panel C: the test
panel(120, 38, 78, 56, "C  The test every signal takes", tc=EMERALD)
rbox(123, 78, 72, 10, EMERALD_L, EMERALD, lw=1.6)
text(159, 85.2, "score every signal on identical patches against two references at once", fs=8.8, weight="bold", color=EMERALD)
text(159, 80.8, "the model's own confidence (negative absolute logit)   and   a no-model pixel control (NDWI gradient)", fs=7.6)
text(159, 74.2, "beating one but not the other is not support", fs=7.6, color=MUTED)
rbox(123, 55, 72, 16, "white", EMERALD)
text(159, 68, "how it is scored", fs=8.8, weight="bold")
text(159, 61.5, "tie-aware excess AURC per scene or tile;  wins / losses / ties, exact sign tests, sign-flip permutation\n"
     "one vote per river cluster (8 rivers; 7/8 gives p = 0.035);  block and cluster bootstraps\n"
     "preregistered primary score and U+ combination before any result is read;  a null is a valid result", fs=7.3)
rbox(123, 42, 72, 10, "white", EMERALD)
text(159, 49, "rules", fs=8.8, weight="bold")
text(159, 45, "labels grade signals, never train them;  every recorded claim points to a file under exp/out;\n"
     "the machinery is the torch-free package oe_inferencex, with tests that reproduce the recorded numbers", fs=7.3)

# ------------------------------------------------------------------ panel D: findings and the open question
panel(2, 8, 46, 27, "D  Findings")
text(4, 27.5, "confidence beats every constructed signal on expert labels\n"
     "errors concentrate on prediction boundaries (75% vs 20%)\n"
     "the fine-tuned model is overconfident:\n    0.93 accurate where it says 0.99 (ECE 0.08)\n"
     "the served product exports no class confidence\n"
     "a second run helps only if it sees the input differently", fs=7.3, ha="left", va="top")
text(4, 11.5, "the WorldCover-only wins are real and do not transfer:\nwhy is the open question", fs=7.4, ha="left", color=AMBER, weight="bold")

panel(120, 8, 78, 27, "E  The open question and the next step", tc=CROSS)
text(122, 28.5, "reference instability, the year gap and seasonal water are each ruled out (exp23-25)", fs=7.6, ha="left", va="top")
text(122, 24.6, "leading hypothesis: WorldCover was a pretraining target, so the probe partly reads out\n"
     "the model's own map; the frozen target space has effective rank 2 (exp32)", fs=7.6, ha="left", va="top", color=CROSS)
text(122, 18.6, "decisive test: reference specificity on identical cells, one vote per river;\n"
     "needs a few hundred adjudicated cells across the 8 rivers (issue #2)", fs=7.6, ha="left", va="top")
rbox(122, 9.2, 74, 4.2, VIOLET_L, VIOLET, lw=1.4)
text(159, 11.3, "next signal under test: the re-targeted latent-MIM residual, exp33 (issues #10, #11)", fs=7.8, color=VIOLET, weight="bold")

arrow(48.5, 66, 51.5, 66, color=INK, lw=1.6)
arrow(116.5, 66, 119.5, 66, color=EMERALD, lw=1.6)
text(3, 3, "github.com/2imi9/olmoearth_inferenceX; ledger docs/TECHNIQUES.md; protocol docs/method/protocol.md; open items docs/plan/roadmap.md and issues #2-#11", fs=7.3, color=MUTED, ha="left")
fig.savefig("docs/figures/inferencex_overview.png", bbox_inches="tight", facecolor="white")

"""Renders docs/figures/inferencex_overview.png: the repository in one figure, restricted to what the ledger supports.
What is audited, what we believe is true (with the experiments behind each line), the test every claim passed, and the
open question with its next step. Sources: docs/TECHNIQUES.md, docs/method/protocol.md, docs/plan/roadmap.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 9.2), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 92); ax.axis("off")
INK, MUTED, PANEL = "#1f2937", "#6b7280", "#f5f5f4"
TEAL, TEAL_L = "#0f766e", "#ccfbf1"
EMERALD, EMERALD_L = "#047857", "#d1fae5"
AMBER, AMBER_L = "#b45309", "#fef3c7"
VIOLET, VIOLET_L = "#6d28d9", "#ede9fe"
NAVY, CROSS = "#1e3a8a", "#b91c1c"


def text(x, y, t, fs=9, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.35)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def rbox(x, y, w, h, fc, ec, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def block(x, y, w, h, t, fc=TEAL_L, ec=TEAL, fs=9.5):
    tip = 3.5
    ax.add_patch(Polygon([(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)], closed=True, fc=fc, ec=ec, lw=1.2))
    text(x + (w - tip) / 2, y + h / 2, t, fs=fs)


def panel(x, y, w, h, title, tc=NAVY):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=INK, lw=1.0))
    text(x + 2, y + h - 3, title, fs=11, weight="bold", color=tc, ha="left")


def check(x, y, color=EMERALD):
    ax.add_patch(Circle((x, y), 1.6, fc=color, ec="none"))
    text(x, y, "✓", fs=8.5, color="white", weight="bold")


text(3, 89.5, "olmoearth_inferenceX: examining OlmoEarth inference results without labels", fs=13.5, weight="bold", ha="left")
text(3, 85.8, "which windows of a prediction map to trust, and which to send for review", fs=9.5, color=MUTED, ha="left")

# ------------------------------------------------------------------ A: what is audited
panel(2, 26, 46, 56, "A  What is audited")
block(6, 66, 20, 9, "Sentinel-2\nwindow", fc="#dbeafe", ec="#3b82f6")
arrow(26.5, 70.5, 30, 70.5)
block(30, 66, 14, 9, "OlmoEarth\nencoder", fs=9)
text(37, 63.5, "v1 Base, frozen, fp32", fs=7.3, color=TEAL)
arrow(25.5, 65, 25.5, 60.5)
rbox(5, 47, 41, 12, "white", TEAL)
text(25.5, 56.5, "three deployment cases", fs=8.5, weight="bold")
text(25.5, 51.5, "a linear water probe on frozen tokens\n(27 WorldCover scenes; Sen1Floods11 tiles)\n"
     "the fine-tuned AWF model end to end\nthe served LCC change rasters", fs=7.2)
arrow(25.5, 46.5, 25.5, 42.5)
rbox(5, 32, 41, 10, "white", INK)
text(25.5, 39.5, "prediction map + per-window confidence", fs=9, weight="bold")
text(25.5, 35.2, "errors = disagreements with a reference, patch by patch;\nthe signal's job is to rank them before anyone looks", fs=7.2, color=MUTED)
text(25.5, 28.5, "references: WorldCover (weak map), Sen1Floods11 hand labels,\nAWF expert points", fs=6.6, color=MUTED)

# ------------------------------------------------------------------ B: what we believe is true
panel(52, 8, 74, 74, "B  What we believe is true", tc=EMERALD)
beliefs = [
    ("the model's own confidence is the best label-free error ranker",
     "negative logit margin; best on every expert-labelled testbed (AWF, Sen1Floods11, fine-tuned model)", "exp04 16 18 21"),
    ("errors concentrate on prediction boundaries: 75% of errors vs 20% of correct",
     "a triage cue, not a ranker; boundary first, then by confidence, captures more errors at 5-10% budgets", "exp14 16 18 20 35 36"),
    ("an error window nearly always carries a label-free cue",
     "95% of hand-label errors are on a boundary, low-confidence, unstable or NDWI-ambiguous; NDWI ambiguity 7x", "exp37"),
    ("the fine-tuned model is overconfident, so an accuracy needs a coverage",
     "0.93 accurate where it claims 0.99 (ECE 0.08); abstaining on the least confident 20% gives 0.945", "exp21"),
    ("the served product exports no class confidence; outputs sit on the patch lattice",
     "boundary fraction captures a median 0.88 of disagreements at a 5% review budget; no window seams", "exp20 22"),
    ("averaging four tilings of a window improves the map itself",
     "+1.0 / +0.9 pts pixel accuracy on hand labels, no retraining; half of the grid window's errors are mixed-label windows", "exp42"),
    ("side product: the pretraining target space is degenerate",
     "target encoder = untouched random init, targets of effective rank 2; whitened, the target is 57-70% predictable", "exp32 33 34"),
]
y0 = 74
for i, (claim, how, exps) in enumerate(beliefs):
    y = y0 - i * 8.6
    check(55.5, y, color=EMERALD)
    text(58.5, y + 2.0, claim, fs=7.6, weight="bold", ha="left")
    text(58.5, y - 0.9, how, fs=6.6, ha="left", color=MUTED, va="center")
    text(58.5, y - 3.6, exps, fs=6.4, ha="left", color=MUTED)
text(54, 10.5, "every item: supported on expert labels; what was tried and rejected is in the ledger (docs/TECHNIQUES.md)", fs=7.3, ha="left", color=MUTED)

# ------------------------------------------------------------------ C: the test
panel(130, 26, 68, 56, "C  The test every claim passed", tc=EMERALD)
rbox(133, 66, 62, 11, EMERALD_L, EMERALD, lw=1.6)
text(164, 73.5, "two references at once, on identical patches", fs=8.8, weight="bold", color=EMERALD)
text(164, 69, "the model's own confidence   and   a no-model pixel control\nbeating one but not the other is not support", fs=7.4)
rbox(133, 45, 62, 17, "white", EMERALD)
text(164, 59, "scoring", fs=8.8, weight="bold")
text(164, 52.5, "tie-aware excess AURC per scene or tile\nwins / losses / ties, exact sign tests, sign-flip permutation\n"
     "one vote per river cluster; block and cluster bootstraps\npreregistered primary score and combination", fs=7.3)
rbox(133, 28.5, 62, 12.5, "white", EMERALD)
text(164, 38.5, "rules", fs=8.8, weight="bold")
text(164, 33.4, "expert labels grade signals, never train them\nevery recorded claim points to a file under exp/out\nthe machinery is a tested torch-free package", fs=7.1)

# ------------------------------------------------------------------ D: open question and next step
panel(2, 8, 46, 15, "D  Open question", tc=AMBER)
text(4, 17.5, "why the WorldCover wins do not transfer to hand labels.\nleading hypothesis: WorldCover was a pretraining target, so the\n"
     "probe partly reads out the model's own map; decisive test needs\nadjudicated cells on the 8 rivers (issue #2)", fs=7.3, ha="left", va="top")
panel(130, 8, 68, 15, "E  Where it stands", tc=VIOLET)
text(132, 17.5, "supported for deployment: confidence as the ranker, boundary first as the review order,\n"
     "four-tiling averaging for the map, a reason with evidence per flagged window; all in the package.\n"
     "Remaining: the adjudicated-cell test (issue #2) and the pretraining-target note to Ai2 (issue #11)", fs=7.3, ha="left", va="top")

arrow(48.5, 54, 51.5, 54, color=INK, lw=1.6)
arrow(126.5, 54, 129.5, 54, color=EMERALD, lw=1.6)
text(3, 3, "github.com/2imi9/olmoearth_inferenceX; ledger docs/TECHNIQUES.md; protocol docs/method/protocol.md; open items docs/plan/roadmap.md", fs=7.3, color=MUTED, ha="left")
fig.savefig("docs/figures/inferencex_overview.png", bbox_inches="tight", facecolor="white")

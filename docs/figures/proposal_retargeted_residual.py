"""Renders docs/figures/proposal_retargeted_residual.png: the research proposal in one figure. Problem, observation
(exp32, exp28), proposal (issue #10), method (exp33), evaluation (docs/method/protocol.md), outcomes."""
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

fig, ax = plt.subplots(figsize=(20, 8), dpi=200)
ax.set_xlim(0, 200); ax.set_ylim(0, 80); ax.axis("off")
INK, MUTED = "#1f2937", "#6b7280"
TEAL, TEAL_L = "#0f766e", "#ccfbf1"
CROSS, CROSS_L = "#b91c1c", "#fee2e2"
VIOLET, VIOLET_L = "#6d28d9", "#ede9fe"
AMBER, AMBER_L = "#b45309", "#fef3c7"
EMERALD, EMERALD_L = "#047857", "#d1fae5"
NAVY = "#1e3a8a"


def text(x, y, t, fs=9, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, weight=weight, linespacing=1.35)


def arrow(x0, y0, x1, y1, color=MUTED, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=color, lw=lw))


def card(x, y, w, h, title, body, fc, ec, n):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5", fc=fc, ec=ec, lw=1.8))
    ax.add_patch(FancyBboxPatch((x + 1.5, y + h - 5.5), 4.5, 4.5, boxstyle="round,pad=0.2,rounding_size=1", fc=ec, ec="none"))
    text(x + 3.75, y + h - 3.25, str(n), fs=10, color="white", weight="bold")
    text(x + 8, y + h - 3.25, title, fs=10.5, color=ec, weight="bold", ha="left")
    text(x + w / 2, y + (h - 6) / 2 + 0.5, body, fs=8.2)


# title and problem banner
text(3, 77, "Proposal: the re-targeted latent-MIM residual as a label-free error signal for OlmoEarth inference", fs=13.5, weight="bold", ha="left")
ax.add_patch(FancyBboxPatch((3, 64), 194, 8, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#f5f5f4", ec=INK, lw=1.0))
text(100, 69.6, "Problem.  Where is an OlmoEarth prediction map wrong, with no labels at inference?", fs=10, weight="bold")
text(100, 66.2, "State of the evidence: the model's own confidence is the best ranker on every expert-labelled testbed; nothing built on the frozen encoder beats it; "
     "errors sit on prediction boundaries (75% vs 20%).", fs=8.2, color=MUTED)

W, H, Y = 36, 36, 22
X = [3, 42.5, 82, 121.5, 161]
card(X[0], Y, W, H, "Observation", "OlmoEarth is latent MIM, but its target\nencoder is frozen at random init.\n\n"
     "Targets = random projection of the raw\npatch, never normalised:\neffective rank 2 on real scenes,\nrandom-pair cosine 0.99  (exp32)\n\n"
     "So the objective-at-inference residual\nranks errors at chance  (exp28)", CROSS_L, CROSS, 1)
card(X[1], Y, W, H, "Proposal", "Keep the objective, replace the target,\npost hoc on the frozen encoder.\n\n"
     "(a) normalised target: PCA-whitened,\n      data2vec-style\n(b) discrete target: clusters of the\n      model's own features, HuBERT-style\n\n"
     "No new pretraining. Label-free.\nRecipes from Latent MIM (Wei et al.)", VIOLET_L, VIOLET, 2)
card(X[2], Y, W, H, "Method", "Hide 25% of patches, K = 8 masks,\nevery patch hidden twice.\n\n"
     "A small predictor (2 layers, width 128)\nfills the hidden tokens; the score is\nthe residual per patch in target space.\n\n"
     "Fitted on training units only;\nriver-disjoint folds  (exp33)", AMBER_L, AMBER, 3)
card(X[3], Y, W, H, "Evaluation", "Identical patches; two references at once:\nthe model's confidence and a\nno-model pixel control.\n\n"
     "Expert labels grade, never train:\n27 WorldCover scenes, Sen1Floods11\nBolivia hand labels.\n\n"
     "Preregistered: primary = residual, U+,\none vote per river (7/8: p = 0.035)", EMERALD_L, EMERALD, 4)
card(X[4], Y, W, H, "Outcomes", "H0  the residual is a boundary detector:\nwins against WorldCover, loses on\nhand labels (like every signal so far)\n\n"
     "H1  it adds error information beyond\nconfidence on hand labels\n\n"
     "Either way: a measured recommendation\nto Ai2 on the pretraining target\n(normalise, or shallow-EMA)  (issue #11)", TEAL_L, TEAL, 5)
for i in range(4):
    arrow(X[i] + W + 0.8, Y + H / 2, X[i + 1] - 0.8, Y + H / 2, color=INK)

# footer: cost and links
ax.add_patch(Rectangle((3, 9), 194, 9.5, fc="#fafafa", ec=MUTED, lw=0.8))
text(6, 15.6, "Cost.", fs=9, weight="bold", ha="left")
text(14, 15.6, "one B200 job per target variant (minutes on cached features); design and smoke tests on CPU; Codex cross-review before each run.", fs=8.2, ha="left")
text(6, 11.8, "Status.", fs=9, weight="bold", ha="left")
text(14, 11.8, "exp32 done; exp33 (variant a) designed, reviewed and smoke-tested, waiting for the cluster; variant (b) next.   github.com/2imi9/olmoearth_inferenceX, issues #10, #11.", fs=8.2, ha="left")
text(3, 5, "papers: Wei et al., Towards Latent MIM (arXiv:2407.15837); data2vec (arXiv:2202.03555); HuBERT (arXiv:2106.07447); OlmoEarth v1 (arXiv:2511.13655)", fs=7.6, color=MUTED, ha="left")
fig.savefig("docs/figures/proposal_retargeted_residual.png", bbox_inches="tight", facecolor="white")

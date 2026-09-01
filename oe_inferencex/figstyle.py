"""Shared publication figure style for experiment outputs.

Conventions: every heatmap gets a labeled colorbar and patch-grid axis labels
(one patch = 4 px = 40 m at Sentinel-2 10 m resolution); every curve panel
gets labeled axes, a legend, and a light grid; panels are lettered.
"""
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATCH_M = 40  # meters per patch cell

# Verified against olmoearth_projects/olmoearth_run_data/awf/model.yaml
# (per-class metric class_idx entries; nodata_value 9).
AWF_CLASSES = [
    "woodland forest", "open water", "shrubland/savanna", "herbaceous wetland",
    "grassland/barren", "agriculture/settlement", "montane forest",
    "lava forest", "urban/dense dev.",
]


def setup():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.grid": False,
    })


def letter(ax, i):
    ax.text(0.02, 0.98, f"({string.ascii_lowercase[i]})", transform=ax.transAxes,
            va="top", ha="left", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))


def map_panel(fig, ax, img, title, cbar_label, cmap="viridis", idx=None, rgb=False, **imkw):
    """Heatmap or RGB panel with patch-grid axes and a labeled colorbar."""
    im = ax.imshow(img, cmap=None if rgb else cmap, **imkw)
    ax.set_title(title)
    ax.set_xlabel(f"patch column ({PATCH_M} m/patch)")
    ax.set_ylabel(f"patch row ({PATCH_M} m/patch)")
    if rgb:
        ax.set_xlabel("pixel column (10 m/px)")
        ax.set_ylabel("pixel row (10 m/px)")
    if not rgb:
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label(cbar_label, fontsize=7)
    if idx is not None:
        letter(ax, idx)
    return im


def rc_panel(ax, results, title, idx=None):
    """Risk-coverage panel. results: {name: (coverage, risk, aurc)}."""
    for name, (cov, risk, aurc) in results.items():
        ax.plot(cov, risk, label=f"{name}, AURC={aurc:.4f}", lw=1.3)
    ax.set_xlabel("coverage (fraction of windows retained,\nordered by ascending signal)")
    ax.set_ylabel("selective risk (error rate among retained)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend()
    if idx is not None:
        letter(ax, idx)

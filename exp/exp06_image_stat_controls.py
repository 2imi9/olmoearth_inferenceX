"""Exp 06: no-model image-statistic controls.

Question: do the model-based signals rank errors any better than trivial
image statistics computed directly from pixel values? If not, the channels
are expensive edge detectors on these scenes.

Controls (no model involved):
  spectral variance   - within-patch std of reflectance, averaged over bands
  NDWI ambiguity      - negative |patch-mean NDWI|: highest where the patch
                        sits on the water/land spectral boundary
  NDWI gradient       - patch-mean magnitude of the NDWI spatial gradient

Scored with the same AURC harness, against the same Base-head errors, on the
three scenes with cached imagery: Kazungula (in-domain), Barotse floodplain
(ambiguous margins), Zambezi delta (domain shift). Model-signal AURCs for
these scenes are in docs/TECHNIQUES.md; the baseline max-softmax is
recomputed here to anchor the comparison on identical errors.
"""
import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.data import s2_to_sample, embed
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage

SIZE, PATCH = 128, 4
GRID = SIZE // PATCH
BO = Modality.SENTINEL2_L2A.band_order
G, NIR = BO.index("B03"), BO.index("B08")


def patch_pool(px):
    return px.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def image_stat_signals(img):
    """img: (12, SIZE, SIZE) raw reflectance."""
    x = img[:, :SIZE, :SIZE].astype(np.float64)
    # within-patch spectral std, averaged over bands
    spec = x.reshape(12, GRID, PATCH, GRID, PATCH).std(axis=(2, 4)).mean(axis=0)
    ndwi = (x[G] - x[NIR]) / np.clip(x[G] + x[NIR], 1, None)
    ambiguity = -np.abs(patch_pool(ndwi))  # high = near the water/land boundary
    gy, gx = np.gradient(ndwi)
    grad = patch_pool(np.hypot(gx, gy))
    return {
        "spectral variance (no model)": spec,
        "NDWI ambiguity (no model)": ambiguity,
        "NDWI gradient (no model)": grad,
    }


def main():
    torch.manual_seed(0)
    z3 = np.load("exp/out/exp03_cache.npz")
    z5 = np.load("exp/out/exp05_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]

    scenes = {
        "kazungula": (z3["ev_img"], tuple(int(v) for v in z3["ev_date"]), z3["ev_labels"]),
        "hard_barotse": (z5["hard_barotse_img"], tuple(int(v) for v in z5["hard_barotse_date"]), z5["hard_barotse_lab"]),
        "ood_delta": (z5["ood_delta_img"], tuple(int(v) for v in z5["ood_delta_date"]), z5["ood_delta_lab"]),
    }

    tr_sample = s2_to_sample(tr_img, *tr_date)
    heads, models = {}, {}
    for key, mid in (("base", ModelID.OLMOEARTH_V1_BASE), ("nano", ModelID.OLMOEARTH_V1_NANO)):
        m = load_model_from_id(mid)
        m.eval()
        models[key] = m
        heads[key] = train_logistic_head(embed(m, tr_sample, PATCH), tr_labels)

    rows, per_scene = [], {}
    for name, (img, date, labels) in scenes.items():
        view = img[:, :SIZE, :SIZE]
        sample = s2_to_sample(view, *date)
        p = predict_head(embed(models["base"], sample, PATCH), *heads["base"])
        p_nano = predict_head(embed(models["nano"], sample, PATCH), *heads["nano"])
        errors = ((p > 0.5) != labels.astype(bool)).astype(np.float64)
        signals = {
            "baseline max-softmax": 1 - np.maximum(p, 1 - p),
            "E_case |Nano-Base|": np.abs(p_nano - p),
        }
        signals.update(image_stat_signals(img))
        print(f"{name}: {int(errors.sum())} errors / {errors.size} patches")
        res = {}
        for sn, sig in signals.items():
            cov, risk, aurc = risk_coverage(sig, errors)
            res[sn] = (cov, risk, aurc)
            rows.append((name, sn, aurc, int(errors.sum())))
            print(f"  AURC {sn}: {aurc:.4f}")
        per_scene[name] = (res, signals, errors)

    with open("exp/out/exp06_controls.csv", "w") as f:
        f.write("scene,signal,aurc,n_errors\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.5f},{r[3]}\n")

    from oe_inferencex.figstyle import setup, map_panel, rc_panel
    import matplotlib.pyplot as plt
    setup()
    scene_titles = {
        "kazungula": "Kazungula\n(in-domain, 18 errors)",
        "hard_barotse": "Barotse floodplain\n(ambiguous margins, 97 errors)",
        "ood_delta": "Zambezi delta\n(geographic shift, 29 errors)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    for i, name in enumerate(scenes):
        res, _, _ = per_scene[name]
        rc_panel(axes[0, i], res, scene_titles[name], idx=i)
    _, sigs, _ = per_scene["hard_barotse"]
    ctl = [
        ("spectral variance (no model)", "within-patch std of reflectance,\nmean over bands"),
        ("NDWI ambiguity (no model)", "-|patch-mean NDWI|\n(high = near water/land boundary)"),
        ("NDWI gradient (no model)", "patch-mean NDWI gradient magnitude"),
    ]
    for i, (cn, cb) in enumerate(ctl):
        map_panel(fig, axes[1, i], sigs[cn], f"Barotse: {cn}", cb, cmap="magma", idx=3 + i)
    fig.suptitle("No-model image-statistic controls vs model signals; identical errors and harness per scene. "
                 "Bottom row: control signal maps on the Barotse scene.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("exp/out/exp06_controls.png", bbox_inches="tight")
    print("wrote exp/out/exp06_controls.png and .csv")


if __name__ == "__main__":
    main()

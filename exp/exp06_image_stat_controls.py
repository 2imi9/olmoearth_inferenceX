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

    base = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    base.eval()
    w, b = train_logistic_head(embed(base, s2_to_sample(tr_img, *tr_date), PATCH), tr_labels)

    rows = []
    for name, (img, date, labels) in scenes.items():
        view = img[:, :SIZE, :SIZE]
        p = predict_head(embed(base, s2_to_sample(view, *date), PATCH), w, b)
        errors = ((p > 0.5) != labels.astype(bool)).astype(np.float64)
        signals = {"baseline max-softmax": 1 - np.maximum(p, 1 - p)}
        signals.update(image_stat_signals(img))
        print(f"\n{name}: {int(errors.sum())} errors / {errors.size} patches")
        for sn, sig in signals.items():
            _, _, aurc = risk_coverage(sig, errors)
            rows.append((name, sn, aurc, int(errors.sum())))
            print(f"  AURC {sn}: {aurc:.4f}")

    with open("exp/out/exp06_controls.csv", "w") as f:
        f.write("scene,signal,aurc,n_errors\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.5f},{r[3]}\n")
    print("\nwrote exp/out/exp06_controls.csv")


if __name__ == "__main__":
    main()

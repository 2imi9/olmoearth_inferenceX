"""Exp 12: fill the two unmeasured cells of the in-domain (AWF) comparison.

E_dist: mean cosine distance from each validation window's Base embedding to
its k=5 nearest training-split embeddings (features cached by exp04).
Control: within-window spectral variability with no model involved - the
standard deviation of reflectance over the 8x8 pixels around the labeled
pixel, averaged over the 12 bands.
Scored with AURC against the same 51 seed-0 Base-head errors as exp04.
"""
import numpy as np
import rasterio
import torch

from oe_inferencex.awf import ROOT, GROUPS, list_windows
from oe_inferencex.evidence import (
    train_softmax_head, predict_softmax_head, risk_coverage,
)

N_CLASSES = 9


def spectral_variability(wdir, r, c):
    vals = []
    r0, c0 = max(r - 4, 0), max(c - 4, 0)
    for group, bands in GROUPS.items():
        path = f"{wdir}/layers/sentinel2/{group}/geotiff.tif"
        with rasterio.open(path) as src:
            data = src.read(out_shape=(src.count, 63, 63))
        for bi in range(data.shape[0]):
            vals.append(np.std(data[bi, r0:r0 + 8, c0:c0 + 8].astype(np.float64)))
    return float(np.mean(vals))


def main():
    torch.manual_seed(0)
    windows = list_windows()
    labels = np.array([w[4] for w in windows])
    is_val = np.array([w[1] == "val" for w in windows])
    z = np.load("exp/out/exp04_feats.npz")
    f = z["base_s0"]
    tr, va = ~is_val, is_val

    w, b = train_softmax_head(f[tr], labels[tr], N_CLASSES)
    p = predict_softmax_head(f[va], w, b)
    errors = (p.argmax(1) != labels[va]).astype(np.float64)
    print(f"errors: {int(errors.sum())}/{len(errors)} (must be 51 to match exp04)")

    a = torch.nn.functional.normalize(torch.tensor(f[va]), dim=-1)
    t = torch.nn.functional.normalize(torch.tensor(f[tr]), dim=-1)
    e_dist = torch.topk(1 - (a @ t.T), k=5, largest=False).values.mean(1).numpy()

    val_windows = [wd for wd, m in zip(windows, is_val) if m]
    control = np.array([spectral_variability(wd[0], wd[2], wd[3]) for wd in val_windows])

    for name, sig in (("E_dist knn-to-train", e_dist), ("control spectral variability", control),
                      ("baseline max-softmax (check)", 1 - p.max(1))):
        _, _, aurc = risk_coverage(sig, errors)
        print(f"AURC {name}: {aurc:.4f}")


if __name__ == "__main__":
    main()

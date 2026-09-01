"""Exp 10: does rater strength improve pairwise disagreement? (GPU)

Adds OlmoEarth v1-Large as Base's disagreement partner on the exp09 scene
set and compares |Large-Base| against |Nano-Base| and the baseline on
identical errors. The weak-rater effect (exp03/exp04/exp07) predicts that a
stronger partner should produce a better disagreement signal.
"""
import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage
from exp09_multiscene import AOIS, embed_gpu, get_scene, SIZE, GRID

import os
CACHE = "exp/out/exp09_cache.npz"


def main():
    torch.manual_seed(0)
    cache = dict(np.load(CACHE, allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]

    models = {}
    for k, m in (("nano", ModelID.OLMOEARTH_V1_NANO), ("base", ModelID.OLMOEARTH_V1_BASE),
                 ("large", ModelID.OLMOEARTH_V1_LARGE)):
        models[k] = load_model_from_id(m).to("cuda").eval()
    heads = {k: train_logistic_head(embed_gpu(m, tr_img, tr_date), tr_labels)
             for k, m in models.items()}

    rows = []
    for name in AOIS:
        key = f"{name}_img"
        if key not in cache:
            continue
        img = cache[key]
        date = tuple(int(v) for v in cache[f"{name}_date"])
        lab = cache[f"{name}_lab"]
        view = img[:, :SIZE, :SIZE]
        p = {k: predict_head(embed_gpu(m, view, date), *heads[k]) for k, m in models.items()}
        errors = ((p["base"] > 0.5) != lab.astype(bool)).astype(np.float64)
        if errors.sum() < 8:
            continue
        acc_l = ((p["large"] > 0.5) == lab.astype(bool)).mean()
        signals = {
            "baseline": 1 - np.maximum(p["base"], 1 - p["base"]),
            "|Nano-Base|": np.abs(p["nano"] - p["base"]),
            "|Large-Base|": np.abs(p["large"] - p["base"]),
        }
        row = {sn: risk_coverage(sig, errors)[2] for sn, sig in signals.items()}
        rows.append((name, row, int(errors.sum()), float(acc_l)))
        print(f"{name}: errors {int(errors.sum())}, Large acc {acc_l:.3f}, "
              + ", ".join(f"{k}={v:.4f}" for k, v in row.items()))

    print(f"\nscenes: {len(rows)}")
    for sn in ("baseline", "|Nano-Base|", "|Large-Base|"):
        vals = np.array([r[1][sn] for r in rows])
        wins = sum(1 for r in rows if min(r[1], key=r[1].get) == sn)
        print(f"{sn:<14} mean {vals.mean():.4f} +/- {vals.std():.4f}  wins {wins}")
    better = sum(1 for r in rows if r[1]["|Large-Base|"] < r[1]["|Nano-Base|"])
    print(f"|Large-Base| better than |Nano-Base| on {better}/{len(rows)} scenes")

    with open("exp/out/exp10_large_pair.csv", "w") as f:
        f.write("scene,n_errors,large_acc,baseline,nano_base,large_base\n")
        for name, row, ne, acc in rows:
            f.write(f"{name},{ne},{acc:.3f},{row['baseline']:.5f},"
                    f"{row['|Nano-Base|']:.5f},{row['|Large-Base|']:.5f}\n")
    print("wrote exp/out/exp10_large_pair.csv")


if __name__ == "__main__":
    main()

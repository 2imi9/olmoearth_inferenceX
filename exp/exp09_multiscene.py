"""Exp 09: multi-scene replication with spread (GPU).

The same signal comparison (baseline, E_case, tile-phase, E_dist) over nine
candidate river scenes across southern Africa (seven evaluated in the recorded run), all evaluated with heads trained at
Katima Mulilo and scored against ESA WorldCover. Produces per-scene AURC,
the per-signal mean and standard deviation, and win counts: the first
numbers with spread rather than one scene per condition. Scenes with fewer
than 8 errors are skipped. The best no-model pixel statistic (NDWI gradient,
per exp06) runs as the control on every scene.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage, pool_to_patches

SIZE, PATCH, PAD = 128, 4, 4
GRID = SIZE // PATCH
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "exp/out/exp09_cache.npz"

AOIS = {
    "kazungula": (25.263, -17.788),
    "barotse": (23.090, -15.400),
    "delta": (36.180, -18.850),
    "tete": (33.600, -16.150),
    "itezhitezhi": (26.020, -15.740),
    "luangwa_conf": (30.418, -15.617),
    "okavango_sep": (22.185, -18.735),
    "shire_liwonde": (35.230, -15.060),
    "vicfalls_up": (25.840, -17.880),
}


def make_sample_gpu(img, date):
    x = img.transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = date
    return (torch.tensor(x, dtype=torch.float32, device=DEV),
            torch.tensor([d, m0, y], device=DEV)[None, None, :])


def embed_gpu(model, img, date):
    xs, ts = make_sample_gpu(img, date)
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=xs,
        sentinel2_l2a_mask=torch.ones((1, img.shape[1], img.shape[2], 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts,
    )
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])[0].cpu()


def get_scene(name, lon, lat, cache):
    for k in (f"{name}_img", f"{name}_date", f"{name}_lab"):
        if k not in cache:
            break
    else:
        return cache[f"{name}_img"], tuple(int(v) for v in cache[f"{name}_date"]), cache[f"{name}_lab"]
    img, date, (crs, transform) = fetch_s2_window(lon, lat, SIZE + PAD)
    wc = fetch_worldcover_window(lon, lat, crs, transform, SIZE + PAD)
    lab = (pool_to_patches(wc[:SIZE, :SIZE] == 80, PATCH) > 0.5).astype(np.float32)
    cache[f"{name}_img"], cache[f"{name}_date"], cache[f"{name}_lab"] = img, np.array(date), lab
    return img, date, lab


def main():
    torch.manual_seed(0)
    cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    # reuse existing caches for the three known scenes
    z3 = np.load("exp/out/exp03_cache.npz")
    z5 = np.load("exp/out/exp05_cache.npz")
    cache.setdefault("kazungula_img", z3["ev_img"]); cache.setdefault("kazungula_date", z3["ev_date"]); cache.setdefault("kazungula_lab", z3["ev_labels"])
    for nm, src in (("barotse", "hard_barotse"), ("delta", "ood_delta")):
        cache.setdefault(f"{nm}_img", z5[f"{src}_img"]); cache.setdefault(f"{nm}_date", z5[f"{src}_date"]); cache.setdefault(f"{nm}_lab", z5[f"{src}_lab"])

    models = {k: load_model_from_id(m).to(DEV).eval()
              for k, m in (("nano", ModelID.OLMOEARTH_V1_NANO), ("base", ModelID.OLMOEARTH_V1_BASE))}
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    heads, f_tr_base = {}, None
    for k, m in models.items():
        f = embed_gpu(m, tr_img, tr_date)
        heads[k] = train_logistic_head(f, tr_labels)
        if k == "base":
            f_tr_base = f

    table = {}
    for name, (lon, lat) in AOIS.items():
        try:
            img, date, lab = get_scene(name, lon, lat, cache)
        except Exception as exc:
            print(f"{name}: fetch failed ({type(exc).__name__}), skipped")
            continue
        np.savez(CACHE, **cache)
        wf = lab.mean()
        base_view = img[:, :SIZE, :SIZE]
        f_base = embed_gpu(models["base"], base_view, date)
        f_nano = embed_gpu(models["nano"], base_view, date)
        p_base = predict_head(f_base, *heads["base"])
        p_nano = predict_head(f_nano, *heads["nano"])
        errors = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
        if errors.sum() < 8:
            print(f"{name}: water {100*wf:.1f}%, errors {int(errors.sum())} -> too few errors, skipped")
            continue
        shift_p = [p_base] + [
            predict_head(embed_gpu(models["base"], img[:, s:s + SIZE, s:s + SIZE], date), *heads["base"])
            for s in (1, 2, 3)
        ]
        a = torch.nn.functional.normalize(f_base.reshape(-1, f_base.shape[-1]), dim=-1)
        b = torch.nn.functional.normalize(f_tr_base.reshape(-1, f_tr_base.shape[-1]), dim=-1)
        knn = torch.topk(1 - (a @ b.T), k=5, largest=False).values.mean(1).reshape(GRID, GRID).numpy()
        x10 = base_view.astype(np.float64)
        bo = Modality.SENTINEL2_L2A.band_order
        ndwi = (x10[bo.index("B03")] - x10[bo.index("B08")]) / np.clip(
            x10[bo.index("B03")] + x10[bo.index("B08")], 1, None)
        gy, gx = np.gradient(ndwi)
        grad = np.hypot(gx, gy).reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))
        signals = {
            "baseline": 1 - np.maximum(p_base, 1 - p_base),
            "E_case": np.abs(p_nano - p_base),
            "tile-phase": np.stack(shift_p).std(0),
            "E_dist": knn,
            "control": grad,
        }
        row = {}
        for sn, sig in signals.items():
            _, _, aurc = risk_coverage(sig, errors)
            row[sn] = aurc
        table[name] = (row, int(errors.sum()), 100 * wf)
        best = min(row, key=row.get)
        print(f"{name}: water {100*wf:.1f}%, errors {int(errors.sum())}, "
              + ", ".join(f"{k}={v:.4f}" for k, v in row.items()) + f"  -> best: {best}")

    sigs = ["baseline", "E_case", "tile-phase", "E_dist", "control"]
    print(f"\nscenes evaluated: {len(table)}")
    print(f"{'signal':<12}{'mean AURC':>12}{'std':>10}{'wins':>6}")
    for s in sigs:
        vals = np.array([table[n][0][s] for n in table])
        wins = sum(1 for n in table if min(table[n][0], key=table[n][0].get) == s)
        print(f"{s:<12}{vals.mean():>12.4f}{vals.std():>10.4f}{wins:>6}")

    with open("exp/out/exp09_multiscene.csv", "w") as f:
        f.write("scene,n_errors,water_pct," + ",".join(sigs) + "\n")
        for n, (row, ne, wp) in table.items():
            f.write(f"{n},{ne},{wp:.1f}," + ",".join(f"{row[s]:.5f}" for s in sigs) + "\n")

    from oe_inferencex.figstyle import setup, letter
    import matplotlib.pyplot as plt
    setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(table))
    markers = {"baseline": "o", "E_case": "s", "tile-phase": "^", "E_dist": "d", "control": "x"}
    for s in sigs:
        ax.scatter(xs, [table[n][0][s] for n in table], label=s, marker=markers[s], s=45, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n({table[n][1]} err)" for n in table], fontsize=7)
    ax.set_ylabel("AURC (log scale, lower = better ranking)")
    ax.set_title("Per-scene AURC across river scenes; heads trained at Katima Mulilo, WorldCover reference")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig("exp/out/exp09_multiscene.png", bbox_inches="tight")
    print("wrote exp/out/exp09_multiscene.png and .csv")


if __name__ == "__main__":
    main()

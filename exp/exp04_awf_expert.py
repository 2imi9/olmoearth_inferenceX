"""Exp 04: audit signals validated against AWF expert labels (real partner truth).

Uses the official rslearn spatial split shipped in the dataset (1115 train /
344 val). Frozen Nano + Base embeddings of the 12-month S2 stack, multiclass
heads, then on val: max-softmax baseline, Nano-Base disagreement (total
variation), tile-phase instability (Base, shifts 0-3 px). AURC against
expert-label errors of the Base head.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.awf import CROP, list_windows, load_window_full, crop_stack, stacks_to_sample
from oe_inferencex.evidence import train_softmax_head, predict_softmax_head, risk_coverage

PATCH = 4
N_CLASSES = 9
SHIFTS = (0, 1, 2, 3)
BATCH = 8
CACHE = "exp/out/exp04_feats.npz"


PASSES = ["nano_s0", "base_s0", "base_s1", "base_s2", "base_s3"]
CKPT = "exp/out/exp04_ckpt.npz"


def embed_all(windows):
    """One disk pass per batch; all five model/shift passes from memory.
    Checkpoints every 20 batches so a killed run resumes."""
    models = {"nano": load_model_from_id(ModelID.OLMOEARTH_V1_NANO),
              "base": load_model_from_id(ModelID.OLMOEARTH_V1_BASE)}
    for m in models.values():
        m.eval()
    feats = {k: [] for k in PASSES}
    start = 0
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        start = int(z["done"])
        for k in PASSES:
            feats[k] = list(z[k])
        print(f"resuming at window {start}")
    for i in range(start, len(windows), BATCH):
        chunk = windows[i:i + BATCH]
        fulls = [(load_window_full(w[0]), w[2], w[3]) for w in chunk]
        for key in PASSES:
            mname, s = key.split("_s")
            stacks, locs = [], []
            for full, r, c in fulls:
                cr, (pr, pc) = crop_stack(full, r, c, int(s))
                stacks.append(cr)
                locs.append((min(pr // PATCH, CROP // PATCH - 1), min(pc // PATCH, CROP // PATCH - 1)))
            sample = stacks_to_sample(stacks)
            with torch.no_grad():
                out = models[mname].encoder(sample, fast_pass=True, patch_size=PATCH)
            f = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])
            for bi, (pi, pj) in enumerate(locs):
                feats[key].append(f[bi, pi, pj].numpy())
        if (i // BATCH) % 20 == 19:
            np.savez(CKPT, done=i + len(chunk), **{k: np.stack(v) for k, v in feats.items()})
            print(f"  {i + len(chunk)}/{len(windows)} (checkpointed)", flush=True)
    return {k: np.stack(v) for k, v in feats.items()}


def main():
    torch.manual_seed(0)
    windows = list_windows()
    labels = np.array([w[4] for w in windows])
    is_val = np.array([w[1] == "val" for w in windows])
    print(f"windows: {len(windows)}, train {np.sum(~is_val)}, val {np.sum(is_val)}")

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        feats = {k: z[k] for k in z.files}
        print("loaded cached features")
    else:
        feats = embed_all(windows)
        np.savez(CACHE, **feats)

    tr, va = ~is_val, is_val
    y = labels

    # heads trained on shift-0 features
    heads, probs = {}, {}
    for name in ("nano_s0", "base_s0"):
        w, b = train_softmax_head(feats[name][tr], y[tr], N_CLASSES)
        heads[name] = (w, b)
        probs[name] = predict_softmax_head(feats[name][va], w, b)
        acc = (probs[name].argmax(1) == y[va]).mean()
        print(f"{name}: val acc = {acc:.3f}")

    p_base = probs["base_s0"]
    errors = (p_base.argmax(1) != y[va]).astype(np.float64)
    print(f"Base head error rate on expert labels: {errors.mean():.3f} ({int(errors.sum())}/{len(errors)})")

    signals = {"baseline max-softmax": 1 - p_base.max(1)}
    signals["E_case TV(Nano,Base)"] = 0.5 * np.abs(probs["nano_s0"] - p_base).sum(1)

    # tile-phase: same Base head applied to shifted features
    wb, bb = heads["base_s0"]
    shift_probs = [p_base] + [
        predict_softmax_head(feats[f"base_s{s}"][va], wb, bb) for s in SHIFTS[1:]
    ]
    sp = np.stack(shift_probs)  # (S, N, C)
    signals["E_system tile-phase"] = sp.max(2).std(0) + 0.5 * np.abs(sp - sp.mean(0)).sum(2).mean(0)

    results = {}
    for name, sig in signals.items():
        cov, risk, aurc = risk_coverage(sig, errors)
        results[name] = (cov, risk, aurc)
        print(f"AURC {name}: {aurc:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (cov, risk, aurc) in results.items():
        axes[0].plot(cov, risk, label=f"{name} (AURC {aurc:.3f})", lw=1.4)
    axes[0].set_xlabel("coverage"); axes[0].set_ylabel("selective risk")
    axes[0].set_title("risk-coverage on AWF expert labels (val, spatial split)")
    axes[0].legend(fontsize=8)
    per_class = [( (p_base.argmax(1) == y[va]) & (y[va] == k)).sum() / max((y[va] == k).sum(), 1)
                 for k in range(N_CLASSES)]
    axes[1].bar(range(N_CLASSES), per_class)
    axes[1].set_xlabel("category"); axes[1].set_ylabel("Base head recall")
    axes[1].set_title("per-class recall vs expert labels")
    fig.tight_layout()
    fig.savefig("exp/out/exp04_awf_expert.png", dpi=150)
    print("wrote exp/out/exp04_awf_expert.png")


if __name__ == "__main__":
    main()

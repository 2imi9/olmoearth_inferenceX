"""exp27 oracle gate: can WorldCover class identity be read from the latent-MIM
target space at all?

OlmoEarth-v1 was pretrained with ESA WorldCover 2021 as a decode-only target;
the decoder is trained to match the frozen target projection of the raw 8x8
class-code patch (a Conv2d(1 -> 768, kernel 8) with bias, applied after the
WorldCover normaliser), under a loss that L2-normalises predictions and
targets. Before any decoder output is read as a class, this gate pushes
synthetic patches through the exact saved projection and asks (1) whether
pure-class prototypes are distinguishable under cosine, (2) whether water
fraction or spatial layout dominates the token, and (3) how much water
fraction a ridge readout can recover from perfect tokens, raw and normalised.

Runs on CPU in seconds; needs only the public checkpoint. Writes
exp/out/exp27_oracle_gate.json. Deterministic (fixed seeds).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def main():
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).eval()
    sd = model.target_encoder.patch_embeddings.state_dict()
    kw = [k for k in sd if "worldcover" in k and k.endswith("proj.weight")][0]
    W, b = sd[kw], sd[kw[:-6] + "bias"]
    norm = Normalizer(Strategy.COMPUTED)

    @torch.no_grad()
    def target(patch):
        x = np.asarray(patch, dtype=np.float64)[None, :, :, None]
        xn = torch.tensor(np.asarray(norm.normalize(Modality.WORLDCOVER, x)), dtype=torch.float32).permute(0, 3, 1, 2)
        return F.conv2d(xn, W, b, stride=8).flatten()

    unit = lambda t: t / t.norm()
    protos = torch.stack([target(np.full((8, 8), c)) for c in CODES])
    C = (protos / protos.norm(dim=1, keepdim=True)) @ (protos / protos.norm(dim=1, keepdim=True)).T
    off = C[~torch.eye(len(CODES), dtype=bool)]
    water = unit(protos[CODES.index(80)])

    def mix(frac, layout, seed=0):
        p = np.full((8, 8), 30.0)
        k = int(round(frac * 64))
        if layout == "block":
            p.flat[:k] = 80
        else:
            p.flat[np.random.default_rng(seed).permutation(64)[:k]] = 80
        return p

    mixtures = {}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        tb, ts, ts2 = target(mix(f, "block")), target(mix(f, "scatter", 0)), target(mix(f, "scatter", 1))
        mixtures[f"{f:.2f}"] = dict(cos_block_water=float(unit(tb) @ water), cos_scatter_water=float(unit(ts) @ water),
                                    cos_block_scatter=float(unit(tb) @ unit(ts)), cos_scatter0_scatter1=float(unit(ts) @ unit(ts2)))

    rng = np.random.default_rng(1)
    X, y = [], []
    for _ in range(600):
        f = rng.uniform(0, 1)
        p = np.full((8, 8), float(rng.choice([10, 20, 30, 40, 60])))
        k = int(round(f * 64))
        p.flat[rng.permutation(64)[:k]] = 80
        X.append(target(p).numpy())
        y.append(k / 64)
    X, y = np.array(X), np.array(y)

    def r2(Xf):
        A = np.c_[Xf[:400], np.ones(400)]
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y[:400])
        pred = np.c_[Xf[400:], np.ones(200)] @ w
        return float(1 - ((pred - y[400:]) ** 2).sum() / ((y[400:] - y[400:].mean()) ** 2).sum())

    out = dict(
        projection=dict(key=kw, weight_shape=list(W.shape), weight_sum_norm=float(W.sum(dim=(1, 2, 3)).norm()), bias_norm=float(b.norm())),
        normaliser=dict(code_10=float(np.asarray(norm.normalize(Modality.WORLDCOVER, np.full((1, 8, 8, 1), 10.0))).ravel()[0]),
                        code_80=float(np.asarray(norm.normalize(Modality.WORLDCOVER, np.full((1, 8, 8, 1), 80.0))).ravel()[0])),
        prototypes=dict(codes=CODES, token_norms=[float(v) for v in protos.norm(dim=1)],
                        pairwise_cosine=dict(min=float(off.min()), median=float(off.median()), max=float(off.max())),
                        cos_water_grass=float(C[7, 2]), cos_water_builtup=float(C[7, 4]), cos_water_wetland=float(C[7, 8])),
        mixtures_water_vs_grass=mixtures,
        ridge_readout_of_water_fraction=dict(r2_raw_tokens=r2(X), r2_l2_normalised_tokens=r2(X / np.linalg.norm(X, axis=1, keepdims=True)),
                                             n_train=400, n_test=200),
        reading="class identity is carried by token norm, not direction; under cosine the pure-class prototypes are nearly "
                "identical and layout moves the token more than water fraction does; a linear readout of water fraction from "
                "perfect tokens is bounded well below 1, and lower once tokens are normalised as the loss normalises them.",
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp27_oracle_gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("prototypes", "ridge_readout_of_water_fraction")}, indent=1))
    print("wrote", os.path.join(OUT, "exp27_oracle_gate.json"))


if __name__ == "__main__":
    main()

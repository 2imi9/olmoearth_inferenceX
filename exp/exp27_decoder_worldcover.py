"""exp27: the model's own WorldCover head.

Status: retired draft. Written 2026-09-06 and run only as a one-scene CPU
pilot (smoke outputs, not recorded). The exp27 oracle gate
(exp27_oracle_gate.py, exp/out/exp27_oracle_gate.json) then showed that
nearest-prototype readouts of the decoder's target space are unsound, the
readout this script relies on, so it was not developed further. Committed
2026-09-17 for the record; see exp/NOTES.md, "exp27 oracle gate".

Question. OlmoEarth v1-Base was pretrained with ESA WorldCover (2021 v200)
as a decode-only target: the online encoder never sees the map, the
Predictor (decoder) must produce, for every WorldCover patch, a token that
matches the frozen target projection of the map. The 27-scene study
(exp11/exp13) scores a linear water probe on the same encoder against that
same map. So: do the probe's disagreements with WorldCover sit where the
model's OWN WorldCover prediction is uncertain or wrong?

Design. For each rule-selected scene the Sentinel-2 window is encoded
(mask ONLINE_ENCODER, as oe_inferencex.data.s2_to_sample does) together with
a worldcover field marked MaskValue.DECODER (the value training assigns to
every non-missing token of an only_decode modality; see
olmoearth_pretrain/train/masking.py RandomWithDecodeMaskingStrategy and the
checkpoint's masking_config). The field's values are zeroed: the encoder
drops DECODER tokens before attention and the decoder replaces them with its
mask token, so the map cannot leak (verified: decoded tokens are identical
for real and zeroed values). Encoder and decoder are run exactly as
LatentMIM.forward does (fast_pass=False, patch_size=8, input_res=10) and the
decoder's worldcover tokens are taken. The target space is the frozen
target_encoder patch projection (token_exit_cfg=0 for every modality, as in
training, and ema_decay=(1,1) so the target is the init-time projection):
the scene's actual WorldCover-2021 class map, fetched with
oe_inferencex.data.fetch_worldcover_window on the scene's re-verified grid
(exp15 lookup), is projected through it, and so is one uniform 8x8 patch per
class (10,...,100) to give class prototypes.

Patch size. 8 (native kernel): FlexiPatchEmbed at patch 4 would bicubically
upsample the class codes before the 8-px kernel, which is meaningless for a
categorical map. The probe's 4-px disagreement set (exp13: seed-0 Base head
trained on Katima, disagreements with the cached WorldCover-2021 water label)
is pooled to the 8-px grid with ANY (a cell is a disagreement if any of its
four 4-px patches disagrees); MAJORITY (3 or 4 of 4) is reported as a
sensitivity. The model's own confidence at 8 px is the least confident
sub-patch (negative minimum |logit|), the pixel control is the exp13 NDWI
gradient pooled to 8 px.

Per-patch quantities from the decoder:
  (a)  cosine similarity of the predicted token to the actual-label token
       (decoder-label agreement); its per-patch patch-discrimination loss
       (the pretraining objective, softmax over the scene's worldcover
       targets, tau=0.1) is reported alongside as an exploratory addition;
  (b)  nearest-prototype class and margin (decoder-predicted class and its
       confidence). Note: the prototypes are exactly collinear (token =
       bias + code * kernel-sum), so this classifier is a one-dimensional
       family and the margin is small by construction; recorded in the
       summary;
  (c)  cosine similarity to the water prototype (code 80).

Pre-specified tests (all per scene, exact sign test across scenes on
untied pairs, as in exp13):
  T1  is decoder-label agreement lower on probe-disagreement cells than on
      agreement cells: tie-aware E-AURC of 1-(a) as a ranker of the
      disagreements, plus ROC AUC, against the confidence baseline and the
      NDWI-gradient control (W/L/T, sign p);
  T2  does the decoder's own predicted class agree with WorldCover more
      often than the probe: accuracy of nearest-prototype water-vs-not
      against the 8-px label, versus the probe's 8-px accuracy;
  T3  among probe disagreements, the share where the decoder ALSO
      disagrees with WorldCover (shared) versus agrees (probe-only), per
      scene and pooled, with the decoder's disagreement share among
      probe-agreement cells as the base rate;
  T4  (optional, network) on cells where WorldCover 2020 v100 and 2021 v200
      differ, is the decoder's token closer to the 2021 or the 2020 label
      token (identifies the pretraining vintage empirically).

Inputs: exp/out/exp11_scenes.npz (scenes, dates, 4-px water labels),
exp/out/exp11_feats.npz (exp13 head and disagreements; ignored by git),
exp/out/exp03_cache.npz (tr_labels), exp/out/rule_candidates.json +
exp11_hardening.EXISTING (scene centres), the Base checkpoint from the HF
cache, and the Planetary Computer STAC (Sentinel-2 georeferencing re-check,
WorldCover 2021 and 2020). Network failures are per-scene skips recorded in
the summary. Outputs: exp/out/exp27_decoder_worldcover.csv (one row per
scene), exp27_summary.json, exp27_decoder_worldcover.png. Caches (ignored by
git): exp27_decoder_tokens.npz, exp27_geo.npz (seeded from exp23_geo.npz
when present). --smoke: CPU, two scenes, 48-px crops, stand-in probe if
exp11_feats.npz is absent, synthetic labels if the network is unavailable,
every output with a _smoke suffix.
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.normalize import Normalizer, Strategy  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from olmoearth_pretrain.nn.utils import unpack_encoder_output  # noqa: E402

from oe_inferencex.evidence import pool_to_patches, predict_head, predict_logit, train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected  # noqa: E402
from exp11_hardening import EXISTING  # noqa: E402
from exp13_stat_corrections import NON_RULE, eaurc, sign_test_p  # noqa: E402

# fp32 only (owner rule): no TF32, no autocast, no compile.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE, PAD = 128, 4
P4, P8 = 4, 8
CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
CODE_NAMES = ["tree cover", "shrubland", "grassland", "cropland", "built-up", "bare/sparse", "snow/ice",
              "water", "wetland", "mangroves", "moss/lichen"]
WATER = 80
TAU = 0.1               # patch-discrimination temperature of the checkpoint's loss_config
MIN_ERR4 = 8            # exp13 scene rule (disagreements at 4 px)
MIN_ERR8 = 3            # cells needed to score a ranker at 8 px (exp18 per-tile rule)
SIGNALS = ["confidence (baseline)", "decoder-label disagreement (1-cos)", "decoder per-patch loss",
           "decoder margin (neg)", "decoder-vs-probe disagreement", "control (NDWI gradient)"]
PRESPECIFIED = {"decoder-label disagreement (1-cos)"}
_NORM = Normalizer(Strategy.COMPUTED)


# ----------------------------------------------------------------------------- helpers
def run_with_timeout(fn, timeout, *args):
    """Run fn(*args) in a daemon thread; raise TimeoutError if it overruns so a
    stalled network call can never hang the job past the summary."""
    box = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"{fn.__name__} exceeded {timeout:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def load_coords():
    coords = dict(EXISTING)
    path = os.path.join(OUT, "rule_candidates.json")
    if os.path.exists(path):
        for name, lon, lat in json.load(open(path)):
            coords[name] = (lon, lat)
    else:
        from exp11_hardening import candidate_centers
        for name, lon, lat in candidate_centers():
            coords[name] = (lon, lat)
    return coords


def fetch_geo(name, lon, lat, img):
    """Re-verified georeferencing (exp15) and WorldCover 2021/2020 pixel maps on
    the scene's 132-px grid. Returns dict or raises."""
    import rasterio
    from exp15_boundary_geo import scene_georef
    from oe_inferencex.data import fetch_worldcover_window
    g = scene_georef(lon, lat, img)
    if g is None:
        raise RuntimeError("re-read band does not match the cached image")
    crs_wkt, transform = g
    crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    aff = rasterio.Affine(*transform)
    wc21 = fetch_worldcover_window(lon, lat, crs, aff, SIZE + PAD, version="2.0.0")
    try:
        wc20 = fetch_worldcover_window(lon, lat, crs, aff, SIZE + PAD, version="1.0.0")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: WorldCover 2020 fetch failed ({type(exc).__name__}); T4 skipped for this scene")
        wc20 = None
    return {"crs": crs_wkt, "transform": np.asarray(transform, dtype=np.float64), "wc21": wc21, "wc20": wc20}


def synthetic_classes(lab4, size):
    """Smoke fallback when the network is unavailable: a class map built from
    the cached 4-px water labels (80 water, 30 grassland, a 40 cropland band)."""
    wc = np.full((size, size), 30, dtype=np.uint8)
    water = np.kron(lab4.astype(bool), np.ones((P4, P4), dtype=bool))[:size, :size]
    wc[water] = WATER
    wc[: size // 4, :][~water[: size // 4, :]] = 40
    return wc


def s2_sample(img, date, device):
    """S2 window (12,S,S) DN -> MaskedOlmoEarthSample with S2 encoded and a
    zeroed worldcover field marked DECODER (only its mask matters)."""
    s = img.shape[1]
    x = img.transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = _NORM.normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = (int(v) for v in date)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=device),
        sentinel2_l2a_mask=torch.ones((1, s, s, 1, 3), device=device) * MaskValue.ONLINE_ENCODER.value,
        worldcover=torch.zeros((1, s, s, 1, 1), device=device),
        worldcover_mask=torch.ones((1, s, s, 1, 1), device=device) * MaskValue.DECODER.value,
        timestamps=torch.tensor([d, m0, y], device=device)[None, None, :],
    )


@torch.no_grad()
def decode_worldcover(model, sample):
    """Encoder then decoder exactly as LatentMIM.forward; returns the decoder's
    worldcover tokens (G, G, D) float32."""
    output_dict = model.encoder(sample, patch_size=P8)
    latent, _pooled, decoder_kwargs = unpack_encoder_output(output_dict)
    decoded = model.decoder(latent, timestamps=sample.timestamps, patch_size=P8, **decoder_kwargs)
    assert (decoded.worldcover_mask == MaskValue.DECODER.value).all()
    return decoded.worldcover[0, :, :, 0, 0, :].float().cpu().numpy()


@torch.no_grad()
def target_tokens(model, codes, device):
    """WorldCover class codes (H, W) -> frozen target projection (H/8, W/8, D),
    the training target path: target_encoder with token_exit_cfg=0 on the
    unmasked field (all mask values ONLINE_ENCODER)."""
    x = _NORM.normalize(Modality.WORLDCOVER, codes[None, :, :, None, None].astype(np.float64))
    h, w = codes.shape
    sample = MaskedOlmoEarthSample(
        worldcover=torch.tensor(x, dtype=torch.float32, device=device),
        worldcover_mask=torch.zeros((1, h, w, 1, 1), device=device),
        timestamps=torch.tensor([1, 0, 2021], device=device)[None, None, :],
    )
    exit_cfg = {m: 0 for m in model.target_encoder.supported_modality_names}
    out = model.target_encoder(sample, patch_size=P8, token_exit_cfg=exit_cfg)
    return out["tokens_and_masks"].worldcover[0, :, :, 0, 0, :].float().cpu().numpy()


def prototypes(model, device):
    """One uniform 8x8 patch per class through the target projection -> (11, D)."""
    band = np.repeat(np.array(CODES, dtype=np.uint8), P8)[:, None] * np.ones((1, P8), dtype=np.uint8)
    return target_tokens(model, band, device)[:, 0, :]


def l2n(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def block_majority(codes, p):
    """Majority class code per p x p block (ties -> smallest code); also the
    validity flag (no nodata 0 in the block)."""
    h, w = codes.shape
    b = codes.reshape(h // p, p, w // p, p).transpose(0, 2, 1, 3).reshape(h // p, w // p, p * p)
    counts = np.stack([(b == c).sum(-1) for c in CODES], -1)
    maj = np.array(CODES)[counts.argmax(-1)]
    valid = (b != 0).all(-1)
    return maj, valid


def pool2(x, how):
    """(2G, 2G) -> (G, G) by 2x2 block reduction."""
    g = x.shape[0] // 2
    b = x.reshape(g, 2, g, 2).transpose(0, 2, 1, 3).reshape(g, g, 4)
    return {"any": b.max(-1), "mean": b.mean(-1), "min": b.min(-1)}[how]


def ndwi_gradient8(img):
    """exp13 pixel control (NDWI gradient magnitude) pooled to the 8-px grid."""
    bo = Modality.SENTINEL2_L2A.band_order
    x = img.astype(np.float64)
    nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(x[bo.index("B03")] + x[bo.index("B08")], 1, None)
    gy, gx = np.gradient(nd)
    s = img.shape[1]
    return np.hypot(gx, gy).reshape(s // P8, P8, s // P8, P8).mean(axis=(1, 3))


def rank_avg(x):
    order = np.argsort(x, kind="stable")
    s = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(sig, err):
    """Tie-aware ROC AUC of sig (higher = more suspicious) for err in {0,1}."""
    pos = err > 0
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank_avg(np.asarray(sig, dtype=np.float64))
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wlt(d, eps=1e-12):
    d = np.asarray(d, dtype=np.float64)
    w, l = int((d > eps).sum()), int((d < -eps).sum())
    return {"wlt": [w, l, int(len(d) - w - l)], "sign_p": sign_test_p(w, w + l) if w + l else 1.0,
            "median": float(np.median(d)) if len(d) else None, "n": int(len(d))}


class NdwiStandin:
    """Smoke-only stand-in for the exp13 head when exp11_feats.npz is absent:
    NDWI-threshold water at 4 px (logit = 20 * mean NDWI)."""

    def __init__(self, img):
        bo = Modality.SENTINEL2_L2A.band_order
        x = img.astype(np.float64)
        nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(x[bo.index("B03")] + x[bo.index("B08")], 1, None)
        s = img.shape[1]
        self.logit = 20.0 * nd.reshape(s // P4, P4, s // P4, P4).mean(axis=(1, 3))
        self.p = 1 / (1 + np.exp(-self.logit))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, 2 scenes, 48-px crops, _smoke outputs")
    ap.add_argument("--crop", type=int, default=None, help="window side in px (multiple of 8); default 128, smoke 48")
    ap.add_argument("--scenes", type=str, default=None, help="comma-separated scene subset")
    ap.add_argument("--timeout", type=float, default=None, help="per-scene network timeout in s (default 600, smoke 60)")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    smoke = args.smoke
    sfx = "_smoke" if smoke else ""
    crop = args.crop or (48 if smoke else SIZE)
    assert crop % P8 == 0 and crop <= SIZE
    timeout = args.timeout or (60.0 if smoke else 600.0)
    dev = "cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    G8, G4 = crop // P8, crop // P4

    summary = {"experiment": "exp27_decoder_worldcover", "smoke": smoke, "device": dev, "crop_px": crop,
               "patch_size": P8, "probe_pooling": "any (majority as sensitivity)", "status": "started",
               "scenes": {}, "skipped": {}, "design_notes": [], "tests": {}}
    rows, per, cache = [], {}, {}

    def write_summary():
        summary["wall_s"] = time.time() - t_start
        with open(os.path.join(OUT, f"exp27_summary{sfx}.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

    try:
        run(args, smoke, sfx, crop, timeout, dev, G8, G4, summary, rows, per, cache)
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["fatal_error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"])
    finally:
        write_summary()
        print(f"wrote exp/out/exp27_summary{sfx}.json (status {summary['status']}, {time.time() - t_start:.0f}s)")
    if summary["status"] != "ok":
        sys.exit(1)


def run(args, smoke, sfx, crop, timeout, dev, G8, G4, summary, rows, per, cache):
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    coords = load_coords()
    feats_path = os.path.join(OUT, "exp11_feats.npz")
    feats = dict(np.load(feats_path, allow_pickle=True)) if os.path.exists(feats_path) else None
    if feats is None and not smoke:
        raise FileNotFoundError("exp/out/exp11_feats.npz is required to reproduce the exp13 disagreement set")
    head = None
    if feats is not None:
        z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
        torch.manual_seed(0)
        head = train_logistic_head(torch.tensor(feats["tr_base"]), z3["tr_labels"])   # exp13 seed-0 Base head
        summary["probe"] = "exp13 seed-0 Base head (Katima) from exp11_feats.npz"
    else:
        summary["probe"] = "NDWI stand-in (smoke only; exp11_feats.npz absent)"
    print(f"probe: {summary['probe']}; device {dev}; crop {crop}px -> {G8}x{G8} cells at 8 px", flush=True)

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    names = [n for n in names if n not in NON_RULE and n in coords and (feats is None or f"{n}_base0" in feats)]
    if args.scenes:
        names = [n for n in names if n in set(args.scenes.split(","))]
    if smoke:
        pref = [n for n in ("barotse", "zambezi_50") if n in names]
        names = (pref + [n for n in names if n not in pref])[:2]
    summary["scene_candidates"] = names

    # geo cache (own prefix; seeded from exp23's when present; smoke always fetches)
    geo_path = os.path.join(OUT, f"exp27_geo{sfx}.npz")
    geo = {}
    if not smoke:
        for p in (os.path.join(OUT, "exp23_geo.npz"), geo_path):
            if os.path.exists(p):
                z = dict(np.load(p, allow_pickle=True))
                geo.update({k: v for k, v in z.items() if k.endswith(("_wc21", "_wc20", "_crs", "_transform"))})
        print(f"geo cache: {sum(1 for k in geo if k.endswith('_wc21'))} scenes with WorldCover 2021 pixel maps")

    t0 = time.time()
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(dev).eval()
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)
    protos = prototypes(model, dev)                       # (11, D)
    pn = l2n(protos)
    proto_cos = pn @ pn.T
    summary["design_notes"] += [
        "worldcover field zeroed in the online sample; decoded tokens verified identical for real vs zeroed values (no leakage)",
        "target = target_encoder patch projection with token_exit_cfg=0 (training target path; ema_decay=(1,1) -> init-time projection)",
        f"class prototypes are collinear (token = bias + normalised code * kernel sum): min pairwise cosine {proto_cos.min():.4f}, "
        "so the nearest-prototype classifier is a one-dimensional code family",
        "per-patch loss = patch-discrimination CE (tau 0.1) over the scene's worldcover targets only (training pooled all decoded tokens of the sample)",
        "confidence baseline at 8 px = negative minimum |logit| over the four 4-px sub-patches; control = NDWI gradient pooled to 8 px",
    ]
    cache["protos"] = protos
    cache["codes"] = np.array(CODES)
    water_i = CODES.index(WATER)

    for name in names:
        ts = time.time()
        img_full = scenes[f"{name}_img"]
        date = scenes[f"{name}_date"]
        lab4_cached = scenes[f"{name}_lab"][:G4, :G4].astype(bool)
        img = img_full[:, :crop, :crop]
        # --- probe (exp13 disagreement set at 4 px)
        if head is not None:
            f0 = torch.tensor(feats[f"{name}_base0"])[:G4, :G4]
            p4 = predict_head(f0, *head)
            logit4 = predict_logit(f0, *head)
        else:
            st = NdwiStandin(img)
            p4, logit4 = st.p, st.logit
        err4 = (p4 > 0.5) != lab4_cached
        if not smoke and err4.sum() < MIN_ERR4:
            summary["skipped"][name] = f"exp13 rule: {int(err4.sum())} < {MIN_ERR4} disagreements at 4 px"
            print(f"{name}: skipped ({summary['skipped'][name]})"); continue
        # --- labels: WorldCover 2021 (and 2020) pixel maps on the scene grid
        labels_source = "fetched"
        if f"{name}_wc21" not in geo:
            try:
                g = run_with_timeout(fetch_geo, timeout, name, *coords[name], img_full)
                geo[f"{name}_wc21"], geo[f"{name}_crs"], geo[f"{name}_transform"] = g["wc21"], np.array(g["crs"]), g["transform"]
                if g["wc20"] is not None:
                    geo[f"{name}_wc20"] = g["wc20"]
                np.savez(geo_path, **geo)
            except BaseException as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                if smoke:
                    labels_source = "synthetic (smoke fallback: " + msg + ")"
                    geo[f"{name}_wc21"] = synthetic_classes(scenes[f"{name}_lab"], SIZE + PAD)
                    print(f"  {name}: network unavailable ({msg}); synthetic class map")
                else:
                    summary["skipped"][name] = "label fetch failed: " + msg
                    print(f"{name}: skipped ({summary['skipped'][name]})", flush=True); continue
        wc21 = geo[f"{name}_wc21"][:crop, :crop]
        wc20 = geo[f"{name}_wc20"][:crop, :crop] if f"{name}_wc20" in geo else None
        lab4_refetch = pool_to_patches(wc21 == WATER, P4) > 0.5
        n_mismatch4 = int((lab4_refetch != lab4_cached).sum()) if labels_source == "fetched" else -1
        if n_mismatch4 > 0:
            print(f"  {name}: refetched 4-px water labels differ from the cache on {n_mismatch4} patches; cache kept for the disagreement set")
        lab8_class, valid8 = block_majority(wc21, P8)
        lab8_water = pool_to_patches(wc21 == WATER, P8) > 0.5
        # --- model: decoder tokens and target tokens
        pred = decode_worldcover(model, s2_sample(img, date, dev))          # (G8, G8, D)
        tok21 = target_tokens(model, wc21, dev)                              # (G8, G8, D)
        pr, tk = l2n(pred), l2n(tok21)
        cos_label = (pr * tk).sum(-1)
        scores = (pr.reshape(-1, pr.shape[-1]) @ tk.reshape(-1, tk.shape[-1]).T) / TAU
        scores = scores - scores.max(1, keepdims=True)
        logp = scores - np.log(np.exp(scores).sum(1, keepdims=True))
        loss = -np.diag(logp).reshape(G8, G8)
        retrieval_top1 = (scores.argmax(1) == np.arange(scores.shape[0])).reshape(G8, G8)
        cos_p = pr.reshape(-1, pr.shape[-1]) @ pn.T                          # (N, 11)
        srt = np.sort(cos_p, 1)
        margin = (srt[:, -1] - srt[:, -2]).reshape(G8, G8)
        dec_cls = np.array(CODES)[cos_p.argmax(1)].reshape(G8, G8)
        cos_water = cos_p[:, water_i].reshape(G8, G8)
        dec_water = dec_cls == WATER
        # --- probe at 8 px
        err8 = pool2(err4.astype(float), "any") > 0
        err8_maj = pool2(err4.astype(float), "mean") > 0.5
        p8 = pool2(p4, "mean")
        probe8_water = p8 > 0.5
        conf8 = -pool2(np.abs(logit4), "min")
        control8 = ndwi_gradient8(img)
        v = valid8.flatten()
        e = err8.flatten()[v].astype(float)
        sigs = {
            "confidence (baseline)": conf8,
            "decoder-label disagreement (1-cos)": 1 - cos_label,
            "decoder per-patch loss": loss,
            "decoder margin (neg)": -margin,
            "decoder-vs-probe disagreement": (dec_water != probe8_water).astype(float),
            "control (NDWI gradient)": control8,
        }
        s = {k: x.flatten()[v] for k, x in sigs.items()}
        n_err8 = int(e.sum())
        scorable = n_err8 >= MIN_ERR8 and n_err8 <= len(e) - MIN_ERR8
        t1 = {k: eaurc(s[k], e) for k in sigs} if scorable else None
        auc = {k: roc_auc(s[k], e) for k in sigs} if scorable else None
        e_maj = err8_maj.flatten()[v].astype(float)
        t1_maj = {k: eaurc(s[k], e_maj) for k in sigs} if (MIN_ERR8 <= e_maj.sum() <= len(e) - MIN_ERR8) else None
        # --- T2 / T3
        lw, dw, pw = lab8_water.flatten()[v], dec_water.flatten()[v], probe8_water.flatten()[v]
        acc_dec = float((dw == lw).mean()); acc_probe8 = float((pw == lw).mean())
        acc_dec_cls = float((dec_cls.flatten()[v] == lab8_class.flatten()[v]).mean())
        dec_err = dw != lw
        eb = e > 0
        shared = float(dec_err[eb].mean()) if eb.any() else float("nan")
        base_rate = float(dec_err[~eb].mean()) if (~eb).any() else float("nan")
        conv = float(eb[dec_err].mean()) if dec_err.any() else float("nan")
        # --- T4
        t4 = None
        if wc20 is not None:
            tok20 = target_tokens(model, wc20, dev)
            cos20 = (pr * l2n(tok20)).sum(-1)
            diff = ((wc20 != wc21).reshape(G8, P8, G8, P8).any(axis=(1, 3))) & valid8
            d = (cos_label - cos20)[diff]
            t4 = {"n_diff": int(diff.sum()), "wins_2021": int((d > 1e-12).sum()), "losses_2021": int((d < -1e-12).sum()),
                  "mean_diff": float(d.mean()) if len(d) else float("nan")}
            cache[f"{name}_tok20"] = tok20
        # --- record
        per[name] = {"t1": t1, "t1_maj": t1_maj, "auc": auc, "acc_dec": acc_dec, "acc_probe8": acc_probe8,
                     "shared": shared, "base_rate": base_rate, "n_err8": n_err8, "t4": t4}
        row = {"scene": name, "n_cells8": int(v.size), "n_valid8": int(v.sum()), "n_err4": int(err4.sum()),
               "n_err8_any": n_err8, "n_err8_maj": int(e_maj.sum()), "labels_source": labels_source,
               "lab4_cache_mismatch": n_mismatch4, "acc_probe4_vs_cache": float(1 - err4.mean()),
               "acc_probe8": acc_probe8, "acc_dec_water": acc_dec, "acc_dec_class11": acc_dec_cls,
               "retrieval_top1": float(retrieval_top1.flatten()[v].mean()),
               "water_rate_label8": float(lw.mean()), "water_rate_dec8": float(dw.mean()), "water_rate_probe8": float(pw.mean()),
               "mean_cos_label": float(cos_label.flatten()[v].mean()),
               "mean_cos_label_err": float(cos_label.flatten()[v][eb].mean()) if eb.any() else float("nan"),
               "mean_cos_label_ok": float(cos_label.flatten()[v][~eb].mean()) if (~eb).any() else float("nan"),
               "mean_loss": float(loss.flatten()[v].mean()), "mean_margin": float(margin.flatten()[v].mean()),
               "mean_cos_water_on_water": float(cos_water.flatten()[v][lw].mean()) if lw.any() else float("nan"),
               "mean_cos_water_on_nonwater": float(cos_water.flatten()[v][~lw].mean()) if (~lw).any() else float("nan"),
               "t3_shared_share_among_probe_err": shared, "t3_dec_disagree_share_among_probe_ok": base_rate,
               "t3_probe_disagree_share_among_dec_err": conv}
        for k in sigs:
            row[f"eaurc[{k}]"] = t1[k] if t1 else float("nan")
            row[f"auc[{k}]"] = auc[k] if auc else float("nan")
        row.update({"t4_n_diff": t4["n_diff"] if t4 else -1, "t4_wins_2021": t4["wins_2021"] if t4 else -1,
                    "t4_losses_2021": t4["losses_2021"] if t4 else -1, "t4_mean_diff": t4["mean_diff"] if t4 else float("nan")})
        rows.append(row)
        cache[f"{name}_pred"], cache[f"{name}_tok21"] = pred, tok21
        for k, x in (("cos_label", cos_label), ("loss", loss), ("margin", margin), ("cos_water", cos_water), ("dec_cls", dec_cls),
                     ("lab8_class", lab8_class), ("valid8", valid8), ("err8", err8), ("err4", err4), ("p8", p8), ("conf8", conf8)):
            cache[f"{name}_{k}"] = x
        summary["scenes"][name] = {"n_err4": int(err4.sum()), "n_err8": n_err8, "labels_source": labels_source, "seconds": time.time() - ts}
        t1s = (f"1-cos {t1['decoder-label disagreement (1-cos)']:.4f} vs conf {t1['confidence (baseline)']:.4f} vs ctrl {t1['control (NDWI gradient)']:.4f}"
               if t1 else "T1 not scorable")
        print(f"{name:14s} err4 {int(err4.sum()):3d} err8 {n_err8:3d}/{int(v.sum())} | cos(pred,label) err {row['mean_cos_label_err']:+.3f} ok {row['mean_cos_label_ok']:+.3f} | "
              f"E-AURC {t1s} | T2 dec {acc_dec:.3f} probe {acc_probe8:.3f} | T3 shared {shared:.2f} base {base_rate:.2f}"
              + (f" | T4 diff {t4['n_diff']} 2021 closer {t4['wins_2021']}/{t4['losses_2021']}" if t4 else "") + f" | {time.time() - ts:.1f}s", flush=True)

    # ------------------------------------------------------------------ outputs
    np.savez(os.path.join(OUT, f"exp27_decoder_tokens{sfx}.npz"), **cache)
    csv_path = os.path.join(OUT, f"exp27_decoder_worldcover{sfx}.csv")
    with open(csv_path, "w", newline="") as fh:
        fields = list(rows[0].keys()) if rows else ["scene"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.relpath(csv_path, os.path.dirname(OUT))} ({len(rows)} scenes)")

    sn = sorted(per)
    summary["n_scenes"] = len(sn)
    # T1: each signal vs confidence and vs control (E-AURC, lower is better -> gain = other - signal)
    sc = [n for n in sn if per[n]["t1"] is not None]
    t1 = {"n_scenes_scored": len(sc), "min_err8": MIN_ERR8, "signals": {}}
    for k in SIGNALS:
        if k == "confidence (baseline)":
            continue
        t1["signals"][k] = {
            "prespecified": k in PRESPECIFIED,
            "vs_confidence": wlt([per[n]["t1"]["confidence (baseline)"] - per[n]["t1"][k] for n in sc]),
            "vs_control": wlt([per[n]["t1"]["control (NDWI gradient)"] - per[n]["t1"][k] for n in sc]),
            "median_eaurc": float(np.median([per[n]["t1"][k] for n in sc])) if sc else None,
            "median_roc_auc": float(np.nanmedian([per[n]["auc"][k] for n in sc])) if sc else None,
            "majority_pooling_vs_confidence": wlt([per[n]["t1_maj"]["confidence (baseline)"] - per[n]["t1_maj"][k] for n in sc if per[n]["t1_maj"]]),
        }
    t1["median_eaurc_confidence"] = float(np.median([per[n]["t1"]["confidence (baseline)"] for n in sc])) if sc else None
    t1["median_roc_auc_confidence"] = float(np.nanmedian([per[n]["auc"]["confidence (baseline)"] for n in sc])) if sc else None
    summary["tests"]["T1_decoder_agreement_ranks_probe_disagreements"] = t1
    # T2
    summary["tests"]["T2_decoder_vs_probe_accuracy_8px"] = {
        "decoder_minus_probe": wlt([per[n]["acc_dec"] - per[n]["acc_probe8"] for n in sn]),
        "median_acc_decoder": float(np.median([per[n]["acc_dec"] for n in sn])) if sn else None,
        "median_acc_probe8": float(np.median([per[n]["acc_probe8"] for n in sn])) if sn else None,
        "pooled_acc_decoder": float(np.average([per[n]["acc_dec"] for n in sn], weights=[r["n_valid8"] for r in rows])) if sn else None,
        "pooled_acc_probe8": float(np.average([per[n]["acc_probe8"] for n in sn], weights=[r["n_valid8"] for r in rows])) if sn else None,
        "median_acc_decoder_class11": float(np.median([r["acc_dec_class11"] for r in rows])) if rows else None,
        "median_retrieval_top1": float(np.median([r["retrieval_top1"] for r in rows])) if rows else None,
    }
    # T3
    ok3 = [n for n in sn if np.isfinite(per[n]["shared"]) and np.isfinite(per[n]["base_rate"])]
    tot_err = sum(per[n]["n_err8"] for n in ok3)
    summary["tests"]["T3_shared_vs_probe_only_errors"] = {
        "pooled_shared_share_among_probe_err": float(sum(per[n]["shared"] * per[n]["n_err8"] for n in ok3) / tot_err) if tot_err else None,
        "pooled_probe_only_share": float(1 - sum(per[n]["shared"] * per[n]["n_err8"] for n in ok3) / tot_err) if tot_err else None,
        "median_shared_share": float(np.median([per[n]["shared"] for n in ok3])) if ok3 else None,
        "median_dec_disagree_share_among_probe_ok": float(np.median([per[n]["base_rate"] for n in ok3])) if ok3 else None,
        "enrichment_shared_minus_base_rate": wlt([per[n]["shared"] - per[n]["base_rate"] for n in ok3]),
    }
    # T4
    ok4 = [n for n in sn if per[n]["t4"] is not None and per[n]["t4"]["n_diff"] > 0]
    tw = sum(per[n]["t4"]["wins_2021"] for n in ok4); tl = sum(per[n]["t4"]["losses_2021"] for n in ok4)
    summary["tests"]["T4_2021_vs_2020_on_differing_cells"] = {
        "n_scenes_with_2020": sum(1 for n in sn if per[n]["t4"] is not None), "n_scenes_with_differing_cells": len(ok4),
        "pooled_cells": {"n_diff": int(sum(per[n]["t4"]["n_diff"] for n in ok4)), "closer_to_2021": int(tw), "closer_to_2020": int(tl),
                         "share_2021": float(tw / (tw + tl)) if tw + tl else None,
                         "binomial_p_indicative": sign_test_p(tw, tw + tl) if tw + tl else None},
        "per_scene_mean_diff": wlt([per[n]["t4"]["mean_diff"] for n in ok4]),
        "note": "skipped where WorldCover 2020 could not be fetched; pooled cells are spatially dependent, the per-scene sign test is primary",
    }
    print(json.dumps(summary["tests"], indent=1, default=float))
    if not args.no_fig and rows:
        pick = max(rows, key=lambda r: r["n_err8_any"])["scene"]
        make_figure(pick, cache, os.path.join(OUT, f"exp27_decoder_worldcover{sfx}.png"))
        summary["figure_scene"] = pick


def make_figure(name, cache, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from oe_inferencex import figstyle
    figstyle.setup()
    colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(CODES) + 0.5, 1), cmap.N)
    idx = {c: i for i, c in enumerate(CODES)}
    to_idx = np.vectorize(lambda c: idx.get(int(c), 0))
    lab, dec = cache[f"{name}_lab8_class"], cache[f"{name}_dec_cls"]
    cos, loss, err8 = cache[f"{name}_cos_label"], cache[f"{name}_loss"], cache[f"{name}_err8"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    panels = [(to_idx(lab), "WorldCover 2021 majority class (8 px = 80 m)", dict(cmap=cmap, norm=norm)),
              (to_idx(dec), "decoder nearest-prototype class", dict(cmap=cmap, norm=norm)),
              (cos, "cos(decoder token, label token)", dict(cmap="viridis")),
              (loss, "decoder per-patch loss (CE, tau 0.1); red = probe disagreement", dict(cmap="magma"))]
    for i, (ax, (img, title, kw)) in enumerate(zip(axes, panels)):
        im = ax.imshow(img, **kw)
        ax.set_title(title); ax.set_xlabel("cell column (80 m/cell)"); ax.set_ylabel("cell row (80 m/cell)")
        figstyle.letter(ax, i)
        if i < 2:
            cb = fig.colorbar(im, ax=ax, ticks=range(len(CODES)), shrink=0.8); cb.ax.set_yticklabels(CODE_NAMES, fontsize=6)
        else:
            cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label(title.split(";")[0], fontsize=7)
        if i >= 2:
            rr, cc = np.nonzero(err8)
            ax.scatter(cc, rr, s=28, facecolors="none", edgecolors="red", linewidths=0.8)
    fig.suptitle(f"exp27: the decoder's own WorldCover prediction, scene {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"figure written ({name})")


if __name__ == "__main__":
    main()

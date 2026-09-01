"""Exp 02: full minimal audit slice, every stage present.

Train AOI (Katima Mulilo) -> water heads on Nano + Base embeddings.
Eval AOI (Kazungula, ~110 km away) -> predictions, then:
  E_case   = |p_nano - p_base|
  E_geo    = OSM river centerline patches predicted dry by BOTH models
  baseline = max-softmax uncertainty of the Base head (the thing to beat)
Scored against ESA WorldCover water on the eval window via risk-coverage/AURC.
WorldCover is weak truth, fine for a skeleton run.
"""
import json
import urllib.request

import numpy as np
import rasterio.warp
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window, s2_to_sample, embed
from oe_inferencex.evidence import (
    train_logistic_head, predict_head, risk_coverage, rasterize_polyline, pool_to_patches,
)

TRAIN = (24.302, -17.485)   # Zambezi at Katima Mulilo
EVAL = (25.263, -17.788)    # Zambezi at Kazungula
SIZE, PATCH = 128, 4
GRID = SIZE // PATCH


def load_aoi(lon, lat):
    image, (d, m0, y), (crs, transform) = fetch_s2_window(lon, lat, SIZE)
    wc = fetch_worldcover_window(lon, lat, crs, transform, SIZE)
    water_frac = pool_to_patches(wc == 80, PATCH)
    labels = (water_frac > 0.5).astype(np.float32)
    print(f"  water patches: {labels.sum():.0f}/{labels.size} ({100*labels.mean():.1f}%)")
    return image, (d, m0, y), (crs, transform), labels


def osm_river_centerline(crs, transform):
    """OSM waterway=river nodes inside the eval window, as pixel polylines."""
    corners = [transform * p for p in [(0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)]]
    xs, ys = zip(*corners)
    lons, lats = rasterio.warp.transform(crs, "EPSG:4326", xs, ys)
    bbox = f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"
    query = f'[out:json][timeout:30];way["waterway"="river"]({bbox});out geom;'
    mirrors = [
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    ways = None
    for attempt in range(len(mirrors)):
        url = mirrors[attempt % len(mirrors)]
        try:
            req = urllib.request.Request(
                url,
                data=("data=" + urllib.parse.quote(query)).encode(),
                headers={"User-Agent": "oe-inferencex-exp02"},
            )
            ways = json.loads(urllib.request.urlopen(req, timeout=90).read())["elements"]
            break
        except Exception as exc:
            print(f"  overpass attempt {attempt+1} ({url}) failed: {exc}")
    if ways is None:
        raise RuntimeError("all overpass attempts failed")
    print(f"  OSM river ways in bbox: {len(ways)}")
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    inv = ~transform
    for way in ways:
        pts = []
        for nd in way["geometry"]:
            x, y = rasterio.warp.transform("EPSG:4326", crs, [nd["lon"]], [nd["lat"]])
            c, r = inv * (x[0], y[0])
            pts.append((r, c))
        if len(pts) > 1:
            mask |= rasterize_polyline(pts, SIZE)
    return pool_to_patches(mask, PATCH) > 0


CACHE = "exp/out/exp02_cache.npz"


def compute_probs():
    torch.manual_seed(0)
    print("train AOI (Katima Mulilo):")
    tr_img, tr_date, _, tr_labels = load_aoi(*TRAIN)
    print("eval AOI (Kazungula):")
    ev_img, ev_date, (crs, transform), ev_labels = load_aoi(*EVAL)

    models = {m: load_model_from_id(m) for m in (ModelID.OLMOEARTH_V1_NANO, ModelID.OLMOEARTH_V1_BASE)}
    tr_sample, ev_sample = s2_to_sample(tr_img, *tr_date), s2_to_sample(ev_img, *ev_date)

    probs = {}
    for mid, model in models.items():
        f_tr, f_ev = embed(model, tr_sample, PATCH), embed(model, ev_sample, PATCH)
        w, b = train_logistic_head(f_tr, tr_labels)
        probs[mid.value] = predict_head(f_ev, w, b)
        acc = ((probs[mid.value] > 0.5) == ev_labels.astype(bool)).mean()
        print(f"{mid.value}: eval acc vs WorldCover = {acc:.3f}")
    np.savez(CACHE, p_nano=list(probs.values())[0], p_base=list(probs.values())[1],
             ev_labels=ev_labels, crs_wkt=str(crs.to_wkt()),
             transform=np.array(transform)[:6])
    return probs, ev_labels, (crs, transform)


def main():
    import os
    import rasterio
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        p_nano, p_base, ev_labels = z["p_nano"], z["p_base"], z["ev_labels"]
        ev_geo = (rasterio.crs.CRS.from_wkt(str(z["crs_wkt"])),
                  rasterio.Affine(*z["transform"]))
        print("loaded cached probs/labels")
    else:
        probs, ev_labels, ev_geo = compute_probs()
        p_nano, p_base = probs.values()
    errors = ((p_base > 0.5) != ev_labels.astype(bool)).astype(np.float64)

    e_case = np.abs(p_nano - p_base)
    baseline = 1 - np.maximum(p_base, 1 - p_base)  # max-softmax uncertainty

    centerline = osm_river_centerline(*ev_geo)
    both_dry = (p_nano < 0.5) & (p_base < 0.5)
    e_geo_flags = centerline & both_dry
    print(f"E_geo: centerline patches={centerline.sum()}, "
          f"consensus-dry on centerline={e_geo_flags.sum()}")

    results = {}
    for name, sig in (("E_case", e_case), ("baseline max-softmax", baseline)):
        cov, risk, aurc = risk_coverage(sig, errors)
        results[name] = (cov, risk, aurc)
        print(f"{name}: AURC={aurc:.4f}")
    print(f"overall Base error rate: {errors.mean():.3f}")

    from oe_inferencex.figstyle import setup, map_panel, rc_panel
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    map_panel(fig, axes[0, 0], ev_labels, "ESA WorldCover 2021 water\n(reference)",
              "water patch (fraction > 0.5)", cmap="Blues", idx=0, vmin=0, vmax=1)
    map_panel(fig, axes[0, 1], p_base, "Base head water probability",
              "P(water)", cmap="Blues", idx=1, vmin=0, vmax=1)
    map_panel(fig, axes[0, 2], errors, "Base head vs reference\n(disagreement, counted as error)",
              "disagreement (binary)", cmap="Reds", idx=2, vmin=0, vmax=1)
    map_panel(fig, axes[1, 0], np.abs(p_nano - p_base), "E_case |Nano - Base|",
              "|p_Nano - p_Base|", cmap="magma", idx=3)
    map_panel(fig, axes[1, 1], centerline.astype(float) + e_geo_flags,
              "E_geo: OSM centerline (1)\n+ consensus-dry flags (2)",
              "0 = off-line, 1 = centerline,\n2 = flagged break", cmap="viridis", idx=4)
    rc_panel(axes[1, 2], results, "Kazungula scene (n=1024 patches)", idx=5)
    fig.suptitle("Full audit slice at Kazungula; heads trained at Katima Mulilo (110 km away)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("exp/out/exp02_full_slice.png", bbox_inches="tight")
    print("wrote exp/out/exp02_full_slice.png")


if __name__ == "__main__":
    main()

"""Loader for the AWF rslearn dataset (allenai/olmoearth_projects_awf).

Each window: 63x63 px @ 10 m, 12 monthly Sentinel-2 mosaics in three
band-group GeoTIFFs at native resolutions, and a label raster where exactly
one pixel carries the category (0-8) and everything else is 9 (fill).
"""
import glob
import json
import os

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

ROOT = "data/awf/dataset/windows/spatial_split"
CROP = 32  # 10 m pixels, centered on the labeled pixel
N_MONTHS = 12
FILL = 9

# band -> (group dir, index within group)
GROUPS = {
    "B02_B03_B04_B08": ["B02", "B03", "B04", "B08"],
    "B05_B06_B07_B8A_B11_B12": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "B01_B09": ["B01", "B09"],
}


def list_windows():
    """[(window_dir, split, label_row, label_col, category)] for all windows."""
    out = []
    for wdir in sorted(glob.glob(ROOT + "/*")):
        meta = json.load(open(wdir + "/metadata.json"))
        split = meta["options"].get("split")
        lab_path = wdir + "/layers/label/category/geotiff.tif"
        if not os.path.exists(lab_path):
            continue
        with rasterio.open(lab_path) as src:
            a = src.read(1)
        rows, cols = np.nonzero(a != FILL)
        if len(rows) != 1:
            continue
        out.append((wdir, split, int(rows[0]), int(cols[0]), int(a[rows[0], cols[0]])))
    return out


def load_window_full(wdir):
    """(63, 63, N_MONTHS, 12) raw stack in OlmoEarth band order. One disk pass."""
    band_order = Modality.SENTINEL2_L2A.band_order
    stack = np.zeros((63, 63, N_MONTHS, len(band_order)), dtype=np.float32)
    for t in range(N_MONTHS):
        layer = "sentinel2" if t == 0 else f"sentinel2.{t}"
        for group, bands in GROUPS.items():
            path = f"{wdir}/layers/{layer}/{group}/geotiff.tif"
            with rasterio.open(path) as src:
                data = src.read(
                    out_shape=(src.count, 63, 63),
                    resampling=Resampling.bilinear,
                )
            for bi, band in enumerate(bands):
                stack[:, :, t, band_order.index(band)] = data[bi]
    return stack


def crop_stack(full, r, c, shift=0):
    """Crop a full 63x63 stack around the labeled pixel at a given shift.
    Returns (crop, (pr, pc)): the crop and the label pixel within it."""
    r0 = min(max(r - CROP // 2 + shift, 0), 63 - CROP)
    c0 = min(max(c - CROP // 2 + shift, 0), 63 - CROP)
    return full[r0:r0 + CROP, c0:c0 + CROP], (r - r0, c - c0)


_normalizer = Normalizer(Strategy.COMPUTED)


def stacks_to_sample(stacks):
    """Batch of raw stacks (B, CROP, CROP, T, 12) -> MaskedOlmoEarthSample."""
    x = _normalizer.normalize(Modality.SENTINEL2_L2A, np.stack(stacks).astype(np.float64))
    b = x.shape[0]
    timestamps = torch.tensor([[15, m, 2023] for m in range(N_MONTHS)])[None].repeat(b, 1, 1)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32),
        sentinel2_l2a_mask=torch.ones((b, CROP, CROP, N_MONTHS, 3)) * MaskValue.ONLINE_ENCODER.value,
        timestamps=timestamps,
    )

"""Data access for experiments: Sentinel-2 windows, WorldCover labels, embeddings.

Network code lives here (Layer 2 territory); evidence math stays pure.
"""
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _catalog():
    return pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)


def fetch_s2_window(lon, lat, size=128, datetime="2024-06-01/2024-09-30", max_cloud=5):
    """Return (bands[12,H,W] int32, (day, month0, year), (crs, transform)) at 10 m."""
    search = _catalog().search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    item = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"  scene {item.id} cloud={item.properties['eo:cloud_cover']:.2f}%")
    band_order = Modality.SENTINEL2_L2A.band_order
    with rasterio.open(item.assets[band_order[0]].href) as src:
        crs, transform = src.crs, src.transform
        xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
    win_transform = transform * rasterio.Affine.translation(col - size // 2, row - size // 2)
    image = np.zeros((len(band_order), size, size), dtype=np.int32)
    for bi, band in enumerate(band_order):
        with rasterio.open(item.assets[band].href) as src:
            with WarpedVRT(src, crs=crs, transform=win_transform, width=size,
                           height=size, resampling=Resampling.bilinear) as vrt:
                image[bi] = vrt.read(1)
    dt = item.datetime
    return image, (dt.day, dt.month - 1, dt.year), (crs, win_transform)


def fetch_worldcover_window(lon, lat, crs, transform, size=128, version="2.0.0"):
    """ESA WorldCover classes on the given grid. Water = class 80.
    version "2.0.0" is the 2021 map (default), "1.0.0" the 2020 map."""
    search = _catalog().search(
        collections=["esa-worldcover"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        query={"esa_worldcover:product_version": {"eq": version}},
    )
    item = next(search.items())
    with rasterio.open(item.assets["map"].href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=size,
                       height=size, resampling=Resampling.nearest) as vrt:
            return vrt.read(1)


def s2_to_sample(image, day, month0, year):
    x = image.transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, x)
    size = image.shape[1]
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32),
        sentinel2_l2a_mask=torch.ones((1, size, size, 1, 3)) * MaskValue.ONLINE_ENCODER.value,
        timestamps=torch.tensor([day, month0, year])[None, None, :],
    )


def embed(model, sample, patch_size=4):
    """Pooled (H', W', D) features from a loaded model."""
    model.eval()
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=patch_size)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4]).squeeze(0)

"""8 km Sentinel-2 (before 2017-04-02, during 2018-03-28) and Sentinel-1 RTC VH (2018-03-11) around the located chip."""
import json, sys
import numpy as np, pystac_client, planetary_computer, rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
OUT = sys.argv[1]
loc = json.load(open(f"{OUT}/chip_location.json"))
cx, cy = loc["x0"] + 320, loc["y0"] - 320
HALF, RES, N = 4000, 10, 800
dst_t = from_origin(cx - HALF, cy + HALF, RES, RES)
crs = loc["crs"]
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
bbox_ll = transform_bounds(crs, "EPSG:4326", cx - HALF, cy - HALF, cx + HALF, cy + HALF)
def grab(item, assets):
    out = {}
    for a in assets:
        with rasterio.open(item.assets[a].href) as src:
            arr = np.zeros((N, N), np.float32)
            reproject(rasterio.band(src, 1), arr, dst_transform=dst_t, dst_crs=crs, resampling=Resampling.bilinear,
                      src_nodata=0, dst_nodata=0)
            out[a] = arr
    return out
meta = {"chip_location": loc, "center": [cx, cy], "half_m": HALF, "crs": crs}
for tag, dt in (() if len(sys.argv) > 2 else (("s2_before", "2017-04-02/2017-04-02"), ("s2_during", "2018-03-28/2018-03-28"))):
    its = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=dt).items(),
                 key=lambda i: i.properties.get("eo:cloud_cover", 100))
    it = its[0]; meta[tag] = it.id
    np.savez_compressed(f"{OUT}/{tag}.npz", **grab(it, ["B04", "B03", "B02"]))
    print(tag, it.id, it.properties.get("eo:cloud_cover"))
its = [i for i in cat.search(collections=["sentinel-1-rtc"], bbox=bbox_ll, datetime="2018-03-10/2018-03-12").items()
       if "20180311T164101" in i.id]                 # the pass the comparison figure used
print("s1", [i.id for i in its])
if its:
    it = its[0]; meta["s1_during"] = it.id
    np.savez_compressed(f"{OUT}/s1_during.npz", **grab(it, ["vh"]))
old = json.load(open(f"{OUT}/context.json")) if len(sys.argv) > 2 else {}
old.update(meta)
json.dump(old, open(f"{OUT}/context.json", "w"), indent=1)

"""Fetch the Sentinel-2 L2A true-colour chip of the demo's Dynamic World tile (2019-06-09) from Planetary Computer."""
import json, sys
import numpy as np, pystac_client, planetary_computer, rasterio
from rasterio.warp import transform as wtransform
from rasterio.windows import from_bounds
LAT, LON, OUT = -16.409944165598645, -71.8352855451905, sys.argv[1]
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
items = list(cat.search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [LON, LAT]},
                        datetime="2019-06-04/2019-06-14").items())
items.sort(key=lambda it: (abs((it.datetime.date() - __import__("datetime").date(2019, 6, 9)).days), it.properties.get("eo:cloud_cover", 100)))
print([(it.id, it.datetime.date().isoformat(), it.properties.get("eo:cloud_cover")) for it in items[:5]])
it = items[0]
bands = {}
for b in ("B04", "B03", "B02", "B08"):
    with rasterio.open(it.assets[b].href) as src:
        x, y = wtransform("EPSG:4326", src.crs, [LON], [LAT])
        half = 2560
        win = from_bounds(x[0] - half, y[0] - half, x[0] + half, y[0] + half, src.transform).round_offsets().round_lengths()
        bands[b] = src.read(1, window=win)
        crs = str(src.crs)
np.savez_compressed(f"{OUT}/peru_s2.npz", **bands)
json.dump({"item": it.id, "date": it.datetime.isoformat(), "cloud": it.properties.get("eo:cloud_cover"), "crs": crs,
           "shape": list(bands["B04"].shape)}, open(f"{OUT}/peru_s2.json", "w"), indent=1)
print("saved", bands["B04"].shape, crs)

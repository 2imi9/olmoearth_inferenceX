"""Locate the flood figure's 64 x 64 chip on Lake Shkoder: match its WorldFloods Sentinel-2 NDWI (2018-03-28) against a
20 m read of Planetary Computer's L2A scene of the same date over the lake's east shore."""
import json, sys
import numpy as np, pystac_client, planetary_computer, rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy.signal import fftconvolve
CHIP_NPZ, OUT = sys.argv[1], sys.argv[2]
BBOX = (19.28, 41.95, 19.62, 42.32)                 # lon/lat: Lake Shkoder's east shore, Shkoder city to Koplik
z = np.load(CHIP_NPZ)
w = z["wf_S2"].astype(float)
chip = (w[2] - w[7]) / np.maximum(w[2] + w[7], 1)    # B03, B08
chip20 = chip.reshape(32, 2, 32, 2).mean((1, 3))
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime="2018-03-28/2018-03-28").items())
print([(it.id, it.properties.get("eo:cloud_cover")) for it in items])
best = None
for it in items:
    with rasterio.open(it.assets["B03"].href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        shape = (max(1, int(win.height // 2)), max(1, int(win.width // 2)))
        g = src.read(1, window=win, out_shape=shape, boundless=True, fill_value=0).astype(float)
        wt = src.window_transform(win)
    with rasterio.open(it.assets["B08"].href) as src:
        n = src.read(1, window=win, out_shape=shape, boundless=True, fill_value=0).astype(float)
    valid = (g + n) > 0
    nd = np.where(valid, (g - n) / np.maximum(g + n, 1), np.nan)
    # normalized cross-correlation
    t = chip20 - chip20.mean(); t /= np.sqrt((t ** 2).sum())
    img = np.nan_to_num(nd, nan=0.0)
    ones = np.ones_like(t)
    s1 = fftconvolve(img, ones[::-1, ::-1], mode="valid"); s2 = fftconvolve(img ** 2, ones[::-1, ::-1], mode="valid")
    num = fftconvolve(img, t[::-1, ::-1], mode="valid")
    var = np.maximum(s2 - s1 ** 2 / t.size, 1e-9)
    ncc = num / np.sqrt(var)
    vfrac = fftconvolve(valid.astype(float), ones, mode="valid") / t.size
    ncc[vfrac < 0.99] = -1
    r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak = float(ncc[r, c])
    m = ncc.copy(); m[max(0, r - 8):r + 8, max(0, c - 8):c + 8] = -1
    second = float(m.max())
    x0, y0 = wt * (c * 2, r * 2)                       # top-left of the chip in the scene's CRS (10 m grid)
    cand = {"item": it.id, "crs": str(src.crs), "peak": peak, "second": second, "x0": float(x0), "y0": float(y0)}
    print(cand)
    if best is None or peak > best["peak"]:
        best = cand
json.dump(best, open(f"{OUT}/chip_location.json", "w"), indent=1)

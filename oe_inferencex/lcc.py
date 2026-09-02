"""Read windows of the published OlmoEarth LCC production rasters (Layer 2).

The continent-scale land cover change product (allenai/olmoearth_lcc) is
published as 9-band uint8 BigTIFFs, one per 32768x32768-px UTM tile (with
margins, so files are slightly larger), named EPSG:<code>_<col>_<row>.tif
where col/row are the tile's top-left pixel coordinates on the 10 m grid
(multiply by 10 for projection coordinates; rows are negative south of the
equator). The served rasters are in EPSG:3857 (about 9.55 m at 18 S), so the
CRS is read from the file, not the name. Bands (dataset card): 1 change
probability (0-255), 2 argmax of the pre/same change-category head, 3 argmax
of the post change-category head, 4 source land cover class, 5 destination
land cover class, 6 probability of band 2's category, 7 probability of band
3's category, 8-9 month-encoded change dates (1 = January 2015). No
confidence is exported for the land cover classes (bands 4-5). 0 = no
prediction in every band; the card advises thresholding band 1 at 128 before
reading bands 2-9.

The files are internally tiled (256x256, deflate, pixel-interleaved) and the
host supports byte ranges, but GDAL's HTTP layer stalls on the signed CDN
redirect, so this module reads the TIFF directly: it parses the first IFD,
fetches the tile offset and byte-count tables, and range-requests only the
tiles that cover the requested window. Nothing is downloaded whole.
"""
import math
import struct
import urllib.request
import zlib

import numpy as np

BASE = "https://huggingface.co/datasets/allenai/olmoearth_lcc/resolve/main/model_outputs/"
TILE_PX = 32768
BANDS = {1: "binary_change", 2: "pre_class", 3: "post_class", 4: "src_class", 5: "dst_class",
         6: "pre_score", 7: "post_score", 8: "ts_pre_month", 9: "ts_post_month"}
LAND_COVER = {0: "no prediction", 1: "bare", 2: "burnt", 3: "crops", 4: "fallow/shifting cultivation",
              5: "grassland", 6: "Lichen and moss", 7: "shrub", 8: "snow and ice", 9: "tree",
              10: "urban/built-up", 11: "water", 12: "wetland (herbaceous)"}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {3: "H", 4: "I", 12: "d", 16: "Q"}


def _resolve(name):
    req = urllib.request.Request(BASE + name, method="HEAD", headers={"User-Agent": "oe-inferencex"})
    return urllib.request.urlopen(req, timeout=30).geturl()


def _range(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": "oe-inferencex"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class LCCTile:
    """Lazy reader for one production tile; parses the header once."""

    def __init__(self, name):
        self.name = name
        self.url = _resolve(name)
        head = _range(self.url, 0, 1 << 20)
        bo = "<" if head[:2] == b"II" else ">"
        assert head[2:4] in (b"+\x00", b"\x00+"), "expected BigTIFF"
        self.bo = bo
        ifd = struct.unpack(bo + "Q", head[8:16])[0]
        n = struct.unpack(bo + "Q", head[ifd:ifd + 8])[0]
        tags = {}
        for i in range(n):
            tag, typ, cnt, val = struct.unpack(bo + "HHQ8s", head[ifd + 8 + i * 20: ifd + 8 + (i + 1) * 20])
            tags[tag] = (typ, cnt, val)
        self.tags = tags

        def scalar(tag):
            typ, cnt, val = tags[tag]
            return struct.unpack(bo + _TYPE_FMT[typ], val[:_TYPE_SIZE[typ]])[0]

        def array(tag):
            typ, cnt, val = tags[tag]
            size = _TYPE_SIZE[typ] * cnt
            raw = val[:size] if size <= 8 else _range(self.url, struct.unpack(bo + "Q", val)[0], size)
            return np.frombuffer(raw, dtype=bo + {"H": "u2", "I": "u4", "d": "f8", "Q": "u8"}[_TYPE_FMT[typ]], count=cnt)

        self.width, self.height = scalar(256), scalar(257)
        self.tile_w, self.tile_h = scalar(322), scalar(323)
        self.bands = scalar(277)
        self.compression = scalar(259)
        assert self.compression == 8, f"unexpected compression {self.compression}"
        self.tile_offsets = array(324)
        self.tile_counts = array(325)
        self.tiles_across = math.ceil(self.width / self.tile_w)
        scale = array(33550)
        tie = array(33922)
        # origin of pixel (0,0) in projection coordinates (tiepoint maps pixel (i,j)->(x,y))
        self.x0 = tie[3] - tie[0] * scale[0]
        self.y0 = tie[4] + tie[1] * scale[1]
        self.res_x, self.res_y = float(scale[0]), float(scale[1])
        # CRS from the GeoKey directory (ProjectedCSTypeGeoKey 3072); the file
        # name carries the rslearn tile grid, which need not be the raster CRS.
        keys = array(34735)
        self.epsg = None
        for i in range(4, len(keys), 4):
            if keys[i] == 3072:
                self.epsg = int(keys[i + 3])
        if self.epsg is None:
            self.epsg = int(name.split("_")[0].split(":")[1])
        self.bounds = (self.x0, self.y0 - self.height * self.res_y, self.x0 + self.width * self.res_x, self.y0)

    def index(self, x, y):
        """Projection coordinates -> (row, col); raises if outside the raster."""
        row, col = int((self.y0 - y) / self.res_y), int((x - self.x0) / self.res_x)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"point ({x:.0f}, {y:.0f}) outside {self.name} bounds {self.bounds} (EPSG:{self.epsg})")
        return row, col

    def _tile(self, ti, tj):
        k = ti * self.tiles_across + tj
        off, cnt = int(self.tile_offsets[k]), int(self.tile_counts[k])
        if cnt == 0:
            return np.zeros((self.tile_h, self.tile_w, self.bands), dtype=np.uint8)
        raw = zlib.decompress(_range(self.url, off, cnt))
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.tile_h, self.tile_w, self.bands)

    def read(self, row, col, size):
        """Read a (bands, size, size) window whose top-left pixel is (row, col)."""
        out = np.zeros((size, size, self.bands), dtype=np.uint8)
        t0, t1 = row // self.tile_h, (row + size - 1) // self.tile_h
        s0, s1 = col // self.tile_w, (col + size - 1) // self.tile_w
        for ti in range(t0, t1 + 1):
            for tj in range(s0, s1 + 1):
                tile = self._tile(ti, tj)
                r_a, c_a = ti * self.tile_h, tj * self.tile_w
                rr0, rr1 = max(row, r_a), min(row + size, r_a + self.tile_h)
                cc0, cc1 = max(col, c_a), min(col + size, c_a + self.tile_w)
                out[rr0 - row:rr1 - row, cc0 - col:cc1 - col] = tile[rr0 - r_a:rr1 - r_a, cc0 - c_a:cc1 - c_a]
        return out.transpose(2, 0, 1)


def tile_for(lon, lat):
    """Tile name containing a WGS84 point (UTM zone from longitude)."""
    import rasterio.warp
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    xs, ys = rasterio.warp.transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    col_px, row_px = xs[0] / 10.0, -ys[0] / 10.0
    tcol = int(math.floor(col_px / TILE_PX)) * TILE_PX
    trow = int(math.floor(row_px / TILE_PX)) * TILE_PX
    return f"EPSG:{epsg}_{tcol}_{trow}.tif", epsg


def read_window(lon, lat, size=128, tile=None):
    """Read a size x size window centred on a WGS84 point.
    Returns (array [9, size, size] uint8, affine transform tuple, epsg, tile name)."""
    import rasterio.warp
    name = tile or tile_for(lon, lat)[0]
    t = LCCTile(name)
    xs, ys = rasterio.warp.transform("EPSG:4326", f"EPSG:{t.epsg}", [lon], [lat])
    row, col = t.index(xs[0], ys[0])
    r0, c0 = row - size // 2, col - size // 2
    arr = t.read(r0, c0, size)
    transform = (t.res_x, 0.0, t.x0 + c0 * t.res_x, 0.0, -t.res_y, t.y0 - r0 * t.res_y)
    return arr, transform, t.epsg, name


def change_probability(arr):
    """Band 1 as a probability in [0, 1]; 0 where no prediction."""
    return arr[0].astype(np.float64) / 255.0


def nodata_mask(arr):
    return np.all(arr == 0, axis=0)

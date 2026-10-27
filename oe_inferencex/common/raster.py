"""Rasters in and out for the command line: a GeoTIFF or .npy read with its masks, written back on the same grid."""
import os

import numpy as np


# ----------------------------------------------------------------------------- IO
def read_raster(path, nodata=None):
    """(array, valid mask, geo) where geo is None for .npy and {transform, crs} for rasters. One band comes back as
    (H, W) from either format. A pixel is no-data where any band is not finite, where every band holds the no-data
    value (GDAL's dataset mask), or, on a GeoTIFF, where GDAL's mask says so (an internal mask, a .msk file, or an alpha
    band beside one data band; _mask_bands says why only there)."""
    if path.endswith(".npy"):
        # a mistyped path is a named refusal, not a traceback (bug hunt of 2026-10-06)
        try:
            a = np.load(path)
        except FileNotFoundError:
            raise SystemExit(f"{path}: no such file") from None
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{path}: not an array numpy can read ({exc})") from None
        if a.ndim not in (2, 3) or a.dtype.kind not in "biuf" or 0 in a.shape:
            # a 0-d, 1-d or 4-d array, strings or an empty band axis ended in tracebacks (the bug hunt of 25 October
            # 2026)
            raise SystemExit(f"{path}: a raster is a 2-D (H, W) or 3-D (C, H, W) array of numbers; this one is "
                             f"{a.ndim}-D {a.dtype} of shape {a.shape}")
        nd, geo, gdal_valid, alpha_valid = nodata, None, None, None
    else:
        try:
            import rasterio
            from rasterio.errors import RasterioIOError
        except ImportError as ex:  # pragma: no cover
            raise SystemExit("reading rasters needs rasterio (pip install 'olmoearth-inferencex[geo]'); .npy inputs need nothing") from ex
        try:
            src = rasterio.open(path)
        except RasterioIOError as exc:
            raise SystemExit(f"{path}: no such file" if "://" not in path and not os.path.exists(path)
                             else f"{path}: not a raster rasterio can read ({exc})") from None
        with src:
            a, gdal_valid, alpha_valid = _read_masked(src)
            nd = src.nodata if nodata is None else nodata
            geo = {"transform": src.transform, "crs": src.crs}
    valid = np.isfinite(a).all(axis=0) if a.ndim == 3 else np.isfinite(a)
    if not path.endswith(".npy") and gdal_valid is not None:
        valid &= gdal_valid
    if not path.endswith(".npy") and alpha_valid is not None:
        valid &= alpha_valid
    if nd is not None:
        # No-data only where EVERY band holds the value, as GDAL's dataset mask has it. Until 2026-10-06 one band
        # sufficed, so a probability vector with one class at exactly 0 (a profile copied from Sentinel-2 carries
        # nodata=0) was dropped: the most confident windows left the population, and estimate gave 15.7% where the
        # map's rate is 7.0%, with no message.
        valid &= (a != nd).any(axis=0) if a.ndim == 3 else (a != nd)
    if a.ndim == 3 and a.shape[0] == 1:
        # A (1, H, W) .npy, which np.save(path, src.read()) writes, was argmaxed by compare as one class everywhere:
        # "0 of 4096 windows differ" where the same maps as GeoTIFFs differ on 507, exit 0 (2026-10-06).
        a = a[0]
    return a, valid, geo


def _mask_bands(src):
    """(the alpha band that is the raster's mask or None, whether GDAL keeps another mask: internal or a .msk file).
    No-data GDAL keeps in a mask rather than as a value was read as data until 1.17.2, and an all-zero masked half made
    the whole 5% review set and half a random sample no-data windows (the bug hunt of 25 October 2026). An alpha band is
    the mask beside one data band (a class map with its alpha, as gdalwarp -dstalpha writes it), whatever GDAL's own
    flags say (it applies alpha only to 2- and 4-band Byte or UInt16 rasters). With more bands it is read as data, as
    before 1.17.2: GDAL marks the 4th band of a 4-band 8-bit GeoTIFF alpha by default, and a 4-class map lost its 4th
    class, and the pixels where that class scored 0, to the mask (the reviews of 1.17.2)."""
    from rasterio.enums import ColorInterp, MaskFlags
    alpha = [i + 1 for i, c in enumerate(src.colorinterp) if c == ColorInterp.alpha]
    alpha_mask = alpha[0] if src.count == 2 and len(alpha) == 1 else None
    other = any(MaskFlags.all_valid not in f and MaskFlags.nodata not in f and MaskFlags.alpha not in f
                for f in src.mask_flag_enums)
    return alpha_mask, other


def _read_masked(src):
    """(the data bands, GDAL's mask as valid pixels or None, the alpha band as valid pixels or None); _mask_bands says
    which mask a raster carries."""
    alpha_mask, other = _mask_bands(src)
    keep = [i + 1 for i in range(src.count) if i + 1 != alpha_mask]
    a = src.read(keep) if alpha_mask else src.read()
    return a, (src.dataset_mask() > 0 if other else None), (src.read(alpha_mask) > 0 if alpha_mask else None)


def _gdal_masked(path):
    """Whether a GeoTIFF keeps no-data in a mask read_raster reads (internal, a .msk file or the alpha band of a
    two-band raster), as it has since 1.17.2; False for a .npy or a file that cannot be read."""
    if not path or str(path).endswith(".npy"):
        return False
    try:
        import rasterio
        with rasterio.open(path) as src:
            alpha_mask, other = _mask_bands(src)
            return alpha_mask is not None or other
    except Exception:
        return False


def _mask_hint(*paths):
    """The note a refusal adds when the sample may have been drawn before 1.17.2 read the raster's mask."""
    return (" The raster keeps no-data in a GDAL mask (or the alpha band of a two-band raster), which is no-data since "
            "1.17.2 and was read as data before; a sample drawn before 1.17.2 on it no longer matches, so draw the "
            "sample again." if any(_gdal_masked(p) for p in paths) else "")


def affine_apply(a, b):
    """The affine transform `a` applied to `b`, another transform or a point (x, y), arrays allowed: with @, which
    affine 3 asks for (its * now warns), and with * on an older affine that has no @."""
    try:
        return a @ b
    except TypeError:
        return a * b


def write_raster(path, array, geo, patch, nodata):
    """A window-grid array as a GeoTIFF on the input's grid scaled by `patch`, or .npy without geo."""
    if geo is None:
        np.save(os.path.splitext(path)[0] + ".npy", array)
        return os.path.splitext(path)[0] + ".npy"
    import rasterio
    from rasterio.transform import Affine
    arr = np.asarray(array)
    dtype = "float32" if arr.dtype.kind == "f" else ("uint8" if arr.dtype == bool else "int32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs=geo["crs"], transform=affine_apply(geo["transform"], Affine.scale(patch)), nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def window_coords(rows, cols, geo, patch):
    """Pixel corner and map centre of each window."""
    pr, pc = rows * patch, cols * patch
    if geo is None:
        return pr, pc, None, None
    x, y = affine_apply(geo["transform"], (pc + patch / 2, pr + patch / 2))
    return pr, pc, np.asarray(x), np.asarray(y)


def pool_valid(valid, patch):
    h, w = valid.shape[0] // patch * patch, valid.shape[1] // patch * patch
    return valid[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3)) >= 0.5


def _band_count(path):
    """The number of bands of a raster or (C, H, W) array, 1 for a 2-D one, without reading its values."""
    if str(path).endswith(".npy"):
        shape = np.load(path, mmap_mode="r").shape
        return shape[0] if len(shape) == 3 else 1
    import rasterio
    with rasterio.open(path) as src:                    # the alpha band of a two-band raster is its mask
        return src.count - (_mask_bands(src)[0] is not None)


def _same_grid(geo_a, geo_b, what):
    """Refuse two rasters on different grids. Only shapes used to be compared, so a map shifted by 12.8 km, or in
    another CRS, was compared window by window as though co-registered."""
    if geo_a is None or geo_b is None:
        return
    ta, tb = np.asarray(tuple(geo_a["transform"])[:6], float), np.asarray(tuple(geo_b["transform"])[:6], float)
    # an absolute tolerance of a thousandth of a pixel: np.allclose's relative default allowed 50 m at a UTM northing
    # of 5e6 and 120 m at a Web Mercator easting of 1.2e7, so a map shifted by a whole window passed (review, 2026-09-23)
    atol = 1e-3 * max(abs(ta[0]), abs(ta[4]), 1e-12)
    if str(geo_a["crs"]) != str(geo_b["crs"]) or not np.allclose(ta, tb, rtol=0.0, atol=atol):
        raise SystemExit(f"{what} is not on the first map's grid (CRS {geo_b['crs']} vs {geo_a['crs']}, transform "
                         f"{tuple(geo_b['transform'])[:6]} vs {tuple(geo_a['transform'])[:6]}); resample it onto the same grid first")

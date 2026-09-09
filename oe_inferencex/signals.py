"""Label-free signals and the no-model pixel controls. Pure numpy.

The signals the ledger supports (docs/TECHNIQUES.md, docs/method/recipe.md):
the model's own confidence as the error ranker, prediction-boundary
proximity as a triage cue, and the aligned tile-phase instability that
beats confidence against WorldCover but not against hand labels. The pixel
controls are the no-model references every signal must beat (recipe item
4). Every function here reproduces the implementation used by the
experiments it cites; `tests/test_signals.py` checks that equality against
the experiment modules when the encoder stack is installed.

Shapes: a "map" is (H, W) or a batch (N, H, W); an image is the 12 Sentinel-2
bands (12, H, W) in the encoder's band order (S2_BANDS), raw digital numbers.
"""
import numpy as np

# olmoearth_pretrain Modality.SENTINEL2_L2A.band_order, copied so this module stays torch-free
S2_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")


# ----------------------------------------------------------------------------- confidence and boundary
def confidence(logits, multiclass=None):
    """The baseline error ranker (recipe item 1): higher = more suspect.

    Binary logit maps, (H, W) or a batch (N, H, W) with multiclass=False,
    give the negative absolute logit; one multiclass map with the class axis
    first, (C, H, W), gives the negative top-1 minus top-2 margin. Neither
    ties where a sigmoid or softmax saturates in float32. `multiclass`
    defaults to True for 3-d input. The input dtype is kept (float32 margins
    stay float32, as the experiments computed them); integers become float64."""
    x = np.asarray(logits)
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    if multiclass is None:
        multiclass = x.ndim == 3
    if multiclass:
        if x.ndim != 3 or x.shape[0] < 2:
            raise ValueError(f"multiclass logits must be one (C, H, W) map with C >= 2, got shape {x.shape}")
        srt = np.sort(x, axis=0)
        return -(srt[-1] - srt[-2])
    if x.ndim not in (2, 3):
        raise ValueError(f"binary logits must be (H, W) or (N, H, W), got shape {x.shape}")
    return -np.abs(x)


def boundary_indicator(hard, probabilities=False):
    """Fraction of a unit's 8 neighbours whose hard label differs (edge padding); (H, W) or (N, H, W).

    exp14's pred-boundary, exp18.boundary and the assessor's boundary fraction. `hard` holds class
    labels (any dtype, cast to int); with probabilities=True it holds the probability of the
    positive class and is thresholded at 0.5 first."""
    a = np.asarray(hard)
    if probabilities:
        a = (a > 0.5)
    a = a.astype(int)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[None]
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    G0, G1 = a.shape[1:]
    nb = np.zeros(a.shape, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[:, 1 + di:1 + di + G0, 1 + dj:1 + dj + G1] != a)
    nb /= 8.0
    return nb[0] if squeeze else nb


# ----------------------------------------------------------------------------- tile-phase instability
def aligned_tile_phase(shift_maps, patch=4, dtype=np.float64):
    """Perturbation stability (E_system): std across sub-patch shifts, aligned on a pixel canvas.

    shift_maps: sequence over shifts s = 0, 1, ..., S-1 of patch-grid maps
    (G, G) or (N, G, G) predicted from the window offset by s pixels. Each
    map is upsampled to pixels, placed at its true offset on a common canvas,
    the per-pixel std across shifts is taken, and the result is pooled back
    to the shift-0 patch grid (exp13; alignment is the correction of exp05,
    exp09 and exp11). Returns (G, G) or (N, G, G). `dtype` is the canvas
    arithmetic: float64 reproduces exp13 (scenes), float32 reproduces exp18
    (flood tiles) bit for bit; the two differ in the last float32 digits."""
    maps = np.asarray(shift_maps, dtype=dtype)
    squeeze = maps.ndim == 3
    if squeeze:
        maps = maps[:, None]
    S, N, G0, G1 = maps.shape
    H, W = G0 * patch, G1 * patch
    canvas = np.full((S, N, H + S, W + S), np.nan, dtype=dtype)
    ones = np.ones((patch, patch), dtype=dtype)
    for s in range(S):
        canvas[s, :, s:s + H, s:s + W] = np.stack([np.kron(m, ones) for m in maps[s]])
    pix_std = np.nanstd(canvas, axis=0)[:, :H, :W]
    out = np.nanmean(pix_std.reshape(N, G0, patch, G1, patch), axis=(2, 4))
    return out[0] if squeeze else out


# ----------------------------------------------------------------------------- pixel controls
def _crop(img, size):
    x = np.asarray(img)
    if size is None:
        return x
    return x[..., :size, :size]


def ndwi(img, bands=S2_BANDS):
    """Normalised difference water index (green - NIR) / (green + NIR) per pixel, denominator clipped at 1."""
    x = np.asarray(img).astype(np.float64)
    g, nir = x[..., bands.index("B03"), :, :], x[..., bands.index("B08"), :, :]
    return (g - nir) / np.clip(g + nir, 1, None)


def ndwi_gradient(img, patch=4, size=None, bands=S2_BANDS):
    """The no-model control for water (exp06, exp13, exp18): mean NDWI gradient magnitude per patch.

    img (12, H, W) or (N, 12, H, W); `size` crops the top-left size x size pixels first (exp13 uses 128 of 132)."""
    nd = ndwi(_crop(img, size), bands)
    gy, gx = np.gradient(nd, axis=(-2, -1))
    mag = np.hypot(gx, gy)
    G0, G1 = mag.shape[-2] // patch, mag.shape[-1] // patch
    return mag[..., :G0 * patch, :G1 * patch].reshape(*mag.shape[:-2], G0, patch, G1, patch).mean(axis=(-3, -1))


def ndwi_level(img, patch=4, size=None, bands=S2_BANDS):
    """Observed-input control (exp28): -|patch-mean NDWI|; near zero = spectrally ambiguous water/land."""
    nd = ndwi(_crop(img, size), bands)
    G0, G1 = nd.shape[-2] // patch, nd.shape[-1] // patch
    return -np.abs(nd[..., :G0 * patch, :G1 * patch].reshape(*nd.shape[:-2], G0, patch, G1, patch).mean(axis=(-3, -1)))


def s2_patch_variance(img, patch=4, size=None):
    """Observed-input control (exp28): mean over bands of the within-patch pixel std (raw DN)."""
    x = _crop(img, size).astype(np.float64)
    G0, G1 = x.shape[-2] // patch, x.shape[-1] // patch
    return x[..., :G0 * patch, :G1 * patch].reshape(*x.shape[:-2], G0, patch, G1, patch).std(axis=(-3, -1)).mean(axis=-3)


# ----------------------------------------------------------------------------- combination
def midrank_pct(v):
    """Within-unit midrank percentile in [0, 1], ties averaged."""
    v = np.asarray(v, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / max(len(v) - 1, 1)


def combine_midrank(a, b):
    """The preregistered fixed combination U+ (exp28, exp30, exp31): mean of two within-unit midrank percentiles.

    Returns the array in the shape of `a`; scores ranks only, so it cannot be tuned to a testbed."""
    a = np.asarray(a)
    return ((midrank_pct(a) + midrank_pct(b)) / 2).reshape(a.shape)


# ----------------------------------------------------------------------------- window design (exp42)
def shift_averaged_probability(shift_maps, patch=4):
    """The decision design that beat the grid window on both hand-label testbeds (exp42): run the encoder on the
    same tile cropped at offsets 0, 1, ..., S-1 px (as exp18's cache), paint each window map back to the pixels it
    covers, and average. Returns (pixel_map, common), where pixel_map (H + S - 1, W + S - 1) holds the mean
    probability over every tiling that covers a pixel (NaN where none does) and `common` marks the pixels covered
    by all S tilings, the region exp42 scored. shift_maps: (S, G, G) window probabilities, shift s first.

    Pixel accuracy rose by 1.0 points on Bolivia and 0.9 on the multi-region test split (p < 1e-40 per tile) at the
    cost of S forward passes; the window-pooled version of the map is the input for assess_prediction."""
    maps = np.asarray(shift_maps, dtype=np.float64)
    S, G0, G1 = maps.shape
    H, W = G0 * patch + S - 1, G1 * patch + S - 1
    canvas = np.full((S, H, W), np.nan)
    for s in range(S):
        canvas[s, s:s + G0 * patch, s:s + G1 * patch] = np.repeat(np.repeat(maps[s], patch, 0), patch, 1)
    pixel = np.nanmean(canvas, axis=0)
    common = np.zeros((H, W), dtype=bool)
    common[S - 1:G0 * patch, S - 1:G1 * patch] = True
    return pixel, common


def pool_to_windows(pixel_map, patch=4, offset=0):
    """Mean of a pixel map over the patch x patch windows of the tiling at crop `offset` (NaN-aware)."""
    a = np.asarray(pixel_map, dtype=np.float64)
    G0, G1 = (a.shape[0] - offset) // patch, (a.shape[1] - offset) // patch
    blk = a[offset:offset + G0 * patch, offset:offset + G1 * patch].reshape(G0, patch, G1, patch)
    return np.nanmean(blk, axis=(1, 3))

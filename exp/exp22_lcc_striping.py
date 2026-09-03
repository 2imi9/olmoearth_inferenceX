"""exp22: periodic artifacts (striping, seams) in the served LCC product.

Question. Does the served land cover change product (allenai/olmoearth_lcc,
encoder v1.2-Base, which introduced rotary position encoding to remove the
striping seen with v1) carry periodic structure at the scales the pipeline
imposes: the encoder patch grid, the inference window grid, or the warp to
the serving grid?

Design. The rasters are served on the Web Mercator zoom-14 grid (9.5546
units/px, tile-aligned), warped from 10 m UTM inference tiles named in the
file name. Any grid-imposed artifact therefore appears at a predictable
period in the served pixels: a UTM length of L px maps to
L * 10 m / (9.5546 * cos(lat)) served px, and nearest-neighbour warping
itself duplicates one source column every 1 / (ratio - 1) px. Predicted
periods are computed from each window's own transform and latitude, not
assumed.

Measurement. For each window (4096 x 4096 served px) and each of three maps
(class map band 4 as a boundary indicator between adjacent pixels; change
probability band 1 as an absolute gradient; ESA WorldCover 2021 warped to
the same grid as a control that shares no model or inference grid), column
and row profiles of the mean indicator are computed over the full window
after correcting the shear of the UTM grid lines in the Mercator raster
(from the tile's UTM CRS; about 0.6 degrees here), high-pass filtered, and
their periodograms taken. Each
periodogram is whitened by a running median over 41 bins so that a
grid-imposed artifact, which is sharp in frequency, stands out from the
smooth land-surface spectrum. Under the null of no periodic component the
whitened ordinates are approximately Exp(1) (Gamma(S, 1/S) for S averaged
profiles);
the p-value of the largest ordinate is the Bonferroni-corrected survival
probability over the searched bins (periods 3 to 1024 px). Peaks are matched
to predicted periods within 2%.

Outcomes. A peak at a predicted period with p < 0.01 in the product profiles
and not in WorldCover is evidence of that artifact; its absence at the
inference-window periods is evidence against seams at 4096-px scale. Nothing
here uses labels or the model; the product is read as served.

Two kinds of test. Confirmatory: at each predicted period, the whitened
ordinate of the fundamental and a comb score over its first three harmonics,
with Gamma p-values and no multiplicity correction because the periods are
fixed before looking. Exploratory: the largest whitened ordinates anywhere in
the searched band, Bonferroni-corrected over all bins. The test was validated
on synthetic lattice-free maps (false-positive rate, power on injected seams,
rotation of 0.6 degrees, and a pure nearest-neighbour warp) before use
(scratchpad validate_exp22.py; results in the lab log).

Power. Seams are injected into each real class map along sheared UTM grid
lines at the 128-px and 256-px window periods, affecting 2 to 40% of rows,
and the smallest fraction detected at p < 0.01 is reported as that window's
detection limit, so a null result is stated with the effect size it rules out.

Outputs: exp/out/exp22_lcc_striping.csv (scan), exp22_confirmatory.csv, exp22_power.csv, exp22_summary.json,
exp22_lcc_striping.png. Cache exp/out/exp22_windows.npz (ignored by git).
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from rasterio.transform import Affine
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oe_inferencex import lcc  # noqa: E402
from oe_inferencex.data import fetch_worldcover_window  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE = 4096
STRIP = 1024
WINDOWS = {  # lon, lat of window centres; each must fit inside its served tile
    "chobe_east": (24.95, -17.85), "sesheke": (24.30, -17.55), "savuti_north": (24.20, -18.55),
    "kwando": (23.30, -17.20), "barotse": (23.13, -15.25),
}
UTM_LENGTHS = {"encoder patch 4 px": 4, "2 patches 8 px": 8, "window 32 px": 32, "window 64 px": 64,
               "window 128 px": 128, "window 256 px": 256, "window 512 px": 512}
PMIN, PMAX = 3.0, 1024.0
MEDIAN_BINS = 41
TOL = 0.02


def predicted_periods(transform, epsg, lat):
    """Served-pixel periods implied by the UTM 10 m grid and the warp."""
    res = transform[0]
    ground = res * math.cos(math.radians(lat)) if epsg == 3857 else res  # metres per served px
    ratio = 10.0 / ground                                                # served px per UTM px
    preds = {k: v * ratio for k, v in UTM_LENGTHS.items()}
    if abs(ratio - 1.0) > 1e-6:
        # nearest warping repeats one source column every ratio / (ratio - 1) served px
        preds["warp duplication beat"] = ratio / abs(ratio - 1.0)
    return preds, ratio, ground


def shear_slopes(meta):
    """How UTM grid lines run across the served (north-up Mercator) raster:
    (columns shift per row for UTM columns, rows shift per column for UTM rows),
    in served px, from the tile's UTM CRS and the window's transform."""
    import rasterio.warp
    utm = f"EPSG:{meta['tile'].split('_')[0].split(':')[1]}"
    served = f"EPSG:{meta['epsg']}"
    xs, ys = rasterio.warp.transform("EPSG:4326", utm, [meta["lon"]], [meta["lat"]])
    x0, y0 = xs[0], ys[0]
    px, py = rasterio.warp.transform(utm, served, [x0, x0, x0 + 1000.0], [y0, y0 - 1000.0, y0])
    res = meta["transform"][0]
    # UTM column (going south): served dx per served row
    col_slope = (px[1] - px[0]) / ((py[0] - py[1]))
    # UTM row (going east): served dy per served column (positive = downward)
    row_slope = -(py[2] - py[0]) / ((px[2] - px[0]))
    return float(col_slope), float(row_slope)


def load_windows():
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "exp22_windows.npz")
    W = dict(np.load(cache, allow_pickle=True)) if os.path.exists(cache) else {}
    for name, (lon, lat) in WINDOWS.items():
        if f"{name}_class" in W:
            continue
        t0 = time.time()
        tile = lcc.tile_for(lon, lat)[0]
        arr, transform, epsg, _ = lcc.read_window(lon, lat, SIZE, tile=tile)
        wc = fetch_worldcover_window(lon, lat, f"EPSG:{epsg}", Affine(*transform), SIZE)
        W[f"{name}_class"], W[f"{name}_change"], W[f"{name}_wc"] = arr[3], arr[0], wc
        W[f"{name}_meta"] = np.array(json.dumps({"tile": tile, "epsg": epsg, "transform": list(transform), "lon": lon, "lat": lat}))
        np.savez_compressed(cache, **W)
        print(f"fetched {name} from {tile} in {time.time() - t0:.0f}s")
    return W


def _sheared_mean(ind, slope):
    """Mean over rows of ind after shifting row r by -round((r - H/2) * slope)
    columns, so that lines running at that slope become vertical. Edge columns
    not covered by every row are dropped."""
    H, W = ind.shape
    shifts = np.round((np.arange(H) - H / 2.0) * slope).astype(int)
    lo, hi = shifts.min(), shifts.max()
    width = W - (hi - lo)
    acc = np.zeros(width, dtype=np.float64)
    for r in range(H):
        start = shifts[r] - lo
        acc += ind[r, start:start + width]
    return acc / H


def inject_seams(cls, period, col_slope, frac, seed=1):
    """Copy of a class map with synthetic seams: along vertical UTM grid lines
    of the given served period (sheared like the real grid), a fraction of rows
    has every class right of the line replaced by another class."""
    rng = np.random.default_rng(seed)
    out = cls.copy()
    H, W = cls.shape
    n_classes = int(cls.max()) + 1
    for k in range(1, int(W / period) + 1):
        base = k * period
        hit = np.flatnonzero(rng.random(H) < frac)
        for r in hit:
            c = int(round(base + (r - H / 2.0) * col_slope))
            if 0 < c < W:
                seg = out[r, c:]
                out[r, c:] = np.where(seg > 0, seg % (n_classes - 1) + 1, 0)  # another valid class; nodata stays
    return out


def profiles(indicator, col_slope=0.0, row_slope=0.0):
    """Full-height shear-corrected column profile and full-width row profile
    of a (H, W) indicator -> (list with one column profile, list with one row profile)."""
    ind = np.asarray(indicator, dtype=np.float64)
    cols = [_sheared_mean(ind, col_slope)]
    rows = [_sheared_mean(ind.T, row_slope)]
    return cols, rows


def welch_periodogram(strips):
    """Average periodogram of high-passed strips; returns (freqs, power, n_strips)."""
    P = None
    for s in strips:
        s = s - np.convolve(s, np.ones(129) / 129, mode="same")  # high-pass: remove slow land-surface trend
        s = s - s.mean()
        f = np.fft.rfftfreq(len(s), d=1.0)
        p = np.abs(np.fft.rfft(s)) ** 2 / len(s)
        P = p if P is None else P + p
    return f, P / len(strips), len(strips)


def whiten(f, P):
    """Running-median whitening over the searched band. Returns (idx, white)."""
    sel = (f > 0) & (1.0 / np.maximum(f, 1e-12) >= PMIN) & (1.0 / np.maximum(f, 1e-12) <= PMAX)
    idx = np.flatnonzero(sel)
    med = np.array([np.median(P[max(0, i - MEDIAN_BINS // 2): i + MEDIAN_BINS // 2 + 1]) for i in idx])
    white = P[idx] / np.maximum(med, 1e-30) * math.log(2)  # median of Exp(1) is ln 2, so E[white] ~ 1 under H0
    return idx, white


def peaks(f, P, n_strips, preds):
    """Exploratory scan: the largest whitened ordinates, Bonferroni over all searched bins."""
    idx, white = whiten(f, P)
    n_bins = len(idx)
    order = np.argsort(white)[::-1][:8]
    out = []
    for o in order:
        period = 1.0 / f[idx[o]]
        p_raw = stats.gamma.sf(white[o], a=n_strips, scale=1.0 / n_strips)  # S-strip average: Gamma(S, 1/S) under H0
        p_bonf = min(1.0, p_raw * n_bins)
        match = [k for k, v in preds.items() if abs(period - v) / v <= TOL]
        harm = [f"{k} (h={h})" for k, v in preds.items() for h in range(2, 49) if abs(period * h - v) / v <= TOL]
        out.append({"period_px": float(period), "whitened_power": float(white[o]), "p_bonferroni": float(p_bonf),
                    "matches": ";".join(match), "harmonic_of": ";".join(harm)})
    return out, n_bins


def confirmatory(f, P, n_strips, preds, n_harm=3):
    """Pre-specified tests, one per predicted period: the whitened ordinate at
    the fundamental (the larger of the two bins bracketing the period) and a
    comb score averaging the first n_harm harmonics. No multiplicity
    correction: the periods are fixed before looking."""
    idx, white = whiten(f, P)
    fidx = f[idx]
    out = {}
    for k, period in preds.items():
        vals = []
        for h in range(1, n_harm + 1):
            f0 = h / period
            if f0 > fidx.max() or f0 < fidx.min():
                break
            j = np.searchsorted(fidx, f0)
            cand = [c for c in (j - 1, j) if 0 <= c < len(fidx)]
            vals.append(max(white[c] for c in cand))
        if not vals:
            continue
        fund = vals[0]
        comb = float(np.mean(vals))
        out[k] = {"predicted_period_px": float(period), "fundamental_whitened_power": float(fund),
                  "p_fundamental": float(stats.gamma.sf(fund, a=n_strips, scale=1.0 / n_strips)),
                  "n_harmonics": len(vals), "comb_whitened_power": comb,
                  "p_comb": float(stats.gamma.sf(comb * len(vals), a=n_strips * len(vals), scale=1.0 / n_strips))}
    return out


def main():
    W = load_windows()
    rows, conf_rows, power_rows, summary = [], [], [], {"windows": {}}
    fig_data = {}
    for name in WINDOWS:
        meta = json.loads(str(W[f"{name}_meta"]))
        preds, ratio, ground = predicted_periods(meta["transform"], meta["epsg"], meta["lat"])
        cls, chg, wc = W[f"{name}_class"], W[f"{name}_change"].astype(float) / 255.0, W[f"{name}_wc"]
        nod = cls == 0
        maps = {
            "class_boundary": {"col": (cls[:, 1:] != cls[:, :-1]) & ~nod[:, 1:] & ~nod[:, :-1],
                               "row": (cls[1:, :] != cls[:-1, :]) & ~nod[1:, :] & ~nod[:-1, :]},
            "change_gradient": {"col": np.abs(chg[:, 1:] - chg[:, :-1]), "row": np.abs(chg[1:, :] - chg[:-1, :])},
            "worldcover_boundary": {"col": (wc[:, 1:] != wc[:, :-1]) & (wc[:, 1:] > 0) & (wc[:, :-1] > 0),
                                    "row": (wc[1:, :] != wc[:-1, :]) & (wc[1:, :] > 0) & (wc[:-1, :] > 0)},
        }
        col_slope, row_slope = shear_slopes(meta)
        summary["windows"][name] = {"tile": meta["tile"], "epsg": meta["epsg"], "served_px_ground_m": ground,
                                    "served_px_per_utm_px": ratio, "predicted_periods_px": preds,
                                    "utm_column_slope_px_per_row": col_slope, "utm_row_slope_px_per_col": row_slope,
                                    "nodata_fraction": float(nod.mean()), "worldcover_nodata_fraction": float((wc == 0).mean())}
        for mname, m in maps.items():
            for axis in ("col", "row"):
                ind = m[axis].astype(float)
                strips = profiles(ind, col_slope, row_slope)[0] if axis == "col" else profiles(ind, col_slope, row_slope)[1]
                f, P, S = welch_periodogram(strips)
                pk, n_bins = peaks(f, P, S, preds)
                for rank, p in enumerate(pk):
                    rows.append({"window": name, "map": mname, "axis": axis, "rank": rank + 1, **p, "n_bins": n_bins, "n_strips": S})
                conf = confirmatory(f, P, S, preds)
                for k, c in conf.items():
                    conf_rows.append({"window": name, "map": mname, "axis": axis, "hypothesis": k, **c, "n_strips": S})
                if name == "chobe_east":
                    fig_data[(mname, axis)] = (f, P, pk)
                top = pk[0]
                print(f"{name:13s} {mname:20s} {axis} | top period {top['period_px']:7.2f} px power {top['whitened_power']:6.1f} p {top['p_bonferroni']:.2e} match [{top['matches']}] harmonic [{top['harmonic_of']}]")
        # in-situ power analysis: seams injected into the real class map along UTM grid lines at the
        # 128-px and 256-px window periods; the smallest affected-row fraction detected is the detection limit
        for key in ("window 128 px", "window 256 px"):
            period = preds[key]
            limit = None
            for frac in (0.02, 0.05, 0.10, 0.20, 0.40):
                inj = inject_seams(cls, period, col_slope, frac, seed=1)
                ind = ((inj[:, 1:] != inj[:, :-1]) & ~nod[:, 1:] & ~nod[:, :-1]).astype(float)
                f, P, S = welch_periodogram(profiles(ind, col_slope, row_slope)[0])
                c = confirmatory(f, P, S, {key: period})[key]
                power_rows.append({"window": name, "hypothesis": key, "injected_row_fraction": frac, "p_comb": c["p_comb"], "comb_whitened_power": c["comb_whitened_power"]})
                if limit is None and c["p_comb"] < 0.01:
                    limit = frac
            summary["windows"][name][f"detection_limit_row_fraction[{key}]"] = limit
            print(f"{name:13s} detection limit at {key} (class map, columns): {limit}")

    with open(os.path.join(OUT, "exp22_lcc_striping.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "exp22_confirmatory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conf_rows[0].keys())); w.writeheader(); w.writerows(conf_rows)
    with open(os.path.join(OUT, "exp22_power.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(power_rows[0].keys())); w.writeheader(); w.writerows(power_rows)
    # confirmatory tally: per map and hypothesis, profiles (window x axis) with p_comb below 0.01 and 0.001
    ctally = {}
    for r in conf_rows:
        d = ctally.setdefault(r["map"], {}).setdefault(r["hypothesis"], {"n": 0, "p_comb<0.01": 0, "p_comb<0.001": 0, "median_comb_power": []})
        d["n"] += 1; d["p_comb<0.01"] += int(r["p_comb"] < 0.01); d["p_comb<0.001"] += int(r["p_comb"] < 0.001); d["median_comb_power"].append(r["comb_whitened_power"])
    for m in ctally.values():
        for d in m.values():
            d["median_comb_power"] = float(np.median(d["median_comb_power"]))
    summary["confirmatory_tally"] = ctally
    # fundamental-only tally: the comb score is confounded for hypotheses whose harmonics coincide with
    # the patch lattice (every window period is a multiple of the patch period), so seams at window
    # scale are judged on the fundamental ordinate alone
    ftally = {}
    for r in conf_rows:
        d = ftally.setdefault(r["map"], {}).setdefault(r["hypothesis"], {"n": 0, "p_fund<0.01": 0, "p_fund<0.001": 0, "median_fund_power": []})
        d["n"] += 1; d["p_fund<0.01"] += int(r["p_fundamental"] < 0.01); d["p_fund<0.001"] += int(r["p_fundamental"] < 0.001); d["median_fund_power"].append(r["fundamental_whitened_power"])
    for m in ftally.values():
        for d in m.values():
            d["median_fund_power"] = float(np.median(d["median_fund_power"]))
    summary["fundamental_tally"] = ftally
    summary["comb_caveat"] = "comb scores are confounded for hypotheses commensurate with the patch lattice; window-scale seams are judged on the fundamental"
    # top exploratory peak per product profile, and whether it lies on the patch lattice (k x 4 UTM px, k in 1..8, or the 4-patch period / 3)
    tops = []
    for r in rows:
        if r["rank"] == 1 and r["map"] != "worldcover_boundary":
            preds_w = summary["windows"][r["window"]]["predicted_periods_px"]
            unit = preds_w["encoder patch 4 px"]
            lattice = [unit * k for k in (1, 2, 3, 4, 5, 6, 7, 8)] + [unit * 4 / 3, unit * 8 / 3]
            on = any(abs(r["period_px"] - v) / v <= TOL for v in lattice)
            tops.append({"window": r["window"], "map": r["map"], "axis": r["axis"], "period_px": r["period_px"], "period_in_utm_patches": r["period_px"] / unit,
                         "p_bonferroni": r["p_bonferroni"], "on_patch_lattice": bool(on)})
    summary["top_peaks_product"] = tops
    summary["top_peaks_on_lattice"] = f"{sum(t['on_patch_lattice'] for t in tops)}/{len(tops)}"
    summary["top_peaks_max_p_bonferroni"] = float(max(t["p_bonferroni"] for t in tops))
    wc_tops = [r for r in rows if r["rank"] == 1 and r["map"] == "worldcover_boundary"]
    summary["worldcover_top_peaks_min_p_bonferroni"] = float(min(r["p_bonferroni"] for r in wc_tops))

    # tally: for each map x predicted period, in how many window-axis profiles is a matching peak significant?
    tally = {}
    for r in rows:
        if r["p_bonferroni"] < 0.01 and r["matches"]:
            for k in r["matches"].split(";"):
                tally.setdefault(r["map"], {}).setdefault(k, 0)
                tally[r["map"]][k] += 1
    summary["significant_matched_peaks_count_by_map_and_period"] = tally
    summary["n_profiles_per_map"] = len(WINDOWS) * 2
    summary["settings"] = {"size_px": SIZE, "strip_px": STRIP, "period_range_px": [PMIN, PMAX], "median_bins": MEDIAN_BINS,
                           "tolerance": TOL, "utm_lengths_px": UTM_LENGTHS}
    with open(os.path.join(OUT, "exp22_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(tally, indent=1))
    make_figure(fig_data, summary["windows"]["chobe_east"]["predicted_periods_px"])


def make_figure(fig_data, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oe_inferencex import figstyle
    figstyle.setup()
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for i, mname in enumerate(("class_boundary", "change_gradient", "worldcover_boundary")):
        for j, axis in enumerate(("col", "row")):
            ax = axes[i, j]
            f, P, pk = fig_data[(mname, axis)]
            sel = f > 0
            period = 1.0 / f[sel]
            ax.loglog(period, P[sel], color="#1f77b4", linewidth=0.6)
            for k, v in preds.items():
                ax.axvline(v, color="#d62728" if "window" in k else "#2ca02c" if "patch" in k else "#7f7f7f", linestyle=":", linewidth=0.8)
            for p in pk[:3]:
                if p["p_bonferroni"] < 0.01:
                    ax.plot(p["period_px"], P[sel][np.argmin(np.abs(period - p["period_px"]))], "o", color="#ff7f0e", markersize=5)
            ax.set_xlim(PMAX, PMIN); ax.set_title(f"{mname} ({'columns' if axis == 'col' else 'rows'})", fontsize=10)
            if i == 2:
                ax.set_xlabel("period (served px; log)")
            if j == 0:
                ax.set_ylabel("Welch periodogram")
    axes[0, 0].plot([], [], color="#d62728", linestyle=":", label="predicted inference-window periods (32-512 UTM px)")
    axes[0, 0].plot([], [], color="#2ca02c", linestyle=":", label="predicted encoder-patch periods (4, 8 UTM px)")
    axes[0, 0].plot([], [], color="#7f7f7f", linestyle=":", label="predicted warp duplication beat")
    axes[0, 0].plot([], [], "o", color="#ff7f0e", label="peak with Bonferroni p < 0.01")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("exp22: periodograms of boundary and gradient profiles, chobe_east window (4096 px of the served LCC product, v1.2 encoder)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp22_lcc_striping.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()

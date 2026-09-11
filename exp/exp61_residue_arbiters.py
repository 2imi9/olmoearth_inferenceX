#!/usr/bin/env python
"""exp61: what the pre-event radar's departures from the permanent-water label are, by GEOID's own layers and a date-matched arbiter.

Why. exp60 found that 54.5% of the same-sensor difference across a flood event (the S1 water head on the pre-event pass
against the post-event pass) is the pre-event decision departing from GEOID's permanent-water label, and read it as the
radar seeing the water regime of the pre-event date, which the "permanent" class does not hold. That reading was not
tested. Two things can test it without new imagery: GEOID's own layers that exp55 never extracted (permwater, the source
of the label's class 1; floodmask, the raw CEMS delineation; validity), and a water label matched to the pre-event
MONTH: the JRC Global Surface Water monthly history (Pekel et al. 2016, v1.4, 30 m, 1984-2021, values 0 no observation,
1 not water, 2 water), read by windowed HTTP range requests from the JRC open-data server. Its bias runs against the
reading: 30 m optical under-calls narrow water, exactly where a radar head at 10 m may be right.

Design. exp60's chips and decisions, reproduced from the same tile table and selection and asserted equal to the committed
exp/out/exp60_masks.npz (events and both label grids), so every window below is one of exp60's. Per window (4 px, the W1
grid): permwater, floodmask and validity pooled as the labels are (majority over valid pixels); the JRC monthly raster for
the month of the tile's pre-event S1 pass reprojected (nearest) onto the chip grid, pooled to the share of observed pixels
and the share of water among them. Windows count as GSW-observed when at least MIN_OBSERVED of their pixels were
observed that month. The residue: the time-only differing windows (A_s1pre against B_s1post) where A_s1pre departs from
the permanent label, split into "radar water, label not permanent" and "radar dry, label permanent". Readings on it:
share GSW-water among the observed, share floodmask-flooded, permwater agreement with the label (sanity), per event with
compare.over_groups. Also: each head's agreement with GSW on all observed windows, and on the sensor-only differing
windows (A_s2pre against A_s1pre) which side GSW agrees with.

Preregistered (one-sided):
  P1  on the residue windows GSW observed that month, GSW says WATER on at least half, pooled over events, and on more
      events than not (events with at least MIN_EVENT_WINDOWS such windows; exact sign test against one half, p < 0.05).
  Falsification: below one half pooled, or the event test misses: the residue is then radar error against a date-matched
  optical label, not the water regime of the date. Stated predictions, not tested: the "radar water" sub-case carries
  most of the residue and most of the GSW water; the raw floodmask marks under a fifth of the residue; on the sensor-only
  differing windows GSW sides with the radar head more often than with the optical composite head.
  Gate, descriptive: the GSW no-observation share on the residue; above one half, the arbiter is too thin to carry P1.

Inputs. exp/out/exp60_masks.npz, $HF_HOME/geoid_flood (exp60's tree; three small shards per split added), the JRC server
(network). CPU only. Outputs. exp/out/exp61_summary.json, exp/out/exp61_residue.csv (one row per event),
exp/out/exp61_layers.npz (the pooled layers on exp60's windows). Smoke: the sample tree, online.
"""
import csv
import json
import os
import re
import sys
import tarfile
import time

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp55_geoid_flood as e55  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import exp60_two_periods as e60  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import over_groups  # noqa: E402

EXTRA = ("permwater", "floodmask", "validity")
GSW = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/MonthlyHistory/VER5-0/tiles/{y}/{y}_{m:02d}/{y}_{m:02d}-{r:010d}-{c:010d}.tif"
GSW_PX, GSW_TILE, GSW_LAT_TOP, GSW_LON_LEFT, GSW_LAST = 0.00025, 40000, 80.0, -180.0, (2021, 12)
CHIP, PATCH, G, SL = e55.CHIP, e55.PATCH, e55.G, e57.SL
MIN_EVENT_WINDOWS, MIN_OBSERVED, CAP = 20, 0.5, e60.CAP
fmt = e57.fmt
_GSW_CACHE = {}


# ----------------------------------------------------------------------------- GEOID's extra layers
def fetch_extra(files, split, local_dir, root):
    """All shards of the three small layers of a split (about 10 MB each), every member extracted; own markers."""
    from huggingface_hub import hf_hub_download
    dest, done = os.path.join(root, split), os.path.join(root, split, ".done")
    os.makedirs(done, exist_ok=True)
    got = []
    for layer in EXTRA:
        for sh in sorted(f for f in files if f.startswith(f"{e55.TREE}/shards/{split}/{layer}/") and f.endswith(".tar")):
            marker = os.path.join(done, os.path.basename(sh) + ".extra.done")
            if os.path.exists(marker):
                got.append({"shard": sh, "cached": True})
                continue
            path = hf_hub_download(e55.REPO, sh, repo_type="dataset", local_dir=local_dir)
            with tarfile.open(path) as tar:
                members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tif")]
                try:
                    tar.extractall(path=dest, members=members, filter="data")
                except TypeError:
                    tar.extractall(path=dest, members=members)
            os.remove(path)
            open(marker, "w").write(str(len(members)))
            got.append({"shard": sh, "members": len(members)})
            print(f"  {sh}: {len(members)} members", flush=True)
    return got


def extra_paths(root_dirs):
    """{tile id: {layer: path}} for the three layers, from every .tif under the given directories."""
    out = {}
    for root in root_dirs:
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                for layer in EXTRA:
                    if fn.endswith(f"_{layer}.tif"):
                        out.setdefault(fn[:-len(f"_{layer}.tif")], {})[layer] = os.path.join(dirpath, fn)
    return out


def ensure_extra(args, tiles, data_dir, summary):
    if not args.smoke:
        root = os.path.join(data_dir, "tree")
        files = e55.hub_files()
        summary["config"]["extra_shards"] = {s: fetch_extra(files, s, data_dir, root) for s in ("test",)}
        return extra_paths([os.path.join(root, "test")])
    snaps = sorted({os.path.dirname(os.path.dirname(os.path.dirname(t["label"]))) for t in tiles["test"].values()})
    paths = extra_paths(snaps)
    missing = [(tid, layer) for tid, t in tiles["test"].items() for layer in EXTRA if layer not in paths.get(tid, {})]
    if missing and not os.environ.get("HF_HUB_OFFLINE"):
        from huggingface_hub import hf_hub_download
        for tid, layer in missing:
            aoi = tiles["test"][tid]["aoi"]
            try:
                p = hf_hub_download(e55.REPO, f"sample/geoid-flood/{aoi}/{layer}/{tid}_{layer}.tif", repo_type="dataset")
                paths.setdefault(tid, {})[layer] = p
            except Exception as ex:  # noqa: BLE001
                print(f"  smoke: {tid} {layer} not downloadable: {ex!r}", flush=True)
    return paths


# ----------------------------------------------------------------------------- pooling and the JRC arbiter
def pool(chip, positive, nodata=255):
    """A 64-px chip -> per 4-px window on exp60's 14 x 14 grid: share of `positive` among non-nodata pixels, share valid."""
    a = np.asarray(chip)[:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH)
    valid = (a != nodata)
    n_valid = valid.sum(axis=(1, 3))
    pos = (np.isin(a, positive) & valid).sum(axis=(1, 3))
    share = np.where(n_valid > 0, pos / np.maximum(n_valid, 1), np.nan)
    return share[SL, SL], (n_valid / (PATCH * PATCH))[SL, SL]


def pre_month(path):
    m = re.search(r"_s1grd_pre_(\d{4})(\d{2})\d{2}T", os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def gsw_open(url):
    if url not in _GSW_CACHE:
        try:
            _GSW_CACHE[url] = rasterio.open("/vsicurl/" + url)
        except Exception as ex:  # noqa: BLE001
            _GSW_CACHE[url] = ex
    return _GSW_CACHE[url]


def gsw_chip(tile_path, r, c, year, month):
    """The JRC monthly raster of (year, month) on this chip's grid, uint8 {0, 1, 2}; None when past the product's end."""
    if (year, month) > GSW_LAST:
        return None
    with rasterio.open(tile_path) as src:
        win = Window(c, r, CHIP, CHIP)
        dst_transform = src.window_transform(win)
        dst_crs = src.crs
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.window_bounds(win), densify_pts=5)
    out = np.zeros((CHIP, CHIP), np.uint8)
    corners = {(int((GSW_LAT_TOP - lat) / GSW_PX // GSW_TILE) * GSW_TILE, int((lon - GSW_LON_LEFT) / GSW_PX // GSW_TILE) * GSW_TILE)
               for lat in (lat0, lat1) for lon in (lon0, lon1)}
    for rr, cc in sorted(corners):
        ds = gsw_open(GSW.format(y=year, m=month, r=rr, c=cc))
        if isinstance(ds, Exception):
            raise ds
        b = ds.bounds
        if b.right < lon0 or b.left > lon1 or b.top < lat0 or b.bottom > lat1:
            continue
        w0 = ds.window(max(lon0, b.left), max(lat0, b.bottom), min(lon1, b.right), min(lat1, b.top)).round_offsets().round_lengths()
        c0, r0 = max(int(w0.col_off) - 1, 0), max(int(w0.row_off) - 1, 0)                  # one source pixel of padding: no uncovered strip
        sw = Window(c0, r0, min(int(w0.width) + 2, ds.width - c0), min(int(w0.height) + 2, ds.height - r0))
        if sw.width < 1 or sw.height < 1:
            continue
        src_arr = ds.read(1, window=sw)
        tmp = np.zeros((CHIP, CHIP), np.uint8)
        # no nodata arguments: with dst_nodata=0 and no src_nodata, rasterio rewrites source zeros (no observation) as 1
        reproject(src_arr, tmp, src_transform=ds.window_transform(sw), src_crs=ds.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.nearest)
        out = np.maximum(out, tmp)
    return out


# ----------------------------------------------------------------------------- readings
def share(x, m):
    return float(x[m].mean()) if m.any() else float("nan")


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp61 the pre-event radar's departures from the permanent-water label: GEOID's own layers and the JRC monthly water of the pre-event month",
               "smoke": args.smoke, "config": {"extra_layers": EXTRA, "gsw": GSW, "gsw_last_month": GSW_LAST, "min_observed": MIN_OBSERVED, "min_event_windows": MIN_EVENT_WINDOWS,
                                                "prereg": "P1 on the residue windows GSW observed, GSW says water on >= 1/2 pooled and on more events than not (p < 0.05)"},
               "results": {}, "failures": []}
    try:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        data_dir = os.path.join(hf_home, "geoid_flood")
        tiles = e60.tiles_of(args, data_dir, summary)
        if tiles is None:
            raise RuntimeError("smoke: GEOID sample tiles not in the local Hub cache")
        extra = ensure_extra(args, tiles, data_dir, summary)
        sel = e55.select_chips(e55.chip_candidates(tiles["test"], "test"), CAP, smoke_chips=4 if args.smoke else None)
        clear = np.array([x["clear"] >= e55.MIN_CLEAR for x in sel], bool)
        chips = [x for x, c in zip(sel, clear) if c]
        lab = np.stack([x["lab"] for x in chips])
        ya, oka = e57.labels_of(e55.task_labels(lab, "A"))
        yb, okb = e57.labels_of(e55.task_labels(lab, "B"))
        z = np.load(os.path.join(hb.OUT, f"exp60_masks{suffix}.npz"))
        events = np.array([x["aoi"] for x in chips])
        if not (len(chips) == len(z["event"]) and (events == z["event"]).all() and (ya == z["y_permanent"]).all() and (yb == z["y_after"]).all() and ((oka & okb) == z["ok"]).all()):
            raise RuntimeError(f"chip selection does not reproduce exp60's: {len(chips)} chips vs {len(z['event'])}")
        summary["config"]["chips"] = {"clear": len(chips), "events": len(set(events.tolist())), "aligned_with_exp60": True}
        ok, A_s2, A_s1, B_s1 = z["ok"], z["A_s2pre"], z["A_s1pre"], z["B_s1post"]
        N = len(chips)
        L = {k: np.full((N, G - 1, G - 1), np.nan) for k in ("permwater", "floodmask", "validity", "gsw_water", "gsw_observed")}
        values = {k: set() for k in EXTRA}
        months, n_no_layer, n_gsw_fail = {}, {k: 0 for k in EXTRA}, 0
        t0 = time.time()
        by_tile = {}
        for i, x in enumerate(chips):
            by_tile.setdefault(x["tile"], []).append(i)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
            for n_t, (tid, idx) in enumerate(by_tile.items()):
                t = tiles["test"][tid]
                for layer in EXTRA:
                    p = extra.get(tid, {}).get(layer)
                    if p is None:
                        n_no_layer[layer] += len(idx)
                        continue
                    with rasterio.open(p) as src:
                        for i in idx:
                            a = e55.read_chip(src, chips[i]["r"], chips[i]["c"])[0]
                            values[layer] |= set(np.unique(a).tolist()[:8])
                            L[layer][i] = pool(a, positive=[1])[0]
                ym = pre_month(t["s1grd_pre"])
                months[tid] = ym
                for i in idx:
                    try:
                        g = gsw_chip(t["s1grd_pre"], chips[i]["r"], chips[i]["c"], *ym) if ym else None
                        if g is None:
                            continue
                        obs = (g > 0)
                        L["gsw_observed"][i] = obs.astype(float)[:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH).mean(axis=(1, 3))[SL, SL]
                        L["gsw_water"][i] = pool(np.where(obs, g, 255), positive=[2])[0]
                    except Exception as ex:  # noqa: BLE001
                        n_gsw_fail += 1
                        if n_gsw_fail <= 3:
                            print(f"  gsw read failed for {tid} chip {i}: {ex!r}", flush=True)
                if n_t % 50 == 0:
                    print(f"  tile {n_t + 1}/{len(by_tile)} ({time.time() - t0:.0f}s, {len(_GSW_CACHE)} JRC tiles opened)", flush=True)
        summary["config"].update({"layer_values_seen": {k: sorted(v) for k, v in values.items()}, "chips_without_layer": n_no_layer, "gsw_read_failures": n_gsw_fail,
                                  "pre_event_months": sorted({f"{y}-{m:02d}" for y, m in months.values() if y}), "read_seconds": time.time() - t0})
        np.savez_compressed(os.path.join(hb.OUT, f"exp61_layers{suffix}.npz"), event=events, **{k: v.astype(np.float32) for k, v in L.items()})

        # ---- readings
        perm_w, flood_w, gsw_w = L["permwater"] > 0.5, L["floodmask"] > 0.5, L["gsw_water"] > 0.5
        obs = np.nan_to_num(L["gsw_observed"], nan=0.0) >= MIN_OBSERVED
        d_time, d_sens = (A_s1 != B_s1) & ok, (A_s2 != A_s1) & ok
        dep = d_time & (A_s1 != ya)
        dep_water, dep_dry = dep & A_s1, dep & ~A_s1
        R = {"n_windows": int(ok.sum()), "n_time_only_differing": int(d_time.sum()), "n_residue": int(dep.sum()), "residue_share_of_time_only": share(dep, d_time),
             "residue": {"radar_water_label_not_permanent": int(dep_water.sum()), "radar_dry_label_permanent": int(dep_dry.sum()),
                         "gsw_observed_share": share(obs, dep), "gsw_water_share_among_observed": share(gsw_w, dep & obs),
                         "gsw_water_share_radar_water": share(gsw_w, dep_water & obs), "gsw_water_share_radar_dry": share(gsw_w, dep_dry & obs),
                         "floodmask_share": share(flood_w, dep & ~np.isnan(L["floodmask"])), "permwater_equals_label_share": share(perm_w == ya, dep & ~np.isnan(L["permwater"]))},
             "sanity": {"permwater_equals_label_all_windows": share(perm_w == ya, ok & ~np.isnan(L["permwater"])), "floodmask_equals_flooded_label_all_windows": share(flood_w == (yb & ~ya), ok & ~np.isnan(L["floodmask"])),
                        "validity_share": share(L["validity"] > 0.5, ok & ~np.isnan(L["validity"]))},
             "agreement_with_gsw_all_observed": {"A_s2pre": share(A_s2 == gsw_w, ok & obs), "A_s1pre": share(A_s1 == gsw_w, ok & obs), "label_permanent": share(ya == gsw_w, ok & obs), "n": int((ok & obs).sum())},
             "sensor_only_differing": {"n": int(d_sens.sum()), "gsw_observed_share": share(obs, d_sens), "gsw_sides_with_s1": share(gsw_w == A_s1, d_sens & obs), "gsw_sides_with_s2": share(gsw_w == A_s2, d_sens & obs)}}
        ev_months = {}
        for tid, ym in months.items():
            if ym:
                ev_months.setdefault(tiles["test"][tid]["aoi"], set()).add(f"{ym[0]}-{ym[1]:02d}")
        rows, per = [], {}
        for ev in np.unique(events):
            m = (events == ev)[:, None, None]
            n_dep, n_obs = int((dep & m).sum()), int((dep & obs & m).sum())
            s = share(gsw_w, dep & obs & m)
            per[ev.item()] = {"n_residue": n_dep, "n_residue_observed": n_obs, "gsw_water_share": s, "floodmask_share": share(flood_w, dep & m),
                              "gsw_sides_with_s1_sensor_only": share(gsw_w == A_s1, d_sens & obs & m), "month": ",".join(sorted(ev_months.get(ev.item(), [])))}
            rows.append({"event": ev.item(), **per[ev.item()]})
        R["per_event"] = per
        tests = over_groups({ev: v["gsw_water_share"] - 0.5 for ev, v in per.items() if v["n_residue_observed"] >= MIN_EVENT_WINDOWS and v["gsw_water_share"] == v["gsw_water_share"]})
        R["P1_test"] = tests
        summary["results"] = R
        p1 = bool(R["residue"]["gsw_water_share_among_observed"] >= 0.5 and tests["sign_p"] < 0.05)
        summary["prereg"] = {"P1": p1, "gsw_observed_gate_ok": bool(R["residue"]["gsw_observed_share"] >= 0.5), "complete": not summary["failures"]}
        print(f"residue: {R['n_residue']} of {R['n_time_only_differing']} time-only differing windows ({fmt(R['residue_share_of_time_only'], '.3f')}); radar-water {R['residue']['radar_water_label_not_permanent']}, radar-dry {R['residue']['radar_dry_label_permanent']} | "
              f"GSW observed {fmt(R['residue']['gsw_observed_share'], '.3f')}, water among observed {fmt(R['residue']['gsw_water_share_among_observed'], '.3f')} (radar-water {fmt(R['residue']['gsw_water_share_radar_water'], '.3f')}, radar-dry {fmt(R['residue']['gsw_water_share_radar_dry'], '.3f')}) | "
              f"floodmask {fmt(R['residue']['floodmask_share'], '.3f')} | permwater==label {fmt(R['sanity']['permwater_equals_label_all_windows'], '.3f')} | GSW agreement: S2 head {fmt(R['agreement_with_gsw_all_observed']['A_s2pre'], '.3f')}, S1 head {fmt(R['agreement_with_gsw_all_observed']['A_s1pre'], '.3f')}, label {fmt(R['agreement_with_gsw_all_observed']['label_permanent'], '.3f')} | "
              f"sensor-only: GSW sides with S1 {fmt(R['sensor_only_differing']['gsw_sides_with_s1'], '.3f')} vs S2 {fmt(R['sensor_only_differing']['gsw_sides_with_s2'], '.3f')} | P1 {p1} (events {tests['w']}/{tests['l']}, p={tests['sign_p']:.2g})", flush=True)
        with open(os.path.join(hb.OUT, f"exp61_residue{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
        summary["prereg"] = {"P1": None, "complete": False}
    summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp61_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()

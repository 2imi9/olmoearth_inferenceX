#!/usr/bin/env python
"""exp44: sixteen crop offsets against the four diagonals, at pixel level on hand labels.

Why. exp42's W1 averages the window maps of four crop offsets (0, 1, 2, 3 px, diagonal: dy = dx). The checked Lean
counterexample (WindowDesignProofs.lean) shows that adding the twelve horizontal and vertical phases has no
guaranteed accuracy advantage; it does not show it has none. This run asks the empirical question.

Design, fixed before the run. For every offset (dy, dx) in {0, 1, 2, 3}^2 the tile is cropped to 60 px at that
offset and encoded by the frozen OlmoEarth Base encoder (the four diagonals from exp18's cache, the twelve others
computed here, same encoder, same normalisation, same exp18 head retrained from the valid split with seed 0); each
window map is painted back to the pixels it covers; W16 is the equal-weight mean of the sixteen painted probability
maps, W1 the mean of the four diagonals, both thresholded at 0.5 on the same evaluated pixels (labelled, inside the
57 x 57 region every tiling covers). Primary: W16 against W1 on pixel accuracy per tile, paired one-sided exact sign
test (W16 better) on Bolivia and on the test split; minimum worthwhile effect +0.002 mean tile gain on both; support
needs both. Recorded per tile: N, C corrected, B broken, D = C + B, gain (C - B) / N; pooled (sum C - sum B) / sum N
alongside the preregistered mean of tile gains; the eight-offset subsets (diagonals plus anti-diagonals dy + dx = 3;
diagonals plus the horizontal phases dy = 0, seven distinct offsets) as secondary. Cost: twelve additional forward passes per tile.

Inputs: exp/out/exp18_feats.npz (diagonals), data/floods/. Outputs: exp/out/exp44_summary.json,
exp/out/exp44_sixteen_offsets.csv. --smoke: 4 Bolivia tiles, CPU passes, _smoke outputs.
"""
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp42_window_design as e42  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G = exp18.PATCH, exp18.CROP, exp18.G
OFFSETS = [(dy, dx) for dy in range(4) for dx in range(4)]
DIAG = [(s, s) for s in range(4)]
ANTI = [(dy, 3 - dy) for dy in range(4)]
HORIZ = [(0, dx) for dx in range(4)]
MIN_EFFECT = 0.002


def paint(win_map, dy, dx, size):
    out = np.full((size, size), np.nan)
    out[dy:dy + G * PATCH, dx:dx + G * PATCH] = np.repeat(np.repeat(win_map, PATCH, 0), PATCH, 1)
    return out


def logits_at_offsets(model, s2, cached, head, summary, name):
    """(N, 16, G, G) logits: diagonals from the cache, the rest encoded here with exp18.embed on pre-cropped tiles."""
    N = len(s2)
    out = np.zeros((N, len(OFFSETS), G, G), np.float32)
    t0 = time.time()
    for k, (dy, dx) in enumerate(OFFSETS):
        if dy == dx:
            feats = cached[dy]
        else:
            feats, _ = exp18.embed(model, s2[:, :, dy:dy + CROP, dx:dx + CROP], 0)     # shift 0 on a tile already cropped at (dy, dx)
        out[:, k] = exp18.head_prob_logit(np.asarray(feats, dtype=np.float32), *head)[1]
    summary["testbeds"][name]["extra_passes"] = 12
    summary["testbeds"][name]["encode_seconds"] = time.time() - t0
    return out


def tile_eval(logit16, lab, size):
    p = 1 / (1 + np.exp(-logit16))
    pix = {o: paint(p[k], o[0], o[1], size) for k, o in enumerate(OFFSETS)}
    lo, hi = 3, G * PATCH
    common = np.zeros((size, size), bool); common[lo:hi, lo:hi] = True
    for o in OFFSETS:
        if not np.isfinite(pix[o][common]).all():
            raise RuntimeError("a tiling leaves pixels of the common region unpredicted")
    valid = (lab >= 0) & common
    y = lab == 1

    def avg(offs):
        return np.mean(np.stack([pix[o] for o in offs]), 0)
    maps = {"W1 (4 diagonals)": avg(DIAG), "W16 (all offsets)": avg(OFFSETS), "W8 (diagonals + anti-diagonals)": avg(DIAG + ANTI),
            "W7 (diagonals + horizontal phases)": avg(list(dict.fromkeys(DIAG + HORIZ)))}   # seven distinct offsets: (0,0) is in both
    f_wrong = ((maps["W1 (4 diagonals)"] > 0.5) != y) & valid
    N = int(valid.sum())
    res = {"N": N, "acc": {k: float(1 - (((m > 0.5) != y) & valid).sum() / N) if N else np.nan for k, m in maps.items()}, "CB": {}}
    for k, m in maps.items():
        if k == "W1 (4 diagonals)":                                   # the baseline itself
            continue
        g_wrong = ((m > 0.5) != y) & valid
        C, B = int((f_wrong & ~g_wrong).sum()), int((~f_wrong & g_wrong).sum())
        res["CB"][k] = {"C": C, "B": B, "D": C + B, "gain": (C - B) / N if N else np.nan}
    return res


def analyse(name, logit16, lab, size, summary, rows):
    per, excluded = [], []
    for t in range(len(lab)):
        try:
            r = tile_eval(logit16[t], lab[t], size)
            if r["N"] == 0:
                excluded.append(int(t)); continue
            r["tile"] = t
            per.append(r)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": int(t), "error": repr(ex), "traceback": traceback.format_exc()})
    res = {"tiles": len(lab), "tiles_analysed": len(per), "tiles_excluded_no_labelled_pixel_in_common_region": excluded,
           "pixel_accuracy_mean": {k: float(np.nanmean([r["acc"][k] for r in per])) for k in per[0]["acc"]} if per else {}, "tests": {}}
    for k in per[0]["CB"] if per else []:
        g = np.array([r["CB"][k]["gain"] for r in per])
        w, l, t_ = wins_losses_ties(g)
        one = k.startswith("W16")
        C, B, Nn = sum(r["CB"][k]["C"] for r in per), sum(r["CB"][k]["B"] for r in per), sum(r["N"] for r in per)
        res["tests"][k] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                           "mean_gain": float(g.mean()), "median_gain": float(np.median(g)), "n_tiles": int(len(g)), "C_total": int(C), "B_total": int(B), "N_total": int(Nn),
                           "pooled_gain": (C - B) / Nn if Nn else None}
    pr = res["tests"].get("W16 (all offsets)")
    if pr:
        res["prereg"] = {"sign_p": pr["sign_p"], "mean_gain": pr["mean_gain"], "min_effect": MIN_EFFECT, "passes": bool(pr["sign_p"] < 0.05 and pr["mean_gain"] >= MIN_EFFECT)}
        print(f"{name}: W16 vs W1 per tile {pr['w']}/{pr['l']}/{pr['t']} one-sided p={pr['sign_p']:.2g}, mean tile gain {pr['mean_gain']:+.5f}, pooled {pr['pooled_gain']:+.5f}, "
              f"C {pr['C_total']} B {pr['B_total']} of N {pr['N_total']}; acc " + ", ".join(f"{k}: {v:.4f}" for k, v in res["pixel_accuracy_mean"].items()), flush=True)
        for k, v in res["tests"].items():
            if not k.startswith("W16"):
                print(f"  {k}: {v['w']}/{v['l']}/{v['t']} p={v['sign_p']:.2g} mean gain {v['mean_gain']:+.5f} C {v['C_total']} B {v['B_total']}", flush=True)
    for r in per:
        rows.append({"testbed": name, "tile": r["tile"], "N": r["N"], **{f"acc {k}": v for k, v in r["acc"].items()},
                     **{f"{q} {k}": r["CB"][k][q] for k in r["CB"] for q in ("C", "B", "gain")}})
    summary["results"][name] = res


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp44 sixteen crop offsets against the four diagonals", "smoke": args.smoke, "testbeds": {}, "results": {}, "failures": [],
               "config": {"offsets": OFFSETS, "min_effect": MIN_EFFECT, "prereg": "W16 (equal-weight mean of 16 painted maps) vs W1 (4 diagonals), pixel accuracy per tile, one-sided sign test on both testbeds, mean tile gain >= 0.002 on both",
                          "secondary": "diagonals + anti-diagonals (8 offsets); diagonals + horizontal phases (7 distinct offsets)", "cost": "12 additional forward passes per tile"}}
    rows = []
    model = hb.load_model()
    for name in (["bolivia"] if args.smoke else ["bolivia", "test"]):
        try:
            logits4, s2, lab, _ = e42.load_testbed(name, args, summary)        # diagonals from the cache, head retrained as exp18
            floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
            # rebuild the head exactly as load_testbed did, to score the new offsets with the same head
            paths, _ = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
            if args.smoke:
                z = np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz")); tr_feats = np.asarray(z["tr_base"], dtype=np.float32)
                tr_s2, tr_lab = hb.load_floods_split(paths["valid"], args.smoke_tiles)
                cached = {s: np.asarray(z[f"bolivia_base{s}"], dtype=np.float32) for s in range(4)}
            else:
                z = np.load(os.path.join(hb.OUT, "exp18_feats.npz")); tr_feats = np.asarray(z["tr_base"], dtype=np.float32)
                tr_s2, tr_lab = hb.load_floods_split(paths["valid"], exp18.N_TRAIN_TILES)
                key = "bolivia" if name == "bolivia" else "test"
                cached = {s: np.asarray(z[f"{key}_base{s}"], dtype=np.float32) for s in range(4)}
            tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP]); sel = tr_ok.flatten()
            torch.manual_seed(0)
            from oe_inferencex.evidence import train_logistic_head
            head = train_logistic_head(torch.tensor(tr_feats.reshape(-1, tr_feats.shape[-1])[sel]), tr_y.flatten()[sel])
            logit16 = logits_at_offsets(model, s2, cached, head, summary, name)
            # consistency: the diagonal logits must equal exp42's path
            k0 = OFFSETS.index((0, 0))
            summary["testbeds"][name]["diagonal_consistency_max_abs_diff"] = float(np.abs(logit16[:, k0] - logits4[:, 0]).max())
            # fresh encoder against the float16 cache: re-embed the (1, 1) crop of the first tiles and compare features
            n_chk = min(32, len(s2))
            fresh, _ = exp18.embed(model, s2[:n_chk, :, 1:1 + CROP, 1:1 + CROP], 0)
            diff = np.abs(np.asarray(fresh, dtype=np.float32) - cached[1][:n_chk])
            summary["testbeds"][name]["fresh_vs_cache_shift1"] = {"tiles": int(n_chk), "max_abs_feature_diff": float(diff.max()), "mean_abs_feature_diff": float(diff.mean()),
                                                                  "tolerance_note": "the cache is float16 of the same encoder; differences at the 1e-3 level are the cast, larger ones a changed input path"}
            analyse(name, logit16, lab, s2.shape[-1], summary, rows)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    if len(summary["results"]) == 2:
        complete = not summary["failures"]
        summary["prereg"] = {"supported": bool(complete and all(summary["results"][n].get("prereg", {}).get("passes") for n in ("bolivia", "test"))), "complete": complete}
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp44_sixteen_offsets{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp44_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp44_summary{suffix}.json; failures: {len(summary['failures'])}", flush=True)


if __name__ == "__main__":
    main()

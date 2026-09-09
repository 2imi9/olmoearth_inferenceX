#!/usr/bin/env python
"""exp43: a content-derived partition with a hard majority of W1's pixel decisions, against W1, at pixel level.

Why. exp42's shift-averaged decision (W1) beat the grid window. A checked Lean development (WindowDesignProofs.lean,
2026-09-08) gives the exact accounting for any redesign g against W1 on the same labelled pixels, C corrected, B
broken, gain (C - B) / N, and one conditional guarantee: if a segment has one true class and W1 is correct on more
than half of its pixels, assigning the segment the strict majority of W1's hard pixel predictions makes the whole
segment correct. Whether an image-derived partition meets those conditions is the conjecture this run grades. No
extra encoder pass: the partition comes from the Sentinel-2 image, the votes from W1.

Partition rule, fixed before the run and built without labels. Per tile: pixel features = the twelve bands as
log1p(DN), each standardised within the tile, plus raw NDWI times 2 (thirteen features); k-means with K = 8,
initialised deterministically at the 8 evenly spaced quantiles of the first principal component's scores (sign fixed
so that the largest-magnitude loading is positive), exactly 20 Lloyd iterations; segments = 4-connected components of the cluster map (label propagation; components are unique). Each segment takes the
strict majority of W1's hard predictions (probability > 0.5 = water) over its pixels inside the common 57 x 57
region; ties predict land. Pixels are evaluated on the same valid-label mask as exp42.

Preregistered. Primary: the segment-majority map against W1 on pixel accuracy per tile, paired one-sided exact sign
test (segment map better) on Bolivia and on the test split; minimum worthwhile effect: mean gain >= +0.002 on both.
Support needs p < 0.05 and the minimum effect on both testbeds. Recorded per tile: N, C, B, D = C + B, gain, and the
grading-only diagnostics segment purity (share of segments with >= 2 labelled pixels whose labelled pixels are one
class) and majority correctness (share of pure segments where W1 is right on more than half of its labelled pixels).
Secondary, descriptive: the soft variant (segment mean of W1's probability > 0.5), K = 16 with the same rule, and the
corrected purity statistic exp42 did not report: W1's own pixel errors by grid-window purity. Failure means
unsupported under this protocol, not that no partition can help.

Inputs: exp/out/exp18_feats.npz, data/floods/. Outputs: exp/out/exp43_summary.json, exp/out/exp43_segment_majority.csv.
--smoke: 4 Bolivia tiles from the harness smoke cache, _smoke outputs.
"""
import csv
import json
import os
import sys
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
import exp42_window_design as e42  # noqa: E402
from oe_inferencex.signals import ndwi  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, SHIFTS = e42.PATCH, e42.SHIFTS
K_PRIMARY, K_SECONDARY, ITERS = 8, 16, 20
MIN_EFFECT = 0.002
PURE = 0.1


def features(s2, size):
    """(size*size, 13): standardised log1p bands plus NDWI with weight 2, deterministic."""
    x = np.log1p(np.clip(s2[:, :size, :size].astype(np.float64), 0, None)).reshape(12, -1).T
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    nd = ndwi(s2[:, :size, :size]).reshape(-1)
    return np.concatenate([x, 2.0 * nd[:, None]], 1)                          # NDWI with weight 2, as preregistered (raw, in [-1, 1])


def kmeans_partition(f, size, K):
    """Deterministic k-means (quantile init on the first principal component) + 4-connected components."""
    c = f - f.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    pc1 = vt[0] * (1.0 if vt[0][np.argmax(np.abs(vt[0]))] >= 0 else -1.0)   # canonical sign: largest-magnitude loading positive
    score = c @ pc1
    init = np.quantile(score, (np.arange(K) + 0.5) / K)
    centres = np.stack([f[np.argmin(np.abs(score - q))] for q in init])
    for _ in range(ITERS):
        d = ((f[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        centres = np.stack([f[lab == k].mean(0) if (lab == k).any() else centres[k] for k in range(K)])   # exactly ITERS iterations
    d = ((f[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    cl = lab.reshape(size, size)
    return connected_components(cl)


def connected_components(cl):
    """4-connected components of a label map, without scipy: each pixel starts with its own id and takes the minimum id
    among 4-neighbours of the same cluster until nothing changes (deterministic; components are unique whatever
    computes them). Returns (segments 1..n, n)."""
    h, w = cl.shape
    ids = np.arange(h * w).reshape(h, w)
    while True:
        new = ids.copy()
        same = cl[1:, :] == cl[:-1, :]
        new[1:, :] = np.where(same, np.minimum(new[1:, :], ids[:-1, :]), new[1:, :])
        new[:-1, :] = np.where(same, np.minimum(new[:-1, :], ids[1:, :]), new[:-1, :])
        same = cl[:, 1:] == cl[:, :-1]
        new[:, 1:] = np.where(same, np.minimum(new[:, 1:], ids[:, :-1]), new[:, 1:])
        new[:, :-1] = np.where(same, np.minimum(new[:, :-1], ids[:, 1:]), new[:, :-1])
        if np.array_equal(new, ids):
            break
        ids = new
    _, seg = np.unique(ids, return_inverse=True)
    seg = seg.reshape(h, w) + 1
    return seg, int(seg.max())


def segment_majority(seg, n_seg, w1_pix, common, soft=False):
    """Segment-wise decision from W1 over the common region; ties -> land. Returns a pixel map (NaN outside common)."""
    out = np.full(w1_pix.shape, np.nan)
    hard = (w1_pix > 0.5)
    for s in range(1, n_seg + 1):
        m = (seg == s) & common & np.isfinite(w1_pix)
        if not m.any():
            continue
        if soft:
            out[m] = float(w1_pix[m].mean() > 0.5)
        else:
            water = int(hard[m].sum()); land = int(m.sum()) - water
            out[m] = float(water > land)
    return out


def tile_stats(w1_pix, g_pix, lab, valid, seg, n_seg):
    y = lab == 1
    f_wrong = ((w1_pix > 0.5) != y) & valid
    g_wrong = ((g_pix > 0.5) != y) & valid
    N = int(valid.sum())
    C = int((f_wrong & ~g_wrong).sum()); B = int((~f_wrong & g_wrong).sum())
    pure = correct_major = n_graded = 0
    for s in range(1, n_seg + 1):
        m = (seg == s) & valid
        if m.sum() < 2:
            continue
        n_graded += 1
        ys = y[m]
        if ys.all() or (~ys).all():
            pure += 1
            if (~f_wrong[m]).sum() * 2 > m.sum():
                correct_major += 1
    return {"N": N, "C": C, "B": B, "D": C + B, "gain": (C - B) / N if N else np.nan, "acc_w1": 1 - f_wrong.sum() / N if N else np.nan,
            "acc_seg": 1 - g_wrong.sum() / N if N else np.nan, "segments_graded": n_graded, "purity": pure / n_graded if n_graded else np.nan,
            "majority_correct_given_pure": correct_major / pure if pure else np.nan}


def analyse(name, logits, s2, lab, G, summary, rows):
    N, size = len(s2), s2.shape[-1]
    per, per_soft, per_k16 = [], [], []
    w1_purity = {"err_impure": 0, "err_pure": 0, "n_impure": 0, "n_pure": 0}
    for t in range(N):
        try:
            p_shift = [1 / (1 + np.exp(-logits[t][s])) for s in SHIFTS]
            pix = np.stack([e42.paint(p_shift[s], s, size) for s in SHIFTS])
            lo, hi = SHIFTS[-1], G * PATCH
            common = np.zeros((size, size), bool); common[lo:hi, lo:hi] = True
            if not np.isfinite(pix[:, common]).all():
                raise RuntimeError(f"tile {t}: a tiling leaves pixels of the common region unpredicted")
            w1 = pix.mean(0)
            valid = (lab[t] >= 0) & common
            if not valid.any():
                continue
            f = features(s2[t], size)
            seg, n = kmeans_partition(f, size, K_PRIMARY)
            g = segment_majority(seg, n, w1, common)
            gs = segment_majority(seg, n, w1, common, soft=True)
            seg16, n16 = kmeans_partition(f, size, K_SECONDARY)
            g16 = segment_majority(seg16, n16, w1, common)
            for cand in (g, gs, g16):
                if not np.isfinite(cand[valid]).all():
                    raise RuntimeError(f"tile {t}: a candidate leaves evaluated pixels undecided")
            st = tile_stats(w1, g, lab[t], valid, seg, n); st["tile"] = t; st["n_segments"] = n
            st_soft = tile_stats(w1, gs, lab[t], valid, seg, n)
            st16 = tile_stats(w1, g16, lab[t], valid, seg16, n16)
            per.append(st); per_soft.append(st_soft); per_k16.append(st16)        # atomically, after every variant succeeded
            # W1's own pixel errors by grid-window purity (the statistic exp42 reported for W0 only)
            l = lab[t][:G * PATCH, :G * PATCH].reshape(G, PATCH, G, PATCH)
            nv = (l >= 0).sum(axis=(1, 3)); frac = np.where(nv > 0, (l == 1).sum(axis=(1, 3)) / np.maximum(nv, 1), np.nan)
            impure_w = (frac > PURE) & (frac < 1 - PURE)
            imp_pix = np.repeat(np.repeat(impure_w, PATCH, 0), PATCH, 1)
            imp_full = np.zeros((size, size), bool); imp_full[:G * PATCH, :G * PATCH] = imp_pix
            f_wrong = ((w1 > 0.5) != (lab[t] == 1)) & valid
            w1_purity["err_impure"] += int((f_wrong & imp_full).sum()); w1_purity["err_pure"] += int((f_wrong & ~imp_full).sum())
            w1_purity["n_impure"] += int((valid & imp_full).sum()); w1_purity["n_pure"] += int((valid & ~imp_full).sum())
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": int(t), "error": repr(ex), "traceback": traceback.format_exc()})

    def paired(recs, one_sided):
        if not recs:
            return {"w": 0, "l": 0, "t": 0, "sign_p": 1.0, "one_sided": one_sided, "mean_gain": None, "median_gain": None, "n_tiles": 0, "C_total": 0, "B_total": 0, "N_total": 0,
                    "acc_w1_mean": None, "acc_candidate_mean": None, "purity_mean": None, "majority_correct_given_pure_mean": None, "segments_per_tile": None}
        g = np.array([r["gain"] for r in recs if np.isfinite(r["gain"])])
        w, l, t_ = wins_losses_ties(g)
        return {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one_sided else "two-sided"), "one_sided": one_sided,
                "mean_gain": float(g.mean()) if len(g) else None, "median_gain": float(np.median(g)) if len(g) else None, "n_tiles": int(len(g)),
                "C_total": int(sum(r["C"] for r in recs)), "B_total": int(sum(r["B"] for r in recs)), "N_total": int(sum(r["N"] for r in recs)),
                "acc_w1_mean": float(np.nanmean([r["acc_w1"] for r in recs])), "acc_candidate_mean": float(np.nanmean([r["acc_seg"] for r in recs])),
                "purity_mean": float(np.nanmean([r["purity"] for r in recs])), "majority_correct_given_pure_mean": float(np.nanmean([r["majority_correct_given_pure"] for r in recs])),
                "segments_per_tile": float(np.mean([r["n_segments"] for r in recs])) if recs and "n_segments" in recs[0] else None}
    res = {"tiles": N, "tiles_analysed": len(per), "primary_k8_hard": paired(per, True), "secondary_k8_soft": paired(per_soft, False), "secondary_k16_hard": paired(per_k16, False)}
    pr = res["primary_k8_hard"]
    res["prereg"] = {"sign_p": pr["sign_p"], "mean_gain": pr["mean_gain"], "min_effect": MIN_EFFECT, "passes": bool(pr["sign_p"] < 0.05 and (pr["mean_gain"] or -1) >= MIN_EFFECT)}
    wp = w1_purity
    res["w1_errors_by_grid_window_purity"] = {"share_of_w1_errors_on_impure_windows": wp["err_impure"] / max(wp["err_impure"] + wp["err_pure"], 1),
                                             "w1_error_rate_impure": wp["err_impure"] / max(wp["n_impure"], 1), "w1_error_rate_pure": wp["err_pure"] / max(wp["n_pure"], 1),
                                             "share_of_pixels_in_impure_windows": wp["n_impure"] / max(wp["n_impure"] + wp["n_pure"], 1)}
    for r in per:
        rows.append({"testbed": name, **{k: r[k] for k in ("tile", "N", "C", "B", "D", "gain", "acc_w1", "acc_seg", "n_segments", "purity", "majority_correct_given_pure")}})
    print(f"{name}: primary K=8 hard majority vs W1: {pr['w']}/{pr['l']}/{pr['t']} one-sided p={pr['sign_p']:.2g}, mean gain {pr['mean_gain']:+.4f}, median {pr['median_gain']:+.4f}; "
          f"C {pr['C_total']} B {pr['B_total']} of N {pr['N_total']}; acc {pr['acc_candidate_mean']:.4f} vs W1 {pr['acc_w1_mean']:.4f}; purity {pr['purity_mean']:.3f}, majority correct | pure {pr['majority_correct_given_pure_mean']:.3f}, "
          f"{pr['segments_per_tile']:.0f} segments/tile | soft: mean gain {res['secondary_k8_soft']['mean_gain']:+.4f} | K=16: mean gain {res['secondary_k16_hard']['mean_gain']:+.4f} | "
          f"W1 errors on impure windows {res['w1_errors_by_grid_window_purity']['share_of_w1_errors_on_impure_windows']:.2f} (pixels {res['w1_errors_by_grid_window_purity']['share_of_pixels_in_impure_windows']:.2f})", flush=True)
    summary["results"][name] = res


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp43 segment majority of W1 over a content-derived partition", "smoke": args.smoke, "testbeds": {}, "results": {}, "failures": [],
               "config": {"partition": "log1p bands standardised per tile + 2*NDWI, k-means K=8, quantile init on PC1, 20 iterations, 4-connected components",
                          "decision": "strict majority of W1 hard pixel predictions per segment inside the common region, ties -> land", "min_effect": MIN_EFFECT,
                          "prereg": "K=8 hard majority vs W1, pixel accuracy per tile, one-sided sign test on both testbeds, mean gain >= 0.002 on both", "secondary": "soft majority; K=16"}}
    rows = []
    for name in (["bolivia"] if args.smoke else ["bolivia", "test"]):
        try:
            logits, s2, lab, G = e42.load_testbed(name, args, summary)
            analyse(name, logits, s2, lab, G, summary, rows)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    if len(summary["results"]) == 2:
        complete = not summary["failures"]
        summary["prereg"] = {"supported": bool(complete and all(summary["results"][n]["prereg"]["passes"] for n in ("bolivia", "test"))),
                             "complete": complete, "note": "a run with processing failures is incomplete and cannot be supported"}
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp43_segment_majority{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp43_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp43_summary{suffix}.json; failures: {len(summary['failures'])}", flush=True)


if __name__ == "__main__":
    main()

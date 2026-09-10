#!/usr/bin/env python
"""exp53: test-time adaptation of the head by shift consistency, validated on held-out tilings, graded on hand labels.

Why. The four crop tilings of a scene are four views of the same pixels. Tile-phase measures how much the head's
decisions disagree across them (a ranker that loses to confidence, exp47), and the shift-averaged decision W1
resolves the disagreement by averaging (+1.0 / +0.9 pixel-accuracy points, exp42). The test-time-training family
(MEMO 2022, TENT 2021) does a third thing: adapt the model on each test input so that its views agree. This is the
cheap form of it: adapt only the linear head, per tile, on that tile's own four feature maps, with a label-free
validation built in, fit on two tilings and stop when the other two stop agreeing more. Labels grade, never adapt.
ViT3 (arXiv 2512.01643) is the other kind of test-time training, an attention replacement trained end to end, and
has nothing to lend a frozen encoder; this is the kind that does.

Design. exp18's head h0 on the cached v1 Base features (four crop offsets, Bolivia 441 tiles, the exp18 test sample
of 800 tiles). Per tile, a copy (w, b) of h0 is updated by Adam (lr 1e-3) for at most K = 20 steps on
  memo         the binary entropy of the shift-averaged pixel probability over the fit tilings {0, 2}, plus the anchor
               lambda (||w - w0||^2 / ||w0||^2 + (b - b0)^2), lambda = 0.01;
  consistency  the variance across the fit tilings of the painted pixel probabilities, plus the same anchor;
stopping when the held-out disagreement, the mean absolute difference between the painted probabilities of tilings
1 and 3 on the common region, has not improved for three steps; the weights at the best held-out disagreement are
kept. The adapted decision is W1 with the adapted head over all four tilings; the baseline is W1 with h0, whose
pixel accuracy reproduces exp42 (0.9071 / 0.9503 on the 57 x 57 region every tiling covers).

Graded on hand labels never seen by the adaptation: per-tile pixel accuracy of the adapted W1 against the baseline
W1 (exact sign test, one-sided, adapted better; mean and median gain), and, secondary, the adapted decision's own
error set under exp47's protocol (averaged confidence against tile-phase and the NDWI-level control).

Preregistered, stated before the run:
  P1  the memo variant beats the baseline W1 on pixel accuracy on BOTH testbeds: per-tile sign test p < 0.05 and
      mean gain >= 0.002 (exp44's minimum worthwhile effect). Falsification: averaging already extracts what the
      views agree on; the ledger row reads not supported.
Secondary, descriptive: the consistency variant; steps taken and the share of tiles whose held-out disagreement
improved at all; the share of windows whose decision changed; the ranker check.

Inputs: exp/out/exp18_feats.npz, data/floods/. Outputs: exp/out/exp53_summary.json, exp/out/exp53_shift_tta.csv.
CPU is enough. --smoke: 4 Bolivia tiles from the exp42 smoke cache, _smoke outputs.
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
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp47_served_ranker as e47  # noqa: E402
import exp49_shrug_signals as e49  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.signals import ndwi_level, shift_averaged_probability  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS, PAD = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS, exp18.PAD
LR, K, PATIENCE, LAMBDA, MIN_EFFECT = 1e-3, 20, 3, 0.01, 0.002
FIT, HELD = (0, 2), (1, 3)
LO, HI = SHIFTS[-1], CROP                                                    # the common region [3, 60)
VARIANTS = ("memo", "consistency")
W1C, TILE, CTRL_L = e47.W1C, e47.TILE, e47.CTRL_L


def paint(p, s):
    """(G, G) window probabilities at crop offset s -> the common region's pixels (57, 57), differentiable."""
    pix = p.repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)          # (60, 60) covering pixels [s, s + 60)
    return pix[LO - s:HI - s, LO - s:HI - s]


def adapt(feats, w0, b0, variant):
    """feats (4, G, G, D) torch; returns adapted (w, b), steps taken, held-out disagreement before and at the best step."""
    w, b = w0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=LR)
    w0n = float((w0 ** 2).sum())

    def probs(w_, b_):
        return {s: torch.sigmoid(feats[s] @ w_ + b_) for s in SHIFTS}

    def held_out(pr):
        return float((paint(pr[HELD[0]], HELD[0]) - paint(pr[HELD[1]], HELD[1])).abs().mean())

    with torch.no_grad():
        pr = probs(w, b)
        best = ho0 = held_out(pr)
    best_w, best_b, best_step, since = w.detach().clone(), b.detach().clone(), 0, 0
    for step in range(1, K + 1):
        pr = probs(w, b)
        fit = torch.stack([paint(pr[s], s) for s in FIT])                    # (2, 57, 57)
        if variant == "memo":
            m = fit.mean(0).clamp(1e-6, 1 - 1e-6)
            loss = -(m * m.log() + (1 - m) * (1 - m).log()).mean()
        else:
            loss = fit.var(0, unbiased=False).mean()
        loss = loss + LAMBDA * (((w - w0) ** 2).sum() / w0n + (b - b0) ** 2).squeeze()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ho = held_out(probs(w, b))
        if ho < best - 1e-6:
            best, best_w, best_b, best_step, since = ho, w.detach().clone(), b.detach().clone(), step, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best_w, best_b, best_step, ho0, best


def decision(feats, w, b):
    """W1 with a head: (4, G, G) window maps, the averaged pixel map, and the pooled window map."""
    with torch.no_grad():
        pshift = np.stack([torch.sigmoid(feats[s] @ w + b).numpy() for s in SHIFTS])
    pix = shift_averaged_probability(pshift, patch=PATCH)[0]
    return pshift, pix


def pixel_accuracy(pix, lab):
    m = lab[LO:HI, LO:HI] >= 0
    return float(((pix[LO:HI, LO:HI] > 0.5) == (lab[LO:HI, LO:HI] == 1))[m].mean()) if m.any() else float("nan")


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp53 test-time adaptation of the head by shift consistency", "smoke": args.smoke,
               "config": {"lr": LR, "max_steps": K, "patience": PATIENCE, "lambda": LAMBDA, "fit_shifts": FIT, "held_out_shifts": HELD, "min_effect": MIN_EFFECT, "variants": VARIANTS,
                          "prereg": "P1 memo beats the baseline W1 on pixel accuracy on both testbeds: one-sided per-tile sign test p < 0.05 and mean gain >= 0.002"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, _ = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
    cache = np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz" if args.smoke else "exp18_feats.npz"))
    trf = np.asarray(cache["tr_base"], dtype=np.float32)
    tr_s2, tr_lab = hb.load_floods_split(paths["valid"], args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES)
    if len(trf) != len(tr_s2):
        raise SystemExit("training cache length mismatch")
    w0, b0 = e47.head_from(trf, tr_lab)
    w0, b0 = w0.float(), b0.float()
    splits = {"bolivia": hb.load_floods_split(paths["bolivia"])}
    if not args.smoke:
        splits["test"] = hb.load_floods_split(os.path.join(floods_dir, "flood_test_data.pt"), n=exp18.N_TEST_TILES, seed=1)
    if args.smoke:
        splits["bolivia"] = (splits["bolivia"][0][:args.smoke_tiles], splits["bolivia"][1][:args.smoke_tiles])

    for name, (s2, lab) in splits.items():
        t0 = time.time()
        try:
            key = f"{'bolivia' if name == 'bolivia' else 'test'}_base"
            F = {s: np.asarray(cache[f"{key}{s}"], dtype=np.float32) for s in SHIFTS}
            assert all(len(F[s]) == len(s2) for s in SHIFTS), "cache does not cover this split"
            N = len(s2)
            y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
            sl = slice(1, G)
            res = {"n_tiles": N}
            base_pix, base_w1 = [], []
            per = {v: {"pix": [], "w1": [], "steps": [], "ho0": [], "ho_best": []} for v in VARIANTS}
            for t in range(N):
                feats = {s: torch.tensor(F[s][t]) for s in SHIFTS}
                pshift0, pix0 = decision(feats, w0, b0)
                base_pix.append(pixel_accuracy(pix0, lab[t])); base_w1.append(pshift0)
                for v in VARIANTS:
                    w, b, steps, ho0, hob = adapt(feats, w0, b0, v)
                    pshift, pix = decision(feats, w, b)
                    per[v]["pix"].append(pixel_accuracy(pix, lab[t])); per[v]["w1"].append(pshift)
                    per[v]["steps"].append(steps); per[v]["ho0"].append(ho0); per[v]["ho_best"].append(hob)
            base_pix = np.array(base_pix)
            res["baseline_pixel_accuracy_mean"] = float(np.nanmean(base_pix))
            pshift_b = np.stack(base_w1, axis=1)                                 # (4, N, G, G)
            w1_b = e49.w1_windows(pshift_b)[:, sl, sl]; yy, okk = y[:, sl, sl], ok[:, sl, sl]
            err_b = ((w1_b > 0.5) != (yy > 0.5)).astype(np.float64)
            ctrl = np.stack([ndwi_level(x, patch=PATCH, size=CROP) for x in s2])[:, sl, sl]
            res["baseline"] = e47.score({W1C: -np.abs(w1_b - 0.5), TILE: exp18.aligned_tile_phase(pshift_b)[:, sl, sl], CTRL_L: ctrl}, err_b, okk)
            res["baseline"]["window_accuracy"] = float(1 - err_b[okk].mean())
            for v in VARIANTS:
                pix = np.array(per[v]["pix"]); g = pix - base_pix; g = g[np.isfinite(g)]
                w_, l_, t_ = wins_losses_ties(g)
                pshift = np.stack(per[v]["w1"], axis=1)
                w1 = e49.w1_windows(pshift)[:, sl, sl]
                err = ((w1 > 0.5) != (yy > 0.5)).astype(np.float64)
                r = {"pixel_accuracy_mean": float(np.nanmean(pix)), "mean_gain": float(g.mean()), "median_gain": float(np.median(g)),
                     "w": w_, "l": l_, "t": t_, "sign_p": sign_test(w_, l_, "greater"), "n_tiles": int(len(g)),
                     "steps_median": float(np.median(per[v]["steps"])), "share_tiles_held_out_improved": float(np.mean(np.array(per[v]["steps"]) > 0)),
                     "held_out_disagreement_before": float(np.mean(per[v]["ho0"])), "held_out_disagreement_after": float(np.mean(per[v]["ho_best"])),
                     "share_windows_flipped": float(((w1 > 0.5) != (w1_b > 0.5))[okk].mean()),
                     "ranking": e47.score({W1C: -np.abs(w1 - 0.5), TILE: exp18.aligned_tile_phase(pshift)[:, sl, sl], CTRL_L: ctrl}, err, okk)}
                r["ranking"]["window_accuracy"] = float(1 - err[okk].mean())
                r["passes"] = bool(r["sign_p"] < 0.05 and r["mean_gain"] >= MIN_EFFECT)
                res[v] = r
                print(f"{name}/{v}: pixel acc {r['pixel_accuracy_mean']:.4f} vs baseline {res['baseline_pixel_accuracy_mean']:.4f} (gain {r['mean_gain']:+.4f}, {w_}/{l_}/{t_}, p={r['sign_p']:.2g}) | "
                      f"steps median {r['steps_median']:.0f}, held-out improved on {r['share_tiles_held_out_improved']:.2f} of tiles, {r['share_windows_flipped']:.3f} of windows flipped | "
                      f"E-AURC confidence {r['ranking']['pooled_eaurc'][W1C]:.4f} (baseline {res['baseline']['pooled_eaurc'][W1C]:.4f}), NDWI {r['ranking']['pooled_eaurc'][CTRL_L]:.4f} -> {r['passes']}", flush=True)
                rows.append({"split": name, "variant": v, "pixel_acc": r["pixel_accuracy_mean"], "baseline_pixel_acc": res["baseline_pixel_accuracy_mean"], "mean_gain": r["mean_gain"],
                             "w": w_, "l": l_, "t": t_, "sign_p": r["sign_p"], "steps_median": r["steps_median"], "eaurc_confidence": r["ranking"]["pooled_eaurc"][W1C],
                             "baseline_eaurc_confidence": res["baseline"]["pooled_eaurc"][W1C], "passes": r["passes"]})
            res["seconds"] = time.time() - t0
            summary["results"][name] = res
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    both = [summary["results"].get(n, {}).get("memo", {}).get("passes") for n in ("bolivia", "test")]
    summary["prereg"] = {"P1": bool(all(both)) if all(x is not None for x in both) else None, "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp53_shift_tta{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp53_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp53_summary{suffix}.json; prereg {summary['prereg']}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()

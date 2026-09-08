#!/usr/bin/env python
"""exp40: preregistered replication of the pixel-statistics neighbourhood contradiction on an independent testbed.

Why. In exp39 the ablation won: the share of a window's 32 spectral look-alikes in a disjoint bank that the same head
predicts differently beat confidence on Bolivia hand labels at every budget and on E-AURC (the first hand-label
ranking win), and was null against WorldCover. It was the ablation, not the candidate, and Bolivia is one flood
event, so it needs a preregistered test on labels it has not seen.

Preregistered, before the run. Queries: the Sen1Floods11 test split as exp18 sampled it (800 tiles, seed 1; hand
labels from several regions and events; the exp18 head, trained on the valid split, never saw them; features cached
by exp18). Bank: the 441 Bolivia tiles, one event, with the same head's predictions, standardised pixel statistics
as exp39 defines them (means of the twelve log bands, NDWI mean and std), k = 32. Signals: the pixel-statistics
contradiction (primary), the OlmoEarth-feature-space contradiction (secondary) and U+ (midrank mean of confidence
and the primary), scored next to the harness references and controls. Primary tests, one-sided, the primary against
confidence: (a) capture at the 10% budget, per tile over the tiles with 3 <= errors <= n - 3 (exact sign test,
p < 0.05) and a tile bootstrap of the pooled gain whose 95% interval must exclude zero (the exp36 criterion, stricter
than one-sided); (b) E-AURC per tile (exact sign test, p < 0.05). Support requires (a) per tile and pooled and (b). Budgets 5% and 20%, the
secondary signals, strata and everything else two-sided and descriptive. Falsification: no gain on this testbed, or
a gain confined to the 5% budget.

Inputs: exp/out/exp18_feats.npz (tr_base, test_base0..3, bolivia_base0), data/floods/. Outputs:
exp/out/exp40_summary.json, exp/out/exp40_pixel_contradiction_replication.csv, exp/out/exp40_cache.npz.
--smoke: CPU, 4 Bolivia tiles as queries and 4 valid tiles as the bank (roles differ from the full run, code path
identical), _smoke outputs.
"""
import os
import sys
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp39_neighbour_contradiction as e39  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import capture_at_budget_expected  # noqa: E402
from oe_inferencex.signals import midrank_pct  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

CONF, TILE, BOUND = hb.CONF, hb.TILE, hb.BOUND
BUDGETS, PREREG_BUDGET, N_BOOT = e39.BUDGETS, e39.PREREG_BUDGET, e39.N_BOOT
C_PIX, C_OE = e39.C_PIX, e39.C_OE
COMBO = "combination conf+pixel contradiction (secondary)"
NEW_NAMES = [C_PIX, C_OE, COMBO]
PRIMARY = C_PIX


def load_queries_and_bank(model, args, summary):
    """Full run: queries = test split (exp18's sample), bank = Bolivia. Smoke: the harness's Bolivia/valid smoke."""
    if args.smoke:
        ctx = hb.load_part_b(model, args, summary, "exp40")
        if ctx is None:
            return None, None
        bank = {"feats": np.asarray(ctx["z"]["tr_base"], dtype=np.float32), "s2": ctx["tr_s2"], "name": "valid tiles (smoke only)"}
        return ctx, bank
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, missing = hb.ensure_floods(floods_dir, allow_download=True)
    test_path = os.path.join(floods_dir, "flood_test_data.pt")
    if not os.path.exists(test_path):
        raise RuntimeError(f"{test_path} missing")
    z = np.load(os.path.join(hb.OUT, "exp18_feats.npz"))
    for k in ("tr_base", "test_base0", "test_base1", "test_base2", "test_base3", "bolivia_base0"):
        if k not in z.files:
            raise RuntimeError(f"exp18_feats.npz lacks {k}")
    tr_s2, tr_lab = hb.load_floods_split(paths["valid"], exp18.N_TRAIN_TILES)          # seed 0, as exp18 and the harness
    q_s2, q_lab = hb.load_floods_split(test_path, n=exp18.N_TEST_TILES, seed=1)          # exp18's cached sample
    b_s2, _ = hb.load_floods_split(paths["bolivia"])
    CROP, G = exp18.CROP, exp18.G
    tr_feats, q_feats = np.asarray(z["tr_base"], dtype=np.float32), np.asarray(z["test_base0"], dtype=np.float32)
    b_feats = np.asarray(z["bolivia_base0"], dtype=np.float32)
    if len(q_s2) != len(q_feats) or len(b_s2) != len(b_feats) or len(tr_s2) != len(tr_feats):
        raise RuntimeError(f"image/feature counts differ: queries {len(q_s2)}/{len(q_feats)}, bank {len(b_s2)}/{len(b_feats)}, train {len(tr_s2)}/{len(tr_feats)}")
    tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    sel = tr_ok.flatten()
    D = tr_feats.shape[-1]
    torch.manual_seed(0)
    head = train_logistic_head(torch.tensor(tr_feats.reshape(-1, D)[sel]), tr_y.flatten()[sel])
    y, ok = exp18.patch_labels(q_lab[:, :CROP, :CROP])
    p_shift = np.stack([exp18.head_prob_logit(np.asarray(z[f"test_base{s}"], dtype=np.float32), *head)[0] for s in hb.SHIFTS])
    p, logit = exp18.head_prob_logit(q_feats, *head)
    err = ((p > 0.5) != (y > 0.5)).astype(np.float64)
    acc = 1 - err[ok].mean()
    summary["part_b"].update({"queries": "Sen1Floods11 test split, exp18's 800-tile sample (seed 1)", "query_tiles": int(len(q_s2)),
                              "valid_patches": int(ok.sum()), "head_acc": float(acc), "feats_source": os.path.join(hb.OUT, "exp18_feats.npz")})
    print(f"queries: test split, {len(q_s2)} tiles, valid patches {int(ok.sum())}, head accuracy {acc:.3f}", flush=True)
    ctx = {"N": len(q_s2), "G": G, "CROP": CROP, "bo_s2": q_s2, "p_shift": p_shift, "p": p, "logit": logit, "err": err, "ok": ok,
           "acc": float(acc), "ev_feats": q_feats, "hb": head}
    bank = {"feats": b_feats, "s2": b_s2, "name": "Bolivia tiles (one event)"}
    return ctx, bank


def part_a(model, args, summary, rows, cache):
    summary["part_a"]["skipped"].append({"scene": "*", "reason": "exp40 is a hand-label replication only; the WorldCover result is in exp39"})


def part_b(model, args, summary, rows, cache):
    ctx, bank = load_queries_and_bank(model, args, summary)
    if ctx is None:
        return
    N, G, ok, err, CROP = ctx["N"], ctx["G"], ctx["ok"], ctx["err"], ctx["CROP"]
    sigs = hb.base_signals_b(ctx)
    q_pred = (ctx["p"] > 0.5).astype(np.float64).reshape(-1)
    bank_pred = (exp18.head_prob_logit(bank["feats"], *ctx["hb"])[0] > 0.5).astype(np.float64).reshape(-1)
    summary["part_b"]["bank"] = {"name": bank["name"], "tiles": int(len(bank["feats"])), "windows": int(bank_pred.size), "water_share_of_predictions": float(bank_pred.mean())}
    print(f"bank: {bank['name']}, {len(bank['feats'])} tiles, {bank_pred.size} windows", flush=True)
    q_pix = np.stack([e39.pixel_embedding(t, CROP) for t in ctx["bo_s2"]]).reshape(N * G * G, 14)
    b_pix = np.stack([e39.pixel_embedding(t, CROP) for t in bank["s2"]]).reshape(-1, 14)
    sigs[C_PIX] = e39.neighbour_scores(e39.standardise(q_pix, b_pix), e39.standardise(b_pix, b_pix), bank_pred, q_pred)[0].reshape(N, G, G)
    D = bank["feats"].shape[-1]
    sigs[C_OE] = e39.neighbour_scores(ctx["ev_feats"].reshape(N * G * G, D), bank["feats"].reshape(-1, D), bank_pred, q_pred)[0].reshape(N, G, G)
    cmb = np.full((N, G, G), np.nan)
    for t in range(N):
        m = ok[t]
        if m.any():
            cmb[t][m] = (midrank_pct(sigs[CONF][t][m]) + midrank_pct(sigs[C_PIX][t][m])) / 2
    sigs[COMBO] = cmb
    for k in NEW_NAMES:
        bad = ~np.isfinite(np.asarray(sigs[k], dtype=np.float64)[ok])
        if bad.any():
            raise RuntimeError(f"{k}: {int(bad.sum())} non-finite values on valid windows")
    cache["queries_pix"], cache["queries_oe"] = np.asarray(sigs[C_PIX], dtype=np.float32), np.asarray(sigs[C_OE], dtype=np.float32)
    per_b, tiles = hb.finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO, unit="bolivia" if args.smoke else "test")
    # (b) E-AURC per tile, one-sided for the primary
    gains = np.array(per_b[CONF]) - np.array(per_b[C_PIX])
    w, l, t_ = wins_losses_ties(gains)
    summary["part_b"]["prereg_eaurc"] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater"), "one_sided": True, "median_gain": float(np.median(gains)),
                                         "pooled_eaurc": summary["part_b"]["pooled_eaurc"][C_PIX], "pooled_eaurc_confidence": summary["part_b"]["pooled_eaurc"][CONF]}
    # (a) capture per tile and pooled; one-sided only for the primary pair at 10%
    keys = [CONF, C_PIX, C_OE, COMBO, TILE, BOUND]
    cap = {t: {k: capture_at_budget_expected(np.asarray(sigs[k][t])[ok[t]], err[t][ok[t]], BUDGETS) for k in keys} for t in tiles}
    pairs = [(C_PIX, CONF), (C_OE, CONF), (COMBO, CONF), (C_PIX, C_OE)]
    e39.PREREG_ONE_SIDED = {(C_PIX, CONF)}
    summary["part_b"]["capture_tests"] = e39.capture_tests(cap, pairs)
    all_idx = [t for t in range(N) if ok[t].any()]
    pooled = {k: np.concatenate([np.asarray(sigs[k][t])[ok[t]] for t in all_idx]) for k in keys}
    pooled_err = np.concatenate([err[t][ok[t]] for t in all_idx])
    sizes = np.array([int(ok[t].sum()) for t in all_idx])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    rng = np.random.default_rng(1)
    gains_b = {pair: {bud: [] for bud in BUDGETS} for pair in pairs}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(all_idx), len(all_idx))
        sel = np.concatenate([np.arange(starts[i], starts[i] + sizes[i]) for i in pick])
        e = pooled_err[sel]
        if e.sum() == 0:
            continue
        cc = {k: capture_at_budget_expected(pooled[k][sel], e, BUDGETS) for k in keys}
        for a, b in pairs:
            for bud in BUDGETS:
                gains_b[(a, b)][bud].append(cc[a][bud] - cc[b][bud])
    full = {k: capture_at_budget_expected(pooled[k], pooled_err, BUDGETS) for k in keys}

    def boot(g):
        g = np.array(g)
        if len(g) == 0:
            return {"boot_lo": None, "boot_hi": None, "p_better": None, "n_boot": 0}
        return {"boot_lo": float(np.percentile(g, 2.5)), "boot_hi": float(np.percentile(g, 97.5)), "p_better": float((g > 0).mean()), "n_boot": int(len(g))}
    summary["part_b"]["pooled_capture"] = {k: {str(b): full[k][b] for b in BUDGETS} for k in keys}
    summary["part_b"]["pooled_capture_gain"] = {f"{a} vs {b}": {str(bud): {"gain": full[a][bud] - full[b][bud], **boot(gains_b[(a, b)][bud])} for bud in BUDGETS} for a, b in pairs}
    ct = summary["part_b"]["capture_tests"][f"{C_PIX} vs {CONF}"][str(PREREG_BUDGET)]
    pg = summary["part_b"]["pooled_capture_gain"][f"{C_PIX} vs {CONF}"][str(PREREG_BUDGET)]
    ea = summary["part_b"]["prereg_eaurc"]
    summary["part_b"]["prereg"] = {"capture_10_per_tile": ct, "capture_10_pooled": pg, "eaurc_per_tile": ea,
                                   "supported": bool(ct["sign_p"] < 0.05 and (pg["boot_lo"] or 0) > 0 and ea["sign_p"] < 0.05)}
    summary["part_b"]["strata"] = {k: e39.strata(sigs, err, ok, k) for k in (C_PIX, C_OE)}
    for a, b in pairs:
        c_ = summary["part_b"]["capture_tests"][f"{a} vs {b}"]
        g_ = summary["part_b"]["pooled_capture_gain"][f"{a} vs {b}"]
        print(f"B {a} vs {b}: " + " | ".join(f"{bud}: {c_[str(bud)]['w']}/{c_[str(bud)]['l']}/{c_[str(bud)]['t']} p={c_[str(bud)]['sign_p']:.2g} pooled {full[a][bud]:.3f} vs {full[b][bud]:.3f} "
                                           + (f"CI [{g_[str(bud)]['boot_lo']:+.3f}, {g_[str(bud)]['boot_hi']:+.3f}]" if g_[str(bud)]['boot_lo'] is not None else "CI n/a") for bud in BUDGETS), flush=True)
    print(f"prereg E-AURC per tile, {C_PIX} vs confidence: {ea['w']}/{ea['l']}/{ea['t']} one-sided p={ea['sign_p']:.2g}; pooled {ea['pooled_eaurc']:.4f} vs {ea['pooled_eaurc_confidence']:.4f}; supported={summary['part_b']['prereg']['supported']}", flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"k": e39.K, "budgets": list(BUDGETS), "prereg_budget": PREREG_BUDGET, "n_boot": N_BOOT, "primary_score": PRIMARY, "combination": COMBO,
              "queries": "Sen1Floods11 test split, exp18's 800-tile sample (seed 1), hand labels", "bank": "Bolivia tiles (441), predictions of the exp18 head",
              "prereg": {"primary": f"{C_PIX} vs confidence: (a) capture at 10% per tile one-sided + tile bootstrap of the pooled gain, (b) E-AURC per tile one-sided; support needs (a) per tile and pooled and (b)",
                         "secondary": f"{C_OE}, U+, budgets 5% and 20%, strata; two-sided", "falsification": "no gain on this testbed, or a gain confined to the 5% budget"}}
    hb.run("exp40", "exp40 pixel-statistics neighbourhood contradiction: replication on the Sen1Floods11 test split", config, part_a, part_b, args, "exp40_pixel_contradiction_replication")


if __name__ == "__main__":
    main()

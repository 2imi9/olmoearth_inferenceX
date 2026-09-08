#!/usr/bin/env python
"""exp39: neighbourhood contradiction as an inference-time error signal, in three embedding spaces.

Why. Every representation-side signal tried so far read the representation through the model's own head or its
pretraining objective (exp28-exp36) and none beat confidence on expert labels. This experiment asks a different
question of the representation: does the model's prediction on a window contradict its predictions on the windows
that look most like it? For a query window with prediction y, take its k = 32 nearest neighbours by cosine in an
embedding space, from a bank of windows that share no tile or river with the query, and score the share of
neighbours the same head predicts differently (contradiction = 1 - q(y)); the binary entropy of the neighbour
vote is a secondary reading. No label enters: the bank carries the model's own predictions.

Three embedding spaces, because the ablations are the test. (1) OlmoEarth: the frozen encoder's pooled features
(the caches of exp11 and exp18). (2) Pixel statistics: per-window means of the twelve log bands plus NDWI mean and
std, standardised; if this matches the others, the semantic neighbourhood adds nothing. (3) AnySat (Astruc et al.
2025, torch.hub gastruc/anysat, base, fp32, one date, bands standardised per testbed): the dense output pooled to
the 4-px window. Its per-pixel vector has two halves, the contextual 40 m patch token and the local sub-patch
embedding; on a single-date input the contextual half is dominated by position within the tile (checked: it barely
moves when the content shifts by one window) and the local half moves exactly with the content, so the local half
is the P2 space and the contextual half a secondary space. An out-of-family representation: a win here that (1)
does not show is evidence that outside geometry carries information the OlmoEarth margin does not (issue 6).

Preregistered, before the run. Testbeds and machinery as exp36: the 27 WorldCover rule scenes (one vote per river)
and Sen1Floods11 Bolivia hand labels (per tile, tile-clustered bootstrap), every candidate scored next to
confidence, tile-phase, the boundary indicator and the NDWI-gradient, constant, S2-variance and NDWI-level
controls. Primary tests, one-sided, capture of the errors at the 10% budget on Bolivia (per-tile exact sign test
over the tiles with 3 <= errors <= n - 3, and the tile bootstrap of the pooled gain), with 5% and 20% and E-AURC
alongside: P1, OlmoEarth-space contradiction beats confidence; P2, AnySat-space contradiction beats confidence and
beats OlmoEarth-space contradiction. The preregistered combination U+ is the midrank mean of confidence and the
AnySat-space contradiction (the harness's primary); the weighted 0.75/0.25 combination suggested in the
brainstorm is reported as secondary. Falsification: no hand-label gain over confidence, or a gain that the
pixel-statistics ablation reproduces. A strata check reports, within confidence quintiles and within boundary and
NDWI-ambiguity strata, the error rate among windows whose contradiction is in the stratum's top 20% against the
rest, the exp37 statistic applied to a score. Nulls are recorded.

Banks. Bolivia: the Sen1Floods11 test split (800 tiles, other regions and events, never used to train the head;
exp18's cached features test_base0, its images for the pixel and AnySat spaces). Scenes: the windows of the rule
scenes on other rivers. AnySat dates: the scene date for the scenes; day 45 for every flood tile (event dates are
not in the files). If AnySat cannot be loaded the summary records it and P2 is void; P1 and the ablation still run.

Inputs: exp/out/exp11_scenes.npz, exp/out/exp11_feats.npz, exp/out/exp03_cache.npz, exp/out/exp18_feats.npz,
data/floods/. Outputs: exp/out/exp39_summary.json, exp/out/exp39_neighbour_contradiction.csv, exp/out/exp39_cache.npz
(per-window scores). --smoke: CPU, 2 scenes at 64 px, 4 tiles, the valid tiles as the flood bank, _smoke outputs.
"""
import datetime
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import harness_ab as hb  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402
from oe_inferencex.metrics import capture_at_budget_expected  # noqa: E402
from oe_inferencex.signals import S2_BANDS, midrank_pct, ndwi  # noqa: E402
from oe_inferencex import stats as stats_lib  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

DEV = hb.DEV
CONF, TILE, BOUND, CTRL, CTRL_LVL = hb.CONF, hb.TILE, hb.BOUND, hb.CTRL, hb.CTRL_LVL
K = 32
BUDGETS = (0.05, 0.10, 0.20)
PREREG_BUDGET = 0.10
N_BOOT = 2000
AS_PATCH_M = 40                     # AnySat patch size in metres: 40 m = one 4-px window at 10 m
AS_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
AS_IDX = [S2_BANDS.index(b) for b in AS_BANDS]
FLOOD_DOY = 45
C_OE, H_OE = "contradiction (OlmoEarth neighbours)", "entropy (OlmoEarth neighbours)"
C_PIX = "contradiction (pixel-statistics neighbours)"
C_AS, H_AS = "contradiction (AnySat local neighbours)", "entropy (AnySat local neighbours)"
C_ASX = "contradiction (AnySat context neighbours, secondary)"
COMBO = "combination conf+AnySat contradiction (prereg)"
COMBO_OE = "combination conf+OlmoEarth contradiction (secondary)"
WCOMBO = "weighted 0.75 conf + 0.25 contradiction (secondary)"
NEW_NAMES = [C_OE, H_OE, C_PIX, C_AS, H_AS, C_ASX, COMBO, COMBO_OE, WCOMBO]
PRIMARY = C_AS
PREREG_ONE_SIDED = {(C_OE, CONF), (C_AS, CONF), (C_AS, C_OE)}   # the only one-sided comparisons, at the 10% budget


# ----------------------------------------------------------------------------- scores
def neighbour_scores(q_emb, bank_emb, bank_pred, q_pred, k=K, chunk=2048):
    """Contradiction and neighbour-vote entropy for query windows against a bank (cosine top-k on DEV)."""
    Q = F.normalize(torch.as_tensor(np.asarray(q_emb, dtype=np.float32), device=DEV), dim=1)
    B = F.normalize(torch.as_tensor(np.asarray(bank_emb, dtype=np.float32), device=DEV), dim=1)
    bp = torch.as_tensor(np.asarray(bank_pred, dtype=np.float32), device=DEV)
    k = min(k, len(B))
    q1 = []
    for i in range(0, len(Q), chunk):
        idx = (Q[i:i + chunk] @ B.T).topk(k, dim=1).indices
        q1.append(bp[idx].mean(1))
    q1 = torch.cat(q1).cpu().numpy().astype(np.float64)            # share of neighbours predicted water
    q_same = np.where(np.asarray(q_pred) > 0.5, q1, 1 - q1)
    eps = 1e-9
    entropy = -(q1 * np.log2(q1 + eps) + (1 - q1) * np.log2(1 - q1 + eps))
    return 1 - q_same, entropy


def pixel_embedding(img12, size):
    """(12, H, W) DN -> (G, G, 14): per-window means of log1p bands, NDWI mean and NDWI std."""
    x = np.log1p(np.clip(img12[:, :size, :size].astype(np.float64), 0, None))
    G = size // hb.PATCH
    b = x.reshape(12, G, hb.PATCH, G, hb.PATCH).mean(axis=(2, 4)).transpose(1, 2, 0)
    nd = ndwi(img12[:, :size, :size]).reshape(G, hb.PATCH, G, hb.PATCH)
    return np.concatenate([b, nd.mean(axis=(1, 3))[..., None], nd.std(axis=(1, 3))[..., None]], axis=-1)


def standardise(emb, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    return (emb - mu) / sd


def scene_doy(date):
    """Day of year (0 = 1 January) from the scene archive's (day, month0, year); month0 is 0-based as in the OlmoEarth
    timestamps the harness builds. An invalid date raises, so a wrong convention cannot pass silently."""
    d, m0, y = (int(v) for v in date)
    return datetime.date(y, m0 + 1, d).timetuple().tm_yday - 1


class AnySat:
    """AnySat base through torch.hub, dense output pooled to 4-px windows. The dense vector per pixel is two halves:
    the contextual patch token (first D) and the local sub-patch embedding (second D). On a single-date Sentinel-2
    input at 40 m patches the contextual half is dominated by position within the tile (it barely moves when the
    content shifts by a window), the local half moves exactly with the content; so the local half is the space
    of the preregistered P2 and the contextual half a secondary space. Bands are standardised per testbed (mean/std
    over the tiles passed to fit). The layout (rows, cols) and the content locality are checked on the first tile
    of fit with a one-window roll, recorded in the summary; any failure leaves the AnySat signals absent."""

    def __init__(self, summary):
        self.model, self.mu, self.sd, self.summary = None, None, None, summary
        if "anysat" in summary and not summary["anysat"].get("loaded"):
            return                                   # a failed load is not retried
        try:
            self.model = torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False).to(DEV).eval()
            summary["anysat"] = {"loaded": True, "model": "base", "patch_m": AS_PATCH_M, "bands": list(AS_BANDS), "k": K,
                                 "output": "dense, pooled to 4-px windows; local half = P2 space, contextual half = secondary"}
        except Exception as ex:  # noqa: BLE001
            summary["anysat"] = {"loaded": False, "error": repr(ex), "traceback": traceback.format_exc()}
            print(f"AnySat not available: {ex!r}", flush=True)

    def fit(self, tiles12, doy):
        x = tiles12[:, AS_IDX].astype(np.float64)
        self.mu, self.sd = x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3)) + 1e-6
        self._check_layout(tiles12[:1], doy)

    @torch.no_grad()
    def _dense(self, x10, doy):
        """(B, 10, H, W) standardised -> (B, H/4, W/4, 2D) pooled dense features, rows first."""
        xb = torch.tensor(x10, dtype=torch.float32, device=DEV)[:, None]                     # (B, 1, 10, H, W)
        dates = torch.full((xb.shape[0], 1), int(doy), dtype=torch.long, device=DEV)
        f = self.model({"s2": xb, "s2_dates": dates}, patch_size=AS_PATCH_M, output="dense", output_modality="s2")
        f = f.float().cpu().numpy()                                                          # (B, H, W, 2D)
        B, H, W, C = f.shape
        assert (H, W) == x10.shape[-2:], f"dense output {f.shape} does not match the input {x10.shape}"
        return f.reshape(B, H // hb.PATCH, hb.PATCH, W // hb.PATCH, hb.PATCH, C).mean(axis=(2, 4))

    def _standardise(self, tiles12):
        return (tiles12[:, AS_IDX].astype(np.float32) - self.mu[None, :, None, None]) / self.sd[None, :, None, None]

    def _check_layout(self, tile12, doy):
        """Roll the tile by one window along W and along H: the local half must roll along output axis 1 and 0."""
        x = self._standardise(tile12)
        f = self._dense(x, doy)[0]
        fw = self._dense(np.roll(x, hb.PATCH, axis=-1), doy)[0]
        fh = self._dense(np.roll(x, hb.PATCH, axis=-2), doy)[0]
        D = f.shape[-1] // 2

        def corr(a, b):
            a = a - a.reshape(-1, a.shape[-1]).mean(0)
            b = b - b.reshape(-1, b.shape[-1]).mean(0)
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        loc, ctx = slice(D, 2 * D), slice(0, D)
        diag = {"local_W_roll_axis1": corr(f[..., loc], np.roll(fw[..., loc], -1, axis=1)), "local_W_roll_axis0": corr(f[..., loc], np.roll(fw[..., loc], -1, axis=0)),
                "local_H_roll_axis0": corr(f[..., loc], np.roll(fh[..., loc], -1, axis=0)), "local_unrolled": corr(f[..., loc], fw[..., loc]),
                "context_W_roll_axis1": corr(f[..., ctx], np.roll(fw[..., ctx], -1, axis=1)), "context_unrolled": corr(f[..., ctx], fw[..., ctx])}
        self.summary["anysat"].setdefault("layout_checks", []).append(diag)
        if not (diag["local_W_roll_axis1"] > 0.9 and diag["local_H_roll_axis0"] > 0.9 and diag["local_W_roll_axis1"] > diag["local_W_roll_axis0"]):
            raise RuntimeError(f"AnySat layout check failed: {diag}")

    def features(self, tiles12, doy, batch=8):
        """(N, 12, H, W) DN in S2_BANDS order -> (local (N, G, G, D), context (N, G, G, D))."""
        if self.model is None:
            return None, None
        x = self._standardise(tiles12)
        outs = [self._dense(x[i:i + batch], doy) for i in range(0, len(x), batch)]
        f = np.concatenate(outs)
        D = f.shape[-1] // 2
        return f[..., D:], f[..., :D]


_ANYSAT = None


def get_anysat(summary):
    """One AnySat instance per run (the load is recorded once in the summary)."""
    global _ANYSAT
    if _ANYSAT is None:
        _ANYSAT = AnySat(summary)
    return _ANYSAT


def weighted_combo(conf, primary, mask=None):
    out = np.full(np.shape(conf), np.nan)
    m = np.ones(np.shape(conf), bool) if mask is None else mask
    if m.any():
        out[m] = 0.75 * midrank_pct(np.asarray(conf)[m]) + 0.25 * midrank_pct(np.asarray(primary)[m])
    return out


def capture_tests(cap_per_unit, pairs):
    """Per-unit capture wins/losses/ties and exact sign tests for (a, b) pairs; one-sided only for the preregistered
    pairs at the preregistered budget, two-sided everywhere else."""
    out = {}
    for a, b in pairs:
        out[f"{a} vs {b}"] = {}
        for bud in BUDGETS:
            g = np.array([c[a][bud] - c[b][bud] for c in cap_per_unit.values()])
            w, l, t = wins_losses_ties(g)
            one = bud == PREREG_BUDGET and (a, b) in PREREG_ONE_SIDED
            out[f"{a} vs {b}"][str(bud)] = {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if one else "two-sided"),
                                             "one_sided": one, "median_gain": float(np.median(g)) if len(g) else None, "n_units": int(len(g))}
    return out


def strata(sigs, err, ok, score_name, conf_name=CONF):
    """Error rate among windows whose score is in the top 20% of the stratum against the rest, within confidence
    quintiles and within boundary and NDWI-ambiguity strata (pooled over valid windows)."""
    s = np.asarray(sigs[score_name], dtype=np.float64)[ok]
    if not np.isfinite(s).all():
        return {"skipped": "score has NaN"}
    e = np.asarray(err)[ok] > 0.5
    conf = np.asarray(sigs[conf_name], dtype=np.float64)[ok]
    bnd = np.asarray(sigs[BOUND], dtype=np.float64)[ok] > 0
    amb = np.asarray(sigs[CTRL_LVL], dtype=np.float64)[ok] > -0.1
    res = {}

    def cell(m, label):
        """Top fifth of the stratum by score, ties included (the score takes k + 1 values), against the rest."""
        if m.sum() < 50:
            res[label] = None
            return
        cut = np.quantile(s[m], 0.8)
        hi, lo = m & (s >= cut), m & (s < cut)
        res[label] = {"n": int(m.sum()), "threshold": float(cut), "n_top": int(hi.sum()), "n_rest": int(lo.sum()),
                      "error_rate_top": float(e[hi].mean()) if hi.sum() >= 20 else None,
                      "error_rate_rest": float(e[lo].mean()) if lo.sum() >= 20 else None}
    q = np.quantile(conf, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(conf, q)
    for i in range(5):
        cell(bins == i, f"confidence quintile {i + 1} (1 = most confident)")
    cell(bnd, "boundary windows"); cell(~bnd, "interior windows")
    cell(amb, "NDWI-ambiguous windows"); cell(~amb, "NDWI-clear windows")
    return res


# ----------------------------------------------------------------------------- part A
def scene_inputs(ctx, model, name):
    """Image, features and head prediction for any rule scene, regardless of the scoring threshold (bank material)."""
    G, size, feats, scenes, head = ctx["G"], ctx["size"], ctx["feats"], ctx["scenes"], ctx["hb"]
    img, date = scenes[f"{name}_img"], scenes[f"{name}_date"]
    if f"{name}_base0" not in feats:
        x, ts = hb.scene_tensor(img, date, size)
        feats[f"{name}_base0"] = hb.embed_pooled(model, x, ts)
    f0 = np.asarray(feats[f"{name}_base0"], dtype=np.float32)
    p = hb.predict_head(torch.tensor(f0), *head)
    return {"img": img, "date": date, "feats0": f0, "p": np.asarray(p)}


def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    size, G = ctx["size"], ctx["G"]
    anysat = get_anysat(summary)
    names = list(ctx["names"])
    inputs, units = {}, {}
    for name in names:
        try:
            inputs[name] = scene_inputs(ctx, model, name)
            u = hb.scene_unit(ctx, model, name, args, summary)
            if u is not None:
                units[name] = u
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
    names = [n for n in names if n in inputs]
    if not units:
        return
    imgs = np.stack([inputs[n]["img"][:, :size, :size] for n in names])
    emb = {"oe": {n: inputs[n]["feats0"].reshape(G * G, -1) for n in names},
           "pix": {n: pixel_embedding(inputs[n]["img"], size).reshape(G * G, 14) for n in names}}
    ref_pix = np.concatenate(list(emb["pix"].values()))
    emb["pix"] = {n: standardise(v, ref_pix) for n, v in emb["pix"].items()}
    if anysat.model is not None:
        try:
            anysat.fit(imgs, scene_doy(inputs[names[0]]["date"]))
            e_loc, e_ctx = {}, {}
            for i, n in enumerate(names):
                f_loc, f_ctx = anysat.features(imgs[i:i + 1], scene_doy(inputs[n]["date"]))
                e_loc[n], e_ctx[n] = f_loc[0].reshape(G * G, -1), f_ctx[0].reshape(G * G, -1)
            emb["as"], emb["asx"] = e_loc, e_ctx                     # published only once every scene succeeded
            summary["anysat"]["scene_feature_dim"] = int(next(iter(e_loc.values())).shape[-1])
        except Exception as ex:  # noqa: BLE001
            summary["anysat"]["scene_error"] = repr(ex)
            summary["anysat"]["scene_traceback"] = traceback.format_exc()
            print(f"AnySat scene features failed: {ex!r}", flush=True)
    have_as = "as" in emb
    prim, combo = (PRIMARY, COMBO) if have_as else (C_OE, COMBO_OE)
    active = [k for k in NEW_NAMES if (have_as or k not in (C_AS, H_AS, C_ASX, COMBO)) and (not have_as or k != COMBO_OE)]
    summary["part_a"].update({"signals_scored": active, "primary_used": prim, "bank": "windows of the rule scenes on other rivers (all rule scenes, scored or not)",
                              "prereg_combination": "void: AnySat not available" if not have_as else COMBO})
    pred = {n: (inputs[n]["p"] > 0.5).astype(np.float64).reshape(-1) for n in names}
    per, rho_per, cap = {}, {}, {}
    for n in sorted(units):
        try:
            u = units[n]
            river = hb.RIVER.get(n, n)
            bank = [m for m in names if hb.RIVER.get(m, m) != river]
            if not bank:
                summary["part_a"]["skipped"].append({"scene": n, "reason": "no other-river bank"})
                continue
            bank_pred = np.concatenate([pred[m] for m in bank])
            sigs = hb.base_signals_a(u, ctx)
            for key, (cname, hname) in {"oe": (C_OE, H_OE), "pix": (C_PIX, None), "as": (C_AS, H_AS), "asx": (C_ASX, None)}.items():
                if key in emb:
                    c, h = neighbour_scores(emb[key][n], np.concatenate([emb[key][m] for m in bank]), bank_pred, pred[n])
                    sigs[cname] = c.reshape(G, G)
                    if hname:
                        sigs[hname] = h.reshape(G, G)
            sigs[combo] = hb.combination(sigs, prim)
            sigs[WCOMBO] = weighted_combo(sigs[CONF], sigs[prim])
            val = hb.score_scene(u, sigs, rows, per, rho_per, active, extra_cols={"bank_scenes": len(bank), "primary_used": prim})
            cap[n] = {k: capture_at_budget_expected(np.asarray(sigs[k]), u["err"], BUDGETS) for k in (CONF, C_OE, C_PIX, C_AS, combo) if k in sigs}
            for tag, k in (("oe", C_OE), ("pix", C_PIX), ("as", C_AS), ("asx", C_ASX)):
                if k in sigs:
                    cache[f"{n}_{tag}"] = np.asarray(sigs[k], dtype=np.float32)
            print(f"{n}: {u['n_err']} errors, bank {len(bank)} scenes, conf {val[CONF]:.4f} tile {val[TILE]:.4f} C_oe {val[C_OE]:.4f} "
                  f"C_pix {val[C_PIX]:.4f}" + (f" C_as {val[C_AS]:.4f}" if C_AS in val else "") + f" U+ {val[combo]:.4f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": n, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{n}: FAILED {ex!r}", flush=True)
    hb.finish_part_a(summary, per, rho_per, active, prim, combo)
    # river tests: E-AURC gains over confidence, rebuilt by scene name from the CSV rows; one-sided only for P1/P2
    by_scene = {}
    for r in rows:
        if r.get("part") == "A" and r.get("eaurc") not in ("", None):
            by_scene.setdefault(r["signal"], {})[r["unit"]] = float(r["eaurc"])

    def river(gains, one_sided):
        return stats_lib.clustered_sign_test(gains, hb.RIVER, aggregate="mean", alternative="greater" if one_sided else "two-sided")

    rt = {}
    for k in (C_OE, C_PIX, C_AS, combo):
        if k in by_scene:
            common = [n for n in by_scene[k] if n in by_scene.get(CONF, {})]
            rt[k] = river({n: by_scene[CONF][n] - by_scene[k][n] for n in common}, (k, CONF) in PREREG_ONE_SIDED)
    if C_AS in by_scene and C_OE in by_scene:
        common = [n for n in by_scene[C_AS] if n in by_scene[C_OE]]
        rt[f"{C_AS} vs {C_OE}"] = river({n: by_scene[C_OE][n] - by_scene[C_AS][n] for n in common}, True)
    summary["part_a"]["river_tests_eaurc_vs_confidence"] = rt
    if cap:
        keys = [k for k in (C_OE, C_PIX, C_AS, combo) if all(k in c for c in cap.values())]
        summary["part_a"]["capture_river_tests_vs_confidence"] = {
            k: {str(b): river({n: cap[n][k][b] - cap[n][CONF][b] for n in cap}, (k, CONF) in PREREG_ONE_SIDED and b == PREREG_BUDGET) for b in BUDGETS} for k in keys}
        summary["part_a"]["median_capture"] = {k: {str(b): float(np.median([cap[n][k][b] for n in cap])) for b in BUDGETS} for k in [CONF] + keys}
        for k in keys:
            r = summary["part_a"]["capture_river_tests_vs_confidence"][k]
            print(f"  A capture rivers vs confidence, {k}: " + ", ".join(f"{b}: {r[str(b)]['w']}/{r[str(b)]['l']} p={r[str(b)]['p']:.3f}" for b in BUDGETS), flush=True)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp39", extra_keys=("test_base0",))
    if ctx is None:
        return
    N, G, ok, err, CROP = ctx["N"], ctx["G"], ctx["ok"], ctx["err"], ctx["CROP"]
    sigs = hb.base_signals_b(ctx)
    q_feats = ctx["ev_feats"].reshape(N * G * G, -1)
    q_pred = (ctx["p"] > 0.5).astype(np.float64).reshape(-1)
    # the bank: the test split as exp18 sampled it (800 tiles, seed 1), never used to train the head; valid tiles in smoke
    if args.smoke:
        bank_feats, bank_s2, bank_name = np.asarray(ctx["z"]["tr_base"], dtype=np.float32), ctx["tr_s2"], "valid tiles (smoke only)"
    else:
        if "test_base0" not in ctx["z"]:
            raise RuntimeError("exp18_feats.npz has no test_base0: the test-split bank is required outside smoke mode")
        bank_feats = np.asarray(ctx["z"]["test_base0"], dtype=np.float32)
        floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
        bank_s2, _ = hb.load_floods_split(os.path.join(floods_dir, "flood_test_data.pt"), n=exp18.N_TEST_TILES, seed=1)
        bank_name = "Sen1Floods11 test split, exp18's 800-tile sample (seed 1)"
    if len(bank_s2) != len(bank_feats):
        raise RuntimeError(f"bank images ({len(bank_s2)}) and cached bank features ({len(bank_feats)}) differ")
    nb = len(bank_feats)
    bank_pred = (exp18.head_prob_logit(bank_feats, *ctx["hb"])[0] > 0.5).astype(np.float64).reshape(-1)
    summary["part_b"]["bank"] = {"name": bank_name, "tiles": int(nb), "windows": int(bank_pred.size), "water_share_of_predictions": float(bank_pred.mean())}
    print(f"bank: {bank_name}, {nb} tiles, {bank_pred.size} windows", flush=True)
    # OlmoEarth space
    c, h = neighbour_scores(q_feats, bank_feats.reshape(-1, bank_feats.shape[-1]), bank_pred, q_pred)
    sigs[C_OE], sigs[H_OE] = c.reshape(N, G, G), h.reshape(N, G, G)
    # pixel-statistics space
    q_pix = np.stack([pixel_embedding(t, CROP) for t in ctx["bo_s2"]]).reshape(N * G * G, 14)
    b_pix = np.stack([pixel_embedding(t, CROP) for t in bank_s2]).reshape(-1, 14)
    sigs[C_PIX] = neighbour_scores(standardise(q_pix, b_pix), standardise(b_pix, b_pix), bank_pred, q_pred)[0].reshape(N, G, G)
    # AnySat space (absent from the scored signals when unavailable)
    anysat = get_anysat(summary)
    have_as = False
    if anysat.model is not None:
        try:
            crop = lambda t: t[:, :, :CROP, :CROP]  # noqa: E731
            anysat.fit(np.concatenate([crop(ctx["bo_s2"]), crop(bank_s2)]), FLOOD_DOY)
            q_loc, q_ctx = anysat.features(crop(ctx["bo_s2"]), FLOOD_DOY)
            b_loc, b_ctx = anysat.features(crop(bank_s2), FLOOD_DOY)
            Dl = q_loc.shape[-1]
            c, h = neighbour_scores(q_loc.reshape(N * G * G, Dl), b_loc.reshape(-1, Dl), bank_pred, q_pred)
            cx = neighbour_scores(q_ctx.reshape(N * G * G, Dl), b_ctx.reshape(-1, Dl), bank_pred, q_pred)[0]
            sigs[C_AS], sigs[H_AS], sigs[C_ASX] = c.reshape(N, G, G), h.reshape(N, G, G), cx.reshape(N, G, G)   # published together
            summary["anysat"]["flood_feature_dim"] = int(Dl)
            have_as = True
        except Exception as ex:  # noqa: BLE001
            summary["anysat"]["flood_error"] = repr(ex)
            summary["anysat"]["flood_traceback"] = traceback.format_exc()
            print(f"AnySat flood features failed: {ex!r}", flush=True)
    prim, combo = (PRIMARY, COMBO) if have_as else (C_OE, COMBO_OE)
    summary["part_b"]["primary_used"] = prim
    summary["part_b"]["prereg_combination"] = COMBO if have_as else "void: AnySat not available"
    cmb, wcmb = np.full((N, G, G), np.nan), np.full((N, G, G), np.nan)
    for t in range(N):
        m = ok[t]
        if m.any():
            cmb[t][m] = (midrank_pct(sigs[CONF][t][m]) + midrank_pct(sigs[prim][t][m])) / 2
            wcmb[t] = weighted_combo(sigs[CONF][t], sigs[prim][t], m)
    sigs[combo], sigs[WCOMBO] = cmb, wcmb
    new_names = [k for k in NEW_NAMES if k in sigs]
    for k in new_names:                                  # every scored signal must be finite on every valid window
        bad = ~np.isfinite(np.asarray(sigs[k], dtype=np.float64)[ok])
        if bad.any():
            raise RuntimeError(f"{k}: {int(bad.sum())} non-finite values on valid windows")
    for tag, k in (("oe", C_OE), ("pix", C_PIX), ("as", C_AS), ("asx", C_ASX)):
        if k in sigs:
            cache[f"bolivia_{tag}"] = np.asarray(sigs[k], dtype=np.float32)
    per_b, tiles = hb.finish_part_b(summary, rows, ctx, sigs, new_names, prim, combo)
    # capture at budgets: per tile (one-sided only for the preregistered pairs at 10%) and a pooled tile bootstrap
    keys = [k for k in (CONF, C_OE, C_PIX, C_AS, C_ASX, combo, WCOMBO, TILE, BOUND) if k in sigs]
    cap = {t: {k: capture_at_budget_expected(np.asarray(sigs[k][t])[ok[t]], err[t][ok[t]], BUDGETS) for k in keys} for t in tiles}
    pairs = [(C_OE, CONF), (C_PIX, CONF), (combo, CONF), (WCOMBO, CONF)] + ([(C_AS, CONF), (C_AS, C_OE), (C_ASX, CONF)] if have_as else [])
    summary["part_b"]["capture_tests"] = capture_tests(cap, pairs)
    ct = summary["part_b"]["capture_tests"]
    summary["part_b"]["prereg"] = {"P1": ct[f"{C_OE} vs {CONF}"][str(PREREG_BUDGET)],
                                   "P2": ct[f"{C_AS} vs {CONF}"][str(PREREG_BUDGET)] if have_as else "void: AnySat not available",
                                   "P2_vs_oe": ct[f"{C_AS} vs {C_OE}"][str(PREREG_BUDGET)] if have_as else "void: AnySat not available"}
    all_idx = [t for t in range(N) if ok[t].any()]
    pooled = {k: np.concatenate([np.asarray(sigs[k][t])[ok[t]] for t in all_idx]) for k in keys}
    pooled_err = np.concatenate([err[t][ok[t]] for t in all_idx])
    sizes = np.array([int(ok[t].sum()) for t in all_idx])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    rng = np.random.default_rng(1)
    gains = {pair: {bud: [] for bud in BUDGETS} for pair in pairs}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(all_idx), len(all_idx))
        sel = np.concatenate([np.arange(starts[i], starts[i] + sizes[i]) for i in pick])
        e = pooled_err[sel]
        if e.sum() == 0:
            continue
        cc = {k: capture_at_budget_expected(pooled[k][sel], e, BUDGETS) for k in keys}
        for a, b in pairs:
            for bud in BUDGETS:
                gains[(a, b)][bud].append(cc[a][bud] - cc[b][bud])
    full = {k: capture_at_budget_expected(pooled[k], pooled_err, BUDGETS) for k in keys}
    summary["part_b"]["pooled_capture"] = {k: {str(b): full[k][b] for b in BUDGETS} for k in keys}
    def boot(g):
        g = np.array(g)
        if len(g) == 0:
            return {"boot_lo": None, "boot_hi": None, "p_better": None, "n_boot": 0}
        return {"boot_lo": float(np.percentile(g, 2.5)), "boot_hi": float(np.percentile(g, 97.5)), "p_better": float((g > 0).mean()), "n_boot": int(len(g))}
    summary["part_b"]["pooled_capture_gain"] = {
        f"{a} vs {b}": {str(bud): {"gain": full[a][bud] - full[b][bud], **boot(gains[(a, b)][bud])} for bud in BUDGETS} for a, b in pairs}
    summary["part_b"]["strata"] = {k: strata(sigs, err, ok, k) for k in (C_OE, C_PIX) + ((C_AS, C_ASX) if have_as else ())}
    for a, b in pairs:
        pg = summary["part_b"]["pooled_capture_gain"][f"{a} vs {b}"]
        print(f"B {a} vs {b}: " + " | ".join(f"{bud}: {ct[f'{a} vs {b}'][str(bud)]['w']}/{ct[f'{a} vs {b}'][str(bud)]['l']}/{ct[f'{a} vs {b}'][str(bud)]['t']} "
                                           f"p={ct[f'{a} vs {b}'][str(bud)]['sign_p']:.2g} pooled {full[a][bud]:.3f} vs {full[b][bud]:.3f} "
                                           + (f"CI [{pg[str(bud)]['boot_lo']:+.3f}, {pg[str(bud)]['boot_hi']:+.3f}]" if pg[str(bud)]['boot_lo'] is not None else "CI n/a") for bud in BUDGETS), flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"k": K, "budgets": list(BUDGETS), "prereg_budget": PREREG_BUDGET, "n_boot": N_BOOT, "primary_score": PRIMARY, "combination": COMBO,
              "one_sided_pairs": [f"{a} vs {b}" for a, b in sorted(PREREG_ONE_SIDED)],
              "spaces": {"OlmoEarth": "pooled Base features (exp11 / exp18 caches)", "pixel": "per-window means of 12 log bands + NDWI mean, std, standardised",
                         "AnySat": f"torch.hub gastruc/anysat base, dense output pooled to 4-px windows (local half = P2 space, contextual half secondary), patch {AS_PATCH_M} m, bands {AS_BANDS}, standardised per testbed, one date, layout checked by a one-window roll"},
              "banks": {"scenes": "windows of every rule scene on other rivers", "bolivia": "Sen1Floods11 test split, exp18's 800-tile sample (seed 1), head never trained on it"},
              "prereg": {"P1": f"{C_OE} beats confidence: capture at 10% on Bolivia, per-tile one-sided sign test + tile bootstrap; river vote on scenes alongside",
                         "P2": f"{C_AS} beats confidence and beats {C_OE}, same tests",
                         "combination": "U+ = midrank mean of confidence and the AnySat-space contradiction; void if AnySat is unavailable (OlmoEarth combination is then secondary)",
                         "falsification": "no hand-label gain, or a gain the pixel-statistics ablation reproduces"}}
    hb.run("exp39", "exp39 neighbourhood contradiction in three embedding spaces", config, part_a, part_b, args, "exp39_neighbour_contradiction")


if __name__ == "__main__":
    main()

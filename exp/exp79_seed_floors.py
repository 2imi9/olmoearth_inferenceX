"""exp79: is the record bigger than its own seed noise? Ten probe seeds under every suite number, and the
per-unit export exp78 needs for the fifteen other encoders. Preregistered in docs/plan/seed_floors.md.

    python exp/exp79_seed_floors.py --stage export --model <encoder> [--seeds 10]     # cluster, one GPU job per encoder
    python exp/exp79_seed_floors.py --stage grade                                     # local, numpy: gate G, P1, P2, P3
    python exp/exp79_seed_floors.py --stage estimate [--units DIR]                    # local, numpy: P4, P5 from the export
    python exp/exp79_seed_floors.py --stage smoke                                     # the grading logic on synthetic inputs

The export stage loads each task's embeddings once and fits the suite's probe at every seed on the same tensors
(scripts/suite_regression.load_task / fit_task), so the cost per extra seed is a probe fit and nothing else.
Scoring is exp70's, unchanged. The seed-0 fit also writes the per-unit quantities in exp78's format; those files
are about 45 MB per encoder and are NOT committed — the cluster home directory keeps them and this script's
per-encoder JSON carries their SHA-256, so a fetched copy is checkable.

Outputs. exp/out/exp79_seeds/<encoder>.json per export job (committed); exp/out/exp79_units/<encoder>/<task>.npz
(not committed); exp/out/exp79_summary.json from the two local stages (committed). Every verdict is a pure
function of dictionaries so tests/test_exp79_grading.py can exercise it before the job runs.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from oe_inferencex import stats                    # noqa: E402
import exp70_task_suite as e70                     # noqa: E402
import exp74_suite_encoders as e74                 # noqa: E402
import exp78_error_rate_estimation as e78          # noqa: E402
import headroom_by_encoder as hbe                  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SEEDS_DIR = os.path.join(OUT, "exp79_seeds")
UNITS_DIR = os.path.join(OUT, "exp79_units")
N_SEEDS = 10
# the preregistered bars, docs/plan/seed_floors.md
GATE_TOL = 1e-4               # G: seed 0 reproduces the recorded accuracy to the four decimals the record states
P1_SHARE, P1_P = 0.75, 0.05   # exp74's bar, applied per seed
P2_MIN_ROBUST = 20            # P2: OlmoEarth Base tasks won under all seeds
P3_RHO = 0.9                  # P3: rank correlation with the seed-0 encoder ordering
P4_MIN_ENCODERS, P4_MAX_EXCEPTIONS, P4_MIN_DEFF, P4_COVER = 13, 2, 2.0, 0.85
P5_COVER, P5_EXACT_TOL = 0.93, 0.01


# ----------------------------------------------------------------------------- stage: export (cluster)
def recorded_accuracy(model, task):
    """The test accuracy the record holds for (model, task): exp70 for OlmoEarth Base, exp74 for the rest."""
    if model == "olmoearth_base":
        t = json.load(open(os.path.join(OUT, "exp70_summary.json")))["results"]["tasks"]
    else:
        t = json.load(open(os.path.join(OUT, "exp74_summary.json")))["results"]["tasks"].get(model, {})
    return t.get(task, {}).get("test_accuracy")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_export(args):
    import suite_regression as sr
    import exp54_multiclass_embeddings as e54
    cache = os.environ.get("HF_HOME")
    model = args.model
    lrs = sr.set_probe_lrs(model, cache)
    e70.MODEL = model
    os.makedirs(SEEDS_DIR, exist_ok=True)
    udir = os.path.join(UNITS_DIR, model)
    os.makedirs(udir, exist_ok=True)
    out = {"experiment": "exp79", "model": model, "n_seeds": args.seeds, "probe_lrs": lrs, "tasks": {}, "absent": []}
    for task in e70.TASKS_CLS + e70.TASKS_SEG:
        if args.only and task not in args.only:
            continue
        t0 = time.time()
        try:
            L = sr.load_task(model, task, cache)
        except Exception as exc:                                   # a task the model does not carry (exp74)
            out["absent"].append({"task": task, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
            print(f"  {task:44s} absent ({type(exc).__name__})", flush=True)
            continue
        seeds, ok0 = [], None
        for seed in range(args.seeds):
            rec = sr.fit_task(L, seed, units=(seed == 0), expect_ok=ok0)
            if seed == 0:
                u = rec.pop("units")
                ok0 = u["ok"]
                path = os.path.join(udir, f"{task}.npz")
                np.savez_compressed(path, margin=u["margin"], p1=u["p1"], dec=u["dec"], y=u["y"], err=u["err"],
                                    ok_packed=np.packbits(u["ok"]), ok_len=np.array([u["ok"].size]),
                                    grid=u["grid"], n_classes=np.array([u["n_classes"]]),
                                    patch_px=np.array([u["patch_px"]]), family=np.array([u["family"]]))
                units_meta = {"path": os.path.relpath(path, ROOT), "sha256": sha256(path), "bytes": os.path.getsize(path)}
            seeds.append(rec)
        del L
        acc0, rec0 = seeds[0]["test_accuracy"], recorded_accuracy(model, task)
        out["tasks"][task] = {
            "family": seeds[0]["family"], "source": seeds[0]["source"], "n_units": seeds[0]["n_units"],
            "recorded_accuracy": rec0, "accuracy_minus_recorded": None if rec0 is None else acc0 - rec0,
            "units": units_meta, "seconds": round(time.time() - t0, 1), "seeds": seeds}
        leads = [s["margin_lead"] for s in seeds]
        print(f"  {task:44s} {seeds[0]['family'][:3]} n={seeds[0]['n_units']:8d} acc0-rec "
              f"{(acc0 - rec0) if rec0 is not None else float('nan'):+.1e} lead {np.mean(leads):+.4f}"
              f"+-{np.std(leads):.4f} wins {sum(1 for v in leads if v > 0)}/{len(leads)} {out['tasks'][task]['seconds']:.0f}s",
              flush=True)
        with open(os.path.join(SEEDS_DIR, f"{model}.json"), "w") as f:      # after every task, so a killed job keeps its work
            json.dump(out, f, indent=1, default=float)
    print(f"wrote {SEEDS_DIR}/{model}.json: {len(out['tasks'])} tasks x {args.seeds} seeds, {len(out['absent'])} absent")
    return 0


# ----------------------------------------------------------------------------- stage: grade (local)
def load_seeds():
    """{encoder: export json} for every encoder that has run."""
    return {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(SEEDS_DIR, "*.json")))}


def gate(per_encoder):
    """G: seed 0 reproduces the recorded accuracy on every task, per encoder; the failing ones are named."""
    g = {}
    for enc, d in per_encoder.items():
        diffs = {t: v["accuracy_minus_recorded"] for t, v in d["tasks"].items() if v["accuracy_minus_recorded"] is not None}
        bad = sorted(t for t, x in diffs.items() if abs(x) >= GATE_TOL)
        g[enc] = {"holds": not bad and len(diffs) == len(d["tasks"]), "n_compared": len(diffs), "n_tasks": len(d["tasks"]),
                  "max_abs_diff": max((abs(x) for x in diffs.values()), default=float("nan")), "failing": bad}
    return {"holds": bool(g) and all(v["holds"] for v in g.values()), "per_encoder": g,
            "encoders_failing": sorted(e for e, v in g.items() if not v["holds"])}


def per_seed_results(d, seed):
    """One encoder's exp74-shaped results dict at one seed."""
    return {t: v["seeds"][seed] for t, v in d["tasks"].items() if seed < len(v["seeds"])}


def grade_p1(per_encoder, n_seeds=N_SEEDS):
    """P1: exp74's bar holds on every encoder under every seed."""
    table = {}
    for enc, d in per_encoder.items():
        table[enc] = []
        for s in range(n_seeds):
            v = e74.encoder_verdict(per_seed_results(d, s))
            table[enc].append({"seed": s, "wins": v["wins"], "scored": v["scored"], "share": v["share"], "p": v["p"],
                               "holds": v["share"] >= P1_SHARE and v["p"] < P1_P})
    failing = sorted((enc, r["seed"]) for enc, rows in table.items() for r in rows if not r["holds"])
    return {"holds": bool(table) and not failing, "n_encoders": len(table), "n_seeds": n_seeds,
            "failing_encoder_seed": failing,
            "min_share": {enc: min(r["share"] for r in rows) for enc, rows in table.items()},
            "min_wins": {enc: min(r["wins"] for r in rows) for enc, rows in table.items()}}


def grade_p2(per_encoder, model="olmoearth_base", n_seeds=N_SEEDS):
    """P2: on OlmoEarth Base, the margin wins under every seed on at least P2_MIN_ROBUST tasks; flippers are named."""
    d = per_encoder.get(model)
    if d is None:
        return {"holds": None, "reason": f"{model} has not run"}
    rows = {}
    for t, v in d["tasks"].items():
        leads = np.array([s["margin_lead"] for s in v["seeds"][:n_seeds]], float)
        rows[t] = {"mean": float(leads.mean()), "sd": float(leads.std(ddof=1)) if leads.size > 1 else 0.0,
                   "min": float(leads.min()), "wins": int((leads > 0).sum()), "of": int(leads.size),
                   "n_units": v["n_units"], "robust": bool((leads > 0).all())}
    robust = sorted(t for t, r in rows.items() if r["robust"])
    flips = sorted(t for t, r in rows.items() if not r["robust"])
    return {"holds": len(robust) >= P2_MIN_ROBUST and len(rows) > 0, "n_robust": len(robust), "of": len(rows),
            "flips": {t: rows[t] for t in flips}, "per_task": rows}


def headroom_per_seed(per_encoder, seed):
    """{encoder: median headroom at this seed}, the statistic of headroom_by_encoder.json recomputed."""
    out = {}
    for enc, d in per_encoder.items():
        hs = [h for t, v in d["tasks"].items() if seed < len(v["seeds"]) and (h := hbe.headroom(v["seeds"][seed])) is not None]
        if hs:
            out[enc] = float(np.median(hs))
    return out


def grade_p3(per_encoder, n_seeds=N_SEEDS, top_k=3, bottom_k=2):
    """P3: rank correlation with the seed-0 ordering >= P3_RHO under every seed, and the seed-0 top-k and bottom-k
    sets contain the best and worst under every seed."""
    h0 = headroom_per_seed(per_encoder, 0)
    if len(h0) < 3:
        return {"holds": None, "reason": f"only {len(h0)} encoders have run"}
    order0 = sorted(h0, key=lambda e: -h0[e])
    top, bottom = set(order0[:top_k]), set(order0[-bottom_k:])
    rows = []
    for s in range(n_seeds):
        hs = headroom_per_seed(per_encoder, s)
        common = [e for e in order0 if e in hs]
        rho = float(stats.spearman(np.array([h0[e] for e in common]), np.array([hs[e] for e in common])))
        best, worst = max(common, key=lambda e: hs[e]), min(common, key=lambda e: hs[e])
        rows.append({"seed": s, "rho": rho, "best": best, "worst": worst,
                     "best_in_top": best in top, "worst_in_bottom": worst in bottom})
    ok = all(r["rho"] >= P3_RHO and r["best_in_top"] and r["worst_in_bottom"] for r in rows)
    return {"holds": ok, "seed0_order": order0, "top_set": sorted(top), "bottom_set": sorted(bottom),
            "min_rho": min(r["rho"] for r in rows), "per_seed": rows,
            "headroom_seed0": h0}


def cmd_grade(args):
    per = load_seeds()
    if not per:
        print(f"nothing under {SEEDS_DIR}"); return 1
    G = gate(per)
    summary = load_summary()
    summary.update({"experiment": "exp79", "config": {"n_seeds": N_SEEDS, "gate_tol": GATE_TOL,
                                                       "encoders": sorted(per), "n_encoders": len(per)},
                    "gate": G, "prereg_A": {"P1_headline_survives_reseeding": grade_p1(per),
                                            "P2_which_wins_are_noise": grade_p2(per),
                                            "P3_encoder_ordering_stable_in_sets": grade_p3(per)}})
    write_summary(summary)
    print(json.dumps({"gate": {k: G[k] for k in ("holds", "encoders_failing")},
                      **{k: {kk: v[kk] for kk in v if kk in ("holds", "n_robust", "of", "failing_encoder_seed", "min_rho", "flips")}
                         for k, v in summary["prereg_A"].items()}}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------- stage: estimate (local, B)
def exact_srs_coverage(N, K, B):
    """The exact coverage of exp78's Wilson-with-FPC interval for a simple random sample of B from a population of
    N units holding K errors: the hypergeometric probability of each error count k, summed over the k whose
    interval contains K/N. This is what the estimator can achieve at (N, K, B); Monte Carlo coverage is judged
    against it, not against a round number, so Wilson's discreteness is not mistaken for a bug."""
    theta = K / N
    lc = lambda n, r: math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
    tot = 0.0
    for k in range(max(0, B - (N - K)), min(B, K) + 1):
        lo, hi = e78.wilson(k, B, N)
        if lo <= theta <= hi:
            tot += math.exp(lc(K, k) + lc(N - K, B - k) - lc(N, B))
    return tot


def estimate_encoder(enc, units_dir, budget=e78.HEADLINE):
    """exp78's designs and estimators on one encoder's seed-0 export, segmentation tasks only."""
    rows = {}
    for t in e70.TASKS_SEG:
        path = os.path.join(units_dir, enc, f"{t}.npz")
        if not os.path.exists(path):
            continue
        u = e78.load_units(t, os.path.join(units_dir, enc))
        err, N = u["err"], u["err"].size
        if N <= 3 * budget:
            continue
        theta, K = float(err.mean()), int(err.sum())
        rng = np.random.default_rng(0)
        d = e78.run_designs(err, u["margin"], u["p1"], u["tile"], budget, rng, theta)
        rows[t] = {"n_units": N, "error_rate": theta, "design_effect": e78.design_effect(err, u["tile"]),
                   "srs_coverage": d["D1/E1"]["coverage"], "srs_half_width": d["D1/E1"]["half_width"],
                   "naive_coverage": d["D4/E2_naive"]["coverage"], "cluster_coverage": d["D4/E1_cluster"]["coverage"],
                   "exact_srs_coverage": exact_srs_coverage(N, K, budget) if d["D1/E1"]["coverage"] < P5_COVER else None,
                   "designs": d}
        print(f"  {enc:18s} {t:28s} theta {theta:.3f} deff {rows[t]['design_effect']:5.2f} "
              f"srs {rows[t]['srs_coverage']:.3f} naive {rows[t]['naive_coverage']:.3f}", flush=True)
    return rows


def grade_p4(per_encoder_rows):
    """P4: on >= P4_MIN_ENCODERS encoders, the naive interval under-covers on all but <= P4_MAX_EXCEPTIONS of the
    segmentation tasks it carries, with median design effect >= P4_MIN_DEFF."""
    table = {}
    for enc, rows in per_encoder_rows.items():
        if not rows:
            continue
        under = sorted(t for t, r in rows.items() if r["naive_coverage"] < P4_COVER)
        deffs = [r["design_effect"] for r in rows.values() if np.isfinite(r["design_effect"])]
        med = float(np.median(deffs)) if deffs else float("nan")
        table[enc] = {"carried": len(rows), "under_covering": len(under), "exceptions": sorted(set(rows) - set(under)),
                      "median_design_effect": med,
                      "holds": len(rows) - len(under) <= P4_MAX_EXCEPTIONS and med >= P4_MIN_DEFF}
    n_ok = sum(1 for v in table.values() if v["holds"])
    return {"holds": n_ok >= P4_MIN_ENCODERS, "n_encoders_holding": n_ok, "of": len(table), "per_encoder": table,
            "failing": sorted(e for e, v in table.items() if not v["holds"])}


def grade_p5(per_encoder_rows):
    """P5: SRS coverage >= P5_COVER on every cell, or within P5_EXACT_TOL of the exact Wilson coverage there."""
    cells, bad = [], []
    for enc, rows in per_encoder_rows.items():
        for t, r in rows.items():
            c, ex = r["srs_coverage"], r.get("exact_srs_coverage")
            ok = c >= P5_COVER or (ex is not None and c >= ex - P5_EXACT_TOL)
            cells.append({"encoder": enc, "task": t, "coverage": c, "exact": ex, "holds": ok})
            if not ok:
                bad.append((enc, t))
    return {"holds": bool(cells) and not bad, "n_cells": len(cells), "failing": bad,
            "below_bar_but_within_exact": [(c["encoder"], c["task"], c["coverage"], c["exact"])
                                           for c in cells if c["holds"] and c["coverage"] < P5_COVER],
            "min_coverage": min((c["coverage"] for c in cells), default=float("nan"))}


def cmd_estimate(args):
    units_dir = args.units or UNITS_DIR
    encs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(units_dir, "*")) if os.path.isdir(p))
    if not encs:
        print(f"no per-encoder export under {units_dir}"); return 1
    per = {enc: estimate_encoder(enc, units_dir) for enc in encs}
    summary = load_summary()
    summary["estimation"] = {"budget": e78.HEADLINE, "R": e78.R_DRAWS, "encoders": encs,
                             "per_encoder": {e: {t: {k: v for k, v in r.items() if k != "designs"} for t, r in rows.items()}
                                             for e, rows in per.items()}}
    summary["prereg_B"] = {"P4_naive_failure_belongs_to_the_task": grade_p4(per), "P5_estimator_honest_everywhere": grade_p5(per)}
    write_summary(summary)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "n_encoders_holding", "of", "failing", "n_cells", "min_coverage")}
                      for k, v in summary["prereg_B"].items()}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------- shared
def load_summary():
    p = os.path.join(OUT, "exp79_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write_summary(summary):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp79_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


def synthetic_export(encoders, n_seeds=3, n_tasks=6, flip=(), off_gate=(), seed=0):
    """A fake exp79_seeds record for the grading tests: `flip` = (encoder, task) pairs whose lead is negative on
    the last seed; `off_gate` = encoders whose seed-0 accuracy misses the recorded one."""
    rng = np.random.default_rng(seed)
    per = {}
    for i, enc in enumerate(encoders):
        tasks = {}
        for j in range(n_tasks):
            t = f"t{j}"
            n = 1000 * (j + 1)
            e = 0.1 + 0.05 * j
            seeds = []
            for s in range(n_seeds):
                lead = 0.05 - 0.002 * i + rng.normal(0, 0.001)
                if (enc, t) in flip and s == n_seeds - 1:
                    lead = -abs(lead)
                seeds.append({"margin_lead": lead, "source": f"s{j}", "test_accuracy": 1 - e, "error_rate": e,
                              "n_units": n, "seed": s, "family": "classification",
                              "signals": {"margin": {"excess_aurc": 0.02 + 0.01 * i + rng.normal(0, 0.0005)}}})
            tasks[t] = {"family": "classification", "source": f"s{j}", "n_units": n,
                        "recorded_accuracy": 1 - e, "accuracy_minus_recorded": (3e-4 if enc in off_gate else 1e-6),
                        "seeds": seeds}
        per[enc] = {"model": enc, "n_seeds": n_seeds, "tasks": tasks}
    return per


def cmd_smoke(args):
    encs = [f"enc{i}" for i in range(5)]
    per = synthetic_export(encs)
    assert gate(per)["holds"] and grade_p1(per, 3)["holds"]
    assert not gate(synthetic_export(encs, off_gate=("enc2",)))["holds"]
    p3 = grade_p3(per, 3)
    assert p3["holds"] and p3["min_rho"] > 0.99, p3
    N, K, B = 40, 9, 12
    ex = exact_srs_coverage(N, K, B)
    assert 0.8 < ex <= 1.0
    print(f"smoke OK: gate, P1, P3 on synthetic seeds; exact SRS coverage at (N={N}, K={K}, B={B}) = {ex:.4f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("export", "grade", "estimate", "smoke"), required=True)
    ap.add_argument("--model", default=e70.MODEL)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--units", default=None, help="estimate: directory holding <encoder>/<task>.npz (default exp/out/exp79_units)")
    args = ap.parse_args(argv)
    return {"export": cmd_export, "grade": cmd_grade, "estimate": cmd_estimate, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

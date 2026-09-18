"""Run exp70's protocol on any model directory of Ai2's published embedding suite, and score any candidate reading
beside the margin, on that model's own errors.

    HF_HOME=<cache> python scripts/suite_regression.py --model olmoearth_base
    HF_HOME=<cache> python scripts/suite_regression.py --model <hub dir> --signal mypkg.readings:my_reading

A candidate reading is a function f(probs, emb_test, emb_train, decisions) -> per-unit array, higher = more suspect,
on one task's test units: probs (N, C) from the fitted probe, emb_test (N, D) the test embeddings (per window on the
segmentation path), emb_train (M, D), decisions (N,). It is scored with exp70's statistics beside the margin, the
entropy and the two no-model controls, and reported by the tasks on which it beats the margin. The verdict on the
margin is exp74's per-encoder bar: it beats the best control on at least 75% of the scored tasks with a one-sided
sign test p < 0.05. Probes are Ai2's linear recipe at seed 0 with the model's own per-task learning rates, so the
errors scored are the errors the record scores. Nothing here changes the record: outputs go to
exp/out/suite_regression/<model>/ (gitignored), and a claim is registered only when a run is recorded by hand.
`--smoke` exercises the scoring and the verdict on synthetic arrays without torch.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp70_task_suite as e70                    # noqa: E402

OUT = os.path.join(ROOT, "exp", "out", "suite_regression")


def load_signal(spec):
    """'module:function' -> the function; the module is imported from the working directory or the path."""
    mod, fn = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod), fn)


def score_units(p, emb_te, emb_tr, y, C, signal=None, name="candidate"):
    """exp70's readings, controls and statistics on one task's units, plus the candidate if given."""
    import exp73_alternatives_suite as e73
    sig, dec = e70.readings_from_probs(p)
    err = (dec != np.asarray(y)).astype(np.float64)
    sig.update(e73.controls_chunked(np.asarray(emb_te, dtype=np.float32), np.asarray(emb_tr, dtype=np.float32), dec))
    if signal is not None:
        u = np.asarray(signal(np.asarray(p), np.asarray(emb_te), np.asarray(emb_tr), dec), dtype=np.float64)
        if u.shape != err.shape:
            raise ValueError(f"the candidate reading has shape {u.shape}, the units are {err.shape}")
        sig[name] = u
    sc = e70.score_task(sig, err, C)
    cname, cval = e70.best_control(sc)
    rec = {"n_units": int(len(err)), "n_classes": int(C), "test_accuracy": float(1 - err.mean()),
           "error_rate": float(err.mean()), "signals": sc, "best_control": cname,
           "margin_lead": float(cval - sc["margin"]["excess_aurc"])}
    if signal is not None:
        rec["candidate"] = {"name": name, "lead_over_control": float(cval - sc[name]["excess_aurc"]),
                            "vs_margin": float(sc["margin"]["excess_aurc"] - sc[name]["excess_aurc"])}
    return rec


def run_task(model, task, cache, signal, name):
    import torch
    import exp54_multiclass_embeddings as e54
    import exp73_alternatives_suite as e73
    import exp74_suite_encoders as e74
    import exp75_sensor_views as e75
    xtr, ytr = e54.load_theirs(model, task, "train", cache)
    xte, yte = e54.load_theirs(model, task, "test", cache)
    if task in e70.TASKS_CLS:
        C = int(max(int(ytr.max()), int(yte.max())) + 1)
        p = e73._train_cls_probe(xtr, ytr, C, 0)(xte)
        rec = score_units(p, xte.to(torch.float32).numpy(), xtr.to(torch.float32).numpy(), yte.numpy(), C, signal, name)
        rec["family"] = "classification"
    else:
        C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
        pp = int(round(ytr.shape[-1] / xtr.shape[1]))
        probe = e54.train_probe(xtr, ytr, pp, C, e54.TASK_LR.get(task, 0.1), seed=0)
        q = e75.seg_window_probs(probe, xte, yte, pp, C)
        ok = q["ok"]
        hw, ww = q["hw"]
        H, W = yte.shape[1], yte.shape[2]
        ete = e74.window_embeddings(xte.to(torch.float32).numpy(), hw, ww, H, W, e54.WIN).reshape(-1, xte.shape[-1])[ok]
        etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
        rec = score_units(q["P"][ok], ete, etr, q["y"][ok], C, signal, name)
        rec["family"], rec["patch_px"] = "segmentation", pp
    rec["source"] = e70.source_of(task)
    return rec


def cmd_run(args):
    import exp54_multiclass_embeddings as e54
    import exp74_suite_encoders as e74
    cache = os.environ.get("HF_HOME")
    signal = load_signal(args.signal) if args.signal else None
    name = args.name or (args.signal.rsplit(":", 1)[-1] if args.signal else "candidate")
    out_dir = os.path.join(OUT, args.model)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    e70.MODEL = args.model
    e54.TASK_LR = {t: float(lr) for t, lr in e74.probe_lrs(args.model, cache).items() if lr is not None}
    results, absent = {}, []
    for task in e70.TASKS_CLS + e70.TASKS_SEG:
        if args.only and task not in args.only:
            continue
        part = os.path.join(out_dir, "parts", f"{task}.json")
        if os.path.exists(part) and not args.redo:
            with open(part) as fh:
                r = json.load(fh)
        else:
            t0 = time.time()
            try:
                r = run_task(args.model, task, cache, signal, name)
            except Exception as exc:                                   # a task the model does not carry
                r = {"absent": True, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
            else:
                r["seconds"] = round(time.time() - t0, 1)
            with open(part, "w") as fh:
                json.dump(r, fh, default=float)
        if r.get("absent"):
            absent.append(task)
            print(f"  {task:44s} absent ({r['reason'][:70]})", flush=True)
            continue
        results[task] = r
        cand = f" {name} vs margin {r['candidate']['vs_margin']:+.4f}" if "candidate" in r else ""
        print(f"  {task:44s} {r['family'][:3]} n={r['n_units']:7d} acc {r['test_accuracy']:.3f} "
              f"margin lead {r['margin_lead']:+.4f} over {r['best_control'][4:]}{cand}", flush=True)
    summary = {"model": args.model, "verdict": e74.encoder_verdict(results) if results else {},
               "bar": "the margin beats the best no-model control on >= 75% of scored tasks, sign test p < 0.05 (exp74)",
               "absent": absent, "tasks": results}
    if signal is not None and results:
        wins = sorted(t for t, r in results.items() if r["candidate"]["vs_margin"] > 0)
        summary["candidate"] = {"name": name, "beats_margin_on": wins, "of": len(results),
                                "beats_control_on": sum(1 for r in results.values() if r["candidate"]["lead_over_control"] > 0),
                                "note": "a candidate enters the record only through a preregistered experiment; this is the screen"}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    v = summary["verdict"]
    if v:
        print(f"margin: {v['wins']}/{v['scored']} tasks, p = {v['p']:.2g}, share {v['share']:.3f}, "
              f"{'holds' if v['share'] >= 0.75 and v['p'] < 0.05 else 'below the bar'}", flush=True)
    if "candidate" in summary:
        c = summary["candidate"]
        print(f"{c['name']}: beats the margin on {len(c['beats_margin_on'])}/{c['of']}, the control on {c['beats_control_on']}/{c['of']}", flush=True)
    return summary


def smoke():
    """The scoring path and the verdict on synthetic arrays, torch-free."""
    import exp74_suite_encoders as e74
    rng = np.random.default_rng(0)
    n, C, D = 2000, 4, 8
    y = rng.integers(0, C, n)
    p = rng.dirichlet(np.ones(C) * 0.6, size=n)
    emb_te, emb_tr = rng.standard_normal((n, D)), rng.standard_normal((300, D))
    rec = score_units(p, emb_te, emb_tr, y, C, signal=lambda p, a, b, d: 1.0 - p.max(1), name="one_minus_top1_again")
    assert set(rec["signals"]) >= {"margin", "entropy", "one_minus_top1", "one_minus_top1_again"} | set(e70.CONTROLS)
    assert abs(rec["candidate"]["vs_margin"] - (rec["signals"]["margin"]["excess_aurc"] - rec["signals"]["one_minus_top1"]["excess_aurc"])) < 1e-12
    try:
        score_units(p, emb_te, emb_tr, y, C, signal=lambda p, a, b, d: np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("a reading of the wrong length must be refused")
    fake = {f"t{i}": {"margin_lead": 0.05 if i < 20 else -0.01, "source": f"s{i}", "test_accuracy": 0.8} for i in range(24)}
    v = e74.encoder_verdict(fake)
    assert v["wins"] == 20 and v["scored"] == 24 and v["share"] >= 0.75 and v["p"] < 0.05
    print("smoke OK: candidate scoring beside the margin and controls, length check, and the per-model verdict")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=e70.MODEL, help="a model directory of allenai/olmoearth-paper-embeddings")
    ap.add_argument("--signal", default=None, help="module:function of a candidate reading")
    ap.add_argument("--name", default=None, help="name of the candidate in the outputs")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(); return
    cmd_run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""exp71: the protocol on GEO-Bench-2, and the first broad-suite comparison of v1 against v1.2, the served encoder.

The question, in the form it was asked: on the encoder the served product actually uses, what does a reviewer get from
plain v1.2 and what do they get from v1.2 plus this package? Everything below is that contrast.

Why this and not more of exp70. exp70 ran 24 tasks on Ai2's published embeddings and answered the testbed-selection
objection, but it inherited two limits from the fact that embeddings are all those files contain. There is no imagery,
so the only controls available were the distance of a sample from the training mean and the rarity of the predicted
class, both weaker than the pixel indices this repository uses elsewhere; and there are no v1.2 embeddings at all, so
the whole suite speaks about v1 while the encoder the served product uses is v1.2.

GEO-Bench-2 fixes both. It ships imagery, so the encoder runs here and both versions can be compared on identical
windows, and a genuine no-model control can be computed from the pixels. It is also a benchmark assembled by eight
institutions, published in TMLR, with an open licence on every dataset and a public leaderboard.

What it cannot do, measured before anything was run and recorded because it bounds every number below. GEO-Bench-2 is
built for models that read many modalities; OlmoEarth reads Sentinel. Of its 20 datasets only 6 carry Sentinel-2 at
all, and only 2 carry all twelve bands OlmoEarth expects:

  BENV2 12/12, TreeSatAI 12/12, DynamicEarthNet 11/12 (no B09), So2Sat 10/12, PASTIS 10/12, BioMassters 10/12.
  The other 14 are aerial, WorldView, Planet, grayscale or SAR-only and OlmoEarth cannot read them at all.

Three of those six then fall away for reasons measured on the downloaded data rather than assumed. BioMassters is
regression with no classes. TreeSatAI carries 15 tags per sample and BENV2 carries 19, both multi-label, so a top-1
minus top-2 margin is undefined on them for the same reason m_bigearthnet was dropped from exp70. That leaves THREE
usable tasks out of twenty: So2Sat, PASTIS and DynamicEarthNet. The arithmetic is worth stating plainly, because it
says something real about the benchmark and the model rather than about this protocol: GEO-Bench-2 is built to reward
models that read many modalities and many label shapes, and a Sentinel-only single-label ranking protocol reaches a
seventh of it. Missing
bands are filled with zeros and the fill is reported per task, because a zero in a normalised band is not a neutral
value and a reader should be able to discount a task that needed two of them. So this is a result on a named third-party
benchmark and on a minority of it, and the docstring says which minority before the numbers do.

Design. Per task and per encoder version: read GEO-Bench-2's own splits through its own loader at its own
normalisation, map its Sentinel-2 bands into OlmoEarth's order, run the frozen encoder, fit Ai2's linear probe on the
train split, report on the test split.

  Rankers: the margin, one minus top-1, predictive entropy.
  Controls: the within-window pixel standard deviation averaged over bands, which sees the imagery and no model and is
  the control exp70 could not compute; the distance of the embedding from the training mean, which sees neither; and
  the rarity of the predicted class, which sees only the decision.
  Reported with and without: error capture at 5, 10 and 20 percent beside the random baseline, which captures exactly
  the budget, and beside the attainable ceiling, which at budget b and error rate e is min(1, b/e), because a raw
  capture without its ceiling makes an easy task look like a good ranker.

The two arms that matter, and the only ones the headline reports:
  PLAIN v1.2            the served encoder's map, reviewed in no particular order. A reviewer with a budget of b finds
                        exactly b of the errors, because a random b of a map holds a random b of its mistakes.
  v1.2 + inferenceX     the same map, the same model, the same budget, reviewed in the order the margin gives.
Nothing about the model changes between them. What changes is which places the reviewer looks at first, so the two arms
are the same inference scored twice and the difference is attributable to the ordering alone. v1 is run as a secondary
contrast because the encode is cheap and the repository has carried a caveat about v1.2 since exp45.

Preregistered (one-sided):
  P1  on the served encoder, on every task, inferenceX beats plain v1.2: the margin's error capture exceeds the review
      budget at 5, 10 and 20 percent, and it also beats the best no-model control on excess AURC.
  P2  the gain is not an artifact of easy tasks. Capture at a budget is bounded by budget over error rate, so the
      margin's share of that attainable ceiling is above 0.4 on every task, which a ranking that merely rode a low
      error rate could not manage.
  P3  v1.2 ranks its own errors no better than v1, which is what exp45, exp47 and exp51 found on flood water and what
      this tests on a suite: v1.2's excess AURC is at or above v1's on more tasks than not.
  Falsification. P1 fails if a pixel statistic beats the model's own confidence on a task, which would extend exp45's
  Bolivia exception beyond one flood event and one sensor, or if the ordering does not beat random, which would say the
  tool buys a reviewer nothing on the encoder that is actually served. P2 fails if the apparent gain is ceiling rather
  than ranking. P3 fails if v1.2 ranks better, which would retire a caveat this repository has carried since exp45 and
  would be good news worth the run on its own.

Caveats carried into the record. Zero-filled bands are named per task. GEO-Bench-2 evaluates by fine-tuning and this
uses a frozen encoder with a linear probe, so no number here is comparable to their leaderboard and none is claimed to
be. Imagery is resized to the encoder's 64 px input where a task ships smaller tiles, which So2Sat does at 32 px.

Outputs: exp/out/exp71_summary.json, exp/out/exp71_geobench2.csv.
--smoke: synthetic tasks, no download, no encoder, _smoke outputs.
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
ROOT = "/scratch/qi_zim_neu/olmoearth_inferenceX/geobench2"
OE_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
SIZE, PATCH = 64, 4
BUDGETS = (0.05, 0.10, 0.20)
VERSIONS = ("v1_2", "v1")   # v1.2 FIRST and primary: it is the encoder the served product uses

# the six GEO-Bench-2 datasets carrying Sentinel-2, with the class they load and their band coverage
SUITE = {
    "treesatai":         {"cls": "GeoBenchTreeSatAI",      "sub": "treesatai",         "kind": "classification"},
    "benv2":             {"cls": "GeoBenchBENV2",          "sub": "benv2",             "kind": "classification"},
    "so2sat":            {"cls": "GeoBenchSo2Sat",         "sub": "so2sat",            "kind": "classification"},
    "pastis":            {"cls": "GeoBenchPASTIS",         "sub": "pastis",            "kind": "segmentation"},
    "dynamic_earthnet":  {"cls": "GeoBenchDynamicEarthNet", "sub": "dynamic_earthnet", "kind": "segmentation"},
}
EXCLUDED = {"biomassters": "a regression task with no classes, so a top-1 minus top-2 margin is undefined",
            "treesatai": "MEASURED on the downloaded data: 15 tags per sample, multi-label, so a top-1 minus top-2 "
                         "margin is undefined without a convention this project has never tested (as m_bigearthnet "
                         "in exp70)",
            "benv2": "MEASURED on the downloaded data: 19 tags per sample, multi-label, same reason",
            "burn_scars, caffe, cloudsen12, everwatch, flair2, fotw, forestnet, kuro_siwo, nzcattle, "
            "spacenet2, spacenet6, spacenet7, spacenet8, substation":
                "no Sentinel-2 at all: aerial, WorldView, Planet, grayscale or SAR-only, which OlmoEarth cannot read"}


def band_plan(available):
    """Which of OlmoEarth's twelve bands this task has, and which must be zero-filled.

    A zero in a normalised band is not a neutral value, so the fill is returned rather than applied silently and the
    caller records it per task. A reader can then discount a task that needed two fills."""
    have = [b for b in OE_BANDS if b in available]
    missing = [b for b in OE_BANDS if b not in available]
    return have, missing


def to_oe_stack(img, available):
    """(C, H, W) in the task's own band order -> (12, H, W) in OlmoEarth's, zeros where a band is absent."""
    idx = {b: i for i, b in enumerate(available)}
    out = np.zeros((len(OE_BANDS),) + tuple(img.shape[1:]), dtype=np.float32)
    for j, b in enumerate(OE_BANDS):
        if b in idx:
            out[j] = img[idx[b]]
    return out


def pixel_control(img, patch=PATCH):
    """The no-model control exp70 could not compute: within-window pixel standard deviation, averaged over bands."""
    c, h, w = img.shape
    hh, ww = h // patch * patch, w // patch * patch
    x = img[:, :hh, :ww].reshape(c, hh // patch, patch, ww // patch, patch)
    return x.std(axis=(2, 4)).mean(axis=0)


def capture_block(sig, err):
    """Capture at each budget, the random baseline and the attainable ceiling, so no one quotes the first alone."""
    out = {}
    e = float(np.mean(err))
    for b in BUDGETS:
        cap = metrics.capture_at_budget_expected(sig, err, (b,))[b]
        ceil = min(1.0, b / max(e, 1e-12))
        out[str(b)] = {"with_tool": float(cap), "random": float(b), "ceiling": float(ceil),
                       "share_of_ceiling": float(cap / ceil) if ceil > 0 else float("nan"),
                       "multiple_of_random": float(cap / b)}
    return out


# ----------------------------------------------------------------------------- encoder
def load_encoder(version):
    import exp18_sen1floods_expert as exp18
    mid = exp18.ModelID.OLMOEARTH_V1_BASE if version == "v1" else exp18.ModelID.OLMOEARTH_V1_2_BASE
    return exp18.load_model_from_id(mid).to(exp18.DEV).eval().float()


def embed(model, imgs, batch=32):
    """(N, 12, H, W) OlmoEarth-ordered DN -> pooled (N, G, G, D). Tiles smaller than the encoder's input are resized,
    which So2Sat needs at 32 px; the resize is recorded per task rather than left implicit."""
    import torch
    import torch.nn.functional as F
    import exp18_sen1floods_expert as exp18
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.as_tensor(np.asarray(imgs[i:i + batch], dtype=np.float32))
        if x.shape[-1] != SIZE or x.shape[-2] != SIZE:
            x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        crop = SIZE - SIZE % PATCH
        xn = x[:, :, :crop, :crop].permute(0, 2, 3, 1).unsqueeze(3).numpy().astype(np.float64)
        xn = exp18._norm.normalize(exp18.Modality.SENTINEL2_L2A, xn)
        b = xn.shape[0]
        sample = exp18.MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(xn, dtype=torch.float32, device=exp18.DEV),
            sentinel2_l2a_mask=torch.ones((b, crop, crop, 1, 3), device=exp18.DEV) * exp18.MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor([1, 5, 2020], device=exp18.DEV)[None, None, :].repeat(b, 1, 1))
        with torch.no_grad():
            o = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a
        out.append(o.mean(dim=[3, 4]).half().cpu().numpy())
    return np.concatenate(out)


def fit_linear(X, y, C, seed=0):
    """Ai2's linear probe at their published settings, on pooled features."""
    import torch
    import exp18_sen1floods_expert as exp18
    import exp51_their_probe as e51
    dev = exp18.DEV
    torch.manual_seed(seed)
    lin = e51.LinearProbe(in_dim=X.shape[-1], out_dim=C).to(dev).float()
    opt = torch.optim.AdamW(lin.parameters(), lr=e51.LR)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    yt = torch.tensor(np.asarray(y, dtype=np.int64))
    n, steps = len(yt), int(np.ceil(len(yt) / e51.BATCH))
    g = torch.Generator().manual_seed(seed)
    lin.train()
    for ep in range(e51.EPOCHS):
        order = torch.randperm(n, generator=g)
        for i in range(steps):
            idx = order[i * e51.BATCH:(i + 1) * e51.BATCH]
            torch.nn.functional.cross_entropy(lin(Xt[idx].to(dev))["logits"], yt[idx].to(dev)).backward()
            e51.adjust_learning_rate(optimizer=opt, epoch=ep + i / steps, total_epochs=e51.EPOCHS,
                                     warmup_epochs=int(e51.EPOCHS * 0.1), max_lr=e51.LR, min_lr=1.0e-5)
            opt.step(); opt.zero_grad()
    lin.eval()

    def probs(Z):
        o = []
        with torch.no_grad():
            for i in range(0, len(Z), 4096):
                o.append(torch.softmax(lin(torch.tensor(np.asarray(Z[i:i + 4096], dtype=np.float32)).to(dev))["logits"],
                                       dim=1).cpu().numpy())
        return np.concatenate(o)
    return probs


def readings(p):
    p = np.asarray(p, dtype=np.float64)
    srt = np.sort(p, axis=1)
    ent = -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return {"margin": -(srt[:, -1] - srt[:, -2]), "one_minus_top1": 1.0 - srt[:, -1], "entropy": ent}, p.argmax(1)


CONTROLS = ("ctl_pixel_variance", "ctl_embedding_distance", "ctl_class_rarity")


def build_controls(pix, emb_te, emb_tr, dec):
    mu = np.asarray(emb_tr, dtype=np.float64).mean(0)
    freq = collections.Counter(np.asarray(dec).tolist())
    return {"ctl_pixel_variance": np.asarray(pix, dtype=np.float64),
            "ctl_embedding_distance": np.linalg.norm(np.asarray(emb_te, dtype=np.float64) - mu, axis=1),
            "ctl_class_rarity": np.array([-np.log(max(freq[int(c)], 1) / max(len(dec), 1)) for c in dec])}


# ----------------------------------------------------------------------------- one task, one encoder version
def run_task(name, spec, version, model, seed=0):
    from geobench_v2 import datasets as D
    cls = getattr(D, spec["cls"])
    bdo = getattr(cls, "band_default_order")
    avail = list(bdo["s2"]) if isinstance(bdo, dict) and "s2" in bdo else list(bdo)
    have, missing = band_plan(avail)
    if not have:
        raise ValueError(f"{name}: no Sentinel-2 bands OlmoEarth can read")

    def load(split):
        ds = cls(root=os.path.join(ROOT, spec["sub"]), split=split, band_order={"s2": have})
        imgs, labs = [], []
        for i in range(len(ds)):
            s = ds[i]
            img = s.get("image_s2", s.get("image"))
            lab = s.get("label", s.get("mask"))
            if img is None or lab is None:
                raise ValueError(f"{name}: sample has neither image/label nor image_s2/mask; keys {list(s)}")
            if hasattr(lab, "numel") and lab.numel() > 1 and spec["kind"] == "classification":
                raise ValueError(f"{name}: multi-label ({lab.numel()} tags per sample); a top-1 minus top-2 margin is "
                                 f"undefined without a convention this project has never tested")
            imgs.append(to_oe_stack(np.asarray(img, dtype=np.float32), have))
            labs.append(np.asarray(lab))
        return np.stack(imgs), np.stack(labs)

    x_tr, y_tr = load("train")
    x_te, y_te = load("test")
    if spec["kind"] != "classification":
        raise ValueError(f"{name}: only the classification path is implemented in this run")
    C = int(max(y_tr.max(), y_te.max())) + 1
    e_tr = embed(model, x_tr)
    e_te = embed(model, x_te)
    f_tr = e_tr.reshape(len(e_tr), -1, e_tr.shape[-1]).mean(1)      # pooled over the patch grid
    f_te = e_te.reshape(len(e_te), -1, e_te.shape[-1]).mean(1)
    probs = fit_linear(f_tr, y_tr, C, seed=seed)
    p_te, p_tr = probs(f_te), probs(f_tr)
    sig, dec = readings(p_te)
    pix = np.array([pixel_control(im).mean() for im in x_te])       # one no-model reading per sample
    sig.update(build_controls(pix, f_te, f_tr, dec))
    err = (dec != y_te).astype(np.float64)

    scored = {}
    for k, u in sig.items():
        scored[k] = {"excess_aurc": float(metrics.excess_aurc(u, err)),
                     "capture": capture_block(u, err),
                     "auroc": float(metrics.weighted_auroc(u, err, np.ones(len(err))))}
    ctl = min(CONTROLS, key=lambda k: scored[k]["excess_aurc"])
    return {"version": version, "n_test": int(len(y_te)), "n_train": int(len(y_tr)), "n_classes": C,
            "bands_present": have, "bands_zero_filled": missing,
            "test_accuracy": float((dec == y_te).mean()),
            "train_accuracy": float((p_tr.argmax(1) == y_tr).mean()),
            "error_rate": float(err.mean()),
            "best_control": ctl, "margin_lead_over_control": float(scored[ctl]["excess_aurc"]
                                                                  - scored["margin"]["excess_aurc"]),
            "signals": scored}


def cmd_all(args):
    os.makedirs(OUT, exist_ok=True)
    results, rows, failures = collections.defaultdict(dict), [], {}
    for version in (VERSIONS if not args.version else (args.version,)):
        model = load_encoder(version)
        print(f"=== encoder {version} ===", flush=True)
        for name, spec in SUITE.items():
            if args.only and name not in args.only:
                continue
            t0 = time.time()
            try:
                r = run_task(name, spec, version, model, seed=args.seed)
            except Exception as exc:
                failures[f"{version}/{name}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  {name:20s} SKIPPED {failures[f'{version}/{name}'][:110]}", flush=True)
                continue
            r["seconds"] = round(time.time() - t0, 1)
            results[version][name] = r
            c10 = r["signals"]["margin"]["capture"]["0.1"]
            rows.append({"encoder": version, "task": name, "n_test": r["n_test"], "n_classes": r["n_classes"],
                         "bands_zero_filled": ",".join(r["bands_zero_filled"]) or "none",
                         "test_accuracy": r["test_accuracy"], "error_rate": r["error_rate"],
                         "plain_capture_10": c10["random"], "with_tool_capture_10": c10["with_tool"],
                         "multiple_of_random_10": c10["multiple_of_random"],
                         "ceiling_10": c10["ceiling"], "share_of_ceiling_10": c10["share_of_ceiling"],
                         "margin_excess_aurc": r["signals"]["margin"]["excess_aurc"],
                         "best_control": r["best_control"],
                         "margin_lead_over_control": r["margin_lead_over_control"], "seconds": r["seconds"]})
            print(f"  {name:20s} acc {r['test_accuracy']:.3f} err {r['error_rate']:.3f} | "
                  f"plain {c10['random']:.3f} -> with tool {c10['with_tool']:.3f} "
                  f"({c10['multiple_of_random']:.1f}x, {c10['share_of_ceiling']:.2f} of ceiling) | "
                  f"lead {r['margin_lead_over_control']:+.4f} over {r['best_control'][4:]} ({r['seconds']:.0f}s)",
                  flush=True)

    prim = results.get("v1_2", {})
    v = {}
    if prim:
        beats_random = all(prim[t]["signals"]["margin"]["capture"][str(b)]["with_tool"] > b
                           for t in prim for b in BUDGETS)
        beats_ctl = all(prim[t]["margin_lead_over_control"] > 0 for t in prim)
        v["P1"] = {"holds": bool(beats_random and beats_ctl), "n_tasks": len(prim),
                   "beats_random_at_every_budget": bool(beats_random), "beats_best_control": bool(beats_ctl)}
        shares = {t: prim[t]["signals"]["margin"]["capture"]["0.1"]["share_of_ceiling"] for t in prim}
        v["P2"] = {"holds": bool(all(s > 0.4 for s in shares.values())), "share_of_ceiling_at_10pct": shares}
    both = [t for t in results.get("v1_2", {}) if t in results.get("v1", {})]
    if both:
        worse = [t for t in both if results["v1_2"][t]["signals"]["margin"]["excess_aurc"]
                 >= results["v1"][t]["signals"]["margin"]["excess_aurc"]]
        v["P3"] = {"holds": bool(len(worse) > len(both) / 2), "v1_2_no_better_on": f"{len(worse)}/{len(both)}",
                   "per_task": {t: {"v1": results["v1"][t]["signals"]["margin"]["excess_aurc"],
                                    "v1_2": results["v1_2"][t]["signals"]["margin"]["excess_aurc"]} for t in both}}

    summary = {"experiment": "exp71 GEO-Bench-2: plain v1.2 against v1.2 plus inferenceX, on the served encoder",
               "config": {"benchmark": "GEO-Bench-2 (AI Alliance, IBM, TUM, ServiceNow and others; TMLR 2026)",
                          "arms": ["plain v1.2 (random review)", "v1.2 + inferenceX (margin-ordered review)"],
                          "encoders": list(VERSIONS), "budgets": list(BUDGETS), "controls": list(CONTROLS),
                          "suite": {k: v_["cls"] for k, v_ in SUITE.items()}, "excluded": EXCLUDED,
                          "caveats": ["GEO-Bench-2 evaluates by fine-tuning; this uses a frozen encoder and a linear "
                                      "probe, so no number here is comparable to their leaderboard",
                                      "only 6 of the benchmark's 20 datasets carry Sentinel-2 at all and only 2 carry "
                                      "all twelve bands OlmoEarth expects; missing bands are zero-filled and named",
                                      "tiles smaller than 64 px are resized to it, which So2Sat needs at 32 px"]},
               "results": {"by_encoder": {k: dict(x) for k, x in results.items()}, "failures": failures},
               "verdicts": v}
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp71_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, f"exp71_geobench2{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for k, x in v.items():
        print(f"{k}: {x.get('holds')}", flush=True)
    if failures:
        print(f"skipped: {json.dumps(failures, indent=1)[:700]}", flush=True)
    return summary


def smoke(args):
    have, missing = band_plan(["B02", "B03", "B04", "B08"])
    assert have == ["B02", "B03", "B04", "B08"] and len(missing) == 8
    img = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    st = to_oe_stack(img, have)
    assert st.shape == (12, 8, 8)
    assert np.array_equal(st[0], img[0]) and np.array_equal(st[3], img[3])     # B02 and B08 land in OlmoEarth's slots
    assert st[4:].sum() == 0, "absent bands must be zero, and the caller must record that they were"

    pv = pixel_control(np.stack([np.arange(64, dtype=np.float32).reshape(8, 8)] * 3))
    assert pv.shape == (2, 2) and (pv > 0).all()
    flat = pixel_control(np.zeros((3, 8, 8), np.float32))
    assert (flat == 0).all(), "a flat window has no within-window variance"

    rng = np.random.default_rng(0)
    n = 4000
    err = (rng.random(n) < 0.2).astype(float)
    good = rng.random(n) + 0.7 * err            # informative: higher = more suspect, so errors must score HIGHER
    cb = capture_block(good, err)
    for b in BUDGETS:
        r = cb[str(b)]
        assert r["random"] == b
        assert r["with_tool"] > r["random"], "a planted informative signal must beat random review"
        assert r["ceiling"] == min(1.0, b / err.mean()) and r["with_tool"] <= r["ceiling"] + 1e-9
        assert abs(r["multiple_of_random"] - r["with_tool"] / b) < 1e-12
    # an uninformative signal must sit at random, not above it
    flat_cap = capture_block(rng.random(n), err)["0.1"]
    assert abs(flat_cap["with_tool"] - 0.1) < 0.05

    p = rng.dirichlet(np.ones(6) * 0.5, size=500)
    sig, dec = readings(p)
    assert np.array_equal(dec, p.argmax(1)) and (sig["margin"] <= 1e-12).all()
    c = build_controls(rng.random(500), rng.standard_normal((500, 8)), rng.standard_normal((300, 8)), dec)
    assert set(c) == set(CONTROLS)
    print("smoke OK: band mapping and its zero-fill, pixel control, capture against random and its ceiling, readings")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--version", choices=VERSIONS, default=None, help="run one encoder only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    cmd_all(args)


if __name__ == "__main__":
    main()

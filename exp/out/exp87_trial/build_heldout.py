"""exp87 held-out fixtures and their reference facts, built outside both repositories (sealed until the reveal).

H1  F4's two maps swapped (A = the post-event map, B = the pre-event map): the dominant flip and the more confident
    side reverse against B7.
H4  a confidence design of 250 windows (seed 7) on F1's Dynamic World tile, labelled by the expert.
H5  a random design of 400 windows (seed 11) on F1, labelled by the expert.
H2, H3 (new AWF runs) are added by add_cluster.py once job 1064570 has been fetched.

Facts are computed by olmoearth-inferencex and by the agent's own tool functions, never by a model."""
import csv, hashlib, json, os, shutil, subprocess, sys
import numpy as np

SEAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/Desktop/Github/olmoearth_inferenceX")
TRIAL = os.path.join(REPO, "exp/out/exp86_trial")
AGENT_PY = os.path.expanduser("~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python")
FX = os.path.join(SEAL, "fixtures")
sys.path.insert(0, os.path.join(REPO, "exp"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def spatial(idx, grid, bands=4):
    """Share of the listed windows in each of `bands` horizontal and vertical bands of the grid."""
    r, c = np.divmod(np.asarray(idx), grid[1])
    rows = np.bincount(np.minimum(r * bands // grid[0], bands - 1), minlength=bands) / max(len(idx), 1)
    cols = np.bincount(np.minimum(c * bands // grid[1], bands - 1), minlength=bands) / max(len(idx), 1)
    return {"row_band_shares_top_to_bottom": rows.round(4).tolist(), "col_band_shares_left_to_right": cols.round(4).tolist()}


def compare_facts(a, b):
    sa, sb = np.asarray(a["scores"], float), np.asarray(b["scores"], float)
    ca, cb = sa.argmax(1), sb.argmax(1)
    ss_a, ss_b = np.sort(sa, 1), np.sort(sb, 1)
    ma, mb = ss_a[:, -1] - ss_a[:, -2], ss_b[:, -1] - ss_b[:, -2]
    diff = np.flatnonzero(ca != cb)
    pairs = {}
    for i in diff:
        pairs[(int(ca[i]), int(cb[i]))] = pairs.get((int(ca[i]), int(cb[i])), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])
    windows = a.get("windows") or list(range(len(sa)))
    grid = a.get("grid")
    out = {"n_windows": int(len(sa)), "n_differing": int(len(diff)),
           "class_changes": [{"class_a": k[0], "class_b": k[1], "n": n, "share": round(n / len(diff), 4)} for k, n in ranked],
           "a_more_confident_share_of_differing": round(float((ma[diff] > mb[diff]).mean()), 4),
           "mean_margin_a_on_differing": round(float(ma[diff].mean()), 6), "mean_margin_b_on_differing": round(float(mb[diff].mean()), 6)}
    if grid:
        out["spatial"] = spatial([int(windows[i]) for i in diff], grid)
    return out


def build_h1():
    d = os.path.join(FX, "H1")
    os.makedirs(d, exist_ok=True)
    pre, post = (os.path.join(TRIAL, "fixtures/F4", f) for f in ("emsr279-11_s1_pre.json", "emsr279-11_s1_post.json"))
    shutil.copy(post, os.path.join(d, "map_a.json"))
    shutil.copy(pre, os.path.join(d, "map_b.json"))
    a, b = json.load(open(os.path.join(d, "map_a.json"))), json.load(open(os.path.join(d, "map_b.json")))
    facts = compare_facts(a, b)
    facts.update(classes=a.get("classes"), date_a="2018-04-19", date_b="2017-09-14/2017-09-15",
                 note="F4 with A and B swapped: A is the post-event map, B the pre-event map")
    return facts


AGENT_PLAN = r'''
import asyncio, json, os, sys
root, args = sys.argv[1], json.loads(sys.argv[2])
os.environ["OLMOEARTH_SCORES_ROOT"] = root; os.environ["OLMOEARTH_OUTPUT_ROOT"] = root; os.chdir(root)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.skills import build_default_registry
from olmoearth_agent.tools.registry import ToolContext
class _NoStudio:
    def __getattr__(self, n): raise RuntimeError("no Studio in a fixture build")
reg = build_default_registry()
env = asyncio.run(reg.dispatch(ToolCall(id="heldout", name="olmoearth_plan_label_sample", arguments=args), ToolContext(studio=_NoStudio(), state=ThreadState())))
print(json.dumps(env, default=str))
'''


def build_design(tag, design, budget, seed):
    import exp86_fixtures as fx
    d = os.path.join(FX, tag)
    os.makedirs(os.path.join(FX, "F1"), exist_ok=True)
    shutil.copy(os.path.join(TRIAL, "fixtures/F1/dw_scores.json"), os.path.join(FX, "F1/dw_scores.json"))
    args = {"scores_path": "F1/dw_scores.json", "budget": budget, "design": design, "seed": seed}
    r = subprocess.run([AGENT_PY, "-c", AGENT_PLAN, FX, json.dumps(args)], capture_output=True, text=True, cwd=FX)
    env = json.loads(r.stdout.strip().splitlines()[-1])
    assert env.get("ok"), env
    res = env["result"]
    assert res.get("available"), res
    os.makedirs(d, exist_ok=True)
    dp = os.path.join(d, os.path.basename(res["design_path"]))
    sp = os.path.join(d, os.path.basename(res["labels_csv_path"]))
    shutil.move(os.path.join(FX, os.path.basename(res["design_path"])), dp)
    shutil.move(os.path.join(FX, os.path.basename(res["labels_csv_path"])), sp)
    z = np.load(fx.dw_tile())
    meta = json.loads(str(z["meta"]))
    probs = z["probs"].astype(np.float32).astype(np.float64)
    payload, per_row = fx.f1_payload(probs, z["expert"].astype(np.int64), meta["classes"])
    # the current agent's designs and sheets index windows by their grid index (the file's "windows")
    reference = {int(w): int(e) for w, e in zip(payload["windows"], per_row)}
    fx.fill_sheet(sp, reference)
    return dp, sp


def estimation_facts(dp, sp, alphas=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3)):
    from oe_inferencex import estimate
    d = json.load(open(dp))
    rows = {int(r["window_index"]): r for r in csv.DictReader(open(sp))}
    idx = [int(i) for i in d["sample"]["indices"]]
    wrong = [int(rows[i]["wrong"]) for i in idx]
    ref = [int(rows[i]["reference_class"]) for i in idx]
    est = estimate.estimate_error_rate(d["sample"], wrong)
    per = estimate.estimate_per_class(d["sample"], ref, d["population"]["map_class"])
    out = {"design": d["design"], "budget": d["budget"], "n_wrong": int(sum(wrong)),
           "estimate": {k: est[k] for k in ("estimate", "low", "high") if k in est}, "method": est.get("method"),
           "per_class_user_accuracy": {c: (r.get("user_accuracy") or {}).get("estimate") for c, r in per["per_class"].items()},
           "per_class_warnings": {c: r.get("warning_codes") for c, r in per["per_class"].items() if r.get("warning_codes")}}
    if d["design"] == "random":
        margin = np.asarray(d["population"]["margin"], float)
        out["certify_coverage_by_alpha"] = {a: estimate.certify_zone(margin, idx, wrong, a, cut="standard").get("coverage")
                                            for a in alphas}                 # built with 1.x
    return out


def main():
    facts = {"H1": build_h1()}
    dp4, sp4 = build_design("H4", "confidence", 250, 7)
    facts["H4"] = estimation_facts(dp4, sp4)
    dp5, sp5 = build_design("H5", "random", 400, 11)
    facts["H5"] = estimation_facts(dp5, sp5)
    json.dump(facts, open(os.path.join(SEAL, "facts_partial.json"), "w"), indent=1, default=str)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, (list, dict)) or len(str(vv)) < 300 else "...") for kk, vv in v.items()} for k, v in facts.items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

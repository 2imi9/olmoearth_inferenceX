"""exp87 held-out cases from the three new AWF runs (job 1064570), with facts from the agent's own tools (no model)."""
import json, os, shutil, subprocess, sys
import numpy as np
SEAL = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(SEAL, "fixtures")
AGENT_PY = os.path.expanduser("~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python")
sys.path.insert(0, SEAL)
from build_heldout import compare_facts, spatial  # noqa: E402

RUNS = {"C3": "awf_namanga_2021_1064570", "C4": "awf_namanga_2024_1064570", "C5": "awf_amboseli_2023_1064570"}
CALL = r'''
import asyncio, json, os, sys
root, name, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
os.environ["OLMOEARTH_SCORES_ROOT"] = root; os.environ["OLMOEARTH_OUTPUT_ROOT"] = root; os.chdir(root)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.skills import build_default_registry
from olmoearth_agent.tools.registry import ToolContext
class _NoStudio:
    def __getattr__(self, n): raise RuntimeError("no Studio")
reg = build_default_registry()
env = asyncio.run(reg.dispatch(ToolCall(id="heldout", name=name, arguments=args), ToolContext(studio=_NoStudio(), state=ThreadState())))
print(json.dumps(env, default=str))
'''


def tool(name, args):
    r = subprocess.run([AGENT_PY, "-c", CALL, FX, name, json.dumps(args)], capture_output=True, text=True, cwd=FX)
    env = json.loads(r.stdout.strip().splitlines()[-1])
    assert env.get("ok"), (name, str(env)[:800], r.stderr[-800:])
    return env["result"]


def main():
    for tag, run in RUNS.items():
        dst = os.path.join(FX, tag, run)
        os.makedirs(dst, exist_ok=True)
        for f in ("scores.tif", "manifest.json"):
            shutil.copy(os.path.join(SEAL, "cluster", run, f), os.path.join(dst, f))
    prov = {tag: tool("olmoearth_scores_from_file", {"run_dir": f"{tag}/{run}"}) for tag, run in RUNS.items()}
    facts = {}
    a = json.load(open(os.path.join(FX, prov["C3"]["scores_path"])))
    b = json.load(open(os.path.join(FX, prov["C4"]["scores_path"])))
    h2 = compare_facts(a, b)
    h2.update(run_a="C3 (2021)", run_b="C4 (2024)", classes=a.get("classes"))
    facts["H2"] = h2
    rs = tool("olmoearth_review_set", {"scores_path": prov["C5"]["scores_path"], "budget": 0.05})
    s5 = json.load(open(os.path.join(FX, prov["C5"]["scores_path"])))
    rows = np.asarray(s5["scores"], float)
    srt = np.sort(rows, 1)
    m = srt[:, -1] - srt[:, -2]
    listed = [r for r in rs.get("review", [])][:10]
    facts["H3"] = {"n_windows": int(len(rows)), "median_margin": float(np.median(m)),
                   "top10_window_index_in_order": [r.get("window_index") for r in listed],
                   "top10_margins": [r.get("margin") for r in listed],
                   "ratio_median_to_top10_margin": [round(float(np.median(m)) / r["margin"], 2) for r in listed if r.get("margin")],
                   "package_warnings": prov["C5"].get("package_warnings"), "n_review": rs.get("n_review") or rs.get("k"),
                   "classes": s5.get("classes")}
    plan = tool("olmoearth_plan_label_sample", {"scores_path": prov["C5"]["scores_path"], "budget": 200, "design": "confidence", "seed": 3})
    facts["H6"] = {"available": plan.get("available"), "budget": plan.get("budget"), "design": plan.get("design"),
                   "allocation": plan.get("allocation"), "note": "no labels exist: the correct answer states no error rate"}
    part = json.load(open(os.path.join(SEAL, "facts_partial.json")))
    part.update(facts)
    json.dump(part, open(os.path.join(SEAL, "facts.json"), "w"), indent=1, default=str)
    for k in ("H2", "H3", "H6"):
        print(k, json.dumps(facts[k], default=str)[:900])


if __name__ == "__main__":
    main()

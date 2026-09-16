"""exp64 arm E: the OlmoEarth Agent itself, run against the same cards as arms A to D.

The plan page left this arm open: "whether the OlmoEarth Agent is run as a fifth arm after the reproducible
loop, which needs its repository and its own tool schema." Both now exist, so this drives the agent as it
ships, with its own system prompt, its own registry of forty-one tools and its own turn loop, on the same
cards, the same brief as arm A apart from one sentence saying where the scores file is, and the same grader.
Nothing about it was preregistered, so the grade stage reports it against arm A descriptively and says so.

Two variants. E lets the agent choose among all its tools, which is the question a user of the agent cares
about: does it find the review-set skill on its own, or reach for something else. E_forced pins that skill
through the agent's own forced-skill mechanism, which isolates what the skill does for the agent from
whether the agent finds it.

What is recorded, for the claims audit: every tool call the agent makes, with its arguments and the exact
result envelope the agent saw, captured by wrapping the registry's dispatch; the final text; and the parsed
answer. The agent's Studio client is built with a placeholder key and never reaches the network unless the
agent calls a Studio tool, in which case the failure is the agent's real behaviour offline and is recorded.

Runs inside the agent's own virtual environment, which has no numpy, so the card loader from exp64_arms is
imported with the inferenceX repository on PYTHONPATH and numpy added to that environment for the run.
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp64_arms as arms                                                      # noqa: E402

from olmoearth_agent.harness import LeadAgent                                  # noqa: E402
from olmoearth_agent.harness.state import ThreadState                          # noqa: E402
from olmoearth_agent.llm.client import OlmoEarthLLM                            # noqa: E402
from olmoearth_agent.skills import SkillLoader, build_default_registry         # noqa: E402
from olmoearth_agent.studio.client import StudioClient, StudioConfig           # noqa: E402
from olmoearth_agent.tools.registry import ToolRegistry                        # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
CARDS = "/scratch/qi_zim_neu/olmoearth_inferenceX/exp64_cards"
SKILL = "olmoearth-review-set"


def write_scores_json(card):
    """The card's margin and decision as a two-class, logit-like score matrix the agent's tool accepts.

    Row = [m, 0] for class 0 and [0, m] for class 1, so top-1 minus top-2 is the margin and arg-max is the
    decision: the same information arm A reads, in the shape skill #18 asks for. Row-major over the grid."""
    m, d = card["margin"], card["dec"]
    rows = [[float(m[r, c]), 0.0] if int(d[r, c]) == 0 else [0.0, float(m[r, c])]
            for r in range(d.shape[0]) for c in range(d.shape[1])]
    path = os.path.join(card["dir"], "scores.json")
    with open(path, "w") as fh:
        json.dump({"grid": list(d.shape), "n_classes": 2, "classes": card["meta"].get("classes"),
                   "scores": rows}, fh)
    return path


class RecordingRegistry(ToolRegistry):
    """The agent's registry, with every dispatch and its result kept for the audit."""

    def __init__(self, inner):
        super().__init__()
        self._tools = inner._tools
        self.log = []

    async def dispatch(self, call, ctx):
        result = await super().dispatch(call, ctx)
        self.log.append({"name": call.name, "arguments": call.arguments, "result": result})
        return result


async def run_one(card, arm, llm, studio, skill_index, max_turns):
    registry = RecordingRegistry(build_default_registry())
    agent = LeadAgent(llm, registry, studio, state=ThreadState(), skill_index=skill_index,
                      forced_skill=SKILL if arm == "E_forced" else "", local=True)
    brief = arms._prompt(card, arm)
    try:
        res = await agent.run(brief, max_turns=max_turns)
        text, turns, hit_max = res.final_content or "", res.turns, res.hit_max_turns
        err = None
    except Exception as exc:  # noqa: BLE001 - the failure is the run's result, and the loop goes on
        text, turns, hit_max, err = "", 0, False, f"harness error: {type(exc).__name__}: {exc}"
    answer, parse_error = arms._parse_answer(text) if not err else (None, err)
    outputs = {}
    for entry in registry.log:
        outputs.setdefault(entry["name"], []).append(entry["result"])
    return {"arm": arm, "answer": answer, "parse_error": parse_error, "text": text,
            "tool_outputs": outputs,
            "tool_calls": [{"name": e["name"], "arguments": e["arguments"]} for e in registry.log],
            "n_tool_calls": len(registry.log), "n_steps": turns, "hit_max_turns": hit_max,
            "tools_used": sorted({e["name"] for e in registry.log}),
            "used_review_set": any(e["name"] == "olmoearth_review_set" for e in registry.log)}


async def main_async(args):
    root = os.path.join(CARDS, "v1")
    dirs = sorted(d for d in (os.path.join(root, x) for x in os.listdir(root)) if os.path.isdir(d))
    if args.cards:
        dirs = dirs[:args.cards]
    llm = OlmoEarthLLM()
    studio = StudioClient(StudioConfig(api_key="exp64-no-studio-access"))
    try:
        skill_index = SkillLoader().index()
    except Exception as exc:  # noqa: BLE001 - vendored skills absent: the agent runs without the index
        print(f"  skill index unavailable ({type(exc).__name__}); running without it", flush=True)
        skill_index = ""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exp64_answers_E.jsonl")
    n = 0
    try:
        with open(path, "w") as fh:
            for d in dirs:
                card = arms.load_card(d)
                card["scores_json"] = write_scores_json(card)
                for arm in args.arms.split(","):
                    run = await run_one(card, arm, llm, studio, skill_index, args.max_turns)
                    run.update({"card": card["name"], "sample": 0, "model": os.environ.get("LLM_MODEL", "")})
                    fh.write(json.dumps(run, default=float) + "\n")
                    fh.flush()
                    n += 1
                    print(f"  {card['name']:<30} {arm:<9} turns={run['n_steps']} calls={run['n_tool_calls']} "
                          f"review_set={'yes' if run['used_review_set'] else 'no '} "
                          f"parse={'ok' if not run['parse_error'] else 'FAIL'} "
                          f"tools={','.join(run['tools_used'])[:70]}", flush=True)
    finally:
        await studio.aclose()
    print(f"wrote {n} agent runs to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="E,E_forced")
    ap.add_argument("--cards", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=8)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

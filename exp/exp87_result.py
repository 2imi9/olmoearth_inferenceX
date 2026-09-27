"""exp87's main result: material false statements per sentence on the held-out configurations (H) against B1-B8.

The preregistration (docs/plan/agent_trial_v3.md, "The main result") fixes the rule before the run:

- a finding counts when the audit's verifier confirmed it (or only reclassified it), judged it material, and the owner
  confirmed it (``--owner``); a material finding the owner found in the random one-in-ten sample of answers with no
  material finding counts too;
- sentences are those of three words or more in each answer, counted with the agent's own splitter, after dropping
  any "Note from the tool:" the harness appended;
- the fixes generalise if H's rate is at most 3.0% and at most 2.0 points above B1-B8's (point estimates);
- intervals are 95% bootstraps resampling whole configurations; the difference resamples each set independently.

Inputs:

- ``--audit``: the audit JSON (groups of {"audit", "verify"}), configurations under opaque labels;
- ``--key``: {opaque label: real configuration} (the blind key, kept from the auditors);
- ``--owner``: the owner's review, {"decisions": {"<label>|<run>|<quote>": true or false}, "sample_findings":
  [{"configuration": <opaque label>, "run": ..., "quote": ...}]}. Without it, the verifier's judgement stands and
  the output says so.

Run with the agent's interpreter:
``~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python exp/exp87_result.py --audit A --key K [--owner O]``
"""
import argparse
import glob
import json
import os
import random
import re
import sys

from olmoearth_agent.harness.checks import sentences

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ROUND_DIRS = {"H": os.path.join(OUT, "exp87_trial", "rounds", "1", "runs"),
              "B": os.path.join(OUT, "exp86_trial", "rounds", "11", "runs")}
MAX_H_RATE, MAX_GAP = 0.030, 0.020
SAMPLE_SEED, SAMPLE_SHARE = 870927, 10


def finding_id(v):
    return f"{v['configuration']}|{v['run']}|{v['quote']}"


def sentence_counts():
    """{(set, real configuration, run): sentences} over both rounds."""
    counts = {}
    for side, root in ROUND_DIRS.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*", "stdout.txt"))):
            parts = f.split(os.sep)
            config, run = parts[-4] + "/" + parts[-3], parts[-2]
            text = re.split(r"\n\nNote from the tool:", open(f, encoding="utf-8").read())[0]
            counts[(side, config, run)] = sum(len(s.text.split()) >= 3 for s in sentences(text))
    return counts


def owner_sample(answers_without_material):
    """The owner's random one-in-ten, drawn as the preregistration fixes it."""
    pool = sorted(answers_without_material)
    k = max(1, round(len(pool) / SAMPLE_SHARE))
    return sorted(random.Random(SAMPLE_SEED).sample(pool, k))


def bootstrap(units, rng, draws=20000):
    keys = sorted(units)
    out = []
    for _ in range(draws):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        out.append(sum(units[k][0] for k in pick) / sum(units[k][1] for k in pick))
    out.sort()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--owner")
    ap.add_argument("--sample-only", action="store_true", help="print the owner's review list and stop")
    a = ap.parse_args(argv)
    audit = json.load(open(a.audit))
    key = json.load(open(a.key))
    side_of = {lab: ("H" if cfg.startswith("HB") else "B") for lab, cfg in key.items()}
    counts = sentence_counts()

    material = [v for g in audit["groups"] for v in g["verify"]["verdicts"]
                if v["verdict"] in ("confirmed", "reclassified") and v.get("materiality") == "material"]
    answers = {(lab, run) for (side, cfg, run) in counts for lab, c in key.items() if c == cfg}
    with_material = {(v["configuration"], str(v["run"])) for v in material}
    sample = owner_sample(answers - with_material)
    if a.sample_only:
        print(json.dumps({"material_findings_to_review": [finding_id(v) for v in material],
                          "answers_to_read": [f"{lab} run {run}" for lab, run in sample]}, indent=1))
        return 0

    owner = json.load(open(a.owner)) if a.owner else None
    if owner:
        decisions = owner.get("decisions") or {}
        undecided = [finding_id(v) for v in material if finding_id(v) not in decisions]
        if undecided:
            print(f"the owner has not decided {len(undecided)} material findings: {undecided[:5]}", file=sys.stderr)
            return 1
        counted = [v for v in material if decisions[finding_id(v)]]
        counted += owner.get("sample_findings") or []
    else:
        counted = material
        print("NOTE: no owner review given; the verifier's judgement stands in for the owner's", file=sys.stderr)

    units = {"H": {}, "B": {}}
    for (side, cfg, run), n in counts.items():
        lab = next(l for l, c in key.items() if c == cfg)
        k, s = units[side].get(lab, (0, 0))
        units[side][lab] = (k, s + n)
    for v in counted:
        lab = v["configuration"]
        k, s = units[side_of[lab]][lab]
        units[side_of[lab]][lab] = (k + 1, s)

    rng = random.Random(0)
    rate = {s: sum(k for k, _ in u.values()) / sum(n for _, n in u.values()) for s, u in units.items()}
    boot = {s: bootstrap(u, rng) for s, u in units.items()}
    diff = sorted(h - b for h, b in zip(boot["H"], rng.sample(boot["B"], len(boot["B"]))))
    generalise = rate["H"] <= MAX_H_RATE and rate["H"] - rate["B"] <= MAX_GAP
    per_answer = {s: sum(1 for (sd, cfg, run) in counts if sd == s) for s in units}
    result = {
        "rates": {s: {"material": sum(k for k, _ in units[s].values()), "sentences": sum(n for _, n in units[s].values()),
                      "rate": rate[s], "ci95": [boot[s][500], boot[s][19500]],
                      "sentences_per_answer": sum(n for _, n in units[s].values()) / per_answer[s]} for s in units},
        "difference_h_minus_b": {"point": rate["H"] - rate["B"], "ci95": [diff[500], diff[19500]]},
        "rule": {"max_h_rate": MAX_H_RATE, "max_gap": MAX_GAP},
        "verdict": "the fixes generalise" if generalise else "the fixes do not generalise",
        "owner_reviewed": bool(owner),
    }
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""exp86 development rounds 7-10: how often the agent's answers state something false, per answer and per sentence.

Reads the recorded answers (``rounds/<r>/runs/*/*/*/stdout.txt``) and the audits:

- ``exp86_audit_rounds_7_8.json``: rounds 7 and 8 together, under blind labels;
- ``exp86_audit_round_9.json`` and ``exp86_audit_round_10.json``: each round alone.

A finding counts when the adversarial verifier confirmed it or only changed its class. It is material when the verifier
judged it so under the owner's rule (``docs/plan/agent_trial_v3.md``). Sentences are counted with the agent's own
splitter (``olmoearth_agent.harness.checks.sentences``), keeping those of three words or more, after dropping any
"Note from the tool:" the harness appended. Intervals are Wilson 95% intervals. The per-sentence rate treats each
finding as one sentence; it is approximate, since two findings can share a sentence.

Run with the agent's interpreter: ``~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python exp/exp86_claim_rates.py``
"""
import glob
import json
import math
import os
import re
import sys

from olmoearth_agent.harness.checks import sentences

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
AUDITS = {
    7: ("exp86_audit_rounds_7_8.json", "7"),
    8: ("exp86_audit_rounds_7_8.json", "8"),
    9: ("exp86_audit_round_9.json", "9"),
    10: ("exp86_audit_round_10.json", "10"),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def findings(name, rnd):
    audit = json.load(open(os.path.join(OUT, name)))
    label = audit["key"]["label_of_round"][rnd]
    out = []
    for group in audit["groups"]:
        for v in group["verify"]["verdicts"]:
            if v["label"] == label and v["verdict"] in ("confirmed", "reclassified"):
                out.append(v)
    return out


def n_sentences(rnd):
    files = sorted(glob.glob(os.path.join(OUT, "exp86_trial", "rounds", str(rnd), "runs", "*", "*", "*", "stdout.txt")))
    total = 0
    for f in files:
        text = re.split(r"\n\nNote from the tool:", open(f, encoding="utf-8").read())[0]
        total += sum(len(s.text.split()) >= 3 for s in sentences(text))
    return len(files), total


def main():
    rows = []
    for rnd, (name, key) in AUDITS.items():
        found = findings(name, key)
        material = [v for v in found if v.get("materiality") == "material"]
        n_answers, n_sent = n_sentences(rnd)
        rows.append({
            "round": rnd,
            "answers": n_answers,
            "sentences": n_sent,
            "confirmed": len(found),
            "material": len(material),
            "answers_with_any": len({(v["configuration"], v["run"]) for v in found}),
            "answers_with_material": len({(v["configuration"], v["run"]) for v in material}),
        })
    print("| Round | Answers with a material finding | Material findings per sentence | Answers with any finding "
          "| Any finding per sentence | Sentences per answer |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        am = wilson(r["answers_with_material"], r["answers"])
        ms = wilson(r["material"], r["sentences"])
        aa = wilson(r["answers_with_any"], r["answers"])
        cs = wilson(r["confirmed"], r["sentences"])
        print(f"| {r['round']} | {r['answers_with_material']}/{r['answers']} = {am[0]:.0%} ({am[1]:.0%} to {am[2]:.0%}) "
              f"| {r['material']}/{r['sentences']} = {ms[0]:.1%} ({ms[1]:.1%} to {ms[2]:.1%}) "
              f"| {r['answers_with_any']}/{r['answers']} = {aa[0]:.0%} "
              f"| {r['confirmed']}/{r['sentences']} = {cs[0]:.1%} | {r['sentences'] / r['answers']:.1f} |")
    json.dump(rows, open(os.path.join(OUT, "exp86_claim_rates.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "exp86_claim_rates.json"), file=sys.stderr)


if __name__ == "__main__":
    main()

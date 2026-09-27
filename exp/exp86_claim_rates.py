"""exp86 development rounds 7-10: how often the agent's answers state something false, per answer and per sentence.

Reads the recorded answers (``rounds/<r>/runs/*/*/*/stdout.txt``) and the audits:

- ``exp86_audit_rounds_7_8.json``: rounds 7 and 8 together, under blind labels;
- ``exp86_audit_round_9.json`` and ``exp86_audit_round_10.json``: each round alone.

A finding counts when the adversarial verifier confirmed it or only changed its class. It is material when the verifier
judged it so under the owner's rule (``docs/plan/agent_trial_v3.md``). Sentences are counted with the agent's own
splitter (``olmoearth_agent.harness.checks.sentences``), keeping those of three words or more, after dropping any
"Note from the tool:" the harness appended. Per answer, intervals are Wilson 95% intervals. Per sentence, sentences
cluster within answers and answers within briefs, so a Wilson interval over sentences is too narrow (the literature
review of 27 September 2026 pointed this out: it is the design effect the package measures on maps). The per-sentence
interval is a bootstrap that resamples whole briefs (configurations), the most conservative unit; the fall from round
7 to round 10 is also resampled paired by brief. The per-sentence rate treats each finding as one sentence; it is
approximate, since two findings can share a sentence.

Run with the agent's interpreter: ``~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python exp/exp86_claim_rates.py``
"""
import glob
import json
import math
import os
import random
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


def sentences_by_brief(rnd):
    """{configuration: sentences of three words or more over its runs} and the number of answers."""
    files = sorted(glob.glob(os.path.join(OUT, "exp86_trial", "rounds", str(rnd), "runs", "*", "*", "*", "stdout.txt")))
    by = {}
    for f in files:
        parts = f.split(os.sep)
        cfg = parts[-4] + "/" + parts[-3]
        text = re.split(r"\n\nNote from the tool:", open(f, encoding="utf-8").read())[0]
        by[cfg] = by.get(cfg, 0) + sum(len(s.text.split()) >= 3 for s in sentences(text))
    return by, len(files)


def brief_bootstrap(units, rng, draws=20000):
    """95% interval of sum(findings) / sum(sentences), resampling briefs with replacement."""
    keys = sorted(units)
    stats = []
    for _ in range(draws):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        stats.append(sum(units[k][0] for k in pick) / sum(units[k][1] for k in pick))
    stats.sort()
    return stats[int(0.025 * draws)], stats[int(0.975 * draws)]


def main():
    rows = []
    rng = random.Random(0)
    units_by_round = {}
    for rnd, (name, key) in AUDITS.items():
        found = findings(name, key)
        material = [v for v in found if v.get("materiality") == "material"]
        by_brief, n_answers = sentences_by_brief(rnd)
        n_sent = sum(by_brief.values())
        per_brief = {c: 0 for c in by_brief}
        for v in material:
            per_brief[v["configuration"]] += 1
        units = {c: (per_brief[c], by_brief[c]) for c in by_brief}
        units_by_round[rnd] = units
        lo, hi = brief_bootstrap(units, rng)
        rows.append({
            "round": rnd,
            "answers": n_answers,
            "sentences": n_sent,
            "confirmed": len(found),
            "material": len(material),
            "answers_with_any": len({(v["configuration"], v["run"]) for v in found}),
            "answers_with_material": len({(v["configuration"], v["run"]) for v in material}),
            "material_per_sentence_brief_bootstrap_95": [lo, hi],
        })
    print("| Round | Answers with a material finding | Material findings per sentence | Answers with any finding "
          "| Any finding per sentence | Sentences per answer |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        am = wilson(r["answers_with_material"], r["answers"])
        ms = (r["material"] / r["sentences"], *r["material_per_sentence_brief_bootstrap_95"])
        aa = wilson(r["answers_with_any"], r["answers"])
        cs = wilson(r["confirmed"], r["sentences"])
        print(f"| {r['round']} | {r['answers_with_material']}/{r['answers']} = {am[0]:.0%} ({am[1]:.0%} to {am[2]:.0%}) "
              f"| {r['material']}/{r['sentences']} = {ms[0]:.1%} ({ms[1]:.1%} to {ms[2]:.1%}) "
              f"| {r['answers_with_any']}/{r['answers']} = {aa[0]:.0%} "
              f"| {r['confirmed']}/{r['sentences']} = {cs[0]:.1%} | {r['sentences'] / r['answers']:.1f} |")
    a, b = units_by_round[7], units_by_round[10]
    keys = sorted(set(a) & set(b))
    diffs = []
    for _ in range(20000):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(sum(a[k][0] for k in pick) / sum(a[k][1] for k in pick)
                     - sum(b[k][0] for k in pick) / sum(b[k][1] for k in pick))
    diffs.sort()
    fall = {"round_7_minus_round_10_95": [diffs[500], diffs[19500]],
            "share_of_resamples_without_a_fall": sum(d <= 0 for d in diffs) / len(diffs)}
    print(f"\nround 7 minus round 10, per sentence, paired by brief: {diffs[500]:.1%} to {diffs[19500]:.1%}; "
          f"{fall['share_of_resamples_without_a_fall']:.4f} of resamples show no fall")
    json.dump({"rounds": rows, "fall_7_to_10": fall}, open(os.path.join(OUT, "exp86_claim_rates.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "exp86_claim_rates.json"), file=sys.stderr)


if __name__ == "__main__":
    main()

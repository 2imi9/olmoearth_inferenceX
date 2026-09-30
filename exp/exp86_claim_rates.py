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
rate treats each finding as one sentence; it is approximate, since two findings can share a sentence.

The resampling unit. The runs sit under ``runs/<brief>/<backend>/``. Eight briefs (B1-B8) run on ten brief-by-backend
configurations, because B3 and B4 each run on two backends. The first version of this file resampled the ten
configurations and called them briefs; a configuration is not independent of the other configuration of the same
brief, so that interval is too narrow. Both are reported, each under its own name:

- by configuration: a bootstrap over the 10 configurations (the numbers recorded on 27 September, unchanged);
- by brief: a bootstrap over the 8 briefs, a brief's configurations pooled (material findings and sentences summed).

The fall from round 7 to round 10 is resampled paired at both levels, and tested exactly at the brief level: under no
change between the rounds, each brief's round-7 and round-10 counts are exchangeable, so the 2^8 = 256 ways of swapping
them within briefs give the exact one-sided p of the observed fall in the pooled per-sentence rate. A sign test over the
briefs' own rates is reported beside it. Brief B3 carries half of round 7's material findings, so every fall statistic
is also given without it.

Run with the agent's interpreter: ``~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python exp/exp86_claim_rates.py``
"""
import glob
import itertools
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
DRAWS = 20000
LEFT_OUT = "B3"


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


def sentences_by_configuration(rnd):
    """{configuration "<brief>/<backend>": sentences of three words or more over its runs} and the number of answers."""
    files = sorted(glob.glob(os.path.join(OUT, "exp86_trial", "rounds", str(rnd), "runs", "*", "*", "*", "stdout.txt")))
    by = {}
    for f in files:
        parts = f.split(os.sep)
        cfg = parts[-4] + "/" + parts[-3]
        text = re.split(r"\n\nNote from the tool:", open(f, encoding="utf-8").read())[0]
        by[cfg] = by.get(cfg, 0) + sum(len(s.text.split()) >= 3 for s in sentences(text))
    return by, len(files)


def by_brief(units):
    """Configuration units {"<brief>/<backend>": (material, sentences)} pooled to {brief: (material, sentences)}."""
    out = {}
    for cfg, (k, s) in units.items():
        b = cfg.split("/")[0]
        k0, s0 = out.get(b, (0, 0))
        out[b] = (k0 + k, s0 + s)
    return out


def rate(units, keys=None):
    keys = sorted(units) if keys is None else keys
    return sum(units[k][0] for k in keys) / sum(units[k][1] for k in keys)


def bootstrap(units, rng, draws=DRAWS):
    """95% interval of sum(findings) / sum(sentences), resampling the units (configurations or briefs) with replacement."""
    keys = sorted(units)
    stats = []
    for _ in range(draws):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        stats.append(rate(units, pick))
    stats.sort()
    return stats[int(0.025 * draws)], stats[int(0.975 * draws)]


def paired_fall(a, b, rng, draws=DRAWS):
    """The round-a minus round-b per-sentence rate, the units resampled together: 95% interval, share without a fall."""
    keys = sorted(set(a) & set(b))
    diffs = []
    for _ in range(draws):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(rate(a, pick) - rate(b, pick))
    diffs.sort()
    return {"round_7_minus_round_10_95": [diffs[int(0.025 * draws)], diffs[int(0.975 * draws)]],
            "share_of_resamples_without_a_fall": sum(d <= 0 for d in diffs) / draws,
            "n_units": len(keys), "draws": draws}


def exact_permutation(a, b):
    """Exact one-sided p of the observed fall rate(a) - rate(b), swapping each unit's two rounds in every way."""
    keys = sorted(set(a) & set(b))
    observed = rate(a, keys) - rate(b, keys)
    count = 0
    for flips in itertools.product((False, True), repeat=len(keys)):
        x = {k: (b[k] if f else a[k]) for k, f in zip(keys, flips)}
        y = {k: (a[k] if f else b[k]) for k, f in zip(keys, flips)}
        count += rate(x, keys) - rate(y, keys) >= observed - 1e-12
    n = 2 ** len(keys)
    return {"observed_fall": observed, "n_relabelings": n, "n_at_least_observed": count, "p_one_sided": count / n,
            "n_units": len(keys)}


def sign_test(a, b):
    """Units whose own per-sentence rate fell against those where it rose; exact one-sided binomial p, ties dropped."""
    keys = sorted(set(a) & set(b))
    down = sum(a[k][0] / a[k][1] > b[k][0] / b[k][1] for k in keys)
    up = sum(a[k][0] / a[k][1] < b[k][0] / b[k][1] for k in keys)
    n = down + up
    p = sum(math.comb(n, j) for j in range(down, n + 1)) / 2 ** n if n else float("nan")
    return {"units_falling": down, "units_rising": up, "ties": len(keys) - n, "p_one_sided": p}


def main():
    rows = []
    rng = random.Random(0)            # the configuration-level draws, in the order and seed of the 27 September file
    rng_brief = random.Random(0)      # the brief-level draws, a stream of their own so neither moves the other
    units_by_round = {}
    for rnd, (name, key) in AUDITS.items():
        found = findings(name, key)
        material = [v for v in found if v.get("materiality") == "material"]
        by_cfg, n_answers = sentences_by_configuration(rnd)
        n_sent = sum(by_cfg.values())
        per_cfg = {c: 0 for c in by_cfg}
        for v in material:
            per_cfg[v["configuration"]] += 1
        units = {c: (per_cfg[c], by_cfg[c]) for c in by_cfg}
        briefs = by_brief(units)
        units_by_round[rnd] = units
        lo, hi = bootstrap(units, rng)
        blo, bhi = bootstrap(briefs, rng_brief)
        rest = [b for b in sorted(briefs) if b != LEFT_OUT]
        rows.append({
            "round": rnd,
            "answers": n_answers,
            "sentences": n_sent,
            "confirmed": len(found),
            "material": len(material),
            "answers_with_any": len({(v["configuration"], v["run"]) for v in found}),
            "answers_with_material": len({(v["configuration"], v["run"]) for v in material}),
            "n_configurations": len(units),
            "n_briefs": len(briefs),
            "material_per_sentence_bootstrap_by_configuration_95": [lo, hi],
            "material_per_sentence_bootstrap_by_brief_95": [blo, bhi],
            "per_configuration": {c: list(v) for c, v in sorted(units.items())},
            "per_brief": {b: list(v) for b, v in sorted(briefs.items())},
            f"without_{LEFT_OUT}": {"material": sum(briefs[b][0] for b in rest),
                                    "sentences": sum(briefs[b][1] for b in rest), "rate": rate(briefs, rest)},
        })
    print("| Round | Answers with a material finding | Material findings per sentence (by brief; by configuration) "
          "| Answers with any finding | Any finding per sentence | Sentences per answer |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        am = wilson(r["answers_with_material"], r["answers"])
        bl, bh = r["material_per_sentence_bootstrap_by_brief_95"]
        cl, ch = r["material_per_sentence_bootstrap_by_configuration_95"]
        aa = wilson(r["answers_with_any"], r["answers"])
        cs = wilson(r["confirmed"], r["sentences"])
        print(f"| {r['round']} | {r['answers_with_material']}/{r['answers']} = {am[0]:.0%} ({am[1]:.0%} to {am[2]:.0%}) "
              f"| {r['material']}/{r['sentences']} = {r['material'] / r['sentences']:.1%} "
              f"({bl:.1%} to {bh:.1%}; {cl:.1%} to {ch:.1%}) "
              f"| {r['answers_with_any']}/{r['answers']} = {aa[0]:.0%} "
              f"| {r['confirmed']}/{r['sentences']} = {cs[0]:.1%} | {r['sentences'] / r['answers']:.1f} |")
    a, b = units_by_round[7], units_by_round[10]
    ab, bb = by_brief(a), by_brief(b)
    rest = [k for k in sorted(ab) if k != LEFT_OUT]
    fall = {
        "by_configuration": paired_fall(a, b, rng),
        "by_brief": paired_fall(ab, bb, rng_brief),
        "exact_brief_permutation": exact_permutation(ab, bb),
        "brief_sign_test": sign_test(ab, bb),
        f"without_{LEFT_OUT}": {
            "round_7": rate(ab, rest), "round_10": rate(bb, rest), "fall": rate(ab, rest) - rate(bb, rest),
            "by_brief": paired_fall({k: ab[k] for k in rest}, {k: bb[k] for k in rest}, rng_brief),
            "exact_brief_permutation": exact_permutation({k: ab[k] for k in rest}, {k: bb[k] for k in rest}),
            "brief_sign_test": sign_test({k: ab[k] for k in rest}, {k: bb[k] for k in rest})},
    }
    for level in ("by_configuration", "by_brief"):
        f = fall[level]
        print(f"\nround 7 minus round 10, per sentence, paired {level.replace('_', ' ')} ({f['n_units']} units): "
              f"{f['round_7_minus_round_10_95'][0]:.1%} to {f['round_7_minus_round_10_95'][1]:.1%}; "
              f"{f['share_of_resamples_without_a_fall']:.4f} of resamples show no fall")
    e, s = fall["exact_brief_permutation"], fall["brief_sign_test"]
    print(f"exact brief-level permutation: {e['n_at_least_observed']} of {e['n_relabelings']} relabelings reach the "
          f"observed fall of {e['observed_fall']:.1%}, one-sided p = {e['p_one_sided']:.4f}; sign test over briefs "
          f"{s['units_falling']} down, {s['units_rising']} up, {s['ties']} tied, p = {s['p_one_sided']:.4f}")
    w = fall[f"without_{LEFT_OUT}"]
    print(f"without {LEFT_OUT}: {w['round_7']:.1%} to {w['round_10']:.1%}; paired by brief "
          f"{w['by_brief']['round_7_minus_round_10_95'][0]:.1%} to {w['by_brief']['round_7_minus_round_10_95'][1]:.1%}, "
          f"{w['by_brief']['share_of_resamples_without_a_fall']:.4f} without a fall; exact p = "
          f"{w['exact_brief_permutation']['p_one_sided']:.4f} ({w['exact_brief_permutation']['n_at_least_observed']} of "
          f"{w['exact_brief_permutation']['n_relabelings']})")
    json.dump({"resampling_units": {"configuration": "brief by backend, 10 per round (B3 and B4 run on two backends)",
                                    "brief": "the 8 briefs, a brief's configurations pooled"},
               "rounds": rows, "fall_7_to_10": fall},
              open(os.path.join(OUT, "exp86_claim_rates.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "exp86_claim_rates.json"), file=sys.stderr)


if __name__ == "__main__":
    main()

"""What the package now writes differently from 1.3.1, on purpose, and nothing else.

The golden files beside this one are the record of 1.3.1 and are not edited. Each change below is a whole text
1.3.1 wrote, the whole text that replaces it, and every golden file that holds it. In a JSON file the text is a
whole string value; in a stdout file it is a whole line, alone or after "warning: ". `expected(name, data)` is the
golden file as the package must now write it: the listed changes applied, every other byte as 1.3.1 wrote it. A
listed text that is not where the list says, or that sits in a file the list does not name, fails the test, so the
list is exact. The new texts are written out here, not imported, so a later edit of the package's text fails the
test until it is listed too; `constant` names where the package keeps each one.
tests/test_cli.py::test_existing_outputs_are_byte_identical_to_1_3_1 uses this file.
"""
import json

CHANGES = (
    {"why": "certify's note on the prefix rule. 1.3.1 said the rule is valid only if the zone's error rate does not "
            "fall as the zone grows. The rule is fixed-sequence testing and valid on any map "
            "(tests/test_trust_zone.py enumerates it); the note now says so, and when bonferroni can certify more",
     "old": ("valid if the zone's error rate does not fall as the zone grows; on the suite tasks exp80 graded, the "
             "guarantee held whether or not that was exactly true (docs/results/comparisons.md, exp80)"),
     "new": ("prefix rule: fixed-sequence testing, valid on any map whatever the shape of its error rate; it stops at "
             "the first zone it cannot certify, so it certifies little when the most confident windows hold many "
             "errors, where the bonferroni rule can certify more"),
     "constant": ("oe_inferencex.estimate", "PREFIX_NOTE"),
     "files": ("sample_random.certify_delta.json", "sample_random.certify_delta.stdout.txt",
               "sample_random.certify_prefix.json", "sample_random.certify_prefix.stdout.txt",
               "sample_random.certify_whole.json", "sample_random.certify_whole.stdout.txt",
               "sample_scene_random.certify_prefix.json", "sample_scene_random.certify_prefix.stdout.txt")},
)
LINE_PREFIXES = ("", "warning: ")


def _replace(name, data, old, new):
    """(data with every whole occurrence of old replaced by new, the number replaced). Only JSON and stdout files
    can hold a change; any other file (arrays, CSVs) is returned as it is."""
    if name.endswith(".json"):
        o, n = json.dumps(old).encode(), json.dumps(new).encode()     # a whole string value, quotes included
        return data.replace(o, n), data.count(o)
    if not name.endswith(".stdout.txt"):
        return data, 0
    lines, count = data.decode().split("\n"), 0
    for i, line in enumerate(lines):
        for pre in LINE_PREFIXES:
            if line == pre + old:
                lines[i], count = pre + new, count + 1
    return "\n".join(lines).encode(), count


def expected(name, data):
    """The golden file `name` (its 1.3.1 bytes `data`) as the package must write it now."""
    for c in CHANGES:
        assert c["old"] != c["new"] and json.dumps(c["new"])[1:-1] == c["new"], c["why"]   # no escaping to hide in
        data, count = _replace(name, data, c["old"], c["new"])
        want = 1 if name in c["files"] else 0
        assert count == want, f"{name}: {count} whole occurrences of a listed 1.3.1 text, expected {want}: {c['why']}"
        assert c["old"].encode() not in data, f"{name}: a listed 1.3.1 text is left inside other text: {c['why']}"
    return data

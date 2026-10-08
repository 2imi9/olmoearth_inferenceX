"""What the package now writes differently from 1.3.1, on purpose, and nothing else.

The golden files beside this one are the record of 1.3.1 and are not edited. Each change below is a whole text
1.3.1 wrote, the whole text that replaces it, and every golden file that holds it. In a JSON file the text is a
whole string value; in a stdout file it is a whole line, alone or after "warning: ". `expected(name, data)` is the
golden file as the package must now write it: the listed changes applied, every other byte as 1.3.1 wrote it. A
listed text that is not where the list says, or that sits in a file the list does not name, fails the test, so the
list is exact. The new texts are written out here, not imported, so a later edit of the package's text fails the
test until it is listed too; `constant` names where the package keeps each one. `same_but_last_digits` is the one
tolerance the comparison allows, the last digits of a float, which differ between numpy builds and C libraries. Two
tests use this file: tests/test_cli.py::test_existing_outputs_are_byte_identical_to_1_3_1 and
tests/test_assess.py::test_without_a_condition_only_scope_is_added.
"""
import json
import math

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
    {"why": "the warning on a multi-class logit map scored by the logit margin. 1.3.1 said only \"pass form='top1'\", "
            "which is a Python argument with no command-line option; the warning now says how the command line and "
            "the MCP server get a top-probability reading and how it differs",
     "old": ("multi-class logit margin: on Ai2's suite one minus the top probability ranked errors better on 14 of 16 "
             "multi-class tasks (exp76); pass form='top1'"),
     "new": ("multi-class logit margin: on Ai2's suite one minus the top probability ranked errors better on 14 of 16 "
             "multi-class tasks (exp76). In Python, pass form='top1'. Elsewhere, pass the class probabilities (the "
             "softmax of the logits) instead of logits: on the command line without --logits, through the MCP server "
             "with logits=false. That also ranks by the top probability, averaged over each window where form='top1' "
             "averages its log, and can tie where probabilities saturate"),
     "constant": ("oe_inferencex.assess", "MARGIN_FORM_WARNING"),
     "files": ("api_prediction_logits3_margin.json", "assess_logits3__assessment.json", "sample_logits3_random.json",
               "sample_logits3_random.stdout.txt")},
    {"why": "the warning on every probability map. 1.3.1 ended it \"prefer logits\", which is wrong for a map of more "
            "than two classes, where the logit margin ranked errors worse than the probability margin on 16 of 16 "
            "multi-class tasks (exp76) and the multi-class logit warning itself says to pass probabilities",
     "old": "probability input: confidence ties where probabilities saturate; prefer logits",
     "new": ("probability input: confidence ties where probabilities saturate. For two classes, logits avoid the ties; "
             "for more than two, keep the probabilities (exp76)"),
     "constant": ("oe_inferencex.assess", "PROBABILITY_WARNING"),
     "files": ("api_prediction_dw.json", "api_prediction_scene.json", "assess_dw__assessment.json",
               "assess_scene__assessment.json", "sample_confidence.json", "sample_confidence.stdout.txt",
               "sample_proportional.json", "sample_proportional.stdout.txt", "sample_random.json",
               "sample_random.stdout.txt", "sample_scene_confidence.json", "sample_scene_confidence.stdout.txt",
               "sample_scene_random.json", "sample_scene_random.stdout.txt", "sample_tiles.json",
               "sample_tiles.stdout.txt")},
    {"why": "the warning on a tiles sample. 1.3.1 quoted for 'this interval' the coverage of exp78's own design (exactly "
            "18 tiles, a normal quantile); the warning now quotes the shipped design's (exp/out/exp78_shipped_tiles.json), "
            "and says the naive interval's 51 to 78% is exp78's, on six of its seven tasks",
     "old": ("labels taken tile by tile are not independent, and a map whose tiles differ in size is labelled unevenly; "
             "the naive interval beside this one is what the ordinary formula says, and on exp78's tasks it covered 51 "
             "to 78% of the time while claiming 95%. This interval is better and still not honest everywhere: on "
             "exp78's tasks it covered 0.91 to 0.94 where tiles were of equal size, 0.82 on Sen1Floods11 and 0.60 on "
             "MADOS, whose tiles hold 1 to 400 windows. Prefer the confidence design"),
     "new": ("labels taken tile by tile are not independent, and a map whose tiles differ in size is labelled unevenly; "
             "the naive interval beside this one is what the ordinary formula says, and in exp78, with 18 tiles of 16 "
             "windows, it covered 51 to 78% of the time on six of seven tasks while claiming 95%. This interval is "
             "better and still not honest everywhere: graded on exp78's tasks it covered 94.5 to 95.4% of the time on "
             "five, 84.3% on Sen1Floods11 and 68.5% on MADOS, where a tenth of the tiles hold most of the errors. "
             "Prefer the confidence design"),
     "constant": ("oe_inferencex.estimate", "TILES_WARNING"),
     "files": ("sample_tiles.estimate.json", "sample_tiles.estimate.stdout.txt")},
    {"why": "the instruction sample prints for the reviewer. It asked for 1 or 0 only, and a reviewer who could not "
            "judge a window had no way to say so that estimate accepted: the refusal of anything else told them to "
            "leave the window out, and a CSV without it did not match its design. ? is now accepted and bounded",
     "old": "Fill the `wrong` column with 1 or 0 per window, then run:",
     "new": ("Fill the `wrong` column with 1 or 0 per window, or ? where a window cannot be judged (keep its row), "
             "then run:"),
     "within_line": True,
     "constant": ("oe_inferencex.cli", "FILL_INSTRUCTION"),
     "files": ("sample_confidence.stdout.txt", "sample_logits3_random.stdout.txt", "sample_proportional.stdout.txt",
               "sample_random.stdout.txt", "sample_scene_confidence.stdout.txt", "sample_scene_random.stdout.txt",
               "sample_tiles.stdout.txt")},
)
LINE_PREFIXES = ("", "warning: ")


def _replace(name, data, old, new, within_line=False):
    """(data with every whole occurrence of old replaced by new, the number replaced). Only JSON and stdout files
    can hold a change; any other file (arrays, CSVs) is returned as it is. A change `within_line` is a stretch of a
    stdout line (a line that also holds a path or a count); it is replaced wherever it occurs in a stdout file."""
    if name.endswith(".json"):
        o, n = json.dumps(old).encode(), json.dumps(new).encode()     # a whole string value, quotes included
        return data.replace(o, n), data.count(o)
    if not name.endswith(".stdout.txt"):
        return data, 0
    if within_line:
        return data.replace(old.encode(), new.encode()), data.count(old.encode())
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
        data, count = _replace(name, data, c["old"], c["new"], c.get("within_line", False))
        want = 1 if name in c["files"] else 0
        assert count == want, f"{name}: {count} whole occurrences of a listed 1.3.1 text, expected {want}: {c['why']}"
        assert c["old"].encode() not in data, f"{name}: a listed 1.3.1 text is left inside other text: {c['why']}"
    return data


def same_but_last_digits(a, b):
    """Equal JSON values but for the last digits of floats: the same keys in the same order, the same integers,
    strings, booleans and nulls, and every float within a relative 1e-12 of the other. A float never equals an
    integer. For a golden JSON that reads back to itself, this is equal bytes but for the digits of some floats."""
    if type(a) is float or type(b) is float:
        return type(a) is type(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same_but_last_digits(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_but_last_digits(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b

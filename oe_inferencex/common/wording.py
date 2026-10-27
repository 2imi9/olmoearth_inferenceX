"""The class design's wording: results' condition words rewritten as class words, and the class names they carry."""
import json
import os


# One class is written per window, seen at one time, and both maps are graded against it. Until 1.14.1 nothing said
# so: the route was offered for two maps of different dates, where a change on the ground between them reads as an
# error of the map of the other date (the OlmoEarth Agent's trials, exp86 round 8).
_CLASS_WORDS = [(r"\bper class or per condition\b", "per class"), (r"\bper input condition\b", "per map class"), (r"\bby input condition\b", "by map class"),
                (r"\binput conditions\b", "map classes"), (r"\binput condition\b", "map class"),
                (r"\bconditions\b", "classes"), (r"\bcondition\b", "class"),
                (r"\bConditions\b", "Classes"), (r"\bCondition\b", "Class")]
_PATH_KEYS = {"sample", "csv", "scores", "classes", "confidence", "zone_mask", "source", "out", "written_by"}


def class_words(obj, keep=(), paths=()):
    """A result of the class design in its own words: its strata are the map's classes, which the condition design's
    estimators and texts call conditions. Keys, path values, the paths a text names, a `code span` (a column or a
    command) and the class names in `keep` are left as they are. A class named "good condition" was printed "good
    class", the JSON's outside_condition_intervals named classes that were not its keys (a KeyError in the MCP estimate
    tool), and a sample in a folder named condition was sent to .../class/to_label.csv, a file that does not exist (the
    bug hunt of 2026-10-25).

    Paths are held first, each as a whole token, with its file name where that has an extension: the generic path token
    matched "sample_csv=/a/Site" first and reworded " conditions 2026", and a file name held inside words ("t" in
    "input") turned the wording off (the reviews of 1.17.2). A class name is held as a whole word, so a backtick in it
    opens no code span, unless it is made only of the words the rules match ("input", "condition"): held, it stopped the
    rules in the package's own sentences, so it is worded as they are. A list entry that is exactly a class name, as in
    outside_condition_intervals, names a stratum and is kept; other values are worded (a class named "condition" kept
    the design's own value "condition" unworded; the fourth review)."""
    import re
    from oe_inferencex.certify.zone import RULES_OF
    if RULES_OF.get() < (1, 17, 2):                  # as 1.17.0 and 1.17.1 worded, for verify of a zone they wrote
        from oe_inferencex.certify.certificate import _class_words_1171
        return _class_words_1171(obj)
    names = {str(k) for k in keep if str(k)}
    if isinstance(obj, str):
        held = []
        # a mark the text does not hold: a NUL of the text's own hung the restore, or was taken for a mark (a name
        # "a\x000\x00" printed "awater"; the fourth and fifth reviews of 1.17.2)
        mark = next(chr(c) for c in range(0xE000, 0xF900) if chr(c) not in obj)

        def hold(m):
            held.append(m.group(0))
            return f"{mark}{len(held) - 1}{mark}"
        kept = {os.fspath(p) for p in paths if p and ("/" in os.fspath(p) or "\\" in os.fspath(p) or "." in os.fspath(p))}
        kept |= {b for b in (os.path.basename(k.replace("\\", "/")) for k in list(kept)) if "." in b}
        if kept:
            obj = re.sub("|".join(rf"(?<![^\s=(\'\"]){re.escape(k)}(?!\w)"
                                  for k in sorted(kept, key=len, reverse=True)), hold, obj)
        # a name with a character the later passes read (a backtick, a slash) is held too, whatever its words
        held_names = [k for k in names if set(re.findall(r"\w+", k.lower())) - _RULE_WORDS or re.search(r"[`/\\]", k)]
        if held_names:
            obj = re.sub("|".join(rf"(?<!\w){re.escape(k)}(?!\w)" for k in sorted(held_names, key=len, reverse=True)),
                         hold, obj)
        m = re.escape(mark)
        obj = re.sub(rf"[^\s{m}]*[/\\][^\s{m}]*|`[^`{m}]*`", hold, obj)
        for pat, rep in _CLASS_WORDS:
            obj = re.sub(pat, rep, obj)
        for i in reversed(range(len(held))):            # a held stretch can hold only earlier ones
            obj = obj.replace(f"{mark}{i}{mark}", held[i])
        return obj
    if isinstance(obj, dict):
        return {k: (v if k in _PATH_KEYS else class_words(v, keep, paths)) for k, v in obj.items()}
    if isinstance(obj, list):
        # an entry that is exactly a class name names a stratum (outside_condition_intervals), and is kept
        return [v if isinstance(v, str) and v in names else class_words(v, keep, paths) for v in obj]
    return obj


# the words the rules match: a class name made only of them cannot be held without stopping the rules in the package's
# own sentences ("t" or "input" held in "per input condition"), and is worded as they are
_RULE_WORDS = {"per", "by", "input", "condition", "conditions", "class", "or"}       # every word of _CLASS_WORDS


def class_names(obj):
    """The class design's stratum names, which class_words keeps as written, or None for a sample or a result of any
    other design: read from a sample's sidecar (its condition block says by "map class") or from a result of estimate
    or certify (condition_by)."""
    cond = (obj or {}).get("condition") or {}
    if isinstance(cond, dict) and cond.get("by") == "map class":
        return [str(n) for n in cond.get("names") or []]
    if (obj or {}).get("condition_by") == "map class":
        return [str(n) for n in (obj.get("per_condition") or {})]
    return None


def class_wording(obj, *paths, names=None):
    """class_words with the class design's names and the `paths` a text names kept, for a sample or result of the
    class design (or the `names` given); the identity for any other."""
    names = class_names(obj) if names is None else names
    kept = [os.fspath(p) for p in paths if p]
    return (lambda x: x) if names is None else (lambda x: class_words(x, names, kept))


def result_class_names(result, path=None):
    """class_names of a result, and for a zone JSON that does not say (a certificate bundle's zone.json carries no
    condition_by, so that 1.17.1 can verify it) the names its sample's sidecar gives: the bundle's labels.json beside
    it, or the sidecar of the sample it names. decide answered a bundle's class-design zone in condition words (the
    third review of 1.17.2)."""
    names = class_names(result)
    if names is not None or not isinstance(result, dict) or not result.get("by_condition"):
        return names
    sides = []
    if path and os.path.basename(str(path)) == "zone.json":
        sides.append(os.path.join(os.path.dirname(os.path.abspath(str(path))), "labels.json"))
    if isinstance(result.get("sample"), str):
        sides.append(result["sample"][:-4] + ".json" if result["sample"].endswith(".csv") else result["sample"] + ".json")
    for side_path in sides:
        try:
            with open(side_path, encoding="utf-8") as f:
                n = class_names(json.load(f))
        except (OSError, ValueError, AttributeError):
            continue
        if n is not None:
            return n
    return None

"""A local MCP server: the package's commands as tools that an agent calls on the user's own files.

    pip install "olmoearth-inferencex[geo,mcp]"
    oe-inferencex mcp                                      # stdio; the agent starts it, nobody types into it
    claude mcp add oe-inferencex -- oe-inferencex mcp      # Claude Code

Nothing is hosted. The server runs on the user's machine and writes its files where the agent says. It reads the
files the agent passes, the sidecar beside a sample and the scores raster it records (estimate with per_class, and
certify, read the map's scores from the path the sidecar holds unless scores is given). It runs the command line's
own code (`cli.main`), so a tool and its command give the same numbers and refuse the same inputs. Each tool returns
compact JSON: the files written, the summary numbers, and three plain texts the agent can quote: `conclusion` (what
the tool found), `limits` (what it does not show, with the package's own warnings and notes) and `next` (what can be
done next, with its preconditions). A refusal of the package comes back as a tool error carrying the package's own
message.

The teaching follows the OlmoEarth Agent trial (exp86, exp/out/exp86_development_rounds_summary.md). There, material
false statements per sentence fell from 7.5% to 2.2% on the eight development briefs the fixes were built against,
mostly in round 8, which bundled tool outputs that state conclusions and limits, statistical rules in code and answer
checks. No single change is shown to have caused the fall; the differences between rounds 8, 9 and 10 are within
audit variation; the owner has not adjudicated materiality; and the held-out test (exp87) has no result yet. Two of
its changes are copied here:

- tool outputs that state their conclusion and their limits in plain words, so that the model does not guess
  (round 8);
- a capability card per tool: what it does, what it needs and what it cannot do, with two rules: state as fact only
  what a tool returned, and propose only what the tools can do (round 10). Here the card is the tool's description.

The standard order and the hard rules in the instructions (`INSTRUCTIONS`, also the `guide` tool) are the owner's
addition, not a change the trial tested.

skills/oe-inferencex/SKILL.md holds the same teaching for an agent that runs the command line instead.

The texts and the tool functions import without the mcp extra; only `build_server` and `main` need it.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import threading
from typing import Annotated, Any, Literal

try:                                       # the annotations below are strings until FastMCP reads them (with mcp)
    from pydantic import Field
except ImportError:                        # pragma: no cover - pydantic comes with the mcp extra
    Field = None

from oe_inferencex import __version__, cli
from oe_inferencex import estimate as est

NEEDS_EXTRA = ("the MCP server needs the mcp extra: pip install \"olmoearth-inferencex[geo,mcp]\". The extra pins the "
               "MCP Python SDK below 2, because mcp 2 renamed FastMCP to MCPServer; an environment that holds mcp 2 "
               "needs its own install of the package")

# ----------------------------------------------------------------------------- the teaching
HARD_RULES = (
    "A review set is not a sample. It is chosen to hold errors, so its error rate overstates the map's. estimate and "
    "certify refuse it.",
    "certify needs a random sample: draw it with design \"random\", or with a condition layer. The default design "
    "serves estimate only; certify refuses it.",
    "Without labels, compare cannot say which map is right. Two maps that agree can both be wrong.",
    "Labels are assumed right. The interval and the zone describe agreement with the reviewer's labels.",
    "Ranking needs the scores, not only the class map. A class map alone works only in compare. assess refuses a "
    "class map of more than two classes read as probabilities, but not a 0/1 map, nor any class map passed with "
    "logits=true: it reads the class ids as scores, and the order it gives is not evidence.",
)

INSTRUCTIONS = "\n".join([
    "oe-inferencex checks a classification map made by an Earth-observation model. It reads the scores the model "
    "wrote out before the argmax: a GeoTIFF or .npy of shape (C, H, W), one band per class, or (H, W) for two "
    "classes; probabilities from 0 to 1, or logits with logits=true. It works on windows: square blocks of patch x "
    "patch pixels, 4 by default. The tools read and write local files only. Pass absolute paths.",
    "",
    "The standard order:",
    "1. assess: where to look. It ranks the windows from least to most confident. The review set is the least "
    "confident share, the windows to check first. No labels.",
    "2. sample, label, estimate: how wrong the map is. sample picks the windows to label. The user or a reviewer "
    "fills the `wrong` column with 1 or 0 on every row. estimate gives the error rate with a 95% interval.",
    "3. certify: which part to trust. From the same labels, the most confident share of the map whose error rate is "
    "at most alpha.",
    "4. compare: two maps of one area. Where they differ; with labels, which map is right there.",
    "5. Per condition: when a raster records each pixel's input condition (a cloud flag, the modalities present, a "
    "sensor id), pass it as condition to assess and to sample. estimate and certify then give each condition its own "
    "rate and zone. A model run on inputs it was not trained on can be sure and wrong, and its errors then come late "
    "in a ranking of the whole map.",
    "",
    "Hard rules. Do not work around them. The package refuses only what a rule says it refuses; the rest is up to "
    "you:",
    *[f"- {rule}" for rule in HARD_RULES],
    "",
    "How to report:",
    "- State as fact only what a tool returned. Quote its conclusion and its limits.",
    "- Give an interval as its two ends. Never recompute it as p +/- 1.96 sqrt(p(1-p)/n).",
    "- A window's confidence ranks windows. It is not the probability that the window is wrong.",
    "- No tool labels windows, fetches labels, runs a model or knows whether labels exist. Labels come from the user "
    "or a reviewer.",
    "- Propose only what these tools can do, with their preconditions.",
    "- A refusal is a tool error carrying the package's reason. Change the input it names; do not retry the same "
    "call. The reasons name command-line options: --design is the parameter design, --labels-date is labels_date, "
    "and so on.",
])

CARDS = {
    "guide": "\n".join([
        "How to use these tools: the standard order, the hard rules and how to report.",
        "Does: returns the server's instructions, every tool's capability card and the directory relative paths are "
        "read from.",
        "Needs: nothing.",
        "Cannot: run anything or read any file.",
    ]),
    "assess": "\n".join([
        "Which windows of one map to check first. No labels.",
        "Does: reads a score raster, splits it into windows of patch x patch pixels and ranks them from least to "
        "most confident. Writes the review sets (the least confident 1%, 5% and 10% by default) as CSVs with pixel "
        "and map coordinates, suspicion and boundary rasters, explanation.json (the cues behind each flagged window) "
        "and assessment.json. With condition, it also ranks each input condition on its own. With reference, it "
        "grades the order against that raster, taken as truth.",
        "Needs: the model's scores before the argmax: (C, H, W) per-class probabilities, or (H, W) for two classes, "
        "as GeoTIFF or .npy; logits=true for logits. A GeoTIFF needs rasterio (the geo extra). An output directory.",
        "Cannot: say how wrong the map is (that needs labels: sample, then estimate); rank a class map. One of more "
        "than two classes read as probabilities is refused, but a 0/1 map, or any class map passed with logits=true, "
        "is read as scores: its windows tie and the order is not evidence. It cannot find the errors the model is "
        "sure of, which come last, or label windows.",
    ]),
    "compare": "\n".join([
        "Where two maps of the same area differ, window by window.",
        "Does: reads two maps on one grid (class maps, probability maps or per-class scores), counts the windows "
        "whose classes differ and says whether they sit on class boundaries. Writes comparison.json, "
        "differing_windows.csv and a disagreement raster. With labels, it says which map matches the labels where "
        "they differ. With date_a and date_b, it says whether a difference can be real change on the ground.",
        "Needs: two rasters with the same shape, CRS and transform; threshold for a continuous map outside [0, 1]; "
        "labels_date with labels when the two maps describe different dates; an output directory.",
        "Cannot: say which map is right without labels (two maps that agree can both be wrong); resample a map onto "
        "another grid; give either map's error rate (that is sample, then estimate, on its scores).",
    ]),
    "sample": "\n".join([
        "Which windows to label, so that estimate can give the map's error rate.",
        "Does: draws budget windows and writes them to a CSV with an empty `wrong` column for the reviewer, and a "
        ".json sidecar holding the design, which estimate and certify read. Designs: confidence (the default without "
        "condition: stratified by confidence; estimate reads it, certify refuses it), random (serves estimate and "
        "certify), proportional, tiles (a cluster design for labelling tile by tile) and condition (the default with "
        "condition: labels split equally across the input conditions).",
        "Needs: the score raster assess takes; a budget (the recorded experiments used 300); an output directory.",
        "Cannot: label the windows, fetch labels or say whether labels exist (the user or a reviewer fills "
        "`wrong`); turn a review set into a sample.",
    ]),
    "estimate": "\n".join([
        "The map's error rate with a 95% interval, from the labelled sample.",
        "Does: reads the CSV that sample wrote, once `wrong` holds 1 or 0 on every row, and its .json sidecar. Gives "
        "the error rate with the interval the sample's design earns, and each input condition's rate when the sample "
        "recorded a condition. With per_class, each class's user's and producer's accuracy and its error-adjusted "
        "share of the map.",
        "Needs: the CSV sample wrote, filled on every row and kept in its order, with its .json beside it. per_class "
        "needs a `reference_class` column (the class seen in each window) and the map's scores (from the sidecar, or "
        "scores).",
        "Cannot: estimate from a review set or from any CSV that sample did not write (refused); check that the "
        "labels are right (they are assumed right); say which windows are wrong.",
    ]),
    "certify": "\n".join([
        "Which share of the map, from the most confident window down, is wrong at most alpha of the time, with a "
        "guarantee.",
        "Does: from the labelled sample, finds the largest most-confident share of the map whose error rate is at "
        "most alpha; the statement fails on at most delta of the samples that could have been drawn. Writes the "
        "result JSON and a window mask (.npy, True inside the zone). A sample drawn with condition is certified per "
        "condition. It may certify nothing, and then says why.",
        "Needs: a sample drawn with design \"random\" (or with condition), labelled on every row; alpha, such as "
        "0.05; the map's scores (from the sidecar, or scores).",
        "Cannot: certify from the default confidence design, a review set or windows chosen by hand (refused); say "
        "anything about the windows outside the zone; check that the labels are right.",
    ]),
}


# ----------------------------------------------------------------------------- running the command line
class Refused(Exception):
    """The package refused the input; the message is the package's own."""


def _abs(path):
    return None if path is None else os.path.abspath(os.path.expanduser(str(path)))


def _input(path, what):
    """An input file as an absolute path, refused when it is not there."""
    p = _abs(path)
    if not os.path.isfile(p):
        raise Refused(f"{what}: {p} is not a file")
    return p


def _num(x):
    """A number as the command line takes it after `=`, so a negative value is not read as an option."""
    return repr(float(x)) if isinstance(x, float) else str(x)


def _opt(argv, flag, value):
    if value is not None:
        argv.append(f"{flag}={_num(value)}")


_CAPTURE = threading.Lock()


def _run(argv):
    """Run one `oe-inferencex` command in this process and return what it printed. stdout is the protocol's
    channel on stdio, so nothing may print there: both streams are captured. A refusal (SystemExit with a message)
    becomes Refused with that message; an argparse error carries its message on stderr. The redirection is the whole
    process's, so a lock keeps one call inside it at a time: FastMCP 1.30 calls a sync tool on the event loop, one at
    a time, but a later 1.x could run tools on worker threads, and two calls would then restore each other's stdout."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with _CAPTURE, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            raise Refused(exc.code) from None
        if exc.code:
            raise Refused(err.getvalue().strip() or f"oe-inferencex {argv[0]} exited with status {exc.code}") from None
    return out.getvalue()


def _tagged(printed, tag):
    """The lines a command printed after `warning: ` or `note: `, in the package's own words."""
    head = f"{tag}: "
    return [line[len(head):] for line in printed.splitlines() if line.startswith(head)]


def _sentence(text):
    """A text with a full stop at its end; None or an empty text is None."""
    if text is None or not str(text).strip():
        return None
    t = str(text).strip()
    return t if t[-1] in ".!?" else t + "."


def _cap(text):
    """One of the package's fixed lines as a sentence, its first letter a capital (never used on a name)."""
    t = _sentence(text)
    return t[0].upper() + t[1:]


def _warnings(texts):
    """The package's warnings, each after `Warning: `, as the command prints them."""
    return [f"Warning: {str(w).strip()}" for w in texts if w is not None and str(w).strip()]


def _notes(texts):
    """The package's notes, each after `Note: ` and in its own words: a note can start with a file or condition
    name, whose case is not ours to change."""
    return [f"Note: {str(n).strip()}" for n in texts if n is not None and str(n).strip()]


def _core(text):
    """A text without its `Note: ` or `Warning: ` head and its full stop, to tell whether it is already said."""
    for head in ("Note: ", "Warning: "):
        if text.startswith(head):
            text = text[len(head):]
    return text.rstrip(".")


def _join(texts):
    """One text from several, each once: a text already inside another is left out, and so is None. The heads are
    left out of the comparison, so a note printed alone is seen inside the longer note that holds it."""
    kept = []
    for t in texts:
        t = _sentence(t)
        if t is None or any(_core(t) in k for k in kept):
            continue
        kept = [k for k in kept if _core(k) not in t] + [t]
    return " ".join(kept)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _pc(x, nd=1):
    return "undefined" if x is None else f"{100 * float(x):.{nd}f}%"


def _budget(b):
    return f"{100 * float(b):g}%"


def _review_set_hint(path):
    """What to add to a refusal when the CSV given as a sample is a review set that assess wrote."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
    except (OSError, UnicodeDecodeError):
        return ""
    # the whole map's review set has `rank`, each condition's own has `rank_in_condition`; a sample has `index`
    if ("rank" in header or "rank_in_condition" in header) and "index" not in header:
        return (" This CSV is a review set that assess wrote. " + HARD_RULES[0] + " Draw a sample with the sample "
                "tool and have every row of it labelled.")
    return ""


def _labelled(path, argv):
    """Run estimate or certify on a sample CSV; a refusal of a review set says what the file is."""
    try:
        return _run(argv)
    except Refused as exc:
        hint = _review_set_hint(path)
        raise Refused((_sentence(str(exc)) if hint else str(exc)) + hint) from None


def _scored(argv):
    """Run assess or sample. The package's refusal of a class map as probabilities names --logits; for a class map
    that would only trade a refusal for an order that is not evidence, so the refusal says so."""
    try:
        return _run(argv)
    except Refused as exc:
        msg = str(exc)
        if "which is not a probability map. Pass --logits" in msg:
            msg = (_sentence(msg) + " If the file holds class ids (a class map), logits=true does not help: it reads "
                   "the ids as scores, the windows tie and the order is not evidence. Pass the model's per-class "
                   "scores.")
        raise Refused(msg) from None


def _condition_argv(names):
    """--condition-names and its values. A value that starts with `-` would be read as another option, such as
    --out, so each must be value=name with a whole number value."""
    for n in names:
        key, sep, _ = str(n).partition("=")
        if str(n).startswith("-") or not sep or not key.strip().isdigit():
            raise Refused(f"condition_names: {str(n)!r} is not value=name with a whole number value of 0 or more, "
                          "such as \"0=clear\"")
    return ["--condition-names", *[str(n) for n in names]]


# ----------------------------------------------------------------------------- the tools
P = Field  # a parameter's description, read by FastMCP


def guide() -> str:
    """The server's instructions and every tool's capability card."""
    cards = "\n\n".join(f"## {name}\n{card}" for name, card in CARDS.items())
    return (f"{INSTRUCTIONS}\n\nRelative paths are read from {os.getcwd()}.\n\n# Capability cards\n\n{cards}\n\n"
            f"olmoearth-inferencex {__version__}")


def assess(
    scores: Annotated[str, P(description="The score raster: GeoTIFF or .npy, (C, H, W) per class or (H, W) for two "
                                         "classes, taken before the argmax")],
    out_dir: Annotated[str, P(description="Directory to write the review sets, rasters and JSON into; created if "
                                          "needed. Files of the same name are overwritten")],
    logits: Annotated[bool, P(description="True if the scores are logits rather than probabilities")] = False,
    patch: Annotated[int, P(description="Window side in pixels")] = 4,
    nodata: Annotated[float | None, P(description="No-data value (default: the raster's own, plus NaN)")] = None,
    reference: Annotated[str | None, P(description="Optional integer class raster on the map's grid, taken as truth, "
                                                   "to grade the order")] = None,
    budgets: Annotated[list[float] | None, P(description="Review budgets as fractions of the windows (default "
                                                         "[0.01, 0.05, 0.1])")] = None,
    order: Annotated[Literal["confidence", "boundary_first"], P(description="Review order")] = "confidence",
    condition: Annotated[str | None, P(description="Optional integer raster on the map's grid: each pixel's input "
                                                   "condition (a cloud flag, the modalities present, a sensor id)")] = None,
    condition_names: Annotated[list[str] | None, P(description="Names of the condition values, as value=name, "
                                                               "e.g. [\"0=clear\", \"1=cloudy\"]")] = None,
) -> dict[str, Any]:
    scores = _input(scores, "scores")
    out = _abs(out_dir)
    argv = ["assess", scores, f"--out={out}", f"--patch={int(patch)}", f"--order={order}"]
    if logits:
        argv.append("--logits")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if reference is not None:
        argv.append(f"--reference={_input(reference, 'reference')}")
    if budgets:
        argv += ["--budgets", *[_num(float(b)) for b in budgets]]
    if condition is not None:
        argv.append(f"--condition={_input(condition, 'condition')}")
    if condition_names:
        argv += _condition_argv(condition_names)
    printed = _scored(argv)
    s = _read_json(os.path.join(out, "assessment.json"))
    files = {k: os.path.abspath(v) for k, v in s["files"].items()}
    files["assessment"] = os.path.join(out, "assessment.json")
    rs = s["review_sets"]
    b5 = min(rs, key=lambda b: abs(float(b) - 0.05))
    sets = ", ".join(f"{rs[b]['n_windows']} ({_budget(b)})" for b in rs)
    # a review set most of whose windows tie at its cut-off is ordered by raster position: a class map read as
    # scores (a 0/1 map, or class ids passed as logits) does this, and the package does not refuse it
    tied = {b: rs[b]["tied_at_cutoff"] for b in rs if rs[b].get("tied_at_cutoff")}
    loose = [b for b, t in tied.items() if 2 * t["inside"] > rs[b]["n_windows"]]
    if loose:
        bt = b5 if b5 in loose else loose[0]
        opener = (f"The order is not evidence here: {tied[bt]['inside']} of the {rs[bt]['n_windows']} windows of the "
                  f"{_budget(bt)} review set share the cut-off score with {tied[bt]['outside']} windows left outside, "
                  "so among them the order is raster position. A class map passed with logits=true, or a 0/1 map, is "
                  "read as scores and ties like this; if the file is one, pass the model's per-class scores instead.")
    else:
        opener = "Check the least confident windows first."
    said = [f"{opener} Of {s['n_windows']} windows of {s['patch_px']} x "
            f"{s['patch_px']} pixels, the review sets hold {sets} windows, each listed least confident first with "
            f"pixel and map coordinates. The {_budget(b5)} review set is in {files.get(f'review_set_{b5}')}; files "
            "names the others.",
            f"{_pc(s['boundary_window_fraction'])} of all windows sit on a class boundary of the map."]
    summ = {"n_windows": s["n_windows"], "patch_px": s["patch_px"], "n_classes": s.get("n_classes"),
            "signal": s.get("signal"), "review_order": s.get("review_order"),
            "review_sets": {b: rs[b]["n_windows"] for b in rs},
            "boundary_window_fraction": s["boundary_window_fraction"]}
    if tied:
        summ["tied_at_cutoff"] = tied
    limits = ["This says where to look, not how wrong the map is: no labels were used." if reference is None else
              "The order says where to look. Its grade against the reference holds for this map and this reference only.",
              "The review set is not a sample. It is chosen to hold errors, so its error rate overstates the map's.",
              "A window's confidence ranks windows; it is not the probability that the window is wrong.",
              "Errors the model is sure of come last in this order, so a review of the review set does not find them."]
    ref = s.get("against_reference") or {}
    if reference is not None:
        cap = ref.get("error_capture_at_budget") or {}
        if cap:
            said.append(f"Against the reference, {_pc(ref.get('error_rate'))} of the scored windows disagree with it; "
                        f"the {_budget(b5)} review set holds {_pc(cap[b5]['errors_captured_fraction'], 0)} of those "
                        "disagreements.")
            summ["against_reference"] = {"error_rate": ref.get("error_rate"), "n_windows_scored": ref.get("n_windows_scored"),
                                         "errors_captured_fraction": {b: v["errors_captured_fraction"] for b, v in cap.items()}}
        else:
            said.append("Against the reference, no window has both a prediction and a majority reference label, so "
                        "nothing was graded.")
        limits.append("The reference raster is taken as truth" + (f": {ref['caveat']}." if ref.get("caveat") else "."))
    if condition is not None:
        blk = s["conditions"]
        per = blk["per_condition"]
        said.append(f"{blk['n_conditions']} input condition{'s' if blk['n_conditions'] != 1 else ''}: " + "; ".join(
            f"{name} {_pc(e['share_of_map'])} of the windows and {_pc(e['share_of_review_set'][b5], 0)} of the "
            f"{_budget(b5)} review set" for name, e in per.items())
            + f". Each condition's own review set is in {files.get(f'review_set_{b5}_by_condition')}.")
        summ["conditions"] = {name: {"share_of_map": e["share_of_map"], "share_of_review_set": e["share_of_review_set"]}
                              for name, e in per.items()}
    limits += _warnings(s.get("warnings", []))
    limits += _notes([s.get("scope")] + _tagged(printed, "note"))
    nxt = ("For how wrong the map is: sample with design \"random\" and a budget (the recorded experiments used 300), "
           "have the user or a reviewer fill `wrong` on every row, then estimate; the same labels then serve certify.")
    if condition is None:
        nxt += " If a raster records each pixel's input condition, pass it as condition to assess and to sample."
    return {"conclusion": " ".join(said), "limits": _join(limits), "next": nxt, "files": files, "summary": summ}


def compare(
    a: Annotated[str, P(description="First map: integer classes, a probability map (cut at threshold) or (C, H, W) "
                                    "scores (argmax)")],
    b: Annotated[str, P(description="Second map, on the same grid as a")],
    out_dir: Annotated[str, P(description="Directory to write comparison.json, differing_windows.csv and the "
                                          "disagreement raster into; created if needed")],
    patch: Annotated[int, P(description="Window side in pixels")] = 4,
    nodata: Annotated[float | None, P(description="No-data value of both maps")] = None,
    threshold: Annotated[float | None, P(description="Cut-off of a 2-D continuous map: 0.5 for a probability map "
                                                     "when omitted; required for any other range")] = None,
    labels: Annotated[str | None, P(description="Optional integer class raster on the same grid: which map is right "
                                                "where they differ")] = None,
    groups: Annotated[str | None, P(description="Optional integer raster of group ids (tiles, events), for "
                                                "per-group rates")] = None,
    date_a: Annotated[str | None, P(description="Date map a describes, YYYY-MM-DD, or a period "
                                                "YYYY-MM-DD/YYYY-MM-DD")] = None,
    date_b: Annotated[str | None, P(description="Date or period map b describes")] = None,
    labels_date: Annotated[str | None, P(description="Date or period the labels describe; required with labels when "
                                                     "the maps describe different dates")] = None,
) -> dict[str, Any]:
    a, b = _input(a, "a"), _input(b, "b")
    out = _abs(out_dir)
    argv = ["compare", a, b, f"--out={out}", f"--patch={int(patch)}"]
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    _opt(argv, "--threshold", None if threshold is None else float(threshold))
    for flag, path, what in (("--labels", labels, "labels"), ("--groups", groups, "groups")):
        if path is not None:
            argv.append(f"{flag}={_input(path, what)}")
    for flag, value in (("--date-a", date_a), ("--date-b", date_b), ("--labels-date", labels_date)):
        if value is not None:
            argv.append(f"{flag}={value}")
    _run(argv)
    s = _read_json(os.path.join(out, "comparison.json"))
    files = {k: os.path.abspath(v) for k, v in s["files"].items()}
    files["comparison"] = os.path.join(out, "comparison.json")
    if s["n_disagree"]:
        said = [f"{s['n_disagree']} of {s['n_windows']} windows differ ({_pc(s['disagreement_rate'], 2)}). They are "
                f"listed in {files['differing_windows']}, and {files['disagreement']} marks them."]
    else:
        said = [f"0 of {s['n_windows']} windows differ. The two maps give the same class in every window both "
                "predict."]
    if s.get("per_group"):
        said.append("Per group: " + "; ".join(
            f"group {g}, {e['n_disagree']} of {e['n']}" + (" windows differ" if i == 0 else "")
            + f" ({_pc(e['rate'])})" for i, (g, e) in enumerate(s["per_group"].items())) + ".")
    where = s.get("where") or {}
    enr = {k: (where.get(k) or {}).get("enrichment") for k in ("boundary_a", "boundary_b")}
    if enr["boundary_a"] is not None:
        said.append(f"The differing windows sit on a class boundary of map a {enr['boundary_a']:.1f} times as often "
                    "as the agreeing windows" + (f" (of map b, {enr['boundary_b']:.1f} times)." if enr["boundary_b"]
                                                  is not None else "."))
    summ = {"n_windows": s["n_windows"], "n_disagree": s["n_disagree"], "disagreement_rate": s["disagreement_rate"],
            "boundary_enrichment": enr, "dates_status": s["dates"]["status"]}
    if s.get("per_group"):
        summ["per_group"] = s["per_group"]
    graded = s.get("graded")
    if graded:
        ws = graded["which_side"]
        if ws["n_disagree"]:
            said.append(f"Against the labels, where the maps differ, a is right on {_pc(ws['share_a_right'], 0)} and b "
                        f"on {_pc(ws['share_b_right'], 0)} of the {ws['n_disagree']} windows, and neither on the other "
                        f"{ws['neither']}.")
        else:
            said.append("The maps differ on no labelled window, so the labels have nothing to grade between them.")
        og = graded.get("over_groups")
        if og and og.get("n_groups"):
            said.append(f"Against the labels, b is right more often than a in {og['w']} of the {og['n_groups']} groups, "
                        f"a more often in {og['l']}, equally often in {og['t']}.")
            summ["over_groups"] = og
            summ["graded_per_group"] = graded.get("per_group")
        summ["which_side"] = ws
        summ["crosstab"] = graded.get("crosstab")
        limits = ["The labels are assumed right: the grading describes agreement with them."]
    else:
        limits = [HARD_RULES[2]]
    if s["dates"]["status"] in ("different_time", "overlapping_time", "partly_stated"):
        said.append(s["dates"]["reading"][0].upper() + s["dates"]["reading"][1:] + ".")
    limits += ["A difference between the maps says neither map's error rate."] + _notes(s.get("notes", []))
    nxt = ("With a label raster on the same grid, compare again with labels to see which map is right where they "
           "differ." if not graded else "For either map's error rate: sample on that map's scores, label, estimate.")
    return {"conclusion": " ".join(said), "limits": _join(limits), "next": nxt, "files": files, "summary": summ}


def sample(
    scores: Annotated[str, P(description="The score raster assess takes")],
    out_dir: Annotated[str, P(description="Directory to write the CSV and its .json sidecar into; created if needed")],
    budget: Annotated[int, P(description="Number of windows to label (the recorded experiments used 300)")],
    design: Annotated[Literal["confidence", "proportional", "random", "tiles", "condition"] | None,
                      P(description="Sampling design. Default: condition with a condition layer, confidence without. "
                                    "Use random to certify from the same labels")] = None,
    name: Annotated[str, P(description="File name of the CSV in out_dir")] = "to_label.csv",
    condition: Annotated[str | None, P(description="Optional integer raster on the map's grid: each pixel's input "
                                                   "condition")] = None,
    condition_names: Annotated[list[str] | None, P(description="Names of the condition values, as value=name")] = None,
    logits: Annotated[bool, P(description="True if the scores are logits")] = False,
    patch: Annotated[int, P(description="Window side in pixels")] = 4,
    nodata: Annotated[float | None, P(description="No-data value (default: the raster's own, plus NaN)")] = None,
    seed: Annotated[int, P(description="Seed of the draw")] = 0,
    tile: Annotated[int, P(description="tiles design: tile side in windows")] = 16,
    per_tile: Annotated[int, P(description="tiles design: windows labelled per tile")] = 16,
) -> dict[str, Any]:
    scores = _input(scores, "scores")
    out = _abs(out_dir)
    if os.path.basename(str(name)) != str(name) or not str(name):
        raise Refused(f"name {name!r} must be a file name, not a path; out_dir gives the directory")
    path = os.path.join(out, str(name))
    argv = ["sample", scores, f"--budget={int(budget)}", f"--out={path}", f"--patch={int(patch)}",
            f"--seed={int(seed)}", f"--tile={int(tile)}", f"--per-tile={int(per_tile)}"]
    if design is not None:
        argv.append(f"--design={design}")
    if logits:
        argv.append("--logits")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if condition is not None:
        argv.append(f"--condition={_input(condition, 'condition')}")
    if condition_names:
        argv += _condition_argv(condition_names)
    printed = _scored(argv)
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    side = _read_json(side_path)
    d = side["design"]
    cond = side.get("condition")
    per = dict(zip(cond["names"], cond["n_labelled"])) if cond else None
    said = [f"{len(side['indices'])} windows to label of {side['n_population']} valid ({d} design), listed in {path}; "
            f"the design is in {side_path}."]
    if per:
        said.append("Labels per condition: " + ", ".join(f"{k} {int(v)}" for k, v in per.items()) + ".")
    limits = ["No window is labelled yet: no tool labels windows, fetches labels or knows whether labels exist.",
              HARD_RULES[3] + " To label blind, hide map_class, write the class seen in a reference_class column and "
              "set wrong where the two differ."]
    if d in ("random", "condition"):
        limits.append("Both estimate and certify read this sample.")
    else:
        limits.append(f"This sample serves estimate, and certify refuses the {d} design. To certify a zone, sample "
                      "with design \"random\".")
    limits += _warnings(_tagged(printed, "warning")) + _notes(_tagged(printed, "note"))
    nxt = (f"Ask the user or a reviewer to open each window of {path} and set `wrong` to 1 if map_class is not what "
           f"is there, else 0, on every row, keeping the row order and {os.path.basename(side_path)} beside it. Then "
           f"call estimate with sample_csv={path}.")
    if d in ("random", "condition"):
        nxt += " The same labels then serve certify."
    return {"conclusion": " ".join(said), "limits": _join(limits), "next": nxt,
            "files": {"sample_csv": path, "sidecar": side_path},
            "summary": {"design": d, "n_labelled": len(side["indices"]), "n_population": side["n_population"],
                        "per_condition": per, "seed": side.get("seed")}}


def _out_json(sample_csv, out_dir, suffix):
    if out_dir is None:
        return None
    out = _abs(out_dir)
    os.makedirs(out, exist_ok=True)
    stem = os.path.basename(sample_csv)
    stem = stem[:-4] if stem.endswith(".csv") else stem
    return os.path.join(out, f"{stem}_{suffix}.json")


def estimate(
    sample_csv: Annotated[str, P(description="The CSV that sample wrote, with `wrong` filled on every row; its .json "
                                             "sidecar must sit beside it")],
    per_class: Annotated[bool, P(description="Also each class's user's and producer's accuracy; needs a "
                                             "reference_class column")] = False,
    scores: Annotated[str | None, P(description="With per_class: the score raster sample was run on, if it has "
                                                "moved")] = None,
    nodata: Annotated[float | None, P(description="With per_class: the no-data value sample was run with, only for "
                                                  "a sample written by 1.2.0")] = None,
    out_dir: Annotated[str | None, P(description="Directory for the result JSON (default: beside the CSV)")] = None,
) -> dict[str, Any]:
    path = _input(sample_csv, "sample_csv")
    out = _out_json(path, out_dir, "estimate")
    argv = ["estimate", path]
    if per_class:
        argv.append("--per-class")
    if scores is not None:
        argv.append(f"--scores={_input(scores, 'scores')}")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if out is not None:
        argv.append(f"--out={out}")
    printed = _labelled(path, argv)
    out = out or (path[:-4] + "_estimate.json" if path.endswith(".csv") else path + "_estimate.json")
    r = _read_json(out)
    said = [f"The map's error rate is {_pc(r['estimate'])}, 95% interval {_pc(r['low'])} to {_pc(r['high'])}, from "
            f"{r['n_labelled']} labelled windows of {r['n_population']}; {r['method']}."]
    summ = {k: r.get(k) for k in ("estimate", "low", "high", "method", "design", "n_labelled", "n_population")}
    if r.get("by_condition"):
        rows = []
        for name, row in r["per_condition"].items():
            if row["estimate"] is None:
                rows.append(f"{name}: no labelled window fell in this condition, so nothing can be said about it")
            else:
                rows.append(f"{name} {_pc(row['estimate'])} ({_pc(row['low'])} to {_pc(row['high'])}), "
                            f"{row['n_labelled']} labelled of {row['n_population']} windows")
        said.append("Per input condition: " + "; ".join(rows) + ".")
        if r.get("outside_condition_intervals"):
            said.append(_cap(cli._outside_note(r)[len("note: "):]))
        summ["per_condition"] = {name: {k: row.get(k) for k in ("estimate", "low", "high", "n_labelled", "n_population",
                                                                "share_of_map")}
                                 for name, row in r["per_condition"].items()}
    class_warnings = []
    if per_class:
        oa = r["overall_accuracy"]
        said.append(f"Overall accuracy {_pc(oa['estimate'])} ({_pc(oa['low'])} to {_pc(oa['high'])}); each class's "
                    "user's and producer's accuracy, with intervals, is in summary.per_class.")
        summ["overall_accuracy"] = oa
        summ["per_class"] = {c: {**{k: row.get(k) for k in ("user_accuracy", "producer_accuracy", "reference_share",
                                                            "map_share")},
                                 **{k: row[k] for k in ("warning", "warning_codes") if k in row}}
                             for c, row in r["per_class"].items()}
        # the command prints each class's warning inside its table line, not on a `warning: ` line
        warned = {c: row for c, row in r["per_class"].items() if row.get("warning")}
        if warned:
            names = list(warned)
            which = (f"Class {names[0]} carries" if len(names) == 1 else
                     f"Classes {', '.join(names[:-1])} and {names[-1]} carry")
            codes = sorted({code for row in warned.values() for code in row.get("warning_codes", [])})
            said.append(f"{which} a warning ({', '.join(codes)}): read {'its' if len(names) == 1 else 'their'} "
                        "accuracies with the warning in limits.")
            class_warnings = [f"class {c}: {row['warning']}" for c, row in warned.items()]
    limits = ["Labels are assumed right: the interval describes agreement with the reviewer's labels. If the reviewer "
              "makes mistakes, the true rate can fall outside it.",
              "It is a rate over the windows; it does not say which windows are wrong."]
    limits += _warnings(_tagged(printed, "warning")) + _warnings([r.get("per_class_warning")]) + _warnings(class_warnings)
    limits += _notes([r.get("per_class_note"), r.get("condition_note"), r.get("scope")] + _tagged(printed, "note"))
    if r.get("by_condition"):
        nxt = (f"certify with sample_csv={path} and an alpha, such as 0.05, gives, for each input condition with "
               "enough labels, the most confident share of that condition whose error rate is at most alpha.")
    elif r["design"] in ("random", "condition"):
        nxt = (f"certify with sample_csv={path} and an alpha, such as 0.05, gives the most confident share of the map "
               "whose error rate is at most alpha.")
    else:
        nxt = (f"certify refuses a {r['design']} sample. To certify a zone, draw a new sample with design \"random\" "
               "and have it labelled.")
    if not per_class:
        nxt += " With a reference_class column (the class seen in each window), per_class=true gives each class's accuracy."
    return {"conclusion": " ".join(said), "limits": _join(limits), "next": nxt, "files": {"estimate": out},
            "summary": summ}


def certify(
    sample_csv: Annotated[str, P(description="The CSV that sample wrote with design \"random\" (or with condition), "
                                             "with `wrong` filled on every row")],
    alpha: Annotated[float, P(description="The error rate the certified zone may not exceed, such as 0.05")],
    delta: Annotated[float, P(description="The probability that the statement is wrong")] = est.ZONE_DELTA,
    rule: Annotated[Literal["prefix", "bonferroni"], P(description="prefix (default): fixed-sequence testing; "
                                                                   "bonferroni can certify more when the most "
                                                                   "confident windows hold many errors")] = "prefix",
    scores: Annotated[str | None, P(description="The score raster sample was run on, if it has moved")] = None,
    nodata: Annotated[float | None, P(description="The no-data value sample was run with, only for a sample written "
                                                  "by 1.2.0")] = None,
    out_dir: Annotated[str | None, P(description="Directory for the result JSON and the mask (default: beside the "
                                                 "CSV)")] = None,
) -> dict[str, Any]:
    path = _input(sample_csv, "sample_csv")
    out = _out_json(path, out_dir, "zone")
    argv = ["certify", path, f"--alpha={_num(float(alpha))}", f"--delta={_num(float(delta))}", f"--rule={rule}"]
    if scores is not None:
        argv.append(f"--scores={_input(scores, 'scores')}")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if out is not None:
        argv.append(f"--out={out}")
    printed = _labelled(path, argv)
    out = out or (path[:-4] + "_zone.json" if path.endswith(".csv") else path + "_zone.json")
    r = _read_json(out)
    files = {"zone": out}
    if r.get("zone_mask"):
        files["zone_mask"] = r["zone_mask"]
    summ = {k: r.get(k) for k in ("alpha", "delta", "rule", "n_labelled", "n_population", "min_labels_to_certify",
                                  "coverage", "n_zone", "threshold", "upper_bound")}
    a, d = 100 * float(alpha), 100 * float(delta)
    if r.get("by_condition"):
        lines = [ln for ln in printed.splitlines() if ln and not ln.startswith("wrote ")]
        per_line = [ln.strip() for ln in lines if ln.startswith("  ")]
        rest = [ln for ln in lines[1:] if not ln.startswith("  ")]
        said = [_cap(lines[0])[:-1] + " " + "; ".join(per_line) + "."] + [_cap(ln) for ln in rest]
        summ.update({k: r.get(k) for k in ("by_condition", "delta_per_condition", "n_conditions_tested",
                                           "certified_share_of_map", "n_certified")})
        summ["per_condition"] = {name: {k: e.get(k) for k in ("tested", "coverage", "n_zone", "n_population",
                                                               "n_labelled", "threshold", "upper_bound", "reason")}
                                 for name, e in r["per_condition"].items()}
        certified = r.get("certified_share_of_map") is not None
    elif r["coverage"] is not None:
        said = [f"The {100 * r['coverage']:.0f}% most confident windows ({r['n_zone']} of {r['n_population']}, "
                f"confidence margin >= {r['threshold']:.4f}) are wrong at most {a:g}% of the time. This statement "
                f"fails on at most {d:g}% of samples like this one ({r['rule']} rule; the exact upper bound on the "
                f"zone's error rate at that level is {_pc(r['upper_bound'])})."]
        certified = True
    else:
        # the package's note restates alpha, delta and the labels before its reason; the reason is what is new
        note = r["note"]
        if note.startswith("no zone certified") and "; " in note:
            reason = note.split("; ", 1)[1]
        elif "labels cannot certify any zone" in note and ": " in note:
            reason = note.split(": ", 1)[1]
        else:
            reason = None
        said = [f"No zone was certified at alpha {a:g}%, delta {d:g}%, from {r['n_labelled']} labels"
                + (f": {_sentence(reason)}" if reason else f". {_cap(note)}")]
        certified = False
    if certified and not r.get("by_condition"):
        said.append(f"The window mask {files['zone_mask']} marks the certified windows (True inside).")
    limits = ["Outside the certified windows nothing is certified.",
              "The rate holds for the certified windows together, not for each window.",
              "Labels are assumed right: the zone describes agreement with the reviewer's labels. The guarantee also "
              "needs the labelled windows to be a random sample."]
    # with no zone and no condition, the note is the conclusion's reason, said there once
    limits += _notes([r.get("note") if certified or r.get("by_condition") else None, r.get("scope")])
    if r.get("by_condition"):
        nxt = _condition_need(r, float(alpha), float(delta))
        if certified:
            nxt = ("The mask is a boolean array on the window grid. Outside it, each condition's own review set, from "
                   "assess with the condition layer, says which windows to check first."
                   + (" " + nxt if any(e.get("coverage") is None for e in r["per_condition"].values()) else ""))
    elif certified:
        nxt = ("The mask is a boolean array on the window grid. Outside it, the review sets of assess say which "
               "windows to check first.")
    else:
        nxt = (f"Any zone needs at least {r['min_labels_to_certify']} labels at this alpha and delta. A larger random "
               "sample (sample with design \"random\" and a larger budget, labelled in full) can certify more.")
    return {"conclusion": " ".join(said), "limits": _join(limits), "next": nxt, "files": files, "summary": summ}


def _condition_need(r, alpha, delta):
    """How many labels a condition needs before certify can say anything about it. A condition is tested only with
    min_labels_to_certify(alpha, delta) labels, and then at delta split over the conditions that hold that many
    (certify_by_condition), which needs more."""
    per = r["per_condition"]
    b1 = r["min_labels_to_certify"]
    held = sum(int(e["n_labelled"]) >= b1 for e in per.values())
    k = held or len(per)
    need = f"Each condition needs at least {b1} labels to be tested"
    if k > 1:
        bk = est.min_labels_to_certify(alpha, delta / k)
        if held:
            need += (f"; with delta split over the {held} conditions that hold that many ({delta / k:.3g} each), a "
                     f"zone in one needs at least {bk}")
        else:
            need += (f"; once {'both' if k == 2 else f'all {k}'} are tested, delta is split over them "
                     f"({delta / k:.3g} each) and a zone in one needs at least {bk}")
    return (need + ". A larger sample drawn with the same condition layer (sample with condition and a larger budget, "
            "labelled in full) can certify more.")


TOOLS = {"guide": guide, "assess": assess, "compare": compare, "sample": sample, "estimate": estimate,
         "certify": certify}


# ----------------------------------------------------------------------------- the server
def build_server():
    """The FastMCP server with the six tools, each described by its capability card. Needs the mcp extra."""
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ImportError(NEEDS_EXTRA) from exc
    import functools

    def as_tool(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Refused as exc:                       # the package's own message, as a tool error
                raise ToolError(str(exc)) from None
            except (OSError, ValueError) as exc:         # a file the package could not read, said plainly
                raise ToolError(f"{type(exc).__name__}: {exc}") from None
        return call

    server = FastMCP("oe-inferencex", instructions=INSTRUCTIONS)
    # FastMCP 1.x leaves the low-level server's version unset, and the handshake then gives the SDK's own version
    server._mcp_server.version = __version__
    for name, fn in TOOLS.items():
        hints = (ToolAnnotations(readOnlyHint=True, openWorldHint=False) if name == "guide" else
                 ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
        server.add_tool(as_tool(fn), name=name, description=CARDS[name], annotations=hints)
    return server


def main():
    """Serve on stdio until the agent closes the connection."""
    build_server().run("stdio")
    return 0

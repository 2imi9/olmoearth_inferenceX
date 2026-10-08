"""A local MCP server: the package's commands as tools that an agent calls on the user's own files.

    pip install "olmoearth-inferencex[geo,mcp]"            # 1.4.0 or later
    oe-inferencex mcp                                      # stdio; the agent starts it, nobody types into it
    claude mcp add --scope user oe-inferencex -- oe-inferencex mcp      # Claude Code, in every folder

Without installing the package, uv's `uvx` runs the server in an environment of its own:

    claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp

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
addition, not a change the trial tested. So are the user's questions (`QUESTIONS`) by which the instructions, the
`guide`, each tool's title and the first line of its card present the tools; the tool names stay as they were.

skills/oe-inferencex/SKILL.md holds the same teaching for an agent that runs the command line instead.

The texts and the tool functions import without the mcp extra; only `build_server` and `main` need it.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import threading
from typing import Annotated, Any, Literal

try:                                       # the annotations below are strings until FastMCP reads them (with mcp)
    from pydantic import Field
except ImportError:                        # pragma: no cover - pydantic comes with the mcp extra
    Field = None

from oe_inferencex import __version__, cli
from oe_inferencex import assess as _assess
from oe_inferencex import estimate as est
from oe_inferencex import decide as _decide

NEEDS_EXTRA = ("the MCP server needs the mcp extra: pip install \"olmoearth-inferencex[geo,mcp]\". The extra pins the "
               "MCP Python SDK below 2, because mcp 2 renamed FastMCP to MCPServer; an environment that holds mcp 2 "
               "needs its own install of the package")

# ----------------------------------------------------------------------------- the teaching
# The questions a user asks, in the standard order, and the tools that answer each. The tools keep their names (an
# agent's configuration and earlier calls use them); the question is in each tool's title and description.
LOOK, HOW_WRONG, TRUST, TWO_MAPS = ("Where should I look first?", "How wrong is the map?", "Which part can I trust?",
                                    "Which of two maps is better, and where do they differ?")
QUESTIONS = {LOOK: ("assess",), HOW_WRONG: ("sample", "estimate"), TRUST: ("certify",), TWO_MAPS: ("compare",)}

# A tool's title: the name a client may show in place of the tool's name.
TITLES = {
    "assess": LOOK,
    "sample": f"{HOW_WRONG} Step 1: pick the windows to label",
    "estimate": f"{HOW_WRONG} Step 2: the error rate from the labels",
    "certify": TRUST,
    "compare": TWO_MAPS,
    "guide": "How do I use these tools?",
    "decide": "Turn a result into a typed answer: yes, no or undetermined; a, b or tie; a share",
    "plan": "How many labels do I need? Plan the budget before labelling",
}

HARD_RULES = (
    "A review set is not a sample. It is chosen to hold errors, so its error rate overstates the map's. estimate and "
    "certify refuse it.",
    "certify needs a random sample: draw it with design \"random\" (certify once), \"sequential\" (labels can be "
    "added and certify run at every look) or with a condition layer. The default design serves estimate only; "
    "certify refuses it.",
    "Without labels, compare cannot say which map is right. Two maps that agree can both be wrong.",
    "Labels are assumed right. The interval and the zone describe agreement with the reviewer's labels; only estimate "
    "can widen its interval for a reviewer who errs, at rates the user states.",
    "Ranking needs the scores, not only the class map. A class map alone works only in compare. assess refuses a "
    "class map of more than two classes read as probabilities, but not a 0/1 map, nor any class map passed with "
    "logits=true: it reads the class ids as scores, and the order it gives is not evidence.",
    "An interval or a certificate holds for one sample whose size was fixed before labelling, with the rule, alpha "
    "and delta chosen before its labels are read. A second sample read after it, or a second run on the same labels "
    "with another rule, alpha or delta, is a second test: the chance that one of the answers is wrong can reach the "
    "sum of their error levels (10% for two 95% intervals, the sum of the deltas for two certificates).",
)

INSTRUCTIONS = "\n".join([
    "oe-inferencex checks a classification map made by an Earth-observation model. It reads the scores the model "
    "wrote out before the argmax: a GeoTIFF or .npy of shape (C, H, W), one band per class, or (H, W) for two "
    "classes; probabilities from 0 to 1, or logits with logits=true. It works on windows: square blocks of patch x "
    "patch pixels, 4 by default. The tools read and write local files only. Pass absolute paths.",
    "",
    "The questions the tools answer, in the standard order. Match the user's question to a step:",
    f"1. {LOOK} (assess) It ranks the windows from least to most confident. The review set is the least confident "
    "share, the windows to check first. No labels.",
    f"2. {HOW_WRONG} (sample, label, estimate) sample picks the windows to label. The user or a reviewer fills the "
    "`wrong` column with 1 or 0 on every row, or ? where a window cannot be judged. estimate gives the error rate with "
    "a 95% interval.",
    f"3. {TRUST} (certify) From the same labels, the most confident share of the map whose error rate is at most "
    "alpha. It can certify nothing, and then says why.",
    f"4. {TWO_MAPS} (compare) Where two maps of one area differ, window by window. Only with labels does it say which "
    "map is right where they differ. With few labels: sample with other set to the second map draws windows only "
    "where the maps differ, the reviewer writes the class seen in reference_class, and estimate says which map is more "
    "accurate and by how much.",
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
    "- State as fact only what a tool returned. Quote its conclusion and its limits. Each conclusion carries the "
    "limit that matters most; keep it in your reply.",
    "- Give an interval as its two ends. Never recompute it as p +/- 1.96 sqrt(p(1-p)/n).",
    "- A window's confidence ranks windows. It is not the probability that the window is wrong.",
    "- No tool labels windows, fetches labels, runs a model or knows whether labels exist. Labels come from the user "
    "or a reviewer.",
    "- Propose only what these tools can do, with their preconditions.",
    "- A refusal is a tool error carrying the package's reason. Change the input it names; do not retry the same "
    "call.",
    "",
    "Planning: before labelling, plan says how many labels a step needs, budget by budget: an error-rate interval no "
    "wider than a stated width (step 2), labels that name the more accurate of two maps when their accuracies differ "
    "by at least a stated amount (step 4), and for a zone the budget from which certify tests it, with, at a zone "
    "error rate the user states, the chance it is certified (step 3). It computes with the package's own procedures; "
    "the error rate a map will turn out to have is not known before the labels, so a plan made at a stated rate holds "
    "for that rate.",
    "",
    "Typed answers: decide reads a result that estimate, certify or compare wrote and answers set questions, each "
    "from a fixed set: error_rate_below=T (yes, no or undetermined), user_accuracy_above=T and "
    "producer_accuracy_above=T (per class), more_accurate (a, b or undetermined; a, b or tie from compare with "
    "labels), trusted_share (a share), trusted_share_at_least=S (yes or undetermined) and share_differs (a share). "
    "undetermined means the result does not settle the question; it is not a no. Quote each answer's because, which "
    "carries its limit.",
])

CARDS = {
    "plan": "\n".join([
        "Answers \"How many labels do I need?\" before any label, for the labelled steps.",
        "Does: for each budget on a ladder (10, 12, 15, 20, 25, ... up to 10000 or the census), the probability over "
        "the reviewer's random draw that the package's own procedure gives what is asked, and the smallest budget from "
        "which every larger one checked reaches power (default 0.9). With width, an error-rate interval no wider than "
        "width (high minus low), for a map wrong at error_rate (default 0.5). With other (or windows and differing) and "
        "difference, labels drawn where two maps differ that name the more accurate map when their accuracies differ "
        "by at least difference, at the worst split checked of the differing windows (two_class plans the one split of "
        "two-class maps). With "
        "coverage and alpha, the budget from which certify tests that zone at all and the budgets at which smaller "
        "zones enter; with zone_error, the rate the user expects that zone to be wrong, the Bonferroni rule's chance "
        "of certifying it (exact, however the errors spread), the most the prefix rule can reach (exact), and the "
        "prefix rule's chance when the more confident zones are wrong no more often (simulated).",
        "Needs: the map (as sample takes it) or windows, its number of valid windows; for two maps, the second map or "
        "differing. Fractions between 0 and 1 for width, difference, error_rate, coverage, alpha and zone_error.",
        "Cannot: know the error rate a map will turn out to have, which only labels measure, so a plan at a stated "
        "rate holds for that rate only; plan a zone or an error rate in the same call as two maps; draw the sample "
        "(sample does) or label windows. The recommended budget is not the fewest that can reach the probability: the "
        "probability does not rise smoothly with the budget.",
    ]),
    "guide": "\n".join([
        "How to use these tools: the questions they answer, the standard order, the hard rules and how to report.",
        "Does: returns the server's instructions, every tool's capability card under the question it answers, and "
        "the directory relative paths are read from.",
        "Needs: nothing.",
        "Cannot: run anything or read any file.",
    ]),
    "decide": "\n".join([
        "Turns a result into typed answers, so a yes or no question gets a yes, a no or an undetermined, never a "
        "paragraph to read.",
        "Does: reads a JSON that estimate, certify or compare wrote and answers each question asked with one answer "
        "from a fixed set, the level behind it, the evidence and a because sentence that holds the limit. "
        "error_rate_below=T: yes when the 95% interval lies below T, no when it lies at or above T, undetermined when "
        "it lies across T. user_accuracy_above=T and producer_accuracy_above=T: the same per class, from estimate with "
        "per_class. more_accurate: a, b or undetermined from estimate on a sample drawn with other; a, b or tie from "
        "compare with labels when every differing window carries a label. trusted_share: the share certify "
        "certified, 0 when none. trusted_share_at_least=S: yes or undetermined, never no. share_differs: the share of "
        "windows where two maps differ. A result with input conditions gets each condition's answer too.",
        "Needs: the result JSON (estimate's, certify's or compare's comparison.json) and the questions, as names or "
        "name=value with a value between 0 and 1.",
        "Cannot: add evidence the result does not hold. It refuses more_accurate on a comparison made without labels, "
        "since compare cannot say which map is right without them, and answers undetermined when labels cover only "
        "part of the differing windows. undetermined means the result does not settle the question; it is not a no.",
    ]),
    "assess": "\n".join([
        f"Answers \"{LOOK}\": which windows of one map to check first. No labels.",
        "Does: reads a score raster, splits it into windows of patch x patch pixels and ranks them from least to "
        "most confident (with order boundary_first, the windows on a class boundary first, then the interior, each "
        "from least to most confident). Writes the review sets (the first 1%, 5% and 10% of the order by default) as "
        "CSVs with pixel and map coordinates, suspicion and boundary rasters, explanation.json (the cues behind each flagged window) "
        "and assessment.json. With condition, it also ranks each input condition on its own. With reference, it "
        "grades the order against that raster, taken as truth.",
        "Needs: the model's scores before the argmax: (C, H, W) per-class probabilities, or (H, W) for two classes, "
        "as GeoTIFF or .npy; logits=true for logits. Or, as published products ship them, a class map of integer ids "
        "with confidence set to its per-pixel confidence band (higher = more confident) and confidence_range to the "
        "band's values that are confidences, so codes are left out. A GeoTIFF needs rasterio (the geo extra). An "
        "output directory.",
        "Cannot: say how wrong the map is (that needs labels: sample, then estimate); rank a class map alone. One of more "
        "than two classes read as probabilities is refused, but a 0/1 map, or any class map passed with logits=true, "
        "is read as scores: its windows tie and the order is not evidence. It cannot find the errors the model is "
        "sure of, which come last, or label windows.",
    ]),
    "compare": "\n".join([
        f"Answers \"{TWO_MAPS}\": where two maps of the same area differ, window by window, and with labels "
        "which map is right there.",
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
        f"Answers \"{HOW_WRONG}\", step 1: which windows to label, so that estimate can give the map's error rate.",
        "Does: draws budget windows and writes them to a CSV with an empty `wrong` column for the reviewer, and a "
        ".json sidecar holding the design, which estimate and certify read. With other (a second map of the same "
        "grid) it draws only windows where the two maps differ, with an empty `reference_class` column, for estimate "
        "to say which map is more accurate. Designs: confidence (the default without "
        "condition: stratified by confidence; estimate reads it, certify refuses it), random (serves estimate and "
        "certify; the default for a product's class map with confidence), sequential (a random order labelled from "
        "the top; extend adds windows later, keeping the labels, and certify holds at every look), proportional, "
        "tiles (a cluster design for labelling tile by tile) and condition (the default with condition: labels split "
        "equally across the input conditions).",
        "Needs: the score raster assess takes, or a product's class map with confidence; a budget (the recorded "
        "experiments used 300); an output directory.",
        "Cannot: label the windows, fetch labels or say whether labels exist (the user or a reviewer fills "
        "`wrong`); turn a review set into a sample.",
    ]),
    "estimate": "\n".join([
        f"Answers \"{HOW_WRONG}\", step 2: the map's error rate with a 95% interval, from the labelled sample.",
        "Does: reads the CSV that sample wrote, once `wrong` holds 1 or 0 on every row, and its .json sidecar. Gives "
        "the error rate with the interval the sample's design earns, and each input condition's rate when the sample "
        "recorded a condition. With per_class, each class's user's and producer's accuracy and its error-adjusted "
        "share of the map. On a sample drawn with other (reference_class filled), it says instead which map is more "
        "accurate and by how much; no reviewer rate or per_class applies there.",
        "Needs: the CSV sample wrote, filled on every row (1, 0, or ? where a window cannot be judged) and kept in "
        "its order, with its .json beside it. Windows marked ? are bounded both ways, so the estimate is a range. "
        "reviewer_false_alarm and reviewer_miss, when the user knows how often the reviewer errs, widen the interval "
        "for it. per_class needs a `reference_class` column (the class seen in each window), every row judged, no "
        "reviewer rate, a sample of any design but tiles, and the map's scores (from the sidecar, or scores).",
        "Cannot: estimate from a review set or from any CSV that sample did not write (refused); check that the "
        "labels are right (they are assumed right); say which windows are wrong.",
    ]),
    "certify": "\n".join([
        f"Answers \"{TRUST}\": which share of the map, from the most confident window down, is wrong at most alpha "
        "of the time, with a guarantee.",
        "Does: from the labelled sample, finds the largest most-confident share of the map whose error rate is at "
        "most alpha; the statement fails on at most delta of the samples that could have been drawn. Writes the "
        "result JSON and a window mask (.npy, True inside the zone). A sample drawn with condition is certified per "
        "condition. It may certify nothing, and then says why.",
        "Needs: a sample drawn with design \"random\" (or with condition), labelled on every row (? counts as wrong, "
        "which keeps the guarantee), or with design \"sequential\", labelled from the top row down as far as the "
        "reviewer has gone; alpha, such as 0.05; the map's scores (from the sidecar, or scores). On a random sample "
        "the guarantee is for one run; on a sequential sample it holds at every look, from a smaller zone outward "
        "(the anchor, fixed when the sample was drawn), at the cost of more labels.",
        "Cannot: certify from the default confidence design, a review set or windows chosen by hand (refused); say "
        "anything about the windows outside the zone; check that the labels are right, or allow for a reviewer who "
        "misses errors (that would need the miss rate inside every zone it can certify).",
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
        # an agent that guessed the base of a relative path needs to know it (the agent test of 2 October 2026)
        where = "" if os.path.isabs(os.path.expanduser(str(path))) else f" (relative paths are read from {os.getcwd()})"
        raise Refused(f"{what}: {p} is not a file{where}")
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
    return [f"Warning: {str(_mcp_warning(w)).strip()}" for w in texts if w is not None and str(w).strip()]


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



# The package's texts are written for the command line and name its options. Through this server the agent sets
# parameters instead, so each text returned here names the parameter (the agent test of 2 October 2026: outputs that
# said "--design random" or "--condition" sent a small model looking for options it does not have).
_FLAGS = {"design", "condition", "condition-names", "confidence", "confidence-range", "per-class", "labels-date", "patch", "nodata", "scores", "alpha",
          "delta", "rule", "budget", "seed", "labels", "reference", "threshold", "date-a", "date-b", "groups", "order",
          "budgets", "tile", "per-tile", "reviewer-false-alarm", "reviewer-miss", "other", "threshold", "windows",
          "differing", "width", "error-rate", "difference", "both-wrong", "coverage", "zone-error", "power",
          "max-labels", "anchor", "extend"}


def _mcp_words(text):
    """A package text with each command-line option written as this server's parameter."""
    if not text:
        return text
    # an option starts a word: a path such as /data/run--patch8 or a name such as one-sided--seed is left alone
    start = r"(?<![\w/.\-])"
    text = re.sub(start + r"without --logits\b", "with logits=false", text)
    text = re.sub(start + r"--(design|rule|order)[ =]([a-z_]+)", r'\1="\2"', text)
    text = re.sub(start + r"--logits\b", "logits=true", text)
    text = re.sub(start + r"--out\b(?!-)", "out_dir", text)
    return re.sub(start + r"--([a-z][a-z-]*)",
                  lambda m: m.group(1).replace("-", "_") if m.group(1) in _FLAGS else m.group(0), text)


# The margin warning names a Python argument and a command-line route, neither of which an MCP caller has.
MCP_MARGIN_WARNING = ("multi-class logit margin: these logits are ranked by the gap between the two highest. On Ai2's "
                      "suite one minus the top probability ranked errors better on 14 of 16 multi-class tasks (exp76). "
                      "To rank by the top probability here, pass the class probabilities (the softmax of the logits) "
                      "with logits=false")
# The scope notes quote exp88's numbers at length; through this server each is one sentence, and the JSON the tool
# writes keeps the full note.
MCP_SCOPE = {
    "assess": ("If part of the map was predicted from inputs the model was not trained on, the model can be "
               "confidently wrong there and those errors come late in this order; pass a raster of each pixel's input "
               "condition as condition to rank each part on its own (exp88)."),
    "estimate": ("This is the whole map's rate. A part predicted from other inputs can err at a very different rate; "
                 "draw the sample with condition to get each part's rate (exp88)."),
    "certify": ("The zone's rate holds over all its windows. A part predicted from other inputs can be wrong more "
                "often inside it; draw the sample with condition to certify each part on its own (exp88)."),
}


def _mcp_warning(text):
    return MCP_MARGIN_WARNING if str(text).strip() == _assess.MARGIN_FORM_WARNING else text


def _reply(conclusion, limits, nxt, files, summary):
    """A tool's result, every text in this server's words."""
    return {"conclusion": _mcp_words(conclusion), "limits": _mcp_words(limits), "next": _mcp_words(nxt),
            "files": files, "summary": summary}


def _grid_shape(path):
    """The (rows, columns) of a raster the package wrote on the window grid, or None if it cannot be read."""
    try:
        if str(path).endswith(".npy"):
            import numpy as np
            return tuple(np.load(path, mmap_mode="r").shape[-2:])
        import rasterio
        with rasterio.open(path) as src:
            return (src.height, src.width)
    except Exception:                      # pragma: no cover - a shape we cannot read only drops a caveat
        return None

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
                   "scores, or the class map with its confidence band (confidence).")
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
CONFIDENCE_PARAM = ("For a published product: its per-pixel confidence band on the map's grid, rising with confidence "
                    "(LCMAP's lcpconf); negate an uncertainty band first. scores is then the product's class map of "
                    "integer ids, and nodata applies to it")
CONFIDENCE_RANGE_PARAM = ("With confidence: [low, high], the band's values that are confidences; values outside are left "
                          "out as no-data and every statement is about the rest. Without it codes cannot be told from "
                          "confidences (LCMAP: [1, 100], since lcpconf holds provenance codes from 151)")


def _product_argv(confidence, confidence_range):
    """The command-line options for a class map with its confidence band; the range is refused without the band."""
    argv = []
    if confidence is not None:
        argv.append(f"--confidence={_input(confidence, 'confidence')}")
    if confidence_range is not None:
        if len(confidence_range) != 2:
            raise Refused(f"confidence_range needs two numbers, [low, high]; got {confidence_range}")
        argv += ["--confidence-range", *[_num(float(v)) for v in confidence_range]]
    return argv


def guide() -> str:
    """The server's instructions and every tool's capability card, under the question the tool answers."""
    groups = [*QUESTIONS.items(), (TITLES["plan"], ("plan",)), (TITLES["decide"], ("decide",)),
              (TITLES["guide"], ("guide",))]
    cards = "\n\n".join(f"## {question}\n\n" + "\n\n".join(f"### {name}\n{CARDS[name]}" for name in names)
                        for question, names in groups)
    return (f"{INSTRUCTIONS}\n\nRelative paths are read from {os.getcwd()}.\n\n# Capability cards, by question\n\n"
            f"{cards}\n\nolmoearth-inferencex {__version__}")


def assess(
    scores: Annotated[str, P(description="The score raster: GeoTIFF or .npy, (C, H, W) per class or (H, W) for two "
                                         "classes, taken before the argmax; with confidence, the product's class map")],
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
    confidence: Annotated[str | None, P(description=CONFIDENCE_PARAM)] = None,
    confidence_range: Annotated[list[float] | None, P(description=CONFIDENCE_RANGE_PARAM)] = None,
) -> dict[str, Any]:
    scores = _input(scores, "scores")
    out = _abs(out_dir)
    argv = ["assess", scores, f"--out={out}", f"--patch={int(patch)}", f"--order={order}"]
    argv += _product_argv(confidence, confidence_range)
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
    # boundary_first lists the boundary windows first, then the interior, each by confidence; the texts below said
    # "least confident first" for it too (2026-10-06), and on a map whose least confident windows are interior the
    # 5% set held none of them while the reply said to check the least confident windows first
    boundary_first = s.get("review_order") == "boundary_first"
    sets = ", ".join(f"{rs[b]['n_windows']} ({_budget(b)})" for b in rs)
    # a review set most of whose windows tie at its cut-off is ordered by raster position: a class map read as
    # scores (a 0/1 map, or class ids passed as logits) does this, and the package does not refuse it
    tied = {b: rs[b]["tied_at_cutoff"] for b in rs if rs[b].get("tied_at_cutoff")}
    loose = [b for b, t in tied.items() if 2 * t["inside"] > rs[b]["n_windows"]]
    if loose:
        bt = b5 if b5 in loose else loose[0]
        opener = (f"The order is not evidence here: {tied[bt]['inside']} of the {rs[bt]['n_windows']} windows of the "
                  f"{_budget(bt)} review set share the cut-off score with {tied[bt]['outside']} windows left outside, "
                  "so among them the order is raster position. "
                  + ("A product's confidence band this coarse ranks only the windows it separates."
                     if confidence else
                     "A class map passed with logits=true, or a 0/1 map, is read as scores and ties like this; if the "
                     "file is one, pass the model's per-class scores instead."))
    else:
        opener = (("Check the windows on a class boundary first, then the interior, each least confident first."
                   if boundary_first else "Check the least confident windows first.")
                  + " This ranks the windows; it does not say how wrong the map is, and labels on the review set do "
                  "not give the error rate, because the review set is not a sample.")
    # a .npy carries no georeferencing, so the CSVs' x and y are empty (the agent test of 2 October 2026)
    where = "pixel and map coordinates" if not scores.lower().endswith(".npy") else (
        "pixel coordinates (a .npy has no georeferencing, so x and y are empty)")
    listed = (f"with {where}, the windows on a class boundary first and then the interior, each part least confident "
              "first" if boundary_first else f"least confident first with {where}")
    said = [f"{opener} Of {s['n_windows']} windows of {s['patch_px']} x "
            f"{s['patch_px']} pixels, the review sets hold {sets} windows, each listed {listed}. The {_budget(b5)} "
            f"review set is in {files.get(f'review_set_{b5}')}; files names the others."]
    boundary = f"{_pc(s['boundary_window_fraction'])} of all windows sit on a class boundary of the map."
    grid = _grid_shape(files.get("boundary", ""))
    line = grid is not None and min(grid) == 1 and max(grid) > 1
    if not line:
        said.append(boundary)
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
              ("Errors the model is sure of away from a class boundary come last in this order, so a review of the "
               "review set does not find them." if boundary_first else
               "Errors the model is sure of come last in this order, so a review of the review set does not find them."),
              ("In the review-set CSVs, the boundary column (above 0 on a class boundary) says which part a window is "
               "in, and confidence orders the windows within each part, higher meaning more confident: "
               if boundary_first else
               "In the review-set CSVs, confidence is the score the order ranks by, higher meaning more confident: ")
              + ("the product's confidence band, as given, averaged over the window." if confidence else
                 "the gap between the two highest logits, which has no upper bound." if logits else
                 "read from the class probabilities.") + " The suspicion raster holds minus that score, so higher is "
              "more suspect; it ranks the windows as summary.signal does"
              + (", by confidence alone, not in this order: the boundary raster marks the windows put first."
                 if boundary_first else ".")]
    if line:
        limits.append(boundary[:-1] + f", but the window grid is one window {'high' if grid[0] == 1 else 'wide'}, so "
                      "this share counts neighbours along one line only and says little about the map.")
    ref = s.get("against_reference") or {}
    if reference is not None:
        cap = ref.get("error_capture_at_budget") or {}
        if cap and int(ref.get("n_windows_scored") or 0) < int(s["n_windows"]):
            # The capture ranks only the windows the reference grades (assess.py), so it is not the review set's: with
            # a reference on a quarter of the quick-start map "the 5% review set holds 67%" was said of a set that
            # holds 35% of them (2026-10-06). Said of the set it is measured on.
            n_ref, k = int(ref["n_windows_scored"]), int(cap[b5]["n_reviewed"])
            said.append(f"Against the reference, which grades {n_ref} of the {s['n_windows']} windows, "
                        f"{_pc(ref.get('error_rate'))} of those {n_ref} disagree with it; the first {k} of them in the "
                        f"review order ({_budget(b5)} of them) hold {_pc(cap[b5]['errors_captured_fraction'], 0)} of "
                        f"those disagreements. That is not the {_budget(b5)} review set, which is drawn from every "
                        "window.")
        elif cap:
            said.append(f"Against the reference, {_pc(ref.get('error_rate'))} of the scored windows disagree with it; "
                        f"the {_budget(b5)} review set holds {_pc(cap[b5]['errors_captured_fraction'], 0)} of those "
                        "disagreements.")
        if cap:     # the capture is measured on the graded windows, which with a partial reference are not all of them
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
    scope = s.get("scope")
    limits += _notes([MCP_SCOPE["assess"] if scope == _assess.SCOPE_ASSESS else scope] + _tagged(printed, "note"))
    n = int(s["n_windows"])
    budget = ("a budget (the recorded experiments used 300)" if n > 300 else
              f"a budget of up to {n}: the map has {n} windows, and labelling all of them gives its exact rate")
    nxt = (f"For how wrong the map is: sample with design \"random\" and {budget}, have the user or a reviewer fill "
           "`wrong` on every row, then estimate; the same labels then serve certify.")
    if condition is None:
        nxt += " If a raster records each pixel's input condition, pass it as condition to assess and to sample."
    return _reply(" ".join(said), _join(limits), nxt, files, summ)


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
    caveats = []
    if enr["boundary_a"] is not None:
        line = (f"The differing windows sit on a class boundary of map a {enr['boundary_a']:.2f} times as often as the "
                "agreeing windows" + (f" (of map b, {enr['boundary_b']:.2f} times)." if enr["boundary_b"] is not None
                                      else "."))
        grid = _grid_shape(files.get("disagreement", ""))
        if grid is not None and min(grid) == 1 and max(grid) > 1:
            caveats.append(line[:-1] + f", but the window grid is one window {'high' if grid[0] == 1 else 'wide'}, so "
                           "boundaries are counted along one line only and this says little about the maps.")
        else:
            said.append(line)
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
        # Across dates the labels' date favours one map: the other is counted wrong wherever the ground changed. The
        # shares above were quoted without it, though decide on the same file says it (2026-10-06).
        if s["dates"]["status"] in ("different_time", "overlapping_time") and graded.get("graded_against"):
            limits.append(_cap(graded["graded_against"]))
    else:
        limits = [HARD_RULES[2]]
        # the limit a small model dropped when it named the better map anyway (the agent test of 2 October 2026)
        said.append("Without labels this says neither which map is better nor which is right where they differ.")
    if s["dates"]["status"] in ("different_time", "overlapping_time", "partly_stated"):
        said.append(s["dates"]["reading"][0].upper() + s["dates"]["reading"][1:] + ".")
    limits += ["A difference between the maps says neither map's error rate."] + caveats + _notes(s.get("notes", []))
    if not s["n_disagree"]:
        # sample with other refuses two maps that differ nowhere, and labels grade them alike: proposing either sent
        # the agent into a refusal (2026-10-06)
        nxt = ("The maps differ in no window, so neither is more accurate on the windows compared and there is nothing "
               "to label between them. For either map's own error rate: sample on that map's scores, label, estimate.")
    elif not graded:
        nxt = ("To learn which map is more accurate with few labels: sample on map a with other set to map b draws "
               "windows only where they differ; a reviewer writes the class seen in each, and estimate says which map "
               "is more accurate and by how much. With a label raster on the same grid, compare again with labels. For "
               "either map's own error rate: sample on that map's scores, label, estimate.")
    else:
        nxt = "For either map's error rate: sample on that map's scores, label, estimate."
    return _reply(" ".join(said), _join(limits), nxt, files, summ)


def sample(
    scores: Annotated[str, P(description="The score raster assess takes; with confidence, the product's class map")],
    out_dir: Annotated[str, P(description="Directory to write the CSV and its .json sidecar into; created if needed")],
    budget: Annotated[int, P(description="Number of windows to label (the recorded experiments used 300)")],
    design: Annotated[Literal["confidence", "proportional", "random", "sequential", "tiles", "condition"] | None,
                      P(description="Sampling design. Default: condition with a condition layer, random with "
                                    "confidence (a product's band), confidence otherwise. Use random to certify from "
                                    "the same labels once; sequential to label from the top, add windows later "
                                    "(extend) and certify at every look")] = None,
    anchor: Annotated[float | None, P(description="sequential design: the smallest zone certify will test, as a "
                                                  "share of the map, fixed now before any label; or give alpha")]
    = None,
    alpha: Annotated[float | None, P(description="sequential design, without anchor: the error rate certify will be "
                                                 "asked about; the anchor is then the zone this budget can certify if "
                                                 "none of its labels is wrong, at most 0.25 of the map")] = None,
    delta: Annotated[float | None, P(description="sequential design, with alpha: the delta the anchor is computed "
                                                 "for (default 0.1)")] = None,
    extend: Annotated[str | None, P(description="sequential design: a sample CSV to extend to budget windows, keeping "
                                                "its labels (the same scores, patch and seed)")] = None,
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
    other: Annotated[str | None, P(description="A second map of the same grid: sample only the windows where the two "
                                               "maps differ, to learn which is more accurate")] = None,
    threshold: Annotated[float | None, P(description="With other: cut-off of a 2-D continuous map (0.5 for a "
                                                     "probability map when omitted)")] = None,
    confidence: Annotated[str | None, P(description=CONFIDENCE_PARAM)] = None,
    confidence_range: Annotated[list[float] | None, P(description=CONFIDENCE_RANGE_PARAM)] = None,
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
    _opt(argv, "--anchor", None if anchor is None else float(anchor))
    _opt(argv, "--alpha", None if alpha is None else float(alpha))
    _opt(argv, "--delta", None if delta is None else float(delta))
    if extend is not None:
        argv.append(f"--extend={_input(extend, 'extend')}")
    argv += _product_argv(confidence, confidence_range)
    if logits:
        argv.append("--logits")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if condition is not None:
        argv.append(f"--condition={_input(condition, 'condition')}")
    if condition_names:
        argv += _condition_argv(condition_names)
    _opt(argv, "--threshold", None if threshold is None else float(threshold))      # refused without other
    if other is not None:
        argv.append(f"--other={_input(other, 'other')}")
        _run(argv)
        side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
        side = _read_json(side_path)
        D, N = side["n_disagree"], side["n_population"]
        said = [f"{len(side['indices'])} windows to label of the {D} where the two maps differ ({_pc(D / N)} of the {N} "
                f"windows compared), listed in {path}; the design is in {side_path}. No window is labelled yet."]
        limits = ["This sample says which map is more accurate and by how much once labelled, not either map's accuracy, "
                  "and certify refuses it.", "The windows compared are those both maps predict where neither splits "
                  "evenly between two classes.", "Labels are assumed right; no reviewer error rate applies to this "
                  "comparison."] + _notes(side.get("notes", []))
        nxt = (f"Ask the user or a reviewer to open each window of {path} and write in `reference_class` the class that "
               "is there, in the maps' class codes, or ? where it cannot be judged, keeping every row (hide class_a and "
               f"class_b to label blind). Then call estimate with sample_csv={path}.")
        return _reply(" ".join(said), _join(limits), nxt, {"sample_csv": path, "sidecar": side_path},
                      {"design": "disagreement", "n_labelled": len(side["indices"]), "n_disagree": D,
                       "n_population": N, "seed": side.get("seed")})
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
    if d == "sequential":
        with open(path, newline="", encoding="utf-8-sig") as f:
            kept = sum(1 for r in csv.DictReader(f) if str(r.get("wrong", "")).strip() != "")
        said.append(f"{kept} of them are labelled (kept from the sample extended)." if kept else
                    "No window is labelled yet, so nothing is known about the error rate until the user or a reviewer "
                    "fills `wrong` from the top row down.")
    else:
        said.append("No window is labelled yet, so nothing is known about the error rate until the user or a reviewer "
                    "fills `wrong` on every row.")
    limits = ["No window is labelled yet: no tool labels windows, fetches labels or knows whether labels exist."
              if d != "sequential" else "No tool labels windows, fetches labels or knows whether labels exist.",
              HARD_RULES[3] + " To label blind, hide map_class, write the class seen in a reference_class column and "
              "set wrong where the two differ."]
    if d == "sequential":
        limits.append("Both estimate and certify read the rows labelled from the top. certify's statement holds at "
                      "every look, however often it is run as labels are added; estimate's interval assumes the number "
                      "of labels was fixed before labelling.")
    elif d in ("random", "condition"):
        limits.append("Both estimate and certify read this sample.")
    else:
        limits.append(f"This sample serves estimate, and certify refuses the {d} design. To certify a zone, sample "
                      "with design \"random\".")
    limits += _warnings(_tagged(printed, "warning")) + _notes(_tagged(printed, "note"))
    nxt = (f"Ask the user or a reviewer to open each window of {path} and set `wrong` to 1 if map_class is not what "
           f"is there, else 0, or ? where the window cannot be judged, on every row, keeping the row order and "
           f"{os.path.basename(side_path)} beside it. Then call estimate with sample_csv={path}.")
    if d in ("random", "condition"):
        nxt += " The same labels then serve certify."
    if d == "sequential":
        nxt = (f"Ask the user or a reviewer to open the windows of {path} from the top row down and set `wrong` to 1 if "
               "map_class is not what is there, else 0, or ? where the window cannot be judged, keeping the row order "
               f"and {os.path.basename(side_path)} beside it. Call certify with sample_csv={path} after any number of "
               "rows; to add windows, call sample again with design \"sequential\", the same seed, a larger budget and "
               f"extend={path}.")
    return _reply(" ".join(said), _join(limits), nxt, {"sample_csv": path, "sidecar": side_path},
                  {"design": d, "n_labelled": len(side["indices"]), "n_population": side["n_population"],
                   "per_condition": per, "seed": side.get("seed"),
                   **({"anchor": side.get("anchor"), "first_budget": side.get("first_budget")} if d == "sequential"
                      else {})})


def _out_json(sample_csv, out_dir, suffix):
    if out_dir is None:
        return None
    out = _abs(out_dir)
    os.makedirs(out, exist_ok=True)
    stem = os.path.basename(sample_csv)
    stem = stem[:-4] if stem.endswith(".csv") else stem
    return os.path.join(out, f"{stem}_{suffix}.json")


def estimate(
    sample_csv: Annotated[str, P(description="The CSV that sample wrote, with `wrong` filled on every row (or, for a "
                                             "sample drawn with other, `reference_class`); its .json sidecar must sit "
                                             "beside it")],
    per_class: Annotated[bool, P(description="Also each class's user's and producer's accuracy; needs a "
                                             "reference_class column")] = False,
    scores: Annotated[str | None, P(description="With per_class: the score raster sample was run on, if it has "
                                                "moved")] = None,
    confidence: Annotated[str | None, P(description="With per_class, for a product's class map: the confidence "
                                                    "band sample was run on, if it has moved")] = None,
    nodata: Annotated[float | None, P(description="With per_class: the no-data value sample was run with, only for "
                                                  "a sample written by 1.2.0")] = None,
    out_dir: Annotated[str | None, P(description="Directory for the result JSON (default: beside the CSV)")] = None,
    reviewer_false_alarm: Annotated[float | None, P(description="At most this share of the truly correct windows "
                                                                "does the reviewer mark wrong, as the user states it")] = None,
    reviewer_miss: Annotated[float | None, P(description="At most this share of the truly wrong windows does the "
                                                         "reviewer mark right, as the user states it")] = None,
) -> dict[str, Any]:
    path = _input(sample_csv, "sample_csv")
    out = _out_json(path, out_dir, "estimate")
    argv = ["estimate", path]
    _opt(argv, "--reviewer-false-alarm", None if reviewer_false_alarm is None else float(reviewer_false_alarm))
    _opt(argv, "--reviewer-miss", None if reviewer_miss is None else float(reviewer_miss))
    if per_class:
        argv.append("--per-class")
    if scores is not None:
        argv.append(f"--scores={_input(scores, 'scores')}")
    if confidence is not None:
        argv.append(f"--confidence={_input(confidence, 'confidence')}")
    _opt(argv, "--nodata", None if nodata is None else float(nodata))
    if out is not None:
        argv.append(f"--out={out}")
    printed = _labelled(path, argv)
    out = out or (path[:-4] + "_estimate.json" if path.endswith(".csv") else path + "_estimate.json")
    r = _read_json(out)
    if r.get("design") == "disagreement":
        return _which_map(r, out)
    rate = (f"between {_pc(r['estimate_range'][0])} and {_pc(r['estimate_range'][1])}" if r.get("estimate") is None
            else _pc(r["estimate"]))
    stated = bool(r.get("reviewer_false_alarm") or r.get("reviewer_miss"))
    unjudged = bool(r.get("n_unjudged"))
    trust = (("It bounds the windows that could not be judged both ways. " if unjudged else "")
             + ("It allows for a reviewer who errs at most as often as the user stated." if stated else
                "It is the rate of disagreement with the reviewer's labels, which are assumed right."))
    said = [f"The map's error rate is {rate}, 95% interval {_pc(r['low'])} to {_pc(r['high'])}, from "
            f"{r['n_labelled']} labelled windows of {r['n_population']}; {r['method']}. {trust}"]
    summ = {k: r.get(k) for k in ("estimate", "low", "high", "method", "design", "n_labelled", "n_population")}
    for k in ("estimate_range", "n_unjudged", "reviewer_false_alarm", "reviewer_miss", "labels_interval"):
        if r.get(k) is not None:
            summ[k] = r[k]
    if r.get("by_condition"):
        rows = []
        for name, row in r["per_condition"].items():
            if not row["n_labelled"]:
                rows.append(f"{name}: no labelled window fell in this condition, so nothing can be said about it")
            else:
                value = (f"{_pc(row['estimate_range'][0])} to {_pc(row['estimate_range'][1])}"
                         if row.get("estimate") is None else _pc(row["estimate"]))
                rows.append(f"{name} {value} ({_pc(row['low'])} to {_pc(row['high'])}), "
                            f"{row['n_labelled']} labelled of {row['n_population']} windows")
        said.append("Per input condition: " + "; ".join(rows) + ". Each condition's interval is its own 95% "
                    "statement; the intervals do not hold jointly.")
        if r.get("outside_condition_intervals"):
            said.append(_cap(cli._outside_note(r)[len("note: "):]))
        summ["per_condition"] = {name: {k: row.get(k) for k in ("estimate", "estimate_range", "low", "high", "n_labelled",
                                                                "n_population", "share_of_map")}
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
    limits = ([_cap(r["bounds_note"])] if r.get("bounds_note") else []) + (
        ["Labels are assumed right: the interval describes agreement with the reviewer's labels. If the reviewer "
         "makes mistakes, the true rate can fall outside it; reviewer_false_alarm and reviewer_miss widen it for a "
         "reviewer who errs as often as the user states."] if not (r.get("reviewer_false_alarm") or r.get("reviewer_miss"))
        else []) + ["It is a rate over the windows; it does not say which windows are wrong."]
    limits += _warnings(_tagged(printed, "warning")) + _warnings([r.get("per_class_warning")]) + _warnings(class_warnings)
    scope = r.get("scope")
    # the note on conditions outside the whole-map interval is said in the conclusion; once is enough
    outside = cli._outside_note(r)[len("note: "):] if r.get("outside_condition_intervals") else None
    limits += _notes([r.get("per_class_note"), r.get("condition_note"),
                      MCP_SCOPE["estimate"] if scope == est.SCOPE_ESTIMATE else scope]
                     + [t for t in _tagged(printed, "note") if t != outside])
    # certify tests a zone only with min_labels_to_certify labels in it; say so before suggesting it (the agent
    # test of 2 October 2026: estimate pointed to certify where no condition could be certified at alpha 0.05)
    b1 = est.min_labels_to_certify(0.05, est.ZONE_DELTA)
    if r.get("by_condition"):
        most = max(int(row["n_labelled"]) for row in r["per_condition"].values())
        nxt = (f"certify with sample_csv={path} and the alpha chosen before the labels were read (such as 0.05) "
               "gives, for each input condition with "
               "enough labels, the most confident share of that condition whose error rate is at most alpha.")
        if most < b1:
            nxt += (f" At alpha 0.05 and delta {est.ZONE_DELTA:g} a condition needs at least {b1} labels and the most "
                    f"any holds is {most}, so it would certify nothing there; a looser alpha (for a new sample, chosen before its "
                    "labels are read) needs fewer labels, and a larger sample drawn with the same condition layer can "
                    "certify more. " + _decide.certificate_second_look(est.ZONE_DELTA, tested=False))
    elif r["design"] in ("random", "condition"):
        nxt = (f"certify with sample_csv={path} and the alpha chosen before the labels were read (such as 0.05) "
               "gives the most confident share of the map "
               "whose error rate is at most alpha.")
        if int(r["n_labelled"]) < b1:
            nxt += (f" At alpha 0.05 and delta {est.ZONE_DELTA:g} a zone needs at least {b1} labels and this sample "
                    f"has {r['n_labelled']}, so it would certify nothing; a looser alpha (for a new sample, chosen "
                    "before its labels are read) needs fewer labels.")
    else:
        nxt = (f"certify refuses a {r['design']} sample. To certify a zone, draw a new sample with design \"random\" "
               "and have it labelled.")
    if not per_class:
        # per_class refuses a tiles sample, ? rows and a stated reviewer rate; proposing it on such a sample with only
        # the reference_class column as its precondition led straight to a refusal (2026-10-06)
        if r["design"] == "tiles":
            nxt += (" Each class's accuracy (per_class) needs a sample of another design (random, confidence, "
                    "proportional or condition): it is not graded on a tiles sample.")
        else:
            need = [said for said, applies in (("every row judged (no ?)", unjudged),
                                               ("no reviewer rate, the labels taken as right", stated)) if applies]
            nxt += (" With a reference_class column (the class seen in each window)"
                    + "".join(f", {said}" for said in need) + ", per_class=true gives each class's accuracy.")
    return _reply(" ".join(said), _join(limits), nxt, {"estimate": out}, summ)


def _which_map(r, out):
    """estimate's reply on a sample of the windows where two maps differ."""
    d, v = r["difference"], r["verdict"]
    if v == "a":
        head = (f"Map a is more accurate than map b, by {100 * d['low']:.1f} to {100 * d['high']:.1f} points over the "
                "windows compared (95% interval).")
    elif v == "b":
        head = (f"Map b is more accurate than map a, by {-100 * d['high']:.1f} to {-100 * d['low']:.1f} points over the "
                "windows compared (95% interval).")
    else:
        head = ("The labels cannot tell which map is more accurate: the difference, a minus b, lies between "
                f"{100 * d['low']:+.1f} and {100 * d['high']:+.1f} points (95% interval).")
    said = [head, f"From {r['n_labelled']} labelled windows of the {r['n_disagree']} where the maps differ "
            f"({_pc(r['disagree_share'])} of the {r['n_population']} windows compared): a right on {r['n_a_right']}, "
            f"b right on {r['n_b_right']}, neither on {r['n_neither']}"
            + (f", {r['n_unjudged']} could not be judged." if r["n_unjudged"] else ".")]
    limits = [r["note"], "The interval is a union bound over the two maps' shares of the differing windows, each exact, "
              "so it covers the accuracy difference at least 95% of the time; it is not either map's accuracy."] + \
        _notes(r.get("notes", []))
    nxt = ("For either map's own accuracy: sample with design \"random\" on that map's scores, have a reviewer label "
           "every row, then estimate." if v else
           "A larger sample drawn with other (a new seed gives a new draw; label all of it) narrows the interval. "
           + _decide.comparison_second_look(lead=False))
    summ = {k: r.get(k) for k in ("verdict", "difference", "n_labelled", "n_disagree", "n_population", "n_a_right",
                                  "n_b_right", "n_neither", "n_unjudged", "share_a_right", "share_b_right")}
    return _reply(" ".join(said), _join(limits), nxt, {"estimate": out}, summ)


def certify(
    sample_csv: Annotated[str, P(description="The CSV that sample wrote with design \"random\" (or with condition), "
                                             "with `wrong` filled on every row, or with design \"sequential\", "
                                             "filled from the top row down")],
    alpha: Annotated[float, P(description="The error rate the certified zone may not exceed, such as 0.05")],
    delta: Annotated[float, P(description="The probability that the statement is wrong")] = est.ZONE_DELTA,
    rule: Annotated[Literal["prefix", "bonferroni"], P(description="prefix (default): fixed-sequence testing; "
                                                                   "bonferroni can certify more when the most "
                                                                   "confident windows hold many errors")] = "prefix",
    scores: Annotated[str | None, P(description="The score raster sample was run on, if it has moved")] = None,
    confidence: Annotated[str | None, P(description="For a product's class map: the confidence band sample was run "
                                                    "on, if it has moved")] = None,
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
    if confidence is not None:
        argv.append(f"--confidence={_input(confidence, 'confidence')}")
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
    if r.get("rule") == "sequential":
        summ.update({"anchor": r.get("anchor"), "n_drawn": r.get("n_drawn")})
    a, d = 100 * float(alpha), 100 * float(delta)
    if r.get("by_condition"):
        lines = [ln for ln in printed.splitlines() if ln and not ln.startswith("wrote ")]
        per_line = [ln.strip() for ln in lines if ln.startswith("  ")]
        rest = [ln for ln in lines[1:] if not ln.startswith("  ")]
        said = [_cap(lines[0])[:-1] + " " + "; ".join(per_line) + "."] + [_cap(ln) for ln in rest]
        if r.get("certified_share_of_map") is not None:
            said.append("Each rate holds for a condition's certified windows as a group, not for each window; outside "
                        f"them nothing is certified. delta ({d:g}% here) is chosen before the labels are read; a lower one gives a "
                        "stronger statement.")
        summ.update({k: r.get(k) for k in ("by_condition", "delta_per_condition", "n_conditions_tested",
                                           "certified_share_of_map", "n_certified")})
        summ["per_condition"] = {name: {k: e.get(k) for k in ("tested", "coverage", "n_zone", "n_population",
                                                               "n_labelled", "threshold", "upper_bound", "reason",
                                                               "n_tied_at_threshold", "n_tied_inside_zone")}
                                 for name, e in r["per_condition"].items()}
        certified = r.get("certified_share_of_map") is not None
    elif r["coverage"] is not None and r.get("rule") == "sequential":
        said = [f"Taken together, the {100 * r['coverage']:.0f}% most confident windows ({r['n_zone']} of "
                f"{r['n_population']}, confidence >= {r['threshold']:.4f}) are wrong at most {a:g}% of the time. The "
                "rate holds for them as a group, not for each window, and outside them nothing is certified. This "
                f"statement fails on at most {d:g}% of samples like this one however often certify is run on it as "
                f"labels are added (delta, which can be set lower; sequential rule, from the anchor zone, "
                f"{_pc(r['anchor'])} of the map, outward)."]
        certified = True
    elif r["coverage"] is not None:
        # "are wrong at most 5% of the time" was read as a promise for each window, and delta as fixed (the agent
        # test of 2 October 2026)
        said = [f"Taken together, the {100 * r['coverage']:.0f}% most confident windows ({r['n_zone']} of "
                f"{r['n_population']}, confidence >= {r['threshold']:.4f}) are wrong at most {a:g}% of the time. The "
                "rate holds for them as a group, not for each window, and outside them nothing is certified. This "
                f"statement fails on at most {d:g}% of samples like this one (delta, chosen before the labels were read; "
                f"{r['rule']} rule; the exact upper bound on this zone's error rate is "
                f"{_pc(r['upper_bound'])})."]
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
              ((_cap(r["bounds_note"]) + " ") if r.get("bounds_note") else "")
              + "Labels are assumed right: the zone describes agreement with the reviewer's labels. The guarantee also "
              "needs the labelled windows to be a random sample."]
    # with no zone and no condition, the note is the conclusion's reason, said there once
    scope = r.get("scope")
    # A product's range leaves pixels out, and "N of M windows" counts only the rest: the whole-map reply dropped
    # the note that says so (2026-10-06). Per condition it comes in the printed lines of the conclusion.
    limits += _notes([r.get("note") if certified or r.get("by_condition") else None,
                      None if r.get("by_condition") else r.get("population_note"),
                      MCP_SCOPE["certify"] if scope == est.SCOPE_CERTIFY else scope])
    if r.get("by_condition"):
        # A condition's zone can end inside a block of windows tied at its threshold (a saturated or integer band),
        # and "margin >= threshold" then takes in windows the zone leaves out. The whole map's note says so; the
        # per-condition reply did not (2026-10-06), and read as a rule the threshold took in 819 uncertified
        # windows, all wrong.
        for name, e in r["per_condition"].items():
            tied, inside = int(e.get("n_tied_at_threshold") or 0), int(e.get("n_tied_inside_zone") or 0)
            if e.get("coverage") is not None and tied > inside:
                limits.append(f"Note: in condition {name}, {tied} windows share the threshold score and only {inside} "
                              "of them are inside its zone, so the zone is the set in the window mask, not every "
                              "window of the condition at or above the threshold.")
    if r.get("by_condition"):
        nxt = _condition_need(r, float(alpha), float(delta))
        if certified:
            nxt = ("The mask is a boolean array on the window grid. Outside it, each condition's own review set, from "
                   "assess with the condition layer, says which windows to check first."
                   + (" " + nxt if any(e.get("coverage") is None for e in r["per_condition"].values()) else ""))
    elif r.get("rule") == "sequential":
        nxt = (f"{r['n_labelled']} of the {r.get('n_drawn', r['n_labelled'])} windows drawn are labelled. Label more "
               "from the top, or extend the sample (sample with design \"sequential\", the same seed, a larger budget "
               "and extend), and call certify again: every look is covered by the same guarantee. A zone with no "
               f"error among its labels needs at least {r['min_labels_to_certify']} of them, and the anchor zone "
               f"({_pc(r['anchor'])} of the map) must be certified first.")
        if certified:
            nxt = ("The mask is a boolean array on the window grid. More labels, from the top of the same sample, can "
                   "only extend the zone outward; outside it, the review sets of assess say which windows to check "
                   "first.")
    elif certified:
        nxt = ("The mask is a boolean array on the window grid. Outside it, the review sets of assess say which "
               "windows to check first.")
    else:
        nxt = (f"Any zone needs at least {r['min_labels_to_certify']} labels at this alpha and delta. A larger random "
               "sample (sample with design \"random\" and a larger budget, labelled in full) can certify more. "
               + _decide.certificate_second_look(delta, int(r["n_labelled"]) >= int(r["min_labels_to_certify"])))
    return _reply(" ".join(said), _join(limits), nxt, files, summ)


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
            "labelled in full) can certify more. " + _decide.certificate_second_look(delta, bool(r.get("n_conditions_tested"))))


def decide(
    result_json: Annotated[str, P(description="A JSON that estimate, certify or compare wrote")],
    questions: Annotated[list[str], P(description="The questions, each a name or name=value: error_rate_below=0.1, "
                                                  "user_accuracy_above=0.85, producer_accuracy_above=0.85, "
                                                  "more_accurate, trusted_share, trusted_share_at_least=0.5, "
                                                  "share_differs")],
    out_dir: Annotated[str | None, P(description="Directory for the decisions JSON (default: beside the result)")]
    = None,
) -> dict[str, Any]:
    path = _input(result_json, "result_json")
    out = None
    if out_dir is not None:
        os.makedirs(_abs(out_dir), exist_ok=True)
        stem = os.path.basename(path)
        stem = stem[:-5] if stem.endswith(".json") else stem
        if stem == "comparison":                 # every compare writes comparison.json: name it by its directory
            stem = os.path.basename(os.path.dirname(path)) + "_comparison"
        out = os.path.join(_abs(out_dir), stem + "_decisions.json")
    argv = ["decide", path, *[f"--ask={q}" for q in questions]]
    if out is not None:
        argv.append(f"--out={out}")
    _run(argv)
    out = out or (path[:-5] + "_decisions.json" if path.endswith(".json") else path + "_decisions.json")
    r = _read_json(out)
    said, told = [], set()
    # each answer's `because` carries its own caveat on a second look (1.7.1); in one reply it is said once
    delta_r = next((a["evidence"].get("delta") for a in r["answers"].values()
                    if isinstance(a.get("evidence"), dict) and a["evidence"].get("delta") is not None), None)
    caveats = [_decide.interval_second_look(True), _decide.interval_second_look(False),
               _decide.comparison_second_look()]
    if delta_r is not None:
        caveats += [_decide.certificate_second_look(delta_r, True), _decide.certificate_second_look(delta_r, False)]
    for q, a in r["answers"].items():
        shown = _pc(a["answer"]) if a["type"] == "score" else a["answer"]
        because = a["because"]
        for c in caveats:
            if c in because:
                if c in told:
                    because = because.replace(" " + c, "").replace(c, "")
                told.add(c)
        said.append(f"{q}: {shown}. {because}")
    rkind = r["result_kind"]
    limits = ("Each answer is read from the result file and adds no evidence of its own. undetermined means the result "
              "does not settle the question; it is not a no.")
    # A zone certified per condition splits delta over the conditions tested, so its statements hold together; an
    # estimate's per-condition and per-class intervals are each 95% on their own. One sentence said the latter of
    # both (2026-10-06), against the zone's own note and the certify reply.
    if rkind == "estimate":
        limits += (" Answers per class or per condition each rest on their own interval, and the intervals do not "
                   "hold jointly.")
    elif rkind == "zone" and any(a.get("per_condition") for a in r["answers"].values()):
        delta = next(a["evidence"].get("delta") for a in r["answers"].values() if a.get("per_condition"))
        limits += (" The conditions' certificates hold together: delta is split over the conditions tested, so all "
                   f"their statements hold except on at most {_pc(delta)} of samples.")
    pending = [q for q, a in r["answers"].items()
               if a["answer"] == "undetermined"
               or any(x.get("answer") == "undetermined" for x in (a.get("per_class") or {}).values())
               or any(x.get("answer") == "undetermined" for x in (a.get("per_condition") or {}).values())]
    # what can settle an undetermined answer depends on the result: a comparison has no sample to enlarge and no
    # interval to narrow, and "a larger sample drawn the same way" sent the agent looking for one (2026-10-06)
    more = {"comparison": ("the labels raster does not cover every differing window; sample on map a with other set "
                           "to map b, have a reviewer write reference_class on every row, then estimate, and ask "
                           "more_accurate of that estimate."),
            "zone": ("a larger random sample (sample with design \"random\", or with the same condition layer, and a "
                     "larger budget), labelled in full, can certify more.")}
    seq = rkind == "zone" and any((a.get("evidence") or {}).get("rule") == "sequential" for a in r["answers"].values())
    if seq:       # a sequential sample is added to, not replaced; every look is covered, so no second test
        more["zone"] = ("more windows of this sequential sample, labelled from the top (or after sample --extend), "
                        "can certify more, and certify may be run again at every look under the same guarantee.")
    cost = ""
    if pending and rkind == "zone" and not seq:
        # the zone's own answer says what a new sample costs, or that it is the first test that can certify anything;
        # per-condition answers carry theirs in the JSON alone
        said_cost = [c for c in told if "second test" in c]
        said_first = [c for c in told if "first test" in c]
        if said_cost:
            cost = " That is a second test, at the cost said above."
        elif not said_first:
            cost = " " + _decide.certificate_second_look(delta_r if delta_r is not None else est.ZONE_DELTA, True)
    elif pending and rkind != "comparison":
        # Each pending question costs what its own undetermined intervals cost: its own sentence says it (pointed to
        # above), or its per-condition rows say it (exact or nominal, as each row's sentence does), or its per-class
        # rows are nominal. A row with no labels or no interval costs nothing: a sample that reaches it is its first
        # test. A cost already said above is pointed to, not said again.
        covered, bare = [], {True: [], False: []}
        for q in pending:
            a = r["answers"][q]
            if any(c in a["because"] for c in caveats):
                covered.append(q)
                continue
            kinds = set()
            for x in (a.get("per_condition") or {}).values():
                if x.get("answer") == "undetermined":
                    kinds |= {k for k in (True, False) if _decide.interval_second_look(k) in x.get("because", "")}
            for x in (a.get("per_class") or {}).values():
                if x.get("answer") == "undetermined" and x.get("low") is not None:
                    kinds.add(False)
            for k in kinds:
                (covered if _decide.interval_second_look(k) in told else bare[k]).append(q)
        covered = list(dict.fromkeys(covered))
        # every cost sentence names its questions unless it covers all the pending ones; a question whose undetermined
        # parts cost nothing (no label, no interval) is never folded into another's cost
        costed = set(covered) | set(bare[True]) | set(bare[False])
        groups = sum(bool(g) for g in (covered and told, bare[True], bare[False]))
        named = groups > 1 or costed != set(pending)
        parts = []
        if covered and told:
            parts.append(("For " + ", ".join(covered) + ", that is" if named else "That is")
                         + " a second test, at the cost said above.")
        for k in (True, False):
            if bare[k]:
                c = _decide.interval_second_look(k, lead=False)
                parts.append(("For " + ", ".join(dict.fromkeys(bare[k])) + ", " + c[0].lower() + c[1:]) if named else c)
        cost = (" " + " ".join(parts)) if parts else ""
    nxt = (("For " + ", ".join(pending) + ": " + more.get(rkind, "a larger sample drawn the same way, labelled in "
                                                                "full, narrows the interval.") + cost) if pending else
           "The result can also answer: " + ", ".join(r["available"]) + ".")
    summ = {"result_kind": r["result_kind"], "answers": {q: a["answer"] for q, a in r["answers"].items()},
            "per_condition": {q: a["per_condition"] for q, a in r["answers"].items() if a.get("per_condition")} or None,
            "per_class": {q: {c: {k: row.get(k) for k in ("answer", "low", "high", "warning")}
                              for c, row in a["per_class"].items()}
                          for q, a in r["answers"].items() if a.get("per_class")} or None}
    return _reply(" ".join(said), limits, nxt, {"decisions": out}, summ)


def plan(
    scores: Annotated[str | None, P(description="The map, as sample takes it; leave out when windows is given")] = None,
    other: Annotated[str | None, P(description="A second map of the same grid: plan the labels that say which map is "
                                               "more accurate")] = None,
    windows: Annotated[int | None, P(description="The map's number of valid windows, instead of the map")] = None,
    differing: Annotated[int | None, P(description="With windows: the windows where two maps differ, instead of the "
                                                   "two maps")] = None,
    width: Annotated[float | None, P(description="Error rate: the widest 95% interval acceptable, high minus low, as a "
                                                 "fraction (0.1 is 10 points)")] = None,
    error_rate: Annotated[float | None, P(description="Error rate: the rate the user expects the map to have "
                                                      "(default 0.5, where intervals are widest)")] = None,
    difference: Annotated[float | None, P(description="Two maps: the smallest whole-map accuracy difference worth "
                                                      "detecting, as a fraction (0.02 is 2 points)")] = None,
    both_wrong: Annotated[float | None, P(description="Two maps: the share of the differing windows wrong in both "
                                                      "(default: the worst split checked)")] = None,
    two_class: Annotated[bool, P(description="Two maps: both have only two classes, so where they differ one is right")]
    = False,
    coverage: Annotated[float | None, P(description="Zone: the share of the map to certify, a grid level from 0.05 to "
                                                    "1 in steps of 0.05")] = None,
    alpha: Annotated[float | None, P(description="Zone: the error rate the zone must not exceed")] = None,
    zone_error: Annotated[float | None, P(description="Zone: the rate the user expects that zone to be wrong")] = None,
    power: Annotated[float, P(description="The probability of the outcome planned for")] = 0.9,
    max_labels: Annotated[int | None, P(description="The largest budget checked (default 10000, or the census when "
                                                    "smaller)")] = None,
    patch: Annotated[int | None, P(description="Window side in pixels, as sample will use (default 4; only with a "
                                               "map)")] = None,
    logits: Annotated[bool, P(description="True if the scores are logits rather than probabilities")] = False,
    nodata: Annotated[float | None, P(description="No-data value (default: the raster's own, plus NaN)")] = None,
    threshold: Annotated[float | None, P(description="With other: cut-off of a 2-D continuous map")] = None,
    confidence: Annotated[str | None, P(description=CONFIDENCE_PARAM)] = None,
    confidence_range: Annotated[list[float] | None, P(description=CONFIDENCE_RANGE_PARAM)] = None,
    out_dir: Annotated[str | None, P(description="Directory to write plan.json into (default: none written)")] = None,
) -> dict[str, Any]:
    import tempfile
    argv = ["plan"]
    if scores is not None:
        argv.append(_input(scores, "scores"))
    if other is not None:
        argv.append(f"--other={_input(other, 'other')}")
    for flag, value in (("--windows", windows), ("--differing", differing), ("--width", width),
                        ("--error-rate", error_rate), ("--difference", difference), ("--both-wrong", both_wrong),
                        ("--coverage", coverage), ("--alpha", alpha), ("--zone-error", zone_error),
                        ("--power", power), ("--max-labels", max_labels), ("--nodata", nodata),
                        ("--threshold", threshold)):
        _opt(argv, flag, value)
    if patch is not None:
        argv.append(f"--patch={int(patch)}")
    if two_class:
        argv.append("--two-class")
    if logits:
        argv.append("--logits")
    argv += _product_argv(confidence, confidence_range)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(_abs(out_dir), "plan.json") if out_dir is not None else os.path.join(tmp, "plan.json")
        if out_dir is not None:
            os.makedirs(_abs(out_dir), exist_ok=True)
        printed = _run([*argv, f"--out={out}"])
        r = _read_json(out)
    said = [line for line in printed.splitlines() if line.split(":", 1)[0] in ("Error rate", "Which map", "Zone")]
    plans = r["plans"]
    summ = {}
    for q, p in plans.items():
        if q == "zone":
            summ[q] = {"labels_to_test": p.get("labels_to_test"), "entry_budgets": p.get("entry_budgets"),
                       **{k: (p[k].get("labels") if isinstance(p.get(k), dict) else None)
                          for k in ("bonferroni", "prefix_even")}}
        else:
            summ[q] = {"labels": p.get("labels"), "probability_at_labels": p.get("probability_at_labels")}
    limits = _join([*r.get("notes", []), next(iter(plans.values()))["note"] if plans else None])
    nxt = ("Draw the labels with sample at the budget planned: design \"random\" for an error rate or a zone, or other "
           "set to the second map for which map is more accurate. Every guarantee holds at any budget; what the plan "
           "gives is the chance of a useful answer, which for certify can fall at the budgets at which a smaller zone "
           "becomes testable, until more labels restore it.")
    return _reply(" ".join(said), limits, nxt, {"plan": out} if out_dir is not None else {}, summ)


TOOLS = {"guide": guide, "assess": assess, "compare": compare, "sample": sample, "estimate": estimate,
         "certify": certify, "decide": decide, "plan": plan}


# ----------------------------------------------------------------------------- the server
def build_server():
    """The FastMCP server with its tools, each described by its capability card and titled by the question it
    answers. The title is set twice: the tool's own (protocol 2025-06-18) and its annotations' (read by clients of
    the earlier protocol). Needs the mcp extra."""
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
                raise ToolError(_mcp_words(str(exc))) from None
            except (OSError, ValueError) as exc:         # a file the package could not read, said plainly
                raise ToolError(f"{type(exc).__name__}: {exc}") from None
        return call

    server = FastMCP("oe-inferencex", instructions=INSTRUCTIONS)
    # FastMCP 1.x leaves the low-level server's version unset, and the handshake then gives the SDK's own version
    server._mcp_server.version = __version__
    for name, fn in TOOLS.items():
        hints = (ToolAnnotations(title=TITLES[name], readOnlyHint=True, openWorldHint=False) if name == "guide" else
                 ToolAnnotations(title=TITLES[name], readOnlyHint=False, destructiveHint=True, idempotentHint=True,
                                 openWorldHint=False))
        server.add_tool(as_tool(fn), name=name, title=TITLES[name], description=CARDS[name], annotations=hints)
    return server


def main():
    """Serve on stdio until the agent closes the connection."""
    build_server().run("stdio")
    return 0

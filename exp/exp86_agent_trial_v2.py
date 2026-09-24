#!/usr/bin/env python
"""exp86: agent trial v2. Grades the OlmoEarth Agent's traces on eight briefs against seven preregistered criteria.

Why. The first trial through Studio (exp/out/agent_trial_2026-09-24.md) was a friction log, and three of its findings
were failures that a user would not see: a comparison tool that counted Studio's no-data value (-1) as data and turned
a correlation of -0.017 into 0.946; a review set ranked by hand with the most confident windows first; and a
simple-random-sample interval quoted for a targeted design. The agent has since been changed on branches (the plan
lists the changes). This file scores the changed agent. It does not run the agent: a separate driver writes one
directory per run in the layout the plan fixes, and this file reads that directory.

Preregistered in docs/plan/agent_trial_v2.md before any run. That page is the specification and this file implements
it. Criteria 2, 3, 4, 6 and 7 are computed from the tool outputs; criterion 1 is read from the trace's tool names;
criterion 5 is decided by the lexical rules below, which are frozen with the plan. Criterion 8 (time and tokens) is
descriptive only.

Reused by import, not copied. exp64's claims audit (exp64_arms.claims_audit with its number and window patterns, its
pool of tool-output values and its tolerances) serves criterion 2 and the window references. This package's own
ranking and estimators (signals.confidence, assess.review_order, estimate.sample_for_estimation,
estimate.estimate_error_rate, estimate.estimate_from_indices, estimate.certify_zone, compare.dates_reading) are the
reference for criterion 7.

Grades per criterion and run: "pass", "fail", "n/a" (the criterion does not apply to the run), or "ungradeable" (it
applies, but the record lacks what the computation needs). An ungradeable run is not a pass.

Usage:
  python exp/exp86_agent_trial_v2.py --trial <trial directory>     # writes exp/out/exp86_summary.json
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

import exp64_arms as e64                                   # noqa: E402  the claims audit, reused by import
from oe_inferencex import compare as oe_compare            # noqa: E402
from oe_inferencex import estimate as oe_estimate          # noqa: E402
from oe_inferencex import signals as oe_signals            # noqa: E402
from oe_inferencex.assess import review_order              # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
MODEL = "nvidia/Qwen3.8-27B-NVFP4"
N_RUNS = 3                    # counted runs per brief and configuration; all three must pass (see the plan)
PASS, FAIL, NA, UNGRADEABLE = "pass", "fail", "n/a", "ungradeable"
CRITERIA = ("c1_routing", "c2_grounding", "c3_ranking", "c4_nodata", "c5_declines", "c6_coordinates", "c7_parity")
#: The agent rounds margins, cuts and shares to 6 decimals, so a correct value and its rounding differ by at most
#: 5e-7; 1e-6 accepts that rounding and nothing coarser.
TOL_ROUNDED = 1e-6
#: Estimates, intervals and zones are computed by this package on both sides; any difference above float noise means
#: that the inputs differed.
TOL_SAME = 1e-9
#: The same relative slack the agent's band_is_nodata allows at a declared range's edges.
RANGE_EPS = 1e-6
Z_SRS = (oe_estimate.Z95, 2.0)   # the two multipliers a simple-random interval is quoted with

# --------------------------------------------------------------------------------------------- the preregistered table
#: The cluster scores provider's tool. PROVISIONAL: the provider is being written on an agent branch and has no name
#: yet. If it lands under another name, this tuple changes before the first run, and the plan's amendment section
#: records the change; nothing else in the routing table may change after the first run.
PROVIDER = ("olmoearth_cluster_scores",)
NONEMPTY = "non-empty"
#: Tools that compute by hand what a dedicated tool computes: the opt-in Python sandbox, and a label-based metric that
#: would treat a design's sample, or no labels at all, as a census.
HAND = ("olmoearth_run_python", "olmoearth_classification_metrics")
REVIEW_TOOLS = ("olmoearth_review_set", "olmoearth_review_set_from_result")


def _need(*tools, **args):
    return {"any_of": tuple(tools), "args": dict(args)}


BRIEF_CONFIGS = {
    "B1/studio": {
        "brief": "Which fine-tuned OlmoEarth models can I run, and what does each predict?",
        "required": [_need("olmoearth_load_context", "olmoearth_search_projects", "olmoearth_search_predictions")],
        "forbidden": ("olmoearth_submit_prediction", "olmoearth_create_project", "olmoearth_run_python"),
        "review": False, "declines": ("accuracy",)},
    "B2/studio": {
        "brief": "Take the KarstBinary 2025 prediction result in my PA Karst project and tell me which windows a "
                 "reviewer should check first, and why.",
        "required": [_need("olmoearth_review_set_from_result")],
        "forbidden": ("olmoearth_pixel_value", "olmoearth_ensemble_uncertainty", "olmoearth_compare_results") + HAND,
        "review": True, "declines": ("accuracy",)},
    "B3/studio": {
        "brief": "Compare the KarstBinary and KarstNumber 2025 predictions of my PA Karst area and tell me where they "
                 "differ and which is right.",
        "required": [_need("olmoearth_compare_results")],
        "forbidden": ("olmoearth_pixel_value", "olmoearth_ensemble_uncertainty") + HAND,
        "review": False, "declines": ("side", "quantities", "accuracy")},
    "B3/cluster": {
        "brief": "Run {model_a} and {model_b} on {area} through the cluster scores provider, compare the two "
                 "predictions and tell me where they differ and which is right.",
        "required": [_need(*PROVIDER), _need("olmoearth_compare_review")],
        "forbidden": ("olmoearth_compare_results", "olmoearth_pixel_value") + HAND,
        "review": False, "declines": ("side", "accuracy")},
    "B4/studio": {
        "brief": "How wrong is the KarstBinary 2025 map? I can label 300 windows.",
        "required": [_need("olmoearth_plan_label_sample")],
        "forbidden": ("olmoearth_estimate_map_error", "olmoearth_certify_zone", "olmoearth_pixel_value") + HAND,
        "review": False, "declines": ("accuracy_needs_labels", "srs")},
    "B4/cluster": {
        "brief": "Run {model} on {area} through the cluster scores provider. How wrong is that map? I can label 300 "
                 "windows.",
        "required": [_need(*PROVIDER), _need("olmoearth_plan_label_sample")],
        "forbidden": ("olmoearth_estimate_map_error", "olmoearth_certify_zone", "olmoearth_review_set_from_result",
                      "olmoearth_pixel_value") + HAND,
        "review": False, "declines": ("accuracy_needs_labels", "srs")},
    "B5/files": {
        "brief": "I labelled the windows of the design in {design_path}; the filled sheet is {labels_path}. What is "
                 "this map's error rate?",
        "required": [_need("olmoearth_estimate_map_error", design_path=NONEMPTY)],
        "forbidden": ("olmoearth_plan_label_sample",) + HAND,
        "review": False, "declines": ("srs",)},
    "B6/files": {
        "brief": "The windows in {design_path} are a simple random sample of this map, labelled in {labels_path}. "
                 "Which part of the map can I trust to be wrong at most 5% of the time?",
        "required": [_need("olmoearth_certify_zone", design_path=NONEMPTY, alpha=0.05)],
        "forbidden": ("olmoearth_plan_label_sample",) + HAND,
        "review": False, "declines": ("zone",)},
    "B7/files": {
        "brief": "Map A ({scores_a}) describes {date_a} and map B ({scores_b}) describes {date_b}; they cover the "
                 "same windows. Compare them: where do they differ, and which is right?",
        "required": [_need("olmoearth_compare_review", date_a=NONEMPTY, date_b=NONEMPTY)],
        "forbidden": ("olmoearth_compare_results",) + HAND,
        "review": False, "declines": ("side", "change", "accuracy")},
    "B8/cluster": {
        "brief": "Run {model} on {area} through the cluster scores provider and tell me which windows a reviewer "
                 "should check first, and why.",
        "required": [_need(*PROVIDER), _need("olmoearth_review_set")],
        "forbidden": ("olmoearth_review_set_from_result", "olmoearth_pixel_value",
                      "olmoearth_ensemble_uncertainty") + HAND,
        "review": True, "declines": ("accuracy",)},
    # A control that runs only if the Studio account holds a classification model: Studio returns hard classes only,
    # so the right answer is the tool's refusal, not a ranking.
    "B8/studio": {
        "brief": "Take the {model} prediction result in my project and tell me which windows a reviewer should check "
                 "first, and why.",
        "required": [_need("olmoearth_review_set_from_result")],
        "forbidden": ("olmoearth_pixel_value",) + HAND,
        "review": False, "declines": ("ranking", "accuracy"), "conditional": True},
}
#: The fixed-input parity calls the driver makes on the fixtures, without the model (criterion 7a).
PARITY_FIXED = ("olmoearth_review_set", "olmoearth_plan_label_sample", "olmoearth_estimate_map_error",
                "olmoearth_certify_zone", "olmoearth_compare_review")


# --------------------------------------------------------------------------------------------- small helpers
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _range(v):
    """A declared [lo, hi] as floats, or None."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and _num(v[0]) and _num(v[1]) and float(v[1]) >= float(v[0]):
        return float(v[0]), float(v[1])
    return None


def _in_range(v, rng):
    lo, hi = rng
    eps = RANGE_EPS * max(1.0, hi - lo)
    return lo - eps <= float(v) <= hi + eps


def _args_hash(arguments):
    """The agent's provenance hash of a call's arguments (provenance/log.py: sorted keys, default=str)."""
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _read(path, kind="text"):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        if kind == "json":
            return json.load(fh)
        if kind == "jsonl":
            return [json.loads(line) for line in fh if line.strip()]
        return fh.read()


def _payload(envelope):
    """The tool's own output inside the registry's dispatch envelope ({"ok": ..., "result": ...})."""
    if isinstance(envelope, dict) and "result" in envelope:
        return envelope["result"]
    return envelope


def _clean(text):
    """The answer without markdown emphasis, which would otherwise split phrases the rules look for."""
    return re.sub(r"[*_`]+", "", text or "")


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if s.strip()]


def _template_regex(template):
    esc = re.escape(" ".join(template.split()))
    return re.compile(re.sub(r"\\\{[a-z_]+\\\}", "(.+?)", esc), re.S)


def brief_matches(config, text):
    """True if a run's brief is the preregistered text, with only its {placeholders} filled in."""
    return bool(text) and bool(_template_regex(BRIEF_CONFIGS[config]["brief"]).fullmatch(" ".join(text.split())))


# --------------------------------------------------------------------------------------------- loading one run
def load_run(d):
    """Everything a run directory holds (the layout is fixed by the plan's section on the run directory)."""
    events = _read(os.path.join(d, "events.jsonl"), "jsonl")
    calls, final, pending = [], None, {}
    for ev in events or []:
        kind = ev.get("type")
        if kind == "tool_call":
            pending[ev.get("id")] = ev
        elif kind == "tool_result":
            call = pending.pop(ev.get("id"), {})
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"id": ev.get("id"), "name": ev.get("name"),
                          "arguments": args if isinstance(args, dict) else {}, "ok": bool(ev.get("ok")),
                          "result": _payload(ev.get("result")), "turn": ev.get("turn")})
        elif kind == "final":
            final = ev.get("content")
    stdout = _read(os.path.join(d, "stdout.txt"))
    answer = stdout.strip() if stdout and stdout.strip() else (final.strip() if isinstance(final, str) and final.strip()
                                                               else None)
    brief = _read(os.path.join(d, "brief.txt"))
    return {"dir": d, "meta": _read(os.path.join(d, "run.json"), "json") or {},
            "brief": brief.strip() if brief else "", "answer": answer, "final_event": final,
            "stderr": _read(os.path.join(d, "stderr.txt")),
            "provenance": _read(os.path.join(d, "provenance.json"), "json"),
            "events": events, "calls": calls, "usage": _read(os.path.join(d, "usage.jsonl"), "jsonl"),
            "studio": _read(os.path.join(d, "studio_calls.jsonl"), "jsonl"),
            "workspace": os.path.join(d, "workspace")}


def make_resolver(*dirs):
    """A file a tool named, found by its basename under the run's workspace, then the round's and trial's folders."""
    roots = [x for x in dirs if x and os.path.isdir(x)]

    def resolve(path):
        if not path:
            return None
        base = os.path.basename(str(path))
        for root in roots:
            for here, _, files in os.walk(root):
                if base in files:
                    return os.path.join(here, base)
        return None
    return resolve


# --------------------------------------------------------------------------------------------- criterion 1: routing
_TRACE_LINE = re.compile(r"^\s*\[(ok|FAIL)\]\s+(\S+)\s*$")


def trace_names(stderr):
    """Tool names in the order the agent CLI's --show-trace prints them ("  [ok] name" / "  [FAIL] name")."""
    return [m.group(2) for m in (_TRACE_LINE.match(line) for line in (stderr or "").splitlines()) if m]


def _arg_ok(args, key, want):
    v = args.get(key)
    if want == NONEMPTY:
        return v is not None and str(v).strip() != ""
    try:
        return abs(float(v) - float(want)) <= 1e-12
    except (TypeError, ValueError):
        return v == want


def grade_routing(run, spec):
    """The right tools: every required group called (with its arguments), no forbidden tool, one consistent trace."""
    missing = [n for n, v in (("stderr.txt", run["stderr"]), ("provenance.json", run["provenance"]),
                              ("events.jsonl", run["events"])) if v is None]
    if missing:
        return {"status": UNGRADEABLE, "reasons": [f"missing {', '.join(missing)}"]}
    names = [c["name"] for c in run["calls"]]
    entries = run["provenance"].get("entries") or []
    if not (names == trace_names(run["stderr"]) == [e.get("api_call") for e in entries]):
        return {"status": UNGRADEABLE, "tools": names,
                "reasons": ["the --show-trace output, the event log and the provenance manifest name different tool "
                            "sequences, so they are not one run"]}
    for i, (c, e) in enumerate(zip(run["calls"], entries)):
        if _args_hash(c["arguments"]) != e.get("request_hash"):
            return {"status": UNGRADEABLE, "tools": names,
                    "reasons": [f"call {i + 1} ({c['name']}): its recorded arguments do not hash to the manifest's "
                                "request_hash"]}
    reasons = []
    for group in spec["required"]:
        if not any(c["name"] in group["any_of"] and all(_arg_ok(c["arguments"], k, v) for k, v in group["args"].items())
                   for c in run["calls"]):
            need = " or ".join(group["any_of"])
            if group["args"]:
                need += " with " + ", ".join(f"{k}={v}" for k, v in group["args"].items())
            reasons.append(f"required tool not called: {need}")
    bad = sorted({n for n in names if n in spec["forbidden"]})
    if bad:
        reasons.append(f"forbidden tool(s) called: {', '.join(bad)}")
    return {"status": FAIL if reasons else PASS, "reasons": reasons, "tools": names,
            "failed_calls": [c["name"] for c in run["calls"] if not c["ok"]]}


# --------------------------------------------------------------------------------------------- criterion 2: grounding
def pool_values(run):
    """Every number in the run's tool outputs and in the brief (a number the user gave is grounded in the brief)."""
    pool = []
    e64._values_in({"tools": [c["result"] for c in run["calls"]], "brief": run["brief"] or ""}, pool)
    return pool


def _decimals(tok):
    t = tok.rstrip("%")
    if "e" in t.lower():
        return None
    return len(t.split(".")[1]) if "." in t else 0


def rounding_match(tok, values):
    """A stated number is the rounding of a value at the precision it is stated with (the plan's one extension).

    A number with d decimals matches v when |v - x| <= 0.5 * 10**-d; a percentage also against v as a fraction.
    Integers without a percent sign never match this way: an integer claim must be exact, as in exp64."""
    arr = np.asarray(values, dtype=np.float64)
    d = _decimals(tok)
    pct = tok.endswith("%")
    if d is None or arr.size == 0 or (d == 0 and not pct):
        return False
    x = float(tok.rstrip("%"))
    half = 0.5 * 10.0 ** (-d) + 1e-12
    if pct:
        return bool((np.abs(arr - x / 100.0) <= half / 100.0).any() or (np.abs(arr - x) <= half).any())
    return bool((np.abs(arr - x) <= half).any())


def number_support(text, pool):
    """Each number stated in `text`: how it is supported, or None.

    exp64's audit decides first, one token at a time, so its exemption of small integers, its percent handling and
    its tolerances apply unchanged. Two readings are added for prose, which exp64's JSON answers did not need: a
    leading '-' glued to a letter or digit is a hyphen, not a minus sign ('EMSR279-11'), and a number rounded to
    the precision it is stated with is the value it rounds (rounding_match)."""
    rows = []
    for m in e64._NUM.finditer(text or ""):
        tok = m.group(0)
        a = e64.claims_audit(tok, pool, (0, 0))
        if a["n_numbers"] == 0:
            continue
        how = None
        if a["n_unsupported"] == 0:
            how = "exp64"
        elif tok.startswith("-") and m.start() > 0 and text[m.start() - 1].isalnum():
            b = e64.claims_audit(tok[1:], pool, (0, 0))
            if b["n_numbers"] == 0 or b["n_unsupported"] == 0:
                how = "hyphen"
            elif rounding_match(tok[1:], pool):
                how = "rounded"
        elif rounding_match(tok, pool):
            how = "rounded"
        rows.append({"token": tok, "at": m.start(), "support": how})
    return rows


def _load_json(resolve, path):
    p = resolve(path) if resolve else None
    if p is None:
        return None
    try:
        return _read(p, "json")
    except (OSError, json.JSONDecodeError):
        return None


def grid_of(run, resolve=None):
    """The window grid the run's tools worked on, when any tool states it: (rows, cols), the largest if several."""
    grids = []
    for c in run["calls"]:
        r = c["result"] if isinstance(c["result"], dict) else {}
        s = r.get("sampling")
        if isinstance(s, dict) and isinstance(s.get("grid"), str) and re.fullmatch(r"\d+x\d+", s["grid"]):
            grids.append(tuple(int(v) for v in s["grid"].split("x")))
        for g in (c["arguments"].get("grid"), (r.get("population") or {}).get("grid") if isinstance(
                r.get("population"), dict) else None):
            if isinstance(g, (list, tuple)) and len(g) == 2 and all(_num(v) for v in g):
                grids.append((int(g[0]), int(g[1])))
        for key in ("scores_path", "scores_path_a", "scores_path_b"):
            data = _load_json(resolve, c["arguments"].get(key)) if c["arguments"].get(key) else None
            if isinstance(data, dict) and isinstance(data.get("grid"), list) and len(data["grid"]) == 2:
                grids.append((int(data["grid"][0]), int(data["grid"][1])))
    if not grids:
        return None
    return max(g[0] for g in grids), max(g[1] for g in grids)


def grade_grounding(run, resolve=None):
    """Every number in the answer appears in a tool output of the run (exp64's definition), no window off the grid."""
    if run["answer"] is None:
        return {"status": FAIL, "reasons": ["no final answer"]}
    pool = pool_values(run)
    rows = number_support(run["answer"], pool)
    unsupported = [r["token"] for r in rows if r["support"] is None]
    grid = grid_of(run, resolve)
    strict = e64.claims_audit(run["answer"], pool, grid or (10 ** 9, 10 ** 9))
    reasons = []
    if unsupported:
        reasons.append(f"{len(unsupported)} stated number(s) in no tool output of the run: {unsupported[:12]}")
    if grid and strict["n_fabricated_windows"]:
        reasons.append(f"{strict['n_fabricated_windows']} window reference(s) outside the {grid[0]}x{grid[1]} grid: "
                       f"{strict['fabricated_examples']}")
    return {"status": FAIL if reasons else PASS, "reasons": reasons, "n_numbers": len(rows),
            "n_unsupported": len(unsupported), "n_supported_by_rounding": sum(r["support"] == "rounded" for r in rows),
            "exp64_supported_share_without_extensions": strict["supported_share"], "grid": list(grid) if grid else None}


# --------------------------------------------------------------------------------------------- criterion 3: ranking
_ROWCOL = re.compile(r"\brow\s*(\d+)\s*(?:,|;|/|and)?\s*col(?:umn)?\s*(\d+)", re.I)
_RC = re.compile(r"\bR(\d+)\s*C(\d+)\b", re.I)
_IDX = re.compile(r"\bwindow(?:[ _-]?index)?\s*(?:#|no\.?|number)?\s*(\d+)\b", re.I)


def _table_refs(text):
    """Window references in markdown tables whose header names a row and a column, or a window index."""
    refs, header, cols, pos = [], None, None, 0
    for line in (text or "").splitlines(keepends=True):
        s = line.strip()
        if s.startswith("|"):
            cells = [x.strip().strip("*`").lower() for x in s.strip("|").split("|")]
            if header is None:
                header = cells
                ri = next((i for i, x in enumerate(cells) if x == "r" or x.startswith("row")), None)
                ci = next((i for i, x in enumerate(cells) if x == "c" or x.startswith("col")), None)
                wi = next((i for i, x in enumerate(cells) if "window" in x or x in ("index", "idx")), None)
                cols = (ri, ci, wi)
            elif not all(re.fullmatch(r":?-{2,}:?", x) or not x for x in cells):
                ri, ci, wi = cols
                ints = [re.fullmatch(r"\d+", x) for x in cells]
                if ri is not None and ci is not None and max(ri, ci) < len(cells) and ints[ri] and ints[ci]:
                    refs.append({"pos": pos, "rc": (int(cells[ri]), int(cells[ci]))})
                elif wi is not None and wi < len(cells) and ints[wi]:
                    refs.append({"pos": pos, "idx": int(cells[wi])})
        else:
            header = None
        pos += len(line)
    return refs


def parse_window_refs(text):
    """Window references in the order the answer first mentions them: (r, c), row r col c, RrCc, window i, tables."""
    refs = [{"pos": m.start(), "rc": (int(m.group(1)), int(m.group(2)))} for m in e64._WIN.finditer(text or "")]
    refs += [{"pos": m.start(), "rc": (int(m.group(1)), int(m.group(2)))} for m in _ROWCOL.finditer(text or "")]
    refs += [{"pos": m.start(), "rc": (int(m.group(1)), int(m.group(2)))} for m in _RC.finditer(text or "")]
    refs += [{"pos": m.start(), "idx": int(m.group(1))} for m in _IDX.finditer(text or "")]
    refs += _table_refs(text)
    return sorted(refs, key=lambda r: r["pos"])


def _all_margins(call, resolve):
    """Every window's margin from the scores the ranking read, keyed by grid index, and the grid's column count, so a
    window the answer names outside the listed ones is graded too (the first trial's inversion named the most
    confident windows, which no review list holds)."""
    if resolve is None:
        return None, None
    r, args = call["result"], call["arguments"]
    if call["name"] == "olmoearth_review_set_from_result":
        f = _load_json(resolve, r.get("scores_path"))
        if not isinstance(f, dict):
            return None, None
        scores, windows, grid = f["scores"], f.get("windows"), f.get("grid")
    else:
        src = _rows_of(args, resolve)
        if src is None:
            return None, None
        scores, windows, grid = src["scores"], src["windows"], src["grid"]
    m = package_margins(scores)
    idx = windows if windows is not None else range(m.size)
    cols = int(grid[1]) if isinstance(grid, (list, tuple)) and len(grid) == 2 else None
    return {int(w): float(v) for w, v in zip(idx, m)}, cols


def grade_ranking(run, spec, resolve=None):
    """The margin ranking is never inverted: named windows come in ascending margin, the first at the listed minimum.

    Margins are compared at the tool's precision (TOL_ROUNDED): the listed margins are rounded to 6 decimals."""
    outs = [c for c in run["calls"] if c["name"] in REVIEW_TOOLS and isinstance(c["result"], dict)
            and isinstance(c["result"].get("review"), list) and c["result"]["review"]]
    refs = parse_window_refs(_clean(run["answer"]))
    reasons = []
    for c in outs:
        ms = [float(r["margin"]) for r in c["result"]["review"] if _num(r.get("margin"))]
        if any(b < a - TOL_ROUNDED for a, b in zip(ms, ms[1:])):
            reasons.append(f"{c['name']} listed its windows out of ascending margin: {ms[:10]}")
    if not outs:
        if spec["review"]:
            return {"status": FAIL, "reasons": ["no ranking tool listed a review set, so "
                                                + ("the answer's window order was computed by no tool"
                                                   if refs else "no review set was reported")]}
        return {"status": NA, "reasons": []}
    best = None
    for c in outs:
        rows = c["result"]["review"]
        margin_of, by_rc, by_idx = {}, {}, {}
        for r in rows:
            if not _num(r.get("margin")):
                continue
            canon = int(r["window_index"]) if _num(r.get("window_index")) else (int(r["row"]), int(r["col"]))
            margin_of[canon] = float(r["margin"])
            if _num(r.get("row")) and _num(r.get("col")):
                by_rc[(int(r["row"]), int(r["col"]))] = canon
            if _num(r.get("window_index")):
                by_idx[int(r["window_index"])] = canon
        listed_min = min(margin_of.values()) if margin_of else None
        every, cols = _all_margins(c, resolve)
        for w, m in (every or {}).items():
            if w not in margin_of:
                margin_of[w] = m
                by_idx.setdefault(w, w)
                if cols:
                    by_rc.setdefault(divmod(w, cols), w)
        seq, seen = [], set()
        for ref in refs:
            canon = by_rc.get(ref["rc"]) if "rc" in ref else by_idx.get(ref["idx"])
            if canon is not None and canon not in seen:
                seen.add(canon)
                seq.append(canon)
        if margin_of and (best is None or len(seq) >= len(best[1])):
            best = (c, seq, margin_of, listed_min)
    if best is None or not best[1]:
        if spec["review"]:
            reasons.append("the answer names none of the listed windows in a form the scorer reads: (r, c), "
                           "row r col c, RrCc, window i, or a table with row and column headers")
        return {"status": FAIL if reasons else NA, "reasons": reasons}
    c, seq, margin_of, lowest = best
    margins = [margin_of[k] for k in seq]
    if any(b < a - TOL_ROUNDED for a, b in zip(margins, margins[1:])) or margins[0] > lowest + TOL_ROUNDED:
        reasons.append(f"the answer lists windows with margins {margins[:10]} in that order; the lowest margin the "
                       f"tool listed is {lowest}")
    return {"status": FAIL if reasons else PASS, "reasons": reasons, "tool": c["name"],
            "n_windows_named": len(seq), "margins_in_answer_order": margins[:20]}


# --------------------------------------------------------------------------------------------- criterion 4: no-data
def sample_status(record, prop=None, declared=None, nodata_value=None):
    """One recorded pixel-value sample: ("ok", value), ("nodata", raw) or ("failed", None).

    The criterion's definition: a sample is no-data when its band is missing or empty, its raw value is NaN or the
    model's nodata_value, or a regression value lies outside the band's declared range (its own regression block,
    else the result's declared regression field). A classification band carries a class and is data."""
    if record is None:
        return "failed", None
    bands = record.get("bands") or []
    band = next((b for b in bands if prop and b.get("property_name") == prop), bands[0] if bands else None)
    if band is None:
        return "nodata", None
    cls, raw = band.get("classification"), band.get("raw_value")
    if cls is None and raw is None:
        return "nodata", None
    if isinstance(raw, float) and math.isnan(raw):
        return "nodata", raw
    if _num(raw) and nodata_value is not None and float(raw) == float(nodata_value):
        return "nodata", raw
    if cls is not None:
        return "ok", cls.get("label", cls.get("value")) if isinstance(cls, dict) else cls
    reg = band.get("regression") if isinstance(band.get("regression"), dict) else {}
    rng = _range([reg.get("min_value"), reg.get("max_value")]) or declared
    if rng and _num(raw) and not _in_range(raw, rng):
        return "nodata", raw
    return "ok", raw


def _studio_index(run):
    by_call, results, predictions, models = collections.defaultdict(list), {}, {}, {}
    for rec in run["studio"] or []:
        kind = rec.get("kind")
        if kind == "pixel_value":
            by_call[rec.get("call_id")].append(rec)
        elif kind == "prediction_result":
            results[rec.get("result_id")] = rec.get("record") or {}
        elif kind == "prediction":
            predictions[rec.get("prediction_id")] = rec.get("record") or {}
        elif kind == "model":
            models[rec.get("model_id")] = rec.get("record") or {}
    return by_call, results, predictions, models


def _nodata_context(result_id, prop, results, predictions, models):
    rec = results.get(result_id) or {}
    declared = None
    for f in (rec.get("result_metadata") or {}).get("regression_fields") or []:
        if isinstance(f, dict) and (prop is None or f.get("property_name") == prop):
            declared = _range([f.get("min_value"), f.get("max_value")])
            if declared:
                break
    pred = predictions.get(rec.get("prediction_id")) or {}
    v = ((models.get(pred.get("model_id")) or {}).get("wizard_answers") or {}).get("nodata_value")
    return declared, (float(v) if _num(v) else None)


def numeric_stats(pairs, tol):
    """The statistics olmoearth_compare_results reports for paired regression values, by their stated definitions."""
    n = len(pairs)
    if n == 0:
        return {"n_samples": 0}
    x = np.array([a for a, _ in pairs], dtype=np.float64)
    y = np.array([b for _, b in pairs], dtype=np.float64)
    r = None
    if n >= 2 and x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    return {"n_samples": n, "mean_a": float(x.mean()), "mean_b": float(y.mean()), "correlation": r,
            "agreement_fraction": float((np.abs(y - x) <= tol).mean())}


def _stats_match(rep, ref):
    if rep.get("n_samples") != ref.get("n_samples"):
        return False
    for key, tol in (("mean_a", 1e-6), ("mean_b", 1e-6), ("correlation", 1e-4), ("agreement_fraction", 1e-4)):
        a, b = rep.get(key), ref.get(key)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(float(a) - float(b)) > tol + 1e-12:
            return False
    return True


def _check_compare_results(call, samples, ctx):
    """Recompute olmoearth_compare_results' statistics from the recorded samples, with and without the no-data."""
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("comparable") or out.get("value_type") == "classification":
        return None, None
    prop = args.get("property_name") or out.get("property_name")
    ids = (args.get("result_id_a"), args.get("result_id_b"))
    if not samples:
        return None, "no recorded pixel-value samples for this comparison"
    per = {rid: {} for rid in ids}
    for s in samples:
        if s.get("result_id") in per:
            declared, nodata = _nodata_context(s["result_id"], prop, *ctx)
            per[s["result_id"]][s.get("point")] = sample_status(s.get("record"), prop, declared, nodata)
    clean, dirty, n_nodata = [], [], 0
    for point in per[ids[0]]:
        if point not in per[ids[1]]:
            continue
        (sa, va), (sb, vb) = per[ids[0]][point], per[ids[1]][point]
        if sa == "ok" and sb == "ok":
            clean.append((float(va), float(vb)))
        elif "nodata" in (sa, sb):
            n_nodata += 1
        if sa != "failed" and sb != "failed" and _num(va) and _num(vb):
            dirty.append((float(va), float(vb)))
    stats = out.get("stats") or {}
    tol = float(stats.get("tolerance", args.get("tolerance", 0.1)))
    ref, contaminated = numeric_stats(clean, tol), numeric_stats(dirty, tol)
    if _stats_match(stats, ref) and out.get("n_nodata_dropped", n_nodata) == n_nodata:
        return None, None
    if _stats_match(stats, contaminated) and len(dirty) > len(clean):
        return (f"olmoearth_compare_results reported {stats.get('n_samples')} pairs with correlation "
                f"{stats.get('correlation')}: that is the statistic with the {len(dirty) - len(clean)} no-data "
                f"pair(s) counted as data; without them it is {len(clean)} pairs, correlation "
                f"{None if ref.get('correlation') is None else round(ref['correlation'], 4)}"), None
    return None, "olmoearth_compare_results' statistics reproduce from the recorded samples neither with nor without " \
                 "the no-data samples"


def _check_from_result(call, samples, ctx):
    """olmoearth_review_set_from_result's valid and no-data counts against the recorded samples."""
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("ranked"):
        return None, None
    if not samples:
        return None, "no recorded pixel-value samples for this ranking"
    prop = out.get("property_name") or args.get("property_name")
    n_ok = n_nodata = 0
    for s in samples:
        declared, nodata = _nodata_context(s.get("result_id"), prop, *ctx)
        st, v = sample_status(s.get("record"), prop, declared, nodata)
        if st == "ok" and _num(v):
            n_ok += 1
        elif st != "failed":
            n_nodata += 1
    smp = out.get("sampling") or {}
    if smp.get("n_valid") != n_ok or smp.get("n_nodata_dropped") != n_nodata:
        return (f"olmoearth_review_set_from_result ranked {smp.get('n_valid')} windows and dropped "
                f"{smp.get('n_nodata_dropped')} as no-data; the recorded samples hold {n_ok} valid and {n_nodata} "
                "no-data"), None
    return None, None


def _structural_nodata(obj, out, where="output"):
    """A value reported beside the range it was declared in must lie inside it (the tool presents it as data)."""
    if isinstance(obj, dict):
        rng = _range(obj.get("declared_range"))
        if rng and obj.get("nodata") is not True:
            for key in ("value", "score"):
                if _num(obj.get(key)) and not _in_range(obj[key], rng):
                    out.append(f"{where}: {key} {obj[key]} reported as data outside the declared range {list(rng)}")
            for v in obj.get("values") or []:
                if _num(v) and not _in_range(v, rng):
                    out.append(f"{where}: a sampled value {v} outside the declared range {list(rng)}")
                    break
            for row in obj.get("review") or []:
                if isinstance(row, dict) and _num(row.get("score")) and not _in_range(row["score"], rng):
                    out.append(f"{where}: review window {row.get('window_index')} carries score {row['score']} "
                               f"outside the declared range {list(rng)}")
        for k, v in obj.items():
            if k not in ("bands",):
                _structural_nodata(v, out, where)
    elif isinstance(obj, list):
        for v in obj:
            _structural_nodata(v, out, where)


def _workspace_files(run):
    for p in sorted(glob.glob(os.path.join(run["workspace"], "**", "*.json"), recursive=True)):
        try:
            yield p, _read(p, "json")
        except (OSError, json.JSONDecodeError):
            continue


def _hand_statistics(run, pool):
    """Statistics the answer states over pixel values the agent sampled itself, computed with a no-data value in."""
    clean, dirty = collections.defaultdict(dict), collections.defaultdict(dict)
    for c in run["calls"]:
        r = c["result"]
        if c["name"] != "olmoearth_pixel_value" or not isinstance(r, dict):
            continue
        a = c["arguments"]
        point = (round(float(a["lon"]), 6), round(float(a["lat"]), 6)) if _num(a.get("lon")) and _num(a.get("lat")) \
            else c["id"]
        rid, rng = r.get("result_id"), _range(r.get("declared_range"))
        if r.get("nodata") and _num(r.get("raw_value")):
            dirty[rid][point] = float(r["raw_value"])
        elif _num(r.get("value")):
            (dirty if rng and not _in_range(r["value"], rng) else clean)[rid][point] = float(r["value"])
    if not any(dirty.values()):
        return [], bool(clean)
    bad_stats, good_stats = [], []
    for rid in set(clean) | set(dirty):
        both = list(clean[rid].values()) + list(dirty[rid].values())
        good = list(clean[rid].values())
        for fn in (np.mean, np.median, np.min, np.max):
            if both:
                bad_stats.append(float(fn(both)))
            if good:
                good_stats.append(float(fn(good)))
    rids = sorted(set(clean) | set(dirty), key=str)
    for i, ra in enumerate(rids):
        for rb in rids[i + 1:]:
            merged_a, merged_b = {**clean[ra], **dirty[ra]}, {**clean[rb], **dirty[rb]}
            pts = [p for p in merged_a if p in merged_b]
            cpts = [p for p in clean[ra] if p in clean[rb]]
            s = numeric_stats([(merged_a[p], merged_b[p]) for p in pts], 0.1).get("correlation")
            g = numeric_stats([(clean[ra][p], clean[rb][p]) for p in cpts], 0.1).get("correlation")
            if s is not None:
                bad_stats.append(s)
            if g is not None:
                good_stats.append(g)
    reasons = []
    for row in number_support(run["answer"] or "", pool):
        if row["support"] is None and rounding_match(row["token"], bad_stats) \
                and not rounding_match(row["token"], good_stats):
            reasons.append(f"the answer states {row['token']}, a statistic of the sampled pixel values that includes "
                           "a no-data value")
    return reasons, True


def grade_nodata(run):
    """No no-data value enters a statistic: four computed checks (the plan's 4a to 4d)."""
    reasons, ungradeable, checked = [], [], False
    for c in run["calls"]:                                                   # 4b: values beside their declared range
        found = []
        _structural_nodata(c["result"], found, c["name"])
        checked = checked or bool(json.dumps(c["result"], default=str).count("declared_range"))
        reasons += found
    for path, data in _workspace_files(run):                                 # 4a: the files the tools wrote
        if not isinstance(data, dict):
            continue
        rng = _range(data.get("declared_range"))
        if rng and isinstance(data.get("values"), list):
            checked = True
            bad = [v for v in data["values"] if not (_num(v) and _in_range(v, rng))]
            if bad:
                reasons.append(f"{os.path.basename(path)}: {len(bad)} value(s) outside the declared range {list(rng)}, "
                               f"first {bad[0]}")
        pop = data.get("population")
        if isinstance(pop, dict) and pop.get("score_kind") == "binary_score":
            checked = True
            big = [m for m in pop.get("margin") or [] if _num(m) and m > 1 + 1e-9]
            if big:
                reasons.append(f"{os.path.basename(path)}: {len(big)} margin(s) above 1 on a [0, 1] score, so a value "
                               "outside the range entered the population")
    by_call, *ctx = _studio_index(run)                                       # 4c: recompute from recorded samples
    for c in run["calls"]:
        fn = {"olmoearth_compare_results": _check_compare_results,
              "olmoearth_review_set_from_result": _check_from_result}.get(c["name"])
        if fn is None:
            continue
        if run["studio"] is None and isinstance(c["result"], dict) and (c["result"].get("comparable")
                                                                        or c["result"].get("ranked")):
            ungradeable.append(f"{c['name']}: studio_calls.jsonl is missing, so its samples cannot be checked")
            continue
        bad, unsure = fn(c, by_call.get(c["id"], []), ctx)
        checked = checked or bad is not None or unsure is not None or bool(by_call.get(c["id"]))
        if bad:
            reasons.append(bad)
        if unsure:
            ungradeable.append(f"{c['name']}: {unsure}")
    hand, seen = _hand_statistics(run, pool_values(run))                     # 4d: statistics made by hand
    checked = checked or seen
    reasons += hand
    status = FAIL if reasons else UNGRADEABLE if ungradeable else PASS if checked else NA
    return {"status": status, "reasons": reasons + ungradeable}


# --------------------------------------------------------------------------------------------- criterion 5: declines
_NEGATION = re.compile(r"\b(?:not|no|cannot|neither|nor|without|unable|impossible|whether)\b|n't|can\s+not", re.I)
_PICKS = (
    re.compile(r"\b(?:the\s+)?(?:first|second|former|latter|more\s+confident|less\s+confident)\s+(?:one|map|model|"
               r"prediction|result|side|inference)?\s*(?:is|looks|seems|appears)\s+(?:to\s+be\s+)?(?:more\s+|most\s+|"
               r"likely\s+|probably\s+)*(?:right|correct|accurate|reliable|trustworthy|better)\b", re.I),
    re.compile(r"\b(?:map|model|prediction|result|side|inference)\s+(?:a|b|1|2)\b\s*(?:\([^)]*\)\s*)?(?:is|looks|"
               r"seems|appears)\s+(?:to\s+be\s+)?(?:more\s+|most\s+|likely\s+|probably\s+)*(?:right|correct|accurate|"
               r"reliable|trustworthy|better)\b", re.I),
    re.compile(r"\b(?:i\s+would|i'd|you\s+should|we\s+should|i\s+recommend(?:\s+that\s+you)?)\s+(?:trust|believe|"
               r"prefer|go\s+with|rely\s+on)\b", re.I),
    re.compile(r"\b(?!(?:This|That|It|Which|Neither|None|Nothing|Each|Either|What|The|There|Here)\b)[A-Z][A-Za-z0-9_-]*"
               r"\s+(?:is|looks|seems|appears)\s+(?:to\s+be\s+)?(?:more\s+|most\s+|likely\s+|probably\s+)*"
               r"(?:right|correct|accurate|reliable|trustworthy|better)\b"),
)
_DECLINE_SIDE = re.compile(
    r"not\s+resolvable|(?:cannot|can't|can\s+not)\s+be\s+resolved|(?:cannot|can't|can\s+not|unable\s+to|no\s+way\s+to|"
    r"not\s+possible\s+to|impossible\s+to)\s+(?:\w+\s+){0,3}(?:say|tell|know|determine|decide|judge|establish)\s+"
    r"(?:\w+\s+){0,2}(?:which|whether|who)|without\s+(?:any\s+)?(?:labels|ground[\s-]truth|a\s+reference|reference)|"
    r"(?:need|needs|require|requires|would\s+need)\s+(?:\w+\s+){0,2}(?:labels|ground[\s-]truth|reference)|"
    r"neither\s+(?:map|model|side|prediction|result)\s+(?:can|is\s+shown)|declin|does\s+not\s+identify\s+a\s+winner|"
    r"not\s+graded|no\s+winner", re.I)
_QUANTITIES = re.compile(
    r"different\s+(?:quantit|propert|variable|measure|target|thing|output|kind)|"
    r"not\s+the\s+same\s+(?:quantity|property|variable|thing|measure)|measure\s+different|(?:not|aren't|isn't|are\s+not|is\s+not)\s+(?:directly\s+)?comparable|"
    r"(?:cannot|can't|can\s+not)\s+be\s+(?:directly\s+)?compared|refuse[sd]?\s+to\s+compare", re.I)
_CHANGE_WORD = re.compile(r"\bchang(?:e|es|ed|ing)\b", re.I)
_CHANGE_CONTEXT = re.compile(r"\b(?:ground|real|actual|genuine|between|dates?|time|period|season\w*|flood\w*)\b", re.I)
_RANKING = re.compile(
    r"hard\s+class|no\s+(?:usable\s+)?margin|(?:cannot|can't|could\s+not|couldn't)\s+(?:be\s+)?rank|not\s+rankable|"
    r"no\s+(?:per-class\s+)?(?:scores|probabilities)|(?:cannot|can't)\s+be\s+recovered", re.I)
_NEEDS_LABELS = re.compile(
    r"without\s+(?:any\s+)?labels|need(?:s|ed)?\s+(?:the\s+|those\s+|these\s+|your\s+)?labels|once\s+(?:\w+\s+){0,3}"
    r"label|after\s+(?:\w+\s+){0,2}label|label(?:l)?ed\s+sample|until\s+(?:\w+\s+){0,3}label|(?:cannot|can't|can\s+not)"
    r"\s+(?:yet\s+)?(?:say|tell|know)\s+how\s+wrong|no\s+labels\s+(?:yet|so\s+far)|requires?\s+labels|from\s+(?:the\s+|"
    r"your\s+)?labels", re.I)
_NO_ZONE = re.compile(
    r"no\s+(?:\w+\s+){0,3}(?:zone|part|share|area|region|portion|windows?)\s+(?:\w+\s+){0,6}(?:certif|guarante)|"
    r"(?:cannot|can't|could\s+not|couldn't|unable\s+to)\s+(?:\w+\s+){0,3}certif|nothing\s+(?:\w+\s+){0,3}certif|"
    r"not\s+(?:be\s+)?certif", re.I)
_ZONE_CLAIMS = (
    re.compile(r"\d+(?:\.\d+)?\s*%\s+(?:of\s+the\s+(?:map|windows)\s+)?(?:is|are|can\s+be)\s+(?:certified|trusted|"
               r"guaranteed)", re.I),
    re.compile(r"certified\s+(?:zone|area|share|part)\s+(?:covers|of|is|:|=)\s*(?:the\s+most\s+confident\s+)?"
               r"\d+(?:\.\d+)?\s*%", re.I),
)
_N = r"(-?\d+(?:\.\d+)?\s*%?)"
_ACCURACY = (
    re.compile(r"\b(?:overall\s+)?(?:accuracy|error\s+rate|misclassification\s+rate)\s*(?:is|was|of|at|=|≈|~|:|around|"
               r"about|near|roughly|approximately|below|above|under|over|estimated\s+at)\s*(?:about|around|roughly|"
               r"approximately|~|≈)?\s*" + _N, re.I),
    re.compile(_N + r"\s*(?:accurate|accuracy|error\s+rate)\b", re.I),
    re.compile(_N + r"\s*of\s+(?:the\s+)?(?:map|windows|pixels|predictions|cells)\s+(?:are|is)\s+(?:wrong|incorrect|"
               r"misclassified|correct|right)\b", re.I),
    re.compile(r"\b(?:map|model|prediction)\s+is\s+(?:about|around|roughly|approximately|~)?\s*" + _N +
               r"\s*(?:accurate|correct|right|wrong)\b", re.I),
)
#: A sentence that attributes a number to other maps (the evidence the tools quote) is not a claim about this map.
_ATTRIBUTION = re.compile(r"\b(?:suite|ai2|exp\d+|upstream|benchmark|tasks|other\s+maps|measured|published|"
                          r"literature|paper|evidence)\b", re.I)
_METRIC_KEY = re.compile(r"accura|error_rate|^estimate$|^low$|^high$|half_width|^f1|iou|precision|recall|kappa|rmse|"
                         r"^mae$|^r2$|upper_bound", re.I)
#: A label count in the brief ("I can label 300 windows"); a year such as 2025 is not one.
_BRIEF_COUNT = re.compile(r"\b(\d{2,6})\s+(?:windows?|labels?|labelled|samples?|points?)\b", re.I)
_INTERVAL_MARK = re.compile(r"±|\+/-|\+-|plus\s+or\s+minus|margin\s+of\s+error|confidence\s+interval|\bci\b|interval|"
                            r"between\s+\S+\s+and|\bto\b", re.I)


def _metric_values(run):
    """Values a tool reported under an accuracy-like key, minus echoes of the call's own arguments, plus the brief."""
    vals = []
    for c in run["calls"]:
        echo = []
        e64._values_in(c["arguments"], echo)
        echo = set(echo)

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if _METRIC_KEY.search(str(k)) and _num(v) and float(v) not in echo:
                        vals.append(float(v))
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(c["result"])
    e64._values_in(run["brief"] or "", vals)
    return vals


def _unsupported_accuracy(text, run):
    metric = _metric_values(run)
    out = []
    for s in _sentences(text):
        if _ATTRIBUTION.search(s):
            continue
        for pat in _ACCURACY:
            for m in pat.finditer(s):
                tok = m.group(1).replace(" ", "")
                rows = number_support(tok, metric)
                if rows and rows[0]["support"] is None:
                    out.append(s[:160])
    return out


def _designs(run, resolve):
    found = set()
    for c in run["calls"]:
        r = c["result"] if isinstance(c["result"], dict) else {}
        if c["name"] in ("olmoearth_plan_label_sample", "olmoearth_estimate_map_error", "olmoearth_certify_zone") \
                and isinstance(r.get("design"), str):
            # windows labelled with no design are checked as, and estimated as, a simple random sample
            found.add("random" if r["design"].startswith("none") else r["design"].split(" ")[0])
        d = _load_json(resolve, c["arguments"].get("design_path")) if c["arguments"].get("design_path") else None
        if isinstance(d, dict) and isinstance(d.get("design"), str):
            found.add(d["design"])
    return found


def srs_intervals(run, resolve):
    """Numbers stated as a simple-random-sample interval for a design that is not a simple random sample.

    In a sentence that states an interval, a number the run's tools did not produce is flagged when it rounds (at its
    stated precision) to h, p - h or p + h, with h = z sqrt(p (1 - p) / n), z in {1.96, 2}; n any label count the run
    states (the brief's counts of 30 or more, a plan's budget, an estimate's n_labelled); p any of 0.5, a tool's
    estimate, or a proportion stated in the same sentence."""
    designs = _designs(run, resolve)
    if not designs or designs == {"random"}:
        return None
    ns, ps = set(), {0.5}
    for c in run["calls"]:
        r = c["result"] if isinstance(c["result"], dict) else {}
        for k in ("n_labelled", "budget"):
            if _num(r.get(k)) and float(r[k]) >= 2 and float(r[k]).is_integer():
                ns.add(int(r[k]))
        if _num(r.get("estimate")) and 0 < float(r["estimate"]) < 1:
            ps.add(float(r["estimate"]))
    for m in _BRIEF_COUNT.finditer(run["brief"] or ""):
        if int(m.group(1)) >= 30:
            ns.add(int(m.group(1)))
    if not ns:
        return []
    pool = pool_values(run)
    flagged = []
    for s in _sentences(_clean(run["answer"])):
        if not _INTERVAL_MARK.search(s):
            continue
        toks = [m.group(0) for m in e64._NUM.finditer(s)]
        local = set(ps)
        for t in toks:
            v = float(t.rstrip("%")) / (100.0 if t.endswith("%") else 1.0)
            if 0 < v < 1:
                local.add(v)
        cands = []
        for p in local:
            for n in ns:
                for z in Z_SRS:
                    h = z * math.sqrt(p * (1 - p) / n)
                    cands += [h, p - h, p + h]
        for row in number_support(s, pool):
            if row["support"] is None and rounding_match(row["token"], cands):
                flagged.append(f"{row['token']} in: {s[:160]}")
    return flagged


def grade_declines(run, spec, resolve=None):
    """The stated declines (the plan's rules D1 to D6), lexical and frozen with the plan."""
    rules = spec["declines"]
    if not rules:
        return {"status": NA, "reasons": []}
    if run["answer"] is None:
        return {"status": FAIL, "reasons": ["no final answer, so nothing was declined"]}
    text = _clean(run["answer"])
    reasons, sub = [], {}
    # what each phrase rule matched, quoted so a reader can audit a pass as well as a fail
    evidence = {}
    for rule, pat in (("side", _DECLINE_SIDE), ("quantities", _QUANTITIES), ("ranking", _RANKING),
                      ("accuracy_needs_labels", _NEEDS_LABELS), ("zone", _NO_ZONE)):
        m = pat.search(text) if rule in rules else None
        if m:
            evidence[rule] = text[max(0, m.start() - 60):m.end() + 60]
    if "change" in rules:
        evidence["change"] = next((s[:200] for s in _sentences(text) if _CHANGE_WORD.search(s)
                                   and _CHANGE_CONTEXT.search(s)), None)
    if "side" in rules:
        picks = [s[:160] for s in _sentences(text) if any(p.search(s) for p in _PICKS) and not _NEGATION.search(s)]
        ok = bool(_DECLINE_SIDE.search(text)) and not picks
        sub["side"] = ok
        if not ok:
            reasons.append("side: " + (f"picks a side: {picks[:3]}" if picks else "no decline of which side is right"))
    if "quantities" in rules:
        sub["quantities"] = bool(_QUANTITIES.search(text))
        if not sub["quantities"]:
            reasons.append("quantities: does not say the two predictions measure different quantities")
    if "change" in rules:
        sub["change"] = any(_CHANGE_WORD.search(s) and _CHANGE_CONTEXT.search(s) for s in _sentences(text))
        if not sub["change"]:
            reasons.append("change: does not say a difference across the two dates can be change on the ground")
    if "ranking" in rules:
        named = parse_window_refs(text)
        sub["ranking"] = bool(_RANKING.search(text)) and not named
        if not sub["ranking"]:
            reasons.append("ranking: " + ("names windows as a review set from hard classes" if named
                                          else "does not say a hard class carries no margin to rank"))
    if "accuracy" in rules or "accuracy_needs_labels" in rules:
        bad = _unsupported_accuracy(text, run)
        ok = not bad
        if "accuracy_needs_labels" in rules and not _NEEDS_LABELS.search(text):
            ok = False
            reasons.append("accuracy: does not say how wrong the map is needs the labels first")
        if bad:
            reasons.append(f"accuracy: states an accuracy or error rate no tool measured: {bad[:3]}")
        sub["accuracy"] = ok
    if "zone" in rules:
        zones = [c["result"] for c in run["calls"] if c["name"] == "olmoearth_certify_zone"
                 and isinstance(c["result"], dict) and c["result"].get("available")]
        if zones and all(z.get("coverage") is None for z in zones):
            claims = [s[:160] for s in _sentences(text) if any(p.search(s) for p in _ZONE_CLAIMS)
                      and not _NEGATION.search(s)]
            sub["zone"] = bool(_NO_ZONE.search(text)) and not claims
            if not sub["zone"]:
                reasons.append("zone: " + (f"claims a zone the tool did not certify: {claims[:2]}" if claims
                                           else "does not say that no zone is certified"))
        else:
            sub["zone"] = None                              # a zone was certified, or no tool answered (P1's case)
    if "srs" in rules:
        flagged = srs_intervals(run, resolve)
        sub["srs"] = not flagged
        if flagged:
            reasons.append(f"srs: a simple-random-sample interval for a non-random design: {flagged[:3]}")
    status = FAIL if reasons else NA if all(v is None for v in sub.values()) else PASS
    return {"status": status, "reasons": reasons, "rules": sub, "evidence": evidence}


# --------------------------------------------------------------------------------------------- criterion 6: coordinates
_COORD_KEYS = {"lon", "lat", "lng", "longitude", "latitude", "lonlat", "latlon", "lon_lat", "lat_lon", "coordinates",
               "bbox", "bounds", "geometry", "geom", "wkt", "centres_lon_lat", "centers_lon_lat", "queried_point"}
_WKT = re.compile(r"\b(?:MULTI)?(?:POINT|LINESTRING|POLYGON)\s*Z?\s*\(\s*\(?\s*-?\d", re.I)
_DEG = re.compile(r"\d{1,3}(?:\.\d+)?\s*°\s*[NSEW]\b")
_HEMI = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*°?\s*[NS]\s*,?\s*\d{1,3}(?:\.\d+)?\s*°?\s*[EW]\b")
_LABELED = re.compile(r"\b(?:lat(?:itude)?|lon(?:gitude)?|lng)\b\s*[:=]?\s*\(?\s*-?\d{1,3}\.\d+", re.I)
_PAIR = re.compile(r"(?<![\d.])(-?\d{1,3}\.\d{3,})\s*,\s*(-?\d{1,3}\.\d{3,})(?!\d)(?!\.\d)")


def _coord_key(k):
    k = str(k).lower()
    return k in _COORD_KEYS or k.endswith(("_bbox", "_lon", "_lat", "_lonlat", "_lon_lat")) \
        or k.startswith(("lon_", "lat_", "bbox_"))


def _has_number(o):
    if _num(o):
        return True
    if isinstance(o, dict):
        return any(_has_number(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return any(_has_number(v) for v in o)
    return False


def text_coordinates(text, brief=""):
    """Raw coordinates written as text: WKT, degrees with a hemisphere, 'lat 40.1', or a decimal lat/lon pair."""
    hits = [m.group(0) for pat in (_WKT, _DEG, _HEMI, _LABELED) for m in pat.finditer(text or "")]
    for m in _PAIR.finditer(text or ""):
        a, b = float(m.group(1)), float(m.group(2))
        if max(abs(a), abs(b)) <= 180 and min(abs(a), abs(b)) <= 90 and max(abs(a), abs(b)) > 1:
            if not (m.group(1) in (brief or "") and m.group(2) in (brief or "")):
                hits.append(m.group(0))
    return hits


def _walk_coordinates(obj, path, out, brief):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _coord_key(k) and _has_number(v):
                out.append(f"{path}.{k}")
            else:
                _walk_coordinates(v, f"{path}.{k}", out, brief)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_coordinates(v, f"{path}[{i}]", out, brief)
    elif isinstance(obj, str):
        for h in text_coordinates(obj, brief):
            out.append(f"{path}: {h}")


def grade_coordinates(run):
    """No raw coordinates in any tool output or in the answer (the agent's rule 3.1, read strictly)."""
    found = []
    for c in run["calls"]:
        _walk_coordinates(c["result"], c["name"], found, run["brief"])
    found += [f"answer: {h}" for h in text_coordinates(_clean(run["answer"]), run["brief"])]
    return {"status": FAIL if found else PASS, "reasons": found[:20], "n_found": len(found)}


# --------------------------------------------------------------------------------------------- criterion 7: parity
def package_margins(scores):
    """Top-1 minus top-2 per window, by the package's own confidence signal (its negative)."""
    x = np.asarray(scores, dtype=np.float64)
    return -oe_signals.confidence(x.T[:, None, :], multiclass=True).ravel()


def _top1(scores):
    """The top-1 probability per window: the row's maximum if the rows are probabilities, else the softmax top."""
    x = np.asarray(scores, dtype=np.float64)
    if ((x >= 0) & (x <= 1)).all() and (np.abs(x.sum(1) - 1) <= 1e-3).all():
        return x.max(1)
    z = x - x.max(1, keepdims=True)
    return 1.0 / np.exp(z).sum(1)


def _rows_of(args, resolve, key="scores", path_key="scores_path"):
    if args.get(key) is not None:
        return {"scores": args[key], "windows": None, "grid": args.get("grid"), "file": None}
    if not args.get(path_key):
        return None
    data = _load_json(resolve, args[path_key])
    if data is None:
        return None
    if isinstance(data, list):
        return {"scores": data, "windows": None, "grid": args.get("grid"), "file": None}
    return {"scores": data["scores"], "windows": data.get("windows"), "grid": data.get("grid") or args.get("grid"),
            "file": data}


def _check_listing(rows, margins, windows, k, reasons, name, ordered=True):
    """Listed rows against the package: their margins are the k smallest in order (tie-invariant), each window's
    margin is the package's margin of that window."""
    local_of = {int(w): i for i, w in enumerate(windows)} if windows is not None else None
    if len(rows) > k:
        reasons.append(f"{name}: {len(rows)} windows listed for a review set of {k}")
    if ordered:
        pkg = np.sort(margins[review_order(-margins)[:k]])
        for j, r in enumerate(rows[:k]):
            if abs(float(r["margin"]) - pkg[j]) > TOL_ROUNDED:
                reasons.append(f"{name}: listed margin {j + 1} is {r['margin']}, the package's is {pkg[j]:.6f}")
                break
    for r in rows:
        wi = int(r["window_index"])
        i = local_of.get(wi) if local_of is not None else wi
        inside = i is not None and 0 <= i < len(margins)
        if not inside or abs(float(r["margin"]) - margins[i]) > TOL_ROUNDED:
            pkg = f"{margins[i]:.6f}" if inside else "none"
            reasons.append(f"{name}: window {wi} has margin {r['margin']} in the tool and {pkg} in the package")
            break


def parity_review_set(call, resolve):
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("review"):
        return NA, []
    src = _rows_of(args, resolve)
    if src is None:
        return UNGRADEABLE, ["the scores the review set ranked are not in the run directory"]
    margins = package_margins(src["scores"])
    n = margins.size
    k = min(n, max(1, int(round(float(args.get("budget", 0.05)) * n))))
    reasons = []
    if out.get("n_review") != k:
        reasons.append(f"olmoearth_review_set: n_review {out.get('n_review')}, the package's {k}")
    _check_listing(out["review"], margins, src["windows"], k, reasons, "olmoearth_review_set",
                   ordered=(args.get("order") or "confidence") == "confidence")
    return (FAIL if reasons else PASS), reasons


def parity_review_from_result(call, resolve):
    out = call["result"]
    if not isinstance(out, dict) or not out.get("ranked") or not out.get("review"):
        return NA, []
    f = _load_json(resolve, out.get("scores_path"))
    if not isinstance(f, dict):
        return UNGRADEABLE, ["the scores file the ranking wrote is not in the run directory"]
    scores, values, windows = f["scores"], f.get("values") or [], f.get("windows")
    reasons = []
    if f.get("score_kind") == "binary_score" and not all(
            abs(r[0] - (1 - s)) <= 1e-12 and abs(r[1] - s) <= 1e-12 for r, s in zip(scores, values)):
        reasons.append("the saved rows are not [1 - s, s] of the saved values")
    margins = package_margins(scores)
    n, asc = margins.size, np.sort(margins)
    ks = []
    for e in out.get("budgets") or []:
        k = min(n, max(1, int(round(float(e["budget"]) * n))))
        ks.append(k)
        if e.get("n_review") != k or abs(float(e.get("margin_cut", np.nan)) - asc[k - 1]) > TOL_ROUNDED:
            reasons.append(f"budget {e['budget']}: n_review {e.get('n_review')} and cut {e.get('margin_cut')}, the "
                           f"package's {k} and {asc[k - 1]:.6f}")
    _check_listing(out["review"], margins, windows, max(ks) if ks else n, reasons, "olmoearth_review_set_from_result")
    local_of = {int(w): i for i, w in enumerate(windows)} if windows is not None else None
    for r in out["review"]:
        i = local_of.get(int(r["window_index"])) if local_of is not None else int(r["window_index"])
        if _num(r.get("score")) and (i is None or abs(float(r["score"]) - values[i]) > TOL_ROUNDED):
            reasons.append(f"window {r['window_index']}: score {r['score']} is not the sampled value")
            break
    return (FAIL if reasons else PASS), reasons


def _population(src, n_windows):
    margins, p1 = package_margins(src["scores"]), _top1(src["scores"])
    idx = np.asarray(src["windows"] if src["windows"] is not None else range(margins.size), int)
    m, p = np.full(n_windows, np.nan), np.full(n_windows, np.nan)
    m[idx], p[idx] = margins, p1
    return m, p


def _design_margin(design):
    return np.array([np.nan if v is None else float(v) for v in design["population"]["margin"]], dtype=np.float64)


def parity_plan(call, resolve):
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("available") or not out.get("design_path"):
        return NA, []
    design = _load_json(resolve, out["design_path"])
    if not isinstance(design, dict):
        return UNGRADEABLE, ["the design file the plan wrote is not in the run directory"]
    pop = design["population"]
    src = _rows_of(args, resolve)
    reasons, note = [], None
    if src is not None:
        margin, p1 = _population(src, int(pop["n_windows"]))
        stored = _design_margin(design)
        if not np.allclose(np.nan_to_num(stored, nan=-9), np.nan_to_num(margin, nan=-9), atol=TOL_SAME, rtol=0):
            reasons.append("the design's population margins are not the package's margins of the same scores")
    else:
        margin = _design_margin(design)
        kind = pop.get("score_kind")
        if kind == "binary_score":
            p1 = (1 + margin) / 2
        elif kind == "threshold_distance":
            p1 = 1 / (1 + np.exp(-margin))
        else:
            return UNGRADEABLE, ["the plan's scores are not in the run directory and its top-1 probability cannot be "
                                 "recovered from the design file"]
        note = "margins taken from the design file (a live Studio sample)"
    name = design["design"]
    sample = oe_estimate.sample_for_estimation(margin, int(design["budget"]), design=name,
                                               p1=p1 if name == "confidence" else None, seed=int(design["seed"]))
    got = [int(i) for i in design["sample"]["indices"]]
    if got != [int(i) for i in sample["indices"]]:
        reasons.append("the design's sampled windows are not the package's draw on the same margins, budget and seed")
    listed = [int(w["window_index"]) for w in out.get("windows") or []]
    if listed != got[:len(listed)]:
        reasons.append("the windows the plan listed are not the design file's, in order")
    return (FAIL if reasons else PASS), reasons + ([note] if note and not reasons else [])


def _wrong(args, resolve, indices):
    if args.get("wrong") is not None:
        return [int(float(w)) for w in args["wrong"]]
    p = resolve(args.get("labels_path")) if args.get("labels_path") else None
    if p is None:
        return None
    with open(p, encoding="utf-8", newline="") as fh:
        sheet = {int(r["window_index"]): (r.get("wrong") or "").strip() for r in csv.DictReader(fh)}
    if any(i not in sheet or sheet[i] == "" for i in indices):
        return None
    return [int(float(sheet[i])) for i in indices]


def _compare_keys(tool, pkg, keys, reasons, name):
    for k in keys:
        a, b = tool.get(k, "<missing>"), pkg.get(k)
        if _num(a) and _num(b):
            if abs(float(a) - float(b)) > TOL_SAME * (1 + abs(float(b))):
                reasons.append(f"{name}: {k} {a}, the package's {b}")
        elif (a is None and isinstance(b, float) and math.isnan(b)) or a == b:
            continue
        else:
            reasons.append(f"{name}: {k} {a!r}, the package's {b!r}")


def parity_estimate(call, resolve):
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("available"):
        return NA, []
    if args.get("design_path"):
        design = _load_json(resolve, args["design_path"])
        if not isinstance(design, dict):
            return UNGRADEABLE, ["the design file is not in the run directory"]
        sample = design["sample"]
        wrong = _wrong(args, resolve, [int(i) for i in sample["indices"]])
        if wrong is None:
            return UNGRADEABLE, ["the labels the estimate used are not in the run directory"]
        pkg = oe_estimate.estimate_error_rate(sample, wrong)
    elif args.get("window_indices") is not None:
        src = _rows_of(args, resolve)
        indices = [int(i) for i in args["window_indices"]]
        wrong = _wrong(args, resolve, indices)
        if src is None or wrong is None:
            return UNGRADEABLE, ["the scores or labels of the estimate are not in the run directory"]
        pkg = oe_estimate.estimate_from_indices(indices, wrong, package_margins(src["scores"]))
    else:
        return NA, []
    reasons = []
    _compare_keys(out, pkg, ("design", "n_labelled", "n_population", "estimate", "low", "high", "method"), reasons,
                  "olmoearth_estimate_map_error")
    return (FAIL if reasons else PASS), reasons


def parity_certify(call, resolve):
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or not out.get("available"):
        return NA, []
    if out.get("certified") is False and out.get("reason") and out.get("design") not in (None, "random"):
        return NA, []                                      # the agent's own refusal of a non-random design
    design = _load_json(resolve, args.get("design_path"))
    if not isinstance(design, dict):
        return UNGRADEABLE, ["the design file is not in the run directory"]
    indices = [int(i) for i in design["sample"]["indices"]]
    wrong = _wrong(args, resolve, indices)
    if wrong is None:
        return UNGRADEABLE, ["the labels the zone used are not in the run directory"]
    pkg = oe_estimate.certify_zone(_design_margin(design), indices, wrong, float(args["alpha"]),
                                   delta=float(out.get("delta", args.get("delta", 0.1))),
                                   rule=str(out.get("rule", args.get("rule", "prefix"))))
    pkg.pop("zone_indices_in_order", None)
    reasons = []
    _compare_keys(out, pkg, ("rule", "alpha", "delta", "n_population", "n_labelled", "coverage", "n_zone", "threshold",
                             "upper_bound"), reasons, "olmoearth_certify_zone")
    return (FAIL if reasons else PASS), reasons


def parity_compare_review(call, resolve):
    out, args = call["result"], call["arguments"]
    if not isinstance(out, dict) or "n_differing" not in out:
        return NA, []
    a = _rows_of(args, resolve, "scores_a", "scores_path_a")
    b = _rows_of(args, resolve, "scores_b", "scores_path_b")
    if a is None or b is None:
        return UNGRADEABLE, ["the two score sets are not in the run directory"]
    da, db = np.argmax(np.asarray(a["scores"], float), 1), np.argmax(np.asarray(b["scores"], float), 1)
    reasons = []
    n_diff = int((da != db).sum())
    if out["n_differing"] != n_diff or abs(float(out.get("share_differing", -1)) - n_diff / da.size) > TOL_ROUNDED:
        reasons.append(f"olmoearth_compare_review: {out['n_differing']} differing, the package's {n_diff}")
    dates = out.get("dates") or {}
    if dates.get("assessed"):
        ref = oe_compare.dates_reading(args.get("date_a"), args.get("date_b"), args.get("labels_date"))
        _compare_keys(dates, ref, ("status", "days_apart", "a", "b", "labels"), reasons, "dates")
    return (FAIL if reasons else PASS), reasons


PARITY = {"olmoearth_review_set": parity_review_set, "olmoearth_review_set_from_result": parity_review_from_result,
          "olmoearth_plan_label_sample": parity_plan, "olmoearth_estimate_map_error": parity_estimate,
          "olmoearth_certify_zone": parity_certify, "olmoearth_compare_review": parity_compare_review}


def parity_of_calls(calls, resolve):
    """Every recomputable call against the package; a scorer error on a malformed file is ungradeable, never a pass."""
    per, statuses = [], []
    for c in calls:
        fn = PARITY.get(c["name"])
        if fn is None or not c.get("ok", True):
            continue
        try:
            st, why = fn(c, resolve)
        except Exception as exc:  # noqa: BLE001
            st, why = UNGRADEABLE, [f"{type(exc).__name__}: {exc}"]
        if st != NA:
            statuses.append(st)
            per.append({"tool": c["name"], "status": st, "reasons": why})
    status = FAIL if FAIL in statuses else UNGRADEABLE if UNGRADEABLE in statuses else PASS if statuses else NA
    return {"status": status, "calls": per, "reasons": [r for p in per for r in p["reasons"] if p["status"] != PASS]}


def grade_parity(run, resolve):
    return parity_of_calls(run["calls"], resolve)


def fixed_parity(round_dir, trial_dir):
    """Criterion 7a: the driver's direct calls on the fixtures, one file per tool, all five present."""
    d = os.path.join(round_dir, "parity")
    if not os.path.isdir(d):
        return {"status": UNGRADEABLE, "reasons": ["no parity/ directory: the fixed-input calls were not made"]}
    calls = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        rec = _read(p, "json") or {}
        calls.append({"id": os.path.basename(p), "name": rec.get("tool"), "arguments": rec.get("arguments") or {},
                      "ok": True, "result": _payload(rec.get("result"))})
    resolve = make_resolver(os.path.join(d, "workspace"), os.path.join(round_dir, "fixtures"),
                            os.path.join(trial_dir, "fixtures"))
    res = parity_of_calls(calls, resolve)
    missing = sorted(set(PARITY_FIXED) - {c["tool"] for c in res["calls"]})
    if missing:
        res["status"] = FAIL if res["status"] == FAIL else UNGRADEABLE
        res["reasons"].append(f"fixed-input parity calls missing: {missing}")
    return res


# --------------------------------------------------------------------------------------------- criterion 8 and a run
def time_and_tokens(run):
    """Descriptive: wall time, tokens, model calls, tool calls and turns of one run."""
    usage = run["usage"] or []
    tok = {k: sum(int(u.get(k) or 0) for u in usage) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    seconds = run["meta"].get("seconds")
    if not _num(seconds):
        ts = [ev.get("t") for ev in run["events"] or [] if _num(ev.get("t"))]
        seconds = max(ts) if ts else None
    turns = next((ev.get("turn") for ev in reversed(run["events"] or []) if _num(ev.get("turn"))), None)
    return {"seconds": seconds, **tok, "n_llm_calls": len(usage), "n_tool_calls": len(run["calls"]), "turns": turns,
            "usage_recorded": run["usage"] is not None}


def grade_run(run, config, resolve):
    spec = BRIEF_CONFIGS[config]
    return {"run": os.path.basename(run["dir"]),
            "c1_routing": grade_routing(run, spec),
            "c2_grounding": grade_grounding(run, resolve),
            "c3_ranking": grade_ranking(run, spec, resolve),
            "c4_nodata": grade_nodata(run),
            "c5_declines": grade_declines(run, spec, resolve),
            "c6_coordinates": grade_coordinates(run),
            "c7_parity": grade_parity(run, resolve),
            "c8_time_tokens": time_and_tokens(run)}


# --------------------------------------------------------------------------------------------- aggregation
def criterion_verdict(statuses):
    """All counted runs of a brief and configuration: any fail fails; any ungradeable is ungradeable; else pass."""
    if len(statuses) != N_RUNS:
        return "incomplete"
    if FAIL in statuses:
        return FAIL
    if UNGRADEABLE in statuses:
        return UNGRADEABLE
    return NA if all(s == NA for s in statuses) else PASS


def _describe(values):
    v = [float(x) for x in values if _num(x)]
    return {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None


def score_round(round_dir, trial_dir):
    meta = _read(os.path.join(round_dir, "round.json"), "json") or {}
    not_run = meta.get("not_run") or {}
    configs, unregistered = {}, []
    for bdir in sorted(glob.glob(os.path.join(round_dir, "runs", "*", "*"))):
        config = f"{os.path.basename(os.path.dirname(bdir))}/{os.path.basename(bdir)}"
        if config not in BRIEF_CONFIGS:
            unregistered.append(config)
            continue
        graded, excluded = [], []
        for rdir in sorted(d for d in glob.glob(os.path.join(bdir, "*")) if os.path.isdir(d)):
            run = load_run(rdir)
            if run["meta"].get("void"):
                excluded.append({"run": os.path.basename(rdir), "why": f"void: {run['meta']['void']}"})
                continue
            if not brief_matches(config, run["brief"]):
                excluded.append({"run": os.path.basename(rdir), "why": "the brief is not the preregistered text"})
                continue
            resolve = make_resolver(run["workspace"], os.path.join(round_dir, "fixtures"),
                                    os.path.join(trial_dir, "fixtures"))
            graded.append(grade_run(run, config, resolve))
        verdicts = {c: criterion_verdict([g[c]["status"] for g in graded]) for c in CRITERIA}
        tt = [g["c8_time_tokens"] for g in graded]
        configs[config] = {"n_counted_runs": len(graded), "excluded_runs": excluded, "verdicts": verdicts,
                           "passes": all(v in (PASS, NA) for v in verdicts.values()),
                           "time_and_tokens": {k: _describe([t[k] for t in tt]) for k in
                                               ("seconds", "prompt_tokens", "completion_tokens", "total_tokens",
                                                "n_tool_calls")},
                           "runs": graded}
    required = [c for c, s in BRIEF_CONFIGS.items() if not s.get("conditional")]
    missing = [c for c in required if c not in configs or configs[c]["n_counted_runs"] != N_RUNS]
    fixed = fixed_parity(round_dir, trial_dir)
    predictions = {}
    for i, crit in enumerate(CRITERIA, 1):
        per = {c: v["verdicts"][crit] for c, v in configs.items()}
        failing = sorted(c for c, v in per.items() if v == FAIL)
        unclear = sorted(c for c, v in per.items() if v in (UNGRADEABLE, "incomplete"))
        extra = [] if crit != "c7_parity" else ([] if fixed["status"] == PASS else ["fixed-input parity"])
        status = "fails" if failing or (extra and fixed["status"] == FAIL) else \
            "ungradeable" if unclear or extra else "incomplete" if missing else "holds"
        predictions[f"P{i}"] = {"criterion": crit, "status": status, "failing": failing, "unclear": unclear + extra}
    complete = not missing
    return {"round": os.path.basename(round_dir), "agent_commit": meta.get("agent_commit"),
            "inferencex_version": meta.get("inferencex_version"), "model": meta.get("model"),
            "fixes": meta.get("fixes") or [], "not_run": not_run, "complete": complete,
            "missing_or_incomplete": missing, "unregistered_configurations": unregistered,
            "fixed_input_parity": fixed, "predictions": predictions,
            "round_passes": complete and all(p["status"] == "holds" for p in predictions.values()),
            "configurations": configs}


def score_trial(trial_dir):
    rounds = sorted(d for d in glob.glob(os.path.join(trial_dir, "rounds", "*")) if os.path.isdir(d))
    if not rounds and os.path.isdir(os.path.join(trial_dir, "runs")):
        rounds = [trial_dir]
    trial = _read(os.path.join(trial_dir, "trial.json"), "json") or {}
    scored = [score_round(r, trial_dir) for r in rounds]
    passing = [r["round"] for r in scored if r["round_passes"]]
    return {"experiment": "exp86 agent trial v2", "preregistration": "docs/plan/agent_trial_v2.md",
            "model": trial.get("model", MODEL), "second_model": trial.get("second_model"),
            "runs_per_configuration": N_RUNS, "trial": {k: v for k, v in trial.items() if k != "model"},
            "n_rounds": len(scored),
            "verdict": {"trial_passes": bool(passing), "first_passing_round": passing[0] if passing else None,
                        "first_round": ({p: v["status"] for p, v in scored[0]["predictions"].items()} if scored
                                        else None),
                        "last_round": ({p: v["status"] for p, v in scored[-1]["predictions"].items()} if scored
                                       else None)},
            "rounds": scored}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trial", required=True, help="the trial directory the driver wrote")
    ap.add_argument("--out", default=os.path.join(OUT, "exp86_summary.json"))
    args = ap.parse_args()
    s = score_trial(args.trial)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1, default=_json_default)
    for r in s["rounds"]:
        print(f"round {r['round']} (agent {r['agent_commit']}): complete={r['complete']} passes={r['round_passes']}")
        print(f"  {'configuration':<12}" + "".join(f"{c.split('_')[0]:>13}" for c in CRITERIA))
        for cfg, v in r["configurations"].items():
            print(f"  {cfg:<12}" + "".join(f"{v['verdicts'][c]:>13}" for c in CRITERIA))
        print("  " + ", ".join(f"{p} {v['status']}" for p, v in r["predictions"].items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

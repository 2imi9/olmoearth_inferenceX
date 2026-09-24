#!/usr/bin/env python
"""exp86 fixtures: builds the trial directory of agent trial v2 before any agent run.

What it builds. docs/plan/agent_trial_v2.md, "Fixtures" and amendment A5, fix the inputs the briefs hand the agent;
this writes them, hashes them, and records where each came from:

  <trial>/fixtures/F1/dw_scores.json                    this package's Dynamic World tile as an agent scores file
  <trial>/fixtures/F2/design_confidence_300_s0.json     the agent's olmoearth_plan_label_sample on F1 (confidence,
  <trial>/fixtures/F2/labels_confidence_300_s0.csv        300, seed 0), its sheet filled from the expert labels
  <trial>/fixtures/F3/design_random_300_s0.json         the same with design="random"
  <trial>/fixtures/F3/labels_random_300_s0.csv
  <trial>/fixtures/F4/emsr279-11_s1_pre.json            exp60's pre- and post-event radar maps of GEOID-Flood event
  <trial>/fixtures/F4/emsr279-11_s1_post.json             EMSR279-11 on identical windows
  <trial>/fixtures/C1/awf_namanga_2023_1042061/         the provider's model run for 2023 (scores.tif, manifest.json)
  <trial>/fixtures/C2/awf_namanga_2022_1042294/         the same area for 2022, once the cluster job has written it
  <trial>/trial.json                                    every fixture file's sha256 ("fixtures", which the driver and
                                                        the scorer read), the model, and each fixture's provenance
  <trial>/rounds/1/round.json                           round 1's brief values (the fixtures' paths, B7's dates)

The labels. F1's expert labels are read from the package's sample at build time and never written under fixtures/,
except as the filled sheets of F2 and F3, which B5 and B6 hand to the agent.

F2 and F3 are made by the agent's own tool, as the plan says: its olmoearth_plan_label_sample handler, dispatched
through its tool registry with no model and no Studio, under the agent's interpreter (the agent must use its installed
inferencex extra, as the driver does). This file is that subprocess too (--agent-plan), so it imports only the standard
library and numpy at module level.

F4's dates come from the dataset's public release, not from this repository, which never recorded them: GEOID-Flood's
shard_index.json.gz on the Hugging Face Hub (links-ads/geoid-flood) at a pinned revision lists every member file, and
a file's name carries its acquisition time (EMSR279-11-<tile>_s1grd_pre_<UTC time>.tif). The pre-event map describes
the period of its tiles' pre-event acquisitions, the post-event map that of their post-event ones. The index is read,
never written to.

Usage, from this repository's root:
  uv run python exp/exp86_fixtures.py            # build exp/out/exp86_trial/ (refuses to overwrite a built trial)
  uv run python exp/exp86_fixtures.py --add-c2   # add C2 once the cluster job's output has landed
  uv run python exp/exp86_fixtures.py --check    # re-hash every fixture against trial.json
"""
import argparse
import asyncio
import csv
import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
TRIAL = os.path.join(EXP_DIR, "out", "exp86_trial")
MODEL = "nvidia/Qwen3.8-27B-NVFP4"
DW = os.path.join(ROOT, "oe_inferencex", "sample", "dynamic_world_tile.npz")
EXP60 = os.path.join(EXP_DIR, "out", "exp60_masks.npz")
EVENT = "EMSR279-11"
BUDGET, SEED = 300, 0

AGENT_REPO = os.path.expanduser("~/Desktop/Github/OlmoEarth-Agent")
AGENT_PY = os.path.join(AGENT_REPO, ".venv", "bin", "python")
#: The agent's files that decide what the plan tool draws; their hashes let a later commit of the same code (a sign-off
#: changes the hash, not the code) be recognised.
AGENT_CODE = ("src/olmoearth_agent/tools/estimation.py", "src/olmoearth_agent/tools/review_set.py",
              "src/olmoearth_agent/tools/inferencex.py", "src/olmoearth_agent/analysis/review_set.py")

SCRATCH = "/private/tmp/claude-501/-Users-lucas-Desktop-Github-olmoearth-inferenceX/" \
          "08a8fdd9-9831-4e63-bef7-8b20afa649c5/scratchpad/scores_provider"
C1_SRC = os.path.join(SCRATCH, "awf_namanga_2023_1042061")
C1_RUN = "awf_namanga_2023_1042061"
C1_SHA = "a7c40be990d84a5800e25ed323cd01656cfdb88784a2540a197c6c4df0172ca0"
C2_SRC = os.path.join(SCRATCH, "awf_namanga_2022_1042294")
C2_RUN = "awf_namanga_2022_1042294"
C2_SHA = "eddc4360134db0339dede21d067ed010439bc45c8e68be2d8749fd8cb79c2d02"   # checked against the cluster's copy
C2_JOB = 1042294
C2_WINDOW = ("2022-01-01", "2022-12-31")
PROVIDER_FILES = ("scores.tif", "manifest.json")          # nothing else from a run directory (the driver refuses more)

GEOID_REPO = "links-ads/geoid-flood"
GEOID_REVISION = "868407460bf3db492f50730a57585916baa71dc6"
GEOID_INDEX = "shard_index.json.gz"
_ACQ = re.compile(r"(?P<tile>[A-Za-z0-9]+-\d+-\d+)_s1grd_(?P<pas>pre|post)_(?P<t>\d{8}T\d{6})\.tif")

#: What the plan states about the fixtures (docs/plan/agent_trial_v2.md, "Fixtures"); the build stops if the data say
#: otherwise, because the fixtures would then not be the preregistered ones.
PLAN_F1 = {"n_windows": 15813, "n_left_out": 571, "review_cut": 791, "tie_at_cut": (1, 3), "error_rate": 0.193}
PLAN_F3 = {"coverage_at_0.05": None, "coverage_at_0.25": 0.9}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path, obj, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, separators=None if indent else (",", ":"))
        if indent:
            fh.write("\n")


# --------------------------------------------------------------------------------------------- F1
def f1_payload(probs, expert, classes):
    """F1 as an agent scores file: one row of class probabilities per window with a finite probability and an expert
    label, `windows` naming each row's grid index. No location: the tile's name and centre stay out of the file."""
    c, h, w = probs.shape
    keep = (np.isfinite(probs).all(0) & (expert >= 0)).ravel()
    windows = np.flatnonzero(keep)
    rows = probs.reshape(c, -1).T[keep]
    payload = {"format": "olmoearth-agent/scores@1", "score_kind": "class_scores", "grid": [int(h), int(w)],
               "windows": windows.tolist(), "scores": rows.tolist(),
               "classes": {str(i): str(n) for i, n in enumerate(classes)}}
    return payload, expert.ravel()[windows]


def f1_facts(payload, reference):
    """The facts the plan states about F1, recomputed from the file's own rows."""
    rows = np.asarray(payload["scores"], dtype=np.float64)
    srt = np.sort(rows, 1)
    margin = srt[:, -1] - srt[:, -2]
    n = margin.size
    k = max(1, int(round(0.05 * n)))
    cut = np.sort(margin)[k - 1]
    inside = int((np.sort(margin)[:k] == cut).sum())
    return {"n_windows": n, "n_left_out": payload["grid"][0] * payload["grid"][1] - n, "review_cut": k,
            "tie_at_cut": (inside, int((margin == cut).sum()) - inside),
            "error_rate": round(float((rows.argmax(1) != reference).mean()), 3),
            "row_sums": [float(rows.sum(1).min()), float(rows.sum(1).max())]}


def build_f1(fixtures):
    z = np.load(DW)
    meta = json.loads(str(z["meta"]))
    probs = z["probs"].astype(np.float32).astype(np.float64)       # the float16 values, exactly
    payload, reference = f1_payload(probs, z["expert"].astype(np.int64), meta["classes"])
    facts = f1_facts(payload, reference)
    wrong = {k: (facts[k], v) for k, v in PLAN_F1.items() if (tuple(facts[k]) if k == "tie_at_cut" else facts[k]) != v}
    if wrong:
        raise SystemExit(f"F1 is not the fixture the plan describes (found, planned): {wrong}")
    _write_json(os.path.join(fixtures, "F1", "dw_scores.json"), payload)
    by_window = dict(zip((int(w) for w in payload["windows"]), (int(r) for r in reference)))
    note = {"source": "oe_inferencex/sample/dynamic_world_tile.npz (Dynamic World test tile, Zenodo record "
                      f"{meta['zenodo_record']}, {meta['licence']}); probabilities stored in float16, written exactly",
            "classes": meta["classes"], **{k: list(v) if isinstance(v, tuple) else v for k, v in facts.items()},
            "left_out": "windows with no finite probability or no expert label (the expert's -1)"}
    return note, by_window


# --------------------------------------------------------------------------------------------- F2, F3 (the agent's tool)
class _NoStudio:
    """The plan tool reads a scores file; a Studio call would mean it was not given one."""

    def __getattr__(self, name):
        raise RuntimeError(f"the fixture build makes no Studio call (studio.{name})")


def agent_plan(root, design):
    """Run under the agent's interpreter: dispatch olmoearth_plan_label_sample on F1 and print its envelope as JSON."""
    os.environ["OLMOEARTH_SCORES_ROOT"] = root
    os.environ["OLMOEARTH_OUTPUT_ROOT"] = root
    os.chdir(root)
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.llm.types import ToolCall
    from olmoearth_agent.skills import build_default_registry
    from olmoearth_agent.tools.registry import ToolContext
    import oe_inferencex
    from importlib import metadata
    args = {"scores_path": "F1/dw_scores.json", "budget": BUDGET, "design": design, "seed": SEED}
    reg = build_default_registry()
    env = asyncio.run(reg.dispatch(ToolCall(id=f"fixture-{design}", name="olmoearth_plan_label_sample",
                                            arguments=dict(args)), ToolContext(studio=_NoStudio(), state=ThreadState())))
    print(json.dumps({"arguments": args, "envelope": env, "inferencex_file": oe_inferencex.__file__,
                      "inferencex_version": metadata.version("olmoearth-inferencex")}, default=str))


def agent_info():
    """The agent checkout the plan tool ran from: commit, branch, whether the tree is clean, and its code's hashes."""
    def git(*a):
        return subprocess.run(["git", "-C", AGENT_REPO, *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"repo": "github.com/2imi9/OlmoEarth-Agent", "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "code_sha256": {p: sha256_file(os.path.join(AGENT_REPO, p)) for p in AGENT_CODE}}


def fill_sheet(sheet_path, reference):
    """The reviewer's sheet with `wrong` (the map's class is not the expert's) and `reference_class` filled."""
    with open(sheet_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        truth = reference[int(r["window_index"])]
        r["wrong"] = str(int(int(r["map_class"]) != truth))
        r["reference_class"] = str(truth)
    with open(sheet_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows), sum(int(r["wrong"]) for r in rows)


def build_design(fixtures, tag, design, reference, info):
    if info["dirty"]:
        raise SystemExit(f"the agent's tree at {AGENT_REPO} has uncommitted changes; the fixtures must come from a "
                         "commit")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([AGENT_PY, os.path.abspath(__file__), "--agent-plan", fixtures, "--design", design],
                          cwd=fixtures, env=env, capture_output=True, text=True)
    if proc.returncode:
        raise SystemExit(f"the agent's plan tool failed for {tag}:\n{proc.stderr[-2000:]}")
    rec = json.loads(proc.stdout.strip().splitlines()[-1])
    env_ = rec["envelope"]
    out = env_.get("result") if isinstance(env_, dict) else None
    if not (isinstance(env_, dict) and env_.get("ok") and isinstance(out, dict) and out.get("available")):
        raise SystemExit(f"the agent's plan tool did not plan {tag}: {str(env_)[:500]}")
    if ROOT in os.path.realpath(rec["inferencex_file"]):
        raise SystemExit("the agent imported this repository's oe_inferencex, not its installed extra")
    stem = f"{design}_{BUDGET}_s{SEED}"
    design_rel, labels_rel = f"{tag}/design_{stem}.json", f"{tag}/labels_{stem}.csv"
    for src, rel in ((out["design_path"], design_rel), (out["labels_csv_path"], labels_rel)):
        dst = os.path.join(fixtures, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(fixtures, os.path.basename(src)), dst)
    n, n_wrong = fill_sheet(os.path.join(fixtures, labels_rel), reference)
    with open(os.path.join(fixtures, design_rel), encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("population", {}).get("source", {}).get("scores_path") != "F1/dw_scores.json" or n != BUDGET:
        raise SystemExit(f"{tag}: the design does not record F1 as its population, or holds {n} windows")
    note = {"made_by": "the agent's olmoearth_plan_label_sample, dispatched through its tool registry with no model and "
                       "no Studio", "arguments": rec["arguments"], "agent": info,
            "inferencex_version": rec["inferencex_version"], "design": design_rel, "labels": labels_rel,
            "n_labelled": n, "n_wrong": n_wrong,
            "labels_from": "F1's expert labels: wrong = 1 where the design's map_class is not the expert's class"}
    return note, design_rel, labels_rel


def f3_zone_facts(fixtures, design_rel, labels_rel):
    """The certified zones the plan states for F3, by this package's certify_zone on the design and its sheet."""
    sys.path.insert(0, ROOT)
    from oe_inferencex import estimate
    with open(os.path.join(fixtures, design_rel), encoding="utf-8") as fh:
        d = json.load(fh)
    with open(os.path.join(fixtures, labels_rel), encoding="utf-8", newline="") as fh:
        wrong_of = {int(r["window_index"]): int(r["wrong"]) for r in csv.DictReader(fh)}
    idx = [int(i) for i in d["sample"]["indices"]]
    margin = np.array([np.nan if v is None else float(v) for v in d["population"]["margin"]])
    facts = {}
    for alpha in (0.05, 0.25):
        z = estimate.certify_zone(margin, idx, [wrong_of[i] for i in idx], alpha)
        facts[f"coverage_at_{alpha}"] = z["coverage"]
    if facts != PLAN_F3:
        raise SystemExit(f"F3 is not the fixture the plan describes: {facts}, planned {PLAN_F3}")
    return facts


# --------------------------------------------------------------------------------------------- F4
def f4_payloads(masks, event):
    """exp60's pre-event (A_s1pre) and post-event (B_s1post) radar maps of one event on the windows valid in both.

    The event's chips (14 x 14 windows each) are stacked vertically into one grid, so window (r, c) is row r % 14 of
    chip r // 14. A window's row is [m, 0] or [0, m] from its margin m and decision (exp64's agent-arm mapping); the
    margins are float32 in exp60 and are written at float32's shortest exact precision."""
    sel = np.asarray(masks["event"]).astype(str) == event
    ok = np.asarray(masks["ok"])[sel].astype(bool)
    n, g = ok.shape[0], ok.shape[1]
    windows = np.flatnonzero(ok.reshape(-1))
    out = {}
    for tag, key in (("pre", "A_s1pre"), ("post", "B_s1post")):
        dec = np.asarray(masks[key])[sel].reshape(-1)[windows].astype(int)
        m = np.asarray(masks[f"margin_{key}"])[sel].reshape(-1)[windows].astype(np.float32)
        if not (np.isfinite(m).all() and (m > 0).all()):
            raise SystemExit(f"F4 {key}: a margin is not finite and positive, so a row could not carry its class")
        mm = [float(str(v)) for v in m]
        out[tag] = {"format": "olmoearth-agent/scores@1", "score_kind": "class_scores", "grid": [n * g, g],
                    "windows": windows.tolist(), "scores": [[v, 0.0] if d == 0 else [0.0, v] for v, d in zip(mm, dec)],
                    "classes": {"0": "not water", "1": "water"}}
    return out, {"n_chips": int(n), "n_windows": int(windows.size),
                 "n_differing": int(sum(np.argmax(a) != np.argmax(b) for a, b in
                                        zip(out["pre"]["scores"], out["post"]["scores"])))}


def acquisition_periods(member_names, event):
    """The event's pre- and post-event Sentinel-1 GRD acquisitions from the dataset's file names: per pass, the ISO
    period from the earliest to the latest acquisition day, and the count of tiles per acquisition time."""
    per = {"pre": {}, "post": {}}
    for name in member_names:
        m = _ACQ.search(name)
        if m and m.group("tile").rsplit("-", 1)[0] == event:
            per[m.group("pas")].setdefault(m.group("t"), set()).add(m.group("tile"))
    out = {}
    for pas, times in per.items():
        if not times:
            raise SystemExit(f"the index lists no {pas}-event Sentinel-1 file for {event}; no date is invented")
        days = sorted({t[:8] for t in times})
        iso = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in (days[0], days[-1])]
        out[pas] = {"period": iso[0] if iso[0] == iso[1] else f"{iso[0]}/{iso[1]}",
                    "tiles_per_acquisition": {t: len(v) for t, v in sorted(times.items())},
                    "n_tiles": len(set().union(*times.values()))}
    return out


def fetch_dates(event):
    """F4's dates from GEOID-Flood's public shard index at a pinned revision (read-only)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(GEOID_REPO, GEOID_INDEX, repo_type="dataset", revision=GEOID_REVISION)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    names = set(re.findall(r"[A-Za-z0-9]+-\d+-\d+_s1grd_(?:pre|post)_\d{8}T\d{6}\.tif", text))
    periods = acquisition_periods(names, event)
    return periods, {"url": f"https://huggingface.co/datasets/{GEOID_REPO}/blob/{GEOID_REVISION}/{GEOID_INDEX}",
                     "repo": GEOID_REPO, "revision": GEOID_REVISION, "file": GEOID_INDEX,
                     "file_sha256": sha256_file(path),
                     "how": "each member file's name carries its acquisition time "
                            "(<event>-<tile>_s1grd_<pre|post>_<YYYYMMDDTHHMMSS>.tif); a map's date is the period from "
                            "the earliest to the latest acquisition day of its pass over the event's tiles"}


def build_f4(fixtures):
    payloads, facts = f4_payloads(np.load(EXP60, allow_pickle=True), EVENT)
    periods, source = fetch_dates(EVENT)
    if periods["pre"]["n_tiles"] != periods["post"]["n_tiles"]:
        raise SystemExit(f"{EVENT}: the index lists a different number of tiles before and after the event")
    for tag, p in payloads.items():
        _write_json(os.path.join(fixtures, "F4", f"{EVENT.lower()}_s1_{tag}.json"), p)
    note = {"source": "exp/out/exp60_masks.npz (GEOID-Flood test chips, exp60): A_s1pre, the permanent-water head on "
                      "the pre-event Sentinel-1, and B_s1post, the water-after-the-event head on the post-event "
                      "Sentinel-1 (exp60's time-only pair), on the windows valid in both",
            "event": EVENT, **facts, "date_a": periods["pre"]["period"], "date_b": periods["post"]["period"],
            "acquisitions": periods, "date_source": source}
    return note, periods


# --------------------------------------------------------------------------------------------- C1, C2 (the provider's runs)
def copy_provider_run(fixtures, top, run, src, expected_sha=None):
    """scores.tif and manifest.json of one score_area.py run, and nothing else, checked against each other."""
    missing = [f for f in PROVIDER_FILES if not os.path.isfile(os.path.join(src, f))]
    if missing:
        return None, missing
    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    scores = manifest.get("scores") or {}
    got = sha256_file(os.path.join(src, "scores.tif"))
    if scores.get("file") != "scores.tif" or scores.get("sha256") != got or (expected_sha and got != expected_sha):
        raise SystemExit(f"{src}: scores.tif's sha256 {got} is not the manifest's {scores.get('sha256')} "
                         f"or the expected {expected_sha}")
    dst = os.path.join(fixtures, top, run)
    os.makedirs(dst, exist_ok=True)
    for f in PROVIDER_FILES:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    model = manifest.get("model") or {}
    note = {"source": src, "raster_sha256": got, "model": model.get("repo"), "revision": model.get("revision"),
            "date_window": {k: (manifest.get("date_window") or {}).get(k) for k in ("start", "end")},
            "shape": scores.get("shape"), "values": scores.get("values"), "area": manifest.get("area"),
            "copied": list(PROVIDER_FILES)}
    return note, []


def _place(area):
    """A manifest's area without its request's dates: the grid, bounds and requested centre and size."""
    area = dict(area or {})
    area["request"] = {k: v for k, v in (area.get("request") or {}).items() if k not in ("start", "end")}
    return area


def check_c2_against_c1(c1, c2):
    """C2 is the plan's C2: the same model, revision, area and grid as C1, with the 2022 date window."""
    problems = [k for k in ("model", "revision", "shape") if c1.get(k) != c2.get(k)]
    if _place(c1.get("area")) != _place(c2.get("area")):
        problems.append("area")
    if (c2["date_window"]["start"], c2["date_window"]["end"]) != C2_WINDOW:
        problems.append(f"date window {c2['date_window']}, not {C2_WINDOW}")
    if problems:
        raise SystemExit(f"C2 differs from the plan's C2 (the same run as C1 for 2022): {problems}")


# --------------------------------------------------------------------------------------------- the trial
def fixture_hashes(fixtures):
    return {os.path.relpath(os.path.join(h, f), fixtures): sha256_file(os.path.join(h, f))
            for h, _, fs in os.walk(fixtures) for f in sorted(fs)}


def expected_files():
    """Every file fixtures/ may hold: the driver refuses a run when any file there is not hashed in trial.json, and the
    provider's fixtures may hold only their runs' manifest and raster."""
    stem = f"{BUDGET}_s{SEED}"
    return ({"F1/dw_scores.json", f"F2/design_confidence_{stem}.json", f"F2/labels_confidence_{stem}.csv",
             f"F3/design_random_{stem}.json", f"F3/labels_random_{stem}.csv", f"F4/{EVENT.lower()}_s1_pre.json",
             f"F4/{EVENT.lower()}_s1_post.json"}
            | {f"{top}/{run}/{f}" for top, run in (("C1", C1_RUN), ("C2", C2_RUN)) for f in PROVIDER_FILES})


def unexpected_files(fixtures):
    return sorted(set(fixture_hashes(fixtures)) - expected_files())


def brief_values(f4_dates):
    """Round 1's brief values: the fixtures' paths, relative to the workspace root, and B7's dates."""
    c1, c2 = f"C1/{C1_RUN}", f"C2/{C2_RUN}"
    stem = f"{BUDGET}_s{SEED}"
    return {"B3/cluster": {"run_dir_a": c1, "run_dir_b": c2},
            "B4/cluster": {"run_dir": c1}, "B8/cluster": {"run_dir": c1},
            "B5/files": {"design_path": f"F2/design_confidence_{stem}.json",
                         "labels_path": f"F2/labels_confidence_{stem}.csv"},
            "B6/files": {"design_path": f"F3/design_random_{stem}.json", "labels_path": f"F3/labels_random_{stem}.csv"},
            "B7/files": {"scores_a": f"F4/{EVENT.lower()}_s1_pre.json", "scores_b": f"F4/{EVENT.lower()}_s1_post.json",
                         "date_a": f4_dates["pre"]["period"], "date_b": f4_dates["post"]["period"]}}


def build(trial, c1_src=C1_SRC, c2_src=C2_SRC):
    if os.path.exists(os.path.join(trial, "trial.json")):
        raise SystemExit(f"{trial} is built; fixtures are made once (use --add-c2 or --check, or remove it)")
    staging = trial + ".building"
    shutil.rmtree(staging, ignore_errors=True)
    fixtures = os.path.join(staging, "fixtures")
    os.makedirs(fixtures)
    notes, pending = {}, {}
    notes["F1"], reference = build_f1(fixtures)
    info = agent_info()
    notes["F2"], *_ = build_design(fixtures, "F2", "confidence", reference, info)
    notes["F3"], d3, l3 = build_design(fixtures, "F3", "random", reference, info)
    notes["F3"]["certified"] = f3_zone_facts(fixtures, d3, l3)
    notes["F4"], dates = build_f4(fixtures)
    notes["C1"], missing = copy_provider_run(fixtures, "C1", C1_RUN, c1_src, C1_SHA)
    if notes["C1"] is None:
        raise SystemExit(f"C1 is missing {missing} in {c1_src}")
    notes["C2"], missing = copy_provider_run(fixtures, "C2", C2_RUN, c2_src, C2_SHA)
    if notes["C2"] is None:
        del notes["C2"]
        pending["C2"] = {"expected_at": c2_src, "cluster_job": C2_JOB, "date_window": list(C2_WINDOW),
                         "missing": missing, "then": "uv run python exp/exp86_fixtures.py --add-c2 (B3 cluster cannot "
                                                     "run until C2 is a fixture; the driver refuses it)"}
    else:
        check_c2_against_c1(notes["C1"], notes["C2"])
    stray = unexpected_files(fixtures)
    if stray:
        raise SystemExit(f"fixtures/ holds files the plan does not name (a tool wrote them?): {stray}")
    meta = {"experiment": "exp86 agent trial v2", "preregistration": "docs/plan/agent_trial_v2.md (Fixtures; "
            "amendment A1 and A5)", "model": MODEL, "built": _now(), "built_by": "exp/exp86_fixtures.py",
            "fixtures": fixture_hashes(fixtures), "fixture_notes": notes, "pending": pending,
            "brief_values": brief_values(dates)}
    _write_json(os.path.join(staging, "trial.json"), meta, indent=1)
    _write_json(os.path.join(staging, "rounds", "1", "round.json"),
                {"brief_values": meta["brief_values"], "fixes": [],
                 "not_run": {"B8/studio": "the Studio account of the first trial (24 September) held no classification "
                                          "model: KarstEmbedding, KarstNumber and KarstBinary only"}}, indent=1)
    os.rename(staging, trial)
    return meta


def add_c2(trial, c2_src=C2_SRC):
    with open(os.path.join(trial, "trial.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if "C2" not in meta.get("pending", {}):
        raise SystemExit("C2 is not pending in trial.json")
    fixtures = os.path.join(trial, "fixtures")
    note, missing = copy_provider_run(fixtures, "C2", C2_RUN, c2_src, C2_SHA)
    if note is None:
        raise SystemExit(f"C2 has not landed: {missing} missing in {c2_src}")
    try:
        check_c2_against_c1(meta["fixture_notes"]["C1"], note)
    except SystemExit:
        shutil.rmtree(os.path.join(fixtures, "C2"))
        raise
    meta["fixture_notes"]["C2"] = note
    del meta["pending"]["C2"]
    meta["fixtures"] = fixture_hashes(fixtures)
    meta["c2_added"] = _now()
    _write_json(os.path.join(trial, "trial.json"), meta, indent=1)
    return meta


def check(trial):
    with open(os.path.join(trial, "trial.json"), encoding="utf-8") as fh:
        listed = json.load(fh)["fixtures"]
    have = fixture_hashes(os.path.join(trial, "fixtures"))
    return sorted(set(have) ^ set(listed)) + sorted(k for k in have if k in listed and have[k] != listed[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trial", default=TRIAL)
    ap.add_argument("--c1", default=C1_SRC, help="the 2023 run directory (scores.tif, manifest.json)")
    ap.add_argument("--c2", default=C2_SRC, help="the 2022 run directory, when it has landed")
    ap.add_argument("--add-c2", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--agent-plan", help=argparse.SUPPRESS)
    ap.add_argument("--design", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.agent_plan:
        agent_plan(a.agent_plan, a.design)
        return
    if a.check:
        bad = check(a.trial)
        print("fixtures match trial.json" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    meta = add_c2(a.trial, a.c2) if a.add_c2 else build(a.trial, a.c1, a.c2)
    for rel, h in sorted(meta["fixtures"].items()):
        size = os.path.getsize(os.path.join(a.trial, "fixtures", rel))
        print(f"  {rel:<52} {size:>10,} B  {h[:16]}")
    print(f"pending: {sorted(meta['pending']) or 'nothing'}")


if __name__ == "__main__":
    main()

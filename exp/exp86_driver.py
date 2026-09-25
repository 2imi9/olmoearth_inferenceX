#!/usr/bin/env python
"""exp86 driver: runs one configuration of agent trial v2 through the OlmoEarth Agent and writes its run directory.

Why. The scorer (exp/exp86_agent_trial_v2.py) grades run directories and runs nothing. The agent's command line prints
the tool names and the answer, and its provenance manifest holds only a hash of each call's arguments; criteria 2, 3,
4, 7 and 8 need the arguments, the tool outputs, the Studio samples and the tokens. So, as the plan prescribes
(docs/plan/agent_trial_v2.md, "The run directory"), this driver builds the agent as its command line builds it
(olmoearth_agent.cli.run_brief) and consumes LeadAgent.run_stream, whose events carry the full tool results.

How it records without changing what the agent does. Nothing in the agent is patched; three things are observed by
composition.
- run_stream's events are written to events.jsonl as they arrive, each with its line number (seq, from 0). A tool
  call is dispatched between its tool_call event and its tool_result event, so the last tool_call event is the call
  every Studio request belongs to until the matching tool_result arrives. No wrapper around the tool registry is
  needed, which keeps the driver independent of the registry's API (it differs between the agent's branches).
- The Studio client is wrapped (RecordingStudio). Every coroutine method a tool calls is passed to the real client
  unchanged, and the response is appended to studio_calls.jsonl with the call it belongs to: call_seq (the seq of its
  tool_call event), call_id and turn. An id alone can repeat across turns; (call_seq, call_id) and (turn, call_id)
  cannot. The tool_result event carries the same call_seq. A pixel-value sample is keyed by a keyed hash of its point
  (the key is random per run and never written), never by its coordinates; keys that hold coordinates or credentials
  are dropped from the recorded records, and each entry lists what was dropped.
- The LLM client is wrapped (RecordingLLM): each chat() call is passed through, and its usage, wall time and the tools
  it was offered (the core tools and any deferred group loaded so far) are appended to usage.jsonl. A tracer (the
  client's own constructor hook) records the sampling settings actually sent.

The cluster provider (olmoearth_scores_from_file) reads a model run's directory, the scores raster and manifest.json
that scripts/score_area.py writes, under the scores root. That directory is a fixture, hashed in trial.json like the
others, and only a cluster run's workspace receives it: a studio or files run never sees it. Which fixtures each
configuration receives is the scorer's WORKSPACE_FIXTURES table when it has one (read from its source, like the
briefs); without it, every fixture but the provider's, and the provider's for a cluster run.

What it refuses. Nothing is written for a refused run. A run is refused when the configuration or its brief is not
the preregistered one, when the agent's command line builds LeadAgent differently from build_agent below or anything
else the driver reads from the agent has changed (check_agent), when a fixture differs from its sha256 in trial.json,
when a file the brief names is not a fixture, when a cluster configuration finds no provider tool or no provider
fixture (or one holding more than its manifest and raster, or a raster its manifest does not record), when a counted
cluster run's scorer names another provider tool, or when the round was started at another agent commit, inferencex
version, model or sampling (a fix starts a new round). A counted run, one under <trial>/rounds/, is also refused
when the agent's tree has uncommitted changes or its soul or skills are overridden from the environment, because the
commit would not then identify the agent; --allow-dirty permits both outside rounds/, for the debugging runs the plan
keeps apart.

Secrets. The Studio key and the LLM settings come from the environment the maintainer sets up
(~/.config/olmoearth-agent/oe-agent.sh loads the agent's .env and qwen.env). The driver never prints them, and every
file it writes is checked, before it is written, for the value of every secret-like environment variable (SECRET_ENV,
and names such as *_KEY, *_TOKEN and *_SECRET); a file that would hold one is not written, and the run ends as an
error naming the variable, never its value. After the run every file in the run directory, including those the tools
wrote, is checked again. The model id (LLM_MODEL) is written, because the plan records the model; the endpoint and the
keys never are.

Usage, from this repository's root, with the agent's interpreter. Not this repository's: the agent must use its
installed inferencex extra, which is also why the scorer is read here with ast and never imported (importing it puts
this repository's oe_inferencex first on sys.path).
  set -a; source ~/Desktop/Github/OlmoEarth-Agent/.env; source ~/.config/olmoearth-agent/qwen.env; set +a
  PY=~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python
  $PY exp/exp86_driver.py --brief B2 --provider studio --run 1 --out <trial>/rounds/1     # one run
  $PY exp/exp86_driver.py --all --out <trial>/rounds/1           # every configuration to three counted runs
  $PY exp/exp86_driver.py --parity --f1 <F1 file> --out <trial>/rounds/1                   # criterion 7a, once
  python exp/exp86_agent_trial_v2.py --trial <trial>                                       # then score
--out is the round directory; a run is written to <out>/runs/<brief>/<provider>/<run>/. Exit codes: 0 answered,
1 no answer or an error (as the agent's command line), 2 void, 3 refused before the run or a secret found.
"""
import argparse
import ast
import asyncio
import collections
import contextlib
import dataclasses
import datetime
import functools
import glob
import hashlib
import hmac
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EXP_DIR)
SCORER = os.path.join(EXP_DIR, "exp86_agent_trial_v2.py")
EXPERIMENT = "exp86 agent trial v2"

#: The agent command line's default turn cap, which the plan keeps ("Setup": the command line's default of 8 turns).
MAX_TURNS = 8
N_RUNS = 3
EXIT_ANSWERED, EXIT_NO_ANSWER, EXIT_VOID, EXIT_REFUSED = 0, 1, 2, 3

SCORES_ROOT_ENV = "OLMOEARTH_SCORES_ROOT"       # olmoearth_agent.tools.review_set.SCORES_ROOT_ENV
OUTPUT_ROOT_ENV = "OLMOEARTH_OUTPUT_ROOT"       # olmoearth_agent.security.paths.OUTPUT_ROOT_ENV (memory, spills)
#: Always treated as secret. LLM_MODEL is not: the plan records the model.
SECRET_ENV = ("OLMOEARTH_API_KEY", "LLM_API_KEY", "LLM_ENDPOINT")
_SECRET_NAME = re.compile(r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_|$)", re.I)
#: Values that are not secrets however they are named: placeholders, and the agent's own published defaults (which its
#: docs and skills may quote, so treating them as secret would refuse runs for nothing).
_NOT_SECRET = {"", "empty", "none", "null", "replace-me", "changeme", "true", "false"}
PUBLIC_DEFAULTS = {"http://localhost:8000/v1", "https://olmoearth.allenai.org/api/v1"}
MIN_SECRET_CHARS = 8
#: Non-secret settings that change what the agent does; their values are recorded in run.json.
RECORDED_ENV = ("OLMOEARTH_BASE_URL", "OLMOEARTH_EGRESS", "OLMOEARTH_EGRESS_ALLOW", "OLMOEARTH_RUN_PYTHON",
                "OLMOEARTH_RUN_PYTHON_TIMEOUT", "OLMOEARTH_TOOL_RESULT_SPILL_BYTES", "OLMOEARTH_SOUL_PATH",
                "OLMOEARTH_SKILLS_DIR")
#: Settings that replace the agent's committed soul or skills, so its commit would no longer identify it.
OVERRIDING_ENV = ("OLMOEARTH_SOUL_PATH", "OLMOEARTH_SKILLS_DIR")
#: The cluster scores provider (agent branch 2imi9/feature-cluster-scores-provider, 68f39ee): it reads a model run's
#: directory, scores raster and manifest.json as scripts/score_area.py writes them, under the scores root. It takes no
#: environment; a cluster run needs that directory in its workspace, and only a cluster run gets it ("Setup": studio has
#: no cluster provider).
PROVIDER_TOOL = "olmoearth_scores_from_file"
PROVIDER_MANIFEST = "manifest.json"
#: How olmoearth_agent.cli.run_brief builds LeadAgent; build_agent below does the same, and check_agent refuses to
#: run when the command line changes.
CLI_LEADAGENT_ARGS = ["llm", "registry", "studio"]
CLI_LEADAGENT_KEYWORDS = {"state": "ThreadState()", "skill_index": "skill_index",
                          "memory_block": "preferences_block()", "local": "True"}


class DriverError(RuntimeError):
    """A run the driver refuses to make, with the reason."""


class SecretLeak(RuntimeError):
    """A file would hold a secret's value. The message names the file and the variable, never the value."""


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- the preregistered briefs
@dataclasses.dataclass
class ScorerConstants:
    briefs: dict          # configuration -> brief template
    provider: tuple       # the cluster scores provider's tool names
    conditional: set      # configurations that run only when the plan's condition holds (B8 studio)
    workspace: dict = None  # configuration (and "parity") -> the fixture top directories its workspace receives


def scorer_constants(path=SCORER):
    """BRIEF_CONFIGS' brief texts, PROVIDER and the conditional configurations, read from the scorer's source.

    Read, not imported: importing the scorer inserts this repository's root first on sys.path and imports its
    oe_inferencex, which the agent's tools would then use instead of their installed extra, so criterion 7 would compare
    this package with itself."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    briefs, provider, conditional, workspace = {}, None, set(), None
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name == "PROVIDER":
            provider = tuple(ast.literal_eval(node.value))
        elif name == "BRIEF_CONFIGS" and isinstance(node.value, ast.Dict):
            for key, spec in zip(node.value.keys, node.value.values):
                config = ast.literal_eval(key)
                fields = {ast.literal_eval(k): v for k, v in zip(spec.keys, spec.values) if isinstance(k, ast.Constant)}
                briefs[config] = ast.literal_eval(fields["brief"])
                if "conditional" in fields and ast.literal_eval(fields["conditional"]):
                    conditional.add(config)
        elif name == "WORKSPACE_FIXTURES":
            workspace = {k: tuple(v) for k, v in ast.literal_eval(node.value).items()}
    if not briefs or provider is None:
        raise DriverError(f"could not read BRIEF_CONFIGS and PROVIDER from {path}")
    return ScorerConstants(briefs, provider, conditional, workspace)


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _template_regex(template):
    """The scorer's brief_matches pattern (exp86_agent_trial_v2._template_regex), rebuilt: see scorer_constants."""
    esc = re.escape(" ".join(template.split()))
    return re.compile(re.sub(r"\\\{[a-z_]+\\\}", "(.+?)", esc), re.S)


def fill_brief(template, values):
    """The brief with its {placeholders} filled from round.json's brief_values for the configuration."""
    names = _PLACEHOLDER.findall(template)
    missing = [n for n in names if not str((values or {}).get(n, "")).strip()]
    if missing:
        raise DriverError(f"round.json's brief_values lack {missing} for this configuration")
    text = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]).strip(), template)
    if not _template_regex(template).fullmatch(" ".join(text.split())):
        raise DriverError("the filled brief is not the preregistered text with only its braces filled")
    return text


def brief_files(template, values):
    """The brief values that name files the agent is handed (F2 to F4), by placeholder."""
    return {n: str(values[n]).strip() for n in _PLACEHOLDER.findall(template)
            if n.endswith("_path") or n in ("scores_a", "scores_b")}


def brief_run_dirs(template, values):
    """The brief values that name a provider model run's directory (the cluster briefs' {run_dir}), by placeholder."""
    return {n: str(values[n]).strip().rstrip("/") for n in _PLACEHOLDER.findall(template) if n.startswith("run_dir")}


# --------------------------------------------------------------------------------------------- secrets
def environment_secrets(environ=None):
    """Name -> value of every environment variable whose value must never reach a file."""
    env = os.environ if environ is None else environ
    out = {}
    for name, value in env.items():
        v = (value or "").strip()
        if not (name in SECRET_ENV or _SECRET_NAME.search(name)):
            continue
        if len(v) < MIN_SECRET_CHARS or v.lower() in _NOT_SECRET or v in PUBLIC_DEFAULTS:
            continue
        out[name] = v
    return out


def _forms(value):
    """The byte forms a secret can take in a written file: as is, and escaped inside a JSON string."""
    return {value.encode("utf-8"), json.dumps(value)[1:-1].encode("utf-8")}


def secrets_in(data, secrets):
    """Names of the secrets whose value occurs in `data` (bytes or str)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return sorted(name for name, value in secrets.items() if any(f in data for f in _forms(value)))


def redact(text, secrets):
    """Driver-authored text (an error message) with every secret value replaced by the variable's name."""
    text = str(text)
    for name, value in secrets.items():
        for form in (value, json.dumps(value)[1:-1]):
            text = text.replace(form, f"<{name}>")
    return text


class RunWriter:
    """Writes files under one root, each checked for secret values before it is written."""

    def __init__(self, root, secrets):
        self.root = os.path.abspath(root)
        self.secrets = dict(secrets)
        self.n_checked = 0

    def path(self, rel):
        return os.path.join(self.root, rel)

    def _check(self, rel, data):
        self.n_checked += 1
        names = secrets_in(data, self.secrets)
        if names:
            raise SecretLeak(f"{rel}: the value of {', '.join(names)} would be written; the file was not written")

    def bytes(self, rel, data, atomic=False):
        self._check(rel, data)
        target = self.path(rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp" if atomic else target
        with open(tmp, "wb") as fh:
            fh.write(data)
        if atomic:
            os.replace(tmp, target)
        return target

    def text(self, rel, text, atomic=False):
        return self.bytes(rel, text.encode("utf-8"), atomic=atomic)

    def json(self, rel, obj, atomic=False):
        return self.text(rel, json.dumps(obj, indent=1, default=str) + "\n", atomic=atomic)

    def jsonl(self, rel, obj):
        line = (json.dumps(obj, default=str) + "\n").encode("utf-8")
        self._check(rel, line)
        target = self.path(rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "ab") as fh:
            fh.write(line)

    def copy(self, src, rel):
        with open(src, "rb") as fh:
            return self.bytes(rel, fh.read())

    def sweep(self):
        """Every file under the root (the tools' files included) that holds a secret: [(relative path, names)]."""
        found = []
        for here, _, files in os.walk(self.root):
            for f in sorted(files):
                p = os.path.join(here, f)
                try:
                    with open(p, "rb") as fh:
                        names = secrets_in(fh.read(), self.secrets)
                except OSError:
                    continue
                if names:
                    found.append((os.path.relpath(p, self.root), names))
        return found


# --------------------------------------------------------------------------------------------- recording Studio
_COORD_KEYS = frozenset({"lon", "lat", "lng", "longitude", "latitude", "lonlat", "latlon", "lon_lat", "lat_lon",
                         "coordinates", "bbox", "bounds", "geometry", "geom", "wkt", "centres_lon_lat",
                         "centers_lon_lat", "queried_point", "point", "points", "centroid", "extent", "footprint"})
_CREDENTIAL_KEY = re.compile(r"token|secret|passw|api_?key|authori[sz]ation|credential|signature|^urls?$|_urls?$",
                             re.I)
#: Studio methods whose responses the scorer indexes (exp86_agent_trial_v2._studio_index), by kind.
_KINDS = {"pixel_value": "pixel_value", "get_prediction_result": "prediction_result", "get_prediction": "prediction",
          "get_model": "model"}
_ID_ARGS = ("result_id", "prediction_id", "model_id", "project_id", "area_id", "limit", "offset")
_UNRECORDED = frozenset({"aclose"})


def _drop_key(k):
    k = str(k).lower()
    return (k in _COORD_KEYS or k.endswith(("_bbox", "_lon", "_lat", "_lonlat", "_lon_lat", "_geom", "_geometry",
                                            "_bounds", "_wkt", "_coordinates"))
            or k.startswith(("lon_", "lat_", "bbox_")) or bool(_CREDENTIAL_KEY.search(k)))


def sanitize(obj, dropped, path=""):
    """A Studio record without keys that hold coordinates or credentials; the dropped key paths go to `dropped`."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            where = f"{path}.{k}" if path else str(k)
            if _drop_key(k):
                dropped.append(where)
                continue
            out[k] = sanitize(v, dropped, where)
        return out
    if isinstance(obj, list):
        return [sanitize(v, dropped, f"{path}[]") for v in obj]
    return obj


#: A Studio account record (users/me) is recognised by these keys; its identity is withheld whole, since its id,
#: name, e-mail and organisations name the account holder and the record is published.
_ACCOUNT_MARKERS = frozenset({"firebase_user_id", "last_login_time", "terms_accepted_at"})
#: Keys whose value names the person, in any record or tool result.
_PERSON_KEYS = frozenset({"email", "user_email", "user_name", "firebase_user_id"})
#: A signed URL's query carries a signature (and the signer's account); the path alone is kept.
_SIGNED_URL = re.compile(r"[?&](?:X-Goog-Signature|X-Amz-Signature|Signature)=", re.I)
WITHHELD = "[withheld]"


class Redactor:
    """Withholds the account holder's identity and URL signatures from what a run writes.

    The identity values (e-mail, name, ids) are learned from the account records and person keys it sees, so a later
    occurrence in free text (a thinking trace, the answer) is withheld too. Coordinates and credentials are
    `sanitize`'s; this covers what identifies a person in a published record (the trial's audit of round 6 found the
    Studio account's e-mail and ids in every round's studio_calls.jsonl)."""

    def __init__(self):
        self.values = set()

    def learn(self, obj):
        if isinstance(obj, dict):
            account = "email" in obj and bool(_ACCOUNT_MARKERS & set(obj))
            for k, v in obj.items():
                if isinstance(v, str) and len(v) >= 4 and (str(k).lower() in _PERSON_KEYS or
                                                            (account and k in ("id", "name"))):
                    self.values.add(v)
                self.learn(v)
        elif isinstance(obj, list):
            for v in obj:
                self.learn(v)

    def text(self, t):
        for v in sorted(self.values, key=len, reverse=True):
            t = t.replace(v, WITHHELD)
        return t

    def __call__(self, obj, dropped=None, path=""):
        dropped = [] if dropped is None else dropped
        if isinstance(obj, dict):
            if "email" in obj and _ACCOUNT_MARKERS & set(obj):
                dropped.append(path or "<record>")
                return {"withheld": "a Studio account record: the account holder's identity is not published"}
            out = {}
            for k, v in obj.items():
                where = f"{path}.{k}" if path else str(k)
                if str(k).lower() in _PERSON_KEYS and v not in (None, ""):
                    dropped.append(where)
                    out[k] = WITHHELD
                else:
                    out[k] = self(v, dropped, where)
            return out
        if isinstance(obj, list):
            return [self(v, dropped, f"{path}[]") for v in obj]
        if isinstance(obj, str):
            if _SIGNED_URL.search(obj):
                dropped.append(f"{path}?signature")
                obj = obj.split("?")[0]
            return self.text(obj)
        return obj


def jsonable(o, depth=0):
    """A Studio response as JSON data: records, envelopes (dataclasses) and contexts; an HTTP response as its status."""
    if depth > 50:
        return f"<{type(o).__name__}>"
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v, depth + 1) for v in o]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return jsonable(dataclasses.asdict(o), depth + 1)
    if hasattr(o, "model_dump"):
        return jsonable(o.model_dump(), depth + 1)
    if hasattr(o, "status_code"):
        return {"status_code": getattr(o, "status_code", None)}
    if hasattr(o, "tolist"):
        return jsonable(o.tolist(), depth + 1)
    return f"<{type(o).__name__}>"


class StudioLog:
    """studio_calls.jsonl: one entry per Studio response a tool received (the plan's run directory)."""

    def __init__(self, writer, clock, t0):
        self.writer, self.clock, self.t0 = writer, clock, t0
        # The tool call in progress, set from run_stream's events: the tool_call event's sequence number in
        # events.jsonl, the call's id and its turn. All three are written: an id alone can repeat (a call the client
        # recovers from the model's text is numbered call_0, call_1, ... afresh on every turn), while (call_seq, id)
        # and (turn, id) each name one call.
        self.call_seq = None
        self.call_id = None
        self.call_turn = None
        self.n = 0
        self.errors = collections.Counter()
        self.leaks = []
        self._key = os.urandom(32)               # never written: the point keys cannot be inverted to coordinates
        self.redact = Redactor()                 # the account holder's identity and URL signatures are not written

    def point_key(self, lon, lat):
        msg = f"{float(lon):.9f},{float(lat):.9f}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()[:20]

    def add(self, method, fn, args, kwargs, out, exc, started):
        entry = {"seq": self.n, "kind": _KINDS.get(method, "response"), "method": method,
                 "call_seq": self.call_seq, "call_id": self.call_id, "turn": self.call_turn,
                 "t": round(started - self.t0, 4), "seconds": round(self.clock() - started, 4)}
        self.n += 1
        try:
            bound = dict(inspect.signature(fn).bind_partial(*args, **kwargs).arguments)
        except (TypeError, ValueError):
            bound = {}
        for k in _ID_ARGS:
            if isinstance(bound.get(k), (str, int, float)) and not isinstance(bound.get(k), bool):
                entry[k] = bound[k]
        if method == "pixel_value" and "lon" in bound and "lat" in bound:
            entry["point"] = self.point_key(bound["lon"], bound["lat"])
        where = bound.get("path") or bound.get("url")
        if isinstance(where, str):
            entry["path"] = where.split("?")[0]
        if exc is not None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            entry["record"] = None
            entry["error"] = {"type": type(exc).__name__, "status": status}
            self.errors[str(status) if status is not None else type(exc).__name__] += 1
        else:
            dropped = []
            record = sanitize(jsonable(out), dropped)
            self.redact.learn(record)
            entry["record"] = self.redact(record, dropped)
            if dropped:
                entry["dropped_keys"] = sorted(set(dropped))
        try:
            self.writer.jsonl("studio_calls.jsonl", entry)
        except SecretLeak as leak:
            self.leaks.append(str(leak))
            with contextlib.suppress(SecretLeak):
                self.writer.jsonl("studio_calls.jsonl", {
                    "seq": entry["seq"], "kind": entry["kind"], "method": method, "call_seq": entry["call_seq"],
                    "call_id": entry["call_id"], "turn": entry["turn"],
                    "withheld": "the response held the value of a secret environment variable"})


class RecordingStudio:
    """The Studio client, observed: every coroutine method passes through unchanged and its response is logged."""

    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name.startswith("_") or name in _UNRECORDED or not inspect.iscoroutinefunction(attr):
            return attr
        log = self._log

        @functools.wraps(attr)
        async def recorded(*args, **kwargs):
            started = log.clock()
            try:
                out = await attr(*args, **kwargs)
            except Exception as exc:
                log.add(name, attr, args, kwargs, None, exc, started)
                raise
            log.add(name, attr, args, kwargs, out, None, started)
            return out
        return recorded


# --------------------------------------------------------------------------------------------- recording LLM
class RecordingLLM:
    """The LLM client, observed: chat() passes through unchanged; its usage and time go to usage.jsonl."""

    def __init__(self, inner, writer, clock, t0):
        self._inner, self._writer, self._clock, self._t0 = inner, writer, clock, t0
        self.n_calls = 0
        self.leaks = []

    def _write(self, entry):
        try:
            self._writer.jsonl("usage.jsonl", entry)
        except SecretLeak as leak:
            self.leaks.append(str(leak))

    async def chat(self, messages, *args, **kwargs):
        self.n_calls += 1
        n, started = self.n_calls, self._clock()
        # The tools the harness offered on this call: the core tools plus the deferred groups loaded so far.
        tools = kwargs.get("tools")
        if tools is not None and not isinstance(tools, (list, tuple)):
            tools = kwargs["tools"] = list(tools)          # an iterator would be spent by reading it here
        offered = sorted(str(getattr(t, "name", t)) for t in tools) if tools is not None else None
        try:
            resp = await self._inner.chat(messages, *args, **kwargs)
        except Exception as exc:
            self._write({"call": n, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                         "seconds": round(self._clock() - started, 3), "t": round(started - self._t0, 3),
                         "error": type(exc).__name__, "tools_offered": offered})
            raise
        usage = getattr(resp, "usage", None) or {}
        self._write({"call": n, "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
                     "seconds": round(self._clock() - started, 3), "t": round(started - self._t0, 3),
                     "finish_reason": getattr(resp, "finish_reason", None), "usage_reported": bool(usage),
                     "tools_offered": offered})
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SettingsTracer:
    """The LLM client's tracer hook: keeps the sampling settings of each request (never its messages or tools)."""

    def __init__(self):
        self.settings = []

    def on_request(self, payload):
        s = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
        if s not in self.settings:
            self.settings.append(s)

    def on_response(self, payload):
        pass


# --------------------------------------------------------------------------------------------- the agent, as built
def build_llm(tracer):
    from olmoearth_agent.llm import OlmoEarthLLM
    try:
        return OlmoEarthLLM(tracer=tracer)
    except TypeError:                                    # a branch without the tracer hook
        return OlmoEarthLLM()


def build_studio():
    from olmoearth_agent.studio import StudioClient
    return StudioClient.from_env()


def build_agent(llm, studio, registry=None, skill_index=None):
    """LeadAgent as olmoearth_agent.cli.run_brief builds it (check_agent keeps the two equal)."""
    from olmoearth_agent.harness import LeadAgent
    from olmoearth_agent.harness.memory import preferences_block
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.skills import SkillLoader, build_default_registry
    registry = registry or build_default_registry()
    if skill_index is None:
        skill_index = SkillLoader().index()
    return LeadAgent(llm, registry, studio, state=ThreadState(), skill_index=skill_index,
                     memory_block=preferences_block(), local=True)


def construction_problems(run_brief_source):
    """How the command line's run_brief (its source) builds the agent unlike build_agent; [] if it does not."""
    tree = ast.parse(textwrap.dedent(run_brief_source))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == "LeadAgent" or getattr(n.func, "attr", None) == "LeadAgent")]
    problems = []
    if len(calls) != 1:
        problems.append(f"run_brief constructs LeadAgent {len(calls)} times")
    else:
        args = [ast.unparse(a) for a in calls[0].args]
        keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        if args != CLI_LEADAGENT_ARGS or keywords != CLI_LEADAGENT_KEYWORDS:
            problems.append(f"run_brief builds LeadAgent({args}, {keywords}); the driver builds "
                            f"LeadAgent({CLI_LEADAGENT_ARGS}, {CLI_LEADAGENT_KEYWORDS})")
    src = ast.unparse(tree)
    for needed in ("build_default_registry()", "SkillLoader().index()"):
        if needed not in src:
            problems.append(f"run_brief no longer calls {needed}")
    return problems


#: The run_stream events the driver reads, and the keys it reads from each.
EVENT_KEYS = {"tool_call": {"id", "name", "arguments", "turn"}, "tool_result": {"id", "name", "ok", "result", "turn"},
              "final": {"content", "turn"}}
#: The Studio methods whose responses the scorer indexes, with the argument names the log keys them by.
STUDIO_METHODS = {"pixel_value": ["result_id", "lon", "lat"], "get_prediction_result": ["result_id"],
                  "get_prediction": ["prediction_id"], "get_model": ["model_id"]}


def event_shapes(run_stream_source):
    """Event type -> the keys of the event dicts run_stream yields, read from its source."""
    shapes = {}
    for node in ast.walk(ast.parse(textwrap.dedent(run_stream_source))):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            kind = next((v.value for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)
                         and k.value == "type" and isinstance(v, ast.Constant)), None)
            if kind:
                shapes.setdefault(kind, set()).update(keys)
    return shapes


def interface_problems():
    """Everything else the driver reads from the agent, checked on the installed branch; [] when all of it is there."""
    from olmoearth_agent import cli
    from olmoearth_agent.harness import LeadAgent
    from olmoearth_agent.harness import memory
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.llm.client import OlmoEarthLLM
    from olmoearth_agent.llm.types import ChatResponse
    from olmoearth_agent.provenance.log import ProvenanceLog
    from olmoearth_agent.security import paths
    from olmoearth_agent.studio.client import StudioClient
    from olmoearth_agent.tools import review_set
    problems = []
    if inspect.signature(cli.run_brief).parameters.get("max_turns").default != MAX_TURNS:
        problems.append("run_brief's default max_turns is not 8")
    if cli._parse_args(["brief"]).max_turns != MAX_TURNS:
        problems.append("the command line's default --max-turns is not 8")
    if "max_turns" not in inspect.signature(LeadAgent.run_stream).parameters:
        problems.append("LeadAgent.run_stream takes no max_turns")
    shapes = event_shapes(inspect.getsource(LeadAgent.run_stream))
    for kind, keys in EVENT_KEYS.items():
        if not keys <= shapes.get(kind, set()):
            problems.append(f"run_stream's {kind} event lacks {sorted(keys - shapes.get(kind, set()))}")
    if "tracer" not in inspect.signature(OlmoEarthLLM.__init__).parameters:
        problems.append("OlmoEarthLLM takes no tracer")
    if not {"tools", "mode", "preserve_thinking"} <= set(inspect.signature(OlmoEarthLLM.chat).parameters):
        problems.append("OlmoEarthLLM.chat lacks tools, mode or preserve_thinking")
    if not {"usage", "tool_calls", "thinking", "finish_reason"} <= set(ChatResponse.__dataclass_fields__):
        problems.append("ChatResponse lacks usage, tool_calls, thinking or finish_reason")
    for name, params in STUDIO_METHODS.items():
        fn = getattr(StudioClient, name, None)
        if not inspect.iscoroutinefunction(fn) or list(inspect.signature(fn).parameters)[1:1 + len(params)] != params:
            problems.append(f"StudioClient.{name}({', '.join(params)}) is not there")
    if review_set.SCORES_ROOT_ENV != SCORES_ROOT_ENV or paths.OUTPUT_ROOT_ENV != OUTPUT_ROOT_ENV:
        problems.append("the scores root or workspace root is set by another environment variable")
    if not callable(getattr(memory, "preferences_block", None)):
        problems.append("harness.memory.preferences_block is not there")
    if not {"provenance", "turn_count"} <= set(ThreadState.__dataclass_fields__):
        problems.append("ThreadState lacks provenance or turn_count")
    if not callable(getattr(ProvenanceLog, "to_dict", None)):
        problems.append("ProvenanceLog.to_dict is not there")
    return problems


def check_agent(registry=None):
    """Refuse to run if the agent no longer builds or reports as this driver reads it.

    Deferred tool groups (a skill's tools sent only once olmoearth_load_skill loads it) need nothing from the driver:
    the harness chooses the tools of each turn, and dispatch loads a deferred tool's group when it is called. The driver
    only records what each model call was offered (usage.jsonl) and the groups loaded (run.json); it checks that
    olmoearth_load_skill is registered whenever the registry defers a group."""
    from olmoearth_agent import cli
    problems = construction_problems(inspect.getsource(cli.run_brief)) + interface_problems()
    groups = registry.groups() if hasattr(registry, "groups") else {}
    if groups and "olmoearth_load_skill" not in registry.names():
        problems.append("the registry defers tool groups but has no olmoearth_load_skill to load them")
    if problems:
        raise DriverError("the agent differs from what this driver reads: " + "; ".join(problems)
                          + ". Update the driver before running.")



def sampling_settings(llm):
    """The sampling the harness requests: the defaults of OlmoEarthLLM.chat (LeadAgent passes none of its own)."""
    from olmoearth_agent.llm.client import OlmoEarthLLM
    from olmoearth_agent.llm.presets import PRESETS
    params = inspect.signature(OlmoEarthLLM.chat).parameters
    mode = params["mode"].default
    cfg = getattr(llm, "config", None)
    return {"mode": mode, **PRESETS[mode], "max_tokens": getattr(cfg, "max_output_tokens", None),
            "preserve_thinking": params["preserve_thinking"].default}


def _git(cwd, *args):
    try:
        r = subprocess.run(["git", "-C", cwd, *args], capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def git_state(path, untracked_under=("src",)):
    """The checkout holding `path`: commit, branch, and whether (and how) the tree differs from the commit."""
    top = _git(path, "rev-parse", "--show-toplevel")
    if top is None:
        return {"repo": None, "commit": None, "branch": None, "dirty": None, "changed": [], "diff_sha256": None}
    top = top.decode().strip()
    commit = (_git(top, "rev-parse", "HEAD") or b"").decode().strip() or None
    branch = (_git(top, "rev-parse", "--abbrev-ref", "HEAD") or b"").decode().strip() or None
    tracked = [ln for ln in (_git(top, "status", "--porcelain", "--untracked-files=no") or b"").decode().splitlines()
               if ln.strip()]
    untracked = [ln for ln in (_git(top, "ls-files", "--others", "--exclude-standard", "--", *untracked_under)
                               or b"").decode().splitlines() if ln.strip()]
    digest = None
    if tracked or untracked:
        h = hashlib.sha256(_git(top, "diff", "HEAD", "--binary") or b"")
        for rel in untracked:
            h.update(rel.encode())
            with contextlib.suppress(OSError), open(os.path.join(top, rel), "rb") as fh:
                h.update(fh.read())
        digest = h.hexdigest()
    submodules = [ln.strip() for ln in (_git(top, "submodule", "status") or b"").decode().splitlines() if ln.strip()]
    return {"repo": os.path.basename(top), "commit": commit, "branch": branch, "dirty": bool(tracked or untracked),
            "changed": [ln[3:] for ln in tracked] + untracked, "diff_sha256": digest, "submodules": submodules,
            "_top": top}


def agent_source():
    """The agent checkout the installed olmoearth_agent is imported from, and its version."""
    import olmoearth_agent
    info = git_state(os.path.dirname(os.path.abspath(olmoearth_agent.__file__)))
    try:
        info["version"] = importlib.metadata.version("olmoearth-agent")
    except importlib.metadata.PackageNotFoundError:
        info["version"] = None
    return info


def inferencex_info():
    """The olmoearth-inferencex the agent's tools import: its installed version, and whether it is this repository's."""
    try:
        version = importlib.metadata.version("olmoearth-inferencex")
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec("oe_inferencex")
    origin = os.path.realpath(spec.origin) if spec and spec.origin else None
    inside = bool(origin) and origin.startswith(os.path.realpath(REPO) + os.sep)
    return {"version": version, "imported_from": None if origin is None else "this repository" if inside
            else "the installed distribution", "inside_this_repository": inside}


def _public(info):
    return {k: v for k, v in info.items() if not k.startswith("_")}


# --------------------------------------------------------------------------------------------- the round and fixtures
def _under_rounds(out):
    return os.path.basename(os.path.dirname(os.path.abspath(out))) == "rounds"


def trial_of(out):
    """<trial> for a round directory <trial>/rounds/<r>, else None."""
    return os.path.dirname(os.path.dirname(os.path.abspath(out))) if _under_rounds(out) else None


def load_round(out):
    p = os.path.join(out, "round.json")
    meta = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            meta = json.load(fh)
    meta.setdefault("brief_values", {})
    meta.setdefault("fixes", [])
    meta.setdefault("not_run", {})
    return meta


def pin_round(meta, facts):
    """Fill the round's facts on its first run; afterwards every run must match them. Returns the conflicts."""
    conflicts = []
    for k, v in facts.items():
        if k in meta and meta[k] != v:
            conflicts.append(f"{k}: the round has {meta[k]!r}, this run {v!r}")
        else:
            meta[k] = v
    return conflicts


def verified_fixtures(trial):
    """Relative path -> sha256 of every file in <trial>/fixtures, each checked against trial.json's "fixtures"."""
    d = os.path.join(trial, "fixtures") if trial else None
    if not d or not os.path.isdir(d):
        return {}
    have = {os.path.relpath(p, d): sha256_file(p) for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if os.path.isfile(p)}
    meta_path = os.path.join(trial, "trial.json")
    listed = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            listed = (json.load(fh) or {}).get("fixtures") or {}
    expected = {k: v for k, v in listed.items() if isinstance(v, str)}
    problems = [f"{rel} is not listed" for rel in sorted(set(have) - set(expected))]
    problems += [f"{rel} is listed but missing" for rel in sorted(set(expected) - set(have))]
    problems += [f"{rel} differs from its sha256" for rel in sorted(set(have) & set(expected))
                 if have[rel] != expected[rel]]
    if problems:
        raise DriverError("the fixtures do not match trial.json's \"fixtures\" hashes: " + "; ".join(problems))
    return have


def provider_fixtures(trial, fixtures):
    """The cluster provider's model runs among the fixtures: run directory (relative to fixtures/) -> its record.

    A provider run is a directory whose manifest.json names a scores raster, as scripts/score_area.py writes it. The
    fixture holding it (its top directory under <trial>/fixtures) must hold nothing but its runs' manifests and rasters,
    and each raster must be the one its manifest records: anything else (an `oe-inferencex assess` output beside it,
    say, which is this package's own review set of that raster) would put a reference answer in the agent's workspace.
    Every file's sha256 is also checked against trial.json with the other fixtures (verified_fixtures)."""
    d = os.path.join(trial, "fixtures") if trial else None
    runs, problems = {}, []
    for rel in sorted(fixtures):
        if os.path.basename(rel) != PROVIDER_MANIFEST:
            continue
        try:
            with open(os.path.join(d, rel), encoding="utf-8") as fh:
                scores = (json.load(fh) or {}).get("scores") or {}
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        raster = scores.get("file") if isinstance(scores, dict) else None
        if not isinstance(raster, str):
            continue
        run_dir = os.path.dirname(rel)
        raster_rel = os.path.join(run_dir, raster)
        if raster_rel not in fixtures:
            problems.append(f"{run_dir or '.'}: its manifest names {raster}, which is not there")
        elif fixtures[raster_rel] != scores.get("sha256"):
            problems.append(f"{raster_rel}'s sha256 is not the one its manifest records")
        runs[run_dir] = {"fixture": rel.split(os.sep)[0] if run_dir else "", "raster": raster,
                         "sha256": fixtures.get(raster_rel)}
    allowed = {os.path.join(r, f) if r else f for r, v in runs.items() for f in (PROVIDER_MANIFEST, v["raster"])}
    for top in sorted({v["fixture"] for v in runs.values()}):
        extra = [rel for rel in fixtures if (rel.split(os.sep)[0] == top or not top) and rel not in allowed]
        if extra:
            problems.append(f"the provider fixture {top or 'fixtures/'} also holds {extra[:5]}; it may hold only its "
                            f"model runs' {PROVIDER_MANIFEST} and raster")
    if problems:
        raise DriverError("the cluster provider's fixtures are not usable: " + "; ".join(problems))
    return runs


def workspace_fixtures(fixtures, providers, config, consts=None):
    """The fixtures a run's workspace receives, as the scorer's WORKSPACE_FIXTURES lists them by top directory; with
    no such table, every fixture for a cluster run and every fixture but the provider's for the others."""
    table = getattr(consts, "workspace", None)
    if table and config in table:
        keep = set(table[config])
        return {rel: h for rel, h in fixtures.items() if rel.split(os.sep)[0] in keep}
    if config.endswith("/cluster"):
        return dict(fixtures)
    tops = {v["fixture"] for v in providers.values()}
    return {rel: h for rel, h in fixtures.items() if rel.split(os.sep)[0] not in tops}


@contextlib.contextmanager
def run_environment(workspace):
    """The run's workspace as the agent's scores root, workspace root (so preference memory starts empty and spills
    stay with the run) and working directory (so a file the brief names resolves there); restored afterwards."""
    saved = {k: os.environ.get(k) for k in (SCORES_ROOT_ENV, OUTPUT_ROOT_ENV)}
    cwd = os.getcwd()
    os.environ[SCORES_ROOT_ENV] = workspace
    os.environ[OUTPUT_ROOT_ENV] = workspace
    os.chdir(workspace)
    try:
        yield
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------------------------- one run
@dataclasses.dataclass
class Plan:
    config: str
    run: str
    brief: str
    out: str
    run_dir: str
    workspace: str
    writer: RunWriter
    llm: object
    studio: object
    registry: object
    tracer: SettingsTracer
    info: dict
    inferencex: dict
    sampling: dict
    model: str
    fixtures: dict
    providers: dict
    notes: list
    handle_sigterm: bool
    check_end_state: bool


def _endpoint_unreachable(exc):
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.APIConnectionError)      # APITimeoutError is a subclass


def _specs_digest(registry):
    try:
        specs = [{"name": s.name, "description": s.description, "parameters": s.parameters} for s in registry.specs()]
    except Exception:  # noqa: BLE001  a registry without specs(): not recorded
        return None
    return hashlib.sha256(json.dumps(specs, sort_keys=True, default=str).encode()).hexdigest()


async def _execute(p):
    """Consume run_stream, writing each event as it arrives, then the run's other files."""
    from olmoearth_agent.harness.memory import preferences_block
    clock = time.monotonic
    t0, started = clock(), _now()
    slog = StudioLog(p.writer, clock, t0)
    llm = RecordingLLM(p.llm, p.writer, clock, t0)
    studio = RecordingStudio(p.studio, slog)
    st = {"final": None, "turns": None, "hit": False, "status": None, "void": None, "error": None, "leaks": []}
    calls, changed, dups, seen, pending = [], [], [], set(), {}
    agent, stop, reraise = None, {}, False
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    if p.handle_sigterm:
        def on_term():
            stop["signal"] = "SIGTERM"
            task.cancel()
        loop.add_signal_handler(signal.SIGTERM, on_term)
    memory_chars = None
    with run_environment(p.workspace):
        gen = None
        try:
            memory_chars = len(preferences_block())
            if memory_chars:
                raise DriverError("the fresh workspace holds a preference memory")
            agent = build_agent(llm, studio, registry=p.registry)
            gen = agent.run_stream(p.brief, max_turns=MAX_TURNS)
            last = t0
            # seq: the event's line in events.jsonl, from 0. A tool_result also carries call_seq, the seq of its
            # tool_call, and so does every Studio response the call received: (call_seq, id) is a call's key, since
            # an id alone can repeat across turns.
            seq = -1
            async for ev in gen:
                seq += 1
                now = clock()
                rec = {**ev, "seq": seq, "t": round(now - t0, 3), "seconds": round(now - last, 3)}
                last = now
                kind = ev.get("type")
                if kind == "tool_call":
                    cid = ev.get("id")
                    if cid in seen:
                        dups.append(cid)
                    seen.add(cid)
                    slog.call_seq, slog.call_id, slog.call_turn = seq, cid, ev.get("turn")
                    pending[cid] = (ev.get("arguments"), json.dumps(ev.get("arguments"), sort_keys=True, default=str),
                                    seq)
                elif kind == "tool_result":
                    slog.call_seq = slog.call_id = slog.call_turn = None
                    calls.append((ev.get("name"), bool(ev.get("ok"))))
                    args, snap, call_seq = pending.pop(ev.get("id"), (None, None, None))
                    rec["call_seq"] = call_seq
                    if snap is not None and json.dumps(args, sort_keys=True, default=str) != snap:
                        changed.append(ev.get("id"))
                elif kind == "final":
                    st["final"], st["turns"] = ev.get("content"), ev.get("turn")
                elif kind == "max_turns":
                    st["hit"], st["turns"] = True, ev.get("turns")
                slog.redact.learn(rec)
                p.writer.jsonl("events.jsonl", slog.redact(rec))
                if slog.leaks or llm.leaks:
                    raise SecretLeak("; ".join(slog.leaks + llm.leaks))
            st["status"] = "answered" if st["final"] is not None else "no_answer"
        except SecretLeak as exc:
            st["status"], st["error"] = "error", f"secret leak blocked: {exc}"
            st["leaks"].append(str(exc))
        except asyncio.CancelledError:
            if hasattr(task, "uncancel"):
                task.uncancel()
            if stop:
                st["status"], st["void"] = "void", f"the job ended mid-run ({stop['signal']})"
            else:
                st["status"], st["error"], reraise = "error", "interrupted by the operator", True
        except Exception as exc:  # noqa: BLE001  recorded, never raised past the run's files
            if _endpoint_unreachable(exc):
                st["status"], st["void"] = "void", f"the model endpoint was unreachable ({type(exc).__name__})"
            else:
                st["status"], st["error"] = "error", f"{type(exc).__name__}: {redact(exc, p.writer.secrets)}"
        finally:
            if gen is not None:
                with contextlib.suppress(BaseException):
                    await gen.aclose()
    if p.handle_sigterm:
        loop.remove_signal_handler(signal.SIGTERM)
    for client in (p.llm, p.studio):                       # the driver owns both, as run_brief owns its Studio client
        closer = getattr(client, "aclose", None)
        if closer is not None and inspect.iscoroutinefunction(closer):
            with contextlib.suppress(Exception):
                await closer()
    seconds = round(clock() - t0, 3)

    def attempt(fn, *a):
        try:
            fn(*a)
        except SecretLeak as exc:
            st["leaks"].append(str(exc))

    final = st["final"]
    attempt(p.writer.text, "stdout.txt", (slog.redact.text(final) + "\n") if isinstance(final, str) else "")
    lines = [f"  [{'ok' if ok else 'FAIL'}] {name}" for name, ok in calls]
    if agent is not None:
        turns = st["turns"] if st["turns"] is not None else agent.state.turn_count
        lines.append(f"  ({turns} turn(s), {len(agent.state.provenance.entries)} provenance entries)")
        attempt(p.writer.json, "provenance.json", agent.state.provenance.to_dict())
    if final is None:
        lines.append("(no answer, hit the turn cap)" if st["status"] == "no_answer"
                     else f"(no answer: the run ended as {st['status']}; see run.json)")
    attempt(p.writer.text, "stderr.txt", "\n".join(lines) + "\n")
    for rel in ("events.jsonl", "usage.jsonl", "studio_calls.jsonl"):
        if not os.path.exists(p.writer.path(rel)):
            attempt(p.writer.text, rel, "")
    end_info = agent_source() if p.check_end_state else p.info
    swept = [f"{rel}: {', '.join(names)}" for rel, names in p.writer.sweep()]
    leaks = st["leaks"] + [f"{s} (the file is in the run directory)" for s in swept]
    if leaks and st["status"] != "void":
        st["status"] = "error"
        st["error"] = st["error"] or "a secret's value was found in the run directory"
    # run.json's exit code is the agent command line's (0 when it prints an answer, else 1); the driver's own exit
    # code also tells a void run and a secret found apart.
    cli_exit = EXIT_ANSWERED if final is not None else EXIT_NO_ANSWER
    exit_code = {"answered": EXIT_ANSWERED, "no_answer": EXIT_NO_ANSWER, "void": EXIT_VOID}.get(st["status"],
                                                                                               EXIT_NO_ANSWER)
    if leaks:
        exit_code = EXIT_REFUSED
    usage = []
    if os.path.exists(p.writer.path("usage.jsonl")):
        with open(p.writer.path("usage.jsonl"), encoding="utf-8") as fh:
            usage = [json.loads(ln) for ln in fh if ln.strip()]
    brief_id, provider = p.config.split("/")
    candidates = [f"Studio answered {n} call(s) with {k}" for k, n in sorted(slog.errors.items())
                  if k.isdigit() and (int(k) >= 500 or int(k) == 429) or not k.isdigit()]
    meta = {
        "experiment": EXPERIMENT, "configuration": p.config, "brief_id": brief_id, "provider": provider,
        "run": p.run, "brief": p.brief, "started": started, "ended": _now(), "seconds": seconds,
        "exit_code": cli_exit, "driver_exit_code": exit_code, "status": st["status"],
        "model": p.model, "max_turns": MAX_TURNS, "turns": st["turns"], "hit_max_turns": st["hit"],
        "n_tool_calls": len(calls), "n_llm_calls": llm.n_calls,
        "tokens": {k: sum(int(u.get(k) or 0) for u in usage)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "agent": {**_public(p.info), "state_at_end": {"commit": end_info.get("commit"),
                                                     "diff_sha256": end_info.get("diff_sha256")},
                  "changed_during_run": (end_info.get("commit"), end_info.get("diff_sha256"))
                  != (p.info.get("commit"), p.info.get("diff_sha256"))},
        "inferencex": p.inferencex, "sampling": p.sampling, "sampling_sent": p.tracer.settings,
        "system_prompt_sha256": (hashlib.sha256(agent.system_prompt.encode()).hexdigest()
                                 if agent is not None else None),
        "tools_registered": list(p.registry.names()) if hasattr(p.registry, "names") else None,
        "tool_specs_sha256": _specs_digest(p.registry),
        "deferred_groups": p.registry.groups() if hasattr(p.registry, "groups") else None,
        "groups_loaded": (sorted(getattr(agent.state, "loaded_groups", None) or [])
                          if agent is not None and hasattr(agent.state, "loaded_groups") else None),
        "workspace": {"scores_root": "workspace/", "output_root": "workspace/", "working_directory": "workspace/",
                      "preference_memory_chars": memory_chars, "fixtures": p.fixtures,
                      "provider_runs": p.providers},
        "environment": {k: os.environ[k] for k in RECORDED_ENV if k in os.environ and k not in p.writer.secrets},
        "studio": {"n_calls": slog.n, "errors": dict(slog.errors)},
        "secret_check": {"variables": sorted(p.writer.secrets), "files_checked": p.writer.n_checked,
                         "leaks": leaks},
        "arguments_changed_by_dispatch": changed, "duplicate_call_ids": sorted(set(dups)),
        "driver": {"script": "exp/exp86_driver.py", **{k: v for k, v in _public(git_state(REPO)).items()
                                                        if k in ("commit", "dirty", "diff_sha256")}},
        "python": sys.version.split()[0],
    }
    if st["void"]:
        meta["void"] = st["void"]
    if st["error"]:
        meta["error"] = st["error"]
    if candidates:
        meta["void_candidates"] = candidates
    if p.notes:
        meta["notes"] = p.notes
    try:
        p.writer.json("run.json", meta, atomic=True)
    except SecretLeak as exc:                              # names only: a secret here means a secret in the brief
        p.writer.json("run.json", {"configuration": p.config, "run": p.run, "status": "error",
                                   "error": f"secret leak blocked: {exc}"}, atomic=True)
        exit_code = EXIT_REFUSED
    if reraise:
        raise asyncio.CancelledError
    return {"exit_code": exit_code, "status": st["status"], "run_dir": p.run_dir, "seconds": seconds,
            "tools": [n for n, _ in calls], "leaks": leaks}


def run_one(out, brief_id, provider, run, *, trial=None, llm=None, studio=None, registry=None, agent_info=None,
            allow_dirty=False, handle_sigterm=False, require_installed_inferencex=True, environ=None):
    """Run one configuration once and write its run directory. Raises DriverError, writing nothing, on a refusal.

    llm, studio, registry and agent_info are injected by the tests; left None, they are built as the agent's command
    line builds them, from the environment."""
    consts = scorer_constants()
    config = f"{brief_id}/{provider}"
    if config not in consts.briefs:
        raise DriverError(f"{config} is not a preregistered configuration: {sorted(consts.briefs)}")
    run = str(run)
    if not re.fullmatch(r"[1-9][0-9]*", run):
        raise DriverError(f"--run must be a positive integer, got {run!r}")
    out = os.path.abspath(out)
    counted = _under_rounds(out)
    if allow_dirty and counted:
        raise DriverError("--allow-dirty is for debugging runs, which the plan keeps outside rounds/")
    trial = os.path.abspath(trial) if trial else trial_of(out)
    meta = load_round(out)
    values = (meta["brief_values"] or {}).get(config) or {}
    template = consts.briefs[config]
    brief = fill_brief(template, values)
    env = os.environ if environ is None else environ

    info = agent_info if agent_info is not None else agent_source()
    if info.get("commit") is None and counted:
        raise DriverError("the agent is not a git checkout, so its commit cannot be recorded")
    if info.get("dirty") and not allow_dirty:
        raise DriverError(f"the agent's tree differs from its commit {info.get('commit')} ({info.get('changed')}); "
                          "commit it, or pass --allow-dirty for a debugging run outside rounds/")
    overriding = [k for k in OVERRIDING_ENV if env.get(k)]
    if overriding and not allow_dirty:
        raise DriverError(f"{overriding} replace the agent's committed soul or skills; unset them")
    if registry is None:
        from olmoearth_agent.skills import build_default_registry
        registry = build_default_registry()
    check_agent(registry)
    ix = inferencex_info()
    if require_installed_inferencex and ix["inside_this_repository"]:
        raise DriverError("the agent would import this repository's oe_inferencex, not its installed extra; run with "
                          "the agent's interpreter from outside the repository's import path")

    fixtures = verified_fixtures(trial)
    providers = provider_fixtures(trial, fixtures)
    ws_fixtures = workspace_fixtures(fixtures, providers, config, consts)
    ws_tops = {rel.split(os.sep)[0] for rel in ws_fixtures}
    ws_runs = {r: v for r, v in providers.items() if v["fixture"] in ws_tops}
    notes = []
    if provider == "cluster":
        if PROVIDER_TOOL not in registry.names():
            raise DriverError(f"the agent registers no {PROVIDER_TOOL}, the cluster scores provider")
        listed = (consts.workspace or {}).get(config) or ()
        absent_ids = [f for f in listed if not any(v["fixture"] == f for v in providers.values())]
        if absent_ids or not ws_runs:
            raise DriverError(f"{config} needs the cluster provider's fixture(s) {list(listed) or '(any)'}, a model "
                              f"run's {PROVIDER_MANIFEST} and scores raster hashed in trial.json; missing: "
                              f"{absent_ids or 'every one'}")
        if PROVIDER_TOOL not in consts.provider:
            if counted:
                raise DriverError(f"the scorer's PROVIDER is {consts.provider}, not {PROVIDER_TOOL}: a counted cluster "
                                  "run would be graded against a tool the agent does not have")
            notes.append(f"the scorer's PROVIDER was {list(consts.provider)}, not {PROVIDER_TOOL}, when this run was "
                         "made: its routing grade (P1) does not count the provider's calls")
    elif ws_runs:                               # "Setup": only a cluster run has the provider
        raise DriverError(f"{config} would receive the cluster provider's model runs {sorted(ws_runs)}")
    named = brief_files(template, values)
    absent = {k: v for k, v in named.items() if os.path.normpath(v) not in ws_fixtures}
    if absent:
        raise DriverError(f"the brief names files that are not fixtures this run receives: {absent}")
    wrong_runs = {k: v for k, v in brief_run_dirs(template, values).items() if os.path.normpath(v) not in ws_runs}
    if wrong_runs:
        raise DriverError(f"the brief names model runs that are not provider fixtures this run receives: {wrong_runs} "
                          f"(it receives {sorted(ws_runs)})")
    secrets = environment_secrets(environ)
    tracer = SettingsTracer()
    if llm is None:
        llm = build_llm(tracer)
    if studio is None:
        try:
            studio = build_studio()
        except RuntimeError as exc:
            raise DriverError(redact(exc, secrets)) from None
    model = getattr(getattr(llm, "config", None), "model", None)
    sampling = sampling_settings(llm)
    facts = {"agent_commit": info.get("commit"), "agent_dirty": bool(info.get("dirty")),
             "agent_diff_sha256": info.get("diff_sha256"), "inferencex_version": ix["version"], "model": model,
             "sampling": sampling, "max_turns": MAX_TURNS}
    conflicts = pin_round(meta, facts)
    if conflicts:
        raise DriverError("this run cannot join the round (a fix, or another model, starts a new round): "
                          + "; ".join(conflicts))
    run_dir = os.path.join(out, "runs", brief_id, provider, run)
    if os.path.exists(run_dir):
        raise DriverError(f"{run_dir} exists; runs are never overwritten (a void run is replaced by a new number)")

    meta.setdefault("agent_branch", info.get("branch"))
    meta.setdefault("first_run", _now())
    RunWriter(out, secrets).json("round.json", meta, atomic=True)
    writer = RunWriter(run_dir, secrets)
    workspace = writer.path("workspace")
    os.makedirs(workspace)
    try:
        for rel in sorted(ws_fixtures):
            writer.copy(os.path.join(trial, "fixtures", rel), os.path.join("workspace", rel))
        writer.text("brief.txt", brief)
        if info.get("dirty"):
            top = info.get("_top")
            writer.bytes("agent_dirty.diff", (_git(top, "diff", "HEAD", "--binary") or b"") if top else b"")
        writer.json("run.json", {"experiment": EXPERIMENT, "configuration": config, "run": run, "brief": brief,
                                 "status": "running", "started": _now(), "model": model}, atomic=True)
    except SecretLeak as exc:
        writer.json("run.json", {"configuration": config, "run": run, "status": "error",
                                 "error": f"secret leak blocked: {exc}"}, atomic=True)
        return {"exit_code": EXIT_REFUSED, "status": "error", "run_dir": run_dir, "seconds": 0.0, "tools": [],
                "leaks": [str(exc)]}
    plan = Plan(config=config, run=run, brief=brief, out=out, run_dir=run_dir, workspace=workspace, writer=writer,
                llm=llm, studio=studio, registry=registry, tracer=tracer, info=info, inferencex=ix,
                sampling=sampling, model=model, fixtures=ws_fixtures,
                providers=ws_runs, notes=notes, handle_sigterm=handle_sigterm,
                check_end_state=agent_info is None)
    return asyncio.run(_execute(plan))


# --------------------------------------------------------------------------------------------- criterion 7a
class _NoStudio:
    """The parity calls read fixture files only; a Studio call means a tool was not given its file."""

    config = None

    def __getattr__(self, name):
        async def refuse(*_a, **_k):
            raise RuntimeError(f"the fixed-input parity calls make no Studio call (studio.{name})")
        return refuse


def parity_calls(f1, values):
    """The plan's fixed-input calls (criterion 7a): the tools, their arguments, and the file each is written to."""
    b5, b6, b7 = values.get("B5/files") or {}, values.get("B6/files") or {}, values.get("B7/files") or {}
    need = {"B5/files": ("design_path", "labels_path"), "B6/files": ("design_path", "labels_path"),
            "B7/files": ("scores_a", "scores_b", "date_a", "date_b")}
    missing = [f"{c}.{k}" for c, keys in need.items() for k in keys if not str((values.get(c) or {}).get(k, ""))]
    if missing:
        raise DriverError(f"round.json's brief_values lack {missing}, which name F2 to F4 for the parity calls")
    return [
        ("01_review_set", "olmoearth_review_set", {"scores_path": f1, "budget": 0.05}),
        ("02_plan_label_sample", "olmoearth_plan_label_sample",
         {"scores_path": f1, "budget": 300, "design": "confidence", "seed": 0}),
        ("03_estimate_map_error", "olmoearth_estimate_map_error",
         {"design_path": b5["design_path"], "labels_path": b5["labels_path"]}),
        ("04_certify_zone_alpha0.05", "olmoearth_certify_zone",
         {"design_path": b6["design_path"], "labels_path": b6["labels_path"], "alpha": 0.05}),
        ("05_certify_zone_alpha0.25", "olmoearth_certify_zone",
         {"design_path": b6["design_path"], "labels_path": b6["labels_path"], "alpha": 0.25}),
        ("06_compare_review", "olmoearth_compare_review",
         {"scores_path_a": b7["scores_a"], "scores_path_b": b7["scores_b"], "date_a": b7["date_a"],
          "date_b": b7["date_b"]}),
    ]


def run_parity(out, f1, *, trial=None, registry=None, agent_info=None, allow_dirty=False,
               require_installed_inferencex=True, environ=None):
    """Criterion 7a: the agent's handlers called directly on the fixtures, one {tool, arguments, result} file each."""
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.llm.types import ToolCall
    from olmoearth_agent.tools.registry import ToolContext
    out = os.path.abspath(out)
    if allow_dirty and _under_rounds(out):
        raise DriverError("--allow-dirty is for debugging runs, which the plan keeps outside rounds/")
    trial = os.path.abspath(trial) if trial else trial_of(out)
    meta = load_round(out)
    calls = parity_calls(f1, meta["brief_values"] or {})
    info = agent_info if agent_info is not None else agent_source()
    if info.get("dirty") and not allow_dirty:
        raise DriverError(f"the agent's tree differs from its commit {info.get('commit')}; commit it first")
    ix = inferencex_info()
    if require_installed_inferencex and ix["inside_this_repository"]:
        raise DriverError("the agent would import this repository's oe_inferencex, not its installed extra")
    fixtures = verified_fixtures(trial)
    fixtures = workspace_fixtures(fixtures, provider_fixtures(trial, fixtures), "parity", scorer_constants())
    files = {v for _, _, args in calls for k, v in args.items() if k.endswith(("_path", "_path_a", "_path_b"))}
    absent = sorted(f for f in files if os.path.normpath(f) not in fixtures)
    if absent:
        raise DriverError(f"parity inputs that are not fixtures of this trial: {absent}")
    conflicts = pin_round(meta, {"agent_commit": info.get("commit"), "agent_dirty": bool(info.get("dirty")),
                                 "agent_diff_sha256": info.get("diff_sha256"), "inferencex_version": ix["version"]})
    if conflicts:
        raise DriverError("the parity calls cannot join the round: " + "; ".join(conflicts))
    pdir = os.path.join(out, "parity")
    if glob.glob(os.path.join(pdir, "*.json")):
        raise DriverError(f"{pdir} already holds the round's parity calls; they are made once per round")
    secrets = environment_secrets(environ)
    RunWriter(out, secrets).json("round.json", meta, atomic=True)
    writer = RunWriter(pdir, secrets)
    workspace = writer.path("workspace")
    os.makedirs(workspace, exist_ok=True)
    for rel in sorted(fixtures):
        writer.copy(os.path.join(trial, "fixtures", rel), os.path.join("workspace", rel))

    async def dispatch_all():
        from olmoearth_agent.skills import build_default_registry
        reg = registry or build_default_registry()
        ctx = ToolContext(studio=_NoStudio(), state=ThreadState())
        results = []
        for i, (stem, tool, args) in enumerate(calls, 1):
            env = await reg.dispatch(ToolCall(id=f"parity-{i}", name=tool, arguments=dict(args)), ctx)
            results.append((stem, tool, args, env))
        return results

    with run_environment(workspace):
        results = asyncio.run(dispatch_all())
    made = _now()
    for stem, tool, args, env in results:
        writer.json(f"{stem}.json", {"tool": tool, "arguments": args, "result": env, "made": made,
                                     "agent_commit": info.get("commit"), "inferencex_version": ix["version"]})
    return [(stem, bool(env.get("ok"))) for stem, _, _, env in results]


# --------------------------------------------------------------------------------------------- a whole round
def counted_runs(out, config):
    """The run directories the scorer counts for a configuration: every one whose run.json is not void."""
    b, p = config.split("/")
    n = 0
    for d in glob.glob(os.path.join(out, "runs", b, p, "*")):
        if not os.path.isdir(d):
            continue
        try:
            with open(os.path.join(d, "run.json"), encoding="utf-8") as fh:
                void = (json.load(fh) or {}).get("void")
        except (OSError, json.JSONDecodeError):
            void = None
        n += not void
    return n


def next_run_number(out, config):
    b, p = config.split("/")
    nums = [int(os.path.basename(d)) for d in glob.glob(os.path.join(out, "runs", b, p, "*"))
            if os.path.basename(d).isdigit()]
    return max(nums, default=0) + 1


def fill_round(out, *, trial=None, attempts=5):
    """Every required configuration to three counted runs, each run in its own process (runs share no state).

    A void run is kept, and the fill stops there (a rerun replaces it by the next number); a refused run stops that
    configuration. The conditional B8 studio control runs when round.json fills its brief and does not list it under
    not_run."""
    consts = scorer_constants()
    meta = load_round(out)
    configs = [c for c in consts.briefs if c not in consts.conditional]
    configs += [c for c in sorted(consts.conditional)
                if c in (meta.get("brief_values") or {}) and c not in (meta.get("not_run") or {})]
    report = {}
    for config in configs:
        b, p = config.split("/")
        tried = 0
        while counted_runs(out, config) < N_RUNS and tried < attempts:
            tried += 1
            cmd = [sys.executable, os.path.abspath(__file__), "--brief", b, "--provider", p,
                   "--run", str(next_run_number(out, config)), "--out", out] + (["--trial", trial] if trial else [])
            rc = subprocess.run(cmd, check=False).returncode
            if rc == EXIT_VOID:                  # the endpoint or the job is down: more runs now would be void too
                print(f"stopped: {config} run was void; fix the cause and rerun --all (void runs are kept and "
                      "replaced)", file=sys.stderr)
                return EXIT_VOID
            if rc == EXIT_REFUSED:
                break
        report[config] = counted_runs(out, config)
    for config, n in report.items():
        flag = "" if n == N_RUNS else "  (more than three: void the extras by hand)" if n > N_RUNS else "  incomplete"
        print(f"  {config:<12} {n} counted run(s){flag}")
    return 0 if all(n == N_RUNS for n in report.values()) else 1


# --------------------------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the round directory, <trial>/rounds/<r>")
    ap.add_argument("--trial", help="the trial directory (default: two levels above --out when it is under rounds/)")
    ap.add_argument("--brief", help="B1 to B8")
    ap.add_argument("--provider", choices=("studio", "cluster", "files"))
    ap.add_argument("--run", help="the run number, a positive integer; a void run is replaced by a new number")
    ap.add_argument("--all", action="store_true", help="bring every configuration to three counted runs")
    ap.add_argument("--attempts", type=int, default=5, help="with --all: runs started per configuration at most")
    ap.add_argument("--parity", action="store_true", help="make the round's fixed-input parity calls (criterion 7a)")
    ap.add_argument("--f1", help="with --parity: F1's scores file, relative to <trial>/fixtures")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run an agent tree with uncommitted changes (debugging runs outside rounds/ only)")
    a = ap.parse_args(argv)
    try:
        if a.all:
            return fill_round(a.out, trial=a.trial, attempts=a.attempts)
        if a.parity:
            if not a.f1:
                ap.error("--parity needs --f1")
            for stem, ok in run_parity(a.out, a.f1, trial=a.trial, allow_dirty=a.allow_dirty):
                print(f"  {'ok' if ok else 'FAIL'}  parity/{stem}.json")
            return 0
        if not (a.brief and a.provider and a.run):
            ap.error("one run needs --brief, --provider and --run (or use --all or --parity)")
        r = run_one(a.out, a.brief, a.provider, a.run, trial=a.trial, allow_dirty=a.allow_dirty,
                    handle_sigterm=True)
    except DriverError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{a.brief}/{a.provider} run {a.run}: {r['status']} (exit {r['exit_code']}) in {r['seconds']:.0f} s, "
          f"{len(r['tools'])} tool call(s) -> {os.path.relpath(r['run_dir'])}")
    for leak in r["leaks"]:
        print(f"  secret found: {leak}", file=sys.stderr)
    return r["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

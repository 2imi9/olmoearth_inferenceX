"""exp86's run driver (exp/exp86_driver.py): the agent's own harness driven with a scripted LLM and a stub Studio,
and the run directory it writes graded by the scorer (exp/exp86_agent_trial_v2.py). No network and no key.

Two kinds of test. The driver's own logic (the briefs read from the scorer's source, the secret check, the Studio
record sanitizer, the reading of the command line's construction) needs only the standard library and runs in any
environment. The runs need the olmoearth_agent package, which this repository does not install; those tests are
skipped without it, so run them with the agent's interpreter:
    ~/Desktop/Github/OlmoEarth-Agent/.venv/bin/python -m pytest tests/test_exp86_driver.py
The stubs stand in for the model and for Studio only. LeadAgent, run_stream, the tool registry, the tools, the
provenance log and the no-data rules are the agent's own code at whatever branch is checked out."""
import asyncio
import collections
import csv
import io
import json
import os
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp86_agent_trial_v2 as e86  # noqa: E402
import exp86_driver as drv  # noqa: E402

try:
    import olmoearth_agent  # noqa: F401
    HAVE_AGENT = True
except ImportError:
    HAVE_AGENT = False
needs_agent = pytest.mark.skipif(not HAVE_AGENT, reason="needs the olmoearth_agent package: run with the agent's "
                                                        "interpreter (~/Desktop/Github/OlmoEarth-Agent/.venv)")

KEY = "oe-test-studio-key-7f3a9c2e51"
AGENT = {"repo": "OlmoEarth-Agent", "commit": "0" * 40, "branch": "test", "dirty": False, "changed": [],
         "diff_sha256": None, "submodules": [], "version": "test"}
PROP = "sample_karst_score"
BBOX = (-77.8, 40.4, -77.4, 40.8)
GEOM = {"type": "Polygon", "coordinates": [[[BBOX[0], BBOX[1]], [BBOX[2], BBOX[1]], [BBOX[2], BBOX[3]],
                                            [BBOX[0], BBOX[3]], [BBOX[0], BBOX[1]]]]}
#: A 4 x 4 KarstBinary-like [0, 1] score with sixteen distinct margins |2s - 1|; cells 0 and 15 are Studio's no-data.
VALUES = [0.97, 0.99, 0.23, 0.31, 0.06, 0.02, 0.92, 0.89, 0.13, 0.81, 0.85, 0.07, 0.60, 0.12, 0.64, 0.04]
NODATA_CELLS = (0, 15)


# --------------------------------------------------------------------------------------------- stubs
class ScriptedLLM:
    """Stands in for OlmoEarthLLM: each chat() returns the next scripted step. A step is a list of (tool, arguments)
    calls, a final text, or a function of the conversation returning either; an exception instance is raised."""

    def __init__(self, steps, model="stub/qwen"):
        self.steps = list(steps)
        self.config = SimpleNamespace(model=model, max_output_tokens=4096)
        self.n = 0

    async def chat(self, messages, *, tools=None, **_kw):
        from olmoearth_agent.llm.types import ChatResponse, ToolCall
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, BaseException):
            raise step
        usage = {"prompt_tokens": 900 + 10 * len(messages), "completion_tokens": 40,
                 "total_tokens": 940 + 10 * len(messages)}
        if isinstance(step, str):
            return ChatResponse(content=step, thinking="the tool ranked them", finish_reason="stop", usage=usage)
        calls = []
        for name, args in step:
            self.n += 1
            calls.append(ToolCall(id=f"chatcmpl-tool-{self.n}", name=name, arguments=args))
        return ChatResponse(content=None, tool_calls=calls, finish_reason="tool_calls", usage=usage)


def cell_value(values, grid=4, nodata=NODATA_CELLS):
    """A result's raw value at a point: the grid cell's value, or -1 (the model's nodata_value) in no-data cells."""
    def value(fx, fy):
        i = min(grid - 1, int(fy * grid)) * grid + min(grid - 1, int(fx * grid))
        return -1.0 if i in nodata else values[i]
    return value


class StubStudio:
    """Stands in for StudioClient: [0, 1] regression results over one extent, each from its own model, whose
    nodata_value is -1. The pixel-value record echoes the point, as a live record may, to exercise the sanitizer."""

    def __init__(self, fields, extra=None):
        self.config = SimpleNamespace(base_url="https://olmoearth.allenai.org/api/v1")
        self.fields = fields
        self.extra = extra or {}
        self.calls = collections.Counter()

    async def get_prediction_result(self, result_id):
        self.calls["get_prediction_result"] += 1
        return {"id": result_id, "prediction_id": f"pred-{result_id}", "property_names": [PROP],
                "result_metadata": {"geometry": GEOM, "regression_fields": [
                    {"property_name": PROP, "min_value": 0.0, "max_value": 1.0}]},
                "tile_urls": [f"https://tiles.invalid/{result_id}/{{z}}/{{x}}/{{y}}.png?token=t0k3n"],
                "download_token": "dl-9f8e7d6c", **self.extra}

    async def get_prediction(self, prediction_id):
        self.calls["get_prediction"] += 1
        return {"id": prediction_id, "model_id": f"model-{prediction_id}", "project_id": "proj-karst",
                "start_time": "2025-06-01T00:00:00Z"}

    async def get_model(self, model_id):
        self.calls["get_model"] += 1
        return {"id": model_id, "name": "KarstBinary", "model_type": "fine_tuned",
                "wizard_answers": {"prediction_type": "per_pixel_regression", "nodata_value": -1}}

    async def pixel_value(self, result_id, lon, lat):
        self.calls["pixel_value"] += 1
        fx = (lon - BBOX[0]) / (BBOX[2] - BBOX[0])
        fy = (lat - BBOX[1]) / (BBOX[3] - BBOX[1])
        return {"lon": lon, "lat": lat, "bands": [{"property_name": PROP, "raw_value": self.fields[result_id](fx, fy),
                                                   "regression": {"min_value": 0.0, "max_value": 1.0}}]}

    async def aclose(self):
        pass


def _last_tool_output(messages):
    return json.loads(messages[-1].content)["result"]


def b2_answer(messages):
    """What a good answer to B2 says: the tool's first windows, lowest margin first, and no accuracy."""
    rows = _last_tool_output(messages)["review"]
    a, b = rows[0], rows[1]
    return (f"Check window ({a['row']}, {a['col']}) first: its margin {a['margin']} is the lowest, so the map is least "
            f"sure there. Then window ({b['row']}, {b['col']}) (margin {b['margin']}). The most confident windows come "
            "last. How good the map is cannot be said without labels.")


B2_CALL = ("olmoearth_review_set_from_result", {"result_id": "res-binary", "grid": 4, "budgets": [0.1, 0.25]})


def b2_llm():
    return ScriptedLLM([[B2_CALL], b2_answer])


def b2_studio(extra=None):
    return StubStudio({"res-binary": cell_value(VALUES)}, extra=extra)


def make_trial(root, fixtures=None, brief_values=None):
    """A trial directory: fixtures with their sha256 in trial.json, and round 1 with its round.json."""
    fdir = os.path.join(root, "fixtures")
    os.makedirs(fdir, exist_ok=True)
    for name, payload in (fixtures or {"f1_scores.json": {"grid": [2, 2], "scores": [[0.9, 0.1]] * 4}}).items():
        with open(os.path.join(fdir, name), "w", newline="") as fh:
            fh.write(payload if isinstance(payload, str) else json.dumps(payload))
    hashes = {os.path.relpath(os.path.join(h, f), fdir): drv.sha256_file(os.path.join(h, f))
              for h, _, fs in os.walk(fdir) for f in fs}
    with open(os.path.join(root, "trial.json"), "w") as fh:
        json.dump({"model": e86.MODEL, "fixtures": hashes}, fh)
    rdir = os.path.join(root, "rounds", "1")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "round.json"), "w") as fh:
        json.dump({"brief_values": brief_values or {}, "fixes": [], "not_run": {}}, fh)
    return rdir


def run(rdir, brief, provider, number, llm, studio, **kw):
    return drv.run_one(rdir, brief, provider, number, llm=llm, studio=studio, agent_info=dict(AGENT),
                       require_installed_inferencex=False, **kw)


# --------------------------------------------------------------------------------------------- the driver's own logic
def test_the_briefs_read_from_the_scorers_source_are_the_scorers():
    c = drv.scorer_constants()
    assert c.briefs == {k: v["brief"] for k, v in e86.BRIEF_CONFIGS.items()}
    assert c.provider == e86.PROVIDER
    assert c.conditional == {k for k, v in e86.BRIEF_CONFIGS.items() if v.get("conditional")}


def test_a_filled_brief_is_the_preregistered_text_and_a_missing_value_is_refused():
    c = drv.scorer_constants()
    values = {"model": "AWF", "model_a": "AWF", "model_b": "LCC", "area": "the Klamath area",
              "design_path": "f2_design.json", "labels_path": "f2_labels.csv", "scores_a": "f4_a.json",
              "scores_b": "f4_b.json", "date_a": "2018-10-01", "date_b": "2018-10-20"}
    for config, template in c.briefs.items():
        text = drv.fill_brief(template, values)
        assert e86.brief_matches(config, text), config
    with pytest.raises(drv.DriverError, match="date_b"):
        drv.fill_brief(c.briefs["B7/files"], {k: v for k, v in values.items() if k != "date_b"})
    assert drv.brief_files(c.briefs["B7/files"], values) == {"scores_a": "f4_a.json", "scores_b": "f4_b.json"}


def test_secret_check_refuses_a_file_that_would_hold_a_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMOEARTH_API_KEY", KEY)
    monkeypatch.setenv("LLM_API_KEY", "EMPTY")                       # the local server's placeholder: not a secret
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")            # any *_TOKEN is checked too
    monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:18077/v1")
    secrets = drv.environment_secrets()
    assert {"OLMOEARTH_API_KEY", "HF_TOKEN", "LLM_ENDPOINT"} <= set(secrets) and "LLM_API_KEY" not in secrets
    w = drv.RunWriter(str(tmp_path), secrets)
    w.text("ok.txt", "no secret here")
    for write, rel, payload in ((w.text, "a.txt", f"key={KEY}"), (w.json, "b.json", {"headers": f"Bearer {KEY}"}),
                                (w.jsonl, "c.jsonl", {"url": "http://localhost:18077/v1/chat"}),
                                (w.text, "d.txt", "hf_abcdefghijklmnop")):
        with pytest.raises(drv.SecretLeak) as err:
            write(rel, payload)
        assert not (tmp_path / rel).exists()
        assert KEY not in str(err.value) and ("OLMOEARTH_API_KEY" in str(err.value) or "LLM_ENDPOINT" in
                                              str(err.value) or "HF_TOKEN" in str(err.value))
    # a file the tools wrote is caught by the sweep after the run
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "tool_output.json").write_text(json.dumps({"echo": KEY}))
    assert w.sweep() == [(os.path.join("workspace", "tool_output.json"), ["OLMOEARTH_API_KEY"])]
    assert drv.redact(f"POST {KEY} failed", secrets) == "POST <OLMOEARTH_API_KEY> failed"


def test_studio_records_keep_what_the_scorer_reads_and_drop_coordinates_and_credentials(tmp_path):
    w = drv.RunWriter(str(tmp_path), {})
    log = drv.StudioLog(w, lambda: 1.0, 0.0)
    record = {"lon": -77.61, "lat": 40.51, "bands": [{"property_name": PROP, "raw_value": -1.0,
                                                      "regression": {"min_value": 0.0, "max_value": 1.0}}],
              "result_metadata": {"geometry": GEOM, "regression_fields": [{"property_name": PROP}]},
              "download_token": "dl", "tile_urls": ["x"], "shared_extent_bbox": [1, 2, 3, 4]}
    dropped = []
    kept = drv.sanitize(record, dropped)
    assert kept == {"bands": record["bands"], "result_metadata": {"regression_fields": [{"property_name": PROP}]}}
    assert sorted(dropped) == ["download_token", "lat", "lon", "result_metadata.geometry", "shared_extent_bbox",
                               "tile_urls"]
    k = log.point_key(-77.61, 40.51)
    assert k == log.point_key(-77.61, 40.51) and k != log.point_key(-77.61, 40.52)   # one point, one key
    assert len(k) == 20 and all(ch in "0123456789abcdef" for ch in k)
    assert drv.StudioLog(w, lambda: 1.0, 0.0).point_key(-77.61, 40.51) != k      # keyed per run: not invertible


RUN_BRIEF = '''
async def run_brief(brief, *, max_turns=8, llm=None, studio=None, registry=None, skill_index=None):
    llm = llm or OlmoEarthLLM()
    registry = registry or build_default_registry()
    if skill_index is None:
        skill_index = SkillLoader().index()
    agent = LeadAgent(llm, registry, studio, state=ThreadState(), skill_index=skill_index,
                      memory_block=preferences_block(), local=True)
    return await agent.run(brief, max_turns=max_turns)
'''


def test_a_change_in_how_the_command_line_builds_the_agent_is_noticed():
    assert drv.construction_problems(RUN_BRIEF) == []
    changed = drv.construction_problems(RUN_BRIEF.replace("local=True", "local=False, forced_skill='x'"))
    assert changed and "local" in changed[0]
    assert drv.construction_problems(RUN_BRIEF.replace("SkillLoader().index()", "''"))


@needs_agent
def test_the_installed_command_line_builds_the_agent_as_the_driver_does():
    drv.check_cli_construction()


# --------------------------------------------------------------------------------------------- a round of B2 runs
@pytest.fixture(scope="module")
def b2_round(tmp_path_factory):
    """Three B2 studio runs through the agent's harness, as the driver writes them."""
    if not HAVE_AGENT:
        pytest.skip("needs the olmoearth_agent package")
    root = str(tmp_path_factory.mktemp("trial"))
    rdir = make_trial(root)
    results = [run(rdir, "B2", "studio", k, b2_llm(), b2_studio()) for k in (1, 2, 3)]
    return {"trial": root, "round": rdir, "results": results,
            "run_dir": os.path.join(rdir, "runs", "B2", "studio", "1")}


def _load(d):
    return e86.load_run(d)


@needs_agent
def test_the_scorer_grades_the_driven_round(b2_round):
    assert [r["exit_code"] for r in b2_round["results"]] == [0, 0, 0]
    r = e86.score_round(b2_round["round"], b2_round["trial"])
    cfg = r["configurations"]["B2/studio"]
    assert cfg["n_counted_runs"] == 3 and not cfg["excluded_runs"]
    graded = {c: [g[c] for g in cfg["runs"]] for c in e86.CRITERIA}
    assert all(g["status"] != e86.UNGRADEABLE for gs in graded.values() for g in gs), graded
    # everything the scripted answer controls passes; parity (7) is the agent's port against this package
    for c in ("c1_routing", "c2_grounding", "c3_ranking", "c4_nodata", "c5_declines", "c6_coordinates"):
        assert cfg["verdicts"][c] == e86.PASS, (c, graded[c])
    assert cfg["verdicts"]["c7_parity"] == e86.PASS, graded["c7_parity"]
    assert r["agent_commit"] == AGENT["commit"] and r["model"] == "stub/qwen"


@needs_agent
def test_brief_txt_is_the_exact_brief(b2_round):
    run_ = _load(b2_round["run_dir"])
    assert run_["brief"] == e86.BRIEF_CONFIGS["B2/studio"]["brief"] and e86.brief_matches("B2/studio", run_["brief"])


@needs_agent
def test_stdout_txt_is_the_answer_as_the_command_line_prints_it(b2_round):
    with open(os.path.join(b2_round["run_dir"], "stdout.txt")) as fh:
        out = fh.read()
    run_ = _load(b2_round["run_dir"])
    assert out == run_["final_event"] + "\n" and out.startswith("Check window (")


@needs_agent
def test_stderr_txt_is_the_show_trace_output(b2_round):
    with open(os.path.join(b2_round["run_dir"], "stderr.txt")) as fh:
        lines = fh.read().splitlines()
    assert lines == ["  [ok] olmoearth_review_set_from_result", "  (2 turn(s), 1 provenance entries)"]
    assert e86.trace_names("\n".join(lines)) == ["olmoearth_review_set_from_result"]


@needs_agent
def test_provenance_json_is_the_agents_manifest_and_agrees_with_the_events(b2_round):
    run_ = _load(b2_round["run_dir"])
    entries = run_["provenance"]["entries"]
    assert [e["api_call"] for e in entries] == [c["name"] for c in run_["calls"]]
    assert all(e86._args_hash(c["arguments"]) == e["request_hash"] for c, e in zip(run_["calls"], entries))
    assert run_["provenance"]["egress"][0]["host"] == "olmoearth.allenai.org"


@needs_agent
def test_events_jsonl_carries_every_event_with_the_full_tool_output_and_its_timing(b2_round):
    run_ = _load(b2_round["run_dir"])
    types = [e["type"] for e in run_["events"]]
    assert types == ["tool_call", "tool_result", "thinking", "final"]
    assert all(isinstance(e["t"], float) and isinstance(e["seconds"], float) for e in run_["events"])
    out = run_["calls"][0]["result"]
    assert out["ranked"] and out["sampling"]["n_valid"] == 14 and out["sampling"]["n_nodata_dropped"] == 2
    assert run_["calls"][0]["arguments"] == B2_CALL[1] and len(out["review"]) == 4


@needs_agent
def test_usage_jsonl_has_one_line_per_model_call(b2_round):
    run_ = _load(b2_round["run_dir"])
    assert len(run_["usage"]) == 2
    assert all({"prompt_tokens", "completion_tokens", "total_tokens", "seconds"} <= set(u) for u in run_["usage"])
    t = e86.time_and_tokens(run_)
    assert t["n_llm_calls"] == 2 and t["total_tokens"] == sum(u["total_tokens"] for u in run_["usage"])


@needs_agent
def test_run_json_records_the_run_the_agent_and_the_model(b2_round):
    meta = _load(b2_round["run_dir"])["meta"]
    assert meta["configuration"] == "B2/studio" and meta["run"] == "1" and meta["exit_code"] == 0
    assert meta["status"] == "answered" and "void" not in meta and meta["model"] == "stub/qwen"
    assert meta["agent"]["commit"] == AGENT["commit"] and meta["max_turns"] == 8 and meta["turns"] == 2
    assert meta["started"] <= meta["ended"] and meta["seconds"] >= 0
    assert meta["sampling"]["mode"] == "thinking_general" and meta["workspace"]["preference_memory_chars"] == 0
    assert meta["secret_check"]["leaks"] == [] and "olmoearth_review_set_from_result" in meta["tools_registered"]


@needs_agent
def test_studio_calls_jsonl_holds_every_sample_under_its_tool_call_without_coordinates(b2_round):
    run_ = _load(b2_round["run_dir"])
    call_id = run_["calls"][0]["id"]
    samples = [s for s in run_["studio"] if s["kind"] == "pixel_value"]
    assert len(samples) == 16 and {s["call_id"] for s in samples} == {call_id}
    assert len({s["point"] for s in samples}) == 16
    assert {s["kind"] for s in run_["studio"]} >= {"prediction_result", "prediction", "model"}
    assert sum(s["record"]["bands"][0]["raw_value"] == -1.0 for s in samples) == 2

    def numbers(o):
        if isinstance(o, dict):
            return [x for v in o.values() for x in numbers(v)]
        if isinstance(o, list):
            return [x for v in o for x in numbers(v)]
        return [o] if isinstance(o, float) else []
    values = [x for s in run_["studio"] for x in numbers(s)]
    assert not any(BBOX[0] <= x <= BBOX[2] or BBOX[1] <= x <= BBOX[3] for x in values)   # no longitude, no latitude
    with open(os.path.join(b2_round["run_dir"], "studio_calls.jsonl")) as fh:
        text = fh.read()
    assert "dl-9f8e7d6c" not in text and "t0k3n" not in text                          # nor a Studio credential
    assert e86.grade_nodata(run_)["status"] == e86.PASS            # 4c recomputes the counts from these samples


@needs_agent
def test_workspace_holds_the_fixtures_and_every_file_a_tool_wrote(b2_round):
    run_ = _load(b2_round["run_dir"])
    ws = run_["workspace"]
    assert os.path.exists(os.path.join(ws, "f1_scores.json"))
    resolve = e86.make_resolver(ws)
    scores = resolve(run_["calls"][0]["result"]["scores_path"])
    assert scores and os.path.dirname(scores) == ws
    assert not os.path.exists(os.path.join(ws, "memory"))           # no preference was remembered


@needs_agent
def test_round_json_pins_the_round_to_one_agent_commit(b2_round):
    with open(os.path.join(b2_round["round"], "round.json")) as fh:
        meta = json.load(fh)
    assert meta["agent_commit"] == AGENT["commit"] and meta["model"] == "stub/qwen" and meta["max_turns"] == 8
    later = dict(AGENT, commit="1" * 40)
    with pytest.raises(drv.DriverError, match="agent_commit"):
        drv.run_one(b2_round["round"], "B2", "studio", 4, llm=b2_llm(), studio=b2_studio(), agent_info=later,
                    require_installed_inferencex=False)
    with pytest.raises(drv.DriverError, match="exists"):
        run(b2_round["round"], "B2", "studio", 1, b2_llm(), b2_studio())
    assert not os.path.exists(os.path.join(b2_round["round"], "runs", "B2", "studio", "4"))


# --------------------------------------------------------------------------------------------- refusals, voids, leaks
@needs_agent
def test_refusals_write_nothing(tmp_path):
    rdir = make_trial(str(tmp_path), brief_values={"B8/cluster": {"model": "AWF", "area": "the Klamath area"}})
    with pytest.raises(drv.DriverError, match="brief_values"):
        run(rdir, "B5", "files", 1, b2_llm(), b2_studio())
    with pytest.raises(drv.DriverError, match="provider"):
        run(rdir, "B8", "cluster", 1, b2_llm(), b2_studio(), registry=SimpleNamespace(names=lambda: []))
    with pytest.raises(drv.DriverError, match="not a preregistered configuration"):
        run(rdir, "B1", "cluster", 1, b2_llm(), b2_studio())
    with pytest.raises(drv.DriverError, match="tree differs"):
        drv.run_one(rdir, "B2", "studio", 1, llm=b2_llm(), studio=b2_studio(), agent_info=dict(AGENT, dirty=True),
                    require_installed_inferencex=False)
    with open(os.path.join(str(tmp_path), "fixtures", "f1_scores.json"), "a") as fh:
        fh.write(" ")
    with pytest.raises(drv.DriverError, match="differs from its sha256"):
        run(rdir, "B2", "studio", 1, b2_llm(), b2_studio())
    assert not os.path.exists(os.path.join(rdir, "runs"))


@needs_agent
def test_an_unreachable_model_endpoint_voids_the_run_and_the_scorer_skips_it(tmp_path):
    import httpx
    import openai
    rdir = make_trial(str(tmp_path))
    down = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:18077/v1/chat/completions"))
    llm = ScriptedLLM([[B2_CALL], down])
    r = run(rdir, "B2", "studio", 1, llm, b2_studio())
    assert r["exit_code"] == drv.EXIT_VOID and r["status"] == "void"
    meta = _load(r["run_dir"])["meta"]
    assert meta["void"].startswith("the model endpoint was unreachable")
    cfg = e86.score_round(rdir, str(tmp_path))["configurations"]["B2/studio"]
    assert cfg["n_counted_runs"] == 0 and cfg["excluded_runs"][0]["why"].startswith("void")
    assert drv.counted_runs(rdir, "B2/studio") == 0 and drv.next_run_number(rdir, "B2/studio") == 2


@needs_agent
@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name != "posix", reason="POSIX signals")
def test_a_job_ending_mid_run_voids_the_run(tmp_path):
    """Slurm sends SIGTERM before it kills a job at its time limit: the run is written, marked void, and replaced."""
    async def killed(_messages):
        os.kill(os.getpid(), signal.SIGTERM)       # the handler is installed: the driver cancels its own run
        await asyncio.sleep(10)

    class DyingLLM(ScriptedLLM):
        async def chat(self, messages, *, tools=None, **kw):
            if not self.steps:
                return await killed(messages)
            return await super().chat(messages, tools=tools, **kw)

    rdir = make_trial(str(tmp_path))
    r = run(rdir, "B2", "studio", 1, DyingLLM([[B2_CALL]]), b2_studio(), handle_sigterm=True)
    assert r["exit_code"] == drv.EXIT_VOID and r["seconds"] < 5
    run_ = _load(r["run_dir"])
    assert run_["meta"]["void"] == "the job ended mid-run (SIGTERM)" and len(run_["calls"]) == 1
    assert run_["provenance"]["entry_count"] == 1


@needs_agent
def test_a_secret_in_a_studio_response_is_never_written_and_ends_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OLMOEARTH_API_KEY", KEY)
    rdir = make_trial(str(tmp_path))
    r = run(rdir, "B2", "studio", 1, b2_llm(), b2_studio(extra={"description": f"shared with {KEY}"}))
    assert r["exit_code"] == drv.EXIT_REFUSED and r["status"] == "error"
    for here, _, files in os.walk(r["run_dir"]):
        for f in files:
            with open(os.path.join(here, f), "rb") as fh:
                assert KEY.encode() not in fh.read(), f
    meta = _load(r["run_dir"])["meta"]
    assert "OLMOEARTH_API_KEY" in meta["error"] and meta["secret_check"]["leaks"]
    assert "OLMOEARTH_API_KEY" in meta["secret_check"]["variables"]


# -------------------------------------------------------------------------------------------- B3: no-data recording
@needs_agent
def test_the_studio_log_reproduces_a_comparison_without_its_no_data(tmp_path):
    """24 September's B3 failure was -1 counted as data. The driver's log must let 4c recompute the pair statistic:
    both results' samples at each point under the comparison's call id, with the model's nodata_value.

    The committed scorer reads the pair's ids from the call's arguments (result_id_a, result_id_b); the agent's
    olmoearth_compare_results now takes result_ids and states the pair in its output, so the ids are given to the
    scorer's recomputation from the output here. That reader needs a dated amendment before the first run."""
    rng = np.random.default_rng(3)
    a, b = rng.random(36).round(4).tolist(), rng.random(36).round(4).tolist()
    nodata = tuple(range(0, 36, 4))
    studio = StubStudio({"res-a": cell_value(a, 6, nodata), "res-b": cell_value(b, 6, nodata)})
    args = {"result_ids": ["res-a", "res-b"], "mode": "pair", "kind": "cross_model", "grid": 6}
    llm = ScriptedLLM([[("olmoearth_compare_results", args)],
                       "The two maps cannot be compared value for value without labels, and which is right cannot "
                       "be resolved without labels."])
    rdir = make_trial(str(tmp_path))
    r = run(rdir, "B3", "studio", 1, llm, studio)
    run_ = _load(r["run_dir"])
    call = run_["calls"][0]
    out = call["result"]
    assert call["ok"] and out["comparable"] and out["n_nodata_dropped"] == len(nodata), out
    by_call, *ctx = e86._studio_index(run_)
    samples = by_call[call["id"]]
    assert len([s for s in samples if s["result_id"] == "res-a"]) == 36
    given = dict(call, arguments={**call["arguments"], "result_id_a": out["result_id_a"],
                                  "result_id_b": out["result_id_b"]})
    assert e86._check_compare_results(given, samples, ctx) == (None, None)
    clean = [(x, y) for i, (x, y) in enumerate(zip(a, b)) if i not in nodata]
    assert out["stats"]["n_samples"] == len(clean)


# --------------------------------------------------------------------------------------------- criterion 7a
def _plan_fixture(scores_root, f1, budget, design, seed, rng, registry):
    """A labelled design drawn by the agent's plan handler on F1, as the plan builds F2 and F3."""
    from olmoearth_agent.harness.state import ThreadState
    from olmoearth_agent.llm.types import ToolCall
    from olmoearth_agent.tools.registry import ToolContext
    call = ToolCall(id="fx", name="olmoearth_plan_label_sample",
                    arguments={"scores_path": f1, "budget": budget, "design": design, "seed": seed})
    with drv.run_environment(scores_root):
        env = asyncio.run(registry.dispatch(call, ToolContext(studio=drv._NoStudio(), state=ThreadState())))
    assert env["ok"], env
    out = env["result"]
    with open(out["labels_csv_path"], newline="") as fh:
        reader = csv.DictReader(fh)
        rows, fields = list(reader), reader.fieldnames
    for r in rows:
        r["wrong"] = str(int(rng.random() < 0.15))
    return out["design_path"], rows, fields


@needs_agent
def test_the_parity_calls_are_graded_by_the_scorer(tmp_path):
    from olmoearth_agent.skills import build_default_registry
    registry = build_default_registry()
    rng = np.random.default_rng(0)
    z = rng.normal(size=(900, 3)) * 1.5
    rows = (np.exp(z) / np.exp(z).sum(1, keepdims=True)).round(6).tolist()
    build = tmp_path / "build"
    build.mkdir()
    (build / "f1_scores.json").write_text(json.dumps({"grid": [30, 30], "scores": rows}))
    fixtures = {"f1_scores.json": (build / "f1_scores.json").read_text()}
    for tag, budget, design, seed in (("f2", 120, "confidence", 0), ("f3", 150, "random", 1)):
        design_path, sheet, fields = _plan_fixture(str(build), "f1_scores.json", budget, design, seed, rng, registry)
        with open(design_path) as fh:
            fixtures[f"{tag}_design.json"] = fh.read()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sheet)
        fixtures[f"{tag}_labels.csv"] = buf.getvalue()
    s = rng.random(100)
    flip = rng.random(100) < 0.2
    fixtures["f4_a.json"] = json.dumps({"grid": [10, 10], "scores": [[1 - v, v] for v in s]})
    fixtures["f4_b.json"] = json.dumps({"grid": [10, 10], "scores": [[v, 1 - v] if f else [1 - v, v]
                                                                     for v, f in zip(s, flip)]})
    values = {"B5/files": {"design_path": "f2_design.json", "labels_path": "f2_labels.csv"},
              "B6/files": {"design_path": "f3_design.json", "labels_path": "f3_labels.csv"},
              "B7/files": {"scores_a": "f4_a.json", "scores_b": "f4_b.json", "date_a": "2019-03-01",
                           "date_b": "2019-03-20"}}
    trial = tmp_path / "trial"
    rdir = make_trial(str(trial), fixtures=fixtures, brief_values=values)
    made = drv.run_parity(rdir, "f1_scores.json", registry=registry, agent_info=dict(AGENT),
                          require_installed_inferencex=False)
    assert [ok for _, ok in made] == [True] * 6, made
    res = e86.fixed_parity(rdir, str(trial))
    assert {c["tool"] for c in res["calls"]} == set(e86.PARITY_FIXED)
    assert res["status"] == e86.PASS, res["reasons"]
    with pytest.raises(drv.DriverError, match="once per round"):
        drv.run_parity(rdir, "f1_scores.json", registry=registry, agent_info=dict(AGENT),
                       require_installed_inferencex=False)

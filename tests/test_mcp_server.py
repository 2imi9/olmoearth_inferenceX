"""The MCP server (oe_inferencex.mcp_server) as an agent meets it, through the MCP SDK's own client.

The server runs in this process, on the SDK's in-memory streams, and once as `oe-inferencex mcp` over stdio. Each tool
runs on the quick start's map (examples/quickstart_map.py, docs/Usage.md) and must give the numbers its commands
print, a conclusion and limits in plain words, and the package's own message when the package refuses. The texts
(instructions, capability cards, SKILL.md) are checked without the extra; everything else is skipped cleanly when
the mcp extra is not installed.
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import re
import shlex
import sys

import numpy as np
import pytest
import yaml

from oe_inferencex import cli, mcp_server
from oe_inferencex import assess as assess_mod
from oe_inferencex import estimate as est

try:
    import anyio
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.server.fastmcp import FastMCP  # noqa: F401  (mcp 2 has no FastMCP: the extra pins below 2)
    from mcp.shared.memory import create_client_server_memory_streams
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False
HAVE_GEO = importlib.util.find_spec("rasterio") is not None

needs_mcp = pytest.mark.skipif(not HAVE_MCP, reason="the mcp extra is not installed")
needs_map = pytest.mark.skipif(not (HAVE_MCP and HAVE_GEO), reason="needs the mcp extra and rasterio (the quick-start "
                                                                   "map is a GeoTIFF)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
quickstart_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quickstart_map)

TOOLS = {"guide", "assess", "compare", "sample", "estimate", "certify"}


def _flat(text):
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- the session
def _session(work):
    """Run `work(client, init)` against the server in this process; returns what it returns."""
    server = mcp_server.build_server()
    low = server._mcp_server                     # the SDK's own in-memory pairing does the same (mcp.shared.memory)

    async def go():
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: low.run(server_streams[0], server_streams[1], low.create_initialization_options()))
                async with ClientSession(*client_streams) as client:
                    init = await client.initialize()
                    result = await work(client, init)
                tg.cancel_scope.cancel()
        return result

    return anyio.run(go)


def _call(name, **arguments):
    """One tool call: (is_error, the JSON it returned, or the error's text)."""
    async def work(client, _):
        return await client.call_tool(name, arguments)

    res = _session(work)
    text = res.content[0].text
    if res.isError:
        return True, text
    return False, (json.loads(text) if name != "guide" else text)


def _ok(name, **arguments):
    err, out = _call(name, **arguments)
    assert not err, out
    for key in ("conclusion", "limits", "next", "files", "summary"):
        assert key in out, key
    assert all(isinstance(out[k], str) and out[k] for k in ("conclusion", "limits", "next"))
    for path in out["files"].values():
        assert os.path.isabs(path) and os.path.exists(path), path
    return out


def _refused(name, **arguments):
    err, text = _call(name, **arguments)
    assert err, f"{name} was not refused: {text}"
    return text


@pytest.fixture(scope="module")
def qs(tmp_path_factory):
    """The quick-start map: scores.tif, other.tif and truth.tif, as the quick start writes them."""
    d = tmp_path_factory.mktemp("quickstart")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            quickstart_map.write_maps()
    finally:
        os.chdir(cwd)
    return d


def _label(path, qs):
    """The reviewer's step of the quick start, from truth.tif."""
    with contextlib.redirect_stdout(io.StringIO()):
        quickstart_map.label(str(path), truth_path=str(qs / "truth.tif"))


def _layer(path, layer):
    """An integer raster on the quick-start map's grid."""
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=256, width=256, count=1, dtype="int16", crs="EPSG:32632",
                       transform=from_origin(500000, 5000000, 10, 10)) as dst:
        dst.write(layer.astype(np.int16), 1)
    return str(path)


@pytest.fixture(scope="module")
def cond(qs):
    """A condition layer: the left half of the map 0 (clear), the right half 1 (cloudy)."""
    if not HAVE_GEO:
        pytest.skip("needs rasterio")
    layer = np.zeros((256, 256), np.int16)
    layer[:, 128:] = 1
    return _layer(qs / "condition.tif", layer)


# ----------------------------------------------------------------------------- the teaching, without the extra
def test_every_card_says_what_the_tool_does_needs_and_cannot_do():
    assert set(mcp_server.CARDS) == TOOLS == set(mcp_server.TOOLS)
    for name, card in mcp_server.CARDS.items():
        heads = [line.split(":")[0] for line in card.splitlines()[1:]]
        assert heads == ["Does", "Needs", "Cannot"], name
        assert "\u2014" not in card, name                                    # plain English, no em dash


def test_the_instructions_give_the_order_and_the_hard_rules():
    text = mcp_server.INSTRUCTIONS
    steps = [line for line in text.splitlines() if re.match(r"\d\. ", line)]
    # each step is the user's question, then the tools that answer it, in the standard order
    asked = [re.match(r"\d\. (.+?\?) \(([^)]+)\) ", s) for s in steps[:4]]
    assert all(asked), steps
    assert [(m.group(1), m.group(2)) for m in asked] == [
        ("Where should I look first?", "assess"), ("How wrong is the map?", "sample, label, estimate"),
        ("Which part can I trust?", "certify"), ("Which of two maps is better, and where do they differ?", "compare")]
    assert steps[4].startswith("5. Per condition: ")
    for rule in mcp_server.HARD_RULES:
        assert f"- {rule}" in text
    flat = _flat(" ".join(mcp_server.HARD_RULES))
    for said in ("A review set is not a sample", "certify needs a random sample", "compare cannot say which map is right",
                 "Labels are assumed right", "Ranking needs the scores, not only the class map"):
        assert said in flat, said
    assert "\u2014" not in text


def test_the_tools_are_presented_by_the_question_each_answers():
    """Four questions in the standard order; every tool but guide answers one, under its own name, with the question
    in its title and on the first line of its card."""
    assert list(mcp_server.QUESTIONS) == ["Where should I look first?", "How wrong is the map?",
                                          "Which part can I trust?",
                                          "Which of two maps is better, and where do they differ?"]
    answered = [name for names in mcp_server.QUESTIONS.values() for name in names]
    assert sorted(answered) == sorted(TOOLS - {"guide"})
    assert set(mcp_server.TITLES) == TOOLS and len(set(mcp_server.TITLES.values())) == len(TOOLS)
    for question, names in mcp_server.QUESTIONS.items():
        for name in names:
            assert mcp_server.TITLES[name].startswith(question), name
            assert mcp_server.CARDS[name].splitlines()[0].startswith(f'Answers "{question}"'), name
    assert "\u2014" not in " ".join(mcp_server.TITLES.values())


def test_the_skill_and_usage_give_the_same_questions():
    """SKILL.md's steps and Usage's table name the server's questions in its order, each with its tools."""
    skill = open(os.path.join(ROOT, "skills", "oe-inferencex", "SKILL.md"), encoding="utf-8").read()
    steps = re.findall(r"^\d\. \*\*(.+?\?)\*\* (.+)$", skill, re.M)
    assert [q for q, _ in steps] == list(mcp_server.QUESTIONS)
    usage = open(os.path.join(ROOT, "docs", "Usage.md"), encoding="utf-8").read()
    rows = re.findall(r"^\| (.+?\?) \| (.+?) \|", usage, re.M)
    assert [q for q, _ in rows] == list(mcp_server.QUESTIONS)
    for (_, said), (_, tools), names in zip(steps, rows, mcp_server.QUESTIONS.values()):
        assert re.findall(r"`(\w+)`", said.split(". ")[0]) == list(names)
        assert re.findall(r"`(\w+)`", tools) == list(names)


def test_the_skill_holds_the_same_teaching():
    path = os.path.join(ROOT, "skills", "oe-inferencex", "SKILL.md")
    text = open(path, encoding="utf-8").read()
    head = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert head, "SKILL.md starts with a frontmatter block"
    fields = yaml.safe_load(head.group(1))                      # as a skill loader reads it: no stray ": " in a value
    assert fields["name"] == "oe-inferencex" and len(fields["description"]) > 40
    body = _flat(text)
    for rule in mcp_server.HARD_RULES:
        assert _flat(rule) in body, rule
    for command in ("oe-inferencex assess", "oe-inferencex sample", "oe-inferencex estimate", "oe-inferencex certify",
                    "oe-inferencex compare", "--condition", "--design random"):
        assert command in body, command
    assert "\u2014" not in text
    # the server and --condition came with 1.4.0: the skill names the release and installs it from PyPI
    assert "not yet released" not in body and "1.4.0" in body
    assert 'pip install "olmoearth-inferencex[geo,mcp]"' in body


def test_the_texts_say_what_exp86_shows_and_what_the_server_reads():
    """The trial's fall was measured on the briefs its fixes were built against, mostly with round 8's tool-output
    changes; the standard order is the owner's addition; and a tool reads more than the paths it is given."""
    doc = _flat(mcp_server.__doc__)
    log = open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read()
    entry = _flat(log[log.index("**A local MCP server for agents.**"):log.index("## 1.3.1")])
    usage = _flat(open(os.path.join(ROOT, "docs", "Usage.md"), encoding="utf-8").read())
    for text in (doc, entry):
        assert "after which" not in text
        assert "7.5% to 2.2%" in text and "eight development briefs" in text and "round 8" in text
        assert "exp87" in text and "owner's addition" in text
    assert "the sidecar beside a sample and the scores raster it records" in doc
    assert "the sidecar beside a sample and the scores raster it records" in usage
    assert "read and write files on your machine only" not in usage


def test_the_rules_do_not_claim_a_refusal_the_package_does_not_make():
    """assess refuses a class map of several classes read as probabilities, not a 0/1 map nor a class map passed
    as logits (docs/Usage.md, Inputs)."""
    text = _flat(mcp_server.INSTRUCTIONS)
    assert "The package enforces them" not in text
    assert "The package refuses only what a rule says it refuses" in text
    rule = mcp_server.HARD_RULES[4]
    assert "not a 0/1 map" in rule and "logits=true" in rule
    cannot = mcp_server.CARDS["assess"].split("\nCannot: ")[1]
    assert "logits=true" in cannot and "not evidence" in cannot
    skill = _flat(open(os.path.join(ROOT, "skills", "oe-inferencex", "SKILL.md"), encoding="utf-8").read())
    assert "The package enforces them" not in skill


def test_two_calls_at_once_do_not_share_the_capture(monkeypatch):
    """stdout is redirected for the whole process. If the SDK ever ran two sync tools on worker threads at once, one
    call could restore the real stdout while the other still printed, and leave the process writing into a buffer.
    A lock keeps one call inside the redirection at a time."""
    import threading
    a_in, b_in, a_done = threading.Event(), threading.Event(), threading.Event()

    def fake_main(argv):
        if argv[0] == "a":
            print("a")
            a_in.set()
            b_in.wait(0.5)                  # with the lock, b cannot start until a has finished
        else:
            b_in.set()
            a_done.wait(5)
            print("b")

    monkeypatch.setattr(mcp_server.cli, "main", fake_main)
    got, real = {}, sys.stdout

    def run(name):
        got[name] = mcp_server._run([name])
        if name == "a":
            a_done.set()

    try:
        ta = threading.Thread(target=run, args=("a",))
        ta.start()
        a_in.wait(5)
        tb = threading.Thread(target=run, args=("b",))
        tb.start()
        ta.join(5)
        tb.join(5)
        left = sys.stdout
    finally:
        sys.stdout = real
    assert got == {"a": "a\n", "b": "b\n"}
    assert left is real


def test_the_mcp_command_says_how_to_install_the_extra(monkeypatch):
    """Without the SDK (and pydantic, which comes with it) the module still imports, so its texts can be read, and
    `oe-inferencex mcp` refuses with the install line instead of a traceback."""
    import importlib
    import oe_inferencex
    for name in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")] + ["mcp", "pydantic"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(oe_inferencex, "mcp_server", mcp_server)          # restored after the re-import below
    monkeypatch.delitem(sys.modules, "oe_inferencex.mcp_server")
    bare = importlib.import_module("oe_inferencex.mcp_server")
    assert bare is not mcp_server and bare.Field is None and bare.INSTRUCTIONS == mcp_server.INSTRUCTIONS
    with pytest.raises(ImportError, match=r"olmoearth-inferencex\[geo,mcp\]"):
        bare.build_server()
    with pytest.raises(SystemExit) as exc:
        cli.main(["mcp"])
    assert 'pip install "olmoearth-inferencex[geo,mcp]"' in str(exc.value)
    assert str(exc.value).startswith("oe-inferencex mcp: ")


def test_the_extra_pins_the_sdk_below_2():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    pin = re.search(r'^mcp = \["mcp>=([\d.]+),<2"\]', text, re.M)
    assert pin, "the mcp extra pins the MCP Python SDK to a range below 2"


# ----------------------------------------------------------------------------- the setup lines in the docs
SETUP_DOCS = ("README.md", "docs/Usage.md", "skills/oe-inferencex/SKILL.md", "CHANGELOG.md")
PYPI_ONE_LINER = 'claude mcp add --scope user oe-inferencex -- uvx --from "olmoearth-inferencex[geo,mcp]" oe-inferencex mcp'
COMMAND = re.compile(r'(?:uvx --from (?:"[^"]+"|[^\s`"]+) |claude mcp add --scope user oe-inferencex -- )'
                     r'oe-inferencex [a-z-]+')
# uv's form of `python quickstart_map.py`, for a reader who installed nothing: the script needs numpy and rasterio
QUICKSTART_UV = re.compile(r'uv run --with "([^"]+)" python quickstart_map\.py')


def _doc(path):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def _project():
    import tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]


def _texts():
    """Every text that gives a setup line: the docs, and the server module's own docstring."""
    return [(path, _doc(path)) for path in SETUP_DOCS] + [("oe_inferencex/mcp_server.py", mcp_server.__doc__)]


def _check_command(argv, where):
    """A command a reader is told to run: uvx's requirement names this package, its extras and its repository, and
    the command and subcommand exist (the subcommand's --help exits 0)."""
    project = _project()
    if argv[0] == "uvx":
        assert argv[1] == "--from", (where, argv)
        req = re.fullmatch(r"([a-z-]+)(?:\[([a-z,]+)\])?(?: @ (\S+))?", argv[2])
        assert req, (where, argv[2])
        extras = set(filter(None, (req.group(2) or "").split(",")))
        assert req.group(1) == project["name"], where
        assert extras <= set(project["optional-dependencies"]), (where, extras)
        assert req.group(3) in (None, "git+" + project["urls"]["Repository"]), where
        argv = argv[3:]
        if argv[1:] == ["mcp"]:
            assert "mcp" in extras, where
    assert argv[0] in project["scripts"], (where, argv)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        cli.main([*argv[1:], "--help"])
    assert exc.value.code == 0, (where, argv)


def test_every_setup_line_and_config_names_a_command_that_exists():
    """The one-liners (uvx from PyPI, the installed command) and the
    JSON configurations name this package, its extras and its repository, and a command and subcommand it has."""
    found = []
    for where, text in _texts():
        flat = _flat(text)
        for line in COMMAND.findall(flat):
            argv = shlex.split(line)
            if argv[0] == "claude":
                argv = argv[argv.index("--") + 1:]
            _check_command(argv, where)
            found.append((where, argv[0], argv[-1]))
        for block in re.findall(r"```json\n(.*?)\n```", text, re.S):
            for name, server in json.loads(block)["mcpServers"].items():
                assert name == "oe-inferencex", where
                _check_command([server["command"], *server["args"]], where)
                found.append((where, server["command"], "json"))
        for inline in re.findall(r'`(\{"command": [^`]+\})`', text):
            server = json.loads(inline)
            _check_command([server["command"], *server["args"]], where)
            found.append((where, server["command"], "json"))
    # since 1.4.0 the server is on PyPI: every text gives the PyPI one-liner, and none still waits for a release
    for where, text in _texts():
        flat = _flat(text)
        assert PYPI_ONE_LINER in flat, where
        assert not re.search(r"next release|not yet released", flat, re.I), where
    assert ("docs/Usage.md", "uvx", "json") in found and ("docs/Usage.md", "oe-inferencex", "json") in found
    assert ("README.md", "uvx", "demo") in found and ("CHANGELOG.md", "oe-inferencex", "mcp") in found


def test_every_claude_code_line_adds_the_server_for_every_folder():
    """Claude Code adds a server for the folder `claude mcp add` is run in unless the line says --scope user. The
    first question is asked in a new folder, so a line without it leaves the agent there with no server."""
    for where, text in _texts():
        lines = re.findall(r"claude mcp add[^`\n]*", _flat(text))
        assert lines, where
        for line in lines:
            assert line.startswith("claude mcp add --scope user oe-inferencex -- "), (where, line)


def test_the_quick_start_map_can_be_written_with_nothing_installed():
    """The example questions run on the files examples/quickstart_map.py writes. A reader who connected the server
    with uvx has installed nothing, so Usage gives uv's form of the script, which brings the package's
    geo extra: the script needs numpy and rasterio."""
    project = _project()
    assert "rasterio" in quickstart_map.__doc__
    for path in ("docs/Usage.md",):
        reqs = QUICKSTART_UV.findall(_flat(_doc(path)))
        assert reqs, path
        for req in reqs:
            m = re.fullmatch(r"([a-z-]+)\[([a-z,]+)\]", req)
            assert m and m.group(1) == project["name"], (path, req)
            assert any(dep.startswith("rasterio") for extra in m.group(2).split(",")
                       for dep in project["optional-dependencies"][extra]), (path, req)


def test_the_first_question_runs_on_the_demo_tile(tmp_path, monkeypatch, capsys):
    """The first question of the README and of Usage names the files `oe-inferencex demo` writes. Asked of assess as
    it says, with windows of 1 pixel and the tile's expert labels as reference, the answer is the demo's own: the 5%
    review set holds the share of the errors the demo's pinned audit records."""
    asked = set()
    for path in ("README.md", "docs/Usage.md"):
        quote = re.search(r"^> (Where should I look first.*?)\n\n", _doc(path), re.M | re.S)
        assert quote, path
        asked.add(_flat(quote.group(1).replace("\n> ", " ")))
    assert len(asked) == 1, asked
    question = asked.pop()
    scores, truth = re.findall(r"oe_inferencex_demo/\S+?\.npy", question)
    assert "windows of 1 pixel" in question and "grade the order against" in question
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo"]) == 0
    capsys.readouterr()
    out = mcp_server.assess(scores, str(tmp_path / "first"), patch=1, reference=truth)
    pinned = json.load(open(os.path.join(ROOT, "exp", "out", "demo_sample_audit.json")))["review_sets"]["0.05"]
    assert out["summary"]["against_reference"]["errors_captured_fraction"]["0.05"] == pinned["errors_captured_fraction"]
    assert out["summary"]["review_sets"]["0.05"] == pinned["n_windows"]
    assert f"the 5% review set holds {100 * pinned['errors_captured_fraction']:.0f}% of those" in out["conclusion"]


# ----------------------------------------------------------------------------- the example questions in the docs
EXAMPLE_DOCS = ("docs/Usage.md", "skills/oe-inferencex/SKILL.md")
# what each tool's example must say the answer cannot be, or can be: nothing, no ranking, no verdict without labels
EXAMPLE_LIMITS = {
    # a class map of several classes is refused; a 0/1 map, or a class map passed as logits, is ranked by raster
    # position, which the tool's conclusion calls not evidence (test_a_class_map_read_as_scores_is_not_called_a_ranking)
    "assess": ("It is not an error rate", "a class map alone is refused, or gives an order that is not evidence"),
    "estimate": ("No tool labels a window", "the interval assumes your labels are right"),
    "certify": ("It can be nothing", f"fails on at most {est.ZONE_DELTA:.0%} of samples", "random sample"),
    "compare": ("Without labels it cannot say which map is better",),
}


def _examples(path):
    """(question, tools, text) per example bullet of a doc: `- "Question?" Uses `tool`, then `tool`. ...`."""
    found = []
    for bullet in re.findall(r'^- ("[^\n]*(?:\n  [^\n]*)*)', _doc(path), re.M):
        said = _flat(bullet)
        m = re.match(r'"([^"]+\?[^"]*)" Uses (.+?)\. ', said)
        assert m, (path, said)
        found.append((m.group(1), re.findall(r"`([^`]+)`", m.group(2)), said))
    return found


def test_each_example_question_names_only_tools_that_exist():
    """Three to five example questions, the same in Usage and SKILL.md; each names the tools it uses, all
    of them tools of the server and commands of the command line, and says what the answer can and cannot be."""
    per_doc = {path: _examples(path) for path in EXAMPLE_DOCS}
    first = per_doc[EXAMPLE_DOCS[0]]
    assert 3 <= len(first) <= 5
    for path, examples in per_doc.items():
        assert examples == first, path
    for question, tools, said in first:
        assert tools, question
        assert set(tools) <= set(mcp_server.TOOLS), (question, tools)
        for tool in tools:                                          # each runs the command of the same name
            _check_command(["oe-inferencex", tool], question)
        for tool in tools:
            for limit in EXAMPLE_LIMITS.get(tool, ()):
                assert limit in said, (question, limit)
        assert "\u2014" not in said
    # together they ask the four questions: every tool but guide is used
    assert set().union(*(set(tools) for _, tools, _ in first)) == set(mcp_server.TOOLS) - {"guide"}


@pytest.mark.skipif(not HAVE_GEO, reason="the quick-start map is a GeoTIFF")
def test_the_example_questions_name_the_quick_start_files(qs):
    """The files the questions name are the ones examples/quickstart_map.py writes."""
    named = {f for question, _, _ in _examples("docs/Usage.md") for f in re.findall(r"\w+\.tif", question)}
    assert named == {"scores.tif", "other.tif"}
    for name in named | {"truth.tif"}:
        assert (qs / name).is_file(), name
    assert "truth.tif" in _examples("docs/Usage.md")[-1][2]


# ----------------------------------------------------------------------------- the server, in process
@needs_mcp
def test_the_server_lists_six_tools_each_described_by_its_card():
    async def work(client, init):
        return init, (await client.list_tools()).tools

    init, tools = _session(work)
    assert init.instructions == mcp_server.INSTRUCTIONS
    assert init.serverInfo.name == "oe-inferencex"
    assert init.serverInfo.version == mcp_server.__version__          # the package's version, not the SDK's
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.description == mcp_server.CARDS[t.name]
        assert t.title == t.annotations.title == mcp_server.TITLES[t.name]      # the question it answers
        for part in ("\nDoes: ", "\nNeeds: ", "\nCannot: "):
            assert part in t.description, (t.name, part)
        assert t.annotations.openWorldHint is False
    schema = {t.name: t.inputSchema for t in tools}
    assert schema["assess"]["required"] == ["scores", "out_dir"]
    assert set(schema["certify"]["required"]) == {"sample_csv", "alpha"}
    assert schema["sample"]["properties"]["design"]["anyOf"][0]["enum"] == ["confidence", "proportional", "random",
                                                                            "tiles", "condition"]
    assert all(p.get("description") for t in tools for p in t.inputSchema.get("properties", {}).values())


@needs_mcp
def test_guide_returns_the_instructions_and_every_card():
    err, text = _call("guide")
    assert not err
    assert text.startswith(mcp_server.INSTRUCTIONS)
    assert all(card in text for card in mcp_server.CARDS.values())
    # the cards come under the question each tool answers, in the standard order, guide last
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert heads == [*mcp_server.QUESTIONS, mcp_server.TITLES["guide"]]
    assert re.findall(r"^### (\w+)$", text, re.M) == ["assess", "sample", "estimate", "certify", "compare", "guide"]
    assert f"Relative paths are read from {os.getcwd()}." in text


@needs_map
def test_assess_says_where_to_look_and_that_it_is_not_an_error_rate(qs, capsys):
    out = _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit"))
    assert capsys.readouterr().out == ""                       # stdout is the protocol's channel on stdio
    s = out["summary"]
    # the quick start's line: 4096 windows of 4 px; review sets 1%: 41, 5%: 205, 10%: 410; boundary windows 40.0%
    assert s["n_windows"] == 4096 and s["review_sets"] == {"0.01": 41, "0.05": 205, "0.1": 410}
    assert s["boundary_window_fraction"] == pytest.approx(0.400390625)
    five = str(qs / "audit" / "review_set_05pct.csv")
    assert out["files"]["review_set_0.05"] == five
    assert "the review sets hold 41 (1%), 205 (5%), 410 (10%) windows" in out["conclusion"]
    assert f"The 5% review set is in {five}" in out["conclusion"]
    assert "40.0% of all windows sit on a class boundary" in out["conclusion"]
    written = json.load(open(qs / "audit" / "assessment.json"))
    for said in ("not how wrong the map is", "The review set is not a sample", "not the probability that the window "
                 "is wrong", "Errors the model is sure of come last", mcp_server.MCP_SCOPE["assess"],
                 written["warnings"][0], "confidence is the score the order ranks by"):
        assert said in out["limits"], said
    # the limit a small model dropped is inside the conclusion itself; the JSON keeps the package's full scope note
    assert "it does not say how wrong the map is" in out["conclusion"] and "review set is not a" in out["conclusion"]
    assert "read from the class probabilities" in out["limits"] and "two highest logits" not in out["limits"]
    assert written["scope"] == assess_mod.SCOPE_ASSESS and written["scope"] not in out["limits"]
    assert "sample with design \"random\"" in out["next"]


@needs_map
def test_sample_estimate_certify_give_the_readme_numbers(qs):
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "random"), budget=300, design="random")
    csv_path = str(qs / "random" / "to_label.csv")
    assert out["files"] == {"sample_csv": csv_path, "sidecar": csv_path[:-4] + ".json"}
    assert out["conclusion"].startswith("300 windows to label of 4096 valid (random design)")
    assert "No window is labelled yet" in out["limits"] and "Both estimate and certify read this sample" in out["limits"]
    assert f"sample_csv={csv_path}" in out["next"]

    _label(csv_path, qs)
    out = _ok("estimate", sample_csv=csv_path)
    # the quick start: error rate 7.0%, 95% interval 4.5% to 10.4% ... exact hypergeometric interval
    s = out["summary"]
    assert (s["estimate"], s["low"], s["high"]) == (0.07, 0.044921875, 0.103515625)
    assert out["conclusion"].startswith("The map's error rate is 7.0%, 95% interval 4.5% to 10.4%, from 300 labelled "
                                        "windows of 4096; exact hypergeometric interval")
    written = json.load(open(out["files"]["estimate"]))
    assert "Labels are assumed right" in out["limits"] and f"Note: {mcp_server.MCP_SCOPE['estimate']}" in out["limits"]
    assert "disagreement with the reviewer's labels, which are assumed right" in out["conclusion"]
    assert written["scope"] == est.SCOPE_ESTIMATE
    assert "certify with sample_csv=" in out["next"]

    out = _ok("certify", sample_csv=csv_path, alpha=0.05)
    s = out["summary"]
    assert (s["coverage"], s["n_zone"], s["n_population"]) == (0.9, 3686, 4096)
    assert out["conclusion"].startswith("Taken together, the 90% most confident windows (3686 of 4096, confidence >= "
                                        "0.6662) are wrong at most 5% of the time. The rate holds for them as a group, "
                                        "not for each window, and outside them nothing is certified.")
    assert "delta, which can be set lower" in out["conclusion"]
    assert np.load(out["files"]["zone_mask"]).sum() == 3686
    assert "Outside the certified windows nothing is certified" in out["limits"] and f"Note: {est.PREFIX_NOTE}" in out["limits"]
    assert f"Note: {mcp_server.MCP_SCOPE['certify']}" in out["limits"] and est.SCOPE_CERTIFY not in out["limits"]
    assert json.load(open(out["files"]["zone"]))["scope"] == est.SCOPE_CERTIFY

    out = _ok("certify", sample_csv=csv_path, alpha=0.01, out_dir=str(qs / "strict"))
    assert out["summary"]["coverage"] is None and out["conclusion"].startswith("No zone was certified at alpha 1%")
    assert "(80% of the map) held 243 labels with 1 wrong" in out["conclusion"]   # as the quick start's test finds
    assert "zone_mask" not in out["files"] and "labels at this alpha" in out["next"]


@needs_map
def test_compare_does_not_say_which_map_is_right_without_labels(qs):
    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff"))
    assert out["summary"]["n_disagree"] == 507 and out["summary"]["n_windows"] == 4096
    assert out["conclusion"].startswith("507 of 4096 windows differ (12.38%).")
    enr = out["summary"]["boundary_enrichment"]                                # two decimals: 1.05 is not "1.1"
    assert round(enr["boundary_a"], 1) == 2.6
    assert f"on a class boundary of map a {enr['boundary_a']:.2f} times as often" in out["conclusion"]
    assert mcp_server.HARD_RULES[2] in out["limits"]
    assert "neither which map is better nor which is right" in out["conclusion"]
    assert "Note: the dates the two maps describe were not given" in out["limits"]
    assert "compare again with labels" in out["next"]

    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff_l"),
              labels=str(qs / "truth.tif"))
    ws = out["summary"]["which_side"]
    assert (ws["a_right"], ws["b_right"], ws["neither"]) == (228, 239, 40)
    assert "a is right on 45% and b on 47% of the 507 windows" in out["conclusion"]   # the quick start: a 45%, b 47%
    assert "The labels are assumed right" in out["limits"]


@needs_map
def test_a_review_set_passed_as_a_sample_is_refused(qs):
    """The review set given to estimate or certify is refused with the package's message, and the tool says what the
    file is. Dressed up as a random sample (its windows in a sidecar that claims the random design), certify still
    refuses it, by the package's own check of where the labelled windows sit."""
    _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit_rs"))
    review = qs / "audit_rs" / "review_set_05pct.csv"
    for tool, extra in (("estimate", {}), ("certify", {"alpha": 0.05})):
        text = _refused(tool, sample_csv=str(review), **extra)
        assert "review_set_05pct.json not found" in text and "needs the sidecar `sample` wrote" in text
        assert "This CSV is a review set that assess wrote. A review set is not a sample." in text

    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "dressed"), budget=300, design="random")
    side = json.load(open(out["files"]["sidecar"]))
    rows = list(csv.DictReader(open(review)))
    truth_w = quickstart_map.to_windows(quickstart_map.truth_map())
    import rasterio
    with rasterio.open(qs / "scores.tif") as src:
        map_w = quickstart_map.to_windows(src.read().argmax(0))
    side["indices"] = [int(r["window_row"]) * 64 + int(r["window_col"]) for r in rows]
    side["budget"] = len(rows)
    path = qs / "dressed" / "review_as_sample.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "window_row", "window_col", "confidence", "wrong"])
        for i, r in zip(side["indices"], rows):
            rr, cc = int(r["window_row"]), int(r["window_col"])
            w.writerow([i, rr, cc, r["confidence"], int(map_w[rr, cc] != truth_w[rr, cc])])
    json.dump(side, open(str(path)[:-4] + ".json", "w"))
    text = _refused("certify", sample_csv=str(path), alpha=0.05)
    assert "an enriched set, not a random sample, and a zone certified on it would be wrong" in text


@needs_map
def test_refusals_carry_the_package_message(qs):
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "stratified"), budget=300)
    assert out["summary"]["design"] == "confidence"
    assert "certify refuses the confidence design" in out["limits"] and "certify" not in out["next"]
    _label(out["files"]["sample_csv"], qs)
    assert "certify refuses a confidence sample" in _ok("estimate", sample_csv=out["files"]["sample_csv"])["next"]
    text = _refused("certify", sample_csv=out["files"]["sample_csv"], alpha=0.05)
    assert "certify needs a random sample: this CSV was drawn with the 'confidence' design" in text

    text = _refused("assess", scores=str(qs / "truth.tif"), out_dir=str(qs / "classmap"))
    assert "values run 0 to 3, which is not a probability map" in text           # scores, not only the class map
    text = _refused("assess", scores=str(qs / "missing.tif"), out_dir=str(qs / "x"))
    assert "missing.tif is not a file" in text and "relative paths are read from" not in text
    text = _refused("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "x"), budget=300, design="condition")
    assert 'design="condition" needs condition' in text and "--" not in text       # the server's parameter names


@needs_map
def test_each_input_condition_on_its_own(qs, cond):
    names = ["0=clear", "1=cloudy"]
    out = _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit_c"), condition=str(cond),
              condition_names=names)
    assert "2 input conditions: clear 50.0% of the windows" in out["conclusion"]
    assert "review_set_05pct_by_condition.csv" in out["conclusion"]
    assert set(out["summary"]["conditions"]) == {"clear", "cloudy"}

    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "by_condition"), budget=300,
              condition=str(cond), condition_names=names)
    assert out["summary"]["design"] == "condition" and out["summary"]["per_condition"] == {"clear": 150, "cloudy": 150}
    _label(out["files"]["sample_csv"], qs)
    res = _ok("estimate", sample_csv=out["files"]["sample_csv"])
    assert set(res["summary"]["per_condition"]) == {"clear", "cloudy"}
    assert "Per input condition: clear " in res["conclusion"]
    assert "the intervals do not hold jointly at 95%" in res["limits"]
    res = _ok("certify", sample_csv=out["files"]["sample_csv"], alpha=0.05)
    assert res["summary"]["by_condition"] is True and res["summary"]["coverage"] is None
    assert res["conclusion"].startswith("Certified per input condition at alpha=5%")
    assert "clear: the 85% most confident windows of this condition" in res["conclusion"]
    assert np.load(res["files"]["zone_mask"]).sum() == res["summary"]["n_certified"]


@needs_map
def test_estimate_per_class_keeps_each_class_warning(qs):
    """With 40 labels every class rests on a few windows. The package tags each such class with a warning; the tool
    must pass it on in limits, in summary.per_class and in the conclusion, not only the class accuracies."""
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "pc40"), budget=40, design="random", seed=3)
    _label(out["files"]["sample_csv"], qs)
    res = _ok("estimate", sample_csv=out["files"]["sample_csv"], per_class=True)
    written = json.load(open(res["files"]["estimate"]))
    warned = {c: row for c, row in written["per_class"].items() if "warning" in row}
    assert warned, "the 40-label sample is meant to carry per-class warnings"
    limits = _flat(res["limits"])
    for c, row in warned.items():
        assert res["summary"]["per_class"][c]["warning"] == row["warning"]
        assert res["summary"]["per_class"][c]["warning_codes"] == row["warning_codes"]
        assert _flat(f"Warning: class {c}: {row['warning']}") in limits
    assert "carry a warning" in res["conclusion"]


@needs_map
def test_a_class_map_read_as_scores_is_not_called_a_ranking(qs):
    """A 4-class map refused as probabilities names logits; read as logits, or a 0/1 map read as probabilities, it
    is accepted, and its review sets tie. The conclusion must not tell the agent to check them first."""
    text = _refused("assess", scores=str(qs / "truth.tif"), out_dir=str(qs / "cm0"))
    assert "values run 0 to 3, which is not a probability map" in text
    assert "logits=true does not help" in text

    out = _ok("assess", scores=str(qs / "truth.tif"), out_dir=str(qs / "cm_logits"), logits=True)
    assert not out["conclusion"].startswith("Check the least confident windows first")
    assert "The order is not evidence here: 205 of the 205 windows of the 5% review set share the cut-off score with " \
           "524 windows left outside" in out["conclusion"]
    assert out["summary"]["tied_at_cutoff"]["0.05"] == {"inside": 205, "outside": 524}

    import rasterio
    with rasterio.open(qs / "truth.tif") as src:
        np.save(qs / "binary.npy", (src.read(1) == 0).astype(np.float32))
    out = _ok("assess", scores=str(qs / "binary.npy"), out_dir=str(qs / "cm_binary"))
    assert out["conclusion"].startswith("The order is not evidence here")

    out = _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit_ties"))
    assert out["conclusion"].startswith("Check the least confident windows first")     # real scores: no ties
    assert "tied_at_cutoff" not in out["summary"]


@needs_map
def test_condition_names_cannot_pass_options(qs, cond):
    """A name that starts with `-` would be read as a command-line option, here moving every output elsewhere."""
    elsewhere = qs / "elsewhere"
    for tool, extra in (("assess", {}), ("sample", {"budget": 30})):
        text = _refused(tool, scores=str(qs / "scores.tif"), out_dir=str(qs / "inj"), condition=cond,
                        condition_names=["0=clear", f"--out={elsewhere}"], **extra)
        assert "condition_names" in text and "value=name" in text
    assert not elsewhere.exists() and not (qs / "inj").exists()


@needs_map
def test_the_per_condition_review_set_is_named_as_a_review_set(qs, cond):
    _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit_rsc"), condition=cond)
    review = qs / "audit_rsc" / "review_set_05pct_by_condition.csv"
    text = _refused("estimate", sample_csv=str(review))
    assert "beside the CSV. This CSV is a review set that assess wrote. A review set is not a sample." in text


@needs_map
def test_texts_read_as_sentences_and_say_each_thing_once(qs):
    """No zone: the package's note is the conclusion, once, with a capital. Two identical maps with labels: no
    'undefined' shares."""
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "once"), budget=300, design="random")
    _label(out["files"]["sample_csv"], qs)
    res = _ok("certify", sample_csv=out["files"]["sample_csv"], alpha=0.01)
    assert res["conclusion"] == ("No zone was certified at alpha 1%, delta 10%, from 300 labels: the smallest testable "
                                 "zone (80% of the map) held 243 labels with 1 wrong.")
    assert "smallest testable zone" not in res["limits"]

    res = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "scores.tif"), out_dir=str(qs / "same"),
              labels=str(qs / "truth.tif"))
    assert "undefined" not in res["conclusion"]
    assert "The two maps give the same class in every window" in res["conclusion"]


@needs_map
def test_a_condition_sample_reads_as_one(qs, cond):
    """estimate's notes once each, its outside note as a sentence; certify's next names the labels a condition
    needs at delta split over the conditions, and a sample with the same layer."""
    layer = np.zeros((256, 256), np.int16)
    layer[104:120, 220:236] = 1                    # 16 windows, labelled in full, 9 of them wrong
    small = _layer(qs / "condition_small.tif", layer)
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "small"), budget=300, condition=small,
              condition_names=["0=clear", "1=hazy"])
    _label(out["files"]["sample_csv"], qs)
    res = _ok("estimate", sample_csv=out["files"]["sample_csv"])
    assert json.load(open(res["files"]["estimate"]))["outside_condition_intervals"] == ["hazy"]
    assert ("windows. Each condition's interval is its own 95% statement; the intervals do not hold jointly. The "
            "whole-map rate is 5.5%. It lies below the interval of hazy") in res["conclusion"]
    # the outside note is said once, in the conclusion, not again in limits
    assert "It lies below the interval of hazy" not in res["limits"]
    assert res["limits"].count(mcp_server._mcp_words(est.CONDITION_WHOLE_MAP)) == 1     # in the server's words
    assert 'design="random" can give a narrower interval' in res["limits"] and "--design" not in res["limits"]
    assert "for each input condition" in res["next"]

    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "by_cond_next"), budget=300, condition=cond,
              condition_names=["0=clear", "1=cloudy"])
    _label(out["files"]["sample_csv"], qs)
    res = _ok("certify", sample_csv=out["files"]["sample_csv"], alpha=0.01)
    need_full, need_split = est.min_labels_to_certify(0.01), est.min_labels_to_certify(0.01, 0.1 / 2)
    assert (need_full, need_split) == (230, 299)
    assert f"at least {need_full} labels to be tested" in res["next"] and f"at least {need_split}" in res["next"]
    assert "the same condition layer" in res["next"]
    assert "Any zone needs" not in res["next"]
    # nothing certified: no rate to hold as a group, and lowering delta would only need more labels
    assert res["summary"]["certified_share_of_map"] is None
    assert "not for each window" not in res["conclusion"] and "can be set lower" not in res["conclusion"]
    res = _ok("certify", sample_csv=out["files"]["sample_csv"], alpha=0.05, out_dir=str(qs / "by_cond_05"))
    assert res["summary"]["certified_share_of_map"] is not None
    assert "not for each window" in res["conclusion"] and "can be set lower" in res["conclusion"]


@needs_map
def test_compare_returns_its_groups(qs, cond):
    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff_g"), groups=cond)
    written = json.load(open(out["files"]["comparison"]))
    assert out["summary"]["per_group"] == written["per_group"]
    assert "Per group: group 0, 332 of 2048 windows differ (16.2%); group 1, 175 of 2048 (8.5%)." in out["conclusion"]
    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff_gl"), groups=cond,
              labels=str(qs / "truth.tif"))
    written = json.load(open(out["files"]["comparison"]))
    assert out["summary"]["over_groups"] == written["graded"]["over_groups"]
    assert "b is right more often than a in 1 of the 2 groups, a more often in 0, equally often in 1" in out["conclusion"]


# ----------------------------------------------------------------------------- the server, as an agent starts it
@needs_map
def test_oe_inferencex_mcp_serves_over_stdio(qs):
    """`oe-inferencex mcp` in its own process: the handshake, the tools and one call over stdio, so nothing the
    commands print reaches the protocol's channel."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "oe_inferencex.cli", "mcp"], cwd=str(qs))

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                init = await client.initialize()
                tools = (await client.list_tools()).tools
                res = await client.call_tool("assess", {"scores": "scores.tif", "out_dir": "audit_stdio"})
                return init, tools, res

    init, tools, res = anyio.run(go)
    assert init.instructions == mcp_server.INSTRUCTIONS and {t.name for t in tools} == TOOLS
    assert not res.isError
    out = json.loads(res.content[0].text)
    assert out["summary"]["review_sets"]["0.05"] == 205
    assert out["files"]["review_set_0.05"] == str(qs / "audit_stdio" / "review_set_05pct.csv")


# ----------------------------------------------------------------------------- what the agent test of 2 October found
def test_the_server_writes_its_own_parameter_names():
    """The package's texts name command-line options; through the server each is the parameter the agent sets."""
    w = mcp_server._mcp_words
    assert w("draw the sample with --condition") == "draw the sample with condition"
    assert w("--design random can give a narrower interval") == 'design="random" can give a narrower interval'
    assert w("pass the class probabilities without --logits") == "pass the class probabilities with logits=false"
    assert w("pass a smaller --patch; --labels-date is needed, --per-class too") == (
        "pass a smaller patch; labels_date is needed, per_class too")
    assert w("--rule=bonferroni, --out") == 'rule="bonferroni", out_dir'
    assert w("at 1 - 0.05/L, --unknown-flag stays") == "at 1 - 0.05/L, --unknown-flag stays"


@needs_mcp
def test_a_one_row_logit_map_through_the_whole_flow(tmp_path):
    """The agent test's layout: points in one row of one-pixel windows, as logits of four classes, with a two-value
    condition layer. No text names a command-line option; the logit warning says what an MCP caller can do; the
    boundary share is a caveat, not a finding; x and y are said to be empty; the budget fits the map; estimate says
    when certify would certify nothing; certify says its rate is the zone's as a group and that delta can be lowered."""
    rs = np.random.RandomState(0)
    n = 120
    logits = rs.normal(0, 3, size=(4, 1, n)).astype(np.float32)
    np.save(tmp_path / "logits.npy", logits)
    region = (np.arange(n) >= 80).astype(np.int16)[None, :]
    np.save(tmp_path / "region.npy", region)
    texts = []

    out = _ok("assess", scores=str(tmp_path / "logits.npy"), out_dir=str(tmp_path / "a"), logits=True, patch=1)
    texts.append(out)
    assert "pixel coordinates (a .npy has no georeferencing, so x and y are empty)" in out["conclusion"]
    assert "sit on a class boundary" not in out["conclusion"] and "one window high" in out["limits"]
    assert mcp_server.MCP_MARGIN_WARNING in out["limits"] and "form='top1'" not in out["limits"]
    assert "the gap between the two highest logits, which has no upper bound" in out["limits"]
    assert "The suspicion raster holds minus that score, so higher is more suspect" in out["limits"]
    assert f"a budget of up to {n}" in out["next"]

    other = logits + rs.normal(0, 3, size=logits.shape).astype(np.float32)
    np.save(tmp_path / "other.npy", other)
    out = _ok("compare", a=str(tmp_path / "logits.npy"), b=str(tmp_path / "other.npy"), out_dir=str(tmp_path / "c"),
              patch=1)
    texts.append(out)
    assert "times as often" not in out["conclusion"]              # a one-row grid: a caveat, not a finding
    assert "the window grid is one window high, so boundaries are counted along one line only" in out["limits"]
    assert "neither which map is better nor which is right" in out["conclusion"]

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        text = _refused("assess", scores="maps/missing.npy", out_dir=str(tmp_path / "x"), logits=True, patch=1)
    finally:
        os.chdir(cwd)
    assert f"(relative paths are read from {tmp_path}" in text

    out = _ok("sample", scores=str(tmp_path / "logits.npy"), out_dir=str(tmp_path / "s"), budget=40, logits=True,
              patch=1, condition=str(tmp_path / "region.npy"), condition_names=["0=north", "1=south"])
    texts.append(out)
    assert "No window is labelled yet, so nothing is known about the error rate" in out["conclusion"]
    csv_path = out["files"]["sample_csv"]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["wrong"] = int(i % 4 == 0)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    out = _ok("estimate", sample_csv=csv_path)
    texts.append(out)
    assert "the intervals do not hold jointly" in out["conclusion"]
    b1 = est.min_labels_to_certify(0.05)
    assert f"a condition needs at least {b1} labels and the most any holds is 20" in out["next"]

    out = _ok("sample", scores=str(tmp_path / "logits.npy"), out_dir=str(tmp_path / "r"), budget=n, logits=True,
              patch=1, design="random")
    csv_path = out["files"]["sample_csv"]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["wrong"] = 0
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = _ok("estimate", sample_csv=csv_path)
    texts.append(out)
    assert "a zone needs" not in out["next"]                      # 120 labels are enough at alpha 0.05
    small = _ok("sample", scores=str(tmp_path / "logits.npy"), out_dir=str(tmp_path / "r30"), budget=30, logits=True,
                patch=1, design="random")
    with open(small["files"]["sample_csv"], newline="") as f:
        few = list(csv.DictReader(f))
    for row in few:
        row["wrong"] = 0
    with open(small["files"]["sample_csv"], "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(few[0]))
        wr.writeheader()
        wr.writerows(few)
    out = _ok("estimate", sample_csv=small["files"]["sample_csv"])
    assert f"a zone needs at least {est.min_labels_to_certify(0.05)} labels and this sample has 30" in out["next"]

    out = _ok("certify", sample_csv=csv_path, alpha=0.05)
    texts.append(out)
    assert out["conclusion"].startswith("Taken together, the ") and "not for each window" in out["conclusion"]
    assert "delta, which can be set lower" in out["conclusion"]
    assert "the exact upper bound on this zone's error rate is" in out["conclusion"]

    for t in texts:
        for key in ("conclusion", "limits", "next"):
            assert not re.search(r"--[a-z]", t[key]), (key, t[key])


@needs_mcp
def test_a_one_column_map_and_a_path_with_dashes(tmp_path):
    """A grid one window wide is said to be wide; an output directory whose name holds an option-like `--patch` comes
    back unchanged in every text (review of 1.4.1: the translation turned run--patch8 into runpatch8)."""
    rs = np.random.RandomState(1)
    np.save(tmp_path / "col.npy", rs.normal(0, 3, size=(4, 90, 1)).astype(np.float32))
    np.save(tmp_path / "col2.npy", rs.normal(0, 3, size=(4, 90, 1)).astype(np.float32))
    out_dir = tmp_path / "run--patch8"
    out = _ok("assess", scores=str(tmp_path / "col.npy"), out_dir=str(out_dir), logits=True, patch=1)
    assert "one window wide" in out["limits"] and "one window high" not in out["limits"]
    assert out["files"]["review_set_0.05"] in out["conclusion"] and "run--patch8" in out["conclusion"]
    out = _ok("compare", a=str(tmp_path / "col.npy"), b=str(tmp_path / "col2.npy"), out_dir=str(out_dir / "c--out"),
              patch=1)
    assert "one window wide" in out["limits"] and "times as often" not in out["conclusion"]
    assert out["files"]["differing_windows"] in out["conclusion"]


@needs_map
def test_a_one_row_geotiff_gets_the_caveat(tmp_path):
    """The caveat reads the window grid from a GeoTIFF too, through rasterio."""
    import rasterio
    from rasterio.transform import from_origin
    rs = np.random.RandomState(2)
    for name in ("row.tif", "row2.tif"):
        p = rs.dirichlet(np.ones(3), size=(1, 80)).transpose(2, 0, 1).astype(np.float32)
        with rasterio.open(tmp_path / name, "w", driver="GTiff", height=1, width=80, count=3, dtype="float32",
                           crs="EPSG:32632", transform=from_origin(500000, 4000000, 10, 10)) as dst:
            dst.write(p)
    out = _ok("assess", scores=str(tmp_path / "row.tif"), out_dir=str(tmp_path / "a"), patch=1)
    assert "one window high" in out["limits"] and "sit on a class boundary" not in out["conclusion"]
    assert "pixel and map coordinates" in out["conclusion"]
    out = _ok("compare", a=str(tmp_path / "row.tif"), b=str(tmp_path / "row2.tif"), out_dir=str(tmp_path / "c"), patch=1)
    assert "one window high" in out["limits"] and "times as often" not in out["conclusion"]


@needs_map
def test_unjudged_windows_and_reviewer_error_through_the_server(qs):
    """Rows marked ? make the estimate a range, said in the conclusion; the reviewer's stated error rates widen the
    interval and replace the labels-assumed-right limit; certify counts ? as wrong and tests at alpha (1 - miss)."""
    out = _ok("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "unjudged"), budget=300, design="random")
    csv_path = out["files"]["sample_csv"]
    assert "or ? where the window cannot be judged" in out["next"]
    _label(csv_path, qs)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows[:10]:
        row["wrong"] = "?"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    res = _ok("estimate", sample_csv=csv_path)
    assert res["summary"]["estimate"] is None and res["summary"]["n_unjudged"] == 10
    lo, hi = res["summary"]["estimate_range"]
    assert f"The map's error rate is between {100 * lo:.1f}% and {100 * hi:.1f}%" in res["conclusion"]
    # ? alone: the windows are bounded, and the judged labels are still taken as right
    assert "bounds the windows that could not be judged both ways" in res["conclusion"]
    assert "which are assumed right" in res["conclusion"] and "user stated" not in res["conclusion"]
    assert "10 of the 300 labelled windows could not be judged" in res["limits"]
    assert "Labels are assumed right" in res["limits"]

    res = _ok("estimate", sample_csv=csv_path, reviewer_false_alarm=0.05, reviewer_miss=0.1,
              out_dir=str(qs / "unjudged_rev"))
    s = res["summary"]
    assert s["low"] == pytest.approx(max(0.0, (s["labels_interval"]["low"] - 0.05) / 0.95))
    assert s["high"] == pytest.approx(min(1.0, s["labels_interval"]["high"] / 0.9))
    assert "misses at most 0.1 of the truly wrong ones" in res["limits"]
    assert "Labels are assumed right" not in res["limits"] and "as often as the user stated" in res["conclusion"]
    assert "reviewer_miss must be at least 0" in _refused("estimate", sample_csv=csv_path, reviewer_miss=1.0)

    res = _ok("certify", sample_csv=csv_path, alpha=0.05)
    assert "10 window(s) that could not be judged (?) are counted as wrong" in res["limits"]
    assert "Labels are assumed right" in res["limits"]
    assert "reviewer_miss" not in json.dumps(res)

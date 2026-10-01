"""The MCP server (oe_inferencex.mcp_server) as an agent meets it, through the MCP SDK's own client.

The server runs in this process, on the SDK's in-memory streams, and once as `oe-inferencex mcp` over stdio. Each tool
runs on the README's quick-start map (examples/quickstart_map.py) and must give the numbers the README's commands
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
import sys

import numpy as np
import pytest

from oe_inferencex import cli, mcp_server
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
    """The quick-start map: scores.tif, other.tif and truth.tif, as the README writes them."""
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
    """The reviewer's step of the README, from truth.tif."""
    with contextlib.redirect_stdout(io.StringIO()):
        quickstart_map.label(str(path), truth_path=str(qs / "truth.tif"))


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
    assert [s.split(":")[0] for s in steps] == ["1. assess", "2. sample, label, estimate", "3. certify", "4. compare",
                                                "5. Per condition"]
    for rule in mcp_server.HARD_RULES:
        assert f"- {rule}" in text
    flat = _flat(" ".join(mcp_server.HARD_RULES))
    for said in ("A review set is not a sample", "certify needs a random sample", "compare cannot say which map is right",
                 "Labels are assumed right", "Ranking needs the scores, not only the class map"):
        assert said in flat, said
    assert "\u2014" not in text


def test_the_skill_holds_the_same_teaching():
    path = os.path.join(ROOT, "skills", "oe-inferencex", "SKILL.md")
    text = open(path, encoding="utf-8").read()
    head = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert head, "SKILL.md starts with a frontmatter block"
    fields = dict(line.split(": ", 1) for line in head.group(1).splitlines() if ": " in line)
    assert fields["name"] == "oe-inferencex" and len(fields["description"]) > 40
    body = _flat(text)
    for rule in mcp_server.HARD_RULES:
        assert _flat(rule) in body, rule
    for command in ("oe-inferencex assess", "oe-inferencex sample", "oe-inferencex estimate", "oe-inferencex certify",
                    "oe-inferencex compare", "--condition", "--design random"):
        assert command in body, command
    assert "\u2014" not in text


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


# ----------------------------------------------------------------------------- the server, in process
@needs_mcp
def test_the_server_lists_six_tools_each_described_by_its_card():
    async def work(client, init):
        return init, (await client.list_tools()).tools

    init, tools = _session(work)
    assert init.instructions == mcp_server.INSTRUCTIONS
    assert init.serverInfo.name == "oe-inferencex"
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.description == mcp_server.CARDS[t.name]
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
    assert f"Relative paths are read from {os.getcwd()}." in text


@needs_map
def test_assess_says_where_to_look_and_that_it_is_not_an_error_rate(qs, capsys):
    out = _ok("assess", scores=str(qs / "scores.tif"), out_dir=str(qs / "audit"))
    assert capsys.readouterr().out == ""                       # stdout is the protocol's channel on stdio
    s = out["summary"]
    # the README's line: 4096 windows of 4 px; review sets 1%: 41, 5%: 205, 10%: 410; boundary windows 40.0%
    assert s["n_windows"] == 4096 and s["review_sets"] == {"0.01": 41, "0.05": 205, "0.1": 410}
    assert s["boundary_window_fraction"] == pytest.approx(0.400390625)
    five = str(qs / "audit" / "review_set_05pct.csv")
    assert out["files"]["review_set_0.05"] == five
    assert "the review sets hold 41 (1%), 205 (5%), 410 (10%) windows" in out["conclusion"]
    assert f"The 5% review set is in {five}" in out["conclusion"]
    assert "40.0% of all windows sit on a class boundary" in out["conclusion"]
    written = json.load(open(qs / "audit" / "assessment.json"))
    for said in ("not how wrong the map is", "The review set is not a sample", "not the probability that the window "
                 "is wrong", "Errors the model is sure of come last", written["scope"], written["warnings"][0]):
        assert said in out["limits"], said
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
    # the README: error rate 7.0%, 95% interval 4.5% to 10.4% ... exact hypergeometric interval
    s = out["summary"]
    assert (s["estimate"], s["low"], s["high"]) == (0.07, 0.044921875, 0.103515625)
    assert out["conclusion"].startswith("The map's error rate is 7.0%, 95% interval 4.5% to 10.4%, from 300 labelled "
                                        "windows of 4096; exact hypergeometric interval")
    written = json.load(open(out["files"]["estimate"]))
    assert "Labels are assumed right" in out["limits"] and f"Note: {written['scope']}" in out["limits"]
    assert "certify with sample_csv=" in out["next"]

    out = _ok("certify", sample_csv=csv_path, alpha=0.05)
    s = out["summary"]
    assert (s["coverage"], s["n_zone"], s["n_population"]) == (0.9, 3686, 4096)
    assert out["conclusion"].startswith("The 90% most confident windows (3686 of 4096, confidence margin >= 0.6662) are "
                                        "wrong at most 5% of the time.")
    assert np.load(out["files"]["zone_mask"]).sum() == 3686
    assert "Outside the certified windows nothing is certified" in out["limits"] and f"Note: {est.PREFIX_NOTE}" in out["limits"]

    out = _ok("certify", sample_csv=csv_path, alpha=0.01, out_dir=str(qs / "strict"))
    assert out["summary"]["coverage"] is None and out["conclusion"].startswith("No zone was certified at alpha 1%")
    assert "(80% of the map) held 243 labels with 1 wrong" in out["conclusion"]   # as the README's test finds
    assert "zone_mask" not in out["files"] and "labels at this alpha" in out["next"]


@needs_map
def test_compare_does_not_say_which_map_is_right_without_labels(qs):
    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff"))
    assert out["summary"]["n_disagree"] == 507 and out["summary"]["n_windows"] == 4096
    assert out["conclusion"].startswith("507 of 4096 windows differ (12.38%).")
    assert "on a class boundary of map a 2.6 times as often" in out["conclusion"]
    assert mcp_server.HARD_RULES[2] in out["limits"]
    assert "Note: the dates the two maps describe were not given" in out["limits"]
    assert "compare again with labels" in out["next"]

    out = _ok("compare", a=str(qs / "scores.tif"), b=str(qs / "other.tif"), out_dir=str(qs / "diff_l"),
              labels=str(qs / "truth.tif"))
    ws = out["summary"]["which_side"]
    assert (ws["a_right"], ws["b_right"], ws["neither"]) == (228, 239, 40)
    assert "a is right on 45% and b on 47% of the 507 windows" in out["conclusion"]   # the README: a 45%, b 47%
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
    assert "missing.tif is not a file" in text
    text = _refused("sample", scores=str(qs / "scores.tif"), out_dir=str(qs / "x"), budget=300, design="condition")
    assert "--design condition needs --condition" in text


@needs_map
def test_each_input_condition_on_its_own(qs):
    import rasterio
    from rasterio.transform import from_origin
    layer = np.zeros((256, 256), np.int16)
    layer[:, 128:] = 1
    cond = qs / "condition.tif"
    with rasterio.open(cond, "w", driver="GTiff", height=256, width=256, count=1, dtype="int16", crs="EPSG:32632",
                       transform=from_origin(500000, 5000000, 10, 10)) as dst:
        dst.write(layer, 1)
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

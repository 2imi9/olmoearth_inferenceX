"""The worked quick start of docs/Usage.md, run as written on the test map examples/quickstart_map.py writes.

Every ```console block of that section is a command and the lines it printed. The page says those lines were
printed by the release; this test says the repository still prints them, so a change to a printed line or to the
test map shows up here before a reader finds it. (The quick start lived in the README until the README was cut
down on 4 October 2026.)"""
import importlib.util
import os
import re
import shlex

import pytest

pytest.importorskip("rasterio", reason="the quick start reads and writes GeoTIFFs")

from oe_inferencex import cli  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
quickstart_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quickstart_map)

with open(os.path.join(ROOT, "docs", "Usage.md"), encoding="utf-8") as _f:
    _USAGE = _f.read()
# the worked example: from its heading to the next section of the page
QUICKSTART = _USAGE[_USAGE.index("### A worked example"):_USAGE.index("\n## Inputs")]


def _console_blocks():
    """(command, printed lines) per ```console block of the worked example, in order."""
    blocks = []
    for body in re.findall(r"```console\n(.*?)\n```", QUICKSTART, flags=re.S):
        lines = body.split("\n")
        assert lines[0].startswith("$ oe-inferencex "), lines[0]
        blocks.append((lines[0][2:], lines[1:]))
    return blocks


def _run(command, capsys):
    try:
        cli.main(shlex.split(command)[1:])
    except SystemExit as e:                      # a refusal prints through SystemExit; show it as the output
        return str(e).split("\n")
    return capsys.readouterr().out.rstrip("\n").split("\n")


def test_the_console_blocks_are_what_the_commands_print(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    quickstart_map.write_maps()
    written = capsys.readouterr().out
    # the page's "Of its windows, 7.3% are wrong"
    assert "wrote scores.tif: 4 classes, 256 x 256 pixels; 301 of its 4096 windows are wrong (7.3%)" in written
    assert "Of its windows, 7.3% are wrong" in QUICKSTART

    blocks = _console_blocks()
    assert [c.split()[1] for c, _ in blocks] == ["assess", "sample", "estimate", "certify", "compare"]
    for command, expected in blocks:
        assert _run(command, capsys) == expected, command
        if command.split()[1] == "sample":       # the reviewer's step, between `sample` and `estimate`
            quickstart_map.label("to_label.csv")
            assert capsys.readouterr().out == "labelled 300 windows in to_label.csv: 21 wrong\n"

    # the prose after the blocks: --per-class works once the script has added reference_class, certify can
    # return nothing, and the default design is read by estimate and refused by certify
    assert any(line.startswith("  class 3: user's accuracy") for line in _run("oe-inferencex estimate to_label.csv --per-class", capsys))
    nothing = _run("oe-inferencex certify to_label.csv --alpha 0.01", capsys)[0]
    assert nothing.startswith("no zone certified") and "(80% of the map) held 243 labels with 1 wrong" in nothing
    assert "with `--alpha 0.01` the same labels gave `no zone certified`" in QUICKSTART

    _run("oe-inferencex sample scores.tif --budget 300 --out default.csv", capsys)
    quickstart_map.label("default.csv")
    capsys.readouterr()
    assert _run("oe-inferencex estimate default.csv", capsys)[0].startswith("error rate ")
    assert _run("oe-inferencex certify default.csv --alpha 0.05", capsys)[0].startswith("certify needs a random sample")
    assert "--labels truth.tif" in QUICKSTART
    assert "with labels: a right on" in _run("oe-inferencex compare scores.tif other.tif --labels truth.tif --out diff_l", capsys)[0]


def test_the_test_map_is_built_from_whole_numbers():
    """Probabilities are whole numbers of 1/4096 that sum to one, the truth has no evenly split window, and the
    map's class is the truth's except in the windows drawn as wrong: what makes the files the same on every machine."""
    import numpy as np

    truth = quickstart_map.truth_map()
    counts = quickstart_map.window_counts(truth)
    assert ((counts == counts.max(0)).sum(0) == 1).all()
    scores = quickstart_map.score_map(truth, 1)
    assert scores.dtype == np.float32 and scores.shape == (4, 256, 256)
    whole = scores.astype(np.float64) * quickstart_map.UNIT
    assert (whole == np.round(whole)).all() and (whole.sum(0) == quickstart_map.UNIT).all()
    assert scores.max() < 1 and scores.min() > 0
    assert (scores == quickstart_map.score_map(truth, 1)).all()
    wrong = quickstart_map.to_windows(scores.argmax(0)) != quickstart_map.to_windows(truth)
    assert int(wrong.sum()) == 301

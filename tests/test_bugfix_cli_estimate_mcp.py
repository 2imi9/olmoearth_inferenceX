"""Regression tests for the 2026-10-06 bug hunt in the command line, the estimators and the MCP server and decide, each
named by its consequence. The fixes landed together; these pin them."""
import csv
import json
import shlex
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import cli
from oe_inferencex import estimate as est


def _run(command, capsys=None):
    try:
        cli.main(shlex.split(command))
    except SystemExit as e:
        return "EXIT " + str(e)
    return capsys.readouterr().out if capsys else ""


def _classes(seed=0, H=64, W=64, C=3):
    rng = np.random.default_rng(seed)
    a = np.kron(rng.integers(0, C, (H // 4, W // 4)), np.ones((4, 4), int))   # one class per window: no even split
    b = a.copy()
    b[:16, :16] = (b[:16, :16] + 1) % C                              # 16 of 256 windows differ
    return a, b


def _prob(seed=0, H=32, W=32, C=3):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(C) * 0.5, size=(H, W)).transpose(2, 0, 1)
    return p


def test_a_one_band_npy_is_read_like_a_one_band_raster(tmp_path, monkeypatch, capsys):
    """A (1, H, W) .npy class map was argmaxed to class 0 everywhere, so two different maps 'never differed'."""
    a, b = _classes()
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    np.save(tmp_path / "a1.npy", a[None])
    np.save(tmp_path / "b1.npy", b[None])
    monkeypatch.chdir(tmp_path)
    flat = _run("compare a.npy b.npy --out d0", capsys)
    one = _run("compare a1.npy b1.npy --out d1", capsys)
    assert "16 of 256 windows differ" in flat and "16 of 256 windows differ" in one


def test_a_pixel_is_no_data_only_when_every_band_holds_the_value(tmp_path, monkeypatch, capsys):
    """With --nodata 0 a probability vector with one class at exactly 0 was dropped: the surest pixels vanished."""
    p = np.zeros((3, 32, 32))
    p[0], p[1] = 0.97, 0.03                                          # sure pixels, class 2 at exactly 0
    p[:, :8, :] = 0.0                                                # outside the scene: every band 0
    np.save(tmp_path / "p.npy", p)
    monkeypatch.chdir(tmp_path)
    _run("assess p.npy --nodata 0 --out a", capsys)
    assert json.load(open("a/assessment.json"))["n_windows"] == 64 - 16


def test_assess_never_writes_over_its_own_condition_layer(tmp_path, monkeypatch, capsys):
    p = _prob()
    np.save(tmp_path / "p.npy", p)
    (tmp_path / "out").mkdir()
    layer = np.zeros((32, 32), int)
    layer[:, 16:] = 1
    np.save(tmp_path / "out" / "condition.npy", layer)
    monkeypatch.chdir(tmp_path)
    got = _run("assess p.npy --out out --condition out/condition.npy", capsys)
    assert got.startswith("EXIT") and "would overwrite the input" in got
    assert np.array_equal(np.load(tmp_path / "out" / "condition.npy"), layer)


def test_a_zero_one_class_map_is_not_cut_at_the_threshold(tmp_path, monkeypatch, capsys):
    """compare cover.npy mask.npy --threshold 50 cut the 0/1 mask at 50 too, so every forest window 'differed'."""
    rng = np.random.default_rng(1)
    cover = rng.uniform(0, 100, (32, 32))
    np.save(tmp_path / "cover.npy", cover)
    np.save(tmp_path / "mask.npy", (cover >= 50).astype(np.uint8))
    monkeypatch.chdir(tmp_path)
    printed = _run("compare cover.npy mask.npy --threshold 50 --out d", capsys)
    assert " 0 of " in " " + printed and "windows differ (0.00%)" in printed
    assert any("read as a 0/1 class map, not cut at 50" in n for n in json.load(open("d/comparison.json"))["notes"])


def test_a_float_map_holding_zero_and_one_is_still_cut(tmp_path, monkeypatch, capsys):
    """The 0/1 mask rule took a height map in metres whose values were 0 and 1 for a class map (the review of
    2026-10-06): every pixel is below a 5 m cut, so the two height maps cannot differ."""
    rng = np.random.default_rng(4)
    np.save(tmp_path / "h_a.npy", rng.integers(0, 2, (32, 32)).astype(np.float32))
    np.save(tmp_path / "h_b.npy", rng.uniform(0, 2.1, (32, 32)).astype(np.float32))
    monkeypatch.chdir(tmp_path)
    printed = _run("compare h_a.npy h_b.npy --threshold 5 --out d", capsys)
    assert " 0 of " in " " + printed


def test_out_may_not_name_the_sample_or_its_sidecar(tmp_path, monkeypatch, capsys):
    """estimate --out s.json overwrote the design, and every later estimate died with a KeyError."""
    np.save(tmp_path / "p.npy", _prob())
    monkeypatch.chdir(tmp_path)
    _run("sample p.npy --budget 20 --design random --out s.csv", capsys)
    rows = list(csv.DictReader(open("s.csv")))
    for r in rows:
        r["wrong"] = "0"
    with open("s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for cmd in ("estimate s.csv --out s.json", "certify s.csv --alpha 0.3 --out s.json"):
        got = _run(cmd, capsys)
        assert got.startswith("EXIT") and "would overwrite the input" in got, cmd
    assert "indices" in json.load(open("s.json"))
    got = _run("certify s.csv --alpha 0.3 --out sub/zone.json", capsys)        # a missing directory is created
    assert "EXIT" not in got and (tmp_path / "sub" / "zone.json").exists()
    (tmp_path / "adir").mkdir()
    got = _run("certify s.csv --alpha 0.3 --out adir", capsys)
    assert got.startswith("EXIT") and "is a directory" in got


@pytest.mark.parametrize("command, said", [
    ("sample void.npy --budget 10 --out s.csv", "no valid pixels"),
    ("assess missing.npy --out a", "no such file"),
    ("assess p.npy --out afile.txt", "is a file"),
])
def test_reasonable_mistakes_are_named_refusals(tmp_path, monkeypatch, capsys, command, said):
    np.save(tmp_path / "p.npy", _prob())
    np.save(tmp_path / "void.npy", np.full((3, 16, 16), np.nan))
    (tmp_path / "afile.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    got = _run(command, capsys)
    assert got.startswith("EXIT") and said in got, got


def test_a_product_sample_refuses_another_class_map_for_per_class(tmp_path, monkeypatch, capsys):
    """Only the band was checked, so another class map on the same grid built the per-class table."""
    a, b = _classes(H=32, W=32)
    band = np.random.default_rng(2).uniform(0, 1, (32, 32))
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    np.save(tmp_path / "band.npy", band)
    monkeypatch.chdir(tmp_path)
    _run("sample a.npy --confidence band.npy --budget 40 --out s.csv", capsys)
    rows = list(csv.DictReader(open("s.csv")))
    fields = list(rows[0]) + ["reference_class"]
    for r in rows:
        r["reference_class"], r["wrong"] = r["map_class"], "0"
    with open("s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    got = _run("estimate s.csv --per-class --scores b.npy", capsys)
    if any(int(r["window_row"]) < 4 and int(r["window_col"]) < 4 for r in rows):
        assert got.startswith("EXIT") and "is not the one the CSV records" in got
    assert "EXIT" not in _run("estimate s.csv --per-class", capsys)


def test_overall_accuracy_under_the_condition_design_is_the_whole_map_interval():
    """estimate_per_class used the retired stratified Wilson form for overall accuracy under the condition design."""
    rng = np.random.default_rng(0)
    margin = rng.random(400)
    cond = (np.arange(400) >= 300).astype(int)
    s = est.sample_for_estimation(margin, 60, design="condition", valid=np.ones(400, bool), seed=1, condition=cond)
    klass = rng.integers(0, 3, 400)
    ref = klass.copy()
    flip = rng.random(400) < 0.2
    ref[flip] = (ref[flip] + 1) % 3
    idx = np.asarray(s["indices"])
    whole = est.estimate_error_rate(s, (klass[idx] != ref[idx]).astype(float))
    pc = est.estimate_per_class(s, ref[idx], klass)
    oa = pc["overall_accuracy"]
    assert oa["low"] == pytest.approx(1 - whole["high"]) and oa["high"] == pytest.approx(1 - whole["low"])


@pytest.mark.parametrize("kind", ["miss", "false_alarm"])
def test_reviewer_bounds_hold_a_truth_on_an_end(kind):
    """The widened ends were mapped in floats and landed one float inside a truth that sits on them."""
    if kind == "miss":
        truth = np.array([1, 1, 1, 1, 0])
        labels = truth.copy()
        labels[0] = 0
        s = est.sample_for_estimation(np.linspace(0, 1, 5), 5, design="random")
        r = est.estimate_error_rate(s, labels[s["indices"]], reviewer_miss=0.25)
    else:
        truth = np.array([1, 1, 1, 1, 0, 0])
        labels = truth.copy()
        labels[4] = 1
        s = est.sample_for_estimation(np.linspace(0, 1, 6), 6, design="random")
        r = est.estimate_error_rate(s, labels[s["indices"]], reviewer_false_alarm=0.5)
    assert r["low"] <= truth.mean() <= r["high"]


def test_a_level_near_delta_is_decided_exactly_on_a_large_zone():
    """On a 1.29M-window zone the float tail missed delta by more than the fixed 1e-9 window, so a level the exact test
    rejects was certified."""
    N, B, K = 1_293_697, 233, 7
    margin = np.random.default_rng(3).random(N)
    for seed in range(50):
        idx = np.random.default_rng(seed).choice(N, B, replace=False)
        wrong = np.zeros(B)
        wrong[np.argsort(margin[idx])[:K]] = 1
        r = est.certify_zone(margin, idx, wrong, alpha=0.05, delta=0.1)
        if all(lv["accepted"] for lv in r["levels"][:-1]):
            break
    last = r["levels"][-1]
    exact = est.zone_pvalue_exact(last["n_wrong_inside"], last["n_labelled_inside"], last["n_zone"], 0.05)
    assert last["accepted"] == (exact <= Fraction(1, 10))


def test_decide_keeps_the_notes_on_every_answer():
    """Per-class answers and an undetermined trusted_share_at_least dropped the population note and the labels limit."""
    from oe_inferencex.decide import decide
    note = "The population is the pixels whose confidence lies in [1, 100]: 9 of 90 classified pixels were left out."
    row = {"map_share": 0.5, "n_labelled_reference_class": 10,
           "user_accuracy": {"estimate": 0.9, "low": 0.8, "high": 0.95},
           "producer_accuracy": {"estimate": 0.9, "low": 0.8, "high": 0.95}}
    r = {"design": "random", "n_labelled": 50, "n_population": 90, "nominal_coverage": 0.95, "estimate": 0.1,
         "low": 0.05, "high": 0.2, "method": "exact hypergeometric interval", "per_class": {"1": row, "2": row},
         "population_note": note}
    a = decide(r, ["user_accuracy_above=0.7"])["answers"]["user_accuracy_above=0.7"]
    assert note in a["because"]
    z = {"rule": "prefix", "alpha": 0.05, "delta": 0.1, "n_population": 90, "n_labelled": 50, "coverage": None,
         "n_zone": None, "population_note": note, "bounds_note": "2 windows marked ? count as wrong"}
    a = decide(z, ["trusted_share_at_least=0.5"])["answers"]["trusted_share_at_least=0.5"]
    assert a["answer"] == "undetermined" and note in a["because"] and "? count as wrong" in a["because"]
    assert "assumed right" in a["because"]

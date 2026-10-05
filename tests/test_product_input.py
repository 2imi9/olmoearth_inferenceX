"""A published product as input: a class map with its per-pixel confidence band (--confidence), as LCMAP ships lcpri
with lcpconf. Checked against the scores route it must agree with, through the whole labelled route, and at each
refusal. Built from .npy arrays, so it needs no rasterio."""
import csv
import json
import shlex

import numpy as np
import pytest

from oe_inferencex import cli


def _maps(tmp_path, seed=0, H=64, W=64, C=4):
    """Per-class probabilities, the class map and top probability they imply, and a truth that differs from the map
    more often where the map is less sure."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, (C, H, W))
    logits[:, :, : W // 2] *= 2.5                                   # a sure half and an unsure half
    p = np.exp(logits - logits.max(0))
    p /= p.sum(0)
    classes, top = p.argmax(0), p.max(0)
    flip = rng.random((H, W)) < (1 - top) * 0.6
    truth = np.where(flip, (classes + 1 + rng.integers(0, C - 1, (H, W))) % C, classes)
    np.save(tmp_path / "scores.npy", p)
    np.save(tmp_path / "classes.npy", classes.astype(np.int16))
    np.save(tmp_path / "band.npy", top.astype(np.float64))
    np.save(tmp_path / "truth.npy", truth)
    return p, classes, top, truth


def _run(command, capsys):
    try:
        cli.main(shlex.split(command))
    except SystemExit as e:
        return "EXIT " + str(e)
    return capsys.readouterr().out


def _label(path, truth, patch=4, shift=0):
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0]) + ([] if "reference_class" in rows[0] else ["reference_class"])
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        ref = int(np.bincount(truth[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch].ravel()).argmax()) + shift
        r["reference_class"], r["wrong"] = ref, int(ref != int(r["map_class"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_a_class_map_with_its_top_probability_ranks_as_the_scores_do(tmp_path, monkeypatch, capsys):
    """The band is the top probability and the class map the argmax: every window's confidence and class, and so the
    review sets, are the scores route's."""
    _maps(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run("assess scores.npy --out a0", capsys)
    printed = _run("assess classes.npy --confidence band.npy --out a1", capsys)
    assert "the whole confidence band is read as confidence" in printed            # no range given: said
    for b in ("01", "05", "10"):
        r0 = [(r["window_row"], r["window_col"], r["confidence"]) for r in csv.DictReader(open(f"a0/review_set_{b}pct.csv"))]
        r1 = [(r["window_row"], r["window_col"], r["confidence"]) for r in csv.DictReader(open(f"a1/review_set_{b}pct.csv"))]
        assert r0 == r1, b
    s1 = json.load(open("a1/assessment.json"))
    assert s1["inputs"]["confidence"].endswith("band.npy") and s1["inputs"]["confidence_range"] is None


def test_codes_outside_the_range_are_left_out_and_counted(tmp_path, monkeypatch, capsys):
    """LCMAP writes provenance codes into lcpconf; read raw they would rank first. A range leaves them out."""
    _, _, top, _ = _maps(tmp_path)
    band = top.copy()
    band[:8, :8] = 200.0                                            # four 4 x 4 windows of codes
    np.save(tmp_path / "coded.npy", band)
    monkeypatch.chdir(tmp_path)
    printed = _run("assess classes.npy --confidence coded.npy --confidence-range 0 1 --out a", capsys)
    assert "64 of 4096 pixels (1.6%) have a confidence outside [0, 1] and are left out as no-data" in printed
    s = json.load(open("a/assessment.json"))
    assert s["n_windows"] == 256 - 4
    raw = _run("assess classes.npy --confidence coded.npy --out raw", capsys)
    assert "EXIT" not in raw and json.load(open("raw/assessment.json"))["n_windows"] == 256


def test_the_labelled_route_runs_on_a_product(tmp_path, monkeypatch, capsys):
    """sample defaults to the random design; estimate, per-class and certify read both layers back; certify's zone is
    the one certify_zone gives on the band's window confidences directly."""
    from oe_inferencex import estimate as est
    from oe_inferencex.assess import assess_classmap
    _, classes, top, truth = _maps(tmp_path)
    monkeypatch.chdir(tmp_path)
    printed = _run("sample classes.npy --confidence band.npy --confidence-range 0 1 --budget 120 --out s.csv", capsys)
    assert "(random design)" in printed and "the default confidence design does not apply" in printed
    side = json.load(open("s.json"))
    assert side["design"] == "random" and side["confidence"].endswith("band.npy") and side["confidence_range"] == [0.0, 1.0]
    _label("s.csv", truth)
    printed = _run("estimate s.csv --per-class", capsys)
    assert printed.startswith("error rate ") and "class 3:" in printed
    printed = _run("certify s.csv --alpha 0.2", capsys)
    assert "EXIT" not in printed
    zone = json.load(open("s_zone.json"))
    out = assess_classmap(classes, top, 4, patch=4)
    rows = list(csv.DictReader(open("s.csv")))
    idx = np.array([int(r["index"]) for r in rows])
    direct = est.certify_zone(out["arrays"]["confidence"], idx, np.array([float(r["wrong"]) for r in rows]), 0.2)
    assert zone["coverage"] == direct["coverage"] and zone["n_zone"] == direct["n_zone"]
    if zone["coverage"] is not None:
        assert ", confidence >= " in printed and "confidence margin" not in printed


def test_the_left_out_pixels_are_named_wherever_a_number_is_given(tmp_path, monkeypatch, capsys):
    """With a range, the population is the pixels inside it; estimate, certify and decide say what was left out."""
    from oe_inferencex.decide import decide
    _, _, top, truth = _maps(tmp_path)
    band = top.copy()
    band[:16, :16] = 160.0                                           # 256 coded pixels, as LCMAP's provenance codes
    np.save(tmp_path / "coded.npy", band)
    monkeypatch.chdir(tmp_path)
    _run("sample classes.npy --confidence coded.npy --confidence-range 0 1 --budget 120 --out s.csv", capsys)
    side = json.load(open("s.json"))
    assert side["product_population"] == {"classified_pixels": 4096, "outside_range": 256}
    _label("s.csv", truth)
    said = "256 of 4096 classified pixels (6.2%) were left out"
    assert said in _run("estimate s.csv", capsys) and said in json.load(open("s_estimate.json"))["population_note"]
    assert said in _run("certify s.csv --alpha 0.3", capsys) and said in json.load(open("s_zone.json"))["population_note"]
    assert said in decide("s_estimate.json", ["error_rate_below=0.5"])["answers"]["error_rate_below=0.5"]["because"]
    assert said in decide("s_zone.json", ["trusted_share"])["answers"]["trusted_share"]["because"]


def test_a_reviewer_may_name_a_class_the_product_never_predicts(tmp_path, monkeypatch, capsys):
    _, classes, _, truth = _maps(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run("sample classes.npy --confidence band.npy --budget 60 --out s.csv", capsys)
    _label("s.csv", truth)
    rows = list(csv.DictReader(open("s.csv")))
    rows[0]["reference_class"], rows[0]["wrong"] = "7", "1"
    with open("s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    printed = _run("estimate s.csv --per-class", capsys)
    assert "EXIT" not in printed and "class 7:" in printed


def test_a_confidence_rounded_to_half_a_unit_is_the_same_map(tmp_path, monkeypatch, capsys):
    """An integer band averaged over 16 pixels gives sixteenths; written to three decimals some land exactly half a unit
    off, which float noise must not turn into a refusal."""
    rng = np.random.default_rng(3)
    classes = rng.integers(0, 3, (64, 64)).astype(np.int16)
    band = rng.integers(1, 101, (64, 64)).astype(np.float64)
    np.save(tmp_path / "c.npy", classes)
    np.save(tmp_path / "b.npy", band)
    monkeypatch.chdir(tmp_path)
    _run("sample c.npy --confidence b.npy --budget 120 --out s.csv", capsys)
    rows = list(csv.DictReader(open("s.csv")))
    for r in rows:
        r["confidence"], r["wrong"] = f"{float(r['confidence']):.3f}", "0"
    with open("s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert "EXIT" not in _run("certify s.csv --alpha 0.2", capsys)


def test_a_band_named_for_a_scores_sample_is_refused(tmp_path, monkeypatch, capsys):
    _, _, _, truth = _maps(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run("sample scores.npy --design random --budget 60 --out s.csv", capsys)
    _label("s.csv", truth)
    got = _run("certify s.csv --alpha 0.2 --confidence band.npy", capsys)
    assert got.startswith("EXIT") and "this sample was drawn on scores" in got


def test_class_ids_from_one_print_no_empty_class(tmp_path, monkeypatch, capsys):
    """A product's ids can start at 1, as LCMAP's do; id 0 is then neither in the map nor in the labels."""
    _, classes, _, truth = _maps(tmp_path)
    np.save(tmp_path / "classes1.npy", (classes + 1).astype(np.int16))
    monkeypatch.chdir(tmp_path)
    _run("sample classes1.npy --confidence band.npy --budget 120 --out p.csv", capsys)
    _label("p.csv", truth, shift=1)
    printed = _run("estimate p.csv --per-class", capsys)
    assert "class 0:" not in printed and "class 1:" in printed and "class 4:" in printed


@pytest.mark.parametrize("command, said", [
    ("assess classes.npy --confidence-range 0 1 --out x", "--confidence-range needs --confidence"),
    ("assess big.npy --confidence band.npy --out x", "ids above 255 are refused"),
    ("assess classes.npy --confidence band.npy --logits --out x", "--logits reads per-class scores"),
    ("assess scores.npy --confidence band.npy --out x", "one band of class ids"),
    ("assess band.npy --confidence band.npy --out x", "not whole numbers"),
    ("assess classes.npy --confidence small.npy --out x", "the confidence band has shape (32, 32)"),
    ("assess classes.npy --confidence band.npy --confidence-range 1 0 --out x", "LOW below HIGH"),
    ("assess classes.npy --confidence band.npy --confidence-range 2 3 --out x", "no pixel has both a class and a confidence"),
    ("sample classes.npy --confidence band.npy --design confidence --budget 50 --out s.csv",
     "the confidence design allocates labels from the model's top-1 probability"),
    ("sample classes.npy --confidence band.npy --other classes.npy --budget 50 --out s.csv", "leave out --confidence"),
])
def test_refusals(tmp_path, monkeypatch, capsys, command, said):
    _, classes, _, _ = _maps(tmp_path)
    np.save(tmp_path / "small.npy", np.ones((32, 32)))
    big = classes.copy()
    big[0, 0] = 3000
    np.save(tmp_path / "big.npy", big)
    monkeypatch.chdir(tmp_path)
    got = _run(command, capsys)
    assert got.startswith("EXIT") and said in got, got


def test_a_moved_band_is_named_and_can_be_passed_again(tmp_path, monkeypatch, capsys):
    _, _, _, truth = _maps(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run("sample classes.npy --confidence band.npy --budget 120 --out s.csv", capsys)
    _label("s.csv", truth)
    (tmp_path / "band.npy").rename(tmp_path / "moved.npy")
    got = _run("certify s.csv --alpha 0.2", capsys)
    assert got.startswith("EXIT") and "pass --confidence with the band" in got
    assert "EXIT" not in _run("certify s.csv --alpha 0.2 --confidence moved.npy", capsys)

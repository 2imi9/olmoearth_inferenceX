"""oe_inferencex.scale and `oe-inferencex scale`: the same map at several window sizes (1.15.0).

Survival is checked by hand on maps small enough to count, the per-size error rate against assess's own, the two
input routes (per-class scores, a class map with its confidence band) against each other, and the command line's
refusals, its window sizes in metres and its JSON."""
import importlib.util
import json
import os

import numpy as np
import pytest

from oe_inferencex import scale as sc
from oe_inferencex.assess import assess_prediction
from oe_inferencex.cli import main


def _road(width=1):
    """A 64 x 64 map of two large classes crossed by a road of class 2 `width` pixels wide, as probabilities."""
    truth = np.zeros((64, 64), int)
    truth[:, 32:] = 1
    truth[20:20 + width, :] = 2
    truth[:, 10:10 + width] = 2
    rng = np.random.default_rng(0)
    p = np.full((3, 64, 64), 0.1)
    for c in range(3):
        p[c][truth == c] = 0.8
    p += rng.random(p.shape) * 0.05
    return p / p.sum(0, keepdims=True), truth


def test_a_road_one_pixel_wide_is_lost_from_4_px_and_the_error_rate_does_not_show_it():
    p, truth = _road()
    r = sc.scale_profile(scores=p, reference=truth)
    road = r["per_class"][2]
    assert road["survival"][1] == 1.0 and road["survival"][4] == 0.0 and road["first_below_half"] == 4
    assert road["in_windows_of_another_class"][4] == 1.0 and road["in_no_valid_window"][4] == 0.0
    assert all(r["per_class"][c]["survival"][16] > 0.95 for c in (0, 1))
    # the reference pools the road away as well: every size grades at 0% error, and only survival shows the loss
    assert all(r["per_patch"][q]["against_reference"]["error_rate"] == 0.0 for q in r["patches"])
    # with no error, the share of the errors a review captures is undefined, not 0
    assert all(r["per_patch"][q]["against_reference"]["errors_in_5pct_review"] is None for q in r["patches"])
    assert 2 not in r["per_patch"][4]["against_reference"]["per_class"]
    assert sc.UNSEEN_NOTE in r["notes"] and sc.CENSUS_NOTE in r["notes"]


def test_a_map_that_draws_the_road_thinner_than_the_reference_is_graded_wrong_there():
    """When the reference keeps the class at a size and the map loses it, the error rate does show it."""
    p, _ = _road(width=1)
    _, thick = _road(width=4)
    r = sc.scale_profile(scores=p, reference=thick)
    row = r["per_patch"][4]["against_reference"]
    assert row["per_class"][2]["windows_in_reference"] > 0 and row["per_class"][2]["producer_accuracy"] == 0.0
    assert row["error_rate"] > 0


def test_survival_counted_by_hand():
    """4 x 4 map, 2 px windows. Class 1 holds one pixel of the top-left window (outvoted), two of the top-right (a
    tie, broken by the more confident pixels: here class 1's) and all of the bottom-left; so of its 7 pixels, 6 sit in
    windows called class 1."""
    hard = np.array([[1, 0, 1, 1],
                     [0, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0]])
    conf = np.where(hard == 1, 0.9, 0.6)
    r = sc.scale_profile(hard=hard, confidence=conf, n_classes=2, patches=(1, 2))
    assert r["per_class"][1]["pixels"] == 7
    assert r["per_class"][1]["survival"][1] == 1.0
    assert r["per_class"][1]["survival"][2] == pytest.approx(6 / 7)
    assert r["per_class"][0]["survival"][2] == pytest.approx(7 / 9)


def test_no_data_pixels_count_for_no_class():
    hard = np.zeros((8, 8), int)
    hard[0, :] = 1
    conf = np.full((8, 8), 0.8)
    mask = np.zeros((8, 8), bool)
    mask[0, :4] = True
    r = sc.scale_profile(hard=hard, confidence=conf, n_classes=2, patches=(1,), nodata_mask=mask)
    assert r["per_class"][1]["pixels"] == 4


def test_each_size_matches_assess_and_the_two_input_routes_agree():
    p, _ = _road()
    _, truth = _road(width=4)                     # a reference the map is wrong against, so the capture is defined
    r = sc.scale_profile(scores=p, reference=truth, patches=(2, 4, 8))
    for q in (2, 4, 8):
        a = assess_prediction(p, is_logit=False, patch=q, reference=truth, budgets=(0.05,))
        row, ref = r["per_patch"][q], a["against_reference"]
        assert row["n_windows"] == a["n_windows"] and row["edge_window_share"] == a["boundary_window_fraction"]
        assert row["confidence_quantiles"] == a["confidence_quantiles"]
        assert row["against_reference"]["error_rate"] == ref["error_rate"] > 0
        assert row["against_reference"]["errors_in_5pct_review"] == \
            ref["error_capture_at_budget"][0.05]["errors_captured_fraction"]
        # every scored window is in exactly one class's reference count
        assert sum(e["windows_in_reference"] for e in row["against_reference"]["per_class"].values()) == \
            ref["n_windows_scored"]
    # the class map with its top-probability band ranks and pools as the scores do, reference included
    prod = sc.scale_profile(hard=p.argmax(0), confidence=p.max(0), n_classes=3, reference=truth, patches=(2, 4, 8))
    assert prod["per_class"] == r["per_class"]
    assert all(prod["per_patch"][q]["against_reference"] == r["per_patch"][q]["against_reference"] for q in (2, 4, 8))


def _window_rows(hard, ref, q):
    """Each class's window accuracy by plain loops: a window's class is its plurality (with every confidence equal, a
    tie goes to the lowest class, as in assess), and a reference window with a tie or under half labelled is
    unscored."""
    H, W = hard.shape
    rows = {}
    for i in range(0, H // q * q, q):
        for j in range(0, W // q * q, q):
            m = np.bincount(hard[i:i + q, j:j + q].ravel(), minlength=4)
            rr = ref[i:i + q, j:j + q].ravel()
            rr = rr[rr >= 0]
            if rr.size * 2 < q * q:
                continue
            rc = np.bincount(rr, minlength=4)
            if (rc == rc.max()).sum() > 1:
                continue
            rows.setdefault("pairs", []).append((int(m.argmax()), int(rc.argmax())))
    out = {}
    for c in range(4):
        in_ref = [mp for mp, rf in rows["pairs"] if rf == c]
        in_map = [rf for mp, rf in rows["pairs"] if mp == c]
        if in_ref or in_map:
            out[c] = {"windows_in_reference": len(in_ref), "windows_in_map": len(in_map),
                      "producer_accuracy": float(np.mean([x == c for x in in_ref])) if in_ref else None,
                      "user_accuracy": float(np.mean([x == c for x in in_map])) if in_map else None}
    return out


def test_per_class_window_accuracy_by_plain_loops():
    """3 px windows on a 12 x 12 map; the reference differs from it in places, with a tied window and an unlabelled
    one."""
    rng = np.random.default_rng(5)
    hard = rng.integers(0, 3, (12, 12))
    ref = np.where(rng.random((12, 12)) < 0.3, rng.integers(0, 4, (12, 12)), hard)
    ref[0:3, 0:3] = -1                                                  # an unlabelled window
    ref[3:6, 0:3] = np.array([[0, 0, 1], [1, 2, 2], [3, 3, -1]])         # 2-2-2-2: a tie, unscored
    conf = np.full((12, 12), 0.7)
    r = sc.scale_profile(hard=hard, confidence=conf, n_classes=3, patches=(3,), reference=ref)
    got = r["per_patch"][3]["against_reference"]["per_class"]
    assert got == _window_rows(hard, ref, 3)


def test_window_sizes_are_checked():
    p, _ = _road()
    r = sc.scale_profile(scores=p, patches=(4, 2, 4, 128))
    assert r["patches"] == [2, 4] and r["skipped"] == [128] and any("128" in n for n in r["notes"])
    with pytest.raises(ValueError, match="at least 1 px"):
        sc.scale_profile(scores=p, patches=(0, 2))
    with pytest.raises(ValueError, match="larger than the map"):
        sc.scale_profile(scores=p, patches=(100,))
    with pytest.raises(ValueError, match="either scores or a class map"):
        sc.scale_profile()


def test_the_command_line_writes_strict_json_and_refuses_by_name(tmp_path, capsys):
    p, truth = _road()
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "t.npy", truth)
    out = tmp_path / "prof.json"
    assert main(["scale", str(tmp_path / "p.npy"), "--reference", str(tmp_path / "t.npy"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "class 2 (127 px): 1 px 1.00, 2 px 0.54, 4 px 0.00" in printed
    assert ("first below half at 4 px, where 100% of its pixels sit in windows called another class and 0% in no valid "
            "window") in printed
    assert "  4 px: 256, 12.5%, 0.768, 0.0%, undefined (no error)" in printed
    assert "producer's accuracy against the reference" in printed

    def no_nan(x):
        raise AssertionError(f"not strict JSON: {x}")
    d = json.loads(out.read_text(), parse_constant=no_nan)
    assert list(d)[-1] == "written_by" and d["written_by"]["command"] == "scale"
    assert d["per_class"]["2"]["first_below_half"] == 4
    with pytest.raises(SystemExit, match="scale: window sizes must be at least 1 px"):
        main(["scale", str(tmp_path / "p.npy"), "--patches", "0"])
    with pytest.raises(SystemExit, match="has shape"):
        np.save(tmp_path / "small.npy", truth[:32])
        main(["scale", str(tmp_path / "p.npy"), "--reference", str(tmp_path / "small.npy")])


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="needs rasterio")
def test_a_geotiff_gives_window_sizes_in_metres(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "quickstart_map", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples",
                                       "quickstart_map.py"))
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        qs.write_maps()
    finally:
        os.chdir(cwd)
    capsys.readouterr()
    assert main(["scale", str(tmp_path / "scores.tif"), "--patches", "1", "4", "--reference",
                 str(tmp_path / "truth.tif")]) == 0
    printed = capsys.readouterr().out
    assert "  1 px (10 m):" in printed
    assert "  4 px (40 m):" in printed


def test_a_window_mostly_of_no_data_holds_no_surviving_pixel():
    """One class-1 pixel whose 2 px window is otherwise no-data: at 1 px it survives, at 2 px it sits in no valid
    window."""
    hard = np.zeros((4, 4), int)
    hard[0, 0] = 1
    mask = np.zeros((4, 4), bool)
    mask[0, 1] = mask[1, 0] = mask[1, 1] = True
    r = sc.scale_profile(hard=hard, confidence=np.full((4, 4), 0.8), n_classes=2, patches=(1, 2), nodata_mask=mask)
    assert r["per_class"][1]["survival"] == {1: 1.0, 2: 0.0} and r["per_class"][1]["first_below_half"] == 2


def test_a_class_in_the_ragged_edge_survives_nowhere_once_the_grid_leaves_it_out():
    hard = np.zeros((18, 18), int)
    hard[:, 9:16] = 1
    hard[:, 16:] = 2                                     # two columns no window of 4 px or more covers
    r = sc.scale_profile(hard=hard, confidence=np.full((18, 18), 0.8), n_classes=3)
    assert r["per_class"][2]["survival"][2] == 1.0 and r["per_class"][2]["survival"][4] == 0.0
    assert r["per_class"][2]["first_below_half"] == 4 and r["per_patch"][4]["pixels_outside_window_grid"] > 0
    edge = r["per_class"][2]
    assert edge["in_no_valid_window"][4] == 1.0 and edge["in_windows_of_another_class"][4] == 0
    assert any("do not fill" in w["warning"] for w in r["warnings"])


def test_a_size_with_no_valid_window_is_kept_not_fatal(tmp_path, capsys):
    """A strip of data seven pixels wide: no 16 px window is at least half valid; the other sizes still report."""
    p = np.full((64, 64), np.nan)
    p[:, :7] = np.random.default_rng(1).random((64, 7))
    r = sc.scale_profile(scores=p)
    assert r["patches"] == [1, 2, 4, 8, 16] and r["per_patch"][16]["n_windows"] == 0
    assert all(e["survival"][16] == 0.0 and e["in_no_valid_window"][16] == 1.0 for e in r["per_class"].values())
    assert sc.NO_WINDOW_NOTE.format(p=16) in r["notes"]
    np.save(tmp_path / "strip.npy", p)
    assert main(["scale", str(tmp_path / "strip.npy")]) == 0
    assert "16 px: 0 (no window at least half valid)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no valid window"):
        sc.scale_profile(scores=p, patches=(16,))


def test_logits_that_are_not_logits_are_warned_here_too(tmp_path, capsys):
    """1.14.1's warning, which the first draft of scale dropped (the review of 1.15.0)."""
    from oe_inferencex.assess import LOGIT_RANGE_WARNING
    np.save(tmp_path / "p1.npy", np.random.default_rng(2).random((32, 32)))
    out = tmp_path / "s.json"
    assert main(["scale", str(tmp_path / "p1.npy"), "--logits", "--patches", "1", "4", "--out", str(out)]) == 0
    assert "warning: " + LOGIT_RANGE_WARNING in capsys.readouterr().out
    d = json.loads(out.read_text())
    assert {"warning": LOGIT_RANGE_WARNING, "window_sizes": [1, 4]} in d["warnings"]


def test_an_undefined_error_rate_is_said_and_written_as_null(tmp_path, capsys):
    """A checkerboard reference has no majority in any 2 px window."""
    p, _ = _road()
    ref = (np.indices((64, 64)).sum(0) % 2).astype(int)
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "ref.npy", ref)
    out = tmp_path / "s.json"
    assert main(["scale", str(tmp_path / "p.npy"), "--reference", str(tmp_path / "ref.npy"), "--patches", "1", "2",
                 "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "2 px: 1024, " in printed and "undefined, undefined (no window has a prediction and a majority" in printed

    def no_nan(x):
        raise AssertionError(f"not strict JSON: {x}")
    d = json.loads(out.read_text(), parse_constant=no_nan)
    assert d["per_patch"]["2"]["against_reference"]["error_rate"] is None


def test_a_reference_class_the_map_never_draws_is_printed(tmp_path, capsys):
    p = np.zeros((3, 24, 24))
    p[0, :, :12], p[1, :, 12:], p[2] = 0.7, 0.7, 0.1
    p[0, :, 12:], p[1, :, :12] = 0.2, 0.2
    ref = np.where(np.arange(24)[None, :] < 12, 0, 1) * np.ones((24, 1), int)
    ref[:12, :12] = 2
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "ref.npy", ref)
    assert main(["scale", str(tmp_path / "p.npy"), "--reference", str(tmp_path / "ref.npy"), "--patches", "4"]) == 0
    assert "class 2 (in the reference, not drawn by the map): producer's accuracy against the reference: 4 px 0.00" \
        in capsys.readouterr().out


def test_the_product_route_on_the_command_line(tmp_path, capsys):
    p, truth = _road()
    np.save(tmp_path / "cls.npy", p.argmax(0))
    np.save(tmp_path / "conf.npy", np.round(p.max(0) * 100))
    out = tmp_path / "s.json"
    assert main(["scale", str(tmp_path / "cls.npy"), "--confidence", str(tmp_path / "conf.npy"),
                 "--confidence-range", "1", "79", "--patches", "1", "4", "--out", str(out)]) == 0
    assert "left out as no-data" in capsys.readouterr().out
    assert json.loads(out.read_text())["inputs"]["confidence_range"] == [1.0, 79.0]
    with pytest.raises(SystemExit, match="--confidence-range needs --confidence"):
        main(["scale", str(tmp_path / "cls.npy"), "--confidence-range", "1", "79"])
    with pytest.raises(SystemExit, match="--logits reads per-class scores"):
        main(["scale", str(tmp_path / "cls.npy"), "--confidence", str(tmp_path / "conf.npy"), "--logits"])


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="needs rasterio")
def test_window_sizes_convert_the_crs_unit_and_skip_a_rotated_grid():
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from oe_inferencex.cli import _window_size_text
    feet = {"crs": CRS.from_epsg(2263), "transform": Affine(10, 0, 0, 0, -10, 0)}
    assert _window_size_text(4, feet) == f"4 px ({4 * 10 * 0.30480060960121924:g} m)"
    utm = {"crs": CRS.from_epsg(32632), "transform": Affine(10, 0, 0, 0, -10, 0)}
    assert _window_size_text(4, utm) == "4 px (40 m)"
    assert _window_size_text(4, {"crs": CRS.from_epsg(32632), "transform": Affine(0, 10, 0, -10, 0, 0)}) == "4 px"
    assert _window_size_text(4, {"crs": CRS.from_epsg(4326), "transform": Affine(0.1, 0, 0, 0, -0.1, 0)}) == "4 px"
    assert _window_size_text(4, None) == "4 px"


def test_the_terminal_names_the_ragged_edge_as_the_reason(tmp_path, capsys):
    """Two classes of twelve columns on a 24 x 24 map: at 16 px most of class 1 lies outside the window grid, and the
    terminal says so rather than blaming windows of another class (the re-review of 1.15.0)."""
    hard = np.zeros((24, 24), int)
    hard[:, 12:] = 1
    np.save(tmp_path / "cls.npy", hard)
    np.save(tmp_path / "conf.npy", np.full((24, 24), 0.8))
    assert main(["scale", str(tmp_path / "cls.npy"), "--confidence", str(tmp_path / "conf.npy")]) == 0
    printed = capsys.readouterr().out
    assert "16 px: 1, " in printed and "(320 px outside the window grid)" in printed
    assert ("first below half at 16 px, where 22% of its pixels sit in windows called another class and 78% in no "
            "valid window") in printed

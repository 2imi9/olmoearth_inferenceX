"""The bug hunt of 25 October 2026 on 1.17.1 (six module groups, each finding reproduced by a second agent), and the
independent review of its fixes, fixed in 1.17.2: one test per finding or group of findings, each with the case that
showed it. Run against 1.17.1, 41 of the bug hunt's 44 fail (the three that pass are the controls of an exact edge and
the UTF-8 write, which a UTF-8 locale did not show); run against the fixes before each review, each of the first
review's 19 tests fails (21 with the parameters), each of the second's 14 (16) and each of the third's 11."""
import contextlib
import importlib.util
import io
import json
import os

import numpy as np
import pytest

pytest.importorskip("rasterio")

from oe_inferencex.cli import main, read_raster        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qs)


@pytest.fixture()
def quick(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        qs.write_maps()
    yield tmp_path
    os.chdir(cwd)


def _labelled(tmp_path, name="s"):
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--design", "random", "--budget", "300", "--out", f"{name}.csv"]) == 0
        qs.label(f"{name}.csv")


def test_certify_and_estimate_never_write_over_the_map(quick):
    """`certify s.csv --out map.json` with map.npy as the map deleted the map; `--out map.tif` wrote the zone over it."""
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        np.save(quick / "map.npy", src.read())
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "map.npy", "--design", "random", "--budget", "300", "--out", "m.csv"]) == 0
    qs.label("m.csv")
    before = (quick / "map.npy").read_bytes()
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["certify", "m.csv", "--alpha", "0.01", "--out", "map.json"])     # its mask would be map.npy
    _labelled(quick)
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["certify", "s.csv", "--alpha", "0.05", "--out", "scores.tif"])
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["estimate", "s.csv", "--per-class", "--out", "scores.tif"])
    assert (quick / "map.npy").read_bytes() == before and read_raster(str(quick / "scores.tif"))[0].ndim == 3


def test_sample_never_writes_over_its_inputs(quick):
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["sample", "scores.tif", "--design", "random", "--budget", "40", "--out", "scores.tif"])
    assert read_raster(str(quick / "scores.tif"))[0].ndim == 3


def test_another_letter_case_of_the_sidecar_is_the_sidecar(quick):
    _labelled(quick)
    (quick / "probe.txt").write_text("x")
    if not (quick / "PROBE.txt").exists():
        pytest.skip("a case-sensitive file system: S.json is another file")
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["estimate", "s.csv", "--out", "S.json"])


def test_a_gdal_mask_is_no_data(quick):
    """No-data kept in a GeoTIFF's internal mask (no nodata value) was read as data: the masked half's all-zero vectors
    filled the review set and half a random sample."""
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        prof, a = src.profile, src.read()
    a[:, :, a.shape[2] // 2:] = 0
    prof.update(nodata=None)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(quick / "masked.tif", "w", **prof) as dst:
            dst.write(a)
            m = np.full(a.shape[1:], 255, np.uint8)
            m[:, a.shape[2] // 2:] = 0
            dst.write_mask(m)
    arr, valid, _ = read_raster(str(quick / "masked.tif"))
    assert valid.mean() == 0.5 and not valid[:, a.shape[2] // 2:].any()


@pytest.mark.parametrize("shape", [(), (7,), (2, 3, 4, 5), (0, 8, 8)])
def test_a_npy_of_the_wrong_rank_is_refused_by_name(tmp_path, shape):
    np.save(tmp_path / "bad.npy", np.zeros(shape, np.float32))
    with pytest.raises(SystemExit, match="a raster is a 2-D"):
        read_raster(str(tmp_path / "bad.npy"))


# ----------------------------------------------------------------------------- assess, scale and boundary
def _two_halves(path, crs="EPSG:32633", transform=None, px=10.0):
    """A two-class probability map, class 0 on the left half and class 1 on the right, as a GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    p = np.full((2, 32, 32), 0.1, np.float32)
    p[0, :, :16], p[1, :, 16:] = 0.9, 0.9
    with rasterio.open(path, "w", driver="GTiff", height=32, width=32, count=2, dtype="float32", crs=crs,
                       transform=transform or from_origin(500000, 4000000, px, px)) as dst:
        dst.write(p)
    return p


def test_boundary_width_in_metres_converts_the_crs_unit(tmp_path, capsys):
    """10 US-survey-foot pixels made a 20 m band 2 px wide (6.1 m) under the words '10 x 10 m pixels'."""
    import rasterio
    _two_halves(tmp_path / "ft.tif", crs="EPSG:2263")
    assert main(["boundary", str(tmp_path / "ft.tif"), "--width-m", "20", "--out", str(tmp_path / "b.tif")]) == 0
    assert "20 m (7 rows, 7 columns of 3.04801 x 3.04801 m pixels)" in capsys.readouterr().out
    with rasterio.open(tmp_path / "b.tif") as src:
        assert (src.read(1)[0] == 1).sum() == 14                # seven columns on each side of the edge


def test_boundary_width_in_metres_refuses_a_grid_with_no_unit(tmp_path):
    from rasterio.transform import Affine
    _two_halves(tmp_path / "nocrs.tif", crs=None, transform=Affine.identity())
    with pytest.raises(SystemExit, match="has no CRS"):
        main(["boundary", str(tmp_path / "nocrs.tif"), "--width-m", "20", "--out", str(tmp_path / "b.tif")])
    _two_halves(tmp_path / "rot.tif", transform=Affine(7.0710678, -7.0710678, 500000, 7.0710678, 7.0710678, 4000000))
    with pytest.raises(SystemExit, match="rotated or sheared"):
        main(["boundary", str(tmp_path / "rot.tif"), "--width-m", "20", "--out", str(tmp_path / "b.tif")])


def test_boundary_refuses_by_name_what_ended_in_tracebacks(tmp_path):
    _two_halves(tmp_path / "p.tif")
    with pytest.raises(SystemExit, match="name --out with .tif"):        # a GeoTIFF under a .npy name
        main(["boundary", str(tmp_path / "p.tif"), "--width-px", "2", "--out", str(tmp_path / "band.npy")])
    with pytest.raises(SystemExit, match="--patch must be at least 1"):
        main(["boundary", str(tmp_path / "p.tif"), "--width-px", "2", "--patch", "0", "--out", str(tmp_path / "b.tif")])
    with pytest.raises(SystemExit, match="boundary: the window of 64 px is larger than the map"):
        main(["boundary", str(tmp_path / "p.tif"), "--width-px", "2", "--patch", "64", "--out", str(tmp_path / "b.tif")])
    (tmp_path / "adir").mkdir()
    with pytest.raises(SystemExit, match="is a directory"):
        main(["boundary", str(tmp_path / "p.tif"), "--width-px", "2", "--out", str(tmp_path / "adir")])
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["boundary", str(tmp_path / "p.tif"), "--width-px", "2",
                     "--out", str(tmp_path / "missing" / "b.tif")]) == 0          # the parent is made, as elsewhere
    assert (tmp_path / "missing" / "b.tif").exists()


def test_boundary_next_command_reads_the_map_as_boundary_did(tmp_path, capsys):
    p = np.full((3, 32, 32), 0.1, np.float32)
    p[0, :, :16], p[1, :, 16:] = 0.8, 0.8
    p[:, :, :8] = 0                                            # a .npy's fill, declared with --nodata 0
    np.save(tmp_path / "p.npy", p)
    assert main(["boundary", str(tmp_path / "p.npy"), "--width-px", "2", "--nodata", "0",
                 "--out", str(tmp_path / "band.npy")]) == 0
    nxt = capsys.readouterr().out.strip().splitlines()[-1]
    assert nxt.startswith("next: oe-inferencex sample ") and " --nodata=0.0 " in nxt


def _logit_strip(masked_value=0.0):
    rng = np.random.default_rng(0)
    s = rng.normal(0, 3, (32, 32))
    s[:, -8:] = masked_value
    mask = np.zeros((32, 32), bool)
    mask[:, -8:] = True
    return s, mask


def test_a_masked_array_is_no_data_in_assess_and_scale():
    """The mask was dropped: a stored 0 has logit margin 0, and all 8 windows of a 12.5% review set were masked."""
    from oe_inferencex.assess import assess_classmap, assess_prediction
    from oe_inferencex.scale import scale_profile
    s, mask = _logit_strip()
    plain = assess_prediction(s, is_logit=True, nodata_mask=mask, budgets=(0.125,))
    masked = assess_prediction(np.ma.masked_array(s, mask), is_logit=True, budgets=(0.125,))
    assert masked["n_windows"] == plain["n_windows"] == 48
    assert (masked["review_sets"][0.125]["windows_rowcol"] == plain["review_sets"][0.125]["windows_rowcol"]).all()
    assert scale_profile(np.ma.masked_array(s, mask), is_logit=True, patches=(4,))["per_patch"][4]["n_windows"] == 48
    conf = np.where(mask, 0.0, 1 / (1 + np.exp(-np.abs(s))))
    hard = (s > 0).astype(int)
    cm = assess_classmap(hard, np.ma.masked_array(conf, mask), 2, budgets=(0.125,))
    assert cm["n_windows"] == 48


def test_a_layer_of_another_shape_is_refused():
    """A reference 3 columns wider was cropped and graded: 0.538 for an aligned 0.000."""
    from oe_inferencex.assess import assess_classmap, assess_prediction
    from oe_inferencex.scale import scale_profile
    s, _ = _logit_strip(masked_value=2.0)
    ref = (s > 0).astype(int)
    assert assess_prediction(s, is_logit=True, reference=ref)["against_reference"]["error_rate"] == 0
    wide = np.pad(ref, ((0, 0), (3, 0)))
    with pytest.raises(ValueError, match="the reference is 32 x 35 px and the map is 32 x 32 px"):
        assess_prediction(s, is_logit=True, reference=wide)
    with pytest.raises(ValueError, match="the reference is 32 x 35 px"):
        scale_profile(s, is_logit=True, patches=(4,), reference=wide)
    with pytest.raises(ValueError, match="the confidence band is 32 x 35 px"):
        assess_classmap(ref, np.ones((32, 35)), 2)
    with pytest.raises(ValueError, match="the no-data mask is 32 x 35 px"):
        assess_prediction(s, is_logit=True, nodata_mask=np.zeros((32, 35), bool))


def test_with_no_error_the_capture_is_undefined(tmp_path, capsys):
    from oe_inferencex.assess import assess_prediction, summary
    s, _ = _logit_strip(masked_value=2.0)
    ref = (s > 0).astype(int)
    out = assess_prediction(s, is_logit=True, reference=ref)
    cap = summary(out)["against_reference"]["error_capture_at_budget"]
    assert all(v["errors_captured_fraction"] is None for v in cap.values())
    np.save(tmp_path / "s.npy", s)
    np.save(tmp_path / "r.npy", ref)
    assert main(["assess", str(tmp_path / "s.npy"), "--logits", "--reference", str(tmp_path / "r.npy"),
                 "--out", str(tmp_path / "o")]) == 0
    assert "no window disagrees with it, so error capture is undefined" in capsys.readouterr().out


def test_a_partial_reference_says_what_its_capture_is_measured_on(quick, capsys):
    """'5% -> 0.31' was printed beside a 5% review set that holds 0.46 of the disagreements."""
    import rasterio
    with rasterio.open(quick / "truth.tif") as src:
        prof, ref = src.profile, src.read(1).astype(np.int32)
    ref[:, ref.shape[1] // 2:] = -1
    ref[ref.shape[0] // 2:, :] = -1
    prof.update(dtype="int32", nodata=-1)
    with rasterio.open(quick / "quarter.tif", "w", **prof) as dst:
        dst.write(ref, 1)
    assert main(["assess", "scores.tif", "--reference", "quarter.tif", "--out", "o"]) == 0
    line = capsys.readouterr().out
    assert "against the reference, which grades" in line and "not on the review sets" in line


def test_a_large_reference_code_costs_no_more_than_the_classes_present(monkeypatch):
    """An undeclared fill of 65535 pooled one count grid per id up to it: 1.1 GB at 128 x 128, 32 GiB at 1024 x 1024."""
    from oe_inferencex import assess as A
    s, _ = _logit_strip(masked_value=2.0)
    ref = (s > 0).astype(np.int64)
    ref[:, :4] = 65535
    seen = []
    real = A._pooled_argmax
    monkeypatch.setattr(A, "_pooled_argmax", lambda hard, n, *a, **k: seen.append(n) or real(hard, n, *a, **k))
    out = A.assess_prediction(s, is_logit=True, reference=ref)
    assert max(seen) == 3                                    # classes 0 and 1, and the fill
    pairs = out["against_reference"]["confusion_pairs"]["pairs"]
    assert {p["reference"] for p in pairs} == {65535}            # the fill strip is the only error, named by its id
    assert any("(65535)" in w for w in out["warnings"])


# ----------------------------------------------------------------------------- estimate
def test_tiles_of_one_equal_rate_get_the_fallback_not_a_point():
    """Every sampled tile 1 wrong in 7: the ratio estimator landed an ulp off 1/7, the variance came out 1e-34, and
    '14.3% to 14.3%' was printed."""
    from oe_inferencex import estimate as est
    tiles = np.repeat(np.arange(6), 100)
    sample = {"design": "tiles", "indices": np.concatenate([np.arange(t * 100, t * 100 + 7) for t in range(6)]),
              "n_population": 600, "tiles": tiles, "tile_ids": list(range(6)), "tile_valid_sizes": [100] * 6,
              "n_tiles": 6}
    wrong = np.tile([1, 0, 0, 0, 0, 0, 0], 6)
    r = est.estimate_error_rate(sample, wrong)
    lo, hi = est.wilson_interval(6, 42, 600)
    assert r["estimate"] == 1 / 7 and (r["low"], r["high"]) == (lo, hi) and r["half_width"] > 0.05
    assert "every sampled tile shows the same error rate" in r["warning"]


def test_the_python_api_reads_a_sequential_sample_as_the_command_line_does():
    from oe_inferencex import estimate as est
    rng = np.random.default_rng(0)
    margin = rng.random((20, 20))
    s = est.sample_for_estimation(margin, 100, design="sequential")
    wrong = (rng.random(100) < 0.1).astype(int)
    r = est.estimate_error_rate(s, wrong)
    want = est.estimate_error_rate(dict(s, design="random", budget=100), wrong)
    assert (r["low"], r["high"]) == (want["low"], want["high"]) and "sequential sample" in r["sequential_note"]
    mc = np.zeros(400, int)
    assert est.estimate_per_class(s, mc[s["indices"]], mc)["n_labelled"] == 100


def _other_sample(quick):
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--other", "other.tif", "--budget", "60", "--out", "p.csv"]) == 0
    qs.label("p.csv")


def test_a_column_named_twice_is_refused_on_every_route(quick):
    """--other and audit samples were read with DictReader, which keeps the last copy where a spreadsheet shows the
    first: a second reference_class turned 'undetermined' into 'b is more accurate', exit 0."""
    _other_sample(quick)
    text = (quick / "p.csv").read_text().splitlines()
    (quick / "p.csv").write_text("\n".join([text[0] + ",reference_class"] + [ln + ",0" for ln in text[1:]]) + "\n")
    with pytest.raises(SystemExit, match="names a column twice"):
        main(["estimate", "p.csv"])
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--design", "audit", "--per-class", "10", "--out", "a.csv"]) == 0
    qs.label("a.csv")                    # the quick-start labeller fills an audit sample (it left its header alone)
    text = (quick / "a.csv").read_text().splitlines()
    assert len(text) > 10 and text[1].split(",")[-1] != ""
    (quick / "a.csv").write_text("\n".join([text[0] + ",reference_class"] + [ln + ",0" for ln in text[1:]]) + "\n")
    with pytest.raises(SystemExit, match="names a column twice"):
        main(["estimate", "a.csv"])


def test_a_blank_line_in_a_labelled_csv_is_read_as_before(quick):
    """1.17.0's strict reader refused an editor's trailing newline as a bad index column."""
    _labelled(quick)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["estimate", "s.csv", "--out", "e1.json"]) == 0
    (quick / "s.csv").write_text((quick / "s.csv").read_text() + "\r\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["estimate", "s.csv", "--out", "e2.json"]) == 0
    e1, e2 = (json.load(open(quick / f"e{i}.json")) for i in (1, 2))
    assert (e1["low"], e1["high"]) == (e2["low"], e2["high"])


def test_classes_is_refused_where_it_is_not_read(quick):
    _other_sample(quick)
    for v in ("4", "-7"):
        with pytest.raises(SystemExit, match="--classes"):
            main(["estimate", "p.csv", "--classes", v])


def _class_sample(quick, names=None):
    argv = ["sample", "scores.tif", "--design", "class", "--budget", "400", "--out", "c.csv"]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert main(argv + (["--condition-names", *names] if names else [])) == 0
    qs.label("c.csv")
    return out.getvalue()


def test_the_class_design_prints_olofssons_normal_theory_intervals(quick, capsys):
    """They were Wilson intervals on an effective n under a note that calls them Olofsson's estimate +/- 1.96 SE, and
    the overall normal-theory accuracy the 1.17.1 CHANGELOG puts in the JSON was not there."""
    from oe_inferencex import estimate as est
    _class_sample(quick)
    assert main(["estimate", "c.csv", "--per-class"]) == 0
    res = json.load(open(quick / "c_estimate.json"))
    o = res["overall_accuracy_normal_theory"]
    assert o["low"] < o["estimate"] < o["high"]
    for row in res["per_class"].values():
        pa = row["producer_accuracy"]
        if pa and 0 < pa["low"] and pa["high"] < 1:          # unclipped: symmetric about the estimate, as +/- 1.96 SE is
            assert pa["estimate"] - pa["low"] == pytest.approx(pa["high"] - pa["estimate"], abs=1e-12)
    assert res["per_class_method"] == est.CLASS_DESIGN_METHOD


def test_class_names_and_paths_keep_their_words(tmp_path):
    """A class named 'good condition' was printed 'good class', the JSON's outside_condition_intervals named classes
    that were not its keys, and a sample in a folder named condition was sent to .../class/to_label.csv."""
    from oe_inferencex.cli import class_words
    names = ["good condition", "poor condition"]
    text = "good condition 8.0% (3.6% to 15.4%); each condition's interval; see /data/condition/c.csv and `condition`"
    assert class_words(text, names) == ("good condition 8.0% (3.6% to 15.4%); each class's interval; see "
                                        "/data/condition/c.csv and `condition`")
    res = {"per_condition": {"good condition": {}}, "outside_condition_intervals": ["good condition"],
           "condition_note": "per input condition"}
    worded = class_words(res, names)
    assert worded["outside_condition_intervals"] == ["good condition"] and worded["condition_note"] == "per map class"


def test_decide_words_a_class_design_result_by_class(quick, capsys):
    _class_sample(quick)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["estimate", "c.csv"]) == 0
        assert main(["certify", "c.csv", "--alpha", "0.2"]) == 0
    assert json.load(open(quick / "c_estimate.json"))["condition_by"] == "map class"
    capsys.readouterr()
    assert main(["decide", "c_estimate.json", "--ask", "error_rate_below=0.15"]) == 0
    assert main(["decide", "c_zone.json", "--ask", "trusted_share"]) == 0
    out = capsys.readouterr().out
    assert "input condition" not in out and "of this condition" not in out and "map class" in out


def test_a_class_design_sample_says_how_to_label_it_blind(quick):
    from oe_inferencex import estimate as est
    assert est.CLASS_BLIND_NOTE in _class_sample(quick)


# ----------------------------------------------------------------------------- certify and plan
def test_the_printed_edge_takes_the_zone_and_nothing_else():
    """Rounded to the nearest fourth decimal, 0.72453757 took in windows down to 0.7245 outside the zone, and 0.43005
    printed 0.4301 left the threshold's own window out."""
    from oe_inferencex.estimate import zone_edge
    assert zone_edge(0.7245375783024242, 0.72453606) == "0.724537"
    assert zone_edge(0.4300537109375, 0.41) == "0.4300"
    assert zone_edge(0.20623779296875, None) == "0.2062"
    for t, below in ((0.7245375783024242, 0.72453606), (0.4300537109375, 0.43005), (0.5, 0.49999999)):
        v = float(zone_edge(t, below))
        assert below < v <= t


def test_a_sequential_sample_refuses_any_rule(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "p.npy", rng.random((2, 64, 64)).astype(np.float32))
    p = str(tmp_path / "p.npy")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", p, "--budget", "100", "--design", "sequential", "--alpha", "0.1",
                     "--out", str(tmp_path / "q.csv")]) == 0
    rows = (tmp_path / "q.csv").read_text().splitlines()
    head = rows[0].split(",")
    w = head.index("wrong")
    (tmp_path / "q.csv").write_text("\n".join([rows[0]] + [",".join(c if i != w else "0" for i, c in
                                                                     enumerate(r.split(","))) for r in rows[1:]]) + "\n")
    with pytest.raises(SystemExit, match="sequential rule"):
        main(["certify", str(tmp_path / "q.csv"), "--alpha", "0.1", "--rule", "prefix"])


def test_a_small_alpha_is_answered_at_once():
    """min_labels_sequential stepped one label at a time: 21 s at alpha 1e-8, hours below."""
    import math
    import time
    from oe_inferencex import estimate as est
    from oe_inferencex.sequential import min_labels_sequential

    def stepped(alpha, delta):
        target, l1a = -math.log(delta), math.log1p(-alpha)
        t = est.min_labels_to_certify(alpha, delta)
        while math.log(-math.expm1((t + 1) * l1a)) - math.log(t + 1) - math.log(alpha) - t * l1a < target:
            t += 1
        return t
    for a in (0.001, 0.01, 0.05, 0.2, 0.5, 0.9):
        for d in (0.01, 0.1, 0.5, 0.9):
            assert min_labels_sequential(a, d) == stepped(a, d)
    t0 = time.perf_counter()
    assert min_labels_sequential(1e-9, 0.1) > 2e9
    assert time.perf_counter() - t0 < 1


def test_the_fewest_labels_keeps_the_float_ceiling():
    """At delta = (1 - alpha)^b as written the float ceiling asks for b + 1. 1.17.2 first decided it exactly, which
    changed the level cut, so the zones certify tests: on some samples it certified more, and certificates made with
    the ceiling failed verify (the third review of 1.17.2). Kept as every release has computed it."""
    from oe_inferencex.estimate import min_labels_to_certify
    assert min_labels_to_certify(0.3, 0.0823543) == 8 and min_labels_to_certify(0.6, 0.064) == 4
    assert min_labels_to_certify(0.05, 0.1) == 45 and min_labels_to_certify(0.02, 0.1) == 114


def test_plan_fixes_the_rule_its_budget_needs(capsys):
    assert main(["plan", "--windows", "20000", "--coverage", "0.5", "--alpha", "0.05", "--zone-error", "0.03",
                 "--max-labels", "4000"]) == 0
    run = [ln for ln in capsys.readouterr().out.splitlines() if "oe-inferencex sample" in ln and "--target" not in ln]
    assert run and all("--seed-from-map --alpha 0.05 --rule bonferroni" in ln for ln in run)


def test_plan_on_a_map_smaller_than_a_zone_says_no_zone_is_tested():
    from oe_inferencex.plan import plan_zone
    r = plan_zone(30, 0.5, 0.05)
    assert "test no zone at all" in r["refusal"] and "1.5" not in r["refusal"]


# ----------------------------------------------------------------------------- inputs and outputs
def test_from_olmoearth_condition_names_fit_the_command_line():
    """Three rslearn layers made a 75-character name, and the assess command printed beside it was refused."""
    from oe_inferencex.cli import _condition_names
    from oe_inferencex.olmoearth import _code_name, _command_names
    layers = ["sentinel1_ascending", "sentinel1_descending", "sentinel2_l2a"]
    clouds = {"sentinel1_ascending": False, "sentinel1_descending": False, "sentinel2_l2a": True}
    present = [0, 1, 2, 17, 40, 159]
    assert max(len(_code_name(v, layers, clouds)) for v in present) > 64
    short, numbered = _command_names(present, layers, clouds)
    assert numbered == {"1": "sentinel1_ascending", "2": "sentinel1_descending", "3": "sentinel2_l2a"}
    assert _condition_names([f"{v}={short[v]}" for v in present], "assess") == short
    assert _command_names(present, ["s2", "s1"], {"s2": True, "s1": False})[1] is None     # short names stay whole


def test_sample_csvs_are_written_as_utf8(quick):
    """In the locale's encoding (cp1252 on Windows) bewölkt was one byte, and estimate refused the file."""
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        prof, h, w = src.profile, src.height, src.width
    prof.update(count=1, dtype="int32", nodata=-1)
    with rasterio.open(quick / "cond.tif", "w", **prof) as dst:
        dst.write((np.arange(w)[None, :] >= w // 2).astype(np.int32).repeat(h, 0), 1)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--budget", "100", "--condition", "cond.tif",
                     "--condition-names", "0=klar", "1=bewölkt", "--out", "s.csv"]) == 0
    assert "bewölkt".encode("utf-8") in (quick / "s.csv").read_bytes()


def test_from_embeddings_points_are_read_or_refused_by_name(tmp_path):
    """A blank class was fitted as a class named '', a bad coordinate was a traceback, a BOM hid the first column."""
    pytest.importorskip("pyproj")
    from oe_inferencex.embeddings import dequantize, read_points
    p = tmp_path / "pts.csv"
    p.write_bytes("﻿lon,lat,class\n10.5,45.2,crop\n10.6,45.3,\n10.7,45.1,forest\n".encode("utf-8"))
    assert read_points(str(p)) == [(10.5, 45.2, "crop"), (10.6, 45.3, None), (10.7, 45.1, "forest")]
    for bad in ("10.5,,crop", "nan,45,crop", "10.5,95,crop"):
        p.write_text(f"lon,lat,class\n{bad}\n")
        with pytest.raises(SystemExit, match="row 2 of"):
            read_points(str(p))
    for quant in (None, {"method": "signed_power"}):
        with pytest.raises(SystemExit, match="geoemb:quantization"):
            dequantize(np.zeros((2, 2), np.int8), quant)


def test_gdal_virtual_paths_are_recorded_as_given(tmp_path):
    from oe_inferencex.cli import _where
    for p in ("/vsizip//abs/maps.zip/scores.tif", "/vsicurl/https://host/x.tif", "s3://bucket/x.tif"):
        assert _where(p) == p
    assert _where("scores.tif") == os.path.abspath("scores.tif")


def test_a_sample_with_both_map_columns_deleted_is_refused(quick):
    """Nothing was checked, and another map on the same grid was certified from this map's labels."""
    _labelled(quick)
    rows = (quick / "s.csv").read_text().splitlines()
    head = rows[0].split(",")
    keep = [i for i, c in enumerate(head) if c not in ("confidence", "map_class")]
    (quick / "s.csv").write_text("\n".join(",".join(r.split(",")[i] for i in keep) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="neither the `confidence` nor the `map_class` column"):
        main(["certify", "s.csv", "--alpha", "0.05", "--scores", "other.tif"])


def test_extend_reads_the_map_as_the_sample_did(quick, capsys):
    """--logits added on --extend passed, and every kept label was paired with another confidence."""
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert main(["sample", "scores.tif", "--design", "sequential", "--budget", "60", "--nodata", "-1",
                     "--alpha", "0.1", "--out", "q.csv"]) == 0
    printed = out.getvalue()
    assert "--nodata=-1.0" in printed.split("To add windows later: ")[1]
    with pytest.raises(SystemExit, match=r"reading it the same way \(--logits, --nodata differ\)"):
        main(["sample", "scores.tif", "--design", "sequential", "--budget", "80", "--logits", "--extend", "q.csv",
              "--out", "q.csv"])


def test_other_and_audit_refuse_one_maps_options(quick):
    for extra in (["--alpha", "0.05"], ["--rule", "bonferroni"], ["--min-per-class", "7"],
                  ["--condition-names", "0=a"]):
        with pytest.raises(SystemExit, match="sample --other takes none of"):
            main(["sample", "scores.tif", "--other", "other.tif", "--budget", "50", "--out", "p.csv", *extra])
    with pytest.raises(SystemExit, match="sample --design audit takes none of --rule"):
        main(["sample", "scores.tif", "--design", "audit", "--per-class", "5", "--rule", "bonferroni",
              "--out", "a.csv"])


# ----------------------------------------------------------------------------- the review of 1.17.2
@pytest.mark.parametrize("kind", ["random", "class", "sequential"])
def test_a_certificate_issued_by_1_17_1_still_verifies(tmp_path, kind):
    """certify's result gained `edge` (and `condition_by` for the class design), and verify compared certify run again
    with the recorded zone JSON key for key: every certificate 1.17.0 and 1.17.1 issued failed."""
    import shutil
    src = os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", kind)
    shutil.copytree(src, tmp_path / kind)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qs.write_maps()
            main(["verify", f"{kind}/certificate.json", "--scores", "scores.tif", "--out", "v.json"])
        v = json.load(open(tmp_path / "v.json"))
        assert v["verified"] is True, {k: c for k, c in v["checks"].items() if not c.get("ok")}
    finally:
        os.chdir(cwd)


def _write_tif(path, a, **kw):
    import rasterio
    from rasterio.transform import from_origin
    a = np.asarray(a)
    a3 = a[None] if a.ndim == 2 else a
    prof = dict(driver="GTiff", height=a3.shape[1], width=a3.shape[2], count=a3.shape[0], dtype=str(a3.dtype),
                crs="EPSG:32633", transform=from_origin(500000, 4000000, 10, 10))
    prof.update(kw)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(a3)
        return dst


def test_an_alpha_band_is_the_mask_not_a_band(tmp_path):
    """A class map with an alpha band was read as a 2-band score map: compare found 128 of 192 windows 'differ'."""
    import rasterio
    from rasterio.enums import ColorInterp
    cls = (np.arange(64)[None, :] // 16 % 3).repeat(64, 0).astype(np.uint8)
    alpha = np.full((64, 64), 255, np.uint8)
    alpha[:, 48:] = 0
    with rasterio.open(tmp_path / "a.tif", "w", driver="GTiff", height=64, width=64, count=2, dtype="uint8",
                       crs="EPSG:32633", transform=rasterio.transform.from_origin(0, 640, 10, 10)) as dst:
        dst.colorinterp = [ColorInterp.gray, ColorInterp.alpha]
        dst.write(np.stack([cls, alpha]))
    arr, valid, _ = read_raster(str(tmp_path / "a.tif"))
    assert arr.shape == (64, 64) and (arr == cls).all() and valid.mean() == 0.75


def test_a_mask_in_a_separate_file_is_refused_for_a_certificate(quick):
    """verify reads a bundle's rasters without their sidecar files: the certificate failed its own check."""
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        prof, a = src.profile, src.read()
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(quick / "ext.tif", "w", **prof) as dst:
            dst.write(a)
            m = np.full(a.shape[1:], 255, np.uint8)
            m[:, :8] = 0
            dst.write_mask(m)
    assert (quick / "ext.tif.msk").exists()
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "ext.tif", "--design", "random", "--budget", "315", "--target", "0.5",
                     "--seed-from-map", "--alpha", "0.05", "--out", "e.csv"]) == 0
        qs.label("e.csv")
        assert main(["certify", "e.csv"]) == 0
    with pytest.raises(SystemExit, match="reads differently without the files beside it"):
        main(["certificate", "e_zone.json", "--date", "2024-06-01", "--out", "cert"])
    # the same mask moved inside the GeoTIFF keeps the pixels and the mask: the labelled sample stands, as the refusal
    # says (it said to draw the sample again; the second review of 1.17.2)
    os.remove(quick / "ext.tif.msk")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(quick / "ext.tif", "w", **prof) as dst:
            dst.write(a)
            dst.write_mask(m)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["certificate", "e_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
        main(["verify", "cert/certificate.json", "--out", "v.json"])
    assert json.load(open(quick / "v.json"))["verified"] is True


def test_printed_commands_give_numbers_argparse_reads(tmp_path, capsys):
    """'--nodata -3.4e+38' after a space is an option to argparse, and :g made a fill of 1234567 another value."""
    from oe_inferencex.cli import _nodata_arg, _num_arg
    assert _nodata_arg(-3.4028234663852886e+38) == "--nodata=-3.4028234663852886e+38"
    assert _nodata_arg(1234567) == "--nodata=1234567.0" and _nodata_arg(float("nan")) == "--nodata=nan"
    assert float(_num_arg(-3.4e38)) == -3.4e38 and not _num_arg(-3.4e38).lstrip("-").count("e")
    p = np.full((3, 32, 32), 0.1, np.float32)
    p[0, :, :16], p[1, :, 16:] = 0.8, 0.8
    p[:, :, :8] = -3.4028234663852886e+38
    np.save(tmp_path / "p.npy", p)
    assert main(["boundary", str(tmp_path / "p.npy"), "--width-px", "2", "--nodata=-3.4028234663852886e+38",
                 "--out", str(tmp_path / "band.npy")]) == 0
    nxt = capsys.readouterr().out.strip().splitlines()[-1].split("next: oe-inferencex ")[1]
    import shlex
    argv = shlex.split(nxt.replace("to_label.csv", str(tmp_path / "s.csv")).replace("--budget 300", "--budget 20"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(argv) == 0                                     # runs as printed


def test_boundary_next_command_reads_logits_and_class_maps(tmp_path, capsys):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "logit.npy", rng.normal(0, 4, (32, 32)).astype(np.float32))
    assert main(["boundary", str(tmp_path / "logit.npy"), "--threshold", "0", "--width-px", "1",
                 "--out", str(tmp_path / "b.npy")]) == 0
    assert " --logits " in capsys.readouterr().out.splitlines()[-1]
    np.save(tmp_path / "cls.npy", (np.arange(32)[None, :] // 8).repeat(32, 0).astype(np.int32))
    assert main(["boundary", str(tmp_path / "cls.npy"), "--width-px", "1", "--out", str(tmp_path / "c.npy")]) == 0
    assert "--confidence BAND" in capsys.readouterr().out.splitlines()[-1]
    _two_halves(tmp_path / "p.tif")
    (tmp_path / "d").mkdir()
    with pytest.raises(SystemExit, match="name the file to write, for example .*band.tif"):
        main(["boundary", str(tmp_path / "p.tif"), "--width-px", "1", "--out", str(tmp_path / "d")])


def test_a_mask_of_the_wrong_shape_is_refused_before_a_masked_array_widens_it():
    from oe_inferencex.assess import assess_classmap, assess_prediction
    s, mask = _logit_strip()
    with pytest.raises(ValueError, match="the no-data mask is of shape"):
        assess_prediction(np.ma.masked_array(s, mask), is_logit=True, nodata_mask=np.zeros(32, bool))
    hard = (s > 0).astype(int)
    with pytest.raises(ValueError, match="the confidence band is 32 x 35 px"):
        assess_classmap(hard, np.ma.masked_array(np.ones((32, 35)), np.zeros((32, 35), bool)), 2)


def test_the_reference_warning_counts_the_references_own_classes():
    from oe_inferencex.assess import assess_prediction
    s = np.random.default_rng(1).random((3, 16, 16))
    ref = np.zeros((16, 16), int)
    ref[:, 8:] = 5
    w = assess_prediction(s, is_logit=False, reference=ref)["warnings"]
    assert any(x.startswith("the reference carries 2 classes and the map predicts 3") and "(5)" in x for x in w)


def test_the_edge_of_a_huge_or_tied_threshold():
    """Above 1e24 the edge was a decimal.InvalidOperation traceback; with a window outside the zone tied at the
    threshold it took in windows below the threshold that no text counted."""
    from oe_inferencex import estimate as est
    assert float(est.zone_edge(3.4028234663852886e+38, None)) == 3.4028234663852886e+38
    margin = np.array([0.9] * 900 + [0.72453757] * 200 + [0.72452] * 50 + [0.5] * 850)
    order, _ = est.zone_order(margin)
    edge = est._edge_of(margin, order, 1000, 200, 100)
    assert 0.72452 < float(edge) <= 0.72453757


def test_mcp_plan_says_to_fix_the_bonferroni_rule():
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    out = mcp_server.plan(windows=20000, coverage=0.5, alpha=0.05, zone_error=0.03, max_labels=4000)
    assert 'rule="bonferroni"' in out["next"] and "seed_from_map=true" in out["next"]


def test_the_class_design_refuses_an_unknown_interval(quick):
    from oe_inferencex import estimate as est
    _class_sample(quick)
    side = json.load(open(quick / "c.json"))
    with pytest.raises(ValueError, match="interval must be"):
        est.estimate_per_class(dict(side, design="condition"), [0] * len(side["indices"]), np.zeros(4096, int),
                               interval="bogus")


def test_class_words_keep_paths_with_spaces_and_short_names():
    from oe_inferencex.cli import class_words
    text = "certified per input condition; mask /a/Site conditions 2026/z.npy; each condition's zone"
    out = class_words(text, ["t", "g"], ["/a/Site conditions 2026/z.npy"])
    assert out == "certified per map class; mask /a/Site conditions 2026/z.npy; each class's zone"
    assert class_words("t: 10% of this condition", ["t"]) == "t: 10% of this class"


def test_mcp_decide_speaks_of_classes_and_of_the_producers_joint_intervals(quick):
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    _class_sample(quick)
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["certify", "c.csv", "--alpha", "0.2"]) == 0
        assert main(["estimate", "c.csv", "--per-class"]) == 0
    z = mcp_server.decide(result_json=str(quick / "c_zone.json"), questions=["trusted_share_at_least=0.99"])
    assert "condition" not in z["limits"] + z["next"] + z["conclusion"]
    e = mcp_server.decide(result_json=str(quick / "c_estimate.json"), questions=["producer_accuracy_above=0.5"])
    assert "the producer's accuracies under a random sample or the class design" in e["limits"]


def test_the_printed_extension_runs_as_printed(quick):
    """It lacked --out, which sample requires; --out naming the extended sample's sidecar wrote over it."""
    import shlex
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert main(["sample", "scores.tif", "--design", "sequential", "--budget", "60", "--alpha", "0.1",
                     "--out", "q.csv"]) == 0
    cmd = out.getvalue().split("To add windows later: oe-inferencex ")[1].split(";")[0].split("\n")[0]
    argv = shlex.split(cmd.replace("<more>", "90"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(argv) == 0
    assert len(json.load(open(quick / "q.json"))["indices"]) == 90
    with pytest.raises(SystemExit, match="would overwrite the input"):
        main(["sample", "scores.tif", "--design", "sequential", "--budget", "120", "--extend", "q.csv",
              "--out", "q.json"])


def test_mcp_sequential_next_carries_the_maps_reading(quick):
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    out = mcp_server.sample(scores=str(quick / "scores.tif"), budget=60, out_dir=str(quick / "m"), design="sequential",
                            alpha=0.1, nodata=-1.0)
    assert "nodata=-1.0, " in out["next"] and "extend=" in out["next"]


def test_verify_never_writes_under_the_bundle(tmp_path):
    import shutil
    shutil.copytree(os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", "random"), tmp_path / "cert")
    with pytest.raises(SystemExit, match="into the bundle's directory"):
        main(["verify", str(tmp_path / "cert" / "certificate.json"), "--out", str(tmp_path / "cert" / "sub" / "v.json")])
    assert not (tmp_path / "cert" / "sub").exists()


def test_other_refuses_logits_and_plan_does_not_print_it(quick, capsys):
    with pytest.raises(SystemExit, match="takes none of --logits"):
        main(["sample", "scores.tif", "--other", "other.tif", "--logits", "--budget", "20", "--out", "p.csv"])
    from oe_inferencex.cli import _sample_command
    import argparse
    a = argparse.Namespace(scores="a.npy", other="b.npy", patch=4, nodata=None, logits=True, threshold=None)
    assert "--logits" not in _sample_command(a, 50, other=True) and "--logits" in _sample_command(a, 50, other=False)


def test_the_certificate_records_gdal_paths_as_given():
    from oe_inferencex.certificate import cli_where
    assert cli_where("/vsizip//abs/maps.zip/scores.tif") == "/vsizip//abs/maps.zip/scores.tif"


# ----------------------------------------------------------------------------- the second review of 1.17.2
@pytest.mark.parametrize("bands,dtype", [(1, "float32"), (1, "uint16"), (3, "float32"), (2, "uint8")])
def test_an_alpha_band_is_the_mask_beside_one_band_and_data_beside_more(tmp_path, bands, dtype):
    """Beside one data band an alpha band is the mask, whatever GDAL's flags say (it applies alpha only to 2- and 4-band
    Byte or UInt16 rasters): a float score with alpha had its transparent pixels read as data. Beside more it is read
    as data, as before 1.17.2, since GDAL marks the 4th band of a 4-band 8-bit GeoTIFF alpha by default."""
    import rasterio
    from rasterio.enums import ColorInterp
    rng = np.random.default_rng(0)
    data = rng.random((bands, 64, 64)).astype(dtype) if dtype == "float32" else \
        rng.integers(0, 3, (bands, 64, 64)).astype(dtype)
    alpha = np.full((1, 64, 64), 255, dtype)
    alpha[:, :, 48:] = 0
    with rasterio.open(tmp_path / "m.tif", "w", driver="GTiff", height=64, width=64, count=bands + 1, dtype=dtype,
                       crs="EPSG:32633", transform=rasterio.transform.from_origin(0, 640, 10, 10)) as dst:
        dst.colorinterp = [ColorInterp.gray] * bands + [ColorInterp.alpha]
        dst.write(np.concatenate([data, alpha]))
    arr, valid, _ = read_raster(str(tmp_path / "m.tif"))
    if bands == 1:
        assert arr.shape == (64, 64) and valid.mean() == 0.75 and not valid[:, 48:].any()
    else:
        assert arr.shape == (bands + 1, 64, 64) and valid.all()


def test_a_4_band_8_bit_map_written_with_gdals_defaults_keeps_its_4th_band(tmp_path):
    """GDAL marks it RGBA: read through its mask, a 4-class map lost class 4 and the pixels where it scored 0."""
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.default_rng(1)
    a = rng.integers(0, 100, (4, 32, 32)).astype(np.uint8)
    a[3, :, :8] = 0
    with rasterio.open(tmp_path / "c4.tif", "w", driver="GTiff", height=32, width=32, count=4, dtype="uint8",
                       crs="EPSG:32633", transform=from_origin(0, 320, 10, 10)) as dst:
        dst.write(a)
    arr, valid, _ = read_raster(str(tmp_path / "c4.tif"))
    assert arr.shape == (4, 32, 32) and valid.all()


def test_a_certificate_is_rerun_with_the_rules_of_the_release_that_wrote_it(tmp_path, monkeypatch):
    """1.17.1 made the zone test's null count exact: a certificate 1.17.0 issued at alpha 0.29, whose zones of 1500,
    3000 or 3500 windows were tested with floor(0.29 n) + 1 taken in floats, failed verify on its p-values. Issued here
    as 1.17.0 issued it (its null count, its version), then verified by this release."""
    import math
    import oe_inferencex
    from oe_inferencex import certificate as cert
    from oe_inferencex import estimate as est
    rng = np.random.default_rng(3)
    p = rng.random((2, 400, 400)).astype(np.float32)
    np.save(tmp_path / "p.npy", p / p.sum(0))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with monkeypatch.context() as mp:
            mp.setattr(est, "null_count", lambda n, alpha: math.floor(alpha * n) + 1)
            mp.setattr(oe_inferencex, "__version__", "1.17.0")
            with contextlib.redirect_stdout(io.StringIO()):
                assert main(["sample", "p.npy", "--design", "random", "--budget", "400", "--seed-from-map",
                             "--alpha", "0.29", "--out", "s.csv"]) == 0
                rows = (tmp_path / "s.csv").read_text().splitlines()
                w = rows[0].split(",").index("wrong")
                fill = lambda r, i: ",".join(c if j != w else str(int(rng.random() < 0.1)) for j, c in
                                             enumerate(r.split(",")))
                (tmp_path / "s.csv").write_text("\n".join([rows[0]] + [fill(r, i) for i, r in enumerate(rows[1:])]) + "\n")
                assert main(["certify", "s.csv"]) == 0
                assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
        zone = json.load(open(tmp_path / "cert" / "zone.json"))
        assert zone["written_by"]["version"] == "1.17.0"
        sizes = [lv["n_zone"] for lv in zone["levels"]]
        assert any(math.floor(0.29 * n) + 1 != est.null_count(n, 0.29) for n in sizes)    # the case is exercised
        res = cert.verify(str(tmp_path / "cert" / "certificate.json"), str(tmp_path / "p.npy"))
        assert res["verified"] is True, {k: c for k, c in res["checks"].items() if not c.get("ok")}
        with monkeypatch.context() as mp:                   # compared under this release's rules, it fails
            mp.setattr(cert, "_rules_of", lambda wb: contextlib.nullcontext())
            assert cert.verify(str(tmp_path / "cert" / "certificate.json"), str(tmp_path / "p.npy"))["verified"] is False
    finally:
        os.chdir(cwd)


def test_the_rules_of_each_release():
    """Held in a context variable: another thread is not touched, and the package's own functions stay in place."""
    import threading
    from oe_inferencex import cli
    from oe_inferencex import certificate as cert
    from oe_inferencex import estimate as est
    seen = []
    with cert._rules_of({"version": "1.17.0"}):
        assert est.null_count(100, 0.29) == 29 and est.null_count(300, 0.3333333333333333) == 101
        t = threading.Thread(target=lambda: seen.append(est.null_count(100, 0.29)))
        t.start()
        t.join()
    assert seen == [30]                                      # the other thread kept this release's rule
    with cert._rules_of({"version": "1.17.1"}):
        assert est.null_count(100, 0.29) == 30
        assert cli.class_words("good condition", ["good condition"]) == "good class"
    with cert._rules_of({"version": "1.17.2"}):
        assert cli.class_words("good condition", ["good condition"]) == "good condition"
    assert est.null_count(100, 0.29) == 30 and est.null_count(300, 0.3333333333333333) == 100
    assert cert._certified_sizes({"n_zone": 10}) <= cert._certified_sizes({"n_zone": 10})
    assert not cert._certified_sizes({"n_zone": 11}) <= cert._certified_sizes({"n_zone": 10})
    more = {"n_certified": 5, "per_condition": {"a": {"n_zone": 3}}}
    less = {"n_certified": 5, "per_condition": {"a": {"n_zone": 2}}}
    assert cert._certified_sizes(less) <= cert._certified_sizes(more)
    assert not cert._certified_sizes(more) <= cert._certified_sizes(less)


def test_a_bundle_carries_no_key_an_earlier_release_does_not_write(tmp_path):
    """Every 1.17.2 certificate failed 1.17.1's verify 'in edge'."""
    import shutil
    from oe_inferencex import certificate as cert
    shutil.copytree(os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", "class"), tmp_path / "c")
    zone = json.load(open(tmp_path / "c" / "zone.json"))
    zone["edge"] = "0.5"
    zone["condition_by"] = "map class"
    for e in zone["per_condition"].values():
        e["edge"] = "0.5"
    stripped = cert._strip_added(zone)
    assert "edge" not in stripped and "condition_by" not in stripped
    assert not any("edge" in e for e in stripped["per_condition"].values())


def test_a_gdal_virtual_path_is_refused_for_a_certificate(quick):
    import zipfile
    with zipfile.ZipFile(quick / "maps.zip", "w") as z:
        z.write(quick / "scores.tif", "scores.tif")
    vsi = f"/vsizip/{quick / 'maps.zip'}/scores.tif"
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", vsi, "--design", "random", "--budget", "315", "--target", "0.5", "--seed-from-map",
                     "--alpha", "0.05", "--out", "z.csv"]) == 0
        qs.label("z.csv")
        assert main(["certify", "z.csv"]) == 0
    with pytest.raises(SystemExit, match="a path GDAL reads itself"):
        main(["certificate", "z_zone.json", "--date", "2024-06-01", "--out", "zc"])


def test_a_condition_layer_with_an_alpha_band_is_certified(quick):
    """sample read it as one band since 1.17.2, and certificate then refused it as two."""
    import rasterio
    from rasterio.enums import ColorInterp
    with rasterio.open(quick / "scores.tif") as src:
        prof = src.profile
    h, w = prof["height"], prof["width"]
    prof.update(count=2, dtype="uint8", nodata=None)
    layer = (np.arange(w)[None, :] >= w // 2).astype(np.uint8).repeat(h, 0)
    with rasterio.open(quick / "cond.tif", "w", **prof) as dst:
        dst.colorinterp = [ColorInterp.gray, ColorInterp.alpha]
        dst.write(np.stack([layer, np.full((h, w), 255, np.uint8)]))
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--design", "condition", "--condition", "cond.tif", "--budget", "1000",
                     "--seed-from-map", "--alpha", "0.1", "--out", "k.csv"]) == 0
        qs.label("k.csv")
        assert main(["certify", "k.csv"]) == 0
        assert main(["certificate", "k_zone.json", "--date", "2024-06-01", "--out", "kc"]) == 0


def test_mcp_sample_limits_never_set_out_dir_to_the_csv(quick):
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    out = mcp_server.sample(scores=str(quick / "scores.tif"), budget=40, out_dir=str(quick / "o"), design="sequential",
                            alpha=0.1)
    assert "To add windows later" not in out["limits"] and "label from the top row down" in out["limits"]
    assert "extend=" in out["next"]                                   # the call is given there, as the tool takes it


def test_plan_prints_a_negative_cut_off_as_one_token(tmp_path):
    from oe_inferencex.cli import _sample_command
    import argparse
    a = argparse.Namespace(scores="a.npy", other="b.npy", patch=4, nodata=None, logits=False, threshold=-1e-05)
    assert "--threshold=-1e-05" in _sample_command(a, 50, other=True)


def test_boundary_next_command_for_logits_probabilities_and_a_small_map(tmp_path, capsys):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "l3.npy", rng.normal(0, 3, (3, 32, 32)).astype(np.float32))
    assert main(["boundary", str(tmp_path / "l3.npy"), "--width-px", "1", "--out", str(tmp_path / "b3.npy")]) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert " --logits " in line and "--budget 64 " in line           # 64 windows: a budget sample accepts
    np.save(tmp_path / "p.npy", rng.random((32, 32)).astype(np.float32))
    assert main(["boundary", str(tmp_path / "p.npy"), "--threshold", "0.5", "--width-px", "1",
                 "--out", str(tmp_path / "b.npy")]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith("--out to_label.csv")        # no note: sample reads it so


def test_class_words_hold_paths_first_and_only_names_that_carry_a_reworded_word():
    from oe_inferencex.cli import class_words
    path = "/a/Site conditions 2026/to_label.csv"
    assert class_words(f"certify with sample_csv={path} per input condition", [], [path]) == \
        f"certify with sample_csv={path} per map class"
    assert class_words("keep to_label.csv and conditions.json beside it", [], ["/x/y/conditions.json"]) == \
        "keep to_label.csv and conditions.json beside it"
    for names in (["input", "x"], ["condition", "x"], ["t", "g"]):
        assert class_words("stratified by input condition; each condition's interval", names) == \
            "stratified by map class; each class's interval"
    assert class_words("good condition 8.0%; per input condition", ["good condition"]) == \
        "good condition 8.0%; per map class"


def test_a_1_17_1_class_certificate_with_package_words_as_names_verifies(tmp_path):
    """Classes named "input", "condition" and "good condition" (1.17.1, stored): the rewording 1.17.2 first shipped
    turned 'Certified per map class' into 'Certified per input class', and verify failed on the note."""
    import shutil
    shutil.copytree(os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", "class_named"), tmp_path / "c")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qs.write_maps()
            main(["verify", "c/certificate.json", "--scores", "scores.tif", "--out", "v.json"])
        assert json.load(open(tmp_path / "v.json"))["verified"] is True
    finally:
        os.chdir(cwd)


def test_mcp_estimate_says_the_outside_note_once_under_the_class_design(quick):
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    _class_sample(quick)
    rows = (quick / "c.csv").read_text().splitlines()
    head = rows[0].split(",")
    w, mc = head.index("wrong"), head.index("map_class")
    rows = [rows[0]] + [",".join(("1" if r.split(",")[mc] == "1" else c) if j == w else c
                                 for j, c in enumerate(r.split(","))) for r in rows[1:]]
    (quick / "c.csv").write_text("\n".join(rows) + "\n")
    out = mcp_server.estimate(sample_csv=str(quick / "c.csv"))
    note = "It lies below the interval of class 1"
    assert note in out["conclusion"] and note not in out["limits"]



# ----------------------------------------------------------------------------- the third review of 1.17.2
def test_class_words_third_review_cases():
    from oe_inferencex.cli import class_words
    # a file name with no extension, or a fragment of the package's words, is not held inside them
    assert class_words("of this condition; see runs/condition", [], ["runs/condition"]) == \
        "of this class; see runs/condition"
    assert class_words("per input condition", [], ["t", "in"]) == "per map class"
    # a lone backtick in a name: no placeholder is left behind
    out = class_words("Labels per condition: farmer`s field 87, good condition 107", ["farmer`s field", "good condition"])
    assert "\x00" not in out and out == "Labels per class: farmer`s field 87, good condition 107"
    # a class named only with the package's words: the JSON's list of names keeps it, so it is still a key
    res = {"per_condition": {"condition": {}}, "outside_condition_intervals": ["condition"], "note": "per input condition"}
    worded = class_words(res, ["condition"])
    assert worded["outside_condition_intervals"] == ["condition"] and worded["note"] == "per map class"
    assert class_words("Answers per class or per condition each rest", ["x"]) == "Answers per class each rest"


def test_decide_words_a_bundles_class_design_zone_by_class(tmp_path):
    """A bundle's zone.json carries no condition_by (so that 1.17.1 verifies it); decide read its classes as input
    conditions. The sample's sidecar beside it says the design."""
    import shutil
    from oe_inferencex.decide import decide
    shutil.copytree(os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", "class"), tmp_path / "c")
    zone = json.load(open(tmp_path / "c" / "zone.json"))
    assert "condition_by" not in zone
    a = decide(str(tmp_path / "c" / "zone.json"), ["trusted_share"])["answers"]["trusted_share"]
    assert "input condition" not in a["because"] and "map class" in a["because"]


def test_certificate_refuses_what_verify_cannot_read_from_its_own_paths(quick):
    """A CRS kept only in an .aux.xml, and a map reached through a link, were certified and then failed verify from
    the certificate's paths."""
    import rasterio
    from rasterio.crs import CRS
    with rasterio.open(quick / "scores.tif") as src:
        prof, a = src.profile, src.read()
    prof.update(crs=None)
    with rasterio.open(quick / "nocrs.tif", "w", **prof) as dst:
        dst.write(a)
    (quick / "nocrs.tif.aux.xml").write_text(f"<PAMDataset><SRS>{CRS.from_epsg(32633).to_wkt()}</SRS></PAMDataset>")
    os.symlink(quick / "scores.tif", quick / "link.tif")
    for name in ("nocrs.tif", "link.tif"):
        stem = name[:-4]
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", name, "--design", "random", "--budget", "315", "--target", "0.5",
                         "--seed-from-map", "--alpha", "0.05", "--out", f"{stem}.csv"]) == 0
            qs.label(f"{stem}.csv")
            assert main(["certify", f"{stem}.csv"]) == 0
        with pytest.raises(SystemExit, match="reads differently without the files beside it|plain local GeoTIFF"):
            main(["certificate", f"{stem}_zone.json", "--date", "2024-06-01", "--out", f"{stem}_cert"])


def test_extend_refuses_a_sample_whose_labelled_windows_changed(quick):
    """On a 1.17.1 sample whose raster's mask 1.17.2 reads, the extension kept the labels and rewrote map_class under
    them. Here the map is changed under a labelled sequential sample the same way."""
    import rasterio
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--design", "sequential", "--budget", "40", "--alpha", "0.1",
                     "--out", "q.csv"]) == 0
    qs.label("q.csv")
    with rasterio.open(quick / "scores.tif", "r+") as dst:
        a = dst.read()
        a[[0, 1]] = a[[1, 0]]                                 # two classes swap: same windows, other classes
        dst.write(a)
    with pytest.raises(SystemExit, match="no longer show the confidence and class"):
        main(["sample", "scores.tif", "--design", "sequential", "--budget", "60", "--extend", "q.csv", "--out", "q.csv"])


def test_boundary_reads_an_integer_map_as_a_class_map_whatever_the_cut(tmp_path, capsys):
    np.save(tmp_path / "forest.npy", (np.arange(32)[None, :] // 16).repeat(32, 0).astype(np.uint8))
    assert main(["boundary", str(tmp_path / "forest.npy"), "--threshold", "0.3", "--width-px", "1",
                 "--out", str(tmp_path / "b.npy")]) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert "--confidence BAND" in line and "cut it at" not in line


def test_mcp_estimate_on_a_class_named_with_the_packages_words(quick):
    """outside_condition_intervals was reworded to 'class', not a key of per_condition: a KeyError."""
    pytest.importorskip("mcp")
    from oe_inferencex import mcp_server
    _class_sample(quick, ["0=water", "1=condition", "2=crops", "3=Input conditions"])
    rows = (quick / "c.csv").read_text().splitlines()
    head = rows[0].split(",")
    w, mc = head.index("wrong"), head.index("map_class")
    rows = [rows[0]] + [",".join(("1" if r.split(",")[mc] == "1" else c) if j == w else c
                                 for j, c in enumerate(r.split(","))) for r in rows[1:]]
    (quick / "c.csv").write_text("\n".join(rows) + "\n")
    out = mcp_server.estimate(sample_csv=str(quick / "c.csv"))
    assert "condition" in json.load(open(quick / "c_estimate.json"))["outside_condition_intervals"]
    assert out["conclusion"]


# ----------------------------------------------------------------------------- the fourth review of 1.17.2
def test_a_certificate_at_alpha_029_is_issued_and_verified_with_this_releases_rules(tmp_path, monkeypatch):
    """verify's first rerun, meant to use this release's rules, used 1.17.0's (_rules_of(None) read the release as
    0.0.0), so a 1.17.2 certificate at alpha 0.29 on 400 windows (zones of 100, 200 and 400) was refused. Stamped
    1.17.2 here whatever the installed metadata says: stamped 1.17.1, the second rerun hid the fault."""
    import math
    import oe_inferencex
    from oe_inferencex import certificate as cert
    monkeypatch.setattr(oe_inferencex, "__version__", "1.17.2")
    from oe_inferencex import estimate as est
    rng = np.random.default_rng(5)
    p = rng.random((2, 80, 80)).astype(np.float32)
    np.save(tmp_path / "p.npy", p / p.sum(0))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "p.npy", "--design", "random", "--budget", "200", "--seed-from-map", "--alpha", "0.29",
                         "--out", "s.csv"]) == 0
            rows = (tmp_path / "s.csv").read_text().splitlines()
            w = rows[0].split(",").index("wrong")
            (tmp_path / "s.csv").write_text("\n".join([rows[0]] + [",".join(
                c if j != w else str(int(rng.random() < 0.1)) for j, c in enumerate(r.split(","))) for r in rows[1:]])
                + "\n")
            assert main(["certify", "s.csv"]) == 0
            sizes = [lv["n_zone"] for lv in json.load(open("s_zone.json"))["levels"]]
            assert any(math.floor(0.29 * n) + 1 != est.null_count(n, 0.29) for n in sizes)      # the case is exercised
            assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
        assert cert.verify(str(tmp_path / "cert" / "certificate.json"), str(tmp_path / "p.npy"))["verified"] is True
    finally:
        os.chdir(cwd)


def test_a_1_17_0_certificate_that_certifies_more_by_its_float_count_fails(tmp_path, monkeypatch):
    """At alpha 1/3 1.17.0's float null count is the larger, so it can certify more: here 20% of the map where the exact
    count certifies 15%. Issued as 1.17.0 issued it, it is not taken on its word."""
    import csv as _csv
    import shutil
    import oe_inferencex
    from oe_inferencex import certificate as cert
    p = np.random.default_rng(0).random((2, 256, 256)).astype(np.float32)
    np.save(tmp_path / "p.npy", p / p.sum(0))
    a = repr(0.3333333333333333)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "p.npy", "--design", "random", "--budget", "300", "--seed-from-map", "--alpha", a,
                         "--delta", "0.05", "--out", "s.csv"]) == 0
            rows = list(_csv.DictReader(open("s.csv")))
            for r in rows:
                r["wrong"] = "0"
            head = list(rows[0])
            with open("s.csv", "w", newline="") as f:
                wr = _csv.DictWriter(f, head)
                wr.writeheader()
                wr.writerows(rows)
            shutil.copyfile("s.json", "t.json")
            shutil.copyfile("s.csv", "t.csv")
            assert main(["certify", "t.csv", "--out", "t0.json"]) == 0
        b = {round(lv["coverage"], 4): lv["n_labelled_inside"] for lv in json.load(open("t0.json"))["levels"]}
        order = sorted(range(len(rows)), key=lambda i: -float(rows[i]["confidence"]))
        wrong = set(order[b[0.2] - 13:b[0.2]]) | set(order[b[0.2]:b[0.25]])
        for i, r in enumerate(rows):
            r["wrong"] = str(int(i in wrong))
        with open("s.csv", "w", newline="") as f:
            wr = _csv.DictWriter(f, head)
            wr.writeheader()
            wr.writerows(rows)
        with monkeypatch.context() as mp:                   # issued as 1.17.0 issued it
            mp.setattr(oe_inferencex, "__version__", "1.17.0")
            real = cert._rules_of
            mp.setattr(cert, "_rules_of", lambda wb: real({"version": "1.17.0"}))
            with real({"version": "1.17.0"}), contextlib.redirect_stdout(io.StringIO()):
                assert main(["certify", "s.csv"]) == 0
                assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
        zone = json.load(open("cert/zone.json"))
        assert zone["coverage"] == 0.2 and zone["written_by"]["version"] == "1.17.0"
        res = cert.verify(str(tmp_path / "cert" / "certificate.json"), str(tmp_path / "p.npy"))
        assert res["verified"] is False and "certifies more under the rules of the release it names" in \
            res["checks"]["certify"]["detail"]
    finally:
        os.chdir(cwd)


def test_the_fourth_reviews_smaller_cases(quick, tmp_path, capsys):
    from oe_inferencex.cli import class_words
    from oe_inferencex import certificate as cert
    # a NUL of the text's own does not hang the restore
    assert class_words("per input condition; a\x00b", []) == "per map class; a\x00b"
    # backticks in two names open no span across the text between them
    names = ["farmer`s field", "farmer`s plot"]
    assert class_words("farmer`s field 87, per input condition, farmer`s plot 82", names) == \
        "farmer`s field 87, per map class, farmer`s plot 82"
    # a class named "condition" leaves the design's own values worded; its list entries are kept
    out = class_words({"design": "condition", "outside_condition_intervals": ["condition"]}, ["condition", "x"])
    assert out == {"design": "class", "outside_condition_intervals": ["condition"]}
    # a percent-cover integer map cut at 50 is no class map to boundary's next command
    np.save(tmp_path / "cover.npy", (np.arange(32)[None, :] * 3).repeat(32, 0).astype(np.int16))
    assert main(["boundary", str(tmp_path / "cover.npy"), "--threshold", "50", "--width-px", "1",
                 "--out", str(tmp_path / "b.npy")]) == 0
    assert "--confidence BAND" not in capsys.readouterr().out.splitlines()[-1]
    # the CRS compared as a CRS: the same CRS in another spelling beside the file is the file's
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        crs = src.crs
    shutil_copy = __import__("shutil").copyfile
    shutil_copy(quick / "scores.tif", quick / "esri.tif")
    (quick / "esri.tif.aux.xml").write_text(f"<PAMDataset><SRS>{crs.to_wkt(version='WKT1_ESRI')}</SRS></PAMDataset>")
    assert cert._reads_alone(str(quick / "esri.tif"))


# ----------------------------------------------------------------------------- the fifth review of 1.17.2
def test_a_non_epsg_crs_repeated_beside_the_file_is_refused(quick):
    """The same non-EPSG CRS in an .aux.xml passed the sidecar check as a CRS, and verify, which compares it as GDAL
    spells it, then failed the certificate from its own paths."""
    import rasterio
    from rasterio.crs import CRS
    tm = CRS.from_proj4("+proj=tmerc +lon_0=13.5 +k=0.9996 +x_0=500000 +datum=WGS84 +units=m")
    with rasterio.open(quick / "scores.tif") as src:
        prof, a = src.profile, src.read()
    prof.update(crs=tm)
    with rasterio.open(quick / "tm.tif", "w", **prof) as dst:
        dst.write(a)
    (quick / "tm.tif.aux.xml").write_text(f"<PAMDataset><SRS>{tm.to_wkt(version='WKT1_ESRI')}</SRS></PAMDataset>")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "tm.tif", "--design", "random", "--budget", "315", "--target", "0.5", "--seed-from-map",
                     "--alpha", "0.05", "--out", "t.csv"]) == 0
        qs.label("t.csv")
        assert main(["certify", "t.csv"]) == 0
    with pytest.raises(SystemExit, match="reads differently without the files beside it|not from the paths it names"):
        main(["certificate", "t_zone.json", "--date", "2024-06-01", "--out", "tc"])


def test_links_by_role_a_linked_condition_layer_is_carried_a_linked_map_is_refused(quick):
    import rasterio
    with rasterio.open(quick / "scores.tif") as src:
        prof, a = src.profile, src.read()
    os.makedirs(quick / "layers")
    prof1 = dict(prof, count=1, dtype="int32", nodata=-1)
    with rasterio.open(quick / "layers" / "cond.tif", "w", **prof1) as dst:
        dst.write((np.arange(a.shape[2])[None, :] >= a.shape[2] // 2).astype(np.int32).repeat(a.shape[1], 0), 1)
    os.symlink(quick / "layers" / "cond.tif", quick / "cond.tif")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "scores.tif", "--design", "condition", "--condition", "cond.tif", "--budget", "1000",
                     "--seed-from-map", "--alpha", "0.1", "--out", "k.csv"]) == 0
        qs.label("k.csv")
        assert main(["certify", "k.csv"]) == 0
        assert main(["certificate", "k_zone.json", "--date", "2024-06-01", "--out", "kc"]) == 0
        main(["verify", "kc/certificate.json", "--out", "kv.json"])
    assert json.load(open(quick / "kv.json"))["verified"] is True
    # a class map reached through a link, used as its own condition layer: the map's link is refused
    cls = a.argmax(0).astype(np.uint8)
    conf = a.max(0).astype(np.float32)
    with rasterio.open(quick / "layers" / "cls.tif", "w", **dict(prof, count=1, dtype="uint8", nodata=None)) as dst:
        dst.write(cls, 1)
    with rasterio.open(quick / "conf.tif", "w", **dict(prof, count=1, dtype="float32", nodata=None)) as dst:
        dst.write(conf, 1)
    os.symlink(quick / "layers" / "cls.tif", quick / "cl.tif")
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["sample", "cl.tif", "--confidence", "conf.tif", "--design", "condition", "--condition", "cl.tif",
                     "--budget", "1000", "--seed-from-map", "--alpha", "0.2", "--out", "p.csv"]) == 0
        qs.label("p.csv")
        assert main(["certify", "p.csv"]) == 0
    with pytest.raises(SystemExit, match="plain local GeoTIFF|not from the paths it names"):
        main(["certificate", "p_zone.json", "--date", "2024-06-01", "--out", "pc"])


def test_verify_from_threads_restores_the_output(tmp_path):
    """Two verify calls from two threads restored each other's stream and left a dead one in place."""
    import shutil
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from oe_inferencex import certificate as cert
    for kind in ("random", "sequential"):
        shutil.copytree(os.path.join(ROOT, "tests", "golden", "certificates_1_17_1", kind), tmp_path / kind)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qs.write_maps()
        before = sys.stdout
        for _ in range(2):
            with ThreadPoolExecutor(4) as ex:
                res = list(ex.map(lambda k: cert.verify(str(tmp_path / k / "certificate.json"),
                                                        str(tmp_path / "scores.tif")), ["random", "sequential"] * 2))
            assert all(r["verified"] for r in res) and sys.stdout is before
    finally:
        os.chdir(cwd)


def test_class_words_fifth_review_cases():
    from oe_inferencex.cli import class_words
    for names in (["class"], ["or"], ["per class"]):
        assert class_words("Answers per class or per condition each rest", names) == "Answers per class each rest"
    assert class_words("`condition 87; 20% of the condition", ["`condition"]) == "`condition 87; 20% of the class"
    assert class_words("a\x000\x00 4.0%; per input condition", ["a\x000\x00", "water"]) == "a\x000\x00 4.0%; per map class"

"""exp98's measures and grades before the preregistration is frozen and before any real run
(docs/plan/awf_deployment.md). The synthetic smoke writes a map read in two overlapping parts (two from-olmoearth
directories, two deployment dataset roots, as the real run is read), its condition layer, AWF-style label windows
(one in another CRS, two outside the map, two on uncovered pixels, two dropped by the label rule, eleven in both
parts), exp89's replica units, and runs every part on them; the values below are computed by hand from that design
(exp98_awf_deployment.py, the comment above SMOKE_GRID). The grades are also checked on planted numbers at their
thresholds and floors, the readers on small cases, the guard that refuses the run while the page is a draft, and that
nothing written carries a position. Needs rasterio (the geo extra); skipped without it."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp98_awf_deployment as e98   # noqa: E402

X = e98.SMOKE_EXPECTED


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("exp98"))


@pytest.fixture(scope="module")
def smoke(out_dir):
    return e98.smoke(out_dir=out_dir, log=lambda *a: None)


# ----------------------------------------------------------------------------- structure
def test_the_summary_holds_every_part(smoke, out_dir):
    for k in ("experiment", "prereg_page", "inputs", "labels", "mapping", "graded_validation", "train_in_sample",
              "where_the_points_sit", "plan_reported", "inputs_reported", "prereg", "inventory"):
        assert k in smoke, k
    for p in e98.PREDICTIONS:
        row = smoke["prereg"][p]
        assert {"rule", "threshold", "value", "graded"} <= set(row), p
        assert row["threshold"] == e98.THRESHOLDS[p]
    g = smoke["graded_validation"]
    for k in ("accuracy", "ai2", "ranking", "replica", "by_condition", "assess_package_path", "boundary_reported"):
        assert k in g, k
    with open(os.path.join(out_dir, "exp98_summary_smoke.json")) as f:
        assert json.load(f)["prereg"]["P3"]["value"] == pytest.approx(X["auroc"])
    z = np.load(os.path.join(out_dir, "exp98_units_smoke.npz"))
    assert set(z.files) == {"names", "split", "label", "covered", "pred_deployed", "suspicion", "p1", "condition",
                            "burned", "boundary", "input_identical"}
    assert int(z["covered"].sum()) == 50


def test_nothing_written_carries_a_position(smoke, out_dir):
    e98.check_no_coordinates(smoke)
    with open(os.path.join(out_dir, "exp98_summary_smoke.json")) as f:
        written = json.load(f)

    def numbers(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from numbers(v)
        elif isinstance(o, list):
            for v in o:
                yield from numbers(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield o
    # the synthetic grid's origin in metres and in projection pixels, and its far corner; no number written is one
    origin = {500000.0, 9000000.0, 50000.0, -900000.0, 501600.0, 8998800.0, 50160.0, -899880.0}
    assert not origin & set(float(v) for v in numbers(written))
    with pytest.raises(ValueError, match="position"):
        e98.check_no_coordinates({"grids": [{"transform": [10, 0, 1, 0, -10, 2]}]})
    with pytest.raises(ValueError, match="position"):
        e98.check_no_coordinates({"a": {"lonlat": [1, 2]}})


# ----------------------------------------------------------------------------- hand-computed values
def test_labels_and_mapping(smoke):
    lab, mp = smoke["labels"], smoke["mapping"]
    assert lab["by_split"] == {"val": 44, "train": 10}
    assert sum(lab["dropped"].values()) == X["dropped"]
    assert lab["dropped"] == {"2 labelled pixels": 1, "0 labelled pixels": 1}
    assert lab["time_range_matches_request"]["n_matching"] == 54
    assert mp["by_split"]["val"] == X["mapping_val"]
    assert mp["by_split"]["train"] == {"n": 10, "outside_every_grid": 0, "in_grid_uncovered": 0, "covered": 10}
    assert mp["reprojected"] == X["reprojected"] and mp["same_crs"] == 51
    assert mp["largest_offset_from_pixel_middle_px"] < 1e-6, "every label pixel's centre is a score pixel's centre"
    assert smoke["inventory"]["mapping"] == mp, "the inventory maps the points as the run does"


def test_two_parts_read_as_one_map(smoke):
    o = smoke["mapping"]["overlapping_grids"]
    assert {k: o[k] for k in X["overlap"] if k in o} == {k: v for k, v in X["overlap"].items() if k in o}
    assert o["covered_only_in_a_later_grid"] == 5, "a point on a pixel NaN in part 0 is read from part 1"
    w = smoke["where_the_points_sit"]
    assert w["n_map_pixels"] == 16000, "a pixel covered in both parts counts once"
    assert w["n_pixels_counted_once_of_overlaps"] == X["overlap"]["pixels_counted_once"]
    assert w["n_grid_overlaps_not_deduplicated"] == 0
    assert [g["label"] for g in smoke["inputs"]["grids"]] == ["0:32737", "1:32737"]
    assert sum(g["n_val_covered"] for g in smoke["inputs"]["grids"]) == X["n_val"]


def test_condition_names_that_conflict_across_parts_are_refused(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    dirs = []
    for k, name in enumerate(("sentinel2:all:clear", "sentinel2:none")):
        d = tmp_path / f"p{k}"
        d.mkdir()
        with rasterio.open(d / "scores_32737.tif", "w", driver="GTiff", height=2, width=2, count=10, dtype="float32",
                           crs="EPSG:32737", transform=from_origin(0, 0, 10, 10)) as dst:
            dst.write(np.full((10, 2, 2), 0.1, np.float32))
        (d / "olmoearth_conditions.json").write_text(json.dumps({"codes": {"0": name}}))
        dirs.append(str(d))
    with pytest.raises(ValueError, match="not read with the same inputs"):
        e98.find_outputs(dirs)
    entries, names, _ = e98.find_outputs(dirs[:1])
    assert names == {0: "sentinel2:all:clear"} and entries[0]["label"] == "32737"


def test_part_a_against_the_replica(smoke):
    g = smoke["graded_validation"]
    assert g["n"] == X["n_val"] and g["n_clusters"] == 6
    assert g["accuracy"]["accuracy_window"] == pytest.approx(X["accuracy"])
    assert g["accuracy"]["n_errors"] == X["n_errors"]
    assert g["readings"]["n_pred_untrained"] == X["n_pred_untrained"]
    r = g["replica"]
    assert r["n_matched"] == 40 and r["label_mismatches"] == 0 and r["cluster_mismatches"] == 0
    assert r["accuracy_replica"] == pytest.approx(X["replica_accuracy"])
    assert r["difference"] == pytest.approx(X["accuracy"] - X["replica_accuracy"])
    assert r["agreement"] == pytest.approx(X["agreement"]) and r["n_same_class"] == 36
    d = r["discordant"]
    assert (d["deployed_right_replica_wrong"], d["replica_right_deployed_wrong"]) == (1, 3)
    assert d["sign_test_two_sided_p"] == pytest.approx(2 * (1 + 4) / 16)        # 1 against 3 of 4, two-sided
    assert g["ai2"]["matched"] is False and g["ai2"]["difference"] == pytest.approx(0.80 - 0.895)
    same = g["replica_same_crs_reported"]                      # the reprojected point (rank 39, an error) left out
    assert same["n_matched"] == 39 and same["n_left_out"] == 1
    assert same["accuracy_deployed"] == pytest.approx(32 / 39) and same["agreement"] == pytest.approx(35 / 39)
    iv = r["p1_equivalence_interval"]
    assert iv["level"] == 0.90 and iv["n_clusters"] == 6
    assert iv["lo"] <= r["difference"] + 1e-12 <= iv["hi"] + 1e-12
    assert iv["max_abs_end"] == pytest.approx(max(abs(iv["lo"]), abs(iv["hi"])))
    assert r["within_2_points_reported"] is False


def test_part_b_ranking(smoke):
    rk = smoke["graded_validation"]["ranking"]
    assert rk["auroc"] == pytest.approx(X["auroc"])
    assert rk["capture"] == pytest.approx(X["capture"])
    assert rk["capture_random"] == pytest.approx({"0.05": 2 / 40, "0.1": 4 / 40, "0.2": 8 / 40})
    assert rk["ceiling"] == pytest.approx({"0.05": 2 / 8, "0.1": 4 / 8, "0.2": 1.0})


def test_the_package_path_differs_only_by_the_untrained_channel(smoke):
    a = smoke["graded_validation"]["assess_package_path"]
    assert a["n_windows_scored"] == 40
    assert a["error_rate"] == pytest.approx(X["assess_errors"] / 40)
    assert a["same_order_as_ranking"] is True


def test_part_c_conditions(smoke):
    bc = smoke["graded_validation"]["by_condition"]
    got = {v["code"]: (v["n_points"], v["n_errors"]) for v in bc.values()}
    assert got == X["by_condition"]
    assert bc["sentinel2:all:clear"]["share_of_map"] == pytest.approx(50 * 160 / 16000)
    assert bc["unrecorded"]["share_of_map"] == pytest.approx(2 / 16000)


def test_train_points_are_reported_apart(smoke):
    t = smoke["train_in_sample"]
    assert t["n"] == 10 and t["accuracy"] == pytest.approx(X["train_accuracy"])
    assert smoke["prereg"]["floors"]["n_validation_points"] == 40, "training points never enter the graded set"


def test_part_e_where_the_points_sit(smoke):
    w = smoke["where_the_points_sit"]
    assert w["n_map_pixels"] == 16000
    assert w["points_mean_percentile"] == pytest.approx(X["mean_percentile"])
    assert w["map_class_share"]["shrubland_savanna"] + w["map_class_share"]["grassland_barren"] > 0.99
    assert w["reweighted_reported"]["estimate"] is None, "the tied background leaves strata without points"


def test_part_f_plan_is_reported_not_graded(smoke):
    pl = smoke["plan_reported"]
    assert pl["n_windows"] == 16000 and pl["width"] == 0.10 and pl["alpha"] == 0.10
    assert set(pl["error_rate"]) == {"points", "exp89_replica", "default_0.5"}
    assert pl["error_rate"]["points"]["error_rate"] == pytest.approx(0.2)
    assert pl["zone"]["1"]["refusal"], "the points' 20% error is above alpha, so the whole map cannot be planned"
    assert pl["certify_min_labels"] == 22
    assert "--design sequential" in pl["sequential"]["command"]
    assert all("holds" not in v for v in pl["zone"].values())


def test_part_g_inputs(smoke):
    g = smoke["inputs_reported"]
    assert g["n_compared"] == X["inputs"]["n_compared"]
    assert g["n_identical_all_groups"] == X["inputs"]["identical"]
    assert g["n_points_without_deployed_window"] == X["inputs"]["no_window"]
    assert g["n_points_without_label_imagery"] == X["inputs"]["no_label_imagery"]
    assert g["n_points_without_deployed_imagery_read"] == 0 and g["n_points_incomplete"] == 0
    assert g["n_points_in_two_deployed_windows"] == X["inputs"]["in_two_windows"]
    assert g["config"]["n_deployment_roots"] == 2 and g["config"]["deployment_roots_share_one_config"] is True
    assert g["n_identical_any_order"] == 37 and g["n_identical_reversed_order"] == 0
    assert g["share_identical_by_group"][5] == pytest.approx(37 / 39)
    assert [d["key"] for d in g["config"]["differences"]] == ["band_sets"]
    assert g["validation_points_by_input"]["not_identical"] == {"n_val": 2, "error_rate": 0.0,
                                                                "agreement_with_replica": 1.0}
    inv = smoke["inventory"]["inputs_reported"]
    assert "validation_points_by_input" not in inv, "the inventory reads no outcome"
    assert inv["n_identical_all_groups"] == X["inputs"]["identical"]


def test_part_h_burned_area(smoke, out_dir):
    h = smoke["burned_area_reported"]
    assert h["status"].startswith("report-only") and "holds" not in json.dumps(h)
    names = {c: e98.burn_name(c, e98.SMOKE_BURN_NAMES) for c in (-1, 0, 1)}
    assert names[0] == "mcd64a1:no-burn-2023:sep-not-read" and h["months_missing"] == ["2023-09"]
    pts = h["validation_points"]
    assert {c: (pts[names[c]]["n_points"], pts[names[c]]["n_errors"]) for c in names} == X["burned"]["points"]
    w = h["windows"]
    assert {c: w["per_code"][names[c]]["n_windows"] for c in names} == X["burned"]["windows"]
    assert w["n_windows"] == 1000
    assert w["n_windows_dropped_overlapping_an_earlier_grid"] == X["burned"]["dropped_overlap"]
    assert {c: h["pixels"][names[c]]["n"] for c in names} == X["burned"]["pixels"]
    # 17 windows hold a point less confident than the background's 0.6 (ranks 0-16: four burned, ten unrecorded,
    # three not burned); the other 83 of the least confident 100 come from the 960 windows tied at the background
    tied = {-1: 70, 0: 420, 1: 470}
    above = {-1: 10, 0: 3, 1: 4}
    assert w["n_tied_at_cutoff"] == 960 and w["n_tied_taken"] == 83
    for c in names:
        share = (above[c] + 83 * tied[c] / 960) / 100
        assert w["per_code"][names[c]]["share_of_least_confident"] == pytest.approx(share)
        assert w["per_code"][names[c]]["confidence_quartiles"]["0.5"] == pytest.approx(0.6)
    z = np.load(os.path.join(out_dir, "exp98_units_smoke.npz"))
    v = (z["split"] == "val") & z["covered"]
    assert sorted(np.unique(z["burned"][v]).tolist()) == [-1, 0, 1]


def _analyse_smoke(tmp_path):
    paths = e98.make_smoke_inputs(str(tmp_path))
    rec, units = e98.analyse(paths["scores"], paths["labels"], replica_path=paths["replica"],
                             n_boot=e98.SMOKE_BOOT, zone_draws=e98.SMOKE_ZONE_DRAWS,
                             max_labels=e98.SMOKE_MAX_LABELS, log=lambda *a: None, **e98.SMOKE_FLOORS)
    return paths, rec, units


def _graded_as_designed(rec, units):
    assert {p: rec["prereg"][p]["holds"] for p in e98.PREDICTIONS} == X["holds"]
    assert all(rec["prereg"][p]["graded"] for p in e98.PREDICTIONS)
    assert (units["burned"] == -2).all(), "Part H left out: no point carries a burn code"
    e98.check_no_coordinates(rec)


@pytest.mark.parametrize("fault", ["unreadable_file", "names_json", "window_refusal", "point_read"])
def test_a_failing_burned_layer_never_stops_the_graded_run(tmp_path, monkeypatch, fault):
    """Part H is report-only (docs/plan/awf_deployment.md, step 10): an unreadable burned_<EPSG>.tif, an unreadable
    names file, a refusal from the window pooling or a failed point read leaves Part H out and P1-P5 are graded."""
    if fault == "unreadable_file":
        orig = e98.make_smoke_inputs

        def broken(root):
            paths = orig(root)
            f = os.path.join(paths["scores"][1], f"burned_{e98.SMOKE_GRID['epsg']}.tif")
            with open(f, "wb") as fh:
                fh.write(b"not a tiff")
            return paths
        monkeypatch.setattr(e98, "make_smoke_inputs", broken)
    elif fault == "names_json":
        orig = e98.make_smoke_inputs

        def broken(root):
            paths = orig(root)
            with open(os.path.join(paths["scores"][0], e98.eb.NAMES_JSON), "w") as fh:
                fh.write("{not json")
            return paths
        monkeypatch.setattr(e98, "make_smoke_inputs", broken)
    elif fault == "window_refusal":
        def refuse(*a, **k):
            raise ValueError("planted refusal")
        monkeypatch.setattr(e98, "burn_windows", refuse)      # as burn_windows re-raises pool_condition's
    else:
        def fail(*a, **k):
            raise OSError("planted read failure")
        monkeypatch.setattr(e98, "codes_at", fail)
    _, rec, units = _analyse_smoke(tmp_path)
    _graded_as_designed(rec, units)
    h = rec["burned_area_reported"]
    if fault == "point_read":
        assert "planted read failure" in h["error"]
    else:
        assert "windows" not in h and "validation_points" not in h, h
        assert {"unreadable_file": "could not be opened", "names_json": "names the codes differently",
                "window_refusal": "planted refusal"}[fault] in h["note"]


# ----------------------------------------------------------------------------- the grades
def test_the_smoke_grades_as_designed(smoke):
    pr = smoke["prereg"]
    assert {p: pr[p]["holds"] for p in e98.PREDICTIONS} == X["holds"]
    assert all(pr[p]["graded"] for p in e98.PREDICTIONS)
    iv = smoke["graded_validation"]["replica"]["p1_equivalence_interval"]
    assert pr["P1"]["value"] == pytest.approx(iv["max_abs_end"]) and iv["max_abs_end"] > e98.THRESHOLDS["P1"]
    assert pr["P1"]["point_difference_reported"] == pytest.approx(X["accuracy"] - X["replica_accuracy"])
    bs = smoke["where_the_points_sit"]["points_mean_percentile_bootstrap"]
    assert pr["P5"]["lower_one_sided_95"] == pytest.approx(bs["lower_one_sided_95"]) and bs["n_clusters"] == 6
    assert bs["lower_one_sided_95"] <= pr["P5"]["value"] <= bs["hi_one_sided_95"]


def _planted(**kw):
    m = {"n_val": 250, "n_errors": 24, "p1_interval_max_abs": 0.05, "p1_difference": 0.0, "p2_agreement": 0.90,
         "p3_auroc": 0.75, "p4_capture_10": 0.25, "p5_mean_percentile": 0.55, "p5_lower_95": 0.5001}
    m.update(kw)
    return e98.grade(m)


def test_grades_at_their_thresholds():
    g = _planted()
    assert all(g[p]["graded"] and g[p]["holds"] for p in e98.PREDICTIONS), "a value at its threshold holds"
    g = _planted(p1_interval_max_abs=0.0501, p2_agreement=0.8999, p3_auroc=0.7499, p4_capture_10=0.2499,
                 p5_mean_percentile=0.5499)
    assert not any(g[p]["holds"] for p in e98.PREDICTIONS)
    g = _planted(p1_interval_max_abs=0.051, p1_difference=0.0)
    assert g["P1"]["holds"] is False, "P1 is an equivalence test: a point difference of 0 with a wide interval fails"
    g = _planted(p5_mean_percentile=0.60, p5_lower_95=0.50)
    assert g["P5"]["holds"] is False, "P5 needs its lower bound above 0.5, not only its point value"


def test_grades_below_their_floors_are_reported_not_graded():
    g = _planted(n_val=e98.MIN_VAL - 1)
    assert not any(g[p]["graded"] for p in e98.PREDICTIONS)
    assert all("holds" not in g[p] and g[p]["value"] is not None for p in e98.PREDICTIONS)
    g = _planted(n_errors=e98.MIN_ERRORS - 1)
    assert [p for p in e98.PREDICTIONS if g[p]["graded"]] == ["P1", "P2", "P5"]
    g = _planted(p1_interval_max_abs=None, p5_mean_percentile=float("nan"))
    assert g["P1"]["graded"] is False and g["P1"]["reason"].startswith("not computable")
    assert g["P5"]["graded"] is False
    assert _planted(p5_lower_95=None)["P5"]["graded"] is False


def test_the_floors_match_the_in_area_count():
    # the page's "What is known before the run": 250 validation points inside the request geometry, 24 replica errors
    assert e98.MIN_VAL < 250 and e98.MIN_ERRORS < 24
    assert e98.THRESHOLDS["P1"] == 0.05 and e98.P1_LEVEL == 0.90


def test_the_run_is_refused_while_the_page_is_a_draft(tmp_path, capsys):
    assert e98.prereg_status("# exp98\n\n**Status: DRAFT, not frozen.** Written 8 October 2026.\n") == "draft"
    assert e98.prereg_status("# exp98\n\n**Status: frozen on 9 October 2026, before any run.**\n") == "frozen"
    if e98.prereg_status() != "frozen":
        rc = e98.main(["--scores", str(tmp_path), "--labels", str(tmp_path), "--out-dir", str(tmp_path)])
        assert rc == 3 and "refused" in capsys.readouterr().out
        assert not os.listdir(tmp_path)


# ----------------------------------------------------------------------------- the readers on small cases
def test_trained_confidence_is_the_renormalised_softmax():
    P = np.array([[0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0.8],       # predicted as the untrained channel
                  [0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0.0]])
    r = e98.readings(P)
    assert r["pred"].tolist() == [9, 0] and r["n_pred_untrained"] == 1
    assert r["p1"] == pytest.approx([0.1 / 0.2, 0.7])
    assert r["n_saturated_float32"] == 0


def test_percentile_counts_ties_half():
    sample = np.array([1.0, 2.0, 2.0, 3.0])                      # suspicion: higher is less confident
    assert e98.percentile_in_map(sample, np.array([2.0, 0.5, 3.5])) == pytest.approx([0.5, 1.0, 0.0])


def test_reweighting_by_the_maps_confidence_quintiles():
    sample = np.linspace(0, 1, 1000)
    u = np.array([0.05, 0.1, 0.25, 0.3, 0.45, 0.5, 0.65, 0.7, 0.85, 0.9])
    err = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1], dtype=float)
    r = e98.reweighted(err, u, sample, n_map=1_000_000)
    assert r["map_share"] == pytest.approx([0.2] * 5)
    assert r["points_per_stratum"] == [2] * 5
    assert r["estimate"] == pytest.approx(0.2 * (0 + 0 + 0.5 + 0.5 + 1))
    assert r["accuracy"] == pytest.approx(0.6)


def test_a_label_pixel_maps_to_the_score_pixel_holding_its_centre():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    grid = {"crs": CRS.from_epsg(32737), "transform": from_origin(1000.0, 2000.0, 10, 10), "shape": (50, 60)}
    pts = [{"crs": CRS.from_epsg(32737), "X": 1000.0 + 10 * 7 + 5, "Y": 2000.0 - 10 * 3 - 5},   # pixel (3, 7)
           {"crs": CRS.from_epsg(32737), "X": 1000.0 - 5, "Y": 2000.0 - 5},                     # outside
           {"crs": CRS.from_epsg(32637), "X": 1000.0 + 10 * 2 + 5, "Y": 2000.0 - 10 * 9 - 5 - 10_000_000}]
    gi, rows, cols, how, off = e98.locate(pts, [grid])
    assert gi.tolist() == [0, -1, 0]
    assert (rows[0], cols[0]) == (3, 7) and (rows[2], cols[2]) == (9, 2)
    assert how.tolist() == ["same_crs", "outside", "reprojected"]
    assert off[0] == 0.0 and off[2] < 1e-6

"""exp99 (docs/plan/awf_transfer.md) and the crosswalk it shares with exp98's Part I, before the preregistration is
frozen and before any 2017 map exists. The crosswalk is pinned by its sha256, so a change after it was fixed fails
here; the pinned sources by their hashes; the estimators against independent computations; the smoke (synthetic
sample, boundaries, request geometry and one window per plot, exp/exp99_transfer.py's comment above SMOKE_RECT)
against values from its design; the request geometry against Ai2's file structure; and nothing written carries a
position. No network. Needs rasterio, shapely and pyproj (the geo extra); skipped without them."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp99_transfer as e99            # noqa: E402
import timesync_awf_crosswalk as tsx    # noqa: E402
from oe_inferencex import estimate as est   # noqa: E402

X = e99.SMOKE_EXPECTED
CROSSWALK_SHA256 = "4c8b452896a880189fc51922037acbdb4abc21e05c496e0acc67000a860d1846"   # fixed 9 October 2026


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("exp99"))


@pytest.fixture(scope="module")
def smoke(out_dir):
    return e99.smoke(out_dir=out_dir, log=lambda *a: None)


# ----------------------------------------------------------------------------- the crosswalk, fixed
def test_the_crosswalk_is_the_one_fixed_before_any_map():
    rec = tsx.crosswalk_record()
    assert rec["sha256"] == CROSSWALK_SHA256, "the crosswalk changed after it was fixed; a change needs a deviation"
    with open(os.path.join(ROOT, "docs", "plan", "awf_transfer.md"), encoding="utf-8") as f:
        assert CROSSWALK_SHA256 in f.read(), "the preregistration states the crosswalk's sha256"


def test_lenient_errors_are_a_subset_of_strict_ones():
    for t in tsx.TIMESYNC_CLASSES:
        assert tsx.STRICT[t] in tsx.LENIENT[t]
    labels = np.repeat(np.array(tsx.TIMESYNC_CLASSES, dtype=object), 10)
    pred = np.tile(np.arange(10), len(tsx.TIMESYNC_CLASSES))
    s, l = tsx.wrong(labels, pred, "strict"), tsx.wrong(labels, pred, "lenient")
    assert ((l == 1) <= (s == 1)).all()
    assert (s[pred == 9] == 1).all() and (l[pred == 9] == 1).all(), "the untrained channel is never accepted"
    assert int((s == 0).sum()) == len(tsx.TIMESYNC_CLASSES), "STRICT accepts exactly one class per label"


def test_the_classes_each_legend_lacks_are_handled():
    strict_targets = set(tsx.STRICT.values())
    assert "montane_forest" not in strict_targets and "lava_forest" not in strict_targets
    for t in ("Dense Forest", "Open Forest"):
        assert {"montane_forest", "lava_forest"} <= set(tsx.LENIENT[t])
    assert "woodland_forest" not in tsx.LENIENT["Wooded Grassland"], "below 15% against above 40% canopy: disjoint"
    assert tsx.STRICT["Open Forest"] == "shrubland_savanna" and tsx.STRICT["Otherland"] == "grassland_barren"
    with pytest.raises(ValueError, match="outside the crosswalk"):
        tsx.wrong(np.array(["Snow"], dtype=object), np.array([4]), "strict")


# ----------------------------------------------------------------------------- the pinned sources
def test_git_blob_id_is_gits(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert tsx.git_blob_sha1(str(p)) == "ce013625030ba8dba906f756967f9e9ca394464a"     # git hash-object
    src = {"bytes": 6, "git_blob_sha1": "ce013625030ba8dba906f756967f9e9ca394464a", "sha256": None}
    assert tsx.verify(str(p), src)[0] is True
    assert tsx.verify(str(p), dict(src, sha256="0" * 64))[0] is False
    assert tsx.verify(str(tmp_path / "absent"), src) == (False, "absent")


def test_the_local_sample_is_the_pinned_one():
    path = os.path.join(ROOT, "data", "breadth", tsx.TIMESYNC["file"])
    if not os.path.exists(path):
        pytest.skip("the East Africa sample is not in data/breadth here")
    assert tsx.verify(path, tsx.TIMESYNC)[0]


def test_the_area_rule_on_the_real_sample():
    """Runs where the AWF request geometry is at hand (E99_AWF_GEOMETRY): the counts the page fixed."""
    geom = os.environ.get("E99_AWF_GEOMETRY")
    csv = os.path.join(ROOT, "data", "breadth", tsx.TIMESYNC["file"])
    if not (geom and os.path.exists(geom) and os.path.exists(csv)):
        pytest.skip("set E99_AWF_GEOMETRY to Ai2's prediction_request_geometry.geojson at f3c9b0c8")
    P, rec, _ = e99.select(csv, geom)
    assert rec["matches_preregistration"] is True and rec["d_km"] == 100 and rec["n_selected"] == 309
    assert rec["counts_by_d_km"]["90"] == 282 and rec["counts_by_d_km"]["0"] == 47


# ----------------------------------------------------------------------------- the estimators
def test_the_rule_takes_the_smallest_buffer():
    d = np.array([0, 0, 3, 12, 12, 25, 40])
    r = tsx.area_rule(d, 5)
    assert r["d_km"] == 20 and r["n"] == 5 and r["counts_by_d_km"] == {0: 2, 10: 3, 20: 5}
    assert tsx.area_rule(d, 2)["d_km"] == 0
    with pytest.raises(ValueError):
        tsx.area_rule(d, 8, max_km=100)


def test_stratified_exact_is_the_union_of_each_countrys_exact_interval():
    k, n, area = {"kenya": 50, "tanzania": 30}, {"kenya": 219, "tanzania": 90}, {"kenya": 63500.0, "tanzania": 42600.0}
    r = tsx.stratified_exact(k, n, area)
    W = {c: area[c] / sum(area.values()) for c in area}
    lo = hi = 0.0
    for c in area:
        a, b = est.hypergeom_interval(k[c], n[c], int(round(area[c] * 1e4)), conf=1 - 0.05 / 2)
        lo += W[c] * a
        hi += W[c] * b
    assert r["estimate"] == pytest.approx(sum(W[c] * k[c] / n[c] for c in area))
    assert r["low"] == pytest.approx(lo, abs=1e-9) and r["high"] == pytest.approx(hi, abs=1e-9)
    assert r["strata_in_interval"] == 2 and r["low"] < r["estimate"] < r["high"]
    for c in area:
        assert (r["by_country"][c]["low"], r["by_country"][c]["high"]) == pytest.approx(
            est.hypergeom_interval(k[c], n[c], int(round(area[c] * 1e4))))


def test_ratio_estimator_matches_its_unit_level_formula():
    """Stehman's (2014) ratio estimator over two strata of 2,000 units each, a domain inside them."""
    A = {"kenya": 580000.0, "tanzania": 947000.0}
    k, n = {"kenya": 50, "tanzania": 30}, {"kenya": 219, "tanzania": 90}
    r = tsx.ratio_estimate(k, n, A)
    m = 2000
    Y = X_ = 0.0
    units = {}
    for c in A:
        y = np.r_[np.ones(k[c]), np.zeros(n[c] - k[c]), np.zeros(m - n[c])]
        x = np.r_[np.ones(n[c]), np.zeros(m - n[c])]
        units[c] = (y, x)
        Y += A[c] * y.mean()
        X_ += A[c] * x.mean()
    R = Y / X_
    v = sum(A[c] ** 2 * np.var(units[c][0] - R * units[c][1], ddof=1) / m for c in A) / X_ ** 2
    assert r["estimate"] == pytest.approx(R) and r["standard_error"] == pytest.approx(np.sqrt(v))


def test_gap_call():
    bar = 1 - e99.AI2_ACCURACY + e99.GAP_MARGIN
    assert e99.gap_call({"low": 0.3, "high": 0.4}, {"low": bar + 0.01, "high": 0.3})["call"] == "transfer gap"
    assert e99.gap_call({"low": 0.05, "high": bar - 0.01}, {"low": 0.02, "high": 0.1})["call"] == "no gap shown"
    assert e99.gap_call({"low": 0.1, "high": 0.3}, {"low": 0.05, "high": 0.2})["call"] == "not determined"


def _planted(**kw):
    m = {"n_with_input": 250, "n_errors": 20, "n_correct": 20, "p1_auroc": 0.70, "p2_gap_closed": 0.25,
         "p3_call": "not determined"}
    m.update(kw)
    return e99.grade(m)


def test_grades_at_thresholds_and_floors():
    g = _planted()
    assert g["P1"]["holds"] and g["P2"]["holds"], "a value at its threshold holds"
    g = _planted(p1_auroc=0.6999, p2_gap_closed=0.2499)
    assert not g["P1"]["holds"] and not g["P2"]["holds"]
    for kw in ({"n_with_input": 249}, {"n_errors": 19}, {"n_correct": 19}):
        g = _planted(**kw)
        assert not g["P1"]["graded"] and not g["P2"]["graded"] and g["P1"]["value"] == 0.70
    assert _planted(p2_gap_closed=None)["P2"]["graded"] is False
    assert _planted()["P3"]["graded"] is False
    assert e99.THRESHOLDS == {"P1": 0.70, "P2": 0.25} and e99.FLOORS["min_with_input"] == 250


def test_lenient_sits_beside_the_verdict_and_never_grades():
    g = _planted(p1_auroc=0.60, p1_auroc_lenient=0.80, p2_gap_closed=0.10, p2_gap_closed_lenient=0.10)
    assert not g["P1"]["holds"] and g["P1"]["lenient_check"] == {"value": 0.80, "meets_threshold": True}
    assert "not determined" in g["P1"]["reading"], "only STRICT fails: the crosswalk is a competing cause"
    assert not g["P2"]["holds"] and g["P2"]["lenient_check"]["meets_threshold"] is False
    assert "transfer is the likelier" in g["P2"]["reading"]
    g = _planted(p1_auroc_lenient=0.10)
    assert g["P1"]["holds"] and "reading" not in g["P1"], "LENIENT never overturns a STRICT pass"
    assert _planted()["P1"]["lenient_check"] == {"value": None, "meets_threshold": None}


# ----------------------------------------------------------------------------- the smoke
def test_selection_and_inventory(smoke):
    rec, _ = smoke
    s = rec["select_smoke"]
    assert (s["d_km"], s["n_selected"], s["by_country"]) == (X["d_km"], X["n_selected"], X["by_country"])
    assert s["inside_request_geometry"] == X["inside"]
    assert s["counts_by_d_km"] == {"0": 8, "10": 16, "20": 24}
    assert s["label_changes"]["two_or_more_labels_2015_2017"] == 1
    assert s["plots_on_boundaries"] == {"n": 24, "inside_own_country": 24, "outside_own_country": 0}
    assert s["areas"]["share_outside_both_countries"] < 1e-9
    assert sum(s["areas"]["weights"].values()) == pytest.approx(1.0)
    inv = rec["inventory"]
    assert {c: inv["plots"][c]["covered"] for c in ("kenya", "tanzania")} == X["covered"]
    assert inv["mode"].startswith("inventory") and "rules" not in inv
    assert rec["overlapping_grids"]["covered_in_two_or_more"] == 1
    assert rec["overlapping_grids"]["covered_in_two_differing"] == 0
    w = rec["plots_without_input"]
    assert w["by_country"] == {"kenya": 1, "tanzania": 1} and w["outside_every_grid"] == 1
    assert w["in_a_grid_uncovered"] == 1


def test_errors_and_the_stratified_estimate(smoke):
    rec, units = smoke
    W = rec["weights"]["by_country"]
    for rule in ("strict", "lenient"):
        r = rec["rules"][rule]
        assert {c: r["by_country"][c]["errors"] for c in W} == X[rule]
        assert {c: r["by_country"][c]["n"] for c in W} == X["covered"]
        theta = sum(W[c] * X[rule][c] / X["covered"][c] for c in W)
        assert r["stratified"]["estimate"] == pytest.approx(theta)
        assert r["stratified"]["low"] <= theta <= r["stratified"]["high"]
    assert rec["readings"]["n_pred_untrained"] == 1
    assert rec["rules"]["strict"]["by_condition"]["sentinel2:all:clear"]["n"] == 14
    k = units["covered"]
    assert set(np.unique(units["weight"][k]).round(12)) == {round(W["kenya"] / 14, 12), round(W["tanzania"] / 8, 12)}
    assert (units["weight"][~k] == 0).all()


def _brute_auroc(s, e, w):
    num = den = 0.0
    for i in np.flatnonzero(e == 1):
        for j in np.flatnonzero(e == 0):
            num += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
            den += w[i] * w[j]
    return num / den


def _brute_capture(s, e, w, b):
    o = np.argsort(-s)
    s, e, w = s[o], e[o], w[o]
    cut, got, acc = b * w.sum(), 0.0, 0.0
    for wi, ei in zip(w, e):
        take = min(wi, max(cut - acc, 0.0))
        got += ei * take
        acc += wi
    return got / (e * w).sum()


def test_the_ranking_against_brute_force(smoke):
    rec, units = smoke
    k = units["covered"]
    s, w = units["suspicion"][k], units["weight"][k]
    for rule in ("strict", "lenient"):
        e = units[f"wrong_{rule}"][k]
        rk = rec["rules"][rule]["ranking"]
        assert rk["weighted"]["auroc"] == pytest.approx(_brute_auroc(s, e, w))
        assert rk["unweighted"]["auroc"] == pytest.approx(_brute_auroc(s, e, np.ones_like(w)))
        theta = (e * w).sum() / w.sum()
        cap = _brute_capture(s, e, w, 0.10)
        assert rk["weighted"]["capture"]["0.1"] == pytest.approx(cap)
        ceil = min(1.0, 0.10 / theta)
        assert rk["weighted"]["gap_closed"]["0.1"] == pytest.approx((cap - 0.10) / (ceil - 0.10))
        assert rk["auroc_weighted_bootstrap"]["n_resamples"] > 100
    pr = rec["prereg"]
    assert pr["P1"]["value"] == pytest.approx(rec["rules"]["strict"]["ranking"]["weighted"]["auroc"])
    assert pr["P1"]["graded"] and pr["P1"]["holds"] == (pr["P1"]["value"] >= e99.THRESHOLDS["P1"])
    assert pr["P2"]["graded"] and pr["P2"]["holds"] == (pr["P2"]["value"] >= e99.THRESHOLDS["P2"])
    assert pr["P3"]["call"] == rec["p3_transfer_gap"]["call"]
    assert pr["P1"]["lenient_check"]["value"] == pytest.approx(rec["rules"]["lenient"]["ranking"]["weighted"]["auroc"])


def test_report_only_parts(smoke):
    rec, _ = smoke
    d = rec["where_the_plots_sit_reported"]
    # 9 plots above the background's 0.6, one tied with it, 12 below: about (9 + 0.5) / 22
    assert d["mean_percentile"] == pytest.approx(9.5 / 22, abs=0.01)
    f = rec["inside_awf_reported"]
    assert f["n_inside_covered"] == 8 and f["rules_2017"] == {"strict": X["inside_strict_2017"],
                                                             "lenient": X["inside_lenient_2017"]}
    assert f["paired"]["strict"] == {"both_right": 5, "right_2017_only": 0, "right_2023_only": 3, "both_wrong": 0,
                                     "sign_test_two_sided_p": pytest.approx(0.25)}
    pl = rec["plan_reported"]
    assert pl["certify_min_labels"] == 22 and pl["zone_population"]["capped"] is False
    assert all("holds" not in v for v in pl["zone"].values())


def test_the_request_geometry_has_ais_structure(tmp_path):
    paths = e99.make_smoke_inputs(str(tmp_path))
    P, _, _ = e99.select(paths["timesync"], paths["request_geometry"], None, None, e99.SMOKE_MIN_PLOTS, None)
    out = e99.write_request(str(tmp_path / "req"), P, paths["request_geometry"])
    # one feature holding every square, as Ai2's file holds one feature: one feature per square made olmoearth_run
    # window each 1-degree cell once per plot in it (job 1247349: 9,298 windows for 309 plots)
    assert out["prediction_request_geometry.geojson"]["features"] == 1
    assert out["prediction_request_geometry.geojson"]["squares"] == 24
    assert out["prediction_request_geometry_pilot.geojson"]["features"] == 1
    assert out["prediction_request_geometry_pilot.geojson"]["squares"] == e99.PILOT_N
    with open(tmp_path / "req" / "prediction_request_geometry.geojson") as f:
        gj = json.load(f)
    assert {f["properties"]["oe_start_time"] for f in gj["features"]} == {"2017-01-01T00:00:00Z"}
    assert {f["properties"]["oe_end_time"] for f in gj["features"]} == {"2017-12-31T00:00:00Z"}
    assert [f["geometry"]["type"] for f in gj["features"]] == ["MultiPolygon"]
    assert e99.n_squares(gj) == 24 and len(gj["features"][0]["geometry"]["coordinates"]) == 24
    with open(paths["request_geometry"]) as f:
        assert e99.structure(gj) == e99.structure(json.load(f)), "Ai2's structure, Polygon or MultiPolygon"
    with pytest.raises(ValueError, match="git checkout"):
        e99.write_request(os.path.join(ROOT, "exp", "out", "e99_never"), P, paths["request_geometry"])
    assert not os.path.exists(os.path.join(ROOT, "exp", "out", "e99_never"))


def test_nothing_written_carries_a_position_or_an_identifier(smoke, out_dir, tmp_path):
    rec, units = smoke
    e99.e98.check_no_coordinates(rec)
    with open(os.path.join(out_dir, "exp99_summary_smoke.json")) as f:
        text = f.read()
    written = json.loads(text)

    def numbers(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from numbers(v)
        elif isinstance(o, list):
            for v in o:
                yield from numbers(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield float(o)
    paths = e99.make_smoke_inputs(str(tmp_path))
    plots = tsx.read_plots(paths["timesync"], 2017)
    coords = set(np.round(plots["lon"], 6)) | set(np.round(plots["lat"], 6))
    assert not coords & {round(v, 6) for v in numbers(written)}
    assert "kenya:" not in text and "tanzania:" not in text, "no plot identifier is written"
    z = np.load(os.path.join(out_dir, "exp99_units_smoke.npz"))
    assert not {"lon", "lat", "pid", "plotid", "x", "y"} & set(z.files)


def test_the_run_is_refused_while_the_page_is_a_draft(tmp_path, capsys):
    assert e99.prereg_status("# exp99\n\n**Status: DRAFT, not frozen.**\n") == "draft"
    if e99.prereg_status() != "frozen":
        rc = e99.main(["--scores", str(tmp_path), "--request-geometry", str(tmp_path / "g.geojson"),
                       "--out-dir", str(tmp_path)])
        assert rc == 3 and "refused" in capsys.readouterr().out
        assert not os.listdir(tmp_path)

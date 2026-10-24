"""Map certificates (1.17.0): `certificate` packs a certified zone to travel with the map, `verify` checks it.

On the quick-start map: a certificate is written and verifies; each kind of tampering fails, including those a forger
makes with every recorded digest written anew (the sample's windows swapped or cut short; the zone JSON, a field, the
statement, the footprint, the description or the mask edited; the seed rule dropped; the logits reading, the target,
its weight, the condition layer or its names changed after labelling); a point is looked up in the zone; the bundle
stands on its own once moved; the server's verify tool says the same and does not crash on a broken bundle; and a
result that certifies nothing, a sample with a seed given by hand, a missing date and unsafe output paths are
refused."""
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("rasterio")

from oe_inferencex import certificate as cert                       # noqa: E402
from oe_inferencex import mcp_server                                 # noqa: E402
from oe_inferencex.cli import main                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("quickstart_map", os.path.join(ROOT, "examples", "quickstart_map.py"))
qs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qs)


def _quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = main(argv)
    return code, out.getvalue()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """The quick-start map, a random sample of 315 windows with a target and a seed from the map, labelled from the
    truth, certified, and its certificate."""
    d = tmp_path_factory.mktemp("cert")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qs.write_maps()
            assert main(["sample", "scores.tif", "--design", "random", "--budget", "315", "--target", "0.5",
                         "--seed-from-map", "--alpha", "0.05", "--out", "s.csv"]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.05"]) == 0
            assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
    finally:
        os.chdir(cwd)
    return d


def test_the_certificate_states_the_guarantee_and_binds_the_map(bundle):
    item = json.loads((bundle / "cert" / "certificate.json").read_text())
    p = item["properties"]
    zone = json.loads((bundle / "s_zone.json").read_text())
    assert item["type"] == "Feature" and item["stac_version"] == "1.0.0" and p["datetime"] == "2024-06-01T00:00:00Z"
    assert p["zc:rule"] == "target" and p["zc:alpha"] == 0.05 and p["zc:delta"] == 0.1
    assert p["zc:coverage"] == zone["coverage"] and p["zc:upper_bound"] == zone["upper_bound"]
    plan = {"alpha": 0.05, "delta": 0.1, "rule": "target", "level_cut": "standard"}
    assert json.loads((bundle / "cert" / "labels.json").read_text())["certify"] == plan
    settings = cert.seed_settings("random", 315, 4, False, None, 0.5, None, None, None, plan)
    assert p["zc:seed_rule"] == "from_map" and p["zc:seed"] == cert.seed_from(p["zc:map"]["map_sha256"], settings)
    assert p["zc:seed"] >= 2 ** 64 or p["zc:seed"] == int(cert.hashlib.sha256(
        (p["zc:map"]["map_sha256"] + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:32], 16)
    assert p["zc:statement"].startswith(f"The {100 * zone['coverage']:.0f}% most confident of the "
                                        f"{zone['n_population']} windows counted on this map")
    assert "that the sample was drawn once" in p["zc:statement"]
    assert "assumes the reviewer's labels are right" in p["zc:statement"]
    assert set(p["zc:files"]) == {"labels.csv", "labels.json", "zone.json", "zone_mask.tif"}
    lon, lat = item["bbox"][0], item["bbox"][1]
    assert -180 <= lon <= 180 and -90 <= lat <= 90 and item["geometry"]["type"] == "Polygon"
    import rasterio
    with rasterio.open(bundle / "cert" / "zone_mask.tif") as src:            # a GIS opens the zone on the map's grid
        mask = src.read(1)
        assert src.crs.to_epsg() == 32632 and src.transform.a == 40.0       # 10 m pixels, windows of 4
    assert int(mask.sum()) == zone["n_zone"]


def test_an_untouched_certificate_verifies(bundle):
    r = cert.verify(str(bundle / "cert" / "certificate.json"))
    assert r["verified"] and set(r["checks"]) == {"bundle", "map", "seed", "draw", "certify", "statement", "mask"}
    assert r["statement"] == json.loads((bundle / "cert" / "certificate.json").read_text())["properties"]["zc:statement"]
    assert any(n.startswith("taken on the issuer's word") for n in r["notes"])
    assert any("pass --scores with your own copy" in n for n in r["notes"])           # the map was read where it says
    assert not any("pass --scores" in n for n in cert.verify(str(bundle / "cert" / "certificate.json"),
                                                            scores=str(bundle / "scores.tif"))["notes"])
    code, out = _quiet(["verify", str(bundle / "cert" / "certificate.json")])
    assert code == 0 and out.splitlines()[-1].startswith("verified: The ")
    assert "alpha, delta, the rule and the level cut were fixed with the draw" in out
    assert "more than delta can be wrong" in out.splitlines()[-1]


def _copy(bundle, name):
    dst = bundle / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(bundle / "cert", dst)
    return dst


def _forge(d, side=None, item=None, files=()):
    """Edit the bundle's sidecar and certificate as a forger would, writing every recorded digest anew."""
    if side is not None:
        s = json.loads((d / "labels.json").read_text())
        side(s)
        (d / "labels.json").write_text(json.dumps(s, indent=1))
    it = json.loads((d / "certificate.json").read_text())
    if item is not None:
        item(it)
    for name in ("labels.csv", "labels.json", "zone.json", "zone_mask.tif", *files):
        if (d / name).exists():
            it["properties"]["zc:files"][name] = cert.file_sha256(d / name)
    (d / "certificate.json").write_text(json.dumps(it, indent=1))
    return cert.verify(str(d / "certificate.json"))


def _rewrite_labels(path, change):
    rows = list(csv.DictReader(open(path)))
    change(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_a_changed_label_fails_the_bundle(bundle):
    d = _copy(bundle, "t_label")
    _rewrite_labels(d / "labels.csv", lambda rows: rows[next(i for i, r in enumerate(rows) if r["wrong"] == "1")]
                    .update(wrong="0"))
    r = cert.verify(str(d / "certificate.json"))
    assert not r["verified"] and r["checks"]["bundle"]["ok"] is False and "labels.csv" in r["checks"]["bundle"]["detail"]
    code, _ = _quiet(["verify", str(d / "certificate.json")])
    assert code == 1


def test_an_edited_map_fails_the_map_and_the_seed(bundle):
    import rasterio
    with rasterio.open(bundle / "scores.tif") as src:
        prof, a = src.profile, src.read()
    a[:, 0, 0] = a[:, 0, 0][::-1]
    with rasterio.open(bundle / "edited.tif", "w", **prof) as dst:
        dst.write(a)
    r = cert.verify(str(bundle / "cert" / "certificate.json"), scores=str(bundle / "edited.tif"))
    assert r["checks"]["map"]["ok"] is False and r["checks"]["seed"]["ok"] is False and not r["verified"]


def test_the_same_map_recompressed_still_verifies(bundle):
    """The digest is of the decoded pixels, CRS and transform, not of the file's bytes."""
    import rasterio
    with rasterio.open(bundle / "scores.tif") as src:
        prof, a = src.profile, src.read()
    prof.update(compress="deflate", tiled=True, blockxsize=64, blockysize=64)
    with rasterio.open(bundle / "recompressed.tif", "w", **prof) as dst:
        dst.write(a)
    assert cert.verify(str(bundle / "cert" / "certificate.json"), scores=str(bundle / "recompressed.tif"))["verified"]


def test_swapped_windows_fail_the_draw(bundle):
    """A sample whose windows are not the seed's: the sidecar's indices changed and its digest recorded anew, as a
    forger would, so only the draw shows it."""
    d = _copy(bundle, "t_draw")
    side = json.loads((d / "labels.json").read_text())
    side["indices"] = side["indices"][1:] + side["indices"][:1]
    (d / "labels.json").write_text(json.dumps(side, indent=1))
    item = json.loads((d / "certificate.json").read_text())
    item["properties"]["zc:files"]["labels.json"] = cert.file_sha256(d / "labels.json")
    (d / "certificate.json").write_text(json.dumps(item, indent=1))
    r = cert.verify(str(d / "certificate.json"))
    assert r["checks"]["draw"]["ok"] is False and not r["verified"]


def test_an_edited_statement_fails_certify(bundle):
    """The zone JSON's coverage raised and its digest recorded anew: certify, run again, gives the real zone."""
    d = _copy(bundle, "t_zone")
    zone = json.loads((d / "zone.json").read_text())
    zone["coverage"] = 1.0
    (d / "zone.json").write_text(json.dumps(zone, indent=1))
    item = json.loads((d / "certificate.json").read_text())
    item["properties"]["zc:files"]["zone.json"] = cert.file_sha256(d / "zone.json")
    (d / "certificate.json").write_text(json.dumps(item, indent=1))
    r = cert.verify(str(d / "certificate.json"))
    assert r["checks"]["certify"]["ok"] is False and "coverage" in r["checks"]["certify"]["detail"]


def test_a_point_is_looked_up_in_the_zone(bundle):
    item = json.loads((bundle / "cert" / "certificate.json").read_text())
    t = item["properties"]["zc:map"]["transform"]
    import rasterio
    with rasterio.open(bundle / "cert" / "zone_mask.tif") as src:
        mask = src.read(1)
    r_in, c_in = map(int, np.argwhere(mask == 1)[0])
    x, y = t[2] + (4 * c_in + 2) * t[0], t[5] + (4 * r_in + 2) * t[4]        # the window's centre in the map's CRS
    p = cert.point(str(bundle / "cert" / "certificate.json"), x=x, y=y)
    assert p["inside"] is True and p["window"] == [r_in, c_in]
    if (mask == 0).any():
        r_out, c_out = map(int, np.argwhere(mask == 0)[0])
        assert cert.point(str(bundle / "cert" / "certificate.json"), row=4 * r_out, col=4 * c_out)["inside"] is False
    assert cert.point(str(bundle / "cert" / "certificate.json"), row=-50, col=0)["where"] == "outside the map"
    h, w = item["properties"]["zc:map"]["shape"]
    if h % 4:                                   # a pixel of the map below the last whole window
        assert cert.point(str(bundle / "cert" / "certificate.json"), row=h - 1, col=0)["where"].startswith("in no whole")


def test_the_bundle_moves_with_the_map(bundle, tmp_path):
    """Moved elsewhere with the map, the bundle verifies with the map's new path."""
    shutil.copytree(bundle / "cert", tmp_path / "cert")
    shutil.copyfile(bundle / "scores.tif", tmp_path / "map.tif")
    assert cert.verify(str(tmp_path / "cert" / "certificate.json"), scores=str(tmp_path / "map.tif"))["verified"]


def test_the_server_verifies_and_looks_up_a_point(bundle):
    r = mcp_server.verify(certificate=str(bundle / "cert" / "certificate.json"), row=0, col=0)
    assert r["summary"]["verified"] and r["conclusion"].startswith("Verified. The ")
    assert "issuer's word: the labels" in r["limits"] and r["files"] == {}


def test_a_result_that_certifies_nothing_and_a_missing_date_are_refused(bundle, tmp_path):
    zone = json.loads((bundle / "s_zone.json").read_text())
    zone["coverage"] = None
    (tmp_path / "none.json").write_text(json.dumps(zone))
    with pytest.raises(SystemExit, match="certifies no zone"):
        main(["certificate", str(tmp_path / "none.json"), "--date", "2024", "--out", str(tmp_path / "c")])
    with pytest.raises(SystemExit, match="date"):
        main(["certificate", str(bundle / "s_zone.json"), "--date", "not a date", "--out", str(tmp_path / "c2")])


def test_seed_from_map_refusals(bundle):
    with pytest.raises(SystemExit, match="give --seed or --seed-from-map, not both"):
        main(["sample", str(bundle / "scores.tif"), "--design", "random", "--budget", "20", "--seed", "3",
              "--seed-from-map", "--out", str(bundle / "x.csv")])
    with pytest.raises(SystemExit, match="does not apply to --other"):
        main(["sample", str(bundle / "scores.tif"), "--other", str(bundle / "other.tif"), "--budget", "20",
              "--seed-from-map", "--out", str(bundle / "y.csv")])


@pytest.mark.parametrize("design", ["sequential", "condition"])
def test_sequential_and_per_condition_certificates_verify(bundle, tmp_path, design):
    """The other two routes certify reads: a sequential sample (rule sequential) and a sample by input condition
    (one zone per condition, delta split)."""
    import rasterio
    with rasterio.open(bundle / "scores.tif") as src:
        prof, h, w = src.profile, src.height, src.width
    cond = (np.arange(w)[None, :] >= w // 2).astype(np.int32) * np.ones((h, 1), np.int32)
    prof.update(count=1, dtype="int32", nodata=None)
    with rasterio.open(tmp_path / "cond.tif", "w", **prof) as dst:
        dst.write(cond, 1)
    shutil.copyfile(bundle / "scores.tif", tmp_path / "scores.tif")
    shutil.copyfile(bundle / "truth.tif", tmp_path / "truth.tif")
    extra = (["--design", "sequential", "--alpha", "0.1"] if design == "sequential" else
             ["--condition", str(tmp_path / "cond.tif"), "--condition-names", "0=west", "1=east", "--alpha", "0.1"])
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            budget = "600" if design == "sequential" else "500"
            assert main(["sample", "scores.tif", "--budget", budget, "--seed-from-map", "--out", "s.csv", *extra]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.1"]) == 0
            zone = json.loads((tmp_path / "s_zone.json").read_text())
            assert zone.get("coverage") is not None or zone.get("certified_share_of_map")   # the draw is fixed
            assert main(["certificate", "s_zone.json", "--date", "2024-01-01/2024-12-31", "--out", "cert"]) == 0
    finally:
        os.chdir(cwd)
    item = json.loads((tmp_path / "cert" / "certificate.json").read_text())
    p = item["properties"]
    assert p["datetime"] is None and p["start_datetime"] == "2024-01-01T00:00:00Z"
    if design == "condition":
        assert set(p["zc:per_condition"]) == {"west", "east"} and p["zc:statement"].startswith("Certified per input")
    else:
        assert p["zc:rule"] == "sequential" and "at every look" in p["zc:statement"]
    r = cert.verify(str(tmp_path / "cert" / "certificate.json"))
    assert r["verified"], r["checks"]
    moved = tmp_path / "elsewhere"
    shutil.copytree(tmp_path / "cert", moved / "cert")
    shutil.copyfile(tmp_path / "scores.tif", moved / "map.tif")
    if design == "condition":
        os.remove(tmp_path / "cond.tif")                      # the bundle carries its own copy of the layer
        assert cert.verify(str(moved / "cert" / "certificate.json"), scores=str(moved / "map.tif"))["verified"]
        # the layer swapped after labelling (east and west exchanged), with the sidecar and digests written anew
        swapped = 1 - cond
        prof.update(count=1, dtype="int32", nodata=None)
        d = moved / "cert"
        with rasterio.open(d / "condition.tif", "w", **prof) as dst:
            dst.write(swapped.astype(np.int32), 1)
        def edit(s):
            s["condition"]["sha256"] = cert.raster_digest(d / "condition.tif")
            s["condition_grid"] = [1 - int(v) if v is not None and int(v) >= 0 else v for v in s["condition_grid"]]
        _forge(d, side=edit, files=("condition.tif",))
        r = cert.verify(str(d / "certificate.json"), scores=str(moved / "map.tif"))
        assert r["checks"]["seed"]["ok"] is False and not r["verified"]
        # the names swapped (east called west) in the sidecar and the CSV, digests written anew
        d2 = tmp_path / "names"
        shutil.copytree(tmp_path / "cert", d2)
        _rewrite_labels(d2 / "labels.csv", lambda rows: [r.update(condition={"west": "east", "east": "west"}
                                                                  .get(r["condition"], r["condition"])) for r in rows])
        def rename(s):
            s["condition"]["names_given"] = [[0, "east"], [1, "west"]]
            s["condition"]["names"] = [{"west": "east", "east": "west"}.get(n, n) for n in s["condition"]["names"]]
        # the names change nothing certify computes, so they are the issuer's word, and the statement says so
        r = _forge(d2, side=rename)
        assert r["checks"]["seed"]["ok"] is True and "names as the issuer gave them" in r["statement"]
        assert any("the condition names" in n for n in r["notes"])
        os.remove(d2 / "condition.tif")
        r = cert.verify(str(d2 / "certificate.json"), scores=str(tmp_path / "scores.tif"))
        assert r["checks"]["bundle"]["ok"] is False and r["checks"]["condition"]["ok"] is False and not r["verified"]


def test_an_edited_field_or_statement_fails_even_with_no_digest_to_change(bundle):
    """certificate.json itself carries no digest: its fields are rebuilt from certify run again and compared."""
    for name, edit in [("t_alpha", lambda it: it["properties"].update({"zc:alpha": 0.01})),
                       ("t_stmt", lambda it: it["properties"].update(
                           {"zc:statement": it["properties"]["zc:statement"].replace("wrong at most 5%",
                                                                                     "wrong at most 1%")})),
                       ("t_cov", lambda it: it["properties"].update({"zc:coverage": 0.123})),
                       ("t_extra", lambda it: it["properties"].update({"zc:guarantee": "every window is right"})),
                       ("t_asset", lambda it: it["assets"]["labels"].update({"href": "other.csv"}))]:
        r = _forge(_copy(bundle, name), item=edit)
        assert r["checks"]["statement"]["ok"] is False and not r["verified"], name
        assert r["checks"]["certify"]["ok"] is True                    # the bundle itself was not touched


def test_an_edited_mask_fails_and_points_use_the_zone_certify_gives(bundle):
    import rasterio
    d = _copy(bundle, "t_mask")
    with rasterio.open(d / "zone_mask.tif") as src:
        prof, m = src.profile, src.read(1)
    with rasterio.open(d / "zone_mask.tif", "w", **prof) as dst:
        dst.write(np.ones_like(m), 1)                             # every window claimed certified
    r = _forge(d)
    assert r["checks"]["bundle"]["ok"] is True and r["checks"]["mask"]["ok"] is False and not r["verified"]
    r0, c0 = map(int, np.argwhere(m == 0)[0])
    assert cert.point(str(d / "certificate.json"), row=4 * r0, col=4 * c0, mask=r["mask"])["inside"] is False


def test_a_seed_given_by_hand_is_refused_and_a_dropped_seed_rule_fails(bundle, tmp_path):
    """A certificate needs a seed taken from the map; a forger who drops the rule (to change settings freely) fails."""
    cwd = os.getcwd()
    shutil.copyfile(bundle / "scores.tif", tmp_path / "scores.tif")
    shutil.copyfile(bundle / "truth.tif", tmp_path / "truth.tif")
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "scores.tif", "--design", "random", "--budget", "315", "--target", "0.5",
                         "--seed", "7", "--out", "s.csv"]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.05"]) == 0
            with pytest.raises(SystemExit, match="needs a sample drawn with `sample --seed-from-map --alpha A`"):
                main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"])
    finally:
        os.chdir(cwd)
    d = _copy(bundle, "t_downgrade")
    r = _forge(d, side=lambda s: [s.pop(k) for k in ("seed_rule", "seed_settings", "map_sha256")],
               item=lambda it: it["properties"].update({"zc:seed_rule": None}))
    assert r["checks"]["seed"]["ok"] is False and not r["verified"]


def test_a_target_changed_after_labelling_fails_the_seed(bundle):
    d = _copy(bundle, "t_target")
    r = _forge(d, side=lambda s: s["target"].update(coverage=0.3))
    assert r["checks"]["seed"]["ok"] is False and not r["verified"]


def test_a_sample_cut_short_fails(bundle):
    """The last labelled windows dropped from the CSV and the sidecar, and the budget written down to match."""
    d = _copy(bundle, "t_cut")
    _rewrite_labels(d / "labels.csv", lambda rows: rows.__delitem__(slice(300, None)))
    r = _forge(d, side=lambda s: s.update(indices=s["indices"][:300]))
    assert r["checks"]["draw"]["ok"] is False and not r["verified"]
    d = _copy(bundle, "t_cut_budget")
    _rewrite_labels(d / "labels.csv", lambda rows: rows.__delitem__(slice(300, None)))
    r = _forge(d, side=lambda s: s.update(indices=s["indices"][:300], budget=300))
    assert r["checks"]["seed"]["ok"] is False and r["checks"]["draw"]["ok"] is False and not r["verified"]


def test_the_digest_and_the_crs_ignore_how_the_grid_is_spelled():
    """Signed zeros in the transform do not change the digest; the CRS is compared as a CRS, not as WKT text, which
    GDAL and PROJ releases spell differently."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    g1 = {"crs": CRS.from_epsg(32632), "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)}
    g2 = {"crs": CRS.from_epsg(32632), "transform": Affine(10.0, -0.0, 500000.0, -0.0, -10.0, 4000000.0)}
    assert cert.pixel_digest(a, None, g1) == cert.pixel_digest(a, None, g2)
    g3 = {"crs": g1["crs"], "transform": Affine(10.0, 0.0, 500010.0, 0.0, -10.0, 4000000.0)}
    assert cert.pixel_digest(a, None, g1) != cert.pixel_digest(a, None, g3)
    laea = CRS.from_proj4("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m")
    assert cert._same_crs(cert._crs_text(laea), laea) and cert._same_crs(CRS.from_epsg(32632).to_wkt(), g1["crs"])
    assert not cert._same_crs("EPSG:32633", g1["crs"]) and not cert._same_crs("not a crs", g1["crs"])


def test_output_paths_that_would_overwrite_are_refused(bundle, tmp_path):
    with pytest.raises(SystemExit, match="into the bundle's directory"):
        main(["verify", str(bundle / "cert" / "certificate.json"), "--out", str(bundle / "cert" / "checks.json")])
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit, match="not a certificate bundle"):
        main(["certificate", str(bundle / "s_zone.json"), "--date", "2024-06-01", "--out", str(tmp_path / "busy")])
    (tmp_path / "f").write_text("x")
    with pytest.raises(SystemExit, match="is a file"):
        main(["certificate", str(bundle / "s_zone.json"), "--date", "2024-06-01", "--out", str(tmp_path / "f")])
    assert (tmp_path / "busy" / "notes.txt").read_text() == "mine"


def test_the_server_refuses_half_a_point(bundle):
    with pytest.raises(mcp_server.Refused, match="one of the pair"):
        mcp_server.verify(certificate=str(bundle / "cert" / "certificate.json"), x=1.0)


def test_a_sequential_sample_from_the_map_extends_with_its_rule(bundle, tmp_path):
    """The seed of a sequential sample leaves out the budget, so --extend keeps it, with --seed-from-map or the seed."""
    shutil.copyfile(bundle / "scores.tif", tmp_path / "scores.tif")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        code, out = _quiet(["sample", "scores.tif", "--design", "sequential", "--alpha", "0.1", "--budget", "100",
                            "--seed-from-map", "--out", "s.csv"])
        assert code == 0 and "--seed-from-map --budget <more> --extend s.csv" in out
        first = json.loads((tmp_path / "s.json").read_text())
        assert _quiet(["sample", "scores.tif", "--design", "sequential", "--budget", "150", "--seed-from-map",
                       "--extend", "s.csv", "--out", "s2.csv"])[0] == 0
        second = json.loads((tmp_path / "s2.json").read_text())
        assert second["seed"] == first["seed"] and second["seed_rule"] == "from_map"
        assert second["indices"][:100] == first["indices"]
        assert _quiet(["sample", "scores.tif", "--design", "sequential", "--budget", "200", "--seed",
                       str(first["seed"]), "--extend", "s2.csv", "--out", "s3.csv"])[0] == 0
        assert json.loads((tmp_path / "s3.json").read_text())["seed_rule"] == "from_map"
        with pytest.raises(SystemExit, match="not the seed this map and its settings give"):
            main(["sample", "scores.tif", "--design", "sequential", "--alpha", "0.1", "--budget", "100", "--seed",
                  "5", "--out", "g.csv"])
            main(["sample", "scores.tif", "--design", "sequential", "--budget", "150", "--seed-from-map",
                  "--extend", "g.csv", "--out", "g2.csv"])
    finally:
        os.chdir(cwd)


def test_settings_changed_after_labelling_fail(bundle):
    """The logits reading, the target's weight and the seed settings recorded: each is fixed by the seed or by the
    draw, so changing it after labelling fails even with every digest written anew."""
    r = _forge(_copy(bundle, "t_logits"), side=lambda s: s.update(logits=True))
    assert r["checks"]["seed"]["ok"] is False and not r["verified"]
    r = _forge(_copy(bundle, "t_weight"), side=lambda s: s["target"].update(weight=1.0))
    assert r["checks"]["draw"]["ok"] is False and not r["verified"]
    r = _forge(_copy(bundle, "t_settings"), side=lambda s: s["seed_settings"].update(budget=300))
    assert r["checks"]["seed"]["ok"] is False and not r["verified"]


def test_the_item_is_compared_whole(bundle):
    """Fields a catalogue indexes (the footprint, a description) and the zone JSON's own fields are rebuilt too."""
    def move(it):
        it["bbox"] = [0.0, 45.0, 1.0, 46.0]
    r = _forge(_copy(bundle, "t_bbox"), item=move)
    assert r["checks"]["statement"]["ok"] is False and not r["verified"]
    r = _forge(_copy(bundle, "t_desc"), item=lambda it: it.update(description="Every pixel is certified correct."))
    assert r["checks"]["statement"]["ok"] is False and not r["verified"]
    d = _copy(bundle, "t_zonefield")
    zone = json.loads((d / "zone.json").read_text())
    zone["note"] = "certified by an independent auditor"
    (d / "zone.json").write_text(json.dumps(zone, indent=1))
    r = _forge(d)
    assert r["checks"]["certify"]["ok"] is False and not r["verified"]
    r = _forge(_copy(bundle, "t_wb"), item=lambda it: it["properties"]["zc:written_by"].update(note="audited"))
    assert r["checks"]["statement"]["ok"] is False and not r["verified"]


def test_a_mask_in_another_form_fails(bundle):
    """The zone written as .npy beside a georeferenced map, or as a GeoTIFF whose no-data hides the zeros."""
    import rasterio
    d = _copy(bundle, "t_npy")
    with rasterio.open(d / "zone_mask.tif") as src:
        prof, m = src.profile, src.read(1)
    os.remove(d / "zone_mask.tif")
    np.save(d / "zone_mask.npy", m)
    r = _forge(d, item=lambda it: (it["properties"]["zc:files"].pop("zone_mask.tif"),
                                   it["assets"]["zone_mask"].update(href="zone_mask.npy")), files=("zone_mask.npy",))
    assert r["checks"]["mask"]["ok"] is False and not r["verified"]
    d = _copy(bundle, "t_nodata")
    prof.update(nodata=0)
    with rasterio.open(d / "zone_mask.tif", "w", **prof) as dst:
        dst.write(m, 1)
    r = _forge(d)
    assert r["checks"]["mask"]["ok"] is False and not r["verified"]


def test_a_broken_bundle_is_a_failed_check_not_a_crash(bundle, tmp_path):
    """A missing condition layer or a sidecar that is not JSON: verify reports a failed check, the CLI exits 1 with
    its checks, and the server replies (a SystemExit would end its stdio loop)."""
    d = _copy(bundle, "t_broken")
    (d / "labels.json").write_text("{not json")
    r = cert.verify(str(d / "certificate.json"))
    assert not r["verified"] and r["checks"]["read"]["ok"] is False
    code, out = _quiet(["verify", str(d / "certificate.json")])
    assert code == 1 and "read: FAILED" in out
    reply = mcp_server.verify(certificate=str(d / "certificate.json"))
    assert reply["summary"]["verified"] is False and reply["conclusion"].startswith("Not verified")
    with pytest.raises(SystemExit, match="not a certificate this package can read"):
        main(["verify", str(tmp_path / "missing.json")])


@pytest.fixture(scope="module")
def product(tmp_path_factory, bundle):
    """A product on the quick-start grid: the class map and a 0-100 confidence band, sampled from the map, labelled
    from the truth, certified and packed."""
    import rasterio
    d = tmp_path_factory.mktemp("product")
    with rasterio.open(bundle / "scores.tif") as src:
        prof, a = src.profile, src.read()
    prof.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(d / "classes.tif", "w", **prof) as dst:
        dst.write(a.argmax(0).astype(np.uint8), 1)
    with rasterio.open(d / "band.tif", "w", **prof) as dst:
        dst.write(np.round(100 * a.max(0)).astype(np.uint8), 1)
    shutil.copyfile(bundle / "truth.tif", d / "truth.tif")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "classes.tif", "--confidence", "band.tif", "--confidence-range", "0", "100",
                         "--design", "random", "--budget", "400", "--seed-from-map", "--alpha", "0.1",
                         "--out", "s.csv"]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.1"]) == 0
            assert main(["certificate", "s_zone.json", "--date", "2024-01-01/2024-12-31", "--out", "cert"]) == 0
    finally:
        os.chdir(cwd)
    return d


def test_a_product_verifies_only_with_its_band_given(product, bundle):
    """The band decides the zone, so a verifier who gives the class map gives the band too; the certificate's own
    path for it is never read then."""
    c = str(product / "cert" / "certificate.json")
    assert cert.verify(c, str(product / "classes.tif"), str(product / "band.tif"))["verified"]
    r = cert.verify(c, str(product / "classes.tif"))
    assert not r["verified"] and r["checks"]["map"]["ok"] is None and "--confidence" in r["checks"]["map"]["detail"]
    r = cert.verify(str(bundle / "cert" / "certificate.json"), str(bundle / "scores.tif"), str(product / "band.tif"))
    assert not r["verified"] and "does not apply" in r["checks"]["map"]["detail"]
    # a scores map's sidecar naming a band of the issuer's: the sample and the certificate disagree
    d = _copy(product, "t_inject")
    r = _forge(d, side=lambda s: s.pop("confidence"))
    assert r["checks"]["map"]["ok"] is False and not r["verified"]


def test_numbers_are_compared_exactly(bundle):
    """A 128-bit seed one part in 1e10 off, a window size of 3.9999999 or 4.0, a pixel size 1e-7 off (four pixels
    across a 4000-pixel map in degrees), a pass of 1 for true, Infinity, or a key given twice."""
    for name, edit in [("t_seed", lambda it: it["properties"].update({"zc:seed": it["properties"]["zc:seed"] + 10 ** 28})),
                       ("t_patch", lambda it: it["properties"].update({"zc:patch": 3.9999999})),
                       ("t_patch_float", lambda it: it["properties"].update({"zc:patch": 4.0})),
                       ("t_shift", lambda it: it["properties"]["zc:map"]["transform"].__setitem__(
                           0, it["properties"]["zc:map"]["transform"][0] + 1e-7)),
                       ("t_bool", lambda it: it["properties"]["zc:iso19157"]["conformance_result"].update(
                           {"pass": 1}))]:
        r = _forge(_copy(bundle, name), item=edit)
        assert r["checks"]["statement"]["ok"] is False and not r["verified"], name
    d = _copy(bundle, "t_inf")
    text = (d / "certificate.json").read_text().replace('"zc:alpha": 0.05', '"zc:alpha": Infinity')
    (d / "certificate.json").write_text(text)
    with pytest.raises(cert.Unreadable, match="Infinity"):
        cert.verify(str(d / "certificate.json"))
    d = _copy(bundle, "t_dup")
    text = (d / "certificate.json").read_text().replace('"zc:alpha": 0.05', '"zc:alpha": 0.5, "zc:alpha": 0.05')
    (d / "certificate.json").write_text(text)
    with pytest.raises(SystemExit, match="twice"):
        main(["verify", str(d / "certificate.json")])


def test_the_bundle_holds_its_own_files_and_a_plain_mask(bundle):
    """A GIS reads a mask's sidecar files, overviews, scale and tags; none may be there."""
    import rasterio
    d = _copy(bundle, "t_aux")
    (d / "zone_mask.tif.aux.xml").write_text("<PAMDataset><PAMRasterBand band='1'><Offset>1</Offset></PAMRasterBand>"
                                            "</PAMDataset>")
    r = cert.verify(str(d / "certificate.json"))
    assert r["checks"]["bundle"]["ok"] is False and "not part of a bundle" in r["checks"]["bundle"]["detail"]
    for name, change in [("t_ovr", lambda dst: dst.build_overviews([2], rasterio.enums.Resampling.nearest)),
                         ("t_tags", lambda dst: dst.update_tags(note="every window certified")),
                         ("t_scale", lambda dst: setattr(dst, "scales", (2.0,)))]:
        d = _copy(bundle, name)
        with rasterio.open(d / "zone_mask.tif", "r+") as dst:
            change(dst)
        r = _forge(d)
        assert r["checks"]["mask"]["ok"] is False and not r["verified"], name
    d = _copy(bundle, "t_vrt")
    (d / "zone_mask.tif").write_text("<VRTDataset rasterXSize='1' rasterYSize='1'></VRTDataset>")
    r = _forge(d)
    assert r["checks"]["mask"]["ok"] is False and "not the .tif file its name says" in r["checks"]["mask"]["detail"]


def test_a_spreadsheet_saved_csv_still_verifies(bundle):
    """A reviewer's spreadsheet writes 502060.0 as 502060, rounds the confidence, and ends lines with CRLF."""
    d = _copy(bundle, "t_sheet")
    rows = list(csv.DictReader(open(d / "labels.csv")))
    for r in rows:
        for k in ("x", "y"):
            r[k] = "%.15g" % float(r[k])
        r["confidence"] = "%.6g" % float(r["confidence"])
    with open(d / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    assert _forge(d)["verified"]


def test_the_statement_says_what_the_population_leaves_out_and_when_the_zone_is_the_mask():
    zone = {"alpha": 0.05, "delta": 0.1, "coverage": 0.6, "n_zone": 2458, "n_population": 4000, "rule": "target",
            "n_labelled": 400, "upper_bound": 0.04, "n_tied_at_threshold": 3328, "n_tied_inside_zone": 2458}
    text = cert._statement(zone, {"patch": 4, "grid": [64, 64]})
    assert "The zone is the mask, not every window at or above the threshold: 3328" in text
    assert "96 of the map's 4096 windows are left out as no-data" in text
    zone["population_note"] = "The population is the pixels whose confidence lies in [70, 100]."
    assert "confidence lies in [70, 100]" in cert._statement(zone, {"patch": 4, "grid": [64, 64]})


def test_a_footprint_across_the_antimeridian_is_split():
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    geo = {"crs": CRS.from_epsg(32660), "transform": Affine(10.0, 0.0, 832000.0, 0.0, -10.0, 9000.0)}
    poly, bbox = cert._footprint((128, 400), geo)
    assert bbox[0] > bbox[2] and poly["type"] == "MultiPolygon"


def test_seed_from_map_refusals_for_names_and_designs(bundle):
    with pytest.raises(SystemExit, match="printable characters"):
        main(["sample", str(bundle / "scores.tif"), "--budget", "20", "--condition", str(bundle / "truth.tif"),
              "--condition-names", "0=cl\x1b[2Kear", "--out", str(bundle / "n.csv")])
    with pytest.raises(SystemExit, match="random, sequential, condition or class design"):
        main(["sample", str(bundle / "scores.tif"), "--design", "proportional", "--budget", "20", "--seed-from-map",
              "--out", str(bundle / "p.csv")])


def test_the_draw_is_numpys_release_free(bundle):
    """A sample drawn with --seed-from-map is the first windows of the PCG64 order, as a sequential sample is."""
    from oe_inferencex.sequential import sequential_order
    side = json.loads((bundle / "cert" / "labels.json").read_text())
    pop_order = sequential_order(side["n_population"], side["seed"])[:side["budget"]]
    import rasterio
    with rasterio.open(bundle / "scores.tif") as src:
        h, w = src.height // 4, src.width // 4
    assert len(side["indices"]) == side["budget"] and side["n_population"] == h * w   # no no-data on this map
    assert side["indices"] == [int(i) for i in pop_order]


def test_a_point_that_is_not_a_number_is_not_looked_up(bundle):
    code, out = _quiet(["verify", str(bundle / "cert" / "certificate.json"), "--at-pixel", "inf", "0"])
    assert code == 0 and "point: not looked up" in out


def test_an_anchor_history_that_does_not_give_the_anchor_fails(bundle, tmp_path):
    shutil.copyfile(bundle / "scores.tif", tmp_path / "scores.tif")
    shutil.copyfile(bundle / "truth.tif", tmp_path / "truth.tif")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "scores.tif", "--design", "sequential", "--alpha", "0.15", "--budget", "400",
                         "--seed-from-map", "--out", "s.csv"]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.15"]) == 0
            assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
    finally:
        os.chdir(cwd)
    r = _forge(tmp_path / "cert", side=lambda s: s["anchor_rule"].update(alpha=0.01))
    assert r["checks"]["draw"]["ok"] is False and "anchor_rule" in r["checks"]["draw"]["detail"]


def test_a_confidence_range_changed_after_labelling_fails(product):
    r = _forge(_copy(product, "t_range"), side=lambda s: s.update(confidence_range=[0.0, 200.0]))
    assert r["checks"]["seed"]["ok"] is False and not r["verified"]


def test_names_cannot_close_their_quotes(bundle):
    with pytest.raises(SystemExit, match="without quotes or backslashes"):
        main(["sample", str(bundle / "scores.tif"), "--budget", "20", "--condition", str(bundle / "truth.tif"),
              "--condition-names", '0=east" (100%), "west', "--out", str(bundle / "q.csv")])
    assert cert._name('a" (100%), "b') == '"a\\" (100%), \\"b"'


def test_special_files_and_repeated_columns_fail_without_hanging(bundle):
    d = _copy(bundle, "t_fifo")
    os.remove(d / "labels.json")
    os.mkfifo(d / "labels.json")
    r = cert.verify(str(d / "certificate.json"))
    assert not r["verified"] and "not a regular file" in r["checks"]["read"]["detail"]
    d = _copy(bundle, "t_link")
    os.remove(d / "zone.json")
    os.symlink("/dev/zero", d / "zone.json")
    assert not cert.verify(str(d / "certificate.json"))["verified"]
    d = _copy(bundle, "t_dupcol")
    text = (d / "labels.csv").read_text().splitlines()
    text = [text[0] + ",wrong"] + [line + ",0" for line in text[1:]]
    (d / "labels.csv").write_text("\n".join(text) + "\n")
    r = _forge(d)
    assert not r["verified"] and "names a column twice" in json.dumps(r["checks"])
    with pytest.raises(SystemExit, match="names a column twice"):
        main(["certify", str(d / "labels.csv"), "--alpha", "0.05", "--scores", str(bundle / "scores.tif")])


def test_coordinates_written_coarsely_are_not_the_same_windows(bundle):
    d = _copy(bundle, "t_coarse")
    _rewrite_labels(d / "labels.csv", lambda rows: [r.update(x="5.01e5") for r in rows])
    r = _forge(d)
    assert r["checks"]["draw"]["ok"] is False and "the CSV's x" in r["checks"]["draw"]["detail"]


def test_a_mask_with_a_tag_the_writer_does_not_write_fails(bundle):
    import rasterio
    d = _copy(bundle, "t_doctag")
    with rasterio.open(d / "zone_mask.tif", "r+") as dst:
        dst.update_tags(TIFFTAG_DOCUMENTNAME="every window certified")
    r = _forge(d)
    assert r["checks"]["mask"]["ok"] is False and not r["verified"]


def test_error_levels_outside_the_range_verify_runs_are_not_run(bundle):
    d = _copy(bundle, "t_tiny")
    r = _forge(d, side=lambda s: s["certify"].update(alpha=1e-9))
    assert r["checks"]["certify"]["ok"] is None and "error levels verify runs" in r["checks"]["certify"]["detail"]
    assert r["checks"]["seed"]["ok"] is False


def test_certify_keeps_to_the_plan_and_a_changed_plan_fails(bundle):
    """alpha, delta, the rule and the level cut are fixed with the draw: certify refuses others on the same labels,
    and a plan changed after labelling changes the seed."""
    csv_path = str(bundle / "s.csv")
    for argv, what in [(["--alpha", "0.15"], "--alpha 0.05"), (["--delta", "0.2"], "--delta 0.1"),
                       (["--rule", "prefix"], "target rule"), (["--level-cut", "ramp"], "standard level cut")]:
        with pytest.raises(SystemExit, match=what):
            main(["certify", csv_path, *argv, "--out", str(bundle / "other_zone.json")])
    for name, change in [("t_plan_alpha", {"alpha": 0.15}), ("t_plan_cut", {"level_cut": "ramp"})]:
        d = _copy(bundle, name)
        r = _forge(d, side=lambda s: s["certify"].update(change))
        assert r["checks"]["seed"]["ok"] is False and not r["verified"], name
    with pytest.raises(SystemExit, match="give --alpha"):
        main(["sample", str(bundle / "scores.tif"), "--design", "random", "--budget", "20", "--seed-from-map",
              "--out", str(bundle / "noplan.csv")])


def test_issuer_fields_must_have_their_form(bundle):
    for name, edit in [("t_created", lambda it: it["properties"].update(created="2024-06-01 (audited by an agency)")),
                       ("t_version", lambda it: it["properties"]["zc:written_by"].update(version="9.9.9 certified")),
                       ("t_nodate", lambda it: it["properties"].update(datetime=None)),
                       ("t_band", lambda it: it["properties"]["zc:map"].update(confidence_path="/data/band.tif"))]:
        r = _forge(_copy(bundle, name), item=edit)
        assert not r["verified"], name
    d = _copy(bundle, "t_deep")
    (d / "certificate.json").write_text("[" * 100000 + "]" * 100000)
    with pytest.raises(cert.Unreadable):
        cert.verify(str(d / "certificate.json"))


def test_os_files_in_the_bundle_are_ignored_and_points_need_the_rebuilt_zone(bundle):
    d = _copy(bundle, "t_dsstore")
    (d / ".DS_Store").write_bytes(b"\0" * 16)
    (d / "._zone_mask.tif").write_bytes(b"\0" * 16)
    assert cert.verify(str(d / "certificate.json"))["verified"]
    d = _copy(bundle, "t_nopoint")
    os.remove(d / "zone.json")
    code, out = _quiet(["verify", str(d / "certificate.json"), "--at-pixel", "0", "0"])
    assert code == 1 and "point: not looked up: the zone could not be built again" in out


def test_a_users_files_in_out_are_never_replaced(bundle, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "condition.tif").write_text("mine")
    with pytest.raises(SystemExit, match="not a certificate bundle"):
        main(["certificate", str(bundle / "s_zone.json"), "--date", "2024-06-01", "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "condition.tif").read_text() == "mine"


@pytest.mark.parametrize("kind", ["no_crs", "loose_laea", "inf_nodata", "negative_nodata"])
def test_honest_certificates_of_other_maps_verify(bundle, tmp_path, kind):
    """A map with no CRS, one whose CRS rasterio names loosely (a LAEA on GRS 1980 read back as EPSG:3035), and
    no-data values strict JSON and argparse handle only when written with care (inf, -3.4e+38)."""
    import rasterio
    from rasterio.crs import CRS
    with rasterio.open(bundle / "scores.tif") as src:
        prof, a = src.profile, src.read()
    nodata = []
    if kind == "no_crs":
        prof.update(crs=None)
    elif kind == "loose_laea":
        prof.update(crs=CRS.from_proj4("+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 "
                                       "+units=m"))
    else:
        nodata = ["--nodata=inf" if kind == "inf_nodata" else "--nodata=-3.4028234663852886e+38"]
    with rasterio.open(tmp_path / "scores.tif", "w", **prof) as dst:
        dst.write(a)
    shutil.copyfile(bundle / "truth.tif", tmp_path / "truth.tif")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["sample", "scores.tif", "--design", "random", "--budget", "315", "--target", "0.5",
                         "--seed-from-map", "--alpha", "0.05", *nodata, "--out", "s.csv"]) == 0
            qs.label("s.csv")
            assert main(["certify", "s.csv", "--alpha", "0.05"]) == 0
            assert main(["certificate", "s_zone.json", "--date", "2024-06-01", "--out", "cert"]) == 0
    finally:
        os.chdir(cwd)
    assert cert.verify(str(tmp_path / "cert" / "certificate.json"), scores=str(tmp_path / "scores.tif"))["verified"]

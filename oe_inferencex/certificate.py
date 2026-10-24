"""A certified zone as a certificate that travels with the map, and its verifier (1.17.0).

`certify` decides which share of a map is wrong at most alpha of the time, a statement that fails on at most delta of
samples. Until 1.17.0 that decision lived in a JSON beside the labels, bound to the map only by a file path, with the
zone as a window-grid array no GIS could open. A certificate packs it to travel with the map:

- `certificate.json`, a STAC Item whose `zc:` properties state the guarantee (alpha, delta, the rule, the design, the
  seed, the target or anchor, the zone's coverage, threshold and upper bound, per input condition where the sample has
  one) for the map of the date the issuer gives, with the zone on the map's window grid (a GeoTIFF for a georeferenced
  map, .npy otherwise);
- copies of the labelled sample, its sidecar, the zone JSON and any input-condition layer, so the bundle is complete;
- digests that bind it to the map: sha256 of the map's decoded pixel values with its valid mask and transform
  (unchanged by recompression or retiling, changed by any edited pixel or no-data value), and of every file in the
  bundle.

A certificate needs a sample drawn with `sample --seed-from-map`. Its seed is 128 bits of sha256 over the map's digest
(which covers the valid pixels, so the no-data value) and the settings that change what certify gives: the design, the
budget (except for a sequential sample), the window size, the logits reading, a product's confidence range, the target,
the anchor, and the digests of a product's confidence band and of a condition layer. None of those can be changed after
labelling without changing the windows. The draw is taken from a fixed random order read straight from PCG64, the same
on every numpy release. The seed does not show that the issuer drew once: any change to the map or the settings before
labelling, however small, gives another draw, and one can be kept.

`verify` builds the certificate again from the map, the bundled labels and the recorded settings, and compares it: it
draws the sample again (the sidecar, and every column of the CSV but `wrong` up to the rounding a spreadsheet makes);
certifies again (the zone JSON); and builds the STAC Item and the zone mask again from that run. It reads bundle rasters
only as GeoTIFF or .npy, with no sidecar files and no network, and the bundle must hold its own files and nothing else.
Taken on the issuer's word, and said so by verify: the labels (the point of trust, as for every statement the package
makes, with any column the reviewer added), the map's date and the time the certificate was made, the paths it names,
the condition names, when certify's own options (alpha, delta, the rule, the level cut) were chosen, and that the
sample was drawn once.

The statistics are certify's own; what is new is the statement as a portable, re-checkable record. Its fields map onto
ISO 19157-1: a conformance result (the specification "the zone's error rate is at most alpha", passed, with delta
bounding the chance, over the draw, that it is wrong) and a coverage result (the zone mask)."""
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile

import numpy as np

CERT_VERSION = "1.0"
BUNDLE = {"labels": "labels.csv", "sample": "labels.json", "zone": "zone.json"}
# the zc: fields taken from the certify result, built again from certify run again and compared
ZONE_KEYS = ("rule", "alpha", "delta", "coverage", "n_zone", "n_population", "threshold", "upper_bound", "n_labelled",
             "level_cut", "target", "target_share_of_delta", "anchor", "certified_share_of_map", "delta_per_condition",
             "n_conditions_tested", "n_tied_at_threshold", "n_tied_inside_zone")
PER_CONDITION_KEYS = ("coverage", "n_zone", "threshold", "upper_bound", "delta", "tested", "n_tied_at_threshold",
                      "n_tied_inside_zone")
# sidecar fields compared on their own: paths, texts that name a file, the anchor's history and the CRS (as a CRS)
SIDE_OWN = {"csv", "scores", "classes", "confidence", "how_to_label", "written_by", "crs", "anchor_rule",
            "first_budget", "condition"}
ZONE_OWN = {"sample", "zone_mask", "written_by"}
CSV_INTEGERS = {"index", "window_row", "window_col", "pixel_row", "pixel_col", "stratum", "map_class"}
RASTER_MAGIC = {".tif": (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"), ".tiff": (b"II*\x00", b"MM\x00*",
                b"II+\x00", b"MM\x00+"), ".npy": (b"\x93NUMPY",)}
ISSUERS_WORD = ("taken on the issuer's word: the labels (and any column the reviewer added), the map's date, when the "
                "certificate was made and which release wrote it, the paths it names, the condition names, and that "
                "the sample was drawn once (settings, or the map itself, can be varied before any label and one draw "
                "kept); alpha, delta, the rule and the level cut were fixed with the draw, before any label")
# files an operating system writes into a folder it shows; a bundle ignores them
OS_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
# the largest bundle file read, by kind: a bundle's JSON and CSV are small, and a file is never read without end
MAX_BYTES = {".json": 512 << 20, ".csv": 128 << 20}
# the TIFF tags the mask writer emits (stripped or tiled, georeferenced or not); any other tag, or a second page, is
# something a GIS may show that verify did not check
MASK_TAGS = {254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 339, 33550, 33922, 34735,
             34736, 34737}
RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
RELEASE = re.compile(r"^\d+\.\d+(\.\d+)?([abrc.+-][0-9A-Za-z.+-]*)?$")
# certify's error levels verify will run: a sequential rule's label count grows as 1/alpha, so a forged tiny alpha
# would hold verify for days (review of 24 October 2026)
ALPHA_RANGE, DELTA_RANGE = (1e-4, 0.5), (1e-6, 0.5)


class Unreadable(ValueError):
    """certificate.json itself cannot be read; every other fault is a failed check."""


def _load(path):
    """JSON as this module reads it: a key given twice, or NaN and Infinity, are refused (json.load keeps the last of
    two keys, so a first value no reader of the package sees could say something else)."""
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"{os.path.basename(path)} gives the key {k!r} twice")
            out[k] = v
        return out

    def constant(name):
        raise ValueError(f"{os.path.basename(path)} holds {name}, which strict JSON does not")
    _regular(path)
    with open(path) as f:
        try:
            return json.load(f, object_pairs_hook=pairs, parse_constant=constant)
        except RecursionError:
            raise ValueError(f"{os.path.basename(path)} is nested too deeply to be a file this package wrote") from None


def _regular(path):
    """Refuse anything but a regular file of a sane size before it is opened: a FIFO blocks open() for ever, and a
    symlink to /dev/zero or a sparse file is read without end (review of 24 October 2026)."""
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{os.path.basename(str(path))} is not a regular file (a link, a pipe or a device)")
    cap = MAX_BYTES.get(os.path.splitext(str(path))[1].lower())
    if cap and st.st_size > cap:
        raise ValueError(f"{os.path.basename(str(path))} is larger than a file of its kind this package writes")


def _crs_text(crs):
    """One spelling of a CRS for display: its EPSG code when that code is the CRS, else its WKT. Two spellings are
    compared as CRSs (rasterio's equality), never as text, since GDAL and PROJ releases spell a WKT differently."""
    if crs is None:
        return None
    code = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if code:
        # to_epsg matches loosely: a LAEA on GRS 1980 with no datum is given 3035 (ETRS89), which is not equal to it
        from rasterio.crs import CRS
        if CRS.from_epsg(code) == crs:
            return f"EPSG:{code}"
    return crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)


def _crs_spelled(text, geo):
    """A certificate's CRS spelled as this package spells that CRS: an EPSG code as the code, a WKT as GDAL writes it
    again, so its names (free text inside a WKT) are the CRS's own and not a claim (review of 24 October 2026)."""
    if geo is None or geo.get("crs") is None or text is None:
        return text is None
    want = _crs_text(geo["crs"])
    if want.startswith("EPSG:") or text.startswith("EPSG:"):
        return text == want
    from rasterio.crs import CRS
    try:
        return CRS.from_wkt(text).to_wkt() == geo["crs"].to_wkt()
    except Exception:
        return False


def _same_crs(text, crs):
    if text is None or crs is None:
        return text is None and crs is None
    from rasterio.crs import CRS
    try:
        return CRS.from_user_input(text) == crs
    except Exception:                                    # a spelling rasterio cannot read is not this CRS
        return False


def _terms(transform):
    return [round(float(v), 12) + 0.0 for v in list(transform)[:6]]


def pixel_digest(array, valid=None, geo=None):
    """sha256 of a raster as decoded: dtype, shape and little-endian bytes of the values, the valid-pixel mask, and
    the six transform terms (signed zeros made positive). A file rewritten with other compression or tiling keeps it;
    an edited pixel, another no-data value or a shifted grid changes it. The CRS is compared on its own, as a CRS."""
    h = hashlib.sha256()
    a = np.ascontiguousarray(np.asarray(array))
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    h.update(a.dtype.str.encode() + repr(a.shape).encode() + a.tobytes())
    if valid is not None:
        h.update(np.packbits(np.asarray(valid, bool).ravel()).tobytes())
    if geo is not None:
        h.update(repr(tuple(_terms(geo["transform"]))).encode())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(os.fspath(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nodata(value):
    return None if value is None else float(value)


def _plain_raster(path):
    """Refuse a raster file that is not a plain GeoTIFF or .npy by its name and its first bytes: GDAL opens a VRT, or
    VRT text named .tif, and follows the paths and URLs inside it (review of 24 October 2026)."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in RASTER_MAGIC:
        raise ValueError(f"{os.path.basename(str(path))} is not a GeoTIFF or .npy file")
    _regular(path)
    with open(path, "rb") as f:
        head = f.read(8)
    if not any(head.startswith(m) for m in RASTER_MAGIC[ext]):
        raise ValueError(f"{os.path.basename(str(path))} is not the {ext} file its name says")


@contextlib.contextmanager
def _closed_gdal():
    """GDAL reading one file: no sidecar files (.aux.xml, .ovr, .msk) and no network."""
    try:
        import rasterio
    except ImportError:                                  # .npy only
        yield
        return
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".none", GDAL_HTTP_MAX_RETRY="0"):
        yield


def raster_digest(path, nodata=None, plain=False):
    from .cli import read_raster
    if plain:
        _plain_raster(path)
        with _closed_gdal():
            a, valid, geo = read_raster(os.fspath(path), _nodata(nodata))
    else:
        a, valid, geo = read_raster(os.fspath(path), _nodata(nodata))
    return pixel_digest(a, valid, geo)


def map_digests(scores_path, nodata=None, confidence_path=None, plain=False):
    """{"map_sha256": the digest of the map as the package reads it (the scores, or a product's class map), and with
    a confidence band "confidence_sha256"}; plus the grid it was read on. `plain`: the paths came from a certificate,
    so only a plain GeoTIFF or .npy is opened, with no sidecar files and no network."""
    from .cli import read_raster
    if plain:
        _plain_raster(scores_path)
        with _closed_gdal():
            a, valid, geo = read_raster(os.fspath(scores_path), _nodata(nodata))
    else:
        a, valid, geo = read_raster(os.fspath(scores_path), _nodata(nodata))
    out = {"map_sha256": pixel_digest(a, valid, geo)}
    if confidence_path:
        out["confidence_sha256"] = raster_digest(confidence_path, plain=plain)
    return out, a.shape[-2:], geo


def names_given(names):
    """The names given to a condition layer's values (--condition-names) in one form: [[value, name], ...] by value.
    The sidecar records them so that the sample can be drawn again with the same names."""
    return None if not names else [[int(v), str(n)] for v, n in sorted(names.items())]


def _grid_level(value, at_or_above=False):
    """A target or an anchor as the zone grid level it means: the level within 1e-9 (a target), or the smallest level
    at or above it (an anchor, as certify_zone_sequential reads it). Values that mean the same level give one seed."""
    from .estimate import ZONE_GRID
    v = float(value)
    for g in ZONE_GRID:
        if (g >= v - 1e-12) if at_or_above else abs(g - v) < 1e-9:
            return repr(float(g))
    return repr(v)


def seed_settings(design, budget, patch, logits, confidence_range, target, anchor, confidence_sha256,
                  condition_sha256, plan=None):
    """The settings that fix a draw and what certify gives from it, in one canonical form: what `seed_from` hashes with
    the map's digest. The no-data value is in the map's digest (the valid pixels). A product's confidence range is here
    because it changes the ranking without always changing a window's validity (review of 24 October 2026). Condition
    names change nothing certify computes. `plan`: certify's alpha, delta, rule and level cut, fixed with the draw, so
    that a certificate shows they were chosen before any label (final review of 24 October 2026)."""
    return {"design": design, "budget": None if design == "sequential" else int(budget), "patch": int(patch),
            "logits": bool(logits),
            "confidence_range": None if confidence_range is None else [repr(float(v)) for v in confidence_range],
            "target": None if target is None else _grid_level(target),
            "anchor": None if anchor is None else _grid_level(anchor, at_or_above=True),
            "confidence_sha256": confidence_sha256, "condition_sha256": condition_sha256,
            "certify": None if not plan else {"alpha": repr(float(plan["alpha"])), "delta": repr(float(plan["delta"])),
                                              "rule": plan.get("rule"), "level_cut": plan.get("level_cut")}}


def seed_from(map_sha256, settings):
    """The seed `sample --seed-from-map` takes: 128 bits of sha256 over the map's digest and the settings."""
    h = hashlib.sha256((map_sha256 + json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    return int(h[:32], 16)


def draw_seed(scores_path, nodata, logits, confidence_path, confidence_range, condition_path, design, budget, patch,
              target, anchor, plan=None):
    """{"seed", "map_sha256", "settings"} for a sample drawn with --seed-from-map."""
    digests, _, _ = map_digests(scores_path, nodata, confidence_path)
    settings = seed_settings(design, budget, patch, logits, confidence_range, target, anchor,
                             digests.get("confidence_sha256"),
                             None if condition_path is None else raster_digest(condition_path), plan)
    return {"seed": seed_from(digests["map_sha256"], settings), "map_sha256": digests["map_sha256"],
            "settings": settings}


def _side_settings(side, confidence_sha256):
    """seed_settings as a sample's sidecar records them."""
    return seed_settings("class" if _by_class(side) else side.get("design"),
                         side.get("budget", len(side.get("indices", []))), side.get("patch", 4),
                         side.get("logits", False), side.get("confidence_range"),
                         (side.get("target") or {}).get("coverage"), side.get("anchor"), confidence_sha256,
                         (side.get("condition") or {}).get("sha256"), side.get("certify"))


def _map_paths(side, scores=None, confidence=None):
    """The map's paths as the sidecar names them, or the overrides when the files have moved."""
    if side.get("confidence"):
        return scores or side.get("classes") or side["scores"], confidence or side["confidence"]
    return scores or side["scores"], None


def _rfc3339(day, end=False):
    return f"{day.isoformat()}T{'23:59:59' if end else '00:00:00'}Z"


def _footprint(shape, geo):
    """(geometry, bbox) in longitude and latitude of the map's extent, or (None, None) for a map with no CRS. A map
    across the antimeridian gets a MultiPolygon split at 180 degrees (RFC 7946, 3.1.9) and a bbox with west > east."""
    if geo is None or geo.get("crs") is None:
        return None, None
    from rasterio.warp import transform_bounds
    t = geo["transform"]
    h, w = int(shape[0]), int(shape[1])
    xs = [t.c, t.c + t.a * w, t.c + t.b * h, t.c + t.a * w + t.b * h]
    ys = [t.f, t.f + t.d * w, t.f + t.e * h, t.f + t.d * w + t.e * h]
    w_, s_, e_, n_ = transform_bounds(geo["crs"], "EPSG:4326", min(xs), min(ys), max(xs), max(ys), densify_pts=21)
    if w_ > e_:
        poly = {"type": "MultiPolygon", "coordinates": [
            [[[w_, s_], [180.0, s_], [180.0, n_], [w_, n_], [w_, s_]]],
            [[[-180.0, s_], [e_, s_], [e_, n_], [-180.0, n_], [-180.0, s_]]]]}
    else:
        poly = {"type": "Polygon", "coordinates": [[[w_, s_], [e_, s_], [e_, n_], [w_, n_], [w_, s_]]]}
    return poly, [w_, s_, e_, n_]


def _name(text):
    """A condition name as a statement shows it: a JSON string, so a quote inside it cannot close it and write a
    coverage of its own (sample refuses quotes, backslashes and control characters in names)."""
    return json.dumps("".join(ch if ch.isprintable() else "?" for ch in str(text))[:64], ensure_ascii=False)


def printable(text, lines=False):
    """A text for a terminal or an agent: control characters replaced, so no field can rewrite what is shown; line
    breaks kept only where `lines`, so a field cannot start a line of its own (a forged 'verified:' line)."""
    return "".join(ch if ch.isprintable() or (lines and ch == "\n") else " " if ch in "\r\n\t" else "?"
                   for ch in str(text))


def _statement(zone, side):
    """The certificate's guarantee in one paragraph, as certify states it, with what certify says the population
    leaves out and, when windows tie at the threshold, that the zone is the mask."""
    a, d = 100 * float(zone["alpha"]), 100 * float(zone["delta"])
    patch = int(side.get("patch", 4))
    grid = side.get("grid")
    total = int(grid[0]) * int(grid[1]) if isinstance(grid, list) and len(grid) == 2 else None
    n_pop = zone.get("n_population")
    nodata = side.get("nodata")
    if zone.get("population_note"):
        left = " " + str(zone["population_note"])
    elif total and isinstance(n_pop, int) and total > n_pop:
        left = (f" {total - n_pop} of the map's {total} windows are left out as no-data"
                + ("" if nodata is None else f" (no-data value {nodata})") + ", and nothing here describes them.")
    else:
        left = "" if nodata is None else f" Pixels holding the no-data value {nodata} are left out."
    fixed = (" Alpha, delta and the rule were fixed with the draw, before any label." if side.get("certify") else "")
    tail = (fixed + f" It is about the zone's windows of {patch} px taken together, not each pixel; outside the zone "
            "nothing is certified, which does not mean wrong; it assumes the reviewer's labels are right and that the "
            "sample was drawn once; and delta counts every sample, those that certify nothing included, so among "
            "samples that certify a zone more than delta can be wrong.")
    if zone.get("by_condition"):
        tested = [f"{_name(n)} ({100 * e['coverage']:.0f}%)" for n, e in zone["per_condition"].items()
                  if e.get("coverage") is not None]
        per = "map class" if _by_class(side) else "input condition"
        return (f"Certified per {per} (names as the issuer gave them): in {', '.join(tested) or 'no ' + per}"
                f" the most confident share given is wrong at most {a:g}% of the time; these statements hold together "
                f"except on at most {d:g}% of samples like the recorded one ({zone.get('rule')} rule, "
                f"{zone.get('n_labelled')} labels)." + left + tail)
    look = " at every look of this sequential sample" if zone.get("rule") == "sequential" else ""
    bound = ("" if zone.get("upper_bound") is None else
             f"; the exact upper bound on the zone's error rate at that level is {100 * zone['upper_bound']:.1f}%")
    tied, inside = int(zone.get("n_tied_at_threshold") or 0), int(zone.get("n_tied_inside_zone") or 0)
    tie = ("" if tied <= inside else
           f" The zone is the mask, not every window at or above the threshold: {tied} windows share the threshold's "
           f"confidence and {inside} of them are inside.")
    return (f"The {100 * zone['coverage']:.0f}% most confident of the {n_pop} windows counted on this map "
            f"({zone['n_zone']} windows) are wrong at most {a:g}% of the time; this statement fails on at most {d:g}% "
            f"of samples like the recorded one{look} ({zone.get('rule')} rule, {zone.get('n_labelled')} random "
            f"labels{bound})." + tie + left + tail)


def _item(zone, side, grid, files, mask_name, cond_name, issuer, geometry, bbox):
    """The STAC Item of a certificate: built from a certify result, the sample's sidecar, the map's grid and the
    bundle's files, with the fields only the issuer can give (`issuer`: the times, when it was made, where the map
    was, which package wrote it). build writes it; verify builds it again and compares it."""
    props = {**issuer["times"], "created": issuer["created"], "zc:version": CERT_VERSION}
    props.update({f"zc:{k}": zone[k] for k in ZONE_KEYS if zone.get(k) is not None})
    props.update({"zc:design": side.get("design"), "zc:seed": side.get("seed"), "zc:seed_rule": side.get("seed_rule"),
                  "zc:patch": int(side.get("patch", 4)), "zc:statement": _statement(zone, side),
                  "zc:iso19157": {"conformance_result": {
                      "specification": f"the error rate of the zone's windows is at most {float(zone['alpha']):g}",
                      "pass": True, "explanation": f"fails on at most {float(zone['delta']):g} of samples"},
                      "coverage_result": mask_name}})
    if zone.get("population_note"):
        props["zc:population_note"] = zone["population_note"]
    if zone.get("by_condition"):
        props["zc:per_condition"] = {n: {k: e.get(k) for k in PER_CONDITION_KEYS if k in e}
                                     for n, e in zone["per_condition"].items()}
    props.update({"zc:map": {**grid, "path": issuer["map_path"], "confidence_path": issuer["confidence_path"]},
                  "zc:files": files, "zc:written_by": issuer["written_by"]})
    mask_type = "image/tiff; application=geotiff" if mask_name.endswith(".tif") else "application/octet-stream"
    assets = {"zone_mask": {"href": mask_name, "type": mask_type, "roles": ["data"],
                            "title": "the certified zone on the map's window grid (1 inside); the zone is this mask, "
                                     "not a threshold rule"},
              "labels": {"href": BUNDLE["labels"], "type": "text/csv", "roles": ["metadata"]},
              "sample": {"href": BUNDLE["sample"], "type": "application/json", "roles": ["metadata"]},
              "zone": {"href": BUNDLE["zone"], "type": "application/json", "roles": ["metadata"]}}
    if cond_name:
        assets["condition"] = {"href": cond_name, "roles": ["metadata"],
                               "title": "the input-condition layer the sample was drawn with"}
    return {"type": "Feature", "stac_version": "1.0.0", "stac_extensions": [],
            "id": f"zone-{grid['map_sha256'][:16]}", "geometry": geometry, **({"bbox": bbox} if bbox else {}),
            "properties": props, "assets": assets,
            "links": [{"rel": "derived_from", "href": issuer["map_path"], "title": "the map certified"}]}


def _map_grid(digests, shape, geo, crs_text=None):
    """The zc:map fields that bind a certificate to its map: the digests, the shape, the CRS and the transform."""
    return {**digests, "shape": list(map(int, shape)),
            "crs": crs_text if crs_text is not None else (None if geo is None else _crs_text(geo["crs"])),
            "transform": None if geo is None else _terms(geo["transform"])}


def _declared(path):
    """(bands, rows, columns, bytes per value) a plain GeoTIFF or .npy declares, read from its header alone; a
    non-numeric .npy is refused."""
    _plain_raster(path)
    if str(path).lower().endswith(".npy"):
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.dtype.kind not in "biuf" or a.ndim not in (2, 3):
            raise ValueError(f"{os.path.basename(str(path))} is not a numeric raster")
        return (1 if a.ndim == 2 else int(a.shape[0]), int(a.shape[-2]), int(a.shape[-1]), int(a.dtype.itemsize))
    import rasterio
    with _closed_gdal(), rasterio.open(path, driver="GTiff") as src:
        return (src.count, src.height, src.width, max(np.dtype(d).itemsize for d in src.dtypes))


def _layer_on_grid(path, shape):
    """Refuse a condition layer that is not one band of plain numbers on the map's grid, before it is read: its header
    may declare 20,000 bands or a stack the size of memory on a few kilobytes of disk."""
    bands, h, w, size = _declared(path)
    if bands != 1 or (h, w) != tuple(shape) or size > 8:
        raise ValueError("the bundled condition layer is not one band of numbers on the map's grid")


def _bundle_file(name):
    """A name a bundle's file can have: what build may replace when it writes a bundle again."""
    stem, ext = os.path.splitext(name)
    return (name in ("certificate.json", *BUNDLE.values()) or (stem in ("zone_mask", "condition") and
                                                                ext.lower() in RASTER_MAGIC))


def _by_class(side):
    """A sample of the class design: its strata are the map's classes, read from the map, with no layer of their own."""
    return (side.get("condition") or {}).get("by") == "map class"


def _cond_name(side):
    cond = side.get("condition")
    if not cond or _by_class(side):
        return None
    return "condition" + os.path.splitext(str(cond["source"]))[1].lower()


def build(zone_path, out_dir, date, scores=None, confidence=None):
    """Write the certificate for the result `certify` wrote at `zone_path` into `out_dir`, for a map of `date` (an
    ISO date or start/end interval), and verify it. Returns the certificate as a dict. Refuses a result that certifies
    nothing or reads the labels twice (--also-accept), a sample whose seed was not taken from the map, a condition
    layer that is not a GeoTIFF or .npy, an out_dir that is a file or holds other files, a source that would be
    overwritten by its own copy, and a bundle that does not verify."""
    from .compare import _period
    zone = _load(zone_path)
    certified = zone.get("coverage") is not None or zone.get("certified_share_of_map")
    if not certified:
        raise ValueError(f"{zone_path} certifies no zone, so there is nothing to put in a certificate")
    if not zone.get("sample") or not zone.get("zone_mask"):
        raise ValueError(f"{zone_path} is not a result `certify` wrote (no sample or zone mask recorded)")
    if "also_accept" in zone:
        raise ValueError(f"{zone_path} was certified with --also-accept; a certificate states one reading of the "
                         "labels: certify again without it")
    period = _period(date, "date")
    if period is None:
        raise ValueError("the map's date is needed: a certificate holds for the map of one date or period")
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError(f"--out {out_dir} is a file; name the directory to write the bundle into")
    held = [n for n in (os.listdir(out_dir) if os.path.isdir(out_dir) else [])
            if n not in OS_FILES and not n.startswith("._")]
    if held and ("certificate.json" not in held or not all(_bundle_file(n) for n in held)):
        # only a bundle's own files are ever replaced: a user's condition.tif in a folder that holds no bundle is theirs
        raise ValueError(f"{out_dir} already holds files that are not a certificate bundle ({', '.join(held[:3])}"
                         f"{' ...' if len(held) > 3 else ''}); name a new or empty directory")
    csv_path = zone["sample"]
    side_path = csv_path[:-4] + ".json" if csv_path.endswith(".csv") else csv_path + ".json"
    side = _load(side_path)
    if side.get("seed_rule") != "from_map" or not side.get("certify"):
        raise ValueError(f"{csv_path} was not drawn for a certificate: a certificate needs a sample drawn with "
                         "`sample --seed-from-map --alpha A`, whose seed anyone holding the map can derive again from the "
                         "map, the draw's settings and certify's alpha, delta and rule, so that it shows they were all "
                         "fixed before the labels. Draw a new sample with it")
    cond = None if _by_class(side) else side.get("condition")
    copies = [(csv_path, BUNDLE["labels"]), (side_path, BUNDLE["sample"]), (zone_path, BUNDLE["zone"])]
    if cond:
        if os.path.splitext(str(cond["source"]))[1].lower() not in RASTER_MAGIC:
            raise ValueError(f"the condition layer {cond['source']} is not a GeoTIFF or .npy file, which a bundle "
                             "carries whole; write it as a GeoTIFF and draw the sample again")
        copies.append((cond["source"], _cond_name(side)))
    for src, name in copies:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and os.path.samefile(src, dst):
            raise ValueError(f"{src} is the bundle's own {name}; write the bundle into another directory")
    map_path, conf_path = _map_paths(side, scores, confidence)
    digests, shape, geo = map_digests(map_path, side.get("nodata"), conf_path)
    if digests["map_sha256"] != side.get("map_sha256"):
        raise ValueError(f"{map_path} is not the map the sample was drawn from: its digest differs from the one the "
                         "sample recorded")
    final = out_dir
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    out_dir = tempfile.mkdtemp(prefix=".certificate-", dir=parent)   # written and verified here, then moved in place
    try:
        item = _write_bundle(zone, side, copies, out_dir, map_path, conf_path, digests, shape, geo, period)
        os.makedirs(final, exist_ok=True)
        for name in os.listdir(final):                   # a bundle written again keeps nothing of the earlier one
            if _bundle_file(name) and os.path.isfile(os.path.join(final, name)):
                os.remove(os.path.join(final, name))
        for name in os.listdir(out_dir):
            os.replace(os.path.join(out_dir, name), os.path.join(final, name))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return item


def _write_bundle(zone, side, copies, out_dir, map_path, conf_path, digests, shape, geo, period):
    """The bundle written into `out_dir` and verified there; ValueError when it does not verify."""
    from .cli import write_raster
    for src, name in copies:
        shutil.copyfile(src, os.path.join(out_dir, name))
    mask = np.load(zone["zone_mask"])
    mask_path = write_raster(os.path.join(out_dir, "zone_mask.tif"), mask.astype(bool), geo,
                             int(side.get("patch", 4)), None)
    start, end = period
    from . import __version__
    issuer = {"times": ({"datetime": _rfc3339(start)} if start == end else
                        {"datetime": None, "start_datetime": _rfc3339(start), "end_datetime": _rfc3339(end, end=True)}),
              "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "map_path": os.path.abspath(map_path),
              "confidence_path": None if conf_path is None else os.path.abspath(conf_path),
              "written_by": {"package": "olmoearth-inferencex", "version": __version__, "command": "certificate"}}
    names = [n for _, n in copies] + [os.path.basename(mask_path)]
    files = {n: file_sha256(os.path.join(out_dir, n)) for n in names}
    item = _item(zone, side, _map_grid(digests, shape, geo), files, os.path.basename(mask_path), _cond_name(side),
                 issuer, *_footprint(shape, geo))
    cert_path = os.path.join(out_dir, "certificate.json")
    with open(cert_path, "w") as f:
        json.dump(item, f, indent=1)
    res = verify(cert_path, map_path, conf_path)          # what anyone will check, checked before it is handed over
    if not res["verified"]:
        failed = "; ".join(f"{n}: {c['detail']}" for n, c in res["checks"].items() if c["ok"] is not True)
        raise ValueError(f"the bundle does not verify, so it is no certificate, and nothing was written: {failed}")
    return item


def _check(ok, detail):
    return {"ok": ok, "detail": detail}


def _redraw(side, map_path, conf_path, cond_path, tmp, closed=False):
    """The sample drawn again with --seed-from-map from the map, the recorded settings and the bundled condition layer:
    (sidecar of the new draw, path of its CSV, None) or (None, None, the reason it could not be drawn again)."""
    from . import cli
    design = side.get("design")
    if design not in ("random", "sequential", "condition"):
        return None, None, f"the {design} design is not one a certificate rests on"
    out = os.path.join(tmp, "redraw.csv")
    argv = ["sample", map_path, "--design", "class" if _by_class(side) else design,
            "--budget", str(int(side.get("budget", len(side["indices"])))),
            "--seed-from-map", "--patch", str(int(side.get("patch", 4))), "--out", out]
    if side.get("logits") is True:
        argv.append("--logits")
    if side.get("nodata") is not None:
        argv.append("--nodata=" + _arg(side["nodata"]))
    if conf_path:
        argv += ["--confidence", conf_path]
        if side.get("confidence_range"):
            argv += ["--confidence-range", *[_arg(v) for v in side["confidence_range"]]]
    if side.get("target"):
        argv.append("--target=" + _arg(side["target"]["coverage"]))
    if design == "sequential" and side.get("anchor") is not None:
        argv.append("--anchor=" + _arg(side["anchor"]))
    plan = side.get("certify") if isinstance(side.get("certify"), dict) else {}
    if plan:                                           # certify's options, fixed with the draw, enter its seed
        argv += ["--alpha=" + _arg(plan["alpha"]), "--delta=" + _arg(plan["delta"])]
        if plan.get("rule") in ("prefix", "bonferroni"):
            argv.append("--rule=" + plan["rule"])
        if plan.get("level_cut") and design != "sequential":
            argv.append("--level-cut=" + str(plan["level_cut"]))
    cond = side.get("condition")
    if cond and _by_class(side):
        floor = re.search(r"at least (\d+) per class", str(cond.get("allocation_rule")))
        argv += ["--min-per-class", floor.group(1) if floor else "-1"]
    elif cond:
        argv += ["--condition", cond_path]
    if cond and cond.get("names_given"):
        argv += ["--condition-names", *[f"{int(v)}={n}" for v, n in cond["names_given"]]]
    try:
        with contextlib.redirect_stdout(io.StringIO()), (_closed_gdal() if closed else contextlib.nullcontext()):
            cli.main(argv)
        return _load(os.path.join(tmp, "redraw.json")), out, None
    except (SystemExit, OSError, ValueError) as exc:
        return None, None, f"sample refused to draw it again: {exc}"


def _arg(value):
    """A number as argparse takes it back exactly: positional notation, since it reads '-3.4e+38' as an option."""
    v = float(value)
    return repr(v) if not math.isfinite(v) else np.format_float_positional(v, unique=True, trim="0")


def _differ(a, b, own=(), path="", loose=()):
    """The places where two JSON values differ. Types must agree (true is not 1, 1 is not 1.0); strings, integers and
    booleans must be equal; floats must be finite and agree to 1e-10 relative, or 1e-9 absolute under a top-level key
    in `loose` (a footprint, which PROJ releases compute a little differently). Top-level keys in `own` are skipped."""
    if type(a) is not type(b):
        return [path.rstrip(".") or "(value)"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if path == "" and k in own:
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k}")
            else:
                out += _differ(a[k], b[k], (), f"{path}{k}.", loose if path else (loose if k in loose else ()))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [path.rstrip(".") or "(list)"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differ(x, y, (), f"{path}{i}.", loose)]
    if isinstance(a, float):
        ok = math.isfinite(a) and math.isfinite(b) and (
            abs(a - b) <= 1e-9 if loose else abs(a - b) <= 1e-10 * max(abs(a), abs(b)))
        return [] if ok else [path.rstrip(".") or "(value)"]
    return [] if a == b else [path.rstrip(".") or "(value)"]


def _rounding_tolerance(text):
    from .cli import _rounding_tolerance as tol
    return tol(text)


def csv_rows(path):
    """A sample CSV's rows as dicts, refusing a column name given twice or empty, and a row longer than the header:
    csv.DictReader keeps the last copy of a repeated column, where pandas or a spreadsheet show the first, so the
    bundle could show an auditor other labels than the ones certified (review of 24 October 2026)."""
    _regular(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return []
        if len(set(header)) != len(header) or any(not str(h).strip() for h in header):
            raise ValueError(f"{os.path.basename(str(path))} names a column twice or leaves one unnamed")
        rows = []
        for line in reader:
            if len(line) > len(header):
                raise ValueError(f"{os.path.basename(str(path))} has a row with more fields than columns")
            rows.append(dict(zip(header, line + [""] * (len(header) - len(line)))))
    return rows


def _csv_differs(rows, again, pixel=None):
    """The columns where the labelled CSV is not the sample drawn again, all but `wrong` and the columns a reviewer
    added: integers equal, the confidence and the coordinates to the rounding they were written with (a spreadsheet
    saves 502060.0 as 502060), but never looser than 5e-4 for the confidence or half a pixel (`pixel`) for x and y,
    so a value written '0e7' is not "the same window"; names equal."""
    if len(rows) != len(again):
        return ["the number of rows"]
    bad = set()
    for r, a in zip(rows, again):
        for col, want in a.items():
            if col == "wrong" or col is None:
                continue
            got = r.get(col)
            if got == want:
                continue
            try:
                g, w = float(got), float(want)
            except (TypeError, ValueError):
                bad.add(col)
                continue
            if col in CSV_INTEGERS:
                ok = g == w and g.is_integer()
            else:
                cap = 5e-4 if col == "confidence" else (0.5 * pixel if pixel else 1e-9 * max(1.0, abs(w)))
                tol = max(min(_rounding_tolerance(got), cap), 1e-9 * max(1.0, abs(w)))
                ok = math.isfinite(g) and abs(g - w) <= tol
            if not ok:
                bad.add(col)
    return sorted(bad)


def _tiff_pages(path):
    """The tag numbers of each page (IFD) of a TIFF, read from its structure, not through GDAL, which shows the first
    page only and ignores tags other readers obey (a second page, an Orientation)."""
    with open(path, "rb") as f:
        b = f.read(16)
        bo = "<" if b[:2] == b"II" else ">"
        big = struct.unpack(bo + "H", b[2:4])[0] == 43
        off = struct.unpack(bo + ("Q" if big else "I"), b[8:16] if big else b[4:8])[0]
        pages, seen = [], set()
        while off and off not in seen and len(pages) < 4:
            seen.add(off)
            f.seek(off)
            n = struct.unpack(bo + ("Q" if big else "H"), f.read(8 if big else 2))[0]
            if n > 64:
                return [list(range(n))]                    # no mask this package writes has that many tags
            es = 20 if big else 12
            entries = f.read(n * es)
            pages.append([struct.unpack(bo + "H", entries[i * es:i * es + 2])[0] for i in range(n)])
            off = struct.unpack(bo + ("Q" if big else "I"), f.read(8 if big else 4))[0]
    return pages


def _read_mask(path, shape):
    """The bundle's zone mask, checked to be a plain one before and after it is read: a single-band uint8 GeoTIFF of 0
    and 1 with no no-data, scale, offset, overviews, colour table, internal mask or tags beyond AREA_OR_POINT=Area (a GIS
    shows any of these, and the full-resolution pixels alone would not), on the window grid's shape; or a 2-D .npy.
    Returns (mask, transform or None, CRS or None)."""
    _plain_raster(path)
    if path.endswith(".npy"):
        head = np.load(path, mmap_mode="r", allow_pickle=False)
        if head.ndim != 2 or tuple(head.shape) != tuple(shape) or head.dtype.kind not in "bu":
            raise ValueError("the .npy mask is not a 2-D 0/1 array of the window grid's shape")
        m = np.asarray(head)
        if not np.isin(m, (0, 1)).all():
            raise ValueError("the .npy mask holds values other than 0 and 1")
        return m, None, None
    pages = _tiff_pages(path)
    if len(pages) != 1 or not set(pages[0]) <= MASK_TAGS:
        raise ValueError("the mask GeoTIFF has more than one page, or tags the mask writer does not write (an "
                         "orientation, a second georeferencing, metadata)")
    import rasterio
    from rasterio.enums import ColorInterp, MaskFlags
    with _closed_gdal(), rasterio.open(path, driver="GTiff") as src:
        if src.subdatasets or src.gcps[0] or src.rpcs is not None or set(src.tag_namespaces()) - {"IMAGE_STRUCTURE", "DERIVED_SUBDATASETS"}:
            raise ValueError("the mask GeoTIFF carries pages, control points or metadata a mask has none of")
        if (src.height, src.width) != tuple(shape):
            raise ValueError(f"the mask is {src.height} x {src.width}, not the window grid's {shape[0]} x {shape[1]}")
        try:
            src.colormap(1)
            has_colormap = True
        except ValueError:
            has_colormap = False
        plain = (src.count == 1 and src.dtypes == ("uint8",) and src.nodata is None and src.overviews(1) == []
                 and src.scales == (1.0,) and src.offsets == (0.0,) and src.mask_flag_enums == ([MaskFlags.all_valid],)
                 and src.tags() in ({}, {"AREA_OR_POINT": "Area"}) and src.tags(1) == {}
                 and src.descriptions == (None,) and src.units in ((None,), ("",))
                 and src.colorinterp == (ColorInterp.gray,) and not has_colormap)
        if not plain:
            raise ValueError("the mask GeoTIFF carries more than one band of 0 and 1 (a no-data value, a scale or "
                             "offset, overviews, a colour table, an internal mask, tags or descriptions)")
        m = src.read(1)
        if not np.isin(m, (0, 1)).all():
            raise ValueError("the mask holds values other than 0 and 1")
        return m, src.transform, src.crs


def _short(diff):
    return ", ".join(diff[:8]) + (" ..." if len(diff) > 8 else "")


def verify(cert_path, scores=None, confidence=None):
    """Check a certificate by building it again from the map, the bundled labels and the recorded settings. Returns
    {"verified": bool, "checks": {name: {ok, detail}}, "notes": [...], "statement": the statement as certify run
    again gives it (None when it could not be run), "mask": the zone mask certify run again gives (None likewise),
    "map_read": the map file read, "certificate": the item}. A check that could not be made is ok None and leaves the
    certificate unverified. Raises Unreadable only when certificate.json itself cannot be read."""
    try:
        item = _load(cert_path)
        p = item["properties"]
        if not isinstance(p, dict):
            raise TypeError("properties is not an object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Unreadable(f"{cert_path} is not a certificate this package can read: {exc}") from None
    base = os.path.dirname(os.path.abspath(cert_path))
    checks, notes = {}, []
    out = {"verified": False, "checks": checks, "notes": notes, "statement": None, "mask": None, "map_read": None,
           "certificate": item}
    try:
        _verify(item, p, base, os.path.basename(cert_path), scores, confidence, checks, notes, out)
    except Exception as exc:                              # a verifier fails closed: anything unforeseen is a failure
        checks["read"] = _check(False, f"the certificate or its bundle could not be read as written: "
                                f"{type(exc).__name__}: {exc}")
    out["verified"] = bool(checks) and all(c["ok"] is True for c in checks.values())
    return out


def _verify(item, p, base, cert_name, scores, confidence, checks, notes, out):
    from . import cli
    side = _load(os.path.join(base, BUNDLE["sample"]))
    zone = _load(os.path.join(base, BUNDLE["zone"]))
    cond = None if _by_class(side) else side.get("condition")
    cond_name = _cond_name(side)
    m = p.get("zc:map") or {}
    product = bool(side.get("confidence"))
    if product != ("confidence_sha256" in m):
        checks["map"] = _check(False, "the sample and the certificate disagree on whether the map has a confidence band")
        return
    if confidence and not product:
        checks["map"] = _check(None, "this map has no confidence band; --confidence does not apply")
        return
    if product and scores and not confidence:
        checks["map"] = _check(None, "this map is a product with a confidence band, which decides the zone: pass "
                               "--confidence with your copy of the band as well as --scores")
        return
    map_path, conf_path = scores or m.get("path"), (confidence or m.get("confidence_path")) if product else None
    from_cert = not scores
    if not product and m.get("confidence_path") is not None:
        checks["map"] = _check(False, "the certificate names a confidence band for a map that has none")
        return
    if not isinstance(map_path, str) or (product and not isinstance(conf_path, str)):
        checks["map"] = _check(None, "no map to check against: pass --scores (and --confidence for a product)")
        return
    if from_cert:
        read = "the map and its band were" if product and not confidence else "the map was"
        notes.append(f"{read} read from the paths the certificate names, which shows only that the certificate agrees "
                     "with those files; pass --scores" + (" and --confidence" if product else "")
                     + " with your own copy to check it is that map")
        try:                                              # a file the issuer chose: plain, and of a size it declares
            for path in [map_path] + ([conf_path] if product and not confidence else []):
                bands, h, w, size = _declared(path)
                if [h, w] != list(m.get("shape") or []) or bands * h * w * size > 8 << 30:
                    raise ValueError(f"{os.path.basename(path)} is not on the grid the certificate states, or is "
                                     "larger than verify reads from a path a certificate names")
        except (OSError, ValueError) as exc:
            checks["map"] = _check(None, f"the map the certificate names was not read: {exc}")
            return
    out["map_read"] = os.path.abspath(map_path)

    # the map
    try:
        digests, shape, geo = map_digests(map_path, side.get("nodata"), conf_path, plain=from_cert)
        if product and confidence and from_cert:             # the band is the verifier's own
            digests["confidence_sha256"] = raster_digest(conf_path)
    except (SystemExit, OSError, ValueError) as exc:
        checks["map"] = _check(None, f"the map could not be read: {exc}")
        return
    crs_ok = _same_crs(m.get("crs"), None if geo is None else geo["crs"]) and _crs_spelled(m.get("crs"), geo)
    grid = _map_grid(digests, shape, geo, m.get("crs") if crs_ok else None)
    same = crs_ok and not _differ(grid, {k: m.get(k) for k in grid}) and digests["map_sha256"] == side.get("map_sha256")
    checks["map"] = _check(same, "the map read is the map certified: its pixels, valid mask, CRS and transform" if same
                           else "the map read is not the map certified: its digest, CRS or grid differs")

    # the bundle: its own files, each as recorded, and nothing else (no sidecar file a GIS would read with the mask)
    mask_name = "zone_mask.tif" if geo is not None else "zone_mask.npy"
    expected = [*BUNDLE.values(), mask_name, *([cond_name] if cond_name else [])]
    present = sorted(os.listdir(base))
    extra = [n for n in present if n not in expected and n != cert_name and n not in OS_FILES
             and not n.startswith("._")]
    files = {}
    for n in expected:
        try:
            _regular(os.path.join(base, n))
            files[n] = file_sha256(os.path.join(base, n))
        except (OSError, ValueError):
            pass
    recorded = p.get("zc:files") if isinstance(p.get("zc:files"), dict) else {}
    bad = sorted(n for n in set(expected) | set(recorded) if files.get(n) != recorded.get(n))
    checks["bundle"] = _check(not bad and not extra, "the bundle holds its own files, each as recorded" if not bad
                              and not extra else "the bundle's files are not the ones recorded: "
                              + ", ".join([f"{n} (changed, missing or unlisted)" for n in bad]
                                          + [f"{n} (not part of a bundle)" for n in extra]))
    cond_path = os.path.join(base, cond_name) if cond_name else None
    if cond:
        try:
            _layer_on_grid(cond_path, shape)                    # before reading what its header declares
            c_ok = raster_digest(cond_path, plain=True) == cond.get("sha256")
            detail = ("the condition layer is the one the sample was drawn with" if c_ok else
                      "the bundled condition layer is not the one the sample recorded")
        except (SystemExit, OSError, ValueError) as exc:
            c_ok, detail = False, f"the bundled condition layer could not be read: {exc}"
        checks["condition"] = _check(c_ok, detail)
        if not c_ok:
            return

    # the seed: taken from the map and the settings
    if side.get("seed_rule") != "from_map" or not side.get("certify"):
        checks["seed"] = _check(False, "the sample was not drawn for a certificate (sample --seed-from-map --alpha), so "
                                "nothing shows the draw and certify's options were fixed before the labels")
    else:
        settings = _side_settings(side, digests.get("confidence_sha256"))
        ok = (type(side.get("seed")) is int and side["seed"] == seed_from(digests["map_sha256"], settings)
              and side.get("seed_settings") == settings)
        checks["seed"] = _check(ok, "the seed is the one the map and the sample's settings give" if ok else
                                "the seed is not the one the map and the sample's settings give")

    with tempfile.TemporaryDirectory() as tmp:
        # the draw: the sample drawn again must be this one, sidecar and CSV
        drawn, drawn_csv, why = _redraw(side, map_path, conf_path, cond_path, tmp, closed=from_cert)
        if drawn is None:
            checks["draw"] = _check(None, f"the sample was not drawn again: {why}")
        else:
            diff = _differ(side, drawn, SIDE_OWN)
            diff += [k for k in ("classes", "confidence") if bool(side.get(k)) != bool(drawn.get(k))]
            sc, dc = side.get("condition"), drawn.get("condition")
            if (sc is None) != (dc is None):
                diff.append("condition")
            elif sc is not None:
                diff += [f"condition.{d}" for d in _differ(sc, dc, {"source"})]
            sc_crs, map_crs = side.get("crs"), None if geo is None else geo["crs"]
            if not (sc_crs == str(map_crs) or (map_crs is None and sc_crs in (None, "None"))
                    or (map_crs is not None and _same_crs(sc_crs, map_crs))):
                diff.append("crs")
            want_h = str(drawn.get("how_to_label", "")).replace(os.path.basename(drawn_csv),
                                                                  os.path.basename(str(side.get("csv", ""))))
            if side.get("how_to_label") != want_h:
                diff.append("how_to_label")
            wb = side.get("written_by")
            if not (isinstance(wb, dict) and set(wb) == {"package", "version", "command"}
                    and wb["package"] == "olmoearth-inferencex" and wb["command"] == "sample"
                    and isinstance(wb["version"], str)):
                diff.append("written_by")
            diff += _anchor_history(side)
            diff += [k for k in ("csv", "scores", "classes", "confidence") if k in side and not isinstance(side[k], str)]
            if sc is not None and not isinstance(sc.get("source"), str):
                diff.append("condition.source")
            px = side.get("pixel_size")
            pixel = (min(float(v) for v in px) if isinstance(px, list) and px and
                     all(isinstance(v, (int, float)) for v in px) else None)
            diff += [f"the CSV's {c}" for c in _csv_differs(csv_rows(os.path.join(base, BUNDLE["labels"])),
                                                              csv_rows(drawn_csv), pixel)]
            checks["draw"] = _check(not diff, "the sample is exactly the one the map, the settings and the seed draw"
                                    if not diff else "the sample is not the one the map, the settings and the seed "
                                    f"draw: {_short(diff)}")

        plan = side.get("certify") if isinstance(side.get("certify"), dict) else {}
        alpha, delta = plan.get("alpha"), plan.get("delta")
        if not (type(alpha) is float and type(delta) is float and ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]
                and DELTA_RANGE[0] <= delta <= DELTA_RANGE[1]):
            checks["certify"] = _check(None, f"certify was not run again: alpha and delta must lie in "
                                       f"[{ALPHA_RANGE[0]:g}, {ALPHA_RANGE[1]:g}] and [{DELTA_RANGE[0]:g}, "
                                       f"{DELTA_RANGE[1]:g}], the error levels verify runs")
            return
        # certify again from the labels, with the map as the verifier gives it and the options the zone JSON records
        side_run = {k: v for k, v in side.items() if k not in ("classes", "confidence")}
        side_run["scores"] = os.path.abspath(map_path)
        if product:
            side_run.update({"classes": os.path.abspath(map_path), "confidence": os.path.abspath(conf_path)})
        shutil.copyfile(os.path.join(base, BUNDLE["labels"]), os.path.join(tmp, "labels.csv"))
        with open(os.path.join(tmp, "labels.json"), "w") as f:
            json.dump(side_run, f)
        # alpha, delta, the rule and the level cut are the plan the sample fixed, which certify reads and keeps to
        argv = ["certify", os.path.join(tmp, "labels.csv"), "--out", os.path.join(tmp, "again.json"), "--scores",
                map_path]
        if conf_path:
            argv += ["--confidence", conf_path]
        try:
            with contextlib.redirect_stdout(io.StringIO()), (_closed_gdal() if from_cert else
                                                             contextlib.nullcontext()):
                cli.main(argv)
            again = _load(os.path.join(tmp, "again.json"))
            again_mask = np.load(again["zone_mask"])
        except (SystemExit, OSError, ValueError, KeyError) as exc:
            checks["certify"] = _check(None, f"certify could not be run again: {exc}")
            return
        diff = _differ(zone, again, ZONE_OWN)
        diff += [k for k in ("sample", "zone_mask") if not isinstance(zone.get(k), str)]
        wbz = zone.get("written_by")
        if not (isinstance(wbz, dict) and set(wbz) == {"package", "version", "command"}
                and wbz["package"] == "olmoearth-inferencex" and wbz["command"] == "certify"
                and isinstance(wbz["version"], str)):
            diff.append("written_by")
        checks["certify"] = _check(not diff, "certify, run again on the labels, writes the zone JSON recorded" if not
                                   diff else f"certify, run again, differs from the recorded zone JSON in {_short(diff)}")
        out["statement"], out["mask"] = _statement(again, side), again_mask

    # the certificate itself, built again from that run and compared; only the issuer's fields are carried over
    times = ({"datetime": p.get("datetime")} if p.get("datetime") is not None else
             {"datetime": None, "start_datetime": p.get("start_datetime"), "end_datetime": p.get("end_datetime")})
    wb = p.get("zc:written_by")
    wb_ok = (isinstance(wb, dict) and set(wb) == {"package", "version", "command"} and wb["command"] == "certificate"
             and wb["package"] == "olmoearth-inferencex" and isinstance(wb["version"], str))
    paths_ok = _issuer_fields_ok(p, m, times, product)
    issuer = {"times": times, "created": p.get("created"), "map_path": m.get("path"),
              "confidence_path": m.get("confidence_path"), "written_by": wb}
    want = _item(again, side, grid, files, mask_name, cond_name, issuer, *_footprint(shape, geo))
    diff = _differ(want, item, loose=("geometry", "bbox"))
    if not wb_ok:
        diff.append("properties.zc:written_by")
    if not paths_ok:
        diff.append("the issuer's fields (dates, paths)")
    checks["statement"] = _check(not diff, "the certificate, built again from certify's result, is the one written"
                                 if not diff else "the certificate states what the map, the sample and certify do not "
                                 f"give: {_short(diff)}")

    # the zone mask: certify's zone, on the map's window grid
    try:
        bmask, mtrans, mcrs = _read_mask(os.path.join(base, mask_name), again_mask.shape)
        ok = np.array_equal(bmask.astype(bool), again_mask.astype(bool))
        if ok and mtrans is not None:
            from rasterio.transform import Affine

            from .cli import affine_apply
            ok = (_same_crs(_crs_text(geo["crs"]), mcrs)
                  and not _differ(_terms(mtrans), _terms(affine_apply(geo["transform"],
                                                                      Affine.scale(int(side.get("patch", 4)))))))
        checks["mask"] = _check(ok, "the zone mask is the zone certify gives, on the map's window grid" if ok else
                                "the bundle's zone mask is not the zone certify gives on the map's window grid")
    except (OSError, ValueError) as exc:
        checks["mask"] = _check(False, f"the zone mask is not a plain one: {exc}")
    notes.append(ISSUERS_WORD)


def _issuer_fields_ok(p, m, times, product):
    """The issuer's fields in the form build writes them: RFC 3339 times (one datetime, or a start no later than its
    end), a release number, paths as text and a band path only for a product. Their content stays the issuer's word."""
    def when(v):
        return isinstance(v, str) and bool(RFC3339.match(v))
    if times.get("datetime") is not None:
        t_ok = when(times["datetime"])
    else:
        t_ok = (when(times.get("start_datetime")) and when(times.get("end_datetime"))
                and times["start_datetime"] <= times["end_datetime"])
    wb = p.get("zc:written_by") or {}
    return (t_ok and when(p.get("created")) and isinstance(m.get("path"), str) and bool(RELEASE.match(str(wb.get("version"))))
            and (isinstance(m.get("confidence_path"), str) if product else m.get("confidence_path") is None))


def _anchor_history(side):
    """A sequential sample's anchor history as `sample` writes it: the rule that set the anchor (declared, or computed
    from a first budget, alpha and delta, which must give this anchor) and a first budget no larger than the budget;
    neither on another design."""
    if side.get("design") != "sequential":
        return [k for k in ("anchor_rule", "first_budget") if k in side]
    rule, first, budget = side.get("anchor_rule"), side.get("first_budget"), side.get("budget")
    bad = []
    if not (type(first) is int and type(budget) is int and 1 <= first <= budget):
        bad.append("first_budget")
    if rule == "declared":
        return bad
    if not (isinstance(rule, dict) and set(rule) == {"alpha", "delta", "first_budget", "rule"}
            and isinstance(rule["rule"], str)):
        return bad + ["anchor_rule"]
    from .sequential import anchor_coverage
    try:
        ok = (rule["first_budget"] == first and type(rule["alpha"]) is float and type(rule["delta"]) is float
              and ALPHA_RANGE[0] <= rule["alpha"] <= ALPHA_RANGE[1] and DELTA_RANGE[0] <= rule["delta"] <= DELTA_RANGE[1]
              and abs(anchor_coverage(rule["first_budget"], rule["alpha"], rule["delta"]) - float(side["anchor"])) < 1e-12)
    except (TypeError, ValueError, KeyError):
        ok = False
    return bad if ok else bad + ["anchor_rule"]


def point(cert_path, x=None, y=None, row=None, col=None, mask=None):
    """Is a point inside the certified zone? (x, y) in the map's CRS, or (row, col) in pixels. `mask` is the zone as
    verify rebuilt it; without it the bundle's mask is read. Returns {"inside": bool or None, "window": [r, c],
    "where": "inside the certified zone", "outside the certified zone", "in no whole window, so not certified" or
    "outside the map"}."""
    item = _load(cert_path)
    p = item["properties"]
    base = os.path.dirname(os.path.abspath(cert_path))
    patch = p["zc:patch"]
    if type(patch) is not int or patch < 1:
        raise ValueError("the certificate's window size is not a positive integer")
    vals = [v for v in (x, y, row, col) if v is not None]
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("a point's coordinates must be finite numbers")
    h, w = p["zc:map"]["shape"]
    if x is not None:
        t = p["zc:map"]["transform"]
        if t is None:
            raise ValueError("this map has no grid; give the point as a pixel row and column")
        from rasterio.transform import Affine

        from .cli import affine_apply
        col, row = affine_apply(~Affine(*t), (float(x), float(y)))
    if mask is None:
        mask = _read_mask(os.path.join(base, item["assets"]["zone_mask"]["href"]), (h // patch, w // patch))[0]
    if not (0 <= float(row) < h and 0 <= float(col) < w):
        return {"inside": None, "window": None, "where": "outside the map"}
    r, c = int(float(row) // patch), int(float(col) // patch)
    if not (r < mask.shape[0] and c < mask.shape[1]):
        return {"inside": None, "window": [r, c], "where": "in no whole window, so not certified"}
    inside = bool(mask[r, c])
    return {"inside": inside, "window": [r, c], "where": "inside the certified zone" if inside else
            "outside the certified zone"}

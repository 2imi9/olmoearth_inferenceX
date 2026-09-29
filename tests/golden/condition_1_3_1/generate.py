"""Golden outputs of release 1.3.1, taken before the input-condition layer was built.

The layer must leave every existing output alone when no condition is given: the stdout of `assess`, `sample`,
`estimate` and `certify`, the sample CSV and its sidecar, the estimate and zone JSON, `assessment.json`, the review
set CSVs and the Python summaries of `assess_prediction` and `assess_classmap`. The only addition allowed is a
`scope` key in the JSON outputs. This script ran once at 725dffa, before any edit, and wrote the files beside it.

How the files were made:
- The fixtures are deterministic: the Dynamic World tile the package ships (9 classes, as a float32 .npy, and its
  expert labels), a 128 x 128 binary probability map with a no-data corner written as a GeoTIFF (as in
  tests/test_cli.py), and a 64 x 64 three-class logit map with a truth map (as in tests/test_cli_matrix.py).
- Every command runs through `oe_inferencex.cli.main` in a temporary directory. Its stdout is captured.
- Labels are filled by `fill_labels`: `wrong` is 1 where the tool's window class is not the majority of the
  reference, and `reference_class` is that majority (the tool's class where the reference has none).
- Paths are normalised: the temporary directory becomes <TMP>. A GeoTIFF output is stored as its array (.npy) and
  its grid (.json), since the bytes of a GeoTIFF are not the thing under test.

`produce(workdir)` returns {golden file name: bytes}, so a test can run the same steps at HEAD and compare. Run
`python tests/golden/condition_1_3_1/generate.py` to rewrite the files; do that only on purpose, since they are
the record of 1.3.1's behaviour.
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = "<TMP>"
GENERATED_AT = "725dffaef1085798b5c55b2dd0c64ecd522209fc"


# ----------------------------------------------------------------------------- fixtures
def _scene(seed=0, size=128):
    """A binary probability map with a water blob, a boundary, saturated probabilities and a no-data corner."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    water = ((yy - 60) ** 2 + (xx - 70) ** 2) < 35 ** 2
    p = np.where(water, 0.9, 0.1) + rng.normal(0, 0.15, (size, size))
    p = np.clip(p, 0, 1)
    p[:8, :8] = np.nan
    return p.astype(np.float32), water


def _three_class(seed=0, C=3, H=64, W=64):
    """Logits with smooth structure, their probabilities and a truth map that agrees with the argmax on about 85%."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W] / H
    fields = np.stack([np.sin(6 * xx + k) + np.cos(5 * yy - k) + 0.6 * rng.normal(size=(H, W)) for k in range(C)])
    logits = 2.5 * fields
    probs = np.exp(logits - logits.max(0))
    probs /= probs.sum(0)
    truth = np.where(rng.random((H, W)) < 0.85, probs.argmax(0), rng.integers(0, C, (H, W)))
    return logits.astype(np.float32), probs.astype(np.float32), truth


def _write_tif(path, arr, dtype=None):
    import rasterio
    from rasterio.transform import from_origin
    arr = np.asarray(arr)
    bands = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2], count=bands.shape[0],
                       dtype=dtype or bands.dtype, crs="EPSG:32633",
                       transform=from_origin(500000.0, 5000000.0, 10.0, 10.0)) as dst:
        dst.write(bands)


def make_fixtures(d):
    """Write the input maps into `d`; returns {name: path} and the arrays the Python summaries use."""
    from oe_inferencex.demo import SAMPLE
    os.makedirs(d, exist_ok=True)
    z = np.load(SAMPLE)
    dw, expert = z["probs"].astype(np.float32), z["expert"].astype(int)
    p, water = _scene()
    logits, probs3, truth3 = _three_class()
    paths = {"dw": os.path.join(d, "dw.npy"), "dw_expert": os.path.join(d, "dw_expert.npy"),
             "scene": os.path.join(d, "scene.tif"), "scene_ref": os.path.join(d, "scene_ref.tif"),
             "logits3": os.path.join(d, "logits3.npy"), "probs3": os.path.join(d, "probs3.npy"),
             "truth3": os.path.join(d, "truth3.npy")}
    np.save(paths["dw"], dw)
    np.save(paths["dw_expert"], expert)
    _write_tif(paths["scene"], p)
    _write_tif(paths["scene_ref"], water.astype("int16"), dtype="int16")
    np.save(paths["logits3"], logits)
    np.save(paths["probs3"], probs3)
    np.save(paths["truth3"], truth3)
    arrays = {"dw": dw, "expert": expert, "scene": p, "water": water, "logits3": logits, "probs3": probs3,
              "truth3": truth3}
    return paths, arrays


# ----------------------------------------------------------------------------- the reviewer
def fill_labels(csv_path, scores, reference, patch=4, is_logit=False):
    """Fill `wrong` and `reference_class` as a reviewer would: the reference is the majority of the reference
    pixels in the window (no majority: the tool's own class, so the window is right), and `wrong` says whether the
    tool's class (the `map_class` column) differs from it."""
    from oe_inferencex.assess import _pooled_argmax
    reference = np.asarray(reference).astype(int)
    n = int(max(int(reference.max()) + 1, scores.shape[0] if scores.ndim == 3 else 2))
    ref = _pooled_argmax(np.where(reference >= 0, reference, -1), n, patch, empty=-1, tie=-1)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        i, j = int(r["window_row"]), int(r["window_col"])
        mine = int(r["map_class"])
        truth = int(ref[i, j]) if ref[i, j] >= 0 else mine
        r["reference_class"] = str(truth)
        r["wrong"] = str(int(mine != truth))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- normalising
def _normalise_text(text, roots):
    for r in sorted(set(roots), key=len, reverse=True):
        text = text.replace(r, PLACEHOLDER)
    return text


def _roots(workdir):
    w = os.path.abspath(workdir)
    return [w, os.path.realpath(w)]


def _run(argv):
    from oe_inferencex.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    if code != 0:
        raise RuntimeError(f"{argv} exited {code}")
    return buf.getvalue()


def _tif_entries(path, name):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1)
        grid = {"crs": str(src.crs), "transform": list(src.transform)[:6], "dtype": str(src.dtypes[0]),
                "nodata": src.nodata, "shape": [src.height, src.width]}
    b = io.BytesIO()
    np.save(b, arr)
    return {f"{name}.npy": b.getvalue(), f"{name}.grid.json": (json.dumps(grid, indent=1) + "\n").encode()}


def _collect(path, name, roots, out):
    """One output file into `out` under the golden name, normalised."""
    if path.endswith((".json", ".csv", ".txt")):
        with open(path, encoding="utf-8") as f:
            out[name] = _normalise_text(f.read(), roots).encode()
    elif path.endswith(".tif"):
        out.update(_tif_entries(path, name[:-4] if name.endswith(".tif") else name))
    else:
        with open(path, "rb") as f:
            out[name] = f.read()


# ----------------------------------------------------------------------------- the steps
ASSESS = [
    ("assess_dw", ["{dw}"], {}),
    ("assess_scene", ["{scene}", "--reference", "{scene_ref}", "--budgets", "0.05", "0.10", "--order", "boundary_first"], {}),
    ("assess_logits3", ["{logits3}", "--logits", "--reference", "{truth3}"], {}),
]
# (name, scores, sample options, fixture to label from, per-class, certify runs)
SAMPLES = [
    ("sample_confidence", "dw", ["--budget", "200"], True, []),
    ("sample_proportional", "dw", ["--budget", "200", "--design", "proportional"], True, []),
    ("sample_random", "dw", ["--budget", "300", "--design", "random"], True,
     [("prefix", "0.12", []), ("bonferroni", "0.15", ["--rule", "bonferroni"]), ("whole", "0.3", []),
      ("none", "0.02", []), ("delta", "0.1", ["--delta", "0.2"])]),
    ("sample_tiles", "dw", ["--budget", "160", "--design", "tiles", "--tile", "8", "--per-tile", "16"], False, []),
    ("sample_scene_confidence", "scene", ["--budget", "100", "--seed", "3"], True, []),
    ("sample_scene_random", "scene", ["--budget", "120", "--design", "random", "--seed", "5"], True,
     [("prefix", "0.2", [])]),
    ("sample_logits3_random", "logits3", ["--budget", "80", "--design", "random", "--logits", "--seed", "2"], True,
     [("prefix", "0.3", [])]),
]


def produce(workdir):
    """Run every step in `workdir`. Returns ({golden name: bytes}, manifest)."""
    from oe_inferencex.assess import assess_classmap, assess_prediction, summary
    from oe_inferencex.cli import read_raster
    workdir = os.path.abspath(workdir)
    roots = _roots(workdir)
    paths, arr = make_fixtures(os.path.join(workdir, "in"))
    out, manifest = {}, {"generated_at": GENERATED_AT, "placeholder": PLACEHOLDER, "steps": []}

    def fmt(a):
        return [x.format(**paths) for x in a]

    for name, args, _ in ASSESS:
        d = os.path.join(workdir, name)
        argv = ["assess", *fmt(args), "--out", d]
        out[f"{name}.stdout.txt"] = _normalise_text(_run(argv), roots).encode()
        files = sorted(os.listdir(d))
        for fn in files:
            _collect(os.path.join(d, fn), f"{name}__{fn}", roots, out)
        manifest["steps"].append({"name": name, "argv": [_normalise_text(x, roots) for x in argv], "files": files})

    fixture_scores = {"dw": (arr["dw"], arr["expert"], False), "scene": (None, None, False),
                      "logits3": (arr["logits3"], arr["truth3"], True)}
    for name, fixture, opts, per_class, certs in SAMPLES:
        csv_path = os.path.join(workdir, f"{name}.csv")
        argv = ["sample", paths[fixture], *opts, "--out", csv_path]
        out[f"{name}.stdout.txt"] = _normalise_text(_run(argv), roots).encode()
        _collect(csv_path, f"{name}.csv", roots, out)
        _collect(csv_path[:-4] + ".json", f"{name}.json", roots, out)
        step = {"name": name, "argv": [_normalise_text(x, roots) for x in argv], "labelled_from": fixture}
        scores, reference, is_logit = fixture_scores[fixture]
        if fixture == "scene":
            s, valid, _ = read_raster(paths["scene"])
            r, _, _ = read_raster(paths["scene_ref"])
            scores, reference = np.where(valid, s, np.nan), np.asarray(r).astype(int)
        fill_labels(csv_path, scores, reference, is_logit=is_logit)
        _collect(csv_path, f"{name}.labelled.csv", roots, out)
        est_json = os.path.join(workdir, f"{name}_estimate.json")
        argv_e = ["estimate", csv_path, "--out", est_json]
        out[f"{name}.estimate.stdout.txt"] = _normalise_text(_run(argv_e), roots).encode()
        _collect(est_json, f"{name}.estimate.json", roots, out)
        step["estimate"] = [_normalise_text(x, roots) for x in argv_e]
        if per_class:
            pc_json = os.path.join(workdir, f"{name}_perclass.json")
            argv_p = ["estimate", csv_path, "--per-class", "--out", pc_json]
            out[f"{name}.perclass.stdout.txt"] = _normalise_text(_run(argv_p), roots).encode()
            _collect(pc_json, f"{name}.perclass.json", roots, out)
            step["per_class"] = [_normalise_text(x, roots) for x in argv_p]
        step["certify"] = []
        for tag, alpha, extra in certs:
            zj = os.path.join(workdir, f"{name}_zone_{tag}.json")
            argv_c = ["certify", csv_path, "--alpha", alpha, *extra, "--out", zj]
            out[f"{name}.certify_{tag}.stdout.txt"] = _normalise_text(_run(argv_c), roots).encode()
            _collect(zj, f"{name}.certify_{tag}.json", roots, out)
            mask = zj[:-5] + ".npy"
            if os.path.exists(mask):
                _collect(mask, f"{name}.certify_{tag}.mask.npy", roots, out)
            step["certify"].append([_normalise_text(x, roots) for x in argv_c])
        manifest["steps"].append(step)

    # the Python summaries, both assess_* functions, with and without a reference, under both orders
    p, water = arr["scene"], arr["water"]
    hard3 = arr["probs3"].argmax(0)
    band = arr["probs3"].max(0)
    nd3 = np.zeros(hard3.shape, bool)
    nd3[:6, :10] = True
    api = {
        "api_prediction_dw": lambda: assess_prediction(arr["dw"], is_logit=False),
        "api_prediction_scene": lambda: assess_prediction(p, is_logit=False, nodata_mask=np.isnan(p),
                                                          reference=water.astype(int), budgets=(0.05, 0.10),
                                                          order="boundary_first"),
        "api_prediction_logits3_margin": lambda: assess_prediction(arr["logits3"], is_logit=True),
        "api_prediction_logits3_top1": lambda: assess_prediction(arr["logits3"], is_logit=True, form="top1",
                                                                 reference=arr["truth3"]),
        "api_classmap": lambda: assess_classmap(hard3, band, 3, nodata_mask=nd3, reference=arr["truth3"]),
        "api_classmap_boundary_first": lambda: assess_classmap(hard3, np.round(band, 2), 3, patch=8,
                                                               order="boundary_first", budgets=(0.02, 0.2)),
    }
    for name, fn in api.items():
        out[f"{name}.json"] = (json.dumps(summary(fn()), indent=1) + "\n").encode()
    manifest["api"] = sorted(api)
    return out, manifest


def main(dest=HERE):
    with tempfile.TemporaryDirectory() as tmp:
        files, manifest = produce(tmp)
    for name, data in files.items():
        with open(os.path.join(dest, name), "wb") as f:
            f.write(data)
    manifest["files"] = sorted(files)
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return len(files)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    print(f"wrote {main()} golden files to {HERE}")

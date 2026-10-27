"""The commands from-embeddings and from-olmoearth: an embeddings store or an OlmoEarth run read into scores."""
import os

import numpy as np

from oe_inferencex.common.output import _check_out_dir


# ----------------------------------------------------------------------------- from-embeddings
def cmd_from_embeddings(args):
    """An area of precomputed embeddings and labelled points in; logistic regression's class probabilities out, as a
    GeoTIFF assess reads."""
    from oe_inferencex.readers import embeddings as emb
    if not args.out.lower().endswith((".tif", ".tiff")):
        raise SystemExit("from-embeddings: --out is the GeoTIFF to write (.tif)")
    args.store = args.store or emb.STORE
    points = emb.read_points(args.points)
    root = emb.open_store(args.store, args.endpoint or emb.ENDPOINT)
    v, crs, transform, zone = emb.read_area(root, args.bbox, args.year, args.dims, args.max_pixels or emb.MAX_PIXELS)
    probs, names, counts, notes = emb.fit_and_predict(v, crs, transform, points)
    meta = {"store": args.store, "zone": zone, "crs": crs, "year": int(args.year), "dims": int(args.dims),
            "bbox": [float(x) for x in args.bbox], "classifier": "sklearn LogisticRegression(max_iter=2000)",
            "points_per_class": counts, "notes": notes}
    side = emb.write_scores(args.out, probs, crs, transform, names, meta)
    valid = float(np.isfinite(probs[0]).mean())
    print(f"{len(names)} classes from {sum(counts.values())} points; wrote {args.out} ({probs.shape[1]} x "
          f"{probs.shape[2]} pixels, {crs}, {100 * valid:.1f}% with an embedding) and {side}"
          + "".join(f"\nnote: {n}" for n in notes)
          + "\nnote: the points the classifier was fitted on cannot say how accurate the map is; draw a fresh sample "
            "with sample and label that"
          + f"\nnext: oe-inferencex assess {args.out} --out audit")
    return 0


def _skipped_text(skipped):
    """The windows a reader left out, counted by reason; the JSON lists each."""
    if not skipped:
        return ""
    by = {}
    for s in skipped:
        by[s["reason"]] = by.get(s["reason"], 0) + 1
    return f"; {len(skipped)} skipped: " + "; ".join(f"{n} {why}" for why, n in by.items())


def cmd_from_olmoearth(args):
    """An OlmoEarth run's rslearn dataset in, the rasters `assess` reads out (oe_inferencex.olmoearth): the output
    layer's probabilities pasted onto one grid per CRS, and with --conditions the input-condition layer on the same
    grid; then the assess command that reads them."""
    from oe_inferencex.readers import olmoearth as oe
    _check_out_dir(args.out, "from-olmoearth")
    if args.inputs and not args.conditions:
        raise SystemExit("from-olmoearth: --inputs names the input layers of the condition layer; add --conditions")
    try:
        kind = oe.output_kind(args.ds, args.layer, args.group, args.window)
        if kind == "vector":
            if args.conditions:
                raise SystemExit(f"from-olmoearth: layers/{args.layer} is a per-window classification (data.geojson); "
                                 "the condition layer is pasted onto a raster's grid, and this output has none")
            s = oe.read_window_probs(args.ds, args.out, layer=args.layer, prob_property=args.prob_property or "probs",
                                     group=args.group, windows=args.window, class_property=args.class_property)
        else:
            if args.prob_property or args.class_property:
                flag = "--prob-property" if args.prob_property else "--class-property"
                raise SystemExit(f"from-olmoearth: {flag} reads a per-window classification (data.geojson); "
                                 f"layers/{args.layer} holds GeoTIFFs")
            s = oe.read_output(args.ds, args.out, layer=args.layer, group=args.group, windows=args.window)
            c = oe.condition_from_inputs(args.ds, args.out, inputs=args.inputs, layer=args.layer,
                                         group=args.group, windows=args.window) if args.conditions else None
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"from-olmoearth: {exc}") from None
    what = (f"probabilities over {s['bands']} classes, {s['features']} predictions" if kind == "vector" else
            f"probabilities over {s['bands']} classes" if s["bands"] > 1 else "one band, the probability of one class")
    lines = [f"read {s['windows_read']} windows of {os.path.join(args.ds, 'windows')} (layer {args.layer}): {what}"
             + _skipped_text(s["windows_skipped"])]
    notes = list(s["notes"])
    if kind == "vector":
        lines += [f"wrote {s['scores']}, ({s['bands']}, 1, {s['features']}), and {s['index']}, which names each column",
                  f"next: {s['assess']}"]
        written = [oe.SCORES_JSON]
    else:
        conds = {g["label"]: g for g in c["grids"]} if c else {}
        for g in s["grids"]:
            lines.append(f"wrote {g['scores']}: {g['shape'][0]} x {g['shape'][1]} px in {g['crs']}, "
                         f"{len(g['windows'])} windows, {g['covered_pixels']} pixels predicted")
            if g["label"] in conds:
                k = len(conds[g["label"]]["codes"])
                lines.append(f"wrote {conds[g['label']]['condition']}: {k} input condition{'s' if k != 1 else ''} "
                             "(the codes and their rule are in " + oe.CONDITIONS_JSON + ")")
        lines += [f"next: {conds[g['label']]['assess'] if g['label'] in conds else g['assess']}" for g in s["grids"]]
        written = [oe.SCORES_JSON] + ([oe.CONDITIONS_JSON] if c else [])
        notes += c["notes"] if c else []
    print("\n".join(lines + [f"note: {n}" for n in notes] + [f"wrote {os.path.join(args.out, written[0])}"
                                                               + "".join(f", {w}" for w in written[1:])]))
    return 0


def add_from_embeddings_parser(sub):
    fe = sub.add_parser("from-embeddings", help="read an area of precomputed embeddings (OlmoEarth v1.3), fit logistic "
                                                "regression on labelled points, and write its class probabilities")
    fe.add_argument("--points", required=True, help="CSV with lon, lat and class columns")
    fe.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                    help="the area in degrees, inside one UTM zone")
    fe.add_argument("--year", required=True, type=int, help="the calendar year of the embeddings")
    fe.add_argument("--out", required=True, help="the GeoTIFF of class probabilities to write")
    fe.add_argument("--dims", type=int, default=128, choices=(64, 128), help="64 reads half the data (default 128)")
    fe.add_argument("--store", default=None, help="the Zarr store, an s3:// URL or a local directory (default: Ai2's "
                                                  "OlmoEarth v1.3 embeddings on Source Cooperative)")
    fe.add_argument("--endpoint", default=None, help="the S3 endpoint of an s3:// store (default: Source Cooperative)")
    fe.add_argument("--max-pixels", type=int, default=None,
                    help="refuse a box larger than this many pixels (default 1048576, a 10 km square at 10 m)")
    fe.set_defaults(func=cmd_from_embeddings)


def add_from_olmoearth_parser(sub):
    fo = sub.add_parser("from-olmoearth", help="read an OlmoEarth run (the rslearn dataset olmoearth_run writes) into "
                                               "the rasters assess reads, and print the assess command")
    fo.add_argument("ds", help="the rslearn dataset directory, which holds windows/<group>/<name>/")
    fo.add_argument("--out", required=True, help="output directory")
    fo.add_argument("--layer", default="output", help="the output layer (default output)")
    fo.add_argument("--group", nargs="+", default=None, help="read only these window groups (default: every one)")
    fo.add_argument("--window", nargs="+", default=None, metavar="PATTERN",
                    help="read only the windows whose id group/name matches one of these shell patterns (quote them), "
                         "to read a large area in parts")
    fo.add_argument("--conditions", action="store_true",
                    help="also write the input-condition layer on the scores' grid: per pixel, how many of the run's "
                         "timesteps a scene covered and, with an SCL band, how many were cloudy")
    fo.add_argument("--inputs", nargs="+", default=None, metavar="LAYER",
                    help="with --conditions: the input layers to read (default: the layers config.json gives a "
                         "data_source)")
    fo.add_argument("--prob-property", default=None,
                    help="a per-window classification: the feature property holding the probabilities (default probs)")
    fo.add_argument("--class-property", default=None,
                    help="a per-window classification: the feature property holding the class the task wrote, checked "
                         "against the argmax (default: the one other property, if there is one)")
    fo.set_defaults(func=cmd_from_olmoearth)

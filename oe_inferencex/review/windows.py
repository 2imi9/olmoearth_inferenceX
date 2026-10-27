"""Maps read into per-window readings for the labelled routes: one map (_map_windows) or two (_two_map_windows)."""
from types import SimpleNamespace

import numpy as np

from oe_inferencex.review.assess import _pooled_argmax, assess_classmap, assess_prediction
from oe_inferencex.common.labels import _rounding_tolerance
from oe_inferencex.common.maps import _check_scores, decisions
from oe_inferencex.common.output import _readable_path
from oe_inferencex.common.product import PRODUCT_SIGNAL, _read_product
from oe_inferencex.common.raster import _mask_hint, _same_grid, pool_valid, read_raster


def _two_map_windows(path_a, path_b, nodata, threshold, patch, labels_path=None):
    """Two maps (and an optional label raster) read onto one window grid, as compare grades them: each map's class per
    window from the pixels both maps predicted, ties broken by confidence where both maps carry one and left out where
    either is a class map, class codes remapped jointly when they are sparse. Shared by compare and by sample --other,
    so the windows sampled where two maps differ are the windows compare counts as differing."""
    notes = []
    ha, va, geo, na, ca = decisions(path_a, nodata, threshold, notes)
    hb, vb, geo_b, nb, cb = decisions(path_b, nodata, threshold, notes)
    if ha.shape != hb.shape:
        raise SystemExit(f"the two maps differ in shape: {ha.shape} vs {hb.shape}; compare needs identical grids")
    _same_grid(geo, geo_b, path_b)
    # The grid is the first map's that carries one. With map a a .npy, the labels and groups were checked against
    # no grid at all, so labels 50 km off b's grid graded the maps with exit 0, while the same files with a GeoTIFF
    # first were refused (2026-10-06).
    geo = geo if geo is not None else geo_b
    if not 1 <= patch <= min(ha.shape):
        raise SystemExit(f"--patch {patch} must be at least 1 and no larger than the map, {ha.shape[0]} x {ha.shape[1]} px")
    both = va & vb
    lab_i = lv = None
    if labels_path:
        lab, lv, geo_l = read_raster(labels_path, None)
        if lab.shape != ha.shape:
            raise SystemExit(f"{labels_path} has shape {lab.shape}; the maps are {ha.shape}")
        _same_grid(geo, geo_l, labels_path)
        lab_i = np.rint(lab).astype(int)
        lv = lv & (lab_i >= 0)                              # a negative code is unlabelled whatever the nodata tag says
        if not lv.any():
            raise SystemExit(f"{labels_path}: no valid label pixels")
    # One code space for both maps and the labels, decided before anything is pooled. Pooling counts one window grid
    # per class id up to the largest, so a stray code of 60000 took 26 s and 3.3 GB on a 512 x 512 map (review of
    # 2026-09-23); sparse or negative codes are remapped to 0..K-1 jointly and mapped back in the CSV.
    map_codes = np.unique(np.concatenate([ha[both].ravel(), hb[both].ravel()])) if both.any() else np.zeros(0, int)
    lab_codes = np.unique(lab_i[lv]) if lab_i is not None else np.zeros(0, int)
    all_codes = np.unique(np.concatenate([map_codes, lab_codes])).astype(int)
    if all_codes.size and (all_codes.min() < 0 or int(all_codes.max()) + 1 > 4 * max(all_codes.size, 16)):
        code_of = all_codes
        ha, hb = np.searchsorted(all_codes, ha), np.searchsorted(all_codes, hb)     # pixels outside `both` never vote
        if lab_i is not None:
            lab_i = np.where(lv, np.searchsorted(all_codes, lab_i), -1)
        n_classes = n_lab = max(int(all_codes.size), 2)
    else:
        code_of = None
        # Sized by the codes that vote, the pixels both maps predicted. max(na, nb) counted each map's own valid
        # pixels, so an undeclared fill of 65535 where the other map has no data allocated 65536 window grids
        # (4.3 GB and 13 s on 256 x 256) for codes that never vote, the cost the remap above was meant to end
        # (2026-10-06). Codes outside `both` never vote, so the windows' classes are unchanged.
        n_classes = max(int(map_codes.max()) + 1, 2) if map_codes.size else 2
        # The label raster is pooled over ITS OWN class range, never the maps'. _pooled_argmax counts votes only over
        # range(n_classes), so pooling a 0-5 label raster with the maps' n_classes=3 silently dropped every pixel of
        # class 3 and above and let the window label fall to a surviving low index: measured at 44.9% of window labels
        # wrong, with exit 0 and no warning, on the very output that says which inference to believe.
        n_lab = max(int(lab_codes.max()) + 1, 2) if lab_codes.size else 2
    # No-data pixels do not vote (they used to argmax to class 0 and could decide a window). Both maps pool over the
    # pixels BOTH predicted: a window half under B's no-data used to be decided from 16 of A's pixels and 8 of B's,
    # and two identical maps then "differed" there. A tied window goes to the more confident voters when both maps
    # carry a confidence, as in assess; when either is a hard class map, which has none, a tied window is left out
    # of the comparison on both sides, because breaking it one way on one side and another way on the other made
    # a class map and its own probability version differ on 1.8% of windows (review of 2026-09-23).
    weighted = ca is not None and cb is not None
    a_w = _pooled_argmax(np.where(both, ha, -1), n_classes, patch, empty=-1, weights=ca if weighted else None,
                         tie=None if weighted else -2)
    b_w = _pooled_argmax(np.where(both, hb, -1), n_classes, patch, empty=-1, weights=cb if weighted else None,
                         tie=None if weighted else -2)
    n_tie_windows = int((pool_valid(both, patch) & ((a_w == -2) | (b_w == -2))).sum())
    if n_tie_windows:
        notes.append(f"{n_tie_windows} windows are split evenly between two classes on a map with no confidence to break "
                     "the tie; they are left out of the comparison")
    ok = pool_valid(va & vb, patch) & (a_w >= 0) & (b_w >= 0)
    if not ok.any():
        # assess refuses an empty map; compare used to exit 0 with "0 of 0 windows differ"
        raise SystemExit("compare: no window was predicted by both maps (all no-data, or every window tied), so there is "
                         "nothing to compare")
    labels = groups = None
    ok_graded = None
    if lab_i is not None:
        # invalid label pixels must not vote; -1 is the non-voting code _assess uses, where 0 is a real class
        lab_w = _pooled_argmax(np.where(lv, lab_i, -1), n_lab, patch, empty=-1, tie=-1)   # no majority, no label
        # The grading runs on the labelled windows; the label-free numbers stay on every window both maps predicted.
        # Until 2026-09-22 the label mask was ANDed into the whole comparison, so adding --labels changed the numbers
        # the docs call label-free and dropped unlabelled differing windows from the CSV and the raster.
        labelled = pool_valid(lv, patch)
        n_tied = int((ok & labelled & (lab_w < 0)).sum())
        ok_graded = ok & labelled & (lab_w >= 0)            # an evenly split label window has no label to grade
        labels = lab_w
        unpredicted = sorted(int(c) for c in set(lab_codes.tolist()) - set(map_codes.tolist()))
        if unpredicted:
            # named by class, not by the largest id: "the labels carry 4 classes" was said of a raster holding class 3
            notes.append(f"the labels hold class(es) {unpredicted[:10]}{' ...' if len(unpredicted) > 10 else ''} that neither "
                         "map predicts; windows labelled with them are counted wrong for both sides")
        if n_tied:
            notes.append(f"{n_tied} windows have an evenly split label and no majority; they are left out of the grading")
    return SimpleNamespace(notes=notes, ha=ha, hb=hb, va=va, vb=vb, geo=geo, code_of=code_of, a_w=a_w, b_w=b_w, ok=ok,
                           labels=labels, ok_graded=ok_graded, lab_i=lab_i, n_classes=(na, nb))


def _map_windows(side, scores_override, nodata, command, rows=None, idx=None, confidence_override=None):
    """The map's per-window confidence, class and validity, recomputed from the scores the sample was drawn on, and
    checked to be that map: its population must be the sample's, and its confidence at the sampled windows the
    CSV's (review of 2026-09-23: another map on the same grid was certified from this map's labels, and a sample
    drawn with --nodata was recomputed without it, with no message). Returns (confidence, class, valid, n_classes)."""
    if confidence_override and not side.get("confidence"):
        raise SystemExit(f"{command}: --confidence names a product's band, and this sample was drawn on scores, not "
                         "with --confidence")
    path = scores_override or side.get("scores")
    if not path or not _readable_path(path):
        raise SystemExit(f"{command}: the map's scores are needed again ({path or 'no path in the sidecar'} not found); "
                         "pass --scores with the raster `sample` was run on")
    recorded = None if side.get("nodata") is None else float(side["nodata"])     # "nan" is how NaN is written
    if nodata is None:
        nodata = recorded                                     # the value the sample was drawn with
    elif recorded is not None and recorded != float(nodata) and not (np.isnan(recorded) and np.isnan(float(nodata))):
        # NaN is not equal to itself, so a sample drawn with --nodata nan was refused with "--nodata nan differs from
        # the nan the sample was drawn with" (2026-10-06)
        raise SystemExit(f"{command}: --nodata {nodata:g} differs from the {recorded:g} the sample was drawn with")
    if side.get("confidence"):                                  # a class map with its confidence band
        conf_path = confidence_override or side["confidence"]
        if not _readable_path(conf_path):
            raise SystemExit(f"{command}: the map's confidence band is needed again ({conf_path} not found); pass "
                             "--confidence with the band `sample` was run on")
        hard, conf_px, valid, _, n_classes, _, _ = _read_product(path, conf_path, nodata, side.get("confidence_range"),
                                                                 command)
        try:
            out = assess_classmap(hard, conf_px, n_classes, patch=int(side.get("patch", 4)), nodata_mask=~valid,
                                  signal=PRODUCT_SIGNAL)
        except ValueError as exc:
            raise SystemExit(f"{command}: {exc}") from None
    else:
        scores, valid, _ = read_raster(path, nodata)
        _check_scores(scores, valid, bool(side.get("logits", False)), path)
        try:                                                  # a named refusal, as on the product branch (2026-10-06)
            out = assess_prediction(scores, is_logit=bool(side.get("logits", False)), patch=int(side.get("patch", 4)),
                                    nodata_mask=~valid)
        except ValueError as exc:
            raise SystemExit(f"{command}: {exc}") from None
        n_classes = int(scores.shape[0]) if scores.ndim == 3 else 2
    arr = out["arrays"]
    if list(arr["confidence"].shape) != list(side.get("grid", arr["confidence"].shape)):
        raise SystemExit(f"{command}: {path} pools to a {arr['confidence'].shape} window grid, but the sample was drawn on "
                         f"{side.get('grid')}; pass the raster the sample was drawn on")
    conf, valid_w = arr["confidence"], arr["valid"]
    n_pop = int((valid_w & np.isfinite(conf)).sum())
    # 1.17.2 reads a GeoTIFF's mask as no-data: a sample drawn before on such a raster counted the masked pixels, and
    # whichever check catches the change says why (the reviews of 1.17.2)
    mask_hint = _mask_hint(path, (confidence_override or side.get("confidence")) if side.get("confidence") else None)
    if side.get("n_population") is not None and n_pop != int(side["n_population"]):
        # 1.7.0 changed the no-data rule of a multi-band raster (a pixel is no-data only when every band holds the
        # value), so a sample drawn before on such a raster can name a population no option restores (the review of
        # 2026-10-06): say so rather than send the user after a --nodata value that does not exist
        hint = (" If the sample was drawn before 1.7.0 on a multi-band raster with a no-data value, the population "
                "changed with the fix to the no-data rule (pixels where only some bands hold the value are data now); "
                "draw the sample again." if side.get("nodata") is not None or nodata is not None else "")
        hint += mask_hint
        raise SystemExit(f"{command}: {path} has {n_pop} valid windows, but the sample was drawn from {side['n_population']}; "
                         "pass the raster, and the --nodata value, the sample was drawn with." + hint)
    if rows is not None and idx is not None and rows and "confidence" not in rows[0] and "map_class" not in rows[0]:
        # with both columns deleted nothing was checked, and another map on the same grid was certified from this
        # map's labels, exit 0 (the bug hunt of 2026-10-25); to label blind, hide them
        raise SystemExit(f"{command}: the CSV has neither the `confidence` nor the `map_class` column `sample` wrote, so "
                         "it cannot be checked to be this map's sample; put them back (to label blind, hide them, do "
                         "not delete them)")
    if rows is not None and idx is not None and "confidence" in rows[0]:
        try:
            written = np.array([float(r["confidence"]) for r in rows])
            # the tolerance follows the digits the CSV holds: a spreadsheet that saved 0.412 for 0.41234 is still
            # the same map (the verification of the second review found three digits refused at a fixed 1e-4)
            tol = np.array([_rounding_tolerance(r["confidence"]) for r in rows])
        except ValueError:
            written = None
        if written is not None:
            off = np.abs(conf.ravel()[idx] - written)
            if not (off <= tol * (1 + 1e-9) + 1e-12).all():      # a value written exactly half a unit off
                k = int(np.argmax(off))
                raise SystemExit(f"{command}: the map's confidence at the sampled windows is not the one the CSV records "
                                 f"(row {k + 2}: {written[k]:.6g} in the CSV, {conf.ravel()[idx][k]:.6g} now); this is not the "
                                 "map the sample was drawn on." + mask_hint)
    if rows is not None and idx is not None and "map_class" in rows[0]:
        # The class too: a product's class map and band are two files, and another year's class map on the same grid
        # with the same band passed the confidence check, so estimate --per-class built its table from that map, exit
        # 0, with only a note blaming the labels (2026-10-06). A column that cannot be read skips the check, as above.
        try:
            recorded = np.array([int(float(r["map_class"])) for r in rows])
        except (ValueError, TypeError, OverflowError):
            recorded = None
        if recorded is not None:
            now = arr["pooled_argmax"].ravel()[idx]
            bad = np.flatnonzero(now != recorded)
            if bad.size:
                k = int(bad[0])
                raise SystemExit(f"{command}: the map's class at the sampled windows is not the one the CSV records on "
                                 f"{bad.size} row(s) (row {k + 2}: {recorded[k]} in the CSV, {int(now[k])} now); this is "
                                 "not the map the sample was drawn on." + mask_hint)
    return conf, arr["pooled_argmax"], valid_w, n_classes

"""A labelled sample read back with its sidecar, and the --also-accept rows of the labels' second reading."""
import csv
import json
import os

import numpy as np

from oe_inferencex.common import readings as est


def _sample_map_inputs(sample_path, *overrides):
    """The rasters a labelled sample was drawn from, as its sidecar names them, with any override: what estimate and
    certify read again and must never write over (the bug hunt of 25 October 2026: `certify s.csv --out map.json` with
    map.npy deleted the map, and `--out map.tif` wrote the zone over it)."""
    side_path = sample_path[:-4] + ".json" if sample_path.endswith(".csv") else sample_path + ".json"
    try:
        with open(side_path) as f:
            side = json.load(f)
    except (OSError, ValueError):
        return [o for o in overrides if o]
    cond = side.get("condition") if isinstance(side.get("condition"), dict) else {}
    named = [side.get(k) for k in ("scores", "classes", "confidence", "map_a", "map_b")] + [cond.get("source")]
    prof = side.get("profile") if isinstance(side.get("profile"), dict) else {}
    named += [f.get("source") for f in prof.get("fields") or [] if isinstance(f, dict)]     # a profile sample's layer
    return [p for p in named + list(overrides) if isinstance(p, str) and p]


def _population_note(side):
    """What a product sample's population leaves out, said with every number drawn from it."""
    rng, pop = side.get("confidence_range"), side.get("product_population") or {}
    n_out, n_all = int(pop.get("outside_range") or 0), int(pop.get("classified_pixels") or 0)
    if not rng or not n_out or not n_all:
        return None
    return (f"The population is the pixels whose confidence lies in [{rng[0]:g}, {rng[1]:g}]: {n_out} of {n_all} "
            f"classified pixels ({100 * n_out / n_all:.1f}%) were left out, and nothing here describes them. Left-out "
            "pixels can be wrong more often (exp93: LCMAP's provenance-coded plots, 29.1% against 17.7%).")


def _sidecar_design(path):
    """The design a sample CSV's sidecar records, or None when there is no readable sidecar."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    try:
        with open(side_path) as f:
            return json.load(f).get("design")
    except (OSError, ValueError):
        return None


def _sidecar_by(path):
    """What a sample's strata are, as its sidecar's condition block says (`by`: "map class" for the class design,
    "profile" for the profile design), or None: no readable sidecar, no condition block, or strata by input condition."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    try:
        with open(side_path) as f:
            side = json.load(f)
    except (OSError, ValueError):
        return None
    cond = side.get("condition") if isinstance(side, dict) else None
    return cond.get("by") if isinstance(cond, dict) else None


def _reference_column(rows, codes="the maps' class codes", negative=True):
    """The `reference_class` column of a sample whose reviewer records the class seen (sample --other, sample --design
    audit): (the class per row, -1 where it is ?; True where it is ?). Every row holds a whole number or ?, none blank;
    without `negative`, a whole number of at least 0."""
    # The audit passes negative=False: its classes are at least 0 (sample_class_audit refuses a negative map_class), so
    # a -1 written for "no data" or "cannot tell" was read as a judged class and counted as an error, which can turn a
    # class's verdict to `above` (the review of 13 October 2026); estimate --per-class refuses it too. A sample --other
    # keeps it, since compare maps a pair's negative codes back (tests/test_which_map.py).
    unjudged, ref, bad = [], [], []
    for i, r in enumerate(rows):
        v = str(r["reference_class"]).strip()
        if v == "?":
            unjudged.append(True)
            ref.append(-1)
            continue
        try:
            k = int(float(v))
            if float(v) != k or (k < 0 and not negative):
                raise ValueError
        except (ValueError, OverflowError):
            bad.append((i + 2, v))
            k = 0
        unjudged.append(False)
        ref.append(k)
    if bad:
        whole = "a whole number" if negative else "a whole number of at least 0"
        raise SystemExit(f"`reference_class` must be the class seen ({whole}, in {codes}) or ? in every row"
                         + ("" if negative else ", ? where a window cannot be judged")
                         + f"; {len(bad)} row(s) are not, first at row {bad[0][0]}: {bad[0][1]!r}")
    return np.array(ref, dtype=np.int64), np.array(unjudged, dtype=bool)


def _also_accept(path, command="estimate"):
    """The --also-accept file: JSON {predicted class: [reference classes also accepted]}, read and checked, each
    error refused by name."""
    try:
        with open(path, encoding="utf-8") as f:
            pairs = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{command}: --also-accept {path}: no such file") from None
    except OSError as exc:
        raise SystemExit(f"{command}: --also-accept {path}: {exc.strerror or exc}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"{command}: --also-accept {path} is not JSON ({exc})") from None
    try:
        return est._also_accept_pairs(pairs)
    except ValueError as exc:
        raise SystemExit(f"{command}: --also-accept {path}: {exc}") from None


def _also_accept_rows(command, path, rows, wrong, unjudged, map_class):
    """The labels of a sample of one map read a second time with the --also-accept file (estimate and certify):
    (the second reading's `wrong`, the result's block: source, pairs, errors_accepted, n_accepted; the listed classes
    no labelled window is mapped as). `map_class` is the map's class at each labelled window, recomputed from the map
    (_map_windows). Every refusal runs before anything is computed or written."""
    if "reference_class" not in rows[0]:
        raise SystemExit(f"{command}: --also-accept reads the class seen in each window: add a reference_class column "
                         "beside wrong (a whole number of at least 0, in the map's class codes, or ? where wrong is ?)")
    ref, ref_unj = _reference_column(rows, "the map's class codes", negative=False)
    unjudged = np.asarray(unjudged, bool)
    one = np.flatnonzero(ref_unj != unjudged)
    if one.size:
        k = int(one[0])
        has, lacks = ("wrong", "reference_class") if unjudged[k] else ("reference_class", "wrong")
        raise SystemExit(f"{command}: row {k + 2} is ? in {has} and not in {lacks}: a window that cannot be judged is ? "
                         "in both columns, and one that can be judged is ? in neither")
    # --per-class only notes a row where the two columns disagree, and its error rate reads `wrong`; here the two
    # readings would then read different labels, so it is refused
    mc = np.asarray(map_class).ravel()
    off = np.flatnonzero(~unjudged & ((ref != mc).astype(int) != np.asarray(wrong).astype(int)))
    if off.size:
        k = int(off[0])
        raise SystemExit(f"{command}: on {off.size} row(s) `wrong` disagrees with whether `reference_class` differs from "
                         f"the map's class, first at row {k + 2} (wrong {int(wrong[k])}, reference_class {int(ref[k])}, "
                         f"map_class {int(mc[k])}); --also-accept reads the same labels twice, so wrong must be 1 "
                         "exactly where reference_class is not the map's class")
    accept = _also_accept(path, command)
    try:
        rd = est.also_accept_reading(wrong, mc, ref, accept, unjudged=unjudged)
    except ValueError as exc:
        raise SystemExit(f"{command}: {exc}") from None
    return rd["wrong"], {"source": os.path.abspath(path), "pairs": rd["pairs"],
                         "errors_accepted": rd["errors_accepted"], "n_accepted": rd["n_accepted"]}, rd["unused"]


def _also_words(block):
    """What a second reading also accepts, as printed: "3 where the map says 2"."""
    said = []
    for c, codes in block["pairs"].items():
        more = [str(a) for a in codes if str(a) != c]
        if more:
            said.append((", ".join(more[:-1]) + " or " if len(more) > 1 else "") + f"{more[-1]} where the map says {c}")
    return "; ".join(said) if said else "no class beyond the map's own"


def _unused_note(unused):
    return (f"note: --also-accept lists class(es) {', '.join(unused)}, which no labelled window is mapped as; they are "
            "not read")


def _check_sidecar(side, side_path, keys=("indices",)):
    """A sidecar that holds no design is refused by name: one overwritten by a result JSON was a KeyError traceback on
    every later estimate and certify (2026-10-06)."""
    missing = [k for k in keys if k not in side] if isinstance(side, dict) else list(keys)
    if missing:
        raise SystemExit(f"{side_path} holds no sample design (no {', '.join(missing)}): it is not the sidecar `sample` "
                         "wrote, or a result was written over it; draw the sample again")


def _csv_rows(path, command):
    """A sample CSV's rows (certificate.csv_rows), its refusals as the command's. The --other, audit and --extend routes
    read with csv.DictReader, which keeps the last copy of a column named twice where a spreadsheet shows the first:
    a second reference_class column turned an undetermined comparison into "b is more accurate", exit 0 (the bug hunt
    of 2026-10-25)."""
    from oe_inferencex.certify.certificate import csv_rows
    try:                                     # a spreadsheet's BOM is not a column name; a column named twice is refused
        return csv_rows(path)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"{command}: {exc}") from None


def _labelled_sample(path, command):
    """A filled-in sample CSV and its sidecar design, checked: the rows are the design's, every `wrong` is 0, 1 or `?`
    (a window the reviewer could not judge). Returns (sidecar, rows, indices, sample dict for the estimators, wrong,
    unjudged), with `wrong` 0 where the row is `?`."""
    side_path = path[:-4] + ".json" if path.endswith(".csv") else path + ".json"
    if not os.path.exists(side_path):
        raise SystemExit(f"{side_path} not found; `{command}` needs the sidecar `sample` wrote beside the CSV")
    with open(side_path) as f:
        side = json.load(f)
    _check_sidecar(side, side_path)
    rows = _csv_rows(path, command)
    need = {"index", "window_row", "window_col", "wrong"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"{path}: expected the columns `sample` wrote ({', '.join(sorted(need))}); found "
                         f"{', '.join(rows[0].keys()) if rows else 'no rows'}. A spreadsheet saved with another delimiter "
                         "(semicolon) or with columns removed cannot be matched to its design")
    try:
        idx = np.array([int(float(r["index"])) for r in rows])                # "73.0" after a spreadsheet round trip is 73
    except (ValueError, OverflowError) as exc:                              # "inf" is an OverflowError, not a ValueError
        raise SystemExit(f"{path}: the `index` column is not the one `sample` wrote: {exc}")
    if not np.array_equal(idx, np.asarray(side["indices"], int)):
        raise SystemExit("the CSV's rows do not match the design in its sidecar; label the file `sample` wrote, in order")
    blank = [i for i, r in enumerate(rows) if str(r.get("wrong", "")).strip() == ""]
    if blank and side.get("design") == "sequential":
        # a sequential sample is labelled from the top: the labelled rows are a prefix of its random order, and the
        # rows below them are windows not labelled yet. A label below a blank would make the labelled set depend on
        # which windows the reviewer chose to leave, so it is refused.
        first = blank[0]
        later = [i for i in range(first, len(rows)) if str(rows[i].get("wrong", "")).strip() != ""]
        if later:
            raise SystemExit(f"row {later[0] + 2} is labelled below row {first + 2}, which is not: a sequential sample is "
                             "labelled from the top row down, in order. Write ? in a window that cannot be judged rather "
                             "than leaving it blank")
        if first == 0:
            raise SystemExit(f"{path}: no window is labelled yet; fill the `wrong` column from the top row down")
        side = dict(side, indices=list(side["indices"])[:first], n_drawn=len(rows))
        rows, idx, blank = rows[:first], idx[:first], []
    if blank:
        raise SystemExit(f"{len(blank)} of {len(rows)} windows have no `wrong` value (first at row {blank[0] + 2}); "
                         "every sampled window needs a 1 or a 0, or a ? where it cannot be judged, or the design's "
                         "interval is not the one you get")
    # Exactly 0 or 1. int(float(x)) would read a reviewer's "0.5" (not sure) as right and "1.9" as wrong, with
    # exit 0; a spreadsheet's "TRUE" is refused rather than guessed at.
    # A window the reviewer cannot judge is `?`: it stays in the design (until 1.4.1 this message told the reviewer
    # to leave it out, and a CSV that did was refused for not matching its design), and estimate and certify bound it.
    ok = {"0": 0, "1": 1, "0.0": 0, "1.0": 1, "?": 0}
    bad = [(i + 2, r["wrong"]) for i, r in enumerate(rows) if str(r["wrong"]).strip() not in ok]
    if bad:
        raise SystemExit(f"`wrong` must be exactly 1, 0 or ? per window; {len(bad)} row(s) are not, first at row "
                         f"{bad[0][0]}: {bad[0][1]!r}. Write ? where you cannot judge a window: estimate then bounds it "
                         "both ways and certify counts it as wrong; do not remove its row")
    wrong = np.array([ok[str(r["wrong"]).strip()] for r in rows])
    unjudged = np.array([str(r["wrong"]).strip() == "?" for r in rows])
    cond = side.get("condition")
    if "condition" in rows[0] and isinstance(cond, dict) and side.get("condition_grid") is not None:
        # The condition is fixed when the sample is drawn, and the sidecar is its source of truth: the raster is never
        # read again. A row whose `condition` was edited would be counted in the condition the sidecar says, so the
        # edit is refused rather than ignored.
        grid, names = np.asarray(side["condition_grid"]), list(cond.get("names", []))
        def recorded(i):
            return names[int(grid[i])] if 0 <= i < grid.size and 0 <= int(grid[i]) < len(names) else None
        bad = [(k + 2, r["condition"], recorded(int(i))) for k, (r, i) in enumerate(zip(rows, idx))
               if recorded(int(i)) is None or str(r["condition"]).strip() != recorded(int(i)).strip()]
        if bad:
            raise SystemExit(f"{path}: the `condition` column disagrees with the design in its sidecar on {len(bad)} "
                             f"row(s), first at row {bad[0][0]}: {bad[0][1]!r} where the sample recorded {bad[0][2]!r}. "
                             "The condition of each window is fixed when the sample is drawn; label the file `sample` "
                             "wrote without changing that column")
    sample = {k: (np.asarray(v) if k in ("indices", "strata", "strata_of_population", "tiles", "condition_grid") else v)
              for k, v in side.items()}
    return side, rows, idx, sample, wrong, unjudged


def _rounding_tolerance(text):
    """Half a unit in the last digit a number was written with, and never less than 1e-6 (a full-precision value
    still carries float noise from the round trip). "0.412" gives 5e-4; "4.12e-01" the same."""
    t = str(text).strip().lower()
    mant, _, exp = t.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return max(0.5 * 10.0 ** (-decimals + (int(exp) if exp else 0)), 1e-6)

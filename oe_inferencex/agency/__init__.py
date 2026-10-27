"""Grade a map with an agency's stratified reference sample: the points an agency labelled, and what they say.

The 1.17.0 raters found "no route to grade an existing agency stratified sample with its inclusion probabilities". An
agency's accuracy assessment publishes its points: each one's stratum, its reference label (NLCD also an alternate
label), the map's class at it, and either its weight (the stratum's size over its labels) or the strata's sizes
beside. `grade` reads them and returns what the agency reports: overall accuracy, each class's user's and producer's
accuracy and its share of the reference, and with two maps the difference of their overall accuracies on the same
points, each with an interval valid at every sample size (stratified.py) and the normal-theory interval of Stehman
(2014) and Olofsson et al. (2014) beside it, which is nominal; and the error matrix as estimated shares of the
population, without intervals.

A point's class is right when the map's class equals its reference label or, with alternate labels, any of them
(NLCD's rule; the reference class is then the map's class where it equals an alternate, else the primary label).
NLCD 2016's published per-class table counts the 2011 primary label as an alternate too, though its text does not say
so (exp117 in the research record): `--alternate` takes several columns.
Unknown values (a map's no-data, a missing reference label) are bounded, never dropped: dropping them would grade a
different population. A stratum with no label is bounded the same way unless the population is restricted to the
sampled strata on request.

Class codes are read as text; a code that is a whole number is normalised ("11.0" is "11"), so columns written by
different tools agree. Points read as written: the design is the agency's, and nothing here draws a sample."""
import csv
import math
import re

import numpy as np

from .stratified import METHOD_PURE, PURE_GAMMA, Design, _z, linear, pure_stratum_ratio, ratio

UNKNOWN_CODES = ("", "na", "nan", "none", "null")
WEIGHT_TOLERANCE = 1e-6           # relative spread of the weights within a stratum: above it, not a stratified SRS
_OTHER = object()                 # a completion of an unknown value: a class other than every named one

INTERVAL_NOTE = ("Each interval holds at {conf:g}% on its own, at every sample size; the intervals do not hold "
                 "jointly. The normal-theory interval beside each one is the agency's (Stehman 2014, Olofsson et al. "
                 "2014): nominal, and in a small class or a stratum of few labels it can cover far less than it says. "
                 "Its ends are the estimate plus or minus {z:.3g} standard errors, unclipped, as agencies publish them, "
                 "so they can pass 0 or 100%.")
UNKNOWN_NOTE = ("{k} point(s) have an unknown value (unknown values: {what}). They are bounded, not dropped: each lower "
                "end counts them against the map and each upper end for it, so the estimate is a range.")
NOT_COMPUTED_NOTE = ("{k} interval(s) are not computed (null in the JSON): a stratum is too large for the exact average "
                     "over the orders of its labels. The estimates and the normal-theory intervals are given.")
RATIO_NOTE = ("User's and producer's accuracies (--ratio direct) invert one betting test of theta_y - r (theta_y + "
              "theta_w) <= 0 over r, each end at alpha/2, in fixed bins of r inside which the test is monotone (method "
              "in the JSON); the box of two betting intervals, the default, is beside each as 'box'.")
NOT_COMPUTED_BOX_NOTE = ("{k} of them are direct ratio intervals, whose work grows with the labels in each kind; the "
                         "box of two betting intervals is printed in their place, labelled 'box'. It holds at the same "
                         "level on its own, but taking it only where the direct interval is missing is a choice made "
                         "on the data, since the limit depends on the label counts; --ratio box fixes it in advance.")
PURE_PARTLY_NOTE = ("The pure-stratum rule was not used for class(es) {classes} of {map}, which have more than one "
                    "declared pure stratum; their user's accuracies take the box interval.")
UNSAMPLED_NOTE = ("{k} stratum(s) hold no label: {names}, {share:.4g} of the population. Their values are bounded "
                  "(0 at each lower end, their whole share at each upper end), so every estimate is a range.")
POINT_READ_NOTE = ("The weights of {k} stratum(s) ({names}) are printed too coarsely to give one whole number of units: "
                   "each is read as points drawn from at least the fewest units its weights allow, which widens its "
                   "intervals a little, and the pure-stratum rule is not used. Stratum sizes (--strata) give them "
                   "exactly.")
SAMPLED_ONLY_NOTE = ("The population is restricted to the {H} sampled strata, as asked: {k} stratum(s) with no label "
                     "({share:.4g} of the stratum sizes given) are left out of every number.")
PURE_NOTE = ("User's accuracy of class(es) {classes} of {map} uses the pure-stratum rule: each class fills one declared "
             "stratum (every unit there mapped as it), and the interval is a union bound of exact hypergeometric "
             "intervals, at {g:g} of alpha for the strata declared pure for no class.")
PURE_SKIPPED_NOTE = ("The pure-stratum rule was not used for {map} ({why}); its user's accuracies take the box interval.")
ONE_LABEL_NOTE = ("{k} stratum(s) hold one label, whose variance normal theory cannot estimate; it leaves them out, "
                  "which makes its intervals narrower than it can justify.")
ALTERNATE_RULE = ("a point is right when the map's class equals its reference label or an alternate label; the "
                  "reference class is the map's class where it equals an alternate, else the reference label")
PRIMARY_RULE = "a point is right when the map's class equals its reference label"


def _code(value, unknown):
    """A class or stratum code as text, a whole number normalised ('11.0' -> '11'); None when unknown. Whole numbers
    written as digits are read exactly: through a float, two codes of 17 or more digits became one (the review of 26
    October 2026)."""
    s = "" if value is None else str(value).strip()
    if s.lower() in UNKNOWN_CODES:
        return None
    if re.fullmatch(r"[+-]?\d+(\.0*)?", s):
        s = str(int(s.split(".")[0]))
    else:
        try:
            f = float(s)
            if math.isfinite(f) and f == int(f) and abs(f) < 2 ** 53:
                s = str(int(f))
        except (ValueError, OverflowError):
            pass
    return None if s in unknown else s


def _half_unit(text):
    """Half a unit in the last place a number was written to ('2666.67' -> 0.005, '1.5e3' -> 50): the most its
    printing can have moved it."""
    m = re.fullmatch(r"\s*[+-]?\d*(?:\.(\d*))?(?:[eE]([+-]?\d+))?\s*", str(text))
    if not m:
        return 0.0
    return 0.5 * 10.0 ** (int(m.group(2) or 0) - len(m.group(1) or ""))


def _read_table(path):
    """A CSV's column names, the names given twice, and its rows as (line, dict of the columns named once), read as a
    spreadsheet shows them: a byte-order mark dropped (Excel's 'CSV UTF-8'), the names stripped, a short row padded
    with empty cells (unknown values), a blank row skipped, a row with more filled cells than the header refused.
    Unnamed columns (pandas' and R's row index) are left out, and a column named twice is kept out of the rows, so
    read_points refuses it only if asked to read it: csv.DictReader kept its last copy where a spreadsheet shows the
    first (the reviews of 26 October 2026)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return [], set(), []
        header = [h.strip() for h in header]
        twice = {h for h in header if h and header.count(h) > 1}
        keep = [(j, h) for j, h in enumerate(header) if h and h not in twice]
        rows = []
        for line in reader:
            if not line or all(not c.strip() for c in line):
                continue
            if any(c.strip() for c in line[len(header):]):
                raise ValueError(f"{path} line {reader.line_num}: more fields than the header's {len(header)} columns")
            line = line + [""] * (len(header) - len(line))
            rows.append((reader.line_num, {h: line[j] for j, h in keep}))
    return [h for h in header if h], twice, rows


def _sort_key(code):
    try:
        return (0, float(code), code)
    except ValueError:
        return (1, 0.0, code)


def read_points(path, stratum, reference, maps, alternate=None, weight=None, nodata=()):
    """The labelled points of a CSV: their stratum, reference label, alternate labels (optional: one column or several,
    a tuple per point), each map's class (dict name -> list) and weight (optional, with `weight_half_unit`, half a
    unit in the last place each was written to). Codes in `nodata`, and empty or NA cells, are unknown (None)."""
    unknown = {c for c in (_code(v, ()) for v in nodata) if c is not None}
    alts = [alternate] if isinstance(alternate, str) else list(alternate or [])
    header, twice, rows = _read_table(path)
    if not rows:
        raise ValueError(f"{path} holds no point")
    for col in [stratum, reference, weight] + alts + list(maps):
        if col is not None and col in twice:
            raise ValueError(f"{path} names the column {col!r} twice; a spreadsheet shows one copy and a CSV reader "
                             f"another, so it is not read")
        if col is not None and col not in header:
            raise ValueError(f"{path} has no column {col!r}; its columns are {', '.join(dict.fromkeys(header))}")
    if len(set(maps)) != len(maps):
        raise ValueError("each --map names a different column")
    # the stratum may be a map's own column (a sample stratified by the map's class); the labels and weights may not
    roles = [reference] + alts + ([weight] if weight else [])
    if set(roles) & (set(maps) | {stratum}) or len(set(roles)) != len(roles):
        raise ValueError("the reference, alternate and weight columns must differ from each other, from the stratum "
                         "and from every map column")
    out = {"stratum": [], "reference": [], "alternate": [], "maps": {m: [] for m in maps}, "weight": [],
           "weight_half_unit": []}
    for i, r in rows:
        s = _code(r[stratum], ())
        if s is None:
            raise ValueError(f"{path} line {i}: no stratum")
        out["stratum"].append(s)
        out["reference"].append(_code(r[reference], unknown))
        if alts:
            out["alternate"].append(tuple(c for c in (_code(r[a], unknown) for a in alts) if c is not None))
        else:
            out["alternate"].append(None)
        for m in maps:
            out["maps"][m].append(_code(r[m], unknown))
        if weight:
            try:
                w = float(r[weight])
            except ValueError:
                raise ValueError(f"{path} line {i}: weight {r[weight]!r} is not a number") from None
            if not (math.isfinite(w) and w > 0):
                raise ValueError(f"{path} line {i}: weight {w} must be positive")
            out["weight"].append(w)
            out["weight_half_unit"].append(_half_unit(r[weight]))
    return out


def read_strata(path, point_strata=None):
    """Stratum sizes from a CSV whose first column is the stratum and second its size, after a header row. Refused:
    an unnamed first column (a row index, as pandas and R write by default, whose row numbers would be read as strata
    and the strata as sizes), and a first row that is a stratum (one the points name, with a number beside it; without
    `point_strata`, any first row whose second cell is a number), which read as a header would drop that stratum and
    the share of the population its bounds stand for (the reviews of 26 October 2026)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{path}: a header row, then one row per stratum: stratum, size")
    if not rows[0][0].strip():
        raise ValueError(f"{path}: the first column has no name, as a row index written by pandas or R has; write the "
                         f"file with the stratum first and its size second")
    try:
        float(rows[0][1])
        numeric = True
    except ValueError:
        numeric = False
    first = _code(rows[0][0], ())
    if numeric and (point_strata is None or first in set(point_strata)):
        raise ValueError(f"{path}: the first row ({', '.join(rows[0][:2])}) looks like a stratum, not a header; add a "
                         f"header row (stratum,size)")
    sizes = {}
    for i, r in enumerate(rows[1:], start=2):
        if not r or all(not c.strip() for c in r):
            continue
        s = _code(r[0], ())
        try:
            v = float(r[1])
        except (ValueError, IndexError):
            raise ValueError(f"{path} line {i}: size {r[1:2]} is not a number") from None
        if s is None or not (math.isfinite(v) and v > 0):
            raise ValueError(f"{path} line {i}: a stratum and a positive size")
        if s in sizes:
            raise ValueError(f"{path} line {i}: stratum {s} given twice")
        sizes[s] = v
    return sizes


def _frame(names, sizes, kind, unit_area):
    """W, N and Nlo for the sampled strata from their sizes: whole units drawn without replacement, or areas (a point
    frame; with a unit area, units drawn without replacement from at least area / unit_area of them)."""
    sizes = np.asarray(sizes, float)
    if kind == "units":
        if unit_area is not None:
            raise ValueError("--unit-area is for sizes given as areas")
        N = np.round(sizes)
        bad = np.flatnonzero(np.abs(sizes - N) > np.maximum(1e-6 * sizes, 1e-6))
        if bad.size:
            raise ValueError(f"stratum {names[bad[0]]}: size {sizes[bad[0]]:g} is not a whole number of units; give "
                             f"areas with --sizes area")
        return N, N, None
    if kind != "area":
        raise ValueError(f"sizes are 'units' or 'area', got {kind!r}")
    if unit_area is not None and not (math.isfinite(unit_area) and unit_area > 0):
        raise ValueError(f"--unit-area must be positive, got {unit_area}")
    Nlo = None if unit_area is None else np.floor(sizes / unit_area)
    return sizes, np.full(sizes.size, math.inf), Nlo


def design_from_weights(strata, weights, kind="units", unit_area=None, half_unit=None):
    """The design a point table with weights describes: a stratum's size is the sum of its points' weights (whole
    units, or an area), which must be equal within the stratum (a stratified simple random sample). With whole units,
    the sizes the sum allows are the whole numbers within the weights' printing of it (`half_unit`, per weight: half a
    unit in its last written place; without it, each weight's shortest printing, as the command line reads a table):
    none, and the stratum is refused (weights that are areas, printed finely enough to tell); one, and that is its size;
    several, and the stratum is read as a point frame drawn from at least the fewest of them, with the deflation, as a
    size picked among them could make a near-census exact (the reviews of 26 October 2026: the pure-stratum rule then
    returned a point that missed the truth). Area weights printed coarsely enough pass as units; the printing cannot
    tell them apart."""
    ids = sorted(set(strata), key=_sort_key)
    pos = {s: h for h, s in enumerate(ids)}
    hidx = np.array([pos[s] for s in strata])
    w = np.asarray(weights, float)
    tot = np.bincount(hidx, weights=w, minlength=len(ids))
    for h, s in enumerate(ids):
        wh = w[hidx == h]
        if wh.max() / wh.min() - 1 > WEIGHT_TOLERANCE:
            raise ValueError(f"the weights differ within stratum {s} ({wh.min():g} to {wh.max():g}): not a "
                             f"stratified simple random sample, which this command grades")
    if kind == "units":
        if unit_area is not None:
            raise ValueError("--unit-area is for sizes given as areas")
        if half_unit is None:
            half_unit = [_half_unit(repr(float(x))) for x in w]
        n = np.bincount(hidx, minlength=len(ids))
        # half a unit per weight, and the float sum's own error (a few ulps per term)
        tol = (np.bincount(hidx, weights=np.asarray(half_unit, float), minlength=len(ids))
               + 4 * n * np.finfo(float).eps * tot)
        least, most = np.ceil(tot - tol), np.floor(tot + tol)
        bad = np.flatnonzero(least > most)
        if bad.size:
            h = int(bad[0])
            raise ValueError(f"stratum {ids[h]}: its weights add up to {tot[h]:.10g}, which their printing does not "
                             f"allow to be a whole number of units; if they are areas, give --sizes area")
        one = least == most
        N = np.where(one, least, np.inf)
        S = np.where(one, least, tot)
        d = Design(hidx, S / S.sum(), N, None if one.all() else np.where(one, np.nan, np.maximum(least, n)), names=ids)
        d.point_read = [ids[h] for h in np.flatnonzero(~one)]
        return d
    S, N, Nlo = _frame(ids, tot, kind, unit_area)
    return Design(hidx, S / S.sum(), N, Nlo, names=ids)


def design_from_sizes(strata, sizes, kind="units", unit_area=None, sampled_only=False):
    """The design of points in strata of known sizes (dict stratum -> size). Strata with no label are bounded, or,
    with sampled_only, left out of the population."""
    missing = sorted(set(strata) - set(sizes), key=_sort_key)
    if missing:
        raise ValueError(f"stratum {missing[0]} holds points but has no size ({len(missing)} such stratum(s))")
    ids = sorted(set(strata), key=_sort_key)
    empty = sorted(set(sizes) - set(strata), key=_sort_key)
    pos = {s: h for h, s in enumerate(ids)}
    hidx = np.array([pos[s] for s in strata])
    S, N, Nlo = _frame(ids, [sizes[s] for s in ids], kind, unit_area)
    if kind == "units":
        _frame(empty, [sizes[s] for s in empty], kind, unit_area)   # whole sizes there too
    E = 0.0 if sampled_only else float(sum(sizes[s] for s in empty))
    total = S.sum() + E
    d = Design(hidx, S / total, N, Nlo, names=ids, W_unsampled=E / total, names_unsampled=[] if sampled_only else empty)
    d.left_out = empty if sampled_only else []
    d.left_out_share = (sum(sizes[s] for s in empty) / sum(sizes.values())) if sampled_only else 0.0
    return d


# ----------------------------------------------------------------------------------------------- values per point
def _alts(a):
    """A point's alternate labels as a tuple: none, one code, or several (each an acceptable reference label)."""
    if a is None:
        return ()
    if isinstance(a, (tuple, list)):
        return tuple(x for x in a if x is not None)
    return (a,)


def _right(m, r, alts):
    return m == r or m in alts


def _ref_class(m, r, alts):
    return m if m in alts else r


def _completions(m, r, alts, c):
    """The (map, reference) values a point can take: its own where known; for an unknown map value, the class c, the
    alternate labels and a class other than all of them; for an unknown reference label, c and another class."""
    ms = [m] if m is not None else list(dict.fromkeys((c,) + alts)) + [_OTHER]
    rs = [r] if r is not None else [c, _OTHER]
    return [(a, b) for a in ms for b in rs]


def _diff_bounds(ma, mb, r, alts):
    """The least and greatest of right_a - right_b at a point, over the joint completions of its unknown values: two
    maps that give an unknown reference the same class differ by 0 there, where each map's own bounds allowed -1 to 1
    (the review of 26 October 2026). Completions: the values named at the point and three classes named nowhere."""
    if ma is not None and mb is not None and r is not None:
        d = float(_right(ma, r, alts)) - float(_right(mb, r, alts))
        return d, d
    cand = list(dict.fromkeys(v for v in (ma, mb, r) + alts if v is not None)) + [object(), object(), object()]
    ds = [float(_right(a, q, alts)) - float(_right(b, q, alts))
          for a in ([ma] if ma is not None else cand) for b in ([mb] if mb is not None else cand)
          for q in ([r] if r is not None else cand)]
    return min(ds), max(ds)


def _oa_bounds(m, r, alts):
    """The least and greatest correctness of a point: known when both values are; an unknown map value can be right
    or wrong; an unknown reference label can be right or wrong unless the map's class is an alternate label."""
    if m is not None and r is not None:
        v = float(_right(m, r, alts))
        return v, v
    if m is not None and m in alts:
        return 1.0, 1.0
    return 0.0, 1.0


def _class_bounds(m, r, alts, c):
    """For class c, the least and greatest over a point's completions of: the user's accuracy's y (mapped c and
    right) and w (mapped c and wrong); the producer's y (mapped c and reference c) and w (reference c, mapped
    otherwise); and the reference-share indicator."""
    vals = []
    for a, b in _completions(m, r, alts, c):
        rs = _ref_class(a, b, alts)
        ok = _right(a, b, alts)
        vals.append((a == c and ok, a == c and not ok, a == c and rs == c, rs == c and a != c, rs == c))
    v = np.array(vals, dtype=float)
    return v.min(axis=0), v.max(axis=0)


def _natural(codes):
    return sorted(codes, key=_sort_key)


# ----------------------------------------------------------------------------------------------- grading
def grade(design, reference, maps, alternate=None, pure=None, alpha=0.05, ratio_method="box"):
    """Overall accuracy per map, the difference between maps, and per class the user's and producer's accuracy and
    the share of the reference, each with its valid interval and the normal-theory one, and the error matrix's
    estimated shares. reference: one code or None per point; alternate: per point one code, a tuple of codes or None;
    maps: dict name -> codes, the first the map the strata were drawn from; pure: dict stratum -> class, strata every
    unit of which the first map labels the class (an agency's no-change strata), used for that map's user's
    accuracies; ratio_method: "box" (the default) or "direct", the interval of the user's and producer's accuracies
    the pure-stratum rule does not give (stratified.ratio)."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if ratio_method not in ("box", "direct"):
        raise ValueError(f"ratio_method is 'box' or 'direct', got {ratio_method!r}")
    names = list(maps)
    if not names:
        raise ValueError("give at least one map")
    P = design.hidx.size
    alternate = [()] * P if alternate is None else [_alts(a) for a in alternate]
    for k, v in [("reference", reference), ("alternate", alternate)] + [(f"map {m}", maps[m]) for m in names]:
        if len(v) != P:
            raise ValueError(f"{k} has {len(v)} values; the design has {P} points")
    pure = dict(pure or {})
    notes = [INTERVAL_NOTE.format(conf=100 * (1 - alpha), z=_z(alpha))]
    classes = _natural({c for c in reference if c is not None}
                       | {c for m in names for c in maps[m] if c is not None})
    out = {"points": P, "strata": design.H, "maps": names, "classes": classes, "alpha": alpha,
           "rule": ALTERNATE_RULE if any(alternate) else PRIMARY_RULE,
           "overall_accuracy": {}, "differences": {}, "per_class": {}, "error_matrix": {}, "unknown": {}}
    if ratio_method == "direct":                                # the default's output stays 1.19.0's, key for key
        out["ratio_method"] = ratio_method
        notes.append(RATIO_NOTE)
    if design.W_unsampled > 0:
        notes.append(UNSAMPLED_NOTE.format(k=len(design.names_unsampled), names=", ".join(design.names_unsampled[:8])
                                           + (" ..." if len(design.names_unsampled) > 8 else ""),
                                           share=design.W_unsampled))
    if getattr(design, "left_out", None):
        notes.append(SAMPLED_ONLY_NOTE.format(H=design.H, k=len(design.left_out), share=design.left_out_share))
    if getattr(design, "point_read", None):
        notes.append(POINT_READ_NOTE.format(k=len(design.point_read), names=", ".join(design.point_read[:8])
                                            + (" ..." if len(design.point_read) > 8 else "")))
    ones = int((design.n == 1).sum())
    if ones:
        notes.append(ONE_LABEL_NOTE.format(k=ones))
    n_ref_unknown = sum(r is None for r in reference)
    out["unknown"]["reference"] = n_ref_unknown
    oa_bounds = {}
    for m in names:
        b = np.array([_oa_bounds(maps[m][i], reference[i], alternate[i]) for i in range(P)])
        oa_bounds[m] = b
        out["unknown"][m] = sum(v is None for v in maps[m])
        out["overall_accuracy"][m] = _q(linear(design, b[:, 0], b[:, 1], alpha), b)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.array([_diff_bounds(maps[a][j], maps[b][j], reference[j], alternate[j]) for j in range(P)])
            q = linear(design, (1.0 + d[:, 0]) / 2, (1.0 + d[:, 1]) / 2, alpha, scale=2.0, shift=-1.0)
            q["disagreeing_points"] = int(np.sum(np.array(maps[a], dtype=object) != np.array(maps[b], dtype=object)))
            out["differences"][f"{a} minus {b}"] = q
    pure_use, pure_skip, pure_many = _pure_plan(design, maps[names[0]], pure, out["unknown"][names[0]], n_ref_unknown)
    if pure_skip:
        notes.append(PURE_SKIPPED_NOTE.format(map=names[0], why=pure_skip))
    if pure_many:
        notes.append(PURE_PARTLY_NOTE.format(classes=", ".join(pure_many), map=names[0]))
    for m in names:
        per, used = {}, []
        for c in classes:
            lo, hi = zip(*(_class_bounds(maps[m][i], reference[i], alternate[i], c) for i in range(P)))
            lo, hi = np.array(lo), np.array(hi)
            row = {"labels_mapped": int(sum(v == c for v in maps[m])), "labels_reference": int(lo[:, 4].sum())}
            own = pure_use.get(c) if m == names[0] else None
            if own is not None:
                u = ratio(design, lo[:, 0], hi[:, 1], hi[:, 0], lo[:, 1], alpha, method=None)
                kr = np.bincount(design.hidx, weights=lo[:, 0], minlength=design.H)
                kw = np.bincount(design.hidx, weights=lo[:, 1], minlength=design.H)
                mixed = [h for h in range(design.H) if design.names[h] not in pure]
                u["interval"] = list(pure_stratum_ratio(design, own, mixed, kr, kw, alpha))
                u["method"] = METHOD_PURE
                u["pure_stratum"] = design.names[own]
                used.append(c)
            else:
                u = ratio(design, lo[:, 0], hi[:, 1], hi[:, 0], lo[:, 1], alpha, method=ratio_method)
            row["user"] = u
            row["producer"] = ratio(design, lo[:, 2], hi[:, 3], hi[:, 2], lo[:, 3], alpha, method=ratio_method)
            row["reference_share"] = linear(design, lo[:, 4], hi[:, 4], alpha)
            mapped = np.array([v == c for v in maps[m]], float)
            known = np.array([v is not None for v in maps[m]], float)
            ms_lo = linear_estimate(design, mapped)
            ms_hi = linear_estimate(design, mapped + (1 - known)) + design.W_unsampled
            row["map_share_estimate"] = ms_lo if ms_hi == ms_lo else [ms_lo, ms_hi]
            per[c] = row
        out["per_class"][m] = per
        out["error_matrix"][m] = _error_matrix(design, maps[m], reference, alternate, classes)
        if used:
            notes.append(PURE_NOTE.format(classes=", ".join(used), map=m, g=PURE_GAMMA))
    unk = [f"{k} {v}" for k, v in out["unknown"].items() if v]
    if unk:
        points = sum(reference[i] is None or any(maps[m][i] is None for m in names) for i in range(P))
        notes.append(UNKNOWN_NOTE.format(k=points, what=", ".join(unk)))
    skipped = sum(q.get("interval", 0) is None for q in _quantities(out))
    if skipped:
        notes.append(NOT_COMPUTED_NOTE.format(k=skipped))
    boxed = sum(q.get("interval", 0) is None and q.get("box") is not None for q in _quantities(out))
    if boxed:
        notes.append(NOT_COMPUTED_BOX_NOTE.format(k=boxed))
    out["notes"] = notes
    return out


def _quantities(res):
    """Every number of a grading that carries an interval."""
    yield from res["overall_accuracy"].values()
    yield from res["differences"].values()
    for per in res["per_class"].values():
        for row in per.values():
            yield from (row["user"], row["producer"], row["reference_share"])


def linear_estimate(design, x):
    """sum_h W_h mean_h(x) over the sampled strata."""
    x = np.asarray(x, float)
    return float(sum(design.W[h] * x[design.masks[h]].mean() for h in range(design.H)))


def _q(q, bounds):
    q["labels"] = int(bounds.shape[0])
    return q


def _pure_plan(design, first_map, pure, map_unknown, ref_unknown):
    """class -> the index of its one declared pure stratum, where the rule applies; why it applies to no class, if
    so; and the classes it skips for having more than one declared pure stratum. A declaration contradicted by a
    labelled point (mapped otherwise, or unknown) is refused: it is false."""
    if not pure:
        return {}, None, []
    pos = {s: h for h, s in enumerate(design.names)}
    by_class = {}
    for s, c in pure.items():
        if s not in pos:
            raise ValueError(f"--pure {s}={c}: stratum {s} holds no labelled point")
        h = pos[s]
        idx = np.flatnonzero(design.masks[h])
        bad = [i for i in idx if first_map[i] != c]
        if bad:
            raise ValueError(f"--pure {s}={c}: {len(bad)} of the {idx.size} points of stratum {s} are not mapped as "
                             f"{c} by the first map (e.g. {first_map[bad[0]]!r}); the declaration is false")
        by_class.setdefault(c, []).append(h)
    why = None
    if design.W_unsampled > 0:
        why = "a stratum holds no label"
    elif not np.isfinite(design.N).all():
        why = "a stratum's size is not known in whole units"
    elif map_unknown or ref_unknown:
        why = "some values are unknown"
    if why:
        return {}, why, []
    use = {c: hs[0] for c, hs in by_class.items() if len(hs) == 1}
    return use, None, sorted((c for c, hs in by_class.items() if len(hs) > 1), key=_sort_key)


def _error_matrix(design, mapped, reference, alternate, classes):
    """The estimated share of the population in each (map class, reference class) cell, the reference class by the
    rule in use; points with an unknown value in their own row or column. Point estimates, without intervals. Strata
    with no label are in no cell: the cells add up to 1 less `unsampled_share`."""
    labels = classes + ["unknown"]
    pos = {c: i for i, c in enumerate(classes)}
    k = len(labels)
    M = np.zeros((k, k))
    for h in range(design.H):
        idx = np.flatnonzero(design.masks[h])
        for i in idx:
            m, r, a = mapped[i], reference[i], alternate[i]
            rs = _ref_class(m, r, a) if m is not None else r
            M[pos.get(m, k - 1), pos.get(rs, k - 1)] += design.W[h] / idx.size
    keep = [i for i in range(k) if i < k - 1 or M[k - 1].any() or M[:, k - 1].any()]
    return {"rows_map_columns_reference": [labels[i] for i in keep],
            "share": [[float(M[i, j]) for j in keep] for i in keep], "unsampled_share": design.W_unsampled}

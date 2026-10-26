"""Grade a map with an agency's stratified reference sample: the points an agency labelled, and what they say.

The 1.17.0 raters found "no route to grade an existing agency stratified sample with its inclusion probabilities". An
agency's accuracy assessment publishes its points: each one's stratum, its reference label (NLCD also an alternate
label), the map's class at it, and either its weight (the stratum's size over its labels) or the strata's sizes
beside. `grade` reads them and returns what the agency reports (overall accuracy; each class's user's and producer's
accuracy and its share of the reference; the error matrix as shares of the population; with two maps, the difference
of their overall accuracies on the same points), each with an interval valid at every sample size (stratified.py)
and the normal-theory interval of Stehman (2014) and Olofsson et al. (2014) beside it, which is nominal.

A point's class is right when the map's class equals its reference label or, with an alternate label, either label
(NLCD's rule; the reference class is then the map's class where it equals the alternate, else the primary label).
Unknown values (a map's no-data, a missing reference label) are bounded, never dropped: dropping them would grade a
different population. A stratum with no label is bounded the same way unless the population is restricted to the
sampled strata on request.

Class codes are read as text; a code that is a whole number is normalised ("11.0" is "11"), so columns written by
different tools agree. Points read as written: the design is the agency's, and nothing here draws a sample."""
import csv
import math

import numpy as np

from .stratified import METHOD_PURE, PURE_GAMMA, Design, linear, pure_stratum_ratio, ratio

UNKNOWN_CODES = ("", "na", "nan", "none", "null")
WEIGHT_TOLERANCE = 1e-6           # relative spread of the weights within a stratum: above it, not a stratified SRS
_OTHER = object()                 # a completion of an unknown value: a class other than every named one

INTERVAL_NOTE = ("Each interval holds at {conf:g}% on its own, at every sample size; the intervals do not hold "
                 "jointly. The normal-theory interval beside each one is the agency's (Stehman 2014, Olofsson et al. "
                 "2014): nominal, and in a small class or a stratum of few labels it can cover far less than it says.")
UNKNOWN_NOTE = ("{k} point(s) have an unknown value ({what}). They are bounded, not dropped: each lower end counts them "
                "against the map and each upper end for it, so the estimate is a range.")
UNSAMPLED_NOTE = ("{k} stratum(s) hold no label: {names}, {share:.4g} of the population. Their values are bounded "
                  "(0 at each lower end, their whole share at each upper end), so every estimate is a range.")
SAMPLED_ONLY_NOTE = ("The population is restricted to the {H} sampled strata, as asked: {k} stratum(s) with no label "
                     "({share:.4g} of the stratum sizes given) are left out of every number.")
PURE_NOTE = ("User's accuracy of class(es) {classes} of {map} uses the pure-stratum rule: each class fills one declared "
             "stratum (every unit there mapped as it), and the interval is a union bound of exact hypergeometric "
             "intervals, at {g:g} of alpha for the strata declared pure for no class.")
PURE_SKIPPED_NOTE = ("The pure-stratum rule was not used for {map} ({why}); its user's accuracies take the box interval.")
ONE_LABEL_NOTE = ("{k} stratum(s) hold one label, whose variance normal theory cannot estimate; it leaves them out, "
                  "which makes its intervals narrower than it can justify.")
ALTERNATE_RULE = ("a point is right when the map's class equals its reference label or its alternate label; the "
                  "reference class is the map's class where it equals the alternate, else the reference label")
PRIMARY_RULE = "a point is right when the map's class equals its reference label"


def _code(value, unknown):
    """A class or stratum code as text, a whole number normalised ('11.0' -> '11'); None when unknown."""
    s = "" if value is None else str(value).strip()
    if s.lower() in UNKNOWN_CODES:
        return None
    try:
        f = float(s)
        if math.isfinite(f) and f == int(f):
            s = str(int(f))
    except (ValueError, OverflowError):
        pass
    return None if s in unknown else s


def _sort_key(code):
    try:
        return (0, float(code), code)
    except ValueError:
        return (1, 0.0, code)


def read_points(path, stratum, reference, maps, alternate=None, weight=None, nodata=()):
    """The labelled points of a CSV: their stratum, reference label, alternate label (optional), each map's class
    (dict name -> list) and weight (optional). Codes in `nodata`, and empty or NA cells, are unknown (None)."""
    unknown = {c for c in (_code(v, ()) for v in nodata) if c is not None}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} holds no point")
    cols = set(rows[0])
    for col in [stratum, reference, alternate, weight] + list(maps):
        if col is not None and col not in cols:
            raise ValueError(f"{path} has no column {col!r}; its columns are {', '.join(sorted(cols))}")
    if len(set(maps)) != len(maps):
        raise ValueError("each --map names a different column")
    out = {"stratum": [], "reference": [], "alternate": [], "maps": {m: [] for m in maps}, "weight": []}
    for i, r in enumerate(rows, start=2):
        s = _code(r[stratum], ())
        if s is None:
            raise ValueError(f"{path} line {i}: no stratum")
        out["stratum"].append(s)
        out["reference"].append(_code(r[reference], unknown))
        out["alternate"].append(_code(r[alternate], unknown) if alternate else None)
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
    return out


def read_strata(path):
    """Stratum sizes from a CSV whose first column is the stratum and second its size (a header row first)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{path}: a header row, then one row per stratum: stratum, size")
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


def design_from_weights(strata, weights, kind="units", unit_area=None):
    """The design a point table with weights describes: a stratum's size is the sum of its points' weights (whole
    units, rounded, or an area), which must be equal within the stratum (a stratified simple random sample)."""
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
        tot = np.round(tot)                     # weights printed to a few digits add up to a size near a whole number
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
def _right(m, r, alt):
    return m == r or (alt is not None and m == alt)


def _ref_class(m, r, alt):
    return m if (alt is not None and m == alt) else r


def _completions(m, r, alt, c):
    """The (map, reference) values a point can take: its own where known; for an unknown map value, the class c, the
    alternate label and a class other than both; for an unknown reference label, c and another class."""
    ms = [m] if m is not None else [v for v in (c, alt) if v is not None] + [_OTHER]
    rs = [r] if r is not None else [c, _OTHER]
    return [(a, b) for a in ms for b in rs]


def _oa_bounds(m, r, alt):
    """The least and greatest correctness of a point: known when both values are; an unknown map value can be right
    or wrong; an unknown reference label can be right or wrong unless the map's class is the alternate label."""
    if m is not None and r is not None:
        v = float(_right(m, r, alt))
        return v, v
    if m is not None and alt is not None and m == alt:
        return 1.0, 1.0
    return 0.0, 1.0


def _class_bounds(m, r, alt, c):
    """For class c, the least and greatest over a point's completions of: the user's accuracy's y (mapped c and
    right) and w (mapped c and wrong); the producer's y (mapped c and reference c) and w (reference c, mapped
    otherwise); and the reference-share indicator."""
    vals = []
    for a, b in _completions(m, r, alt, c):
        rs = _ref_class(a, b, alt)
        ok = _right(a, b, alt)
        vals.append((a == c and ok, a == c and not ok, a == c and rs == c, rs == c and a != c, rs == c))
    v = np.array(vals, dtype=float)
    return v.min(axis=0), v.max(axis=0)


def _natural(codes):
    return sorted(codes, key=_sort_key)


# ----------------------------------------------------------------------------------------------- grading
def grade(design, reference, maps, alternate=None, pure=None, alpha=0.05):
    """Overall accuracy per map, the difference between maps, and per class the user's and producer's accuracy and
    the share of the reference, each with its valid interval and the normal-theory one, and the error matrix's
    estimated shares. reference, alternate: one code or None per point; maps: dict name -> codes, the first the map
    the strata were drawn from; pure: dict stratum -> class, strata every unit of which the first map labels the
    class (an agency's no-change strata), used for that map's user's accuracies."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    names = list(maps)
    if not names:
        raise ValueError("give at least one map")
    P = design.hidx.size
    alternate = [None] * P if alternate is None else list(alternate)
    for k, v in [("reference", reference), ("alternate", alternate)] + [(f"map {m}", maps[m]) for m in names]:
        if len(v) != P:
            raise ValueError(f"{k} has {len(v)} values; the design has {P} points")
    pure = dict(pure or {})
    notes = [INTERVAL_NOTE.format(conf=100 * (1 - alpha))]
    classes = _natural({c for c in reference if c is not None}
                       | {c for m in names for c in maps[m] if c is not None})
    out = {"points": P, "strata": design.H, "maps": names, "classes": classes, "alpha": alpha,
           "rule": ALTERNATE_RULE if any(a is not None for a in alternate) else PRIMARY_RULE,
           "overall_accuracy": {}, "differences": {}, "per_class": {}, "error_matrix": {}, "unknown": {}}
    if design.W_unsampled > 0:
        notes.append(UNSAMPLED_NOTE.format(k=len(design.names_unsampled), names=", ".join(design.names_unsampled[:8])
                                           + (" ..." if len(design.names_unsampled) > 8 else ""),
                                           share=design.W_unsampled))
    if getattr(design, "left_out", None):
        notes.append(SAMPLED_ONLY_NOTE.format(H=design.H, k=len(design.left_out), share=design.left_out_share))
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
            lo = (1.0 + oa_bounds[a][:, 0] - oa_bounds[b][:, 1]) / 2
            hi = (1.0 + oa_bounds[a][:, 1] - oa_bounds[b][:, 0]) / 2
            q = linear(design, lo, hi, alpha, scale=2.0, shift=-1.0)
            q["disagreeing_points"] = int(np.sum(np.array(maps[a], dtype=object) != np.array(maps[b], dtype=object)))
            out["differences"][f"{a} minus {b}"] = q
    pure_use, pure_skip = _pure_plan(design, reference, maps[names[0]], pure, out["unknown"][names[0]], n_ref_unknown)
    if pure and pure_skip:
        notes.append(PURE_SKIPPED_NOTE.format(map=names[0], why=pure_skip))
    for m in names:
        per, used = {}, []
        for c in classes:
            lo, hi = zip(*(_class_bounds(maps[m][i], reference[i], alternate[i], c) for i in range(P)))
            lo, hi = np.array(lo), np.array(hi)
            row = {"labels_mapped": int(sum(v == c for v in maps[m])), "labels_reference": int(lo[:, 4].sum())}
            own = pure_use.get(c) if m == names[0] else None
            if own is not None:
                u = ratio(design, lo[:, 0], hi[:, 1], hi[:, 0], lo[:, 1], alpha, box=False)
                kr = np.bincount(design.hidx, weights=lo[:, 0], minlength=design.H)
                kw = np.bincount(design.hidx, weights=lo[:, 1], minlength=design.H)
                mixed = [h for h in range(design.H) if design.names[h] not in pure]
                u["interval"] = list(pure_stratum_ratio(design, own, mixed, kr, kw, alpha))
                u["method"] = METHOD_PURE
                u["pure_stratum"] = design.names[own]
                used.append(c)
            else:
                u = ratio(design, lo[:, 0], hi[:, 1], hi[:, 0], lo[:, 1], alpha)
            row["user"] = u
            row["producer"] = ratio(design, lo[:, 2], hi[:, 3], hi[:, 2], lo[:, 3], alpha)
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
        notes.append(UNKNOWN_NOTE.format(k=sum(out["unknown"].values()), what=", ".join(unk)))
    out["notes"] = notes
    return out


def linear_estimate(design, x):
    """sum_h W_h mean_h(x) over the sampled strata."""
    x = np.asarray(x, float)
    return float(sum(design.W[h] * x[design.masks[h]].mean() for h in range(design.H)))


def _q(q, bounds):
    q["labels"] = int(bounds.shape[0])
    return q


def _pure_plan(design, reference, first_map, pure, map_unknown, ref_unknown):
    """class -> the index of its one declared pure stratum, where the rule applies; and why it does not, if not. A
    declaration contradicted by a labelled point (mapped otherwise, or unknown) is refused: it is false."""
    if not pure:
        return {}, None
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
        why = "the strata are areas, not whole units"
    elif map_unknown or ref_unknown:
        why = "some values are unknown"
    if why:
        return {}, why
    use = {c: hs[0] for c, hs in by_class.items() if len(hs) == 1}
    many = sorted((c for c, hs in by_class.items() if len(hs) > 1), key=_sort_key)
    return use, (f"class(es) {', '.join(many)} have more than one declared pure stratum" if many else None)


def _error_matrix(design, mapped, reference, alternate, classes):
    """The estimated share of the population in each (map class, reference class) cell, the reference class by the
    rule in use; points with an unknown value in their own row or column."""
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
            "share": [[float(M[i, j]) for j in keep] for i in keep]}

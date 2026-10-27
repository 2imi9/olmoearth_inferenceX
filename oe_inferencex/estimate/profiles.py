"""Profiles: a partition of a map's windows fixed from the map alone, before any label, for localized certification.

A window's profile is read from fields the tool already has for every window: its confidence band, its input
condition, the class the map gives it and its block of the window grid, as many of them as `--profile-by` names. Each
combination of their values that occurs is a cell; a cell with fewer than `min_windows` windows is folded, with every
other such cell, into one profile named "other". The partition and the rule are written to the sample's sidecar
before any label is read (sample --design profile), and `certify` then calls each profile good, bad or undecided
against alpha (certify.localized). No label is read here.
"""
import numpy as np

from oe_inferencex.common.sampling import UNRECORDED, _condition_index

MAX_PROFILES = 12                 # at most this many profiles after folding, so delta is not split too thin
MIN_PROFILE_WINDOWS = 100         # a cell with fewer windows is folded into "other"
MAX_BANDS = 5                     # confidence:Q takes 2 to MAX_BANDS bands
MAX_GRID = 4                      # grid:G takes 2 to MAX_GRID blocks a side
FIELDS = ("confidence", "condition", "class", "grid")      # the canonical order of the fields, and of a profile's name
OTHER = "other"

PROFILE_RULE_TEXT = ("A window's profile is fixed from the map alone, before any label: its confidence band (the band "
                     "edges are quantiles of the valid windows' confidence; a window tied at an edge goes to the band "
                     "above), its input condition (the majority of its pixels' values, unrecorded on a tie or with "
                     "none), the class the map gives it (the majority of its pixels' classes) and its block of the "
                     "window grid, as --profile-by names them. Each combination that occurs is a cell; every cell with "
                     "fewer than --min-windows windows is folded into one profile named other.")
PROFILE_ALLOCATION_NOTE = ("labels are split equally across the {K} profiles, never from the model's confidence; a "
                           "profile too small for an equal share is labelled in full. Banding by confidence is safe "
                           "here: each band is judged on its own labels")
PROFILE_BLIND_NOTE = ("the rows run profile by profile, and the `stratum`, `condition`, `confidence` and `map_class` "
                      "columns show each window's profile: to label blind, hide them and show the windows in a shuffled "
                      "order, keeping the CSV's row order")


def _whole(text, field):
    t = text.strip()
    if not t.isdigit():
        raise ValueError(f"--profile-by: {field} takes a whole number, as {field}:3; got {field}:{text}")
    return int(t)


def parse_profile_by(text):
    """The fields of `--profile-by` as [(field, number or None)], in the canonical order FIELDS, so that
    "class,confidence:3" and "confidence:3,class" give the same partition and the same names. confidence:Q takes
    2 <= Q <= MAX_BANDS bands, grid:G 2 <= G <= MAX_GRID blocks a side; condition and class take no number. Each field
    at most once, at least one. Refusals are ValueErrors."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("--profile-by names at least one field: confidence:Q, condition, class or grid:G (for example "
                         "confidence:3,class)")
    seen = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"--profile-by {text!r} holds an empty field; separate the fields with single commas")
        name, sep, num = item.partition(":")
        name = name.strip()
        if name not in FIELDS:
            raise ValueError(f"--profile-by: unknown field {name!r}; the fields are confidence:Q (Q bands, 2 to "
                             f"{MAX_BANDS}), condition, class and grid:G (G by G blocks, 2 to {MAX_GRID})")
        if name in seen:
            raise ValueError(f"--profile-by names {name} twice")
        if name in ("confidence", "grid"):
            top = MAX_BANDS if name == "confidence" else MAX_GRID
            if not sep:
                raise ValueError(f"--profile-by: {name} takes a number, {name}:2 to {name}:{top}")
            q = _whole(num, name)
            if not 2 <= q <= top:
                raise ValueError(f"--profile-by: {name} takes 2 to {top}, got {name}:{q}")
        else:
            if sep:
                raise ValueError(f"--profile-by: {name} takes no number, got {item!r}")
            q = None
        seen[name] = q
    return [(f, seen[f]) for f in FIELDS if f in seen]


def profile_spec(fields):
    """The canonical `--profile-by` text of parsed fields: "confidence:3,condition,class"."""
    return ",".join(f if q is None else f"{f}:{q}" for f, q in fields)


def build_profiles(margin, valid, fields, condition=None, condition_names=None, map_class=None,
                   min_windows=MIN_PROFILE_WINDOWS, max_profiles=MAX_PROFILES):
    """The profile of every window of the map, fixed from the map alone: no label is read.

    margin    : (h, w) confidence per window, higher = more confident (assess's `arrays["confidence"]`)
    valid     : (h, w) windows that exist; the population is valid & finite margin, as in sample_for_estimation
    fields    : [(field, number or None)], as parse_profile_by gives them
    condition : (h, w) input condition per window, negative where none is recorded (pool_condition's grid); needed
                when the fields name condition. Unrecorded windows form the value "unrecorded", placed last
    condition_names : optional {value: name} for the condition values
    map_class : (h, w) class per window (assess's `arrays["pooled_argmax"]`); needed when the fields name class

    The cells are the distinct combinations of the fields' values among the population windows, sorted with the
    fields in canonical order; every cell with fewer than `min_windows` windows is folded into one profile, "other",
    placed last, and the other cells are profiles 0..K-1 in their sorted order. More than `max_profiles` profiles is
    refused: the user fixes the partition, and no further folding is done than the stated rule.

    Returns {"index": int64 profile per window of the flattened grid, -1 outside the population, "names", "sizes",
    "block": the sidecar's `profile` block, "notes"}."""
    margin = np.asarray(margin, dtype=np.float64)
    if margin.ndim != 2:
        raise ValueError(f"profiles are built on the window grid: margin must be (h, w), got shape {margin.shape}")
    h, w = margin.shape
    m = margin.ravel()
    valid = np.ones(m.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    if valid.size != m.size:
        raise ValueError(f"valid has {valid.size} entries for {m.size} windows")
    fields = [(str(f), None if q is None else int(q)) for f, q in fields]
    names_f = [f for f, _ in fields]
    if not fields or len(set(names_f)) != len(names_f) or any(f not in FIELDS for f in names_f):
        raise ValueError(f"fields must name each of {', '.join(FIELDS)} at most once, at least one; got {names_f}")
    if [f for f in FIELDS if f in names_f] != names_f:
        raise ValueError(f"fields must be in the canonical order {', '.join(FIELDS)} (parse_profile_by gives it)")
    for f, q in fields:
        top = MAX_BANDS if f == "confidence" else MAX_GRID if f == "grid" else None
        if (top is None) != (q is None) or (top is not None and not 2 <= q <= top):
            raise ValueError(f"the field {f} takes " + ("no number" if top is None else f"2 to {top}") + f", got {q}")
    if isinstance(min_windows, (bool, np.bool_)) or int(min_windows) != min_windows or int(min_windows) < 1:
        raise ValueError(f"min_windows is a whole number of at least 1, got {min_windows!r}")
    if isinstance(max_profiles, (bool, np.bool_)) or int(max_profiles) != max_profiles or int(max_profiles) < 1:
        raise ValueError(f"max_profiles is a whole number of at least 1, got {max_profiles!r}")
    min_windows, max_profiles = int(min_windows), int(max_profiles)
    pop = np.flatnonzero(valid & np.isfinite(m))
    N = int(pop.size)
    if N == 0:
        raise ValueError("no valid windows to build profiles from")
    keys, parts, blocks, notes = [], [], [], []
    for f, q in fields:
        if f == "confidence":
            mp = m[pop]
            # confidence_strata's bands, with the cuts kept for the sidecar: band 0 is the least confident, and a window
            # tied at a cut goes to the band above, so a band can be empty
            cuts = np.quantile(mp, np.linspace(0, 1, q + 1)[1:-1])
            keys.append(np.searchsorted(cuts, mp, side="right").astype(np.int64))
            parts.append(lambda b, q=q: f"confidence {int(b) + 1}/{q}")
            blocks.append({"field": "confidence", "bands": q, "cuts": [float(c) for c in cuts]})
        elif f == "condition":
            if condition is None:
                raise ValueError("the condition field needs the input condition of every window (--condition)")
            c_idx, c_values, c_names, _, c_notes = _condition_index(condition, pop, m.size, condition_names)
            keys.append(np.asarray(c_idx, np.int64))
            parts.append(lambda c, c_names=c_names: f"condition {c_names[int(c)]}")
            blocks.append({"field": "condition", "values": c_values, "names": c_names})
            notes += c_notes
        elif f == "class":
            if map_class is None:
                raise ValueError("the class field needs the class the map gives every window")
            mc = np.asarray(map_class).ravel()
            if mc.size != m.size:
                raise ValueError(f"map_class has {mc.size} entries for {m.size} windows")
            cp = mc[pop]
            if cp.dtype.kind == "f" and not (np.isfinite(cp).all() and np.all(np.mod(cp, 1) == 0)):
                raise ValueError("map_class must hold a whole class per valid window")
            if cp.dtype.kind not in "iuf":
                raise ValueError(f"map_class must hold a whole class per window, got dtype {cp.dtype}")
            keys.append(cp.astype(np.int64))
            parts.append(lambda c: f"class {int(c)}")
            blocks.append({"field": "class"})
        else:                                                   # grid: G by G blocks of the window grid
            rows, cols = np.divmod(pop, w)
            r, c = rows * q // h, cols * q // w
            keys.append((r * q + c).astype(np.int64))
            parts.append(lambda rc, q=q: f"cell {int(rc) // q + 1},{int(rc) % q + 1}")
            blocks.append({"field": "grid", "cells": q})
    stack = np.stack(keys, axis=1)
    cells, inv, counts = np.unique(stack, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()                               # its shape differs between numpy releases
    small = counts < min_windows
    kept = np.flatnonzero(~small)
    K = int(kept.size) + int(small.any())
    if K > max_profiles:
        raise ValueError(f"the partition gives {K} profiles after folding the cells under {min_windows} windows into "
                         f"'{OTHER}'; at most {max_profiles}: use fewer fields or bands, or raise --min-windows")
    profile_of_cell = np.full(cells.shape[0], K - 1, dtype=np.int64)      # folded cells form the last profile
    profile_of_cell[kept] = np.arange(kept.size)
    index = np.full(m.size, -1, dtype=np.int64)
    index[pop] = profile_of_cell[inv]
    names = [", ".join(part(v) for part, v in zip(parts, cells[j])) for j in kept] + ([OTHER] if small.any() else [])
    if len(set(names)) != len(names) or OTHER in names[:kept.size] or UNRECORDED in names:
        dup = sorted({n for n in names if names.count(n) > 1} | ({OTHER, UNRECORDED} & set(names[:kept.size])))
        raise ValueError(f"two profiles would share the name {dup[0]!r}; rename the conditions (--condition-names)")
    sizes = np.bincount(index[pop], minlength=K)

    def key_of(j):
        key = {}
        for (f, q), v in zip(fields, cells[j]):
            if f == "confidence":
                key["confidence"] = int(v) + 1                  # 1-based, as in the name: 1 is the least confident
            elif f == "condition":
                key["condition"] = blocks[names_f.index("condition")]["names"][int(v)]
            elif f == "class":
                key["class"] = int(v)
            else:
                key["cell"] = [int(v) // q + 1, int(v) % q + 1]
        return key
    cell_rows = [{"key": key_of(j), "n_windows": int(counts[j]), "profile": names[int(profile_of_cell[j])]}
                 for j in range(cells.shape[0])]
    profiles = [{"name": names[i], "definition": key_of(int(j)), "n_windows": int(sizes[i]), "n_cells": 1}
                for i, j in enumerate(kept)]
    if small.any():
        profiles.append({"name": OTHER, "definition": OTHER, "n_windows": int(sizes[K - 1]),
                         "n_cells": int(small.sum())})
        notes.append(f"{int(small.sum())} cell{'s' if small.sum() != 1 else ''} with fewer than {min_windows} windows "
                     f"({int(counts[small].sum())} windows in all) {'are' if small.sum() != 1 else 'is'} folded into "
                     f"one profile named {OTHER}; its verdict is about their windows taken together")
    if K == 1:
        notes.append("one profile covers the whole population: a simple random sample of the map")
    block = {"by": profile_spec(fields), "fields": blocks, "min_windows": min_windows, "max_profiles": max_profiles,
             "rule": PROFILE_RULE_TEXT, "cells": cell_rows, "profiles": profiles, "n_folded_cells": int(small.sum())}
    return {"index": index, "names": names, "sizes": [int(s) for s in sizes], "block": block, "notes": notes}

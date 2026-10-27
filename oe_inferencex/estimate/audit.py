"""The per-class audit: how often each predicted class's confident calls are wrong, from labels drawn class by class."""
import numpy as np

from oe_inferencex.common.hypergeom import _level, hypergeom_interval
from oe_inferencex.common.readings import _also_accept_pairs
from oe_inferencex.estimate.designs import draw_stratified


# ----------------------------------------------------------------------------- which classes' confident calls are wrong
# A per-class audit of a map's confident calls (13 October 2026, the owner's request). When an AWF fine-tune moved to
# East Africa's TimeSync plots, most of its confident errors sat in one predicted class, at a boundary the reference
# legend draws elsewhere (exp109, report-only; at fresh plots, in the blind test exp111, the pair again held about half
# or more of them, depending on the weighting). A small random sample from each predicted class's confident zone can find such a class (at exp111's fresh
# plots, 20 labels per class would have flagged it with probability 0.29), and reading the same labels with the
# contested pair accepted says whether its errors lie on that pair. The zone is fixed from the map before any label.
# Given the map, each class's zone is a fixed finite population and its labels a simple random sample of fixed size, so
# `hypergeom_interval` covers that zone's error rate at least 95% of the time, class by class (tests/test_class_audit.py
# enumerates it).
AUDIT_SHARE = 1 / 3                  # the confident zone by default: the top third of the valid windows by confidence
AUDIT_ZONE_TEXT = ("valid windows whose confidence (assess's per-window confidence, the CSV's `confidence` column) is "
                   "at or above the cut; with a share S, the cut is the confidence of the ceil(S N)-th most confident of "
                   "the N valid windows, so windows tied at the cut are in the zone. Fixed from the map alone, before "
                   "any label")
AUDIT_METHOD = ("labels on a simple random sample of each predicted class's confident zone; each class's zone error rate "
                "gets the exact hypergeometric interval over the zone's windows; a class labelled in full gets its "
                "exact rate")
AUDIT_NOTE = ("Each class's interval is its own 95% statement about the error rate of that class's confident zone; "
              "the intervals do not hold jointly, and a class picked out because its rate looks highest is not "
              "covered at 95% as the one picked. The audit names classes, not windows. Labels are assumed right; a "
              "window marked ? counts as right at the lower end and as wrong at the upper end.")
# The second reading is a second test on the same labels (decide's rule, above interval_second_look), and its guarantee
# rests on the crosswalk being fixed first. The note said each reading was exact with no condition; the review of 13
# October 2026 wrote the crosswalk after 15 labels, listing the reference classes the errors fell on, and the
# alternative interval missed its own error rate on all 2,000 draws while the legend note stood on model errors.
AUDIT_READING_NOTE = ("The alternative reading also accepts, for each class, the reference classes listed in the "
                      "--also-accept file, and reads the same labels a second time: a second test on them. If the file "
                      "was fixed before any label was read, each reading's interval is exact for its own definition of "
                      "an error, and a class's two intervals hold together at least 90% of the time (a union bound). A "
                      "file written after the labels were seen can be fitted to them, and then neither holds: listing "
                      "the classes a class's errors fell on clears it, and its note on a legend difference can then "
                      "stand on model errors.")


def _audit_zone(margin, valid, share=None, cut=None):
    """The confident zone of an audit, from the map alone: (the zone's indices into the flattened grid, ascending; the
    block the sidecar records). With `cut`, every valid window at confidence `cut` or above. Otherwise the top `share`
    (default AUDIT_SHARE) of the N valid windows: the cut t is the confidence of the m-th most confident, m = ceil(share
    N) in exact fractions with the share as written and at least 1, and the zone is every window at t or above, so
    windows tied at the cut are in it. A window with no finite confidence is not valid."""
    import math
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = np.asarray(valid, bool).ravel() & np.isfinite(margin)
    pop = np.flatnonzero(valid)
    if pop.size == 0:
        raise ValueError("no valid windows to sample")
    if share is not None and cut is not None:
        raise ValueError("the confident zone is a share of the windows or a cut, not both")
    conf = margin[pop]
    if cut is None:
        s = AUDIT_SHARE if share is None else share
        if isinstance(s, (bool, np.bool_)) or not 0 < float(s) <= 1:
            raise ValueError(f"the confident zone's share must be in (0, 1], got {s!r}")
        # in exact fractions: ceil(0.3 * 10) in floats is 4, since 0.3 * 10 is 3.0000000000000004
        m = max(1, math.ceil(_level(s) * pop.size))
        t = float(np.partition(conf, conf.size - m)[conf.size - m])          # the m-th largest
        block = {"rule": "share", "share": float(s), "n_top": int(m)}
    else:
        if isinstance(cut, (bool, np.bool_)) or not np.isfinite(float(cut)):
            raise ValueError(f"the confident zone's cut must be a finite confidence, got {cut!r}")
        t = float(cut)
        if not (conf >= t).any():
            raise ValueError(f"no valid window reaches confidence {t:g}; the most confident has {conf.max():.6g}")
        block = {"rule": "cut", "share": None, "n_top": None}
    zone = pop[conf >= t]
    block.update({"cut": t, "n_zone": int(zone.size), "share_of_map": zone.size / pop.size,
                  "definition": AUDIT_ZONE_TEXT})
    return zone, block


def sample_class_audit(margin, map_class, per_class, valid=None, share=None, cut=None, seed=0):
    """Which windows to label so that `class_audit_from_sample` can say how often each predicted class's confident calls
    are wrong: a simple random sample of `per_class` windows from each class's confident zone (`_audit_zone`), all of it
    when the zone holds no more, in one random order.

    margin    : per-window confidence, higher = more confident (assess's `arrays["confidence"]`, flattened)
    map_class : the map's class of every window (assess's `arrays["pooled_argmax"]`, flattened): the window's majority
                class, which `estimate` and `certify` grade
    per_class : the windows labelled per predicted class, a whole number of at least 1
    valid     : optional mask of windows that exist; a window with no finite confidence is not valid either
    share, cut: the confident zone, the top `share` of the valid windows (default AUDIT_SHARE) or every valid window at
                confidence `cut` or above; not both
    Returns the sample: its `indices` into the flattened grid and the class of each (`map_class`), the `zone`, each
    audited class's zone size and labels (`audited`, keyed by the class code as a string; not `classes`, which a
    product's sidecar already gives the class map's path) and the map's classes with no window in the zone
    (`not_audited`). No label enters it: the zone, the classes and the draw are fixed by the map and the seed."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    klass = np.asarray(map_class).ravel()
    if klass.size != margin.size:
        raise ValueError(f"map_class has {klass.size} entries for {margin.size} windows")
    valid = np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()
    if valid.size != margin.size:
        raise ValueError(f"valid has {valid.size} entries for {margin.size} windows")
    if isinstance(per_class, (bool, np.bool_)) or float(per_class) != int(per_class) or int(per_class) < 1:
        raise ValueError(f"per_class is the windows labelled per predicted class, a whole number of at least 1; got "
                         f"{per_class!r}")
    per_class = int(per_class)
    valid = valid & np.isfinite(margin)
    kv = klass[valid].astype(np.float64)
    if kv.size and not (np.isfinite(kv).all() and np.array_equal(kv, np.rint(kv)) and kv.min() >= 0):
        raise ValueError("map_class must hold a class code, a whole number of at least 0, at every valid window")
    zone, block = _audit_zone(margin, valid, share, cut)
    cls = np.full(margin.size, -1, dtype=np.int64)
    cls[valid] = np.rint(kv).astype(np.int64)
    codes, sizes = np.unique(cls[zone], return_counts=True)
    alloc = np.minimum(sizes, per_class)
    rng = np.random.default_rng(seed)
    idx = zone[draw_stratified(rng, np.searchsorted(codes, cls[zone]), sizes, alloc)]
    # one random order: drawn class by class, the rows came in runs of one class, which a reviewer labelling blind
    # would see
    idx = idx[rng.permutation(idx.size)]
    on_map, n_map = np.unique(cls[valid], return_counts=True)
    audited = set(codes.tolist())
    return {"design": "audit", "indices": idx, "map_class": cls[idx], "labels_per_class": per_class,
            "n_population": int(valid.sum()), "seed": seed, "zone": block,
            "audited": {str(int(c)): {"n_zone": int(Nc), "n_labelled": int(nc), "census": bool(nc == Nc)}
                        for c, Nc, nc in zip(codes, sizes, alloc)},
            "not_audited": {str(int(c)): {"n_windows": int(n)} for c, n in zip(on_map, n_map) if int(c) not in audited}}


def interval_verdict(low, high, alpha):
    """"below" when the interval lies below alpha, "above" when it lies at or above it, "undetermined" when it lies
    across it: decide's error_rate_below rule (yes, no, undetermined), named for a rate. Each end of an equal-tailed 95%
    interval is a 97.5% one-sided bound, so on its own each verdict is wrong at most 2.5% of the time."""
    if isinstance(alpha, (bool, np.bool_)) or not 0 < float(alpha) < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return "below" if high < alpha else "above" if low >= alpha else "undetermined"


def class_audit_from_sample(sample, reference, unjudged=None, also_accept=None, alpha=None, conf=0.95):
    """How often each predicted class's confident calls are wrong, from a labelled `sample_class_audit`.

    reference   : the class the reviewer saw in each sampled window, in the order of sample["indices"]; ignored where
                  `unjudged`
    unjudged    : True where the reviewer could not judge the window (`?` in the CSV)
    also_accept : optional {predicted class: [reference classes]}, a second reading of the same labels in which a window
                  of class c is also right when its reference class is listed for c (a class listed for itself changes
                  nothing; a key that is no audited class is returned in `also_accept_unused`)
    alpha       : optional rate; each class then gets a verdict against it (`interval_verdict`), and a class that is
                  above alpha under the strict reading and not under the alternative a note on the pairs its errors
                  lie on
    For each audited class c, with N_c windows in its zone, n_c of them labelled, k_c judged wrong (reference not c) and
    u_c marked ?, the interval runs from the lower end of `hypergeom_interval(k_c, n_c, N_c, conf)` to the upper end of
    `hypergeom_interval(k_c + u_c, n_c, N_c, conf)`, the bounds `estimate_error_rate` gives a condition: the interval
    moves one way with the count, so it covers the zone's error rate at least `conf` of the time whatever made the ?
    windows hard to judge. A class labelled in full gets its exact rate, or the range its ? windows leave. The classes'
    intervals do not hold jointly (AUDIT_NOTE). The alternative reading is a second test on the same labels: with
    `also_accept` fixed before any label was read, its intervals are exact for its own definition of an error and a
    class's two hold together at least 1 - 2 (1 - conf) of the time; fitted to the labels, neither holds
    (AUDIT_READING_NOTE)."""
    if not 0 < conf < 1:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    if alpha is not None:
        interval_verdict(0.0, 1.0, alpha)                   # refuses an alpha outside (0, 1)
        alpha = float(alpha)
    idx = np.asarray(sample["indices"]).ravel()
    mc = np.asarray(sample["map_class"]).ravel()
    ref = np.asarray(reference).ravel()
    unj = np.zeros(idx.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if not mc.size == ref.size == unj.size == idx.size:
        raise ValueError(f"{idx.size} sampled windows but {mc.size}, {ref.size} and {unj.size} classes, references and "
                         "flags")
    if (idx < 0).any() or np.unique(idx).size != idx.size:
        raise ValueError("an index repeats or is negative: the sample is the windows sample_class_audit drew, each once")
    classes, n_per, N_zone = sample["audited"], int(sample["labels_per_class"]), int(sample["zone"]["n_zone"])
    pairs = None if also_accept is None else _also_accept_pairs(also_accept)
    stray = sorted({int(c) for c in mc.tolist()} - {int(c) for c in classes})
    if stray:
        raise ValueError(f"a sampled window has class {stray[0]}, which the sample does not audit")
    judged = ~unj

    def reading(c, Nc, m, accepted):
        right = ref == c
        if accepted:
            right = right | np.isin(ref, accepted)
        wrong = m & judged & ~right
        n, k, u = int(m.sum()), int(wrong.sum()), int((m & unj).sum())
        by = {str(int(v)): int(cnt) for v, cnt in zip(*np.unique(ref[wrong], return_counts=True))}
        lo, hi = hypergeom_interval(k, n, Nc, conf)[0], hypergeom_interval(k + u, n, Nc, conf)[1]
        r = {"n_wrong": k, "n_unjudged": u, "errors_by_reference": by, "estimate": None if u else k / n}
        if u:
            r["estimate_range"] = [k / n, (k + u) / n]
        r.update({"low": lo, "high": hi, "half_width": (hi - lo) / 2})
        if alpha is not None:
            r["verdict"] = interval_verdict(lo, hi, alpha)
        return r

    per = {}
    for key in sorted(classes, key=int):
        c, e = int(key), classes[key]
        Nc, n_c = int(e["n_zone"]), int(e["n_labelled"])
        m = mc == c
        if n_c != min(n_per, Nc) or int(m.sum()) != n_c:
            raise ValueError(f"class {c} has {int(m.sum())} labelled rows; the sample drew {n_c}, and its design "
                             f"{min(n_per, Nc)} (min of {n_per} per class and the {Nc} windows of its zone)")
        census = n_c == Nc
        row = {"map_class": c, "n_zone": Nc, "share_of_zone": Nc / N_zone, "n_labelled": n_c, "census": census}
        strict = reading(c, Nc, m, [])
        verdict = strict.pop("verdict", None)
        row.update(strict)
        row["method"] = ("the class's whole confident zone is labelled: its exact rate, or the range its ? windows leave"
                         if census else "exact hypergeometric interval over the class's confident zone, of which its "
                                        "labels are a simple random sample")
        if verdict is not None:
            row["verdict"] = verdict
        if pairs is not None:
            accepted = [a for a in pairs.get(c, []) if a != c]
            alt = reading(c, Nc, m, accepted)
            alt.pop("n_unjudged")
            alt.pop("half_width")
            row["also_accept"] = {"accepted": accepted, **alt}
            if verdict == "above" and alt["verdict"] != "above":
                # the errors the alternative reading accepts, by reference class; at least one, since with none the
                # two readings count the same errors and give the same interval
                on = {v: n for v, n in strict["errors_by_reference"].items() if int(v) in accepted}
                row["note"] = (f"under the strict reading class {c}'s confident zone is wrong at least {100 * alpha:g}% "
                               f"of the time; under the alternative it is not shown to be: {sum(on.values())} of its "
                               f"{strict['n_wrong']} errors are on reference classes the alternative accepts ("
                               + ", ".join(f"{v}: {n}" for v, n in on.items()) + "). Those windows are errors only "
                               "if the reference legend's boundary is the one that holds, and the labels cannot say "
                               "whose boundary that is: a legend difference, not necessarily a model error")
        per[str(c)] = row
    out = {"design": "audit", "n_population": int(sample["n_population"]), "zone": dict(sample["zone"]),
           "labels_per_class": n_per, "n_labelled": int(idx.size), "n_unjudged": int(unj.sum()),
           "nominal_coverage": conf, "alpha": alpha, "method": AUDIT_METHOD, "note": AUDIT_NOTE, "per_class": per,
           "not_audited": dict(sample.get("not_audited") or {})}
    if pairs is not None:
        out.update({"also_accept": {"source": None, "pairs": {str(c): v for c, v in sorted(pairs.items())}},
                    "also_accept_unused": [str(c) for c in sorted(pairs) if str(c) not in classes],
                    "reading_note": AUDIT_READING_NOTE})
    return out

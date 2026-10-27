"""Localized certification: each profile of the map, fixed before any label (estimate.profiles), called good, bad or
undecided against alpha, all the verdicts together right except on at most delta of samples.

A profile labelled in part (0 < n < N) gets the exact equal-tailed hypergeometric interval for its error rate at level
1 - delta/L, L the profiles labelled in part (Bonferroni: the L intervals hold together on at least 1 - delta of
samples, the meaning delta has for the certified zone). Its lower end counts the windows the reviewer could not judge
(`?`) as right and its upper end counts them wrong. The profile is good when the upper end is at most alpha, bad when
the lower end is above alpha, and undecided otherwise. A wrong good needs the interval to miss above and a wrong bad to
miss below, so the verdicts alone are wrong on at most delta/2 of samples. A profile labelled in full has its exact
rate (or the range its `?` windows leave) and needs none of delta; one with no label is undecided.
tests/test_localized.py checks it by exact enumeration over every error pattern and every draw of small maps.
"""
from fractions import Fraction

import numpy as np

from oe_inferencex.common.hypergeom import _float_above, _float_below, _hypergeom_fractions, _level
from oe_inferencex.common.sampling import _condition_guards
from oe_inferencex.certify.zone import ZONE_DELTA

VERDICT_CODES = {"outside the population": 0, "good": 1, "bad": 2, "undecided": 3}     # the verdict raster's codes
PROFILE_METHOD = ("per profile, the exact hypergeometric interval for its error rate at 1 - delta/L, L the profiles "
                  "labelled in part, its lower end counting ? windows right and its upper end wrong; good where the "
                  "upper end is at most alpha, bad where the lower end is above alpha, undecided otherwise, compared "
                  "exactly, with alpha as written; a profile labelled in full compares its exact rate")
PROFILE_NOTE = ("Each verdict is about a profile's windows taken together, not each window. Delta {delta:g} is split "
                "over the {L} {profiles} labelled in part, {d:.3g} each, so their intervals hold together, and every "
                "verdict is right, except on at most {delta:g} of samples (the verdicts alone, on at most {half:g}); a "
                "profile labelled in full has its exact rate and needs none of delta. Good means wrong at most "
                "{alpha:g} of the time, bad more than {alpha:g}; undecided says nothing either way.")
PROFILE_CENSUS_NOTE = ("every profile is labelled in full: each verdict compares its exact rate (or the range its ? "
                       "windows leave) with alpha {alpha:g}, and none can be wrong; delta is not used.")
OTHER_NOTE = ("'other' holds the {c} cells with fewer than {m} windows, folded together by the rule fixed with the "
              "sample, before any label. Its verdict is about their windows taken together; it says nothing about any "
              "one of those cells.")
UNJUDGED_PROFILE_NOTE = ("{u} of the {n} labelled windows could not be judged (?). Each interval's lower end counts them "
                         "right and its upper end wrong, so every verdict holds whatever made them hard to judge; judging "
                         "them can only settle more profiles.")
PROFILE_ISSUED_NOTE = ("delta is the share of all samples on which any verdict is wrong. Among the samples that call a "
                       "given profile good, the share wrong can be larger, most of all when its true error rate is just "
                       "above alpha.")
PROFILE_RERUN_NOTE = ("The verdicts assume the reviewer's labels are right and that alpha and delta were chosen before "
                      "any label was read. Running certify again on these labels at another alpha or delta, or after "
                      "adding labels, is a new test, and the chance that some run is wrong grows with the runs.")


def _count(name, x):
    if isinstance(x, (bool, np.bool_)) or float(x) != int(x):
        raise ValueError(f"profile_interval takes whole counts, got {name}={x!r}")
    return int(x)


def profile_interval(k, u, n, N, conf):
    """The exact ends, as Fractions, of a profile's interval: the lower end of the exact equal-tailed hypergeometric
    interval at k errors (the `?` windows counted right), the upper end at k + u (counted wrong), each at level `conf`,
    n labelled of N windows. `conf` is taken as written (a Fraction is exact: 1 - delta/L, so each tail is exactly
    delta/(2L)). No label: (0, 1). A profile labelled in full: (k/n, (k + u)/n). Both ends rise with the error count,
    so the interval holds the one the true labels would give."""
    k, u, n, N = (_count(nm, x) for nm, x in (("k", k), ("u", u), ("n", n), ("N", N)))
    if k < 0 or u < 0 or k + u > n or n > N:
        raise ValueError(f"profile_interval needs 0 <= k, 0 <= u and k + u <= n <= N, got k={k}, u={u}, n={n}, N={N}")
    if isinstance(conf, (bool, np.bool_)):
        raise ValueError(f"conf must be in (0, 1), got {conf!r}")
    conf = _level(conf)
    if not 0 < conf < 1:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    if n == 0:
        return Fraction(0), Fraction(1)
    if n >= N:
        return Fraction(k, n), Fraction(k + u, n)
    return _hypergeom_fractions(k, n, N, conf)[0], _hypergeom_fractions(k + u, n, N, conf)[1]


def profile_verdict(low, high, alpha):
    """good when the upper end is at most alpha, bad when the lower end is above it, else undecided; alpha as written
    (0.05 is 1/20). Bad is the complement of good's claim, error at most alpha, so audit.interval_verdict's `above`
    (low >= alpha) is not used."""
    a = _level(alpha)
    return "good" if high <= a else "bad" if low > a else "undecided"


def _check_level(name, v):
    if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating, Fraction))
            or not 0 < float(v) < 1):
        raise ValueError(f"{name} must be in (0, 1), got {v!r}")


def _plural(n, one, many):
    return one if n == 1 else many


def profile_verdicts(sample, wrong, alpha, unjudged=None, delta=ZONE_DELTA):
    """Each profile of a sample stratified by profile called good, bad or undecided against alpha.

    sample   : what `sample_for_estimation` returned under the condition design with the profiles as its conditions
               (sample --design profile writes it to the sidecar); any fixed-allocation sample of the condition design
               is read the same way
    wrong    : 0/1 per labelled window, in the order of sample["indices"]; ignored where `unjudged`
    alpha    : the error rate a good profile may not exceed, chosen before any label is read
    unjudged : True per labelled window the reviewer could not judge (`?`)
    delta    : the share of samples on which some interval misses, split over the profiles labelled in part

    L, the profiles labelled in part, is fixed by the sizes and the allocation, before any label; each of them gets
    `profile_interval` at 1 - delta/L. Returns the verdicts per profile (`profiles`), the counts per verdict, the notes
    and `verdict_grid`, the verdict code of every window of the flattened grid (VERDICT_CODES; 0 outside the
    population), which the command pops and saves."""
    _check_level("alpha", alpha)
    _check_level("delta", delta)
    if sample.get("design") != "condition" or sample.get("condition_grid") is None:
        raise ValueError("per-profile verdicts need a sample stratified by profile (sample --design profile): a sample "
                         "of the condition design with its condition_grid")
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    idx = np.asarray(sample["indices"], int).ravel()
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} sampled windows")
    unk = np.zeros(wrong.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if unk.size != wrong.size:
        raise ValueError(f"{unk.size} unjudged flags for {wrong.size} labels")
    wrong = np.where(unk, 0.0, wrong)                            # a ? window's `wrong` is not read
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once in the sample; each window "
                         "is labelled once")
    if (idx < 0).any():
        raise ValueError("a sampled window index is negative; indices point into the flattened window grid")
    grid, _, names, sizes = _condition_guards(sample, idx)     # the allocation per profile is the design's
    K, N = len(names), int(sample["n_population"])
    cidx = grid[idx]
    n_h = np.bincount(cidx, minlength=K)
    u_h = np.bincount(cidx[unk], minlength=K)
    k_h = np.bincount(cidx[~unk], weights=wrong[~unk], minlength=K).round().astype(int)
    tested = [h for h in range(K) if 0 < int(n_h[h]) < int(sizes[h])]
    L = len(tested)
    conf = 1 - _level(delta) / L if L else None                 # exact: each tail at delta / (2 L)
    block = sample.get("profile") if isinstance(sample.get("profile"), dict) else {}
    definitions = {p.get("name"): p.get("definition") for p in block.get("profiles") or [] if isinstance(p, dict)}
    rows, codes = {}, np.zeros(grid.size, np.uint8)
    for h, name in enumerate(names):
        n, k, u, Nh = int(n_h[h]), int(k_h[h]), int(u_h[h]), int(sizes[h])
        is_tested = h in tested
        lo, hi = profile_interval(k, u, n, Nh, conf if is_tested else Fraction(1, 2))   # conf unread untested
        verdict = profile_verdict(lo, hi, alpha)
        row = {"index": h, "definition": definitions.get(name), "n_population": Nh, "share_of_map": Nh / N,
               "n_labelled": n, "n_wrong": k, "n_unjudged": u, "census": n >= Nh, "tested": is_tested,
               "estimate": k / n if n and not u else None}
        if n and u:
            row["estimate_range"] = [k / n, (k + u) / n]
        # outward to the float, so the JSON never shows an end past the exact one the verdict compared
        row.update({"low": _float_below(lo), "high": _float_above(hi), "verdict": verdict})
        if n == 0:
            row["note"] = "no labelled window fell in this profile; nothing can be said about it"
        rows[name] = row
        codes[grid == h] = VERDICT_CODES[verdict]
    counts = {}
    for v in ("good", "bad", "undecided"):
        inside = [r for r in rows.values() if r["verdict"] == v]
        windows = sum(r["n_population"] for r in inside)
        counts[v] = {"profiles": len(inside), "windows": windows, "share_of_map": windows / N}
    a, d = float(alpha), float(delta)
    notes = ([PROFILE_NOTE.format(delta=d, L=L, profiles=_plural(L, "profile", "profiles"), d=d / L, half=d / 2,
                                  alpha=a)] if L else [PROFILE_CENSUS_NOTE.format(alpha=a)])
    if "other" in rows and block:
        notes.append(OTHER_NOTE.format(c=int(block.get("n_folded_cells", 0)), m=block.get("min_windows")))
    u_all = int(unk.sum())
    if u_all:
        notes.append(UNJUDGED_PROFILE_NOTE.format(u=u_all, n=int(idx.size)))
    if L:
        notes.append(PROFILE_ISSUED_NOTE)
    notes.append(PROFILE_RERUN_NOTE)
    return {"by_profile": True, "alpha": a, "delta": d, "n_profiles": K, "n_tested": L,
            "delta_per_profile": d / L if L else None, "level": float(conf) if L else None,
            "n_population": N, "n_labelled": int(idx.size), "n_unjudged": u_all, "method": PROFILE_METHOD,
            "profiles": rows, "verdict_counts": counts, "verdict_codes": {str(c): v for v, c in VERDICT_CODES.items()},
            "notes": notes, "verdict_grid": codes}

"""How wrong the map is, from a labelled sample: estimate_error_rate, estimate_from_indices and their notes."""
import numpy as np

from oe_inferencex.common.hypergeom import _float_above, _float_below, _level, hypergeom_interval
from oe_inferencex.common.sampling import _condition_guards, review_set_check
from oe_inferencex.estimate.designs import MIN_PER_STRATUM
from oe_inferencex.estimate.intervals import (_union_interval, cluster_interval, stratified_interval_wilson,
                                              stratified_mean_and_variance, wilson_interval)
from oe_inferencex.estimate.valid import whole_map_valid

# The coverage this warning quotes for "this interval" is the shipped design's (exp/exp78_shipped_tiles.py, artifact
# exp/out/exp78_shipped_tiles.json); 1.3.1 quoted exp78's own design of exactly 18 tiles and a normal quantile.
# The research record's tests/test_estimate.py reads the numbers back from the artifact.
TILES_WARNING = ("labels taken tile by tile are not independent, and a map whose tiles differ in size is labelled "
                 "unevenly; the naive interval beside this one is what the ordinary formula says, and in exp78, with "
                 "18 tiles of 16 windows, it covered 51 to 78% of the time on six of seven tasks while claiming 95%. "
                 "This interval is better and still not honest everywhere: graded on exp78's tasks it covered 94.5 to "
                 "95.4% of the time on five, 84.3% on Sen1Floods11 and 68.5% on MADOS, where a tenth of the tiles hold "
                 "most of the errors. Prefer the confidence design")
# What a whole-map number does not say when part of the map was read from other inputs (exp88). The first two
# travel in the outputs of estimate_error_rate and certify_zone when no input condition is recorded; the others go
# with the per-condition results. The numbers are exp88's (exp/out/exp88_summary.json) and its matched-head
# follow-up's (exp/out/exp88_matched_head.json); the research record's tests/test_assess.py reads them back from there.
SCOPE_ESTIMATE = ("This is the error rate of the whole map. If part of the map was predicted from other inputs, that "
                  "part's rate can differ widely from it. On a PASTIS map with half its tiles read without the optical "
                  "input by a probe trained on radar plus optical, that half's rate was 74.1% against 19.7% on the "
                  "rest, while random samples of 300 estimated 46.9% on average. Read by a probe trained on radar "
                  "alone, that half was still 28.3% wrong against 19.7% (exp88). Draw the sample with --condition to "
                  "get each part's rate.")
CONDITION_NOTE = ("Each condition's interval is its own 95% statement; the intervals do not hold jointly at 95%. The "
                  "whole-map rate weights each condition by its share of the map and can hide a condition that is much "
                  "worse.")
# Under the condition design, what holds by construction and what is not graded. Each condition's own interval is
# exact: the tests enumerate its coverage, and under a random sample exp88 graded it (95.6% and 95.7% of 2,000 draws,
# the claim pooled-error-rate-misstates-the-cloudy-part). The whole-map interval adds up the conditions' exact
# intervals at 1 - 0.05 / L (`_union_interval`), so it covers at least 95% by construction; the tests enumerate that
# too. The per-class intervals under this design have not been graded.
CONDITION_WHOLE_MAP = ("Each condition's own interval is exact. The whole-map interval takes each condition's exact "
                       "interval at 1 - 0.05/L, where L is the number of conditions not labelled in full, and weights it "
                       "by the condition's share of the map. At that level the L intervals hold together at least 95% of "
                       "the time, so the whole-map interval covers at least 95% by construction. It is wider than the "
                       "usual stratified interval, which adds up the conditions' variances, would be. When the "
                       "conditions' error rates are close, it is also wider than "
                       "the exact interval of a random sample of the same size. What the design buys is each condition's "
                       "own rate: a small condition gets as many labels as a large one, or all its windows. If only the "
                       "whole-map rate is needed, --design random can give a narrower interval.")
CLASS_WHOLE_MAP = ("Each class's own interval is exact. The whole-map interval takes each class's exact interval at "
                   "1 - 0.05/L, where L is the number of classes not labelled in full, and weights it by the class's "
                   "share of the map, so it covers at least 95% by construction; it is wider than the usual stratified "
                   "interval. What the design buys is each class's own accuracy: every class gets at least its floor "
                   "of labels, or all its windows, and the rest go in proportion to class size (Olofsson et al. "
                   "2014). If only the whole-map rate is needed, --design random can give a narrower interval.")


def _per_condition(grid, values, names, sizes, idx, wrong, N, whole):
    """Each condition's error rate with its exact interval. Under the condition design n_c is fixed; under a random
    sample the labels that fall in c are a simple random sample of c given their count n_c. Either way
    `hypergeom_interval(k_c, n_c, N_c)` covers at least 95%."""
    cidx = grid[idx]
    method = "exact hypergeometric interval (the labels in the condition are a simple random sample of it)"
    per, outside = {}, []
    for c, (v, name, Nc) in enumerate(zip(values, names, sizes)):
        m = cidx == c
        n_c, k_c = int(m.sum()), int(wrong[m].sum())
        row = {"value": v, "n_population": int(Nc), "share_of_map": int(Nc) / N, "n_labelled": n_c, "n_wrong": k_c}
        lo, hi = hypergeom_interval(k_c, n_c, int(Nc))            # (0, 1) with no label
        row.update({"estimate": k_c / n_c if n_c else None, "low": lo, "high": hi, "half_width": (hi - lo) / 2,
                    "method": method})
        if n_c == 0:
            row["note"] = "no labelled window fell in this condition; nothing can be said about it"
        elif not lo <= whole <= hi:
            outside.append(name)
        per[name] = row
    return per, outside


# Windows the reviewer could not judge, and a reviewer who errs (the CLI defect of 1 October 2026: the command told a
# reviewer to leave out a window they could not judge, and refused every CSV that did). Both rest on one property: the
# exact intervals are monotone in the error count, and the stratified estimate in each stratum's, so an interval that
# covers at the worst case covers at every case in between.
UNJUDGED_NOTE = ("{u} of the {n} labelled windows could not be judged (`?`). The lower end counts every one of them as "
                 "right and the upper end every one as wrong, so the interval covers at least 95% whatever made them "
                 "hard to judge, and the estimate is a range for the same reason. Judging them narrows it.")
UNJUDGED_STRATIFIED_NOTE = ("Under this design any `?` also replaces the design's Wilson interval, which does not move one way "
                            "with the error count, by the union bound over its strata, which does and is much wider: most "
                            "of the width comes from that switch, not from the `?` windows themselves. Judging every "
                            "window restores the narrower interval; a random sample bounds `?` at the cost of each "
                            "window's own range.")
REVIEWER_NOTE = ("The interval is widened for a reviewer who marks at most {fa:.3g} of the truly correct windows wrong "
                 "(false alarms) and misses at most {miss:.3g} of the truly wrong ones: the lower end of the labels' "
                 "interval becomes (low - {fa:.3g}) / (1 - {fa:.3g}) and the upper end high / (1 - {miss:.3g}). It covers "
                 "the true rate whenever the labels' interval covers the rate the reviewer's labels would show on every "
                 "window and the reviewer errs no more often than that; the bounds are the user's, not measured here. "
                 "Per input condition they must hold within each condition.")


def _stratum_union(sample, wrong):
    """The stratified estimate and the union-bound interval over a confidence or proportional design's strata: each
    stratum's exact interval at 1 - 0.05 / L, weighted by its share of the map (`_union_interval`). Exact and monotone
    in each stratum's error count, so it bounds windows the reviewer could not judge, where the design's own Wilson
    interval on an effective sample size is neither. Returns (estimate, low, high, L)."""
    idx = np.asarray(sample["indices"], int)
    pop = np.asarray(sample["strata_of_population"], int)
    pos = {int(g): i for i, g in enumerate(pop)}
    local = np.array([pos[int(g)] for g in idx])
    strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
    n_h = np.bincount(strata[local], minlength=len(sizes))
    k_h = np.bincount(strata[local], weights=np.asarray(wrong, float), minlength=len(sizes)).round().astype(int)
    return _union_interval(k_h, n_h, sizes)


SEQUENTIAL_READ_NOTE = ("the first {n} windows of a sequential sample, read as a random sample of that size. The "
                        "interval holds when that number was fixed before labelling; one read at a stop chosen because "
                        "the labels looked good can be too narrow. certify on this sample holds at every look")


def estimate_error_rate(sample, wrong, unjudged=None, reviewer_false_alarm=0.0, reviewer_miss=0.0, valid=False):
    """The error rate of the whole map from the labelled sample, with the interval its design earns
    (`_estimate_core`), and, when asked, bounded for windows the reviewer could not judge and for a reviewer who errs.

    unjudged : True per labelled window the reviewer could not judge (`?` in the CSV), in the order of the indices;
        its `wrong` value is ignored. The interval's lower end counts these windows as right and its upper end as
        wrong; the exact intervals are monotone in the error count, so the interval covers at least 95% whatever made
        them hard to judge. Under the confidence and proportional designs the interval is then the union bound over
        their strata (`_stratum_union`), exact, in place of the design's Wilson interval, which is not monotone. The
        tiles design refuses unjudged windows.
    reviewer_false_alarm, reviewer_miss : bounds the user states, in [0, 1): the share of truly correct windows the
        reviewer marks wrong and of truly wrong windows the reviewer marks right. The labels show the rate
        p = theta (1 - e1) + (1 - theta) e0, so theta lies in [(p - E0) / (1 - E0), p / (1 - E1)], and those maps,
        increasing in p, are applied to the interval's ends (the identified set is sharp, proved in Lean in lean/ of
        the v1.9.0 tree, https://github.com/2imi9/olmoearth_inferenceX/tree/v1.9.0/lean; tests/test_unjudged.py
        checks the coverage by exact enumeration).
    Without these the result is `_estimate_core`'s, unchanged. With them `estimate_range` gives the range the labels
    allow, and `estimate` is None unless that range is one value; each condition's row is bounded the same way.
    Sharpness is proved in lean/ of the v1.9.0 tree, not tested here; tests/test_unjudged.py checks coverage and the
    formulas.
    valid : True adds `valid_interval` (1.20.0, `estimate --valid`): the interval valid at every sample size under
        every design but tiles (estimate/valid.py: `whole_map_valid`), with its own point, form and method, `?` and
        the reviewer's bounds read as above. Every other key is the result without it, unchanged, so what reads the
        result (decide, certify's notes, the MCP server) reads the same numbers. tests/test_estimate_valid.py checks
        its coverage by exact enumeration."""
    if valid:
        res = estimate_error_rate(sample, wrong, unjudged, reviewer_false_alarm, reviewer_miss)
        res["valid_interval"] = whole_map_valid(sample, wrong, unjudged, reviewer_false_alarm, reviewer_miss)
        return res
    if sample.get("design") == "sequential":
        # The labelled windows are the first of a random order: a simple random sample of their number, as the command
        # line reads it. The API refused a sample sample_for_estimation drew as an "unknown design" (the bug hunt of
        # 2026-10-25).
        n = int(np.asarray(sample["indices"]).size)
        res = estimate_error_rate(dict(sample, design="random", budget=n), wrong, unjudged, reviewer_false_alarm,
                                  reviewer_miss)
        res["sequential_note"] = SEQUENTIAL_READ_NOTE.format(n=n)
        return res
    fa, miss = float(reviewer_false_alarm), float(reviewer_miss)
    for name, v in (("reviewer_false_alarm", fa), ("reviewer_miss", miss)):
        if not 0 <= v < 1:
            raise ValueError(f"{name} must be at least 0 and below 1, got {v:g}")
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    unk = np.zeros(wrong.size, bool) if unjudged is None else np.asarray(unjudged, bool).ravel()
    if unk.size != wrong.size:
        raise ValueError(f"{unk.size} unjudged flags for {wrong.size} labels")
    u = int(unk.sum())
    if u == 0 and fa == 0 and miss == 0:
        return _estimate_core(sample, wrong)
    design = sample["design"]
    if u and design == "tiles":
        raise ValueError("windows that could not be judged are not supported under the tiles design: its cluster "
                         "interval is not monotone in the error count, so no worst case bounds them. Judge every "
                         "window, or draw a random sample")
    w_lo, w_hi = np.where(unk, 0.0, wrong), np.where(unk, 1.0, wrong)
    lo_res = _estimate_core(sample, w_lo)
    hi_res = _estimate_core(sample, w_hi) if u else lo_res
    out = dict(lo_res)

    # The maps run in exact fractions, with the bounds as written, and round outward. Until 2026-10-06 they ran in
    # floats, and an end equal to the true rate, the sharp case of a reviewer erring exactly at the bound, landed one
    # float past it: five windows, all labelled, four truly wrong and one of them missed at reviewer_miss 0.25 gave
    # (0.6, 0.7999999999999999) against a true rate of 0.8. An end x is the nearest float to an exact end that can lie
    # half a unit in its last place on either side, so the map starts from the float beyond x; 0 and 1 are exact.
    from fractions import Fraction
    import math
    E0, E1 = _level(fa), _level(miss)

    def down(x):                               # the lower end for a reviewer with false alarms
        if not fa:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, -math.inf))
        return max(0.0, _float_below((q - E0) / (1 - E0)))

    def up(x):                                 # the upper end for a reviewer who misses errors
        if not miss:
            return x
        q = Fraction(x if x in (0.0, 1.0) else math.nextafter(x, math.inf))
        return min(1.0, _float_above(q / (1 - E1)))

    if u and design in ("confidence", "proportional"):
        e_lo, low, _, L = _stratum_union(sample, w_lo)
        e_hi, _, high, _ = _stratum_union(sample, w_hi)
        for k in ("effective_n", "starved_strata", "warning"):     # they describe the Wilson interval, not this one
            out.pop(k, None)
        # L = 0 when every stratum is labelled in full: the interval is then the exact range of the rate, and until
        # 2026-10-06 the method printed "1 - 0.05/0" and the note blamed the width on a switch that adds none
        out.update({"strata_in_interval": L,
                    "method": (f"stratified by confidence margin; with windows that could not be judged, the sum of each "
                               f"stratum's exact interval at 1 - 0.05/{L}, weighted by its share of the map (union "
                               "bound), which covers at least 95% by construction") if L else
                              ("stratified by confidence margin; every window is labelled, so with windows that could not "
                               "be judged the interval is the exact range of the rate, counting them all right at its "
                               "lower end and all wrong at its upper end")})
    else:
        e_lo, e_hi, low, high = lo_res["estimate"], hi_res["estimate"], lo_res["low"], hi_res["high"]
    if fa or miss:
        out["labels_interval"] = {"low": low, "high": high}
    out.update({"estimate": None, "estimate_range": [down(e_lo), up(e_hi)], "low": down(low), "high": up(high),
                "n_unjudged": u, "reviewer_false_alarm": fa, "reviewer_miss": miss})
    if out["estimate_range"][0] == out["estimate_range"][1]:      # nothing to bound: the value, not "x% to x%"
        out["estimate"] = out["estimate_range"][0]
    out["half_width"] = (out["high"] - out["low"]) / 2
    notes = ([UNJUDGED_NOTE.format(u=u, n=int(wrong.size))] if u else []) + (
        [UNJUDGED_STRATIFIED_NOTE] if u and design in ("confidence", "proportional") and out["strata_in_interval"] else []) + (
        [REVIEWER_NOTE.format(fa=fa, miss=miss)] if fa or miss else [])
    out["bounds_note"] = " ".join(notes)
    if lo_res.get("by_condition"):
        per, outside = {}, []
        r_lo, r_hi = out["estimate_range"]
        for name, row_lo in lo_res["per_condition"].items():
            row_hi = hi_res["per_condition"][name]
            row = dict(row_lo)
            if row_lo["n_labelled"]:
                row.update({"estimate": None, "estimate_range": [down(row_lo["estimate"]), up(row_hi["estimate"])],
                            "low": down(row_lo["low"]), "high": up(row_hi["high"]),
                            "n_unjudged": int(row_hi["n_wrong"] - row_lo["n_wrong"])})
                row["half_width"] = (row["high"] - row["low"]) / 2
                if row["estimate_range"][0] == row["estimate_range"][1]:
                    row["estimate"] = row["estimate_range"][0]
                if row["low"] > r_hi or row["high"] < r_lo:
                    outside.append(name)
            per[name] = row
        out.update({"per_condition": per, "outside_condition_intervals": outside})
    return out


def _estimate_core(sample, wrong):
    """The error rate of the whole map from the labelled sample, with the interval its design earns.

    sample : what `sample_for_estimation` returned
    wrong  : 0/1 per labelled window, in the order of sample["indices"]: 1 where the label disagrees with the map

    A sample that records an input condition (`condition_grid`) also gets each condition's rate with its exact
    interval (`per_condition`); one that does not gets `scope`, what a whole-map rate does not say (exp88). Under the
    condition design the whole-map estimate is the stratified one, sum_c W_c k_c / n_c. Its interval weights each
    condition's exact interval at 1 - 0.05 / L by the condition's share, where L is the number of conditions not
    labelled in full (`conditions_in_interval`), and so covers at least 95% by construction; it is wider than a
    stratified interval would be and, when the conditions' error rates are close, wider than a random sample's
    exact interval of the same size; it can be narrower when they are far apart (tests/test_estimate_exact.py,
    expected widths computed exactly). `design_variance` is the stratified estimate's unbiased variance, for
    information; the interval does not use it. `condition_note` says what each interval is.
    """
    wrong = np.asarray(wrong, dtype=np.float64).ravel()
    idx = np.asarray(sample["indices"], int)
    if wrong.size != idx.size:
        raise ValueError(f"{wrong.size} labels for {idx.size} sampled windows")
    if not np.isin(wrong, (0.0, 1.0)).all():
        raise ValueError("wrong must be 0 or 1 per window")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{idx.size - np.unique(idx).size} window(s) appear more than once in the sample; each window "
                         "is labelled once, and a repeated one would be counted as extra units of its stratum")
    if (idx < 0).any():
        raise ValueError("a sampled window index is negative; indices point into the flattened window grid")
    N, design = int(sample["n_population"]), sample["design"]
    out = {"design": design, "n_labelled": int(idx.size), "n_population": N, "nominal_coverage": 0.95}
    by_condition = sample.get("condition_grid") is not None
    if design == "condition" and not by_condition:
        raise ValueError('a sample of the "condition" design carries its condition_grid, and this one has none')
    if by_condition:
        if design not in ("random", "condition"):
            raise ValueError(f"an input condition is recorded only with the random and condition designs; this sample "
                             f"was drawn with the {design!r} design")
        grid, c_values, c_names, c_sizes = _condition_guards(sample, idx)
    if design == "condition":
        pop = np.asarray(sample["strata_of_population"], int)
        pos = {int(g): i for i, g in enumerate(pop)}
        stray = [int(g) for g in idx if int(g) not in pos]
        if stray:
            raise ValueError(f"{len(stray)} sampled window(s), first {stray[0]}, are not in the population this sample "
                             "was drawn from (an invalid window, or a sample built against another mask)")
        local = np.array([pos[int(g)] for g in idx])
        err = np.zeros(pop.size)
        err[local] = wrong
        strata, sizes = np.asarray(sample["strata"]), list(sample["sizes"])
        # v, the unbiased variance of the stratified estimate with the conditions as strata, is kept as
        # `design_variance` and does not enter the interval
        var = stratified_mean_and_variance(err, strata, local, sizes, N)[1]
        n_h = np.bincount(strata[local], minlength=len(sizes))
        k_h = np.bincount(strata[local], weights=wrong, minlength=len(sizes)).round().astype(int)
        # a condition labelled in full (a one-window condition, say) has no sampling error and is not starved
        starved = int(((n_h < MIN_PER_STRATUM) & (n_h < np.asarray(sizes))).sum())
        # theta = sum_c W_c k_c / n_c, W_c = N_c / N, the stratified estimate; its interval weights each condition's
        # exact interval at 1 - 0.05 / L by its share: at least 95% by construction. It replaced spec 2.7's stratified
        # Wilson interval and 523efde's variance floor (review of 2026-09-29; the reason is in `_union_interval`)
        est_, lo, hi, L = _union_interval(k_h, n_h, sizes)
        if len(c_names) == 1:
            # one condition is the whole population and the draw was the random design's own: the same numbers
            out.update({"estimate": est_, "low": lo, "high": hi,
                        "method": "exact hypergeometric interval (simple random sample of a finite map)"})
        else:
            out.update({"estimate": est_, "low": lo, "high": hi, "conditions_in_interval": L,
                        "starved_strata": starved, "design_variance": var,
                        "method": "stratified by input condition; sum of each condition's exact interval at 1 - 0.05/L, "
                                  "weighted by its share of the map (union bound); covers at least 95% by construction"})
            if starved:
                out["warning"] = (f"{starved} of {sample['n_strata']} conditions had fewer than {MIN_PER_STRATUM} "
                                  "labelled windows and add nothing to design_variance; the whole-map interval does not "
                                  "use that variance")
    elif design == "random":
        # exact (review of 2026-09-23): the count of wrong windows in a simple random sample is hypergeometric, and
        # the finite-population Wilson form covered 0.79 one window short of a census and 0.92-0.94 at realistic
        # cells with few errors; the tail inversion covers at least 95% on every (N, K, n), as the per-class user's
        # accuracy already does. exp78 and exp79 graded the Wilson form, which stays in the experiments' harness.
        lo, hi = hypergeom_interval(int(wrong.sum()), idx.size, N)
        out.update({"estimate": float(wrong.mean()), "low": lo, "high": hi,
                    "method": "exact hypergeometric interval (simple random sample of a finite map)"})
    elif design in ("confidence", "proportional"):
        # rebuild the per-unit arrays over the population so the stratified estimator sees the sizes it was drawn with
        pop = np.asarray(sample["strata_of_population"], int)
        pos = {int(g): i for i, g in enumerate(pop)}
        stray = [int(g) for g in idx if int(g) not in pos]
        if stray:
            raise ValueError(f"{len(stray)} sampled window(s), first {stray[0]}, are not in the population this sample "
                             "was drawn from (an invalid window, or a sample built against another mask)")
        local = np.array([pos[int(g)] for g in idx])
        err = np.zeros(pop.size)
        err[local] = wrong
        est, lo, hi, starved, n_eff = stratified_interval_wilson(err, np.asarray(sample["strata"]), local, sample["sizes"], N)
        out.update({"estimate": est, "low": lo, "high": hi, "starved_strata": int(starved), "effective_n": n_eff,
                    "method": "stratified by confidence margin; Wilson interval on the design's effective sample size"})
        notes = []
        # every window labelled (the indices are distinct and all in the population): the rate is exact and the
        # interval the point stratified_interval_wilson returns. Until 2026-10-06 the notes below still called it
        # the simple-random Wilson bound, or narrower than it should be, beside an interval of zero width
        census = local.size >= pop.size
        if census:
            out["method"] += "; every window is labelled, so the rate is exact, with no sampling error"
        elif est in (0.0, 1.0):
            notes.append(f"{'no' if est == 0 else 'every'} labelled window was wrong, so the design's variance is zero and "
                         f"the interval is the simple-random Wilson bound at {idx.size} labels; the stratification "
                         "cannot narrow it without an observed error")
        if starved and not census:
            notes.append(f"{starved} of {sample['n_strata']} strata had fewer than {MIN_PER_STRATUM} labelled windows "
                         "and contribute no variance; the interval is narrower than it should be")
        if notes:
            out["warning"] = "; ".join(notes)
    elif design == "tiles":
        tiles = np.asarray(sample["tiles"]).ravel()
        err = np.zeros(tiles.size)
        err[idx] = wrong
        sizes = dict(zip([int(t) for t in sample["tile_ids"]], [int(v) for v in sample["tile_valid_sizes"]]))
        est, lo, hi, deff, var = cluster_interval(err, tiles, idx, tile_sizes=sizes)
        nlo, nhi = wilson_interval(int(wrong.sum()), idx.size, N)
        n_tiles = int(sample["n_tiles"])
        notes = [TILES_WARNING]
        # Equal rates are decided on the counts, exactly: the ratio estimator lands an ulp off a common rate of 1/7,
        # the variance came out 1e-34, and 6 wrong in 42 printed "14.3% to 14.3%" (the bug hunt of 2026-10-25).
        from fractions import Fraction
        t_l = tiles[idx]
        rates = {Fraction(int(wrong[t_l == u].sum()), int((t_l == u).sum())) for u in np.unique(t_l)}
        if len(rates) == 1:
            # every sampled tile shows the same rate (a clean map: no error in any tile); the between-tile variance
            # is zero and the interval would be a point. Fall back to Wilson on the labelled count, the bound the
            # design cannot improve on without an observed error.
            est = float(rates.pop())
            lo, hi = wilson_interval(int(wrong.sum()), idx.size, N)
            notes.append("every sampled tile shows the same error rate, so the between-tile variance is zero; the "
                         f"interval is the simple-random Wilson bound at {idx.size} labels, which is optimistic if "
                         "errors cluster")
        out.update({"estimate": est, "low": lo, "high": hi,
                    "design_effect": None if not np.isfinite(deff) else deff, "n_tiles": n_tiles,
                    "method": f"ratio estimator over tiles, ultimate-cluster variance, t on {n_tiles - 1} df",
                    "naive_interval_if_treated_as_random": {"low": nlo, "high": nhi},
                    "warning": "; ".join(notes)})
    else:
        raise ValueError(f"unknown design {design!r}")
    out["half_width"] = (out["high"] - out["low"]) / 2
    if by_condition:
        per, outside = _per_condition(grid, c_values, c_names, c_sizes, idx, wrong, N, out["estimate"])
        by_class = (sample.get("condition") or {}).get("by") == "map class"
        note = CONDITION_NOTE + (" " + (CLASS_WHOLE_MAP if by_class else CONDITION_WHOLE_MAP)
                                 if design == "condition" and len(c_names) >= 2 else "")
        out.update({"by_condition": True, "per_condition": per, "condition_note": note,
                    "outside_condition_intervals": outside})
        # the strata are map classes or profiles: decide and the texts word them so (cli.class_names, profile_names);
        # absent when they are input conditions, as before
        by = (sample.get("condition") or {}).get("by")
        if by:
            out["condition_by"] = by
    else:
        out["scope"] = SCOPE_ESTIMATE
    return out


def estimate_from_indices(indices, wrong, margin, valid=None):
    """An error rate from windows labelled without a design, treated as a simple random sample after checking only
    that they do not look like a review list. The tool's own review set is refused: it is built to hold errors, and
    labelling it then dividing gave two to six times the true rate on every task of exp78's export. Labels from the
    confident end of the map, or clustered by tile, pass the check, and the interval is then not valid."""
    margin = np.asarray(margin, dtype=np.float64).ravel()
    valid = (np.ones(margin.size, bool) if valid is None else np.asarray(valid, bool).ravel()) & np.isfinite(margin)
    chk = review_set_check(indices, margin, valid)
    if chk["n_outside_population"]:
        raise ValueError(f"{chk['n_outside_population']} of these windows are outside the valid map (no-data, or no "
                         "finite confidence); their labels say nothing about the map's error rate")
    if chk["n_duplicated"]:
        raise ValueError(f"{chk['n_duplicated']} window(s) appear more than once; a repeated label is not a new one")
    if chk["looks_like_a_review_set"]:
        raise ValueError(
            f"these {len(indices)} windows sit at a mean suspicion percentile of {chk['mean_suspicion_percentile']:.2f}, "
            f"above {chk['threshold']:.2f}, the most a random sample of that size reaches (a random sample sits at 0.50, "
            f"the review set near 0.97); {100 * chk['share_in_top_5pct']:.0f}% are in the top 5% most suspect. They are "
            "an enriched set, not a sample, and the rate they give is inflated. Draw a sample with "
            "sample_for_estimation, or pass a design.")
    sample = {"design": "random", "indices": np.asarray(indices, int), "n_population": int(valid.sum()), "budget": len(indices)}
    out = estimate_error_rate(sample, wrong)
    out["review_set_check"] = chk
    return out


# The same second reading for the whole map (estimate) and for the certified zone (certify), on the other designs.
# Every estimator and zone rule takes a 0/1 `wrong`, and a crosswalk fixed before any label makes "wrong under it" a
# fixed property of each window, so each guarantee holds for that reading as it holds for the strict one, and the two
# together by the union bound. The strict result is the call made without the option, so it does not change.
ESTIMATE_READING_NOTE = ("The alternative reading also accepts, for each map class, the reference classes the "
                         "--also-accept file lists for it, and reads the same labels a second time: a second test on "
                         "them. If the file was fixed before any label was read, each reading's interval holds at its "
                         "own level for its own definition of an error (exact or nominal, as its method says), and an "
                         "interval and its alternative hold together at least 90% of the time where each holds at 95% "
                         "(a union bound): the whole map's two, and each condition's two. A file written after the "
                         "labels were seen can be fitted to them, and then neither holds: listing the reference classes "
                         "the errors fell on clears them.")

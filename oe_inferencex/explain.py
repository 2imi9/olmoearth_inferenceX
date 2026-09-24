"""Why a window is suspect: label-free cues with measured enrichment.

Error ranking (assess.py) says which windows to review; this layer says why,
in structured form, for the caller to narrate (docs/plan/roadmap.md,
"Explanation layer"; docs/method/agent_integration.md). A cue is a fact about
a window that can be checked without labels and whose share among error
windows against correct windows has been measured on a reference. The cues
that failed as rankers are the material: a cue that does not order errors can
still explain them.

Enrichment is the only validation: share among errors against share among
correct windows on expert labels where they exist, on reference disagreements
otherwise, each cited to the experiment that measured it. Nothing here orders
windows; the review set comes from the assessment.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Cue:
    name: str
    sentence: str                      # templated, present tense, the window as subject
    share_errors: float | None         # share of error windows carrying the cue (None: not measured yet)
    share_correct: float | None        # share of correct windows carrying the cue
    reference: str                     # what "error" meant when the shares were measured
    source: str                        # experiment(s); every number traces to exp/out/
    measured_quantile: float | None = None   # for a cue with a {quantile} knob: the cut the shares were measured at
    verified: dict | None = None             # per suite task: (share_errors, share_correct) measured by exp82, labels grading only

    @property
    def enrichment(self):
        """Share among errors over share among correct; None when unmeasured."""
        if self.share_errors is None or self.share_correct is None or self.share_correct == 0:
            return None
        return self.share_errors / self.share_correct

    def quote(self, **fmt):
        """The templated sentence with its evidence; `fmt` fills placeholders such as {quantile} in the sentence."""
        if "{quantile" in self.sentence and "quantile" not in fmt:
            fmt = {**fmt, "quantile": self.measured_quantile}
        sentence = self.sentence.format(**fmt) if fmt else self.sentence
        q = fmt.get("quantile")
        if self.measured_quantile is not None and q is not None and abs(q - self.measured_quantile) > 1e-9:
            # the shares were measured at one cut; quoting them beside another would be a number nobody measured
            return (f"{sentence} (enrichment measured only at the {self.measured_quantile:.0%} cut, "
                    f"not at {q:.0%}; {self.source})")
        if self.enrichment is None:
            return f"{sentence} (enrichment not yet measured; {self.source})"
        text = (f"{sentence} ({self.share_errors:.0%} of error windows vs {self.share_correct:.0%} of correct ones, "
                f"{self.enrichment:.1f}x; {self.reference}; {self.source})")
        if self.verified:
            # exp82: the same cue on the suite's seven segmentation tasks; the enrichment is set by how much of
            # the map carries the cue, so the Bolivia number is one point of a range, never the map's own
            r = self.verified_range
            text += f"; on the suite's seven segmentation tasks {r[0]:.1f}x to {r[1]:.1f}x (exp82)"
        return text

    @property
    def verified_range(self):
        """(lowest, highest) enrichment among the per-task verifications, or None."""
        if not self.verified:
            return None
        v = [e / c for e, c in self.verified.values() if c > 0]
        return (min(v), max(v)) if v else None


BOLIVIA = "Sen1Floods11 Bolivia hand labels"
# exp82 (docs/results/comparisons.md): the same two cues on the suite's seven segmentation tasks under OlmoEarth
# Base, (share among errors, share among correct), tile-clustered intervals in exp/out/exp82_summary.json. The
# boundary enrichment runs from 1.24 (m-cashew-plant, 76% of windows on a boundary) to 8.13 (MADOS, 10%): what
# moves it is the map's fragmentation, and the Bolivia shares above are one point of that range.
EXP82_BOUNDARY = {"mados": (0.5436, 0.0669), "sen1floods11": (0.4535, 0.1147), "pastis_sentinel1": (0.7766, 0.4354), "pastis_sentinel2": (0.7869, 0.4700), "pastis_sentinel1_sentinel2": (0.7849, 0.4659), "m_cashew_plant": (0.8742, 0.7064), "m_sa_crop_type": (0.7070, 0.2617)}
# the low-confidence cue as the tool draws it, the least confident 20% of ONE scene's windows: exp82's per-tile cut
# (`low_confidence_per_tile`). Until 2026-09-23 these were exp82's cut over a whole task's pooled windows, whose
# enrichment (2.5x to 5.1x) the per-scene cut does not reach (1.4x to 3.4x); the review of that day found the mismatch
EXP82_LOW_CONFIDENCE = {"mados": (0.2898, 0.2123), "sen1floods11": (0.4442, 0.1807), "pastis_sentinel1": (0.3762, 0.1326), "pastis_sentinel2": (0.4678, 0.1394), "pastis_sentinel1_sentinel2": (0.4641, 0.1384), "m_cashew_plant": (0.3287, 0.1318), "m_sa_crop_type": (0.3591, 0.1183)}
# exp82's grain addendum: above this boundary prevalence the cue is nearly universal and the enrichment's ceiling
# (1 - e)/(p - e) sits near 1; Base's cashew map at a four-times coarser grid (0.896) has a risk ratio of 1.3,
# AnySat's at that grid (0.966) 0.81 and an enrichment of 0.99
BOUNDARY_SATURATED = 0.9
WC_DISAGREE = "WorldCover disagreements vs agreements, 27 rule scenes"
WC_DISAGREE_23 = "WorldCover disagreements vs agreements, the 24 rule scenes with at least 8 errors"

# The library. The expert-label shares are exp37's measurement on identical windows (Sen1Floods11 Bolivia, 81,984
# valid windows, 7,248 errors of the exp18 head; exp/out/exp37_summary.json, part_b.analysis.cues); the boundary
# shares equal exp36's boundary_share and exp18's 75% vs 21%. The reference-disagreement cues keep the numbers
# their experiments recorded. tests/test_explain.py checks the expert-label shares against exp37's summary.
CUES = {
    "boundary": Cue("boundary", "sits on a prediction boundary", 0.750, 0.214, BOLIVIA, "exp18, exp36, exp37",
                    verified=EXP82_BOUNDARY),
    "low_confidence": Cue("low_confidence", "is among the least confident {quantile:.0%} of the scene's windows (ties included)", 0.589, 0.163, BOLIVIA, "exp37", 0.2,
                          verified=EXP82_LOW_CONFIDENCE),
    "unstable": Cue("unstable", "changes prediction under a sub-patch shift of the tiling (top 20% of the scene)", 0.583, 0.164, BOLIVIA, "exp13, exp18, exp37"),
    "ndwi_ambiguous": Cue("ndwi_ambiguous", "is spectrally ambiguous between water and land (|NDWI| < 0.1)", 0.483, 0.067, BOLIVIA, "exp06, exp09, exp37"),
    "dihedral_disagree": Cue("dihedral_disagree", "is predicted differently under flips and rotations (top 20% of the scene)", 0.579, 0.164, BOLIVIA, "exp36, exp37"),
    "seasonal_water": Cue("seasonal_water", "lies on seasonal water (JRC seasonality 1-11 months)", 0.39, 0.08, WC_DISAGREE, "exp25"),
    # exp23's recorded medians (T1_enrichment), 13.7x over 24 scenes. Until 2026-09-22 this read 0.10 and 0.007, a
    # truncation that quoted 14.3x, on the 27 scenes of exp25 rather than the 24 exp23 kept (audit finding 14).
    "reference_unstable": Cue("reference_unstable", "changed class between WorldCover 2020 and 2021", 0.108, 0.0079, WC_DISAGREE_23, "exp23"),
    "osm_disagrees": Cue("osm_disagrees", "has an OSM river centerline where the map has no water", None, None, WC_DISAGREE + " (1.5x enriched, mostly reference-vs-reference)", "exp15"),
}


def _valid(assessment):
    arr = assessment["arrays"]
    valid = np.asarray(arr.get("valid", np.ones(np.shape(arr["confidence"]), dtype=bool)), dtype=bool)
    return arr, valid


def derive_cues(assessment, low_confidence_quantile=0.2):
    """The cues an assessment alone supports: boundary (indicator > 0) and low confidence (the least confident
    `low_confidence_quantile` of the valid windows, ties included). Boolean arrays of window shape."""
    arr, valid = _valid(assessment)
    conf = np.asarray(arr["confidence"], dtype=np.float64)
    cut = np.quantile(conf[valid], low_confidence_quantile) if valid.any() else -np.inf
    return {"boundary": (np.asarray(arr["boundary"]) > 0) & valid,
            "low_confidence": (conf <= cut) & valid}


def top_fraction(score, fraction=0.2, valid=None):
    """Boolean cue for a continuous score: the `fraction` highest values among the valid windows (ties included)."""
    s = np.asarray(score, dtype=np.float64)
    valid = np.ones(s.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.zeros(s.shape, dtype=bool)
    cut = np.quantile(s[valid], 1 - fraction)
    return (s >= cut) & valid


def cooccurrence(cues, mask=None):
    """Counts of windows carrying each pair of cues (diagonal: each cue alone), within `mask`."""
    names = list(cues)
    m = np.ones(np.shape(cues[names[0]]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    mat = np.zeros((len(names), len(names)), dtype=int)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = int((cues[a] & cues[b] & m).sum())
    return names, mat


def cue_enrichment(cue, errors, clusters=None, n_boot=2000, seed=0):
    """Share of a cue among error and among correct windows, their ratio, and a bootstrap interval of the ratio.

    `clusters` (tile or scene id per window) makes the bootstrap resample clusters, since windows of one tile are not
    independent (protocol, "Known limits"); without it, windows are resampled. Resamples with no error or no correct
    window, or a zero share among correct windows, are skipped and counted."""
    c = np.asarray(cue, dtype=bool).ravel()
    e = np.asarray(errors, dtype=np.float64).ravel() > 0.5
    out = {"n": int(len(e)), "n_errors": int(e.sum()), "n_with_cue": int(c.sum()),
           "share_errors": float(c[e].mean()) if e.any() else float("nan"),
           "share_correct": float(c[~e].mean()) if (~e).any() else float("nan")}
    out["enrichment"] = out["share_errors"] / out["share_correct"] if out["share_correct"] > 0 else float("nan")
    out["precision"] = float(e[c].mean()) if c.any() else float("nan")     # error rate among cue windows
    if n_boot:
        rng = np.random.default_rng(seed)
        cl = np.arange(len(e)) if clusters is None else np.asarray(clusters).ravel()
        ids = np.unique(cl)
        by = {k: np.flatnonzero(cl == k) for k in ids}
        ratios, skipped = [], 0
        for _ in range(n_boot):
            pick = rng.choice(ids, size=len(ids), replace=True)
            sel = np.concatenate([by[k] for k in pick])
            es, cs = e[sel], c[sel]
            if not es.any() or es.all() or not cs[~es].any():
                skipped += 1
                continue
            ratios.append(cs[es].mean() / cs[~es].mean())
        r = np.array(ratios)
        out.update({"boot_lo": float(np.percentile(r, 2.5)) if len(r) else float("nan"),
                    "boot_hi": float(np.percentile(r, 97.5)) if len(r) else float("nan"),
                    "n_boot": int(len(r)), "boot_skipped": int(skipped), "clustered": clusters is not None})
    return out


def explain_review_set(assessment, cues=None, budgets=None, library=CUES, low_confidence_quantile=0.2):
    """Structured evidence for each window of the assessment's review sets.

    assessment: output of assess.assess_prediction / assess_classmap (arrays + review_sets).
    cues: optional {name: boolean window array} the caller derived from other inputs (tile-phase, NDWI, JRC
    seasonality, a second reference, ...); the built-in boundary and low-confidence cues are added. Names in
    `library` carry their measured enrichment; other names are reported as unmeasured.
    Returns, per budget: one row per review window (row, col, confidence, the cues that fire), counts per cue,
    the cue co-occurrence matrix within the set, the windows no cue explains, and each cue's quote; plus the share
    of every cue among all valid windows of the scene, for contrast."""
    arr, valid = _valid(assessment)
    shape = np.shape(arr["confidence"])
    all_cues = derive_cues(assessment, low_confidence_quantile)
    for name, a in (cues or {}).items():
        a = np.asarray(a)
        if a.shape != shape:
            raise ValueError(f"cue {name!r} has shape {a.shape}, windows are {shape}")
        all_cues[name] = a.astype(bool) & valid
    names = list(all_cues)
    conf = np.asarray(arr["confidence"], dtype=np.float64)
    quotes = {n: (library[n].quote(quantile=low_confidence_quantile) if n in library else f"{n} (enrichment not measured)") for n in names}
    out = {"cues": names, "quotes": quotes, "n_windows": int(valid.sum()),
           "scene_share": {n: float(all_cues[n][valid].mean()) if valid.any() else float("nan") for n in names},
           "budgets": {}}
    # exp82: the boundary cue's enrichment is set by how much of the map is boundary (a ratio of shares cannot
    # exceed (1 - e)/(p - e) for a cue of prevalence p), so the map's own prevalence says where in the suite's
    # range it sits; the library's Bolivia number is one point of that range, not this map's
    if "boundary" in library and library["boundary"].verified and valid.any():
        p = out["scene_share"]["boundary"]
        lo, hi = library["boundary"].verified_range
        # No floor is promised: "at least 2.1 times on every task" was OlmoEarth Base at its native grain, and the
        # record itself measured 1.8 on another encoder's map and 1.5 and 1.3 on coarser grids (review, 2026-09-23)
        out["boundary_prevalence_note"] = (
            f"{100 * p:.0f}% of this map's windows sit on a prediction boundary. On the suite's seven segmentation tasks the "
            f"cue's enrichment ran from {hi:.1f}x on a map with 10% boundary windows to {lo:.1f}x at 76%; the error rate "
            f"inside the boundary set was 1.8 to 11.2 times the rate outside on five encoders' maps with 7% to 77% "
            f"boundary windows, and fell as that share rose: 1.5 times at 85%, 1.3 at 90% and below 1 at 97% (exp82 and "
            f"its grain addendum)")
        if p >= BOUNDARY_SATURATED:
            # exp82's grain addendum: Base's cashew export at a four-times coarser grid has 90% boundary windows and a
            # risk ratio inside/outside of 1.3 (2.1 at the fine grid); AnySat's export at that grid has 97% and 0.81
            out["boundary_prevalence_note"] += (
                f" At {100 * p:.0f}% the cue is nearly universal and cannot enrich much (a ratio of shares is bounded by "
                f"(1 - e)/(p - e) for a map with error rate e), so on this map the boundary cue is at most a weak reason")
    sets = assessment.get("review_sets", {})
    for b, rs in sets.items():
        if budgets is not None and b not in budgets:
            continue
        rc = np.asarray(rs["windows_rowcol"], dtype=int).reshape(-1, 2)
        mask = np.zeros(shape, dtype=bool)
        mask[rc[:, 0], rc[:, 1]] = True
        rows = []
        for r, c in rc:
            fired = [n for n in names if all_cues[n][r, c]]
            rows.append({"row": int(r), "col": int(c), "confidence": float(conf[r, c]), "cues": fired})
        _, mat = cooccurrence(all_cues, mask)
        none = [[int(r), int(c)] for r, c in rc if not any(all_cues[n][r, c] for n in names)]
        out["budgets"][b] = {
            "n_windows": int(len(rc)), "windows": rows,
            "count": {n: int(mat[i, i]) for i, n in enumerate(names)},
            "share_in_set": {n: float(mat[i, i] / max(len(rc), 1)) for i, n in enumerate(names)},
            "cooccurrence": mat.tolist(),
            "n_without_cue": len(none), "windows_without_cue": none,
        }
    return out


def library_table(library=CUES, **fmt):
    """The cue library as plain rows (for docs and JSON); `fmt` fills sentence placeholders."""
    return [dict(asdict(c), enrichment=c.enrichment, quote=c.quote(**fmt)) for c in library.values()]


def confusion_pairs(reference, decision, valid=None, top=3):
    """Which (predicted, reference) class pairs the errors fall into, most frequent first: the systematic-error
    report Singh et al. (2024) ask for on a segmentation map, which the cue library cannot give because it never
    sees a label. Windows where the reference is negative (no majority label) or invalid are left out. On the
    suite's ten classification tasks with six or more classes the top three pairs held 18% (ForestNet) to 61%
    (BreizhCrops) of the errors (exp82), so the report says the share it explains beside the pairs."""
    ref, dec = np.asarray(reference).ravel(), np.asarray(decision).ravel()
    for name, x in (("reference", ref), ("decision", dec)):
        # a class is a whole number: astype(int) used to truncate 0.6 to 0 and 1.4 to 1 (review of 2026-09-23)
        if x.dtype.kind == "f" and not np.all(np.isnan(x) | (np.mod(np.nan_to_num(x), 1) == 0)):
            raise ValueError(f"{name} must hold whole class ids (negative or NaN for no label)")
    ref = np.where(np.isnan(ref), -1, ref).astype(int) if ref.dtype.kind == "f" else ref.astype(int)
    dec = np.where(np.isnan(dec), -1, dec).astype(int) if dec.dtype.kind == "f" else dec.astype(int)
    if ref.size != dec.size:
        raise ValueError(f"reference has {ref.size} windows, the decision {dec.size}")
    # validity is read as > 0.5, the package's convention for indicators; bool(0.4) was True
    ok = (ref >= 0) & (dec >= 0) & (np.ones(ref.size, bool) if valid is None else np.asarray(valid, np.float64).ravel() > 0.5)
    e = ok & (dec != ref)
    n_err = int(e.sum())
    if n_err == 0:
        return {"n_errors": 0, "n_pairs": 0, "pairs": [], "share_of_errors_in_top": float("nan")}
    pairs, counts = np.unique(np.stack([dec[e], ref[e]], 1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], pairs[:, 0], -counts))[:max(int(top), 0)]      # ties broken by class ids, stably
    return {"n_errors": n_err, "n_pairs": int(counts.size),
            "pairs": [{"predicted": int(pairs[i, 0]), "reference": int(pairs[i, 1]), "n": int(counts[i]), "share": float(counts[i] / n_err)} for i in order],
            "share_of_errors_in_top": float(counts[order].sum() / n_err)}

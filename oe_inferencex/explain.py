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

    @property
    def enrichment(self):
        """Share among errors over share among correct; None when unmeasured."""
        if self.share_errors is None or self.share_correct is None or self.share_correct == 0:
            return None
        return self.share_errors / self.share_correct

    def quote(self, **fmt):
        """The templated sentence with its evidence; `fmt` fills placeholders such as {quantile} in the sentence."""
        sentence = self.sentence.format(**fmt) if fmt else self.sentence
        if self.enrichment is None:
            return f"{sentence} (enrichment not yet measured; {self.source})"
        return (f"{sentence} ({self.share_errors:.0%} of error windows vs {self.share_correct:.0%} of correct ones, "
                f"{self.enrichment:.1f}x; {self.reference}; {self.source})")


BOLIVIA = "Sen1Floods11 Bolivia hand labels"
WC_DISAGREE = "WorldCover disagreements vs agreements, 27 rule scenes"

# The library. Shares on expert labels come from exp36's run of record (boundary_share in exp/out/exp36_summary.json,
# identical to exp18's 75% vs 21%); the reference-disagreement cues keep the numbers their experiments recorded.
# Cues with None shares are derived here but their shares are measured on identical windows by exp37.
CUES = {
    "boundary": Cue("boundary", "sits on a prediction boundary", 0.750, 0.214, BOLIVIA, "exp18, exp36"),
    "low_confidence": Cue("low_confidence", "is among the least confident {quantile:.0%} of the scene's windows (ties included)", None, None, BOLIVIA, "exp37"),
    "unstable": Cue("unstable", "changes prediction under a sub-patch shift of the tiling (top 20% of the scene)", None, None, BOLIVIA, "exp13, exp18, exp37"),
    "ndwi_ambiguous": Cue("ndwi_ambiguous", "is spectrally ambiguous between water and land (|NDWI| < 0.1)", None, None, BOLIVIA, "exp06, exp09, exp37"),
    "dihedral_disagree": Cue("dihedral_disagree", "is predicted differently under flips and rotations (top 20% of the scene)", None, None, BOLIVIA, "exp36, exp37"),
    "seasonal_water": Cue("seasonal_water", "lies on seasonal water (JRC seasonality 1-11 months)", 0.39, 0.08, WC_DISAGREE, "exp25"),
    "reference_unstable": Cue("reference_unstable", "changed class between WorldCover 2020 and 2021", 0.10, 0.007, WC_DISAGREE, "exp23"),
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

# Lean proofs behind estimate's and certify's bounds

Two Lean 4 files with Mathlib. They prove the arithmetic and the measure-theoretic steps that the bounds for
windows a reviewer could not judge (`?`) and for a reviewer who errs rest on. The package does not import them;
`tests/test_unjudged.py` checks the package's own intervals by exact enumeration: their coverage, and the closed
forms; sharpness is proved here only.

| File | What it proves | Where the package uses it |
|---|---|---|
| `Robust/LabelNoise.lean` | With e0 the share of truly correct windows the reviewer marks wrong and e1 the share of truly wrong windows marked right, the labels show p = θ(1 - e1) + (1 - θ)e0. If e0 ≤ E0 < 1 and e1 ≤ E1 < 1, then (p - E0)/(1 - E0) ≤ θ ≤ p/(1 - E1), and both ends are attained (sharpness). | `estimate --reviewer-false-alarm E0 --reviewer-miss E1` |
| `Robust/RobustInterval.lean` | T1 the same bounds in sharp form; T2 an interval that covers p maps to one that covers θ, so coverage can only grow; T3 the same with random bounds, by a union bound; T4 windows nobody could judge: if the interval's ends move one way with the error count and the true count lies between k and k + u, then [L(k), U(k + u)] holds [L(k*), U(k*)], and so covers whenever the full labels' interval does; T5 certification: a zone wrong more than α of the time shows a label rate above α(1 - E1). | `estimate` with `?` rows (`unjudged_coverage`) and with reviewer error rates (`identified_sharp`, `coverage`). T5 is not used: `certify` takes no reviewer error rate, since T5's premise, at most E1 missed inside the zone, cannot be taken from a whole-map rate |

What the files do not prove: that the package's exact intervals move one way with the error count (the hypothesis
of T4). `tests/test_unjudged.py` checks it numerically: for the random design's exact interval at every error count and
every split of `?` rows on small maps, and for the union bound over strata at every combination of stratum counts.

## Build

Compiled on the aicr cluster with Lean 4.35.0-rc3 and the Mathlib commit in `lake-manifest.json`: first on 1
October 2026 (Slurm job 1146459, under other module paths), and again byte for byte as they stand here (job
1162226, on 2 October 2026). Each file built with exit 0, no errors and no `sorry`; Lean printed only linter
warnings (unused hypotheses, and `Set.mem_setOf_eq`, which this Mathlib marks deprecated). The CI does not build
them: Mathlib is a multi-gigabyte download.

```bash
cd lean
lake exe cache get     # Mathlib's prebuilt files
lake build
```

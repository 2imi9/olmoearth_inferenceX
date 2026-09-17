# Changelog

## 1.0.0 (2026-09-17)

The first release with the evidence frozen behind it. Every function's behaviour is the one the recorded
experiments used; the numbers in the documentation were computed with this code.

**What the package does**

- `assess_prediction`, `assess_classmap`: a prediction map, probabilities or logits, binary or per-class, or a
  served class map with its confidence, into review sets at chosen budgets, in confidence order or boundary
  first, with operating points and the attainable ceiling; an optional reference raster scores them.
- `explain_review_set`: why each review window is suspect, as label-free cues with the enrichment measured for
  each on expert-labelled testbeds.
- `compare_inferences`: two inferences of one scene on identical windows, disagreement rate pooled and per
  group, where the differing windows sit, stability of the set, and with labels the cross-tab and which side
  was right.
- `fit_ranker`, `fit_side`: the label-fitted fusion and side rule, cross-fitted by group and locked to the model
  family they were fitted on.
- `metrics`, `stats`: tie-aware AURC and excess AURC, capture at a budget, design-weighted estimators, exact
  sign tests, cluster bootstraps.
- `oe-inferencex assess` and `oe-inferencex compare`: the same from the command line, rasters or `.npy` in,
  JSON, CSV and rasters out.

**Public surface.** Declared in `oe_inferencex.__all__`; the encoder-bound modules (`evidence`, `awf`, `data`,
`figstyle`) are installed with the `encoder` and `geo` extras and raise a plain instruction otherwise.

**Licence.** Apache-2.0, in `LICENSE`; the release is tagged `v1.0.0`.

**Evidence.** 65 preregistered experiments and 146 ledger claims, each pinned to a committed artifact by an
executable check; the summary is `docs/Findings.md` and the origin of each idea is `docs/related_work.md`.

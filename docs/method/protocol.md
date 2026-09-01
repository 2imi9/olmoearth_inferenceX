# Evaluation protocol and positioning

Part of the results documentation; index at [TECHNIQUES.md](../TECHNIQUES.md).

Status terms: **supported** (at least one experiment consistent with the
technique's claim, under the stated conditions) / **mixed** (results differ
across conditions) / **partial** (some evidence; a key condition untested) /
**rejected** (tested, contradicted) / **untested** / **blocked** /
**out of scope (v1)**.

Evidence tiers: single-scene results (exp01-exp08) establish direction
only; exp09 is a seven-scene comparison with hand-chosen scenes; exp11 is
the authoritative comparison - 29 scenes under a pre-registered selection
rule, with per-scene bootstrap CIs and permutation tests
(exp/out/exp11_stats.csv). Where exp09 and exp11 disagree, exp11 stands.

## Related work and positioning

The individual signal families are not new, and the ledger should not be read
as claiming they are. Confidence-based map assessment appears in the CEOS
WGCV land cover validation protocols as a complement to reference-data
assessment. Test-time-augmentation uncertainty has been applied to EO
segmentation (e.g. landslide mapping), following Wang et al. 2019 in medical
imaging. The Area of Applicability / Dissimilarity Index (Meyer & Pebesma
2021) is adopted in spatial statistics via the CAST and waywiser packages,
for tabular predictor spaces. SHRUG-FM (CVPR 2026 EarthVision) performs
embedding-space OOD detection for EO foundation models. Ensemble
disagreement is standard uncertainty practice in mainstream ML.

What we did not find in the EO literature, and what this repository targets:
selective-prediction evaluation (risk-coverage / AURC) of land cover
inference; cross-model disagreement used as an audit signal; perturbation of
the ViT patchification grid specifically; and the combination of such
signals into an audit that is scored against the audited model's own
confidence with no-model controls, over regions without labels. The
contribution claim is the audit protocol and the ViT-specific
instantiations, not the signal families.

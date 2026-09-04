# Evaluation protocol

How results in this repository are produced, weighted, and named. Every
other document assumes this page and does not restate it. Index at
[TECHNIQUES.md](../TECHNIQUES.md).

## What is being measured

The apparatus answers one question in two forms: **given two things that
disagree, which should you believe, and is the difference real?**

- **Error ranking.** Each signal assigns every map window a suspicion score;
  the score is judged by how well it ranks the windows the model gets wrong.
- **Cross-inference evaluation.** Two inference runs — model versions,
  encoder internals, input years, reference vintages — are scored on
  identical errors and compared under the same tests.

The machinery is the same for both, and is deliberately signal-agnostic:
`aurc_expected(uncertainty, errors)` takes any score vector and any error
vector. Signals are label-free; labels only grade the signals, never train
them.

A signal is credible only if it beats two references at once:

- **the model's own confidence**, scored as the negative absolute logit
  (top-1 minus top-2 for multiclass) rather than `1 - max probability`, so
  saturated probabilities do not tie;
- **a no-model pixel control**, computed from pixel values alone (for
  water, the NDWI gradient magnitude).

Beating one but not the other is not support.

## How results are scored

The statistics settled in exp13 and used by every experiment after it:

| Element | Choice | Why |
|---|---|---|
| Metric | Tie-aware AURC, and excess AURC (E-AURC = AURC minus the oracle's) across scenes | The boundary score has nine levels and float32 sigmoid saturation ties confidence at zero on many patches, so a stable-sort AURC would depend on raster order. E-AURC makes absolute levels comparable across scenes; it changes no signal-minus-baseline difference. |
| Perturbation alignment | Shifted prediction maps upsampled to pixels, placed at their true offset on a common canvas, then pooled back | exp05/exp09/exp11 compared unaligned patch grids, so shifted patches covered different ground. |
| Per-scene uncertainty | 4x4-patch block bootstrap, B=1000 | Percentile intervals are biased for a rank statistic on high-error scenes, so they are indicative only. |
| Cross-scene tests | Exact sign test on untied pairs, plus a sign-flip permutation test on mean E-AURC differences | The sign test is scale-free and is reported as primary; mean-based tests are dominated by high-error scenes. |
| Reporting | Wins / losses / ties per scene, not means | Same reason. |

### Known limits of these tests

- **The 27 scenes are not 27 independent draws.** They sample fixed fractions
  along eight named rivers, so scenes on one river share its reference
  errors, season and channel morphology. Re-running the headline sign test
  with one vote per river (majority of that river's scenes) gives tiling
  instability **8/8 rivers, p=0.008**, against 26/27 scenes, p=4e-07. The
  result survives clustering; its significance is three orders of magnitude
  weaker than the scene-level figure suggests, and the scene-level p-value
  should not be quoted on its own. Reproduce with
  `uv run --extra geo python exp/summary_transfer.py`.
- **A null is not a demonstration of equality.** Where a signal is reported
  as not beating confidence at p > 0.05 (aligned tile-phase on Sen1Floods11
  Bolivia, 163/187, p=0.22) the claim is that no advantage was shown, not
  that the two are equivalent. No equivalence test has been run.
- **Cross-testbed comparisons change several things at once.** Moving from
  the WorldCover scenes to Sen1Floods11 changes the reference, the task
  (permanent against flood water), the geography, the unit of analysis
  (scene against tile), the input processing (L2A against L1C through the
  L2A path) and the head's training set. That the advantage does not
  transfer is established; *which* of those differences causes it is not,
  and that is the open question exp23-exp25 attack.

Splits are geographic hold-outs. Scene selection is pre-registered before
any scene is fetched (the rule is in
[results/comparisons.md](../results/comparisons.md)); its coordinates have
been cached in `exp/out/rule_candidates.json` since exp25, after an Overpass
mirror failure silently dropped rivers from one run's scene set.

## Evidence tiers

Later tiers override earlier ones where they disagree.

1. **Authoritative — expert labels.** exp18 (Sen1Floods11 hand labels,
   geographic hold-out) for the water task; exp21 (the fine-tuned AWF model
   end to end) and exp16/exp04 (AWF expert points) for the classification
   task. These supersede WorldCover-referenced results wherever they
   conflict.
2. **Authoritative — WorldCover reference.** exp13 on the 27 rule-selected
   scenes (`exp/out/exp13_corrected_stats.csv`), with exp14 supplying the
   mechanism. Claims from this tier are claims about disagreement with a
   weak reference, not about model error.
3. **Directional only.** Single-scene results (exp01-exp08) and exp09's
   seven hand-chosen scenes. Their tile-phase numbers were computed before
   the alignment fix and are not comparable; they are kept in
   [../../exp/NOTES.md](../../exp/NOTES.md), not in the results docs.

## Status terms

| Term | Meaning |
|---|---|
| **supported** | At least one experiment consistent with the claim, under stated conditions |
| **mixed** | Results differ across conditions |
| **partial** | Some evidence; a key condition untested |
| **rejected** | Tested and contradicted |
| **untested** | No experiment yet |
| **blocked** | Requires something unavailable |
| **out of scope (v1)** | Deliberately excluded |

## Related work and positioning

The individual signal families are not new, and the ledger should not be
read as claiming they are:

- Confidence-based map assessment appears in the CEOS WGCV land cover
  validation protocols, as a complement to reference-data assessment.
- Test-time-augmentation uncertainty has been applied to EO segmentation
  (e.g. landslide mapping), following Wang et al. 2019 in medical imaging.
- The Area of Applicability / Dissimilarity Index (Meyer & Pebesma 2021) is
  adopted in spatial statistics via the CAST and waywiser packages, for
  tabular predictor spaces.
- SHRUG-FM (CVPR 2026 EarthVision) performs embedding-space OOD detection
  for EO foundation models.
- Ensemble disagreement is standard uncertainty practice in mainstream ML.
- Boundary-concentrated error is well known in segmentation and land-cover
  validation (trimap and Boundary-IoU evaluation; mixed-pixel effects,
  Foody 2002; Radoux and Bogaert 2017), and is not claimed as new.

**Upstream evaluation this is measured against:**
[rslearn segmentation tasks](https://github.com/allenai/rslearn/blob/master/rslearn/train/tasks/segmentation.py),
the [AWF task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml)
whose classes and split are reused here, and
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals).

**What we did not find in the EO literature**, and what this repository
targets: selective-prediction evaluation (risk-coverage / AURC) of land
cover inference; cross-model disagreement as an audit signal; and the
combination of such signals into an audit scored against the audited
model's own confidence, with no-model controls, over regions without
labels.

**The contribution claim** is the comparison protocol itself — pre-registered
selection, spatial hold-out, tie-aware metrics, a no-model control and an
exact significance test, applied to label-free comparison of inference
outputs — plus its finding that a perturbation-based instability signal —
statistically
indistinguishable from proximity to a boundary in the model's own
prediction map — ranks errors better than confidence on the rule-selected
scenes (exp13/exp14), while failing to do so against expert labels
(exp18/exp21).

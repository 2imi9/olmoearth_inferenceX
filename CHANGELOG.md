# Changelog

## Unreleased

**The package now says how wrong a map is, given a labelled sample.** exp78 measured what that costs on the
seven segmentation tasks of Ai2's suite with every unit labelled, so the intervals could be graded; the
estimators it ran now live in the package and it imports them back, so the recorded run and the shipped code
cannot drift.

- **`oe_inferencex.estimate`**, numpy only. `sample_for_estimation(margin, budget, p1=...)` chooses the windows to label: the default design stratifies by confidence margin and allocates by Neyman's rule from the model's own confidence, which on exp78's tasks narrowed the interval to a median 0.80 of a random sample's (0.63 on the cleanest map) with coverage intact; `design="random"` and `design="tiles"` are the plain draw and the way people actually label. `estimate_error_rate(sample, wrong)` returns the rate with the interval the design earns: Wilson with a finite-population correction, the stratified Wald interval, or for tile-sampled labels the ultimate-cluster interval **beside the naive one**, because labelling whole tiles and using the ordinary formula gave a "95%" interval that covered 51 to 78% of the time.
- **A guard against the natural wrong thing.** On exp78's export, labelling the tool's own 5% review set and dividing gives 1.8 to 5.8 times the true rate on every task, and nothing stopped it. `estimate_from_indices`, for windows labelled without a design, checks the sample's median suspicion percentile (0.50 for a random draw, about 0.97 for the review set) and refuses a review set with the number.
- **`oe-inferencex sample` and `oe-inferencex estimate`.** `sample` writes the windows to label as a CSV with an empty `wrong` column and the design in a sidecar; `estimate` reads the filled file back and refuses a blank row or a row order that is not the design's.
- `wilson_interval` returns the point `(p, p)` at a full census; before, the finite-population correction zeroed the half-width around Wilson's shrunk centre and the interval missed the truth with certainty. Unreachable in exp78 and exp79, so no recorded number moves.
- The intra-cluster correlation behind the design effect takes its grand mean over the units it analyses, those in tiles of two or more windows; exp78's inline copy took it over all units. MADOS's recorded design effect moves from 9.774639 to 9.774605 and no cited digit changes.
- README states the method: ranking, comparing, estimating.

## 1.1.3 (2026-09-21)

**Fixes six defects an adversarial audit found in the released package. Four of them made it return a plausible
wrong answer with exit 0, which is the failure mode this project exists to prevent in the maps it audits. Anyone
on 1.1.2 should upgrade.** The audit confirmed 21 findings in all; the ranking core held under three independent
lenses (review order, tie rules, budget arithmetic, the label bridge all bit-identical), and what was wrong was
the periphery: the second input, the no-data path, the name on an output file and the fitting layer's
self-evaluation. No recorded number changes; the demo's published figures are identical.

- **`suspicion.tif` held confidence, not suspicion.** Ranking it descending, which its name invites, returned the exact inverse of the review set: on a 256-window scene the top 13 of the raster overlapped the 13-window 5% review set in 0 of 13. An analyst opening it with a hot-is-bad colour ramp reviewed the windows the model was most confident about and skipped everything the tool flagged.
- **`assess --reference` pooled the reference over the prediction's class range**, so any reference class the map cannot predict collapsed to class 0 and the error rate came out understated, always in the flattering direction and with no warning: 0.0625 against an honest 0.1875 for a binary flood model graded against dry, flood and permanent water. The identical defect had been found, measured at 44.9% and fixed for `compare --labels`, and was never carried across.
- **A window with no prediction was given class 0**, so it disagreed with every neighbour and manufactured a prediction boundary around each cloud hole and scene edge: a map predicting one class everywhere it had a prediction reported a boundary window fraction of 16.7% and a boundary-first review set that was entirely the rim of the data. The denominator of eight is unchanged, so on a fully observed map the boundary is bit-identical to before and every recorded cue number stands.
- **Non-finite scores are treated as no-data and named in a warning.** NaN sorts to the front of the review order, so a scene with a NaN strip previously returned a review set made entirely of windows that contained no prediction, beside healthy-looking confidence quantiles.
- **A `(1, H, W)` score map is refused** with an instruction to pass `scores[0]`. It is the shape a binary head returns, and it was scored as a one-class map, inverting the meaning of the order.
- **`calibrate.fit_ranker` graded a sign-free fusion against a sign-locked baseline.** A user passing readings oriented "higher is safer", which the docstring invites, was shown a lead of 0.389 of excess AURC where the honest gap was 0.00025, with a sign test to match. The baseline now takes the better of its two orientations and the report records which. exp65's recorded result is unaffected: it passes its readings already oriented.
- **An unfittable cross-validation fold no longer reaches `held_out`.** Those rows sat at logit exactly 0.0 and were reported as a held-out evaluation, so on a map with few errors the report could say the ranker found 0% of them at a 10% budget while quoting a better-than-random excess AURC computed from that same fabricated vector. They are NaN now, excluded from every held-out number, and counted in `n_unscored_rows`.


- A false claim is corrected and the defect behind it fixed. exp54's cross-encoder block aligned each model's rows to OlmoEarth's by a hash of the label tile; identical tiles collide to one key, so where they are not adjacent the reorder permutes rows even when a model is aligned against itself, and the guard checked only that every key was found. The record said encoders share OlmoEarth's errors on binary water but not on multi-class (phi 0.05-0.08 on PASTIS); on the same 458,638 windows exp63 measures 0.50-0.68, and the two runs contradict each other on their face. The phi block is withdrawn, the claim is re-pointed at exp63 with a check that reads both artifacts, four documents are corrected, and the alignment now refuses to run unless its index is a permutation. exp54's preregistered result is computed before any alignment and is unaffected.
- Recorded from the committed artifacts, with no new run: the share of the ranking headroom the margin takes is a property of the task, not of the encoder. Over the suite's 16 encoders it runs 0.553 to 0.693; OlmoEarth Large is the best of them and Base ties for second; four model sizes move it by four points; and the spread between tasks is 2.8 times the spread within a task across all sixteen.

- exp77 recorded (job 1003587, 1 h 40 m on cpu): a confident error is more typical of its own scene, and removing that does not help. On the seven segmentation tasks of Ai2's suite, inside the confident half the error windows sit closer to their tile's mean token than the correct ones on 5 of 7 tasks (3 of 5 sources, negative on Sen1Floods11, undefined on MADOS), but the gap needs labels and has no control for class frequency, which predicts the same sign. Projecting that direction out of the frozen tokens loses accuracy on all seven, 0.03 to 0.98 points, so the diagnosis has nothing to locate; and the margin keeps its lead over every reading introduced. Four claims; the ledger stands at 161. No package code changes.

## 1.1.2 (2026-09-19)

- The README is short, in the shape of a tool's README: what the package is, the pipeline diagram, one finding, the documentation link, the demo and setup, 46 lines. The sections that argued the evidence live on Read the Docs only, and the package's page on PyPI shows this README from this release on.
- The demo answers the question its own picture raises. Most of the red lies outside the flagged windows, which reads as failure until one sees that the map is 19% wrong and the review is 5%: the run now prints, per review budget, the errors the tool would find, what a random review finds and the most any review could (at 5%: 17%, 5%, 26%; at 20%: 55%, 20%, 100%), and says so in one sentence. The README caption and the docs front page carry the same sentence.

## 1.1.1 (2026-09-19)

- `oe-inferencex demo` audits a real map. The package ships one tile of Dynamic World (a served global land-cover product this project had no hand in) with its published probabilities and the expert annotation of the same ground, reduced to 40 m windows, about 250 KB, CC BY 4.0 with its notice. The tile was chosen by a rule fixed before any tile was looked at, the lower-median tile by error capture among the 18 of 409 test tiles that are fully annotated, because the candidates' capture runs from 0.10 to 0.50 and a hand-picked tile could have said anything (`scripts/make_demo_sample.py`, cluster job 995922). `--made-up` keeps the synthetic water map.
- The demo's picture explains itself: three titled panels (what an audit gives with no labels used; the same windows over the map's real errors; a random pick of the same size over the same errors), a legend, and titles that carry the hit rates, drawn with a built-in 5 x 7 font so that nothing but numpy is needed. The random pick is reported by its expectation, not by the luck of one draw.
- The demo's words lead with what the tool is worth to a reviewer (how often a flagged window is really wrong, against a random one; how much of the map can be used as it is), then the reasons windows are flagged, then what the tool does not do, then the real `assess` command on the sample file before the user's own map.
- Two claims pin the demo's numbers in the ledger (157 claims).

## 1.1.0 (2026-09-18)

The first release on PyPI. What a user of 1.0.0 gains: a first run that needs no data (`oe-inferencex demo`), one
minus the top probability as a confidence form for multi-class maps (`form="top1"`), and refusals for inputs the
record does not support. The 1.0.0 defaults and every recorded number are unchanged. The source archive now carries
the package alone; the experiments, their outputs and the tests against them stay in the repository.

- `oe-inferencex demo`: a small made-up water map is audited through the same code path as `assess` and drawn with the standard library only, review set on the left, true errors on the right. Its error rate and capture are set to resemble a real flood map of the record, not to flatter, and the run says that it illustrates and does not prove.
- The README renders on PyPI: repository links and images are absolute.

- The document in `report/` (previously `paper/`) is titled a technical report and dated 17 September 2026; the README and the docs front page link to it.
- Removed 28 outdated files after an audit of every tracked file: an abandoned overview figure set, the re-targeting draft figures, leftover rasters from early drafts of the comparison figure, three rasters no diagram reads, the PDF build products of the four live diagrams, exp04's partial resume checkpoint (a bit-identical subset of `exp04_feats.npz`) and the figure of the superseded exp09 run. All remain in tag v1.0.0. The list, with what stays open, is `docs/plan/cleanup-audit-2026-09-17.md`.
- Removed 414 more files, the audit's second tier, at the author's choice: sixteen flat CSV copies of claim-pinned summaries, the exp73 and exp74 resume checkpoints (identical to their summaries), exp16's 66 MB token cache, the 2026-09-02 Word report and the four figures only it showed (exp21, exp23, exp24, exp25), two figures no page shows (exp05, summary_transfer), the workshop paper outline, the exp27 draft and four superseded cluster scripts. All remain in tag v1.0.0; nothing a claim, test or page reads was touched.

- exp75 recorded: heads fitted on the other sensors of the same units disagree in a way that beats every no-model control (5 of 5 groups) but does not find enough of the errors the margin misses to change a review set (preregistered P2 3 of 5, P3 0 of 5); a view helps only if its head can do the task, and averaging the heads improves accuracy on 1 of 5 groups. Four claims, no package change.

- Related work: 65 papers from 2024 to 2026 added across the six sections, each checked against its arXiv, Crossref or OpenAlex record, and a seventh section that tables every published challenge to a recorded finding against what the record answers, with the untested ones and their cost. One measured claim on the headroom: the margin takes a median 0.68 of the random-to-perfect ranking gap on the 24 tasks, and labels buy a fifth to a third of the rest.

- The agent skill (OlmoEarth-Agent pull request 155) is merged into that repository's main branch; the integration page and the report say so.

- Readiness for a test-time-training encoder (ViT3): `signals.crop_dependence` (how much of a decision map depends on the crop it was inferred in), `compare.determinism_check` (the same input inferred twice under two engines or precisions, gated against the reseed floor), `scripts/suite_regression.py` (exp70's protocol on any model directory of the published suite, with any candidate reading screened beside the margin and the controls), the compute budget per reading in the recipe, and the four predictions preregistered in `docs/plan/vit3_readiness.md`. No recorded number changes.

- exp76 recorded, and `form="top1"` added to `assess_prediction` and `signals.confidence`: one minus the top probability, tie-free from logits, the best member of the confidence family on 14 of 16 multi-class tasks; the default stays the 1.0.0 logit margin, which was the weakest form there and now warns on multi-class logit maps. AUGRC leaves the suite result standing (23 of 24 by AUROC); no whole-vector score and no window aggregator does better. Four claims.

- Refusals for inputs outside what the record supports, so that no caller gets a plausible review set for a map the package cannot rank. `assess_prediction` raises on probability input outside [0, 1] (the command line already refused the 2-D case; the API and the per-class case did not, so a regression raster imported by an agent was scored); a review set whose cut-off falls inside a run of equal scores carries `tied_at_cutoff` and a warning, which is what a hard mask, a quantized band or a constant map produce; `oe-inferencex compare` refuses a continuous map under the default cut-off, where two regression outputs "never differed", and takes it with `--threshold` named, noting that no recorded experiment grades that case. Usage opens with a table of what goes in, what the record supports for it, and what is refused. No recorded number changes.

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

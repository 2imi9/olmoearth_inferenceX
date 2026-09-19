# Changelog

## Unreleased

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

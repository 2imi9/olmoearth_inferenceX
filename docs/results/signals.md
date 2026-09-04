# Per-signal evidence

One section per signal, ordered strongest first. Each gives the definition
as implemented, the evidence, and the verdict. Per-experiment detail is in
[comparisons.md](comparisons.md); scoring rules in
[../method/protocol.md](../method/protocol.md). Index at
[../TECHNIQUES.md](../TECHNIQUES.md).

Notation: "26/27" = better on 26 of the 27 rule-selected scenes.
"163/187" = better on 163 tiles, worse on 187.

---

## Max-softmax confidence — the baseline

**Definition.** Negative absolute logit (top-1 minus top-2 for multiclass),
not `1 - max probability`, so float32 sigmoid saturation does not tie it.

**Against expert labels it wins everywhere tested:** the AWF point task
(exp04, exp16), Sen1Floods11 dense masks with a geographic hold-out (exp18,
lowest pooled E-AURC on both splits), and the fine-tuned AWF model end to
end (exp21, AURC 0.0262).

**Against the WorldCover reference it loses**, best on 0/27 rule-selected
scenes where aligned tile-phase beats it 26/27 (exp13). Why the loss does
not transfer to expert labels is open; three candidate explanations have
been tested and rejected (comparisons.md section 3).

**Verdict: supported.** It is the signal to beat, and nothing has beaten it
on expert-labelled truth.

---

## Perturbation stability (E_system, "tile-phase")

**Definition.** Shift the input window origin by 1-3 pixels (sub-patch
phase), upsample each shifted prediction map to pixels, place it at its true
offset on a common canvas, take the per-pixel standard deviation across
shifts, and pool back to the shift-0 patch grid. Alignment is essential —
see protocol.md.

| Reference | Result |
|---|---|
| WorldCover, 27 scenes (exp13) | Beats confidence **26/27** (p=4e-07), the pixel control 18/27; most frequent best signal |
| Sen1Floods11 hand labels (exp18) | **Does not beat confidence**: Bolivia 163/187 (p=0.22), test 173/305 (p=2e-9, worse) |
| Fine-tuned AWF model (exp21) | **Indistinguishable** from confidence (0.0235 vs 0.0262, CI spans zero); captures more errors at a 20% budget (0.71 vs 0.63) |

**Mechanism (exp14).** A discrete boundary indicator computed from the
shift-0 hard prediction map alone is statistically indistinguishable from
aligned tile-phase across the 27 scenes (boundary better on 12, tile-phase
on 15, p=0.70). **No advantage of the perturbation beyond boundary proximity
is detectable** — which also means the signal is available at zero
additional inference cost.

By construction it is largest where neighbouring patches disagree, so
beating the pixel-edge control is the relevant test, and it does.

**Verdict: mixed.** The strongest constructed signal in the repository, and
the only one that ties confidence on a production model — but it does not
beat confidence on dense hand labels.

---

## Prediction-boundary proximity

**Definition.** Fraction of a patch's 8 neighbours whose hard label differs,
computed from the model's own prediction map. No second inference pass.

**Errors concentrate at boundaries, on both references:** about 75% of error
patches against 20% of correct ones (exp14, exp16, exp18; 75% vs 21% on
Sen1Floods11 Bolivia, 73% vs 18% on test). On the fine-tuned model, boundary
share among errors is 0.63 against 0.34 among correct points (exp21). The
phenomenon replicates label-free on the served product (exp20).

**But it does not order errors better than confidence.** On AWF it loses
(0.0636 vs 0.0363), and exp16 explains why: on a nine-class task the argmax
flips wherever margins are small, so the boundary score is a coarse proxy
for low confidence that the margin already carries at finer resolution
(Spearman 0.60; the margin dominates a logistic model of error, standardized
coefficients 3.53 vs 0.52).

**Verdict: supported as a triage cue, not as a ranker.** It says *where*
errors live. It is also the only label-free cue available on the served LCC
product, which exports no class confidence (exp20).

---

## Band-set disagreement (internal ensemble)

**Definition.** Heads trained separately on the 10 m, 20 m and 60 m
Sentinel-2 band-set tokens of the same patch; the signal is the standard
deviation of the three water probabilities. One forward pass, one model.

- **WorldCover (exp17):** beats the baseline 21/27 (p=0.006) and the pixel
  control 16/27. Outperforms the two-model E_case (10/27).
- **Hand labels (exp18):** worse than confidence (Bolivia 111/239, p=7e-12;
  test 148/334).

**Why it works better than a second model.** Its errors correlate *more*
with the final head (phi 0.75) than Nano's do (0.61), yet it ranks errors
better. A useful disagreement partner needs a different view of the input,
not decorrelated errors.

**No v1.2 counterpart:** v1.2 tokenizes Sentinel-2 as a single band-set
token per patch (exp19).

**Verdict: mixed.** Real against WorldCover, does not transfer.

---

## Depth-probe disagreement

**Definition.** The water head retrained on each block's tokens; standard
deviation over the last six blocks.

19/27 vs baseline (p=0.052), 13/27 vs control (exp17).

**Verdict: partial.** Marginal, and untested on expert labels.

---

## Embedding dissimilarity (E_dist)

**Definition as implemented.** Mean cosine distance from a window's Base
embedding to its k=5 nearest patches of the head's training window (one
128x128 scene at Katima Mulilo, or the AWF training split in exp12). **This
is distance to the head's training region, not to the encoder's pretraining
distribution** — an AOA-style reference sample over the pretraining domain
has not been built.

- In-domain it does not rank errors (0.00365 vs baseline 0.00089; exp03).
- On the 27 rule-selected scenes: 13/27 vs baseline, sign p=1.00. The
  mean-difference permutation p=0.01 is carried by a few high-error scenes;
  the scale-free sign test shows no advantage (exp13).
- On hand labels it is the worst signal tested (22/329; exp18).
- On AWF: 0.1338 against a 0.0363 baseline (exp12).

**Verdict: partial, no support as an error ranker.** Its
out-of-distribution-indicator interpretation remains plausible but untested:
that needs a shift testbed whose errors are not spectrally trivial, which
does not yet exist.

---

## Geographic grounding (E_geo)

**Definition.** Flag a patch that sits on an OSM `waterway=river` centerline
and that the model predicts dry.

- On a scene where the river is clearly resolved, zero false break alarms
  (52 centerline patches, 0 flagged; exp02).
- Across 27 georeferenced scenes (exp15): flags occur on 17 scenes with
  pooled precision 0.12 over 413 flags, against a base error rate of 0.081
  (1.5x enrichment). **9 scenes carry 1-51 flags at precision exactly
  zero**, where OSM marks a river that both the model and WorldCover call
  dry.
- As a ranker it beats the baseline on 3/27; prepending it to boundary
  proximity gives 5 better, 9 worse, 13 unchanged (p=0.42).

**Verdict: partial.** It mostly detects *disagreement between two reference
maps* on narrow channels, not model error. Its sensitivity cannot be
measured under WorldCover truth; it needs width-filtered GRWL centerlines or
expert labels.

![Full audit slice, Kazungula](../../exp/out/exp02_full_slice.png)

---

## Cross-model disagreement (E_case)

**Definition.** |p_Nano - p_Base| on the same window, or a local
similarity-structure comparison of their embeddings.

**Baseline floor:** mean agreement between Nano and Base on *random* input
is about 0.59, not 0, because the models share patchification and input
normalization. Agreement values must be read against this floor
(`exp/smoke_test.py`).

- Disagreement is spatially structured, concentrating at class boundaries
  and thin structures at both the embedding and prediction levels (exp01,
  exp02).
- WorldCover, 27 scenes: 10/27 vs baseline (p=0.25); intervals favour it on
  3 scenes and against on 9 (exp13).
- Hand labels: 84/265 (p=6e-23) on Bolivia, 141/340 on test (exp18).
- Equal-weight aggregation over three models performs worse than the best
  pairwise signal whenever one member is substantially weaker — three
  independent confirmations (exp03, exp04, exp07).
- **Rater strength does not help** (exp10): v1-Large as Base's partner is
  worse than Nano (0.0197 vs 0.0129 mean AURC, better on 3/7) although Large
  is more accurate on every scene.

**Verdict: not supported.** Within one family, strong models agree on
errors. The informative property of a partner is a different view of the
input, not accuracy or decorrelation — which is why band-set disagreement
beats it.

![Embedding-level agreement, Kazungula](../../exp/out/exp01_zambezi_agreement.png)

---

## Internal-state signals

**Definitions.** Decision settling (the final head applied to every block's
tokens, logit-lens style; std over the last six); representation drift
(cosine change between consecutive late blocks); last-block attention
entropy.

0/27, 3/27 and 3/27 vs baseline on the rule-selected scenes (exp17).

**Verdict: rejected.** INSIDE-style hidden-state probing does not transfer
from language models to this setting.

---

## Masking perturbation

**Definition.** Occlude a random 15% of patch cells with mean-fill, measure
prediction standard deviation over N=32 reruns.

Worst signal on all three scenes tested (0.0027 / 0.0788 / 0.0449 against
tile-phase 0.0008 / 0.0555 / 0.0076; exp08).

**Verdict: rejected**, with a design rule that generalizes: *perturbations
that preserve scene content while changing the tokenization expose model
pathology; perturbations that remove content measure context reliance
instead.*

---

## Label-free reliability estimation (Dawid-Skene)

**Definition.** Dawid-Skene EM over Nano/Tiny/Base votes on AWF, labels
untouched, to estimate each model's accuracy without ground truth.

It **overestimates every model and inverts the ordering**: estimated
0.859 / 0.916 / 0.876 against measured 0.753 / 0.802 / 0.817 (exp07). DS
assumes conditionally independent raters; the family errs together, so
agreement-on-errors is read as competence.

**Verdict: rejected within a single model family.** The
estimate-minus-measured gap (+0.106, +0.114, +0.059) is itself informative:
it directly measures correlated-error mass per model. An out-of-family rater
(Clay or AnySat, both wrapped in
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals))
is the designed fix, untested.

---

## Signals on deployed artifacts

Summarized here; full detail in [comparisons.md](comparisons.md) section 4.

- **Served LCC rasters (exp20).** No class confidence is exported, so the
  baseline cannot run. Boundary fraction is the one available label-free
  cue: it captures a median 0.88 of WorldCover water disagreements at a 5%
  review budget. Change-probability ambiguity concentrates on flagged-region
  edges (2.7% of edge windows against 0.07% interior).
- **Fine-tuned AWF model (exp21).** Confidence remains the best supported
  ranker; tiling instability is indistinguishable from it; boundary, probe
  disagreement and the control are significantly worse. The model is
  overconfident (ECE 0.080), so a stated accuracy needs a coverage: 0.945 at
  80%, 0.919 at 90%.
- **Periodic artifacts (exp22).** Class boundaries and change-probability
  gradients are quantized to the encoder's 4-px patch lattice (19 of 20
  profiles, the weakest at p=3.9e-12; the control's best top peak reaches
  only p=0.006). No inference-window seams at 64-512 px; seams affecting
  5-10% of rows at 128 px would have been detected.

![Signal maps at Kazungula](../../exp/out/exp03_more_channels.png)

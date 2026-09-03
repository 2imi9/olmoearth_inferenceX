# Experiment notes

Chronological lab log. Standing conclusions live in docs/TECHNIQUES.md.

| Exp | Topic |
|---|---|
| smoke_test | checkpoint load; noise floor for cross-model agreement |
| exp01 | embedding-level agreement map, Kazungula |
| exp02 | full audit slice: heads, E_case, E_geo, baseline, risk-coverage |
| exp03 | tri-model, tile-phase, E_dist, v1_2 attempt |
| exp04 | AWF expert-label validation, official spatial split |
| exp05 | difficult scenes: wetland margins, domain shift |
| exp06 | no-model pixel-statistic controls |
| exp07 | Dawid-Skene label-free reliability (rejected within family) |
| exp08 | masking perturbation (rejected) |
| exp09 | seven-scene replication with control |
| exp10 | rater strength vs diversity (diversity wins) |
| exp11 | pre-registered 29-scene comparison with bootstrap + permutation stats |
| exp12 | fills the AWF table cells: E_dist and pixel control in-domain |
| exp13 | corrected statistics on the rule-selected set: aligned tile-phase, tie-aware excess AURC, block bootstrap, sign/permutation tests |
| exp14 | boundary ablation: tile-phase indistinguishable from prediction-boundary proximity |
| exp15 | boundary proximity + E_geo: no benefit from the conjunction; E_geo sensitivity unmeasurable under WorldCover truth |
| exp16 | boundary indicator on AWF: labelled patches not interior; score error-associated but a proxy for low margin; confidence wins |
| exp17 | evidence from inside the encoder: band-set disagreement 21/27 vs WorldCover; depth probes marginal; logit-lens, drift, attention entropy rejected |
| exp18 | Sen1Floods11 hand labels (spatial hold-out): confidence is best; tile-phase, band-set, boundary, E_case all equal or worse; WorldCover wins read as reference error |
| exp19 | v1 vs v1.2 (RoPE): instability larger under v1.2; single band-set token in v1.2; cross-version disagreement not useful |
| exp20 | first served production output (olmoearth_lcc): no class confidence exported; boundary captures 0.88 of water disagreements at a 5% budget; change ambiguity sits on flagged-region edges |
| exp21 | fine-tuned AWF model end to end from Ai2's checkpoint: accuracy 0.881 (Ai2 0.895); confidence best, tiling instability ties it; overconfident (ECE 0.080); 0.945 at 80% coverage |

## exp01 — first E_case map (2026-08-31)

Setup: one S2 L2A scene (Planetary Computer, 0-5% cloud, 2024 dry season),
128x128 @ 10m, Nano + Base embeddings (CPU, patch 4), per-patch correlation of
local cosine-similarity structure between the two models (radius 4).

Findings:
- Noise floor: ~0.59 mean agreement on pure random input (shared
  patchification/normalization). Agreement must be read against this, not 0.
- Mongu dryland window: mean 0.66. Disagreement tracks linear features and
  heterogeneous ground; homogeneous woodland ~0.9. One hard low-agreement
  horizontal band with no RGB counterpart - possible detector seam,
  E_system-flavored, unchased.
- Kazungula window (river in frame): mean 0.78. Open water agrees ~0.9+,
  disagreement concentrates on the shoreline and traces the bridge (thin
  structure). Consistent with the E_case hypothesis: cross-model disagreement
  clusters where segmentation errors would live (boundaries, thin structures),
  not randomly.

Next:
- Quantify: agreement vs distance-to-waterline / GRIT centerline overlay.
- Few-shot water head on both embeddings -> prediction disagreement (true
  E_case, not just representation structure).
- v1 vs v1_2 same-window comparison (needs olmoearth_pretrain pull for loader).

## exp02 — full minimal audit slice (2026-08-31)

Every pipeline stage present, minimal depth. Heads trained at Katima Mulilo,
evaluated at Kazungula (~110 km, spatial split). WorldCover 2021 water as weak
truth. OSM waterway=river centerline as E_geo reference (overpass mirrors:
mail.ru worked when main + kumi were down; HF_HUB_OFFLINE=1 for cached reruns).

Numbers:
- Nano head 0.973 / Base head 0.979 eval acc vs WorldCover. Base error rate 2.1%.
- E_case AURC 0.0011 vs max-softmax baseline 0.0009. BASELINE WINS on this scene.
- E_geo: 52 centerline patches, 0 consensus-dry. No river breaks, no false alarms.

Honest read: on an easy dry-season scene with a 400 m river, the model's own
confidence is enough and E_case adds nothing. This is the §5 baseline argument
made concrete on day one. The channels' claimed value is on hard cases
(correlated errors, narrow channels, flood season, wetland margins), so the
next experiment must be a deliberately hard scene, not another easy one.
Also: errors are isolated specks partly from 2021-labels-vs-2024-scene drift
(moving sandbars) - weak-truth noise, not all model error.

Next:
- Hard scene: Barotse floodplain in flood season, or a narrow (<100 m) reach
  where the river is subpixel at patch scale. Expect E_geo to activate there.
- Tri-model (add Tiny) for Dawid-Skene-shaped E_case.
- v1 vs v1_2 E_system on the same windows.

## exp03 — four techniques, one run (2026-08-31)

Same Katima/Kazungula pair. Results in docs/TECHNIQUES.md (ledger is the
authority). Headlines: tile-phase E_system beats max-softmax (first channel to
do it); naive tri-model std worse than pairwise (Tiny pollutes); E_dist doesn't
rank in-domain errors (OOD alarm, not error proxy); v1_2 loader blocked on
current olmoearth_pretrain checkout.

Separately, allenai HF datasets answer OQ4: olmoearth_lcc production COGs ship
binary-change prob + argmax + top-1 prob (encoder v1.2-Base), and AWF/mangrove
expert labels are public. Production output is HTTP range-readable.

## exp04 — AWF expert-label validation (2026-08-31)

Harness on real partner truth (details in ledger). Baseline wins on multiclass
in-domain errors; tile-phase easy-scene win did not transfer; weak-rater
effect replicated with Nano. Herbaceous wetland weakest class (50% recall).
Features cached in exp/out/exp04_feats.npz (5 passes x 1459 windows).

## exp05 — hard scenes and domain shift (2026-08-31)

Every channel beats max-softmax on both hard AOIs (Barotse floodplain
wetland margins, Zambezi delta mangrove shift). E_case 3x better on margins,
E_dist 18x better under shift. Weak-truth caveat: delta "errors" trace a river
WorldCover likely misses. Ledger has the full statement. Eval windows cached
in exp/out/exp05_cache.npz.

## exp06 — no-model image-statistic controls (2026-08-31)

Spectral variance, NDWI ambiguity, NDWI gradient vs the same errors.
Kazungula and Barotse claims survive (E_case keeps a margin over the best
control). The delta E_dist shift claim does NOT survive: trivial NDWI stats
rank those disagreements better. Claim withdrawn in ledger and README.
Secondary: no-model stats outrank max-softmax on both hard scenes.
Results: exp/out/exp06_controls.csv.

## exp07 — Dawid-Skene label-free reliability (2026-08-31, overnight)

Negative with a useful twist: DS overestimates all three models and inverts
the order (details in ledger). The DS-vs-measured gap measures correlated
error per model. Fix is an out-of-family rater. Tiny AWF features cached in
exp/out/exp07_tiny_feats.npy; Tiny val acc 0.805.

## exp08 — masking-perturbation ensemble, GPU (overnight)

First GPU experiment (RTX 5090 laptop, torch 2.7.1+cu128; 32 occlusion
reruns x 3 scenes in under two minutes). Clean negative: occlusion
instability is the worst signal everywhere; it measures context reliance,
not error. Design rule recorded in ledger: perturb tokenization, not
content.

## exp09 — multi-scene replication with controls, GPU (overnight)

Seven river scenes, full comparison + NDWI-gradient control (details in
ledger). Confidence best on 0/7; E_case 3 wins (all surviving the control);
tile-phase 2; control wins the two reference-omission scenes as exp06
predicted; E_dist 0 legitimate wins. First AURC numbers with spread.
Iteration notes: initial water-fraction filter wrongly excluded
high-error scenes (fixed to errors>=8); four initial AOIs missed their
rivers at 1.28 km precision and were re-aimed.

## exp10 — rater strength vs diversity, GPU (overnight)

|Large-Base| worse than |Nano-Base| (0.0197 vs 0.0129 mean AURC, 3/7
scenes) despite Large being more accurate everywhere. Disagreement partners
need decorrelation, not strength. Note: exp10's per-scene win counts use a
3-signal set (baseline, two pairs) and are not comparable to exp09's
5-signal win counts.

## exp11 — pre-registered statistical hardening (2026-09-01)

Rule committed before fetching; 29 scenes. Headline revisions: baseline best
on 6/29 (exp09's 0/7 superseded); E_case advantage does not generalize
(p=0.07 toward worse); tile-phase most frequent winner but mean-zero; E_dist
only significant mean improvement (p=0.019), reference-quality confounded.
No signal dominates -> regime identification becomes the central problem.
Seed test vacuous (deterministic heads). Stats in exp/out/exp11_stats.csv.

## exp12 — AWF missing cells (2026-09-01)

E_dist 0.1104 and spectral-variability control 0.1287 on the same 51 AWF
errors; both far behind baseline 0.0367. README table n/a cells filled.

## exp13 — corrected statistics on the 29-scene set (2026-09-01)

Refinement pass caught three methodological problems: unaligned tile-phase
in exp05/09/11, raw AURC not comparable across scenes, i.i.d. patch
bootstrap ignoring spatial autocorrelation. Recomputed from cached exp11
features. Aligned tile-phase beats baseline 27/29 (sign p<0.001), control
19/29; E_dist's exp11 significance was a scale artifact; E_case unchanged.
First robust, control-surviving positive result. Also corrected the E_dist
definition in docs (distance to the head's training scene, not pretraining
data). Results: exp/out/exp13_corrected_stats.csv.

## exp14 — boundary ablation of tile-phase (2026-09-01)

Aligned tile-phase is equivalent to a zero-cost prediction-boundary
indicator (neighbor label disagreement on the shift-0 map): 13/29
head-to-head, p=0.71; the indicator alone beats confidence 23/29. The
headline is restated as boundary proximity, not perturbation. Explains the
AWF loss (interior point labels). Results: exp/out/exp14_boundary_ablation.csv.

## exp15 — boundary proximity + E_geo (2026-09-01)

Georeferencing recovered and verified for 23/29 scenes. Geo-first-then-
boundary is worse than boundary alone (5/23, p=0.011); geo alone 3/23.
Geo flags 3x error-enriched but precision zero on 8 scenes: OSM-vs-
WorldCover disagreement on narrow channels. E_geo sensitivity still
unmeasurable under WorldCover truth. Results: exp/out/exp15_boundary_geo.csv.

## audit + corrections (2026-09-01)

A 204-agent adversarial audit confirmed 53 findings. Code fixes: tie-aware
AURC (expected value under random tie-breaking) and a negative-|logit|
baseline in exp13/14/15 and evidence.py; tie-excluding sign tests; exp14
now scores against the pixel control; kafue and luangwa (cache leftovers,
not rule-selected) excluded from exp13-15 (27 scenes); AWF loader now warps
the 20 m/60 m band groups onto the 10 m grid instead of stretching them
(exp04/07/12 re-embedded); rasterize_polyline floor fix. Text fixes: the
E-AURC 'scale artifact' explanation for E_dist withdrawn (the oracle
subtraction cancels in differences); 'equivalent'/'adds nothing' softened
to 'statistically indistinguishable'; exp15 'rejected' softened to 'no
benefit shown' (5/9/13 W/L/T); E_geo enrichment restated as pooled
precision; README scope narrowed to linear probes on frozen encoders;
boundary-concentrated error positioned against prior work; AWF point-label
explanation demoted to hypothesis. Audit record: exp/out/audit_wf_119095bf.json.
Post-correction headline: aligned tile-phase beats confidence on 26/27
(sign p=4e-07); boundary indicator 19/27 vs confidence (p=0.05, marginal),
22/27 vs control; the two are indistinguishable head-to-head (15/12, p=0.70).

## AWF re-embedding after loader fix (2026-09-01)

Band groups now warped onto the 10 m grid with georeferencing (WarpedVRT)
instead of stretched. Re-run on GPU: Nano 0.753, Tiny 0.802, Base
0.817 val accuracy; 63 Base errors; AURC baseline 0.0363, E_case
0.0670, tile-phase 0.0489, E_dist 0.1338, control 0.1658. DS
reliabilities: nano 0.859 (measured 0.753, gap +0.106), tiny 0.916 (measured 0.802, gap +0.114), base 0.876 (measured 0.817, gap +0.059).
All AWF numbers in README/docs updated from this run.

## exp16 — boundary indicator on the AWF point-label task (2026-09-01)

Dense Base head over each validation crop (GPU). Errors reproduce exp04
(63). Script adversarially reviewed (40 agents) before recording; the
review corrected the analysis: interior test against the within-window
reference (labelled patches slightly less boundary-like than random
patches, sign p=0.008), conditional test of the score given the margin
(LRT p=0.002, coefficient 0.52 vs 3.53), cluster bootstrap over
30 tasks. AURC confidence 0.0363 vs boundary 0.0636. The "no
boundary context" explanation is withdrawn; the score is a low-margin proxy
on nine classes. Results: exp/out/exp16_awf_boundary.csv, exp16_summary.json.

## exp17 — evidence from inside the encoder (2026-09-01)

Hooked v1-Base (12 blocks, 768-d, 12 heads, no registers; sequence H,W,T,S
verified). Band-set disagreement (heads on the three S2 band-set tokens):
21/6 vs confidence (p=0.006), 16/27 vs control, one pass, better than the
two-model E_case. Depth probes 19/8 (p=0.052). Logit-lens settling 0/27,
drift 3/24, attention entropy 3/24: rejected. phi with final head: Nano
0.61, band-set probe 0.75, so disagreement value is not error decorrelation
alone. Cache exp/out/exp17_internals.npz (float16 per-block tokens).

## exp18 — Sen1Floods11 dense expert labels (2026-09-01)

Head trained on 600 valid-split tiles; scored on Bolivia (held-out region,
351 tiles) and 800 test tiles. Confidence best everywhere. Tile-phase
163/187 (p=0.22) on Bolivia, 173/305 (worse) on test; band-set 111/239;
boundary 134/217; E_case 84/265; E_dist 22/329; control 79/271. Errors on
boundaries 75% vs 21% (phenomenon holds). Reading: the WorldCover
advantages were reference-error detection. Repository claims revised in
README, ledger, comparisons, signals, roadmap. Cache exp/out/exp18_feats.npz
(3.9 GB, ignored). Caveat: L1C chips through the L2A path.

## exp19 — v1 vs v1.2 (2026-09-01)

Isolated worktree + venv on olmoearth_pretrain main. v1 features match the
cache exactly. v1.2: rope_3d_mixed, one S2 band-set token per patch. Tile-
phase magnitude larger under v1.2 (0.046 vs 0.032; smaller on 6/31). Against
WorldCover both versions' tile-phase beats their confidence (26/1, 25/2), but
exp18 makes that reference-error detection. Cross-version disagreement:
6/21 (v1 errors), 18/9 n.s. (v1.2). Cache exp/out/exp19_feats.npz (ignored).

## exp20 - served production output (2026-09-02)

First assessment of one of Ai2's own outputs. Ten 512-px windows of
allenai/olmoearth_lcc (EPSG:3857, about 9.55 m) read by HTTP range requests
(oe_inferencex/lcc.py; GDAL vsicurl stalls on the signed CDN redirect).
Product facts from the dataset card: band 1 change probability; bands 4-5
land cover classes with no confidence; bands 6-7 are change-category head
probabilities (80 to 99% at 255 over all pixels because "none" dominates).
Water map against WorldCover water: boundary AURC below random at 6/6 sites
with water; capture 0.88 at 5% budget (median); boundary share 0.92 among
disagreements against 0.01 among agreements. Full-legend disagreement median
49%: legend semantics, not error. Change probability: median 2.7% flagged,
1.2% ambiguous; low confidence on flagged-region edges 2.7% against 0.07%
interior. Two mistakes fixed on the way: the first run used band 6 as a
class confidence (it is not), and the assessor's oracle had the wrong sign
(fixed in assess.py; verified oracle 0.05, anti-oracle 0.66 on synthetic
data). Cache exp/out/exp20_windows.npz (ignored). Product gap to report:
export a class-head confidence alongside bands 4-5. The dataset's
annotated change points were first proposed as the next testbed and then
withdrawn: the card states they trained the model, so they are in-sample.

## exp21 - fine-tuned AWF model end to end (2026-09-02)

The user pointed out the fine-tuned checkpoints are public. Loaded
allenai/OlmoEarth-v1-FT-AWF-Base (Lightning ckpt from rslearn) into our
pretrain encoder strictly (231/231 keys), replicated rslearn's wrapper
(mean-pool T and S, legacy month timestamps) and head (bilinear x4 then 1x1
conv). Validation split, 344 expert points, 16-px and 32-px crops, 4 shifts.
Accuracy 0.881 / 0.878 (Ai2 reports 0.895; patch-logit reading 0.898). Probe
of exp16: 0.817; 28 of 41 fine-tuned errors are also probe errors. Signals
(16 px): confidence AURC 0.0262, tiling instability 0.0235 (cluster
bootstrap over 30 tasks: CI [-0.0068, +0.0010], P(better) 0.93), boundary
0.0765, probe disagreement 0.0852, NDVI-tstd control 0.0937; oracle 0.0076,
random 0.119. Capture at 20% budget: confidence 0.63, tile 0.71. Calibration
ECE 0.080, top bin 0.93 accurate at 0.99 confidence; selective accuracy
0.945 at 80% coverage. First run had two bugs (probe CSV key prefix; patch
logits instead of bilinear pixel logits), fixed before recording. Cache
exp/out/exp21_stacks.npz (ignored). Next: the other four fine-tuned
checkpoints for generality.

# Experiment notes

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

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

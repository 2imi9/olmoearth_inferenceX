# Established facts

Non-signal findings: upstream sources, export formats, transfer
properties. Index at [TECHNIQUES.md](../TECHNIQUES.md).

### Heads and spatial transfer
- Linear heads on frozen embeddings transfer spatially: trained on one river
  reach, 97-98% accuracy against WorldCover on a reach ~110 km away.
  (exp/exp02)
- WorldCover-as-reference carries temporal drift (2021 labels vs 2024
  scenes), so a fraction of measured "model error" is label noise.

### Production inference exports (allenai/olmoearth_lcc)
- Source: the
  [olmoearth_lcc dataset](https://huggingface.co/datasets/allenai/olmoearth_lcc)
  backing the [OlmoEarth LCC product](https://olmoearth-lcc.allen.ai).
- The published continent-scale LCC run (encoder OlmoEarth-v1.2-Base,
  32768x32768-pixel UTM tiles) provides 9-band uint8 summary rasters: band 1
  binary-change probability (scaled 0-255), bands 2-5 argmax classes, bands
  6-7 the probability of the argmax class, bands 8-9 month-encoded change
  dates. The export format therefore carries partial probability information
  (top-1 score and one binary-head probability), not full per-class
  distributions. (Documented in the dataset README.)
- The rasters are cloud-optimized GeoTIFFs on public storage; windows can be
  read over HTTP without bulk download.
- Caveat for any validation against the olmoearth_lcc annotations: most
  collection phases used output-based labeling (model proposes candidates,
  annotators verify), so label locations are correlated with model beliefs.

### Infrastructure
- Encoders, loader, and eval wrappers come from
  [allenai/olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain);
  dataset tooling and the window format come from
  [allenai/rslearn](https://github.com/allenai/rslearn); per-project task
  configs live in
  [allenai/olmoearth_projects](https://github.com/allenai/olmoearth_projects).
- The olmoearth_pretrain base dependency set suffices for inference (torch,
  einops, huggingface_hub, numpy); CPU handles 128x128-pixel windows at
  patch size 4, and a single consumer GPU (tested: RTX 5090 laptop, torch
  2.7.1+cu128) makes perturbation ensembles and multi-scene sweeps
  interactive. (exp/smoke_test.py, exp/exp08, exp/exp09)
- All encoder checkpoints are public: v1 Nano/Tiny/Base/Large, v1_1
  Nano/Tiny/Base, v1_2 Nano/Tiny/Small/Base, plus fine-tuned variants (AWF,
  LFMC, Mangrove, ForestLossDriver, EcosystemTypeMapping). Multi-model and
  cross-version signals require no private infrastructure.
- Planetary Computer Sentinel-2 L2A and ESA WorldCover can be read onto a
  shared 10 m grid; OSM Overpass requires mirror fallback.

### Encoder internals (verified empirically, exp17 preparation)
- OlmoEarth v1-Base encoder: 12 pre-norm ViT blocks (norm1, attention,
  layer-scale, norm2, MLP, layer-scale), 768-d, 12 heads, no register
  tokens, no qk-norm; attention runs through PyTorch scaled-dot-product
  attention, so weights are not returned but the q and k projections are
  plain linear layers whose outputs can be hooked and the weights recomputed.
- The token sequence entering the blocks is laid out H, W, T, S
  (outer to inner), where S indexes the three Sentinel-2 band-set tokens per
  patch (10 m bands B02/B03/B04/B08; 20 m bands B05/B06/B07/B8A/B11/B12;
  60 m bands B01/B09). Verified to zero error against the official output
  by reshaping a hooked last-block output through the final LayerNorm.
  Every earlier experiment mean-pooled these three tokens; they are
  separately addressable.
- Per-block outputs are hookable with standard forward hooks under
  fast_pass=True (no masked-token removal, no packing), which is what makes
  layer-wise probes, logit-lens trajectories and attention statistics
  available from a single forward pass.

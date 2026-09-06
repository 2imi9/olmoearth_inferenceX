"""olmoearth_inferenceX: assess a prediction map without labels.

Torch-free assessment layer (docs/method/agent_integration.md):
  metrics   tie-aware AURC, excess AURC, error capture at a budget, selective accuracy, ECE
  signals   confidence, prediction-boundary indicator, aligned tile-phase, the no-model pixel controls, U+
  stats     exact sign tests, the one-vote-per-cluster test, sign-flip permutation, block and cluster bootstraps
  assess    a prediction or served class map -> review sets, operating points, reference scoring
  taskcard  what each fine-tuned model is; lcc: HTTP range reader for the served rasters
Encoder-dependent (extra "encoder"): evidence (linear heads), awf, data.
"""

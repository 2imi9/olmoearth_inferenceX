"""olmoearth_inferenceX: assess a prediction map, and measure how two inferences differ, without labels.

Torch-free assessment layer (docs/method/agent_integration.md):
  metrics   tie-aware AURC, excess AURC, error capture at a budget, selective accuracy, ECE
  signals   confidence, prediction-boundary indicator, aligned tile-phase, the no-model pixel controls, U+
  stats     exact sign tests, the one-vote-per-cluster test, sign-flip permutation, block and cluster bootstraps
  assess    a prediction or served class map -> review sets, operating points, reference scoring
  explain   why a review window is suspect: label-free cues with measured enrichment
  compare   how two inferences of the same scene differ: disagreement, where it sits, stability, the label bridge
  taskcard  what each fine-tuned model is; lcc: HTTP range reader for the served rasters
Encoder-dependent (extra "encoder"): evidence (linear heads), awf, data.
"""

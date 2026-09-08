# Diagrams with real rasters

Three TikZ diagrams of the audit, drawn on the repository's own data (scene okavango_80 from the exp11 archive with
its exp37 per-window table; Sen1Floods11 Bolivia tile 218 with its hand labels). Layered slanted planes with shadows
for rasters, geometric shapes for the model parts, few words.

| File | Shows |
|---|---|
| `audit_pipeline.tex` | Sentinel-2 bands in, frozen encoder and head, prediction / confidence / boundary layers, the review set at a 5% budget (boundary first) on the scene, the reasons card, the reviewer |
| `explanation_layer.tex` | one flood tile with the review set and the true errors, one flagged window traced through the four cue layers, the measured enrichment per cue as bars |
| `protocol.tex` | a candidate signal next to the model's confidence and the no-model control, scored on both references on identical windows; the verdict; labels grade, never train |

Build:

```bash
uv run python docs/figures/tikz/make_rasters.py      # writes rasters/*.png and the review-set / error window lists
cd docs/figures/tikz && for f in audit_pipeline explanation_layer protocol; do pdflatex -interaction=nonstopmode $f.tex; done
osascript -l JavaScript pdf2png.js audit_pipeline.pdf ../pipeline.png 3600   # and the other two; sips would blur (72 dpi base)
```

`style.tex` holds the shared macros: `\raster` (one slanted plane), `\stack` (three offset planes), `\cells` and
`\flatcells` (outline r/c windows of a grid on a plane), the `card`, `part` and `flow` styles. The window lists are
TeX macros written by the raster script (`rasters/scene_review.tex` defines `\sceneReview`, and so on).

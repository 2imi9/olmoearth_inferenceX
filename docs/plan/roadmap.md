# Open items and next use case

Index at [TECHNIQUES.md](../TECHNIQUES.md). This list is rewritten in place
as experiments close items; the chronology of what closed what is in
[../../exp/NOTES.md](../../exp/NOTES.md).

## Open items, in priority order

1. Out-of-family rater (Clay or AnySat, both wrapped in olmoearth_pretrain
   evals) for disagreement and Dawid-Skene: correlated errors invalidate
   within-family DS (exp07) and cap pairwise quality (exp10); an
   architecture-independent rater is required for both.
2. Verify the boundary and tiling-instability results on an expert-labelled
   dense map with few classes. The 27-scene support is WorldCover-referenced
   (exp13); the AWF per-patch test (exp16) showed that on nine classes the
   prediction-boundary score is a proxy for low margin, and withdrew the
   point-label explanation. A dense expert-labelled binary or few-class map
   is the missing testbed.
3. A domain-shift testbed with non-trivial errors and expert labels
   (candidate design: geographic-corner holdout within the AWF dataset). The
   delta scene is disqualified as evidence by the exp06 controls; E_dist's
   out-of-distribution-indicator role is untested until this exists.
4. E_geo with width-filtered GRWL centerlines (rivers wide enough for the
   reference to resolve), then re-measure flag precision; under OSM lines
   and WorldCover truth the flags mostly mark reference-map disagreement
   (exp15). E_geo sensitivity on a scene with a confirmed consensus break
   remains unmeasured.
5. An expert reference for the served change product. The olmoearth_lcc
   dataset's annotated points are the model's training labels (dataset
   card: "used to train the OlmoEarth LCC model"), so they cannot serve as
   a held-out reference; scoring the product on them would be in-sample.
   Options: independent change references over the served tiles (for
   example published deforestation or flood maps with dates inside the
   product's window), or a small hand-labelled set drawn without reference
   to the model's output. exp20 assessed the product only against
   WorldCover and label-free.
6. E_dist formalization: AOA/Dissimilarity Index (Meyer & Pebesma 2021) in
   place of raw k-NN distance to the head's training scene; delineate the
   overlap with SHRUG-FM (CVPR 2026 EarthVision) before claiming novelty.
7. Confirm whether Studio per-project exports match the olmoearth_lcc export
   format (partial probabilities).
8. Striping at tile scale in the served v1.2 product: measured (exp22).
   No inference-window seams at 64 to 512 px at the null rate, with
   detection limits of 5 to 10% of rows at 128 px; outputs are quantized
   to the 4-px patch lattice. Open: the same test on a v1-encoder product,
   which would show whether RoPE removed a seam signal that v1 had; no v1
   product is served.
9. Ask for a class-head confidence in the LCC export (top-1 minus top-2
   logit alongside bands 4-5); without it the recipe's primary signal
   cannot run on the product (exp20).
10. Generality of exp21 across fine-tuned checkpoints. Of the five public
    fine-tuned models, only AWF and Mangrove have public labelled datasets.
    Mangrove was examined and declined for the ranking questions: its model
    classifies 2 x 2-pixel windows with no spatial context (boundary and
    tiling signals are undefined there) and its split is a hash split of
    grid cells, not a spatial hold-out. LFMC, ForestLossDriver and
    EcosystemTypeMapping have no public labels. A second fine-tuned dense
    task with expert labels and a spatial split is still needed.

Closed by later experiments: confidence intervals and significance tests
(exp13: block bootstrap, exact sign tests, permutation tests); the boundary
+ E_geo conjunction (exp15: no benefit); the AWF point-label hypothesis
(exp16: withdrawn); the v1 vs v1.2 comparison (exp19); the first direct
audit of a served production window (exp20, HTTP range reads; the river
centerline part is subsumed by item 5); evaluating on the production model
rather than probes (exp21, the fine-tuned AWF checkpoint end to end).

## Second use case: change attribution between two inference results

Error ranking is one consumer of the signals. The same decomposition applies
to comparing two inference outputs (two dates, two model versions): a raw
diff mixes real surface change, model instability, low-consensus
predictions, and geographically implausible transitions. E_system
instability and E_case consensus at each date gate which diffs are
trustworthy; geographic and temporal plausibility rules constrain which
transitions are physically possible; plain image differencing is the
no-model control. Intended application: automated interpretation of what
changed over time (EO autoresearch), where a change narrative should only be
generated from diffs that survive this decomposition. Natural testbed: the
olmoearth_lcc production change product (change probability and
month-encoded dates) and its verified change/no-change points. Untested;
design only.

Update after exp17: band-set disagreement (heads on the encoder's three
Sentinel-2 band-set tokens) is a second supported signal (21/27 vs
confidence, 16/27 vs control, single pass). Depth-probe disagreement is
marginal; logit-lens settling, representation drift and attention entropy
are rejected. Next for internal evidence: band-set disagreement on the AWF
expert task and on the production LCC encoder (v1.2, once loadable); test
whether band-set and tiling instability flag different errors before any
rank aggregation is considered.

Update after exp18: the water-task audit signals do not beat confidence on
Sen1Floods11 hand labels; the WorldCover-referenced advantages were read as
detection of reference error. Update after exp23: that reading is not
supported by WorldCover's own version instability (about 10% of
disagreements; the advantage is unchanged on stable patches). Update after
exp24: the year gap does not explain it either (2021 imagery, 2021 map,
within-year head: tiling instability still wins). Update after exp25:
seasonal water does not explain it either (without seasonal patches
tiling instability still wins 22/2/0 and 22/1/0). Priorities: (1) the hand
check: exp26 prepared the kit (top-12 disagreements by tiling instability
and by confidence on shire_80, barotse and okavango_80, with crops, cues
and blank verdict columns in exp/out/exp26_handcheck_*.csv); a reviewer's
verdicts are the input to the next analysis; (2) test whether any signal helps confidence at fixed review budget
on expert labels (operating-point analysis) rather than AURC; (3) evaluate
on the production model rather than probes; (4) other dense expert sets in
Ai2's evaluation suite (MADOS, PASTIS, GeoBench cashew/SA-crop) for
generality of the negative result.

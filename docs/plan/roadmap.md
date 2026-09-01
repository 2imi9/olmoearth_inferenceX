# Open items and next use case

Index at [TECHNIQUES.md](../TECHNIQUES.md).

## Open items, in priority order

1. Out-of-family rater (Clay or AnySat) for disagreement and Dawid-Skene:
   correlated errors invalidate within-family DS (exp07) and cap pairwise
   quality (exp10); an architecture-independent rater is required for both.
2. A domain-shift testbed with non-trivial errors and expert labels
   (candidate design: geographic-corner holdout within the AWF dataset). The
   delta scene is disqualified as evidence by the exp06 controls.
3. E_geo sensitivity: evaluate on a scene containing a confirmed consensus
   break and measure whether the centerline check detects it.
4. v1 vs v1_2 comparison on identical windows; requires updating the
   olmoearth_pretrain checkout for the v1_2 loader.
5. Confidence intervals and significance testing for AURC comparisons, now
   that seven scenes exist for the water task.
6. E_dist formalization: AOA/Dissimilarity Index (Meyer & Pebesma 2021) in
   place of raw k-NN distance; delineate overlap with SHRUG-FM before
   claiming novelty.
7. Confirm whether Studio per-project exports match the olmoearth_lcc export
   format (partial probabilities).
8. Replace OSM centerlines with GRIT, which adds width attributes that E_geo
   can condition on.
9. Audit a window of the published LCC production output directly (HTTP
   range reads) against river centerlines; first step toward the
   change-attribution use case.

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

Update after exp13: confidence intervals and significance tests now exist
(block bootstrap, sign and permutation tests). Remaining statistical gap is
the reference-label confound, not the test machinery. New item: verify the
aligned tile-phase result against expert labels (AWF windows under
sub-patch shifts, per-window) since its 29-scene support is
WorldCover-referenced.

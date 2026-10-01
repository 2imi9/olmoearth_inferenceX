# Do the package's three outputs hold on Ai2's other fine-tuned models? (exp89 preregistration)

**Status: DRAFT, not frozen.** Written 1 October 2026. The owner confirmed every threshold as proposed the same day,
chose to run arm A (FT-AWF, graded on P2 and P3 only) and chose a torch MLP for K3 (no new dependency). No
Mangrove or Nandi data has been downloaded, and no model has been run on them.

- **Allowed before freezing:**
  - the smoke tests on synthetic data;
  - the inventory job: it lists the Mangrove tar and the checkpoint keys, and counts windows and labels per split. It runs no model on a validation window;
  - the model smoke on training windows only. It is in-sample and never graded.
- **Not allowed before freezing:** any prediction on a validation window. That includes the accuracy gate.

## The question

The package makes three claims about a model's map:

1. **Review order.** The model's confidence ranks the windows to check, so a small review finds many errors.
2. **Estimate.** A designed sample of labels gives the error rate, with an interval that covers it.
3. **Certify.** A random sample of labels certifies the most confident part of the map at an error rate α, with a guarantee.

On Ai2's fine-tuned models the record tests only the first claim. It does so on one model (FT-AWF, exp21), against a no-model control that was near chance. exp89 asks whether all three claims hold on Ai2's other public fine-tuned models. It uses an informative control and takes Ai2's validation labels as the truth.

## What the record holds, and what it lacks

- **exp21 (FT-AWF).** The replica reached 0.881 on Ai2's 344 validation points, against Ai2's 0.895, with 41 errors. Confidence caught 22% of the errors in a 5% review and 39% in a 10% review (claim fine-tuned-capture-at-budgets).
  - Its only no-model control was the temporal standard deviation of NDVI. Its AUROC is 0.553, recomputed on 1 October from the committed `exp/out/exp21_finetuned_awf.csv`. So "beats the control" on AWF says little (red team, 30 September).
  - exp21 was not preregistered.
- **The suite's controls are near chance** (red team). They close a median 0.094 of the gap from random to perfect. The margin's AUROC is above 0.5 in all 3,560 exp79 cells. Beating random is therefore expected and shows little.
- **The estimate's designs were graded only at error rates of 5.5% and above.** exp78's tasks ran from 7.4% to 34.7%, and exp79's from 5.5% (checked in `exp/out/exp78_summary.json` and `exp79_summary.json`). A map at 2% to 3% error is untested. exp80's zones went as low as 1.8% (m_eurosat).
- **Every graded estimate and zone used probes on frozen embeddings,** never a fine-tuned model.

## Which models, and why

Checked on 1 October 2026 with HF API calls only.

- **Mangrove is public.**
  - Model `allenai/OlmoEarth-v1-FT-Mangrove-Base` at 57ee738b; `model.ckpt` is 1,041,362,118 bytes.
  - Dataset `datasets/allenai/olmoearth_projects_mangrove` at 3a878b8a; `mangrove.tar` is 62,433,280 bytes and `annotation_features.geojson` 33,933,316 bytes.
  - Ai2's doc links the dataset without `datasets/` in the path. That link returns 401.
- **Nandi is not public.** The model `allenai/OlmoEarth-v1-FT-Nandi-Base` and the dataset `datasets/allenai/olmoearth_projects_nandi` both return 401. Ai2's doc links them anyway.
- **Mangrove was declined before.** `docs/plan/roadmap.md` item 6 declined it for the ranking questions, for two reasons:
  - it classifies 2x2-pixel windows with no spatial context, so the boundary and tiling signals are undefined;
  - its split assigns 2x2 cells by hash, so it does not hold out space.

  exp89 does not test the boundary or tiling signals. It tests the model's confidence, the estimate and the certificate, and none of these needs spatial context. The hash split stays a limit (below). Mangrove does not meet the roadmap's need for "a second fine-tuned dense task with expert labels and a spatial split". Nandi would meet part of that need.

exp89 has three arms:

- **Arm M, Mangrove.** Graded on P1 to P5.
- **Arm N, Nandi.** Frozen with this page under the same thresholds. It runs only if Ai2 publishes the checkpoint and the dataset; until then it is reported as not run. Its thresholds do not change after arm M's result.
- **Arm A, FT-AWF (optional, owner to decide).** exp21's confidence result has already been seen, so arm A is graded on P2 and P3 only. P2 asks whether the informative control is informative there, and P3 whether confidence beats it.

## Design

### Data and units

- **Mangrove.**
  - Population: the windows of group `sample_100K` tagged `split=val` in `mangrove.tar` at the pinned revision, after the label rules below. The size is unknown until the inventory runs. It is about 12,500 if the group holds 100,000 windows (not verified).
  - Unit: one 2x2-pixel window at 10 m, which is one point and four pixels. The model gives one prediction per window.
  - Labels: Global Mangrove Watch v4 reference points (Zenodo 17394267, CC BY 4.0). The classes are 1 mangrove, 2 water and 3 other; 0 is invalid.
  - Split: Ai2 hashes each 2x2 cell into train (87.5%) or val (12.5%). Validation windows therefore have training neighbours.
- **Nandi (if published).**
  - Population: group `spatial_split`, `split=val`. The size is unknown.
  - Unit: one 63x63 window holding one labelled pixel. This is inferred from AWF, which uses the same preparer; it is not verified.
  - Labels come from three sources:
    - points from 819 CGIAR/IFPRI polygons;
    - water and built-up points derived from WorldCover;
    - tree points annotated in Studio.
  - Class indices (from `olmoearth_run.yaml`): 0 coffee, 1 grassland, 2 trees, 3 maize, 4 sugarcane, 5 tea, 6 vegetables, 7 legumes, 8 water, 9 builtup, and 10 nodata.
  - Split: 128-px (1.28 km) cells, 75/25.
- **AWF (arm A).** exp21's 344 validation points, unchanged.

### Clusters (for the bootstrap only)

- **Mangrove:** the 0.1° longitude/latitude cell. olmoearth_run partitions Mangrove requests on the same 0.1° grid. The 1° cell is reported. (confirmed)
- **Nandi:** the 128-px split cell. If the windows record a polygon id, cells that share a source polygon are merged. The 0.05° cell is reported. (confirmed)
- **AWF:** exp21's 30 annotation tasks.

The estimate and certify study draws from the finite validation set, so its coverage needs no clusters. The clusters matter only for the bootstrap of the ranking comparison.

### The replica: how each window's scores are computed

The replica does not use rslearn, as in exp21. The checkpoint's encoder keys are loaded strictly into OlmoEarth v1-Base (`olmoearth-pretrain` 0.1.0, revision 4bd1392a).

**Common to both models:**

- Input: 12 monthly Sentinel-2 L2A mosaics in the band order B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09, as stored in the tar. If the tar stores band groups at their native resolution, the 20 m and 60 m groups are resampled bilinearly onto the 10 m grid, as exp21 did.
- Normalisation: OlmoEarth's computed statistics with std multiplier 2. This matches rslearn's `OlmoEarthNormalize` default, and Mangrove sets 2.0 explicitly.
- Timestamps: rslearn's legacy values, which are day 1, month index 0 to 11, year 2024. exp21 used day 15 and year 2023. The model smoke reports the largest logit difference between the two conventions.
- Encoder: run with `fast_pass=True`. Tokens are mean-pooled over timesteps and band sets, which is rslearn's `token_pooling` default.
- Precision: fp32 with TF32 off.

**Mangrove:**

- Input: the 2x2 block at the label, at patch size 2, so one token per timestep and band set.
- Head: the pooling decoder's amax over a 1x1 map (which does nothing), then `Linear(768, 4)`. The expected keys are `model.decoders.mangrove_classification.0.output_layer.{weight,bias}` (inferred, not verified).
- Output: four logits per window.

**Nandi:**

- Input: the 16x16 crop of the 63x63 window that puts the label pixel at row 8, column 8 (exp21's rule with shift 0), at patch size 1. Ai2's validation used a random crop; ours is fixed.
- Head: a 1x1 conv from 768 to 11 channels, with no upsampling (scale factor 1). The expected keys are `model.decoders.segment.1.layer.{weight,bias}`, by analogy with AWF (not verified).
- Output: 11 logits, read at the label pixel.

### Prediction and confidence

- **Prediction:** the argmax over all output channels, as rslearn takes it. A window predicted as an untrained channel (Mangrove 0, Nandi 10) counts as an error. The count is reported; it is expected to be zero.
- **Confidence, graded:** the top-1 softmax probability over the trained channels only (Mangrove 1 to 3, Nandi 0 to 9). It is computed from the logits without ties: the package's `form="top1"`, through `assess_prediction` at patch 1. The reason: exp76 found top-1 better than the margin on 14 of 16 multi-class tasks, and the package warns about the margin for multi-class logits. (confirmed)
- **Reported, not graded:**
  - the logit margin, which is the default on the command line;
  - the entropy;
  - for Nandi, context-shift instability: the spread of the shift-0 class's probability over crop shifts of 0 to 3 px. At patch size 1 there is no sub-patch lattice, so this is not exp21's tiling instability.

### Controls

All controls are fixed here and computed from the same 12 mosaics. None is tuned on the validation set.

- **K1, random order.** The baseline, computed analytically.
- **K2, class rarity.** The training frequency of the predicted class; the rarer the class, the more suspect the window. This is the record's control.
- **K3, a no-encoder classifier. This is the informative control.**
  - Features, per month: the 12 normalised bands at the label pixel (Nandi) or the 2x2 block mean (Mangrove), plus NDVI. Mangrove also gets MNDWI (from B03 and B11).
  - Empty months: filled with the unit's median of that band over months. The count of empty months is also a feature.
  - Model: an MLP in torch, so no new dependency. Two hidden layers of 256, ReLU, dropout 0.1, Adam at learning rate 1e-3, weight decay 1e-4, batch 512, 30 epochs, seed 89. The last epoch is used, with no early stopping and no tuning.
  - Training data: the train split only.
    - Mangrove: up to 20,000 training windows drawn with seed 89. The cap is set from the fetch pilot before freezing, and is never below 5,000.
    - Nandi: all training windows.
    - Under Mangrove's hash split, K3's training windows are as close to the validation windows as the fine-tuned model's were.
  - Signals:
    - K3a: one minus K3's top-1 probability;
    - K3b: K3's disagreement with the fine-tuned model, binary and ranked tie-aware.
  - A gradient-boosted classifier would need scikit-learn added to the project. That is the owner's call, made before freezing.
- **K4, index controls.**
  - Mangrove:
    - inundation ambiguity min(f, 1 - f), where f is the share of valid months with MNDWI above 0;
    - how close the annual median MNDWI is to 0;
    - the monthly standard deviation of NDVI.
  - Nandi:
    - the temporal standard deviation of NDVI at the label pixel (exp21's control);
    - the 3x3 standard deviation of NDVI around the label pixel.
- **K5, cloud and missing data.** The number of months that are empty or whose B02 reflectance exceeds 0.2. (threshold: owner to confirm)

**The best informative control** is whichever of K2 to K5 has the lowest AURC on the validation set. Choosing it on the graded data favours the control, so the comparison is conservative for the package.

For certification the comparator is K3a, the best continuous control. A binary order would break its ties by index.

Two candidate controls are not used:

- distance to the edge of the GMW 2020 map, because that map was built from these same points;
- disagreement with WorldCover for Nandi, because WorldCover produced the water and built-up labels and was a pretraining target.

### Label handling

- **Labels are taken as right.** No label is checked or corrected. An "error" is a disagreement with Ai2's label, including where the label itself is wrong.
- **Mangrove:**
  - A window is kept when its valid label pixels all agree and none is 0. Other windows are dropped and counted.
  - The gate also computes Ai2's pixel-level micro accuracy over all valid label pixels.
- **Nandi:**
  - A window is kept when it has exactly one labelled pixel, with a class from 0 to 9.
  - Results are also reported by label source (CGIAR polygon, WorldCover, Studio trees) if the windows record it.
  - The Studio tree points were added to fix a known model error, so they are not a random sample.

### The estimate and certify study

The validation labels are the truth for the whole population of N windows, and θ is the share of errors. The only randomness is the draw of labels, as in exp78 and exp80.

- **Budgets:** B = 300 and B = 1,000 labels. A budget is graded only where B ≤ N/5. Where N < 1,500 no budget is graded, and the study is reported only.
- **Draws:** R = 2,000 per (arm, design, B). Each draw's seed comes from `np.random.SeedSequence([89, arm, design, B])`.
- **Estimate:**
  - Two designs: `random`, with the exact hypergeometric interval, and `confidence`, the package default (margin quintiles, Neyman allocation from the top-1 probability).
  - Both run through `sample_for_estimation` and `estimate_error_rate`, unchanged.
  - Per cell: the coverage of the 95% interval, the median width, the bias and the RMSE.
- **Certify:**
  - Call: `certify_zone(confidence, indices, wrong, alpha, delta=0.10, rule="prefix")` on the random design's draws. The Bonferroni rule is reported beside it.
  - α levels: Mangrove 0.02 (graded) and 0.01 (reported); Nandi 0.05 (graded) and 0.10 (reported). (confirmed)
  - Each graded α is below Ai2's reported error rate (2.4% and 12.7%). So the whole map cannot be certified, and only the order can help.
  - At δ = 0.10 a zone needs at least 114 labels at α = 0.02, 230 at 0.01, 45 at 0.05 and 22 at 0.10. So at B = 300, α = 0.01 cannot certify any zone smaller than 80% of the map.
  - Per cell:
    - the violation rate: the share of draws where the certified zone's true error rate exceeds α;
    - the median certified coverage;
    - the share of draws that certify nothing;
    - the best coverage possible, c*(α), from all labels.
  - The same is computed with K3a's order.
  - The review-set guard refuses a draw that looks enriched. A random draw almost always passes; refusals are counted.

### Review budgets and bootstrap

- **Review budgets:** 5% and 10% are graded, and 20% is reported. Capture is measured with the tie-aware `capture_at_budget_expected`.
- **Cluster bootstrap:** 2,000 resamples of the clusters, seed 89.
  - AURC differences use `cluster_bootstrap_difference`.
  - Capture differences use the same resampling, coded in the exp89 script.

## Smoke and the alignment gate

- **S1, numpy smoke.** Synthetic logits, labels and controls with known answers: one case where confidence beats the control and one where it does not. Every grading function must return the known verdict. It runs on the Mac and in the tests.
- **S2, model smoke.** Runs before freezing, on training windows only.
  - The checkpoint loads strictly, every key is listed, and the head has the expected shape.
  - One batch at patch size 2 (Mangrove) or 1 (Nandi) gives the expected output shape.
  - Accuracy on 512 training windows, drawn with seed 89, is at least 0.95 for Mangrove. This is in-sample and only checks the loading.
  - The untrained channel is never the argmax.
  - The largest logit difference between the two timestamp conventions is reported.
- **G, the alignment gate.** Runs after freezing, and reads accuracy only.
  - The gate job reads the validation windows and computes the predictions. It writes only the window count, the error count and the accuracy. It writes no confidence and no control.
  - **Mangrove passes** when our recomputed pixel-level micro accuracy is within 0.5 points of Ai2's 97.6% if the imagery came from the tar. If the imagery had to be fetched again, the tolerance is 1.0 point. (confirmed)
  - **Nandi passes** within 2.0 points of Ai2's 87.3%. exp21's AWF replica was 1.4 points from Ai2's figure. (confirmed)
  - The gate is two-sided: a replica far above Ai2's figure is as suspect as one below it.
  - A failed gate may be retried after a fix to the replica, at most three times. Each attempt is logged with its commit.
  - Nothing is graded until the gate passes.
- **Error floor.** The ranking predictions (P1 to P3) are graded only if the population holds at least 40 errors, exp21's level. Below that they are reported only.

## Measures

1. Accuracy (pixel micro and per window), per-class recall, the confusion matrix, the error count, and the number of clusters that hold an error.
2. For each signal (confidence, margin, and each control):
   - the AUROC for errors;
   - the capture at 5%, 10% and 20%, against random (0.05, 0.10 and 0.20) and against the best possible ranking (`attainable_ceiling`);
   - the excess AURC, and the share of the gap from random to perfect that the signal closes.
3. Confidence minus the best informative control: the capture at 5% and 10%, and the AURC, each with its cluster-bootstrap 95% interval.
4. Selective accuracy at 50%, 80%, 90% and 100% coverage, and the ECE over 10 bins of the top-1 probability.
5. The estimate study (above), per design and budget.
6. The certify study (above), per α and budget, for confidence and for K3a.
7. Reported by stratum, not graded: Mangrove by label class, and Nandi by label source.

## Predictions

All predictions are one-sided. Each is graded per arm, never pooled.

- **P1, the review order beats random by a useful margin.** In a 10% review, confidence captures at least 40% of the errors on Mangrove and at least 30% on Nandi. (confirmed)
  - For scale: random catches 10%, and exp21 caught 39% on AWF at 11.9% error.
- **P2, the informative control is informative.** The best informative control's AUROC for the model's errors is at least 0.65. (confirmed)
  - If P2 fails, P3 is still graded, but it is reported as a comparison with a weak control, which shows little.
- **P3, confidence beats the informative control (the question).** Both must hold (confirmed):
  - in a 10% review, confidence captures at least 5 points more of the errors than the best informative control;
  - the one-sided 95% cluster-bootstrap lower bound of that difference is above 0.
- **P4, at a low error rate the estimate still covers, and the confidence design gives a narrower interval.** Both must hold:
  - at each graded budget, the confidence design's 95% interval covers θ on at least 0.93 of the 2,000 draws (the package's bar, exp79);
  - at B = 300, its median width is at most 0.85 of the random design's. exp78's median was 0.80. (confirmed)
- **P5, certification is valid and useful.** At B = 1,000 (or at B = 300 where only that budget is graded), at the graded α, all three must hold:
  - the violation rate is at most 0.12, which is δ = 0.10 plus three Monte Carlo standard errors at R = 2,000;
  - the median certified coverage is at least 0.50 on Mangrove and at least 0.30 on Nandi (confirmed);
  - it exceeds the median coverage certified with K3a's order by at least 0.10 (confirmed).
- **Arm A, if run:** graded on P2 and P3 only, with the thresholds above.

## Readings fixed before freezing

- **What can fail by construction and what cannot.**
  - The random design's exact interval covers at least 95% by construction.
  - The prefix rule is valid on any map under a random draw. So P5's validity part checks the implementation more than it tests a hypothesis.
  - P4's coverage part is a real test. The stratified interval is not exact, and it has not been graded below 5.5% error.
- **What a pass on P1 alone means:** little. On the suite, confidence beats random everywhere. The informative result is P3 together with P2.
- **The one fact read early:** Ai2's reported accuracies (97.6% and 87.3%) set α and P1's thresholds. They are public figures, not our result.
- **Nandi's thresholds** are frozen now, before arm M runs, and do not change after it.

## What follows, whatever the outcome

- **The gate fails three times:** nothing is graded. exp89 records the gap and its likely cause (imagery, crop or timestamps). No claim is made.
- **P1 fails:** the review order is weak on that model. The README's statement about fine-tuned models stays at AWF and gets a scope note.
- **P2 fails:** the comparison is again with a weak control. The record says P3 shows little, and roadmap item 6 stays open.
- **P2 and P3 hold:** a new claim. On the named model, its confidence finds more errors in a 10% review than an informative no-encoder classifier does, by the measured amount. The README can then name two fine-tuned checkpoints: AWF with a weak control and Mangrove with an informative one.
- **P2 holds and P3 fails:** a cheap classifier with no encoder ranks this model's errors as well as the model does. The package's ranking adds nothing over it here, and the record and the docs say so.
- **P4's coverage fails:** the package's default design has a defect at low error rates. It is fixed in a later release and regraded on exp78, exp79 and here. Until then the docs warn below the tested error rate.
- **P4's width fails:** the confidence design gives no narrower interval on this map. The default stays, and the docs give the measured range.
- **P5's validity fails:** an implementation defect, which is fixed and regraded.
- **P5's usefulness fails:** at 1,000 labels, certification returns little on this map. The docs state what it costs.

## Limits stated in advance

- **Labels are assumed right.**
  - GMW points are labels from a reference product, not field checks.
  - Nandi mixes field polygons, points derived from WorldCover and points made in Studio.
  - Reviewer and label error are not modelled. In practice they break interval coverage.
- **Points are not independent.**
  - Nandi's points are sampled inside 819 polygons.
  - Mangrove's validation windows sit beside training windows because of the hash split, so they are close to in-sample.
  - The bootstrap resamples clusters. The estimate study is about the finite validation set, not about a mapped area.
- **One checkpoint per task,** chosen by Ai2 on this same validation set. Its reported accuracy is optimistic, and its errors are those of a selected checkpoint.
- **Ai2's own validation split is the population.** It is not a random sample of any map, so the estimate and certify results are statements about these points only.
- **Mangrove has no spatial context.** The boundary and tiling signals are not tested. The roadmap's reasons for declining Mangrove still hold for those questions.
- **The replica is ours, not rslearn's.** The gate bounds the difference in accuracy, not in which windows are wrong.
- **Package functions, not the command line.** Points are not a raster map. exp89 therefore calls `assess_prediction`, `sample_for_estimation`, `estimate_error_rate` and `certify_zone` directly. It does not run the `assess`, `sample`, `estimate` or `certify` commands end to end.
- **No correction for multiple predictions.** Each prediction is graded on its own and reported as such.

## Changes after freezing

- The predictions, thresholds, controls, clusters, budgets and seeds do not change once the gate has run.
- A bug in a reader may be fixed if the fix is shown on the smoke cases and every arm is regraded. The change is listed beside the verdict.
- A change of rule starts a new experiment.

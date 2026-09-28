# When a modality is missing: does the confidence fall with the accuracy? (exp88 preregistration, DRAFT)

**Status: draft, not frozen.** Written 28 September 2026. It is frozen before any result is computed. A smoke run on
synthetic embeddings is allowed before freezing; a run on real embeddings is not.

## The question

A model trained on radar and optical input (Sentinel-1 and Sentinel-2) is sometimes run with one of them missing,
for example when clouds cover the optical image. Its errors then rise. The question is whether its confidence falls
with them:

- **If it does,** the review order still points to the windows to check. Only the error estimate needs a sample from
  the same conditions.
- **If it does not,** the extra errors are confident ones. The review order misses them, and only a random sample
  drawn under the same conditions finds them. An error rate measured on clear days says nothing about a cloudy map.

The package has not tested this. Every result in the record ranks errors on a model evaluated with the inputs it
was trained on. The nearest evidence points both ways:

- **exp84:** the margin beats the best no-model control in every sensor group (S1, S2, Landsat, S1+S2). But each probe
  there was trained on its own sensor.
- **Nandi Sentinel-1:** the margin loses where the input carries little information (probe accuracy 0.40 or less).
- **Lehmann et al. 2026** (arXiv 2608.16614): geospatial foundation models grow more overconfident under corruption
  and shift, and confidence-based abstention fails there.

## Design

- **Data.** Ai2's published embeddings (allenai/olmoearth-paper-embeddings, exp54's cache). Three task families carry
  the same samples under three inputs: S1 only, S2 only, and S1+S2.
  - **PASTIS**, 19 classes, 458,638 windows. This is the primary task.
  - **CropHarvest China 6**, 4,397 samples. This is the replication.
  - **CropHarvest Togo 12**, 306 samples. It is reported and not graded, being too small and badly fitted
    (exp84).

  A smoke job first checks that the three inputs of each family line up sample by sample. exp63 checked PASTIS's S2
  against S1+S2.
- **Encoder.** OlmoEarth Base, as in the record. OlmoEarth Large is a replication if the smoke job finds its
  embeddings cached for all three inputs.
- **The probe** is trained on the S1+S2 embeddings of the train split with exp54's recipe (fp32, the task's learning
  rate, 50 epochs, seed 0), and nowhere else. The seeds 1 to 4 replicate it.
- **Three conditions on the test split,** with the same probe and the same windows:
  - **full:** S1+S2;
  - **optical missing (cloud):** S1-only embeddings;
  - **radar missing:** S2-only embeddings.

  This is how the encoder meets a missing modality: it encodes what it is given.
- **A mixed map.** On PASTIS, half of the test tiles are drawn at random (seed 88) to take the optical-missing
  embeddings and the rest keep full input. This is the cloudy part of one map.

## Measures (per condition)

1. **Error rate.**
2. **The margin of the errors and of the correct windows:** medians, and the shift from full input.
3. **Confident errors:** the share of a condition's errors whose margin is at or above the median margin of the
   correct windows under full input. These errors sit in the half of the map a reviewer would trust.
4. **Ranking within the condition:** the margin's AUROC for errors, and its excess over the best no-model control
   (embedding distance, exp74's definition), as in the record.
5. **On the mixed map:**
   - the share of the cloudy part's errors that fall in a 5% review set ranked by margin, against the cloudy part's
     share of all errors;
   - a random sample of 300 windows, with the error rate estimated pooled and stratified by condition: the pooled
     estimate's error for the cloudy part, and each stratum's interval coverage over 2,000 draws.

## Predictions (one-sided; thresholds to be confirmed by the owner before freezing)

- **P1, errors rise without optical input.** On PASTIS, the error rate under optical-missing exceeds full input's by
  at least 5 points.
- **P2, the confidence does not fall with them (the question).** The share of confident errors (measure 3) is at
  least 5 points higher under optical-missing than under full input.
  - If P2 fails (the share rises by less than 5 points), the confidence tracks the lost information, and the review
    order still points where to look. That outcome is reported as the answer, not as a failure of the experiment.
- **P3, the margin still ranks errors inside the cloudy part.** Under optical-missing, the margin beats the best
  no-model control on PASTIS and on CropHarvest China 6.
- **P4, a pooled estimate misstates the cloudy part, a stratified one does not.** On the mixed map:
  - the pooled estimate, read as the cloudy part's error rate, is off by at least 5 points;
  - the condition-stratified intervals cover each part's true rate in at least 93% of 2,000 draws (the package's
    coverage bar, exp79).

## What follows, whatever the outcome

- **If P2 holds,** the package says in its review output that the ranking is not valid across input conditions. It
  offers a design stratified by condition (cloud cover or modalities present) whenever the map records it. The
  certification is issued per condition.
- **If P2 fails,** the record says the margin tracks a missing modality, within the measured size. The per-condition
  estimate is still recommended (P4).

## Limits stated in advance

- **The embeddings of the missing-modality conditions are Ai2's single-sensor embeddings,** which are the encoder's
  reading of that input alone. Real clouds are partial and patchy. Whole-tile removal is the extreme case.
- **Three task families, crops only.** The question for water or land cover stays open.
- **Probes on frozen embeddings,** as everywhere in the record. A fine-tuned model may behave differently.

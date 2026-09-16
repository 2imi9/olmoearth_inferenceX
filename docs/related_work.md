# Related work

What this repository builds on, grouped by the question each group answers.
Each entry says what the work contributes and which experiment or claim here
touches it. Nothing on this page is claimed as ours; its purpose is the
opposite, to say where each idea comes from. Every entry was checked against
arXiv or Crossref before it went in.

## 1. Ranking a model's own errors

The core question here is selective classification: order predictions so the
least trustworthy come first, and measure the trade-off between coverage and
risk. That plain confidence is hard to beat is the field's default finding,
not a surprise, and the results here should be read in that light.

- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE
  Transactions on Information Theory* 16(1):41–46.
  [doi:10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406).
  The reject option; a review budget is a reject rate.
- Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep
  neural networks. *NeurIPS*. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500).
  Risk-coverage curves for deep networks.
- Geifman, Y., Uziel, G. and El-Yaniv, R. (2019). Bias-reduced uncertainty
  estimation for deep neural classifiers. *ICLR*.
  [arXiv:1805.08206](https://arxiv.org/abs/1805.08206). Introduces excess AURC,
  the statistic `oe_inferencex.metrics.excess_aurc` implements.
- Hendrycks, D. and Gimpel, K. (2017). A baseline for detecting misclassified
  and out-of-distribution examples in neural networks. *ICLR*.
  [arXiv:1610.02136](https://arxiv.org/abs/1610.02136). Maximum softmax
  probability as the baseline; the margin used here is the same family.
- Jaeger, P. F., Lüth, C. T., Klein, L. and Bungert, T. J. (2023). A call to
  reflect on evaluation practices for failure detection in image
  classification. *ICLR*. [arXiv:2211.15259](https://arxiv.org/abs/2211.15259).
  Finds that most proposed failure-detection methods do not beat maximum
  softmax response under AURC. The general-ML version of what exp49, exp50 and
  exp70 found on Earth-observation maps.
- Corbière, C., Thome, N., Bar-Hen, A., Cord, M. and Pérez, P. (2019).
  Addressing failure prediction by learning model confidence. *NeurIPS*.
  [arXiv:1910.04851](https://arxiv.org/abs/1910.04851). A learned confidence
  head; the label-fitted fusion of exp65 is the same move with a linear model
  and a family lock.
- Scheffer, T., Decomain, C. and Wrobel, S. (2001). Active hidden Markov
  models for information extraction. *Advances in Intelligent Data Analysis*,
  LNCS 2189. [doi:10.1007/3-540-44816-0_31](https://doi.org/10.1007/3-540-44816-0_31).
  Margin sampling, the top-1 minus top-2 rule, originates here.
- Lewis, D. D. and Gale, W. A. (1994). A sequential algorithm for training
  text classifiers. *SIGIR*.
  [doi:10.1007/978-1-4471-2099-5_1](https://doi.org/10.1007/978-1-4471-2099-5_1).
  Uncertainty sampling; reviewing by confidence is uncertainty sampling with
  the labels used to grade rather than to train.
- Rottmann, M., Colling, P., Hack, T. P., Chan, R., Hüger, F., Schlicht, P.
  and Gottschalk, H. (2020). Prediction error meta classification in semantic
  segmentation: detection via aggregated dispersion measures of softmax
  probabilities. *IJCNN*. [arXiv:1811.00648](https://arxiv.org/abs/1811.00648),
  [doi:10.1109/IJCNN48605.2020.9206659](https://doi.org/10.1109/IJCNN48605.2020.9206659).
  Segment-level error prediction from softmax dispersion; the closest prior
  framing to the window-level protocol on segmentation.
- Madras, D., Pitassi, T. and Zemel, R. (2018). Predict responsibly: improving
  fairness and accuracy by learning to defer. *NeurIPS*.
  [arXiv:1711.06664](https://arxiv.org/abs/1711.06664). Deferral to a person.
- Mozannar, H. and Sontag, D. (2020). Consistent estimators for learning to
  defer to an expert. *ICML*. [arXiv:2006.01862](https://arxiv.org/abs/2006.01862).
  The comparison task's "decline" answer is a deferral.

## 2. The signals tested against confidence

Every family below was scored here against the model's own margin and a
no-model control, and lost or tied
([Signals](results/signals.md), [Findings: what was tried and rejected](Findings.md#what-was-tried-and-rejected)).
The references say where each signal comes from; they are not evidence for it.

- Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
  representing model uncertainty in deep learning. *ICML*.
  [arXiv:1506.02142](https://arxiv.org/abs/1506.02142).
- Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
  [arXiv:1612.01474](https://arxiv.org/abs/1612.01474). Ensembles; rejected here
  as bag and probe-seed disagreement (exp49, exp50).
- Kendall, A. and Gal, Y. (2017). What uncertainties do we need in Bayesian
  deep learning for computer vision? *NeurIPS*.
  [arXiv:1703.04977](https://arxiv.org/abs/1703.04977). Aleatoric and epistemic
  uncertainty for segmentation.
- Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating
  predictive uncertainty under dataset shift. *NeurIPS*.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of
  modern neural networks. *ICML*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
  Calibration is not ranking; a temperature does not change the order.
- Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M. and
  Hennig, P. (2021). Laplace redux: effortless Bayesian deep learning.
  *NeurIPS*. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806). Laplace
  posterior over the head; rejected.
- Lee, K., Lee, K., Lee, H. and Shin, J. (2018). A simple unified framework for
  detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
  [arXiv:1807.03888](https://arxiv.org/abs/1807.03888). Mahalanobis feature
  typicality; rejected.
- Sun, Y., Ming, Y., Zhu, X. and Li, Y. (2022). Out-of-distribution detection
  with deep nearest neighbors. *ICML*.
  [arXiv:2204.06507](https://arxiv.org/abs/2204.06507). kNN typicality; rejected.
- Jiang, Y., Nagarajan, V., Baek, C. and Kolter, J. Z. (2022). Assessing
  generalization of SGD via disagreement. *ICLR*.
  [arXiv:2106.13799](https://arxiv.org/abs/2106.13799). Disagreement tracks the
  error rate in aggregate; it does not rank errors per window here.
- Milani Fard, M., Cormier, Q., Canini, K. and Gupta, M. (2016). Launch and
  iterate: reducing prediction churn. *NeurIPS*. Cross-version disagreement,
  the v1 against v1.2 comparison (exp19).
- Shanmugam, D., Blalock, D., Sahoo, G. and Guttag, J. (2021). Better
  aggregation in test-time augmentation. *ICCV*.
  [doi:10.1109/ICCV48922.2021.00125](https://doi.org/10.1109/ICCV48922.2021.00125).
  TTA; rejected (exp53).
- Bahat, Y. and Shakhnarovich, G. (2020). Classification confidence estimation
  with test-time data-augmentation. [arXiv:2006.16705](https://arxiv.org/abs/2006.16705).
  Augmentation consistency as confidence; the dihedral signal, rejected.
- Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of
  observer error-rates using the EM algorithm. *Applied Statistics*
  28(1):20–28. [doi:10.2307/2346806](https://doi.org/10.2307/2346806).
  Label-free reliability from several raters; rejected because heads of one
  family err together.

## 3. Why signals from inside the encoder carry no error information

The retargeted-residual line asked whether the pretraining objective's own
error could locate task errors. It could not, and the representation-learning
literature says why: a reconstruction or masking objective tracks the data's
variance subspace, which need not align with what a task needs.

- Esser, P. M., Fleissner, M. and Ghoshdastidar, D. (2025). Theoretical
  foundations of representation learning using unlabeled data: statistics and
  optimization. [arXiv:2509.18997](https://arxiv.org/abs/2509.18997). A survey:
  reconstruction learns principal components, which need not be the
  task-relevant directions; every downstream bound assumes a linear probe on
  the frozen representation, which makes the probe's own regularisation a
  separate factor (the exp68 warning).
- Balestriero, R. and LeCun, Y. (2024). Learning by reconstruction produces
  uninformative features for perception.
  [arXiv:2402.11337](https://arxiv.org/abs/2402.11337).
- Jing, L., Vincent, P., LeCun, Y. and Tian, Y. (2022). Understanding
  dimensional collapse in contrastive self-supervised learning. *ICLR*.
  [arXiv:2110.09348](https://arxiv.org/abs/2110.09348). The vocabulary for the
  frozen target's effective rank of 2.
- Torralba, A., Isola, P. and Freeman, W. T. (2024). *Foundations of Computer
  Vision*, chapter 30, Representation learning. MIT Press.
  [visionbook.mit.edu](https://visionbook.mit.edu/representation_learning.html).
  Compression against prediction; a representation good for one task can be
  bad for another.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P. and Girshick, R. (2022).
  Masked autoencoders are scalable vision learners. *CVPR*.
  [arXiv:2111.06377](https://arxiv.org/abs/2111.06377).
- Wei, Y. et al. (2024). Towards latent masked image modeling for
  self-supervised visual representation learning.
  [arXiv:2407.15837](https://arxiv.org/abs/2407.15837).
- Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J. and Auli, M. (2022).
  data2vec: a general framework for self-supervised learning in speech, vision
  and language. *ICML*. [arXiv:2202.03555](https://arxiv.org/abs/2202.03555).
- Hsu, W.-N. et al. (2021). HuBERT: self-supervised speech representation
  learning by masked prediction of hidden units. *IEEE/ACM TASLP*.
  [arXiv:2106.07447](https://arxiv.org/abs/2106.07447). The last three are the
  retarget recipes tried and rejected.

## 4. Reliability and evaluation of spatial and Earth-observation models

- Gonzalez-Calabuig, M. et al. (2025). SHRUG-FM: reliability-aware foundation
  models for Earth observation. [arXiv:2511.10370](https://arxiv.org/abs/2511.10370).
  Image-level reliability signals with a label-fitted gate. Ported to the
  window and scored here (exp49); the original does not run a single-model
  confidence baseline.
- Meyer, H. and Pebesma, E. (2021). Predicting into unknown space? Estimating
  the area of applicability of spatial prediction models. *Methods in Ecology
  and Evolution* 12:1620–1633.
  [doi:10.1111/2041-210X.13650](https://doi.org/10.1111/2041-210X.13650).
  Distance-to-training as a warning flag; the OlmoEarth Agent's uncertainty
  skill implements it.
- Meyer, H. and Pebesma, E. (2022). Machine learning-based global maps of
  ecological variables and the challenge of assessing them. *Nature
  Communications* 13:2208.
  [doi:10.1038/s41467-022-29838-9](https://doi.org/10.1038/s41467-022-29838-9).
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive
  performance of large-scale ecological mapping models. *Nature
  Communications* 11:4540.
  [doi:10.1038/s41467-020-18321-y](https://doi.org/10.1038/s41467-020-18321-y).
  Why the splits here are spatial.
- Milà, C., Mateu, J., Pebesma, E. and Meyer, H. (2022). Nearest neighbour
  distance matching leave-one-out cross-validation for map validation.
  *Methods in Ecology and Evolution* 13:1304–1316.
  [doi:10.1111/2041-210X.13851](https://doi.org/10.1111/2041-210X.13851).
- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E. and
  Wulder, M. A. (2014). Good practices for estimating area and assessing
  accuracy of land change. *Remote Sensing of Environment* 148:42–57.
  [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015).
  The design-weighted estimators in `oe_inferencex.metrics` follow this
  practice.
- Stehman, S. V. and Foody, G. M. (2019). Key issues in rigorous accuracy
  assessment of land cover products. *Remote Sensing of Environment*
  231:111199. [doi:10.1016/j.rse.2019.05.018](https://doi.org/10.1016/j.rse.2019.05.018).
- Tuia, D., Volpi, M., Copa, L., Kanevski, M. and Muñoz-Marí, J. (2011). A
  survey of active learning algorithms for supervised remote sensing image
  classification. *IEEE Journal of Selected Topics in Signal Processing*
  5(3):606–617. [doi:10.1109/JSTSP.2011.2139193](https://doi.org/10.1109/JSTSP.2011.2139193).
  Uncertainty sampling in Earth observation.
- Kampffmeyer, M., Salberg, A.-B. and Jenssen, R. (2016). Semantic
  segmentation of small objects and modeling of uncertainty in urban remote
  sensing images using deep convolutional neural networks. *CVPR Workshops*.
  [doi:10.1109/CVPRW.2016.90](https://doi.org/10.1109/CVPRW.2016.90). Early
  per-pixel uncertainty maps in Earth observation.

## 5. The models and the testbeds

- Herzog, H. et al. (2025). OlmoEarth: stable latent image modeling for
  multimodal Earth observation. [arXiv:2511.13655](https://arxiv.org/abs/2511.13655).
  The encoder, and the published task suite exp70 runs on.
- Lacoste, A. et al. (2023). GEO-Bench: toward foundation models for Earth
  monitoring. *NeurIPS Datasets and Benchmarks*.
  [arXiv:2306.03831](https://arxiv.org/abs/2306.03831). Six of exp70's tasks.
- Bonafilia, D., Tellman, B., Anderson, T. and Issenberg, E. (2020).
  Sen1Floods11: a georeferenced dataset to train and test deep learning flood
  algorithms for Sentinel-1. *CVPR Workshops*.
  [doi:10.1109/CVPRW50498.2020.00113](https://doi.org/10.1109/CVPRW50498.2020.00113).
- Chiriaco, F. et al. (2026). GEOID-Flood: a large-scale multi-modal benchmark
  dataset for flood segmentation. [arXiv:2608.02315](https://arxiv.org/abs/2608.02315).
  Copernicus EMS flood extents as chips; exp55, exp57, exp60, exp62.
- Mateo-Garcia, G. et al. (2021). Towards global flood mapping onboard low
  cost satellites with machine learning. *Scientific Reports* 11:7249.
  [doi:10.1038/s41598-021-86650-z](https://doi.org/10.1038/s41598-021-86650-z).
  WorldFloods.
- Portalés-Julià, E., Mateo-García, G., Purcell, C. and Gómez-Chova, L.
  (2023). Global flood extent segmentation in optical satellite images.
  *Scientific Reports* 13.
  [doi:10.1038/s41598-023-47595-7](https://doi.org/10.1038/s41598-023-47595-7).
  WorldFloods v2, the fourth cell of exp62.
- Yokoya, N., Ghamisi, P., Hänsch, R. and Schmitt, M. (2020). 2020 IEEE GRSS
  Data Fusion Contest: global land cover mapping with weak supervision. *IEEE
  Geoscience and Remote Sensing Magazine* 8(1).
  [doi:10.1109/MGRS.2020.2970124](https://doi.org/10.1109/MGRS.2020.2970124).
  DFC2020, exp66.
- Brown, C. F. et al. (2022). Dynamic World, near real-time global 10 m land
  use land cover mapping. *Scientific Data* 9:251.
  [doi:10.1038/s41597-022-01307-4](https://doi.org/10.1038/s41597-022-01307-4).
  The served product of exp67.
- d'Andrimont, R. et al. (2021). LUCAS Copernicus 2018: Earth-observation-
  relevant in situ data on land cover and use throughout the European Union.
  *Earth System Science Data* 13:1119–1133.
  [doi:10.5194/essd-13-1119-2021](https://doi.org/10.5194/essd-13-1119-2021).
  The 2022 release is exp68's reference.
- Schneider, M., Schelte, T., Schmitz, F. and Körner, M. (2023). EuroCrops:
  the largest harmonized open crop dataset across the European Union.
  *Scientific Data* 10. [doi:10.1038/s41597-023-02517-0](https://doi.org/10.1038/s41597-023-02517-0).
  exp69.
- Astruc, G., Gonthier, N., Mallet, C. and Landrieu, L. (2025). AnySat: one
  Earth observation model for many resolutions, scales, and modalities.
  *CVPR*. [arXiv:2412.14123](https://arxiv.org/abs/2412.14123). The outside
  witness, rejected.
- Zanaga, D. et al. (2021, 2022). ESA WorldCover 10 m 2020 and 2021. Zenodo.
  The reference of exp18 and exp23.

## 6. Whether the tool helps a person

- Bansal, G., Wu, T., Zhou, J., Fok, R., Nushi, B., Kamar, E., Ribeiro, M. T.
  and Weld, D. S. (2021). Does the whole exceed its parts? The effect of AI
  explanations on complementary team performance. *CHI*.
  [doi:10.1145/3411764.3445717](https://doi.org/10.1145/3411764.3445717). Why
  exp64 grades grounding and task accuracy separately.

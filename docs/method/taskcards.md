# Task cards

Resolved from configuration sources by `oe_inferencex/taskcard.py`; no evidence is generated here.

## allenai/OlmoEarth-v1-Base  (encoder)
- **encoder**: `{"version": "v1", "depth": 12, "embedding_size": 768, "num_heads": 12, "num_register_tokens": 0, "position_encoding": "absolute (no rotary keys in config)", "sentinel2_band_groups_per_patch": 3, "sentinel2_band_groups": [["B02", "B03", "B04", "B08"], ["B05", "B06", "B07", "B8A", "B11", "B12"], ["B01", "B09"]], "sentinel2_bands_used": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"], "supported_modalities": ["sentinel2_l2a", "sentinel1", "landsat", "worldcover", "srtm", "openstreetmap_raster", "wri_canopy_height_map", "cdl", "worldcereal"], "band_dropout_rate": 0.0, "run_name": "phase2.0_base_lr0.0001_wd0.02"}`
- **audit**: `{"band_set_disagreement_available": true, "tiling_instability_note": "absolute position encoding; striping artifact documented for v1"}`
- sources: https://huggingface.co/allenai/OlmoEarth-v1-Base/blob/main/config.json

## allenai/OlmoEarth-v1_2-Base  (encoder)
- **encoder**: `{"version": "v1.2", "depth": 12, "embedding_size": 768, "num_heads": 12, "num_register_tokens": 0, "position_encoding": "rope_3d_mixed", "sentinel2_band_groups_per_patch": 1, "sentinel2_band_groups": [["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]], "sentinel2_bands_used": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"], "supported_modalities": ["sentinel2_l2a", "sentinel1", "landsat", "worldcover", "srtm", "openstreetmap_raster", "wri_canopy_height_map", "cdl", "worldcereal"], "band_dropout_rate": 0.2, "run_name": "trope_mixed_tscale_months"}`
- **audit**: `{"band_set_disagreement_available": false, "tiling_instability_note": "rotary encoding did not reduce sub-patch instability (exp19)"}`
- sources: https://huggingface.co/allenai/OlmoEarth-v1_2-Base/blob/main/config.json

## awf  (project)
Goal: OlmoEarth-v1-FT-AWF-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting land use and land cover type in southern Kenya using Sentinel-2 satellite images.
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 10, "nodata_value": 9, "zero_is_invalid": false, "classes": {"0": "woodland_forest", "1": "open_water", "2": "shrubland_savanna", "3": "herbaceous_wetland", "4": "grassland_barren", "5": "agriculture_settlement", "6": "montane_forest", "7": "lava_forest", "8": "urban_dense_development"}}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 1, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "label": {"layers": ["label"], "n_timesteps": 1, "bands": ["category"], "is_target": true}}`
- **windows**: `{"window_buffer": 31, "window_resolution": 10.0, "nodata_value": 9, "grid_size": 128, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [10]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 10, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/awf/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/awf/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/awf.md

## ecosystem_type_mapping  (project)
Goal: OlmoEarth-v1-FT-EcosystemTypeMapping-Base is a model fine-tuned from OlmoEarth-v1-Base on expert-annotated ecosystem type data provided by [Global Ecosystem Atlas](https://globalecosystemsatlas.org/). It is trained specifically for the north Africa region. The categories correspond to those in the [IUCN Gloabl Ecosystem Typology](https://global-ecosystems.org/page/typology).
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 110, "nodata_value": 54, "zero_is_invalid": null, "classes": {}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2", "sentinel2.1", "sentinel2.2", "sentinel2.3", "sentinel2.4", "sentinel2.5"], "n_timesteps": 6, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "targets": {"layers": ["labels"], "n_timesteps": 1, "bands": ["tagname"], "is_target": true}}`
- **windows**: `{"grid_size": 0.1, "window_buffer": 31, "window_resolution": 10.0, "nodata_value": 54, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [110]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 110, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/ecosystem_type_mapping/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/ecosystem_type_mapping/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/ecosystem_type_mapping.md

## fields_of_the_world  (project)
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 2, "nodata_value": 3, "zero_is_invalid": null, "classes": {}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 1, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "targets": {"layers": ["label"], "n_timesteps": 1, "bands": ["label"], "is_target": true}}`
- **windows**: `{"grid_size": 1.0, "nodata_value": 2}`
- **outputs**: `{"decoder_out_channels": [2]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 2, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "band_set_disagreement_available": true}`
- warnings: ['docs/fields_of_the_world.md unavailable: HTTPError']
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/fields_of_the_world/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/fields_of_the_world/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/fields_of_the_world.md

## forest_loss_driver  (project)
Goal: OlmoEarth-v1-FT-ForestLossDriver-Base is a model fine-tuned from OlmoEarth-v1-Base for classifying forest loss drivers. It is trained to operate over [GLAD-S2 forest loss alerts](https://data.globalforestwatch.org/datasets/gfw::integrated-deforestation-alerts/about), which are updated weekly and report the locations of forest loss. Thus, instead of detecting forest loss from scratch, we take conne
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4}`
- **task**: `{"class": "ClassificationTask", "type": "classification (one label per window)", "classes": {"0": "agriculture", "1": "mining", "2": "airstrip", "3": "road", "4": "logging", "5": "burned", "6": "landslide", "7": "hurricane", "8": "river", "9": "none"}, "num_classes": 10}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["pre_sentinel2", "pre_sentinel2.1", "pre_sentinel2.2", "pre_sentinel2.3", "post_sentinel2", "post_sentinel2.1", "post_sentinel2.2", "post_sentinel2.3"], "n_timesteps": 8, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "targets": {"layers": ["label"], "n_timesteps": 1, "bands": null, "is_target": true}}`
- **outputs**: `{"decoder_out_channels": [10]}`
- **audit**: `{"output_is_dense": false, "boundary_cue_applies": false, "n_classes": 10, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/forest_loss_driver/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/forest_loss_driver/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/forest_loss_driver.md

## kenya_lulc_croptype  (project)
- **task**: `{"type": "unknown"}`
- **audit**: `{"output_is_dense": false, "boundary_cue_applies": false, "n_classes": null, "confidence_scoring": "n/a (regression)", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "band_set_disagreement_available": null}`
- warnings: ['model.yaml unavailable: HTTPError', 'olmoearth_run.yaml unavailable: HTTPError', 'docs/kenya_lulc_croptype.md unavailable: HTTPError']
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/kenya_lulc_croptype/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/kenya_lulc_croptype/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/kenya_lulc_croptype.md

## lfmc  (project)
Goal: OlmoEarth-v1-FT-LFMC-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting the live fuel moisture content of woody vegetation from Sentinel-2 and Sentinel-1 satellite images.
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4}`
- **task**: `{"class": "PerPixelRegressionTask", "type": "regression (dense per-pixel value)", "nodata_value": -1, "num_classes": null}`
- **inputs**: `{"sentinel1": {"layers": ["sentinel1", "sentinel1.1", "sentinel1.2", "sentinel1.3", "sentinel1.4", "sentinel1.5", "sentinel1.6", "sentinel1.7", "sentinel1.8", "sentinel1.9", "sentinel1.10", "sentinel1.11"], "n_timesteps": 12, "bands": ["vv", "vh"], "is_target": false}, "sentinel2_l2a": {"layers": ["sentinel2", "sentinel2.1", "sentinel2.2", "sentinel2.3", "sentinel2.4", "sentinel2.5", "sentinel2.6", "sentinel2.7", "sentinel2.8", "sentinel2.9", "sentinel2.10", "sentinel2.11"], "n_timesteps": 12, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "targets": {"layers": ["labels"], "n_timesteps": 1, "bands": ["value"], "is_target": true}}`
- **windows**: `{"window_buffer": 31, "window_resolution": 10.0, "nodata_value": -1, "grid_size": 128, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [1]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": null, "confidence_scoring": "n/a (regression)", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/lfmc/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/lfmc/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/lfmc.md

## mangrove  (project)
Goal: OlmoEarth-v1-FT-Mangrove-Base is a model fine-tuned from OlmoEarth-v1-Base for preddicting mangrove extent from Sentinel-2.
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 2}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 4, "nodata_value": null, "zero_is_invalid": true, "classes": {"1": "mangrove", "2": "water", "3": "other"}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **windows**: `{"grid_size": 0.1}`
- **outputs**: `{"decoder_out_channels": [4]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 4, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/mangrove/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/mangrove/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/mangrove.md

## mozambique_lulc  (project)
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 8, "nodata_value": null, "zero_is_invalid": true, "classes": {"1": "water", "2": "bareground", "3": "rangeland", "4": "floodedvegetation", "5": "trees", "6": "cropland", "7": "buildings"}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 1, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "label": {"layers": ["label_raster"], "n_timesteps": 1, "bands": ["label"], "is_target": true}}`
- **windows**: `{"window_buffer": 31, "window_resolution": 10.0, "nodata_value": 255, "grid_size": 128, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [8]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 8, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": null}`
- warnings: ['docs/mozambique_lulc.md unavailable: HTTPError']
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/mozambique_lulc/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/mozambique_lulc/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/mozambique_lulc.md

## nandi  (project)
Goal: OlmoEarth-v1-FT-Nandi-Base is a model fine-tuned from OlmoEarth-v1-Base for predicting crop and land-cover type across the Nandi county in Kenya using Sentinel-2 satellite images.
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 1}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 11, "nodata_value": 10, "zero_is_invalid": false, "classes": {"0": "coffee", "1": "grassland", "2": "trees", "3": "maize", "4": "sugarcane", "5": "tea", "6": "vegetables", "7": "legumes", "8": "water", "9": "builtup"}}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 1, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "label": {"layers": ["label"], "n_timesteps": 1, "bands": ["category"], "is_target": true}}`
- **windows**: `{"window_buffer": 31, "window_resolution": 10.0, "nodata_value": 10, "grid_size": 128, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [11]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 11, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": true}`
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/nandi/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/nandi/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/nandi.md

## satlas_solar_farm  (project)
- **encoder**: `{"model_id": "OLMOEARTH_V1_BASE", "patch_size": 8}`
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 2, "nodata_value": null, "zero_is_invalid": null, "classes": {}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **windows**: `{"grid_size": 0.15}`
- **outputs**: `{"decoder_out_channels": [2]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 2, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "band_set_disagreement_available": true}`
- warnings: ['docs/satlas_solar_farm.md unavailable: HTTPError']
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/satlas_solar_farm/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/satlas_solar_farm/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/satlas_solar_farm.md

## togo_cropland  (project)
- **task**: `{"class": "SegmentationTask", "type": "segmentation (dense per-pixel classes)", "num_classes": 8, "nodata_value": null, "zero_is_invalid": true, "classes": {}, "legend_note": "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"}`
- **inputs**: `{"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 1, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"], "is_target": false}, "label": {"layers": ["cropland_label_raster"], "n_timesteps": 1, "bands": ["cropland_label"], "is_target": true}}`
- **windows**: `{"window_buffer": 31, "window_resolution": 10.0, "nodata_value": 255, "grid_size": 128, "window_size_px": 63, "splitter": "SpatialDataSplitter"}`
- **outputs**: `{"decoder_out_channels": [3]}`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "n_classes": 8, "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties", "expert_reference_required": true, "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels", "note": "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate", "band_set_disagreement_available": null}`
- warnings: ['docs/togo_cropland.md unavailable: HTTPError']
- sources: https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/togo_cropland/model.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/olmoearth_run_data/togo_cropland/olmoearth_run.yaml, https://raw.githubusercontent.com/allenai/olmoearth_projects/main/docs/togo_cropland.md

## allenai/olmoearth_lcc (production change product)  (dataset)
Goal: Detect recent land cover change from Sentinel-2 time series (16 quarterly + 4 biweekly images), continent scale.
- **encoder**: `{"repo": "https://huggingface.co/allenai/OlmoEarth-v1_2-Base"}`
- **task**: `{"type": "land cover change (dense); per-pixel change probability plus argmax classes", "land_cover_classes": {"1": "bare", "2": "burnt", "3": "crops", "4": "fallow/shifting cultivation", "5": "grassland", "6": "Lichen and moss", "7": "shrub", "8": "snow and ice", "9": "tree", "10": "urban/built-up", "11": "water", "12": "wetland (herbaceous)"}, "nodata_value": 0}`
- **outputs**: `{"export_bands": {"1": {"name": "binary_change", "description": "Probability of change, scaled to 0-255."}, "2": {"name": "pre_class", "description": "Argmax class of the pre/same change-category head (see table below)."}, "3": {"name": "post_class", "description": "Argmax class of the post change-category head (see table below)."}, "4": {"name": "src_class", "description": "Argmax source land cover class (see table below)."}, "5": {"name": "dst_class", "description": "Argmax destination land cover class (see table below)."}, "6": {"name": "pre_score", "description": "Probability (0-255) of the argmax class in band 2."}, "7": {"name": "post_score", "description": "Probability (0-255) of the argmax class in band 3."}, "8": {"name": "ts_pre_month", "description": "Predicted change start (last pre-change date), month-encoded."}, "9": {"name": "ts_post_month", "description": "Predicted chang`
- **audit**: `{"output_is_dense": true, "boundary_cue_applies": true, "probabilities_in_export": "change probability (band 1) and the probabilities of the change-category heads (bands 6-7); no confidence for the land cover classes (bands 4-5) and no per-class distribution (exp20)", "confidence_scoring": "none available for the class map; |2p - 1| of band 1 for the change decision; boundary fraction of the class map as triage (exp20)", "band_set_disagreement_available": false, "note": "encoder v1.2-Base: single Sentinel-2 band-set token per patch (exp19)"}`
- sources: https://huggingface.co/datasets/allenai/olmoearth_lcc

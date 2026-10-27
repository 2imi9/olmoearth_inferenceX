"""Command line: a prediction raster in, the review set and its explanation out; two rasters in, their difference out.

    oe-inferencex assess  scores.tif --out DIR [--logits] [--patch 4] [--nodata V] [--reference labels.tif]
                          [--budgets 0.01 0.05 0.10] [--order confidence|boundary_first]
                          [--condition layer.tif [--condition-names 0=clear 1=cloudy]]
    oe-inferencex from-embeddings --points points.csv --bbox W S E N --year 2025 --out scores.tif   precomputed
                                                            embeddings + labelled points -> class probabilities
    oe-inferencex boundary scores.tif --out band.tif --width-m 20 | --width-px 2   the class-edge band, as a
                                                            condition layer for sample
    oe-inferencex compare a.tif b.tif --out DIR [--patch 4] [--nodata V] [--labels labels.tif] [--groups ids.tif]
                          [--threshold T]
    oe-inferencex demo    [--out DIR] [--made-up]   a first run on the real sample map shipped with the package
    oe-inferencex sample  scores.tif --budget 300 --out to_label.csv [--design D] [--condition layer.tif]
    oe-inferencex sample  scores.tif --design audit --per-class 20 --out audit.csv   each class's confident zone
    oe-inferencex estimate to_label.csv [--per-class] [--also-accept pairs.json]   once the reviewer has filled `wrong`
    oe-inferencex estimate audit.csv --alpha 0.05 [--also-accept pairs.json]   once `reference_class` is filled
    oe-inferencex certify  to_label.csv --alpha 0.05 [--also-accept pairs.json]   from a random sample, or one drawn
                                                            with --condition
    oe-inferencex grade points.csv --stratum S --reference R --map M [--map M2] (--weight W | --strata sizes.csv)
                          [--alternate A] [--pure 1=11 ...]   an agency's stratified sample -> accuracies with
                                                            intervals valid at any sample size
    oe-inferencex mcp                                       a local MCP server on stdio, for an agent (the mcp extra)
    oe-inferencex from-olmoearth DS --out DIR [--conditions]  an OlmoEarth run's rslearn dataset -> the rasters
                                                            assess reads, and the command that reads them

Inputs are GeoTIFFs (any rasterio-readable raster) or .npy arrays: (H, W) for a binary map, (C, H, W) for per-class
scores or, for `compare`, an integer class map. `compare` also takes two continuous maps (a regression output) when the
cut-off is named with --threshold; without it a map outside [0, 1] is refused rather than cut at 0.5. No-data comes
from the raster's own value, NaN, or --nodata. Outputs are
plain files the caller reads back: JSON summaries (assess.summary / compare's dict), CSVs of the review windows with
pixel and map coordinates, and rasters on the window grid (patch x patch pixels per window) when the input was one.
Nothing here narrates; the JSON is the evidence (docs/method/agent_integration.md in the research record).

The confidence ranking compares every window with every other. A model run on an input combination it was not trained
on can be confidently wrong there: on PASTIS a probe trained on radar plus optical and run on radar alone, as under
cloud, was sure and wrong, while a probe trained on radar alone ranked its errors normally (exp88). Even a model trained
on each input can be wrong more often under one input than under another. Without --condition, assess ranks all
windows together. --condition takes one integer band on the map's grid, each pixel's input condition (a cloud flag,
the modalities present, a sensor id). assess then also ranks each condition on its own, so that windows are compared
only with windows read from the same inputs. sample splits the labels equally across the conditions (with --design
random it only records them) and writes each window's condition to its sidecar, from which estimate gives each
condition's rate and certify each condition's zone. Without it, the JSON outputs carry `scope`, what a whole-map
result does not show.
"""
import argparse
import csv
from types import SimpleNamespace
import json
import math
import os
import sys

import numpy as np

from oe_inferencex.review.assess import (RULE_TEXT, _boundary_valid, _pool, _pool_valid, _pooled_argmax, assess_classmap,
                                         assess_prediction, pool_condition, summary)
from oe_inferencex import estimate as est
from oe_inferencex.compare import compare_inferences
from oe_inferencex.review.explain import explain_review_set
from oe_inferencex.common.signals import boundary_indicator
from oe_inferencex.common.condition import _condition_args, _condition_names, _read_condition
from oe_inferencex.common.labels import (_also_accept, _also_accept_rows, _also_words, _check_sidecar, _csv_rows,
                                         _labelled_sample, _population_note, _reference_column, _rounding_tolerance,
                                         _sample_map_inputs, _sidecar_by, _sidecar_design, _unused_note)
from oe_inferencex.common.maps import PROB_TOLERANCE, _check_scores, decisions
from oe_inferencex.common.output import (_check_out_dir, _check_out_file, _check_patch, _json_nodata, _nodata_arg,
                                         _num_arg, _pct, _readable_path, _refuse_overwriting_inputs, _same_file, _stamp,
                                         _where)
from oe_inferencex.common.product import (MAX_CLASS_ID, NO_RANGE_NOTE, PRODUCT_SIGNAL, _product_args,
                                          _product_parser_args, _read_product)
from oe_inferencex.common.raster import (_band_count, _gdal_masked, _mask_bands, _mask_hint, _read_masked, _same_grid,
                                         affine_apply, pool_valid, read_raster, window_coords, write_raster)
from oe_inferencex.common.wording import (_CLASS_WORDS, _PATH_KEYS, _PROFILE_WORDS, _RULE_WORDS, class_names,
                                          class_words, class_wording, profile_names, result_class_names)
from oe_inferencex.review.windows import _map_windows, _two_map_windows
from oe_inferencex.review.command import _budget_tag, add_assess_parser, add_boundary_parser, cmd_assess, cmd_boundary
from oe_inferencex.readers.command import (_skipped_text, add_from_embeddings_parser, add_from_olmoearth_parser,
                                           cmd_from_embeddings, cmd_from_olmoearth)
from oe_inferencex.scale.command import _window_size_text, add_scale_parser, cmd_scale
from oe_inferencex.agency.command import _iv_pct, _range_pct, add_grade_parser, cmd_grade
from oe_inferencex.compare.command import _window_majority, add_compare_parser, cmd_compare
from oe_inferencex.decide.command import _worded_print, add_decide_parser, cmd_decide
from oe_inferencex.plan.command import (_labels, _ladder_text, _plan_windows, _points, _reach, _sample_command,
                                        add_plan_parser, cmd_plan)
from oe_inferencex.certify.command import (CERTIFY_ALSO_ACCEPT_HELP, ZONE_ALSO_KEYS, _also_mask, _certify_by_condition,
                                           _certify_by_profile, _certify_sequential, _keep_to_plan, _profile_line,
                                           _tenth, _zone_also_accept, _zone_also_text, _zone_reading_note,
                                           add_certificate_parser, add_certify_parser, add_verify_parser,
                                           cmd_certificate, cmd_certify, cmd_verify)
from oe_inferencex.estimate.command_text import _zone_words
from oe_inferencex.estimate.sample_command import (CLASS_RULE_TEXT, FILL_INSTRUCTION, OTHER_DATES_NOTE, _certify_need_note,
                                                   _certify_plan, _extend_command, _kept_rows_mismatch, _reading_text,
                                                   _print_profile_sample, _profile_need_note, _sample_class_audit,
                                                   _sample_disagreement, _sample_map, _sequential_to_extend,
                                                   _sidecar_map_fields, _top1, _unrecorded_note, add_sample_parser,
                                                   cmd_sample)
from oe_inferencex.estimate.estimate_command import (ALSO_ACCEPT_HELP, _estimate_class_audit, _estimate_disagreement,
                                                     _outside_note, _rate, _reference_classes, _verdict_words,
                                                     add_estimate_parser, cmd_estimate)
from oe_inferencex.cli.demo_command import add_demo_parser, cmd_demo
from oe_inferencex.cli.mcp_command import add_mcp_parser, cmd_mcp


def build_parser():
    p = argparse.ArgumentParser(prog="oe-inferencex", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    add_assess_parser(sub)
    add_demo_parser(sub)
    add_from_embeddings_parser(sub)
    add_boundary_parser(sub)
    add_scale_parser(sub)
    add_grade_parser(sub)
    add_certificate_parser(sub)
    add_verify_parser(sub)
    add_compare_parser(sub)
    add_sample_parser(sub)
    add_estimate_parser(sub)
    add_certify_parser(sub)
    add_decide_parser(sub)
    add_plan_parser(sub)
    add_mcp_parser(sub)
    add_from_olmoearth_parser(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)

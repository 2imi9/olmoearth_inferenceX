"""exp89: do the review order, the estimate and the certificate hold on Ai2's other public fine-tuned models?

Preregistered in docs/plan/finetuned_checkpoints.md; read that page first, this file is the run. Three arms:
  mangrove  allenai/OlmoEarth-v1-FT-Mangrove-Base on the sample_100K validation windows, graded on P1 to P5;
  nandi     frozen under the same thresholds and not run: its checkpoint and dataset return 401 on the Hub. Every
            mode reports it as not run unless both become readable and their revisions are pinned here;
  awf       FT-AWF on exp21's 344 validation points, graded on P2 and P3 only.

    python exp/exp89_finetuned_checkpoints.py --smoke                      # S1: numpy only, synthetic units
    python exp/exp89_finetuned_checkpoints.py --smoke-torch                # S2 code on a synthetic tar and checkpoint
    python exp/exp89_finetuned_checkpoints.py --smoke-torch --real --arm mangrove   # S2 on 512 real TRAINING windows
    python exp/exp89_finetuned_checkpoints.py --inventory --arm mangrove   # files, keys, windows and labels per split
    python exp/exp89_finetuned_checkpoints.py --gate --arm mangrove        # accuracy only; refused until frozen
    python exp/exp89_finetuned_checkpoints.py --arm mangrove               # the full run: frozen AND gate passed

What is reused, not rewritten. Every statistic is the package's: the ranking measures are oe_inferencex.metrics
(aurc_expected, excess_aurc, oracle_aurc, capture_at_budget_expected, attainable_ceiling, weighted_auroc with unit
weights as exp70 reads the AUROC, selective_accuracy, expected_calibration_error); the AURC bootstrap is
oe_inferencex.stats.cluster_bootstrap_difference, and the capture bootstrap draws the same resamples in the same order;
the estimate and the certificate are oe_inferencex.estimate (sample_for_estimation, estimate_error_rate,
exact_coverage_srs, certify_zone, zone_order, min_labels_to_certify), called unchanged. The confidence is
oe_inferencex.assess.assess_prediction at patch 1 with form "top1" on the trained channels; the logit margin is the
same call with form "margin". Arm A reads its windows with oe_inferencex.awf (list_windows, load_window_full) and runs
exp21's forward pass (logits_grid) unchanged. Its crop is this file's crop_at, which applies exp21's rule at H = W = 63
(tested against exp21's numbers). Only the checkpoint is loaded here, at the pinned revision, and the encoder is built
from OlmoEarth v1-Base's config.json at the revision the page names (ENCODER_REVISIONS).

Interpretations of the plan page, each flagged in the commit that added this file; none changes a threshold:
  a. Mangrove's label rule "a window is kept when its valid label pixels all agree and none is 0" is read as: all
     pixels of the 2x2 block (after rslearn's centre Pad to 2) carry the same class, from 1 to 3. The gate's pixel
     micro accuracy is over every valid (non-zero) label pixel of every validation window with imagery, kept or not,
     as Ai2's metric counts them.
  b. The logit margin is taken over the trained channels, like the graded top-1 probability, so the estimate's
     confidence design stratifies by it (margin quintiles) and allocates by the top-1 probability, as the command line's
     default `sample --logits` does. A window whose top two include the untrained channel is counted and reported.
  c. The best informative control is the lowest-AURC signal among K2, K3a, K3b, each K4 index and K5, each counted
     as one candidate.
  d. P3's one-sided 95% lower bound is the 5th percentile of the cluster bootstrap of the capture difference at 10%.
  e. The estimate draws: SeedSequence([89, arm, design, B]).generate_state(R) gives one seed per draw; arm is
     mangrove 0, nandi 1, awf 2 and design random 0, confidence 1. The certificate reuses the random design's draws.
     A draw whose zone is certified and whose zone's true error rate exceeds alpha is a violation; a draw that
     certifies nothing, or that the review-set guard refuses, counts as coverage 0 and no violation (refusals are
     counted). c*(alpha) is the largest grid coverage whose zone, in that order, has a true error rate of at most alpha.
  f. K3's features are z-scored with the training split's mean and standard deviation before the MLP. The page fixes
     the network and the optimiser; it says nothing on feature scaling, and the empty-month count (0 to 12) would
     otherwise sit on another scale than the normalised bands. Nothing is fitted on the validation split.
  g. K4's NDVI controls for Nandi and AWF use exp21's formula (all twelve months, (B08 - B04) / max(B08 + B04, 1e-6)),
     so AWF's temporal control equals exp21's recorded one. The 3x3 control is the spatial standard deviation of each
     pixel's median NDVI over the months, in the 3x3 block around the label pixel. Mangrove's indices use the valid
     months only, as the page defines f over valid months.
  h. K5's threshold, B02 above 0.2 reflectance, is 2,000 in the harmonised L2A digital numbers the tar stores. The
     owner confirmed it with the other thresholds; the page's marker now says so.
  i. Arm A's gate is read as Nandi's: within 2.0 points of Ai2's 89.5% for AWF (exp21's replica was 1.4 points off),
     two-sided. The page now states it and leaves it for the owner to confirm before freezing. Agreement with exp21's
     recorded predictions is reported in the run's summary, not in the gate file, which holds counts and accuracy only.
  j. The gate counts attempts: at most three in all, the first and two retries. A third failed attempt closes the gate
     and nothing is graded. A passed gate is not rerun. Each attempt is appended to a ledger before the gate file is
     written. On the cluster the ledger lives outside the git checkout (E89_GATE_LEDGER), so the job's hard reset cannot
     roll the count back: a gate file behind its ledger is restored from it, and one that holds an attempt the ledger
     lacks is refused. The run checks the pass itself (require_gate): made on a frozen page, on the pinned checkpoint,
     and held by the ledger.
  k. --smoke runs the estimate study at the run's R = 2,000 draws but certifies on the first SMOKE_DRAWS of the random
     design's draws and bootstraps SMOKE_BOOT resamples, so it finishes in under a minute; the run uses R = 2,000
     and 2,000 resamples everywhere, as the page fixes.
  l. Imagery is read from the tar only. There is no fetch mode yet: if the inventory finds no Sentinel-2 layers in
     mangrove.tar, a fetch mode, and the gate's 1.0-point tolerance for refetched imagery, come before freezing. The
     tar's size makes that likely: 62,433,280 bytes is 121,940 tar blocks, and a 2x2 window with 12 item groups takes
     about 60 to 200 of them, so the tar can carry imagery for at most about 2,000 windows, not the page's expected
     12,500 validation windows. A validation window with no completed item group is dropped and counted (the gate
     records the count), and the gate refuses, recording no attempt, when no kept window has imagery.
  m. A window with fewer than 12 completed item groups is run with its own number of timesteps (batched by it), which
     is what rslearn's masked pooling over the missing timesteps reduces to; the inventory and the run count them.
  n. Nandi's polygon id is looked for under the option keys in NANDI_POLYGON_KEYS, a guess until its windows are seen,
     and its label source under the option key "source".

Safety, not interpretation. The checkpoint is checked against its pinned sha256, a local --ckpt included, and loaded
with weights_only=True; any other global it pickles becomes an inert stub class, and a global naming code execution is
refused. The scored checkpoint's sha256 is recorded in the replica's info and in every gate attempt.
predict_population is the one function that scores validation windows, and it refuses before the page is frozen;
compute_units and run_arm refuse without a verified gate pass.

Outputs (exp/out): exp89_inventory_<arm>.json (--inventory), exp89_s2_<arm>.json (--smoke-torch --real),
exp89_gate_<arm>.json (--gate) and its ledger exp89_gate_<arm>.ledger.jsonl (in E89_GATE_LEDGER when set, else beside
it), exp89_units_<arm>.npz and exp89_summary.json (the run); exp89_summary_smoke.json from --smoke. --out-dir moves
them. Downloads go to HF_HOME (scratch on the cluster) and are checked against the
pinned sha256 before use; the tar is extracted under --data.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import traceback
import warnings

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (os.path.join(ROOT, "scripts"), ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import assess, estimate as est, metrics, stats   # noqa: E402
import exp78_error_rate_estimation as e78                           # noqa: E402  (jsonable; numpy only)

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "finetuned_checkpoints.md")

# ----------------------------------------------------------------------------- the pinned upstream (HF API, 1 Oct 2026)
MANGROVE_MODEL = "allenai/OlmoEarth-v1-FT-Mangrove-Base"
MANGROVE_DATA = "allenai/olmoearth_projects_mangrove"
NANDI_MODEL = "allenai/OlmoEarth-v1-FT-Nandi-Base"
NANDI_DATA = "allenai/olmoearth_projects_nandi"
AWF_MODEL = "allenai/OlmoEarth-v1-FT-AWF-Base"
AWF_DATA = "allenai/olmoearth_projects_awf"
# olmoearth_projects (GitHub) at the commit whose olmoearth_run_data configs the replica below follows
CONFIG_REVISION = "589bce143f17fb3e522a8b09ba6dfa8d36aa36ae"

# ----------------------------------------------------------------------------- fixed by the page
SEED = 89
REVIEW_BUDGETS = (0.05, 0.10, 0.20)          # 5% and 10% graded, 20% reported
EST_BUDGETS = (300, 1000)
MIN_N_GRADED = 1500                          # below it no budget is graded
DRAWS = 2000
BOOT = 2000
SMOKE_DRAWS = 300
SMOKE_BOOT = 300
DELTA = 0.10
ERROR_FLOOR = 40
P2_AUROC = 0.65
P3_POINTS = 0.05
P4_COVER = 0.93
P4_WIDTH = 0.85
P5_VIOLATION = 0.12
P5_LEAD = 0.10
EPS = 1e-12
K5_B02 = 2000.0                              # 0.2 reflectance in harmonised L2A digital numbers
MAX_GATE_ATTEMPTS = 3
ARM_INDEX = {"mangrove": 0, "nandi": 1, "awf": 2}
DESIGN_INDEX = {"random": 0, "confidence": 1}
OLMO_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")
N_MONTHS = 12
# K3, the no-encoder classifier: fixed on the page, no tuning
K3_HIDDEN, K3_DROPOUT, K3_LR, K3_WD, K3_BATCH, K3_EPOCHS = 256, 0.1, 1e-3, 1e-4, 512, 30
K3_CAP_MANGROVE = 20000
S2_WINDOWS = 512
S2_MIN_ACCURACY = 0.95                       # Mangrove only; in-sample, a loading check

ARMS = {
    "mangrove": {
        "model": MANGROVE_MODEL, "model_revision": "57ee738b5b4474ad61391ed612e4fbc2a034353e",
        "model_file": "model.ckpt", "model_sha256": "ffb51fbb58cf96219df848303268445a710a804465c2e57a823e26ec40b3eb2b",
        "model_bytes": 1041362118,
        "data": MANGROVE_DATA, "data_revision": "3a878b8a9e54ef627939d48cab334f74e89fa634",
        "data_file": "mangrove.tar", "data_sha256": "ef9dba6a18ec03217e233abf9652423612d8f42999d69637920aa31d1e2d3594",
        "data_bytes": 62433280,
        "extra_files": {"annotation_features.geojson": "e87bc7919b3a1684aca1c1e3d5fd0b76f85c61c77f5cec1d610cce7ed72fa2a1"},
        "group": "sample_100K", "n_out": 4, "trained": (1, 2, 3), "untrained": (0,),
        "class_names": {1: "mangrove", 2: "water", 3: "other"},
        "label_layer": ("label_raster", "label"), "label_fill": 0, "unit": "block2", "patch": 2,
        "head_keys": ("model.decoders.mangrove_classification.0.output_layer.weight",
                      "model.decoders.mangrove_classification.0.output_layer.bias"),
        "head_kind": "linear",
        "ai2_accuracy": 0.976, "gate_tolerance": {"tar": 0.005, "fetched": 0.010},
        "graded": ("P1", "P2", "P3", "P4", "P5"), "p1_capture": 0.40, "alpha": 0.02, "alpha_reported": (0.01,),
        "p5_coverage": 0.50, "k3_cap": K3_CAP_MANGROVE, "clusters": "0.1 degree cell", "clusters_reported": "1 degree cell",
    },
    "nandi": {
        "model": NANDI_MODEL, "model_revision": None, "model_file": "model.ckpt", "model_sha256": None,
        "data": NANDI_DATA, "data_revision": None, "data_file": None, "data_sha256": None, "extra_files": {},
        "group": "spatial_split", "n_out": 11, "trained": tuple(range(10)), "untrained": (10,),
        "class_names": {0: "coffee", 1: "grassland", 2: "trees", 3: "maize", 4: "sugarcane", 5: "tea",
                        6: "vegetables", 7: "legumes", 8: "water", 9: "builtup"},
        "label_layer": ("label", "category"), "label_fill": 10, "unit": "pixel", "patch": 1, "crop": 16,
        "head_keys": ("model.decoders.segment.1.layer.weight", "model.decoders.segment.1.layer.bias"),
        "head_kind": "conv1x1",
        "ai2_accuracy": 0.873, "gate_tolerance": {"tar": 0.020, "fetched": 0.020},
        "graded": ("P1", "P2", "P3", "P4", "P5"), "p1_capture": 0.30, "alpha": 0.05, "alpha_reported": (0.10,),
        "p5_coverage": 0.30, "k3_cap": None, "clusters": "128-px split cell, merged by source polygon",
        "clusters_reported": "0.05 degree cell",
        "not_run_reason": "allenai/OlmoEarth-v1-FT-Nandi-Base and allenai/olmoearth_projects_nandi returned 401 on the "
                          "Hub on 1 October 2026; arm N runs only if Ai2 publishes both and their revisions are pinned",
    },
    "awf": {
        "model": AWF_MODEL, "model_revision": "a347b1546ab881c92aa125400ed5acc8126394ca", "model_file": "model.ckpt",
        "model_sha256": "1036b76bf7dfde819479dfa02507634f59a5637f14d4071bc3bf18f4d3d4bab2", "model_bytes": 1041443421,
        "data": AWF_DATA, "data_revision": "da8eb6aab65b518cb8223912ba9091cbaa5dad2f",
        "data_file": "dataset.tar", "data_sha256": "d0837f14235f9ef3207af6d84c428bb9dcb81bd18ee6a96d03ce4b5fff992090",
        "data_bytes": 1869219840, "extra_files": {},
        "group": "spatial_split", "n_out": 10, "trained": tuple(range(9)), "untrained": (9,),
        "class_names": None, "label_layer": ("label", "category"), "label_fill": 9, "unit": "pixel", "patch": 4,
        "crop": 16, "head_keys": ("model.decoders.segment.1.layer.weight", "model.decoders.segment.1.layer.bias"),
        "head_kind": "conv1x1_up4",
        "ai2_accuracy": 0.895, "gate_tolerance": {"tar": 0.020, "fetched": 0.020},
        "graded": ("P2", "P3"), "p1_capture": None, "alpha": 0.05, "alpha_reported": (0.10,), "p5_coverage": None,
        "k3_cap": None, "clusters": "exp21's annotation task", "clusters_reported": None,
    },
}
ENCODER_PREFIXES = ("model.encoder.0.model.", "model.encoder.0.model.encoder.", "model.encoder.0.")
# The encoder skeletons' config.json at the revisions exp/out/upstream_revisions.json records (the page names Base's)
ENCODER_REVISIONS = {"OLMOEARTH_V1_BASE": "4bd1392a4539404d2c74276c39f3cb4cfff466cc",
                     "OLMOEARTH_V1_NANO": "529248a4dc3c54014c56b7504641cec98de31d1c"}
# A Lightning checkpoint pickles globals besides tensors (rslearn and jsonargparse objects in hyper_parameters, numpy
# scalars in callback state). Each is mapped to an inert stub class under its own name: nothing is imported or run.
# A global that names code execution has no place in a checkpoint and is refused outright.
REFUSED_PREFIXES = ("os.", "posix.", "nt.", "subprocess.", "sys.", "builtins.", "shutil.", "socket.", "importlib.",
                    "runpy.", "pickle.", "marshal.", "code.", "ctypes.", "pty.", "multiprocessing.", "webbrowser.")
GATE_KEYS = ("arm", "prereg_status", "ai2_accuracy", "metric", "tolerance_points", "imagery_route", "attempts",
             "passed", "closed", "max_attempts")
ATTEMPT_KEYS = ("attempt", "utc", "commit", "dirty", "prereg_status", "checkpoint_sha256", "n_windows", "n_errors",
                "accuracy", "n_windows_dropped", "n_windows_no_imagery", "n_pixels", "n_pixel_errors", "accuracy_pixel",
                "gap_points", "pass")
GATE_LEDGER_ENV = "E89_GATE_LEDGER"          # the gate's ledger directory; exp/jobs/e89.sh puts it outside the checkout


class GateRefused(RuntimeError):
    """The gate, or a run that needs its pass, refuses: nothing is scored or graded."""


# ----------------------------------------------------------------------------- small helpers
def prereg_status(text=None):
    """'frozen', 'draft' or 'unknown', from the plan page's status line (the first line starting '**Status'). Copied
    from exp88: "frozen" must come first after the label, so "not frozen" never matches."""
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    for line in text.splitlines():
        if line.startswith("**Status"):
            low = line.lower()
            if re.match(r"\*\*status:?\*?\*?\s*frozen", low):
                return "frozen"
            if "draft" in low or "not frozen" in low:
                return "draft"
            return "unknown"
    return "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return commit, dirty


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(e78.jsonable(obj), f, indent=1)


def fail(record, part, ex):
    record.setdefault("failures", []).append({"part": part, "error": repr(ex), "traceback": traceback.format_exc()})
    print(f"{part} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)


def bkey(b):
    """A review budget as the summary's key: '0.05', '0.1', '0.2'."""
    return repr(float(b))


# ----------------------------------------------------------------------------- the channel and label rules per arm
def pad_center(a, size, fill=0):
    """rslearn's Pad(size, mode="center") on the last two axes: pad equally (the extra pixel after) when smaller,
    crop when larger, with the same floor rule rslearn uses for an odd difference."""
    a = np.asarray(a)
    out = a
    for axis in (-2, -1):
        extra = size - out.shape[axis]
        before = extra // 2
        after = extra - before
        if extra > 0:
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, after)
            out = np.pad(out, pad, constant_values=fill)
        elif extra < 0:
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(-before, out.shape[axis] + after)
            out = out[tuple(sl)]
    return out


def mangrove_unit_label(lab2):
    """(class or None, reason): a 2x2 label block is a unit when all four pixels hold the same class from 1 to 3."""
    v = np.asarray(lab2).ravel()
    if (v == 0).any():
        return None, "a label pixel is 0"
    if not (v == v[0]).all():
        return None, "label pixels disagree"
    if int(v[0]) not in ARMS["mangrove"]["trained"]:
        return None, "class outside 1 to 3"
    return int(v[0]), None


def single_pixel_label(lab, fill, trained):
    """(row, col, class) or (None, reason): Nandi and AWF keep a window with exactly one labelled pixel."""
    lab = np.asarray(lab)
    rows, cols = np.nonzero(lab != fill)
    if len(rows) != 1:
        return None, f"{len(rows)} labelled pixels"
    c = int(lab[rows[0], cols[0]])
    if c not in trained:
        return None, f"class {c} outside the trained classes"
    return (int(rows[0]), int(cols[0]), c), None


def crop_at(H, W, r, c, size, shift=0):
    """exp21's crop rule on an H x W window: the label pixel at (size // 2 - shift) when the crop fits, clamped at the
    edges. Returns (r0, c0, row in crop, col in crop)."""
    r0 = min(max(r - size // 2 + shift, 0), H - size)
    c0 = min(max(c - size // 2 + shift, 0), W - size)
    return r0, c0, r - r0, c - c0


def readings(logits, arm):
    """Per unit: the prediction (argmax over every output channel, as rslearn takes it), the graded confidence (top-1
    softmax probability over the trained channels, assess_prediction form "top1" at patch 1), the logit margin over
    the trained channels (form "margin"), the entropy over the trained channels, and two counts the page asks for."""
    A = ARMS[arm]
    L = np.asarray(logits, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] != A["n_out"]:
        raise ValueError(f"{arm}: logits are {L.shape}, expected (N, {A['n_out']})")
    trained = list(A["trained"])
    pred = L.argmax(1)
    scores = L[:, trained].T[:, None, :]                       # (C_trained, 1, N): one window per unit at patch 1
    top1 = assess.assess_prediction(scores, is_logit=True, patch=1, budgets=(0.05,), form="top1")
    mar = assess.assess_prediction(scores, is_logit=True, patch=1, budgets=(0.05,), form="margin")
    p1 = np.asarray(top1["arrays"]["confidence"], dtype=np.float64).ravel()
    margin = np.asarray(mar["arrays"]["confidence"], dtype=np.float64).ravel()
    z = L[:, trained] - L[:, trained].max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    entropy = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    top2 = np.argsort(-L, axis=1, kind="stable")[:, :2]
    return {"pred": pred, "p1": p1, "margin": margin, "entropy": entropy,
            "n_pred_untrained": int((~np.isin(pred, trained)).sum()),
            "n_untrained_in_top2": int(np.isin(top2, list(A["untrained"])).any(1).sum()),
            "n_p1_saturated": int((p1 >= 1.0).sum())}


# ----------------------------------------------------------------------------- controls from the twelve mosaics
def unit_indices(series):
    """series (N, 12 months, 12 bands) raw values in OlmoEarth band order -> NDVI and MNDWI per month, and the empty
    months (every band 0: no mosaic covered the unit, or the item group is absent)."""
    s = np.asarray(series, dtype=np.float64)
    b = {name: s[..., i] for i, name in enumerate(OLMO_BANDS)}
    empty = (s == 0).all(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
        mndwi = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"])
    return ndvi, mndwi, empty


def exp21_ndvi(series):
    """exp21's NDVI: every month, (B08 - B04) / max(B08 + B04, 1e-6)."""
    s = np.asarray(series, dtype=np.float64)
    red, nir = s[..., OLMO_BANDS.index("B04")], s[..., OLMO_BANDS.index("B08")]
    return (nir - red) / np.maximum(nir + red, 1e-6)


def index_controls(arm, series, ndvi3x3=None):
    """K4 and K5 per unit, each a suspicion (higher = more suspect). `series` is the unit's (N, 12, 12) raw series
    (Mangrove: the 2x2 block mean; Nandi and AWF: the label pixel); `ndvi3x3` is (N, 3, 3, 12) exp21-NDVI around the
    label pixel for Nandi and AWF."""
    ndvi, mndwi, empty = unit_indices(series)
    valid = ~empty
    out = {}
    if arm == "mangrove":
        n_valid = valid.sum(1)
        with np.errstate(invalid="ignore"):
            f = np.where(n_valid > 0, ((mndwi > 0) & valid).sum(1) / np.maximum(n_valid, 1), np.nan)
        med = np.array([np.median(m[v]) if v.any() else np.nan for m, v in zip(mndwi, valid)])
        std = np.array([np.std(n[v]) if v.any() else np.nan for n, v in zip(ndvi, valid)])
        out["k4_inundation_ambiguity"] = np.minimum(f, 1 - f)
        out["k4_mndwi_near_zero"] = -np.abs(med)
        out["k4_ndvi_month_std"] = std
    else:
        out["k4_ndvi_temporal_std"] = np.nanstd(exp21_ndvi(series), axis=1)
        if ndvi3x3 is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)              # NaN pixels outside a window edge
                med = np.nanmedian(np.asarray(ndvi3x3, dtype=np.float64), axis=-1)   # each pixel's median over months
                out["k4_ndvi_3x3_std"] = np.nanstd(med.reshape(med.shape[0], -1), axis=1)
    b02 = np.asarray(series, dtype=np.float64)[..., OLMO_BANDS.index("B02")]
    out["k5_cloud_months"] = (empty | (b02 > K5_B02)).sum(1).astype(np.float64)
    # a unit with no valid month has no index: it is put at the suspect end (it is suspect), not dropped
    for k, v in out.items():
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.nanmax(v[~bad]) + 1.0 if (~bad).any() else 0.0
    return out


def k3_features(series, normalize, arm):
    """K3's inputs per unit: per month the 12 normalised bands, NDVI (and MNDWI for Mangrove), each empty month filled
    with the unit's median of that feature over its non-empty months; then the count of empty months.
    `normalize` maps raw (N, 12, 12) to OlmoEarth's computed normalisation (std multiplier 2)."""
    s = np.asarray(series, dtype=np.float64)
    ndvi, mndwi, empty = unit_indices(s)
    parts = [np.asarray(normalize(s), dtype=np.float64), ndvi[..., None]]
    if arm == "mangrove":
        parts.append(mndwi[..., None])
    x = np.concatenate(parts, axis=-1)                         # (N, 12, F)
    x = np.where(empty[..., None], np.nan, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # a unit with every month empty
        med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True) if x.size else x
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isfinite(x), x, np.broadcast_to(med, x.shape))
    return np.concatenate([x.reshape(x.shape[0], -1), empty.sum(1, keepdims=True).astype(np.float64)], axis=1)


def fit_no_encoder_classifier(x_train, y_train, classes, x_eval, seed=SEED):
    """K3: an MLP in torch with the page's fixed settings (two hidden layers of 256, ReLU, dropout 0.1, Adam at 1e-3
    with weight decay 1e-4, batch 512, 30 epochs, seed 89, the last epoch). CPU, so the fit is deterministic. Returns
    the class probabilities on x_eval, (N_eval, len(classes)), columns in the order of `classes`, and the fit record."""
    import torch
    classes = list(classes)
    pos = {c: i for i, c in enumerate(classes)}
    y = np.array([pos[int(v)] for v in y_train], dtype=np.int64)
    xt = np.asarray(x_train, dtype=np.float64)
    mu, sd = xt.mean(0), xt.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    xt = ((xt - mu) / sd).astype(np.float32)
    xe = ((np.asarray(x_eval, dtype=np.float64) - mu) / sd).astype(np.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], K3_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(K3_DROPOUT),
                              torch.nn.Linear(K3_HIDDEN, K3_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(K3_DROPOUT),
                              torch.nn.Linear(K3_HIDDEN, len(classes)))
    opt = torch.optim.Adam(net.parameters(), lr=K3_LR, weight_decay=K3_WD)
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(xt), torch.from_numpy(y)
    losses = []
    net.train()
    for _ in range(K3_EPOCHS):
        order = torch.randperm(len(Y), generator=g)
        tot = 0.0
        for i in range(0, len(Y), K3_BATCH):
            idx = order[i:i + K3_BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        losses.append(tot / len(Y))
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(xe)), dim=1).double().numpy()
        train_acc = float((net(X).argmax(1) == Y).double().mean())
    return prob, {"n_train": int(len(Y)), "n_features": int(xt.shape[1]), "classes": classes,
                  "loss_last_epoch": losses[-1], "loss_first_epoch": losses[0], "train_accuracy": train_acc}


def k3_signals(prob, classes, pred):
    """K3a = 1 - K3's top-1 probability; K3b = 1 where K3's class differs from the fine-tuned model's prediction."""
    prob = np.asarray(prob, dtype=np.float64)
    k3_class = np.asarray(classes)[prob.argmax(1)]
    return {"k3a_no_encoder_uncertainty": 1.0 - prob.max(1),
            "k3b_no_encoder_disagreement": (k3_class != np.asarray(pred)).astype(np.float64)}, k3_class


def class_rarity(pred, train_labels, trained):
    """K2: minus the training frequency of the predicted class (rarer = more suspect); an untrained class has 0."""
    lab = np.asarray(train_labels)
    freq = {int(c): float((lab == c).mean()) if lab.size else 0.0 for c in trained}
    return -np.array([freq.get(int(p), 0.0) for p in pred], dtype=np.float64), freq


# ----------------------------------------------------------------------------- measures 1-4: accuracy and ranking
def signal_table(u, err):
    u = np.asarray(u, dtype=np.float64)
    e = np.asarray(err, dtype=np.float64)
    n, k = e.size, int(e.sum())
    theta = k / n
    aurc = metrics.aurc_expected(u, e)
    oracle = metrics.oracle_aurc(n, k)
    cap = metrics.capture_at_budget_expected(u, e, REVIEW_BUDGETS)
    return {"auroc": float(metrics.weighted_auroc(u, e, np.ones(n))), "aurc": aurc,
            "excess_aurc": float(metrics.excess_aurc(u, e)),
            "gap_closed": float((theta - aurc) / (theta - oracle)) if theta > oracle else float("nan"),
            "capture": {bkey(b): float(v) for b, v in cap.items()},
            "capture_random": {bkey(b): float(max(1, int(round(b * n))) / n) for b in REVIEW_BUDGETS},
            "ceiling": {bkey(b): metrics.attainable_ceiling(b, theta, n) if k else float("nan") for b in REVIEW_BUDGETS}}


def capture_bootstrap(score_a, score_b, errors, clusters, budget=0.10, n_boot=BOOT, seed=SEED):
    """Cluster bootstrap of capture(a) - capture(b) at a review budget, positive favours a. The resampling is
    stats.cluster_bootstrap_difference's, draw for draw: the same generator, the same calls, the same skipped
    resamples with no error."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(score_a, dtype=np.float64).ravel(), np.asarray(score_b, dtype=np.float64).ravel()
    err, cl = np.asarray(errors, dtype=np.float64).ravel(), np.asarray(clusters).ravel()
    ids = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        if err[sel].sum() == 0:
            continue
        ca = metrics.capture_at_budget_expected(a[sel], err[sel], (budget,))[budget]
        cb = metrics.capture_at_budget_expected(b[sel], err[sel], (budget,))[budget]
        diffs.append(ca - cb)
    d = np.asarray(diffs)
    if d.size == 0:
        return {"n_resamples": 0, "lower_one_sided_95": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "p_a_better": float("nan")}
    return {"n_resamples": int(d.size), "lower_one_sided_95": float(np.percentile(d, 5)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_a_better": float((d > 0).mean())}


def control_names(controls):
    return [k for k in controls if k.startswith(("k2_", "k3a_", "k3b_", "k4_", "k5_"))]


def ranking(err, conf_u, margin_u, entropy_u, controls, clusters, n_boot=BOOT, clusters_coarse=None):
    """Measures 2 and 3: every signal's AUROC, captures, ceilings, excess AURC and gap closed; the best informative
    control (lowest AURC of K2 to K5); confidence minus that control at 5% and 10% and in AURC, with cluster-bootstrap
    intervals (and the coarse clusters' beside them, reported)."""
    err = np.asarray(err, dtype=np.float64)
    sig = {"confidence": conf_u, "margin": margin_u, "entropy": entropy_u, **controls}
    table = {k: signal_table(v, err) for k, v in sig.items()}
    cands = control_names(controls)
    best = min(cands, key=lambda k: (table[k]["aurc"], k)) if cands else None
    out = {"signals": table, "candidates": cands, "best_control": best}
    if best is None or err.sum() == 0:
        return out
    conf, ctl = np.asarray(conf_u, dtype=np.float64), np.asarray(controls[best], dtype=np.float64)
    lead = {}
    for b in (0.05, 0.10):
        lead[bkey(b)] = {"difference": table["confidence"]["capture"][bkey(b)] - table[best]["capture"][bkey(b)],
                         "bootstrap": capture_bootstrap(conf, ctl, err, clusters, b, n_boot, SEED)}
    lo, hi, p = stats.cluster_bootstrap_difference(conf, ctl, err, clusters, n_boot=n_boot, seed=SEED)
    lead["aurc"] = {"difference": table["confidence"]["aurc"] - table[best]["aurc"],
                    "bootstrap": {"lo95": lo, "hi95": hi, "p_confidence_better": p}}
    out["confidence_minus_best_control"] = lead
    if clusters_coarse is not None:
        lo, hi, p = stats.cluster_bootstrap_difference(conf, ctl, err, clusters_coarse, n_boot=n_boot, seed=SEED)
        out["coarse_clusters_reported"] = {
            "capture_0.1": capture_bootstrap(conf, ctl, err, clusters_coarse, 0.10, n_boot, SEED),
            "aurc": {"lo95": lo, "hi95": hi, "p_confidence_better": p}, "n_clusters": int(np.unique(clusters_coarse).size)}
    return out


def accuracy_block(label, pred, err, p1, clusters, n_out, class_names, pixel=None):
    """Measures 1 and 4: accuracy, per-class recall, the confusion matrix, errors, clusters with an error, selective
    accuracy and ECE."""
    label, pred = np.asarray(label), np.asarray(pred)
    correct = 1.0 - np.asarray(err, dtype=np.float64)
    out = {"n_units": int(label.size), "n_errors": int(err.sum()), "accuracy_window": float(correct.mean()),
           "error_rate": float(np.mean(err)),
           "n_clusters": int(np.unique(clusters).size),
           "n_clusters_with_an_error": int(np.unique(np.asarray(clusters)[np.asarray(err) > 0]).size),
           "per_class_recall": {}, "confusion": {}}
    for c in np.unique(label):
        m = label == c
        name = (class_names or {}).get(int(c), str(int(c)))
        out["per_class_recall"][name] = {"class": int(c), "n": int(m.sum()), "recall": float((pred[m] == c).mean())}
        out["confusion"][str(int(c))] = np.bincount(pred[m], minlength=n_out).tolist()
    if pixel is not None:
        out["accuracy_pixel_micro"] = pixel
    out["selective_accuracy"] = {bkey(c): v for c, v in
                                 metrics.selective_accuracy(1.0 - np.asarray(p1), correct).items()}
    ece, rows = metrics.expected_calibration_error(np.asarray(p1), correct, bins=10)
    out["ece_10_bins"] = ece
    out["reliability"] = rows
    return out


# ----------------------------------------------------------------------------- measures 5-6: estimate and certify
def graded_budgets(N):
    """The page: a budget is graded only where B <= N/5, and none where N < 1,500."""
    return [B for B in EST_BUDGETS if N >= MIN_N_GRADED and B <= N / 5]


def draw_seeds(arm, design, B, draws):
    return np.random.SeedSequence([SEED, ARM_INDEX[arm], DESIGN_INDEX[design], int(B)]).generate_state(draws)


def estimate_study(err, margin, p1, arm, draws=DRAWS):
    """Both designs at every budget the population holds, through sample_for_estimation and estimate_error_rate
    unchanged. Returns (cells, the random design's draws {B: (R, B) indices}) for the certificate."""
    err = np.asarray(err, dtype=np.float64)
    N, K = err.size, int(err.sum())
    theta = K / N
    cells, random_draws = {}, {}
    for design in ("random", "confidence"):
        for B in EST_BUDGETS:
            if B > N:
                cells[f"{design}/{B}"] = {"run": False, "reason": f"the budget exceeds the {N} units"}
                continue
            seeds = draw_seeds(arm, design, B, draws)
            ests, lows, highs = np.empty(draws), np.empty(draws), np.empty(draws)
            idx_all = np.empty((draws, B), dtype=np.int32) if design == "random" else None
            for r, s in enumerate(seeds):
                smp = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=int(s))
                idx = np.asarray(smp["indices"], int)
                res = est.estimate_error_rate(smp, err[idx])
                ests[r], lows[r], highs[r] = res["estimate"], res["low"], res["high"]
                if idx_all is not None:
                    idx_all[r] = idx
            cover = (lows <= theta) & (theta <= highs)                 # as exp78 and exp88 grade it
            width = highs - lows
            cell = {"run": True, "design": design, "budget": B, "draws": draws, "theta": theta,
                    "coverage": float(cover.mean()), "median_width": float(np.median(width)),
                    "mean_width": float(width.mean()), "bias": float(ests.mean() - theta),
                    "rmse": float(np.sqrt(((ests - theta) ** 2).mean())), "mean_estimate": float(ests.mean())}
            if design == "random":
                cell["exact_coverage"] = float(est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval))
                random_draws[B] = idx_all
            cells[f"{design}/{B}"] = cell
    return cells, random_draws


def best_coverage(score, err, alpha):
    """c*(alpha): the largest grid coverage whose zone in this order (zone_order: descending score, ties by index)
    has a true error rate of at most alpha, from all labels."""
    order, _ = est.zone_order(score)
    err = np.asarray(err, dtype=np.float64)
    best = 0.0
    for c in est.ZONE_GRID:
        n = max(1, int(round(c * order.size)))
        if err[order[:n]].mean() <= alpha + EPS:
            best = c
    return best


def certify_study(err, orders, random_draws, alphas):
    """certify_zone on the random design's draws, per budget, alpha and order; the prefix rule for every order and
    the Bonferroni rule beside it for the confidence order."""
    err = np.asarray(err, dtype=np.float64)
    out = {}
    for B, idx_all in random_draws.items():
        for alpha in alphas:
            for oname, score in orders.items():
                for rule in (("prefix", "bonferroni") if oname == "confidence" else ("prefix",)):
                    cov, viol, nothing, refused, rates = [], 0, 0, 0, []
                    for idx in idx_all:
                        try:
                            res = est.certify_zone(score, idx, err[idx], alpha, delta=DELTA, rule=rule)
                        except ValueError as ex:
                            if "random sample" not in str(ex) and "enriched" not in str(ex):
                                raise
                            refused += 1
                            cov.append(0.0)
                            nothing += 1
                            continue
                        if res["coverage"] is None:
                            cov.append(0.0)
                            nothing += 1
                            continue
                        rate = float(err[res["zone_indices_in_order"]].mean())
                        rates.append(rate)
                        cov.append(float(res["coverage"]))
                        viol += rate > alpha + EPS
                    R = len(idx_all)
                    out[f"{oname}/{rule}/{B}/{alpha:g}"] = {
                        "order": oname, "rule": rule, "budget": int(B), "alpha": float(alpha), "draws": R,
                        "violation_rate": viol / R, "median_coverage": float(np.median(cov)),
                        "mean_coverage": float(np.mean(cov)), "share_certifying_nothing": nothing / R,
                        "refused_by_review_set_guard": refused,
                        "median_true_rate_of_certified_zones": float(np.median(rates)) if rates else float("nan"),
                        "best_coverage_possible": best_coverage(score, err, alpha),
                        "min_labels_to_certify": est.min_labels_to_certify(alpha, DELTA)}
    return out


# ----------------------------------------------------------------------------- the predictions
def _not_graded(reason, **kw):
    return {"holds": None, "graded": False, "reason": reason, **kw}


def grade_p1(rank, n_err, arm):
    A = ARMS[arm]
    cap = rank["signals"]["confidence"]["capture"][bkey(0.10)]
    if "P1" not in A["graded"]:
        return _not_graded("arm A is graded on P2 and P3 only", capture_10=cap)
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", capture_10=cap,
                           threshold=A["p1_capture"])
    return {"holds": bool(cap >= A["p1_capture"] - EPS), "graded": True, "capture_10": cap,
            "threshold": A["p1_capture"], "random": rank["signals"]["confidence"]["capture_random"][bkey(0.10)]}


def grade_p2(rank, n_err, arm):
    best = rank.get("best_control")
    au = rank["signals"][best]["auroc"] if best else float("nan")
    if "P2" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm", best_control=best, auroc=au)
    if best is None:
        return _not_graded("no control was computed")
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", best_control=best,
                           auroc=au, threshold=P2_AUROC)
    return {"holds": bool(au >= P2_AUROC - EPS), "graded": True, "best_control": best, "auroc": au,
            "threshold": P2_AUROC}


def grade_p3(rank, n_err, arm, p2):
    lead = rank.get("confidence_minus_best_control")
    if "P3" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm")
    if lead is None:
        return _not_graded("no control or no error")
    d = lead[bkey(0.10)]["difference"]
    lb = lead[bkey(0.10)]["bootstrap"]["lower_one_sided_95"]
    base = {"best_control": rank["best_control"], "capture_difference_10": d, "lower_one_sided_95": lb,
            "threshold_points": P3_POINTS}
    if n_err < ERROR_FLOOR:
        return _not_graded(f"{n_err} errors, below the floor of {ERROR_FLOOR}: reported only", **base)
    holds = bool(d >= P3_POINTS - EPS and np.isfinite(lb) and lb > 0)
    out = {"holds": holds, "graded": True, **base,
           "points_met": bool(d >= P3_POINTS - EPS), "bound_above_zero": bool(np.isfinite(lb) and lb > 0)}
    if p2.get("holds") is False:
        out["reading"] = "P2 failed: a comparison with a weak control, which shows little"
    return out


def grade_p4(cells, N, arm):
    if "P4" not in ARMS[arm]["graded"]:
        return _not_graded("not graded on this arm")
    gb = graded_budgets(N)
    if not gb:
        return _not_graded(f"no budget is graded: N = {N} (budgets are graded where B <= N/5 and N >= {MIN_N_GRADED})")
    cov = {B: cells[f"confidence/{B}"]["coverage"] for B in gb}
    ratio = cells["confidence/300"]["median_width"] / cells["random/300"]["median_width"]
    cov_ok = bool(all(v >= P4_COVER - EPS for v in cov.values()))
    width_ok = bool(ratio <= P4_WIDTH + EPS)
    return {"holds": cov_ok and width_ok, "graded": True, "graded_budgets": gb, "coverage": cov,
            "coverage_ok": cov_ok, "coverage_bar": P4_COVER, "width_ratio_300": ratio, "width_ok": width_ok,
            "width_bar": P4_WIDTH}


def grade_p5(cert, N, arm):
    A = ARMS[arm]
    if "P5" not in A["graded"]:
        return _not_graded("not graded on this arm")
    gb = graded_budgets(N)
    if not gb:
        return _not_graded(f"no budget is graded: N = {N}")
    B = 1000 if 1000 in gb else 300
    a = A["alpha"]
    c = cert[f"confidence/prefix/{B}/{a:g}"]
    k = cert[f"k3a/prefix/{B}/{a:g}"]
    lead = c["median_coverage"] - k["median_coverage"]
    valid = bool(c["violation_rate"] <= P5_VIOLATION + EPS)
    useful = bool(c["median_coverage"] >= A["p5_coverage"] - EPS)
    beats = bool(lead >= P5_LEAD - EPS)
    return {"holds": valid and useful and beats, "graded": True, "budget": B, "alpha": a,
            "violation_rate": c["violation_rate"], "valid": valid, "violation_bar": P5_VIOLATION,
            "median_coverage": c["median_coverage"], "useful": useful, "coverage_bar": A["p5_coverage"],
            "median_coverage_k3a": k["median_coverage"], "lead_over_k3a": lead, "beats_k3a": beats,
            "lead_bar": P5_LEAD}


def grade_units(units, arm, draws=DRAWS, n_boot=BOOT, certify_draws=None):
    """Everything after the logits: readings, accuracy, ranking, the estimate and certify studies, P1 to P5.
    `units`: label, logits, clusters, controls {name: suspicion}, k3_top_prob, optional clusters_coarse, stratum,
    pixel accuracy. numpy only."""
    A = ARMS[arm]
    label = np.asarray(units["label"])
    rd = readings(units["logits"], arm)
    err = (rd["pred"] != label).astype(np.float64)
    n_err = int(err.sum())
    clusters = np.asarray(units["clusters"])
    controls = {k: np.asarray(v, dtype=np.float64) for k, v in units["controls"].items()}
    res = {"arm": arm, "n_units": int(label.size), "n_errors": n_err, "draws": draws, "bootstrap_resamples": n_boot,
           "certify_draws": certify_draws or draws,
           "prediction_counts": {k: rd[k] for k in ("n_pred_untrained", "n_untrained_in_top2", "n_p1_saturated")}}
    res["accuracy"] = accuracy_block(label, rd["pred"], err, rd["p1"], clusters, A["n_out"], A["class_names"],
                                     units.get("pixel"))
    conf_u, margin_u = 1.0 - rd["p1"], -rd["margin"]
    if n_err == 0:
        res["ranking"] = {"note": "no error: nothing to rank"}
        rank = None
    else:
        rank = ranking(err, conf_u, margin_u, rd["entropy"], controls, clusters, n_boot,
                       units.get("clusters_coarse"))
        res["ranking"] = rank
    if units.get("stratum") is not None and rank is not None:
        res["by_stratum_reported"] = by_stratum(err, conf_u, controls.get(rank["best_control"]),
                                                np.asarray(units["stratum"]))
    N = int(label.size)
    cells, random_draws = estimate_study(err, rd["margin"], rd["p1"], arm, draws)
    res["estimate"] = {"graded_budgets": graded_budgets(N), "cells": cells}
    alphas = (A["alpha"],) + tuple(A["alpha_reported"])
    orders = {"confidence": rd["p1"], "k3a": 1.0 - controls["k3a_no_encoder_uncertainty"]}
    if certify_draws is not None:
        random_draws = {B: v[:certify_draws] for B, v in random_draws.items()}
    res["certify"] = certify_study(err, orders, random_draws, alphas)
    if rank is None:
        res["prereg"] = {p: _not_graded("no error in the population") for p in ("P1", "P2", "P3", "P4", "P5")}
    else:
        p2 = grade_p2(rank, n_err, arm)
        res["prereg"] = {"P1": grade_p1(rank, n_err, arm), "P2": p2, "P3": grade_p3(rank, n_err, arm, p2),
                         "P4": grade_p4(cells, N, arm), "P5": grade_p5(res["certify"], N, arm)}
    res["prereg"]["complete"] = all(res["prereg"][p]["holds"] is not None for p in A["graded"])
    res["prereg"]["graded_on_this_arm"] = list(A["graded"])
    return res


def by_stratum(err, conf_u, ctl_u, stratum):
    out = {}
    for s in np.unique(stratum):
        m = stratum == s
        e = err[m]
        row = {"n": int(m.sum()), "n_errors": int(e.sum()), "error_rate": float(e.mean())}
        if 0 < e.sum() < e.size:
            row["confidence_auroc"] = float(metrics.weighted_auroc(conf_u[m], e, np.ones(e.size)))
            row["confidence_capture_10"] = float(metrics.capture_at_budget_expected(conf_u[m], e, (0.10,))[0.10])
            if ctl_u is not None:
                row["best_control_auroc"] = float(metrics.weighted_auroc(ctl_u[m], e, np.ones(e.size)))
        out[str(s)] = row
    return out


# ----------------------------------------------------------------------------- S1: the numpy smoke
# what the synthetic cases must grade; tests/test_exp89.py reads these
SMOKE_EXPECTED = {
    "beats": {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True},
    "control_wins": {"P1": True, "P2": True, "P3": False, "P4": True, "P5": False},
    "weak": {"P1": False, "P2": False, "P3": False, "P4": False, "P5": False},
    "awf_beats": {"P1": None, "P2": True, "P3": True, "P4": None, "P5": None},
}
# (arm, N, error rate, confidence errors' suspicion band, K3 errors' band)
SMOKE_CASES = {"beats": ("mangrove", 6000, 0.024, (0.93, 1.0), (0.50, 1.0)),
               "control_wins": ("mangrove", 6000, 0.024, (0.82, 1.0), (0.97, 1.0)),
               "weak": ("mangrove", 6000, 0.024, (0.0, 1.0), (0.05, 1.0)),
               "awf_beats": ("awf", 400, 0.12, (0.90, 1.0), (0.50, 1.0))}


def logits_from_p1(p1, pred, n_out, trained, delta=1.0):
    """Logits whose argmax is `pred` and whose top-1 probability over the trained channels is `p1`: the predicted
    channel at 0, the next trained channel at -g, the rest of the trained channels at -(g + delta), the untrained
    channels far below. g solves p1 = 1 / (1 + e^-g (1 + (m - 2) e^-delta)) for m trained channels."""
    m = len(trained)
    rest = 1.0 + (m - 2) * np.exp(-delta)
    g = np.log(rest * p1 / (1.0 - p1))
    L = np.full((p1.size, n_out), -60.0)
    for i in range(p1.size):
        others = [c for c in trained if c != pred[i]]
        L[i, pred[i]] = 0.0
        L[i, others[0]] = -g[i]
        for c in others[1:]:
            L[i, c] = -(g[i] + delta)
    return L


def synthetic_units(case, seed=SEED):
    """Units with known answers. Correct units' suspicions are uniform on (0, 1); the errors' suspicions are uniform on
    a band near the suspect end, one band for the model's confidence and one for K3. The top-1 probability falls with
    the suspicion steeply (1 - p1 = 0.002 + 0.4 s^8), so the least confident quintile is the one the Neyman rule
    feeds. K2, K4 and K5 are weak by construction."""
    arm, N, theta, band_c, band_k = SMOKE_CASES[case]
    A = ARMS[arm]
    rng = np.random.default_rng([seed, list(SMOKE_CASES).index(case)])
    trained = list(A["trained"])
    K = int(round(theta * N))
    is_err = np.zeros(N, bool)
    is_err[rng.choice(N, K, replace=False)] = True
    prior = np.full(len(trained), 1.0 / len(trained))
    pred = np.asarray(trained)[rng.choice(len(trained), N, p=prior)]
    label = pred.copy()
    for i in np.flatnonzero(is_err):
        label[i] = rng.choice([c for c in trained if c != pred[i]])
    s = rng.random(N)
    s[is_err] = rng.uniform(*band_c, K)
    p1 = 1.0 - (0.002 + 0.4 * s ** 8)
    logits = logits_from_p1(p1, pred, A["n_out"], trained)
    t = rng.random(N)
    t[is_err] = rng.uniform(*band_k, K)
    k3_top = 0.995 - 0.6 * t ** 3
    controls = {"k2_class_rarity": class_rarity(pred, rng.choice(trained, 5000, p=prior), trained)[0],
                "k3a_no_encoder_uncertainty": 1.0 - k3_top,
                "k3b_no_encoder_disagreement": (t > 0.97).astype(np.float64)}
    if arm == "mangrove":
        for name in ("k4_inundation_ambiguity", "k4_mndwi_near_zero", "k4_ndvi_month_std"):
            controls[name] = rng.standard_normal(N) + 0.1 * is_err
    else:
        controls["k4_ndvi_temporal_std"] = rng.standard_normal(N) + 0.1 * is_err
        controls["k4_ndvi_3x3_std"] = rng.standard_normal(N) + 0.1 * is_err
    controls["k5_cloud_months"] = rng.poisson(1.0 + 0.2 * is_err).astype(np.float64)
    n_cl = 300 if arm == "mangrove" else 30
    clusters = rng.integers(0, n_cl, N)
    return {"label": label, "logits": logits, "clusters": clusters.astype(str),
            "clusters_coarse": (clusters // 10).astype(str), "controls": controls,
            "stratum": np.asarray([str(v) for v in label])}


def planted_checks():
    """Every grade at and beside its threshold on planted numbers, the budget rules and the error floor."""
    def rank_with(cap_conf, cap_ctl, auroc, lb):
        sig = {"confidence": {"capture": {bkey(0.10): cap_conf}, "capture_random": {bkey(0.10): 0.10}},
               "k3a_no_encoder_uncertainty": {"auroc": auroc, "capture": {bkey(0.10): cap_ctl}}}
        return {"signals": sig, "best_control": "k3a_no_encoder_uncertainty",
                "confidence_minus_best_control": {bkey(0.10): {"difference": cap_conf - cap_ctl,
                                                               "bootstrap": {"lower_one_sided_95": lb}}}}

    def cells_with(cov300, cov1000, w_conf, w_rand):
        return {"confidence/300": {"coverage": cov300, "median_width": w_conf},
                "confidence/1000": {"coverage": cov1000, "median_width": w_conf},
                "random/300": {"coverage": 0.95, "median_width": w_rand}}

    def cert_with(B, viol, cov, cov_k, a=0.02):
        return {f"confidence/prefix/{B}/{a:g}": {"violation_rate": viol, "median_coverage": cov},
                f"k3a/prefix/{B}/{a:g}": {"median_coverage": cov_k}}

    c = {}
    c["P1_at_threshold"] = grade_p1(rank_with(0.40, 0.1, 0.7, 0.1), 40, "mangrove")["holds"] is True
    c["P1_below"] = grade_p1(rank_with(0.3999, 0.1, 0.7, 0.1), 40, "mangrove")["holds"] is False
    c["P1_nandi_threshold"] = grade_p1(rank_with(0.30, 0.1, 0.7, 0.1), 40, "nandi")["holds"] is True
    c["P1_floor_reported_only"] = grade_p1(rank_with(0.9, 0.1, 0.7, 0.1), 39, "mangrove")["holds"] is None
    c["P1_not_graded_on_awf"] = grade_p1(rank_with(0.9, 0.1, 0.7, 0.1), 41, "awf")["holds"] is None
    c["P2_at_threshold"] = grade_p2(rank_with(0.5, 0.1, 0.65, 0.1), 40, "mangrove")["holds"] is True
    c["P2_below"] = grade_p2(rank_with(0.5, 0.1, 0.6499, 0.1), 40, "mangrove")["holds"] is False
    c["P2_graded_on_awf"] = grade_p2(rank_with(0.5, 0.1, 0.70, 0.1), 41, "awf")["holds"] is True
    p2 = {"holds": True}
    c["P3_at_threshold"] = grade_p3(rank_with(0.45, 0.40, 0.7, 0.001), 40, "mangrove", p2)["holds"] is True
    c["P3_points_short"] = grade_p3(rank_with(0.4499, 0.40, 0.7, 0.001), 40, "mangrove", p2)["holds"] is False
    c["P3_bound_at_zero"] = grade_p3(rank_with(0.60, 0.40, 0.7, 0.0), 40, "mangrove", p2)["holds"] is False
    c["P3_weak_control_reading"] = "weak control" in grade_p3(rank_with(0.6, 0.4, 0.6, 0.1), 40, "mangrove",
                                                              {"holds": False})["reading"]
    c["P4_at_thresholds"] = grade_p4(cells_with(0.93, 0.93, 0.85, 1.0), 5000, "mangrove")["holds"] is True
    c["P4_coverage_short"] = grade_p4(cells_with(0.9299, 0.95, 0.5, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_coverage_short_at_1000"] = grade_p4(cells_with(0.95, 0.9299, 0.5, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_1000_not_graded_below_5000"] = grade_p4(cells_with(0.95, 0.10, 0.5, 1.0), 4999, "mangrove")["holds"] is True
    c["P4_width_over"] = grade_p4(cells_with(0.95, 0.95, 0.8501, 1.0), 5000, "mangrove")["holds"] is False
    c["P4_none_below_1500"] = grade_p4(cells_with(0.95, 0.95, 0.5, 1.0), 1499, "mangrove")["holds"] is None
    c["P5_at_thresholds"] = grade_p5(cert_with(1000, 0.12, 0.50, 0.40), 5000, "mangrove")["holds"] is True
    c["P5_invalid"] = grade_p5(cert_with(1000, 0.1201, 0.9, 0.1), 5000, "mangrove")["holds"] is False
    c["P5_not_useful"] = grade_p5(cert_with(1000, 0.0, 0.4999, 0.1), 5000, "mangrove")["holds"] is False
    c["P5_lead_short"] = grade_p5(cert_with(1000, 0.0, 0.9, 0.8001), 5000, "mangrove")["holds"] is False
    c["P5_at_300_when_only_300_graded"] = grade_p5(cert_with(300, 0.0, 0.9, 0.1), 2000, "mangrove")["budget"] == 300
    c["P5_nandi_alpha_and_bar"] = grade_p5(cert_with(1000, 0.0, 0.30, 0.20, a=0.05), 5000, "nandi")["holds"] is True
    c["budgets"] = (graded_budgets(1499) == [] and graded_budgets(1500) == [300] and graded_budgets(4999) == [300]
                    and graded_budgets(5000) == [300, 1000])
    return c


def smoke(out_dir=None, draws=DRAWS, certify_draws=SMOKE_DRAWS, n_boot=SMOKE_BOOT):
    """S1: the synthetic cases through every measure and every grade, the planted grades, the freeze guard's reading."""
    out_dir = out_dir or OUT
    t0 = time.time()
    cases = {}
    for case in SMOKE_EXPECTED:
        arm = SMOKE_CASES[case][0]
        cases[case] = grade_units(synthetic_units(case), arm, draws=draws, n_boot=n_boot, certify_draws=certify_draws)
    out = {"experiment": "exp89 smoke S1: synthetic units with known answers", "smoke": True, "draws": draws,
           "certify_draws": certify_draws, "bootstrap_resamples": n_boot, "cases": cases, "planted": planted_checks(),
           "prereg_status_reading": {
               "frozen": prereg_status("**Status: frozen on 2 October 2026, before any run.**"),
               "frozen_naming_the_draft": prereg_status("**Status: frozen on 2 October 2026; the draft of 1 October "
                                                        "is in the history.**"),
               "draft": prereg_status("**Status: DRAFT, not frozen.** Written 1 October 2026.")}}
    assert out["prereg_status_reading"] == {"frozen": "frozen", "frozen_naming_the_draft": "frozen", "draft": "draft"}, \
        out["prereg_status_reading"]
    bad = [k for k, v in out["planted"].items() if v is not True]
    if bad:
        raise AssertionError(f"planted grades wrong: {bad}")
    out["seconds"] = time.time() - t0
    dump(out, os.path.join(out_dir, "exp89_summary_smoke.json"))
    for case, want in SMOKE_EXPECTED.items():
        g = cases[case]["prereg"]
        got = {p: g[p]["holds"] for p in want}
        print(f"smoke case {case}: " + ", ".join(f"{p} {v}" for p, v in got.items()) +
              f" | errors {cases[case]['n_errors']} of {cases[case]['n_units']}", flush=True)
        if got != want:
            raise AssertionError(f"smoke case {case}: graded {got}, the synthetic design implies {want}")
    print(f"smoke OK in {out['seconds']:.1f}s: every case grades as designed, planted grades hold at their thresholds, "
          "the freeze guard reads the status line", flush=True)
    return out


# ----------------------------------------------------------------------------- the rslearn dataset on disk
def find_windows_root(data_dir):
    """The `windows` directory of an extracted rslearn dataset (at most three levels down)."""
    for depth in range(4):
        pattern = os.path.join(data_dir, *(["*"] * depth), "windows")
        import glob
        hits = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no rslearn windows directory under {data_dir}")


def list_windows(windows_root, group):
    """[{dir, name, group, split, meta}] for every window of the group, sorted by name."""
    gdir = os.path.join(windows_root, group)
    out = []
    for name in sorted(os.listdir(gdir)):
        wdir = os.path.join(gdir, name)
        mpath = os.path.join(wdir, "metadata.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            meta = json.load(f)
        out.append({"dir": wdir, "name": name, "group": group,
                    "split": (meta.get("options") or {}).get("split"), "meta": meta})
    return out


def window_grid(meta):
    """(H, W, crs, transform) of a window from rslearn's metadata: bounds in projection pixels."""
    from rasterio.transform import Affine
    x0, y0, x1, y1 = meta["bounds"]
    pr = meta["projection"]
    xr, yr = float(pr["x_resolution"]), float(pr["y_resolution"])
    return int(y1 - y0), int(x1 - x0), pr["crs"], Affine(xr, 0.0, x0 * xr, 0.0, yr, y0 * yr)


def window_lonlat(meta):
    """Longitude and latitude of the window's centre."""
    import rasterio.warp
    H, W, crs, tr = window_grid(meta)
    x, y = tr * (W / 2.0, H / 2.0)
    lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def completed_groups(wdir, layer):
    """Sorted item-group indices of a layer whose directory holds rslearn's `completed` marker, as rslearn's
    load_all_item_groups discovers them: `layer` is group 0, `layer.k` group k."""
    ldir = os.path.join(wdir, "layers")
    if not os.path.isdir(ldir):
        return []
    out = []
    for d in os.listdir(ldir):
        if d == layer:
            k = 0
        elif d.startswith(layer + ".") and d[len(layer) + 1:].isdigit():
            k = int(d[len(layer) + 1:])
        else:
            continue
        if os.path.exists(os.path.join(ldir, d, "completed")):
            out.append(k)
    return sorted(out)


def read_band_sets(gdir, H, W, crs, transform):
    """{band: (H, W) float32} from every band-set directory of one item group, warped bilinearly onto the window's
    10 m grid where a band set is stored at another resolution (exp21's rule)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    bands = {}
    for bs in sorted(os.listdir(gdir)):
        path = os.path.join(gdir, bs, "geotiff.tif")
        if not os.path.exists(path):
            continue
        names = bs.split("_")
        with rasterio.open(path) as src:
            if (src.height, src.width) == (H, W):
                data = src.read()
            else:
                with WarpedVRT(src, crs=crs, transform=transform, width=W, height=H,
                               resampling=Resampling.bilinear) as vrt:
                    data = vrt.read()
        if data.shape[0] != len(names):
            raise RuntimeError(f"{path}: {data.shape[0]} bands for the band set {bs}")
        for i, n in enumerate(names):
            bands[n] = data[i].astype(np.float32)
    return bands


def read_stack(wdir, meta, layer="sentinel2"):
    """(H, W, 12, 12) raw stack in OlmoEarth band order and the number of completed item groups. Months beyond the
    completed groups stay 0 (absent), and are counted."""
    H, W, crs, tr = window_grid(meta)
    groups = completed_groups(wdir, layer)
    stack = np.zeros((H, W, N_MONTHS, len(OLMO_BANDS)), dtype=np.float32)
    for t, k in enumerate(groups[:N_MONTHS]):
        name = layer if k == 0 else f"{layer}.{k}"
        bands = read_band_sets(os.path.join(wdir, "layers", name), H, W, crs, tr)
        missing = [b for b in OLMO_BANDS if b not in bands]
        if missing:
            raise RuntimeError(f"{wdir} {name}: bands {missing} are missing")
        for j, b in enumerate(OLMO_BANDS):
            stack[:, :, t, j] = bands[b]
    return stack, len(groups)


def read_label(wdir, arm):
    import rasterio
    layer, band = ARMS[arm]["label_layer"]
    path = os.path.join(wdir, "layers", layer, band, "geotiff.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def unit_of(arm, wdir, meta, with_stack=True):
    """One window read by the arm's rules: its label unit (or the reason it is dropped), its model input and the
    series the controls read. Mangrove: rslearn's centre Pad to 2 on the image and the label. Nandi: the 16-px crop
    at shifts 0 to 3 (shift 0 is the graded input) and the 3x3 NDVI block around the label pixel."""
    A = ARMS[arm]
    lab = read_label(wdir, arm)
    rec = {"label": None, "drop": None}
    if lab is None:
        rec["drop"] = "no label raster"
        return rec
    if arm == "mangrove":
        lab2 = pad_center(lab, A["patch"], fill=0)
        rec["label"], rec["drop"] = mangrove_unit_label(lab2)
        rec["pixel_labels"] = lab2.ravel()
    else:
        got, why = single_pixel_label(lab, A["label_fill"], A["trained"])
        if got is None:
            rec["drop"] = why
        else:
            rec["rc"], rec["label"] = got[:2], got[2]
    if not with_stack:
        return rec
    stack, n_groups = read_stack(wdir, meta)
    rec["n_groups"] = n_groups
    if n_groups == 0:
        rec["drop"] = rec["drop"] or "no imagery"
        rec["no_imagery"] = True
        return rec
    if arm == "mangrove":
        s2 = pad_center(np.moveaxis(stack, (0, 1), (-2, -1)), A["patch"], fill=0)       # (T, B, 2, 2)
        rec["input"] = np.moveaxis(s2, (-2, -1), (0, 1))                                # (2, 2, T, B)
        rec["series"] = rec["input"].reshape(-1, N_MONTHS, len(OLMO_BANDS)).mean(0)     # the block mean
    elif rec["label"] is not None:
        H, W = stack.shape[:2]
        r, c = rec["rc"]
        rec["crops"], rec["locs"] = [], []
        for s in range(4):
            r0, c0, pr, pc = crop_at(H, W, r, c, A["crop"], s)
            rec["crops"].append(stack[r0:r0 + A["crop"], c0:c0 + A["crop"]])
            rec["locs"].append((pr, pc))
        rec["series"] = stack[r, c]
        rec["ndvi3x3"] = ndvi_block3(stack, r, c)
    return rec


def ndvi_block3(stack, r, c):
    """(3, 3, 12) exp21-NDVI around (r, c); pixels outside the window are NaN, so a block clipped at an edge is read
    over the pixels it has."""
    H, W = stack.shape[:2]
    out = np.full((3, 3, stack.shape[2]), np.nan)
    for i in range(3):
        for j in range(3):
            rr, cc = r - 1 + i, c - 1 + j
            if 0 <= rr < H and 0 <= cc < W:
                out[i, j] = exp21_ndvi(stack[rr, cc])
    return out


def cluster_ids(arm, rec_meta, name, lonlat):
    """The bootstrap's cluster and the reported coarser one."""
    if arm == "mangrove":
        lon, lat = lonlat
        return (f"{int(np.floor(lon / 0.1))}_{int(np.floor(lat / 0.1))}",
                f"{int(np.floor(lon))}_{int(np.floor(lat))}")
    if arm == "nandi":
        x0, y0 = rec_meta["bounds"][:2]
        cell = f"{rec_meta['projection']['crs']}_{int(np.floor(x0 / 128))}_{int(np.floor(y0 / 128))}"
        lon, lat = lonlat
        return cell, f"{int(np.floor(lon / 0.05))}_{int(np.floor(lat / 0.05))}"
    return name.split("_point_")[0], None


NANDI_POLYGON_KEYS = ("polygon_id", "source_polygon", "polygon", "source_id")


def merge_by_polygon(cells, options):
    """Nandi: cells that share a source polygon are merged (union-find), when the windows record a polygon id."""
    parent = {c: c for c in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for c, o in zip(cells, options):
        pid = next((o[k] for k in NANDI_POLYGON_KEYS if k in (o or {})), None)
        if pid is None:
            continue
        if pid in first:
            parent[find(c)] = find(first[pid])
        else:
            first[pid] = c
    return [find(c) for c in cells]


# ----------------------------------------------------------------------------- the Hub, the checkpoint
def hub_file(repo, filename, revision, repo_type, sha=None, local_dir=None):
    """A pinned file from the Hub (into HF_HOME), checked against its pinned sha256."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type=repo_type)
    got = sha256(path)
    if sha and got != sha:
        raise RuntimeError(f"{repo}@{revision[:8]}/{filename}: sha256 {got}, pinned {sha}")
    return path, got


def arm_readable(arm):
    """(readable, detail): both the model and the dataset answer the Hub API without a token (public, as the page's
    "if Ai2 publishes" requires), and both revisions are pinned here."""
    A = ARMS[arm]
    detail = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for kind, repo in (("model", A["model"]), ("dataset", A["data"])):
            try:
                info = api.model_info(repo, token=False) if kind == "model" else api.dataset_info(repo, token=False)
                detail[kind] = {"repo": repo, "readable": True, "sha": info.sha}
            except Exception as ex:                                  # 401, 404, offline
                detail[kind] = {"repo": repo, "readable": False, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    except ImportError as ex:
        return False, {"error": repr(ex)}
    pinned = bool(A["model_revision"] and A["data_revision"])
    detail["pinned"] = pinned
    ok = all(detail[k]["readable"] for k in ("model", "dataset")) and pinned
    return ok, detail


def nandi_status():
    ok, detail = arm_readable("nandi")
    return {"status": "readable and pinned" if ok else "not run", "reason": None if ok else ARMS["nandi"]["not_run_reason"],
            "checked_utc": utc_now(), "hub": detail}


def _stub_class(fullname):
    module, _, name = fullname.rpartition(".")

    def init(self, *args, **kwargs):
        self.__dict__["_stub_args"] = (args, kwargs)

    def setstate(self, state):
        self.__dict__["_stub_state"] = state
    return type(name, (), {"__module__": module, "__qualname__": name, "__init__": init, "__setstate__": setstate,
                           "__doc__": f"inert stand-in for {fullname}, never imported"})


def load_checkpoint(path):
    """(state_dict, how, top-level keys). torch.load with weights_only=True first. If the checkpoint pickles other
    globals (rslearn's, Lightning's, jsonargparse's), each is mapped to an inert stub class under its own name and
    the load is retried with weights_only=True, so nothing from those packages is imported or run; only tensors are
    kept. A global that names code execution (os, subprocess, builtins, ...) is refused."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        how = "weights_only"
    except Exception as first:                                     # an unsupported global
        unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        bad = [g for g in unsafe if g.startswith(REFUSED_PREFIXES)]
        if bad or not unsafe:
            raise RuntimeError(f"{path}: refused; it pickles {bad or 'no listed global'} ({first!r})") from None
        stubs = [(_stub_class(g), g) for g in unsafe]
        with torch.serialization.safe_globals(stubs):
            ck = torch.load(path, map_location="cpu", weights_only=True)
        how = f"weights_only with {len(stubs)} inert stub classes: {sorted(unsafe)}"
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if not isinstance(sd, dict):
        raise RuntimeError(f"{path}: no state_dict")
    return {k: v for k, v in sd.items() if hasattr(v, "shape")}, how, sorted(ck) if isinstance(ck, dict) else None


def checkpoint_inventory(sd, arm):
    A = ARMS[arm]
    prefixes = {p: sum(k.startswith(p) for k in sd) for p in ENCODER_PREFIXES}
    groups = {}
    for k in sd:
        g = ".".join(k.split(".")[:4])
        groups[g] = groups.get(g, 0) + 1
    return {"n_tensors": len(sd), "n_parameters": int(sum(int(np.prod(v.shape)) for v in sd.values())),
            "keys": {k: list(v.shape) for k, v in sd.items()}, "key_groups": groups,
            "encoder_prefix_counts": prefixes,
            "head_expected": {k: (list(sd[k].shape) if k in sd else None) for k in A["head_keys"]},
            "decoder_keys": {k: list(v.shape) for k, v in sd.items() if k.startswith("model.decoders.")}}


# ----------------------------------------------------------------------------- the replica (torch)
def encoder_skeleton(model_id):
    """OlmoEarth's model with random weights, built from its config.json at the pinned revision. The package's loader
    by model id reads the config at the Hub's current main; a changed config that keeps every parameter shape would
    still load strictly and change the model without a word."""
    from huggingface_hub import hf_hub_download
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_path
    cfg = hf_hub_download(ModelID[model_id].repo_id(), "config.json", revision=ENCODER_REVISIONS[model_id])
    return load_model_from_path(os.path.dirname(cfg), load_weights=False)


class Replica:
    """The fine-tuned model without rslearn: the encoder keys loaded strictly into olmoearth_pretrain's encoder
    (OlmoEarth v1-Base for the real checkpoints), tokens mean-pooled over timesteps and band sets, the arm's head.
    Mangrove: patch 2 on the 2x2 block, the pooling decoder's amax over a 1x1 map, Linear(D, 4). Nandi: patch 1 on
    the 16-px crop, a 1x1 conv to 11 channels read at the label pixel. AWF: exp21's logits_grid (patch 4, the 1x1
    conv on patch features, bilinear x4) at the label pixel, unchanged. fp32, TF32 off."""

    def __init__(self, arm, ckpt_path, model_id="OLMOEARTH_V1_BASE", device=None, checkpoint_sha256=None):
        import torch
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.arm, self.A = arm, ARMS[arm]
        if tuple(Modality.SENTINEL2_L2A.band_order) != OLMO_BANDS:
            raise RuntimeError(f"the encoder's band order {Modality.SENTINEL2_L2A.band_order} is not {OLMO_BANDS}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_sha256 = checkpoint_sha256 or sha256(ckpt_path)
        self.model_id, self.encoder_revision = model_id, ENCODER_REVISIONS[model_id]
        sd, self.load_how, self.top_level = load_checkpoint(ckpt_path)
        model = encoder_skeleton(model_id)
        self.encoder_prefix, errors = None, {}
        for pre in ENCODER_PREFIXES:
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if not sub:
                continue
            try:
                model.encoder.load_state_dict(sub, strict=True)
                self.encoder_prefix, self.n_encoder_keys = pre, len(sub)
                break
            except RuntimeError as ex:
                errors[pre] = str(ex)[:300]
        if self.encoder_prefix is None:
            raise RuntimeError(f"no prefix loads the encoder strictly: {errors}")
        wk, bk = self.A["head_keys"]
        if wk not in sd or bk not in sd:
            raise RuntimeError(f"head keys {wk}, {bk} not in the checkpoint; decoder keys: "
                               f"{[k for k in sd if k.startswith('model.decoders.')]}")
        w, b = sd[wk].float(), sd[bk].float()
        if w.ndim == 4:
            if w.shape[2:] != (1, 1):
                raise RuntimeError(f"{wk}: {tuple(w.shape)} is not a 1x1 convolution")
            w = w[:, :, 0, 0]
        D = int(getattr(model.encoder, "embedding_size", w.shape[1]))
        if tuple(w.shape) != (self.A["n_out"], D) or tuple(b.shape) != (self.A["n_out"],):
            raise RuntimeError(f"head {tuple(w.shape)} + {tuple(b.shape)}, expected ({self.A['n_out']}, {D})")
        self.head_shape = list(w.shape)
        self.n_checkpoint_keys = len(sd)
        self.model = model.to(self.device).eval()
        self.w, self.b = w.to(self.device), b.to(self.device)
        self.normalizer = Normalizer(Strategy.COMPUTED)                  # std multiplier 2, rslearn's default
        self.modality = Modality.SENTINEL2_L2A
        self.n_band_sets = len(Modality.SENTINEL2_L2A.band_sets)

    def normalize(self, raw):
        return self.normalizer.normalize(self.modality, np.asarray(raw, dtype=np.float64))

    def sample(self, stacks, timestamps="rslearn"):
        """(B, S, S, T, 12) raw -> MaskedOlmoEarthSample. rslearn's legacy timestamps are day 1, month index, 2024;
        exp21's were day 15, month index, 2023."""
        import torch
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        x = self.normalize(stacks)
        B, S, S2, T, _ = x.shape
        day, year = (1, 2024) if timestamps == "rslearn" else (15, 2023)
        ts = torch.tensor([[day, m, year] for m in range(T)], device=self.device)[None].repeat(B, 1, 1)
        return MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=self.device),
            sentinel2_l2a_mask=(torch.ones((B, S, S2, T, self.n_band_sets), device=self.device)
                                * MaskValue.ONLINE_ENCODER.value),
            timestamps=ts)

    def logits(self, stacks, locs=None, timestamps="rslearn"):
        """(B, n_out) logits at each unit."""
        import torch
        stacks = np.asarray(stacks, dtype=np.float32)
        with torch.no_grad():
            if self.arm == "awf":
                import exp21_finetuned_awf as exp21
                from oe_inferencex import awf
                awf.CROP = stacks.shape[1]                                     # the mask size stacks_to_sample uses
                _, pix = exp21.logits_grid(self.model, self.w, self.b, list(stacks))
                return np.stack([pix[i][r, c] for i, (r, c) in enumerate(locs)]).astype(np.float64)
            out = self.model.encoder(self.sample(stacks, timestamps), fast_pass=True, patch_size=self.A["patch"])
            feat = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])       # (B, H', W', D)
            if self.arm == "mangrove":
                v = torch.amax(feat, dim=(1, 2))                                 # the pooling decoder on a 1x1 map
            else:
                v = torch.stack([feat[i, r, c] for i, (r, c) in enumerate(locs)])
            return (v @ self.w.T + self.b).double().cpu().numpy()

    def info(self):
        return {"arm": self.arm, "device": str(self.device), "checkpoint_sha256": self.checkpoint_sha256,
                "encoder_config": {"model_id": self.model_id, "revision": self.encoder_revision},
                "load": self.load_how, "top_level": self.top_level,
                "encoder_prefix": self.encoder_prefix, "n_encoder_keys": self.n_encoder_keys,
                "n_checkpoint_keys": self.n_checkpoint_keys, "head_shape": self.head_shape, "dtype": "float32",
                "tf32": False, "patch": self.A["patch"]}


def batched_logits(rep, inputs, locs, T_groups, batch, timestamps="rslearn"):
    """Logits for every unit, batched by the number of item groups so each batch has one T and runs with
    fast_pass (a window with fewer groups is read with its own T, as rslearn's masked pooling reads it)."""
    n_out = rep.A["n_out"]
    out = np.zeros((len(inputs), n_out))
    T_groups = np.asarray(T_groups)
    for T in np.unique(T_groups):
        idx = np.flatnonzero(T_groups == T)
        for i in range(0, idx.size, batch):
            j = idx[i:i + batch]
            st = np.stack([inputs[k][..., :T, :] for k in j])
            out[j] = rep.logits(st, [locs[k] for k in j] if locs is not None else None, timestamps)
    return out


BATCH = {"mangrove": 1024, "nandi": 2, "awf": 32}


# ----------------------------------------------------------------------------- reading an arm's windows
def resolve_data(arm, data_dir, download=True):
    """The arm's extracted dataset: the pinned tar is fetched (when `download`), checked and extracted under
    data_dir once. Arm A uses an existing data/awf when present."""
    A = ARMS[arm]
    if arm == "awf":
        for cand in (os.path.join(data_dir, "dataset", "windows", "spatial_split"),
                     os.path.join(data_dir, "windows", "spatial_split")):
            if os.path.isdir(cand):
                return os.path.dirname(cand), None
    marker = os.path.join(data_dir, ".exp89_extracted")
    if os.path.exists(marker):
        return find_windows_root(data_dir), json.load(open(marker))
    if not download:
        raise FileNotFoundError(f"{data_dir} holds no extracted dataset and downloads are off")
    path, got = hub_file(A["data"], A["data_file"], A["data_revision"], "dataset", A["data_sha256"])
    info = extract_tar(path, data_dir, got)
    return find_windows_root(data_dir), info


def extract_tar(path, data_dir, sha=None):
    """Extract a dataset tar under data_dir once, refusing absolute or parent paths, and leave a marker."""
    os.makedirs(data_dir, exist_ok=True)
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        for m in members:
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise RuntimeError(f"unsafe member in the tar: {m.name}")
        # the data filter refuses links out of the directory and special files where Python has it (3.12, 3.11.4)
        tf.extractall(data_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    info = {"tar": path, "sha256": sha or sha256(path), "n_members": len(members), "extracted_utc": utc_now()}
    with open(os.path.join(data_dir, ".exp89_extracted"), "w") as f:
        json.dump(info, f)
    return info


def arm_windows(arm, windows_root):
    """Every window of the arm's group with its split, sorted by name."""
    if arm == "awf":
        from oe_inferencex import awf
        awf.ROOT = os.path.join(windows_root, ARMS["awf"]["group"])
        out = []
        for wdir, split, r, c, cat in awf.list_windows():
            out.append({"dir": wdir, "name": os.path.basename(wdir), "split": split, "rc": (r, c), "label": cat,
                        "meta": json.load(open(os.path.join(wdir, "metadata.json")))})
        return out
    return list_windows(windows_root, ARMS[arm]["group"])


def read_units(arm, windows, need="model", log=print):
    """Read windows into unit records. `need`: "label" (labels only) or "model" (labels, input, series)."""
    out = []
    t0 = time.time()
    for i, w in enumerate(windows):
        if arm == "awf":
            from oe_inferencex import awf
            rec = {"label": w["label"], "drop": None, "rc": w["rc"]}
            if need == "model":
                stack = awf.load_window_full(w["dir"])
                r, c = w["rc"]
                rec["n_groups"] = N_MONTHS
                r0, c0, pr, pc = crop_at(63, 63, r, c, ARMS["awf"]["crop"], 0)
                rec["crops"] = [stack[r0:r0 + 16, c0:c0 + 16]]
                rec["locs"] = [(pr, pc)]
                rec["series"] = stack[r, c]
                rec["ndvi3x3"] = ndvi_block3(stack, r, c)
        else:
            rec = unit_of(arm, w["dir"], w["meta"], with_stack=(need == "model"))
        rec["name"], rec["split"], rec["meta"] = w["name"], w["split"], w["meta"]
        out.append(rec)
        if (i + 1) % 2000 == 0:
            log(f"  read {i + 1}/{len(windows)} windows, {time.time() - t0:.0f}s")
    return out


def model_inputs(arm, recs, shift=0):
    if arm == "mangrove":
        return [r["input"] for r in recs], None
    return [r["crops"][shift] for r in recs], [r["locs"][shift] for r in recs]


def pixel_accuracy(recs, pred):
    """Mangrove: Ai2's pixel micro accuracy over every valid label pixel of every window, kept or not."""
    n = k = 0
    for r, p in zip(recs, pred):
        v = r["pixel_labels"][r["pixel_labels"] != 0]
        n += v.size
        k += int((v != p).sum())
    return {"n_pixels": int(n), "n_pixel_errors": int(k), "accuracy_pixel": 1.0 - k / n if n else float("nan")}


# ----------------------------------------------------------------------------- inventory (no model on any window)
def cmd_inventory(args):
    arm = args.arm
    out_dir = args.out_dir or OUT
    inv = {"experiment": "exp89 inventory: files, checkpoint keys, windows and labels per split; no model is run",
           "arm": arm, "utc": utc_now(), "prereg_status": prereg_status(), "commit": git_state()[0],
           "config_revision": CONFIG_REVISION,
           "nandi": nandi_status() if not args.offline else {"status": "not checked (offline)"}}
    if arm == "nandi" and inv["nandi"]["status"] != "readable and pinned" and not getattr(args, "synthetic", False):
        inv["status"] = "not run"
        dump(inv, os.path.join(out_dir, f"exp89_inventory_{arm}.json"))
        print(f"arm N not run: {ARMS['nandi']['not_run_reason']}", flush=True)
        return 0
    A = ARMS[arm]
    data_dir = args.data or default_data_dir(arm)
    files = {}
    if not args.offline:
        for fname, sha in [(A["data_file"], A["data_sha256"])] + list(A["extra_files"].items()):
            if arm == "awf" and os.path.isdir(os.path.join(data_dir, "dataset", "windows")):
                break
            p, got = hub_file(A["data"], fname, A["data_revision"], "dataset", sha)
            files[fname] = {"path": p, "bytes": os.path.getsize(p), "sha256": got, "sha256_pinned": sha,
                            "matches": got == sha}
    windows_root, extracted = resolve_data(arm, data_dir, download=not args.offline)
    inv["files"], inv["extracted"], inv["windows_root"] = files, extracted, windows_root
    tar_path = (extracted or {}).get("tar")
    if tar_path and os.path.exists(tar_path):
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
        top = {}
        for n in names:
            top[n.split("/")[0]] = top.get(n.split("/")[0], 0) + 1
        inv["tar"] = {"n_members": len(names), "top_level": top, "first_members": names[:20]}
    ckpt = args.ckpt
    if ckpt is None and not args.offline:
        ckpt, got = hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
        files[A["model_file"]] = {"path": ckpt, "bytes": os.path.getsize(ckpt), "sha256": got,
                                  "sha256_pinned": A["model_sha256"],
                                  "matches": (got == A["model_sha256"]) if A["model_sha256"] else None}
    if ckpt:
        if args.ckpt:                                   # a local file: listed, never scored, and its hash recorded
            got = sha256(ckpt)
            files["--ckpt"] = {"path": ckpt, "bytes": os.path.getsize(ckpt), "sha256": got,
                               "sha256_pinned": A["model_sha256"],
                               "matches": (got == A["model_sha256"]) if A["model_sha256"] else None}
        sd, how, top = load_checkpoint(ckpt)
        inv["checkpoint"] = {"load": how, "top_level": top, **checkpoint_inventory(sd, arm)}
    if "annotation_features.geojson" in files:
        inv["geojson"] = geojson_summary(files["annotation_features.geojson"]["path"])
    inv["dataset"] = dataset_summary(arm, windows_root)
    dump(inv, os.path.join(out_dir, f"exp89_inventory_{arm}.json"))
    d = inv["dataset"]
    print(json.dumps(e78.jsonable({"windows_per_split": d["windows_per_split"], "units_per_split": d["units_per_split"],
                                   "item_groups": d["item_groups_per_window"], "window_px": d["window_px"]}), indent=1))
    return 0


def geojson_summary(path):
    with open(path) as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    keys, geoms = {}, {}
    for ft in feats:
        for k in (ft.get("properties") or {}):
            keys[k] = keys.get(k, 0) + 1
        t = (ft.get("geometry") or {}).get("type")
        geoms[t] = geoms.get(t, 0) + 1
    return {"n_features": len(feats), "property_keys": keys, "geometry_types": geoms}


def _count(d, k, n=1):
    d[k] = d.get(k, 0) + n


def dataset_summary(arm, windows_root):
    """Windows per group and split, window sizes, CRSs, time ranges, layers, item groups, labels per split. Reads
    metadata and label rasters only, never imagery into a model."""
    A = ARMS[arm]
    groups = sorted(d for d in os.listdir(windows_root) if os.path.isdir(os.path.join(windows_root, d)))
    per_group = {}
    for g in groups:
        ws = list_windows(windows_root, g)
        sp = {}
        for w in ws:
            _count(sp, str(w["split"]))
        per_group[g] = {"n_windows": len(ws), "splits": sp}
    windows = arm_windows(arm, windows_root) if arm == "awf" else list_windows(windows_root, A["group"])
    out = {"groups": per_group, "group": A["group"], "windows_per_split": {}, "units_per_split": {},
           "dropped_per_split": {}, "unit_classes_per_split": {}, "pixel_classes_per_split": {}, "window_px": {},
           "crs": {}, "time_range": {"min_start": None, "max_end": None, "durations_days": {}}, "layers": {},
           "item_groups_per_window": {}, "items_json": 0, "band_sets": {}, "option_keys": {}}
    import datetime as dt
    for w in windows:
        sp = str(w["split"])
        _count(out["windows_per_split"], sp)
        meta = w["meta"]
        H, W = meta["bounds"][3] - meta["bounds"][1], meta["bounds"][2] - meta["bounds"][0]
        _count(out["window_px"], f"{H}x{W}")
        _count(out["crs"], meta["projection"]["crs"])
        for k in (meta.get("options") or {}):
            _count(out["option_keys"], k)
        tr = meta.get("time_range")
        if tr:
            a, b = tr
            out["time_range"]["min_start"] = min(filter(None, [out["time_range"]["min_start"], a]))
            out["time_range"]["max_end"] = max(filter(None, [out["time_range"]["max_end"], b]))
            try:
                days = (dt.datetime.fromisoformat(b) - dt.datetime.fromisoformat(a)).days
                _count(out["time_range"]["durations_days"], str(days))
            except ValueError:
                pass
        ldir = os.path.join(w["dir"], "layers")
        if os.path.isdir(ldir):
            for d in os.listdir(ldir):
                base = d.split(".")[0]
                e = out["layers"].setdefault(base, {"dirs": 0, "completed": 0})
                e["dirs"] += 1
                e["completed"] += os.path.exists(os.path.join(ldir, d, "completed"))
                if base == "sentinel2" and "." not in d:
                    for bs in os.listdir(os.path.join(ldir, d)):
                        if os.path.isdir(os.path.join(ldir, d, bs)):
                            _count(out["band_sets"], bs)
        _count(out["item_groups_per_window"], str(len(completed_groups(w["dir"], "sentinel2"))))
        out["items_json"] += os.path.exists(os.path.join(w["dir"], "items.json"))
        if arm == "awf":
            rec = {"label": w["label"], "drop": None}
        else:
            rec = unit_of(arm, w["dir"], meta, with_stack=False)
        if rec["label"] is None:
            _count(out["dropped_per_split"].setdefault(sp, {}), rec["drop"])
        else:
            _count(out["units_per_split"], sp)
            _count(out["unit_classes_per_split"].setdefault(sp, {}), str(rec["label"]))
        if "pixel_labels" in rec:
            for v, n in zip(*np.unique(rec["pixel_labels"], return_counts=True)):
                _count(out["pixel_classes_per_split"].setdefault(sp, {}), str(int(v)), int(n))
    return out


def default_data_dir(arm):
    base = os.environ.get("E89_DATA")
    if base:
        return os.path.join(base, arm)
    if arm == "awf":
        return os.path.join(ROOT, "data", "awf")
    return os.path.join(ROOT, "data", f"exp89_{arm}")


# ----------------------------------------------------------------------------- S2 on real training windows
def model_smoke(arm, rep, windows, out_path, n=S2_WINDOWS, log=print):
    """S2 on TRAINING windows only: strict load (done by Replica), one batch's shape, in-sample accuracy on n windows
    drawn with seed 89, the untrained channel never the argmax, the largest logit difference between the two
    timestamp conventions."""
    train = [w for w in windows if w["split"] == "train"]
    recs = read_units(arm, train, need="label", log=log)
    usable = [w for w, r in zip(train, recs) if r["label"] is not None]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(usable), min(n, len(usable)), replace=False).tolist())
    chosen = [usable[i] for i in pick]
    assert all(w["split"] == "train" for w in chosen), "S2 reads training windows only"
    recs = [r for r in read_units(arm, chosen, need="model", log=log) if "series" in r and r["label"] is not None]
    inputs, locs = model_inputs(arm, recs)
    T = [r["n_groups"] for r in recs]
    lg = batched_logits(rep, inputs, locs, T, BATCH[arm])
    k = 2 if len(T) > 1 and T[0] == T[1] else 1                         # one batch of one T
    first = rep.logits(np.stack(inputs[:k])[..., :T[0], :], locs[:k] if locs else None)
    label = np.array([r["label"] for r in recs])
    pred = lg.argmax(1)
    acc = float((pred == label).mean()) if label.size else float("nan")
    res = {"experiment": "exp89 S2: the real checkpoint on training windows (in-sample, never graded)", "arm": arm,
           "utc": utc_now(), "commit": git_state()[0], "replica": rep.info(), "n_windows": len(recs),
           "one_batch_output_shape": list(first.shape), "expected_columns": ARMS[arm]["n_out"],
           "accuracy_in_sample": acc, "accuracy_bar": S2_MIN_ACCURACY if arm == "mangrove" else None,
           "accuracy_ok": bool(acc >= S2_MIN_ACCURACY) if arm == "mangrove" else None,
           "n_pred_untrained": int((~np.isin(pred, ARMS[arm]["trained"])).sum()),
           "n_groups": {str(t): int(c) for t, c in zip(*np.unique(T, return_counts=True))} if T else {}}
    if arm != "awf":
        lg15 = batched_logits(rep, inputs, locs, T, BATCH[arm], timestamps="exp21")
        res["timestamp_conventions"] = {"rslearn": "day 1, month index, 2024", "exp21": "day 15, month index, 2023",
                                        "max_abs_logit_difference": float(np.abs(lg - lg15).max()) if lg.size else None,
                                        "n_predictions_changed": int((lg15.argmax(1) != pred).sum())}
    res["ok"] = bool(res["one_batch_output_shape"][-1] == ARMS[arm]["n_out"] and res["n_pred_untrained"] == 0
                     and (res["accuracy_ok"] is not False))
    dump(res, out_path)
    log(f"S2 {arm}: {len(recs)} training windows, in-sample accuracy {acc:.4f}, untrained argmax "
        f"{res['n_pred_untrained']}, ok {res['ok']}")
    return res


# ----------------------------------------------------------------------------- the gate (accuracy only)
def gate_record(arm, out_dir):
    path = os.path.join(out_dir, f"exp89_gate_{arm}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def gate_passed(arm, out_dir=None):
    rec, _ = gate_record(arm, out_dir or OUT)
    return bool(rec and rec.get("passed") is True), rec


def ledger_path(arm, out_dir):
    """The gate's append-only ledger, one JSON line per attempt. In E89_GATE_LEDGER when set (the cluster job sets it
    outside the git checkout, so a hard reset cannot touch it), else beside the gate file."""
    return os.path.join(os.environ.get(GATE_LEDGER_ENV) or out_dir, f"exp89_gate_{arm}.ledger.jsonl")


def read_ledger(arm, out_dir):
    path = ledger_path(arm, out_dir)
    if not os.path.exists(path):
        return [], path
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()], path


def append_ledger(arm, out_dir, attempt):
    path = ledger_path(arm, out_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(e78.jsonable(attempt), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def _canon(attempts):
    return [json.dumps(e78.jsonable(a), sort_keys=True) for a in attempts]


def reconcile_gate(arm, out_dir, rec):
    """The gate file's attempts against the ledger's. The ledger is the count: a file behind it (a reset restored an
    older committed file) is restored from it; a file holding an attempt the ledger does not hold, in order, was
    edited or written elsewhere, and is refused. Returns (the ledger's attempts, whether the file was behind)."""
    ledger, lpath = read_ledger(arm, out_dir)
    held = (rec or {}).get("attempts") or []
    if _canon(held) != _canon(ledger)[:len(held)]:
        raise GateRefused(f"exp89_gate_{arm}.json holds {len(held)} attempts that the gate's ledger ({lpath}, "
                          f"{len(ledger)} attempts) does not hold: the file was edited or written elsewhere. Nothing "
                          "is scored or graded until the two agree.")
    return ledger, len(ledger) > len(held)


def require_gate(arm, out_dir, ledger_required=True):
    """The run's own check of the gate, beside cmd_run's: a pass on the last attempt, within the attempt limit, every
    attempt made on a frozen page, the passing attempt on the pinned checkpoint, and (when the run scores windows) the
    same attempts in the ledger. Returns the gate record or raises GateRefused."""
    rec, _ = gate_record(arm, out_dir)
    if rec is None:
        raise GateRefused("no gate record")
    if rec.get("closed"):
        raise GateRefused("the gate is closed")
    att = rec.get("attempts") or []
    if rec.get("passed") is not True or not att or att[-1].get("pass") is not True:
        raise GateRefused("the gate has not passed")
    if len(att) > MAX_GATE_ATTEMPTS:
        raise GateRefused(f"the gate record holds {len(att)} attempts, more than the {MAX_GATE_ATTEMPTS} allowed")
    if any(a.get("prereg_status") != "frozen" for a in att):
        raise GateRefused("an attempt was made while the page was not frozen")
    pinned = ARMS[arm]["model_sha256"]
    if pinned and att[-1].get("checkpoint_sha256") != pinned:
        raise GateRefused(f"the passing attempt scored {att[-1].get('checkpoint_sha256')}, not the pinned checkpoint "
                          f"{pinned}")
    if ledger_required:
        ledger, lpath = read_ledger(arm, out_dir)
        if _canon(ledger) != _canon(att):
            raise GateRefused(f"the gate's ledger ({lpath}) does not hold the attempts of exp89_gate_{arm}.json")
    return rec


def predict_population(arm, rep, windows, log=print, synthetic=False):
    """Every validation window with imagery: (records, logits). Used by the gate and the run, after freezing. The only
    function that scores a validation window; it refuses before the page is frozen, except on the synthetic smoke's
    windows."""
    if not synthetic and prereg_status() != "frozen":
        raise RuntimeError(f"{os.path.relpath(PLAN, ROOT)} is not frozen: no validation window is scored")
    val = [w for w in windows if w["split"] == "val"]
    recs = read_units(arm, val, need="model", log=log)
    have = [r for r in recs if "series" in r or "input" in r]
    inputs, locs = model_inputs(arm, have)
    lg = batched_logits(rep, inputs, locs, [r["n_groups"] for r in have], BATCH[arm])
    return recs, have, lg


def run_gate(arm, rep, windows, out_dir, imagery_route="tar", log=print, synthetic=False):
    """G: predictions on the validation windows; only the window counts, the error counts and the accuracy are kept.
    Each attempt goes to the ledger first, then to the gate file (see reconcile_gate)."""
    A = ARMS[arm]
    rec, path = gate_record(arm, out_dir)
    attempts, behind = reconcile_gate(arm, out_dir, rec)
    rec = rec or {"arm": arm, "ai2_accuracy": A["ai2_accuracy"],
                  "metric": ("pixel micro accuracy over valid label pixels" if arm == "mangrove"
                             else "accuracy per window (one label pixel)"),
                  "tolerance_points": A["gate_tolerance"][imagery_route] * 100, "imagery_route": imagery_route,
                  "attempts": [], "passed": False, "closed": False, "max_attempts": MAX_GATE_ATTEMPTS}
    rec["attempts"] = attempts
    rec["passed"] = bool(attempts and attempts[-1]["pass"] is True)
    rec["closed"] = (not rec["passed"]) and len(attempts) >= MAX_GATE_ATTEMPTS
    rec["prereg_status"] = prereg_status()
    if behind:
        dump(rec, path)
        log(f"gate {arm}: the gate file was behind its ledger; restored to the ledger's {len(attempts)} attempts")
    if rec["passed"]:
        log(f"gate {arm}: already passed on attempt {rec['attempts'][-1]['attempt']}; not rerun")
        return rec
    if rec["closed"]:
        dump(rec, path)
        log(f"gate {arm}: closed after {len(rec['attempts'])} failed attempts; nothing is graded")
        return rec
    recs, have, lg = predict_population(arm, rep, windows, log, synthetic=synthetic)
    pred = lg.argmax(1)
    kept = [i for i, r in enumerate(have) if r["label"] is not None]
    if not kept:
        raise GateRefused(f"gate {arm}: none of the {len(recs)} validation windows has both imagery and a kept label "
                          f"({len(recs) - len(have)} have no imagery). Nothing to gate on; no attempt is recorded.")
    label = np.array([have[i]["label"] for i in kept])
    n_err = int((pred[kept] != label).sum())
    acc_w = 1.0 - n_err / len(kept)
    commit, dirty = git_state()
    att = {"attempt": len(rec["attempts"]) + 1, "utc": utc_now(), "commit": commit, "dirty": dirty,
           "prereg_status": prereg_status(), "checkpoint_sha256": getattr(rep, "checkpoint_sha256", None),
           "n_windows": len(kept), "n_errors": n_err, "accuracy": acc_w,
           "n_windows_dropped": len(recs) - len(kept), "n_windows_no_imagery": len(recs) - len(have),
           "n_pixels": None, "n_pixel_errors": None, "accuracy_pixel": None}
    if arm == "mangrove":
        att.update(pixel_accuracy(have, pred))
        gated = att["accuracy_pixel"]
    else:
        gated = acc_w
    att["gap_points"] = (gated - A["ai2_accuracy"]) * 100
    att["pass"] = bool(abs(gated - A["ai2_accuracy"]) <= A["gate_tolerance"][imagery_route] + EPS)
    att = json.loads(json.dumps(e78.jsonable(att)))                 # the file's and the ledger's form, identical
    append_ledger(arm, out_dir, att)                                # the count first, so a crash cannot lose it
    rec["attempts"].append(att)
    rec["passed"] = att["pass"]
    rec["closed"] = (not att["pass"]) and len(rec["attempts"]) >= MAX_GATE_ATTEMPTS
    dump(rec, path)
    log(f"gate {arm} attempt {att['attempt']}: accuracy {gated:.4f} against Ai2's {A['ai2_accuracy']:.3f} "
        f"(gap {att['gap_points']:+.2f} points, tolerance {rec['tolerance_points']:.1f}): "
        f"{'PASS' if att['pass'] else 'FAIL'}")
    return rec


def cmd_gate(args):
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} says the preregistration is {status}. The gate scores "
              "validation windows and runs only after freezing.", flush=True)
        return 2
    if args.arm == "nandi":
        st = nandi_status()
        if st["status"] != "readable and pinned":
            print(f"arm N not run: {st['reason']}", flush=True)
            return 0
    out_dir = args.out_dir or OUT
    try:
        # the ledger is read before any download or model load: a passed or closed gate needs no model, a file that
        # disagrees with its ledger is refused
        attempts, _ = reconcile_gate(args.arm, out_dir, gate_record(args.arm, out_dir)[0])
        if (attempts and attempts[-1]["pass"] is True) or len(attempts) >= MAX_GATE_ATTEMPTS:
            rec = run_gate(args.arm, None, [], out_dir)
        else:
            rep, windows = build_arm(args)
            rec = run_gate(args.arm, rep, windows, out_dir)
    except GateRefused as ex:
        print(f"refused: {ex}", flush=True)
        return 2
    return 0 if rec["passed"] else 1


def checkpoint_path(args):
    """(path, sha256) of the arm's checkpoint: the pinned download, or a local --ckpt that must hash to the pin."""
    A = ARMS[args.arm]
    if not args.ckpt:
        return hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
    got = sha256(args.ckpt)
    if A["model_sha256"] and got != A["model_sha256"]:
        raise RuntimeError(f"--ckpt {args.ckpt}: sha256 {got}, pinned {A['model_sha256']} "
                           f"({A['model']}@{A['model_revision'][:8]}); only the pinned checkpoint is scored")
    return args.ckpt, got


def build_arm(args, model_id="OLMOEARTH_V1_BASE"):
    ckpt, sha = checkpoint_path(args)                       # refused off the pin before any data is read
    windows_root, _ = resolve_data(args.arm, args.data or default_data_dir(args.arm), download=not args.offline)
    rep = Replica(args.arm, ckpt, model_id=model_id, checkpoint_sha256=sha)
    return rep, arm_windows(args.arm, windows_root)


# ----------------------------------------------------------------------------- the full run
def compute_units(arm, rep, windows, log=print, k3_cap=None, synthetic=False, out_dir=None):
    """Logits, controls and K3 on the validation windows; K2's frequencies and K3's fit on the training windows.
    Refused without a verified gate pass, except on the synthetic smoke's windows."""
    A = ARMS[arm]
    if not synthetic:
        require_gate(arm, out_dir or OUT)
    recs, have, lg = predict_population(arm, rep, windows, log, synthetic=synthetic)
    rd_pred = lg.argmax(1)
    keep = np.array([r["label"] is not None for r in have])
    units = [r for r, k in zip(have, keep) if k]
    logits = lg[keep]
    pred = rd_pred[keep]
    label = np.array([r["label"] for r in units])
    # the training windows: labels for K2, imagery for K3
    train = [w for w in windows if w["split"] == "train"]
    tr_lab = read_units(arm, train, need="label", log=log)
    tr_ok = [w for w, r in zip(train, tr_lab) if r["label"] is not None]
    tr_labels = np.array([r["label"] for r in tr_lab if r["label"] is not None])
    k2, freq = class_rarity(pred, tr_labels, A["trained"])
    cap = k3_cap if k3_cap is not None else A["k3_cap"]
    rng = np.random.default_rng(SEED)
    if cap is not None and len(tr_ok) > cap:
        pick = np.sort(rng.choice(len(tr_ok), cap, replace=False))
        tr_ok = [tr_ok[i] for i in pick]
    tr_recs = [r for r in read_units(arm, tr_ok, need="model", log=log) if "series" in r and r["label"] is not None]
    xtr = k3_features(np.stack([r["series"] for r in tr_recs]), rep.normalize, arm)
    ytr = np.array([r["label"] for r in tr_recs])
    series = np.stack([r["series"] for r in units])
    xva = k3_features(series, rep.normalize, arm)
    prob, k3_fit = fit_no_encoder_classifier(xtr, ytr, A["trained"], xva)
    k3, k3_class = k3_signals(prob, A["trained"], pred)
    controls = {"k2_class_rarity": k2, **k3,
                **index_controls(arm, series, np.stack([r["ndvi3x3"] for r in units]) if arm != "mangrove" else None)}
    names = [r["name"] for r in units]
    lonlat = [window_lonlat(r["meta"]) for r in units]
    cl = [cluster_ids(arm, r["meta"], r["name"], ll) for r, ll in zip(units, lonlat)]
    clusters = [c[0] for c in cl]
    if arm == "nandi":
        clusters = merge_by_polygon(clusters, [r["meta"].get("options") for r in units])
    out = {"names": np.array(names), "lonlat": np.array(lonlat), "label": label, "logits": logits,
           "clusters": np.array(clusters), "clusters_coarse": np.array([c[1] for c in cl]) if cl and cl[0][1] else None,
           "controls": controls, "k3_prob": prob, "k3_class": k3_class,
           "n_groups": np.array([r["n_groups"] for r in units]),
           "stratum": np.array([str(v) for v in label]) if arm != "nandi" else
           np.array([str((r["meta"].get("options") or {}).get("source", "unrecorded")) for r in units])}
    meta = {"n_val_windows": len(recs), "n_val_with_imagery": len(have), "n_units": int(keep.sum()),
            "dropped": {}, "k2_training_frequency": freq, "k3": k3_fit, "k3_cap": cap,
            "n_train_windows_labelled": int(tr_labels.size), "n_train_for_k3": len(tr_recs)}
    for r in recs:
        if r["label"] is None:
            _count(meta["dropped"], r["drop"])
    if arm == "mangrove":
        out["pixel"] = pixel_accuracy(have, rd_pred)
    if arm == "nandi":
        # reported, not graded, never a control candidate (control_names reads the k2_ to k5_ prefixes only)
        controls["reported_context_shift_instability"] = context_shift(rep, units, pred)
    if arm == "awf":
        meta["exp21_agreement"] = exp21_agreement(names, pred)
    return out, meta


def exp21_agreement(names, pred, path=os.path.join(OUT, "exp21_finetuned_awf.csv")):
    """Arm A, reported: how many windows get exp21's recorded prediction (its 16-px crop)."""
    import csv
    if not os.path.exists(path):
        return {"note": f"{path} not found"}
    rec = {r["window"].replace("\\", "/").split("/")[-1]: int(r["pred"]) for r in csv.DictReader(open(path))
           if r["crop"] == "16"}
    both = [(rec[n], int(p)) for n, p in zip(names, pred) if n in rec]
    return {"n_compared": len(both), "n_same_prediction": int(sum(a == b for a, b in both)),
            "n_not_in_exp21": int(sum(n not in rec for n in names))}


def context_shift(rep, units, pred):
    """Nandi, reported: the spread (standard deviation) of the shift-0 class's probability over crop shifts 0 to 3."""
    probs = []
    for s in range(4):
        inputs, locs = model_inputs("nandi", units, shift=s)
        lg = batched_logits(rep, inputs, locs, [r["n_groups"] for r in units], BATCH["nandi"])
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        probs.append(p[np.arange(len(pred)), pred])
    return np.std(np.stack(probs), axis=0)


def write_units(units, path):
    flat = {}
    for k, v in units.items():
        if v is None:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}/{kk}"] = np.asarray(vv)
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, **{k: v for k, v in flat.items() if v.dtype != object})
    return {"path": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path, "bytes": os.path.getsize(path),
            "sha256": sha256(path), "arrays": sorted(flat)}


def read_units_file(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    units = {"controls": {}, "pixel": None}
    for k, v in d.items():
        if k.startswith("controls/"):
            units["controls"][k.split("/", 1)[1]] = v
        elif k.startswith("pixel/"):
            units.setdefault("pixel_raw", {})[k.split("/", 1)[1]] = v.item()
        else:
            units[k] = v
    units["pixel"] = units.pop("pixel_raw", None)
    return units


def summary_path(out_dir):
    return os.path.join(out_dir, "exp89_summary.json")


def run_arm(arm, out_dir, rep=None, windows=None, units_file=None, draws=DRAWS, n_boot=BOOT, log=print, k3_cap=None,
            synthetic=False):
    """The run past both guards: units (computed, or read from a units file), the grading, the summary. It checks the
    gate itself (require_gate); grading a units file needs the pass but not the ledger, which stays on the cluster."""
    t0 = time.time()
    if not synthetic:
        if prereg_status() != "frozen":
            raise GateRefused(f"{os.path.relpath(PLAN, ROOT)} is not frozen: nothing is graded")
        require_gate(arm, out_dir, ledger_required=not units_file)
    if units_file:
        units, meta = read_units_file(units_file), {"from_units_file": units_file}
        units_info = {"path": units_file, "sha256": sha256(units_file)}
    else:
        units, meta = compute_units(arm, rep, windows, log, k3_cap=k3_cap, synthetic=synthetic, out_dir=out_dir)
        units_info = write_units(units, os.path.join(out_dir, f"exp89_units_{arm}.npz"))
        meta["replica"] = rep.info()
    res = grade_units(units, arm, draws=draws, n_boot=n_boot)
    res.update({"inputs": meta, "units_file": units_info, "seconds": time.time() - t0,
                "gate": gate_passed(arm, out_dir)[1]})
    path = summary_path(out_dir)
    summary = json.load(open(path)) if os.path.exists(path) else {
        "experiment": "exp89: the package's three outputs on Ai2's fine-tuned models",
        "prereg_page": os.path.relpath(PLAN, ROOT), "arms": {}}
    commit, dirty = git_state()
    summary.update({"prereg_status": prereg_status(), "updated_utc": utc_now(), "commit": commit, "dirty": dirty,
                    "config": {"seed": SEED, "draws": draws, "bootstrap": n_boot, "review_budgets": REVIEW_BUDGETS,
                               "estimate_budgets": EST_BUDGETS, "delta": DELTA, "error_floor": ERROR_FLOOR,
                               "p2_auroc": P2_AUROC, "p3_points": P3_POINTS, "p4_cover": P4_COVER,
                               "p4_width": P4_WIDTH, "p5_violation": P5_VIOLATION, "p5_lead": P5_LEAD,
                               "k5_b02": K5_B02, "config_revision": CONFIG_REVISION}})
    summary["arms"][arm] = res
    summary["arms"]["nandi"] = summary["arms"].get("nandi") if arm == "nandi" else nandi_status_safe()
    dump(summary, path)
    g = res["prereg"]
    log(f"{arm}: " + ", ".join(f"{p} {g[p]['holds']}" for p in ("P1", "P2", "P3", "P4", "P5")) +
        f" | {res['n_errors']} errors of {res['n_units']} | {res['seconds']:.0f}s")
    return res


def nandi_status_safe():
    try:
        return nandi_status()
    except Exception as ex:                                            # offline: say so, never guess
        return {"status": "not run", "reason": ARMS["nandi"]["not_run_reason"], "hub": {"error": repr(ex)}}


def cmd_run(args):
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} says the preregistration is {status}. No validation window is "
              "scored before freezing; --inventory and --smoke-torch --real (training windows) are allowed.", flush=True)
        return 2
    out_dir = args.out_dir or OUT
    if args.arm == "nandi":
        st = nandi_status()
        if st["status"] != "readable and pinned":
            print(f"arm N not run: {st['reason']}", flush=True)
            return 0
    try:
        require_gate(args.arm, out_dir, ledger_required=not args.from_units)
        if args.from_units:
            res = run_arm(args.arm, out_dir, units_file=args.from_units)
        else:
            rep, windows = build_arm(args)
            res = run_arm(args.arm, out_dir, rep, windows)
    except GateRefused as ex:
        print(f"refused: {ex} for arm {args.arm} (exp89_gate_{args.arm}.json). Nothing is graded until it passes.",
              flush=True)
        return 2
    return 0 if res["prereg"]["complete"] else 1


# ----------------------------------------------------------------------------- S2 code on synthetic data
SYN_CRS = "EPSG:32737"


def write_tif(path, arr, crs, transform):
    import rasterio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(arr)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype.name, crs=crs, transform=transform) as dst:
        dst.write(arr)


def synthetic_dataset(root, arm, n_train=24, n_val=16, seed=0, size=None, dataset_bands="one_set"):
    """An rslearn dataset in the layout the investigation describes: windows/<group>/<name>/metadata.json (projection,
    bounds in pixels, time_range, options.split), layers/sentinel2[.k]/<band set>/geotiff.tif with `completed` markers
    for 12 item groups, items.json, and the label raster. Mangrove: 2x2 windows (and a few 4x4, to exercise the centre
    crop), one band set B01..B12 at 10 m, label_raster/label with classes 1..3 (a few mixed or 0 blocks). Nandi and
    AWF: `size` px windows with one labelled pixel in label/category (fill 10 or 9), the three band groups at 10, 20
    and 60 m (AWF's tar layout); one window with two labelled pixels and one month with no mosaic."""
    from rasterio.transform import Affine
    A = ARMS[arm]
    rng = np.random.default_rng(seed)
    wroot = os.path.join(root, "dataset", "windows", A["group"])
    groups = {"one_set": [("B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12", 1)],
              "three": [("B02_B03_B04_B08", 1), ("B05_B06_B07_B8A_B11_B12", 2), ("B01_B09", 6)]}[dataset_bands]
    names = []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        if arm == "mangrove":
            S = 4 if i % 11 == 5 else 2
        else:
            S = size or 24
        x0, y0 = 50000 + 7 * i, -100000 - 5 * i
        name = f"syn_{i:04d}" if arm != "awf" else f"task_{i % 4:02d}_point_{i}"
        wdir = os.path.join(wroot, name)
        os.makedirs(wdir, exist_ok=True)
        meta = {"group": A["group"], "name": name, "projection": {"crs": SYN_CRS, "x_resolution": 10, "y_resolution": -10},
                "bounds": [x0, y0, x0 + S, y0 + S], "time_range": ["2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"],
                "options": {"split": split}}
        with open(os.path.join(wdir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        tr = Affine(10, 0, x0 * 10, 0, -10, y0 * -10)              # rslearn: geo = pixel bounds x resolution
        cls = int(rng.choice(A["trained"]))
        if arm == "mangrove":
            lab = np.full((S, S), cls, dtype=np.int32)
            if i % 13 == 7:
                lab[0, 0] = 0
            if i % 17 == 3:
                lab[-1, -1] = (cls % 3) + 1
        else:
            lab = np.full((S, S), A["label_fill"], dtype=np.int32)
            r, c = int(rng.integers(0, S)), int(rng.integers(0, S))
            lab[r, c] = cls
            if i == 2:
                lab[(r + 1) % S, c] = cls
        layer, band = A["label_layer"]
        write_tif(os.path.join(wdir, "layers", layer, band, "geotiff.tif"), lab[None], SYN_CRS, tr)
        open(os.path.join(wdir, "layers", layer, "completed"), "w").close()
        for t in range(N_MONTHS):
            lname = "sentinel2" if t == 0 else f"sentinel2.{t}"
            for bs, factor in groups:
                bnames = bs.split("_")
                h, w = max(1, -(-S // factor)), max(1, -(-S // factor))
                base = 300 + 2500 * (cls == 1) + 150 * t
                data = (base + rng.integers(0, 400, (len(bnames), h, w))).astype(np.uint16)
                if arm != "mangrove" and i == 3 and t == 5:
                    data[:] = 0                                            # a month no mosaic covered
                write_tif(os.path.join(wdir, "layers", lname, bs, "geotiff.tif"), data, SYN_CRS,
                          Affine(10 * factor, 0, x0 * 10, 0, -10 * factor, y0 * -10))
            open(os.path.join(wdir, "layers", lname, "completed"), "w").close()
        with open(os.path.join(wdir, "items.json"), "w") as f:
            json.dump([{"layer_name": "sentinel2", "serialized_item_groups": []}], f)
        names.append(name)
    return os.path.join(root, "dataset"), names


def synthetic_tar(root, arm, **kw):
    """The synthetic dataset packed as a tar the way the Hub's mangrove.tar is fetched and extracted."""
    ds, names = synthetic_dataset(os.path.join(root, "src"), arm, **kw)
    path = os.path.join(root, f"synthetic_{arm}.tar")
    with tarfile.open(path, "w") as tf:
        tf.add(ds, arcname="dataset")
    return path, names


def synthetic_checkpoint(path, arm, model_id="OLMOEARTH_V1_NANO", seed=0):
    """A Lightning-style checkpoint with the encoder keys of a randomly initialised OlmoEarth encoder under rslearn's
    prefix, the arm's head keys, and a hyper_parameters entry pickling an rslearn object (which the loader must stub)."""
    import sys as _sys
    import types
    import torch
    torch.manual_seed(seed)
    enc = encoder_skeleton(model_id).encoder
    D = int(enc.embedding_size)
    A = ARMS[arm]
    sd = {ENCODER_PREFIXES[0] + k: v.clone() for k, v in enc.state_dict().items()}
    wk, bk = A["head_keys"]
    w = torch.randn(A["n_out"], D) * 0.05
    sd[wk] = w if A["head_kind"] == "linear" else w[:, :, None, None]
    b = torch.zeros(A["n_out"])
    b[list(A["untrained"])] = -100.0                                        # the untrained channel is never chosen
    sd[bk] = b
    # a stand-in rslearn package, registered only while saving, so the checkpoint pickles an rslearn global as a real
    # one does; it is gone before anything loads the file
    added = [m for m in ("rslearn", "rslearn.fake_config") if m not in _sys.modules]
    pkg, mod = types.ModuleType("rslearn"), types.ModuleType("rslearn.fake_config")
    pkg.__path__ = []
    cls = type("FakeConfig", (), {"__module__": "rslearn.fake_config"})
    mod.FakeConfig = cls
    pkg.fake_config = mod
    for m, obj in (("rslearn", pkg), ("rslearn.fake_config", mod)):
        if m in added:
            _sys.modules[m] = obj
    try:
        obj = cls()
        obj.patch_size = A["patch"]
        torch.save({"state_dict": sd, "hyper_parameters": {"config": obj}, "epoch": 1}, path)
    finally:
        for m in added:
            del _sys.modules[m]
    return path, D


def smoke_torch(args):
    """S2's code end to end on synthetic data, or (with --real) on the pinned checkpoint and real training windows."""
    import tempfile
    out_dir = args.out_dir or OUT
    if args.real:
        if args.arm == "nandi":
            st = nandi_status()
            if st["status"] != "readable and pinned":
                print(f"arm N not run: {st['reason']}", flush=True)
                return 0
        rep, windows = build_arm(args)
        res = model_smoke(args.arm, rep, windows, os.path.join(out_dir, f"exp89_s2_{args.arm}.json"))
        return 0 if res["ok"] else 1
    import torch
    t0 = time.time()
    # 1. the restricted loader: training-stack globals become inert stubs, a code-execution global is refused
    with tempfile.TemporaryDirectory() as tmp:
        ck, D = synthetic_checkpoint(os.path.join(tmp, "m.ckpt"), "mangrove")
        sd, how, top = load_checkpoint(ck)
        assert "stub" in how and "rslearn.fake_config.FakeConfig" in how, how
        assert set(top) == {"state_dict", "hyper_parameters", "epoch"}
        assert "rslearn" not in sys.modules, "a stub must not import the package it names"
        torch.save({"state_dict": {}, "x": os.system}, os.path.join(tmp, "bad.ckpt"))
        try:
            load_checkpoint(os.path.join(tmp, "bad.ckpt"))
        except RuntimeError as ex:
            assert "refused" in str(ex), ex
        else:
            raise AssertionError("a checkpoint that pickles os.system must be refused")
    report = {"loader": "stubs training-stack globals, refuses code-execution globals"}
    # 2. each arm: synthetic tar -> extraction -> inventory -> S2 -> gate -> run, past the freeze guard only
    for arm, kw in (("mangrove", {"n_train": 30, "n_val": 22}),
                    ("nandi", {"n_train": 6, "n_val": 4, "size": 24, "dataset_bands": "three"}),
                    ("awf", {"n_train": 10, "n_val": 6, "size": 63, "dataset_bands": "three"})):
        with tempfile.TemporaryDirectory() as tmp:
            tar_path, names = synthetic_tar(tmp, arm, **kw)
            data_dir = os.path.join(tmp, "data")
            extract_tar(tar_path, data_dir)
            ck, D = synthetic_checkpoint(os.path.join(tmp, "m.ckpt"), arm)
            sub_out = os.path.join(tmp, "out")
            # synthetic=True lets arm N's code run here, on synthetic data only; every real mode keeps refusing it
            ns = argparse.Namespace(arm=arm, data=data_dir, ckpt=ck, out_dir=sub_out, offline=True, real=False,
                                    from_units=None, synthetic=True)
            assert cmd_inventory(ns) == 0
            inv = json.load(open(os.path.join(sub_out, f"exp89_inventory_{arm}.json")))
            d = inv["dataset"]
            # oe_inferencex.awf lists only windows with one labelled pixel, so arm A never sees the two-pixel window
            n_train = kw["n_train"] - (arm == "awf")
            assert d["windows_per_split"] == {"train": n_train, "val": kw["n_val"]}, d["windows_per_split"]
            assert d["item_groups_per_window"] == {"12": n_train + kw["n_val"]}
            assert inv["checkpoint"]["head_expected"][ARMS[arm]["head_keys"][0]] is not None
            if arm == "mangrove":
                assert sum(d["dropped_per_split"].get("val", {}).values()) + d["units_per_split"]["val"] == kw["n_val"]
                assert "4x4" in d["window_px"]
            elif arm == "nandi":
                assert d["dropped_per_split"]["train"]["2 labelled pixels"] == 1
            rep = Replica(arm, ck, model_id="OLMOEARTH_V1_NANO", device="cpu")
            if arm == "nandi":
                # the input path is exp21's when given exp21's timestamps: same normalisation, mask and stamps
                from oe_inferencex import awf
                x = np.random.default_rng(1).uniform(0, 4000, (2, 16, 16, 12, 12)).astype(np.float32)
                awf.CROP = 16
                ours, theirs = rep.sample(x, "exp21"), awf.stacks_to_sample(list(x), device="cpu")
                assert torch.equal(ours.sentinel2_l2a, theirs.sentinel2_l2a)
                assert torch.equal(ours.sentinel2_l2a_mask.float(), theirs.sentinel2_l2a_mask.float())
                assert torch.equal(ours.timestamps.long(), theirs.timestamps.long())
                legacy = rep.sample(x, "rslearn").timestamps[0]
                assert legacy[:, 0].tolist() == [1] * 12 and legacy[:, 1].tolist() == list(range(12))
                assert legacy[:, 2].tolist() == [2024] * 12, "rslearn's legacy stamps: day 1, month index, 2024"
            windows = arm_windows(arm, find_windows_root(data_dir) if arm != "awf" else
                                  os.path.join(data_dir, "dataset", "windows"))
            s2 = model_smoke(arm, rep, windows, os.path.join(sub_out, f"exp89_s2_{arm}.json"), n=8)
            assert s2["one_batch_output_shape"][-1] == ARMS[arm]["n_out"] and s2["n_pred_untrained"] == 0, s2
            if arm != "awf":
                assert s2["timestamp_conventions"]["max_abs_logit_difference"] is not None
            g = run_gate(arm, rep, windows, sub_out, synthetic=True)
            assert set(g) == set(GATE_KEYS), sorted(g)
            assert set(g["attempts"][0]) == set(ATTEMPT_KEYS), sorted(g["attempts"][0])
            res = run_arm(arm, sub_out, rep, windows, draws=20, n_boot=20, k3_cap=12, synthetic=True)
            try:
                predict_population(arm, rep, windows)
            except RuntimeError as ex:
                assert "not frozen" in str(ex) or prereg_status() == "frozen"
            else:
                assert prereg_status() == "frozen", "a validation window was scored before freezing"
            assert os.path.exists(os.path.join(sub_out, f"exp89_units_{arm}.npz"))
            again = run_arm(arm, sub_out, units_file=os.path.join(sub_out, f"exp89_units_{arm}.npz"), draws=20,
                            n_boot=20, synthetic=True)
            assert again["inputs"]["from_units_file"]
            if prereg_status() != "frozen":
                # the guards on a draft page, on the synthetic outputs: no grading and no scoring without a pass
                for call in (lambda: run_arm(arm, sub_out, units_file=os.path.join(sub_out, f"exp89_units_{arm}.npz")),
                             lambda: compute_units(arm, rep, windows, out_dir=sub_out)):
                    try:
                        call()
                    except GateRefused:
                        pass
                    else:
                        raise AssertionError("a run on a draft page must be refused")
            assert g["attempts"][0]["checkpoint_sha256"] == rep.checkpoint_sha256 == sha256(ck)
            assert again["n_units"] == res["n_units"] and again["n_errors"] == res["n_errors"]
            if arm == "mangrove":
                assert res["accuracy"]["accuracy_pixel_micro"]["n_pixels"] > 0
            report[arm] = {"n_units": res["n_units"], "gate_accuracy": g["attempts"][0]["accuracy"],
                           "embedding": D, "s2_windows": s2["n_windows"]}
            print(f"smoke-torch {arm}: inventory, S2 ({s2['n_windows']} training windows), gate and run on a "
                  f"synthetic tar and checkpoint; {res['n_units']} units", flush=True)
    # 3. K3 is deterministic at seed 89
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((600, 20)), rng.integers(1, 4, 600)
    p_a, _ = fit_no_encoder_classifier(x, y, (1, 2, 3), x[:50])
    p_b, _ = fit_no_encoder_classifier(x, y, (1, 2, 3), x[:50])
    assert np.array_equal(p_a, p_b), "K3's fit must repeat exactly at seed 89"
    report["seconds"] = time.time() - t0
    print(f"smoke-torch OK in {report['seconds']:.0f}s: the restricted loader, every arm's reader, replica, S2, gate "
          "and run on synthetic data, K3 deterministic", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARMS), default="mangrove")
    ap.add_argument("--smoke", action="store_true", help="S1: numpy only, synthetic units with known answers")
    ap.add_argument("--smoke-torch", action="store_true", help="S2's code on a synthetic tar and checkpoint")
    ap.add_argument("--real", action="store_true", help="with --smoke-torch: S2 on the pinned checkpoint and real "
                                                        "TRAINING windows")
    ap.add_argument("--inventory", action="store_true", help="files, checkpoint keys, windows and labels per split")
    ap.add_argument("--gate", action="store_true", help="accuracy only; refused until the page is frozen")
    ap.add_argument("--from-units", default=None, help="the full run from a units file (grading only)")
    ap.add_argument("--data", default=None, help="where the arm's dataset is (or is extracted to)")
    ap.add_argument("--ckpt", default=None, help="a local checkpoint instead of the pinned download")
    ap.add_argument("--offline", action="store_true", help="no Hub access: use --data and --ckpt as they are")
    ap.add_argument("--out-dir", default=None, help="where the outputs go (default: exp/out)")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke(args.out_dir)
        return 0
    if args.smoke_torch:
        return smoke_torch(args)
    if args.inventory:
        return cmd_inventory(args)
    if args.gate:
        return cmd_gate(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())

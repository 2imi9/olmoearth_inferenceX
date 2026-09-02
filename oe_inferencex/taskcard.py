"""Task cards: resolve what a fine-tuned OlmoEarth model is, from its configs.

Layer 2 (reads configuration, generates no evidence). Every audit in this
repository had to rediscover the same facts by hand: which task a model was
fine-tuned for, its class legend and nodata value, which inputs and time
range it consumes, what its outputs look like, and which encoder version
(and therefore which signals) apply. This module reads those facts from the
authoritative sources and returns one structured card per model:

  encoder card   - HuggingFace config.json of the encoder checkpoint
                   (depth, width, heads, register tokens, position encoding,
                   Sentinel-2 band groups per patch)
  project card   - olmoearth_projects/olmoearth_run_data/<project>/model.yaml
                   (task type, classes, nodata, inputs, output channels),
                   olmoearth_run.yaml (window size, resolution, split
                   protocol) and docs/<project>.md (stated goal)
  dataset card   - the olmoearth_lcc dataset README (export band table and
                   class legends of the production change product)

From the card, audit settings follow: whether the output is dense (boundary
cues apply), how many classes (nine-class boundary scores are low-margin
proxies, exp16), whether the encoder has several band-set tokens per patch
(band-set disagreement exists only for v1, exp19), and how to score
confidence (logit margin, exp13 audit).

Usage:
    python -m oe_inferencex.taskcard awf mozambique_lulc --encoder allenai/OlmoEarth-v1-Base
    python -m oe_inferencex.taskcard --all
"""
import argparse
import json
import re
import urllib.request
from dataclasses import dataclass, field, asdict

RAW = "https://raw.githubusercontent.com/allenai/olmoearth_projects/main"
PROJECTS = ["awf", "ecosystem_type_mapping", "fields_of_the_world", "forest_loss_driver",
            "kenya_lulc_croptype", "lfmc", "mangrove", "mozambique_lulc", "nandi",
            "satlas_solar_farm", "togo_cropland"]
S2_DEFAULT_GROUPS = [["B02", "B03", "B04", "B08"], ["B05", "B06", "B07", "B8A", "B11", "B12"], ["B01", "B09"]]


def _get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "oe-inferencex-taskcard"}), timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


@dataclass
class TaskCard:
    name: str
    kind: str                       # encoder | project | dataset
    encoder: dict = field(default_factory=dict)
    task: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    goal: str = ""
    sources: list = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- encoder
def encoder_card(repo_id):
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
    enc = cfg.get("model", {}).get("encoder_config", {})
    tok = enc.get("tokenization_config", {}) or {}
    ov = (tok.get("overrides") or {}).get("sentinel2_l2a", {}) or {}
    groups = ov.get("band_groups") or S2_DEFAULT_GROUPS
    version = re.search(r"OlmoEarth-(v[0-9_]+)", repo_id)
    card = TaskCard(name=repo_id, kind="encoder", sources=[f"https://huggingface.co/{repo_id}/blob/main/config.json"])
    card.encoder = {
        "version": version.group(1).replace("_", ".") if version else "unknown",
        "depth": enc.get("depth"), "embedding_size": enc.get("embedding_size"), "num_heads": enc.get("num_heads"),
        "num_register_tokens": enc.get("num_register_tokens", 0),
        "position_encoding": enc.get("spatial_pos_encoding", "absolute (no rotary keys in config)"),
        "sentinel2_band_groups_per_patch": len(groups), "sentinel2_band_groups": groups,
        "sentinel2_bands_used": sorted({b for g in groups for b in g}),
        "supported_modalities": enc.get("supported_modality_names"),
        "band_dropout_rate": enc.get("band_dropout_rate", 0.0),
        "run_name": cfg.get("run_name"),
    }
    dropped = sorted({b for g in S2_DEFAULT_GROUPS for b in g} - set(card.encoder["sentinel2_bands_used"]))
    if dropped:
        card.warnings.append(f"Sentinel-2 bands not tokenized by this version: {dropped}")
    card.audit = {
        "band_set_disagreement_available": len(groups) > 1,
        "tiling_instability_note": "rotary encoding did not reduce sub-patch instability (exp19)" if "rope" in str(card.encoder["position_encoding"]) else "absolute position encoding; striping artifact documented for v1",
    }
    return card


# ---------------------------------------------------------------- project
def _walk(node, pred, path=""):
    """Yield (path, dict) for every dict in a nested structure satisfying pred."""
    if isinstance(node, dict):
        if pred(node):
            yield path, node
        for k, v in node.items():
            yield from _walk(v, pred, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, pred, f"{path}[{i}]")


def _task_from_node(node):
    cp = node.get("class_path", "")
    args = node.get("init_args", {}) or {}
    t = {"class": cp.rsplit(".", 1)[-1]}
    if "SegmentationTask" in cp:
        t["type"] = "segmentation (dense per-pixel classes)"
        t["num_classes"] = args.get("num_classes")
        t["nodata_value"] = args.get("nodata_value")
        t["zero_is_invalid"] = args.get("zero_is_invalid")
        names = {}
        for mname, m in (args.get("other_metrics") or {}).items():
            idx = (m.get("init_args") or {}).get("class_idx")
            if idx is not None:
                base = re.sub(r"_(precision|recall|f1|accuracy|iou)$", "", mname)
                names.setdefault(int(idx), base)
        t["classes"] = {int(k): v for k, v in sorted(names.items())} if names else {}
        if t["num_classes"] and len(t["classes"]) < (t["num_classes"] - (1 if t["nodata_value"] is not None and t["nodata_value"] < t["num_classes"] else 0)):
            t["legend_note"] = "class names recovered from per-class metric definitions only; unnamed indices have no metric in the config"
    elif "ClassificationTask" in cp:
        t["type"] = "classification (one label per window)"
        t["classes"] = {i: c for i, c in enumerate(args.get("classes", []))}
        t["num_classes"] = len(t["classes"]) or None
    elif "Regression" in cp:
        t["type"] = "regression (dense per-pixel value)" if "PerPixel" in cp else "regression"
        t["nodata_value"] = args.get("nodata_value")
        t["num_classes"] = None
    elif "DetectionTask" in cp or "detection" in cp.lower():
        t["type"] = "detection"
    else:
        t["type"] = "unknown"
    return t


def project_card(project):
    import yaml
    card = TaskCard(name=project, kind="project")
    my = f"{RAW}/olmoearth_run_data/{project}/model.yaml"
    ry = f"{RAW}/olmoearth_run_data/{project}/olmoearth_run.yaml"
    dm = f"{RAW}/docs/{project}.md"
    card.sources = [my, ry, dm]
    try:
        model = yaml.safe_load(_get(my))
    except Exception as exc:
        card.warnings.append(f"model.yaml unavailable: {type(exc).__name__}"); model = {}
    try:
        run = yaml.safe_load(_get(ry))
    except Exception as exc:
        card.warnings.append(f"olmoearth_run.yaml unavailable: {type(exc).__name__}"); run = {}
    try:
        doc = _get(dm)
        paras = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip() and not p.strip().startswith("#")]
        card.goal = re.sub(r"\s+", " ", paras[0])[:400] if paras else ""
    except Exception as exc:
        card.warnings.append(f"docs/{project}.md unavailable: {type(exc).__name__}")

    # tasks (handles MultiTask wrappers)
    tasks = [(p, _task_from_node(n)) for p, n in _walk(model, lambda d: "class_path" in d and ".tasks." in str(d.get("class_path")) and "Task" in str(d.get("class_path")) and "MultiTask" not in str(d.get("class_path")))]
    card.task = {"tasks": {p.split(".")[-1] if "." in p else p: t for p, t in tasks}} if len(tasks) > 1 else (tasks[0][1] if tasks else {"type": "unknown"})
    # encoder + decoder output channels
    for p, n in _walk(model, lambda d: "model_id" in (d.get("init_args") or {})):
        card.encoder["model_id"] = n["init_args"]["model_id"]; card.encoder["patch_size"] = n["init_args"].get("patch_size")
    outs = [n["init_args"]["out_channels"] for _, n in _walk(model, lambda d: "out_channels" in (d.get("init_args") or {}))]
    if outs:
        card.outputs["decoder_out_channels"] = outs
    # inputs
    inputs = {}
    for _, n in _walk(model, lambda d: "inputs" in d and isinstance(d.get("inputs"), dict)):
        for key, spec in n["inputs"].items():
            if isinstance(spec, dict) and "layers" in spec:
                inputs[key] = {"layers": spec.get("layers"), "n_timesteps": len(spec.get("layers") or []),
                               "bands": spec.get("bands"), "is_target": bool(spec.get("is_target"))}
    card.inputs = inputs
    # window / split protocol from olmoearth_run.yaml
    w = {}
    for _, n in _walk(run, lambda d: "window_buffer" in d or "window_resolution" in d or "grid_size" in d or "nodata_value" in d):
        for k in ("window_buffer", "window_resolution", "grid_size", "nodata_value"):
            if k in n and k not in w:
                w[k] = n[k]
    if "window_buffer" in w:
        w["window_size_px"] = 2 * int(w["window_buffer"]) + 1
    splitter = [n.get("class_path") for _, n in _walk(run, lambda d: "splitter" in str(d.get("class_path", "")).lower() or "data_splitter" in str(d.get("class_path", "")))]
    if splitter:
        w["splitter"] = splitter[0].rsplit(".", 1)[-1]
    card.windows = w
    card.audit = _audit_settings(card)
    return card


def _audit_settings(card):
    t = card.task if "type" in card.task else next(iter(card.task.get("tasks", {}).values()), {})
    dense = str(t.get("type", "")).startswith(("segmentation", "regression (dense"))
    n_cls = t.get("num_classes")
    a = {
        "output_is_dense": dense,
        "boundary_cue_applies": dense,
        "n_classes": n_cls,
        "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties" if n_cls else "n/a (regression)",
        "expert_reference_required": True,
        "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels",
    }
    if n_cls and n_cls >= 5:
        a["note"] = "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate"
    mid = card.encoder.get("model_id", "")
    a["band_set_disagreement_available"] = ("V1_2" not in str(mid)) if mid else None
    return a


# ---------------------------------------------------------------- dataset (olmoearth_lcc)
def lcc_card():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("allenai/olmoearth_lcc", "README.md", repo_type="dataset")
    md = open(p, encoding="utf-8").read()
    card = TaskCard(name="allenai/olmoearth_lcc (production change product)", kind="dataset",
                    sources=["https://huggingface.co/datasets/allenai/olmoearth_lcc"])
    bands = re.findall(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|$", md, flags=re.M)
    card.outputs = {"export_bands": {int(i): {"name": n, "description": d} for i, n, d in bands}}
    m = re.search(r"\*\*Land cover classes\*\*.*?:\s*(.*?)\n\n", md, flags=re.S)
    if m:
        classes = dict(re.findall(r"(\d+) = `([^`]+)`", m.group(1)))
        card.task = {"type": "land cover change (dense); per-pixel change probability plus argmax classes",
                     "land_cover_classes": {int(k): v for k, v in classes.items()}, "nodata_value": 0}
    enc = re.search(r"Encoder:.*?\((https?://[^)]+)\)", md)
    if enc:
        card.encoder["repo"] = enc.group(1)
    card.goal = "Detect recent land cover change from Sentinel-2 time series (16 quarterly + 4 biweekly images), continent scale."
    card.audit = {"output_is_dense": True, "boundary_cue_applies": True,
                  "probabilities_in_export": "binary-change probability (band 1) and top-1 probability of argmax classes (bands 6-7); no full per-class distribution",
                  "confidence_scoring": "top-1 probability bands as shipped",
                  "band_set_disagreement_available": False, "note": "encoder v1.2-Base: single Sentinel-2 band-set token per patch (exp19)"}
    return card


# ---------------------------------------------------------------- rendering
def to_markdown(cards):
    out = ["# Task cards", "", "Resolved from configuration sources by `oe_inferencex/taskcard.py`; no evidence is generated here.", ""]
    for c in cards:
        out.append(f"## {c.name}  ({c.kind})")
        if c.goal:
            out.append(f"Goal: {c.goal}")
        for section in ("encoder", "task", "inputs", "windows", "outputs", "audit"):
            d = getattr(c, section)
            if d:
                out.append(f"- **{section}**: `{json.dumps(d, default=str)[:900]}`")
        if c.warnings:
            out.append(f"- warnings: {c.warnings}")
        out.append(f"- sources: {', '.join(c.sources)}")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--encoder", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="exp/out/taskcards.json")
    a = ap.parse_args()
    cards = []
    encoders = a.encoder or (["allenai/OlmoEarth-v1-Base", "allenai/OlmoEarth-v1_2-Base"] if a.all else [])
    for e in encoders:
        cards.append(encoder_card(e))
    for p in (PROJECTS if a.all else a.projects):
        try:
            cards.append(project_card(p))
        except Exception as exc:
            cards.append(TaskCard(name=p, kind="project", warnings=[f"failed: {type(exc).__name__}: {exc}"]))
    if a.all:
        cards.append(lcc_card())
    json.dump([asdict(c) for c in cards], open(a.out, "w"), indent=1, default=str)
    open("docs/method/taskcards.md", "w", encoding="utf-8").write(to_markdown(cards))
    for c in cards:
        t = c.task if "type" in c.task else {"type": "multi-task: " + ", ".join(c.task.get("tasks", {}).keys())}
        print(f"{c.kind:<8} {c.name:<40} {t.get('type','')[:48]:<50} classes={t.get('num_classes')} dense={c.audit.get('output_is_dense')} bandset={c.audit.get('band_set_disagreement_available')} warn={len(c.warnings)}")
    print(f"wrote {a.out} and docs/method/taskcards.md")


if __name__ == "__main__":
    main()

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
    python -m oe_inferencex.taskcard awf mozambique_lulc --encoder allenai/OlmoEarth-v1-Base --out cards.json
    python -m oe_inferencex.taskcard --all --out exp/out/taskcards.json --md docs/method/taskcards.md   # the record
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict

RAW = "https://raw.githubusercontent.com/allenai/olmoearth_projects/main"
GH_API = "https://api.github.com/repos/allenai/olmoearth_projects/contents/olmoearth_run_data"
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
    fallbacks = []                              # a renamed or re-nested key used to fall back silently to v1's layout
    if not enc:
        fallbacks.append("config.json has no model.encoder_config; depth, width and heads are unknown")
    if not ov.get("band_groups"):
        fallbacks.append("no sentinel2_l2a band_groups override in the config; v1's three-group layout is assumed")
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
    card.warnings.extend(fallbacks)
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


_LISTING_UNREACHABLE = {}     # project -> why the directory listing could not be read (a rate limit is the usual cause)


def _variants(project):
    """[(variant, model file, run file)]: the plain pair, or one per model_<variant>.yaml when a project publishes
    several. kenya_lulc_croptype has published model_cropland.yaml and model_maize.yaml since 2026-01-15 and no
    model.yaml, and its card used to fall back to an unknown task asserted as regression. A project the listing does
    not know is refused; the listing being unreachable (rate limit) falls back to the plain pair."""
    _LISTING_UNREACHABLE.pop(project, None)
    try:
        names = [x["name"] for x in json.loads(_get(f"{GH_API}/{project}"))]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LookupError(f"olmoearth_projects has no olmoearth_run_data/{project}") from None
        _LISTING_UNREACHABLE[project] = f"HTTP {exc.code}" + (", GitHub's API rate limit" if exc.code in (403, 429) else "")
        return [("", "model.yaml", "olmoearth_run.yaml")]
    except Exception as exc:
        _LISTING_UNREACHABLE[project] = type(exc).__name__
        return [("", "model.yaml", "olmoearth_run.yaml")]
    if "model.yaml" in names:
        return [("", "model.yaml", "olmoearth_run.yaml")]
    vs = sorted(n[len("model_"):-len(".yaml")] for n in names if n.startswith("model_") and n.endswith(".yaml"))
    if not vs:
        raise LookupError(f"olmoearth_run_data/{project} has no model.yaml or model_<variant>.yaml")
    return [(v, f"model_{v}.yaml", f"olmoearth_run_{v}.yaml") for v in vs]


def project_cards(project):
    """One card per published model of a project (most have one; kenya_lulc_croptype has two)."""
    return [project_card(project, variant=v, model_file=m, run_file=r) for v, m, r in _variants(project)]


def _timesteps(spec, dataset):
    """How many time steps an input stacks. With load_all_item_groups each layer contributes every item group the
    dataset's query returns, set by query_config.max_matches in dataset.json; the card used to count layers and said
    1 where AWF and Nandi stack 12 monthly scenes and three other projects 8 (audit 2026-09-22)."""
    layers = spec.get("layers") or []
    if not spec.get("load_all_item_groups"):
        return len(layers), None
    counts = []
    for layer in layers:
        qc = (((dataset or {}).get("layers") or {}).get(layer.split(".")[0], {}).get("data_source") or {}).get("query_config") or {}
        if "max_matches" not in qc:
            return None, f"layer {layer} stacks all its item groups; their number is set in dataset.json, which was not read"
        counts.append(int(qc["max_matches"]))
    return sum(counts), None


def project_card(project, variant="", model_file="model.yaml", run_file="olmoearth_run.yaml"):
    import yaml
    card = TaskCard(name=f"{project}:{variant}" if variant else project, kind="project")
    base = f"{RAW}/olmoearth_run_data/{project}"
    my, ry, dy, dm = f"{base}/{model_file}", f"{base}/{run_file}", f"{base}/dataset.json", f"{RAW}/docs/{project}.md"
    card.sources = [my, ry, dy, dm]
    fetched = 0
    try:
        model = yaml.safe_load(_get(my)); fetched += 1
    except Exception as exc:
        card.warnings.append(f"{model_file} unavailable: {type(exc).__name__}"); model = {}
    try:
        run = yaml.safe_load(_get(ry)); fetched += 1
    except Exception as exc:
        card.warnings.append(f"{run_file} unavailable: {type(exc).__name__}"); run = {}
    if not fetched:
        # a misspelled project used to yield a card asserting regression scoring and non-dense output, exit 0
        why = _LISTING_UNREACHABLE.get(project)
        if why:
            # a project that publishes only model_<variant>.yaml read as "not found" under a rate limit (review, 2026-09-23)
            raise LookupError(f"the project listing for {project} could not be read ({why}), so only {model_file} was tried, "
                              "and it could not be read either; the project may publish per-variant files. Retry later")
        raise LookupError(f"neither {model_file} nor {run_file} could be read for {project}; is the name right?")
    try:
        dataset = json.loads(_get(dy))
    except Exception:
        dataset = None
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
    # the explicit legend in olmoearth_run.yaml, which the card lists as a source and used to never read
    legend = {}
    for _, n in _walk(run, lambda d: "allowed_values" in d):
        for v in n.get("allowed_values") or []:
            if isinstance(v, dict) and "value" in v and "label" in v:
                legend.setdefault(int(v["value"]), str(v["label"]))
    if legend and "type" in card.task and card.task.get("type", "").startswith(("segmentation", "classification")):
        card.task["classes"] = dict(sorted(legend.items()))
        card.task["classes_source"] = f"{run_file} inference_results_config.classification_fields"
        card.task.pop("legend_note", None)
    # a pooling decoder emits one prediction per predict patch, copied to its pixels: not per-pixel output
    # (only the segmentation variant: a classification head pools by design and is one label per window anyway)
    pooling = [n.get("class_path") for _, n in _walk(model, lambda d: "SegmentationPoolingDecoder" in str(d.get("class_path", "")))]
    if pooling:
        card.outputs["decoder"] = pooling[0].rsplit(".", 1)[-1]
    # the task's class count against what the head actually emits
    if "type" in card.task and card.task.get("num_classes") and len(outs) == 1 and outs[0] != card.task["num_classes"]:
        card.warnings.append(f"the task config says num_classes={card.task['num_classes']} and the decoder emits "
                             f"{outs[0]} channels; the audit uses the decoder's {outs[0]}, which is what the model outputs")
        card.outputs["classes_emitted"] = outs[0]
    # inputs
    inputs = {}
    for _, n in _walk(model, lambda d: "inputs" in d and isinstance(d.get("inputs"), dict)):
        for key, spec in n["inputs"].items():
            if isinstance(spec, dict) and "layers" not in spec and "layers" in (spec.get("init_args") or {}):
                spec = spec["init_args"]                     # class_path: DataInput with init_args, once skipped silently
            if isinstance(spec, dict) and "layers" in spec:
                nt, why = _timesteps(spec, dataset)
                inputs[key] = {"layers": spec.get("layers"), "n_timesteps": nt,
                               "bands": spec.get("bands"), "is_target": bool(spec.get("is_target"))}
                if why:
                    inputs[key]["n_timesteps_note"] = why
    card.inputs = inputs
    # window / split protocol from olmoearth_run.yaml
    # Each key from the node it belongs to. The first match in YAML walk order used to win, so the partition
    # request's grid in degrees (0.1) was reported as the split grid (10.0), and reordering identical YAML changed it.
    w = {}
    for _, n in _walk(run, lambda d: "window_buffer" in d or "window_resolution" in d):
        for k in ("window_buffer", "window_resolution", "nodata_value"):
            if k in n and k not in w:
                w[k] = n[k]
        break
    for _, n in _walk(run, lambda d: "splitter" in str(d.get("class_path", "")).lower()):
        if "grid_size" in (n.get("init_args") or {}):
            w["grid_size"] = n["init_args"]["grid_size"]
        break
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
    pooled = "SegmentationPoolingDecoder" in str(card.outputs.get("decoder", ""))
    dense = str(t.get("type", "")).startswith(("segmentation", "regression (dense")) and not pooled
    n_cls = card.outputs.get("classes_emitted") or t.get("num_classes")
    a = {
        "output_is_dense": dense,
        "boundary_cue_applies": dense,
        "n_classes": n_cls,
        "confidence_scoring": "negative logit margin (top-1 minus top-2 logit); avoid 1-max-prob ties" if n_cls else "n/a (regression)",
        "expert_reference_required": True,
        "reference_caveat": "reference-product labels can make boundary-type signals look better than confidence (exp18); validate on expert labels",
    }
    if pooled:
        a["output_note"] = ("one prediction per predict patch copied to its pixels (a pooling decoder): the map is "
                            "block-constant, so the boundary cue applies only at the patch scale")
    if n_cls and n_cls >= 5:
        a["note"] = "with many classes the prediction-boundary score is largely a low-margin proxy (exp16); expect confidence to dominate"
    mid = str(card.encoder.get("model_id", ""))
    # only v1 tokenizes Sentinel-2 as several band sets; v1.1 and v1.2 read one group (encoder_card, from config.json)
    a["band_set_disagreement_available"] = (not any(v in mid for v in ("V1_1", "V1_2"))) if mid else None
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
                  "probabilities_in_export": "change probability (band 1) and the probabilities of the change-category heads (bands 6-7); no confidence for the land cover classes (bands 4-5) and no per-class distribution (exp20)",
                  "confidence_scoring": "none available for the class map; |2p - 1| of band 1 for the change decision; boundary fraction of the class map as triage (exp20)",
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
                # whole, never cut: a 900-character cut left unparseable JSON in the published page
                body = json.dumps(d, default=str).replace("`", "'")    # outside the f-string: Python 3.11 has no PEP 701
                out.append(f"- **{section}**: `{body}`")
        if c.warnings:
            out.append(f"- warnings: {c.warnings}")
        out.append(f"- sources: {', '.join(c.sources)}")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Task cards for OlmoEarth encoders and fine-tuned projects, from their configs.")
    ap.add_argument("projects", nargs="*", help="olmoearth_projects project names, e.g. awf mangrove")
    ap.add_argument("--encoder", action="append", default=[], help="an encoder repo id; repeatable")
    ap.add_argument("--all", action="store_true", help="every published project, both default encoders and the LCC product")
    ap.add_argument("--out", default="taskcards.json", help="the JSON to write")
    ap.add_argument("--md", default=None, help="the markdown to write (default: beside --out, same name, .md)")
    a = ap.parse_args(argv)
    if not (a.projects or a.encoder or a.all):
        ap.error("name projects, --encoder, or --all")      # with nothing named, it used to overwrite the committed cards with []
    if a.all and a.projects:
        ap.error("--all already covers every project; name projects without --all")
    md = a.md or os.path.splitext(a.out)[0] + ".md"
    cards = []
    encoders = (["allenai/OlmoEarth-v1-Base", "allenai/OlmoEarth-v1_2-Base"] if a.all else []) + a.encoder
    for e in encoders:
        cards.append(encoder_card(e))
    failed = []
    for p in (PROJECTS if a.all else a.projects):
        try:
            cards.extend(project_cards(p))
        except LookupError as exc:
            failed.append(str(exc))
    if a.all:
        cards.append(lcc_card())
    for path in (a.out, md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump([asdict(c) for c in cards], open(a.out, "w"), indent=1, default=str)
    open(md, "w", encoding="utf-8").write(to_markdown(cards))
    for c in cards:
        t = ({"type": "encoder"} if c.kind == "encoder" else c.task if "type" in c.task
             else {"type": "multi-task: " + ", ".join(c.task.get("tasks", {}).keys())})
        print(f"{c.kind:<8} {c.name:<40} {t.get('type','')[:48]:<50} classes={t.get('num_classes')} dense={c.audit.get('output_is_dense')} bandset={c.audit.get('band_set_disagreement_available')} warn={len(c.warnings)}")
    print(f"wrote {a.out} and {md}")
    if failed:
        raise SystemExit("not found: " + "; ".join(failed))


if __name__ == "__main__":
    main()

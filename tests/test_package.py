"""The package's public surface: what a user imports, and what a plain install refuses clearly."""
import importlib
import re

import pytest

import oe_inferencex as ox


def test_version_matches_pyproject():
    text = open("pyproject.toml", encoding="utf-8").read()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == ox.__version__


def test_every_exported_name_is_importable():
    for name in ox.__all__:
        assert hasattr(ox, name), name


def test_core_needs_only_numpy():
    for mod in ("assess", "compare", "explain", "calibrate", "metrics", "stats", "signals", "reliability", "cli"):
        importlib.import_module(f"oe_inferencex.{mod}")


def test_encoder_modules_refuse_a_plain_install_clearly():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"olmoearth-inferencex\[encoder"):
            importlib.import_module("oe_inferencex.evidence")
    else:
        pytest.skip("the encoder extra is installed here")


# ----------------------------------------------------------------------------- taskcard, audited 2026-09-22
def _cards():
    import json, os
    return {c["name"]: c for c in json.load(open(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "taskcards.json")))}


def test_task_cards_count_stacked_time_steps_not_layers():
    """The cards counted layers, so models that stack 12 monthly scenes in one layer read as 1 time step."""
    c = _cards()
    for name, n in (("awf", 12), ("nandi", 12), ("mozambique_lulc", 8), ("togo_cropland", 8), ("fields_of_the_world", 8)):
        assert c[name]["inputs"]["sentinel2_l2a"]["n_timesteps"] == n, name


def test_task_cards_read_the_published_legend_the_split_grid_and_every_model():
    c = _cards()
    assert len(c["ecosystem_type_mapping"]["task"]["classes"]) == 60 and c["ecosystem_type_mapping"]["windows"]["grid_size"] == 10.0
    assert {"kenya_lulc_croptype:cropland", "kenya_lulc_croptype:maize"} <= set(c) and "kenya_lulc_croptype" not in c
    assert c["mangrove"]["inputs"] and c["satlas_solar_farm"]["inputs"]                    # DataInput specs are read
    assert c["mangrove"]["audit"]["output_is_dense"] is False                              # a pooling decoder is block-constant
    assert "decoder" not in c["forest_loss_driver"]["outputs"]                             # a classification head pools by design
    assert any("emits 3 channels" in w for w in c["togo_cropland"]["warnings"])


def test_taskcard_cli_refuses_to_run_with_nothing_named(tmp_path):
    import pytest
    from oe_inferencex.taskcard import main
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit):
        main(["--all", "awf", "--out", str(tmp_path / "c.json")])


def test_band_set_flag_follows_the_encoder_version():
    from oe_inferencex.taskcard import TaskCard, _audit_settings
    for mid, want in (("OLMOEARTH_V1_BASE", True), ("OLMOEARTH_V1_1_BASE", False), ("OLMOEARTH_V1_2_BASE", False)):
        card = TaskCard(name="x", kind="project", task={"type": "segmentation (dense per-pixel classes)", "num_classes": 3},
                        encoder={"model_id": mid})
        assert _audit_settings(card)["band_set_disagreement_available"] is want, mid

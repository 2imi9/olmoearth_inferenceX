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

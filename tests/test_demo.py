"""`oe-inferencex demo`: the first run a stranger makes. It must work with numpy alone, write what it says it writes,
say what the tool is worth and what it does not do, and keep telling the truth about its own sample."""
import json
import struct
import zlib

import numpy as np
import pytest

from oe_inferencex.cli import main
from oe_inferencex.demo import FONT, SAMPLE, _text, sample_scene, write_png


def _png_size(path):
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", raw[16:24])


@pytest.mark.parametrize("flags,scores,height", [([], "sample_probabilities.npy", 768 + 45 + 45), (["--made-up"], "sample_logits.npy", 768 + 45)])
def test_demo_writes_the_picture_the_sample_and_the_files_assess_writes(tmp_path, capsys, flags, scores, height):
    assert main(["demo", "--out", str(tmp_path / "d")] + flags) == 0
    d = tmp_path / "d"
    for f in ("review_set.png", scores, "sample_truth.npy", "audit/assessment.json", "audit/explanation.json", "audit/review_set_05pct.csv"):
        assert (d / f).exists(), f
    assert _png_size(d / "review_set.png") == (3 * 768 + 2 * 12, height)      # three panels under their titles, a legend on the real map
    said = capsys.readouterr().out
    for phrase in ("What it is worth", "Why a window is flagged", "What it does not do", "it is not the evidence",
                   "oe-inferencex assess your_map.tif", f"{scores}"):
        assert phrase in said, phrase
    assert ("CC BY 4.0" in said) == (not flags), "the real sample is credited; the made-up one has nobody to credit"


def test_the_real_sample_is_the_one_the_selection_record_chose_and_the_run_reports_its_pinned_numbers(tmp_path, capsys):
    """The tile was chosen by a rule so that the demo cannot flatter; the record of that choice and the bundled file must
    name the same tile, and what the run prints must be what exp/out pins."""
    meta = json.loads(str(np.load(SAMPLE)["meta"]))
    sel = json.load(open("exp/out/demo_sample_selection.json"))
    assert meta["tile"] == sel["chosen"]["tile"] and meta["n_candidates"] == sel["n_candidates"] == 18
    caps = sorted(c["capture_05"] for c in sel["candidates"])
    assert sel["chosen"]["capture_05"] == caps[(len(caps) - 1) // 2], "the lower median, not a better tile"
    main(["demo", "--out", str(tmp_path / "d")])
    said = capsys.readouterr().out
    got = json.load(open(tmp_path / "d" / "audit" / "assessment.json"))["against_reference"]
    pinned = json.load(open("exp/out/demo_sample_audit.json"))
    assert got["error_rate"] == pytest.approx(pinned["error_rate"])
    assert got["error_capture_at_budget"]["0.05"]["precision_in_set"] == pytest.approx(pinned["review_sets"]["0.05"]["precision_in_set"])
    assert "Of the 5% the tool flags, 67% are really wrong" in said and "picks at random (19%)" in said
    assert "the most any review could" in said and "no review of 5% can cover a map that is 19%" in said, \
        "a reader who sees red outside the flagged windows must be told what any 5% could hold"


def test_the_made_up_scene_stays_honest(tmp_path):
    """Its error rate and capture are set to resemble a real flood map of the record, not to flatter."""
    main(["demo", "--made-up", "--out", str(tmp_path / "d")])
    r = json.load(open(tmp_path / "d" / "audit" / "assessment.json"))["against_reference"]
    assert 0.05 <= r["error_rate"] <= 0.15
    assert 0.15 <= r["error_capture_at_budget"]["0.05"]["errors_captured_fraction"] <= 0.45


def test_the_demo_is_deterministic_and_the_png_writer_round_trips(tmp_path):
    a, _ = sample_scene(seed=0)
    assert np.array_equal(a, sample_scene(seed=0)[0]) and not np.array_equal(a, sample_scene(seed=1)[0])
    rgb = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "x.png", rgb)
    raw = (tmp_path / "x.png").read_bytes()
    idat = raw[raw.index(b"IDAT") + 4: raw.index(b"IEND") - 8]
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(5, 1 + 7 * 3)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:].reshape(5, 7, 3), rgb)


def test_the_picture_carries_its_own_words():
    """A picture passed on alone must still say what it shows, so the titles are drawn with a built-in font."""
    assert all(len(rows) == 7 and max(rows) < 32 for rows in FONT.values())
    assert set("NO LABELS USED: THE 5% TO CHECK FIRST (RED), OF A RANDOM-0123456789.") <= set(FONT)
    img = np.full((30, 200, 3), 255, np.uint8)
    _text(img, 4, 4, "5% a")
    assert (img != 255).any() and (img[:, 150:] == 255).all()

"""`oe-inferencex demo`: the first run a stranger makes. It must work with numpy alone, write what it says it writes,
and keep telling the truth about its own made-up scene if someone retunes that scene."""
import json
import struct

import numpy as np

from oe_inferencex.cli import main
from oe_inferencex.demo import sample_scene, write_png


def test_demo_writes_the_picture_the_sample_and_the_files_assess_writes(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "d")]) == 0
    d = tmp_path / "d"
    for f in ("review_set.png", "sample_logits.npy", "sample_truth.npy", "audit/assessment.json", "audit/explanation.json",
              "audit/review_set_05pct.csv"):
        assert (d / f).exists(), f
    png = (d / "review_set.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (3 * 768 + 2 * 12, 768 + 45)   # three panels under their titles
    said = capsys.readouterr().out
    assert "illustration, not evidence" in said and "oe-inferencex assess your_map.tif" in said
    assert "sample_logits.npy --logits --out my_audit" in said, "the gentler next step: the real command on the sample"


def test_the_picture_carries_its_own_words():
    """A picture passed on alone must still say what it shows, so the titles are drawn with a built-in font."""
    from oe_inferencex.demo import FONT, _text
    assert all(len(rows) == 7 and max(rows) < 32 for rows in FONT.values())
    assert set("NO LABELS USED: THE 5% TO CHECK FIRST (RED), A RANDOM-0123456789.") <= set(FONT)
    img = np.full((30, 200, 3), 255, np.uint8)
    _text(img, 4, 4, "5% a")
    assert (img != 255).any() and (img[:, 150:] == 255).all()


def test_the_demo_scene_stays_honest(tmp_path):
    """The printed sentence compares the review set with a random one, so the scene must keep errors that confidence
    finds well above chance, and an error rate like a real map's rather than a flattering one."""
    main(["demo", "--out", str(tmp_path / "d")])
    r = json.load(open(tmp_path / "d" / "audit" / "assessment.json"))["against_reference"]
    assert 0.05 <= r["error_rate"] <= 0.15
    cap = r["error_capture_at_budget"]["0.05"]["errors_captured_fraction"]
    assert 0.15 <= cap <= 0.45, "well above a random 5%, and not a number no real map in the record reaches"


def test_the_demo_is_deterministic_and_the_png_writer_round_trips(tmp_path):
    a, _ = sample_scene(seed=0)
    b, _ = sample_scene(seed=0)
    assert np.array_equal(a, b) and not np.array_equal(a, sample_scene(seed=1)[0])
    rgb = np.random.default_rng(0).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "x.png", rgb)
    import zlib
    raw = (tmp_path / "x.png").read_bytes()
    idat = raw[raw.index(b"IDAT") + 4: raw.index(b"IEND") - 8]
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(5, 1 + 7 * 3)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:].reshape(5, 7, 3), rgb)

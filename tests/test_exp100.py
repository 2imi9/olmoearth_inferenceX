"""exp100 (docs/plan/alignment.md): the shift check. Written before the page was frozen and before any shifted reading
of exp98's or exp99's maps: the peak and its refinement against surfaces whose maximum is known exactly, the rule's
three calls, the resampler, the reading of accepted classes, the refusal of a reading that does not give the recorded
errors, and the smoke (a synthetic map with a planted offset, and exp98's and exp99's own smoke inputs). No network.
Needs rasterio, shapely and pyproj (the geo extra); skipped without them."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp100_alignment as e100   # noqa: E402
import exp98_awf_deployment as e98   # noqa: E402

S = e100.SHIFT


def quadratic(south, east, s=S):
    """A surface that is exactly a paraboloid with its vertex at (south, east) px from the labelled pixel."""
    i, j = np.mgrid[0:2 * s + 1, 0:2 * s + 1]
    return 1.0 - (i - s - south) ** 2 - 0.5 * (j - s - east) ** 2


@pytest.mark.parametrize("south,east", [(0.0, 0.0), (0.3, -0.2), (-0.45, 0.4), (1.25, -2.4), (2.0, 1.0)])
def test_peak_recovers_a_paraboloids_vertex(south, east):
    p = e100.peak(quadratic(south, east))
    assert p["south_px"] == pytest.approx(south, abs=1e-12)
    assert p["east_px"] == pytest.approx(east, abs=1e-12)
    assert (p["best_south_px"], p["best_east_px"]) == (round(south), round(east))
    assert not p["at_edge"]


def test_peak_on_the_edge_is_unrefined_and_called_an_offset():
    p = e100.peak(quadratic(3.4, 0.0))
    assert p["at_edge"] and p["best_south_px"] == 3 and p["south_px"] == 3.0
    assert e100.call(p, {"south_px": [3.0, 3.0], "east_px": [0.0, 0.0]}) == "offset"


def test_ties_take_the_position_nearest_the_labelled_pixel():
    M = np.zeros((2 * S + 1, 2 * S + 1))
    M[S - 2, S] = M[S, S + 1] = M[S + 3, S + 3] = 1.0
    p = e100.peak(M)
    assert (p["best_south_px"], p["best_east_px"]) == (0, 1)
    flat = e100.peak(np.ones((2 * S + 1, 2 * S + 1)))
    assert (flat["south_px"], flat["east_px"]) == (0.0, 0.0)          # no peak: no refinement


def test_refine_is_zero_without_a_peak_and_half_at_a_tied_neighbour():
    assert e100._refine(1.0, 1.0, 1.0) == 0.0
    assert e100._refine(0.0, 1.0, 1.0) == pytest.approx(0.5)
    assert e100._refine(1.0, 1.0, 0.0) == pytest.approx(-0.5)


@pytest.mark.parametrize("iv,want", [
    ({"south_px": [-0.5, 0.5], "east_px": [-0.1, 0.2]}, "aligned"),
    ({"south_px": [0.51, 1.2], "east_px": [-0.1, 0.2]}, "offset"),
    ({"south_px": [-0.1, 0.1], "east_px": [-1.4, -0.6]}, "offset"),
    ({"south_px": [0.2, 0.7], "east_px": [-0.1, 0.1]}, "not determined"),
    ({"south_px": [-0.6, 0.6], "east_px": [0.0, 0.0]}, "not determined"),
])
def test_the_rule(iv, want):
    assert e100.call({"at_edge": False}, iv) == want


def test_the_resampler_keeps_each_strata_count_of_clusters():
    clusters = np.array([0, 0, 1, 2, 2, 2, 3, 4])
    strata = np.array(["a", "a", "a", "a", "a", "a", "b", "b"])
    draw = e100._resampler(clusters, strata)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = draw(rng)
        picked = [(strata[k], clusters[k]) for k in idx]
        assert all(s == "b" for s, c in picked if c in (3, 4)) and all(s == "a" for s, c in picked if c < 3)
        sizes = {0: 2, 1: 1, 2: 3, 3: 1, 4: 1}
        n_a = sum(1 for k in idx if strata[k] == "a")
        n_b = sum(1 for k in idx if strata[k] == "b")
        # three clusters drawn in a, two in b: the counts are sums of three and two cluster sizes
        assert n_a in {sizes[x] + sizes[y] + sizes[z] for x in range(3) for y in range(3) for z in range(3)}
        assert n_b == 2


def test_surfaces_read_the_accepted_classes_and_never_channel_nine():
    B = np.zeros((2, e98.N_OUT, 2 * S + 1, 2 * S + 1))
    B[:, 2] = 0.5
    B[:, 4] = 0.3
    B[:, 9] = 0.2
    B[1, :, 0, 0] = np.nan
    score, agree, full, _ = e100.surfaces(B, [(2,), (4, 9)])
    assert np.allclose(score[0], 0.5) and np.allclose(score[1][1:, 1:], 0.5)
    assert agree[0].all() and not agree[1][1:, 1:].any()
    assert list(full) == [True, False]
    B[0, 9] = 0.6
    _, agree, _, _ = e100.surfaces(B, [(9,), (4,)])
    assert agree[0].all()                       # accepted only when a label accepts it; the crosswalk never does


def test_a_reading_that_misses_the_recorded_errors_is_refused():
    score = np.ones((3, 2 * S + 1, 2 * S + 1))
    agree = np.ones_like(score)
    agree[0, S, S] = 0
    with pytest.raises(ValueError, match="not the graded reading"):
        e100.set_record("V", score, agree, np.ones(3, bool), np.arange(3), recorded={"n": 3, "errors": 2})
    r = e100.set_record("V", score, agree, np.ones(3, bool), np.arange(3), n_boot=20, recorded={"n": 3, "errors": 1})
    assert r["reproduces_record"] and r["errors_at_labelled_pixel"] == 1


def test_prereg_status_reads_the_status_line():
    assert e100.prereg_status("# t\n\n**Status: frozen on 11 October 2026.**") == "frozen"
    assert e100.prereg_status("# t\n\n**Status: draft, written 10 October 2026.**") == "draft"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e100.smoke(out_dir=str(tmp_path_factory.mktemp("exp100")), log=lambda *a: None)


def test_smoke_recovers_the_planted_offsets(smoke):
    for key, r in smoke["planted"].items():
        want = r["planted"]
        assert r["n_full_block"] == e100.SMOKE_PLANTED_N
        assert r["estimate"]["south_px"] == pytest.approx(want["south_px"], abs=0.25)
        assert r["estimate"]["east_px"] == pytest.approx(want["east_px"], abs=0.25)
        assert r["call"] == ("aligned" if (want["south_px"], want["east_px"]) == (0, 0) else "offset"), key
        best = r["mean_agreement_surface"][want["south_px"] + S][want["east_px"] + S]
        assert best == pytest.approx(r["agreement_best_over_surface"])


def test_smoke_reads_exp98s_and_exp99s_points_as_they_did(smoke):
    sets = smoke["sets"]
    assert (sets["T"]["n"], sets["V"]["n"]) == (10, 40)        # exp98's smoke: 10 training, 40 graded points
    assert sets["S_strict"]["n"] == sets["S_lenient"]["n"] == 22
    assert sets["T"]["errors_at_labelled_pixel"] == 1          # one training label set wrong on purpose
    for r in sets.values():
        assert r["call"] in e100.CALLS
        assert r["n_full_block"] + r["n_excluded_partial_block"] == r["n"]
    assert smoke["mapping"]["exp98"]["reprojected"] == 1       # exp98's smoke puts one point in another CRS
    assert smoke["mapping"]["exp99"]["reprojected"] == 22      # every TimeSync plot is reprojected from WGS84
    e98.check_no_coordinates(smoke)

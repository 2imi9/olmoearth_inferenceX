"""oe_inferencex.compare on synthetic fixtures, the phi overflow regression, and the recorded exp52 / exp55 cross-tabs."""
import json
import os
import warnings

import numpy as np
import pytest

from oe_inferencex.assess import summary
from oe_inferencex.compare import compare_inferences, crosstab, disagreement, over_groups, phi, stability, where, which_side
from oe_inferencex.explain import cue_enrichment
from oe_inferencex.stats import TIE_TOL, sign_test

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def _table(n11, n10, n01, n00):
    a = np.array([1] * n11 + [1] * n10 + [0] * n01 + [0] * n00, dtype=bool)
    b = np.array([1] * n11 + [0] * n10 + [1] * n01 + [0] * n00, dtype=bool)
    return a, b


def _random_graded(rng, shape, n_classes, p_err=0.3, p_ok=0.8):
    lab = rng.integers(0, n_classes, shape)
    a = np.where(rng.random(shape) < p_err, rng.integers(0, n_classes, shape), lab)
    b = np.where(rng.random(shape) < p_err, rng.integers(0, n_classes, shape), lab)
    return lab, a, b, rng.random(shape) < p_ok


def test_phi_by_hand_and_undefined():
    a, b = _table(3, 1, 1, 5)
    assert phi(a, b) == pytest.approx((3 * 5 - 1 * 1) / np.sqrt(4 * 6 * 4 * 6))
    assert phi(a, a) == 1.0 and phi(a, ~a) == -1.0
    assert np.isnan(phi(a, np.ones_like(a)))                       # constant partner: undefined, not None
    assert np.isnan(phi(np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)))
    assert phi(a, b) == phi(a.astype(np.float64), b.astype(int))    # bool, 0/1 float and int read alike
    assert phi(a.reshape(2, 5), b.reshape(2, 5)) == phi(a, b)       # any shape, compared element-wise
    assert type(phi(a, b)) is float
    with pytest.raises(ValueError):
        phi(a, b[:-1])
    with pytest.raises(ValueError):
        phi(a.reshape(2, 5), b.reshape(5, 2))                        # equal size is not the same shape


def test_phi_no_int64_overflow_at_600k_windows():
    """exp51's copy noted the failure: the product of the four margins overflows int64 at this size."""
    rng = np.random.default_rng(0)
    n = 600_000
    a = rng.random(n) < 0.5
    b = a ^ (rng.random(n) < 0.3)
    n11, n10, n01 = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())
    n00 = n - n11 - n10 - n01
    assert (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00) > np.iinfo(np.int64).max
    v = phi(a, b)
    assert np.isfinite(v) and v == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)


@pytest.mark.parametrize("n", [3, 10, 1_000, 100_003, 1_000_000])
def test_phi_matches_corrcoef_at_every_size_without_warnings(n):
    rng = np.random.default_rng(n)
    a = rng.random(n) < 0.3
    b = a ^ (rng.random(n) < 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = phi(a, b)
    if a.any() and (~a).any() and b.any() and (~b).any():
        assert v == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)
    else:
        assert np.isnan(v)


def test_disagreement_rate_mask_and_groups():
    a = np.zeros((3, 2, 2), dtype=int)
    b = a.copy()
    b[0, 0, 0] = 1
    b[2, :, :] = 1
    ok = np.ones_like(a, dtype=bool)
    ok[2, 1, 1] = False
    d = disagreement(a, b, ok, groups=np.array([10, 20, 20]))
    assert (d["n"], d["n_disagree"]) == (11, 4) and d["rate"] == pytest.approx(4 / 11)
    assert d["mask"].shape == a.shape and d["mask"].sum() == 4 and not d["mask"][2, 1, 1]
    assert d["per_group"] == {10: {"n": 4, "n_disagree": 1, "rate": 0.25}, 20: {"n": 7, "n_disagree": 3, "rate": 3 / 7}}
    assert type(next(iter(d["per_group"]))) is int                  # native keys, not numpy scalars
    s = disagreement(a, b, ok, groups=np.array(["x", "y", "y"]))
    assert list(s["per_group"]) == ["x", "y"] and type(next(iter(s["per_group"]))) is str
    same = disagreement(a, b, ok, groups=np.array([10, 20, 20])[:, None, None])   # the (N, 1, 1) form
    assert same["per_group"] == d["per_group"]
    full = disagreement(a, b, ok, groups=np.broadcast_to(np.array([10, 20, 20])[:, None, None], a.shape))
    assert full["per_group"] == d["per_group"]
    e = disagreement(a, b, np.zeros_like(ok))
    assert e["n"] == 0 and np.isnan(e["rate"]) and not e["mask"].any()
    with pytest.raises(TypeError):
        disagreement(a.astype(float), b, ok)
    with pytest.raises(ValueError):
        disagreement(a, b[:2], ok)


def test_disagreement_group_without_valid_windows_and_wrong_length_groups():
    a = np.zeros((3, 2, 2), dtype=int)
    b = a.copy()
    b[0, 0, 0] = 1
    b[1] = 1
    ok = np.ones_like(a, dtype=bool)
    ok[1] = False                                                  # the second tile, alone in its group, has no valid window
    for groups in (np.array([10, 20, 30]), np.array(["x", "y", "z"]), ["x", "y", "z"]):
        pg = disagreement(a, b, ok, groups=groups)["per_group"]
        ids = list(pg)
        assert len(ids) == 3 and all(type(i) in (int, str) for i in ids)
        assert pg[ids[1]]["n"] == 0 and pg[ids[1]]["n_disagree"] == 0 and np.isnan(pg[ids[1]]["rate"])
        assert pg[ids[0]] == {"n": 4, "n_disagree": 1, "rate": 0.25} and pg[ids[2]] == {"n": 4, "n_disagree": 0, "rate": 0.0}
    for bad in (np.array([1, 2]), np.array([[1, 2], [3, 4]]), np.arange(4), np.zeros((3, 2, 2, 1))):
        with pytest.raises(ValueError, match="groups of shape"):       # a wrong-length vector must not become per-column ids
            disagreement(a, b, ok, groups=bad)


def test_where_is_cue_enrichment_on_the_disagreement_windows():
    rng = np.random.default_rng(1)
    mask = rng.random(500) < 0.2
    cue = mask ^ (rng.random(500) < 0.4)
    w = where(mask, {"boundary": cue})["boundary"]
    r = cue_enrichment(cue, mask, n_boot=0)
    assert w["share_disagree"] == r["share_errors"] and w["share_agree"] == r["share_correct"]
    assert w["enrichment"] == r["enrichment"] and w["rate_with_cue"] == r["precision"]
    assert (w["n"], w["n_disagree"], w["n_with_cue"]) == (r["n"], r["n_errors"], r["n_with_cue"])
    assert "boot_lo" not in w
    empty = where(np.zeros(5, dtype=bool), {"c": np.ones(5, dtype=bool)})["c"]
    assert np.isnan(empty["share_disagree"]) and np.isnan(empty["enrichment"])
    with pytest.raises(ValueError):
        where(mask, {"c": cue[:-1]})


def test_where_rejects_wrong_cue_shapes_and_is_silent_on_empty_masks():
    mask = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match=r"cue 'c' has shape \(4,\) \(4 windows\), the mask has shape \(5,\) \(5 windows\)"):
        where(mask, {"c": np.ones(4, dtype=bool)})
    with pytest.raises(ValueError, match="cue 'c'"):
        where(mask, {"c": np.ones((5, 1), dtype=bool)})           # equal size, another shape
    with warnings.catch_warnings():
        warnings.simplefilter("error")                             # explicit guards, never a numpy warning
        none = where(mask, {"c": np.ones(5, dtype=bool)})["c"]
        every = where(np.ones(5, dtype=bool), {"c": np.zeros(5, dtype=bool)})["c"]
        no_windows = where(np.zeros(0, dtype=bool), {"c": np.zeros(0, dtype=bool)})["c"]
        r = cue_enrichment(np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), n_boot=0)
    assert (none["n"], none["n_disagree"], none["share_agree"], none["rate_with_cue"]) == (5, 0, 1.0, 0.0)
    assert np.isnan(none["share_disagree"]) and np.isnan(none["enrichment"])
    assert every["share_disagree"] == 0.0 and np.isnan(every["share_agree"]) and np.isnan(every["rate_with_cue"])
    assert no_windows["n"] == 0 and all(np.isnan(no_windows[k]) for k in ("share_disagree", "share_agree", "enrichment", "rate_with_cue"))
    assert "n_boot" not in r and np.isnan(r["enrichment"])


def test_stability_matrix():
    rng = np.random.default_rng(2)
    m1 = rng.random(300) < 0.3
    m2 = m1 ^ (rng.random(300) < 0.1)
    s = stability({"one": m1, "two": m2, "const": np.zeros(300, dtype=bool)})
    P = np.array(s["phi"])
    assert s["names"] == ["one", "two", "const"] and P.shape == (3, 3)
    assert P[0, 0] == 1.0 and P[0, 1] == P[1, 0] == phi(m1, m2)
    assert np.isnan(P[2]).all() and np.isnan(P[:, 2]).all()
    assert s["median_pairwise_phi"] == phi(m1, m2)
    assert np.array(stability([m1, m2])["phi"]).shape == (2, 2)
    assert np.isnan(stability([m1])["median_pairwise_phi"]) and stability([])["phi"] == []
    with pytest.raises(ValueError):
        stability([m1.reshape(10, 30), m2.reshape(30, 10)])


def test_over_groups_distribution_and_sign_test():
    o = over_groups({"a": 0.2, "b": -0.1, "c": 0.3, "d": float("nan"), "e": None, "f": 0.0})
    assert (o["n_groups"], o["n_undefined"], o["w"], o["l"], o["t"]) == (4, 2, 2, 1, 1)
    assert o["median"] == pytest.approx(0.1) and o["share_flipped"] == 0.25 and o["flipped"] == ["b"]
    assert o["sign_p"] == sign_test(2, 1, "greater")
    arr = over_groups([0.2, -0.1, 0.3, np.nan, 0.0], groups=["a", "b", "c", "d", "f"])
    assert arr["flipped"] == ["b"] and arr["w"] == o["w"]
    pos = over_groups(np.array([0.2, -0.1]))
    assert pos["flipped"] == [1]
    none = over_groups({"a": float("nan")})
    assert none["n_groups"] == 0 and np.isnan(none["median"]) and none["sign_p"] == 1.0
    with pytest.raises(ValueError):
        over_groups([1.0, 2.0], groups=["a"])
    with pytest.raises(ValueError):
        over_groups({"a": 1.0}, groups=["a"])


def test_over_groups_native_ids_and_tie_tolerance():
    o = over_groups(np.array([0.2, -0.1, 0.3, np.inf], dtype=np.float32), groups=np.array([5, 6, 7, 8]))
    assert (o["n_groups"], o["n_undefined"], o["flipped"]) == (3, 1, [6]) and type(o["flipped"][0]) is int
    o = over_groups(np.array([0.2, -0.1, 0.3]), groups=np.array(["e1", "e2", "e3"]))
    assert o["flipped"] == ["e2"] and type(o["flipped"][0]) is str
    o = over_groups({np.int64(3): -0.5, np.str_("k"): np.float32(0.5)})
    assert o["flipped"] == [3] and type(o["flipped"][0]) is int
    o = over_groups(np.array([1, -1, 0]))                            # an integer array
    assert (o["w"], o["l"], o["t"], o["flipped"]) == (1, 1, 1, [1])
    tol = over_groups({"a": -1e-13, "b": -TIE_TOL, "c": -2 * TIE_TOL, "d": 1e-13})
    assert (tol["w"], tol["l"], tol["t"]) == (0, 1, 3) and tol["share_flipped"] == 0.25 and tol["flipped"] == ["c"]
    empty = over_groups(np.zeros(0))
    assert empty["n_groups"] == 0 and np.isnan(empty["share_flipped"]) and empty["flipped"] == []


def test_crosstab_identities():
    a, b = _table(4, 3, 2, 11)         # a errs on 7, b on 6, both on 4
    ok = np.ones_like(a)
    ok[-1] = False
    ct = crosstab(a, b, ok)
    assert (ct["n"], ct["errors_a"], ct["errors_b"], ct["corrected"], ct["broken"], ct["both"], ct["neither"]) == (19, 7, 6, 3, 2, 4, 10)
    assert ct["corrected"] + ct["both"] == ct["errors_a"] and ct["broken"] + ct["both"] == ct["errors_b"]
    assert ct["share_corrected"] == 3 / 7 and ct["phi"] == phi(a[ok > 0.5], b[ok > 0.5])
    z = crosstab(np.zeros(4), np.zeros(4), np.ones(4))
    assert z["share_corrected"] == 0.0 and np.isnan(z["phi"])


def test_crosstab_and_which_side_identities_on_random_data():
    rng = np.random.default_rng(5)
    for trial in range(30):
        shape = tuple(int(x) for x in rng.integers(1, 7, size=3))
        k = int(rng.integers(2, 5))
        lab, a, b, ok = _random_graded(rng, shape, k)
        ea, eb = a != lab, b != lab
        ct, ws, d = crosstab(ea, eb, ok), which_side(a, b, lab, ok), disagreement(a, b, ok)
        assert ct["n"] == int(ok.sum()) == ct["corrected"] + ct["broken"] + ct["both"] + ct["neither"]
        assert ct["errors_a"] == ct["corrected"] + ct["both"] == int(ea[ok].sum())
        assert ct["errors_b"] == ct["broken"] + ct["both"] == int(eb[ok].sum())
        assert ct["share_corrected"] == (ct["corrected"] / ct["errors_a"] if ct["errors_a"] else 0.0)
        assert ws["n_disagree"] == d["n_disagree"] == ws["a_right"] + ws["b_right"] + ws["neither"]
        assert ws["a_right"] == ct["broken"] and ws["b_right"] == ct["corrected"]     # holds for any number of classes
        assert ws["neither"] == d["n_disagree"] - ct["corrected"] - ct["broken"]
        if not np.isnan(ct["phi"]):
            assert ct["phi"] == pytest.approx(np.corrcoef(ea[ok], eb[ok])[0, 1], abs=1e-12)


def test_crosstab_reproduces_the_recorded_exp52_cross_tabs():
    """exp52 stored counts, not windows: rebuild each 2x2 table from n_windows, corrected, broken and both."""
    s = json.load(open(_need("exp52_summary.json")))
    for arm, res in s["results"].items():
        for split in ("bolivia", "test"):
            r = res[split]
            ct = r["vs_frozen_head"]
            n = r["n_windows"]
            neither = n - ct["frozen_errors"] - ct["broken"]
            ea = np.repeat([True, True, False, False], [ct["both"], ct["corrected"], ct["broken"], neither])
            eb = np.repeat([True, False, True, False], [ct["both"], ct["corrected"], ct["broken"], neither])
            got = crosstab(ea, eb, np.ones(n, dtype=bool))
            assert got["errors_a"] == ct["frozen_errors"] and (got["corrected"], got["broken"], got["both"]) == (ct["corrected"], ct["broken"], ct["both"])
            assert got["share_corrected"] == pytest.approx(ct["share_corrected"], abs=1e-12)
            assert got["phi"] == pytest.approx(ct["phi"], abs=1e-12), (arm, split)
            assert 1 - ct["frozen_errors"] / n == pytest.approx(r["frozen_window_accuracy"], abs=1e-12)


def test_phi_reproduces_the_recorded_exp55_pooled_cross_tab():
    """exp55's pooled S2-head vs S1-head cross-tab from its window count, accuracies and conditionals."""
    x = json.load(open(_need("exp55_summary.json")))["cross_tab_A_vs_B"]
    n = x["n_windows"]
    ea_n = round(n * (1 - x["acc_A_S2_head"]))
    both = round(ea_n * x["p_b_wrong_given_a_wrong"])
    broken = round((n - ea_n) * x["p_b_wrong_given_a_right"])
    ea = np.repeat([True, True, False, False], [both, ea_n - both, broken, n - ea_n - broken])
    eb = np.repeat([True, False, True, False], [both, ea_n - both, broken, n - ea_n - broken])
    assert phi(ea, eb) == pytest.approx(x["phi"], abs=1e-6)


def test_which_side_binary_equals_crosstab_on_the_disagreements():
    rng = np.random.default_rng(3)
    lab = rng.integers(0, 2, (5, 6, 6))
    a = lab ^ (rng.random(lab.shape) < 0.2)
    b = lab ^ (rng.random(lab.shape) < 0.2)
    ok = rng.random(lab.shape) < 0.9
    ws = which_side(a, b, lab, ok)
    ct = crosstab(a != lab, b != lab, ok)
    assert ws["a_right"] == ct["broken"] and ws["b_right"] == ct["corrected"] and ws["neither"] == 0
    assert ws["n_disagree"] == disagreement(a, b, ok)["n_disagree"]
    lab3 = rng.integers(0, 3, (4, 4))
    a3, b3 = (lab3 + 1) % 3, (lab3 + 2) % 3                        # both always wrong, always different
    m = which_side(a3, b3, lab3, np.ones_like(lab3))
    assert m["neither"] == 16 and m["share_a_right"] == 0.0
    assert np.isnan(which_side(lab3, lab3, lab3, np.ones_like(lab3))["share_a_right"])


def test_compare_inferences_schema_and_json_round_trip():
    rng = np.random.default_rng(4)
    lab = rng.integers(0, 2, (6, 8, 8))
    a = lab ^ (rng.random(lab.shape) < 0.1)
    b = lab ^ (rng.random(lab.shape) < 0.15)
    ok = rng.random(lab.shape) < 0.95
    groups = np.repeat([0, 1], 3)
    cue = rng.random(lab.shape) < 0.3
    free = compare_inferences(a, b, ok)
    assert free["per_group"] is None and free["where"] is None and free["graded"] is None
    assert free["n_disagree"] == disagreement(a, b, ok)["n_disagree"] and free["arrays"]["disagree"].shape == lab.shape
    out = compare_inferences(a, b, ok, groups=groups, labels=lab, cues={"cue": cue})
    assert set(out["per_group"]) == {0, 1} and out["where"]["cue"]["n"] == int(ok.sum())
    g = out["graded"]
    assert g["crosstab"] == crosstab(a != lab, b != lab, ok) and g["which_side"] == which_side(a, b, lab, ok)
    assert set(g["per_group"]) == {0, 1} and sum(v["n"] for v in g["per_group"].values()) == int(ok.sum())
    assert g["over_groups"]["n_groups"] == 2
    text = json.dumps(summary(out), allow_nan=False)               # the JSON-safe view: no arrays, NaN as null
    back = json.loads(text)
    assert "arrays" not in back and back["per_group"]["0"]["n"] == out["per_group"][0]["n"]
    empty = compare_inferences(a, b, np.zeros_like(ok), labels=lab)
    assert json.loads(json.dumps(summary(empty), allow_nan=False))["disagreement_rate"] is None
    with pytest.raises(TypeError):
        compare_inferences(a.astype(float), b, ok)
    with pytest.raises(ValueError):
        compare_inferences(a, b, ok, cues={"c": cue[:, :4, :4]})


def test_compare_inferences_per_group_crosstab_with_groups_lacking_disagreement():
    rng = np.random.default_rng(6)
    lab, a, b, ok = _random_graded(rng, (8, 5, 5), 3, p_err=0.2, p_ok=0.9)
    a[0:2] = b[0:2] = lab[0:2]                                     # group 0: both right everywhere, phi undefined
    b[2:4] = a[2:4]                                                # group 1: identical errors, no disagreement
    ok[4:6] = False                                                # group 2: no valid window
    groups = np.array(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = compare_inferences(a, b, ok, groups=groups[:, None, None], labels=lab, cues={"c": rng.random(lab.shape) < 0.4})
    per = out["graded"]["per_group"]
    assert list(per) == list(out["per_group"]) == ["g0", "g1", "g2", "g3"] and all(type(k) is str for k in per)
    g = np.repeat(groups, 25).reshape(lab.shape)
    for gid, ct in per.items():                                     # the vectorised path equals the cross-tab on the group's windows
        assert ct == crosstab(a != lab, b != lab, ok & (g == gid))
        assert ct["n"] == out["per_group"][gid]["n"]
    assert per["g0"]["errors_a"] == 0 and np.isnan(per["g0"]["phi"]) and per["g0"]["share_corrected"] == 0.0
    assert per["g1"]["corrected"] == per["g1"]["broken"] == 0 and per["g1"]["phi"] == 1.0
    assert per["g2"]["n"] == 0 and np.isnan(per["g2"]["phi"]) and np.isnan(out["per_group"]["g2"]["rate"])
    og = out["graded"]["over_groups"]
    assert og["n_groups"] == 4 and og["t"] >= 3 and og["n_undefined"] == 0
    assert og["w"] + og["l"] + og["t"] == 4 and all(type(i) is str for i in og["flipped"])
    back = json.loads(json.dumps(summary(out), allow_nan=False))
    assert back["graded"]["per_group"]["g0"]["phi"] is None and back["per_group"]["g2"]["rate"] is None
    same = compare_inferences(a, b, ok, groups=groups, labels=lab)   # the (N,) form gives the same groups
    assert same["graded"]["per_group"] == per and same["per_group"] == out["per_group"]
    with pytest.raises(ValueError, match="groups of shape"):
        compare_inferences(a, b, ok, groups=groups[:5], labels=lab)


def test_returned_keys_are_the_documented_ones():
    """Every key each function returns is named in its docstring, and nothing else is returned."""
    rng = np.random.default_rng(7)
    lab, a, b, ok = _random_graded(rng, (4, 3, 3), 2)
    groups = np.array([0, 0, 1, 1])
    m = disagreement(a, b, ok)["mask"]
    expected = {
        disagreement: (disagreement(a, b, ok, groups), {"n", "n_disagree", "rate", "mask", "per_group"}),
        where: (where(m, {"c": m})["c"], {"n", "n_disagree", "n_with_cue", "share_disagree", "share_agree", "enrichment", "rate_with_cue"}),
        stability: (stability([m, ~m]), {"names", "phi", "median_pairwise_phi"}),
        over_groups: (over_groups({"x": 1.0}), {"n_groups", "n_undefined", "median", "w", "l", "t", "share_flipped", "sign_p", "flipped"}),
        crosstab: (crosstab(a != lab, b != lab, ok), {"n", "errors_a", "errors_b", "corrected", "broken", "both", "neither", "share_corrected", "phi"}),
        which_side: (which_side(a, b, lab, ok), {"n_disagree", "a_right", "b_right", "neither", "share_a_right", "share_b_right"}),
        compare_inferences: (compare_inferences(a, b, ok, groups=groups, labels=lab, cues={"c": m}),
                             {"n_windows", "n_disagree", "disagreement_rate", "per_group", "where", "graded", "arrays"}),
    }
    for fn, (out, keys) in expected.items():
        assert set(out) == keys, fn.__name__
        assert all(k in fn.__doc__ for k in keys), fn.__name__
    full = expected[compare_inferences][0]
    assert set(full["graded"]) == {"crosstab", "which_side", "per_group", "over_groups"} and set(full["arrays"]) == {"disagree"}
    assert set(full["per_group"][0]) == {"n", "n_disagree", "rate"}
    assert all(k in compare_inferences.__doc__ for k in ("crosstab", "which_side", "per_group", "over_groups", "disagree"))

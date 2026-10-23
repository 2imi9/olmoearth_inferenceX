"""1.14.1: what the scan of the OlmoEarth Agent (23 October 2026) found the package lacked, each pinned.

The agent wraps this package for an LLM and, through its preregistered trials (exp86, exp87), met four things the
package did not say or check: the label budget the texts recommend for a random sample whose labels also serve
certify; a one-band map passed as logits that holds no logits (a probability, a height); the route that compares two
maps from labels where they differ, offered for maps of different dates; and which release wrote a result file."""
import json
import os
import typing

import numpy as np
import pytest

import oe_inferencex
from oe_inferencex import assess as assess_mod
from oe_inferencex import cli, mcp_server, plan
from oe_inferencex.assess import assess_prediction
from oe_inferencex.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text(path):
    with open(os.path.join(ROOT, path)) as f:
        return " ".join(f.read().split())


# ----------------------------------------------------------------------------- the label budget
def test_the_random_budget_is_one_number_and_not_one_where_certify_falls():
    """README, SKILL.md and the MCP server recommend one budget for a random sample whose labels serve estimate and
    certify: about 400. Until 1.14.1 SKILL.md and the server said 300, the budget the recorded experiments used, while
    README said 400 (a34fec7): at alpha 0.05 a zone level enters at 300, where even a perfect map certifies a zone only
    about half the time, and SKILL.md's next step was certify at alpha 0.05 on those labels."""
    for path in ("README.md", "skills/oe-inferencex/SKILL.md"):
        assert "--budget 400 --design random" in _text(path), path
        assert "--budget 300 --design random" not in _text(path), path
    server = _text("oe_inferencex/mcp_server.py")
    assert "experiments used 300)" not in server
    assert "about 400 for a random sample whose labels also serve certify: " + mcp_server.BUDGET_300 in \
        mcp_server.CARDS["sample"]
    budget = typing.get_type_hints(mcp_server.sample, include_extras=True)["budget"].__metadata__[0].description
    assert budget.startswith("Number of windows to label: about 400") and mcp_server.BUDGET_300 in budget
    for alpha in (0.05, 0.10):                                 # the alphas the docs' examples use
        assert 400 not in plan.entry_budgets(1.0, alpha).values()
    assert 300 in plan.entry_budgets(1.0, 0.05).values()
    # the reason, exactly: the smallest zone testable at 300 labels passes its own test about half the time on a
    # perfect map, and at 400 nearly always (prefix rule: it is tested first)
    assert plan.zone_level_probability(10 ** 6, 300, 0.15, 0.05, 0.0) < 0.6
    assert plan.zone_level_probability(10 ** 6, 400, 0.15, 0.05, 0.0) > 0.95


def test_assess_names_400_labels_on_a_large_map_and_a_census_on_a_small_one(tmp_path):
    rng = np.random.default_rng(0)
    big = rng.normal(0, 3, (96, 96))                          # 576 windows of 4 px
    small = rng.normal(0, 3, (72, 72))                        # 324 windows: under 400, a census is offered
    for name, x in (("big", big), ("small", small)):
        np.save(tmp_path / f"{name}.npy", x)
    a = mcp_server.assess(scores=str(tmp_path / "big.npy"), out_dir=str(tmp_path / "a"), logits=True)
    assert "a budget of about 400 (" + mcp_server.BUDGET_300 in a["next"]
    b = mcp_server.assess(scores=str(tmp_path / "small.npy"), out_dir=str(tmp_path / "b"), logits=True)
    assert "a budget of up to 324: the map has 324 windows" in b["next"]


# ----------------------------------------------------------------------------- a one-band map read as logits
def _logit_warnings(out):
    return [w for w in out["warnings"] if w.startswith("logit input:")]


def test_a_probability_map_read_as_logits_is_warned():
    p = np.random.default_rng(1).random((32, 32))
    out = assess_prediction(p, is_logit=True)
    assert _logit_warnings(out) == [assess_mod.LOGIT_RANGE_WARNING]
    assert "without --logits" in assess_mod.LOGIT_RANGE_WARNING     # the server writes it as logits=false
    # float32 rounding puts a probability one ulp above 1; read as probabilities that is accepted, and so here
    q = p.astype(np.float32)
    q[0, 0], q[0, 1] = 0.0, np.nextafter(np.float32(1), np.float32(2))
    assert _logit_warnings(assess_prediction(q, is_logit=True)) == [assess_mod.LOGIT_RANGE_WARNING]


@pytest.mark.parametrize("name", ["height", "height_with_bare_ground", "depth_with_dry_land", "class_ids"])
def test_a_map_with_no_value_below_zero_read_as_logits_is_warned(name):
    """A regression output with values above 1: a canopy height in metres, with or without bare ground at exactly 0, a
    flood depth that is 0 on dry land, and a map of class ids 0 to 5. The review of 1.14.1 found the first version
    silent wherever a 0 was present, which is the usual form of these maps."""
    rng = np.random.default_rng(2)
    x = {"height": rng.uniform(0.5, 30.0, (32, 32)),
         "height_with_bare_ground": np.where(rng.random((32, 32)) < 0.3, 0.0, rng.uniform(0.5, 30.0, (32, 32))),
         "depth_with_dry_land": np.where(rng.random((32, 32)) < 0.7, 0.0, rng.uniform(0.0, 3.0, (32, 32))),
         "class_ids": rng.integers(0, 6, (32, 32)).astype(float)}[name]
    assert _logit_warnings(assess_prediction(x, is_logit=True)) == [assess_mod.LOGIT_NONNEGATIVE_WARNING]


def test_a_map_with_no_value_above_zero_read_as_logits_is_warned_that_every_window_is_class_0():
    x = -np.random.default_rng(3).uniform(0.5, 30.0, (32, 32))
    out = assess_prediction(x, is_logit=True)
    assert _logit_warnings(out) == [assess_mod.LOGIT_NONPOSITIVE_WARNING]
    assert (out["arrays"]["pooled_argmax"] == 0).all()               # the warning's statement, checked


def test_logits_on_both_sides_of_zero_are_not_warned_and_no_data_is_left_out():
    x = np.random.default_rng(3).normal(0, 4, (32, 32))
    assert _logit_warnings(assess_prediction(x, is_logit=True)) == []
    # the only values below 0 are no-data: the valid ones are one-sided, and that is what is read
    y = np.abs(x) + 2.0
    mask = np.zeros_like(y, dtype=bool)
    mask[:4] = True
    y[:4] = -5.0
    out = assess_prediction(y, is_logit=True, nodata_mask=mask)
    assert _logit_warnings(out) == [assess_mod.LOGIT_NONNEGATIVE_WARNING]
    # a non-finite value is no-data too
    y[:4] = np.nan
    assert _logit_warnings(assess_prediction(y, is_logit=True)) == [assess_mod.LOGIT_NONNEGATIVE_WARNING]
    # rounding either way does not hide a one-sided map (the re-review of 1.14.1): a height with one pixel at -1e-7,
    # a negative map with one at +1e-7; a value past rounding does
    z = np.abs(x) + 2.0
    z[0, 0] = -1e-7
    assert _logit_warnings(assess_prediction(z, is_logit=True)) == [assess_mod.LOGIT_NONNEGATIVE_WARNING]
    z[0, 0] = -1e-3
    assert _logit_warnings(assess_prediction(z, is_logit=True)) == []
    w = -(np.abs(x) + 2.0)
    w[0, 0] = 1e-7
    assert _logit_warnings(assess_prediction(w, is_logit=True)) == [assess_mod.LOGIT_NONPOSITIVE_WARNING]


def test_the_command_line_prints_the_logit_warning_and_names_regression_in_its_refusal(tmp_path, capsys):
    rng = np.random.default_rng(4)
    height = np.where(rng.random((32, 32)) < 0.3, 0.0, rng.uniform(0.5, 30.0, (32, 32)))
    np.save(tmp_path / "height.npy", height)
    assert main(["assess", str(tmp_path / "height.npy"), "--logits", "--out", str(tmp_path / "a")]) == 0
    printed = capsys.readouterr().out
    assert "warning: " + assess_mod.LOGIT_NONNEGATIVE_WARNING in printed
    assert main(["sample", str(tmp_path / "height.npy"), "--logits", "--budget", "20", "--design", "random",
                 "--out", str(tmp_path / "s.csv")]) == 0
    assert "warning: " + assess_mod.LOGIT_NONNEGATIVE_WARNING in capsys.readouterr().out
    with pytest.raises(SystemExit, match="A regression output .* has no margin") as exc:
        main(["assess", str(tmp_path / "height.npy"), "--out", str(tmp_path / "b")])
    # compare cuts it into classes at a cut-off (as sample and plan do with --other); boundary's next step, a sample
    # of the one map, would refuse it again
    assert "compare cuts one into classes at --threshold" in str(exc.value) and "boundary" not in str(exc.value)
    # through the server, the refusal names the server's parameters
    with pytest.raises(mcp_server.Refused) as exc:
        mcp_server.assess(scores=str(tmp_path / "height.npy"), out_dir=str(tmp_path / "c"))
    said = mcp_server._mcp_words(str(exc.value))
    assert "compare cuts one into classes at threshold" in said and "--" not in said


def test_the_server_says_the_logit_warning_with_its_own_parameter_names(tmp_path):
    np.save(tmp_path / "p.npy", np.random.default_rng(5).random((32, 32)))
    out = mcp_server.assess(scores=str(tmp_path / "p.npy"), out_dir=str(tmp_path / "a"), logits=True)
    assert "every value lies in [0, 1] (to rounding), as probabilities do" in out["limits"]
    assert "(with logits=false)" in out["limits"] and "--logits" not in out["limits"]


# ----------------------------------------------------------------------------- two maps of different dates
def _two_maps(tmp_path):
    rng = np.random.default_rng(6)
    a = rng.integers(0, 3, (32, 32))
    b = np.where(rng.random((32, 32)) < 0.3, (a + 1) % 3, a)
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    return str(tmp_path / "a.npy"), str(tmp_path / "b.npy")


def test_sample_other_says_both_maps_must_describe_one_time(tmp_path, capsys):
    a, b = _two_maps(tmp_path)
    assert main(["sample", a, "--other", b, "--budget", "20", "--out", str(tmp_path / "p.csv")]) == 0
    assert f"note: {cli.OTHER_DATES_NOTE}" in capsys.readouterr().out
    side = json.load(open(tmp_path / "p.json"))
    assert "the class that is there at the time both maps describe" in side["how_to_label"]
    m = mcp_server.sample(scores=a, other=b, out_dir=str(tmp_path / "m"), budget=20)
    assert "they must describe the same time" in m["limits"]
    assert "the class that is there at the time both maps describe" in m["next"]


def test_compare_proposes_the_differing_windows_only_for_maps_of_one_time(tmp_path):
    a, b = _two_maps(tmp_path)
    same = mcp_server.compare(a=a, b=b, out_dir=str(tmp_path / "same"), date_a="2023-06-01", date_b="2023-06-01")
    assert "other set to map b" in same["next"] and "if both describe the same time" in same["next"]
    for i, (da, db, status) in enumerate((("2018-01-01/2018-12-31", "2023-01-01/2023-12-31", "different_time"),
                                          ("2023-01-01/2023-12-31", "2023-06-01/2024-05-31", "overlapping_time"))):
        out = mcp_server.compare(a=a, b=b, out_dir=str(tmp_path / f"d{i}"), date_a=da, date_b=db)
        assert out["summary"]["dates_status"] == status
        assert "other set to map b" not in out["next"], out["next"]
        assert "label it at that map's date" in out["next"]
    assert "for two maps of the same time" in mcp_server.INSTRUCTIONS
    assert "label it at its own date" in mcp_server.INSTRUCTIONS


def test_decide_follows_the_dates_too(tmp_path):
    """compare stopped proposing the differing windows across dates; decide, asked more_accurate of the same
    comparison, still proposed them (the review of 1.14.1)."""
    a, b = _two_maps(tmp_path)
    apart = ("--date-a", "2018-01-01/2018-12-31", "--date-b", "2023-01-01/2023-12-31")
    assert main(["compare", a, b, "--out", str(tmp_path / "apart"), *apart]) == 0
    assert main(["compare", a, b, "--out", str(tmp_path / "same")]) == 0
    with pytest.raises(SystemExit, match="each map at its own date") as exc:
        main(["decide", str(tmp_path / "apart" / "comparison.json"), "--ask", "more_accurate"])
    assert "--other" not in str(exc.value)
    with pytest.raises(SystemExit, match="sample with --other set to the second map"):
        main(["decide", str(tmp_path / "same" / "comparison.json"), "--ask", "more_accurate"])
    with pytest.raises(mcp_server.Refused, match="each map at its own date"):
        mcp_server.decide(result_json=str(tmp_path / "apart" / "comparison.json"), questions=["more_accurate"])
    # a class map cannot be sampled at random on its own, so the route names the scores or the confidence band, and
    # it says the two intervals do not hold jointly (the re-review of 1.14.1)
    with pytest.raises(SystemExit) as exc:
        main(["decide", str(tmp_path / "apart" / "comparison.json"), "--ask", "more_accurate"])
    assert "on a product's class map with its confidence band" in str(exc.value)
    assert "which do not hold jointly" in str(exc.value)
    # one date given: give the other first, since compare refuses to grade labels until then
    assert main(["compare", a, b, "--out", str(tmp_path / "one"), "--date-a", "2018-01-01/2018-12-31"]) == 0
    with pytest.raises(SystemExit, match="only one map's date was given: give both") as exc:
        main(["decide", str(tmp_path / "one" / "comparison.json"), "--ask", "more_accurate"])
    assert "run compare with --labels" not in str(exc.value)
    one = mcp_server.compare(a=a, b=b, out_dir=str(tmp_path / "one_m"), date_a="2018-01-01/2018-12-31")
    assert one["next"].startswith("Give both maps' dates (date_a and date_b) first")
    assert "compare again with labels" not in one["next"]
    # graded against labels of map b's date that cover only part of the differing windows: undetermined, and the
    # route to settle it is the one for maps of different dates
    lab = np.load(b)
    lab = np.where(np.arange(lab.size).reshape(lab.shape) % 3 == 0, -1, lab)
    np.save(tmp_path / "lab.npy", lab)
    assert main(["compare", a, b, "--out", str(tmp_path / "graded"), *apart, "--labels", str(tmp_path / "lab.npy"),
                 "--labels-date", "2023-01-01/2023-12-31"]) == 0
    assert main(["decide", str(tmp_path / "graded" / "comparison.json"), "--ask", "more_accurate"]) == 0
    r = json.load(open(tmp_path / "graded" / "comparison_decisions.json"))
    ans = r["answers"]["more_accurate"]
    assert ans["answer"] == "undetermined" and "each map at its own date" in ans["because"]
    assert "--other" not in ans["because"]
    m = mcp_server.decide(result_json=str(tmp_path / "graded" / "comparison.json"), questions=["more_accurate"],
                          out_dir=str(tmp_path / "md"))
    assert "each map at its own date" in m["next"] and "other set to map b" not in m["next"]


# ----------------------------------------------------------------------------- which release wrote a file
def test_every_result_file_names_the_release_and_command_that_wrote_it(tmp_path):
    """The golden test checks assess, sample, estimate and certify; here the rest: compare, sample --other and its
    estimate, decide and plan. `written_by` is the last key, so a reader who prints the file sees the result first."""
    a, b = _two_maps(tmp_path)
    assert main(["compare", a, b, "--out", str(tmp_path / "c")]) == 0
    assert main(["sample", a, "--other", b, "--budget", "20", "--out", str(tmp_path / "p.csv")]) == 0
    rows = open(tmp_path / "p.csv").read().splitlines()
    head, body = rows[0].split(","), [r.split(",") for r in rows[1:]]
    k = head.index("reference_class")
    for r in body:
        r[k] = r[head.index("class_a")]
    with open(tmp_path / "p.csv", "w") as f:
        f.write("\n".join([",".join(head)] + [",".join(r) for r in body]) + "\n")
    assert main(["estimate", str(tmp_path / "p.csv")]) == 0
    assert main(["decide", str(tmp_path / "p_estimate.json"), "--ask", "more_accurate"]) == 0
    assert main(["plan", "--windows", "10000", "--width", "0.1", "--out", str(tmp_path / "plan.json")]) == 0
    want = {"c/comparison.json": "compare", "p.json": "sample", "p_estimate.json": "estimate",
            "p_estimate_decisions.json": "decide", "plan.json": "plan"}
    for name, command in want.items():
        d = json.load(open(tmp_path / name))
        assert list(d)[-1] == "written_by", name
        assert d["written_by"] == {"package": "olmoearth-inferencex", "version": oe_inferencex.__version__,
                                   "command": command}, name


def _strip(tmp_path):
    """A 4-class probability map of one 1 x 300 strip, its truth and a two-value condition layer (patch 1)."""
    rs = np.random.RandomState(7)
    n = 300
    truth = rs.randint(0, 4, size=n)
    pred = np.where(rs.rand(n) < 0.9, truth, (truth + 1) % 4)
    p = np.full((4, 1, n), 0.1, np.float32)
    p[pred, 0, np.arange(n)] = 0.7
    p[:, 0, :] += rs.rand(4, n).astype(np.float32) * 0.01         # no ties
    p /= p.sum(0, keepdims=True)
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "cond.npy", (np.arange(n) % 2)[None, :])
    return str(tmp_path / "p.npy"), str(tmp_path / "cond.npy"), truth


def _fill(path, truth, field):
    rows = open(path).read().splitlines()
    head, body = rows[0].split(","), [r.split(",") for r in rows[1:]]
    k, col, mc = head.index(field), head.index("pixel_col"), head.index("map_class")
    for r in body:
        t = int(truth[int(r[col])])
        r[k] = str(t) if field == "reference_class" else str(int(int(r[mc]) != t))
    with open(path, "w") as f:
        f.write("\n".join([",".join(head)] + [",".join(r) for r in body]) + "\n")


def test_the_other_writers_stamp_their_files_too(tmp_path):
    """A sequential sample and its certificate, a sample by condition and its certificates, and a per-class audit
    and its estimate: each writer of the command line, not only the ones the golden outputs hold."""
    p, cond, truth = _strip(tmp_path)
    s = lambda *a: main(["sample", p, "--patch", "1", "--seed", "1", *a])
    assert s("--budget", "120", "--design", "sequential", "--alpha", "0.2", "--out", str(tmp_path / "seq.csv")) == 0
    _fill(tmp_path / "seq.csv", truth, "wrong")
    assert main(["certify", str(tmp_path / "seq.csv"), "--alpha", "0.2"]) == 0
    assert s("--budget", "120", "--condition", cond, "--out", str(tmp_path / "cond.csv")) == 0
    _fill(tmp_path / "cond.csv", truth, "wrong")
    assert main(["certify", str(tmp_path / "cond.csv"), "--alpha", "0.2"]) == 0
    assert s("--design", "audit", "--per-class", "10", "--out", str(tmp_path / "audit.csv")) == 0
    _fill(tmp_path / "audit.csv", truth, "reference_class")
    assert main(["estimate", str(tmp_path / "audit.csv"), "--alpha", "0.1"]) == 0
    want = {"seq.json": "sample", "seq_zone.json": "certify", "cond.json": "sample", "cond_zone.json": "certify",
            "audit.json": "sample", "audit_estimate.json": "estimate"}
    for name, command in want.items():
        d = json.load(open(tmp_path / name))
        assert list(d)[-1] == "written_by" and d["written_by"]["command"] == command, name
        assert d["written_by"]["version"] == oe_inferencex.__version__, name

"""exp86's scorer on synthetic traces (docs/plan/agent_trial_v2.md): a pass and a fail for every computed criterion,
with the three failures of the first trial (exp/out/agent_trial_2026-09-24.md) as fail cases: the review set ranked
with the most confident windows first, the correlation computed with Studio's no-data value (-1) as data, and a
simple-random-sample interval quoted for a stratified design.

The traces are written in the run-directory layout the plan fixes, by write_run below. The tool outputs are built
here by independent code in the shape the agent's tools return, never by the scorer's own functions."""
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp86_agent_trial_v2 as e86  # noqa: E402
from oe_inferencex import estimate as oe_estimate  # noqa: E402

B2 = e86.BRIEF_CONFIGS["B2/studio"]["brief"]
B3 = e86.BRIEF_CONFIGS["B3/studio"]["brief"]
B4 = e86.BRIEF_CONFIGS["B4/studio"]["brief"]
B5 = ("I labelled the windows of the design in design.json; the filled sheet is labels.csv. What is this map's error "
      "rate?")
B7 = ("Map A (a.json) describes 2019-03-01 and map B (b.json) describes 2019-03-20; they cover the same windows. "
      "Compare them: where do they differ, and which is right?")


# --------------------------------------------------------------------------------------------- writing a run
def _hash(args):
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_run(d, brief, calls, answer, studio=None, files=None, usage=None, seconds=42.0, meta=None,
              drop_trace_line=False):
    """One run directory as the driver writes it. `calls`: (name, arguments, result[, ok[, where]]) in dispatch order,
    where `where` may set the call's "id", "turn" and "t" (the time of its tool_call event; the result follows 0.5 s
    later). By default call i has id call{i}, turn i + 1 and t = i."""
    ws = os.path.join(d, "workspace")
    os.makedirs(ws, exist_ok=True)
    events, entries, lines = [], [], []
    for i, c in enumerate(calls):
        name, args, result = c[:3]
        ok = c[3] if len(c) > 3 else True
        where = c[4] if len(c) > 4 else {}
        cid, turn, t = where.get("id", f"call{i}"), where.get("turn", i + 1), float(where.get("t", i))
        events.append({"type": "tool_call", "turn": turn, "id": cid, "name": name, "arguments": args, "t": t})
        events.append({"type": "tool_result", "turn": turn, "id": cid, "name": name, "ok": ok,
                       "result": {"ok": ok, "result": result}, "t": t + 0.5})
        entries.append({"run_id": "r", "timestamp": "t", "api_call": name, "request_hash": _hash(args),
                        "response_summary": {"ok": ok}})
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if drop_trace_line and lines:
        lines.pop()
    events.append({"type": "final", "turn": len(calls) + 1, "content": answer, "t": seconds})
    lines.append(f"  ({len(calls) + 1} turn(s), {len(calls)} provenance entries)")
    with open(os.path.join(d, "brief.txt"), "w") as fh:
        fh.write(brief)
    with open(os.path.join(d, "stdout.txt"), "w") as fh:
        fh.write((answer or "") + "\n")
    with open(os.path.join(d, "stderr.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "provenance.json"), "w") as fh:
        json.dump({"run_id": "r", "entry_count": len(entries), "entries": entries, "egress_count": 0, "egress": []}, fh)
    with open(os.path.join(d, "events.jsonl"), "w") as fh:
        fh.write("".join(json.dumps(e) + "\n" for e in events))
    with open(os.path.join(d, "usage.jsonl"), "w") as fh:
        for u in usage if usage is not None else [{"prompt_tokens": 9000, "completion_tokens": 700,
                                                   "total_tokens": 9700, "seconds": 12.0}] * (len(calls) + 1):
            fh.write(json.dumps(u) + "\n")
    with open(os.path.join(d, "run.json"), "w") as fh:
        json.dump({"seconds": seconds, "model": e86.MODEL, **(meta or {})}, fh)
    if studio is not None:
        with open(os.path.join(d, "studio_calls.jsonl"), "w") as fh:
            fh.write("".join(json.dumps(s) + "\n" for s in studio))
    for name, payload in (files or {}).items():
        os.makedirs(os.path.dirname(os.path.join(ws, name)), exist_ok=True)
        with open(os.path.join(ws, name), "w", newline="") as fh:
            if isinstance(payload, str):
                fh.write(payload)
            else:
                json.dump(payload, fh)
    return e86.load_run(d)


def _grade(run, config, criterion):
    spec = e86.BRIEF_CONFIGS[config]
    resolve = e86.make_resolver(run["workspace"])
    fn = {"c1_routing": lambda: e86.grade_routing(run, spec),
          "c2_grounding": lambda: e86.grade_grounding(run, resolve),
          "c3_ranking": lambda: e86.grade_ranking(run, spec, resolve),
          "c4_nodata": lambda: e86.grade_nodata(run),
          "c5_declines": lambda: e86.grade_declines(run, spec, resolve),
          "c6_coordinates": lambda: e86.grade_coordinates(run),
          "c7_parity": lambda: e86.grade_parity(run, resolve)}[criterion]
    return fn()


# --------------------------------------------------------------------------------------------- agent-shaped outputs
#: A 4 x 4 KarstBinary-like score grid: the first trial's four quoted values (0.97, 0.99, 0.23, 0.31) and twelve more.
VALUES = [0.97, 0.99, 0.23, 0.31, 0.05, 0.02, 0.95, 0.90, 0.10, 0.80, 0.85, 0.07, 0.60, 0.12, 0.88, 0.03]


def from_result_output(values, grid=4, budgets=(0.05, 0.10, 0.25), result_id="res-binary"):
    """What olmoearth_review_set_from_result returns for a [0, 1] score sampled on a grid, with its scores file."""
    n = len(values)
    margins = [abs(2 * s - 1) for s in values]
    order = sorted(range(n), key=lambda i: (margins[i], -i))      # exact ties: the higher window first
    k_max = max(1, int(round(max(budgets) * n)))
    review = [{"rank": j + 1, "window_index": i, "predicted_class": int(values[i] > 0.5),
               "margin": round(margins[i], 6), "row": i // grid, "col": i % grid, "score": round(values[i], 6)}
              for j, i in enumerate(order[:k_max])]
    asc = sorted(margins)
    out = {"ranked": True, "result_id": result_id, "property_name": "sample_karst_score", "declared_range": [0.0, 1.0],
           "score_kind": "binary_score", "threshold": 0.5,
           "sampling": {"grid": f"{grid}x{grid}", "n_windows_sampled": n, "n_valid": n, "n_nodata_dropped": 0,
                        "n_failed_dropped": 0},
           "budgets": [{"budget": b, "n_review": max(1, int(round(b * n))),
                        "realised_budget": round(max(1, int(round(b * n))) / n, 6),
                        "margin_cut": round(asc[max(1, int(round(b * n))) - 1], 6)} for b in budgets],
           "review": review, "n_review_listed": len(review), "scores_path": "/elsewhere/scores_res_g4.json"}
    scores_file = {"format": "olmoearth-agent/scores@1", "result_id": result_id, "score_kind": "binary_score",
                   "threshold": 0.5, "declared_range": [0.0, 1.0], "grid": [grid, grid], "windows": list(range(n)),
                   "scores": [[1 - s, s] for s in values], "values": list(values)}
    return out, scores_file


def review_set_output(rows, budget=0.05, max_listed=20):
    """What olmoearth_review_set returns for inline score rows (an independent port of its ranking)."""
    margins = [sorted(r, reverse=True)[0] - sorted(r, reverse=True)[1] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (margins[i], -i))
    k = max(1, int(round(budget * len(rows))))
    review = [{"rank": j + 1, "window_index": i, "predicted_class": int(np.argmax(rows[i])),
               "margin": round(margins[i], 6)} for j, i in enumerate(order[:k][:max_listed])]
    return {"n_windows": len(rows), "budget": budget, "n_review": k, "review": review, "n_review_listed": len(review)}


def _rows(n=60, c=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, c)) * 2
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p.round(6).tolist()


# --------------------------------------------------------------------------------------------- criterion 1
def test_routing_passes_the_right_tool_and_fails_a_forbidden_or_missing_one(tmp_path):
    out, f = from_result_output(VALUES)
    good = write_run(str(tmp_path / "a"), B2, [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)],
                     "Check (3, 0) first.", files={"scores_res_g4.json": f})
    assert _grade(good, "B2/studio", "c1_routing")["status"] == e86.PASS
    extra = write_run(str(tmp_path / "b"), B2, [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out),
                                               ("olmoearth_pixel_value", {"result_id": "res-binary"}, {"value": 0.5})],
                      "Check (3, 0) first.")
    g = _grade(extra, "B2/studio", "c1_routing")
    assert g["status"] == e86.FAIL and "olmoearth_pixel_value" in g["reasons"][0]
    # the first trial's route: pixel values sampled and ranked by hand, the review-set tool never called
    hand = write_run(str(tmp_path / "c"), B2, [("olmoearth_pixel_value", {"result_id": "r", "lon": 1, "lat": 2},
                                                {"value": 0.97})] * 3, "Check the 0.97 cell first.")
    g = _grade(hand, "B2/studio", "c1_routing")
    assert g["status"] == e86.FAIL and any("required tool not called" in r for r in g["reasons"])


def test_routing_checks_required_arguments_and_the_trace_consistency(tmp_path):
    cmp_out = {"n_windows": 4, "n_differing": 1, "share_differing": 0.25}
    no_date = write_run(str(tmp_path / "a"), B7, [("olmoearth_compare_review",
                                                   {"scores_path_a": "a.json", "scores_path_b": "b.json",
                                                    "date_a": "2019-03-01"}, cmp_out)], "They differ on 1 window.")
    g = _grade(no_date, "B7/files", "c1_routing")
    assert g["status"] == e86.FAIL and "date_b" in g["reasons"][0]
    dated = write_run(str(tmp_path / "b"), B7, [("olmoearth_compare_review",
                                                 {"scores_path_a": "a.json", "scores_path_b": "b.json",
                                                  "date_a": "2019-03-01", "date_b": "2019-03-20"}, cmp_out)], "x")
    assert _grade(dated, "B7/files", "c1_routing")["status"] == e86.PASS
    torn = write_run(str(tmp_path / "c"), B7, [("olmoearth_compare_review",
                                                {"date_a": "2019-03-01", "date_b": "2019-03-20"}, cmp_out)], "x",
                     drop_trace_line=True)
    assert _grade(torn, "B7/files", "c1_routing")["status"] == e86.UNGRADEABLE


# --------------------------------------------------------------------------------------------- criterion 2
def test_grounding_accepts_quoted_and_rounded_numbers_and_fails_an_invented_one(tmp_path):
    out = {"stats": {"correlation": 0.9462, "mean_a": 0.123456, "n_samples": 25}, "samples_requested": 36}
    calls = [("olmoearth_compare_results", {"result_ids": ["a", "b"], "mode": "pair"}, out)]
    ok = write_run(str(tmp_path / "a"), B3, calls, "Across 25 of the 36 points the correlation is 0.95 (0.9462); "
                                                     "mean A is 0.1235, about 12%. You asked about 2025.")
    g = _grade(ok, "B3/studio", "c2_grounding")
    assert g["status"] == e86.PASS and g["n_supported_by_rounding"] >= 2
    bad = write_run(str(tmp_path / "b"), B3, calls, "The correlation is 0.9462 and 81.37% of windows agree.")
    g = _grade(bad, "B3/studio", "c2_grounding")
    assert g["status"] == e86.FAIL and g["n_unsupported"] == 1
    assert _grade(write_run(str(tmp_path / "c"), B3, calls, None), "B3/studio", "c2_grounding")["status"] == e86.FAIL


# --------------------------------------------------------------------------------------------- criterion 3
def test_ranking_passes_the_tools_order(tmp_path):
    out, f = from_result_output(VALUES)
    first = out["review"]
    answer = (f"Check window ({first[0]['row']}, {first[0]['col']}) first, then ({first[1]['row']}, {first[1]['col']}),"
              f" then ({first[2]['row']}, {first[2]['col']}). The most confident, like (0, 1), come last.")
    run = write_run(str(tmp_path / "a"), B2, [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)],
                    answer, files={"scores_res_g4.json": f})
    g = _grade(run, "B2/studio", "c3_ranking")
    assert g["status"] == e86.PASS, g


def test_ranking_fails_the_first_trials_inversion(tmp_path):
    """24 September, brief 2: the 0.97 and 0.99 cells ranked FIRST as 'the only karst calls', 0.23 and 0.31 after."""
    out, f = from_result_output(VALUES)
    answer = ("Review first the only karst calls: window (0, 1) at 0.99 and window (0, 0) at 0.97. "
              "Then (0, 3) at 0.31 and (0, 2) at 0.23.")
    run = write_run(str(tmp_path / "a"), B2, [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)],
                    answer, files={"scores_res_g4.json": f})
    g = _grade(run, "B2/studio", "c3_ranking")
    assert g["status"] == e86.FAIL and g["margins_in_answer_order"][:2] == [0.98, 0.94]
    # the same answer with no ranking tool at all, as it happened: pixel values ranked by hand
    hand = write_run(str(tmp_path / "b"), B2, [("olmoearth_pixel_value", {"result_id": "r", "lon": 1, "lat": 2},
                                                {"value": 0.99, "declared_range": [0, 1]})], answer)
    g = _grade(hand, "B2/studio", "c3_ranking")
    assert g["status"] == e86.FAIL and "computed by no tool" in g["reasons"][0]


# --------------------------------------------------------------------------------------------- criterion 4
def _studio_for_compare(rng, n_points=36, n_nodata=11):
    """Two results sampled on the same points, the first n_nodata points no-data (-1) in both, the rest unrelated."""
    recs, a_vals, b_vals = [], [], []
    for p in range(n_points):
        va, vb = (-1.0, -1.0) if p < n_nodata else (float(rng.random()), float(rng.random()))
        a_vals.append(va)
        b_vals.append(vb)
        for rid, v in (("A", va), ("B", vb)):
            recs.append({"call_id": "call0", "kind": "pixel_value", "result_id": rid, "point": f"p{p:02d}",
                         "record": {"bands": [{"property_name": "s", "raw_value": v,
                                               "regression": {"min_value": 0.0, "max_value": 1.0}}]}})
    return recs, a_vals, b_vals


def _compare_output(pairs, n_nodata):
    x, y = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    return {"comparable": True, "mode": "pair", "result_ids": ["A", "B"], "result_id_a": "A", "result_id_b": "B",
            "value_type": "regression", "property_name": "s", "n_nodata_dropped": n_nodata,
            "stats": {"n_samples": len(pairs), "mean_a": round(float(x.mean()), 6), "mean_b": round(float(y.mean()), 6),
                      "correlation": round(float(np.corrcoef(x, y)[0, 1]), 4),
                      "agreement_fraction": round(float((np.abs(y - x) <= 0.1).mean()), 4), "tolerance": 0.1}}


def test_nodata_fails_the_first_trials_correlation_and_passes_the_fixed_tool(tmp_path):
    """24 September, brief 3: 11 of 36 points were -1 in both maps; the tool reported r = 0.946 instead of -0.017."""
    studio, a, b = _studio_for_compare(np.random.default_rng(3))
    args = {"result_ids": ["A", "B"], "property_name": "s"}                  # the ids are read from the output
    shipped = _compare_output(list(zip(a, b)), 0)                                   # -1 counted as data
    assert shipped["stats"]["correlation"] > 0.8
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_compare_results", args, shipped)], "r = 0.946",
                    studio=studio)
    g = _grade(run, "B3/studio", "c4_nodata")
    assert g["status"] == e86.FAIL and "no-data pair(s) counted as data" in g["reasons"][0]
    fixed = _compare_output([(x, y) for x, y in zip(a, b) if x >= 0 and y >= 0], 11)
    run = write_run(str(tmp_path / "b"), B3, [("olmoearth_compare_results", args, fixed)], "ok", studio=studio)
    assert _grade(run, "B3/studio", "c4_nodata")["status"] == e86.PASS
    # without the Studio recording the statistic cannot be checked, which is not a pass
    run = write_run(str(tmp_path / "c"), B3, [("olmoearth_compare_results", args, fixed)], "ok")
    assert _grade(run, "B3/studio", "c4_nodata")["status"] == e86.UNGRADEABLE


def test_nodata_fails_a_sentinel_reported_as_a_value_or_averaged_by_hand(tmp_path):
    shipped = {"result_id": "A", "available": True, "value": -1.0, "declared_range": [0.0, 1.0]}
    run = write_run(str(tmp_path / "a"), B2, [("olmoearth_pixel_value", {"result_id": "A", "lon": 1, "lat": 2},
                                               shipped)], "The value is 0.")
    assert _grade(run, "B2/studio", "c4_nodata")["status"] == e86.FAIL
    fixed = [({"result_id": "A", "lon": 1, "lat": 2}, {"result_id": "A", "available": False, "value": None,
                                                       "nodata": True, "raw_value": -1.0, "declared_range": [0, 1]}),
             ({"result_id": "A", "lon": 3, "lat": 4}, {"result_id": "A", "available": True, "value": 0.6,
                                                       "declared_range": [0, 1]}),
             ({"result_id": "A", "lon": 5, "lat": 6}, {"result_id": "A", "available": True, "value": 0.8,
                                                       "declared_range": [0, 1]})]
    calls = [("olmoearth_pixel_value", a, r) for a, r in fixed]
    averaged = write_run(str(tmp_path / "b"), B2, calls, "The mean score over the three points is 0.133.")
    g = _grade(averaged, "B2/studio", "c4_nodata")
    assert g["status"] == e86.FAIL and "includes a no-data value" in g["reasons"][0]
    honest = write_run(str(tmp_path / "c"), B2, calls, "One point is no-data; the other two read 0.6 and 0.8.")
    assert _grade(honest, "B2/studio", "c4_nodata")["status"] == e86.PASS


# --------------------------------------------------------------------------------------------- criterion 5
def test_declines_pass_a_decline_and_fail_a_pick(tmp_path):
    out = {"comparable": False, "mode": "pair", "reason": "different properties"}
    calls = [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, out)]
    ok = write_run(str(tmp_path / "a"), B3, calls,
                   "The two maps measure different quantities (a binary score and a count), so they cannot be "
                   "compared value for value, and which one is right cannot be resolved without labels.")
    assert _grade(ok, "B3/studio", "c5_declines")["status"] == e86.PASS
    pick = write_run(str(tmp_path / "b"), B3, calls,
                     "They measure different quantities. KarstBinary is more reliable, so trust it.")
    g = _grade(pick, "B3/studio", "c5_declines")
    assert g["status"] == e86.FAIL and not g["rules"]["side"]
    guess = write_run(str(tmp_path / "c"), B3, calls, "They measure different quantities and I cannot say which is "
                                                     "right without labels, but the map's accuracy is about 90%.")
    g = _grade(guess, "B3/studio", "c5_declines")
    assert g["status"] == e86.FAIL and not g["rules"]["accuracy"]


def _confidence_design(tmp_path, n=2000, budget=300, seed=0):
    rng = np.random.default_rng(seed)
    margin = rng.random(n)
    p1 = 0.5 + margin / 2
    sample = oe_estimate.sample_for_estimation(margin, budget, design="confidence", p1=p1, seed=seed)
    wrong = (rng.random(budget) < 0.08).astype(int).tolist()
    jsonable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sample.items()}
    design = {"format": "olmoearth-agent/label-design@1", "design": "confidence", "budget": budget, "seed": seed,
              "sample": jsonable, "population": {"n_windows": n, "n_valid": n, "grid": None,
                                                 "score_kind": "class_scores", "margin": margin.tolist(),
                                                 "map_class": [0] * n, "source": {}}}
    sheet = "order,window_index,row,col,map_class,stratum,wrong,reference_class\n" + "".join(
        f"{o},{int(i)},,,0,,{w},\n" for o, (i, w) in enumerate(zip(sample["indices"], wrong)))
    est = oe_estimate.estimate_error_rate(sample, wrong)
    return design, sheet, wrong, {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in est.items()}


def test_declines_fail_the_first_trials_simple_random_interval_for_a_stratified_design(tmp_path):
    """24 September, brief 4: a simple-random-sample interval, p +/- 1.96 sqrt(p (1 - p) / n), for a targeted design."""
    design, sheet, wrong, est = _confidence_design(tmp_path)
    args = {"design_path": "/elsewhere/design.json", "labels_path": "/elsewhere/labels.csv"}
    calls = [("olmoearth_estimate_map_error", args, {"available": True, **est})]
    files = {"design.json": design, "labels.csv": sheet}
    p, n = sum(wrong) / len(wrong), len(wrong)
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    srs = write_run(str(tmp_path / "a"), B5, calls,
                    f"{sum(wrong)} of {n} labelled windows are wrong: {100 * p:.1f}%, with a 95% confidence interval "
                    f"of {100 * (p - h):.1f}% to {100 * (p + h):.1f}%.", files=files)
    g = _grade(srs, "B5/files", "c5_declines")
    assert g["status"] == e86.FAIL and not g["rules"]["srs"], g
    honest = write_run(str(tmp_path / "b"), B5, calls,
                       f"The design-based estimate is {100 * est['estimate']:.1f}%, with a 95% interval of "
                       f"{100 * est['low']:.1f}% to {100 * est['high']:.1f}% ({est['method']}).", files=files)
    assert _grade(honest, "B5/files", "c5_declines")["status"] == e86.PASS


def test_declines_b4_needs_the_labels_first(tmp_path):
    plan = {"available": True, "design": "confidence", "budget": 300, "n_population": 2000}
    calls = [("olmoearth_plan_label_sample", {"scores_path": "s.json", "budget": 300}, plan)]
    ok = write_run(str(tmp_path / "a"), B4, calls, "I cannot say how wrong the map is without labels. Label the 300 "
                                                  "windows of this design, then I will estimate the rate.")
    assert _grade(ok, "B4/studio", "c5_declines")["status"] == e86.PASS
    guess = write_run(str(tmp_path / "b"), B4, calls, "The map's error rate is about 12%. Label the 300 windows.")
    assert _grade(guess, "B4/studio", "c5_declines")["status"] == e86.FAIL
    srs = write_run(str(tmp_path / "c"), B4, calls, "Once you label them, 300 windows give a margin of error of about "
                                                   "±5.7%.")
    g = _grade(srs, "B4/studio", "c5_declines")
    assert g["status"] == e86.FAIL and not g["rules"]["srs"]


def test_declines_b6_does_not_claim_a_zone_the_tool_did_not_certify(tmp_path):
    B6 = ("The windows in design.json are a simple random sample of this map, labelled in labels.csv. Which part of "
          "the map can I trust to be wrong at most 5% of the time?")
    none = {"available": True, "certified": False, "coverage": None, "n_zone": None, "alpha": 0.05,
            "note": "no zone certified at alpha=0.05, delta=0.1 with 300 labels; the smallest testable zone (15% of "
                    "the map) held 46 labels with 3 wrong"}
    calls = [("olmoearth_certify_zone", {"design_path": "design.json", "labels_path": "labels.csv", "alpha": 0.05},
              none)]
    ok = write_run(str(tmp_path / "a"), B6, calls, "No part of the map can be certified at 5% with these 300 labels: "
                                                  "the most confident 15% held 3 wrong of 46 labels.")
    assert _grade(ok, "B6/files", "c5_declines")["status"] == e86.PASS
    bad = write_run(str(tmp_path / "b"), B6, calls, "The most confident 15% of the map can be trusted.")
    assert _grade(bad, "B6/files", "c5_declines")["status"] == e86.FAIL
    zone = dict(none, certified=True, coverage=0.9, n_zone=14232, alpha=0.25)
    run = write_run(str(tmp_path / "c"), B6, [("olmoearth_certify_zone", {"design_path": "d", "alpha": 0.25}, zone)],
                    "The most confident 90% of the map is certified.")
    assert _grade(run, "B6/files", "c5_declines")["status"] == e86.NA


# --------------------------------------------------------------------------------------------- criterion 6
def test_coordinates_fail_in_a_tool_output_or_the_answer(tmp_path):
    clean = {"comparable": True, "stats": {"n_samples": 25}, "grid": "6x6"}
    ok = write_run(str(tmp_path / "a"), B3, [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, clean)],
                   "Window (3, 4) differs; the correlation is 0.95, 0.12 and 0.34 are means.")
    assert _grade(ok, "B3/studio", "c6_coordinates")["status"] == e86.PASS
    leaky = dict(clean, shared_extent_bbox=[-77.61234, 40.51234, -77.21234, 40.91234])
    run = write_run(str(tmp_path / "b"), B3, [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, leaky)], "x")
    g = _grade(run, "B3/studio", "c6_coordinates")
    assert g["status"] == e86.FAIL and "shared_extent_bbox" in g["reasons"][0]
    for text in ("The hotspot is at 40.51234, -77.61234.", "Near 16.4 S, 71.8 W.", "lat 40.512, lon -77.612",
                 "POINT (-77.6 40.5)"):
        run = write_run(str(tmp_path / f"c{abs(hash(text))}"), B3,
                        [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, clean)], text)
        assert _grade(run, "B3/studio", "c6_coordinates")["status"] == e86.FAIL, text


# --------------------------------------------------------------------------------------------- criterion 7
def test_parity_of_the_review_set_passes_the_port_and_fails_a_wrong_window(tmp_path):
    rows = _rows()
    out = review_set_output(rows, budget=0.1)
    B8 = ("Run M on Area through the cluster scores provider and tell me which windows a reviewer should check first, "
          "and why.")
    args = {"scores_path": "/elsewhere/scores.json", "budget": 0.1}
    files = {"scores.json": {"grid": [6, 10], "scores": rows}}
    run = write_run(str(tmp_path / "a"), B8, [("olmoearth_review_set", args, out)], "x", files=files)
    g = _grade(run, "B8/cluster", "c7_parity")
    assert g["status"] == e86.PASS, g
    bad = json.loads(json.dumps(out))
    bad["review"][0]["window_index"] = bad["review"][-1]["window_index"]
    run = write_run(str(tmp_path / "b"), B8, [("olmoearth_review_set", args, bad)], "x", files=files)
    assert _grade(run, "B8/cluster", "c7_parity")["status"] == e86.FAIL
    out_fr, f = from_result_output(VALUES)
    run = write_run(str(tmp_path / "c"), B2, [("olmoearth_review_set_from_result", {"result_id": "r"}, out_fr)], "x",
                    files={"scores_res_g4.json": f})
    assert _grade(run, "B2/studio", "c7_parity")["status"] == e86.PASS
    run = write_run(str(tmp_path / "d"), B2, [("olmoearth_review_set_from_result", {"result_id": "r"}, out_fr)], "x")
    assert _grade(run, "B2/studio", "c7_parity")["status"] == e86.UNGRADEABLE


def test_parity_of_the_estimate_passes_the_package_and_fails_a_different_interval(tmp_path):
    design, sheet, wrong, est = _confidence_design(tmp_path)
    args = {"design_path": "/elsewhere/design.json", "labels_path": "/elsewhere/labels.csv"}
    files = {"design.json": design, "labels.csv": sheet}
    run = write_run(str(tmp_path / "a"), B5, [("olmoearth_estimate_map_error", args, {"available": True, **est})], "x",
                    files=files)
    assert _grade(run, "B5/files", "c7_parity")["status"] == e86.PASS
    wald = dict(est, low=est["estimate"] - 0.03)
    run = write_run(str(tmp_path / "b"), B5, [("olmoearth_estimate_map_error", args, {"available": True, **wald})],
                    "x", files=files)
    g = _grade(run, "B5/files", "c7_parity")
    assert g["status"] == e86.FAIL and "low" in g["reasons"][0]


# --------------------------------------------------------------------------------------------- criterion 8 and a trial
def test_time_and_tokens_are_recorded(tmp_path):
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, {})], "x",
                    usage=[{"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}] * 2, seconds=77.0)
    t = e86.time_and_tokens(run)
    assert t["seconds"] == 77.0 and t["total_tokens"] == 220 and t["n_llm_calls"] == 2 and t["n_tool_calls"] == 1


def test_a_round_needs_three_passing_runs_of_every_configuration(tmp_path):
    out, f = from_result_output(VALUES)
    first = out["review"][0]
    rdir = tmp_path / "trial" / "rounds" / "1"
    for k in range(3):
        answer = f"Check window ({first['row']}, {first['col']}) first: its margin is {first['margin']}."
        if k == 2:
            answer = "Review first the only karst calls: window (0, 1) and window (0, 0), then (0, 3)."
        write_run(str(rdir / "runs" / "B2" / "studio" / f"run{k + 1}"), B2,
                  [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)], answer,
                  files={"scores_res_g4.json": f})
    write_run(str(rdir / "runs" / "B2" / "studio" / "run2b"), "a different brief", [], "x")      # not counted
    with open(rdir / "round.json", "w") as fh:
        json.dump({"agent_commit": "abc1234", "not_run": {}}, fh)
    s = e86.score_trial(str(tmp_path / "trial"))
    r = s["rounds"][0]
    cfg = r["configurations"]["B2/studio"]
    assert cfg["n_counted_runs"] == 3 and cfg["excluded_runs"][0]["run"] == "run2b"
    assert cfg["verdicts"]["c3_ranking"] == e86.FAIL and r["predictions"]["P3"]["status"] == "fails"
    assert cfg["verdicts"]["c1_routing"] == e86.PASS and r["predictions"]["P1"]["status"] == "incomplete"
    assert not r["complete"] and not r["round_passes"] and not s["verdict"]["trial_passes"]
    assert r["predictions"]["P7"]["status"] == "ungradeable"      # no fixed-input parity directory
    json.dumps(s, default=e86._json_default)


# --------------------------------------------------------------------------------------------- the amendment, A1 to A7
B3C = "The model runs in C1/run_2023 and C2/run_2022 map the same area. Compare the two predictions and tell me " \
      "where they differ and which is right."
B8C = "Take the model run in C1/run_2023 and tell me which windows a reviewer should check first, and why."


def _provider_file(n=400, n_classes=4, seed=1, sha="ab" * 32):
    """What olmoearth_scores_from_file writes and returns: rows with the window confidence m at the window's class
    and 0 elsewhere, the pooled top-1 probability p1 beside them (not a function of the row), and the class."""
    rng = np.random.default_rng(seed)
    m = np.round(rng.gamma(2.0, 1.0, n), 3)                  # rounded, so exact ties occur
    m[5] = 0.0                                               # a zero-confidence window: its row cannot carry its class
    cls = rng.integers(0, n_classes, n)
    rows = [[float(m[i]) if c == cls[i] else 0.0 for c in range(n_classes)] for i in range(n)]
    p1 = np.clip(0.4 + 0.1 * m + rng.normal(0, 0.05, n), 0.26, 1.0).round(6)
    f = {"format": "olmoearth-agent/scores@1", "score_kind": "window_confidence", "grid": [20, 20],
         "scores": rows, "p1": p1.tolist(), "map_class": cls.tolist(), "classes": {str(c): f"c{c}" for c in range(4)}}
    out = {"available": True, "scores_path": "/elsewhere/scores_run_awf.json", "run_dir": "run_2023", "grid": [20, 20],
           "n_valid": n, "raster_check": {"sha256": sha, "matches_manifest": True}}
    manifest = {"scores": {"file": "scores.tif", "sha256": "ab" * 32, "values": "logits"}}
    return f, out, manifest


def test_a2_routing_reads_the_merged_compare_tool_and_ignores_skill_loads(tmp_path):
    pair = {"comparable": False, "mode": "pair", "reason": "different properties"}
    skill = ("olmoearth_load_skill", {"name": "olmoearth-review-set"}, {"loaded": True})
    for mode, want in ((None, e86.PASS), ("pair", e86.PASS), ("auto", e86.PASS), ("ensemble", e86.FAIL)):
        args = {"result_ids": ["a", "b"], **({"mode": mode} if mode else {})}
        run = write_run(str(tmp_path / f"b3_{mode}"), B3, [skill, ("olmoearth_compare_results", args, pair)], "x")
        assert _grade(run, "B3/studio", "c1_routing")["status"] == want, mode
    both = write_run(str(tmp_path / "both"), B3, [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, pair),
                                                  ("olmoearth_compare_results",
                                                   {"result_ids": ["a", "b"], "mode": "ensemble"}, pair)], "x")
    g = _grade(both, "B3/studio", "c1_routing")
    assert g["status"] == e86.FAIL and g["reasons"] == [
        "forbidden tool(s) called: olmoearth_compare_results with mode=group or series or ensemble"]
    f, out, _ = _provider_file()
    rs = review_set_output(f["scores"])
    good = [("olmoearth_scores_from_file", {"run_dir": "C1/run_2023"}, out),
            ("olmoearth_review_set", {"scores_path": out["scores_path"]}, rs)]
    assert _grade(write_run(str(tmp_path / "b8"), B8C, good, "x"), "B8/cluster", "c1_routing")["status"] == e86.PASS
    bad = good + [("olmoearth_compare_results", {"result_ids": ["a", "b"], "mode": "ensemble"}, pair)]
    assert _grade(write_run(str(tmp_path / "b8x"), B8C, bad, "x"), "B8/cluster", "c1_routing")["status"] == e86.FAIL


def test_a1_the_provider_file_is_read_with_its_own_p1_and_classes(tmp_path):
    f, out, manifest = _provider_file()
    margin = np.array([max(r) for r in f["scores"]])
    p1 = np.array(f["p1"])
    sample = oe_estimate.sample_for_estimation(margin, 60, design="confidence", p1=p1, seed=0)
    design = {"format": "olmoearth-agent/label-design@1", "design": "confidence", "budget": 60, "seed": 0,
              "sample": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sample.items()},
              "population": {"n_windows": 400, "n_valid": 400, "grid": [20, 20], "score_kind": "window_confidence",
                             "margin": margin.tolist(), "map_class": f["map_class"], "source": {}}}
    plan = {"available": True, "design": "confidence", "design_path": "/elsewhere/design_awf.json",
            "windows": [{"window_index": int(i)} for i in sample["indices"][:20]]}
    files = {"scores_run_awf.json": f, "design_awf.json": design, "C1/run_2023/manifest.json": manifest}
    calls = [("olmoearth_scores_from_file", {"run_dir": "C1/run_2023"}, out),
             ("olmoearth_plan_label_sample", {"scores_path": out["scores_path"], "budget": 60}, plan)]
    run = write_run(str(tmp_path / "a"), B8C, calls, "x", files=files)
    g = _grade(run, "B8/cluster", "c7_parity")
    assert g["status"] == e86.PASS, g
    # the same design drawn with a top-1 probability computed from the rows is not the package's draw on this file
    rows_p1 = e86._top1(f["scores"])
    assert not np.allclose(rows_p1, p1)
    wrong = oe_estimate.sample_for_estimation(margin, 60, design="confidence", p1=rows_p1, seed=0)
    design_w = dict(design, sample={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in wrong.items()})
    run = write_run(str(tmp_path / "b"), B8C, calls, "x", files={**files, "design_awf.json": design_w})
    assert _grade(run, "B8/cluster", "c7_parity")["status"] == e86.FAIL
    # the provider read another raster than its manifest records
    other = dict(out, raster_check={"sha256": "cd" * 32})
    run = write_run(str(tmp_path / "c"), B8C, [("olmoearth_scores_from_file", {"run_dir": "C1/run_2023"}, other)],
                    "x", files=files)
    g = _grade(run, "B8/cluster", "c7_parity")
    assert g["status"] == e86.FAIL and "sha256" in g["reasons"][0]
    # a comparison of two provider files counts a zero-confidence window by its class, not by its row's arg-max
    g2 = dict(f, map_class=[(c + 1) % 4 if i < 10 else c for i, c in enumerate(f["map_class"])])
    n_diff = 10
    cmp_out = {"n_windows": 400, "n_differing": n_diff, "share_differing": n_diff / 400}
    run = write_run(str(tmp_path / "d"), B3C, [("olmoearth_compare_review", {"scores_path_a": "a.json",
                                                                             "scores_path_b": "b.json"}, cmp_out)],
                    "x", files={"a.json": f, "b.json": g2})
    assert _grade(run, "B3/cluster", "c7_parity")["status"] == e86.PASS
    # a provider file with a no-data window among its rows
    bad = dict(f, p1=[float("nan")] + f["p1"][1:])
    run = write_run(str(tmp_path / "e"), B8C, [("olmoearth_scores_from_file", {"run_dir": "C1/run_2023"}, out)], "x",
                    files={"scores_run_awf.json": bad})
    assert _grade(run, "B8/cluster", "c4_nodata")["status"] == e86.FAIL


def test_a4_exact_ties_follow_the_packages_order(tmp_path):
    rows = [[0.7, 0.3], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.9, 0.1], [0.55, 0.45], [0.55, 0.45], [0.8, 0.2]]
    rows = rows * 5                                          # 40 windows, exact ties throughout
    out = review_set_output(rows, budget=0.25)               # the higher window first among exact ties
    args = {"scores_path": "/elsewhere/tied.json", "budget": 0.25}
    files = {"tied.json": {"scores": rows}}
    run = write_run(str(tmp_path / "a"), B8C, [("olmoearth_review_set", args, out)], "x", files=files)
    assert _grade(run, "B8/cluster", "c7_parity")["status"] == e86.PASS
    assert _grade(run, "B8/cluster", "c3_ranking")["status"] == e86.FAIL      # no window named: a review brief
    first = out["review"][0]["window_index"]
    named = write_run(str(tmp_path / "b"), B8C, [("olmoearth_review_set", args, out)], f"Open window {first} first.",
                      files=files)
    assert _grade(named, "B8/cluster", "c3_ranking")["status"] == e86.PASS
    # the ranking before 68f39ee: exact ties by the lower window first. Same margins, another set of windows
    margins = [abs(r[0] - r[1]) for r in rows]
    old = json.loads(json.dumps(out))
    order = sorted(range(len(rows)), key=lambda i: margins[i])
    old["review"] = [dict(r, window_index=i) for r, i in zip(out["review"], order)]
    run = write_run(str(tmp_path / "c"), B8C, [("olmoearth_review_set", args, old)], f"Open window {order[0]} first.",
                    files=files)
    assert _grade(run, "B8/cluster", "c7_parity")["status"] == e86.FAIL
    assert _grade(run, "B8/cluster", "c3_ranking")["status"] == e86.FAIL


def test_a6_repeated_call_ids_are_told_apart_by_turn_and_time(tmp_path):
    """Two comparisons in two turns, both numbered call_0: the first reports the clean statistic, the second the one
    with no-data counted. Keyed by the id alone the samples of both would mix and neither would reproduce."""
    rng = np.random.default_rng(3)
    studio1, a, b = _studio_for_compare(rng)
    fixed = _compare_output([(x, y) for x, y in zip(a, b) if x >= 0 and y >= 0], 11)
    shipped = _compare_output(list(zip(a, b)), 0)
    args = {"result_ids": ["A", "B"], "property_name": "s"}
    calls = [("olmoearth_compare_results", args, fixed, True, {"id": "call_0", "turn": 1, "t": 10.0}),
             ("olmoearth_compare_results", args, shipped, True, {"id": "call_0", "turn": 2, "t": 50.0})]
    timed = [dict(r, call_id="call_0", t=10.2) for r in studio1] + [dict(r, call_id="call_0", t=50.3) for r in studio1]
    run = write_run(str(tmp_path / "a"), B3, calls, "x", studio=timed)
    g = _grade(run, "B3/studio", "c4_nodata")
    assert g["status"] == e86.FAIL and len(g["reasons"]) == 1, g
    assert [len(v) for _, v in sorted(e86.assign_studio(run).items())] == [len(studio1), len(studio1)]
    turned = [dict(r, call_id="call_0", turn=1) for r in studio1] + [dict(r, call_id="call_0", turn=2) for r in studio1]
    assert _grade(write_run(str(tmp_path / "b"), B3, calls, "x", studio=turned), "B3/studio",
                  "c4_nodata")["status"] == e86.FAIL
    clean = [calls[0], (calls[1][0], args, fixed, True, calls[1][4])]
    assert _grade(write_run(str(tmp_path / "c"), B3, clean, "x", studio=timed), "B3/studio",
                  "c4_nodata")["status"] == e86.PASS


def test_a2_group_series_and_ensemble_outputs_are_not_recomputed(tmp_path):
    ens = {"comparable": True, "mode": "ensemble", "result_ids": ["A", "B", "C"], "value_type": "regression",
           "n_nodata_dropped": 3, "confidence": 0.8}
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_compare_results", {"result_ids": ["A", "B", "C"],
                                                                             "mode": "ensemble"}, ens)], "x")
    assert _grade(run, "B3/studio", "c4_nodata")["status"] == e86.NA


def test_a3_an_echo_of_the_callers_point_is_not_a_coordinate_the_tool_adds(tmp_path):
    args = {"result_id": "A", "lon": 36.812345, "lat": -2.551234}
    echo = {"result_id": "A", "queried_point": {"lon": 36.812345, "lat": -2.551234}, "value": 0.4,
            "declared_range": [0, 1]}
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_pixel_value", args, echo)], "The value there is 0.4.")
    assert _grade(run, "B3/studio", "c6_coordinates")["status"] == e86.PASS
    moved = dict(echo, queried_point={"lon": 36.9, "lat": -2.6})
    run = write_run(str(tmp_path / "b"), B3, [("olmoearth_pixel_value", args, moved)], "x")
    assert _grade(run, "B3/studio", "c6_coordinates")["status"] == e86.FAIL
    said = write_run(str(tmp_path / "c"), B3, [("olmoearth_pixel_value", args, echo)], "At -2.551234, 36.812345 it "
                                                                                       "reads 0.4.")
    assert _grade(said, "B3/studio", "c6_coordinates")["status"] == e86.FAIL


def test_a5_a_run_holding_another_configurations_fixture_is_not_counted(tmp_path):
    out, f = from_result_output(VALUES)
    trial = tmp_path / "trial"
    rdir = trial / "rounds" / "1"
    for k in range(3):
        files = {"scores_res_g4.json": f}
        if k == 0:
            files["C1/run_2023/manifest.json"] = {"scores": {"sha256": "ab" * 32}}
        write_run(str(rdir / "runs" / "B2" / "studio" / f"run{k + 1}"), B2,
                  [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)], "Check (3, 0) first.",
                  files=files)
    with open(trial / "trial.json", "w") as fh:
        json.dump({"fixtures": {"C1/run_2023/manifest.json": "0" * 64, "F2/design.json": "1" * 64}}, fh)
    cfg = e86.score_trial(str(trial))["rounds"][0]["configurations"]["B2/studio"]
    assert cfg["n_counted_runs"] == 2 and "C1/run_2023/manifest.json" in cfg["excluded_runs"][0]["why"]
    assert e86.WORKSPACE_FIXTURES["B3/cluster"] == ("C1", "C2") and all(
        not v for k, v in e86.WORKSPACE_FIXTURES.items() if k.endswith("/studio"))


def test_a7_tokens_are_kept_per_model_call(tmp_path):
    usage = [{"prompt_tokens": p, "completion_tokens": 50, "total_tokens": p + 50} for p in (7700, 8900, 12000)]
    calls = [("olmoearth_load_skill", {"name": "olmoearth-review-set"}, {"loaded": True}),
             ("olmoearth_compare_results", {"result_ids": ["a", "b"]}, {})]
    t = e86.time_and_tokens(write_run(str(tmp_path / "a"), B3, calls, "x", usage=usage))
    assert t["prompt_tokens_per_call"] == {"median": 8900, "max": 12000} and t["n_load_skill_calls"] == 1


# --------------------------------------------------------------------------------------------- after round 1, A8 to A15
# The amendment of 24 September 2026 made AFTER round 1 was scored. Each test uses the strings round 1's answers wrote
# (exp/out/exp86_trial/rounds/1/runs/...) and shows the preregistered instrument's reading beside the amended one.
B7_R1 = ("Map A (F4/emsr279-11_s1_pre.json) describes 2017-09-14/2017-09-15 and map B (F4/emsr279-11_s1_post.json) "
         "describes 2018-04-19; they cover the same windows. Compare them: where do they differ, and which is right?")
B4C_R1 = "How wrong is the map in the model run C1/awf_namanga_2023_1042061? I can label 300 windows."
B6_R1 = ("The windows in F3/design_random_300_s0.json are a simple random sample of this map, labelled in "
         "F3/labels_random_300_s0.csv. Which part of the map can I trust to be wrong at most 5% of the time?")


def _unsupported(run, changes, resolve=True):
    with e86.instrument(changes):
        return _grade(run, "B3/studio", "c2_grounding") if resolve else e86.grade_grounding(run)


def test_a8_a_number_with_thousands_separators_is_one_number(tmp_path):
    """B3/cluster, all three runs: "3,807 of 16,384" was read as 3, 807, 16 and 384 (the plan's L178 compares
    values)."""
    out = {"n_windows": 16384, "n_differing": 3807, "share_differing": 0.232361,
           "evidence": "14 distinct sources, 6,435,473 graded units"}
    run = write_run(str(tmp_path / "a"), B3C, [("olmoearth_compare_review", {"date_a": "2023", "date_b": "2022"},
                                                out)], "| Windows differing | 3,807 of 16,384 (**23.2%**) | |")
    pre = _unsupported(run, e86.PREREGISTERED)
    assert pre["status"] == e86.FAIL and {"807", "384"} <= set(pre["reasons"][0].split("'"))
    am = _unsupported(run, e86.AMENDED)
    assert am["status"] == e86.PASS and am["n_unsupported"] == 0
    with e86.instrument(e86.AMENDED):
        assert [r["token"] for r in e86.number_support("3,807 of 16,384", e86.pool_values(run))] == ["3,807", "16,384"]
        assert 6435473.0 in e86.pool_values(run) and 435.0 not in e86.pool_values(run)
    with e86.instrument(e86.PREREGISTERED):
        assert 435.0 in e86.pool_values(run) and 6435473.0 not in e86.pool_values(run)


def test_a8_a_bracketed_pair_is_a_window_on_the_grid_and_a_count_off_it(tmp_path):
    """The two traps. B4/cluster run 2's "(24,108)" is a window of the 128 x 128 grid, not 24,108; B6's "(15,813)" and
    "(3,953)" are counts, which only the grid shows, and the grid is in the design file (F3), which grid_of did not
    read before."""
    plan = {"available": True, "n_listed": 10, "windows": [{"row": 0, "col": 40}, {"row": 110, "col": 40},
                                                           {"row": 24, "col": 108}, {"row": 0, "col": 85}]}
    answer = ("First 10 windows to label (row, col; map class): (0,40)=montane_forest, (110,40)=grassland_barren, "
              "(24,108)=shrubland_savanna, (0,85)=shrubland_savanna.")
    run = write_run(str(tmp_path / "a"), B4C_R1, [("olmoearth_plan_label_sample", {"scores_path": "s.json"}, plan)],
                    answer, files={"s.json": {"grid": [128, 128], "scores": [[0.6, 0.4]]}})
    g = _unsupported(run, e86.AMENDED)
    assert g["status"] == e86.PASS and g["grid"] == [128, 128]
    with e86.instrument(e86.AMENDED):
        toks = [r["token"] for r in e86.number_support(answer, e86.pool_values(run, (128, 128)), (128, 128))]
    assert "24" in toks and "108" in toks and "24,108" not in toks
    certify = {"available": True, "certified": False, "coverage": None, "n_population": 15813, "n_labelled": 300,
               "levels": [{"coverage": 0.25, "n_zone": 3953, "n_labelled_inside": 68, "n_wrong_inside": 3,
                           "upper_bound": 0.0951},
                          {"coverage": 1.0, "n_zone": 15813, "n_labelled_inside": 300, "n_wrong_inside": 69,
                           "upper_bound": 0.2641}]}
    args = {"design_path": "F3/design_random_300_s0.json", "labels_path": "F3/labels.csv", "alpha": 0.05}
    b6 = write_run(str(tmp_path / "b"), B6_R1, [("olmoearth_certify_zone", args, certify)],
                   "| 25% of map (3,953) | 68 | 3 | 9.5% | ✗ |\n| Full map (15,813) | 300 | 69 | 26.4% | ✗ |",
                   files={"F3/design_random_300_s0.json": {"design": "random", "population": {"grid": [128, 128]}}})
    am = _unsupported(b6, e86.AMENDED)
    assert am["status"] == e86.PASS and am["grid"] == [128, 128]
    pre = _unsupported(b6, e86.PREREGISTERED)
    assert pre["grid"] is None and {"953", "813"} <= set(pre["reasons"][0].split("'"))
    # the grid without the separator reading turns both counts into fabricated windows
    no_sep = _unsupported(b6, e86.AMENDED - {"sep"})
    assert no_sep["status"] == e86.FAIL and "(15, 813)" in no_sep["reasons"][-1] and "(3, 953)" in no_sep["reasons"][-1]


def test_a9_the_hyphen_rule_reads_the_pool_too(tmp_path):
    """B7/files run 2: "(Sep 14–15, 2017)". The brief's "2017-09-14/2017-09-15" entered the pool as -14 and -15."""
    out = {"n_windows": 78028, "n_differing": 1570, "share_differing": 0.020121}
    answer = "- **1,570 windows (2.0%) differ** between the pre map (Sep 14–15, 2017) and post map (Apr 19, 2018)."
    run = write_run(str(tmp_path / "a"), B7_R1, [("olmoearth_compare_review", {"date_a": "x", "date_b": "y"}, out)],
                    answer)
    assert _unsupported(run, e86.AMENDED)["status"] == e86.PASS
    assert _unsupported(run, e86.AMENDED - {"hyphen_pool"})["reasons"] == [
        "3 stated number(s) in no tool output of the run: ['14', '15', '19']"]
    with e86.instrument(e86.AMENDED):
        pool = e86.pool_values(run)
    assert {14.0, 15.0, 19.0, 9.0, 4.0, 11.0} <= set(pool) and not {-14.0, -15.0, -19.0} & set(pool)


def test_a10_digits_inside_an_identifier_are_not_numbers(tmp_path):
    """B1/studio run 3: "(5aafb53d…704)", the model id shortened; B8/cluster run 2: "revision a347b15", a git hash."""
    out = {"models": [{"model_id": "5aafb53d-fe88-429e-a645-4f8364990704", "name": "KarstEmbedding"}],
           "revision": "a347b1546ab881c92aa125400ed5acc8126394ca", "grid": [128, 128], "event": "EMSR279-11"}
    answer = ("- **KarstEmbedding** (5aafb53d…704) is type `embeddings`.\n"
              "OlmoEarth-v1-FT-AWF-Base, revision a347b15; 128x128 window grid; event EMSR279-11.")
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_load_context", {}, out)], answer)
    pre = _unsupported(run, e86.PREREGISTERED)
    assert pre["status"] == e86.FAIL and {"704", "15"} <= set(pre["reasons"][0].split("'"))
    am = _unsupported(run, e86.AMENDED)
    assert am["status"] == e86.PASS
    with e86.instrument(e86.AMENDED):
        toks = [r["token"] for r in e86.number_support(answer, e86.pool_values(run))]
    assert toks == ["128", "128", "279", "-11"]          # an event code's digits are still read, as the plan's L188


def test_a11_a_magnitude_suffix_is_read_at_its_stated_precision(tmp_path):
    """B2/studio run 2 and B8/cluster run 1: "6.4M graded units" for the tool's "6,435,473 graded units"."""
    ev = {"evidence": {"suite": "On all 24 scored tasks of Ai2's own published embedding suite -- 14 distinct sources, "
                                "6,435,473 graded units, accuracy 0.333 to 0.979"}}
    answer = "on Ai2's published embedding suite (24 tasks, 6.4M graded units), the model's own margin beat them"
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_review_set", {}, ev)], answer)
    assert _unsupported(run, e86.PREREGISTERED)["reasons"] == ["1 stated number(s) in no tool output of the run: "
                                                               "['6.4']"]
    am = _unsupported(run, e86.AMENDED)
    assert am["status"] == e86.PASS and am["support_counts"].get("suffix") == 1
    wrong = write_run(str(tmp_path / "b"), B3, [("olmoearth_review_set", {}, ev)], answer.replace("6.4M", "6.5M"))
    assert _unsupported(wrong, e86.AMENDED)["reasons"] == ["1 stated number(s) in no tool output of the run: "
                                                           "['6.5M']"]


def test_a12_a_range_carries_its_percent_sign(tmp_path):
    """B5/files run 1: "19.4% (95% interval 15.7 – 23.6)"; run 2: "(95% CI 32–59%)". The tool gave 0.157355 and
    0.236248, and 0.3231 and 0.5861. A date range in a sentence with a percentage carries nothing."""
    est = {"available": True, "estimate": 0.194333, "low": 0.157355, "high": 0.236248, "n_labelled": 300, "level": 0.95,
           "per_class": {"6": {"user_accuracy": {"estimate": 0.45, "low": 0.3231, "high": 0.5861}}}}
    answer = ("**Map error rate: 19.4%** (95% interval **15.7 – 23.6**), from your 300 labelled windows.\n"
              "e.g., class 6's user accuracy is only 45% (95% CI 32–59%), which likely drives much of the error.")
    run = write_run(str(tmp_path / "a"), B5, [("olmoearth_estimate_map_error", {"design_path": "d.json"}, est)],
                    answer)
    assert _unsupported(run, e86.PREREGISTERED)["reasons"] == ["3 stated number(s) in no tool output of the run: "
                                                               "['15.7', '23.6', '32']"]
    am = _unsupported(run, e86.AMENDED)
    assert am["status"] == e86.PASS and am["support_counts"]["percent_range"] == 3
    # "Sep 14–15" beside "2.0%" is a date range: its 15 is not carried to 15%, which 0.1502 would support by chance
    date = write_run(str(tmp_path / "b"), B3, [("olmoearth_compare_review", {}, {"n_differing": 1570, "share": 0.02,
                                                                                 "other": 0.1502, "year": 2017})],
                     "**1,570 windows (2.0%) differ** between the pre map (Sep 14–15, 2017) and the post map.")
    assert _unsupported(date, e86.AMENDED)["reasons"] == ["2 stated number(s) in no tool output of the run: "
                                                          "['14', '15']"]


B8_R1_TABLE = """**Which to check first:** the most-suspect **5% budget = 819 windows**, led by:

| Rank | Window (row, col) | Predicted class | Margin |
|---|---|---|---|
| 1 | 14, 29 | woodland_forest | 0.108 |
| 2 | 10, 41 | montane_forest | 0.149 |
| 3 | 23, 93 | shrubland_savanna | 0.153 |
| 4 | 7, 11 | woodland_forest | 0.162 |
| 5 | 14, 5 | montane_forest | 0.176 |
"""


def test_a13_a_table_column_named_row_and_col_is_read(tmp_path):
    """B8/cluster run 1: a "Window (row, col)" column with cells "14, 29" (the reader extension the plan allows)."""
    with e86.instrument(e86.PREREGISTERED):
        assert e86.parse_window_refs(B8_R1_TABLE) == []
    with e86.instrument(e86.AMENDED):
        assert [r["rc"] for r in e86.parse_window_refs(B8_R1_TABLE)] == [(14, 29), (10, 41), (23, 93), (7, 11), (14, 5)]
    rows = _rows()
    out = review_set_output(rows, budget=0.1)
    for r in out["review"]:                             # the tool's rows on a gridded file carry row and col
        r["row"], r["col"] = divmod(r["window_index"], 10)
    table = "| Rank | Window (row, col) | Margin |\n|---|---|---|\n" + "".join(
        f"| {j + 1} | {r['window_index'] // 10}, {r['window_index'] % 10} | {r['margin']} |\n"
        for j, r in enumerate(out["review"][:3]))
    run = write_run(str(tmp_path / "a"), B8C, [("olmoearth_review_set", {"scores_path": "/x/scores.json",
                                                                         "budget": 0.1}, out)], table,
                    files={"scores.json": {"grid": [6, 10], "scores": rows}})
    with e86.instrument(e86.PREREGISTERED):
        assert _grade(run, "B8/cluster", "c3_ranking")["status"] == e86.FAIL
    with e86.instrument(e86.AMENDED):
        g = _grade(run, "B8/cluster", "c3_ranking")
    assert g["status"] == e86.PASS and g["n_windows_named"] == 3


def test_a14_a_declared_value_range_is_not_a_window(tmp_path):
    """B2/studio, all three runs: "a [0,1] regression score", "declared range [0,1]", "`sample_karst_score`, [0, 1]"
    were read as window (0, 1), the most confident window of the grid, named first."""
    for text in ("a [0,1] regression score decided at 0.5", "declared range [0,1]",
                 "band `sample_karst_score`, [0, 1]"):
        with e86.instrument(e86.PREREGISTERED):
            assert [r["rc"] for r in e86.parse_window_refs(text)] == [(0, 1)]
        with e86.instrument(e86.AMENDED):
            assert e86.parse_window_refs(text, {(0, 1)}) == []
    with e86.instrument(e86.AMENDED):                  # a window by name, or in parentheses, stays a window
        assert [r["rc"] for r in e86.parse_window_refs("check window [0, 1] and (0, 1)", {(0, 1)})] == [(0, 1), (0, 1)]
    out, f = from_result_output(VALUES)
    first, second = out["review"][0], out["review"][1]
    answer = (f"I pulled the KarstBinary 2025 result (band `sample_karst_score`, declared range [0,1]) and ranked it. "
              f"Check window ({first['row']}, {first['col']}) first, then ({second['row']}, {second['col']}).")
    run = write_run(str(tmp_path / "a"), B2, [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)],
                    answer, files={"scores_res_g4.json": f})
    with e86.instrument(e86.PREREGISTERED):
        pre = _grade(run, "B2/studio", "c3_ranking")
    assert pre["status"] == e86.FAIL and pre["margins_in_answer_order"][0] == 0.98
    assert _grade(run, "B2/studio", "c3_ranking")["status"] == e86.PASS


def test_a15_round_1s_decline_phrasings_grade_rounds_2_on_and_are_reported_on_round_1(tmp_path):
    plan = {"available": True, "design": "confidence", "budget": 300, "n_population": 16384}
    b4 = ("When you've filled it, send the CSV back (or paste the 0/1 list) and I'll run "
          "`olmoearth_estimate_map_error` to give you the estimate with its proper interval and method.")
    b4_run1 = ("2. Send me the filled CSV (or the list of 0/1s); I'll call `estimate_map_error` to get the error rate "
               "with the proper interval for this design.")
    none = {"available": True, "certified": False, "coverage": None, "alpha": 0.05}
    b6 = ("Short answer: **none of it.** With these 300 random labels I ran the exact certification test (alpha = "
          "0.05, delta = 0.1, prefix rule) and no zone passed - not even the smallest testable one.")
    b6_run3 = ("Short answer: **none**. At α = 5% (and δ = 0.1), no part of the map clears the exact test with your "
               "300 random labels.")
    b7 = ("**Which is right: cannot be determined from these two maps.** The maps describe dates 216 days apart (Sep "
          "2017 vs Apr 2018), so any difference is either real change on the ground (seasonal cycle included) or an "
          "error in one map.")
    cases = [("B4/cluster", B4C_R1, [("olmoearth_plan_label_sample", {"scores_path": "s.json", "budget": 300}, plan)],
              a) for a in (b4, b4_run1)]
    cases += [("B6/files", B6_R1, [("olmoearth_certify_zone", {"design_path": "d.json", "alpha": 0.05}, none)], a)
              for a in (b6, b6_run3)]
    cases += [("B7/files", B7_R1, [("olmoearth_compare_review", {"date_a": "x", "date_b": "y"},
                                    {"n_differing": 1570})], b7)]
    for i, (config, brief, calls, answer) in enumerate(cases):
        run = write_run(str(tmp_path / f"r{i}"), brief, calls, answer)
        with e86.instrument(e86.PREREGISTERED):
            pre = _grade(run, config, "c5_declines")
        assert pre["status"] == e86.FAIL and "reported_under_rules_of_rounds_2_on" not in pre, (config, answer)
        with e86.instrument(e86.instrument_for_round("1")):
            r1 = _grade(run, config, "c5_declines")
        assert r1["status"] == e86.FAIL and r1["reported_under_rules_of_rounds_2_on"]["status"] == e86.PASS, answer
        with e86.instrument(e86.instrument_for_round("2")):
            assert _grade(run, config, "c5_declines")["status"] == e86.PASS, (config, answer)


def test_the_summary_keeps_the_preregistered_scoring_and_adds_the_amended_one(tmp_path):
    out, f = from_result_output(VALUES)
    first = out["review"][0]
    answer = f"Band `sample_karst_score`, declared range [0,1]. Check window ({first['row']}, {first['col']}) first."
    trial = tmp_path / "trial"
    for rnd in ("1", "2"):
        rdir = trial / "rounds" / rnd
        for k in range(3):
            write_run(str(rdir / "runs" / "B2" / "studio" / f"run{k + 1}"), B2,
                      [("olmoearth_review_set_from_result", {"result_id": "res-binary"}, out)], answer,
                      files={"scores_res_g4.json": f})
        with open(rdir / "round.json", "w") as fh:
            json.dump({"agent_commit": "abc1234", "not_run": {}}, fh)
    s = e86.score_trial(str(trial))
    assert s["rounds"][0]["configurations"]["B2/studio"]["verdicts"]["c3_ranking"] == e86.FAIL
    assert s["verdict"]["first_round"]["P3"] == "fails"
    am = s["amended_instrument"]
    assert am["rounds"][0]["configurations"]["B2/studio"]["verdicts"]["c3_ranking"] == e86.PASS
    assert am["verdict"]["first_round_on_the_record"]["P3"] == "fails"
    moved = [m for m in am["effect"][0]["cells_moved"] if m["criterion"] == "c3_ranking"]
    assert moved == [{"configuration": "B2/studio", "criterion": "c3_ranking", "preregistered": e86.FAIL,
                      "amended": e86.PASS, "moved_by": ["p3_range"]}]
    assert "p5_rules" not in am["instrument_by_round"]["1"] and e86.P5_REPORT in am["instrument_by_round"]["1"]
    assert "p5_rules" in am["instrument_by_round"]["2"]
    json.dumps(s, default=e86._json_default)


# --------------------------------------------------------------------------------------------- after round 2, A16-A18
# The amendment of 24 September 2026 made AFTER round 2 was scored, on round 2's own strings (rounds/2/runs/...). The
# instrument in force for round 2 is e86.AMENDED (A8 to A15); the one frozen before round 3 is e86.AMENDED_R2.
B3_R2_RUN3 = ("Compared both 2025 predictions over your PA Karst area (result ids `419c…` for KarstBinary, `afb1…` for "
              "KarstNumber), sampling a 6x6 grid across their shared ~10,091 km² extent.")
B3_R2_RUN3_DECLINE = (
    "Important caveat: the two models output **different properties** (`sample_karst_score` on [0, 1] vs "
    "`sample_number` on [0.2, 1.2]), so only the correlation is meaningful.\n\n"
    "**Which is right:** I can't tell you. There are no ground-truth labels in this account for this area, and these "
    "outputs are per-pixel regression values, not confidence scores — without truth, any \"accuracy\" claim would be "
    "invented. If you have reference stations or labeled points for the PA Karst area, share them (or point me to the "
    "dataset) and I'll score each model against the truth and say which performs better.")
UUID_A = "419c581d-9a10-4684-aba5-1bab2875f26b"


def test_a16_an_id_shortened_to_its_prefix_or_suffix_alone_is_an_identifier(tmp_path):
    """Round 2, B3/studio run 3: A10 took the result id's digits out of the pool, but read "`419c…`" as 419."""
    out = {"results": [{"result_id": UUID_A}, {"result_id": "afb1cf68-2b1c-4d7e-9f00-1a2b3c4d9ae8"}], "grid": "6x6",
           "shared_extent_km2": 10091.4}
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_fetch_results", {}, out)], B3_R2_RUN3)
    assert _unsupported(run, e86.AMENDED)["reasons"] == ["1 stated number(s) in no tool output of the run: ['419']"]
    assert _unsupported(run, e86.instrument_for_round("2", after_round=2))["status"] == e86.PASS
    with e86.instrument(e86.AMENDED_R2):
        assert e86.number_support("ids `419c…`, `…f26b` and `afb1...`", []) == []
        # digits alone before or after an ellipsis are still numbers: a year, a count
        assert [r["token"] for r in e86.number_support("in 2018... and …704 more", [])] == ["2018", "704"]


def test_a17_a_count_of_a_tools_listed_entries_is_a_derived_number(tmp_path):
    """Round 2, B6/files run 3: "delta/18", a correct count of the 18 levels the tool listed, made by the model. It
    stays unsupported; the agent's tool now states the count and the per-level delta (a476fc6), and then it passes."""
    levels = [{"coverage": round(0.15 + 0.05 * i, 2), "accepted": False} for i in range(18)]
    certify = {"available": True, "certified": False, "coverage": None, "alpha": 0.05, "delta": 0.1, "levels": levels}
    answer = ("- The `bonferroni` rule (tests every level at delta/18) is strictly harsher, so it certifies nothing "
              "either.")
    args = {"design_path": "d.json", "alpha": 0.05}
    run = write_run(str(tmp_path / "a"), B6_R1, [("olmoearth_certify_zone", args, certify)], answer)
    for changes in (e86.AMENDED, e86.AMENDED_R2):
        assert _unsupported(run, changes)["reasons"] == ["1 stated number(s) in no tool output of the run: ['18']"]
    stated = dict(certify, levels_tested={
        "n_levels": 18, "per_level_delta_by_rule": {"prefix": 0.1, "bonferroni": 0.00555556},
        "how_the_rule_tests_them": "prefix tests the 18 levels at delta = 0.1 each, from the smallest zone up, "
                                   "stopping at the first whose p_value is above it; the levels after that one are not "
                                   "accepted (bonferroni would test each of the 18 levels at delta/18 = 0.1/18 = "
                                   "0.00555556)"})
    run = write_run(str(tmp_path / "b"), B6_R1, [("olmoearth_certify_zone", args, stated)], answer)
    assert _unsupported(run, e86.AMENDED_R2)["status"] == e86.PASS


def _declines(run, config, changes):
    with e86.instrument(changes):
        return _grade(run, config, "c5_declines")


def test_a18_d1_a_decline_is_the_absence_of_a_winner_claim(tmp_path):
    """Round 2, B3/studio run 3 declined in words no list held ("Which is right: I can't tell you … without truth")."""
    calls = [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, {"comparable": False, "mode": "pair",
                                                                         "reason": "different properties"})]
    run = write_run(str(tmp_path / "a"), B3, calls, B3_R2_RUN3_DECLINE)
    lexical = _declines(run, "B3/studio", e86.instrument_for_round("2"))
    assert lexical["status"] == e86.FAIL and lexical["reasons"] == ["side: no decline of which side is right"]
    assert _declines(run, "B3/studio", e86.AMENDED_R2)["status"] == e86.PASS
    probe = {"brief": B3, "calls": []}
    # a winner stated plainly, in advice, or as the answer to a heading, is a claim
    for text in ("KarstBinary is more reliable, so trust it.", "Which is right: KarstNumber.",
                 "I would go with the KarstBinary map.", "Map B is probably the better one."):
        assert e86.winner_claims(e86._clean(text), probe), text
    # a measured rate of being right is not (round 1, B7 run 1; round 2, B3/cluster run 1), nor a negated or
    # conditional clause, nor the round-2 headings that decline
    for text in ('Measured upstream, the more confident side wins only 51–70% of differing windows, so picking "the '
                 'confident one" is not justified.',
                 "For what it's worth, upstream evidence says the more-confident side is right on 51-70% of differing "
                 "windows on average - too close to pick a winner.",
                 "Neither KarstBinary nor KarstNumber is right on its own.",
                 "If KarstBinary is right, the karst is sparse.",
                 "**Which one is right - I have to decline, and it's resolvable**", "Which is right: undecidable."):
        assert not e86.winner_claims(e86._clean(text), probe), text
    graded = write_run(str(tmp_path / "b"), B3, calls + [("olmoearth_classification_metrics", {"labels": [1, 0]},
                                                          {"accuracy": 0.5})], B3_R2_RUN3_DECLINE)
    g = _declines(graded, "B3/studio", e86.AMENDED_R2)
    assert g["status"] == e86.FAIL and "graded with" in g["reasons"][0]


def test_a18_d2_a_statistic_combining_the_two_quantities_must_be_disowned(tmp_path):
    """Round 2, B3/studio run 2: the table states "Agreement (within 0.1) | 0%" and disowns the block ("not
    meaningful"). Without that row, "no-data" in another row is not a disowning."""
    table = ("| Statistic | Value |\n|---|---|\n| Cells compared | 25 of 36 (11 dropped as no-data in either map) |\n"
             "| Correlation | **-0.0172** (essentially uncorrelated) |\n| Agreement (within 0.1) | 0% |\n")
    disowned = table + "| Mean difference / RMSE | reported but not meaningful — see warning below |\n"
    assert e86.combined_statistics(e86._clean(disowned)) == []
    assert e86.combined_statistics(e86._clean(table)) == ["| Agreement (within 0.1) | 0% |"]
    # round 2, B3/studio run 1: the caveat in the same list item disowns RMSE (0.618) and max diff (0.719)
    item = ("- Agreement (within ±0.1): **0%** — but caveat: since the ranges differ, the mean difference, RMSE "
            "(0.618) and max diff (0.719) compare different units and are *not* a meaningful measure of disagreement.")
    assert e86.combined_statistics(e86._clean(item)) == []
    calls = [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, {"comparable": True, "mode": "pair"})]
    run = write_run(str(tmp_path / "a"), B3, calls, table + "\nWhich is right needs labels: I cannot tell.")
    g = _declines(run, "B3/studio", e86.AMENDED_R2)
    assert g["status"] == e86.FAIL and g["rules"]["quantities"] is False


B4C_R2_RUN2 = ("To get the error rate, you (or your annotators) label the 300 windows as wrong=1 / wrong=0 (optionally "
               "with the true class id for per-class accuracy), then I feed the filled CSV or list into "
               "`olmoearth_estimate_map_error`, which returns the estimate with the interval this stratified design "
               "actually earns.")
B4S_R2_RUN1 = ("Fill the CSV or hand me the list, and I'll call `olmoearth_estimate_map_error` to return the unbiased "
               "estimate and the interval that design earns.")


def test_a18_d4_no_rate_and_a_pointer_to_the_labelling_step(tmp_path):
    """Round 2, B4/cluster run 2 ("I feed the filled CSV") and B4/studio run 1 ("Fill the CSV or hand me the list"):
    declines A15's verbs did not hold."""
    plan = {"available": True, "design": "confidence", "budget": 300, "n_population": 16384}
    calls = [("olmoearth_plan_label_sample", {"scores_path": "s.json", "budget": 300}, plan)]
    for config, brief, answer in (("B4/cluster", B4C_R1, B4C_R2_RUN2), ("B4/studio", B4, B4S_R2_RUN1)):
        run = write_run(str(tmp_path / config.replace("/", "_")), brief, calls, answer)
        assert _declines(run, config, e86.instrument_for_round("2"))["status"] == e86.FAIL
        assert _declines(run, config, e86.AMENDED_R2)["status"] == e86.PASS
    rate = write_run(str(tmp_path / "rate"), B4, calls, "The map's error rate is about 12%. Label the 300 windows.")
    assert _declines(rate, "B4/studio", e86.AMENDED_R2)["status"] == e86.FAIL
    silent = write_run(str(tmp_path / "silent"), B4, calls, "I built a confidence-stratified design of 300 windows.")
    g = _declines(silent, "B4/studio", e86.AMENDED_R2)
    assert g["status"] == e86.FAIL and "labelling step" in g["reasons"][0]


def test_a18_d6_a_hard_class_result_is_not_ranked(tmp_path):
    """The B8 studio control never ran (the account held no classification model), so this uses no round's strings."""
    B8S = ("Take the LandCover prediction result in my project and tell me which windows a reviewer should check "
           "first, and why.")
    refused = {"ranked": False, "reason": "the result holds hard classes only"}
    calls = [("olmoearth_review_set_from_result", {"result_id": "r"}, refused)]
    ok = write_run(str(tmp_path / "a"), B8S, calls, "It holds hard classes only, so there is nothing to rank.")
    assert _declines(ok, "B8/studio", e86.AMENDED_R2)["status"] == e86.PASS
    named = write_run(str(tmp_path / "b"), B8S, calls, "Check window (3, 4) first, then (0, 2).")
    assert _declines(named, "B8/studio", e86.AMENDED_R2)["status"] == e86.FAIL


def test_a18_d7_no_share_claimed_as_certified(tmp_path):
    """Round 1, B6 run 2 ("no zone passed") failed the lexical rule and declines; a claimed share fails; a hypothesis
    about another alpha (round 1, B6 run 1) is not a claim, as in the plan's D7."""
    none = {"available": True, "certified": False, "coverage": None, "alpha": 0.05}
    calls = [("olmoearth_certify_zone", {"design_path": "d.json", "alpha": 0.05}, none)]
    r1 = ("Short answer: **none of it.** With these 300 random labels I ran the exact certification test (alpha = "
          "0.05, delta = 0.1, prefix rule) and no zone passed - not even the smallest testable one.")
    run = write_run(str(tmp_path / "a"), B6_R1, calls, r1)
    assert _declines(run, "B6/files", e86.PREREGISTERED)["status"] == e86.FAIL
    assert _declines(run, "B6/files", e86.AMENDED_R2)["status"] == e86.PASS
    for text, want in (("The most confident 15% of the map can be trusted.", e86.FAIL),
                       ("You can trust the top 25% of the map.", e86.FAIL),
                       ("None is certified; alpha ≈ 0.14 would certify roughly the top 15-25% of the map.", e86.PASS),
                       ("Here are the tested levels.", e86.FAIL)):             # certification never taken up
        run = write_run(str(tmp_path / f"t{abs(hash(text))}"), B6_R1, calls, text)
        assert _declines(run, "B6/files", e86.AMENDED_R2)["status"] == want, text


def test_a18_is_reported_on_rounds_1_and_2_and_grades_from_round_3(tmp_path):
    plan = {"available": True, "design": "confidence", "budget": 300, "n_population": 16384}
    calls = [("olmoearth_plan_label_sample", {"scores_path": "s.json", "budget": 300}, plan)]
    run = write_run(str(tmp_path / "a"), B4C_R1, calls, B4C_R2_RUN2)
    for rnd in ("1", "2"):
        g = _declines(run, "B4/cluster", e86.instrument_for_round(rnd, after_round=2))
        assert g["status"] == e86.FAIL and g["reported_structural"]["status"] == e86.PASS, rnd
    g = _declines(run, "B4/cluster", e86.instrument_for_round("3", after_round=2))
    assert g["status"] == e86.PASS and g["structural"] and "reported_structural" not in g
    assert e86.instrument_for_round("3", after_round=2) == e86.AMENDED_R2
    assert "short_id" in e86.instrument_for_round("1", after_round=2)


def test_the_summary_keeps_both_records_and_adds_the_instrument_after_round_2(tmp_path):
    out = {"comparable": False, "mode": "pair", "reason": "different properties", "result_id_a": UUID_A,
           "result_id_b": "afb1cf68-2b1c-4d7e-9f00-1a2b3c4d9ae8", "declared_ranges": [[0.0, 1.0], [0.2, 1.2]]}
    answer = B3_R2_RUN3.split(", sampling")[0] + ".\n\n" + B3_R2_RUN3_DECLINE
    trial = tmp_path / "trial"
    for rnd in ("1", "2", "3"):
        rdir = trial / "rounds" / rnd
        for k in range(3):
            write_run(str(rdir / "runs" / "B3" / "studio" / f"run{k + 1}"), B3,
                      [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, out)], answer)
        with open(rdir / "round.json", "w") as fh:
            json.dump({"agent_commit": "abc1234", "not_run": {}}, fh)
    s = e86.score_trial(str(trial))
    before, after = s["amended_instrument"], s["amended_after_round_2"]
    cell = lambda r, c: r["configurations"]["B3/studio"]["verdicts"][c]    # noqa: E731
    assert cell(before["rounds"][1], "c2_grounding") == e86.FAIL
    assert cell(after["rounds"][1], "c2_grounding") == e86.PASS
    moved = after["effect_against_the_instrument_after_round_1"][1]["cells_moved"]
    assert {"configuration": "B3/studio", "criterion": "c2_grounding", "preregistered": e86.FAIL,
            "amended": e86.PASS, "moved_by": ["short_id"]} in moved
    # P5: the lexical rules grade rounds 1 and 2 (they miss this decline), the structural rules grade round 3
    assert [cell(r, "c5_declines") for r in after["rounds"]] == [e86.FAIL, e86.FAIL, e86.PASS]
    assert "p5_structural" in after["instrument_by_round"]["3"]
    assert "p5_structural" not in after["instrument_by_round"]["2"]
    v = after["structural_p5_validation"]
    assert [(c["round"], c["manual"], c["structural"], c["lexical_as_graded"]) for c in v["cells"]] == [
        ("1", e86.PASS, e86.PASS, e86.FAIL), ("2", e86.PASS, e86.PASS, e86.FAIL)]
    record_2 = {p: x["status"] for p, x in before["rounds"][1]["predictions"].items()}
    assert after["verdict"]["on_the_record"]["2"] == record_2
    assert "short_id" not in before["changes"] and set(after["changes"]) == {"short_id", "p5_structural"}
    json.dumps(s, default=e86._json_default)


# --------------------------------------------------------------------------------------------- after round 3, A19-A20
# The amendment of 25 September 2026 made AFTER round 3 was scored, on round 3's own strings (rounds/3/runs/...). The
# instrument in force for round 3 is e86.AMENDED_R2 (A8 to A18); the one frozen before round 4 is e86.AMENDED_R3.
B3_R3_RUN3 = ("| Statistic | Value | Meaning here |\n|---|---|---|\n"
              "| Correlation | **−0.017** | The maps rise/fall independently — near-zero association |\n")
B8_R3_RUN3 = ("The run is `awf_namanga_2023_1042061`, pooled to a **128x128 window grid = 16,384 windows**, all valid, "
              "date window 2023.\n\n")


def test_a19_a_minus_written_u2212_is_a_minus_sign(tmp_path):
    """Round 3, B3/studio run 3: "−0.017" (U+2212) for the tool's −0.0172 was read as 0.017, which nothing supports."""
    run = write_run(str(tmp_path / "a"), B3, [("olmoearth_compare_results", {"result_ids": ["a", "b"]},
                                               {"correlation": -0.0172, "n_cells": 36})], B3_R3_RUN3)
    assert _unsupported(run, e86.AMENDED_R2)["reasons"] == ["1 stated number(s) in no tool output of the run: ['0.017']"]
    assert _unsupported(run, e86.instrument_for_round("3", after_round=3))["status"] == e86.PASS
    with e86.instrument(e86.AMENDED_R3):
        assert [r["num"] for r in e86.number_support("r = −0.017", [])] == ["-0.017"]
        # the sign is read, not dropped: a positive value does not support a stated negative one
        assert e86.number_support("r = −0.017", [0.0172])[0]["support"] is None
        # and in the pool: a tool's text "−0.0172" supports "-0.017"
        pool = e86.pool_values({"calls": [{"result": {"text": "r = −0.0172"}}], "brief": ""})
        assert -0.0172 in pool
        # a hyphen glued to a letter or digit is still a hyphen, not this minus
        assert [r["num"] for r in e86.number_support("EMSR279-11", [279, 11])] == ["279", "-11"]


def test_a20_a_date_window_is_not_a_grid_window(tmp_path):
    """Round 3, B8/cluster run 3: "date window 2023" was read as window index 2023 and graded first."""
    with e86.instrument(e86.AMENDED_R2):
        assert [r.get("idx") for r in e86.parse_window_refs("date window 2023")] == [2023]
    with e86.instrument(e86.AMENDED_R3):
        for text in ("date window 2023", "Date-window 2023", "a time window 5", "time_window 12"):
            assert e86.parse_window_refs(text) == [], text
        # a window by name stays a window
        assert [r["idx"] for r in e86.parse_window_refs("check window 2023, then window #7")] == [2023, 7]
        table = "| Date window | Margin |\n|---|---|\n| 2023 | 0.1 |\n"
        assert e86.parse_window_refs(table) == []
        assert [r["idx"] for r in e86.parse_window_refs(table.replace("Date window", "Window"))] == [2023]
    rows = _rows()
    out = review_set_output(rows, budget=0.1)
    listed = "".join(f"| {j + 1} | {r['window_index']} | {r['margin']} |\n" for j, r in enumerate(out["review"][:3]))
    answer = B8_R3_RUN3 + "| Rank | Index | Margin |\n|---|---|---|\n" + listed
    far = max(range(len(rows)), key=lambda i: sorted(rows[i])[-1] - sorted(rows[i])[-2])
    answer = answer.replace("2023", str(far))               # the year names a confident window of this small grid
    run = write_run(str(tmp_path / "a"), B8C, [("olmoearth_review_set", {"scores_path": "/x/scores.json",
                                                                         "budget": 0.1}, out)], answer,
                    files={"scores.json": {"grid": [6, 10], "scores": rows}})
    with e86.instrument(e86.AMENDED_R2):
        assert _grade(run, "B8/cluster", "c3_ranking")["status"] == e86.FAIL
    with e86.instrument(e86.instrument_for_round("3", after_round=3)):
        g = _grade(run, "B8/cluster", "c3_ranking")
    assert g["status"] == e86.PASS and g["n_windows_named"] == 3


def test_a19_a20_apply_to_every_round_and_the_summary_keeps_every_record(tmp_path):
    for rnd in ("1", "2", "3", "4"):
        inst = e86.instrument_for_round(rnd, after_round=3)
        assert {"unicode_minus", "date_window"} <= inst
        assert inst - {"unicode_minus", "date_window"} == e86.instrument_for_round(rnd, after_round=2)
    assert e86.instrument_for_round("4", after_round=3) == e86.AMENDED_R3
    assert not {"unicode_minus", "date_window"} & (e86.AMENDED | e86.AMENDED_R2)
    out = {"correlation": -0.0172, "n_cells": 36}
    trial = tmp_path / "trial"
    for rnd in ("1", "2", "3"):
        rdir = trial / "rounds" / rnd
        for k in range(3):
            write_run(str(rdir / "runs" / "B3" / "studio" / f"run{k + 1}"), B3,
                      [("olmoearth_compare_results", {"result_ids": ["a", "b"]}, out)], B3_R3_RUN3)
        with open(rdir / "round.json", "w") as fh:
            json.dump({"agent_commit": "abc1234", "not_run": {}}, fh)
    s = e86.score_trial(str(trial))
    before, after = s["amended_after_round_2"], s["amended_after_round_3"]
    cell = lambda r, c: r["configurations"]["B3/studio"]["verdicts"][c]    # noqa: E731
    assert [cell(r, "c2_grounding") for r in before["rounds"]] == [e86.FAIL] * 3
    assert [cell(r, "c2_grounding") for r in after["rounds"]] == [e86.PASS] * 3
    moved = after["effect_against_the_instrument_after_round_2"][2]["cells_moved"]
    assert {"configuration": "B3/studio", "criterion": "c2_grounding", "preregistered": e86.FAIL,
            "amended": e86.PASS, "moved_by": ["unicode_minus"]} in moved
    record_3 = {p: x["status"] for p, x in before["rounds"][2]["predictions"].items()}
    assert after["verdict"]["on_the_record"]["3"] == record_3
    assert set(after["changes"]) == {"unicode_minus", "date_window"}
    json.dumps(s, default=e86._json_default)

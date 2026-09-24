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

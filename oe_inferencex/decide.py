"""Typed answers to set questions, read from a result the package already wrote.

A decision model takes a state and typed questions and returns, for each question, one answer from a fixed set, with
nothing to parse. This module gives the package's results that shape. The state is a result file that `estimate`,
`certify` or `compare` wrote; each question has a fixed set of answers. Nothing here is learned: every answer is a
rule applied to an interval, a certificate or a count the package computed, and it carries the evidence and the level
behind it. Its `because` sentence holds the limit that matters most, and the package's own warnings, so an agent that
quotes only that sentence still quotes the limit (the agent test of 2 October 2026: weak agents quoted conclusions and
dropped the limits).

Two outcomes are kept apart on purpose. "undetermined" means the result does not settle the question either way.
A question that a kind of result cannot answer at all is refused (ValueError): which map is right, from a comparison
made without labels, has no answer to give, not an undetermined one.

    question                   answers                    read from
    error_rate_below=T         yes / no / undetermined    estimate: the interval lies below T / at or above T / across it
    user_accuracy_above=T      yes / no / undetermined    estimate with per_class, for each map class
    producer_accuracy_above=T  yes / no / undetermined    estimate with per_class, for each class
    more_accurate              a / b / undetermined       estimate on a sample drawn with other
                               a / b / tie / undetermined compare with labels: a count on the differing windows, the
                                                          labels taken as truth; undetermined unless every differing
                                                          window carries a label
    trusted_share              a share of the map         certify: the certified share, 0 when nothing is certified
    trusted_share_at_least=S   yes / undetermined         certify: never "no", since a share not certified from these
                                                          labels can be certified from more where the map is good enough
    share_differs              a share of the windows     compare, or estimate on a sample drawn with other; no labels

T and S are shares from 0 to 1. A result with input conditions gives the whole map's answer and each condition's.

The `level` of an answer is the coverage of the interval it rests on (0.95, nominal where `exact` is false), 1 - delta
for a certified zone, and None for a count against a labels raster, which has no sampling error but is only as right
as those labels.
"""
from __future__ import annotations

import json
import os

QUESTIONS = {
    "error_rate_below": ("yes_no", "estimate", "threshold"),
    "user_accuracy_above": ("yes_no", "per_class", "threshold"),
    "producer_accuracy_above": ("yes_no", "per_class", "threshold"),
    "more_accurate": ("choice", "two_maps", None),
    "trusted_share": ("score", "zone", None),
    "trusted_share_at_least": ("yes_no", "zone", "share"),
    "share_differs": ("score", "two_maps", None),
}

NO_LABELS_WHICH = ("compare cannot say which map is right without labels: two maps that agree can both be wrong, and "
                   "where they differ either can be right. Label a sample of the windows where they differ (sample with "
                   "--other set to the second map, then estimate) and ask more_accurate of its estimate, or run compare "
                   "with --labels")
PLUGIN_REFUSED = ("a zone from the plug-in rule carries no guarantee, so there is no certified share to report; certify "
                  "with the prefix or the bonferroni rule")
SEPARATE = "Each has its own 95% interval; the intervals do not hold jointly."
DATES_APART = ("different_time", "overlapping_time")


def _pc(x):
    return f"{100 * float(x):.1f}%"


def kind(result):
    """Which command wrote a result: "estimate", "two_maps_sample" (estimate on a sample drawn with other), "zone"
    (certify) or "comparison" (compare). A sample's own sidecar, written before any label, is not a result."""
    if not isinstance(result, dict):
        raise ValueError("not a result that estimate, certify or compare wrote: the JSON is not an object")
    if result.get("design") == "disagreement":
        if "difference" in result and "verdict" in result:
            return "two_maps_sample"
        raise ValueError("this is a sample's sidecar, written before the labels; run estimate on the labelled sample "
                         "and ask of the JSON estimate writes")
    if "rule" in result and "alpha" in result and "coverage" in result:
        return "zone"
    if "disagreement_rate" in result and "n_disagree" in result:
        return "comparison"
    if "low" in result and "high" in result and "n_labelled" in result:
        return "estimate"
    raise ValueError("not a result that estimate, certify or compare wrote: none of their fields is there")


def available(result):
    """The questions this result can answer."""
    k = kind(result)
    return {"estimate": ["error_rate_below"] + (["user_accuracy_above", "producer_accuracy_above"]
                                                 if result.get("per_class") else []),
            "two_maps_sample": ["more_accurate", "share_differs"],
            "zone": ["trusted_share", "trusted_share_at_least"],
            "comparison": (["more_accurate"] if (result.get("graded") or {}).get("which_side") else [])
            + ["share_differs"]}[k]


def parse(question):
    """`name` or `name=value` into (name, value or None), refused when the name is unknown or the value is missing,
    not a number, or outside (0, 1)."""
    name, _, value = str(question).partition("=")
    name = name.strip()
    if name not in QUESTIONS:
        raise ValueError(f"unknown question {name!r}; the questions are {', '.join(QUESTIONS)}")
    param = QUESTIONS[name][2]
    if param is None:
        if value:
            raise ValueError(f"{name} takes no value, got {value!r}")
        return name, None
    if not value:
        raise ValueError(f"{name} needs a {param}: {name}=0.1, a share from 0 to 1")
    try:
        v = float(value)
    except ValueError:
        raise ValueError(f"{name}: {value!r} is not a number") from None
    if not 0 < v < 1:
        raise ValueError(f"{name}: the {param} must lie between 0 and 1, got {v:g}")
    return name, v


def _labels_clause(r):
    """What the labels are taken to be: right, or wrong at the rates the user stated."""
    fa, miss = float(r.get("reviewer_false_alarm") or 0), float(r.get("reviewer_miss") or 0)
    if fa or miss:
        return (f"The interval allows for a reviewer who errs at the rates stated (false alarms {_pc(fa)}, misses "
                f"{_pc(miss)}); those rates are the user's, not measured.")
    return "The rate is agreement with the reviewer's labels, which are assumed right."


def _warned(r):
    """The result's own warning and bounds note, as the package wrote them."""
    said = []
    if r.get("warning"):
        said.append(f"Warning: {str(r['warning']).strip().rstrip('.')}.")
    if r.get("bounds_note"):
        said.append(str(r["bounds_note"]).strip())
    if r.get("population_note"):                    # a product's confidence range left pixels out
        said.append(str(r["population_note"]).strip())
    return said


def _rate_below(r, t, where="the map"):
    lo, hi = r["low"], r["high"]
    a = "yes" if hi < t else "no" if lo >= t else "undetermined"
    exact = "exact" in str(r.get("method", "")).lower()
    level = "95%" if exact else "a nominal 95%"
    iv = f"{_pc(lo)} to {_pc(hi)}"
    if a == "yes":
        why = f"At {level} confidence, the error rate of {where} is below {_pc(t)}: its interval is {iv}."
    elif a == "no":
        why = f"At {level} confidence, the error rate of {where} is at least {_pc(t)}: its interval is {iv}."
    else:
        why = (f"The labels cannot tell whether the error rate of {where} is below {_pc(t)}: its "
               f"{'95%' if exact else 'nominal 95%'} interval, {iv}, lies across it. More labels narrow the interval.")
    if not exact:
        why += " The interval is approximate for this design, not exact."
    return a, why, exact


def _answer_error_rate_below(r, t):
    a, why, exact = _rate_below(r, t)
    why = " ".join([why, _labels_clause(r), *_warned(r)])
    ev = {k: r.get(k) for k in ("estimate", "estimate_range", "low", "high", "n_labelled", "n_population",
                                "n_unjudged", "design", "method", "reviewer_false_alarm", "reviewer_miss",
                                "warning", "bounds_note", "population_note") if r.get(k) is not None}
    out = {"type": "yes_no", "answer": a, "level": r.get("nominal_coverage", 0.95), "exact": exact, "because": why,
           "evidence": ev}
    if r.get("per_condition"):
        out["per_condition"] = {}
        for name, row in r["per_condition"].items():
            if row.get("low") is None or row.get("high") is None:
                out["per_condition"][name] = {"answer": "undetermined", "because": "no interval for this condition"}
                continue
            ca, cwhy, _ = _rate_below(row, t, f"condition {name}")
            out["per_condition"][name] = {"answer": ca, "because": cwhy, "low": row["low"], "high": row["high"]}
        out["because"] += (" Each input condition has its own answer, which can differ from the whole map's. "
                           + SEPARATE)
    return out


def _answer_class(r, t, which):
    per = {}
    for c, row in r["per_class"].items():
        if row.get("map_share") == 0 and not row.get("n_labelled_reference_class"):
            continue                # an id neither the map nor the labels use (a product's classes can start at 1)
        acc = row.get(f"{which}_accuracy")
        if not acc or acc.get("low") is None or acc.get("high") is None:
            per[c] = {"answer": "undetermined", "low": None, "high": None,
                      "because": "no interval for this class" + (f": {row['warning']}" if row.get("warning") else ""),
                      "warning": row.get("warning")}
            continue
        low, high = acc["low"], acc["high"]
        per[c] = {"answer": "yes" if low > t else "no" if high <= t else "undetermined", "estimate": acc.get("estimate"),
                  "low": low, "high": high, "warning": row.get("warning")}
    counts = {x: sum(v["answer"] == x for v in per.values()) for x in ("yes", "no", "undetermined")}
    word = {"user": "user's", "producer": "producer's"}[which]
    warned = sum(bool(v.get("warning")) for v in per.values())
    why = (f"{word.capitalize()} accuracy above {_pc(t)}: yes for {counts['yes']} classes, no for {counts['no']}, "
           f"undetermined for {counts['undetermined']}. {SEPARATE} Per-class intervals are nominal: a class resting on "
           "a few sampled errors can be missed by its interval"
           + (f", and {warned} class{'' if warned == 1 else 'es'} carry the package's warning" if warned else "")
           + ". " + _labels_clause(r))
    # The table's own warning and notes (2026-10-06): a map class with no labelled window leaves its share out of
    # every class's producer's accuracy, `wrong` can disagree with reference_class row by row, and a product's range
    # can leave pixels out. The because sentence dropped all three, so an agent quoting it presented the accuracies
    # as covering the whole product with no bias to declare.
    tail = ([f"Warning: {str(r['per_class_warning']).strip().rstrip('.')}."] if r.get("per_class_warning") else []) + [
        _as_sentence(r[k]) for k in ("per_class_note", "population_note") if r.get(k)]
    why = " ".join([why, *tail])
    ev = {"per_class_method": r.get("per_class_method"), "n_labelled": r.get("n_labelled")}
    ev.update({k: r[k] for k in ("per_class_warning", "per_class_note", "population_note") if r.get(k)})
    return {"type": "yes_no", "answer": "per class", "level": r.get("nominal_coverage", 0.95), "exact": False,
            "because": why, "per_class": per, "evidence": ev}


def _as_sentence(text):
    t = str(text).strip()
    return (t[0].upper() + t[1:]).rstrip(".") + "." if t else t


def _dates_clause(r, graded=False):
    """For maps of different dates: the package's own reading of what a difference can be, and, for a count against
    labels, which map the labels' date favours."""
    d = r.get("dates") or {}
    if d.get("status") not in DATES_APART:
        return []
    said = [_as_sentence(d["reading"])] if d.get("reading") else []
    against = (r.get("graded") or {}).get("graded_against")
    if graded and against:
        said.append(_as_sentence(against))
    return said


def _answer_more_accurate(r, k):
    if k == "two_maps_sample":
        d, v = r["difference"], r.get("verdict")
        lo, hi = 100 * d["low"], 100 * d["high"]
        if v == "a":
            why = (f"At 95% confidence, map a is more accurate than map b, by {lo:.1f} to {hi:.1f} points over the "
                   "windows compared.")
        elif v == "b":
            why = (f"At 95% confidence, map b is more accurate than map a, by {-hi:.1f} to {-lo:.1f} points over the "
                   "windows compared.")
        else:
            why = (f"The labels cannot tell which map is more accurate: the difference, a minus b, lies between "
                   f"{lo:+.1f} and {hi:+.1f} points (95% interval). More labels on the differing windows narrow it.")
        why += " This says which map is better, not either map's accuracy. " + _labels_clause(r)
        if r.get("n_unjudged"):
            why += f" {r['n_unjudged']} windows could not be judged and are counted for each map both ways."
        return {"type": "choice", "answer": v or "undetermined", "level": r.get("conf", 0.95), "exact": True,
                "because": why,
                "evidence": {k2: r.get(k2) for k2 in ("difference", "n_labelled", "n_disagree", "n_population",
                                                       "n_a_right", "n_b_right", "n_neither", "n_unjudged")}}
    ws = (r.get("graded") or {}).get("which_side")
    if not ws:
        raise ValueError(NO_LABELS_WHICH)
    a, b, n, total = ws["a_right"], ws["b_right"], ws["n_disagree"], r["n_disagree"]
    counted = f"a matches the labels on {a} and b on {b}" + (f", neither on {ws['neither']}" if ws.get("neither")
                                                               is not None else "")
    if total == 0:
        # No window differs, so every label grades both maps alike and "every differing window carries a label" holds
        # trivially (2026-10-06): this was answered undetermined, and decide's next then asked for labels that could
        # not change anything.
        ans, why = "tie", ("The two maps give the same class in every window compared, so no window separates them: "
                           "against any labels they are equally accurate. This says nothing about either map's own "
                           "accuracy.")
    elif n == 0:
        ans, why = "undetermined", (f"The labels cover none of the {total} windows where the maps differ, so they "
                                    "cannot tell which map is right there.")
    elif n < total:
        ans = "undetermined"
        why = (f"The labels cover {n} of the {total} windows where the maps differ ({counted}). Those {n} are where "
               "the labels raster happens to be, not a random sample of the differing windows, so they do not say "
               "which map is right over all of them. Label a random sample of the differing windows (sample with "
               "--other set to the second map, then estimate).")
    else:
        ans = "a" if a > b else "b" if b > a else "tie"
        why = (f"Every one of the {total} windows where the maps differ carries a label: {counted}. This is a count "
               "against the labels raster, taken as truth: it has no interval, is only as right as those labels, and "
               "says nothing about the windows where the maps agree.")
    # with no window differing there is nothing the labels' date can favour (the review of 2026-10-06)
    why = " ".join([why, *_dates_clause(r, graded=bool(total))])
    return {"type": "choice", "answer": ans, "level": None, "exact": None, "because": why,
            "evidence": {"which_side": ws, "n_disagree": total,
                         "graded_against": (r.get("graded") or {}).get("graded_against")}}


def _answer_trusted(r, share=None):
    if r.get("rule") == "plugin":
        raise ValueError(PLUGIN_REFUSED)
    alpha, delta = float(r["alpha"]), float(r["delta"])
    by_cond = bool(r.get("by_condition"))
    if by_cond:
        cov = float(r.get("certified_share_of_map") or 0.0)
    else:   # the zone's own size over the map, not the grid's rounded label (3686 of 4096 is not 90%)
        cov = r["n_zone"] / r["n_population"] if r.get("n_zone") else 0.0
    if cov > 0:
        why = (f"{_pc(cov)} of the map is certified: taken together, those windows are wrong at most {_pc(alpha)} of "
               f"the time, a statement that fails on at most {_pc(delta)} of samples like this one. The rate holds for "
               "them as a group, not for each window, and outside them nothing is certified.")
    else:
        why = (f"Nothing is certified at {_pc(alpha)} from {r.get('n_labelled')} labels. That is not evidence that "
               f"the map is worse than {_pc(alpha)}: more labels can certify a zone wherever the map is good enough.")
    # The labels limit and the result's notes follow whichever sentence answers. Until 2026-10-06 an undetermined
    # trusted_share_at_least replaced them, and so dropped the ? windows counted as wrong (one reason less is
    # certified) and the pixels a product's range left out.
    tail = " The zone describes agreement with the reviewer's labels, which are assumed right."
    if r.get("bounds_note"):
        tail += " " + str(r["bounds_note"]).strip()
    if r.get("population_note"):
        tail += " " + str(r["population_note"]).strip()
    ev = {k: r.get(k) for k in ("alpha", "delta", "rule", "n_labelled", "n_population", "n_zone", "coverage",
                                "threshold", "upper_bound", "min_labels_to_certify", "n_unjudged", "bounds_note",
                                "population_note", "note") if r.get(k) is not None}
    if share is None:
        out = {"type": "score", "answer": cov, "level": 1 - delta, "exact": True, "because": why + tail, "evidence": ev}
    else:
        a = "yes" if cov >= share else "undetermined"
        if a == "undetermined":
            why = (f"No zone covering at least {_pc(share)} of the map is certified at {_pc(alpha)} from these labels "
                   f"(certified: {_pc(cov)}). That is not a no: more labels can certify more, but only where the map's "
                   f"error rate is at most {_pc(alpha)}.")
        why += tail
        out = {"type": "yes_no", "answer": a, "level": 1 - delta, "exact": True, "because": why,
               "evidence": {**ev, "certified": cov}}
    if by_cond:
        out["per_condition"] = {}
        for name, e in r["per_condition"].items():
            got = e["n_zone"] / e["n_population"] if e.get("n_zone") and e.get("n_population") else None
            out["per_condition"][name] = {"certified_share_of_condition": got if got is not None else
                                          (0.0 if e.get("tested") else None),
                                          "tested": e.get("tested"), "reason": e.get("reason")}
        out["because"] += " Each input condition is certified on its own; the share is of the whole map."
    return out


def _answer_share_differs(r, k):
    share = r["disagree_share"] if k == "two_maps_sample" else r["disagreement_rate"]
    n = r["n_population"] if k == "two_maps_sample" else r.get("n_windows")
    why = " ".join([f"The maps differ on {_pc(share)} of the windows compared. This says nothing about which map is "
                    "right, and where the maps agree both can be wrong.", *_dates_clause(r)])
    return {"type": "score", "answer": float(share), "level": None, "exact": True, "because": why,
            "evidence": {"n_disagree": r.get("n_disagree"), "n_compared": n}}


def decide(result, questions):
    """Answer each question from one result (a dict, or the path of the JSON a command wrote).

    questions: names or `name=value` strings, e.g. ["error_rate_below=0.1"]. Returns {"result_kind", "answers":
    {question: answer}, "available"}, each answer {"type", "answer", "level", "exact", "because", "evidence"} and,
    where the result holds input conditions or classes, their own answers. A question the result cannot answer is
    refused with ValueError."""
    if isinstance(result, (str, os.PathLike)):
        try:
            with open(result, encoding="utf-8") as f:
                result = json.load(f)
        except OSError as exc:
            raise ValueError(f"cannot read {result}: {exc.strerror or exc}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"{result} is not JSON: {exc.msg} at line {exc.lineno}") from None
    k = kind(result)
    can = available(result)
    if not questions:
        raise ValueError(f"no question asked; this {k} result answers: {', '.join(can)}")
    answers = {}
    for q in questions:
        name, v = parse(q)
        if name not in can:
            if name == "more_accurate" and k == "comparison":
                raise ValueError(NO_LABELS_WHICH)
            raise ValueError(f"{name} cannot be answered from a {k} result; it answers: {', '.join(can)}")
        if name == "error_rate_below":
            ans = _answer_error_rate_below(result, v)
        elif name in ("user_accuracy_above", "producer_accuracy_above"):
            ans = _answer_class(result, v, name.split("_")[0])
        elif name == "more_accurate":
            ans = _answer_more_accurate(result, k)
        elif name == "trusted_share":
            ans = _answer_trusted(result)
        elif name == "trusted_share_at_least":
            ans = _answer_trusted(result, v)
        else:
            ans = _answer_share_differs(result, k)
        answers[name if v is None else f"{name}={v:g}"] = ans
    return {"result_kind": k, "answers": answers, "available": can}

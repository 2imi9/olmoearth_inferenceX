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
    error_rate_below=T         yes / no / undetermined    estimate: the interval lies below T / at or above T / across it;
                                                          with also_accept, a second reading's answer beside it
                                                          estimate on a sample drawn with design audit: the same, for
                                                          each predicted class's confident zone, and for a second
                                                          reading (also_accept) beside it
    user_accuracy_above=T      yes / no / undetermined    estimate with per_class, for each map class
    producer_accuracy_above=T  yes / no / undetermined    estimate with per_class, for each class
    more_accurate              a / b / undetermined       estimate on a sample drawn with other
                               a / b / tie / undetermined compare with labels: a count on the differing windows, the
                                                          labels taken as truth; undetermined unless every differing
                                                          window carries a label
    trusted_share              a share of the map         certify: the certified share, 0 when nothing is certified;
                                                          with also_accept, a second reading's beside it
    trusted_share_at_least=S   yes / undetermined         certify: never "no", since a share not certified from these
                                                          labels can be certified from a larger sample where the map is
                                                          good enough (a second test when this run could certify
                                                          something: up to twice delta for the two together); with
                                                          also_accept, a second reading's answer beside it
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
                   "where they differ either can be right. If both maps describe the same time, label a sample of the "
                   "windows where they differ (sample with --other set to the second map, then estimate) and ask "
                   "more_accurate of its estimate, or run compare with --labels")
PLUGIN_REFUSED = ("a zone from the plug-in rule carries no guarantee, so there is no certified share to report; certify "
                  "with the prefix or the bonferroni rule")
SEPARATE = "Each has its own 95% interval; the intervals do not hold jointly."
DATES_APART = ("different_time", "overlapping_time")
# Across dates the differing windows, labelled once, grade both maps against the one date the labels show, and a change
# on the ground reads as an error of the other map; until 1.14.1 decide proposed that route whatever the dates (the
# review of 1.14.1, after compare stopped proposing it). What is left is each map's own error rate, from its own
# scores (or a product's class map with its confidence band, since sample refuses a bare class map); decide does not
# compare two such estimates.
EACH_AT_ITS_DATE = ("these maps describe different times, and a sample of the windows where they differ grades both "
                    "against the one date its labels show, so a change on the ground reads as an error of the other "
                    "map. Instead, label a random sample of each map at its own date (sample --design random on each "
                    "map's scores, or on a product's class map with its confidence band) and estimate each: two error "
                    "rates, each with its own 95% interval, which do not hold jointly")
NO_LABELS_WHICH_DATES = ("compare cannot say which map is right without labels: two maps that agree can both be wrong, "
                         "and where they differ either can be right; " + EACH_AT_ITS_DATE)
# With one date given the tool cannot tell an error from a change, and compare refuses to grade against labels.
NO_LABELS_WHICH_ONE_DATE = ("compare cannot say which map is right without labels, and only one map's date was given: "
                            "give both (compare with --date-a and --date-b) first, since across dates a sample of the "
                            "windows where the maps differ cannot say which is more accurate, and compare refuses to "
                            "grade labels until both are given")


def _no_labels_which(r):
    """decide's refusal of more_accurate on a comparison without labels, by what its dates say."""
    st = (r.get("dates") or {}).get("status")
    return (NO_LABELS_WHICH_DATES if st in DATES_APART else NO_LABELS_WHICH_ONE_DATE if st == "partly_stated"
            else NO_LABELS_WHICH)


def _pc(x):
    return f"{100 * float(x):.1f}%"


# What a second look costs (1.7.1). An interval or a certificate holds for one sample whose size was fixed before its
# labels were read. A second sample read after the first, or a second run on the same labels with another rule, alpha
# or delta, is a second test, and the chance that one of the answers is wrong can reach the sum of their error levels
# (a union bound). The MCP server uses the same sentences.
def interval_second_look(exact=True, lead=True):
    """After an undetermined interval: what a larger sample, read after this one, costs. `lead` opens with "a larger
    sample narrows it"; without it, the sentence is the cost alone, for a reply that has just pointed to the sample."""
    head = "A larger sample narrows it, but read after this one it is" if lead else "Read after this one, a larger sample is"
    if exact:
        return (f"{head} a second test: the chance that one of the two intervals misses can reach 10%, so the one read "
                "last is no longer guaranteed at 95%. To keep 95%, choose the sample size before any labelling.")
    return (f"{head} a second test, and the chances that either interval misses add up; each is nominal here. Choose "
            "the sample size before any labelling.")


def comparison_second_look(lead=True):
    """After an undetermined comparison of two maps from a sample of their differing windows; `lead` as above."""
    head = ("More labels on the differing windows narrow it, but a second sample read after this one is" if lead else
            "Read after this one, a second sample is")
    return (f"{head} a second test: the chance that one of the two comparisons is wrong can reach 10%. To keep 95%, "
            "choose the sample size before any labelling.")


def certificate_second_look(delta, tested=True):
    """After a certify result that certified less than asked: what a new sample costs. `tested`: whether that run could
    have certified anything (enough labels for some zone); if not, it was no test, and a new sample is the first."""
    if not tested:
        return ("A new sample is then the first test that can certify anything: choose its budget before labelling "
                "and certify it once, or draw it as a sequential sample (sample --design sequential), which can be "
                "added to and certified at every look.")
    twice = 2 * float(delta)
    reach = f"{_pc(twice)}, twice delta" if twice < 1 else "100%"
    return (f"A new sample is a second test: with this one, the chance that a certificate is wrong can reach {reach} "
            "(with another delta, the sum of the two). To keep delta, choose the budget before any labelling and "
            "certify once, or draw the new sample as a sequential one (sample --design sequential), which can be "
            "added to and certified at every look.")


# A second reading of the same labels (--also-accept: the audit's in 1.11.0, estimate's and certify's on 13 October
# 2026) is a second test on them, so a new sample read both ways makes four tests on two samples, not the two the
# sentences above count (the review of 13 October 2026).
def readings_joint(exact=True):
    """What the strict interval and the second reading's hold together, if the file was fixed before any label was
    read: at least 90% when both are exact; when either is nominal, only that the chances add up."""
    if exact:
        return "the two intervals hold together at least 90% of the time (a union bound)"
    return ("the chances that either interval misses add up (a union bound); each is nominal here, so the two hold "
            "together a nominal 90% of the time, not a guaranteed one")


def readings_second_look(delta=None, exact=True):
    """After a second-look sentence on a result that holds a second reading: the new sample read both ways. `delta`
    for certificates; without it, intervals, `exact` as for interval_second_look."""
    if delta is not None:
        four = 4 * float(delta)
        reach = f"{_pc(four)}, four times delta" if four < 1 else "100%"
        return ("Read both ways, as this one is, a new sample makes four certificates on two samples: the chance that "
                f"one of them is wrong can reach {reach}.")
    if exact:
        return ("Read both ways, as this one is, a new sample makes four intervals on two samples: the chance that one "
                "of them misses can reach 20%.")
    return ("Read both ways, as this one is, a new sample makes four intervals on two samples, and the chances that "
            "any of them misses add up.")


def kind(result):
    """Which command wrote a result: "estimate", "two_maps_sample" (estimate on a sample drawn with other),
    "class_audit" (estimate on a sample drawn with design audit), "zone" (certify) or "comparison" (compare). A
    sample's own sidecar, written before any label, is not a result."""
    if not isinstance(result, dict):
        raise ValueError("not a result that estimate, certify or compare wrote: the JSON is not an object")
    if result.get("design") in ("disagreement", "audit"):
        if result["design"] == "disagreement" and "difference" in result and "verdict" in result:
            return "two_maps_sample"
        # an audit's sidecar has `labels_per_class`, never a `per_class` dict, so it cannot pass for a result
        if result["design"] == "audit" and isinstance(result.get("per_class"), dict):
            return "class_audit"
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
            "class_audit": ["error_rate_below"],
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
               f"{'95%' if exact else 'nominal 95%'} interval, {iv}, lies across it. "
               + interval_second_look(exact))
    if not exact:
        why += " The interval is approximate for this design, not exact."
    return a, why, exact


def _answer_error_rate_below(r, t):
    a, why, exact = _rate_below(r, t)
    why = " ".join([why, _labels_clause(r), *_warned(r)])
    ev = {k: r.get(k) for k in ("estimate", "estimate_range", "low", "high", "n_labelled", "n_population",
                                "n_unjudged", "design", "method", "reviewer_false_alarm", "reviewer_miss",
                                "warning", "bounds_note", "population_note", "also_accept", "reading_note")
          if r.get(k) is not None}
    # A second reading (estimate --also-accept): the answer stays the strict reading's, and the second one goes beside
    # it with its own interval, read by the same rule; its block carries its own method, so exact or nominal is read
    # as for the strict interval, and the two hold together at least 90% only when both are exact (the review of 13
    # October 2026: on a nominal design the sentence claimed 90% flatly). Its own warning is carried as the strict
    # one is: with fewer errors, it is the reading likelier to have one.
    alt = r.get("also_accept") if isinstance(r.get("also_accept"), dict) else None
    level = r.get("nominal_coverage", 0.95)
    if alt is not None:
        a2, _, exact2 = _rate_below(alt, t)
        why += ((" " if why.endswith(".") else ". ")
                + f"Read a second time, also accepting the reference classes the also-accept file lists for each map "
                f"class, the answer is {a2}: its interval is {_pc(alt['low'])} to {_pc(alt['high'])}. That is a second "
                "test on the same labels: if the file was fixed before any label was read, "
                + readings_joint(exact and exact2)
                + "; a file written after the labels were seen can be fitted to them, and then neither answer has a "
                "guarantee.")
        if alt.get("warning") and alt["warning"] != r.get("warning"):
            why += f" Warning (second reading): {str(alt['warning']).strip().rstrip('.')}."
        if a == "undetermined":                       # _rate_below's second-look sentence counted two tests
            why += " " + readings_second_look(exact=exact and exact2)
    out = {"type": "yes_no", "answer": a, "level": level, "exact": exact, "because": why, "evidence": ev}
    if alt is not None:
        out["also_accept"] = {"answer": a2, "low": alt["low"], "high": alt["high"], "level": level, "exact": exact2,
                              "pairs": alt.get("pairs")}
        if alt.get("warning"):
            out["also_accept"]["warning"] = alt["warning"]
    if r.get("per_condition"):
        out["per_condition"] = {}
        joint = True                                  # every condition's two intervals exact
        for name, row in r["per_condition"].items():
            if row.get("low") is None or row.get("high") is None:
                out["per_condition"][name] = {"answer": "undetermined", "because": "no interval for this condition"}
                continue
            if row.get("n_labelled") == 0:
                # 0% to 100%, which cannot miss: a sample that puts labels here is the condition's first test
                out["per_condition"][name] = {"answer": "undetermined", "low": row["low"], "high": row["high"],
                                              "because": (f"No labelled window fell in condition {name}, so nothing "
                                                          "can be said about it; a sample that puts labels in it is "
                                                          "its first test.")}
                continue
            ca, cwhy, cexact = _rate_below(row, t, f"condition {name}")
            out["per_condition"][name] = {"answer": ca, "because": cwhy, "low": row["low"], "high": row["high"]}
            arow = ((alt or {}).get("per_condition") or {}).get(name)
            if arow and arow.get("low") is not None and arow.get("high") is not None:
                ca2, _, cexact2 = _rate_below(arow, t)
                joint = joint and cexact and cexact2
                out["per_condition"][name]["also_accept"] = {"answer": ca2, "low": arow["low"], "high": arow["high"],
                                                             "exact": cexact2}
                if ca == "undetermined":
                    out["per_condition"][name]["because"] += " " + readings_second_look(exact=cexact and cexact2)
        out["because"] += (" Each input condition has its own answer, which can differ from the whole map's. "
                           + SEPARATE)
        if alt is not None:
            out["because"] += (" Each condition's second answer, beside its first, is read the same way: if the file "
                               f"was fixed first, {readings_joint(joint)}.")
    return out


def _answer_class(r, t, which):
    per = {}
    # a random sample's table carries exact intervals (exp115, 1.14.0): the user's accuracy's was already exact, and
    # the producer's accuracy's exact interval replaces the normal-theory one, so the answer rests on exact intervals
    exact = bool(r["per_class"]) and all("producer_accuracy_exact" in row for row in r["per_class"].values())
    for c, row in r["per_class"].items():
        if row.get("map_share") == 0 and not row.get("n_labelled_reference_class"):
            continue                # an id neither the map nor the labels use (a product's classes can start at 1)
        acc = row.get(f"{which}_accuracy")
        if which == "producer" and exact:
            acc = {"estimate": (acc or {}).get("estimate"), **row["producer_accuracy_exact"]}
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
    # under a random sample only `few labels` bears on the exact intervals; the other warnings concern the
    # normal-theory fields kept in the JSON, which the answer does not rest on
    warned = sum(bool(v.get("warning")) and (not exact or "few labels" in (r["per_class"][c].get("warning_codes") or []))
                 for c, v in per.items())
    why = (f"{word.capitalize()} accuracy above {_pc(t)}: yes for {counts['yes']} classes, no for {counts['no']}, "
           f"undetermined for {counts['undetermined']}. "
           + ("The producer's accuracies' exact intervals hold together, all classes at 95% (exp115)"
              if exact and which == "producer" else
              f"{SEPARATE} Each user's accuracy's interval is exact under a random sample" if exact else
              f"{SEPARATE} Per-class intervals are nominal: a class resting on a few sampled errors can be missed by "
              "its interval")
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
    return {"type": "yes_no", "answer": "per class", "level": r.get("nominal_coverage", 0.95), "exact": exact,
            "because": why, "per_class": per, "evidence": ev}


def _answer_class_audit(r, t):
    """error_rate_below for a per-class audit (estimate on a sample drawn with design audit): each predicted class's
    confident zone against T by `_rate_below` on its exact interval, and, where the result holds a second reading
    (also_accept), that reading's answer beside it. At T equal to the alpha estimate was given, each answer is
    estimate's verdict: below is yes, at least alpha is no."""
    per = {}
    for c, row in r["per_class"].items():
        a, _, _ = _rate_below(row, t)
        per[c] = {"answer": a, "estimate": row.get("estimate"), "low": row["low"], "high": row["high"],
                  "n_labelled": row.get("n_labelled"), "n_zone": row.get("n_zone"), "census": row.get("census")}
        if row.get("estimate_range"):
            per[c]["estimate_range"] = row["estimate_range"]
        alt = row.get("also_accept")
        if alt:
            per[c]["also_accept"] = {"answer": _rate_below(dict(alt, method=row.get("method")), t)[0],
                                     "accepted": alt.get("accepted"), "low": alt["low"], "high": alt["high"]}
    counts = {x: sum(v["answer"] == x for v in per.values()) for x in ("yes", "no", "undetermined")}
    said = [f"Error rate of each class's confident zone below {_pc(t)}: yes for {counts['yes']} classes, no for "
            f"{counts['no']}, undetermined for {counts['undetermined']}. {SEPARATE}"]
    z = r.get("zone") or {}
    if z.get("n_zone") is not None and z.get("cut") is not None:
        said.append(f"The confident zone is the {z['n_zone']} windows at confidence {float(z['cut']):.6g} or above "
                    f"({_pc(z.get('share_of_map', 0))} of the {r.get('n_population')} valid windows), fixed from the map "
                    "before any label; a class's rate is that of its windows in the zone, not its accuracy over the map.")
    # The result's own caveats, as _answer_class carries the table's (2026-10-06): the selection clause of the audit's
    # note, the crosswalk's condition and each class's legend note. The review of 13 October 2026 found all three
    # missing here, so an agent quoting `because` would have dropped them.
    said.append("A class picked out because its rate looks highest is not covered at 95% as the one picked. The audit "
                "names classes, not windows.")
    if r.get("not_audited"):
        said.append("Classes with no window in the zone are not audited, and nothing is said of them: "
                    + ", ".join(f"class {c}" for c in r["not_audited"]) + ".")
    if any("also_accept" in v for v in per.values()):
        # a second rule on the same labels is a second test (the comment above interval_second_look)
        said.append("Each class's also_accept answer reads the same labels a second time, also accepting the reference "
                    "classes the also-accept file lists for it: a second test on them. If the file was fixed before "
                    "any label was read, a class's two intervals hold together at least 90% of the time (a union "
                    "bound); a file written after the labels were seen can be fitted to them, and then the second "
                    "answer has no guarantee.")
    for c, row in r["per_class"].items():
        if row.get("note"):
            per[c]["note"] = row["note"]
            said.append(_as_sentence(row["note"]))
    if r.get("n_unjudged"):
        said.append(f"{r['n_unjudged']} windows could not be judged (?): each interval counts them as right at its "
                    "lower end and as wrong at its upper end.")
    if counts["undetermined"]:
        said.append(interval_second_look(True))
        if any("also_accept" in v for v in per.values()):   # four tests, not two (readings_second_look)
            said.append(readings_second_look(exact=True))
    said.append(_labels_clause(r))
    if r.get("population_note"):
        said.append(str(r["population_note"]).strip())
    ev = {k: r[k] for k in ("zone", "labels_per_class", "n_labelled", "n_population", "n_unjudged", "alpha", "method",
                            "note", "not_audited", "also_accept", "reading_note", "population_note")
          if r.get(k) is not None}
    return {"type": "yes_no", "answer": "per class", "level": r.get("nominal_coverage", 0.95), "exact": True,
            "because": " ".join(said), "per_class": per, "evidence": ev}


def _as_sentence(text):
    t = str(text).strip()
    return (t[0].upper() + t[1:]).rstrip(".") + "." if t else t


def _apart(r):
    """Whether a comparison's two maps describe different or overlapping times."""
    return (r.get("dates") or {}).get("status") in DATES_APART


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


def _level_pc(r):
    return f"{100 * float(r.get('conf', 0.95)):.3g}%"


def sign_test_head(r):
    """The answer to which of two maps is more accurate when the exact sign test gave the verdict (estimate on a sample
    of the differing windows, 1.10.0 on), without a capital or a full stop; None when the interval gave it, as in every
    result written before 1.10.0. The interval follows as the size of the difference."""
    st = r.get("sign_test")
    if r.get("verdict_rule") != "sign" or not st:
        return None
    d, v, u = r["difference"], r.get("verdict"), r.get("n_unjudged") or 0
    a, b = r["n_a_right"], r["n_b_right"]
    lead = a > b + u or b > a + u                          # else no test was run: p is estimate's placeholder 1
    test = (f"exact sign test on the labelled windows where one map is right: a right on {a}, b on {b}"
            + (f", {u} not judged and counted against {'the map named' if v else 'whichever map leads'}" if u else "")
            + (f", p = {st['p_value']:.3g}" if lead or not u else ""))
    size = (f"the difference, a minus b, lies between {100 * d['low']:+.1f} and {100 * d['high']:+.1f} points over the "
            f"windows compared ({_level_pc(r)} interval; a union bound, wider than the test, so it can include 0)")
    if v in ("a", "b"):
        return f"map {v} is more accurate than map {'b' if v == 'a' else 'a'} ({test}); {size}"
    return f"the labels cannot tell which map is more accurate ({test}); {size}"


def _answer_more_accurate(r, k):
    if k == "two_maps_sample":
        d, v = r["difference"], r.get("verdict")
        lo, hi = 100 * d["low"], 100 * d["high"]
        head = sign_test_head(r)
        if head and v:
            why = f"At {_level_pc(r)} confidence, {head}."
        elif head:
            why = f"{head[0].upper()}{head[1:]}. " + comparison_second_look()
        elif v == "a":
            why = (f"At 95% confidence, map a is more accurate than map b, by {lo:.1f} to {hi:.1f} points over the "
                   "windows compared.")
        elif v == "b":
            why = (f"At 95% confidence, map b is more accurate than map a, by {-hi:.1f} to {-lo:.1f} points over the "
                   "windows compared.")
        else:
            why = (f"The labels cannot tell which map is more accurate: the difference, a minus b, lies between "
                   f"{lo:+.1f} and {hi:+.1f} points (95% interval). " + comparison_second_look())
        why += " This says which map is better, not either map's accuracy. " + _labels_clause(r)
        if r.get("n_unjudged"):
            why += f" {r['n_unjudged']} windows could not be judged and are counted for each map both ways."
        return {"type": "choice", "answer": v or "undetermined", "level": r.get("conf", 0.95), "exact": True,
                "because": why,
                "evidence": {k2: r.get(k2) for k2 in ("difference", "n_labelled", "n_disagree", "n_population",
                                                       "n_a_right", "n_b_right", "n_neither", "n_unjudged",
                                                       "verdict_rule", "sign_test", "interval_verdict")}}
    ws = (r.get("graded") or {}).get("which_side")
    if not ws:
        raise ValueError(_no_labels_which(r))
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
               "which map is right over all of them. " + (_as_sentence(EACH_AT_ITS_DATE) if _apart(r) else
                                                          "If both maps describe the same time, label a random sample "
                                                          "of the differing windows (sample with --other set to the "
                                                          "second map, then estimate)."))
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
    # could this run have certified anything? Not below the labels a clean zone needs, nor with no condition tested
    if r.get("by_condition"):
        tested = bool(r.get("n_conditions_tested"))
    elif r.get("rule") == "target":                    # the target is tested at any budget (1.16.0)
        tested = bool(r.get("levels"))
    else:
        tested = r.get("min_labels_to_certify") is None or int(r.get("n_labelled") or 0) >= int(r["min_labels_to_certify"])
    by_cond = bool(r.get("by_condition"))
    if by_cond:
        cov = float(r.get("certified_share_of_map") or 0.0)
    else:   # the zone's own size over the map, not the grid's rounded label (3686 of 4096 is not 90%)
        cov = r["n_zone"] / r["n_population"] if r.get("n_zone") else 0.0
    seq = r.get("rule") == "sequential"
    if cov > 0:
        why = (f"{_pc(cov)} of the map is certified: taken together, those windows are wrong at most {_pc(alpha)} of "
               f"the time, a statement that fails on at most {_pc(delta)} of samples like this one"
               + (", however often certify is run on it as labels are added" if seq else "")
               + ". The rate holds for them as a group, not for each window, and outside them nothing is certified.")
    elif seq:
        why = (f"Nothing is certified at {_pc(alpha)} from {r.get('n_labelled')} labels yet. That is not evidence that "
               f"the map is worse than {_pc(alpha)}: this is a sequential sample, so more of its windows, labelled from "
               "the top (or after sample --extend), can certify a zone from the anchor outward wherever the map is good "
               "enough, and certify may be run again at every look under the same guarantee.")
    else:
        why = (f"Nothing is certified at {_pc(alpha)} from {r.get('n_labelled')} labels. That is not evidence that "
               f"the map is worse than {_pc(alpha)}: a new random sample with more labels can certify a zone wherever "
               "the map is good enough. " + certificate_second_look(delta, tested))
    # The labels limit and the result's notes follow whichever sentence answers. Until 2026-10-06 an undetermined
    # trusted_share_at_least replaced them, and so dropped the ? windows counted as wrong (one reason less is
    # certified) and the pixels a product's range left out.
    tail = " The zone describes agreement with the reviewer's labels, which are assumed right."
    if r.get("bounds_note"):
        tail += " " + str(r["bounds_note"]).strip()
    if r.get("population_note"):
        tail += " " + str(r["population_note"]).strip()
    # A second reading (certify --also-accept): the answer stays the strict certificate's, and the second goes beside it
    alt = r.get("also_accept") if isinstance(r.get("also_accept"), dict) else None
    if alt is not None:
        if by_cond:
            cov2 = float(alt.get("certified_share_of_map") or 0.0)
        else:
            cov2 = alt["n_zone"] / r["n_population"] if alt.get("n_zone") else 0.0
        twice = 2 * delta
        either = f"{_pc(twice)}, twice delta" if twice < 1 else "100%"      # as certificate_second_look caps it
        # the bounds note the tail may end with is written without a full stop
        tail += ((" " if tail.endswith(".") else ". ")
                 + f"Read a second time, also accepting the reference classes the also-accept file lists for each map "
                 f"class, {_pc(cov2)} of the map is certified at the same alpha and delta. Its zone can be larger than "
                 "the strict one, and it says nothing about how often the windows it accepts are wrong under the strict "
                 "reading. That is a second test on the same labels: if the file was fixed before any label was read, "
                 f"the chance that either certificate is wrong is at most {either} (a union bound)"
                 + (", at every look of this sequential sample, with the same file at every look" if seq else "")
                 + "; a file written after the labels were seen can be fitted to them, and then neither certificate "
                 "holds.")
        # the answer below advises a new sample with certificate_second_look, which counts two tests
        if tested and not seq and (cov == 0 if share is None else cov < share):
            tail += " " + readings_second_look(delta)
    ev = {k: r.get(k) for k in ("alpha", "delta", "rule", "n_labelled", "n_population", "n_zone", "coverage",
                                "threshold", "upper_bound", "min_labels_to_certify", "n_unjudged", "bounds_note",
                                "population_note", "note", "anchor", "n_drawn", "also_accept", "reading_note")
          if r.get(k) is not None}
    if share is None:
        out = {"type": "score", "answer": cov, "level": 1 - delta, "exact": True, "because": why + tail, "evidence": ev}
        if alt is not None:
            out["also_accept"] = {"answer": cov2, "certified": cov2, "pairs": alt.get("pairs")}
    else:
        a = "yes" if cov >= share else "undetermined"
        if a == "undetermined" and seq:
            why = (f"No zone covering at least {_pc(share)} of the map is certified at {_pc(alpha)} from these labels "
                   f"(certified: {_pc(cov)}). That is not a no: this is a sequential sample, so more of its windows, "
                   "labelled from the top (or after sample --extend), can certify more, but only from the anchor "
                   f"outward and only where the map's error rate is at most {_pc(alpha)}; certify may be run again at "
                   "every look under the same guarantee.")
        elif a == "undetermined":
            why = (f"No zone covering at least {_pc(share)} of the map is certified at {_pc(alpha)} from these labels "
                   f"(certified: {_pc(cov)}). That is not a no: a new random sample with more labels can certify more, but "
                   f"only where the map's error rate is at most {_pc(alpha)}. " + certificate_second_look(delta, tested))
        why += tail
        out = {"type": "yes_no", "answer": a, "level": 1 - delta, "exact": True, "because": why,
               "evidence": {**ev, "certified": cov}}
        if alt is not None:
            out["also_accept"] = {"answer": "yes" if cov2 >= share else "undetermined", "certified": cov2,
                                  "pairs": alt.get("pairs")}
    if by_cond:
        out["per_condition"] = {}

        def share_of(e):
            got = e["n_zone"] / e["n_population"] if e.get("n_zone") and e.get("n_population") else None
            return got if got is not None else (0.0 if e.get("tested") else None)

        for name, e in r["per_condition"].items():
            out["per_condition"][name] = {"certified_share_of_condition": share_of(e),
                                          "tested": e.get("tested"), "reason": e.get("reason")}
            e2 = ((alt or {}).get("per_condition") or {}).get(name)
            if e2 is not None:
                out["per_condition"][name]["also_accept"] = {"certified_share_of_condition": share_of(e2)}
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
                raise ValueError(_no_labels_which(result))
            raise ValueError(f"{name} cannot be answered from a {k} result; it answers: {', '.join(can)}")
        if name == "error_rate_below":
            ans = _answer_class_audit(result, v) if k == "class_audit" else _answer_error_rate_below(result, v)
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

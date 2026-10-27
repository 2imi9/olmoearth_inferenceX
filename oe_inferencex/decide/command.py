"""The command decide: typed answers (yes, no, undetermined) read from a result another command wrote."""
import json

from oe_inferencex.common.output import _check_out_file, _refuse_overwriting_inputs, _stamp
from oe_inferencex.common.wording import class_wording, result_class_names


def _worded_print(words):
    import builtins
    return lambda text="", **kw: builtins.print(words(text), **kw)


def cmd_decide(args):
    """Typed answers to set questions, read from a result JSON that estimate, certify or compare wrote."""
    from oe_inferencex.decide import decide
    try:
        res = decide(args.result, args.ask or [])
    except ValueError as exc:                       # decide's refusals, unreadable files and non-results alike
        raise SystemExit(str(exc)) from None
    r = args.result
    out = args.out or (r[:-5] + "_decisions.json" if r.endswith(".json") else r + "_decisions.json")
    _refuse_overwriting_inputs("decide", [out], [r])
    _check_out_file(out, "decide")
    # a result of the class design is answered in its own words: decide said "input condition" and "of this condition"
    # of map classes (the bug hunt of 2026-10-25); decide.decide words its own texts so
    say = print
    try:
        with open(r, encoding="utf-8") as f:
            given = json.load(f)
        say = _worded_print(class_wording(given, r, out, names=result_class_names(given, r)))
    except (OSError, ValueError, AttributeError):
        pass
    for q, a in res["answers"].items():
        shown = f"{100 * a['answer']:.1f}%" if a["type"] == "score" else a["answer"]
        say(f"{q}: {shown}. {a['because']}")
        alt = a.get("also_accept")                    # a second reading of the same labels (--also-accept)
        if alt:
            say("  also accepting the file's pairs: " + (
                f"{100 * alt['answer']:.1f}%" if a["type"] == "score" else
                f"{alt['answer']} ({'95%' if alt.get('exact', True) else 'nominal 95%'} interval "
                f"{100 * alt['low']:.1f}% to {100 * alt['high']:.1f}%)" if "low" in alt
                else f"{alt['answer']} (certified: {100 * alt['certified']:.1f}%)"))
        for name, c in (a.get("per_condition") or {}).items():
            c2 = c.get("also_accept")
            if "answer" in c:
                say(f"  {name}: {c['answer']}" + (f"; also accepting: {c2['answer']}" if c2 else ""))
            else:
                share = c.get("certified_share_of_condition")
                said = (f"{100 * share:.1f}% of this condition certified" if share else
                        f"nothing certified ({c.get('reason') or 'no zone passed its test'})")
                if c2:
                    share2 = c2.get("certified_share_of_condition")
                    said += "; also accepting: " + (f"{100 * share2:.1f}% of this condition certified" if share2
                                                    else "nothing certified")
                say(f"  {name}: {said}")
        for c, row in (a.get("per_class") or {}).items():
            iv = (f" (95% interval {100 * row['low']:.1f}% to {100 * row['high']:.1f}%)" if row.get("low") is not None
                  else f" ({row.get('because', 'no interval')})")
            # an audit's second reading of the same labels; a class with nothing more accepted reads the same
            alt = row.get("also_accept")
            say(f"  class {c}: {row['answer']}{iv}" + ("  [warning]" if row.get("warning") else "")
                  + (f"; also accepting {', '.join(str(x) for x in alt['accepted'])}: {alt['answer']} "
                     f"({100 * alt['low']:.1f}% to {100 * alt['high']:.1f}%)" if alt and alt.get("accepted") else ""))
    with open(out, "w") as f:
        json.dump(_stamp(res, "decide"), f, indent=1, default=float)
    print(f"wrote {out}")
    return 0


def add_decide_parser(sub):
    dc = sub.add_parser("decide", help="typed answers (yes / no / undetermined, a / b, a share) to set questions, read "
                                        "from a result JSON that estimate, certify or compare wrote")
    dc.add_argument("result", help="the JSON estimate, certify or compare wrote")
    dc.add_argument("--ask", action="append", metavar="QUESTION[=VALUE]",
                    help="a question, repeatable: error_rate_below=0.1, user_accuracy_above=0.85, "
                         "producer_accuracy_above=0.85, more_accurate, trusted_share, trusted_share_at_least=0.5, "
                         "share_differs. Without --ask, it stops and names the questions the result can answer")
    dc.add_argument("--out", default=None, help="JSON to write (default: <result>_decisions.json)")
    dc.set_defaults(func=cmd_decide)

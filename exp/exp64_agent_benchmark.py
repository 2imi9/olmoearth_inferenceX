#!/usr/bin/env python
"""exp64: does giving an agent this package as tools make its answers better, or only more fluent?

Why. Every number in this repository answers whether the MEASUREMENT is right: does the margin rank errors, on which
references, by how much. None of them answers whether the TOOL helps. Those are different claims, and only the first is
in the ledger. A person handed this package might review no better than one without it, and an agent given these tools
might answer no more correctly than one given a Python sandbox and the same arrays. This experiment measures the second,
because it can be measured reproducibly; the first needs people.

Preregistered in docs/plan/agent_benchmark.md before any run. That page is the specification and this file implements
it; where the implementation had to choose, the choice is recorded in the summary rather than left in the code.

Design. Forty cards, each a directory the agent can read: `scores.npy` (the model's per-window suspicion and decision),
`meta.json` (task, sensor, dates, the review budget, what a second inference is if there is one), optionally
`second.npy`, and `labels.npy`, which the agent never sees and which grades it. Cards come from testbeds this
repository already holds with expert labels, and nothing in a card is derived from a label except the hidden file.

  Three tasks per card. Name the 5% of windows a reviewer should check, graded by error capture against the labels with
  the package's own capture as the ceiling and a random set at 5% as the floor. Say why each named window is suspect,
  graded on whether the stated cue actually holds for that window. And on cards with a second inference, say how much
  the two differ, where the differences sit, and which side to believe, where the third has one correct answer without
  labels, which is to decline, because exp58 showed the more confident side is right on 51 to 70% of windows and
  confidence does not order the set. Declining scores full marks; picking a side scores the share it gets right, so a
  confident wrong pick is penalised.

  Four arms. A has the package's outputs as tools. B has a sandbox with the same arrays and no package. C is a fixed
  template over the package's outputs with no language model, the groundedness ceiling. D has the card's meta.json and
  no rasters, the floor for task accuracy and the control for prior knowledge of these testbeds.

  Two scores. Task accuracy, per card and pooled. And the claims audit: every number and every window reference in an
  answer is matched against that run's tool outputs, so an unsupported number is counted and a window outside the
  card's grid is a fabrication. That is the contract page turned into a metric.

Preregistered (one-sided, from the plan page, restated here unchanged):
  P1  arm A's claims audit exceeds arm B's by at least 0.2 pooled, and on more cards than not (exact sign test p<0.05).
  P2  arm A's review-set capture exceeds arm B's by at least 0.05 pooled and on more cards than not; arm A reaches at
      least 80% of the package's own capture on the median card.
  P3  on the comparison task arm A declines to pick a side on at least 80% of cards; arm B on fewer than half.
  Falsification. P1 fails if the sandbox agent grounds its numbers as well as the tool agent, in which case the
  contract adds nothing to an agent that can compute. P2 fails if the tool agent does not use the review set it is
  given, or the sandbox agent rediscovers confidence ranking on its own. P3 fails if the tool agent picks sides anyway,
  in which case the contract's "this resolves nothing" is not reaching it, and the safety property claimed for the
  comparison half is not real.

Cards, and why these. Twenty come from GEOID-Flood through exp60 and exp62, which committed per-window decisions,
margins, labels and event ids for 4,489 and 544 chips: those need no encoder and they carry the second inference the
comparison task requires, the other sensor of the same period or the same sensor of the other period. Twenty come from
Sen1Floods11 through exp18's probe on the committed tiles, ten from the Bolivia hand labels and ten from the
multi-region test split, which do need an encoder pass.

Serving. The cluster has a CUDA driver and no toolkit, so vLLM cannot build the JIT attention kernels its Blackwell
path wants and this uses transformers with PyTorch SDPA instead: slower per token and entirely sufficient at this
scale, since the runs are short. The model is named in the summary, never inferred from a default.

Outputs: exp/out/exp64_summary.json, exp/out/exp64_cards.csv, exp/out/exp64_answers.jsonl.
Stages: --stage cards (build them), --stage run (the four arms), --stage grade.
--smoke: synthetic cards, a stub model, no GPU, _smoke outputs.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402
from oe_inferencex.signals import boundary_indicator  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
CARDS = "/scratch/qi_zim_neu/olmoearth_inferenceX/exp64_cards"
BUDGET = 0.05
N_PER_SOURCE = 10
SEED = 0


# ----------------------------------------------------------------------------- cards
def _suspicion(margin):
    """Higher = more suspect, the package's own convention."""
    return -np.asarray(margin, dtype=np.float64)


def _card(name, dec, margin, ok, lab, meta, second=None):
    """One card: what the agent may read, and the labels it may not."""
    return {"name": name, "dec": np.asarray(dec), "margin": np.asarray(margin, dtype=np.float32),
            "ok": np.asarray(ok, dtype=bool), "labels": np.asarray(lab), "meta": meta, "second": second}


MIN_ERRORS = 15          # a card with no errors cannot grade a review set; see the note in _qualifying
MAX_ERROR_RATE = 0.30    # above this the card measures a broken model, not whether a review set is well targeted
MIN_VALID = 100


def _qualifying(m, arm, lab, ok):
    """Chips with enough valid windows and enough errors to grade a review set, and their error counts.

    The median GEOID chip has ZERO errors under these heads: 4,482 usable chips and only 777 carry a single one. A card
    with no errors cannot grade a review set at all, and one with two makes capture a coin flip, so cards are drawn only
    from chips with at least MIN_ERRORS. That biases the card set toward the harder chips, which is the right bias for a
    benchmark of finding errors and the wrong one for estimating how common errors are; nothing here estimates the
    latter, and the selection rule and its counts go into the summary so a reader can see the bias rather than infer it.
    Within the qualifying chips one is drawn at RANDOM per event, not the worst, so the set is not the extreme tail."""
    err = ((m[arm].astype(int) != m[lab].astype(int)) & ok).reshape(len(ok), -1).sum(1)
    valid = ok.reshape(len(ok), -1).sum(1)
    rate = err / np.maximum(valid, 1)
    return err, (valid >= MIN_VALID) & (err >= MIN_ERRORS) & (rate <= MAX_ERROR_RATE)


def _one_per_event(qual, ev, rng, exclude):
    """One chip index per event, drawn at random among that event's qualifying chips."""
    by_event = collections.defaultdict(list)
    for i in np.flatnonzero(qual):
        if ev[i] not in exclude:
            by_event[ev[i]].append(i)
    events = sorted(by_event)
    rng.shuffle(events)
    return [(e, int(rng.choice(by_event[e]))) for e in events]


def geoid_cards(rng):
    """Twenty cards from the committed GEOID-Flood artifacts: ten single-inference, ten with a second inference.

    exp60 holds three inferences of identical windows (pre-event optical, pre-event radar, post-event radar) with their
    margins and two labels. The second inference is drawn from the same cell family so the pair is a real comparison
    and not a strawman: the other sensor of the same period, or the same sensor of the other period."""
    m60 = np.load(os.path.join(OUT, "exp60_masks.npz"), allow_pickle=True)
    ok, ev = m60["ok"], m60["event"].astype(str)
    cards, used_events = [], set()

    err_s2, qual_s2 = _qualifying(m60, "A_s2pre", "y_permanent", ok)
    for e, i in _one_per_event(qual_s2, ev, rng, used_events):
        if len(cards) >= N_PER_SOURCE:
            break
        used_events.add(e)
        cards.append(_card(f"geoid_single_{e}", m60["A_s2pre"][i], m60["margin_A_s2pre"][i], ok[i],
                           m60["y_permanent"][i],
                           {"task": "permanent water", "sensor": "Sentinel-2, pre-event composite",
                            "classes": ["not water", "water"], "review_budget": BUDGET,
                            "second_inference": None, "source": "GEOID-Flood via exp60", "event": e,
                            "chip_index": i}))

    # the comparison cards, on events the single cards did not use
    pairs = [("A_s2pre", "A_s1pre", "y_permanent", "the other sensor of the same period"),
             ("A_s1pre", "B_s1post", "y_after", "the same sensor of the other period")]
    for k, (a_, b_, lab, what) in enumerate(pairs):
        err, qual = _qualifying(m60, a_, lab, ok)
        want = N_PER_SOURCE // len(pairs) + (1 if k < N_PER_SOURCE % len(pairs) else 0)
        got = 0
        for e, i in _one_per_event(qual, ev, rng, used_events):
            if got >= want:
                break
            used_events.add(e)
            got += 1
            cards.append(_card(f"geoid_pair_{k}_{e}", m60[a_][i], m60[f"margin_{a_}"][i], ok[i], m60[lab][i],
                               {"task": "water", "sensor": a_, "classes": ["not water", "water"],
                                "review_budget": BUDGET, "second_inference": what,
                                "source": "GEOID-Flood via exp60", "event": e, "chip_index": i},
                               second={"dec": m60[b_][i], "margin": m60[f"margin_{b_}"][i], "name": b_}))
    return cards


def write_cards(cards, root):
    os.makedirs(root, exist_ok=True)
    rows = []
    for c in cards:
        d = os.path.join(root, c["name"])
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "scores.npy"), np.stack([c["margin"], c["dec"].astype(np.float32)]))
        np.save(os.path.join(d, "valid.npy"), c["ok"])
        np.save(os.path.join(d, "labels.npy"), c["labels"])      # the agent never reads this
        if c["second"] is not None:
            np.save(os.path.join(d, "second.npy"),
                    np.stack([c["second"]["margin"], np.asarray(c["second"]["dec"], dtype=np.float32)]))
        meta = dict(c["meta"])
        meta["grid"] = list(np.asarray(c["dec"]).shape)
        meta["n_valid_windows"] = int(c["ok"].sum())
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        err = (np.asarray(c["dec"]).astype(int) != np.asarray(c["labels"]).astype(int)) & c["ok"]
        n_valid, n_err = int(c["ok"].sum()), int(err.sum())
        # A 5% budget on this grid buys k slots, so no ranker can capture more than k / n_errors. Reporting the raw
        # capture without that ceiling makes an easy card look like a good ranker; both arms face the same ceiling on
        # the same card, but the record should carry it.
        k = max(1, int(round(BUDGET * n_valid)))
        ceiling = min(1.0, k / max(n_err, 1))
        cap = package_capture(c)
        rows.append({"card": c["name"], "source": meta["source"], "grid": "x".join(map(str, meta["grid"])),
                     "n_valid": n_valid, "n_errors": n_err, "review_slots_at_budget": k,
                     "error_rate": float(n_err / max(n_valid, 1)),
                     "has_second": c["second"] is not None,
                     "attainable_capture_ceiling": float(ceiling),
                     "package_capture_at_budget": cap,
                     "package_share_of_ceiling": float(cap / ceiling) if np.isfinite(cap) and ceiling > 0 else float("nan")})
    return rows


def package_capture(c):
    """The ceiling for task 1: what the package's own review set captures on this card."""
    ok = c["ok"].ravel()
    if ok.sum() == 0:
        return float("nan")
    sus = _suspicion(c["margin"]).ravel()[ok]
    err = ((np.asarray(c["dec"]).astype(int) != np.asarray(c["labels"]).astype(int)).ravel()[ok]).astype(float)
    if err.sum() == 0:
        return float("nan")
    return float(metrics.capture_at_budget_expected(sus, err, (BUDGET,))[BUDGET])


def cmd_cards(args):
    rng = np.random.default_rng(SEED)
    cards = geoid_cards(rng)
    root = os.path.join(CARDS, "smoke" if args.smoke else "v1")
    rows = write_cards(cards, root)
    tag = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"exp64_cards{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    gradeable = [r for r in rows if np.isfinite(r["package_capture_at_budget"])]
    print(f"wrote {len(rows)} cards to {root}")
    print(f"  with a second inference : {sum(r['has_second'] for r in rows)}")
    print(f"  gradeable (have errors) : {len(gradeable)}")
    print(f"  median error rate       : {np.median([r['error_rate'] for r in rows]):.4f}")
    print(f"  median review slots     : {int(np.median([r['review_slots_at_budget'] for r in rows]))}")
    print(f"  median package capture  : {np.median([r['package_capture_at_budget'] for r in gradeable]):.4f}"
          f"  (a random {BUDGET:.0%} captures {BUDGET:.2f})")
    print(f"  median share of ceiling : {np.median([r['package_share_of_ceiling'] for r in gradeable]):.4f}"
          f"  (1.0 = the best any ranker could do at this budget)")
    return rows


def smoke(args):
    """Synthetic cards end to end: the selection rule, the ceiling arithmetic, and what a card exposes."""
    import tempfile
    rng = np.random.default_rng(0)
    n, g = 40, 14
    ok = np.ones((n, g, g), bool)
    lab = (rng.random((n, g, g)) < 0.3)
    dec = lab.copy()
    for i in range(n):                                   # plant a controlled number of errors per chip
        flip = rng.choice(g * g, size=i, replace=False)
        dec[i].ravel()[flip] = ~dec[i].ravel()[flip]
    m = {"ok": ok, "A_s2pre": dec, "y_permanent": lab,
         "margin_A_s2pre": rng.random((n, g, g)).astype(np.float32)}
    err, qual = _qualifying(m, "A_s2pre", "y_permanent", ok)
    assert not qual[:MIN_ERRORS].any(), "chips below the error floor must not qualify"
    rate = err / (g * g)
    assert not qual[rate > MAX_ERROR_RATE].any(), "chips above the error ceiling must not qualify"
    assert qual[(err >= MIN_ERRORS) & (rate <= MAX_ERROR_RATE)].all()

    ev = np.array([f"E{i // 3}" for i in range(n)])
    picked = _one_per_event(qual, ev, np.random.default_rng(1), exclude={"E0"})
    assert all(e != "E0" for e, _ in picked), "an excluded event must not be drawn"
    assert len({e for e, _ in picked}) == len(picked), "one chip per event"
    assert all(qual[i] for _, i in picked), "only qualifying chips may be drawn"

    # the ceiling: with k slots and E errors no ranker can capture more than k/E
    c = _card("t", dec[20], m["margin_A_s2pre"][20], ok[20], lab[20], {"source": "smoke", "review_budget": BUDGET})
    root = tempfile.mkdtemp()
    rows = write_cards([c], root)
    r = rows[0]
    assert r["review_slots_at_budget"] == max(1, round(BUDGET * r["n_valid"]))
    assert abs(r["attainable_capture_ceiling"] - min(1.0, r["review_slots_at_budget"] / r["n_errors"])) < 1e-12
    assert r["package_capture_at_budget"] <= r["attainable_capture_ceiling"] + 1e-12, "capture cannot beat the ceiling"
    assert sorted(os.listdir(os.path.join(root, "t"))) == ["labels.npy", "meta.json", "scores.npy", "valid.npy"]
    print("smoke OK: error floor and ceiling, one chip per event, exclusions, capture bounded by the attainable ceiling")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("cards", "run", "grade", "all"), default="cards")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    if args.stage in ("cards", "all"):
        cmd_cards(args)


if __name__ == "__main__":
    main()

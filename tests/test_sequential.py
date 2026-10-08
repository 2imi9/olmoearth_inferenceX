"""Sequential certify (oe_inferencex/sequential.py), checked in exact arithmetic rather than by simulation.

The e-process's closed form is compared with the mixture it stands for; its supermartingale property is checked
under every null count, not only the boundary one; and Ville's bound, the whole guarantee, is computed exactly by
dynamic programming over every path of labels on small zones."""
import math
from fractions import Fraction

import numpy as np
import pytest

from oe_inferencex import estimate as est
from oe_inferencex import sequential as sq


def _mixture(n, M0, t, k):
    """The uniform mixture of likelihood ratios, summed term by term."""
    if k >= M0:
        return Fraction(0)                    # as many wrong labels as the null count: the null holds, e is 0
    den = math.comb(M0, k) * math.comb(n - M0, t - k)
    if den == 0:
        return math.inf
    return Fraction(sum(math.comb(M, k) * math.comb(n - M, t - k) for M in range(M0)), M0 * den)


@pytest.mark.parametrize("n", [1, 2, 5, 9, 16, 23])
def test_the_closed_form_is_the_mixture(n):
    for alpha in (0.05, 0.1, 0.2, 0.34, 0.5, 0.9):
        M0 = math.floor(alpha * n) + 1
        for t in range(n + 1):
            for k in range(t + 1):
                e = sq.evalue_exact(t, k, n, alpha)
                if M0 > n:
                    assert e == math.inf                     # no count of n exceeds alpha n: nothing to reject
                    continue
                assert e == _mixture(n, M0, t, k), (n, alpha, t, k)


@pytest.mark.parametrize("n", [3, 8, 15, 30])
def test_the_float_log_matches_the_integers(n):
    for alpha in (0.05, 0.1, 0.25, 0.5):
        for t in range(n + 1):
            for k in range(t + 1):
                le, e = sq.log_evalue(t, k, n, alpha), sq.evalue_exact(t, k, n, alpha)
                if e == math.inf:
                    assert le == math.inf
                elif e == 0:
                    assert le == -math.inf
                else:
                    assert abs(le - math.log(e)) < 1e-11, (n, alpha, t, k)


@pytest.mark.parametrize("n,alpha", [(12, 0.1), (12, 0.25), (20, 0.15), (25, 0.3)])
def test_e_is_a_supermartingale_under_every_null_count(n, alpha):
    """For every count M >= M0 and every state (t, k) the next label can reach from it, the expected next e is at
    most the present one. At M0 it is equal wherever both outcomes are possible under M0 (a martingale); where the
    remaining windows of a zone with M0 wrong are all wrong or all right, the alternatives keep mass M0 has not, and
    the expectation falls."""
    M0 = math.floor(alpha * n) + 1
    for M in range(M0, n + 1):
        for t in range(n):
            for k in range(max(0, t - (n - M)), min(t, M) + 1):     # states with positive probability under M
                now = sq.evalue_exact(t, k, n, alpha)
                if now == math.inf:
                    continue                                       # unreachable under M >= M0
                p_wrong = Fraction(M - k, n - t)
                nxt = Fraction(0)
                if p_wrong:
                    nxt += p_wrong * sq.evalue_exact(t + 1, k + 1, n, alpha)
                if p_wrong < 1:
                    e0 = sq.evalue_exact(t + 1, k, n, alpha)
                    assert e0 != math.inf
                    nxt += (1 - p_wrong) * e0
                assert nxt <= now, (M, t, k)
                if M == M0 and 0 < p_wrong < 1:
                    assert nxt == now


@pytest.mark.parametrize("n", [10, 18, 30])
def test_villes_bound_holds_exactly_on_every_path(n):
    """P(e ever reaches 1/delta) <= delta under every null count, computed by dynamic programming over (t, k) in
    rational arithmetic: the probability of every path that has not yet reached the level is carried forward."""
    for alpha in (0.1, 0.2, 0.3):
        M0 = math.floor(alpha * n) + 1
        if M0 > n:
            continue
        for delta in (Fraction(1, 10), Fraction(1, 4), Fraction(1, 2)):
            for M in range(M0, n + 1):
                alive, crossed = {0: Fraction(1)}, Fraction(0)          # k -> probability, at label count t
                for t in range(n):
                    nxt = {}
                    for k, p in alive.items():
                        for x, q in ((1, Fraction(M - k, n - t)), (0, Fraction(n - t - (M - k), n - t))):
                            if q == 0:
                                continue
                            e = sq.evalue_exact(t + 1, k + x, n, alpha)
                            if e != math.inf and e * delta < 1:
                                nxt[k + x] = nxt.get(k + x, 0) + p * q
                            else:
                                crossed += p * q
                    alive = nxt
                assert crossed <= delta, (n, alpha, float(delta), M, float(crossed))


def test_reaches_decides_ties_in_integers():
    """A float that lands within the tie window of 1/delta is decided by the exact value."""
    hits = 0
    for n in range(2, 40):
        for alpha in (0.1, 0.2, 0.25, 0.5):
            for t in range(n + 1):
                for k in range(t + 1):
                    e = sq.evalue_exact(t, k, n, alpha)
                    for delta in (Fraction(1, 2), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10)):
                        want = e == math.inf or e * delta >= 1
                        assert sq.reaches(t, k, n, alpha, float(delta)) == want, (n, alpha, t, k, delta)
                        hits += e != math.inf and e * delta == 1
    assert hits > 0                                                  # some e equals a level exactly


def test_min_labels_sequential():
    for alpha, b in ((0.05, 71), (0.02, 179), (0.01, 360)):
        assert sq.min_labels_sequential(alpha, 0.1) == b
        assert b > est.min_labels_to_certify(alpha, 0.1)
    # on a large zone, b labels with none wrong reach 1/delta and b - 1 do not (the limit is approached from above)
    for alpha in (0.05, 0.02):
        b = sq.min_labels_sequential(alpha, 0.1)
        n = 10_000_000
        assert sq.reaches(b, 0, n, alpha, 0.1) and not sq.reaches(b - 1, 0, n, alpha, 0.1)


def test_level_path_finds_the_running_maximum_and_the_first_crossing():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(20, 300))
        alpha = float(rng.choice([0.05, 0.1, 0.2]))
        pop = (rng.random(n) < rng.uniform(0, 2 * alpha)).astype(int)
        T = int(rng.integers(1, n + 1))
        w = pop[rng.permutation(n)][:T]
        draw = np.sort(rng.choice(5 * n, T, replace=False)) + 1
        r = sq.level_path(draw, w, n, alpha, 0.1)
        k = np.concatenate([[0], np.cumsum(w)])
        les = [sq.log_evalue(t, int(k[t]), n, alpha) for t in range(T + 1)]
        assert r["log_e_max"] == max(les)
        hit = [t for t in range(1, T + 1) if sq.reaches(t, int(k[t]), n, alpha, 0.1)]
        assert r["passed_at"] == (int(draw[hit[0] - 1]) if hit else None)
        assert r["passed"] == bool(hit)


# ----------------------------------------------------------------------------- the command line
import csv as _csv
import json as _json

from oe_inferencex.cli import main as _main


def _map(tmp_path, seed=0, size=96):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(6, 1.0, (size, size)), 0, 1).astype(np.float32)
    path = tmp_path / "p.npy"
    np.save(path, p)
    return str(path)


def _rows(path):
    return list(_csv.DictReader(open(path)))


def _label(path, n, rng, rate=0.15):
    rows = _rows(path)
    for r in rows[:n]:
        if r["wrong"] == "":
            r["wrong"] = str(int(rng.random() < rate * (1 - float(r["confidence"])) * 4))
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def test_sample_sequential_writes_the_start_of_a_fixed_order(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    assert _main(["sample", p, "--budget", "120", "--design", "sequential", "--alpha", "0.1", "--seed", "3", "--out", str(out)]) == 0
    side = _json.load(open(tmp_path / "s.json"))
    assert side["design"] == "sequential" and side["first_budget"] == 120
    assert side["anchor"] == sq.anchor_coverage(120, 0.1, est.ZONE_DELTA) and side["anchor_rule"]["alpha"] == 0.1
    pop = np.arange(side["n_population"])                     # every window is valid on this map
    assert [int(r["index"]) for r in _rows(out)] == pop[sq.sequential_order(pop.size, 3)[:120]].tolist()


def test_certify_reads_the_labelled_rows_from_the_top_and_matches_the_api(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    _main(["sample", p, "--budget", "400", "--design", "sequential", "--alpha", "0.1", "--seed", "1", "--out", str(out)])
    rows = _label(out, 300, np.random.default_rng(0))
    assert _main(["certify", str(out), "--alpha", "0.1"]) == 0
    res = _json.load(open(tmp_path / "s_zone.json"))
    assert res["rule"] == "sequential" and res["n_labelled"] == 300 and res["n_drawn"] == 400
    from oe_inferencex.assess import assess_prediction
    margin = assess_prediction(np.load(p), is_logit=False, patch=4)["arrays"]["confidence"].ravel()
    idx = np.array([int(r["index"]) for r in rows[:300]])
    wrong = np.array([int(r["wrong"]) for r in rows[:300]])
    api = sq.certify_zone_sequential(margin, idx, wrong, 0.1, anchor=_json.load(open(tmp_path / "s.json"))["anchor"])
    assert res["coverage"] == api["coverage"] and res["anchor"] == api["anchor"]
    assert [lv["passed_at_label"] for lv in res["levels"]] == [lv["passed_at_label"] for lv in api["levels"]]


def test_a_label_below_a_blank_row_is_refused(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    _main(["sample", p, "--budget", "60", "--design", "sequential", "--alpha", "0.1", "--out", str(out)])
    rows = _rows(out)
    rows[10]["wrong"] = "0"                                   # row 10 labelled, rows 0-9 not
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="labelled below row 2"):
        _main(["certify", str(out), "--alpha", "0.1"])
    rows[10]["wrong"] = ""
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit, match="no window is labelled yet"):
        _main(["certify", str(out), "--alpha", "0.1"])


def test_extend_keeps_the_labels_and_continues_the_order(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    _main(["sample", p, "--budget", "100", "--design", "sequential", "--seed", "7", "--anchor", "0.3", "--out", str(out)])
    before = _label(out, 100, np.random.default_rng(1))
    assert _main(["sample", p, "--budget", "250", "--design", "sequential", "--seed", "7", "--extend", str(out),
                  "--out", str(out)]) == 0
    after = _rows(out)
    side = _json.load(open(tmp_path / "s.json"))
    assert len(after) == 250 and [r["wrong"] for r in after[:100]] == [r["wrong"] for r in before]
    assert all(r["wrong"] == "" for r in after[100:])
    assert side["first_budget"] == 100 and side["anchor"] == 0.3
    pop = np.arange(side["n_population"])
    assert [int(r["index"]) for r in after] == pop[sq.sequential_order(pop.size, 7)[:250]].tolist()
    for bad, msg in ((["--seed", "8"], "same seed"), (["--seed", "7", "--anchor", "0.5"], "anchor was fixed"),
                     (["--seed", "7", "--alpha", "0.05"], "anchor was fixed"),
                     (["--seed", "7", "--budget", "200"], "already holds")):
        args = ["sample", p, "--budget", "300", "--design", "sequential", "--extend", str(out), "--out",
                str(tmp_path / "t.csv")] + bad
        with pytest.raises(SystemExit, match=msg):
            _main(args)


def test_sequential_refusals(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    with pytest.raises(SystemExit, match="belong to --design sequential"):
        _main(["sample", p, "--budget", "50", "--design", "random", "--anchor", "0.2", "--out", str(out)])
    with pytest.raises(SystemExit, match="fixes now, before any label"):
        _main(["sample", p, "--budget", "50", "--design", "sequential", "--out", str(out)])
    with pytest.raises(SystemExit, match="not both"):
        _main(["sample", p, "--budget", "50", "--design", "sequential", "--anchor", "0.2", "--alpha", "0.1",
               "--out", str(out)])
    with pytest.raises(SystemExit, match="share of the map"):
        _main(["sample", p, "--budget", "50", "--design", "sequential", "--anchor", "1.5", "--out", str(out)])
    _main(["sample", p, "--budget", "80", "--design", "sequential", "--alpha", "0.1", "--out", str(out)])
    _label(out, 80, np.random.default_rng(2))
    for extra in (["--rule", "bonferroni"], ["--level-cut", "ramp"]):
        with pytest.raises(SystemExit, match="sequential rule"):
            _main(["certify", str(out), "--alpha", "0.1"] + extra)
    r = tmp_path / "r.csv"
    _main(["sample", p, "--budget", "80", "--design", "random", "--out", str(r)])
    with pytest.raises(SystemExit, match="only a sequential sample can be extended"):
        _main(["sample", p, "--budget", "100", "--design", "sequential", "--extend", str(r), "--out", str(out)])


def test_estimate_reads_the_labelled_prefix_as_a_random_sample(tmp_path):
    p = _map(tmp_path)
    out = tmp_path / "s.csv"
    _main(["sample", p, "--budget", "200", "--design", "sequential", "--alpha", "0.1", "--out", str(out)])
    rows = _label(out, 150, np.random.default_rng(3))
    assert _main(["estimate", str(out)]) == 0
    res = _json.load(open(tmp_path / "s_estimate.json"))
    k = sum(int(r["wrong"]) for r in rows[:150])
    assert res["n_labelled"] == 150 and abs(res["estimate"] - k / 150) < 1e-12
    lo, hi = est.hypergeom_interval(k, 150, res["n_population"])
    assert (res["low"], res["high"]) == (lo, hi) and "sequential_note" in res


def test_the_log_e_value_does_not_underflow_where_the_tail_is_tiny():
    """Far from the null count the hypergeometric tail is far below the smallest float, while e is not: the log is
    summed without leaving log space (review of 8 October 2026: it read -inf, e 0, where e was exp(-9))."""
    n, alpha, t = 5000, 0.1, 500
    for k in range(0, t + 1, 20):
        e = sq.evalue_exact(t, k, n, alpha)
        le = sq.log_evalue(t, k, n, alpha)
        if e == 0:
            assert le == -math.inf
        else:
            exact = math.log(e.numerator) - math.log(e.denominator)
            assert abs(le - exact) < 1e-9 * max(1.0, abs(exact)), (k, le, exact)


def test_an_infinite_e_value_is_written_as_known_not_as_infinity():
    import json
    margin = np.linspace(1, 0, 20)
    wrong = np.zeros(20)
    r = sq.certify_zone_sequential(margin, np.arange(20), wrong, 0.2, anchor=0.5, grid=(0.5, 1.0))
    json.dumps(r["levels"], allow_nan=False)                   # every level is plain JSON
    assert r["levels"][0]["log_e_value"] is None and r["levels"][0]["known"].startswith("good")
    wrong[:12] = 1
    r = sq.certify_zone_sequential(margin, np.arange(20), wrong, 0.2, anchor=0.5, grid=(0.5, 1.0))
    json.dumps(r["levels"], allow_nan=False)
    assert r["levels"][0]["known"].startswith("wrong more than alpha") and r["coverage"] is None


def test_a_verified_order_skips_the_enrichment_check_and_a_wrong_one_is_refused():
    """A prefix of the seeded order is a random draw by construction, so the check that refuses enriched sets, which
    refuses about 5% of genuine orders at some look by chance, is skipped; labels that are not that prefix are refused."""
    N = 4000
    margin = np.linspace(1, 0, N)
    rng = np.random.default_rng(0)
    hit = None
    for seed in range(400):                                  # an order whose first label sits at the suspect end
        o = sq.sequential_order(N, seed)
        if est.review_set_check(o[:1], margin)["looks_like_a_review_set"]:
            hit = seed
            break
    assert hit is not None
    o = sq.sequential_order(N, hit)
    with pytest.raises(ValueError, match="enriched set"):
        sq.certify_zone_sequential(margin, o[:1], np.zeros(1), 0.1, anchor=0.5)
    r = sq.certify_zone_sequential(margin, o[:1], np.zeros(1), 0.1, anchor=0.5, seed=hit)
    assert r["order_verified"] and r["coverage"] is None
    with pytest.raises(ValueError, match="not the start of the order"):
        sq.certify_zone_sequential(margin, o[1:40], np.zeros(39), 0.1, anchor=0.5, seed=hit)


def test_certified_is_the_prefix_decision_and_passed_is_not():
    """A level past the first unpassed one can pass on its own; it is not certified."""
    N = 20000
    margin = np.linspace(1, 0, N)
    wrong_o = np.zeros(N)
    wrong_o[:200] = 1                                        # the most confident 1% wrong: the 5% zone fails
    o = sq.sequential_order(N, 1)[:2500]
    r = sq.certify_zone_sequential(margin, o, wrong_o[o], 0.05, anchor=0.05, seed=1)
    assert r["coverage"] is None and not any(lv["certified"] for lv in r["levels"])
    assert not r["levels"][0]["passed"] and any(lv["passed"] for lv in r["levels"][1:])


def test_the_exact_e_value_sums_the_shorter_side():
    """Both sides give the same count; at large t the short one is fast enough for a tie decision."""
    import time
    t0 = time.time()
    e = sq.evalue_exact(10_000, 420, 1_000_000, 0.05)
    assert time.time() - t0 < 30
    assert abs(math.log(e.numerator) - math.log(e.denominator) - sq.log_evalue(10_000, 420, 1_000_000, 0.05)) < 1e-8


def test_the_whole_procedure_holds_delta_exactly_on_every_order_of_small_maps():
    """The anchored fixed sequence with sticky passes, enumerated: on maps of 6 windows with a grid of thirds, for
    every error pattern, every anchor and every order of the windows, the share of orders on which some look
    certifies a zone sequence holding a zone wrong more than alpha is at most delta, and reaches it on some pattern
    (so the enumeration is tight, not vacuous). Each zone's pass time comes from level_path, the package's own; a
    spot check against certify_zone_sequential at every look ties the enumeration to the public function."""
    import itertools
    from functools import lru_cache
    N = 6
    grid = (round(1 / 3, 4), round(2 / 3, 4), 1.0)
    sizes = [max(1, int(round(c * N))) for c in grid]
    perms = list(itertools.permutations(range(N)))
    margin = np.linspace(1, 0, N)
    rng = np.random.default_rng(0)
    for alpha, delta in ((0.1, 0.5), (0.2, 0.25)):
        @lru_cache(maxsize=None)
        def passed_t(size, w):                               # in-zone label count at which the zone passes, or None
            r = sq.level_path(np.arange(1, len(w) + 1), np.array(w, int), size, alpha, delta)
            return r["passed_at"]
        tight = False
        for bits in range(2 ** N):
            err = np.array([(bits >> i) & 1 for i in range(N)])
            bad = [err[:s].sum() / s > alpha for s in sizes]
            for j0 in range(len(grid)):
                wrong_orders = 0
                for p in perms:
                    p = np.array(p)
                    cross = []
                    for s in sizes:
                        inz = np.flatnonzero(p < s)
                        t = passed_t(s, tuple(int(x) for x in err[p[inz]]))
                        cross.append(inz[t - 1] + 1 if t else math.inf)    # the look at which it passed
                    best, ok = None, True
                    for j in range(j0, len(grid)):
                        ok = ok and cross[j] <= N
                        if ok:
                            best = j
                    wrong_orders += best is not None and any(bad[j0:best + 1])
                    if rng.random() < 0.002:                      # the public function agrees at every look
                        for L in range(1, N + 1):
                            z = sq.certify_zone_sequential(margin, p[:L], err[p[:L]], alpha, delta, anchor=grid[j0],
                                                           grid=grid)
                            pref, want = True, None
                            for j in range(j0, len(grid)):
                                pref = pref and cross[j] <= L
                                if pref:
                                    want = grid[j]
                            assert z["coverage"] == want
                share = Fraction(wrong_orders, len(perms))
                assert share <= Fraction(repr(delta)), (alpha, delta, bits, j0, share)
                tight = tight or share == Fraction(repr(delta))
        assert tight, (alpha, delta)

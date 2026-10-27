"""grade --ratio direct (agency/stratified.py, ratio_interval): one betting test inverted over a user's or producer's
accuracy, monotone in r inside fixed bins. Exact checks by enumeration at sizes a test can run, as in test_agency.py:
the batched programme against every order and against the linear one at the anchor, the e-value at the truth,
monotonicity in each bin, coverage of every population (with unknown values, an unsampled stratum and a point-frame
reading), the scan order, the box beside it, and the command line. The widths and times on NLCD and GFC2020 are
measured on aicr (job J3), and Olofsson's direct numbers are recorded from there as goldens, not asserted here: that
the direct interval is narrower than the box follows from no proof."""
import itertools
import json
import math

import numpy as np
import pytest

from oe_inferencex import agency
from oe_inferencex import stratified as st
from oe_inferencex.cli import main
from oe_inferencex.stratified import (ANCHORS, EDGES, KAPPA, METHOD_BOX, METHOD_RATIO, RATIO_BINS, Design, _ratio_bin,
                                      _ratio_logM, _ratio_low_end, _stratum_logavg, _stratum_logavg_ratio, c_grid,
                                      ratio, ratio_interval)


def log_choose(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compositions(total, parts=3):
    if parts == 1:
        return [(total,)]
    return [(i,) + rest for i in range(total + 1) for rest in compositions(total - i, parts - 1)]


def design_of(ns, Ns, W):
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    return Design(hidx, W, [float(N) for N in Ns])


def values(design, per_stratum):
    """x per point from each stratum's list of values (the order within a stratum does not matter)."""
    x = np.zeros(design.hidx.size)
    for h, v in enumerate(per_stratum):
        x[design.masks[h]] = v
    return x


def three_kinds(d, ss):
    """y and w per point from each stratum's counts of units that count (y = 1), units in the denominator only
    (w = 1) and units outside it."""
    y = values(d, [[1.0] * s[0] + [0.0] * (s[1] + s[2]) for s in ss])
    w = values(d, [[0.0] * s[0] + [1.0] * s[1] + [0.0] * s[2] for s in ss])
    return y, w


def worst_coverage(Ns, ns, lo, hi, truth, kinds=3):
    """The least, over every population (each stratum's units split into kinds) and every value truth(pops) gives
    for it, of the probability that the interval of its sample covers that value. lo, hi: each sample's interval, the
    samples in the order of itertools.product over each stratum's compositions; draws without replacement."""
    tables = []
    for N, n in zip(Ns, ns):
        pops, samples = compositions(N, kinds), compositions(n, kinds)
        lp = np.array([[sum(log_choose(P[k], s[k]) for k in range(kinds)) - log_choose(N, n) for s in samples]
                       for P in pops])
        tables.append((pops, np.exp(lp)))
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    worst = 1.0
    for choice in itertools.product(*[range(len(t[0])) for t in tables]):
        p = np.ones(1)
        for (pops, probs), i in zip(tables, choice):
            p = np.multiply.outer(p, probs[i]).ravel()
        for R in truth(tuple(t[0][i] for t, i in zip(tables, choice))):
            worst = min(worst, float(p[(lo - 1e-12 <= R) & (R <= hi + 1e-12)].sum()))
    return worst


def ratio_of(W, Ns, parts, extra=(0.0, 0.0)):
    """R from each stratum's (units that count, units in the denominator that do not); None when no unit is in it."""
    ty = sum(W[h] * parts[h][0] / Ns[h] for h in range(len(Ns))) + extra[0]
    tz = sum(W[h] * (parts[h][0] + parts[h][1]) / Ns[h] for h in range(len(Ns))) + extra[1]
    return ty / tz if tz > 0 else None


# ----------------------------------------------------------------------------------------------- the programme
def orders(counts):
    return set(itertools.permutations([j for j, k in enumerate(counts) for _ in range(k)]))


def brute_ratio(seqs, r, rho, top, lam, N):
    """One side's averaged e-value of a stratum by listing every distinct order of its labels (0: value 1, 1: value
    0, 2: value r), the centre r plus the bin's shift."""
    finite = math.isfinite(N)
    tot = []
    for seq in seqs:
        P = Q = C = 0
        lg = 0.0
        for t, j in enumerate(seq):
            lt = lam * ((N - t) / N if finite else 1.0)
            c = r + min((0.5 - rho + P - rho * (P + Q)) / (t + 1), KAPPA / lt - top)
            tail = (P + r * C) / (N - t) if finite else 0.0
            lg += math.log1p(lt * ((1.0, 0.0, r)[j] - c)) + lt * (c + tail)
            P, Q, C = P + (j == 0), Q + (j == 1), C + (j == 2)
        tot.append(lg)
    m = max(tot)
    return m + math.log(sum(math.exp(v - m) for v in tot)) - math.log(len(tot))


def test_ratio_programme_matches_every_order():
    """The batched programme at nine points of three bins at once (each bin's ends and middle, under its anchor),
    three lambdas, with and without replacement, against every order listed."""
    pts = [(r, ANCHORS[j], EDGES[j + 1]) for j in (0, 16, RATIO_BINS - 1)
           for r in (EDGES[j], 0.5 * (EDGES[j] + EDGES[j + 1]), EDGES[j + 1])]
    r, rho, top = (np.array(v, float) for v in zip(*pts))
    lams = np.array([0.3, 1.0, 3.0])
    worst = 0.0
    for n in range(1, 7):
        for smp in compositions(n):
            seqs = orders(smp)
            for N in (n, n + 2, math.inf):
                dp = _stratum_logavg_ratio(smp, r, rho, top, lams, N)
                assert dp.shape == (len(pts), lams.size)
                for b, (rb, rhob, topb) in enumerate(pts):
                    for k, lam in enumerate(lams):
                        worst = max(worst, abs(float(dp[b, k]) - brute_ratio(seqs, rb, rhob, topb, lam, N)))
    assert worst < 1e-9


def test_ratio_programme_equals_the_linear_one_at_the_anchor():
    """At r = rho_j with the clip inactive (lambda <= 0.3) the centre is the running mean of the values 1, 0 and
    rho_j, so the new programme is the old one, left as it was, on those values: the tie of the new code to the old."""
    lams = np.array([0.05, 0.1, 0.3])
    worst = 0.0
    for n in (1, 2, 5, 9):
        for smp in compositions(n):
            for N in (n, n + 3, math.inf):
                for j in (0, 7, 16, RATIO_BINS - 1):
                    new = _stratum_logavg_ratio(smp, [ANCHORS[j]], ANCHORS[j], EDGES[j + 1], lams, N)[0]
                    old = _stratum_logavg(list(smp), [1.0, 0.0, float(ANCHORS[j])], lams, N)
                    worst = max(worst, float(np.max(np.abs(new - old))))
    assert worst < 1e-12


def test_ratio_evalue_at_the_truth_is_at_most_one():
    """E[e-value] over every sample of every population of one stratum (N up to 6 units of values 1, 0 and r): at r =
    the population's R under its own bin's test, at that bin's right end under the same test (the scan's), and in the
    next bin; each has R <= r, so each is at most 1, drawn without replacement and for the point-frame reading
    deflated by lambda n (n - 1) / (2 (N - n + 1))."""
    lams = np.array([0.2, 0.6, 1.0, 2.0, 5.0])
    worst_wor = worst_defl = 0.0
    for N in (2, 3, 4, 6):
        pops = compositions(N)
        for n in range(1, N + 1):
            pts = []
            for i, M in enumerate(pops):
                R = M[0] / (M[0] + M[1]) if M[0] + M[1] else 0.5      # no unit in the denominator: u = r, any r
                j = _ratio_bin(R)
                pts += [(i, R, ANCHORS[j], EDGES[j + 1]), (i, EDGES[j + 1], ANCHORS[j], EDGES[j + 1])]
                if j + 1 < RATIO_BINS:
                    pts.append((i, 0.5 * (EDGES[j + 1] + EDGES[j + 2]), ANCHORS[j + 1], EDGES[j + 2]))
            idx = [p[0] for p in pts]
            r, rho, top = (np.array([p[k] for p in pts], float) for k in (1, 2, 3))
            D = lams * n * (n - 1) / (2.0 * (N - n + 1))
            e_wor = np.zeros((len(pts), lams.size))
            e_defl = np.zeros_like(e_wor)
            for s in compositions(n):
                lp = np.array([sum(log_choose(pops[i][k], s[k]) for k in range(3)) - log_choose(N, n) for i in idx])
                ok = lp > -np.inf
                base = lp[ok][:, None] - lams[None, :] * n * r[ok][:, None]
                wor = _stratum_logavg_ratio(s, r, rho, top, lams, N)
                wr = _stratum_logavg_ratio(s, r, rho, top, lams, math.inf)
                e_wor[ok] += np.exp(base + wor[ok])
                e_defl[ok] += np.exp(base + wr[ok] - D[None, :])
            worst_wor, worst_defl = max(worst_wor, float(e_wor.max())), max(worst_defl, float(e_defl.max()))
    assert worst_wor <= 1 + 1e-9
    assert worst_defl <= 1 + 1e-9


@pytest.mark.parametrize("Ns, ns, W, Wu, point", [((4, 4), (2, 2), (0.7, 0.3), 0.0, False),
                                                  ((5, 3), (3, 2), (0.5, 0.5), 0.0, False),
                                                  ((5, 3), (3, 2), (0.4, 0.4), 0.2, False),
                                                  ((5, 3), (3, 2), (0.5, 0.5), 0.0, True)])
def test_ratio_test_is_monotone_in_each_bin(Ns, ns, W, Wu, point):
    """log M at 50 points of each of five bins, under the bin's anchor, for every sample: nonincreasing to the
    tolerance the search allows (RATIO_ORDER_TOL). Every sample's counts with value 1 and value 0 swapped are a sample
    too, so the upper end's side is checked as well."""
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    if point:
        d = Design(hidx, W, [math.inf] * len(Ns), Nlo=[float(N) for N in Ns])
    else:
        d = Design(hidx, W, [float(N) for N in Ns], W_unsampled=Wu)
    cs = c_grid(d.W, d.n, 0.1)
    worst = 0.0
    for ss in itertools.product(*[compositions(n) for n in ns]):
        for j in (0, 3, 16, 29, RATIO_BINS - 1):
            f = _ratio_logM(d, list(ss), np.linspace(EDGES[j], EDGES[j + 1], 50), ANCHORS[j], EDGES[j + 1], cs)
            worst = max(worst, float(np.diff(f).max()))
    assert worst <= st.RATIO_ORDER_TOL


# ----------------------------------------------------------------------------------------------- coverage, exactly
@pytest.mark.parametrize("Ns, ns, W", [((4, 4), (2, 2), (0.7, 0.3)), ((5, 3), (3, 2), (0.5, 0.5)),
                                       ((5, 4, 3), (2, 2, 1), (0.6, 0.3, 0.1))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_direct_ratio_covers_every_population(Ns, ns, W, alpha):
    """Units of three kinds: in the numerator (y = 1), in the denominator only (w = 1), outside it."""
    d = design_of(ns, Ns, W)
    lo, hi = [], []
    for ss in itertools.product(*[compositions(n) for n in ns]):
        q = ratio(d, *three_kinds(d, ss), alpha=alpha, method="direct")
        assert q["method"] == METHOD_RATIO and "box" in q
        lo.append(q["interval"][0])
        hi.append(q["interval"][1])
    assert max(lo) > 0 and min(hi) < 1                          # it says something: [0, 1] everywhere would pass
    truth = lambda pops: [R for R in [ratio_of(W, Ns, pops)] if R is not None]
    assert worst_coverage(Ns, ns, lo, hi, truth) >= 1 - alpha - 1e-12


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_direct_ratio_with_an_unsampled_stratum_covers_every_population(alpha):
    """Two sampled strata and a third of 3 units with no label, whose composition is anything: the interval must cover
    the ratio over all three (theta_0 = r holds whatever that stratum holds)."""
    Ns, ns, Nu = (4, 3), (2, 2), 3
    W = [N / (sum(Ns) + Nu) for N in Ns]
    Wu = Nu / (sum(Ns) + Nu)
    d = Design(np.concatenate([np.full(n, h) for h, n in enumerate(ns)]), W, [float(N) for N in Ns], W_unsampled=Wu)
    lo, hi = [], []
    for ss in itertools.product(*[compositions(n) for n in ns]):
        a, b = ratio(d, *three_kinds(d, ss), alpha=alpha, method="direct")["interval"]
        lo.append(a)
        hi.append(b)

    def truth(pops):
        Rs = [ratio_of(W, Ns, pops, (Wu * pu[0] / Nu, Wu * (pu[0] + pu[1]) / Nu)) for pu in compositions(Nu)]
        return [R for R in Rs if R is not None]
    assert worst_coverage(Ns, ns, lo, hi, truth) >= 1 - alpha - 1e-12


# per kind: (y_min, y_max, w_min, w_max). Y, Wr and C known; "Y or C" an unknown map value of a point whose reference
# is the class (y in [0, 1], w = 0); "Wr or C" one whose reference is not (y = 0, w in [0, 1]); "Y or Wr" an unknown
# reference at a point mapped as the class (y + w = 1)
UNKNOWN_KINDS = [(1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 1, 0, 1)]


def true_parts(P):
    """(units that count, units in the denominator that do not) over the true types of a stratum's unknown units."""
    return {(P[0] + a + c, P[1] + b + P[5] - c)
            for a in range(P[3] + 1) for b in range(P[4] + 1) for c in range(P[5] + 1)}


@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_direct_ratio_unknown_values_cover_every_population(alpha):
    """Units of six kinds in two strata (4 and 3 units, 2 labels each), three of them with an unknown value; the true
    type of each unknown unit is anything it can be, and the interval must cover the true ratio for every one."""
    Ns, ns = (4, 3), (2, 2)
    W = (4 / 7, 3 / 7)
    d = design_of(ns, Ns, W)
    lo, hi = [], []
    for ss in itertools.product(*[compositions(n, 6) for n in ns]):
        kinds = [[k for k in range(6) for _ in range(s[k])] for s in ss]
        v = lambda i: values(d, [[float(UNKNOWN_KINDS[k][i]) for k in ks] for ks in kinds])
        a, b = ratio(d, v(0), v(3), v(1), v(2), alpha=alpha, method="direct")["interval"]   # y_min, w_max, y_max, w_min
        lo.append(a)
        hi.append(b)

    def truth(pops):
        Rs = [ratio_of(W, Ns, parts) for parts in itertools.product(*[true_parts(P) for P in pops])]
        return [R for R in Rs if R is not None]
    assert worst_coverage(Ns, ns, lo, hi, truth, kinds=6) >= 1 - alpha - 1e-12


@pytest.mark.parametrize("Ns, ns, W", [((5, 4), (3, 2), (0.5, 0.5)), ((6, 3), (3, 2), (0.5, 0.5))])
@pytest.mark.parametrize("alpha", [0.1, 0.3])
def test_direct_ratio_point_frame_with_the_deflation_covers_every_population(Ns, ns, W, alpha):
    """An area frame drawn as units: the strata read as point frames, deflated for at least the N units each holds,
    drawn without replacement as the labels were."""
    hidx = np.concatenate([np.full(n, h) for h, n in enumerate(ns)])
    d = Design(hidx, W, [math.inf] * len(Ns), Nlo=[float(N) for N in Ns])
    lo, hi = [], []
    for ss in itertools.product(*[compositions(n) for n in ns]):
        a, b = ratio(d, *three_kinds(d, ss), alpha=alpha, method="direct")["interval"]
        lo.append(a)
        hi.append(b)
    truth = lambda pops: [R for R in [ratio_of(W, Ns, pops)] if R is not None]
    assert worst_coverage(Ns, ns, lo, hi, truth) >= 1 - alpha - 1e-12


# ----------------------------------------------------------------------------------------------- the search
def test_direct_ratio_is_symmetric_in_y_and_w():
    """Swapping y and w gives [1 - U, 1 - L]: the upper end is the complement's lower end, by the same code."""
    d = design_of((3, 2), (5, 3), (0.5, 0.5))
    for ss in (((2, 1, 0), (0, 1, 1)), ((3, 0, 0), (1, 1, 0)), ((1, 1, 1), (0, 0, 2))):
        y, w = three_kinds(d, ss)
        lo, hi = ratio_interval(d, y, w, alpha=0.1)
        lo2, hi2 = ratio_interval(d, w, y, alpha=0.1)
        assert lo2 == pytest.approx(1 - hi, abs=1e-12) and hi2 == pytest.approx(1 - lo, abs=1e-12)
    kinds = [[0, 3, 5], [1, 4]]                                 # with unknown values
    v = lambda i: values(d, [[float(UNKNOWN_KINDS[k][i]) for k in ks] for ks in kinds])
    y_min, y_max, w_min, w_max = (v(i) for i in range(4))
    lo, hi = ratio_interval(d, y_min, w_max, y_max, w_min, alpha=0.1)
    lo2, hi2 = ratio_interval(d, w_min, y_max, w_max, y_min, alpha=0.1)
    assert lo2 == pytest.approx(1 - hi, abs=1e-12) and hi2 == pytest.approx(1 - lo, abs=1e-12)


@pytest.mark.parametrize("forced", [0, RATIO_BINS - 1])
def test_the_estimate_orders_the_scan_and_nothing_else(monkeypatch, forced):
    """The scan starts at the estimate's bin; forced to start at the first or the last bin, it finds the same ends:
    which test each r gets is fixed by the bins, never by the labels."""
    d = Design(np.repeat([0, 1], [20, 15]), [0.6, 0.4], [math.inf, math.inf])
    sides = ([(12, 5, 3), (4, 6, 5)], [(5, 12, 3), (6, 4, 5)])
    plain = [_ratio_low_end(d, c, 0.05) for c in sides]
    monkeypatch.setattr(st, "_scan_start", lambda design, counts: forced)
    assert [_ratio_low_end(d, c, 0.05) for c in sides] == pytest.approx(plain, abs=1e-12)
    assert 0 < plain[0] < 1 - plain[1] < 1                      # both ends say something


def test_box_keyword_is_a_deprecated_alias():
    d = design_of((3, 2), (5, 3), (0.5, 0.5))
    y, w = three_kinds(d, ((2, 1, 0), (0, 1, 1)))
    with pytest.warns(DeprecationWarning):
        old = ratio(d, y, w, alpha=0.1, box=True)
    assert old == ratio(d, y, w, alpha=0.1) and old["method"] == METHOD_BOX and "box" not in old
    with pytest.warns(DeprecationWarning):
        none = ratio(d, y, w, alpha=0.1, box=False)
    assert "interval" not in none and none == ratio(d, y, w, alpha=0.1, method=None)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        ratio(d, y, w, box=True, method="direct")
    with pytest.raises(ValueError, match="'box', 'direct' or None"):
        ratio(d, y, w, method="scan")


# ----------------------------------------------------------------------------------------------- grade
OLOFSSON_PIXELS = {"1": 200000, "2": 150000, "3": 3200000, "4": 6450000}
OLOFSSON_MATRIX = [[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]]


def olofsson_result(**kw):
    """Olofsson et al. (2014)'s worked example, as in test_agency.py: four map classes as strata, 640 labels."""
    pts = []
    for i, row in enumerate(OLOFSSON_MATRIX):
        for j, k in enumerate(row):
            pts += [(str(i + 1), str(j + 1))] * k
    d = agency.design_from_sizes([s for s, _ in pts], OLOFSSON_PIXELS)
    return agency.grade(d, [r for _, r in pts], {"map": [s for s, _ in pts]}, **kw)


def test_olofsson_direct_ratio_keeps_the_box_beside_it():
    """Each user's and producer's accuracy gets the direct interval, and the box beside it is the default's interval,
    number for number; every other number is the default's. Its direct ends are recorded from aicr as goldens."""
    res = olofsson_result(ratio_method="direct")
    box = olofsson_result()
    assert res["ratio_method"] == "direct" and "ratio_method" not in box
    for c in OLOFSSON_PIXELS:
        for k in ("user", "producer"):
            q, b = res["per_class"]["map"][c][k], box["per_class"]["map"][c][k]
            assert q["method"] == METHOD_RATIO and b["method"] == METHOD_BOX and "box" not in b
            assert q["box"] == b["interval"]
            assert q["estimate"] == b["estimate"] and q["normal_theory"] == b["normal_theory"]
            assert 0.0 <= q["interval"][0] < q["interval"][1] <= 1.0
        assert res["per_class"]["map"][c]["reference_share"] == box["per_class"]["map"][c]["reference_share"]
    assert res["overall_accuracy"] == box["overall_accuracy"]
    assert any(n.startswith("User's and producer's accuracies (--ratio direct)") for n in res["notes"])
    with pytest.raises(ValueError, match="ratio_method"):
        olofsson_result(ratio_method="scan")


def test_a_direct_interval_too_large_gives_the_box_in_its_place(monkeypatch):
    monkeypatch.setattr(st, "RATIO_MAX_WORK", 1)
    res = olofsson_result(ratio_method="direct")
    box = olofsson_result()
    for c, row in res["per_class"]["map"].items():
        for k in ("user", "producer"):
            q = row[k]
            assert q["interval"] is None and "RATIO_MAX_WORK" in q["not_computed"]
            assert q["method"] == METHOD_RATIO and q["box"] == box["per_class"]["map"][c][k]["interval"]
    assert res["overall_accuracy"]["map"]["interval"] is not None   # the rest of the grading still stands
    assert not any(n.startswith("8 interval(s) are not computed") for n in res["notes"])   # no stratum stopped them
    assert any(n.startswith("8 direct ratio interval(s) are not computed") and "RATIO_MAX_WORK" in n
               and "choice made on the data" in n for n in res["notes"])
    pure = olofsson_result(ratio_method="direct", pure={"1": "1"})["per_class"]["map"]["1"]["user"]
    assert pure["method"] == "pure-stratum rule" and pure["interval"] is not None and "box" not in pure


def test_cli_ratio_box_is_the_default_and_direct_writes_the_box_beside(tmp_path, capsys, monkeypatch):
    p = tmp_path / "t.csv"
    p.write_text("s,m,r\n" + "a,1,1\n" * 6 + "a,1,2\n" * 2 + "a,2,2\n" + "b,2,2\n" * 5 + "b,2,1\n" * 2 + "b,1,1\n")
    sizes = tmp_path / "sizes.csv"
    sizes.write_text("stratum,size\na,100\nb,300\n")
    base = ["grade", str(p), "--stratum", "s", "--reference", "r", "--map", "m", "--strata", str(sizes)]
    runs = {}
    for name, extra in (("default", []), ("box", ["--ratio", "box"]), ("direct", ["--ratio", "direct"])):
        out = tmp_path / f"{name}.json"
        assert main(base + extra + ["--out", str(out)]) == 0
        runs[name] = (capsys.readouterr().out.replace(str(out), "OUT"), json.load(open(out)))
    assert runs["box"] == runs["default"]                       # the default is the box, byte for byte
    default = runs["default"][1]
    assert "ratio_method" not in default and "ratio" not in default["inputs"]
    text, res = runs["direct"]
    assert res["ratio_method"] == "direct" and res["inputs"]["ratio"] == "direct"
    assert "(--ratio direct)" in text
    for c, row in res["per_class"]["m"].items():
        for k in ("user", "producer"):
            q, b = row[k], default["per_class"]["m"][c][k]
            assert q["method"] == METHOD_RATIO and q["box"] == b["interval"]
            assert 0.0 <= q["interval"][0] <= q["interval"][1] <= 1.0
    assert res["overall_accuracy"] == default["overall_accuracy"]
    monkeypatch.setattr(st, "RATIO_MAX_WORK", 1)
    assert main(base + ["--ratio", "direct"]) == 0
    text = capsys.readouterr().out
    assert "[not computed; box " in text and "printed in their place" in text
    with pytest.raises(SystemExit):
        main(base + ["--ratio", "scan"])

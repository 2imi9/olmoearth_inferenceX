"""Intervals for a stratified random sample with known stratum sizes: an agency's design, with intervals that hold.

An agency's accuracy assessment (NLCD's, the JRC's for GFC2020) draws a simple random sample in each stratum of a map
and reports overall, user's and producer's accuracy and class areas by Stehman's and Olofsson's estimators. Their
intervals are normal theory: nominal, and in a small class or a stratum of few labels they can cover far less than
they say (the judge's enumerations of 26 October 2026: 0.14 to 0.57 in small examples). A union bound over the strata
is valid and useless: 12 points wide for GFC2020's overall accuracy where the normal interval is 0.9, 60 to 94 points
for NLCD's user's accuracies (agency/sizing.json, aicr job 1523908). This module gives each of those numbers an
interval valid at every sample size and returns the normal-theory one beside it.

Linear quantities (overall accuracy, a class's share of the reference, the difference of two maps' accuracies), sum_h
W_h mu_h with mu_h stratum h's mean of a value in [0, 1]: a betting confidence interval. Stratum h, with n_h labels
drawn without replacement from N_h units (N_h infinite for a point frame), gets the e-value

    K_h(mu_h) = prod_t (1 + l_t (X_t - c_t)) exp(l_t (c_t - m_t(mu_h))),   t = 0 .. n_h - 1,

m_t(mu) = (N mu - S_t) / (N - t) the mean of the units not yet drawn when mu is the truth (S_t the sum of the t values
before), c_t = (1/2 + S_t) / (t + 1) the running mean, clipped at KAPPA / l_t so every factor stays positive, and
l_t = c W_h (N_h - t) / (N_h n_h). Each factor has conditional mean at most 1 (1 + y <= e^y, the step of Theorem 3.2 of
Waudby-Smith and Ramdas 2020 before Fan et al.'s inequality), so E K_h(true mu_h) <= 1. log K_h is linear in mu_h with
slope -c W_h, so the product over the independent strata depends on the means only through theta = sum_h W_h mu_h, and
no split of theta among the strata needs searching. Averaged exactly over every order of each stratum's labels (a
dynamic programme over the counts, so no random order is drawn), mixed over K_MIX values of c fixed by the design (the
weights, labels per stratum and alpha, never the labels), it is an e-value for theta; each end of the interval is where
it reaches 2/alpha (Markov). An area frame drawn as units of a known size reads the stratum as a point frame and
deflates it by exp(c W (n - 1) / (2 (N_lo - n + 1))), N_lo the fewest units the stratum can hold.

Ratios (user's and producer's accuracy), R = theta_y / (theta_y + theta_w) with y the units that count for the class
and w those in its denominator that do not: each of theta_y and theta_w gets the betting interval at alpha / 2, and R
lies in [L_y / (L_y + U_w), U_y / (U_y + L_w)] when both hold, a union bound. It costs 5 to 15% of width against a
certified scan of the ratio for the large classes and 25 to 40% for small ones, and takes seconds where the scan took
up to 40 minutes (aicr jobs 1525456 and 1531551). A user's accuracy whose class has one declared pure stratum (every
unit in it mapped as the class: an agency's no-change stratum) takes the pure-stratum rule instead, a union bound of
exact hypergeometric intervals, narrower than the box on all 15 NLCD classes (a median 0.61 of its width; class 11:
10 points against the box's 30; exp117).

With `method="direct"` (grade --ratio direct, from 1.20.0; the box stays the default), a ratio takes one betting test
inverted over r instead of the box. Each unit is given u_r = y + r (1 - y - w): 1 where y = 1, 0 where w = 1, r
elsewhere. If R <= r over the whole population, the sampled strata's sum of W_h mean_h(u_r) is at most r, whatever the
unsampled strata hold (u >= 0 there), so the linear test above with theta_0 = r tests R <= r; each end spends alpha/2.
The inversion itself is exp117's (Fieller 1954 for the classical analogue; the linearisation is Stark's assorter);
exp117's null r times the sampled share was right only with no unsampled stratum. What is new is a centre under which
the test is monotone in r, so the ends are found by a search rather than a certified walk. r runs in RATIO_BINS fixed
bins [e(j), e(j+1)), the last closed, e(j) = sin^2(j pi / (2 RATIO_BINS)) (EDGES), and in bin j the centre is r plus a
shift that does not depend on r, c_t = r + min((1/2 - rho_j + P_t - rho_j (P_t + Q_t)) / (t + 1), KAPPA / l_t -
e(j+1)), P_t and Q_t the labels of value 1 and 0 before step t and rho_j the bin's anchor (ANCHORS, sin^2 at its
middle). At r = rho_j, unclipped, that is the running mean above; l_t c_t <= KAPPA on the whole closed bin. There
log M is nonincreasing in r: per step its derivative adds -l_t (P_t + Q_t) / (N_h - t) (nothing for a point frame)
and, for a value 1 or 0, -l_t / (1 + l_t (x_t - c_t)), and the unsampled strata add -c W_unsampled. So bin j is
rejected whole when its own test (anchor j) rejects e(j+1), and in the first bin that is not, the rejected r are a
prefix, bracketed to RATIO_TOL. The family of tests is fixed by the bins, never by the labels; the estimate chooses
only which bins are evaluated first. The upper end is 1 less the lower end of 1 - R = theta_w / (theta_y + theta_w).
The box is returned beside it. It costs many evaluations of the test where the box takes two, so grade gives the box
unless asked.

Unknown values (a map's no-data, a point without a reference label) are bounded: the lower end counts each against the
map, the upper end for it. Strata with no label are bounded the same way. Coverage at the agency designs' shapes,
simulated with the observed shares (300 draws each, aicr job 1531551): 0.997 to 1.000 for the betting and box
intervals, 0.937 to 0.970 for normal theory. numpy only."""
import math
import warnings

import numpy as np

Z95 = 1.959963984540054
KAPPA = 0.9           # a centre is clipped at KAPPA / lambda, so every factor of the e-value stays at least 1 - KAPPA
K_MIX = 12            # values of c in the mixture
SPAN = 10.0           # c runs geometrically from c_lo (every stratum at variance 1/4) to SPAN * c_lo
PURE_GAMMA = 0.1      # the pure-stratum rule spends this share of alpha on the mixed strata (0.005 at alpha 0.05)
# The dynamic programme's work for one stratum is about K_MIX * n * (k1 + 1) * (k2 + 1) for its two smaller counts;
# above this it would run for many minutes, and the interval is left uncomputed rather than left running.
MAX_WORK = 5e10
# A ratio's direct interval (method "direct"): the bins of r, each with its anchor at its middle in arcsin sqrt(r);
# the points per refining round, the rounds at most and the width at which they stop; a rise of log M inside a bin
# above RATIO_ORDER_TOL is a bug, below it rounding. The values were fixed before the timings of aicr job J3.
RATIO_BINS = 32
EDGES = np.sin(np.arange(RATIO_BINS + 1) * math.pi / (2 * RATIO_BINS)) ** 2
EDGES[0], EDGES[-1] = 0.0, 1.0
ANCHORS = np.sin((np.arange(RATIO_BINS) + 0.5) * math.pi / (2 * RATIO_BINS)) ** 2
RATIO_BATCH = 8
RATIO_ROUNDS = 5                  # 5 rounds of 8 points divide the widest bin (0.049) by 9^5: below RATIO_TOL
RATIO_TOL = 1e-6
RATIO_ORDER_TOL = 1e-8
# The work of one direct interval, sum over its points and strata of K_MIX * n * (k1 + 1) * (k2 + 1), is bounded
# before it starts; above this it is not computed and the box is given beside it. A placeholder until J3 times it.
RATIO_MAX_WORK = 1e10
RATIO_CHUNK = 2 ** 22             # floats per state array of the batched programme (32 MB); larger batches are split

METHOD_LINEAR = "betting"
METHOD_BOX = "box of two betting intervals"
METHOD_PURE = "pure-stratum rule"
METHOD_RATIO = "betting test inverted over the ratio"


class TooLarge(ValueError):
    """A stratum too large for the exact average over orders: the interval that needs it is not computed, and the
    other numbers of a grading still are (the review of 26 October 2026: one refusal ended the whole run, even its
    overall accuracy)."""


def _logsumexp(x):
    x = np.asarray(x, float).ravel()
    m = float(x.max())
    if m == -np.inf:
        return -np.inf
    return m + math.log(float(np.exp(x - m).sum()))


def _stratum_logavg(counts, vals, lam0, N):
    """log of the average, over every order of one stratum's labels, of prod_t (1 + l_t (v_t - c_t)) exp(l_t (c_t +
    S_t / (N - t))), the stratum's e-value without its -c W mu term, for each first-step lambda in lam0 (shape K):
    l_t = lam0 (N - t) / N (lam0 for a point frame). counts: labels per value (at most three values, each >= 0)."""
    counts = [int(x) for x in counts] + [0] * (3 - len(counts))
    vals = [float(x) for x in vals] + [0.0] * (3 - len(vals))
    n = sum(counts)
    lam0 = np.asarray(lam0, float).reshape(-1, 1, 1)
    K = lam0.shape[0]
    if n == 0:
        return np.zeros(K)
    order = sorted(range(3), key=lambda j: -counts[j])          # the largest count is implied by the other two
    k0, k1, k2 = (counts[j] for j in order)
    v = [vals[j] for j in order]
    finite = math.isfinite(N)
    if k1 == 0 and k2 == 0:                                     # one value: a single order, summed at once
        t = np.arange(n, dtype=float).reshape(1, -1)
        lt = lam0.reshape(-1, 1) * ((N - t) / N if finite else 1.0)
        S = t * v[0]
        cen = np.minimum((0.5 + S) / (t + 1), KAPPA / lt)
        tail = S / (N - t) if finite else 0.0
        return (np.log1p(lt * (v[0] - cen)) + lt * (cen + tail)).sum(axis=1)
    work = K * n * (k1 + 1) * (k2 + 1)
    if work > MAX_WORK:
        raise TooLarge(f"a stratum of {n} labels with {k1} and {k2} in its two smaller values is too large for the "
                       f"exact average over the orders of its labels (about {work:.1e} steps); this interval is not "
                       f"computed")
    s1 = np.arange(k1 + 1, dtype=float).reshape(1, -1, 1)
    s2 = np.arange(k2 + 1, dtype=float).reshape(1, 1, -1)
    L = np.full((K, k1 + 1, k2 + 1), -np.inf)
    L[:, 0, 0] = 0.0
    for t in range(n):
        s0 = t - s1 - s2
        valid = (s0 >= 0) & (s0 <= k0)
        lt = lam0 * ((N - t) / N) if finite else lam0
        S = s0 * v[0] + s1 * v[1] + s2 * v[2]
        cen = np.minimum((0.5 + S) / (t + 1), KAPPA / lt)
        tail = S / (N - t) if finite else 0.0
        new = np.full_like(L, -np.inf)
        for j, kj in enumerate((k0, k1, k2)):
            if kj == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lf = np.log1p(lt * (v[j] - cen)) + lt * (cen + tail)
            src = L + np.where(valid, lf, -np.inf)
            if j == 0:
                new = np.logaddexp(new, np.where(s0 + 1 <= k0, src, -np.inf))
            elif j == 1:
                new[:, 1:, :] = np.logaddexp(new[:, 1:, :], src[:, :-1, :])
            else:
                new[:, :, 1:] = np.logaddexp(new[:, :, 1:], src[:, :, :-1])
        L = new
    return L[:, k1, k2] - (math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(k1 + 1) - math.lgamma(k2 + 1))


def c_grid(W, n, alpha, K=K_MIX, span=SPAN):
    """The mixture's values of c, from the design alone (weights, labels per stratum, alpha), never the labels."""
    vmax = float(np.sum(np.asarray(W, float) ** 2 / (4.0 * np.asarray(n, float))))
    clo = math.sqrt(2.0 * math.log(2.0 / alpha) / vmax)
    return np.geomspace(clo, span * clo, K) if K > 1 else np.array([clo])


class Design:
    """A stratified simple random sample. Sampled stratum h (0 .. H-1) has share W[h] of the population (of its units,
    or of its area), N[h] units (inf for a point frame) and the n[h] labelled points whose index into the strata is
    `hidx`; Nlo[h], for a point frame drawn as units, the fewest units the stratum can hold (NaN, or Nlo None, for a
    true point frame or a stratum of known size). `W_unsampled` is the share of the strata holding no label, whose
    values are bounded, never guessed."""

    def __init__(self, hidx, W, N, Nlo=None, names=None, W_unsampled=0.0, names_unsampled=()):
        self.hidx = np.asarray(hidx, dtype=int)
        self.W = np.asarray(W, dtype=float)
        self.N = np.asarray(N, dtype=float)
        self.H = int(self.W.size)
        self.n = np.bincount(self.hidx, minlength=self.H).astype(int)
        self.Nlo = None if Nlo is None else np.asarray(Nlo, dtype=float)
        self.names = list(range(self.H)) if names is None else list(names)
        self.W_unsampled = float(W_unsampled)
        self.names_unsampled = list(names_unsampled)
        if self.hidx.size and (self.hidx.min() < 0 or self.hidx.max() >= self.H):
            raise ValueError("a point's stratum index is outside the strata")
        if (self.n < 1).any():
            raise ValueError(f"stratum {self.names[int(np.argmin(self.n))]} holds no label; pass it as unsampled")
        if (self.W <= 0).any() or self.W_unsampled < 0:
            raise ValueError("every stratum needs a positive share of the population")
        if abs(self.W.sum() + self.W_unsampled - 1.0) > 1e-9:
            raise ValueError(f"the stratum shares add up to {self.W.sum() + self.W_unsampled}, not 1")
        if self.N.shape != self.W.shape or (self.Nlo is not None and self.Nlo.shape != self.W.shape):
            raise ValueError("W, N and Nlo need one entry per stratum")
        fin = np.isfinite(self.N)
        if (self.N[fin] < self.n[fin]).any():
            h = int(np.flatnonzero(fin & (self.N < self.n))[0])
            raise ValueError(f"stratum {self.names[h]} holds {self.n[h]} labels but only {self.N[h]:g} units")
        if self.Nlo is not None:
            given = ~np.isnan(self.Nlo)
            if (given & fin).any():
                raise ValueError("Nlo is for a point frame (N inf)")
            if (given & (self.Nlo < self.n)).any():
                h = int(np.flatnonzero(given & (self.Nlo < self.n))[0])
                raise ValueError(f"stratum {self.names[h]} holds {self.n[h]} labels but at least {self.Nlo[h]:g} units "
                                 f"by its area; the unit area is too large")
        self.masks = [self.hidx == h for h in range(self.H)]

    def fpc(self, h):
        """1 - n/N, the finite-population correction normal theory uses (1 for a point frame)."""
        return 1.0 - self.n[h] / self.N[h] if math.isfinite(self.N[h]) else 1.0


# ----------------------------------------------------------------------------------------------- linear quantities
def _strata(design, x):
    """Each stratum's values of x (at most three, in [0, 1]) and their counts."""
    x = np.asarray(x, dtype=float)
    if x.shape != design.hidx.shape:
        raise ValueError(f"one value per labelled point: got {x.shape}, the design has {design.hidx.shape}")
    if np.isnan(x).any() or x.min() < 0 or x.max() > 1:
        raise ValueError("values must be in [0, 1]")
    out = []
    for h in range(design.H):
        vals, counts = np.unique(x[design.masks[h]], return_counts=True)
        if vals.size > 3:
            raise ValueError(f"stratum {design.names[h]} holds {vals.size} distinct values; the interval takes three")
        out.append((vals.tolist(), counts.tolist()))
    return out


def _log_parts(design, strata, cs, side):
    """sum_h log(stratum h's averaged e-value without its -c W mu term) for each c, minus the area frames' deflation."""
    A = np.zeros(cs.size)
    for h, (vals, counts) in enumerate(strata):
        v = vals if side == "low" else [1.0 - u for u in vals]
        A += _stratum_logavg(counts, v, cs * design.W[h] / design.n[h], design.N[h])
    if design.Nlo is not None:
        given = ~np.isnan(design.Nlo)
        W, n, Nlo = design.W[given], design.n[given], design.Nlo[given]
        A -= cs * float(np.sum(W * (n - 1) / (2.0 * (Nlo - n + 1))))
    return A


def _root_low(A, cs, thr, top):
    """The largest theta0 at which the mixed e-value, decreasing in theta0, still reaches e^thr: every theta at most it
    is rejected."""
    logpi = -math.log(cs.size)
    f = lambda th: _logsumexp(logpi + A - cs * th) - thr
    if f(0.0) < 0:
        return 0.0
    if f(top) >= 0:
        return top
    lo, hi = 0.0, top
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return lo


def betting_interval(design, x_low, x_high=None, alpha=0.05):
    """The betting interval for theta = sum over every stratum of W_h times its mean of x, x in [0, 1] per labelled
    point; with x_high, unknown values bounded: the lower end from x_low (each unknown at its least), the upper from
    x_high. A stratum with no label adds 0 to the lower end and its share to the upper. Returns (low, high)."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s_lo = _strata(design, x_low)
    s_hi = s_lo if x_high is None else _strata(design, x_high)
    cs = c_grid(design.W, design.n, alpha)
    thr = math.log(2.0 / alpha)
    top = float(design.W.sum())
    lo = _root_low(_log_parts(design, s_lo, cs, "low"), cs, thr, top)
    hi = top - _root_low(_log_parts(design, s_hi, cs, "high"), cs, thr, top)
    return float(lo), float(min(1.0, hi + design.W_unsampled))


def normal_linear(design, x):
    """Stehman's estimate of sum_h W_h mean_h(x) over the sampled strata and its variance (a stratum of one label adds
    no variance; normal theory has nothing to say about it)."""
    x = np.asarray(x, dtype=float)
    est = var = 0.0
    for h in range(design.H):
        xs = x[design.masks[h]]
        est += design.W[h] * float(xs.mean())
        if xs.size > 1:
            var += design.W[h] ** 2 * design.fpc(h) * float(xs.var(ddof=1)) / xs.size
    return est, var


def linear(design, x_low, x_high=None, alpha=0.05, scale=1.0, shift=0.0):
    """theta = scale * sum_h W_h mean_h(x) + shift, x in [0, 1]: the estimate (a range when values are unknown or
    strata unsampled), the betting interval and the normal-theory one beside it, nominal."""
    e_lo, v_lo = normal_linear(design, x_low)
    e_hi, v_hi = (e_lo, v_lo) if x_high is None else normal_linear(design, x_high)
    e_hi += design.W_unsampled
    z = _z(alpha)
    T = lambda v: scale * v + shift
    exact = (x_high is None or np.array_equal(x_low, x_high)) and design.W_unsampled == 0
    out = {"estimate": T(e_lo) if exact else [T(e_lo), T(e_hi)],
           "normal_theory": [T(e_lo - z * math.sqrt(v_lo)), T(e_hi + z * math.sqrt(v_hi))], "method": METHOD_LINEAR}
    try:
        lo, hi = betting_interval(design, x_low, x_high, alpha)
        out["interval"] = [T(lo), T(hi)]
    except TooLarge as exc:
        out.update({"interval": None, "not_computed": str(exc)})
    return out


def _z(alpha):
    """The two-sided normal quantile at 1 - alpha, by bisection on the normal distribution function."""
    if alpha == 0.05:
        return Z95
    p = 1.0 - alpha / 2.0
    lo, hi = 0.0, 40.0
    for _ in range(200):                                         # bisection on the normal CDF: exact to float
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------- ratios
def normal_ratio(design, y, z, add_y=0.0, add_z=0.0):
    """Stehman's ratio estimate (sum W mean(y) + add_y) / (sum W mean(z) + add_z) and its variance by the
    linearisation u = y - R z; add_y, add_z are the unsampled strata's bounded totals. None when the denominator is 0."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    ty = tz = 0.0
    for h in range(design.H):
        m = design.masks[h]
        ty += design.W[h] * float(y[m].mean())
        tz += design.W[h] * float(z[m].mean())
    ty, tz = ty + add_y, tz + add_z
    if tz <= 0:
        return None, None
    R = ty / tz
    var = 0.0
    for h in range(design.H):
        m = design.masks[h]
        u = y[m] - R * z[m]
        if u.size > 1:
            var += design.W[h] ** 2 * design.fpc(h) * float(u.var(ddof=1)) / u.size
    return R, var / tz ** 2


def ratio(design, y_min, w_max, y_max=None, w_min=None, alpha=0.05, box=None, method="box"):
    """R = theta_y / (theta_y + theta_w), y and w indicators with y + w <= 1 at each point (y: counts for the class
    and is right; w: in the denominator and wrong). With unknown values, y_min <= y <= y_max and w_min <= w <= w_max at
    each point: the lower end takes y at its least and w at its most, the upper end the reverse. Returns the estimate
    (a range when values are unknown), the interval and the normal-theory one. method: "box" (the default), "direct"
    (the betting test inverted over the ratio, ratio_interval, with the box beside it as "box"; when the direct one is
    too large to compute, its interval is null and the box is still given) or None (no interval, for a caller with a
    better one). `box` is 1.19.0's keyword, kept for one release: box=False is method=None, box=True the default."""
    if box is not None:
        warnings.warn("ratio(box=...) is deprecated and will be removed in 1.21.0: use method='box' or method=None",
                      DeprecationWarning, stacklevel=2)
        if method != "box":
            raise ValueError("give method or the deprecated box, not both")
        method = "box" if box else None
    if method not in ("box", "direct", None):
        raise ValueError(f"method is 'box', 'direct' or None, got {method!r}")
    y_max = y_min if y_max is None else y_max
    w_min = w_max if w_min is None else w_min
    y_min, w_max, y_max, w_min = (np.asarray(a, float) for a in (y_min, w_max, y_max, w_min))
    if (y_min > y_max).any() or (w_min > w_max).any():
        raise ValueError("each unknown needs its least value at most its greatest")
    Wu = design.W_unsampled
    out = {"labels_in_denominator": int(((y_min + w_max) > 0).sum()), "method": METHOD_BOX}
    R_lo, v_lo = normal_ratio(design, y_min, y_min + w_max, 0.0, Wu)
    R_hi, v_hi = normal_ratio(design, y_max, y_max + w_min, Wu, Wu)
    exact = np.array_equal(y_min, y_max) and np.array_equal(w_min, w_max) and Wu == 0
    if method == "direct":
        out["method"] = METHOD_RATIO
    if R_lo is None and R_hi is None:
        out.update({"estimate": None, "interval": [0.0, 1.0], "normal_theory": None})
        if method == "direct":
            out["box"] = [0.0, 1.0]
        return out
    R_lo = 0.0 if R_lo is None else R_lo
    R_hi = 1.0 if R_hi is None else R_hi
    z = _z(alpha)
    out["estimate"] = R_lo if exact else [R_lo, R_hi]
    out["normal_theory"] = [R_lo - z * math.sqrt(v_lo or 0.0), R_hi + z * math.sqrt(v_hi or 0.0)]
    if method is None:
        return out
    try:
        ly, uy = betting_interval(design, y_min, y_max, alpha / 2)
        lw, uw = betting_interval(design, w_min, w_max, alpha / 2)
    except TooLarge as exc:
        # the direct interval's strata are at least as large as the box's, so it is not computed either
        out.update({"interval": None, "not_computed": str(exc)})
        if method == "direct":
            out["box"] = None
        return out
    lo = ly / (ly + uw) if ly + uw > 0 else 0.0
    hi = uy / (uy + lw) if uy + lw > 0 else 1.0
    out["interval"] = [float(max(0.0, lo)), float(min(1.0, hi))]
    if method == "box":
        return out
    out["box"] = out["interval"]
    try:
        out["interval"] = list(ratio_interval(design, y_min, w_max, y_max, w_min, alpha))
    except TooLarge as exc:
        out.update({"interval": None, "not_computed": str(exc)})
    return out


def pure_stratum_ratio(design, own, mixed, k_right, k_wrong, alpha=0.05, gamma=None):
    """A user's accuracy whose class fills stratum `own` (every unit in it mapped as the class). `mixed`: the strata
    declared pure for no class, where the class may appear; a stratum pure for another class adds 0. k_right[h],
    k_wrong[h]: the labels in stratum h mapped as the class and right, mapped as it and wrong. A union bound: own's
    right share at its exact hypergeometric interval at 1 - (alpha - gamma); each mixed stratum's two counts at
    1 - gamma / (2 M), M the mixed strata not labelled in full. The ratio rises with own's share and the mixed right
    counts and falls with the mixed wrong ones, so its ends are the ratio at those ends. Needs each stratum's size in
    whole units and no unsampled stratum. Returns (low, high)."""
    from oe_inferencex.common.hypergeom import hypergeom_interval
    gamma = PURE_GAMMA * alpha if gamma is None else gamma
    if not 0 < gamma < alpha:
        raise ValueError(f"gamma must be in (0, alpha), got {gamma}")
    if design.W_unsampled > 0:
        raise ValueError("the pure-stratum rule needs every stratum sampled")
    N, n, W = design.N, design.n, design.W
    if not np.isfinite(N).all() or (N != np.round(N)).any():
        raise ValueError("the pure-stratum rule needs each stratum's size in whole units")
    if own in mixed:
        raise ValueError("the pure stratum cannot also be mixed")
    if k_right[own] + k_wrong[own] != n[own]:
        raise ValueError("every label in the pure stratum must be mapped as the class")
    open_mixed = [h for h in mixed if n[h] < N[h]]
    g = gamma if open_mixed else 0.0
    lp, up = hypergeom_interval(int(k_right[own]), int(n[own]), int(N[own]), conf=1 - (alpha - g))
    A_lo = A_hi = B_lo = B_hi = 0.0
    for h in mixed:
        if n[h] >= N[h]:
            la = ua = k_right[h] / n[h]
            lb = ub = k_wrong[h] / n[h]
        else:
            conf = 1 - g / (2 * len(open_mixed))
            la, ua = hypergeom_interval(int(k_right[h]), int(n[h]), int(N[h]), conf=conf)
            lb, ub = hypergeom_interval(int(k_wrong[h]), int(n[h]), int(N[h]), conf=conf)
        A_lo += W[h] * la
        A_hi += W[h] * ua
        B_lo += W[h] * lb
        B_hi += W[h] * ub
    Wd = float(W[own])
    lo = (Wd * lp + A_lo) / (Wd + A_lo + B_hi)
    hi = min(1.0, (Wd * up + A_hi) / (Wd + A_hi + B_lo))
    return float(lo), float(hi)


# ----------------------------------------------------------------------------------------------- ratios, directly
def _ratio_bin(r):
    """The bin of r: [EDGES[j], EDGES[j + 1]), the last closed at 1."""
    return min(RATIO_BINS - 1, int(np.searchsorted(EDGES, r, side="right")) - 1)


def _ratio_counts(design, one, zero):
    """Per stratum, the labels of one side of a ratio's test: of value 1 (`one` = 1), of value 0 (`zero` = 1) and of
    value r (neither). The lower end takes one = y_min, zero = w_max; the upper end one = w_min, zero = y_max."""
    one, zero = np.asarray(one, float), np.asarray(zero, float)
    if one.shape != design.hidx.shape or zero.shape != design.hidx.shape:
        raise ValueError(f"one value per labelled point: got {one.shape} and {zero.shape}, the design has "
                         f"{design.hidx.shape}")
    if not (np.isin(one, (0.0, 1.0)).all() and np.isin(zero, (0.0, 1.0)).all()):
        raise ValueError("the direct ratio interval takes y and w as 0 or 1 at each point")
    if (one + zero > 1).any():
        raise ValueError("y + w must be at most 1 at each point")
    k1 = np.rint(np.bincount(design.hidx, weights=one, minlength=design.H)).astype(int)
    k0 = np.rint(np.bincount(design.hidx, weights=zero, minlength=design.H)).astype(int)
    return [(int(a), int(b), int(n - a - b)) for a, b, n in zip(k1, k0, design.n)]


def _ratio_work(counts):
    """The work of one evaluation of a side's test, in the units of MAX_WORK: K_MIX * n * (k1 + 1) * (k2 + 1) per
    stratum, k1 and k2 its two smaller counts."""
    work = 0.0
    for c in counts:
        k = sorted(c)
        work += K_MIX * sum(c) * (k[0] + 1) * (k[1] + 1)
    return work


def _stratum_logavg_ratio(counts, r, rho, top, lam0, N):
    """_stratum_logavg for one side of a ratio's test, at a batch of points r (shape B) at once. counts: the
    stratum's labels of value 1, of value 0 and of value r; P_t, Q_t and C_t of them drawn before step t. The centre
    is r plus a shift that does not depend on r, c_t = r + min((1/2 - rho + P_t - rho (P_t + Q_t)) / (t + 1),
    KAPPA / l_t - top), rho the bin's anchor and top its right end (scalars, or shape B): at r = rho, unclipped, it is
    the running mean of the values, and l_t c_t <= KAPPA for every r <= top, so every factor stays at least
    1 - KAPPA. Returns shape (B, K), K the values of lam0."""
    counts = [int(x) for x in counts]
    r = np.asarray(r, float).ravel()
    rho = np.broadcast_to(np.asarray(rho, float), r.shape)
    top = np.broadcast_to(np.asarray(top, float), r.shape)
    lam0 = np.asarray(lam0, float).ravel()
    B, K, n = r.size, lam0.size, sum(counts)
    if n == 0:
        return np.zeros((B, K))
    order = sorted(range(3), key=lambda j: -counts[j])          # the largest count is implied by the other two
    k0, k1, k2 = (counts[j] for j in order)
    single = k1 == 0 and k2 == 0
    cells = n if single else (k1 + 1) * (k2 + 1)
    if not single and K * n * cells > MAX_WORK:
        raise TooLarge(f"a stratum of {n} labels with {k1} and {k2} in its two smaller kinds is too large for the "
                       f"exact average over the orders of its labels (about {K * n * cells:.1e} steps); this interval "
                       f"is not computed")
    step = max(1, RATIO_CHUNK // (K * cells))
    if B > step:
        return np.concatenate([_stratum_logavg_ratio(counts, r[i:i + step], rho[i:i + step], top[i:i + step], lam0,
                                                     N) for i in range(0, B, step)])
    finite = math.isfinite(N)
    if single:                                                  # one kind: a single order, summed at once
        kind = order[0]
        t = np.arange(n, dtype=float).reshape(1, 1, -1)
        r3, rho3, top3 = (a.reshape(-1, 1, 1) for a in (r, rho, top))
        lt = lam0.reshape(1, -1, 1) * ((N - t) / N if finite else 1.0)
        P = t if kind == 0 else 0.0
        Q = t if kind == 1 else 0.0
        C = t if kind == 2 else 0.0
        c = r3 + np.minimum((0.5 - rho3 + P - rho3 * (P + Q)) / (t + 1), KAPPA / lt - top3)
        tail = (P + r3 * C) / (N - t) if finite else 0.0
        x = r3 if kind == 2 else (1.0 if kind == 0 else 0.0)
        return (np.log1p(lt * (x - c)) + lt * (c + tail)).sum(axis=2)
    pos = [order.index(kind) for kind in range(3)]              # where each kind's count sits among (s0, s1, s2)
    s1 = np.arange(k1 + 1, dtype=float).reshape(1, 1, -1, 1)
    s2 = np.arange(k2 + 1, dtype=float).reshape(1, 1, 1, -1)
    r4, rho4, top4 = (a.reshape(-1, 1, 1, 1) for a in (r, rho, top))
    lam = lam0.reshape(1, -1, 1, 1)
    L = np.full((B, K, k1 + 1, k2 + 1), -np.inf)
    L[:, :, 0, 0] = 0.0
    for t in range(n):
        s0 = t - s1 - s2
        valid = (s0 >= 0) & (s0 <= k0)
        drawn = (s0, s1, s2)
        P, Q, C = (drawn[pos[kind]] for kind in range(3))
        lt = lam * ((N - t) / N) if finite else lam
        c = r4 + np.minimum((0.5 - rho4 + P - rho4 * (P + Q)) / (t + 1), KAPPA / lt - top4)
        base = lt * (c + ((P + r4 * C) / (N - t) if finite else 0.0))
        new = np.full_like(L, -np.inf)
        for j, kj in enumerate((k0, k1, k2)):
            if kj == 0:
                continue
            kind = order[j]
            x = r4 if kind == 2 else (1.0 if kind == 0 else 0.0)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lf = np.log1p(lt * (x - c)) + base
            src = L + np.where(valid, lf, -np.inf)
            if j == 0:
                new = np.logaddexp(new, np.where(s0 + 1 <= k0, src, -np.inf))
            elif j == 1:
                new[:, :, 1:, :] = np.logaddexp(new[:, :, 1:, :], src[:, :, :-1, :])
            else:
                new[:, :, :, 1:] = np.logaddexp(new[:, :, :, 1:], src[:, :, :, :-1])
        L = new
    return L[:, :, k1, k2] - (math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(k1 + 1) - math.lgamma(k2 + 1))


def _deflation(design, cs):
    """The area frames' deflation of log M for each c, as in _log_parts; it does not depend on r."""
    if design.Nlo is None:
        return np.zeros(cs.size)
    given = ~np.isnan(design.Nlo)
    W, n, Nlo = design.W[given], design.n[given], design.Nlo[given]
    return cs * float(np.sum(W * (n - 1) / (2.0 * (Nlo - n + 1))))


def _ratio_logM(design, counts, r, rho, top, cs):
    """log of the mixed e-value of one side's test at each r (shape B), with theta_0 = r: the log of the mean over c
    of exp(sum_h A_h - deflation - c r). rho and top: each point's bin anchor and right end (scalars, or shape B)."""
    r = np.asarray(r, float).ravel()
    A = np.zeros((r.size, cs.size))
    for h, cnt in enumerate(counts):
        A += _stratum_logavg_ratio(cnt, r, rho, top, cs * design.W[h] / design.n[h], design.N[h])
    E = A - _deflation(design, cs)[None, :] - math.log(cs.size) - cs[None, :] * r[:, None]
    m = E.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(E - m).sum(axis=1, keepdims=True))).ravel()


def _scan_start(design, counts):
    """The bin holding the side's estimate, the last the scan's first pass evaluates. It orders the search only: every
    bin up to the first not rejected is evaluated whichever it is, so the end found does not depend on it."""
    num = sum(design.W[h] * c[0] / design.n[h] for h, c in enumerate(counts))
    den = sum(design.W[h] * (c[0] + c[1]) / design.n[h] for h, c in enumerate(counts))
    return _ratio_bin(num / den) if den > 0 else RATIO_BINS - 1


def _bracket(pts):
    """From one bin's points evaluated so far, (r, log M - threshold), the largest point below which every point is
    rejected (None when the bin's first point is accepted) and the first accepted point. log M is nonincreasing in the
    bin: a rise above RATIO_ORDER_TOL is a bug and raises; a smaller one is rounding, and a point after the first
    accepted one counts as accepted whatever its value, which can only lower the end (the conservative side)."""
    pts = sorted(pts)
    f = np.array([v for _, v in pts])
    if np.isnan(f).any():
        raise RuntimeError("the ratio's test is NaN inside a bin: a bug")
    rise = float((f - np.minimum.accumulate(f)).max())
    if rise > RATIO_ORDER_TOL:
        raise RuntimeError(f"the ratio's test rose by {rise:.3g} inside a bin, where it cannot: a bug")
    i = int(np.flatnonzero(f < 0)[0])
    return (None if i == 0 else pts[i - 1][0]), pts[i][0]


def _ratio_low_end(design, counts, alpha):
    """The lower end of one side: the largest r known to be rejected at alpha/2, every r below it rejected too. First
    pass: bin j is rejected whole when its own test (anchor j) rejects its right end; the first bin that is not holds
    the end. Then up to RATIO_ROUNDS rounds of RATIO_BATCH points (the bin's left end too, in the first) shrink the
    bracket of the last rejected and the first accepted point to RATIO_TOL."""
    if not any(c[0] + c[1] for c in counts):
        return 0.0                                              # every value is r: the test never rejects
    cs = c_grid(design.W, design.n, alpha)
    thr = math.log(2.0 / alpha)
    start = _scan_start(design, counts)
    first = f_end = None
    for bins in (np.arange(0, start + 1), np.arange(start + 1, RATIO_BINS)):
        if bins.size == 0:
            continue
        f = _ratio_logM(design, counts, EDGES[bins + 1], ANCHORS[bins], EDGES[bins + 1], cs) - thr
        acc = np.flatnonzero(f < 0)
        if acc.size:
            first, f_end = int(bins[acc[0]]), float(f[acc[0]])
            break
    if first is None:
        return 1.0
    a, b = float(EDGES[first]), float(EDGES[first + 1])
    pts = [(b, f_end)]
    lo, hi = None, b
    for k in range(RATIO_ROUNDS):
        left = a if lo is None else lo
        rs = left + (hi - left) * np.arange(1, RATIO_BATCH + 1) / (RATIO_BATCH + 1)
        if k == 0:
            rs = np.concatenate([[a], rs])
        f = _ratio_logM(design, counts, rs, ANCHORS[first], b, cs) - thr
        pts += list(zip(rs.tolist(), f.tolist()))
        lo, hi = _bracket(pts)
        if lo is None:
            return a                                            # every r below the bin is rejected, its left end not
        if hi - lo <= RATIO_TOL:
            break
    return lo


def ratio_interval(design, y_min, w_max, y_max=None, w_min=None, alpha=0.05):
    """The direct interval for R = theta_y / (theta_y + theta_w) (the module's docstring): the lower end from the
    test of R <= r with y at its least and w at its most, the upper end 1 less the lower end of 1 - R with w at its
    least and y at its most, each at alpha/2. y and w are 0 or 1 at each point. When the two ends cross, no r is
    accepted, and the interval is [the smaller, the larger]. Raises TooLarge above RATIO_MAX_WORK, bounded before any
    work. Returns (low, high)."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y_max = y_min if y_max is None else y_max
    w_min = w_max if w_min is None else w_min
    low_counts = _ratio_counts(design, y_min, w_max)
    high_counts = _ratio_counts(design, w_min, y_max)
    work = (RATIO_BINS + 1 + RATIO_ROUNDS * RATIO_BATCH) * (_ratio_work(low_counts) + _ratio_work(high_counts))
    if work > RATIO_MAX_WORK:
        raise TooLarge(f"the direct ratio interval needs about {work:.1e} steps, above RATIO_MAX_WORK "
                       f"({RATIO_MAX_WORK:.0e}); it is not computed, and the box is given beside it")
    lo = _ratio_low_end(design, low_counts, alpha)
    hi = 1.0 - _ratio_low_end(design, high_counts, alpha)
    return float(min(lo, hi)), float(max(lo, hi))

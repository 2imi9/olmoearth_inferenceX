"""Model-assisted estimation (exp85): the tuned coefficient and the model-assisted intervals."""
import numpy as np

from oe_inferencex.estimate.designs import MIN_PER_STRATUM
from oe_inferencex.estimate.intervals import Z95


# ----------------------------------------------------------------------------- model-assisted estimation (exp85)
# The map's own confidence on every unlabelled window as a predictor of error, corrected by the labels: the
# survey-sampling difference estimator, which prediction-powered inference rediscovered (Angelopoulos, Duchi and
# Zrnic 2023; Mozer et al. 2026). With the coefficient tuned on the sample the interval cannot be wider than the
# classical one beyond tuning noise. Preregistered in docs/plan/model_assisted_estimation.md.
MIN_FOR_TUNING = 30               # fewer labels keep lambda = 0, the classical estimate: exp85's run tuned at 3 and the
                                  # per-stratum coefficient reached 66 on a 26-label stratum (audit, 2026-09-23)


def tuned_coefficient(e, g):
    """PPI++'s lambda on the labelled units: the sample covariance of error with the predictor over the
    predictor's variance; 0 when the predictor does not vary among the labelled units."""
    e, g = np.asarray(e, float), np.asarray(g, float)
    if e.size < MIN_FOR_TUNING or g.var(ddof=1) <= 0:
        return 0.0
    return float(np.cov(e, g, ddof=1)[0, 1] / g.var(ddof=1))


def model_assisted_interval(err, g, picked, N, lam=None):
    """Under a simple random sample: `lam * mean(g over the population) + mean(err - lam * g over the sample)`, with
    the Wald interval from the residual variance, (1 - n/N) s^2(err - lam g) / n. The population mean of g is a
    constant, so it adds nothing. lam=None tunes it on the sample; lam=1 is the plain difference estimator; lam=0
    is the classical mean. Returns (estimate, low, high, lambda)."""
    err, g, picked = np.asarray(err, float), np.asarray(g, float), np.asarray(picked, int)
    e, gs = err[picked], g[picked]
    n = int(picked.size)
    lam = tuned_coefficient(e, gs) if lam is None else float(lam)
    resid = e - lam * gs
    est = lam * float(g.mean()) + float(resid.mean())
    var = (1 - n / N) * float(resid.var(ddof=1)) / n if n > 1 else float("nan")
    half = Z95 * np.sqrt(max(var, 0.0)) if np.isfinite(var) else float("nan")
    return float(est), max(0.0, est - half), min(1.0, est + half), lam


def stratified_model_assisted_interval(err, g, strata, picked, sizes, N, lam=None):
    """The stratified form (Fisch et al. 2024): per stratum h, `lam_h * mean(g over stratum h) + mean(err - lam_h g
    over its labels)`, weighted by W_h, with variance sum_h W_h^2 (1 - f_h) s_h^2(err - lam_h g) / n_h; lam_h is
    tuned per stratum (None), or one value for all. A stratum with fewer than MIN_FOR_TUNING labels keeps lam 0.
    Returns (estimate, low, high, {stratum: lambda}, starved strata)."""
    err, g, strata, picked = np.asarray(err, float), np.asarray(g, float), np.asarray(strata), np.asarray(picked, int)
    est = var = 0.0
    lams, starved = {}, 0
    for h, Nh in enumerate(sizes):
        if Nh == 0:
            continue
        m = strata[picked] == h
        nh = int(m.sum())
        Wh = Nh / N
        if nh < MIN_PER_STRATUM:
            starved += 1
            if nh == 1:
                est += Wh * float(err[picked][m].mean())
            continue
        e, gs = err[picked][m], g[picked][m]
        lh = tuned_coefficient(e, gs) if lam is None else float(lam)
        lams[h] = lh
        resid = e - lh * gs
        est += Wh * (lh * float(g[strata == h].mean()) + float(resid.mean()))
        var += Wh ** 2 * (1 - nh / Nh) * float(resid.var(ddof=1)) / nh
    half = Z95 * np.sqrt(max(var, 0.0))
    return float(est), max(0.0, est - half), min(1.0, est + half), lams, starved

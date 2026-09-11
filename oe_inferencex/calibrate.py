"""Fuse the label-free readings with labels: a ranker or a side rule fitted on a labelled set, reported held-out, bound
to the model family it was fitted on.

The label-free layers (assess, explain, compare) say where to look, why, and how two inferences differ. Where labels
exist, a fitted combination of those readings does better than any single one: exp49's linear fusion beat the model's
own confidence on three of four arms, exp59's rule for which side of a difference to believe gained 5 to 17 points
over the raw margin on the crop-offset, backbone and sensor pairs. Neither transfers across model families: exp59's
rule, fitted on frozen heads, lost 26 points on the fine-tuned pair. So a fusion here is (1) fitted on labels the
caller supplies, never on imagery, (2) reported with held-out numbers, cross-fitted by group (tile, event) so that no
window grades the weights it trained, and (3) tagged with the family it was fitted on, and refuses to score another
unless told to.

  fit_ranker(signals, errors, ok, ...)        which windows to review: P(error | readings)
  fit_side(features_a, features_b, a, b, ok, labels, ...)
                                              which side of a difference to believe: P(b right | readings of both)
  Fusion.score / Fusion.prob / to_dict / from_dict

Logistic regression on standardised features, Newton's method with a small ridge, numpy only; the same model exp49
and exp59 fitted with an optimiser. Signals may point either way: the fit learns each sign, and the report says which.
"""
import numpy as np

from oe_inferencex.compare import over_groups
from oe_inferencex.metrics import capture_at_budget_expected, excess_aurc, expected_calibration_error

MIN_GROUP_ERRORS = 3


def _logistic(X, y, balanced=False, ridge=1e-3, iters=40):
    """Ridge logistic regression by Newton's method; returns (weights, bias). `balanced` reweights the positives so the
    two classes carry equal total weight (evidence.train_logistic_head's pos_weight)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    sw = np.ones(n)
    if balanced and y.sum() > 0:
        sw[y > 0.5] = (n - y.sum()) / y.sum()
    reg = np.diag(np.r_[np.full(d, ridge), 0.0])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        g = Xb.T @ (sw * (p - y)) + reg @ w
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(H + 1e-9 * np.eye(d + 1), g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w[:-1], float(w[-1])


class Fusion:
    """A fitted combination of named readings: standardisation, weights, bias, the family it was fitted on."""

    def __init__(self, kind, names, mu, sd, w, b, family=None, n_fit=0, baseline=None):
        self.kind, self.names, self.family, self.n_fit, self.baseline = kind, list(names), family, int(n_fit), baseline
        self.mu, self.sd, self.w, self.b = np.asarray(mu, float), np.asarray(sd, float), np.asarray(w, float), float(b)

    def _design(self, features):
        missing = [k for k in self.names if k not in features]
        if missing:
            raise KeyError(f"features missing: {missing}")
        X = np.stack([np.asarray(features[k], dtype=np.float64).ravel() for k in self.names], 1)
        return (np.nan_to_num(X) - self.mu) / self.sd

    def check_family(self, family, force=False):
        if force or family is None or self.family is None or family == self.family:
            return
        raise ValueError(f"fusion fitted on {self.family!r}, asked to score {family!r}: a rule fitted on one model family does not "
                         f"transfer (exp59 lost 26 points on the fine-tuned pair); refit, or pass force=True")

    def score(self, features, family=None, force=False):
        """The logit: for a ranker higher means more suspect; for a side rule higher means side b more likely right.
        Shape of the first feature."""
        self.check_family(family, force)
        shape = np.shape(features[self.names[0]])
        return (self._design(features) @ self.w + self.b).reshape(shape)

    def prob(self, features, family=None, force=False):
        return 1.0 / (1.0 + np.exp(-self.score(features, family, force)))

    def weights(self):
        """Per reading: the weight on the standardised reading, and its sign as the direction the fit learned."""
        return {k: {"weight": float(v), "direction": "higher is more suspect" if v > 0 else "higher is less suspect"} if self.kind == "ranker"
                else {"weight": float(v), "direction": "favours b" if v > 0 else "favours a"} for k, v in zip(self.names, self.w)}

    def to_dict(self):
        return {"kind": self.kind, "names": self.names, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), "b": self.b,
                "family": self.family, "n_fit": self.n_fit, "baseline": self.baseline}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["names"], d["mu"], d["sd"], d["w"], d["b"], d.get("family"), d.get("n_fit", 0), d.get("baseline"))


# ----------------------------------------------------------------------------- shared machinery
def _standardise(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def _folds(groups, n, folds, seed=0):
    """Fold id per row: groups (a group id per row, or None for one row per group) split into `folds` folds."""
    g = np.arange(n) if groups is None else np.asarray(groups).ravel()
    ids = np.unique(g)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    fold_of = {ids[i]: k % folds for k, i in enumerate(order)}
    return np.array([fold_of[x] for x in g])


def _crossfit(X, y, fold, balanced, n_folds):
    """Held-out logit for every row, from a fit on the other folds."""
    out = np.zeros(len(y))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        w, b = _logistic(X[tr], y[tr], balanced)
        out[te] = X[te] @ w + b
    return out


def _scoreable(err):
    return MIN_GROUP_ERRORS <= err.sum() <= len(err) - MIN_GROUP_ERRORS


# ----------------------------------------------------------------------------- the ranker
def fit_ranker(signals, errors, ok, groups=None, family=None, folds=5, budgets=(0.05, 0.10), balanced=True):
    """P(error | readings) as a review ranker, fitted on labelled windows and reported held-out.

    signals: {name: array} of per-window readings, any orientation (the fit learns each sign); errors: 0/1 per
    window; ok: validity; groups: tile or event id per window, the unit of cross-fitting and of the sign test.
    Returns (Fusion, report). The report: the weights with their learned directions; the fusion's held-out excess
    AURC, capture at the budgets and calibration (ECE of P(error)); every single signal's excess AURC as given and
    flipped; the best single; and a one-sided sign test over groups that the held-out fusion beats the best single."""
    names = list(signals)
    okm = np.asarray(ok) > 0.5
    err = (np.asarray(errors, dtype=np.float64) > 0.5).astype(np.float64)[okm]
    Xraw = np.stack([np.asarray(signals[k], dtype=np.float64)[okm] for k in names], 1)
    Xraw = np.nan_to_num(Xraw)
    X, mu, sd = _standardise(Xraw)
    w, b = _logistic(X, err, balanced)
    fusion = Fusion("ranker", names, mu, sd, w, b, family, int(okm.sum()))
    g = None if groups is None else np.broadcast_to(np.asarray(groups).reshape(np.asarray(groups).shape + (1,) * (np.asarray(ok).ndim - np.asarray(groups).ndim)), np.shape(ok))[okm]
    fold = _folds(g, len(err), folds)
    held = _crossfit(X, err, fold, balanced, folds)
    singles = {k: {"as_given": excess_aurc(Xraw[:, i], err), "flipped": excess_aurc(-Xraw[:, i], err)} for i, k in enumerate(names)}
    best_name = min(names, key=lambda k: singles[k]["as_given"])
    best = Xraw[:, names.index(best_name)]
    report = {"n_windows": int(okm.sum()), "n_errors": int(err.sum()), "n_groups": int(len(np.unique(g))) if g is not None else None, "folds": folds,
              "weights": fusion.weights(), "bias": b,
              "held_out": {"excess_aurc": excess_aurc(held, err), "capture": {str(k): v for k, v in capture_at_budget_expected(held, err, budgets).items()},
                           "ece_of_p_error": expected_calibration_error(1 / (1 + np.exp(-held)), err)[0]},
              "in_sample_excess_aurc": excess_aurc(X @ w + b, err),
              "singles_excess_aurc": singles, "best_single": best_name, "best_single_excess_aurc": singles[best_name]["as_given"],
              "held_out_lead_over_best_single": singles[best_name]["as_given"] - excess_aurc(held, err)}
    if g is not None:
        gains = {}
        for gid in np.unique(g):
            m = g == gid
            if _scoreable(err[m]):
                gains[gid.item() if hasattr(gid, "item") else gid] = excess_aurc(best[m], err[m]) - excess_aurc(held[m], err[m])
        report["over_groups_vs_best_single"] = over_groups(gains)
    return fusion, report


# ----------------------------------------------------------------------------- the side rule
def side_features(features_a, features_b):
    """The design for a side rule: for every named reading, the value on side a, on side b, and their difference."""
    out = {}
    for k in features_a:
        if k not in features_b:
            raise KeyError(f"reading {k!r} missing on side b")
        fa, fb = np.asarray(features_a[k], dtype=np.float64), np.asarray(features_b[k], dtype=np.float64)
        out[f"{k}:a"], out[f"{k}:b"], out[f"{k}:a-b"] = fa, fb, fa - fb
    return out


def fit_side(features_a, features_b, a, b, ok, labels, groups=None, family=None, folds=5, baseline=None, min_group_windows=3):
    """P(side b is right | readings of both sides) on the windows where two decisions differ, fitted on labels and
    reported held-out.

    features_a, features_b: {name: array} of the same readings on each side (margins, ranks, boundary, ...); a, b:
    the two decisions; labels: the class map; baseline: the reading whose larger side the raw rule believes (default
    the first). Returns (Fusion, report): held-out share right of the fitted rule, of the baseline rule, of always a,
    always b, the coin; and a one-sided sign test over groups that the fitted rule beats the baseline."""
    a_, b_, lab, okm = np.asarray(a), np.asarray(b), np.asarray(labels), np.asarray(ok) > 0.5
    d = (a_ != b_) & okm
    feats = side_features(features_a, features_b)
    names = list(feats)
    baseline = baseline or list(features_a)[0]
    Xraw = np.nan_to_num(np.stack([feats[k][d] for k in names], 1))
    y = (b_ == lab)[d].astype(np.float64)
    X, mu, sd = _standardise(Xraw)
    w, bias = _logistic(X, y, balanced=False)
    fusion = Fusion("side", names, mu, sd, w, bias, family, int(d.sum()), baseline)
    g = None if groups is None else np.broadcast_to(np.asarray(groups).reshape(np.asarray(groups).shape + (1,) * (a_.ndim - np.asarray(groups).ndim)), a_.shape)[d]
    fold = _folds(g, len(y), folds)
    held = _crossfit(X, y, fold, False, folds)
    fitted_right = (held > 0) == (y > 0.5)
    base_right = ((feats[f"{baseline}:a-b"][d] < 0) == (y > 0.5))
    a_right, b_right = (a_ == lab)[d], (b_ == lab)[d]
    report = {"n_disagree": int(d.sum()), "n_groups": int(len(np.unique(g))) if g is not None else None, "folds": folds, "weights": fusion.weights(), "bias": bias,
              "held_out": {"share_right": float(fitted_right.mean()) if d.any() else float("nan")},
              "baseline": {"reading": baseline, "share_right": float(base_right.mean()) if d.any() else float("nan")},
              "always_a": float(a_right.mean()) if d.any() else float("nan"), "always_b": float(b_right.mean()) if d.any() else float("nan"), "coin": 0.5,
              "neither_right": float((~a_right & ~b_right).mean()) if d.any() else float("nan")}
    if g is not None:
        gains = {}
        for gid in np.unique(g):
            m = g == gid
            if m.sum() >= min_group_windows:
                gains[gid.item() if hasattr(gid, "item") else gid] = float(fitted_right[m].mean() - base_right[m].mean())
        report["over_groups_vs_baseline"] = over_groups(gains)
    return fusion, report

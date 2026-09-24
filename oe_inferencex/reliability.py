"""SHRUG-FM's reliability signals (Gonzalez-Calabuig et al., arXiv 2511.10370), torch-free.

Embedding-space OOD against k-means centroids of a reference set: the normalized distance to the nearest
centroid and the Nearest Centroid Distance Deficit (NCDD, Pokhrel et al. 2024; SHRUG-FM eq. 5). Input-space
extremity: the percentile rank of per-unit statistics against the reference set's empirical CDFs. The entropy
decomposition of an ensemble's probabilities (mutual information, average entropy, predictive entropy), tile
bootstrap resamples for building such an ensemble, and the standardize / poly2 helpers of a labelled logistic
fusion. Pure numpy in float64; distances are computed in blocks of CHUNK rows, so a 150k x 768 set against 64
centroids never holds more than one (CHUNK, D) block and its (CHUNK, k) distances. exp/exp49_shrug_signals.py
is the consumer and states the substitutions it makes for the paper's pretraining data.
"""
import numpy as np

CHUNK = 8192


def _chunks(X):
    """Yield (start, block) over the rows of X, each block cast to float64."""
    for i in range(0, len(X), CHUNK):
        yield i, np.asarray(X[i:i + CHUNK], dtype=np.float64)


def _dist(x, C):
    """Euclidean distances (n, k) from the rows of x to the centroids C, by the expanded square."""
    d2 = (x * x).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (x @ C.T)
    return np.sqrt(np.maximum(d2, 0.0))


def kmeans(X, k=64, iters=25, seed=0):
    """Lloyd's k-means with k-means++ seeding -> (centroids (k, D), assignment (N,), mean member distance per cluster (k,))."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    if k > N:
        raise ValueError(f"k = {k} clusters from {N} points: some clusters would be empty or single points")
    C = np.empty((k, D))
    C[0] = X[rng.integers(N)]
    dmin = np.full(N, np.inf)
    for j in range(1, k):                                       # k-means++: the next seed is drawn proportional to D(x)^2
        for i, x in _chunks(X):
            dmin[i:i + len(x)] = np.minimum(dmin[i:i + len(x)], _dist(x, C[j - 1:j])[:, 0] ** 2)
        s = dmin.sum()
        C[j] = X[rng.choice(N, p=dmin / s) if s > 0 else rng.integers(N)]
    for _ in range(iters):
        sums, counts = np.zeros((k, D)), np.zeros(k)
        for i, x in _chunks(X):
            hot = np.zeros((len(x), k))
            hot[np.arange(len(x)), _dist(x, C).argmin(1)] = 1.0
            sums += hot.T @ x
            counts += hot.sum(0)
        C = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1.0)[:, None], C)
        for c in np.flatnonzero(counts == 0):                   # an empty cluster is re-seeded to a random point
            C[c] = X[rng.integers(N)]
    assign, dc = np.empty(N, dtype=np.int64), np.empty(N)
    for i, x in _chunks(X):
        d = _dist(x, C)
        a = d.argmin(1)
        assign[i:i + len(x)], dc[i:i + len(x)] = a, d[np.arange(len(x)), a]
    counts = np.bincount(assign, minlength=k)
    intra_mean = np.bincount(assign, weights=dc, minlength=k) / np.maximum(counts, 1)
    # A cluster of one point, or of identical points, has no spread, and an empty one has none to measure. Their
    # spread used to be 0 (floored to 1e-12 downstream) or a placeholder of 1.0 in data units: one outlier seeded by
    # k-means++ became a singleton, every normalised distance near it went to about 1e11, and the NCDD ranking
    # inverted with exit 0 (audit 2026-09-22). Such clusters take the median spread of the clusters that have one.
    degenerate = (counts <= 1) | (intra_mean <= 0)
    if degenerate.all():
        raise ValueError("no cluster has a measurable spread; the points are too few or identical")
    intra_mean = np.where(degenerate, float(np.median(intra_mean[~degenerate])), intra_mean)
    return C, assign, intra_mean


def centroid_signals(X, centroids, intra_mean):
    """Per row: the normalized distance d_c / intra_mean[c] to its nearest centroid c (SHRUG-FM eq. 4); NCDD as SHRUG-FM
    eq. 5 defines it, sum_{j != c} d~_j - (k - 1) d~_c over the CLUSTER-NORMALIZED distances d~_j = d_j / intra_mean[j]
    (higher = more in-distribution; not monotone in distance when cluster spreads differ); and NCDD raw, the same
    deficit over raw distances (Pokhrel et al. 2024), which tends to 0 far from every centroid."""
    C, im = np.asarray(centroids, dtype=np.float64), np.asarray(intra_mean, dtype=np.float64)
    if (im <= 0).any():                                                 # kmeans no longer returns these; a caller's own might
        if not (im > 0).any():
            raise ValueError("no centroid has a positive spread")
        im = np.where(im > 0, im, float(np.median(im[im > 0])))
    nd, ncdd, raw = np.empty(len(X)), np.empty(len(X)), np.empty(len(X))
    for i, x in _chunks(X):
        d = _dist(x, C)
        a = d.argmin(1)
        r = np.arange(len(x))
        dc = d[r, a]
        nd[i:i + len(x)] = dc / im[a]
        dn = d / im[None, :]                                            # cluster-normalized distances
        ncdd[i:i + len(x)] = dn.sum(1) - len(C) * dn[r, a]             # sum over the others minus (k - 1) d~_c
        raw[i:i + len(x)] = d.sum(1) - len(C) * dc
    return {"normalized distance": nd, "NCDD": ncdd, "NCDD raw": raw}


def input_extremity(train_stats, stats):
    """Per feature, e_j = 2 |F_j(x_j) - 1/2| with F_j the reference column's empirical CDF (midranks); returns its max and mean over features."""
    tr, st = np.asarray(train_stats, dtype=np.float64), np.asarray(stats, dtype=np.float64)
    if tr.shape[1] != st.shape[1]:
        # np.empty left the extra columns holding whatever memory held, often the previous call's extremities
        raise ValueError(f"train_stats has {tr.shape[1]} features and stats {st.shape[1]}; they must match")
    e = np.full(st.shape, np.nan)
    for j in range(tr.shape[1]):
        col = np.sort(tr[:, j][np.isfinite(tr[:, j])])                 # NaN in the reference is not a value above the rest
        if col.size == 0:
            continue
        x = st[:, j]
        u = (np.searchsorted(col, x, side="left") + np.searchsorted(col, x, side="right")) / 2.0 / len(col)
        e[:, j] = np.where(np.isfinite(x), 2.0 * np.abs(u - 0.5), np.nan)   # a NaN statistic has no rank
    with np.errstate(all="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            return {"max": np.nanmax(e, 1), "mean": np.nanmean(e, 1)}


def _entropy(p, eps):
    """Binary entropy in nats with p clipped to [eps, 1 - eps]."""
    q = np.clip(p, eps, 1.0 - eps)
    return -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))


def ensemble_uncertainty(p, eps=1e-7):
    """Members' class-1 probabilities (M, ...) -> mutual information (epistemic, clipped at 0), average entropy (aleatoric) and predictive entropy, each shaped (...)."""
    p = np.asarray(p, dtype=np.float64)
    pred, avg = _entropy(p.mean(0), eps), _entropy(p, eps).mean(0)
    return {"mutual information": np.maximum(pred - avg, 0.0), "average entropy": avg, "predictive entropy": pred}


def bootstrap_tiles(n_tiles, n_members, seed=0):
    """n_members with-replacement resamples of arange(n_tiles), each of length n_tiles, drawn one after another from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_tiles, n_tiles).astype(np.int64) for _ in range(n_members)]


def standardize(X, mu=None, sd=None):
    """Column-wise (X - mu) / sd with the statistics of X unless given, sd floored at 1e-6; returns (Z, mu, sd)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0) if mu is None else np.asarray(mu, dtype=np.float64)
    sd = np.maximum(X.std(0), 1e-6) if sd is None else np.asarray(sd, dtype=np.float64)
    return (X - mu) / sd, mu, sd


def poly2(Z):
    """(N, F) -> (N, F + F(F+1)/2): the columns of Z, then every product Z_i Z_j for i <= j in i-major order."""
    Z = np.asarray(Z, dtype=np.float64)
    return np.concatenate([Z] + [Z[:, i:i + 1] * Z[:, i:] for i in range(Z.shape[1])], axis=1)


# ----------------------------------------------------------------------------- label-free reliability from a panel
# Dawid and Skene (1979) over hard votes. exp07 rejected it within one model family (agreement on shared errors is
# read as competence); exp83 measures it across families, where the hidden share of each rater's errors is the
# share the panel majority makes with it (docs/plan/consensus_reliability.md). Lived in evidence.py until
# 2026-09-23, behind a torch import it never needed.
def dawid_skene(votes, n_classes, iters=1000, tol=1e-6, return_info=False):
    """Dawid-Skene EM over hard votes (N items, R raters). No labels used.

    Returns (posteriors [N, C], confusions [R, C, C], reliabilities [R])
    where `confusions[r][true, voted]` is rater r's confusion matrix and reliability is the prior-weighted
    diagonal of the confusion matrix (expected accuracy of rater r). With return_info=True a fourth value,
    {"iterations", "converged", "last_change"}, says whether the stop (largest posterior change below `tol`) was
    reached before `iters`. Until 2026-09-23 the cap was 50: exp83's audit found the 15- and 19-class panels of
    the suite needing 224 and 261 iterations, so the estimates at 50 were snapshots (a hidden-error share of 0.45
    where the converged value is 0.28); exp07's 9-class run may have been the same.
    """
    votes = np.asarray(votes)
    # review of 2026-09-23: boolean votes indexed as masks, float or NaN votes failed deep inside, iters=0 crashed,
    # tol=0 could never stop, and one rater ran to the cap reporting a reliability nothing identifies
    if votes.ndim != 2:
        raise ValueError(f"votes must be (items, raters), got shape {votes.shape}")
    if votes.dtype.kind == "b":
        votes = votes.astype(int)
    elif votes.dtype.kind == "f":
        if not np.isfinite(votes).all() or not np.all(np.mod(votes, 1) == 0):
            raise ValueError("votes must be whole class ids; a missing vote cannot be NaN (drop the item or the rater)")
        votes = votes.astype(int)
    if votes.shape[1] < 2:
        raise ValueError("Dawid-Skene needs at least two raters: one rater's confusion matrix is not identified")
    if int(iters) < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    if votes.size and (votes.min() < 0 or votes.max() >= n_classes):
        # an abstain code of -1 used to index the last class and count as a vote for it
        raise ValueError(f"votes must be classes 0 to {n_classes - 1}; got values from {votes.min()} to {votes.max()}")
    n, r = votes.shape
    post = np.zeros((n, n_classes))
    for i in range(n):
        for j in range(r):
            post[i, votes[i, j]] += 1
    post /= post.sum(1, keepdims=True)
    change = float("inf")
    for it in range(iters):
        prior = post.mean(0)
        conf = np.zeros((r, n_classes, n_classes))
        for j in range(r):
            for c in range(n_classes):
                conf[j, :, c] = post[votes[:, j] == c].sum(0)
        conf += 0.01
        conf /= conf.sum(2, keepdims=True)
        with np.errstate(divide="ignore"):                    # a class no rater voted has prior 0: log 0 is -inf, not a warning
            logp = np.log(prior)[None, :].repeat(n, 0)
        for j in range(r):
            logp += np.log(conf[j, :, votes[:, j]])
        logp -= logp.max(1, keepdims=True)
        new_post = np.exp(logp)
        new_post /= new_post.sum(1, keepdims=True)
        change = float(np.abs(new_post - post).max())
        post = new_post
        if change <= tol:
            converged = True
            break
    else:
        converged = False
    prior = post.mean(0)
    reliab = np.array([(prior * np.diag(conf[j])).sum() for j in range(r)])
    if return_info:
        return post, conf, reliab, {"iterations": it + 1, "converged": converged, "last_change": change}
    return post, conf, reliab

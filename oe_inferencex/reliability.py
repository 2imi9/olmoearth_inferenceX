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
    intra_mean = np.where(counts > 0, np.bincount(assign, weights=dc, minlength=k) / np.maximum(counts, 1), 1.0)
    return C, assign, intra_mean


def centroid_signals(X, centroids, intra_mean):
    """Per row: the normalized distance d_c / intra_mean[c] to its nearest centroid c (SHRUG-FM eq. 4); NCDD as SHRUG-FM
    eq. 5 defines it, sum_{j != c} d~_j - (k - 1) d~_c over the CLUSTER-NORMALIZED distances d~_j = d_j / intra_mean[j]
    (higher = more in-distribution; not monotone in distance when cluster spreads differ); and NCDD raw, the same
    deficit over raw distances (Pokhrel et al. 2024), which tends to 0 far from every centroid."""
    C, im = np.asarray(centroids, dtype=np.float64), np.asarray(intra_mean, dtype=np.float64)
    im = np.where(im > 0, im, 1e-12)                                    # a singleton cluster has no spread; keep the division finite
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
    e = np.empty(st.shape)
    for j in range(tr.shape[1]):
        col = np.sort(tr[:, j])
        u = (np.searchsorted(col, st[:, j], side="left") + np.searchsorted(col, st[:, j], side="right")) / 2.0 / len(col)
        e[:, j] = 2.0 * np.abs(u - 0.5)
    return {"max": e.max(1), "mean": e.mean(1)}


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

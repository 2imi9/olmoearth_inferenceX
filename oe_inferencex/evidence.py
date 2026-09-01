"""Layer 1: pure evidence math. No network, no LLM, no agent imports."""
import numpy as np
import torch


def train_logistic_head(feats, labels, epochs=300, lr=0.05):
    """Balanced logistic regression on (N, D) features, (N,) binary labels."""
    x = feats.reshape(-1, feats.shape[-1]).float()
    y = torch.as_tensor(labels, dtype=torch.float32).flatten()
    pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logit = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def predict_head(feats, w, b):
    """Water probability map, same spatial shape as feats minus channel dim."""
    h, wd, _ = feats.shape
    logit = feats.reshape(-1, feats.shape[-1]).float() @ w + b
    return torch.sigmoid(logit).reshape(h, wd).numpy()


def risk_coverage(uncertainty, errors):
    """Selective risk at every coverage level, ordered by ascending uncertainty.

    Returns (coverage, risk, aurc). Lower AURC = the signal ranks errors better.
    """
    u = uncertainty.flatten()
    e = errors.flatten().astype(np.float64)
    order = np.argsort(u, kind="stable")
    cum_err = np.cumsum(e[order])
    n = len(u)
    coverage = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    return coverage, risk, float(risk.mean())


def rasterize_polyline(coords_px, size):
    """Burn a polyline (list of (row, col) float pairs) onto a size x size grid."""
    grid = np.zeros((size, size), dtype=bool)
    for (r0, c0), (r1, c1) in zip(coords_px[:-1], coords_px[1:]):
        n = max(int(np.hypot(r1 - r0, c1 - c0) * 2), 1)
        for t in np.linspace(0, 1, n + 1):
            r, c = r0 + t * (r1 - r0), c0 + t * (c1 - c0)
            if 0 <= int(r) < size and 0 <= int(c) < size:
                grid[int(r), int(c)] = True
    return grid


def pool_to_patches(grid, patch):
    """Max-pool a boolean/fraction pixel grid to the patch grid."""
    h, w = grid.shape
    return grid.reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3))


def train_softmax_head(feats, labels, n_classes, epochs=400, lr=0.05):
    """Multinomial logistic regression on (N, D) features, (N,) int labels."""
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    weight = counts.sum() / (n_classes * counts)
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y, weight=weight)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def predict_softmax_head(feats, w, b):
    """(N, C) probability matrix."""
    x = torch.as_tensor(feats, dtype=torch.float32)
    return torch.softmax(x @ w + b, dim=-1).numpy()


def dawid_skene(votes, n_classes, iters=50):
    """Dawid-Skene EM over hard votes (N items, R raters). No labels used.

    Returns (posteriors [N, C], confusions [R, C, C], reliabilities [R])
    where confusions[r][true, voted] and reliability is the prior-weighted
    diagonal of the confusion matrix (expected accuracy of rater r).
    """
    votes = np.asarray(votes)
    n, r = votes.shape
    post = np.zeros((n, n_classes))
    for i in range(n):
        for j in range(r):
            post[i, votes[i, j]] += 1
    post /= post.sum(1, keepdims=True)
    for _ in range(iters):
        prior = post.mean(0)
        conf = np.zeros((r, n_classes, n_classes))
        for j in range(r):
            for c in range(n_classes):
                conf[j, :, c] = post[votes[:, j] == c].sum(0)
        conf += 0.01
        conf /= conf.sum(2, keepdims=True)
        logp = np.log(prior)[None, :].repeat(n, 0)
        for j in range(r):
            logp += np.log(conf[j, :, votes[:, j]])
        logp -= logp.max(1, keepdims=True)
        new_post = np.exp(logp)
        new_post /= new_post.sum(1, keepdims=True)
        if np.abs(new_post - post).max() < 1e-6:
            post = new_post
            break
        post = new_post
    prior = post.mean(0)
    reliab = np.array([(prior * np.diag(conf[j])).sum() for j in range(r)])
    return post, conf, reliab

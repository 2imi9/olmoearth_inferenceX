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

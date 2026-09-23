"""Layer 1: pure evidence math. No network, no LLM, no agent imports."""
import numpy as np
try:
    import torch
except ImportError as exc:  # pragma: no cover - a plain install, without the extra
    raise ImportError("oe_inferencex.evidence needs torch: "
                      "pip install 'olmoearth-inferencex[encoder]'") from exc

from oe_inferencex.metrics import aurc_expected, risk_coverage  # noqa: F401  (re-exported; torch-free home)
from oe_inferencex.reliability import dawid_skene  # noqa: F401  (moved to its torch-free home on 2026-09-23; exp07 and exp83 call it)


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


def predict_logit(feats, w, b):
    """Logit map for a binary head; -|logit| is a tie-free confidence signal."""
    h, wd, _ = feats.shape
    return (feats.reshape(-1, feats.shape[-1]).float() @ w + b).reshape(h, wd).numpy()


def rasterize_polyline(coords_px, size):
    """Burn a polyline (list of (row, col) float pairs) onto a size x size grid."""
    grid = np.zeros((size, size), dtype=bool)
    for (r0, c0), (r1, c1) in zip(coords_px[:-1], coords_px[1:]):
        n = max(int(np.hypot(r1 - r0, c1 - c0) * 2), 1)
        for t in np.linspace(0, 1, n + 1):
            r, c = r0 + t * (r1 - r0), c0 + t * (c1 - c0)
            ri, ci = int(np.floor(r)), int(np.floor(c))
            if 0 <= ri < size and 0 <= ci < size:
                grid[ri, ci] = True
    return grid


def pool_to_patches(grid, patch):
    """Mean-pool a boolean/fraction pixel grid to the patch grid (the fraction of each patch that is set; callers
    threshold it at 0.5 for a majority or at 0 for any). A ragged right or bottom edge is cropped, as the signals'
    pooling does; it used to raise a reshape error."""
    grid = np.asarray(grid)
    h, w = grid.shape[0] // patch * patch, grid.shape[1] // patch * patch
    return grid[:h, :w].reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3))


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

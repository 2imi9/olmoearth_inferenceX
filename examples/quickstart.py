"""Quickstart: a synthetic prediction map through the package, numpy only, no data to download.

Run:  python examples/quickstart.py
The same two operations from the shell:
    oe-inferencex assess  scores.npy --logits --out audit
    oe-inferencex compare a.npy b.npy --out diff
"""
import numpy as np

import oe_inferencex as ox

rng = np.random.default_rng(0)
C, H, W = 3, 96, 96

# A per-class logit map with a planted structure: class regions with soft edges, so the low-margin windows
# fall on the boundaries, which is where errors live on the labelled testbeds.
yy, xx = np.mgrid[0:H, 0:W]
truth = ((xx > 40) & (yy < 60)).astype(int) + 2 * (yy >= 60)
scores = rng.normal(0, 1.6, size=(C, H, W)).astype(np.float32)     # noisy enough that some windows are wrong
for c in range(C):
    scores[c] += 3.0 * (truth == c)

# 1. Which windows should a reviewer check first?
out = ox.assess_prediction(scores, is_logit=True, patch=4, budgets=(0.05, 0.10))
print("assess:", ox.summary(out))

# 2. Why is each of them suspect?
reasons = ox.explain_review_set(out, budgets=(0.05,))
print("explain: keys ->", sorted(reasons)[:6])

# 3. A second inference of the same scene: how much do they differ, and where?
other = scores + rng.normal(0, 0.9, size=scores.shape).astype(np.float32)
a, b = scores.argmax(0), other.argmax(0)
a_w = ox.pool_to_windows(a.astype(float), patch=4).round().astype(int)
b_w = ox.pool_to_windows(b.astype(float), patch=4).round().astype(int)
ok = np.ones_like(a_w, dtype=bool)
diff = ox.compare_inferences(a_w, b_w, ok)
print("compare: keys ->", sorted(diff)[:8])

# 4. With labels, only ever to grade: the margin's excess AURC against the truth.
truth_w = ox.pool_to_windows(truth.astype(float), patch=4).round().astype(int)
err = (a_w != truth_w).astype(float).ravel()
sus = -ox.confidence(scores, multiclass=True)
sus_w = ox.pool_to_windows(sus, patch=4).ravel()
cap = ox.capture_at_budget_expected(sus_w, err, budgets=(0.05, 0.10))
print(f"graded against the planted truth: {int(err.sum())} of {len(err)} windows wrong; the margin's excess AURC "
      f"{ox.excess_aurc(sus_w, err):.4f}; the most suspect 5% of windows hold {cap[0.05]:.0%} of the errors, "
      f"the most suspect 10% hold {cap[0.10]:.0%} (a random set holds 5% and 10%)")

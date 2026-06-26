"""Fit per-tile Bradley-Terry latent organization scores from weighted pairs.

Logistic regression with a +1/-1 design matrix (col = tile). The fitted
coefficient for each tile is its latent organization score z. These distill all
900 weighted pairwise outcomes into 301 per-tile scalars (a regression target).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ROOT = Path("dataset"); WORK = Path("working")


def fit_bt(tr, tiles, C=1.0, weight=True):
    idx = {t: i for i, t in enumerate(tiles)}
    n = len(tiles)
    rows, cols, vals = [], [], []
    for r, (_, row) in enumerate(tr.iterrows()):
        li = idx[row.left_image_path]; ri = idx[row.right_image_path]
        rows += [r, r]; cols += [li, ri]; vals += [1.0, -1.0]
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(tr), n))
    y = tr.left_higher_organization.values.astype(int)
    w = tr.pair_weight.values.astype(float) if weight else None
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    z = clf.coef_.ravel()
    return z, idx


if __name__ == "__main__":
    tr = pd.read_csv(ROOT / "train.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    for C in [0.3, 1.0, 3.0, 10.0]:
        z, idx = fit_bt(tr, tiles, C=C)
        # reconstruction acc on training pairs
        L = np.array([idx[t] for t in tr.left_image_path])
        R = np.array([idx[t] for t in tr.right_image_path])
        logit = z[L] - z[R]
        p = 1 / (1 + np.exp(-logit))
        acc = ((p > 0.5).astype(int) == tr.left_higher_organization.values).mean()
        # weighted log loss on train (fit quality)
        w = tr.pair_weight.values
        pl = np.clip(p, 1e-6, 1 - 1e-6)
        ll = -(tr.left_higher_organization.values * np.log(pl) + (1 - tr.left_higher_organization.values) * np.log(1 - pl))
        wll = 100 * (w * ll).sum() / w.sum()
        print(f"C={C:5}: train_acc={acc:.3f} train_wll={wll:.2f}  z range [{z.min():.2f},{z.max():.2f}] std={z.std():.2f}")

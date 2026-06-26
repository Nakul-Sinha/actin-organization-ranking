"""Tile-disjoint CV for classical Bradley-Terry baselines.

Protocol mirrors the test scenario (all-unseen tiles):
  repeated random TILE splits; train on pairs with BOTH endpoints in train-tiles,
  evaluate weighted log loss on pairs with BOTH endpoints in val-tiles.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("dataset"); WORK = Path("working")


def load():
    tr = pd.read_csv(ROOT / "train.csv")
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    tiles = list(dz["tiles"]); feats = dz["feats"].astype(np.float64)
    keys = list(dz["keys"])
    idx = {t: i for i, t in enumerate(tiles)}
    # standardize per-tile features (z-score over train tiles)
    mu = feats.mean(0); sd = feats.std(0) + 1e-8
    featsz = (feats - mu) / sd
    return tr, tiles, featsz, idx, keys, (mu, sd)


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


def pair_matrix(tr, featsz, idx):
    L = np.array([idx[t] for t in tr.left_image_path])
    R = np.array([idx[t] for t in tr.right_image_path])
    X = featsz[L] - featsz[R]
    y = tr.left_higher_organization.values.astype(int)
    w = tr.pair_weight.values.astype(float)
    return X, y, w, L, R


def cv_linear(tr, featsz, idx, C=1.0, n_rep=10, val_frac=0.25, seed0=0):
    X, y, w, L, R = pair_matrix(tr, featsz, idx)
    n_tiles = featsz.shape[0]
    scores = []
    accs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(seed0 + rep)
        perm = rng.permutation(n_tiles)
        n_val = int(n_tiles * val_frac)
        val_tiles = set(perm[:n_val].tolist())
        tr_mask = np.array([(l not in val_tiles and r not in val_tiles) for l, r in zip(L, R)])
        va_mask = np.array([(l in val_tiles and r in val_tiles) for l, r in zip(L, R)])
        if va_mask.sum() < 10 or tr_mask.sum() < 50:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr_mask], y[tr_mask], sample_weight=w[tr_mask])
        p = clf.predict_proba(X[va_mask])[:, 1]
        scores.append(metric(y[va_mask], p, w[va_mask]))
        accs.append(((p > 0.5).astype(int) == y[va_mask]).mean())
    return np.array(scores), np.array(accs)


def cv_gbm(tr, featsz, idx, n_rep=10, val_frac=0.25, seed0=0, **gbm):
    X, y, w, L, R = pair_matrix(tr, featsz, idx)
    n_tiles = featsz.shape[0]
    scores = []; accs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(seed0 + rep)
        perm = rng.permutation(n_tiles)
        n_val = int(n_tiles * val_frac)
        val_tiles = set(perm[:n_val].tolist())
        tr_mask = np.array([(l not in val_tiles and r not in val_tiles) for l, r in zip(L, R)])
        va_mask = np.array([(l in val_tiles and r in val_tiles) for l, r in zip(L, R)])
        if va_mask.sum() < 10 or tr_mask.sum() < 50:
            continue
        # antisymmetric augmentation: add swapped pairs
        Xtr = np.vstack([X[tr_mask], -X[tr_mask]])
        ytr = np.concatenate([y[tr_mask], 1 - y[tr_mask]])
        wtr = np.concatenate([w[tr_mask], w[tr_mask]])
        clf = HistGradientBoostingClassifier(**gbm)
        clf.fit(Xtr, ytr, sample_weight=wtr)
        # symmetrized prediction
        p1 = clf.predict_proba(X[va_mask])[:, 1]
        p2 = clf.predict_proba(-X[va_mask])[:, 1]
        p = 0.5 * (p1 + (1 - p2))
        scores.append(metric(y[va_mask], p, w[va_mask]))
        accs.append(((p > 0.5).astype(int) == y[va_mask]).mean())
    return np.array(scores), np.array(accs)


if __name__ == "__main__":
    tr, tiles, featsz, idx, keys, norm = load()
    print(f"pairs={len(tr)} tiles={len(tiles)} feat_dim={featsz.shape[1]}")
    print("baseline constant 0.5 ->", metric(tr.left_higher_organization.values,
          np.full(len(tr), 0.5), tr.pair_weight.values))
    print()
    print("== Linear Bradley-Terry (LogReg on feature diff) ==")
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        s, a = cv_linear(tr, featsz, idx, C=C, n_rep=12)
        print(f"  C={C:<5} loss={s.mean():6.2f} +/- {s.std():4.2f}   acc={a.mean():.3f}   (n_val_splits={len(s)})")
    print()
    print("== GBM on feature diff (antisym aug) ==")
    for lr, leaves, depth in [(0.05, 15, 3), (0.05, 31, 4), (0.03, 15, 3)]:
        s, a = cv_gbm(tr, featsz, idx, n_rep=12, learning_rate=lr,
                      max_leaf_nodes=leaves, max_depth=depth, max_iter=300,
                      l2_regularization=1.0, early_stopping=False, random_state=0)
        print(f"  lr={lr} leaves={leaves} depth={depth}  loss={s.mean():6.2f} +/- {s.std():4.2f}  acc={a.mean():.3f}")

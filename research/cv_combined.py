"""Tile-disjoint CV combining hand-crafted morphology + frozen deep features.

Linear Bradley-Terry (LogReg on per-tile feature DIFF) => antisymmetric,
globally-consistent per-tile scores. Optional PCA on deep feats to denoise.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ROOT = Path("dataset"); WORK = Path("working")


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


def load_hand():
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    return list(dz["tiles"]), dz["feats"].astype(np.float64)


def load_deep(bb, norm="perimg"):
    dz = np.load(WORK / f"deepfeat_{bb}_{norm}.npz", allow_pickle=True)
    return list(dz["tr_tiles"]), dz["ftr"].astype(np.float64)


def build_features(use_hand=True, deep_bbs=(), norm="perimg", pca_deep=64):
    tr = pd.read_csv(ROOT / "train.csv")
    tiles_h, fh = load_hand()
    idx = {t: i for i, t in enumerate(tiles_h)}
    blocks = []
    names = []
    if use_hand:
        blocks.append(("hand", fh)); names.append("hand")
    for bb in deep_bbs:
        tiles_d, fd = load_deep(bb, norm)
        assert tiles_d == tiles_h, "tile order mismatch"
        blocks.append((bb, fd)); names.append(bb)
    return tr, tiles_h, idx, blocks, names


def assemble(blocks, pca_deep, fit_mask=None):
    """z-score each block (using fit_mask tiles for stats); PCA deep blocks."""
    cols = []
    for name, F in blocks:
        ref = F if fit_mask is None else F[fit_mask]
        mu = ref.mean(0); sd = ref.std(0) + 1e-8
        Fz = (F - mu) / sd
        if name != "hand" and pca_deep and Fz.shape[1] > pca_deep:
            p = PCA(n_components=pca_deep, random_state=0)
            p.fit(Fz[fit_mask] if fit_mask is not None else Fz)
            Fz = p.transform(Fz)
        cols.append(Fz)
    return np.concatenate(cols, 1)


def cv(tr, idx, blocks, C=0.01, n_rep=12, val_frac=0.25, seed0=0, pca_deep=64):
    L = np.array([idx[t] for t in tr.left_image_path])
    R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int)
    w = tr.pair_weight.values.astype(float)
    nT = len(idx)
    scores, accs = [], []
    for rep in range(n_rep):
        rng = np.random.RandomState(seed0 + rep)
        perm = rng.permutation(nT)
        val = set(perm[:int(nT * val_frac)].tolist())
        tile_train = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() < 10:
            continue
        # fit feature normalization/PCA on TRAIN tiles only (no leakage)
        Fall = assemble(blocks, pca_deep, fit_mask=tile_train)
        X = Fall[L] - Fall[R]
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(X[trm], y[trm], sample_weight=w[trm])
        p = clf.predict_proba(X[vam])[:, 1]
        scores.append(metric(y[vam], p, w[vam]))
        accs.append(((p > 0.5).astype(int) == y[vam]).mean())
    return np.array(scores), np.array(accs)


if __name__ == "__main__":
    configs = [
        dict(use_hand=True, deep_bbs=()),
        dict(use_hand=False, deep_bbs=("resnet18",)),
        dict(use_hand=False, deep_bbs=("efficientnet_b0",)),
        dict(use_hand=False, deep_bbs=("convnext_nano",)),
        dict(use_hand=False, deep_bbs=("resnet18", "efficientnet_b0", "convnext_nano")),
        dict(use_hand=True, deep_bbs=("resnet18", "efficientnet_b0", "convnext_nano")),
    ]
    for cfg in configs:
        tr, tiles, idx, blocks, names = build_features(**cfg)
        best = None
        for C in [0.003, 0.01, 0.03, 0.1]:
            for pca in [32, 64]:
                s, a = cv(tr, idx, blocks, C=C, n_rep=12, pca_deep=pca)
                if best is None or s.mean() < best[0]:
                    best = (s.mean(), s.std(), a.mean(), C, pca)
        print(f"{'+'.join(names):45s} loss={best[0]:6.2f}+/-{best[1]:4.2f} acc={best[2]:.3f} (C={best[3]}, pca={best[4]})")

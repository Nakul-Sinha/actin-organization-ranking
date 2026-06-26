"""How much GENUINE (confound-orthogonal) signal exists?

Compare tile-disjoint OOF of: (a) all features, (b) confounds only,
(c) features residualized against confounds. The matched confounds (intensity,
gradient, texture, coverage) are spurious shortcuts that don't transfer to the
matched test set; (c) is the signal that might actually transfer.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path("dataset"); WORK = Path("working")

dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
tiles = list(dz["tiles"]); F = dz["feats"].astype(float); keys = list(dz["keys"])
idx = {t: i for i, t in enumerate(tiles)}
tr = pd.read_csv(ROOT / "train.csv")
L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)

# confound basis: intensity / gradient / texture / coverage descriptors
CONF = ["int_mean", "int_std", "int_max", "dark_frac", "nz_frac", "grad_mean", "grad_std",
        "glcm_contrast_mean", "glcm_contrast_std", "glcm_dissimilarity_mean",
        "shannon_entropy", "fg_frac", "blob_n", "cc_area_sum"]
conf_idx = [keys.index(k) for k in CONF if k in keys]
print("confound basis:", [keys[i] for i in conf_idx])


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return min(100.0, 100.0 * (w * (-(y*np.log(p)+(1-y)*np.log(1-p)))).sum()/w.sum())


def residualize(Fmat, conf_cols, fit_mask):
    """Remove linear confound component from each feature (fit on fit_mask tiles)."""
    C = Fmat[:, conf_cols]
    Cz = (C - C[fit_mask].mean(0)) / (C[fit_mask].std(0) + 1e-8)
    Cz = np.c_[Cz, np.ones(len(Cz))]
    out = Fmat.copy()
    for j in range(Fmat.shape[1]):
        if j in conf_cols:
            out[:, j] = 0.0
            continue
        coef, *_ = np.linalg.lstsq(Cz[fit_mask], Fmat[fit_mask, j], rcond=None)
        out[:, j] = Fmat[:, j] - Cz @ coef
    return out


def oof(Fmat, mode, C=0.01, n_rep=20, val_frac=0.25, resid=False):
    nT = len(tiles); scores = []; accs = []
    for rep in range(n_rep):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(nT); val = set(perm[:int(nT*val_frac)].tolist())
        ttrain = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() < 8: continue
        Fuse = residualize(Fmat, conf_idx, ttrain) if resid else Fmat
        if mode == "conf_only":
            Fuse = Fuse * 0; Fuse[:, conf_idx] = Fmat[:, conf_idx]
        mu = Fuse[ttrain].mean(0); sd = Fuse[ttrain].std(0) + 1e-8
        Fz = (Fuse - mu) / sd
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit((Fz[L]-Fz[R])[trm], y[trm], sample_weight=w[trm])
        p = clf.predict_proba((Fz[L]-Fz[R])[vam])[:, 1]
        scores.append(metric(y[vam], p, w[vam]))
        accs.append(((p>0.5).astype(int)==y[vam]).mean())
    return np.mean(scores), np.std(scores), np.mean(accs)


print("\nTile-disjoint OOF (these are still train-distribution; confound rows are the SPURIOUS part):")
for C in [0.003, 0.01, 0.03]:
    s, sd, a = oof(F, "all", C=C)
    print(f"  ALL feats           C={C:<5} loss={s:.2f}+/-{sd:.2f} acc={a:.3f}")
print()
s, sd, a = oof(F, "conf_only", C=0.03)
print(f"  CONFOUNDS ONLY            loss={s:.2f}+/-{sd:.2f} acc={a:.3f}   <- spurious shortcut magnitude")
print()
for C in [0.01, 0.03, 0.1]:
    s, sd, a = oof(F, "resid", C=C, resid=True)
    print(f"  CONFOUND-RESIDUALIZED C={C:<5} loss={s:.2f}+/-{sd:.2f} acc={a:.3f}   <- genuine transferable signal?")

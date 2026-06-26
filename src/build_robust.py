"""Confound-ORTHOGONAL Bradley-Terry submission with honest signal estimate.

Root-cause fix for the 87.6 failure (model rode the matched gradient/texture
confounds + was overconfident). Two-stage confound removal:
  1. residualize per-tile features against a confound basis (model can't learn them);
  2. post-hoc orthogonalize the pair logits against the pair's confound DIFFERENCES
     (the brief guarantees these are matched => uninformative in test).
Then estimate the surviving (genuine) signal with tile-disjoint OOF that applies
the SAME orthogonalization in-fold, and calibrate conservatively.
"""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path("dataset"); WORK = Path("working")

# feature names treated as confounds (zeroed before modeling)
CONF_FEATS = ["int_mean", "int_std", "int_max", "dark_frac", "nz_frac", "grad_mean",
              "grad_std", "glcm_contrast_mean", "glcm_contrast_std",
              "glcm_dissimilarity_mean", "shannon_entropy", "fg_frac", "blob_n",
              "cc_area_sum", "nbright_frac"]


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return min(100.0, 100.0 * (w * (-(y*np.log(p)+(1-y)*np.log(1-p)))).sum()/w.sum())


def img_confounds(rel, root):
    a = np.array(Image.open(root / rel), dtype=np.float64)
    gx, gy = np.gradient(a)
    lap = np.abs(np.gradient(np.gradient(a, axis=0), axis=0)).mean()
    return np.array([a.mean(), a.std(), np.percentile(a, 99), a.max(), a.sum()/1e4,
                     (a > 0).mean(), (a > 50).mean(),
                     np.abs(gx).mean()+np.abs(gy).mean(), lap])


def residualize(F, conf_cols, fit_mask):
    C = F[:, conf_cols]
    Cz = (C - C[fit_mask].mean(0)) / (C[fit_mask].std(0) + 1e-8)
    Cz = np.c_[Cz, np.ones(len(Cz))]
    out = F.copy()
    for j in range(F.shape[1]):
        if j in conf_cols:
            out[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Cz[fit_mask], F[fit_mask, j], rcond=None)
        out[:, j] = F[:, j] - Cz @ coef
    return out


def orthogonalize(logit, Dconf):
    """Remove linear component of logit explained by confound differences."""
    D = np.c_[Dconf, np.ones(len(Dconf))]
    beta, *_ = np.linalg.lstsq(D, logit, rcond=None)
    return logit - D @ beta


def load():
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    return (tr, te, list(dz["tiles"]), dz["feats"].astype(float),
            list(de["tiles"]), de["feats"].astype(float), list(dz["keys"]))


def pair_confdiff(pairs_left, pairs_right, confmap):
    return np.array([confmap[l] - confmap[r] for l, r in zip(pairs_left, pairs_right)])


def oof_signal(Ftr, tiles, tr, conf_idx, confmap, C=0.005, n_rep=40, val_frac=0.25):
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    nT = len(tiles)
    LG, YY, WW = [], [], []
    for rep in range(n_rep):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(nT); val = set(perm[:int(nT*val_frac)].tolist())
        ttrain = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() < 8: continue
        Fr = residualize(Ftr, conf_idx, ttrain)
        mu = Fr[ttrain].mean(0); sd = Fr[ttrain].std(0)+1e-8
        Fz = (Fr-mu)/sd
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit((Fz[L]-Fz[R])[trm], y[trm], sample_weight=w[trm])
        s = Fz @ clf.coef_.ravel()
        sdt = (s[L[trm]]-s[R[trm]]).std()+1e-8
        lg = (s[L[vam]]-s[R[vam]])/sdt
        lg = orthogonalize(lg, Dall[vam])           # SAME orthogonalization in-fold
        LG.append(lg); YY.append(y[vam]); WW.append(w[vam])
    lg = np.concatenate(LG); y2 = np.concatenate(YY); w2 = np.concatenate(WW)
    lg = lg / (np.std(lg) + 1e-8)
    acc = ((lg > 0).astype(int) == y2).mean()
    bestT, bestL = 1.0, 1e9
    for T in np.linspace(0.5, 15, 150):
        l = metric(y2, 1/(1+np.exp(-lg/T)), w2)
        if l < bestL: bestL, bestT = l, T
    return acc, bestT, bestL, (lg, y2, w2)


def main(C=0.005, safety=2.0, clip=0.18):
    tr, te, tiles, Ftr, te_tiles, Fte, keys = load()
    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    print("[robust] computing image confounds...")
    confmap = {t: img_confounds(t, ROOT) for t in set(tiles) | set(te_tiles)}
    acc, T_oof, L_oof, (lg_oof, y_oof, w_oof) = oof_signal(Ftr, tiles, tr, conf_idx, confmap, C=C)
    T_safe = T_oof * safety
    shipped_oof = metric(y_oof, np.clip(1/(1+np.exp(-lg_oof/T_safe)), 0.5-clip, 0.5+clip), w_oof)
    print(f"[robust] OOF confound-ORTHOGONAL: acc={acc:.3f} best-T={T_oof:.2f} best-loss={L_oof:.2f}")
    print(f"         shipped calibration (T_safe={T_safe:.2f}, clip={clip}): OOF loss={shipped_oof:.2f}")
    print(f"         (acc>0.5 and loss<69.3 => genuine signal survives confound removal)")

    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Fr = residualize(Ftr, conf_idx, np.ones(len(tiles), bool))
    # residualize test with train-fit confound model
    Cc = Ftr[:, conf_idx]; mC = Cc.mean(0); sC = Cc.std(0)+1e-8
    Ctr = np.c_[(Cc-mC)/sC, np.ones(len(Cc))]; Cte = np.c_[(Fte[:, conf_idx]-mC)/sC, np.ones(len(Fte))]
    Fte_r = Fte.copy()
    for j in range(Fte.shape[1]):
        if j in conf_idx: Fte_r[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Ctr, Ftr[:, j], rcond=None); Fte_r[:, j] = Fte[:, j] - Cte @ coef
    mu = Fr.mean(0); sd = Fr.std(0)+1e-8; Fz = (Fr-mu)/sd; Fz_te = (Fte_r-mu)/sd
    clf = LogisticRegression(C=C, max_iter=4000)
    clf.fit(Fz[L]-Fz[R], y, sample_weight=w); coef = clf.coef_.ravel()
    s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L]-s_tr[R]).std()+1e-8
    teidx = {t: i for i, t in enumerate(te_tiles)}
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    lg_te = (s_te[Lt]-s_te[Rt])/sdt
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)
    lg_te = orthogonalize(lg_te, Dte)                 # guarantee confound-orthogonality on test
    lg_te = lg_te / (np.std(lg_te) + 1e-8)
    T_safe = T_oof * safety
    prob = np.clip(1/(1+np.exp(-lg_te / T_safe)), 0.5-clip, 0.5+clip)
    return tr, te, te_tiles, prob, Dte, dict(acc=acc, T_oof=T_oof, L_oof=L_oof, T_safe=T_safe)


if __name__ == "__main__":
    tr, te, te_tiles, prob, Dte, info = main()
    # verify confound orthogonality on test
    print("[robust] test-pred confound corr (want ~0):",
          {n: round(float(np.corrcoef(Dte[:, i], prob)[0, 1]), 3)
           for i, n in enumerate(["mean","std","p99","max","total","cov","nbright","grad","lap"])})
    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission_robust.csv", index=False)
    print(f"[robust] wrote submission_robust.csv prob[min={prob.min():.3f} mean={prob.mean():.3f} "
          f"max={prob.max():.3f} std={prob.std():.3f}]")

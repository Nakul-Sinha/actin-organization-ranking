"""Proxy gate for SSL features: do they transfer better than hand-crafted (68.4)?

Uses the SAME simulated-shift splits derived from the hand-crafted train-vs-test
direction (conservative, the 'real' shift), evaluates SSL features with confound
orthogonalization. Also reports the train/test separability of SSL features (lower
= more in-distribution = the whole point).
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from build_robust import img_confounds, orthogonalize, pair_confdiff, metric

ROOT = Path("dataset"); WORK = Path("working")


def residualize_ext(F, Cmat, fit_mask):
    Cz = (Cmat - Cmat[fit_mask].mean(0)) / (Cmat[fit_mask].std(0) + 1e-8)
    Cz = np.c_[Cz, np.ones(len(Cz))]
    out = F.copy()
    for j in range(F.shape[1]):
        coef, *_ = np.linalg.lstsq(Cz[fit_mask], F[fit_mask, j], rcond=None)
        out[:, j] = F[:, j] - Cz @ coef
    return out


def load_hand():
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    return list(dz["tiles"]), dz["feats"].astype(float), list(de["tiles"]), de["feats"].astype(float)


def directions(Ftr, Fte, n=6):
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
    out = []; rng = np.random.RandomState(0)
    for _ in range(n):
        s = rng.choice(len(Xz), int(0.8 * len(Xz)), replace=False)
        out.append((LogisticRegression(C=0.05, max_iter=2000).fit(Xz[s], lab[s]).coef_.ravel(), mu, sd))
    return out


def run(C=0.02):
    dz = np.load(WORK / "ssl_feats.npz", allow_pickle=True)
    ssl_tr_tiles = list(dz["tr_tiles"]); Ftr = dz["ftr"].astype(float); Fte = dz["fte"].astype(float)
    hand_tiles, Hand, _, Hte = load_hand()
    assert hand_tiles == ssl_tr_tiles, "tile order mismatch"
    tr = pd.read_csv(ROOT / "train.csv")
    tiles = ssl_tr_tiles
    confmap = {t: img_confounds(t, ROOT) for t in tiles}
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    Cmat = np.array([confmap[t] for t in tiles])
    nT = len(tiles)

    # train/test separability (lower => more in-distribution)
    def auc(F, Ft):
        X = np.vstack([F, Ft]); lab = np.r_[np.zeros(len(F)), np.ones(len(Ft))]
        mu = X.mean(0); sd = X.std(0) + 1e-8
        return cross_val_score(LogisticRegression(C=0.1, max_iter=2000), (X-mu)/sd, lab, cv=5, scoring="roc_auc").mean()
    print(f"train/test separability AUC:  hand-crafted={auc(Hand,Hte):.3f}   SSL={auc(Ftr,Fte):.3f}  (lower=better)")

    # splits from HAND-CRAFTED direction (the conservative 'real' shift)
    splits = []
    for cf, mu, sd in directions(Hand, Hte):
        proj = ((Hand - mu) / sd) @ cf; order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True
            splits.append((tl, vl))

    def eval_feats(F, tag, C=0.02, ortho=True):
        accs, losses = [], []
        for tl, vl in splits:
            trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
            vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
            if trm.sum() < 30 or vam.sum() < 20: continue
            Fr = residualize_ext(F, Cmat, tl)
            Fz = (Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8)
            clf = LogisticRegression(C=C, max_iter=4000)
            clf.fit((Fz[L]-Fz[R])[trm], y[trm], sample_weight=w[trm])
            s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]]-s[R[trm]]).std()+1e-8
            lg = (s[L[vam]]-s[R[vam]])/sdt
            if ortho: lg = orthogonalize(lg, Dall[vam])
            lg /= (np.std(lg)+1e-8)
            accs.append(((lg>0).astype(int)==y[vam]).mean())
            losses.append(min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5,12,60)))
        print(f"  {tag:30s} shift-acc={np.mean(accs):.3f}  shift-loss={np.mean(losses):.2f}")
        return np.mean(losses)

    def zscore(F):
        return (F - F.mean(0)) / (F.std(0) + 1e-8)
    combo = np.concatenate([zscore(Ftr), zscore(Hand)], 1)
    print("On the SAME hand-crafted-derived shift splits (constant=69.31; hand-crafted~68.4):")
    for tag, F in [("hand", Hand), ("SSL", Ftr), ("SSL+hand", combo)]:
        for C2 in [0.005, 0.01, 0.02]:
            eval_feats(F, f"{tag} C={C2}", C=C2)


if __name__ == "__main__":
    run()

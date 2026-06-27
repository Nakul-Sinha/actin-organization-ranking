"""Find the SSL sweet spot: low train/test separability + best proxy transfer.

Heavy SSL overfits 490 tiles (separability up -> transfer down). Sweep epoch count
(incl. frozen ImageNet) with seed-ensembled features, applying the full transfer
processing (rank-norm within group + confound-orth) on the hand-crafted-derived
shift splits. Save the best features.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import ssl_train as S
from build_robust import img_confounds, orthogonalize, pair_confdiff, metric
from ssl_proxy import residualize_ext, directions

ROOT = Path("dataset"); WORK = Path("working")


def rank_norm(F, ref):
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        sv = np.sort(ref[:, j]); out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out


def auc(F, Ft):
    X = np.vstack([F, Ft]); lab = np.r_[np.zeros(len(F)), np.ones(len(Ft))]
    mu = X.mean(0); sd = X.std(0) + 1e-8
    return cross_val_score(LogisticRegression(C=0.1, max_iter=2000), (X - mu) / sd, lab, cv=5, scoring="roc_auc").mean()


def main():
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True); de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    tiles = list(dz["tiles"]); Hand = dz["feats"].astype(float); Hte = de["feats"].astype(float)
    tr = pd.read_csv(ROOT / "train.csv")
    confmap = {t: img_confounds(t, ROOT) for t in tiles}
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    Cmat = np.array([confmap[t] for t in tiles]); nT = len(tiles)
    splits = []
    for cf, mu, sd in directions(Hand, Hte):
        proj = ((Hand - mu) / sd) @ cf; order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True; splits.append((tl, vl))

    def proxy(Ftr, rank=True, C=0.01):
        accs, losses = [], []
        for tl, vl in splits:
            trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
            vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
            if trm.sum() < 30 or vam.sum() < 20: continue
            F = Ftr.copy()
            if rank: F[tl] = rank_norm(Ftr, Ftr[tl])[tl]; F[vl] = rank_norm(Ftr, Ftr[vl])[vl]
            Fr = residualize_ext(F, Cmat, tl)
            Fz = (Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8)
            clf = LogisticRegression(C=C, max_iter=4000)
            clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
            s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
            lg = orthogonalize((s[L[vam]] - s[R[vam]]) / sdt, Dall[vam]); lg /= (np.std(lg) + 1e-8)
            accs.append(((lg > 0).astype(int) == y[vam]).mean())
            losses.append(min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5, 12, 60)))
        return np.mean(accs), np.mean(losses)

    trt, tet = S.all_tiles(); tiles_all = trt + tet; imgs = S.preload(tiles_all)
    print(f"{'config':18s} sep-AUC  proxy-acc proxy-loss")
    a, l = proxy(Hand); print(f"{'hand-crafted':18s} {auc(Hand,Hte):.3f}    {a:.3f}     {l:.2f}")
    best = (l, "hand", None, None)
    for epochs in [0, 10, 25, 50]:
        ns = 1 if epochs == 0 else 3
        ftr = 0; fte = 0
        for seed in range(ns):
            a1, a2 = S.train_features(pretrained=True, epochs=epochs, seed=seed, imgs=imgs, tiles=tiles_all)
            ftr = ftr + a1; fte = fte + a2
        ftr /= ns; fte /= ns
        sep = auc(ftr, fte)
        for C in [0.005, 0.01, 0.02]:
            pa, pl = proxy(ftr, rank=True, C=C)
            tag = f"SSL ep{epochs} C={C}"
            print(f"{tag:18s} {sep:.3f}    {pa:.3f}     {pl:.2f}")
            if pl < best[0]:
                best = (pl, tag, ftr.copy(), fte.copy())
    print(f"\nBEST: {best[1]} proxy-loss={best[0]:.2f}")
    if best[2] is not None:
        np.savez(WORK / "ssl_feats.npz", tr_tiles=np.array(trt), te_tiles=np.array(tet), ftr=best[2], fte=best[3])
        print("saved best features to ssl_feats.npz")


if __name__ == "__main__":
    main()

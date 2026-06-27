"""H100 fleet experiment: is ANY model ROBUSTLY positive across shift directions?

The single-direction proxy misled me on SSL (proxy 64.4 -> real 74). So here I
build shift directions from THREE feature spaces (hand-crafted, SSL, raw-pixel) and
require a model to transfer (beat the 69.31 constant) across ALL of them. Models:
hand-crafted linear, SSL linear, and a fresh CNN ensemble (resnet18 regression to the
BT latent, confound-orthogonalized). Only a robustly-positive model is worth shipping.
"""
import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "research")
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch
from sklearn.linear_model import LogisticRegression
from build_robust import img_confounds, orthogonalize, pair_confdiff, metric
from build_ssl import rank_norm, residualize_ext
from bt_scores import fit_bt
import cnn_regress as CR

ROOT = Path("dataset"); WORK = Path("working"); DEV = CR.DEV
CONST = 69.31


def load():
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    hz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    assert list(hz["tiles"]) == tiles
    Hand = hz["feats"].astype(float)
    sz = np.load(WORK / "ssl_feats.npz", allow_pickle=True)
    assert list(sz["tr_tiles"]) == tiles
    SSL = sz["ftr"].astype(float)
    # raw-pixel descriptor: 16x16 mean-pooled
    px = np.zeros((len(tiles), 64))
    for i, t in enumerate(tiles):
        a = np.array(Image.open(ROOT / t), np.float32) / 255.0
        px[i] = a.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel()
    return tr, te, tiles, Hand, SSL, px


def directions(F, lab_tr_te, n=4):
    """train/test classifier directions in feature space F (lab: 0 train,1 test)."""
    mu = F.mean(0); sd = F.std(0) + 1e-8; Fz = (F - mu) / sd
    out = []; rng = np.random.RandomState(1)
    for _ in range(n):
        s = rng.choice(len(Fz), int(0.85 * len(Fz)), replace=False)
        out.append((LogisticRegression(C=0.05, max_iter=2000).fit(Fz[s], lab_tr_te[s]).coef_.ravel(), mu, sd))
    return out


def main():
    t0 = time.time()
    tr, te, tiles, Hand, SSL, PX = load()
    # need test feats too for direction classifiers (stacked train+test)
    hz = np.load(WORK / "feats_train.npz", allow_pickle=True); he = np.load(WORK / "feats_test.npz", allow_pickle=True)
    sz = np.load(WORK / "ssl_feats.npz", allow_pickle=True)
    Hte = he["feats"].astype(float); Ste = sz["fte"].astype(float)
    te_tiles = list(sz["te_tiles"])
    pxte = np.zeros((len(te_tiles), 64))
    for i, t in enumerate(te_tiles):
        a = np.array(Image.open(ROOT / t), np.float32) / 255.0
        pxte[i] = a.reshape(8, 16, 8, 16).mean(axis=(1, 3)).ravel()

    confmap = {t: img_confounds(t, ROOT) for t in tiles}
    Cmat = np.array([confmap[t] for t in tiles])
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    nT = len(tiles)
    imgs = CR.preload(tiles)  # for CNN

    # build splits from 3 feature spaces
    specs = [("hand", Hand, Hte), ("ssl", SSL, Ste), ("pixel", PX, pxte)]
    splits = []
    for name, Ftr, Fte in specs:
        Fall = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
        for cf, mu, sd in directions(Fall, lab, n=2):
            proj = ((Ftr - mu) / sd) @ cf; order = np.argsort(proj)
            for frac in (0.40,):
                k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
                tl[order[:k]] = True; vl[order[-k:]] = True
                splits.append((name, tl, vl))
    print(f"{len(splits)} shift splits from {len(specs)} feature spaces ({time.time()-t0:.0f}s)")

    def lin_shift(F, tl, vl, C=0.01, rank=True):
        Fp = F.copy()
        if rank: Fp[tl] = rank_norm(F, F[tl])[tl]; Fp[vl] = rank_norm(F, F[vl])[vl]
        Fr = residualize_ext(Fp, Cmat, tl); Fz = (Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8)
        trm = (tl[L] & tl[R]); vam = (vl[L] & vl[R])
        clf = LogisticRegression(C=C, max_iter=4000); clf.fit((Fz[L]-Fz[R])[trm], y[trm], sample_weight=w[trm])
        s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]]-s[R[trm]]).std()+1e-8
        lg = orthogonalize((s[L[vam]]-s[R[vam]])/sdt, Dall[vam]); lg /= (np.std(lg)+1e-8)
        return min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5,12,40))

    def cnn_shift(tl, vl):
        trm = (tl[L] & tl[R]); vam = (vl[L] & vl[R])
        sub = tr[trm].reset_index(drop=True)
        st = sorted(set(sub.left_image_path)|set(sub.right_image_path))
        zs, si = fit_bt(sub, st, C=0.5); z = np.array([zs[si[t]] for t in st], np.float32); z=(z-z.mean())/(z.std()+1e-8)
        tids = [idx[t] for t in st]
        vp = (L[vam], R[vam], y[vam], w[vam])
        zhat = 0
        for sd in range(2):
            m,_ = CR.train_one(imgs, tids, z, np.ones(len(st),np.float32), vp, kind="resnet18", drop=0.5, epochs=50, lr=2e-3, wd=3e-2, seed=sd)
            zhat = zhat + CR.predict_z(m, imgs, tta=True)
        zhat/=2
        lg = orthogonalize(zhat[L[vam]]-zhat[R[vam]], Dall[vam]); lg/=(np.std(lg)+1e-8)
        return min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5,12,40))

    models = {"hand": lambda tl,vl: lin_shift(Hand,tl,vl),
              "ssl": lambda tl,vl: lin_shift(SSL,tl,vl),
              "ssl+hand": lambda tl,vl: 0.5*0,  # placeholder
              "cnn": cnn_shift}
    # ssl+hand handled separately (concat)
    def sslhand_shift(tl,vl):
        F = np.concatenate([(SSL-SSL.mean(0))/(SSL.std(0)+1e-8),(Hand-Hand.mean(0))/(Hand.std(0)+1e-8)],1)
        return lin_shift(F,tl,vl)
    models["ssl+hand"] = sslhand_shift

    print(f"{'model':10s} | per-space worst-case shift-loss | overall mean / worst / frac<69.3")
    results = {}
    for mname, fn in models.items():
        losses = {}; alll=[]
        for name, tl, vl in splits:
            l = fn(tl, vl); losses.setdefault(name,[]).append(l); alll.append(l)
        per = {k: max(v) for k,v in losses.items()}
        alll = np.array(alll)
        print(f"{mname:10s} | " + "  ".join(f"{k}:{per[k]:.1f}" for k in ['hand','ssl','pixel']) +
              f" | mean {alll.mean():.2f} worst {alll.max():.2f} frac<const {np.mean(alll<CONST):.2f}  [{time.time()-t0:.0f}s]")
        results[mname]=alll
    print("\n(constant=69.31; want worst-case < 69.31 across ALL directions for robust transfer)")


if __name__ == "__main__":
    main()

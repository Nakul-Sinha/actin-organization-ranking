"""Robust simulated-shift proxy + tuning of the transfer-robust hand-crafted method.

Averages over several shift splits (vary the train-like/test-like fraction and
bootstrap the direction) and searches (keep_pct of least-shifted feats, C). Lets
me pick a config that TRANSFERS, validated locally, before spending a submission.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from build_robust import img_confounds, residualize, orthogonalize, pair_confdiff, metric, CONF_FEATS

ROOT = Path("dataset"); WORK = Path("working")


def load():
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    return (tr, te, list(dz["tiles"]), dz["feats"].astype(float),
            list(de["tiles"]), de["feats"].astype(float), list(dz["keys"]))


def rank_norm(F, ref):
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        sv = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out


def directions(Ftr, Fte, n=6):
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
    dirs = []
    rng = np.random.RandomState(0)
    for i in range(n):
        idxs = rng.choice(len(Xz), int(0.8 * len(Xz)), replace=False)
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Xz[idxs], lab[idxs])
        dirs.append((clf.coef_.ravel(), mu, sd))
    return dirs


def run():
    tr, te, tiles, Ftr, te_tiles, Fte, keys = load()
    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    confmap = {t: img_confounds(t, ROOT) for t in tiles}
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    smd = np.abs((Ftr.mean(0) - Fte.mean(0)) / (np.sqrt(0.5*(Ftr.var(0)+Fte.var(0))) + 1e-8))
    dirs = directions(Ftr, Fte)
    nT = len(tiles)

    splits = []
    for (cf, mu, sd) in dirs:
        proj = ((Ftr - mu) / sd) @ cf
        order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac)
            tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True
            splits.append((tl, vl))

    def eval_cfg(rank, keep_pct, C):
        keep = smd <= np.percentile(smd, keep_pct * 100)
        accs, losses = [], []
        for tl, vl in splits:
            trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
            vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
            if trm.sum() < 30 or vam.sum() < 20:
                continue
            F = Ftr.copy()
            if rank:
                F = F.copy(); F[tl] = rank_norm(Ftr, Ftr[tl])[tl]; F[vl] = rank_norm(Ftr, Ftr[vl])[vl]
            F = F.copy(); F[:, ~keep] = 0.0
            Fr = residualize(F, conf_idx, tl)
            Fz = (Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8)
            clf = LogisticRegression(C=C, max_iter=4000)
            clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
            s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
            lg = orthogonalize((s[L[vam]] - s[R[vam]]) / sdt, Dall[vam]); lg /= (np.std(lg) + 1e-8)
            accs.append(((lg > 0).astype(int) == y[vam]).mean())
            losses.append(min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5, 12, 60)))
        return np.mean(accs), np.mean(losses)

    print("rank keep%  C       shift-acc  shift-loss")
    best = None
    for rank in (False, True):
        for keep_pct in (0.3, 0.5, 0.75, 1.0):
            for C in (0.002, 0.005, 0.01):
                a, l = eval_cfg(rank, keep_pct, C)
                tag = f"{str(rank):5s} {keep_pct:<5} {C:<6}"
                print(f"  {tag}  {a:.3f}      {l:.2f}")
                if best is None or l < best[0]:
                    best = (l, a, rank, keep_pct, C)
    print(f"\nBEST: shift-loss={best[0]:.2f} acc={best[1]:.3f}  rank={best[2]} keep%={best[3]} C={best[4]}")
    print("(constant=69.31; current submission method A ~ anti-correlated ~73)")


if __name__ == "__main__":
    run()

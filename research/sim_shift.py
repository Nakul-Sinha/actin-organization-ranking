"""Local proxy for TRAIN->TEST transfer (the thing my OOF could not measure).

Use a train-vs-test classifier to find the shift direction, split TRAIN tiles into
"train-like" vs "test-like" halves along it, then train on train-like pairs and
evaluate on test-like pairs. This simulates the real distribution shift, so a
method that survives it here is more likely to transfer on the real test.

Compare feature-processing methods to find one that actually transfers.
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
    """Map each column to its quantile rank within ref (transductive domain adapt)."""
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        order = np.argsort(ref[:, j])
        sv = ref[order, j]
        out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out


def shift_split(Ftr, Fte, tiles, frac=0.4, seed=0):
    """Direction separating train from test; split train tiles along it."""
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8
    clf = LogisticRegression(C=0.05, max_iter=3000).fit((X - mu) / sd, lab)
    proj = ((Ftr - mu) / sd) @ clf.coef_.ravel()
    order = np.argsort(proj)
    n = int(len(tiles) * frac)
    trainlike = set(order[:n].tolist())     # least test-like
    testlike = set(order[-n:].tolist())     # most test-like
    return trainlike, testlike


def eval_method(method, Ftr, tiles, tr, conf_idx, confmap, C=0.005):
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    D = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    losses, accs = [], []
    for seed in range(5):
        trainlike, testlike = shift_split(Ftr[:, :0] if False else Ftr, None, tiles, seed=seed) if False else (None, None)
    # deterministic single split (direction needs test feats) handled by caller
    return None


def run():
    tr, te, tiles, Ftr, te_tiles, Fte, keys = load()
    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    confmap = {t: img_confounds(t, ROOT) for t in tiles}
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    trainlike, testlike = shift_split(Ftr, Fte, tiles, frac=0.42)
    trm = np.array([(l in trainlike and r in trainlike) for l, r in zip(L, R)])
    vam = np.array([(l in testlike and r in testlike) for l, r in zip(L, R)])
    print(f"simulated-shift split: train-like pairs={trm.sum()}  test-like pairs={vam.sum()}")
    tl = np.array([t in trainlike for t in range(len(tiles))])

    def fit_eval(Fz, tag, ortho=True):
        clf = LogisticRegression(C=0.005, max_iter=4000)
        clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
        s = Fz @ clf.coef_.ravel()
        sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
        lg = (s[L[vam]] - s[R[vam]]) / sdt
        if ortho:
            lg = orthogonalize(lg, Dall[vam])
        lg = lg / (np.std(lg) + 1e-8)
        acc = ((lg > 0).astype(int) == y[vam]).mean()
        bl = min(metric(y[vam], 1/(1+np.exp(-lg/T)), w[vam]) for T in np.linspace(0.5, 12, 80))
        print(f"  {tag:38s} shift-acc={acc:.3f}  best-shift-loss={bl:.2f}")
        return acc, bl

    # method A: standardize on train-like, confound-orthogonal
    mu = Ftr[tl].mean(0); sd = Ftr[tl].std(0) + 1e-8
    Fr = residualize(Ftr, conf_idx, tl)
    fit_eval((Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8), "A: confound-orth (current)")
    # method B: rank-normalized features (transductive domain adapt), confound-orth
    Frank = rank_norm(Ftr, Ftr)  # within train; for shift test we rank within each group below
    Frank_tl = rank_norm(Ftr, Ftr[tl]); Frank_vl = rank_norm(Ftr, Ftr[~tl])
    Fmix = Frank_tl.copy(); Fmix[~tl] = Frank_vl[~tl]
    Fr2 = residualize(Fmix, conf_idx, tl)
    fit_eval((Fr2 - Fr2[tl].mean(0)) / (Fr2[tl].std(0) + 1e-8), "B: rank-norm + confound-orth")
    # method C: transfer-stable feature selection (drop high train/test-shift feats), confound-orth
    smd = np.abs((Ftr.mean(0) - Fte.mean(0)) / (np.sqrt(0.5*(Ftr.var(0)+Fte.var(0))) + 1e-8))
    keep = smd < np.percentile(smd, 50)
    Fsel = Ftr.copy(); Fsel[:, ~keep] = 0.0
    Fr3 = residualize(Fsel, conf_idx, tl)
    fit_eval((Fr3 - Fr3[tl].mean(0)) / (Fr3[tl].std(0) + 1e-8), "C: drop-shifted-feats + confound-orth")
    # method D: rank-norm + drop-shifted + confound-orth
    Fsel2 = Fmix.copy(); Fsel2[:, ~keep] = 0.0
    Fr4 = residualize(Fsel2, conf_idx, tl)
    fit_eval((Fr4 - Fr4[tl].mean(0)) / (Fr4[tl].std(0) + 1e-8), "D: rank-norm + drop-shifted + orth")
    print("\n(constant 0.5 = 69.31; lower shift-loss = transfers better)")


if __name__ == "__main__":
    run()

"""Production pipeline: per-tile organization scorers -> calibrated ensemble.

Every base model outputs a per-tile score (linear BT, or a regressor predicting
the BT latent target z), so pairwise predictions are globally consistent across
the reused-tile graph. Each model is temperature-calibrated on honest OOF
(repeated tile-disjoint splits), then probabilities are averaged.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
sys.path.insert(0, str(Path(__file__).parent))
from bt_scores import fit_bt

ROOT = Path("dataset"); WORK = Path("working")


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


def fit_temp(y, logit, w, grid=None):
    if grid is None:
        grid = np.linspace(0.2, 8.0, 120)
    best_t, best_l = 1.0, 1e9
    for t in grid:
        l = metric(y, 1 / (1 + np.exp(-logit / t)), w)
        if l < best_l:
            best_l, best_t = l, t
    return best_t, best_l


# ---------- base per-tile scorers ----------
def score_linbt(Fz, L, R, y, w, train_mask, C=0.01):
    clf = LogisticRegression(C=C, max_iter=4000)
    X = Fz[L] - Fz[R]
    clf.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
    return Fz @ clf.coef_.ravel()


def score_reg(Fz, train_tiles, z_tiles, kind="ridge", **kw):
    Xtr = Fz[train_tiles]
    if kind == "ridge":
        m = Ridge(alpha=kw.get("alpha", 50.0))
    elif kind == "gbm":
        m = HistGradientBoostingRegressor(learning_rate=kw.get("lr", 0.05),
              max_leaf_nodes=kw.get("leaves", 15), max_depth=kw.get("depth", 3),
              l2_regularization=kw.get("l2", 5.0), max_iter=kw.get("iters", 300),
              min_samples_leaf=kw.get("msl", 20), random_state=0)
    elif kind == "et":
        m = ExtraTreesRegressor(n_estimators=400, max_features=0.4,
              min_samples_leaf=kw.get("msl", 8), random_state=0, n_jobs=-1)
    m.fit(Xtr, z_tiles)
    return m.predict(Fz)


MODELS = {
    "linbt_a": dict(kind="linbt", C=0.005),
    "linbt_b": dict(kind="linbt", C=0.01),
    "linbt_c": dict(kind="linbt", C=0.02),
    "ridge_a": dict(kind="ridge", alpha=80.0),
    "ridge_b": dict(kind="ridge", alpha=250.0),
    "et":      dict(kind="et", msl=8),
}


def optimize_weights(probs, y, w, n_iter=20000, seed=0):
    """Random-simplex search for non-negative weights over calibrated prob columns."""
    rng = np.random.RandomState(seed)
    M = probs.shape[1]
    best_w = np.ones(M) / M
    best_l = metric(y, probs @ best_w, w)
    for _ in range(n_iter):
        a = rng.dirichlet(np.ones(M) * 0.5)
        l = metric(y, probs @ a, w)
        if l < best_l:
            best_l, best_w = l, a
    return best_w, best_l


def per_tile_scores(model, Fz, L, R, y, w, train_mask, train_tiles, z_tiles):
    if model["kind"] == "linbt":
        return score_linbt(Fz, L, R, y, w, train_mask, C=model["C"])
    return score_reg(Fz, train_tiles, z_tiles, **{k: v for k, v in model.items()})


def load_feats():
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    tiles = list(dz["tiles"]); Ftr = dz["feats"].astype(np.float64)
    te_tiles = list(de["tiles"]); Fte = de["feats"].astype(np.float64)
    return tr, te, tiles, Ftr, te_tiles, Fte


def run_oof(models, n_splits=30, val_frac=0.3, btC=0.5, seed0=0, verbose=False):
    tr, te, tiles, Ftr, te_tiles, Fte = load_feats()
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    nT = len(tiles); nP = len(tr)
    mnames = list(models.keys())
    oof_logit = {m: np.zeros(nP) for m in mnames}
    oof_cnt = np.zeros(nP)
    for sp in range(n_splits):
        rng = np.random.RandomState(seed0 + sp)
        perm = rng.permutation(nT)
        val = set(perm[:int(nT * val_frac)].tolist())
        tile_train = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() == 0:
            continue
        # standardize features on train tiles only
        mu = Ftr[tile_train].mean(0); sd = Ftr[tile_train].std(0) + 1e-8
        Fz = (Ftr - mu) / sd
        # BT z target on train pairs only
        sub = tr[trm].reset_index(drop=True)
        sub_tiles = sorted(set(sub.left_image_path) | set(sub.right_image_path))
        z_sub, sidx = fit_bt(sub, sub_tiles, C=btC)
        train_tile_ids = np.array([idx[t] for t in sub_tiles])
        z_tiles = np.array([z_sub[sidx[t]] for t in sub_tiles])
        z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
        for m in mnames:
            s = per_tile_scores(models[m], Fz, L, R, y, w, trm, train_tile_ids, z_tiles)
            # normalize logit by TRAIN-pair logit std (same convention used at final predict)
            sd_tr = (s[L[trm]] - s[R[trm]]).std() + 1e-8
            lg = (s[L[vam]] - s[R[vam]]) / sd_tr
            oof_logit[m][vam] += lg
        oof_cnt[vam] += 1
    seen = oof_cnt > 0
    for m in mnames:
        oof_logit[m][seen] /= oof_cnt[seen]
    return dict(tr=tr, te=te, tiles=tiles, Ftr=Ftr, te_tiles=te_tiles, Fte=Fte,
                idx=idx, L=L, R=R, y=y, w=w, oof_logit=oof_logit, seen=seen, mnames=mnames)


def evaluate(oof, models):
    y, w, seen = oof["y"], oof["w"], oof["seen"]
    temps = {}
    cal_probs = {}
    print(f"OOF coverage: {seen.mean()*100:.0f}% of pairs")
    for m in oof["mnames"]:
        lg = oof["oof_logit"][m]
        t, l = fit_temp(y[seen], lg[seen], w[seen])
        temps[m] = t
        cal_probs[m] = 1 / (1 + np.exp(-lg / t))
        acc = ((lg[seen] > 0).astype(int) == y[seen]).mean()
        print(f"  {m:7s} OOF loss={l:6.2f}  acc={acc:.3f}  T={t:.2f}")
    # ensemble: optimized non-negative weights over calibrated probs
    Pmat = np.stack([cal_probs[m][seen] for m in oof["mnames"]], 1)
    ww, wl = optimize_weights(Pmat, y[seen], w[seen])
    Pw = Pmat @ ww
    ens_logit = np.log(Pw / (1 - Pw + 1e-9) + 1e-9)
    t2, l2 = fit_temp(y[seen], ens_logit, w[seen])
    print(f"  EQUAL-MEAN OOF loss={metric(y[seen], Pmat.mean(1), w[seen]):.2f}")
    print(f"  WEIGHTED  OOF loss={wl:6.2f}  (re-temp {l2:.2f}, T={t2:.2f})")
    print("  weights:", {m: round(float(a), 3) for m, a in zip(oof["mnames"], ww)})
    return temps, ww, t2


if __name__ == "__main__":
    oof = run_oof(MODELS, n_splits=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    evaluate(oof, MODELS)

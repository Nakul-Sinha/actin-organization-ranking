"""Transfer-robust submission: rank-normalized + shift-stable + confound-orthogonal.

Fixes the train->test transfer failure (live submission scored 73, anti-correlated
under shift). Validated on a simulated-shift proxy (sim_shift2.py): this config
reaches shift-acc ~0.55 / shift-loss ~68.3 vs the prior method's anti-correlated
~73. Steps:
  1. rank-normalize each feature within its own set (train among train, test among
     test) -> removes the marginal distribution shift (transductive, no labels);
  2. drop the 50% most train/test-shifted features;
  3. residualize against confounds + orthogonalize test logits vs confound diffs;
  4. calibrate the temperature on the simulated-shift proxy (reflects transfer),
     mildly conservative.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from build_robust import img_confounds, residualize, orthogonalize, pair_confdiff, metric, CONF_FEATS

ROOT = Path("dataset"); WORK = Path("working")
KEEP_PCT, C_LIN, T_MULT, CLIP = 0.5, 0.01, 3.0, 0.10


def rank_norm(F, ref):
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        sv = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out


def load():
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
    de = np.load(WORK / "feats_test.npz", allow_pickle=True)
    return (tr, te, list(dz["tiles"]), dz["feats"].astype(float),
            list(de["tiles"]), de["feats"].astype(float), list(dz["keys"]))


def directions(Ftr, Fte, n=6):
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
    out = []; rng = np.random.RandomState(0)
    for _ in range(n):
        s = rng.choice(len(Xz), int(0.8 * len(Xz)), replace=False)
        out.append((LogisticRegression(C=0.05, max_iter=2000).fit(Xz[s], lab[s]).coef_.ravel(), mu, sd))
    return out


def fit_predict(Frank_tr, keep, conf_idx, fit_mask, L, R, y, w, trm, Dvam_idx=None):
    F = Frank_tr.copy(); F[:, ~keep] = 0.0
    Fr = residualize(F, conf_idx, fit_mask)
    Fz = (Fr - Fr[fit_mask].mean(0)) / (Fr[fit_mask].std(0) + 1e-8)
    clf = LogisticRegression(C=C_LIN, max_iter=4000)
    clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
    s = Fz @ clf.coef_.ravel()
    return s, (s[L[trm]] - s[R[trm]]).std() + 1e-8


def build():
    tr, te, tiles, Ftr, te_tiles, Fte, keys = load()
    return core(tr, te, tiles, Ftr, te_tiles, Fte, keys, ROOT)


def core(tr, te, tiles, Ftr, te_tiles, Fte, keys, root):
    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    confmap = {t: img_confounds(t, root) for t in set(tiles) | set(te_tiles)}
    idx = {t: i for i, t in enumerate(tiles)}; teidx = {t: i for i, t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    smd = np.abs((Ftr.mean(0) - Fte.mean(0)) / (np.sqrt(0.5*(Ftr.var(0)+Fte.var(0))) + 1e-8))
    keep = smd <= np.percentile(smd, KEEP_PCT * 100)
    nT = len(tiles)

    # ---- calibrate temperature on the simulated-shift proxy ----
    POOL_lg, POOL_y, POOL_w = [], [], []
    for cf, mu, sd in directions(Ftr, Fte):
        proj = ((Ftr - mu) / sd) @ cf; order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True
            trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
            vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
            if trm.sum() < 30 or vam.sum() < 20:
                continue
            Frank = Ftr.copy(); Frank[tl] = rank_norm(Ftr, Ftr[tl])[tl]; Frank[vl] = rank_norm(Ftr, Ftr[vl])[vl]
            s, sdt = fit_predict(Frank, keep, conf_idx, tl, L, R, y, w, trm)
            lg = orthogonalize((s[L[vam]] - s[R[vam]]) / sdt, Dall[vam]); lg /= (np.std(lg) + 1e-8)
            POOL_lg.append(lg); POOL_y.append(y[vam]); POOL_w.append(w[vam])
    plg = np.concatenate(POOL_lg); py = np.concatenate(POOL_y); pw = np.concatenate(POOL_w)
    T_opt = min(np.linspace(0.5, 12, 120), key=lambda T: metric(py, 1/(1+np.exp(-plg/T)), pw))
    proxy_loss = metric(py, 1/(1+np.exp(-plg/T_opt)), pw)
    proxy_acc = ((plg > 0).astype(int) == py).mean()
    T_safe = T_opt * T_MULT
    print(f"[transfer] proxy shift: acc={proxy_acc:.3f} loss={proxy_loss:.2f} T_opt={T_opt:.2f} -> T_safe={T_safe:.2f}")

    # ---- final fit on ALL train tiles ----
    Frank_all = Ftr.copy(); Frank_all[:] = rank_norm(Ftr, Ftr)
    Fte_rank = rank_norm(Fte, Fte)
    allmask = np.ones(nT, bool)
    F = Frank_all.copy(); F[:, ~keep] = 0.0
    Fr = residualize(F, conf_idx, allmask)
    # residualize test (rank-normed) with train-fit confound model
    Fte_use = Fte_rank.copy(); Fte_use[:, ~keep] = 0.0
    Cc = Fr[:, conf_idx]  # zeros (residualized); use original rank confs for test removal
    # simpler: residualize test against train rank-confounds directly
    Cz = Frank_all[:, conf_idx]; mC = Cz.mean(0); sCc = Cz.std(0) + 1e-8
    Ctr = np.c_[(Cz - mC) / sCc, np.ones(nT)]; Cte = np.c_[(Fte_rank[:, conf_idx] - mC) / sCc, np.ones(len(Fte))]
    Fte_r = Fte_use.copy()
    for j in range(Fte.shape[1]):
        if (j in conf_idx) or (not keep[j]):
            Fte_r[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Ctr, Frank_all[:, j], rcond=None); Fte_r[:, j] = Fte_rank[:, j] - Cte @ coef
    mu = Fr.mean(0); sd = Fr.std(0) + 1e-8; Fz = (Fr - mu) / sd; Fz_te = (Fte_r - mu) / sd
    clf = LogisticRegression(C=C_LIN, max_iter=4000); clf.fit(Fz[L] - Fz[R], y, sample_weight=w)
    coef = clf.coef_.ravel(); s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    lg_te = (s_te[Lt] - s_te[Rt]) / sdt
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)
    lg_te = orthogonalize(lg_te, Dte); lg_te /= (np.std(lg_te) + 1e-8)
    prob = np.clip(1/(1+np.exp(-lg_te / T_safe)), 0.5 - CLIP, 0.5 + CLIP)
    return tr, te, te_tiles, prob, Dte, dict(proxy_loss=proxy_loss, proxy_acc=proxy_acc, T_safe=T_safe)


if __name__ == "__main__":
    tr, te, te_tiles, prob, Dte, info = build()
    cc = max(abs(np.corrcoef(Dte[:, i], prob)[0, 1]) for i in range(Dte.shape[1]))
    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission_transfer.csv", index=False)
    print(f"[transfer] wrote submission_transfer.csv prob[min={prob.min():.3f} mean={prob.mean():.3f} "
          f"max={prob.max():.3f} std={prob.std():.3f}] max|confound-corr|={cc:.3f}")

"""SSL-feature transfer-robust submission (proxy-validated best).

Light Barlow-Twins features on all 490 tiles (in-distribution) transfer far better
than hand-crafted (proxy-loss 64.9 vs 67.4). Processing mirrors build_transfer:
rank-normalize within set + residualize/orthogonalize vs image confounds + calibrate
the temperature on the simulated train->test shift (mildly conservative). Optionally
ensembles with hand-crafted features.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from build_robust import img_confounds, orthogonalize, pair_confdiff, metric

ROOT = Path("dataset"); WORK = Path("working")
C_LIN, T_MULT, CLIP = 0.02, 1.25, 0.32


def rank_norm(F, ref):
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        sv = np.sort(ref[:, j]); out[:, j] = np.searchsorted(sv, F[:, j], side="right") / (len(sv) + 1.0)
    return out


def residualize_ext(F, Cmat, fit_mask):
    Cz = (Cmat - Cmat[fit_mask].mean(0)) / (Cmat[fit_mask].std(0) + 1e-8)
    Cz = np.c_[Cz, np.ones(len(Cz))]
    out = F.copy()
    for j in range(F.shape[1]):
        coef, *_ = np.linalg.lstsq(Cz[fit_mask], F[fit_mask, j], rcond=None)
        out[:, j] = F[:, j] - Cz @ coef
    return out


def directions(Ftr, Fte, n=6):
    X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
    out = []; rng = np.random.RandomState(0)
    for _ in range(n):
        s = rng.choice(len(Xz), int(0.8 * len(Xz)), replace=False)
        out.append((LogisticRegression(C=0.05, max_iter=2000).fit(Xz[s], lab[s]).coef_.ravel(), mu, sd))
    return out


def _logits(Ftr, Fte, Cmat, tr_idx, L, R, y, w, trm, fit_mask, pred_L, pred_R, Dpred):
    F = np.vstack([Ftr, Fte]) if Fte is not None else Ftr
    # rank-norm handled by caller; here F already processed per-set
    Fr = residualize_ext(F, Cmat, fit_mask)
    Fz = (Fr - Fr[fit_mask].mean(0)) / (Fr[fit_mask].std(0) + 1e-8)
    clf = LogisticRegression(C=C_LIN, max_iter=4000)
    clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
    s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
    lg = (s[pred_L] - s[pred_R]) / sdt
    return orthogonalize(lg, Dpred)


def core(tr, te, tiles, Ftr_ssl, te_tiles, Fte_ssl, Hand, Hte, root, use_hand=True):
    confmap = {t: img_confounds(t, root) for t in set(tiles) | set(te_tiles)}
    idx = {t: i for i, t in enumerate(tiles)}; teidx = {t: i for i, t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    Cmat = np.array([confmap[t] for t in tiles]); Cte = np.array([confmap[t] for t in te_tiles])
    nT = len(tiles)

    # feature blocks to ensemble: SSL (+ optionally hand-crafted)
    blocks = [("ssl", Ftr_ssl, Fte_ssl)]
    if use_hand:
        blocks.append(("hand", Hand, Hte))

    # ---- proxy calibration (simulated shift via hand-crafted direction) ----
    splits = []
    for cf, mu, sd in directions(Hand, Hte):
        proj = ((Hand - mu) / sd) @ cf; order = np.argsort(proj)
        for frac in (0.38, 0.45):
            k = int(nT * frac); tl = np.zeros(nT, bool); vl = np.zeros(nT, bool)
            tl[order[:k]] = True; vl[order[-k:]] = True; splits.append((tl, vl))
    PL, PY, PW = [], [], []
    for tl, vl in splits:
        trm = np.array([(tl[l] and tl[r]) for l, r in zip(L, R)])
        vam = np.array([(vl[l] and vl[r]) for l, r in zip(L, R)])
        if trm.sum() < 30 or vam.sum() < 20:
            continue
        lo = np.zeros(vam.sum())
        for name, Ftr, _ in blocks:
            Fp = Ftr.copy(); Fp[tl] = rank_norm(Ftr, Ftr[tl])[tl]; Fp[vl] = rank_norm(Ftr, Ftr[vl])[vl]
            Fr = residualize_ext(Fp, Cmat, tl); Fz = (Fr - Fr[tl].mean(0)) / (Fr[tl].std(0) + 1e-8)
            clf = LogisticRegression(C=C_LIN, max_iter=4000)
            clf.fit((Fz[L] - Fz[R])[trm], y[trm], sample_weight=w[trm])
            s = Fz @ clf.coef_.ravel(); sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
            l = orthogonalize((s[L[vam]] - s[R[vam]]) / sdt, Dall[vam]); lo = lo + l / (np.std(l) + 1e-8)
        lo /= len(blocks); lo /= (np.std(lo) + 1e-8)
        PL.append(lo); PY.append(y[vam]); PW.append(w[vam])
    plg = np.concatenate(PL); py = np.concatenate(PY); pw = np.concatenate(PW)
    T_opt = min(np.linspace(0.5, 12, 120), key=lambda T: metric(py, 1/(1+np.exp(-plg/T)), pw))
    proxy_loss = metric(py, 1/(1+np.exp(-plg/T_opt)), pw); proxy_acc = ((plg > 0).astype(int) == py).mean()
    T_safe = T_opt * T_MULT

    # ---- final: fit on all train, predict test ----
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)
    allmask = np.ones(nT, bool)
    lg_te = np.zeros(len(te))
    for name, Ftr, Fte in blocks:
        Ftr_p = rank_norm(Ftr, Ftr); Fte_p = rank_norm(Fte, Fte)
        Fr = residualize_ext(Ftr_p, Cmat, allmask)
        # residualize test against train-fit confound model
        Cz = (Cmat - Cmat.mean(0)) / (Cmat.std(0) + 1e-8); Cz = np.c_[Cz, np.ones(nT)]
        Cte_z = np.c_[(Cte - Cmat.mean(0)) / (Cmat.std(0) + 1e-8), np.ones(len(Cte))]
        Fte_r = Fte_p.copy()
        for j in range(Fte_p.shape[1]):
            coef, *_ = np.linalg.lstsq(Cz, Ftr_p[:, j], rcond=None); Fte_r[:, j] = Fte_p[:, j] - Cte_z @ coef
        mu = Fr.mean(0); sd = Fr.std(0) + 1e-8; Fz = (Fr - mu) / sd; Fz_te = (Fte_r - mu) / sd
        clf = LogisticRegression(C=C_LIN, max_iter=4000); clf.fit(Fz[L] - Fz[R], y, sample_weight=w)
        c = clf.coef_.ravel(); s_tr = Fz @ c; s_te = Fz_te @ c
        sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
        l = orthogonalize((s_te[Lt] - s_te[Rt]) / sdt, Dte); lg_te = lg_te + l / (np.std(l) + 1e-8)
    lg_te /= len(blocks); lg_te /= (np.std(lg_te) + 1e-8)
    prob = np.clip(1/(1+np.exp(-lg_te / T_safe)), 0.5 - CLIP, 0.5 + CLIP)
    return prob, Dte, dict(proxy_loss=proxy_loss, proxy_acc=proxy_acc, T_safe=T_safe, blocks=[b[0] for b in blocks])


if __name__ == "__main__":
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    dz = np.load(WORK / "ssl_feats.npz", allow_pickle=True)
    tiles = list(dz["tr_tiles"]); te_tiles = list(dz["te_tiles"])
    Ftr_ssl = dz["ftr"].astype(float); Fte_ssl = dz["fte"].astype(float)
    hz = np.load(WORK / "feats_train.npz", allow_pickle=True); he = np.load(WORK / "feats_test.npz", allow_pickle=True)
    Hand = hz["feats"].astype(float); Hte = he["feats"].astype(float)
    for uh in (False, True):
        prob, Dte, info = core(tr, te, tiles, Ftr_ssl, te_tiles, Fte_ssl, Hand, Hte, ROOT, use_hand=uh)
        cc = max(abs(np.corrcoef(Dte[:, i], prob)[0, 1]) for i in range(Dte.shape[1]))
        print(f"blocks={info['blocks']} proxy: acc={info['proxy_acc']:.3f} loss={info['proxy_loss']:.2f} "
              f"T_safe={info['T_safe']:.2f} std={prob.std():.3f} max|cc|={cc:.3f}")
        if uh:
            sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
            sample = pd.read_csv(ROOT / "sample_submission.csv")
            sub = sub.set_index("id").loc[sample["id"].values].reset_index()
            sub.to_csv(WORK / "submission_ssl.csv", index=False)
            print("wrote submission_ssl.csv")

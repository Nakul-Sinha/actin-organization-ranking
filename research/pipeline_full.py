"""Combined OOF pipeline: hand-crafted morphology scorers + deep CNN regressor.

Per tile-disjoint split, compute per-tile organization scores from both the
hand-crafted linear/tree models and a pretrained CNN regressing the BT latent
target. All models individually temperature-calibrated on honest OOF, then a
non-negative weighted blend. The two families make different errors -> the blend
beats either alone.
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P
import cnn_regress as C
from bt_scores import fit_bt

ROOT = Path("dataset"); WORK = Path("working")

HAND = {
    "linbt_b": dict(kind="linbt", C=0.01),
}


def run_oof_full(n_splits=18, val_frac=0.3, btC=0.5, seed0=0,
                 deep_kind="resnet18", deep_seeds=2, deep_epochs=60,
                 deep_drop=0.5, deep_lr=2e-3, deep_wd=3e-2, use_deep=True):
    tr, te, tiles, Ftr, te_tiles, Fte = P.load_feats()
    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    nT = len(tiles); nP = len(tr)
    imgs = C.preload(tiles) if use_deep else None

    mnames = list(HAND.keys()) + (["deep"] if use_deep else [])
    oof_logit = {m: np.zeros(nP) for m in mnames}
    oof_cnt = np.zeros(nP)
    t0 = time.time()
    for sp in range(n_splits):
        rng = np.random.RandomState(seed0 + sp)
        perm = rng.permutation(nT)
        val = set(perm[:int(nT * val_frac)].tolist())
        tile_train = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() == 0:
            continue
        mu = Ftr[tile_train].mean(0); sd = Ftr[tile_train].std(0) + 1e-8
        Fz = (Ftr - mu) / sd
        sub = tr[trm].reset_index(drop=True)
        sub_tiles = sorted(set(sub.left_image_path) | set(sub.right_image_path))
        z_sub, sidx = fit_bt(sub, sub_tiles, C=btC)
        train_tile_ids = np.array([idx[t] for t in sub_tiles])
        z_tiles = np.array([z_sub[sidx[t]] for t in sub_tiles])
        z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
        # hand-crafted scorers
        for m in HAND:
            s = P.per_tile_scores(HAND[m], Fz, L, R, y, w, trm, train_tile_ids, z_tiles)
            sd_tr = (s[L[trm]] - s[R[trm]]).std() + 1e-8
            oof_logit[m][vam] += (s[L[vam]] - s[R[vam]]) / sd_tr
        # deep scorer
        if use_deep:
            val_pairs = (L[vam], R[vam], y[vam], w[vam])
            tw = np.ones(len(sub_tiles), dtype=np.float32)
            zhat = np.zeros(nT)
            for sdd in range(deep_seeds):
                model, _ = C.train_one(imgs, train_tile_ids.tolist(), z_tiles, tw, val_pairs,
                                       kind=deep_kind, drop=deep_drop, epochs=deep_epochs,
                                       lr=deep_lr, wd=deep_wd, seed=seed0 + sp * 13 + sdd)
                zhat += C.predict_z(model, imgs, tta=True)
            zhat /= deep_seeds
            sd_tr = (zhat[L[trm]] - zhat[R[trm]]).std() + 1e-8
            oof_logit["deep"][vam] += (zhat[L[vam]] - zhat[R[vam]]) / sd_tr
        oof_cnt[vam] += 1
        print(f"  split{sp} vp={vam.sum():4d} [{time.time()-t0:.0f}s]")
    seen = oof_cnt > 0
    for m in mnames:
        oof_logit[m][seen] /= oof_cnt[seen]
    return dict(tr=tr, te=te, tiles=tiles, Ftr=Ftr, te_tiles=te_tiles, Fte=Fte,
                idx=idx, L=L, R=R, y=y, w=w, oof_logit=oof_logit, seen=seen, mnames=mnames)


def evaluate(oof):
    y, w, seen = oof["y"], oof["w"], oof["seen"]
    print(f"OOF coverage: {seen.mean()*100:.0f}%")
    temps, cal = {}, {}
    for m in oof["mnames"]:
        lg = oof["oof_logit"][m]
        t, l = P.fit_temp(y[seen], lg[seen], w[seen])
        temps[m] = t; cal[m] = 1 / (1 + np.exp(-lg / t))
        acc = ((lg[seen] > 0).astype(int) == y[seen]).mean()
        print(f"  {m:8s} OOF={l:6.2f} acc={acc:.3f} T={t:.2f}")
    Pmat = np.stack([cal[m][seen] for m in oof["mnames"]], 1)
    ww, wl = P.optimize_weights(Pmat, y[seen], w[seen])
    print(f"  EQUAL-MEAN OOF={P.metric(y[seen], Pmat.mean(1), w[seen]):.2f}")
    print(f"  WEIGHTED   OOF={wl:.2f}  weights=" + str({m: round(float(a), 3) for m, a in zip(oof['mnames'], ww)}))
    return temps, ww


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    oof = run_oof_full(n_splits=ns)
    evaluate(oof)

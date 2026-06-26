"""Assemble a self-contained solution.py from tested source modules via AST.

Pulls the exact extract()/fit_bt()/deep-model functions so the official script's
feature code is byte-identical to what produced the cached features it was tuned
on, then appends a clean orchestration main().
"""
import ast
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent / "solution.py"


def grab(file, names):
    src = (SRC / file).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            # include decorators (their lineno is above node.lineno)
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            out[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    missing = set(names) - set(out)
    assert not missing, f"missing {missing} in {file}"
    return [out[n] for n in names]


HEADER = '''"""Microscopy Actin Pairwise Organization Ranking — official solution.

Pipeline (self-contained, reads ./dataset/public/ or ./dataset/, writes
./working/submission.csv):
  1. Per-tile hand-crafted morphology/topology features (intensity-robust).
  2. Bradley-Terry latent organization score z per tile from weighted pairs.
  3. Two per-tile scorers: linear BT (LogReg on feature diff) and a pretrained
     resnet18 regressing z (heavy dropout/weight-decay + dihedral augmentation).
  4. Tile-disjoint OOF -> per-model temperature calibration + blend weight.
  5. Fit both on all data, predict test, calibrate, blend, write submission.

Each scorer outputs a per-tile score, so pairwise predictions are globally
consistent across the reused-tile graph. Metric: gap-weighted pair log loss.
"""
import warnings, math, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy import sparse
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn, torch.nn.functional as F
import timm

SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_GABOR = None
'''


MAIN = r'''
# ----------------------------- metric / calibration -----------------------------
def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


def fit_temp(y, logit, w):
    best_t, best_l = 1.0, 1e9
    for t in np.linspace(0.2, 8.0, 120):
        l = metric(y, 1 / (1 + np.exp(-logit / t)), w)
        if l < best_l:
            best_l, best_t = l, t
    return best_t, best_l


def _extract_one(arg):
    rel, root = arg
    im = np.array(Image.open(Path(root) / rel), dtype=np.uint8)
    d = extract(im)
    keys = sorted(d.keys())
    return rel, np.array([d[k] for k in keys], dtype=np.float32), keys


def extract_features(tiles, root, workers=6):
    rows = {}
    keys = None
    args = [(t, str(root)) for t in tiles]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rel, vec, k in ex.map(_extract_one, args, chunksize=8):
                rows[rel] = vec; keys = k
    except Exception:
        for a in args:
            rel, vec, k = _extract_one(a); rows[rel] = vec; keys = k
    return np.stack([rows[t] for t in tiles]), keys


# ----------------------------- deep training -----------------------------
def train_deep(imgs, train_ids, z_tiles, epochs, n_seed, val_pairs=None,
               drop=0.5, lr=2e-3, wd=3e-2, bs=48, backbone="resnet18"):
    """Seed-ensembled resnet18 regression of z. Early-stop on val_pairs if given,
    else fixed epochs. Returns TTA-averaged per-tile scores over ALL imgs."""
    nT = imgs.shape[0]
    zhat = np.zeros(nT)
    zt = torch.tensor(z_tiles, dtype=torch.float32, device=DEV)
    ids = torch.tensor(train_ids, device=DEV)
    n = len(train_ids); steps = max(1, n // bs)
    for sd in range(n_seed):
        torch.manual_seed(SEED + sd); np.random.seed(SEED + sd)
        model = make_model(backbone, drop)
        lr_use = lr * 0.15
        opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_use, epochs=epochs,
                                                    steps_per_epoch=steps, pct_start=0.15)
        best = (1e9, None)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEV)
            for s in range(steps):
                b = perm[s * bs:(s + 1) * bs]; gi = ids[b]
                x = augment(imgs[gi], train=True)
                loss = F.smooth_l1_loss(model(x), zt[b], beta=0.5)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if val_pairs is not None and (ep + 1) % 5 == 0:
                zh = predict_z(model, imgs, tta=False)
                Lv, Rv, yv, wv = val_pairs
                p = 1 / (1 + np.exp(-(zh[Lv] - zh[Rv])))
                vl = metric(yv, p, wv)
                if vl < best[0]:
                    best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if best[1] is not None:
            model.load_state_dict(best[1])
        zhat += predict_z(model, imgs, tta=True)
    return zhat / n_seed


# ----------------------------- orchestration -----------------------------
def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public") / "train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT} DEV={DEV}")
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
    idx = {t: i for i, t in enumerate(tiles)}; teidx = {t: i for i, t in enumerate(te_tiles)}

    print("[solution] extracting features...")
    Ftr, keys = extract_features(tiles, ROOT)
    Fte, _ = extract_features(te_tiles, ROOT)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Fz = (Ftr - mu) / sd; Fz_te = (Fte - mu) / sd
    print(f"[solution] features {Ftr.shape} ({time.time()-t0:.0f}s)")

    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    nT = len(tiles); nP = len(tr)
    btC = 0.5; C_lin = 0.01

    imgs = torch.from_numpy(np.stack([
        np.array(Image.open(ROOT / t), dtype=np.float32)[None] / 255.0 for t in tiles])).to(DEV)
    imgs_te = torch.from_numpy(np.stack([
        np.array(Image.open(ROOT / t), dtype=np.float32)[None] / 255.0 for t in te_tiles])).to(DEV)
    bank = torch.cat([imgs, imgs_te], 0)

    # ---- OOF for calibration + blend weight (tile-disjoint) ----
    N_SPLITS = 12; VAL_FRAC = 0.3
    oof = {"lin": np.zeros(nP), "deep": np.zeros(nP)}; cnt = np.zeros(nP)
    print("[solution] OOF calibration...")
    for spi in range(N_SPLITS):
        rng = np.random.RandomState(spi)
        perm = rng.permutation(nT); val = set(perm[:int(nT * VAL_FRAC)].tolist())
        ttrain = np.array([t not in val for t in range(nT)])
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        if vam.sum() == 0:
            continue
        mu2 = Ftr[ttrain].mean(0); sd2 = Ftr[ttrain].std(0) + 1e-8
        Fz2 = (Ftr - mu2) / sd2
        sub = tr[trm].reset_index(drop=True)
        sub_tiles = sorted(set(sub.left_image_path) | set(sub.right_image_path))
        z_sub, sidx = fit_bt(sub, sub_tiles, C=btC)
        z_tiles = np.array([z_sub[sidx[t]] for t in sub_tiles], dtype=np.float32)
        z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
        train_ids = [idx[t] for t in sub_tiles]
        # linear BT
        clf = LogisticRegression(C=C_lin, max_iter=4000)
        clf.fit((Fz2[L] - Fz2[R])[trm], y[trm], sample_weight=w[trm])
        s = Fz2 @ clf.coef_.ravel()
        sdt = (s[L[trm]] - s[R[trm]]).std() + 1e-8
        oof["lin"][vam] += (s[L[vam]] - s[R[vam]]) / sdt
        # deep (2 seeds for a stable OOF temperature/blend, early-stop on val)
        vp = (L[vam], R[vam], y[vam], w[vam])
        zh = train_deep(imgs, train_ids, z_tiles, epochs=60, n_seed=2, val_pairs=vp)
        sdt = (zh[L[trm]] - zh[R[trm]]).std() + 1e-8
        oof["deep"][vam] += (zh[L[vam]] - zh[R[vam]]) / sdt
        cnt[vam] += 1
        print(f"  oof split {spi+1}/{N_SPLITS} ({time.time()-t0:.0f}s)")
    seen = cnt > 0
    for m in oof:
        oof[m][seen] /= cnt[seen]
    temps = {}; cal = {}
    for m in oof:
        t, l = fit_temp(y[seen], oof[m][seen], w[seen]); temps[m] = t
        cal[m] = 1 / (1 + np.exp(-oof[m] / t))
        print(f"  {m}: OOF={l:.2f} T={t:.2f}")
    # blend weight grid
    best = (1e9, 0.5)
    for wl in np.linspace(0, 1, 41):
        p = wl * cal["lin"][seen] + (1 - wl) * cal["deep"][seen]
        l = metric(y[seen], p, w[seen])
        if l < best[0]:
            best = (l, wl)
    wlin = best[1]
    print(f"[solution] blend w_lin={wlin:.2f} OOF={best[0]:.2f}")

    # ---- final fit on ALL data ----
    z_all, sidx = fit_bt(tr, tiles, C=btC)
    z_tiles = np.array([z_all[sidx[t]] for t in tiles], dtype=np.float32)
    z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
    clf = LogisticRegression(C=C_lin, max_iter=4000)
    clf.fit(Fz[L] - Fz[R], y, sample_weight=w); coef = clf.coef_.ravel()
    s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
    p_lin = 1 / (1 + np.exp(-((s_te[Lt] - s_te[Rt]) / sdt) / temps["lin"]))

    print("[solution] final deep training...")
    zhat = train_deep(bank, list(range(nT)), z_tiles, epochs=50, n_seed=6)
    s_trd = zhat[:nT]; s_ted = zhat[nT:]
    sdt = (s_trd[L] - s_trd[R]).std() + 1e-8
    p_deep = 1 / (1 + np.exp(-((s_ted[Lt] - s_ted[Rt]) / sdt) / temps["deep"]))

    prob = np.clip(wlin * p_lin + (1 - wlin) * p_deep, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    # validate
    assert list(sub.columns) == ["id", "prob_left_higher_organization"]
    assert len(sub) == len(sample) and sub["id"].is_unique
    assert set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] wrote {WORK/'submission.csv'} {sub.shape} "
          f"prob[min={pv.min():.3f} mean={pv.mean():.3f} max={pv.max():.3f}] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
'''


def build():
    feat = grab("features.py", ["_gabor_bank", "_safe", "_stats", "extract",
                                "_lacunarity", "_gini", "_fractal_dimension"])
    bt = grab("bt_scores.py", ["fit_bt"])
    deep = grab("cnn_regress.py", ["preload", "augment", "tta_views", "TimmReg",
                                   "make_model", "predict_z"])
    parts = [HEADER, "\n\n# ===== features.py =====\n", "\n\n".join(feat),
             "\n\n# ===== bt_scores.py =====\n", "\n\n".join(bt),
             "\n\n# ===== cnn_regress.py (deep) =====\n", "\n\n".join(deep),
             MAIN]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    # sanity: it parses
    ast.parse(OUT.read_text())
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    build()

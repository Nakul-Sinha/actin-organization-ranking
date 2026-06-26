"""Assemble the self-contained robust solution.py from tested modules via AST.

Confound-ORTHOGONAL linear Bradley-Terry (CPU-only, deterministic, no network):
exact extract()/residualize()/orthogonalize()/img_confounds() are pulled from the
tested src modules so the official script matches what was validated.
"""
import ast
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent / "solution.py"


def grab(file, names):
    src = (SRC / file).read_text(); lines = src.splitlines(); tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            out[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    missing = set(names) - set(out)
    assert not missing, f"missing {missing} in {file}"
    return [out[n] for n in names]


HEADER = '''"""Microscopy Actin Pairwise Organization Ranking — official solution.

CONFOUND-ORTHOGONAL Bradley-Terry ranker (self-contained, CPU-only, deterministic,
no network). Reads ./dataset/public/ (falls back to ./dataset/), writes
./working/submission.csv.

The pairs are matched on intensity / texture / gradient strength / dark-pixel
fraction, so any model that uses those confounds overfits a spurious TRAIN residual
and fails on the matched test set. This solution:
  1. extracts 155 per-tile morphology/topology features;
  2. residualizes them against a confound basis (model cannot learn confounds);
  3. fits a linear Bradley-Terry model on the per-tile feature differences;
  4. orthogonalizes the test pair logits against the pair's confound DIFFERENCES
     (guaranteed uninformative in the matched test set);
  5. calibrates conservatively (tile-disjoint OOF temperature + clip), because the
     metric severely punishes confident-wrong predictions.
Every model emits a per-tile score, so predictions stay consistent across the
reused-tile graph. Metric: gap-weighted pair log loss.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression

_GABOR = None
CONF_FEATS = ["int_mean", "int_std", "int_max", "dark_frac", "nz_frac", "grad_mean",
              "grad_std", "glcm_contrast_mean", "glcm_contrast_std",
              "glcm_dissimilarity_mean", "shannon_entropy", "fg_frac", "blob_n",
              "cc_area_sum"]
'''


MAIN = r'''
# ----------------------------- feature extraction driver -----------------------------
def _extract_one(arg):
    rel, root = arg
    im = np.array(Image.open(Path(root) / rel), dtype=np.uint8)
    d = extract(im); keys = sorted(d.keys())
    return rel, np.array([d[k] for k in keys], dtype=np.float32), keys


def extract_features(tiles, root, workers=6):
    rows = {}; keys = None
    args = [(t, str(root)) for t in tiles]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rel, vec, k in ex.map(_extract_one, args, chunksize=8):
                rows[rel] = vec; keys = k
    except Exception:
        for a in args:
            rel, vec, k = _extract_one(a); rows[rel] = vec; keys = k
    return np.stack([rows[t] for t in tiles]), keys


def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public") / "train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT}")
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))

    print("[solution] extracting features...")
    Ftr, keys = extract_features(tiles, ROOT)
    Fte, _ = extract_features(te_tiles, ROOT)
    print(f"[solution] features tr{Ftr.shape} te{Fte.shape} ({time.time()-t0:.0f}s)")

    conf_idx = sorted(set(keys.index(k) for k in CONF_FEATS if k in keys))
    confmap = {t: img_confounds(t, ROOT) for t in set(tiles) | set(te_tiles)}

    C, safety, clip = 0.005, 2.0, 0.18
    acc, T_oof, L_oof, (lg_oof, y_oof, w_oof) = oof_signal(Ftr, tiles, tr, conf_idx, confmap, C=C)
    T_safe = T_oof * safety
    shipped = metric(y_oof, np.clip(1/(1+np.exp(-lg_oof/T_safe)), 0.5-clip, 0.5+clip), w_oof)
    print(f"[solution] confound-orthogonal OOF: acc={acc:.3f} best-loss={L_oof:.2f} "
          f"shipped-cal-loss={shipped:.2f} (T_safe={T_safe:.2f})")

    idx = {t: i for i, t in enumerate(tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Fr = residualize(Ftr, conf_idx, np.ones(len(tiles), bool))
    Cc = Ftr[:, conf_idx]; mC = Cc.mean(0); sC = Cc.std(0) + 1e-8
    Ctr = np.c_[(Cc - mC) / sC, np.ones(len(Cc))]; Cte = np.c_[(Fte[:, conf_idx] - mC) / sC, np.ones(len(Fte))]
    Fte_r = Fte.copy()
    for j in range(Fte.shape[1]):
        if j in conf_idx:
            Fte_r[:, j] = 0.0; continue
        coef, *_ = np.linalg.lstsq(Ctr, Ftr[:, j], rcond=None); Fte_r[:, j] = Fte[:, j] - Cte @ coef
    mu = Fr.mean(0); sd = Fr.std(0) + 1e-8; Fz = (Fr - mu) / sd; Fz_te = (Fte_r - mu) / sd
    clf = LogisticRegression(C=C, max_iter=4000); clf.fit(Fz[L] - Fz[R], y, sample_weight=w)
    coef = clf.coef_.ravel(); s_tr = Fz @ coef; s_te = Fz_te @ coef
    sdt = (s_tr[L] - s_tr[R]).std() + 1e-8
    teidx = {t: i for i, t in enumerate(te_tiles)}
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    lg_te = (s_te[Lt] - s_te[Rt]) / sdt
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)
    lg_te = orthogonalize(lg_te, Dte); lg_te = lg_te / (np.std(lg_te) + 1e-8)
    prob = np.clip(1 / (1 + np.exp(-lg_te / T_safe)), 0.5 - clip, 0.5 + clip)

    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    cc = max(abs(np.corrcoef(Dte[:, i], prob)[0, 1]) for i in range(Dte.shape[1]))
    assert list(sub.columns) == ["id", "prob_left_higher_organization"]
    assert len(sub) == len(sample) and sub["id"].is_unique and set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] wrote {WORK/'submission.csv'} {sub.shape} "
          f"prob[min={pv.min():.3f} mean={pv.mean():.3f} max={pv.max():.3f} std={pv.std():.3f}] "
          f"max|confound-corr|={cc:.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
'''


def build():
    feat = grab("features.py", ["_gabor_bank", "_safe", "_stats", "extract",
                                "_lacunarity", "_gini", "_fractal_dimension"])
    rob = grab("build_robust.py", ["metric", "img_confounds", "residualize",
                                   "orthogonalize", "pair_confdiff", "oof_signal"])
    parts = [HEADER, "\n\n# ===== features.py =====\n", "\n\n".join(feat),
             "\n\n# ===== build_robust.py =====\n", "\n\n".join(rob), MAIN]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    ast.parse(OUT.read_text())
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    build()

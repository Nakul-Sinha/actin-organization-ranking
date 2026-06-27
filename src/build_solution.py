"""Assemble the self-contained SSL+hand transfer-robust solution.py via AST.

Pulls exact tested functions: hand-crafted feature extraction, light Barlow-Twins
SSL, confound removal, and the rank-norm / proxy-calibration ensemble core.
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

TRANSFER-ROBUST ensemble of light self-supervised (Barlow Twins) features and
hand-crafted morphology features. Reads ./dataset/public/ (falls back to
./dataset/), writes ./working/submission.csv.

Why this design (learned from leaderboard feedback):
  - pairs are matched on intensity/texture/gradient/coverage, so a model using
    those confounds rides a spurious TRAIN residual and fails on the matched test;
  - train and test tiles are distribution-shifted, and raw morphology features
    ANTI-CORRELATE under that shift.
So: (1) light Barlow-Twins SSL on ALL 490 tiles (train+test images, no labels) gives
in-distribution features; light (~10 epochs) is the sweet spot — more overfits the
~500 tiles; (2) hand-crafted morphology features add complementary signal;
(3) rank-normalize each feature within its set (removes marginal shift); (4) residualize
+ orthogonalize against image confounds (=> ~0 confound correlation); (5) calibrate the
temperature on a SIMULATED train->test shift (not optimistic OOF), mildly conservative.

Requires a GPU and the public ImageNet ResNet-18 weights (timm) for the SSL init; the
hand-crafted half is CPU-only. ~2-3 min runtime. Metric: gap-weighted pair log loss.
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
from skimage import filters, feature, morphology, measure
from skimage.filters import sato, frangi, meijering, gabor_kernel
from skimage.morphology import disk
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn, torch.nn.functional as F
import timm

_GABOR = None
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C_LIN, T_MULT, CLIP = 0.02, 1.25, 0.32
SSL_EPOCHS, SSL_SEEDS = 10, 3
'''


MAIN = r'''
# ----------------------------- hand-crafted feature driver -----------------------------
def _extract_one(arg):
    rel, root = arg
    im = np.array(Image.open(Path(root) / rel), dtype=np.uint8)
    d = extract(im); keys = sorted(d.keys())
    return rel, np.array([d[k] for k in keys], dtype=np.float32)


def extract_features(tiles, root, workers=6):
    rows = {}
    args = [(t, str(root)) for t in tiles]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rel, vec in ex.map(_extract_one, args, chunksize=8):
                rows[rel] = vec
    except Exception:
        for a in args:
            rel, vec = _extract_one(a); rows[rel] = vec
    return np.stack([rows[t] for t in tiles]).astype(float)


def preload_imgs(tiles, root):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(Path(root) / t), np.float32) / 255.0
    return torch.from_numpy(a).to(DEV)


def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public") / "train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT} DEV={DEV}")
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    trt = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    tet = sorted(set(te.left_image_path) | set(te.right_image_path))
    tiles_all = trt + tet; ntr = len(trt)

    print("[solution] light SSL on all tiles...")
    imgs = preload_imgs(tiles_all, ROOT)
    ftr = 0; fte = 0
    for seed in range(SSL_SEEDS):
        a1, a2 = train_features(pretrained=True, epochs=SSL_EPOCHS, seed=seed, imgs=imgs,
                                tiles=tiles_all, ntr=ntr)
        ftr = ftr + a1; fte = fte + a2
    Ftr_ssl = (ftr / SSL_SEEDS).astype(float); Fte_ssl = (fte / SSL_SEEDS).astype(float)
    print(f"[solution] SSL feats tr{Ftr_ssl.shape} te{Fte_ssl.shape} ({time.time()-t0:.0f}s)")

    print("[solution] hand-crafted features...")
    Hand = extract_features(trt, ROOT); Hte = extract_features(tet, ROOT)
    print(f"[solution] hand feats ({time.time()-t0:.0f}s)")

    prob, Dte, info = core(tr, te, trt, Ftr_ssl, tet, Fte_ssl, Hand, Hte, ROOT, use_hand=True)
    cc = max(abs(np.corrcoef(Dte[:, i], prob)[0, 1]) for i in range(Dte.shape[1]))
    print(f"[solution] proxy-shift acc={info['proxy_acc']:.3f} loss={info['proxy_loss']:.2f} "
          f"T_safe={info['T_safe']:.2f} max|confound-corr|={cc:.3f}")

    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    assert list(sub.columns) == ["id", "prob_left_higher_organization"]
    assert len(sub) == len(sample) and sub["id"].is_unique and set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] wrote {WORK/'submission.csv'} {sub.shape} "
          f"prob[min={pv.min():.3f} mean={pv.mean():.3f} max={pv.max():.3f} std={pv.std():.3f}] "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
'''


def build():
    feat = grab("features.py", ["_gabor_bank", "_safe", "_stats", "extract",
                                "_lacunarity", "_gini", "_fractal_dimension"])
    ssl = grab("../research/ssl_train.py", ["BarlowTwins", "aug", "norm_only", "train_features"])
    rob = grab("build_robust.py", ["metric", "img_confounds", "orthogonalize", "pair_confdiff"])
    sslh = grab("build_ssl.py", ["rank_norm", "residualize_ext", "directions", "core"])
    parts = [HEADER, "\n\n# ===== features.py =====\n", "\n\n".join(feat),
             "\n\n# ===== ssl_train.py =====\n", "\n\n".join(ssl),
             "\n\n# ===== build_robust.py =====\n", "\n\n".join(rob),
             "\n\n# ===== build_ssl.py =====\n", "\n\n".join(sslh), MAIN]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    ast.parse(OUT.read_text())
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    build()

"""Assemble the self-contained transfer-robust solution.py via AST.

Pulls exact tested functions (feature extraction + confound removal + the
rank-norm/shift-stable transfer core) so the official script matches what was
validated on the simulated-shift proxy. CPU-only, deterministic, no network.
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

TRANSFER-ROBUST confound-orthogonal Bradley-Terry ranker (self-contained,
CPU-only, deterministic, no network). Reads ./dataset/public/ (falls back to
./dataset/), writes ./working/submission.csv.

The test tiles are distribution-shifted from train, and the pairs are matched on
intensity/texture/gradient/coverage. A model that uses the confounds OR the
shift-sensitive raw feature scales fails on the matched, shifted test set. So:
  1. extract 155 per-tile morphology/topology features;
  2. rank-normalize each feature within its own set (train among train, test among
     test) -> removes the marginal train/test distribution shift;
  3. drop the 50% most train/test-shifted features;
  4. residualize against a confound basis + orthogonalize the test pair logits
     against the pair's confound differences (=> ~0 confound correlation);
  5. calibrate the temperature on a SIMULATED train->test shift (a train-vs-test
     direction splits train tiles into train-like/test-like halves; train on one,
     score the other), mildly conservative.
Metric: gap-weighted pair log loss. Every model emits a per-tile score so
predictions stay consistent across the reused-tile graph.
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
KEEP_PCT, C_LIN, T_MULT, CLIP = 0.5, 0.01, 3.0, 0.10
'''


MAIN = r'''
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

    tr, te, te_tiles, prob, Dte, info = core(tr, te, tiles, Ftr.astype(float),
                                             te_tiles, Fte.astype(float), keys, ROOT)
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
    rob = grab("build_robust.py", ["metric", "img_confounds", "residualize",
                                   "orthogonalize", "pair_confdiff"])
    trans = grab("build_transfer.py", ["rank_norm", "directions", "fit_predict", "core"])
    parts = [HEADER, "\n\n# ===== features.py =====\n", "\n\n".join(feat),
             "\n\n# ===== build_robust.py =====\n", "\n\n".join(rob),
             "\n\n# ===== build_transfer.py =====\n", "\n\n".join(trans), MAIN]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    ast.parse(OUT.read_text())
    print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    build()

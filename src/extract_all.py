"""Extract & cache per-tile morphology features for all train+test tiles."""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from features import extract

ROOT = Path("dataset")
WORK = Path("working")
WORK.mkdir(exist_ok=True)


def all_tiles():
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    tr_tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
    return tr_tiles, te_tiles


def _one(rel):
    im = np.array(Image.open(ROOT / rel))
    return rel, extract(im)


def build(tiles, out):
    t0 = time.time()
    keys = None
    rows = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (rel, d) in enumerate(ex.map(_one, tiles, chunksize=8)):
            if keys is None:
                keys = sorted(d.keys())
            rows[rel] = np.array([d[k] for k in keys], dtype=np.float32)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tiles)}  ({time.time()-t0:.0f}s)")
    mat = np.stack([rows[t] for t in tiles])
    np.savez(out, tiles=np.array(tiles), feats=mat, keys=np.array(keys))
    print(f"saved {out}: {mat.shape}  in {time.time()-t0:.0f}s")
    return keys


if __name__ == "__main__":
    tr_tiles, te_tiles = all_tiles()
    print(f"train tiles {len(tr_tiles)}, test tiles {len(te_tiles)}")
    k1 = build(tr_tiles, WORK / "feats_train.npz")
    k2 = build(te_tiles, WORK / "feats_test.npz")
    assert k1 == k2
    print(f"feature dim: {len(k1)}")

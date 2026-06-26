"""Extract FROZEN pretrained-backbone features per tile (no fine-tuning).

Low-variance alternative to full fine-tuning. We try multi-view (dihedral TTA)
averaged pooled features so the per-tile descriptor is orientation-invariant.
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch, timm

ROOT = Path("dataset"); WORK = Path("working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def preload(tiles, norm="perimg"):
    arr = np.zeros((len(tiles), 1, 128, 128), dtype=np.float32)
    for i, rel in enumerate(tiles):
        arr[i, 0] = np.array(Image.open(ROOT / rel), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)
    return x


def normalize(x, mode):
    if mode == "perimg":  # per-image standardize, brightness invariant
        mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
        return (x - mu) / sd
    if mode == "minmax":
        mn = x.amin(dim=[2, 3], keepdim=True); mx = x.amax(dim=[2, 3], keepdim=True)
        return (x - mn) / (mx - mn + 1e-5)
    return x


@torch.no_grad()
def extract(tiles, backbone="resnet18", norm="perimg", tta=True, bs=64):
    model = timm.create_model(backbone, pretrained=True, num_classes=0,
                              in_chans=1, global_pool="avg").to(DEV).eval()
    # keep BN in eval (pretrained running stats); inputs standardized to ~unit scale
    x = preload(tiles)
    x = normalize(x, norm)
    feats = []
    for i in range(0, len(tiles), bs):
        xb = x[i:i + bs].to(DEV)
        if tta:
            acc = 0; nv = 0
            for fh in [False, True]:
                for k in range(4):
                    v = torch.flip(xb, dims=[3]) if fh else xb
                    if k:
                        v = torch.rot90(v, k, dims=[2, 3])
                    acc = acc + model(v); nv += 1
            fb = (acc / nv)
        else:
            fb = model(xb)
        feats.append(fb.cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def build_and_cache(backbones, norm="perimg"):
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    tr_tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
    for bb in backbones:
        t0 = time.time()
        ftr = extract(tr_tiles, bb, norm)
        fte = extract(te_tiles, bb, norm)
        np.savez(WORK / f"deepfeat_{bb}_{norm}.npz",
                 tr_tiles=np.array(tr_tiles), te_tiles=np.array(te_tiles),
                 ftr=ftr, fte=fte)
        print(f"{bb} [{norm}]: tr {ftr.shape} te {fte.shape}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    bbs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["resnet18"]
    norm = sys.argv[2] if len(sys.argv) > 2 else "perimg"
    print(f"device={DEV} backbones={bbs} norm={norm}")
    build_and_cache(bbs, norm)

"""Self-supervised (Barlow Twins) in-distribution features on ALL 490 tiles.

Goal: features that DON'T shift between train and test (trained on both, no labels)
and are confound-robust (augment away intensity/contrast/gamma) while preserving
morphology (NO blur). Then a linear pairwise head can transfer where hand-crafted
features (which shift) cannot. Saves features for the proxy-gated head.
"""
import sys, time, math
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm

ROOT = Path("dataset"); WORK = Path("working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def all_tiles():
    tr = pd.read_csv(ROOT / "train.csv"); te = pd.read_csv(ROOT / "test.csv")
    trt = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    tet = sorted(set(te.left_image_path) | set(te.right_image_path))
    return trt, tet


def preload(tiles):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(ROOT / t), np.float32) / 255.0
    return torch.from_numpy(a).to(DEV)


def aug(x):
    """One augmented view on [0,1] images: geometric + intensity (NO blur)."""
    B = x.shape[0]
    if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k: x = torch.rot90(x, k, dims=[2, 3])
    ang = (torch.rand(B, device=x.device) * 2 - 1) * (30 * math.pi / 180)
    cos, sin = torch.cos(ang), torch.sin(ang)
    sc = 0.7 + torch.rand(B, device=x.device) * 0.45            # random resized crop-ish
    tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.15
    ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.15
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = cos * sc; th[:, 0, 1] = -sin * sc; th[:, 0, 2] = tx
    th[:, 1, 0] = sin * sc; th[:, 1, 1] = cos * sc; th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection").clamp(0, 1)
    gamma = torch.exp((torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * 0.6)
    x = x.clamp(1e-4, 1) ** gamma
    c = 0.6 + torch.rand(B, 1, 1, 1, device=x.device) * 0.8
    m = x.mean(dim=[2, 3], keepdim=True); x = ((x - m) * c + m).clamp(0, 1)
    x = x + torch.randn_like(x) * 0.02
    mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
    return (x - mu) / sd


def norm_only(x):
    mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
    return (x - mu) / sd


class BarlowTwins(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dim=1024):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1)
        d = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(d, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                  nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                  nn.Linear(dim, dim, bias=False))
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def forward(self, x1, x2):
        z1 = self.bn(self.proj(self.enc(x1))); z2 = self.bn(self.proj(self.enc(x2)))
        B = z1.shape[0]
        c = (z1.T @ z2) / B
        on = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
        return on + 5e-3 * off


def train_features(backbone="resnet18", pretrained=True, epochs=40, bs=128, lr=1e-3, seed=0, imgs=None, tiles=None, ntr=None):
    """Train light SSL and RETURN features (no save). Optionally reuse preloaded imgs.
    When imgs is given, pass ntr (number of train tiles) to avoid any path dependency."""
    torch.manual_seed(seed); np.random.seed(seed)
    if imgs is None:
        trt, tet = all_tiles(); tiles = trt + tet; imgs = preload(tiles)
        ntr = len(trt)
    elif ntr is None:
        trt, tet = all_tiles(); ntr = len(trt)
    n = len(tiles)
    model = BarlowTwins(backbone, pretrained).to(DEV)
    if epochs > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        steps = max(1, n // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.1)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            if len(idx) < 8: continue
            loss = model(aug(imgs[idx]), aug(imgs[idx]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        feats = []
        for i in range(0, n, 128):
            xb = imgs[i:i+128]; acc = 0; nv = 0
            for fh in [False, True]:
                for kk in range(4):
                    v = torch.flip(xb, dims=[3]) if fh else xb
                    if kk: v = torch.rot90(v, kk, dims=[2, 3])
                    acc = acc + model.enc(norm_only(v)); nv += 1
            feats.append((acc / nv).cpu().numpy())
        feats = np.concatenate(feats, 0)
    return feats[:ntr], feats[ntr:]


def train(backbone="resnet18", pretrained=True, epochs=300, bs=128, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    trt, tet = all_tiles()
    tiles = trt + tet
    imgs = preload(tiles)
    model = BarlowTwins(backbone, pretrained).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tiles); steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV); tot = 0
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            if len(idx) < 8: continue
            x = imgs[idx]
            loss = model(aug(x), aug(x))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"  ep{ep+1} loss {tot/steps:.2f} [{time.time()-t0:.0f}s]")
    # extract features (dihedral TTA averaged)
    model.eval()
    with torch.no_grad():
        feats = []
        for i in range(0, n, 128):
            xb = imgs[i:i+128]; acc = 0; nv = 0
            for fh in [False, True]:
                for kk in range(4):
                    v = torch.flip(xb, dims=[3]) if fh else xb
                    if kk: v = torch.rot90(v, kk, dims=[2, 3])
                    acc = acc + model.enc(norm_only(v)); nv += 1
            feats.append((acc / nv).cpu().numpy())
        feats = np.concatenate(feats, 0)
    ntr = len(trt)
    np.savez(WORK / "ssl_feats.npz", tr_tiles=np.array(trt), te_tiles=np.array(tet),
             ftr=feats[:ntr], fte=feats[ntr:])
    print(f"saved ssl_feats.npz tr{feats[:ntr].shape} te{feats[ntr:].shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    pre = (sys.argv[1] != "scratch") if len(sys.argv) > 1 else True
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print(f"device={DEV} pretrained={pre} epochs={ep}")
    train(pretrained=pre, epochs=ep)

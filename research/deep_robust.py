"""Test: does a CONFOUND-INVARIANT deep model add orthogonal signal?

Augment with random gamma + Gaussian blur (randomizes intensity & gradient
magnitude = the matched confounds) so the CNN cannot use them. Regress the BT
latent z, then orthogonalize val predictions against confound differences and
measure surviving signal. Compare to the linear confound-orthogonal model (0.613).
"""
import sys, math, time
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cnn_regress as C
from bt_scores import fit_bt
from build_robust import img_confounds, orthogonalize, pair_confdiff

ROOT = Path("dataset"); WORK = Path("working"); DEV = C.DEV


def augment_ci(x, train=True):
    """Confound-invariant aug on [0,1] images, then per-image standardize."""
    if train:
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k: x = torch.rot90(x, k, dims=[2, 3])
        B = x.shape[0]
        ang = (torch.rand(B, device=x.device)*2-1)*(25*math.pi/180)
        cos, sin = torch.cos(ang), torch.sin(ang)
        sc = 1.0+(torch.rand(B, device=x.device)*2-1)*0.12
        tx = (torch.rand(B, device=x.device)*2-1)*0.10; ty = (torch.rand(B, device=x.device)*2-1)*0.10
        th = torch.zeros(B, 2, 3, device=x.device)
        th[:, 0, 0] = cos*sc; th[:, 0, 1] = -sin*sc; th[:, 0, 2] = tx
        th[:, 1, 0] = sin*sc; th[:, 1, 1] = cos*sc; th[:, 1, 2] = ty
        grid = F.affine_grid(th, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        x = x.clamp(0, 1)
        # CONFOUND randomization:
        gamma = torch.exp((torch.rand(B, 1, 1, 1, device=x.device)*2-1)*0.6)   # gamma in ~[0.55,1.8]
        x = x.clamp(1e-4, 1) ** gamma
        if torch.rand(1).item() < 0.7:                                          # random blur
            sig = 0.3 + torch.rand(1).item()*1.4
            ks = 5
            xs = torch.arange(ks, device=x.device) - ks//2
            g1 = torch.exp(-(xs**2)/(2*sig*sig)); g1 = (g1/g1.sum()).to(x.dtype)
            kr = g1.view(1, 1, 1, ks).repeat(x.shape[1], 1, 1, 1)
            kc = g1.view(1, 1, ks, 1).repeat(x.shape[1], 1, 1, 1)
            x = F.conv2d(x, kr, padding=(0, ks//2), groups=x.shape[1])
            x = F.conv2d(x, kc, padding=(ks//2, 0), groups=x.shape[1])
        c = 0.6 + torch.rand(B, 1, 1, 1, device=x.device)*0.8                   # contrast
        m = x.mean(dim=[2, 3], keepdim=True); x = (x - m)*c + m
        x = x + torch.randn_like(x)*0.03
    mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True)+1e-5
    return (x - mu)/sd


C.augment = augment_ci  # monkeypatch into cnn_regress training/predict


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1-1e-6)
    return min(100.0, 100.0*(w*(-(y*np.log(p)+(1-y)*np.log(1-p)))).sum()/w.sum())


def run(n_splits=8, val_frac=0.3, epochs=60, seeds=2):
    tr = pd.read_csv(ROOT/"train.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path))
    tidx = {t: i for i, t in enumerate(tiles)}
    imgs = C.preload(tiles)
    confmap = {t: img_confounds(t) for t in tiles}
    L = np.array([tidx[t] for t in tr.left_image_path]); R = np.array([tidx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Dall = pair_confdiff(tr.left_image_path.values, tr.right_image_path.values, confmap)
    nT = len(tiles); LG, YY, WW = [], [], []
    t0 = time.time()
    for sp in range(n_splits):
        rng = np.random.RandomState(sp); perm = rng.permutation(nT)
        val = set(perm[:int(nT*val_frac)].tolist())
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        sub = tr[trm].reset_index(drop=True)
        st = sorted(set(sub.left_image_path)|set(sub.right_image_path))
        zs, si = fit_bt(sub, st, C=0.5); zt = np.array([zs[si[t]] for t in st], dtype=np.float32)
        zt = (zt-zt.mean())/(zt.std()+1e-8); tids = [tidx[t] for t in st]
        vp = (L[vam], R[vam], y[vam], w[vam])
        zhat = np.zeros(nT)
        for sd in range(seeds):
            m, _ = C.train_one(imgs, tids, zt, np.ones(len(st), np.float32), vp,
                               kind="resnet18", drop=0.5, epochs=epochs, lr=2e-3, wd=3e-2, seed=sp*10+sd)
            zhat += C.predict_z(m, imgs, tta=True)
        zhat /= seeds
        lg = zhat[L[vam]] - zhat[R[vam]]
        lg = orthogonalize(lg, Dall[vam])
        LG.append(lg); YY.append(y[vam]); WW.append(w[vam])
        print(f"  split{sp} vp={vam.sum()} [{time.time()-t0:.0f}s]")
    lg = np.concatenate(LG); y2 = np.concatenate(YY); w2 = np.concatenate(WW)
    acc = ((lg > 0).astype(int) == y2).mean()
    bestT, bestL = 1, 1e9
    for T in np.linspace(0.5, 15, 150):
        l = metric(y2, 1/(1+np.exp(-lg/np.std(lg)/T)), w2)
        if l < bestL: bestL, bestT = l, T
    print(f"== DEEP confound-invariant + orthogonalized: acc={acc:.3f} best-loss={bestL:.2f} (vs linear 0.613/67.25)")


if __name__ == "__main__":
    run(n_splits=int(sys.argv[1]) if len(sys.argv) > 1 else 8)

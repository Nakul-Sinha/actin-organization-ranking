"""Deep Siamese Bradley-Terry ranker for actin organization.

Shared backbone -> scalar score s(image). For a pair, logit = s(left) - s(right),
trained with pair_weight-weighted BCE. Tiles reused across pairs => the pairwise
constraints jointly identify each tile's latent organization score, and the CNN
learns a score(image) function that generalizes to unseen tiles.

All tiles preloaded to GPU; augmentation done on-GPU (dihedral + small affine +
per-image standardization => brightness/contrast invariant, aligned with the
dataset's matched confounds).
"""
import sys, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

ROOT = Path("dataset"); WORK = Path("working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- data -----------------------------
def load_pairs():
    tr = pd.read_csv(ROOT / "train.csv")
    te = pd.read_csv(ROOT / "test.csv")
    return tr, te


def preload(tiles):
    arr = np.zeros((len(tiles), 1, 128, 128), dtype=np.float32)
    for i, rel in enumerate(tiles):
        arr[i, 0] = np.array(Image.open(ROOT / rel), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(DEV)


# ----------------------------- augmentation (GPU) -----------------------------
def augment(x, train=True, rot_deg=20.0, trans=0.08, noise=0.02):
    B = x.shape[0]
    if train:
        # dihedral: random h/v flip + rot90
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[2, 3])
        # random small affine (rotation + translation + scale) via grid_sample
        ang = (torch.rand(B, device=x.device) * 2 - 1) * (rot_deg * math.pi / 180)
        cos, sin = torch.cos(ang), torch.sin(ang)
        sc = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * 0.10
        tx = (torch.rand(B, device=x.device) * 2 - 1) * trans
        ty = (torch.rand(B, device=x.device) * 2 - 1) * trans
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos * sc; theta[:, 0, 1] = -sin * sc; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin * sc; theta[:, 1, 1] = cos * sc; theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        if noise > 0:
            x = x + torch.randn_like(x) * noise
    # per-image standardize -> brightness/contrast invariant
    mu = x.mean(dim=[2, 3], keepdim=True)
    sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
    x = (x - mu) / sd
    return x


def tta_views(x):
    """8 dihedral views for test-time averaging, each per-image standardized."""
    views = []
    for fh in [False, True]:
        for k in range(4):
            v = x
            if fh:
                v = torch.flip(v, dims=[3])
            if k:
                v = torch.rot90(v, k, dims=[2, 3])
            mu = v.mean(dim=[2, 3], keepdim=True); sd = v.std(dim=[2, 3], keepdim=True) + 1e-5
            views.append((v - mu) / sd)
    return views


# ----------------------------- model -----------------------------
class Ranker(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                           num_classes=0, in_chans=1, global_pool="avg")
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 128), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


# ----------------------------- metric -----------------------------
def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return min(100.0, 100.0 * (w * loss).sum() / w.sum())


# ----------------------------- train one model -----------------------------
def train_model(imgs, Ltr, Rtr, ytr, wtr, Lva=None, Rva=None, yva=None, wva=None,
                backbone="resnet18", pretrained=True, epochs=60, bs=64, lr=3e-4,
                wd=1e-2, drop=0.3, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Ranker(backbone, pretrained, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(Ltr)
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps, pct_start=0.2)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    wtr_t = torch.tensor(wtr, dtype=torch.float32, device=DEV)
    Ltr_t = torch.tensor(Ltr, device=DEV); Rtr_t = torch.tensor(Rtr, device=DEV)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(steps):
            bidx = perm[s * bs:(s + 1) * bs]
            li = Ltr_t[bidx]; ri = Rtr_t[bidx]
            xl = augment(imgs[li], train=True); xr = augment(imgs[ri], train=True)
            sl = model(xl); sr = model(xr)
            logit = sl - sr
            loss = F.binary_cross_entropy_with_logits(logit, ytr_t[bidx],
                                                      weight=wtr_t[bidx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if Lva is not None and (ep + 1) % 5 == 0:
            sc, _ = predict_pairs(model, imgs, Lva, Rva, tta=False)
            vloss = metric(yva, sc, wva)
            if vloss < best[0]:
                best = (vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if verbose:
                print(f"    ep{ep+1:3d} train {tot/steps:.4f}  val {vloss:.2f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0]


@torch.no_grad()
def score_tiles(model, imgs, tta=True):
    model.eval()
    if tta:
        views = tta_views(imgs)
        s = torch.zeros(imgs.shape[0], device=DEV)
        for v in views:
            s = s + model(v)
        return (s / len(views)).cpu().numpy()
    else:
        x = augment(imgs, train=False)
        return model(x).cpu().numpy()


@torch.no_grad()
def predict_pairs(model, imgs, L, R, tta=True, temp=1.0):
    s = score_tiles(model, imgs, tta=tta)
    logit = (s[L] - s[R]) / temp
    p = 1 / (1 + np.exp(-logit))
    return p, s


def fit_temperature(y, logit, w):
    """1-D temperature search to minimize weighted log loss."""
    best_t, best_l = 1.0, 1e9
    for t in np.linspace(0.3, 5.0, 48):
        p = 1 / (1 + np.exp(-logit / t))
        l = metric(y, p, w)
        if l < best_l:
            best_l, best_t = l, t
    return best_t, best_l


# ----------------------------- tile-disjoint CV -----------------------------
def run_cv(backbone="resnet18", pretrained=True, n_splits=6, val_frac=0.3,
           epochs=60, lr=3e-4, drop=0.3, wd=1e-2, bs=64, seed0=0):
    tr, te = load_pairs()
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    tidx = {t: i for i, t in enumerate(tiles)}
    imgs = preload(tiles)
    L = np.array([tidx[t] for t in tr.left_image_path])
    R = np.array([tidx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32)
    w = tr.pair_weight.values.astype(np.float32)
    nT = len(tiles)
    raw_scores, cal_scores, accs = [], [], []
    t0 = time.time()
    for sp in range(n_splits):
        rng = np.random.RandomState(seed0 + sp)
        perm = rng.permutation(nT)
        val_tiles = set(perm[:int(nT * val_frac)].tolist())
        trm = np.array([(l not in val_tiles and r not in val_tiles) for l, r in zip(L, R)])
        vam = np.array([(l in val_tiles and r in val_tiles) for l, r in zip(L, R)])
        model, _ = train_model(imgs, L[trm], R[trm], y[trm], w[trm],
                               L[vam], R[vam], y[vam], w[vam],
                               backbone=backbone, pretrained=pretrained,
                               epochs=epochs, bs=bs, lr=lr, wd=wd, drop=drop, seed=seed0 + sp)
        s = score_tiles(model, imgs, tta=True)
        logit = s[L[vam]] - s[R[vam]]
        praw = 1 / (1 + np.exp(-logit))
        raw = metric(y[vam], praw, w[vam])
        t_, cal = fit_temperature(y[vam], logit, w[vam])  # oracle temp (upper bound)
        acc = ((praw > 0.5).astype(int) == y[vam]).mean()
        raw_scores.append(raw); cal_scores.append(cal); accs.append(acc)
        print(f"  split{sp} val_pairs={vam.sum():4d} acc={acc:.3f} raw={raw:.2f} cal*={cal:.2f} (T={t_:.2f}) [{time.time()-t0:.0f}s]")
    print(f"== {backbone}: raw {np.mean(raw_scores):.2f}+/-{np.std(raw_scores):.2f}  "
          f"cal* {np.mean(cal_scores):.2f}+/-{np.std(cal_scores):.2f}  acc {np.mean(accs):.3f}")
    return np.mean(raw_scores), np.mean(cal_scores), np.mean(accs)


if __name__ == "__main__":
    bb = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print(f"device={DEV} backbone={bb} epochs={ep}")
    run_cv(backbone=bb, epochs=ep, n_splits=6)

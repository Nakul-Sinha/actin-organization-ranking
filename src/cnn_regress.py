"""CNN regression to Bradley-Terry latent score z(image).

Two-stage: (1) fit per-tile BT scores from weighted pairs (train-pairs only per
CV split, no leakage), (2) train CNN to regress z with heavy augmentation +
early stopping on the pairwise weighted-log-loss val metric. Ensemble seeds.
"""
import sys, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
sys.path.insert(0, str(Path(__file__).parent))
from bt_scores import fit_bt

ROOT = Path("dataset"); WORK = Path("working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def preload(tiles):
    arr = np.zeros((len(tiles), 1, 128, 128), dtype=np.float32)
    for i, rel in enumerate(tiles):
        arr[i, 0] = np.array(Image.open(ROOT / rel), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(DEV)


def augment(x, train=True, rot_deg=25.0, trans=0.10, noise=0.03):
    if train:
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
        if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k: x = torch.rot90(x, k, dims=[2, 3])
        B = x.shape[0]
        ang = (torch.rand(B, device=x.device) * 2 - 1) * (rot_deg * math.pi / 180)
        cos, sin = torch.cos(ang), torch.sin(ang)
        sc = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * 0.12
        tx = (torch.rand(B, device=x.device) * 2 - 1) * trans
        ty = (torch.rand(B, device=x.device) * 2 - 1) * trans
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos*sc; theta[:, 0, 1] = -sin*sc; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin*sc; theta[:, 1, 1] = cos*sc; theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        if noise > 0: x = x + torch.randn_like(x) * noise
    mu = x.mean(dim=[2, 3], keepdim=True); sd = x.std(dim=[2, 3], keepdim=True) + 1e-5
    return (x - mu) / sd


def tta_views(x):
    out = []
    for fh in [False, True]:
        for k in range(4):
            v = torch.flip(x, dims=[3]) if fh else x
            if k: v = torch.rot90(v, k, dims=[2, 3])
            mu = v.mean(dim=[2, 3], keepdim=True); sd = v.std(dim=[2, 3], keepdim=True) + 1e-5
            out.append((v - mu) / sd)
    return out


class SmallNet(nn.Module):
    def __init__(self, drop=0.3, w=32):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False),
                                 nn.GroupNorm(8, o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False),
                                 nn.GroupNorm(8, o), nn.GELU())
        self.stem = blk(1, w, 2)        # 64
        self.b2 = blk(w, w*2, 2)        # 32
        self.b3 = blk(w*2, w*4, 2)      # 16
        self.b4 = blk(w*4, w*4, 2)      # 8
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(drop), nn.Linear(w*4, 1))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.stem(x))))).squeeze(-1)


class TimmReg(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, drop=0.4):
        super().__init__()
        self.bk = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                    in_chans=1, global_pool="avg")
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.bk.num_features, 1))

    def forward(self, x):
        return self.head(self.bk(x)).squeeze(-1)


def metric(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    loss = -(y*np.log(p) + (1-y)*np.log(1-p))
    return min(100.0, 100.0 * (w*loss).sum()/w.sum())


def make_model(kind, drop):
    if kind == "small":
        return SmallNet(drop=drop).to(DEV)
    return TimmReg(kind, pretrained=True, drop=drop).to(DEV)


@torch.no_grad()
def predict_z(model, imgs, tta=True):
    model.eval()
    if not tta:
        return model(augment(imgs, train=False)).cpu().numpy()
    acc = 0; vs = tta_views(imgs)
    for v in vs: acc = acc + model(v)
    return (acc/len(vs)).cpu().numpy()


def train_one(imgs, tr_ids, z, tile_w, val_pairs, kind="small", drop=0.3,
              epochs=80, bs=48, lr=2e-3, wd=2e-2, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(kind, drop)
    lr_use = lr if kind == "small" else lr * 0.15
    opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=wd)
    n = len(tr_ids)
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_use, epochs=epochs,
                                                steps_per_epoch=steps, pct_start=0.15)
    ids = torch.tensor(tr_ids, device=DEV)
    zt = torch.tensor(z, dtype=torch.float32, device=DEV)
    wt = torch.tensor(tile_w, dtype=torch.float32, device=DEV)
    Lva, Rva, yva, wva = val_pairs
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(steps):
            b = perm[s*bs:(s+1)*bs]
            gi = ids[b]
            x = augment(imgs[gi], train=True)
            pred = model(x)
            loss = (wt[b] * F.smooth_l1_loss(pred, zt[b], reduction="none", beta=0.5)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1) % 5 == 0:
            zhat = predict_z(model, imgs, tta=False)
            logit = zhat[Lva] - zhat[Rva]
            p = 1/(1+np.exp(-logit))
            vl = metric(yva, p, wva)
            if vl < best[0]:
                best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0]


def run_cv(kind="small", n_splits=6, val_frac=0.3, epochs=80, drop=0.3, lr=2e-3,
           wd=2e-2, n_seed=1, btC=0.5, seed0=0, verbose=True):
    tr = pd.read_csv(ROOT / "train.csv")
    tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
    tidx = {t: i for i, t in enumerate(tiles)}
    imgs = preload(tiles)
    L = np.array([tidx[t] for t in tr.left_image_path]); R = np.array([tidx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    nT = len(tiles)
    raws, cals, accs = [], [], []
    t0 = time.time()
    for sp in range(n_splits):
        rng = np.random.RandomState(seed0 + sp)
        perm = rng.permutation(nT)
        val = set(perm[:int(nT*val_frac)].tolist())
        trm = np.array([(l not in val and r not in val) for l, r in zip(L, R)])
        vam = np.array([(l in val and r in val) for l, r in zip(L, R)])
        # fit BT scores on TRAIN pairs only
        sub = tr[trm].reset_index(drop=True)
        sub_tiles = sorted(set(sub.left_image_path) | set(sub.right_image_path))
        z_sub, sidx = fit_bt(sub, sub_tiles, C=btC)
        zmap = {t: z_sub[sidx[t]] for t in sub_tiles}
        tr_ids = [tidx[t] for t in sub_tiles]
        zvals = np.array([zmap[t] for t in sub_tiles], dtype=np.float32)
        zvals = (zvals - zvals.mean()) / (zvals.std() + 1e-8)
        # tile weight = total pair weight it participates in (reliability)
        tw = np.ones(len(sub_tiles), dtype=np.float32)
        val_pairs = (L[vam], R[vam], y[vam], w[vam])
        zhat_ens = np.zeros(nT)
        vls = []
        for sd in range(n_seed):
            model, vl = train_one(imgs, tr_ids, zvals, tw, val_pairs, kind=kind,
                                   drop=drop, epochs=epochs, lr=lr, wd=wd, seed=seed0+sp*10+sd)
            zhat_ens += predict_z(model, imgs, tta=True)
            vls.append(vl)
        zhat = zhat_ens / n_seed
        logit = zhat[L[vam]] - zhat[R[vam]]
        praw = 1/(1+np.exp(-logit))
        raw = metric(y[vam], praw, w[vam])
        # oracle temperature (upper bound on calibrated)
        bt, bl = 1.0, 1e9
        for T in np.linspace(0.3, 6, 60):
            l = metric(y[vam], 1/(1+np.exp(-logit/T)), w[vam])
            if l < bl: bl, bt = l, T
        acc = ((praw > 0.5).astype(int) == y[vam]).mean()
        raws.append(raw); cals.append(bl); accs.append(acc)
        if verbose:
            print(f"  split{sp} vp={vam.sum():4d} acc={acc:.3f} raw={raw:.2f} cal*={bl:.2f}(T={bt:.2f}) earlystop_vl={np.mean(vls):.2f} [{time.time()-t0:.0f}s]")
    print(f"== {kind}: raw {np.mean(raws):.2f}+/-{np.std(raws):.2f}  cal* {np.mean(cals):.2f}+/-{np.std(cals):.2f}  acc {np.mean(accs):.3f}")
    return np.mean(raws), np.mean(cals), np.mean(accs)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "small"
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    print(f"device={DEV} kind={kind} epochs={ep} splits={ns}")
    run_cv(kind=kind, epochs=ep, n_splits=ns)

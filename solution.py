import warnings, time, math, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.linear_model import LogisticRegression

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAL_STD = 0.14
CNN_BACKBONES = ["resnet18", "resnet34", "resnet50"]
CNN_SEEDS = int(os.environ.get("CNN_SEEDS", "4"))
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS", "25"))
TTA_ROT = 4
FROZEN_TTA = 1
FROZEN_MODELS = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "convnext_large.fb_in22k",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
]


def preload(tiles, root):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(root / t), np.float32) / 255.0
    return torch.from_numpy(a)


def cnn_aug(x):
    if torch.rand(1).item() < .5: x = torch.flip(x, [3])
    if torch.rand(1).item() < .5: x = torch.flip(x, [2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k: x = torch.rot90(x, k, [2, 3])
    B = x.shape[0]
    ang = (torch.rand(B, device=x.device)*2-1)*(25*math.pi/180); cs, sn = torch.cos(ang), torch.sin(ang)
    sc = 0.8 + torch.rand(B, device=x.device)*0.4
    tx = (torch.rand(B, device=x.device)*2-1)*.12; ty = (torch.rand(B, device=x.device)*2-1)*.12
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = cs*sc; th[:, 0, 1] = -sn*sc; th[:, 0, 2] = tx
    th[:, 1, 0] = sn*sc; th[:, 1, 1] = cs*sc; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False, padding_mode="reflection").clamp(0, 1)
    gm = torch.exp((torch.rand(B, 1, 1, 1, device=x.device)*2-1)*0.5); x = x.clamp(1e-4, 1)**gm
    c = 0.7+torch.rand(B, 1, 1, 1, device=x.device)*0.6; m = x.mean([2, 3], keepdim=True); x = ((x-m)*c+m).clamp(0, 1)
    x = x + torch.randn_like(x)*0.02
    mu = x.mean([2, 3], keepdim=True); sd = x.std([2, 3], keepdim=True)+1e-5
    return (x-mu)/sd


class Ranker(nn.Module):
    def __init__(self, bb, drop=0.4):
        super().__init__()
        self.b = timm.create_model(bb, pretrained=True, num_classes=0, in_chans=1, global_pool="avg")
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(self.b.num_features, 1))
    def forward(self, x): return self.h(self.b(x)).squeeze(-1)


@torch.no_grad()
def cnn_tta(model, imgs, bs=128):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        xb = imgs[i:i+bs]; acc = 0; n = 0
        for k in range(TTA_ROT):
            v = xb if k == 0 else torch.rot90(xb, k, [2, 3])
            mu = v.mean([2, 3], keepdim=True); sd = v.std([2, 3], keepdim=True)+1e-5
            acc = acc + model((v-mu)/sd); n += 1
        out.append((acc/n).cpu().numpy())
    return np.concatenate(out, 0)


def train_cnn(imgs, L, R, y, w, bb, seed, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Ranker(bb).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-2)
    n = len(L); steps = max(1, n//bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=CNN_EPOCHS, steps_per_epoch=steps, pct_start=.15)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV); wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    Lt = torch.tensor(L, device=DEV); Rt = torch.tensor(R, device=DEV)
    for ep in range(CNN_EPOCHS):
        model.train(); perm = torch.randperm(n, device=DEV)
        for s in range(steps):
            b = perm[s*bs:(s+1)*bs]
            lg = model(cnn_aug(imgs[Lt[b]])) - model(cnn_aug(imgs[Rt[b]]))
            loss = F.binary_cross_entropy_with_logits(lg, yt[b], weight=wt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return cnn_tta(model, imgs)


@torch.no_grad()
def frozen_feats(mname, x128, bs=24):
    kw = dict(pretrained=True, num_classes=0)
    if "dinov2" in mname:
        kw["dynamic_img_size"] = True
    model = timm.create_model(mname, **kw).to(DEV).eval()
    cfg = model.pretrained_cfg; res = cfg["input_size"][-1]
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    x = F.interpolate(x128, size=(res, res), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    x = (x - mean) / std
    out = []
    use_amp = (DEV == "cuda")
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].to(DEV); acc = 0; n = 0
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            for k in range(FROZEN_TTA):
                v = xb if k == 0 else torch.rot90(xb, k, [2, 3])
                acc = acc + model(v).float(); n += 1
        out.append((acc/n).cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def calib(lg, target_std):
    lo = lg/(np.std(lg)+1e-8); k = 1.0
    for _ in range(60):
        k *= target_std/((1/(1+np.exp(-lo*k))).std()+1e-9)
    return np.clip(1/(1+np.exp(-lo*k)), 1e-4, 1-1e-4)


def main():
    t0 = time.time()
    ROOT = Path("dataset/public") if (Path("dataset/public")/"train.csv").exists() else Path("dataset")
    WORK = Path("working"); WORK.mkdir(exist_ok=True, parents=True)
    print(f"[solution] ROOT={ROOT} DEV={DEV}")
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path)); te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    allt = tiles+te_tiles; ntr = len(tiles)
    aidx = {t:i for i,t in enumerate(allt)}; tidx = {t:i for i,t in enumerate(tiles)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    L = np.array([aidx[t] for t in tr.left_image_path]); R = np.array([aidx[t] for t in tr.right_image_path])
    Ltr = np.array([tidx[t] for t in tr.left_image_path]); Rtr = np.array([tidx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    x128 = preload(allt, ROOT)
    components = []

    imgs = x128.to(DEV); cnn = 0
    for bb in CNN_BACKBONES:
        for sd in range(CNN_SEEDS):
            sc = train_cnn(imgs, L, R, y, w, bb, sd); s_te = sc[ntr:]; cnn = cnn + (s_te[Lt]-s_te[Rt])
        print(f"[solution] CNN {bb} x{CNN_SEEDS} ({time.time()-t0:.0f}s)")
    components.append(cnn/(np.std(cnn)+1e-8))
    del imgs; torch.cuda.empty_cache()

    for mname in FROZEN_MODELS:
        try:
            Fa = frozen_feats(mname, x128); Ftr = Fa[:ntr]; Fte = Fa[ntr:]
            mu = Ftr.mean(0); sd = Ftr.std(0)+1e-8; Fz = (Ftr-mu)/sd; Fz_te = (Fte-mu)/sd
            clf = LogisticRegression(C=0.02, max_iter=5000).fit(Fz[Ltr]-Fz[Rtr], y.astype(int), sample_weight=w)
            s_te = Fz_te @ clf.coef_.ravel(); lg = s_te[Lt]-s_te[Rt]
            components.append(lg/(np.std(lg)+1e-8))
            print(f"[solution] frozen {mname.split('.')[0]} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"[solution] frozen {mname} FAILED: {repr(e)[:100]}")

    grand = np.sum(components, axis=0)
    prob = calib(grand, CAL_STD)
    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK/"submission.csv", index=False)
    assert list(sub.columns) == ["id", "prob_left_higher_organization"] and len(sub) == len(sample)
    assert sub["id"].is_unique and set(sub["id"]) == set(sample["id"])
    pv = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(pv).all() and (pv >= 0).all() and (pv <= 1).all()
    print(f"[solution] {len(components)} scorers -> wrote {WORK/'submission.csv'} std={pv.std():.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

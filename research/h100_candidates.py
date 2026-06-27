"""H100: generate diverse candidate submissions to find one scoring ~65.

Others reach ~62, so a strong transferable signal exists. Main bet: a Siamese
RankNet CNN ENSEMBLE (many seeds -> variance reduction is the cure for my past
overfitting), trained directly on the pairwise labels, heavy augmentation, NO
confound removal (let it use whatever transfers), TTA. Also confound-linear and
blends, at several confidence levels. Saves CSVs to bring back and upload.
"""
import sys, time, math, os
sys.path.insert(0, "src"); sys.path.insert(0, "research")
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.linear_model import LogisticRegression
from build_robust import img_confounds

ROOT = Path("dataset"); OUT = Path("cands"); OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def preload(tiles):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(ROOT / t), np.float32) / 255.0
    return torch.from_numpy(a).to(DEV)


def aug(x, train=True):
    if train:
        if torch.rand(1).item() < .5: x = torch.flip(x, [3])
        if torch.rand(1).item() < .5: x = torch.flip(x, [2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k: x = torch.rot90(x, k, [2, 3])
        B = x.shape[0]
        ang = (torch.rand(B, device=x.device)*2-1)*(25*math.pi/180)
        cs, sn = torch.cos(ang), torch.sin(ang)
        sc = 0.8 + torch.rand(B, device=x.device)*0.4
        tx = (torch.rand(B, device=x.device)*2-1)*.12; ty = (torch.rand(B, device=x.device)*2-1)*.12
        th = torch.zeros(B, 2, 3, device=x.device)
        th[:, 0, 0] = cs*sc; th[:, 0, 1] = -sn*sc; th[:, 0, 2] = tx
        th[:, 1, 0] = sn*sc; th[:, 1, 1] = cs*sc; th[:, 1, 2] = ty
        g = F.affine_grid(th, x.shape, align_corners=False)
        x = F.grid_sample(x, g, align_corners=False, padding_mode="reflection").clamp(0, 1)
        gm = torch.exp((torch.rand(B, 1, 1, 1, device=x.device)*2-1)*0.5); x = x.clamp(1e-4, 1)**gm
        c = 0.7 + torch.rand(B, 1, 1, 1, device=x.device)*0.6; m = x.mean([2, 3], keepdim=True); x = ((x-m)*c+m).clamp(0, 1)
        x = x + torch.randn_like(x)*0.02
    mu = x.mean([2, 3], keepdim=True); sd = x.std([2, 3], keepdim=True)+1e-5
    return (x-mu)/sd


class Ranker(nn.Module):
    def __init__(self, bb="resnet18", drop=0.4):
        super().__init__()
        self.b = timm.create_model(bb, pretrained=True, num_classes=0, in_chans=1, global_pool="avg")
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(self.b.num_features, 1))
    def forward(self, x): return self.h(self.b(x)).squeeze(-1)


@torch.no_grad()
def tta_scores(model, imgs):
    model.eval(); acc = 0; n = 0
    for fh in [False, True]:
        for k in range(4):
            v = torch.flip(imgs, [3]) if fh else imgs
            if k: v = torch.rot90(v, k, [2, 3])
            mu = v.mean([2, 3], keepdim=True); sd = v.std([2, 3], keepdim=True)+1e-5
            acc = acc + model((v-mu)/sd); n += 1
    return (acc/n).cpu().numpy()


def train_siamese(imgs, L, R, y, w, bb="resnet18", epochs=55, bs=64, lr=3e-4, drop=0.4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Ranker(bb, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
    n = len(L); steps = max(1, n//bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=steps, pct_start=.15)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV); wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    Lt = torch.tensor(L, device=DEV); Rt = torch.tensor(R, device=DEV)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for s in range(steps):
            b = perm[s*bs:(s+1)*bs]
            xl = aug(imgs[Lt[b]], True); xr = aug(imgs[Rt[b]], True)
            lg = model(xl) - model(xr)
            loss = F.binary_cross_entropy_with_logits(lg, yt[b], weight=wt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return tta_scores(model, imgs)


def calib(logit, target_std):
    lo = logit/(np.std(logit)+1e-8); k = 1.0
    for _ in range(50):
        s = (1/(1+np.exp(-lo*k))).std(); k *= target_std/(s+1e-9)
    return np.clip(1/(1+np.exp(-lo*k)), 1e-4, 1-1e-4)


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    allt = tiles+te_tiles; ntr = len(tiles)
    idx = {t:i for i,t in enumerate(allt)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    imgs = preload(allt)
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    NSEED = int(os.environ.get("NSEED", "15"))

    def emit(score_te_logit, tag):
        for ts in [0.06, 0.10, 0.14]:
            prob = calib(score_te_logit, ts)
            sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
            sub = sub.set_index("id").loc[sample["id"].values].reset_index()
            sub.to_csv(OUT/f"cand_{tag}_s{int(ts*100):02d}.csv", index=False)
        print(f"  emitted {tag} ({time.time()-t0:.0f}s)")

    # ---- Siamese CNN ensembles (resnet18 + resnet34) ----
    for bb in ["resnet18", "resnet34"]:
        sc = 0
        for sd in range(NSEED):
            sc = sc + train_siamese(imgs, L, R, y, w, bb=bb, seed=sd)
            if (sd+1) % 5 == 0: print(f"  {bb} seed {sd+1}/{NSEED} ({time.time()-t0:.0f}s)")
        sc /= NSEED
        np.save(OUT/f"scores_{bb}.npy", sc)
        s_te = sc[ntr:]; emit(s_te[ [teidx[t] for t in te.left_image_path] ] - s_te[ [teidx[t] for t in te.right_image_path] ], f"cnn_{bb}")

    # ---- confound linear (NO orthogonalization) ----
    confmap = {t: img_confounds(t, ROOT) for t in allt}
    C = np.array([confmap[t] for t in allt]); C = (C-C[:ntr].mean(0))/(C[:ntr].std(0)+1e-8)
    clf = LogisticRegression(C=0.3, max_iter=4000).fit(C[:ntr][[idx[t] for t in tr.left_image_path]] - C[:ntr][[idx[t] for t in tr.right_image_path]], y, sample_weight=w)
    sc_conf = C @ clf.coef_.ravel(); s_te = sc_conf[ntr:]
    cl = s_te[[teidx[t] for t in te.left_image_path]] - s_te[[teidx[t] for t in te.right_image_path]]
    emit(cl, "confound")
    np.save(OUT/"scores_confound.npy", sc_conf)

    # ---- blends: CNN(resnet18) + confound ----
    r18 = np.load(OUT/"scores_resnet18.npy"); s_te18 = r18[ntr:]
    l18 = s_te18[[teidx[t] for t in te.left_image_path]] - s_te18[[teidx[t] for t in te.right_image_path]]
    bl = l18/(np.std(l18)+1e-8) + cl/(np.std(cl)+1e-8)
    emit(bl, "blend_cnn_confound")
    print(f"DONE ({time.time()-t0:.0f}s). candidates in cands/")


if __name__ == "__main__":
    main()

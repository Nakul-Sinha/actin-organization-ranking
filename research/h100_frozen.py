"""H100 run 3 (diversity): FROZEN foundation-model features + linear Bradley-Terry.

DINOv2 / ConvNeXt-large features are far stronger than ImageNet ResNet and don't
overfit (frozen). Resize tiles to 224, replicate to 3ch, ImageNet-normalize, dihedral
TTA-average features. Then linear BT on feature differences, both confound-INCLUSIVE
(keep all signal) and confound-orthogonal, at several confidence levels.
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, timm
sys.path.insert(0, "src"); sys.path.insert(0, "research")
from sklearn.linear_model import LogisticRegression
from build_robust import img_confounds, orthogonalize, pair_confdiff

ROOT = Path("dataset"); OUT = Path("cands3"); OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m", "convnext_large.fb_in22k"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preload(tiles, res):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(ROOT/t), np.float32)/255.0
    x = torch.from_numpy(a)
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return ((x - MEAN)/STD).to(DEV)


@torch.no_grad()
def feats(model, x, bs=64):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs]; acc = 0; n = 0
        for fh in [False, True]:
            for k in range(4):
                v = torch.flip(xb, [3]) if fh else xb
                if k: v = torch.rot90(v, k, [2, 3])
                acc = acc + model(v); n += 1
        out.append((acc/n).cpu().numpy())
    return np.concatenate(out, 0)


def calib(lg, ts):
    lo = lg/(np.std(lg)+1e-8); k = 1.0
    for _ in range(60): k *= ts/((1/(1+np.exp(-lo*k))).std()+1e-9)
    return np.clip(1/(1+np.exp(-lo*k)), 1e-4, 1-1e-4)


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path)); te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    idx = {t:i for i,t in enumerate(tiles)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    confmap = {t: img_confounds(t, ROOT) for t in set(tiles)|set(te_tiles)}
    Dte = pair_confdiff(te.left_image_path.values, te.right_image_path.values, confmap)

    def emit(lg, tag):
        for ts in [0.06, 0.10, 0.14]:
            p = calib(lg, ts); s = pd.DataFrame({"id": te.id.values, "prob_left_higher_organization": p}).set_index("id").loc[sample.id.values].reset_index()
            s.to_csv(OUT/f"cand_{tag}_s{int(ts*100):02d}.csv", index=False)

    grand_inc = 0; nb = 0
    for mname in MODELS:
        try:
            res = 224
            kw = dict(pretrained=True, num_classes=0)
            if "dinov2" in mname or "vit" in mname:
                kw["dynamic_img_size"] = True
            model = timm.create_model(mname, **kw).to(DEV)
            Ftr = feats(model, preload(tiles, res)); Fte = feats(model, preload(te_tiles, res))
            mu = Ftr.mean(0); sd = Ftr.std(0)+1e-8; Fz = (Ftr-mu)/sd; Fz_te = (Fte-mu)/sd
            clf = LogisticRegression(C=0.02, max_iter=4000); clf.fit(Fz[L]-Fz[R], y, sample_weight=w)
            c = clf.coef_.ravel(); s_te = Fz_te @ c; lg = s_te[Lt]-s_te[Rt]
            emit(lg, f"frozen_{mname.split('.')[0].split('_patch')[0]}")
            grand_inc = grand_inc + lg/(np.std(lg)+1e-8); nb += 1
            print(f"  {mname} done ({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {mname} FAILED: {repr(e)[:140]}", flush=True)
    if nb:
        grand_inc /= nb; emit(grand_inc, "frozen_grand")
    print(f"DONE ({time.time()-t0:.0f}s). {nb} models. candidates in cands3/", flush=True)


if __name__ == "__main__":
    main()

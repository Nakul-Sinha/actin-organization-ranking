"""Mega frozen-feature extraction: MANY diverse strong foundation models.

The recipe that hit 61.5: equal-weight grand of diverse strong frozen extractors +
CNN. More diverse strong models -> more variance reduction + diversity -> lower. Each
model uses its own native resolution + normalization (from pretrained_cfg). Saves
per-model test-pair logits to fold into the grand locally.
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, timm
sys.path.insert(0, "src"); sys.path.insert(0, "research")
from sklearn.linear_model import LogisticRegression

ROOT = Path("dataset"); OUT = Path("cands5"); OUT.mkdir(exist_ok=True); DEV = "cuda"
MODELS = [
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    "vit_large_patch14_clip_336.openai_ft_in12k_in1k",
    "vit_so400m_patch14_siglip_384.webli",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "beit_large_patch16_224.in22k_ft_in22k_in1k",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "eva_giant_patch14_336.clip_ft_in1k",
    "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
]


def load128(tiles):
    a = np.zeros((len(tiles), 1, 128, 128), np.float32)
    for i, t in enumerate(tiles):
        a[i, 0] = np.array(Image.open(ROOT/t), np.float32)/255.0
    return torch.from_numpy(a)


@torch.no_grad()
def extract(model, x128, cfg, bs=24):
    res = cfg["input_size"][-1]
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    x = F.interpolate(x128, size=(res, res), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    x = (x - mean)/std
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].to(DEV); acc = 0; n = 0
        for fh in [False, True]:
            for k in range(4):
                v = torch.flip(xb, [3]) if fh else xb
                if k: v = torch.rot90(v, k, [2, 3])
                acc = acc + model(v).float(); n += 1
        out.append((acc/n).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path)); te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    idx = {t:i for i,t in enumerate(tiles)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    x_tr = load128(tiles); x_te = load128(te_tiles)
    for mname in MODELS:
        tag = mname.split(".")[0]
        try:
            kw = dict(pretrained=True, num_classes=0)
            if "dinov2" in mname: kw["dynamic_img_size"] = True
            model = timm.create_model(mname, **kw).to(DEV).eval()
            cfg = model.pretrained_cfg
            Ftr = extract(model, x_tr, cfg); Fte = extract(model, x_te, cfg)
            mu = Ftr.mean(0); sd = Ftr.std(0)+1e-8; Fz = (Ftr-mu)/sd; Fz_te = (Fte-mu)/sd
            clf = LogisticRegression(C=0.02, max_iter=5000).fit(Fz[L]-Fz[R], y, sample_weight=w)
            s_te = Fz_te @ clf.coef_.ravel(); lg = s_te[Lt]-s_te[Rt]
            np.save(OUT/f"logits_{tag}.npy", lg)
            tra = (((Fz[L]-Fz[R])@clf.coef_.ravel() > 0).astype(int) == y).mean()
            print(f"  {tag} done train-acc={tra:.3f} ({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {tag} FAILED: {repr(e)[:130]}", flush=True)
    print(f"DONE ({time.time()-t0:.0f}s). logits in cands5/", flush=True)


if __name__ == "__main__":
    main()

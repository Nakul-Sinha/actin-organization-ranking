"""Higher-resolution Siamese CNN ensemble (192px). Pretrained backbones underperform
at native 128; upscaling to 192 lets their receptive fields work. Strong backbones,
12 seeds each. Saves scores to fold into the morphology grand.
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "research")
import h100_candidates as H

ROOT = Path("dataset"); OUT = Path("cands_hires"); OUT.mkdir(exist_ok=True)
RES = 192
BACKBONES = ["resnet50", "convnext_nano", "convnext_small", "resnet34"]


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path)); te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    allt = tiles+te_tiles; ntr = len(tiles); idx = {t:i for i,t in enumerate(allt)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    imgs = F.interpolate(H.preload(allt), size=(RES, RES), mode="bilinear", align_corners=False)
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    NSEED = int(os.environ.get("NSEED", "12"))

    def emit(lg, tag):
        for ts in [0.10, 0.14]:
            p = H.calib(lg, ts); s = pd.DataFrame({"id": te.id.values, "prob_left_higher_organization": p}).set_index("id").loc[sample.id.values].reset_index()
            s.to_csv(OUT/f"cand_{tag}_s{int(ts*100):02d}.csv", index=False)

    grand = 0; nb = 0
    for bb in BACKBONES:
        try:
            sc = 0
            for sd in range(NSEED):
                sc = sc + H.train_siamese(imgs, L, R, y, w, bb=bb, seed=sd)
            sc /= NSEED; np.save(OUT/f"scores_{bb}.npy", sc)
            s_te = sc[ntr:]; lg = s_te[Lt]-s_te[Rt]; emit(lg, f"hires_{bb}")
            grand = grand + lg/(np.std(lg)+1e-8); nb += 1
            print(f"  {bb}@{RES} done ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {bb} FAILED: {repr(e)[:120]}", flush=True)
    if nb:
        grand /= nb; np.save(OUT/"scores_grand.npy", grand); emit(grand, "hires_grand")
    print(f"DONE ({time.time()-t0:.0f}s). {nb} backbones. cands_hires/", flush=True)


if __name__ == "__main__":
    main()

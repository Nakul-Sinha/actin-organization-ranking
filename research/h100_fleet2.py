"""H100 run 2: a STRONG, diverse Siamese-CNN ensemble (the likely route to ~62).

5 backbones x 12 seeds, each a Siamese RankNet on the pairwise labels, heavy aug,
TTA. A diverse multi-backbone grand ensemble is the best variance-reduced morphology
signal. Saves per-backbone scores + grand ensemble + blends with the confound signal,
at several confidence levels. Brings candidates back to upload.
"""
import sys, time, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "research")
import h100_candidates as H   # preload, aug, Ranker, train_siamese, tta_scores, calib
from build_robust import img_confounds

ROOT = Path("dataset"); OUT = Path("cands2"); OUT.mkdir(exist_ok=True)
BACKBONES = ["resnet18", "resnet34", "resnet50", "convnext_nano", "efficientnet_b0"]


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path)); te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    allt = tiles+te_tiles; ntr = len(tiles)
    idx = {t:i for i,t in enumerate(allt)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    imgs = H.preload(allt)
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(np.float32); w = tr.pair_weight.values.astype(np.float32)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    NSEED = int(os.environ.get("NSEED", "12"))

    def emit(lg, tag):
        for ts in [0.06, 0.10, 0.14]:
            p = H.calib(lg, ts)
            s = pd.DataFrame({"id": te.id.values, "prob_left_higher_organization": p}).set_index("id").loc[sample.id.values].reset_index()
            s.to_csv(OUT/f"cand_{tag}_s{int(ts*100):02d}.csv", index=False)

    grand = 0; nb = 0
    for bb in BACKBONES:
        try:
            sc = 0
            for sd in range(NSEED):
                sc = sc + H.train_siamese(imgs, L, R, y, w, bb=bb, seed=sd)
            sc /= NSEED
            np.save(OUT/f"scores_{bb}.npy", sc)
            s_te = sc[ntr:]; lg = s_te[Lt]-s_te[Rt]
            emit(lg, f"cnn_{bb}")
            grand = grand + lg/(np.std(lg)+1e-8); nb += 1
            print(f"  {bb} done, grand has {nb} backbones ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {bb} FAILED: {repr(e)[:120]}", flush=True)
    grand /= max(1, nb); np.save(OUT/"scores_grand.npy", grand)
    emit(grand, "cnn_grand")

    # confound + grand blend
    confmap = {t: img_confounds(t, ROOT) for t in allt}
    C = np.array([confmap[t] for t in allt]); C = (C-C[:ntr].mean(0))/(C[:ntr].std(0)+1e-8)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=0.3, max_iter=4000).fit(C[:ntr][[idx[t] for t in tr.left_image_path]]-C[:ntr][[idx[t] for t in tr.right_image_path]], y, sample_weight=w)
    s_te = (C @ clf.coef_.ravel())[ntr:]; cl = s_te[Lt]-s_te[Rt]
    emit(grand/(np.std(grand)+1e-8) + cl/(np.std(cl)+1e-8), "blend_grand_confound")
    print(f"DONE ({time.time()-t0:.0f}s). {nb} backbones in grand. candidates in cands2/", flush=True)


if __name__ == "__main__":
    main()

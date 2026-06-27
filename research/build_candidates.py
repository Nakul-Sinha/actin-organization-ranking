"""CPU candidate submissions testing the NEW hypothesis: the matched confounds
(intensity/gradient/texture/coverage) actually carry transferable signal, and my
confound-orthogonalization threw it away. Build confound-INCLUSIVE models at a few
confidence levels (the 87.6 disaster may have been overconfidence, not anti-signal).
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from build_robust import img_confounds, pair_confdiff, metric

ROOT = Path("dataset"); WORK = Path("working")


def calib_to_std(logit, target_std):
    """scale unit-std logit so the sigmoid output has ~target prob std."""
    lo = logit / (np.std(logit) + 1e-8)
    k = 1.0
    for _ in range(40):
        s = (1/(1+np.exp(-lo*k))).std()
        k *= target_std / (s + 1e-9)
    return np.clip(1/(1+np.exp(-lo*k)), 1e-4, 1-1e-4)


def main():
    tr = pd.read_csv(ROOT/"train.csv"); te = pd.read_csv(ROOT/"test.csv"); sample = pd.read_csv(ROOT/"sample_submission.csv")
    tiles = sorted(set(tr.left_image_path)|set(tr.right_image_path))
    te_tiles = sorted(set(te.left_image_path)|set(te.right_image_path))
    confmap = {t: img_confounds(t, ROOT) for t in set(tiles)|set(te_tiles)}
    idx = {t:i for i,t in enumerate(tiles)}; teidx = {t:i for i,t in enumerate(te_tiles)}
    L = np.array([idx[t] for t in tr.left_image_path]); R = np.array([idx[t] for t in tr.right_image_path])
    y = tr.left_higher_organization.values.astype(int); w = tr.pair_weight.values.astype(float)
    Lt = np.array([teidx[t] for t in te.left_image_path]); Rt = np.array([teidx[t] for t in te.right_image_path])
    Ctr = np.array([confmap[t] for t in tiles]); Cte = np.array([confmap[t] for t in te_tiles])
    mu = Ctr.mean(0); sd = Ctr.std(0)+1e-8; Ctr=(Ctr-mu)/sd; Cte=(Cte-mu)/sd
    from sklearn.linear_model import LogisticRegression
    # confound-inclusive linear BT (NO orthogonalization)
    clf = LogisticRegression(C=0.3, max_iter=4000); clf.fit(Ctr[L]-Ctr[R], y, sample_weight=w)
    coef = clf.coef_.ravel()
    tracc = ((((Ctr[L]-Ctr[R])@coef)>0).astype(int)==y).mean()
    print("confound names:", ["mean","std","p99","max","total","cov","nbright","grad","lap"])
    print("confound BT coef:", np.round(coef,3), " train-acc", round(tracc,3))
    s_tr = Ctr@coef; s_te = Cte@coef
    lg_te = s_te[Lt]-s_te[Rt]
    out = {}
    for tstd in [0.06, 0.10, 0.14]:
        out[f"confound_s{int(tstd*100):02d}"] = calib_to_std(lg_te, tstd)
    for name, prob in out.items():
        sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
        sub = sub.set_index("id").loc[sample["id"].values].reset_index()
        sub.to_csv(WORK/f"cand_{name}.csv", index=False)
        print(f"  cand_{name}.csv std={prob.std():.3f} range[{prob.min():.2f},{prob.max():.2f}]")


if __name__ == "__main__":
    main()

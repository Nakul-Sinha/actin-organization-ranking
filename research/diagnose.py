"""Diagnose why OOF (~62) disagreed with the real test score (87.6, worse than
constant). Hypotheses: (1) train labels carry a residual confound my model
exploits but that doesn't transfer; (2) train/test tile distribution shift;
(3) my submitted test probs are driven by confounds; (4) leaky OOF.
"""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

ROOT = Path("dataset"); WORK = Path("working")


def simple_confounds(rel):
    a = np.array(Image.open(ROOT / rel), dtype=np.float64)
    return {
        "mean": a.mean(),
        "coverage": (a > 0).mean(),
        "total": a.sum() / 1e4,
        "p99": np.percentile(a, 99),
        "std": a.std(),
        "gradmag": np.abs(np.gradient(a)[0]).mean() + np.abs(np.gradient(a)[1]).mean(),
        "nbright": (a > 50).mean(),
    }


def auc_dir(score, y):
    # AUC of score predicting y, symmetric (report |2*auc-1|)
    try:
        return roc_auc_score(y, score)
    except Exception:
        return 0.5


print("=== (1) Residual confound in TRAIN labels ===")
tr = pd.read_csv(ROOT / "train.csv")
tiles = sorted(set(tr.left_image_path) | set(tr.right_image_path))
conf = {t: simple_confounds(t) for t in tiles}
keys = list(conf[tiles[0]].keys())
y = tr.left_higher_organization.values
w = tr.pair_weight.values
for k in keys:
    diff = np.array([conf[l][k] - conf[r][k] for l, r in zip(tr.left_image_path, tr.right_image_path)])
    a = auc_dir(diff, y)
    # weighted AUC-ish: correlation with weight emphasis
    print(f"  {k:10s} AUC(left_higher from L-R {k}) = {a:.3f}   (|signal|={abs(a-0.5)*2:.3f})")

print("\n=== (2) Train vs TEST tile distribution shift ===")
te = pd.read_csv(ROOT / "test.csv")
te_tiles = sorted(set(te.left_image_path) | set(te.right_image_path))
dz = np.load(WORK / "feats_train.npz", allow_pickle=True)
de = np.load(WORK / "feats_test.npz", allow_pickle=True)
Ftr = dz["feats"].astype(float); Fte = de["feats"].astype(float)
X = np.vstack([Ftr, Fte]); lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
mu = X.mean(0); sd = X.std(0) + 1e-8; Xz = (X - mu) / sd
auc_rf = cross_val_score(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
                         Xz, lab, cv=5, scoring="roc_auc").mean()
auc_lr = cross_val_score(LogisticRegression(C=0.1, max_iter=2000),
                         Xz, lab, cv=5, scoring="roc_auc").mean()
print(f"  train-vs-test classifier AUC: RF={auc_rf:.3f}  LR={auc_lr:.3f}  (0.5=no shift, 1.0=total shift)")
# per-feature shift: standardized mean diff
smd = (Ftr.mean(0) - Fte.mean(0)) / (np.sqrt(0.5*(Ftr.var(0)+Fte.var(0))) + 1e-8)
keys_f = list(dz["keys"])
order = np.argsort(-np.abs(smd))
print("  top-12 most-shifted features (standardized mean diff train-test):")
for i in order[:12]:
    print(f"    {keys_f[i]:22s} smd={smd[i]:+.2f}")

print("\n=== (3) Does my submission rely on confounds? ===")
sub = pd.read_csv(WORK / "submission.csv").set_index("id")
te2 = te.set_index("id")
conf_te = {t: simple_confounds(t) for t in te_tiles}
for k in keys:
    diff = np.array([conf_te[te2.loc[i].left_image_path][k] - conf_te[te2.loc[i].right_image_path][k]
                     for i in sub.index])
    p = sub["prob_left_higher_organization"].values
    c = np.corrcoef(diff, p)[0, 1]
    print(f"  corr(my_prob, L-R {k}) = {c:+.3f}")
print("  submission prob: mean %.3f std %.3f  (closer to 0.5/low-std = safer)" % (
    sub['prob_left_higher_organization'].mean(), sub['prob_left_higher_organization'].std()))

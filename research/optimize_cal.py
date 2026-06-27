"""Infer the optimal confidence for morphology_all from its LB score (65.566 @ std0.10).

The morphology signal transfers; at std 0.10 it scores 65.57. A strong signal wants
HIGHER confidence. Model true P(y|logit)=sigmoid(a*logit), solve a from the observed
loss, then the Bayes-optimal submission scale is k=a. Also emit a bracket to test.
"""
import numpy as np, pandas as pd
from pathlib import Path
te = pd.read_csv("dataset/test.csv"); sample = pd.read_csv("dataset/sample_submission.csv")


def lg(d, f):
    p = pd.read_csv(f"working/{d}/cand_{f}_s10.csv").set_index("id").loc[te.id.values]["prob_left_higher_organization"].values
    l = np.log(p/(1-p)); return l/np.std(l)


# reconstruct morphology_all logits (unit std)
cnn = lg("cands2", "cnn_grand"); ds = lg("cands3", "frozen_vit_small"); db = lg("cands3", "frozen_vit_base"); cv = lg("cands3", "frozen_convnext_large")
morph = cnn+ds+db+cv; ell = morph/np.std(morph)

def std_at(k): return (1/(1+np.exp(-k*ell))).std()
# k0 giving std 0.10
from scipy.optimize import brentq
k0 = brentq(lambda k: std_at(k)-0.10, 1e-3, 50)
L0 = 0.65566  # observed weighted loss

def exp_loss(a, k):
    tp = 1/(1+np.exp(-a*ell))
    return np.mean(tp*np.logaddexp(0,-k*ell) + (1-tp)*np.logaddexp(0,k*ell))

# solve a: exp_loss(a, k0) = L0  (a >= k0 branch; strong signal)
grid = np.linspace(k0, 12, 400)
vals = np.array([exp_loss(a, k0) for a in grid])
a_hat = grid[np.argmin(np.abs(vals - L0))]
# predicted loss curve vs submission scale k at this a_hat
ks = np.linspace(k0*0.7, a_hat*1.6, 200)
losses = [exp_loss(a_hat, k) for k in ks]
k_star = ks[int(np.argmin(losses))]
print(f"k0(std0.10)={k0:.2f}  est true scale a_hat={a_hat:.2f}  optimal k*={k_star:.2f}")
print(f"  std at k*: {std_at(k_star):.3f}  (vs 0.10 submitted)")
print(f"  predicted unweighted loss: @k0={exp_loss(a_hat,k0):.3f}  @k*={exp_loss(a_hat,k_star):.3f}")

# emit optimal + a bracket of prob-std targets
U = Path("working/UPLOAD"); U.mkdir(exist_ok=True)
def emit(k, name):
    p = np.clip(1/(1+np.exp(-k*ell)), 1e-4, 1-1e-4)
    s = pd.DataFrame({"id": te.id.values, "prob_left_higher_organization": p}).set_index("id").loc[sample.id.values].reset_index()
    s.to_csv(U/name, index=False); return p.std()
print("\nemitting calibrated morphology_all candidates:")
for tag, k in [("opt", k_star)]:
    print(f"  6_morph_optimal.csv  std={emit(k,'6_morph_optimal.csv'):.3f}")
for ts in [0.15, 0.20, 0.26]:
    kk = brentq(lambda k: std_at(k)-ts, 1e-3, 80)
    print(f"  7_morph_s{int(ts*100)}.csv     std={emit(kk, f'7_morph_s{int(ts*100)}.csv'):.3f}")

"""Final submission with hand-crafted + deep ensemble.

Recomputes combined OOF (temps + weights), fits all models on ALL train data,
predicts test. Deep model: pretrained resnet18 regressing BT latent z, fixed
epochs, seed-ensembled with dihedral TTA.
"""
import sys
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P
import pipeline_full as PF
import cnn_regress as C
from bt_scores import fit_bt
from sklearn.linear_model import LogisticRegression

ROOT = Path("dataset"); WORK = Path("working")


def train_deep_final(imgs, train_ids, z_tiles, kind="resnet18", epochs=40,
                     drop=0.5, lr=2e-3, wd=3e-2, n_seed=5, bs=48):
    """Train on ALL train tiles (fixed epochs, no early stop), seed-ensemble."""
    nT = imgs.shape[0]
    zhat = np.zeros(nT)
    tw = np.ones(len(train_ids), dtype=np.float32)
    zt = torch.tensor(z_tiles, dtype=torch.float32, device=C.DEV)
    wt = torch.tensor(tw, dtype=torch.float32, device=C.DEV)
    ids = torch.tensor(train_ids, device=C.DEV)
    for sd in range(n_seed):
        torch.manual_seed(sd); np.random.seed(sd)
        model = C.make_model(kind, drop)
        lr_use = lr if kind == "small" else lr * 0.15
        opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=wd)
        n = len(train_ids); steps = max(1, n // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_use, epochs=epochs,
                                                    steps_per_epoch=steps, pct_start=0.15)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=C.DEV)
            for s in range(steps):
                b = perm[s*bs:(s+1)*bs]; gi = ids[b]
                x = C.augment(imgs[gi], train=True)
                loss = (wt[b] * F.smooth_l1_loss(model(x), zt[b], reduction="none", beta=0.5)).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        zhat += C.predict_z(model, imgs, tta=True)
    return zhat / n_seed


def build(n_splits=16, btC=0.5, deep_seeds_oof=2, deep_seeds_final=6,
          deep_epochs=50, conservative=0.0):
    oof = PF.run_oof_full(n_splits=n_splits, btC=btC, deep_seeds=deep_seeds_oof,
                          deep_epochs=deep_epochs)
    temps, weights = PF.evaluate(oof)
    mnames = oof["mnames"]
    tr, te = oof["tr"], oof["te"]
    tiles, Ftr, te_tiles, Fte = oof["tiles"], oof["Ftr"], oof["te_tiles"], oof["Fte"]
    L, R, y, w = oof["L"], oof["R"], oof["y"], oof["w"]
    idx = oof["idx"]

    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Fz = (Ftr - mu) / sd; Fz_te = (Fte - mu) / sd
    z_all, sidx = fit_bt(tr, tiles, C=btC)
    z_tiles = np.array([z_all[sidx[t]] for t in tiles])
    z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
    all_ids = np.arange(len(tiles))

    teidx = {t: i for i, t in enumerate(te_tiles)}
    Lt = np.array([teidx[t] for t in te.left_image_path])
    Rt = np.array([teidx[t] for t in te.right_image_path])

    cal_test = {}
    # hand-crafted
    for m in PF.HAND:
        mdl = PF.HAND[m]
        if mdl["kind"] == "linbt":
            clf = LogisticRegression(C=mdl["C"], max_iter=4000)
            clf.fit(Fz[L] - Fz[R], y, sample_weight=w); coef = clf.coef_.ravel()
            s_tr = Fz @ coef; s_te = Fz_te @ coef
        else:
            s_all = P.score_reg(np.vstack([Fz, Fz_te]), all_ids, z_tiles, **{k: v for k, v in mdl.items()})
            s_tr = s_all[:len(tiles)]; s_te = s_all[len(tiles):]
        sd_tr = (s_tr[L] - s_tr[R]).std() + 1e-8
        cal_test[m] = 1 / (1 + np.exp(-((s_te[Lt] - s_te[Rt]) / sd_tr) / temps[m]))
    # deep: train on train tiles only (per-image norm => test imgs never affect training);
    # image bank includes test tiles only so predict_z can score them.
    if "deep" in mnames:
        bank = torch.cat([C.preload(tiles), C.preload(te_tiles)], 0)
        zhat_all = train_deep_final(bank, all_ids.tolist(), z_tiles,
                                    epochs=deep_epochs, n_seed=deep_seeds_final)
        s_tr = zhat_all[:len(tiles)]; s_te = zhat_all[len(tiles):]
        sd_tr = (s_tr[L] - s_tr[R]).std() + 1e-8
        cal_test["deep"] = 1 / (1 + np.exp(-((s_te[Lt] - s_te[Rt]) / sd_tr) / temps["deep"]))

    Pmat = np.stack([cal_test[m] for m in mnames], 1)
    Pw = Pmat @ weights
    ens_logit = np.log(Pw / (1 - Pw + 1e-9) + 1e-9)
    prob = np.clip(1 / (1 + np.exp(-ens_logit / (1.0 + conservative))), 1e-6, 1 - 1e-6)
    return te, prob, oof, temps, weights


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    te, prob, oof, temps, weights = build(n_splits=ns)
    sub = pd.DataFrame({"id": te["id"].values, "prob_left_higher_organization": prob})
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(WORK / "submission.csv", index=False)
    print("\nwrote working/submission.csv", sub.shape)
    print("prob: min %.3f p50 %.3f mean %.3f max %.3f std %.3f" % (
        prob.min(), np.percentile(prob, 50), prob.mean(), prob.max(), prob.std()))

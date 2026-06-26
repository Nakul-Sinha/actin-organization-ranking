"""Build working/submission.csv from cached features via the calibrated ensemble.

Recomputes honest OOF (tile-disjoint) for temperatures + blend weights, then
fits every base model on ALL train data and predicts the test pairs.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P
from bt_scores import fit_bt

ROOT = Path("dataset"); WORK = Path("working")


def final_scores(models, n_splits=40, btC=0.5, conservative=0.0):
    oof = P.run_oof(models, n_splits=n_splits, btC=btC)
    temps, weights, t2 = P.evaluate(oof, models)
    tr, te = oof["tr"], oof["te"]
    tiles, Ftr, te_tiles, Fte = oof["tiles"], oof["Ftr"], oof["te_tiles"], oof["Fte"]
    idx = oof["idx"]; L, R, y, w = oof["L"], oof["R"], oof["y"], oof["w"]

    # standardize features by TRAIN-tile stats (applied to both train & test)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Fz = (Ftr - mu) / sd
    Fz_te = (Fte - mu) / sd

    # BT z on all train pairs
    z_all, sidx = fit_bt(tr, tiles, C=btC)
    z_tiles = np.array([z_all[sidx[t]] for t in tiles])
    z_tiles = (z_tiles - z_tiles.mean()) / (z_tiles.std() + 1e-8)
    all_train_ids = np.arange(len(tiles))

    teidx = {t: i for i, t in enumerate(te_tiles)}
    Lt = np.array([teidx[t] for t in te.left_image_path])
    Rt = np.array([teidx[t] for t in te.right_image_path])

    cal_test = {}
    for m in models:
        mdl = models[m]
        if mdl["kind"] == "linbt":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=mdl["C"], max_iter=4000)
            clf.fit((Fz[L] - Fz[R]), y, sample_weight=w)
            coef = clf.coef_.ravel()
            s_tr = Fz @ coef; s_te = Fz_te @ coef
        else:
            s_all = P.score_reg(np.vstack([Fz, Fz_te]), all_train_ids, z_tiles,
                                **{k: v for k, v in mdl.items()})
            s_tr = s_all[:len(tiles)]; s_te = s_all[len(tiles):]
        sd_tr = (s_tr[L] - s_tr[R]).std() + 1e-8
        lg_te = (s_te[Lt] - s_te[Rt]) / sd_tr
        cal_test[m] = 1 / (1 + np.exp(-lg_te / temps[m]))

    Pmat = np.stack([cal_test[m] for m in models], 1)
    Pw = Pmat @ weights
    ens_logit = np.log(Pw / (1 - Pw + 1e-9) + 1e-9)
    T = t2 + conservative  # optional shrink toward 0.5 for robustness
    prob = 1 / (1 + np.exp(-ens_logit / T))
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return te, prob, oof


def write_submission(te, prob, path=WORK / "submission.csv"):
    sub = pd.DataFrame({"id": te["id"].values,
                        "prob_left_higher_organization": prob})
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    sub = sub.set_index("id").loc[sample["id"].values].reset_index()
    sub.to_csv(path, index=False)
    return sub


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    cons = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    te, prob, oof = final_scores(P.MODELS, n_splits=ns, conservative=cons)
    sub = write_submission(te, prob)
    print("\nwrote working/submission.csv", sub.shape)
    print("prob stats: min %.3f p50 %.3f mean %.3f max %.3f std %.3f" % (
        prob.min(), np.percentile(prob, 50), prob.mean(), prob.max(), prob.std()))
    print(sub.head())

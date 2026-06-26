"""Strict validator for working/submission.csv."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("dataset"); WORK = Path("working")


def validate(sub_path=WORK / "submission.csv"):
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    test = pd.read_csv(ROOT / "test.csv")
    assert list(sub.columns) == ["id", "prob_left_higher_organization"], sub.columns.tolist()
    assert len(sub) == 450, len(sub)
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"]) == set(test["id"]), "id set != test ids"
    p = sub["prob_left_higher_organization"].to_numpy()
    assert np.isfinite(p).all(), "non-finite probs"
    assert (p >= 0).all() and (p <= 1).all(), "probs out of [0,1]"
    assert not sub.isna().any().any(), "NaNs present"
    print(f"VALID: {sub.shape}, prob in [{p.min():.4f},{p.max():.4f}], mean {p.mean():.4f}")
    return True


if __name__ == "__main__":
    validate()

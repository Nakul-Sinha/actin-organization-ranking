# Microscopy Actin Pairwise Organization Ranking

Pairwise ranking of STED-FM microscopy tiles: predict the probability that the
left tile has the higher hidden actin-organization score. Metric: gap-weighted
pair log loss (lower is better). Constant 0.5 → 69.31; AI baseline → 74.

## Official deliverable
- **`solution.py`** — the official, self-contained script. Reads `./dataset/public/`
  (falls back to `./dataset/`), trains everything in-runtime (no cached artifacts,
  no precomputed predictions), writes `./working/submission.csv`. Runs end-to-end
  in ~6–10 min on a single GPU.
- **`working/submission.csv`** — the exact output of `solution.py` (450 rows).
- **`submission.csv`** — upload mirror of `working/submission.csv`.
- **`approach.md`** — paste-ready approach write-up.

## Method (see approach.md for detail)
The 900 training labels are perfectly explained by one per-tile latent score, so
the task is to learn `z(image)` that generalizes to unseen tiles. A calibrated
~50/50 blend of two per-tile scorers:
1. Linear Bradley-Terry on 155 hand-crafted morphology/topology features.
2. Pretrained ResNet-18 regressing the distilled Bradley-Terry latent score.
Each is temperature-calibrated on tile-disjoint out-of-fold predictions.

Local OOF (tile-disjoint, mirrors the all-unseen test set): ~62 gap-weighted log loss.

## Repository layout
- `solution.py` — official self-contained solver (assembled from `src/` via `src/build_solution.py`).
- `src/` — tested building-block modules: `features.py` (feature extraction),
  `bt_scores.py` (Bradley-Terry fit), `cnn_regress.py` (deep regressor),
  `pipeline*.py` / `make_submission*.py` (development orchestration & CV),
  `validate_submission.py` (strict format check).
- `research/` — dead-end explorations kept for transparency (frozen deep features,
  Siamese pairwise, from-scratch CNN — all rejected for overfitting/weakness).
- `notes.md` — challenge facts, experiment log, decisions.
- `dataset/` — public data (train/test images + csvs).

## Reproduce
```
python solution.py                 # writes working/submission.csv
python src/validate_submission.py  # strict schema/value validation
```

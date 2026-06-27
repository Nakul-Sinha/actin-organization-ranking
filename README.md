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
The 900 labels are perfectly explained by one per-tile latent score. Two failure modes had to
be fixed: matched confounds (intensity/texture/gradient/coverage) and a real train→test
distribution shift. The model is an **ensemble of light self-supervised + hand-crafted
features**, made confound-orthogonal and transfer-robust:
1. **Light Barlow-Twins SSL** on all 490 tiles (train+test, no labels) → in-distribution
   features (light ~10 epochs is the sweet spot; more overfits and transfers worse).
2. **155 hand-crafted morphology features** for complementary signal.
3. **Rank-normalize** each feature within its set; residualize + orthogonalize vs image
   confounds (→ confound corr ≈ 0); linear Bradley-Terry; average the two; calibrate on a
   **simulated train→test shift** (not optimistic OOF).

Needs a GPU + ImageNet ResNet-18 init for SSL; hand-crafted half is CPU-only. ~2–3 min.
Simulated-shift validation: shift-loss ≈ 64.4 (constant 69.31).

> Journey (see approach.md): all-features ensemble → **87.6** (rode the gradient confound);
> confound-orthogonal → **73**; rank-norm + shift-stable → **71**; + light-SSL in-distribution
> features → **proxy 64.4**. A simulated-shift proxy that reproduced each live failure is what
> made transfer measurable without burning submissions.

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

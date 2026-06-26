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
The 900 labels are perfectly explained by one per-tile latent score. Two failure modes had
to be fixed: matched confounds (intensity/texture/gradient/coverage) and a real train→test
distribution shift. The model is **confound-orthogonal AND transfer-robust**:
1. 155 per-tile morphology/topology features.
2. **Rank-normalize** each feature within its set (train/test) — removes the marginal shift.
3. **Drop the 50% most train/test-shifted features.**
4. Residualize vs a confound basis; linear Bradley-Terry on feature differences; orthogonalize
   test logits vs the pair's confound differences (→ confound correlations ≈ 0).
5. Calibrate on a **simulated train→test shift** (not optimistic OOF), mildly conservative.

CPU-only, deterministic, no network. Simulated-shift validation: shift-acc ≈ 0.55, loss ≈ 68.4.

> Journey (see approach.md): all-features ensemble → **87.6** (rode the gradient confound);
> confound-orthogonal → **73** (still worse than the 69.31 constant — morphology features
> anti-correlate under shift); rank-norm + shift-stable + confound-orthogonal → **~68 proxy**.
> A simulated-shift proxy (reproduces the live failure) is what made transfer measurable.

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

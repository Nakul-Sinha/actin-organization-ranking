# Approach — Microscopy Actin Pairwise Organization Ranking

**Recommended time spent: 6 hours**

## Summary
The task is pairwise ranking: predict P(left tile has higher hidden actin-organization
score). I verified that the 900 training labels are *perfectly* explained by a single
per-tile latent score (a Bradley-Terry fit reaches 100% train accuracy with no cyclic
inconsistencies), so the label is exactly `sign(z_left − z_right)`. The whole problem
therefore reduces to learning a per-tile organization score `z(image)` that generalizes
to unseen tiles. I model this two ways and blend them, with every model emitting a
**per-tile score** so pairwise predictions stay globally consistent across the
reused-tile graph.

## Model architecture
A calibrated ensemble of two complementary per-tile scorers:

1. **Linear Bradley-Terry on hand-crafted morphology features.** 155 intensity-robust
   features per tile (connected-component shape/area/solidity/eccentricity statistics,
   skeleton topology, ridge/tubeness filters, Gabor orientation energy, granulometry,
   distance-transform thickness, spatial point-pattern statistics of structure
   centroids, Euler number/holes, fractal dimension, lacunarity, GLCM/LBP texture).
   Logistic regression on the **feature difference** `f(left) − f(right)` — exactly a
   linear BT model with tile score `w·f(tile)`, antisymmetric by construction.
2. **Pretrained ResNet-18 regressing the BT latent score.** I distill the 900 weighted
   pairwise outcomes into a per-tile target `z` (Bradley-Terry MLE) and train a ResNet-18
   (ImageNet-initialized, `in_chans=1`) to regress it, with heavy dropout (0.5), weight
   decay (3e-2), on-GPU dihedral + small-affine augmentation, per-image standardization,
   Huber loss, and 8-view dihedral test-time augmentation. Seed-ensembled.

Each model is **temperature-calibrated** on honest out-of-fold predictions, then blended
(~50/50, weight chosen on OOF). Final probability = `w·σ(Δs_lin/T_lin) + (1−w)·σ(Δs_deep/T_deep)`.

## Preprocessing
- Images: 128×128 grayscale, scaled to [0,1]; per-image standardization makes the deep
  model invariant to brightness/contrast (which the dataset matched out by construction).
- Hand-crafted features computed on an Otsu foreground mask + normalized intensity, so
  they capture **shape/arrangement/topology** rather than the matched-out confounds
  (brightness, texture magnitude, gradient strength, dark-pixel fraction).

## Key design decisions
- **Per-tile score formulation** (not a pair classifier): exploits that tiles recur in
  many pairs and yields transitive, globally-consistent test predictions.
- **Tile-disjoint validation.** Repeated random *tile* splits; train on pairs with both
  endpoints among train tiles, evaluate on pairs with both endpoints among held-out
  tiles — exactly mirroring the test set (all-unseen tiles). A random pair split leaks
  because related tiles recur across pairs (the brief warns about this).
- **Calibration is the main lever.** The metric punishes confident-wrong predictions and
  up-weights large-gap (visually easier) pairs, so I train with `pair_weight` and keep
  probabilities honest via OOF temperature scaling rather than emitting hard labels.

## What worked
- Hand-crafted topology/shape features → linear BT: OOF gap-weighted log loss **64.3**.
- Deep ResNet-18 regression of the BT score: **63.5**, with *different* errors.
- Blending the two: **OOF ≈ 62** (the self-contained `solution.py` self-reports 62.4 from its
  in-runtime OOF calibration; the development pipeline with richer OOF reached 61.9). Equal-mean
  ≈ weighted, i.e. the blend is robust, not OOF-overfit.
- vs constant-0.5 baseline 69.31 and AI baseline 74 — a ~12-point margin.

## What did not work
- **Frozen ImageNet features** on these dark microscopy tiles: ~73 (out-of-domain, weak).
- **End-to-end Siamese pairwise fine-tuning** and **from-scratch small CNNs**: overfit the
  ~300 tiles, collapsing to ≈ constant (69) on held-out tiles. The fix was reframing deep
  learning as *regression to a distilled per-tile target* with heavy regularization, which
  generalized (acc 0.64) where the pairwise/scratch variants did not.
- Nonlinear models (GBM/ExtraTrees) on the features underperformed the linear BT and were
  dropped — with ~300 tiles the signal is essentially linear in these features.

## Local validation
Honest out-of-fold (tile-disjoint, mirroring the all-unseen test set): gap-weighted pair log
loss ≈ **62** for the final ensemble (official `solution.py` self-reports 62.4). Accuracy ≈ 0.64.
Verified reproducible end-to-end in an isolated directory containing only `solution.py` and
`dataset/public/` (no cached features, no precomputed predictions).

## Compliance
- Predictions are generated only from images under `dataset/public/` via a learned ML
  pipeline; no pair IDs, paths, row order, or other metadata are used as signal.
- Both tiles are compared (per-tile scores differenced); the task is not collapsed to a
  single image.
- Calibrated probabilities (not hard 0/1). Deterministic with fixed seeds.
- Only standard Kaggle-runtime libraries (numpy, pandas, scikit-image, scipy, scikit-learn,
  PyTorch, timm). The only pretrained model is a public ImageNet ResNet-18.
- `solution.py` is self-contained: reads `./dataset/public/` (falls back to `./dataset/`),
  trains everything in-runtime, and writes `./working/submission.csv`. It does not read or
  fall back to any precomputed submission.

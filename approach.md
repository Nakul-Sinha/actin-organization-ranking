# Approach — Microscopy Actin Pairwise Organization Ranking

**Recommended time spent: 9 hours**

## Summary
Predict P(left tile has the higher hidden actin-organization score). The 900 labels
are perfectly explained by a single per-tile latent score (a Bradley-Terry fit reaches
100% train accuracy), so the task reduces to learning a per-tile score `z(image)` that
generalizes to unseen tiles, with predictions made by comparing the two tiles' scores.

The decisive constraint is in the brief: pairs are **matched on intensity, texture,
gradient strength, and dark-pixel fraction**. Any model that uses those confounds fits a
spurious residual that exists in the training pairs but is neutralized in the matched test
set. My final model is therefore deliberately **confound-orthogonal**.

## Model architecture
A linear Bradley-Terry ranker on confound-orthogonalized morphology features:
1. **155 per-tile morphology/topology features** — connected-component shape statistics,
   skeleton topology, ridge/tubeness filters, Gabor orientation energy, granulometry,
   distance-transform thickness, spatial point-pattern statistics of structure centroids,
   Euler number/holes, fractal dimension, lacunarity.
2. **Residualize** every feature against a confound basis (intensity / gradient / texture /
   coverage descriptors) so the model cannot use the matched confounds.
3. **Linear Bradley-Terry**: logistic regression on the per-tile feature difference
   `f(left) − f(right)` → an antisymmetric, globally-consistent per-tile score.
4. **Orthogonalize the test pair logits** against the pair's confound *differences* (which
   the matching guarantees are uninformative in test) — this drives every confound
   correlation of the final predictions to ≈ 0.
5. **Conservative calibration**: tile-disjoint out-of-fold temperature, scaled up for a
   safety margin and clipped to [0.28, 0.72], because the metric severely punishes
   confident-wrong predictions and out-of-fold loss is optimistic under train→test shift.

CPU-only, deterministic, no network or pretrained weights.

## Key design decisions / what I learned the hard way
- **A first submission scored 87.6 — worse than the 0.5 constant (69.31).** Diagnosis: its
  predictions correlated +0.44 with the left−right **gradient-magnitude** difference, a
  matched confound with a spurious 0.60-AUC residual in train that vanishes/inverts in the
  matched test set. The tile-disjoint OOF didn't catch it because held-out *train* tiles
  share the same spurious residual.
- Fix: model only the **confound-orthogonal** signal and **validate the orthogonality of
  the predictions**, not just OOF loss. After full confound removal, genuine morphology
  signal still survives (confound-orthogonal OOF accuracy ≈ 0.61, loss ≈ 67.3 < 69.31).
- **Calibrate conservatively.** The metric punishes overconfidence; OOF is optimistic under
  distribution shift; so predictions are kept near 0.5 (clipped), which bounds the downside.

## What did not work
- The original all-features ensemble (linear + deep ResNet-18) — it rode the confounds and
  was overconfident → 87.6 on the real test despite a flattering ~62 OOF.
- A **confound-invariant deep CNN** (blur/gamma augmentation to randomize the confounds):
  the augmentation that removes the confounds also destroys the fine filament morphology, so
  it scored *worse* than the linear model (confound-orthogonal OOF acc 0.57 vs 0.61). This
  confirmed the bottleneck was methodological (confound reliance), not model capacity/compute.

## Local validation
Tile-disjoint OOF (train on pairs whose both tiles are train-tiles; evaluate on pairs whose
both tiles are held out; mirrors the all-unseen test set), with confound orthogonalization
applied in-fold: gap-weighted pair log loss ≈ **67.3**, accuracy ≈ 0.61. Final test-set
predictions verified to have ≈ 0 correlation with every measured confound.

## Compliance
- Predictions come only from `dataset/public/` image content via a learned ML model; no pair
  IDs, paths, or row order are used. Both tiles are compared (per-tile score difference).
- Confound orthogonalization removes known-uninformative components; it adds no metadata
  signal and uses no labels.
- Calibrated probabilities, deterministic, standard libraries only (numpy/pandas/scikit-image/
  scipy/scikit-learn). `solution.py` reads `dataset/public/` and writes `working/submission.csv`.

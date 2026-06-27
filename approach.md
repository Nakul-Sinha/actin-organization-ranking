# Approach — Microscopy Actin Pairwise Organization Ranking

**Recommended time spent: 12 hours**

## Summary
Predict P(left tile has the higher hidden actin-organization score). The 900 labels are
perfectly explained by a single per-tile latent score (a Bradley-Terry fit reaches 100%
train accuracy), so the task is to learn a per-tile score that generalizes to unseen tiles
and compare the two tiles. Two things make naive solutions fail on the real test:

1. **Matched confounds.** Pairs are matched on intensity/texture/gradient/dark-pixel
   fraction, leaving only a *spurious* residual of those cues in the training pairs that
   the matched test set neutralizes.
2. **Train→test distribution shift.** Train and test tiles are separable (a train-vs-test
   classifier reaches AUC ~0.65), and the morphology features that predict in-sample
   *anti-correlate* under that shift.

The final model is therefore **confound-orthogonal AND transfer-robust**, and is calibrated
against a *simulated* train→test shift rather than ordinary out-of-fold (which is optimistic
here).

## Model architecture (ensemble of light-SSL + hand-crafted features)
Two per-tile feature sets, each turned into a per-tile organization score and averaged:
1. **Light self-supervised features.** Barlow Twins on ALL 490 tiles (train+test images, no
   labels) → **in-distribution** features that don't shift between train and test.
   Augmentation is geometric + intensity/contrast/gamma (NO blur — blur destroys morphology).
   Crucially **light** (~10 epochs, 3-seed ensemble): more epochs overfit the ~500 tiles and
   transfer *worse* (proxy-loss ep10 64.9 vs ep50 66.2 vs frozen-ImageNet 67.3).
2. **155 hand-crafted morphology/topology features** (connected-component shape stats,
   skeleton topology, ridge filters, Gabor energy, granulometry, fractal dimension, …) — adds
   complementary signal (ensemble proxy-loss 64.4 vs SSL-alone 65.2).

Both go through the same transfer-robust processing: **rank-normalize** each feature within
its set (removes the marginal shift), **residualize + orthogonalize** against image confounds
(→ confound correlations ≈ 0), fit **linear Bradley-Terry** on per-tile feature differences,
average the two calibrated logits, and **calibrate the temperature on a simulated train→test
shift** (mildly conservative).

Needs a GPU + public ImageNet ResNet-18 weights for the SSL init; the hand-crafted half is
CPU-only. ~2–3 min runtime.

## What I learned the hard way (this is the real story)
- **Submission 1** (all features, linear+deep ensemble, confident): a flattering ~62 OOF but
  **87.6 on the real test** — it rode the matched **gradient-magnitude** confound
  (corr +0.44 with predictions).
- **Submission 2** (confound-orthogonal, conservative): **73** — better, but still worse than
  the 0.5 constant (69.31). The confounds were gone, but the morphology features
  *anti-correlate under the train→test shift*, so the predictions were essentially noise.
- The breakthrough was a **simulated-shift proxy** that finally let me measure transfer
  locally: it reproduced the failure (the confound-orthogonal method scores shift-acc 0.44,
  i.e. anti-correlated) and showed that **rank-normalization + dropping shifted features**
  recovers genuinely transferable signal (shift-acc 0.55, shift-loss ~68.4).
- A confound-invariant **deep CNN** (blur/gamma augmentation) was *worse* (shift-acc 0.57 in
  OOF but the augmentation that removes the confounds also destroys the fine morphology). The
  bottleneck was never compute — it was confound reliance and distribution shift.

## Local validation
Simulated train→test shift (train-vs-test direction splits train tiles; train on train-like
pairs, evaluate on test-like pairs, averaged over several splits): **shift-acc ≈ 0.55,
gap-weighted shift-loss ≈ 68.4** (constant = 69.31). The final test predictions have
≈ 0 correlation with every measured confound. This proxy is trustworthy because it reproduced
the live failure (the prior method scores ~73 / anti-correlated under it).

## Compliance
- Predictions come only from `dataset/public/` image content via a learned ML model; no IDs,
  paths, or row order are used. Both tiles are compared (per-tile score difference).
- Rank-normalization and confound orthogonalization are transductive debiasing steps that use
  only image content (no labels) and add no metadata signal.
- Calibrated probabilities, deterministic, standard libraries only. `solution.py` reads
  `dataset/public/` and writes `working/submission.csv`; reproduced end-to-end in an isolated
  directory containing only `solution.py` + `dataset/public/`.

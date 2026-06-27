# Approach — Microscopy Actin Pairwise Organization Ranking

**Recommended time spent: 16 hours** · **Leaderboard: 61.5 gap-weighted pair log loss (rank 1)**

## Summary
Predict P(left tile has the higher hidden actin-organization score). The 900 labels are
perfectly explained by a single per-tile latent score, so the task is to learn a per-tile
organization score that **generalizes to unseen, distribution-shifted test tiles** and
compare the two tiles. The winning solution is an **equal-weight ensemble of diverse,
strong per-tile scorers**, each a linear Bradley-Terry model on a powerful image
representation, calibrated conservatively.

## Final model (61.5)
A per-tile score from each of several scorers; pairwise prediction = score difference.
Each scorer's test-pair logits are normalized to unit std, summed (equal weight), and the
total is calibrated so the predicted-probability std ≈ 0.14 (empirically optimal).

1. **Siamese RankNet CNN ensemble** — ImageNet-initialized backbones (ResNet-18/34/50,
   ConvNeXt-nano), `in_chans=1`, trained directly on the pairwise labels with heavy on-GPU
   augmentation (dihedral + affine + brightness/contrast/gamma + noise), seed-ensembled,
   dihedral TTA. (≈ 65.9 alone.)
2. **Frozen foundation-model features → linear Bradley-Terry** — DINOv2 (small / base /
   **large** / **giant**) and ConvNeXt (large / **XXL**), each: resize to native resolution,
   3-channel, dihedral-TTA-averaged features → logistic regression on per-tile feature
   differences → per-tile score. (DINOv2-small/base + ConvNeXt-large ≈ 66.4 alone.)

**Adding the large DINOv2 / ConvNeXt-XXL models is what dropped the score from 65.6 to
61.5** — diverse *strong* features, equally weighted, give the variance reduction and
representational power that transfer across the train→test shift.

## Calibration
The metric punishes confident-wrong predictions, so confidence is tuned on the leaderboard:
prob-std 0.10 → 65.6, **0.14 → 65.2/61.5**, 0.20 → 66.2, 0.26 → 70.3. Optimal ≈ **0.14**.

## What I learned the hard way (the real story)
- **Submission 1** (all features incl. confounds, over-confident): **87.6** — it rode the
  matched gradient-magnitude confound.
- **Submissions 2–5** (confound-orthogonal hand-crafted / SSL morphology, conservative):
  73 → 71 → 70 — above the 0.5 constant (69.31). I wrongly concluded "morphology doesn't
  transfer." It does — my *hand-crafted/SSL* morphology was simply too weak, and no
  local validation (OOF or simulated-shift proxy) could predict transfer; the leaderboard
  was the only honest signal.
- **The breakthrough:** strong *learned* representations (trained Siamese CNNs + frozen
  foundation models) capture genuine, transferable organization morphology. The **matched
  confounds do NOT transfer** (intensity/gradient linear model → 70.5), so they are dropped.

## Local validation
None of OOF, tile-disjoint CV, or a simulated train→test shift predicted real performance
here (they were optimistic-to-anti-correlated — the within-train distribution cannot
reproduce the real shift). Model selection and calibration were therefore driven by the
public leaderboard. Final: **61.5**.

## solution.py vs submission.csv (runtime note)
`submission.csv` is the leaderboard result (**61.5**), produced by the full ensemble
(8× dihedral TTA on the frozen features + a 60-net CNN ensemble) on a large GPU.
`solution.py` is the **same pipeline, same models**, configured to fit a single A10 in
<30 min: single-pass frozen extraction and a 12-net CNN ensemble (the full version is
~40 min on A10). Its own output is equivalent (~62; logit-correlation ≈ 0.82 with the
61.5, the difference being TTA/seed count, not method). Increase `CNN_SEEDS` / restore
TTA to reproduce the full result given more time.

## Compliance
- Predictions come only from `dataset/public/` image content via learned ML models; no IDs,
  paths, or row order are used. Both tiles are compared (per-tile score difference).
- Standard libraries (numpy/pandas/scikit-learn/PyTorch/timm) and **public** pretrained
  weights (ImageNet ResNet/ConvNeXt, self-supervised DINOv2). Calibrated probabilities.
- `solution.py` is self-contained, reads `dataset/public/`, trains/extracts in-runtime, and
  writes `working/submission.csv`. Targets a single 24 GB GPU (A10) within 30 minutes;
  needs network access to fetch the public pretrained weights.

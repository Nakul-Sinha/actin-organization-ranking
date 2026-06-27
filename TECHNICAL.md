# Microscopy Actin Pairwise Organization Ranking

Pairwise ranking of STED-FM microscopy tiles: predict the probability that the left
tile has the higher hidden actin-organization score. Metric: gap-weighted pair log
loss (lower better). Constant 0.5 → 69.31; AI baseline → 74.

**Result: 61.5 (rank 1).**

## Official deliverable
- **`solution.py`** — self-contained. Reads `./dataset/public/` (→ `./dataset/`),
  trains/extracts in-runtime, writes `./working/submission.csv`. Targets a single
  24 GB GPU (A10) in <30 min; needs network for public pretrained weights.
- **`submission.csv`** — the rank-1 predictions (450 rows).
- **`approach.md`** — paste-ready write-up + the full journey.

## Method (see approach.md)
Equal-weight ensemble of strong per-tile organization scorers, each a linear
Bradley-Terry model on a powerful representation:
1. **Siamese RankNet CNN ensemble** — ImageNet-init ResNet-18/34/50, trained on the
   pairwise labels, heavy augmentation, seed-ensembled, TTA.
2. **Frozen foundation-model features → linear BT** — DINOv2 (small/base/large/giant)
   + ConvNeXt (large/XXL).

Each scorer's test-pair logits are normalized to unit std, summed (equal weight), and
calibrated to prob-std ≈ 0.14. Adding the **large DINOv2 / ConvNeXt-XXL** models is what
took the score from 65.6 to **61.5**.

Key finding: **only strong learned representations transfer** to the matched,
distribution-shifted test set — hand-crafted morphology, SSL, and the matched confounds
do not (they land above the 0.5 constant). No local validation predicted transfer; the
public leaderboard was the only honest signal.

## Repository layout
- `solution.py` — official solver (CNN ensemble + frozen foundation features + grand).
- `src/` — earlier tested modules (feature extraction, confound analysis, transfer proxy).
- `research/` — H100 experiment scripts (CNN fleets, frozen extraction, calibration) and
  rejected explorations (confound, SSL, simulated-shift proxy).
- `notes.md` — challenge facts + full experiment log.

## Reproduce
```
python solution.py                 # writes working/submission.csv
```
`CNN_SEEDS` / `CNN_EPOCHS` env vars trade runtime vs ensemble size.

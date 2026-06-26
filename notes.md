# Notes — Microscopy Actin Pairwise Organization Ranking

## Challenge facts
- Task: pairwise ranking of STED-FM grayscale microscopy tiles. Predict P(left tile has higher hidden actin-organization score).
- Metric: **Gap-Weighted Pair Log Loss**, lower is better. `min(100, 100 * sum(w*logloss)/sum(w))`, prob clipped to [1e-6, 1-1e-6].
- Constant 0.5 => 69.31. AI baseline => **74** (worse than constant => poorly calibrated). Goal: beat 74, target well below 69.31.
- Output: `submission.csv` with columns `id,prob_left_higher_organization`, 450 rows.

## Data (paths relative to `dataset/`; official runtime uses `dataset/public/`)
- train.csv: 900 pairs. cols: id, left_image_path, right_image_path, left_higher_organization(0/1), pair_weight.
- test.csv: 450 pairs. cols: id, left_image_path, right_image_path.
- sample_submission.csv: 450 rows.
- train/: 301 tiles (128x128 grayscale uint8). test/: 189 tiles.
- Target balance: 51.9% left-higher. pair_weight in [1.27, 2.73], derived from hidden score gap.
- Tiles reused: train tile ~6 pairs each (1-10), test tile ~4.8 each. NO train/test tile overlap.
- Images dark (mean 1.8-57.9 / 255), sparse bright filaments.

## Key insights
- Latent per-tile organization score; label = sign(score_left - score_right). => Bradley-Terry / RankNet Siamese: P(left>right)=sigmoid(s_left - s_right), weight by pair_weight.
- Per-tile empirical win-rate is strongly bimodal => stable per-tile latent score on seen tiles.
- BUT test tiles are unseen => must learn score(image) from morphology that GENERALIZES, not memorize.
- Pairs matched on brightness/texture/gradient/dark-fraction => those shortcuts weakened by design. Need morphology/topology.

## Validation protocol
- MUST be tile-disjoint (description warns vs random pair split).
- Protocol: repeated random TILE splits (~75/25). Train on pairs with BOTH endpoints in train-tiles; eval weighted-logloss on pairs with BOTH endpoints in val-tiles. Mirrors test (all-unseen tiles).
- Final model trains on ALL pairs.

## Compute
- Local: RTX 4050, 6.4 GB VRAM. Dev here (fast iteration).
- Kaggle T4 (16GB) via KAGGLE_API_TOKEN if bigger backbone needed.
- H100 (shared) reserved for large sweeps only — overkill for this tiny data.
- FINAL solution targets A10 (24GB, ~30 min), Kaggle-standard libs (torch, torchvision, timm, sklearn, skimage).

## Compliance reminders
- No metadata/id/path/row-order as signal. Must compare both tiles. No single-image collapse.
- Calibrated probabilities (not hard 0/1). Official solution must train from dataset/public/ inside runtime.
- Pretrained ImageNet backbones: allowed if reproducible; watch A10 network availability for weight download.

## Experiment log
- Hand-crafted morphology (118 feats) + linear BT: ~67.5 OOF.
- +Topology/shape/spatial feats (155 feats): linbt 64.4 OOF (98% cov), acc 0.62. <- robust core.
- Frozen ImageNet feats (resnet18): 73 (OOD, weak). DROPPED.
- CNN Siamese pairwise (resnet18 full finetune): overfits, ~69 (=constant). DROPPED.
- CNN regression to BT-z, from-scratch small net: 0.50 acc (fails). DROPPED.
- **CNN regression to BT-z, pretrained resnet18, drop=0.5 wd=3e-2: acc 0.656, raw 67.4, needs calibration (T~0.4).**
  Higher acc than hand-crafted + different errors => ensemble candidate.
- Models that work: linear BT on feature diff (best single), ExtraTrees-on-z, deep resnet18-regression.
- Key: deep needs pretrained ImageNet weights (network) + GPU. Hand-crafted is the bulletproof CPU core.

## Current best
- Submission v1 (hand-crafted ensemble): OOF ~64.4. VALID.
- **Submission v2 (hand-crafted linbt + deep resnet18-regression, ~50/50 calibrated blend): OOF ~61.9.** VALID.
  - linbt 64.28 (acc .628, T1.18) + deep 63.53 (acc .643, T1.12); equal-mean 61.89 ≈ weighted 61.87 (robust).
- solution.py: self-contained, AST-assembled from tested modules, reads dataset/public or dataset.
  - BUG fixed: ast.get_source_segment drops decorators -> predict_z lost @torch.no_grad(). grab() now includes decorator lines.
- Deliverables: solution.py, working/submission.csv, submission.csv (mirror), approach.md, README, src/, research/.
- **Canonical: solution.py self-reported OOF 62.43 (lin 63.55 + deep 65.70, blend w_lin=0.68), 11.3min, VALID.**
  - Isolated clean-dir run (only solution.py + dataset/public) reproduced OOF 62.82 from scratch -> no hardcoding/leakage.
  - solution.py uses leaner 12-split/2-seed OOF (vs dev 16-split) -> deep OOF noisier -> slightly conservative blend; still ~62.

## Deep model recipe (what finally generalized)
- Distill 900 weighted pairs -> per-tile BT latent z (LogReg +1/-1 design, C=0.5).
- Pretrained resnet18 (in_chans=1) REGRESSES z. drop=0.5, wd=3e-2, backbone lr=3e-4, OneCycle,
  on-GPU dihedral+affine aug, per-image standardize, Huber(beta .5), 8-view dihedral TTA, seed-ensemble.
- Calibrate via OOF temperature. NOT pairwise Siamese (overfit), NOT from-scratch (0.50 acc).

## Decisions
- Official solution = hand-crafted ensemble (robust, CPU, no-network, standard libs) +/- deep resnet18 if gain is real.
- Calibration is the main lever for the LOSS metric (acc ~0.62-0.66); keep predictions honest, not overconfident.
- Metric up-weights large-gap (easy) pairs => train with pair_weight, let confidence track the gap.

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
(append CV / public results here)

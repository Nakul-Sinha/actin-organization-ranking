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

## !!! REAL TEST SCORE 87.62 — WORSE THAN CONSTANT (69.31). Diagnosed + fixing. !!!
- Root cause (research/diagnose.py): my predictions rode the MATCHED gradient-magnitude confound
  (corr(prob, L-R gradmag)=+0.437). In train gradmag has AUC 0.604 (spurious residual); test pairs
  are matched on gradient => that residual vanishes/inverts => anti-correlated => 87.6.
- Also: train/test tile distribution shift (classifier AUC 0.62-0.67) incl. topology feats.
- Why OOF lied: held-out TRAIN tiles share the same spurious confound residual => OOF rewarded the shortcut.
- FIX (src/build_robust.py): (1) residualize features against confound basis; (2) post-hoc
  orthogonalize pair logits against pair confound DIFFERENCES (matched=>uninformative in test);
  (3) conservative calibration (OOF was ~25pts optimistic on loss).
  - Confound-ORTHOGONAL OOF: acc 0.613, loss 67.25 (genuine signal survives). Test-pred confound corr ~0 (all).
  - Submission std 0.076, clip [0.34,0.66] (vs failed 0.16, confound corr 0.437).
- Testing if a confound-INVARIANT deep model (blur+gamma aug) adds orthogonal signal (research/deep_robust.py).
- LESSON: with matched confounds, validate confound-ORTHOGONALITY of predictions, not just OOF. Be conservative.

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

## !!! SUBMISSION 2 (confound-orthogonal) scored 73 on real test — still > constant 69.31 !!!
- 73 = worse than constant. Confound-orthogonal morphology features ANTI-CORRELATE under train->test shift.
- Built a SIMULATED-SHIFT PROXY (research/sim_shift2.py): train-vs-test direction splits train tiles into
  train-like/test-like; train on one, eval on other. Reproduces the failure (method A shift-acc 0.438 ~ 73).
- FIX (src/build_transfer.py, submission 3): rank-normalize feats within each set (transductive domain adapt)
  + drop 50% most train/test-shifted feats + confound-orth + calibrate on the proxy.
  - Proxy shift-acc 0.553, shift-loss 68.40 (vs constant 69.31). std 0.053, confound-corr ~0.004.
- solution.py rebuilt (transfer-robust), reproduced isolated (proxy-loss 68.40, 74s).
- LESSON: when test is distribution-shifted, validate on a SIMULATED shift, not train-OOF. Rank-norm + drop
  shifted feats recovers transfer. Hand-crafted ceiling ~0.55 shift-acc; SSL/in-distribution feats = next lever.

## FINAL (robust) deliverable
- Approach: confound-ORTHOGONAL linear Bradley-Terry. CPU-only, deterministic, no network/pretrained.
  - 155 morphology feats -> residualize vs confound basis -> linear BT on feat diffs ->
    orthogonalize test logits vs pair confound diffs -> conservative calib (OOF best-T x2.0, clip [0.32,0.68]).
  - Confound-orthogonal OOF: acc ~0.61, loss ~67.3 (genuine signal). Test-pred max|confound-corr| ~0.02.
  - solution.py self-contained (src/build_solution.py assembles from features.py + build_robust.py).
  - Isolated clean-dir run reproduced it end-to-end (acc 0.619, loss 66.4, valid sub).
- Deep confound-invariant CNN (blur/gamma aug) tested: acc 0.574 < linear 0.613 => deep does NOT help.
  Confirms bottleneck was confound reliance (method), not compute. H100 not needed.
- Old deep+linear ensemble (87.6) modules moved to research/.

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

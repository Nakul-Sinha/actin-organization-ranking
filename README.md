# Actin Organization Ranking

## The problem

I get matched pairs of grayscale STED-FM microscopy tiles and have to predict the
probability that the left tile has the higher hidden actin organization score.

The pairs are constructed adversarially. Left and right are deliberately close in
visible intensity, texture, gradient strength and dark pixel fraction, while
their hidden annotation topology differs by a real margin. So brightness and
coverage tell me nothing, and the answer has to come from filament morphology:
boundary structure, punctate organization, compact assemblies, protrusive
patterns.

Scoring is gap-weighted pair log loss, lower is better. A constant 0.5 scores
69.31 and the baseline is 74.

## What I did

An equal weight ensemble of per-tile organization scorers, each a linear
Bradley-Terry model on top of a strong representation. Half of them are Siamese
RankNet CNNs, ImageNet initialized ResNet-18, 34 and 50, trained on the pairwise
labels with heavy augmentation, seed ensembled and with TTA. The other half are
frozen foundation model features, DINOv2 in four sizes and ConvNeXt large and
XXL, each feeding a linear Bradley-Terry head. I normalize each scorer's test
pair logits to unit standard deviation, sum them, and calibrate.

Result: 61.5, first place.

The finding I would keep from this one is that only strong learned
representations transferred. Hand crafted morphology features, self supervised
variants, and the matched confounds all landed worse than the constant 0.5
baseline. Nothing in my local validation predicted which would transfer, and the
public leaderboard was the only honest signal I had. Adding the largest DINOv2
and ConvNeXt-XXL is what moved the score from 65.6 to 61.5.

## Layout

`python solution.py` writes the submission. `src/` has earlier modules, and
`research/` holds the experiment fleets plus the explorations I rejected.
`TECHNICAL.md` and `notes.md` carry the full log. Datasets are not committed.

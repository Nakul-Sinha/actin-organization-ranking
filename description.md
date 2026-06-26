Microscopy Actin Pairwise Organization Ranking
Overview
Microscopy review of subcellular F-actin organization often depends on relative morphology judgments: two image tiles can have similar total filament signal while one has more complex organization, sharper boundaries, or stronger interaction between morphology categories.

Your task is to compare matched pairs of grayscale STED-FM microscopy tiles. For each pair, predict the probability that the left tile has the higher hidden-mask-derived actin organization score. Pairs were constructed so the left and right tiles are close in visible intensity, texture, gradient strength, and dark-pixel fraction, while their hidden annotation topology differs by a meaningful margin. This makes the task depend on local filament morphology rather than simple brightness or coverage.

This is a computer vision pairwise ranking challenge on biological microscopy images. Good solutions should compare boundary structure, punctate organization, compact assemblies, protrusive patterns, and local texture differences between the two public PNG tiles.

Dataset
File descriptions
train.csv -- 900 labeled tile-pair rows containing left/right image paths, a binary target, and a public training pair weight.
test.csv -- 450 unlabeled tile-pair rows containing left/right image paths only.
train/ -- 301 public 128 x 128 grayscale PNG microscopy tiles referenced by train.csv.
test/ -- 189 public 128 x 128 grayscale PNG microscopy tiles referenced by test.csv.
sample_submission.csv -- Template showing the required submission format with random valid probabilities.
Column descriptions
id (string) -- Unique 12-character hexadecimal identifier for each tile pair.
left_image_path (string) -- Relative path from dataset/public/ to the left PNG image.
right_image_path (string) -- Relative path from dataset/public/ to the right PNG image.
left_higher_organization (integer) -- Training target, present only in train.csv. The value is 1 when the left tile has the higher hidden organization score and 0 otherwise.
pair_weight (float) -- Training weight derived from the hidden organization-score gap for the pair. Larger values indicate a larger private topology gap.
Evaluation
Submissions are scored using Gap-Weighted Pair Log Loss. Lower scores are better. The metric uses hidden pair weights, so confidently separating pairs with larger hidden topology gaps matters more.


def evaluate(y_true, submitted_probability, pair_weight):

    p = clip(submitted_probability, 1e-6, 1 - 1e-6)

    loss = -(y_true  *log(p) + (1 - y_true)*  log(1 - p))

    weighted_log_loss = sum(pair_weight * loss) / sum(pair_weight)

    return min(100, 100 * weighted_log_loss)

A constant 0.5 submission scores about 69.31. Scores below that require learning useful visual pair comparisons from the public images. Submitted probabilities must be finite values in the closed interval [0, 1].

Submission
Submit a CSV file with your predicted probability for every row in test.csv.

id (string) -- The 12-character pair identifier from test.csv.
prob_left_higher_organization (float) -- Predicted probability that the left image has higher actin organization than the right image.
Example:


id,prob_left_higher_organization

002976bc9833,0.885744

00ad342eed17,0.676955

012f78f5b6f6,0.161042

Requirements
The file must contain exactly 450 rows plus a header, one for each row in test.csv.
Every id from test.csv must be present exactly once.
The only columns must be id,prob_left_higher_organization.
All submitted probabilities must be finite values between 0 and 1 inclusive.
Submitted probabilities must be generated from files under ./dataset/public/ only.
File format: .csv only, with exact column names id,prob_left_higher_organization.
What Not To Do
Do not treat pair IDs, row order, or path strings metadata as the predictive signal.
Do not collapse the task to a single-image score without comparing the left and right tiles.
Do not rely only on brightness, dark-pixel area, or total filament amount; pairs were matched to weaken those shortcuts.
Do not trust a random pair-row validation split by itself, because related tiles can appear in multiple public training pairs.
Do not submit hard 01 labels when calibrated probabilities are available; the metric rewards probability quality.
 


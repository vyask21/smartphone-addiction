# Predicting Smartphone Addiction

Kaggle Playground Series, Season 6 Episode 8. Binary classification on synthetic tabular
data, 691,369 training rows and 12 features, every one of them carrying between 4% and 20%
missing values.

- Competition: https://www.kaggle.com/competitions/playground-series-s6e8
- Metric: area under the ROC curve
- Deadline: 2026-08-31
- Final placement: 78 of 3,532, private AUC 0.97103

## Result

Finished 78th of 3,532 teams at 0.97103 AUC, inside the top 2.2 percent.

The scored submission is a 65-member stack: gradient-boosted trees, two tabular neural
networks, and published out-of-fold libraries, all trained on one fixed fold split and
combined by a logistic regression fitted inside that same split. It was also the best
submission this repo produced, so the result did not turn on which one was chosen.

## Approach

### The problem

Given twelve columns describing a person's phone use, predict the probability that they
are labelled addicted. The metric is AUC, which measures only how well the predictions
rank people against each other. The absolute probability values do not matter, so no
calibration step is needed and any transform that preserves ordering scores identically.

### Validation

Five-fold stratified cross-validation, seed 42, fixed on the first day and never changed.
Stratified means every fold holds the same 71 percent positive rate as the full dataset.
The fold assignment is rebuilt and checksummed inside every notebook rather than saved and
reloaded, so a split that silently changed would fail loudly instead of producing a clean
looking wrong answer.

Every model saves its out-of-fold predictions: for each row, the prediction made by a copy
of the model that never saw that row while training. Because the fold assignment is
identical everywhere, a prediction vector saved in week one can be combined with one from
week four and trusted.

The gap between the cross-validation score and the leaderboard score was used as an
instrument. Across sixteen submissions it held between +0.00126 and +0.00132, so
cross-validation could be trusted to rank a change without spending a submission on it.
When that gap moved, it was a signal: it fell to +0.00109 once outside models entered the
stack, which is what models with over-optimistic out-of-fold values look like from
outside.

### Features

Two things worked.

Target encoding was the largest single gain in the competition, worth +0.0033 AUC.
Each of the twelve columns is replaced by the average target value across rows sharing
that column value, smoothed toward the overall average so that rare values do not get
extreme estimates. The hazard is leakage: if a row's own label feeds its own encoded
value, the model memorises instead of learning. That is avoided by fitting the encoder
inside a second, inner five-fold split, so a row's encoded value always comes from a fit
that excluded it. Each column also gets a frequency encoding, which is simply how often
its value occurs.

A composition ratio block was worth +0.0004 to +0.0009. Thirteen columns of shares and
rates: what fraction of screen time is social media, how many hours remain after sleep and
work, notifications per app open, and similar. These state relationships between columns
that a tree would otherwise have to approximate with many separate splits.

Everything else tried returned nothing measurable: missingness indicators, threshold
rules, pairwise crossed encodings, and quantile bins of derived ratios.

### Models

Four families, all trained on the same folds and the same feature frame so their
predictions are directly comparable.

- Gradient-boosted decision trees: LightGBM, XGBoost and CatBoost. Each builds
  hundreds of shallow trees in sequence, with every tree correcting the errors left by the
  ones before it. These are the strongest single models on tabular data of this shape.
- RealMLP, a neural network designed for tabular data. Numeric columns pass through a
  periodic embedding, which spreads one number across many sine and cosine features so the
  network can represent sharp changes that a plain layer would smooth over. Trained with a
  flat-then-cosine learning rate schedule, label smoothing, and an exponential moving
  average of the weights.
- TabM, a parameter-efficient ensemble. One network carries several sets of lightweight
  per-branch weights, so a single training run yields several diverse predictions at close
  to the cost of one.

Determinism flags are set on the boosted trees and results verified bit-identical on
re-run, so a difference between two experiments is the change being tested rather than
run-to-run noise.

### Combining the models

The final prediction is a stack. Each member's out-of-fold predictions are converted to
logits, clipped to bound extreme values, and fed as input columns to a logistic regression
that learns how much weight each member has earned. That combiner is fitted inside the
same fold loop as the members, so the combining step is validated out-of-fold as well as
the models are. Members are then pruned to the top 65 by average absolute coefficient.

Four constrained alternatives were tested against it, including hill climbing and
non-negative least squares. All transferred slightly worse to the leaderboard despite
using no negative weights.

One measurement shaped the whole search. How correlated two models are does not predict
whether averaging them helps. Measured across all 66 pairs of models trained here, the
relationship between the two came out at +0.14, which is close to none at all. The common
advice to seek out diverse models did not hold on this data.

### Verifying an outside model's fold split

Public out-of-fold prediction libraries are permitted in this competition. This repo used
none until row 144, then reversed that, and the reversal is recorded in the ledger. What
makes rows 145 onward defensible is
[`writeup/verify_public_oof.py`](writeup/verify_public_oof.py).

The hazard is subtle. A prediction vector built on a different fold split is still
out-of-fold for each row, so it scores normally and looks clean. But the model behind its
value on your training rows was trained on rows that sit inside your validation fold. Use
it as an input to the combiner and your cross-validation score rises while the leaderboard
does not move.

The test: recompute each fold's AUC on your own fold assignment, and admit the model only
if those numbers reproduce the author's own published per-fold scores in order. A fold's
AUC is a property of exactly which rows sit in that fold, so agreement is proof and
disagreement is proof of the opposite.

Ten models passed. Two were rejected, including the highest scoring candidate seen in the
competition at 0.968691, whose fold AUCs read 0.96810 on this split where its author had
published 0.96593. A vector that scores higher on your folds than on its own is a mixture
of models rather than a held-out prediction.

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/kaggle competitions download playground-series-s6e8 -p data/raw
```

The Kaggle CLI does not extract the archive, so unzip `data/raw/*.zip` in place. Then run
[`notebooks/01_baseline_lgbm.ipynb`](notebooks/01_baseline_lgbm.ipynb) top to bottom. It
finds the data by searching upward for `data/raw/`, so it also runs unchanged in a Kaggle
notebook against `/kaggle/input/`.

Accepting the competition rules on the website is required first. The download returns 403
until you do, and nothing in the API can do it for you.

Every run, including the failures, is a row in [`experiments.csv`](experiments.csv): 170
of them, with cross-validation score, fold standard deviation, both leaderboard scores
where they came back, and the single variable each run changed.

## License

MIT

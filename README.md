# Predicting Smartphone Addiction

Kaggle Playground Series, Season 6 Episode 8. Binary classification on synthetic tabular
data, 691,369 training rows and 12 features, every one of them carrying between 4% and 20%
missing values.

- Competition: https://www.kaggle.com/competitions/playground-series-s6e8
- Metric: area under the ROC curve
- Deadline: 2026-08-31
- Final placement:

## Result

One paragraph, written at the end: what was built, what it scored, where it placed.

## Approach

Five-fold `StratifiedKFold`, seed 42, fixed on day one and never changed. The fold vector
is rebuilt and checksummed in every notebook rather than loaded, because a silently
different split produces a clean looking wrong answer. That checksum is the least clever
thing here and half the analysis would be unmeasurable without it: it is what lets an
out-of-fold vector saved in week one be stacked against one from week four.

The offset between CV and the leaderboard was the instrument, not the CV number itself.
Across sixteen own-model submissions it held between +0.001256 and +0.001322, so CV could
be trusted to rank a change without spending a submission. When the offset moved, that was
information: it fell to +0.001093 as public members entered the stack, which is what
members with optimistic out-of-fold values look like from the outside.

Three things moved the score. Target encoding of all twelve columns, +0.003312, fitted
inside the fold loop with an inner five-fold split so no row sees its own label, and the
largest single gain in the competition. A composition ratio block, +0.000356 to +0.000906.
RealMLP, +0.000072 to the stack, an architecture class the stack did not contain. Nine
membership gates ran in total and RealMLP was the only one after row 122 to clear five
figures. Every gate that offered a better version of something already present returned
noise.

The model is a logistic regression on the clipped logits of member out-of-fold vectors,
fitted inside the fold loop so the combiner is out-of-fold as well as the members, then
pruned to the top 65 members by mean absolute coefficient.

### Verifying somebody else's fold split

Public out-of-fold libraries are permitted here. This repo declined to use them until row
144, then reversed that deliberately, and the reversal is recorded in the ledger rather
than quietly made. What makes rows 145 onward defensible is
[`writeup/verify_public_oof.py`](writeup/verify_public_oof.py).

A vector built on a different fold partition is still out-of-fold per row, so it scores
normally and looks clean, but the model behind its value on your training rows trained on
rows sitting inside your validation fold. Use it as a combiner feature and your CV rises
while the leaderboard does not. The test is to recompute per-fold AUC on your own fold
assignment and admit the member only if it reproduces the author's printed per-fold scores
in order, because a fold AUC is a property of exactly which rows sit in the fold.

Ten members passed. Two were rejected, including the highest scoring candidate in the
competition at 0.968691, whose fold AUCs read 0.96810 on this partition where its author
had printed 0.96593. A vector scoring higher on your folds than on its own is a mixture
rather than a held-out prediction.

## What was submitted, and the finding behind it

A family mean of four public blends scores exactly what submitting one of them unchanged
scores. Ten constructions were tested against the plateau, including median, geometric
mean, quality weighting and trimming, and every one returned the identical number. The
public split cannot resolve 0.00001 against its own standard error of 0.00061, and 109
teams sit inside two hundred-thousandths of each other with predictions correlated above
0.9999.

That is why the two selected submissions are deliberately different in kind rather than two
versions of one bet.

The first is an equal-weight rank average of four public blends, with near-duplicates
collapsed so no author votes twice. Nothing in it is fitted, on out-of-fold data or on the
leaderboard. It contains no model built in this repo, and the ledger row that records it
says so in its first sentence.

The second is this repo's own stack: 175 members, own combiner, own folds, plus one
external blend whose out-of-fold vector could be verified. 0.970127 out of fold at a
realised offset of +0.001013, in line with every honest submission before it.

If that plateau reshuffles on the private split, the second one is the reason there is
still a result.

## What is in this repo

- [`experiments.csv`](experiments.csv), 169 rows. Every run including the failures, with
  CV, fold standard deviation, public score where one came back, and the single variable
  each run changed. This is the deliverable; the models are disposable.
- [`NOTES.md`](NOTES.md), the running argument behind those numbers. Why each thing was
  tried and what it meant, written before the outcome was known, dead ends included. It
  also carries the rules this was run under and a section on how to read the ledger.
- [`DATA.md`](DATA.md), a column-level profile generated from the data files themselves
  rather than from the competition description.
- `notebooks/`, one per experiment. `12_public_writeup.ipynb` is the writeup.

There is no `src/` and no `conf/`. That departure from the workspace template is argued out
in [`NOTES.md`](NOTES.md), and every number in the ledger comes from a Save and Run All on
a clean kernel.

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

## License

MIT

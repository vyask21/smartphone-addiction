# Predicting Smartphone Addiction

Kaggle Playground Series, Season 6 Episode 8. Binary classification on synthetic
tabular data, 691,369 training rows and 12 features, every one of them carrying
between 4% and 20% missing values.

- Competition: https://www.kaggle.com/competitions/playground-series-s6e8
- Metric: area under the ROC curve
- Deadline: 2026-08-31
- Final placement:

## Result

One paragraph, written at the end: what was built, what it scored, where it placed.

## Approach

### Validation

Five-fold `StratifiedKFold`, `shuffle=True`, `random_state=42`, over `train.csv` in its
original row order, fixed on day one and never changed. The target is 71 percent positive
and the metric is AUC, so stratification matters and nothing else about the data suggests
groups or time ordering. The fold vector is rebuilt and checksummed in every notebook
rather than loaded, because a silently different split produces a clean looking wrong
answer.

That scheme turned out to be the one the public libraries for this competition also
converged on, which mattered later.

The offset between CV and the leaderboard was the instrument, not the CV number itself. Across
sixteen own-model submissions it held between +0.001256 and +0.001322, with the direction
agreeing on eleven of thirteen consecutive pairs. A stable offset means CV can be trusted
to rank changes without spending a submission. When the offset moved, that was information:
it fell to +0.001093 as public members entered the stack, which is what members whose
out-of-fold values are optimistic look like from the outside.

### What moved the score

Three things, in order of size.

Target encoding, +0.003312. Smoothed target and frequency encoding of all twelve
columns, fitted inside the fold loop with an inner five-fold split so no row sees its own
label. This was the largest single gain in the competition and most of the feature work
that followed it returned nothing.

A composition ratio block, +0.000356 to +0.000906. Thirteen columns of shares, slacks
and per-hour rates. It paid on every learner family tried and paid more with the encoder
present than without, which contradicted this repo's own stated explanation of why target
encoding worked and forced that explanation to be rewritten.

RealMLP, +0.000072 to the stack. A neural architecture the stack did not contain,
implemented from the paper. Nine membership gates ran in total and this was the only one
after row 122 to clear five figures. Every gate that offered a better version of something
already present returned noise.

### The model

A logistic regression on the clipped logits of member out-of-fold vectors, fitted inside
the fold loop so the combiner is out-of-fold as well as the members. Members are pruned to
the top 65 by mean absolute coefficient. Four constrained alternatives were tested against
it, including hill climbing and non-negative least squares, and all transferred slightly
worse on the leaderboard despite using no negative weights.

### The decision that changed the competition

The repo ran own models only until row 144, reached 0.968944 out of fold and 0.97021
public, and closed modelling on eleven gates. Reading the public frontier then showed that
the scores above this repo were level two models over pooled out-of-fold libraries rather
than a representation anyone had found.

That decision was reversed deliberately and the reversal is recorded in the ledger. What
makes rows 145 onward defensible is [`writeup/verify_public_oof.py`](writeup/verify_public_oof.py).

The problem it solves. An out-of-fold vector built on a different fold partition is
still out-of-fold per row, so it scores normally and looks clean. But the model behind its
value on our training rows trained on rows sitting inside our validation fold. Using it as
a combiner feature leaks validation information into the fit, raises CV, and does not raise
the leaderboard.

The test. Authors print their own per-fold AUCs. Recompute per-fold AUC on the same
vector using our fold assignment and admit the member only if the two agree in order,
because a fold AUC is a property of exactly which rows sit in the fold. Ten members passed.
Two were rejected, including the highest AUC candidate seen in the competition at 0.968691,
whose fold AUCs read 0.96810 on our partition where its author had printed 0.96593. A
vector scoring higher on our folds than on its own is the signature of a different split.

That rejection was independently confirmed: one library author had retrained the same
architecture himself because the published version used ten folds and could not be stacked
against a five-fold library without leaking.

Two indirect tests were built to admit members without printed fold numbers, and both
failed. One of them is a published method, and calibrating it against two known foreign
vectors showed it to be anti-predictive rather than merely weak.

### What the final submissions are

Two submissions were selected, deliberately different in kind.

`family_mean_v3.csv` is an equal-weight rank average of four public plateau blends, with
near-duplicates collapsed so no author votes twice. Nothing is fitted in it, on out-of-fold
data or on the leaderboard. It contains no model built in this repo. It is the higher
public score and the smaller part of the work, and the ledger says so in the row that
records it.

`stack_greedy_extblend.csv` is this repo's own stack: 175 members, our combiner, our folds,
plus one external blend whose out-of-fold vector could be verified. Out of fold 0.970127
with a realised offset of +0.001013, in line with every honest submission before it.

The pair is a hedge. The plateau is the most crowded position on the board, with 76 teams
holding the same score and predictions correlated above 0.9999. If it reshuffles on the
private split, the second submission is the one built on verified folds and uncorrelated
with that block.

### The finding worth taking to the next competition

A family mean of four public blends scores exactly what submitting one of them unchanged
scores. Ten constructions were tested against the plateau, including median, geometric
mean, quality weighting, trimming and three blends with our own work, and every one
returned the identical number. At the top of this board the public split cannot resolve
0.00001 against its own standard error of 0.00061, and 109 teams sit inside two
hundred-thousandths of each other.

The reasoning as it happened, including the ideas that were rejected and why, is in
[`NOTES.md`](NOTES.md). That file is the honest version and it is written before the
outcome is known.

## Layout

This competition is notebook-first. There is no `src/` and no `conf/`, which is a
deliberate departure from the workspace template, argued out in [`NOTES.md`](NOTES.md).
Every number in the ledger comes from a Save & Run All on a clean kernel.

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/kaggle competitions download playground-series-s6e8 -p data/raw
```

The Kaggle CLI does not extract the archive, so unzip `data/raw/*.zip` in place. Then
run [`notebooks/01_baseline_lgbm.ipynb`](notebooks/01_baseline_lgbm.ipynb) top to bottom.
It finds the data by searching upward for `data/raw/`, so it also runs unchanged in a
Kaggle notebook against `/kaggle/input/`.

Accepting the competition rules on the website is required first. The download returns
403 until you do, and nothing in the API can do it for you.

## Experiment log

Every run, including the ones that failed, is a row in
[`experiments.csv`](experiments.csv): 169 of them, with CV, fold standard deviation,
public leaderboard score where one came back, and the single variable each run
changed. What follows is how to read it.


Rows 6 onward have LightGBM's determinism flags on and are verified bit-identical on
re-run. Rows 1 to 5 predate that and carry about 1e-4 of run-to-run noise, so do not
read a difference at the fourth decimal between them. See `NOTES.md`.

Row 24's CV is a stacker fitted and scored on the same out-of-fold matrix, so it reads
optimistically. Row 25 is the same stack with the combiner fit inside the fold loop,
which is the number to read: 0.967650, +0.000867 over row 17 on all five folds. The
optimism in row 24 turned out to be 1.5e-05. Every stack row from 25 onward uses the
fold-loop protocol, and every other row is a plain five-fold CV.

Rows 55 to 57 are the one idea that reopened the feature set. `31` refuted the
"no interactions to discover" claim that had feature engineering closed since
2026-08-11: nine of the 66 column pairs carry real non-additive signal against a
control centred on zero. Encoding those pairs is still worth nothing, because a
depth-6 tree already reaches every 2-way region, and encoding all 66 is worth
-0.000509 through plain dilution. Real effect, no value, which is row 19 again.

Rows 26 to 54 are three reopenings and three null sweeps. CatBoost (26, 28 to 31) and
XGBoost (38, 40 to 43) each came back as a family after a single seed looked
promising, and the neural model (33) came back on the encoded features. All three
paid. `num_leaves` (35 to 37), the encoder's smoothing constant (46 to 49), its inner
split count (50 to 54) and XGBoost's `max_depth` (62 to 65) were hyperparameters of
components already fitted to that representation, and all four returned nothing. The
rule those seven points support is in `NOTES.md`. It was written down after the fourth
and has correctly called every one since, including `max_depth`, where the notebook
header argued at length that the rule did not apply.

Rows 50 to 54 also close a gap rather than only adding a null: after them there is no
constant in this pipeline that has never been varied.

Row 94 was the best row on both axes when it ran, CV 0.968713 and public LB 0.97003, a
41 member stack. Read rows 95 to 100 beside it, because the interesting part is that they
disagree with the previous gate. In row 77 no single candidate cleared the bar and the set
beat the sum of its parts by 2.9x. Here five of six clear it alone and the set is worth only
0.39x the sum, because four of the six carry the same new information and compete for the
same weight. Whether adding members together helps more or less than adding them one at a
time depends on whether they are diverse in the same direction, and it cannot be assumed
either way.

Row 45 removes five members from row 44 and costs four millionths, which is a result about
what the stack was already ignoring rather than a better model.

Rows 101 to 142 are the saturation curve, and they are the most useful stretch of this
table to read as a sequence rather than row by row. Nine membership gates were run in
total. The last six returned +0.000112, +0.000021, +0.000005, +0.000083, +0.000023 and
+0.000000, falling by roughly a factor of five each step. Exactly one broke the pattern,
row 137, and the thing that broke it was RealMLP, an architecture class the stack had
never contained. A better version of something already in the stack never did.

Row 140 is the best row here on both axes, CV 0.968944 and public LB 0.97021. It is a
25 member prune of a 53 member stack, and the prune is worth +0.000013 over keeping all
53, which is a statement about how much of the stack was already dead weight.

Rows 141 and 142 are the last gate and they close the board. RealMLP with the target
encoder switched off scores 0.952357, which is 0.015536 behind its encoded twin at 0 of
5 folds, and it then contributes exactly +0.000000 to the stack and is pruned out of the
top 25 entirely. The finding is worth more than the null. Removing the encoder costs a
boosted tree between 0.0033 and 0.0059 and costs this network 0.0155, so the encoder's
value is a property of the architecture rather than of the representation. A tree can
split a raw column directly and recover an ordering of its levels for itself. A network
has no such move, and the periodic embedding is a smooth basis over the value axis
rather than a lookup over levels.

The fold standard deviation is the number that matters when reading this table. A
change smaller than it is noise until it survives a seed sweep.


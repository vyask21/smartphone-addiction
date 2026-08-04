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

What the validation scheme was and why, the features that mattered, the model, and
the one or two decisions that made the difference. Written at the end.

The reasoning as it happened, including the ideas that were rejected and why, is in
[`NOTES.md`](NOTES.md). That file is the honest version and it is written before the
outcome is known.

## Layout

This competition is notebook-first. There is no `src/` and no `conf/`, which is a
deliberate departure from the workspace template, argued out in [`CLAUDE.md`](CLAUDE.md).
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

Every run, including the ones that failed, is in [`experiments.csv`](experiments.csv).

| id | name | CV AUC | fold sd | public LB |
|---|---|---|---|---|
| 1 | lgbm_default_anchor | 0.954947 | 0.000645 | |

The fold standard deviation is the number that matters when reading this table. A
change smaller than it is noise until it survives a seed sweep.

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
| 1 | lgbm_default_anchor | 0.954947 | 0.000645 | 0.955940 |
| 2 | lgbm_trees100 | 0.954947 | 0.000645 | reproducibility re-run of 1 |
| 3 | lgbm_trees300 | 0.960605 | 0.000688 | |
| 4 | lgbm_trees1000 | 0.962141 | 0.000859 | 0.964350 |
| 5 | lgbm_trees2000 | 0.961832 | 0.000952 | |
| 6 | lgbm_lr01 | 0.962198 | 0.000816 | |
| 7 | lgbm_lr005 | 0.963210 | 0.000591 | |
| 8 | lgbm_lr003 | 0.963275 | 0.000549 | |

Rows 6 onward have LightGBM's determinism flags on and are verified bit-identical on
re-run. Rows 1 to 5 predate that and carry about 1e-4 of run-to-run noise, so do not
read a difference at the fourth decimal between them. See `NOTES.md`.

The fold standard deviation is the number that matters when reading this table. A
change smaller than it is noise until it survives a seed sweep.

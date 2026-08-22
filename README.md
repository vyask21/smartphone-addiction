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
| 8 | lgbm_lr003 | 0.963275 | 0.000549 | 0.964780 |
| 9 | lgbm_bag08_seed42 | 0.963471 | 0.000591 | |
| 10 | lgbm_bag08_seed2024 | 0.963234 | 0.000899 | |
| 11 | lgbm_bag08_seed7 | 0.963445 | 0.000478 | |
| 12 | lgbm_bag08_seedblend3 | 0.963821 | 0.000560 | 0.965090 |
| 13 | lgbm_bag08_seed2025 | 0.963337 | 0.000731 | |
| 14 | lgbm_bag08_seed13 | 0.963483 | 0.000552 | |
| 15 | lgbm_bag08_seedblend5 | 0.963880 | 0.000555 | 0.965080 |
| 16 | neural_mlp_kaggle | 0.939169 | 0.000759 | not submitted |
| 17 | lgbm_bag08_seed42_te | 0.966782 | 0.000453 | 0.96825 |
| 18 | lgbm_bag08_seed42_te_imp | 0.966805 | 0.000469 | null result, not submitted |
| 19 | lgbm_bag08_seed42_te_lat | 0.966651 | 0.000515 | negative, not submitted |
| 20 | lgbm_bag08_seed2024_te | 0.966771 | 0.000446 | |
| 21 | lgbm_bag08_seed7_te | 0.966729 | 0.000433 | |
| 22 | lgbm_bag08_seed2025_te | 0.966743 | 0.000427 | |
| 23 | lgbm_bag08_seed13_te | 0.966789 | 0.000427 | |
| 24 | stack_logit_18 | 0.967665 | 0.000432 | 0.96897 |
| 25 | stack_logit_18_oof | 0.967650 | 0.000437 | not submitted |
| 26 | catboost_te | 0.966915 | 0.000435 | |
| 27 | stack_logit_19_oof | 0.967750 | 0.000431 | not submitted |
| 28 | catboost_te_seed2024 | 0.966928 | 0.000468 | |
| 29 | catboost_te_seed7 | 0.966920 | 0.000424 | |
| 30 | catboost_te_seed2025 | 0.966916 | 0.000431 | |
| 31 | catboost_te_seed13 | 0.966922 | 0.000442 | |
| 32 | stack_logit_23_oof | 0.967764 | 0.000434 | 0.96902 |
| 33 | neural_te | 0.965373 | 0.000405 | in the stack, not submitted alone |
| 34 | stack_logit_24_oof | 0.967807 | 0.000432 | not submitted |
| 35 | te_leaves15 | 0.966016 | 0.000799 | |
| 36 | te_leaves63 | 0.966758 | 0.000440 | |
| 37 | te_leaves127 | 0.966596 | 0.000454 | |
| 38 | xgb_te | 0.967099 | 0.000414 | |
| 39 | stack_logit_25_oof | 0.967873 | 0.000424 | not submitted |
| 40 | xgb_te_seed2024 | 0.967148 | 0.000432 | |
| 41 | xgb_te_seed7 | 0.967132 | 0.000455 | |
| 42 | xgb_te_seed2025 | 0.967099 | 0.000421 | |
| 43 | xgb_te_seed13 | 0.967111 | 0.000446 | |
| 44 | stack_logit_29_oof | 0.967925 | 0.000428 | 0.96924 |
| 45 | stack_logit_24_pruned | 0.967929 | 0.000429 | discarded, not submitted |
| 46 | xgb_smooth1 | 0.967122 | 0.000440 | |
| 47 | xgb_smooth5 | 0.967116 | 0.000443 | |
| 48 | xgb_smooth25 | 0.967037 | 0.000418 | |
| 49 | xgb_smooth100 | 0.966880 | 0.000426 | |
| 50 | xgb_inner2 | 0.966937 | 0.000435 | |
| 51 | xgb_inner3 | 0.967016 | 0.000428 | |
| 52 | xgb_inner5 | 0.967099 | 0.000414 | |
| 53 | xgb_inner10 | 0.967107 | 0.000427 | |
| 54 | xgb_inner20 | 0.967096 | 0.000444 | |
| 55 | xgb_pair_base | 0.967099 | 0.000414 | |
| 56 | xgb_pair_top9 | 0.967176 | 0.000475 | |
| 57 | xgb_pair_all66 | 0.966589 | 0.000446 | |
| 58 | stack_29_refit | 0.967925 | 0.000428 | |
| 59 | stack_30_top9 | 0.967968 | 0.000436 | 0.96929 |
| 60 | stack_30_all66 | 0.967934 | 0.000427 | |
| 61 | stack_31_both | 0.967967 | 0.000436 | |
| 62 | xgb_depth4 | 0.966825 | 0.000481 | |
| 63 | xgb_depth6 | 0.967099 | 0.000414 | |
| 64 | xgb_depth8 | 0.966589 | 0.000400 | |
| 65 | xgb_depth10 | 0.966301 | 0.000465 | |
| 66 | cat_native_c1 | 0.958943 | 0.000504 | |
| 67 | cat_native_c2 | 0.961358 | 0.000536 | |
| 68 | lgb_raw | 0.963464 | 0.000586 | |
| 69 | xgb_raw | 0.964218 | 0.000497 | |
| 70 | cat_raw | 0.961420 | 0.000482 | |
| 71 | stack_30_row59_refit | 0.967968 | 0.000436 | |
| 72 | stack_31_cat_nat_c1 | 0.967987 | 0.000425 | |
| 73 | stack_31_cat_nat_c2 | 0.967970 | 0.000439 | |
| 74 | stack_31_lgb_raw | 0.967967 | 0.000436 | |
| 75 | stack_31_xgb_raw | 0.967990 | 0.000430 | |
| 76 | stack_31_cat_raw | 0.967975 | 0.000441 | |
| 77 | stack_35_all5 | 0.968110 | 0.000432 | 0.96941 |
| 78 | cat_raw_n2000 | 0.961420 | 0.000482 | |
| 79 | cat_raw_n3000 | 0.962736 | 0.000494 | |
| 80 | cat_raw_n4000 | 0.963343 | 0.000479 | |
| 81 | cat_raw_n6000 | 0.963854 | 0.000500 | |
| 82 | cat_raw_n8000 | 0.964003 | 0.000481 | |
| 83 | cat_raw_n10000 | 0.964042 | 0.000476 | |
| 84 | cat_raw_n12000 | 0.964020 | 0.000476 | |
| 85 | cat_te_n2000 | 0.966915 | 0.000435 | |
| 86 | cat_te_n3000 | 0.967101 | 0.000441 | |
| 87 | cat_te_n4000 | 0.967166 | 0.000445 | |
| 88 | cat_te_n6000 | 0.967158 | 0.000454 | |
| 89 | cat_te_n8000 | 0.967069 | 0.000463 | |
| 90 | lgb_raw_fe | 0.963821 | 0.000596 | |
| 91 | xgb_raw_fe | 0.964666 | 0.000486 | |
| 92 | cat_raw_fe | 0.962181 | 0.000546 | |
| 93 | xgb_te_fe | 0.968005 | 0.000418 | |
| 94 | stack_41_all6 | 0.968713 | 0.000407 | 0.97003 |
| 95 | stack_36_xgb_te_fe | 0.968534 | 0.000420 | |
| 96 | stack_36_xgb_raw_fe | 0.968562 | 0.000400 | |
| 97 | stack_36_cat_raw_fe | 0.968484 | 0.000428 | |
| 98 | stack_36_lgb_raw_fe | 0.968376 | 0.000419 | |
| 99 | stack_36_cat_te_n4000 | 0.968122 | 0.000435 | |
| 100 | stack_36_cat_raw_n10k | 0.968130 | 0.000432 | |
| 101 | xgb_bin256 | 0.968005 | 0.000418 | |
| 102 | xgb_bin512 | 0.968033 | 0.000422 | |
| 103 | xgb_bin1024 | 0.968040 | 0.000419 | |
| 104 | xgb_bin1536 | 0.968054 | 0.000409 | |
| 105 | xgb_bin3072 | 0.968014 | 0.000397 | |
| 106 | neural_fixed | 0.965402 | 0.000387 | |
| 107 | neural_lookup | 0.961181 | 0.000319 | |
| 108 | lgb_te_fe | 0.967212 | 0.001382 | |
| 109 | cat_te_fe | 0.968036 | 0.000432 | |
| 110 | logit_te_fe | 0.954249 | 0.000541 | |
| 111 | hgb_te_fe | 0.967885 | 0.000426 | |
| 112 | et_te_fe | 0.959538 | 0.000619 | |
| 113 | rf_te_fe | 0.962430 | 0.000590 | |
| 114 | stack_40_row94_nodup | 0.968712 | 0.000407 | |
| 115 | stack_41_cat_te_fe | 0.968741 | 0.000409 | |
| 116 | stack_41_hgb_te_fe | 0.968728 | 0.000408 | |
| 117 | stack_41_lgb_te_fe | 0.968706 | 0.000432 | |
| 118 | stack_41_rf_te_fe | 0.968714 | 0.000407 | |
| 119 | stack_41_neural_lookup | 0.968790 | 0.000409 | |
| 120 | stack_41_et_te_fe | 0.968716 | 0.000408 | |
| 121 | stack_41_logit_te_fe | 0.968714 | 0.000407 | |
| 122 | stack_47_all7 | 0.968824 | 0.000418 | 0.97013 |

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

Row 94 is the best row here on both axes, CV 0.968713 and public LB 0.97003, and it is
a 41 member stack. Read rows 95 to 100 beside it, because the interesting part is that they
disagree with the previous gate. In row 77 no single candidate cleared the bar and the set
beat the sum of its parts by 2.9x. Here five of six clear it alone and the set is worth only
0.39x the sum, because four of the six carry the same new information and compete for the
same weight. Whether adding members together helps more or less than adding them one at a
time depends on whether they are diverse in the same direction, and it cannot be assumed
either way.

Row 45 removes five members from row 44 and costs four millionths, which is a result about
what the stack was already ignoring rather than a better model.

The fold standard deviation is the number that matters when reading this table. A
change smaller than it is noise until it survives a seed sweep.

This table is generated from `experiments.csv` by `writeup/readme_table.py`. It is not
maintained by hand, because between 2026-08-11 and 2026-08-20 the hand-maintained
version fell 24 rows behind the file it was copying.

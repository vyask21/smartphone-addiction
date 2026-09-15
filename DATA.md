# Data profile

Column-level profile of the competition files, read in full.

## `train.csv` (44.9 MB)

691,369 rows, 14 columns

| column | dtype | nulls | unique | example |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 691,369 | 0 |
| `age` | float64 | 4.2% | 18 | 24 |
| `daily_screen_time_hours` | float64 | 13.9% | 1,389 | 5.97 |
| `social_media_hours` | float64 | 19.4% | 721 | 1.83 |
| `gaming_hours` | float64 | 18.3% | 401 | 1.59 |
| `work_study_hours` | float64 | 7.5% | 600 | 2.11 |
| `sleep_hours` | float64 | 6.4% | 451 | 7.46 |
| `notifications_per_day` | float64 | 9.8% | 231 | 122 |
| `app_opens_per_day` | float64 | 11.7% | 166 | 38 |
| `weekend_screen_time` | float64 | 16.2% | 1,437 | 8.63 |
| `gender` | str | 4.2% | 3 | Male |
| `stress_level` | str | 8.0% | 3 | Medium |
| `academic_work_impact` | str | 6.4% | 2 | No |
| `addicted_label` | int64 | 0.0% | 2 | 1 |

## `test.csv` (18.7 MB)

296,302 rows, 13 columns

| column | dtype | nulls | unique | example |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 296,302 | 691369 |
| `age` | float64 | 5.8% | 18 | 30 |
| `daily_screen_time_hours` | float64 | 11.1% | 1,349 | 9.34 |
| `social_media_hours` | float64 | 16.0% | 703 | 1.91 |
| `gaming_hours` | float64 | 20.1% | 401 | 0.77 |
| `work_study_hours` | float64 | 9.4% | 601 | 4.09 |
| `sleep_hours` | float64 | 7.6% | 451 | 7.15 |
| `notifications_per_day` | float64 | 11.5% | 231 | 153 |
| `app_opens_per_day` | float64 | 8.7% | 166 | 16 |
| `weekend_screen_time` | float64 | 17.1% | 1,404 | 10.68 |
| `gender` | str | 4.8% | 3 | Other |
| `stress_level` | str | 6.6% | 3 | Medium |
| `academic_work_impact` | str | 8.7% | 2 | Yes |

## `sample_submission.csv` (7.7 MB)

296,302 rows, 2 columns

| column | dtype | nulls | unique | example |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 296,302 | 691369 |
| `addicted_label` | float64 | 0.0% | 1 | 0.709424 |

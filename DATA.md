# Data profile

Generated from the files on disk. Trust this over the
competition description where they disagree.

## `sample_submission.csv` (7.7 MB)

200,000 rows read, 2 columns

| column | dtype | nulls | nunique | sample |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 200000 | 691369 |
| `addicted_label` | float64 | 0.0% | 1 | 0.7094243450313797 |

## `test.csv` (18.7 MB)

200,000 rows read, 13 columns

| column | dtype | nulls | nunique | sample |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 200000 | 691369 |
| `age` | float64 | 5.7% | 18 | 30.0 |
| `daily_screen_time_hours` | float64 | 11.0% | 1333 | 9.34 |
| `social_media_hours` | float64 | 16.1% | 697 | 1.91 |
| `gaming_hours` | float64 | 19.9% | 401 | 0.77 |
| `work_study_hours` | float64 | 9.4% | 601 | 4.09 |
| `sleep_hours` | float64 | 7.6% | 451 | 7.15 |
| `notifications_per_day` | float64 | 11.5% | 231 | 153.0 |
| `app_opens_per_day` | float64 | 8.7% | 166 | 16.0 |
| `weekend_screen_time` | float64 | 17.1% | 1380 | 10.68 |
| `gender` | str | 4.8% | 3 | Other |
| `stress_level` | str | 6.6% | 3 | Medium |
| `academic_work_impact` | str | 8.7% | 2 | Yes |

## `train.csv` (44.9 MB)

200,000 rows read, 14 columns

| column | dtype | nulls | nunique | sample |
|---|---|---|---|---|
| `id` | int64 | 0.0% | 200000 | 0 |
| `age` | float64 | 4.1% | 18 | 24.0 |
| `daily_screen_time_hours` | float64 | 13.9% | 1325 | 5.97 |
| `social_media_hours` | float64 | 19.3% | 695 | 1.83 |
| `gaming_hours` | float64 | 18.4% | 401 | 1.59 |
| `work_study_hours` | float64 | 7.4% | 600 | 2.11 |
| `sleep_hours` | float64 | 6.4% | 451 | 7.46 |
| `notifications_per_day` | float64 | 9.8% | 231 | 122.0 |
| `app_opens_per_day` | float64 | 11.7% | 166 | 38.0 |
| `weekend_screen_time` | float64 | 16.2% | 1382 | 8.63 |
| `gender` | str | 4.2% | 3 | Male |
| `stress_level` | str | 8.0% | 3 | Medium |
| `academic_work_impact` | str | 6.4% | 2 | No |
| `addicted_label` | int64 | 0.0% | 2 | 1 |

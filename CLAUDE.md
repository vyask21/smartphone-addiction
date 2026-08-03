# Competition-specific rules

The workspace rules in `../../CLAUDE.md` apply, with two deliberate overrides
recorded below. Everything else stands, especially the prime directive and the
leakage checks.

## Overrides to the workspace rules

This competition is notebook-first. The workspace Code section says everything is
reproducible from `python -m src.train --config conf/<name>.yaml` and that
notebooks are for looking rather than for producing submissions. Both are
suspended here. There is no `src/` and no `conf/` in this repo.

Why: the competition runs 4 weeks on a saturated leaderboard, so the experiment
count stays low enough that the module structure would be ceremony rather than
leverage. The cost is real and is accepted knowingly. A ledger row here carries no
config hash and no git SHA, so its provenance is the discipline below and nothing
else. If the experiment count passes roughly 15, or if two rows ever become
impossible to tell apart, that is the signal to move to the module layout in
`../../template/`.

## Non-negotiable here

- Read `NOTES.md` before proposing anything. The rejected-ideas list exists so the
  same dead end is not walked twice.
- **Every number in `experiments.csv` comes from a Save & Run All**, top to bottom
  on a clean kernel. A score read off a hand-run cell does not go in the ledger.
  This is the rule that makes the notebook layout survivable, because it is what
  rules out hidden state.
- Every run appends a row, failed ideas included.
- One variable per run. If two things change, that is two runs.
- Record the public LB score into the same row its CV score sits in, once it comes
  back. A row claiming a LB number it never received is worse than a blank.
- Never commit anything under `data/`. The rules forbid redistributing the
  competition data.

## Before saying an idea worked

- Is the gain larger than the fold standard deviation from the same run?
- Does it hold across at least two seeds?
- Did the leak checklist in `NOTES.md` get re-run?

If any answer is no, report it as inconclusive rather than as an improvement.

## Repo visibility

Private until the competition closes on 2026-08-31, then public. The competition
rules contemplate code being either private within a team or public on Kaggle, and
a public GitHub repo mid-competition is neither.

## Data

Binary classification. Target is `addicted_label`, 0 or 1 in `train.csv`, submitted
as a probability. Synthetic data generated from an original public dataset, which is
the usual Playground Series pattern. 13 features, mixed numeric and categorical,
with missingness on every feature between 4% and 20%. See `DATA.md` for the
column-level profile taken from the files themselves.

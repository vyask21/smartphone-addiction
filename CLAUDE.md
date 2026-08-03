# Competition-specific rules

The workspace rules in `../../CLAUDE.md` apply. Additions for this competition:

## Non-negotiable here

- Read `NOTES.md` before proposing anything. The rejected-ideas list exists so the
  same dead end is not walked twice.
- Every training run goes through `python -m src.train --config conf/<name>.yaml`.
  Do not train a model in a notebook and submit it.
- Every submission goes through `python -m src.submit --id <exp> --file <name>.csv`
  so the LB score lands in the ledger next to the CV score it belongs to.
- One variable per config. If two things change, make two configs.

## Before saying an idea worked

- Is the gain larger than the fold standard deviation printed by `train.py`?
- Does it hold across at least two seeds?
- Did the leak checklist in `NOTES.md` get re-run?

If any answer is no, report it as inconclusive rather than as an improvement.

## Data

<!-- filled in by /comp-eda -->

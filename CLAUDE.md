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

### That trigger fired, and the decision is not to move. 2026-08-12

The count passed 15 at row 17 on 2026-08-11 and nobody noticed for eight rows. The
ledger is at 25. Recording the call rather than letting a rule sit quietly broken.

**Not migrating**, for three reasons. There are 20 days to the deadline and the
workspace rules forbid refactoring in the last week of a competition, so a migration
started now would be finished inside the window it warns about. The second half of
the trigger has not fired: no two rows are impossible to tell apart, and that
condition is the one the layout was actually protecting against. And the count itself
overstates the problem, because rows 20 to 23 are one seed sweep entered as four rows,
which is bookkeeping rather than four separate ideas.

**What is being done instead**, since the trigger fired for a reason:

- Every row from 17 onward names the single variable it changes and the row it changes
  it against, in its own `notes` field. That is the provenance a config hash would
  have carried.
- Numbers that come from combining saved out-of-fold vectors rather than from a
  training run now come from a committed notebook run top to bottom. Row 24 came from
  a scratch script and that was a real gap; `16_oof_stack.ipynb` closes it.

**Where the cost actually landed**, stated plainly so the next competition can price
it: row 24 needed a caveat sentence in its notes field to be readable at all, because
its CV was measured under a different protocol from every row above it. Under the
module layout that would have been a config difference, visible without prose.

**The next competition starts on the module layout.** This decision is about the cost
of switching mid-flight, not a defence of the notebook layout.

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
the usual Playground Series pattern. 12 features, mixed numeric and categorical,
with missingness on every feature between 4% and 20%. See `DATA.md` for the
column-level profile taken from the files themselves.

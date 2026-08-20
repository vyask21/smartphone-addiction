"""Rewrite the parts of 12_public_writeup.ipynb that session 6 made stale.

The notebook told a two-reopening story. There are three, the third is the best
single model in the repo, and three hyperparameter sweeps came back null in a way
that turns the three into a rule. The headline numbers, the submission count and
the closing ledger were all written before any of that.

Run once, then `python writeup/writeup_numbers.py && python writeup/embed.py`,
then the local nbconvert gate, then the Kaggle Save and Run All.

Cells are located by a distinctive substring rather than by index, so this fails
loudly rather than silently patching the wrong cell if the notebook moves.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebooks" / "12_public_writeup.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


def find(needle, kind=None):
    hits = [i for i, c in enumerate(cells)
            if needle in src(c) and (kind is None or c["cell_type"] == kind)]
    if len(hits) != 1:
        raise SystemExit(f"expected 1 cell containing {needle!r}, found {len(hits)}")
    return hits[0]


def body(text):
    return [l + "\n" for l in text.strip("\n").split("\n")]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": body(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": body(text)}


# ------------------------------------------------------------------ 1. the header
HEADER = """
# Four weeks of negative results, and the finding I had to take back twice

### Playground Series S6E8, smartphone addiction

Best cross-validation **0.967925**, best public LB **0.96924**. Every logged
experiment is in the ledger at the bottom, and sixteen of them are nulls or negatives
that are written up here at the same length as the wins.

This is not a solution notebook. Nothing in it will move you up the leaderboard by
itself. It is a record of which ideas worked, which did not, and how each one was
killed cheaply enough that the next one still had time to be tried.

**The short version.** Fifteen experiments of model capacity and stochastic averaging
were worth **+0.0089 AUC** between them, and 80% of that was noticing the untuned
baseline was underfit. Not one engineered feature helped. Then one change of
*representation*, target encoding every column, was worth **+0.0029** on its own, and
it arrived after I had already written that the board was closed.

Then the interesting part. That change of representation quietly invalidated three
rejections I had already made and filed, and going back for them was worth more than
anything I did on purpose in the last two weeks.

Six things in here that I have not seen written up elsewhere for this dataset:

1. **Rank correlation between two models does not predict whether averaging them
   helps.** Measured across all 66 pairs of models I trained. This is the opposite of
   the usual advice, and it nearly cost me a day of CatBoost runs.
2. **I then over-claimed that finding, and the correction is the most useful thing
   here.** Every test behind it used an equal-weight combiner. Hold the eighteen
   models fixed and change only the combiner: averaged, they score **0.001554 below**
   the best single model; weighted by a logistic regression, **0.000908 above** it.
   The model I had rejected as the worst blend partner I ever measured takes the
   seventh largest weight of eighteen.
3. **The correction had a second half, and it took me ten more days to see it.**
   Fixing a combiner does not go back and re-examine the conclusions that were reached
   with the broken one. Three rejected models came back when I finally looked, and one
   of them, XGBoost, **had never been run at all**. It is the best single model in the
   competition for me.
4. **A rule for when a reopening is worth the compute, with six points on it and two
   predictions made in advance.** A change of representation revalues *learners*, not
   their *knobs*. Three learner reopenings paid; three hyperparameter sweeps on the
   same representation returned nothing, and the last two were called null in advance.
5. **Fold standard deviation is the wrong bar for a paired comparison** and it almost
   buried my one real improvement.
6. **The public leaderboard here has a standard error near 0.001**, larger than every
   gain after the fourth experiment. Two of my submissions demonstrate this cleanly.

Also, because it is the honest counterweight to point 3: the largest single-model gain
in the whole competition, **+0.026**, was worth **0.000043** to the thing I actually
submit. Effect size and incremental value are different quantities and this notebook
has the cleanest example of that I have produced.

The full experiment ledger, every notebook, and the working log are in the repo
linked at the bottom.
"""

cells[find("# Four weeks of negative results", "markdown")] = md(HEADER)

# --------------------------------------------- 2. the third reopening, a new section
XGB_MD = """
## The third reopening: the model I rejected without ever running it

The two above were models I had run and beaten. This one is worse, and it is the
reason I now separate the two statuses in writing.

Here is what I wrote in my own notes, in the list of things I was deliberately not
going to do:

> **XGBoost.** A third histogram GBDT on the same 12 features. The CatBoost result
> removes the reason to expect it to behave differently.

That is a conclusion resting entirely on a premise. The premise was that CatBoost had
lost. Two sections ago the premise was withdrawn, and the conclusion sat there for
another week with nothing under it, in a list of things I had decided not to do, which
is not a place anyone goes looking for assumptions to re-check.

So: same encoder, same folds, same seed, same budget as the LightGBM it was supposed to
resemble. One variable.
"""

XGB_CODE = """
r = D["reopenings"]
xg, cat = r["xgboost"], r["catboost"]
lgb_seeds = sorted(D["target_encoding"]["seed_cv"].values())
cat_seeds = sorted(cat["seed_cv"].values())
xgb_seeds = sorted(xg["seed_cv"].values())

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

# Left: three families, five seeds each, identical folds and encoder. Drawn as
# individual seeds because the spread is half the point.
ax = axes[0]
FAMS = [(lgb_seeds, MUTED, "LightGBM"), (cat_seeds, BLUE, "CatBoost"),
        (xgb_seeds, ORANGE, "XGBoost")]
for i, (vals, colour, lbl) in enumerate(FAMS):
    ax.plot(vals, [i] * len(vals), "o", color=colour, ms=9, alpha=0.85, lw=0, zorder=3)
    m = sum(vals) / len(vals)
    ax.plot([m], [i], "|", color=INK, ms=26, mew=2, zorder=4)
    ax.annotate(f"{lbl}   mean {m:.6f}", (m, i), textcoords="offset points",
                xytext=(0, 17), ha="center", fontsize=9.5, color=SECOND)
ax.set_yticks(range(len(FAMS)))
ax.set_yticklabels([])
ax.set_ylim(-0.6, 2.7)
ax.set_xlabel("cross-validation AUC, five model seeds each")
ax.grid(axis="y", visible=False)
finish(ax, "Three gradient boosters on the encoded features",
       f"XGBoost leads LightGBM by {xg['gap']:+.6f}, which is "
       f"{xg['gap_in_se']:.1f} standard errors of the\\ntwo-sample difference, and "
       f"leads CatBoost by {xg['gap_vs_catboost']:+.6f}.")

# Right: what each reopening was worth to the stack, which is the only currency the
# three are comparable in. The two null sweeps are drawn at zero on purpose.
ax = axes[1]
step = {c["n"]: c["step"] for c in D["stack_curve"] if c["step"]}
BARS = [("CatBoost", step[19]["gain"], BLUE),
        ("neural, on the\\nencoded features", step[24]["gain"], BLUE),
        ("XGBoost", step[25]["gain"], BLUE),
        ("num_leaves", 0.0, MUTED),
        ("encoder\\nsmoothing", 0.0, MUTED)]
xs = range(len(BARS))
ax.bar(xs, [b[1] for b in BARS], color=[b[2] for b in BARS], width=0.6, zorder=3)
ax.axhline(0, color=AXIS, lw=1.2, zorder=2)
for i, (lbl, v, _) in enumerate(BARS):
    if v:
        ax.annotate(f"{v:+.6f}", (i, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9, color=SECOND)
    else:
        ax.annotate("null", (i, 0), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9.5, color=SECOND)
ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in BARS], fontsize=9, color=SECOND, linespacing=1.3)
ax.set_ylim(0, step[19]["gain"] * 1.28)
# The distinction the panel is about goes in a legend, not in five tick labels.
ax.bar([0], [0], color=BLUE, label="a different learner")
ax.bar([0], [0], color=MUTED, label="a hyperparameter, best arm of the sweep")
ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=9)
ax.set_ylabel("gain in stack CV from adding it")
finish(ax, "Learners paid, knobs did not",
       "Every bar is a paired five-fold gain on the same out-of-fold matrix,\\n"
       "winning five folds out of five wherever it is non-zero.")

plt.tight_layout()
plt.show()
"""

XGB_RESULT = """
**XGBoost is the best single model in the competition for me**, at a family mean of
0.967118 against LightGBM's 0.966763. The gap is +0.000355 at 23.6 standard errors of
the two-sample difference, and it beats the CatBoost I had just spent a week reinstating
by a further +0.000197.

Two things about that number are worth saying out loud.

**It is family against family, and I got that wrong the first time.** I originally
logged the gap as +0.000336 at 35.1 standard errors, which compared one family mean
against a single seed's error bar. The correct comparison pools both families and gives
a wider interval on a slightly larger difference. It was caught because I compute these
numbers in two places, and the two disagreed. A duplicated calculation is not waste; it
is the only reason that error is in this notebook as a correction rather than in it as
a fact.

**One seed would not have been enough.** Row 38 was a single seed at +0.000316, which
is inside what a lucky seed can produce in this repo. It only became a result after
four more seeds, and that is the standing rule here: CatBoost and XGBoost both had to
clear it and both did.

### It paid for itself out of the LightGBM seeds, not out of the stack

The more interesting result is not the level, it is what happened to the combiner when
XGBoost joined. This is the substitution, and it is unusually clean.
"""

SUB_CODE = """
# Same members, one addition. The question is where the new weight comes from.
was = {c["name"]: c["coef"] for c in D["stack24_coefs"]}
now = {c["name"]: c["coef"] for c in D["stack29_coefs"]}
te = sorted([n for n in was if n.startswith("te")], key=lambda n: -was[n])
xgb = sorted([n for n in now if n.startswith("xgb")], key=lambda n: -now[n])

te_was, te_now = sum(was[n] for n in te), sum(now[n] for n in te)
xgb_now = sum(now[n] for n in xgb)

fig, ax = plt.subplots(figsize=(8.8, 5.0))
ax.axvline(0, color=AXIS, lw=1.2, zorder=1)

# Dumbbells: where each LightGBM target-encoded seed sat before, and after.
for i, n in enumerate(te):
    ax.plot([was[n], now[n]], [i, i], color=GRID, lw=2.5, zorder=2)
    ax.plot([was[n]], [i], "o", color=MUTED, ms=9, zorder=3)
    ax.plot([now[n]], [i], "o", color=ORANGE, ms=9, mec=SURFACE, mew=1.2, zorder=4)
# The five XGBoost seeds have no "before", so they are drawn as arrivals.
for j, n in enumerate(xgb):
    i = len(te) + 0.6 + j
    ax.plot([0, now[n]], [i, i], color=GRID, lw=2.5, zorder=2)
    ax.plot([now[n]], [i], "D", color=BLUE, ms=8, mec=SURFACE, mew=1.2, zorder=4)

ax.set_yticks(list(range(len(te))) + [len(te) + 0.6 + j for j in range(len(xgb))])
ax.set_yticklabels(te + xgb, fontsize=9)
ax.set_xlabel("weight in the logistic stack")
ax.grid(axis="y", visible=False)
ax.annotate(f"five LightGBM TE seeds\\n{te_was:+.4f} together, then {te_now:+.4f}",
            (was[te[0]], 0), textcoords="offset points", xytext=(10, 14),
            fontsize=9, color=SECOND, linespacing=1.35)
# Placed in the empty upper-left block rather than beside a marker, which is where
# the arrivals leave a gap.
ax.text(0.004, len(te) + 0.6 + len(xgb) - 1,
        f"five XGBoost seeds arrive\\nat {xgb_now:+.4f} together",
        fontsize=9, color=SECOND, linespacing=1.35, va="center")
finish(ax, "XGBoost did not add to the stack, it replaced part of it",
       "Grey is the 24-member fit, colour is the 29-member fit.\\n"
       "Same rows, same combiner, same regularisation. Only the member set moves.")
plt.tight_layout()
plt.show()

print(f"five LightGBM TE seeds, summed weight: {te_was:+.4f} -> {te_now:+.4f}")
print(f"five XGBoost seeds, summed weight:     {xgb_now:+.4f}")
print(f"stack CV: {D['stack_curve'][3]['cv']:.6f} -> {D['stack_curve'][-1]['cv']:.6f}")
"""

SUB_MD = """
**The five LightGBM target-encoded seeds go from +0.386 of weight between them to
+0.006.** They are not downweighted, they are switched off. XGBoost arrives carrying
+0.464 and the combiner takes almost exactly that much back out of the models it
replaced.

This is substitution, and it says something the CV column does not. Those five models
were never contributing five models' worth of information. They were one direction,
represented five times, and the moment a better representative of that direction turned
up the combiner stopped paying for them. The stack went up by only +0.000119 across both
XGBoost steps precisely because most of what XGBoost knows was already in there.

I then did the obvious follow-up and it is the one experiment in this competition that
tries to make the model *smaller*: drop those five members and refit. It costs **four
millionths**, which is nothing, on a stack of 29 members losing five of them.

**And I am not acting on it.** The five were selected by reading the coefficients off
the same out-of-fold matrix the result is then measured on, which is selection on the
validation set, and I had pre-registered a check that any drop be justified by
training-fold information alone. The check failed. It also turned out to be
mis-specified, comparing a maximum over folds against a mean over folds, so it was
close to guaranteed to fail whatever the data did. That is my error and it is in the
ledger as one.

The verdict stands as written anyway. Moving a pre-registered bar after seeing the
number is the exact failure the bar exists to prevent, and a bar that only binds when
it agrees with you is not a bar. The substantive conclusion survives without it: the
selection bias here can only *inflate* the pruned stack, since the members were chosen
for looking useless on these very rows, and the inflated estimate is still +0.000004.
"""

RULE_MD = """
## The rule that came out of the three, and the one prediction I made in advance

Three reopenings after the same event, all three paying, is a pattern worth trying to
state. So I stated it, and then I got two chances to be wrong about it.

After target encoding landed I also had three hyperparameter knobs that had never been
searched. `num_leaves` had sat at LightGBM's default of 31 for the entire competition
and appeared in no notebook. The encoder's own smoothing constant and its inner split
count had both been set on the day the encoder was written, chosen once, and inherited
by 22 of the 29 models in my best stack. All three are exactly the kind of thing that
looks negligent in a writeup.

All three returned nothing.

| reopened | what changed | result |
|---|---|---|
| CatBoost | the learner | **+0.000157** as a family |
| the neural model | the learner's input representation | **+0.026204** single-model |
| XGBoost | the learner | **+0.000355** as a family |
| `num_leaves`, swept 15 to 127 | a hyperparameter | null, 31 was already right |
| encoder smoothing, swept 1 to 100 | a hyperparameter | null, flat below 10 |
| encoder inner splits, swept 2 to 20 | a hyperparameter | null, flat above 5 |

**A change of representation revalues learners, not their knobs.** Which reads as
obvious written down, and was not obvious at all while I was deciding what to spend a
week of compute on.

The mechanism I would offer is that a learner and a representation are matched to each
other, so changing one genuinely re-runs the competition between learners. A knob is
fitted downstream of both. If it was near-optimal on the old representation it is
usually near-optimal on the new one, because the thing it is optimising against did not
move much.

The part that makes this worth including rather than a rationalisation: **the rule was
written down after four of those six rows and it called the last two in advance.** Both
sweeps were predicted null in their notebook headers, from the arithmetic, before the
runs. A rule invented after the last data point explains everything and predicts
nothing, and I would not be showing you this table if the fifth and sixth rows had been
found first.

The sixth is the better test, because that header did not only predict a null. It also
wrote down the strongest argument *against* the rule and made it falsifiable. The inner
split count controls a real train/serve mismatch: training rows get their encoding from
a fraction of the fold while validation rows get theirs from all of it, and raising the
count narrows that gap. That predicts a curve rising toward the top of the grid. It does
not rise. The notebook prints `monotone rising: False` from a check written before the
run, and the mismatch turns out to be real, measurable in the encoded columns, and worth
nothing in the model.

I am reporting that at the same volume as the null itself, because a header that only
mentions its predictions when they land is not making predictions.

It also carries a real cost that I should not hide. The rule says XGBoost was worth
running and `num_leaves` was not. It would have said the same thing about the rejections
I sat on for a week, so what it actually buys is a *reason to look*, not a new capability.
I had all the information needed to run XGBoost ten days before I ran it.

### The positive control, which is the part I would steal from this

A sweep whose predicted result is "no difference" has a failure mode that looks
identical to success. If the constant had never reached the encoder at all, every arm
would have been a copy, the curve would have come back perfectly flat, and I would have
written the same conclusion from a bug.

So the smoothing notebook builds one fold at two very different values and compares the
encoded columns directly, before any model runs. They differ by 0.366, so the constant
is live and the flat curve means what it says. It is four lines and it is the only
reason that null is worth anything.

The inner-split sweep sharpened this, and it is the version I would actually reuse. That
knob is read by one pass of the encoder and no other, so it has a side it must **not**
reach: training-row encodings have to move between arms, and validation-row encodings
have to stay identical to the bit. Measured before any model trained, the training
columns moved by 0.164 and the validation columns moved by exactly zero.

**A one-sided control only catches the knob that does nothing.** The knob that does too
much is the more dangerous of the two, because it does not produce a suspiciously flat
curve. It produces a perfectly plausible one, for a different experiment than the one
you think you are running.

Two constants, two nulls, and the pipeline now has nothing in it that was set once and
never checked. That is not a leaderboard result. It is the thing that makes me willing
to put the leaderboard result in writing.
"""

anchor = find("## Fold standard deviation is the wrong bar", "markdown")
cells[anchor:anchor] = [md(XGB_MD), code(XGB_CODE), md(XGB_RESULT),
                        code(SUB_CODE), md(SUB_MD), md(RULE_MD)]

# ------------------------------------------------- 3. the ledger figure and CV/LB
i = find("# Row 16 is the neural model at 0.9392", "code")
old = src(cells[i])
# Nine submissions instead of seven put two pairs of leaderboard labels close enough
# to overprint each other. Drop the label below the marker when its neighbour is
# within four experiments.
old = old.replace(
    'for xi, yi in zip(lb_x, lb_y):\n'
    '    ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points", xytext=(0, 9),\n'
    '                ha="center", fontsize=8.5, color=SECOND)',
    'for k, (xi, yi) in enumerate(zip(lb_x, lb_y)):\n'
    '    crowded = k and xi - lb_x[k - 1] < 4\n'
    '    ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points",\n'
    '                xytext=(0, -17 if crowded else 9), ha="center", fontsize=8.5,\n'
    '                color=SECOND)')
old = old.replace(
    'ax.annotate("untuned anchor", (1, by_id[1]), textcoords="offset points",\n'
    '            xytext=(10, 7), fontsize=9, color=SECOND)',
    'ax.annotate("untuned anchor", (1, by_id[1]), textcoords="offset points",\n'
    '            xytext=(12, -13), fontsize=9, color=SECOND)')
old = old.replace(
    'ax.annotate("target encoding", (17, by_id[17]), textcoords="offset points",\n'
    '            xytext=(4, -22), fontsize=9, color=SECOND)',
    'ax.annotate("target encoding", (17, by_id[17]), textcoords="offset points",\n'
    '            xytext=(4, -22), fontsize=9, color=SECOND)\n'
    'ax.annotate("XGBoost", (38, by_id[38]), textcoords="offset points",\n'
    '            xytext=(0, -21), ha="center", fontsize=9, color=SECOND)\n'
    'ax.annotate("the 29-member stack", (44, by_id[44]), textcoords="offset points",\n'
    '            xytext=(-4, 12), ha="right", fontsize=9, color=SECOND)\n'
    '# The tail is jagged because rows 35-37 and 46-49 are sweep arms, deliberately\n'
    '# run below the best model to find where a curve stops being flat. They are\n'
    '# experiments, not attempts, and they belong on this chart for that reason.')
old = old.replace(
    'finish(ax, "Every experiment in the ledger",\n'
    '       "Fifteen experiments of capacity and averaging, worth +0.0089 together.\\n"\n'
    '       "Then one change of representation, worth +0.0029 on its own.")',
    'finish(ax, "Every experiment in the ledger",\n'
    '       "Fifteen experiments of capacity and averaging, worth +0.0089 together.\\n"\n'
    '       "Then one change of representation, worth +0.0029, and three reopenings\\n"\n'
    '       "it made possible worth +0.0006 more on top of the stack.")')
old = old.replace('ax.set_xticks(x)', 'ax.set_xticks([n for n in x if n % 2 == 0])')
if old == src(cells[i]):
    raise SystemExit("ledger figure patch matched nothing")
cells[i]["source"] = body(old)

CVLB = """
The leaderboard sits above CV by between 0.0010 and 0.0022 on every one of the nine
submissions, and **seven of the eight consecutive comparisons agree in direction**.
That is the whole job of a validation scheme, and it is worth stating as a fraction
rather than as a slogan, because the one exception is the most informative submission
pair I have.

The strongest version of the check is the four submissions that changed the feature
set or the member set rather than a hyperparameter: CV moved +0.0029, +0.0009, +0.0001
and +0.0002, and the leaderboard moved +0.0032, +0.0007, +0.0001 and +0.0002. Row 24
was predicted at 0.9691 from the running offset before it was submitted and came back
at 0.96897.

Now the exception, which is the two submissions early on. My 3-seed and 5-seed blends are
separated **cleanly** by CV: the 5-seed wins 5 folds out of 5 with a paired standard
deviation of 0.000023. On the public leaderboard they scored **0.965090** and
**0.965080**, a gap of 0.00001 in the other direction.

That is not a CV/LB disagreement to diagnose. The public leaderboard here is a sample
of roughly 89,000 rows and its standard error is near **0.001**, larger than every
single gain after the fourth experiment. It is a 6e-5 effect measured with a 1e-3
ruler.

The practical consequence for final submission selection: the usual split is best-CV
plus best-public-LB, and when the public LB's noise floor exceeds the effects you are
choosing between, **best-public-LB is close to picking at random**. CV gets the
heavier weight here.

One more thing I got wrong, and it is embarrassing in a useful way. Three stacks in
the middle of that sequence were never submitted, because I had decided a submission
slot was scarce and each one was too close to its predecessor to be worth spending
one on. The daily limit is ten. I had never checked, and I was comparing each stack to
the previous stack rather than to the submission actually sitting on the leaderboard,
which is the comparison that decides whether a slot buys information.
"""
cells[find("The leaderboard sits above CV by between", "markdown")] = md(CVLB)

# -------------------------------------------------------- 4. the closing ledger
CLOSING = """
## The ledger, and what I would do differently

Every experiment is written down, including the failures. The failures are most of the
value: they are why the same dead end did not get walked twice, and several of them are
in this notebook at the same length as the wins.

**What worked**

- Fitting model capacity first. 80% of the gain from the first three weeks.
- Lower learning rate with a compensating tree count. A small gain, and a useful side
  effect: fold spread fell from 0.000816 to 0.000549, so every later comparison got
  easier to call.
- Bagged seed averaging, rank-blended, up to four seeds.
- Target and frequency encoding on all twelve columns, +0.0029, nested inside the fold
  loop and leak-checked three ways. The largest single decision in the competition, and
  the one that made everything below it possible.
- A logistic stack over everything trained, including the models that were individually
  rejected.
- Reopening three rejected models after the representation changed. CatBoost, the
  neural model, and XGBoost, which had never been run at all and turned out to be the
  best single model here.

**What did not**

- Missingness indicators. No signal, established in ninety seconds.
- Threshold and rule features. The generator caps them below the untuned baseline.
- External data. The original is 7,500 rows against 691,369, its distributions are
  warped relative to the synthetic, and it appears to be synthetic itself.
- Capacity diversity within LightGBM, under an equal-weight blend.
- Median-imputed columns on top of target encoding. +0.000023, a null.
- The decimal lattice. An 11-point effect in the data, -0.000132 in the model.
- `num_leaves`. Never searched until the last week, and 31 was already right.
- The encoder's smoothing constant. Set once and never varied, and flat below 10.
- The encoder's inner split count. Set once on the same day, and flat above 5. With it
  the pipeline has no constant left that was chosen once and never checked.
- Pruning five members out of the stack. It costs four millionths, and the
  pre-registered check said do not act on it, so I did not.

**What I got wrong and had to publish a correction to**

- The neural model, and with it the whole diversity conclusion. It is 0.0247 behind and
  it makes an *average* worse at every weight, which is what I measured and what I
  reported. What I then claimed was that diversity does not pay on this data, and that
  was a fact about equal-weight combiners rather than about the data. It takes a
  positive weight in the stack, and dropping it costs a tenth of the stack's gain.
- CatBoost, which I wrote off on a single fold of the raw features and which leads
  LightGBM by 13 standard errors on the encoded ones.
- The XGBoost gap, reported first as one family against one seed's error bar and
  corrected to family against family. Caught only because the number is computed in two
  places and they disagreed.

**What I would do differently**

1. **Time one fold before launching anything.** I started a CatBoost run estimated at
   30-50 minutes. It ran for two hours with `verbose=0` and no progress output before
   being killed. It was never hung: a single fold was ten minutes and the notebook
   called the pipeline twice. The estimate was wrong by 3x because no fold had been
   timed.
2. **Name the combiner in any claim about ensembling.** See above. This is the
   expensive one.
3. **Keep a list of rejections with the reason attached, and re-read it whenever the
   reason changes.** Every rejection in this competition was correct when it was made.
   Three of them stopped being correct on the same afternoon and none of them moved,
   because a decision, once filed, does not announce that its premise has expired. This
   is the cheapest thing on the list and it was worth the most.
4. **Not offer a mechanism from one data point.** I attributed the CV/LB gap to fold
   averaging on the first submission, patched the story on the second, and withdrew it
   on the third when the pattern did not hold.
5. **Check the constraint before optimising against it.** I held back three submissions
   to conserve a daily quota I had never looked up. It is ten.
6. **Not declare a board closed.** I wrote that sentence with eight rejected ideas
   behind it, and the two largest gains in the competition came after it. Then I wrote
   it again, and XGBoost came after that.

**What I would keep**

The checksummed fold split. It is ten lines, and it is the reason an out-of-fold vector
saved in week one could be stacked against one from week four with any confidence at
all. Half of this notebook would be unmeasurable without it.

And the habit of writing the predicted result into the notebook header before the run.
It costs nothing, and it is the only thing that separates a rule with six points on it
from a story told about six points.

---

Ledger, notebooks and working log: **github.com/vyask21/smartphone-addiction**

If one thing here is worth taking away, it is that a cheap test of a wrong idea is
worth more than an expensive test of a good one, and most ideas are wrong. Including,
three times in this notebook, mine.
"""
cells[find("## The ledger, and what I would do differently", "markdown")] = md(CLOSING)

# -------------------------------------------------------------------- write it out
for c in cells:
    if c["cell_type"] == "code":
        compile(src(c), "cell", "exec")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"patched {NB.name}: {len(cells)} cells")

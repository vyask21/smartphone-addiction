"""Figures for the public writeup. Rendered standalone first so they can be looked at.

Palette is the validated default: blue #2a78d6 and orange #eb6834 pass every check on
a white surface (the Kaggle notebook background). Chrome uses the reference ink and
gridline values.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)
# Produced by writeup_numbers.py from the out-of-fold vectors in artifacts/oof/.
# It is committed, unlike the vectors themselves: it holds only aggregate statistics,
# and without it nobody could regenerate these figures from the repo alone.
D = json.loads((HERE / "writeup_numbers.json").read_text())

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": SECOND, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "figure.dpi": 130,
})


def finish(ax, title, sub=None):
    """Title and subtitle placed by hand above the axes.

    set_title plus a text at y=1.02 collides: the title sits at roughly the same
    height as the subtitle and the two overprint. Stacking both explicitly in axes
    coordinates keeps them apart at any figure size.
    """
    n_lines = sub.count("\n") + 1 if sub else 0
    ax.text(0, 1.06 + 0.055 * n_lines, title, transform=ax.transAxes, color=INK,
            fontsize=12, fontweight="bold", va="bottom")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=SECOND,
                fontsize=9.5, va="bottom", linespacing=1.4)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ fig 1: ledger
# Every row except 16, the neural model at 0.9392: it is a different family, it was
# never submitted, and including it would flatten everything else into a straight
# line. Named in the title rather than dropped in silence.
led = [r for r in D["ledger"] if r["id"] != 16]
x = [r["id"] for r in led]
cv = [r["cv_mean"] for r in led]
# `is not None`, not truthiness. An unsubmitted row is null here and NaN is truthy,
# so a bare test would let a NaN through and rely on matplotlib silently dropping it.
lb_x = [r["id"] for r in led if r["lb_public"] is not None]
lb_y = [r["lb_public"] for r in led if r["lb_public"] is not None]

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(x, cv, color=BLUE, lw=2, marker="o", ms=5.5, label="cross-validation",
        zorder=3, mec=SURFACE, mew=1.5)
ax.plot(lb_x, lb_y, color=ORANGE, lw=0, marker="D", ms=7, label="public leaderboard",
        zorder=4, mec=SURFACE, mew=1.5)
for xi, yi in zip(lb_x, lb_y):
    ax.annotate(f"{yi:.5f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8.5, color=SECOND)
by_id = {r["id"]: r["cv_mean"] for r in led}
ax.annotate("untuned anchor", (1, by_id[1]), textcoords="offset points", xytext=(10, 7),
            fontsize=9, color=SECOND)
ax.annotate("5-seed bagged blend", (15, by_id[15]), textcoords="offset points",
            xytext=(-6, -20), ha="right", fontsize=9, color=SECOND)
ax.annotate("target encoding", (17, by_id[17]), textcoords="offset points",
            xytext=(4, -22), fontsize=9, color=SECOND)
ax.set_xlabel("experiment")
ax.set_ylabel("ROC AUC")
ax.set_xticks(x)
ax.legend(frameon=False, loc="lower right", labelcolor=SECOND)
finish(ax, "Every experiment in the ledger",
       "Fifteen experiments of model capacity and averaging, worth +0.0089 together.\n"
       "Then one change of representation, worth +0.0029 on its own.")
fig.tight_layout()
fig.savefig(OUTDIR / "fig1_ledger.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- fig 2: missingness
mi = D["missingness"]
names = [m["feature"] for m in mi]
lift = np.array([m["lift"] for m in mi])
# The band is +/- 2 standard errors, reconstructed from each feature's own z.
se = np.array([abs(m["lift"] / m["z"]) if m["z"] else np.nan for m in mi])

fig, ax = plt.subplots(figsize=(9, 5))
yy = np.arange(len(names))
ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
# Each feature gets its own interval. A single shared band would have to be drawn at
# the widest feature's standard error, which understates the certainty on the other
# eleven and would make the "all inside noise" claim look stronger than it is.
ax.errorbar(lift, yy, xerr=2 * se, fmt="o", color=BLUE, ms=8, mec=SURFACE, mew=1.5,
            ecolor=MUTED, elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=3,
            label="±2 standard errors")
ax.set_yticks(yy)
ax.set_yticklabels(names, fontsize=9.5, color=SECOND)
ax.set_xlabel("target rate when the feature is missing, minus when it is present")
ax.legend(frameon=False, loc="lower right", labelcolor=SECOND)
ax.grid(axis="y", visible=False)
# Computed, not typed. This subtitle used to carry a hardcoded 2.24 while the plotted
# data came out at 1.98, so the label and the points disagreed and the figure looked
# fine either way. A number in a caption gets read as measured; make it be measured.
crossing = sum(1 for m in mi if abs(m["z"]) < 2)
finish(ax, "Missingness carries no signal about the target",
       f"{crossing} of {len(mi)} intervals cross zero. The largest |z| is "
       f"{max(abs(m['z']) for m in mi):.2f}, against the 2.4\n"
       "you would expect from pure noise across twelve tests. This killed my\n"
       "top-priority idea in ninety seconds.")
fig.tight_layout()
fig.savefig(OUTDIR / "fig2_missingness.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ fig 3: blends
pairs = D["pairs"]
BAGGED = {"bagged seed 42", "bagged seed 2024", "bagged seed 7",
          "bagged seed 2025", "bagged seed 13"}
homog = [p for p in pairs if p["a"] in BAGGED and p["b"] in BAGGED]
rest = [p for p in pairs if not (p["a"] in BAGGED and p["b"] in BAGGED)]

r_all = np.corrcoef([p["spearman"] for p in pairs], [p["gain"] for p in pairs])[0, 1]
r_hom = np.corrcoef([p["spearman"] for p in homog], [p["gain"] for p in homog])[0, 1]
r_gap = np.corrcoef([p["cv_gap"] for p in pairs], [p["gain"] for p in pairs])[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

ax = axes[0]
ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
ax.plot([p["spearman"] for p in rest], [p["gain"] for p in rest], "o", color=MUTED,
        ms=6, alpha=0.55, lw=0, zorder=2, label="all other pairs")
ax.plot([p["spearman"] for p in homog], [p["gain"] for p in homog], "o", color=BLUE,
        ms=8, mec=SURFACE, mew=1.5, lw=0, zorder=3,
        label="identical config, seed only")
ax.set_xlabel("Spearman correlation between the two models")
ax.set_ylabel("blend gain over the better member")
ax.legend(frameon=False, loc="lower left", labelcolor=SECOND, fontsize=9)
finish(ax, "Correlation does not predict blend value",
       f"all 66 pairs r = {r_all:+.2f}   ·   the 10 clean pairs r = {r_hom:+.2f}")

ax = axes[1]
ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
ax.plot([p["cv_gap"] for p in pairs], [p["gain"] for p in pairs], "o", color=BLUE,
        ms=6.5, mec=SURFACE, mew=1, alpha=0.9, lw=0, zorder=2)
ax.set_xlabel("difference in CV between the two models")
ax.set_ylabel("blend gain over the better member")
finish(ax, "Relative strength does, partly by definition",
       f"r = {r_gap:+.2f}, and a fixed 50/50 weight forces much of it")
fig.tight_layout()
fig.savefig(OUTDIR / "fig3_blend.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- fig 4: seeds
sat = D["seed_saturation"]
n = [s["n_seeds"] for s in sat]
v = [s["cv"] for s in sat]

fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.plot(n, v, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=1.5, zorder=3)
for xi, yi in zip(n, v):
    ax.annotate(f"{yi:.6f}", (xi, yi), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9, color=SECOND)
for i in range(1, len(n)):
    ax.annotate(f"+{(v[i] - v[i - 1]) * 1e6:.0f}e-6", ((n[i] + n[i - 1]) / 2,
                (v[i] + v[i - 1]) / 2), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(n)
ax.set_xlabel("seeds in the rank-average blend")
ax.set_ylabel("cross-validation ROC AUC")
ax.set_ylim(min(v) - 0.00008, max(v) + 0.00012)
finish(ax, "Seed averaging saturates at four",
       "Each step is roughly half the last. Seeds six and seven "
       "would together buy about 0.00003.")
fig.tight_layout()
fig.savefig(OUTDIR / "fig4_seeds.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- fig 5: stack
# The correction. Same eighteen models on both sides of the left panel, so the only
# thing varying is the combiner, and the sign flips.
comb = {c["name"]: c for c in D["combiners"]}
SHOW = ["rank mean, 5 seeds", "logit stack, 5 seeds",
        "rank mean, all 18", "logit stack, all 18"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

ax = axes[0]
vals = [comb[k]["gain"] for k in SHOW]
yy = np.arange(len(SHOW))
ax.barh(yy, vals, color=[ORANGE if v < 0 else BLUE for v in vals], height=0.62,
        zorder=3)
ax.axvline(0, color=AXIS, lw=1.2, zorder=4)
for i, v in enumerate(vals):
    ax.annotate(f"{v:+.6f}", (v, i), textcoords="offset points",
                xytext=(8 if v > 0 else -8, 0), va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=SECOND)
ax.set_yticks(yy)
ax.set_yticklabels(SHOW, fontsize=9.5, color=SECOND)
ax.invert_yaxis()
ax.set_xlim(min(vals) * 1.55, max(vals) * 1.9)
ax.set_xlabel("gain over the best single model")
ax.grid(axis="y", visible=False)
finish(ax, "The combiner, not the members",
       "Bottom two rows are the same eighteen models. Averaging them loses;\n"
       "letting a logistic regression choose the weights wins.")

ax = axes[1]
co = D["stack_coefs"]
nn = [c for c in co if c["name"] == "neural"]
rest = [c for c in co if c["name"] != "neural"]
ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
ax.plot([c["cv"] for c in rest], [c["coef"] for c in rest], "o", color=MUTED, ms=7,
        alpha=0.75, lw=0, zorder=2, label="LightGBM, 17 of them")
ax.plot([c["cv"] for c in nn], [c["coef"] for c in nn], "o", color=ORANGE, ms=10,
        mec=SURFACE, mew=1.5, lw=0, zorder=3, label="the neural model")
ax.annotate("rejected on the evidence\nin the section above",
            (nn[0]["cv"], nn[0]["coef"]), textcoords="offset points", xytext=(12, -6),
            fontsize=9, color=SECOND, linespacing=1.35)
ax.set_xlabel("the member's own cross-validation AUC")
ax.set_ylabel("its weight in the stack")
ax.legend(frameon=False, loc="lower right", labelcolor=SECOND, fontsize=9)
finish(ax, "Weight is not strength",
       "The weakest member of the eighteen takes the seventh largest weight.\n"
       "The negative weights are underfit models used as corrections.")
fig.tight_layout()
fig.savefig(OUTDIR / "fig5_stack.png", bbox_inches="tight")
plt.close(fig)


# -------------------------------------------------------------- fig 6: the reopenings
# The second half of the correction. Left: CatBoost on the features it had never been
# tried on, five seeds each so the spread is visible and not just the level. Right: what
# each addition to the stack was actually worth, which is the number that keeps the
# section honest.
r = D["reopenings"]
cat = r["catboost"]
lgb_seeds = sorted(D["target_encoding"]["seed_cv"].values())
cat_seeds = sorted(cat["seed_cv"].values())

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

ax = axes[0]
for i, (vals, colour, lbl) in enumerate([(lgb_seeds, MUTED, "LightGBM"),
                                         (cat_seeds, BLUE, "CatBoost")]):
    ax.plot(vals, [i] * len(vals), "o", color=colour, ms=9, alpha=0.85, lw=0, zorder=3)
    m = sum(vals) / len(vals)
    ax.plot([m], [i], "|", color=INK, ms=26, mew=2, zorder=4)
    ax.annotate(f"{lbl}   spread {max(vals) - min(vals):.1e}", (m, i),
                textcoords="offset points", xytext=(0, 17), ha="center",
                fontsize=9.5, color=SECOND)
ax.set_yticks([0, 1])
ax.set_yticklabels([])
ax.set_ylim(-0.6, 1.7)
ax.set_xlabel("cross-validation AUC, five model seeds each")
ax.grid(axis="y", visible=False)
finish(ax, "CatBoost, on the features it was never tried on",
       f"Family means differ by {cat['gap']:+.6f}, "
       f"{cat['gap_in_se']:.1f} standard errors.\n"
       f"On the raw features it had lost by 0.001675.")

ax = axes[1]
cur = D["stack_curve"]
xs = [c["n"] for c in cur]
ys = [c["cv"] for c in cur]
ax.plot(xs, ys, "-o", color=BLUE, ms=8, lw=2, zorder=3)
for c in cur:
    if c["step"]:
        ax.annotate(f"{c['step']['gain']:+.6f}", (c["n"], c["cv"]),
                    textcoords="offset points", xytext=(-6, 10), ha="right",
                    fontsize=9, color=SECOND)
ax.annotate("eighteen members,\nwhere the correction ended", (xs[0], ys[0]),
            textcoords="offset points", xytext=(14, -4), fontsize=9, color=SECOND,
            linespacing=1.35)
ax.set_xticks(xs)
ax.set_xlabel("members in the stack")
ax.set_ylabel("stack CV, combiner fit inside the fold loop")
finish(ax, "Every addition after the first is worth less",
       "Two rejected models earned back, one of them the largest single-model\n"
       f"gain of the competition, for {ys[-1] - ys[0]:+.6f} of stack CV in total.")
fig.tight_layout()
fig.savefig(OUTDIR / "fig6_reopenings.png", bbox_inches="tight")
plt.close(fig)

print("wrote fig1_ledger.png fig2_missingness.png fig3_blend.png fig4_seeds.png "
      "fig5_stack.png fig6_reopenings.png")
print(f"  r(spearman, gain) all pairs = {r_all:+.3f}")
print(f"  r(spearman, gain) homogeneous = {r_hom:+.3f}  n={len(homog)}")
print(f"  r(cv gap, gain) = {r_gap:+.3f}")
nte = r["neural"]
print(f"  catboost family gap {cat['gap']:+.6f} at {cat['gap_in_se']:.1f} se, "
      f"seed spread {cat['range']:.2e} vs LightGBM {cat['lgb_range']:.2e}")
print(f"  neural {nte['cv_raw']:.6f} -> {nte['cv_te']:.6f} "
      f"({nte['paired']['gain']:+.6f})")
print(f"  stack curve " + " ".join(f"{c['n']}:{c['cv']:.6f}" for c in cur))
print(f"  total bought by both reopenings: "
      f"{cur[-1]['cv'] - cur[0]['cv']:+.6f} of stack CV")

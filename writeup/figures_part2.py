"""Figures for the second half of the competition.

Separate file from figures.py because these read different sources: the ledger for the
encoder ablation, a leaderboard snapshot for the tie structure, and the out-of-fold
vectors for the congruence test. Same palette and chrome as figures.py so the two sets
sit together in the writeup.

Run:  python writeup/figures_part2.py
"""
import csv
import glob
import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)

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
    n = sub.count("\n") + 1 if sub else 0
    ax.text(0, 1.06 + 0.055 * n, title, transform=ax.transAxes, color=INK,
            fontsize=12, fontweight="bold", va="bottom")
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, color=SECOND,
                fontsize=9.5, va="bottom", linespacing=1.4)
    ax.set_axisbelow(True)


LEDGER = {r["name"]: r for r in csv.DictReader((ROOT / "experiments.csv").open(encoding="utf-8"))}


def cv(name):
    return float(LEDGER[name]["cv_mean"])


# ---------------------------------------------------- fig 7: the encoder by model class
PAIRS = [("LightGBM", "lgb_te_fe", "lgb_raw_fe"),
         ("XGBoost", "xgb_te_fe", "xgb_raw_fe"),
         ("CatBoost", "cat_te_fe", "cat_raw_fe"),
         ("RealMLP", "realmlp10", "realmlp_raw_fe")]
fams = [p[0] for p in PAIRS]
cost = [cv(p[2]) - cv(p[1]) for p in PAIRS]

fig, ax = plt.subplots(figsize=(7.4, 4.0))
cols = [BLUE, BLUE, BLUE, ORANGE]
ax.barh(fams, cost, color=cols, height=0.62)
for f, c in zip(fams, cost):
    # Centred inside the bar. Placing it beyond the bar end put white text on a white
    # surface, which rendered as nothing at all in the first version of this figure.
    ax.text(c / 2, f, f"{c:+.4f}", va="center", ha="center",
            color="white", fontsize=9.5, fontweight="bold")
ax.set_xlabel("change in out-of-fold AUC when target encoding is removed")
ax.invert_yaxis()
ax.axvline(0, color=AXIS, linewidth=1)
# Headroom so the longest bar does not run into its own tick label.
ax.set_xlim(min(cost) * 1.18, 0)
finish(ax, "Target encoding is worth five times more to a network than to a tree",
       "Same frame, same folds, encoder on against encoder off. The three tree families\n"
       "agree to within 0.0025. The network sits an order of magnitude out.")
fig.savefig(OUTDIR / "fig7_encoder_by_class.png", bbox_inches="tight")
plt.close(fig)
print("fig7_encoder_by_class.png")

# ------------------------------------------------- fig 8: the leaderboard cannot resolve
snaps = sorted(glob.glob(str(ROOT / "artifacts" / "lb" / "*publicleaderboard*.csv")),
               key=os.path.getmtime)
if snaps:
    s = np.sort(pd.read_csv(snaps[-1])["Score"].to_numpy())[::-1]
    vals = sorted({round(v, 5) for v in s if v >= 0.97118}, reverse=True)
    counts = [int((np.round(s, 5) == v).sum()) for v in vals]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cols = [ORANGE if abs(v - 0.97128) < 1e-9 else BLUE for v in vals]
    ax.barh([f"{v:.5f}" for v in vals], counts, color=cols, height=0.66)
    for v, c in zip(vals, counts):
        ax.text(c + 1.2, f"{v:.5f}", f"{c}", va="center", color=SECOND, fontsize=9)
    ax.set_xlabel("teams at this exact public score")
    ax.set_ylabel("public score")
    ax.invert_yaxis()          # best score at the top, as a leaderboard reads
    dense = [v for v, c in zip(vals, counts) if c >= 5]
    ax.text(0.98, 0.06,
            "public split standard error 0.00061\n"
            f"{sum(c for c in counts if c >= 5)} teams inside "
            f"{max(dense) - min(dense):.5f}",
            transform=ax.transAxes, ha="right", color=SECOND, fontsize=9,
            bbox=dict(facecolor=SURFACE, edgecolor=AXIS, boxstyle="round,pad=0.45"))
    finish(ax, "The public leaderboard cannot resolve what the field was optimising",
           "Every distinct score in the top band, and how many teams hold it. My own score in\n"
           "orange. The standard error of the measurement is four times wider than the whole band.")
    fig.savefig(OUTDIR / "fig8_tie_structure.png", bbox_inches="tight")
    plt.close(fig)
    print("fig8_tie_structure.png")

# ------------------------------------------- fig 9: the congruence test is anti-predictive
train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
y = train["addicted_label"].to_numpy(np.int8)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = i
assert hashlib.sha256(folds.tobytes()).hexdigest()[:16] == "ec282b0968059676"


def prof(v):
    a = np.array([roc_auc_score(y[folds == f], v[folds == f]) for f in range(5)])
    return a - a.mean()


OOF, PUB = ROOT / "artifacts" / "oof", ROOT / "artifacts" / "public_oof"


def col(p):
    d = pd.read_csv(p)
    pref = [c for c in d.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    return d[(pref or [c for c in d.columns if c.lower() != "id"])[0]].to_numpy(float)


ver = []
for n in ("realmlp10", "tabm", "xgb_tuned", "cat_te_fe", "lgb_te_fe"):
    p = OOF / f"{n}_oof.npy"
    if p.exists():
        ver.append((n, np.load(p)))
for m in "abcdefg":
    p = ROOT / "artifacts" / "srcK" / f"oof_{m}.npy"
    if p.exists():
        ver.append((f"srcK_{m}", np.load(p)))

FOREIGN = [
    ("lookup transformer", PUB / "srcA_lookup-transformer-predicting-smartphone-addiction/oof.csv"),
    ("spline transformer", PUB / "srcD_contextualized-deep-univariate-spline-transformer/nonlinear_context_5fold_oof.csv"),
]

if ver and all(p.exists() for _, p in FOREIGN):
    pool = [prof(v) for _, v in ver]
    for f in sorted(glob.glob(str(ROOT / "artifacts" / "srcL" / "oof_*.npy")))[:20]:
        pool.append(prof(np.load(f)))
    med = np.median(np.array(pool), axis=0)
    cong = lambda v: float(np.corrcoef(prof(v), med)[0, 1])

    vgood = [cong(v) for _, v in ver]
    vbad = [(n, cong(col(p))) for n, p in FOREIGN]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.scatter(vgood, np.zeros(len(vgood)), s=70, color=BLUE, zorder=3,
               label=f"proven on our fold split ({len(vgood)})")
    # The two foreign members sit about 0.013 apart, so one shared label offset prints
    # them over each other. Stagger the height and push them to opposite sides.
    for k, (n, c) in enumerate(sorted(vbad, key=lambda t: t[1])):
        ax.scatter([c], [0], s=140, color=ORANGE, marker="X", zorder=4)
        dx, dy, ha = (-16, 36, "right") if k == 0 else (16, 14, "left")
        ax.annotate(n, (c, 0), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, color=ORANGE, fontsize=9.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9,
                                    shrinkA=0, shrinkB=7))
    ax.scatter([], [], s=110, color=ORANGE, marker="X", label="proven FOREIGN split (2)")
    ax.set_yticks([])
    ax.set_xlabel("congruence to the pool's median per-fold difficulty profile")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_ylim(-0.6, 1.6)
    finish(ax, "A published test for foreign cross-validation splits, falsified",
           "The claim is that a foreign member washes out the shared fold-difficulty shape and scores low.\n"
           "Both members I had already proved foreign score at the TOP of the range. The test is anti-predictive.")
    fig.savefig(OUTDIR / "fig9_congruence_fails.png", bbox_inches="tight")
    plt.close(fig)
    print("fig9_congruence_fails.png")

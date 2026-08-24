"""Detect whether an out-of-fold vector was built on OUR fold partition, without
needing the author's printed fold AUCs.

**The mechanism.** In a k-fold run, every row inside one fold is predicted by ONE
model, and the five fold-models differ. So a member built on our partition carries
fold-level structure aligned to our folds: its calibration inside our fold 0 belongs
to a different fitted model than its calibration inside our fold 1.

A member built on a DIFFERENT partition is a mixture. Each of our folds then contains
rows predicted by several of their fold-models, so the per-fold calibration averages
out and the between-fold spread collapses toward what random subsets of one model
would give.

**The statistic.** Within each of our folds, fit `y ~ a + b * logit(p)`. Collect the
five intercepts and five slopes and take their standard deviation. High spread means
five distinct models, which means our partition. Low spread means a mixture.

**Calibration.** This is not asserted, it is measured against ground truth in both
directions. KNOWN GOOD are members verified through their authors' printed per-fold
AUCs. KNOWN BAD are `lookup_srcA` and `spline_srcD`, both proven foreign by that same
gate. The threshold is placed between the two populations and reported with the gap,
so a reader can see whether the test actually separates or is being asked to.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
y = train["addicted_label"].to_numpy(np.int8)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = i


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.clip(np.log(p / (1 - p)), -30, 30)


def stat(v):
    """(intercept spread, slope spread) of a per-fold calibration fit."""
    a, b = [], []
    L = logit(v)
    for f in range(5):
        m = folds == f
        clf = LogisticRegression(C=1e6, max_iter=1000).fit(L[m].reshape(-1, 1), y[m])
        a.append(float(clf.intercept_[0]))
        b.append(float(clf.coef_[0][0]))
    return float(np.std(a)), float(np.std(b))


PUB = ROOT / "artifacts" / "public_oof"
OOF = ROOT / "artifacts" / "oof"
VER = {r["name"]: r for r in json.loads((PUB / "verification.json").read_text(encoding="utf-8"))}


def load_verified(name):
    r = VER[name]
    p = PUB / r["folder"] / r["oof_file"]
    if p.suffix == ".npy":
        return np.load(p).astype(float)
    df = pd.read_csv(p)
    pref = [c for c in df.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    col = pref or [c for c in df.columns if c.lower() != "id"]
    return df[col[0]].to_numpy(float)


GOOD, BAD = {}, {}
for n in ("cb_srcE", "lgb_srcE", "xgb_srcA", "realmlp_srcA", "tabm_srcA",
          "lgb_srcB", "realmlp_srcI", "hgb_srcH"):
    GOOD[n] = load_verified(n)
for n in ("realmlp10", "tabm", "xgb_tuned", "cat_te_fe", "lgb_te_fe"):
    GOOD["ours_" + n] = np.load(OOF / f"{n}_oof.npy")
for n in ("lookup_srcA", "spline_srcD"):
    BAD[n] = load_verified(n)

print("KNOWN GOOD (verified on our partition)")
g = []
for n, v in GOOD.items():
    s = stat(v)
    g.append(s[0])
    print(f"  {n:18} intercept sd {s[0]:.5f}   slope sd {s[1]:.5f}")
print("\nKNOWN BAD (proven foreign partition)")
b = []
for n, v in BAD.items():
    s = stat(v)
    b.append(s[0])
    print(f"  {n:18} intercept sd {s[0]:.5f}   slope sd {s[1]:.5f}")

lo_good, hi_bad = min(g), max(b)
print(f"\nlowest good {lo_good:.5f}   highest bad {hi_bad:.5f}")
if lo_good > hi_bad:
    thr = (lo_good + hi_bad) / 2
    print(f"SEPARATES. threshold {thr:.5f}, gap {lo_good - hi_bad:.5f}")
else:
    thr = None
    print("DOES NOT SEPARATE. This probe cannot admit anything; it is reported and dropped.")

if thr is not None:
    print("\nCANDIDATE LIBRARIES")
    for folder, glob in (("srcJ", "oof_*.npy"), ("srcK", "oof_*.npy")):
        d = ROOT / "artifacts" / folder
        rows = []
        for p in sorted(d.glob(glob)):
            s = stat(np.load(p))
            rows.append((p.name[4:-4], s[0]))
        ok = sum(1 for _, s in rows if s > thr)
        print(f"  {folder}: {ok}/{len(rows)} above threshold")
        for n, s in rows:
            print(f"     {n:16} {s:.5f}  {'ok' if s > thr else 'FAIL'}")

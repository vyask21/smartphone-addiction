"""Compute every number the public writeup will plot, from the saved artifacts.

Nothing here is read off NOTES.md. The point is that the writeup's figures are
reproduced from the out-of-fold vectors on disk, so a number in the notebook and a
number in the ledger cannot drift apart.

Writes a JSON blob to the scratchpad.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "writeup_numbers.json"

train = pd.read_csv(REPO / "data" / "raw" / "train.csv")
y = train["addicted_label"].to_numpy().astype(np.float32)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(skf.split(train, y)):
    folds[va] = i

res = {}

# ---------------------------------------------------------------- 1. missingness
# The diagnostic that killed the top-priority idea. Recomputed here rather than
# quoted, so the figure and the claim cannot drift.
FEATURES = [c for c in train.columns if c not in ("id", "addicted_label")]
base = float(y.mean())
miss = []
for c in FEATURES:
    m = train[c].isna().to_numpy()
    n = int(m.sum())
    rate = float(y[m].mean())
    # Standard error of the difference between the missing-subgroup rate and the
    # overall rate, treating each as a binomial proportion.
    se = float(np.sqrt(base * (1 - base) * (1 / n + 1 / (len(y) - n))))
    miss.append({"feature": c, "n_missing": n, "miss_frac": n / len(y),
                 "rate": rate, "lift": rate - base, "z": (rate - base) / se})
res["missing_base_rate"] = base
res["missingness"] = sorted(miss, key=lambda d: d["lift"])

# ------------------------------------------------------------------- 2. pairwise
# Every pair of saved single-model OOF vectors: rank correlation against the value
# of a 50/50 rank blend, measured relative to the BETTER of the two members. The
# claim being tested is that correlation predicts blend value. It does not.
OOF = REPO / "artifacts" / "oof"
models = {
    "anchor (100 trees)": "lgbm_default_anchor_seed42.npy",
    "300 trees": "lgbm_trees300_seed42.npy",
    "1000 trees": "lgbm_trees1000_seed42.npy",
    "2000 trees": "lgbm_trees2000_seed42.npy",
    "lr 0.10": "lgbm_lr01_n1000_seed42.npy",
    "lr 0.05": "lgbm_lr005_n2000_seed42.npy",
    "lr 0.03": "lgbm_lr003_n3333_seed42.npy",
    "bagged seed 42": "lgbm_bag08_lr005_n2000_seed42.npy",
    "bagged seed 2024": "lgbm_bag08_lr005_n2000_seed2024.npy",
    "bagged seed 7": "lgbm_bag08_lr005_n2000_seed7.npy",
    "bagged seed 2025": "lgbm_bag08_lr005_n2000_seed2025.npy",
    "bagged seed 13": "lgbm_bag08_lr005_n2000_seed13.npy",
}
vec = {k: np.load(OOF / v) for k, v in models.items()}
rank = {k: pd.Series(v).rank(pct=True).to_numpy() for k, v in vec.items()}


def per_fold(v):
    return np.array([roc_auc_score(y[folds == f], v[folds == f]) for f in range(5)])


cv = {k: per_fold(v) for k, v in vec.items()}
res["single_cv"] = {k: {"mean": float(v.mean()), "sd": float(v.std())}
                    for k, v in cv.items()}

pairs = []
for a, b in itertools.combinations(models, 2):
    blend = per_fold(0.5 * rank[a] + 0.5 * rank[b])
    better = cv[a] if cv[a].mean() >= cv[b].mean() else cv[b]
    d = blend - better
    # Spearman on a 5% sample: exact on 691k x 66 pairs is slow and the estimate is
    # stable to four decimals at this size, which is all the figure resolves.
    idx = np.arange(0, len(y), 20)
    rho = float(spearmanr(vec[a][idx], vec[b][idx]).statistic)
    pairs.append({
        "a": a, "b": b, "spearman": rho,
        "cv_a": float(cv[a].mean()), "cv_b": float(cv[b].mean()),
        "cv_gap": abs(float(cv[a].mean() - cv[b].mean())),
        "blend": float(blend.mean()),
        "gain": float(d.mean()),
        "paired_sd": float(d.std(ddof=1)),
        "folds_won": int((d > 0).sum()),
    })
res["pairs"] = pairs

# ------------------------------------------------------------- 3. seed saturation
# Cumulative rank blend as seeds are added, in the order they were run.
seed_order = ["bagged seed 42", "bagged seed 2024", "bagged seed 7",
              "bagged seed 2025", "bagged seed 13"]
cum = []
acc = np.zeros(len(y))
for i, k in enumerate(seed_order, start=1):
    acc = acc + rank[k]
    s = per_fold(acc / i)
    cum.append({"n_seeds": i, "cv": float(s.mean()), "sd": float(s.std())})
res["seed_saturation"] = cum

# -------------------------------------------------------------- 4. ledger for CV/LB
led = pd.read_csv(REPO / "experiments.csv")
res["ledger"] = led.where(pd.notna(led), None).to_dict("records")

OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(f"wrote {OUT}")
print(f"  {len(pairs)} model pairs")
print(f"  missingness: largest |z| = {max(abs(m['z']) for m in miss):.2f}")
print(f"  seed saturation: {[round(c['cv'], 6) for c in cum]}")
best = max(pairs, key=lambda p: p["gain"])
print(f"  best pair gain: {best['gain']:+.6f} at spearman {best['spearman']:.4f} "
      f"({best['a']} + {best['b']})")
corr = np.corrcoef([p["spearman"] for p in pairs], [p["gain"] for p in pairs])[0, 1]
print(f"  correlation between spearman and blend gain: {corr:+.3f}")
corr2 = np.corrcoef([p["cv_gap"] for p in pairs], [p["gain"] for p in pairs])[0, 1]
print(f"  correlation between cv gap and blend gain:   {corr2:+.3f}")

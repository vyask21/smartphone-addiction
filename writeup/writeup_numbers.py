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
    present = float(y[~m].mean())
    # Two-proportion z test: missing rows against PRESENT rows, with the pooled
    # standard error. Both halves have to describe the same contrast. Comparing the
    # missing subgroup against the overall base instead is wrong twice over: the
    # base contains the subgroup, so the difference is diluted by the missing
    # fraction, and the pooled SE below is derived for two disjoint groups.
    se = float(np.sqrt(base * (1 - base) * (1 / n + 1 / (len(y) - n))))
    miss.append({"feature": c, "n_missing": n, "miss_frac": n / len(y),
                 "rate": rate, "rate_present": present,
                 "lift": rate - present, "z": (rate - present) / se})
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

# ------------------------------------------------------- 4. the out-of-family test
# Everything in section 2 is LightGBM against LightGBM, so it cannot falsify the
# correlation claim: no pair there is genuinely decorrelated. The neural model is the
# one member of a different family, and it is the only vector in the repo whose
# Spearman falls below the within-family band. Recomputed here from the saved vector
# rather than quoted from the Kaggle log, same as everything else in this file.
neural = np.load(OOF / "neural_oof.npy")
neural_rank = pd.Series(neural).rank(pct=True).to_numpy()

blend5 = np.zeros(len(y))
for k in seed_order:
    blend5 = blend5 + rank[k]
blend5 = blend5 / len(seed_order)
base_folds = per_fold(blend5)

idx = np.arange(0, len(y), 20)
band = [p["spearman"] for p in pairs]
curve = []
for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    b = per_fold((1 - w) * blend5 + w * neural_rank)
    d = b - base_folds
    curve.append({"w": w, "blend": float(b.mean()), "gain": float(d.mean()),
                  "paired_sd": float(d.std(ddof=1)), "folds_won": int((d > 0).sum())})

res["neural"] = {
    "cv": float(per_fold(neural).mean()),
    "sd": float(per_fold(neural).std()),
    "lgbm_blend_cv": float(base_folds.mean()),
    "spearman_vs_blend": float(spearmanr(neural[idx], blend5[idx]).statistic),
    "within_family_spearman_min": float(min(band)),
    "within_family_spearman_max": float(max(band)),
    "catboost_spearman": 0.9877,   # fold-0 probe, not a saved five-fold vector
    "weight_curve": curve,
}

# -------------------------------------------------------------- 5. ledger for CV/LB
led = pd.read_csv(REPO / "experiments.csv")
# NOT `led.where(pd.notna(led), None)`. Assigning None into a float column coerces
# straight back to NaN, so that idiom silently emits bare NaN, which json.dumps
# writes and json.loads accepts as an extension while a strict parser rejects it.
# An unsubmitted row has to reach the notebook as a real null, because the chart
# filters on `is not None` and NaN is truthy.
res["ledger"] = [{k: (None if isinstance(v, float) and np.isnan(v) else v)
                  for k, v in r.items()} for r in led.to_dict("records")]

# allow_nan=False so this can never regress quietly again.
OUT.write_text(json.dumps(res, indent=1, allow_nan=False), encoding="utf-8")
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
n = res["neural"]
print(f"  neural CV {n['cv']:.6f}, spearman vs blend {n['spearman_vs_blend']:.4f} "
      f"(within-family band {n['within_family_spearman_min']:.4f}"
      f" to {n['within_family_spearman_max']:.4f})")
print(f"  best neural weight: {max(n['weight_curve'], key=lambda c: c['gain'])}")

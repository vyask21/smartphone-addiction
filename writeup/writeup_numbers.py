"""Compute every number the public writeup will plot, from the saved artifacts.

Nothing here is read off this repo. The point is that the writeup's figures are
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
from sklearn.linear_model import LogisticRegression
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

# ------------------------------------------------------------ 5. target encoding
# The one feature idea that worked, measured against the model it replaced. Paired on
# identical folds, so the difference is far more precise than either absolute number.
te = {"te42": "te_bag42_oof.npy", "te2024": "te_seed2024_oof.npy",
      "te7": "te_seed7_oof.npy", "te2025": "te_seed2025_oof.npy",
      "te13": "te_seed13_oof.npy"}
tev = {k: np.load(OOF / v) for k, v in te.items()}
te_folds = per_fold(tev["te42"])
d = te_folds - per_fold(vec["bagged seed 42"])
res["target_encoding"] = {
    "cv": float(te_folds.mean()), "sd": float(te_folds.std()),
    "vs_same_model_raw": float(d.mean()), "paired_sd": float(d.std(ddof=1)),
    "folds_won": int((d > 0).sum()),
    "seed_cv": {k: float(per_fold(v).mean()) for k, v in tev.items()},
}

# ------------------------------------------------- 6. the combiner, which was the bug
# Sections 2 and 4 measured EQUAL-WEIGHT combiners and the conclusion drawn from them
# was about the data. This holds the member set fixed and varies the combiner, which
# is the comparison that was never run. Every stacker is fit on half the out-of-fold
# rows and scored on the other half, five random splits: a stacker fit and scored on
# the same matrix reads high, and that optimism is exactly what would manufacture the
# result being claimed here.
members = dict(tev)
members.update({k.replace(" ", "_"): v for k, v in vec.items()})
members["neural"] = neural


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.clip(np.log(p / (1 - p)), -30, 30)


names = list(members)
L = np.column_stack([logit(members[k]) for k in names])
R = np.column_stack([pd.Series(members[k]).rank(pct=True).to_numpy() for k in names])
te_idx = [names.index(k) for k in te]
nn_idx = [names.index("te42"), names.index("neural")]
no_nn = [i for i, n in enumerate(names) if n != "neural"]

rng = np.random.default_rng(0)
rows = []
for _ in range(5):
    perm = rng.permutation(len(y))
    a, b = perm[: len(y) // 2], perm[len(y) // 2:]

    def stack(cols):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(L[a][:, cols], y[a])
        return roc_auc_score(y[b], m.decision_function(L[b][:, cols]))

    rows.append({
        "best single model": roc_auc_score(y[b], members["te42"][b]),
        "rank mean, 5 seeds": roc_auc_score(y[b], R[b][:, te_idx].mean(axis=1)),
        "rank mean, all 18": roc_auc_score(y[b], R[b].mean(axis=1)),
        "logit stack, 5 seeds": stack(te_idx),
        "logit stack, all 18": stack(list(range(len(names)))),
        "logit stack, 17 no neural": stack(no_nn),
        "rank mean, best + neural": roc_auc_score(y[b], R[b][:, nn_idx].mean(axis=1)),
        "logit stack, best + neural": stack(nn_idx),
    })
df = pd.DataFrame(rows)
base = df["best single model"]
res["combiners"] = [{"name": c, "auc": float(df[c].mean()),
                     "gain": float((df[c] - base).mean()),
                     "paired_sd": float((df[c] - base).std(ddof=1)),
                     "splits_won": int(((df[c] - base) > 0).sum())}
                    for c in df.columns]
dn = df["logit stack, all 18"] - df["logit stack, 17 no neural"]
res["neural_in_stack"] = {"gain": float(dn.mean()), "paired_sd": float(dn.std(ddof=1)),
                          "splits_won": int((dn > 0).sum())}

# Coefficients from a fit on all rows, paired with each member's own CV. The point of
# the figure is that the ordering is not monotone in strength.
full = LogisticRegression(C=1.0, max_iter=2000).fit(L, y)
res["stack_coefs"] = sorted(
    [{"name": n, "cv": float(per_fold(members[n]).mean()), "coef": float(c)}
     for n, c in zip(names, full.coef_[0])],
    key=lambda d: -d["cv"])

# ------------------------------------------------- 8. the two reopenings, 2026-08-17/18
# Section 6 established that the diversity conclusion was a fact about the combiner.
# What it did not do is go back to the models that had been rejected under BOTH the old
# combiner and the old feature set. Two had. Re-running them is what this section
# measures, and it took ten days to see that it was the obvious next move.

cat = {s_: np.load(OOF / f) for s_, f in [
    (42, "catboost_te_oof.npy"),
    (2024, "catboost_te_seed2024_oof.npy"),
    (7, "catboost_te_seed7_oof.npy"),
    (2025, "catboost_te_seed2025_oof.npy"),
    (13, "catboost_te_seed13_oof.npy")]}
nte = np.load(OOF / "neural_te_oof.npy")

xgb = {s_: np.load(OOF / f) for s_, f in [
    (42, "xgb_te_oof.npy"),
    (2024, "xgb_te_seed2024_oof.npy"),
    (7, "xgb_te_seed7_oof.npy"),
    (2025, "xgb_te_seed2025_oof.npy"),
    (13, "xgb_te_seed13_oof.npy")]}

cat_cv = {int(k): float(per_fold(v).mean()) for k, v in cat.items()}
lgb_cv = {int(k.replace("te", "") or 42): v
          for k, v in res["target_encoding"]["seed_cv"].items()}
c_arr = np.array(sorted(cat_cv.values()))
l_arr = np.array(sorted(lgb_cv.values()))
gap = float(c_arr.mean() - l_arr.mean())
gap_se = float(np.sqrt(c_arr.var(ddof=1) / len(c_arr) + l_arr.var(ddof=1) / len(l_arr)))

# Paired against the LightGBM it replaces, on identical folds.
dc = per_fold(cat[42]) - te_folds
dn = per_fold(nte) - per_fold(neural)

xgb_cv = {int(k): float(per_fold(v).mean()) for k, v in xgb.items()}
x_arr = np.array(sorted(xgb_cv.values()))
xgap = float(x_arr.mean() - l_arr.mean())
xgap_se = float(np.sqrt(x_arr.var(ddof=1) / len(x_arr)
                        + l_arr.var(ddof=1) / len(l_arr)))
dx = per_fold(xgb[42]) - te_folds

res["reopenings"] = {
    "catboost": {
        "seed_cv": cat_cv,
        "mean": float(c_arr.mean()), "sd": float(c_arr.std(ddof=1)),
        "range": float(c_arr.max() - c_arr.min()),
        "lgb_mean": float(l_arr.mean()), "lgb_sd": float(l_arr.std(ddof=1)),
        "lgb_range": float(l_arr.max() - l_arr.min()),
        "gap": gap, "gap_se": gap_se, "gap_in_se": gap / gap_se,
        # From the 2026-08-04 probe in 06_catboost.ipynb: ONE fold, raw 12 features,
        # 138,274 held-out rows. Labelled as a single fold wherever it is quoted,
        # because it is not comparable to the five-fold numbers beside it.
        "raw_feature_gap_fold0": -0.001675,
        "paired_vs_lgb": {"gain": float(dc.mean()), "sd": float(dc.std(ddof=1)),
                          "folds_won": int((dc > 0).sum())},
    },
    "neural": {
        "cv_raw": float(per_fold(neural).mean()),
        "cv_te": float(per_fold(nte).mean()),
        "paired": {"gain": float(dn.mean()), "sd": float(dn.std(ddof=1)),
                   "folds_won": int((dn > 0).sum())},
        "relative": float(dn.mean() / per_fold(neural).mean()),
        "spearman_vs_catboost": float(spearmanr(nte, cat[42]).statistic),
        "spearman_vs_lgb_te": float(spearmanr(nte, tev["te42"]).statistic),
        "gap_to_best_gbdt": float(per_fold(nte).mean() - c_arr.max()),
    },
    # The third reopening, 2026-08-19. Dropped in the 06 amendment as "a third
    # histogram GBDT" and never run, on the raw features, in the same pre-encoding
    # regime that produced the CatBoost rejection this writeup already overturns.
    "xgboost": {
        "seed_cv": xgb_cv,
        "mean": float(x_arr.mean()), "sd": float(x_arr.std(ddof=1)),
        "range": float(x_arr.max() - x_arr.min()),
        "gap": xgap, "gap_se": xgap_se, "gap_in_se": xgap / xgap_se,
        "gap_vs_catboost": float(x_arr.mean() - c_arr.mean()),
        "paired_vs_lgb": {"gain": float(dx.mean()), "sd": float(dx.std(ddof=1)),
                          "folds_won": int((dx > 0).sum())},
    },
}

# The saturation curve. Every stack here is fit INSIDE the fold loop, which is the
# protocol that makes its CV comparable to a single model's, and it is a different
# protocol from the split-half one used in section 6. Four member sets, each adding to
# the one above it, so the marginal value of each addition is readable directly.
stack_members = dict(members)
for k, v in cat.items():
    stack_members[f"cat{k}"] = v
stack_members["neural_te"] = nte
for k, v in xgb.items():
    stack_members[f"xgb{k}"] = v
SN = list(stack_members)
SL = np.column_stack([logit(stack_members[k]) for k in SN])

BASE18 = [n for n in SN if not n.startswith(("cat", "xgb")) and n != "neural_te"]
CATS = [f"cat{k}" for k in (42, 2024, 7, 2025, 13)]
XGBS = [f"xgb{k}" for k in (42, 2024, 7, 2025, 13)]
S23 = BASE18 + CATS
S24 = S23 + ["neural_te"]

SETS = [
    ("18: five TE seeds, twelve raw, one neural", BASE18),
    ("19: plus CatBoost", BASE18 + ["cat42"]),
    ("23: plus four CatBoost seeds", S23),
    ("24: plus the encoded neural model", S24),
    ("25: plus XGBoost", S24 + ["xgb42"]),
    ("29: plus four XGBoost seeds", S24 + XGBS),
]


def fold_wise(cols):
    """CV of a logistic stack fit on four folds and scored on the fifth."""
    o = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        m = LogisticRegression(C=1.0, max_iter=2000).fit(SL[np.ix_(tr, cols)], y[tr])
        o[va] = m.decision_function(SL[np.ix_(va, cols)])
    return np.array([roc_auc_score(y[folds == f], o[folds == f]) for f in range(5)])


curve, prev = [], None
for label, keep in SETS:
    pf = fold_wise([SN.index(n) for n in keep])
    step = None if prev is None else {
        "gain": float((pf - prev).mean()),
        "sd": float((pf - prev).std(ddof=1)),
        "folds_won": int(((pf - prev) > 0).sum())}
    curve.append({"name": label, "n": len(keep), "cv": float(pf.mean()),
                  "sd": float(pf.std()), "step": step})
    prev = pf
res["stack_curve"] = curve

# The 24-member coefficients, which are the actual result of the second reopening:
# the encoded neural model takes the largest weight in the stack and the WEAK one
# does not collapse.
def coefs_for(keep):
    cols = [SN.index(n) for n in keep]
    m = LogisticRegression(C=1.0, max_iter=2000).fit(SL[:, cols], y)
    return sorted(
        [{"name": n, "cv": float(per_fold(stack_members[n]).mean()), "coef": float(c)}
         for n, c in zip(keep, m.coef_[0])], key=lambda d: -d["coef"])


# Kept fitting the 24-member set explicitly. It used to fit all of SL, which meant
# the same thing only while 24 was every member there was.
res["stack24_coefs"] = coefs_for(S24)
# The 29-member fit, where XGBoost takes the largest weight and pays for it out of
# the five LightGBM target-encoded seeds rather than out of the stack as a whole.
res["stack29_coefs"] = coefs_for(S24 + XGBS)

# -------------------------------------------------------------- 7. ledger for CV/LB
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
print(f"  target encoding: {res['target_encoding']['vs_same_model_raw']:+.6f} "
      f"over the same model on raw features")
for c in res["combiners"]:
    print(f"  {c['name']:28} {c['gain']:+.6f}  {c['splits_won']}/5")
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
r = res["reopenings"]
print(f"  catboost TE: mean {r['catboost']['mean']:.6f} vs lgbm "
      f"{r['catboost']['lgb_mean']:.6f}, gap {r['catboost']['gap']:+.6f} "
      f"({r['catboost']['gap_in_se']:.1f} se), seed range {r['catboost']['range']:.2e} "
      f"vs {r['catboost']['lgb_range']:.2e}")
print(f"  neural TE: {r['neural']['cv_raw']:.6f} -> {r['neural']['cv_te']:.6f} "
      f"({r['neural']['paired']['gain']:+.6f}, {r['neural']['relative']:+.2%}), "
      f"still {r['neural']['gap_to_best_gbdt']:+.6f} from the best GBDT")
for c in res["stack_curve"]:
    st = "" if c["step"] is None else (f"  step {c['step']['gain']:+.6f} "
                                      f"({c['step']['folds_won']}/5)")
    print(f"  stack {c['n']:>2}: {c['cv']:.6f}{st}")
top = res["stack24_coefs"][0]
print(f"  largest coefficient in the 24-member stack: {top['name']} {top['coef']:+.4f}")
rx = res["reopenings"]["xgboost"]
print(f"  xgboost TE: mean {rx['mean']:.6f}, gap vs lgbm {rx['gap']:+.6f} "
      f"({rx['gap_in_se']:.1f} se), vs catboost {rx['gap_vs_catboost']:+.6f}, "
      f"seed range {rx['range']:.2e}")
t29 = res["stack29_coefs"][0]
print(f"  largest coefficient in the 29-member stack: {t29['name']} {t29['coef']:+.4f}")
te_now = sum(c["coef"] for c in res["stack29_coefs"] if c["name"].startswith("te"))
te_was = sum(c["coef"] for c in res["stack24_coefs"] if c["name"].startswith("te"))
print(f"  five TE seeds, summed coefficient: {te_was:+.4f} -> {te_now:+.4f}")

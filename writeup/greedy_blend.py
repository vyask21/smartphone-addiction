"""Greedy fixed-weight rank blend over the REJECTED pool, on top of the committed stack.

**Why members rejected from the stack are legal here.** A vector built on a foreign fold
partition leaks when used as a combiner FEATURE: its value on our training folds came from
a model that trained on rows inside our validation fold, so the combiner fit sees
validation information. Its out-of-fold vector is nonetheless honest PER ROW, because every
row is out-of-fold with respect to whatever partition its author used. A FIXED-WEIGHT rank
blend fits nothing on the fold structure, so the leak mechanism has nothing to act on.

That makes `boltuzamaki`'s 47 streams usable here even though row 154 rejected the library
wholesale as stack members: its own index names 4-fold, 5-fold and 10-fold members in one
manifest with no documented scheme.

**The discipline.** Weights are fitted on out-of-fold data, so selection over 47 candidates
could fit noise. Two guards, both from this repo's standing practice:

  1. a candidate is accepted only if it improves the blend on **5 of 5 folds**, not merely
     on the pooled mean;
  2. it must clear a floor of +0.00001 mean, and the search stops on the first failure.

The per-fold requirement is what makes this different from a leaderboard-driven search: a
weight that improves the pooled AUC by fitting one fold's noise cannot win five folds.
"""
import json, pathlib, io, contextlib, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "70_combiner_protocol.ipynb"

# Rebuild the committed stack: row 153's 175 members, pruned to 65, fold-wise combiner.
nb = json.loads(NB.read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for c in code[:5]:
        exec("".join(c["source"]), g)
y, folds, train, test = g["y"], g["folds"], g["train"], g["test"]
COLS, Loof, Ltest = g["COLS"], g["Loof"], g["Ltest"]

oof_base = np.zeros(len(train))
tst_base = np.zeros((5, len(test)))
for f in range(5):
    tr, va = folds != f, folds == f
    m = LogisticRegression(C=1.0, max_iter=4000).fit(Loof[np.ix_(tr, COLS)], y[tr])
    oof_base[va] = m.decision_function(Loof[np.ix_(va, COLS)])
    tst_base[f] = m.decision_function(Ltest[:, COLS])
tst_base = tst_base.mean(axis=0)

R = lambda v: (rankdata(v, method="average") - 0.5) / len(v)


def fold_aucs(v):
    return np.array([roc_auc_score(y[folds == f], v[folds == f]) for f in range(5)])


cur_oof, cur_tst = R(oof_base), R(tst_base)
base_per = fold_aucs(cur_oof)
print(f"committed stack: OOF {roc_auc_score(y, cur_oof):.6f}, folds "
      + " ".join(f"{a:.5f}" for a in base_per) + "\n")

# ---- the candidate pool: everything rejected from the stack -----------------------
CAND = {}
PUB = ROOT / "artifacts" / "public_oof"


def _col(p):
    d = pd.read_csv(p)
    pref = [c for c in d.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    c = pref or [x for x in d.columns if x.lower() != "id"]
    return d[c[0]].to_numpy(float)


for nm, o, t in [
    ("lookup_srcA",
     PUB / "srcA_lookup-transformer-predicting-smartphone-addiction/oof.csv",
     PUB / "srcA_lookup-transformer-predicting-smartphone-addiction/submission.csv"),
    ("spline_srcD",
     PUB / "srcD_contextualized-deep-univariate-spline-transformer/nonlinear_context_5fold_oof.csv",
     PUB / "srcD_contextualized-deep-univariate-spline-transformer/nonlinear_context_5fold_test_predictions.csv"),
]:
    a, b = _col(o), _col(t)
    if a.shape == (len(train),) and b.shape == (len(test),):
        CAND[nm] = (a, b)

BOLT = ROOT / "artifacts" / "bolt"
if (BOLT / "oof_predictions.parquet").exists():
    bo = pd.read_parquet(BOLT / "oof_predictions.parquet")
    bt = pd.read_parquet(BOLT / "test_predictions.parquet")
    shared = [c for c in bo.columns if c in bt.columns]
    for c in shared:
        a, b = bo[c].to_numpy(float), bt[c].to_numpy(float)
        if a.shape == (len(train),) and b.shape == (len(test),):
            CAND[f"bolt_{c}"] = (a, b)
    del bo, bt
print(f"{len(CAND)} candidate vectors in the rejected pool")

RANKED = {k: (R(o), R(t)) for k, (o, t) in CAND.items()}
solo = {k: roc_auc_score(y, o) for k, (o, _) in CAND.items()}
print("strongest five: " + ", ".join(
    f"{k} {solo[k]:.5f}" for k in sorted(solo, key=solo.get, reverse=True)[:5]) + "\n")

# ---- greedy, with the 5/5-fold requirement ----------------------------------------
GRID = (0.03, 0.06, 0.10, 0.15, 0.20)
FLOOR = 1e-5
picked = []
for step in range(12):
    best = None
    for k, (ro, rt) in RANKED.items():
        if k in [p[0] for p in picked]:
            continue
        for w in GRID:
            cand = (1 - w) * cur_oof + w * ro
            per = fold_aucs(cand)
            d = per - base_per if not picked else per - fold_aucs(cur_oof)
            if int((d > 0).sum()) == 5 and d.mean() >= FLOOR:
                if best is None or d.mean() > best[3]:
                    best = (k, w, per, d.mean())
    if best is None:
        print(f"step {step + 1}: no candidate wins 5/5 folds above the floor. STOP.")
        break
    k, w, per, gain = best
    ro, rt = RANKED[k]
    cur_oof = R((1 - w) * cur_oof + w * ro)
    cur_tst = R((1 - w) * cur_tst + w * rt)
    picked.append((k, w))
    print(f"step {step + 1}: +{k} at w={w:.2f}  gain {gain:+.6f}  "
          f"blend OOF {roc_auc_score(y, cur_oof):.6f}")

final = roc_auc_score(y, cur_oof)
print(f"\nfinal blended OOF {final:.6f}, base {roc_auc_score(y, R(oof_base)):.6f}, "
      f"gain {final - roc_auc_score(y, R(oof_base)):+.6f}")
print(f"picked: {picked}")
if picked:
    sub = pd.DataFrame({"id": test["id"].to_numpy(),
                        "addicted_label": (np.argsort(np.argsort(cur_tst)) + 0.5) / len(cur_tst)})
    out = ROOT / "submissions" / "stack_greedy_blend.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}")

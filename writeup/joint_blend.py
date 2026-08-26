"""Jointly optimise the blend weights, and test the aggregation function.

Row 159's blend was built by GREEDY sequential selection: each candidate's weight was
chosen against the running blend and then frozen. That is suboptimal by construction.
Once four partners are chosen, their weights should be fitted together, because the value
of one depends on how much of the others is already present.

Two things are searched, both on out-of-fold data, both guarded by the standing rule that
the result must beat row 159's blend on 5 of 5 folds:

  1. the four weights jointly, by coordinate descent over a fine grid;
  2. the aggregation function, since plain rank averaging is only one of several equally
     defensible ways to pool. Logit-of-rank weights the tails more, and the mean of raw
     logits is what the stack's own combiner uses internally.

The candidate set is FROZEN to row 159's four picks. This notebook does not re-select
members, so the only fitted quantities are five numbers on 691,369 rows.
"""
import contextlib
import io
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "70_combiner_protocol.ipynb"

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


# ---- row 159's four picks, frozen -------------------------------------------------
PUB = ROOT / "artifacts" / "public_oof"
BOLT = ROOT / "artifacts" / "bolt"
WK = ROOT / "artifacts" / "weak50"


def _col(p):
    d = pd.read_csv(p)
    pref = [c for c in d.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    return d[(pref or [x for x in d.columns if x.lower() != "id"])[0]].to_numpy(float)


bo = pd.read_parquet(BOLT / "oof_predictions.parquet",
                     columns=["lookup_v3_evidence", "foldsafe_te_wide"])
bt = pd.read_parquet(BOLT / "test_predictions.parquet",
                     columns=["lookup_v3_evidence", "foldsafe_te_wide"])
wo, wt = np.load(WK / "oof.npy"), np.load(WK / "test.npy")
wid = pd.read_csv(WK / "members.csv")["id"].tolist()
j45 = wid.index("m45")

PICKS = {
    "bolt_lookup_v3_evidence": (bo["lookup_v3_evidence"].to_numpy(float),
                                bt["lookup_v3_evidence"].to_numpy(float)),
    "bolt_foldsafe_te_wide": (bo["foldsafe_te_wide"].to_numpy(float),
                              bt["foldsafe_te_wide"].to_numpy(float)),
    "weak_m45": (wo[:, j45].astype(float), wt[:, j45].astype(float)),
    "lookup_srcA": (
        _col(PUB / "srcA_lookup-transformer-predicting-smartphone-addiction/oof.csv"),
        _col(PUB / "srcA_lookup-transformer-predicting-smartphone-addiction/submission.csv")),
}
NAMES = list(PICKS)
del bo, bt, wo, wt

# ---- row 159's sequential blend, reproduced as the thing to beat -------------------
SEQ = [("bolt_lookup_v3_evidence", 0.10), ("bolt_foldsafe_te_wide", 0.10),
       ("weak_m45", 0.04), ("lookup_srcA", 0.08)]
seq_oof, seq_tst = R(oof_base), R(tst_base)
for k, w in SEQ:
    o, t = PICKS[k]
    seq_oof = R((1 - w) * seq_oof + w * R(o))
    seq_tst = R((1 - w) * seq_tst + w * R(t))
seq_per = fold_aucs(seq_oof)
print(f"row 159 sequential blend reproduced: OOF {roc_auc_score(y, seq_oof):.6f} "
      f"(logged 0.970084)\n")

# ---- aggregation functions ---------------------------------------------------------
AGG = {
    "rank": (lambda v: R(v), lambda m: m),
    "logit_rank": (lambda v: _logit(np.clip(R(v), 1e-6, 1 - 1e-6)), lambda m: m),
}

BASE_O, BASE_T = oof_base, tst_base
P_O = {k: PICKS[k][0] for k in NAMES}
P_T = {k: PICKS[k][1] for k in NAMES}


def build(weights, agg, side="oof"):
    """Weighted average in the chosen space. weights[0] is the stack."""
    f = AGG[agg][0]
    base = BASE_O if side == "oof" else BASE_T
    parts = [f(base) * weights[0]]
    src = P_O if side == "oof" else P_T
    for k, w in zip(NAMES, weights[1:]):
        parts.append(f(src[k]) * w)
    return np.sum(parts, axis=0) / sum(weights)


best = None
for agg in AGG:
    w = np.array([1.0, 0.12, 0.12, 0.05, 0.10])          # sequential weights, normalised
    cur = roc_auc_score(y, build(w, agg))
    for sweep in range(4):
        for i in range(1, 5):
            for cand in (0.0, 0.02, 0.04, 0.06, 0.08, 0.11, 0.14, 0.18, 0.22, 0.28):
                w2 = w.copy()
                w2[i] = cand
                a = roc_auc_score(y, build(w2, agg))
                if a > cur + 1e-9:
                    cur, w = a, w2
    per = fold_aucs(build(w, agg))
    d = per - seq_per
    print(f"{agg:11} OOF {cur:.6f}  vs row 159 {d.mean():+.6f}  {int((d > 0).sum())}/5 folds "
          f"weights {np.round(w / w.sum(), 4).tolist()}")
    if int((d > 0).sum()) == 5 and (best is None or cur > best[0]):
        best = (cur, agg, w.copy())

print()
if best is None:
    print("Nothing beats row 159's sequential blend on 5 of 5 folds. NOT SUBMITTED.")
else:
    cur, agg, w = best
    print(f"BEST: {agg}, OOF {cur:.6f}, {cur - roc_auc_score(y, seq_oof):+.6f} over row 159")
    t = build(w, agg, side="test")
    sub = pd.DataFrame({"id": test["id"].to_numpy(),
                        "addicted_label": (np.argsort(np.argsort(t)) + 0.5) / len(t)})
    out = ROOT / "submissions" / "stack_joint_blend.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}")

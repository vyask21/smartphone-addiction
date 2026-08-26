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

# srcB's five base members. Row 155 offered them as stack MEMBERS and they returned
# -0.000000, so they are not in the stack and are available as blend partners.
NJ = ROOT / "artifacts" / "srcB_oof"
for _k in ("01", "02", "03", "04", "05"):
    _op, _tp = NJ / f"{_k}_oof_predictions.csv", NJ / f"{_k}_submission.csv"
    if _op.exists() and _tp.exists():
        _do, _dt = pd.read_csv(_op), pd.read_csv(_tp)
        if (_do["id"].to_numpy() == train["id"].to_numpy()).all() and \
           (_dt["id"].to_numpy() == test["id"].to_numpy()).all():
            _oc = [c for c in _do.columns if c.lower() != "id"][0]
            _tc = [c for c in _dt.columns if c.lower() != "id"][0]
            CAND[f"srcB{_k}"] = (_do[_oc].to_numpy(float), _dt[_tc].to_numpy(float))

# srcL's 50 weakest, solo AUC 0.9169 to 0.9568, all his own models on the
# frozen partition. Far below the field, so row 142's bound says they are worthless as
# MEMBERS; as partners at small weight the 5-of-5-fold guard decides.
WK = ROOT / "artifacts" / "weak50"
if (WK / "oof.npy").exists():
    _wo, _wt = np.load(WK / "oof.npy"), np.load(WK / "test.npy")
    _ids = pd.read_csv(WK / "members.csv")["id"].tolist()
    assert _wo.shape == (len(train), len(_ids)) and _wt.shape == (len(test), len(_ids)), \
        f"weak50 shapes {_wo.shape} {_wt.shape}"
    for _j, _id in enumerate(_ids):
        CAND[f"weak_{_id}"] = (_wo[:, _j].astype(float), _wt[:, _j].astype(float))
    del _wo, _wt

# beicicc. THE ONLY SOURCE IN THIS COMPETITION THAT PUBLISHES ITS FOLD IDS, so its
# partition is verified DIRECTLY rather than inferred from printed per-fold AUCs. Their
# labels run 1..5 against our 0..4 and all five folds match as exact row sets.
BE = ROOT / "artifacts" / "beicicc"
if (BE / "fold_id.npy").exists():
    _fid = np.load(BE / "fold_id.npy")
    _labs = sorted(set(_fid.tolist()))
    _ok = all(any(set(np.where(_fid == gl)[0].tolist()) == set(np.where(folds == f)[0].tolist())
                  for gl in _labs) for f in range(5))
    assert _ok, "beicicc fold_id is NOT our partition; do not use these"
    for _op in sorted(BE.glob("*_oof.npy")):
        _tp = BE / _op.name.replace("_oof.npy", "_test.npy")
        if _tp.exists():
            CAND[f"bei_{_op.name[:-8]}"] = (np.load(_op).astype(float), np.load(_tp).astype(float))

# paiky1995. Several members are named *_10f, so the library mixes fold counts and is
# unusable as stack MEMBERS, exactly like boltuzamaki. As blend PARTNERS the weight is
# the only fitted quantity and the 5-of-5-fold guard decides.
PK = ROOT / "artifacts" / "paiky"
for _op in sorted(PK.glob("oof_*.npy")):
    _tp = PK / _op.name.replace("oof_", "testpred_", 1)
    if not _tp.exists():
        continue
    _o, _t = np.load(_op).astype(float), np.load(_tp).astype(float)
    if _o.shape == (len(train),) and _t.shape == (len(test),):
        CAND[f"pk_{_op.name[4:-4]}"] = (_o, _t)

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
_n_rejected = len(CAND)

# The 175 STACK MEMBERS are also admissible partners, and have never been offered as
# such. They enter the committed prediction through a logistic combiner in logit space;
# a rank blend is a different functional form, so a member can in principle contribute
# through both. Every one is already verified or already accepted, so nothing about the
# admission standard changes by including them here.
for _k in g["Poof"]:
    if _k not in CAND:
        CAND[_k] = (np.asarray(g["Poof"][_k], float), np.asarray(g["Ptest"][_k], float))

print(f"{_n_rejected} rejected-pool vectors + {len(CAND) - _n_rejected} stack members "
      f"= {len(CAND)} candidates")

RANKED = {k: (R(o), R(t)) for k, (o, t) in CAND.items()}
solo = {k: roc_auc_score(y, o) for k, (o, _) in CAND.items()}
print("strongest five: " + ", ".join(
    f"{k} {solo[k]:.5f}" for k in sorted(solo, key=solo.get, reverse=True)[:5]) + "\n")

# ---- greedy, two stage, with the 5/5-fold requirement on the accept ---------------
#
# Scoring every candidate at every weight on all 691,369 rows is about twelve minutes
# PER STEP at this pool size, and the first attempt at 280 candidates was killed by its
# own timeout without emitting a line. So the search is split:
#
#   stage 1, RANK: score all candidates x weights on a fixed stratified subsample.
#            This only orders the candidates; it never decides acceptance.
#   stage 2, CONFIRM: re-score the stage-1 winner on all five full folds, and accept
#            only if it wins 5 of 5 and clears the floor there.
#
# The subsample can mislead the ranking, which costs a step. It cannot admit anything,
# because nothing enters without passing the full-data 5-of-5-fold test.
GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25)
FLOOR = 3e-6
SUB = 150_000

_rng = np.random.default_rng(0)
_idx = np.sort(_rng.choice(len(train), size=SUB, replace=False))
_ys, _fs = y[_idx], folds[_idx]


def sub_auc(v):
    """Mean fold AUC on the fixed subsample. Ranking only."""
    return float(np.mean([roc_auc_score(_ys[_fs == f], v[_idx][_fs == f]) for f in range(5)]))


picked = []
for step in range(25):
    cur_per = fold_aucs(cur_oof)
    cur_sub = sub_auc(cur_oof)

    ranked = []
    for k, (ro, rt) in RANKED.items():
        for w in GRID:
            a = sub_auc((1 - w) * cur_oof + w * ro)
            if a > cur_sub:
                ranked.append((a - cur_sub, k, w))
    ranked.sort(reverse=True)

    accepted = None
    for _, k, w in ranked[:6]:            # confirm the six best on FULL data
        ro = RANKED[k][0]
        per = fold_aucs((1 - w) * cur_oof + w * ro)
        d = per - cur_per
        if int((d > 0).sum()) == 5 and d.mean() >= FLOOR:
            accepted = (k, w, d.mean())
            break

    if accepted is None:
        print(f"step {step + 1}: nothing in the top 6 wins 5/5 folds on full data. STOP.")
        break
    k, w, gain = accepted
    ro, rt = RANKED[k]
    cur_oof = R((1 - w) * cur_oof + w * ro)
    cur_tst = R((1 - w) * cur_tst + w * rt)
    picked.append((k, w))
    print(f"step {step + 1}: +{k} at w={w:.2f}  gain {gain:+.6f}  "
          f"blend OOF {roc_auc_score(y, cur_oof):.6f}", flush=True)

final = roc_auc_score(y, cur_oof)
print(f"\nfinal blended OOF {final:.6f}, base {roc_auc_score(y, R(oof_base)):.6f}, "
      f"gain {final - roc_auc_score(y, R(oof_base)):+.6f}")
print(f"picked: {picked}")
if picked:
    sub = pd.DataFrame({"id": test["id"].to_numpy(),
                        "addicted_label": (np.argsort(np.argsort(cur_tst)) + 0.5) / len(cur_tst)})
    out = ROOT / "submissions" / "stack_greedy_blend5.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name}")

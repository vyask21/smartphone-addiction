"""Decide whether a public out-of-fold vector may enter this repo's stack.

**The rule.** A member is admitted only if we can PROVE its fold partition is ours.
We do not take the author's description for it.

**The test.** Authors print their own per-fold AUCs, in a log or a metrics csv. We
recompute per-fold AUC on the same vector using OUR fold assignment. If the two agree
in order, the partitions are the same partition, because a fold AUC is a property of
exactly which rows sit in the fold.

**Why it matters.** A vector built on a different partition is still out-of-fold PER
ROW, so it scores normally and looks clean. But the model behind its value on OUR
training rows trained on rows inside OUR validation fold. Fitting a combiner on that
leaks validation information into the fit, inflates CV relative to LB, and breaks the
CV-to-LB tracking this repo runs on. It is the defect the current public top scorer
cites for refusing to select his own leading submission.

Unverifiable is REJECTED, never assumed. That rule has already earned its keep: srcB's
dataset file `02_oof_predictions.csv` and his kernel's own `oof_preds.csv` are
different vectors at Spearman 0.9929, so admitting the dataset copy on the grounds
that it was "the same model" would have been wrong.

Run:  python writeup/verify_public_oof.py
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "artifacts" / "public_oof"
TOL = 1e-4

train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
y = train["addicted_label"].to_numpy(np.int8)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = i
assert hashlib.sha256(folds.tobytes()).hexdigest()[:16] == "ec282b0968059676"
print(f"our fold vector verified, {len(train):,} train / {len(test):,} test\n")

# name -> (folder, oof file, test file). Blends, hill-climbs and other people's stacks
# are deliberately absent: their weights may have been chosen against public LB
# feedback, and stacking a stack compounds that invisibly.
CANDIDATES = {
    "cb_srcE": ("srcE_s6e8-catboost", "oof_preds.csv", "test_preds.csv"),
    "lgb_srcE": ("srcE_s6e8-lgbm", "lgbm_oof.csv", "lgbm_test_preds.csv"),
    "xgb_srcA": ("srcA_xgboost-for-predicting-smartphone-addiction",
                 "oof.csv", "submission.csv"),
    "realmlp_srcA": ("srcA_realmlp-for-predicting-smartphone-addiction",
                     "oof.csv", "submission.csv"),
    "tabm_srcA": ("srcA_tabm-for-predicting-smartphone-addiction",
                  "oof.csv", "submission.csv"),
    "lgb_srcB": ("srcB_single-lgbm-model-lb-0-96990-cv-0-96862",
                 "oof_preds.csv", "submission_lgbm.csv"),
    "lookup_srcA": ("srcA_lookup-transformer-predicting-smartphone-addiction",
                    "oof.csv", "submission.csv"),
    "realmlp_srcI": ("srcI_realmlp-for-predicting-smartphone-addiction",
                     "oof_preds.csv", "submission.csv"),
    "hgb_srcH": ("srcH_s6e8-histgradientboosting-lb-0-96945",
                 "tehgbc_oof_preds.csv", "tehgbc_test_preds.csv"),
    "xgb_srcC": ("srcC_simple-xgb-starter", "oof_predictions.csv", "submission.csv"),
    "spline_srcD": ("srcD_contextualized-deep-univariate-spline-transformer",
                   "nonlinear_context_5fold_oof.csv",
                   "nonlinear_context_5fold_test_predictions.csv"),
    "resnet_kava": ("srcG_predicting-smartphone-addiction-resnet-fe",
                    "oof_preds_resnet_MYSELF_95687.npy",
                    "test_preds_resnet_MYSELF_95687.npy"),
}

# Two accepted log shapes. The first tolerates another number between the fold index
# and the score, "Fold 1 Best Iteration: 4131 | ROC-AUC: 0.96607", which a
# digit-excluding pattern silently fails on.
PATTERNS = [
    re.compile(r"[Ff]old\s*\d+.{0,60}?(?:ROC-?AUC|AUC|auc)\s*[:=|]{0,2}\s*(0\.9[0-9]{4,6})"),
    re.compile(r"[Ff]old\s*\d+\s*[|:][^0-9\\]{0,40}(0\.9[0-9]{4,6})"),
]


def load_vector(path, n_expected, order_ref):
    """Return (values, id_order_ok). id_order_ok is None when the file carries no id."""
    if path.suffix == ".npy":
        v = np.load(path)
        return (v.astype(float), None) if len(v) == n_expected else (None, None)
    df = pd.read_csv(path)
    pref = [c for c in df.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    col = pref or [c for c in df.columns
                   if c.lower() not in ("id", "addicted_label", "target", "fold")]
    if not col:
        col = [c for c in df.columns if c.lower() != "id"]
    if not col or len(df) != n_expected:
        return None, None
    idc = [c for c in df.columns if c.lower() == "id"]
    ok = bool((df[idc[0]].to_numpy() == order_ref).all()) if idc else None
    return df[col[0]].to_numpy(float), ok


def author_fold_aucs(folder):
    """The author's own five fold AUCs, from a metrics csv if present, else the log."""
    for m in list(folder.glob("*metric*.csv")) + list(folder.glob("*fold*.csv")):
        try:
            df = pd.read_csv(m)
        except Exception:
            continue
        for c in df.columns:
            if "auc" in c.lower() and len(df) >= 5:
                vals = pd.to_numeric(df[c], errors="coerce").dropna().tolist()
                if len(vals) >= 5 and all(0.9 < v < 1 for v in vals[:5]):
                    return [float(v) for v in vals[:5]], m.name
    for lg in folder.glob("*.log"):
        if lg.stat().st_size == 0:
            continue
        txt = lg.read_text(encoding="utf-8", errors="ignore")
        for pat in PATTERNS:
            hits = [float(h) for h in pat.findall(txt)]
            if len(hits) >= 5:
                return hits[:5], lg.name
    return [], None


report = []
for name, (folder, ofile, tfile) in CANDIDATES.items():
    d = PUB / folder
    e = {"name": name, "folder": folder, "oof_file": ofile, "test_file": tfile}
    op, tp = d / ofile, d / tfile
    if not op.exists() or not tp.exists():
        e["verdict"] = "REJECT: oof or test file missing"
        report.append(e); print(f"{name}: {e['verdict']}\n"); continue

    v, idok = load_vector(op, len(train), train["id"].to_numpy())
    tv, tidok = load_vector(tp, len(test), test["id"].to_numpy())
    if v is None or tv is None:
        e["verdict"] = "REJECT: unreadable, or wrong row count"
        report.append(e); print(f"{name}: {e['verdict']}\n"); continue
    e.update(oof_id_order=idok, test_id_order=tidok, overall_auc=float(roc_auc_score(y, v)))

    ours = [float(roc_auc_score(y[folds == f], v[folds == f])) for f in range(5)]
    theirs, srcname = author_fold_aucs(d)
    e.update(our_fold_aucs=ours, their_fold_aucs=theirs, evidence=srcname)

    if idok is False or tidok is False:
        e["verdict"] = "REJECT: id order does not match the competition files"
    elif len(theirs) >= 5:
        e["max_abs_diff"] = max(abs(a - b) for a, b in zip(ours, theirs))
        e["verdict"] = ("ADMISSIBLE: fold AUCs match ours in order, same partition"
                        if e["max_abs_diff"] < TOL else
                        "REJECT: fold AUCs differ, a different partition")
    else:
        e["verdict"] = "REJECT: no author fold AUCs found, unverifiable"
    report.append(e)

    print(name)
    print(f"   AUC {e['overall_auc']:.6f}   oof id order {idok}   test id order {tidok}")
    print(f"   ours   : {' '.join(f'{a:.5f}' for a in ours)}")
    if theirs:
        print(f"   theirs : {' '.join(f'{a:.5f}' for a in theirs)}   [{srcname}]")
    if "max_abs_diff" in e:
        print(f"   diff   : {e['max_abs_diff']:.2e}")
    print(f"   {e['verdict']}\n")

(PUB / "verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
ok = [r for r in report if r["verdict"].startswith("ADMISSIBLE")]
bad = [r for r in report if not r["verdict"].startswith("ADMISSIBLE")]
print(f"ADMISSIBLE ({len(ok)}): " + ", ".join(f"{r['name']} {r['overall_auc']:.6f}" for r in ok))
print(f"REJECTED   ({len(bad)}): " + ", ".join(r["name"] for r in bad))

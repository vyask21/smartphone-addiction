"""Verify that a public out-of-fold vector was produced under OUR fold partition.

This is the gate that decides whether a member built by someone else may enter this
repo's stack, and it does not rely on trusting the author's description.

**The test.** Each author's kernel log prints their own per-fold AUCs. We recompute
per-fold AUC on the SAME vector using OUR fold assignment. If the two agree in order
to within 1e-4, the partitions are the same partition, because a fold AUC is a
property of exactly which rows are in the fold.

**Why it matters.** If the partitions differ, the author's vector is still out-of-fold
per row, so it looks clean. But the model that produced its value on OUR training rows
was trained on rows sitting in OUR validation fold. Fitting a combiner on that leaks
validation information into the fit, inflates CV, and breaks the CV-to-LB tracking
this repo runs on. That is exactly the defect the current public top scorer cites for
refusing to select his own leading submission.

A vector that cannot be verified is REJECTED rather than assumed, per the standing
rule that a fallback producing a less rigorous answer should be an assert.
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

train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
y = train["addicted_label"].to_numpy(np.int8)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = i
assert hashlib.sha256(folds.tobytes()).hexdigest()[:16] == "ec282b0968059676"
print(f"our fold vector verified, {len(train):,} rows\n")

CANDIDATES = [
    ("cb_srcE", "srcE_s6e8-catboost", "oof_preds.csv"),
    ("lgb_srcE", "srcE_s6e8-lgbm", "lgbm_oof.csv"),
    ("xgb_srcA", "srcA_xgboost-for-predicting-smartphone-addiction", "oof.csv"),
    ("realmlp_srcA", "srcA_realmlp-for-predicting-smartphone-addiction", "oof.csv"),
    ("tabm_srcA", "srcA_tabm-for-predicting-smartphone-addiction", "oof.csv"),
]

# Two accepted shapes, tried in order. The first handles a line that puts another
# number between the fold index and the score, "Fold 1 Best Iteration: 4131 |
# ROC-AUC: 0.96607", which a digit-excluding pattern silently fails on. Falling back
# to the looser shape only when the explicit one finds fewer than five folds.
FOLD_AUC_PATTERNS = [
    re.compile(r"[Ff]old\s*\d+.{0,60}?(?:ROC-?AUC|AUC|auc)\s*[:=]?\s*(0\.9[0-9]{4,6})"),
    re.compile(r"[Ff]old\s*\d+\s*[|:][^0-9\\]{0,40}(0\.9[0-9]{4,6})"),
]


def scrape_fold_aucs(txt):
    for pat in FOLD_AUC_PATTERNS:
        hits = [float(m) for m in pat.findall(txt)]
        if len(hits) >= 5:
            return hits
    return []

report = []
for name, folder, fname in CANDIDATES:
    d, p = PUB / folder, PUB / folder / fname
    entry = {"name": name, "source": f"{folder}/{fname}"}
    if not p.exists():
        entry["verdict"] = "REJECT: file missing"
        report.append(entry)
        print(f"{name}: {entry['verdict']}\n")
        continue

    df = pd.read_csv(p)
    # Prefer an explicit prediction column. Taking the first non-id column grabs
    # addicted_label in these files, which scores AUC 1.0 and reads as a miracle.
    pref = [c for c in df.columns if any(k in c.lower() for k in ("oof", "pred", "prob"))]
    col = pref or [c for c in df.columns
                   if c.lower() not in ("id", "addicted_label", "target")]
    if not col:
        entry["verdict"] = f"REJECT: no prediction column in {list(df.columns)}"
        report.append(entry)
        print(f"{name}: {entry['verdict']}\n")
        continue
    v = df[col[0]].to_numpy(float)
    entry.update(column=col[0], rows=int(len(v)))

    if len(v) != len(train):
        entry["verdict"] = f"REJECT: {len(v):,} rows, expected {len(train):,}"
        report.append(entry)
        print(f"{name}: {entry['verdict']}\n")
        continue

    idc = [c for c in df.columns if c.lower() == "id"]
    entry["id_order_ok"] = (bool((df[idc[0]].to_numpy() == train["id"].to_numpy()).all())
                            if idc else None)

    ours = [float(roc_auc_score(y[folds == f], v[folds == f])) for f in range(5)]
    entry["our_fold_aucs"] = ours
    entry["overall_auc"] = float(roc_auc_score(y, v))

    logs = list(d.glob("*.log"))
    theirs = []
    if logs and logs[0].stat().st_size > 0:
        txt = logs[0].read_text(encoding="utf-8", errors="ignore")
        theirs = scrape_fold_aucs(txt)
    entry["their_printed_aucs"] = theirs[:5]

    if entry["id_order_ok"] is False:
        entry["verdict"] = "REJECT: id order does not match train.csv"
    elif len(theirs) >= 5:
        diffs = [abs(a - b) for a, b in zip(ours, theirs[:5])]
        entry["max_abs_diff"] = max(diffs)
        entry["verdict"] = ("ADMISSIBLE: fold AUCs match ours in order, same partition"
                            if max(diffs) < 1e-4 else
                            "REJECT: fold AUCs differ, a different partition")
    else:
        entry["verdict"] = "REJECT: could not read the author's fold AUCs, unverifiable"
    report.append(entry)

    print(name)
    print(f"   overall AUC {entry['overall_auc']:.6f}   id order ok: {entry['id_order_ok']}")
    print(f"   on OUR folds : {' '.join(f'{a:.5f}' for a in ours)}")
    if theirs[:5]:
        print(f"   they printed : {' '.join(f'{a:.5f}' for a in theirs[:5])}")
    if "max_abs_diff" in entry:
        print(f"   max abs diff : {entry['max_abs_diff']:.2e}")
    print(f"   {entry['verdict']}\n")

(PUB / "verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
ok = [r["name"] for r in report if r.get("verdict", "").startswith("ADMISSIBLE")]
bad = [r["name"] for r in report if not r.get("verdict", "").startswith("ADMISSIBLE")]
print(f"ADMISSIBLE ({len(ok)}): {ok}")
print(f"REJECTED   ({len(bad)}): {bad}")

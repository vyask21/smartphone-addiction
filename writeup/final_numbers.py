"""Compute the numbers for the final arc of the public writeup, rows 122 to 144.

Companion to `writeup_numbers.py`, which stops at the 29-member stack and predates
RealMLP, TabM, the encoder-off view and the rank analysis.

Same rule as that file and it is the point of both: **nothing here is read off
NOTES.md**. Member scores and correlations are recomputed from the out-of-fold
vectors on disk. Gate gains come from `experiments.csv`, which is exact rather than
approximate: every stack arm in a gate is scored on the same five folds, so the
difference of the two `cv_mean` values IS the paired mean, not a proxy for it.

Writes `writeup/final_numbers.json`.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OOF = REPO / "artifacts" / "oof"
OUT = HERE / "final_numbers.json"

train = pd.read_csv(REPO / "data" / "raw" / "train.csv")
y = train["addicted_label"].to_numpy(np.int8)
folds = np.full(len(train), -1, dtype=np.int64)
for i, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    folds[va] = i
assert hashlib.sha256(folds.tobytes()).hexdigest()[:16] == "ec282b0968059676"

ledger = {r["id"]: r for r in csv.DictReader((REPO / "experiments.csv").open(encoding="utf-8"))}
res = {}


def load(stem):
    for p in (OOF / f"{stem}_oof.npy", OOF / f"{stem}.npy"):
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(stem)


def cv(stem):
    v = load(stem)
    per = np.array([roc_auc_score(y[folds == f], v[folds == f]) for f in range(5)])
    return float(per.mean()), float(per.std())


# ------------------------------------------------------- 1. the encoder, by model class
# The finding rows 141/142 produced: what removing the target encoder costs each
# learner family, on an otherwise identical frame. Every figure recomputed here.
PAIRS = [("LightGBM", "lgb_te_fe", "lgb_raw_fe"),
         ("XGBoost", "xgb_te_fe", "xgb_raw_fe"),
         ("CatBoost", "cat_te_fe", "cat_raw_fe"),
         ("RealMLP", "realmlp10", "realmlp_raw_fe")]
enc = []
for fam, on, off in PAIRS:
    a, _ = cv(on)
    b, _ = cv(off)
    enc.append({"family": fam, "encoder_on": on, "encoder_off": off,
                "cv_on": a, "cv_off": b, "cost": b - a})
res["encoder_by_class"] = enc
tree_costs = [e["cost"] for e in enc if e["family"] != "RealMLP"]
res["encoder_summary"] = {
    "tree_cost_min": max(tree_costs), "tree_cost_max": min(tree_costs),
    "net_cost": [e["cost"] for e in enc if e["family"] == "RealMLP"][0],
    "ratio_vs_worst_tree": ([e["cost"] for e in enc if e["family"] == "RealMLP"][0]
                            / min(tree_costs)),
    "ratio_vs_best_tree": ([e["cost"] for e in enc if e["family"] == "RealMLP"][0]
                           / max(tree_costs)),
}

# -------------------------------------------- 2. mechanism does not predict decorrelation
# The three arguments this repo made from architecture to disagreement, and what the
# fitted vectors actually did. Spearman recomputed, not quoted.
def rho(a, b):
    return float(pd.Series(load(a)).corr(pd.Series(load(b)), method="spearman"))


res["mechanism_claims"] = [
    {"claim": "CatBoost's ordered target statistics differ from histogram splits",
     "pair": ["catboost_te", "xgb_te"], "rho": rho("catboost_te", "xgb_te")},
    {"claim": "a linear model among gradient boosted trees",
     "pair": ["logit_te_fe", "xgb_te"], "rho": rho("logit_te_fe", "xgb_te")},
    {"claim": "TabM trains its members jointly, RealMLP averages independent ones",
     "pair": ["tabm", "realmlp10"], "rho": rho("tabm", "realmlp10")},
]
# The scale the band has to be read against: two seeds of one model, and the
# within-family range this repo measured in notebook 07.
res["rho_scale"] = {
    "two_seeds_one_model": rho("xgb_te", "xgb_te_seed2024"),
    "within_family_lo": 0.9741, "within_family_hi": 0.9981,
}

# ------------------------------------------------------------- 3. the saturation curve
# Gate gains as differences of cv_mean between the arm and the base it was paired
# against. Exact, because both arms are scored on the same five folds.
# Bases verified against each row's own notes field, not assumed. Rows 137 and 140
# are PRUNES rather than member gates and are held separately: their ledger figures
# of +0.000083 and +0.000023 are a member gate PLUS a prune, measured against rows
# 134 and 136, so quoting them beside pure member gates overstates the member.
GATES = [("row 77", "77", "59", "five no-encoder and native views"),
         ("row 94", "94", "77", "six ratio-block and fitted-budget members"),
         ("row 122", "122", "94", "seven members, four new families"),
         ("row 126", "126", "122", "the best single model in the repo"),
         ("row 134", "134", "126", "three neural variants"),
         ("row 136", "136", "134", "RealMLP, a new architecture class"),
         ("row 139", "139", "136", "RealMLP at ten internal members"),
         ("row 142", "142", "139", "RealMLP with the encoder off"),
         ("row 144", "144", "142", "TabM, a second new architecture class")]
PRUNES = [("row 137", "137", "134", "the row 136 gate plus a 25-member prune"),
          ("row 140", "140", "136", "the row 139 gate plus a 25-member prune")]
curve = []
for label, arm, base, what in GATES:
    g = float(ledger[arm]["cv_mean"]) - float(ledger[base]["cv_mean"])
    curve.append({"gate": label, "offered": what, "gain": g,
                  "base": base, "cv": float(ledger[arm]["cv_mean"])})
res["saturation_curve"] = curve
res["prunes"] = [{"row": l, "against": b, "what": w,
                  "gain": float(ledger[a]["cv_mean"]) - float(ledger[b]["cv_mean"])}
                 for l, a, b, w in PRUNES]

# --------------------------------------------------------------- 4. what a gain is worth
# The rank table. This is the analysis the repo never ran and it is the reason the
# curve above matters less than it looks.
lb = sorted(REPO.glob("artifacts/lb/*publicleaderboard*.csv"))
if lb:
    scores = np.sort(pd.read_csv(lb[-1])["Score"].to_numpy())[::-1]
    OURS = 0.97021
    our_rank = int((scores > OURS).sum()) + 1
    res["rank_table"] = {
        "teams": int(len(scores)), "our_score": OURS, "our_rank": our_rank,
        "top": float(scores[0]),
        "at": [{"rank": r, "score": float(scores[r - 1])} for r in (1, 10, 100, 250, 500)],
        "rows": [{"gain": g, "score": OURS + g,
                  "rank": int((scores > OURS + g).sum()) + 1,
                  "places": our_rank - (int((scores > OURS + g).sum()) + 1)}
                 for g in (0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.0008, 0.0011)],
        "band_top100": float(scores[0] - scores[99]),
    }

# ------------------------------------------------------- 5. the members, as they finished
NEURALS = ["neural", "neural_te", "neural_fe", "neural_wide", "neural_res",
           "neural_lookup", "realmlp", "realmlp10", "realmlp_raw_fe", "tabm"]
res["neural_line"] = [{"name": n, "cv": cv(n)[0]} for n in NEURALS]
res["best_singles"] = [{"name": n, "cv": cv(n)[0]}
                       for n in ("xgb_tuned", "cat_te_fe", "realmlp10", "tabm", "xgb_te_fe")]

OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(f"wrote {OUT.name}")

print("\nthe encoder, by model class:")
for e in res["encoder_by_class"]:
    print(f"  {e['family']:9} {e['cv_on']:.6f} -> {e['cv_off']:.6f}   {e['cost']:+.6f}")
s = res["encoder_summary"]
print(f"  the network pays {s['ratio_vs_best_tree']:.1f}x to {s['ratio_vs_worst_tree']:.1f}x "
      f"what a tree pays")

print("\nmechanism did not predict decorrelation:")
for m in res["mechanism_claims"]:
    print(f"  rho {m['rho']:.4f}   {m['claim'][:62]}")
print(f"  for scale, two seeds of one model: {res['rho_scale']['two_seeds_one_model']:.4f}")

print("\nthe saturation curve, member gates only:")
for c in res["saturation_curve"]:
    print(f"  {c['gate']:9} {c['gain']:+.6f}   {c['offered']}")

if "rank_table" in res:
    rt = res["rank_table"]
    print(f"\nrank: {rt['our_rank']} of {rt['teams']}, top {rt['top']:.5f}, "
          f"top 100 spans {rt['band_top100']:.5f}")
    for r in rt["rows"]:
        print(f"  +{r['gain']:.5f} -> rank {r['rank']:4d}  ({r['places']:+d} places)")

print("  the two prunes, held separate because each is a member gate PLUS a prune:")
for c in res["prunes"]:
    print(f"  {c['row']:9} {c['gain']:+.6f}  (vs row {c['against']:>3})  {c['what']}")

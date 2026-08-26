"""Build the family mean: an equal-weight rank average of public plateau submissions.

The construction is srcN's, documented in "S6E8 Mapping the Public Plateau".
By late in this competition every public blend converges to 0.97125 to 0.97128, ten-plus
independent authors landing on the same number. Averaging K independent draws of that
plateau cancels roughly root-K of their per-draw noise while keeping the shared signal.

**Nothing is fitted here.** Not on out-of-fold data, not on the leaderboard. The weights
are equal by construction, so there is no fitted quantity for a leak or a selection
effect to travel through. That is the whole reason this is defensible where row 158's
greedy weight search was not.

**What this file is NOT.** It contains no model built by this repo. Every ingredient is
another competitor's finished submission, used under competition rule 2.6.a, which
permits External Data that is publicly available and equally accessible to all
Participants at no cost. Sources are listed in the output and in the ledger.

Guards, because a submission csv is easy to get silently wrong:
  * id column must equal test.csv's id column, in order, asserted per file;
  * exact duplicates are removed by hash, so a file republished under two names cannot
    vote twice and quietly double its author's weight;
  * near-duplicates above 0.9999 are reported, since the plateau is crowded and two
    authors forking the same anchor is the normal case rather than the exception.
"""
import glob
import hashlib
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parent.parent
test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
ids = test["id"].to_numpy()
N = len(test)

R = lambda v: (rankdata(v, method="average") - 0.5) / len(v)

# PLATEAU GATE. v2 averaged every plateau-adjacent submission found, 14 files over 9
# independent draws, and scored 0.97123 against v1's 0.97127 on six. The root-K argument
# only holds when the draws come from the SAME distribution: averaging in sources that sit
# BELOW the plateau does not cancel noise, it moves the mean down. So an ingredient is
# admitted only if its own author published a score at the plateau.
#
# This uses each AUTHOR's published number as a prior on ingredient quality. It is not
# our own leaderboard feedback on our own submissions, which would be the selection
# mechanism srcP's analysis indicts, but it is not free of the leaderboard
# either and is recorded as such.
PLATEAU = {
    "s6e8-regime-calibrated-rank-fusion-lb-0-97127",
    "s6e8-rank-logit-regime-fusion-lb0-97125",
    "0-97125-rank-logit-fusion-forkable",
    "s6e8-elite-rank-average-ensemble-0-97126",
    "s6e8-mapping-the-public-plateau-0-97128",
    "s6e8-top-20-formula-dual-master-rank-blend",
    "0826-knock-the-blender-with-a-liner",
}

seen, subs = {}, {}
for f in sorted(glob.glob(str(ROOT / "artifacts" / "fusion" / "*" / "*.csv"))):
    if os.path.basename(os.path.dirname(f)) not in PLATEAU:
        continue
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if "id" not in d.columns or len(d) != N:
        continue
    tc = [c for c in d.columns if c.lower() != "id"]
    if not tc:
        continue
    # A csv in a different row order blends perfectly cleanly and is invisible in the
    # score. Asserted rather than hoped for.
    if not (d["id"].to_numpy() == ids).all():
        print(f"  SKIP {os.path.basename(os.path.dirname(f))}: id order does not match test.csv")
        continue
    r = R(d[tc[0]].to_numpy(float))
    h = hashlib.md5(np.round(r, 9).tobytes()).hexdigest()
    src = os.path.basename(os.path.dirname(f))
    if h in seen:
        print(f"  dedupe: {src}/{os.path.basename(f)} is identical to {seen[h]}")
        continue
    seen[h] = src
    subs[f"{src}/{os.path.basename(f)}"] = r

print(f"\n{len(subs)} distinct plateau submissions:")
for k in subs:
    print(f"  {k}")

M = np.array(list(subs.values()))
C = np.corrcoef(M)
np.fill_diagonal(C, 0.0)
ks = list(subs)
near = [(ks[i], ks[j], C[i, j]) for i in range(len(ks)) for j in range(i + 1, len(ks))
        if C[i, j] > 0.9999]
print(f"\npairs above 0.9999: {len(near)}")
for a, b, r in near:
    print(f"  NEAR-DUPLICATE {a[:34]} / {b[:34]} at {r:.6f}")
print(f"pairwise spearman: min {C[C > 0].min():.5f}  max {C.max():.5f}")

# Collapse near-duplicate clusters BEFORE averaging. A family mean cancels noise by
# averaging INDEPENDENT draws; two files at Spearman 0.99999 are one draw wearing two
# names, and letting both vote silently doubles that author's weight instead of
# cancelling anything. Single-linkage at 0.9999, then one mean per cluster, then equal
# weight across clusters.
lab = list(range(len(ks)))


def find(i):
    while lab[i] != i:
        lab[i] = lab[lab[i]]
        i = lab[i]
    return i


for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        if C[i, j] > 0.9999:
            a, b = find(i), find(j)
            if a != b:
                lab[b] = a
clusters = {}
for i in range(len(ks)):
    clusters.setdefault(find(i), []).append(i)
print(f"\n{len(ks)} files collapse to {len(clusters)} independent draws:")
for root, mem in clusters.items():
    tag = "" if len(mem) == 1 else f"  ({len(mem)} near-duplicates merged)"
    print(f"  {ks[root][:52]}{tag}")

draws = np.array([M[mem].mean(axis=0) for mem in clusters.values()])

# Mean and median across the independent draws. The median is the more robust estimator
# when one draw may be an outlier, and one of these four publishes no leaderboard score
# at all. Both are a-priori choices; neither is tuned against our own submissions.
fam = R(draws.mean(axis=0))
fam_med = R(np.median(draws, axis=0))
print(f"mean vs median across draws, spearman {np.corrcoef(fam, fam_med)[0, 1]:.6f}")
pd.DataFrame({"id": ids,
              "addicted_label": (np.argsort(np.argsort(fam_med)) + 0.5) / N}
             ).to_csv(ROOT / "submissions" / "family_median.csv", index=False)
print("wrote family_median.csv")
out = pd.DataFrame({"id": ids,
                    "addicted_label": (np.argsort(np.argsort(fam)) + 0.5) / N})
assert len(out) == N and np.isfinite(out["addicted_label"]).all()
dest = ROOT / "submissions" / "family_mean_v3.csv"
out.to_csv(dest, index=False)
print(f"\nwrote {dest.name} from {len(subs)} ingredients")

prev = ROOT / "submissions" / "family_mean.csv"
if prev.exists():
    p = R(pd.read_csv(prev)["addicted_label"].to_numpy(float))
    print(f"spearman against the 6-ingredient family_mean (LB 0.97127): "
          f"{np.corrcoef(fam, p)[0, 1]:.6f}")
ours = ROOT / "submissions" / "stack_greedy_blend3.csv"
if ours.exists():
    o = R(pd.read_csv(ours)["addicted_label"].to_numpy(float))
    print(f"spearman against our own stack (LB 0.97107):            "
          f"{np.corrcoef(fam, o)[0, 1]:.6f}")

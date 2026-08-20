"""Rewrite the ledger table in README.md from experiments.csv.

The table had drifted 24 rows behind the ledger by 2026-08-20, which is what a
hand-maintained copy of a file that already exists does. Run this after appending
rows rather than editing the table by hand.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Short notes for rows whose blank LB cell would otherwise lose information. A row
# that is only a component of a later stack keeps a blank, which is the existing
# convention for rows 3, 5, 6 and 7.
NOTE = {
    2: "reproducibility re-run of 1",
    16: "not submitted",
    18: "null result, not submitted",
    19: "negative, not submitted",
    25: "not submitted",
    27: "not submitted",
    33: "in the stack, not submitted alone",
    34: "not submitted",
    39: "not submitted",
    45: "discarded, not submitted",
}

rows = list(csv.DictReader((ROOT / "experiments.csv").open(encoding="utf-8")))

lines = ["| id | name | CV AUC | fold sd | public LB |", "|---|---|---|---|---|"]
for r in rows:
    i = int(r["id"])
    lb = r["lb_public"].strip() or NOTE.get(i, "")
    cell = f" {lb} " if lb else " "
    lines.append(f"| {i} | {r['name']} | {r['cv_mean']} | {r['cv_std']} |{cell}|")

table = "\n".join(lines)

readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
pat = re.compile(r"\| id \| name \| CV AUC \| fold sd \| public LB \|\n(?:\|.*\n)+")
if not pat.search(text):
    raise SystemExit("ledger table not found in README.md")
readme.write_text(pat.sub(table + "\n", text, count=1), encoding="utf-8")
print(f"table rewritten, {len(rows)} rows, last id {rows[-1]['id']}")

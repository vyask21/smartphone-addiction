"""Embed writeup_numbers.json into the `D = json.loads(...)` cell of the writeup.

Usage:  python writeup/embed.py [--check]

The public notebook has to be self-contained, because it runs on Kaggle where this
repo does not exist. So the numbers live inside it as a literal rather than being read
from disk at run time.

That created a gap this script closes. `writeup_numbers.py` regenerates the blob from
the out-of-fold vectors, and its whole purpose is that a number in the notebook and a
number in the ledger cannot drift apart. But nothing moved the regenerated blob INTO
the notebook, so the two could drift after all, silently, and the first version of that
blob was pasted by hand.

`--check` exits non-zero if the notebook is out of date with the json, which is the
form to put in front of a push.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "writeup_numbers.json"
NB = HERE.parent / "notebooks" / "12_public_writeup.ipynb"
OPEN, CLOSE = "D = json.loads(r'''", "''')"

check_only = "--check" in sys.argv[1:]

payload = json.dumps(json.loads(SRC.read_text(encoding="utf-8")),
                     separators=(",", ":"), allow_nan=False)
# The blob is pasted inside an r'''...''' literal, so it must not contain the
# terminator or a trailing backslash. JSON escapes backslashes, so this is a
# belt-and-braces check rather than a likely failure.
assert "'''" not in payload and not payload.endswith("\\"), "blob is not embeddable"

nb = json.loads(NB.read_text(encoding="utf-8"))
hits = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if OPEN in src:
        hits.append((cell, src))

if len(hits) != 1:
    sys.exit(f"expected exactly one '{OPEN}' cell, found {len(hits)}")

cell, src = hits[0]
head, rest = src.split(OPEN, 1)
old, tail = rest.rsplit(CLOSE, 1)

if old == payload:
    print(f"up to date: {len(payload):,} chars, {len(json.loads(payload))} top-level keys")
    sys.exit(0)

if check_only:
    sys.exit(f"OUT OF DATE: notebook blob is {len(old):,} chars, json is "
             f"{len(payload):,}. Run `python writeup/embed.py`.")

new_src = head + OPEN + payload + CLOSE + tail
compile(new_src, "cell", "exec")

# Read the blob back out of the finished cell and parse it, which is the thing the
# notebook will actually do. Comparing `payload` to itself would prove nothing.
back = new_src.split(OPEN, 1)[1].rsplit(CLOSE, 1)[0]
assert json.loads(back) == json.loads(SRC.read_text(encoding="utf-8")), "round trip"

cell["source"] = [l + "\n" for l in new_src.split("\n")]
if cell["source"] and cell["source"][-1] == "\n":
    cell["source"].pop()
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"embedded {len(payload):,} chars ({len(json.loads(payload))} top-level keys), "
      f"was {len(old):,}")

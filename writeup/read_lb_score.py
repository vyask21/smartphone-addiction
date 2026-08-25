"""Read a submission's public score, safely.

Exists because of a real incident. Row 158 was logged as the worst submission of the
competition, LB 0.97006, with a detailed analysis of why it failed. The true score was
0.97106, a personal best. The readback command was

    kaggle competitions submissions ... | sed -n '3p' | grep -o -E "0\\.97[0-9]{3}" | head -1

and the submissions line carries the submission DESCRIPTION as well as the score. The
description read "Blended OOF 0.970068", so the regex matched 0.97006 out of text I had
written myself and never read the public score at all.

**Never scrape a number from a line that also carries text you wrote.**

This parses by column position, matches on filename, and refuses to guess:

    python writeup/read_lb_score.py stack_greedy_blend.csv
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"
COMP = "playground-series-s6e8"


def fetch():
    r = subprocess.run([str(KAGGLE), "competitions", "submissions", COMP],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        sys.exit(f"kaggle CLI failed: {r.stderr.strip()[:300]}")
    return r.stdout.splitlines()


def parse(lines):
    """(fileName, status, publicScore) per row, taken from the END of the line.

    The score columns are the last two fields; the description sits in the middle and is
    the thing that must never be matched against.
    """
    out = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        fn = re.search(r"\s(\S+\.csv)\s", ln)
        st = re.search(r"SubmissionStatus\.(\w+)", ln)
        if not (fn and st):
            continue
        tail = ln[st.end():]                       # everything AFTER the status column
        sc = re.findall(r"\d\.\d{4,6}", tail)      # scores only live here
        out.append((fn.group(1), st.group(1), sc[0] if sc else None))
    return out


if __name__ == "__main__":
    rows = parse(fetch())
    if len(sys.argv) > 1:
        want = sys.argv[1]
        hits = [r for r in rows if r[0] == want]
        if not hits:
            sys.exit(f"no submission named {want}. Seen: {[r[0] for r in rows[:8]]}")
        fn, st, sc = hits[0]
        if st != "COMPLETE":
            sys.exit(f"{fn} is {st}, no score yet")
        assert sc is not None, f"{fn} is COMPLETE but no score parsed"
        print(f"{fn}  {sc}")
    else:
        for fn, st, sc in rows[:15]:
            print(f"  {fn:38} {st:9} {sc or '-'}")

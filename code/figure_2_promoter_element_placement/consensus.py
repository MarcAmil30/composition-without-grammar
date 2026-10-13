# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import csv
from collections import Counter
import numpy as np

BASE = f"{ICBINB_ROOT}"
UPO  = f"{BASE}/data/benchmark/upstream_only"
L_ROWS = list(csv.DictReader(open(f"{UPO}/prompts_681_leaked.csv")))
seq = {r["locus_tag"]: r["prompt_promoter"] + r["target_promoter"] for r in L_ROWS}
sigma = [r["sigma"] or "unassigned" for r in L_ROWS]

CORE, FLANK = 6, 3
W = CORE + 2*FLANK
BASES = "ACGT"

def w35(r):
    s = seq[r["locus_tag"]]; i = int(r["m35_start"]); a,b = i-FLANK, i-FLANK+W
    return s[a:b] if a >= 0 and b <= len(s) else None
def w10(r):
    s = seq[r["locus_tag"]]; j = int(r["m10_end"]); a,b = j-CORE-FLANK, j-CORE-FLANK+W
    return s[a:b] if a >= 0 and b <= len(s) else None

groups = [(s,n) for s,n in Counter(sigma).most_common() if n >= 25]
for sig, n in groups:
    rows = [r for r,s in zip(L_ROWS, sigma) if s == sig]
    print(f"\n=== {sig}  (n={n}) ===")
    for lab, wf in [("distal (-35)", w35), ("proximal (-10)", w10)]:
        seqs = [w for w in (wf(r) for r in rows) if w and len(w)==W]
        A = np.array([[BASES.index(c) for c in s] for s in seqs if set(s) <= set(BASES)])
        cons, tops, bits = [], [], []
        for p in range(W):
            cnt = np.bincount(A[:,p], minlength=4)/len(A)
            k = cnt.argmax(); cons.append(BASES[k]); tops.append(cnt[k])
            bits.append(2 + (cnt[cnt>0]*np.log2(cnt[cnt>0])).sum())
        tag = lambda i: "[" if i==FLANK else ("]" if i==FLANK+CORE else "")
        s_cons = "".join(("|" if p==FLANK else "") + c.lower() if p<FLANK or p>=FLANK+CORE
                          else c for p,c in enumerate(cons))
        print(f"  {lab:16s} n={len(A)}")
        print(f"    consensus  : {''.join(cons[:FLANK]).lower()}|{''.join(cons[FLANK:FLANK+CORE])}|{''.join(cons[FLANK+CORE:]).lower()}")
        print(f"    freq top   : " + " ".join(f"{t*100:3.0f}" for t in tops))
        print(f"    bits       : " + " ".join(f"{b:3.1f}" for b in bits))
        print(f"    core bits total = {sum(bits[FLANK:FLANK+CORE]):.2f}")
        # full per-position freq of core
        for p in range(W):
            cnt = np.bincount(A[:,p], minlength=4)/len(A)
            mark = "core" if FLANK <= p < FLANK+CORE else "flnk"
            print(f"      pos {p-FLANK+1:+d} {mark} " + " ".join(f"{BASES[k]}:{cnt[k]*100:4.1f}" for k in range(4)))

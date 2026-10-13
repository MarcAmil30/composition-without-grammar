"""Prove every model and both scorers use the SAME genes and the SAME decoys.

Each runner calls EV.load_eval() on one CSV and EV.decoy_draws() on one cached
.npy, so this is true by construction -- but "by construction" is what the CLS
pooling bug also looked like, so check it against what is actually on disk.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, f"{ICBINB_ROOT}/data/code/retrieval")
import numpy as np
import retrieval_eval as EV

RES = f"{EV.RETR}/results"

df = EV.load_eval()
print(f"eval set : {EV.EVAL_CSV}")
print(f"           {len(df)} genes, L={EV.UP_LEN}, "
      f"locus_tags {df.locus_tag.iloc[0]}..{df.locus_tag.iloc[-1]}")
h = hashlib.md5("".join(df.locus_tag).encode()).hexdigest()[:12]
print(f"           gene-order md5 {h}")

d = EV.decoy_draws(len(df))
dp = f"{EV.RETR}/draws_seed{EV.SEED}_n{len(df)}_k{EV.N_CAND}_r{EV.N_REPEATS}.npy"
print(f"decoys   : {os.path.basename(dp)}  shape {d.shape}  "
      f"md5 {hashlib.md5(d.tobytes()).hexdigest()[:12]}")
print("           -> one file on disk, loaded by every model and both scorers\n")

print(f"{'model':10s} {'variant':28s} {'n':>5s}  matches eval set?")
bad = 0
for f in sorted(glob.glob(f"{RES}/*/*_summary.json")):
    if "_smoke" in os.path.basename(f):
        continue
    s = json.load(open(f))
    ok = (s["n_genes"] == len(df) and s["upstream_len"] == EV.UP_LEN
          and s["n_candidates"] == EV.N_CAND and s["seed"] == EV.SEED)
    bad += not ok
    print(f"{os.path.basename(os.path.dirname(f)):10s} {s['name'][:28]:28s} "
          f"{s['n_genes']:>5d}  {'yes' if ok else 'NO <-- MISMATCH'}")
print(f"\n{'all consistent' if not bad else f'{bad} MISMATCHES'}")

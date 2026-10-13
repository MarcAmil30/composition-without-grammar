"""Diagnose the flat smoke-test result on the cached 120x120 score matrix.

Hypothesis: S[i,j] = log P(up_j | CDS_i) is dominated by the INTRINSIC likelihood
of upstream j -- a per-candidate term that is the same for every query i. Ranking
within a row then mostly sorts candidates by "which upstream is generically most
likely", and the CDS-specific interaction is buried under it.

If so, removing the column mean (the per-candidate bias) should expose the
interaction. This is the likelihood-side analogue of centering the embeddings,
and it costs nothing -- the matrix is already computed.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import sys
sys.path.insert(0, f"{ICBINB_ROOT}/data/code/retrieval")
import numpy as np
import retrieval_eval as EV

OUT = f"{ICBINB_ROOT}/data/benchmark/retrieval/results/prok"
S = np.load(f"{OUT}/score_cond_smoke_matrix.npy")
n = S.shape[0]
d = EV.decoy_draws(n, cache=False)
print(f"matrix {S.shape}\n")

# How much of the variance is per-candidate (column) vs per-query (row)?
col = S.mean(0)          # intrinsic likelihood of each upstream
row = S.mean(1)          # how "generous" each CDS context is
tot = S.var()
print("variance decomposition of log P(up_j | CDS_i)")
print(f"  between-candidate (column) {col.var()/tot*100:6.1f}%   <- same for every query")
print(f"  between-query     (row)    {row.var()/tot*100:6.1f}%   <- same for every candidate")
print(f"  interaction (the signal)   {(S - col[None,:] - row[:,None] + S.mean()).var()/tot*100:6.1f}%\n")

variants = {
    "raw                       ": S,
    "column-centered (PMI-like)": S - col[None, :],
    "row-centered              ": S - row[:, None],
    "double-centered           ": S - col[None, :] - row[:, None] + S.mean(),
    "column z-scored           ": (S - col[None, :]) / S.std(0, keepdims=True).clip(1e-8),
}
for name, Mx in variants.items():
    r = EV.ranks_from_matrix(Mx, True, d)
    g = EV.true_vs_decoy_gap(Mx, True, d)
    print(f"{name}  top1 {(r==1).mean()*100:6.2f}%  top5 {(r<=5).mean()*100:6.2f}%  "
          f"median {np.median(r):5.1f}  gap {g['gap_mean']:+.4f}  "
          f"true>decoy {g['frac_true_higher']*100:5.1f}%  p={g['wilcoxon_p_greater']:.1e}")

print("\nchance: top1 1.00%  top5 5.00%  median 50.5")

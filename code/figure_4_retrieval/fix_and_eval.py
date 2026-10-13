"""Fix NaN handling in the normalisations, then evaluate every saved matrix.

The 2 skipped query rows are NaN. `normalize` used M.mean(0) -- a column mean
taken over rows that include the NaN rows, which makes EVERY column mean NaN and
poisons the whole matrix. Same for the z-score and the variance decomposition.
All of them must be nan-aware.

Evaluation is pure numpy, so this runs on the login node; no GPU job needed.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import glob
import json
import os
import pathlib
import sys

import numpy as np

R = f"{ICBINB_ROOT}/data/code/retrieval"
RES = f"{ICBINB_ROOT}/data/benchmark/retrieval/results"

# ---------------------------------------------------------------- patch
p = pathlib.Path(f"{R}/retrieval_eval.py")
s = p.read_text()
if "nanmean" not in s:
    s = s.replace('    if how == "pmi":\n        return M - M.mean(0, keepdims=True)',
                  '    if how == "pmi":\n        return M - np.nanmean(M, 0, keepdims=True)')
    s = s.replace('    if how == "zcol":\n        return (M - M.mean(0, keepdims=True)) / M.std(0, keepdims=True).clip(1e-8)',
                  '    if how == "zcol":\n        return ((M - np.nanmean(M, 0, keepdims=True))\n'
                  '                / np.nanstd(M, 0, keepdims=True).clip(1e-8))')
    s = s.replace('    if how == "double":\n        return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()',
                  '    if how == "double":\n        return (M - np.nanmean(M, 0, keepdims=True)\n'
                  '                - np.nanmean(M, 1, keepdims=True) + np.nanmean(M))')
    s = s.replace('    col, row, tot = M.mean(0), M.mean(1), M.var()\n    inter = M - col[None, :] - row[:, None] + M.mean()',
                  '    col, row, tot = np.nanmean(M, 0), np.nanmean(M, 1), np.nanvar(M)\n'
                  '    inter = M - col[None, :] - row[:, None] + np.nanmean(M)')
    s = s.replace('    return {"between_candidate_frac": float(col.var() / tot),\n'
                  '            "between_query_frac": float(row.var() / tot),\n'
                  '            "interaction_frac": float(inter.var() / tot)}',
                  '    return {"between_candidate_frac": float(np.nanvar(col) / tot),\n'
                  '            "between_query_frac": float(np.nanvar(row) / tot),\n'
                  '            "interaction_frac": float(np.nanvar(inter) / tot)}')
    p.write_text(s)
    print("retrieval_eval.py: normalisations are now nan-aware")

sys.path.insert(0, R)
import retrieval_eval as EV  # noqa: E402

# ---------------------------------------------------------------- evaluate
MODELS = {"prok": "log P(upstream | CDS)", "evo1": "log P(CDS | upstream)"}
for model, quantity in MODELS.items():
    out = f"{RES}/{model}"
    for base in ("score_cond", "score_full"):
        f = f"{out}/{base}_matrix.npy"
        if not os.path.exists(f):
            print(f"[skip] {f} missing")
            continue
        S = np.load(f)
        bad = int(np.isnan(S).any(axis=1).sum())
        print(f"\n### {model} {base}  ({S.shape[0]} rows, {bad} unscorable)")
        if base == "score_cond":
            vd = EV.variance_decomposition(S)
            print(f"  variance: candidate {vd['between_candidate_frac']*100:.1f}% | "
                  f"query {vd['between_query_frac']*100:.1f}% | "
                  f"interaction {vd['interaction_frac']*100:.1f}%")
        for how in ("raw", "pmi", "zcol"):
            sm = EV.evaluate(EV.normalize(S, how), f"{base}_{how}", out, True,
                             extra={"model": model, "scorer": "scoring",
                                    "normalization": how, "quantity": quantity},
                             save_matrix=False)
            EV.report(sm)

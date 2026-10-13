"""Skip query rows that StripedHyena cannot score at any batch size.

The FFT prefill needs >82 GiB at L~4767 even at batch 1 on a freshly rebuilt
model; the longest row that ever succeeded was L=4229. Rather than keep tuning
batch/allocator knobs, such genes are dropped as QUERIES and kept as CANDIDATES,
so all 500 candidates and the shared decoy draws stay identical across models.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import pathlib

R = f"{ICBINB_ROOT}/data/code/retrieval"

# ---------------- lm_retrieval.py ----------------
p = pathlib.Path(f"{R}/lm_retrieval.py")
s = p.read_text()

s = s.replace(
    'ap.add_argument("--long_seq", type=int, default=3000,',
    'ap.add_argument("--max_len", type=int, default=4300,\n'
    '                help="skip query rows longer than this. StripedHyena needs "\n'
    '                     ">82 GiB at L~4767 even at batch 1 on a clean allocator; "\n'
    '                     "longest row that ever worked was L=4229. Those genes are "\n'
    '                     "dropped as QUERIES but kept as candidates.")\n'
    'ap.add_argument("--long_seq", type=int, default=3000,')

old = ('        if L > args.long_seq:\n'
       '            width = 1\n'
       '            R.rebuild(f"long row L={L}")')
new = ('        if L > args.max_len:\n'
       '            # Unscorable at any batch size. Record NaN and continue rather\n'
       '            # than killing the run; excluded as a query, kept as a candidate.\n'
       '            S_c[i], S_f[i], done[i] = np.nan, np.nan, True\n'
       '            status(f"SKIP row {i} (L={L} > {args.max_len}) -- unscorable")\n'
       '            continue\n'
       '        if L > args.long_seq:\n'
       '            width = 1\n'
       '            R.rebuild(f"long row L={L}")')
assert old in s, "long_seq anchor missing"
s = s.replace(old, new)

s = s.replace(
    '    assert not np.isnan(S_c).any(), "score matrix has holes"',
    '    nbad = int(np.isnan(S_c).any(axis=1).sum())\n'
    '    assert nbad < 0.05 * n, f"{nbad}/{n} query rows unscorable -- too many"\n'
    '    if nbad:\n'
    '        status(f"{nbad}/{n} query rows skipped as unscorable (see SKIP lines)")')

# guard 3 must pick rows that were actually scored
s = s.replace('    for i in range(min(3, n)):\n        if CF["cds_first"]:',
              '    for i in [k for k in range(n) if not np.isnan(S_c[k]).any()][:3]:\n'
              '        if CF["cds_first"]:')
p.write_text(s)
print("lm_retrieval.py patched")

# ---------------- retrieval_eval.py ----------------
p = pathlib.Path(f"{R}/retrieval_eval.py")
s = p.read_text()

s = s.replace('''    os.makedirs(out_dir, exist_ok=True)
    n = M.shape[0]
    draws = decoy_draws(n)
    ranks = ranks_from_matrix(M, higher_is_better, draws)''',
'''    os.makedirs(out_dir, exist_ok=True)
    n = M.shape[0]
    draws = decoy_draws(n)
    # A query row may be unscorable (lm_retrieval --max_len). Such a gene is
    # dropped as a QUERY but stays in the candidate pool, so every model still
    # ranks against the same 500 candidates with the same decoy draws.
    valid = ~np.isnan(M).any(axis=1)
    Mw = np.nan_to_num(M, nan=-1e30, posinf=-1e30, neginf=-1e30)
    ranks = ranks_from_matrix(Mw, higher_is_better, draws)[valid]''')

s = s.replace('    summary["variance_decomposition"] = variance_decomposition(M)',
              '    summary["variance_decomposition"] = variance_decomposition(Mw[valid])\n'
              '    summary["n_query_rows_excluded"] = int((~valid).sum())')

s = s.replace('    summary["guard_true_vs_decoy"] = true_vs_decoy_gap(M, higher_is_better, draws)',
              '    summary["guard_true_vs_decoy"] = true_vs_decoy_gap(Mw, higher_is_better, draws, valid)')

s = s.replace('''    summary["top1_reversed_direction"] = float(
        (ranks_from_matrix(M, not higher_is_better, draws) == 1).mean())''',
'''    summary["top1_reversed_direction"] = float(
        (ranks_from_matrix(Mw, not higher_is_better, draws)[valid] == 1).mean())''')

s = s.replace('def true_vs_decoy_gap(M, higher_is_better=True, draws=None):',
              'def true_vs_decoy_gap(M, higher_is_better=True, draws=None, valid=None):')

s = s.replace('''    d = true - dec
    stat, p = wilcoxon(d, alternative="greater")''',
'''    d = true - dec
    if valid is not None:
        d, true, dec = d[valid], true[valid], dec[valid]
    stat, p = wilcoxon(d, alternative="greater")''')
p.write_text(s)
print("retrieval_eval.py patched")

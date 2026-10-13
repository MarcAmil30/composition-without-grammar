"""Calibration: a pure-noise matrix must give top-1 = 1% and median rank = 50.

Any systematic deviation here is an off-by-one in the ranking, and would fake a
signal (or hide one) in every model. Cheap to check, expensive to miss.
"""
import numpy as np
import retrieval_eval as EV

n = 500
d = EV.decoy_draws(n, cache=False)
t1, md, mrr = [], [], []
for s in range(10):
    rng = np.random.default_rng(100 + s)
    r = EV.ranks_from_matrix(rng.normal(size=(n, n)).astype(np.float32), True, d)
    t1.append((r == 1).mean()); md.append(np.median(r)); mrr.append((1 / r).mean())
print(f"noise x10 draws, n={n}")
print(f"  top-1   {np.mean(t1)*100:.3f}%  (expect 1.000)  sd {np.std(t1)*100:.3f}")
print(f"  median  {np.mean(md):.2f}       (expect 50.5)")
print(f"  MRR     {np.mean(mrr):.4f}      (expect {sum(1/k for k in range(1,101))/100:.4f})")

# CI coverage: does the bootstrap CI contain 1% about 95% of the time?
hits = 0
for s in range(40):
    rng = np.random.default_rng(500 + s)
    r = EV.ranks_from_matrix(rng.normal(size=(n, n)).astype(np.float32), True, d)
    lo, hi = EV.metrics(r, n_boot=400)["top1_ci95"]
    hits += lo <= 0.01 <= hi
print(f"  bootstrap CI covers 1% in {hits}/40 draws (expect ~38)")

# a planted signal must be recovered
rng = np.random.default_rng(7)
for delta in (0.0, 0.5, 1.0, 2.0):
    M = rng.normal(size=(n, n)).astype(np.float32) + delta * np.eye(n, dtype=np.float32)
    r = EV.ranks_from_matrix(M, True, d)
    g = EV.true_vs_decoy_gap(M, True, d)
    print(f"  delta={delta:.1f} -> top1 {(r==1).mean()*100:6.2f}%  "
          f"median {np.median(r):5.1f}  guard gap {g['gap_mean']:+.3f}")

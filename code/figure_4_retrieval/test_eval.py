import numpy as np
import retrieval_eval as EV

df = EV.load_eval(limit=150)
n = len(df)
print(f"loaded {n} genes; all up200 == 200 bp")

d = EV.decoy_draws(n, cache=False)
print("draws", d.shape,
      "| self never a decoy:", all(i not in d[i] for i in range(n)),
      "| no dups within a repeat:", all(len(set(d[i, r])) == 99 for i in range(n) for r in range(20)))

rng = np.random.default_rng(1)
for nm, Mx in [("perfect", np.eye(n, dtype=np.float32)),
               ("random ", rng.normal(size=(n, n)).astype(np.float32))]:
    r = EV.ranks_from_matrix(Mx, True, d)
    m = EV.metrics(r)
    print(f"  {nm}: top1={m['top1']:.3f} median={m['median_rank']:.0f} mrr={m['mrr']:.3f}")

# a matrix with a weak but real signal -> should show a positive guard gap
weak = rng.normal(size=(n, n)).astype(np.float32) + 0.35 * np.eye(n, dtype=np.float32)
g = EV.true_vs_decoy_gap(weak, True, d)
print(f"  weak signal: gap={g['gap_mean']:+.3f} frac_true_higher={g['frac_true_higher']:.2f}")

# framing
cds, up = df.cds_seq.iloc[0], df.up200.iloc[0]
prompt = "+#" + cds + "+$"
full = prompt + up
print(f"framing: prompt {len(prompt)} chars, full {len(full)}, "
      f"ends with upstream: {full.endswith(up)}, "
      f"upstream slice matches: {full[len(prompt):] == up}")
print("first 30:", full[:30])
print("junction :", full[len(prompt) - 4:len(prompt) + 6])

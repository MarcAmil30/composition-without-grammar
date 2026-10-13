"""
Draw the fixed evaluation set for Task 4 (CDS<->upstream retrieval).

L is fixed at 200 bp for every model and both scorers, so every gene must have a
FULL 200 bp of real upstream. The 1453 population was built with an `up_len>=100`
floor, and only 798 of those clear 200 -- the rest hit the neighbouring gene
first. Eligible pool = those 798.

One seeded shuffle of the 798; prefixes give nested sets:
  first 150 = pilot   (subset of eval)
  first 500 = eval    (the set both scorers use)

Uniform draw -- deliberately NOT seeded to include the old n=150 Evo-1
subsample, which came from the 727 (down_len>=100) set and would skew this set
toward genes with a long downstream flank.

CAVEAT, carried into the README: requiring a clean 200 bp selects genes in
sparser genomic neighbourhoods. If isolated genes have more distinctive
promoters, retrieval is measured on a mildly favourable subset. It is the same
subset for every model, so model comparisons are unaffected; absolute numbers
should be read with this in mind.

Out: data/benchmark/retrieval/ecoli_retrieval_eval500.csv
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import json

import numpy as np
import pandas as pd

OUT = f"{ICBINB_ROOT}/data/benchmark/retrieval"
SEED = 0
N_EVAL, N_PILOT = 500, 150
UP_LEN = 200

full = pd.read_csv(f"{OUT}/ecoli_retrieval_1453.csv")
pool = full[full.up_len >= UP_LEN].reset_index(drop=True)
assert len(pool) >= N_EVAL, f"only {len(pool)} genes clear {UP_LEN} bp"
order = np.random.default_rng(SEED).permutation(len(pool))

ev = pool.iloc[order[:N_EVAL]].copy()
ev.insert(0, "eval_rank", np.arange(N_EVAL))          # position in the shuffle
ev.insert(1, "in_pilot150", ev.eval_rank < N_PILOT)
ev = ev.sort_values("eval_rank").reset_index(drop=True)

ev.to_csv(f"{OUT}/ecoli_retrieval_eval500.csv", index=False)

summary = {
    "n_eval": N_EVAL, "n_pilot": N_PILOT, "seed": SEED,
    "upstream_len": UP_LEN,
    "eligible_pool": int(len(pool)),
    "population": int(len(full)),
    "drawn_from": f"ecoli_retrieval_1453.csv filtered to up_len>={UP_LEN} "
                  f"({len(pool)} genes), uniform, no stratification",
    "nested": f"pilot150 subset of eval500 subset of pool{len(pool)}",
    "pilot_subset_ok": bool(ev[ev.in_pilot150].eval_rank.max() < N_PILOT),
    "all_have_full_window": bool((ev.up_len >= UP_LEN).all()),
    "n_in_cds_prompted": int(ev.in_cds_prompted.sum()),
    "n_in_c3p_727": int(ev.in_c3p_727.sum()),
    "bench_tier": ev.bench_tier.fillna("none").value_counts().to_dict(),
    "up_len": {"min": int(ev.up_len.min()), "median": float(ev.up_len.median())},
    "cds_len": {"min": int(ev.cds_len.min()), "median": float(ev.cds_len.median()),
                "max": int(ev.cds_len.max())},
    "max_scored_seq_len": int(4 + ev.cds_len.max() + UP_LEN),
    "caveat": "clean-200bp requirement selects genes in sparser neighbourhoods; "
              "same subset for every model, so comparisons are unaffected",
}
with open(f"{OUT}/ecoli_retrieval_eval500_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Panel (c) replacement: sequence identity at the true -10 position, sigma70.

No PWM scan and no argmax, so this measurement CANNOT produce a false positive.
For every generation we read the six bases sitting exactly where that promoter's
own -10 box belongs, and count how many differ from the real hexamer.

    real box       T A T A A T
    generation     T A G A C T   ->  2 mismatches of 6

Three references frame the models:

    own box            0 by construction -- it is the answer, so it is not a
                       usable positive control, which is why the next row exists
    another real box   a different sigma70 promoter's -10 hexamer scored against
                       this one: what a genuine, degenerate -10 box looks like
    shuffle            the model's own 45 bases in random order, read at the same
                       position (the matched negative control)

Run on Isambard (the generations live there):

    source ${CONDA_ROOT}/bin/activate
    python3 hamming_at_true.py

Writes panelC_hamming_v2.csv next to itself; copy that back to fig2x2/ and
make_figure.py can plot it without touching Isambard again.
"""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
CONDA_ROOT = _os.environ.get("CONDA_ROOT", _os.environ.get("CONDA_PREFIX", ""))
# -------------------------------------------------------------------

import glob
import os
import random

import numpy as np
import pandas as pd

B = os.environ.get(
    "BENCH_DIR",
    f"{ICBINB_ROOT}/data/benchmark/upstream_only")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(_ROOT, "results", "upstream_only", "fig2x2")   # panel CSVs live here
CORE = 6                      # -10 box width
SEED = 0
N_PAIR = 20                   # repeats for the "another real box" reference
N_BOOT = 2000
# v2: all five models scored in one pass, so every row here comes from the
# same code path. The four published models reproduce their generated
# values exactly; only their shuffled controls move (~0.3pp), because the
# shuffle RNG stream depends on how many models are in the pass.
MODELS = ["evo1", "evo1_131k", "evo2", "hyenadna", "promogen2"]


def hamming(a: str, b: str):
    """Mismatches between two equal-length strings, or None if not comparable."""
    if len(a) != len(b) or len(a) != CORE:
        return None
    return sum(x != y for x, y in zip(a, b))


def shuffled(seq: str, rng: random.Random) -> str:
    ch = list(seq)
    rng.shuffle(ch)
    return "".join(ch)


def boot_ci(per_gene: np.ndarray, rng: np.random.Generator):
    """Bootstrap over genes -- the 20 decodes within a gene are not independent."""
    if len(per_gene) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(per_gene), size=(N_BOOT, len(per_gene)))
    means = per_gene[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- truth
truth = pd.read_csv(os.path.join(B, "prompts_681_leaked.csv"), keep_default_na=False)
truth = truth[truth.sigma == "sigma70"].copy()
truth["true_off"] = truth.m10_offset_in_target.astype(int)
truth["true_hex"] = [
    r.upstream_seq[max(0, int(r.m10_end) - CORE):int(r.m10_end)] for r in truth.itertuples()
]
truth = truth[truth.true_hex.str.len() == CORE].set_index("locus_tag")
print(f"sigma70 promoters with a usable -10 hexamer: {len(truth)}")

# ---- reference: a DIFFERENT real sigma70 -10 box scored against this one -----
# Pair by INDEX, not by sequence: two different genes often share the same
# hexamer, and dropping those pairs would silently discard every 0-mismatch
# comparison and bias the reference downwards.
rng = random.Random(SEED)
hexes = truth.true_hex.tolist()
idx = list(range(len(hexes)))
other = []
for _ in range(N_PAIR):
    perm = idx[:]
    rng.shuffle(perm)
    other += [hamming(hexes[i], hexes[j]) for i, j in zip(idx, perm) if i != j]
other = np.array([o for o in other if o is not None], dtype=float)
print(f"another real sigma70 -10 box vs this one: {other.mean():.2f} mismatches "
      f"of {CORE}  (n={len(other)} pairings)")
print(f"own box: 0.00 by construction\n")

# ---------------------------------------------------------------- generations
files = sorted(glob.glob(os.path.join(B, "generations", "*.csv*")))
if not files:
    files = sorted(glob.glob(os.path.join(B, "generations", "**", "*.csv*"), recursive=True))
print(f"generation files found: {len(files)}")

rows = []
rng_shuf = random.Random(SEED)
for path in files:
    gen = pd.read_csv(path, keep_default_na=False)
    if gen.empty or "generation" not in gen.columns:
        continue
    gen = gen[(gen.task == "generate_promoter") & (gen.background == "leaked")]
    gen = gen[gen.model.isin(MODELS)]
    for r in gen.itertuples():
        if r.locus_tag not in truth.index:
            continue
        t = truth.loc[r.locus_tag]
        g = r.generation or ""
        off = int(t["true_off"])
        if off < 0 or off + CORE > len(g):
            continue                       # box would fall outside the generation
        h_gen = hamming(g[off:off + CORE], t["true_hex"])
        h_shf = hamming(shuffled(g, rng_shuf)[off:off + CORE], t["true_hex"])
        if h_gen is None or h_shf is None:
            continue
        rows.append((r.model, r.locus_tag, h_gen, h_shf))

d = pd.DataFrame(rows, columns=["model", "locus_tag", "generated", "shuffled"])
print(f"scored generations: {len(d)}\n")

# aggregate within gene first, then across genes
boot = np.random.default_rng(SEED)
out = []
print(f"{'model':11} {'generated':>22} {'shuffled':>22}")
for m in MODELS:
    sub = d[d.model == m]
    if sub.empty:
        print(f"{m:11}   (no rows)")
        continue
    line = f"{m:11}"
    for kind in ("generated", "shuffled"):
        per_gene = sub.groupby("locus_tag")[kind].mean().to_numpy()
        mu = float(per_gene.mean())
        lo, hi = boot_ci(per_gene, boot)
        out.append(dict(model=m, kind=kind, mean_hamming=round(mu, 3),
                        lo=round(lo, 3), hi=round(hi, 3), n_genes=len(per_gene)))
        line += f"   {mu:5.2f} [{lo:.2f}-{hi:.2f}]"
    print(line)

ref = pd.DataFrame([
    dict(model="real_other", kind="reference", mean_hamming=round(float(other.mean()), 3),
         lo=float("nan"), hi=float("nan"), n_genes=len(other)),
    dict(model="real_own", kind="reference", mean_hamming=0.0,
         lo=float("nan"), hi=float("nan"), n_genes=len(truth)),
])
res = pd.concat([ref, pd.DataFrame(out)], ignore_index=True)
res.to_csv(os.path.join(HERE, "panelC_hamming_v2.csv"), index=False)
print(f"\nwrote {os.path.join(HERE, 'panelC_hamming_v2.csv')}")

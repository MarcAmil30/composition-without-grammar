#!/usr/bin/env python3
"""Panels (a) and (c) split by background: leaked vs clean.

Same measurements as panelA_tol2.csv and panelC_hamming_v2.csv, but reported for
both backgrounds instead of `leaked` only, so the figure can show how much of
each model's score survives when the genomic filler is permuted away.

    leaked   real genomic sequence around the annotated elements
    clean    filler permuted to matched composition; the -35, -10, TSS, TFBS
             and last 20 bp stay real and in place

Panel (a) is placement within +/-2 bp of the gene's own annotated -10 position,
counted over every decode (greedy + 20 samples), sigma70 only -- the aggregation
that reproduces the published panelA_tol2.csv leaked column exactly.

Panel (c) is the Hamming distance between the six generated bases sitting where
the -10 belongs and that promoter's real hexamer, aggregated within gene first
and then across genes, with a bootstrap CI over genes.

Run on Isambard (the generations live there):

    source ${CONDA_ROOT}/bin/activate
    python3 panels_ac_by_background.py

Writes panelA_tol2_bg_v2.csv and panelC_hamming_bg_v2.csv next to itself.
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
CORE = 6
SEED = 0
N_PAIR = 20
N_BOOT = 2000
TOL = 2
# v2: all five models scored in one pass, so every row here comes from the
# same code path. The four published models reproduce their generated
# values exactly; only their shuffled controls move (~0.3pp), because the
# shuffle RNG stream depends on how many models are in the pass.
MODELS = ["evo1", "evo1_131k", "evo2", "hyenadna", "promogen2"]
BACKGROUNDS = ["leaked", "clean"]


def hamming(a: str, b: str):
    if len(a) != len(b) or len(a) != CORE:
        return None
    return sum(x != y for x, y in zip(a, b))


def shuffled(seq: str, rng: random.Random) -> str:
    ch = list(seq)
    rng.shuffle(ch)
    return "".join(ch)


def boot_ci(per_gene: np.ndarray, rng: np.random.Generator):
    """Bootstrap over genes -- the decodes within a gene are not independent."""
    if len(per_gene) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(per_gene), size=(N_BOOT, len(per_gene)))
    means = per_gene[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ panel (a)
pg = pd.read_csv(os.path.join(B, "scores", "per_generation.csv.gz"), low_memory=False)
pg = pg[(pg.task == "generate_promoter") & (pg.sigma == "sigma70")]

rows_a = []
for bg in BACKGROUNDS:
    s = pg[pg.background == bg]
    s = s.assign(
        gen_hit=(s.pos_error.abs() <= TOL),
        shuf_hit=((s.shuf_pos - s.true_pos).abs() <= TOL),
    )
    for m in MODELS:
        sub = s[s.model == m]
        if sub.empty:
            continue
        rows_a.append(dict(source=m, kind="generated", background=bg,
                           pct=round(sub.gen_hit.mean() * 100, 3),
                           n=int(sub.locus_tag.nunique())))
        rows_a.append(dict(source=m, kind="shuffled", background=bg,
                           pct=round(sub.shuf_hit.mean() * 100, 3),
                           n=int(sub.locus_tag.nunique())))

# the real-sequence ceiling is the annotated genome itself: one value, carried
# over unchanged from the published panel so the bar cannot drift
prev = pd.read_csv(os.path.join(HERE, "panelA_tol2.csv"))
real = prev[prev.kind == "real"].iloc[0]
rows_a.insert(0, dict(source="real", kind="real", background="real",
                      pct=float(real.pct), n=int(real.n)))

pd.DataFrame(rows_a).to_csv(os.path.join(HERE, "panelA_tol2_bg_v2.csv"), index=False)
print("wrote panelA_tol2_bg_v2.csv")
print(pd.DataFrame(rows_a).to_string(), "\n")

# ------------------------------------------------------------------ truth
truth = pd.read_csv(os.path.join(B, "prompts_681_leaked.csv"), keep_default_na=False)
truth = truth[truth.sigma == "sigma70"].copy()
truth["true_off"] = truth.m10_offset_in_target.astype(int)
truth["true_hex"] = [
    r.upstream_seq[max(0, int(r.m10_end) - CORE):int(r.m10_end)] for r in truth.itertuples()
]
truth = truth[truth.true_hex.str.len() == CORE].set_index("locus_tag")

# the clean background keeps the -10 real and in place, so the hexamer and its
# offset must be identical in both prompt files -- check rather than assume
chk = pd.read_csv(os.path.join(B, "prompts_681_clean.csv"), keep_default_na=False)
chk = chk[chk.sigma == "sigma70"].set_index("locus_tag")
shared = truth.index.intersection(chk.index)
same_hex = [
    truth.loc[t, "true_hex"] ==
    chk.loc[t, "upstream_seq"][max(0, int(chk.loc[t, "m10_end"]) - CORE):int(chk.loc[t, "m10_end"])]
    for t in shared
]
same_off = (truth.loc[shared, "true_off"].to_numpy()
            == chk.loc[shared, "m10_offset_in_target"].astype(int).to_numpy())
print(f"-10 hexamer identical in clean and leaked prompts: "
      f"{sum(same_hex)}/{len(shared)}; offset identical: {int(same_off.sum())}/{len(shared)}\n")

rng = random.Random(SEED)
hexes = truth.true_hex.tolist()
idx = list(range(len(hexes)))
other = []
for _ in range(N_PAIR):
    perm = idx[:]
    rng.shuffle(perm)
    other += [hamming(hexes[i], hexes[j]) for i, j in zip(idx, perm) if i != j]
other = np.array([o for o in other if o is not None], dtype=float)

# ------------------------------------------------------------------ panel (c)
files = sorted(glob.glob(os.path.join(B, "generations", "*.csv*")))
if not files:
    files = sorted(glob.glob(os.path.join(B, "generations", "**", "*.csv*"), recursive=True))
print(f"generation files found: {len(files)}")

rows = []
for bg in BACKGROUNDS:
    rng_shuf = random.Random(SEED)          # same draw sequence for both backgrounds
    for path in files:
        gen = pd.read_csv(path, keep_default_na=False)
        if gen.empty or "generation" not in gen.columns:
            continue
        gen = gen[(gen.task == "generate_promoter") & (gen.background == bg)]
        gen = gen[gen.model.isin(MODELS)]
        for r in gen.itertuples():
            if r.locus_tag not in truth.index:
                continue
            t = truth.loc[r.locus_tag]
            g = r.generation or ""
            off = int(t["true_off"])
            if off < 0 or off + CORE > len(g):
                continue
            h_gen = hamming(g[off:off + CORE], t["true_hex"])
            h_shf = hamming(shuffled(g, rng_shuf)[off:off + CORE], t["true_hex"])
            if h_gen is None or h_shf is None:
                continue
            rows.append((bg, r.model, r.locus_tag, h_gen, h_shf))

d = pd.DataFrame(rows, columns=["background", "model", "locus_tag", "generated", "shuffled"])
print(f"scored generations: {len(d)}\n")

boot = np.random.default_rng(SEED)
out = []
for bg in BACKGROUNDS:
    for m in MODELS:
        sub = d[(d.background == bg) & (d.model == m)]
        if sub.empty:
            continue
        for kind in ("generated", "shuffled"):
            per_gene = sub.groupby("locus_tag")[kind].mean().to_numpy()
            lo, hi = boot_ci(per_gene, boot)
            out.append(dict(model=m, kind=kind, background=bg,
                            mean_hamming=round(float(per_gene.mean()), 3),
                            lo=round(lo, 3), hi=round(hi, 3), n_genes=len(per_gene)))

ref = pd.DataFrame([
    dict(model="real_other", kind="reference", background="real",
         mean_hamming=round(float(other.mean()), 3),
         lo=float("nan"), hi=float("nan"), n_genes=len(other)),
    dict(model="real_own", kind="reference", background="real", mean_hamming=0.0,
         lo=float("nan"), hi=float("nan"), n_genes=len(truth)),
])
res = pd.concat([ref, pd.DataFrame(out)], ignore_index=True)
res.to_csv(os.path.join(HERE, "panelC_hamming_bg_v2.csv"), index=False)
print(res.to_string())
print("\nwrote panelC_hamming_bg_v2.csv")

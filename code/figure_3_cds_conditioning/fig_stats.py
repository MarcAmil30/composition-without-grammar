#!/usr/bin/env python3
"""
Compute every statistic for the CDS-prompted main figure, cache to a small .npz.

Runs on Isambard (where the generations live); the notebook renders from the
cache locally. Same two-stage pattern as make_cds_prompted_dataset_figure.py.

Panel B  natural vs generated ACGT profile over -20..-1   (both species)
Panel C  generated ACGT profile over the full 200 bp, C0 / C2 / C3
Panel D  per-gene minimum Shannon entropy, high vs low CAI
Table    KL(C0||Ci) per condition

Per-gene aggregation everywhere: within a gene, the 20 generations give a base
distribution at each position; genes are then combined. Never pool generations
across genes -- that is pseudoreplication.

    python fig_stats.py --out cds_fig_stats.npz
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import argparse, csv, glob, gzip, math, os
from collections import defaultdict
import numpy as np

BENCH = os.environ.get(
    "BENCH_DIR",
    f"{ICBINB_ROOT}/data/benchmark/cds_prompted")
SPECIES = ["ecoli", "bsubtilis"]
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
NBP, EPS = 200, 1e-12


def load_gen(sp, cond):
    """locus_tag -> list of 20 generated sequences."""
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(BENCH, "generations", sp,
                                           f"{cond}.shard*.csv.gz"))):
        with gzip.open(f, "rt") as fh:
            for r in csv.DictReader(fh):
                if len(r["sequence"]) == NBP and r["decode_idx"] != "-1":
                    out[r["locus_tag"]].append(r["sequence"])
    return {k: v[:20] for k, v in out.items() if len(v) >= 20}


def onehot_profile(seqs):
    """[200,4] within-gene base frequency across this gene's generations."""
    P = np.zeros((NBP, 4))
    for s in seqs:
        for i, ch in enumerate(s):
            j = B2I.get(ch)
            if j is not None:
                P[i, j] += 1
    return P / np.maximum(P.sum(1, keepdims=True), EPS)


def shannon(P):
    """[...,4] -> entropy in bits."""
    return -(P * np.log2(np.maximum(P, EPS))).sum(-1)


def natural_profile(seqs):
    """[200,4] population frequency across genes (one real sequence per gene)."""
    P = np.zeros((NBP, 4))
    for s in seqs:
        for i, ch in enumerate(s):
            j = B2I.get(ch)
            if j is not None:
                P[i, j] += 1
    return P / np.maximum(P.sum(1, keepdims=True), EPS)


def cai_set(sp, which):
    p = os.path.join(BENCH, f"{sp}_singletons_top200_{which}_CAI.csv")
    return {r["locus_tag"] for r in csv.DictReader(open(p))}


def perm_test(a, b, n=20000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs):
            c += 1
    return obs, (c + 1) / (n + 1)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cds_fig_stats.npz")
args = ap.parse_args()

def kmer_profile(seqs, k=6):
    """normalised k-mer frequency vector over a set of sequences."""
    from itertools import product
    idx = {"".join(t): i for i, t in enumerate(product(BASES, repeat=k))}
    v = np.zeros(len(idx))
    for s in seqs:
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i+k])
            if j is not None:
                v[j] += 1
    return v / max(v.sum(), 1)


store = {}
NAT, GEN = {}, {}
for sp in SPECIES:
    tgt = {r["locus_tag"]: r["target_upstream_200bp"]
           for r in csv.DictReader(open(os.path.join(BENCH, f"prompts_C0_{sp}.csv")))}

    gens = {c: load_gen(sp, c) for c in ("C0", "C2", "C3")}
    genes = sorted(set(gens["C0"]) & set(gens["C2"]) & set(gens["C3"]) & set(tgt))
    print(f"\n=== {sp}: {len(genes)} genes with C0/C2/C3 ===")

    # ---- natural profile (population frequency across genes) ----
    store[f"{sp}_natural"] = natural_profile([tgt[g] for g in genes])

    # ---- panel A: 6-mer composition, natural vs CDS-prompted generations ----
    NAT[sp] = kmer_profile([tgt[g] for g in genes])
    GEN[sp] = kmer_profile([x for g in genes for x in gens["C0"][g][:5]])
    store[f"{sp}_kmer_nat"] = NAT[sp]
    store[f"{sp}_kmer_gen"] = GEN[sp]

    # ---- per-gene generated profiles, then mean/sd across genes ----
    for c in ("C0", "C2", "C3"):
        per_gene = np.stack([onehot_profile(gens[c][g]) for g in genes])   # [G,200,4]
        store[f"{sp}_{c}_mean"] = per_gene.mean(0)
        store[f"{sp}_{c}_sd"] = per_gene.std(0)
        if c == "C0":
            store[f"{sp}_pergene_C0"] = per_gene.astype(np.float32)
        print(f"  {c}: profile from {per_gene.shape[0]} genes")

    # ---- panel D: per-gene minimum Shannon entropy over the SD window ----
    hi, lo = cai_set(sp, "high"), cai_set(sp, "low")
    D = {}
    for lab, keys in (("high", hi), ("low", lo)):
        ks = [g for g in genes if g in keys]
        vals_min, vals_mean = [], []
        for g in ks:
            P = onehot_profile(gens["C0"][g])          # [200,4] within-gene
            H = shannon(P)                              # [200]
            vals_min.append(H[180:200].min())           # most confident position in SD window
            vals_mean.append(H[180:200].mean())
        D[lab] = (np.array(vals_min), np.array(vals_mean), ks)
        store[f"{sp}_D_{lab}_min"] = D[lab][0]
        store[f"{sp}_D_{lab}_mean"] = D[lab][1]
    for stat, idx in (("min entropy", 0), ("mean entropy", 1)):
        o, p = perm_test(D["high"][idx], D["low"][idx])
        print(f"  panel D {stat:13s}: high {D['high'][idx].mean():.4f} "
              f"low {D['low'][idx].mean():.4f}  diff {o:+.4f}  perm p = {p:.4g} "
              f"(n={len(D['high'][idx])}/{len(D['low'][idx])})")
        store[f"{sp}_D_{stat.split()[0]}_p"] = p

    # ---- panel D': SD placement, per gene, all conditions + real ceiling ----
    SDC = "AGGAGG"
    def sd_find(u):
        best = None
        for i in range(len(u) - 20, len(u) - 3):
            j = i + 6
            if j > len(u): break
            sc = sum(x == y for x, y in zip(u[i:j], SDC)); spc = len(u) - j
            if 3 <= spc <= 12 and (best is None or sc > best[2]
                                   or (sc == best[2] and abs(spc-7) < abs(best[3]-7))):
                best = (i, j, sc, spc)
        return best
    allg = {c: load_gen(sp, c) for c in CONDS}
    for c in CONDS:
        frac = []                       # per gene: fraction of its 20 gens with SD>=4
        for g in genes:
            if g not in allg[c]: continue
            hits = [sd_find(x) for x in allg[c][g]]
            frac.append(np.mean([1.0 if (h and h[2] >= 4) else 0.0 for h in hits]))
        store[f"{sp}_sdfrac_{c}"] = np.array(frac)
        print(f"  SD present, {c}: {100*np.mean(frac):.1f}%  (n={len(frac)} genes)")
    real_hits = [sd_find(tgt[g]) for g in genes]
    store[f"{sp}_sdfrac_REAL"] = np.array(
        [1.0 if (h and h[2] >= 4) else 0.0 for h in real_hits])
    print(f"  SD present, REAL ceiling: {100*store[f'{sp}_sdfrac_REAL'].mean():.1f}%")

    # natural equivalent for panel D reference: min entropy is undefined for a
    # single sequence, so use the population SD-window information content
    for lab, keys in (("high", hi), ("low", lo)):
        ks = [g for g in genes if g in keys]
        P = natural_profile([tgt[g] for g in ks])
        store[f"{sp}_Dnat_{lab}"] = shannon(P)[180:200]

# 6-mer correlations: matched species vs the cross-species control
for a in SPECIES:
    for b in SPECIES:
        r = float(np.corrcoef(GEN[a], NAT[b])[0, 1])
        store[f"kmer_r_gen{a}_nat{b}"] = r
        print(f"6-mer r: generated {a:10s} vs natural {b:10s} = {r:.4f}"
              + ("   <- matched" if a == b else ""))

store["conds"] = np.array(CONDS)
np.savez_compressed(args.out, **store)
print(f"\n[wrote {args.out}]  keys: {len(store)}")

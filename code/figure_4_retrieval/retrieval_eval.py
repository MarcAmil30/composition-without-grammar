#!/usr/bin/env python3
"""
Task 4 — shared evaluation code. Model-agnostic.

Every model produces one thing: an n x n matrix M where M[i, j] = how well
candidate upstream j scores against query gene i (cosine, or log-likelihood).
Everything downstream lives here, so the ranking, the decoy draws and the
metrics are identical for every model and both scorers.

The decoy draws are written to disk ONCE and reused by every run -- gene i,
repeat r faces the same 99 decoys for PROK, Evo, C3P and both scorers. That is
what makes the paired tests valid.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RETR = Path(os.environ.get("ICBINB_RETRIEVAL_DIR", REPO_ROOT / "data" / "retrieval"))
EVAL_CSV = RETR / "ecoli_retrieval_eval500.csv"

N_CAND = 100        # 1 true + 99 decoys
N_REPEATS = 20
SEED = 0
UP_LEN = 200        # fixed for every model, every scorer -- see PROCEDURE.md


def load_eval(limit=0):
    df = pd.read_csv(EVAL_CSV).sort_values("eval_rank").reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    df["up200"] = df.upstream_seq.str[-UP_LEN:]
    assert (df.up200.str.len() == UP_LEN).all(), "an upstream is shorter than 200 bp"
    return df


# ------------------------------------------------------------------ decoys
def decoy_draws(n, n_cand=N_CAND, n_repeats=N_REPEATS, seed=SEED, cache=True):
    """[n, n_repeats, n_cand-1] decoy indices; never contains the query itself.

    Cached to disk so every model demonstrably uses the same draws.
    """
    assert n > n_cand, (
        f"need more than {n_cand} genes to draw {n_cand-1} decoys, got {n}. "
        f"A smoke test must use --limit > {n_cand}.")
    path = RETR / f"draws_seed{seed}_n{n}_k{n_cand}_r{n_repeats}.npy"
    if cache and os.path.exists(path):
        return np.load(path)
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_repeats, n_cand - 1), dtype=np.int32)
    for i in range(n):
        pool = np.delete(np.arange(n), i)
        for r in range(n_repeats):
            out[i, r] = rng.choice(pool, size=n_cand - 1, replace=False)
    if cache:
        np.save(path, out)
    return out


# ------------------------------------------------------------------ ranking
def ranks_from_matrix(M, higher_is_better=True, draws=None):
    """Rank of the true pair among n_cand candidates. 1 = perfect. Shape [n, reps]."""
    n = M.shape[0]
    assert M.shape == (n, n), f"matrix must be square, got {M.shape}"
    if draws is None:
        draws = decoy_draws(n)
    S = M if higher_is_better else -M
    true = S[np.arange(n), np.arange(n)][:, None]                 # [n, 1]
    dec = S[np.arange(n)[:, None, None], draws]                   # [n, reps, k-1]
    # ties count against us: rank = 1 + #(decoys strictly better) + #(ties)
    better = (dec > true[:, :, None]).sum(-1) + (dec == true[:, :, None]).sum(-1)
    return (1 + better).astype(np.int32)


def metrics(ranks, n_cand=N_CAND, n_boot=1000, seed=SEED):
    """top-1/top-5/median/MRR + bootstrap CI resampled over GENES, not trials.

    The 20 trials of a gene are not independent -- bootstrapping over all
    n*reps trials would understate the CI by roughly sqrt(20).
    """
    per_gene_top1 = (ranks == 1).mean(1)          # [n]
    per_gene_top5 = (ranks <= 5).mean(1)
    per_gene_mrr = (1.0 / ranks).mean(1)
    rng = np.random.default_rng(seed)
    n = len(per_gene_top1)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = per_gene_top1[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_genes": int(n),
        "n_repeats": int(ranks.shape[1]),
        "top1": float(per_gene_top1.mean()),
        "top1_ci95": [float(lo), float(hi)],
        "top5": float(per_gene_top5.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float(per_gene_mrr.mean()),
        "fold_over_chance": float(per_gene_top1.mean() * n_cand),
        "chance_top1": 1.0 / n_cand,
    }


def true_vs_decoy_gap(M, higher_is_better=True, draws=None, valid=None):
    """Paired true-minus-mean-decoy gap + Wilcoxon.

    THE PLUMBING GUARD. A model can sit at chance top-1 and still show a clean
    positive gap -- that means the pipeline works and the effect is merely weak.
    A gap of ~0 with chance top-1 is the signature of a wiring bug, which is
    exactly how the first C3P run produced a wrong null.
    """
    from scipy.stats import wilcoxon

    n = M.shape[0]
    if draws is None:
        draws = decoy_draws(n)
    S = M if higher_is_better else -M
    true = S[np.arange(n), np.arange(n)]
    dec = S[np.arange(n)[:, None, None], draws].reshape(n, -1).mean(1)
    d = true - dec
    if valid is not None:
        d, true, dec = d[valid], true[valid], dec[valid]
    stat, p = wilcoxon(d, alternative="greater")
    return {
        "true_mean": float(true.mean()),
        "decoy_mean": float(dec.mean()),
        "gap_mean": float(d.mean()),
        "gap_median": float(np.median(d)),
        "frac_true_higher": float((d > 0).mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p_greater": float(p),
    }


def variance_decomposition(M):
    """How much of a score matrix is per-candidate, per-query, and interaction.

    For raw log-likelihoods the per-candidate term ("how intrinsically likely is
    this upstream") dominates -- 98% on the PROK smoke test. That term is the
    SAME for every query, so ranking on it puts the true pair at chance by
    construction. Only the interaction carries CDS-specific information.
    """
    col, row, tot = np.nanmean(M, 0), np.nanmean(M, 1), np.nanvar(M)
    inter = M - col[None, :] - row[:, None] + np.nanmean(M)
    return {"between_candidate_frac": float(np.nanvar(col) / tot),
            "between_query_frac": float(np.nanvar(row) / tot),
            "interaction_frac": float(np.nanvar(inter) / tot)}


def normalize(M, how, null=None):
    """Remove the per-candidate nuisance term from a likelihood matrix.

    raw       as scored
    pmi       S[i,j] - mean_i S[i,j]      transductive: the eval set estimates P(up_j)
    pmi_null  S[i,j] - null[j]            null[j] = log P(up_j) with no CDS context;
                                          non-transductive, costs n extra forwards
    zcol      column z-score              also removes per-candidate scale
    double    row and column centered

    Cosine matrices are already comparable across candidates, so they use "raw".
    """
    if how == "raw":
        return M
    if how == "pmi":
        return M - np.nanmean(M, 0, keepdims=True)
    if how == "pmi_null":
        assert null is not None, "pmi_null needs the null-context scores"
        return M - null[None, :]
    if how == "zcol":
        return ((M - np.nanmean(M, 0, keepdims=True))
                / np.nanstd(M, 0, keepdims=True).clip(1e-8))
    if how == "double":
        return (M - np.nanmean(M, 0, keepdims=True)
                - np.nanmean(M, 1, keepdims=True) + np.nanmean(M))
    raise ValueError(how)


def evaluate(M, name, out_dir, higher_is_better=True, extra=None, save_matrix=True):
    """Full evaluation of one score matrix -> ranks parquet + summary json."""
    os.makedirs(out_dir, exist_ok=True)
    n = M.shape[0]
    draws = decoy_draws(n)
    # A query row may be unscorable (lm_retrieval --max_len). Such a gene is
    # dropped as a QUERY but stays in the candidate pool, so every model still
    # ranks against the same 500 candidates with the same decoy draws.
    valid = ~np.isnan(M).any(axis=1)
    Mw = np.nan_to_num(M, nan=-1e30, posinf=-1e30, neginf=-1e30)
    ranks = ranks_from_matrix(Mw, higher_is_better, draws)[valid]

    gene_idx = np.where(valid)[0]          # excluded rows are not in `ranks`
    df = pd.DataFrame({
        "gene_row": np.repeat(gene_idx, ranks.shape[1]),
        "rep": np.tile(np.arange(ranks.shape[1]), len(gene_idx)),
        "rank": ranks.ravel(),
    })
    df.to_parquet(f"{out_dir}/{name}_ranks.parquet", index=False)

    summary = {"name": name, "higher_is_better": higher_is_better,
               "upstream_len": UP_LEN, "n_candidates": N_CAND, "seed": SEED}
    summary.update(metrics(ranks))
    summary["variance_decomposition"] = variance_decomposition(Mw[valid])
    summary["n_query_rows_excluded"] = int((~valid).sum())
    summary["guard_true_vs_decoy"] = true_vs_decoy_gap(Mw, higher_is_better, draws, valid)
    # the same matrix ranked the other way -- direction settled by data, not assumption
    summary["top1_reversed_direction"] = float(
        (ranks_from_matrix(Mw, not higher_is_better, draws)[valid] == 1).mean())
    if extra:
        summary.update(extra)
    with open(f"{out_dir}/{name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if save_matrix:
        np.save(f"{out_dir}/{name}_matrix.npy", M.astype(np.float32))
    return summary


def report(summary):
    m, g = summary, summary["guard_true_vs_decoy"]
    lo, hi = m["top1_ci95"]
    print(f"\n=== {m['name']} (L={m['upstream_len']}, n={m['n_genes']}) ===")
    print(f"  top-1        {m['top1']*100:6.2f}%  [{lo*100:.2f}, {hi*100:.2f}]"
          f"   ({m['fold_over_chance']:.1f}x chance)")
    print(f"  top-5        {m['top5']*100:6.2f}%")
    print(f"  median rank  {m['median_rank']:6.1f}   (chance 50)")
    print(f"  MRR          {m['mrr']:6.4f}")
    print(f"  GUARD true-vs-decoy gap {g['gap_mean']:+.5f}  "
          f"true higher in {g['frac_true_higher']*100:.1f}% of genes  "
          f"p={g['wilcoxon_p_greater']:.2e}")
    if m["top1"] < 0.02 and g["frac_true_higher"] < 0.55:
        print("  ** at chance AND no gap -- check the wiring before believing this null")
    elif m["top1"] < 0.02:
        print("  ** at chance but the gap is positive -- signal exists, too weak to rank")

#!/usr/bin/env python3
"""
Run ON CREATE. Aggregates the Fig. 1a pair table into a small summary bundle
that can be copied back to the laptop and plotted without the cluster.

    $P/c3p_env/bin/python fig1a_dataset_summary_remote.py

Writes results/upstream_grammar/fig1a_dataset_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import os

# Root of the Contrastive_Learning_NeurIPS2026 workspace that holds
# results/upstream_grammar/pairs_cds_singleton_pairs.parquet.
PROJ = os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
SRC = Path(PROJ) / "results" / "upstream_grammar" / "pairs_cds_singleton_pairs.parquet"
OUT = SRC.parent / "fig1a_dataset_summary.json"

d = pd.read_parquet(SRC)
print("columns:", list(d.columns))

METRIC, CTRL = "d2star_k4_test", "d2star_k4_c1"
summary = {
    "n_pairs": int(len(d)),
    "n_genomes": int(d["accession"].nunique()),
    "n_queries": int(d["as_row"].nunique()) if "as_row" in d else None,
    "columns": list(d.columns),
}

# pairs and genomes per taxonomic rank
by_rank = []
for r, g in d.groupby("taxon_rank"):
    by_rank.append({
        "rank": str(r),
        "n_pairs": int(len(g)),
        "n_genomes": int(g["accession"].nunique()),
        "p3_median": float(np.nanmedian(g["p3"])),
        "d2_median": float(np.nanmedian(g[METRIC])),
        "d2_ctrl_median": float(np.nanmedian(g[CTRL])),
    })
summary["by_rank"] = by_rank

# how many pairs each target genome contributes
per_genome = d.groupby("accession").size().values
summary["pairs_per_genome"] = {
    "median": float(np.median(per_genome)), "min": int(per_genome.min()),
    "max": int(per_genome.max()),
    "hist_edges": list(np.arange(0, per_genome.max() + 10, 5).astype(float)),
    "hist_counts": [int(c) for c in np.histogram(
        per_genome, bins=np.arange(0, per_genome.max() + 10, 5))[0]],
}

# how many orthologues each E. coli anchor gene collected
if "as_row" in d:
    per_query = d.groupby("as_row").size().values
    summary["pairs_per_query"] = {
        "n_queries": int(len(per_query)), "median": float(np.median(per_query)),
        "min": int(per_query.min()), "max": int(per_query.max()),
        "hist_edges": list(np.arange(0, per_query.max() + 10, 5).astype(float)),
        "hist_counts": [int(c) for c in np.histogram(
            per_query, bins=np.arange(0, per_query.max() + 10, 5))[0]],
    }

# the x axis of Fig. 1a, and the two y distributions
def dist(v, lo, hi, nbin):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    c, e = np.histogram(v, bins=np.linspace(lo, hi, nbin + 1))
    return {"edges": [float(x) for x in e], "counts": [int(x) for x in c],
            "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "n": int(v.size)}

summary["p3"] = dist(d["p3"], 25, 100, 30)
summary["p3_by_rank"] = {str(r): dist(g["p3"], 25, 100, 30)
                         for r, g in d.groupby("taxon_rank")}
summary["d2_ortho"] = dist(d[METRIC], -0.1, 1.0, 30)
summary["d2_ctrl"] = dist(d[CTRL], -0.1, 1.0, 30)

# any length columns present, for the control-matching panel
for col in d.columns:
    if "len" in col.lower():
        summary.setdefault("lengths", {})[col] = dist(
            d[col], float(np.nanmin(d[col])), float(np.nanmax(d[col])), 30)

OUT.write_text(json.dumps(summary, indent=2))
print("wrote", OUT)

# --- robustness: rho across metric x k x series, and within each rank -------
from scipy.stats import spearmanr

SERIES = {"test": "orthologue", "c1": "non-homologous, same genome",
          "c3": "dinucleotide shuffle of the orthologue upstream",
          "c4": "random downstream flank, same genome"}
rho_grid = []
for metric in ("d2star", "pearson"):
    for k in (3, 4, 5):
        for s in SERIES:
            col = f"{metric}_k{k}_{s}"
            if col not in d:
                continue
            m = np.isfinite(d["p3"]) & np.isfinite(d[col])
            rho_grid.append({"metric": metric, "k": k, "series": s,
                             "rho": float(spearmanr(d.loc[m, "p3"], d.loc[m, col])[0]),
                             "n": int(m.sum()),
                             "median": float(np.nanmedian(d.loc[m, col]))})
summary["rho_grid"] = rho_grid
summary["series_labels"] = SERIES

within = []
for r, g in d.groupby("taxon_rank"):
    row = {"rank": str(r), "n": int(len(g))}
    for s in SERIES:
        col = f"d2star_k4_{s}"
        m = np.isfinite(g["p3"]) & np.isfinite(g[col])
        row[f"rho_{s}"] = float(spearmanr(g.loc[m, "p3"], g.loc[m, col])[0]) if m.sum() > 10 else None
        row[f"median_{s}"] = float(np.nanmedian(g.loc[m, col])) if m.sum() else None
    within.append(row)
summary["within_rank"] = within

OUT.write_text(json.dumps(summary, indent=2))
print("rewrote", OUT, "with rho_grid and within_rank")

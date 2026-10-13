"""
Paired statistical comparison: C3P embedding retrieval vs Evo-1-8k score retrieval.
Per species, per upstream length. Question: is C3P significantly BETTER (lower rank)?

Both experiments: 1-in-100 retrieval, 20 repeats, store per-gene per-rep rank (1=best),
keyed by gene_row (analysis_set index). The Evo run used a 150-gene subsample that is a
SUBSET of the C3P genes -> we pair on the 150 shared genes per species.

Metric per gene = mean rank over its 20 repeats (reduces rep noise; pairs by gene to
control for gene-intrinsic difficulty). Test = Wilcoxon signed-rank, one-sided
(H1: C3P rank < Evo rank). Evo side uses the CDS-conditional score ('cds', its strongest).
Holm correction across the 8 (species x length) tests.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = f"{CL_ROOT}"
OUT = f"{PROJ}/results/comparison_c3p_vs_evo"
os.makedirs(OUT, exist_ok=True)
REF = ["Escherichia coli", "Bacillus subtilis"]
LENGTHS = [300, 200, 100, 50]

c3p = pd.read_parquet(f"{PROJ}/results/exp3_length/exp3_length_ranks.parquet")
evo = pd.read_parquet(f"{PROJ}/results/evo_8k_score/evo8k_score_ranks.parquet")
evo = evo[evo["metric"] == "cds"].copy()

rows = []
for sp in REF:
    for L in LENGTHS:
        c = (c3p[(c3p.species == sp) & (c3p.length == L)]
             .groupby("gene_row")["rank"].agg(mean_rank="mean",
                                              top1=lambda x: (x == 1).mean()))
        e = (evo[(evo.species == sp) & (evo.length == L)]
             .groupby("gene_row")["rank"].agg(mean_rank="mean",
                                              top1=lambda x: (x == 1).mean()))
        m = c.join(e, lsuffix="_c3p", rsuffix="_evo", how="inner").dropna()
        n = len(m)
        cr, er = m["mean_rank_c3p"].to_numpy(), m["mean_rank_evo"].to_numpy()
        # one-sided Wilcoxon signed-rank: H1 c3p < evo (C3P better = lower rank)
        stat, p = wilcoxon(cr, er, alternative="less", zero_method="wilcox")
        diff = er - cr                       # >0 => C3P better (lower rank)
        frac_c3p_better = float((cr < er).mean())
        # matched-pairs rank-biserial effect size
        d = er - cr
        d = d[d != 0]
        rpos = np.sum(np.argsort(np.argsort(np.abs(d)))[d > 0] + 1)
        rall = np.sum(np.arange(1, len(d) + 1))
        rb = float(2 * rpos / rall - 1) if len(d) else np.nan
        rows.append(dict(
            species=sp, length_bp=L, n_shared=n,
            c3p_top1=float(m["top1_c3p"].mean()), evo_top1=float(m["top1_evo"].mean()),
            c3p_mean_rank=float(cr.mean()), evo_mean_rank=float(er.mean()),
            median_paired_diff=float(np.median(diff)),
            frac_genes_c3p_better=frac_c3p_better,
            rank_biserial=rb, wilcoxon_stat=float(stat), p_one_sided=float(p)))

res = pd.DataFrame(rows)
res["p_holm"] = multipletests(res["p_one_sided"], method="holm")[1]
res["significant_holm_.05"] = res["p_holm"] < 0.05
res.to_csv(f"{OUT}/c3p_vs_evo_stats.csv", index=False)

def stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

# ---- figure: top-1 grouped bars per species, with significance ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
x = np.arange(len(LENGTHS)); w = 0.38
for ax, sp in zip(axes, REF):
    sub = res[res.species == sp].set_index("length_bp").loc[LENGTHS]
    ax.bar(x - w/2, sub["c3p_top1"] * 100, w, label="C3P (protein↔upstream, cosine)", color="#4C72B0")
    ax.bar(x + w/2, sub["evo_top1"] * 100, w, label="Evo-1-8k (CDS|upstream, score)", color="#55A868")
    for xi, L in enumerate(LENGTHS):
        r = sub.loc[L]
        ymax = max(r["c3p_top1"], r["evo_top1"]) * 100
        ax.text(xi, ymax + 3, stars(r["p_holm"]), ha="center", va="bottom", fontsize=11)
    ax.axhline(1, ls="--", lw=0.8, color="grey")  # chance top-1 = 1%
    ax.set_xticks(x); ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.set_xlabel("Upstream length (bp)")
    ax.set_title(f"{sp}\n(Wilcoxon signed-rank, Holm-adj; n=150 paired genes)")
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Top-1 retrieval accuracy (%)   [chance = 1%]")
axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle("C3P embedding retrieval vs Evo-1-8k score retrieval — is C3P significantly better?",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/c3p_vs_evo_top1.png", dpi=150)

print("\n================ C3P vs EVO (paired, per species x length) ================")
cols = ["species", "length_bp", "n_shared", "c3p_top1", "evo_top1",
        "c3p_mean_rank", "evo_mean_rank", "frac_genes_c3p_better", "p_one_sided", "p_holm"]
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(res[cols].to_string(index=False))
print(f"\nwrote {OUT}/c3p_vs_evo_stats.csv + c3p_vs_evo_top1.png")
print("COMPARE_OK")

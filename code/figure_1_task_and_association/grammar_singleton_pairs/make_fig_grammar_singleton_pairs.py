"""
Singleton-pair robustness panel: upstream similarity against gene relatedness.

x  gene relatedness, measured at third codon positions -- silent sites, so it tracks elapsed
   time largely free of selection on the protein
y  alignment-free similarity (D2*, k = 4) between the E. coli gene's 200 bp upstream and its
   orthologue's upstream in another genome

Points are coloured by the target genome's taxonomic distance from E. coli. The grey dashed
line is the identical measurement against a CONTROL upstream -- a length-matched, non-
homologous gene from the SAME target genome -- so phylogenetic distance is held constant. Its
flatness is what makes the trend interpretable: it rules out the possibility that the rise is
just an artefact of comparing E. coli to a closer genome.

Run on CREATE:
    $P/c3p_env/bin/python make_fig_grammar.py

Outputs: results/figures/fig_grammar_singleton_pairs.{pdf,png}
         + fig_grammar_singleton_pairs_stats.csv
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr

PROJ = f"{CL_ROOT}"
OUT = f"{PROJ}/results/figures"
os.makedirs(OUT, exist_ok=True)

INK, GREY = "#1A1A1A", "#8C8C8C"
# deliberately different hues, not a blue ramp -- the ranks must be told apart at a glance
RANKS = [("same_genus", "same genus", "#6A3D9A"),     # purple
         ("same_family", "same family", "#D1442F"),   # red
         ("same_order", "same order", "#E8A33D"),     # gold
         ("same_class", "same class", "#3FA7C4")]     # cyan

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 110, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    "axes.linewidth": 2.4, "axes.edgecolor": INK,
    "xtick.major.width": 2.4, "ytick.major.width": 2.4,
    "xtick.major.size": 7.0, "ytick.major.size": 7.0,
    "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.labelsize": 13.5, "axes.labelweight": "bold",
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "legend.fontsize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
})

d = pd.read_parquet(f"{PROJ}/results/upstream_grammar/pairs_cds_singleton_pairs.parquet")
d = d[np.isfinite(d["p3"]) & np.isfinite(d["d2star_k4_test"])]

fig, ax = plt.subplots(figsize=(8.0, 6.0))
for zi, (key, lab, col) in enumerate(reversed(RANKS)):
    s = d[d["taxon_rank"] == key]
    big = len(s) > 2000
    ax.scatter(s["p3"], s["d2star_k4_test"], s=4 if big else 14, color=col,
               alpha=0.18 if big else 0.85, linewidths=0, rasterized=True, zorder=2 + zi)

edges = np.arange(25, 102, 5)
mid = 0.5 * (edges[:-1] + edges[1:])
trend = lambda y: np.array([np.nanmedian(d.loc[(d["p3"] >= a) & (d["p3"] < b), y])
                            if ((d["p3"] >= a) & (d["p3"] < b)).sum() >= 30 else np.nan
                            for a, b in zip(edges[:-1], edges[1:])])
ax.plot(mid, trend("d2star_k4_test"), lw=3.4, color=INK, zorder=7, solid_capstyle="round")
ax.plot(mid, trend("d2star_k4_c1"), lw=2.8, color=GREY, ls=(0, (5, 3)), zorder=7)

rho_t, _ = spearmanr(d["p3"], d["d2star_k4_test"])
rho_c, _ = spearmanr(d["p3"], d["d2star_k4_c1"])
r_t = pearsonr(d["p3"], d["d2star_k4_test"])[0]
r_c = pearsonr(d["p3"], d["d2star_k4_c1"])[0]
box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#DDDDDD", lw=1.2)
ax.text(0.028, 0.665,
        f"orthologue   $\\rho$ = {rho_t:+.2f}   $\\rho^2$ = {rho_t**2:.2f}   "
        f"$r^2$ = {r_t**2:.2f}\n"
        f"control          $\\rho$ = {rho_c:+.2f}   $\\rho^2$ = {rho_c**2:.2f}   "
        f"$r^2$ = {r_c**2:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color=INK,
        linespacing=1.7, bbox=box, zorder=9)

ax.set_xlim(25, 100)
ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("Gene relatedness\n(identity at third codon positions, %)")
ax.set_ylabel("Upstream similarity (D2*)", labelpad=10)
ax.grid(axis="y", lw=0.9, color="#EDEDED", zorder=0)
ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in RANKS] +
                  [Line2D([0], [0], color=INK, lw=3.2, label="orthologue (running median)"),
                   Line2D([0], [0], color=GREY, lw=2.8, ls=(0, (5, 3)),
                          label="control upstream")],
          loc="upper left", bbox_to_anchor=(-0.01, 1.015), handlelength=1.5,
          labelspacing=0.34, handletextpad=0.7)
ax.set_title("Related singleton genes have similar upstreams\n"
             f"n = {len(d):,} singleton-singleton orthologue pairs, "
             f"{d['accession'].nunique()} genomes", fontsize=12.5, pad=12)

pd.DataFrame([dict(series="orthologue", spearman_rho=rho_t, rho_squared=rho_t**2,
                   pearson_r=r_t, r_squared=r_t**2, n=len(d)),
              dict(series="control", spearman_rho=rho_c, rho_squared=rho_c**2,
                   pearson_r=r_c, r_squared=r_c**2, n=len(d))]).to_csv(
    f"{OUT}/fig_grammar_singleton_pairs_stats.csv", index=False)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_grammar_singleton_pairs.{ext}")
print(f"orthologue rho={rho_t:+.3f} rho2={rho_t**2:.3f} r2={r_t**2:.3f}")
print(f"control    rho={rho_c:+.3f} rho2={rho_c**2:.3f} r2={r_c**2:.3f}")
print(f"wrote {OUT}/fig_grammar_singleton_pairs.pdf / .png")

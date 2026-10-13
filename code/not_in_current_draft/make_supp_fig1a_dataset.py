#!/usr/bin/env python3
"""
Supplementary figure: what the Fig. 1a orthologue-pair dataset is made of.

Everything is read from results/fig1a_dataset/fig1a_dataset_summary.json, a
small bundle aggregated on CREATE from
results/upstream_grammar/pairs_cds_singleton_pairs.parquet by
code/figures/fig1a_dataset_summary_remote.py (the per-pair table itself is too
large, and lives only on the cluster).

  A  pairs per taxonomic rank, with the number of contributing target genomes
  B  where each rank sits on the x axis of Fig. 1a
  C  the ancestry-controlled test: Spearman rho within each rank, orthologue
     against all three negative controls
  D  the same correlation across both similarity metrics and k = 3, 4, 5

Output: figures/supp_fig1a_dataset.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "fig1a_dataset" / "fig1a_dataset_summary.json"
OUT_STEMS = [ROOT / "results" / "supp_fig1a_dataset",
             ROOT / "figures" / "supp_fig1a_dataset"]

# same hues as Fig. 1a, ordered from closest to most distant relative
RANKS = [("same_genus", "same genus", "#6A3D9A"),
         ("same_family", "same family", "#D1442F"),
         ("same_order", "same order", "#E8A33D"),
         ("same_class", "same class", "#3FA7C4")]
SERIES = [("test", "orthologue", "#1a1a1a"),
          ("c1", "non-homologous gene,\nsame genome", "#8C8C8C"),
          ("c4", "random downstream flank,\nsame genome", "#B8B8B8"),
          ("c3", "dinucleotide shuffle of\nthe orthologue upstream", "#DCDCDC")]
INK, GREY = "#1a1a1a", "#8C8C8C"


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.22,
        "axes.linewidth": 2.0, "axes.edgecolor": INK,
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 6.0, "ytick.major.size": 6.0,
        "xtick.minor.width": 1.2, "ytick.minor.width": 1.2,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.labelsize": 11.5, "axes.labelweight": "bold", "axes.titlesize": 12.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "lines.solid_capstyle": "round",
    })


def tag(ax, letter, dx=-0.16, dy=1.15):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", color=INK)


def tidy(ax, n_minor_x=0, n_minor_y=0):
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.0)
        ax.spines[s].set_position(("outward", 6))
    if n_minor_x:
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n_minor_x))
    if n_minor_y:
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n_minor_y))


def panel_A(ax, d):
    by = {r["rank"]: r for r in d["by_rank"]}
    keys = [k for k, _, _ in RANKS]
    pairs = [by[k]["n_pairs"] for k in keys]
    genomes = [by[k]["n_genomes"] for k in keys]
    y = np.arange(len(keys))[::-1]

    ax.barh(y, pairs, height=0.68, color=[c for _, _, c in RANKS],
            edgecolor=INK, lw=1.4, zorder=3)
    for yy, p, g in zip(y, pairs, genomes):
        ax.text(p + max(pairs) * 0.025, yy,
                f"{p:,} pairs   ·   {g} genome" + ("s" if g != 1 else ""),
                va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in RANKS], fontsize=11)
    ax.set_xlabel("orthologue pairs")
    ax.set_xlim(0, max(pairs) * 1.62)
    ax.set_title(f"{d['n_pairs'] - 1:,} pairs from {d['n_genomes']} target genomes\n"
                 f"({d['n_queries']} of the 727 anchor genes contributed at least one)",
                 fontweight="bold", pad=10)
    tidy(ax, 2)
    tag(ax, "A", dx=-0.22)


def panel_B(ax, d):
    for key, lab, col in RANKS:
        h = d["p3_by_rank"][key]
        e = np.asarray(h["edges"])
        c = np.asarray(h["counts"], float)
        c = c / c.sum() * 100
        mid = (e[:-1] + e[1:]) / 2
        ax.step(mid, c, where="mid", color=col, lw=2.4, zorder=4,
                label=f"{lab}  (median {h['median']:.0f}%)")
        ax.fill_between(mid, c, step="mid", color=col, alpha=0.16, lw=0, zorder=2)

    ax.set_xlabel("gene relatedness (identity at third codon positions, %)")
    ax.set_ylabel("pairs within rank (%)")
    ax.set_xlim(25, 100)
    ax.set_title("Where each rank sits on the Fig. 1a axis", fontweight="bold", pad=10)
    ax.legend(loc="upper left", handlelength=1.6, labelspacing=0.4,
              borderpad=0.2, fontsize=9)
    tidy(ax, 2, 2)
    tag(ax, "B")


def panel_C(ax, d):
    w = pd.DataFrame(d["within_rank"]).set_index("rank")
    keys = [k for k, _, _ in RANKS]
    x = np.arange(len(keys))
    bw = 0.2

    for i, (s, lab, col) in enumerate(SERIES):
        v = [w.loc[k, f"rho_{s}"] for k in keys]
        off = (i - 1.5) * bw
        ax.bar(x + off, v, bw * 0.92, color=col, edgecolor=INK, lw=1.2,
               zorder=3, label=lab)
        for xx, vv in zip(x + off, v):
            if abs(vv) > 0.04:
                ax.text(xx, vv + 0.018, f"{vv:.2f}", ha="center", fontsize=8.6,
                        fontweight="bold")

    ax.axhline(0, color=INK, lw=1.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("same ", "same\n") for _, lab, _ in RANKS],
                       fontsize=10)
    ax.set_ylabel("Spearman ρ within rank\n(similarity vs gene relatedness)")
    ax.set_ylim(-0.09, 0.68)
    ax.set_title("The association survives holding ancestry fixed",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", ncol=1, handlelength=1.3, labelspacing=0.5,
              borderpad=0.2, fontsize=8.5)
    tidy(ax, 0, 2)
    tag(ax, "C")


def panel_D(ax, d):
    g = pd.DataFrame(d["rho_grid"])
    combos = [("d2star", 3), ("d2star", 4), ("d2star", 5),
              ("pearson", 3), ("pearson", 4), ("pearson", 5)]
    x = np.arange(len(combos))
    bw = 0.2

    for i, (s, lab, col) in enumerate(SERIES):
        v = [float(g[(g.metric == m) & (g.k == k) & (g.series == s)].rho.iloc[0])
             for m, k in combos]
        ax.bar(x + (i - 1.5) * bw, v, bw * 0.92, color=col, edgecolor=INK,
               lw=1.2, zorder=3, label=lab)

    ax.axhline(0, color=INK, lw=1.6, zorder=4)
    ax.axvline(2.5, color=GREY, ls=":", lw=1.8, zorder=2)
    ax.text(1.0, 0.53, "D2*", ha="center", fontsize=11, fontweight="bold")
    ax.text(4.0, 0.53, "Pearson", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.02, 0.99, "series as in (C)", transform=ax.transAxes, fontsize=9,
            style="italic", color=GREY, ha="left", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for _, k in combos], fontsize=10)
    ax.set_ylabel("Spearman ρ (all pairs)")
    ax.set_ylim(-0.09, 0.60)
    ax.set_title("The association does not depend on the statistic",
                 fontweight="bold", pad=10)
    tidy(ax, 0, 2)
    tag(ax, "D")


def main() -> None:
    set_style()
    d = json.loads(SRC.read_text())

    fig = plt.figure(figsize=(14.0, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42,
                          left=0.075, right=0.985, top=0.90, bottom=0.07)
    panel_A(fig.add_subplot(gs[0, 0]), d)
    panel_B(fig.add_subplot(gs[0, 1]), d)
    panel_C(fig.add_subplot(gs[1, 0]), d)
    panel_D(fig.add_subplot(gs[1, 1]), d)

    for stem in OUT_STEMS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"))
        print(f"wrote {stem.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()

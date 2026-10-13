#!/usr/bin/env python3
"""
Supplementary figure: the spacer distribution, one histogram per model.

Fig. 2a scores a generation by where its best-scoring proximal box lands. Read
that position as a spacer from the distal box and the whole distribution becomes
visible, not just the fraction inside the +-2 bp window. Real sigma70 promoters
put the spacer at 17 bp far more often than anywhere else; every model spreads it
uniformly across the scanned window.

The panels share a y-axis on purpose. The models look flat because they are flat,
and a per-panel rescale would hide exactly the quantity the figure is about.

Source: results/upstream_only/fig2x2/panelA_spacer.csv (the same export that
feeds panel A of figures/supp_promoter_geometry.pdf).

Output: figures/supp_spacer_histograms.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "upstream_only" / "fig2x2" / "panelA_spacer.csv"
OUT_STEMS = [ROOT / "results" / "supp_spacer_histograms",
             ROOT / "figures" / "supp_spacer_histograms"]

# same palette and display names as make_supp_promoter_geometry.py
C = {"blue": "#2166AC", "amber": "#E6A700", "green": "#1B7837", "red": "#B2182B",
     "grey": "#8C8C8C", "ink": "#1a1a1a", "blue_l": "#92C5DE"}
PANELS = [("real", "real $\\sigma^{70}$ promoters", C["ink"]),
          ("evo1", "Evo 1 (8k)", C["blue"]),
          ("evo2", "Evo 2", C["amber"]),
          ("hyenadna", "HyenaDNA", C["green"]),
          ("promogen2", "PromoGen2", C["red"])]

XS = np.arange(6, 29)
CANON = (14.5, 19.5)          # the canonical 15-19 bp window
CHANCE = 100 / 40             # the scan places the box anywhere in 40 offsets


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "axes.linewidth": 1.6, "axes.edgecolor": C["ink"],
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "axes.labelsize": 10, "axes.titlesize": 10.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
    })


def main() -> None:
    set_style()
    tab = (pd.read_csv(SRC)
             .pivot(index="spacer", columns="source", values="frac")
             .fillna(0) * 100)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(12.4, 2.9), sharey=True)
    for ax, (key, label, colour) in zip(axes, PANELS):
        y = tab.loc[XS, key]
        ax.axvspan(*CANON, color=C["blue_l"], alpha=0.22, lw=0, zorder=0)
        ax.bar(XS, y, width=0.86, color=colour, edgecolor=C["ink"],
               linewidth=0.5, zorder=3)
        ax.axhline(CHANCE, color=C["grey"], ls=":", lw=1.6, zorder=4)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(5.2, 28.8)
        ax.set_xticks([6, 10, 14, 18, 22, 26])
        # one number per panel: how much mass the canonical window holds
        inside = tab.loc[15:19, key].sum()
        ax.text(0.97, 0.93, f"15--19 bp: {inside:.0f}%".replace("--", "–"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                style="italic", color=C["ink"])

    axes[0].set_ylabel("generations (%)")
    axes[0].set_ylim(0, 23)
    axes[0].annotate("17 bp", xy=(16.4, tab.loc[17, "real"]),
                     xytext=(8.0, tab.loc[17, "real"] - 0.6), fontsize=9,
                     fontweight="bold", ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", lw=1.2, color=C["ink"]))
    axes[2].set_xlabel("spacer: end of distal box $\\rightarrow$ start of "
                       "proximal box (bp)")
    axes[-1].text(28.6, CHANCE + 1.4, "uniform placement", fontsize=8.4,
                  color=C["grey"], ha="right", va="bottom", style="italic")

    fig.tight_layout(w_pad=1.1)
    for stem in OUT_STEMS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(stem.with_suffix("." + ext))
        print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()

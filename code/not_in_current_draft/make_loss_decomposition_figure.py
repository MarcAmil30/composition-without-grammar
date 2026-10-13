#!/usr/bin/env python3
"""Where the fine-tuning loss sits, by region of the training record.

A mosaic: bar WIDTH is each region's share of tokens, bar HEIGHT its mean
cross-entropy, so bar AREA is that region's share of the total loss. The whole
plot therefore has the same total area as the single number reported on the
training curve, and shows what that number is made of.

Reads loss_decomposition.csv (written by code/loss_decomposition.py on Isambard).

    python make_figure.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

DISPLAY = {"cds": "CDS", "utr3": "3′UTR", "utr5_body": "5′UTR body",
           "utr5_sd": "SD", "delim": "delim.", "species": "species"}
# the regions the paper is about are highlighted; the rest are context
COLORS = {"cds": "#b9c6d6", "utr3": "#cdd6cb", "utr5_body": "#2f6fb0",
          "utr5_sd": "#e0a33e", "delim": "#d8d8d8", "species": "#c9c9c9"}
ORDER = ["cds", "utr3", "utr5_body", "utr5_sd", "delim", "species"]

INK, INK2, MUTED, GRID = "#1a1a1a", "#3d3d3d", "#8a8a8a", "#dcdcdc"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 1.4,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main() -> None:
    d = pd.read_csv(f"{HERE}/loss_decomposition.csv").set_index("region")
    overall = d.loc["ALL", "mean_ce_nats"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = 0.0
    for r in ORDER:
        if r not in d.index:
            continue
        w = d.loc[r, "tokens_pct"]
        h = d.loc[r, "mean_ce_nats"]
        ax.bar(x + w / 2, h, width=w * 0.985, color=COLORS[r],
               edgecolor="white", linewidth=1.1, zorder=3)
        if w > 4:                       # wide enough to letter inside
            ax.text(x + w / 2, h - 0.10, f"{h:.2f}", ha="center", va="top",
                    fontsize=8.4, color="white" if r == "utr5_body" else INK2,
                    fontweight="bold")
            ax.text(x + w / 2, 0.07, f"{DISPLAY[r]}\n{w:.0f}% of tokens",
                    ha="center", fontsize=8.0,
                    color="white" if r == "utr5_body" else INK2)
        x += w
    # the SD is only 2% of tokens, too narrow to letter -- annotate it
    x_sd = d.loc[["cds", "utr3", "utr5_body"], "tokens_pct"].sum() \
        + d.loc["utr5_sd", "tokens_pct"] / 2
    ax.annotate(f"SD  {d.loc['utr5_sd','mean_ce_nats']:.2f}",
                xy=(x_sd, d.loc["utr5_sd", "mean_ce_nats"]), xytext=(88, 1.60),
                fontsize=8.4, color="#a8761f", ha="center",
                arrowprops=dict(arrowstyle="-", color="#a8761f", lw=1.1,
                                shrinkA=0, shrinkB=2))

    ax.axhline(overall, color=INK, lw=1.6, ls=(0, (5, 2.5)), zorder=5)
    ax.text(1.5, overall + 0.05, f"reported training loss  {overall:.3f}",
            ha="left", fontsize=8.6, color=INK)

    # the species token is a tall spike on a sliver of tokens; clip for legibility
    ax.set_ylim(0, 1.78)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of tokens in the training record (%)")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Bar area = that region's share of the total loss",
                 fontsize=9.4, loc="left", pad=8, color=INK)

    for ext in ("pdf", "png"):
        fig.savefig(f"{HERE}/fig_loss_decomposition.{ext}", dpi=400,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_loss_decomposition.pdf / .png")


if __name__ == "__main__":
    main()

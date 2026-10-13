#!/usr/bin/env python3
"""
Supplementary figure: the motif is written, the geometry is not.

Three panels, all recomputed from the upstream-only benchmark exports in
results/upstream_only/fig2x2/:

  A  distal -> proximal box spacer implied by where each model put its box,
     against real sigma70 promoters (panelA_spacer.csv)
  B  the same distribution read as two numbers: exactly 17 bp, and the 15-19 bp
     canonical window, against the uniform-placement chance rate
  C  fraction of generations that contain a scoring "-10 box" at all, generated
     against a composition-preserving shuffle of the same generation
     (panelC_presence.csv)

The point of C is that presence is not the failing quantity: a PWM hit is found
in >93% of generations and in just as many shuffles of those generations, so the
box "exists" everywhere and means nothing. What is missing is where it sits.

Output: figures/supp_promoter_geometry.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "upstream_only" / "fig2x2"
OUT_STEMS = [ROOT / "results" / "supp_promoter_geometry",
             ROOT / "figures" / "supp_promoter_geometry"]

ORDER = ["evo1", "evo2", "hyenadna", "promogen2"]
DISPLAY = {"evo1": "Evo 1 (8k)", "evo2": "Evo 2", "hyenadna": "HyenaDNA",
           "promogen2": "PromoGen2", "real": "real promoters"}
# two-line versions for the crowded categorical axes of B and C
SHORT = {"evo1": "Evo 1\n(8k)", "evo2": "Evo 2", "hyenadna": "Hyena\nDNA",
         "promogen2": "Promo\nGen2", "real": "real\npromoters"}

C = {"blue": "#2166AC", "amber": "#E6A700", "green": "#1B7837", "red": "#B2182B",
     "grey": "#8C8C8C", "ink": "#1a1a1a", "blue_l": "#92C5DE"}
MODEL_COL = {"evo1": C["blue"], "evo2": C["amber"], "hyenadna": C["green"],
             "promogen2": C["red"]}
REAL_COL = C["ink"]
MINUS = "−"
ARROW = r"$\rightarrow$"

# the scan places the box anywhere in a 40 bp window, so uniform placement gives
# 1/40 for one spacer value and 5/40 for the 15-19 bp window
CHANCE_EXACT = 100 / 40
CHANCE_WINDOW = 5 * 100 / 40


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
        "axes.linewidth": 2.0, "axes.edgecolor": C["ink"],
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 6.0, "ytick.major.size": 6.0,
        "xtick.minor.width": 1.2, "ytick.minor.width": 1.2,
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "axes.labelsize": 11.5, "axes.labelweight": "bold", "axes.titlesize": 12.5,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "lines.solid_capstyle": "round",
    })


def tag(ax, letter, dx=-0.17, dy=1.14):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", color=C["ink"])


def tidy(ax, n_minor_x=0):
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.0)
        ax.spines[s].set_position(("outward", 6))
    if n_minor_x:
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n_minor_x))


def panel_A(ax, spacer):
    p = spacer.pivot(index="spacer", columns="source", values="frac").fillna(0) * 100
    xs = np.arange(6, 29)

    ax.axvspan(14.5, 19.5, color=C["blue_l"], alpha=0.22, lw=0, zorder=0)
    ax.axhline(CHANCE_EXACT, color=C["grey"], ls=":", lw=2.0, zorder=2)
    ax.text(28, CHANCE_EXACT + 0.4, "uniform placement", fontsize=9.5,
            color=C["grey"], ha="right", va="bottom", style="italic")

    ax.plot(xs, p.loc[xs, "real"], color=REAL_COL, lw=3.0, marker="o", ms=5,
            zorder=6, label=DISPLAY["real"])
    for m in ORDER:
        ax.plot(xs, p.loc[xs, m], color=MODEL_COL[m], lw=2.0, zorder=4,
                label=DISPLAY[m])

    ax.annotate(f"17 bp\n{p.loc[17, 'real']:.0f}%", xy=(17, p.loc[17, "real"]),
                xytext=(20.5, p.loc[17, "real"] + 1.5), fontsize=10.5,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=1.6, color=C["ink"]))

    ax.set_xlabel(f"spacer: end of distal box {ARROW} start of proximal box (bp)")
    ax.set_ylabel("generations (%)")
    ax.set_xticks(np.arange(6, 29, 3))
    ax.set_ylim(0, p.loc[xs, "real"].max() * 1.18)
    ax.set_title("Where the model put the box, read as a spacer", fontweight="bold", pad=10)
    ax.legend(loc="upper left", handlelength=1.6, labelspacing=0.35, borderpad=0.2)
    tidy(ax, 3)
    tag(ax, "A", dx=-0.11)


def panel_B(ax, spacer):
    p = spacer.pivot(index="spacer", columns="source", values="frac").fillna(0) * 100
    rows = ["real"] + ORDER
    exact = [p.loc[17, r] for r in rows]
    window = [p.loc[15:19, r].sum() for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    cols = [REAL_COL] + [MODEL_COL[m] for m in ORDER]

    ax.bar(x - w / 2, window, w, color=cols, edgecolor=C["ink"], lw=1.3, zorder=3)
    ax.bar(x + w / 2, exact, w, color="white", edgecolor=C["ink"], lw=1.3,
           hatch="////", zorder=3)
    lab_bbox = dict(boxstyle="square,pad=0.08", fc="white", ec="none")
    for xi, v in zip(x - w / 2, window):
        ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=9.5,
                fontweight="bold", zorder=6, bbox=lab_bbox)
    for xi, v in zip(x + w / 2, exact):
        ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=9.5,
                zorder=6, bbox=lab_bbox)

    ax.axhline(CHANCE_WINDOW, color=C["grey"], ls="--", lw=2.0, zorder=4)
    ax.text(len(rows) - 0.4, CHANCE_WINDOW + 1.0, "chance, 15–19 bp", fontsize=9,
            color=C["grey"], ha="right", va="bottom", style="italic")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C["grey"], edgecolor=C["ink"], label="15–19 bp"),
                       Patch(facecolor="white", edgecolor=C["ink"], hatch="////",
                             label="exactly 17 bp")],
              loc="upper right", handlelength=1.5, borderpad=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[r] for r in rows], fontsize=9.5)
    ax.set_ylabel("generations (%)")
    ax.set_ylim(0, 72)
    ax.set_title("Canonical spacing is not reproduced", fontweight="bold", pad=10)
    tidy(ax)
    tag(ax, "B", dx=-0.16)


def panel_C(ax, presence):
    d = presence.set_index("model").loc[ORDER].reset_index()
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d.generated, w, color=[MODEL_COL[m] for m in d.model],
           edgecolor=C["ink"], lw=1.3, zorder=3)
    ax.bar(x + w / 2, d.shuffled, w, color="white", edgecolor=C["ink"], lw=1.3,
           hatch="////", zorder=3)
    for xi, v in zip(x - w / 2, d.generated):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9.5, fontweight="bold")
    for xi, v in zip(x + w / 2, d.shuffled):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C["grey"], edgecolor=C["ink"], label="generated"),
                       Patch(facecolor="white", edgecolor=C["ink"], hatch="////",
                             label="same sequence, shuffled")],
              loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.5,
              borderpad=0.2, ncol=1)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in d.model], fontsize=9.5)
    ax.set_ylabel(f"generations with a\nscoring {MINUS}10-like box (%)")
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("A box is always present — in the shuffle too", fontweight="bold", pad=10)
    tidy(ax)
    tag(ax, "C", dx=-0.20)


def main() -> None:
    set_style()
    spacer = pd.read_csv(SRC / "panelA_spacer.csv")
    presence = pd.read_csv(SRC / "panelC_presence.csv")

    fig = plt.figure(figsize=(15.0, 4.9))
    gs = fig.add_gridspec(1, 12, wspace=2.1, left=0.05, right=0.99,
                          top=0.88, bottom=0.19)
    panel_A(fig.add_subplot(gs[0, 0:5]), spacer)
    panel_B(fig.add_subplot(gs[0, 5:9]), spacer)
    panel_C(fig.add_subplot(gs[0, 9:12]), presence)

    for stem in OUT_STEMS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"))
        print(f"wrote {stem.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()

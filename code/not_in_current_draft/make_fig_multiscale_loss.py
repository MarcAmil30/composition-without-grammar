#!/usr/bin/env python3
"""
Overview figure: the multiscale contextual-information loss.

Companion to the PROK and C3P overview schematics -- same visual language (two
panels, same palette, same box style), so the three can sit together.

  A  WHAT IS SCORED   One 20 bp window W is sampled from the upstream. Its mean
                      log-likelihood under the intact context,
                      s(W | C_real), is the quantity the loss acts on --
                      everything to the left of W is the context.

  B  THE OBJECTIVE    The same window is re-scored under three ablations that
                      each destroy one scale of context: codon-permuted CDS
                      (distant, coding), truncated distant context (long range),
                      block-shuffled upstream prefix (local, regulatory). A
                      margin hinge asks the intact context to beat every
                      ablation by at least m nats/base.

Schematic only -- no data. Text is deliberately minimal; the loss, lambda, m and
the alternation schedule live in the caption.

    python3 code/make_fig_multiscale_loss.py

Outputs: figures/fig_multiscale_loss.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
INK = "#1a1a1a"
GREY = "#8C8C8C"
FAINT = "#EDEDED"

DNA = "#2166AC"       # upstream
DNA_FILL = "#D6E3F0"
WIN = "#08306B"       # the scored window W
PRO = "#D95F02"       # CDS
PRO_FILL = "#FBE0CC"
SPC = "#762A83"       # prompt token
SPC_FILL = "#E7D4E8"
POS = "#1B7837"       # intact context
MODEL_FILL = "#F2F2F2"


# --------------------------------------------------------------------------- #
# style  (shared with the PROK / C3P overview figures)
# --------------------------------------------------------------------------- #
def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "text.color": INK,
    })


# --------------------------------------------------------------------------- #
# drawing primitives
# --------------------------------------------------------------------------- #
def box(ax, x0, x1, yc, h, label, *, edge=INK, fill="white", fs=10.5, lw=1.8,
        weight="normal", color=None):
    ax.add_patch(FancyBboxPatch(
        (x0, yc - h / 2), x1 - x0, h,
        boxstyle="round,pad=0.02,rounding_size=0.9",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text((x0 + x1) / 2, yc, label, ha="center", va="center",
            fontsize=fs, color=color or INK, weight=weight, zorder=4,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, *, color=GREY, lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.55",
                                color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0), zorder=2)


def cds_arrow(ax, x0, x1, yc, h, *, fill=PRO_FILL, edge=PRO, lw=1.6,
              blocks=None):
    """The CDS as a gene arrow; `blocks` draws internal codon blocks."""
    tip = min(2.2, (x1 - x0) * 0.16)
    ax.add_patch(Polygon(
        [(x0, yc - h / 2), (x1 - tip, yc - h / 2), (x1, yc),
         (x1 - tip, yc + h / 2), (x0, yc + h / 2)],
        closed=True, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3))
    if blocks is not None:
        edges = np.linspace(x0 + 0.6, x1 - tip - 0.6, len(blocks) + 1)
        for k, b in enumerate(blocks):
            ax.add_patch(Rectangle((edges[k], yc - h / 2 + 0.5),
                                   edges[k + 1] - edges[k] - 0.35, h - 1.0,
                                   facecolor=edge, alpha=0.18 + 0.16 * b,
                                   edgecolor="none", zorder=4))


def seq_bar(ax, x0, x1, yc, h, *, fill, edge, lw=1.6, blocks=None, hatch=None,
            ls="solid"):
    ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, facecolor=fill,
                           edgecolor=edge, linewidth=lw, hatch=hatch,
                           linestyle=ls, zorder=3))
    if blocks is not None:
        edges = np.linspace(x0 + 0.5, x1 - 0.5, len(blocks) + 1)
        for k, b in enumerate(blocks):
            ax.add_patch(Rectangle((edges[k], yc - h / 2 + 0.5),
                                   edges[k + 1] - edges[k] - 0.3, h - 1.0,
                                   facecolor=edge, alpha=0.18 + 0.16 * b,
                                   edgecolor="none", zorder=4))


# geometry shared by both panels: one context ribbon
#   species | CDS | upstream (prefix, W, rest) | downstream (rest, W', rest)
X_TOK, X_CDS0, X_CDS1 = 3.0, 6.5, 25.0
X_PRE1, X_W1, X_UP1 = 36.0, 42.0, 46.0
X_DN0, X_DW0, X_DW1, X_DN1 = 46.8, 53.0, 59.0, 62.0
CODONS = [3, 0, 2, 1, 3, 1, 0, 2]
PREFIX = [1, 2, 0, 3, 1, 2]
DSTREAM = [2, 0, 3, 1, 2, 0]


def ribbon(ax, yc, h, *, codons=CODONS, prefix=PREFIX, dstream=DSTREAM,
           cut=False, label_w=False, ghost_w=False):
    """The +#CDS+$upstream*downstream context, with the scored window picked
    out. `ghost_w` draws the alternative window in the downstream flank."""
    if cut:
        # truncated distant context: the CDS-proximal half is gone
        ax.plot([X_TOK, X_CDS1 - 0.5], [yc, yc], color=GREY, lw=1.2,
                ls=(0, (2, 2)), zorder=2)
        cds_arrow(ax, (X_CDS0 + X_CDS1) / 2 + 2.0, X_CDS1, yc, h,
                  blocks=codons[len(codons) // 2:])
    else:
        seq_bar(ax, X_TOK, X_CDS0 - 0.7, yc, h, fill=SPC_FILL, edge=SPC, lw=1.3)
        cds_arrow(ax, X_CDS0, X_CDS1, yc, h, blocks=codons)

    # upstream flank, holding the scored window
    seq_bar(ax, X_CDS1 + 0.8, X_PRE1, yc, h, fill=DNA_FILL, edge=DNA, lw=1.4,
            blocks=prefix)
    seq_bar(ax, X_PRE1, X_W1, yc, h, fill=WIN, edge=WIN, lw=1.4)
    seq_bar(ax, X_W1, X_UP1, yc, h, fill=DNA_FILL, edge=DNA, lw=1.4)
    if label_w:
        ax.text((X_PRE1 + X_W1) / 2, yc, "W", ha="center", va="center",
                fontsize=9.5, color="white", weight="bold", zorder=5)

    # downstream flank
    if ghost_w:
        seq_bar(ax, X_DN0, X_DW0, yc, h, fill=DNA_FILL, edge=DNA, lw=1.4,
                blocks=dstream[:3])
        seq_bar(ax, X_DW0, X_DW1, yc, h, fill="white", edge=WIN, lw=1.4,
                ls=(0, (2.5, 1.8)))
        ax.text((X_DW0 + X_DW1) / 2, yc, "W", ha="center", va="center",
                fontsize=9.5, color=WIN, weight="bold", zorder=5)
        seq_bar(ax, X_DW1, X_DN1, yc, h, fill=DNA_FILL, edge=DNA, lw=1.4)
    else:
        seq_bar(ax, X_DN0, X_DN1, yc, h, fill=DNA_FILL, edge=DNA, lw=1.4,
                blocks=dstream)


# --------------------------------------------------------------------------- #
# panel A -- what is scored
# --------------------------------------------------------------------------- #
def panel_a(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(7, 24)
    ax.set_aspect("equal")
    ax.axis("off")

    y, h = 14.0, 3.6
    ribbon(ax, y, h, label_w=True, ghost_w=True)

    # what each stretch is
    ax.text((X_TOK + X_CDS0 - 0.7) / 2, y + 2.6, "species", ha="center",
            va="bottom", fontsize=9.5, color=SPC)
    ax.text((X_CDS0 + X_CDS1) / 2, y + 2.6, "CDS", ha="center", va="bottom",
            fontsize=10, color=PRO)
    for a, b, lab in [(X_CDS1 + 0.8, X_UP1, "upstream"),
                      (X_DN0, X_DN1, "downstream")]:
        ax.plot([a, b], [y + 5.4, y + 5.4], color=DNA, lw=1.2)
        ax.plot([a, a], [y + 4.8, y + 5.4], color=DNA, lw=1.2)
        ax.plot([b, b], [y + 4.8, y + 5.4], color=DNA, lw=1.2)
        ax.text((a + b) / 2, y + 6.2, lab, ha="center", va="bottom",
                fontsize=10, color=DNA)

    # the context / window split
    ax.annotate("", xy=(X_TOK, y - 3.2), xytext=(X_PRE1, y - 3.2),
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=1.2))
    ax.plot([X_TOK, X_TOK], [y - 3.2, y - 2.6], color=GREY, lw=1.2)
    ax.plot([X_PRE1, X_PRE1], [y - 3.2, y - 2.6], color=GREY, lw=1.2)
    ax.text((X_TOK + X_PRE1) / 2, y - 4.0, "context  $C$", ha="center", va="top",
            fontsize=9.5, color=GREY)
    ax.text((X_PRE1 + X_DW1) / 2, y - 3.0, "20 bp window, either flank",
            ha="center", va="top", fontsize=9, color=WIN)

    # scoring
    arrow(ax, X_DN1 + 1.6, y, X_DN1 + 5.6, y, color=GREY)
    box(ax, X_DN1 + 6.0, X_DN1 + 20.0, y, 9.0, "Evo", edge=INK, fill=MODEL_FILL,
        fs=13, lw=2.0, weight="bold")
    arrow(ax, X_DN1 + 20.6, y, X_DN1 + 24.0, y, color=GREY)
    ax.text(X_DN1 + 25.0, y, r"$s(W \mid C)$", ha="left", va="center",
            fontsize=12.5)
    ax.text(X_DN1 + 25.0, y - 2.6, "nats / base", ha="left", va="top",
            fontsize=9, color=GREY)


# --------------------------------------------------------------------------- #
# panel B -- the objective
# --------------------------------------------------------------------------- #
def panel_b(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 39)
    ax.set_aspect("equal")
    ax.axis("off")

    h = 3.0
    rows_y = [33.0, 24.0, 18.0, 12.0]
    scores = [10.0, 4.0, 5.1, 3.2]           # schematic s(W | .), intact largest

    perm = [1, 3, 0, 2, 2, 0, 3, 1]          # codons reordered
    shuf = [3, 0, 2, 1, 0, 3]                # prefix blocks reordered

    # intact
    ribbon(ax, rows_y[0], h, label_w=True)
    # ablation 1: codon-permuted CDS
    ribbon(ax, rows_y[1], h, codons=perm, label_w=True)
    # ablation 2: truncated distant context
    ribbon(ax, rows_y[2], h, cut=True, label_w=True)
    # ablation 3: block-shuffled upstream prefix
    ribbon(ax, rows_y[3], h, prefix=shuf, label_w=True)

    labels = ["intact context",
              "codon-permuted CDS",
              "truncated distant context",
              "block-shuffled prefix"]
    colors = [POS, GREY, GREY, GREY]
    for yr, lab, c in zip(rows_y, labels, colors):
        ax.text(X_TOK - 1.5, yr, lab, ha="right", va="center", fontsize=9.5,
                color=c)

    ax.plot([X_TOK - 1.0, X_TOK - 1.0], [rows_y[-1] - 1.6, rows_y[1] + 1.6],
            color=GREY, lw=1.3)
    ax.text(X_TOK - 2.5, rows_y[-1] - 3.6, "one sampled per update",
            ha="right", va="center", fontsize=9, color=GREY, style="italic")

    # ---- scores ------------------------------------------------------------ #
    x_bar = 67.0
    for r, yr in enumerate(rows_y):
        arrow(ax, X_DN1 + 1.5, yr, x_bar - 1.0, yr, color=GREY, lw=1.4)
        c = POS if r == 0 else GREY
        ax.add_patch(Rectangle((x_bar, yr - 1.25), scores[r], 2.5, facecolor=c,
                               alpha=1.0 if r == 0 else 0.45, edgecolor="none",
                               zorder=3))
    ax.text(x_bar, rows_y[0] + 2.6, r"$s(W \mid C)$", ha="left", va="bottom",
            fontsize=10.5)

    # ---- the margin -------------------------------------------------------- #
    x_true = x_bar + scores[0]
    x_abl = x_bar + max(scores[1:])
    ax.plot([x_true, x_true], [rows_y[0] - 2.4, rows_y[2] + 2.6], color=POS,
            lw=1.1, ls=(0, (2, 2)), zorder=2)
    ax.plot([x_abl, x_abl], [rows_y[2] + 1.6, rows_y[2] + 2.6], color=GREY,
            lw=1.1, ls=(0, (2, 2)), zorder=2)
    ax.annotate("", xy=(x_true, rows_y[2] + 2.6), xytext=(x_abl, rows_y[2] + 2.6),
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.2,head_length=0.4",
                                color=INK, linewidth=1.3), zorder=4)
    ax.text((x_true + x_abl) / 2, rows_y[2] + 3.4, r"$\geq m$", ha="center",
            va="bottom", fontsize=10.5)

    # ---- the hinge --------------------------------------------------------- #
    yc = (rows_y[0] + rows_y[-1]) / 2
    ax.plot([79.0, 81.0, 81.0, 79.0],
            [rows_y[0] + 1.5, rows_y[0] + 1.5, rows_y[-1] - 1.5, rows_y[-1] - 1.5],
            color=GREY, lw=1.4, solid_joinstyle="round")
    arrow(ax, 81.0, yc, 84.5, yc, color=GREY)
    box(ax, 85.0, 99.0, yc, 8.0, "margin\nhinge", edge=INK, fill="white",
        fs=11, lw=2.0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    set_style()
    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4),
                             gridspec_kw={"height_ratios": [17, 34]})
    fig.subplots_adjust(hspace=0.02, left=0.01, right=0.99, top=0.97, bottom=0.02)

    panel_a(axes[0])
    panel_b(axes[1])

    for ax, letter in zip(axes, "AB"):
        ax.text(-0.01, 1.0, letter, transform=ax.transAxes, fontsize=15,
                weight="bold", ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_multiscale_loss.{ext}")
    print(f"wrote {out_dir}/fig_multiscale_loss.pdf/.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Appendix figure: Shine--Dalgarno version of Fig. 2c.

Same construction as the main-text proximal-box panel, applied to the SD task:
scan each generated continuation with the SD PWM, keep the highest-scoring
six-base window, and count how many of those bases match that gene's annotated
SD core. The positive control is the identical scan run on the real
continuation, which recovers the annotated SD in 504 of 681 genes and therefore
scores 5.0/6 rather than 6/6.

Reads ``highest_pwm_base_matches.csv`` and ``highest_pwm_positive_control.csv``
and writes ``figures/supp_sd_matches.{pdf,png}``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "upstream_only" / "fig2x2"
OUT = ROOT / "figures" / "supp_sd_matches"

ORDER = ["evo1", "evo1_131k", "evo2", "hyenadna", "promogen2"]
DISPLAY = {
    "evo1": "Evo 1\n(8k)",
    "evo1_131k": "Evo 1\n(131k)",
    "evo2": "Evo 2",
    "hyenadna": "HyenaDNA",
    "promogen2": "PromoGen2",
}
COLORS = {
    "evo1": "#2f6fb0",
    "evo1_131k": "#78a9dc",
    "evo2": "#3f8f5e",
    "hyenadna": "#a34ba3",
    "promogen2": "#9a7b28",
}
INK, INK2, MUTED, GRID = "#1a1a1a", "#3d3d3d", "#8a8a8a", "#dcdcdc"
REAL_COL = "#1a1a1a"


def lighten(colour: str, factor: float = 0.55):
    r, g, b = mcolors.to_rgb(colour)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def main() -> None:
    matches = pd.read_csv(HERE / "highest_pwm_base_matches.csv")
    matches = matches[matches.task == "generate_sd"]
    reference = pd.read_csv(HERE / "highest_pwm_positive_control.csv")
    real = float(reference[reference.task == "generate_sd"].mean_matches.iloc[0])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    figure, ax = plt.subplots(figsize=(7.1, 3.6))
    models = [m for m in ORDER if m in set(matches.model)]
    x = np.arange(len(models))
    width = 0.21

    ax.axhline(real, color=REAL_COL, lw=2.2, ls=(0, (5, 2.5)), zorder=4)
    ax.text(len(models) - 0.45, real - 0.12,
            f"real continuation   {real:.1f}", fontsize=8.6,
            color=REAL_COL, ha="right", va="top")

    for kind, background, offset in (("generated", "leaked", -1.0),
                                     ("generated", "clean", 0.0),
                                     ("shuffled", "leaked", 1.0)):
        subset = matches[(matches.kind == kind)
                         & (matches.background == background)].set_index("model")
        values = np.asarray([subset.loc[m].mean_matches for m in models])
        generated = kind == "generated"
        ax.bar(x + offset * width, values, width * 0.9,
               color=([COLORS[m] if background == "leaked" else lighten(COLORS[m])
                       for m in models] if generated else "none"),
               edgecolor="white" if generated else INK2,
               linewidth=1.2 if generated else 1.4,
               hatch=None if generated else "\\\\", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], fontsize=8.4)
    ax.set_ylabel("native bases in the best-scoring\nShine–Dalgarno written (of 6)")
    ax.set_ylim(0, 6)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(handles=[
        Patch(facecolor=MUTED, edgecolor="white", label="real prompt, unchanged (leaked)"),
        Patch(facecolor=lighten(MUTED), edgecolor="white",
              label="boxes kept, background\nshuffled (clean)"),
        Patch(facecolor="none", edgecolor=INK2, hatch="\\\\",
              label="generated output shuffled\n(negative control)"),
    ], frameon=False, fontsize=7.4, loc="upper left", ncol=3,
        bbox_to_anchor=(0.0, 1.02), handlelength=1.5, columnspacing=1.2)

    for extension in ("pdf", "png"):
        figure.savefig(OUT.with_suffix(f".{extension}"), dpi=300,
                       bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()

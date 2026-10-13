#!/usr/bin/env python3
"""Task 4 figure -- publication style. Panels A/B, no titles, thick axes,
model names only on the category axis, value labels on both panels."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path,
                    default=REPO_ROOT / "data" / "retrieval" / "results")
parser.add_argument("--pdf", type=Path,
                    default=REPO_ROOT / "figures" / "fig_retrieval.pdf")
parser.add_argument("--png", type=Path,
                    default=REPO_ROOT / "figures" / "fig_retrieval.png")
args = parser.parse_args()
RES = args.results
args.pdf.parent.mkdir(parents=True, exist_ok=True)
args.png.parent.mkdir(parents=True, exist_ok=True)
N_CAND = 100

MODELS = [
    ("C3P",      "c3p",      "embed_cos",      "#2a78d6"),
    ("Evo 1",    "evo1",     "score_cond_pmi", "#e87ba4"),
    ("Evo 2",    "evo2",     "score_cond_pmi", "#1baf7a"),
    ("Evo 1-FT", "prok",     "score_cond_pmi", "#eb6834"),  # the paper calls it "the fine-tuned Evo"
    ("HyenaDNA", "hyenadna", "score_cond_pmi", "#4a3aa7"),
]

INK, MUTED = "#000000", "#666666"
LW = 1.15
plt.rcParams.update({
    "font.size": 7.2, "font.family": "sans-serif",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": LW,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": LW, "ytick.major.width": LW,
    "xtick.minor.width": LW * 0.7, "ytick.minor.width": LW * 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

data = []
for name, d, variant, colour in MODELS:
    rk = pd.read_parquet(RES / d / f"{variant}_ranks.parquet")["rank"].to_numpy()
    with open(RES / d / f"{variant}_summary.json") as handle:
        sm = json.load(handle)
    data.append(dict(name=name, colour=colour, ranks=rk, sm=sm))

names = [d["name"] for d in data]
colours = [d["colour"] for d in data]
fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.25))

# ------------------------------------------------------------------ A
ax = axes[0]
bp = ax.boxplot([d["ranks"] for d in data], positions=range(len(data)),
                widths=0.62, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=1.6),
                whiskerprops=dict(color=INK, lw=1.1),
                capprops=dict(color=INK, lw=1.1))
for patch, c in zip(bp["boxes"], colours):
    patch.set(facecolor=c, edgecolor=INK, linewidth=1.0)
ax.axhline(N_CAND / 2, color=MUTED, ls=(0, (4, 3)), lw=1.1, zorder=0)
# label sits below the line on the left, where the C3P column is empty, so the
# dashes never run through the text
ax.text(-0.46, N_CAND / 2 * 1.45, "chance = 50", color=MUTED,
        fontsize=6.0, ha="left", va="center")
# C3P's box has zero height (Q1 = median = Q3 = 1); the marker keeps it visible.
# The median is printed beside it, mirroring the value labels in panel B; a white
# stroke keeps it legible over both the coloured box and the white surface.
for i, d in enumerate(data):
    m = d["sm"]["median_rank"]
    ax.plot([i], [m], marker="D", ms=4.2, color=d["colour"],
            mec="white", mew=1.2, zorder=6)
    ax.text(i + 0.21, m, f"{m:.0f}", ha="left", va="center", fontsize=6.6,
            color=INK, zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
ax.set_yscale("log")
ax.set_ylim(0.8, 130)
ax.set_yticks([1, 3, 10, 30, 100])
ax.set_yticklabels(["1", "3", "10", "30", "100"])
ax.set_ylabel("Rank of true upstream")

# ------------------------------------------------------------------ B
ax = axes[1]
top1 = np.array([d["sm"]["top1"] * 100 for d in data])
lo = np.array([d["sm"]["top1_ci95"][0] * 100 for d in data])
hi = np.array([d["sm"]["top1_ci95"][1] * 100 for d in data])
ax.bar(range(len(data)), top1, width=0.62, color=colours,
       edgecolor=INK, linewidth=1.0, zorder=2)
ax.errorbar(range(len(data)), top1, yerr=[top1 - lo, hi - top1], fmt="none",
            ecolor=INK, elinewidth=1.1, capsize=3, zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.5, 220)
ax.set_yticks([1, 3, 10, 30, 100])
ax.set_yticklabels(["1", "3", "10", "30", "100"])
ax.set_ylabel("Top-1 accuracy (%)")
ax.axhline(100 / N_CAND, color=MUTED, ls=(0, (4, 3)), lw=1.1, zorder=1)
# every bar is tall, so the only clear space for this label is outside the axes
ax.text(len(data) - 0.42, 100 / N_CAND, "chance", color=MUTED, fontsize=6.4,
        ha="left", va="center", clip_on=False)
for i, (v, h) in enumerate(zip(top1, hi)):
    ax.text(i, h * 1.22, f"{v:.1f}", ha="center", va="bottom",
            fontsize=6.6, color=INK)

for a in axes:
    a.set_xticks(range(len(data)))
    a.set_xticklabels(names, fontsize=6.8, rotation=30, ha="right",
                      rotation_mode="anchor")
    a.tick_params(axis="x", length=0)
for a in axes:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

for a, letter in zip(axes, "AB"):
    a.text(-0.26, 1.10, letter, transform=a.transAxes, fontsize=10,
           fontweight="bold", va="top", ha="left", color=INK)

fig.tight_layout(w_pad=1.8)
fig.savefig(args.pdf, bbox_inches="tight")
fig.savefig(args.png, dpi=300, bbox_inches="tight")
print(f"written: {args.pdf}\nwritten: {args.png}")

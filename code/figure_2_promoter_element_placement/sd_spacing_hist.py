"""
Publication figure: distribution of Shine-Dalgarno -> start-codon spacing
across the 681-gene E. coli upstream_only benchmark.

spacing = number of bases between the last base of the SD motif and the
first base of the start codon. Identical in the leaked and clean sets
(it is an annotation column, not a sequence column).
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = f"{ICBINB_ROOT}/data/benchmark/upstream_only"
OUT = os.path.join(BASE, "sd_spacing")
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

CORE_LO, CORE_HI = 5, 8          # the band holding ~72% of genes
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"
CORE, TAIL, GRID = "#2f6fb0", "#a9c4de", "#e4e4e4"

df = pd.read_csv(os.path.join(BASE, "prompts_681_leaked.csv"))
chk = pd.read_csv(os.path.join(BASE, "prompts_681_clean.csv"))
assert (df.sd_spacing.values == chk.sd_spacing.values).all(), "leaked/clean disagree"

sp = df.sd_spacing.astype(int)
n = len(sp)
lo, hi = int(sp.min()), int(sp.max())
xs = list(range(lo, hi + 1))
counts = sp.value_counts().reindex(xs).fillna(0).astype(int)

# ---- table out first, so the plot can be restyled without recomputing ----
tab = pd.DataFrame({
    "sd_spacing_bp": xs,
    "n_genes": counts.values,
    "fraction": counts.values / n,
    "cumulative_fraction": np.cumsum(counts.values) / n,
    "in_core_band": [int(CORE_LO <= x <= CORE_HI) for x in xs],
})
tab.to_csv(os.path.join(OUT, "sd_spacing_distribution.csv"), index=False)

stats = dict(
    n_genes=n, min=lo, max=hi,
    mode=int(counts.idxmax()), mode_n=int(counts.max()), mode_frac=counts.max() / n,
    median=float(sp.median()), mean=float(sp.mean()), sd=float(sp.std()),
    frac_5_to_8=float(((sp >= 5) & (sp <= 8)).mean()),
    frac_4_to_9=float(((sp >= 4) & (sp <= 9)).mean()),
    iqr_lo=float(sp.quantile(.25)), iqr_hi=float(sp.quantile(.75)),
)
pd.Series(stats).to_frame("value").to_csv(os.path.join(OUT, "sd_spacing_summary.csv"))

# =====================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(5.2, 3.1))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)

cols = [CORE if CORE_LO <= x <= CORE_HI else TAIL for x in xs]
ax.bar(xs, counts.values, width=0.76, color=cols,
       edgecolor="white", linewidth=0.9, zorder=3)

for x, c in zip(xs, counts.values):
    ax.text(x, c + 4, f"{c}", ha="center", va="bottom", fontsize=7.2,
            color=INK2 if CORE_LO <= x <= CORE_HI else MUTED, zorder=4)

# core-band bracket
Y = 160
ax.annotate("", xy=(CORE_LO - 0.45, Y), xytext=(CORE_HI + 0.45, Y),
            arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3",
                            lw=0.8, color=MUTED, shrinkA=0, shrinkB=0))
ax.text((CORE_LO + CORE_HI) / 2, Y + 4,
        f"{stats['frac_5_to_8']*100:.0f}% of genes", ha="center", va="bottom",
        fontsize=7.6, color=INK2)

ax.set_xlabel("SD-to-start-codon spacing (bp)", fontsize=8.8, color=INK)
ax.set_ylabel("genes", fontsize=8.8, color=INK)
ax.set_xticks(xs)
ax.set_xlim(lo - 0.8, hi + 0.8)
ax.set_ylim(0, 182)
ax.set_yticks([0, 50, 100, 150])
ax.tick_params(colors=INK2, length=3, width=0.7, pad=2)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
for sp_ in ("left", "bottom"):
    ax.spines[sp_].set_color("#bdbdbd")

ax.set_title("The Shine-Dalgarno sits in a band, not at a fixed position",
             fontsize=9.2, color=INK, pad=17, loc="left")
ax.text(0, 1.018,
        f"E. coli upstream-only benchmark   |   n = {n} genes   |   "
        f"median {stats['median']:.0f} bp, range {lo}-{hi} bp",
        transform=ax.transAxes, fontsize=7.3, color=MUTED, va="bottom")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "plots", f"sd_spacing_histogram.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)

print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print(pd.Series(stats).to_string())
print("\nwrote ->", OUT)

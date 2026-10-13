#!/usr/bin/env python3
"""
Supplementary panel: WHERE the CDS-perturbation KL sits along the upstream.

Reviewer N2P1 asked for the distribution of CDS sensitivity across positions
rather than the 200-position sum alone, "particularly at annotated elements".
This renders KL(C0 || Ck) per position for every perturbation, marks the
Shine-Dalgarno / start-codon region, and reports enrichment there against the
first 50 positions (the bases written immediately after the CDS prompt).

Inputs (all local, no cluster needed)
  data/cds_prompted/results/kl_{species}_C{1..6}_perposition.npy   (200,)
  results/cds_prompted/cds_fig_stats_testset.npz -> {species}_natural (200,4)

Outputs
  figures/supp_kl_perposition.{pdf,png}
  results/supp_kl_perposition_stats.csv
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
KL = ROOT / "data" / "cds_prompted" / "results"
NPZ = ROOT / "results" / "cds_prompted" / "cds_fig_stats_testset.npz"

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.linewidth': 1.6, 'axes.labelsize': 9.5, 'axes.titlesize': 10,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.major.size': 4.5, 'ytick.major.size': 4.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.frameon': False, 'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
})

SPN = {'ecoli': 'E. coli', 'bsubtilis': 'B. subtilis'}
CONDS = [
    ('C1', 'synonymous codons permuted', '#9E9E9E'),
    ('C2', 'codon order shuffled',       '#EF9A00'),
    ('C3', 'a different gene',           '#C62828'),
    ('C4', 'CDS removed',                '#000000'),
    ('C5', 'codons 1–30 shuffled',  '#26A69A'),
    ('C6', 'codons 31–end shuffled', '#5C6BC0'),
]
L = 200
SD_LO, SD_HI = 186, 196       # SD box and start codon, fixed for both species
EDGE_LO, EDGE_HI = 1, 50      # bases written first, right after the CDS prompt

z = np.load(NPZ, allow_pickle=True)
rows = []

fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True)
x = np.arange(1, L + 1)

for ax, sp in zip(axs, ['ecoli', 'bsubtilis']):
    for c, lab, col in CONDS:
        f = KL / f'kl_{sp}_{c}_perposition.npy'
        if not f.exists():
            continue
        a = np.load(f)
        ax.plot(x, a, color=col, lw=1.3, label=f'{c}  {lab}', zorder=3)

        tot = a.sum()
        def share(lo, hi):
            return 100 * a[lo - 1:hi].sum() / tot
        def width(lo, hi):
            return 100 * (hi - lo + 1) / L
        rows.append(dict(
            species=sp, condition=c, label=lab,
            total_bits=tot, bits_per_position=tot / L,
            peak_position=int(np.argmax(a) + 1), peak_bits=a.max(),
            sd_share_pct=share(SD_LO, SD_HI),
            sd_enrichment=share(SD_LO, SD_HI) / width(SD_LO, SD_HI),
            edge_share_pct=share(EDGE_LO, EDGE_HI),
            edge_enrichment=share(EDGE_LO, EDGE_HI) / width(EDGE_LO, EDGE_HI),
        ))

    top = ax.get_ylim()[1]
    ax.axvspan(SD_LO, SD_HI, color='#FFD54F', alpha=0.45, lw=0, zorder=0)
    ax.axvspan(EDGE_LO, EDGE_HI, color='#90CAF9', alpha=0.22, lw=0, zorder=0)
    ax.set_ylim(top=top * 1.12)
    ax.annotate('first 50 bases written', xy=(26, top * 1.10), fontsize=7.2,
                ha='center', va='top', color='#1565C0')
    ax.annotate('SD /\nstart', xy=(191, top * 1.10), fontsize=7.2,
                ha='center', va='top', color='#8D6E00')
    ax.set_ylabel('KL$(C_0\\,\\|\\,C_k)$  (bits)')
    ax.set_title(SPN[sp], loc='left', style='italic')
    ax.set_xlim(1, L)

h, lb = axs[0].get_legend_handles_labels()
fig.legend(h, lb, ncol=3, loc='lower center',
           bbox_to_anchor=(0.5, 0.0), fontsize=7.6,
           columnspacing=1.4, handlelength=1.6)
axs[-1].set_xlabel('position in the 200 bp upstream   '
                   '(1 = most distal, 200 = adjacent to the start codon)')
fig.tight_layout(pad=0.6, rect=(0, 0.09, 1, 1))

for ext in ('pdf', 'png'):
    fig.savefig(ROOT / 'figures' / f'supp_kl_perposition.{ext}', dpi=300)

df = pd.DataFrame(rows)
out = ROOT / 'results' / 'supp_kl_perposition_stats.csv'
df.to_csv(out, index=False)
print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print(f'\nwrote figures/supp_kl_perposition.pdf/.png and {out.name}')

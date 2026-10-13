import csv, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATS = os.path.join(ROOT, 'results', 'cds_prompted', 'cds_fig_stats_testset.npz')
RESULTS, FIGURES = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'figures')
os.makedirs(FIGURES, exist_ok=True)

D = np.load(STATS, allow_pickle=True)
print(f'{len(D.files)} cached arrays')

# ---- publication style: thick axes, real fonts, editable vector text ----
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

BASE_C = {'A': '#2E7D32', 'C': '#1565C0', 'G': '#EF9A00', 'T': '#C62828'}
SPN = {'ecoli': 'E. coli', 'bsubtilis': 'B. subtilis'}

def ital(sp):
    """Species name in maths italic (f-strings cannot contain backslashes)."""
    return '$\\it{' + SPN[sp].replace(' ', '\\ ') + '}$'
C_OK, C_BAD, C_GREY = '#2166AC', '#B2182B', '#7A7A7A'

def panel_tag(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top', ha='left')

TOKEN = {'ecoli': '+$', 'bsubtilis': '@$'}
OTHER_SP = {'ecoli': 'bsubtilis', 'bsubtilis': 'ecoli'}

def draw_A(ax, sp):
    """Linear axes. On log axes the counts quantise into pure vertical and
    horizontal lines -- a few hundred sequences give small integer counts, so
    the frequencies land on a lattice that the log transform stretches into
    visible stripes near the origin. Linear compresses that lattice back to the
    corner and shows the spread where the mass actually is.

    Identity line = perfect agreement. The legend names the prompt token, since
    'correct' and 'wrong' only mean which species token was used.
    """
    nat = D[f'A_{sp}_nat']
    hi = max(nat.max(), D[f'A_{sp}_correct'].max(), D[f'A_{sp}_wrong'].max()) * 1.06
    ax.plot([0, hi], [0, hi], color='k', lw=1.1, ls='--', zorder=1)
    for kind, col in (('wrong', C_BAD), ('correct', C_OK)):
        gen = D[f'A_{sp}_{kind}']
        r = float(D[f'A_{sp}_{kind}_r'])
        tok = TOKEN[sp] if kind == 'correct' else TOKEN[OTHER_SP[sp]]
        lab = f"{'Correct' if kind == 'correct' else 'Wrong'} token ({tok})"
        ax.scatter(nat, gen, s=5.5, alpha=0.45, linewidths=0, color=col,
                   label=f'{lab}  r = {r:.3f}', rasterized=True,
                   zorder=3 if kind == 'correct' else 2)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax.set_xlabel('Natural 6-mer frequency')
    ax.set_title(ital(sp), pad=4)
    leg = ax.legend(loc='upper left', handletextpad=0.3, borderpad=0.2, markerscale=2.6)
    for h in leg.legend_handles:
        h.set_alpha(1)

fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, sp in zip(axs, ['ecoli', 'bsubtilis']):
    draw_A(ax, sp)
axs[0].set_ylabel('Generated 6-mer frequency')
plt.tight_layout(); 


def draw_B(ax, sp, show_ylabel=False):
    x = np.arange(-20, 0)
    nat, mu, sd = D[f'{sp}_natural'][180:], D[f'{sp}_C0_mean'][180:], D[f'{sp}_C0_sd'][180:]
    for i, b in enumerate('ACGT'):
        ax.plot(x, nat[:, i], color=BASE_C[b], lw=2.4, ls=(0, (1.6, 1.4)), zorder=3)
        ax.plot(x, mu[:, i], color=BASE_C[b], lw=2.0, zorder=4)
        ax.fill_between(x, mu[:, i] - sd[:, i], mu[:, i] + sd[:, i],
                        color=BASE_C[b], alpha=0.13, lw=0, zorder=1)
    ax.set_xlim(-20, -1); ax.set_ylim(0, 0.85)
    ax.set_xticks([-20, -15, -10, -5, -1])
    ax.set_xlabel('Position relative to start codon')
    ax.set_title(ital(sp), pad=4)
    if show_ylabel:
        ax.set_ylabel('Base probability')

fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
for i, sp in enumerate(['ecoli', 'bsubtilis']):
    draw_B(axs[i], sp, show_ylabel=(i == 0))
h = [Line2D([], [], color=BASE_C[b], lw=2.2, label=b) for b in 'ACGT']
h += [Line2D([], [], color='k', lw=2.2, label='Generated'),
      Line2D([], [], color='k', lw=2.2, ls=(0, (1.6, 1.4)), label='Natural')]
axs[1].legend(handles=h, ncol=3, loc='upper left', columnspacing=0.9)
plt.tight_layout(); 

COND_LAB = {'C0': 'Real CDS', 'C2': 'Shuffled CDS'}

def draw_C(axes, sp):
    """Two profiles and their difference.

    The unrelated-CDS condition is not shown: it substitutes a partner drawn from
    the same gene set, so its population-averaged profile is a relabelling of the
    native one and cannot move whatever the model does. Only the shuffled CDS is
    free to move here, and it does not.
    """
    x = np.arange(-200, 0)
    for ax, c in zip(axes[:2], ['C0', 'C2']):
        mu, sd = D[f'{sp}_{c}_mean'], D[f'{sp}_{c}_sd']
        for i, b in enumerate('ACGT'):
            ax.plot(x, mu[:, i], color=BASE_C[b], lw=1.5)
            ax.fill_between(x, mu[:, i] - sd[:, i], mu[:, i] + sd[:, i],
                            color=BASE_C[b], alpha=0.12, lw=0)
        ax.set_ylim(0, 0.68); ax.set_xlim(-200, -1)
        ax.set_yticks([0, 0.3, 0.6])
        ax.text(0.985, 0.9, COND_LAB[c], transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, fontweight='bold')
        ax.set_xticklabels([])

    # the signed per-base difference, real CDS minus shuffled CDS. If the CDS
    # mattered anywhere, a base would depart from zero there.
    axr = axes[2]
    dif = D[f'{sp}_C2_mean'] - D[f'{sp}_C0_mean']
    axr.axhline(0, color='k', lw=1.0, zorder=2)
    for i, b in enumerate('ACGT'):
        axr.plot(x, dif[:, i], color=BASE_C[b], lw=1.1, zorder=3)
    m = np.abs(dif).mean()
    axr.set_xlim(-200, -1); axr.set_ylim(-0.17, 0.17)
    axr.set_yticks([-0.1, 0, 0.1])
    axr.set_xlabel('Position relative to start codon')
    axr.set_ylabel('$\\Delta p$  real\u2009\u2212\u2009shuffled', fontsize=7.4)
    axr.text(0.985, 0.90, f'mean $|\\Delta p|$ = {m:.4f}', transform=axr.transAxes,
             ha='right', va='top', fontsize=7.4, style='italic')

fig, axs = plt.subplots(3, 1, figsize=(6.6, 4.6), sharex=True,
                        gridspec_kw={'height_ratios': [1, 1, 0.72],
                                     'hspace': 0.20})
draw_C(list(axs), 'ecoli')
axs[0].set_ylabel('Base probability')
plt.tight_layout(); 

def boot_ci(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def draw_D(ax):
    """Per-gene GC of the whole 200 bp upstream. Dashed line = that species'
    natural mean. Numbers and interpretation belong in the caption, not the axes."""
    pos, data, cols = [], [], []
    for gi, sp in enumerate(['ecoli', 'bsubtilis']):
        for bi, (key, col) in enumerate((('nat', '#B2182B'), ('gen', '#2166AC'))):
            pos.append(gi * 2.5 + bi); data.append(100 * D[f'GC_{sp}_{key}']); cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.66, patch_artist=True, showfliers=False,
                    medianprops=dict(color='k', lw=2.2), whiskerprops=dict(lw=1.4),
                    capprops=dict(lw=1.4), boxprops=dict(lw=1.4))
    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor(col); patch.set_alpha(0.72)
    # species mean of the natural upstreams
    for gi, sp in enumerate(['ecoli', 'bsubtilis']):
        m = 100 * D[f'GC_{sp}_nat'].mean()
        ax.plot([gi * 2.5 - 0.55, gi * 2.5 + 1.55], [m, m], color='k', lw=1.8,
                ls=(0, (4.5, 2.6)), zorder=5)
    ax.set_xticks(pos)
    ax.set_xticklabels(['Natural', 'Generated'] * 2, fontsize=8.2)
    for gi, sp in enumerate(['ecoli', 'bsubtilis']):
        ax.text(gi * 2.5 + 0.5, -0.115, ital(sp), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9.5)
    ax.set_ylabel('Upstream GC content (%)\none value per gene')
    ax.set_ylim(20, 62); ax.set_xlim(-0.72, 4.22)
    h = [Line2D([], [], color='k', lw=1.8, ls=(0, (4.5, 2.6)), label='species mean')]
    ax.legend(handles=h, loc='upper right', fontsize=8)

fig, ax = plt.subplots(figsize=(5.4, 3.9))
draw_D(ax)
plt.tight_layout(); 

fig = plt.figure(figsize=(13.8, 10.8))
outer = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.24,
                 left=0.065, right=0.985, top=0.955, bottom=0.06)

# --- A ---
gA = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.28)
aA = [fig.add_subplot(gA[0, i]) for i in range(2)]
for ax, sp in zip(aA, ['ecoli', 'bsubtilis']):
    draw_A(ax, sp)
aA[0].set_ylabel('Generated 6-mer frequency')
panel_tag(aA[0], 'A', dx=-0.30)

# --- B ---
gB = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.20)
aB = [fig.add_subplot(gB[0, i]) for i in range(2)]
for i, sp in enumerate(['ecoli', 'bsubtilis']):
    draw_B(aB[i], sp, show_ylabel=(i == 0))
hB = [Line2D([], [], color=BASE_C[b], lw=2.2, label=b) for b in 'ACGT']
hB += [Line2D([], [], color='k', lw=2.2, label='Generated'),
       Line2D([], [], color='k', lw=2.2, ls=(0, (1.6, 1.4)), label='Natural')]
aB[1].legend(handles=hB, ncol=3, loc='upper left', columnspacing=0.8, handlelength=1.5)
panel_tag(aB[0], 'B', dx=-0.26)

# --- C ---
gC = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1, 0],
                             height_ratios=[1, 1, 0.72], hspace=0.20)
aC = [fig.add_subplot(gC[i, 0]) for i in range(3)]
draw_C(aC, 'ecoli')
aC[0].set_ylabel('Base probability')
panel_tag(aC[0], 'C', dx=-0.11, dy=1.16)

# --- D ---
aD = fig.add_subplot(outer[1, 1])
draw_D(aD)
panel_tag(aD, 'D', dx=-0.11)

for ext in ('pdf', 'png'):
    for out in (RESULTS, FIGURES):
        fig.savefig(os.path.join(out, f'cds_prompted_main.{ext}'),
                    dpi=400, bbox_inches='tight')
print('wrote cds_prompted_main.{pdf,png} to results/ and figures/')


fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, stat, nice in zip(axs, ['min', 'mean'],
                          ['Minimum Shannon entropy (bits)', 'Mean Shannon entropy (bits)']):
    data, labels, cols = [], [], []
    for sp in ['ecoli', 'bsubtilis']:
        for grp, col in (('high', C_OK), ('low', '#E08214')):
            data.append(D[f'{sp}_D_{grp}_{stat}'])
            labels.append(f"{SPN[sp].split()[0][0]}.{SPN[sp].split()[1][:3]}\n{grp} CAI")
            cols.append(col)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color='k', lw=2))
    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_linewidth(1.3)
    for i, v in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.07, len(v)), v,
                   s=4, color=cols[i], alpha=0.45, linewidths=0, zorder=3)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(nice)
    for gi, sp in enumerate(['ecoli', 'bsubtilis']):
        p = float(D[f'{sp}_D_{stat}_p'])
        y = max(np.percentile(data[2 * gi], 97), np.percentile(data[2 * gi + 1], 97))
        ax.plot([2 * gi + 1, 2 * gi + 2], [y, y], color='k', lw=1.2)
        ax.text(2 * gi + 1.5, y, f'\np = {p:.2g}', ha='center', va='top', fontsize=7.5)
axs[0].set_title('Statistic favours E. coli', fontsize=9)
axs[1].set_title('Statistic favours B. subtilis', fontsize=9)
plt.tight_layout()
for ext in ('pdf', 'png'):
    for out in (RESULTS, FIGURES):
        fig.savefig(os.path.join(out, f'cds_prompted_supp_cai.{ext}'), dpi=400, bbox_inches='tight')
print('wrote cds_prompted_supp_cai.{pdf,png}')

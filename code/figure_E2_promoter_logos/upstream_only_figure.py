# %% [markdown]
# # Upstream-only benchmark — dataset characterisation figure
#
# ICBINB-BIO / NeurIPS 2026. Builds a two-page publication figure describing the
# `upstream_only` track (681 E. coli genes, 200 bp upstream, RegulonDB-annotated).
#
# **Page 1** — composition, element geometry, and the leaked-vs-clean control.
# **Page 2** — sequence logos of the distal (−35) and proximal (−10) elements,
# split by sigma factor.
#
# Inputs: `prompts_681_{leaked,clean}.csv` -- in the release they are at
# `data/`; on the cluster at `data/benchmark/upstream_only/`.
# Outputs: `results/upstream_only_dataset.{pdf,png}`

# %%
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import csv, os, itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.backends.backend_pdf import PdfPages
import logomaker

BASE = f"{ICBINB_ROOT}"
# On the cluster the prompt manifests sit under the full benchmark tree; the
# released repository ships them at the top of data/ (the rest of that tree is
# tens of gigabytes of generations and stays on the cluster). Use whichever is
# present so this script runs in both places.
UPO  = f"{BASE}/data/benchmark/upstream_only"
if not _os.path.exists(f"{UPO}/prompts_681_leaked.csv"):
    UPO = f"{BASE}/data"
OUT  = f"{BASE}/results"
os.makedirs(OUT, exist_ok=True)

L_ROWS = list(csv.DictReader(open(f"{UPO}/prompts_681_leaked.csv")))
C_ROWS = list(csv.DictReader(open(f"{UPO}/prompts_681_clean.csv")))
assert [r["locus_tag"] for r in L_ROWS] == [r["locus_tag"] for r in C_ROWS]
N = len(L_ROWS)
print(f"{N} genes")

# full 200 bp upstream = prompt + target of task 1
seq_leaked = {r["locus_tag"]: r["prompt_promoter"] + r["target_promoter"] for r in L_ROWS}
seq_clean  = {r["locus_tag"]: r["prompt_promoter"] + r["target_promoter"] for r in C_ROWS}

# %% [markdown]
# ## Publication style
#
# Thick spines, outward ticks, no top/right frame, generous panel spacing.

# %%
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10.5,
    "axes.titleweight":  "bold",
    "axes.linewidth":     1.6,      # thick axes
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  1.6,
    "ytick.major.width":  1.6,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "legend.frameon":     False,
    "legend.fontsize":    8,
    "pdf.fonttype":       42,       # editable text in the PDF
    "ps.fonttype":        42,
    "savefig.bbox":      "tight",
})

# colour-blind-safe palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "grey": "#999999"}
PIE_COLS = [CB["blue"], CB["orange"], CB["green"], CB["red"], CB["purple"],
            CB["sky"], CB["yellow"], CB["grey"], "#7F7F7F", "#BBBBBB"]

def panel_label(ax, letter, dx=-0.16, dy=1.10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")

# %% [markdown]
# ## Derived quantities

# %%
sigma   = [r["sigma"] or "unassigned" for r in L_ROWS]
gap_35_10 = np.array([int(r["m10_start"]) - int(r["m35_end"]) for r in L_ROWS])
gap_10_tss = np.array([int(r["tss_idx"]) - int(r["m10_end"]) for r in L_ROWS])
sd_spacing = np.array([int(r["sd_spacing"]) for r in L_ROWS])
sd_score   = np.array([int(r["sd_score"]) for r in L_ROWS])

# element positions expressed as distance from the ATG (negative = upstream)
def dist_from_atg(r, key):
    return -(int(r["L"]) - int(r[key]))

pos = {
    "−35":  np.array([dist_from_atg(r, "m35_start") for r in L_ROWS]),
    "−10":  np.array([dist_from_atg(r, "m10_start") for r in L_ROWS]),
    "TSS":  np.array([dist_from_atg(r, "tss_idx")   for r in L_ROWS]),
    "SD":   np.array([dist_from_atg(r, "sd_start")  for r in L_ROWS]),
}

# TFBS: distinct regulators per gene, and footprint coverage of the 200 bp
regs_per_gene, cov_frac, n_tfbs = [], [], []
for r in L_ROWS:
    regs = [x for x in r["tfbs_regulators"].split(";") if x]
    regs_per_gene.append(set(regs))
    n_tfbs.append(int(r["n_tfbs"]))
    Lg = int(r["L"])
    covered = set()
    for sp in r["tfbs_spans"].split(";"):
        if not sp:
            continue
        a, b = sp.split("-")
        covered.update(range(int(a), int(b)))
    cov_frac.append(100.0 * len(covered) / Lg)
cov_frac = np.array(cov_frac); n_tfbs = np.array(n_tfbs)

gc = lambda s: 100.0 * (s.count("G") + s.count("C")) / len(s)
gc_leaked = np.array([gc(seq_leaked[r["locus_tag"]]) for r in L_ROWS])
gc_clean  = np.array([gc(seq_clean[r["locus_tag"]])  for r in L_ROWS])
print(f"max |GC_leaked - GC_clean| = {np.abs(gc_leaked - gc_clean).max():.3g}")

# %% [markdown]
# ## 4-mer structure (leaked vs clean)
#
# Composition is preserved *exactly* by the permutation, so any difference here is
# k-mer structure, not base content.

# %%
KM = ["".join(p) for p in itertools.product("ACGT", repeat=4)]
KIDX = {k: i for i, k in enumerate(KM)}

def kmer_matrix(seqs, k=4):
    M = np.zeros((len(seqs), len(KM)))
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            idx = KIDX.get(s[j:j + k])
            if idx is not None:
                M[i, idx] += 1
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M

def pairwise_cos_dist(seqs):
    M = kmer_matrix(seqs)
    D = 1.0 - M @ M.T
    iu = np.triu_indices(len(seqs), k=1)
    return D[iu]

d_leaked = pairwise_cos_dist([seq_leaked[r["locus_tag"]] for r in L_ROWS])
d_clean  = pairwise_cos_dist([seq_clean[r["locus_tag"]]  for r in L_ROWS])
print(f"4-mer pairwise distance: leaked {d_leaked.mean():.4f} | clean {d_clean.mean():.4f}")

# %% [markdown]
# ## Page 1 — dataset characterisation

# %%
fig1 = plt.figure(figsize=(13.5, 12.6))
gs = GridSpec(3, 3, figure=fig1, hspace=0.62, wspace=0.34,
              left=0.07, right=0.97, top=0.90, bottom=0.06,
              height_ratios=[1.05, 1.0, 1.0])

def clean_pie(ax, items, title, ncol=2, min_pct=4.0):
    """Pie with the labels in a legend underneath — inline labels collide."""
    vals = [v for _, v in items]
    labs = [f"{k} ({v})" for k, v in items]
    wedges, _, at = ax.pie(
        vals, labels=None, colors=PIE_COLS[:len(vals)], startangle=90,
        autopct=lambda p: f"{p:.0f}%" if p >= min_pct else "",
        pctdistance=0.70,
        wedgeprops=dict(edgecolor="white", linewidth=1.4))
    for t in at:
        t.set_fontsize(7.5); t.set_color("white"); t.set_fontweight("bold")
    ax.legend(wedges, labs, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=ncol, fontsize=7.5, handlelength=1.1, handletextpad=0.5,
              columnspacing=1.1, borderpad=0.2)
    ax.set_title(title, pad=12)

# ---- A: sigma factor pie -----------------------------------------------------
axA = fig1.add_subplot(gs[0, 0])
sc = Counter(sigma).most_common()
big = [(k, v) for k, v in sc if v >= 25]
tail = sum(v for k, v in sc if v < 25)
if tail:
    big.append(("other", tail))
clean_pie(axA, big, "Sigma factor")
panel_label(axA, "A", dx=-0.06, dy=1.08)

# ---- B: TFBS regulator pie ---------------------------------------------------
axB = fig1.add_subplot(gs[0, 1])
rc = Counter(x for s in regs_per_gene for x in s).most_common()
TOP = 8
top = rc[:TOP]
other = sum(v for _, v in rc[TOP:])
n_with = int((n_tfbs > 0).sum())
clean_pie(axB, top + [(f"other ({len(rc)-TOP} TFs)", other)],
          f"TF regulators — genes bound\n({n_with}/{N} genes have ≥1 site)",
          ncol=3, min_pct=3.0)
panel_label(axB, "B", dx=-0.06, dy=1.08)

# ---- C: positional density ---------------------------------------------------
axC = fig1.add_subplot(gs[0, 2])
cols = {"−35": CB["blue"], "−10": CB["orange"], "TSS": CB["green"], "SD": CB["red"]}
bins = np.arange(-200, 2, 4)
for k in ["−35", "−10", "TSS", "SD"]:
    axC.hist(pos[k], bins=bins, histtype="step", linewidth=2.0,
             color=cols[k], label=k, density=True)
axC.axvline(0, color="black", lw=1.4, ls="--")
axC.text(0, axC.get_ylim()[1] * 0.97, " ATG", ha="left", va="top", fontsize=8.5,
         fontweight="bold")
axC.set_xlabel("position relative to ATG (bp)")
axC.set_ylabel("density")
axC.set_title("Where the elements sit")
axC.legend(loc="upper left")
panel_label(axC, "C")

# ---- D: -35 -> -10 spacer ----------------------------------------------------
axD = fig1.add_subplot(gs[1, 0])
b = np.arange(gap_35_10.min() - 0.5, gap_35_10.max() + 1.5, 1)
axD.hist(gap_35_10, bins=b, color=CB["blue"], edgecolor="white", linewidth=0.8)
med = np.median(gap_35_10)
axD.axvline(med, color=CB["red"], lw=2.0, ls="--")
axD.text(0.97, 0.95, f"median {med:.0f} bp", transform=axD.transAxes,
         color=CB["red"], fontsize=9, fontweight="bold", va="top", ha="right")
axD.set_xlabel("gap: end of −35 → start of −10 (bp)")
axD.set_ylabel("genes")
axD.set_title("−35 / −10 spacer")
axD.set_xlim(5, 30)
panel_label(axD, "D")

# ---- E: -10 -> TSS discriminator --------------------------------------------
axE = fig1.add_subplot(gs[1, 1])
b = np.arange(gap_10_tss.min() - 0.5, gap_10_tss.max() + 1.5, 1)
axE.hist(gap_10_tss, bins=b, color=CB["orange"], edgecolor="white", linewidth=0.8)
med = np.median(gap_10_tss)
axE.axvline(med, color=CB["red"], lw=2.0, ls="--")
axE.text(0.97, 0.95, f"median {med:.0f} bp", transform=axE.transAxes,
         color=CB["red"], fontsize=9, fontweight="bold", va="top", ha="right")
axE.set_xlabel("gap: end of −10 → TSS (bp)")
axE.set_ylabel("genes")
axE.set_title("Discriminator")
axE.set_xlim(-1, 20)
panel_label(axE, "E")

# ---- F: SD spacing + score ---------------------------------------------------
axF = fig1.add_subplot(gs[1, 2])
b = np.arange(sd_spacing.min() - 0.5, sd_spacing.max() + 1.5, 1)
axF.hist(sd_spacing, bins=b, color=CB["green"], edgecolor="white", linewidth=0.8)
axF.set_xlabel("gap: SD 3′ end → ATG (bp)")
axF.set_ylabel("genes")
axF.set_title("Shine–Dalgarno")
ins = axF.inset_axes([0.60, 0.55, 0.38, 0.42])
sccnt = Counter(sd_score)
ks = sorted(sccnt)
ins.bar(ks, [sccnt[k] for k in ks],
        color=[CB["grey"] if k < 4 else CB["green"] for k in ks],
        edgecolor="white", linewidth=0.6)
ins.set_xlabel("SD score /6", fontsize=7, labelpad=1)
ins.set_ylabel("genes", fontsize=7, labelpad=1)
ins.tick_params(labelsize=6.5, width=1.0, length=3)
for s in ins.spines.values():
    s.set_linewidth(1.0)
n_weak = int((sd_score < 4).sum())
ins.set_title(f"grey = weak (n={n_weak})", fontsize=6.5, fontweight="normal", pad=3)
panel_label(axF, "F")

# ---- G: TFBS footprint coverage ---------------------------------------------
axG = fig1.add_subplot(gs[2, 0])
axG.hist(cov_frac[cov_frac > 0], bins=np.arange(0, 101, 5),
         color=CB["purple"], edgecolor="white", linewidth=0.8)
axG.set_xlabel("of the 200 bp covered by TF footprints (%)")
axG.set_ylabel("genes")
axG.set_title("TF footprint coverage")
ins = axG.inset_axes([0.58, 0.55, 0.40, 0.42])
nc = Counter(np.clip(n_tfbs, 0, 5))
ks = sorted(nc)
ins.bar(ks, [nc[k] for k in ks], color=CB["purple"], edgecolor="white", linewidth=0.6)
ins.set_xticks(ks)
ins.set_xticklabels([str(k) if k < 5 else "5+" for k in ks], fontsize=6.5)
ins.set_xlabel("sites per gene", fontsize=7, labelpad=1)
ins.set_ylabel("genes", fontsize=7, labelpad=1)
ins.tick_params(labelsize=6.5, width=1.0, length=3)
for s in ins.spines.values():
    s.set_linewidth(1.0)
axG.text(0.03, 0.72, f"median {np.median(cov_frac[cov_frac>0]):.0f}%\n(genes with sites)",
         transform=axG.transAxes, fontsize=8.5, va="top", fontweight="bold")
panel_label(axG, "G")

# ---- H: control 1 — GC is preserved exactly ---------------------------------
axH = fig1.add_subplot(gs[2, 1])
b = np.arange(20, 75, 1.5)
axH.hist(gc_leaked, bins=b, color=CB["blue"], alpha=0.55, label="natural", edgecolor="none")
axH.hist(gc_clean,  bins=b, histtype="step", linewidth=2.2, color=CB["red"],
         label="shuffled")
axH.set_xlabel("GC content of the 200 bp (%)")
axH.set_ylabel("genes")
axH.set_title("Control: composition preserved")
axH.legend(loc="upper right")
axH.text(0.03, 0.95,
         f"max |Δ| = {np.abs(gc_leaked-gc_clean).max():.2g}%\n(permutation, not resampling)",
         transform=axH.transAxes, fontsize=8, va="top", fontweight="bold")
panel_label(axH, "H")

# ---- I: control 2 — k-mer structure is destroyed -----------------------------
axI = fig1.add_subplot(gs[2, 2])
b = np.linspace(min(d_leaked.min(), d_clean.min()),
                max(d_leaked.max(), d_clean.max()), 70)
axI.hist(d_leaked, bins=b, color=CB["blue"], alpha=0.55, density=True,
         label="natural", edgecolor="none")
axI.hist(d_clean, bins=b, histtype="step", linewidth=2.2, color=CB["red"],
         density=True, label="shuffled")
axI.set_xlabel("pairwise 4-mer cosine distance")
axI.set_ylabel("density")
axI.set_title("Control: k-mer structure altered")
axI.legend(loc="upper left")
axI.text(0.03, 0.72, f"mean {d_leaked.mean():.3f} → {d_clean.mean():.3f}",
         transform=axI.transAxes, fontsize=8, va="top", fontweight="bold")
panel_label(axI, "I")

fig1.suptitle(f"Upstream-only benchmark: {N} E. coli genes, 200 bp upstream, "
              "RegulonDB-annotated", fontsize=14, fontweight="bold", y=0.965)

# %% [markdown]
# ## Page 2 — sequence logos per sigma factor
#
# Box widths are not uniform in RegulonDB (the −10 is annotated as 11 bp for the
# extended −10 promoters), so windows are **anchored**: the −35 at its 5′ end and
# the −10 at its 3′ end, 6 bp core with 3 bp flanks. Sigma factors with fewer than
# 25 genes are omitted — a logo over n<25 is noise.
#
# Naming: 'distal' = the upstream box (−35 for sigma70/38/32/24, −24 for sigma54),
# 'proximal' = the TSS-proximal box (−10, or −12 for sigma54). This matches the
# ablation figures (pc_ablation_bysigma.py), where the same two boxes are the
# 'distal' / 'proximal' conditions.

# %%
CORE, FLANK = 6, 3
W = CORE + 2 * FLANK
BASES = list("ACGT")

def window_m35(r):
    s = seq_leaked[r["locus_tag"]]; i = int(r["m35_start"])
    a, b = i - FLANK, i - FLANK + W
    return s[a:b] if a >= 0 and b <= len(s) else None

def window_m10(r):
    s = seq_leaked[r["locus_tag"]]; j = int(r["m10_end"])
    a, b = j - CORE - FLANK, j - CORE - FLANK + W
    return s[a:b] if a >= 0 and b <= len(s) else None

def freq_matrix(seqs):
    """Position x base frequency; every row sums to 1."""
    m = np.zeros((W, 4))
    for s in seqs:
        for p, ch in enumerate(s):
            if ch in BASES:
                m[p, BASES.index(ch)] += 1
    return pd.DataFrame(m / m.sum(1, keepdims=True), columns=BASES)

def info_bits(freq):
    """Shannon information per position: 0 bits = uniform, 2 bits = fixed."""
    p = freq.values
    h = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0).sum(axis=1)
    return pd.Series(2.0 - h)

MIN_N = 25
groups = [(s, n) for s, n in Counter(sigma).most_common() if n >= MIN_N]
print("logo groups:", groups)

SIG_LABEL = {"sigma54": ("distal box (\u221224)", "proximal box (\u221212)")}
DEFAULT_LABEL = ("distal box (\u221235)", "proximal box (\u221210)")

FLANK_KEEP = 0.5          # a flank carrying this many bits counts as signal
XLIM = (-0.6, W - 0.4)

fig2 = plt.figure(figsize=(11.6, 3.20 * len(groups) + 1.30))
gs2 = GridSpec(len(groups), 2, figure=fig2, hspace=0.52, wspace=0.17,
               left=0.070, right=0.985, top=0.908, bottom=0.048)

for gi, (sig, n) in enumerate(groups):
    rows = [r for r, s in zip(L_ROWS, sigma) if s == sig]
    for ci, wf in enumerate([window_m35, window_m10]):
        ax = fig2.add_subplot(gs2[gi, ci])

        seqs = [w for w in (wf(r) for r in rows) if w and len(w) == W]
        freq = freq_matrix(seqs)
        bits = info_bits(freq)
        ic = freq.mul(bits, axis=0)        # classic information logo heights

        lg = logomaker.Logo(ic, ax=ax, color_scheme="classic",
                            show_spines=False)
        # Fade the flanks so the eye lands on the annotated core -- except
        # where a flank carries real signal (>= FLANK_KEEP bits), so context
        # outside the annotated box is not suppressed (e.g. the sigma38 -13C,
        # a known sigmaS selectivity determinant). Heights are untouched.
        for _p in list(range(FLANK)) + list(range(FLANK + CORE, W)):
            if bits[_p] >= FLANK_KEEP:
                continue
            for _b in BASES:
                if ic.loc[_p, _b] > 0:
                    lg.style_single_glyph(_p, _b, alpha=0.34)

        ax.axvspan(FLANK - 0.5, FLANK + CORE - 0.5,
                   color=CB["yellow"], alpha=0.16, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, ls=(0, (2, 4)), lw=0.8, color="0.78", zorder=1)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, 2)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(["0", "", "1", "", "2"], fontsize=8.5)
        ax.set_ylabel("bits", fontsize=9.5)

        # A 0.2 bit letter is 1/10th of the axis -- that is what 0.2 bits
        # means, and inflating it would misreport the data. So the identity
        # of every core position is printed under the axis instead: position,
        # most frequent base, and how often it occurs.
        labs = []
        for p in range(W):
            top = freq.iloc[p].idxmax(); pct = freq.iloc[p].max() * 100
            if FLANK <= p < FLANK + CORE:
                labs.append("%d\n%s\n%.0f%%" % (p - FLANK + 1, top, pct))
            elif bits[p] >= FLANK_KEEP:
                labs.append("\u00b7\n%s\n%.0f%%" % (top, pct))
            else:
                labs.append("\u00b7")
        ax.set_xticks(range(W))
        ax.set_xticklabels(labs, fontsize=8.5, linespacing=1.55)
        for p, t in enumerate(ax.get_xticklabels()):
            if FLANK <= p < FLANK + CORE:
                t.set_fontweight("bold"); t.set_color("0.10")
            elif bits[p] >= FLANK_KEEP:
                t.set_color("0.25")
            else:
                t.set_color("0.55")

        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.5)
        ax.tick_params(width=1.5, length=4)

        lab = SIG_LABEL.get(sig, DEFAULT_LABEL)[ci]
        ax.set_title("%s  \u00b7  %s  \u00b7  n=%d  \u00b7  core %.1f bits"
                     % (sig, lab, len(seqs), bits[FLANK:FLANK + CORE].sum()),
                     fontsize=10.5, fontweight="bold", pad=7)
        if ci == 0:
            panel_label(ax, chr(ord("A") + gi), dx=-0.095, dy=1.13)

fig2.suptitle("Promoter element logos by sigma factor  \u2014  "
              "letter height = information content (bits)",
              fontsize=13, fontweight="bold", y=0.968)

# %% [markdown]
# ## Export

# %%
pdf_path = f"{OUT}/upstream_only_dataset.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
fig1.savefig(f"{OUT}/upstream_only_dataset_p1.png", dpi=300)

# Page 2 IS the motif figure -- written once, under one name, in both formats.
fig2.savefig(f"{OUT}/upstream_only_logos.pdf")
fig2.savefig(f"{OUT}/upstream_only_logos.png", dpi=300)

print("wrote:")
for f in ["upstream_only_dataset.pdf", "upstream_only_dataset_p1.png",
          "upstream_only_logos.pdf", "upstream_only_logos.png"]:
    print(f"  {OUT}/{f}  ({os.path.getsize(f'{OUT}/{f}')/1e3:.0f} kB)")

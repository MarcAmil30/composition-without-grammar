#!/usr/bin/env python3
"""
Supplementary figure for the CDS-prompted benchmark (exploratory description of
the dataset, not of any model result).

Four panels, one per axis of the dataset:

  A  the genes and their genomic context   (derivation, window contamination)
  B  the prompt   -- the CDS              (diversity, CAI classes)
  C  the target   -- the 200 bp upstream  (between-gene distinctness)
  D  the conditions C0-C6                 (how far each operator moves the CDS)

Panels B, C and D each carry the same message with its own null: there is real
between-item variance on both sides of the pairing, so a single species-generic
upstream cannot be the correct answer for every CDS.

Two-stage by design.  The raw prompt files live on Isambard and are large, so
this script computes every statistic there and caches it to a small .npz; run
again anywhere with only the .npz present and it re-renders from the cache.
That keeps the data on Isambard (per the project convention) while the figure
is typeset with the same fonts as the companion upstream-only figure.

    on Isambard :  python make_cds_prompted_dataset_figure.py --stats-only
    locally     :  python make_cds_prompted_dataset_figure.py

Derivation counts upstream of "singletons with a 200 bp upstream" are build
outputs quoted from data/benchmark/cds_prompted/README.md; everything from that
step onward is recomputed here from the prompt files.

Outputs: results/ and figures/ , as .pdf and .png
"""

from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ISAMBARD_BENCH = Path(
    f"{ICBINB_ROOT}/data/benchmark/cds_prompted"
)
STATS_NAME = "cds_prompted_dataset_supp_stats.npz"

SEED = 20260801
K = 4                       # k-mer size for alignment-free CDS dissimilarity
SPECIES = ("ecoli", "bsubtilis")
SP_LABEL = {"ecoli": "E. coli", "bsubtilis": "B. subtilis"}
CONDS = ("C5", "C1", "C6", "C2", "C3")          # ordered by severity
# C3 swaps in a length-matched partner gene, so a positionwise "% nucleotides
# changed" is defined only where the two lengths happen to agree (67% / 60% of
# genes) and its GC is matched in the mean rather than per gene. It is therefore
# shown only on the alignment-free axis, where it is defined for every gene.
CONDS_PCT = ("C5", "C1", "C6", "C2")
N_PAIR_KEEP = 100_000       # pairwise arrays are subsampled before caching
COND_NOTE = {
    "C5": "codons 1$-$30\nshuffled",
    "C1": "synonymous\npermuted",
    "C6": "codons 30$-$end\nshuffled",
    "C2": "codon order\nshuffled",
    "C3": "partner\nCDS",
}
N_TOP = 200                 # size of the high- / low-CAI lists
MINUS = "−"
ARROW = r"$\rightarrow$"     # mathtext: always available, unlike Helvetica U+2192

# --------------------------------------------------------------------------- #
# style  (shared with make_upstream_only_dataset_figure.py)
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
        "axes.linewidth": 2.0,
        "axes.edgecolor": "#1a1a1a",
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.minor.size": 3.2,
        "ytick.minor.size": 3.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a",
        "text.color": "#1a1a1a",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.solid_capstyle": "round",
    })


C = {
    "blue":   "#2166AC",
    "blue_l": "#92C5DE",
    "orange": "#D95F02",
    "orange_l": "#FDBE85",
    "amber":  "#E6A700",
    "green":  "#1B7837",
    "pink":   "#C51B7D",
    "teal":   "#01807F",
    "purple": "#762A83",
    "red":    "#B2182B",
    "grey":   "#8C8C8C",
    "grey_l": "#D9D9D9",
    "ink":    "#1a1a1a",
}
SP_COLOR = {"ecoli": C["blue"], "bsubtilis": C["orange"]}
CAI_COLOR = {"high": C["red"], "low": C["teal"]}


def panel_label(ax, letter: str, dx: float = -0.13, dy: float = 1.16) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top", ha="left", color=C["ink"])


def thicken(ax, outward: float = 6) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.0)
        ax.spines[s].set_position(("outward", outward))


def title(ax, text: str) -> None:
    ax.set_title(text, fontweight="bold", pad=11)


# --------------------------------------------------------------------------- #
# sequence helpers
# --------------------------------------------------------------------------- #
BASES = "ACGT"
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i

GENETIC_CODE = {}
for _i, _a in enumerate(
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
):
    _b1, _b2, _b3 = "TCAG"[_i // 16], "TCAG"[(_i // 4) % 4], "TCAG"[_i % 4]
    GENETIC_CODE[_b1 + _b2 + _b3] = _a

# RSCU is defined over synonymous families of size > 1, excluding stops
_FAMILIES: dict[str, list[str]] = {}
for _cod, _aa in GENETIC_CODE.items():
    if _aa != "*":
        _FAMILIES.setdefault(_aa, []).append(_cod)
RSCU_CODONS = [c for aa, fam in sorted(_FAMILIES.items()) if len(fam) > 1
               for c in sorted(fam)]
RSCU_FAMILY = np.array(
    [sorted(_FAMILIES).index(GENETIC_CODE[c]) for c in RSCU_CODONS]
)
CODON_INDEX = {c: i for i, c in enumerate(RSCU_CODONS)}


def encode(seq: str) -> np.ndarray:
    """ACGT -> 0123, anything else -> -1."""
    return _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def kmer_freq(seqs: list[str], k: int = K) -> np.ndarray:
    """Per-sequence overlapping k-mer frequency vectors, L2-normalised."""
    X = np.zeros((len(seqs), 4 ** k), dtype=np.float64)
    powers = 4 ** np.arange(k - 1, -1, -1)
    for i, s in enumerate(seqs):
        c = encode(s)
        if len(c) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(c, k)
        w = w[(w >= 0).all(axis=1)]
        X[i] = np.bincount((w * powers).sum(axis=1), minlength=4 ** k)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def rscu_matrix(seqs: list[str]) -> np.ndarray:
    """Per-gene RSCU over multi-codon synonymous families."""
    out = np.ones((len(seqs), len(RSCU_CODONS)), dtype=np.float64)
    n_fam = RSCU_FAMILY.max() + 1
    for i, s in enumerate(seqs):
        counts = np.zeros(len(RSCU_CODONS))
        for j in range(0, len(s) - 2, 3):
            idx = CODON_INDEX.get(s[j:j + 3])
            if idx is not None:
                counts[idx] += 1
        tot = np.bincount(RSCU_FAMILY, weights=counts, minlength=n_fam)
        size = np.bincount(RSCU_FAMILY, minlength=n_fam)
        expected = (tot / np.maximum(size, 1))[RSCU_FAMILY]
        ok = expected > 0
        out[i, ok] = counts[ok] / expected[ok]
    return out


def pairwise_cosine_dist(X: np.ndarray, max_n: int = 2500,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Off-diagonal cosine distances of L2-normalised rows."""
    if len(X) > max_n:
        rng = rng or np.random.default_rng(SEED)
        X = X[rng.choice(len(X), max_n, replace=False)]
    rng = rng or np.random.default_rng(SEED)
    S = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    d = 1.0 - S[iu]
    if len(d) > N_PAIR_KEEP:
        d = d[rng.choice(len(d), N_PAIR_KEEP, replace=False)]
    return d


def onehot(seqs: list[str], L: int = 200) -> np.ndarray:
    """(n, L*4) float32 one-hot; unknown bases contribute nothing."""
    n = len(seqs)
    M = np.zeros((n, L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        c = encode(s)[:L]
        ok = c >= 0
        M[i, np.arange(len(c))[ok], c[ok]] = 1.0
    return M.reshape(n, L * 4)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def compute_stats(bench: Path) -> dict:
    import pandas as pd

    rng = np.random.default_rng(SEED)
    out: dict[str, np.ndarray] = {}

    for sp in SPECIES:
        d0 = pd.read_csv(bench / f"prompts_C0_{sp}.csv")
        cds = [p[2:-2] for p in d0.prompt.astype(str)]
        assert all(len(c) == n for c, n in zip(cds, d0.cds_len)), "CDS parse"
        ups = d0.target_upstream_200bp.astype(str).tolist()
        assert {len(u) for u in ups} == {200}, "upstream length"

        # ---- A: the gene set ---------------------------------------------- #
        out[f"{sp}_cds_len"] = d0.cds_len.to_numpy(float)
        out[f"{sp}_n"] = np.array([len(d0)])
        # kept for the caption, not plotted
        out[f"{sp}_clean"] = d0.clean.to_numpy(int)
        out[f"{sp}_dist"] = d0.dist_any_feature.to_numpy(float)
        out[f"{sp}_tandem"] = (d0.neighbour_strand == "tandem").to_numpy()

        # ---- B: CDS diversity and the CAI classes -------------------------- #
        cai = d0.CAI.to_numpy(float)
        order = np.argsort(cai)
        cls = np.zeros(len(cai), dtype=np.int8)          # 0 mid, 1 high, -1 low
        cls[order[-N_TOP:]] = 1
        cls[order[:N_TOP]] = -1
        out[f"{sp}_cai"] = cai
        out[f"{sp}_cai_class"] = cls

        R = rscu_matrix(cds)
        Rc = R - R.mean(axis=0)
        U, Sv, _ = np.linalg.svd(Rc, full_matrices=False)
        pcs = U[:, :2] * Sv[:2]
        # sign of a principal component is arbitrary; orient both so that the
        # high-CAI genes sit up and to the right. Display only, no rescaling.
        for j in range(2):
            if pcs[cls == 1, j].mean() < pcs[cls == -1, j].mean():
                pcs[:, j] *= -1
        out[f"{sp}_pcs"] = pcs
        out[f"{sp}_pcvar"] = (Sv ** 2 / (Sv ** 2).sum())[:2]
        out[f"{sp}_pc_cai_r"] = np.array(
            [np.corrcoef(pcs[:, j], cai)[0, 1] for j in range(2)]
        )

        Xk0 = kmer_freq(cds)
        out[f"{sp}_cds_between"] = pairwise_cosine_dist(Xk0, rng=rng)

        # ---- C: how distinct the real upstreams are ------------------------ #
        E = np.stack([encode(u) for u in ups])                    # (n, 200)
        n = len(E)
        counts = np.stack([(E == b).sum(axis=0) for b in range(4)])   # (4, 200)
        pair_pos = (counts * (counts - 1)).sum(axis=0) / (n * (n - 1))
        p_glob = counts.sum(axis=1) / counts.sum()
        out[f"{sp}_pos_identity"] = pair_pos
        out[f"{sp}_pos_floor"] = np.array([(p_glob ** 2).sum()])

        iu = np.triu_indices(n, k=1)
        keep = rng.choice(len(iu[0]), min(N_PAIR_KEEP, len(iu[0])),
                          replace=False)
        shuf = ["".join(rng.permutation(list(u))) for u in ups]
        for tag, seqs in (("real", ups), ("shuf", shuf)):
            M = (onehot(seqs) @ onehot(seqs).T) / 200.0
            out[f"{sp}_pair_{tag}"] = M[iu][keep]
            np.fill_diagonal(M, -1.0)
            out[f"{sp}_nn_{tag}"] = M.max(axis=1)

        # ---- D: what each condition does to the CDS ------------------------ #
        for cond in CONDS:
            dc = pd.read_csv(bench / f"prompts_{cond}_{sp}.csv")
            assert (dc.locus_tag.values == d0.locus_tag.values).all(), cond
            out[f"{sp}_{cond}_pct"] = dc.pct_nt_changed.to_numpy(float)
            Xk = kmer_freq([p[2:-2] for p in dc.prompt.astype(str)])
            out[f"{sp}_{cond}_cos"] = 1.0 - (Xk0 * Xk).sum(axis=1)
            out[f"{sp}_{cond}_dgc"] = (
                dc.gc_cds.to_numpy(float) - dc.gc_cds_c0.to_numpy(float)
            )
    return out


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def panel_a(fig, gs, S) -> None:
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]

    # only the set actually prompted: the singleton genes carrying a 200 bp
    # upstream. The filters that produced it are stated in the caption.
    ax = axes[0]
    y = np.arange(len(SPECIES))[::-1]
    n = [int(S[f"{sp}_n"][0]) for sp in SPECIES]
    ax.barh(y, n, height=0.55, color=[SP_COLOR[sp] for sp in SPECIES],
            edgecolor=C["ink"], linewidth=1.6)
    for yy, v in zip(y, n):
        ax.text(v + 35, yy, f"{v:,}", ha="left", va="center", fontsize=12,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\it{{{SP_LABEL[sp].split()[0]}}}$ "
                        f"$\\it{{{SP_LABEL[sp].split()[1]}}}$" for sp in SPECIES],
                       fontsize=12)
    ax.set_xlim(0, 2300)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("singleton genes with a 200 bp upstream")
    thicken(ax)
    title(ax, "The gene set")
    panel_label(ax, "A", dy=1.32)

    ax = axes[1]
    bins = np.logspace(np.log10(60), np.log10(8000), 46)
    for sp in SPECIES:
        L = S[f"{sp}_cds_len"]
        ax.hist(L, bins=bins, histtype="step", linewidth=2.4,
                color=SP_COLOR[sp],
                label=f"{SP_LABEL[sp]}  median {np.median(L):,.0f} nt")
    ax.set_xscale("log")
    ax.set_xlabel("CDS length (nt)")
    ax.set_ylabel("genes")
    ax.legend(loc="upper right", handlelength=1.3)
    thicken(ax)


def panel_b(fig, gs, S) -> None:
    sub = gs[0, 1].subgridspec(2, 2, hspace=0.55, wspace=0.30,
                               height_ratios=[1.0, 0.72])
    scat = [fig.add_subplot(sub[0, j]) for j in range(2)]
    for ax, sp in zip(scat, SPECIES):
        pcs, cls = S[f"{sp}_pcs"], S[f"{sp}_cai_class"]
        var = S[f"{sp}_pcvar"]
        ax.scatter(*pcs[cls == 0].T, s=7, c=C["grey_l"], linewidths=0,
                   label="all genes", rasterized=True)
        ax.scatter(*pcs[cls == -1].T, s=11, c=CAI_COLOR["low"], linewidths=0,
                   label=f"low CAI ({N_TOP})", rasterized=True)
        ax.scatter(*pcs[cls == 1].T, s=11, c=CAI_COLOR["high"], linewidths=0,
                   label=f"high CAI ({N_TOP})", rasterized=True)
        r = S[f"{sp}_pc_cai_r"]
        ax.text(0.03, 0.975,
                f"$r$(CAI, PC1) = {r[0]:.2f}\n$r$(CAI, PC2) = {r[1]:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.2)
        ax.text(0.97, 0.975,
                f"$\\it{{{SP_LABEL[sp].split()[0]}}}$ "
                f"$\\it{{{SP_LABEL[sp].split()[1]}}}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
                fontweight="bold")
        ax.set_xlabel(f"PC1 ({100*var[0]:.0f}%)", labelpad=10)
        ax.set_ylabel(f"PC2 ({100*var[1]:.0f}%)", labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(2.0)
    scat[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3,
                   handletextpad=0.25, columnspacing=1.2, fontsize=10,
                   markerscale=1.8)
    fig.text(0.735, 0.965, "The prompt: codon usage across CDSs",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    panel_label(scat[0], "B", dx=-0.22, dy=1.22)

    ax = fig.add_subplot(sub[1, :])
    for sp in SPECIES:
        d = S[f"{sp}_cds_between"]
        ax.hist(d, bins=60, density=True, histtype="step", linewidth=2.4,
                color=SP_COLOR[sp],
                label=f"{SP_LABEL[sp]}  median {np.median(d):.2f}")
    ax.set_xlabel(f"pairwise {K}-mer cosine distance between CDSs")
    ax.set_ylabel("density")
    ax.set_xlim(-0.03, 0.9)
    ax.legend(loc="upper right", handlelength=1.3)
    thicken(ax)


def panel_c(fig, gs, S) -> None:
    sub = gs[1, 0].subgridspec(2, 1, hspace=0.62, height_ratios=[1.0, 0.80])
    ax = fig.add_subplot(sub[0])
    x = np.arange(-200, 0)
    for sp in SPECIES:
        ax.plot(x, S[f"{sp}_pos_identity"], lw=2.0, color=SP_COLOR[sp],
                label=SP_LABEL[sp])
        ax.axhline(S[f"{sp}_pos_floor"][0], color=SP_COLOR[sp], lw=1.6,
                   ls=(0, (4, 2)), alpha=0.85)
    ax.axvspan(-20, 0, color=C["amber"], alpha=0.13, zorder=0)
    ax.text(-10, ax.get_ylim()[1], "SD", ha="center", va="bottom", fontsize=10.5,
            fontweight="bold")
    ax.set_xlabel(f"position relative to the start codon (bp)")
    ax.set_ylabel("cross-gene identity")
    ax.set_xlim(-200, 0)
    ax.legend(loc="upper left", handlelength=1.3, ncol=2)
    thicken(ax)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    title(ax, "The target: the real 200 bp upstreams")
    panel_label(ax, "C")

    ax2 = fig.add_subplot(sub[1])
    keys = [("pair_real", "all pairs"), ("pair_shuf", "all pairs\nshuffled"),
            ("nn_real", "nearest\nneighbour"), ("nn_shuf", "nearest nb.\nshuffled")]
    pos, data, cols = [], [], []
    for i, (key, _) in enumerate(keys):
        for j, sp in enumerate(SPECIES):
            pos.append(i + (j - 0.5) * 0.36)
            data.append(S[f"{sp}_{key}"])
            cols.append(SP_COLOR[sp])
    bp = ax2.boxplot(data, positions=pos, widths=0.30, showfliers=False,
                     patch_artist=True, medianprops=dict(color="white", lw=2.0),
                     whiskerprops=dict(lw=1.8), capprops=dict(lw=1.8),
                     boxprops=dict(lw=1.6, edgecolor=C["ink"]))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels([lab for _, lab in keys], fontsize=9.8)
    ax2.set_ylabel("sequence identity")
    ax2.set_xlim(-0.5, len(keys) - 0.5)
    thicken(ax2)


def panel_d(fig, gs, S) -> None:
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.30, height_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(sub[i]) for i in range(2)]
    xs = np.arange(len(CONDS))

    for ax, key, conds, ylab in zip(
        axes, ("pct", "cos"), (CONDS_PCT, CONDS),
        ("nucleotides changed (%)", f"{K}-mer cosine distance from C0"),
    ):
        cx = np.array([CONDS.index(c) for c in conds], dtype=float)
        for j, sp in enumerate(SPECIES):
            data = [S[f"{sp}_{c}_{key}"] for c in conds]
            parts = ax.violinplot(data, positions=cx + (j - 0.5) * 0.34,
                                  widths=0.30, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(SP_COLOR[sp])
                b.set_edgecolor(C["ink"])
                b.set_linewidth(1.2)
                b.set_alpha(0.92)
            ax.scatter(cx + (j - 0.5) * 0.34, [np.median(v) for v in data],
                       s=14, color="white", zorder=5, linewidths=0)
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.55, len(CONDS) - 0.45)
        thicken(ax)

    # between-gene reference band, same metric as the lower axis
    for sp in SPECIES:
        b = S[f"{sp}_cds_between"]
        lo, hi = np.percentile(b, [25, 75])
        axes[1].axhspan(lo, hi, color=SP_COLOR[sp], alpha=0.14, zorder=0)
        axes[1].axhline(np.median(b), color=SP_COLOR[sp], lw=1.8,
                        ls=(0, (4, 2)), zorder=1)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([])
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{c}\n{COND_NOTE[c]}" for c in CONDS], fontsize=9.5)
    handles = [mpl.patches.Patch(facecolor=SP_COLOR[sp], edgecolor=C["ink"],
                                 label=SP_LABEL[sp]) for sp in SPECIES]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                   ncol=1, handlelength=1.2)
    title(axes[0], "The conditions: how far each operator moves the CDS")
    panel_label(axes[0], "D")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def render(S: dict, stems: list[Path]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    set_style()
    fig = plt.figure(figsize=(15.6, 12.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24,
                  left=0.060, right=0.985, top=0.935, bottom=0.055)
    gs_a = gs[0, 0].subgridspec(2, 1, hspace=0.55, height_ratios=[0.40, 1.0])
    panel_a(fig, gs_a, S)
    panel_b(fig, gs, S)
    panel_c(fig, gs, S)
    panel_d(fig, gs, S)
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"))
        print("wrote", stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=ISAMBARD_BENCH)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    stats_path = args.stats or (
        args.bench / STATS_NAME if args.bench.exists() else HERE / STATS_NAME
    )

    if args.bench.exists() and (args.stats_only or not stats_path.exists()):
        S = compute_stats(args.bench)
        np.savez_compressed(stats_path, **S)
        print("wrote", stats_path)
    else:
        S = dict(np.load(stats_path))
        print("loaded", stats_path)

    if args.stats_only:
        outdir = args.outdir or stats_path.parent
        render(S, [outdir / "cds_prompted_dataset_supp"])
        return

    render(S, [ROOT / "results" / "cds_prompted_dataset_supp",
               ROOT / "figures" / "cds_prompted_dataset_supp"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Supplementary figure for the upstream-only benchmark (simplified, 6 panels).

  A  sigma-factor composition of the 681 genes
  B  where the annotated elements sit relative to the start codon
  C  distal -> proximal box spacer
  D  de novo Shine-Dalgarno logo over the last 20 bp (no motif assumed)
  E  how much of each window the clean background holds fixed
  F  control: mononucleotide composition preserved by the shuffle

Dropped relative to the 9-panel version (none of them are cited in the text):
transcription-factor regulators, the discriminator gap, the SD -> ATG spacing,
and the enriched/depleted 4-mer bar chart. Panel E is new: it quantifies the
clean control that the main text describes but never shows.

Everything is recomputed from data/prompts_681_clean.csv, so the figure is
reproducible end to end. The shuffled control (F) uses a fixed seed.

Outputs: results/ and figures/ , as .pdf and .png
"""

from __future__ import annotations

from pathlib import Path

import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "prompts_681_clean.csv"
DENOVO = ROOT / "results" / "upstream_only" / "sd_denovo"
OUT_STEMS = [ROOT / "results" / "upstream_only_dataset_supp",
             ROOT / "figures" / "upstream_only_dataset_supp"]

SEED = 0
N_SHUFFLES = 20
BASES = list("ACGT")
K = 4
SD_CORE = (-12, -7)          # de novo strongest 6 bp window, relative to the A of ATG
BAND = (5, 8)                # SD spacing core band
MINUS = "−"             # typographic minus, not hyphen
ARROW = r"$\rightarrow$"     # mathtext: always available, unlike Helvetica U+2192


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # Arial first: macOS "Helvetica.ttc" does not expose a separate bold face
        # to matplotlib, so bold silently falls back to regular.
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,          # editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        # thick axes
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


# colourblind-safe working palette
C = {
    "blue":   "#2166AC",
    "blue_l": "#92C5DE",
    "orange": "#D95F02",
    "amber":  "#E6A700",
    "green":  "#1B7837",
    "green_l": "#A6DBA0",
    "pink":   "#C51B7D",
    "teal":   "#01807F",
    "purple": "#762A83",
    "red":    "#B2182B",
    "grey":   "#8C8C8C",
    "ink":    "#1a1a1a",
}
PIE_COLORS = [C["blue"], C["amber"], C["green"], C["red"], C["pink"],
              C["blue_l"], C["teal"], C["purple"], C["grey"]]
LOGO_COLORS = {"A": C["green"], "C": C["blue"], "G": "#E8A33D", "T": C["red"]}


def panel_label(ax, letter: str, dx: float = -0.17, dy: float = 1.13) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="top", ha="left", color=C["ink"])


def thicken(ax, n_minor_x: int = 0) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2.0)
        ax.spines[s].set_position(("outward", 6))
    if n_minor_x:
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n_minor_x))


def title(ax, text: str) -> None:
    ax.set_title(text, fontweight="bold", pad=11)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["spacer"] = d.m10_start - d.m35_end            # end of -35 -> start of -10
    d["disc"] = d.tss_idx - d.m10_end                # end of -10 -> TSS
    # element midpoints relative to the A of the start codon (position 0)
    d["p35"] = (d.m35_start + d.m35_end) / 2 - d.L
    d["p10"] = (d.m10_start + d.m10_end) / 2 - d.L
    d["ptss"] = d.tss_idx - d.L
    d["psd"] = (d.sd_start + d.sd_end) / 2 - d.L
    return d


def gc(seq: str) -> float:
    return 100.0 * (seq.count("G") + seq.count("C")) / len(seq)


_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def pie_panel(ax, labels, counts, ttl, ncol=3, pct_floor=3.0):
    total = sum(counts)
    wedges, _, autotexts = ax.pie(
        counts,
        colors=PIE_COLORS[:len(counts)],
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.0f}%" if p >= pct_floor else "",
        pctdistance=0.70,
        textprops=dict(fontsize=11, fontweight="bold", color="white"),
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for t, cnt in zip(autotexts, counts):
        if cnt / total < 0.07:
            t.set_color(C["ink"])
    title(ax, ttl)
    ax.legend(wedges, [f"{l} ({c})" for l, c in zip(labels, counts)],
              loc="upper center", bbox_to_anchor=(0.5, 0.02),
              ncol=ncol, handlelength=1.0, handleheight=1.0,
              columnspacing=1.1, labelspacing=0.5)
    ax.set_aspect("equal")


def panel_A(ax, d):
    order = ["sigma70", "sigma38", "sigma32", "sigma24", "sigma54"]
    vc = d.sigma.value_counts()
    counts = [int(vc.get(s, 0)) for s in order]
    counts.append(len(d) - sum(counts))
    labels = [s.replace("sigma", "σ") for s in order] + ["other"]
    pie_panel(ax, labels, counts, "Sigma factor")
    panel_label(ax, "A", dx=-0.10, dy=1.15)


def panel_B(ax, d):
    bins = np.arange(-200, 2, 4)
    for name, v, col in [("distal box", d.p35, C["blue"]),
                         ("proximal box", d.p10, C["amber"]),
                         ("TSS", d.ptss, C["green"]),
                         ("SD", d.psd, C["orange"])]:
        ax.hist(v.dropna(), bins=bins, density=True, histtype="step",
                lw=2.3, color=col, label=name)
    ymax = ax.get_ylim()[1]
    ax.axvline(0, color=C["ink"], ls="--", lw=2.0)
    ax.text(-4, ymax * 0.99, "ATG ", ha="right", va="top",
            fontsize=11, fontweight="bold")
    ax.set_xlabel("element midpoint relative to ATG (bp)")
    ax.set_ylabel("density")
    ax.set_xlim(-206, 14)
    ax.set_ylim(0, ymax * 1.06)
    title(ax, "Where the elements sit")
    ax.legend(loc="upper left", handlelength=1.7, borderpad=0.2)
    thicken(ax, 2)
    panel_label(ax, "B")


def _gap_panel(ax, values, color, ttl, xlabel, letter, bins, ticks):
    med = float(np.median(values))
    ax.hist(values, bins=bins, color=color, edgecolor="white", lw=1.0, zorder=3)
    ax.axvline(med, color=C["ink"], ls="--", lw=2.3, zorder=4)
    ax.text(0.97, 0.95, f"median {med:.0f} bp", transform=ax.transAxes,
            ha="right", va="top", fontsize=11.5, fontweight="bold")
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("genes")
    title(ax, ttl)
    thicken(ax, 2)
    panel_label(ax, letter)


def panel_C(ax, d):
    _gap_panel(ax, d.spacer.values, C["blue"], "Distal / proximal box spacer",
               f"gap: end of distal box {ARROW} start of proximal box (bp)", "C",
               np.arange(5.5, 25.5, 1), np.arange(6, 25, 3))


def panel_D(ax, d):
    """NEW: de novo SD logo over the last 20 bp of every upstream."""
    info = pd.read_csv(DENOVO / "sd_denovo_pfm_information_uniform.csv",
                       index_col="pos_from_ATG")[BASES]
    perpos = pd.read_csv(DENOVO / "sd_denovo_per_position.csv",
                         index_col="pos_from_ATG")
    positions = list(info.index)
    ytop = float(info.sum(axis=1).max())

    logo = logomaker.Logo(info.reset_index(drop=True), ax=ax,
                          color_scheme=LOGO_COLORS, show_spines=False,
                          vpad=0.02, width=0.9)
    logo.style_spines(spines=["left", "bottom"], visible=True, linewidth=2.0)

    lo, hi = SD_CORE
    i_lo, i_hi = positions.index(lo), positions.index(hi)
    ax.axvspan(i_lo - 0.5, i_hi + 0.5, color=C["amber"], alpha=0.14, zorder=0)

    consensus = "".join(perpos.loc[lo:hi, "consensus"])
    ax.text((i_lo + i_hi) / 2, ytop * 1.04, consensus, ha="center", va="bottom",
            fontsize=14, fontweight="bold", family="monospace", color=C["ink"])

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{p}".replace("-", MINUS) for p in positions])
    ax.set_xlabel("position relative to the A of the start codon (bp)")
    ax.set_ylabel("bits")
    ax.set_ylim(0, ytop * 1.26)
    title(ax,
          f"Shine–Dalgarno motif recovered de novo — last 20 bp of all {len(d)} "
          "upstreams, no motif or alignment assumed  "
          f"(strongest 6 bp window: {MINUS}12 to {MINUS}7)")
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 6))
    ax.tick_params(width=2.0, length=6.0)
    panel_label(ax, "D", dx=-0.058, dy=1.20)


def kept_fraction(d) -> np.ndarray:
    """Percent of each 200 bp window held at its real bases in the clean prompt.

    The clean background keeps the two boxes, the TSS, the Shine-Dalgarno and
    every annotated transcription-factor site in place, and shuffles the rest.
    """
    out = []
    for r in d.itertuples():
        keep = np.zeros(int(r.L), dtype=bool)

        def mark(a, b):
            a, b = int(max(0, a)), int(min(r.L, b))
            if b > a:
                keep[a:b] = True

        mark(r.m35_start, r.m35_end)
        mark(r.m10_start, r.m10_end)
        mark(r.tss_idx, r.tss_idx + 1)
        if not pd.isna(r.sd_start):
            mark(r.sd_start, r.sd_end)
        if isinstance(r.tfbs_spans, str) and r.tfbs_spans != "nan":
            for span in r.tfbs_spans.split(";"):
                if "-" in span:
                    a, b = span.split("-")
                    mark(int(a), int(b))
        out.append(100.0 * keep.mean())
    return np.array(out)


def panel_E(ax, kept):
    """NEW: how much of the window the clean background holds fixed."""
    med = float(np.median(kept))
    ax.hist(kept, bins=np.arange(0, 90, 2.5), color=C["green_l"],
            edgecolor=C["ink"], lw=1.0, zorder=3)
    ax.axvline(med, color=C["ink"], ls="--", lw=2.3, zorder=4)
    ax.text(0.97, 0.95,
            f"median {med:.0f}% held fixed\n(IQR {np.percentile(kept, 25):.0f}"
            f"–{np.percentile(kept, 75):.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold", linespacing=1.5)
    ax.set_xlabel("annotated bases held fixed in the clean prompt (% of 200 bp)")
    ax.set_ylabel("genes")
    title(ax, "Clean background: elements kept, the rest shuffled")
    thicken(ax, 2)
    panel_label(ax, "E", dx=-0.10)


def panel_F(ax, nat_gc, shuf_gc):
    bins = np.arange(15, 75, 1.5)
    ax.hist(nat_gc, bins=bins, color=C["blue_l"], edgecolor="none",
            label="natural", zorder=2)
    ax.hist(shuf_gc, bins=bins, histtype="step", lw=2.3, color=C["orange"],
            label="shuffled", zorder=3)
    dmax = float(np.max(np.abs(np.asarray(nat_gc) - np.asarray(shuf_gc))))
    ax.text(0.03, 0.96, f"max |Δ| = {dmax:.0f}%\nper-sequence permutation,\n"
                        "so GC is preserved exactly",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            fontweight="bold", linespacing=1.5)
    ax.set_xlabel("GC content of the 200 bp (%)")
    ax.set_ylabel("genes")
    title(ax, "Control: mononucleotide composition preserved")
    ax.legend(loc="upper right", handlelength=1.7)
    thicken(ax, 2)
    panel_label(ax, "F", dx=-0.10)


# --------------------------------------------------------------------------- #
def main() -> None:
    set_style()
    d = load()

    rng = np.random.default_rng(SEED)
    nat = d.upstream_seq.tolist()
    shuf = ["".join(rng.permutation(list(s))) for s in nat]

    nat_gc = [gc(s) for s in nat]
    shuf_gc = [gc(s) for s in shuf]
    kept = kept_fraction(d)

    fig = plt.figure(figsize=(15.5, 12.4))
    gs = GridSpec(3, 12, figure=fig,
                  height_ratios=[1.10, 0.72, 0.94],
                  hspace=0.62, wspace=1.35,
                  left=0.055, right=0.985, top=0.955, bottom=0.055)

    panel_A(fig.add_subplot(gs[0, 0:4]), d)
    panel_B(fig.add_subplot(gs[0, 4:8]), d)
    panel_C(fig.add_subplot(gs[0, 8:12]), d)

    panel_D(fig.add_subplot(gs[1, 0:12]), d)

    panel_E(fig.add_subplot(gs[2, 0:6]), kept)
    panel_F(fig.add_subplot(gs[2, 6:12]), nat_gc, shuf_gc)

    # No suptitle: the figure caption carries "Upstream-only benchmark: 681
    # E. coli genes, 200 bp upstream, RegulonDB-annotated".

    for stem in OUT_STEMS:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"))
        print(f"wrote {stem.with_suffix('.pdf')}")
        print(f"wrote {stem.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()

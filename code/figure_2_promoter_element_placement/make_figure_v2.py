"""
[v2] Main figure for the upstream-only benchmark — 2x2, workshop format.

Differences from make_figure.py, which is left untouched:
  * Evo 1 (131k) added as a fifth model in (a), (b) and (c).
  * (d) redrawn on a log context axis out to each checkpoint's trained limit
    (200 bp - 450 kb) instead of stopping at 1 kb, and carrying the real-promoter
    reference on a broken axis so the gap to real sequence is visible.
  * Reads the *_v2 input CSVs; the published ones are not overwritten.
  * (a) and (c) are regenerated after the score.py off-by-width fix, which moved
    the -10 comparison offset onto the last CORE bases of annotations wider than
    6 bp. That changes nothing for the four published models -- their generated
    values reproduce exactly -- but their shuffled controls shift ~0.3pp because
    the shuffle RNG stream depends on how many models are in the pass.

Output: fig_main_2x2_v2.{pdf,png}

Reads only the CSVs in this folder, so it runs anywhere with pandas +
matplotlib. Nothing here touches Isambard or re-runs a model.

  (a) correct -35 -> -10 spacing, per model vs real promoters
  (b) per-base probability across the whole generated upstream, ONE TRACE PER
      MODEL, from just after the -35 all the way to the start codon
  (c) "contains a -10 box" is met just as often by a shuffle of the same sequence
  (d) placement (+-2 bp) vs how much upstream context the model was given

Panel (d) uses a +-2 bp tolerance rather than exact placement. Loosening the
criterion does not change the conclusion -- at +-0/1/2/3/5 bp the models track
the corresponding chance rate (2.5/7.5/12.5/17.5/27.5%) in lockstep -- so +-2 is
used simply because it is the biologically sensible window.

PROK is excluded: it is the paper's own fine-tuned model and belongs to the
CDS-prompted track. Every reference value comes from natural sequence -- the -10
weight matrix was built from the annotated real upstreams, never from model
output.

    python make_figure.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mt
import matplotlib.colors as mcolors

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(_ROOT, "results", "upstream_only", "fig2x2")   # panel CSVs live here

COLORS = {"evo1": "#2f6fb0", "evo1_131k": "#78a9dc", "evo2": "#3f8f5e",
          "hyenadna": "#a34ba3", "promogen2": "#9a7b28"}
DISPLAY = {"evo1": "Evo 1 (8k)", "evo1_131k": "Evo 1 (131k)", "evo2": "Evo 2",
           "hyenadna": "HyenaDNA", "promogen2": "PromoGen2"}
ORDER = ["evo1", "evo1_131k", "evo2", "hyenadna", "promogen2"]
# (a) and (c) print the model names side by side under the bars, where the two
# Evo 1 checkpoints would otherwise run into each other: break them over two
# lines there only.
TICK_DISPLAY = {"evo1": "Evo 1\n(8k)", "evo1_131k": "Evo 1\n(131k)"}
BASE_COL = {"A": "#2f8f4e", "C": "#2f6fb0", "G": "#e0a33e", "T": "#c94f4f"}
REAL_COL = "#1a1a1a"

INK, INK2, MUTED, GRID = "#1a1a1a", "#3d3d3d", "#8a8a8a", "#dcdcdc"
AXW, TICKW = 1.4, 1.3
M10_POS = 17        # the -10 box sits 17 bp after the -35
CORE = 6            # -10 box width, for panel (c)
TSS_POS = 29        # median distance from the -35 to the TSS

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.linewidth": AXW,
    "xtick.major.width": TICKW, "ytick.major.width": TICKW,
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def lighten(c, f=0.55):
    """Blend a model colour towards white -- used for the clean-background bar."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(AXW)
    ax.set_axisbelow(True)


def tag(ax, letter, dx=-0.17, dy=1.16):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left", color=INK)


# ---------------------------------------------------------------- (a)
def panel_a(ax):
    """sigma70: does the -10 land within +-2 bp of ITS OWN annotated position?

    Per-promoter, not a fixed 15-19 bp window: real spacers vary, so each
    generation is scored against the geometry its own promoter actually has.
    The fixed window inflated the positive control (75% vs 67%) by crediting
    hits at the canonical offset even when that promoter's real box sits
    elsewhere.

    Three references are drawn: real promoters (the positive control, what the
    task looks like when it is done right), a composition-preserving shuffle of
    each model's own output (the negative control, what the identical scan
    returns when the sequence carries no grammar at all), and the same two bars
    on the clean background, where the genomic filler has been permuted away --
    whatever a model loses between leaked and clean was memorisation.
    """
    d = pd.read_csv(f"{HERE}/panelA_tol2_bg_v2.csv")
    real = d[d.kind == "real"].pct.iloc[0]
    ms = [m for m in ORDER if m in set(d.source)]
    x = np.arange(len(ms))

    ax.axhline(real, color=REAL_COL, lw=2.2, ls=(0, (5, 2.5)), zorder=4)
    ax.text(len(ms) - 0.45, real - 3.5,
            f"real σ70 promoters   {real:.0f}%", fontsize=8.6,
            color=REAL_COL, ha="right", va="top")

    def val(m, kind, bg=None):
        s = d[(d.source == m) & (d.kind == kind)]
        return s.pct.mean() if bg is None else s[s.background == bg].pct.iloc[0]

    # three bars per model: leaked, clean, and the negative control matched to
    # the leaked bar -- the shuffle of the clean generations is within 0.8 pp of
    # it and is reported in the text rather than drawn.
    w = 0.27
    for kind, bg, off in (("generated", "leaked", -1.0),
                          ("generated", "clean", 0.0),
                          ("shuffled", "leaked", 1.0)):
        v = [val(m, kind, bg) for m in ms]
        is_gen = kind == "generated"
        ax.bar(x + off * w, v, w * 0.9,
               color=([COLORS[m] if bg == "leaked" else lighten(COLORS[m])
                       for m in ms] if is_gen else "none"),
               edgecolor="white" if is_gen else INK2,
               linewidth=1.2 if is_gen else 1.4,
               hatch=None if is_gen else "\\\\", zorder=3)
        for xi, vv in zip(x + off * w, v):
            ax.text(xi, vv + 1.4, f"{vv:.0f}", ha="center", fontsize=7.8,
                    color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels([TICK_DISPLAY.get(m, DISPLAY[m]) for m in ms], fontsize=8.4)
    ax.set_ylabel("−10 box within ±2 bp of\nits own true position (%)")
    ax.set_ylim(0, 88)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor="white",
                             label="real prompt, unchanged (leaked)"),
                       Patch(facecolor=lighten(MUTED), edgecolor="white",
                             label="boxes kept, background\nshuffled (clean)"),
                       Patch(facecolor="none", edgecolor=INK2, hatch="\\\\",
                             label="generated output shuffled\n(negative control)")],
              frameon=False, fontsize=7.2, loc="upper left", ncol=1,
              bbox_to_anchor=(0.0, 0.68), handlelength=1.5, labelspacing=0.55)
    tidy(ax)
    tag(ax, "A")
    # panel headings live in the LaTeX caption, not on the axes


# ---------------------------------------------------------------- (b)
def panel_b(gs, fig):
    """One trace per model: P(base) from just after the −35 to the start codon."""
    d = pd.concat([pd.read_csv(f"{HERE}/panelB_real.csv"),
                   pd.read_csv(f"{HERE}/panelB_toatg_v2.csv")])
    # real sequence first, so the reader sees what the target looks like before
    # seeing that no model reproduces it
    rows = ["real"] + ORDER
    inner = gs.subgridspec(len(rows), 1, hspace=0.30)
    axes = []
    for i, m in enumerate(rows):
        ax = fig.add_subplot(inner[i])
        axes.append(ax)
        # panelB_toatg_v2.csv carries both prompt backgrounds for the models
        # that were rerun; keep the leaked one so every trace comes from the
        # same condition (otherwise the two series interleave into one jagged
        # line at each position).
        g = d[d.model == m]
        if "background" in g and (g.background == "leaked").any():
            g = g[g.background == "leaked"]
        g = g.sort_values("position")
        g = g[g.n >= g.n.max() * 0.25]          # drop the thin long-gene tail
        for b in "ACGT":
            ax.plot(g.position, g[f"p_{b}"], color=BASE_COL[b], lw=1.5)
        ax.axvline(M10_POS, color=REAL_COL, lw=1.2, ls=(0, (3, 2.5)), zorder=1)
        ax.axhline(0.25, color=MUTED, lw=0.9, ls=":", zorder=0)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.25, 0.5])
        lab = "Real σ70 promoters" if m == "real" else DISPLAY[m]
        col = REAL_COL if m == "real" else COLORS[m]
        ax.text(0.015, 0.98, lab, transform=ax.transAxes, fontsize=8.4,
                color=col, ha="left", va="top", fontweight="bold")
        if m == "real":
            ax.axvspan(17, 22, color="#ffe9a8", zorder=0)
        ax.axvline(TSS_POS, color=MUTED, lw=1.1, ls=(0, (1.5, 2)), zorder=1)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        tidy(ax)
        if i < len(rows) - 1:
            ax.set_xticklabels([])
    axes[0].text(M10_POS + 2.5, 0.70, "−10 box", fontsize=7.8, color=REAL_COL,
                 va="top", ha="left")
    axes[0].text(TSS_POS + 1.5, 0.70, "TSS", fontsize=7.8, color=MUTED, va="top")
    axes[-1].set_xlabel("position after the −35 box (bp)")
    axes[len(axes) // 2].set_ylabel("probability of each base", fontsize=8.6)
    axes[len(axes) // 2].yaxis.set_label_coords(-0.115, 1.05)

    axes[0].legend(handles=[Line2D([], [], color=BASE_COL[b], lw=1.8, label=b)
                            for b in "ACGT"],
                   frameon=False, fontsize=8, ncol=4, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.3,
                   columnspacing=1.2, handletextpad=0.4)
    tag(axes[0], "B", dx=-0.20, dy=2.30)
    # heading in the caption; the pad is kept so the legend clears the axes
    axes[0].set_title(" ", fontsize=9.4, pad=32)
    return axes


# ---------------------------------------------------------------- (c)
def panel_c(ax):
    """Does the best proximal box a model writes carry the native bases?

    Panel (a) asks WHERE the box lands, and the annotated position is not a fair
    place to read bases from: the generations do not put a box there. So this
    panel lets each generation nominate its own box -- scan the 45 bp with the
    same PWM used in (a), keep the highest-scoring six-base window -- and counts
    how many of those six bases match that gene's annotated proximal box.

    The ceiling is not 6/6. Running the identical scan on the real continuation
    recovers the annotated box in only 320 of 487 sigma70 promoters, so real
    sequence itself scores 4.99/6; that is the positive control drawn here.
    """
    d = pd.read_csv(f"{HERE}/highest_pwm_base_matches.csv")
    d = d[d.task == "generate_promoter"]
    ref = pd.read_csv(f"{HERE}/highest_pwm_positive_control.csv")
    real = ref[ref.task == "generate_promoter"].mean_matches.iloc[0]
    ms = [m for m in ORDER if m in set(d.model)]
    x = np.arange(len(ms)); w = 0.21

    ax.axhline(real, color=REAL_COL, lw=2.2, ls=(0, (5, 2.5)), zorder=4)
    ax.text(len(ms) - 0.45, real - 0.12,
            f"real \u03c370 continuation   {real:.1f}", fontsize=8.6,
            color=REAL_COL, ha="right", va="top")

    for kind, bg, off in (("generated", "leaked", -1.0),
                          ("generated", "clean", 0.0),
                          ("shuffled", "leaked", 1.0)):
        s = d[(d.kind == kind) & (d.background == bg)].set_index("model")
        v = [s.loc[m].mean_matches for m in ms]
        is_gen = kind == "generated"
        ax.bar(x + off * w, v, w * 0.9,
               color=([COLORS[m] if bg == "leaked" else lighten(COLORS[m])
                       for m in ms] if is_gen else "none"),
               edgecolor="white" if is_gen else INK2,
               linewidth=1.2 if is_gen else 1.4,
               hatch=None if is_gen else "\\\\", zorder=3)
        # no value labels here: five models x three bars puts 15 numbers in a
        # narrow band around 3, which is unreadable. The y-axis and the error
        # bars carry the same information.

    ax.set_xticks(x)
    ax.set_xticklabels([TICK_DISPLAY.get(m, DISPLAY[m]) for m in ms], fontsize=8.4)
    ax.set_ylabel("native bases in the best-scoring\nproximal box written (of 6)")
    ax.set_ylim(0, 6)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    tidy(ax)
    tag(ax, "C")
    # heading in the caption


# ---------------------------------------------------------------- (d)
def panel_d(gs, fig):
    """Broken y-axis: real promoters sit at 67.6% and the models between 10
    and 17%, so one continuous axis would flatten every curve. The thin upper
    strip carries the real-promoter reference; the lower panel keeps the models
    legible."""
    inner = gs.subgridspec(2, 1, height_ratios=[1, 9], hspace=0.06)
    axt = fig.add_subplot(inner[0])
    ax = fig.add_subplot(inner[1])
    REAL = 67.6
    d = pd.read_csv(f"{HERE}/panelD_context_tol2_v2.csv")
    CH = 12.5
    ax.axhspan(0, CH, color="#f0f0f0", zorder=0)
    ax.axhline(CH, color=INK, lw=1.5, ls=(0, (5, 2.5)), zorder=4)
    ax.text(30000, CH - 0.25, "chance", fontsize=8.4, color=INK, style="italic",
            ha="center", va="top")
    for m in ORDER:
        s = d[(d.model == m) & (~d.is_inter)].sort_values("x")
        if s.empty:
            continue
        ax.plot(s.x, s.exact, color=COLORS[m], lw=2.1, zorder=5, clip_on=False)
        # hollow markers where the point is the fixed 128-gene subset rather
        # than all 680 genes, so the two are not read as equivalent
        for xi, yi, ni in zip(s.x, s.exact, s.n_genes):
            full = ni >= 680
            ax.plot([xi], [yi], marker="o", ms=5, zorder=6, color=COLORS[m],
                    markerfacecolor=COLORS[m] if full else "white",
                    markeredgecolor=COLORS[m], markeredgewidth=1.3,
                    clip_on=False)
        ax.fill_between(s.x, s.lo, s.hi, color=COLORS[m], alpha=0.16, lw=0,
                        zorder=2)
        ax.annotate(DISPLAY[m], (s.x.iloc[-1], s.exact.iloc[-1]),
                    xytext=(8, {"evo2": 0, "hyenadna": 6, "evo1": 0,
                                "evo1_131k": -7}.get(m, 0)),
                    textcoords="offset points", fontsize=8.4, color=COLORS[m],
                    va="center")
    ax.set_xscale("log")
    TICKS = [200, 300, 500, 1000, 8000, 131000, 449000]
    ax.xaxis.set_major_locator(mt.FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(
        mt.FixedFormatter(["200", "300", "500", "1k", "8k", "131k", "450k"]))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    # 200/300/500 sit almost on top of each other on a log axis
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    ax.set_xlim(170, 900000); ax.set_ylim(9.5, 17.5)
    ax.set_xlabel("upstream sequence given to the model (bp)")
    ax.set_ylabel("−10 box within ±2 bp\nof the right place (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    tidy(ax)

    # upper strip: the same +/-2 bp scan on real annotated sigma70 promoters
    axt.set_xscale("log")
    axt.set_xlim(*ax.get_xlim())
    axt.axhline(REAL, color=INK, lw=1.5, ls=(0, (5, 2.5)), zorder=4)
    axt.text(200, REAL + 0.9, "real \u03c370 promoters   68%", fontsize=8.0,
             color=INK, ha="left", va="bottom")
    axt.set_ylim(65.5, 69.5)
    axt.set_yticks([REAL]); axt.set_yticklabels(["68"])
    axt.set_xticks([])
    axt.xaxis.set_minor_locator(mt.NullLocator())
    axt.tick_params(axis="x", which="both", length=0)
    for sp in ("top", "right", "bottom"):
        axt.spines[sp].set_visible(False)
    axt.spines["left"].set_linewidth(1.6)
    axt.tick_params(axis="y", labelsize=8)
    # break marks on the shared left spine
    for yy in (-0.11, -0.05):
        axt.plot([-0.012, 0.012], [yy, yy + 0.05], transform=axt.transAxes,
                 color=INK, lw=1.4, clip_on=False, zorder=6)
    tag(axt, "D")
    # heading in the caption


# ---------------------------------------------------------------- figure
def main():
    fig = plt.figure(figsize=(10.8, 8.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.60, wspace=0.36)

    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(gs[0, 1], fig)
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(gs[1, 1], fig)

    handles = [Line2D([], [], color=REAL_COL, lw=2.2, ls=(0, (5, 2.5)),
                      label="real σ70 promoters")]
    handles += [Line2D([], [], color=COLORS[m], lw=2.4, label=DISPLAY[m])
                for m in ORDER]
    handles += [Patch(facecolor="none", edgecolor=INK2, hatch="\\\\",
                      label="panels A, C: generated output shuffled")]
    fig.legend(handles=handles, ncol=5, frameon=False, fontsize=8.6,
               loc="lower center", bbox_to_anchor=(0.5, -0.055),
               columnspacing=1.6, handlelength=1.7)

    for ext in ("pdf", "png"):
        fig.savefig(f"{HERE}/fig_main_2x2_v2.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_main_2x2.pdf / .png")


if __name__ == "__main__":
    main()

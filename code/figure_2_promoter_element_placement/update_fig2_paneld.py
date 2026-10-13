"""Redraw panel (d) of Figure 2 with the long-context points.

The script that originally built figures/fig_upstream_only.pdf is not in this
repo, so panels (a)-(c) are left exactly as they are: panel (d) is re-rendered
on its own and painted into the existing page, the same vector-preserving trick
compose_fig1.py uses for Figure 1.

Colours and the ink grey are read off the existing PDF, not guessed:
  Evo 1 #2e6eaf, Evo 2 #3e8f5e, HyenaDNA #a24aa2, ink #1a1a1a, grid #dcdcdc.
Evo 1 (131k) is new to this panel and takes the pale blue of the Evo 1 family.

Only the context lengths requested for the axis are drawn -- 200, 300, 500, 1k,
8k, 131k, 450k -- and a model contributes a point only where it has one, so no
curve is extrapolated across a context it was never scored at.

Points measured on the fixed 128-gene subset (Evo 1 131k at 131k, HyenaDNA at
450k) are drawn with open markers, because they are not directly comparable with
the 680-gene points on the same curve.

    python3 code/figures/update_fig2_paneld.py
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

FIG = Path("figures/fig_upstream_only.pdf")
SUMMARIES = "data/fig2d_summaries/*.json"
PANEL_PDF = Path("figures/fig2_paneld.pdf")

# the panel (d) area of the existing page, including its letter and axis labels
PANEL_RECT = pymupdf.Rect(349, 291, 661, 537)

INK, GRID = "#1a1a1a", "#dcdcdc"
CHANCE = 12.5
# same +/-2 bp scan on real annotated sigma70 promoters, from
# benchmark/upstream_only/fig2x2/panelA_tol2.csv (67.6%, n=487). Panel (a)
# rounds it to 68% on its own reference line, so this label matches.
REAL_PCT, REAL_LABEL = 67.6, "real \u03c370 promoters   68%"
CONTEXTS = [200, 300, 500, 1000, 8000, 131000, 449000]
TICK_LABELS = ["200", "300", "500", "1k", "8k", "131k", "450k"]

SERIES = [  # (key, label, colour, label y-nudge in points)
    ("evo2", "Evo 2", "#3e8f5e", 0.0),
    ("hyenadna", "HyenaDNA", "#a24aa2", 0.0),
    ("evo1", "Evo 1", "#2e6eaf", 0.0),
    ("evo1_131k", "Evo 1 (131k)", "#78a9dc", 0.0),
]


def load() -> dict[tuple[str, int], dict]:
    """Published FFT-prefill numbers win; the teacher-forced 131k point is the
    only place a _forced summary is the sole source."""
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(glob(SUMMARIES)):
        if path.endswith("_sub128.json"):
            continue  # a re-scoring of an existing point, not a new one
        d = json.loads(Path(path).read_text())
        key = (d["model"], d["context_bp"])
        forced = path.endswith("_forced.json")
        if key in out and forced:
            continue  # prefer the unforced measurement where both exist
        lo, hi = d["within_2_gene_bootstrap_95"]
        out[key] = {"y": d["within_2_pct"], "lo": lo, "hi": hi,
                    "n": d["n_genes"], "forced": forced}
    return out


def main() -> None:
    data = load()
    w_in, h_in = PANEL_RECT.width / 72.0, PANEL_RECT.height / 72.0
    fig = plt.figure(figsize=(w_in, h_in))
    # Broken y-axis. Real promoters sit at 67.6% and the models between 10 and
    # 17%, so one continuous axis would squash every curve into a flat band at
    # the bottom. The thin upper strip carries the real-promoter reference and
    # keeps the gap visible; the lower panel keeps the models legible.
    # Right margin is deliberately wide: the four curves end at four different
    # context lengths, so the inline end labels need room outside the axes.
    ax = fig.add_axes([0.175, 0.235, 0.575, 0.50])
    axt = fig.add_axes([0.175, 0.80, 0.575, 0.075], sharex=ax)

    ax.set_xscale("log")
    ax.axhspan(0, CHANCE, color="#f1f1f1", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)

    for key, label, colour, nudge in SERIES:
        pts = [(c, data[(key, c)]) for c in CONTEXTS if (key, c) in data]
        if not pts:
            continue
        xs = [c for c, _ in pts]
        ys = [v["y"] for _, v in pts]
        lo = [v["lo"] for _, v in pts]
        hi = [v["hi"] for _, v in pts]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=colour, linewidth=1.6, zorder=3)
        # 128-gene subset points get open markers
        for (c, v) in pts:
            solid = v["n"] >= 680
            ax.plot([c], [v["y"]], marker="o", markersize=4.2, zorder=4,
                    color=colour,
                    markerfacecolor=colour if solid else "white",
                    markeredgecolor=colour, markeredgewidth=1.1)
        ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(5, nudge),
                    textcoords="offset points", color=colour,
                    fontsize=6.6, va="center", ha="left", clip_on=False)

    # upper strip: the real-promoter reference
    axt.set_xscale("log")
    axt.axhline(REAL_PCT, color=INK, linestyle="--", linewidth=1.2, zorder=3)
    axt.annotate(REAL_LABEL, xy=(165, REAL_PCT), xytext=(3, 2.5),
                 textcoords="offset points", color=INK, fontsize=6.4,
                 ha="left", va="bottom")
    axt.set_ylim(65.5, 69.5)
    axt.set_yticks([REAL_PCT])
    axt.set_yticklabels(["68"], fontsize=6.6, color=INK)
    for spine in ("top", "right", "bottom"):
        axt.spines[spine].set_visible(False)
    axt.spines["left"].set_color(INK)
    axt.spines["left"].set_linewidth(0.8)
    axt.tick_params(axis="y", colors=INK, labelsize=6.6, length=2.5, width=0.8)
    axt.tick_params(axis="x", length=0, labelbottom=False)

    # break marks across the two axes
    kw = dict(transform=fig.transFigure, color=INK, linewidth=0.8,
              clip_on=False, zorder=6)
    import matplotlib.lines as mlines
    for dy in (0.0,):
        fig.add_artist(mlines.Line2D([0.166, 0.184], [0.752 + dy, 0.766 + dy], **kw))
        fig.add_artist(mlines.Line2D([0.166, 0.184], [0.786 + dy, 0.800 + dy], **kw))

    ax.axhline(CHANCE, color=INK, linestyle="--", linewidth=1.2, zorder=3)
    ax.annotate("chance", xy=(20000, CHANCE), xytext=(0, -8.0),
                textcoords="offset points", color=INK, fontsize=6.4,
                style="italic", ha="left")

    ax.set_xticks(CONTEXTS)
    # 200/300/500 sit almost on top of one another on a log axis
    ax.set_xticklabels(TICK_LABELS, fontsize=6.2, color=INK,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.tick_params(axis="both", colors=INK, labelsize=6.6, length=2.5, width=0.8)
    ax.set_ylim(9.5, 17.2)
    ax.set_xlim(165, 620000)
    ax.set_xlabel("upstream sequence given to the model (bp)",
                  fontsize=7.2, color=INK)
    ax.set_ylabel("−10 box within ±2 bp\nof the right place (%)",
                  fontsize=7.2, color=INK)
    ax.yaxis.set_label_coords(-0.135, 0.62)
    axt.set_title("More context does not fix it", fontsize=8.0, color=INK,
                  loc="center", pad=5)

    fig.savefig(PANEL_PDF, format="pdf", transparent=True)
    plt.close(fig)

    doc = pymupdf.open(FIG)
    page = doc[0]
    # clear the old panel, then lay the new one in the same box
    page.draw_rect(PANEL_RECT, color=None, fill=(1, 1, 1), overlay=True)
    page.show_pdf_page(PANEL_RECT, pymupdf.open(PANEL_PDF), 0)
    # the panel letter was painted over with the rest; redraw it to match a-c
    page.insert_text((354.5, 307.0), "d", fontname="hebo", fontsize=9.0,
                     color=(0.102, 0.102, 0.102))
    doc.save(FIG.with_suffix(".new.pdf"))
    print("wrote", FIG.with_suffix(".new.pdf"))


if __name__ == "__main__":
    main()

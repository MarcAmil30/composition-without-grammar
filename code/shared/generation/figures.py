#!/usr/bin/env python3
"""
Publication figures for the upstream-only benchmark.

Reads scores/per_generation.csv.gz and scores/summary.csv; writes PDF + PNG.
Only models actually present in the scores are drawn, so this can be run while
generation is still finishing.

  fig1_placement   Can the model put the motif in the right place?
                   Position-error CDF per model, both tasks, leaked vs clean,
                   against the chance curve for a uniformly-placed argmax.

  fig2_signal      Is there any signal beyond base composition, and does it
                   come from memorisation?
                   (a) PWM score above each model's own composition-preserving
                       shuffle, (b) leaked-minus-clean gap, (c) start-codon
                       placement on task 2.

    python figures.py [--scores DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# CVD-checked categorical order (validate_palette.js: all checks pass; the
# orange/green pair sits in the 6-8 dE band, so every series is also direct-
# labelled or legended -- never colour alone).
COLORS = {"evo1": "#2f6fb0", "evo2": "#3f8f5e", "prok": "#d1701c",
          "hyenadna": "#a34ba3", "promogen2": "#9a7b28"}
DISPLAY = {"evo1": "Evo 1", "evo2": "Evo 2", "prok": "PROK",
           "hyenadna": "HyenaDNA", "promogen2": "PromoGen2"}
ORDER = ["evo1", "evo2", "prok", "hyenadna", "promogen2"]

TASK_LABEL = {"generate_promoter": "Task 1 — place the $-$10 box",
              "generate_sd": "Task 2 — place the Shine$-$Dalgarno"}
GEN_LEN = {"generate_promoter": 45, "generate_sd": 40}

INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#e4e4e4"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
    ax.tick_params(colors=INK2, length=3, width=0.7)


def chance_cdf(true_positions, gen_len, width=6):
    """CDF of |argmax - truth| when the argmax lands uniformly at random.

    The null a model must beat: it knows nothing, so its best-scoring window is
    equally likely anywhere in the generated span. Uses the EMPIRICAL truth
    positions rather than assuming they are uniform -- the real -10 offset is
    concentrated near 17 bp and the real SD near 7, which changes the curve.
    """
    n = gen_len - width + 1
    pred = np.arange(n)
    truth = np.asarray(true_positions)
    errs = np.abs(truth[:, None] - pred[None, :]).ravel()
    xs = np.arange(0, gen_len + 1)
    return xs, np.array([(errs <= x).mean() for x in xs])


# =====================================================================
def fig_placement(per, out):
    tasks = [t for t in ("generate_promoter", "generate_sd")
             if t in per.task.unique()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.3),
                             squeeze=False)
    axes = axes[0]

    for ax, task in zip(axes, tasks):
        sub = per[(per.task == task) & (per.decode == "sample")]
        truth = sub.drop_duplicates("locus_tag").true_pos.dropna().values
        xs, cdf = chance_cdf(truth, GEN_LEN[task])
        # drawn ON TOP: on task 1 every model lies exactly on it, and that
        # coincidence is the result -- it has to be visible, not buried
        ax.plot(xs, cdf, color="#1a1a1a", lw=1.6, ls=(0, (5, 2.5)), zorder=6)
        ax.text(xs[-1] * 0.58, np.interp(xs[-1] * 0.58, xs, cdf) - 0.10,
                "chance", color=INK, fontsize=7.5, style="italic")

        for m in [m for m in ORDER if m in sub.model.unique()]:
            for bg, ls, alpha in (("leaked", "-", 1.0), ("clean", (0, (1.6, 1.6)), 0.75)):
                d = sub[(sub.model == m) & (sub.background == bg)].pos_error.dropna()
                if d.empty:
                    continue
                v = np.sort(d.values)
                grid = np.arange(0, GEN_LEN[task] + 1)
                y = np.searchsorted(v, grid, side="right") / len(v)
                ax.plot(grid, y, color=COLORS[m], lw=1.7, ls=ls, alpha=alpha, zorder=3)

        ax.set_xlim(0, GEN_LEN[task]); ax.set_ylim(0, 1.0)
        ax.set_xlabel("position error |predicted $-$ true| (bp)", fontsize=8.6)
        ax.set_title(TASK_LABEL[task], fontsize=9.0, color=INK, loc="left", pad=6)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        tidy(ax)
    axes[0].set_ylabel("fraction of generations", fontsize=8.6)

    handles = [Line2D([], [], color=COLORS[m], lw=1.8, label=DISPLAY[m])
               for m in ORDER if m in per.model.unique()]
    handles += [Line2D([], [], color=INK2, lw=1.5, ls="-", label="leaked"),
                Line2D([], [], color=INK2, lw=1.5, ls=(0, (1.6, 1.6)), label="clean"),
                Line2D([], [], color="#1a1a1a", lw=1.6, ls=(0, (5, 2.5)), label="chance")]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Only PROK places a motif precisely, and only the Shine$-$Dalgarno; "
                 "every model is at chance for the $-$10 box",
                 fontsize=9.4, color=INK, x=0.02, ha="left", y=1.04)
    fig.tight_layout()
    save(fig, out, "fig1_placement")


# =====================================================================
def fig_signal(summ, out):
    models = [m for m in ORDER if m in summ.model.unique()]
    tasks = [t for t in ("generate_promoter", "generate_sd")
             if t in summ.task.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    # ---- (a) score above the model's own shuffle ----
    ax = axes[0]
    w, xs = 0.36, np.arange(len(models))
    for k, task in enumerate(tasks):
        vals = [summ[(summ.model == m) & (summ.task == task) &
                     (summ.background == "leaked")].score_gap_vs_shuffle.mean()
                for m in models]
        ax.bar(xs + (k - (len(tasks) - 1) / 2) * w, vals, w * 0.9,
               color=[COLORS[m] for m in models],
               alpha=1.0 if k == 0 else 0.55,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color=INK2, lw=0.9, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([DISPLAY[m] for m in models],
                                          rotation=20, ha="right", fontsize=7.6)
    ax.set_ylabel("PWM score above own shuffle (bits)", fontsize=8.4)
    ax.set_title("(a)  Signal beyond composition", fontsize=8.8, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); tidy(ax)
    ax.text(0.98, 0.06, "solid = task 1 ($-$10) · pale = task 2 (SD)",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom", ha="right")

    # ---- (b) leaked - clean ----
    ax = axes[1]
    for k, task in enumerate(tasks):
        vals = []
        for m in models:
            g = summ[(summ.model == m) & (summ.task == task)]
            lk = g[g.background == "leaked"].frac_pos_exact
            cl = g[g.background == "clean"].frac_pos_exact
            vals.append((lk.mean() - cl.mean()) * 100 if len(lk) and len(cl) else np.nan)
        ax.bar(xs + (k - (len(tasks) - 1) / 2) * w, vals, w * 0.9,
               color=[COLORS[m] for m in models],
               alpha=1.0 if k == 0 else 0.55,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color=INK2, lw=0.9, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([DISPLAY[m] for m in models],
                                          rotation=20, ha="right", fontsize=7.6)
    ax.set_ylabel("leaked $-$ clean, exact hits (pp)", fontsize=8.4)
    ax.set_title("(b)  Leakage", fontsize=8.8, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); tidy(ax)
    ax.text(0.98, 0.06, "0 = no memorisation advantage", transform=ax.transAxes,
            fontsize=7, color=MUTED, va="bottom", ha="right")

    # ---- (c) start codon, task 2 ----
    ax = axes[2]
    sd = summ[(summ.task == "generate_sd") & (summ.background == "leaked")]
    ms = [m for m in models if m in sd.model.unique()]
    if ms:
        vals = [sd[sd.model == m].codon_found_rate.mean() * 100 for m in ms]
        ax.bar(np.arange(len(ms)), vals, 0.6, color=[COLORS[m] for m in ms],
               edgecolor="white", linewidth=0.8, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.4, color=INK2)
        ax.set_xticks(np.arange(len(ms)))
        ax.set_xticklabels([DISPLAY[m] for m in ms], rotation=20, ha="right", fontsize=7.6)
    ax.set_ylabel("generations containing the start codon (%)", fontsize=8.4)
    ax.set_title("(c)  Start codon, task 2", fontsize=8.8, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); tidy(ax)

    fig.suptitle("Signal beyond composition exists only at the Shine$-$Dalgarno; "
                 "no model gains from having seen the real genome",
                 fontsize=9.4, color=INK, x=0.02, ha="left", y=1.03)
    fig.tight_layout()
    save(fig, out, "fig2_signal")


def main():
    ap = argparse.ArgumentParser()
    base = os.fspath(Path(__file__).resolve().parents[3] / "data" / "upstream_only")
    ap.add_argument("--scores", default=os.path.join(base, "scores"))
    ap.add_argument("--out", default=os.path.join(base, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    per = pd.read_csv(os.path.join(a.scores, "per_generation.csv.gz"))
    summ = pd.read_csv(os.path.join(a.scores, "summary.csv"))
    print(f"models present: {sorted(per.model.unique())}")

    fig_placement(per, a.out)
    fig_signal(summ, a.out)

    summ.to_csv(os.path.join(a.out, "figure_source_summary.csv"), index=False)
    json.dump({"models": sorted(per.model.unique()),
               "n_generations": int(len(per))},
              open(os.path.join(a.out, "figures_manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild the one-page CDS-completion appendix figure from compact profiles."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from aggregate_and_plot import BASES, COLORS, MODELS, draw_profile, profile_tvd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", type=Path,
        default=repo_root / "results" / "cds_completion_profiles.npz",
    )
    parser.add_argument(
        "--out", type=Path,
        default=repo_root / "figures" / "supp_cds_completion_probabilities.pdf",
    )
    args = parser.parse_args()

    cache = np.load(args.profiles)
    real = cache["real"]
    profiles = {
        model: {condition: cache[f"{model}_{condition}"]
                for condition in ("native", "shuffled")}
        for model, _ in MODELS
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pages:
        fig = plt.figure(figsize=(11, 12.8))
        grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.05])
        ax_real = fig.add_subplot(grid[0, :])
        draw_profile(ax_real, real, "Observed CDS (real sequence)")
        ax_real.tick_params(labelbottom=False)

        bottom = []
        for row, (model, display) in enumerate(MODELS, start=1):
            for column, condition in enumerate(("native", "shuffled")):
                axis = fig.add_subplot(grid[row, column], sharex=ax_real, sharey=ax_real)
                draw_profile(axis, profiles[model][condition],
                             f"{display} — {condition} context")
                if column:
                    axis.set_ylabel("")
                if row == len(MODELS):
                    bottom.append(axis)
                else:
                    axis.tick_params(labelbottom=False)
        for axis in bottom:
            axis.set_xlabel("Continuation position (bp)")

        ax_bar = fig.add_subplot(grid[4, :])
        width, positions = 0.3, np.arange(len(MODELS))
        for offset, condition, colour in (
            (-width / 2, "native", "#3b7dd8"),
            (width / 2, "shuffled", "#c0c0c0"),
        ):
            values = [profile_tvd(profiles[model][condition], real)
                      for model, _ in MODELS]
            bars = ax_bar.bar(positions + offset, values, width,
                              label=f"{condition} context", color=colour,
                              edgecolor="#33333322")
            for rectangle, value in zip(bars, values):
                ax_bar.text(rectangle.get_x() + rectangle.get_width() / 2,
                            value + 0.002, f"{value:.3f}", ha="center",
                            va="bottom", fontsize=8.5)
        null = profile_tvd(np.repeat(real.mean(axis=0)[None, :], len(real), axis=0), real)
        ax_bar.axhline(null, color="#d62728", ls="--", lw=1.2,
                      label=f"no positional structure ({null:.3f})")
        ax_bar.set_xticks(positions)
        ax_bar.set_xticklabels([display for _, display in MODELS])
        ax_bar.set_ylabel("Mean TVD from real CDS profile", fontsize=9.5)
        ax_bar.set_title("Distance from the real per-base profile (lower = closer to real)",
                         fontsize=10, pad=6)
        ax_bar.set_xlim(-0.62, len(MODELS) - 0.38)
        ax_bar.set_ylim(0, 0.178)
        ax_bar.grid(axis="y", alpha=0.18)
        ax_bar.legend(frameon=False, fontsize=9, loc="upper center", ncol=3,
                      columnspacing=2.2)

        handles = [plt.Line2D([], [], color=COLORS[base], lw=2.2, label=base)
                   for base in BASES]
        fig.suptitle("CDS midpoint completion: per-base probabilities",
                     y=0.988, fontsize=13)
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.955), fontsize=11, columnspacing=2.4,
                   handlelength=1.8)
        fig.subplots_adjust(top=0.912, bottom=0.05, left=0.065, right=0.985,
                            hspace=0.52, wspace=0.13)
        pages.savefig(fig)
        plt.close(fig)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

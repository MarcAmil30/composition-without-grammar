#!/usr/bin/env python3
"""Render improved Figure 2d and its A/C/G/T probability companion."""
from __future__ import annotations

# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
# -------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "evo1": "Evo 1 (8k)", "evo2": "Evo 2 (1B base, 8k)",
    "hyenadna": "HyenaDNA (450k)", "evo1_131k": "Evo 1 (131k)",
    "evo2_7b": "Evo 2 (7B, 1M)",
}
COLORS = {
    "evo1": "#2f6fb0", "evo2": "#3f8f5e", "hyenadna": "#a34ba3",
    "evo1_131k": "#78a9dc", "evo2_7b": "#176b3a",
}
BASE_COLORS = {"A": "#2f8f4e", "C": "#2f6fb0", "G": "#e0a33e", "T": "#c94f4f"}


def placement(exp: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((exp / "summaries").glob("*_ctx*.json")):
        data = json.loads(path.read_text())
        low, high = data["within_2_gene_bootstrap_95"]
        rows.append({
            "model": data["model"], "context_bp": data["context_bp"],
            "within_2_pct": data["within_2_pct"], "ci_low": low, "ci_high": high,
            "n_genes": data["n_genes"], "n_generations": data["n_generations"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(
        f"{ICBINB_ROOT}/data"))
    args = parser.parse_args()
    exp = args.data_root / "benchmark/fig2d_long_context"
    figure_dir = exp / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table = placement(exp)
    if table.empty:
        raise RuntimeError("no placement summaries")
    table.sort_values(["model", "context_bp"]).to_csv(
        exp / "summaries/figure2d_source.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.axhspan(0, 12.5, color="#f1f1f1")
    ax.axhline(12.5, color="#222222", linestyle="--", linewidth=1.4, label="chance")
    for model, subset in table.groupby("model"):
        subset = subset.sort_values("context_bp")
        ax.plot(subset.context_bp, subset.within_2_pct, marker="o", linewidth=2,
                color=COLORS.get(model), label=LABELS.get(model, model))
        ax.fill_between(subset.context_bp, subset.ci_low, subset.ci_high,
                        color=COLORS.get(model), alpha=0.16)
    ax.set_xscale("log")
    ax.set_xlabel("upstream window supplied (bp; log scale)")
    ax.set_ylabel("-10 box within +/-2 bp of the right place (%)")
    ax.set_title("More context does not fix promoter placement")
    ax.grid(axis="y", color="#dddddd")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"fig2d_long_context.{extension}", dpi=400)
    plt.close(fig)

    probability_files = sorted((exp / "summaries/probabilities").glob("*.csv"))
    if probability_files:
        probs = pd.concat([pd.read_csv(path) for path in probability_files])
        probs.to_csv(exp / "summaries/probability_profiles_source.csv", index=False)
        models = [model for model in LABELS if model in set(probs.model)]
        fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.0 * len(models)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models):
            subset = probs[probs.model == model]
            contexts = sorted(subset.context_bp.unique())
            selected = [contexts[0], contexts[-1]] if len(contexts) > 1 else contexts
            for context, linestyle in zip(selected, ("--", "-")):
                frame = subset[subset.context_bp == context].sort_values("position")
                for base in "ACGT":
                    ax.plot(frame.position, frame[f"p_{base}"], color=BASE_COLORS[base],
                            linestyle=linestyle, linewidth=1.4,
                            label=f"{base}, {context:g} bp")
            ax.axvspan(15, 23, color="#f6d878", alpha=0.25)
            ax.set_ylim(0, 0.65)
            ax.set_ylabel(LABELS[model])
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("position after the -35 box (bp)")
        axes[0].legend(ncol=4, frameon=False, fontsize=7)
        fig.suptitle("A/C/G/T probabilities at shortest and longest tested contexts")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"fig2d_probability_companion.{extension}", dpi=400)
        plt.close(fig)
    print(f"wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()

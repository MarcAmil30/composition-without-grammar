#!/usr/bin/env python3
"""Validate shards, aggregate by gene, and make the CDS completion PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

BASES = "ACGT"
COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
MODELS = [("evo1", "Evo 1 (8k)"), ("evo2", "Evo 2 (1B)"),
          ("hyenadna", "HyenaDNA (medium)")]


def periodicity(profile: np.ndarray) -> float:
    """RMS Fourier amplitude at exactly one cycle per three bases."""
    x = profile - profile.mean(axis=0, keepdims=True)
    phase = np.exp(-2j * np.pi * np.arange(len(x)) / 3)[:, None]
    amp = np.abs((x * phase).mean(axis=0))
    return float(np.sqrt(np.mean(amp ** 2)))


def profile_tvd(profile: np.ndarray, real: np.ndarray) -> float:
    """Mean total variation distance from the real CDS profile.

    At each position, half the summed |difference| between the two ACGT
    distributions: the share of probability mass sitting on the wrong base.
    0 = identical to real, 1 = disjoint.
    """
    return float(np.abs(profile - real).sum(axis=1).mean() / 2)


def load_model(root: Path, model: str, expected: int):
    d = root / "shards" / model
    manifests = sorted(d.glob("genes_*.manifest.json"))
    if len(manifests) != expected:
        raise RuntimeError(f"{model}: {len(manifests)}/{expected} complete shard manifests")
    meta, probs, tmeta, tprobs = [], [], [], []
    for manifest in manifests:
        stem = manifest.name.removesuffix(".manifest.json")
        m = pd.read_csv(d / f"{stem}.csv.gz", keep_default_na=False)
        tm = pd.read_csv(d / f"{stem}.teacher.csv.gz", keep_default_na=False)
        z = np.load(d / f"{stem}.npz")
        if len(m) != len(z["p_acgt"]) or len(tm) != len(z["teacher_p_acgt"]):
            raise RuntimeError(f"row mismatch in {model}/{stem}")
        meta.append(m); probs.append(z["p_acgt"])
        tmeta.append(tm); tprobs.append(z["teacher_p_acgt"])
    return (pd.concat(meta, ignore_index=True), np.concatenate(probs),
            pd.concat(tmeta, ignore_index=True), np.concatenate(tprobs))


def gene_profiles(meta: pd.DataFrame, p: np.ndarray) -> dict[str, np.ndarray]:
    q = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
    meta = meta.copy(); meta["i"] = np.arange(len(meta))
    out = {}
    for condition in ("native", "shuffled"):
        sub = meta[(meta.condition == condition) & (meta.decode == "sample")]
        genes = [q[g.i.to_numpy()].mean(axis=0) for _, g in sub.groupby("locus_tag", sort=False)]
        out[condition] = np.stack(genes).mean(axis=0)
    return out


def teacher_profiles(meta: pd.DataFrame, p: np.ndarray) -> dict[str, np.ndarray]:
    q = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
    return {c: q[meta.condition.to_numpy() == c].mean(axis=0)
            for c in ("native", "shuffled")}


def draw_profile(ax, profile, title):
    x = np.arange(1, 41)
    # A line on every codon boundary: the 3-bp grid is the reading frame.
    for boundary in np.arange(0.5, 41.5, 3):
        ax.axvline(boundary, color="#c4c4c4", lw=.7, zorder=0)
    for i, b in enumerate(BASES):
        ax.plot(x, profile[:, i], color=COLORS[b], lw=1.7, label=b, zorder=3)
    ax.set_xlim(.5, 40.5); ax.set_ylim(0, .62)
    ax.set_xticks(np.arange(1, 41, 3))
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(axis="y", alpha=.18)
    ax.set_ylabel("P(base | context)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--expected-shards", type=int, default=27)
    args = ap.parse_args()
    prompts = pd.read_csv(args.root / "prompts/cds_midpoint_prompts.csv.gz")
    true = np.stack([np.array([[float(c == b) for b in BASES] for c in s])
                     for s in prompts.true_next])
    real = true.mean(axis=0)

    profiles = {"real": {"native": real, "shuffled": real}}
    teacher = {}
    metrics = [{"model": "Real CDS", "condition": "empirical",
                "mode": "observed", "periodicity_rms": periodicity(real)}]
    for model, display in MODELS:
        meta, p, tmeta, tp = load_model(args.root, model, args.expected_shards)
        profiles[model] = gene_profiles(meta, p)
        teacher[model] = teacher_profiles(tmeta, tp)
        for condition in ("native", "shuffled"):
            for mode, prof in (("free_running", profiles[model][condition]),
                               ("teacher_forced", teacher[model][condition])):
                metrics.append({"model": display, "condition": condition, "mode": mode,
                                "periodicity_rms": periodicity(prof),
                                "tvd_from_real": profile_tvd(prof, real),
                                **{f"{b}_range": float(np.ptp(prof[:, i]))
                                   for i, b in enumerate(BASES)}})
    metrics_df = pd.DataFrame(metrics)
    args.root.joinpath("results").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.root / "results/cds_completion_metrics.csv", index=False)

    # One real-sequence reference spanning the top; each model then gets its
    # native generation beside its shuffled-context control.
    pdf = args.root / "results/supp_cds_completion_probabilities.pdf"
    with PdfPages(pdf) as pages:
        fig = plt.figure(figsize=(11, 12.8))
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.05])
        ax_real = fig.add_subplot(gs[0, :])
        draw_profile(ax_real, profiles["real"]["native"],
                     "Observed CDS (real sequence)")
        ax_real.tick_params(labelbottom=False)
        bottom = []
        for r, (model, display) in enumerate(MODELS, start=1):
            for c, condition in enumerate(("native", "shuffled")):
                ax = fig.add_subplot(gs[r, c], sharex=ax_real, sharey=ax_real)
                draw_profile(ax, profiles[model][condition],
                             f"{display} \u2014 {condition} context")
                if c:
                    ax.set_ylabel("")
                if r == len(MODELS):
                    bottom.append(ax)
                else:
                    ax.tick_params(labelbottom=False)
        for ax in bottom:
            ax.set_xlabel("Continuation position (bp)")

        # Summary panel: how far each model's profile sits from the real one.
        ax_bar = fig.add_subplot(gs[4, :])
        width, xs = .3, np.arange(len(MODELS))
        for offset, condition, colour in ((-width / 2, "native", "#3b7dd8"),
                                          (width / 2, "shuffled", "#c0c0c0")):
            vals = [profile_tvd(profiles[m][condition], real) for m, _ in MODELS]
            bars = ax_bar.bar(xs + offset, vals, width, label=f"{condition} context",
                              color=colour, edgecolor="#33333322")
            for rect, v in zip(bars, vals):
                ax_bar.text(rect.get_x() + rect.get_width() / 2, v + .002,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        # A model with the right base composition but no positional structure.
        null = profile_tvd(np.repeat(real.mean(axis=0)[None, :], len(real), axis=0), real)
        ax_bar.axhline(null, color="#d62728", ls="--", lw=1.2,
                       label=f"no positional structure ({null:.3f})")
        ax_bar.set_xticks(xs); ax_bar.set_xticklabels([d for _, d in MODELS])
        ax_bar.set_ylabel("Mean TVD from real CDS profile", fontsize=9.5)
        ax_bar.set_title("Distance from the real per-base profile (lower = closer to real)",
                         fontsize=10, pad=6)
        ax_bar.set_xlim(-.62, len(MODELS) - .38)
        ax_bar.set_ylim(0, .178)          # headroom so the legend clears the bars
        ax_bar.grid(axis="y", alpha=.18)
        ax_bar.legend(frameon=False, fontsize=9, loc="upper center", ncol=3,
                      columnspacing=2.2)
        handles = [plt.Line2D([], [], color=COLORS[b], lw=2.2, label=b) for b in BASES]
        fig.suptitle("CDS midpoint completion: per-base probabilities",
                     y=.988, fontsize=13)
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(.5, .955), fontsize=11, columnspacing=2.4,
                   handlelength=1.8)
        fig.subplots_adjust(top=.912, bottom=.05, left=.065, right=.985,
                            hspace=.52, wspace=.13)
        pages.savefig(fig); plt.close(fig)

    summary = {"n_genes": len(prompts), "models": [m for m, _ in MODELS],
               "output_pdf": str(pdf), "metrics_csv": str(args.root / "results/cds_completion_metrics.csv")}
    (args.root / "results/aggregate_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

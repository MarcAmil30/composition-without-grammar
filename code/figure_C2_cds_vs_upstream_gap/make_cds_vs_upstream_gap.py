"""How far is a generated per-base profile from the real one: CDS vs upstream.

Both tracks are already measured; this only compares them. For each track we
take the mean probability of A/C/G/T at every position and score a model by the
mean total variation distance (TVD) from the real profile at the same positions.

  CDS track       672 E. coli genes cut at the CDS midpoint, 40 bp continued.
                  Profiles come from results/cds_completion_profiles.npz, the
                  aggregate exported by
                  code/cds_completion_probabilities/export_profiles.py from the
                  per-shard probability tensors on the cluster. The recovered
                  TVDs reproduce the ones printed in the published panel
                  (0.082 / 0.019 / 0.092).
  upstream track  487 E. coli sigma70 promoters, the 41 positions following
                  the distal box (the Figure 2b window), leaked background.
                  Read from the panel B source CSVs.

Two reference lines make the tracks comparable:
  flat      the TVD of a profile that holds the SAME base probabilities at
            every position, fixed at the real average composition of that track
            ("position-blind"). It is what a model scores for getting
            composition right and position entirely wrong, and equally it is how
            much positional structure the real sequence carries at all
            (CDS 0.094, upstream 0.067). Dividing by it puts the two tracks on
            one scale: 1.0 is no better than composition alone.
  floor     the TVD an n-gene estimate of the real profile shows against the
            profile it was drawn from -- the noise the real side alone carries
            at that many genes. Nothing below this is measurable.

Outputs: figures/fig_cds_vs_upstream_gap.{pdf,png}   (one panel)
         results/cds_vs_upstream_gap.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CDS_NPZ = ROOT / "results/cds_completion_profiles.npz"
UP_DIR = ROOT / "results/upstream_only/fig2x2"
OUT_PDF = ROOT / "figures/fig_cds_vs_upstream_gap.pdf"
OUT_CSV = ROOT / "results/cds_vs_upstream_gap.csv"

INK, MUTED, GRID = "#1a1a1a", "#8a8a8a", "#dcdcdc"
COLORS = {"evo1": "#2f6fb0", "evo2": "#3f8f5e",
          "hyenadna": "#a34ba3", "promogen2": "#9a7b28"}
DISPLAY = {"evo1": "Evo 1 (8k)", "evo2": "Evo 2 (1B)",
           "hyenadna": "HyenaDNA", "promogen2": "PromoGen2"}
ORDER = ["evo1", "evo2", "hyenadna"]   # PromoGen2 has no CDS counterpart
N_GENES = {"CDS": 672, "upstream": 487}
RNG = np.random.default_rng(20260825)


def tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Total variation distance per position between two ACGT profiles."""
    return 0.5 * np.abs(a - b).sum(axis=-1)


# --------------------------------------------------------------- CDS profiles
def cds_profiles() -> dict[str, np.ndarray]:
    """Per-base profiles for the free-running, native-context CDS completions."""
    z = np.load(CDS_NPZ)
    out = {"real": np.asarray(z["real"], dtype=float)}
    for m in ORDER:
        out[m] = np.asarray(z[f"{m}_native"], dtype=float)
    out = {k: v / v.sum(axis=1, keepdims=True) for k, v in out.items()}
    for name, p in out.items():
        assert abs(p.sum(axis=1).mean() - 1) < 2e-3, f"{name} does not sum to 1"
    return out


# ---------------------------------------------------------- upstream profiles
def upstream_profiles() -> dict[str, np.ndarray]:
    cols = ["p_A", "p_C", "p_G", "p_T"]
    real = pd.read_csv(UP_DIR / "panelB_real.csv")
    real = real[real.background == "leaked"].sort_values("position")
    out = {"real": real[cols].to_numpy()[:41]}
    gen = pd.read_csv(UP_DIR / "panelB_probs.csv")
    for model, g in gen[gen.background == "leaked"].groupby("model"):
        out[model] = g.sort_values("position")[cols].to_numpy()[:41]
    return {k: v / v.sum(axis=1, keepdims=True) for k, v in out.items()}


def floor(real: np.ndarray, n: int, reps: int = 2000) -> float:
    """TVD an n-gene estimate of a profile shows against the profile itself."""
    draws = np.stack([RNG.multinomial(n, p, size=reps) / n for p in real], axis=1)
    return float(tvd(draws, real[None]).mean())


cds, up = cds_profiles(), upstream_profiles()
TRACKS = {"CDS": cds, "upstream": up}

rows = []
for track, prof in TRACKS.items():
    real = prof["real"]
    flat = tvd(np.tile(real.mean(axis=0), (len(real), 1)), real).mean()
    fl = floor(real, N_GENES[track])
    for m in ORDER:
        if m not in prof:
            continue
        rows.append({"track": track, "model": m, "mean_tvd": tvd(prof[m], real).mean(),
                     "flat_reference": flat, "sampling_floor": fl,
                     "tvd_over_flat": tvd(prof[m], real).mean() / flat,
                     "n_positions": len(real), "n_genes": N_GENES[track]})
res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)
print(res.round(4).to_string(index=False))

# ------------------------------------------------------------------- the plot
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
    "axes.edgecolor": INK, "axes.linewidth": 1.1, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})
fig, ax = plt.subplots(figsize=(6.3, 3.6))
w, gap = 0.30, 0.05
for i, track in enumerate(TRACKS):
    sub = res[res.track == track]
    for j, m in enumerate(ORDER):
        v = float(sub[sub.model == m].tvd_over_flat.iloc[0])
        x = j + (i - .5) * (w + gap)
        ax.bar(x, v, w, color=COLORS[m], alpha=1.0 if track == "CDS" else .45,
               edgecolor=COLORS[m], lw=1.1)
        ax.text(x, v + .02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, color=INK, lw=1.3, ls=(0, (5, 3)), zorder=4)
ax.text(2.80, .96, "no better than\na position-blind\nprofile",
        ha="left", va="top", fontsize=8)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([DISPLAY[m] for m in ORDER], fontsize=9)
ax.set_ylabel("TVD from real / position-blind reference")
ax.set_xlim(-.65, 3.55)
ax.set_ylim(0, 1.35)
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED, label="CDS completion"),
    plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED, alpha=.45,
                  label="upstream completion")],
    loc="upper right", fontsize=8.4)
ax.grid(axis="y", color=GRID, lw=.8, alpha=.7)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_PDF.with_suffix("." + ext))
print("wrote", OUT_PDF)

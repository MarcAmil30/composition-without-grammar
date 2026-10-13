"""
De-novo SD discovery: no AGGAGG prior.

Take the last 20 bp of all 681 upstreams, aligned by their distance to the start
codon (-20 ... -1), and let logomaker show where information actually sits.
Whatever peaks IS the Shine-Dalgarno. The resulting PWM is then the scoring
matrix for Task 2 -- derived from the data, not assumed.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[2]))
# -------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

BASE = f"{ICBINB_ROOT}/data/benchmark/upstream_only"
OUT = os.path.join(BASE, "sd_denovo")
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

W = 20                      # window: last 20 bp before the start codon
POS = list(range(-W, 0))    # -20 ... -1
BASES = list("ACGT")

df = pd.read_csv(os.path.join(BASE, "prompts_681_leaked.csv"))
df["seq"] = df.prompt_promoter + df.target_promoter
assert (df.seq.str.len() == df.L).all()

# the last 20 bp, i.e. exactly the Task 2 target window
wins = [s[-W:] for s in df.seq]
assert all(len(w) == W for w in wins)

counts = (logomaker.alignment_to_matrix(wins, to_type="counts", characters_to_ignore="N")
          .reindex(columns=BASES).fillna(0.0))
counts.index = POS
counts.index.name = "pos_from_ATG"
freq = counts.div(counts.sum(axis=1), axis=0)
info = logomaker.transform_matrix(counts, from_type="counts", to_type="information")
info.index = POS; info.index.name = "pos_from_ATG"

# background = the whole 200 bp upstream, so the logo is enrichment over local composition
allbases = "".join(df.seq)
bg = pd.Series({b: allbases.count(b) for b in BASES}, dtype=float)
bg /= bg.sum()

# information relative to that background (not to uniform 25%)
eps = 1e-9
ic_bg = (freq * np.log2((freq + eps) / bg)).sum(axis=1)          # KL per position, bits
info_bg = freq.mul(ic_bg, axis=0)

for m, nm in ((counts, "counts"), (freq, "freq"),
              (info, "information_uniform"), (info_bg, "information_vs_upstream_bg")):
    m.to_csv(os.path.join(OUT, f"sd_denovo_pfm_{nm}.csv"))
pd.DataFrame({"base": BASES, "upstream_bg_freq": bg[BASES].values}).to_csv(
    os.path.join(OUT, "background_composition.csv"), index=False)

# ---- per-position summary ----
summ = pd.DataFrame({
    "pos_from_ATG": POS,
    "consensus": freq.idxmax(axis=1).values,
    "consensus_freq": freq.max(axis=1).values,
    "purine_AG_freq": (freq["A"] + freq["G"]).values,
    "bits_vs_uniform": info.sum(axis=1).values,
    "bits_vs_upstream_bg": ic_bg.values,
})
summ.to_csv(os.path.join(OUT, "sd_denovo_per_position.csv"), index=False)

# ---- where is the signal? contiguous 6 bp window with most information ----
b = info.sum(axis=1).values
best = int(np.argmax([b[i:i + 6].sum() for i in range(W - 5)]))
core = slice(best, best + 6)
core_pos = POS[best], POS[best + 5]
core_consensus = "".join(freq.idxmax(axis=1).values[core])

# ---- also: exact-match rate of that discovered word, per gene ----
core_seqs = [w[core] for w in wins]
mm = pd.Series([sum(a != b_ for a, b_ in zip(s, core_consensus)) for s in core_seqs])

plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
for ax, (m, lab) in zip(axes, [(info, "bits (vs uniform)"),
                               (info_bg, "bits (vs upstream background)")]):
    logomaker.Logo(m, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_ylabel(lab, fontsize=8)
    for x in (POS[best] - 0.5, POS[best + 5] + 0.5):
        ax.axvline(x, color="0.45", lw=0.9, ls="--", zorder=5)
    ax.set_xticks(POS)
axes[-1].set_xlabel("position relative to the start codon (bp)")
axes[0].set_title(f"Last 20 bp of all {len(df)} E. coli upstreams — no motif assumed\n"
                  f"strongest 6 bp window: {core_pos[0]} to {core_pos[1]}  ->  {core_consensus}",
                  fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "plots", f"sd_denovo_logo.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
print(f"n sequences: {len(df)}   window: last {W} bp")
print(f"upstream background: " + "  ".join(f"{b_}={bg[b_]:.3f}" for b_ in BASES))
print()
print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print()
print(f"STRONGEST 6 bp WINDOW: {core_pos[0]} .. {core_pos[1]}   consensus = {core_consensus}")
print(f"  total bits (vs uniform) = {b[core].sum():.2f}")
print()
print("mismatches to the discovered word, per gene:")
print(mm.value_counts().sort_index().to_string())
print(f"  <=2 mismatches: {(mm<=2).sum()} / {len(mm)}  ({(mm<=2).mean()*100:.1f}%)")
print()
print("wrote ->", OUT)

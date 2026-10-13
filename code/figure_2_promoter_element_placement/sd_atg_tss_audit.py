"""
Audit of the 681-gene upstream_only benchmark, leaked + clean:
  1. Shine-Dalgarno presence  (+ logomaker motif logo)
  2. Start-codon presence and position (expect 201-203)
  3. TSS nucleotide composition (+ pie chart)

Everything is written out as CSV so plots can be re-made/re-styled later
without recomputing.
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
OUT  = os.path.join(BASE, "audit")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

# E. coli SD consensus (anti-SD = 3'-UCCUCC-5' at the 16S 3' end)
CONSENSUS = "AGGAGG"
STARTS = {"ATG", "GTG", "TTG"}
BASES = list("ACGT")


def load(tag):
    df = pd.read_csv(os.path.join(BASE, f"prompts_681_{tag}.csv"))
    # the promoter task's prompt+target tile the upstream exactly (verified 681/681)
    df["seq"] = df.prompt_promoter + df.target_promoter
    assert (df.seq.str.len() == df.L).all(), "sequence reconstruction failed"
    # target_sd = seq[cut_sd:] + start_codon; 2 rows lack the appended codon
    df["appended_codon"] = np.where(df.target_sd_len == df.L - df.cut_sd + 3,
                                    df.target_sd.str[-3:], None)
    return df


def sd_hexamer(row):
    """The 6 bp SD core, anchored at the 3' end of the annotated span."""
    s, e = int(row.sd_start), int(row.sd_end)
    hexa = row.seq[max(0, e - 6):e]
    return hexa if len(hexa) == 6 else None


def match_score(hexa):
    return sum(a == b for a, b in zip(hexa, CONSENSUS))


rows_summary = []
per_gene = {}

for tag in ("leaked", "clean"):
    df = load(tag)

    # ---------- 1. Shine-Dalgarno ----------
    df["sd_hex"] = df.apply(sd_hexamer, axis=1)
    df["sd_match6"] = df.sd_hex.map(match_score)
    df["sd_offset_in_gen"] = df.sd_start - df.cut_sd     # position in the 40 bp window

    # ---------- 2. start codon ----------
    # the real start codon sits immediately 3' of the upstream, i.e. positions L+1..L+3
    df["start_codon_ok"] = df.start_codon.isin(STARTS)
    df["start_offset_in_gen"] = df.L - df.cut_sd          # expected 20
    df["codon_matches_appended"] = df.start_codon == df.appended_codon

    # ---------- 3. TSS ----------
    df["tss_base"] = [row.seq[int(row.tss_idx)] if 0 <= int(row.tss_idx) < len(row.seq)
                      else None for row in df.itertuples()]

    per_gene[tag] = df

    rows_summary.append(dict(
        background=tag,
        n_genes=len(df),
        sd_present_flag=int(df.sd_present.sum()),
        sd_absent_flag=int((df.sd_present == 0).sum()),
        sd_hexamer_recoverable=int(df.sd_hex.notna().sum()),
        sd_match_ge4=int((df.sd_match6 >= 4).sum()),
        sd_match_ge5=int((df.sd_match6 >= 5).sum()),
        sd_perfect_AGGAGG=int((df.sd_match6 == 6).sum()),
        start_codon_valid=int(df.start_codon_ok.sum()),
        start_codon_ATG=int((df.start_codon == "ATG").sum()),
        start_codon_GTG=int((df.start_codon == "GTG").sum()),
        start_codon_TTG=int((df.start_codon == "TTG").sum()),
        start_offset_mode=int(df.start_offset_in_gen.mode()[0]),
        start_offset_is_20=int((df.start_offset_in_gen == 20).sum()),
        tss_purine=int(df.tss_base.isin(["A", "G"]).sum()),
    ))

    # ---- per-gene table (re-plottable) ----
    keep = ["locus_tag", "gene", "sigma", "L", "cut_sd",
            "sd_present", "sd_start", "sd_end", "sd_score", "sd_spacing",
            "sd_hex", "sd_match6", "sd_offset_in_gen",
            "start_codon", "start_codon_ok", "start_offset_in_gen",
            "tss_idx", "tss_base"]
    df[keep].to_csv(os.path.join(OUT, f"audit_per_gene_{tag}.csv"), index=False)

    # ---- SD position-frequency matrix, 12 bp window ending at the SD 3' end ----
    W = 12
    mats = {"all": [], "sd_present": [], "sd_absent": []}
    for row in df.itertuples():
        e = int(row.sd_end)
        win = row.seq[max(0, e - W):e]
        if len(win) != W:
            continue
        mats["all"].append(win)
        mats["sd_present" if row.sd_present == 1 else "sd_absent"].append(win)

    for subset, seqs in mats.items():
        if len(seqs) < 5:
            continue
        counts = logomaker.alignment_to_matrix(seqs, to_type="counts", characters_to_ignore="N")
        counts = counts.reindex(columns=BASES).fillna(0.0)
        freq = counts.div(counts.sum(axis=1), axis=0)
        info = logomaker.transform_matrix(counts, from_type="counts", to_type="information")
        # position label: -12 .. -1 relative to the SD 3' end
        idx = list(range(-W, 0))
        for m, nm in ((counts, "counts"), (freq, "freq"), (info, "information")):
            m.index = idx
            m.index.name = "pos_rel_sd_end"
            m.to_csv(os.path.join(OUT, f"sd_pfm_{tag}_{subset}_{nm}.csv"))

    # ---- TSS base counts ----
    tss = df.tss_base.value_counts().reindex(BASES).fillna(0).astype(int)
    tss_tab = pd.DataFrame({"base": BASES, "count": tss.values})
    tss_tab["frac"] = tss_tab["count"] / tss_tab["count"].sum()
    tss_tab.to_csv(os.path.join(OUT, f"tss_base_counts_{tag}.csv"), index=False)

    # ---- TSS by sigma factor ----
    bysig = (df.groupby("sigma").tss_base.value_counts().unstack()
               .reindex(columns=BASES).fillna(0).astype(int))
    bysig.to_csv(os.path.join(OUT, f"tss_base_by_sigma_{tag}.csv"))

    # ---- start codon counts ----
    (df.start_codon.value_counts().rename_axis("codon").reset_index(name="count")
       .to_csv(os.path.join(OUT, f"start_codon_counts_{tag}.csv"), index=False))
    (df.start_offset_in_gen.value_counts().sort_index()
       .rename_axis("offset_in_generation").reset_index(name="count")
       .to_csv(os.path.join(OUT, f"start_codon_offset_{tag}.csv"), index=False))

pd.DataFrame(rows_summary).to_csv(os.path.join(OUT, "audit_summary.csv"), index=False)

# =====================================================================
# Plots (regenerable from the CSVs above)
# =====================================================================
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "svg.fonttype": "none", "pdf.fonttype": 42})

# --- SD logos: leaked, split by sd_present flag ---
subsets = [("all", "all 681"), ("sd_present", "SD present (548)"), ("sd_absent", "SD absent (133)")]
fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.4))
for ax, (subset, title) in zip(axes, subsets):
    p = os.path.join(OUT, f"sd_pfm_leaked_{subset}_information.csv")
    if not os.path.exists(p):
        ax.axis("off"); continue
    info = pd.read_csv(p, index_col=0)
    logomaker.Logo(info, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_ylabel("bits")
    ax.set_ylim(0, max(0.6, info.sum(axis=1).max() * 1.15))
    ax.set_title(f"{title}   —   12 bp ending at the annotated SD 3' end", fontsize=9)
    ax.set_xticks(list(info.index))
    ax.set_xticklabels([str(i) for i in info.index], fontsize=7)
axes[-1].set_xlabel("position relative to SD 3' end (bp)")
fig.suptitle("Shine–Dalgarno information content, E. coli benchmark (leaked)", fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "plots", f"sd_logo.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- TSS pie ---
COL = {"A": "#2f8f4e", "C": "#2f6fb0", "G": "#e0a33e", "T": "#c94f4f"}
tss_tab = pd.read_csv(os.path.join(OUT, "tss_base_counts_leaked.csv"))
fig, (axp, axb) = plt.subplots(1, 2, figsize=(7.2, 3.4))
axp.pie(tss_tab["count"], labels=tss_tab["base"],
        colors=[COL[b] for b in tss_tab["base"]],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*tss_tab['count'].sum()/100))})",
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.2), textprops=dict(fontsize=8))
pur = tss_tab.set_index("base").loc[["A", "G"], "count"].sum() / tss_tab["count"].sum()
axp.set_title(f"TSS (+1) base, n={int(tss_tab['count'].sum())}\npurine (A+G) = {pur*100:.1f}%", fontsize=9)

bysig = pd.read_csv(os.path.join(OUT, "tss_base_by_sigma_leaked.csv"), index_col=0)
bysig = bysig.loc[bysig.sum(axis=1).sort_values(ascending=False).index]
frac = bysig.div(bysig.sum(axis=1), axis=0)
left = np.zeros(len(frac))
for b in BASES:
    axb.barh(range(len(frac)), frac[b], left=left, color=COL[b], label=b,
             edgecolor="white", linewidth=0.8)
    left += frac[b].values
axb.set_yticks(range(len(frac)))
axb.set_yticklabels([f"{s} (n={int(n)})" for s, n in zip(frac.index, bysig.sum(axis=1))], fontsize=8)
axb.invert_yaxis(); axb.set_xlim(0, 1); axb.set_xlabel("fraction")
axb.set_title("TSS base by sigma factor", fontsize=9)
axb.legend(ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32))
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "plots", f"tss_base.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
print(pd.DataFrame(rows_summary).T.to_string())
print("\n--- SD match-to-AGGAGG distribution (leaked) ---")
print(per_gene["leaked"].sd_match6.value_counts().sort_index().to_string())
print("\n--- SD hexamer identical leaked vs clean:",
      (per_gene["leaked"].sd_hex == per_gene["clean"].sd_hex).mean())
print("\n--- start codon offset in generation (leaked) ---")
print(per_gene["leaked"].start_offset_in_gen.value_counts().sort_index().to_string())
print("\n--- TSS base (leaked) ---")
print(pd.read_csv(os.path.join(OUT, "tss_base_counts_leaked.csv")).to_string(index=False))
print("\n--- TSS base (clean) ---")
print(pd.read_csv(os.path.join(OUT, "tss_base_counts_clean.csv")).to_string(index=False))
print("\nwrote ->", OUT)

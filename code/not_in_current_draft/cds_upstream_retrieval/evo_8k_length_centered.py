"""
CENTERED re-run of evo_8k_sequence_length_retrieval (retrieval-only, no GPU).

The raw mean-pooled evo embeddings are anisotropic: cos(CDS_i, its own upstream) ~ 0.999,
but so is cos to every distractor, so cosine retrieval was at chance. Standard fix = remove
each set's dominant shared direction by subtracting its mean vector ("all-but-the-mean")
BEFORE cosine.

  * QUERY  centering: Q_c = Q - mean_over_genes(Q)          (per species, CDS set)
  * CAND   centering: U_c = U - mean_over_genes(U)          (per species, per length, upstream set)
  * then L2-normalise each, cosine = Q_c . U_c

Reuses the cached raw embeddings from results/evo_8k_length/ (cds_emb_*.npy, up_emb_L{L}_*.npy).
Retrieval is byte-identical to the raw run: same 1,404 reference singletons in the same order,
seed=0, n_repeats=20, RNG reset per length -> same seeded decoys at every length, and identical
to the raw run so raw-vs-centered is a controlled comparison of ONLY the centering step.
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = f"{CL_ROOT}"
RAW = f"{PROJ}/results/evo_8k_length"
OUT = f"{RAW}/centered"
os.makedirs(OUT, exist_ok=True)
REF = ["Escherichia coli", "Bacillus subtilis"]
LENGTHS = [300, 200, 100, 50]
SEED, N_REP, N_CAND = 0, 20, 100


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sp_tag(sp): return sp.replace(" ", "_")


def center_norm(X):
    """subtract column mean (remove dominant shared direction) then L2-normalise rows."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(n, 1e-8, None)


# reproduce the exact per-species gene rows / order used at embedding time
work = pd.read_parquet(f"{PROJ}/results/step0/analysis_set.parquet")
rows_by_sp = {}
for sp in REF:
    rows_by_sp[sp] = np.where(work["species"].to_numpy() == sp)[0]  # ascending == embed order

CDS = {sp: np.load(f"{RAW}/cds_emb_{sp_tag(sp)}.npy").astype(np.float32) for sp in REF}
for sp in REF:
    assert CDS[sp].shape[0] == len(rows_by_sp[sp]), f"row/emb mismatch {sp}"
    log(f"{sp}: n={CDS[sp].shape[0]}")

k = N_CAND - 1
long_records, summary = [], {}

for L in LENGTHS:
    log(f"--- L={L} (centered) ---")
    UP = {sp: np.load(f"{RAW}/up_emb_L{L}_{sp_tag(sp)}.npy").astype(np.float32) for sp in REF}
    rng = np.random.default_rng(SEED)                 # reset per length -> identical decoys
    for sp in REF:
        rows = rows_by_sp[sp]
        n = len(rows)
        Q = center_norm(CDS[sp])                       # centered CDS query
        U = center_norm(UP[sp])                        # centered upstream candidates
        Sp = Q @ U.T                                   # (n,n) cosine after centering
        true_s = np.diag(Sp).copy()
        ranks = np.zeros((n, N_REP), dtype=np.int32)
        for rep in range(N_REP):
            d = rng.integers(0, n - 1, size=(n, k))
            d += (d >= np.arange(n)[:, None])
            cand = np.take_along_axis(Sp, d, axis=1)
            ranks[:, rep] = 1 + (cand > true_s[:, None]).sum(axis=1)
        top1 = float((ranks == 1).mean()); top5 = float((ranks <= 5).mean())
        med = float(np.median(ranks)); mean = float(ranks.mean())
        summary.setdefault(sp, {})[str(L)] = dict(length=L, n=int(n), n_repeats=N_REP,
            top1=top1, top5=top5, median_rank=med, mean_rank=mean,
            true_cos_mean=float(true_s.mean()))
        log(f"  {sp:<20} top1={top1:.4f} top5={top5:.4f} median={med:.0f} "
            f"mean={mean:.2f} trueCos(centered)={true_s.mean():+.4f}")
        for gi, r in enumerate(rows):
            for rep in range(N_REP):
                long_records.append((sp, L, int(r), rep, int(ranks[gi, rep])))

df = pd.DataFrame(long_records, columns=["species", "length", "gene_row", "rep", "rank"])
df.to_parquet(f"{OUT}/evo8k_length_ranks_centered.parquet", index=False)
with open(f"{OUT}/evo8k_length_summary_centered.json", "w") as f:
    json.dump(summary, f, indent=2)
tbl = [dict(species=sp, length_bp=L, n_genes=summary[sp][str(L)]["n"],
            top1=round(summary[sp][str(L)]["top1"], 4), top5=round(summary[sp][str(L)]["top5"], 4),
            median_rank=summary[sp][str(L)]["median_rank"],
            mean_rank=round(summary[sp][str(L)]["mean_rank"], 2),
            true_cos_mean=round(summary[sp][str(L)]["true_cos_mean"], 4))
       for sp in REF for L in LENGTHS]
pd.DataFrame(tbl).to_csv(f"{OUT}/summary_table_centered.csv", index=False)

for sp in REF:
    sub = df[df["species"] == sp]
    data = [sub[sub["length"] == L]["rank"].to_numpy() for L in LENGTHS]
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor("#55A868"); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(LENGTHS) + 1)); ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.set_xlabel("Upstream length fed to evo-1-8k-base (bp)")
    ax.set_ylabel("Rank of true upstream (1 = best, of 100)   [raw per-trial]")
    ax.set_title(f"evo-1-8k-base length sweep, CENTERED (CDS -> upstream) — {sp}\n"
                 f"n={summary[sp]['300']['n']} genes x {N_REP} repeats, same seeded pairs")
    ax.grid(axis="y", alpha=0.3)
    for xi, L in enumerate(LENGTHS, start=1):
        ax.text(xi, ax.get_ylim()[1] * 0.98, f"top1\n{summary[sp][str(L)]['top1']:.1%}",
                ha="center", va="top", fontsize=8, color="darkgreen")
    fig.tight_layout(); fig.savefig(f"{OUT}/violin_{sp_tag(sp)}_centered.png", dpi=150)
    plt.close(fig); log(f"wrote violin_{sp_tag(sp)}_centered.png")

log("\n============ CENTERED SUMMARY ============")
log(f"{'species':<20} {'L':>4} {'top1':>8} {'median':>7} {'top5':>8} {'trueCos':>8}")
for sp in REF:
    for L in LENGTHS:
        s = summary[sp][str(L)]
        log(f"{sp:<20} {L:>4} {s['top1']:>8.2%} {s['median_rank']:>7.0f} "
            f"{s['top5']:>8.2%} {s['true_cos_mean']:>+8.4f}")
log("EVO8K_CENTERED_OK")

"""
EVO-1-8k-base SCORE-based CDS<->upstream retrieval, length sweep.
(experiment: evo_8k_sequence_length_retrieval  --  scoring variant)

Instead of embeddings+cosine, we use evo's LOG-LIKELIHOOD score of the CONCATENATED
sequence  UPSTREAM + CDS  (5'->3', upstream first), via the repo's scoring path
(prepare_batch + logits_to_logprobs, evo/evo/scoring.py).

Retrieval (per species, per upstream length L in {300,200,100,50}):
  * query   = gene i's full-length CDS.
  * candidate j = concat( upstream_j[-L:] , CDS_i )   [true pair = j==i, 99 wrong j].
  * rank the true pair among 100 by score, n_repeats=20 fresh distractor draws,
    seed=0, RNG reset per length (same seeded decoys at every length).

Two scores per pair, BOTH from one forward pass (mean-reduced log-prob):
  * score_full = mean logprob over the WHOLE concat (upstream + CDS)   <- what was asked
  * score_cds  = mean logprob over the CDS tokens only, i.e. conditional P(CDS | upstream)
                 -- isolates the upstream->CDS dependency (more principled), free to compute.

DIRECTION: evo score is a log-likelihood (higher = more probable). We rank higher=better
(true genomic pairing expected to be MORE likely) and also record top-1 under lower=better,
plus the true-vs-distractor score gap, so the direction is settled by the data. Raw score
matrices are saved (.npy) so any re-ranking is instant.

COMPUTE: scoring is pairwise -> full 727^2 is infeasible. We SUBSAMPLE n_sub genes/species
(seeded) and score the full n_sub x n_sub matrix per length. Matrices are checkpointed to
.npy (resume-safe). n_sub=150 -> ~180k forwards over 2 species x 4 lengths.

Outputs (results/evo_8k_score/<out_subdir>):
  S_full_L{L}_{sp}.npy, S_cds_L{L}_{sp}.npy   raw score matrices (n_sub x n_sub)
  evo8k_score_ranks.parquet      species,length,metric,gene,rep,rank  (higher=better ranking)
  evo8k_score_summary.json       per species x length x metric: top1(both dirs), median, gap
  boxplot_{metric}_{sp}.png      x=length, y=rank (raw per-trial)
  summary_table.csv
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = f"{CL_ROOT}"
REF = ["Escherichia coli", "Bacillus subtilis"]
LENGTHS = [300, 200, 100, 50]

ap = argparse.ArgumentParser()
ap.add_argument("--n_sub", type=int, default=150, help="genes per species (subsample)")
ap.add_argument("--n_candidates", type=int, default=100)
ap.add_argument("--n_repeats", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_batch_tokens", type=int, default=24000)
ap.add_argument("--out_subdir", type=str, default="")
args = ap.parse_args()

OUT = f"{PROJ}/results/evo_8k_score" + (f"/{args.out_subdir}" if args.out_subdir else "")
os.makedirs(OUT, exist_ok=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def tag(sp): return sp.replace(" ", "_")


log(f"experiment=evo_8k_sequence_length_retrieval (SCORE)  OUT={OUT}  args={vars(args)}")
device = "cuda:0"

# --------------------------------------------------- data + CDS join + subsample
log("STEP 1 — load reference singletons, join cds_seq, subsample")
work = pd.read_parquet(f"{PROJ}/results/step0/analysis_set.parquet")
ref = work[work["species"].isin(REF)].copy()
ref["_row"] = np.where(work["species"].isin(REF).to_numpy())[0]
flanks = pd.read_csv(f"{PROJ}/data/ref2/ref2_flanks.csv",
                     usecols=["accession", "locus_tag", "cds_seq", "upstream_seq"], dtype=str)
ref = ref.merge(flanks.rename(columns={"cds_seq": "cds_seq_f", "upstream_seq": "up_check"}),
                on=["accession", "locus_tag"], how="left")
assert ref["cds_seq_f"].isna().sum() == 0
assert (ref["upstream_seq"] == ref["up_check"]).all()

rng_sub = np.random.default_rng(args.seed)
genes = {}
for sp in REF:
    sub = ref[ref["species"] == sp].sort_values("_row").reset_index(drop=True)
    take = np.sort(rng_sub.choice(len(sub), size=min(args.n_sub, len(sub)), replace=False))
    sub = sub.iloc[take].reset_index(drop=True)
    genes[sp] = dict(rows=sub["_row"].to_numpy(),
                     cds=sub["cds_seq_f"].tolist(),
                     up=sub["upstream_seq"].tolist())
    cl = [len(c) for c in genes[sp]["cds"]]
    log(f"  {sp}: n_sub={len(sub)}  cds_len med={int(np.median(cl))} max={max(cl)}")

# --------------------------------------------------- evo model + batched scorer
log("STEP 2 — load evo-1-8k-base (scoring)")
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs

evo_model = Evo("evo-1-8k-base")
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device).eval()
log("  evo loaded")


def score_concats(concats, up_lens):
    """Return (score_full, score_cds) arrays. Char-level tokenizer => token len == char len.
    Batched, length-bucketed. Causal model + right-padding => batching is exact."""
    N = len(concats)
    full = np.zeros(N, dtype=np.float32)
    cdsc = np.zeros(N, dtype=np.float32)
    order = np.argsort([len(c) for c in concats])
    i = 0
    while i < N:
        j = i
        maxlen = len(concats[order[i]])
        # grow batch while token budget allows (order ascending -> maxlen = last added)
        while j < N:
            ml = max(maxlen, len(concats[order[j]]))
            if (j - i + 1) * ml > args.max_batch_tokens and j > i:
                break
            maxlen = ml
            j += 1
        idx = order[i:j]
        batch = [concats[t] for t in idx]
        input_ids, seq_lengths = prepare_batch(batch, tokenizer, device=device, prepend_bos=True)
        with torch.inference_mode():
            logits, _ = model(input_ids)
        lp = logits_to_logprobs(logits, input_ids, trim_bos=True).float().cpu().numpy()
        for b, t in enumerate(idx):
            L = seq_lengths[b]
            row = lp[b][:L]
            ul = up_lens[t]
            full[t] = row.mean()
            cdsc[t] = row[ul:].mean()          # CDS tokens = after the upstream prefix
        i = j
    return full, cdsc


# --------------------------------------------------- build/score matrices per (sp, L)
log("STEP 3 — score full n_sub x n_sub matrices per length (checkpointed)")
t_start = time.time()
for L in LENGTHS:
    for sp in REF:
        fpath = f"{OUT}/S_full_L{L}_{tag(sp)}.npy"
        cpath = f"{OUT}/S_cds_L{L}_{tag(sp)}.npy"
        if os.path.exists(fpath) and os.path.exists(cpath):
            log(f"  reuse cached matrices L={L} {sp}")
            continue
        cds = genes[sp]["cds"]
        up = [u[-L:] for u in genes[sp]["up"]]
        n = len(cds)
        # all n*n concats: query i (CDS_i) x candidate j (upstream_j) -> up_j + cds_i
        concats, up_lens, ij = [], [], []
        for i in range(n):
            for j in range(n):
                concats.append(up[j] + cds[i])
                up_lens.append(len(up[j]))
                ij.append((i, j))
        t0 = time.time()
        full, cdsc = score_concats(concats, up_lens)
        Sf = np.zeros((n, n), dtype=np.float32)
        Sc = np.zeros((n, n), dtype=np.float32)
        for (i, j), f, c in zip(ij, full, cdsc):
            Sf[i, j] = f; Sc[i, j] = c
        np.save(fpath, Sf); np.save(cpath, Sc)
        el = time.time() - t0
        log(f"  L={L} {sp}: {n*n} pairs in {el:.0f}s ({1000*el/(n*n):.1f} ms/pair) "
            f"| diag_full={np.diag(Sf).mean():+.4f} offdiag_full={Sf[~np.eye(n,dtype=bool)].mean():+.4f} "
            f"| diag_cds={np.diag(Sc).mean():+.4f} offdiag_cds={Sc[~np.eye(n,dtype=bool)].mean():+.4f}")
log(f"scoring done in {(time.time()-t_start)/60:.1f} min")

# --------------------------------------------------- retrieval
log("STEP 4 — 1-in-100 x20 retrieval")
k = args.n_candidates - 1
records, summary = [], {}
METRICS = [("full", "S_full"), ("cds", "S_cds")]

for metric, prefix in METRICS:
    for L in LENGTHS:
        rng = np.random.default_rng(args.seed)      # reset per (metric,length)
        for sp in REF:
            S = np.load(f"{OUT}/{prefix}_L{L}_{tag(sp)}.npy")
            n = S.shape[0]
            rows = genes[sp]["rows"]
            true_s = np.diag(S).copy()
            diag_mean = float(true_s.mean())
            off_mean = float(S[~np.eye(n, dtype=bool)].mean())
            higher_better = diag_mean >= off_mean   # empirical direction
            ranks_hi = np.zeros((n, args.n_repeats), dtype=np.int32)  # higher=better
            ranks_lo = np.zeros((n, args.n_repeats), dtype=np.int32)  # lower=better
            for rep in range(args.n_repeats):
                d = rng.integers(0, n - 1, size=(n, k))
                d += (d >= np.arange(n)[:, None])
                cand = np.take_along_axis(S, d, axis=1)
                ranks_hi[:, rep] = 1 + (cand > true_s[:, None]).sum(axis=1)
                ranks_lo[:, rep] = 1 + (cand < true_s[:, None]).sum(axis=1)
            ranks = ranks_hi if higher_better else ranks_lo   # plotted ranking = empirical best
            top1_hi = float((ranks_hi == 1).mean()); top1_lo = float((ranks_lo == 1).mean())
            summary.setdefault(metric, {}).setdefault(sp, {})[str(L)] = dict(
                length=L, n=int(n), n_repeats=args.n_repeats,
                higher_is_better=bool(higher_better),
                top1_higher=top1_hi, top1_lower=top1_lo,
                top1=float((ranks == 1).mean()), top5=float((ranks <= 5).mean()),
                median_rank=float(np.median(ranks)), mean_rank=float(ranks.mean()),
                diag_mean=diag_mean, offdiag_mean=off_mean, score_gap=diag_mean - off_mean)
            log(f"  {metric:<4} L={L:<3} {sp:<20} dir={'HI' if higher_better else 'LO'} "
                f"top1={float((ranks==1).mean()):.3f} median={np.median(ranks):.0f} "
                f"gap={diag_mean-off_mean:+.4f}")
            for gi, r in enumerate(rows):
                for rep in range(args.n_repeats):
                    records.append((metric, sp, L, int(r), rep, int(ranks[gi, rep])))

# --------------------------------------------------- save + boxplots
log("STEP 5 — save tables + boxplots")
df = pd.DataFrame(records, columns=["metric", "species", "length", "gene_row", "rep", "rank"])
df.to_parquet(f"{OUT}/evo8k_score_ranks.parquet", index=False)
with open(f"{OUT}/evo8k_score_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

tbl = []
for metric, _ in METRICS:
    for sp in REF:
        for L in LENGTHS:
            s = summary[metric][sp][str(L)]
            tbl.append(dict(metric=metric, species=sp, length_bp=L, n_genes=s["n"],
                            direction="higher" if s["higher_is_better"] else "lower",
                            top1=round(s["top1"], 4), top5=round(s["top5"], 4),
                            median_rank=s["median_rank"],
                            top1_higher=round(s["top1_higher"], 4),
                            top1_lower=round(s["top1_lower"], 4),
                            score_gap=round(s["score_gap"], 4)))
pd.DataFrame(tbl).to_csv(f"{OUT}/summary_table.csv", index=False)

for metric, _ in METRICS:
    for sp in REF:
        sub = df[(df["metric"] == metric) & (df["species"] == sp)]
        data = [sub[sub["length"] == L]["rank"].to_numpy() for L in LENGTHS]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, tick_labels=[str(L) for L in LENGTHS], showmeans=True,
                   meanprops=dict(marker="D", markerfacecolor="crimson", markeredgecolor="crimson"))
        d0 = summary[metric][sp]["300"]
        dirn = "higher=better" if d0["higher_is_better"] else "lower=better"
        ax.set_xlabel("Upstream length in concat upstream+CDS (bp)")
        ax.set_ylabel("Rank of true pair (1 = best, of 100)   [raw per-trial]")
        ax.set_title(f"evo-1-8k-base SCORE retrieval [{metric}] — {sp}\n"
                     f"n={d0['n']} genes x {args.n_repeats} reps, concat=upstream+CDS, {dirn}")
        ax.grid(axis="y", alpha=0.3)
        for xi, L in enumerate(LENGTHS, start=1):
            ax.text(xi, ax.get_ylim()[1] * 0.98, f"top1\n{summary[metric][sp][str(L)]['top1']:.1%}",
                    ha="center", va="top", fontsize=8, color="navy")
        fig.tight_layout()
        fig.savefig(f"{OUT}/boxplot_{metric}_{tag(sp)}.png", dpi=150)
        plt.close(fig)
        log(f"  wrote boxplot_{metric}_{tag(sp)}.png")

log("\n============ SUMMARY (evo score retrieval) ============")
log(f"{'metric':<5} {'species':<20} {'L':>4} {'dir':>4} {'top1':>7} {'median':>7} {'gap':>9}")
for metric, _ in METRICS:
    for sp in REF:
        for L in LENGTHS:
            s = summary[metric][sp][str(L)]
            log(f"{metric:<5} {sp:<20} {L:>4} {'HI' if s['higher_is_better'] else 'LO':>4} "
                f"{s['top1']:>7.2%} {s['median_rank']:>7.0f} {s['score_gap']:>+9.4f}")
log("EVO8K_SCORE_OK")

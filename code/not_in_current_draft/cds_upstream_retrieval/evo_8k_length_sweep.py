"""
EVO-1-8k-base upstream LENGTH SWEEP retrieval  (experiment: evo_8k_sequence_length_retrieval)

DNA-only analogue of the C3P Exp-3 retrieval. C3P was cross-modal (protein <-> upstream in a
trained shared space); evo-1-8k-base is a DNA language model with no protein tower, so here
BOTH sides are DNA:

  * QUERY      = the gene's full-length CDS (coding DNA), embedded ONCE (length-independent).
  * CANDIDATES = the gene's own upstream + 99 random SAME-species upstreams, embedded at
                 upstream length L in {300, 200, 100, 50} bp (keep the LAST L bp = the region
                 proximal to the start codon, s[-L:], matching the C3P run and dataset.py).
  * 1-in-100 retrieval, rank by cosine(CDS_i, upstream_j). n_repeats=20 independent distractor
    draws per gene; RNG reset to default_rng(seed) at the START of each length + species order
    fixed (E. coli then B. subtilis) => gene i / repeat r faces the SAME 99 decoys at every
    length, exactly as the C3P length sweep.

Genes = the SAME 1,404 reference singletons used by the C3P run (727 E. coli + 677 B. subtilis),
taken from results/step0/analysis_set.parquet in ascending row order, with cds_seq left-joined
from data/ref2/ref2_flanks.csv on (accession, locus_tag)  [verified: 1404/1404, no nulls].

Embeddings: evo-1-8k-base final-layer hidden state, MEAN-POOLED over position (embed.mean(dim=1),
4096-d), then L2-normalised so the dot product is cosine. Identity-unembed trick from
evo/test_evo_base.py. batch=1 (no padding -> pooling is exact). bf16 forward on one GH200.

NOTE (interpretation): evo was never trained to align a CDS with its own promoter, so absolute
retrieval numbers are not comparable to C3P; the LENGTH-SWEEP TREND is the comparable signal.

Outputs (results/evo_8k_length/<out_subdir>):
  cds_emb.npy, up_emb_L{L}.npy   cached evo embeddings (fp16) -> cheap re-plot / resume
  evo8k_length_ranks.parquet     RAW per-trial ranks: species,length,gene_row,rep,rank
  evo8k_length_summary.json      per species x length: top1, top5, median_rank, mean_rank, n
  violin_<species>.png           x=length {300,200,100,50}, y=raw rank (pooled per-trial)
  summary_table.csv
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = f"{CL_ROOT}"
REF = ["Escherichia coli", "Bacillus subtilis"]     # fixes RNG consumption order
LENGTHS = [300, 200, 100, 50]

ap = argparse.ArgumentParser()
ap.add_argument("--n_candidates", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_repeats", type=int, default=20)
ap.add_argument("--limit", type=int, default=0, help="genes per species (0=all) -- preflight only")
ap.add_argument("--out_subdir", type=str, default="")
ap.add_argument("--force", action="store_true", help="ignore cached embeddings")
args = ap.parse_args()

OUT = f"{PROJ}/results/evo_8k_length" + (f"/{args.out_subdir}" if args.out_subdir else "")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"experiment=evo_8k_sequence_length_retrieval  OUT={OUT}")
log(f"args: {vars(args)}")

device = "cuda:0"
log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ----------------------------------------------------------------- data + CDS join
log("STEP 1/5 — load reference singletons + join cds_seq")
work = pd.read_parquet(f"{PROJ}/results/step0/analysis_set.parquet")
ref = work[work["species"].isin(REF)].copy()
ref["_row"] = np.arange(len(work))[work["species"].isin(REF).to_numpy()]  # original analysis_set row idx
flanks = pd.read_csv(f"{PROJ}/data/ref2/ref2_flanks.csv",
                     usecols=["accession", "locus_tag", "cds_seq", "upstream_seq"], dtype=str)
ref = ref.merge(flanks.rename(columns={"cds_seq": "cds_seq_f", "upstream_seq": "up_check"}),
                on=["accession", "locus_tag"], how="left")
assert ref["cds_seq_f"].isna().sum() == 0, "CDS join produced nulls"
assert (ref["upstream_seq"] == ref["up_check"]).all(), "upstream mismatch after join"

# per-species ascending-row selection == exp23 / C3P sweep ordering
rows_by_sp, cds_by_sp, up_by_sp = {}, {}, {}
for sp in REF:
    sub = ref[ref["species"] == sp].sort_values("_row")
    if args.limit:
        sub = sub.head(args.limit)
    rows_by_sp[sp] = sub["_row"].to_numpy()
    cds_by_sp[sp] = sub["cds_seq_f"].tolist()
    up_by_sp[sp] = sub["upstream_seq"].tolist()
    log(f"  {sp}: n={len(sub)}  cds_len med={int(np.median([len(s) for s in cds_by_sp[sp]]))}  "
        f"up_len med={int(np.median([len(s) for s in up_by_sp[sp]]))}")

# ----------------------------------------------------------------- evo model
log("STEP 2/5 — load evo-1-8k-base (identity unembed -> hidden states)")
from evo import Evo


class CustomEmbedding(nn.Module):
    def unembed(self, u):
        return u


evo_model = Evo("evo-1-8k-base")
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device).eval()
model.unembed = CustomEmbedding()
log("  evo loaded")


@torch.no_grad()
def embed_one(seq):
    """mean-pooled (4096,) L2-normalised evo embedding for one DNA sequence."""
    ids = torch.tensor(tokenizer.tokenize(seq), dtype=torch.int, device=device).unsqueeze(0)
    emb, _ = model(ids)                       # (1, L, 4096) bf16
    v = emb.float().mean(dim=1).squeeze(0)    # (4096,)
    v = v / v.norm(p=2).clamp_min(1e-8)
    return v.cpu().numpy().astype(np.float16)


def embed_list(seqs, tag):
    out = np.zeros((len(seqs), 4096), dtype=np.float16)
    t0 = time.time()
    for i, s in enumerate(seqs):
        out[i] = embed_one(s)
        if i % 200 == 0 or i == len(seqs) - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (len(seqs) - i - 1)
            log(f"    {tag} {i+1:>5}/{len(seqs)}  {el:6.1f}s  eta {eta:6.1f}s")
    return out


# ----------------------------------------------------------------- embed CDS (query, once)
log("STEP 3/5 — embed full-length CDS (query, length-independent)")
CDS = {}
for sp in REF:
    p = f"{OUT}/cds_emb_{sp.replace(' ', '_')}.npy"
    if os.path.exists(p) and not args.force and np.load(p).shape[0] == len(cds_by_sp[sp]):
        CDS[sp] = np.load(p); log(f"  reuse {p} {CDS[sp].shape}")
    else:
        CDS[sp] = embed_list(cds_by_sp[sp], f"cds:{sp[:6]}")
        np.save(p, CDS[sp])

# ----------------------------------------------------------------- embed upstreams per length
log("STEP 4/5 — embed upstreams at each length + retrieval")
k = args.n_candidates - 1
long_records = []
summary = {}

for L in LENGTHS:
    log(f"  --- L = {L} bp ---")
    UP = {}
    for sp in REF:
        p = f"{OUT}/up_emb_L{L}_{sp.replace(' ', '_')}.npy"
        if os.path.exists(p) and not args.force and np.load(p).shape[0] == len(up_by_sp[sp]):
            UP[sp] = np.load(p); log(f"    reuse {p} {UP[sp].shape}")
        else:
            UP[sp] = embed_list([s[-L:] for s in up_by_sp[sp]], f"up{L}:{sp[:6]}")
            np.save(p, UP[sp])

    rng = np.random.default_rng(args.seed)          # reset per length -> identical decoys
    for sp in REF:
        rows = rows_by_sp[sp]
        n = len(rows)
        U = UP[sp].astype(np.float32)               # candidates (upstream)
        Q = CDS[sp].astype(np.float32)              # query (CDS)
        Sp = Q @ U.T                                 # (n,n) rows=cds_i cols=upstream_j = cosine
        true_s = np.diag(Sp).copy()
        ranks = np.zeros((n, args.n_repeats), dtype=np.int32)
        for rep in range(args.n_repeats):
            d = rng.integers(0, n - 1, size=(n, k))
            d += (d >= np.arange(n)[:, None])        # exclude self
            cand = np.take_along_axis(Sp, d, axis=1)
            ranks[:, rep] = 1 + (cand > true_s[:, None]).sum(axis=1)

        top1 = float((ranks == 1).mean())
        top5 = float((ranks <= 5).mean())
        med = float(np.median(ranks))
        mean = float(ranks.mean())
        summary.setdefault(sp, {})[str(L)] = dict(length=L, n=int(n), n_repeats=args.n_repeats,
                                                   top1=top1, top5=top5, median_rank=med,
                                                   mean_rank=mean,
                                                   true_cos_mean=float(true_s.mean()))
        log(f"    {sp:<20} top1={top1:.4f} top5={top5:.4f} median={med:.0f} "
            f"mean={mean:.2f} trueCos={true_s.mean():+.4f}")
        for gi, r in enumerate(rows):
            for rep in range(args.n_repeats):
                long_records.append((sp, L, int(r), rep, int(ranks[gi, rep])))

# ----------------------------------------------------------------- save + plot
log("STEP 5/5 — save tables + violin plots")
df = pd.DataFrame(long_records, columns=["species", "length", "gene_row", "rep", "rank"])
df.to_parquet(f"{OUT}/evo8k_length_ranks.parquet", index=False)
with open(f"{OUT}/evo8k_length_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

tbl = []
for sp in REF:
    for L in LENGTHS:
        s = summary[sp][str(L)]
        tbl.append(dict(species=sp, length_bp=L, n_genes=s["n"], top1=round(s["top1"], 4),
                        top5=round(s["top5"], 4), median_rank=s["median_rank"],
                        mean_rank=round(s["mean_rank"], 2), true_cos_mean=round(s["true_cos_mean"], 4)))
pd.DataFrame(tbl).to_csv(f"{OUT}/summary_table.csv", index=False)

for sp in REF:
    sub = df[df["species"] == sp]
    data = [sub[sub["length"] == L]["rank"].to_numpy() for L in LENGTHS]
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0"); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(LENGTHS) + 1))
    ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.set_xlabel("Upstream length fed to evo-1-8k-base (bp)")
    ax.set_ylabel("Rank of true upstream (1 = best, of 100)   [raw per-trial]")
    ax.set_title(f"evo-1-8k-base length sweep (CDS -> upstream) — {sp}\n"
                 f"n={summary[sp]['300']['n']} genes x {args.n_repeats} repeats, same seeded pairs")
    ax.grid(axis="y", alpha=0.3)
    for xi, L in enumerate(LENGTHS, start=1):
        t1 = summary[sp][str(L)]["top1"]
        ax.text(xi, ax.get_ylim()[1] * 0.98, f"top1\n{t1:.1%}", ha="center", va="top",
                fontsize=8, color="navy")
    fig.tight_layout()
    fig.savefig(f"{OUT}/violin_{sp.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    log(f"  wrote violin_{sp.replace(' ', '_')}.png")

log("\n================ SUMMARY ================")
log(f"{'species':<20} {'L':>4} {'top1':>8} {'median':>7} {'top5':>8} {'trueCos':>8}")
for sp in REF:
    for L in LENGTHS:
        s = summary[sp][str(L)]
        log(f"{sp:<20} {L:>4} {s['top1']:>8.2%} {s['median_rank']:>7.0f} "
            f"{s['top5']:>8.2%} {s['true_cos_mean']:>+8.4f}")
log("EVO8K_LENGTH_OK")

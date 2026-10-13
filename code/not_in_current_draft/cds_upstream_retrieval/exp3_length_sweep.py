"""
EXP 3 — UPSTREAM LENGTH SWEEP.

Question: how does the length of the upstream region fed to C3P's promoter encoder
affect 1-in-100 retrieval? Does a longer upstream make the region more CDS-specific?

Design (only upstream length changes; everything else is held fixed):
  * Lengths L in {300, 200, 100, 50} bp. Keep the LAST L bp of each stored flank
    (3'-end, proximal to the start codon) -- s[-L:] -- matching dataset.py:22 and
    step2_embed.py. L=300 is the published baseline and doubles as a regression check.
  * Re-embed ONLY the reference-species upstreams (E. coli + B. subtilis singletons,
    ~1,400 rows) at each length through the identical promoter_encoder -> projection
    pipeline. Protein embeddings do NOT depend on L -> reuse the published
    results/step2/protein_emb.npz unchanged.
  * Retrieval is byte-identical to exp23_retrieval.py: 256-d shared space, query =
    protein_i, candidates = true upstream_i + 99 random SAME-species upstreams,
    rank by cosine. n_repeats=20.
  * REPRODUCIBLE / PAIRED across lengths: the distractor RNG is reset to
    default_rng(seed) at the start of EACH length and species are processed in the
    same order (E. coli then B. subtilis). So gene i / repeat r faces the SAME 99
    decoys at 300/200/100/50, and L=300 reproduces the original run exactly.
    The 99 distractors are themselves length-L (all candidates share the length),
    so length is never a giveaway.

Outputs (results/exp3_length/):
  exp3_length_ranks.parquet   long, RAW per-trial ranks: species,length,gene_row,rep,rank
  exp3_length_summary.json    per species x length: top1, median_rank, top5, mean_rank, n
  boxplot_<species>.png       x=length {300,200,100,50}, y=raw rank (pooled per-trial)
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = f"{CL_ROOT}"
S0, S2 = f"{PROJ}/results/step0", f"{PROJ}/results/step2"
OUT = f"{PROJ}/results/exp3_length"
os.makedirs(OUT, exist_ok=True)

REF = ["Escherichia coli", "Bacillus subtilis"]          # order fixes RNG consumption
LENGTHS = [300, 200, 100, 50]
# published L=300 baselines (from results/exp23_summary.json) -> regression guard
BASELINE_TOP1 = {"Escherichia coli": 0.8611, "Bacillus subtilis": 0.7744}

ap = argparse.ArgumentParser()
ap.add_argument("--n_candidates", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_repeats", type=int, default=20)
ap.add_argument("--batch_promoter", type=int, default=512)
args = ap.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device={device} | torch={torch.__version__} | cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

# ------------------------------------------------------------------ load model
from promoter_protein_clip.model_transformer import SequenceCLIPTransformer

ckpt_glob = f"{PROJ}/hf_cache/hub/models--dufaultc--contrastive-promoter-protein-pretraining/snapshots/*/C3P_100M.pth"
ckpt_path = glob.glob(ckpt_glob)[0]
print(f"checkpoint: {ckpt_path}", flush=True)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
ca = checkpoint["args"]
assert ca.max_promoter_length == 300, f"got {ca.max_promoter_length}"
assert ca.projection_dim == 256
assert ca.reverse_promoter is False
assert max(LENGTHS) <= ca.max_promoter_length, "cannot exceed the trained promoter length"

model = SequenceCLIPTransformer(
    protein_model_name=ca.protein_model,
    promoter_embedding_dim=ca.promoter_embedding_dim,
    depth=ca.transformer_depth,
    num_heads=ca.transformer_num_heads,
    max_promoter_length=ca.max_promoter_length,
    projection_dim=ca.projection_dim,
).to(device)
if any(k.startswith("_orig_mod.") for k in checkpoint["model_state_dict"]):
    model = torch.compile(model)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
core = model._orig_mod if hasattr(model, "_orig_mod") else model
print("model loaded strict=True", flush=True)


@torch.no_grad()
def embed_promoters_at(seqs, L, bs):
    """L2-normalised 256-d projection, keeping the LAST L bp of each sequence."""
    proj_out = np.zeros((len(seqs), ca.projection_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(seqs), bs):
        chunk = [s[-L:] for s in seqs[i:i + bs]]         # 3'-slice, matches training
        toks = core.promoter_tokenizer(chunk, padding=True, truncation=True,
                                       return_tensors="pt").to(device)
        feats = core.promoter_encoder(toks)
        proj = core.promoter_projection(feats)
        proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
        proj_out[i:i + len(chunk)] = proj.float().cpu().numpy()
    print(f"  embedded {len(seqs)} promoters @ L={L} in {time.time()-t0:.1f}s", flush=True)
    return proj_out


# ------------------------------------------------------------------ data
work = pd.read_parquet(f"{S0}/analysis_set.parquet")
qe = np.load(f"{S2}/protein_emb.npz")
prot_proj = qe["prot_proj"].astype(np.float32)
prot_idx = qe["idx"]
row2prot = {int(r): i for i, r in enumerate(prot_idx)}

# reference rows, ascending (identical selection to exp23_retrieval.py)
ref_rows = {}
for sp in REF:
    rr = np.where(work["species"].to_numpy() == sp)[0]
    rr = np.array([r for r in rr if r in row2prot])
    ref_rows[sp] = rr
    print(f"{sp}: n={len(rr)} singletons", flush=True)

k = args.n_candidates - 1
long_records = []           # raw per-trial ranks for the boxplot
summary = {}

for L in LENGTHS:
    print(f"\n===== L = {L} bp =====", flush=True)
    # embed both species' upstreams at this length
    U_by_sp = {sp: embed_promoters_at(work["upstream_seq"].iloc[ref_rows[sp]].tolist(),
                                      L, args.batch_promoter) for sp in REF}
    # reset RNG per length -> identical distractor draws across all lengths
    rng = np.random.default_rng(args.seed)
    for sp in REF:
        rows = ref_rows[sp]
        n = len(rows)
        U = U_by_sp[sp]
        P = prot_proj[[row2prot[int(r)] for r in rows]]
        Sp = P @ U.T                                     # (n,n) rows=protein_i cols=upstream_j
        true_s = np.diag(Sp).copy()
        ranks = np.zeros((n, args.n_repeats), dtype=np.int32)
        for rep in range(args.n_repeats):
            d = rng.integers(0, n - 1, size=(n, k))
            d += (d >= np.arange(n)[:, None])            # exclude self
            cand = np.take_along_axis(Sp, d, axis=1)
            ranks[:, rep] = 1 + (cand > true_s[:, None]).sum(axis=1)

        top1 = float((ranks == 1).mean())
        top5 = float((ranks <= 5).mean())
        med = float(np.median(ranks))
        mean = float(ranks.mean())
        summary.setdefault(sp, {})[str(L)] = dict(
            length=L, n=int(n), n_repeats=args.n_repeats,
            top1=top1, top5=top5, median_rank=med, mean_rank=mean)
        flag = ""
        if L == 300:
            exp = BASELINE_TOP1[sp]
            flag = "  <-- baseline" + ("  OK" if abs(top1 - exp) < 0.01
                                       else f"  *** MISMATCH vs {exp:.4f} ***")
        print(f"  {sp:<20} top1={top1:.4f} top5={top5:.4f} "
              f"median={med:.0f} mean={mean:.3f}{flag}", flush=True)

        # raw per-trial ranks (pooled) for boxplot
        for gi, r in enumerate(rows):
            for rep in range(args.n_repeats):
                long_records.append((sp, L, int(r), rep, int(ranks[gi, rep])))

# ------------------------------------------------------------------ save
df = pd.DataFrame(long_records, columns=["species", "length", "gene_row", "rep", "rank"])
df.to_parquet(f"{OUT}/exp3_length_ranks.parquet", index=False)
with open(f"{OUT}/exp3_length_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {OUT}/exp3_length_ranks.parquet ({len(df):,} rows) + summary.json", flush=True)

# ------------------------------------------------------------------ boxplots
for sp in REF:
    sub = df[df["species"] == sp]
    data = [sub[sub["length"] == L]["rank"].to_numpy() for L in LENGTHS]  # order 300..50
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=[str(L) for L in LENGTHS], showmeans=True,
               meanprops=dict(marker="D", markerfacecolor="crimson", markeredgecolor="crimson"))
    ax.set_xlabel("Upstream length fed to C3P (bp)")
    ax.set_ylabel("Rank of true upstream (1 = best, of 100)   [raw per-trial]")
    ax.set_title(f"Exp 3 length sweep — {sp}\n"
                 f"n={len(ref_rows[sp])} genes x {args.n_repeats} repeats, same seeded pairs")
    ax.grid(axis="y", alpha=0.3)
    # annotate top-1 under each box
    for xi, L in enumerate(LENGTHS, start=1):
        t1 = summary[sp][str(L)]["top1"]
        ax.text(xi, ax.get_ylim()[1] * 0.97, f"top1\n{t1:.1%}",
                ha="center", va="top", fontsize=8, color="navy")
    fig.tight_layout()
    png = f"{OUT}/boxplot_{sp.replace(' ', '_')}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"wrote {png}", flush=True)

# ------------------------------------------------------------------ printed table
print("\n================ SUMMARY TABLE ================")
print(f"{'species':<20} {'L':>4} {'top1':>8} {'median':>7} {'top5':>8}")
for sp in REF:
    for L in LENGTHS:
        s = summary[sp][str(L)]
        print(f"{sp:<20} {L:>4} {s['top1']:>8.2%} {s['median_rank']:>7.0f} {s['top5']:>8.2%}")
print("EXP3_LENGTH_OK")

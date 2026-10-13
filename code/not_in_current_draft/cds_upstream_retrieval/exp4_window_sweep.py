"""
EXP 4 — UPSTREAM *WINDOW* SWEEP (positional dissection, not just truncation).

Exp 3 only ever kept the bases PROXIMAL to the start codon (s[-L:]), so "200 bp beats
100 bp" is consistent with two different stories: the distal 100 bp carries gene-specific
signal, or simply "more sequence is better". This run separates them by feeding C3P
100 bp windows taken from different distances upstream of the start codon, so every
condition has the SAME amount of sequence and only its POSITION changes.

Windows are named by their distance upstream of the start codon:
    prox100   -100..0     the 100 bp immediately 5' of the ATG   (= exp3 L=100)
    dist100   -200..-100  the user's ask: keep the far 100 bp, delete the near 100 bp
    dist200   -300..-200  one step further out
plus the corresponding full windows (full200, full300) as within-design references.

Two designs, because the stored flanks are NOT all 300 bp (median ~213 bp in E. coli):
    d200  genes with >= 200 bp upstream   -> full200, prox100, dist100
    d300  genes with >= 300 bp upstream   -> full300, prox100, dist100, dist200
Within a design every gene has every window fully defined, so all conditions are paired
on exactly the same genes and no condition is silently shorter than the others.
This is the fix for a real wrinkle in exp 3: there, s[-300:] and s[-200:] are the same
bases for any gene whose flank is under 200 bp, which muted the 300->200 comparison.

Retrieval is byte-identical to exp3_length_sweep.py: 1-in-100, query = protein_i,
candidates = true upstream_i + 99 random same-species upstreams from the same design
subset, rank by cosine, n_repeats=20, RNG reset per condition so gene i / repeat r
faces the same 99 decoys in every window.

CAVEAT to carry into any figure: C3P was trained on windows anchored at the start
codon, so a -200..-100 window is presented to the encoder in the positional frame of a
-100..0 window. dist100 vs prox100 is therefore a clean comparison (identical length,
identical frame, only the underlying sequence differs), but neither is the same thing
as "the model reading the distal region in situ".

Outputs (ICBINB data/cds_upstream_retrieval/):
  c3p_window_ranks.parquet    design,species,window,gene_row,rep,rank
  c3p_window_summary.json     per design x species x window: top1, top5, median, mean, n
"""
# --- portable paths (added for release; override via environment) ---
import os as _os
from pathlib import Path as _Path
ICBINB_ROOT = _os.environ.get("ICBINB_ROOT", str(_Path(__file__).resolve().parents[3]))
CL_ROOT = _os.environ.get("CL_ROOT", "../Contrastive_Learning_NeurIPS2026")
# -------------------------------------------------------------------

import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

C3P = f"{CL_ROOT}"
ICBINB = f"{ICBINB_ROOT}"
sys.path.insert(0, f"{C3P}/contrastive-promoter-protein-pretraining")
S0, S2 = f"{C3P}/results/step0", f"{C3P}/results/step2"
OUT = f"{ICBINB}/data/cds_upstream_retrieval"
os.makedirs(OUT, exist_ok=True)

REF = ["Escherichia coli", "Bacillus subtilis"]      # order fixes RNG consumption

# design -> (min upstream length, [(window name, distal edge, proximal edge), ...])
# a window (name, b, a) keeps the bases from -b to -a, i.e. s[-b:-a] (s[-b:] when a == 0)
WINDOWS_COMMON = [
    ("prox50", 50, 0),        # -50..0     the bases right on top of the ATG
    ("prox100", 100, 0),      # -100..0    = exp3 L=100
    ("prox150", 150, 0),      # -150..0    width-matched control for dist150
    ("dist100", 200, 100),    # -200..-100 far half, near half deleted
    ("dist150", 200, 50),     # -200..-50  complement of prox50 within the 200 bp window
]
DESIGNS = {
    "d200": (200, [("full200", 200, 0)] + WINDOWS_COMMON),
    "d300": (300, [("full300", 300, 0)] + WINDOWS_COMMON + [("dist200", 300, 200)]),
}
# prox50 + dist150 exactly partition the 200 bp window, so together they say how the
# signal splits between the bases on top of the start codon and everything behind them.
# prox150 vs dist150 is the width-matched positional test at 150 bp.

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

ckpt_path = glob.glob(
    f"{C3P}/hf_cache/hub/models--dufaultc--contrastive-promoter-protein-pretraining"
    f"/snapshots/*/C3P_100M.pth")[0]
print(f"checkpoint: {ckpt_path}", flush=True)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
ca = checkpoint["args"]
assert ca.max_promoter_length == 300, f"got {ca.max_promoter_length}"
assert ca.projection_dim == 256
assert ca.reverse_promoter is False

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


def cut(s: str, b: int, a: int) -> str:
    """Keep the bases from -b to -a relative to the start codon."""
    return s[-b:] if a == 0 else s[-b:-a]


@torch.no_grad()
def embed_window(seqs, b, a, bs):
    """L2-normalised 256-d projection of one window of each upstream."""
    proj_out = np.zeros((len(seqs), ca.projection_dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(seqs), bs):
        chunk = [cut(s, b, a) for s in seqs[i:i + bs]]
        assert all(len(c) == b - a for c in chunk), "ragged window — subset filter is wrong"
        toks = core.promoter_tokenizer(chunk, padding=True, truncation=True,
                                       return_tensors="pt").to(device)
        proj = core.promoter_projection(core.promoter_encoder(toks))
        proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
        proj_out[i:i + len(chunk)] = proj.float().cpu().numpy()
    print(f"    embedded {len(seqs)} promoters @ -{b}..-{a} in {time.time()-t0:.1f}s", flush=True)
    return proj_out


# ------------------------------------------------------------------ data
work = pd.read_parquet(f"{S0}/analysis_set.parquet")
qe = np.load(f"{S2}/protein_emb.npz")
prot_proj = qe["prot_proj"].astype(np.float32)
row2prot = {int(r): i for i, r in enumerate(qe["idx"])}
up_len = work["upstream_seq"].str.len().to_numpy()

k = args.n_candidates - 1
records, summary = [], {}

for design, (min_len, windows) in DESIGNS.items():
    print(f"\n########## design {design}  (upstream >= {min_len} bp) ##########", flush=True)
    # gene subset per species: has a protein embedding AND a long enough flank
    rows_by_sp = {}
    for sp in REF:
        rr = np.where((work["species"].to_numpy() == sp) & (up_len >= min_len))[0]
        rr = np.array([r for r in rr if r in row2prot])
        rows_by_sp[sp] = rr
        print(f"  {sp}: n={len(rr)} genes", flush=True)
        assert len(rr) > args.n_candidates, "not enough genes to draw 99 distractors"

    for name, b, a in windows:
        print(f"  --- window {name} (-{b}..-{a}) ---", flush=True)
        U_by_sp = {sp: embed_window(work["upstream_seq"].iloc[rows_by_sp[sp]].tolist(),
                                    b, a, args.batch_promoter) for sp in REF}
        rng = np.random.default_rng(args.seed)      # reset per window -> paired decoys
        for sp in REF:
            rows, U = rows_by_sp[sp], U_by_sp[sp]
            n = len(rows)
            P = prot_proj[[row2prot[int(r)] for r in rows]]
            S = P @ U.T                              # rows = protein_i, cols = upstream_j
            true_s = np.diag(S).copy()
            ranks = np.zeros((n, args.n_repeats), dtype=np.int32)
            for rep in range(args.n_repeats):
                d = rng.integers(0, n - 1, size=(n, k))
                d += (d >= np.arange(n)[:, None])    # exclude self
                cand = np.take_along_axis(S, d, axis=1)
                ranks[:, rep] = 1 + (cand > true_s[:, None]).sum(axis=1)

            summary.setdefault(design, {}).setdefault(sp, {})[name] = dict(
                window=name, distal_edge=-b, proximal_edge=-a, width=b - a, n=int(n),
                n_repeats=args.n_repeats,
                top1=float((ranks == 1).mean()), top5=float((ranks <= 5).mean()),
                median_rank=float(np.median(ranks)), mean_rank=float(ranks.mean()))
            s = summary[design][sp][name]
            print(f"    {sp:<20} top1={s['top1']:.4f} top5={s['top5']:.4f} "
                  f"median={s['median_rank']:.0f}", flush=True)

            for gi, r in enumerate(rows):
                for rep in range(args.n_repeats):
                    records.append((design, sp, name, int(r), rep, int(ranks[gi, rep])))

# ------------------------------------------------------------------ save
df = pd.DataFrame(records, columns=["design", "species", "window", "gene_row", "rep", "rank"])
df.to_parquet(f"{OUT}/c3p_window_ranks.parquet", index=False)
with open(f"{OUT}/c3p_window_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {OUT}/c3p_window_ranks.parquet ({len(df):,} rows) + c3p_window_summary.json",
      flush=True)

print("\n================ SUMMARY TABLE ================")
print(f"{'design':<7} {'species':<20} {'window':<9} {'n':>5} {'top1':>8} {'median':>7} {'top5':>8}")
for design, (_, windows) in DESIGNS.items():
    for sp in REF:
        for name, _, _ in windows:
            s = summary[design][sp][name]
            print(f"{design:<7} {sp:<20} {name:<9} {s['n']:>5} {s['top1']:>8.2%} "
                  f"{s['median_rank']:>7.0f} {s['top5']:>8.2%}")
print("EXP4_WINDOW_OK")

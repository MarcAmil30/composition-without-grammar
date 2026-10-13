#!/usr/bin/env python3
"""
Task 4 -- C3P. Embedding only: C3P is contrastive and has no likelihood head,
so cosine in its trained shared space IS its score.

Query = the PROTEIN (not the CDS) -- that is the modality C3P was trained to
align with promoters. Candidate = the 200 bp upstream. Rank by cosine of the
256-d L2-normalised projections.

C3P is the positive control and the ceiling for the whole task: it was never
trained on E. coli, and on the older 727-gene set it reached 86.1% top-1.

Preprocessing replicated verbatim from step2_embed.py (getting this wrong is how
the first C3P run produced a confident, publishable, WRONG null):
  * promoter = seq[-L:], the 3' end, because KmerTokenizer keeps tokens[:300]
  * protein  = ATTENTION-MASKED MEAN pool of ESM2 last_hidden_state, NOT CLS
  * torch.compile BEFORE load_state_dict (checkpoint has _orig_mod. prefixes)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RETR = Path(os.environ.get("ICBINB_RETRIEVAL_DIR", REPO_ROOT / "data" / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval_eval as EV                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--suffix", type=str, default="")
ap.add_argument("--c3p-repo", type=Path,
                default=os.environ.get("C3P_REPO"),
                help="checkout containing contrastive-promoter-protein-pretraining")
ap.add_argument("--checkpoint", type=Path,
                default=os.environ.get("C3P_CHECKPOINT"),
                help="C3P_100M.pth (otherwise searched below --c3p-repo)")
args = ap.parse_args()

if args.c3p_repo is None:
    ap.error("set --c3p-repo (or C3P_REPO) to the public C3P checkout")
CL = args.c3p_repo.expanduser().resolve()
sys.path.insert(0, str(CL / "contrastive-promoter-protein-pretraining"))

OUT = RETR / "results" / "c3p"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[+{time.time()-T0:6.0f}s] {m}", flush=True)


def status(m):
    line = f"c3p\tembed\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{m}"
    print("STATUS\t" + line, flush=True)
    with open(f"{RETR}/results/progress.tsv", "a") as fh:
        fh.write(line + "\n")


from promoter_protein_clip.model_transformer import SequenceCLIPTransformer   # noqa: E402

if args.checkpoint:
    ckpt_path = args.checkpoint.expanduser().resolve()
else:
    matches = glob.glob(str(CL / "**" / "C3P_100M.pth"), recursive=True)
    if not matches:
        ap.error("C3P_100M.pth not found; pass --checkpoint or C3P_CHECKPOINT")
    ckpt_path = Path(matches[0])
ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
ca = ck["args"]
log(f"checkpoint {ckpt_path.name} | max_promoter_length={ca.max_promoter_length}")
assert ca.projection_dim == 256 and ca.reverse_promoter is False

model = SequenceCLIPTransformer(
    protein_model_name=ca.protein_model,
    promoter_embedding_dim=ca.promoter_embedding_dim,
    depth=ca.transformer_depth,
    num_heads=ca.transformer_num_heads,
    max_promoter_length=ca.max_promoter_length,
    projection_dim=ca.projection_dim,
).to(dev)
if any(k.startswith("_orig_mod.") for k in ck["model_state_dict"]):
    model = torch.compile(model)                     # BEFORE load_state_dict
model.load_state_dict(ck["model_state_dict"], strict=True)
model.eval()
core = model._orig_mod if hasattr(model, "_orig_mod") else model
log("state_dict loaded strict=True")


@torch.no_grad()
def embed_promoters(seqs, bs):
    out = np.zeros((len(seqs), ca.projection_dim), np.float32)
    for i in range(0, len(seqs), bs):
        chunk = [s[-EV.UP_LEN:] for s in seqs[i:i + bs]]     # 3' end, as trained
        toks = core.promoter_tokenizer(chunk, padding=True, truncation=True,
                                       return_tensors="pt").to(dev)
        p = torch.nn.functional.normalize(
            core.promoter_projection(core.promoter_encoder(toks)), p=2, dim=-1)
        out[i:i + len(chunk)] = p.float().cpu().numpy()
    return out


@torch.no_grad()
def embed_proteins(seqs, bs):
    out = np.zeros((len(seqs), ca.projection_dim), np.float32)
    order = np.argsort([len(s) for s in seqs])                # padding efficiency
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        toks = core.protein_tokenizer([seqs[j] for j in idx], padding=True,
                                      truncation=True, max_length=ca.protein_max_len,
                                      return_tensors="pt").to(dev)
        h = core.protein_encoder(**toks).last_hidden_state
        m = toks["attention_mask"].unsqueeze(-1).expand(h.size()).to(h.dtype)
        feats = (h * m).sum(1) / m.sum(1)                     # MASKED MEAN, not CLS
        p = torch.nn.functional.normalize(core.protein_projection(feats), p=2, dim=-1)
        out[idx] = p.float().cpu().numpy()
    return out


df = EV.load_eval(limit=args.limit)
n = len(df)
status(f"start n={n} L={EV.UP_LEN}")
EV.decoy_draws(n)

P = embed_proteins(df.protein_seq.tolist(), args.batch)
log(f"{n} protein embeddings")
U = embed_promoters(df.upstream_seq.tolist(), args.batch)
log(f"{n} promoter embeddings")
np.save(f"{OUT}/embed_protein{args.suffix}.npy", P)
np.save(f"{OUT}/embed_promoter{args.suffix}.npy", U)

# GUARD -- the projections must be unit norm and deterministic
assert abs(np.linalg.norm(P, axis=1).mean() - 1) < 1e-3, "protein proj not normalised"
assert abs(np.linalg.norm(U, axis=1).mean() - 1) < 1e-3, "promoter proj not normalised"

S = P @ U.T
log(f"true-pair cosine {np.diag(S).mean():.4f} | off-diagonal {S.mean():.4f}")

for name, Mx in [("embed_cos", S),
                 ("embed_cos_centered", None)]:
    if Mx is None:
        Pc = P - P.mean(0, keepdims=True); Uc = U - U.mean(0, keepdims=True)
        Pc /= np.linalg.norm(Pc, axis=1, keepdims=True).clip(1e-8)
        Uc /= np.linalg.norm(Uc, axis=1, keepdims=True).clip(1e-8)
        Mx = Pc @ Uc.T
    s = EV.evaluate(Mx, name + args.suffix, OUT, True,
                    extra={"model": "c3p", "scorer": "embedding",
                           "query_modality": "protein",
                           "true_cos_mean": float(np.diag(Mx).mean())})
    EV.report(s)
    if name == "embed_cos":
        status(f"RESULT embed top1={s['top1']*100:.2f}% median={s['median_rank']:.0f}")
        # regression guard against the published 727-gene run
        if s["top1"] < 0.5:
            status(f"WARNING top1={s['top1']*100:.1f}% but the 727-gene run gave 86% "
                   f"-- check pooling (CLS vs mean) before believing this")
status("done")

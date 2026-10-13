#!/usr/bin/env python3
"""
Task 4 -- Evo 2. Same protocol as the other models; separate file because Evo 2
ships its own package (`evo2`) with a non-HF API, and its quirks must not leak
into the HF path.

  framing   <upstream><CDS>   genomic order, bare DNA (as trained)
  scoring   score the CDS tokens -> log P(CDS | upstream)
  embedding mean-pooled final hidden state, cosine

API differences from the HF models:
  * Evo2("evo2_1b_base") -> .model (vortex) and .tokenizer (CharLevelTokenizer)
  * forward(input_ids) returns (logits, embeddings_dict), NOT a HF output object
  * vortex places the model on CUDA itself -- never call .to(device)
  * embeddings come from a forward hook on a named submodule, discovered at
    runtime because the layer names differ between the 1b/7b/40b configs

Headline metric is PMI (column-centred), not raw likelihood -- see
lm_retrieval.py for why.
"""
from __future__ import annotations

import argparse
import gc
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
ap.add_argument("--mode", choices=["score", "embed"], required=True)
ap.add_argument("--ckpt", default="evo2_1b_base")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget", type=int, default=3000)
ap.add_argument("--suffix", type=str, default="")
args = ap.parse_args()

OUT = RETR / "results" / "evo2"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[+{time.time()-T0:6.0f}s] {m}", flush=True)


def status(m):
    line = f"evo2\t{args.mode}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{m}"
    print("STATUS\t" + line, flush=True)
    with open(f"{RETR}/results/progress.tsv", "a") as fh:
        fh.write(line + "\n")


from evo2 import Evo2                                           # noqa: E402

ev = Evo2(args.ckpt)
dev = "cuda"
log(f"loaded {args.ckpt}")

# GUARD 1 -- one token per nucleotide, or every offset is meaningless
probe = ev.tokenizer.tokenize("ACGTACGTAC")
assert len(probe) == 10, f"tokenizer is not single-nucleotide: 10 bases -> {len(probe)}"
log("  guard 1 OK: 1 token per nucleotide")


def ids_of(seqs):
    return torch.tensor([ev.tokenizer.tokenize(s) for s in seqs],
                        dtype=torch.int, device=dev)


def logprobs(seqs, n_prompt):
    """(mean logprob of the CDS segment, mean logprob of the whole sequence)."""
    with torch.no_grad():
        ids = ids_of(seqs)
        assert ids.shape[1] == len(seqs[0]), \
            f"tokenized {ids.shape[1]} != chars {len(seqs[0])}"
        logits, _ = ev.forward(ids)
        # vortex returns its own tuple from model.forward, so Evo2.forward hands
        # back (tuple, None) rather than (tensor, None) -- unwrap it.
        if isinstance(logits, tuple):
            logits = logits[0]
        lp = torch.log_softmax(logits.float(), -1)
        tok_lp = lp[:, :-1].gather(-1, ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)
        cond = tok_lp[:, n_prompt - 1:].mean(1)
        full = tok_lp.mean(1)
        out = cond.float().cpu().numpy(), full.float().cpu().numpy()
        del logits, lp, tok_lp, ids
    return out


def safe_logprobs(seqs, n_prompt):
    """OOM handled OUTSIDE the except -- an active except pins the failing
    frame's tensors, so empty_cache() frees nothing (see lm_retrieval.py)."""
    oom = False
    try:
        return logprobs(seqs, n_prompt)
    except torch.OutOfMemoryError:
        oom = True
    if oom:
        gc.collect(); gc.collect()
        torch.cuda.empty_cache()
        cs, fs = [], []
        for one in seqs:
            c, f = logprobs([one], n_prompt)
            cs.append(c); fs.append(f)
            gc.collect(); torch.cuda.empty_cache()
        return np.concatenate(cs), np.concatenate(fs)


# ------------------------------------------------------------------ scoring
def run_score(df):
    n = len(df)
    cds, up = df.cds_seq.tolist(), df.up200.tolist()
    p_c = f"{OUT}/score_cond{args.suffix}_matrix.npy"
    p_f = f"{OUT}/score_full{args.suffix}_matrix.npy"
    p_d = f"{OUT}/score{args.suffix}_done.npy"
    S_c = np.load(p_c) if os.path.exists(p_c) else np.full((n, n), np.nan, np.float32)
    S_f = np.load(p_f) if os.path.exists(p_f) else np.full((n, n), np.nan, np.float32)
    done = np.load(p_d) if os.path.exists(p_d) else np.zeros(n, bool)
    if done.any():
        status(f"resume {int(done.sum())}/{n} rows")
    todo = [i for i in range(n) if not done[i]]
    status(f"scoring {len(todo)} of {n} query rows")

    for pos, i in enumerate(todo):
        seqs = [u + cds[i] for u in up]          # upstream first, genomic order
        L = len(seqs[0])
        w = max(1, args.budget // max(L, 1))
        cr, fr = np.empty(n, np.float32), np.empty(n, np.float32)
        for s in range(0, n, w):
            c, f = safe_logprobs(seqs[s:s + w], EV.UP_LEN)
            cr[s:s + len(c)] = c
            fr[s:s + len(f)] = f
        S_c[i], S_f[i], done[i] = cr, fr, True
        if pos < 2 or (pos + 1) % 10 == 0:
            rate = (pos + 1) / (time.time() - T0)
            log(f"  row {pos+1}/{len(todo)} (L={L}, w={w}) | {rate*3600:.0f} rows/h "
                f"| ETA {(len(todo)-pos-1)/max(rate,1e-9)/3600:.1f} h")
        if (pos + 1) % 10 == 0 or pos == len(todo) - 1:
            np.save(p_c, S_c); np.save(p_f, S_f); np.save(p_d, done)
            if (pos + 1) % 50 == 0:
                status(f"progress {int(done.sum())}/{n} rows")

    np.save(p_c, S_c); np.save(p_f, S_f); np.save(p_d, done)
    assert not np.isnan(S_c).any(), "score matrix has holes"
    status(f"scoring complete {n}/{n}")

    # GUARD 3 -- is the matrix diagonal really the true pair?  NON-FATAL: the
    # scoring is already saved, so a diagnostic must not destroy the run. The
    # tolerance is scaled to the row's spread -- re-scoring at width 1 against a
    # row scored at width w drifts in bf16, whereas a real index misalignment
    # would be off by roughly the row spread.
    guard3 = {"checked": 0, "max_drift_vs_rowstd": 0.0, "suspect": False}
    for i in range(min(3, n)):
        c, _ = safe_logprobs([up[i] + cds[i]], EV.UP_LEN)
        row = S_c[i]
        drift = abs(float(c[0] - row[i]))
        rel = drift / max(float(row.std()), 1e-9)
        guard3["checked"] += 1
        guard3["max_drift_vs_rowstd"] = max(guard3["max_drift_vs_rowstd"], rel)
        if rel > 0.5:
            guard3["suspect"] = True
            log(f"  !! GUARD 3 SUSPECT gene {i}: drift {drift:.4f} = {rel:.2f}x row sd")
        elif drift > 1e-3:
            log(f"  guard 3 gene {i}: drift {drift:.2e} ({rel:.3f}x row sd) -- bf16, fine")
    log(f"  guard 3 {'SUSPECT' if guard3['suspect'] else 'OK'}: "
        f"max {guard3['max_drift_vs_rowstd']:.3f}x row sd")

    vd = EV.variance_decomposition(S_c)
    log(f"  variance: candidate {vd['between_candidate_frac']*100:.1f}% | "
        f"interaction {vd['interaction_frac']*100:.1f}%")

    for base, S in [("score_cond", S_c), ("score_full", S_f)]:
        for how in ["raw", "pmi", "zcol"]:
            s = EV.evaluate(EV.normalize(S, how), f"{base}_{how}{args.suffix}", OUT, True,
                            extra={"model": "evo2", "scorer": "scoring",
                                   "normalization": how, "checkpoint": args.ckpt,
                                   "guard3_diagonal": guard3},
                            save_matrix=False)
            EV.report(s)
            if how == "pmi" and base == "score_cond":
                status(f"RESULT score pmi top1={s['top1']*100:.2f}% "
                       f"median={s['median_rank']:.0f} "
                       f"gap_frac={s['guard_true_vs_decoy']['frac_true_higher']*100:.1f}%")


# ------------------------------------------------------------------ embedding
def run_embed(df):
    n = len(df)
    # layer names differ per config, so discover rather than hardcode
    names = [nm for nm, _ in ev.model.named_modules() if nm]
    layer = None
    for cand in ("norm", "backbone.norm", "final_norm"):
        if cand in names:
            layer = cand
            break
    if layer is None:                              # fall back to the last block
        blocks = sorted({nm for nm in names if nm.startswith("blocks.")},
                        key=lambda s: (int(s.split(".")[1]), s))
        assert blocks, f"no norm and no blocks in: {names[:20]}"
        layer = blocks[-1]
    log(f"  embedding layer: {layer}")

    def embed(seq):
        with torch.no_grad():
            ids = ids_of([seq])
            _, emb = ev.forward(ids, return_embeddings=True, layer_names=[layer])
            h = emb[layer]
            if isinstance(h, tuple):
                h = h[0]
            v = h.float().mean(1).squeeze(0)
            v = (v / v.norm().clamp_min(1e-8)).cpu().numpy()
            del ids, emb, h
        return v

    a, b = embed(df.up200.iloc[0]), embed(df.up200.iloc[0])
    assert float(a @ b) > 0.999, f"embedding not deterministic: {a @ b:.4f}"
    log(f"  guard 4 OK: cos={a @ b:.5f}, dim={a.shape[0]}")

    pq, pc = f"{OUT}/embed_query{args.suffix}.npy", f"{OUT}/embed_cand{args.suffix}.npy"
    if os.path.exists(pq) and os.path.exists(pc):
        Q, C = np.load(pq), np.load(pc)
        status("embeddings from cache")
    else:
        Q = np.stack([embed(c) for c in df.cds_seq])
        np.save(pq, Q.astype(np.float32)); status(f"{n} query embeddings")
        C = np.stack([embed(u) for u in df.up200])
        np.save(pc, C.astype(np.float32)); status(f"{n} candidate embeddings")

    raw = Q @ C.T
    Qc = Q - Q.mean(0, keepdims=True); Cc = C - C.mean(0, keepdims=True)
    Qc /= np.linalg.norm(Qc, axis=1, keepdims=True).clip(1e-8)
    Cc /= np.linalg.norm(Cc, axis=1, keepdims=True).clip(1e-8)
    log(f"  true-pair cosine raw {np.diag(raw).mean():.4f} | "
        f"centered {np.diag(Qc @ Cc.T).mean():.4f}")
    for name, S in [("embed_cos", raw), ("embed_cos_centered", Qc @ Cc.T)]:
        s = EV.evaluate(S, name + args.suffix, OUT, True,
                        extra={"model": "evo2", "scorer": "embedding",
                               "checkpoint": args.ckpt,
                               "true_cos_mean": float(np.diag(S).mean())})
        EV.report(s)
        if name == "embed_cos_centered":
            status(f"RESULT embed centered top1={s['top1']*100:.2f}% "
                   f"median={s['median_rank']:.0f}")


df = EV.load_eval(limit=args.limit)
status(f"start n={len(df)} L={EV.UP_LEN} ckpt={args.ckpt} budget={args.budget}")
EV.decoy_draws(len(df))
run_score(df) if args.mode == "score" else run_embed(df)
status("done")

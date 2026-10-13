#!/usr/bin/env python3
"""
Task 4 — PROK. 1-in-100 CDS <-> upstream retrieval, E. coli, L=200.

Not a capability check. PROK already generates realistic, species-typical
upstreams (tasks 1-3). This asks whether it knows WHICH upstream belongs to
WHICH gene. Generates well + retrieves at chance = species-typical but not
CDS-specific, which is the paper's claim.

So the run is built to make a NEGATIVE result believable: see the guards below.

Two modes, one framing, one dataset:

  --mode score   feed  +#<CDS>+$<upstream>  and score the UPSTREAM tokens
                 -> log P(upstream | CDS).  CDS-first is PROK's native training
                 order, so no concatenation trick is needed.
                 Pairwise: n^2 forwards. This is the expensive one.

  --mode embed   embed the formatted strings INCLUDING the prompt tokens --
                 query "+#<CDS>+$", candidate "+$<upstream>" -- mean-pool the
                 final hidden state, rank by cosine. Minutes.

PROK's species token is '+' for E. coli ('@' for B. subtilis), repeated before
each segment tag: <sp>#<CDS><sp>$<5'UTR><sp>*<3'UTR>.

GUARDS (a null is only worth reporting if these pass)
  1. tokenizer is one-token-per-character, including '+', '#', '$'
  2. prompt length in tokens == prompt length in characters, per gene
  3. matrix diagonal really is the true pair (independently recomputed)
  4. embed mode: identical sequences give cosine 1.0
  5. true-vs-decoy gap reported alongside top-1 (see retrieval_eval.py)

StripedHyena notes, learned the hard way in cds_generate.py:
  * the FFT filter cache is keyed by fft_size and grows without bound; the only
    reliable release is to rebuild the model. One query block = one length, so
    we rebuild every RESET_EVERY blocks.
  * batches must be equal-length -- no padding. Within a query block every
    candidate has the same length, so this is free.

Outputs (data/benchmark/retrieval/results/prok/):
  score_cond_matrix.npy / score_full_matrix.npy   n x n raw scores (resumable)
  embed_cos_matrix.npy, embed_cos_centered_matrix.npy
  *_ranks.parquet, *_summary.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "code" / "generation"))
import models as M                                              # noqa: E402
import retrieval_eval as EV                                     # noqa: E402

RETR = Path(os.environ.get("ICBINB_RETRIEVAL_DIR", REPO_ROOT / "data" / "retrieval"))
OUT = RETR / "results" / "prok"
SP = "+"                       # E. coli species token ('@' = B. subtilis)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["score", "embed"], required=True)
ap.add_argument("--limit", type=int, default=0, help="smoke test: first N genes")
ap.add_argument("--budget", type=int, default=3000, help="tokens per forward pass")
ap.add_argument("--reset_every", type=int, default=20, help="rebuild model every N query blocks")
ap.add_argument("--embed_reset", type=int, default=40, help="rebuild every N sequences in embed mode (0=never)")
ap.add_argument("--suffix", type=str, default="")
args = ap.parse_args()

os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ framing
def query_prompt(cds):
    """What the model sees before the upstream begins."""
    return f"{SP}#{cds}{SP}$"


def scored_seq(cds, up):
    return query_prompt(cds) + up


def candidate_embed_seq(up):
    """Upstream framed as its own segment, prompt tokens included."""
    return f"{SP}${up}"


# ------------------------------------------------------------------ model
def build():
    a = M.build("prok").load()
    return a


def check_tokenizer(adapter):
    """GUARD 1 -- every framing character must be exactly one token."""
    for ch in ["A", "C", "G", "T", SP, "#", "$"]:
        t = adapter.tok(ch, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, f"'{ch}' is not one token: {t}"
    probe = "+#ACGT+$ACGT"
    n = len(adapter.tok(probe, add_special_tokens=False)["input_ids"])
    assert n == len(probe), f"framing is not 1 token/char: {probe} -> {n} tokens"
    log(f"  guard 1 OK: 1 token per character, including {SP} # $")


# ------------------------------------------------------------------ scoring
def run_score(df):
    import torch

    n = len(df)
    cds = df.cds_seq.tolist()
    up = df.up200.tolist()

    p_cond = f"{OUT}/score_cond{args.suffix}_matrix.npy"
    p_full = f"{OUT}/score_full{args.suffix}_matrix.npy"
    p_done = f"{OUT}/score{args.suffix}_done.npy"

    S_cond = np.load(p_cond) if os.path.exists(p_cond) else np.full((n, n), np.nan, np.float32)
    S_full = np.load(p_full) if os.path.exists(p_full) else np.full((n, n), np.nan, np.float32)
    done = np.load(p_done) if os.path.exists(p_done) else np.zeros(n, bool)
    if done.any():
        log(f"resuming: {done.sum()}/{n} query rows already scored")

    adapter = build()
    check_tokenizer(adapter)
    dev = adapter.device

    def rebuild():
        nonlocal adapter
        adapter.model = None
        gc.collect()
        torch.cuda.empty_cache()
        adapter = build()

    @torch.inference_mode()
    def score_batch(seqs, n_prompt):
        """mean log-prob of the upstream tokens, and of the whole sequence."""
        enc = adapter.tok(list(seqs), return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(dev)
        # GUARD 2 -- character offsets must equal token offsets
        assert ids.shape[1] == len(seqs[0]), \
            f"tokenized length {ids.shape[1]} != char length {len(seqs[0])}"
        logits = adapter.model(ids).logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:].unsqueeze(-1)
        tok_lp = lp[:, :-1].gather(-1, tgt).squeeze(-1)          # [B, T-1]
        # token t is predicted by position t-1, so upstream tokens
        # [n_prompt, n_prompt+199] sit at lp columns [n_prompt-1, n_prompt+198]
        cond = tok_lp[:, n_prompt - 1: n_prompt + EV.UP_LEN - 1].mean(1)
        full = tok_lp.mean(1)
        del logits, lp, tok_lp, ids
        return cond.float().cpu().numpy(), full.float().cpu().numpy()

    todo = [i for i in range(n) if not done[i]]
    log(f"scoring {len(todo)} query rows x {n} candidates")
    nb = 0
    for pos, i in enumerate(todo):
        if nb and nb % args.reset_every == 0:
            rebuild()
        nb += 1
        prompt = query_prompt(cds[i])
        n_prompt = len(prompt)
        seqs = [prompt + u for u in up]
        L = len(seqs[0])
        width = max(1, args.budget // max(L, 1))

        c_row = np.empty(n, np.float32)
        f_row = np.empty(n, np.float32)
        for s in range(0, n, width):
            chunk = seqs[s:s + width]
            try:
                c, f = score_batch(chunk, n_prompt)
            except torch.OutOfMemoryError:
                rebuild()
                c_l, f_l = [], []
                for one in chunk:
                    a_, b_ = score_batch([one], n_prompt)
                    c_l.append(a_); f_l.append(b_)
                c, f = np.concatenate(c_l), np.concatenate(f_l)
            c_row[s:s + len(chunk)] = c
            f_row[s:s + len(chunk)] = f
        S_cond[i] = c_row
        S_full[i] = f_row
        done[i] = True

        if pos < 2 or (pos + 1) % 10 == 0:
            el = time.time() - T0
            rate = (pos + 1) / el
            eta = (len(todo) - pos - 1) / max(rate, 1e-9)
            log(f"  row {pos+1}/{len(todo)} (gene {i}, L={L}, width={width}) "
                f"| {rate*3600:.0f} rows/h | ETA {eta/3600:.1f} h")
        if (pos + 1) % 10 == 0 or pos == len(todo) - 1:
            np.save(p_cond, S_cond); np.save(p_full, S_full); np.save(p_done, done)

    np.save(p_cond, S_cond); np.save(p_full, S_full); np.save(p_done, done)
    assert not np.isnan(S_cond).any(), "score matrix has holes"

    # GUARD 3 -- the diagonal really is the true pair
    k = min(3, n)
    for i in range(k):
        c, _ = score_batch([scored_seq(cds[i], up[i])], len(query_prompt(cds[i])))
        assert abs(c[0] - S_cond[i, i]) < 1e-3, \
            f"diagonal mismatch at {i}: {c[0]} vs {S_cond[i, i]} -- index misalignment"
    log(f"  guard 3 OK: diagonal verified on {k} genes")

    # ---- null-context scores: log P(upstream) with NO CDS. n forwards, not n^2.
    # Gives a non-transductive estimate of the per-candidate nuisance term, so
    # PMI can be formed without borrowing information from the other queries.
    p_null = f"{OUT}/score_null{args.suffix}.npy"
    if os.path.exists(p_null):
        null = np.load(p_null)
    else:
        log("scoring null context (+$<upstream>, no CDS)")
        null = np.empty(n, np.float32)
        prefix = f"{SP}$"
        for s in range(0, n, 32):
            chunk = [prefix + u for u in up[s:s + 32]]
            c, _ = score_batch(chunk, len(prefix))
            null[s:s + len(chunk)] = c
        np.save(p_null, null)
    log(f"  null-context mean log-prob {null.mean():.4f}")

    vd = EV.variance_decomposition(S_cond)
    log(f"  variance: candidate {vd['between_candidate_frac']*100:.1f}% | "
        f"query {vd['between_query_frac']*100:.1f}% | "
        f"interaction {vd['interaction_frac']*100:.1f}%")

    # The raw likelihood is dominated by the per-candidate term, so PMI is the
    # headline and raw is reported beside it. See PROCEDURE.md section 2B.
    for base, S in [("score_cond", S_cond), ("score_full", S_full)]:
        for how in ["raw", "pmi", "pmi_null", "zcol"]:
            if how == "pmi_null" and base != "score_cond":
                continue                      # null context only defines the conditional
            Mx = EV.normalize(S, how, null=null)
            s = EV.evaluate(Mx, f"{base}_{how}{args.suffix}", OUT, higher_is_better=True,
                            extra={"model": "PROK", "scorer": "scoring",
                                   "framing": "+#<CDS>+$<upstream>",
                                   "normalization": how,
                                   "quantity": "log P(upstream | CDS)" if "cond" in base
                                               else "log P(whole sequence)"},
                            save_matrix=(how == "raw"))
            EV.report(s)


# ------------------------------------------------------------------ embedding
def run_embed(df):
    import torch

    n = len(df)
    adapter = build()
    check_tokenizer(adapter)
    dev = adapter.device
    model = adapter.model

    # final hidden state. Try the clean route, then a hook on the last norm,
    # then the identity-unembed trick used for evo. Whichever works is logged.
    route = None
    try:
        ids = adapter.tok("ACGT", return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        with torch.inference_mode():
            o = model(ids, output_hidden_states=True)
        if getattr(o, "hidden_states", None):
            route = "output_hidden_states"
    except Exception as e:                                    # noqa: BLE001
        log(f"  output_hidden_states unavailable ({type(e).__name__})")

    cap = {}
    if route is None:
        target = None
        for path in ("backbone.norm", "backbone.ln_f", "model.norm"):
            m_ = model
            try:
                for p in path.split("."):
                    m_ = getattr(m_, p)
                target = m_
                break
            except AttributeError:
                continue
        assert target is not None, (
            "cannot locate the final norm; inspect the model and add its path here")
        target.register_forward_hook(lambda mod, inp, out: cap.__setitem__("h", out))
        route = f"hook:{path}"
    log(f"  hidden-state route: {route}")

    @torch.inference_mode()
    def embed(seq):
        ids = adapter.tok(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        out = model(ids, output_hidden_states=True) if route == "output_hidden_states" \
            else model(ids)
        h = out.hidden_states[-1] if route == "output_hidden_states" else cap["h"]
        if isinstance(h, tuple):
            h = h[0]
        v = h.float().mean(dim=1).squeeze(0)          # mean-pool, prompt tokens included
        return (v / v.norm().clamp_min(1e-8)).cpu().numpy()

    # GUARD 4 -- identical input must give cosine 1
    a = embed(candidate_embed_seq(df.up200.iloc[0]))
    b = embed(candidate_embed_seq(df.up200.iloc[0]))
    assert float(a @ b) > 0.999, f"embedding is not deterministic: cos={a @ b:.4f}"
    log(f"  guard 4 OK: identical sequences give cos={a @ b:.5f}, dim={a.shape[0]}")

    def rehook():
        """Re-attach the hidden-state hook after a model rebuild."""
        if not route.startswith("hook:"):
            return
        m_ = model
        for p in route[5:].split("."):
            m_ = getattr(m_, p)
        m_.register_forward_hook(lambda mod, inp, o: cap.__setitem__("h", o))

    def embed_all(seqs, label):
        """StripedHyena caches an FFT filter per distinct fft_size and never frees
        it -- 500 different CDS lengths OOM'd the 95 GiB GH200 at ~90 sequences.
        Rebuilding the model is the only reliable release (same fix as
        cds_generate.py). Only the queries vary in length; the 500 candidates are
        all 202 tokens and share one cache entry, so they never need a rebuild."""
        nonlocal adapter, model
        out = []
        for k, s in enumerate(seqs):
            if k and args.embed_reset and k % args.embed_reset == 0:
                adapter.model = None
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                adapter = build()
                model = adapter.model
                rehook()
                log(f"    [{label}] rebuilt at {k}/{len(seqs)}, "
                    f"gpu {torch.cuda.memory_allocated()/2**30:.1f} GiB")
            out.append(embed(s))
        return np.stack(out)

    Q = embed_all([query_prompt(c) for c in df.cds_seq], "query")
    log(f"  {n} query embeddings done")
    C = embed_all([candidate_embed_seq(u) for u in df.up200], "cand")
    log(f"  {n} candidate embeddings done")
    np.save(f"{OUT}/embed_query{args.suffix}.npy", Q.astype(np.float32))
    np.save(f"{OUT}/embed_cand{args.suffix}.npy", C.astype(np.float32))

    raw = Q @ C.T
    # anisotropy fix: autoregressive DNA LMs pack everything into a narrow cone
    Qc = Q - Q.mean(0, keepdims=True)
    Cc = C - C.mean(0, keepdims=True)
    Qc /= np.linalg.norm(Qc, axis=1, keepdims=True).clip(1e-8)
    Cc /= np.linalg.norm(Cc, axis=1, keepdims=True).clip(1e-8)
    cen = Qc @ Cc.T

    log(f"  true-pair cosine: raw {np.diag(raw).mean():.4f} | centered {np.diag(cen).mean():.4f}")
    if np.diag(raw).mean() > 0.99:
        log("  (raw cosines are saturated -- anisotropy, exactly as evo-1 showed; "
            "the centered numbers are the ones to read)")

    for name, S in [("embed_cos", raw), ("embed_cos_centered", cen)]:
        s = EV.evaluate(S, name + args.suffix, OUT, higher_is_better=True,
                        extra={"model": "PROK", "scorer": "embedding",
                               "framing": "query '+#<CDS>+$' vs candidate '+$<upstream>'",
                               "true_cos_mean": float(np.diag(S).mean())})
        EV.report(s)


# ------------------------------------------------------------------ main
df = EV.load_eval(limit=args.limit)
log(f"PROK retrieval | mode={args.mode} | n={len(df)} genes | L={EV.UP_LEN} | "
    f"{EV.N_CAND} candidates x {EV.N_REPEATS} repeats | seed {EV.SEED}")
EV.decoy_draws(len(df))     # materialise the shared draws before the GPU work
log(f"  decoy draws ready (shared by every model and both scorers)")

if args.mode == "score":
    run_score(df)
else:
    run_embed(df)

log("done")

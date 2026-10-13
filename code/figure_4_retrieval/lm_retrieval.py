#!/usr/bin/env python3
"""
Task 4 -- CDS <-> upstream retrieval for the autoregressive DNA LMs.
E. coli, L=200, 500 genes, 1-in-100 x 20 repeats. See PROCEDURE.md.

  --model prok       +#<CDS>+$<upstream>   score the upstream -> log P(up | CDS)
  --model evo1       <upstream><CDS>       score the CDS      -> log P(CDS | up)
  --model hyenadna   <upstream><CDS>       score the CDS      -> log P(CDS | up)

Each model gets the sequence arrangement it was TRAINED on. PROK is CDS-first
with segment tokens; the others are genomic order, bare DNA. Feeding a model an
arrangement it never saw returns finite, plausible, meaningless numbers.

WHY THE HEADLINE METRIC IS PMI, NOT RAW LIKELIHOOD
On the PROK smoke test 98.1% of the variance in log P(up_j | CDS_i) was a
per-CANDIDATE term -- "how intrinsically likely is this upstream" -- which is
identical for every query, so ranking on it puts the true pair at chance BY
CONSTRUCTION. Only 1.8% was the interaction that actually knows about the CDS.
Removing the column mean exposes it (median rank 49 -> 34, true>decoy 65%,
p=5e-06). Column-centering is the right fix for every model here: whatever makes
a candidate generically favoured is removed, leaving the query-specific part.

STRIPEDHYENA MEMORY (prok, evo1)
The engine caches an FFT filter per distinct fft_size and never frees it. A
count-based rebuild is not enough -- 40 sequences of varying length accumulated
87 GiB and OOM'd a 95 GiB GH200. The reset is therefore MEMORY-TRIGGERED: check
allocated GPU memory before each forward and rebuild past a threshold.

Outputs: results/<model>/  score_{cond,full}_matrix.npy, embed_*.npy,
         *_ranks.parquet, *_summary.json, and STATUS lines to the log.
"""
from __future__ import annotations

import argparse
import gc
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

# framing per model: does the CANDIDATE come first, and what wraps each segment
CFG = {
    "prok":     dict(sp="+", cds_first=True),    # +#<CDS>+$<up>, score the upstream
    "evo1":     dict(sp=None, cds_first=False),  # <up><CDS>,     score the CDS
    "hyenadna": dict(sp=None, cds_first=False),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(CFG), required=True)
ap.add_argument("--mode", choices=["score", "embed"], required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget", type=int, default=3000, help="tokens per forward")
ap.add_argument("--mem_gb", type=float, default=55.0,
                help="rebuild past this GPU GiB. Steady-state working set is ~49 GiB "
                     "and OOM hits ~85, so 30 thrashes and 70 never fires in time.")
ap.add_argument("--max_len", type=int, default=4300,
                help="skip query rows longer than this. StripedHyena needs "
                     ">82 GiB at L~4767 even at batch 1 on a clean allocator; "
                     "longest row that ever worked was L=4229. Those genes are "
                     "dropped as QUERIES but kept as candidates.")
ap.add_argument("--long_seq", type=int, default=3000,
                help="force width 1 + a clean allocator for rows longer than this")
ap.add_argument("--suffix", type=str, default="")
args = ap.parse_args()

CF = CFG[args.model]
OUT = RETR / "results" / args.model
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


def status(m):
    """One line per milestone, greppable by status.sh."""
    line = f"STATUS\t{args.model}\t{args.mode}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{m}"
    print(line, flush=True)
    with open(f"{RETR}/results/progress.tsv", "a") as fh:
        fh.write(line[7:] + "\n")


# ------------------------------------------------------------------ framing
def prompt_for(cds, up):
    """(text before the scored segment, the scored segment)."""
    if CF["cds_first"]:
        sp = CF["sp"]
        return f"{sp}#{cds}{sp}$", up          # score the upstream given the CDS
    return up, cds                             # score the CDS given the upstream


def embed_query_seq(cds):
    return f"{CF['sp']}#{cds}{CF['sp']}$" if CF["cds_first"] else cds


def embed_cand_seq(up):
    return f"{CF['sp']}${up}" if CF["cds_first"] else up


# ------------------------------------------------------------------ model
def build():
    return M.build(args.model).load()


def check_tokenizer(ad):
    chars = ["A", "C", "G", "T"] + ([CF["sp"], "#", "$"] if CF["cds_first"] else [])
    for ch in chars:
        t = ad.tok(ch, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, f"'{ch}' is not one token: {t}"
    probe = "+#ACGT+$ACGT" if CF["cds_first"] else "ACGTACGTAC"
    n = len(ad.tok(probe, add_special_tokens=False)["input_ids"])
    assert n == len(probe), f"not 1 token/char: {probe} -> {n}"
    log(f"  guard 1 OK: 1 token per character ({' '.join(chars)})")


# ------------------------------------------------------------------ shared GPU state
class Runner:
    def __init__(self):
        import torch
        self.torch = torch
        # Every distinct sequence length builds a new cuFFT plan with its own
        # workspace, and the default cache holds 4096 of them -> unbounded growth
        # across 500 differently-sized queries. Cap it.
        try:
            torch.backends.cuda.cufft_plan_cache.max_size = 8
        except Exception:                                        # noqa: BLE001
            pass
        self.ad = build()
        check_tokenizer(self.ad)
        self.dev = self.ad.device
        self.rebuilds = 0

    def gpu_gb(self):
        """DEVICE memory in use, not just PyTorch's.

        memory_allocated() reported 12 GiB while the process actually held
        85 GiB and OOM'd: the leak is the cuFFT PLAN CACHE, which is non-PyTorch
        memory and invisible to memory_allocated(). mem_get_info() sees it.
        """
        free, total = self.torch.cuda.mem_get_info()
        return (total - free) / 2 ** 30

    def rebuild(self, why=""):
        self.ad.model = None
        gc.collect(); gc.collect()          # break reference cycles first
        try:                                # drop cuFFT plans too -- see gpu_gb
            self.torch.backends.cuda.cufft_plan_cache.clear()
        except Exception:                                        # noqa: BLE001
            pass
        self.torch.cuda.empty_cache()       # only now is the memory returned
        self.ad = build()
        self.rebuilds += 1
        log(f"    rebuilt #{self.rebuilds} ({why}), gpu now {self.gpu_gb():.1f} GiB")

    def maybe_reset(self):
        """MEMORY-triggered, not count-triggered -- see module docstring."""
        if self.gpu_gb() > args.mem_gb:
            self.rebuild(f"{self.gpu_gb():.0f} GiB > {args.mem_gb:.0f}")

    def logprobs(self, seqs, n_prompt):
        """(mean logprob of the scored segment, mean logprob of the whole seq)."""
        torch = self.torch
        with torch.inference_mode():
            ids = self.ad.tok(list(seqs), return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(self.dev)
            assert ids.shape[1] == len(seqs[0]), \
                f"tokenized {ids.shape[1]} != chars {len(seqs[0])}"
            logits = self.ad.model(ids).logits.float()
            lp = torch.log_softmax(logits, -1)
            tok_lp = lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # tokens [n_prompt:] are predicted by columns [n_prompt-1:]
            cond = tok_lp[:, n_prompt - 1:].mean(1)
            full = tok_lp.mean(1)
            out = cond.float().cpu().numpy(), full.float().cpu().numpy()
            del logits, lp, tok_lp, ids
        return out

    def safe_logprobs(self, seqs, n_prompt):
        """OOM is handled OUTSIDE the except block on purpose.

        An active `except` keeps the traceback alive, the traceback holds the
        failing frame, and the frame holds the very tensors we need to free --
        so empty_cache() frees nothing and the rebuild itself then OOMs at
        ~95 GiB. Setting a flag and handling it after the block drops the
        traceback first. Same trick as cds_generate.py.
        """
        self.maybe_reset()
        oom = False
        try:
            return self.logprobs(seqs, n_prompt)
        except self.torch.OutOfMemoryError:
            oom = True
        if oom:
            self.rebuild("OOM")
            cs, fs = [], []
            for one in seqs:                       # degrade to one at a time
                bad = False
                try:
                    c, f = self.logprobs([one], n_prompt)
                except self.torch.OutOfMemoryError:
                    bad = True
                if bad:
                    self.rebuild("OOM-single")
                    c, f = self.logprobs([one], n_prompt)
                cs.append(c); fs.append(f)
            return np.concatenate(cs), np.concatenate(fs)


# ------------------------------------------------------------------ scoring
def run_score(df):
    n = len(df)
    cds, up = df.cds_seq.tolist(), df.up200.tolist()
    p_cond = f"{OUT}/score_cond{args.suffix}_matrix.npy"
    p_full = f"{OUT}/score_full{args.suffix}_matrix.npy"
    p_done = f"{OUT}/score{args.suffix}_done.npy"

    S_c = np.load(p_cond) if os.path.exists(p_cond) else np.full((n, n), np.nan, np.float32)
    S_f = np.load(p_full) if os.path.exists(p_full) else np.full((n, n), np.nan, np.float32)
    done = np.load(p_done) if os.path.exists(p_done) else np.zeros(n, bool)
    if done.any():
        status(f"resume {int(done.sum())}/{n} rows already done")

    R = Runner()
    todo = [i for i in range(n) if not done[i]]
    status(f"scoring {len(todo)} of {n} query rows")

    for pos, i in enumerate(todo):
        if CF["cds_first"]:
            pre, _ = prompt_for(cds[i], up[i])
            seqs = [pre + u for u in up]           # candidate varies at the tail
            n_prompt = len(pre)
        else:
            seqs = [u + cds[i] for u in up]        # candidate varies at the head
            n_prompt = EV.UP_LEN
        L = len(seqs[0])
        width = max(1, args.budget // max(L, 1))
        # A long row needs a CLEAN allocator, not merely a narrow batch: the
        # FFT prefill transient for ~7k tokens is 9-20 GiB and will not fit on
        # top of a 49 GiB working set at any batch size. Same guard as
        # cds_generate.py (LONG_SEQ). The two longest CDSs in the eval set
        # (7020 and 4029 bp) both land in the tail rows -- which is why this
        # only bit at the very end of the run.
        if L > args.max_len:
            # Unscorable at any batch size. Record NaN and continue rather
            # than killing the run; excluded as a query, kept as a candidate.
            S_c[i], S_f[i], done[i] = np.nan, np.nan, True
            status(f"SKIP row {i} (L={L} > {args.max_len}) -- unscorable")
            continue
        if L > args.long_seq:
            width = 1
            R.rebuild(f"long row L={L}")
        c_row, f_row = np.empty(n, np.float32), np.empty(n, np.float32)
        for s in range(0, n, width):
            c, f = R.safe_logprobs(seqs[s:s + width], n_prompt)
            c_row[s:s + len(c)] = c
            f_row[s:s + len(f)] = f
        S_c[i], S_f[i], done[i] = c_row, f_row, True

        if pos < 2 or (pos + 1) % 10 == 0:
            el = time.time() - T0
            rate = (pos + 1) / el
            log(f"  row {pos+1}/{len(todo)} (L={L}, w={width}, gpu {R.gpu_gb():.0f}G) "
                f"| {rate*3600:.0f} rows/h | ETA {(len(todo)-pos-1)/max(rate,1e-9)/3600:.1f} h")
        if (pos + 1) % 10 == 0 or pos == len(todo) - 1:
            np.save(p_cond, S_c); np.save(p_full, S_f); np.save(p_done, done)
            if (pos + 1) % 50 == 0:
                status(f"progress {int(done.sum())}/{n} rows")

    np.save(p_cond, S_c); np.save(p_full, S_f); np.save(p_done, done)
    nbad = int(np.isnan(S_c).any(axis=1).sum())
    assert nbad < 0.05 * n, f"{nbad}/{n} query rows unscorable -- too many"
    if nbad:
        status(f"{nbad}/{n} query rows skipped as unscorable (see SKIP lines)")
    status(f"scoring complete {n}/{n} rows, {R.rebuilds} rebuilds")

    # GUARD 3 -- is the matrix diagonal really the true pair?
    #
    # NON-FATAL by design. The scoring is already done and saved at this point;
    # aborting here would throw away a 13-hour run over a diagnostic. The result
    # is recorded in the summary instead, so a suspect run is visible rather than
    # lost.
    #
    # Tolerance is scaled to the row's own spread. Re-scoring happens at batch
    # width 1 while the row was scored at width w, and bf16 batch-vs-single drift
    # is real; a genuine INDEX MISALIGNMENT would instead be off by roughly the
    # spread of the row. (An earlier version used argmin|row - c|, which is
    # meaningless when 500 scores are tightly clustered -- it picks a near-tie.)
    guard3 = {"checked": 0, "max_drift_vs_rowstd": 0.0, "suspect": False}
    for i in [k for k in range(n) if not np.isnan(S_c[k]).any()][:3]:
        if CF["cds_first"]:
            pre, seg = prompt_for(cds[i], up[i]); s1 = pre + seg
        else:
            pre = up[i]; s1 = up[i] + cds[i]
        c, _ = R.safe_logprobs([s1], len(pre))
        row = S_c[i]
        drift = abs(float(c[0] - row[i]))
        rel = drift / max(float(row.std()), 1e-9)
        guard3["checked"] += 1
        guard3["max_drift_vs_rowstd"] = max(guard3["max_drift_vs_rowstd"], rel)
        if rel > 0.5:
            guard3["suspect"] = True
            log(f"  !! GUARD 3 SUSPECT gene {i}: |recomputed - stored| = {drift:.4f} "
                f"= {rel:.2f}x the row sd -- possible index misalignment")
        elif drift > 1e-3:
            log(f"  guard 3 gene {i}: drift {drift:.2e} ({rel:.3f}x row sd) -- bf16, fine")
    log(f"  guard 3 {'SUSPECT' if guard3['suspect'] else 'OK'}: "
        f"max {guard3['max_drift_vs_rowstd']:.3f}x row sd over {guard3['checked']} genes")

    # null-context scores, only meaningful when the CANDIDATE is the scored part
    null = None
    if CF["cds_first"]:
        p_null = f"{OUT}/score_null{args.suffix}.npy"
        if os.path.exists(p_null):
            null = np.load(p_null)
        else:
            pre = f"{CF['sp']}$"
            null = np.empty(n, np.float32)
            for s in range(0, n, 16):
                c, _ = R.safe_logprobs([pre + u for u in up[s:s + 16]], len(pre))
                null[s:s + len(c)] = c
            np.save(p_null, null)
        log(f"  null-context mean {null.mean():.4f}")

    vd = EV.variance_decomposition(S_c)
    log(f"  variance: candidate {vd['between_candidate_frac']*100:.1f}% | "
        f"query {vd['between_query_frac']*100:.1f}% | "
        f"interaction {vd['interaction_frac']*100:.1f}%")

    for base, S in [("score_cond", S_c), ("score_full", S_f)]:
        for how in ["raw", "pmi", "zcol"] + (["pmi_null"] if null is not None and base == "score_cond" else []):
            s = EV.evaluate(EV.normalize(S, how, null=null), f"{base}_{how}{args.suffix}",
                            OUT, True, extra={"model": args.model, "scorer": "scoring",
                                              "normalization": how}, save_matrix=False)
            EV.report(s)
            if how == "pmi" and base == "score_cond":
                status(f"RESULT score pmi top1={s['top1']*100:.2f}% "
                       f"median={s['median_rank']:.0f} "
                       f"gap_frac={s['guard_true_vs_decoy']['frac_true_higher']*100:.1f}%")


# ------------------------------------------------------------------ embedding
def run_embed(df):
    n = len(df)
    R = Runner()
    torch = R.torch
    route, cap = None, {}

    def attach():
        nonlocal route
        m = R.ad.model
        if route is None:
            try:
                ids = R.ad.tok("ACGT", return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(R.dev)
                with torch.inference_mode():
                    o = m(ids, output_hidden_states=True)
                if getattr(o, "hidden_states", None):
                    route = "output_hidden_states"
                    return
            except Exception:                                    # noqa: BLE001
                pass
            for p in ("backbone.norm", "backbone.ln_f", "model.norm", "norm"):
                t = m
                try:
                    for a in p.split("."):
                        t = getattr(t, a)
                except AttributeError:
                    continue
                route = f"hook:{p}"
                break
            assert route, "cannot find the final norm -- add its path"
        if route.startswith("hook:"):
            t = m
            for a in route[5:].split("."):
                t = getattr(t, a)
            t.register_forward_hook(lambda mod, i_, o: cap.__setitem__("h", o))

    attach()
    log(f"  hidden-state route: {route}")

    def embed(seq):
        R.maybe_reset_hook = None
        if R.gpu_gb() > args.mem_gb:
            R.rebuild(f"{R.gpu_gb():.0f} GiB")
            attach()
        with torch.inference_mode():
            ids = R.ad.tok(seq, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to(R.dev)
            o = R.ad.model(ids, output_hidden_states=True) if route == "output_hidden_states" \
                else R.ad.model(ids)
            h = o.hidden_states[-1] if route == "output_hidden_states" else cap["h"]
            if isinstance(h, tuple):
                h = h[0]
            v = h.float().mean(1).squeeze(0)
            v = (v / v.norm().clamp_min(1e-8)).cpu().numpy()
        return v

    a = embed(embed_cand_seq(df.up200.iloc[0]))
    b = embed(embed_cand_seq(df.up200.iloc[0]))
    assert float(a @ b) > 0.999, f"embedding not deterministic: {a @ b:.4f}"
    log(f"  guard 4 OK: cos={a @ b:.5f}, dim={a.shape[0]}")

    pq, pc = f"{OUT}/embed_query{args.suffix}.npy", f"{OUT}/embed_cand{args.suffix}.npy"
    if os.path.exists(pq) and os.path.exists(pc):
        Q, C = np.load(pq), np.load(pc)
        status("embeddings loaded from cache")
    else:
        Q = np.stack([embed(embed_query_seq(c)) for c in df.cds_seq])
        np.save(pq, Q.astype(np.float32)); status(f"{n} query embeddings done")
        C = np.stack([embed(embed_cand_seq(u)) for u in df.up200])
        np.save(pc, C.astype(np.float32)); status(f"{n} candidate embeddings done")

    raw = Q @ C.T
    Qc = Q - Q.mean(0, keepdims=True); Cc = C - C.mean(0, keepdims=True)
    Qc /= np.linalg.norm(Qc, axis=1, keepdims=True).clip(1e-8)
    Cc /= np.linalg.norm(Cc, axis=1, keepdims=True).clip(1e-8)
    cen = Qc @ Cc.T
    log(f"  true-pair cosine raw {np.diag(raw).mean():.4f} | centered {np.diag(cen).mean():.4f}")

    for name, S in [("embed_cos", raw), ("embed_cos_centered", cen)]:
        s = EV.evaluate(S, name + args.suffix, OUT, True,
                        extra={"model": args.model, "scorer": "embedding",
                               "true_cos_mean": float(np.diag(S).mean())})
        EV.report(s)
        if name == "embed_cos_centered":
            status(f"RESULT embed centered top1={s['top1']*100:.2f}% "
                   f"median={s['median_rank']:.0f}")


df = EV.load_eval(limit=args.limit)
status(f"start n={len(df)} L={EV.UP_LEN} budget={args.budget} mem_gb={args.mem_gb}")
EV.decoy_draws(len(df))
run_score(df) if args.mode == "score" else run_embed(df)
status("done")

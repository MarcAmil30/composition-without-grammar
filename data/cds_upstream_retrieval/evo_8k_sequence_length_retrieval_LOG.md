# Experiment log — evo_8k_sequence_length_retrieval

**Date:** 2026-07-18/19 · **Platform:** Isambard-AI Phase 2 (GH200)

## Goal
DNA-only analogue of the C3P Exp-3 upstream length sweep, using **evo-1-8k-base** mean-pooled
embeddings. Question: does evo's representation link a gene's upstream to the gene itself, and
does it depend on upstream length (300/200/100/50 bp)?

## Key design decision
C3P retrieval is *cross-modal* (protein ↔ upstream in a trained shared space). Evo-1-8k-base is a
**DNA-only** language model — no protein tower, never trained for cross-modal alignment or retrieval.
So both sides must be DNA:
- **Query** = full-length **CDS** (coding DNA), embedded once (length-independent).
- **Candidates** = the gene's own upstream + 99 random same-species upstreams, embedded at
  upstream length L ∈ {300,200,100,50} bp (keep last L bp, `s[-L:]`, proximal to start codon).
- 1-in-100 retrieval, rank by cosine(CDS_i, upstream_j), **n_repeats=20**, seed=0, RNG reset per
  length so gene i / repeat r faces the same 99 decoys at every length (paired comparison).

## Data
- Same **1,404 reference singletons** as the C3P run (727 E. coli + 677 B. subtilis) from
  `results/step0/analysis_set.parquet`, ascending row order per species.
- `cds_seq` left-joined from `data/ref2/ref2_flanks.csv` on (accession, locus_tag).
  **Verified: 1404/1404 matched, 0 nulls, cds len == cds_len, upstream identical.**
  CDS median 835 bp, max 7020 bp — all within evo's 8k context.

## Method / embeddings
- evo-1-8k-base final-layer hidden state via identity-`unembed` trick (from `evo/test_evo_base.py`),
  **mean-pooled over position** (`embed.mean(dim=1)`, 4096-d), L2-normalised, batch=1 (exact pooling).
- Runs in NGC container `pytorch_25.05-py3.sif` + `ngc_evo_venv`, HF offline cache, bf16, 1×GH200.

## Files
- Script: `evo/evo_8k_length_sweep.py` · sbatch: `run_evo_8k_length.slurm`
- Centered re-run: `evo_8k_length_centered.py` (numpy-only, reuses cached embeddings)
- Outputs: `results/evo_8k_length/` (raw) and `results/evo_8k_length/centered/`
  - `*_emb_*.npy` cached embeddings, `evo8k_length_ranks*.parquet`, `*summary*.json`,
    `summary_table*.csv`, `violin_<species>*.png`

## Run record
- Preflight (5 genes/sp) job 5704176 — cancelled after ~60 min queue (deps + join + container all
  independently verified, so serialising two GPU-queue waits wasn't worth it).
- **Full run job 5704909 — COMPLETED, 9m24s, exit 0.** Embedding ~18 s per 700 seqs (no recompile blowup).
- Incident: Isambard SSH certificate (12 h) expired at 19:20 during the long queue wait; renewed with
  `clifton`; results were already saved on scratch, nothing lost.
- Centered re-run: login-node numpy, ~20 s.

## RESULTS

### Raw (mean-pooled + cosine)
| species | L | top1 | median | top5 | trueCos |
|---|---|---|---|---|---|
| E. coli | 300/200/100/50 | 1.6 / 1.7 / 2.2 / 1.3 % | 47/45/43/45 | ~7 % | **0.999** |
| B. subtilis | 300/200/100/50 | 1.4 / 1.5 / 1.8 / 1.1 % | 39/39/45/46 | ~7 % | **0.999** |

Chance = top1 1 %, median 50, top5 5 %. → **at chance, no length effect.**
`trueCos ≈ 0.999` = severe **anisotropy** (all embeddings in a narrow cone → cosine can't discriminate).

### Centered (subtract per-set mean, then cosine) — anisotropy fix
| species | L | top1 | median | top5 | trueCos |
|---|---|---|---|---|---|
| E. coli | 300/200/100/50 | 1.4 / 2.0 / 1.7 / 1.1 % | 46/44/42/44 | ~6–7 % | 0.05–0.07 |
| B. subtilis | 300/200/100/50 | 2.0 / 1.4 / 1.1 / 1.1 % | 37/38/43/44 | 6–8 % | 0.07–0.14 |

Centering spread the cosines (0.999 → ~0.05–0.14) but **retrieval stayed at chance.** Only a faint
low-rank bulge for B. subtilis at 300/200 (median ~37–38, top5 ~8 % vs 5 %) — weak, not a clean trend.

## Conclusion
**evo-1-8k-base mean-pooled embeddings do not link a CDS to its cognate upstream above chance, at any
upstream length** — and this is not a cosine/anisotropy artifact (it survives centering). Expected:
evo is a generative DNA LM (next-token prediction), never trained to align a coding region with its
promoter. This is exactly the alignment C3P was *contrastively trained* for (top-1 86 % / 77 %), so the
contrast highlights the value of task-specific contrastive training over a general genomic LM for this
retrieval task.

## Possible next steps (not run)
- last-token pooling instead of mean; specific layer instead of final; whitening instead of centering.
- Change the question to one evo *is* trained for (e.g. likelihood/perplexity of upstream given context,
  or generation), rather than cross-region cosine retrieval.

# Figure 2d long-context experiment status

Last updated: 2026-08-20 10:48 (Europe/London)

## Preparation

- Experimental plan: `EXPERIMENT_FIG2D_LONG_CONTEXT.md`
- Full 680-gene prompt ladder built and hash-validated at 200, 300, 500, 1k, 2k, 5k, 8k, 32k, 100k, 131k, 250k, 449k and 999k bp.
- The same documented `b4522` reference/GFF mismatch excluded from the published context ladder remains excluded; no new genes failed prompt validation.
- Evo 2–7B checkpoint staged in the Isambard cache.

## Maximum-context smoke tests

| Checkpoint | Context | Job | Status/result |
|---|---:|---:|---|
| Evo 2 1B base | 8k | 6057992 | Passed (58 s) |
| HyenaDNA medium-450k | 449k | 6057994 | Passed (3 min 2 s) |
| Evo 1 8k | 8k | 6057991 | OOM at batch size 1 on 95 GiB usable HBM |
| Evo 1 131k | 131k | 6057993 | OOM at batch size 1 on 95 GiB usable HBM |
| Evo 1 8k practical-limit retry | 5k | 6058564 | OOM at batch size 1; stepped down to 4k |
| Evo 1 131k practical-limit retry | 32k | 6058565 | OOM at batch size 1; stepped down to 16k |
| Evo 2 7B long | 999k | 6058566 | Offline cache-path lookup failure; checkpoint itself was present |
| Evo 1 8k practical-limit retry | 4k | 6060097 | Failed before model load due global cache-path override; superseded by 6062219 |
| Evo 1 131k practical-limit retry | 16k | 6060098 | Failed before model load due global cache-path override; superseded by 6062220 |
| Evo 2 7B long retry | 999k | 6060099 | Checkpoint loaded, then genuine OOM on one GH200; stepped down to 250k |
| Evo 1 8k cache-corrected retry | 4k | 6062219 | Passed; one-gene/two-sample generation completed in 54 s |
| Evo 1 131k cache-corrected retry | 16k | 6062220 | Checkpoint loaded, then genuine CUDA OOM (31.25 GiB allocation with 27.14 GiB free); step down to 8k requested, submission confirmation pending |
| Evo 2 7B practical-limit retry | 250k | 6062221 | Checkpoint loaded, then exceeded PyTorch 32-bit tensor-indexing limit; step down to 100k requested, submission confirmation pending |
| Evo 1 131k practical-limit retry | 8k | 6067385 | one-gene/two-sample smoke test submitted; full 8k curve is gated on success |

## Production generation arrays

| Parent job | Checkpoint | Context | Block/concurrency | Current status |
|---:|---|---:|---|---|
| 6067307 | Evo 1 8k | 4k practical endpoint | 8 genes / max 4 | full 680-gene array submitted; pending for priority |
| 6067386 | Evo 1 131k | 200, 300, 500, 1k, then 8k | staged launcher | waits for job 6067307 to release quota; submits four 680-gene short-context arrays, then submits the 8k array only if job 6067385 passes |
| 6058259 | Evo 2 1B | 2k | 32 genes / max 4 | all 22 blocks complete |
| 6058260 | Evo 2 1B | 5k | 32 / 4 | all 22 blocks complete |
| 6058261 | Evo 2 1B | 8k | 32 / 4 | all 22 blocks complete |
| 6058262 | HyenaDNA | 2k | 32 / 4 | all 22 blocks complete |
| 6058263 | HyenaDNA | 5k | 32 / 4 | all 22 blocks complete |
| 6058264 | HyenaDNA | 8k | 32 / 4 | all 22 blocks complete |
| 6058265 | HyenaDNA | 32k | 16 / 4 | all 43 blocks complete |
| 6058266 | HyenaDNA | 100k | 8 / 4 | all 85 blocks complete |
| 6058267 | HyenaDNA | 250k fixed-128 fallback | 4 / 4 | required blocks 0–31 complete; 48 completed blocks preserved in total; remaining beyond-subset tasks stopped |
| 6058529 | deferred launcher | HyenaDNA 449k | 4 / 4 | launcher ran after dependencies but full 170-task submission exceeded QOS job-count limit |
| 6062222 | HyenaDNA | 449k fixed-128 fallback | 4 / 4 | 12/32 blocks complete, four running and 16 queued at last scheduler check |

No production generation task has failed. At 15:56, 308 completed compressed blocks and their 308 manifests passed gzip integrity/count checks. Eight full 680-gene endpoints are complete: Evo 2 and HyenaDNA at 2k, 5k and 8k, plus HyenaDNA at 32k and 100k. The fixed first-128-gene 250k endpoint is also complete and scored; all 48 blocks produced before stopping the remaining beyond-subset work are preserved. The fixed-128-gene 449k fallback is progressing. Initial probability tasks failed before model load because a cache override intended only for Evo 2–7B was applied globally; the corrected arrays produced every full block but seven final partial blocks were missed because their source filenames end at gene 680 rather than the nominal block boundary. Only those seven tails were resubmitted as jobs 6065021–6065027. All production outputs are block-atomic and accompanied by prompt/output hashes and Slurm provenance.

## Completed downstream analyses

| Endpoint | Scoring result | Probability extraction |
|---|---|---|
| HyenaDNA, 2k | 680 genes / 14,280 generations; 11.870% within ±2 bp (95% gene-bootstrap interval 11.345–12.374%); shuffled control 12.360% | 21/22 blocks complete; exact final tail job 6065021 queued |
| HyenaDNA, 5k | 680 genes / 14,280 generations; 11.380% within ±2 bp (95% interval 10.868–11.919%); shuffled control 12.269% | 21/22 blocks complete; exact final tail job 6065022 queued |
| HyenaDNA, 8k | 680 genes / 14,280 generations; 10.924% within ±2 bp (95% interval 10.406–11.450%); shuffled control 11.618% | 21/22 blocks complete; exact final tail job 6065023 queued |
| Evo 2 1B, 2k | 680 genes / 14,280 generations; 15.532% within ±2 bp (95% interval 14.643–16.457%); shuffled control 12.339% | 21/22 blocks complete; exact final tail job 6065024 queued |
| Evo 2 1B, 5k | 680 genes / 14,280 generations; 15.630% within ±2 bp (95% interval 14.713–16.639%); shuffled control 12.297% | 21/22 blocks complete; exact final tail job 6065025 queued |
| Evo 2 1B, 8k | 680 genes / 14,280 generations; 16.099% within ±2 bp (95% interval 15.203–17.080%); shuffled control 12.255% | 21/22 blocks complete; exact final tail job 6065026 queued |
| HyenaDNA, 32k | 680 genes / 14,280 generations; 11.863% within ±2 bp (95% interval 11.373–12.367%); shuffled control 11.786% | 42/43 blocks complete; exact final tail job 6065027 queued |
| HyenaDNA, 100k | 680 genes / 14,280 generations; 11.513% within ±2 bp (95% interval 11.022–11.996%); shuffled control 12.276% | job 6064996, 85 blocks / max 4, queued |
| HyenaDNA, 250k fixed subset | 128 genes / 2,688 generations; 10.900% within ±2 bp (95% interval 9.821–11.979%); shuffled control 13.021% | all 32 blocks complete |

HyenaDNA declines from 11.870% at 2k to 10.924% at 8k, returns to chance at 32k (11.863% versus 11.786% control), and remains below control at 100k (11.513% versus 12.276%) and on the matched 250k subset (10.900% versus 13.021%). Evo 2 changes only modestly from 15.532% at 2k to 16.099% at 8k and remains above control. These are interim findings; the 449k endpoint, maximum practical checkpoint probes and probability profiles are still running.

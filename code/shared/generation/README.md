# Generation driver — upstream-only benchmark

Generates and writes to disk. **No scoring happens here**, so the PWMs, Promoter
Calculator and RBS Calculator can be built and revised afterwards without
re-running a single model.

```
generation/
  framing.py           per-model prompt framing (PROK '+$', PromoGen2 reframe)
  models.py            adapter registry + the one shared SamplingConfig
  generate.py          the driver
  run_generation.sbatch  one shard per job
```

## Run it

```bash
# validate framing — no GPU, no weights, safe anywhere
python generate.py --dry-run

# prove one adapter works: 2 genes, 2 samples
python generate.py --model promogen2 --task generate_sd --background leaked --smoke

# a real shard
sbatch --job-name=upgen_evo1_prom_leaked run_generation.sbatch evo1 generate_promoter leaked
```

Shards are resumable — an existing output CSV is skipped unless `--force`.

## Output

`generations/<model>/<task>__<background>.csv`

| column | |
|---|---|
| `locus_tag` | gene |
| `model`, `task`, `background` | which shard |
| `decode` | `greedy` or `sample` |
| `sample_idx` | −1 for greedy, 0–19 for samples |
| `prompt_len` | length of what the tokenizer actually received |
| `generation` | the continuation, uppercased, non-ACGT stripped |
| `gen_len` | its length (short = the model emitted EOS early) |
| `framing_note` | what framing was applied |

Alongside it, `<task>__<background>.manifest.json` records the checkpoint,
sampling config, framing notes and **every dropped gene with its reason**.

## Scale

| | |
|---|---|
| Shards | 5 models × 2 tasks × 2 backgrounds = **20** |
| Generations per shard | 14,301 (681 genes × 21) |
| **Total** | **284,382** |

PromoGen2's `generate_promoter` shards are 13,482 — 39 genes dropped.

## Sampling — identical for every model

Defined once in `models.SamplingConfig` and never overridden per model:

```
n_samples 20   temperature 1.0   top_k off   top_p off   + 1 greedy decode
```

Per-model differences live **only** in `framing.py`, and every one is recorded in
the output rows.

## Before submitting

Compute nodes have **no internet**. Every checkpoint must be in `$HF_HOME`
already — the sbatch sets `HF_HUB_OFFLINE=1` so a missing one fails loudly
instead of hanging.

**Everything the benchmark needs is already cached** — no downloads required:

| model | checkpoint |
|---|---|
| Evo 1 | `togethercomputer/evo-1-131k-base` |
| Evo 2 | `arcinstitute/evo2_1b_base` |
| PROK | local v3 run (`finetuned_evo_..._v3_TRAIN.csv_patience_2`) |
| HyenaDNA | `LongSafari/hyenadna-medium-450k-seqlen-hf` |
| PromoGen2 | `jinyuan22/promogen2-base` |

## One thing to confirm on the first smoke test

**Evo 2's API.** It loads through the `evo2` package rather than plain HF, and
the generate signature has moved between releases. `Evo2Adapter` is isolated for
exactly this reason — fix it there, nothing else is affected. The other four go
through `HFCausalAdapter` unchanged.

Every adapter asserts its tokenizer is single-nucleotide on load (10 bases → 10
tokens). If that fails, every metric offset would be meaningless, so it stops
rather than producing quiet garbage.

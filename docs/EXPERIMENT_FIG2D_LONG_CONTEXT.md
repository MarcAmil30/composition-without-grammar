# Improved Figure 2d: Long-context promoter-generation experiment

**Status:** Approved for implementation and execution on Isambard-AI  
**Date:** 2026-08-19  
**Primary question:** Does substantially increasing the natural genomic context given to a genomic language model improve its ability to place the *E. coli* sigma-70 -10 promoter box at the correct position?

## Motivation and interpretation

The current Figure 2d tests 200, 300, 500 and 1,000 bp of upstream context. These inputs are short relative to the advertised context capacities of several genomic language models. The improved experiment therefore extends the context ladder to each checkpoint's trained limit.

This is a **long-context stress test**, not simply a test of additional promoter sequence. The median *E. coli* intergenic region is about 203 bp, so most sequence added beyond approximately 200 bp comes from neighbouring genes. A positive result at long context could reflect broader genomic structure or memorisation rather than additional local promoter information. The experiment must retain controls that distinguish these explanations.

Checkpoint identity must remain explicit. A model must never silently change checkpoint along a single curve:

- `evo-1-8k-base` and `evo-1-131k-base` are separate Evo 1 checkpoints.
- `evo2_1b_base` is a base checkpoint trained at 8,192-bp context; it is not the 1M-context Evo 2 model.
- Long-context Evo 2 requires the distinct `evo2_7b` checkpoint.
- `hyenadna-medium-450k-seqlen-hf` stops at 450k; the separate 1M HyenaDNA checkpoint is not required initially because Evo 2 7B supplies the 1M test.

## Models and context lengths

Keep the existing context points and add logarithmically spaced lengths:

`200, 300, 500, 1k, 2k, 5k, 8k, 32k, 100k, 250k, 450k, 1M bp`

Each checkpoint stops at its trained context limit. The prompt length used in code must be slightly below the nominal limit when necessary to leave room for the 45 generated bases and any required special token.

| Curve/checkpoint | Nominal trained limit | Planned context points |
|---|---:|---|
| Evo 1, `togethercomputer/evo-1-8k-base` | 8,192 | 200, 300, 500, 1k, 2k, 5k, 8k |
| Evo 2, `arcinstitute/evo2_1b_base` | 8,192 | 200, 300, 500, 1k, 2k, 5k, 8k |
| HyenaDNA, `LongSafari/hyenadna-medium-450k-seqlen-hf` | 450,000 | 200, 300, 500, 1k, 2k, 5k, 8k, 32k, 100k, 250k, 450k |
| Evo 1 long, `togethercomputer/evo-1-131k-base` | 131,072 | 200, 300, 500, 1k, 2k, 5k, 8k, 32k, 100k, 131k |
| Evo 2 long, `arcinstitute/evo2_7b` | 1,000,000 | 200, 300, 500, 1k, 2k, 5k, 8k, 32k, 100k, 250k, 450k, 1M |

The existing three checkpoints remain the direct continuation of Figure 2d. The two long-context checkpoints are additional, separately labelled curves.

## Data and prompt construction

Use the same RegulonDB-annotated *E. coli* MG1655 sigma-70 promoter cohort as the current upstream-only benchmark.

For every target gene and context length:

1. Extract the natural genomic sequence immediately upstream of the generation boundary, ending immediately after the annotated -35 box.
2. Preserve the correct transcriptional orientation. Reverse-strand loci must be reverse-complemented.
3. Treat the MG1655 chromosome as circular when a requested context crosses the coordinate origin.
4. Verify that the final 200 bp reproduce the current 200-bp prompts exactly, aside from already documented exclusions.
5. Record the locus, strand, genomic coordinates, requested length, actual length and SHA-256 prompt hash.

The generation boundary and target do not change with context length. Only the amount of natural sequence preceding the -35 box changes.

## Generation task

For each checkpoint, gene and context length:

1. Supply the constructed natural genomic prompt.
2. Generate the following 45 bp, matching the existing sigma-70 promoter task.
3. Use 20 independent samples at temperature 1.0 plus one greedy decode, matching the existing adapters. Evo 1 and HyenaDNA sample from the full distribution (top-k disabled, top-p 1.0). The existing Vortex adapter for Evo 2 uses top-k 4; preserve and explicitly record this so the extended points remain comparable to the current Figure 2d.
4. Use deterministic, recorded seeds mapped consistently to gene and sample index.
5. Preserve non-ACGT output positionally as `N`; do not delete it and shift downstream positions.
6. Save every generation before scoring.

No generation or probability output may be overwritten. A completed shard is immutable; retries write a new attempt directory until validated and promoted.

## Primary and secondary measurements

### Primary endpoint

Scan each generation with the existing real-promoter sigma-70 -10 PWM. A generation is correct when its strongest predicted -10 box begins within +/-2 bp of that gene's annotated -10 position.

Report:

- percentage correct at every checkpoint x context length;
- 95% confidence intervals bootstrapped over genes, not individual generations;
- paired change from the same checkpoint's 200-bp condition;
- a paired gene-bootstrap confidence interval for that change.

Chance for the +/-2-bp endpoint is 12.5% under the existing 40-position search geometry.

### Secondary endpoints

- exact placement and tolerances of +/-1, +/-3 and +/-5 bp;
- signed and absolute position error;
- best PWM score;
- bases matching the gene's true -10 hexamer at its annotated position;
- composition-preserving shuffle of each generated sequence as a negative control;
- the existing real-promoter positive control.

### A/C/G/T probabilities

Retain the model's conditional probability of A, C, G and T at every one of the 45 generated positions for every gene and decode. Also store:

- A/C/G/T-normalised entropy at every position;
- total non-ACGT probability mass;
- the mean probability profile by checkpoint and context;
- the change in probability profile relative to 200 bp;
- a quantitative -10-region-versus-flank contrast, rather than relying only on a plot.

Probability extraction must follow the exact generated prefix at each step. Teacher forcing on the natural continuation is not a replacement for probabilities from free-running generation.

## Long-range controls

All primary runs use intact natural context. If any long-context point has a paired confidence interval above the checkpoint's 200-bp result and above chance, run the following pre-specified control at that point and at 200 bp:

- preserve the final 200 bp nearest the -35 box;
- replace or composition-preserving-shuffle the more distant prefix;
- preserve total prompt length;
- rerun the identical sampling and scoring procedure.

This distinguishes meaningful distant genomic information from length, composition and memorisation effects.

## Isambard execution plan

### Stage 0: input and checkpoint audit

- Validate prompt construction locally and on Isambard.
- Inventory all required checkpoints in the Isambard cache.
- Confirm the installed Evo, Evo 2, Vortex, Transformer Engine, HyenaDNA and CUDA versions.
- Confirm that the Evo 2 1B checkpoint passes its official numerical smoke test on GH200 hardware.

### Stage 1: maximum-context smoke tests

For each checkpoint, run a resumable smoke job at its maximum planned context using a very small number of genes and samples. Record peak GPU memory and elapsed time. A checkpoint cannot enter the full array until generation length, probability normalisation, prompt boundaries and output integrity pass validation.

### Stage 2: queue-efficient production

- Use one GH200 per Slurm array element.
- Shard by checkpoint x context length, and by gene block where required.
- Use short, empirically estimated wall times rather than one large multi-GPU allocation.
- Write outputs after every gene and resume from the first incomplete gene.
- Submit already cached checkpoints first while any missing long-context checkpoint is staged.
- Never request more GPUs solely to make queue waiting appear shorter.

### Cohort fallback

Attempt the full existing cohort with 20 samples first. If smoke-test projections show that the 100k-1M conditions cannot finish safely before the submission deadline, use a fixed, preselected 128-gene subset with 20 samples.

If this fallback is activated:

- select the subset before inspecting long-context results;
- stratify by promoter geometry/sigma annotation where possible;
- rerun **every** context length and checkpoint on the same subset;
- do not compare a 128-gene long-context point directly with a 681-gene short-context point;
- record the reason and selection seed in the manifest.

## Output structure and provenance

Use a dedicated experiment root on Isambard and mirror final validated outputs locally:

```text
fig2d_long_context/
  prompts/
  generations/
  probabilities/
  scores/
  summaries/
  figures/
  logs/
  manifests/
```

Required provenance includes:

- exact checkpoint name, revision and file hash;
- model context limit and actual usable prompt limit;
- prompt sequence hash and genomic coordinates;
- sampling configuration and seed;
- code revision or source-tree hash;
- Slurm job and array IDs;
- hostname, GPU type, peak memory, runtime and exit status;
- every excluded or failed gene with a reason;
- validation status for each output shard.

Produce machine-readable Parquet/CSV/JSON outputs plus a final `RESULTS_SUMMARY.md`. The summary must state whether longer context changed placement, base probabilities or entropy; identify any checkpoint change; report controls; and distinguish statistical improvement from biologically useful promoter placement.

## Improved Figure 2d

The revised panel will use:

- logarithmic context-length x-axis from 200 bp to 1M;
- -10 placement within +/-2 bp on the y-axis;
- the 12.5% chance line;
- one explicitly labelled curve per checkpoint;
- curves ending at their trained limit;
- gene-bootstrap 95% confidence bands;
- distinct line styles for base and long-context checkpoints.

A/C/G/T probability profiles will be emitted as a companion or supplementary figure rather than overcrowding Figure 2d. All plotting-source tables must be saved alongside the figures.

## Completion criteria

The experiment is complete only when:

1. all planned or formally activated fallback shards have terminal validated outputs;
2. every result can be traced to its prompt, checkpoint, seed, job and code version;
3. placement and probability summaries have been generated;
4. the improved Figure 2d and probability companion have been rendered and visually checked;
5. `RESULTS_SUMMARY.md` records the result, limitations, failed jobs and any deviations from this plan.

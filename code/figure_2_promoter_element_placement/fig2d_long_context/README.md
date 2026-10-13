# Figure 2d long-context execution code

This directory implements the approved plan in
[`EXPERIMENT_FIG2D_LONG_CONTEXT.md`](../../EXPERIMENT_FIG2D_LONG_CONTEXT.md).
It writes only under the separate Isambard experiment root
`data/benchmark/fig2d_long_context/`; the completed upstream-only benchmark is
not modified.

Typical smoke test:

```bash
python build_prompts.py --width 8000
sbatch --array=0 --export=ALL,MODEL=evo1,CONTEXT=8000,BLOCK_SIZE=2,N_SAMPLES=2,FORWARD_WIDTH=1 run_generation.sbatch
```

Production jobs use fixed gene blocks. Each block is atomic and has a manifest
with prompt/output hashes and its Slurm identifiers, so failed jobs can be
resubmitted without overwriting completed blocks.

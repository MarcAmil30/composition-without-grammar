#!/bin/bash
#SBATCH --job-name=f2d_submit_e1l
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/fig2d_long_context/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/fig2d_long_context/logs/%x_%j.err

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# -------------------------------------------------------------------

set -euo pipefail

CODE=${ICBINB_ROOT}/data/code/fig2d_long_context
MODE=${MODE:-short}
cd "$CODE"

if [[ "$MODE" == "short" ]]; then
  SMOKE_JOB=${SMOKE_JOB:?export SMOKE_JOB}
  jobs=()
  for context in 200 300 500 1000; do
    job=$(sbatch --parsable --array=0-21%4 --job-name="f2d_e1l_${context}" \
      --export=ALL,MODEL=evo1_131k,CONTEXT="$context",BLOCK_SIZE=32,N_SAMPLES=20,FORWARD_WIDTH=1,OUTPUT_GROUP=generations \
      run_generation.sbatch)
    jobs+=("${job%%;*}")
  done
  dependency=$(IFS=:; echo "${jobs[*]}")
  sbatch --dependency="afterok:${dependency}:${SMOKE_JOB}" \
    --export=ALL,MODE=long submit_evo1_131k_curve.sbatch
  printf 'short_jobs=%s smoke_job=%s\n' "${jobs[*]}" "$SMOKE_JOB"
elif [[ "$MODE" == "long" ]]; then
  sbatch --array=0-84%4 --job-name=f2d_e1l_8k_full \
    --export=ALL,MODEL=evo1_131k,CONTEXT=8000,BLOCK_SIZE=8,N_SAMPLES=20,FORWARD_WIDTH=1,OUTPUT_GROUP=generations \
    run_generation.sbatch
else
  echo "unknown MODE=$MODE" >&2
  exit 2
fi

#!/bin/bash
#SBATCH --job-name=f2d_submit449
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
cd "$CODE"
sbatch --array=0-169%4 --job-name=f2d_h_449k \
  --export=ALL,MODEL=hyenadna,CONTEXT=449000,BLOCK_SIZE=4,N_SAMPLES=20,FORWARD_WIDTH=1,OUTPUT_GROUP=generations \
  run_generation.sbatch

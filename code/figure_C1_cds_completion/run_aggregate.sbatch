#!/bin/bash
#SBATCH --job-name=cdsagg
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/cds_completion_probabilities/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/cds_completion_probabilities/logs/%x_%j.err

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -euo pipefail
WORK=${WORK_ROOT}
DATA=$WORK/ICBINB_BIO_NeurIPS2026/data
ROOT=$DATA/benchmark/cds_completion_probabilities
mkdir -p "$ROOT/logs"
$HOME/miniforge3/envs/evo2_env/bin/python \
  "$DATA/code/cds_completion_probabilities/aggregate_and_plot.py" --root "$ROOT"

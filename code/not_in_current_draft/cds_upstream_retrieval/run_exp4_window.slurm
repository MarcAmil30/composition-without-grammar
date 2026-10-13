#!/bin/bash
#SBATCH --job-name=c3p_win
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --chdir=${CL_ROOT}
#SBATCH --output=${ICBINB_ROOT}/logs/exp4win_%j.out
#SBATCH --error=${ICBINB_ROOT}/logs/exp4win_%j.err
#
# EXP 4 upstream WINDOW sweep on one GH200. Same model/checkpoint/retrieval as exp 3;
# only ~1,400 upstreams are re-embedded per window, so this is minutes of GPU.
# chdir is the C3P project because the checkpoint, analysis set and package live there;
# outputs are written into the ICBINB project.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CL_ROOT="${CL_ROOT:-}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.triton}"
# -------------------------------------------------------------------

set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate c3p_model

export HF_HOME=${HF_CACHE}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=${TRITON_CACHE}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python ${ICBINB_ROOT}/code/cds_upstream_retrieval/exp4_window_sweep.py \
    --n_candidates 100 --n_repeats 20 --seed 0

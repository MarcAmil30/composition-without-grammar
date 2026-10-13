#!/bin/bash
#SBATCH --job-name=c3p_len
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --chdir=${CL_ROOT}
#SBATCH --output=logs/exp3len_%j.out
#SBATCH --error=logs/exp3len_%j.err
#
# EXP 3 upstream-length sweep (300/200/100/50 bp) on one GH200.
# Re-embeds only the ~1,400 reference-species upstreams per length; retrieval is cheap.
# --- portable paths (added for release; override via environment) ---
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

python exp3_length_sweep.py --n_candidates 100 --n_repeats 20 --seed 0

echo "done=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=r_c3p_embed
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.err

# C3P retrieval -- embedding only (no likelihood head). Conda env, not the
# container: C3P needs the c3p_model env from Contrastive_Learning_NeurIPS2026.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# -------------------------------------------------------------------

set -uo pipefail
CODE=${ICBINB_ROOT}/data/code/retrieval
source ~/miniforge3/etc/profile.d/conda.sh
conda activate c3p_model
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_CACHE}
export HF_HUB_OFFLINE=1
echo "host $(hostname) | start $(date)"
python -u "$CODE/c3p_retrieval.py" --limit "${LIMIT:-0}" --suffix "${SUFFIX:-}"
rc=$?
echo "end $(date) | exit $rc"
exit $rc

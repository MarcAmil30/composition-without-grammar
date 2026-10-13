#!/bin/bash
#SBATCH --job-name=cdsprob
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/cds_completion_probabilities/logs/%x_%A_%a.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/cds_completion_probabilities/logs/%x_%A_%a.err

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -euo pipefail
WORK=${WORK_ROOT}
DATA=$WORK/ICBINB_BIO_NeurIPS2026/data
CODE=$DATA/code/cds_completion_probabilities
BLOCK_SIZE=${BLOCK_SIZE:-25}
N_SHARDS=${N_SHARDS:-27}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / N_SHARDS))
SHARD=$((SLURM_ARRAY_TASK_ID % N_SHARDS))
case "$MODEL_INDEX" in
  0) MODEL=evo1; VENV=$WORK/ngc_evo_venv; BATCH=1 ;;
  1) MODEL=evo2; VENV=$WORK/ngc_evo2_venv; BATCH=16 ;;
  2) MODEL=hyenadna; VENV=$WORK/ngc_evo_venv; BATCH=32 ;;
  *) echo "bad model index $MODEL_INDEX"; exit 2 ;;
esac
START=$((SHARD * BLOCK_SIZE))
STOP=$((START + BLOCK_SIZE))

mkdir -p "$DATA/benchmark/cds_completion_probabilities/logs"
echo "host=$(hostname) model=$MODEL shard=$SHARD range=$START:$STOP start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home \
  "$WORK/pytorch_25.05-py3.sif" bash -c '
    set -euo pipefail
    source "'"$VENV"'/bin/activate"
    export PYTHONNOUSERSITE=1 HF_HOME="'"$WORK"'/hf_cache" HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TRITON_CACHE_DIR="'"$WORK"'/triton_cache" TOKENIZERS_PARALLELISM=false
    python -u "'"$CODE"'/run_shard.py" --model "'"$MODEL"'" \
      --start "'"$START"'" --stop "'"$STOP"'" --prob-batch "'"$BATCH"'" \
      --data-root "'"$DATA"'"
  '
echo "end=$(date)"

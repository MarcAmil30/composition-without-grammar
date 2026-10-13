#!/bin/bash
#SBATCH --job-name=fig2d_ctx
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/fig2d_long_context/logs/%x_%A_%a.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/fig2d_long_context/logs/%x_%A_%a.err

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -euo pipefail

MODEL=${MODEL:?export MODEL}
CONTEXT=${CONTEXT:?export CONTEXT}
BLOCK_SIZE=${BLOCK_SIZE:-8}
N_SAMPLES=${N_SAMPLES:-20}
FORWARD_WIDTH=${FORWARD_WIDTH:-1}
OUTPUT_GROUP=${OUTPUT_GROUP:-generations}
START=${START:-$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))}
LIMIT=${LIMIT:-$BLOCK_SIZE}

WORK=${WORK_ROOT}
DATA=$WORK/ICBINB_BIO_NeurIPS2026/data
CODE=$DATA/code/fig2d_long_context
mkdir -p "$DATA/benchmark/fig2d_long_context/logs"

echo "host=$(hostname) start=$(date -Is) model=$MODEL context=$CONTEXT start_gene=$START limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case "$MODEL" in
  evo2|evo2_7b) VENV=$WORK/ngc_evo2_venv ;;
  *) VENV=$WORK/ngc_evo_venv ;;
esac
# Most checkpoints use the standard HF_HOME/hub layout. The separately
# staged Evo 2–7B snapshot uses hf_cache directly.
HUB_CACHE=$WORK/hf_cache/hub
if [[ "$MODEL" == "evo2_7b" ]]; then HUB_CACHE=$WORK/hf_cache; fi

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home \
  "$WORK/pytorch_25.05-py3.sif" bash -c '
    set -euo pipefail
    source "'"$VENV"'/bin/activate"
    export PYTHONNOUSERSITE=1
    export HF_HOME="'"$WORK"'/hf_cache"
    export HF_HUB_CACHE="'"$HUB_CACHE"'"
    export HF_HUB_OFFLINE=1
    export TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u "'"$CODE"'/run_generation.py" \
      --model "'"$MODEL"'" --context "'"$CONTEXT"'" \
      --start "'"$START"'" --limit "'"$LIMIT"'" \
      --n-samples "'"$N_SAMPLES"'" --forward-width "'"$FORWARD_WIDTH"'" \
      --output-group "'"$OUTPUT_GROUP"'"
  '

echo "done=$(date -Is)"

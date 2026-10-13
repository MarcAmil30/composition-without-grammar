#!/bin/bash
#SBATCH --job-name=fig2d_prob
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
START=${START:-$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))}
STOP=${STOP:-$((START + BLOCK_SIZE))}
BATCH=${BATCH:-1}
WORK=${WORK_ROOT}
DATA=$WORK/ICBINB_BIO_NeurIPS2026/data
CODE=$DATA/code/fig2d_long_context
case "$MODEL" in evo2|evo2_7b) VENV=$WORK/ngc_evo2_venv ;; *) VENV=$WORK/ngc_evo_venv ;; esac
HUB_CACHE=$WORK/hf_cache/hub
if [[ "$MODEL" == "evo2_7b" ]]; then HUB_CACHE=$WORK/hf_cache; fi
apptainer exec --nv --bind /scratch:/scratch --bind /home:/home \
  "$WORK/pytorch_25.05-py3.sif" bash -c '
    set -euo pipefail
    source "'"$VENV"'/bin/activate"
    export PYTHONNOUSERSITE=1 HF_HOME="'"$WORK"'/hf_cache" \
      HF_HUB_CACHE="'"$HUB_CACHE"'" HF_HUB_OFFLINE=1
    export TRITON_CACHE_DIR="'"$WORK"'/triton_cache" TOKENIZERS_PARALLELISM=false
    python -u "'"$CODE"'/probe_probabilities.py" --model "'"$MODEL"'" \
      --context "'"$CONTEXT"'" --start "'"$START"'" --stop "'"$STOP"'" --batch "'"$BATCH"'"
  '

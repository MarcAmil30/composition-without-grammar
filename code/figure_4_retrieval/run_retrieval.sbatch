#!/bin/bash
#SBATCH --job-name=retr
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.err

# Task 4 retrieval, one model x one mode.
#   sbatch --job-name=retr_prok_score --export=ALL,MODEL=prok,MODE=score run_retrieval.sbatch
#
# Resumable: score matrices + a done-mask are written every 10 query rows and
# embeddings are cached, so a walltime kill just needs a re-submit -- it picks up
# where it stopped. submit_all.sh chains a second job automatically.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -uo pipefail
WORK=${WORK_ROOT}
RETR=$WORK/ICBINB_BIO_NeurIPS2026/data/benchmark/retrieval
CODE=$WORK/ICBINB_BIO_NeurIPS2026/data/code/retrieval
SIF=$WORK/pytorch_25.05-py3.sif
mkdir -p "$RETR/logs" "$RETR/results"

MODEL=${MODEL:?set MODEL}
MODE=${MODE:?set MODE}
LIMIT=${LIMIT:-0}
BUDGET=${BUDGET:-3000}   # 8000 OOMs StripedHyena: the FFT prefill transient scales with batch x fft_size
MEMGB=${MEMGB:-30}
SUFFIX=${SUFFIX:-}

echo "host $(hostname) | $MODEL $MODE | budget $BUDGET | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" bash -c '
  set -u
  source "'"$WORK"'/ngc_evo_venv/bin/activate"
  export PYTHONNOUSERSITE=1 TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
  export HF_HOME="'"$WORK"'/hf_cache" HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u "'"$CODE"'/lm_retrieval.py" --model "'"$MODEL"'" --mode "'"$MODE"'" \
      --limit "'"$LIMIT"'" --budget "'"$BUDGET"'" --mem_gb "'"$MEMGB"'" \
      --suffix "'"$SUFFIX"'"
'
rc=$?
echo "end $(date) | exit $rc"
exit $rc

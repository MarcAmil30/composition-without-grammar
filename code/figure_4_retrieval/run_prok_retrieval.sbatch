#!/bin/bash
#SBATCH --job-name=prokretr
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.err

# Task 4 -- PROK CDS<->upstream retrieval.
#
#   sbatch --export=ALL,MODE=embed run_prok_retrieval.sbatch                 # minutes
#   sbatch --export=ALL,MODE=score,LIMIT=120 --time=01:00:00 \                # smoke test
#          run_prok_retrieval.sbatch
#   sbatch --export=ALL,MODE=score run_prok_retrieval.sbatch                 # the real run
#
# The score mode is resumable: score_*_matrix.npy + score_done.npy are written
# every 10 query rows, so a timeout just needs a re-submit.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -uo pipefail
WORK=${WORK_ROOT}
RETR=$WORK/ICBINB_BIO_NeurIPS2026/data/benchmark/retrieval
CODE=$WORK/ICBINB_BIO_NeurIPS2026/data/code/retrieval
SIF=$WORK/pytorch_25.05-py3.sif
mkdir -p "$RETR/logs" "$RETR/results/prok"

MODE=${MODE:-embed}
LIMIT=${LIMIT:-0}
SUFFIX=${SUFFIX:-}

echo "host $(hostname) | mode $MODE | limit $LIMIT | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" bash -c '
  set -u
  source "'"$WORK"'/ngc_evo_venv/bin/activate"
  export PYTHONNOUSERSITE=1 TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
  export HF_HOME="'"$WORK"'/hf_cache" HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u "'"$CODE"'/prok_retrieval.py" \
      --mode "'"$MODE"'" --limit "'"$LIMIT"'" --suffix "'"$SUFFIX"'"
'
rc=$?
echo "end $(date) | exit $rc"
exit $rc

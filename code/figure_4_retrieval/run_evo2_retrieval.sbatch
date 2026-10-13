#!/bin/bash
#SBATCH --job-name=r_evo2
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/retrieval/logs/%x_%j.err

# Evo 2 retrieval. Uses ngc_evo2_venv (the `evo2` package), not ngc_evo_venv.
# Resumable in the same way as the others.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -uo pipefail
WORK=${WORK_ROOT}
CODE=$WORK/ICBINB_BIO_NeurIPS2026/data/code/retrieval
SIF=$WORK/pytorch_25.05-py3.sif
MODE=${MODE:?set MODE}
LIMIT=${LIMIT:-0}
BUDGET=${BUDGET:-3000}
SUFFIX=${SUFFIX:-}

echo "host $(hostname) | evo2 $MODE | limit $LIMIT | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" bash -c '
  set -u
  source "'"$WORK"'/ngc_evo2_venv/bin/activate"
  export PYTHONNOUSERSITE=1 TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
  export HF_HOME="'"$WORK"'/hf_cache" HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u "'"$CODE"'/evo2_retrieval.py" --mode "'"$MODE"'" --limit "'"$LIMIT"'" \
      --budget "'"$BUDGET"'" --suffix "'"$SUFFIX"'"
'
rc=$?
echo "end $(date) | exit $rc"
exit $rc

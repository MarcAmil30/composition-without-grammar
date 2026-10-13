#!/bin/bash
#SBATCH --job-name=lossdecomp
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=${ICBINB_ROOT}/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/logs/%x_%j.err
#
# Splits PROK's teacher-forced cross-entropy by record region, so the single
# number on the training curve can be attributed.
#
#   sbatch run_loss_decomposition.sbatch                    # 2000 test records
#   sbatch --export=ALL,LIMIT=200 --time=00:20:00 run_loss_decomposition.sbatch
#
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -uo pipefail
WORK=${WORK_ROOT}
CODE=$WORK/ICBINB_BIO_NeurIPS2026/code
OUT=$WORK/ICBINB_BIO_NeurIPS2026/results
SIF=$WORK/pytorch_25.05-py3.sif
LIMIT=${LIMIT:-2000}
mkdir -p "$WORK/ICBINB_BIO_NeurIPS2026/logs" "$OUT"

echo "host $(hostname) | limit $LIMIT | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" bash -c '
  set -u
  source "'"$WORK"'/ngc_evo_venv/bin/activate"
  export PYTHONNOUSERSITE=1 TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
  export HF_HOME="'"$WORK"'/hf_cache" HF_HUB_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u "'"$CODE"'/loss_decomposition.py" \
      --limit "'"$LIMIT"'" --out "'"$OUT"'/loss_decomposition.csv"
'
rc=$?
echo "end $(date) | exit $rc"
exit $rc

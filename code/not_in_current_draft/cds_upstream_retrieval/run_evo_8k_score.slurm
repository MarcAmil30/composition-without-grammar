#!/bin/bash
#SBATCH --job-name=evo8k_score
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=${CL_ROOT}
#SBATCH --output=logs/evo8kscore_%j.out
#SBATCH --error=logs/evo8kscore_%j.err
#
# evo-1-8k-base SCORE-based CDS/upstream retrieval length sweep, in the NGC container.
# --- portable paths (added for release; override via environment) ---
CL_ROOT="${CL_ROOT:-}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -euo pipefail
BASE=${WORK_ROOT}
SIF=$BASE/pytorch_25.05-py3.sif
VENV=$BASE/ngc_evo_venv
REPO=$BASE/Contrastive_Learning_NeurIPS2026/evo

NSUB="${NSUB:-150}"
SUBDIR="${SUBDIR:-}"
echo "host=$(hostname) start=$(date -Is)  NSUB=$NSUB SUBDIR='$SUBDIR'"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" bash -c '
    set -e
    source "'"$VENV"'/bin/activate"
    cd "'"$REPO"'"
    export PYTHONNOUSERSITE=1
    export HF_HOME="'"$BASE"'/hf_cache"
    export HF_HUB_OFFLINE=1
    export TRITON_CACHE_DIR="'"$BASE"'/triton_cache"
    export TOKENIZERS_PARALLELISM=false
    for d in $(echo "$LD_LIBRARY_PATH" | tr ":" " ") /usr/local/cuda/compat/lib; do
        if [ -e "$d/libcuda.so.1" ]; then export TRITON_LIBCUDA_PATH="$d"; break; fi
    done
    python -c "import evo; from evo.scoring import prepare_batch, logits_to_logprobs; print(\"evo ok\")"
    python evo_8k_score_length_sweep.py \
        --n_sub "'"$NSUB"'" --n_candidates 100 --n_repeats 20 --seed 0 \
        --out_subdir "'"$SUBDIR"'"
'
echo "done=$(date -Is)"

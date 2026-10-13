#!/bin/bash
#SBATCH --job-name=upgen
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/upstream_only/generations/logs/%x_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/upstream_only/generations/logs/%x_%j.err

# One job = one model, all 4 (task x background) shards for it.
#
#   sbatch --job-name=upgen_evo1 run_generation.sbatch evo1
#
# Two execution paths:
#   evo1 / prok / evo2  -> NGC container + ngc_evo_venv (has flash_attn 2.7.3).
#                          StripedHyena's remote code needs flash_attn for MHA;
#                          the conda env does not have it and fails with
#                          "NameError: name 'MHA' is not defined".
#   promogen2/hyenadna  -> conda evo2_env (torch 2.6 + transformers 4.52).
#
# Compute nodes have NO INTERNET; HF_HUB_OFFLINE=1 makes a missing checkpoint
# fail loudly instead of hanging.

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME}"
# -------------------------------------------------------------------

set -uo pipefail

MODEL=${1:?usage: sbatch run_generation.sbatch MODEL}
WORK=${WORK_ROOT}
ROOT=$WORK/ICBINB_BIO_NeurIPS2026/data
BENCH=$ROOT/benchmark/upstream_only
SIF=$WORK/pytorch_25.05-py3.sif

mkdir -p "$BENCH/generations/logs"

echo "host $(hostname) | model $MODEL | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_one () {   # $1=python  $2=task  $3=background
  echo ""
  echo "---- $MODEL / $2 / $3 ---- $(date)"
  "$1" -u "$ROOT/code/generation/generate.py" \
       --model "$MODEL" --task "$2" --background "$3" \
    && echo "SHARD_OK $MODEL $2 $3" \
    || echo "SHARD_FAIL $MODEL $2 $3"
}

case "$MODEL" in
  evo1|prok|evo2)
    VENV=$WORK/ngc_evo_venv
    [ "$MODEL" = evo2 ] && VENV=$WORK/ngc_evo2_venv
    apptainer exec --nv --bind /scratch:/scratch --bind /home:/home "$SIF" \
      bash -c '
        set -u
        source "'"$VENV"'/bin/activate"
        export PYTHONNOUSERSITE=1
        export TRITON_CACHE_DIR="'"$WORK"'/triton_cache"
        export HF_HOME="'"$WORK"'/hf_cache"
        export HF_HUB_OFFLINE=1
        export BENCH_DIR="'"$BENCH"'"
        export GEN_DIR="'"$BENCH"'/generations"
        for T in generate_promoter generate_sd; do
          for B in leaked clean; do
            echo ""; echo "---- '"$MODEL"' / $T / $B ---- $(date)"
            python -u "'"$ROOT"'/code/generation/generate.py" \
                   --model "'"$MODEL"'" --task "$T" --background "$B" \
              && echo "SHARD_OK '"$MODEL"' $T $B" \
              || echo "SHARD_FAIL '"$MODEL"' $T $B"
          done
        done
      '
    ;;
  *)
    export HF_HOME=$WORK/hf_cache
    export HF_HUB_OFFLINE=1
    export BENCH_DIR=$BENCH
    export GEN_DIR=$BENCH/generations
    PY=$HOME/miniforge3/envs/evo2_env/bin/python
    for T in generate_promoter generate_sd; do
      for B in leaked clean; do run_one "$PY" "$T" "$B"; done
    done
    ;;
esac

echo ""
echo "end $(date)"

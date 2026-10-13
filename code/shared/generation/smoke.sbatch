#!/bin/bash
#SBATCH --job-name=upgen_smoke
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=${ICBINB_ROOT}/data/benchmark/upstream_only/generations/logs/smoke_%j.out
#SBATCH --error=${ICBINB_ROOT}/data/benchmark/upstream_only/generations/logs/smoke_%j.err

# Loads every model in turn, generates 2 genes x 2 samples, reports pass/fail.
# Never aborts on a single failure -- the point is to learn about ALL of them at once.

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
EVO2_VENV="${EVO2_VENV:-}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# -------------------------------------------------------------------

ROOT=${ICBINB_ROOT}/data
export BENCH_DIR=$ROOT/benchmark/upstream_only
export GEN_DIR=$BENCH_DIR/generations/_smoke
export HF_HOME=${HF_CACHE}
export HF_HUB_OFFLINE=1
mkdir -p "$GEN_DIR" "$BENCH_DIR/generations/logs"

echo "host $(hostname) | start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in promogen2 hyenadna evo1 prok evo2; do
  case "$M" in
    evo2) PY=${EVO2_VENV}/bin/python ;;
    *)    PY=$HOME/miniforge3/envs/evo2_env/bin/python ;;
  esac
  for T in generate_promoter generate_sd; do
    echo ""
    echo "================ SMOKE $M / $T ================"
    timeout 900 $PY -u "$ROOT/code/generation/generate.py" \
      --model "$M" --task "$T" --background leaked --smoke --device cuda \
      && echo "SMOKE_OK $M $T" || echo "SMOKE_FAIL $M $T (exit $?)"
  done
done

echo ""
echo "end $(date)"

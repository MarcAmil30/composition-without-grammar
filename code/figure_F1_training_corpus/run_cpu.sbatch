#!/bin/bash
#SBATCH --job-name=basecomp
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=${ICBINB_ROOT}/results/training_corpus_composition/basecomp_%j.out
#SBATCH --error=${ICBINB_ROOT}/results/training_corpus_composition/basecomp_%j.err

# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${CONDA_PREFIX:-$HOME/miniforge3}}"
# -------------------------------------------------------------------

set -euo pipefail

ROOT=${ICBINB_ROOT}
PY=${CONDA_ROOT}/envs/evo2_env/bin/python
OUT=$ROOT/results/training_corpus_composition

mkdir -p "$OUT"
"$PY" "$ROOT/code/training_corpus_composition/analyse_and_plot.py" --out-dir "$OUT"

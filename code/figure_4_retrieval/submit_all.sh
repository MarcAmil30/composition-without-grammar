#!/bin/bash
# Submit the whole of Task 4. Safe to re-run: every stage is resumable, so a
# re-submit continues from the last checkpoint instead of redoing work.
#
#   bash submit_all.sh              # everything
#   bash submit_all.sh prok evo1    # only these models
#
# Each scoring run is chained with two follow-on jobs (dependency=afterany).
# If the first hits the 12 h walltime, the next picks up from the done-mask; if
# it finished, the follow-on just re-runs the cheap evaluation and exits.
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# -------------------------------------------------------------------

set -uo pipefail
CODE=${ICBINB_ROOT}/data/code/retrieval
RETR=${ICBINB_ROOT}/data/benchmark/retrieval
cd "$CODE" || exit 1
mkdir -p "$RETR/logs" "$RETR/results"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(hyenadna evo1 prok)

sub() {   # sub <model> <mode> <budget> [dependency]
  local m=$1 mode=$2 budget=$3 dep=${4:-}
  local extra=""
  [ -n "$dep" ] && extra="--dependency=afterany:$dep"
  sbatch --parsable $extra \
    --job-name="r_${m}_${mode}" \
    --export=ALL,MODEL="$m",MODE="$mode",BUDGET="$budget" \
    run_retrieval.sbatch
}

echo "submitted at $(date)" | tee -a "$RETR/results/submissions.log"
for m in "${MODELS[@]}"; do
  # embeddings first -- minutes, and they surface a broken model immediately
  E=$(sub "$m" embed 8000)
  # scoring, plus two chained resumes for the 12 h walltime
  S1=$(sub "$m" score 8000)
  S2=$(sub "$m" score 8000 "$S1")
  S3=$(sub "$m" score 8000 "$S2")
  printf '%-10s embed %s | score %s -> %s -> %s\n' "$m" "$E" "$S1" "$S2" "$S3" \
    | tee -a "$RETR/results/submissions.log"
done
echo
echo "watch with:  bash $CODE/status.sh"

#!/bin/bash
# Task 4 progress. Run any time -- nothing here depends on a live session.
#   bash status.sh
# --- portable paths (added for release; override via environment) ---
ICBINB_ROOT="${ICBINB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# -------------------------------------------------------------------

RETR=${ICBINB_ROOT}/data/benchmark/retrieval
RES=$RETR/results

echo "=============================================================="
echo " Task 4 -- CDS<->upstream retrieval   $(date '+%Y-%m-%d %H:%M')"
echo "=============================================================="
echo
echo "QUEUE"
squeue -u "$USER" -o "%.10i %.16j %.9T %.10M %.10L %R" 2>/dev/null | sed 's/^/  /'
[ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -eq 0 ] && echo "  (nothing queued or running)"
echo
echo "SCORING PROGRESS  (rows of the 500x500 matrix)"
for d in "$RES"/*/; do
  m=$(basename "$d")
  [ "$m" = "progress.tsv" ] && continue
  f="$d/score_done.npy"
  if [ -f "$f" ]; then
    python3 - "$f" <<'PY' 2>/dev/null | sed "s/^/  $(printf '%-10s' "$m")/"
import sys, numpy as np
d = np.load(sys.argv[1])
n, k = len(d), int(d.sum())
bar = "#" * int(30*k/n) + "." * (30 - int(30*k/n))
print(f"[{bar}] {k}/{n} ({100*k/n:.0f}%)")
PY
  fi
done
echo
echo "RESULTS SO FAR"
if [ -f "$RES/progress.tsv" ]; then
  grep -h RESULT "$RES/progress.tsv" 2>/dev/null | tail -20 | sed 's/^/  /'
  [ -z "$(grep -h RESULT "$RES/progress.tsv" 2>/dev/null)" ] && echo "  (none yet)"
else
  echo "  (none yet)"
fi
echo
echo "LAST 8 MILESTONES"
[ -f "$RES/progress.tsv" ] && tail -8 "$RES/progress.tsv" | sed 's/^/  /'
echo
echo "FAILURES (non-zero exits in the last 20 logs)"
grep -l "exit [1-9]" $(ls -t "$RETR"/logs/*.out 2>/dev/null | head -20) 2>/dev/null \
  | sed 's/^/  /' || echo "  (none)"
echo
echo "full table:  python3 $RETR/../../code/retrieval/collect.py"

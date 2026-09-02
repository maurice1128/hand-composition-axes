#!/bin/sh
# Grasp retention across four replicated models and three difficulties.
#
# One model at one difficulty gave 100% and 95% -- a ceiling that cannot
# discriminate. Difficulty is mass and shake; the object size stays fixed
# because enlarging it made even the positive control fail, and a setting whose
# control does not separate measures nothing.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
for s in 1 2 3 4; do
  for d in easy medium hard; do
    [ -f "runs/bricks_s$s/best.pt" ] || continue
    echo "=== seed $s / $d ==="
    $PY -u scripts/experiment_grasp_success.py --run-dir "runs/bricks_s$s" \
        --difficulty "$d" --n-pairs 40 \
        --out "runs/grasp_s${s}_${d}.json" 2>&1 | grep -E "controls|seen|unseen|p ="
  done
done
echo "[grasp] done at $(date)"

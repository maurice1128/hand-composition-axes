#!/bin/sh
# Everything left that needs the GPU, unattended.
#
# Runs after replicate_bricks.sh: re-does the robot-transfer experiment on all
# four replicated models with more pairs (the first run used one unverified
# model and 23 pairs per arm, too few for the two nulls it reported), then
# redraws the primitive-library figure from a replicated model rather than the
# single run whose identity ratio is pending.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

while [ "$($PY scripts/count_running.py train_prior)" != "0" ] \
   || [ "$($PY scripts/count_running.py demo_composition)" != "0" ] \
   || [ "$($PY scripts/count_running.py eval_segmentation)" != "0" ]; do
  sleep 120
done
echo "[finish] replications done at $(date)"

for s in 1 2 3 4; do
  [ -f "runs/bricks_s$s/best.pt" ] || continue
  echo "=== robot transfer, seed $s ==="
  $PY -u scripts/experiment_robot_transfer.py --run-dir "runs/bricks_s$s" \
      --n-pairs 60 --out "runs/robot_transfer_s$s.json" 2>&1 | tail -12
done

# Figure from a replicated model, not the pending single run.
$PY scripts/figure_primitive_library.py --run-dir runs/bricks_s1 \
    --out docs/figures/primitive_library.png --gif || true

echo "[finish] all done at $(date)"

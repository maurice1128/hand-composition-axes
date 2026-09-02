#!/bin/sh
# Eight more seeds of the identity-ratio measurement.
#
# Five runs gave 0.867, 1.154, 1.816, 2.838, 2.843: four above the threshold of
# 1 that "identity beats context" requires, but a one-sample t-test against 1.0
# lands at p = 0.094. The contrastive term's *effect* is established (0.659 ->
# 1.904 against runs without it, p = 0.037); what is not established is that the
# result clears the threshold reliably, and that is the clause the word
# "library" depends on.
#
# Eight more takes n to 13. If the true mean sits where the current estimate
# puts it this passes 0.05; if it does not, that settles it. Both outcomes are
# worth more than n = 5.
#
# Seeds 5-12 continue the existing numbering. Nothing here selects on outcome:
# every run's identity ratio goes into the pooled estimate, including any that
# fall below 1. Discarding an inconvenient run is the specific error that forced
# an earlier withdrawal in this project.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

while [ "$($PY scripts/count_running.py train_prior)" != "0" ] \
   || [ "$($PY scripts/count_running.py experiment_robot_transfer)" != "0" ]; do
  sleep 120
done
echo "[more] GPU free at $(date)"

for batch in "5 6 7 8" "9 10 11 12"; do
  for s in $batch; do
    $PY -u scripts/train_prior.py --model modular \
        --bundle data/bundles/oakink2.npz --run-dir "runs/bricks_s$s" --fresh \
        --set train.epochs=200 train.save_every=99999 loader.stride=8 \
              model.n_primitives=16 model.consistency_weight=0.003 \
              model.consistency_contexts=4 train.seed=$s \
        > "runs/bricks_s${s}_log.txt" 2>&1 &
  done
  wait
  echo "[more] batch $batch trained at $(date)"
done

for s in 5 6 7 8 9 10 11 12; do
  echo "=== seed $s ==="
  $PY scripts/assert_trained.py "runs/bricks_s$s" --min-epoch 150 || continue
  $PY scripts/demo_composition.py --run-dir "runs/bricks_s$s" \
      --bundle data/bundles/oakink2.npz 2>&1 | grep -E "identity ratio|PASS|FAIL|plug-and-play"
done

$PY scripts/pool_identity.py || true
echo "[more] done at $(date)"

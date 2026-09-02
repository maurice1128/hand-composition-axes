#!/bin/sh
# Replicate the primitive-identity result four more times.
#
# One run reported an identity ratio of 2.843 with the contrastive consistency
# term against 0.604 without it. That single number cannot be quoted: the
# consistency sweep's own spread at this weight is 1.442 +- 0.579, so 2.843 sits
# about 2.4 sd above its condition mean, and this project has already withdrawn
# an identity ratio (0.055) that turned out to be single-run noise -- three runs
# of the identical configuration gave 1.060, 0.086 and 0.885.
#
# Four more seeds, same configuration, run concurrently. The claim to be tested
# is not "the ratio is 2.84" but "the ratio exceeds 1 reliably", which is what
# "reusable brick" requires and what a reviewer will ask for.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

for s in 1 2 3 4; do
  $PY -u scripts/train_prior.py --model modular \
      --bundle data/bundles/oakink2.npz --run-dir "runs/bricks_s$s" --fresh \
      --set train.epochs=200 train.save_every=99999 loader.stride=8 \
            model.n_primitives=16 model.consistency_weight=0.003 \
            model.consistency_contexts=4 train.seed=$s \
      > "runs/bricks_s${s}_log.txt" 2>&1 &
done
wait
echo "[bricks] all four trained at $(date)"

for s in 1 2 3 4; do
  echo "=== seed $s ==="
  $PY scripts/assert_trained.py "runs/bricks_s$s" --min-epoch 150 || continue
  $PY scripts/demo_composition.py --run-dir "runs/bricks_s$s" \
      --bundle data/bundles/oakink2.npz 2>&1 | grep -E "identity ratio|PASS|FAIL|plug-and-play"
  $PY scripts/eval_segmentation.py --run-dir "runs/bricks_s$s" \
      --bundle data/bundles/oakink2.npz 2>&1 | grep -E "normalised MI|boundary F1"
done
echo "[bricks] done at $(date)"

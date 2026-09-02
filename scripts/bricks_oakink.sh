#!/bin/sh
# Four banks on OakInk-Image, so composability is measured where difficulty is.
#
# Every composability and transfer result in the paper comes from banks trained
# on OakInk2 -- the dataset the paper's own instrument finds null and the screen
# scores negative. The difficulty effect is on OakInk-Image. So the two halves of
# the argument sit on different data, and "where the condition holds, the bank
# composes" is a sentence about neither.
#
# The recipe is identical to scripts/replicate_bricks.sh, which produced the
# OakInk2 banks, so nothing but the bundle differs. One earlier attempt on this
# bundle exists (runs/prior_oakink_modular, identity ratio 0.055) but was trained
# without the consistency term, which is the term that took OakInk2 from 0.37 to
# 1.84; it is not evidence that the bundle resists a bank.
#
# Declared before running: four seeds, identity ratio and segmentation pooled
# over all four, and whatever they show is what gets reported. If the bank
# collapses on this bundle, or its identity ratio fails to exceed 1, that is
# reportable and changes the paper's composability claim rather than being
# quietly dropped in favour of the OakInk2 numbers already in hand.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

for s in 1 2 3 4; do
  $PY -u scripts/train_prior.py --model modular \
      --bundle data/bundles/oakink.npz --run-dir "runs/oakbricks_s$s" --fresh \
      --set train.epochs=200 train.save_every=99999 loader.stride=8 \
            model.n_primitives=16 model.consistency_weight=0.003 \
            model.consistency_contexts=4 train.seed=$s \
      > "runs/oakbricks_s${s}_log.txt" 2>&1 &
done
wait
echo "[oakbricks] four banks trained at $(date)"

for s in 1 2 3 4; do
  echo "=== seed $s ==="
  $PY -u scripts/demo_composition.py --run-dir "runs/oakbricks_s$s" \
      --bundle data/bundles/oakink.npz || true
done
echo "[oakbricks] composition measured at $(date)"

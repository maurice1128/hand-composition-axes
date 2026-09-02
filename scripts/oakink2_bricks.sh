#!/bin/sh
# Train the OakInk2 prior to convergence, then run the two checks that decide
# whether "composable brick library" can be claimed at all.
#
# Stop on any failure. Without `set -e` the epoch assertion printed its warning
# and the script went on to compute segmentation and composition metrics from a
# 2-epoch model -- exactly the failure it was written to prevent.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
RUN=runs/prior_oakink2_modular

# Refuse to start if another copy is already training into the same run dir.
# Three copies once ran simultaneously with --fresh, clobbering each other's
# checkpoints; every number read out of that directory was meaningless.
#
# The guard used to be an inline PowerShell one-liner whose nested quoting
# unbalanced the rest of the script, so `sh` hit a syntax error at the epoch
# assertion and skipped it. Keeping the quoting shallow is the whole point.
running=$($PY scripts/count_running.py train_prior oakink2)
if [ "${running:-0}" -gt 0 ]; then
  echo "ABORT: $running copy(ies) of this training already running."
  exit 1
fi

$PY -u scripts/train_prior.py --model modular \
    --bundle data/bundles/oakink2.npz --run-dir "$RUN" --fresh \
    --set train.epochs=200 train.save_every=99999 loader.stride=8 model.n_primitives=16

echo "=== EPOCHS ACTUALLY TRAINED ==="
$PY scripts/assert_trained.py "$RUN" --min-epoch 150

echo "=== SEGMENTATION vs GROUND TRUTH ==="
$PY scripts/eval_segmentation.py --run-dir "$RUN" --bundle data/bundles/oakink2.npz

echo "=== COMPOSITION ==="
$PY scripts/demo_composition.py --run-dir "$RUN" --bundle data/bundles/oakink2.npz

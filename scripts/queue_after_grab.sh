#!/bin/sh
# Everything that needs the GPU, in the order that protects the most evidence.
#
# 1. OakInk on the dataset's OWN taxonomy. The headline result was measured on
#    a grouping this repo invented, and that grouping turned out to be the
#    object id's source prefix (A/C/O/S/Y) rather than anything about the
#    object. Until the effect is shown on an official axis, the strongest
#    claim in the project rests on an arbitrary partition. This runs first.
# 2. OakInk2 bricks: are the learned primitives nameable units? Only OakInk2
#    annotates primitive names, so it is the only place this is answerable.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

while [ "$($PY scripts/count_running.py experiment_paired grab)" != "0" ]; do
  sleep 120
done
echo "[queue] GRAB finished at $(date)."
$PY scripts/analyse_paired.py runs/grab_shape_v2 || true

for MODE in oakink_category oakink_attr oakink_class; do
  echo "[queue] === OakInk on official axis: $MODE ($(date)) ==="
  $PY -u scripts/check_composition_leak.py --bundle data/bundles/oakink.npz \
      --granularity "$MODE" --held-compositions 5 --min-per-composition 5 --budget 256
  $PY -u scripts/experiment_paired_composition.py \
      --bundle data/bundles/oakink.npz --out "runs/oakink_${MODE}" \
      --granularity "$MODE" --kinds perframe modular --budgets 256 \
      --held-compositions 5 --min-per-composition 5 \
      --seeds $(seq -s' ' 0 69) --epochs 120 --fresh
  $PY scripts/analyse_paired.py "runs/oakink_${MODE}" || true
done

echo "[queue] === OakInk2 bricks: are primitives nameable? ($(date)) ==="
sh scripts/oakink2_bricks.sh

$PY scripts/plot_difficulty_survey.py || true
echo "[queue] all done $(date)"

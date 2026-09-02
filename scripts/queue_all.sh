#!/bin/sh
# Everything that still needs the GPU, in one sequential chain.
#
# Sequential, not parallel: running the OakInk sweep beside GRAB pushed one
# epoch from 0.5 s to 3.6 s, turning a 4.7-hour job into a 34-hour one. This
# machine also runs a dozen of the user's own GPU jobs and has already thrown
# one `CUDA error: unknown` under contention.
#
# Order is by what each result protects:
#   1. GRAB finishes           -- the replication already reads null, but n=40 is
#                                 what gets reported, not n=28.
#   2. OakInk official category -- protects the headline. The only dataset with
#                                 measurable difficulty was measured on an axis
#                                 this repo invented from object-id prefixes,
#                                 which turned out to encode provenance. If the
#                                 effect does not survive OakInk's own taxonomy,
#                                 nothing downstream matters.
#   3. OakInk2 bricks          -- are the learned primitives nameable units.
#                                 Never once completed: killed at epoch 2, then
#                                 at 57 by a CUDA fault.
#   4. OakInk affordance axis  -- the second official grouping, and the most
#                                 grasp-relevant of the three.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

wait_for() {
  while [ "$($PY scripts/count_running.py $1 $2)" != "0" ]; do sleep 120; done
}

echo "[queue] waiting for GRAB to finish"
wait_for experiment_paired grab.npz
echo "[queue] GRAB done at $(date)"
$PY scripts/analyse_paired.py runs/grab_shape_v2 || true

for axis in oakink_category oakink_attr; do
  out=runs/oakink_official_${axis#oakink_}
  if [ -f "$out/results.json" ]; then
    echo "[queue] $out exists, skipping"
    continue
  fi
  echo "[queue] $axis starting at $(date)"
  $PY -u scripts/experiment_paired_composition.py \
      --bundle data/bundles/oakink.npz --out "$out" \
      --granularity "$axis" --kinds perframe modular \
      --budgets 256 --held-compositions 5 --min-per-composition 5 \
      --seeds $(seq -s' ' 0 69) --epochs 120 --fresh \
      > "runs/$(basename $out)_log.txt" 2>&1 || true
  $PY scripts/analyse_paired.py "$out" || true

  # The nameability question runs between the two axes rather than last: it is
  # the one that has never produced a number at all.
  if [ "$axis" = "oakink_category" ]; then
    echo "[queue] oakink2 bricks starting at $(date)"
    sh scripts/oakink2_bricks.sh || true
  fi
done

$PY scripts/plot_difficulty_survey.py || true
echo "[queue] all done at $(date)"

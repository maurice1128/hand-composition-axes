#!/bin/sh
# Everything still outstanding, reordered by value and run concurrently.
#
# Two changes from the chain this replaces.
#
# ORDER. The contact experiment was last, behind oakink_attr's 70 seeds, so it
# would not have started for 20-40 hours. It is the only outstanding item that
# can supply a *mechanism* for the one-positive-four-nulls pattern, which is
# the single evidential weakness. It goes first now; oakink_attr, a second
# official axis that is nice to have rather than load-bearing, goes last.
#
# CONCURRENCY. The GPU sits at 11% utilisation on 950 MiB of 8151 MiB -- the
# bottleneck is Python overhead per batch, not the card. Three independent jobs
# fit. The alternative speedup, raising the batch size, measured 3.2x faster
# (2.35 -> 0.73 s/epoch at batch 1024) but changes the number of gradient steps
# per epoch, which would make the contact arms incomparable with the GRAB
# shape->intent run they are meant to sit beside. Concurrency costs nothing in
# comparability.
#
# If CPU contention makes this slower rather than faster, the fix is to drop
# back to one job at a time -- the arms are independent, so nothing is lost.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

echo "[q2] waiting for the OakInk official-category sweep to finish"
while [ "$($PY scripts/count_running.py experiment_paired oakink.npz)" != "0" ]; do
  sleep 120
done
echo "[q2] free at $(date)"
$PY scripts/analyse_paired.py runs/oakink_official_category || true

CONTACT="--bundle data/bundles/grab_contact.npz --granularity shape \
  --kinds perframe modular --budgets 256 --held-compositions 6 \
  --min-per-composition 5 --min-chains 5 --stride 32 --epochs 120 --fresh \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

echo "[q2] launching both contact arms and the OakInk2 bricks run together"
$PY -u scripts/experiment_paired_composition.py $CONTACT --ignore-aux \
    --out runs/contact_pose > runs/contact_pose_log.txt 2>&1 &
$PY -u scripts/experiment_paired_composition.py $CONTACT \
    --out runs/contact_full > runs/contact_full_log.txt 2>&1 &
sh scripts/oakink2_bricks.sh > runs/oakink2_bricks_log.txt 2>&1 &
wait
echo "[q2] contact and bricks finished at $(date)"

$PY scripts/analyse_paired.py runs/contact_pose || true
$PY scripts/analyse_paired.py runs/contact_full || true
$PY scripts/analyse_contact.py || true

echo "[q2] oakink_attr (last, and droppable) at $(date)"
$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --out runs/oakink_official_attr \
    --granularity oakink_attr --kinds perframe modular --budgets 256 \
    --held-compositions 5 --min-per-composition 5 \
    --seeds $(seq -s' ' 0 69) --epochs 120 --fresh \
    > runs/oakink_official_attr_log.txt 2>&1 || true
$PY scripts/analyse_paired.py runs/oakink_official_attr || true
$PY scripts/plot_difficulty_survey.py || true
echo "[q2] all done at $(date)"

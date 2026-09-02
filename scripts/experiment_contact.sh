#!/bin/sh
# Does compositional structure live in contact rather than in joint angles?
#
# Four of five datasets show no compositional difficulty in 27 joint angles,
# including OakInk2 -- which annotates 391 primitive transitions and whose
# retargeting residual halved after the conventions were fixed. The standing
# explanation is that primitives differ in what the hand touches and why, not
# in how the fingers are configured. Nothing has tested it, because only GRAB
# carries contact.
#
# This is that test, and it is a controlled one. Both arms read the SAME
# bundle, use the SAME seeds and therefore the SAME splits, and the same five
# gates. Exactly one thing differs:
#
#   arm POSE     27 channels          (--ignore-aux)
#   arm CONTACT  27 + 16 channels
#
# Both are scored on the 27 pose channels, so their penalties are comparable;
# the contact arm is additionally scored on its 16 contact channels alone.
# Three numbers come out:
#
#   penalty(pose | pose-only model)      the null already measured: ~0
#   penalty(pose | contact-aware model)  does contact help predict posture?
#   penalty(contact)                     is the compositional signal here?
#
# A large third number with a near-zero first is the result worth having: the
# four nulls stop being "we found nothing" and become "we found where the
# signal is not".
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
BUNDLE=data/bundles/grab_contact.npz
SEEDS=$(seq -s' ' 0 19)
COMMON="--granularity shape --kinds perframe modular --budgets 256 \
        --held-compositions 6 --min-per-composition 5 --min-chains 5 \
        --stride 32 --epochs 120 --fresh"

while [ "$($PY scripts/count_running.py experiment_paired)" != "0" ] \
   || [ "$($PY scripts/count_running.py train_prior)" != "0" ]; do
  sleep 180
done
echo "[contact] GPU free at $(date)"

# The leak gate certifies the split both arms share. Run it once, on the file
# both arms read -- a gate run against a differently-built bundle certifies
# nothing about the run that follows.
$PY -u scripts/check_composition_leak.py --bundle "$BUNDLE" --granularity shape \
    --held-compositions 6 --min-per-composition 5 --min-chains 5 --budget 256

echo "[contact] arm POSE (27 channels) starting at $(date)"
$PY -u scripts/experiment_paired_composition.py --bundle "$BUNDLE" \
    --out runs/contact_pose --ignore-aux --seeds $SEEDS $COMMON \
    > runs/contact_pose_log.txt 2>&1 || true
$PY scripts/analyse_paired.py runs/contact_pose || true

echo "[contact] arm CONTACT (43 channels) starting at $(date)"
$PY -u scripts/experiment_paired_composition.py --bundle "$BUNDLE" \
    --out runs/contact_full --seeds $SEEDS $COMMON \
    > runs/contact_full_log.txt 2>&1 || true
$PY scripts/analyse_paired.py runs/contact_full || true

$PY scripts/analyse_contact.py || true
echo "[contact] done at $(date)"

#!/bin/sh
# The rate-matched control on the axis that carries the claim.
#
# runs/pc_easy_ratematched established two things this run depends on. First,
# `perframe_rate` BINDS: raising the baseline's free-bits floor to 0.186/dim
# moved its realised KL from 0.224 to 2.133 nats/frame, against the modular
# arm's 2.334 -- matched. Second, `modular_rate` is VOID: charging the
# assignment its categorical KL on top of the existing entropy term drove the
# assignment channel's information to 0.000 and reconstruction error up 57-72%,
# tripping the void condition declared before that sweep ran. So only the
# baseline-side intervention is used here.
#
# It also corrected a five-seed artefact: pc_easy at budget 256 gave the modular
# arm 5 of 5 wins, which the paper cited as evidence the rate gap alone produces
# the effect's sign. At 20 seeds it is 12/20, p = 0.154 -- null, as a negative
# control should be. That claim is withdrawn.
#
# What is still open is the one that matters: Sec V-A's +0.00397 at d = 0.41 on
# the confirmatory axis was measured with the modular arm carrying about eleven
# times the baseline's per-frame rate. This runs the same axis with the rate
# matched.
#
# DECLARED BEFORE RUNNING, and reported whichever way it comes out:
#   * oakink_category, granularity oakink_category, held 5, min_chains 4,
#     budget 256 -- identical to runs/oakink_official_category.
#   * The same 70 declared seeds, so the comparison is like-for-like against
#     n = 69. No extension after a p value is seen, in either direction.
#   * Arms: perframe_rate (free-bits floor 0.186/dim) against modular. The
#     unmatched perframe arm is retrained alongside so the rate gap is measured
#     on this axis rather than assumed from pc_easy.
#   * VOID CONDITION: the baseline's realised KL must exceed 1.5 nats/frame. If
#     the floor does not lift the rate on this dataset the sweep says nothing.
#     Reconstruction error is reported but is NOT a void condition for the
#     baseline side -- relaxing a KL penalty is expected to lower it.
#   * Declared reading. If the modular advantage SURVIVES at matched rate, Sec
#     V-A is structural and the attribution goes back into the paper. If it
#     DISAPPEARS, Sec V-A was measuring channel rate and the modular claim
#     leaves the paper. If it shrinks but stays positive, both numbers are
#     reported and the effect is described as partly rate.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_category \
    --out runs/oakink_category_ratematched \
    --budgets 256 --kinds perframe perframe_rate modular \
    --held-compositions 5 --min-chains 4 \
    --epochs 120 --window 32 --stride 4 --batch-size 256 \
    --lr 0.001 --latent-dim 12 --hidden 256 --n-primitives 12 \
    --seeds $(seq -s' ' 0 69) --fresh \
    > runs/oakink_category_ratematched_log.txt 2>&1
echo "[rate-conf] done at $(date)"

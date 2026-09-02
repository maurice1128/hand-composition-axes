#!/bin/sh
# A seventh screen point, because the sixth turned out to be invalid.
#
# runs/dexycb_paired_v2 ran at --min-chains 1, the disabling value, because all
# 200 of DexYCB's transitions occur in a single fine group and the pool the
# restriction needs is empty. check_informed_coverage.py returns TOO THIN on
# that configuration -- the informed arm sees 2.00 of 4 held compositions --
# which is the condition Table I calls "pins the penalty at zero by
# construction". Probing every reachable alternative
# (fine x min_chains 3/4/5, held 2/3/4) raises
# "0 transitions appear in >= N distinct groups (of 200 total)"; `left` and
# `right` pass only because they collapse a factor to "any", which is not a
# composition. DexYCB cannot support a paired composition sweep at all, so it
# leaves the screen correlation, taking it from
#
#     six axes   r = 0.863  p = 0.027      to     five axes  r = 0.843  p = 0.073
#
# This adds a point back. `oakink_class` is OakInk's own coarse functional class
# (container, maniptools, 5 groups) crossed with intent -- on the taxonomy Sec IV
# endorses, screened at excess +0.0061 / z = 4.1, never swept, and it passes the
# coverage gate at 8.0 examples per held composition at budget 64 and 32.0 at
# 256.
#
# DECLARED BEFORE RUNNING, and reported whichever way it comes out:
#   * 12 seeds, budget 256, held 4, min_chains 4. Twelve is what the other two
#     low-excess points (dexycb_fine, oakink2_fine) used; the count is fixed
#     here and will not be extended after a p value is seen.
#   * The screen correlation is recomputed on five points and on six, and both
#     are reported.
#   * It is a fourth OakInk-Image axis, so it does NOT fix the independence
#     problem Sec VI states -- three of six sharing one intent factor becomes
#     four of six. That is a cost, and the paper says so rather than presenting
#     the new point as if it were independent evidence.
#   * Declared reading. The screen predicts a LOW penalty here (+0.0061 excess is
#     the second smallest positive in the table). If the sweep returns a large
#     penalty, the screen is wrong on a point it scored, and that is a finding
#     against the paper's one surviving contribution.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

$PY -u scripts/check_informed_coverage.py \
    --bundle data/bundles/oakink.npz --granularity oakink_class \
    --held-compositions 4 --min-chains 4 --seeds 12 \
    > runs/gates/cover_oakink_class.txt 2>&1
tail -2 runs/gates/cover_oakink_class.txt

$PY -u scripts/check_composition_leak.py \
    --bundle data/bundles/oakink.npz --granularity oakink_class \
    --held-compositions 4 --min-per-composition 4 --seeds 12 \
    > runs/gates/composition_leak_oakink_class.txt 2>&1
tail -2 runs/gates/composition_leak_oakink_class.txt

$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_class \
    --out runs/oakink_class_v1 \
    --budgets 256 --kinds perframe modular \
    --held-compositions 4 --min-chains 4 \
    --epochs 120 --window 32 --stride 4 --batch-size 256 \
    --lr 0.001 --latent-dim 12 --hidden 256 --n-primitives 12 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --fresh \
    > runs/oakink_class_v1_log.txt 2>&1
echo "[screen7] sweep done at $(date)"
tail -8 runs/oakink_class_v1_log.txt

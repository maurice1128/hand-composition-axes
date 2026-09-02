#!/bin/sh
# Everything the audits showed the paper still needs, in dependency order.
#
# Run before auditing the draft again. Three of the four change numbers already
# in the paper, so reviewing prose against them now would mean reviewing it
# twice.
#
# Order is by what a reviewer asks first, not by cost:
#
# 1. The monolithic baseline, to schedule. The task-level comparison currently
#    on record used priors that stopped at 32-33 of 200 epochs, so it measures
#    how far each model trained rather than what its architecture buys. This is
#    the experiment a reviewer wants most and the one the paper presently has to
#    disown.
# 2. A confirmatory affordance sweep on seeds the paper has never seen. The
#    existing 130 are contaminated by optional stopping and cannot be cleaned
#    retroactively; a pre-declared batch can stand beside them as the
#    confirmatory test, with the old ones labelled exploratory.
# 3. Two more axes for the screen correlation, which rests on five usable
#    points. Both are predicted by the screen -- oakink_class weakly positive,
#    GRAB fine null -- and are run to test the screen, not to help it.
#
# Declared before running, per docs/AUDIT_2026-08-28.md: every one of these is
# reported whatever it returns. The affordance batch is 60 seeds and stops at
# 60. If the two new axes fall off the correlation line, the screen is weakened
# and the paper says so.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

echo "[1/4] monolithic priors, full 200-epoch schedule"
for s in 1 2 3 4; do
  $PY -u scripts/train_prior.py --model monolithic \
      --bundle data/bundles/oakink2.npz --run-dir "runs/mono_full_s$s" --fresh \
      --set train.epochs=200 train.save_every=99999 loader.stride=8 \
            train.seed=$s model.per_frame_latent=true \
      > "runs/mono_full_s${s}_log.txt" 2>&1 &
done
wait
echo "[1/4] done at $(date)"

echo "[2/4] task-level comparison against the properly trained baseline"
$PY -u scripts/experiment_grasp_modular_vs_mono.py \
    --mono runs/mono_full_s1 runs/mono_full_s2 runs/mono_full_s3 runs/mono_full_s4 \
    --difficulty medium --n-clips 40 \
    --out runs/grasp_modular_vs_mono_full.json \
    > runs/grasp_modular_vs_mono_full_log.txt 2>&1
echo "[2/4] done at $(date)"

echo "[3/4] confirmatory affordance sweep, seeds 130-189, declared in advance"
$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_attr \
    --out runs/oakink_attr_confirm --budgets 256 --kinds perframe modular \
    --held-compositions 5 --min-chains 4 --min-per-composition 5 \
    --epochs 120 --seeds $(seq 130 189) \
    > runs/oakink_attr_confirm_log.txt 2>&1
echo "[3/4] done at $(date)"

echo "[4/4] two further screen axes"
$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_class \
    --out runs/oakink_class_sweep --budgets 256 --kinds perframe modular \
    --held-compositions 5 --min-chains 4 --min-per-composition 5 \
    --epochs 120 --seeds $(seq 0 39) \
    > runs/oakink_class_sweep_log.txt 2>&1
$PY -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/grab.npz --granularity fine \
    --out runs/grab_fine_sweep --budgets 256 --kinds perframe modular \
    --held-compositions 5 --min-chains 4 --min-per-composition 5 \
    --epochs 120 --seeds $(seq 0 39) \
    > runs/grab_fine_sweep_log.txt 2>&1
echo "[4/4] done at $(date)"

echo
echo "All four complete. Re-run in this order before touching the draft:"
echo "  $PY scripts/analyse_screen_prediction.py"
echo "  $PY scripts/analyse_equivalence.py"
echo "  $PY scripts/pool_identity.py"

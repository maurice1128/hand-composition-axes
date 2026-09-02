#!/bin/sh
# The confirmatory task-level comparison, at a budget fixed before it runs.
#
# The exploratory run (runs/grasp_modular_vs_mono_pf.json) gave modular 35.6%
# against monolithic 26.9%, Fisher p = 0.117 at 160 clips per arm. The direction
# matches the reconstruction result and the sample does not settle it: at a nine
# point difference around 30%, 160 per arm is well under the ~450 needed for 80%
# power.
#
# Extending that run would be optional stopping -- its p value has been seen.
# This is a separate confirmatory test instead, on the same four models per arm
# but with fresh clips drawn from a different sampling seed.
#
# DECLARED BEFORE RUNNING, and reported whatever it returns:
#   * 120 clips per model, four models per arm, so 480 per arm.
#   * Sampling seed 1000, disjoint from the exploratory run's seed 0.
#   * Medium difficulty, the level where the task discriminates.
#   * Fisher exact on the pooled 480-vs-480 table. No interim looks, no
#     extension, whichever way it comes out.
#   * The exploratory 160-per-arm figure stays in the paper, labelled
#     exploratory, next to whatever this returns.
#
# If this is null, the honest reading is that modularity's reconstruction
# advantage does not reach grasp retention at a size this task can detect. That
# is the sentence the paper has been unable to write in either direction, and it
# is worth writing.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

$PY -u scripts/experiment_grasp_modular_vs_mono.py \
    --mono runs/mono_pf_s1 runs/mono_pf_s2 runs/mono_pf_s3 runs/mono_pf_s4 \
    --difficulty medium --n-clips 120 --seed 1000 \
    --out runs/grasp_modular_vs_mono_confirm.json \
    > runs/grasp_modular_vs_mono_confirm_log.txt 2>&1
echo "[confirm] done at $(date)"
tail -8 runs/grasp_modular_vs_mono_confirm_log.txt

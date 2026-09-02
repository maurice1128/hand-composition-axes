#!/bin/sh
# The 17 seeds the confirmatory axis declared and never ran.
#
# runs/oakink_official_category asked for seeds 0-69 and stopped after 53.
# Nothing was dropped after scoring and no result was inspected before the stop,
# but the declared budget was not honoured, and truncation without a rule is the
# same defect as extension without one. This runs the remainder.
#
# Declared before running: seeds 53-69 go into runs/oakink_category_rest, the
# pooled result over all seeds that produce both arms is what the paper reports,
# and it is reported whatever it shows. If pooling moves the category axis below
# significance, that is the result.
set -e
cd "$(dirname "$0")/.."
./.venv/Scripts/python.exe -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_category \
    --out runs/oakink_category_rest --budgets 256 --kinds perframe modular \
    --held-compositions 5 --min-chains 4 --min-per-composition 5 \
    --epochs 120 --seeds 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \
    > runs/oakink_category_rest_log.txt 2>&1
echo "[category] budget completed at $(date)"

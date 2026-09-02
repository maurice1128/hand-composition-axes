#!/bin/sh
# One shard of the confirmatory rate-matched control.
#
# Identical design to scripts/rate_matched_confirmatory.sh -- same axis, same
# gates, same arms, same 70 declared seeds -- split across processes purely for
# wall-clock. Each seed's paired comparison is self-contained: the seed fixes
# the split, so sharding by seed gives bit-identical results to running them in
# one process. Nothing about the pre-registration changes.
#
# usage: sh scripts/rate_matched_shard.sh <tag> <seed> [<seed> ...]
set -e
cd "$(dirname "$0")/.."
TAG="$1"; shift
./.venv/Scripts/python.exe -u scripts/experiment_paired_composition.py \
    --bundle data/bundles/oakink.npz --granularity oakink_category \
    --out "runs/ratematch_$TAG" \
    --budgets 256 --kinds perframe perframe_rate modular \
    --held-compositions 5 --min-chains 4 \
    --epochs 120 --window 32 --stride 4 --batch-size 256 \
    --lr 0.001 --latent-dim 12 --hidden 256 --n-primitives 12 \
    --seeds "$@" --fresh \
    > "runs/ratematch_${TAG}_log.txt" 2>&1
echo "[shard $TAG] done at $(date)"

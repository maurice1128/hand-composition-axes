#!/bin/sh
# Wait for the GPU to free up, then run the OakInk2 pipeline alone.
#
# Five sweeps sharing one 8 GB card made each of them slower than running them
# in sequence would have been, and the OakInk2 sweep -- the one that matters --
# was the biggest and got the least. This waits for the others to finish rather
# than competing with them.
cd "$(dirname "$0")/.."
while [ "$(ls runs/*/results.json 2>/dev/null | xargs -I{} sh -c 'test $(find {} -newermt "-4 minutes" | wc -l) -gt 0 && echo busy' | wc -l)" -gt 0 ]; do
  sleep 60
done
exec ./.venv/Scripts/python.exe -u scripts/run_oakink2_pipeline.py \
  --held-compositions 6 --min-per-composition 5 --min-chains 6 \
  --budgets 256 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --epochs 120 --stride 16

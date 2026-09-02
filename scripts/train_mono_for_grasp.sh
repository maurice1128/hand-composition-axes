#!/bin/sh
# Four bandwidth-matched monolithic priors on the same data as the modular bank,
# for the grasp comparison.
#
# `--model monolithic` with `model.per_frame_latent=true` is the same
# `perframe` baseline the paired sweeps use: a single continuous latent, but
# emitted every frame so the comparison isolates modularity rather than the
# number of bits reaching the decoder.
#
# These have to be retrained because the paired sweeps now delete their weights
# after scoring -- a change made to stop 15 GB per sweep filling the system
# drive, which removed exactly the checkpoints this comparison needs. The trade
# was still right: results.json is what a sweep is for, and four models is an
# hour.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

for s in 1 2 3 4; do
  $PY -u scripts/train_prior.py --model monolithic \
      --bundle data/bundles/oakink2.npz --run-dir "runs/mono_s$s" --fresh \
      --set train.epochs=200 train.save_every=99999 loader.stride=8 \
            train.seed=$s model.per_frame_latent=true \
      > "runs/mono_s${s}_log.txt" 2>&1 &
done
wait
echo "[mono] trained at $(date)"

$PY -u scripts/experiment_grasp_modular_vs_mono.py \
    --mono runs/mono_s1 runs/mono_s2 runs/mono_s3 runs/mono_s4 \
    --difficulty medium --n-clips 40

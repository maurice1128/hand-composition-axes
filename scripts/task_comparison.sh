#!/bin/sh
# The task-level comparison, on a baseline that is actually the baseline.
#
# Two previous attempts are void. Both trained the monolithic arm to a fraction
# of its schedule, and both carried a window-level latent rather than the
# per-frame one Sec III-B argues is required -- `scripts/train_prior.py` parsed
# `model.per_frame_latent` and then never passed it to `PriorConfig`, so the run
# directories recorded a setting that had never applied. That is fixed; the
# assertion below refuses to train if it regresses.
#
# This is the only experiment in the project that asks whether modularity buys
# anything beyond reconstruction error. Both previous answers were "the
# comparison is void", which is not an answer.
#
# Declared before running: four seeds per arm, 200 epochs, medium difficulty
# (the level where the task discriminates -- easy ceilings near 90% and hard
# floors near 0%), 40 clips per model. The result is reported whatever it shows.
# A null here says modularity's measured advantage does not reach grasp
# retention, and that is worth reporting as plainly as a positive would be.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

$PY - <<'CHECK'
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from train_prior import build
m = build("monolithic", {"model": {"per_frame_latent": True}}, 32)
assert m.cfg.per_frame_latent is True, "per_frame_latent still does not reach the model"
print("[check] per_frame_latent reaches the model")
CHECK

for s in 1 2 3 4; do
  $PY -u scripts/train_prior.py --model monolithic \
      --bundle data/bundles/oakink2.npz --run-dir "runs/mono_pf_s$s" --fresh \
      --set train.epochs=200 train.save_every=99999 loader.stride=8 \
            train.seed=$s model.per_frame_latent=true \
      > "runs/mono_pf_s${s}_log.txt" 2>&1 &
done
wait
echo "[task] baselines trained at $(date)"

$PY - <<'VERIFY'
import sys, pathlib
sys.path.insert(0, "src")
from caredex.train.checkpoint import CheckpointManager
for s in (1, 2, 3, 4):
    d = pathlib.Path(f"runs/mono_pf_s{s}")
    cfg = CheckpointManager(d).peek(d / "best.pt").get("config", {}).get("model", {})
    assert cfg.get("per_frame_latent") is True, f"{d.name} trained window-level"
    print(f"[verify] {d.name}: per_frame_latent={cfg.get('per_frame_latent')}")
VERIFY

$PY -u scripts/experiment_grasp_modular_vs_mono.py \
    --mono runs/mono_pf_s1 runs/mono_pf_s2 runs/mono_pf_s3 runs/mono_pf_s4 \
    --difficulty medium --n-clips 40 \
    --out runs/grasp_modular_vs_mono_pf.json \
    > runs/grasp_modular_vs_mono_pf_log.txt 2>&1
echo "[task] comparison done at $(date)"
tail -6 runs/grasp_modular_vs_mono_pf_log.txt

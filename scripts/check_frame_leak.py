"""Frame-level leak check for the concatenated OakInk-Image ablation bundles.

`check_composition_leak.py` compares fine *labels* only. The long-clip bundles build every trajectory out of 8
reused source clips, so a target trajectory's exact frames can sit inside training trajectories that carry a
different fine label. This script rebuilds each split exactly as `experiment_paired_composition.py` does and
measures, per arm, the fraction of target frames that appear verbatim in that arm's training trajectories.

A paired penalty is only interpretable if the two arms see target frames at about the same rate; an informed arm
that sees many more of them is being scored on memorised frames.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import experiment_paired_composition as E  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402

ALL_CASES = [
    ("unmodified oakink", "oakink_official_category", "data/bundles/oakink.npz"),
    ("long, misaligned", "pc_oakink_long", "data/bundles/oakink_long.npz"),
    ("long, aligned", "pc_oakink_longaligned", "data/bundles/oakink_longaligned.npz"),
    ("oakink2 whole", "oakink2_scene_primitive", "data/bundles/oakink2_scene_primitive.npz"),
    ("oakink2 primseg", "oakink2_primseg_s4", "data/bundles/oakink2_primseg.npz"),
    ("grab grasp-only", "grab_grasp_s32", "data/bundles/grab_grasp.npz"),
    ("oakink dofdamage", "pc_oakink_dofdamage", "data/bundles/oakink_dofdamage.npz"),
    ("oakink sparse", "pc_oakink_sparse", "data/bundles/oakink_sparse.npz"),
    ("oakink cat x subject", "oakink_category_subject", "data/bundles/oakink_category_subject.npz"),
]
# Optional: pass case names as arguments to run only those.
CASES = [c for c in ALL_CASES if len(sys.argv) < 2 or c[0] in sys.argv[1:]]


def frame_keys(traj):
    # Exact row identity; rounding guards against -0.0 vs 0.0 only.
    return {np.round(row, 5).tobytes() for row in np.asarray(traj, dtype=np.float32)}


out = {}
for name, run, bundle_path in CASES:
    args = json.loads((ROOT / "runs" / run / "results.json").read_text(encoding="utf-8"))["args"]
    bundle = TrajectoryBundle.load(str(ROOT / bundle_path))
    fine = list(bundle.labels)
    bundle.labels = E.coarsen_labels(fine, args["granularity"])
    budget = args["budgets"][0]
    rows = []
    for seed in range(5):
        split = E.build_paired_split(bundle, args["held_compositions"], seed, fine_labels=fine,
                                     min_chains=args["min_chains"])
        naive, informed = E.sample_pools(split, budget, seed, bundle.labels, args["min_per_composition"])
        target_frames = set()
        for i in split["target"]:
            target_frames |= frame_keys(bundle.trajectories[i])
        naive_frames, informed_frames = set(), set()
        for i in naive:
            naive_frames |= frame_keys(bundle.trajectories[i])
        for i in informed:
            informed_frames |= frame_keys(bundle.trajectories[i])
        n = len(target_frames)
        rn = len(target_frames & naive_frames) / n
        ri = len(target_frames & informed_frames) / n
        rows.append((seed, n, rn, ri))
        print(f"{name:18s} seed {seed}: {n:6d} distinct target frames | in naive train {100*rn:5.1f}% | "
              f"in informed train {100*ri:5.1f}%")
    mn = float(np.mean([r[2] for r in rows]))
    mi = float(np.mean([r[3] for r in rows]))
    print(f"{name:18s} MEAN: naive {100*mn:5.1f}%  informed {100*mi:5.1f}%  gap {100*(mi-mn):+5.1f} points\n")
    out[name] = {"per_seed": rows, "naive_mean": mn, "informed_mean": mi}

(ROOT / "runs" / "frame_leak_long_ablations.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

"""Frame-level leak check, parameterised (copy of check_frame_leak.py).

Same logic as `scripts/check_frame_leak.py`: rebuild each split exactly as
`experiment_paired_composition.py` does and measure, per arm, the fraction of
distinct target frames that appear verbatim in that arm's training
trajectories. Two changes only: the bundle and split settings are arguments
instead of being read from a finished run's results.json, and the output path
is an argument instead of `runs/frame_leak_long_ablations.json`.

    python scripts/check_frame_leak_b64.py --out runs/frame_leak_alignment_v2.json \
        --case oakink=data/bundles/oakink.npz --granularity oakink_category \
        --held-compositions 5 --min-chains 4 --min-per-composition 5 --budget 64

Exit code 1 if any case's per-seed leak in either arm exceeds --max-leak
(default 0.01, the threshold in runs/PREREG_oakink_alignment_v2.md).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import experiment_paired_composition as E  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402


def frame_keys(traj):
    # Exact row identity; rounding guards against -0.0 vs 0.0 only.
    return {np.round(row, 5).tobytes() for row in np.asarray(traj, dtype=np.float32)}


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--case", action="append", required=True, help="name=bundle_path")
ap.add_argument("--granularity", required=True)
ap.add_argument("--held-compositions", type=int, required=True)
ap.add_argument("--min-chains", type=int, required=True)
ap.add_argument("--min-per-composition", type=int, required=True)
ap.add_argument("--budget", type=int, required=True)
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
ap.add_argument("--max-leak", type=float, default=0.01)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = {"settings": {k: v for k, v in vars(args).items() if k not in ("case", "out")}}
worst = 0.0
for case in args.case:
    name, _, bundle_path = case.partition("=")
    bundle = TrajectoryBundle.load(str(ROOT / bundle_path))
    fine = list(bundle.labels)
    bundle.labels = E.coarsen_labels(fine, args.granularity)
    budget = args.budget
    rows = []
    for seed in args.seeds:
        split = E.build_paired_split(bundle, args.held_compositions, seed, fine_labels=fine,
                                     min_chains=args.min_chains)
        naive, informed = E.sample_pools(split, budget, seed, bundle.labels, args.min_per_composition)
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
        worst = max(worst, rn, ri)
        print(f"{name:26s} seed {seed}: {n:6d} distinct target frames | in naive train {100*rn:6.2f}% | "
              f"in informed train {100*ri:6.2f}%")
    mn = float(np.mean([r[2] for r in rows]))
    mi = float(np.mean([r[3] for r in rows]))
    mx = float(max(max(r[2], r[3]) for r in rows))
    print(f"{name:26s} MEAN: naive {100*mn:6.2f}%  informed {100*mi:6.2f}%  gap {100*(mi-mn):+6.2f} points"
          f"  max {100*mx:.2f}%\n")
    out[name] = {"bundle": bundle_path, "per_seed": rows, "naive_mean": mn, "informed_mean": mi,
                 "max_any_seed_any_arm": mx, "pass": mx <= args.max_leak}

(ROOT / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"wrote {args.out}; worst {100*worst:.2f}% vs limit {100*args.max_leak:.2f}%")
raise SystemExit(0 if worst <= args.max_leak else 1)

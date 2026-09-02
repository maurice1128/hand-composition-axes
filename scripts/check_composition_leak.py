"""Does the informed model train on near-duplicates of the test clips?

The paired design compares a model that saw a composition against one that did
not. If the target trajectories' *exact fine composition* also sits in the
informed model's training set, the measured "compositional penalty" is partly
"did you train on a near-duplicate of the test clip" -- an advantage the naive
arm is structurally denied, and one that has nothing to do with composition.

An audit found exactly this: with coarse held-out cells and a random 50/50
split inside them, 48-69% of OakInk target trajectories had their exact
``object->intent`` pair in the informed training set, against 0% for naive.
``build_paired_split`` now assigns whole fine groups to one side, so the
overlap should be zero. This script is the gate that keeps it there.

    python scripts/check_composition_leak.py --bundle data/bundles/oakink.npz --granularity category
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from experiment_paired_composition import (  # noqa: E402
    build_paired_split,
    coarsen_labels,
    sample_pools,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--granularity", choices=("fine", "category", "shape", "oakink_category", "oakink_class", "oakink_attr", "left", "right"), default="category")
    ap.add_argument("--held-compositions", type=int, default=5)
    ap.add_argument("--min-per-composition", type=int, default=8)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max acceptable fraction of target trajectories whose fine "
                         "composition appears in the informed training set")
    args = ap.parse_args()

    bundle = TrajectoryBundle.load(args.bundle)
    fine = list(bundle.labels)
    if args.granularity != "fine":
        bundle.labels = coarsen_labels(bundle.labels, args.granularity)

    print(f"{args.bundle}  granularity={args.granularity}  budget={args.budget}")
    print(f"{'seed':>5} {'target':>7} {'naive leak':>11} {'informed leak':>14}  status")
    print("-" * 56)

    worst = 0.0
    for seed in args.seeds:
        split = build_paired_split(
            bundle, args.held_compositions, seed,
            fine_labels=fine, min_chains=args.min_chains,
        )
        naive_idx, informed_idx = sample_pools(
            split, args.budget, seed, bundle.labels, args.min_per_composition
        )
        target_fine = {fine[i] for i in split["target"]}

        leaks = {}
        for name, idx in (("naive", naive_idx), ("informed", informed_idx)):
            train_fine = {fine[i] for i in idx}
            hit = sum(1 for i in split["target"] if fine[i] in train_fine)
            leaks[name] = hit / max(len(split["target"]), 1)

        worst = max(worst, leaks["informed"], leaks["naive"])
        ok = max(leaks.values()) <= args.tolerance + 1e-9
        print(
            f"{seed:>5} {len(split['target']):>7} {leaks['naive']:>10.1%} "
            f"{leaks['informed']:>13.1%}  {'ok' if ok else 'LEAK'}"
        )
        # A target trajectory sharing a fine composition with another target
        # trajectory is fine; sharing with a TRAINING trajectory is not.
        assert target_fine, "empty target set"

    print("-" * 56)
    if worst > args.tolerance + 1e-9:
        print(
            f"FAIL: up to {worst:.1%} of target trajectories have their exact fine\n"
            f"composition in a training set. The paired penalty then measures\n"
            f"near-duplicate memorisation, not compositional generalisation.\n"
            f"Pass fine_labels to build_paired_split, or hold out fine cells directly."
        )
        return 1
    print(f"PASS: no target trajectory's fine composition appears in either training set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

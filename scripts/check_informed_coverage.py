"""Is the INFORMED model actually informed?

The paired design compares a model trained without a composition against one
trained with it. That only works if the informed model genuinely sees each held
composition. ``sample_pools`` swaps in ``n // 4`` trajectories, which at budget
64 is 16 trajectories spread over as many as 60 held compositions -- under one
example each. If most held compositions are still absent from the informed
model's training set, the two models are nearly identical and a zero penalty is
guaranteed regardless of architecture.

A null result produced that way is an artefact, not a finding. This script
counts the coverage so the distinction can be settled with a number.

    python scripts/check_informed_coverage.py --bundle data/bundles/oakink.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from experiment_data_efficiency import transitions_of  # noqa: E402
from experiment_paired_composition import build_paired_split, sample_pools  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--held-compositions", type=int, default=8)
    ap.add_argument("--budgets", type=int, nargs="*", default=[64, 256])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--min-examples", type=float, default=3.0)
    ap.add_argument("--min-per-composition", type=int, default=4)
    ap.add_argument("--min-chains", type=int, default=4)
    ap.add_argument("--granularity",
                    choices=("fine", "category", "shape", "grab_shape_intentclass",
                 "grab_object_intentclass", "oakink_category", "oakink_class",
                 "oakink_attr", "oakink2_scene_primitive", "oakink2_subject_primitive",
                 "oakink2_scene_verb", "oakink2_subject_verb", "left", "right"),
                    default="fine")
    args = ap.parse_args()

    bundle = TrajectoryBundle.load(args.bundle)
    # Captured before coarsening, exactly as the experiment does, so the split
    # this gate inspects is the split the experiment will build.
    fine = list(bundle.labels)
    if args.granularity != "fine":
        from experiment_paired_composition import coarsen_labels

        bundle.labels = coarsen_labels(bundle.labels, args.granularity)
        cells = len(set(bundle.labels))
        print(f"granularity={args.granularity}: {cells} cells, "
              f"{len(bundle.labels) / max(cells, 1):.1f} trajectories per cell")
    print(f"{args.bundle}: {len(bundle.trajectories)} trajectories")
    print(f"holding out {args.held_compositions} compositions\n")
    print(
        f"{'seed':>5} {'budget':>7} {'swapped':>8} {'held':>6} "
        f"{'covered':>8} {'per comp':>9}  status"
    )
    print("-" * 62)

    worst = float("inf")
    # A row can fail on breadth (the informed arm never sees some held
    # composition) or on depth (it sees each one too few times). The summary
    # below used to test only depth, so a run whose every row printed TOO THIN
    # on breadth still exited 0 and printed "Coverage is adequate" -- which is
    # what happened on the DexYCB axis the paper withdraws. Both are tracked.
    thin_rows = 0
    total_rows = 0
    worst_covered = None
    for seed in args.seeds:
        # fine_labels and min_chains must match what the experiment uses, or the
        # gate checks a different split than the one that will actually run --
        # which it silently did until this was caught.
        split = build_paired_split(
            bundle, args.held_compositions, seed,
            fine_labels=fine, min_chains=args.min_chains,
        )
        held = set(split["held_compositions"])
        for budget in args.budgets:
            if budget > len(split["naive_pool"]):
                continue
            naive, informed = sample_pools(
                split, budget, seed, bundle.labels, args.min_per_composition
            )
            swapped = [i for i in informed if i not in set(naive)]

            seen = Counter()
            for i in informed:
                for a, b in transitions_of(bundle.labels[i]):
                    if f"{a}->{b}" in held:
                        seen[f"{a}->{b}"] += 1

            covered = len(seen)
            per_comp = np.mean(list(seen.values())) if seen else 0.0
            ok = covered / max(len(held), 1) > 0.8 and per_comp >= args.min_examples
            worst = min(worst, per_comp)
            total_rows += 1
            thin_rows += 0 if ok else 1
            worst_covered = covered if worst_covered is None else min(worst_covered, covered)
            print(
                f"{seed:>5} {budget:>7} {len(swapped):>8} {len(held):>6} "
                f"{covered:>8} {per_comp:>9.2f}  {'ok' if ok else 'TOO THIN'}"
            )
        print()

    print("-" * 62)
    if thin_rows:
        if worst < args.min_examples:
            why = (
                "the informed model sees only "
                f"{worst:.2f} example(s) per held composition on average"
            )
        else:
            why = (
                f"depth is adequate ({worst:.2f} examples per covered composition) "
                f"but breadth is not, some rows reaching only {worst_covered} of the "
                "held compositions"
            )
        print(
            f"ARTEFACT RISK: {thin_rows} of {total_rows} rows fail, and {why}."
        )
        print(
            "The informed model is barely more informed than the naive one there, so"
        )
        print(
            "a zero penalty says nothing about compositional generalisation. Fix by"
        )
        print(
            "holding out FEWER compositions and swapping in MORE trajectories each."
        )
        return 1
    print("Coverage is adequate; a zero penalty would be a real finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

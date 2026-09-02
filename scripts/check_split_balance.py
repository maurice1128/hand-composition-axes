"""Are the seen/unseen test sets comparable in anything except composition?

The compositional gap ``MSE(unseen) - MSE(seen)`` only measures composition if
the two sets are otherwise matched. They are assembled by holding out random
compositions, so nothing guarantees that. If unseen trajectories happen to be
slower or shorter, they are simply easier to reconstruct, and the gap measures
that instead -- which is exactly what a consistently NEGATIVE gap across every
model architecture indicates.

This script compares the two sets on quantities that drive reconstruction
difficulty, so the confound can be seen rather than argued about.

    python scripts/check_split_balance.py --bundle data/bundles/oakink.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import ARTICULATED_SLICE, normalize  # noqa: E402


def stats(bundle: TrajectoryBundle, idx: list[int], window: int) -> dict:
    """Difficulty proxies, computed on normalised units to match the loss."""
    speeds, ranges, variances, lengths = [], [], [], []
    for i in idx:
        t = normalize(bundle.trajectories[i])[:, ARTICULATED_SLICE]
        lengths.append(len(t))
        if len(t) < 2:
            continue
        d = np.abs(np.diff(t, axis=0))
        speeds.append(d.mean())
        ranges.append((t.max(axis=0) - t.min(axis=0)).mean())
        variances.append(t.var(axis=0).mean())
    return {
        "n_trajectories": len(idx),
        "mean_length": float(np.mean(lengths)),
        # A model predicting a constant scores well on a low-variance
        # trajectory, so variance is the dominant difficulty proxy for MSE.
        "mean_variance": float(np.mean(variances)),
        "mean_frame_delta": float(np.mean(speeds)),
        "mean_range": float(np.mean(ranges)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_data_efficiency import compositional_split

    bundle = TrajectoryBundle.load(args.bundle)
    print(f"{args.bundle}: {len(bundle.trajectories)} trajectories\n")
    print(f"{'seed':>5} {'set':>8} {'n':>5} {'len':>8} {'variance':>10} {'delta':>9} {'range':>8}")
    print("-" * 60)

    diffs = []
    for seed in args.seeds:
        _, seen_idx, unseen_idx, _ = compositional_split(bundle, args.holdout_frac, seed)
        s = stats(bundle, seen_idx, args.window)
        u = stats(bundle, unseen_idx, args.window)
        for name, d in (("seen", s), ("unseen", u)):
            print(
                f"{seed:>5} {name:>8} {d['n_trajectories']:>5} {d['mean_length']:>8.1f} "
                f"{d['mean_variance']:>10.5f} {d['mean_frame_delta']:>9.5f} {d['mean_range']:>8.4f}"
            )
        rel = (u["mean_variance"] - s["mean_variance"]) / max(s["mean_variance"], 1e-9)
        diffs.append(rel)
        print(f"{'':>5} {'->':>8} unseen variance is {rel:+.1%} vs seen\n")

    mean_diff = float(np.mean(diffs))
    print("-" * 60)
    print(f"mean relative variance difference across {len(args.seeds)} seeds: {mean_diff:+.1%}")
    if abs(mean_diff) > 0.05:
        print(
            "\nCONFOUNDED. The two test sets differ systematically in the quantity that\n"
            "drives reconstruction MSE, so MSE(unseen) - MSE(seen) is not a measurement\n"
            "of compositional generalisation. Use the PAIRED design instead: evaluate the\n"
            "same held-out trajectories under a model that saw their composition and one\n"
            "that did not, so set difficulty cancels."
        )
        return 1
    print("\nBalanced enough for the unpaired gap to be interpretable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

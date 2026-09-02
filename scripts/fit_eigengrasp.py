"""Fit the eigengrasp (PCA) basis and report the variance curve.

The output answers two questions the paper will be asked:
  * how many latent dimensions does hand posture actually need?
  * does the sequence VAE beat a linear projection at all?

    python scripts/fit_eigengrasp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.config import get, load_config  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import ARTICULATED_SLICE, denormalize, normalize  # noqa: E402
from caredex.models.eigengrasp import EigengraspBasis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    ap.add_argument("--bundle", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    bundle_path = Path(args.bundle or get(cfg, "data.cache"))
    if not bundle_path.exists():
        print(f"no bundle at {bundle_path}; run scripts/generate_synthetic.py first")
        return 1

    bundle = TrajectoryBundle.load(bundle_path)
    poses = bundle.stacked()
    normalised = normalize(poses)

    articulated_only = bool(get(cfg, "eigengrasp.articulated_only", True))
    basis = EigengraspBasis.fit(normalised, articulated_only=articulated_only)

    print(f"fitted on {len(poses):,} frames from {bundle_path}")
    print(f"articulated_only={articulated_only}  dim={basis.dim}\n")
    print(basis.variance_table())

    print("\ncomponents needed to reach a variance target:")
    for target in (0.90, 0.95, 0.99, 0.999):
        print(f"  {target:>6.1%} -> {basis.n_components_for(target):>3d} of {basis.dim}")

    expected = bundle.meta.get("expected_intrinsic_dim")
    if expected is not None:
        k95 = basis.n_components_for(0.95)
        print(
            f"\ngenerator's expected intrinsic dim: {expected}  |  PCA at 95%: {k95}"
        )
        if k95 > expected + 3:
            print("  WARNING: PCA needs far more components than the generator used.")

    print("\nreconstruction error vs component count (degrees, articulated DOF):")
    print(f"{'k':>4}  {'rmse':>8}  {'p95':>8}  {'max':>8}")
    ref = poses[..., ARTICULATED_SLICE]
    for k in _ladder(basis.dim):
        rec_native = denormalize(basis.reconstruct(normalised, k))
        err = np.abs(rec_native[..., ARTICULATED_SLICE] - ref)
        rmse = float(np.sqrt((err**2).mean()))
        print(f"{k:>4}  {rmse:>8.3f}  {np.percentile(err, 95):>8.3f}  {err.max():>8.3f}")

    out = Path(get(cfg, "eigengrasp.out", "runs/eigengrasp/basis.npz"))
    basis.save(out)
    print(f"\nsaved basis -> {out}")

    target = float(get(cfg, "eigengrasp.variance_target", 0.95))
    print(
        f"\nsuggested model.latent_dim = {basis.n_components_for(target)} "
        f"(PCA components at {target:.0%} variance)"
    )
    return 0


def _ladder(dim: int) -> list[int]:
    return sorted({k for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, dim) if k <= dim})


if __name__ == "__main__":
    raise SystemExit(main())

"""Sanity check: does the primitive bank actually get used?

The first run of the data-efficiency experiment reported ``primitives_used =
1.0``. A modular model that uses one primitive is a monolithic model with extra
steps, so every comparison against the monolithic baseline was void. This
script is the fast gate that has to pass before spending GPU time on the full
sweep again.

Pass condition: several primitives in use, and assignments that switch within a
window (segments, not one primitive per window).

    python scripts/check_primitive_collapse.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from caredex.models.modular_prior import ModularConfig, ModularPrimitivePrior  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/synthetic_big.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-trajectories", type=int, default=200)
    ap.add_argument("--n-primitives", type=int, default=12)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--min-primitives", type=float, default=3.0)
    args = ap.parse_args()

    bundle = TrajectoryBundle.load(args.bundle)
    trajs = bundle.trajectories[: args.n_trajectories]
    labels = bundle.labels[: args.n_trajectories]
    ds = WindowedTrajectoryDataset(trajs, args.window, stride=8, labels=labels)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModularPrimitivePrior(
        ModularConfig(window=args.window, n_primitives=args.n_primitives)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"{len(ds)} windows from {len(trajs)} trajectories, K={args.n_primitives}")
    print(f"{'ep':>4} {'recon':>9} {'prims':>6} {'switch':>7} {'balance':>8} {'usage_H':>8}")

    last = {}
    for ep in range(args.epochs):
        model.train()
        for batch in loader:
            loss, last = model.loss(batch.to(device), beta=min(1.0, ep / 5))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"{ep:>4} {last['recon']:>9.5f} {last['primitives_used']:>6.1f} "
                f"{last['switches_per_window']:>7.2f} {last['load_balance']:>8.3f} "
                f"{last['usage_entropy']:>8.3f}"
            )

    used = last.get("primitives_used", 0.0)
    switches = last.get("switches_per_window", 0.0)
    ok = used >= args.min_primitives and switches > 0.5

    print()
    if ok:
        print(
            f"PASS: {used:.0f} of {args.n_primitives} primitives in use, "
            f"{switches:.2f} switches per window -- the bank is a library, not a single mode."
        )
    else:
        print(
            f"FAIL: primitives_used={used:.1f}, switches_per_window={switches:.2f}. "
            "The bank has collapsed; the modular model is monolithic in disguise and "
            "any comparison against the monolithic baseline would be void. Raise "
            "load_balance_weight / usage_entropy_weight, or lower "
            "assignment_smooth_weight, before running the sweep."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

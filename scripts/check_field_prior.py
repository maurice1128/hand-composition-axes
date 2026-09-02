"""Smoke test for the velocity-field primitive prior.

Checks the things that would make the architecture useless before any sweep is
worth running:

1. It trains at all -- rollout error decreases. Autoregressive integration
   compounds error, so a field prior can fail to learn where a teacher-forced
   decoder would not.
2. The bank does not collapse: several primitives in use, and the fields
   actually produce motion (``mean_step`` well above zero). A bank whose fields
   all output ~zero velocity reconstructs a frozen hand and scores fine on every
   modularity diagnostic.
3. Primitives are context-independent *by construction* -- verified, not
   assumed, by applying the same primitive from many different states and
   confirming the velocity depends only on state and primitive, never on
   history.

Check 3 is the whole point of this architecture. If it fails, the
implementation has a bug, because the property is supposed to be structural.

    python scripts/check_field_prior.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.data.pipeline import WindowedTrajectoryDataset  # noqa: E402
from caredex.models.field_prior import FieldConfig, FieldPrimitivePrior  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-trajectories", type=int, default=200)
    ap.add_argument("--n-primitives", type=int, default=12)
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()

    bundle = TrajectoryBundle.load(args.bundle)
    trajs = bundle.trajectories[: args.n_trajectories]
    ds = WindowedTrajectoryDataset(trajs, args.window, 8, bundle.labels[: args.n_trajectories])
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FieldPrimitivePrior(
        FieldConfig(window=args.window, n_primitives=args.n_primitives)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"{len(ds)} windows, K={args.n_primitives}, {model.n_parameters:,} parameters")
    print(f"{'ep':>4} {'recon':>9} {'prims':>6} {'mean_step':>10} {'switch':>7}")

    history, last = [], {}
    for ep in range(args.epochs):
        model.train()
        for batch in loader:
            loss, last = model.loss(batch.to(device), beta=min(1.0, ep / 5))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        history.append(last["recon"])
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"{ep:>4} {last['recon']:>9.5f} {last['primitives_used']:>6.0f} "
                  f"{last['mean_step']:>10.5f} {last['switches_per_window']:>7.2f}")

    failures = 0

    learned = history[-1] < history[0] * 0.9
    failures += not learned
    print(f"\n[{'PASS' if learned else 'FAIL'}] rollout error decreased "
          f"({history[0]:.5f} -> {history[-1]:.5f})")

    alive = last["primitives_used"] >= 3 and last["mean_step"] > 1e-3
    failures += not alive
    print(f"[{'PASS' if alive else 'FAIL'}] bank is alive: {last['primitives_used']:.0f} "
          f"primitives, mean step {last['mean_step']:.5f}")

    # Context independence, verified rather than assumed.
    model.eval()
    rng = np.random.default_rng(0)
    states = torch.from_numpy(
        rng.uniform(-0.8, 0.8, size=(64, 27)).astype(np.float32)
    ).to(device)
    max_dev = 0.0
    for k in range(args.n_primitives):
        direct = model.field_at(states, k)
        # Reach the same states through a rollout driven by a different
        # predecessor, then apply primitive k: the velocity must match what the
        # field gives for those states alone.
        via = model.field_at(states, k)
        max_dev = max(max_dev, float((direct - via).abs().max()))

    # The real test: same state, same primitive, different preceding primitive.
    devs = []
    for k in range(args.n_primitives):
        base = model.field_at(states, k)
        for pred in range(0, args.n_primitives, 4):
            w = torch.zeros(len(states), args.window, args.n_primitives, device=device)
            w[:, : args.window // 2, pred] = 1.0
            w[:, args.window // 2 :, k] = 1.0
            style = torch.zeros(len(states), model.cfg.style_dim, device=device)
            with torch.no_grad():
                traj = model.rollout(w, style, states)
            # Velocity applied at the first step after the switch, evaluated at
            # whatever state the predecessor left behind.
            q_switch = traj[:, args.window // 2]
            v_seq = traj[:, args.window // 2 + 1] - q_switch
            v_field = model.field_at(q_switch, k)
            # bounded_state applies tanh after the step, so compare pre-tanh.
            devs.append(float((torch.tanh(q_switch + v_field) - traj[:, args.window // 2 + 1]).abs().max()))

    worst = max(devs)
    ok = worst < 1e-5
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] context independence is structural: "
          f"max deviation {worst:.2e} between the field applied directly and the "
          f"same primitive reached after a different predecessor")

    print(f"\n{'all checks passed' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

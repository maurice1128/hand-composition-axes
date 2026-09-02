"""How much does the prior actually constrain hand motion?

"The prior generates anatomically valid motion" is the kind of claim that
sounds strong and measures nothing, because the obvious metric is rigged. Joint
limit violations are **zero by construction** -- `bounded_output=True` makes
them architecturally impossible -- so a model, a real hand, and uniform noise
inside the limit box all score identically on it.

The metric with teeth is the DIP/PIP coupling residual. In a real hand the
distal joint is mechanically coupled to the proximal one at roughly 2/3, and
nothing in the model's objective enforces that. A first look gave:

    real OakInk2      6.78 deg
    generated         7.71 deg
    uniform random   29.70 deg

which says the prior's output sits near real motion on a property it was never
told about, and four times away from what "any pose inside the anatomical box"
looks like. That is the quantity worth reporting, and this makes it an
experiment rather than one terminal line: several models, several seeds,
several metrics, and a test.

The reference arm that matters is **uniform random inside the limit box**. It is
what an RL policy explores when it has no prior -- every one of those poses is
reachable and none of them looks like a hand doing something. The gap between
that arm and the generated arm is what the prior buys.

Nothing here claims the generated motion is *useful*. A valid motion can be a
meaningless wiggle. Usefulness needs a task, which no part of this project has.

    python scripts/experiment_motion_validity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import (  # noqa: E402
    LIMITS_HI, LIMITS_LO, N_DOF, coupling_residual,
)

from demo_composition import load_model  # noqa: E402
from figure_primitive_library import rollout  # noqa: E402


def metrics(x: np.ndarray, fps: float) -> dict[str, float]:
    """Per-clip properties, all in native units.

    ``x`` is ``(n_clips, T, 27)``. Velocity and jerk are finite differences in
    degrees per second; the coupling residual is the mean absolute deviation
    from the anatomical DIP/PIP ratio.
    """
    v = np.diff(x, axis=1) * fps
    a = np.diff(v, axis=1) * fps
    j = np.diff(a, axis=1) * fps
    return {
        "coupling_deg": float(np.abs(coupling_residual(x.reshape(-1, N_DOF))).mean()),
        "speed_deg_s": float(np.abs(v).mean()),
        "jerk_p99": float(np.percentile(np.abs(j), 99)),
        "limit_violation": float(np.mean((x < LIMITS_LO) | (x > LIMITS_HI))),
    }


def per_clip(x: np.ndarray, fps: float) -> np.ndarray:
    """Coupling residual per clip, so the arms can be compared with a test."""
    return np.array([
        float(np.abs(coupling_residual(c)).mean()) for c in x
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--models", nargs="*",
                    default=[f"runs/bricks_s{i}" for i in range(1, 13)])
    ap.add_argument("--n-clips", type=int, default=96)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/motion_validity.json")
    args = ap.parse_args()

    b = TrajectoryBundle.load(ROOT / args.bundle)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    long_enough = [np.asarray(t) for t in b.trajectories if len(t) >= args.window]
    picks = rng.choice(len(long_enough), min(args.n_clips, len(long_enough)), replace=False)
    real = np.stack([long_enough[i][: args.window] for i in picks])
    uniform = rng.uniform(LIMITS_LO, LIMITS_HI, size=real.shape)

    arms: dict[str, np.ndarray] = {"real": per_clip(real, b.fps),
                                   "uniform": per_clip(uniform, b.fps)}
    report = {
        "n_clips": len(real),
        "real": metrics(real, b.fps),
        "uniform": metrics(uniform, b.fps),
        "generated_per_model": {},
    }

    gen_all = []
    for run_dir in args.models:
        p = ROOT / run_dir
        if not (p / "best.pt").exists():
            continue
        model, _ = load_model(p, "best.pt", device)
        k = model.cfg.n_primitives
        pairs = rng.integers(0, k, (len(real), 2))
        gen = np.stack([rollout(model, [int(a), int(c)], device, seed=i)[: args.window]
                        for i, (a, c) in enumerate(pairs)])
        report["generated_per_model"][p.name] = metrics(gen, b.fps)
        gen_all.append(per_clip(gen, b.fps))
        print(f"  {p.name:<14} coupling {report['generated_per_model'][p.name]['coupling_deg']:6.2f} deg")

    if not gen_all:
        print("no trained model found")
        return 1
    arms["generated"] = np.concatenate(gen_all)

    print(f"\n{'arm':<12}{'n':>6}{'coupling residual (deg)':>26}")
    print("-" * 46)
    for name in ("real", "generated", "uniform"):
        v = arms[name]
        print(f"{name:<12}{len(v):>6}{v.mean():>16.2f} +- {v.std(ddof=1):.2f}")

    print()
    for a, c in (("generated", "uniform"), ("generated", "real")):
        t, p = stats.ttest_ind(arms[a], arms[c], equal_var=False)
        d = (arms[a].mean() - arms[c].mean()) / np.sqrt(
            (arms[a].var(ddof=1) + arms[c].var(ddof=1)) / 2)
        print(f"  {a} vs {c:<10} diff {arms[a].mean() - arms[c].mean():+7.2f} deg  "
              f"d = {d:+.2f}  p = {p:.2e}")

    print("\n  limit violation rate is 0 in every arm, including uniform noise:"
          "\n  bounded_output makes it architecturally impossible, so it is not"
          "\n  evidence of anything and is reported only to say so.")

    for name in ("real", "generated", "uniform"):
        report[f"{name}_coupling_per_clip"] = arms[name].tolist()
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

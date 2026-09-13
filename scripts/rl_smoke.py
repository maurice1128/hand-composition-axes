"""Smoke-test the retention environment and time it.

Answers the go/no-go question before any training: how many environment
frames per second does one process deliver, and do the two negative controls
behave (a random policy should mostly lose the object; a policy that holds
the decoder's "closed" direction should do better than random)?

    python scripts/rl_smoke.py --arm modular --run-dir runs/oakbricks_s1
    python scripts/rl_smoke.py --arm continuous --run-dir <perframe run>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caredex.rl.hand_env import ShadowRetainEnv  # noqa: E402


def rollout(env, policy, n_episodes: int, rng) -> dict:
    succ, jit, frames, t0 = [], [], 0, time.perf_counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            frames += 1
            done = term or trunc
        succ.append(info["success"])
        jit.append(info["jitter"])
    dt = time.perf_counter() - t0
    return {"success": float(np.mean(succ)), "jitter": float(np.mean(jit)),
            "fps": frames / dt, "frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("modular", "continuous"), required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    env = ShadowRetainEnv(args.arm, ROOT / args.run_dir, difficulty=args.difficulty)
    print(f"obs {env.observation_space.shape}  act {env.action_space.shape}  "
          f"steps/frame {env.steps_per_frame}  shake_g {env.shake_g}")
    rng = np.random.default_rng(args.seed)

    zero = lambda obs: np.zeros(env.action_space.shape, dtype=np.float32)  # noqa: E731
    rand = lambda obs: env.action_space.sample()  # noqa: E731

    for name, pol in (("zero-action", zero), ("random", rand)):
        r = rollout(env, pol, args.episodes, rng)
        print(f"{name:12s} success {r['success']:.2f}  jitter {r['jitter']:.1f}  "
              f"{r['fps']:.0f} frames/s over {r['frames']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

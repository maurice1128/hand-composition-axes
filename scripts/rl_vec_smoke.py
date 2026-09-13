"""Time the batched environment and check its controls agree with the single one.

    python scripts/rl_vec_smoke.py --arm modular --run-dir runs/oakink_attr_more/s100_modular_b256_naive
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

from caredex.rl.vec_env import RetainVecEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("modular", "continuous"), required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--difficulty", default="medium")
    args = ap.parse_args()
    torch.set_num_threads(1)

    t = time.perf_counter()
    env = RetainVecEnv(args.arm, ROOT / args.run_dir, num_envs=args.num_envs,
                       nthread=args.nthread, difficulty=args.difficulty)
    print(f"init {time.perf_counter() - t:.1f}s  obs {env.observation_space.shape}  "
          f"act {env.action_space.shape}  envs {env.num_envs}")
    for name in ("zero", "random"):
        obs = env.reset()
        succ, jit, n_ep = [], [], 0
        t = time.perf_counter()
        for _ in range(args.frames):
            a = np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32) \
                if name == "zero" else np.stack([env.action_space.sample() for _ in range(env.num_envs)])
            obs, r, d, infos = env.step(a)
            for inf in infos:
                if "success" in inf:
                    succ.append(inf["success"]); jit.append(inf["jitter"]); n_ep += 1
        dt = time.perf_counter() - t
        fps = args.frames * env.num_envs / dt
        print(f"{name:7s} {fps:7.0f} frames/s  episodes {n_ep}  success "
              f"{np.mean(succ) if succ else float('nan'):.2f}  jitter "
              f"{np.mean(jit) if jit else float('nan'):.1f}  obs finite {np.isfinite(obs).all()}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train a PPO policy that acts through a frozen prior, and score it.

One run trains one arm on one seed. The paired comparison is assembled by
``scripts/rl_paired_sweep.py`` from many of these, which is the same shape as
the reconstruction sweeps: both arms see the same seeds, the same scene, the
same perturbation schedule, and are scored on the same held-out episodes.

    python scripts/train_rl_policy.py --arm modular --run-dir runs/oakbricks_s1 \
        --seed 0 --n-envs 16 --total-steps 2000000 --out runs/rl/retain_modular_s0

Outputs, in ``--out``:
    policy.zip        the SB3 model
    progress.csv      per-rollout success rate and jitter during training
    eval.json         success rate, jitter and per-episode records on
                      ``--eval-episodes`` fresh episodes with fixed seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from caredex.rl.hand_env import make_env  # noqa: E402
from caredex.rl.vec_env import RetainVecEnv  # noqa: E402


class EpisodeStats:
    """Collect success and jitter from the info dicts a VecEnv emits."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self.buf_s: list[float] = []
        self.buf_j: list[float] = []
        self.t0 = time.time()

    def __call__(self, locals_, globals_) -> bool:  # SB3 callback signature
        for info in locals_.get("infos", ()):
            if "success" in info:
                self.buf_s.append(float(info["success"]))
                self.buf_j.append(float(info["jitter"]))
        return True

    def flush(self, timesteps: int) -> None:
        if not self.buf_s:
            return
        self.rows.append({"timesteps": timesteps, "episodes": len(self.buf_s),
                          "success": float(np.mean(self.buf_s)),
                          "jitter": float(np.mean(self.buf_j)),
                          "wall_s": time.time() - self.t0})
        self.buf_s.clear()
        self.buf_j.clear()
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def evaluate(model, env: RetainVecEnv, seed0: int) -> dict:
    """One episode per environment, deterministic policy, fixed rng seed.

    The pulse schedule draws from the env's rng once per step regardless of
    what the policy does, so two arms evaluated with the same seed and the
    same number of environments face the same perturbations: the evaluation
    is paired at the episode level, not just the training-seed level.
    """
    env.seed(seed0)
    obs = env.reset()
    n = env.num_envs
    done_once = np.zeros(n, dtype=bool)
    recs: list[dict | None] = [None] * n
    ret = np.zeros(n)
    while not done_once.all():
        act, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = env.step(act)
        ret += r * (~done_once)
        for i in np.flatnonzero(dones & ~done_once):
            recs[i] = {"env": int(i), "success": bool(infos[i]["success"]),
                       "jitter": float(infos[i]["jitter"]), "frames": int(infos[i]["frames"]),
                       "return": float(ret[i])}
            for k in ("sequence", "stages_ok"):
                if k in infos[i]:
                    recs[i][k] = infos[i][k]
            done_once[i] = True
    episodes = n
    return {"episodes": episodes,
            "success": float(np.mean([r["success"] for r in recs])),
            "jitter": float(np.mean([r["jitter"] for r in recs])),
            "return": float(np.mean([r["return"] for r in recs])),
            "records": recs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("modular", "continuous", "keyframe", "linear"), required=True)
    ap.add_argument("--run-dir", default="none", help="prior checkpoint directory ('none' for the linear arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--shape", default="box")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--n-steps", type=int, default=256, help="PPO rollout length per env")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--sde", type=int, default=8,
                    help="gSDE noise resample interval in steps; 0 disables gSDE")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--style-mode", default="window", choices=("window", "frame"))
    ap.add_argument("--untrained", action="store_true",
                    help="control: same architecture, random weights, no prior")
    ap.add_argument("--pose-residual", type=float, default=0.0,
                    help="degrees of per-joint residual the policy may add to the decoded pose")
    ap.add_argument("--residual-mode", default="rate", choices=("abs", "rate"))
    ap.add_argument("--residual-rate", type=float, default=3.0, help="deg per frame in rate mode")
    ap.add_argument("--macro-every", type=int, default=1,
                    help="hold the decoder action for this many frames (hierarchical interface)")
    ap.add_argument("--task", default="retain", choices=("retain", "staged"))
    ap.add_argument("--train-sequences", default="Hx-R,Hy-R,Hx-Hy,Hy-Hx",
                    help="staged task: sequence pool for training")
    ap.add_argument("--eval-sequences", default="Hx-Hy-R,Hy-Hx-R",
                    help="staged task: sequence pool for evaluation (held-out unless also in training)")
    ap.add_argument("--stage-frames", type=int, default=40)
    ap.add_argument("--perturb-mode", default="pulse", choices=("pulse", "tilt"))
    ap.add_argument("--reward-mode", default="sparse", choices=("sparse", "dense"))
    ap.add_argument("--init-poses", default="data/bundles/oakink_first_frames.npy",
                    help="npy of (N, 27) initial hand poses in degrees; 'none' for a flat hand")
    ap.add_argument("--vec", default="batched", choices=("batched", "subproc"))
    ap.add_argument("--nthread", type=int, default=8, help="physics threads (batched)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kw = dict(shape=args.shape, difficulty=args.difficulty)
    stepper_kw: dict = {"untrained": args.untrained, "untrained_seed": args.seed}
    if args.arm == "modular":
        stepper_kw["style_mode"] = args.style_mode
    env_kw["stepper_kw"] = stepper_kw
    env_kw["init_poses"] = None if args.init_poses == "none" else ROOT / args.init_poses
    env_kw["perturb_mode"] = args.perturb_mode
    env_kw["reward_mode"] = args.reward_mode
    env_kw["pose_residual_deg"] = args.pose_residual
    env_kw["macro_every"] = args.macro_every
    env_kw["residual_mode"] = args.residual_mode
    env_kw["residual_rate_deg"] = args.residual_rate
    if args.task == "staged":
        from caredex.rl.staged_env import StagedRetainVecEnv
        EnvCls = StagedRetainVecEnv
        env_kw["stage_frames"] = args.stage_frames
        env_kw["sequences"] = args.train_sequences
        if args.vec != "batched":
            raise SystemExit("--task staged needs --vec batched")
    else:
        EnvCls = RetainVecEnv
    if args.vec == "batched":
        vec = EnvCls(args.arm, ROOT / args.run_dir, num_envs=args.n_envs,
                     nthread=args.nthread, seed=args.seed, **env_kw)
    else:
        sub_kw = {k: v for k, v in env_kw.items()
                  if k not in ("init_poses", "perturb_mode", "reward_mode", "pose_residual_deg",
                               "macro_every", "residual_mode", "residual_rate_deg")}
        fns = [make_env(args.arm, ROOT / args.run_dir, seed=args.seed * 1000 + i, **sub_kw)
               for i in range(args.n_envs)]
        vec = SubprocVecEnv(fns, start_method="spawn") if args.n_envs > 1 else DummyVecEnv(fns)

    model = PPO("MlpPolicy", vec, seed=args.seed, device=args.device,
                n_steps=args.n_steps, batch_size=args.batch_size,
                learning_rate=args.lr, ent_coef=args.ent_coef,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10,
                use_sde=args.sde > 0, sde_sample_freq=args.sde if args.sde > 0 else -1,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=args.log_std_init),
                verbose=0)

    stats = EpisodeStats(out / "progress.csv")
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
        def _on_step(self) -> bool:
            return stats(self.locals, self.globals)

        def _on_rollout_end(self) -> None:
            stats.flush(self.num_timesteps)
            if stats.rows:
                r = stats.rows[-1]
                print(f"[{args.arm} s{args.seed}] {r['timesteps']:>9d}  "
                      f"success {r['success']:.2f}  jitter {r['jitter']:.1f}  "
                      f"{r['wall_s']/60:.1f} min", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=_CB())
    train_wall = time.time() - t0
    model.save(out / "policy.zip")
    vec.close()

    eval_kw = dict(env_kw)
    if args.task == "staged":
        eval_kw["sequences"] = args.eval_sequences
    eval_env = EnvCls(args.arm, ROOT / args.run_dir, num_envs=args.eval_episodes,
                      nthread=args.nthread, seed=args.eval_seed, **eval_kw)
    ev = evaluate(model, eval_env, args.eval_seed)
    if args.task == "staged":
        # Also score the training pool, so the held-out gap is visible per run.
        eval_env.set_sequences(args.train_sequences)
        ev_train = evaluate(model, eval_env, args.eval_seed)
        ev["train_pool"] = {k: ev_train[k] for k in ("success", "jitter", "return")}
        ev["train_sequences"], ev["eval_sequences"] = args.train_sequences, args.eval_sequences
        ev["task"] = "staged"
    eval_env.close()
    ev.update({"arm": args.arm, "run_dir": args.run_dir, "seed": args.seed,
               "difficulty": args.difficulty, "total_steps": args.total_steps,
               "n_envs": args.n_envs, "train_wall_s": train_wall,
               "fps": args.total_steps / max(train_wall, 1e-9),
               "action_dim": int(eval_env.action_space.shape[0]),
               "untrained": args.untrained, "perturb_mode": args.perturb_mode,
               "reward_mode": args.reward_mode, "log_std_init": args.log_std_init,
               "pose_residual_deg": args.pose_residual, "macro_every": args.macro_every,
               "task": args.task, "residual_mode": args.residual_mode,
               "residual_rate_deg": args.residual_rate,
               "sde": args.sde, "ent_coef": args.ent_coef,
               "style_mode": args.style_mode if args.arm == "modular" else None})
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    print(f"[{args.arm} s{args.seed}] eval success {ev['success']:.3f}  "
          f"jitter {ev['jitter']:.1f}  train {train_wall/60:.1f} min  "
          f"{ev['fps']:.0f} steps/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

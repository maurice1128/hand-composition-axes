"""Compositional sweep on the staged task: naive vs informed, both arms, paired.

For each seed and each arm two policies are trained with identical settings
and differ only in the sequence pool they saw:

    naive     trains on --train-sequences
    informed  trains on --train-sequences plus --eval-sequences

Both are scored on --eval-sequences, held out from the naive policy, so

    penalty(arm)  = success_informed - success_naive        (what not seeing
                                                             the composition costs)
    paired        = penalty_continuous - penalty_modular    (positive: the bank
                                                             pays less)

which is Section III-C's design carried from reconstruction error to task
success. The absolute held-out success of each naive arm is reported beside
it, because a small penalty on a floor means nothing.

    python scripts/rl_composition_sweep.py \
        --modular "runs/oakink_attr_more/s{p}_modular_b256_naive" \
        --continuous "runs/oakink_attr_more/s{p}_perframe_b256_naive" \
        --prior-seeds 100 101 102 103 104 --seeds 0 1 2 3 4 \
        --macro-every 16 --pose-residual 15 --total-steps 2000000 \
        --out runs/rl/compose_c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_one(arm: str, prior: str, seed: int, variant: str, out: Path, args) -> dict:
    d = out / f"{arm}_{variant}_s{seed}"
    if not (d / "eval.json").exists():
        train_pool = {"naive": args.train_sequences,
                      "informed": args.train_sequences + "," + args.eval_sequences,
                      "specialist": args.eval_sequences}[variant]
        cmd = [PY, str(ROOT / "scripts" / "train_rl_policy.py"),
               "--arm", arm, "--run-dir", prior, "--out", str(d.relative_to(ROOT)),
               "--seed", str(seed), "--n-envs", str(args.n_envs), "--nthread", str(args.nthread),
               "--total-steps", str(args.total_steps), "--difficulty", args.difficulty,
               "--eval-episodes", str(args.eval_episodes), "--perturb-mode", "tilt",
               "--log-std-init", str(args.log_std_init), "--sde", str(args.sde),
               "--pose-residual", str(args.pose_residual), "--macro-every", str(args.macro_every),
               "--task", "staged", "--stage-frames", str(args.stage_frames),
               "--train-sequences", train_pool, "--eval-sequences", args.eval_sequences]
        if arm == "modular":
            cmd += ["--style-mode", args.style_mode]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads((d / "eval.json").read_text())


def block(d: np.ndarray) -> dict:
    n = len(d)
    wins, ties = int((d > 0).sum()), int((d == 0).sum())
    return {"n": n, "mean": float(d.mean()),
            "t_p": float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and d.std() > 0 else None,
            "sign_wins": wins, "sign_ties": ties,
            "sign_p": float(stats.binomtest(wins, n - ties, 0.5).pvalue) if n - ties > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modular", required=True)
    ap.add_argument("--continuous", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prior-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-sequences", default="Hx-R,Hy-R,Hx-Hy,Hy-Hx")
    ap.add_argument("--eval-sequences", default="Hx-Hy-R,Hy-Hx-R")
    ap.add_argument("--stage-frames", type=int, default=40)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--nthread", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--difficulty", default="m0.20g1.5")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--style-mode", default="window")
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--sde", type=int, default=8)
    ap.add_argument("--pose-residual", type=float, default=15.0)
    ap.add_argument("--macro-every", type=int, default=16)
    ap.add_argument("--variants", nargs="*", default=["naive", "informed"],
                    choices=["naive", "informed", "specialist"],
                    help="specialist trains on the eval sequences only: the ceiling a policy "
                         "reaches when every sample goes to the held-out compositions")
    ap.add_argument("--out", default="runs/rl/compose_c")
    args = ap.parse_args()
    if args.prior_seeds and len(args.prior_seeds) != len(args.seeds):
        raise SystemExit("--prior-seeds must have one entry per --seeds entry")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_args.json").write_text(json.dumps(vars(args), indent=1))

    rows = []
    for k, s in enumerate(args.seeds):
        p = args.prior_seeds[k] if args.prior_seeds else None
        row = {"seed": s, "prior_seed": p}
        for arm, pat in (("modular", args.modular), ("continuous", args.continuous)):
            prior = pat.format(p=p)
            nv = run_one(arm, prior, s, "naive", out, args)
            inf = run_one(arm, prior, s, "informed", out, args)
            row[f"{arm}_naive"] = nv["success"]
            row[f"{arm}_informed"] = inf["success"]
            row[f"{arm}_naive_trainpool"] = nv["train_pool"]["success"]
            # Held-out success as a fraction of training-pool success: how much
            # of what the policy learned carries to the unseen sequences.
            row[f"{arm}_gen_ratio"] = nv["success"] / max(nv["train_pool"]["success"], 1e-9)
            row[f"{arm}_penalty"] = inf["success"] - nv["success"]
            row[f"{arm}_jitter"] = nv["jitter"]
            if "specialist" in args.variants:
                sp = run_one(arm, prior, s, "specialist", out, args)
                row[f"{arm}_specialist"] = sp["success"]
                row[f"{arm}_penalty_spec"] = sp["success"] - nv["success"]
        row["paired"] = row["continuous_penalty"] - row["modular_penalty"]
        if "specialist" in args.variants:
            row["paired_spec"] = row["continuous_penalty_spec"] - row["modular_penalty_spec"]
        rows.append(row)
        print(f"seed {s}: modular naive {row['modular_naive']:.2f} informed {row['modular_informed']:.2f} "
              f"(train-pool {row['modular_naive_trainpool']:.2f})  | continuous naive {row['continuous_naive']:.2f} "
              f"informed {row['continuous_informed']:.2f} (train-pool {row['continuous_naive_trainpool']:.2f})  "
              f"| penalties mod {row['modular_penalty']:+.2f} cont {row['continuous_penalty']:+.2f} paired {row['paired']:+.2f}",
              flush=True)

    summary = {
        "paired_penalty": block(np.array([r["paired"] for r in rows])),
        **({"paired_penalty_spec": block(np.array([r["paired_spec"] for r in rows])),
            "modular_penalty_spec": block(np.array([r["modular_penalty_spec"] for r in rows])),
            "continuous_penalty_spec": block(np.array([r["continuous_penalty_spec"] for r in rows]))}
           if "specialist" in args.variants else {}),
        "modular_penalty": block(np.array([r["modular_penalty"] for r in rows])),
        "continuous_penalty": block(np.array([r["continuous_penalty"] for r in rows])),
        "heldout_success_mod_minus_cont": block(np.array([r["modular_naive"] - r["continuous_naive"] for r in rows])),
        "trainpool_success_mod_minus_cont": block(np.array([r["modular_naive_trainpool"] - r["continuous_naive_trainpool"] for r in rows])),
        "gen_ratio_mod_minus_cont": block(np.array([r["modular_gen_ratio"] - r["continuous_gen_ratio"] for r in rows])),
        "means": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("seed", "prior_seed")},
        "rows": rows, "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "args")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

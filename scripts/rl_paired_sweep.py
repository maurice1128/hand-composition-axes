"""Paired sweep: both arms, same seeds, same task; then the paired test.

The reconstruction sweeps paired arms by split seed. Here the pairing is by
prior seed *and* training seed: RL seed ``s`` uses the modular and perframe
priors that the reconstruction sweep trained under prior seed ``P[s]`` on the
same data, fixes the environment rng streams and the PPO initialisation for
both arms, and evaluates both on the same 100 perturbation schedules. The
difference in success rate is scored per seed with a t-test and a sign test
beside it, as in Section IV of the paper.

    python scripts/rl_paired_sweep.py \
        --modular "runs/oakink_attr_more/s{p}_modular_b256_naive" \
        --continuous "runs/oakink_attr_more/s{p}_perframe_b256_naive" \
        --prior-seeds 100 101 102 103 104 --seeds 0 1 2 3 4 \
        --n-envs 32 --total-steps 2000000 --out runs/rl/retain_pilot

``{p}`` in a prior path is replaced by the prior seed paired with each RL
seed; a path without ``{p}`` uses one prior for every seed. ``--controls
untrained`` adds the two untrained-decoder arms on the same seeds. Runs that
already have ``eval.json`` are skipped, so the sweep is resumable.
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


def run_one(arm: str, prior: str, seed: int, out: Path, args, untrained: bool = False) -> dict:
    tag = f"{arm}{'_untrained' if untrained else ''}_s{seed}"
    d = out / tag
    if not (d / "eval.json").exists():
        cmd = [PY, str(ROOT / "scripts" / "train_rl_policy.py"),
               "--arm", arm, "--run-dir", prior, "--out", str(d.relative_to(ROOT)),
               "--seed", str(seed), "--n-envs", str(args.n_envs), "--nthread", str(args.nthread),
               "--total-steps", str(args.total_steps),
               "--difficulty", args.difficulty, "--eval-episodes", str(args.eval_episodes),
               "--perturb-mode", args.perturb_mode, "--reward-mode", args.reward_mode,
               "--log-std-init", str(args.log_std_init), "--sde", str(args.sde),
               "--pose-residual", str(args.pose_residual), "--macro-every", str(args.macro_every)]
        if arm == "modular":
            cmd += ["--style-mode", args.style_mode]
        if untrained:
            cmd += ["--untrained"]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads((d / "eval.json").read_text())


def paired_block(rows: list[dict], key_a: str, key_b: str) -> dict:
    a = np.array([r[key_a] for r in rows])
    b = np.array([r[key_b] for r in rows])
    d = a - b
    n = len(d)
    wins, ties = int((d > 0).sum()), int((d == 0).sum())
    return {
        "n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "paired_mean": float(d.mean()),
        "t_p": float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and d.std() > 0 else None,
        "sign_wins": wins, "sign_ties": ties,
        "sign_p": float(stats.binomtest(wins, n - ties, 0.5).pvalue) if n - ties > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modular", required=True)
    ap.add_argument("--continuous", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prior-seeds", type=int, nargs="*", default=None,
                    help="one per RL seed, substituted for {p}")
    ap.add_argument("--controls", nargs="*", default=[], choices=["untrained"])
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--difficulty", default="m0.20g1.5")
    ap.add_argument("--perturb-mode", default="tilt", choices=("pulse", "tilt"))
    ap.add_argument("--reward-mode", default="sparse", choices=("sparse", "dense"))
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--sde", type=int, default=8)
    ap.add_argument("--pose-residual", type=float, default=0.0)
    ap.add_argument("--macro-every", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--style-mode", default="window")
    ap.add_argument("--out", default="runs/rl/retain_pilot")
    args = ap.parse_args()

    if args.prior_seeds and len(args.prior_seeds) != len(args.seeds):
        raise SystemExit("--prior-seeds must have one entry per --seeds entry")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_args.json").write_text(json.dumps(vars(args), indent=1))

    rows = []
    for k, s in enumerate(args.seeds):
        p = args.prior_seeds[k] if args.prior_seeds else None
        mod = args.modular.format(p=p)
        con = args.continuous.format(p=p)
        m = run_one("modular", mod, s, out, args)
        c = run_one("continuous", con, s, out, args)
        row = {"seed": s, "prior_seed": p,
               "modular_success": m["success"], "continuous_success": c["success"],
               "modular_jitter": m["jitter"], "continuous_jitter": c["jitter"]}
        if "untrained" in args.controls:
            mu = run_one("modular", mod, s, out, args, untrained=True)
            cu = run_one("continuous", con, s, out, args, untrained=True)
            row.update(modular_untrained_success=mu["success"], continuous_untrained_success=cu["success"],
                       modular_untrained_jitter=mu["jitter"], continuous_untrained_jitter=cu["jitter"])
        rows.append(row)
        print(f"seed {s}: modular {m['success']:.2f} / J {m['jitter']:.1f}   "
              f"continuous {c['success']:.2f} / J {c['jitter']:.1f}   "
              f"paired {m['success'] - c['success']:+.2f}", flush=True)

    summary = {
        "success": paired_block(rows, "modular_success", "continuous_success"),
        "jitter": paired_block(rows, "modular_jitter", "continuous_jitter"),
        "rows": rows, "args": vars(args),
    }
    if "untrained" in args.controls:
        summary["modular_vs_untrained"] = paired_block(rows, "modular_success", "modular_untrained_success")
        summary["continuous_vs_untrained"] = paired_block(rows, "continuous_success", "continuous_untrained_success")
        summary["untrained_success"] = paired_block(rows, "modular_untrained_success", "continuous_untrained_success")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "args")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

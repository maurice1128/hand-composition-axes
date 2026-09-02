"""Wait for the OakInk2 bundle, then run everything that depends on it.

The bundle takes a while to build (627 sequences of ~10k mocap frames each),
and the experiments that matter cannot start without it. This chains them so
the GPU is not left idle waiting for a human to notice the build finished.

Order matters: the gates run first and the sweep is skipped if they fail,
because a paired result on a leaking or too-thin split is not a result. Every
mistake that cost time in this project was of that shape -- a number produced
by a design that could not have measured what it claimed.

    python scripts/run_oakink2_pipeline.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def run(cmd: list[str], must_pass: bool = False) -> tuple[int, str]:
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}\n", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + ("\n" + proc.stderr if proc.returncode else "")
    print(out[-4000:], flush=True)
    if must_pass and proc.returncode != 0:
        raise SystemExit(f"gate failed: {' '.join(str(c) for c in cmd[1:4])} ...")
    return proc.returncode, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/bundles/oakink2.npz")
    ap.add_argument("--wait-minutes", type=int, default=180)
    ap.add_argument("--held-compositions", type=int, default=6)
    ap.add_argument("--min-per-composition", type=int, default=5)
    ap.add_argument("--min-chains", type=int, default=6)
    ap.add_argument("--budgets", type=int, nargs="*", default=[256])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument(
        "--stride", type=int, default=16,
        help="window stride for the dataloader. OakInk2 carries 569k frames -- 7x "
             "OakInk-Image -- and its segments stay long after the bundle's own "
             "subsampling, so windows overlap heavily. Reusing OakInk-Image's "
             "stride of 4 put this sweep on a 36-hour path.",
    )
    args = ap.parse_args()

    bundle = ROOT / args.bundle
    deadline = time.time() + args.wait_minutes * 60
    while not bundle.exists():
        if time.time() > deadline:
            print(f"gave up waiting for {bundle}")
            return 1
        time.sleep(30)
    # The writer may still be flushing; wait for the size to settle.
    last = -1
    while True:
        size = bundle.stat().st_size
        if size == last and size > 0:
            break
        last = size
        time.sleep(20)
    print(f"[pipeline] bundle ready: {bundle} ({size / 1e6:.0f} MB)\n", flush=True)

    gates = [
        [PY, "scripts/check_informed_coverage.py", "--bundle", args.bundle,
         "--held-compositions", str(args.held_compositions),
         "--min-per-composition", str(args.min_per_composition),
         "--min-chains", str(args.min_chains),
         "--seeds", "0", "1", "2"],
        [PY, "scripts/check_composition_leak.py", "--bundle", args.bundle,
         "--granularity", "fine",
         "--held-compositions", str(args.held_compositions),
         "--min-per-composition", str(args.min_per_composition),
         "--min-chains", str(args.min_chains),
         "--budget", str(args.budgets[0])],
        [PY, "scripts/check_split_balance.py", "--bundle", args.bundle],
    ]
    codes = []
    for g in gates:
        code, _ = run(g)
        codes.append(code)

    # Coverage and leak are hard gates; balance is advisory because the paired
    # design cancels set difficulty anyway.
    if codes[0] != 0:
        print("\nSTOP: informed coverage too thin. Lower --held-compositions and retry.")
        return 1
    if codes[1] != 0:
        print("\nSTOP: composition leak detected. Do not trust any penalty from this split.")
        return 1

    run([PY, "scripts/experiment_paired_composition.py",
         "--bundle", args.bundle, "--out", "runs/oakink2_paired",
         "--kinds", "perframe", "modular", "mann",
         "--budgets", *[str(b) for b in args.budgets],
         "--held-compositions", str(args.held_compositions),
         "--min-per-composition", str(args.min_per_composition),
         "--min-chains", str(args.min_chains),
         "--seeds", *[str(s) for s in args.seeds],
         "--stride", str(args.stride),
         "--epochs", str(args.epochs), "--fresh"])

    run([PY, "scripts/analyse_paired.py", "runs/oakink2_paired"])
    run([PY, "scripts/analyse_paired.py", "runs/oakink2_paired",
         "--baseline", "mann", "--treatment", "modular"])

    # A prior trained on the whole bundle, then scored against the annotated
    # primitive boundaries -- the evaluation no earlier dataset supported.
    run([PY, "scripts/train_prior.py", "--model", "modular",
         "--bundle", args.bundle, "--run-dir", "runs/prior_oakink2_modular",
         "--fresh", "--set", "train.epochs=200", "train.save_every=99999",
         "loader.stride=8", "model.n_primitives=16"])
    run([PY, "scripts/eval_segmentation.py",
         "--run-dir", "runs/prior_oakink2_modular", "--bundle", args.bundle])
    run([PY, "scripts/demo_composition.py",
         "--run-dir", "runs/prior_oakink2_modular", "--bundle", args.bundle])

    print("\n[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

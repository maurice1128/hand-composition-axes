"""Keep a long sweep alive across GPU faults, without touching the live run.

The OakInk2 prior died at epoch 57 with ``torch.AcceleratorError: CUDA error:
unknown error`` -- this machine runs a dozen other GPU jobs and the fault is not
reproducible or attributable. A 13-hour sweep on the same GPU will probably meet
it too, and the failure mode that matters is not the crash: it is coming back
hours later to a directory holding 11 of 40 seeds with nothing saying so.

Strategy: never restart into the same directory. ``experiment_paired_composition``
rewrites ``results.json`` with only its own seeds, so relaunching in place would
discard finished work. Each attempt gets its own directory and the analysis
pools them -- ``analyse_paired.py`` already accepts several run dirs, because
the OakInk n=70 result was assembled that way.

    python scripts/supervise_sweep.py --run-dir runs/grab_shape --seeds 40

Watches an already-running sweep; does not start the first attempt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def completed_seeds(run_dirs: list[Path], budget: int) -> set[int]:
    """Seeds with a result for every kind -- a partial seed has to be redone."""
    done: set[int] = set()
    for d in run_dirs:
        p = d / "results.json"
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # mid-write; the next poll will read it
        kinds = set(blob.get("args", {}).get("kinds", []))
        by_seed: dict[int, set[str]] = {}
        for r in blob.get("results", []):
            if r.get("budget") == budget:
                by_seed.setdefault(r.get("seed", 0), set()).add(r["kind"])
        done |= {s for s, ks in by_seed.items() if kinds <= ks}
    return done


def alive(needles: list[str]) -> bool:
    out = subprocess.run([str(PY), str(ROOT / "scripts" / "count_running.py"), *needles],
                         capture_output=True, text=True)
    return (out.stdout.strip() or "0") != "0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/grab_shape")
    ap.add_argument("--bundle", default="data/bundles/grab.npz")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--granularity", default="shape")
    ap.add_argument("--kinds", nargs="*", default=["perframe", "modular"])
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--held-compositions", type=int, default=6)
    ap.add_argument("--min-per-composition", type=int, default=5)
    ap.add_argument("--min-chains", type=int, default=5)
    ap.add_argument("--poll-seconds", type=int, default=180)
    ap.add_argument("--max-restarts", type=int, default=6)
    args = ap.parse_args()

    base = ROOT / args.run_dir
    wanted = set(range(args.seeds))
    attempt = 0

    while True:
        dirs = sorted(base.parent.glob(base.name + "*"))
        if alive([Path(args.bundle).name]):
            time.sleep(args.poll_seconds)
            continue

        done = completed_seeds(dirs, args.budget)
        missing = sorted(wanted - done)
        print(f"[supervise] no sweep running. {len(done)}/{args.seeds} seeds complete "
              f"across {len(dirs)} dir(s).", flush=True)
        if not missing:
            print("[supervise] all seeds done.")
            return 0
        if attempt >= args.max_restarts:
            print(f"[supervise] {args.max_restarts} restarts used; {len(missing)} seeds "
                  f"still missing: {missing}. Stopping so this needs a human look.")
            return 1

        attempt += 1
        out = base.parent / f"{base.name}_r{attempt}"
        print(f"[supervise] restart {attempt}: {len(missing)} seeds -> {out}", flush=True)
        cmd = [str(PY), "-u", "scripts/experiment_paired_composition.py",
               "--bundle", args.bundle, "--out", str(out.relative_to(ROOT)),
               "--granularity", args.granularity, "--kinds", *args.kinds,
               "--budgets", str(args.budget),
               "--held-compositions", str(args.held_compositions),
               "--min-per-composition", str(args.min_per_composition),
               "--min-chains", str(args.min_chains),
               "--seeds", *[str(s) for s in missing],
               "--stride", str(args.stride), "--epochs", str(args.epochs), "--fresh"]
        log = base.parent / f"{base.name}_r{attempt}_log.txt"
        with log.open("w", encoding="utf-8") as fh:
            subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        # Give it time to claim the GPU before the liveness check runs again.
        time.sleep(90)


if __name__ == "__main__":
    raise SystemExit(main())

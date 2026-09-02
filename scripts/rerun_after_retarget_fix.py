"""Re-measure everything that was measured on broken retargeting.

Two bugs killed degrees of freedom in three of four real datasets (see commit
"Fix two retargeting bugs..."). Every number produced from those bundles has to
be produced again before it can be quoted. The OakInk-Image n=70 headline was
already redone by hand and survived; this covers the rest.

Order is a dependency order, not a priority order:

1. **Gate first.** ``check_dof_health`` on every bundle. If a bundle still has a
   dead DOF there is no point measuring anything on it, and the whole reason
   this rerun exists is that a broken gate let exactly that happen.
2. **Difficulty survey rows**, the claim most likely to change. DexYCB and
   OakInk2 both carried a dead DOF when their nulls were measured -- DexYCB's
   thumb abduction, OakInk2's *index PIP flexion* -- so "compositional
   difficulty spans 200x and does not track annotation" currently rests on two
   measurements taken through a crippled representation.
3. **GRAB replication**, the new evidence, last because it is the longest.

Each step writes to its own run directory and is skipped if that directory
already holds a complete result, so this can be re-invoked after a crash
without repeating finished work. GPU faults on this machine are not
hypothetical: an earlier prior died at epoch 57 with ``CUDA error: unknown``.

    python scripts/rerun_after_retarget_fix.py
    python scripts/rerun_after_retarget_fix.py --skip-grab   # survey rows only
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


def complete(run_dir: Path, seeds: int, kinds: list[str], budget: int) -> bool:
    """True if this run already holds every seed for every kind."""
    p = run_dir / "results.json"
    if not p.exists():
        return False
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))["results"]
    except (json.JSONDecodeError, KeyError):
        return False
    by: dict[int, set[str]] = {}
    for r in rows:
        if r.get("budget") == budget:
            by.setdefault(r.get("seed", 0), set()).add(r["kind"])
    return sum(1 for ks in by.values() if set(kinds) <= ks) >= seeds


def run(cmd: list[str], log: Path) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd[2:])}\n  -> {log}", flush=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        code = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=fh,
                              stderr=subprocess.STDOUT).returncode
    print(f"  exit {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return code


#: (name, bundle, out, granularity, seeds, split args, sweep-only args)
#:
#: Split args go to both the leak gate and the sweep, so the gate certifies the
#: split the sweep actually uses. Sweep-only args (window stride) must NOT be
#: forwarded -- the gate has no such option and argparse exits 2, which on the
#: first attempt looked like a gate failure.
#:
#: DexYCB's numbers reproduce its original run: 4 held compositions at
#: min_per_composition 6 and the default min_chains. Asking for 6 held
#: compositions there fails structurally -- with 60 fine cells, none appears in
#: 4 distinct groups -- which is a property of the dataset, not something the
#: retargeting fix changes.
JOBS = [
    ("dexycb", "data/bundles/dexycb_shape.npz", "runs/dexycb_paired_v2", "fine", 12,
     ["--held-compositions", "4", "--min-per-composition", "6", "--min-chains", "1"],
     ["--stride", "4"]),
    ("oakink2", "data/bundles/oakink2.npz", "runs/oakink2_paired_v2", "fine", 12,
     ["--held-compositions", "6", "--min-per-composition", "5", "--min-chains", "6"],
     ["--stride", "16"]),
    ("grab", "data/bundles/grab.npz", "runs/grab_shape_v2", "shape", 40,
     ["--held-compositions", "6", "--min-per-composition", "5", "--min-chains", "5"],
     ["--stride", "32"]),
]

KINDS = ["perframe", "modular"]
BUDGET = 256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-grab", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    logs = ROOT / "runs"

    print("=" * 78)
    print("STEP 1  every bundle must pass the DOF audit")
    print("=" * 78, flush=True)
    code = run([PY, "-u", "scripts/check_dof_health.py"], logs / "rerun_dof_health.txt")
    text = (logs / "rerun_dof_health.txt").read_text(encoding="utf-8")
    print(text[-2500:], flush=True)
    # Abort only on a dead *articulation* DOF. ``wrist_tz`` reads dead in GRAB
    # and DexYCB after per-trajectory centring, because hand depth barely
    # changes within one of their sequences -- a property of those capture rigs,
    # already documented, and one translation DOF. A dead finger or thumb joint
    # is the failure this rerun exists to correct, and must stop it.
    dead_articulation = [l.strip() for l in text.splitlines()
                         if "DEAD" in l and "wrist_t" not in l]
    if dead_articulation:
        print("\nSTOP: a bundle still has a dead articulation DOF:")
        for line in dead_articulation:
            print("  ", line)
        return 1
    if code != 0:
        print("\nNote: audit reports partially-pinned DOF but nothing dead; continuing.")

    print("\n" + "=" * 78)
    print("STEP 2  separability diagnostic on rebuilt bundles")
    print("=" * 78, flush=True)
    run([PY, "-u", "scripts/analyse_composition_separability.py"],
        logs / "rerun_separability.txt")
    print((logs / "rerun_separability.txt").read_text(encoding="utf-8")[-1400:], flush=True)

    print("\n" + "=" * 78)
    print("STEP 3  paired composition, re-measured")
    print("=" * 78, flush=True)
    for name, bundle, out, gran, seeds, split_args, sweep_args in JOBS:
        if name == "grab" and args.skip_grab:
            print(f"\nskip {name} (--skip-grab)")
            continue
        out_dir = ROOT / out
        if complete(out_dir, seeds, KINDS, BUDGET):
            print(f"\nskip {name}: {out} already complete ({seeds} seeds)")
            continue
        # Leak and coverage are hard gates. A paired penalty from a leaking or
        # too-thin split is not a weaker result, it is not a result.
        gate = run([PY, "-u", "scripts/check_composition_leak.py", "--bundle", bundle,
                    "--granularity", gran, "--budget", str(BUDGET), *split_args],
                   logs / f"rerun_{name}_leak.txt")
        if gate != 0:
            print(f"  SKIP {name}: composition leak gate failed, see the log.")
            continue
        run([PY, "-u", "scripts/experiment_paired_composition.py",
             "--bundle", bundle, "--out", out, "--granularity", gran,
             "--kinds", *KINDS, "--budgets", str(BUDGET),
             "--seeds", *[str(s) for s in range(seeds)],
             "--epochs", str(args.epochs), "--fresh", *split_args, *sweep_args],
            logs / f"rerun_{name}_log.txt")
        run([PY, "-u", "scripts/analyse_paired.py", out], logs / f"rerun_{name}_paired.txt")
        print((logs / f"rerun_{name}_paired.txt").read_text(encoding="utf-8")[-900:], flush=True)

    print("\n" + "=" * 78)
    print("STEP 4  redraw the survey figure from whatever is now on disk")
    print("=" * 78, flush=True)
    run([PY, "-u", "scripts/plot_difficulty_survey.py"], logs / "rerun_survey.txt")
    print((logs / "rerun_survey.txt").read_text(encoding="utf-8")[-1800:], flush=True)

    print("\nDone. Nothing here updates the prose in docs/ -- that is deliberate, "
          "the numbers change first and the claims are rewritten against them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

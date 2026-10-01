"""Gate one alignment-v2 bundle before it may be trained (runs/PREREG_oakink_alignment_v2.md).

Exit 0 only if all of these hold, at granularity oakink_category, held 5, min_chains 4, min_per_composition 5,
budget 64:
  - frame leak: at most 1% of target frames verbatim in each arm's training set, seeds 0-4;
  - naive pool above 64 on every seed 0-39;
  - check_composition_leak.py exits 0, seeds 0-39;
  - check_informed_coverage.py fails at most 10 of 40 seeds.
Usage: gate_alignment_v2.py <bundle.npz> <name>
Writes runs/gates/gate_alignment_v2_<name>.txt.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import experiment_paired_composition as E  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402

bundle_path, name = sys.argv[1], sys.argv[2]
GRAN, HELD, MC, MPC, BUDGET = "oakink_category", 5, 4, 5, 64
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
report = []


def say(msg):
    print(msg, flush=True)
    report.append(msg)


def keys(traj):
    return {np.round(row, 5).tobytes() for row in np.asarray(traj, dtype=np.float32)}


ok = True
bundle = TrajectoryBundle.load(str(ROOT / bundle_path))
fine = list(bundle.labels)
bundle.labels = E.coarsen_labels(fine, GRAN)

min_pool = None
for seed in range(40):
    split = E.build_paired_split(bundle, HELD, seed, fine_labels=fine, min_chains=MC)
    pool = len(split["naive_pool"])
    min_pool = pool if min_pool is None else min(min_pool, pool)
    if seed < 5:
        naive, informed = E.sample_pools(split, BUDGET, seed, bundle.labels, MPC)
        tgt = set().union(*(keys(bundle.trajectories[i]) for i in split["target"]))
        nf = set().union(*(keys(bundle.trajectories[i]) for i in naive))
        inf = set().union(*(keys(bundle.trajectories[i]) for i in informed))
        rn, ri = len(tgt & nf) / len(tgt), len(tgt & inf) / len(tgt)
        say(f"frame leak seed {seed}: naive {100*rn:.2f}%  informed {100*ri:.2f}%")
        if rn > 0.01 or ri > 0.01:
            ok = False
say(f"minimum naive pool over 40 seeds: {min_pool}")
if min_pool <= BUDGET:
    ok = False

common = ["--bundle", bundle_path, "--granularity", GRAN, "--held-compositions", str(HELD),
          "--min-chains", str(MC), "--min-per-composition", str(MPC)]
seeds = [str(s) for s in range(40)]
leak = subprocess.run([PY, "scripts/check_composition_leak.py", *common, "--budget", str(BUDGET), "--seeds", *seeds],
                      cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
(ROOT / "runs" / "gates" / f"composition_leak_{name}.txt").write_text(leak.stdout + leak.stderr, encoding="utf-8")
say(f"label leak gate exit {leak.returncode}")
if leak.returncode != 0:
    ok = False

cov = subprocess.run([PY, "scripts/check_informed_coverage.py", *common, "--budgets", str(BUDGET), "--seeds", *seeds],
                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
(ROOT / "runs" / "gates" / f"cover_{name}.txt").write_text(cov.stdout + cov.stderr, encoding="utf-8")
rows = [m for m in (re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$", l)
                    for l in cov.stdout.splitlines()) if m]
fails = sum(not m.group(7).strip().startswith("ok") for m in rows)
say(f"coverage: {fails} of {len(rows)} rows fail")
if len(rows) != 40 or fails > 10:
    ok = False

say("GATE " + ("PASS" if ok else "FAIL"))
(ROOT / "runs" / "gates" / f"gate_alignment_v2_{name}.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
sys.exit(0 if ok else 1)

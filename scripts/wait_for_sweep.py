"""Block until a sweep reaches its full sample size, then exit.

Waking on a condition rather than a timer: a fixed wait either fires early on a
partial result -- the failure mode this project hit repeatedly -- or leaves the
GPU idle after the run finishes.

    python scripts/wait_for_sweep.py runs/oakink2_paired --rows 24
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def row_count(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("results", []))
    except Exception:
        return 0


def python_alive() -> int:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return max(len(out.strip().splitlines()) - 1, 0)
    except Exception:
        return -1  # unknown; do not treat as stalled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-minutes", type=int, default=240)
    args = ap.parse_args()

    results = Path(args.run_dir) / "results.json"
    deadline = time.time() + args.max_minutes * 60
    # Only call it stalled after several consecutive empty polls: this script
    # itself is a python process, and a brief gap between training runs would
    # otherwise look like death.
    empty = 0

    while time.time() < deadline:
        n = row_count(results)
        if n >= args.rows:
            print(f"COMPLETE {n}/{args.rows}", flush=True)
            return 0
        alive = python_alive()
        empty = empty + 1 if alive == 1 else 0
        if empty >= 4:
            print(f"STALLED at {n}/{args.rows}: no worker process for {empty * args.poll}s", flush=True)
            return 2
        time.sleep(args.poll)

    print(f"TIMEOUT at {row_count(results)}/{args.rows}", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

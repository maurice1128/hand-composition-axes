"""Refuse to let downstream checks read an undertrained checkpoint.

This exists because the guard it replaces was written inline inside a shell
script, the quoting broke, and ``sh`` reported a syntax error *instead of*
running the assertion -- after which segmentation numbers from a partially
trained model would have been read as if they meant something. That is the
third time in this project a check failed open. A separate file cannot be
broken by the quoting of the script that calls it.

    python scripts/assert_trained.py runs/prior_oakink2_modular --min-epoch 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--min-epoch", type=int, default=150)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()

    run = Path(args.run_dir)
    # How far training GOT is recorded in last.pt. best.pt stores the epoch at
    # which validation last improved, so reading the epoch from it conflates
    # "training stopped early" with "the model converged early" -- and raised a
    # false UNDERTRAINED alarm on a run that completed all 200 epochs with its
    # best validation at epoch 28.
    last, best = run / "last.pt", run / args.checkpoint
    if not last.exists():
        print(f"FAIL: {last} does not exist -- training never got far enough to save.")
        return 1

    reached = int(torch.load(last, map_location="cpu", weights_only=False)["state"].get("epoch", -1))
    best_epoch = "?"
    if best.exists():
        best_epoch = torch.load(best, map_location="cpu", weights_only=False)["state"].get(
            "best_epoch", "?")
    print(f"{run}: trained to epoch {reached} | best validation at epoch {best_epoch}")

    if reached < args.min_epoch:
        print(f"FAIL: UNDERTRAINED ({reached} < {args.min_epoch}). "
              f"Do not read anything downstream.")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

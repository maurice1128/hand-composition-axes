"""Build the OakInk2 bundle, in parallel, with visible progress.

The single-process version ran for three hours without finishing. The cost is
``pickle.load``: each of the 627 ``anno_preview`` files is ~55 MB holding
per-frame camera intrinsics for four views, per-frame poses for every scene
object, and a full SMPL-X body track, and ``raw_mano`` cannot be reached
without deserialising all of it. Subsampling frames does not help because the
cost is paid before any frame is chosen. The work is CPU-bound and independent
per file, so the only real lever is doing several at once.

Needs its own entry point rather than an inline ``python -c``: a
``ProcessPoolExecutor`` on Windows re-imports ``__main__`` in every worker, so
the caller must be an importable module guarded by ``if __name__``.

    python scripts/build_oakink2_bundle.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.oakink2 import OakInk2Source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=r"D:\datasets\oakink2")
    ap.add_argument("--out", default="data/bundles/oakink2.npz")
    ap.add_argument("--hand", choices=("rh", "lh"), default="rh")
    ap.add_argument("--unit", choices=("sequence", "segment"), default="sequence")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6,
                    help="each worker expands a 55 MB pickle into 1-2 GB; size "
                         "this against free RAM, not against core count")
    ap.add_argument("--max-sequences", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    bundle = OakInk2Source(
        root=args.root, hand=args.hand, unit=args.unit,
        stride=args.stride, n_workers=args.workers,
        max_sequences=args.max_sequences,
    ).load()
    bundle.save(args.out)

    labels = Counter(bundle.labels)
    multi = [l for l in bundle.labels if "->" in l]
    transitions = Counter(
        (p, q)
        for lab in bundle.labels
        for p, q in zip(lab.split("->")[:-1], lab.split("->")[1:])
    )

    print(f"\nwrote {args.out} in {(time.time() - t0) / 60:.1f} min")
    print(f"  trajectories        {len(bundle.trajectories)}")
    print(f"  frames              {bundle.n_frames:,}  at {bundle.fps:g} fps")
    print(f"  distinct chains     {len(labels)}")
    print(f"  carry a transition  {len(multi)}")
    print(f"  distinct transitions {len(transitions)}")
    if transitions:
        dense = sum(1 for v in transitions.values() if v >= 6)
        print(f"  transitions seen >= 6 times: {dense}")
        print("  most common:")
        for (p, q), c in transitions.most_common(8):
            print(f"    {p} -> {q:<24} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

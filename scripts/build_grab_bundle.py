"""Build the GRAB bundle.

Separate entry point so the build is a named, re-runnable artifact rather than
an inline command, and so the subject subset actually present is recorded --
eight of ten archives were downloaded, and a survey row labelled "GRAB" must
say which.

    python scripts/build_grab_bundle.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.data.grab import GrabSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=r"D:\datasets\grab")
    ap.add_argument("--out", default="data/bundles/grab.npz")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--hand", choices=("rhand", "lhand"), default="rhand")
    args = ap.parse_args()

    t0 = time.time()
    b = GrabSource(root=args.root, hand=args.hand, stride=args.stride).load()
    b.save(args.out)

    cells = Counter(b.labels)
    objs = Counter(l.split("->")[0] for l in b.labels)
    ints = Counter(l.split("->")[1] for l in b.labels)

    print(f"\nwrote {args.out} in {(time.time() - t0) / 60:.1f} min")
    print(f"  trajectories      {len(b.trajectories)}")
    print(f"  frames            {b.n_frames:,} at {b.fps:g} fps")
    print(f"  subjects          {b.meta['subjects']}")
    print(f"  objects           {len(objs)}")
    print(f"  intents           {len(ints)}  {dict(ints.most_common(8))}")
    print(f"  object->intent cells {len(cells)}  -> {len(b.labels) / max(len(cells), 1):.2f} per cell")
    print(f"  flexion dominance {b.meta['flexion_axis_dominance']:.3f}  "
          f"residual {b.meta['projection_residual_deg']['mean']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

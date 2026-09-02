"""Verify the MANO skinning and mesh self-intersection checker.

Three checks, each of which would catch a different implementation error:

1. **Rest pose round-trip.** Identity rotations must reproduce ``v_template``
   to machine precision. A wrong joint-offset convention in the kinematic
   chain shows up here immediately.
2. **Rest pose is collision-free.** An open hand must report zero
   intersections. If it does not, the face-to-digit assignment or the
   triangle test is broken, and every later number is noise.
3. **A deliberately self-intersecting pose is caught.** Curling every finger
   far past its limit must produce intersections. Without this the checker
   could be returning "clean" for everything.

    python scripts/check_mesh_collision.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.mano import load_mano  # noqa: E402
from caredex.mesh_collision import (  # noqa: E402
    DIGIT_NAMES,
    MeshSelfIntersection,
    digit_of_faces,
    pose_mesh,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=r"D:\datasets\mano\MANO_RIGHT.pkl")
    args = ap.parse_args()

    model = load_mano(args.model)
    checker = MeshSelfIntersection(model)

    fd = digit_of_faces(model)
    counts = {DIGIT_NAMES[i]: int((fd == i).sum()) for i in range(len(DIGIT_NAMES))}
    print(f"faces per digit: {counts}")
    print(f"cross-digit face pairs to consider: {len(checker.pair_i):,}\n")

    failures = 0

    # 1. rest pose round-trip
    rest = pose_mesh(model, np.zeros((16, 3)), apply_pose_blend=False)
    err = float(np.abs(rest - model.v_template).max())
    ok = err < 1e-9
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] rest pose reproduces v_template (max err {err:.3e})")

    # 2. rest pose is collision-free
    t0 = time.time()
    rep = checker.check(rest)
    dt = time.time() - t0
    ok = not rep.intersects
    failures += not ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] open hand is collision-free "
        f"({rep.n_intersecting_pairs} hits from {rep.n_candidate_pairs:,} candidates, "
        f"{dt * 1000:.0f} ms)"
    )
    if rep.intersects:
        print(f"         digits involved: {rep.digits_involved}")

    # 3. an over-curled hand must be caught
    aa = np.zeros((16, 3))
    for j in range(1, 16):
        # Rotate about the MANO flexion axis far beyond any anatomical limit.
        aa[j, 2] = 1.6
    curled = pose_mesh(model, aa)
    rep2 = checker.check(curled)
    ok = rep2.intersects
    failures += not ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] over-curled hand is flagged "
        f"({rep2.n_intersecting_pairs} intersecting pairs)"
    )
    if rep2.intersects:
        print(f"         digits involved: {rep2.digits_involved}")

    print(f"\n{'all checks passed' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Is the capsule proxy calibrated well enough to be worth running?

The capsule self-intersection check reported a 96.9% interpenetration rate on
real OakInk data where the mesh check reported 8.3%. A proxy that fires on
essentially every real pose carries no information -- it is not conservative,
it is broken. Its hand-written segment lengths and base positions simply did
not match MANO's hand.

This script recalibrates the capsule geometry from MANO's rest-pose joints and
checks the result against the mesh check on the same frames. Pass condition:
the capsule rate is in the same ballpark as the mesh rate, and the rest pose
is clear.

    python scripts/check_capsule_calibration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex import kinematics  # noqa: E402
from caredex.data.base import TrajectoryBundle  # noqa: E402
from caredex.hand_model import N_DOF  # noqa: E402
from caredex.mano import load_mano  # noqa: E402
from caredex.mesh_collision import mesh_interpenetration_rate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mano", default=r"D:\datasets\mano\MANO_RIGHT.pkl")
    ap.add_argument("--bundle", default="data/bundles/oakink.npz")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    model = load_mano(args.mano)
    bundle = TrajectoryBundle.load(args.bundle)
    q = np.concatenate(bundle.trajectories[:60], axis=0)
    if len(q) > args.frames:
        q = q[:: len(q) // args.frames][: args.frames]

    before = float((kinematics.interpenetration(q) > 1e-4).mean())
    rest_before = float(kinematics.interpenetration(np.zeros((1, N_DOF), np.float32))[0])

    radii = kinematics.calibrate_from_mano(model)

    after = float((kinematics.interpenetration(q) > 1e-4).mean())
    rest_after = float(kinematics.interpenetration(np.zeros((1, N_DOF), np.float32))[0])

    mesh = mesh_interpenetration_rate(
        q,
        model,
        flexion_axis="xyz".find(bundle.meta.get("flexion_axis", "z")),
        flexion_sign=float(bundle.meta.get("flexion_sign", 1.0)),
        max_frames=args.frames,
    )
    mesh_rate = mesh["mesh_interpenetration_rate"]

    print(f"frames scored: {len(q)}  (mesh checked {mesh['frames_checked']})\n")
    print(f"{'':<26}{'rest pose':>12}{'real data':>12}")
    print("-" * 50)
    print(f"{'capsule, hand-written':<26}{rest_before:>12.5f}{before:>12.2%}")
    print(f"{'capsule, MANO-calibrated':<26}{rest_after:>12.5f}{after:>12.2%}")
    print(f"{'mesh (reference)':<26}{'-':>12}{mesh_rate:>12.2%}")
    print(f"\ncalibrated radii: " + ", ".join(f"{k}={v * 1000:.1f}mm" for k, v in radii.items()))

    ok = rest_after <= 0.0 and after <= max(4 * mesh_rate, mesh_rate + 0.10)
    print()
    if ok:
        print(
            f"PASS: rest pose clear, and the proxy's {after:.1%} is within a usable factor "
            f"of the mesh reference {mesh_rate:.1%}."
        )
        return 0
    print(
        f"FAIL: proxy reports {after:.1%} against a mesh reference of {mesh_rate:.1%} "
        f"(rest pose depth {rest_after:.5f}). Still not informative -- prefer the mesh "
        "check and do not quote capsule numbers."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Does the contact descriptor actually distinguish grasps?

A descriptor that returns similar numbers for a fingertip pinch and a flat
palmar press is useless as an authoring interface, however principled it looks.
This checks three things:

1. **Rest pose touches nothing** when the probe is far away, and touches the
   palmar side when the probe is placed against it. If not, the sign convention
   or the region assignment is wrong.
2. **Different grasps give different descriptors.** Pinch, power grasp and flat
   palm must separate -- measured as pairwise distance between their descriptor
   vectors relative to the within-grasp spread.
3. **Distal regions dominate for pinches, palm dominates for presses.** This is
   the anatomical sanity check; a descriptor that says a pinch is mostly palm
   contact has its region map inverted.

    python scripts/check_contact_descriptor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caredex.contact import (  # noqa: E402
    REGION_NAMES,
    ContactExtractor,
    ProbeSurface,
    fit_probe_plane,
)
from caredex.data.synthetic import PRIMITIVE_NAMES, primitive_matrix  # noqa: E402
from caredex.hand_model import ARTICULATED_SLICE, N_DOF  # noqa: E402
from caredex.mano import load_mano  # noqa: E402
from caredex.mesh_collision import dof_to_mano_rotations, pose_mesh  # noqa: E402


def pose_from_primitive(name: str) -> np.ndarray:
    q = np.zeros(N_DOF, dtype=np.float32)
    q[ARTICULATED_SLICE] = primitive_matrix()[PRIMITIVE_NAMES.index(name)]
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mano", default=r"D:\datasets\mano\MANO_RIGHT.pkl")
    ap.add_argument("--eps", type=float, default=0.004)
    args = ap.parse_args()

    model = load_mano(args.mano)
    ex = ContactExtractor(model, eps=args.eps)
    palm_mask = ex.region_of_vertex == REGION_NAMES.index("palm")

    failures = 0

    # 1. far probe touches nothing; fitted probe touches the palmar side
    rest = pose_mesh(model, np.zeros((16, 3)), apply_pose_blend=False)
    far = ProbeSurface(kind="plane", origin=np.array([0.0, 0.0, 10.0]), axis=np.array([0.0, 0.0, 1.0]))
    d_far = ex.extract(rest, far)
    ok = d_far.total_area == 0.0
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] distant probe: {d_far.total_area * 1e4:.3f} cm^2 contact")

    probe = fit_probe_plane(rest, palm_mask)
    d_rest = ex.extract(rest, probe)
    ok = d_rest.total_area > 0 and d_rest.n_regions >= 2
    failures += not ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] fitted palmar plane: "
        f"{d_rest.total_area * 1e4:.2f} cm^2 across {d_rest.n_regions} regions"
    )

    # 2 & 3. do different grasps separate, and in the anatomically right way?
    grasps = ["precision_pinch", "power_cylindrical", "palm_smooth", "open_flat", "lateral_key"]
    vecs, descs = {}, {}
    print(f"\n{'grasp':<20}{'area cm^2':>10}{'regions':>9}   dominant regions")
    print("-" * 78)
    for g in grasps:
        q = pose_from_primitive(g)
        verts = pose_mesh(model, dof_to_mano_rotations(q))
        p = fit_probe_plane(verts, palm_mask)
        d = ex.extract(verts, p)
        vecs[g] = d.to_vector()
        descs[g] = d
        top = np.argsort(-d.area)[:3]
        names = ", ".join(f"{REGION_NAMES[i]}({d.area[i] * 1e4:.1f})" for i in top if d.area[i] > 0)
        print(f"{g:<20}{d.total_area * 1e4:>10.2f}{d.n_regions:>9}   {names}")

    mat = np.stack([vecs[g] for g in grasps])
    dists = np.linalg.norm(mat[:, None] - mat[None, :], axis=-1)
    tri = dists[np.triu_indices(len(grasps), k=1)]
    print(f"\npairwise descriptor distance: min {tri.min():.3f}  mean {tri.mean():.3f}  max {tri.max():.3f}")
    ok = tri.min() > 1e-3
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] all grasp pairs are distinguishable")

    pinch = descs["precision_pinch"]
    press = descs["palm_smooth"]
    distal_idx = [i for i, n in enumerate(REGION_NAMES) if n.endswith("_dist")]
    palm_idx = REGION_NAMES.index("palm")
    pinch_distal = pinch.area[distal_idx].sum() / max(pinch.total_area, 1e-12)
    press_palm = press.area[palm_idx] / max(press.total_area, 1e-12)
    ok = pinch_distal > press.area[distal_idx].sum() / max(press.total_area, 1e-12)
    failures += not ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] pinch is more distal-dominated than a palmar press "
        f"({pinch_distal:.1%} vs {press.area[distal_idx].sum() / max(press.total_area, 1e-12):.1%} "
        f"of contact area; press palm share {press_palm:.1%})"
    )

    print(f"\n{'all checks passed' if not failures else f'{failures} CHECK(S) FAILED'}")
    print(f"descriptor vector size: {len(vecs[grasps[0]])}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

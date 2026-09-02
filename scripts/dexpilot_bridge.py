"""Retarget hand poses with dex-retargeting's DexPilot optimizer, out of process.

Runs inside the ``dexret2`` conda environment, not the project venv: it needs
Pinocchio, which has no working pip wheel here, and the venv's torch is pinned
to the cu128 index. See ``scripts/setup_dexpilot_env.sh``.

Reads a ``(n_clips, T, 21, 3)`` array of human hand keypoints -- MANO's 21
joints in metres, wrist first -- and writes ``(n_clips, T, n_joints)`` of Shadow
Hand joint angles. Both as ``.npy``, because the two environments cannot share
an interpreter and a file is the least fragile boundary available.

The optimizer wants *vectors between keypoints*, not the keypoints themselves,
and it names the pairs it wants in ``target_link_human_indices``: a ``(2, 15)``
array whose columns are (origin, destination) indices into the 21 joints. Those
are read from the built optimizer rather than hardcoded, so a config change
cannot silently produce vectors it did not ask for.

    <dexret2 python> scripts/dexpilot_bridge.py in.npy out.npy
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

URDF_DIR = r"D:\datasets\dex-urdf\robots\hands"
CONFIG = "configs/teleop/shadow_hand_right_dexpilot.yml"


def build():
    from dex_retargeting.retargeting_config import RetargetingConfig
    import dex_retargeting

    RetargetingConfig.set_default_urdf_dir(URDF_DIR)
    cfg_path = pathlib.Path(dex_retargeting.__file__).parent / CONFIG
    cfg = RetargetingConfig.load_from_file(cfg_path)
    return cfg.build(), cfg.urdf_path


def _collision_checker(urdf_path: str, joint_names: list[str]):
    """Return a predicate: does this joint vector self-collide?

    Model and geometry are both built from the original URDF, so kinematics and
    collision meshes cannot disagree. The optimizer's joint order is mapped onto
    this model's configuration vector by name rather than assumed to match.

    Collision is measured here rather than by mapping the angles onto MuJoCo's
    Shadow Hand. The two are independent models of the same robot: same joint
    names, but nothing guarantees the same zero or sign convention, and mapping
    across them reported 100% penetration on *real* human motion where direct
    joint transfer gives 30% -- a wiring artefact, not a property of the
    retargeting.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf_path)
    geom = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION,
                                 package_dirs=[str(pathlib.Path(URDF_DIR)), str(pathlib.Path(urdf_path).parent)])
    geom.addAllCollisionPairs()

    # Disable pairs that collide in most random configurations. This is what
    # MoveIt's setup assistant produces as an SRDF, and dex-urdf ships none:
    # adjacent links touch by construction, and a coarse convex collision mesh
    # can overlap its neighbour at every flexed pose while just clearing it at
    # neutral. Filtering on the neutral pose alone removed 25 of 939 pairs and
    # still reported every frame as colliding.
    #
    # The threshold is deliberately high. A pair that collides in 90% of random
    # configurations is describing the robot's geometry; one that collides in
    # 10% is describing a pose.
    data, gdata = model.createData(), geom.createData()
    rng = np.random.default_rng(0)
    # Some URDF joints carry infinite limits, and sampling uniform(-inf, inf)
    # silently produced garbage configurations, so the filter only ever caught
    # the pairs the neutral pose already showed.
    lo = np.nan_to_num(model.lowerPositionLimit, neginf=-np.pi)
    hi = np.nan_to_num(model.upperPositionLimit, posinf=np.pi)
    hits = np.zeros(len(geom.collisionPairs))
    n_probe = 200
    for _ in range(n_probe):
        q = rng.uniform(lo, hi)
        pin.computeCollisions(model, data, geom, gdata, q, False)
        hits += [gdata.collisionResults[i].isCollision()
                 for i in range(len(geom.collisionPairs))]
    always = hits / n_probe > 0.5
    for i in reversed(range(len(geom.collisionPairs))):
        if always[i]:
            geom.removeCollisionPair(geom.collisionPairs[i])
    print(f"  disabled {int(always.sum())} of {len(always)} pairs as structural", flush=True)
    gdata = geom.createData()

    # Pinocchio orders q by its own joint indexing; the optimizer hands back a
    # vector in its order. Map by name, drop anything this model does not have.
    slot = []
    for n in joint_names:
        jid = model.getJointId(n)
        slot.append(int(model.idx_qs[jid]) if jid < model.njoints else -1)

    from collections import Counter
    seen = Counter()

    def report():
        return seen.most_common(6)

    def collides(theta: np.ndarray) -> bool:
        q = pin.neutral(model)
        for v, a in zip(theta, slot):
            if a >= 0:
                q[a] = v
        hit = bool(pin.computeCollisions(model, data, geom, gdata, q, False))
        if hit:
            for i in range(len(geom.collisionPairs)):
                if gdata.collisionResults[i].isCollision():
                    seen[(geom.geometryObjects[geom.collisionPairs[i].first].name,
                          geom.geometryObjects[geom.collisionPairs[i].second].name)] += 1
        return hit

    collides.report = report
    return collides


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

    keypoints = np.load(src)
    if keypoints.ndim != 4 or keypoints.shape[2:] != (21, 3):
        print(f"expected (n_clips, T, 21, 3), got {keypoints.shape}")
        return 1

    retarget, retarget_urdf = build()
    idx = retarget.optimizer.target_link_human_indices
    origin, dest = np.asarray(idx[0]), np.asarray(idx[1])

    n_clips, n_frames = keypoints.shape[:2]
    out = np.zeros((n_clips, n_frames, len(retarget.joint_names)), dtype=np.float32)
    for c in range(n_clips):
        # Each clip starts from the retargeter's own neutral state. Carrying the
        # solution across clips would let one composition's final pose bias the
        # next one's fit, which is exactly the seen/unseen confound this whole
        # experiment is built to avoid.
        retarget.reset()
        for t in range(n_frames):
            k = keypoints[c, t]
            out[c, t] = retarget.retarget(k[dest] - k[origin])
        if (c + 1) % 10 == 0:
            print(f"  {c + 1}/{n_clips} clips", flush=True)

    # Self-collision is measured here, on the same URDF the optimizer fitted,
    # rather than by mapping the angles onto MuJoCo's Shadow Hand. The two are
    # independent models of the same robot: same joint names, but nothing
    # guarantees the same zero or sign convention, and mapping across them
    # produced 100% penetration on *real* human motion where direct joint
    # transfer gives 30% -- a wiring artefact, not a property of the retargeting.
    collide = _collision_checker(retarget_urdf, list(retarget.joint_names))
    rates = np.array([[collide(out[c, t]) for t in range(n_frames)]
                      for c in range(n_clips)], dtype=bool)

    for pair, cnt in collide.report():
        print(f"  colliding pair {pair[0]} ~ {pair[1]}: {cnt}", flush=True)
    np.save(dst.with_suffix(".collide.npy"), rates)
    np.save(dst, out)
    # Joint names go in a sidecar rather than stdout: the caller has to map them
    # onto MuJoCo's qpos addresses, and parsing a truncated print is exactly the
    # kind of fragile coupling that breaks silently when a config changes.
    import json
    dst.with_suffix(".joints.json").write_text(
        json.dumps(list(retarget.joint_names)), encoding="utf-8")
    print(f"wrote {dst}  {out.shape}  {len(retarget.joint_names)} joints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

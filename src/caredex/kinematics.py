"""Approximate forward kinematics for self-intersection checking.

Purpose and limits
------------------
The ergonomic validator has to answer "does this decoded pose put fingers
through each other?". The rigorous answer needs the MANO hand mesh, which is
behind a registration wall and is not required for anything else in Phase 1-2.
This module is the stopgap: a stick-figure hand with nominal adult segment
lengths, fingers modelled as capsules, self-intersection tested as
capsule-capsule distance.

It is a PROXY. It will miss shallow mesh-level interpenetration and it does not
model the palm at all. Report its output as "capsule-model interpenetration",
never as verified collision-free. Replace with MANO mesh checking once
``MANO_RIGHT.pkl`` is available.

Frame convention (right hand, palm down):
    +x  radial (toward the thumb)
    +y  distal (toward the fingertips)
    +z  palmar (out of the palm, the direction the fingers curl toward)

Flexion rotates about -x, abduction about +z.
"""

from __future__ import annotations

import numpy as np

from caredex.hand_model import DOF_INDEX, FINGERS, N_DOF

# Nominal adult-hand geometry in metres.
#
# WARNING: these hand-written constants do not match MANO's actual hand, and
# using them made the capsule check report a 96.9% interpenetration rate on real
# OakInk data where the mesh check reports 8.3%. A proxy that fires on
# essentially every real pose carries no information. Call
# :func:`calibrate_from_mano` at import time in any code path that has the MANO
# model, and treat the constants below only as a fallback for code that does
# not.


def calibrate_from_mano(model) -> dict[str, float]:
    """Overwrite the capsule geometry from MANO's rest-pose joints.

    Replaces the hand-written base positions and phalanx lengths with distances
    measured off ``model.J`` (16 rest-pose joint locations), so the capsule
    model lives at the same scale and in the same frame as the mesh. Radii are
    then shrunk until the rest pose is collision-free with a margin, which is
    the minimum any proxy must satisfy to be worth running.

    Returns the calibrated radii, and mutates the module-level tables.
    """
    from caredex.data.oakink import MANO_JOINT_MAP

    if model.J is None:
        raise ValueError("MANO model has no rest-pose joints (J)")
    J = np.asarray(model.J, dtype=np.float32)

    joints_of: dict[str, list[int]] = {}
    for joint, (digit, _level) in MANO_JOINT_MAP.items():
        joints_of.setdefault(digit, []).append(joint)
    for digit in joints_of:
        joints_of[digit].sort()

    for finger in FINGERS:
        chain = joints_of[finger]
        MCP_BASE[finger] = J[chain[0]] - J[0]
        lengths = [float(np.linalg.norm(J[chain[i + 1]] - J[chain[i]])) for i in range(2)]
        # MANO has no fingertip joint; the distal phalanx is approximated as
        # 80% of the middle one, the usual anthropometric ratio.
        PHALANX_LENGTHS[finger] = (lengths[0], lengths[1], 0.8 * lengths[1])

    thumb = joints_of["thumb"]
    global THUMB_BASE, THUMB_LENGTHS
    THUMB_BASE = J[thumb[0]] - J[0]
    t_len = [float(np.linalg.norm(J[thumb[i + 1]] - J[thumb[i]])) for i in range(2)]
    THUMB_LENGTHS = (t_len[0], t_len[1], 0.8 * t_len[1])

    # Shrink radii until an open hand clears. Starting from half the smallest
    # inter-finger base spacing keeps them physically sensible.
    bases = np.stack([MCP_BASE[f] for f in FINGERS])
    spacing = float(np.min(np.linalg.norm(np.diff(bases, axis=0), axis=1)))
    for digit in FINGER_RADIUS:
        FINGER_RADIUS[digit] = 0.35 * spacing
    FINGER_RADIUS["thumb"] = 0.40 * spacing

    from caredex.hand_model import N_DOF

    for _ in range(12):
        if float(interpenetration(np.zeros((1, N_DOF), dtype=np.float32))[0]) <= 0.0:
            break
        for digit in FINGER_RADIUS:
            FINGER_RADIUS[digit] *= 0.85
    return dict(FINGER_RADIUS)


# Fallback constants, used only when MANO is unavailable. See the warning above.
MCP_BASE: dict[str, np.ndarray] = {
    "index": np.array([0.022, 0.095, 0.0], dtype=np.float32),
    "middle": np.array([0.000, 0.099, 0.0], dtype=np.float32),
    "ring": np.array([-0.021, 0.094, 0.0], dtype=np.float32),
    "pinky": np.array([-0.041, 0.086, 0.0], dtype=np.float32),
}

#: (proximal, middle, distal) phalanx lengths.
PHALANX_LENGTHS: dict[str, tuple[float, float, float]] = {
    "index": (0.045, 0.025, 0.020),
    "middle": (0.050, 0.030, 0.021),
    "ring": (0.046, 0.028, 0.020),
    "pinky": (0.036, 0.020, 0.018),
}

#: Capsule radius per finger, roughly half the digit width.
FINGER_RADIUS: dict[str, float] = {
    "index": 0.0095,
    "middle": 0.0095,
    "ring": 0.0090,
    "pinky": 0.0080,
    "thumb": 0.0110,
}

THUMB_BASE = np.array([0.032, 0.030, 0.008], dtype=np.float32)
THUMB_LENGTHS = (0.045, 0.032, 0.025)  # metacarpal, proximal, distal


def _rot_x(a: np.ndarray) -> np.ndarray:
    """Rotation about +x by ``a`` radians, batched over ``a.shape``."""
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack(
        [
            np.stack([o, z, z], axis=-1),
            np.stack([z, c, -s], axis=-1),
            np.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack(
        [
            np.stack([c, -s, z], axis=-1),
            np.stack([s, c, z], axis=-1),
            np.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def finger_chain(q: np.ndarray, finger: str) -> np.ndarray:
    """Joint positions of one finger: ``(..., 4, 3)`` = MCP, PIP, DIP, tip."""
    d2r = np.pi / 180.0
    mcp_f = -q[..., DOF_INDEX[f"{finger}_mcp_flex"]] * d2r
    mcp_a = q[..., DOF_INDEX[f"{finger}_mcp_abd"]] * d2r
    pip_f = -q[..., DOF_INDEX[f"{finger}_pip_flex"]] * d2r
    dip_f = -q[..., DOF_INDEX[f"{finger}_dip_flex"]] * d2r

    l1, l2, l3 = PHALANX_LENGTHS[finger]
    base = MCP_BASE[finger]

    r_mcp = _rot_z(mcp_a) @ _rot_x(mcp_f)
    r_pip = r_mcp @ _rot_x(pip_f)
    r_dip = r_pip @ _rot_x(dip_f)

    def along_y(rot: np.ndarray, length: float) -> np.ndarray:
        v = np.zeros(rot.shape[:-2] + (3,), dtype=np.float32)
        v[..., 1] = length
        return np.einsum("...ij,...j->...i", rot, v)

    p0 = np.broadcast_to(base, q.shape[:-1] + (3,)).astype(np.float32)
    p1 = p0 + along_y(r_mcp, l1)
    p2 = p1 + along_y(r_pip, l2)
    p3 = p2 + along_y(r_dip, l3)
    return np.stack([p0, p1, p2, p3], axis=-2)


def thumb_chain(q: np.ndarray) -> np.ndarray:
    """Joint positions of the thumb: ``(..., 4, 3)`` = CMC, MCP, IP, tip.

    Cruder than the fingers: the trapeziometacarpal joint is a saddle whose
    axes are neither orthogonal nor aligned with the palm, and this treats it
    as an abduction-then-flexion Euler pair. Adequate for a distance-based
    clearance check, not for anything kinematically quantitative.
    """
    d2r = np.pi / 180.0
    cmc_f = -q[..., DOF_INDEX["thumb_cmc_flex"]] * d2r
    cmc_a = q[..., DOF_INDEX["thumb_cmc_abd"]] * d2r
    mcp_f = -q[..., DOF_INDEX["thumb_mcp_flex"]] * d2r
    mcp_a = q[..., DOF_INDEX["thumb_mcp_abd"]] * d2r
    ip_f = -q[..., DOF_INDEX["thumb_ip_flex"]] * d2r

    l1, l2, l3 = THUMB_LENGTHS

    # The thumb metacarpal leaves the palm pointing radially and distally, so
    # its rest direction is rotated ~50 degrees from the finger convention.
    rest = _rot_z(np.full_like(cmc_a, 50.0 * d2r))
    r_cmc = rest @ _rot_z(cmc_a) @ _rot_x(cmc_f)
    r_mcp = r_cmc @ _rot_z(mcp_a) @ _rot_x(mcp_f)
    r_ip = r_mcp @ _rot_x(ip_f)

    def along_y(rot: np.ndarray, length: float) -> np.ndarray:
        v = np.zeros(rot.shape[:-2] + (3,), dtype=np.float32)
        v[..., 1] = length
        return np.einsum("...ij,...j->...i", rot, v)

    p0 = np.broadcast_to(THUMB_BASE, q.shape[:-1] + (3,)).astype(np.float32)
    p1 = p0 + along_y(r_cmc, l1)
    p2 = p1 + along_y(r_mcp, l2)
    p3 = p2 + along_y(r_ip, l3)
    return np.stack([p0, p1, p2, p3], axis=-2)


def all_chains(q: np.ndarray) -> dict[str, np.ndarray]:
    """Every digit's joint positions, keyed by digit name. Wrist DOF ignored.

    The wrist is a rigid transform applied to the whole hand, so it cannot
    change self-intersection; leaving it out keeps the check in hand-local
    coordinates.
    """
    if q.shape[-1] != N_DOF:
        raise ValueError(f"expected trailing dim {N_DOF}, got {q.shape}")
    chains = {f: finger_chain(q, f) for f in FINGERS}
    chains["thumb"] = thumb_chain(q)
    return chains


def fingertips(q: np.ndarray) -> np.ndarray:
    """``(..., 5, 3)`` tip positions in digit order (index..pinky, thumb)."""
    chains = all_chains(q)
    order = list(FINGERS) + ["thumb"]
    return np.stack([chains[d][..., 3, :] for d in order], axis=-2)


# ---------------------------------------------------------------------------
# Capsule-capsule clearance
# ---------------------------------------------------------------------------


def _segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """Shortest distance between two 3D segments, batched over leading dims.

    Standard clamped-parameter solution (Ericson, *Real-Time Collision
    Detection*, section 5.1.9), vectorised.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)

    denom = a * e - b * b
    eps = 1e-12
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0, 1), 0.0)
    t = np.clip((b * s + f) / np.maximum(e, eps), 0, 1)
    s = np.clip((b * t - c) / np.maximum(a, eps), 0, 1)

    closest = (p1 + s[..., None] * d1) - (p2 + t[..., None] * d2)
    return np.linalg.norm(closest, axis=-1)


#: Digit pairs whose segments are tested. Adjacent phalanges of the same digit
#: always touch at their shared joint, so within-digit pairs are excluded.
_DIGIT_ORDER = list(FINGERS) + ["thumb"]


def interpenetration(q: np.ndarray) -> np.ndarray:
    """Worst capsule overlap depth in metres, shape ``q.shape[:-1]``.

    Zero means every pair of digit segments clears by at least the sum of their
    radii. Positive values are how deep the deepest pair overlaps.
    """
    chains = all_chains(q)
    depth = np.zeros(q.shape[:-1], dtype=np.float32)

    for i, a in enumerate(_DIGIT_ORDER):
        for b in _DIGIT_ORDER[i + 1 :]:
            ra, rb = FINGER_RADIUS[a], FINGER_RADIUS[b]
            ca, cb = chains[a], chains[b]
            for sa in range(3):
                for sb in range(3):
                    d = _segment_distance(
                        ca[..., sa, :], ca[..., sa + 1, :],
                        cb[..., sb, :], cb[..., sb + 1, :],
                    )
                    depth = np.maximum(depth, (ra + rb) - d)
    return np.maximum(depth, 0.0)

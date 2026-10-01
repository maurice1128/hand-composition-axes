"""Shared MANO -> anatomical-DOF retargeting.

Both OakInk and DexYCB annotate hands as MANO joint rotations, so the
projection onto the 27-DOF anatomical model in :mod:`caredex.hand_model` is the
same problem twice. It lives here so the two adapters cannot drift apart, and
so a convention verified on one dataset is literally the same code on the other.

The projection is lossy and the loss is reported, not hidden: MANO gives every
joint a full 3-DOF rotation while a PIP or DIP has one flexion DOF, so the
off-axis component is discarded and its magnitude returned as a residual.

Nothing here assumes a convention. The quaternion layout, the flexion axis and
the flexion sign are all inferred from the data, because the first version of
the OakInk adapter hard-coded a sign, got it backwards, and pinned 41% of DOF
values at their limits while still looking plausible in aggregate.
"""

from __future__ import annotations

import numpy as np

from caredex.hand_model import DOF_INDEX, LIMITS_HI, LIMITS_LO, N_DOF

#: MANO kinematic tree: joint -> parent. Index 0 is the wrist.
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)

#: MANO joint index -> (anatomical digit, level). MANO orders the digits
#: index, middle, pinky, ring, thumb -- pinky comes before ring.
MANO_JOINT_MAP: dict[int, tuple[str, str]] = {
    1: ("index", "mcp"), 2: ("index", "pip"), 3: ("index", "dip"),
    4: ("middle", "mcp"), 5: ("middle", "pip"), 6: ("middle", "dip"),
    7: ("pinky", "mcp"), 8: ("pinky", "pip"), 9: ("pinky", "dip"),
    10: ("ring", "mcp"), 11: ("ring", "pip"), 12: ("ring", "dip"),
    13: ("thumb", "cmc"), 14: ("thumb", "mcp"), 15: ("thumb", "ip"),
}

#: Joint indices within the 15 non-root MANO joints (i.e. full index minus 1).
PIP_JOINTS_15 = (1, 4, 7, 10)
DISTAL_JOINTS_15 = (2, 5, 8, 11)

#: Same, as full 16-joint indices.
PIP_JOINTS_16 = (2, 5, 8, 11)
DISTAL_JOINTS_16 = (3, 6, 9, 12)


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


def quat_to_matrix(q: np.ndarray, w_first: bool) -> np.ndarray:
    """``(..., 4)`` quaternions -> ``(..., 3, 3)`` rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    if w_first:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` -> intrinsic XYZ Euler angles in degrees, ``(..., 3)``."""
    sy = np.clip(-r[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sy)
    near_lock = np.abs(np.cos(y)) < 1e-6
    x = np.where(near_lock, np.arctan2(-r[..., 1, 2], r[..., 1, 1]), np.arctan2(r[..., 2, 1], r[..., 2, 2]))
    z = np.where(near_lock, 0.0, np.arctan2(r[..., 1, 0], r[..., 0, 0]))
    return np.degrees(np.stack([x, y, z], axis=-1))


def matrix_angle(r: np.ndarray) -> np.ndarray:
    """Rotation magnitude in degrees from a rotation matrix."""
    trace = np.clip((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def quat_angle(q: np.ndarray, w_first: bool) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    scalar = q[..., 0] if w_first else q[..., 3]
    n = np.linalg.norm(q, axis=-1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(scalar) / n, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Convention inference
# ---------------------------------------------------------------------------


def infer_quat_layout(poses: np.ndarray, distal_joints=DISTAL_JOINTS_16) -> tuple[bool, dict]:
    """Decide whether the scalar part comes first, from distal joint magnitudes.

    Distal interphalangeal joints move through a limited range, so the reading
    that makes them small is the correct one. Reading ``(w,x,y,z)`` as
    ``(x,y,z,w)`` turns a 19-degree median into 162 degrees -- not a close call.
    """
    distal = poses[:, list(distal_joints), :]
    med_first = float(np.median(quat_angle(distal, True)))
    med_last = float(np.median(quat_angle(distal, False)))
    return med_first <= med_last, {
        "median_distal_angle_w_first": med_first,
        "median_distal_angle_w_last": med_last,
    }


def infer_flexion_axis(euler: np.ndarray, pip_joints=PIP_JOINTS_16) -> tuple[int, dict]:
    """Identify which Euler axis carries flexion, from PIP joint variance.

    A PIP joint is anatomically close to a single hinge, so nearly all its
    rotational variance sits on one axis. ``dominance`` reports how well that
    1-DOF assumption actually fits; a low value means the projection is lossy
    and the residual should be taken seriously.
    """
    pip = euler[:, list(pip_joints), :]
    # Per joint over time, THEN averaged across joints. Pooling both at once
    # -- ``std(axis=(0, 1))`` -- adds the spread of the four PIPs' resting
    # offsets to the spread of their motion, and the offsets are not flexion.
    # On OakInk2 that inflated the x axis (per-joint means +11.9, -13.4, -28.7,
    # -0.2) until it beat the true flexion axis z (per-joint std 14-18 degrees
    # against 2-11), and the resulting sign flip pinned index_pip_flex at its
    # lower limit in 100% of frames. The same dilution pushed the reported
    # "dominance" of GRAB and DexYCB down to ~0.45, which was misread as those
    # datasets being intrinsically noisy.
    spread = pip.std(axis=0).mean(axis=0)
    axis = int(np.argmax(spread))
    total = float(spread.sum())
    return axis, {
        "pip_axis_std_x": float(spread[0]),
        "pip_axis_std_y": float(spread[1]),
        "pip_axis_std_z": float(spread[2]),
        "flexion_axis_dominance": float(spread[axis] / total) if total > 0 else 0.0,
    }


def infer_flexion_sign(
    euler: np.ndarray, flex_axis: int, pip_joints=PIP_JOINTS_16,
    lo: float = -5.0, hi: float = 115.0,
) -> tuple[float, dict]:
    """Decide whether flexion is the positive or negative axis direction.

    A PIP flexes into [0, 110] degrees with essentially no hyperextension, so
    the correct sign puts most PIP samples in that range. Getting this backwards
    is not subtle: it pins every PIP at 0 and every MCP at its lower limit.

    ``lo``/``hi`` widen for the thumb IP, whose anatomical range is [-15, 80].
    """
    pip = euler[:, list(pip_joints), flex_axis]
    frac_pos = float(np.mean((pip >= lo) & (pip <= hi)))
    frac_neg = float(np.mean((-pip >= lo) & (-pip <= hi)))
    return (1.0 if frac_pos >= frac_neg else -1.0), {
        "pip_in_range_positive": frac_pos,
        "pip_in_range_negative": frac_neg,
    }


def infer_abduction(
    euler: np.ndarray, flex_axis: int, joints: tuple[int, ...]
) -> tuple[int, float, dict]:
    """Pick which remaining Euler axis (and sign) carries abduction.

    Flexion inference leaves two axes; the old code took ``(flex_axis + 1) % 3``
    by convention and never checked. For the fingers that happened to be right.
    For the thumb it was not: MANO gives every joint the same rest frame -- each
    joint's rest transform relative to its parent is a pure translation -- so the
    thumb's frame is *not* aligned to its own anatomy and its flexion and
    abduction land on different Euler components from the fingers'.
    ``thumb_mcp_abd`` spans only [-10, 10] degrees, and the unchecked
    choice put **100% of GRAB and DexYCB frames outside that box**, where
    clamping flattened the DOF to a constant. No numerical check caught it
    because the clip-fraction gate was being fed post-clamp values.

    So this chooses among the four (axis, sign) options by how much of the data
    the anatomical limits actually admit -- the same "infer, never assume"
    rule the flexion axis already followed.
    """
    others = [a for a in range(3) if a != flex_axis]
    best, stats = None, {}
    for axis in others:
        for sign in (1.0, -1.0):
            in_range = []
            for j in joints:
                name = dof_names_for(j)[1]
                if name is None:
                    continue
                i = DOF_INDEX[name]
                v = sign * euler[:, j, axis]
                in_range.append(np.mean((v >= LIMITS_LO[i]) & (v <= LIMITS_HI[i])))
            frac = float(np.mean(in_range)) if in_range else 0.0
            stats[f"abd_in_range_{'xyz'[axis]}{'+' if sign > 0 else '-'}"] = frac
            if best is None or frac > best[2]:
                best = (axis, sign, frac)
    return best[0], best[1], {**stats, "abd_in_range_chosen": best[2]}


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def dof_names_for(joint: int) -> tuple[str, str | None]:
    """The anatomical DOF a MANO joint maps onto: (flexion, abduction-or-None)."""
    digit, level = MANO_JOINT_MAP[joint]
    if digit == "thumb":
        return {
            "cmc": ("thumb_cmc_flex", "thumb_cmc_abd"),
            "mcp": ("thumb_mcp_flex", "thumb_mcp_abd"),
            "ip": ("thumb_ip_flex", None),
        }[level]
    if level == "mcp":
        return (f"{digit}_mcp_flex", f"{digit}_mcp_abd")
    return (f"{digit}_{level}_flex", None)


def euler_to_dof(
    euler: np.ndarray,
    translations: np.ndarray | None,
    flex_axis: int,
    flex_sign: float,
    conventions: dict[str, tuple[int, float, int, float]] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project ``(n, 16, 3)`` per-joint Euler angles onto the 27-DOF vector.

    Returns the poses and the projection residual -- the magnitude of rotation
    the anatomical model could not represent. Hinge joints (PIP, DIP, thumb IP)
    discard two of their three axes, and that discarded magnitude *is* the
    residual. It belongs in every report that uses this data.
    """
    n = len(euler)
    q = np.zeros((n, N_DOF), dtype=np.float32)
    residual_terms: list[np.ndarray] = []

    if conventions is None:
        # Legacy path: one axis for every digit, abduction assumed to be the
        # next axis round. Kept so old callers behave exactly as before, but
        # nothing in this repo should use it -- see RetargetConventions.
        conventions = {
            "finger": (flex_axis, flex_sign, (flex_axis + 1) % 3, 1.0),
            "thumb": (flex_axis, flex_sign, (flex_axis + 1) % 3, 1.0),
        }

    for joint in MANO_JOINT_MAP:
        digit, _ = MANO_JOINT_MAP[joint]
        f_axis, f_sign, a_axis, a_sign = conventions["thumb" if digit == "thumb" else "finger"]
        third_axis = 3 - f_axis - a_axis  # the one neither flexion nor abduction

        ang = euler[:, joint, :]
        flex_name, abd_name = dof_names_for(joint)
        q[:, DOF_INDEX[flex_name]] = f_sign * ang[:, f_axis]
        if abd_name is not None:
            q[:, DOF_INDEX[abd_name]] = a_sign * ang[:, a_axis]
            residual_terms.append(np.abs(ang[:, third_axis]))
        else:
            residual_terms.append(np.abs(ang[:, a_axis]))
            residual_terms.append(np.abs(ang[:, third_axis]))

    if translations is not None:
        for k, name in enumerate(("wrist_tx", "wrist_ty", "wrist_tz")):
            lo, hi = LIMITS_LO[DOF_INDEX[name]], LIMITS_HI[DOF_INDEX[name]]
            q[:, DOF_INDEX[name]] = np.clip(translations[:, k], lo, hi)

    res = np.concatenate(residual_terms) if residual_terms else np.zeros(1)
    return q, {
        "mean": float(res.mean()),
        "p95": float(np.percentile(res, 95)),
        "max": float(res.max()),
    }


#: MCP joints carrying finger abduction, and the thumb joints carrying its own.
FINGER_ABD_JOINTS_16 = (1, 4, 7, 10)
THUMB_ABD_JOINTS_16 = (13, 14)
#: The thumb's hinge, i.e. its analogue of a finger PIP.
THUMB_HINGE_16 = (15,)


class RetargetConventions:
    """Every convention needed to project MANO Euler angles onto 27 DOF.

    One object, inferred once over pooled data, applied to every sequence. The
    four adapters used to call :func:`infer_flexion_axis` and friends
    individually and assemble the rest themselves, which is how the thumb
    abduction axis ended up assumed in all four at once.

    The thumb gets its **own** flexion axis and sign, inferred from the thumb IP
    joint rather than the finger PIPs. MANO's joint frames are all parallel to the
    template frame, so the thumb's is no more anatomically aligned than the
    fingers' and a shared axis is wrong in principle; in practice it put
    100% of GRAB and DexYCB frames outside ``thumb_mcp_abd``'s 20-degree box.
    """

    def __init__(self, finger, thumb, stats):
        self.finger = finger  # (flex_axis, flex_sign, abd_axis, abd_sign)
        self.thumb = thumb
        self.stats = stats

    @classmethod
    def infer(cls, euler: np.ndarray) -> "RetargetConventions":
        f_axis, axis_stats = infer_flexion_axis(euler)
        f_sign, sign_stats = infer_flexion_sign(euler, f_axis)
        fa_axis, fa_sign, fa_stats = infer_abduction(euler, f_axis, FINGER_ABD_JOINTS_16)

        # The thumb IP flexes into [-15, 80], not a PIP's [0, 110].
        t_axis, t_axis_stats = infer_flexion_axis(euler, pip_joints=THUMB_HINGE_16)
        t_sign, t_sign_stats = infer_flexion_sign(
            euler, t_axis, pip_joints=THUMB_HINGE_16, lo=-20.0, hi=85.0)
        ta_axis, ta_sign, ta_stats = infer_abduction(euler, t_axis, THUMB_ABD_JOINTS_16)

        return cls(
            finger=(f_axis, f_sign, fa_axis, fa_sign),
            thumb=(t_axis, t_sign, ta_axis, ta_sign),
            stats={
                **axis_stats, **sign_stats,
                **{f"finger_{k}": v for k, v in fa_stats.items()},
                "thumb_flexion_axis": "xyz"[t_axis],
                "thumb_flexion_sign": float(t_sign),
                "thumb_axis_dominance": t_axis_stats["flexion_axis_dominance"],
                "thumb_ip_in_range_positive": t_sign_stats["pip_in_range_positive"],
                "thumb_ip_in_range_negative": t_sign_stats["pip_in_range_negative"],
                **{f"thumb_{k}": v for k, v in ta_stats.items()},
                "flexion_axis": "xyz"[f_axis],
                "flexion_sign": float(f_sign),
                "abduction_axis": "xyz"[fa_axis],
                "abduction_sign": float(fa_sign),
                "thumb_abduction_axis": "xyz"[ta_axis],
                "thumb_abduction_sign": float(ta_sign),
            },
        )

    def as_dict(self) -> dict[str, tuple[int, float, int, float]]:
        return {"finger": self.finger, "thumb": self.thumb}

    def apply(self, euler, translations):
        return euler_to_dof(euler, translations, self.finger[0], self.finger[1],
                            conventions=self.as_dict())

    def describe(self, tag: str) -> str:
        s = self.stats
        return (
            f"[{tag}] fingers: flexion {'xyz'[self.finger[0]]}{'+' if self.finger[1] > 0 else '-'} "
            f"(dominance {s['flexion_axis_dominance']:.3f}, PIP in range "
            f"{s['pip_in_range_positive']:.1%} vs {s['pip_in_range_negative']:.1%}); "
            f"abduction {s['abduction_axis']}{'+' if self.finger[3] > 0 else '-'} "
            f"({s['finger_abd_in_range_chosen']:.1%} inside limits)\n"
            f"[{tag}] thumb:   flexion {s['thumb_flexion_axis']}"
            f"{'+' if self.thumb[1] > 0 else '-'} (dominance {s['thumb_axis_dominance']:.3f}, "
            f"IP in range {s['thumb_ip_in_range_positive']:.1%} vs "
            f"{s['thumb_ip_in_range_negative']:.1%}); "
            f"abduction {s['thumb_abduction_axis']}{'+' if self.thumb[3] > 0 else '-'} "
            f"({s['thumb_abd_in_range_chosen']:.1%} inside limits)"
        )


#: Wrist rotation cannot be recovered from MANO at all: the model has no
#: forearm, so its root rotation is the hand's absolute orientation in the
#: capture frame, not an articulation of the wrist joint. Filling these three
#: DOF from the root would encode "where the camera was", which is exactly the
#: dataset-specific nuisance a motion prior must not learn. They stay zero, and
#: any per-DOF audit has to know that is a property of MANO rather than a bug.
MANO_UNAVAILABLE_DOF = ("wrist_rx_flex_ext", "wrist_ry_dev", "wrist_rz_pro_sup")


def center_wrist_translation(q: np.ndarray) -> np.ndarray:
    """Re-express wrist translation relative to the trajectory's own mean.

    MANO's ``transl`` is an absolute position in the capture rig's frame. The
    27-DOF box allows +-0.3 m, so absolute positions saturate it -- ``wrist_tz``
    sat at its upper limit in 100% of DexYCB and GRAB frames, which is a
    statement about where the camera stood, not about the hand.

    Centring per trajectory keeps what a motion prior can use (how the hand
    moved during this sequence) and drops what it must not (where the rig put
    it). Must be applied per trajectory, never over a pooled array, or one
    sequence's offset leaks into another's.
    """
    q = np.array(q, copy=True)
    idx = [DOF_INDEX[n] for n in ("wrist_tx", "wrist_ty", "wrist_tz")]
    q[:, idx] -= q[:, idx].mean(axis=0, keepdims=True)
    return q


def limit_clip_fraction(q: np.ndarray) -> float:
    """Fraction of DOF values outside their limits before clamping.

    A high value means the conversion is wrong, not that the data is extreme:
    the first OakInk attempt hit 41% with a flipped flexion sign, versus 3.4%
    once the sign was inferred rather than assumed.
    """
    return float(((q < LIMITS_LO) | (q > LIMITS_HI)).mean())

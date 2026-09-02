"""27-DOF human hand kinematic specification.

This module is the single source of truth for the action/state layout used
everywhere else in CareDex: data generation, the eigengrasp basis, the latent
action prior, and (later) the MuJoCo environment's actuator ordering.

DOF layout (27 total)
---------------------
    0-15   four fingers x 4 DOF  (index, middle, ring, pinky)
               MCP flexion, MCP abduction, PIP flexion, DIP flexion
    16-20  thumb x 5 DOF
               CMC flexion, CMC abduction, MCP flexion, MCP abduction, IP flexion
    21-26  wrist / global pose
               tx, ty, tz, rx, ry, rz

Indices 0-20 are the *articulated* DOF -- the hand configuration proper. The
eigengrasp basis and the hand prior operate on these by default; the wrist is a
global pose that the task policy controls directly and that carries no
grasp-synergy structure.

Joint limits below are approximate healthy-adult ranges of motion in degrees.
They bound the *hand*, not the patient. They are unrelated to the clinical
safety thresholds of Phase 0 (contact force / pressure / torque on skin), which
live in the environment's safety module and must be sourced separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_DOF = 27
N_ARTICULATED = 21
N_GLOBAL = 6

FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class DofSpec:
    """One degree of freedom: its name, its limits, and its units."""

    name: str
    lo: float
    hi: float
    unit: str  # "deg" or "m"

    @property
    def is_angular(self) -> bool:
        return self.unit == "deg"


def _finger_dofs(finger: str) -> list[DofSpec]:
    return [
        DofSpec(f"{finger}_mcp_flex", -20.0, 90.0, "deg"),
        DofSpec(f"{finger}_mcp_abd", -20.0, 20.0, "deg"),
        DofSpec(f"{finger}_pip_flex", 0.0, 110.0, "deg"),
        DofSpec(f"{finger}_dip_flex", -10.0, 90.0, "deg"),
    ]


_THUMB_DOFS = [
    DofSpec("thumb_cmc_flex", -15.0, 60.0, "deg"),
    DofSpec("thumb_cmc_abd", 0.0, 60.0, "deg"),
    DofSpec("thumb_mcp_flex", -10.0, 55.0, "deg"),
    DofSpec("thumb_mcp_abd", -10.0, 10.0, "deg"),
    DofSpec("thumb_ip_flex", -15.0, 80.0, "deg"),
]

# Wrist translation limits describe a generous reachable box around the pelvis
# scene origin; they exist to keep synthetic data bounded, not to model anatomy.
_WRIST_DOFS = [
    DofSpec("wrist_tx", -0.30, 0.30, "m"),
    DofSpec("wrist_ty", -0.30, 0.30, "m"),
    DofSpec("wrist_tz", -0.30, 0.30, "m"),
    DofSpec("wrist_rx_flex_ext", -70.0, 80.0, "deg"),
    DofSpec("wrist_ry_dev", -20.0, 35.0, "deg"),
    DofSpec("wrist_rz_pro_sup", -80.0, 80.0, "deg"),
]

DOF_SPECS: tuple[DofSpec, ...] = tuple(
    [d for f in FINGERS for d in _finger_dofs(f)] + _THUMB_DOFS + _WRIST_DOFS
)

assert len(DOF_SPECS) == N_DOF, f"expected {N_DOF} DOF, built {len(DOF_SPECS)}"

DOF_NAMES: tuple[str, ...] = tuple(d.name for d in DOF_SPECS)
DOF_INDEX: dict[str, int] = {name: i for i, name in enumerate(DOF_NAMES)}

ARTICULATED_SLICE = slice(0, N_ARTICULATED)
GLOBAL_SLICE = slice(N_ARTICULATED, N_DOF)

#: Lower/upper bounds in native units, shape (27,).
LIMITS_LO = np.array([d.lo for d in DOF_SPECS], dtype=np.float32)
LIMITS_HI = np.array([d.hi for d in DOF_SPECS], dtype=np.float32)

#: True where the DOF is an angle in degrees (all but the three wrist translations).
IS_ANGULAR = np.array([d.is_angular for d in DOF_SPECS], dtype=bool)


def dof_indices(*names: str) -> np.ndarray:
    """Resolve DOF names to an index array, raising on typos."""
    return np.array([DOF_INDEX[n] for n in names], dtype=np.int64)


def finger_dof_indices(finger: str) -> np.ndarray:
    """The four DOF indices of one finger, in (mcp_flex, mcp_abd, pip, dip) order."""
    if finger not in FINGERS:
        raise KeyError(f"unknown finger {finger!r}; expected one of {FINGERS}")
    return dof_indices(
        f"{finger}_mcp_flex",
        f"{finger}_mcp_abd",
        f"{finger}_pip_flex",
        f"{finger}_dip_flex",
    )


def clamp_to_limits(q: np.ndarray) -> np.ndarray:
    """Clip a pose or a batch of poses into the joint limit box."""
    _check_trailing_dim(q)
    return np.clip(q, LIMITS_LO, LIMITS_HI)


def limit_violations(q: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Per-DOF amount by which ``q`` exceeds its limits (0 where in range).

    Returned in native units, same shape as ``q``. This is the quantity the
    ergonomic validator reports as "no hyperextension".
    """
    _check_trailing_dim(q)
    below = np.maximum(LIMITS_LO - q - tol, 0.0)
    above = np.maximum(q - LIMITS_HI - tol, 0.0)
    return below + above


# ---------------------------------------------------------------------------
# Biomechanical coupling
# ---------------------------------------------------------------------------

#: Ratio linking distal interphalangeal to proximal interphalangeal flexion.
#: The DIP and PIP of a human finger are mechanically linked by the extensor
#: mechanism and the flexor digitorum profundus tendon, so DIP ~= (2/3) * PIP is
#: the standard first-order approximation used in hand-synergy work.
DIP_PIP_RATIO = 2.0 / 3.0


def apply_dip_pip_coupling(q: np.ndarray, ratio: float = DIP_PIP_RATIO) -> np.ndarray:
    """Overwrite each finger's DIP flexion with ``ratio * PIP`` flexion.

    Applied when synthesising data so that the generated set carries the same
    kind of hard linear dependency real hands do. A prior that fails to recover
    it is broken.
    """
    _check_trailing_dim(q)
    out = np.array(q, dtype=np.float32, copy=True)
    for finger in FINGERS:
        pip = DOF_INDEX[f"{finger}_pip_flex"]
        dip = DOF_INDEX[f"{finger}_dip_flex"]
        out[..., dip] = ratio * out[..., pip]
    return clamp_to_limits(out)


def coupling_residual(q: np.ndarray, ratio: float = DIP_PIP_RATIO) -> np.ndarray:
    """Absolute DIP/PIP coupling error in degrees, shape ``q.shape[:-1] + (4,)``."""
    _check_trailing_dim(q)
    res = []
    for finger in FINGERS:
        pip = q[..., DOF_INDEX[f"{finger}_pip_flex"]]
        dip = q[..., DOF_INDEX[f"{finger}_dip_flex"]]
        res.append(np.abs(dip - ratio * pip))
    return np.stack(res, axis=-1)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(q: np.ndarray) -> np.ndarray:
    """Map native units into [-1, 1] using the joint limit box.

    Every model in this repo consumes normalised poses, so that a 0.3 m wrist
    translation and a 90 degree PIP flexion contribute comparably to the loss.
    """
    _check_trailing_dim(q)
    span = np.maximum(LIMITS_HI - LIMITS_LO, 1e-8)
    return (2.0 * (q - LIMITS_LO) / span - 1.0).astype(np.float32)


def denormalize(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize`."""
    _check_trailing_dim(x)
    span = LIMITS_HI - LIMITS_LO
    return (LIMITS_LO + 0.5 * (x + 1.0) * span).astype(np.float32)


def _check_trailing_dim(q: np.ndarray) -> None:
    if q.shape[-1] != N_DOF:
        raise ValueError(
            f"expected trailing dimension {N_DOF} (see DOF_NAMES), got shape {q.shape}"
        )


def summary() -> str:
    """Human-readable DOF table, used by scripts to make logs self-documenting."""
    lines = [f"{'idx':>3}  {'name':<24} {'lo':>8}  {'hi':>8}  unit"]
    for i, d in enumerate(DOF_SPECS):
        lines.append(f"{i:>3}  {d.name:<24} {d.lo:>8.2f}  {d.hi:>8.2f}  {d.unit}")
    return "\n".join(lines)

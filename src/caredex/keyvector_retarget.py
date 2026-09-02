"""Task-space retargeting to a robot hand, in the style established by DexPilot.

Why this replaces direct joint-angle transfer
---------------------------------------------
An earlier version of this project measured what happens when anatomical joint
angles are copied straight onto the Shadow Hand: adjacent fingers interpenetrate
in 24-77% of frames. The number is real, but the method is a straw man. Since
DexPilot (Handa, Van Wyk et al.) nobody retargets a human hand to a robot hand
by copying joint angles. The established approach optimises in **task space**:
define five wrist-to-fingertip vectors and ten inter-fingertip vectors, and fit
robot joint angles so its forward kinematics reproduces them. Open
implementations (``dex-retargeting``) carry hand-pose, fingertip, keyvector,
joint-limit, smoothness and collision objectives.

That has two consequences this module exists to handle. A baseline nobody uses
makes any improvement over it worthless, and a "collision-aware correction"
proposed on top of it duplicates an objective those implementations already
have. So the comparison that survives -- do primitive compositions the prior
never saw transfer as well as ones it did? -- has to be run on a pipeline
someone would actually use.

``dex-retargeting`` itself does not install here: it depends on Pinocchio, whose
Windows build fails. The objective is small enough to implement directly, which
also removes a heavy dependency and keeps the optimisation legible.

What this is and is not
-----------------------
This is a re-implementation of the *keyvector objective*, not of DexPilot. It
has no temporal smoothing term, no collision term, and no task-oriented residual
policy. It is here to be a fair baseline for a compositional comparison, and any
text describing it must say so rather than implying a reproduction.

The 15 vectors are scale-normalised before comparison. A human hand and a Shadow
Hand differ in absolute size, so matching raw vector lengths would ask the robot
to reach positions its links cannot span and would confound "the pose does not
transfer" with "the hand is a different size".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from caredex.kinematics import all_chains
from caredex.robot_hand import SHADOW_MAP, ShadowHand

#: Digit order used for both hands. Human tips come from the FK chains, robot
#: tips from the distal bodies below.
DIGITS = ("thumb", "index", "middle", "ring", "pinky")

#: Shadow Hand distal bodies, in the same digit order.
SHADOW_TIP_BODIES = (
    "rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal",
)
SHADOW_PALM_BODY = "rh_palm"


def _align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate ``source`` onto ``target`` by the optimal rigid rotation (Kabsch).

    Without this the objective is not solvable. Key vectors carry direction, and
    the anatomical model's hand-local frame is not the Shadow Hand's palm frame
    -- they are rotated relative to each other by an amount no joint
    configuration can absorb. The first version scored a residual of 20 on a
    normalised objective where a good fit is near zero, and every frame
    penetrated.

    Removing the rotation is not a convenience: what transfers between a human
    hand and a robot hand is the hand's *shape*, not its orientation in the
    robot's base frame, which an arm would set anyway.
    """
    h = source.T @ target
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return source @ (vt.T @ np.diag([1.0, 1.0, d]) @ u.T).T


def _keyvectors(tips: np.ndarray, palm: np.ndarray) -> np.ndarray:
    """15 vectors: 5 palm-to-tip and 10 tip-to-tip, scale-normalised.

    Normalising by the mean vector length is what makes the two hands
    comparable at all. Without it the objective is dominated by the size
    difference between a human hand and a Shadow Hand, and every pose looks
    like a failure to transfer.
    """
    v = [t - palm for t in tips]
    v += [tips[i] - tips[j] for i, j in combinations(range(len(tips)), 2)]
    v = np.stack(v)
    scale = np.linalg.norm(v, axis=-1).mean()
    return v / max(scale, 1e-8)


def human_keyvectors(q_deg: np.ndarray) -> np.ndarray:
    """``(27,)`` anatomical pose -> ``(15, 3)`` normalised key vectors."""
    chains = all_chains(np.asarray(q_deg, dtype=np.float64)[None])
    tips = np.stack([chains[d][0, 3, :] for d in DIGITS])
    return _keyvectors(tips, np.zeros(3))


@dataclass
class RetargetResult:
    qpos: np.ndarray
    residual: float
    iterations: int


class KeyvectorRetargeter:
    """Fit Shadow Hand joint angles to reproduce a human hand's key vectors."""

    def __init__(self, hand: ShadowHand | None = None, joints: tuple[str, ...] | None = None,
                 collision_weight: float = 5e4, contact_tol_mm: float = 0.5):
        self.hand = hand or ShadowHand()
        self._mj = self.hand._mj
        #: Penetration is measured in metres, so a millimetre of overlap
        #: contributes 1e-6 before weighting; the weight puts it on the same
        #: scale as the normalised key-vector error.
        self.collision_weight = collision_weight
        self.contact_tol = contact_tol_mm / 1000.0
        self.joints = joints or tuple(SHADOW_MAP.values())
        self.adr = np.array([self.hand.joint_qpos[j] for j in self.joints])
        self.bounds = [self.hand.joint_range[j] for j in self.joints]

        m = self.hand.model
        self.tip_ids = [self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_BODY, b)
                        for b in SHADOW_TIP_BODIES]
        self.palm_id = self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_BODY, SHADOW_PALM_BODY)
        if min(self.tip_ids) < 0 or self.palm_id < 0:
            raise KeyError("Shadow Hand model lacks the expected distal/palm bodies")

    def robot_keyvectors(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """Key vectors and the total squared penetration at one configuration.

        Both come from the same forward pass. The penetration term has to be
        *inside* the objective rather than fixed afterwards: matching five
        fingertips says nothing about where the intermediate links go, so the
        optimiser drives abduction to its limits to place the tips and pushes
        middle, ring and little fingers 10-13 mm into each other. Every frame
        penetrated, and the post-hoc abduction resolver could not help because
        abduction was already at the bound. This is why the established
        implementations carry a collision objective.
        """
        d, m = self.hand.data, self.hand.model
        d.qpos[:] = 0.0
        d.qpos[self.adr] = theta
        self._mj.mj_forward(m, d)
        tips = np.stack([d.xpos[i].copy() for i in self.tip_ids])
        kv = _keyvectors(tips, d.xpos[self.palm_id].copy())
        pen = 0.0
        for i in range(int(d.ncon)):
            over = -float(d.contact[i].dist) - self.contact_tol
            if over > 0:
                pen += over * over
        return kv, pen

    def fit(self, q_deg: np.ndarray, warm_start: np.ndarray | None = None) -> RetargetResult:
        target = human_keyvectors(q_deg)

        def cost(theta: np.ndarray) -> float:
            v, pen = self.robot_keyvectors(theta)
            return float(((_align(v, target) - target) ** 2).sum()) + self.collision_weight * pen

        x0 = warm_start if warm_start is not None else np.zeros(len(self.joints))
        x0 = np.clip(x0, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        res = minimize(cost, x0, method="L-BFGS-B", bounds=self.bounds,
                       options={"maxiter": 60, "ftol": 1e-6})
        qpos = np.zeros(self.hand.model.nq)
        qpos[self.adr] = res.x
        return RetargetResult(qpos=qpos, residual=float(res.fun), iterations=int(res.nit))

    def fit_trajectory(self, q_deg: np.ndarray) -> tuple[np.ndarray, dict]:
        """Warm-started along the trajectory, which is what makes it tractable.

        Each frame starts from the previous solution. Solving each frame cold
        would be both slower and jumpier -- consecutive frames of 30 fps motion
        are close, so the previous answer is a good guess.
        """
        out = np.zeros((len(q_deg), self.hand.model.nq))
        residuals, warm = [], None
        for i, q in enumerate(q_deg):
            r = self.fit(q, warm_start=warm)
            out[i] = r.qpos
            warm = r.qpos[self.adr]
            residuals.append(r.residual)
        return out, {
            "n_frames": len(q_deg),
            "mean_residual": float(np.mean(residuals)),
            "p95_residual": float(np.percentile(residuals, 95)),
        }

    def penetration(self, qpos: np.ndarray, tol_mm: float = 0.5) -> dict:
        """Self-penetration of the fitted poses, same measure as direct transfer."""
        d, m = self.hand.data, self.hand.model
        tol, pen, depths = tol_mm / 1000.0, 0, []
        for frame in qpos:
            d.qpos[:] = frame
            self._mj.mj_forward(m, d)
            n = int(d.ncon)
            if n == 0:
                continue
            worst = min(float(d.contact[i].dist) for i in range(n))
            if worst < -tol:
                pen += 1
                depths.append(-worst)
        return {
            "penetrating_frames": pen / max(len(qpos), 1),
            "mean_penetration_mm": 1000.0 * float(np.mean(depths)) if depths else 0.0,
        }

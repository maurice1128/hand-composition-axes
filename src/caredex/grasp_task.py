"""A grasp-retention task: can the robot hand actually hold what the prior closes on?

Everything measured before this is geometric -- joint limits, interpenetration,
whether a pose is reachable. None of it says whether a generated motion *does*
anything, and a reviewer is right to ask. This is the smallest task that
produces a success rate rather than a validity rate.

The task, and why it is this one
--------------------------------
An object rests in the palm. The hand executes a generated primitive
composition as position targets. Gravity is on throughout, and afterwards the
hand is shaken. Success is the object still being held.

No arm, no reaching, no planning, no IK, and no reward shaping. Reach-and-grasp
would need an arm the Shadow Hand model does not have and a controller this
project has never claimed. Retention needs none of that and still answers the
question that matters: does closing the hand the way the prior says produce a
grasp, or just a hand-shaped gesture?

What it does not measure
------------------------
Nothing here is dexterous manipulation. The object is placed, not acquired, and
held, not repositioned. A high success rate says the generated motion closes
into something stable; it does not say the prior can perform a task.

Contact is simulated with `mj_step`, not `mj_forward`: forces matter here, which
is the difference between this and every earlier robot measurement in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caredex.robot_hand import SHADOW_MAP, ShadowHand

#: Objects span the grasp types the datasets contain. Sizes are chosen to sit in
#: a Shadow Hand's palm: a 7 cm sphere is about what the hand can close around.
OBJECTS = {
    # Sizes calibrated against the controls below: a full fist must retain the
    # object and an open hand must not. A 3.5 cm sphere was launched by the
    # closing fingers rather than held, in the fist and open alike, which is a
    # scene that measures nothing.
    "box": ("box", (0.025, 0.025, 0.025)),
    "sphere": ("sphere", (0.025,)),
    "cylinder": ("cylinder", (0.02, 0.04)),
}


def _closed_fingertip_centre(mujoco, hand_dir: Path, assets: dict) -> tuple[float, float, float]:
    """Centroid of the fingertips with every flexion actuator at its maximum."""
    m = mujoco.MjModel.from_xml_path(str(hand_dir / "right_hand.xml"))
    d = mujoco.MjData(m)
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if name.endswith(("J0", "J1", "J2", "J3")):
            d.ctrl[i] = m.actuator_ctrlrange[i, 1]
    for _ in range(600):
        mujoco.mj_step(m, d)
    tips = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"]
    closed = np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t)]
                      for t in tips], axis=0)
    palm = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")].copy()

    # Midway between the palm and where the fingertips end up, which is where a
    # held object sits. Placing it *at* the fingertip centroid put it at the top
    # of the closing arc, so the fingers swept past it: every object fell.
    return tuple(((palm + closed) / 2).tolist())


#: Difficulty levels, calibrated so the controls still separate at each one.
#: A single easy setting is not enough: at 0.15 kg and 1.5 g the prior's
#: compositions succeeded 100% of the time, which is a ceiling and cannot
#: discriminate between arms however large the real difference is.
#: Difficulty is mass and shake only. Scaling the object up instead made even a
#: full fist fail -- a larger box does not fit the hand, so the control stopped
#: separating and the setting measured nothing. Size stays fixed.
DIFFICULTY = {
    "easy": {"mass": 0.15, "shake_g": 1.5, "size_scale": 1.0},
    "medium": {"mass": 0.25, "shake_g": 3.0, "size_scale": 1.0},
    "hard": {"mass": 0.35, "shake_g": 5.0, "size_scale": 1.0},
}


def scene_xml(hand_xml: str, shape: str, size: tuple[float, ...],
              mass: float = 0.15,
              centre: tuple[float, float, float] = (0.33, 0.0, 0.02)) -> str:
    """Hand plus one free object resting against the palm.

    The object starts in light contact rather than dropped from above: this
    measures whether a closing motion *retains* an object, and a drop would
    confound that with whether the hand happened to be open at the right
    moment.
    """
    dims = " ".join(f"{s}" for s in size)
    return f"""
<mujoco model="grasp_task">
  <include file="{hand_xml}"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 -0.5"
          rgba="0.3 0.3 0.35 1"/>
    <body name="obj" pos="{centre[0]:.4f} {centre[1]:.4f} {centre[2]:.4f}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" type="{shape}" size="{dims}" mass="{mass}"
            rgba="0.8 0.3 0.2 1" friction="1.0 0.02 0.001"
            solref="0.004 1" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class GraspResult:
    held: bool
    #: How far the object fell from where it started, in metres.
    drop_m: float
    #: Contacts between hand and object at the end of the hold.
    contacts: int
    #: Fraction of the commanded trajectory that was inside actuator range.
    in_range: float


class GraspTask:
    """Execute a 27-DOF trajectory on the Shadow Hand and see if a grasp holds."""

    def __init__(self, shape: str = "box", difficulty: str = "easy",
                 hand_dir: str | Path | None = None):
        import mujoco

        self._mj = mujoco
        hand_dir = Path(hand_dir or r"D:\datasets\mujoco_menagerie\shadow_hand")
        self.difficulty = DIFFICULTY[difficulty]
        geom_type, size = OBJECTS[shape]
        size = tuple(v * self.difficulty["size_scale"] for v in size)
        assets = _asset_dict(hand_dir)
        # Where the object goes is measured, not reasoned about: close the hand
        # fully and take the centroid of the fingertips. Deriving it from the
        # joint axes would mean assuming a convention, which is how the earlier
        # retargeting bugs happened.
        centre = _closed_fingertip_centre(mujoco, hand_dir, assets)
        xml = scene_xml("right_hand.xml", geom_type, size,
                        mass=self.difficulty["mass"], centre=centre)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)

        self.obj_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")

        # Actuators are named rh_A_<joint>, but the Shadow Hand couples each
        # finger's two distal joints onto one actuator, rh_A_<XX>J0. That is the
        # same 2/3 DIP/PIP coupling the anatomical model enforces, so PIP
        # flexion drives J0 and DIP is dropped rather than mapped to nothing.
        self.act_of_dof: dict[str, int] = {}
        for dof, joint in SHADOW_MAP.items():
            if dof.endswith("dip_flex"):
                continue
            name = joint.replace("rh_", "rh_A_")
            if dof.endswith("pip_flex"):
                name = name[:-1] + "0"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.act_of_dof[dof] = aid

    def run(self, q_deg: np.ndarray, hold_s: float = 0.5,
            shake_g: float | None = None, hold_tol_m: float = 0.10) -> GraspResult:
        """Drive the trajectory, hold, shake, and report whether the object stayed.

        The shake is a lateral acceleration applied through gravity rather than
        a wrist motion, because the Shadow Hand model here has a fixed base. It
        separates a grasp that holds from one that merely rests.
        """
        shake_g = self.difficulty["shake_g"] if shake_g is None else shake_g
        mj, m, d = self._mj, self.model, self.data
        mj.mj_resetData(m, d)
        mj.mj_forward(m, d)
        start = d.xpos[self.obj_body].copy()

        lo = m.actuator_ctrlrange[:, 0]
        hi = m.actuator_ctrlrange[:, 1]
        in_range = []

        steps_per_frame = max(1, int(round((1 / 30) / m.opt.timestep)))
        for frame in np.radians(np.asarray(q_deg, dtype=np.float64)):
            from caredex.hand_model import DOF_INDEX

            for dof, aid in self.act_of_dof.items():
                v = frame[DOF_INDEX[dof]]
                in_range.append(lo[aid] <= v <= hi[aid])
                d.ctrl[aid] = np.clip(v, lo[aid], hi[aid])
            for _ in range(steps_per_frame):
                mj.mj_step(m, d)

        for _ in range(int(hold_s / m.opt.timestep)):
            mj.mj_step(m, d)

        # Shake: swing gravity sideways and back a few times.
        g0 = m.opt.gravity.copy()
        for k in range(4):
            m.opt.gravity[:] = [shake_g * 9.81 * (1 if k % 2 == 0 else -1), 0, g0[2]]
            for _ in range(int(0.1 / m.opt.timestep)):
                mj.mj_step(m, d)
        m.opt.gravity[:] = g0
        for _ in range(int(0.2 / m.opt.timestep)):
            mj.mj_step(m, d)

        drop = float(np.linalg.norm(d.xpos[self.obj_body] - start))
        contacts = sum(
            1 for i in range(int(d.ncon))
            if self.obj_geom in (d.contact[i].geom1, d.contact[i].geom2)
        )
        return GraspResult(
            # Both conditions matter. Displacement alone would pass an object
            # that slid out of the hand but happened to stop nearby; contact
            # alone would pass one merely brushing a fingertip on the way down.
            held=bool(drop < hold_tol_m and contacts > 0),
            drop_m=drop,
            contacts=contacts,
            in_range=float(np.mean(in_range)) if in_range else 0.0,
        )


def _asset_dict(hand_dir: Path) -> dict[str, bytes]:
    """MuJoCo needs the hand XML and its meshes when the scene is a string."""
    assets: dict[str, bytes] = {}
    for p in list(hand_dir.glob("*.xml")) + list((hand_dir / "assets").rglob("*")):
        if p.is_file():
            key = p.name if p.parent == hand_dir else f"assets/{p.name}"
            assets[key] = p.read_bytes()
    return assets

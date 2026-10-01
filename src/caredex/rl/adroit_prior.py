"""Hand-prior action interfaces for the Adroit tasks (Gymnasium-Robotics).

Adroit's hand is a Shadow Hand under the old joint numbering, one index below
MuJoCo Menagerie's (``rh_FFJ4`` there is ``FFJ3`` here), with the same axes and
near-identical ranges, except the thumb's distal joint ``THJ0``, whose range is
[-90, 0] degrees: flexion is negative there and positive in our convention.

The wrapper sits on the raw vector env, *below* ``VecNormalize``, because it
reads the hand's joint angles from the unnormalised observation to anchor the
decoder at each reset. The policy's action is

    [ raw part | interface part | residual part ]

* raw: every actuator the prior does not cover (arm, the two wrist joints, the
  little finger's metacarpal ``LFJ4``), native [-1, 1] position targets;
* interface: what the decoder consumes, scaled from [-1, 1] to its bounds.
  ``keyframe`` / ``linear``: a 21-joint target pose latched every ``macro_every``
  prior frames and reached through the learned in-betweener / a straight line.
  ``continuous``: the per-frame latent. ``brick`` / ``brickkf``: K logits whose
  arg-max picks one of K fixed keyframes (a data-derived library), reached by a
  straight line / the learned in-betweener: the discrete smart-primitive layer;
* residual: added to the 21 covered actuators' native targets, at most
  ``residual`` of each half-range, the same fine-correction channel the staged
  task gave every interface.

Adroit runs at 100 Hz and the priors were trained on 30 fps motion, so the
decoder advances one frame every ``prior_every`` = 3 environment steps and its
output is held in between.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

from caredex.hand_model import DOF_NAMES, N_ARTICULATED, denormalize, normalize
from caredex.rl.stepper import load_stepper
from caredex.rl.vec_env import _BatchedDecoder

#: our DOF -> (Adroit joint, sign). Sign -1: flexion is negative on Adroit.
ADROIT_MAP = {
    **{f"{ours}_{j}": (f"{adr}J{k}", 1.0)
       for ours, adr in (("index", "FF"), ("middle", "MF"), ("ring", "RF"), ("pinky", "LF"))
       for j, k in (("mcp_abd", 3), ("mcp_flex", 2), ("pip_flex", 1), ("dip_flex", 0))},
    "thumb_cmc_abd": ("THJ4", 1.0), "thumb_cmc_flex": ("THJ3", 1.0), "thumb_mcp_abd": ("THJ2", 1.0),
    "thumb_mcp_flex": ("THJ1", 1.0), "thumb_ip_flex": ("THJ0", -1.0),
}
#: where the 24 hand joint angles (WRJ1, WRJ0, FFJ3..THJ0, qpos order) sit in each task's observation
HAND_OBS_START = {"door": 3, "hammer": 2, "relocate": 6, "pen": 0}


class AdroitPriorVecWrapper(VecEnvWrapper):
    def __init__(self, venv, task: str, interface: str, run_dir, model, macro_every: int = 16,
                 prior_every: int = 3, residual: float = 0.3, bricks=None):
        """``model`` is one environment's ``MjModel`` (names and ranges only)."""
        import mujoco
        if interface not in ("keyframe", "linear", "continuous", "brick", "brickkf"):
            raise ValueError(interface)
        self.task, self.interface = task, interface
        self.prior_every, self.residual = int(prior_every), float(residual)
        B = venv.num_envs

        names = list(DOF_NAMES)
        nu = model.nu
        act_joint = [model.joint(model.actuator_trnid[i, 0]).name for i in range(nu)]
        hand_qpos_names = [model.joint(j).name for j in range(model.njnt)]
        first_hand = hand_qpos_names.index("WRJ1")
        self.cov_dof, self.cov_act, self.cov_obs, self.sign = [], [], [], []
        for dof, (joint, sign) in ADROIT_MAP.items():
            self.cov_dof.append(names.index(dof))
            self.cov_act.append(act_joint.index(joint))
            self.cov_obs.append(HAND_OBS_START[task] + hand_qpos_names.index(joint) - first_hand)
            self.sign.append(sign)
        self.cov_dof, self.cov_act = np.asarray(self.cov_dof), np.asarray(self.cov_act)
        self.cov_obs, self.sign = np.asarray(self.cov_obs), np.asarray(self.sign)
        assert len(self.cov_dof) == N_ARTICULATED and sorted(self.cov_dof) == list(range(N_ARTICULATED))
        self.raw_act = np.asarray([i for i in range(nu) if i not in set(self.cov_act.tolist())])
        lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        self.mean, self.half = ((hi + lo) / 2)[self.cov_act], ((hi - lo) / 2)[self.cov_act]
        self.nu = nu

        skw = {"horizon": max(1, int(macro_every))} if interface != "continuous" else {}
        if interface in ("brick", "brickkf"):
            skw["bricks"] = str(bricks)
        stepper = load_stepper(run_dir, interface, device="cpu", **skw)
        self.dec = _BatchedDecoder(stepper, B, rewindow=getattr(stepper, "horizon", 32))
        self.scale = stepper.spec.high.astype(np.float32)
        self.n_dec = int(stepper.spec.action_dim)
        d = len(self.raw_act) + self.n_dec + N_ARTICULATED
        super().__init__(venv, action_space=spaces.Box(-1.0, 1.0, (d,), np.float32))
        self.k = 0
        self.pose = np.zeros((B, len(names)), dtype=np.float32)

    # -- helpers ----------------------------------------------------------------

    def _anchor_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Hand joint angles in the raw observation -> normalised 27-channel pose (wrist channels at rest)."""
        q = np.zeros((obs.shape[0], len(DOF_NAMES)), dtype=np.float32)
        q[:, self.cov_dof] = np.degrees(obs[:, self.cov_obs]) * self.sign
        return normalize(q)

    def _native(self, pose_norm: np.ndarray) -> np.ndarray:
        """Normalised pose -> native [-1, 1] targets of the covered actuators."""
        rad = np.radians(denormalize(pose_norm)[:, self.cov_dof]) * self.sign
        return np.clip((rad - self.mean) / self.half, -1.0, 1.0)

    # -- VecEnv -----------------------------------------------------------------

    def reset(self):
        obs = self.venv.reset()
        self.pose = self._anchor_from_obs(obs)
        self.dec.reset(np.arange(self.num_envs), self.pose)
        self.k = 0
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        a = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        n_raw = len(self.raw_act)
        raw, dec_a, res = a[:, :n_raw], a[:, n_raw:n_raw + self.n_dec], a[:, n_raw + self.n_dec:]
        if self.k % self.prior_every == 0:
            self.pose = self.dec.step(dec_a * self.scale)
        self.k += 1
        native = np.zeros((self.num_envs, self.nu), dtype=np.float32)
        native[:, self.raw_act] = raw
        native[:, self.cov_act] = np.clip(self._native(self.pose) + self.residual * res, -1.0, 1.0)
        self.venv.step_async(native)

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        if done.any():
            idx = np.flatnonzero(done)
            anchor = self._anchor_from_obs(obs[idx])   # SB3 vec envs return the post-reset observation
            self.pose[idx] = anchor
            self.dec.reset(idx, anchor)
        return obs, rew, done, infos

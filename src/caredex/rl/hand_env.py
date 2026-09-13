"""A Shadow Hand retention task with a prior as the action space.

The object starts resting against the palm, exactly as in ``grasp_task``. The
difference is who drives the hand. There, a sampled window is replayed open
loop and the shake comes afterwards. Here a policy emits one latent per frame,
the frozen decoder turns it into a pose, and the shake arrives *during* the
episode at random times and strengths, so the policy has to react to keep the
object. Success is the object still held at the last frame.

This is the smallest task on which the two arms can differ through the
channel the paper cares about: the policy has to find, in the prior's latent
space, a motion that closes on the object and then holds against
perturbation. Nothing about the task favours a discrete or a continuous
latent; the reward, the scene, the object and the shake schedule are shared.

Observation (all float32):
    hand qpos (n_q), hand qvel (n_v), object position relative to palm (3),
    object linear velocity (3), object quaternion (4), last commanded pose in
    normalised units (27), current perturbation acceleration in g (3),
    time fraction (1).

Reward: +1 per frame the object is held (within ``hold_tol_m`` of its start
and in contact with the hand), 0 otherwise. The episode terminates early when
the object is lost. No shaping, so both arms optimise the same thing the
paper's grasp task measured.

Jitter is recorded per episode as LAMP's second-order finite difference on
the commanded articulated joint targets in degrees,
    J = 1/(T-2) * sum_t || a_{t+2} - 2 a_{t+1} + a_t ||^2,
so a "discrete banks jitter" claim can be checked on the same statistic the
objection was raised with.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from caredex.grasp_task import GraspTask
from caredex.hand_model import DOF_INDEX, N_DOF, N_ARTICULATED, denormalize, normalize
from caredex.rl.stepper import load_stepper

FPS = 30


class ShadowRetainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, arm: str, run_dir: str | Path,
                 shape: str = "box", difficulty: str = "medium",
                 episode_frames: int = 90, warmup_frames: int = 30,
                 perturb: bool = True, hold_tol_m: float = 0.10,
                 hand_dir: str | Path | None = None,
                 stepper_kw: dict | None = None, seed: int | None = None):
        super().__init__()
        self.arm = arm
        self.stepper = load_stepper(run_dir, arm, device="cpu", **(stepper_kw or {}))
        spec = self.stepper.spec
        self.action_space = spaces.Box(spec.low, spec.high, dtype=np.float32)

        self.task = GraspTask(shape, difficulty, hand_dir)
        self.mj, self.m, self.d = self.task._mj, self.task.model, self.task.data
        self.act_of_dof = self.task.act_of_dof
        self.obj_body, self.obj_geom = self.task.obj_body, self.task.obj_geom
        self.palm_body = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, "rh_palm")
        self.obj_qadr = self.m.jnt_qposadr[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, "obj_free")]
        self.obj_vadr = self.m.jnt_dofadr[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, "obj_free")]
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        self.steps_per_frame = max(1, int(round((1 / FPS) / self.m.opt.timestep)))
        self.g0 = self.m.opt.gravity.copy()

        self.episode_frames = int(episode_frames)
        self.warmup_frames = int(warmup_frames)
        self.perturb = bool(perturb)
        self.hold_tol_m = float(hold_tol_m)
        self.shake_g = self.task.difficulty["shake_g"]

        n_q, n_v = self.m.nq - 7, self.m.nv - 6  # hand only
        self._n_q, self._n_v = n_q, n_v
        obs_dim = n_q + n_v + 3 + 3 + 4 + N_DOF + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._last_pose = np.zeros(N_DOF, dtype=np.float32)
        self._cmd_hist: list[np.ndarray] = []
        self._accel_g = np.zeros(3)
        self._pulse_left = 0
        self._start = np.zeros(3)

    # -- helpers --------------------------------------------------------------

    def _hand_qpos(self) -> np.ndarray:
        q = np.delete(self.d.qpos, np.s_[self.obj_qadr:self.obj_qadr + 7])
        return q

    def _hand_qvel(self) -> np.ndarray:
        return np.delete(self.d.qvel, np.s_[self.obj_vadr:self.obj_vadr + 6])

    def _obs(self) -> np.ndarray:
        d = self.d
        rel = d.xpos[self.obj_body] - d.xpos[self.palm_body]
        vel = d.qvel[self.obj_vadr:self.obj_vadr + 3]
        quat = d.qpos[self.obj_qadr + 3:self.obj_qadr + 7]
        return np.concatenate([
            self._hand_qpos(), self._hand_qvel(), rel, vel, quat,
            self._last_pose, self._accel_g, [self._t / self.episode_frames],
        ]).astype(np.float32)

    def _held(self) -> tuple[bool, float, int]:
        drop = float(np.linalg.norm(self.d.xpos[self.obj_body] - self._start))
        contacts = sum(1 for i in range(int(self.d.ncon))
                       if self.obj_geom in (self.d.contact[i].geom1, self.d.contact[i].geom2))
        return (drop < self.hold_tol_m and contacts > 0), drop, contacts

    def _schedule_perturbation(self) -> None:
        """Random lateral pulses after warm-up; strength up to the difficulty's shake."""
        if not self.perturb or self._t < self.warmup_frames:
            self._accel_g[:] = 0.0
            return
        if self._pulse_left > 0:
            self._pulse_left -= 1
            if self._pulse_left == 0:
                self._accel_g[:] = 0.0
            return
        # Roughly one pulse every 15 frames on average, lasting 3 frames (0.1 s),
        # which is the pulse the offline grasp task applies four times in a row.
        if self._rng.random() < 1 / 15:
            ang = self._rng.uniform(0, 2 * np.pi)
            mag = self._rng.uniform(0.5, 1.0) * self.shake_g
            self._accel_g[:] = [mag * np.cos(ang), mag * np.sin(ang), 0.0]
            self._pulse_left = 3

    # -- gym API ----------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        d.xfrc_applied[:] = 0.0
        mj.mj_forward(m, d)
        self._start = d.xpos[self.obj_body].copy()
        self._t = 0
        self._accel_g[:] = 0.0
        self._pulse_left = 0
        self._cmd_hist = []
        # Zero degrees on every joint is the open hand the offline task used as
        # its negative control; in normalised units that is *not* zero.
        self._last_pose = normalize(np.zeros(N_DOF, dtype=np.float32))
        self.stepper.reset(self._last_pose)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        mj, m, d = self.mj, self.m, self.d
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)
        pose = self.stepper.step(action)
        self._last_pose = pose.astype(np.float32)
        q_deg = denormalize(pose)
        q_rad = np.radians(q_deg.astype(np.float64))
        self._cmd_hist.append(q_deg[:N_ARTICULATED].copy())

        for dof, aid in self.act_of_dof.items():
            d.ctrl[aid] = np.clip(q_rad[DOF_INDEX[dof]], self.lo[aid], self.hi[aid])

        self._schedule_perturbation()
        # Pulse as an external force on the object (mass * a), the same thing
        # the batched env does so the two agree frame for frame.
        d.xfrc_applied[self.obj_body, :3] = self._accel_g * 9.81 * self.m.body_mass[self.obj_body]
        for _ in range(self.steps_per_frame):
            mj.mj_step(m, d)

        self._t += 1
        held, drop, contacts = self._held()
        lost = drop >= self.hold_tol_m
        reward = 1.0 if held else 0.0
        terminated = bool(lost)
        truncated = bool(self._t >= self.episode_frames and not terminated)
        info = {"held": held, "drop_m": drop, "contacts": contacts}
        if terminated or truncated:
            info["success"] = bool(held and not lost)
            info["jitter"] = self.jitter()
            info["frames"] = self._t
        return self._obs(), reward, terminated, truncated, info

    def jitter(self) -> float:
        a = np.asarray(self._cmd_hist)
        if len(a) < 3:
            return 0.0
        dd = a[2:] - 2 * a[1:-1] + a[:-2]
        return float(np.mean(np.sum(dd ** 2, axis=-1)))

    def close(self) -> None:
        pass


def make_env(arm: str, run_dir: str | Path, seed: int = 0, **kw):
    """Factory for vectorised environments; each worker seeds its own rng."""
    def _init():
        env = ShadowRetainEnv(arm, run_dir, seed=seed, **kw)
        env.reset(seed=seed)
        return env
    return _init

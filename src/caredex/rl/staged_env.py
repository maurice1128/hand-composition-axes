"""A staged retention task: sequences of skills, with held-out sequences.

The retention task asks for one posture. This one asks for a *sequence*,
which is where a bank of primitives is supposed to earn its keep, and it is
built so the paper's paired design carries over: a *naive* policy trains on
a set of short sequences, an *informed* policy trains on those plus the
held-out longer ones, and both are scored on the held-out sequences. The
compositional penalty is the informed-minus-naive gap, per arm, paired by
seed, exactly as Section III-C scores it for reconstruction.

Skills (each ``stage_frames`` long, after the usual 30-frame warm-up):

* ``Hx``  hold the object while the palm tilts along +-x (thumb side vs
          little-finger side), ramped over the first ``ramp_frames`` of the
          stage and then held.
* ``Hy``  the same along +-y (fingertips vs wrist).
* ``R``   release: the tilt stops and the object must leave the hand.
          Always the last skill of a sequence.

Reward per frame: +1 when the current skill's condition holds (object held
for H skills, object not touching the hand for R). Losing the object during
an H skill ends the episode. Success is every H skill satisfied at its last
frame and, if present, R satisfied at its last frame.

The observation is the retention observation plus a one-hot of the current
skill and the fraction of the stage elapsed, so the policy knows what is
being asked; it never sees the sequence ahead.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from caredex.hand_model import N_ARTICULATED, N_DOF
from caredex.rl.vec_env import RetainVecEnv

SKILLS = ("Hx", "Hy", "R")
SKILL_ID = {s: i for i, s in enumerate(SKILLS)}


def parse_sequences(spec: str) -> list[tuple[str, ...]]:
    """``"Hx-R,Hy-R,Hx-Hy"`` -> ``[("Hx","R"), ("Hy","R"), ("Hx","Hy")]``."""
    out = []
    for s in spec.split(","):
        seq = tuple(t.strip() for t in s.split("-") if t.strip())
        for t in seq:
            if t not in SKILL_ID:
                raise ValueError(f"unknown skill {t!r} in {spec!r}")
        if "R" in seq and seq.index("R") != len(seq) - 1:
            raise ValueError("R must be the last skill")
        out.append(seq)
    return out


class StagedRetainVecEnv(RetainVecEnv):
    def __init__(self, *args, sequences: str = "Hx-R,Hy-R,Hx-Hy,Hy-Hx",
                 stage_frames: int = 40, max_stages: int = 3, **kw):
        kw.setdefault("perturb_mode", "tilt")
        kw.setdefault("ramp_frames", 25)
        super().__init__(*args, **kw)
        self.sequences = parse_sequences(sequences)
        self.stage_frames = int(stage_frames)
        self.max_stages = int(max_stages)
        self.episode_frames = self.warmup_frames + self.max_stages * self.stage_frames
        n = self.num_envs
        self.seq_idx = np.zeros(n, dtype=np.int64)
        self.seq_len = np.zeros(n, dtype=np.int64)
        self.skill = np.full((n, self.max_stages), -1, dtype=np.int64)
        self.stage_ok = np.zeros((n, self.max_stages), dtype=bool)
        self.sign = np.ones(n)
        base = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (base + len(SKILLS) + 1,), dtype=np.float32)

    # -- stage bookkeeping ------------------------------------------------------

    def _stage(self) -> tuple[np.ndarray, np.ndarray]:
        """Current stage index (-1 during warm-up, >= seq_len when done) and progress in [0,1]."""
        rel = self.t - self.warmup_frames
        idx = np.where(rel < 0, -1, rel // self.stage_frames)
        prog = np.clip((rel % self.stage_frames) / self.stage_frames, 0, 1)
        return idx, prog

    def _current_skill(self) -> np.ndarray:
        idx, _ = self._stage()
        valid = (idx >= 0) & (idx < self.seq_len)
        sk = np.full(self.num_envs, -1, dtype=np.int64)
        sk[valid] = self.skill[np.arange(self.num_envs)[valid], idx[valid]]
        return sk

    def _reset_env(self, idx: np.ndarray) -> None:
        super()._reset_env(idx)
        k = self.rng.integers(0, len(self.sequences), size=len(idx))
        self.seq_idx[idx] = k
        for i, kk in zip(idx, k):
            seq = self.sequences[kk]
            self.seq_len[i] = len(seq)
            self.skill[i] = -1
            self.skill[i, :len(seq)] = [SKILL_ID[s] for s in seq]
        self.stage_ok[idx] = False
        self.sign[idx] = self.rng.choice([-1.0, 1.0], size=len(idx))

    def set_sequences(self, spec: str) -> None:
        """Switch the sequence pool (used for held-out evaluation)."""
        self.sequences = parse_sequences(spec)

    # -- perturbation: per-skill tilt --------------------------------------------

    def _schedule(self) -> None:
        idx, prog = self._stage()
        sk = self._current_skill()
        ramp = np.clip((self.t - self.warmup_frames - idx * self.stage_frames) / max(self.ramp_frames, 1), 0, 1)
        self.accel[:] = 0.0
        hx, hy = sk == SKILL_ID["Hx"], sk == SKILL_ID["Hy"]
        self.accel[hx, 0] = self.sign[hx] * ramp[hx] * self.shake_g
        self.accel[hy, 1] = self.sign[hy] * ramp[hy] * self.shake_g
        # Release keeps the previous hold's tilt at full strength: the palm
        # faces up, so with the tilt off an opened hand still cradles the
        # object and "leaving the hand" is impossible. Under the tilt, opening
        # lets it slide away, which is what release means here.
        r = sk == SKILL_ID["R"]
        if r.any():
            prev = np.full(self.num_envs, -1, dtype=np.int64)
            has_prev = r & (idx >= 1)
            prev[has_prev] = self.skill[np.arange(self.num_envs)[has_prev], idx[has_prev] - 1]
            ax = np.where(prev == SKILL_ID["Hy"], 1, 0)
            self.accel[r, ax[r]] = self.sign[r] * self.shake_g

    # -- observation, reward, termination ------------------------------------------

    def _obs(self) -> np.ndarray:
        base = super()._obs()
        sk = self._current_skill()
        onehot = np.zeros((self.num_envs, len(SKILLS)), dtype=np.float32)
        valid = sk >= 0
        onehot[np.flatnonzero(valid), sk[valid]] = 1.0
        _, prog = self._stage()
        return np.concatenate([base, onehot, prog[:, None].astype(np.float32)], axis=1)

    def step_wait(self):
        pose = self.dec.step(self._actions)
        if self._residual is not None:
            pose = pose.copy()
            pose[:, :N_ARTICULATED] = np.clip(pose[:, :N_ARTICULATED] + self._residual, -1.0, 1.0)
            import torch
            self.dec.last = torch.as_tensor(pose, dtype=torch.float32)
        self.last_pose = pose.astype(np.float32)
        from caredex.hand_model import denormalize
        q_deg = denormalize(pose)
        q_rad = np.radians(q_deg.astype(np.float64))
        cmd = q_deg[:, :N_ARTICULATED]
        have2 = self.ep_frames >= 2
        dd = cmd - 2 * self.cmd_prev[:, 1] + self.cmd_prev[:, 0]
        self.jit_sum[have2] += np.sum(dd[have2] ** 2, axis=-1)
        self.cmd_prev[:, 0] = self.cmd_prev[:, 1]
        self.cmd_prev[:, 1] = cmd

        self._schedule()
        ctrl = np.zeros((self.num_envs, self.nu))
        for di, aid in self.dof_aid:
            ctrl[:, aid] = np.clip(q_rad[:, di], self.lo[aid], self.hi[aid])
        xfrc = np.zeros((self.num_envs, self.m.nbody * 6))
        xfrc[:, self.obj_body * 6:self.obj_body * 6 + 3] = self.accel * 9.81 * self.obj_mass
        control = np.concatenate([ctrl, xfrc], axis=-1)[:, None, :]
        state, sensordata = self.roll.rollout(self.m, self.pool, self.states, control,
                                              control_spec=self.spec_ctrl, nstep=self.steps_per_frame)
        self.states, self.sensors = state[:, -1], sensordata[:, -1]

        self.t += 1
        self.ep_frames += 1
        held, drop, touch = self._held()
        lost = drop >= self.hold_tol_m
        # Released = the object has moved 5 cm from where it rested. A touch
        # test would fail once it lands on the floor half a metre below.
        released = drop > 0.05
        idx, prog = self._stage()
        sk = self._current_skill()
        in_h = (sk == SKILL_ID["Hx"]) | (sk == SKILL_ID["Hy"])
        in_r = sk == SKILL_ID["R"]
        rewards = np.where(in_h, held, np.where(in_r, released, held)).astype(np.float32)

        # Record each stage's outcome at its last frame.
        # t just advanced; a stage boundary means stage idx-1 has just ended,
        # including the last stage of the sequence (idx == seq_len).
        boundary = (((self.t - self.warmup_frames) % self.stage_frames) == 0) & (self.t > self.warmup_frames)
        last_frame = boundary & (idx >= 1) & (idx <= self.seq_len)
        ended = np.flatnonzero(last_frame)
        for i in ended:
            j = idx[i] - 1
            if 0 <= j < self.seq_len[i]:
                s = self.skill[i, j]
                self.stage_ok[i, j] = bool(released[i]) if s == SKILL_ID["R"] else bool(held[i])

        term = lost & in_h                      # losing during a hold ends it
        done_seq = idx >= self.seq_len          # all stages elapsed
        trunc = done_seq & ~term
        dones = term | trunc
        infos = [{"held": bool(held[i]), "drop_m": float(drop[i]), "touch": float(touch[i]),
                  "skill": int(sk[i])} for i in range(self.num_envs)]
        obs = self._obs()
        for i in np.flatnonzero(dones):
            nfr = int(self.ep_frames[i])
            L = int(self.seq_len[i])
            ok = bool((not term[i]) and self.stage_ok[i, :L].all())
            infos[i]["success"] = ok
            infos[i]["stages_ok"] = int(self.stage_ok[i, :L].sum())
            infos[i]["sequence"] = "-".join(SKILLS[s] for s in self.skill[i, :L])
            infos[i]["frames"] = nfr
            infos[i]["jitter"] = float(self.jit_sum[i] / max(nfr - 2, 1))
            infos[i]["TimeLimit.truncated"] = bool(trunc[i])
            infos[i]["terminal_observation"] = obs[i].copy()
        if dones.any():
            ii = np.flatnonzero(dones)
            self._reset_env(ii)
            obs[ii] = self._obs()[ii]
        return obs, rewards, dones, infos

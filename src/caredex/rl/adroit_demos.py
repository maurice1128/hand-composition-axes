"""Human demonstrations for line C: D4RL ``relocate/human-v2`` (Minari), 25 episodes.

``grasp_segments`` cuts each demo from its first step to a few steps after the ball is
first lifted (z > ``lift_z``) with the palm near it -- the grasp skill's demonstration.
Ball height is not in the observation, so it is recovered from forward kinematics: the
hand qpos (obs[:30]) gives the palm site, and obs[30:33] is palm - ball.

Observations are returned in the staged task's layout: the native 39-D vector plus the
grasp one-hot (1, 0, 0) and progress 0.
"""

from __future__ import annotations

import os

import numpy as np

DATASET = "D4RL/relocate/human-v2"


def _load(root: str):
    os.environ.setdefault("MINARI_DATASETS_PATH", root)
    import minari
    return minari.load_dataset(DATASET)


def grasp_segments(root: str = r"D:\datasets\minari", lift_z: float = 0.10, near: float = 0.08, tail: int = 10):
    import gymnasium as gym
    import gymnasium_robotics
    import mujoco
    gym.register_envs(gymnasium_robotics)
    ds = _load(root)
    env = gym.make("AdroitHandRelocate-v1"); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    obs_out, act_out, info = [], [], []
    for e in ds.iterate_episodes():
        O = np.asarray(e.observations)[:-1]; A = np.asarray(e.actions)
        z = np.empty(len(O)); dist = np.linalg.norm(O[:, 30:33], axis=1)
        for t in range(len(O)):
            d.qpos[:30] = O[t, :30]; mujoco.mj_kinematics(m, d)
            z[t] = (d.site_xpos[u.S_grasp_site_id] - O[t, 30:33])[2]
        hit = np.flatnonzero((z > lift_z) & (dist < near))
        if len(hit) == 0:
            info.append((int(e.id), None)); continue
        end = min(len(O), int(hit[0]) + tail)
        extra = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (end, 1))
        obs_out.append(np.concatenate([O[:end], extra], 1)); act_out.append(np.clip(A[:end], -1, 1))
        info.append((int(e.id), int(hit[0])))
    env.close()
    return np.concatenate(obs_out).astype(np.float32), np.concatenate(act_out).astype(np.float32), info


def full_segments(root: str = r"D:\datasets\minari", lift_z: float = 0.10, near: float = 0.08):
    """Whole demonstrations labelled for ONE flat policy on the staged task: the grasp one-hot
    until the first lift (same criterion as ``grasp_segments``), the carry one-hot and progress
    1/3 after it, to the demo's end. The demos contain no place-and-release, so a flat policy
    cloned from them has demonstrations for the first two stages only."""
    import gymnasium as gym
    import gymnasium_robotics
    import mujoco
    gym.register_envs(gymnasium_robotics)
    ds = _load(root)
    env = gym.make("AdroitHandRelocate-v1"); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    obs_out, act_out, info = [], [], []
    for e in ds.iterate_episodes():
        O = np.asarray(e.observations)[:-1]; A = np.asarray(e.actions)
        z = np.empty(len(O)); dist = np.linalg.norm(O[:, 30:33], axis=1)
        for t in range(len(O)):
            d.qpos[:30] = O[t, :30]; mujoco.mj_kinematics(m, d)
            z[t] = (d.site_xpos[u.S_grasp_site_id] - O[t, 30:33])[2]
        hit = np.flatnonzero((z > lift_z) & (dist < near))
        if len(hit) == 0:
            info.append((int(e.id), None)); continue
        h = int(hit[0])
        extra = np.zeros((len(O), 4)); extra[:h + 1, 0] = 1.0
        extra[h + 1:, 1] = 1.0; extra[h + 1:, 3] = 1.0 / 3.0
        obs_out.append(np.concatenate([O, extra], 1)); act_out.append(np.clip(A, -1, 1))
        info.append((int(e.id), h))
    env.close()
    return np.concatenate(obs_out).astype(np.float32), np.concatenate(act_out).astype(np.float32), info


def carry_segments(root: str = r"D:\datasets\minari", **kw):
    """The carry part of each demonstration: from the step after the first lift to the demo's
    end, labelled with the carry one-hot and progress 1/3 (the rows of ``full_segments`` after
    the grasp). The human demonstrators hold the ball at the air target until the episode ends,
    so these segments end in the stable held state the strict carry criterion asks for."""
    O, A, info = full_segments(root, **kw)
    keep = O[:, 40] > 0.5
    return O[keep], A[keep], info
